from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ACTION_LABELS = [
    "target_x_b",
    "target_y_b",
    "target_z_b",
    "rot6d_0",
    "rot6d_1",
    "rot6d_2",
    "rot6d_3",
    "rot6d_4",
    "rot6d_5",
    "gripper_width",
]

REQUIRED_ALIGNED_FILES = ("timestamps.npy", "rgb.npy", "action.npy")


@dataclass(frozen=True)
class ExportStats:
    saved_frames: int
    saved_thumbs: int
    skipped_frames: int
    skipped_thumbs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a static HTML viewer for LabPick CAFE trajectory records.")
    parser.add_argument(
        "--record_dir",
        type=Path,
        default=Path("/tmp/lab_pick_cafe_records"),
        help="Directory containing record_XXXXXX CAFE records.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for index.html, frames/, and thumbs/. Defaults to <record_dir>/vis_traj_html.",
    )
    parser.add_argument("--jpeg_quality", type=int, default=85, help="JPEG quality for exported frames and thumbnails.")
    parser.add_argument("--thumb_width", type=int, default=160, help="Thumbnail width in pixels.")
    parser.add_argument("--frame_width", type=int, default=640, help="Main exported frame width in pixels.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate images even if they already exist.")
    return parser.parse_args()


def require_pillow():
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Pillow is required to export trajectory frames. Install Pillow in this environment and rerun."
        ) from exc
    return Image


def discover_record_dirs(record_dir: Path) -> list[Path]:
    if not record_dir.is_dir():
        raise SystemExit(f"record_dir does not exist or is not a directory: {record_dir}")

    records = [path for path in sorted(record_dir.glob("record_*")) if path.is_dir()]
    if not records:
        raise SystemExit(f"No record_* directories found in: {record_dir}")
    return records


def validate_record(record: Path) -> None:
    aligned = record / "aligned_60Hz"
    missing = [name for name in REQUIRED_ALIGNED_FILES if not (aligned / name).is_file()]
    if missing:
        missing_text = ", ".join(missing)
        raise SystemExit(f"{record} is missing required aligned_60Hz files: {missing_text}")


def load_mmap(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def native_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return native_value(value.item())
        return [native_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return native_value(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [native_value(item) for item in value]
    return str(value)


def read_metadata(record: Path) -> dict[str, Any]:
    metadata_path = record / "metadata.npz"
    if not metadata_path.is_file():
        return {}

    metadata: dict[str, Any] = {}
    with np.load(metadata_path, allow_pickle=False) as data:
        for key in data.files:
            metadata[key] = native_value(data[key])
    return metadata


def finite_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def array_summary(array: np.ndarray, names: list[str] | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if array.size == 0:
        return summary

    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    flat = values.reshape(values.shape[0], -1)
    mins = np.nanmin(flat, axis=0)
    maxs = np.nanmax(flat, axis=0)
    means = np.nanmean(flat, axis=0)
    limit = min(flat.shape[1], 12)
    labels = names or [f"dim_{index}" for index in range(flat.shape[1])]
    summary["dims"] = [
        {
            "name": labels[index] if index < len(labels) else f"dim_{index}",
            "min": finite_float(mins[index]),
            "max": finite_float(maxs[index]),
            "mean": finite_float(means[index]),
        }
        for index in range(limit)
    ]
    if flat.shape[1] > limit:
        summary["truncated_dims"] = flat.shape[1] - limit
    return summary


def optional_summaries(aligned: Path) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    optional_specs = {
        "ft": ("ft.npy", ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]),
        "xyz": ("xyz.npy", ["x", "y", "z"]),
        "width": ("width.npy", ["gripper_width"]),
    }

    for key, (filename, names) in optional_specs.items():
        path = aligned / filename
        if path.is_file():
            array = load_mmap(path)
            summary = array_summary(array, names)
            if key == "ft" and array.size:
                values = np.asarray(array, dtype=np.float64).reshape(array.shape[0], -1)
                if values.shape[1] >= 3:
                    force_norm = np.linalg.norm(values[:, :3], axis=1)
                    summary["force_norm_n"] = {
                        "min": finite_float(np.nanmin(force_norm)),
                        "max": finite_float(np.nanmax(force_norm)),
                        "mean": finite_float(np.nanmean(force_norm)),
                    }
            summaries[key] = summary
    return summaries


def rgb_to_image(Image, frame: np.ndarray):
    rgb = np.asarray(frame)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Expected RGB frame with shape (H, W, 3+), got {rgb.shape}")
    rgb = rgb[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def resize_to_width(Image, image, width: int):
    if width <= 0 or image.width == width:
        return image.copy()
    height = max(1, round(image.height * (width / image.width)))
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    return image.resize((width, height), resampling)


def save_jpeg(image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=quality, optimize=True)


def relative_path(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def export_record_images(
    *,
    record_name: str,
    rgb: np.ndarray,
    output_dir: Path,
    Image,
    jpeg_quality: int,
    frame_width: int,
    thumb_width: int,
    overwrite: bool,
) -> tuple[list[str], list[str], ExportStats]:
    frames_dir = output_dir / "frames" / record_name
    thumbs_dir = output_dir / "thumbs" / record_name
    frames_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[str] = []
    thumb_paths: list[str] = []
    saved_frames = 0
    saved_thumbs = 0
    skipped_frames = 0
    skipped_thumbs = 0

    for index in range(int(rgb.shape[0])):
        frame_path = frames_dir / f"frame_{index:06d}.jpg"
        thumb_path = thumbs_dir / f"frame_{index:06d}.jpg"
        frame_paths.append(relative_path(frame_path, output_dir))
        thumb_paths.append(relative_path(thumb_path, output_dir))

        need_frame = overwrite or not frame_path.is_file()
        need_thumb = overwrite or not thumb_path.is_file()
        if not need_frame and not need_thumb:
            skipped_frames += 1
            skipped_thumbs += 1
            continue

        image = rgb_to_image(Image, rgb[index])
        if need_frame:
            save_jpeg(resize_to_width(Image, image, frame_width), frame_path, jpeg_quality)
            saved_frames += 1
        else:
            skipped_frames += 1
        if need_thumb:
            save_jpeg(resize_to_width(Image, image, thumb_width), thumb_path, jpeg_quality)
            saved_thumbs += 1
        else:
            skipped_thumbs += 1

    return frame_paths, thumb_paths, ExportStats(saved_frames, saved_thumbs, skipped_frames, skipped_thumbs)


def collect_record_manifest(
    *,
    record: Path,
    output_dir: Path,
    Image,
    jpeg_quality: int,
    frame_width: int,
    thumb_width: int,
    overwrite: bool,
) -> tuple[dict[str, Any], ExportStats]:
    validate_record(record)
    aligned = record / "aligned_60Hz"
    timestamps = load_mmap(aligned / "timestamps.npy")
    rgb = load_mmap(aligned / "rgb.npy")
    actions = load_mmap(aligned / "action.npy")

    if rgb.ndim != 4:
        raise SystemExit(f"{record} aligned_60Hz/rgb.npy must have shape (T, H, W, C), got {rgb.shape}")
    if actions.ndim != 2:
        raise SystemExit(f"{record} aligned_60Hz/action.npy must have shape (T, D), got {actions.shape}")
    frame_count = min(int(timestamps.shape[0]), int(rgb.shape[0]), int(actions.shape[0]))
    if frame_count == 0:
        raise SystemExit(f"{record} has zero aligned frames")

    if frame_count != int(rgb.shape[0]) or frame_count != int(actions.shape[0]) or frame_count != int(timestamps.shape[0]):
        print(
            f"[WARN] {record.name}: timestamps/rgb/action lengths differ; using first {frame_count} aligned samples.",
            file=sys.stderr,
        )

    frame_paths, thumb_paths, stats = export_record_images(
        record_name=record.name,
        rgb=rgb[:frame_count],
        output_dir=output_dir,
        Image=Image,
        jpeg_quality=jpeg_quality,
        frame_width=frame_width,
        thumb_width=thumb_width,
        overwrite=overwrite,
    )

    action_dim = int(actions.shape[1])
    action_labels = [ACTION_LABELS[index] if index < len(ACTION_LABELS) else f"action_{index}" for index in range(action_dim)]
    timestamps_list = [finite_float(value) for value in np.asarray(timestamps[:frame_count], dtype=np.float64).tolist()]
    action_values = np.asarray(actions[:frame_count], dtype=np.float32)

    summary = {
        "record_dir": str(record),
        "frame_count": frame_count,
        "duration_s": finite_float(timestamps_list[-1] - timestamps_list[0]) if frame_count > 1 else 0.0,
        "timestamp_start_s": timestamps_list[0],
        "timestamp_end_s": timestamps_list[-1],
        "rgb_shape": list(rgb.shape),
        "action_shape": list(actions.shape),
        "metadata": read_metadata(record),
        "optional": optional_summaries(aligned),
    }

    return (
        {
            "name": record.name,
            "summary": summary,
            "frames": frame_paths,
            "thumbs": thumb_paths,
            "timestamps": timestamps_list,
            "actions": action_values.tolist(),
            "action_labels": action_labels,
        },
        stats,
    )


def make_manifest(records: list[dict[str, Any]], record_dir: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "title": "LabPick CAFE Trajectory Viewer",
        "record_dir": str(record_dir),
        "output_dir": str(output_dir),
        "generated_by": "scripts/vis/vis_traj.py",
        "action_labels": ACTION_LABELS,
        "records": records,
    }


def html_document(manifest: dict[str, Any]) -> str:
    manifest_json = json.dumps(manifest, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    return (
        r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LabPick CAFE Trajectory Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --panel-soft: #eef1f4;
      --text: #19202a;
      --muted: #687383;
      --line: #d9dee6;
      --accent: #0f766e;
      --accent-soft: #d6f2ef;
      --danger: #c2410c;
      --shadow: 0 10px 28px rgba(25, 32, 42, 0.08);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }

    button,
    input {
      font: inherit;
    }

    button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 6px;
      padding: 7px 11px;
      cursor: pointer;
    }

    button:hover {
      border-color: var(--accent);
    }

    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }

    input[type="range"] {
      width: 100%;
      touch-action: none;
    }

    input[type="number"] {
      width: 72px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 8px;
      background: #fff;
      color: var(--text);
    }

    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(260px, 330px) minmax(0, 1fr);
    }

    .sidebar {
      background: #fff;
      border-right: 1px solid var(--line);
      padding: 18px 16px;
      overflow: auto;
      max-height: 100vh;
    }

    .brand {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 16px;
    }

    .brand h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.15;
      font-weight: 700;
      letter-spacing: 0;
    }

    .brand span {
      color: var(--muted);
      white-space: nowrap;
      font-size: 12px;
    }

    .record-list {
      display: grid;
      gap: 8px;
    }

    .record-row {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      display: grid;
      gap: 6px;
      background: #fff;
      cursor: pointer;
    }

    .record-row.active {
      border-color: var(--accent);
      background: var(--accent-soft);
    }

    .record-row-main,
    .record-row-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
    }

    .record-name {
      font-weight: 700;
    }

    .badge {
      font-size: 12px;
      border-radius: 999px;
      padding: 2px 8px;
      background: var(--panel-soft);
      color: var(--muted);
    }

    .badge.fail {
      color: var(--danger);
      background: #ffedd5;
    }

    .badge.ok {
      color: #047857;
      background: #d1fae5;
    }

    .main {
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      gap: 14px;
      padding: 18px;
      min-width: 0;
    }

    .topbar,
    .viewer-grid,
    .thumb-strip {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .topbar {
      padding: 12px 14px;
      display: grid;
      gap: 10px;
    }

    .controls {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }

    .frame-readout {
      margin-left: auto;
      color: var(--muted);
      white-space: nowrap;
    }

    .viewer-grid {
      min-width: 0;
      padding: 14px;
      display: grid;
      grid-template-columns: minmax(320px, 0.88fr) minmax(360px, 1.12fr);
      gap: 14px;
    }

    .image-pane {
      min-width: 0;
      display: grid;
      gap: 10px;
      align-content: start;
    }

    .frame-image-wrap {
      background: #111827;
      border-radius: 8px;
      overflow: hidden;
      min-height: 260px;
      display: grid;
      place-items: center;
    }

    #frameImage {
      display: block;
      width: 100%;
      height: auto;
      max-height: 70vh;
      object-fit: contain;
    }

    .stats-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }

    .stat {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      min-width: 0;
    }

    .stat-label {
      color: var(--muted);
      font-size: 12px;
    }

    .stat-value {
      font-weight: 700;
      overflow-wrap: anywhere;
    }

    .chart-pane {
      min-width: 0;
      display: grid;
      grid-template-rows: auto minmax(280px, 1fr) auto;
      gap: 10px;
    }

    .action-controls {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-content: start;
    }

    .action-toggle {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
      background: #fff;
      font-size: 12px;
      white-space: nowrap;
    }

    .swatch {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
    }

    .chart-wrap {
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 280px;
      overflow: hidden;
      background: #fff;
    }

    #actionCanvas {
      display: block;
      width: 100%;
      height: 360px;
      cursor: crosshair;
      touch-action: none;
    }

    .action-values {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px 10px;
      font-size: 12px;
    }

    .action-value {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      border-bottom: 1px solid var(--panel-soft);
      padding-bottom: 4px;
      min-width: 0;
    }

    .action-value span:first-child {
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .thumb-strip {
      padding: 12px;
      overflow: auto;
      min-height: 128px;
    }

    .thumb-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(104px, 1fr));
      gap: 8px;
    }

    .thumb {
      border: 2px solid transparent;
      border-radius: 7px;
      background: #fff;
      padding: 0;
      overflow: hidden;
      position: relative;
      aspect-ratio: 4 / 3;
    }

    .thumb.active {
      border-color: var(--accent);
    }

    .thumb img {
      width: 100%;
      height: 100%;
      display: block;
      object-fit: cover;
    }

    .thumb span {
      position: absolute;
      left: 4px;
      bottom: 4px;
      padding: 1px 5px;
      border-radius: 999px;
      background: rgba(17, 24, 39, 0.72);
      color: #fff;
      font-size: 11px;
    }

    @media (max-width: 980px) {
      .app {
        grid-template-columns: 1fr;
      }

      .sidebar {
        max-height: none;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }

      .viewer-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <h1>LabPick CAFE</h1>
        <span id="recordCount"></span>
      </div>
      <div id="recordList" class="record-list"></div>
    </aside>

    <main class="main">
      <section class="topbar">
        <div class="controls">
          <button id="prevButton" type="button">Prev</button>
          <button id="playButton" class="primary" type="button">Play</button>
          <button id="nextButton" type="button">Next</button>
          <label>FPS <input id="fpsInput" type="number" min="1" max="60" value="12"></label>
          <div id="frameReadout" class="frame-readout"></div>
        </div>
        <input id="frameSlider" type="range" min="0" max="0" value="0">
      </section>

      <section class="viewer-grid">
        <div class="image-pane">
          <div class="frame-image-wrap">
            <img id="frameImage" alt="trajectory frame">
          </div>
          <div class="stats-grid">
            <div class="stat">
              <div class="stat-label">Record</div>
              <div id="recordName" class="stat-value"></div>
            </div>
            <div class="stat">
              <div class="stat-label">Timestamp</div>
              <div id="timestampValue" class="stat-value"></div>
            </div>
            <div class="stat">
              <div class="stat-label">Success</div>
              <div id="successValue" class="stat-value"></div>
            </div>
          </div>
        </div>

        <div class="chart-pane">
          <div id="actionControls" class="action-controls"></div>
          <div class="chart-wrap">
            <canvas id="actionCanvas"></canvas>
          </div>
          <div id="actionValues" class="action-values"></div>
        </div>
      </section>

      <section class="thumb-strip">
        <div id="thumbGrid" class="thumb-grid"></div>
      </section>
    </main>
  </div>

  <script id="manifest" type="application/json">__MANIFEST_JSON__</script>
  <script>
    const manifest = JSON.parse(document.getElementById("manifest").textContent);
    const colors = ["#0f766e", "#dc2626", "#2563eb", "#d97706", "#7c3aed", "#059669", "#be123c", "#0891b2", "#9333ea", "#4d7c0f", "#334155", "#ea580c"];
    const state = {
      recordIndex: 0,
      frameIndex: 0,
      playing: false,
      timer: null,
      visibleDims: new Set(),
      draggingChart: false,
    };

    const recordList = document.getElementById("recordList");
    const recordCount = document.getElementById("recordCount");
    const frameImage = document.getElementById("frameImage");
    const frameSlider = document.getElementById("frameSlider");
    const frameReadout = document.getElementById("frameReadout");
    const playButton = document.getElementById("playButton");
    const prevButton = document.getElementById("prevButton");
    const nextButton = document.getElementById("nextButton");
    const fpsInput = document.getElementById("fpsInput");
    const recordName = document.getElementById("recordName");
    const timestampValue = document.getElementById("timestampValue");
    const successValue = document.getElementById("successValue");
    const actionControls = document.getElementById("actionControls");
    const actionValues = document.getElementById("actionValues");
    const thumbGrid = document.getElementById("thumbGrid");
    const thumbStrip = document.querySelector(".thumb-strip");
    const canvas = document.getElementById("actionCanvas");
    const ctx = canvas.getContext("2d");

    function currentRecord() {
      return manifest.records[state.recordIndex];
    }

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function fmt(value, digits = 4) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
      return Number(value).toFixed(digits);
    }

    function successLabel(record) {
      const success = record.summary.metadata.success;
      if (success === true) return "true";
      if (success === false) return "false";
      return "n/a";
    }

    function buildRecordList() {
      recordCount.textContent = `${manifest.records.length} records`;
      recordList.innerHTML = "";
      manifest.records.forEach((record, index) => {
        const row = document.createElement("button");
        row.type = "button";
        row.className = "record-row";
        row.dataset.index = index;
        const success = successLabel(record);
        const successClass = success === "true" ? "ok" : success === "false" ? "fail" : "";
        row.innerHTML = `
          <div class="record-row-main">
            <span class="record-name">${record.name}</span>
            <span class="badge ${successClass}">${success}</span>
          </div>
          <div class="record-row-meta">
            <span>${record.summary.frame_count} frames</span>
            <span>${fmt(record.summary.duration_s, 2)}s</span>
          </div>
        `;
        row.addEventListener("click", () => setRecord(index));
        recordList.appendChild(row);
      });
    }

    function buildActionControls(record) {
      actionControls.innerHTML = "";
      state.visibleDims.clear();
      record.action_labels.forEach((label, index) => {
        state.visibleDims.add(index);
        const item = document.createElement("label");
        item.className = "action-toggle";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = true;
        checkbox.addEventListener("change", () => {
          if (checkbox.checked) state.visibleDims.add(index);
          else state.visibleDims.delete(index);
          drawChart();
          renderActionValues();
        });
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.backgroundColor = colors[index % colors.length];
        const text = document.createElement("span");
        text.textContent = label;
        item.append(checkbox, swatch, text);
        actionControls.appendChild(item);
      });
    }

    function buildThumbs(record) {
      thumbGrid.innerHTML = "";
      record.thumbs.forEach((src, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "thumb";
        button.dataset.index = index;
        const img = document.createElement("img");
        img.loading = "lazy";
        img.decoding = "async";
        img.src = src;
        img.alt = `${record.name} frame ${index}`;
        const label = document.createElement("span");
        label.textContent = index;
        button.append(img, label);
        button.addEventListener("click", () => setFrame(index, true));
        thumbGrid.appendChild(button);
      });
    }

    function setRecord(index) {
      stopPlayback();
      state.recordIndex = clamp(index, 0, manifest.records.length - 1);
      state.frameIndex = 0;
      const record = currentRecord();

      document.querySelectorAll(".record-row").forEach((row) => {
        row.classList.toggle("active", Number(row.dataset.index) === state.recordIndex);
      });

      frameSlider.max = Math.max(0, record.frames.length - 1);
      buildActionControls(record);
      buildThumbs(record);
      setFrame(0, false);
    }

    function setFrame(index, scrollThumb) {
      const record = currentRecord();
      if (!record || record.frames.length === 0) return;
      state.frameIndex = clamp(Math.round(index), 0, record.frames.length - 1);

      frameImage.src = record.frames[state.frameIndex];
      frameSlider.value = state.frameIndex;
      frameReadout.textContent = `Frame ${state.frameIndex + 1} / ${record.frames.length}`;
      recordName.textContent = record.name;
      timestampValue.textContent = `${fmt(record.timestamps[state.frameIndex], 4)} s`;
      successValue.textContent = successLabel(record);

      document.querySelectorAll(".thumb.active").forEach((thumb) => thumb.classList.remove("active"));
      const activeThumb = thumbGrid.querySelector(`[data-index="${state.frameIndex}"]`);
      if (activeThumb) {
        activeThumb.classList.add("active");
        if (scrollThumb) scrollThumbWithinStrip(activeThumb);
      }

      drawChart();
      renderActionValues();
    }

    function scrollThumbWithinStrip(activeThumb) {
      const stripRect = thumbStrip.getBoundingClientRect();
      const thumbRect = activeThumb.getBoundingClientRect();
      const padding = 8;

      if (thumbRect.top < stripRect.top) {
        thumbStrip.scrollTop -= stripRect.top - thumbRect.top + padding;
      } else if (thumbRect.bottom > stripRect.bottom) {
        thumbStrip.scrollTop += thumbRect.bottom - stripRect.bottom + padding;
      }

      if (thumbRect.left < stripRect.left) {
        thumbStrip.scrollLeft -= stripRect.left - thumbRect.left + padding;
      } else if (thumbRect.right > stripRect.right) {
        thumbStrip.scrollLeft += thumbRect.right - stripRect.right + padding;
      }
    }

    function renderActionValues() {
      const record = currentRecord();
      const values = record.actions[state.frameIndex] || [];
      actionValues.innerHTML = "";
      record.action_labels.forEach((label, index) => {
        const row = document.createElement("div");
        row.className = "action-value";
        const name = document.createElement("span");
        name.textContent = label;
        const value = document.createElement("strong");
        value.textContent = fmt(values[index], 5);
        value.style.color = colors[index % colors.length];
        row.append(name, value);
        actionValues.appendChild(row);
      });
    }

    function selectedDims(record) {
      const dims = [...state.visibleDims].filter((index) => index < record.action_labels.length);
      return dims.length ? dims : [0];
    }

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(320, Math.floor(rect.width));
      const height = Math.max(280, Math.floor(rect.height));
      if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { width, height };
    }

    function actionDomain(record, dims) {
      let min = Infinity;
      let max = -Infinity;
      record.actions.forEach((row) => {
        dims.forEach((dim) => {
          const value = Number(row[dim]);
          if (Number.isFinite(value)) {
            min = Math.min(min, value);
            max = Math.max(max, value);
          }
        });
      });
      if (!Number.isFinite(min) || !Number.isFinite(max)) {
        return [-1, 1];
      }
      if (Math.abs(max - min) < 1e-9) {
        const pad = Math.max(1e-3, Math.abs(max) * 0.1);
        return [min - pad, max + pad];
      }
      const pad = (max - min) * 0.08;
      return [min - pad, max + pad];
    }

    function drawChart() {
      const record = currentRecord();
      if (!record) return;
      const { width, height } = resizeCanvas();
      const dims = selectedDims(record);
      const [minY, maxY] = actionDomain(record, dims);
      const left = 56;
      const right = 18;
      const top = 16;
      const bottom = 34;
      const plotW = Math.max(1, width - left - right);
      const plotH = Math.max(1, height - top - bottom);
      const n = record.actions.length;

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);

      ctx.strokeStyle = "#e5e7eb";
      ctx.lineWidth = 1;
      ctx.fillStyle = "#687383";
      ctx.font = "12px system-ui, sans-serif";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let tick = 0; tick <= 4; tick += 1) {
        const y = top + (plotH * tick) / 4;
        const value = maxY - ((maxY - minY) * tick) / 4;
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(width - right, y);
        ctx.stroke();
        ctx.fillText(fmt(value, 3), left - 8, y);
      }

      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let tick = 0; tick <= 4; tick += 1) {
        const x = left + (plotW * tick) / 4;
        const index = Math.round(((n - 1) * tick) / 4);
        const t = record.timestamps[index] ?? 0;
        ctx.beginPath();
        ctx.moveTo(x, top);
        ctx.lineTo(x, top + plotH);
        ctx.stroke();
        ctx.fillText(`${fmt(t, 2)}s`, x, top + plotH + 8);
      }

      ctx.save();
      ctx.rect(left, top, plotW, plotH);
      ctx.clip();
      dims.forEach((dim) => {
        ctx.strokeStyle = colors[dim % colors.length];
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        let started = false;
        for (let index = 0; index < n; index += 1) {
          const value = Number(record.actions[index][dim]);
          if (!Number.isFinite(value)) {
            started = false;
            continue;
          }
          const x = left + (n <= 1 ? 0 : (plotW * index) / (n - 1));
          const y = top + plotH - ((value - minY) / (maxY - minY)) * plotH;
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      });
      ctx.restore();

      const cursorX = left + (n <= 1 ? 0 : (plotW * state.frameIndex) / (n - 1));
      ctx.strokeStyle = "#111827";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cursorX, top);
      ctx.lineTo(cursorX, top + plotH);
      ctx.stroke();

      ctx.strokeStyle = "#9ca3af";
      ctx.strokeRect(left, top, plotW, plotH);
    }

    function chartIndexFromEvent(event) {
      const record = currentRecord();
      const rect = canvas.getBoundingClientRect();
      const left = 56;
      const right = 18;
      const plotW = Math.max(1, rect.width - left - right);
      const x = clamp(event.clientX - rect.left - left, 0, plotW);
      return Math.round((x / plotW) * (record.frames.length - 1));
    }

    function startPlayback() {
      if (state.playing) return;
      state.playing = true;
      playButton.textContent = "Pause";
      const fps = clamp(Number(fpsInput.value) || 12, 1, 60);
      state.timer = window.setInterval(() => {
        const record = currentRecord();
        const next = state.frameIndex + 1;
        setFrame(next >= record.frames.length ? 0 : next, false);
      }, 1000 / fps);
    }

    function stopPlayback() {
      state.playing = false;
      playButton.textContent = "Play";
      if (state.timer !== null) {
        window.clearInterval(state.timer);
        state.timer = null;
      }
    }

    playButton.addEventListener("click", () => {
      if (state.playing) stopPlayback();
      else startPlayback();
    });
    prevButton.addEventListener("click", () => setFrame(state.frameIndex - 1, true));
    nextButton.addEventListener("click", () => setFrame(state.frameIndex + 1, true));
    fpsInput.addEventListener("change", () => {
      if (state.playing) {
        stopPlayback();
        startPlayback();
      }
    });
    frameSlider.addEventListener("input", () => setFrame(Number(frameSlider.value), false));

    canvas.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      state.draggingChart = true;
      canvas.setPointerCapture(event.pointerId);
      setFrame(chartIndexFromEvent(event), false);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (state.draggingChart) {
        event.preventDefault();
        setFrame(chartIndexFromEvent(event), false);
      }
    });
    canvas.addEventListener("pointerup", (event) => {
      state.draggingChart = false;
      canvas.releasePointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointercancel", () => {
      state.draggingChart = false;
    });
    window.addEventListener("resize", drawChart);

    buildRecordList();
    setRecord(0);
  </script>
</body>
</html>
"""
    ).replace("__MANIFEST_JSON__", manifest_json)


def write_html(output_dir: Path, manifest: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.html"
    index_path.write_text(html_document(manifest), encoding="utf-8")
    return index_path


def main() -> int:
    args = parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg_quality must be between 1 and 100")
    if args.thumb_width <= 0:
        raise SystemExit("--thumb_width must be positive")
    if args.frame_width <= 0:
        raise SystemExit("--frame_width must be positive")

    Image = require_pillow()
    record_dir = args.record_dir.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else record_dir / "vis_traj_html")
    records = discover_record_dirs(record_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_records: list[dict[str, Any]] = []
    total_saved_frames = 0
    total_saved_thumbs = 0
    total_skipped_frames = 0
    total_skipped_thumbs = 0

    print(f"[INFO] record_dir={record_dir}")
    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] discovered {len(records)} records")

    for record in records:
        print(f"[INFO] exporting {record.name}")
        record_manifest, stats = collect_record_manifest(
            record=record,
            output_dir=output_dir,
            Image=Image,
            jpeg_quality=args.jpeg_quality,
            frame_width=args.frame_width,
            thumb_width=args.thumb_width,
            overwrite=args.overwrite,
        )
        manifest_records.append(record_manifest)
        total_saved_frames += stats.saved_frames
        total_saved_thumbs += stats.saved_thumbs
        total_skipped_frames += stats.skipped_frames
        total_skipped_thumbs += stats.skipped_thumbs
        print(
            "[INFO] "
            f"{record.name}: frames={record_manifest['summary']['frame_count']} "
            f"saved_frames={stats.saved_frames} saved_thumbs={stats.saved_thumbs} "
            f"skipped_frames={stats.skipped_frames} skipped_thumbs={stats.skipped_thumbs}"
        )

    manifest = make_manifest(manifest_records, record_dir, output_dir)
    index_path = write_html(output_dir, manifest)
    print(
        "[INFO] export complete: "
        f"saved_frames={total_saved_frames} saved_thumbs={total_saved_thumbs} "
        f"skipped_frames={total_skipped_frames} skipped_thumbs={total_skipped_thumbs}"
    )
    print(f"[INFO] open {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
python scripts/vis/vis_traj.py --record_dir ./dataset/success
"""
