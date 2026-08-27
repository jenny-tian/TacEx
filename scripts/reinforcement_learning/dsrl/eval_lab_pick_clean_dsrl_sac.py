#!/usr/bin/env python3
"""Launch clean DSRL evaluation in the configured IsaacLab Python."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    isaac_python_value = os.environ.get("TACEX_ISAAC_PYTHON")
    if not isaac_python_value:
        raise RuntimeError("Set TACEX_ISAAC_PYTHON to the IsaacLab Python executable.")
    isaac_python = Path(isaac_python_value)
    if not isaac_python.is_file():
        raise FileNotFoundError(f"Isaac Python not found: {isaac_python}")
    if not any(
        argument == "--bc_policy" or argument.startswith("--bc_policy=")
        for argument in sys.argv[1:]
    ):
        raise SystemExit("Pass --bc_policy /path/to/flow_matching_checkpoint.pt")

    env = os.environ.copy()
    for variable in ("LD_PRELOAD", "VGL_ISACTIVE", "VGL_DISPLAY", "DISPLAY"):
        env.pop(variable, None)
    python_paths = [
        repo_root / "scripts" / "lerobot" / "src",
        repo_root / "source" / "tacex",
        repo_root / "source" / "tacex_assets",
        repo_root / "source" / "tacex_tasks",
        repo_root / "bc_policy",
    ]
    if env.get("PYTHONPATH"):
        python_paths.append(Path(env["PYTHONPATH"]))
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in python_paths)
    command = [
        str(isaac_python),
        str(Path(__file__).with_name("eval_lab_pick_clean_dsrl_sac_runtime.py")),
        "--headless",
        "--enable_cameras",
        *sys.argv[1:],
    ]
    print("[COMMAND]", " ".join(command), flush=True)
    os.chdir(repo_root)
    os.execvpe(str(isaac_python), command, env)


if __name__ == "__main__":
    main()
