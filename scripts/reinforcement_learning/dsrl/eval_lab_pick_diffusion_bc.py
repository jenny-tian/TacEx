#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    inference_deps = repo_root / ".cache" / "lerobot_inference"
    if not inference_deps.is_dir():
        raise FileNotFoundError(
            f"Missing isolated LeRobot inference dependencies at {inference_deps}. "
            "Follow scripts/reinforcement_learning/dsrl/README.md."
        )
    if not any(arg == "--policy" or arg.startswith("--policy=") for arg in sys.argv[1:]):
        raise SystemExit("Pass --policy /path/to/diffusion/checkpoints/last/pretrained_model")

    isaac_python = Path(
        os.environ.get(
            "TACEX_ISAAC_PYTHON",
            "/home/tjx/miniforge3/envs/env_isaaclab/bin/python",
        )
    )
    if not isaac_python.is_file():
        raise FileNotFoundError(f"Isaac Python not found: {isaac_python}")

    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    env.pop("VGL_ISACTIVE", None)
    env.pop("VGL_DISPLAY", None)
    env.pop("DISPLAY", None)
    python_paths = [
        inference_deps,
        repo_root / "scripts" / "lerobot" / "src",
        repo_root / "source" / "tacex",
        repo_root / "source" / "tacex_assets",
        repo_root / "source" / "tacex_tasks",
    ]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        python_paths.append(Path(existing_pythonpath))
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in python_paths)

    command = [
        str(isaac_python),
        str(Path(__file__).with_name("eval_lab_pick_diffusion_bc_runtime.py")),
        *sys.argv[1:],
    ]
    print("[COMMAND]", " ".join(command), flush=True)
    os.chdir(repo_root)
    os.execvpe(str(isaac_python), command, env)


if __name__ == "__main__":
    main()
