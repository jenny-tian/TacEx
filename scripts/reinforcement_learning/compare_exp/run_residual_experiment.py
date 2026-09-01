#!/usr/bin/env python3
"""Launch one exact-episode residual-RL comparison run."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    value = os.environ.get("TACEX_ISAAC_PYTHON")
    if not value:
        raise RuntimeError("Set TACEX_ISAAC_PYTHON to the IsaacLab Python executable.")
    isaac_python = Path(value).expanduser().resolve()
    if not isaac_python.is_file():
        raise FileNotFoundError(isaac_python)
    env = os.environ.copy()
    for variable in ("LD_PRELOAD", "VGL_ISACTIVE", "VGL_DISPLAY", "DISPLAY"):
        env.pop(variable, None)
    roots = [
        repo_root / "scripts" / "reinforcement_learning" / "dsrl",
        repo_root / "scripts" / "reinforcement_learning" / "vlm_dsrl",
        repo_root / "scripts" / "reinforcement_learning" / "compare_exp",
        repo_root / "scripts" / "bc_training",
        repo_root / "source" / "tacex",
        repo_root / "source" / "tacex_assets",
        repo_root / "source" / "tacex_tasks",
        repo_root / "bc_policy",
    ]
    if env.get("PYTHONPATH"):
        roots.append(Path(env["PYTHONPATH"]))
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in roots)
    command = [
        str(isaac_python),
        str(Path(__file__).with_name("run_residual_experiment_runtime.py")),
        "--headless",
        "--enable_cameras",
        *sys.argv[1:],
    ]
    print("[COMMAND]", " ".join(command), flush=True)
    os.chdir(repo_root)
    os.execvpe(str(isaac_python), command, env)


if __name__ == "__main__":
    main()

