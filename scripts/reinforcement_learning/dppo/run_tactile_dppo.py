#!/usr/bin/env python3
"""Launch one exact-episode tactile DPPO run in the Isaac environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    isaac_python = Path(
        os.environ.get(
            "TACEX_ISAAC_PYTHON",
            "/home/limx/anaconda3/envs/env_isaaclab/bin/python",
        )
    ).expanduser().resolve()
    if not isaac_python.is_file():
        raise FileNotFoundError(f"Isaac Python not found: {isaac_python}")
    environment = os.environ.copy()
    for variable in ("LD_PRELOAD", "VGL_ISACTIVE", "VGL_DISPLAY", "DISPLAY"):
        environment.pop(variable, None)
    roots = [
        repo_root / "scripts" / "reinforcement_learning" / "dppo",
        repo_root / "scripts" / "reinforcement_learning" / "dsrl",
        repo_root / "scripts" / "reinforcement_learning" / "compare_exp",
        repo_root / "scripts" / "lerobot" / "src",
        repo_root / "source" / "tacex",
        repo_root / "source" / "tacex_assets",
        repo_root / "source" / "tacex_tasks",
    ]
    if environment.get("PYTHONPATH"):
        roots.append(Path(environment["PYTHONPATH"]))
    environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in roots)
    environment.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    command = [
        str(isaac_python),
        str(Path(__file__).with_name("run_tactile_dppo_runtime.py")),
        "--headless",
        "--enable_cameras",
        *sys.argv[1:],
    ]
    print("[COMMAND]", " ".join(command), flush=True)
    os.chdir(repo_root)
    os.execvpe(str(isaac_python), command, environment)


if __name__ == "__main__":
    main()
