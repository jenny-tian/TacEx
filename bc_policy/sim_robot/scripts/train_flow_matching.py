from __future__ import annotations

import os
import sys
from pathlib import Path


def _reexec_with_cudnn_library_path() -> None:
    python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    cudnn_dir = Path(sys.prefix) / "lib" / python_dir / "site-packages" / "nvidia" / "cudnn" / "lib"
    if not cudnn_dir.is_dir():
        return
    current_paths = [value for value in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if value]
    cudnn_path = str(cudnn_dir)
    if cudnn_path in current_paths:
        return
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join([cudnn_path, *current_paths])
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


_reexec_with_cudnn_library_path()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sim_robot.training.trainer import main


if __name__ == "__main__":
    main()

