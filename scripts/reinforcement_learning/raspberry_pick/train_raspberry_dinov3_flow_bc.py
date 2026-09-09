#!/usr/bin/env python3
"""Train RaspberryPick BC with frozen DINOv3 and conditional Flow Matching."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


os.environ["TACEX_PICK_VARIANT"] = "raspberry"
TARGET = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "reinforcement_learning"
    / "grape_pick"
    / "train_grape_dinov3_flow_bc.py"
)
runpy.run_path(str(TARGET), run_name="__main__")
