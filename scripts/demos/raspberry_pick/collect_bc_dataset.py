#!/usr/bin/env python3
"""Collect randomized dual-camera RaspberryPick BC demonstrations."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


os.environ["TACEX_PICK_VARIANT"] = "raspberry"
TARGET = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "demos"
    / "grape_pick"
    / "collect_dataset.py"
)
runpy.run_path(str(TARGET), run_name="__main__")
