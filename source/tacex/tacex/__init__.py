"""
Python module for simulating GelSight sensors inside Isaac Sim/Lab
"""

from .gelsight_sensor import GelSightSensor
from .gelsight_sensor_cfg import GelSightSensorCfg
from .gelsight_sensor_data import GelSightSensorData

__all__ = ["GelSightSensor", "GelSightSensorCfg", "GelSightSensorData"]

# The example extension is GUI-only; keep sensor imports usable in headless Isaac.
try:
    from .ui_extension_example import UsdrtExamplePythonExtension
except ModuleNotFoundError as exc:
    if exc.name != "omni.ui":
        raise
else:
    __all__.append("UsdrtExamplePythonExtension")
