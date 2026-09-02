import gymnasium as gym

from .lab_surface_env_cfg import LabSurfaceForceScanEnvCfg

gym.register(
    id="TacEx-LabSurface-ForceScan-v0",
    entry_point=f"{__name__}.lab_surface_env:LabSurfaceForceScanEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": LabSurfaceForceScanEnvCfg},
)
