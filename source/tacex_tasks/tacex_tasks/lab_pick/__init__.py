import gymnasium as gym

from .lab_pick_env_cfg import LabPickCoverslipEnvCfg, LabPickCupEnvCfg, LabPickSlideEnvCfg


gym.register(
    id="TacEx-LabPick-Slide-Direct-v0",
    entry_point=f"{__name__}.lab_pick_env:LabPickEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": LabPickSlideEnvCfg},
)

gym.register(
    id="TacEx-LabPick-Coverslip-Direct-v0",
    entry_point=f"{__name__}.lab_pick_env:LabPickEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": LabPickCoverslipEnvCfg},
)

gym.register(
    id="TacEx-LabPick-Cup-Direct-v0",
    entry_point=f"{__name__}.lab_pick_env:LabPickEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": LabPickCupEnvCfg},
)
