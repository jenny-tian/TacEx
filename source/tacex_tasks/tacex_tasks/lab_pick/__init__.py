import gymnasium as gym

from . import agents
from .lab_pick_env_cfg import (
    LabPickCoverslipEnvCfg,
    LabPickCupEnvCfg,
    LabPickSlideEnvCfg,
    LabPickSlideDSRLBaseEnvCfg,
    LabPickSlideCleanResidualSACEnvCfg,
    LabPickSlideSACEnvCfg,
)

SAC_CFG_ENTRY_POINT = f"{agents.__name__}:skrl_sac_cfg.yaml"
DSRL_SAC_CFG_ENTRY_POINT = f"{agents.__name__}:skrl_dsrl_sac_cfg.yaml"
CLEAN_RESIDUAL_SAC_CFG_ENTRY_POINT = f"{agents.__name__}:skrl_clean_residual_sac_cfg.yaml"

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


gym.register(
    id="TacEx-LabPick-Slide-SAC-v0",
    entry_point=f"{__name__}.lab_pick_env:LabPickEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LabPickSlideSACEnvCfg,
        "skrl_sac_cfg_entry_point": SAC_CFG_ENTRY_POINT,
    },
)


gym.register(
    id="TacEx-LabPick-Slide-DSRL-Base-v0",
    entry_point=f"{__name__}.lab_pick_env:LabPickEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LabPickSlideDSRLBaseEnvCfg,
        "skrl_sac_cfg_entry_point": DSRL_SAC_CFG_ENTRY_POINT,
    },
)


gym.register(
    id="TacEx-LabPick-Slide-Clean-Residual-SAC-v0",
    entry_point=f"{__name__}.lab_pick_env:LabPickEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LabPickSlideCleanResidualSACEnvCfg,
        "skrl_clean_sac_cfg_entry_point": CLEAN_RESIDUAL_SAC_CFG_ENTRY_POINT,
    },
)
