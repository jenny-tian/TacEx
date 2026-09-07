import gymnasium as gym

from .grape_pick_env_cfg import GrapePickEnvCfg


gym.register(
    id="TacEx-GrapePick-Force-v0",
    entry_point=f"{__name__}.grape_pick_env:GrapePickEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": GrapePickEnvCfg},
)
