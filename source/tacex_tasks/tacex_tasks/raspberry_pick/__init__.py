import gymnasium as gym

from .raspberry_pick_env_cfg import RaspberryPickEnvCfg


gym.register(
    id="TacEx-RaspberryPick-Force-v0",
    entry_point=f"{__name__}.raspberry_pick_env:RaspberryPickEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": RaspberryPickEnvCfg},
)

