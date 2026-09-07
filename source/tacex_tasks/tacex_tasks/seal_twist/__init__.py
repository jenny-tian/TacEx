import gymnasium as gym

from .seal_twist_env_cfg import SealTwistEnvCfg


gym.register(
    id="TacEx-SealTwist-Torque-v0",
    entry_point=f"{__name__}.seal_twist_env:SealTwistEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": SealTwistEnvCfg},
)
