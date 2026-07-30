"""UI compatibility helpers for headless Isaac Sim applications."""

try:
    from isaaclab.envs.ui import BaseEnvWindow as BaseEnvWindow
except ModuleNotFoundError as exc:
    if exc.name != "omni.ui":
        raise

    class BaseEnvWindow:
        """Placeholder used only while the ``omni.ui`` extension is disabled."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError("Environment windows require a non-headless Isaac Sim application.")
