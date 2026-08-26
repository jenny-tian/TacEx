from __future__ import annotations

import importlib.util
import os
import sys


_CLEAN_MARKER = "TACEX_DSRL_VGL_CLEAN"


def reexec_without_virtualgl() -> None:
    """Restart the current command without TurboVNC/VirtualGL CUDA interposition.

    The workstation exports ``LD_PRELOAD=libdlfaker.so:libvglfaker.so`` for the
    remote desktop. Those libraries intercept ``dlopen`` and prevent cuDNN 9
    from loading ``libcudnn_graph``. Removing the variables after importing
    PyTorch is too late, so DSRL entrypoints call this before importing torch.
    """

    if os.environ.get(_CLEAN_MARKER) == "1":
        return
    preload = os.environ.get("LD_PRELOAD", "")
    if "libdlfaker" not in preload and "libvglfaker" not in preload:
        return

    clean_env = os.environ.copy()
    clean_env.pop("LD_PRELOAD", None)
    clean_env.pop("VGL_ISACTIVE", None)
    clean_env.pop("VGL_DISPLAY", None)
    clean_env.pop("DISPLAY", None)
    clean_env[_CLEAN_MARKER] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], clean_env)


def disable_optional_transformers_discovery() -> None:
    """Keep LeRobot's eager policy imports from loading unrelated VLA stacks.

    The Isaac environment already contains a newer Transformers installation,
    while the isolated Diffusion inference dependencies intentionally pin the
    Hugging Face Hub version expected by LeRobot 0.4.3. Isaac also exposes a
    boto3/botocore combination that is irrelevant to local inference and
    version-incompatible with Accelerate's optional SageMaker path. DSRL only
    needs the Diffusion policy, so hide both optional stacks during discovery.
    """

    original_find_spec = importlib.util.find_spec

    def find_spec(name: str, package: str | None = None):
        root_name = name.partition(".")[0]
        if root_name in {"transformers", "boto3"}:
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = find_spec
