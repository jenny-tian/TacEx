from __future__ import annotations


try:
    from tacex_tasks.lab_pick.visuals import (
        SLIDE_VISUAL_DIFFUSE_COLOR,
        SLIDE_VISUAL_OPACITY,
        SLIDE_VISUAL_ROUGHNESS,
    )
except ModuleNotFoundError:
    SLIDE_VISUAL_DIFFUSE_COLOR = (0.25, 0.75, 1.0)
    SLIDE_VISUAL_OPACITY = 0.9
    SLIDE_VISUAL_ROUGHNESS = 0.18
