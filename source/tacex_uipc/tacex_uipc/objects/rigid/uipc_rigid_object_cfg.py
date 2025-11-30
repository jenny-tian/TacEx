from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import omni.physics.tensors.impl.api as physx
import omni.usd
import usdrt
import usdrt.UsdGeom
from isaacsim.core.prims import XFormPrim
from pxr import UsdGeom

try:
    from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
except ImportError:
    import warnings

    warnings.warn("_debug_draw failed to import", ImportWarning)
    draw = None

import numpy as np

import warp as wp
from uipc import builtin, view
from uipc.constitution import AffineBodyConstitution, ElasticModuli, StableNeoHookean
from uipc.geometry import extract_surface, flip_inward_triangles, label_surface, label_triangle_orient, tetmesh
from uipc.unit import MPa

import isaaclab.utils.string as string_utils
from isaaclab.assets import AssetBase, AssetBaseCfg
from isaaclab.utils import configclass

wp.init()


from tacex_uipc.utils import MeshGenerator, TetMeshCfg

from .uipc_rigid_object import UipcRigidObject

from ..uipc_object import UipcObjectCfg


@configclass
class UipcRigidObjectCfg(UipcObjectCfg):
    class_type: type = UipcRigidObject

    # contact_model:

    @configclass
    class AffineBodyConstitutionCfg:
        # class_type = AffineBodyConstitution # doesn't work, cause no builtin signature found for AffineBodyConstitution class
        m_kappa: float = 100.0
        """Stiffness (hardness) of the object
        in [MPa]

        E.g. 100.0 MPa = hard-rubber-like material
        """

        kinematic: bool = False
        """Makes the DoF of the ABD body fixed.

        """

    constitution_cfg: AffineBodyConstitutionCfg = AffineBodyConstitutionCfg()
