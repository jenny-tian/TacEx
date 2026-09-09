from __future__ import annotations

import math
from pathlib import Path

from tacex_tasks.grape_pick.grape_pick_env import GrapePickEnv

from .raspberry_pick_env_cfg import (
    LEAF_COLORS,
    RASPBERRY_DARK,
    RASPBERRY_HIGHLIGHT,
    RASPBERRY_RED,
)
from .raspberry_pick_env_cfg import RaspberryPickEnvCfg


class RaspberryPickEnv(GrapePickEnv):
    """Harvest one delicate raspberry from a layered outdoor-style grove."""

    cfg: RaspberryPickEnvCfg

    def __init__(self, cfg: RaspberryPickEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Semantic aliases keep the inherited, thoroughly tested force and
        # tether implementation while exposing raspberry-specific state.
        self.raspberry = self.grape
        self.raspberry_reset_pos_w = self.grape_reset_pos_w
        self.raspberry_reset_quat_w = self.grape_reset_quat_w

    def _spawn_leaf_visuals(self):
        """Create layered serrated leaf meshes instead of spherical foliage."""

        try:
            import omni.usd
            from pxr import Gf, Sdf, UsdGeom, UsdShade
        except Exception:
            return
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        self._spawn_textured_backdrops(stage, Gf, Sdf, UsdGeom, UsdShade)

        leaf_materials = []
        for index, color in enumerate(LEAF_COLORS):
            leaf_materials.append(
                self._make_preview_material(
                    stage,
                    Gf,
                    Sdf,
                    UsdShade,
                    f"/World/envs/env_0/Looks/RaspberryLeaf_{index}",
                    color,
                    roughness=0.67 + 0.035 * index,
                )
            )
        vein_material = self._make_preview_material(
            stage,
            Gf,
            Sdf,
            UsdShade,
            "/World/envs/env_0/Looks/RaspberryLeafVein",
            (0.28, 0.39, 0.105),
            roughness=0.82,
        )

        for index, (center, angle, length, width, _color) in enumerate(
            self.cfg.leaf_visuals
        ):
            self._spawn_serrated_leaf(
                stage,
                Gf,
                UsdGeom,
                UsdShade,
                index,
                center,
                angle,
                length,
                width,
                leaf_materials[index % len(leaf_materials)],
                vein_material,
            )
        self._spawn_background_raspberries(stage, Gf, Sdf, UsdGeom, UsdShade)
        self._spawn_target_raspberry_visual()

    @staticmethod
    def _spawn_textured_backdrops(stage, Gf, Sdf, UsdGeom, UsdShade) -> None:
        """Add CC0 raspberry foliage at the two sight-line termini."""

        texture_path = (
            Path(__file__).resolve().parent / "assets" / "raspberry_grove_cc0.jpg"
        )
        if not texture_path.is_file():
            return

        material_path = "/World/envs/env_0/Looks/RaspberryGroveBackdrop"
        material = UsdShade.Material.Define(stage, material_path)
        surface = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.88)
        surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

        reader = UsdShade.Shader.Define(stage, f"{material_path}/TexCoordReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        texture = UsdShade.Shader.Define(stage, f"{material_path}/DiffuseTexture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(str(texture_path))
        )
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            texture.ConnectableAPI(), "rgb"
        )
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")

        panels = (
            (
                "rear",
                (
                    (-0.90, 0.552, -0.12),
                    (1.55, 0.552, -0.12),
                    (1.55, 0.552, 1.30),
                    (-0.90, 0.552, 1.30),
                ),
                ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            ),
            (
                "side",
                (
                    (1.34, -1.10, -0.12),
                    (1.34, 0.58, -0.12),
                    (1.34, 0.58, 1.30),
                    (1.34, -1.10, 1.30),
                ),
                ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            ),
        )
        for name, coordinates, texcoords in panels:
            mesh = UsdGeom.Mesh.Define(
                stage, f"/World/envs/env_0/raspberry_grove_{name}_backdrop"
            )
            mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in coordinates])
            mesh.CreateFaceVertexCountsAttr([4])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)
            st_primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
            )
            st_primvar.Set([Gf.Vec2f(*uv) for uv in texcoords])
            UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)

    @staticmethod
    def _make_preview_material(
        stage,
        Gf,
        Sdf,
        UsdShade,
        path: str,
        color: tuple[float, float, float],
        *,
        roughness: float,
    ):
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    @staticmethod
    def _spawn_serrated_leaf(
        stage,
        Gf,
        UsdGeom,
        UsdShade,
        index: int,
        center,
        angle: float,
        length: float,
        width: float,
        leaf_material,
        vein_material,
    ) -> None:
        """Build one gently folded leaflet with serrated margins and veins."""

        cosine = math.cos(angle)
        sine = math.sin(angle)
        major = (cosine, 0.0, sine)
        minor = (-sine, 0.0, cosine)
        rows = tuple(-0.5 + item / 24.0 for item in range(25))
        points = []
        for row_index, u_value in enumerate(rows):
            profile = max(0.035, math.sin(math.pi * (u_value + 0.5)))
            serration = 1.0 + (0.075 if row_index % 2 else -0.025)
            half_width = 0.5 * width * profile * serration
            for side in (-1.0, 0.0, 1.0):
                ridge = -0.0065 * (1.0 - side * side) * profile
                edge_curl = 0.0025 * side * math.sin(2.0 * math.pi * (u_value + 0.5))
                points.append(
                    Gf.Vec3f(
                        center[0]
                        + major[0] * u_value * length
                        + minor[0] * side * half_width,
                        center[1] + ridge + edge_curl,
                        center[2]
                        + major[2] * u_value * length
                        + minor[2] * side * half_width,
                    )
                )

        faces = []
        for row_index in range(len(rows) - 1):
            left, middle, right = row_index * 3, row_index * 3 + 1, row_index * 3 + 2
            next_left, next_middle, next_right = left + 3, middle + 3, right + 3
            faces.extend(
                (
                    (left, next_left, middle),
                    (next_left, next_middle, middle),
                    (middle, next_middle, right),
                    (right, next_middle, next_right),
                )
            )

        path = f"/World/envs/env_0/raspberry_leaf_{index:02d}"
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr([vertex for face in faces for vertex in face])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(leaf_material)

        # Midrib and four lateral veins sit just in front of the folded leaf.
        vein_points = []
        vein_faces = []

        def add_vein(start, end, thickness):
            direction_x = end[0] - start[0]
            direction_z = end[2] - start[2]
            magnitude = max(math.hypot(direction_x, direction_z), 1.0e-6)
            offset_x = -direction_z / magnitude * thickness
            offset_z = direction_x / magnitude * thickness
            base = len(vein_points)
            vein_points.extend(
                (
                    Gf.Vec3f(start[0] + offset_x, start[1], start[2] + offset_z),
                    Gf.Vec3f(start[0] - offset_x, start[1], start[2] - offset_z),
                    Gf.Vec3f(end[0] - offset_x, end[1], end[2] - offset_z),
                    Gf.Vec3f(end[0] + offset_x, end[1], end[2] + offset_z),
                )
            )
            vein_faces.append((base, base + 1, base + 2, base + 3))

        front_y = center[1] - 0.0072
        start = (
            center[0] - major[0] * length * 0.43,
            front_y,
            center[2] - major[2] * length * 0.43,
        )
        end = (
            center[0] + major[0] * length * 0.45,
            front_y,
            center[2] + major[2] * length * 0.45,
        )
        add_vein(start, end, 0.00055)
        for side in (-1.0, 1.0):
            for u_value in (-0.25, -0.05, 0.16, 0.32):
                profile = math.sin(math.pi * (u_value + 0.5))
                vein_start = (
                    center[0] + major[0] * u_value * length,
                    front_y - 0.0001,
                    center[2] + major[2] * u_value * length,
                )
                vein_end = (
                    vein_start[0] + minor[0] * side * width * profile * 0.34,
                    front_y - 0.0001,
                    vein_start[2] + minor[2] * side * width * profile * 0.34,
                )
                add_vein(vein_start, vein_end, 0.00024)

        veins = UsdGeom.Mesh.Define(stage, f"{path}_veins")
        veins.CreatePointsAttr(vein_points)
        veins.CreateFaceVertexCountsAttr([4] * len(vein_faces))
        veins.CreateFaceVertexIndicesAttr([vertex for face in vein_faces for vertex in face])
        veins.CreateSubdivisionSchemeAttr("none")
        veins.CreateDoubleSidedAttr(True)
        UsdShade.MaterialBindingAPI(veins.GetPrim()).Bind(vein_material)

    def _spawn_background_raspberries(self, stage, Gf, Sdf, UsdGeom, UsdShade) -> None:
        materials = []
        colors = (RASPBERRY_DARK, RASPBERRY_RED, RASPBERRY_HIGHLIGHT)
        for index, color in enumerate(colors):
            materials.append(
                self._make_preview_material(
                    stage,
                    Gf,
                    Sdf,
                    UsdShade,
                    f"/World/envs/env_0/Looks/GroveRaspberry_{index}",
                    color,
                    roughness=0.34 + 0.045 * index,
                )
            )
        calyx_material = self._make_preview_material(
            stage,
            Gf,
            Sdf,
            UsdShade,
            "/World/envs/env_0/Looks/GroveRaspberryCalyx",
            LEAF_COLORS[2],
            roughness=0.76,
        )

        rings = (
            (0.015, 0.006, 5),
            (0.007, 0.012, 8),
            (-0.003, 0.014, 9),
            (-0.013, 0.010, 7),
            (-0.022, 0.004, 3),
        )
        for cluster_index, (center, scale) in enumerate(self.cfg.background_raspberries):
            root = f"/World/envs/env_0/background_raspberry_{cluster_index:02d}"
            UsdGeom.Xform.Define(stage, root)
            drupelet_index = 0
            for ring_index, (height, radius, count) in enumerate(rings):
                phase = 0.42 * (ring_index % 2)
                for item in range(count):
                    angle = 2.0 * math.pi * item / count + phase
                    sphere = UsdGeom.Sphere.Define(
                        stage, f"{root}/drupelet_{drupelet_index:02d}"
                    )
                    sphere.CreateRadiusAttr(0.0055 * scale)
                    UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp().Set(
                        Gf.Vec3d(
                            center[0] + scale * radius * math.cos(angle),
                            center[1] + scale * radius * math.sin(angle),
                            center[2] + scale * height,
                        )
                    )
                    UsdShade.MaterialBindingAPI(sphere.GetPrim()).Bind(
                        materials[(drupelet_index + cluster_index) % len(materials)]
                    )
                    drupelet_index += 1

            calyx_points = []
            calyx_faces = []
            top_z = center[2] + scale * 0.020
            for leaf_index in range(5):
                angle = 2.0 * math.pi * leaf_index / 5.0
                tangent = (-math.sin(angle), math.cos(angle))
                radial = (math.cos(angle), math.sin(angle))
                base = len(calyx_points)
                calyx_points.extend(
                    (
                        Gf.Vec3f(
                            center[0] + scale * 0.0025 * tangent[0],
                            center[1] + scale * 0.0025 * tangent[1],
                            top_z,
                        ),
                        Gf.Vec3f(
                            center[0] + scale * 0.015 * radial[0],
                            center[1] + scale * 0.015 * radial[1],
                            top_z + scale * 0.007,
                        ),
                        Gf.Vec3f(
                            center[0] - scale * 0.0025 * tangent[0],
                            center[1] - scale * 0.0025 * tangent[1],
                            top_z,
                        ),
                    )
                )
                calyx_faces.append((base, base + 1, base + 2))
            calyx = UsdGeom.Mesh.Define(stage, f"{root}/calyx")
            calyx.CreatePointsAttr(calyx_points)
            calyx.CreateFaceVertexCountsAttr([3] * len(calyx_faces))
            calyx.CreateFaceVertexIndicesAttr(
                [vertex for face in calyx_faces for vertex in face]
            )
            calyx.CreateSubdivisionSchemeAttr("none")
            calyx.CreateDoubleSidedAttr(True)
            UsdShade.MaterialBindingAPI(calyx.GetPrim()).Bind(calyx_material)

    def _spawn_target_raspberry_visual(self):
        """Build the target from glossy drupelets and a five-leaf green calyx."""

        try:
            import omni.usd
            from pxr import Gf, Sdf, UsdGeom, UsdShade
        except Exception:
            return
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        root = "/World/envs/env_0/target_raspberry"
        if not stage.GetPrimAtPath(root).IsValid():
            return

        colors = (
            RASPBERRY_HIGHLIGHT,
            (0.84, 0.020, 0.060),
            RASPBERRY_DARK,
        )
        materials = []
        for index, color in enumerate(colors):
            material_path = f"/World/envs/env_0/Looks/RaspberryDrupelet_{index}"
            material = UsdShade.Material.Define(stage, material_path)
            shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*color)
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.34 + 0.045 * index)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            materials.append(material)

        # Rings narrow toward the bottom to create the characteristic hanging,
        # softly conical raspberry silhouette from the reference photograph.
        rings = (
            (0.016, 0.008, 5),
            (0.008, 0.014, 8),
            (-0.002, 0.015, 9),
            (-0.012, 0.012, 7),
            (-0.021, 0.006, 4),
        )
        drupelet_index = 0
        for ring_index, (z_value, ring_radius, count) in enumerate(rings):
            phase = 0.45 * (ring_index % 2)
            for item in range(count):
                angle = 2.0 * math.pi * item / count + phase
                local = Gf.Vec3d(
                    ring_radius * math.cos(angle),
                    ring_radius * math.sin(angle),
                    z_value,
                )
                path = f"{root}/drupelet_{drupelet_index:02d}"
                sphere = UsdGeom.Sphere.Define(stage, path)
                sphere.CreateRadiusAttr(0.0062 if ring_index < 4 else 0.0057)
                UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp().Set(local)
                UsdShade.MaterialBindingAPI(sphere.GetPrim()).Bind(
                    materials[(drupelet_index + ring_index) % len(materials)]
                )
                drupelet_index += 1

        calyx_path = "/World/envs/env_0/Looks/RaspberryCalyx"
        calyx_material = UsdShade.Material.Define(stage, calyx_path)
        calyx_shader = UsdShade.Shader.Define(stage, f"{calyx_path}/PreviewSurface")
        calyx_shader.CreateIdAttr("UsdPreviewSurface")
        calyx_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*LEAF_COLORS[2])
        )
        calyx_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.68)
        calyx_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        calyx_material.CreateSurfaceOutput().ConnectToSource(
            calyx_shader.ConnectableAPI(), "surface"
        )
        for index in range(5):
            angle = 2.0 * math.pi * index / 5.0
            tangent = (-math.sin(angle), math.cos(angle))
            radial = (math.cos(angle), math.sin(angle))
            points = (
                Gf.Vec3f(0.003 * tangent[0], 0.003 * tangent[1], 0.019),
                Gf.Vec3f(0.018 * radial[0], 0.018 * radial[1], 0.026),
                Gf.Vec3f(-0.003 * tangent[0], -0.003 * tangent[1], 0.019),
            )
            mesh = UsdGeom.Mesh.Define(stage, f"{root}/calyx_leaf_{index}")
            mesh.CreatePointsAttr(points)
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)
            UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(calyx_material)

    def _get_rewards(self):
        reward = super()._get_rewards()
        log = self.extras.get("log", {})
        self.extras["log"] = {
            key.replace("GrapePick/", "RaspberryPick/"): value for key, value in log.items()
        }
        return reward
