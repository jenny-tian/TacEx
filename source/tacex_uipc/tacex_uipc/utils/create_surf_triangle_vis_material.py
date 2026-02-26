# Tip: Node id names can differe depending on your application.
# To find out what node id's you have, use this:
# from pxr import Sdr
# reg = Sdr.Registry()
# print("\n".join(sorted(reg.GetNodeNames())))

# -> Run this script inside the Isaac Sim script editor

from pxr import UsdGeom, UsdShade, Gf, Sdf
import omni.usd
from isaaclab.sim.utils import bind_visual_material

# Config
OUTLINE_COLOR = Gf.Vec3f(0.8, 0.8, 0.8)
OUTLINE_WIDTH = 0.05
BASE_COLOR = Gf.Vec3f(0.0, 0.0, 0.8)  # underlying fill color


# Helper: create barycentric primvar on mesh (vertex interpolation)
def add_barycentric_primvar(mesh_prim: UsdGeom.Mesh, primvar_name="baryCoord"):
    """
    Adds a vec3f primvar per-vertex storing barycentric coordinates for each triangle.
    Works for triangle-only topologies.
    """
    # Get indices and points
    points_attr = mesh_prim.GetPointsAttr()
    points = points_attr.Get()
    if points is None:
        print("Mesh has no points; cannot add barycentric primvar.")
        return None

    # Get face vertex counts and indices
    fv_counts = mesh_prim.GetFaceVertexCountsAttr().Get()
    fv_indices = mesh_prim.GetFaceVertexIndicesAttr().Get()
    if fv_counts is None or fv_indices is None:
        print("Mesh missing face vertex counts/indices.")
        return None

    # Ensure triangles only
    if any(c != 3 for c in fv_counts):
        print("Mesh contains non-triangle faces; barycentric generation expects triangles.")
        # We can triangulate externally; here we bail.
        return None

    # Build per-vertex barycentric list length = num points
    # For meshes with shared vertices across triangles we must create per-vertex-per-face mapping (non-indexed)
    # USD supports "primvar interpolation='fvar' or 'vertex'". To avoid complications, we'll create a primvar with
    # interpolation='vertex' and set a value for each vertex index. However if a vertex is shared between triangles
    # its barycentric value must be consistent; that would not show triangle outlines correctly. Therefore the robust method
    # is to split vertices so each triangle has unique vertices. But modifying topology is intrusive.
    # Alternative: store barycentrics as "faceVarying" primvar where value per face-vertex is supported.
    primvarsAPI = UsdGeom.PrimvarsAPI(mesh_prim)
    primvar = primvarsAPI.CreatePrimvar(primvar_name, Sdf.ValueTypeNames.Float3Array, UsdGeom.Tokens.faceVarying)
    # Build face-varying array: one vec3 per face-vertex entry
    bary_vals = []
    idx_iter = iter(fv_indices)
    for i, count in enumerate(fv_counts):
        # should be 3
        # For triangle: assign (1,0,0),(0,1,0),(0,0,1)
        bary_vals.append(Gf.Vec3f(1.0, 0.0, 0.0))
        bary_vals.append(Gf.Vec3f(0.0, 1.0, 0.0))
        bary_vals.append(Gf.Vec3f(0.0, 0.0, 1.0))
        # advance iterator by 3
        for _ in range(count):
            next(idx_iter, None)
    primvar.Set(bary_vals)
    print(f"Added face-varying barycentric primvar '{primvar_name}' to {mesh_prim.GetPath()}")

    # print("MESH Debug")
    # mesh = mesh_prim.GetPrim()
    # print(mesh.GetAttribute("primvars:baryCoord").Get())
    # print(mesh.GetAttribute("faceVertexCounts").Get())
    # print(mesh.GetAttribute("faceVertexIndices").Get())

    return primvar


def create_triangle_outline_material(
    mat_path="/Materials/TriangleOutlineMat",
    primvar_name="baryCoord",
    outline_color=OUTLINE_COLOR,
    outline_width=OUTLINE_WIDTH,
    base_color=BASE_COLOR,
):
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage available. Open a stage in Isaac Sim first.")

    # from pxr import Sdr

    # reg = Sdr.Registry()

    # def has_node(id_):
    #     return id_ in reg.GetNodeNames()

    # print(
    #     "Shader node availability:",
    #     "\n ND_UsdPrimvarReader_vector3",
    #     has_node("ND_UsdPrimvarReader_vector3"),
    #     "\n separate3",
    #     has_node("ND_extract_vector3"),
    #     "\n min",
    #     has_node("ND_min_float"),
    #     "\n smooth",
    #     has_node("ND_smoothstep_float"),
    #     "\n sub",
    #     has_node("ND_subtract_float"),
    #     "\n make3",
    #     has_node("ND_combine3_vector3"),  # ND_make_float3
    # )

    material = UsdShade.Material.Define(stage, mat_path)

    # Read values of the primvar
    primvar_reader = UsdShade.Shader.Define(stage, mat_path + "/PrimvarReader_float3")
    primvar_reader.CreateIdAttr("ND_UsdPrimvarReader_vector3")
    primvar_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(primvar_name)

    # Separate components of the primvar
    outx = UsdShade.Shader.Define(stage, mat_path + "/PrimvarX")
    outx.CreateIdAttr("ND_extract_vector3")
    outx.CreateInput("in", Sdf.ValueTypeNames.Float3).ConnectToSource(primvar_reader.ConnectableAPI(), "result")
    outx.CreateInput("index", Sdf.ValueTypeNames.Int).Set(0)

    outy = UsdShade.Shader.Define(stage, mat_path + "/PrimvarY")
    outy.CreateIdAttr("ND_extract_vector3")
    outy.CreateInput("in", Sdf.ValueTypeNames.Float3).ConnectToSource(primvar_reader.ConnectableAPI(), "result")
    outy.CreateInput("index", Sdf.ValueTypeNames.Int).Set(1)

    outz = UsdShade.Shader.Define(stage, mat_path + "/PrimvarZ")
    outz.CreateIdAttr("ND_extract_vector3")
    outz.CreateInput("in", Sdf.ValueTypeNames.Float3).ConnectToSource(primvar_reader.ConnectableAPI(), "result")
    outz.CreateInput("index", Sdf.ValueTypeNames.Int).Set(2)

    # min(x,y)
    min_xy = UsdShade.Shader.Define(stage, mat_path + "/MinXY")
    min_xy.CreateIdAttr("ND_min_float")
    min_xy.CreateInput("in1", Sdf.ValueTypeNames.Float).ConnectToSource(outx.ConnectableAPI(), "out")
    min_xy.CreateInput("in2", Sdf.ValueTypeNames.Float).ConnectToSource(outy.ConnectableAPI(), "out")
    # min_xy.CreateOutput("out", Sdf.ValueTypeNames.Float)

    # min(min_xy, z)
    min_xyz = UsdShade.Shader.Define(stage, mat_path + "/MinXYZ")
    min_xyz.CreateIdAttr("ND_min_float")
    min_xyz.CreateInput("in1", Sdf.ValueTypeNames.Float).ConnectToSource(min_xy.ConnectableAPI(), "out")
    min_xyz.CreateInput("in2", Sdf.ValueTypeNames.Float).ConnectToSource(outz.ConnectableAPI(), "out")
    # min_xyz.CreateOutput("out", Sdf.ValueTypeNames.Float)

    # smoothstep(0, outline_width, min_xyz)
    smooth = UsdShade.Shader.Define(stage, mat_path + "/Smoothstep")
    smooth.CreateIdAttr("ND_smoothstep_float")
    smooth.CreateInput("low", Sdf.ValueTypeNames.Float).Set(0.0)
    smooth.CreateInput("high", Sdf.ValueTypeNames.Float).Set(outline_width)
    smooth.CreateInput("in", Sdf.ValueTypeNames.Float).ConnectToSource(min_xyz.ConnectableAPI(), "out")
    # smooth.CreateOutput("out", Sdf.ValueTypeNames.Float)

    # edge mask = 1 - smooth
    one_minus = UsdShade.Shader.Define(stage, mat_path + "/OneMinus")
    one_minus.CreateIdAttr("ND_subtract_float")
    one_minus.CreateInput("in1", Sdf.ValueTypeNames.Float).Set(1.0)
    one_minus.CreateInput("in2", Sdf.ValueTypeNames.Float).ConnectToSource(smooth.ConnectableAPI(), "out")
    # one_minus.CreateOutput("out", Sdf.ValueTypeNames.Float)

    # Make outline color (float3)
    outline_col_node = UsdShade.Shader.Define(stage, mat_path + "/OutlineColor")
    outline_col_node.CreateIdAttr("ND_combine3_color3")
    outline_col_node.CreateInput("in1", Sdf.ValueTypeNames.Float).Set(float(outline_color[0]))
    outline_col_node.CreateInput("in2", Sdf.ValueTypeNames.Float).Set(float(outline_color[1]))
    outline_col_node.CreateInput("in3", Sdf.ValueTypeNames.Float).Set(float(outline_color[2]))
    # outline_col_node.CreateOutput("out", Sdf.ValueTypeNames.Float3)

    # Make base color (float3)
    base_col_node = UsdShade.Shader.Define(stage, mat_path + "/BaseColor")
    base_col_node.CreateIdAttr("ND_combine3_color3")
    base_col_node.CreateInput("in1", Sdf.ValueTypeNames.Float).Set(float(base_color[0]))
    base_col_node.CreateInput("in2", Sdf.ValueTypeNames.Float).Set(float(base_color[1]))
    base_col_node.CreateInput("in3", Sdf.ValueTypeNames.Float).Set(float(base_color[2]))
    # base_col_node.CreateOutput("out", Sdf.ValueTypeNames.Float3)

    # Mix colors: mix(outline, base, smooth) where smooth is 0 at edges -> use one_minus as factor for outline
    mix = UsdShade.Shader.Define(stage, mat_path + "/MixColor")
    mix.CreateIdAttr("ND_mix_color3")
    mix.CreateInput("fg", Sdf.ValueTypeNames.Float3).ConnectToSource(outline_col_node.ConnectableAPI(), "out")
    mix.CreateInput("bg", Sdf.ValueTypeNames.Float3).ConnectToSource(base_col_node.ConnectableAPI(), "out")
    mix.CreateInput("mix", Sdf.ValueTypeNames.Float).ConnectToSource(one_minus.ConnectableAPI(), "out")
    # mix.CreateOutput("out", Sdf.ValueTypeNames.Float3)

    # # Preview surface
    # pbr = UsdShade.Shader.Define(stage, mat_path + "/PreviewSurface")
    # pbr.CreateIdAttr("ND_UsdPreviewSurface_surfaceshader")
    # # connect with color from Mix node
    # pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(mix.ConnectableAPI(), "out")
    # pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    # pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    # PBR surface - preview surface won't work with standalone IsaacLab script
    pbr = UsdShade.Shader.Define(stage, mat_path + "/SurfaceShader")
    pbr.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
    # connect with color from Mix node
    pbr.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(mix.ConnectableAPI(), "out")
    pbr.CreateInput("base_diffuse_roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    pbr.CreateInput("base_metalness", Sdf.ValueTypeNames.Float).Set(0.0)

    # Bind shading outputs
    material.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "out")
    return material


def create_cube(mesh_path="/World/CubeSurface"):
    # --- Cube vertices (unit cube) ---
    vertices = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
        (0.0, 0.0, 1.0),  # 4
        (1.0, 0.0, 1.0),  # 5
        (1.0, 1.0, 1.0),  # 6
        (0.0, 1.0, 1.0),  # 7
    ]

    # --- Tetrahedralization of the cube (5 tets) ---
    tets = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6), (1, 4, 5, 6), (3, 4, 6, 7)]

    # Helper: build outer surface triangles by collecting all faces and removing internal ones
    from collections import Counter

    # Collect face occurrences
    face_counts = Counter()
    face_map = {}  # map sorted face -> original winding (we'll pick one)
    for tet in tets:
        # original oriented faces for visualization (consistent winding)
        a, b, c, d = tet
        oriented_faces = [(a, b, c), (a, d, b), (b, d, c), (c, d, a)]
        for of in oriented_faces:
            key = tuple(sorted(of))
            face_counts[key] += 1
            # store the oriented version (last wins) for the key
            face_map[key] = of

    # Keep only boundary faces (count == 1)
    boundary_faces = [face_map[k] for k, v in face_counts.items() if v == 1]

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage available. Open a stage in Isaac Sim first.")

    # Create mesh prim for boundary surface
    mesh_path = Sdf.Path(mesh_path)
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)

    # Points
    points = [Gf.Vec3f(x, y, z) for (x, y, z) in vertices]
    mesh.GetPointsAttr().Set(points)

    # Build face topology (triangles)
    faceVertexCounts = [3] * len(boundary_faces)
    faceVertexIndices = []
    # We'll create a primvar "edgeFactor" per-face-vertex that encodes distance-to-edge mask.
    # For simplicity, we set equal barycentric coordinates for vertices, but to get crisp lines
    # we encode a per-vertex attribute that the shader can use to detect edges: set:
    #   for each triangle vertex, set value 1.0 at the vertex and 0.0 at the opposite edge midpoint — the shader will use barycentric interpolation.
    # A compact approach: provide a "edgeAnchor" vec3 per face-vertex that is (1,0,0), (0,1,0), (0,0,1).
    # The interpolated value near an edge will have a small component; shader computes min(component) to get distance to nearest edge.
    edge_anchor_vals = []

    for tri in boundary_faces:
        for vi in tri:
            faceVertexIndices.append(int(vi))
        # append anchors for the 3 face-vertices
        edge_anchor_vals.extend([Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0), Gf.Vec3f(0.0, 0.0, 1.0)])

    mesh.GetFaceVertexCountsAttr().Set(faceVertexCounts)
    mesh.GetFaceVertexIndicesAttr().Set(faceVertexIndices)
    return mesh


# Apply to a selected mesh or example mesh
def assign_material_to_mesh_with_usd(gprim: UsdGeom.Mesh, material=None):
    # Create material if not provided
    if material is None:
        material = create_triangle_outline_material()

    # Create material binding
    # UsdShade.MaterialBindingAPI(gprim).Bind(material)
    bind_visual_material(gprim.GetPath(), material.GetPath(), omni.usd.get_context().get_stage())

    print(f"Assigned material to {gprim.GetPath()}")


if __name__ == "__main__":
    # Example usage:

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage available. Open a stage in Isaac Sim first.")

    mesh_path = "/World/MeshExample"
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    if not mesh_prim.IsValid():
        mesh = create_cube(mesh_path=mesh_path)
    else:
        mesh = UsdGeom.Mesh(mesh_prim)
    # Ensure that gmesh has barycentric primvar
    primvar = add_barycentric_primvar(mesh, primvar_name="baryCoord")
    if primvar is None:
        print("Primvar creation failed.")

    mat = create_triangle_outline_material(
        mat_path="/World/Materials/TriangleOutlineMat",
        primvar_name="baryCoord",
        outline_color=OUTLINE_COLOR,
        outline_width=OUTLINE_WIDTH,
        base_color=BASE_COLOR,
    )
    assign_material_to_mesh_with_usd(mesh, material=mat)
