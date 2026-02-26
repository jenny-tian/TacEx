# Isaac Sim 5.1 script — per-vertex "strain" -> remap(min,max)-> 1D colormap -> emissiveColor
from pxr import Usd, UsdGeom, UsdShade, Sdf
from PIL import Image
import numpy as np
import os
import omni.usd

# -------- CONFIG ----------
mesh_path = "/World/Cube"  # path to mesh prim with per-vertex primvar "strain"
colormap_filename = "colormap_1x256.png"
strain_min = 0.0
strain_max = 10.0
material_path = Sdf.Path("/World/Materials/colormapMaterial")
# ---------------------------

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage available. Open a stage in Isaac Sim first.")

# Determine output directory for colormap (same folder as stage if possible)
root_path = stage.GetRootLayer().realPath
out_dir = "/home/dh/Projects/Public_TacEx/TacEx"
colormap_path = os.path.join(out_dir, colormap_filename)

# Create 1x256 colormap (HSV sweep vivid)
w = 256
h = 1
arr = np.zeros((h, w, 4), dtype=np.uint8)


def hsv_to_rgb(hv, s=1.0, v=1.0):
    i = int(hv * 6.0) % 6
    f = hv * 6.0 - int(hv * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(r * 255), int(g * 255), int(b * 255)


for x in range(w):
    u = x / (w - 1)
    r, g, b = hsv_to_rgb(u)
    arr[0, x] = (r, g, b, 255)

Image.fromarray(arr, mode="RGBA").save(colormap_path)
print("Wrote colormap to:", colormap_path)

# Create material
mat_prim = stage.DefinePrim(material_path, "Material")
material = UsdShade.Material(mat_prim)

# Texture (UsdUVTexture)
tex_path = material_path.AppendChild("ColormapTexture")
tex = UsdShade.Shader.Define(stage, tex_path)
tex.CreateIdAttr("UsdUVTexture")
# Set file (use relative basename so it resolves next to stage)
tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("/home/dh/Projects/Public_TacEx/TacEx/colormap_1x256.png")
tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
tex.CreateInput("st", Sdf.ValueTypeNames.Float2)  # will be connected

# Primvar reader for "strain"
primvar_path = material_path.AppendChild("PrimvarStrain")
primvar = UsdShade.Shader.Define(stage, primvar_path)
primvar.CreateIdAttr("UsdPrimvarReader_float")
primvar.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("strain")
primvar.CreateOutput("result", Sdf.ValueTypeNames.Float)

# Arithmetic nodes to remap: (strain - min) / (max - min) -> clamp(0,1) -> make float2(u, 0.5)
# Subtract
sub_path = material_path.AppendChild("Sub_Min")
sub = UsdShade.Shader.Define(stage, sub_path)
sub.CreateIdAttr("ND_math_sub")  # placeholder ID; OK to record graph
sub.CreateInput("in1", Sdf.ValueTypeNames.Float)
sub.CreateInput("in2", Sdf.ValueTypeNames.Float).Set(float(strain_min))
sub.CreateOutput("out", Sdf.ValueTypeNames.Float)

# Divide
div_path = material_path.AppendChild("Div_Range")
div = UsdShade.Shader.Define(stage, div_path)
div.CreateIdAttr("ND_math_div")
div.CreateInput("in1", Sdf.ValueTypeNames.Float)
den = max(1e-8, float(strain_max - strain_min))
div.CreateInput("in2", Sdf.ValueTypeNames.Float).Set(den)
div.CreateOutput("out", Sdf.ValueTypeNames.Float)

# Clamp
clamp_path = material_path.AppendChild("Clamp_0_1")
clamp = UsdShade.Shader.Define(stage, clamp_path)
clamp.CreateIdAttr("ND_clamp")
clamp.CreateInput("in", Sdf.ValueTypeNames.Float)
clamp.CreateInput("min", Sdf.ValueTypeNames.Float).Set(0.0)
clamp.CreateInput("max", Sdf.ValueTypeNames.Float).Set(1.0)
clamp.CreateOutput("out", Sdf.ValueTypeNames.Float)

# Compose float2 (u,v)
compose_path = material_path.AppendChild("ComposeUV")
compose = UsdShade.Shader.Define(stage, compose_path)
compose.CreateIdAttr("ND_compose_float2")
compose.CreateInput("u", Sdf.ValueTypeNames.Float)
compose.CreateInput("v", Sdf.ValueTypeNames.Float).Set(0.5)
compose.CreateOutput("out", Sdf.ValueTypeNames.Float2)

# Connect the chain
primvar.GetOutput("result").ConnectToSource(sub.GetInput("in1"))
sub.GetOutput("out").ConnectToSource(div.GetInput("in1"))
div.GetOutput("out").ConnectToSource(clamp.GetInput("in"))
clamp.GetOutput("out").ConnectToSource(compose.GetInput("u"))
compose.GetOutput("out").ConnectToSource(tex.GetInput("st"))

# UsdPreviewSurface
preview_path = material_path.AppendChild("PreviewSurface")
preview = UsdShade.Shader.Define(stage, preview_path)
preview.CreateIdAttr("UsdPreviewSurface")
preview.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)

# Connect tex.rgb -> preview.emissiveColor
tex.CreateOutput("rgb", Sdf.ValueTypeNames.Color3f)
preview.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
tex.GetOutput("rgb").ConnectToSource(preview.GetInput("emissiveColor"))

# Bind preview shader to material surface output
# validate objects and ports
out = material.GetSurfaceOutput()
assert out, "material.GetSurfaceOutput() returned None"
assert material.GetPrim().IsValid(), "material prim invalid"
assert preview and preview.GetPrim().IsValid(), "preview shader prim invalid"

# preview.GetInput("surface").ConnectToSource(material.GetSurfaceOutput())

# material.GetSurfaceOutput().ConnectToSource(preview, "surface")
# material.GetSurfaceOutput().ConnectToSource(preview, "surface", UsdShade.AttributeType.Input)
material.CreateSurfaceOutput().ConnectToSource(preview.GetOutput("surface"))

# Bind material to mesh
mesh_prim = stage.GetPrimAtPath(mesh_path)
if not mesh_prim:
    raise RuntimeError(f"Mesh prim not found at {mesh_path}")
UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)

print("Material created and bound to", mesh_path)
