# Copyright 2026 the Dapple Authors
# SPDX-License-Identifier: Apache-2.0 OR MIT

"""Renders material_wall's swatches as one labeled contact sheet in
headless Blender (Cycles).

Usage:

    blender --background --python render.py -- <wall-dir> [samples] [width]

`<wall-dir>` is `cargo run --release -p material_wall`'s output: `wall.json`
lists every swatch (its family, label, directory, physical size, tiling and
lab verdict) in wall order, and each swatch's directory holds its export as
`dapple_export` writes it: the textures and `manifest.json`, which says
what each holds. The sheet is written to `<wall-dir>/wall.png`; `samples`
defaults to 64 (denoised) and `width` to 3000 pixels, the height following
from the layout.

Layout: families in the index's order, each a run of swatches in its own
order (a package's presets from fresh to weathered). The widest family
sets the sheet's width; a family starts a new row unless it fits beside
the one before it, a wider gap apart. Every swatch shows the same physical extent, `SHOWN` meters
square, so scales compare across the wall: a material whose tile is
smaller repeats, one whose tile is larger shows a part of it. Under each
swatch, its label and family; a swatch whose lab report failed is labeled
in red. A caption under the wall gives the count, the extent and the bake
resolution.

Each swatch is a shallow pillow, curved across both axes, so that metals
mirror a band of the surroundings instead of one point and relief reads
under the low light; its UVs follow arc length along both axes, so the
texture is not stretched. Rows start at domain `y = 0` at the image top
(glTF's convention), so `v = 1 - y`, and domain `+y` is up the wall.

Light: the surroundings are a procedural sky behind the camera, dark below
the horizon and bright above it, with two softboxes in it (a broad warm one
above the horizon, a narrow cool one low to the right), so a mirror-like
metal shows a horizon line, the softboxes and dark ground; a low sun from
the upper left rakes across the wall to show relief; the swatches hang on a
near-black board. The camera is orthographic, square on to the wall. The
view transform is Khronos PBR Neutral, which keeps base colors' hue and
saturation where AgX bleaches bright metals toward white.

Each material is built from its manifest by the repository's
`tools/blender/dapple_export.py`, displaced by its height where the export
holds one (the base normal map, derived from the same height, is left out;
a coat keeps its own normal map).
"""

import json
import math
import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 3, "tools", "blender"))
import dapple_export  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
if not argv:
    sys.exit(__doc__)
wall_dir = os.path.abspath(argv[0])
samples = int(argv[1]) if len(argv) > 1 else 64
width_px = int(argv[2]) if len(argv) > 2 else 3000

with open(os.path.join(wall_dir, "wall.json")) as f:
    index = json.load(f)
if index.get("format") != "dapple-material-wall" or index.get("version") != 1:
    sys.exit(f"{wall_dir}: not a version 1 material wall index")

# --- Layout --------------------------------------------------------------

SHOWN = 1.0  # meters of material every swatch shows, across and up
GAP = 0.2  # between swatches of a family
GROUP_GAP = 0.6  # between families sharing a row
LABEL = 0.34  # below each swatch, for its two label lines
ROW_GAP = 0.12  # between a row's labels and the next row's swatches
MARGIN = 0.3  # around the wall
CAPTION = 0.16  # below the last row, for the sheet's caption
RX, RZ = 2.6, 4.0  # the pillow's radii of curvature across and up
DENSITY = 256  # mesh vertices per meter, before the bump on top

families = []
rows = {}
for swatch in index["swatches"]:
    if swatch["family"] not in rows:
        families.append(swatch["family"])
        rows[swatch["family"]] = []
    rows[swatch["family"]].append(swatch)


def span(n):
    """The width of `n` swatches side by side."""
    return n * SHOWN + (n - 1) * GAP


# The widest family sets the wall's width; families that fit beside the
# one before them share its row, `GROUP_GAP` apart, in index order.
capacity = max(span(len(r)) for r in rows.values())
lines = [[]]
used = 0.0
for family in families:
    w = span(len(rows[family]))
    start = used + GROUP_GAP if lines[-1] else 0.0
    if lines[-1] and start + w > capacity + 1e-6:
        lines.append([])
        start = 0.0
    lines[-1].append((family, start))
    used = start + w
wall_w = capacity + 2.0 * MARGIN
row_h = SHOWN + LABEL + ROW_GAP
wall_h = len(lines) * row_h - ROW_GAP + 2.0 * MARGIN + CAPTION

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# Cycles on the GPU where there is one.
scene.render.engine = "CYCLES"
cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
try:
    cycles_prefs.compute_device_type = "METAL"
    cycles_prefs.get_devices()
    for device in cycles_prefs.devices:
        device.use = True
    scene.cycles.device = "GPU" if any(d.type == "METAL" for d in cycles_prefs.devices) else "CPU"
except TypeError:
    scene.cycles.device = "CPU"
print("cycles device:", scene.cycles.device)
scene.cycles.samples = samples
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 6
scene.cycles.glossy_bounces = 4
# Khronos PBR Neutral: a swatch should show the material's own color.
scene.view_settings.view_transform = "Khronos PBR Neutral"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0

# --- Light -------------------------------------------------------------


def world():
    """The surroundings the metals mirror: by direction, dark ground,
    a bright horizon fading to a blue zenith, and two softboxes."""
    w = bpy.data.worlds.new("surroundings")
    tree = w.node_tree
    nodes, links = tree.nodes, tree.links
    coords = nodes.new("ShaderNodeTexCoord")
    xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(coords.outputs["Generated"], xyz.inputs["Vector"])
    # Sky by elevation (z of the direction).
    ramp = nodes.new("ShaderNodeValToRGB")
    elevation = nodes.new("ShaderNodeMapRange")
    elevation.inputs["From Min"].default_value = -1.0
    elevation.inputs["From Max"].default_value = 1.0
    links.new(xyz.outputs["Z"], elevation.inputs["Value"])
    links.new(elevation.outputs["Result"], ramp.inputs["Fac"])
    stops = [
        (0.0, (0.012, 0.011, 0.010)),
        (0.40, (0.035, 0.032, 0.029)),
        (0.43, (0.22, 0.22, 0.21)),
        (0.52, (0.17, 0.19, 0.22)),
        (1.0, (0.10, 0.14, 0.22)),
    ]
    elements = ramp.color_ramp.elements
    for e, (position, color) in zip(elements, (stops[0], stops[-1])):
        e.position, e.color = position, (*color, 1.0)
    for position, color in stops[1:-1]:
        elements.new(position).color = (*color, 1.0)
    color = ramp.outputs["Color"]

    def window(axis_value, lo, hi, soft):
        """A soft 0..1 mask of `lo < value < hi`."""
        a = nodes.new("ShaderNodeMapRange")
        a.interpolation_type = "SMOOTHSTEP"
        a.inputs["From Min"].default_value = lo - soft
        a.inputs["From Max"].default_value = lo + soft
        links.new(axis_value, a.inputs["Value"])
        b = nodes.new("ShaderNodeMapRange")
        b.interpolation_type = "SMOOTHSTEP"
        b.inputs["From Min"].default_value = hi + soft
        b.inputs["From Max"].default_value = hi - soft
        links.new(axis_value, b.inputs["Value"])
        m = nodes.new("ShaderNodeMath")
        m.operation = "MULTIPLY"
        links.new(a.outputs["Result"], m.inputs[0])
        links.new(b.outputs["Result"], m.inputs[1])
        return m.outputs["Value"]

    def softbox(x_range, z_range, behind, radiance, tint, base):
        """Adds a softbox seen in directions with x and z in the ranges and
        y below `behind` (behind the camera, which looks along +y)."""
        mask = nodes.new("ShaderNodeMath")
        mask.operation = "MULTIPLY"
        links.new(window(xyz.outputs["X"], *x_range, 0.03), mask.inputs[0])
        links.new(window(xyz.outputs["Z"], *z_range, 0.02), mask.inputs[1])
        back = nodes.new("ShaderNodeMath")
        back.operation = "LESS_THAN"
        links.new(xyz.outputs["Y"], back.inputs[0])
        back.inputs[1].default_value = behind
        both = nodes.new("ShaderNodeMath")
        both.operation = "MULTIPLY"
        links.new(mask.outputs["Value"], both.inputs[0])
        links.new(back.outputs["Value"], both.inputs[1])
        mix = nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "ADD"
        links.new(both.outputs["Value"], mix.inputs["Factor"])
        links.new(base, mix.inputs["A"])
        mix.inputs["B"].default_value = tuple(radiance * c for c in tint) + (1.0,)
        return mix.outputs["Result"]

    # A broad warm softbox above the horizon, a little left of the camera.
    color = softbox((-0.45, 0.2), (0.06, 0.32), -0.3, 0.7, (1.0, 0.95, 0.88), color)
    # A narrow cool strip low and to the right.
    color = softbox((0.42, 0.55), (-0.02, 0.22), -0.3, 0.8, (0.85, 0.92, 1.0), color)
    background = nodes["Background"]
    background.inputs["Strength"].default_value = 1.0
    links.new(color, background.inputs["Color"])
    scene.world = w


world()

# The key: a low sun from the upper left, raking across the wall.
sun_data = bpy.data.lights.new("key", type="SUN")
sun_data.energy = 3.2
sun_data.angle = math.radians(2.0)
sun_data.color = (1.0, 0.94, 0.86)
sun = bpy.data.objects.new("key", sun_data)
toward_sun = Vector((-0.50, -0.40, 0.77)).normalized()
sun.rotation_euler = toward_sun.to_track_quat("Z", "Y").to_euler()
scene.collection.objects.link(sun)

# --- Geometry ------------------------------------------------------------


def board():
    """The near-black board the swatches hang on, just behind them."""
    bpy.ops.mesh.primitive_plane_add(
        size=1.0, location=(wall_w / 2.0, 0.03, wall_h / 2.0), rotation=(math.pi / 2.0, 0.0, 0.0)
    )
    obj = bpy.context.active_object
    obj.scale = (wall_w + 4.0, wall_h + 4.0, 1.0)
    mat = bpy.data.materials.new("board")
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.018, 0.018, 0.019, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.85
    obj.data.materials.append(mat)


def pillow(name, size):
    """A swatch's mesh: `SHOWN` meters of arc across and up, bowed toward
    the camera (−y), its UVs in the texture's units for a material whose
    tile is `size` meters."""
    n = max(2, int(SHOWN * DENSITY))
    half = SHOWN / 2.0
    edge_x = RX * math.cos(half / RX)
    edge_z = RZ * math.cos(half / RZ)
    verts = []
    for j in range(n + 1):
        t = SHOWN * j / n - half
        for i in range(n + 1):
            s = SHOWN * i / n - half
            depth = (RX * math.cos(s / RX) - edge_x) + (RZ * math.cos(t / RZ) - edge_z)
            verts.append((RX * math.sin(s / RX), -depth, RZ * math.sin(t / RZ)))
    row = n + 1
    faces = [
        (j * row + i, j * row + i + 1, (j + 1) * row + i + 1, (j + 1) * row + i)
        for j in range(n)
        for i in range(n)
    ]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    uvs = mesh.uv_layers.new(name="UVMap")
    flat = []
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            vi = mesh.loops[li].vertex_index
            i, j = vi % row, vi // row
            # Domain x across, domain y up the wall; v = 1 - y.
            flat += (SHOWN * i / n / size[0], 1.0 - SHOWN * j / n / size[1])
    uvs.data.foreach_set("uv", flat)
    mesh.shade_smooth()
    return mesh


def label(lines, location):
    """Centered label lines, top down, as (text, size, color) under a
    swatch: emissive, so they read the same under any light."""
    y = location[2]
    for text, size, color in lines:
        curve = bpy.data.curves.new(f"label {text}", type="FONT")
        curve.body = text
        curve.size = size
        curve.align_x = "CENTER"
        curve.align_y = "TOP"
        obj = bpy.data.objects.new(f"label {text}", curve)
        obj.location = (location[0], location[1], y)
        obj.rotation_euler = (math.pi / 2.0, 0.0, 0.0)
        obj.visible_shadow = False
        key = "label %.3f %.3f %.3f" % color
        mat = bpy.data.materials.get(key)
        if mat is None:
            mat = bpy.data.materials.new(key)
            tree = mat.node_tree
            tree.nodes.remove(tree.nodes["Principled BSDF"])
            emission = tree.nodes.new("ShaderNodeEmission")
            emission.inputs["Color"].default_value = (*color, 1.0)
            emission.inputs["Strength"].default_value = 1.0
            tree.links.new(emission.outputs["Emission"], tree.nodes["Material Output"].inputs["Surface"])
        curve.materials.append(mat)
        scene.collection.objects.link(obj)
        y -= size * 1.25


meshes = {}


def place(swatch, x, z0):
    """Hangs `swatch` centered on `x` with its foot at `z0`, labeled
    beneath."""
    size = tuple(swatch["size"])
    if size not in meshes:
        meshes[size] = pillow(f"pillow {size}", size)
    mesh = meshes[size].copy()
    obj = bpy.data.objects.new(f"swatch {swatch['directory']}", mesh)
    obj.location = (x, 0.0, z0 + SHOWN / 2.0)
    scene.collection.objects.link(obj)
    mat = dapple_export.material(
        os.path.join(wall_dir, swatch["directory"]),
        name=swatch["directory"],
        displacement=True,
        normal_maps=("geometry_coat_normal",),
    )
    mesh.materials.append(mat)
    passed = swatch["passed"]
    title = swatch["label"].replace("_", " ")
    if not passed:
        title += "  FAILED"
    label(
        [
            (title, 0.085, (0.85, 0.85, 0.82) if passed else (0.95, 0.12, 0.08)),
            (swatch["family"].replace("_", " "), 0.06, (0.42, 0.42, 0.40)),
        ],
        (x, -0.02, z0 - 0.06),
    )


board()
for r, line in enumerate(lines):
    # Rows from the top of the wall down.
    z0 = wall_h - MARGIN - r * row_h - SHOWN
    for family, start in line:
        for c, swatch in enumerate(rows[family]):
            place(swatch, MARGIN + start + c * (SHOWN + GAP) + SHOWN / 2.0, z0)

failed = sum(1 for s in index["swatches"] if not s["passed"])
label(
    [
        (
            f"dapple_library: {len(index['swatches'])} swatches, each {SHOWN:g} m square, "
            f"baked at {index['texels']}\u00b2 per tile"
            + (f"; {failed} failed the lab report" if failed else "; every lab report passes"),
            0.07,
            (0.5, 0.5, 0.48),
        )
    ],
    (wall_w / 2.0, -0.02, MARGIN + CAPTION - 0.02),
)

# --- Camera --------------------------------------------------------------

camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
camera.data.type = "ORTHO"
camera.data.ortho_scale = max(wall_w, wall_h)
camera.data.clip_end = 100.0
camera.location = (wall_w / 2.0, -20.0, wall_h / 2.0)
camera.rotation_euler = (math.pi / 2.0, 0.0, 0.0)
scene.collection.objects.link(camera)
scene.camera = camera
scene.render.resolution_x = width_px
scene.render.resolution_y = round(width_px * wall_h / wall_w)
scene.render.resolution_percentage = 100

scene.render.filepath = os.path.join(wall_dir, "wall.png")
scene.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True)
print("wrote", scene.render.filepath)
