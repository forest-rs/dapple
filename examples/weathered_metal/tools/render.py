# Copyright 2026 the Dapple Authors
# SPDX-License-Identifier: Apache-2.0 OR MIT

"""Renders weathered_metal's baked presets in headless Blender (Cycles).

Usage:

    blender --background --python render.py -- <family-dir> strip [samples]
    blender --background --python render.py -- <family-dir> hero <preset> [samples]

`<family-dir>` is one family of `cargo run --release -p weathered_metal`'s
output (`<out>/copper`, `<out>/steel` or `<out>/painted`): it lists its
presets in `presets.txt`, fresh to weathered, and holds each preset's
`maps/` as `dapple_export` writes it: the textures and `manifest.json`,
which says what each holds.

- `strip` writes `<family-dir>/strip.png`: every preset side by side, each
  as a flat panel with a cylinder standing in front of it, labeled.
- `hero <preset>` writes `<family-dir>/hero-<preset>.png`: a close view of
  one preset's panel and cylinder at a grazing angle.

Metals show what they reflect, so the scene gives them something to
reflect: a sky beyond the panels, a dark ground for a horizon, and behind
the camera a dark wall with a long softbox on it that reads as a bright
band in the panels, a small cool box low to one side, a tall strip light
to the other that draws a line down each cylinder, and a low key light
that rakes across the panels to show relief. The view is white-balanced on a
grey card, as glazed_brick's renders are.

Mapping: the maps are one 1 m tile; a panel shows 0.7 m × 0.9 m of it,
a cylinder of 1 m circumference wraps it once. Rows start at domain
`y = 0` at the image top (glTF's convention), so `v = 1 − y`.

Each preset's material is built from its manifest by the repository's
`tools/blender/dapple_export.py`, which maps OpenPBR onto Principled BSDF:
`base_color` is the metal's F0, anisotropy turns about the UV tangent by
the axis the manifest gives, and the coat's tint and IOR come from the
manifest. The height displaces the geometry in meters over the range the
manifest gives, so the base normal map, which is derived from the same
height, is left out; the coat keeps its own normal map.

Cycles renders on the GPU (Metal) when one is available.
"""

import math
import os
import sys

import bmesh
import bpy
from mathutils import Vector

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 3, "tools", "blender"))
import dapple_export  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
if len(argv) < 2:
    sys.exit(__doc__)
family_dir = argv[0]
mode = argv[1]
if mode == "hero":
    hero_preset = argv[2]
    rest = argv[3:]
else:
    hero_preset = None
    rest = argv[2:]
samples = int(rest[0]) if rest else 192

with open(os.path.join(family_dir, "presets.txt")) as f:
    presets = [line.strip() for line in f if line.strip()]
if hero_preset is not None:
    if hero_preset not in presets:
        sys.exit(f"no preset {hero_preset!r} in {presets}")

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
scene.cycles.max_bounces = 8
scene.cycles.glossy_bounces = 6
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Medium High Contrast"
scene.view_settings.exposure = 0.0

# --- Light -------------------------------------------------------------

# A sky, kept dim: the softboxes and the key do the work.
SUN_ELEVATION = math.radians(24.0)
SUN_AZIMUTH = math.radians(200.0)  # in front of the panels (−y), to their left
world = bpy.data.worlds.new("sky")
nodes = world.node_tree.nodes
sky = nodes.new("ShaderNodeTexSky")
for sky_type in ("NISHITA", "MULTIPLE_SCATTERING", "HOSEK_WILKIE"):
    try:
        sky.sky_type = sky_type
        break
    except TypeError:
        continue
if hasattr(sky, "sun_elevation"):
    sky.sun_elevation = SUN_ELEVATION
    sky.sun_rotation = SUN_AZIMUTH + math.pi / 2.0
    if hasattr(sky, "sun_disc"):
        sky.sun_disc = False
background = nodes["Background"]
background.inputs["Strength"].default_value = 0.12
world.node_tree.links.new(sky.outputs["Color"], background.inputs["Color"])
scene.world = world

# The key: a low sun raking across the panels from the left.
sun_data = bpy.data.lights.new("key", type="SUN")
sun_data.energy = 3.0
sun_data.angle = math.radians(1.5)
sun_data.color = (1.0, 0.93, 0.84)
sun = bpy.data.objects.new("key", sun_data)
toward_sun = Vector(
    (
        math.cos(SUN_ELEVATION) * math.cos(SUN_AZIMUTH),
        math.cos(SUN_ELEVATION) * math.sin(SUN_AZIMUTH),
        math.sin(SUN_ELEVATION),
    )
)
sun.rotation_euler = toward_sun.to_track_quat("Z", "Y").to_euler()
scene.collection.objects.link(sun)


def area_light(name, location, target, size, energy, color=(1.0, 1.0, 1.0)):
    """A rectangular area light facing `target`, visible in reflections."""
    data = bpy.data.lights.new(name, type="AREA")
    data.shape = "RECTANGLE"
    data.size, data.size_y = size
    data.energy = energy
    data.color = color
    obj = bpy.data.objects.new(name, data)
    obj.location = location
    obj.rotation_euler = (Vector(target) - Vector(location)).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(obj)
    return obj


def ground():
    bpy.ops.mesh.primitive_plane_add(size=200.0, location=(0.0, 0.0, 0.0))
    obj = bpy.context.active_object
    mat = bpy.data.materials.new("ground")
    tree = mat.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.05, 0.048, 0.045, 1.0)
    ramp.color_ramp.elements[1].color = (0.11, 0.105, 0.1, 1.0)
    tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.85
    obj.data.materials.append(mat)


ground()


def backdrop(distance, width):
    """A dark wall behind the camera, `distance` from the panels: what the
    metal mirrors between the lights. It casts no shadow."""
    bpy.ops.mesh.primitive_plane_add(
        size=1.0, location=(0.0, -distance, 3.0), rotation=(math.pi / 2.0, 0.0, 0.0)
    )
    obj = bpy.context.active_object
    obj.scale = (width, 6.0, 1.0)
    obj.visible_shadow = False
    mat = bpy.data.materials.new("backdrop")
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.03, 0.03, 0.032, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.9
    obj.data.materials.append(mat)


def studio(distance, width):
    """The lights the metal mirrors: a long softbox above the camera on the
    backdrop, a smaller cool one low and well to the left, where the
    cylinders catch it but the panels do not mirror it, and a tall strip to
    the right that draws a line down each cylinder."""
    backdrop(distance + 1.5, width + 12.0)
    area_light("softbox", (0.0, -distance - 1.3, 1.2), (0.0, 0.0, 0.5), (width + 1.0, 0.5), 28.0)
    area_light(
        "low box", (-width / 2.0 - 2.0, -1.2, 0.35), (0.0, 0.0, 0.4), (1.0, 1.0), 10.0,
        (0.88, 0.94, 1.0),
    )
    area_light("strip", (width / 2.0 + 1.6, -distance * 0.45, 1.0), (0.0, 0.0, 0.4), (0.25, 2.5), 22.0)

# --- Materials -----------------------------------------------------------


def material(preset):
    """The preset's material, built from its export's manifest, displaced
    by its height."""
    return dapple_export.material(
        os.path.join(family_dir, preset, "maps"),
        name=preset,
        displacement=True,
        normal_maps=("geometry_coat_normal",),
    )


# --- Geometry ------------------------------------------------------------

PANEL = (0.7, 0.9)  # meters of the tile a panel shows
PANEL_UV = (0.1, 0.05)  # where in the tile it starts (domain x, y)
RADIUS = 1.0 / (2.0 * math.pi)  # one tile around
CYLINDER_HEIGHT = 0.55
SPACING = 0.95  # between presets along x


def panel(name, x, mat, density):
    """A panel facing −y, `PANEL` in size, its foot 3 cm up."""
    w, h = PANEL
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=int(w * density),
        y_subdivisions=int(h * density),
        size=1.0,
        calc_uvs=True,
        location=(x, 0.0, 0.03 + h / 2.0),
        rotation=(math.pi / 2.0, 0.0, 0.0),
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (w, h, 1.0)
    uvs = obj.data.uv_layers.active
    uvs.name = "UVMap"
    for loop in uvs.data:
        u, v = loop.uv
        # Domain x across, domain y up the panel; v = 1 − y.
        loop.uv = (PANEL_UV[0] + u * w, 1.0 - (PANEL_UV[1] + v * h))
    obj.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return obj


def cylinder(name, location, mat, density):
    """A cylinder of one tile's circumference, its axis vertical, with a
    seam column of its own so its UVs run from 0 to 1 around."""
    around = int(1.0 * density)
    up = int(CYLINDER_HEIGHT * density)
    verts, faces, uv = [], [], []
    for j in range(up + 1):
        z = CYLINDER_HEIGHT * j / up
        for i in range(around + 1):
            # u around from the side facing the camera (−y).
            a = 2.0 * math.pi * i / around
            verts.append((RADIUS * math.sin(a), -RADIUS * math.cos(a), z))
    row = around + 1
    for j in range(up):
        for i in range(around):
            faces.append((j * row + i, j * row + i + 1, (j + 1) * row + i + 1, (j + 1) * row + i))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    uvs = mesh.uv_layers.new(name="UVMap")
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            vi = mesh.loops[li].vertex_index
            i, j = vi % row, vi // row
            # Domain y up the side, from 0.1 m; v = 1 − y.
            uvs.data[li].uv = (i / around, 1.0 - (0.1 + CYLINDER_HEIGHT * j / up))
    mesh.shade_smooth()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = (location[0], location[1], 0.0)
    scene.collection.objects.link(obj)
    # The seam column shares positions with the first: weld it, so it
    # displaces without a crack; UVs are per corner and survive.
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bm.to_mesh(mesh)
    bm.free()
    mesh.shade_smooth()
    obj.data.materials.append(mat)
    # A cap, plain metal-grey, so the top does not read as a hole.
    bpy.ops.mesh.primitive_circle_add(
        vertices=128,
        radius=RADIUS + 0.002,
        fill_type="NGON",
        location=(location[0], location[1], CYLINDER_HEIGHT + 0.004),
    )
    cap = bpy.context.active_object
    cap_mat = bpy.data.materials.get("cap") or bpy.data.materials.new("cap")
    cap_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (
        0.06,
        0.06,
        0.06,
        1.0,
    )
    cap_mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.6
    cap.data.materials.append(cap_mat)
    return obj


def label(text, location, size=0.045):
    curve = bpy.data.curves.new(f"label {text}", type="FONT")
    curve.body = text.replace("_", " ")
    curve.size = size
    curve.align_x = "CENTER"
    obj = bpy.data.objects.new(f"label {text}", curve)
    obj.location = location
    obj.rotation_euler = (math.radians(70.0), 0.0, 0.0)
    mat = bpy.data.materials.get("label") or bpy.data.materials.new("label")
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.55, 0.55, 0.52, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.7
    curve.materials.append(mat)
    scene.collection.objects.link(obj)


camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
scene.collection.objects.link(camera)
scene.camera = camera


def aim(eye, target, lens):
    camera.location = eye
    camera.rotation_euler = (Vector(target) - Vector(eye)).to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = lens


def white_balance(at):
    """Balances the view on a grey card facing the camera, as a
    photographer would: the sky otherwise tints every neutral."""
    bpy.ops.mesh.primitive_plane_add(
        size=0.3, location=at, rotation=(math.pi / 2.0, 0.0, 0.0)
    )
    card = bpy.context.active_object
    mat = bpy.data.materials.new("grey card")
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.18, 0.18, 0.18, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Specular IOR Level"].default_value = 0.0
    card.data.materials.append(mat)
    settings = (scene.render.resolution_x, scene.render.resolution_y, scene.cycles.samples)
    format_ = scene.render.image_settings.file_format
    scene.render.resolution_x = scene.render.resolution_y = 32
    scene.cycles.samples = 64
    scene.render.image_settings.file_format = "OPEN_EXR"
    aim((at[0], at[1] - 1.2, at[2]), at, 200.0)
    path = os.path.join(family_dir, "white-balance-probe.exr")
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    probe = bpy.data.images.load(path)
    px = list(probe.pixels)
    count = len(px) // 4
    mean = [sum(px[c::4]) / count for c in range(3)]
    bpy.data.images.remove(probe)
    os.remove(path)
    bpy.data.objects.remove(card)
    scene.render.resolution_x, scene.render.resolution_y, scene.cycles.samples = settings
    scene.render.image_settings.file_format = format_
    scene.view_settings.use_white_balance = True
    scene.view_settings.white_balance_whitepoint = [c / mean[1] for c in mean]
    print("white point:", mean)


def shoot(name):
    scene.render.filepath = os.path.join(family_dir, name)
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)


if mode == "strip":
    shown = presets
    xs = [(i - (len(shown) - 1) / 2.0) * SPACING for i in range(len(shown))]
    for preset, x in zip(shown, xs):
        mat = material(preset)
        panel(f"panel {preset}", x, mat, 700)
        cylinder(f"cylinder {preset}", (x + 0.2, -0.3), mat, 900)
        label(preset, (x - 0.12, -0.52, 0.004))
    width = xs[-1] - xs[0] + SPACING
    lens = 50.0
    distance = 0.55 * width * 1.02 / math.tan(math.atan(18.0 / lens))
    studio(distance, width)
    scene.render.resolution_x = 2400
    scene.render.resolution_y = 860
    white_balance((0.0, -0.6, 0.5))
    aim((0.0, -distance, 0.62), (0.0, 0.0, 0.42), lens)
    shoot("strip.png")
else:
    mat = material(hero_preset)
    panel(f"panel {hero_preset}", 0.0, mat, 1600)
    cylinder(f"cylinder {hero_preset}", (0.3, -0.36), mat, 1600)
    studio(2.5, 1.5)
    scene.render.resolution_x = 2000
    scene.render.resolution_y = 1250
    white_balance((0.0, -0.6, 0.5))
    # From the left, looking along the panel at a grazing angle with the
    # cylinder in front of it, so the key rakes across the relief.
    aim((-0.72, -0.66, 0.66), (0.14, -0.06, 0.4), 40.0)
    shoot(f"hero-{hero_preset}.png")
