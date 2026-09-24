# Copyright 2026 the Dapple Authors
# SPDX-License-Identifier: Apache-2.0 OR MIT

"""Renders glazed_brick's packed maps as a short wall in headless Blender.

Usage: blender --background --python render.py -- <gallery-dir>

Reads `<gallery-dir>/maps/` (base_color.png, orm.png, normal.png,
height.png and height.txt, as `cargo run --release -p glazed_brick` writes
them) and writes into <gallery-dir>:

- `blender-front.png`: the wall face on, under sky and a low sun;
- `blender-grazing.png`: along the wall at a grazing angle, where the
  glaze's gloss and the sky's reflection show;
- `blender-close.png`: a close, oblique view of a few bricks;
- `blender-grazing-normal-map.png`: the grazing view on a flat plane shaded
  by `normal.png` alone, to check the packed normal map against the
  displaced geometry.

The wall is 2 m wide (two repeats of the 1 m tile) and 1 m tall, displaced
by the height map in meters. The maps' rows start at domain `y = 0` at the
image top (glTF's convention), so the wall's UVs run `v = 1` at its foot to
`v = 0` at its top: domain `+y` is up the wall, and the tangent frame's
bitangent points down it, as the glTF normal map's green channel expects.

Cycles renders on the GPU (Metal) when one is available.
"""

import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

gallery = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "."
maps = os.path.join(gallery, "maps")
with open(os.path.join(maps, "height.txt")) as f:
    height_lo, height_hi = (float(v) for v in f.read().split())

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
print("cycles device:", scene.cycles.device, [d.name for d in cycles_prefs.devices if d.use])
scene.cycles.samples = 256
scene.cycles.use_denoising = True
scene.view_settings.view_transform = "AgX"
scene.render.resolution_x = 1600
scene.render.resolution_y = 1000

# Sky and a sun from the same direction: low, from the upper left, so it
# rakes across the bricks.
SUN_ELEVATION = math.radians(28.0)
SUN_AZIMUTH = math.radians(215.0)  # from +x toward +y: in front of the wall, to its left
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
background.inputs["Strength"].default_value = 0.35
world.node_tree.links.new(sky.outputs["Color"], background.inputs["Color"])
scene.world = world

sun_data = bpy.data.lights.new("sun", type="SUN")
sun_data.energy = 4.0
sun_data.angle = math.radians(1.0)
sun = bpy.data.objects.new("sun", sun_data)
# A sun lamp shines along its local -Z.
toward_sun = Vector(
    (
        math.cos(SUN_ELEVATION) * math.cos(SUN_AZIMUTH),
        math.cos(SUN_ELEVATION) * math.sin(SUN_AZIMUTH),
        math.sin(SUN_ELEVATION),
    )
)
sun.rotation_euler = toward_sun.to_track_quat("Z", "Y").to_euler()
scene.collection.objects.link(sun)

# Ground.
bpy.ops.mesh.primitive_plane_add(size=200.0, location=(0.0, 0.0, 0.0))
ground = bpy.context.active_object
grey = bpy.data.materials.new("ground")
grey.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.2, 0.19, 0.17, 1.0)
grey.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
ground.data.materials.append(grey)


def image(name, colorspace):
    img = bpy.data.images.load(os.path.join(maps, name))
    img.colorspace_settings.name = colorspace
    return img


base_color = image("base_color.png", "sRGB")
orm = image("orm.png", "Non-Color")
normal = image("normal.png", "Non-Color")
height = image("height.png", "Non-Color")


def material(name, displaced):
    mat = bpy.data.materials.new(name)
    tree = mat.node_tree
    n, links = tree.nodes, tree.links
    bsdf = n["Principled BSDF"]
    bsdf.inputs["IOR"].default_value = 1.5

    uv = n.new("ShaderNodeUVMap")

    def texture(img, interpolation="Linear"):
        t = n.new("ShaderNodeTexImage")
        t.image = img
        t.extension = "REPEAT"
        t.interpolation = interpolation
        links.new(uv.outputs["UV"], t.inputs["Vector"])
        return t

    links.new(texture(base_color).outputs["Color"], bsdf.inputs["Base Color"])
    split = n.new("ShaderNodeSeparateColor")
    links.new(texture(orm).outputs["Color"], split.inputs["Color"])
    links.new(split.outputs["Green"], bsdf.inputs["Roughness"])
    links.new(split.outputs["Blue"], bsdf.inputs["Metallic"])
    output = n["Material Output"]
    if displaced:
        # Height in meters: lo + h · (hi − lo) = (h − midlevel) · scale.
        scale = height_hi - height_lo
        displacement = n.new("ShaderNodeDisplacement")
        displacement.inputs["Scale"].default_value = scale
        displacement.inputs["Midlevel"].default_value = -height_lo / scale
        links.new(texture(height, "Cubic").outputs["Color"], displacement.inputs["Height"])
        links.new(displacement.outputs["Displacement"], output.inputs["Displacement"])
        mat.displacement_method = "BOTH"
    else:
        normal_map = n.new("ShaderNodeNormalMap")
        normal_map.uv_map = "UVMap"
        links.new(texture(normal).outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def wall(name, subdivisions, mat):
    """A 2 m × 1 m wall facing −y, its foot 5 cm above the ground."""
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=2 * subdivisions,
        y_subdivisions=subdivisions,
        size=1.0,
        calc_uvs=True,
        location=(0.0, 0.0, 0.55),
        rotation=(math.pi / 2.0, 0.0, 0.0),
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (2.0, 1.0, 1.0)
    uvs = obj.data.uv_layers.active
    uvs.name = "UVMap"
    co = np.empty(2 * len(uvs.data), dtype=np.float32)
    uvs.data.foreach_get("uv", co)
    co = co.reshape(-1, 2)
    co[:, 0] *= 2.0  # two repeats across
    co[:, 1] = 1.0 - co[:, 1]  # domain y = 0 (the image top) at the foot
    uvs.data.foreach_set("uv", co.ravel())
    obj.data.materials.append(mat)
    return obj


camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
scene.collection.objects.link(camera)
scene.camera = camera


def shoot(name, eye, target, lens):
    camera.location = eye
    direction = Vector(target) - Vector(eye)
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = lens
    scene.render.filepath = os.path.join(gallery, name)
    bpy.ops.render.render(write_still=True)


displaced = wall("wall", 1000, material("glazed_brick", True))
shoot("blender-front.png", (0.0, -3.1, 0.62), (0.0, 0.0, 0.55), 50.0)
shoot("blender-grazing.png", (-1.25, -0.42, 0.72), (0.55, 0.0, 0.52), 40.0)
shoot("blender-close.png", (0.12, -0.36, 0.66), (0.34, 0.0, 0.58), 50.0)

# The packed normal map alone, on a flat plane.
bpy.data.objects.remove(displaced)
wall("flat", 1, material("glazed_brick_normal_map", False))
shoot("blender-grazing-normal-map.png", (-1.25, -0.42, 0.72), (0.55, 0.0, 0.52), 40.0)
