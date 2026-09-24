# Copyright 2026 the Dapple Authors
# SPDX-License-Identifier: Apache-2.0 OR MIT

"""Renders library_swatches' maps as a labeled swatch sheet in headless Blender.

Usage: blender --background --python render.py -- <swatch-dir>

Reads `<swatch-dir>/swatches.txt` (one swatch name per line) and, per
swatch, `<name>-base.png` (sRGB), `<name>-normal.png` and
`<name>-roughness.png` (non-color), as `cargo run --release -p
library_swatches` writes them, and writes `<swatch-dir>/blender-sheet.png`:
each swatch a 1 m square under a sky and a sun raking at 35 degrees, seen
from straight above, named beneath.

The maps' first row is the domain's top (+y), which is Blender's image
top (v = 1), and the normal map's green is domain +y, so the standard
OpenGL tangent frame of a plane's UVs reads it directly.

Cycles renders on the GPU (Metal) when one is available.
"""

import math
import os
import sys

import bpy
from mathutils import Vector

folder = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "."
with open(os.path.join(folder, "swatches.txt")) as f:
    names = [line.strip() for line in f if line.strip()]

COLUMNS = 5
GAP = 0.25
LABEL = 0.22

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
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
scene.cycles.samples = 64
scene.cycles.use_denoising = True
scene.view_settings.view_transform = "AgX"
scene.view_settings.exposure = -0.3

# A dim sky and a raking sun from the upper left.
world = bpy.data.worlds.new("sky")
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.6, 0.7, 0.9, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35
scene.world = world
sun_data = bpy.data.lights.new("sun", type="SUN")
sun_data.energy = 4.0
sun_data.angle = math.radians(1.0)
sun = bpy.data.objects.new("sun", sun_data)
elevation, azimuth = math.radians(35.0), math.radians(135.0)
toward = Vector(
    (
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    )
)
sun.rotation_euler = toward.to_track_quat("Z", "Y").to_euler()
scene.collection.objects.link(sun)


def image(path, colorspace):
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = colorspace
    return img


def material(name):
    mat = bpy.data.materials.new(name)
    tree = mat.node_tree
    n, links = tree.nodes, tree.links
    bsdf = n["Principled BSDF"]

    def texture(suffix, colorspace):
        t = n.new("ShaderNodeTexImage")
        t.image = image(os.path.join(folder, f"{name}-{suffix}.png"), colorspace)
        t.extension = "REPEAT"
        return t

    links.new(texture("base", "sRGB").outputs["Color"], bsdf.inputs["Base Color"])
    rough = n.new("ShaderNodeSeparateColor")
    links.new(texture("roughness", "Non-Color").outputs["Color"], rough.inputs["Color"])
    links.new(rough.outputs["Red"], bsdf.inputs["Roughness"])
    normal_map = n.new("ShaderNodeNormalMap")
    links.new(texture("normal", "Non-Color").outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


label_mat = bpy.data.materials.new("label")
label_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.9, 0.9, 0.9, 1.0)

rows = (len(names) + COLUMNS - 1) // COLUMNS
step_y = 1.0 + GAP + LABEL
for k, name in enumerate(names):
    col, row = k % COLUMNS, k // COLUMNS
    x = col * (1.0 + GAP)
    y = -row * step_y
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(x, y, 0.0))
    plane = bpy.context.active_object
    plane.name = name
    plane.data.materials.append(material(name))
    bpy.ops.object.text_add(location=(x - 0.5, y - 0.5 - 0.16, 0.0))
    text = bpy.context.active_object
    text.data.body = name.replace("_", " ")
    text.data.size = 0.12
    text.data.materials.append(label_mat)

# A dark ground under everything.
width = COLUMNS * (1.0 + GAP) - GAP
height = rows * step_y
bpy.ops.mesh.primitive_plane_add(size=1.0, location=(width / 2 - 0.5, -height / 2 + 0.5, -0.01))
ground = bpy.context.active_object
ground.scale = (width + 1.0, height + 1.0, 1.0)
ground_mat = bpy.data.materials.new("ground")
ground_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1.0)
ground.data.materials.append(ground_mat)

camera_data = bpy.data.cameras.new("camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = max(width, height) + 0.4
camera = bpy.data.objects.new("camera", camera_data)
camera.location = (width / 2 - 0.5, -height / 2 + 0.5 + LABEL / 2, 10.0)
scene.collection.objects.link(camera)
scene.camera = camera
scene.render.resolution_x = 2000
scene.render.resolution_y = int(2000 * (height + 0.4) / (width + 0.4))
scene.render.filepath = os.path.join(folder, "blender-sheet.png")
bpy.ops.render.render(write_still=True)
print("wrote", scene.render.filepath)
