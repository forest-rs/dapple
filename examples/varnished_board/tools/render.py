# Copyright 2026 the Dapple Authors
# SPDX-License-Identifier: Apache-2.0 OR MIT

"""Render package swatches: blender --background --python render.py -- <gallery-dir>.

The 80 × 20 cm faces share geometry, UVs and lighting. Domain +x follows
the long edge; image rows start at domain y = -0.1. Flipping UV v makes
the glTF normal maps' tangent frame match this image convention. Each
variant's material is built from its `dapple_export` manifest by the
repository's `tools/blender/dapple_export.py`: Principled approximates
OpenPBR, and the coat's tint (which glTF cannot carry), IOR and normals
come from the manifest. The dark backing is a display mount, not
synthesized end grain.
"""

import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 3, "tools", "blender"))
import dapple_export  # noqa: E402

gallery = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
try:
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "METAL"
    scene.cycles.device = "GPU" if any(d.use for d in prefs.devices) else "CPU"
except TypeError:
    scene.cycles.device = "CPU"
scene.cycles.samples = 128
scene.cycles.seed = 0
scene.cycles.use_denoising = True
scene.view_settings.view_transform = "AgX"
scene.view_settings.exposure = -0.5
scene.render.resolution_x = 1800
scene.render.resolution_y = 1200
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"

world = bpy.data.worlds.new("Studio")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.5, 0.5, 0.5, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.15
scene.world = world


def plain(name, color, roughness=0.8):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def package_material(variant):
    return dapple_export.material(os.path.join(gallery, variant), name=variant)


def point_at(obj, point):
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat("-Z", "Y").to_euler()


def light(name, location, energy, size, size_y):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "RECTANGLE"
    data.size = size
    data.size_y = size_y
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.location = location
    point_at(obj, (0, 0, 0))


light("Long softbox", (-0.15, 0.65, 0.9), 16, 1.1, 0.22)
light("Soft fill", (-0.7, -0.25, 0.8), 5.5, 0.7, 0.7)
light("Edge light", (0.6, 0.35, 0.5), 3, 0.5, 0.12)

bpy.ops.mesh.primitive_plane_add(size=200)
bpy.context.object.data.materials.append(plain("Charcoal stage", (0.025, 0.03, 0.035)))
mount = plain("Sample backing", (0.018, 0.021, 0.025))
ink = plain("Lettering", (0.65, 0.69, 0.72))


def label(body, x, y, size):
    data = bpy.data.curves.new(body, "FONT")
    data.body = body
    data.size = size
    obj = bpy.data.objects.new(body, data)
    scene.collection.objects.link(obj)
    obj.location = (x, y, 0.001)
    data.materials.append(ink)


for variant, y, caption in [
    ("default", 0.16, "01   DEFAULT / AMBER VARNISH"),
    ("dark_and_worn", -0.16, "02   DARK AND WORN"),
]:
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, y, 0.014))
    obj = bpy.context.object
    obj.name = variant + " mount"
    obj.dimensions = (0.806, 0.206, 0.024)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mount)
    bevel = obj.modifiers.new("Soft mount edges", "BEVEL")
    bevel.width = 0.002
    bevel.segments = 3
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, y, 0.0261))
    face = bpy.context.object
    face.name = variant + " package face"
    face.scale = (0.8, 0.2, 1)
    for loop in face.data.uv_layers.active.data:
        loop.uv.y = 1 - loop.uv.y
    face.data.materials.append(package_material(variant))
    label(caption, -0.4, y - 0.135, 0.011)

label("DAPPLE / VARNISHED OAK", -0.4, 0.35, 0.025)
label("JSON PACKAGE     /     80 x 20 cm     /     MATCHED GRAIN", -0.4, 0.318, 0.010)

camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
scene.collection.objects.link(camera)
scene.camera = camera


def shoot(name, eye, target, scale):
    camera.location = eye
    point_at(camera, target)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = scale
    scene.render.filepath = os.path.join(gallery, name + ".png")
    bpy.ops.render.render(write_still=True)


shoot("comparison", (0, -0.15, 1.65), (0, 0.025, 0), 1.14)
shoot("grazing", (0.12, -0.9, 0.55), (0, 0.02, 0.025), 1.03)
# Save a self-contained scene for inspection and relighting.
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(gallery, "varnished-board.blend"))
