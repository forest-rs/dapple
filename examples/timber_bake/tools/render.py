# Copyright 2026 the Dapple Authors
# SPDX-License-Identifier: Apache-2.0 OR MIT

"""Renders timber_bake's `timbers.glb` with headless Blender.

Usage: blender --background --python render.py -- <output-dir>

Writes `timbers.png` (the whole group), `timbers-beam-end.png` (the beam's
end grain) and `timbers-rafter-end.png` (the rafter's plumb cut) into
<output-dir>.
"""

import math
import os
import sys

import bpy
from mathutils import Vector

out = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "."

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=os.path.join(out, "timbers.glb"))
scene = bpy.context.scene

# Ground plane, a neutral grey.
bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0.0, 0.0, 0.0))
ground = bpy.context.active_object
grey = bpy.data.materials.new("ground")
grey.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.32, 0.32, 0.3, 1.0)
ground.data.materials.append(grey)

world = bpy.data.worlds.new("sky")
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.6, 0.68, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6
scene.world = world

sun_data = bpy.data.lights.new("sun", type="SUN")
sun_data.energy = 3.5
sun_data.angle = math.radians(3.0)
sun = bpy.data.objects.new("sun", sun_data)
sun.rotation_euler = (math.radians(50.0), math.radians(10.0), math.radians(35.0))
scene.collection.objects.link(sun)

camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
scene.collection.objects.link(camera)
scene.camera = camera

scene.render.engine = "CYCLES"
scene.cycles.samples = 64
scene.cycles.use_denoising = True
scene.view_settings.view_transform = "AgX"
scene.render.resolution_x = 1400
scene.render.resolution_y = 1000


def shoot(name, eye, target, lens):
    camera.location = eye
    direction = Vector(target) - Vector(eye)
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = lens
    scene.render.filepath = os.path.join(out, name)
    bpy.ops.render.render(write_still=True)


# Exedra is z-up; the glTF importer brings the y-up file back to z-up.
shoot("timbers.png", (1.9, -2.6, 1.6), (-0.25, -0.15, 0.62), 36.0)
shoot("timbers-beam-end.png", (1.4, -0.5, 1.55), (0.8, 0.0, 1.3), 50.0)
shoot("timbers-rafter-end.png", (0.25, -1.0, 0.95), (-0.39, -0.38, 0.66), 50.0)
