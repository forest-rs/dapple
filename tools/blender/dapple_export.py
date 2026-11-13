# Copyright 2026 the Dapple Authors
# SPDX-License-Identifier: Apache-2.0 OR MIT

"""Builds Blender materials from the directories `dapple_export` writes.

Import it from a render script:

    sys.path.insert(0, "<repository>/tools/blender")
    import dapple_export
    material = dapple_export.material("<export-dir>")

`material(directory)` reads `manifest.json` and wires a Principled BSDF from
its account of every OpenPBR parameter: each texture is loaded with the
color space the manifest gives, each channel decoded over its range, and
each constant set, so nothing here knows file names, color spaces or IORs.
Parameters the manifest says glTF cannot carry (a coat tint, the height)
come from the export's extra textures or constants, as any other; a lost map
uses the mean the manifest gives for it. What Principled has no input for
(occlusion, which path tracing computes itself; coat darkening; subsurface
color, which Principled takes from the base color) is listed, not guessed.

The mapping onto Principled is approximate where the models differ:
`specular_weight` w is "Specular IOR Level" w / 2 (0.5 leaves the
reflectance alone); `emission_luminance` in cd/m² becomes "Emission
Strength" through 683 lm/W; glTF's anisotropy strength and fuzz as sheen
are passed through. The anisotropy direction is not wired: Blender
stretches along the UV tangent.

Run directly, it checks Blender's own glTF importer instead:

    blender --background --python dapple_export.py -- <export-dir> [--render <out.png>]

which imports `material.gltf`, prints how the importer wired each
Principled input, and with `--render` renders the imported quad facing a
camera under a sun and sky.
"""

import json
import math
import os
import sys

import bpy

# OpenPBR parameter -> Principled BSDF input, for parameters that map one to
# one (after the decoding below).
SOCKETS = {
    "base_color": "Base Color",
    "base_metalness": "Metallic",
    "specular_roughness": "Roughness",
    "specular_ior": "IOR",
    "specular_color": "Specular Tint",
    "specular_roughness_anisotropy": "Anisotropic",
    "transmission_weight": "Transmission Weight",
    "subsurface_weight": "Subsurface Weight",
    "coat_weight": "Coat Weight",
    "coat_roughness": "Coat Roughness",
    "coat_ior": "Coat IOR",
    "coat_color": "Coat Tint",
    "fuzz_weight": "Sheen Weight",
    "fuzz_color": "Sheen Tint",
    "fuzz_roughness": "Sheen Roughness",
    "emission_color": "Emission Color",
    "geometry_opacity": "Alpha",
    "base_diffuse_roughness": "Diffuse Roughness",
}

# Luminous efficacy at 555 nm: candela per square meter per W/(sr m²).
LUMENS_PER_WATT = 683.0


def manifest(directory):
    """The manifest `dapple_export` wrote into `directory`, checked."""
    with open(os.path.join(directory, "manifest.json")) as f:
        m = json.load(f)
    if m.get("format") != "dapple-material" or m.get("version") != 1:
        raise ValueError(f"{directory}: not a version 1 dapple manifest")
    return m


class _Builder:
    def __init__(self, directory, name, uv_map, extension):
        self.directory = directory
        self.manifest = manifest(directory)
        self.uv_map = uv_map
        self.mat = bpy.data.materials.new(name or self.manifest["provenance"]["subject"])
        self.nodes = self.mat.node_tree.nodes
        self.links = self.mat.node_tree.links
        self.bsdf = self.nodes["Principled BSDF"]
        tiling = self.manifest["provenance"]["tiling"]
        self.extension = extension or ("REPEAT" if tiling["x"] or tiling["y"] else "EXTEND")
        self.uv = self.nodes.new("ShaderNodeUVMap")
        self.uv.uv_map = uv_map
        self.images = {}
        self.parameters = {p["name"]: p for p in self.manifest["parameters"]}
        self.unused = []

    def image(self, texture):
        """The image node of the texture named `texture`, loaded once."""
        if texture in self.images:
            return self.images[texture]
        entry = next(t for t in self.manifest["textures"] if t["name"] == texture)
        img = bpy.data.images.load(os.path.join(self.directory, entry["file"]))
        img.colorspace_settings.name = "sRGB" if entry["color_space"] == "srgb" else "Non-Color"
        # Channels hold independent data: never premultiply them by alpha.
        img.alpha_mode = "CHANNEL_PACKED"
        node = self.nodes.new("ShaderNodeTexImage")
        node.image = img
        node.extension = self.extension
        self.links.new(self.uv.outputs["UV"], node.inputs["Vector"])
        self.images[texture] = node
        return node

    def channels(self, location):
        """An output socket holding the location's channels, decoded to
        `range` (a scalar for one channel, a color for three)."""
        node = self.image(location["texture"])
        start, end = location["channels"]["start"], location["channels"]["end"]
        if end - start == 3 and start == 0:
            out = node.outputs["Color"]
        elif end - start == 1 and start == 3:
            out = node.outputs["Alpha"]
        elif end - start == 1:
            split = self.nodes.new("ShaderNodeSeparateColor")
            self.links.new(node.outputs["Color"], split.inputs["Color"])
            out = split.outputs[("Red", "Green", "Blue")[start]]
        else:
            raise ValueError(f"cannot read channels {start}..{end} of {location['texture']}")
        lo, hi = location["range"]
        if (lo, hi) != (0.0, 1.0):
            if end - start == 3:
                scale = self.nodes.new("ShaderNodeVectorMath")
                scale.operation = "MULTIPLY_ADD"
                self.links.new(out, scale.inputs[0])
                scale.inputs[1].default_value = (hi - lo,) * 3
                scale.inputs[2].default_value = (lo,) * 3
                out = scale.outputs["Vector"]
            else:
                scale = self.nodes.new("ShaderNodeMath")
                scale.operation = "MULTIPLY_ADD"
                self.links.new(out, scale.inputs[0])
                scale.inputs[1].default_value = hi - lo
                scale.inputs[2].default_value = lo
                out = scale.outputs["Value"]
        return out

    def value(self, name):
        """The parameter's constant (or default, or a lost map's mean), or
        None when a texture holds it."""
        held = self.parameters[name]["held"]
        if held["kind"] == "texture":
            return None
        return held["value"] if held["kind"] != "lost" else held["mean"]

    def set(self, socket, value):
        """Sets a Principled input to a scalar or color."""
        target = self.bsdf.inputs[socket]
        if len(value) == 3:
            target.default_value = (*value, 1.0)
        else:
            target.default_value = value[0]

    def scalar(self, socket, name, transform):
        """Wires a scalar parameter through `transform` (a constant
        function and a node operation)."""
        fn, op, k = transform
        held = self.parameters[name]["held"]
        if held["kind"] == "texture":
            out = self.channels(held)
            node = self.nodes.new("ShaderNodeMath")
            node.operation = op
            self.links.new(out, node.inputs[0])
            node.inputs[1].default_value = k
            self.links.new(node.outputs["Value"], self.bsdf.inputs[socket])
        else:
            self.bsdf.inputs[socket].default_value = fn(self.value(name)[0])

    def normal(self, name, socket):
        held = self.parameters[name]["held"]
        if held["kind"] != "texture":
            return
        # The Normal Map node decodes [0, 1] to [-1, 1] itself.
        color = self.channels({**held, "range": [0.0, 1.0]})
        if held["encoding"] == "normal_y_down":
            # Flip green into Blender's (OpenGL's) +Y-up tangent frame.
            split = self.nodes.new("ShaderNodeSeparateColor")
            self.links.new(color, split.inputs["Color"])
            flip = self.nodes.new("ShaderNodeMath")
            flip.operation = "SUBTRACT"
            flip.inputs[0].default_value = 1.0
            self.links.new(split.outputs["Green"], flip.inputs[1])
            join = self.nodes.new("ShaderNodeCombineColor")
            self.links.new(split.outputs["Red"], join.inputs["Red"])
            self.links.new(flip.outputs["Value"], join.inputs["Green"])
            self.links.new(split.outputs["Blue"], join.inputs["Blue"])
            color = join.outputs["Color"]
        node = self.nodes.new("ShaderNodeNormalMap")
        node.uv_map = self.uv_map
        self.links.new(color, node.inputs["Color"])
        self.links.new(node.outputs["Normal"], self.bsdf.inputs[socket])

    def build(self, displacement, normal_maps):
        products = {}
        for name, socket in SOCKETS.items():
            p = self.parameters.get(name)
            if p is None or socket not in self.bsdf.inputs:
                continue
            held = p["held"]
            if held["kind"] == "texture" and held["encoding"] == "product":
                # Channels shared with another parameter hold the product:
                # wire them to the color and the weight's input to 1.
                products.setdefault((held["texture"], held["channels"]["start"]), []).append(p)
                continue
            if name == "specular_weight":
                continue
            if held["kind"] == "texture":
                self.links.new(self.channels(held), self.bsdf.inputs[socket])
            else:
                self.set(socket, self.value(name))
            if held["kind"] == "lost":
                print(f"{name}: lost; using its mean {held['mean']}")
        for group in products.values():
            color = next(p for p in group if p["name"].endswith("_color"))
            scale = math.prod(p["held"]["range"][1] for p in group)
            out = self.channels({**color["held"], "range": [0.0, scale]})
            self.links.new(out, self.bsdf.inputs[SOCKETS[color["name"]]])
            for p in group:
                if p is not color:
                    self.bsdf.inputs[SOCKETS[p["name"]]].default_value = 1.0
        # Emission: luminance in cd/m² over the emission color.
        lum = self.parameters["emission_luminance"]["held"]
        if lum["kind"] == "texture" and lum["encoding"] == "product":
            # The color's channels hold the luminance over its largest value.
            strength = lum["range"][1] / LUMENS_PER_WATT
            self.bsdf.inputs["Emission Strength"].default_value = strength
        else:
            self.scalar(
                "Emission Strength",
                "emission_luminance",
                (lambda v: v / LUMENS_PER_WATT, "MULTIPLY", 1.0 / LUMENS_PER_WATT),
            )
        self.scalar(
            "Specular IOR Level",
            "specular_weight",
            (lambda v: v * 0.5, "MULTIPLY", 0.5),
        )
        if normal_maps:
            self.normal("geometry_normal", "Normal")
            self.normal("geometry_coat_normal", "Coat Normal")
        height = self.parameters["height"]["held"]
        if displacement and height["kind"] == "texture":
            # Height in meters: lo + t (hi - lo) = (t - midlevel) scale.
            lo, hi = height["range"]
            node = self.nodes.new("ShaderNodeDisplacement")
            node.inputs["Scale"].default_value = hi - lo
            node.inputs["Midlevel"].default_value = -lo / (hi - lo) if hi > lo else 0.0
            image = self.image(height["texture"])
            image.interpolation = "Cubic"
            self.links.new(image.outputs["Color"], node.inputs["Height"])
            self.links.new(
                node.outputs["Displacement"], self.nodes["Material Output"].inputs["Displacement"]
            )
            self.mat.displacement_method = "BOTH"
        if self.parameters["geometry_thin_walled"]["held"]["value"] != [0.0]:
            self.mat.use_backface_culling = False
        for name in ("occlusion", "coat_darkening", "subsurface_color", "geometry_tangent"):
            if self.parameters[name]["held"]["kind"] != "default":
                self.unused.append(name)
        if self.unused:
            print(f"{self.mat.name}: Principled has no input for {', '.join(self.unused)}")
        return self.mat



def material(
    directory, name=None, uv_map="UVMap", extension=None, displacement=False, normal_maps=True
):
    """A Principled BSDF material built from the export in `directory`.

    `extension` is the image nodes' extension mode; by default `REPEAT`
    when the material tiles along either axis and `EXTEND` otherwise.
    `displacement` wires the height, when the export holds one as a
    texture, into true displacement in meters; `normal_maps=False` leaves
    the base and coat normal maps out, for geometry that is displaced by
    the same relief.
    """
    return _Builder(directory, name, uv_map, extension).build(displacement, normal_maps)


def _check_import(directory, render):
    """Imports `material.gltf` with Blender's glTF importer and prints how
    each Principled input is wired; renders the quad with `render`."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    path = os.path.join(directory, "material.gltf")
    result = bpy.ops.import_scene.gltf(filepath=path)
    print("import:", result, "blender", bpy.app.version_string)
    for mat in bpy.data.materials:
        tree = mat.node_tree
        bsdf = next(n for n in tree.nodes if n.type == "BSDF_PRINCIPLED")
        print(f"material {mat.name}:")
        for socket in bsdf.inputs:
            if socket.is_linked:
                link = socket.links[0]
                print(f"  {socket.name} <- {link.from_node.name}.{link.from_socket.name}")
            elif socket.name in SOCKETS.values() or socket.name in (
                "Specular IOR Level",
                "Emission Strength",
            ):
                value = socket.default_value
                if hasattr(value, "__len__"):
                    value = tuple(round(v, 4) for v in value)
                print(f"  {socket.name} = {value}")
    if not render:
        return
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.view_settings.view_transform = "AgX"
    scene.render.resolution_x = scene.render.resolution_y = 1024
    world = bpy.data.worlds.new("sky")
    scene.world = world
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.6, 0.65, 0.75, 1.0)
    background.inputs["Strength"].default_value = 0.4
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(60.0), 0.0, math.radians(-30.0))
    scene.collection.objects.link(sun)
    size = manifest(directory)["provenance"]["grid"]["size"]
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = max(size) * 1.05
    # glTF's +Z (the quad's facing) is Blender's -Y.
    camera.location = (0.0, -3.0, 0.0)
    camera.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    scene.collection.objects.link(camera)
    scene.camera = camera
    scene.render.filepath = render
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if not args:
        sys.exit("usage: blender --background --python dapple_export.py -- <dir> [--render <png>]")
    out = args[args.index("--render") + 1] if "--render" in args else None
    _check_import(os.path.abspath(args[0]), out and os.path.abspath(out))
