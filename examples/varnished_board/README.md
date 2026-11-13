# Varnished board package render

Bake the actual `dapple_library/packages/varnished_board.json` package at its
defaults and with its `dark_and_worn` preset, then compare them in Blender:

```sh
cargo run --release -p varnished_board
blender --background --python examples/varnished_board/tools/render.py -- .local/gallery/varnished-board
```

An optional output directory is the Rust executable's first argument; pass
the same directory to Blender. Outputs include `comparison.png`, `grazing.png`,
and a self-contained `varnished-board.blend` scene with packed textures.
Each variant directory holds its `dapple_export` output (`material.gltf`,
which opens by itself in any glTF viewer, its textures, and
`manifest.json`), module diagnostics and material validation report. A
failed material report fails the bake.

Both samples cover the same 80 × 20 cm cut, with identical instance paths,
grain, UVs and lighting. Maps are 1536 × 384 with clamped edges. The grey
backings are display mounts; only their top faces show the package material.

The renderer approximates OpenPBR with Blender's Principled BSDF, building
each material from its manifest with `tools/blender/dapple_export.py`: the
glTF profile's base color, `orm`, normal, clearcoat and clearcoat normal
maps, packed with 16-bit normals (`PackSettings::sixteen_bit_normals`) to
preserve the varnish's orange peel, plus the raw profile's `coat_color.png`
(linear coat tint), which glTF cannot carry, and the IORs the manifest
states. The raw profile's 16-bit `height.png` (its range in meters is in
the manifest) is written too but not rendered: normals carry the surface
relief; geometry is not displaced. OpenPBR's coat darkening has no
Principled equivalent; the manifest lists it with every other loss.

No ADR is needed: this example joins existing package, material and encoder
APIs without changing their ownership or public semantics. It adds no core
dependencies or public APIs.
