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
Each variant directory holds its baked maps, module diagnostics, material
validation report, and lowering report. A failed material report fails the bake.

Both samples cover the same 80 × 20 cm cut, with identical instance paths,
grain, UVs and lighting. Maps are 1536 × 384 with clamped edges. The grey
backings are display mounts; only their top faces show the package material.

The renderer approximates OpenPBR with Blender's Principled BSDF. It uses
the glTF profile's base color, `orm`, normal, clearcoat and clearcoat normal
maps, packed with 16-bit normals (`PackSettings::sixteen_bit_normals`) to
preserve the varnish's orange peel, plus the raw profile's `coat_color.png`
(linear coat tint), which glTF cannot carry. The raw profile's 16-bit
`height.png`, over the range `height.txt` gives in meters, is written too but
not rendered: normals carry the surface relief; geometry is not displaced.
Base and coat IOR are 1.5. OpenPBR's coat darkening has no exact Principled
equivalent; the bake records lowering losses explicitly.

No ADR is needed: this example joins existing package, material and encoder
APIs without changing their ownership or public semantics. It adds no core
dependencies or public APIs.
