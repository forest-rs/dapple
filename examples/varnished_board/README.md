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
the glTF base, roughness, metalness, normal and clearcoat maps, plus sidecar
coat tint and independent 16-bit coat normal maps to preserve the varnish's
tint and orange peel. Normals carry the surface relief; geometry is not displaced.
Base and coat IOR are 1.5. OpenPBR's coat darkening has no exact Principled
equivalent; the bake records lowering losses explicitly.

No ADR is needed: this example joins existing package, material and encoder
APIs without changing their ownership or public semantics. It adds no core
dependencies or public APIs.
