# weathered_metal

Bakes `dapple_library`'s metals at every preset of their packages, gates
each bake on `dapple_lab`'s material report, exports the result for glTF
through `dapple_export`, and renders the presets in Blender.

| Family | Package | Presets, fresh to weathered |
|---|---|---|
| `copper` | `weathered_copper.json`: `copper` → `patina` | `bright`, `tarnished`, `streaked`, `verdigris` |
| `steel` | `rusted_steel.json`: `steel` → `rust` | `fresh`, `flash_rusted`, `rusting`, `pitted` |
| `painted` | `painted_steel.json`: `steel` → `paint` → `rust` | `new`, `chipped`, `weeping`, `derelict` |

## Bake

```sh
cargo run --release -p weathered_metal -- [copper|steel|painted|all] [output-dir]
```

The default output is the git-ignored `.local/gallery/weathered-metal`. Each
preset is a 1 m wrapping tile at 2048² in `<output-dir>/<family>/<preset>/`:

- `maps/`: the export: `material.gltf` (the material on a 1 m quad); the
  glTF profile's `base_color.png`, 16-bit `normal.png`, `orm.png`,
  `anisotropy.png` (brushed metal), `clearcoat.png` and
  `clearcoat_normal.png` (paint); the raw profile's `coat_color.png` and
  16-bit `height.png` for what glTF cannot carry; and `manifest.json`,
  which says what every parameter is, which channels hold it and how to
  decode them (the height's range in meters, the anisotropy axis's
  frame), what is a constant (the coat's IOR) and what glTF loses.
- previews: `base-color.png`, `roughness.png`, `metalness.png`,
  `height.png`, `surface.png` (metal grey, verdigris green, rust orange,
  paint blue) and `raking.png`.
- `report.json`: the lab's material report, whose verdict the manifest
  records too (ranges, seams against the
  tiling promise, lowering losses, a whole-material transform) and the
  metals' relationships: where corrosion or paint wins the surface,
  metalness is at most a half; corrosion is rougher than the bare metal;
  bright copper keeps copper's calibrated F0. The run fails if any check
  does.
- `diagnostics.txt`: every module instance and every operation's
  approximation report.

Per family, `progression.json` checks the presets against each other
(roughness rises with weathering, bare metals lose metalness),
`timings.txt` records each bake, and `presets.txt` lists the presets in
order.

## Render

With Blender 4.2 or later (Cycles; Metal on Apple GPUs):

```sh
blender --background --python examples/weathered_metal/tools/render.py -- <output-dir>/copper strip
blender --background --python examples/weathered_metal/tools/render.py -- <output-dir>/copper hero streaked
```

`strip` writes `<family>/strip.png`, every preset as a displaced panel with a
cylinder in front of it; `hero <preset>` writes `<family>/hero-<preset>.png`,
a close view at a grazing angle. An optional last argument sets the sample
count (default 192). The scene is procedural: a sky, a dark backdrop with a
softbox behind the camera, a strip light, and a low key light; no assets are
downloaded. Each material is built from its manifest by the repository's
`tools/blender/dapple_export.py`, displaced by its height.

`cargo test -p weathered_metal` runs the same lab gate on each family at
its defaults on a 12² grid.
