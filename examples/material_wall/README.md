# material_wall

One image of everything `dapple_library` makes: every package preset and
every library module that is a complete surface, each baked as a 1 m
swatch, gated on `dapple_lab`'s material report, exported through
`dapple_export`, and rendered in Blender as one labeled contact sheet. Run
it after a library change to see the whole library at once.

## What is on the wall

The wall enumerates the library rather than keeping its own list:

- **Packages**: every package in `dapple_library::packages::PACKAGES`, at
  every preset it ships, in the package's order (fresh to weathered). A
  package's family is its name: `varnished_board`, `weathered_copper`,
  `rusted_steel`, `painted_steel`. A package without presets is shown at
  its defaults.
- **Modules**: every module `dapple_library::packages::registry()`
  registers that is a complete surface by itself, at its defaults. The
  rule: **every input optional, and a `material` output.** That keeps the
  bare metals (`copper`, `steel`), the assets (`glazed_brick_wall`,
  `stone_sill`, `threshold`), the masonry, the bark and the bodies
  (`ceramic_body`, `mortar`, `stone`, `wood`), and leaves out the building
  blocks that need an input material (`finish`, `grime`, `streaks`,
  `efflorescence`, `edge_wear`, `moss`, `patina`, `rust`, `paint`) or a
  host resource (`by_example`); they are on the wall inside the assets and
  packages that use them. A module a package re-expresses
  (`dapple_library.X` beside `dapple_library.packages.X`) is shown as the
  package's presets instead.

The registry does not say which family a module belongs to, so
`FAMILIES` in `src/main.rs` is a short explicit table that places the
library's modules in rows (`metal`, `assets`, `masonry`, `bark`,
`bodies`); a module it does not list still appears, in an `other` row.
The run prints what it left out and why, and `wall.json` records it.

Not on the wall: the older field-program materials (`oak`, `brick`,
`gravel`, `parquet` and the `glazed_brick` program), which are functions
that build `dapple_field` programs rather than registered modules; and
variations a module offers beyond its defaults (a birch high on the stem,
an ashlar on a host's bond: see `library_swatches`).

## Bake

```sh
cargo run --release -p material_wall -- [--size <texels>] [--no-cache] [--budget-mib <MiB>] [output-dir]
```

The default output is the git-ignored `.local/gallery/material-wall`. Every
swatch is a 1 m wrapping tile at 512² by default (`--size`): the wall is an
overview, not a close-up. Every swatch is built through one
`InstanceCache` (512 MiB by default; `--no-cache` turns it off). A
package's presets share one top-level instance name, so a step whose
arguments do not change between presets, such as the bare steel under each
painted preset, is built once.

Each swatch is gated on the lab's material report (ranges, invalid values,
seams against the tiling promise, lowering losses, a whole-material
transform). A failing swatch is still exported and marked in the index;
the run fails at the end, once every swatch is baked.

Each swatch is written to `<output-dir>/<family>/<label>/`: the export
(`material.gltf`, its textures, the raw profile's extra textures such as
16-bit `height.png`, and `manifest.json`), `report.json` (the lab report)
and `diagnostics.txt` (every module instance and operation). `wall.json`
indexes them for the render script: per swatch its label, family, preset,
module, directory, physical size, tiling, build time, cache hits and
misses, and verdict; for the run, the grid, total build time, the cache's
counts, and the modules left out. The run prints a summary table.

## Render

With Blender 4.2 or later (Cycles; Metal on Apple GPUs):

```sh
blender --background --python examples/material_wall/tools/render.py -- <output-dir> [samples] [width]
```

writes `<output-dir>/wall.png`, about 3000 × 4300 pixels for the current
library at the default 64 samples (denoised), in under a minute on an
Apple M3 Max. Families keep their swatches together in index order; a
family shares a row with the one before it when both fit in the widest
family's width. Every swatch shows the same 1 m of material, so scales
compare across the wall, on a shallow pillow curved across both axes so
metals mirror a band of the surroundings and relief reads under a low sun
from the upper left. Each material is built from its manifest by the
repository's `tools/blender/dapple_export.py` and displaced by its height.
Under each swatch are its label and family; a failing swatch is labeled
in red. The layout follows `wall.json`, so new packages, presets and
modules need no edit here.

## Test

`cargo test -p material_wall` checks that every package preset is on the
wall and every registered module is either on it or left out with a
reason (so a package added to `PACKAGES` appears automatically), and builds
every swatch on a 32² grid through one cache, the same gate and the same
export (in memory, so it runs on `wasm32-wasip1` too), checking that
presets reuse their shared steps.
