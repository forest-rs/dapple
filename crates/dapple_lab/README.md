# dapple_lab

A headless material laboratory for [dapple](../../README.md).

- `report`: named measurements (with units and bounds) and checks,
  written as deterministic JSON; not tied to materials, so other harnesses
  can reuse it.
- `measure`: value statistics and a resolution-independent feature size.
- `material`: material reports (ranges, invalid values, seams against the
  tiling promise, lowering losses) and relationship checks (transforms
  leave no channel behind, resolution keeps feature size, incremental
  equals clean).
- `preview`: tiled, raking-light and mip previews and contact sheets;
  PNG output behind `std`.

`#![no_std]` with `alloc`.
