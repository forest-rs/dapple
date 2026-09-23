# dapple_imaging

Coverage masks for [dapple](../../docs/design.md) from vector shapes.

- Leaf contours, tile layouts and decals are recorded as an
  [`imaging`](https://github.com/forest-rs/imaging) scene, in domain units.
- `rasterize` turns a scene into a coverage mask over a
  `dapple_raster::Realization`: each texel holds the fraction of its area
  the shapes cover, so masks line up with the fields realized beside them.
  Wrapping realizations wrap shapes across the tile's edges.
- `coverage_image` makes a mask a footprint-filtered field for
  `dapple_field` programs (`Op::Sample`), with mips of exact area means.

Rasterization is `imaging_vello_cpu`'s 8-bit CPU pipeline at a pinned git
revision: coverage is quantized to 1/255, and curves are flattened within a
quarter texel. Golden tests pin the output.

`#![no_std]` with `alloc`.
