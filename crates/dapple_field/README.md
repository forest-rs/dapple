# dapple_field

Deterministic, band-limited procedural fields for [dapple](../../README.md).

- Typed domains: `Plane`, and `Periodic` fields that tile by construction;
  solid `Space` and `Periodic3` domains for 3D fields.
- Footprint band limiting, so realized textures and mips do not alias.
- Value and gradient noise, fBm and ridged fractals, cellular noise, in 2D
  and 3D (`Noise3`, `Fractal3`, `Cellular3`).
- Keyed hashing shared with sylva; no sequential random streams.
- `program`: fields as fingerprinted DAG values, evaluated bit-identically,
  with typed ports: scalars, masks, identifiers, vectors, linear colors and
  normals, checked when the program is built. One IR covers planar and solid
  nodes; `Op::Slice` turns a solid field into a planar one, and
  `SolidProgram::eval_chart` evaluates a solid field at surface points.
- `image`: sampled images, texels read back as a field (`Op::Sample`):
  bilinear, periodic-aware, footprint-filtered through mip chains.

Results are bit-exact across platforms. `#![no_std]` with `alloc`.
