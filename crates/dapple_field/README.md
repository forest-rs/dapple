# dapple_field

Deterministic, band-limited procedural fields for [dapple](../../README.md).

- Typed domains: `Plane`, and `Periodic` fields that tile by construction.
- Footprint band limiting, so realized textures and mips do not alias.
- Value and gradient noise, fBm and ridged fractals, cellular noise.
- Keyed hashing shared with sylva; no sequential random streams.
- `program`: fields as fingerprinted DAG values, evaluated bit-identically.

Results are bit-exact across platforms. `#![no_std]` with `alloc`.
