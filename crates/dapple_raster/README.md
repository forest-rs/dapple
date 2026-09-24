# dapple_raster

Deterministic rasters for [dapple](../../README.md): explicit realization of
`dapple_field` fields at a resolution, and neighborhood operations that state
their parameters in domain units and their kernel footprints in texels.

- `realize` with a `Realization`: one period of a periodic field (wrapping
  edges) or any region (clamping edges).
- `GaussianBlur`, `HeightToNormal`, `AmbientOcclusion`, `DistanceTransform`,
  `Morphology`, and `Streak` (one-sided trails of a physical length).
- Measured value shaping: `Histogram`, exact `percentiles` and
  `PercentileRemap` (controlled normalization, coverage by percentile).
- Every op states its `OpCategory`: local stencil, separable pass,
  reduction, global transform or iterative solve.
- `SampledField`: a raster as a field again, with bilinear filtering shared
  with `dapple_field::SampleImage`. `Edge` is `dapple_field`'s.

Wrapping rasters are tori: every operation commutes exactly with rolling the
raster, so results tile. Results are bit-exact across platforms.

`#![no_std]` with `alloc`.
