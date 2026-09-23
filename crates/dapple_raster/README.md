# dapple_raster

Deterministic rasters for [dapple](../../README.md): explicit realization of
`dapple_field` fields at a resolution, and neighborhood operations that state
their parameters in domain units and their kernel footprints in texels.

- `realize` with a `Realization`: one period of a periodic field (wrapping
  edges) or any region (clamping edges).
- `GaussianBlur`, `HeightToNormal`, `AmbientOcclusion`, `DistanceTransform`.
- `SampledField`: a raster as a field again, with bilinear filtering.

Wrapping rasters are tori: every operation commutes exactly with rolling the
raster, so results tile. Results are bit-exact across platforms.

`#![no_std]` with `alloc`.
