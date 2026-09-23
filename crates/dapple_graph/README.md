# dapple_graph

Incremental material graphs for [dapple](../../docs/design.md), on
`execution_graph`.

- **Field nodes** add one `dapple_field` op to their inputs' programs,
  importing shared subgraphs once.
- **Realize nodes** turn a scalar periodic field into a raster.
- **Raster nodes** apply a `dapple_raster` operation.

Each node's parameters are one of its inputs, so editing them re-runs exactly
that node and its dependents. Values on edges are reference-counted programs
and rasters with fingerprints.

`execution_graph` is a git dependency pinned to the `forest-rs/execution`
commit that added its executor seam, until a release contains it.

`#![no_std]` with `alloc`.
