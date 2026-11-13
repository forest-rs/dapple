# dapple_graph

Incremental material graphs for [dapple](../../docs/design.md), on
`execution_graph`.

- **Field nodes** add one `dapple_field` op to their inputs' programs,
  importing shared subgraphs once.
- **Realize nodes** turn a periodic field into a raster of its type.
- **Raster nodes** apply a `dapple_raster` operation.
- **Normals nodes** realize a scalar field's normals from its gradient.
- **Mip nodes** filter a scalar raster down one level, bit for bit as
  `dapple_encode::data_mips`; chains of them build whole mip chains.
- **Reduce nodes** reduce a raster of any type one mip level under an
  explicit, type-checked reduction policy.
- **Sample nodes** read a raster of any type and its mips back as a field
  under an explicit sampling policy (nearest for identifiers), so realized,
  blurred or eroded rasters feed further field ops; their change regions
  come from the sampled rasters' changed tiles.

Every node's type is known when it is added (`MaterialGraph::port`): a node
that would refuse its inputs (a blur of identifiers, an average of normals,
linear sampling of identifiers) is refused then, and so is an edit that would
make any node's inputs invalid.

Each node's parameters are one of its inputs, so editing them re-runs exactly
that node and its dependents; a node whose output does not change cuts its
dependents off. Realize, raster, normals and mip nodes recompute only the
tiles an edit reaches. A node nothing reads can be removed. Values on edges
are reference-counted programs and rasters with fingerprints.

A run reports to [`footfall`](https://crates.io/crates/footfall): a scope per
`run` and per node execution (`dapple_graph.run`, `dapple_graph.node`, naming
the node's kind and key) and, once per run, a counter for each `TileReport`
field (`dapple_graph.tiles_recomputed` and so on). With no backend installed
this costs a level check per emission; a host that installs one, such as
`footfall::runtime`, sees where time went. Nothing is read back, so results,
fingerprints and `TileReport`s are identical either way. The crate docs list
every name.

`#![no_std]` with `alloc`.
