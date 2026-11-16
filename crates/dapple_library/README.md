# dapple_library

Ready-made dapple materials, each built as a `dapple_field` program so its
graph is inspectable, fingerprinted and evaluable like any other.

- `oak`: oak bark (planar and tileable) and solid oak wood around a trunk
  axis, for sawn faces and surface charts, its rings and rays band-limited
  by the footprint.
- `brick`: a running-bond brick wall with chipped edges, tileable.
- `gravel`: scattered pebbles of three sizes over sand, tileable.
- `parquet`: oak planks laid in herringbone, tileable.
- `glazed_brick`: keyed glazed bricks whose identity drives their shape,
  glaze thickness and color, and chips.
- `modules`: reusable material modules (`dapple_material`): ceramic body,
  mortar, stone, oak, finishes (glaze, varnish, oil) and weathering (grime,
  streaks, efflorescence), and assets composed of them: a glazed brick
  wall, a stone sill over brickwork, a varnished board and a threshold.
- metals: copper cladding and riveted steel plate, calibrated to tabulated
  F0, with a copper patina (tarnish and verdigris), rust, and chipping
  paint; packaged as weathered copper, rusted steel and painted steel,
  each with presets from fresh to heavily weathered.
- `packages`: the varnished board, the metals and the glazed brick wall as
  declarative module steps. The wall exposes the native asset's parameters
  and ships `clean`, `weathered` and `mossy` presets. `GlazedBricks` supplies
  its geometry and glaze maps on a 1 m wrapping grid; height detail, mask
  selection and region assignment assemble them before the finish and
  guarded weathering layers. Existing native child paths and output bits
  are preserved.
