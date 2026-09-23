# dapple_elements

Keyed element sets for [dapple](../../README.md): placed things (bricks,
tiles, stones, flakes) that exist before they are drawn.

- **Identity is not position.** An `ElementKey` is a keyed hash of the
  layout's author-given `LayoutId`, the element's anchor and slot. Moving,
  resizing or re-glazing an element changes its content fingerprint and
  never its key.
- **Layout is separate from realization.** A layout (`RunningBond`)
  produces an `ElementSet`, a column table in canonical key order; one set
  drives every output.
- **Surface programs are inspectable values.** A `SurfaceProgram` declares
  named typed inputs with an execution scope (material, element, sample),
  typed outputs, resource dependencies and a body of `Node`s. A
  `ProgramInstance` binds the inputs with a stable instance identity, and
  scopes are checked: a per-element value never depends on the sample
  position.
- **Compositing ownership is explicit.** `composite` writes each output
  under coverage compositing, and a winner label per texel that is a
  *summary*; `Realized::contributors` recomputes every contributor on
  request. Element identity and surface-material identity are separate
  outputs.
- **Incremental equals clean.** `Realized::update` recomputes only the
  tiles an edit reaches (old and new bounds) and is bit-identical to a clean
  realization.

`#![no_std]` with `alloc`.
