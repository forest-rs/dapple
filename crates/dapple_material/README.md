# dapple_material

Typed material values for [dapple](../../README.md), the operations that
combine them, and parameterized modules that build them.

- **Material values.** A `Material` binds every OpenPBR parameter (the
  `openpbr` crate's vocabulary) to a constant or a typed map, with auxiliary
  channels (height in meters, occlusion, region and surface identity) kept
  apart, all on one grid.
- **Four operations with their own contracts.** `select` decides once per
  texel for every channel; `apply_detail` perturbs height and normals in a
  stated layer, with an identity, and is the only place reoriented normal
  mapping lives; `coat` sets OpenPBR's coat over an untouched base;
  `deposit` covers a material, changing coverage, height, identity and
  optics together. `transform` moves every channel. Each returns a
  `Report` naming what it approximated (roughness in α, averaged normals,
  mixed layer stacks, collapsed coats, winner labels, resampling) where it
  happened.
- **Programs over materials.** `program::evaluate` runs a
  `dapple_field` scoped program per texel with inputs bound to channels,
  raster-pass results and positions, checking scopes.
- **Modules.** A `Module` publishes an `Interface`: typed parameters with
  units, ranges and defaults; material, map and host-resolved resource
  inputs; named outputs; a versioned `ModuleId`. Names are borrowed for
  native modules and owned for modules loaded as data (`dapple_package`).
  `Context::instantiate` checks bindings, derives seeds from the instance path, and records every
  operation's report against the instance that made it.
- **Lowering.** `lower::maps` turns a material into `dapple_encode`'s maps
  and names the parameters they cannot carry.

`#![no_std]` with `alloc`.
