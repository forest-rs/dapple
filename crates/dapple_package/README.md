# dapple_package

Portable material packages for [dapple](../../README.md).

- **Source** (`Package`): the editable JSON form of a material module:
  its interface (typed parameters, inputs including resource requirements,
  outputs with semantic declarations), required engine capabilities and
  module dependencies, presets, and a body that composes registered
  modules. Versioned, strict about unknown fields and versions, and
  canonical, so fingerprints are stable.
- **Execution artifact** (`CompiledPackage`): the source compiled against a
  `Registry` of modules and capabilities; a `dapple_material` module that
  instantiates like any other.
- **Presets** (`Preset`, `PresetSet`): named parameter sets, such as a fit's
  result, for packaged or native modules.

`#![no_std]` with `alloc`.
