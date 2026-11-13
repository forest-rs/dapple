# dapple_export

Exports [dapple](../../docs/design.md) materials as self-describing results.

`export` lowers and packs a `dapple_material::Material` and returns an
`Exported` value: the packed textures, and one account per OpenPBR parameter
and auxiliary channel (occlusion, height) saying whether it is the
specification's default, a constant, which channels of which texture hold it
and how to decode them, or lost, with why. The account merges lowering's
dropped parameters and packing's unsupported and approximated ones, and
says whether the profile's own material (the glTF material) carries each,
so a consumer reads one thing. What the profile cannot carry comes from
the raw profile as extra textures or constants.

Everything a writer needs is in that value, so other writers (a USD
adapter, an engine importer) read it rather than files:

- `gltf::material` builds the glTF material, its textures, images and
  sampler (repeating along the axes the material tiles), with
  `KHR_materials_specular`, `_ior`, `_anisotropy`, `_transmission`,
  `_diffuse_transmission`, `_clearcoat`, `_sheen` and `_emissive_strength`
  as the material needs them. `gltf::document` wraps it in a complete glTF
  2.0 file: one quad the material's physical size, with normals, tangents
  and UVs, so every bake opens by itself in Blender or any glTF viewer.
- `Exported::manifest` describes the export as a versioned JSON manifest
  (`format` `"dapple-material"`, `version` 1): packing settings,
  provenance (subject, module instance and argument fingerprint, grid,
  physical size, tiling), conventions (meters, cd/m², linear Rec. 709,
  normal and axis frames per texture location), the account, textures with
  their color spaces and bit depths, files with content hashes, and the
  material laboratory's verdict. `Manifest::from_json` reads it back and
  refuses other versions and unknown fields.
- `Exported::write_dir` (feature `std`) writes each texture as PNG,
  `material.gltf` and `manifest.json`, removing an old manifest first and
  writing the new one last, so a manifest always vouches for complete
  files; `Manifest::changed_files` says which files no longer match.

JSON goes through `serde_json`, already the workspace's choice for
`dapple_package` and `no_std` with `alloc`: one schema serves writing and
reading, and output is deterministic (fixed field order, shortest
round-tripping numbers). Content hashes reuse `dapple_field`'s keyed hash;
they detect changes, not tampering. Textures are PNG, the only lossless
format glTF core reads; 16-bit normal maps stay 16-bit, which glTF allows
for PNG.

`#![no_std]` with `alloc`; the `std` feature adds directory writing.
