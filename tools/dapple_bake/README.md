# dapple_bake

Bakes the dapple material recipes a manifest lists into textures: dapple's
counterpart to Lightweald's `texture_bake`, for procedural sources.

```sh
cargo run -p dapple_bake --release -- --manifest tools/dapple_bake/materials/materials.toml
```

Options: `--out <dir>` (default the repository's git-ignored
`.local/gallery/dapple-bake`, which survives `cargo clean`), `--encoding
uncompressed,bc,astc` (Lightweald pool encodings besides the uncompressed KTX2;
default `bc`), `--quality fast|balanced|best`, `--force`, and `--profile <capture.ffcap>`.

- **Manifest** (`materials/materials.toml`): one `[[material]]` per material,
  with its `id`, its `recipe` file, the `profiles` to write (`lightweald`,
  `gltf`, `raw`), a mip `filter` and an optional `alpha_cutoff`.
- **Recipes** (`materials/recipes/*.toml`) are `dapple_graph` recipes in TOML:
  labeled field, realize and raster nodes, and outputs naming which raster
  nodes fill which material maps. `oak_bark.toml` is the field gallery's bark
  study as a recipe.
- **Output** goes to `<out>/<profile>/<material>/`: KTX2 for Lightweald, PNG
  for glTF and raw. Maps a profile cannot carry are listed.
- **Caching:** each material's stamp records its recipe fingerprint and the
  bake settings, its written files, and a completion count. An unchanged
  material is skipped only when its complete stamp matches and every listed
  file exists. `--force` re-bakes. Bakes preserve earlier outputs when a
  profile, texture or material is dropped; remove unwanted outputs explicitly.

## Profiling a bake

`--profile <capture.ffcap>` installs a [`footfall`](https://crates.io/crates/footfall)
runtime and writes what it recorded to a capture the `footfall` command reads:

```sh
cargo run -p dapple_bake --release -- --manifest tools/dapple_bake/materials/materials.toml \
    --profile bake.ffcap
footfall analyze summary bake.ffcap
```

The capture holds a scope for the whole manifest, one per material, and one
per step (`images`, which includes the `dapple_graph` run and its node
scopes, `pack`, `ktx2`, `compress`, `png`, `write`), with counters for
materials baked or skipped and files and bytes written; the names are listed
in the library's `observe` module docs. Material ids are not scope arguments
(arguments are static), so each material's scope has an `index` and a
`dapple_bake.material_id` message just before it carries the id. Profiling
changes nothing the bake writes, and the tool says so if the runtime evicted
events before they were written. Without the flag nothing is recorded.
