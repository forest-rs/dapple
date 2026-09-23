# dapple_bake

Bakes the dapple material recipes a manifest lists into textures: dapple's
counterpart to Lightweald's `texture_bake`, for procedural sources.

```sh
cargo run -p dapple_bake --release -- --manifest tools/dapple_bake/materials/materials.toml
```

Options: `--out <dir>` (default `target/dapple-bake`), `--encoding
uncompressed,bc,astc` (Lightweald pool encodings besides the uncompressed KTX2;
default `bc`), `--quality fast|balanced|best`, and `--force`.

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
  bake settings. An unchanged material is skipped, and `--force` re-bakes.
