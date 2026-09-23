# dapple_encode

Mip chains, specular anti-aliasing, channel packing, and texture files for
[dapple](../../docs/design.md).

Each kind of data gets its own mip rule:

- **Color:** filtered in linear light, premultiplied where there is opacity.
- **Coverage:** alpha-tested masks keep their coverage at the cutoff on every
  level (Castaño 2010).
- **Normals:** renormalized, with the lost length moved into roughness as
  variance (Toksvig), so distant bumpy surfaces keep their energy.
- **Data:** filtered as-is.

Box (exact area, any size) and Kaiser-windowed sinc filters follow the image's
edge policy, so wrapping textures tile on every level.

`pack` turns OpenPBR maps into 8-bit textures for a profile:
- `Lightweald`: the `lightweald_material` slots, with a two-channel normal map
  and an occlusion/roughness/metalness texture;
- `Gltf`: glTF 2.0 textures;
- `Raw`: one texture per parameter.

`ktx2::write` writes a texture with its whole mip chain as KTX2, and
`png::write` (feature `std`) writes level 0 as PNG. The KTX2 files are
uncompressed; block compression comes later.

`#![no_std]` with `alloc`; the `std` feature adds PNG output.
