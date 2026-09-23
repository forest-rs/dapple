# dapple_compress

Block compression and Zstandard supercompression of
[dapple](../../docs/design.md) textures, through `ctt`.

`compress` writes a `dapple_encode::EncodedTexture` as KTX2 in its kind's
Lightweald pool format for an `Encoding`, keeping dapple's own mip chain:

| Kind | Uncompressed | BC | ASTC |
|---|---|---|---|
| color | RGBA8 sRGB | BC7 sRGB | ASTC 4 × 4 sRGB |
| data | RGBA8 | BC7 | ASTC 4 × 4 |
| normal | RG8 | BC5 | ASTC 4 × 4 (x, y, 0, 1) |

`Encoding`, `Quality` and Zstandard supercompression mirror Lightweald's
`texture_bake`, and the output loads through `lightweald_texture_io`. `ctt`
is pinned to an exact version because compressed bytes depend on the
encoders.

Standard library only, with C++ encoders built by `ctt`; not for `no_std` or
WebAssembly. Pure-Rust uncompressed output is `dapple_encode::ktx2`.
