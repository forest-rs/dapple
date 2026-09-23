// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Block compression and Zstandard supercompression of dapple textures.
//!
//! [`compress`] writes an [`EncodedTexture`] from `dapple_encode` as a KTX2
//! file in its [`TextureKind`]'s pool format for an [`Encoding`], keeping
//! dapple's own mip chain. The formats are those of Lightweald's texture
//! pools, as its `texture_bake` tool writes them:
//!
//! | Kind | Uncompressed | BC | ASTC |
//! |---|---|---|---|
//! | color | RGBA8 sRGB | BC7 sRGB | ASTC 4 × 4 sRGB |
//! | data | RGBA8 | BC7 | ASTC 4 × 4 |
//! | normal | RG8 | BC5 | ASTC 4 × 4 (x, y, 0, 1) |
//!
//! `ctt` is used only as the encoder and container writer: Intel's ISPC
//! compressor for BC7 and BC5, ARM's astcenc for ASTC. Mip levels are never
//! regenerated, so what dapple filtered, coverage-corrected or
//! variance-folded is what gets encoded. Dependency-free uncompressed output
//! stays in `dapple_encode::ktx2`.
//!
//! **Determinism.** Compressed bytes depend on the encoders, so `ctt` is
//! pinned to an exact version (0.5.0). Output is repeatable for a given
//! version and platform; encoders may differ across CPU architectures, so
//! golden tests pin the decoded format and level structure rather than
//! compressed bytes.

use ctt::{
    AlphaMode, ColorSpace, Container, ConvertSettings, Format, Image, Ktx2Supercompression,
    PipelineOutput, Surface, TargetFormat,
};
use dapple_encode::{EncodedTexture, PixelFormat, TextureKind};

/// Which pool encoding to write.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Encoding {
    /// RGBA8 (sRGB for color) or RG8 for normals.
    Uncompressed,
    /// BC7 (sRGB for color) or BC5 for normals: desktop GPUs and Apple
    /// silicon Macs.
    #[default]
    Bc,
    /// ASTC 4 × 4 (sRGB for color): Apple and mobile GPUs.
    Astc,
}

/// Encoder effort.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Quality {
    /// Quick previews.
    Fast,
    /// Good quality at a moderate speed.
    #[default]
    Balanced,
    /// The best quality the encoders offer; slow.
    Best,
}

/// How to compress.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CompressSettings {
    /// Target pool encoding.
    pub encoding: Encoding,
    /// Encoder effort.
    pub quality: Quality,
    /// Supercompress levels with Zstandard (smaller files; loaders undo it).
    pub zstd: bool,
}

impl CompressSettings {
    /// `encoding` at balanced quality with Zstandard supercompression, as
    /// `texture_bake` defaults.
    #[must_use]
    pub const fn new(encoding: Encoding) -> Self {
        Self {
            encoding,
            quality: Quality::Balanced,
            zstd: true,
        }
    }
}

/// Why compression failed.
#[derive(Debug)]
pub enum CompressError {
    /// A side is not a power of two (texture pools are sized in powers of
    /// two).
    NotPowerOfTwo {
        /// Width in texels.
        width: u32,
        /// Height in texels.
        height: u32,
    },
    /// A block-compressed texture smaller than a 4 × 4 block.
    TooSmall {
        /// Width in texels.
        width: u32,
        /// Height in texels.
        height: u32,
    },
    /// The texture's texel format has no pool counterpart (16-bit and float
    /// data).
    UnsupportedFormat(PixelFormat),
    /// The encoder or container writer failed.
    Encode(ctt::Error),
}

impl core::fmt::Display for CompressError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotPowerOfTwo { width, height } => {
                write!(f, "{width} × {height} is not a power of two in each side")
            }
            Self::TooSmall { width, height } => {
                write!(f, "{width} × {height} is smaller than a 4 × 4 block")
            }
            Self::UnsupportedFormat(format) => {
                write!(f, "{format:?} textures have no pool format")
            }
            Self::Encode(error) => write!(f, "encoding failed: {error}"),
        }
    }
}

impl std::error::Error for CompressError {}

/// The KTX2 (Vulkan) format of `kind` in `encoding`.
#[must_use]
pub fn target_format(kind: TextureKind, encoding: Encoding) -> Format {
    match (encoding, kind) {
        (Encoding::Uncompressed, TextureKind::Color) => Format::R8G8B8A8_SRGB,
        (Encoding::Uncompressed, TextureKind::Data) => Format::R8G8B8A8_UNORM,
        (Encoding::Uncompressed, TextureKind::Normal) => Format::R8G8_UNORM,
        (Encoding::Bc, TextureKind::Color) => Format::BC7_SRGB_BLOCK,
        (Encoding::Bc, TextureKind::Data) => Format::BC7_UNORM_BLOCK,
        (Encoding::Bc, TextureKind::Normal) => Format::BC5_UNORM_BLOCK,
        (Encoding::Astc, TextureKind::Color) => Format::ASTC_4x4_SRGB_BLOCK,
        (Encoding::Astc, TextureKind::Data | TextureKind::Normal) => Format::ASTC_4x4_UNORM_BLOCK,
    }
}

/// One level as RGBA8 texels in the layout the pool expects: color and data
/// as they are (one channel in red), normals as `(x, y, 0, 255)`.
fn rgba8(kind: TextureKind, format: PixelFormat, level: &[u8]) -> Vec<u8> {
    let channels = format.channels();
    level
        .chunks_exact(channels)
        .flat_map(|t| match (kind, channels) {
            (TextureKind::Normal, _) => [t[0], t[1], 0, 255],
            (_, 1) => [t[0], 0, 0, 255],
            (_, 2) => [t[0], t[1], 0, 255],
            (_, _) => [t[0], t[1], t[2], t[3]],
        })
        .collect()
}

/// Writes `texture` and its mip chain as a KTX2 file in the pool format of
/// its kind for `settings.encoding`.
///
/// Every level is passed to the encoder as-is; none is regenerated. Like
/// Lightweald's pools, sides must be powers of two, and block-compressed
/// textures at least 4 × 4.
pub fn compress(
    texture: &EncodedTexture,
    settings: CompressSettings,
) -> Result<Vec<u8>, CompressError> {
    let (width, height) = (texture.width, texture.height);
    if !width.is_power_of_two() || !height.is_power_of_two() {
        return Err(CompressError::NotPowerOfTwo { width, height });
    }
    if settings.encoding != Encoding::Uncompressed && (width < 4 || height < 4) {
        return Err(CompressError::TooSmall { width, height });
    }
    if texture.format.bytes_per_channel() != 1 {
        return Err(CompressError::UnsupportedFormat(texture.format));
    }
    let (color_space, alpha) = match texture.kind {
        TextureKind::Color => (ColorSpace::Srgb, AlphaMode::Straight),
        // Data channels are independent: no premultiplication.
        TextureKind::Data | TextureKind::Normal => (ColorSpace::Linear, AlphaMode::Opaque),
    };
    let levels = texture
        .levels
        .iter()
        .enumerate()
        .map(|(index, level)| {
            let w = (width >> index).max(1);
            let h = (height >> index).max(1);
            Surface {
                data: rgba8(texture.kind, texture.format, level),
                width: w,
                height: h,
                depth: 1,
                stride: w * 4,
                slice_stride: 0,
                format: Format::R8G8B8A8_UNORM,
                color_space,
                alpha,
            }
        })
        .collect();

    // ctt's encoders name the linear block formats; the surfaces' color space
    // tags color as sRGB.
    let target = match target_format(texture.kind, settings.encoding) {
        Format::BC7_SRGB_BLOCK => compressed(Format::BC7_UNORM_BLOCK),
        Format::ASTC_4x4_SRGB_BLOCK => compressed(Format::ASTC_4x4_UNORM_BLOCK),
        format if settings.encoding == Encoding::Uncompressed => TargetFormat::Uncompressed(format),
        format => compressed(format),
    };
    let output = ctt::convert(
        Image {
            surfaces: vec![levels],
            kind: ctt::TextureKind::Texture2D,
        },
        ConvertSettings {
            format: Some(target),
            container: Container::Ktx2(
                settings
                    .zstd
                    .then_some(Ktx2Supercompression::Zstd { level: 15 }),
            ),
            quality: match settings.quality {
                Quality::Fast => ctt::Quality::Fast,
                Quality::Balanced => ctt::Quality::Basic,
                Quality::Best => ctt::Quality::Slow,
            },
            allow_discarding_alpha: true,
            mipmap: false,
            ..Default::default()
        },
    )
    .map_err(CompressError::Encode)?;
    match output {
        PipelineOutput::Encoded(bytes) => Ok(bytes),
        PipelineOutput::Raw(_) => unreachable!("a KTX2 container was requested"),
    }
}

fn compressed(format: Format) -> TargetFormat {
    TargetFormat::Compressed {
        format,
        encoder: ctt::encoders::Encoder::Auto,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dapple_encode::{Edge, Filter, Image as Map, MaterialMaps, PackSettings, Profile, pack};

    fn u32_at(b: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
    }

    fn material() -> MaterialMaps {
        let n = 16_u32;
        let mut color = Vec::new();
        let mut normal = Vec::new();
        let mut rough = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let (fx, fy) = (x as f32 / 15.0, y as f32 / 15.0);
                color.extend_from_slice(&[fx, fy, 0.5]);
                let (dx, dy) = (fx - 0.5, fy - 0.5);
                let l = (dx * dx + dy * dy + 1.0).sqrt();
                normal.extend_from_slice(&[dx / l, dy / l, 1.0 / l]);
                rough.push(0.2 + 0.5 * fx);
            }
        }
        let map = |c, v| Map::new(n, n, c, Edge::Wrap, v).unwrap();
        MaterialMaps {
            base_color: Some(map(3, color)),
            normal: Some(map(3, normal)),
            specular_roughness: Some(map(1, rough)),
            ..MaterialMaps::default()
        }
    }

    #[test]
    fn every_kind_and_encoding_keeps_the_chain() {
        let settings = PackSettings {
            filter: Filter::Box,
            ..PackSettings::default()
        };
        let bundle = pack(&material(), Profile::Lightweald, &settings).unwrap();
        for encoding in [Encoding::Uncompressed, Encoding::Bc, Encoding::Astc] {
            for zstd in [false, true] {
                for texture in &bundle.textures {
                    let settings = CompressSettings {
                        zstd,
                        ..CompressSettings::new(encoding)
                    };
                    let file = compress(texture, settings).unwrap();
                    let format = target_format(texture.kind, encoding);
                    assert_eq!(
                        u32_at(&file, 12),
                        format.value(),
                        "{} {encoding:?}",
                        texture.name
                    );
                    assert_eq!((u32_at(&file, 20), u32_at(&file, 24)), (16, 16));
                    assert_eq!(u32_at(&file, 40), 5, "16, 8, 4, 2, 1");
                    assert_eq!(
                        u32_at(&file, 44),
                        u32::from(zstd) * 2,
                        "Zstandard is scheme 2"
                    );
                    let again = compress(texture, settings).unwrap();
                    assert_eq!(file, again, "compression is repeatable");
                }
            }
        }
    }

    #[test]
    fn uncompressed_output_keeps_dapples_levels() {
        let bundle = pack(&material(), Profile::Lightweald, &PackSettings::default()).unwrap();
        let normal = bundle.texture("normal").unwrap();
        let file = compress(
            normal,
            CompressSettings {
                zstd: false,
                ..CompressSettings::new(Encoding::Uncompressed)
            },
        )
        .unwrap();
        // Every level's bytes match dapple's own (RG8 normals, no mips rebuilt).
        for (index, level) in normal.levels.iter().enumerate() {
            let entry = 80 + index * 24;
            let offset = usize::try_from(u64::from_le_bytes(
                file[entry..entry + 8].try_into().unwrap(),
            ))
            .unwrap();
            assert_eq!(
                &file[offset..offset + level.len()],
                level.as_slice(),
                "level {index}"
            );
        }
    }

    #[test]
    fn rejects_what_pools_cannot_hold() {
        let odd = Map::new(12, 8, 3, Edge::Clamp, vec![0.5; 12 * 8 * 3]).unwrap();
        let maps = MaterialMaps {
            base_color: Some(odd),
            ..MaterialMaps::default()
        };
        let bundle = pack(&maps, Profile::Lightweald, &PackSettings::default()).unwrap();
        let settings = CompressSettings::new(Encoding::Bc);
        assert!(matches!(
            compress(&bundle.textures[0], settings),
            Err(CompressError::NotPowerOfTwo { .. })
        ));
        let tiny = Map::new(2, 2, 3, Edge::Clamp, vec![0.5; 12]).unwrap();
        let maps = MaterialMaps {
            base_color: Some(tiny),
            ..MaterialMaps::default()
        };
        let bundle = pack(&maps, Profile::Lightweald, &PackSettings::default()).unwrap();
        assert!(matches!(
            compress(&bundle.textures[0], settings),
            Err(CompressError::TooSmall { .. })
        ));
    }
}
