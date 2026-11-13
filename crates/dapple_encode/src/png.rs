// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! PNG output of level 0.
//!
//! PNG holds one image, so [`write()`] stores level 0 only; use
//! [`crate::ktx2::write`] to keep the mip chain. One-channel textures become
//! grayscale PNG files. PNG has no two-channel color type, so two-channel textures
//! are written as RGB with blue 0. 16-bit textures become 16-bit grayscale
//! or RGB; PNG has no float samples, so float textures are refused.
//!
//! Files carry no color-space chunk (`sRGB`, `gAMA`, `iCCP`), sRGB textures
//! included: glTF 2.0 (section 3.8.3.2, PNG support) says such chunks should
//! not be present and that loaders should ignore them, since the material
//! slot that reads an image decides its transfer function. Whoever reads a
//! texture learns its encoding from [`EncodedTexture::format`] instead.

use std::vec::Vec;

use crate::pack::EncodedTexture;
use crate::quantize::PixelFormat;

/// Why PNG encoding failed.
#[derive(Debug)]
pub enum PngError {
    /// PNG cannot hold the texture's format.
    UnsupportedFormat(PixelFormat),
    /// The encoder failed.
    Encoding(png::EncodingError),
}

impl core::fmt::Display for PngError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedFormat(format) => write!(f, "PNG cannot hold {format:?} texels"),
            Self::Encoding(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for PngError {}

impl From<png::EncodingError> for PngError {
    fn from(error: png::EncodingError) -> Self {
        Self::Encoding(error)
    }
}

/// Level 0 of `texture` as PNG bytes.
pub fn write(texture: &EncodedTexture) -> Result<Vec<u8>, PngError> {
    let level = &texture.levels[0];
    let mut depth = png::BitDepth::Eight;
    let (color, data) = match texture.format {
        PixelFormat::R8Unorm => (png::ColorType::Grayscale, level.clone()),
        PixelFormat::Rg8Unorm => (
            png::ColorType::Rgb,
            level
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|rg| [rg[0], rg[1], 0])
                .collect(),
        ),
        PixelFormat::Rgba8Unorm | PixelFormat::Rgba8Srgb => (png::ColorType::Rgba, level.clone()),
        PixelFormat::R16Unorm | PixelFormat::Rgb16Unorm => {
            depth = png::BitDepth::Sixteen;
            // KTX2 levels are little-endian; PNG samples are big-endian.
            let swapped = level
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|b| [b[1], b[0]])
                .collect();
            let color = if texture.format == PixelFormat::R16Unorm {
                png::ColorType::Grayscale
            } else {
                png::ColorType::Rgb
            };
            (color, swapped)
        }
        PixelFormat::R32Float => return Err(PngError::UnsupportedFormat(texture.format)),
    };
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, texture.width, texture.height);
        encoder.set_color(color);
        encoder.set_depth(depth);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&data)?;
    }
    Ok(out)
}
