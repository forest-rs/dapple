// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! 8-bit quantization.

/// Texel formats of encoded textures.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum PixelFormat {
    /// One linear 8-bit channel.
    R8Unorm,
    /// Two linear 8-bit channels.
    Rg8Unorm,
    /// Four linear 8-bit channels.
    Rgba8Unorm,
    /// RGB encoded with the sRGB transfer function; alpha linear.
    Rgba8Srgb,
}

impl PixelFormat {
    /// Bytes per texel.
    #[must_use]
    pub const fn bytes_per_texel(self) -> usize {
        match self {
            Self::R8Unorm => 1,
            Self::Rg8Unorm => 2,
            Self::Rgba8Unorm | Self::Rgba8Srgb => 4,
        }
    }

    /// True when RGB use the sRGB transfer function.
    #[must_use]
    pub const fn is_srgb(self) -> bool {
        matches!(self, Self::Rgba8Srgb)
    }
}

/// The sRGB transfer function (IEC 61966-2-1) of a linear value, clamped to
/// `[0, 1]` first.
#[must_use]
pub fn linear_to_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * libm::powf(v, 1.0 / 2.4) - 0.055
    }
}

/// A `[0, 1]` value as an 8-bit unsigned normalized integer, rounding to
/// nearest. Values outside `[0, 1]` clamp; NaN becomes 0.
#[must_use]
pub fn quantize_unorm8(v: f32) -> u8 {
    let v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the value is clamped to [0, 255.5) before narrowing"
    )]
    let q = libm::floorf(v * 255.0 + 0.5) as u8;
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantization_rounds_and_clamps() {
        assert_eq!(quantize_unorm8(0.0), 0);
        assert_eq!(quantize_unorm8(1.0), 255);
        assert_eq!(quantize_unorm8(0.5), 128);
        assert_eq!(quantize_unorm8(-3.0), 0);
        assert_eq!(quantize_unorm8(7.0), 255);
        assert_eq!(quantize_unorm8(f32::NAN), 0);
    }

    #[test]
    fn srgb_matches_reference_points() {
        // Linear 0.5 is sRGB 0.7354, which quantizes to 188.
        assert_eq!(quantize_unorm8(linear_to_srgb(0.5)), 188);
        // Linear 0.2159 is sRGB 0.5020, just above the 127.5 boundary.
        assert_eq!(quantize_unorm8(linear_to_srgb(0.2159)), 128);
        assert_eq!(linear_to_srgb(0.0), 0.0);
        assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-6);
    }
}
