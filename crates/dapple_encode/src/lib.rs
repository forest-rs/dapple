// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Mip chains, specular anti-aliasing, channel packing, and texture files for
//! dapple.
//!
//! A generic downsampler treats every texture as color. Dapple does not: each
//! kind of data has its own mip rule, and a [`Profile`] decides how the
//! results are packed for a consumer.
//!
//! - **Color** ([`color_mips`]): filtered in linear light. With opacity, the
//!   color is filtered premultiplied, so transparent texels do not bleed their
//!   color into their neighbors.
//! - **Coverage** ([`preserve_coverage`]): an alpha-tested mask keeps the
//!   fraction of texels at or above its cutoff on every level (Castaño 2010).
//!   Without this, alpha-tested foliage thins away with distance.
//! - **Normals** ([`normal_mips`]): unit normals are averaged, and the length
//!   the average loses measures how much the normals disagreed. The chain
//!   stores renormalized normals and moves that variance into roughness
//!   (Toksvig), so distant bumpy surfaces widen their highlight instead of
//!   turning into sparkling mirrors.
//! - **Data** ([`data_mips`]): filtered as-is.
//!
//! [`Filter::Box`] averages each destination texel's exact source area and
//! handles any size; [`Filter::Kaiser`] is a Kaiser-windowed sinc with less
//! blur. Filtering follows the image's [`Edge`]: a wrapping image stays
//! tileable on every level.
//!
//! [`pack()`] turns a [`MaterialMaps`] set into a [`Bundle`] of 8-bit textures
//! for a [`Profile`]: [`Profile::Lightweald`] (the slots of
//! `lightweald_material`), [`Profile::Gltf`], or [`Profile::Raw`]. Each
//! texture carries its whole mip chain. [`ktx2::write`] writes one as a KTX2
//! file with those mips; `png::write` (behind the `std` feature) writes
//! level 0 as PNG.
//!
//! **Determinism.** Filter weights are computed in `f64` with `libm`, sums
//! run in a fixed order, and coverage search runs a fixed number of steps, so
//! equal inputs give equal bytes on every platform.
//!
//! ```
//! use dapple_encode::{Filter, Image, MaterialMaps, PackSettings, Profile, ktx2, pack};
//! use dapple_raster::Edge;
//!
//! let flat = Image::new(4, 4, 3, Edge::Wrap, vec![0.0, 0.0, 1.0].repeat(16))?;
//! let maps = MaterialMaps { normal: Some(flat), ..MaterialMaps::default() };
//! let bundle = pack(&maps, Profile::Lightweald, &PackSettings::default())?;
//! let normal = bundle.texture("normal").expect("normals were supplied");
//! assert_eq!(normal.levels.len(), 3); // 4×4, 2×2, 1×1
//! let file = ktx2::write(normal);
//! assert_eq!(&file[..12], &ktx2::IDENTIFIER);
//! # Ok::<(), dapple_encode::EncodeError>(())
//! ```

#![no_std]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod filter;
pub mod ktx2;
mod mips;
mod pack;
#[cfg(feature = "std")]
pub mod png;
mod quantize;

#[cfg(test)]
mod golden_tests;

use alloc::vec::Vec;
use core::fmt;

use dapple_raster::{Edge, Raster};

pub use filter::{Filter, KAISER_BETA, KAISER_RADIUS};
pub use mips::{
    CoverageLevel, MipChain, NormalChain, color_mips, data_mips, normal_mips, preserve_coverage,
};
pub use pack::{Bundle, EncodedTexture, MaterialMaps, PackReport, PackSettings, Profile, pack};
pub use quantize::{PixelFormat, linear_to_srgb, quantize_unorm8};

/// Largest supported image dimension.
pub const MAX_DIMENSION: u32 = 1 << 15;

/// A rejected image, parameter or material set.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum EncodeError {
    /// A dimension is zero or exceeds [`MAX_DIMENSION`].
    InvalidSize {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
    },
    /// The channel count is not 1 to 4.
    InvalidChannels(usize),
    /// The value buffer length is not `width * height * channels`.
    LengthMismatch {
        /// Expected length.
        expected: usize,
        /// Supplied length.
        found: usize,
    },
    /// An input has the wrong number of channels for its role.
    ChannelMismatch {
        /// The role, for example `"normal"`.
        role: &'static str,
        /// Channels the role needs.
        expected: usize,
        /// Channels supplied.
        found: usize,
    },
    /// Inputs of one material have different sizes or edge policies.
    Incompatible {
        /// The role that differs from the first input.
        role: &'static str,
    },
    /// A parameter is outside its documented range.
    InvalidParameter {
        /// Parameter name.
        name: &'static str,
    },
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSize { width, height } => write!(f, "invalid image size {width}×{height}"),
            Self::InvalidChannels(n) => write!(f, "{n} channels; images have 1 to 4"),
            Self::LengthMismatch { expected, found } => {
                write!(f, "expected {expected} values, got {found}")
            }
            Self::ChannelMismatch {
                role,
                expected,
                found,
            } => write!(f, "{role} needs {expected} channels, got {found}"),
            Self::Incompatible { role } => {
                write!(
                    f,
                    "{role} differs in size or edge policy from the other inputs"
                )
            }
            Self::InvalidParameter { name } => write!(f, "parameter {name} is out of range"),
        }
    }
}

impl core::error::Error for EncodeError {}

/// A row-major image of 1 to 4 interleaved linear `f32` channels.
///
/// Row `y` increases with the source domain's y axis, as in
/// [`dapple_raster::Raster`]; file writers store row 0 first. The [`Edge`]
/// policy says how filters read past the border.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    width: u32,
    height: u32,
    channels: usize,
    edge: Edge,
    values: Vec<f32>,
}

impl Image {
    /// Builds an image from interleaved row-major `values`.
    pub fn new(
        width: u32,
        height: u32,
        channels: usize,
        edge: Edge,
        values: Vec<f32>,
    ) -> Result<Self, EncodeError> {
        if width == 0 || height == 0 || width > MAX_DIMENSION || height > MAX_DIMENSION {
            return Err(EncodeError::InvalidSize { width, height });
        }
        if !(1..=4).contains(&channels) {
            return Err(EncodeError::InvalidChannels(channels));
        }
        let expected = width as usize * height as usize * channels;
        if values.len() != expected {
            return Err(EncodeError::LengthMismatch {
                expected,
                found: values.len(),
            });
        }
        Ok(Self {
            width,
            height,
            channels,
            edge,
            values,
        })
    }

    /// Texels per row.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Rows.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Channels per texel.
    #[must_use]
    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// The edge policy.
    #[must_use]
    pub const fn edge(&self) -> Edge {
        self.edge
    }

    /// The interleaved row-major values.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// The channels of texel `(x, y)`.
    ///
    /// # Panics
    ///
    /// Panics when `(x, y)` is outside the image.
    #[must_use]
    pub fn texel(&self, x: u32, y: u32) -> &[f32] {
        assert!(x < self.width && y < self.height, "texel out of range");
        let start = (y as usize * self.width as usize + x as usize) * self.channels;
        &self.values[start..start + self.channels]
    }

    /// One channel as a single-channel image.
    ///
    /// # Panics
    ///
    /// Panics when `channel` is not below [`Self::channels`].
    #[must_use]
    pub fn channel(&self, channel: usize) -> Self {
        assert!(channel < self.channels, "channel out of range");
        Self {
            width: self.width,
            height: self.height,
            channels: 1,
            edge: self.edge,
            values: self
                .values
                .chunks_exact(self.channels)
                .map(|t| t[channel])
                .collect(),
        }
    }

    fn texels(&self) -> usize {
        self.width as usize * self.height as usize
    }

    fn same_grid(&self, other: &Self) -> bool {
        self.width == other.width && self.height == other.height && self.edge == other.edge
    }
}

impl From<&Raster<f32>> for Image {
    fn from(raster: &Raster<f32>) -> Self {
        Self {
            width: raster.width(),
            height: raster.height(),
            channels: 1,
            edge: raster.edge(),
            values: raster.values().to_vec(),
        }
    }
}

impl<const N: usize> From<&Raster<[f32; N]>> for Image {
    /// Interleaves an `N`-channel raster.
    ///
    /// # Panics
    ///
    /// Panics unless `N` is 1 to 4.
    fn from(raster: &Raster<[f32; N]>) -> Self {
        assert!((1..=4).contains(&N), "images have 1 to 4 channels");
        Self {
            width: raster.width(),
            height: raster.height(),
            channels: N,
            edge: raster.edge(),
            values: raster.values().iter().flatten().copied().collect(),
        }
    }
}
