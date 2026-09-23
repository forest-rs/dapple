// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Coverage masks for dapple from vector shapes.
//!
//! Leaf contours, tile layouts and decals are shapes, not fields. This crate
//! records them as an [`imaging`] scene and rasterizes the scene into a
//! coverage mask: each texel holds the fraction of its area the shapes
//! cover. The mask is a raster input to `dapple_raster` operations, or,
//! through [`coverage_image`], a footprint-filtered field for
//! `dapple_field` programs (`Op::Sample`).
//!
//! **Units.** Scenes are drawn in domain units. A [`Realization`] names the
//! region, the texel grid and the edge policy, exactly as for
//! `dapple_raster::realize`, so a mask lines up texel for texel with the
//! fields realized beside it:
//!
//! - [`rasterize`] maps the realization's region onto its grid and covers
//!   each texel with its area: the texel is the footprint.
//! - A wrapping realization is one period of a periodic domain. The scene
//!   is also drawn one period away in every direction, so a shape that
//!   crosses the tile's edge reappears on the opposite side and the mask
//!   tiles. Shapes must lie within one period of the tile for this to hold.
//!
//! **Coverage.** The mask is the scene's composited alpha: opaque brushes
//! give area coverage, a brush's alpha scales it, and colors are ignored.
//! Coverage is quantized to 1/255, the precision of the 8-bit pipeline.
//! Curves are flattened into chords within a quarter texel (the renderer's
//! fixed tolerance), so a convex curved outline sits up to a quarter texel
//! inside the true curve: realize masks with a few texels across their
//! smallest curves.
//!
//! **Determinism.** Rasterization is `imaging_vello_cpu`'s CPU renderer
//! with its `OptimizeSpeed` (8-bit) pipeline, at a pinned git revision of
//! `forest-rs/imaging`. Equal scenes and realizations give equal masks,
//! which golden tests pin; a revision bump that changes coverage shows up
//! there.
//!
//! ```
//! use dapple_field::Domain;
//! use dapple_imaging::imaging::kurbo::Circle;
//! use dapple_imaging::imaging::peniko::Color;
//! use dapple_imaging::imaging::{Painter, record::Scene};
//! use dapple_imaging::rasterize;
//! use dapple_raster::Realization;
//!
//! // A disk of radius 0.25 at the corner of a unit tile: it wraps into all
//! // four corners.
//! let mut scene = Scene::new();
//! Painter::new(&mut scene).fill(Circle::new((0.0, 0.0), 0.25), Color::WHITE).draw();
//! let torus = Domain::periodic(1, 1).unwrap();
//! let mask = rasterize(&scene, Realization::period(torus, 64, 64).unwrap()).unwrap();
//! // Chords sit inside the curve, so the area comes out a little short.
//! let area: f32 = mask.values().iter().sum::<f32>() / (64.0 * 64.0);
//! assert!((area - core::f32::consts::PI / 16.0).abs() < 5e-3);
//! ```

#![no_std]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use dapple_field::hash::hash;
use dapple_field::program::Fingerprint;
use dapple_field::{DomainError, Edge, ImageLevel, SampleImage};
use dapple_raster::{Raster, RasterError, Realization};
use imaging::kurbo::Affine;
use imaging::record::Scene;
use imaging_vello_cpu::VelloCpuRenderer;

pub use imaging;

/// The largest realization side the renderer takes, in texels.
pub const MAX_SIZE: u32 = u16::MAX as u32;

/// Why a scene could not become a mask.
#[derive(Debug)]
#[non_exhaustive]
pub enum ImagingError {
    /// The realization is wider or taller than [`MAX_SIZE`].
    TooLarge {
        /// Texels per row.
        width: u32,
        /// Rows.
        height: u32,
    },
    /// The renderer refused the scene, for example an unbalanced clip or
    /// group stack, or an image brush.
    Render(imaging_vello_cpu::Error),
    /// The mask raster could not be built.
    Raster(RasterError),
    /// The mip chain could not be built into a sampled image.
    Image(DomainError),
}

impl fmt::Display for ImagingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge { width, height } => write!(
                f,
                "a {width}×{height} realization exceeds the renderer's {MAX_SIZE} texels per side"
            ),
            Self::Render(e) => write!(f, "rendering failed: {e}"),
            Self::Raster(e) => write!(f, "invalid mask raster: {e}"),
            Self::Image(e) => write!(f, "invalid mask image: {e}"),
        }
    }
}

impl core::error::Error for ImagingError {}

impl From<imaging_vello_cpu::Error> for ImagingError {
    fn from(e: imaging_vello_cpu::Error) -> Self {
        Self::Render(e)
    }
}

impl From<RasterError> for ImagingError {
    fn from(e: RasterError) -> Self {
        Self::Raster(e)
    }
}

impl From<DomainError> for ImagingError {
    fn from(e: DomainError) -> Self {
        Self::Image(e)
    }
}

/// Rasterizes `scene`, drawn in domain units, into a coverage mask over
/// `realization`.
///
/// Texel `(x, y)` covers the domain rectangle from `origin + (x, y) · texel`
/// to `origin + (x + 1, y + 1) · texel` and holds the fraction of it the
/// scene covers, in steps of 1/255. The raster has the realization's grid
/// and edge policy. A wrapping realization also draws the scene one period
/// away in every direction, so shapes crossing the tile's edge wrap.
///
/// # Errors
///
/// [`ImagingError::TooLarge`] beyond [`MAX_SIZE`] texels per side, and
/// [`ImagingError::Render`] when the renderer refuses the scene.
pub fn rasterize(scene: &Scene, realization: Realization) -> Result<Raster, ImagingError> {
    let (width, height) = (realization.width(), realization.height());
    let (Ok(w), Ok(h)) = (u16::try_from(width), u16::try_from(height)) else {
        return Err(ImagingError::TooLarge { width, height });
    };
    let (origin, texel) = (realization.origin(), realization.texel());
    let to_texels = Affine::scale_non_uniform(1.0 / f64::from(texel.x), 1.0 / f64::from(texel.y))
        * Affine::translate((-f64::from(origin.x), -f64::from(origin.y)));
    let mut framed = Scene::new();
    match realization.domain().period() {
        Some([px, py]) if realization.edge() == Edge::Wrap => {
            for j in -1..=1 {
                for i in -1..=1 {
                    let shift = (
                        f64::from(i * px.cast_signed()),
                        f64::from(j * py.cast_signed()),
                    );
                    framed.append_transformed(scene, to_texels * Affine::translate(shift));
                }
            }
        }
        _ => framed.append_transformed(scene, to_texels),
    }
    let image = VelloCpuRenderer::new(w, h).render_scene(&framed, w, h)?;
    let values = image
        .data
        .chunks_exact(4)
        .map(|rgba| f32::from(rgba[3]) / 255.0)
        .collect();
    Ok(Raster::from_values(
        width,
        height,
        origin,
        texel,
        realization.edge(),
        values,
    )?)
}

/// Rasterizes `scene` into a footprint-filtered image for `Op::Sample`.
///
/// Level 0 is [`rasterize`]'s mask. Each coarser level halves each side
/// (rounding down, to 1 × 1) and holds the exact area mean of level 0 over
/// its texels, so a coarse footprint reads the coverage of the area it
/// spans. The image's derivation is a fingerprint of level 0's texels and
/// grid.
///
/// # Errors
///
/// As [`rasterize`], plus [`ImagingError::Image`] if the levels do not form
/// a valid image.
pub fn coverage_image(
    scene: &Scene,
    realization: Realization,
) -> Result<SampleImage, ImagingError> {
    let mask = rasterize(scene, realization)?;
    let (width, height) = (mask.width(), mask.height());
    let mut levels = vec![ImageLevel::new(
        width,
        height,
        mask.texel(),
        mask.values().to_vec(),
    )?];
    let (mut w, mut h) = (width, height);
    while w > 1 || h > 1 {
        (w, h) = ((w / 2).max(1), (h / 2).max(1));
        let texel = realization.resized(w, h)?.texel();
        levels.push(ImageLevel::new(w, h, texel, area_mean(&mask, w, h))?);
    }
    Ok(SampleImage::new(
        realization.domain(),
        realization.origin(),
        levels,
        derivation(mask.digest()),
    )?)
}

/// `mask` resampled onto a `width` × `height` grid over the same region,
/// each texel the exact mean of the mask over its area.
fn area_mean(mask: &Raster, width: u32, height: u32) -> Vec<f32> {
    let columns = spans(mask.width(), width);
    let rows = spans(mask.height(), height);
    let mut values = Vec::with_capacity(columns.len() * rows.len());
    for row in &rows {
        for column in &columns {
            let mut sum = 0.0_f64;
            for &(y, wy) in row {
                for &(x, wx) in column {
                    sum += f64::from(mask.at(i64::from(x), i64::from(y))) * wx * wy;
                }
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a mean of values in [0, 1]"
            )]
            values.push(sum as f32);
        }
    }
    values
}

/// For each of `coarse` equal spans of `fine` texels, the fine texels it
/// overlaps and their weights, which sum to 1.
fn spans(fine: u32, coarse: u32) -> Vec<Vec<(u32, f64)>> {
    let scale = f64::from(fine) / f64::from(coarse);
    (0..coarse)
        .map(|c| {
            let (lo, hi) = (f64::from(c) * scale, f64::from(c + 1) * scale);
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "texel indices within the fine grid"
            )]
            let (first, last) = (libm::floor(lo) as u32, (libm::ceil(hi) as u32).min(fine));
            (first..last)
                .map(|t| {
                    let overlap = hi.min(f64::from(t + 1)) - lo.max(f64::from(t));
                    (t, overlap / scale)
                })
                .collect()
        })
        .collect()
}

/// The fingerprint of a mask image whose level 0 has this digest.
fn derivation(digest: u64) -> Fingerprint {
    let [lo, hi] = [0x636f_7665_7261_6730, 0x636f_7665_7261_6731] // "coverag0", "coverag1"
        .map(|seed| hash(seed, &[digest]));
    Fingerprint((u128::from(hi) << 64) | u128::from(lo))
}

#[cfg(test)]
mod tests;
