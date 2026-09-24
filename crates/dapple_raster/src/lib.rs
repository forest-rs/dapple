// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Deterministic rasters for dapple: realization and neighborhood operations.
//!
//! A field has no resolution; a raster does. [`realize`] makes that step
//! explicit: a [`Realization`] names the region, the texel grid and the
//! [`Edge`] policy, and the field is evaluated at texel centers with a
//! one-texel footprint. Everything downstream sees that resolution and
//! nothing else. [`SampledField`] turns a raster back into a field.
//!
//! Raster operations implement [`RasterOp`]. Each states its parameters in
//! domain units (a blur of 0.01 units stays 0.01 units at any resolution) and
//! declares its kernel footprint in texels at a given texel size, so later
//! tiled realization can derive tile dependencies from it:
//!
//! - [`GaussianBlur`]: separable Gaussian with a physical standard deviation.
//! - [`HeightToNormal`]: surface normals from a height raster, in a declared
//!   frame.
//! - [`AmbientOcclusion`]: horizon-based occlusion from a height raster.
//! - [`DistanceTransform`]: exact Euclidean distance to the nearest feature
//!   texel (Felzenszwalb–Huttenlocher).
//! - [`Morphology`]: dilate, erode, open and close by an exact disk.
//! - [`Streak`]: a one-sided directional smear that fades over a physical
//!   length, for trails running from their sources.
//! - [`PercentileRemap`] and [`Histogram`] ([`shaping`]): measured value
//!   shaping, remapping by exact order statistics.
//! - Directional processing: [`SlopeSample`] (sampling along a height's fall
//!   lines), [`Displace`] (vector displacement) and [`Advect`] (carrying
//!   values along a flow's streamlines).
//! - [`synthesis::ByExample`]: Heitz and Neyret's histogram-preserving
//!   by-example blending of an exemplar.
//!
//! Each op states its [`OpCategory`] (local stencil, separable pass,
//! reduction, global transform or iterative solve), so schedulers know what
//! runs in parallel and what a tile depends on.
//!
//! **Edges.** A raster realized over one period of a periodic field wraps
//! ([`Edge::Wrap`]): every operation treats it as a torus, so its results tile
//! exactly and commute with rolling the raster. Any other raster clamps at its
//! border ([`Edge::Clamp`]).
//!
//! **Determinism.** Operations use plain `f32`/`f64` arithmetic in a fixed
//! order and `libm` for transcendentals, so equal inputs give equal bits on
//! every platform. [`Raster::digest`] fingerprints the exact bits.
//!
//! ```
//! use dapple_field::{Basis, Domain, Fractal, FractalParams};
//! use dapple_raster::{GaussianBlur, RasterOp, Realization, realize};
//! use glam::Vec2;
//!
//! let domain = Domain::periodic(1, 1).unwrap();
//! let fbm = Fractal::new(Basis::Gradient, domain, Vec2::splat(4.0), 1, FractalParams::default())?;
//! let height = realize(&fbm, Realization::period(domain, 64, 64)?)?;
//! let soft = GaussianBlur { sigma: 0.02 }.apply(&height)?;
//! assert_eq!(GaussianBlur { sigma: 0.02 }.footprint(height.texel()), Some([4, 4]));
//! assert_eq!(soft.width(), 64);
//! # Ok::<(), Box<dyn core::error::Error>>(())
//! ```

#![no_std]

extern crate alloc;

mod ao;
mod blur;
mod directional;
mod distance;
mod morphology;
mod normal;
pub mod shaping;
mod streak;
pub mod synthesis;
pub mod typed;

use alloc::vec::Vec;
use core::fmt;

use dapple_field::hash::hash;
use dapple_field::image::{bilinear, texel_coordinates};
use dapple_field::raster::Region;
use dapple_field::{Domain, Footprint, ScalarField};
use glam::Vec2;

pub use ao::AmbientOcclusion;
pub use blur::GaussianBlur;
pub use directional::{Advect, Displace, SlopeMode, SlopeSample};
pub use distance::DistanceTransform;
pub use morphology::{Morphology, MorphologyOp};
pub use normal::HeightToNormal;
pub use shaping::{Histogram, PercentileRemap, percentiles};
pub use streak::{Flow, Streak};

/// Largest texel count per raster, so indices and sizes stay exact.
pub const MAX_TEXELS: u64 = 1 << 28;

/// How a raster continues past its border: `dapple_field`'s [`Edge`], shared
/// with sampled images.
pub use dapple_field::Edge;

/// A rejected realization or raster operation.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum RasterError {
    /// Width or height is zero, or the texel count exceeds [`MAX_TEXELS`].
    InvalidSize {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
    },
    /// A region or texel size is non-finite or not positive.
    InvalidRegion,
    /// A wrapping realization needs a periodic domain.
    NotPeriodic,
    /// The field's domain differs from the realization's.
    DomainMismatch {
        /// The realization's domain.
        expected: Domain,
        /// The field's domain.
        found: Domain,
    },
    /// The value buffer length is not `width * height`.
    LengthMismatch {
        /// Expected length.
        expected: usize,
        /// Supplied length.
        found: usize,
    },
    /// An operation parameter is outside its documented range.
    InvalidParameter {
        /// Parameter name.
        name: &'static str,
    },
}

impl fmt::Display for RasterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSize { width, height } => {
                write!(f, "invalid raster size {width}×{height}")
            }
            Self::InvalidRegion => f.write_str("region or texel size must be finite and positive"),
            Self::NotPeriodic => f.write_str("wrapping realization needs a periodic domain"),
            Self::DomainMismatch { expected, found } => {
                write!(f, "field domain {found:?} differs from {expected:?}")
            }
            Self::LengthMismatch { expected, found } => {
                write!(f, "expected {expected} values, got {found}")
            }
            Self::InvalidParameter { name } => write!(f, "parameter {name} is out of range"),
        }
    }
}

impl core::error::Error for RasterError {}

fn check_size(width: u32, height: u32) -> Result<usize, RasterError> {
    let count = u64::from(width) * u64::from(height);
    if width == 0 || height == 0 || count > MAX_TEXELS {
        return Err(RasterError::InvalidSize { width, height });
    }
    Ok(usize::try_from(count).expect("MAX_TEXELS fits usize"))
}

/// A half-open rectangle of texels: columns `x0..x1`, rows `y0..y1`.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct TexelRect {
    /// First column.
    pub x0: u32,
    /// First row.
    pub y0: u32,
    /// One past the last column.
    pub x1: u32,
    /// One past the last row.
    pub y1: u32,
}

impl TexelRect {
    /// Every texel of a `width` × `height` grid.
    #[must_use]
    pub const fn full(width: u32, height: u32) -> Self {
        Self {
            x0: 0,
            y0: 0,
            x1: width,
            y1: height,
        }
    }

    /// Whether the rectangle holds no texels.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.x0 >= self.x1 || self.y0 >= self.y1
    }

    /// Number of texels.
    #[must_use]
    pub const fn area(&self) -> u64 {
        if self.is_empty() {
            0
        } else {
            (self.x1 - self.x0) as u64 * (self.y1 - self.y0) as u64
        }
    }

    /// The rectangle clipped to a `width` × `height` grid.
    #[must_use]
    pub fn clipped(self, width: u32, height: u32) -> Self {
        Self {
            x0: self.x0.min(width),
            y0: self.y0.min(height),
            x1: self.x1.min(width),
            y1: self.y1.min(height),
        }
    }
}

/// A row-major grid of texel values over a region of a domain.
///
/// Texel `(x, y)` covers `origin + texel * [x, x + 1) × [y, y + 1)`; its value
/// stands for that square, and its center is at `origin + texel * (x + ½, y + ½)`.
/// Row `y` increases with the domain's y axis.
#[derive(Clone, Debug, PartialEq)]
pub struct Raster<T = f32> {
    width: u32,
    height: u32,
    origin: Vec2,
    texel: Vec2,
    edge: Edge,
    values: Vec<T>,
}

impl<T: Copy> Raster<T> {
    /// Builds a raster from row-major `values`.
    pub fn from_values(
        width: u32,
        height: u32,
        origin: Vec2,
        texel: Vec2,
        edge: Edge,
        values: Vec<T>,
    ) -> Result<Self, RasterError> {
        let expected = check_size(width, height)?;
        if !(origin.is_finite() && texel.is_finite() && texel.x > 0.0 && texel.y > 0.0) {
            return Err(RasterError::InvalidRegion);
        }
        if values.len() != expected {
            return Err(RasterError::LengthMismatch {
                expected,
                found: values.len(),
            });
        }
        Ok(Self {
            width,
            height,
            origin,
            texel,
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

    /// Domain position of texel `(0, 0)`'s minimum corner.
    #[must_use]
    pub const fn origin(&self) -> Vec2 {
        self.origin
    }

    /// Texel size in domain units, per axis.
    #[must_use]
    pub const fn texel(&self) -> Vec2 {
        self.texel
    }

    /// The edge policy.
    #[must_use]
    pub const fn edge(&self) -> Edge {
        self.edge
    }

    /// The row-major values.
    #[must_use]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// The value at `(x, y)`, continued past the border by the edge policy.
    #[must_use]
    pub fn at(&self, x: i64, y: i64) -> T {
        let (x, y) = self.resolve(x, y);
        self.values[y * self.width as usize + x]
    }

    fn resolve(&self, x: i64, y: i64) -> (usize, usize) {
        let (w, h) = (i64::from(self.width), i64::from(self.height));
        let (x, y) = match self.edge {
            Edge::Wrap => (x.rem_euclid(w), y.rem_euclid(h)),
            Edge::Clamp => (x.clamp(0, w - 1), y.clamp(0, h - 1)),
        };
        (
            usize::try_from(x).expect("resolved index is in range"),
            usize::try_from(y).expect("resolved index is in range"),
        )
    }

    /// Whether `other` has the same size, placement, texel size and edge
    /// policy, whatever its values.
    #[must_use]
    pub fn same_grid<U>(&self, other: &Raster<U>) -> bool {
        self.width == other.width
            && self.height == other.height
            && self.origin == other.origin
            && self.texel == other.texel
            && self.edge == other.edge
    }

    fn check_rect(&self, rect: TexelRect) -> Result<(), RasterError> {
        if rect.x1 > self.width || rect.y1 > self.height {
            return Err(RasterError::InvalidSize {
                width: rect.x1,
                height: rect.y1,
            });
        }
        Ok(())
    }

    /// Recomputes the texels of `rect` by `f`, leaving the rest.
    pub(crate) fn map_rect(&mut self, rect: TexelRect, mut f: impl FnMut(i64, i64) -> T) {
        let w = self.width as usize;
        for y in rect.y0..rect.y1 {
            for x in rect.x0..rect.x1 {
                self.values[y as usize * w + x as usize] = f(i64::from(x), i64::from(y));
            }
        }
    }

    /// Copies the texels of `rect` from `source`, which must share this grid.
    ///
    /// # Errors
    ///
    /// [`RasterError::LengthMismatch`] when the grids differ, and
    /// [`RasterError::InvalidSize`] when `rect` leaves the grid.
    pub fn copy_rect(&mut self, source: &Self, rect: TexelRect) -> Result<(), RasterError> {
        if !self.same_grid(source) {
            return Err(RasterError::LengthMismatch {
                expected: self.values.len(),
                found: source.values.len(),
            });
        }
        self.check_rect(rect)?;
        self.map_rect(rect, |x, y| source.at(x, y));
        Ok(())
    }

    /// A raster with the same grid and `values` computed per texel by `f`.
    pub(crate) fn map_texels<U>(&self, mut f: impl FnMut(i64, i64) -> U) -> Raster<U> {
        let mut values = Vec::with_capacity(self.values.len());
        for y in 0..i64::from(self.height) {
            for x in 0..i64::from(self.width) {
                values.push(f(x, y));
            }
        }
        Raster {
            width: self.width,
            height: self.height,
            origin: self.origin,
            texel: self.texel,
            edge: self.edge,
            values,
        }
    }

    /// The raster shifted by `(dx, dy)` texels with wraparound: output texel
    /// `(x, y)` is input texel `(x - dx, y - dy)`. Origin and edge are kept.
    #[must_use]
    pub fn rolled(&self, dx: i64, dy: i64) -> Self {
        let wrap = Self {
            edge: Edge::Wrap,
            ..self.clone()
        };
        let mut out = wrap.map_texels(|x, y| wrap.at(x - dx, y - dy));
        out.edge = self.edge;
        out
    }
}

/// Values that [`Raster::digest`] can fingerprint.
pub trait DigestValue: Copy {
    /// Appends this value's exact bits as hash words.
    fn words(self, out: &mut impl FnMut(u64));
}

impl DigestValue for f32 {
    fn words(self, out: &mut impl FnMut(u64)) {
        out(u64::from(self.to_bits()));
    }
}

impl DigestValue for u32 {
    fn words(self, out: &mut impl FnMut(u64)) {
        out(u64::from(self));
    }
}

impl<const N: usize> DigestValue for [f32; N] {
    fn words(self, out: &mut impl FnMut(u64)) {
        for v in self {
            out(u64::from(v.to_bits()));
        }
    }
}

impl<T: DigestValue> Raster<T> {
    /// Whether the texels of `rect` have the same bits in `self` and
    /// `other`, which must share this grid.
    #[must_use]
    pub fn rect_bits_eq(&self, other: &Self, rect: TexelRect) -> bool {
        if !self.same_grid(other) || self.check_rect(rect).is_err() {
            return false;
        }
        let w = self.width as usize;
        (rect.y0..rect.y1).all(|y| {
            (rect.x0..rect.x1).all(|x| {
                let i = y as usize * w + x as usize;
                let words = |value: T| {
                    let (mut out, mut n) = ([0_u64; 8], 0);
                    value.words(&mut |v| {
                        out[n] = v;
                        n += 1;
                    });
                    (out, n)
                };
                words(self.values[i]) == words(other.values[i])
            })
        })
    }

    /// A 64-bit digest of the grid, edge policy and exact value bits, for
    /// golden tests.
    #[must_use]
    pub fn digest(&self) -> u64 {
        let mut h = hash(
            0x7261_7374_6572, // "raster"
            &[
                u64::from(self.width),
                u64::from(self.height),
                u64::from(self.origin.x.to_bits()),
                u64::from(self.origin.y.to_bits()),
                u64::from(self.texel.x.to_bits()),
                u64::from(self.texel.y.to_bits()),
                match self.edge {
                    Edge::Wrap => 0,
                    Edge::Clamp => 1,
                },
            ],
        );
        for value in &self.values {
            value.words(&mut |w| h = hash(h, &[w]));
        }
        h
    }
}

/// How a field is realized: region, grid and edge policy.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Realization {
    region: Region,
    width: u32,
    height: u32,
    edge: Edge,
    domain: Domain,
}

impl Realization {
    /// One full period of a periodic `domain`, wrapping at the edges.
    ///
    /// The field realized must have exactly this domain.
    pub fn period(domain: Domain, width: u32, height: u32) -> Result<Self, RasterError> {
        check_size(width, height)?;
        let region = Region::period(domain).ok_or(RasterError::NotPeriodic)?;
        Ok(Self {
            region,
            width,
            height,
            edge: Edge::Wrap,
            domain,
        })
    }

    /// An arbitrary region of any domain, clamping at the edges.
    pub fn region(region: Region, width: u32, height: u32) -> Result<Self, RasterError> {
        check_size(width, height)?;
        if !(region.origin.is_finite()
            && region.size.is_finite()
            && region.size.x > 0.0
            && region.size.y > 0.0)
        {
            return Err(RasterError::InvalidRegion);
        }
        Ok(Self {
            region,
            width,
            height,
            edge: Edge::Clamp,
            domain: Domain::Plane,
        })
    }

    /// The same region, domain and edge policy on a `width` × `height` grid,
    /// for example the next mip level.
    pub fn resized(&self, width: u32, height: u32) -> Result<Self, RasterError> {
        check_size(width, height)?;
        Ok(Self {
            width,
            height,
            ..*self
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

    /// Domain position of texel `(0, 0)`'s minimum corner.
    #[must_use]
    pub const fn origin(&self) -> Vec2 {
        self.region.origin
    }

    /// The edge policy: wrapping over one period, clamping otherwise.
    #[must_use]
    pub const fn edge(&self) -> Edge {
        self.edge
    }

    /// The domain realized: the periodic domain of a wrapping realization,
    /// [`Domain::Plane`] otherwise.
    #[must_use]
    pub const fn domain(&self) -> Domain {
        self.domain
    }

    /// Texel size in domain units.
    #[must_use]
    pub fn texel(&self) -> Vec2 {
        self.region.size / Vec2::new(self.width as f32, self.height as f32)
    }
}

/// Evaluates `field` at every texel center of `realization`.
///
/// Rows are evaluated with [`ScalarField::eval_batch`], so fields with
/// per-batch setup set up once per row.
/// Each texel is evaluated with a footprint of its larger side, so fields
/// band-limit to the raster's resolution. A wrapping realization requires the
/// field's domain to be the realization's periodic domain; a clamping one
/// accepts any field.
pub fn realize(field: &impl ScalarField, realization: Realization) -> Result<Raster, RasterError> {
    if realization.edge == Edge::Wrap && field.domain() != realization.domain {
        return Err(RasterError::DomainMismatch {
            expected: realization.domain,
            found: field.domain(),
        });
    }
    let texel = realization.texel();
    if !(texel.is_finite() && texel.x > 0.0 && texel.y > 0.0) {
        return Err(RasterError::InvalidRegion);
    }
    let footprint = Footprint::new(texel.max_element()).ok_or(RasterError::InvalidRegion)?;
    let count = check_size(realization.width, realization.height)?;
    let mut values = alloc::vec![0.0; count];
    let mut points = Vec::with_capacity(realization.width as usize);
    for (y, row) in values
        .chunks_exact_mut(realization.width as usize)
        .enumerate()
    {
        points.clear();
        for x in 0..realization.width {
            let center = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            points.push(realization.region.origin + center * texel);
        }
        field.eval_batch(&points, footprint, row);
    }
    Raster::from_values(
        realization.width,
        realization.height,
        realization.region.origin,
        texel,
        realization.edge,
        values,
    )
}

/// Evaluates `field` at the texel centers of `rect` in `realization`,
/// writing them into `output` and leaving its other texels.
///
/// Each texel gets exactly the value [`realize`] gives it, bit for bit, so a
/// raster refreshed rectangle by rectangle equals a full realization.
///
/// # Errors
///
/// As [`realize`], plus [`RasterError::LengthMismatch`] when `output` is not
/// on the realization's grid and [`RasterError::InvalidSize`] when `rect`
/// leaves it.
pub fn realize_into(
    field: &impl ScalarField,
    realization: Realization,
    rect: TexelRect,
    output: &mut Raster,
) -> Result<(), RasterError> {
    if realization.edge == Edge::Wrap && field.domain() != realization.domain {
        return Err(RasterError::DomainMismatch {
            expected: realization.domain,
            found: field.domain(),
        });
    }
    let texel = realization.texel();
    if !(texel.is_finite() && texel.x > 0.0 && texel.y > 0.0) {
        return Err(RasterError::InvalidRegion);
    }
    let footprint = Footprint::new(texel.max_element()).ok_or(RasterError::InvalidRegion)?;
    check_realization_grid(&realization, output)?;
    output.check_rect(rect)?;
    if rect.is_empty() {
        return Ok(());
    }
    let w = realization.width as usize;
    let span = (rect.x1 - rect.x0) as usize;
    let mut points = Vec::with_capacity(span);
    for y in rect.y0..rect.y1 {
        points.clear();
        for x in rect.x0..rect.x1 {
            let center = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            points.push(realization.region.origin + center * texel);
        }
        let start = y as usize * w + rect.x0 as usize;
        field.eval_batch(&points, footprint, &mut output.values[start..start + span]);
    }
    Ok(())
}

/// Unit normals of the height field `scale * field` at every texel center of
/// `realization`, from the field's gradient.
///
/// The frame is [`HeightToNormal`]'s: `+X` along the domain x axis, `+Y`
/// along the domain y axis, `+Z` out of the surface, and
/// `n = normalize(-scale ∂f/∂x, -scale ∂f/∂y, 1)`. Where the field's
/// derivatives are analytic ([`ScalarField::eval_gradient`]), the normals
/// carry no finite-difference error and no blur from a two-texel stencil,
/// and they do not depend on neighboring texels, so tiles and whole passes
/// agree trivially. The footprint is the realization's, as in [`realize`].
///
/// # Errors
///
/// As [`realize`], plus [`RasterError::InvalidParameter`] for a non-finite
/// `scale`.
pub fn realize_normals(
    field: &impl ScalarField,
    realization: Realization,
    scale: f32,
) -> Result<Raster<[f32; 3]>, RasterError> {
    let footprint = normals_setup(field, &realization, scale)?;
    let count = check_size(realization.width, realization.height)?;
    let mut values = Vec::with_capacity(count);
    for y in 0..realization.height {
        for x in 0..realization.width {
            values.push(normal_at(field, &realization, footprint, scale, x, y));
        }
    }
    Raster::from_values(
        realization.width,
        realization.height,
        realization.region.origin,
        realization.texel(),
        realization.edge,
        values,
    )
}

/// [`realize_normals`] for the texels of `rect` only, written into
/// `output` on the realization's grid; each texel equals the whole pass's.
///
/// # Errors
///
/// As [`realize_normals`], plus [`RasterError::LengthMismatch`] when
/// `output` is not on the realization's grid and
/// [`RasterError::InvalidSize`] when `rect` leaves it.
pub fn realize_normals_into(
    field: &impl ScalarField,
    realization: Realization,
    scale: f32,
    rect: TexelRect,
    output: &mut Raster<[f32; 3]>,
) -> Result<(), RasterError> {
    let footprint = normals_setup(field, &realization, scale)?;
    check_realization_grid(&realization, output)?;
    output.check_rect(rect)?;
    output.map_rect(rect, |x, y| {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "map_rect passes texel indices of the grid"
        )]
        let (x, y) = (x as u32, y as u32);
        normal_at(field, &realization, footprint, scale, x, y)
    });
    Ok(())
}

fn normals_setup(
    field: &impl ScalarField,
    realization: &Realization,
    scale: f32,
) -> Result<Footprint, RasterError> {
    if !scale.is_finite() {
        return Err(RasterError::InvalidParameter { name: "scale" });
    }
    if realization.edge == Edge::Wrap && field.domain() != realization.domain {
        return Err(RasterError::DomainMismatch {
            expected: realization.domain,
            found: field.domain(),
        });
    }
    let texel = realization.texel();
    if !(texel.is_finite() && texel.x > 0.0 && texel.y > 0.0) {
        return Err(RasterError::InvalidRegion);
    }
    Footprint::new(texel.max_element()).ok_or(RasterError::InvalidRegion)
}

fn normal_at(
    field: &impl ScalarField,
    realization: &Realization,
    footprint: Footprint,
    scale: f32,
    x: u32,
    y: u32,
) -> [f32; 3] {
    let center = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
    let p = realization.region.origin + center * realization.texel();
    let (_, g) = field.eval_gradient(p, footprint);
    glam::Vec3::new(-g.x * scale, -g.y * scale, 1.0)
        .normalize()
        .to_array()
}

/// Checks that `output` lies on `realization`'s grid.
fn check_realization_grid<T: Copy>(
    realization: &Realization,
    output: &Raster<T>,
) -> Result<(), RasterError> {
    let texel = realization.texel();
    if output.width != realization.width
        || output.height != realization.height
        || output.origin != realization.region.origin
        || output.texel != texel
        || output.edge != realization.edge
    {
        return Err(RasterError::LengthMismatch {
            expected: check_size(realization.width, realization.height)?,
            found: output.values.len(),
        });
    }
    Ok(())
}

/// How an operation reaches across a raster, for scheduling: what can run
/// in parallel, what a tile depends on, how work is reported and where it
/// can be cancelled.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum OpCategory {
    /// Each output texel reads a bounded neighborhood ([`RasterOp::footprint`]);
    /// tiles recompute locally and in parallel.
    LocalStencil,
    /// Passes along one axis at a time (separable filters, scans, distance
    /// transforms); lines are independent, and a line may reach across the
    /// whole raster.
    SeparablePass,
    /// Measures the whole raster into a small summary (a histogram,
    /// percentiles) in a fixed order.
    Reduction,
    /// A reduction followed by a pointwise map that uses it: every input
    /// texel may change every output texel.
    GlobalTransform,
    /// Repeats a pass until it converges or a bounded number of times.
    IterativeSolve,
}

/// An operation on a scalar raster.
pub trait RasterOp {
    /// The output value type.
    type Output: Copy;

    /// Texels read on each side of an output texel, per axis, at `texel`
    /// size; `None` when the whole raster may contribute.
    fn footprint(&self, texel: Vec2) -> Option<[u32; 2]>;

    /// How the operation reaches across the raster. The default is a
    /// [`OpCategory::LocalStencil`] for operations with a footprint and a
    /// [`OpCategory::GlobalTransform`] otherwise.
    fn category(&self) -> OpCategory {
        if self.footprint(Vec2::ONE).is_some() {
            OpCategory::LocalStencil
        } else {
            OpCategory::GlobalTransform
        }
    }

    /// Applies the operation.
    fn apply(&self, input: &Raster) -> Result<Raster<Self::Output>, RasterError>;

    /// Recomputes the texels of `rect` in `output`, a previous result on
    /// `input`'s grid, leaving its other texels.
    ///
    /// Each recomputed texel equals [`RasterOp::apply`]'s, bit for bit. The
    /// default applies the whole operation and copies `rect`; local
    /// operations override it to read only `rect` grown by
    /// [`RasterOp::footprint`].
    ///
    /// # Errors
    ///
    /// As [`RasterOp::apply`], plus [`RasterError::LengthMismatch`] when
    /// `output` is not on `input`'s grid and [`RasterError::InvalidSize`] when
    /// `rect` leaves it.
    fn apply_into(
        &self,
        input: &Raster,
        rect: TexelRect,
        output: &mut Raster<Self::Output>,
    ) -> Result<(), RasterError> {
        let full = self.apply(input)?;
        output.copy_rect(&full, rect)
    }
}

/// Checks that `output` shares `input`'s grid and holds `rect`.
pub(crate) fn check_into<T: Copy>(
    input: &Raster,
    rect: TexelRect,
    output: &Raster<T>,
) -> Result<(), RasterError> {
    if !input.same_grid(output) {
        return Err(RasterError::LengthMismatch {
            expected: input.values.len(),
            found: output.values.len(),
        });
    }
    output.check_rect(rect)
}

/// A scalar raster sampled back into a field with bilinear filtering.
///
/// A wrapping raster over one period is a periodic field with that period;
/// any other raster is a plane field that clamps outside its region. The
/// footprint is ignored: detail is already limited to the raster's
/// resolution.
#[derive(Clone, Debug)]
pub struct SampledField {
    raster: Raster,
    domain: Domain,
}

impl SampledField {
    /// Wraps `raster`. `domain` is used for wrapping rasters and must be the
    /// periodic domain whose single period the raster covers.
    pub fn new(raster: Raster, domain: Domain) -> Result<Self, RasterError> {
        let domain = match raster.edge {
            Edge::Clamp => Domain::Plane,
            Edge::Wrap => {
                let [px, py] = domain.period().ok_or(RasterError::NotPeriodic)?;
                let covered = raster.texel * Vec2::new(raster.width as f32, raster.height as f32);
                if covered != Vec2::new(px as f32, py as f32) || raster.origin != Vec2::ZERO {
                    return Err(RasterError::InvalidRegion);
                }
                domain
            }
        };
        Ok(Self { raster, domain })
    }

    /// The sampled raster.
    #[must_use]
    pub fn raster(&self) -> &Raster {
        &self.raster
    }
}

impl ScalarField for SampledField {
    fn domain(&self) -> Domain {
        self.domain
    }

    /// The bilinear sample at `p`, whatever the footprint: one raster level
    /// is not band-limited. Sample through a mip chain with
    /// `dapple_field::SampleImage` to filter by footprint.
    fn eval(&self, p: Vec2, _footprint: Footprint) -> f32 {
        let t = texel_coordinates(
            p,
            self.raster.origin,
            self.raster.texel,
            self.domain.period(),
        );
        bilinear(
            &self.raster.values,
            [self.raster.width, self.raster.height],
            self.raster.edge,
            t,
        )
    }
}

#[cfg(test)]
mod golden_tests;
#[cfg(test)]
mod rect_tests;
