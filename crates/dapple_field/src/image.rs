// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sampled images: texels read back as a field.
//!
//! Realization turns a field into texels; sampling turns texels back into a
//! field. [`SampleImage`] holds a raster's texels, optionally with its mip
//! chain, and evaluates them bilinearly at any point of its domain. Field
//! programs sample one through [`Op::Sample`](crate::program::Op::Sample).
//!
//! **Coordinates.** Texel centers sit at `origin + (i + 0.5) · texel`. On a
//! periodic domain a point is first reduced into the period exactly, in
//! `f64`, so every repeat of a point samples identical bits. Past the border,
//! texels continue by the image's [`Edge`] policy.
//!
//! **Footprints.** A single-level image is not band-limited: it samples level
//! 0 whatever the footprint. With mips, a footprint `w` wide reads the level
//! whose texel matches it, at level-of-detail `log2(w / texel₀)`, blending the
//! two nearest levels linearly (trilinear filtering). Footprints finer than
//! level 0 read level 0; coarser than the last level read the last level.
//!
//! **Types and policies.** An image keeps the [`PortType`] of its texels:
//! one channel for scalars and masks, two for 2D vectors and directions
//! (as doubled-angle vectors), three for 3D vectors, colors and normals, and
//! `u32` identifiers. How texels are interpolated is an explicit
//! [`SamplePolicy`], checked against the type when the image is built:
//!
//! | Policy | Types | Retains | Loses |
//! |---|---|---|---|
//! | [`Linear`](SamplePolicy::Linear) | all but identifiers | per component, the texels' (and levels') weighted mean: a mask stays in `[0, 1]`, a direction's doubled-angle vector stays sign-free, a normal keeps the mean length that measures its spread (it is never renormalized) | the texels' individual values |
//! | [`Nearest`](SamplePolicy::Nearest) | all | the nearest texel of the level nearest the footprint, exactly | continuity across texel borders |
//!
//! Identifiers only ever sample nearest: blending two cell labels names
//! neither cell.
//!
//! The same sampling serves `dapple_raster`'s `SampledField`, through
//! [`bilinear`] and [`texel_coordinates`].

use alloc::sync::Arc;
use alloc::vec::Vec;

use glam::Vec2;

use crate::domain::{Domain, DomainError, Footprint};
use crate::program::{Fingerprint, StaticBounds};
use crate::types::{PortType, Value};

/// How a [`SampleImage`] interpolates its texels; see the [module
/// docs](self) for what each policy retains and loses.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum SamplePolicy {
    /// Bilinear within a level and linear between the two levels a
    /// footprint falls between, per component, never renormalized.
    Linear,
    /// The nearest texel of the level nearest the footprint.
    Nearest,
}

impl SamplePolicy {
    /// The policy used for `port` when a consumer has no reason to choose:
    /// [`Nearest`](Self::Nearest) for identifiers, [`Linear`](Self::Linear)
    /// otherwise.
    #[must_use]
    pub const fn default_for(port: PortType) -> Self {
        match port {
            PortType::Id => Self::Nearest,
            _ => Self::Linear,
        }
    }

    /// Whether this policy is meaningful for `port` values: everything but
    /// linear interpolation of identifiers.
    #[must_use]
    pub const fn permits(self, port: PortType) -> bool {
        !matches!((self, port), (Self::Linear, PortType::Id))
    }

    /// A word identifying the policy, for fingerprints.
    #[must_use]
    pub const fn word(self) -> u64 {
        match self {
            Self::Linear => 0,
            Self::Nearest => 1,
        }
    }
}

/// How texels continue past their border.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Edge {
    /// The texels are one period of a torus: coordinates wrap.
    Wrap,
    /// Coordinates clamp to the nearest border texel.
    Clamp,
}

/// Continuous texel coordinates of `p` (texel centers at integers) on a
/// grid starting at `origin` with texels `texel` wide.
///
/// With a `period`, `p` is reduced into `[0, period)` exactly in `f64` first,
/// so every repeat of a point gives identical coordinates.
#[must_use]
pub fn texel_coordinates(p: Vec2, origin: Vec2, texel: Vec2, period: Option<[u32; 2]>) -> Vec2 {
    let local = |v: f32, origin: f32, texel: f32, period: Option<u32>| {
        let mut v = f64::from(v) - f64::from(origin);
        if let Some(period) = period {
            // `fmod` is exact, so every repeat reduces to the same value.
            let period = f64::from(period);
            v = libm::fmod(v, period);
            if v < 0.0 {
                v += period;
            }
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "texel coordinates are bounded by the image size"
        )]
        let t = (v / f64::from(texel) - 0.5) as f32;
        t
    };
    Vec2::new(
        local(p.x, origin.x, texel.x, period.map(|p| p[0])),
        local(p.y, origin.y, texel.y, period.map(|p| p[1])),
    )
}

/// The row-major index of texel `(x, y)` of a `width` × `height` grid,
/// continued past the border by `edge`.
fn texel_index(width: u32, height: u32, edge: Edge, x: i64, y: i64) -> usize {
    let (w, h) = (i64::from(width), i64::from(height));
    let (x, y) = match edge {
        Edge::Wrap => (x.rem_euclid(w), y.rem_euclid(h)),
        Edge::Clamp => (x.clamp(0, w - 1), y.clamp(0, h - 1)),
    };
    usize::try_from(y * w + x).expect("resolved index is in range")
}

/// The texel at `(x, y)` of a row-major `width` × `height` grid, continued
/// past the border by `edge`.
fn at(values: &[f32], width: u32, height: u32, edge: Edge, x: i64, y: i64) -> f32 {
    values[texel_index(width, height, edge, x, y)]
}

/// The four texels around `t` and the fractional offsets.
fn corners(values: &[f32], size: [u32; 2], edge: Edge, t: Vec2) -> ([f32; 4], Vec2) {
    let (fx, fy) = (libm::floorf(t.x), libm::floorf(t.y));
    #[expect(
        clippy::cast_possible_truncation,
        reason = "floored coordinates are bounded by the image size plus one"
    )]
    let (x, y) = (fx as i64, fy as i64);
    let [w, h] = size;
    let texel = |dx: i64, dy: i64| at(values, w, h, edge, x + dx, y + dy);
    (
        [texel(0, 0), texel(1, 0), texel(0, 1), texel(1, 1)],
        Vec2::new(t.x - fx, t.y - fy),
    )
}

/// The bilinear sample of a row-major `size[0]` × `size[1]` grid at
/// continuous texel coordinates `t` (texel centers at integers), continued
/// past the border by `edge`.
///
/// # Panics
///
/// Panics if `values` is shorter than `size[0] · size[1]` or either size is
/// zero.
#[must_use]
pub fn bilinear(values: &[f32], size: [u32; 2], edge: Edge, t: Vec2) -> f32 {
    let ([a, b, c, d], f) = corners(values, size, edge, t);
    let top = a + (b - a) * f.x;
    let bottom = c + (d - c) * f.x;
    top + (bottom - top) * f.y
}

/// [`bilinear`]'s value and its gradient with respect to `t`, in value units
/// per texel.
#[must_use]
pub fn bilinear_gradient(values: &[f32], size: [u32; 2], edge: Edge, t: Vec2) -> (f32, Vec2) {
    let ([a, b, c, d], f) = corners(values, size, edge, t);
    let top = a + (b - a) * f.x;
    let bottom = c + (d - c) * f.x;
    let value = top + (bottom - top) * f.y;
    let dx = (b - a) + ((d - c) - (b - a)) * f.y;
    (value, Vec2::new(dx, bottom - top))
}

/// The texels of an [`ImageLevel`].
#[derive(Clone, Debug)]
enum Texels {
    /// `channels` interleaved `f32` components per texel.
    F32 { channels: u8, values: Arc<[f32]> },
    /// One identifier per texel.
    Id(Arc<[u32]>),
}

/// One mip level of a [`SampleImage`].
#[derive(Clone, Debug)]
pub struct ImageLevel {
    width: u32,
    height: u32,
    texel: Vec2,
    texels: Texels,
}

impl ImageLevel {
    /// A `width` × `height` level of row-major scalar `values`, with texels
    /// `texel` domain units wide.
    pub fn new(
        width: u32,
        height: u32,
        texel: Vec2,
        values: impl Into<Arc<[f32]>>,
    ) -> Result<Self, DomainError> {
        Self::with_channels(width, height, texel, 1, values)
    }

    /// A level of `channels` (1 to 3) interleaved components per texel, row
    /// major: 2D vectors and directions have two, 3D vectors, colors and
    /// normals three.
    pub fn with_channels(
        width: u32,
        height: u32,
        texel: Vec2,
        channels: u8,
        values: impl Into<Arc<[f32]>>,
    ) -> Result<Self, DomainError> {
        let values = values.into();
        if !(1..=3).contains(&channels) {
            return Err(DomainError::InvalidParameter { name: "channels" });
        }
        let texels = u64::from(width) * u64::from(height);
        if values.len() as u64 != texels * u64::from(channels) {
            return Err(DomainError::InvalidParameter {
                name: "image level",
            });
        }
        check_level(width, height, texel, texels)?;
        Ok(Self {
            width,
            height,
            texel,
            texels: Texels::F32 { channels, values },
        })
    }

    /// A level of row-major identifiers.
    pub fn ids(
        width: u32,
        height: u32,
        texel: Vec2,
        ids: impl Into<Arc<[u32]>>,
    ) -> Result<Self, DomainError> {
        let ids = ids.into();
        check_level(width, height, texel, ids.len() as u64)?;
        Ok(Self {
            width,
            height,
            texel,
            texels: Texels::Id(ids),
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

    /// A texel's size in domain units.
    #[must_use]
    pub const fn texel(&self) -> Vec2 {
        self.texel
    }

    /// Components per texel: 1 for scalars and identifiers.
    #[must_use]
    pub const fn channels(&self) -> u8 {
        match self.texels {
            Texels::F32 { channels, .. } => channels,
            Texels::Id(_) => 1,
        }
    }

    /// The row-major texels, components interleaved; empty for an
    /// identifier level.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        match &self.texels {
            Texels::F32 { values, .. } => values,
            Texels::Id(_) => &[],
        }
    }

    /// The row-major identifiers; empty for a level of numbers.
    #[must_use]
    pub fn id_values(&self) -> &[u32] {
        match &self.texels {
            Texels::Id(ids) => ids,
            Texels::F32 { .. } => &[],
        }
    }

    /// Whether the level holds `port` values.
    fn holds(&self, port: PortType) -> bool {
        match self.texels {
            Texels::Id(_) => port == PortType::Id,
            Texels::F32 { channels, .. } => {
                port != PortType::Id && usize::from(channels) == port.components()
            }
        }
    }

    /// The texel at `(x, y)` as a `port` value, continued by `edge`.
    fn value_at(&self, port: PortType, edge: Edge, x: i64, y: i64) -> Value {
        let i = texel_index(self.width, self.height, edge, x, y);
        match &self.texels {
            Texels::Id(ids) => Value::Id(ids[i]),
            Texels::F32 { channels, values } => {
                let n = usize::from(*channels);
                let mut v = [0.0; 3];
                v[..n].copy_from_slice(&values[i * n..(i + 1) * n]);
                to_value(port, v)
            }
        }
    }

    /// The bilinear sample at continuous texel coordinates `t`, per
    /// component.
    fn bilinear_components(&self, edge: Edge, t: Vec2) -> [f32; 3] {
        let Texels::F32 { channels, values } = &self.texels else {
            return [0.0; 3];
        };
        let n = usize::from(*channels);
        let (fx, fy) = (libm::floorf(t.x), libm::floorf(t.y));
        #[expect(
            clippy::cast_possible_truncation,
            reason = "floored coordinates are bounded by the image size plus one"
        )]
        let (x, y) = (fx as i64, fy as i64);
        let f = Vec2::new(t.x - fx, t.y - fy);
        let index = |dx: i64, dy: i64| texel_index(self.width, self.height, edge, x + dx, y + dy);
        let [a, b, c, d] = [index(0, 0), index(1, 0), index(0, 1), index(1, 1)];
        let mut out = [0.0; 3];
        for (k, o) in out.iter_mut().enumerate().take(n) {
            let (a, b, c, d) = (
                values[a * n + k],
                values[b * n + k],
                values[c * n + k],
                values[d * n + k],
            );
            let top = a + (b - a) * f.x;
            let bottom = c + (d - c) * f.x;
            *o = top + (bottom - top) * f.y;
        }
        out
    }

    fn same_shape(&self, other: &Self) -> bool {
        (self.width, self.height) == (other.width, other.height)
            && self.channels() == other.channels()
            && matches!(self.texels, Texels::Id(_)) == matches!(other.texels, Texels::Id(_))
            && self.texel.to_array().map(f32::to_bits) == other.texel.to_array().map(f32::to_bits)
    }
}

/// Checks a level's size, texel and texel count.
fn check_level(width: u32, height: u32, texel: Vec2, count: u64) -> Result<(), DomainError> {
    if width == 0
        || height == 0
        || count != u64::from(width) * u64::from(height)
        || !(texel.is_finite() && texel.x > 0.0 && texel.y > 0.0)
    {
        return Err(DomainError::InvalidParameter {
            name: "image level",
        });
    }
    Ok(())
}

/// The first `port.components()` of `v` as a `port` value.
fn to_value(port: PortType, v: [f32; 3]) -> Value {
    match port.components() {
        1 => Value::Scalar(v[0]),
        2 => Value::Vector2(Vec2::new(v[0], v[1])),
        _ => Value::Vector3(glam::Vec3::from_array(v)),
    }
}

/// Sampled texels, with an optional mip chain, as a field over a domain.
///
/// An image's identity is its [`derivation`](Self::derivation): a
/// fingerprint of how the texels were produced, supplied by whoever built
/// them (a material graph uses its raster nodes' fingerprints). Equality and
/// program fingerprints compare derivations and shapes, never texels, so two
/// images with one derivation must hold the same texels of the same type.
///
/// An image keeps its texels' [`PortType`] and samples them under an
/// explicit [`SamplePolicy`]; see the [module docs](self).
#[derive(Clone, Debug)]
pub struct SampleImage {
    port: PortType,
    policy: SamplePolicy,
    domain: Domain,
    origin: Vec2,
    edge: Edge,
    levels: Arc<[ImageLevel]>,
    derivation: Fingerprint,
    bounds: StaticBounds,
}

impl PartialEq for SampleImage {
    fn eq(&self, other: &Self) -> bool {
        self.derivation == other.derivation
            && self.port == other.port
            && self.policy == other.policy
            && self.domain == other.domain
            && self.edge == other.edge
            && self.origin.to_array().map(f32::to_bits) == other.origin.to_array().map(f32::to_bits)
            && self.levels.len() == other.levels.len()
            && self
                .levels
                .iter()
                .zip(other.levels.iter())
                .all(|(a, b)| a.same_shape(b))
    }
}

impl SampleImage {
    /// A scalar image of `levels`, finest first, derived as `derivation`,
    /// sampled [`Linear`](SamplePolicy::Linear)ly; see [`Self::typed`].
    ///
    /// # Errors
    ///
    /// As [`Self::typed`].
    pub fn new(
        domain: Domain,
        origin: Vec2,
        levels: Vec<ImageLevel>,
        derivation: Fingerprint,
    ) -> Result<Self, DomainError> {
        Self::typed(
            PortType::Scalar,
            SamplePolicy::Linear,
            domain,
            origin,
            levels,
            derivation,
        )
    }

    /// An image of `port` values in `levels`, finest first, derived as
    /// `derivation` and sampled under `policy`.
    ///
    /// A [`Domain::Periodic`] image wraps: level 0 must cover exactly one
    /// period from the origin. A [`Domain::Plane`] image clamps, from
    /// `origin`. Every level must cover level 0's extent to within one part
    /// in 10⁵, each coarser than the last.
    ///
    /// # Errors
    ///
    /// [`DomainError::InvalidParameter`] for a policy that `port` does not
    /// permit (named `"sample policy"`), levels that do not hold `port`
    /// values, no levels, a non-finite origin, a periodic image not covering
    /// exactly one period, or levels that do not cover the same extent with
    /// growing texels.
    pub fn typed(
        port: PortType,
        policy: SamplePolicy,
        domain: Domain,
        origin: Vec2,
        levels: Vec<ImageLevel>,
        derivation: Fingerprint,
    ) -> Result<Self, DomainError> {
        if !policy.permits(port) {
            return Err(DomainError::InvalidParameter {
                name: "sample policy",
            });
        }
        let invalid = DomainError::InvalidParameter { name: "image" };
        let base = levels.first().ok_or(invalid)?;
        if !origin.is_finite() || levels.iter().any(|level| !level.holds(port)) {
            return Err(invalid);
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "image sizes are far below f32's exact integer range"
        )]
        let extent =
            |level: &ImageLevel| Vec2::new(level.width as f32, level.height as f32) * level.texel;
        let covered = extent(base);
        let edge = match domain.period() {
            Some([px, py]) => {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "periods are below 2^24, exact in f32"
                )]
                let period = Vec2::new(px as f32, py as f32);
                if covered != period || origin != Vec2::ZERO {
                    return Err(invalid);
                }
                Edge::Wrap
            }
            None => Edge::Clamp,
        };
        for pair in levels.windows(2) {
            let (fine, coarse) = (&pair[0], &pair[1]);
            let drift = (extent(coarse) - covered).abs().max_element();
            if drift > covered.max_element() * 1e-5
                || coarse.texel.x < fine.texel.x
                || coarse.texel.y < fine.texel.y
            {
                return Err(invalid);
            }
        }
        let bounds = if port.is_scalar() {
            let bounds = image_bounds(&levels, edge);
            match policy {
                SamplePolicy::Linear => bounds,
                // Nearest texels jump at texel borders.
                SamplePolicy::Nearest => StaticBounds {
                    slope: None,
                    ..bounds
                },
            }
        } else {
            StaticBounds::default()
        };
        Ok(Self {
            port,
            policy,
            domain,
            origin,
            edge,
            levels: levels.into(),
            derivation,
            bounds,
        })
    }

    /// The type of the texels' values.
    #[must_use]
    pub const fn port(&self) -> PortType {
        self.port
    }

    /// How the texels are interpolated.
    #[must_use]
    pub const fn policy(&self) -> SamplePolicy {
        self.policy
    }

    /// The domain the image is a field over.
    #[must_use]
    pub const fn domain(&self) -> Domain {
        self.domain
    }

    /// Where level 0's first texel begins.
    #[must_use]
    pub const fn origin(&self) -> Vec2 {
        self.origin
    }

    /// The edge policy: wrapping on a periodic domain, clamping otherwise.
    #[must_use]
    pub const fn edge(&self) -> Edge {
        self.edge
    }

    /// The levels, finest first.
    #[must_use]
    pub fn levels(&self) -> &[ImageLevel] {
        &self.levels
    }

    /// The fingerprint of how the texels were produced.
    #[must_use]
    pub const fn derivation(&self) -> Fingerprint {
        self.derivation
    }

    /// Bounds on the image's values and slope at every point and footprint,
    /// computed once from the texels of every level.
    ///
    /// Bilinear sampling stays within the texels' range, and along each axis
    /// changes no faster than the largest step between neighboring texels
    /// (wrapping across the period on a periodic image) over a texel's
    /// width. Blending two mip levels by footprint mixes their values with
    /// weights constant in the point, so the bounds over all levels hold.
    /// Any non-finite texel leaves both bounds unknown.
    #[must_use]
    pub const fn static_bounds(&self) -> StaticBounds {
        self.bounds
    }

    /// The footprint's level of detail, `log2(width / texel₀)`, or `None`
    /// when it reads level 0 alone.
    fn detail(&self, footprint: Footprint) -> Option<f32> {
        let width = footprint.width();
        if self.levels.len() == 1 || width <= 0.0 {
            return None;
        }
        let detail = libm::log2f(width / self.levels[0].texel.max_element());
        (!detail.is_nan() && detail > 0.0).then_some(detail)
    }

    /// The levels a footprint reads and the weight of the coarser one.
    fn levels_for(&self, footprint: Footprint) -> (usize, usize, f32) {
        let last = self.levels.len() - 1;
        let Some(detail) = self.detail(footprint) else {
            return (0, 0, 0.0);
        };
        #[expect(clippy::cast_precision_loss, reason = "level counts are tiny")]
        if detail >= last as f32 {
            return (last, last, 0.0);
        }
        let floor = libm::floorf(detail);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "0 <= floor < last"
        )]
        let fine = floor as usize;
        (fine, fine + 1, detail - floor)
    }

    /// The level nearest a footprint: its level of detail rounded, clamped
    /// to the chain.
    fn nearest_level(&self, footprint: Footprint) -> usize {
        let last = self.levels.len() - 1;
        self.detail(footprint).map_or(0, |detail| {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a positive level of detail, clamped to the chain"
            )]
            let level = libm::floorf(detail + 0.5) as usize;
            level.min(last)
        })
    }

    /// The continuous texel coordinates of `p` in `level`.
    fn coordinates(&self, level: &ImageLevel, p: Vec2) -> Vec2 {
        texel_coordinates(p, self.origin, level.texel, self.domain.period())
    }

    fn level_sample(&self, level: usize, p: Vec2) -> (f32, Vec2) {
        let level = &self.levels[level];
        let t = self.coordinates(level, p);
        let (value, per_texel) =
            bilinear_gradient(level.values(), [level.width, level.height], self.edge, t);
        (value, per_texel / level.texel)
    }

    /// The nearest texel of the level nearest `footprint`.
    fn nearest(&self, p: Vec2, footprint: Footprint) -> Value {
        let level = &self.levels[self.nearest_level(footprint)];
        let t = self.coordinates(level, p);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "texel coordinates are bounded by the image size"
        )]
        let (x, y) = (
            libm::floorf(t.x + 0.5) as i64,
            libm::floorf(t.y + 0.5) as i64,
        );
        level.value_at(self.port, self.edge, x, y)
    }

    /// The image's value at `p`, filtered for `footprint`, under the
    /// image's [policy](Self::policy).
    #[must_use]
    pub fn sample_value(&self, p: Vec2, footprint: Footprint) -> Value {
        match self.policy {
            SamplePolicy::Nearest => self.nearest(p, footprint),
            SamplePolicy::Linear if self.port.is_scalar() => {
                Value::Scalar(self.linear_scalar(p, footprint))
            }
            SamplePolicy::Linear => {
                let (fine, coarse, weight) = self.levels_for(footprint);
                let at = |level: usize| {
                    let level = &self.levels[level];
                    level.bilinear_components(self.edge, self.coordinates(level, p))
                };
                let mut v = at(fine);
                if fine != coarse {
                    let b = at(coarse);
                    for (a, b) in v.iter_mut().zip(b) {
                        *a += (b - *a) * weight;
                    }
                }
                to_value(self.port, v)
            }
        }
    }

    fn linear_scalar(&self, p: Vec2, footprint: Footprint) -> f32 {
        let (fine, coarse, weight) = self.levels_for(footprint);
        let at = |level: usize| {
            let level = &self.levels[level];
            bilinear(
                level.values(),
                [level.width, level.height],
                self.edge,
                self.coordinates(level, p),
            )
        };
        let a = at(fine);
        if fine == coarse {
            return a;
        }
        let b = at(coarse);
        a + (b - a) * weight
    }

    /// The value at `p` of a scalar or mask image, filtered for `footprint`
    /// ([`Self::sample_value`]). Other types give their first component
    /// (an identifier as a float).
    #[must_use]
    pub fn sample(&self, p: Vec2, footprint: Footprint) -> f32 {
        if self.policy == SamplePolicy::Linear && self.port.is_scalar() {
            return self.linear_scalar(p, footprint);
        }
        self.sample_value(p, footprint).component(0).unwrap_or(0.0)
    }

    /// [`Self::sample`]'s value and its gradient in value units per domain
    /// unit: exact for the bilinear patch around `p`, blended like the value
    /// between levels. Nearest sampling is flat between texel borders, so
    /// its gradient is zero, as it is for non-scalar images.
    #[must_use]
    pub fn sample_gradient(&self, p: Vec2, footprint: Footprint) -> (f32, Vec2) {
        if self.policy != SamplePolicy::Linear || !self.port.is_scalar() {
            return (self.sample(p, footprint), Vec2::ZERO);
        }
        let (fine, coarse, weight) = self.levels_for(footprint);
        let (a, ga) = self.level_sample(fine, p);
        if fine == coarse {
            return (a, ga);
        }
        let (b, gb) = self.level_sample(coarse, p);
        (a + (b - a) * weight, ga + (gb - ga) * weight)
    }
}

/// [`SampleImage::static_bounds`] of `levels`.
fn image_bounds(levels: &[ImageLevel], edge: Edge) -> StaticBounds {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    let mut slope = 0.0_f64;
    for level in levels {
        let (w, h) = (level.width as usize, level.height as usize);
        let v = level.values();
        if v.iter().any(|x| !x.is_finite()) {
            return StaticBounds::default();
        }
        for &x in v {
            lo = lo.min(x);
            hi = hi.max(x);
        }
        let wrap = matches!(edge, Edge::Wrap);
        let mut step = [0.0_f64; 2];
        for y in 0..h {
            for x in 0..w {
                let here = f64::from(v[y * w + x]);
                let right = if x + 1 < w {
                    Some(x + 1)
                } else if wrap {
                    Some(0)
                } else {
                    None
                };
                let below = if y + 1 < h {
                    Some(y + 1)
                } else if wrap {
                    Some(0)
                } else {
                    None
                };
                if let Some(r) = right {
                    step[0] = step[0].max((f64::from(v[y * w + r]) - here).abs());
                }
                if let Some(b) = below {
                    step[1] = step[1].max((f64::from(v[b * w + x]) - here).abs());
                }
            }
        }
        let texel = level.texel();
        slope = slope.max(step[0] / f64::from(texel.x) + step[1] / f64::from(texel.y));
    }
    #[expect(clippy::cast_possible_truncation, reason = "rounded up after widening")]
    let slope = (slope * (1.0 + 1e-5)) as f32;
    StaticBounds {
        range: Some([lo, hi]),
        slope: Some(slope.next_up()).filter(|s| s.is_finite()),
    }
}
