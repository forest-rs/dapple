// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Rasters that keep their semantic type, and explicit reduction policies.
//!
//! A [`TypedRaster`] is a raster plus the [`PortType`] its values mean, with
//! storage to match: `u32` for identifiers, one channel for scalars and
//! masks, two for vectors and directions, three for vectors, colors and
//! normals. Realizing a typed program ([`realize_value`]) produces one.
//!
//! A type says which operations are meaningful; it does not say which
//! meaningful operation a consumer needs. So reducing a raster (building a
//! mip level) takes an explicit [`ReductionPolicy`], checked against the
//! type ([`ReductionPolicy::check`]) and part of the result's derivation
//! ([`ReductionPolicy::fingerprint`]). Each policy states what it retains
//! and what it loses:
//!
//! | Policy | Types | Retains | Loses |
//! |---|---|---|---|
//! | [`Average`](ReductionPolicy::Average) | scalars, masks, vectors, colors | the area mean | variation within a texel |
//! | [`ThresholdCoverage`](ReductionPolicy::ThresholdCoverage) | masks | the fraction of the footprint at or above a cutoff | the mean |
//! | [`Axial`](ReductionPolicy::Axial) | directions | the mean doubled-angle vector: the dominant axis and, in its length, the agreement | which way individual directions pointed |
//! | [`NormalMean`](ReductionPolicy::NormalMean) | normals | the unnormalized mean normal, whose length measures the spread | unit length; renormalize where a consumer asks |
//! | [`IdPoint`](ReductionPolicy::IdPoint) | identifiers | the level-0 identifier at the texel's center | every other identifier the texel covers |
//! | [`IdMode`](ReductionPolicy::IdMode) | identifiers | the most frequent level-0 identifier in the texel's footprint, ties to the smallest | every other identifier the texel covers |
//!
//! Every level of a chain is computed from **level 0** over the level's
//! footprint, never from the level above, so each policy's promise is stated
//! over the original texels: a repeated majority is not the majority of the
//! footprint, and repeated thresholds drift. Identifier reductions are lossy
//! summaries either way.

use alloc::vec::Vec;
use core::fmt;

use dapple_field::hash::hash;
use dapple_field::image::texel_coordinates;
use dapple_field::program::ValueProgram;
use dapple_field::{Domain, Edge, Footprint, PortType, SamplePolicy, Value};
use glam::{Vec2, Vec3};

use crate::{Raster, RasterError, Realization, check_size};

/// Texel storage for a [`TypedRaster`].
#[derive(Clone, Debug, PartialEq)]
pub enum Storage {
    /// One `f32` per texel: scalars and masks.
    F32(Raster<f32>),
    /// One `u32` per texel: identifiers.
    U32(Raster<u32>),
    /// Two `f32` per texel: 2D vectors and directions.
    F32x2(Raster<[f32; 2]>),
    /// Three `f32` per texel: 3D vectors, colors and normals.
    F32x3(Raster<[f32; 3]>),
}

impl Storage {
    /// The storage a port type is kept in.
    #[must_use]
    pub const fn kind_for(port: PortType) -> StorageKind {
        match port {
            PortType::Scalar | PortType::Mask => StorageKind::F32,
            PortType::Id => StorageKind::U32,
            PortType::Vector2 | PortType::Direction => StorageKind::F32x2,
            PortType::Vector3 | PortType::Color(_) | PortType::Normal(_) => StorageKind::F32x3,
        }
    }

    /// This storage's kind.
    #[must_use]
    pub const fn kind(&self) -> StorageKind {
        match self {
            Self::F32(_) => StorageKind::F32,
            Self::U32(_) => StorageKind::U32,
            Self::F32x2(_) => StorageKind::F32x2,
            Self::F32x3(_) => StorageKind::F32x3,
        }
    }

    fn shape(&self) -> (u32, u32, Vec2, Vec2, Edge) {
        macro_rules! shape {
            ($r:expr) => {
                ($r.width(), $r.height(), $r.origin(), $r.texel(), $r.edge())
            };
        }
        match self {
            Self::F32(r) => shape!(r),
            Self::U32(r) => shape!(r),
            Self::F32x2(r) => shape!(r),
            Self::F32x3(r) => shape!(r),
        }
    }
}

/// The storage kinds of [`Storage`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum StorageKind {
    /// [`Storage::F32`].
    F32,
    /// [`Storage::U32`].
    U32,
    /// [`Storage::F32x2`].
    F32x2,
    /// [`Storage::F32x3`].
    F32x3,
}

/// A raster that keeps the semantic type of its values.
#[derive(Clone, Debug, PartialEq)]
pub struct TypedRaster {
    port: PortType,
    storage: Storage,
}

impl TypedRaster {
    /// Pairs `storage` with the type its values mean.
    ///
    /// # Errors
    ///
    /// [`TypedError::StorageMismatch`] when `port` is not kept in
    /// `storage`'s kind.
    pub fn new(port: PortType, storage: Storage) -> Result<Self, TypedError> {
        if Storage::kind_for(port) != storage.kind() {
            return Err(TypedError::StorageMismatch {
                port,
                found: storage.kind(),
            });
        }
        Ok(Self { port, storage })
    }

    /// The type the values mean.
    #[must_use]
    pub const fn port(&self) -> PortType {
        self.port
    }

    /// The texels.
    #[must_use]
    pub const fn storage(&self) -> &Storage {
        &self.storage
    }

    /// Texels per row.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.storage.shape().0
    }

    /// Rows.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.storage.shape().1
    }

    /// Domain position of texel `(0, 0)`'s minimum corner.
    #[must_use]
    pub fn origin(&self) -> Vec2 {
        self.storage.shape().2
    }

    /// Texel size in domain units.
    #[must_use]
    pub fn texel(&self) -> Vec2 {
        self.storage.shape().3
    }

    /// The edge policy.
    #[must_use]
    pub fn edge(&self) -> Edge {
        self.storage.shape().4
    }

    /// The value at texel `(x, y)`, with the raster's edge policy.
    #[must_use]
    pub fn value_at(&self, x: i64, y: i64) -> Value {
        match &self.storage {
            Storage::F32(r) => Value::Scalar(r.at(x, y)),
            Storage::U32(r) => Value::Id(r.at(x, y)),
            Storage::F32x2(r) => Value::Vector2(Vec2::from_array(r.at(x, y))),
            Storage::F32x3(r) => Value::Vector3(Vec3::from_array(r.at(x, y))),
        }
    }

    /// A hash of the type and every texel's bits.
    #[must_use]
    pub fn digest(&self) -> u64 {
        let texels = match &self.storage {
            Storage::F32(r) => r.digest(),
            Storage::U32(r) => r.digest(),
            Storage::F32x2(r) => r.digest(),
            Storage::F32x3(r) => r.digest(),
        };
        hash(port_word(self.port), &[texels])
    }
}

/// A stable word for a port type, for fingerprints.
#[must_use]
pub fn port_word(port: PortType) -> u64 {
    use dapple_field::{NormalFrame, Primaries};
    match port {
        PortType::Scalar => 1,
        PortType::Mask => 2,
        PortType::Id => 3,
        PortType::Vector2 => 4,
        PortType::Vector3 => 5,
        PortType::Color(Primaries::Rec709) => 6,
        PortType::Normal(NormalFrame::Domain) => 7,
        PortType::Direction => 8,
    }
}

/// How a raster is reduced to a coarser level. See the [module
/// docs](self) for what each policy retains and loses.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "kind", rename_all = "snake_case"))]
pub enum ReductionPolicy {
    /// The area mean, per component.
    Average,
    /// For masks: the area mean, rescaled so the fraction of texels at or
    /// above `cutoff` equals the fraction of the level-0 footprint at or
    /// above it (Castaño 2010), for alpha-tested cutouts.
    ThresholdCoverage {
        /// The consumer's alpha cutoff, in `(0, 1)`.
        cutoff: f32,
    },
    /// For directions: the mean doubled-angle vector.
    Axial,
    /// For normals: the unnormalized mean, keeping the length that measures
    /// the normals' spread.
    NormalMean,
    /// For identifiers: the level-0 identifier at the texel's center.
    IdPoint,
    /// For identifiers: the most frequent level-0 identifier in the texel's
    /// footprint, ties to the smallest.
    IdMode,
}

impl ReductionPolicy {
    /// The policy used when a consumer names none.
    #[must_use]
    pub const fn default_for(port: PortType) -> Self {
        match port {
            PortType::Scalar
            | PortType::Mask
            | PortType::Vector2
            | PortType::Vector3
            | PortType::Color(_) => Self::Average,
            PortType::Direction => Self::Axial,
            PortType::Normal(_) => Self::NormalMean,
            PortType::Id => Self::IdMode,
        }
    }

    /// Whether this policy is meaningful for `port`.
    ///
    /// # Errors
    ///
    /// [`TypedError::PolicyRefused`] when it is not, for example averaging
    /// identifiers or thresholding a color; [`TypedError::InvalidCutoff`]
    /// for a threshold outside `(0, 1)`.
    pub fn check(self, port: PortType) -> Result<(), TypedError> {
        let permitted = match self {
            Self::Average => matches!(
                port,
                PortType::Scalar
                    | PortType::Mask
                    | PortType::Vector2
                    | PortType::Vector3
                    | PortType::Color(_)
            ),
            Self::ThresholdCoverage { cutoff } => {
                if !(cutoff > 0.0 && cutoff < 1.0) {
                    return Err(TypedError::InvalidCutoff);
                }
                port == PortType::Mask
            }
            Self::Axial => port == PortType::Direction,
            Self::NormalMean => matches!(port, PortType::Normal(_)),
            Self::IdPoint | Self::IdMode => port == PortType::Id,
        };
        if permitted {
            Ok(())
        } else {
            Err(TypedError::PolicyRefused { port, policy: self })
        }
    }

    /// Words identifying this policy, for derivation fingerprints.
    #[must_use]
    pub fn fingerprint(self) -> [u64; 2] {
        match self {
            Self::Average => [1, 0],
            Self::ThresholdCoverage { cutoff } => [2, u64::from(cutoff.to_bits())],
            Self::Axial => [3, 0],
            Self::NormalMean => [4, 0],
            Self::IdPoint => [5, 0],
            Self::IdMode => [6, 0],
        }
    }
}

/// A typed-raster failure.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TypedError {
    /// A port type paired with storage of another kind.
    StorageMismatch {
        /// The port type.
        port: PortType,
        /// The storage supplied.
        found: StorageKind,
    },
    /// A reduction policy that is not meaningful for the type.
    PolicyRefused {
        /// The raster's type.
        port: PortType,
        /// The refused policy.
        policy: ReductionPolicy,
    },
    /// A coverage threshold outside `(0, 1)`.
    InvalidCutoff,
    /// A raster-level failure.
    Raster(RasterError),
}

impl fmt::Display for TypedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StorageMismatch { port, found } => {
                write!(f, "{port:?} values cannot be stored as {found:?}")
            }
            Self::PolicyRefused { port, policy } => {
                write!(f, "{policy:?} is not a reduction for {port:?} values")
            }
            Self::InvalidCutoff => f.write_str("coverage cutoff must lie in (0, 1)"),
            Self::Raster(e) => write!(f, "raster: {e}"),
        }
    }
}

impl core::error::Error for TypedError {}

impl From<RasterError> for TypedError {
    fn from(e: RasterError) -> Self {
        Self::Raster(e)
    }
}

/// Evaluates a typed program at every texel center of `realization`, with
/// the same points and footprint as [`crate::realize`].
///
/// # Errors
///
/// As [`crate::realize`].
pub fn realize_value(
    program: &ValueProgram,
    realization: Realization,
) -> Result<TypedRaster, TypedError> {
    let domain = program.domain();
    if realization.edge == Edge::Wrap && domain != realization.domain {
        return Err(RasterError::DomainMismatch {
            expected: realization.domain,
            found: domain,
        }
        .into());
    }
    let texel = realization.texel();
    if !(texel.is_finite() && texel.x > 0.0 && texel.y > 0.0) {
        return Err(RasterError::InvalidRegion.into());
    }
    let footprint = Footprint::new(texel.max_element()).ok_or(RasterError::InvalidRegion)?;
    let count = check_size(realization.width, realization.height)?;
    let port = program.output_type();
    let point = |i: usize| {
        let (x, y) = (
            i % realization.width as usize,
            i / realization.width as usize,
        );
        #[expect(
            clippy::cast_precision_loss,
            reason = "realizations are far below f32's exact integer range"
        )]
        let center = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
        realization.region.origin + center * texel
    };
    let (w, h, origin, edge) = (
        realization.width,
        realization.height,
        realization.region.origin,
        realization.edge,
    );
    let eval = |i| program.eval(point(i), footprint);
    let storage = match Storage::kind_for(port) {
        StorageKind::F32 => Storage::F32(Raster::from_values(
            w,
            h,
            origin,
            texel,
            edge,
            (0..count)
                .map(|i| eval(i).scalar().unwrap_or(0.0))
                .collect(),
        )?),
        StorageKind::U32 => Storage::U32(Raster::from_values(
            w,
            h,
            origin,
            texel,
            edge,
            (0..count)
                .map(|i| match eval(i) {
                    Value::Id(id) => id,
                    _ => 0,
                })
                .collect(),
        )?),
        StorageKind::F32x2 => Storage::F32x2(Raster::from_values(
            w,
            h,
            origin,
            texel,
            edge,
            (0..count)
                .map(|i| match eval(i) {
                    Value::Vector2(v) => v.to_array(),
                    _ => [0.0; 2],
                })
                .collect(),
        )?),
        StorageKind::F32x3 => Storage::F32x3(Raster::from_values(
            w,
            h,
            origin,
            texel,
            edge,
            (0..count)
                .map(|i| match eval(i) {
                    Value::Vector3(v) => v.to_array(),
                    _ => [0.0; 3],
                })
                .collect(),
        )?),
    };
    TypedRaster::new(port, storage)
}

/// The size of mip level `level` of an `n`-texel axis: halved and rounded
/// up per level, at least one.
#[must_use]
pub const fn level_size(n: u32, level: u32) -> u32 {
    let mut n = n;
    let mut k = 0;
    while k < level {
        n = if n <= 1 { 1 } else { n.div_ceil(2) };
        k += 1;
    }
    n
}

/// Level `level` (≥ 1) of `base`'s chain under `policy`, computed from the
/// level-0 texels each level texel covers.
///
/// Level `k` texel `(x, y)` covers the level-0 texels
/// `[x·2ᵏ, (x+1)·2ᵏ) × [y·2ᵏ, (y+1)·2ᵏ)`, clipped to the raster.
///
/// # Errors
///
/// As [`ReductionPolicy::check`].
pub fn reduce(
    base: &TypedRaster,
    policy: ReductionPolicy,
    level: u32,
) -> Result<TypedRaster, TypedError> {
    policy.check(base.port)?;
    let (w0, h0) = (base.width(), base.height());
    let (w, h) = (level_size(w0, level), level_size(h0, level));
    let step = 1_u64 << level.min(31);
    #[expect(
        clippy::cast_precision_loss,
        reason = "mip sizes are far below f32's exact integer range"
    )]
    let texel = base.texel() * Vec2::new(w0 as f32 / w as f32, h0 as f32 / h as f32);
    let footprint = |x: u32, y: u32| {
        let x0 = u64::from(x) * step;
        let y0 = u64::from(y) * step;
        (
            x0..(x0 + step).min(u64::from(w0)),
            y0..(y0 + step).min(u64::from(h0)),
        )
    };
    let texels = (0..h).flat_map(|y| (0..w).map(move |x| (x, y)));
    let make = |storage| TypedRaster::new(base.port, storage);
    let (origin, edge) = (base.origin(), base.edge());
    let average = |fetch: &dyn Fn(usize) -> [f32; 3], n: usize| -> Vec<[f32; 3]> {
        texels
            .clone()
            .map(|(x, y)| {
                let (xs, ys) = footprint(x, y);
                let mut sum = [0.0_f64; 3];
                let mut count = 0_u32;
                for yy in ys {
                    for xx in xs.clone() {
                        let v = fetch(index(xx, yy, w0));
                        for c in 0..n {
                            sum[c] += f64::from(v[c]);
                        }
                        count += 1;
                    }
                }
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the mean of f32 values is narrowed back to f32"
                )]
                sum.map(|s| (s / f64::from(count)) as f32)
            })
            .collect()
    };
    match (&base.storage, policy) {
        (Storage::F32(r), ReductionPolicy::Average) => {
            let values = average(&|i| [r.values()[i], 0.0, 0.0], 1)
                .into_iter()
                .map(|v| v[0])
                .collect();
            make(Storage::F32(Raster::from_values(
                w, h, origin, texel, edge, values,
            )?))
        }
        (Storage::F32(r), ReductionPolicy::ThresholdCoverage { cutoff }) => {
            let mean: Vec<f32> = average(&|i| [r.values()[i], 0.0, 0.0], 1)
                .into_iter()
                .map(|v| v[0])
                .collect();
            let target = fraction_at_least(r.values(), cutoff);
            let scale = coverage_scale(&mean, cutoff, target);
            let values = mean.iter().map(|&v| (v * scale).clamp(0.0, 1.0)).collect();
            make(Storage::F32(Raster::from_values(
                w, h, origin, texel, edge, values,
            )?))
        }
        (Storage::F32x2(r), ReductionPolicy::Average | ReductionPolicy::Axial) => {
            let values = average(
                &|i| {
                    let [a, b] = r.values()[i];
                    [a, b, 0.0]
                },
                2,
            )
            .into_iter()
            .map(|[a, b, _]| [a, b])
            .collect();
            make(Storage::F32x2(Raster::from_values(
                w, h, origin, texel, edge, values,
            )?))
        }
        (Storage::F32x3(r), ReductionPolicy::Average | ReductionPolicy::NormalMean) => {
            let values = average(&|i| r.values()[i], 3);
            make(Storage::F32x3(Raster::from_values(
                w, h, origin, texel, edge, values,
            )?))
        }
        (Storage::U32(r), ReductionPolicy::IdPoint) => {
            let values = texels
                .map(|(x, y)| {
                    let (xs, ys) = footprint(x, y);
                    let cx = xs.start + (xs.end - xs.start - 1) / 2;
                    let cy = ys.start + (ys.end - ys.start - 1) / 2;
                    r.values()[index(cx, cy, w0)]
                })
                .collect();
            make(Storage::U32(Raster::from_values(
                w, h, origin, texel, edge, values,
            )?))
        }
        (Storage::U32(r), ReductionPolicy::IdMode) => {
            let mut ids: Vec<u32> = Vec::new();
            let values = texels
                .map(|(x, y)| {
                    let (xs, ys) = footprint(x, y);
                    ids.clear();
                    for yy in ys {
                        for xx in xs.clone() {
                            ids.push(r.values()[index(xx, yy, w0)]);
                        }
                    }
                    mode(&mut ids)
                })
                .collect();
            make(Storage::U32(Raster::from_values(
                w, h, origin, texel, edge, values,
            )?))
        }
        (_, policy) => Err(TypedError::PolicyRefused {
            port: base.port,
            policy,
        }),
    }
}

/// Levels 1 to `levels` of `base`'s chain under `policy`.
///
/// # Errors
///
/// As [`reduce`].
pub fn reduce_chain(
    base: &TypedRaster,
    policy: ReductionPolicy,
    levels: u32,
) -> Result<Vec<TypedRaster>, TypedError> {
    (1..=levels).map(|k| reduce(base, policy, k)).collect()
}

/// The row-major index of level-0 texel `(x, y)` in a `width`-wide raster.
#[expect(
    clippy::cast_possible_truncation,
    reason = "texel indices are bounded by a checked raster size"
)]
fn index(x: u64, y: u64, width: u32) -> usize {
    (y * u64::from(width) + x) as usize
}

/// The most frequent value, ties to the smallest. Sorts `ids`.
fn mode(ids: &mut [u32]) -> u32 {
    ids.sort_unstable();
    let (mut best, mut best_count) = (0, 0_usize);
    let mut i = 0;
    while i < ids.len() {
        let mut j = i;
        while j < ids.len() && ids[j] == ids[i] {
            j += 1;
        }
        if j - i > best_count {
            (best, best_count) = (ids[i], j - i);
        }
        i = j;
    }
    best
}

/// The fraction of `values` at or above `cutoff`.
fn fraction_at_least(values: &[f32], cutoff: f32) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "texel counts are far below f64's exact integer range"
    )]
    let fraction = values.iter().filter(|&&v| v >= cutoff).count() as f64 / values.len() as f64;
    fraction
}

/// The scale whose clamped, scaled `values` put `target` of them at or above
/// `cutoff`, by bisection with a fixed iteration count (deterministic).
fn coverage_scale(values: &[f32], cutoff: f32, target: f64) -> f32 {
    let (mut lo, mut hi) = (0.0_f32, 64.0_f32);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        let scaled: Vec<f32> = values.iter().map(|&v| (v * mid).clamp(0.0, 1.0)).collect();
        if fraction_at_least(&scaled, cutoff) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

/// Reads a typed chain back at domain point `p` with footprint `footprint`
/// (domain units) under `policy`, as a reference for
/// `dapple_field::SampleImage`: with [`SamplePolicy::Nearest`], the nearest
/// texel of the level nearest the footprint; with
/// [`SamplePolicy::Linear`], bilinear per component in the two levels the
/// footprint falls between, blended linearly, with no renormalization.
///
/// `levels[0]` is level 0. A periodic `domain` wraps `p` exactly into one
/// period first. Returns `None` for no levels or a policy the type does not
/// permit.
#[must_use]
pub fn sample(
    levels: &[TypedRaster],
    domain: Domain,
    policy: SamplePolicy,
    p: Vec2,
    footprint: f32,
) -> Option<Value> {
    let base = levels.first()?;
    if !policy.permits(base.port) {
        return None;
    }
    let texel0 = base.texel().max_element();
    let last = levels.len() - 1;
    let detail = if levels.len() > 1 && footprint > 0.0 && texel0 > 0.0 {
        let d = libm::log2f(footprint / texel0);
        if d > 0.0 { d } else { 0.0 }
    } else {
        0.0
    };
    let period = match domain {
        Domain::Periodic { period } => Some(period),
        _ => None,
    };
    let coordinates =
        |raster: &TypedRaster| texel_coordinates(p, raster.origin(), raster.texel(), period);
    if policy == SamplePolicy::Nearest {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a non-negative level of detail, clamped to the chain"
        )]
        let level = (libm::floorf(detail + 0.5) as usize).min(last);
        let raster = &levels[level];
        let t = coordinates(raster);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "texel coordinates are bounded by the raster size"
        )]
        return Some(raster.value_at(
            libm::floorf(t.x + 0.5) as i64,
            libm::floorf(t.y + 0.5) as i64,
        ));
    }
    let mix = |a: Value, b: Value, s: f32| match (a, b) {
        (Value::Scalar(a), Value::Scalar(b)) => Value::Scalar(a + (b - a) * s),
        (Value::Vector2(a), Value::Vector2(b)) => Value::Vector2(a + (b - a) * s),
        (Value::Vector3(a), Value::Vector3(b)) => Value::Vector3(a + (b - a) * s),
        (a, _) => a,
    };
    let bilinear = |raster: &TypedRaster| {
        let t = coordinates(raster);
        let (fx, fy) = (libm::floorf(t.x), libm::floorf(t.y));
        let f = Vec2::new(t.x - fx, t.y - fy);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "texel coordinates are bounded by the raster size"
        )]
        let (x, y) = (fx as i64, fy as i64);
        let top = mix(raster.value_at(x, y), raster.value_at(x + 1, y), f.x);
        let bottom = mix(
            raster.value_at(x, y + 1),
            raster.value_at(x + 1, y + 1),
            f.x,
        );
        mix(top, bottom, f.y)
    };
    #[expect(clippy::cast_precision_loss, reason = "level counts are tiny")]
    if detail >= last as f32 {
        return Some(bilinear(&levels[last]));
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "0 <= floor < last"
    )]
    let fine = libm::floorf(detail) as usize;
    let a = bilinear(&levels[fine]);
    let weight = detail - libm::floorf(detail);
    if weight == 0.0 {
        return Some(a);
    }
    Some(mix(a, bilinear(&levels[fine + 1]), weight))
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use dapple_field::program::{Op, ProgramBuilder};
    use dapple_field::{NormalFrame, Primaries};

    use super::*;

    fn typed<T: Copy>(
        port: PortType,
        w: u32,
        h: u32,
        values: Vec<T>,
        wrap: fn(Raster<T>) -> Storage,
    ) -> TypedRaster {
        let raster = Raster::from_values(w, h, Vec2::ZERO, Vec2::ONE, Edge::Wrap, values).unwrap();
        TypedRaster::new(port, wrap(raster)).unwrap()
    }

    #[test]
    fn storage_must_match_type() {
        let r = Raster::from_values(1, 1, Vec2::ZERO, Vec2::ONE, Edge::Wrap, vec![0.0]).unwrap();
        assert!(matches!(
            TypedRaster::new(PortType::Id, Storage::F32(r)),
            Err(TypedError::StorageMismatch { .. })
        ));
    }

    #[test]
    fn policies_are_checked_against_the_type() {
        use ReductionPolicy as P;
        let color = PortType::Color(Primaries::Rec709);
        let normal = PortType::Normal(NormalFrame::Domain);
        assert!(P::Average.check(PortType::Id).is_err());
        assert!(P::Average.check(normal).is_err());
        assert!(P::Average.check(PortType::Direction).is_err());
        assert!(P::ThresholdCoverage { cutoff: 0.5 }.check(color).is_err());
        assert!(
            P::ThresholdCoverage { cutoff: 0.5 }
                .check(PortType::Mask)
                .is_ok()
        );
        assert_eq!(
            P::ThresholdCoverage { cutoff: 1.5 }.check(PortType::Mask),
            Err(TypedError::InvalidCutoff)
        );
        assert!(P::Axial.check(PortType::Direction).is_ok());
        assert!(P::NormalMean.check(normal).is_ok());
        assert!(P::IdMode.check(PortType::Scalar).is_err());
        for port in [
            PortType::Scalar,
            PortType::Mask,
            PortType::Id,
            PortType::Vector2,
            PortType::Vector3,
            color,
            normal,
            PortType::Direction,
        ] {
            assert!(P::default_for(port).check(port).is_ok(), "{port:?}");
        }
        // Reducing with a refused policy fails rather than guessing.
        let ids = typed(PortType::Id, 2, 2, vec![1_u32, 2, 3, 4], Storage::U32);
        assert!(matches!(
            reduce(&ids, P::Average, 1),
            Err(TypedError::PolicyRefused { .. })
        ));
    }

    #[test]
    fn policies_are_part_of_the_fingerprint() {
        use ReductionPolicy as P;
        let all = [
            P::Average,
            P::ThresholdCoverage { cutoff: 0.5 },
            P::ThresholdCoverage { cutoff: 0.25 },
            P::Axial,
            P::NormalMean,
            P::IdPoint,
            P::IdMode,
        ];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a.fingerprint(), b.fingerprint(), "{a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn id_mode_uses_the_level_zero_footprint() {
        // The level-1 modes of the 2×2 blocks are 7, 9, 9 and 7 (a tie, to
        // the smallest), so reducing level 1 again would give 7; over the
        // 4×4 level-0 footprint, 9 holds ten texels to 7's five.
        #[rustfmt::skip]
        let values = vec![
            7, 7, 9, 9,
            7, 5, 9, 5,
            9, 9, 9, 9,
            9, 9, 7, 7,
        ];
        let ids = typed(PortType::Id, 4, 4, values, Storage::U32);
        let level1 = reduce(&ids, ReductionPolicy::IdMode, 1).unwrap();
        assert_eq!(level1.value_at(0, 0), Value::Id(7));
        let level2 = reduce(&ids, ReductionPolicy::IdMode, 2).unwrap();
        assert_eq!(level2.value_at(0, 0), Value::Id(9));
        // A tie resolves to the smallest identifier.
        let tie = typed(PortType::Id, 2, 1, vec![8_u32, 3], Storage::U32);
        let level1 = reduce(&tie, ReductionPolicy::IdMode, 1).unwrap();
        assert_eq!(level1.value_at(0, 0), Value::Id(3));
        // The point policy takes the footprint's center texel.
        let level1 = reduce(&ids, ReductionPolicy::IdPoint, 1).unwrap();
        assert_eq!(level1.value_at(0, 0), Value::Id(7));
        assert_eq!(level1.value_at(1, 0), Value::Id(9));
    }

    #[test]
    fn axial_reduction_ignores_sign_and_measures_agreement() {
        let dir = |theta: f32| [libm::cosf(2.0 * theta), libm::sinf(2.0 * theta)];
        let pi = core::f32::consts::PI;
        // θ and θ + π are the same axis.
        let same = typed(
            PortType::Direction,
            2,
            1,
            vec![dir(0.3), dir(0.3 + pi)],
            Storage::F32x2,
        );
        let Value::Vector2(v) = reduce(&same, ReductionPolicy::Axial, 1)
            .unwrap()
            .value_at(0, 0)
        else {
            panic!()
        };
        assert!((v.length() - 1.0).abs() < 1e-5);
        // Perpendicular axes cancel: no dominant direction.
        let crossed = typed(
            PortType::Direction,
            2,
            1,
            vec![dir(0.0), dir(pi / 2.0)],
            Storage::F32x2,
        );
        let Value::Vector2(v) = reduce(&crossed, ReductionPolicy::Axial, 1)
            .unwrap()
            .value_at(0, 0)
        else {
            panic!()
        };
        assert!(v.length() < 1e-5);
    }

    #[test]
    fn normal_mean_keeps_the_length_that_measures_spread() {
        let tilt = |s: f32| Vec3::new(s, 0.0, 1.0).normalize().to_array();
        let normals = typed(
            PortType::Normal(NormalFrame::Domain),
            2,
            2,
            vec![tilt(0.8), tilt(-0.8), tilt(-0.8), tilt(0.8)],
            Storage::F32x3,
        );
        let Value::Vector3(v) = reduce(&normals, ReductionPolicy::NormalMean, 1)
            .unwrap()
            .value_at(0, 0)
        else {
            panic!()
        };
        let expected = 1.0 / libm::sqrtf(1.64);
        assert!((v.length() - expected).abs() < 1e-5, "{v}");
        assert!(v.x.abs() < 1e-6);
    }

    #[test]
    fn threshold_coverage_preserves_level_zero_coverage() {
        // A thin cutout: 4 of 16 texels are opaque, spread so each 2×2 block
        // averages 0.25 and would vanish under a 0.5 cutoff.
        #[rustfmt::skip]
        let values = vec![
            1.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0_f32,
        ];
        let mask = typed(PortType::Mask, 4, 4, values, Storage::F32);
        let averaged = reduce(&mask, ReductionPolicy::Average, 1).unwrap();
        let Storage::F32(a) = averaged.storage() else {
            panic!()
        };
        assert!(a.values().iter().all(|&v| v < 0.5));
        let policy = ReductionPolicy::ThresholdCoverage { cutoff: 0.5 };
        let kept = reduce(&mask, policy, 1).unwrap();
        let Storage::F32(k) = kept.storage() else {
            panic!()
        };
        let coverage = k.values().iter().filter(|&&v| v >= 0.5).count();
        assert_eq!(coverage, 4, "{:?}", k.values());
    }

    #[test]
    fn levels_are_computed_from_level_zero() {
        let values: Vec<f32> = (0..64).map(|i| (i * 37 % 11) as f32).collect();
        let scalar = typed(PortType::Scalar, 8, 8, values.clone(), Storage::F32);
        let level2 = reduce(&scalar, ReductionPolicy::Average, 2).unwrap();
        assert_eq!((level2.width(), level2.height()), (2, 2));
        assert_eq!(level2.texel(), Vec2::splat(4.0));
        let direct: f32 = (0..4)
            .flat_map(|y| (0..4).map(move |x| (y, x)))
            .map(|(y, x)| values[y * 8 + x])
            .sum::<f32>()
            / 16.0;
        assert_eq!(level2.value_at(0, 0), Value::Scalar(direct));
        assert_eq!(
            reduce_chain(&scalar, ReductionPolicy::Average, 3)
                .unwrap()
                .len(),
            3
        );
    }

    #[test]
    fn typed_programs_realize_with_their_type() {
        let domain = Domain::periodic(1, 1).unwrap();
        let mut b = ProgramBuilder::new();
        let c = b.add(Op::Constant { domain, value: 0.6 }).unwrap();
        let id = b
            .add(Op::ToId {
                input: c,
                levels: 10,
            })
            .unwrap();
        let program = b.finish_value(id).unwrap();
        let realization = Realization::period(domain, 4, 4).unwrap();
        let raster = realize_value(&program, realization).unwrap();
        assert_eq!(raster.port(), PortType::Id);
        assert_eq!(raster.value_at(2, 3), Value::Id(6));
        let levels = [
            raster.clone(),
            reduce(&raster, ReductionPolicy::IdMode, 1).unwrap(),
        ];
        assert_eq!(
            sample(
                &levels,
                domain,
                SamplePolicy::Nearest,
                Vec2::new(0.4, 0.9),
                0.5
            ),
            Some(Value::Id(6))
        );
        assert_ne!(raster.digest(), 0);
    }
}
