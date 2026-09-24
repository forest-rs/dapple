// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The material operations, each with its own contract.
//!
//! | Operation | Contract |
//! |---|---|
//! | [`select`] | One weight per texel, decided once, selects between two materials in every channel. Where the weight is fractional, parameters that do not mix linearly are approximated and reported. |
//! | [`apply_detail`] | Height or normal perturbation in a stated layer and frame, with an identity ([`Detail::identity`]) that leaves the material unchanged bit for bit. Reoriented normal mapping lives here and only here. |
//! | [`coat`] | Sets OpenPBR's coat, a dielectric layer over the base, and leaves the base parameters alone. A second coat over a first collapses into one and is reported. |
//! | [`deposit`] | A covering (dirt, salts, moss) that changes coverage, height, surface identity and optical properties together: it adds its thickness where it lies and is selected over the base, coat included, by its coverage. |
//! | [`transform`] | Moves every bound channel, parameters and auxiliary channels alike, and turns vector channels with the texels. |
//!
//! Every operation returns a [`Report`]: approximations are reported where
//! a richer combination collapses, not only when a packer lowers the
//! result. Dapple does not evaluate BSDFs; the reports say what a
//! renderer receives instead of what was meant.

use alloc::vec;
use alloc::vec::Vec;

use dapple_field::{NormalFrame, PortType, Value};
use dapple_raster::typed::TypedRaster;
use dapple_raster::{HeightToNormal, Raster, RasterOp};
use glam::{Vec2, Vec3};
use openpbr::Param;

use crate::material::{Aux, Channel, ChannelId, Grid, Material, MaterialError};
use crate::report::{ApproximationKind, Report};

/// How a selection weight becomes the per-texel decision.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Transition {
    /// The weight itself: 0 selects the first material, 1 the second.
    Mask,
    /// Height-based: the second material wins where it stands higher,
    /// with the weight biasing heights by up to `depth` meters either way
    /// (`(h₂ − h₁) + (2w − 1)·depth`) and the transition spread over
    /// `±contrast` meters. Weights of exactly 0 and 1 still select a whole
    /// material.
    Height {
        /// How far the weight moves the decision, in meters.
        depth: f32,
        /// Half the width of the transition, in meters; positive.
        contrast: f32,
    },
}

/// Selects between `a` (weight 0) and `b` (weight 1) with one decision per
/// texel shared by every channel.
///
/// Where the weight is fractional, colors, weights, heights and occlusion
/// mix linearly; coat parameters mix weighted by each side's coat weight,
/// so where only one side is coated its coat is kept exactly; roughness mixes in `α = r²`
/// ([`ApproximationKind::RoughnessInAlpha`]); normals and tangents average
/// and renormalize ([`ApproximationKind::NormalsAveraged`]); indices of
/// refraction interpolate ([`ApproximationKind::IorInterpolated`]);
/// differing coats mix as [`ApproximationKind::LayersMixed`]; and
/// identifiers take the larger weight's ([`ApproximationKind::WinnerLabel`]).
/// Each is reported only on texels where it made a difference.
///
/// # Errors
///
/// [`MaterialError::GridMismatch`] when the materials and the weight are
/// not on one grid, [`MaterialError::UniformConflict`] when they disagree
/// on a uniform parameter, [`MaterialError::InvalidParameter`] for a bad
/// transition.
pub fn select(
    a: &Material,
    b: &Material,
    weight: &Raster,
    transition: Transition,
) -> Result<(Material, Report), MaterialError> {
    let grid = a.grid();
    if b.grid() != grid || !grid.holds_raster(weight) {
        return Err(MaterialError::GridMismatch);
    }
    let w: Vec<f32> = match transition {
        Transition::Mask => weight.values().iter().map(|w| w.clamp(0.0, 1.0)).collect(),
        Transition::Height { depth, contrast } => {
            if !(depth.is_finite() && contrast.is_finite() && contrast > 0.0) {
                return Err(MaterialError::InvalidParameter("transition"));
            }
            let h = ChannelId::Aux(Aux::Height);
            weight
                .values()
                .iter()
                .enumerate()
                .map(|(i, &w)| {
                    let w = w.clamp(0.0, 1.0);
                    if w == 0.0 || w == 1.0 {
                        return w;
                    }
                    let (ha, hb) = (scalar(a.value(h, i)), scalar(b.value(h, i)));
                    let d = (hb - ha) + (2.0 * w - 1.0) * depth;
                    smoothstep(-contrast, contrast, d)
                })
                .collect()
        }
    };
    let mut report = Report::new("select");
    let m = mix(a, b, &w, &mut report)?;
    Ok((m, report))
}

fn scalar(v: Value) -> f32 {
    v.component(0).unwrap_or(0.0)
}

fn vec3(v: Value) -> Vec3 {
    match v {
        Value::Vector3(v) => v,
        other => Vec3::splat(scalar(other)),
    }
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// How a channel mixes under a fractional weight.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Rule {
    Linear,
    Alpha,
    Orientation,
    Ior,
    Winner,
}

fn rule(c: ChannelId) -> Rule {
    match c {
        ChannelId::Aux(Aux::Region | Aux::Surface) => Rule::Winner,
        ChannelId::Aux(_) => Rule::Linear,
        ChannelId::Param(p) => match p {
            Param::SpecularRoughness | Param::CoatRoughness | Param::FuzzRoughness => Rule::Alpha,
            Param::GeometryNormal
            | Param::GeometryCoatNormal
            | Param::GeometryTangent
            | Param::GeometryCoatTangent => Rule::Orientation,
            Param::SpecularIor | Param::CoatIor | Param::ThinFilmIor => Rule::Ior,
            _ => Rule::Linear,
        },
    }
}

fn all_channels() -> impl Iterator<Item = ChannelId> {
    Param::ALL
        .into_iter()
        .map(ChannelId::Param)
        .chain(Aux::ALL.into_iter().map(ChannelId::Aux))
}

/// Mixes `a` and `b` channel by channel under per-texel weights `w`.
fn mix(
    a: &Material,
    b: &Material,
    w: &[f32],
    report: &mut Report,
) -> Result<Material, MaterialError> {
    let grid = a.grid();
    let mut out = Material::new(grid);
    let partial = |i: usize| w[i] > 0.0 && w[i] < 1.0;
    report.partial += (0..w.len()).filter(|&i| partial(i)).count() as u64;
    let (lo, hi) = w
        .iter()
        .fold((f32::MAX, f32::MIN), |(l, h), &x| (l.min(x), h.max(x)));
    for c in all_channels() {
        let (ca, cb) = (a.channel(c), b.channel(c));
        if ca.is_none() && cb.is_none() {
            continue;
        }
        let (ka, kb) = (a.constant(c), b.constant(c));
        if let ChannelId::Param(p) = c
            && p.info().uniform
        {
            if ka != kb {
                return Err(MaterialError::UniformConflict(p));
            }
            out.set(
                c,
                Channel::Constant(ka.expect("uniform parameters are constant")),
            )?;
            continue;
        }
        if let Some(k) = ka
            && ka == kb
        {
            out.set(c, Channel::Constant(k))?;
            continue;
        }
        if hi <= 0.0 {
            if let Some(ch) = ca {
                out.set(c, ch.clone())?;
            }
            continue;
        }
        if lo >= 1.0 {
            if let Some(ch) = cb {
                out.set(c, ch.clone())?;
            }
            continue;
        }
        let r = rule(c);
        let coated = is_coat_param(c);
        let coat_weight = ChannelId::Param(Param::CoatWeight);
        let mut approximated = 0_u64;
        let values = (0..grid.len()).map(|i| {
            let (va, vb) = (a.value(c, i), b.value(c, i));
            let mut t = w[i];
            if coated && t > 0.0 && t < 1.0 {
                // A coat parameter matters only where there is coat: weight
                // each side by its coat's presence, so a side without coat
                // leaves the other's coat parameters exact.
                let (ca, cb) = (
                    scalar(a.value(coat_weight, i)),
                    scalar(b.value(coat_weight, i)),
                );
                let total = ca * (1.0 - t) + cb * t;
                if total > 0.0 {
                    t = cb * t / total;
                }
            }
            if t <= 0.0 {
                return va;
            }
            if t >= 1.0 {
                return vb;
            }
            if va != vb {
                approximated += u64::from(r != Rule::Linear);
            }
            match r {
                Rule::Linear | Rule::Ior => lerp(va, vb, t),
                Rule::Alpha => {
                    let (ra, rb) = (scalar(va), scalar(vb));
                    let alpha = ra * ra + (rb * rb - ra * ra) * t;
                    Value::Scalar(libm::sqrtf(alpha.max(0.0)))
                }
                Rule::Orientation => {
                    let (na, nb) = (vec3(va), vec3(vb));
                    Value::Vector3((na + (nb - na) * t).normalize_or(na))
                }
                Rule::Winner => {
                    if t >= 0.5 {
                        vb
                    } else {
                        va
                    }
                }
            }
        });
        let values: Vec<Value> = values.collect();
        out.set(c, Channel::Map(grid.typed(c.port(), values)?))?;
        let kind = match r {
            Rule::Linear => None,
            Rule::Alpha => Some(ApproximationKind::RoughnessInAlpha),
            Rule::Orientation => Some(ApproximationKind::NormalsAveraged),
            Rule::Ior => Some(ApproximationKind::IorInterpolated),
            Rule::Winner => Some(ApproximationKind::WinnerLabel),
        };
        if let Some(kind) = kind {
            report.add(c, kind, approximated);
        }
    }
    // Coats: a fractional mixture of two different layer stacks.
    let coat = |m: &Material, i: usize| {
        (
            scalar(m.value(ChannelId::Param(Param::CoatWeight), i)),
            m.value(ChannelId::Param(Param::CoatColor), i),
            m.value(ChannelId::Param(Param::CoatRoughness), i),
        )
    };
    let layered = (0..grid.len())
        .filter(|&i| {
            if !partial(i) {
                return false;
            }
            let ((wa, ca, ra), (wb, cb, rb)) = (coat(a, i), coat(b, i));
            wa != wb || (wa > 0.0 && (ca != cb || ra != rb))
        })
        .count() as u64;
    report.add(
        ChannelId::Param(Param::CoatWeight),
        ApproximationKind::LayersMixed,
        layered,
    );
    Ok(out)
}

/// Parameters of the coat layer, whose mixing is weighted by coat presence.
fn is_coat_param(c: ChannelId) -> bool {
    matches!(
        c,
        ChannelId::Param(
            Param::CoatColor
                | Param::CoatRoughness
                | Param::CoatRoughnessAnisotropy
                | Param::CoatIor
                | Param::CoatDarkening
                | Param::GeometryCoatNormal
                | Param::GeometryCoatTangent
        )
    )
}

fn lerp(a: Value, b: Value, t: f32) -> Value {
    dapple_field::scoped::zip(a, b, |x, y| x + (y - x) * t)
}

/// Which layer a [`Detail`] perturbs.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Layer {
    /// The base surface: its height (displacement) and `geometry_normal`.
    Base,
    /// The coat's surface only (orange peel, brush marks in a varnish):
    /// `geometry_coat_normal`, never the displacement.
    Coat,
    /// Both: displacement and both normals, for detail the coat follows.
    Both,
}

/// A geometric perturbation for [`apply_detail`].
///
/// Normals are unit vectors in the domain frame
/// ([`NormalFrame::Domain`]), `+Z` out of the surface; heights are meters.
#[derive(Clone, Debug, PartialEq)]
pub struct Detail {
    /// Height to add, in meters.
    pub height: Option<Raster>,
    /// A detail normal to compose onto the layer's normal.
    pub normal: Option<TypedRaster>,
    /// How strongly to apply it, per texel in `[0, 1]`; everywhere 1 when
    /// absent. Heights scale by it; normals tilt from `+Z` toward the
    /// detail by it and renormalize.
    pub strength: Option<Raster>,
    /// The layer perturbed.
    pub layer: Layer,
}

impl Detail {
    /// The identity: no height, no normal. Applying it returns the
    /// material unchanged.
    #[must_use]
    pub const fn identity(layer: Layer) -> Self {
        Self {
            height: None,
            normal: None,
            strength: None,
            layer,
        }
    }
}

/// Reoriented normal mapping (Barré-Brisebois and Hill, 2012): `detail`
/// rotated from `+Z` onto `base`, so both survive. `+Z` detail returns
/// `base` exactly.
#[must_use]
pub fn reoriented(base: Vec3, detail: Vec3) -> Vec3 {
    if detail == Vec3::Z {
        return base;
    }
    let t = base + Vec3::Z;
    let u = detail * Vec3::new(-1.0, -1.0, 1.0);
    (t * (t.dot(u) / t.z) - u).normalize_or(base)
}

/// Applies `detail` to `base` in its stated layer.
///
/// - [`Layer::Base`] and [`Layer::Both`] add the height (scaled by the
///   strength) to [`Aux::Height`]. When the base has a bound
///   `geometry_normal`, the added height's slope is composed onto it by
///   reoriented normal mapping, so normal and displacement stay in step.
/// - A detail normal is composed onto the layer's normal by reoriented
///   normal mapping. A base `geometry_normal` that is not bound is derived
///   from the base height when there is one
///   ([`ApproximationKind::NormalFromHeight`]), and is `+Z` otherwise.
/// - [`Layer::Coat`] turns a height into a coat normal instead of
///   displacement: the coat's surface ripples, the body does not.
///
/// # Errors
///
/// [`MaterialError::GridMismatch`] for detail off the grid, and
/// [`MaterialError::TypeMismatch`] for a detail normal that is not a
/// domain-frame normal.
pub fn apply_detail(base: &Material, detail: &Detail) -> Result<(Material, Report), MaterialError> {
    let grid = base.grid();
    let mut report = Report::new("detail");
    let mut out = base.clone();
    for r in [&detail.height, &detail.strength].into_iter().flatten() {
        if !grid.holds_raster(r) {
            return Err(MaterialError::GridMismatch);
        }
    }
    let normal_channel = ChannelId::Param(Param::GeometryNormal);
    if let Some(n) = &detail.normal {
        if !grid.holds(n) {
            return Err(MaterialError::GridMismatch);
        }
        if n.port() != PortType::Normal(NormalFrame::Domain) {
            return Err(MaterialError::TypeMismatch(normal_channel));
        }
    }
    let strength = |i: usize| detail.strength.as_ref().map_or(1.0, |s| s.values()[i]);
    if let Some(s) = &detail.strength {
        report.partial = s.values().iter().filter(|&&v| v > 0.0 && v < 1.0).count() as u64;
    }
    let scaled_height = detail.height.as_ref().map(|h| {
        h.values()
            .iter()
            .enumerate()
            .map(|(i, &v)| v * strength(i))
            .collect::<Vec<f32>>()
    });
    // The height's own slope, for layers whose normals must follow it.
    let height_normals = match &scaled_height {
        Some(v) => Some(HeightToNormal { scale: 1.0 }.apply(&grid.raster(v.clone())?)?),
        None => None,
    };
    let displaces = matches!(detail.layer, Layer::Base | Layer::Both);
    if displaces && let Some(dh) = &scaled_height {
        let h = ChannelId::Aux(Aux::Height);
        let values: Vec<Value> = (0..grid.len())
            .map(|i| Value::Scalar(scalar(base.value(h, i)) + dh[i]))
            .collect();
        out.set(h, Channel::Map(grid.typed(PortType::Scalar, values)?))?;
    }
    let detail_normal = |i: usize| -> Vec3 {
        let Some(n) = &detail.normal else {
            return Vec3::Z;
        };
        let s = strength(i);
        let n = vec3(n.value(i));
        if s >= 1.0 {
            n
        } else {
            (Vec3::Z + (n - Vec3::Z) * s).normalize_or(Vec3::Z)
        }
    };
    let targets: &[Param] = match detail.layer {
        Layer::Base => &[Param::GeometryNormal],
        Layer::Coat => &[Param::GeometryCoatNormal],
        Layer::Both => &[Param::GeometryNormal, Param::GeometryCoatNormal],
    };
    for &p in targets {
        let c = ChannelId::Param(p);
        let bound = base.param(p).is_some();
        // Height composes into a normal that is bound (so it stays in step
        // with the displacement) or into the coat's (which never displaces).
        let fold_height = height_normals.is_some() && (bound || p == Param::GeometryCoatNormal);
        if detail.normal.is_none() && !fold_height {
            continue;
        }
        let derived = if !bound && p == Param::GeometryNormal && base.aux(Aux::Height).is_some() {
            report.add(c, ApproximationKind::NormalFromHeight, grid.len() as u64);
            Some(HeightToNormal { scale: 1.0 }.apply(&out.height()?)?)
        } else {
            None
        };
        let values: Vec<Value> = (0..grid.len())
            .map(|i| {
                let mut n = match &derived {
                    Some(d) => Vec3::from_array(d.values()[i]),
                    None => vec3(base.value(c, i)),
                };
                if fold_height && derived.is_none() {
                    let hn = height_normals.as_ref().expect("checked").values()[i];
                    n = reoriented(n, Vec3::from_array(hn));
                }
                Value::Vector3(reoriented(n, detail_normal(i)))
            })
            .collect();
        out.set(
            c,
            Channel::Map(grid.typed(PortType::Normal(NormalFrame::Domain), values)?),
        )?;
    }
    Ok((out, report))
}

/// A dielectric coat for [`coat`], in OpenPBR's terms.
#[derive(Clone, Debug, PartialEq)]
pub struct Coating {
    /// `coat_weight`: the fraction covered, a mask.
    pub weight: Channel,
    /// `coat_color`: the coat's absorption tint, linear.
    pub color: Channel,
    /// `coat_roughness`.
    pub roughness: Channel,
    /// `coat_ior`.
    pub ior: Channel,
    /// `coat_darkening`.
    pub darkening: Channel,
    /// The coat's thickness in meters, added to the height where it lies
    /// (scaled by its weight); `None` for a coat too thin to displace.
    pub thickness: Option<Raster>,
}

impl Coating {
    /// A clear, smooth coat of refractive index 1.5 covering everything.
    #[must_use]
    pub fn clear() -> Self {
        Self {
            weight: Channel::Constant(Value::Scalar(1.0)),
            color: Channel::Constant(Value::Vector3(Vec3::ONE)),
            roughness: Channel::Constant(Value::Scalar(0.0)),
            ior: Channel::Constant(Value::Scalar(1.5)),
            darkening: Channel::Constant(Value::Scalar(1.0)),
            thickness: None,
        }
    }
}

/// Lays `coating` over `base`, leaving the base parameters alone: the base
/// stays the substrate, seen through the coat.
///
/// Over an uncoated base the coat parameters are set as given, which is
/// exact. Where the base is already coated, OpenPBR's one coat cannot hold
/// two, so they collapse ([`ApproximationKind::CoatsCollapsed`]): weights
/// combine as coverage `1 − (1 − w₁)(1 − w₂)`, the new tint multiplies the
/// old one where it lies, roughness adds lobe widths in `α²`
/// (`r⁴ = r₂⁴ + w₁ r₁⁴`), and the new coat's index and darkening win.
///
/// # Errors
///
/// [`MaterialError::GridMismatch`] or [`MaterialError::TypeMismatch`] for
/// a channel that does not fit.
pub fn coat(base: &Material, coating: &Coating) -> Result<(Material, Report), MaterialError> {
    let grid = base.grid();
    let mut report = Report::new("coat");
    let mut out = base.clone();
    let param = |p| ChannelId::Param(p);
    let mut scratch = Material::new(grid);
    for (p, ch) in [
        (Param::CoatWeight, &coating.weight),
        (Param::CoatColor, &coating.color),
        (Param::CoatRoughness, &coating.roughness),
        (Param::CoatIor, &coating.ior),
        (Param::CoatDarkening, &coating.darkening),
    ] {
        scratch.set(param(p), ch.clone())?;
    }
    if let Some(t) = &coating.thickness
        && !grid.holds_raster(t)
    {
        return Err(MaterialError::GridMismatch);
    }
    let weight = |i: usize| scalar(scratch.value(param(Param::CoatWeight), i));
    report.partial = (0..grid.len())
        .filter(|&i| {
            let w = weight(i);
            w > 0.0 && w < 1.0
        })
        .count() as u64;
    let uncoated = base.constant(param(Param::CoatWeight)) == Some(Value::Scalar(0.0));
    if uncoated {
        for p in [
            Param::CoatWeight,
            Param::CoatColor,
            Param::CoatRoughness,
            Param::CoatIor,
            Param::CoatDarkening,
        ] {
            out.set(
                param(p),
                scratch.channel(param(p)).expect("set above").clone(),
            )?;
        }
    } else {
        let n = grid.len();
        let mut collapsed = 0_u64;
        let (mut w, mut color, mut rough, mut ior, mut dark) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for i in 0..n {
            let wb = scalar(base.value(param(Param::CoatWeight), i));
            let wc = weight(i);
            let get = |m: &Material, p| m.value(param(p), i);
            if wb > 0.0 && wc > 0.0 {
                collapsed += 1;
            }
            if wc <= 0.0 {
                w.push(Value::Scalar(wb));
                color.push(get(base, Param::CoatColor));
                rough.push(get(base, Param::CoatRoughness));
                ior.push(get(base, Param::CoatIor));
                dark.push(get(base, Param::CoatDarkening));
                continue;
            }
            if wb <= 0.0 {
                w.push(Value::Scalar(wc));
                color.push(get(&scratch, Param::CoatColor));
                rough.push(get(&scratch, Param::CoatRoughness));
                ior.push(get(&scratch, Param::CoatIor));
                dark.push(get(&scratch, Param::CoatDarkening));
                continue;
            }
            w.push(Value::Scalar(1.0 - (1.0 - wb) * (1.0 - wc)));
            let old_tint = Vec3::ONE + (vec3(get(base, Param::CoatColor)) - Vec3::ONE) * wb;
            color.push(Value::Vector3(
                vec3(get(&scratch, Param::CoatColor)) * old_tint,
            ));
            let (rc, rb) = (
                scalar(get(&scratch, Param::CoatRoughness)),
                scalar(get(base, Param::CoatRoughness)),
            );
            let r4 = rc * rc * rc * rc + wb * rb * rb * rb * rb;
            rough.push(Value::Scalar(libm::sqrtf(libm::sqrtf(r4))));
            ior.push(get(&scratch, Param::CoatIor));
            dark.push(get(&scratch, Param::CoatDarkening));
        }
        for (p, values) in [
            (Param::CoatWeight, w),
            (Param::CoatColor, color),
            (Param::CoatRoughness, rough),
            (Param::CoatIor, ior),
            (Param::CoatDarkening, dark),
        ] {
            out.set(param(p), Channel::Map(grid.typed(param(p).port(), values)?))?;
        }
        report.add(
            param(Param::CoatWeight),
            ApproximationKind::CoatsCollapsed,
            collapsed,
        );
    }
    if let Some(t) = &coating.thickness {
        let h = ChannelId::Aux(Aux::Height);
        let values: Vec<Value> = (0..grid.len())
            .map(|i| Value::Scalar(scalar(base.value(h, i)) + t.values()[i] * weight(i)))
            .collect();
        out.set(h, Channel::Map(grid.typed(PortType::Scalar, values)?))?;
    }
    Ok((out, report))
}

/// A covering for [`deposit`]: its own material, where it lies, and how
/// thick it is.
#[derive(Clone, Debug, PartialEq)]
pub struct Deposit {
    /// The deposit's own material (dirt, salts): its parameters, and its
    /// [`Aux::Surface`] identity.
    pub material: Material,
    /// Where it lies, a mask in `[0, 1]`.
    pub coverage: Raster,
    /// Its thickness at full coverage, in meters.
    pub thickness: f32,
}

/// Covers `base` with `deposit`.
///
/// The deposit lies on top: the height rises by `coverage · thickness`;
/// the deposit's material is selected over the base by coverage, the coat
/// included, since what a deposit covers it also hides; the surface
/// identity becomes the deposit's where it covers at least half a texel.
/// Element and region ownership ([`Aux::Region`]) stays the base's: dirt on
/// a brick is still on that brick. Partial coverage reports as
/// [`select`] does, with [`ApproximationKind::LayersMixed`] wherever a
/// coated base is partly covered.
///
/// # Errors
///
/// As [`select`], and [`MaterialError::InvalidParameter`] for a thickness
/// that is not finite.
pub fn deposit(base: &Material, deposit: &Deposit) -> Result<(Material, Report), MaterialError> {
    let grid = base.grid();
    if deposit.material.grid() != grid || !grid.holds_raster(&deposit.coverage) {
        return Err(MaterialError::GridMismatch);
    }
    if !deposit.thickness.is_finite() {
        return Err(MaterialError::InvalidParameter("thickness"));
    }
    let mut on_top = deposit.material.clone();
    let h = ChannelId::Aux(Aux::Height);
    let raised: Vec<Value> = (0..grid.len())
        .map(|i| Value::Scalar(scalar(base.value(h, i)) + deposit.thickness))
        .collect();
    on_top.set(h, Channel::Map(grid.typed(PortType::Scalar, raised)?))?;
    match base.aux(Aux::Region) {
        Some(r) => on_top.set_aux(Aux::Region, r.clone())?,
        None => on_top.clear(ChannelId::Aux(Aux::Region)),
    }
    let w: Vec<f32> = deposit
        .coverage
        .values()
        .iter()
        .map(|c| c.clamp(0.0, 1.0))
        .collect();
    let mut report = Report::new("deposit");
    let m = mix(base, &on_top, &w, &mut report)?;
    Ok((m, report))
}

/// A material-wide coordinate transform: a mirror, quarter turns and an
/// offset, applied in that order to the material's content.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MaterialTransform {
    /// Mirror across the domain's y axis first (`x → −x`).
    pub flip_x: bool,
    /// Quarter turns counterclockwise, about the grid's center.
    pub quarter_turns: u8,
    /// Then move by this offset, in domain units.
    pub offset: Vec2,
}

impl MaterialTransform {
    /// A pure translation.
    #[must_use]
    pub const fn offset(offset: Vec2) -> Self {
        Self {
            flip_x: false,
            quarter_turns: 0,
            offset,
        }
    }
}

/// Moves every bound channel of `m` by `t`.
///
/// Texels move together in every channel, and vector channels turn with
/// them: normals and tangents rotate (and mirror) their x and y. Offsets of
/// whole texels and quarter turns are exact permutations; a fractional
/// offset resamples bilinearly (identifiers by nearest texel) and reports
/// [`ApproximationKind::Resampled`] for each channel it touched. Constant
/// channels stay constant, turned where they are vectors.
///
/// # Errors
///
/// [`MaterialError::InvalidParameter`] for an odd number of quarter turns
/// on a grid that is not square, or an offset that is not finite.
pub fn transform(m: &Material, t: MaterialTransform) -> Result<(Material, Report), MaterialError> {
    let grid = m.grid();
    let turns = t.quarter_turns % 4;
    if turns % 2 == 1 && grid.width != grid.height {
        return Err(MaterialError::InvalidParameter("quarter_turns"));
    }
    if !t.offset.is_finite() {
        return Err(MaterialError::InvalidParameter("offset"));
    }
    let mut report = Report::new("transform");
    let shift = t.offset / grid.texel;
    let whole = shift.round();
    let exact = (shift - whole).abs().max_element() < 1e-4;
    let (w, h) = (i64::from(grid.width), i64::from(grid.height));
    // The source position, in texels (centers at integers), of output
    // texel (x, y): undo the offset, the turns, then the mirror.
    let source = |x: i64, y: i64| -> Vec2 {
        #[expect(clippy::cast_precision_loss, reason = "texel indices are small")]
        let mut p = Vec2::new(x as f32, y as f32) - shift;
        #[expect(clippy::cast_precision_loss, reason = "grid sizes are small")]
        let c = Vec2::new((w - 1) as f32, (h - 1) as f32) * 0.5;
        for _ in 0..turns {
            // Undo one counterclockwise quarter turn about the center.
            let d = p - c;
            p = c + Vec2::new(d.y, -d.x);
        }
        if t.flip_x {
            p.x = 2.0 * c.x - p.x;
        }
        p
    };
    let turn_vector = |v: Vec3| -> Vec3 {
        let mut v = v;
        if t.flip_x {
            v.x = -v.x;
        }
        for _ in 0..turns {
            v = Vec3::new(-v.y, v.x, v.z);
        }
        v
    };
    let mut out = Material::new(grid);
    for (c, ch) in m.bound() {
        let port = c.port();
        let turns_vectors = matches!(
            c,
            ChannelId::Param(
                Param::GeometryNormal
                    | Param::GeometryCoatNormal
                    | Param::GeometryTangent
                    | Param::GeometryCoatTangent
            )
        );
        let fix = |v: Value| -> Value {
            if turns_vectors {
                Value::Vector3(turn_vector(vec3(v)))
            } else {
                v
            }
        };
        let moved = match ch {
            Channel::Constant(v) => Channel::Constant(fix(*v)),
            Channel::Map(r) => {
                let mut values = Vec::with_capacity(grid.len());
                for y in 0..h {
                    for x in 0..w {
                        let p = source(x, y);
                        let v = if exact || port == PortType::Id {
                            #[expect(
                                clippy::cast_possible_truncation,
                                reason = "positions are within a few grids of the origin"
                            )]
                            let (sx, sy) = (libm::roundf(p.x) as i64, libm::roundf(p.y) as i64);
                            r.value_at(sx, sy)
                        } else {
                            bilinear(r, p, turns_vectors)
                        };
                        values.push(fix(v));
                    }
                }
                if !exact {
                    report.add(c, ApproximationKind::Resampled, grid.len() as u64);
                }
                Channel::Map(grid.typed(r.port(), values)?)
            }
        };
        out.set(c, moved)?;
    }
    Ok((out, report))
}

/// `r` at texel position `p` (centers at integers), bilinear with the
/// raster's edge policy; unit vectors are renormalized.
fn bilinear(r: &TypedRaster, p: Vec2, unit: bool) -> Value {
    let (fx, fy) = (libm::floorf(p.x), libm::floorf(p.y));
    let (tx, ty) = (p.x - fx, p.y - fy);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "positions are within a few grids of the origin"
    )]
    let (x0, y0) = (fx as i64, fy as i64);
    let at = |x, y| r.value_at(x, y);
    let top = lerp(at(x0, y0), at(x0 + 1, y0), tx);
    let bottom = lerp(at(x0, y0 + 1), at(x0 + 1, y0 + 1), tx);
    let v = lerp(top, bottom, ty);
    match v {
        Value::Vector3(n) if unit => Value::Vector3(n.normalize_or(Vec3::Z)),
        other => other,
    }
}

/// A grid-sized raster of `value`, for masks and heights built by hand.
///
/// # Errors
///
/// Never for a valid grid.
pub fn filled(grid: Grid, value: f32) -> Result<Raster, MaterialError> {
    Ok(grid.raster(vec![value; grid.len()])?)
}
