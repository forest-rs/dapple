// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Curve networks: joints, cracks, veins and grooves as polylines with arc
//! length and width.
//!
//! A [`Curve`] is a polyline, open or closed, with a width at each vertex
//! interpolated along its arc length. A [`CurveNetwork`] holds curves over
//! a domain and answers, for any point, the nearest curve with the point's
//! coordinates in the curve's frame ([`CurveSample`]): distance, arc length
//! along it, signed offset across it (positive to the left of its
//! direction), its tangent and its width there. The network is exposed:
//!
//! - **as fields** ([`CurveNetwork::field`]): distance, along, across, or a
//!   stroke of the width profile with its edge antialiased over the
//!   footprint, each a `dapple_field::ScalarField` to realize or bake;
//! - **as element layouts** ([`CurveNetwork::stitches`]): elements every
//!   `spacing` domain units along each curve, turned to its tangent and
//!   keyed by curve and stitch index: stitches along a seam, studs along a
//!   groove, fibers along a vein.
//!
//! Every quantity is in domain units, so spacing and widths never depend on
//! a realization's resolution. On a periodic domain curves repeat with the
//! period: the nearest point looks at the neighboring repeats too.
//! [`CurveNetwork::intersections`] lists where curves cross.
//!
//! Evaluation visits every segment of every curve (in the nine nearest
//! repeats on a periodic domain); networks are expected to be modest.

use alloc::vec::Vec;
use core::fmt;

use dapple_field::{Domain, Footprint, ScalarField, Value};
use glam::Vec2;

use crate::identity::{Anchor, ElementKey, LayoutId};
use crate::set::{AttributeDecl, Element, ElementError, ElementSet, Outline, Placement};

/// A curve failure.
#[derive(Clone, Debug, PartialEq)]
pub enum CurveError {
    /// Too few points (two for an open curve, three for a closed one), a
    /// width per point missing, a point or width not finite, a negative
    /// width, or a curve of no length.
    InvalidCurve,
    /// A stitch spacing that is not positive and finite.
    InvalidSpacing,
    /// The stitches do not make a valid element set.
    Elements(ElementError),
}

impl fmt::Display for CurveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCurve => f.write_str("a curve needs finite points and widths"),
            Self::InvalidSpacing => f.write_str("stitch spacing must be positive"),
            Self::Elements(e) => e.fmt(f),
        }
    }
}

impl core::error::Error for CurveError {}

/// A polyline with a width profile.
#[derive(Clone, Debug, PartialEq)]
pub struct Curve {
    points: Vec<Vec2>,
    widths: Vec<f32>,
    closed: bool,
    /// Arc length at each vertex, and the total at the end (after the
    /// closing segment of a closed curve).
    lengths: Vec<f32>,
}

impl Curve {
    /// An open polyline through `points`, `widths[i]` wide at `points[i]`.
    ///
    /// # Errors
    ///
    /// [`CurveError::InvalidCurve`].
    pub fn open(points: Vec<Vec2>, widths: Vec<f32>) -> Result<Self, CurveError> {
        Self::new(points, widths, false)
    }

    /// A closed polyline through `points`, back to the first.
    ///
    /// # Errors
    ///
    /// [`CurveError::InvalidCurve`].
    pub fn closed(points: Vec<Vec2>, widths: Vec<f32>) -> Result<Self, CurveError> {
        Self::new(points, widths, true)
    }

    fn new(points: Vec<Vec2>, widths: Vec<f32>, closed: bool) -> Result<Self, CurveError> {
        let minimum = if closed { 3 } else { 2 };
        if points.len() < minimum
            || widths.len() != points.len()
            || !points.iter().all(|p| p.is_finite())
            || !widths.iter().all(|w| w.is_finite() && *w >= 0.0)
        {
            return Err(CurveError::InvalidCurve);
        }
        let mut lengths = Vec::with_capacity(points.len() + 1);
        let mut total = 0.0_f32;
        lengths.push(0.0);
        let count = if closed {
            points.len()
        } else {
            points.len() - 1
        };
        for i in 0..count {
            total += (points[(i + 1) % points.len()] - points[i]).length();
            lengths.push(total);
        }
        if !(total > 0.0 && total.is_finite()) {
            return Err(CurveError::InvalidCurve);
        }
        Ok(Self {
            points,
            widths,
            closed,
            lengths,
        })
    }

    /// The vertices.
    #[must_use]
    pub fn points(&self) -> &[Vec2] {
        &self.points
    }

    /// Whether the curve closes back to its first point.
    #[must_use]
    pub const fn is_closed(&self) -> bool {
        self.closed
    }

    /// The arc length.
    #[must_use]
    pub fn length(&self) -> f32 {
        *self.lengths.last().expect("at least one segment")
    }

    fn segments(&self) -> usize {
        self.lengths.len() - 1
    }

    /// Segment `i`'s ends and widths.
    fn segment(&self, i: usize) -> (Vec2, Vec2, f32, f32) {
        let j = (i + 1) % self.points.len();
        (
            self.points[i],
            self.points[j],
            self.widths[i],
            self.widths[j],
        )
    }

    /// The point, unit tangent and width at arc length `s`, clamped to the
    /// curve (wrapped on a closed one).
    #[must_use]
    pub fn at(&self, s: f32) -> CurvePoint {
        let length = self.length();
        let s = if self.closed {
            let r = s - length * libm::floorf(s / length);
            if r >= length { 0.0 } else { r }
        } else {
            s.clamp(0.0, length)
        };
        // The last segment starting at or before `s`.
        let i = self.lengths[..self.segments()]
            .partition_point(|&l| l <= s)
            .saturating_sub(1);
        let (a, b, wa, wb) = self.segment(i);
        let span = self.lengths[i + 1] - self.lengths[i];
        let t = if span > 0.0 {
            ((s - self.lengths[i]) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        CurvePoint {
            position: a + (b - a) * t,
            tangent: (b - a).normalize_or_zero(),
            width: wa + (wb - wa) * t,
        }
    }
}

/// A point on a curve.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CurvePoint {
    /// Where it is.
    pub position: Vec2,
    /// The curve's unit direction there.
    pub tangent: Vec2,
    /// The curve's width there.
    pub width: f32,
}

/// A point's coordinates in the frame of its nearest curve.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CurveSample {
    /// The curve's position in the network.
    pub curve: u32,
    /// Distance to the curve.
    pub distance: f32,
    /// Arc length from the curve's start to the nearest point.
    pub along: f32,
    /// Signed offset: the distance, positive to the left of the curve's
    /// direction.
    pub across: f32,
    /// The curve's unit direction at the nearest point.
    pub tangent: Vec2,
    /// The curve's width at the nearest point.
    pub width: f32,
}

/// Where two curve segments cross.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Intersection {
    /// The crossing point.
    pub point: Vec2,
    /// The first curve and the arc length on it.
    pub a: (u32, f32),
    /// The second curve and the arc length on it.
    pub b: (u32, f32),
}

/// Which quantity a [`CurveField`] returns.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum CurveOutput {
    /// [`CurveSample::distance`].
    Distance,
    /// [`CurveSample::along`].
    Along,
    /// [`CurveSample::across`].
    Across,
    /// Coverage of the stroke of the width profile: the share of a
    /// footprint-wide box across the curve that the band within half the
    /// width covers. Exact for a straight stroke, so a stroke thinner than
    /// a texel keeps its area.
    Stroke,
}

/// Curves over a domain: see the [module docs](self).
#[derive(Clone, Debug, PartialEq)]
pub struct CurveNetwork {
    domain: Domain,
    curves: Vec<Curve>,
}

impl CurveNetwork {
    /// The network of `curves` over `domain`.
    #[must_use]
    pub const fn new(domain: Domain, curves: Vec<Curve>) -> Self {
        Self { domain, curves }
    }

    /// The domain.
    #[must_use]
    pub const fn domain(&self) -> Domain {
        self.domain
    }

    /// The curves.
    #[must_use]
    pub fn curves(&self) -> &[Curve] {
        &self.curves
    }

    /// The repeats of the plane to search: one, or nine on a torus.
    fn shifts(&self) -> impl Iterator<Item = Vec2> + '_ {
        let period = self.domain.period().map(|[x, y]| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "periods are below 2^24, exact in f32"
            )]
            let p = Vec2::new(x as f32, y as f32);
            p
        });
        let reach: i32 = if period.is_some() { 1 } else { 0 };
        (-reach..=reach).flat_map(move |j| {
            (-reach..=reach).map(move |i| {
                #[expect(clippy::cast_precision_loss, reason = "shifts of -1, 0 or 1")]
                let k = Vec2::new(i as f32, j as f32);
                period.map_or(Vec2::ZERO, |p| k * p)
            })
        })
    }

    /// The nearest curve to `p` and `p`'s coordinates in its frame, or
    /// `None` for an empty network. Ties go to the earlier curve and
    /// segment.
    #[must_use]
    pub fn nearest(&self, p: Vec2) -> Option<CurveSample> {
        let mut best: Option<(f32, CurveSample)> = None;
        for shift in self.shifts() {
            let q = p + shift;
            for (c, curve) in self.curves.iter().enumerate() {
                for i in 0..curve.segments() {
                    let (a, b, wa, wb) = curve.segment(i);
                    let ab = b - a;
                    let len2 = ab.length_squared();
                    let t = if len2 > 0.0 {
                        ((q - a).dot(ab) / len2).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let foot = a + ab * t;
                    let d2 = (q - foot).length_squared();
                    if best.as_ref().is_some_and(|(b2, _)| *b2 <= d2) {
                        continue;
                    }
                    let tangent = ab.normalize_or_zero();
                    let distance = libm::sqrtf(d2);
                    let side = tangent.perp_dot(q - foot);
                    best = Some((
                        d2,
                        CurveSample {
                            curve: u32::try_from(c).expect("fewer than 2^32 curves"),
                            distance,
                            along: curve.lengths[i] + (curve.lengths[i + 1] - curve.lengths[i]) * t,
                            across: if side < 0.0 { -distance } else { distance },
                            tangent,
                            width: wa + (wb - wa) * t,
                        },
                    ));
                }
            }
        }
        best.map(|(_, s)| s)
    }

    /// Every crossing of two segments, of different curves or of one
    /// curve's non-adjacent segments, in curve and arc-length order.
    #[must_use]
    pub fn intersections(&self) -> Vec<Intersection> {
        let mut out = Vec::new();
        let shifts: Vec<Vec2> = self.shifts().collect();
        for (ci, a) in self.curves.iter().enumerate() {
            for (cj, b) in self.curves.iter().enumerate().skip(ci) {
                for i in 0..a.segments() {
                    for j in 0..b.segments() {
                        if ci == cj {
                            let n = a.segments();
                            let adjacent =
                                j <= i || j == i + 1 || (a.closed && i == 0 && j == n - 1);
                            if adjacent {
                                continue;
                            }
                        }
                        let (p0, p1, _, _) = a.segment(i);
                        let (q0, q1, _, _) = b.segment(j);
                        for &shift in &shifts {
                            if ci == cj && shift != Vec2::ZERO {
                                continue;
                            }
                            if let Some((t, u)) = cross(p0, p1, q0 + shift, q1 + shift) {
                                out.push(Intersection {
                                    point: p0 + (p1 - p0) * t,
                                    a: (
                                        u32::try_from(ci).expect("fewer than 2^32 curves"),
                                        a.lengths[i] + (a.lengths[i + 1] - a.lengths[i]) * t,
                                    ),
                                    b: (
                                        u32::try_from(cj).expect("fewer than 2^32 curves"),
                                        b.lengths[j] + (b.lengths[j + 1] - b.lengths[j]) * u,
                                    ),
                                });
                            }
                        }
                    }
                }
            }
        }
        out.sort_by(|x, y| {
            x.a.0
                .cmp(&y.a.0)
                .then(x.a.1.total_cmp(&y.a.1))
                .then(x.b.0.cmp(&y.b.0))
                .then(x.b.1.total_cmp(&y.b.1))
        });
        out
    }

    /// One quantity of the network as a scalar field.
    #[must_use]
    pub fn field(&self, output: CurveOutput) -> CurveField<'_> {
        CurveField {
            network: self,
            output,
        }
    }

    /// Elements every `spacing` domain units along each curve, the first
    /// half a spacing from its start (open curves hold as many as fit),
    /// each `half_size` in extent, centered on the curve and turned to its
    /// tangent, with attributes from `attributes` for each key, curve point
    /// and arc length.
    ///
    /// Stitch `k` of curve `c` has the key of anchor `[c, k]` in `layout`,
    /// so changing the spacing moves stitches and adds or removes them at
    /// the ends without renaming the others. Positions come from arc
    /// length alone, independent of any realization.
    ///
    /// # Errors
    ///
    /// [`CurveError::InvalidSpacing`], or [`CurveError::Elements`] when
    /// the stitches do not make a valid set (attributes that do not fit
    /// `schema`, say).
    pub fn stitches(
        &self,
        layout: LayoutId,
        spacing: f32,
        half_size: Vec2,
        schema: Vec<AttributeDecl>,
        mut attributes: impl FnMut(ElementKey, CurvePoint, f32) -> Vec<Value>,
    ) -> Result<ElementSet, CurveError> {
        if !(spacing.is_finite() && spacing > 0.0) {
            return Err(CurveError::InvalidSpacing);
        }
        let mut elements = Vec::new();
        for (c, curve) in self.curves.iter().enumerate() {
            let length = curve.length();
            let fits = libm::floorf(length / spacing);
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a non-negative whole count, far below u32::MAX for sane spacings"
            )]
            let count = fits.min(1.0e6) as u32;
            for k in 0..count {
                #[expect(clippy::cast_precision_loss, reason = "stitch counts are small")]
                let s = (k as f32 + 0.5) * spacing;
                let point = curve.at(s);
                let key = ElementKey::new(
                    layout,
                    Anchor([
                        i32::try_from(c).expect("fewer than 2^31 curves"),
                        i32::try_from(k).expect("fewer than 2^31 stitches"),
                    ]),
                    0,
                );
                elements.push(Element {
                    key,
                    placement: Placement {
                        center: point.position,
                        rotation: libm::atan2f(point.tangent.y, point.tangent.x),
                    },
                    half_size,
                    outline: Outline::Rectangle,
                    variant: 0,
                    attributes: attributes(key, point, s),
                });
            }
        }
        ElementSet::new(schema, elements).map_err(CurveError::Elements)
    }
}

/// Where segments `p0 p1` and `q0 q1` cross, as their parameters, or
/// `None` when they are parallel or miss.
fn cross(p0: Vec2, p1: Vec2, q0: Vec2, q1: Vec2) -> Option<(f32, f32)> {
    let r = p1 - p0;
    let s = q1 - q0;
    let denom = r.perp_dot(s);
    if denom == 0.0 {
        return None;
    }
    let d = q0 - p0;
    let t = d.perp_dot(s) / denom;
    let u = d.perp_dot(r) / denom;
    ((0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u)).then_some((t, u))
}

/// One quantity of a [`CurveNetwork`] as a scalar field.
///
/// Distance, along and across are point values: they do not band-limit.
/// The stroke is box-filtered across the curve over the footprint, so its
/// coverage is exact for straight strokes at any resolution, however thin.
#[derive(Copy, Clone, Debug)]
pub struct CurveField<'a> {
    network: &'a CurveNetwork,
    output: CurveOutput,
}

impl ScalarField for CurveField<'_> {
    fn domain(&self) -> Domain {
        self.network.domain
    }

    fn eval(&self, p: Vec2, footprint: Footprint) -> f32 {
        let Some(s) = self.network.nearest(p) else {
            return match self.output {
                CurveOutput::Distance => f32::INFINITY,
                _ => 0.0,
            };
        };
        match self.output {
            CurveOutput::Distance => s.distance,
            CurveOutput::Along => s.along,
            CurveOutput::Across => s.across,
            CurveOutput::Stroke => {
                // The share of a footprint-wide box across the curve, centered
                // `distance` from it, that the band of the curve's width
                // covers.
                let (d, half) = (s.distance, 0.5 * s.width);
                let f = footprint.width();
                if f > 0.0 {
                    let lo = (d - half).max(-0.5 * f);
                    let hi = (d + half).min(0.5 * f);
                    ((hi - lo) / f).clamp(0.0, 1.0)
                } else if d <= half {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    fn torus() -> Domain {
        Domain::periodic(1, 1).unwrap()
    }

    #[test]
    fn curves_measure_arc_length_and_width() {
        let c = Curve::open(
            vec![Vec2::ZERO, Vec2::new(0.3, 0.0), Vec2::new(0.3, 0.4)],
            vec![0.01, 0.02, 0.04],
        )
        .unwrap();
        assert!((c.length() - 0.7).abs() < 1e-6);
        let mid = c.at(0.5);
        assert!((mid.position - Vec2::new(0.3, 0.2)).length() < 1e-6);
        assert_eq!(mid.tangent, Vec2::Y);
        assert!((mid.width - 0.03).abs() < 1e-6);
        assert!(Curve::open(vec![Vec2::ZERO], vec![0.0]).is_err());
        assert!(Curve::open(vec![Vec2::ZERO, Vec2::ZERO], vec![0.0, 0.0]).is_err());
        let ring =
            Curve::closed(vec![Vec2::ZERO, Vec2::X, Vec2::new(1.0, 1.0)], vec![0.0; 3]).unwrap();
        assert!((ring.length() - (2.0 + core::f32::consts::SQRT_2)).abs() < 1e-5);
        // Closed curves wrap their arc length.
        assert_eq!(ring.at(ring.length() + 0.5).position, ring.at(0.5).position);
    }

    #[test]
    fn samples_carry_the_curve_frame() {
        let line = Curve::open(
            vec![Vec2::new(0.2, 0.5), Vec2::new(0.8, 0.5)],
            vec![0.02; 2],
        )
        .unwrap();
        let net = CurveNetwork::new(Domain::Plane, vec![line]);
        let s = net.nearest(Vec2::new(0.3, 0.55)).unwrap();
        assert!((s.distance - 0.05).abs() < 1e-6);
        assert!((s.along - 0.1).abs() < 1e-6);
        assert!(s.across > 0.0, "left of a rightward curve is up");
        let below = net.nearest(Vec2::new(0.3, 0.45)).unwrap();
        assert!(below.across < 0.0);
        // The stroke covers half the width either side.
        let stroke = net.field(CurveOutput::Stroke);
        let fp = Footprint::new(0.001).unwrap();
        assert_eq!(stroke.eval(Vec2::new(0.5, 0.505), fp), 1.0);
        assert_eq!(stroke.eval(Vec2::new(0.5, 0.52), fp), 0.0);
        assert!((stroke.eval(Vec2::new(0.5, 0.51), fp) - 0.5).abs() < 1e-3);
    }

    #[test]
    fn periodic_networks_repeat() {
        let line = Curve::open(
            vec![Vec2::new(0.1, 0.02), Vec2::new(0.9, 0.02)],
            vec![0.0; 2],
        )
        .unwrap();
        let net = CurveNetwork::new(torus(), vec![line]);
        // Just below y = 1 is just below the line's repeat at y = 1.02.
        let s = net.nearest(Vec2::new(0.5, 0.99)).unwrap();
        assert!((s.distance - 0.03).abs() < 1e-5, "{s:?}");
        let d = net.field(CurveOutput::Distance);
        let fp = Footprint::POINT;
        assert!((d.eval(Vec2::new(0.4, 0.3), fp) - d.eval(Vec2::new(1.4, -0.7), fp)).abs() < 1e-5);
    }

    #[test]
    fn crossings_are_found_on_both_curves() {
        let a = Curve::open(vec![Vec2::new(0.0, 0.5), Vec2::new(1.0, 0.5)], vec![0.0; 2]).unwrap();
        let b = Curve::open(
            vec![Vec2::new(0.25, 0.0), Vec2::new(0.25, 1.0)],
            vec![0.0; 2],
        )
        .unwrap();
        let net = CurveNetwork::new(Domain::Plane, vec![a, b]);
        let x = net.intersections();
        assert_eq!(x.len(), 1);
        assert!((x[0].point - Vec2::new(0.25, 0.5)).length() < 1e-6);
        assert!((x[0].a.1 - 0.25).abs() < 1e-6 && (x[0].b.1 - 0.5).abs() < 1e-6);
        // A figure eight crosses itself once.
        let eight = Curve::closed(
            vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 1.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(0.0, 1.0),
            ],
            vec![0.0; 4],
        )
        .unwrap();
        let net = CurveNetwork::new(Domain::Plane, vec![eight]);
        assert_eq!(net.intersections().len(), 1);
    }

    #[test]
    fn stitches_follow_arc_length() {
        let seam = Curve::open(
            vec![
                Vec2::new(0.1, 0.1),
                Vec2::new(0.5, 0.1),
                Vec2::new(0.5, 0.3),
            ],
            vec![0.0; 3],
        )
        .unwrap();
        let net = CurveNetwork::new(torus(), vec![seam]);
        let layout = LayoutId::named("test.seam");
        let set = net
            .stitches(
                layout,
                0.04,
                Vec2::new(0.01, 0.002),
                Vec::new(),
                |_, _, _| Vec::new(),
            )
            .unwrap();
        // 0.6 of seam at 4 cm: 15 stitches.
        assert_eq!(set.len(), 15);
        let key = ElementKey::new(layout, Anchor([0, 12]), 0);
        let i = set.index_of(key).unwrap();
        // Stitch 12 is at 0.5 along: past the corner, heading up.
        let p = set.placement(i);
        assert!((p.center - Vec2::new(0.5, 0.2)).length() < 1e-6);
        assert!((p.rotation - core::f32::consts::FRAC_PI_2).abs() < 1e-6);
        // A wider spacing keeps the keys of the stitches that remain.
        let wide = net
            .stitches(
                layout,
                0.05,
                Vec2::new(0.01, 0.002),
                Vec::new(),
                |_, _, _| Vec::new(),
            )
            .unwrap();
        assert_eq!(wide.len(), 12);
        assert!(wide.keys().iter().all(|k| set.index_of(*k).is_some()));
        assert!(matches!(
            net.stitches(layout, 0.0, Vec2::ONE, Vec::new(), |_, _, _| Vec::new()),
            Err(CurveError::InvalidSpacing)
        ));
    }
}
