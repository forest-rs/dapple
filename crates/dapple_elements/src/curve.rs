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

    /// A crossing's topological location and arc length. Interior crossings
    /// keep their segment identity even when their arc lengths round alike.
    fn crossing_location(&self, segment: usize, t: f64) -> (CurveLocation, f32) {
        if t == 0.0 {
            (CurveLocation::Vertex(segment), self.lengths[segment])
        } else if t == 1.0 {
            let vertex = (segment + 1) % self.points.len();
            (CurveLocation::Vertex(vertex), self.lengths[vertex])
        } else {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "arc lengths are stored as f32"
            )]
            let along = (f64::from(self.lengths[segment])
                + f64::from(self.lengths[segment + 1] - self.lengths[segment]) * t)
                as f32;
            (CurveLocation::Segment(segment), along)
        }
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
    /// a texel keeps its area. Where strokes overlap, the one that covers
    /// most wins (they do not add), so a wide curve is not cut short by a
    /// hairline crossing it.
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

    /// The most any one segment's stroke covers a `footprint`-wide box
    /// centered at `p`: see [`CurveOutput::Stroke`].
    fn stroke(&self, p: Vec2, footprint: f32) -> f32 {
        let mut best = 0.0_f32;
        for shift in self.shifts() {
            let q = p + shift;
            for curve in &self.curves {
                for i in 0..curve.segments() {
                    let (a, b, wa, wb) = curve.segment(i);
                    let ab = b - a;
                    let len2 = ab.length_squared();
                    let t = if len2 > 0.0 {
                        ((q - a).dot(ab) / len2).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let distance = (q - (a + ab * t)).length();
                    let half = 0.5 * (wa + (wb - wa) * t);
                    best = best.max(stroke_coverage(distance, half, footprint));
                }
            }
            if best >= 1.0 {
                break;
            }
        }
        best
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
    /// curve's non-adjacent segments, in curve and arc-length order. A
    /// crossing at a shared vertex is reported once, not once per segment.
    #[must_use]
    pub fn intersections(&self) -> Vec<Intersection> {
        let mut found = Vec::new();
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
                        for (image, &shift) in shifts.iter().enumerate() {
                            if ci == cj && shift != Vec2::ZERO {
                                continue;
                            }
                            if let Some((t, u)) = cross(p0, p1, q0 + shift, q1 + shift) {
                                let (mut left, mut sa) = a.crossing_location(i, t);
                                let (mut right, mut sb) = b.crossing_location(j, u);
                                if ci == cj && left > right {
                                    core::mem::swap(&mut left, &mut right);
                                }
                                if ci == cj && sa > sb {
                                    core::mem::swap(&mut sa, &mut sb);
                                }
                                let point = [0, 1].map(|axis| {
                                    let (lo, hi) = (f64::from(p0[axis]), f64::from(p1[axis]));
                                    #[expect(
                                        clippy::cast_possible_truncation,
                                        reason = "curve positions are stored as f32"
                                    )]
                                    let coordinate = (lo + (hi - lo) * t) as f32;
                                    coordinate
                                });
                                let crossing = Intersection {
                                    point: Vec2::from_array(point),
                                    a: (u32::try_from(ci).expect("fewer than 2^32 curves"), sa),
                                    b: (u32::try_from(cj).expect("fewer than 2^32 curves"), sb),
                                };
                                found.push((left, right, image, crossing));
                            }
                        }
                    }
                }
            }
        }
        // Only reports of the same vertex/segment pair in the same periodic
        // image repeat. Arc-length proximity cannot identify a crossing:
        // distinct ones can be arbitrarily close, or round to the same f32.
        found.sort_by_key(|(left, right, image, crossing)| {
            (crossing.a.0, crossing.b.0, *left, *right, *image)
        });
        found.dedup_by_key(|(left, right, image, crossing)| {
            (crossing.a.0, crossing.b.0, *left, *right, *image)
        });
        let mut out: Vec<Intersection> = found
            .into_iter()
            .map(|(_, _, _, crossing)| crossing)
            .collect();
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
///
/// Compute parameters in f64 so an interior crossing near an endpoint is
/// not rounded to the endpoint before its topological identity is recorded.
fn cross(p0: Vec2, p1: Vec2, q0: Vec2, q1: Vec2) -> Option<(f64, f64)> {
    let difference = |a: Vec2, b: Vec2| {
        [
            f64::from(a.x) - f64::from(b.x),
            f64::from(a.y) - f64::from(b.y),
        ]
    };
    let perp_dot = |a: [f64; 2], b: [f64; 2]| a[0] * b[1] - a[1] * b[0];
    let r = difference(p1, p0);
    let s = difference(q1, q0);
    let denom = perp_dot(r, s);
    if denom == 0.0 {
        return None;
    }
    let d = difference(q0, p0);
    let t = perp_dot(d, s) / denom;
    let u = perp_dot(d, r) / denom;
    ((0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u)).then_some((t, u))
}

/// Shared endpoints have a vertex identity; each segment's interior is
/// distinct from its endpoints and from every other segment's interior.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum CurveLocation {
    Vertex(usize),
    Segment(usize),
}

/// The share of a `footprint`-wide box across a curve, centered `distance`
/// from it, that the band `half` either side of the curve covers.
fn stroke_coverage(distance: f32, half: f32, footprint: f32) -> f32 {
    if footprint > 0.0 {
        let lo = (distance - half).max(-0.5 * footprint);
        let hi = (distance + half).min(0.5 * footprint);
        ((hi - lo) / footprint).clamp(0.0, 1.0)
    } else if distance <= half {
        1.0
    } else {
        0.0
    }
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
        if self.output == CurveOutput::Stroke {
            return self.network.stroke(p, footprint.width());
        }
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
            CurveOutput::Stroke => unreachable!("strokes are evaluated above"),
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
    fn strokes_take_the_widest_covering_stroke() {
        // A wide horizontal curve and a hairline crossing it: a point inside
        // the wide band but nearer the hairline is still covered.
        let wide =
            Curve::open(vec![Vec2::new(0.0, 0.5), Vec2::new(1.0, 0.5)], vec![0.2; 2]).unwrap();
        let thin =
            Curve::open(vec![Vec2::new(0.5, 0.0), Vec2::new(0.5, 1.0)], vec![0.0; 2]).unwrap();
        let net = CurveNetwork::new(Domain::Plane, vec![wide, thin]);
        let stroke = net.field(CurveOutput::Stroke);
        let fp = Footprint::new(0.001).unwrap();
        // 0.05 above the wide curve's axis, 0.02 from the hairline.
        assert_eq!(stroke.eval(Vec2::new(0.52, 0.55), fp), 1.0);

        // The same holds within one curve that doubles back across itself,
        // wide on the way out and a hairline on the way back.
        let loop_back = Curve::open(
            vec![
                Vec2::new(0.0, 0.5),
                Vec2::new(1.0, 0.5),
                Vec2::new(1.0, 0.9),
                Vec2::new(0.5, 0.9),
                Vec2::new(0.5, 0.0),
            ],
            vec![0.2, 0.2, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let net = CurveNetwork::new(Domain::Plane, vec![loop_back]);
        let stroke = net.field(CurveOutput::Stroke);
        assert_eq!(stroke.eval(Vec2::new(0.52, 0.55), fp), 1.0);
        assert_eq!(stroke.eval(Vec2::new(0.52, 0.8), fp), 0.0, "uncovered");
    }

    #[test]
    fn a_crossing_at_a_vertex_is_reported_once() {
        // Give every curve the same explicit vertex. Merely centering two
        // rounded endpoints around `bend` can put their segment slightly off
        // it, making two real crossings that must stay distinct.
        for scale in [1.0_f32, 100.0, 1000.0] {
            let at = |x: f32, y: f32| Vec2::new(x, y) * scale;
            let bend = at(0.517, 0.483);
            let bent = Curve::open(vec![at(0.0, 0.4), bend, at(1.0, 0.31)], vec![0.0; 3]).unwrap();
            let mut curves = vec![bent];
            for k in 0..24_u8 {
                let angle = 0.13 + 0.131 * f32::from(k);
                let d = Vec2::new(libm::cosf(angle), libm::sinf(angle)) * scale * 0.4;
                curves.push(Curve::open(vec![bend - d, bend, bend + d], vec![0.0; 3]).unwrap());
            }
            let net = CurveNetwork::new(Domain::Plane, curves);
            let x = net.intersections();
            for a in 0..25 {
                for b in a + 1..25 {
                    let count = x.iter().filter(|c| c.a.0 == a && c.b.0 == b).count();
                    assert_eq!(count, 1, "scale {scale}: curves {a} and {b}");
                }
            }
        }
    }

    #[test]
    fn a_vertex_crossing_another_segments_interior_is_reported_once() {
        for scale in [1.0_f32, 100.0, 1000.0] {
            let bend = Vec2::new(0.517, 0.483) * scale;
            let bent = Curve::open(
                vec![
                    Vec2::new(0.0, 0.4) * scale,
                    bend,
                    Vec2::new(1.0, 0.31) * scale,
                ],
                vec![0.0; 3],
            )
            .unwrap();
            let line = Curve::open(
                vec![Vec2::new(bend.x, -scale), Vec2::new(bend.x, 2.0 * scale)],
                vec![0.0; 2],
            )
            .unwrap();
            let crossings = CurveNetwork::new(Domain::Plane, vec![bent, line]).intersections();
            assert_eq!(crossings.len(), 1, "scale {scale}: {crossings:?}");
            assert_eq!(crossings[0].point, bend);
        }
    }

    #[test]
    fn nearby_crossings_with_long_arc_lengths_stay_distinct() {
        let a = Curve::open(vec![Vec2::ZERO, Vec2::new(2000.0, 0.0)], vec![0.0; 2]).unwrap();
        let b = Curve::open(
            vec![
                Vec2::new(0.0, -1.0),
                Vec2::new(1000.0, -1.0),
                Vec2::new(1000.0, 0.001),
                Vec2::new(1000.005, -0.001),
            ],
            vec![0.0; 4],
        )
        .unwrap();
        let crossings = CurveNetwork::new(Domain::Plane, vec![a, b]).intersections();
        assert_eq!(crossings.len(), 2, "{crossings:?}");
        assert!(crossings[0].point.x < crossings[1].point.x);
    }

    #[test]
    fn distinct_crossings_near_a_vertex_are_not_vertex_reports() {
        let a = Curve::open(vec![Vec2::ZERO, Vec2::new(2000.0, 0.0)], vec![0.0; 2]).unwrap();
        let b = Curve::open(
            vec![
                Vec2::new(0.0, -1.0),
                Vec2::new(1000.0, 1e-7),
                Vec2::new(2000.0, -1.0),
            ],
            vec![0.0; 3],
        )
        .unwrap();
        let crossings = CurveNetwork::new(Domain::Plane, vec![a, b]).intersections();
        assert_eq!(crossings.len(), 2, "{crossings:?}");
        assert!(crossings[0].point.x < 1000.0 && crossings[1].point.x > 1000.0);
    }

    #[test]
    fn a_crossing_at_the_closing_vertex_is_reported_once() {
        let closed = Curve::closed(vec![Vec2::ZERO, Vec2::X, Vec2::Y], vec![0.0; 3]).unwrap();
        let line =
            Curve::open(vec![Vec2::new(-1.0, -1.0), Vec2::splat(0.25)], vec![0.0; 2]).unwrap();
        let crossings = CurveNetwork::new(Domain::Plane, vec![closed, line]).intersections();
        assert_eq!(crossings.len(), 1, "{crossings:?}");
        assert_eq!(crossings[0].point, Vec2::ZERO);
        assert_eq!(crossings[0].a.1, 0.0);
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
