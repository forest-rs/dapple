// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Pointwise value shaping: tone curves and color ramps.
//!
//! Both are plain data, so programs that use them stay inspectable and
//! fingerprintable:
//!
//! - [`ToneCurve`]: a monotone cubic through control points (Fritsch and
//!   Carlson), so a curve through increasing points never overshoots, and
//!   is flat past its ends;
//! - [`ColorRamp`]: linear colors at increasing stops, interpolated in
//!   linear light (a gradient map), constant past its ends.
//!
//! Measured shaping, such as histogram and percentile remapping, needs the
//! whole raster and lives in `dapple_raster`.

use alloc::vec::Vec;
use core::fmt;

use glam::Vec3;

/// Why a curve or ramp was refused.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ShapingError {
    /// Fewer than two points or stops.
    TooFew,
    /// Positions are not finite and strictly increasing, or a value is not
    /// finite.
    NotIncreasing,
}

impl fmt::Display for ShapingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::TooFew => "a curve or ramp needs at least two points",
            Self::NotIncreasing => "positions must be finite and strictly increasing",
        })
    }
}

impl core::error::Error for ShapingError {}

/// A monotone cubic tone curve: see the [module docs](self).
#[derive(Clone, Debug, PartialEq)]
pub struct ToneCurve {
    points: Vec<[f32; 2]>,
    tangents: Vec<f32>,
}

impl ToneCurve {
    /// The curve through `points`, `[x, y]` with strictly increasing `x`.
    ///
    /// Tangents follow Fritsch and Carlson: the harmonic mean of the
    /// neighboring secants, zero at a local extremum, so the curve is
    /// monotone wherever its points are.
    ///
    /// # Errors
    ///
    /// [`ShapingError`] for fewer than two points, or positions that are
    /// not finite and increasing.
    pub fn new(points: &[[f32; 2]]) -> Result<Self, ShapingError> {
        check(points.iter().map(|p| p[0]), points.len())?;
        if points.iter().any(|p| !p[1].is_finite()) {
            return Err(ShapingError::NotIncreasing);
        }
        let n = points.len();
        let secant = |i: usize| {
            let (a, b) = (points[i], points[i + 1]);
            (b[1] - a[1]) / (b[0] - a[0])
        };
        let mut tangents = Vec::with_capacity(n);
        for i in 0..n {
            let t = if i == 0 {
                secant(0)
            } else if i == n - 1 {
                secant(n - 2)
            } else {
                let (s0, s1) = (secant(i - 1), secant(i));
                if s0 * s1 <= 0.0 {
                    0.0
                } else {
                    let (h0, h1) = (
                        points[i][0] - points[i - 1][0],
                        points[i + 1][0] - points[i][0],
                    );
                    // Weighted harmonic mean (Fritsch–Butland), which keeps
                    // the interpolant monotone.
                    let w0 = 2.0 * h1 + h0;
                    let w1 = h1 + 2.0 * h0;
                    (w0 + w1) / (w0 / s0 + w1 / s1)
                }
            };
            tangents.push(t);
        }
        Ok(Self {
            points: points.to_vec(),
            tangents,
        })
    }

    /// The identity on `[0, 1]`.
    #[must_use]
    pub fn identity() -> Self {
        Self::new(&[[0.0, 0.0], [1.0, 1.0]]).expect("two increasing points")
    }

    /// The control points.
    #[must_use]
    pub fn points(&self) -> &[[f32; 2]] {
        &self.points
    }

    /// The curve at `x`: the first or last point's `y` outside the points.
    #[must_use]
    pub fn eval(&self, x: f32) -> f32 {
        let p = &self.points;
        let n = p.len();
        if x.is_nan() || x <= p[0][0] {
            return p[0][1];
        }
        if x >= p[n - 1][0] {
            return p[n - 1][1];
        }
        let i = p.partition_point(|q| q[0] <= x) - 1;
        let (a, b) = (p[i], p[i + 1]);
        let h = b[0] - a[0];
        let t = (x - a[0]) / h;
        let (t2, t3) = (t * t, t * t * t);
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        h00 * a[1] + h10 * h * self.tangents[i] + h01 * b[1] + h11 * h * self.tangents[i + 1]
    }

    pub(crate) fn words(&self, w: &mut Vec<u64>) {
        w.push(self.points.len() as u64);
        for q in &self.points {
            w.extend([u64::from(q[0].to_bits()), u64::from(q[1].to_bits())]);
        }
    }
}

/// A gradient map of linear colors: see the [module docs](self).
#[derive(Clone, Debug, PartialEq)]
pub struct ColorRamp {
    stops: Vec<(f32, Vec3)>,
}

impl ColorRamp {
    /// A ramp through `stops`, `(position, linear color)` with strictly
    /// increasing positions.
    ///
    /// # Errors
    ///
    /// [`ShapingError`] for fewer than two stops, or positions that are not
    /// finite and increasing, or colors that are not finite.
    pub fn new(stops: &[(f32, Vec3)]) -> Result<Self, ShapingError> {
        check(stops.iter().map(|s| s.0), stops.len())?;
        if stops.iter().any(|s| !s.1.is_finite()) {
            return Err(ShapingError::NotIncreasing);
        }
        Ok(Self {
            stops: stops.to_vec(),
        })
    }

    /// The stops.
    #[must_use]
    pub fn stops(&self) -> &[(f32, Vec3)] {
        &self.stops
    }

    /// The color at `t`, linear between the stops around it.
    #[must_use]
    pub fn eval(&self, t: f32) -> Vec3 {
        let s = &self.stops;
        let n = s.len();
        if t.is_nan() || t <= s[0].0 {
            return s[0].1;
        }
        if t >= s[n - 1].0 {
            return s[n - 1].1;
        }
        let i = s.partition_point(|q| q.0 <= t) - 1;
        let (a, b) = (s[i], s[i + 1]);
        let u = (t - a.0) / (b.0 - a.0);
        a.1 + (b.1 - a.1) * u
    }

    pub(crate) fn words(&self, w: &mut Vec<u64>) {
        w.push(self.stops.len() as u64);
        for (t, c) in &self.stops {
            w.extend([
                u64::from(t.to_bits()),
                u64::from(c.x.to_bits()),
                u64::from(c.y.to_bits()),
                u64::from(c.z.to_bits()),
            ]);
        }
    }
}

fn check(positions: impl Iterator<Item = f32>, len: usize) -> Result<(), ShapingError> {
    if len < 2 {
        return Err(ShapingError::TooFew);
    }
    let mut last = f32::NEG_INFINITY;
    for x in positions {
        if !(x.is_finite() && x > last) {
            return Err(ShapingError::NotIncreasing);
        }
        last = x;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curves_pass_through_their_points_and_stay_monotone() {
        let c = ToneCurve::new(&[[0.0, 0.0], [0.3, 0.1], [0.5, 0.8], [1.0, 1.0]]).unwrap();
        for p in c.points() {
            assert!((c.eval(p[0]) - p[1]).abs() < 1e-6, "passes through {p:?}");
        }
        let mut last = c.eval(-1.0);
        for i in 0..=1000 {
            let y = c.eval(i as f32 / 1000.0);
            assert!(y >= last - 1e-6, "monotone at {i}");
            assert!((0.0..=1.0).contains(&y), "no overshoot at {i}");
            last = y;
        }
        assert_eq!(c.eval(2.0), 1.0, "flat past the end");
        assert_eq!(ToneCurve::identity().eval(0.25), 0.25, "identity");
    }

    #[test]
    fn ramps_interpolate_in_linear_light() {
        let r = ColorRamp::new(&[(0.0, Vec3::ZERO), (1.0, Vec3::new(1.0, 0.5, 0.0))]).unwrap();
        assert_eq!(r.eval(0.5), Vec3::new(0.5, 0.25, 0.0), "midpoint");
        assert_eq!(r.eval(-1.0), Vec3::ZERO, "clamped below");
        assert_eq!(
            ColorRamp::new(&[(0.0, Vec3::ZERO)]),
            Err(ShapingError::TooFew),
            "one stop"
        );
        assert_eq!(
            ToneCurve::new(&[[0.0, 0.0], [0.0, 1.0]]),
            Err(ShapingError::NotIncreasing),
            "repeated x"
        );
    }
}
