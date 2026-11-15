// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Periodic waves, box-filtered in closed form.
//!
//! A [`WaveRamp`] is a clamped linear ramp of a period-1 wave of its phase:
//! a ring's darkening toward its latewood, or a stripe around each whole
//! phase. Its antiderivative over the phase is piecewise quadratic, so its
//! mean over any interval of phase is exact arithmetic, not sampling: as a
//! footprint covers more of a period the ramp fades to its mean over the
//! period, and a stripe narrower than the footprint spreads to its share of
//! it, instead of aliasing as point samples of [`Op::Fract`] do.
//!
//! [`Op::Fract`]: crate::program::Op::Fract

/// The shape of a period-1 wave of a phase `x`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum WaveShape {
    /// `x − ⌊x⌋`, rising from 0 to 1 over each period and dropping back.
    Sawtooth,
    /// The distance from `x` to the nearest whole number, from 0 at each
    /// whole phase to ½ halfway between.
    Triangle,
}

impl WaveShape {
    /// The wave at `t` in `[0, 1)`.
    #[inline(always)]
    fn at(self, t: f32) -> f32 {
        match self {
            Self::Sawtooth => t,
            Self::Triangle => t.min(1.0 - t),
        }
    }

    /// The wave's slope at `t` in `[0, 1)`.
    #[inline(always)]
    fn slope(self, t: f32) -> f32 {
        match self {
            Self::Sawtooth => 1.0,
            Self::Triangle => {
                if t < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
        }
    }
}

/// `to[0] + (to[1] − to[0]) · clamp((w(x) − shift − from[0]) / (from[1] −
/// from[0]), 0, 1)` for the wave `w` of `shape`, as [`Op::Wave`] evaluates
/// it; equal `from` ends make a hard step up at `from[0] + shift`.
///
/// [`Op::Wave`]: crate::program::Op::Wave
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct WaveRamp {
    pub(crate) shape: WaveShape,
    pub(crate) from: [f32; 2],
    pub(crate) to: [f32; 2],
}

/// `∫ clamp((v − a) / m, 0, 1) dv` up to `u`, zero at and below `a`: a rising
/// ramp from `a` over `m ≥ 0`.
#[inline(always)]
fn rising(u: f32, a: f32, m: f32) -> f32 {
    let d = u - a;
    if d <= 0.0 {
        0.0
    } else if d < m {
        d * d / (2.0 * m)
    } else {
        d - 0.5 * m
    }
}

impl WaveRamp {
    /// The clamped ramp of a wave value `u` with the ramp's start at `c`
    /// (`from[0]` plus the shift), in `[0, 1]`.
    #[inline(always)]
    fn ramp(&self, u: f32, c: f32) -> f32 {
        let k = self.from[1] - self.from[0];
        if k == 0.0 {
            if u > c { 1.0 } else { 0.0 }
        } else {
            ((u - c) / k).clamp(0.0, 1.0)
        }
    }

    /// The ramp's slope at wave value `u`: `1 / k` inside it, 0 outside.
    #[inline(always)]
    fn ramp_slope(&self, u: f32, c: f32) -> f32 {
        let k = self.from[1] - self.from[0];
        let inside = if k > 0.0 {
            u > c && u < c + k
        } else {
            u < c && u > c + k
        };
        if inside { 1.0 / k } else { 0.0 }
    }

    /// An antiderivative of the ramp over the wave value.
    #[inline(always)]
    fn antiderivative(&self, u: f32, c: f32) -> f32 {
        let k = self.from[1] - self.from[0];
        if k >= 0.0 {
            rising(u, c, k)
        } else {
            // A falling ramp is one minus the rising ramp from `c + k`.
            u - rising(u, c + k, -k)
        }
    }

    /// `∫₀ᵗ ramp(w(τ)) dτ` for `t` in `[0, 1]`, and its derivative by `c`.
    #[inline(always)]
    fn cumulative(&self, t: f32, c: f32) -> (f32, f32) {
        let h = |u: f32| self.antiderivative(u, c) - self.antiderivative(0.0, c);
        // ∂/∂c of `h(u)` is `ramp(0) − ramp(u)`: the antiderivative is a
        // function of `u − c`, up to a constant that cancels.
        let dh = |u: f32| self.ramp(0.0, c) - self.ramp(u, c);
        match self.shape {
            WaveShape::Sawtooth => (h(t), dh(t)),
            WaveShape::Triangle => {
                if t <= 0.5 {
                    (h(t), dh(t))
                } else {
                    // Mirrored about ½: the rise to ½, then the same ramp
                    // read back down.
                    (2.0 * h(0.5) - h(1.0 - t), 2.0 * dh(0.5) - dh(1.0 - t))
                }
            }
        }
    }

    /// The ramp at phase `x` and `shift`, averaged over the phases `width`
    /// wide around `x`, with its derivatives by `x` and by `shift`.
    ///
    /// A `width` too small to separate the interval's ends in `f32` is a
    /// point sample; an infinite or NaN one gives the period's mean. The
    /// mean divides by the interval's width as rounded, so it is the mean
    /// over the interval the ends bound, however far from 0 the phase is.
    pub(crate) fn eval(&self, x: f32, shift: f32, width: f32) -> (f32, f32, f32) {
        let c = self.from[0] + shift;
        let span = self.to[1] - self.to[0];
        let (x0, x1) = (x - 0.5 * width, x + 0.5 * width);
        let d = x1 - x0;
        if !(d > 0.0 && d.is_finite()) {
            if d == 0.0 || x.is_nan() {
                let t = crate::program::fract(x);
                let u = self.shape.at(t);
                let slope = self.ramp_slope(u, c);
                return (
                    self.to[0] + span * self.ramp(u, c),
                    span * slope * self.shape.slope(t),
                    -span * slope,
                );
            }
            // The mean over a whole period.
            let (whole, dwhole) = self.cumulative(1.0, c);
            return (self.to[0] + span * whole, 0.0, span * dwhole);
        }
        let (n0, n1) = (libm::floorf(x0), libm::floorf(x1));
        let (t0, t1) = ((x0 - n0).clamp(0.0, 1.0), (x1 - n1).clamp(0.0, 1.0));
        let periods = n1 - n0;
        let (whole, dwhole) = self.cumulative(1.0, c);
        let (g0, dg0) = self.cumulative(t0, c);
        let (g1, dg1) = self.cumulative(t1, c);
        let area = periods * whole + (g1 - g0);
        let darea = periods * dwhole + (dg1 - dg0);
        let edge = |t: f32| self.ramp(self.shape.at(t), c);
        (
            self.to[0] + span * area / d,
            span * (edge(t1) - edge(t0)) / d,
            span * darea / d,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RAMPS: [WaveRamp; 5] = [
        WaveRamp {
            shape: WaveShape::Sawtooth,
            from: [0.4, 0.95],
            to: [0.0, 1.0],
        },
        WaveRamp {
            shape: WaveShape::Sawtooth,
            from: [0.12, 0.3],
            to: [1.0, 0.0],
        },
        WaveRamp {
            shape: WaveShape::Triangle,
            from: [0.07, 0.12],
            to: [1.0, 0.0],
        },
        WaveRamp {
            shape: WaveShape::Triangle,
            from: [0.2, 0.2],
            to: [-1.0, 2.0],
        },
        WaveRamp {
            shape: WaveShape::Sawtooth,
            from: [0.5, 0.5],
            to: [0.0, 1.0],
        },
    ];

    /// The ramp averaged over `[x − w/2, x + w/2]` by the midpoint rule.
    fn reference(ramp: &WaveRamp, x: f32, shift: f32, width: f32) -> f32 {
        let n = 40_000;
        let mut sum = 0.0_f64;
        for i in 0..n {
            let at = x - 0.5 * width + width * (i as f32 + 0.5) / n as f32;
            sum += f64::from(ramp.eval(at, shift, 0.0).0);
        }
        #[expect(clippy::cast_possible_truncation, reason = "a mean of f32 values")]
        let mean = (sum / f64::from(n)) as f32;
        mean
    }

    #[test]
    fn the_average_is_the_integral() {
        for ramp in RAMPS {
            for &x in &[0.0, 0.31, 0.95, 3.7, -2.2, 41.0] {
                for &shift in &[0.0, 0.03, -0.05] {
                    for &width in &[0.01, 0.2, 0.5, 1.0, 1.7, 6.3] {
                        let (value, _, _) = ramp.eval(x, shift, width);
                        let expected = reference(&ramp, x, shift, width);
                        assert!(
                            (value - expected).abs() < 2e-3,
                            "{ramp:?} at {x} shift {shift} width {width}: {value} vs {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn wide_footprints_fade_to_the_mean() {
        for ramp in RAMPS {
            let mean = ramp.eval(0.0, 0.0, f32::INFINITY).0;
            let expected = reference(&ramp, 0.5, 0.0, 1.0);
            assert!(
                (mean - expected).abs() < 1e-3,
                "{ramp:?}: {mean} vs {expected}"
            );
            for &x in &[0.0, 0.37, 12.5] {
                let wide = ramp.eval(x, 0.0, 400.0).0;
                assert!(
                    (wide - mean).abs() < 1e-3,
                    "{ramp:?} at {x}: {wide} vs {mean}"
                );
            }
        }
    }

    #[test]
    fn a_point_is_the_ramp_itself() {
        let late = RAMPS[0];
        assert_eq!(late.eval(0.2, 0.0, 0.0).0, 0.0);
        assert!((late.eval(0.675, 0.0, 0.0).0 - 0.5).abs() < 1e-6);
        assert_eq!(late.eval(1.97, 0.0, 0.0).0, 1.0);
        let stripe = RAMPS[2];
        assert_eq!(stripe.eval(3.02, 0.0, 0.0).0, 1.0);
        assert_eq!(stripe.eval(2.98, 0.0, 0.0).0, 1.0);
        assert_eq!(stripe.eval(3.5, 0.0, 0.0).0, 0.0);
        // A shift moves the threshold: a wider stripe.
        assert_eq!(stripe.eval(3.1, 0.05, 0.0).0, 1.0);
    }

    #[test]
    fn derivatives_match_differences() {
        for ramp in RAMPS {
            for &width in &[0.0, 0.05, 0.4, 2.5] {
                for &x in &[0.23, 0.71, 5.43] {
                    for &shift in &[0.0, 0.021] {
                        let (_, dx, ds) = ramp.eval(x, shift, width);
                        let h = 1e-3;
                        let fx = (ramp.eval(x + h, shift, width).0
                            - ramp.eval(x - h, shift, width).0)
                            / (2.0 * h);
                        let fs = (ramp.eval(x, shift + h, width).0
                            - ramp.eval(x, shift - h, width).0)
                            / (2.0 * h);
                        assert!((dx - fx).abs() < 2e-2 * (1.0 + fx.abs()), "{ramp:?} d/dx");
                        assert!((ds - fs).abs() < 2e-2 * (1.0 + fs.abs()), "{ramp:?} d/ds");
                    }
                }
            }
        }
    }
}
