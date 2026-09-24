// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Seam measurement: whether a raster that claims to tile does.
//!
//! A raster over one period of a periodic domain is a torus: its last row
//! is followed by its first. If the content really is periodic, the step
//! across that wrap is like the steps elsewhere; a seam (a feature whose
//! frequency does not divide the period, a gradient from the foot of a wall
//! to its top) makes it stand out.
//!
//! [`seam`] measures, along one axis and averaged over every line, the step
//! between each pair of neighbors and the change of step at each texel (a
//! kink, as where a wave's phase jumps), and compares those across the wrap
//! with the **roughest** position inside. Periodic features that happen to
//! line up with the wrap (a mortar joint on the tile's edge, cell borders
//! crowding a lattice line) line up with interior positions as well, so the
//! comparison does not mistake them for seams; a seam is a wrap rougher
//! than anything in the interior. [`Seam::is_seamless`] allows up to
//! [`SEAM_TOLERANCE`] times the roughest interior position.
//!
//! This is a raster-only test: it catches gross seams and never cries wolf
//! on periodic content, but a subtle seam no rougher than the interior
//! passes. Where the content is a formula, evaluating it across the wrap is
//! exact; see `dapple_material::program`. Values are compared per component
//! by absolute difference; identifiers by whether they differ.

use crate::Raster;

/// The largest ratio of the wrap to the roughest interior position
/// [`Seam::is_seamless`] accepts.
pub const SEAM_TOLERANCE: f32 = 1.5;

/// An axis of a raster.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Axis {
    /// Across columns: the wrap from the last column to the first.
    X,
    /// Across rows: the wrap from the last row to the first.
    Y,
}

/// What [`seam`] measured along one axis.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Seam {
    /// The axis.
    pub axis: Axis,
    /// The mean step across the wrap.
    pub wrap_step: f32,
    /// The mean step at the roughest interior position.
    pub interior_step: f32,
    /// The larger of the wrap's step and change of step over those of the
    /// roughest interior positions; 1 where all are zero.
    pub ratio: f32,
}

impl Seam {
    /// Whether the wrap is no rougher than the interior, within
    /// [`SEAM_TOLERANCE`].
    #[must_use]
    pub fn is_seamless(&self) -> bool {
        self.ratio <= SEAM_TOLERANCE
    }
}

/// A texel value [`seam`] can compare.
pub trait SeamValue: Copy {
    /// The distance between two values.
    fn distance(self, other: Self) -> f32;

    /// The size of the second difference `a − 2b + c`: how much the step
    /// changes at `b`.
    fn bend(a: Self, b: Self, c: Self) -> f32;
}

impl SeamValue for f32 {
    fn distance(self, other: Self) -> f32 {
        (self - other).abs()
    }

    fn bend(a: Self, b: Self, c: Self) -> f32 {
        (a - 2.0 * b + c).abs()
    }
}

impl SeamValue for u32 {
    fn distance(self, other: Self) -> f32 {
        f32::from(u8::from(self != other))
    }

    fn bend(_: Self, _: Self, _: Self) -> f32 {
        0.0
    }
}

impl<const N: usize> SeamValue for [f32; N] {
    fn distance(self, other: Self) -> f32 {
        self.iter().zip(other).map(|(a, b)| (a - b).abs()).sum()
    }

    fn bend(a: Self, b: Self, c: Self) -> f32 {
        (0..N).map(|i| (a[i] - 2.0 * b[i] + c[i]).abs()).sum()
    }
}

/// Measures the seam of `r` along `axis`: see the [module docs](self).
///
/// Rasters under four texels along the axis have no interior to compare
/// and measure a ratio of 1.
#[must_use]
pub fn seam<T: SeamValue>(r: &Raster<T>, axis: Axis) -> Seam {
    let (w, h) = (i64::from(r.width()), i64::from(r.height()));
    let (lines, len) = match axis {
        Axis::X => (h, w),
        Axis::Y => (w, h),
    };
    let at = |line: i64, k: i64| match axis {
        Axis::X => r.at(k.rem_euclid(len), line),
        Axis::Y => r.at(line, k.rem_euclid(len)),
    };
    if len < 4 {
        return Seam {
            axis,
            wrap_step: 0.0,
            interior_step: 0.0,
            ratio: 1.0,
        };
    }
    // Per position, averaged over the lines: the step to the next texel
    // (position `len − 1` is the wrap) and the change of step at the texel
    // (positions 0 and `len − 1` straddle the wrap).
    let size = usize::try_from(len).expect("raster sizes fit usize");
    let mut step = alloc::vec![0.0_f64; size];
    let mut bend = alloc::vec![0.0_f64; size];
    for line in 0..lines {
        for (k, (s, b2)) in (0..len).zip(step.iter_mut().zip(bend.iter_mut())) {
            let (a, b, c) = (at(line, k - 1), at(line, k), at(line, k + 1));
            *s += f64::from(b.distance(c));
            *b2 += f64::from(T::bend(a, b, c));
        }
    }
    let last = size - 1;
    let wrap = step[last];
    let wrap2 = bend[0].max(bend[last]);
    let interior = step[..last].iter().copied().fold(0.0, f64::max);
    let interior2 = bend[1..last].iter().copied().fold(0.0, f64::max);
    let ratio_of = |w: f64, i: f64| -> f32 {
        #[expect(clippy::cast_possible_truncation, reason = "a ratio of sums")]
        if i > 1e-12 {
            (w / i) as f32
        } else if w > 1e-12 {
            f32::INFINITY
        } else {
            1.0
        }
    };
    #[expect(clippy::cast_precision_loss, reason = "line counts fit f64")]
    let n = lines as f64;
    #[expect(clippy::cast_possible_truncation, reason = "means of f32 steps")]
    Seam {
        axis,
        wrap_step: (wrap / n) as f32,
        interior_step: (interior / n) as f32,
        ratio: ratio_of(wrap, interior).max(ratio_of(wrap2, interior2)),
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use glam::Vec2;

    use super::*;
    use crate::Edge;

    fn raster(f: impl Fn(u32, u32) -> f32) -> Raster {
        let n = 64;
        let mut v = Vec::new();
        for y in 0..n {
            for x in 0..n {
                v.push(f(x, y));
            }
        }
        Raster::from_values(n, n, Vec2::ZERO, Vec2::splat(1.0 / 64.0), Edge::Wrap, v).unwrap()
    }

    #[test]
    fn periodic_content_is_seamless_and_mismatched_content_is_not() {
        use core::f32::consts::TAU;
        // Four whole cycles per period along x; 4.5 along y.
        let r = raster(|x, y| {
            libm::sinf(TAU * 4.0 * x as f32 / 64.0) + libm::sinf(TAU * 4.5 * y as f32 / 64.0)
        });
        assert!(seam(&r, Axis::X).is_seamless(), "{:?}", seam(&r, Axis::X));
        assert!(!seam(&r, Axis::Y).is_seamless(), "{:?}", seam(&r, Axis::Y));
        let gradient = raster(|_, y| y as f32);
        assert!(
            !seam(&gradient, Axis::Y).is_seamless(),
            "a foot-to-top gradient"
        );
    }
}
