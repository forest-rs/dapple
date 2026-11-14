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
//! [`seam`] measures, along one axis and on every line, the step between
//! each pair of neighbors and the change of step at each texel (a kink, as
//! where a wave's phase jumps), and compares those across the wrap with the
//! **roughest** position inside. Periodic features that happen to line up
//! with the wrap (a mortar joint on the tile's edge, cell borders crowding
//! a lattice line) line up with interior positions as well, so the
//! comparison does not mistake them for seams; a seam is a wrap rougher
//! than anything in the interior.
//!
//! Each position is summed up over the lines twice: by the mean, and by
//! the median line. A seam runs the length of the wrap, so it shows in
//! both. A feature that merely lies on the wrap (a paint chip, a fleck of
//! bare metal, a streak) crosses only some lines: it can make the wrap the
//! roughest position on average, since a sparse feature may have no twin
//! inside and one a texel wide changes the step at its texel by twice its
//! height, but it cannot move the median line. So the wrap counts as only
//! as much rougher as both comparisons agree, the smaller of the two, and
//! [`Seam::is_seamless`] allows up to [`SEAM_TOLERANCE`].
//!
//! This is a raster-only test: it catches gross seams that run along most
//! of the wrap and never cries wolf on features that cross under half of
//! its lines, but a subtle seam no rougher than the interior passes, as
//! does one confined to a minority of lines. A lone feature that crosses
//! most of the lines and is rougher than anything inside (a crease the
//! height of the tile, or the only high-contrast course boundary in it) is
//! indistinguishable from a seam when it lies on the wrap. Where the
//! content is a formula, evaluating it across the wrap is exact; see
//! `dapple_material::program`. Values are compared per component by
//! absolute difference; identifiers by whether they differ.

use alloc::vec::Vec;

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
    /// How many times rougher the wrap is than the roughest interior
    /// position; 1 means no rougher, and 1 where all are zero.
    ///
    /// The step and the change of step are each compared twice, by the
    /// mean over lines and by the median line, and count as far as both
    /// agree (the smaller ratio); this is the larger of the two measures.
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
    // Per position, over the lines: the step to the next texel (position
    // `len − 1` is the wrap) and the change of step at the texel (positions
    // 0 and `len − 1` straddle the wrap).
    let size = usize::try_from(len).expect("raster sizes fit usize");
    let count = usize::try_from(lines).expect("raster sizes fit usize");
    let (mut steps, mut bends) = (Vec::with_capacity(count), Vec::with_capacity(count));
    let (mut step, mut bend) = (Vec::with_capacity(size), Vec::with_capacity(size));
    for k in 0..len {
        steps.clear();
        bends.clear();
        for line in 0..lines {
            let (a, b, c) = (at(line, k - 1), at(line, k), at(line, k + 1));
            steps.push(b.distance(c));
            bends.push(T::bend(a, b, c));
        }
        step.push(Lines::of(&mut steps));
        bend.push(Lines::of(&mut bends));
    }
    let last = size - 1;
    let roughest = |p: &[Lines]| p.iter().map(|l| l.sum).fold(0.0, f64::max);
    let (wrap, interior) = (step[last].sum, roughest(&step[..last]));
    let ratio =
        compare(&step[last..], &step[..last]).max(compare(&[bend[0], bend[last]], &bend[1..last]));
    #[expect(clippy::cast_precision_loss, reason = "line counts fit f64")]
    let n = lines as f64;
    #[expect(clippy::cast_possible_truncation, reason = "means of f32 steps")]
    Seam {
        axis,
        wrap_step: (wrap / n) as f32,
        interior_step: (interior / n) as f32,
        ratio,
    }
}

/// One measure at one position, summed up over the lines.
#[derive(Copy, Clone, Debug)]
struct Lines {
    /// The sum over the lines, in line order: the mean, times the count.
    sum: f64,
    /// The median line: what more than half of the lines reach.
    median: f64,
}

impl Lines {
    /// Sums up `values`, one per line (reordering them).
    fn of(values: &mut [f32]) -> Self {
        let sum = values.iter().fold(0.0, |s, v| s + f64::from(*v));
        let middle = (values.len() - 1) / 2;
        let (_, median, _) = values.select_nth_unstable_by(middle, f32::total_cmp);
        Self {
            sum,
            median: f64::from(*median),
        }
    }
}

/// How many times rougher the roughest of the `wrap` positions is than the
/// roughest of the `interior` ones, as far as the mean over lines and the
/// median line agree.
fn compare(wrap: &[Lines], interior: &[Lines]) -> f32 {
    let by = |stat: fn(&Lines) -> f64| {
        let roughest = |p: &[Lines]| p.iter().map(stat).fold(0.0, f64::max);
        let (w, i) = (roughest(wrap), roughest(interior));
        #[expect(clippy::cast_possible_truncation, reason = "a ratio of sums")]
        if i > 1e-12 {
            (w / i) as f32
        } else if w > 1e-12 {
            f32::INFINITY
        } else {
            1.0
        }
    };
    by(|l| l.sum).min(by(|l| l.median))
}

#[cfg(test)]
mod tests {
    use alloc::boxed::Box;
    use alloc::vec::Vec;
    use core::f32::consts::TAU;

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

    /// Content of period `n` texels along both axes, at `(x, y)` in texels.
    type Content = Box<dyn Fn(f32, f32) -> f32>;

    /// A pseudo-random number in [0, 1) for `(seed, i)`.
    fn unit(seed: u64, i: u64) -> f32 {
        let mut x = seed
            .wrapping_mul(0x1000_0001)
            .wrapping_add(i)
            .wrapping_add(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^= x >> 31;
        (x >> 40) as f32 / (1_u64 << 24) as f32
    }

    /// The offset from `b` to `a` on a circle of circumference `n`, in
    /// [−n/2, n/2).
    fn around(a: f32, b: f32, n: f32) -> f32 {
        let mut d = (a - b) % n;
        if d < -n / 2.0 {
            d += n;
        }
        if d >= n / 2.0 {
            d -= n;
        }
        d
    }

    /// Periodic content of the family `name` with period `n`; `seed`
    /// picks the instance. Features are placed anywhere, so some straddle
    /// the wrap.
    fn family(name: &str, n: f32, seed: u64) -> Content {
        use libm::{floorf, sinf};
        let u = move |i: u64| unit(seed, i);
        let count = |prefix: &str| -> u64 { name[prefix.len()..].parse().expect("a count") };
        match name {
            // Smooth noise: a few waves of whole cycles per period.
            "smooth" => Box::new(move |x, y| {
                (0..6)
                    .map(|i| {
                        let kx = floorf(u(i * 5) * 7.0) - 3.0;
                        let ky = floorf(u(i * 5 + 1) * 7.0) - 3.0;
                        u(i * 5 + 2) * sinf(TAU * (kx * x + ky * y) / n + u(i * 5 + 3) * TAU)
                    })
                    .sum()
            }),
            // Value noise on a periodic lattice of 4 to 9 cells.
            "value" => {
                let l = 4.0 + floorf(u(99) * 6.0);
                Box::new(move |x, y| {
                    let (fx, fy) = (x / n * l, y / n * l);
                    let (col, row) = (floorf(fx), floorf(fy));
                    let ease = |t: f32| t * t * (3.0 - 2.0 * t);
                    let (tx, ty) = (ease(fx - col), ease(fy - row));
                    #[expect(clippy::cast_possible_truncation, reason = "small lattice indices")]
                    let at = |i: f32, j: f32| u((i % l) as u64 * 131 + (j % l) as u64);
                    let a = at(col, row) + (at(col + 1.0, row) - at(col, row)) * tx;
                    let next = row + 1.0;
                    let b = at(col, next) + (at(col + 1.0, next) - at(col, next)) * tx;
                    a + (b - a) * ty
                })
            }
            // Paint chips: rectangles 1 to 4 texels across, a rim around a
            // brighter inside, on a flat background.
            _ if name.starts_with("chips") => {
                let k = count("chips");
                Box::new(move |x, y| {
                    let mut v = 0.2;
                    for i in 0..k {
                        let (dx, dy) = (around(x, u(i * 7) * n, n), around(y, u(i * 7 + 1) * n, n));
                        let w = 1.0 + floorf(u(i * 7 + 2) * 4.0);
                        let h = 1.0 + floorf(u(i * 7 + 3) * 4.0);
                        if dx.abs() < w / 2.0 && dy.abs() < h / 2.0 {
                            v = if dx.abs() < w / 2.0 - 1.0 && dy.abs() < h / 2.0 - 1.0 {
                                0.9
                            } else {
                                0.5 + 0.3 * u(i * 7 + 4)
                            };
                        }
                    }
                    v
                })
            }
            // Flecks: single texels of partial metalness on none.
            _ if name.starts_with("specks") => {
                let k = count("specks");
                Box::new(move |x, y| {
                    let mut v = 0.0;
                    for i in 0..k {
                        let cx = floorf(u(i * 7) * n) + 0.5;
                        let cy = floorf(u(i * 7 + 1) * n) + 0.5;
                        if around(x, cx, n).abs() < 0.5 && around(y, cy, n).abs() < 0.5 {
                            v = 0.05 + 0.3 * u(i * 7 + 2);
                        }
                    }
                    v
                })
            }
            // Smooth bumps 1 to 6 texels in radius.
            _ if name.starts_with("blobs") => {
                let k = count("blobs");
                Box::new(move |x, y| {
                    (0..k)
                        .map(|i| {
                            let (dx, dy) =
                                (around(x, u(i * 7) * n, n), around(y, u(i * 7 + 1) * n, n));
                            let r = 1.0 + u(i * 7 + 2) * 5.0;
                            let t = (1.0 - (dx * dx + dy * dy) / (r * r)).max(0.0);
                            t * t
                        })
                        .sum()
                })
            }
            // Hard-edged cells: the nearest of `k` seeds' flat values.
            _ if name.starts_with("cells") => {
                let k = count("cells");
                Box::new(move |x, y| {
                    let mut best = (f32::MAX, 0);
                    for i in 0..k {
                        let (dx, dy) = (around(x, u(i * 3) * n, n), around(y, u(i * 3 + 1) * n, n));
                        if dx * dx + dy * dy < best.0 {
                            best = (dx * dx + dy * dy, i);
                        }
                    }
                    u(best.1 * 3 + 2)
                })
            }
            // Hard stripes across x, across y, and slanting across both.
            "stripes_x" | "stripes_y" => {
                let k = 2.0 + floorf(u(5) * 5.0);
                let across_x = name == "stripes_x";
                Box::new(move |x, y| {
                    let t = if across_x { x } else { y } / n * k;
                    if t - floorf(t) < 0.4 { 1.0 } else { 0.0 }
                })
            }
            "stripes_diag" => {
                let (a, b) = (1.0 + floorf(u(5) * 3.0), 1.0 + floorf(u(6) * 3.0));
                Box::new(move |x, y| {
                    let t = (a * x + b * y) / n;
                    if t - floorf(t) < 0.5 { 1.0 } else { 0.3 }
                })
            }
            // A slanting wave.
            "waves" => {
                let (a, b) = (1.0 + floorf(u(5) * 3.0), -1.0 - floorf(u(6) * 3.0));
                Box::new(move |x, y| sinf(TAU * (a * x + b * y) / n))
            }
            // A single feature on a flat ground: a rectangle, or one texel.
            "rect" => {
                let (cx, cy) = (u(1) * n, u(2) * n);
                let (w, h) = (2.0 + u(3) * 6.0, 2.0 + u(4) * 6.0);
                Box::new(move |x, y| {
                    let inside =
                        around(x, cx, n).abs() < w / 2.0 && around(y, cy, n).abs() < h / 2.0;
                    if inside { 1.0 } else { 0.3 }
                })
            }
            "dot" => {
                let (cx, cy) = (floorf(u(1) * n) + 0.5, floorf(u(2) * n) + 0.5);
                Box::new(move |x, y| {
                    let inside = around(x, cx, n).abs() < 0.5 && around(y, cy, n).abs() < 0.5;
                    if inside { 1.0 } else { 0.0 }
                })
            }
            // Binary masks: value noise above a level, in percent.
            _ if name.starts_with("mask") => {
                #[expect(clippy::cast_precision_loss, reason = "a percentage")]
                let level = count("mask") as f32 / 100.0;
                let noise = family("value", n, seed);
                Box::new(move |x, y| if noise(x, y) > level { 1.0 } else { 0.0 })
            }
            // Dirt streaks: shallow runs down the surface, 1 to 2.5 texels
            // wide and 30% to 70% of the period long.
            _ if name.starts_with("streaks") => {
                let k = count("streaks");
                Box::new(move |x, y| {
                    0.9 - (0..k)
                        .map(|i| {
                            let (dx, dy) =
                                (around(x, u(i * 7) * n, n), around(y, u(i * 7 + 1) * n, n));
                            let half = n * (0.15 + 0.2 * u(i * 7 + 2));
                            let w = 0.7 + u(i * 7 + 3) * 1.5;
                            let along = (1.0 - (dy / half) * (dy / half)).max(0.0);
                            let across = (1.0 - (dx / w) * (dx / w)).max(0.0);
                            (0.02 + 0.03 * u(i * 7 + 4)) * along * across
                        })
                        .sum::<f32>()
                })
            }
            _ => panic!("no family {name}"),
        }
    }

    /// `content` sampled at the texel centers of an `n` × `n` period.
    fn sampled(n: u32, content: impl Fn(f32, f32) -> f32) -> Raster {
        let mut v = Vec::new();
        for y in 0..n {
            for x in 0..n {
                v.push(content(x as f32 + 0.5, y as f32 + 0.5));
            }
        }
        let texel = Vec2::splat(1.0 / n as f32);
        Raster::from_values(n, n, Vec2::ZERO, texel, Edge::Wrap, v).unwrap()
    }

    /// A seam to inject along an axis, at a strength relative to the
    /// content's range.
    #[derive(Copy, Clone, Debug)]
    enum Injected {
        /// A ramp across the period: a cliff of its height at the wrap.
        Ramp,
        /// An offset faded in over the last eighth of the period: a cliff
        /// at the wrap that the interior approaches smoothly.
        Edge,
        /// The content generated with a period longer by this share.
        Period,
        /// 2.5 cycles of a wave added: its phase reverses at the wrap.
        HalfCycle,
    }

    /// The family `name` with period `n` and `seam` injected along `axis`
    /// at `strength`.
    fn injected(
        name: &str,
        n: u32,
        seed: u64,
        seam: Injected,
        axis: Axis,
        strength: f32,
    ) -> Raster {
        let size = n as f32;
        let base = family(name, size, seed);
        let plain = sampled(n, &base);
        let (lo, hi) = plain
            .values()
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
        let height = strength * (hi - lo);
        let along = |x: f32, y: f32| if axis == Axis::X { x } else { y };
        match seam {
            Injected::Ramp => sampled(n, |x, y| base(x, y) + height * along(x, y) / size),
            Injected::Edge => sampled(n, |x, y| {
                let fade = (1.0 - (size - 0.5 - along(x, y)) / (size / 8.0)).max(0.0);
                base(x, y) + height * fade
            }),
            Injected::Period => sampled(n, |x, y| {
                let s = 1.0 + strength;
                if axis == Axis::X {
                    base(x * s, y)
                } else {
                    base(x, y * s)
                }
            }),
            Injected::HalfCycle => sampled(n, |x, y| {
                base(x, y) + height * libm::sinf(TAU * 2.5 * along(x, y) / size)
            }),
        }
    }

    /// Seams are reported on every kind of content, at strengths short of
    /// glaring: a ramp or a faded offset of a tenth of the content's range
    /// at 64 texels, a period 5% too long. Sparse or hard-edged content
    /// hides a seam among its own edges, so it needs a stronger one.
    #[test]
    fn seams_are_reported_on_every_kind_of_content() {
        use Injected::{Edge as Faded, HalfCycle, Period, Ramp};
        let reported = |name: &str, n: u32, seams: &[(Injected, f32)]| {
            for &(injected_seam, strength) in seams {
                for seed in 1..=4 {
                    for axis in [Axis::X, Axis::Y] {
                        let r = injected(name, n, seed, injected_seam, axis, strength);
                        let s = seam(&r, axis);
                        assert!(
                            !s.is_seamless(),
                            "{name} #{seed} at {n}: {injected_seam:?} {strength} along {axis:?} missed: {s:?}"
                        );
                    }
                }
            }
        };
        let smooth = [(Ramp, 0.1), (Faded, 0.1), (Period, 0.05), (HalfCycle, 0.25)];
        reported("smooth", 64, &smooth);
        let value = [(Ramp, 0.1), (Faded, 0.1), (Period, 0.1), (HalfCycle, 0.5)];
        reported("value", 64, &value);
        reported("waves", 64, &[(Ramp, 0.1), (Faded, 0.1)]);
        reported("blobs6", 64, &[(Ramp, 0.2), (Faded, 0.2)]);
        reported("specks6", 64, &[(Ramp, 0.1), (Faded, 0.1)]);
        reported("stripes_diag", 64, &[(Ramp, 0.15), (Faded, 0.15)]);
        reported("chips8", 64, &[(Ramp, 0.4), (Faded, 0.4)]);
        reported("cells12", 64, &[(Ramp, 0.4), (Faded, 0.3)]);
        reported("mask50", 64, &[(Ramp, 1.0), (Faded, 1.0)]);
        let smooth = [(Ramp, 0.2), (Faded, 0.15), (Period, 0.2), (HalfCycle, 0.5)];
        reported("smooth", 32, &smooth);
        reported("value", 32, &[(Ramp, 0.3), (Faded, 0.3), (Period, 0.2)]);
        reported("waves", 32, &[(Ramp, 0.4), (Faded, 0.3)]);
        reported("blobs6", 32, &[(Ramp, 0.3), (Faded, 0.3)]);
        reported("specks6", 32, &[(Ramp, 0.2), (Faded, 0.2)]);
        reported("stripes_diag", 32, &[(Ramp, 0.3), (Faded, 0.4)]);
        reported("chips8", 32, &[(Ramp, 0.75), (Faded, 0.75)]);
        reported("cells12", 32, &[(Ramp, 0.5), (Faded, 0.5)]);
        reported("mask50", 32, &[(Ramp, 1.5), (Faded, 1.5)]);
    }

    /// Periodic content is seamless under every cyclic shift along either
    /// axis: periodic content is the same content wherever the period
    /// starts, so no choice of start may fail it. Among the shifts are the
    /// ones that put a chip's edge, a fleck one texel wide or a streak's
    /// center on the wrap, which the mean over lines alone took for seams.
    ///
    /// Hard-edged cells are checked at 32 texels and up: on coarser grids,
    /// or with denser cells, borders can bunch along the texels beside the
    /// wrap across more than half the lines, which the kink measure cannot
    /// tell from a crease.
    #[test]
    fn periodic_content_is_seamless_under_every_cyclic_shift() {
        let holds = |name: &str, n: u32, seed: u64| {
            let r = sampled(n, family(name, n as f32, seed));
            for s in 0..i64::from(n) {
                for (axis, rolled) in [(Axis::X, r.rolled(s, 0)), (Axis::Y, r.rolled(0, s))] {
                    let m = seam(&rolled, axis);
                    assert!(
                        m.is_seamless(),
                        "{name} #{seed} at {n}, started {s} texels on along {axis:?}: {m:?}"
                    );
                }
            }
        };
        let local = [
            "chips1", "chips2", "chips4", "specks1", "specks2", "specks6",
        ];
        let single = ["dot", "rect", "blobs1", "blobs6", "streaks1"];
        let periodic = [
            "smooth",
            "value",
            "waves",
            "stripes_x",
            "stripes_y",
            "stripes_diag",
        ];
        let masks = ["mask20", "mask50", "mask70"];
        for n in [16, 24, 32, 48, 64] {
            for name in local.iter().chain(&single).chain(&periodic).chain(&masks) {
                for seed in 1..=3 {
                    holds(name, n, seed);
                }
            }
        }
        for n in [24, 32, 48, 64] {
            for seed in 1..=3 {
                holds("streaks3", n, seed);
            }
        }
        for n in [32, 48, 64] {
            for name in ["cells4", "cells12"] {
                for seed in [1, 3] {
                    holds(name, n, seed);
                }
            }
        }
    }
}
