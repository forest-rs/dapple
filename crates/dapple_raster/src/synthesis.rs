// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! By-example synthesis: Heitz and Neyret's histogram-preserving blending.
//!
//! [`ByExample`] tiles an exemplar image over any grid with no visible
//! repetition and no blurred seams, keeping the exemplar's histogram:
//!
//! 1. Each channel of the exemplar is **Gaussianized**: every texel is
//!    replaced by the standard normal quantile of its rank, so the
//!    channel's histogram becomes a standard Gaussian. The sorted original
//!    values are kept as the inverse lookup.
//! 2. The output is covered by a lattice of triangles, each vertex taking a
//!    patch of the Gaussianized exemplar at a random offset keyed by the
//!    vertex. A point blends the three patches of its triangle by its
//!    barycentric weights, **variance-preservingly**: `Σ wᵢ Gᵢ / √Σ wᵢ²`, so
//!    the blend is still standard Gaussian instead of washing out to the
//!    mean, as a linear blend would.
//! 3. The blend goes back through each channel's inverse lookup, which
//!    restores the exemplar's histogram.
//!
//! Channels are Gaussianized separately, which keeps each channel's
//! histogram but only approximately their correlations (the paper
//! decorrelates colors first; that is left for when a workload needs it).
//! The exemplar's texel size is its physical scale, so features keep their
//! size in domain units on any grid. On a wrapping grid the lattice fits a
//! whole number of cells per period and its vertices wrap, so the result
//! tiles. Offsets are keyed hashes of the vertex, so results do not depend
//! on evaluation order.
//!
//! Reference: Heitz and Neyret, *High-Performance By-Example Noise using a
//! Histogram-Preserving Blending Operator*, HPG 2018.

use alloc::vec::Vec;

use dapple_field::hash::{hash, unit_f32};
use dapple_field::image::bilinear;
use glam::Vec2;

use crate::{Edge, OpCategory, Raster, RasterError};

/// The standard normal quantile (Acklam's rational approximation, relative
/// error under `1.2e-9`), in `f64`.
fn normal_quantile(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    let tail = |q: f64| {
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    if p < 0.024_25 {
        tail(libm::sqrt(-2.0 * libm::log(p)))
    } else if p > 1.0 - 0.024_25 {
        -tail(libm::sqrt(-2.0 * libm::log(1.0 - p)))
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
}

/// The standard normal cumulative distribution.
fn normal_cdf(g: f64) -> f64 {
    0.5 * libm::erfc(-g / core::f64::consts::SQRT_2)
}

/// One Gaussianized channel and its inverse lookup.
struct Channel {
    gaussian: Vec<f32>,
    sorted: Vec<f32>,
}

impl Channel {
    fn new(values: Vec<f32>) -> Self {
        let n = values.len();
        let mut order: Vec<usize> = (0..n).collect();
        // Ties keep index order, so the ranks are deterministic.
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]).then(a.cmp(&b)));
        let mut gaussian = alloc::vec![0.0_f32; n];
        for (rank, &i) in order.iter().enumerate() {
            #[expect(clippy::cast_precision_loss, reason = "exemplar sizes fit f64")]
            let p = (rank as f64 + 0.5) / n as f64;
            #[expect(clippy::cast_possible_truncation, reason = "quantiles are small")]
            let g = normal_quantile(p) as f32;
            gaussian[i] = g;
        }
        let sorted = order.iter().map(|&i| values[i]).collect();
        Self { gaussian, sorted }
    }

    /// The value whose rank a standard Gaussian `g` has.
    fn inverse(&self, g: f32) -> f32 {
        let n = self.sorted.len();
        #[expect(clippy::cast_precision_loss, reason = "exemplar sizes fit f64")]
        let at = normal_cdf(f64::from(g)) * n as f64 - 0.5;
        let at = at.clamp(0.0, (n - 1) as f64);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "at is in [0, n - 1]"
        )]
        let i = libm::floor(at) as usize;
        let j = (i + 1).min(n - 1);
        #[expect(clippy::cast_possible_truncation, reason = "a fraction")]
        let t = (at - libm::floor(at)) as f32;
        self.sorted[i] + (self.sorted[j] - self.sorted[i]) * t
    }
}

/// Histogram-preserving by-example synthesis: see the [module docs](self).
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ByExample {
    /// The lattice cell, in domain units: the size of the patches blended.
    /// On a wrapping grid it is adjusted to fit the period.
    pub cell: f32,
    /// Keys the patch offsets.
    pub seed: u64,
}

impl ByExample {
    /// The op's category: a reduction (the histograms) and a pointwise map.
    #[must_use]
    pub const fn category(&self) -> OpCategory {
        OpCategory::GlobalTransform
    }

    /// Synthesizes `exemplar`'s texture over `grid`'s texels.
    ///
    /// The exemplar's texel size is its physical scale; its origin and edge
    /// policy are ignored. It must be at least three cells across, so every
    /// patch stays inside it.
    ///
    /// # Errors
    ///
    /// [`RasterError::InvalidParameter`] for a cell that is not finite and
    /// positive, or an exemplar too small for it.
    pub fn apply<const N: usize, T: Copy>(
        &self,
        exemplar: &Raster<[f32; N]>,
        grid: &Raster<T>,
    ) -> Result<Raster<[f32; N]>, RasterError> {
        if !(self.cell.is_finite() && self.cell > 0.0) {
            return Err(RasterError::InvalidParameter { name: "cell" });
        }
        let ex_texel = exemplar.texel();
        #[expect(clippy::cast_precision_loss, reason = "image sizes are small")]
        let ex_size = Vec2::new(exemplar.width() as f32, exemplar.height() as f32) * ex_texel;
        #[expect(clippy::cast_precision_loss, reason = "image sizes are small")]
        let extent = Vec2::new(grid.width() as f32, grid.height() as f32) * grid.texel();
        let wraps = grid.edge() == Edge::Wrap;
        // Whole cells per period on a wrapping grid.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "cell counts are small and positive"
        )]
        let count = |e: f32| (libm::roundf(e / self.cell).max(1.0)) as i64;
        let cells = [count(extent.x), count(extent.y)];
        #[expect(clippy::cast_precision_loss, reason = "cell counts are small")]
        let cell = if wraps {
            Vec2::new(extent.x / cells[0] as f32, extent.y / cells[1] as f32)
        } else {
            Vec2::splat(self.cell)
        };
        let margin = cell * 1.5;
        if (ex_size - 2.0 * margin).min_element() <= 0.0 {
            return Err(RasterError::InvalidParameter { name: "exemplar" });
        }
        let channels: Vec<Channel> = (0..N)
            .map(|c| Channel::new(exemplar.values().iter().map(|t| t[c]).collect()))
            .collect();
        let size = [exemplar.width(), exemplar.height()];
        let offset = |vx: i64, vy: i64| -> Vec2 {
            let (vx, vy) = if wraps {
                (vx.rem_euclid(cells[0]), vy.rem_euclid(cells[1]))
            } else {
                (vx, vy)
            };
            #[expect(clippy::cast_sign_loss, reason = "hashing the bits")]
            let words = [vx as u64, vy as u64];
            let h = hash(self.seed, &words);
            let u = Vec2::new(unit_f32(h), unit_f32(hash(h, &[1])));
            margin + u * (ex_size - 2.0 * margin)
        };
        let mut out = Vec::with_capacity(grid.values().len());
        for y in 0..grid.height() {
            for x in 0..grid.width() {
                #[expect(clippy::cast_precision_loss, reason = "texel indices are small")]
                let p = grid.texel() * Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let q = p / cell;
                let (fx, fy) = (libm::floorf(q.x), libm::floorf(q.y));
                let f = q - Vec2::new(fx, fy);
                #[expect(clippy::cast_possible_truncation, reason = "lattice indices are small")]
                let (i, j) = (fx as i64, fy as i64);
                let tri: [(i64, i64, f32); 3] = if f.x + f.y < 1.0 {
                    [(i, j, 1.0 - f.x - f.y), (i + 1, j, f.x), (i, j + 1, f.y)]
                } else {
                    [
                        (i + 1, j + 1, f.x + f.y - 1.0),
                        (i, j + 1, 1.0 - f.x),
                        (i + 1, j, 1.0 - f.y),
                    ]
                };
                let norm = libm::sqrtf(tri.iter().map(|t| t.2 * t.2).sum::<f32>());
                let mut texel = [0.0_f32; N];
                for (c, channel) in channels.iter().enumerate() {
                    let mut g = 0.0;
                    for &(vx, vy, w) in &tri {
                        #[expect(clippy::cast_precision_loss, reason = "lattice indices are small")]
                        let vertex = Vec2::new(vx as f32, vy as f32) * cell;
                        let at = (offset(vx, vy) + (p - vertex)) / ex_texel - 0.5;
                        g += w * bilinear(&channel.gaussian, size, Edge::Clamp, at);
                    }
                    texel[c] = channel.inverse(g / norm);
                }
                out.push(texel);
            }
        }
        Raster::from_values(
            grid.width(),
            grid.height(),
            grid.origin(),
            grid.texel(),
            grid.edge(),
            out,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_invert_the_cdf() {
        for p in [0.001, 0.02, 0.3, 0.5, 0.8, 0.999] {
            assert!((normal_cdf(normal_quantile(p)) - p).abs() < 1e-8, "{p}");
        }
    }

    #[test]
    fn synthesis_keeps_the_exemplars_histogram() {
        // A skewed exemplar: squared hash noise in blocks of 8 × 8 texels
        // (features, not texel noise, which any resampling would average),
        // 128 texels of 5 mm.
        let n = 128_u32;
        let values: Vec<[f32; 1]> = (0..n * n)
            .map(|i| {
                let block = (i % n) / 8 + 1000 * ((i / n) / 8);
                let u = unit_f32(hash(3, &[u64::from(block)]));
                [u * u]
            })
            .collect();
        let exemplar =
            Raster::from_values(n, n, Vec2::ZERO, Vec2::splat(0.005), Edge::Clamp, values).unwrap();
        let grid = Raster::from_values(
            128,
            128,
            Vec2::ZERO,
            Vec2::splat(1.0 / 128.0),
            Edge::Wrap,
            alloc::vec![(); 128 * 128],
        )
        .unwrap();
        let op = ByExample { cell: 0.1, seed: 9 };
        let out = op.apply(&exemplar, &grid).unwrap();
        let mean = |v: &[[f32; 1]]| v.iter().map(|t| f64::from(t[0])).sum::<f64>() / v.len() as f64;
        let (a, b) = (mean(exemplar.values()), mean(out.values()));
        // A linear blend would pull the mean of u² toward 1/3 and shrink
        // the spread; the histogram-preserving blend keeps both.
        assert!((a - b).abs() < 0.03, "means {a} and {b}");
        let spread = |v: &[[f32; 1]], m: f64| {
            libm::sqrt(
                v.iter().map(|t| (f64::from(t[0]) - m).powi(2)).sum::<f64>() / v.len() as f64,
            )
        };
        let (sa, sb) = (spread(exemplar.values(), a), spread(out.values(), b));
        assert!((sa - sb).abs() < 0.03, "spreads {sa} and {sb}");
        assert_eq!(out, op.apply(&exemplar, &grid).unwrap(), "deterministic");
        assert!(
            ByExample { cell: 0.3, seed: 9 }
                .apply(&exemplar, &grid)
                .is_err(),
            "a cell too large for the exemplar"
        );
    }
}
