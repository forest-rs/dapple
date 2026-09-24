// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Measurements on rasters: value statistics and feature size.

use dapple_raster::Raster;

/// Statistics of a stream of values.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Stats {
    /// The smallest finite value.
    pub min: f64,
    /// The largest finite value.
    pub max: f64,
    /// The mean of the finite values.
    pub mean: f64,
    /// Values that were NaN or infinite.
    pub non_finite: u64,
    /// Finite values counted.
    pub count: u64,
}

impl Stats {
    /// The statistics of `values`.
    pub fn of(values: impl IntoIterator<Item = f32>) -> Self {
        let mut s = Self {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            mean: 0.0,
            non_finite: 0,
            count: 0,
        };
        let mut sum = 0.0;
        for v in values {
            if v.is_finite() {
                let v = f64::from(v);
                s.min = s.min.min(v);
                s.max = s.max.max(v);
                sum += v;
                s.count += 1;
            } else {
                s.non_finite += 1;
            }
        }
        if s.count > 0 {
            #[expect(clippy::cast_precision_loss, reason = "texel counts fit f64")]
            let n = s.count as f64;
            s.mean = sum / n;
        } else {
            (s.min, s.max) = (0.0, 0.0);
        }
        s
    }
}

/// A resolution-independent feature size of a scalar raster, in domain
/// units: `√(var f / mean |∇f|²)`, the correlation length of a smooth
/// random field. Realizing the same field at a finer resolution changes it
/// only as far as the finer texels resolve detail the coarser ones could
/// not, so it tests that physical feature size stays put.
///
/// A constant raster has no features and measures 0.
#[must_use]
pub fn feature_size(r: &Raster) -> f64 {
    let (w, h) = (i64::from(r.width()), i64::from(r.height()));
    let t = r.texel();
    let (mut sum, mut sum2, mut grad2) = (0.0_f64, 0.0_f64, 0.0_f64);
    for y in 0..h {
        for x in 0..w {
            let v = f64::from(r.at(x, y));
            sum += v;
            sum2 += v * v;
            let gx = f64::from(r.at(x + 1, y) - r.at(x - 1, y)) / (2.0 * f64::from(t.x));
            let gy = f64::from(r.at(x, y + 1) - r.at(x, y - 1)) / (2.0 * f64::from(t.y));
            grad2 += gx * gx + gy * gy;
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "texel counts fit f64")]
    let n = (w * h) as f64;
    let var = (sum2 / n - (sum / n) * (sum / n)).max(0.0);
    let grad2 = grad2 / n;
    if grad2 > 0.0 {
        libm::sqrt(var / grad2)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use dapple_field::{Basis, Domain, Fractal, FractalParams};
    use dapple_raster::{Realization, realize};
    use glam::Vec2;

    use super::*;

    #[test]
    fn feature_size_does_not_move_with_resolution() {
        let domain = Domain::periodic(1, 1).unwrap();
        let fbm = Fractal::new(
            Basis::Gradient,
            domain,
            Vec2::splat(8.0),
            1,
            FractalParams {
                octaves: 2,
                ..FractalParams::default()
            },
        )
        .unwrap();
        let sizes: Vec<f64> = [128, 256, 512]
            .iter()
            .map(|&n| {
                feature_size(&realize(&fbm, Realization::period(domain, n, n).unwrap()).unwrap())
            })
            .collect();
        for s in &sizes[1..] {
            assert!((s / sizes[0] - 1.0).abs() < 0.05, "{sizes:?}");
        }
        let s = Stats::of([1.0, f32::NAN, 3.0]);
        assert_eq!((s.min, s.max, s.mean, s.non_finite), (1.0, 3.0, 2.0, 1));
    }
}
