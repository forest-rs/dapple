// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Exact Euclidean distance transform.

use alloc::vec::Vec;

use glam::Vec2;

use crate::{Edge, Raster, RasterError, RasterOp};

/// Distance in domain units from each texel center to the nearest *feature*
/// texel center, where features are texels with value `>= threshold`.
///
/// The transform is exact Euclidean, including for non-square texels, using
/// the separable lower-envelope algorithm of Felzenszwalb and Huttenlocher in
/// `f64`. On a wrapping raster distances are measured on the torus: each 1D
/// pass runs over three copies of its line and keeps the middle one, which is
/// exact because the nearest feature on a torus is within one period. A
/// raster without features is `f32::INFINITY` everywhere.
///
/// The footprint is unbounded: any feature can be the nearest.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DistanceTransform {
    /// Values at or above this are features.
    pub threshold: f32,
}

impl RasterOp for DistanceTransform {
    type Output = f32;

    fn category(&self) -> crate::OpCategory {
        crate::OpCategory::SeparablePass
    }

    fn footprint(&self, _texel: Vec2) -> Option<[u32; 2]> {
        None
    }

    fn apply(&self, input: &Raster) -> Result<Raster, RasterError> {
        if self.threshold.is_nan() {
            return Err(RasterError::InvalidParameter { name: "threshold" });
        }
        let (w, h) = (input.width() as usize, input.height() as usize);
        let texel = input.texel();
        let wrap = input.edge() == Edge::Wrap;
        // Squared distances along each column first, then along each row.
        let mut squared: Vec<f64> = input
            .values()
            .iter()
            .map(|v| {
                if *v >= self.threshold {
                    0.0
                } else {
                    f64::INFINITY
                }
            })
            .collect();
        let mut line = Vec::new();
        let mut out = Vec::new();
        for x in 0..w {
            line.clear();
            line.extend((0..h).map(|y| squared[y * w + x]));
            transform_1d(&line, f64::from(texel.y), wrap, &mut out);
            for (y, d) in out.iter().enumerate() {
                squared[y * w + x] = *d;
            }
        }
        for y in 0..h {
            line.clear();
            line.extend_from_slice(&squared[y * w..(y + 1) * w]);
            transform_1d(&line, f64::from(texel.x), wrap, &mut out);
            squared[y * w..(y + 1) * w].copy_from_slice(&out);
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "distances are reported in f32"
        )]
        Ok(input.map_texels(|x, y| {
            let i = usize::try_from(y).expect("row") * w + usize::try_from(x).expect("column");
            libm::sqrt(squared[i]) as f32
        }))
    }
}

/// 1D squared-distance transform of `f` with sample spacing `step`:
/// `out[q] = min_p f[p] + ((q - p) * step)²`.
fn transform_1d(f: &[f64], step: f64, wrap: bool, out: &mut Vec<f64>) {
    let n = f.len();
    let (samples, offset) = if wrap { (3 * n, n) } else { (n, 0) };
    let value = |i: usize| f[i % n];
    // Lower envelope of the parabolas rooted at finite samples.
    let mut roots: Vec<usize> = Vec::with_capacity(samples);
    let mut bounds: Vec<f64> = Vec::with_capacity(samples + 1);
    for q in (0..samples).filter(|&q| value(q).is_finite()) {
        let pos = |i: usize| i as f64 * step;
        loop {
            let Some(&p) = roots.last() else {
                roots.push(q);
                bounds.clear();
                bounds.push(f64::NEG_INFINITY);
                break;
            };
            let (qs, ps) = (pos(q), pos(p));
            let s = ((value(q) + qs * qs) - (value(p) + ps * ps)) / (2.0 * (qs - ps));
            if s <= *bounds.last().expect("one bound per root") {
                roots.pop();
                bounds.pop();
                if roots.is_empty() {
                    continue;
                }
            } else {
                roots.push(q);
                bounds.push(s);
                break;
            }
        }
    }
    out.clear();
    if roots.is_empty() {
        out.resize(n, f64::INFINITY);
        return;
    }
    bounds.push(f64::INFINITY);
    let mut k = 0;
    for q in offset..offset + n {
        let qs = q as f64 * step;
        while bounds[k + 1] < qs {
            k += 1;
        }
        let p = roots[k];
        let d = (q as f64 - p as f64) * step;
        out.push(d * d + value(p));
    }
    debug_assert_eq!(out.len(), n, "one output per sample");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute force over all features, on the torus when wrapping.
    fn brute(input: &Raster, threshold: f32) -> Vec<f32> {
        let (w, h) = (i64::from(input.width()), i64::from(input.height()));
        let t = input.texel();
        let mut out = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let mut best = f64::INFINITY;
                for fy in 0..h {
                    for fx in 0..w {
                        if input.at(fx, fy) < threshold {
                            continue;
                        }
                        let (mut dx, mut dy) = ((fx - x).abs(), (fy - y).abs());
                        if input.edge() == Edge::Wrap {
                            dx = dx.min(w - dx);
                            dy = dy.min(h - dy);
                        }
                        let (dx, dy) = (dx as f64 * f64::from(t.x), dy as f64 * f64::from(t.y));
                        best = best.min(dx * dx + dy * dy);
                    }
                }
                #[expect(clippy::cast_possible_truncation, reason = "test comparison")]
                out.push(libm::sqrt(best) as f32);
            }
        }
        out
    }

    #[test]
    fn matches_brute_force() {
        for edge in [Edge::Wrap, Edge::Clamp] {
            for seed in 0..6_u32 {
                let (w, h) = (13 + seed, 9 + seed % 3);
                let values: Vec<f32> = (0..w * h)
                    .map(|i| {
                        let r = dapple_field::hash::hash(u64::from(seed), &[u64::from(i)]);
                        if r.is_multiple_of(11) { 1.0 } else { 0.0 }
                    })
                    .collect();
                let texel = Vec2::new(0.1, 0.07 + seed as f32 * 0.01);
                let raster = Raster::from_values(w, h, Vec2::ZERO, texel, edge, values).unwrap();
                let exact = DistanceTransform { threshold: 0.5 }.apply(&raster).unwrap();
                let expected = brute(&raster, 0.5);
                for (i, (a, b)) in exact.values().iter().zip(&expected).enumerate() {
                    assert!(
                        a == b || (a - b).abs() <= 1e-6 * b.max(1.0),
                        "{edge:?} seed {seed} texel {i}: {a} vs {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn empty_rasters_are_infinitely_far() {
        let r = Raster::from_values(3, 2, Vec2::ZERO, Vec2::ONE, Edge::Wrap, alloc::vec![0.0; 6])
            .unwrap();
        let d = DistanceTransform { threshold: 0.5 }.apply(&r).unwrap();
        assert!(d.values().iter().all(|v| v.is_infinite()));
    }
}
