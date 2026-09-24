// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Separable Gaussian blur in domain units.

use alloc::vec::Vec;

use glam::Vec2;

use crate::{Raster, RasterError, RasterOp, TexelRect, check_into};

/// Gaussian blur with standard deviation `sigma` in domain units.
///
/// The kernel is sampled at texel offsets out to `ceil(3σ)` texels per axis
/// and normalized to sum to one, so constant rasters stay constant. The two
/// passes run horizontally, then vertically, each summing from the most
/// negative offset to the most positive. A `sigma` below 1/1000 of a texel on
/// an axis leaves that axis unchanged.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussianBlur {
    /// Standard deviation in domain units; finite and non-negative.
    pub sigma: f32,
}

impl GaussianBlur {
    fn kernel(self, sigma_texels: f32) -> Vec<f32> {
        if sigma_texels < 1e-3 {
            return alloc::vec![1.0];
        }
        let radius = radius(sigma_texels);
        let denom = 2.0 * sigma_texels * sigma_texels;
        let mut weights: Vec<f32> = (-radius..=radius)
            .map(|i| {
                #[expect(clippy::cast_precision_loss, reason = "kernel offsets are small")]
                let d = i as f32;
                libm::expf(-(d * d) / denom)
            })
            .collect();
        let sum: f32 = weights.iter().sum();
        for w in &mut weights {
            *w /= sum;
        }
        weights
    }
}

fn radius(sigma_texels: f32) -> i64 {
    if sigma_texels < 1e-3 {
        return 0;
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "radius is checked against the texel limit by the caller"
    )]
    let r = libm::ceilf(3.0 * sigma_texels) as i64;
    r
}

impl RasterOp for GaussianBlur {
    type Output = f32;

    fn category(&self) -> crate::OpCategory {
        crate::OpCategory::SeparablePass
    }

    fn footprint(&self, texel: Vec2) -> Option<[u32; 2]> {
        let r = |t: f32| u32::try_from(radius(self.sigma / t)).ok();
        Some([r(texel.x)?, r(texel.y)?])
    }

    fn apply(&self, input: &Raster) -> Result<Raster, RasterError> {
        let (kx, ky) = self.prepare(input)?;
        let rx = half(&kx);
        let ry = half(&ky);
        let horizontal = input.map_texels(|x, y| {
            kx.iter()
                .zip(-rx..=rx)
                .fold(0.0, |sum, (w, i)| sum + w * input.at(x + i, y))
        });
        Ok(horizontal.map_texels(|x, y| {
            ky.iter()
                .zip(-ry..=ry)
                .fold(0.0, |sum, (w, i)| sum + w * horizontal.at(x, y + i))
        }))
    }

    fn apply_into(
        &self,
        input: &Raster,
        rect: TexelRect,
        output: &mut Raster,
    ) -> Result<(), RasterError> {
        let (kx, ky) = self.prepare(input)?;
        check_into(input, rect, output)?;
        if rect.is_empty() {
            return Ok(());
        }
        let rx = half(&kx);
        let ry = half(&ky);
        // The horizontal pass over just the rows the vertical taps reach.
        // Reading `input` at an unresolved row resolves it exactly as the full
        // pass resolves the row it reads back, so the sums are identical.
        let (x0, y0) = (i64::from(rect.x0), i64::from(rect.y0));
        let columns = usize::try_from(rect.x1 - rect.x0).expect("columns fit usize");
        let first = y0 - ry;
        let last = i64::from(rect.y1) + ry;
        let mut rows = Vec::with_capacity(columns * usize::try_from(last - first).unwrap_or(0));
        for y in first..last {
            for x in x0..i64::from(rect.x1) {
                rows.push(
                    kx.iter()
                        .zip(-rx..=rx)
                        .fold(0.0, |sum, (w, i)| sum + w * input.at(x + i, y)),
                );
            }
        }
        let at = |x: i64, y: i64| {
            let row = usize::try_from(y - first).expect("row within the pass");
            let column = usize::try_from(x - x0).expect("column within the pass");
            rows[row * columns + column]
        };
        output.map_rect(rect, |x, y| {
            ky.iter()
                .zip(-ry..=ry)
                .fold(0.0, |sum, (w, i)| sum + w * at(x, y + i))
        });
        Ok(())
    }
}

fn half(kernel: &[f32]) -> i64 {
    i64::try_from(kernel.len() / 2).expect("kernel length fits i64")
}

impl GaussianBlur {
    /// Validates `sigma` against `input` and builds both kernels.
    fn prepare(&self, input: &Raster) -> Result<(Vec<f32>, Vec<f32>), RasterError> {
        if !(self.sigma.is_finite() && self.sigma >= 0.0) {
            return Err(RasterError::InvalidParameter { name: "sigma" });
        }
        let texel = input.texel();
        let limit = i64::from(input.width().max(input.height())) * 64;
        for t in [texel.x, texel.y] {
            if radius(self.sigma / t) > limit {
                return Err(RasterError::InvalidParameter { name: "sigma" });
            }
        }
        Ok((
            self.kernel(self.sigma / texel.x),
            self.kernel(self.sigma / texel.y),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Edge;

    fn raster(width: u32, height: u32, edge: Edge, values: Vec<f32>) -> Raster {
        Raster::from_values(width, height, Vec2::ZERO, Vec2::splat(0.1), edge, values).unwrap()
    }

    #[test]
    fn constants_stay_constant_and_mass_is_preserved() {
        let flat = raster(8, 8, Edge::Clamp, alloc::vec![0.25; 64]);
        let blurred = GaussianBlur { sigma: 0.15 }.apply(&flat).unwrap();
        for v in blurred.values() {
            assert!((v - 0.25).abs() < 1e-6, "{v}");
        }

        let mut impulse = alloc::vec![0.0; 64];
        impulse[27] = 1.0;
        let spread = GaussianBlur { sigma: 0.1 }
            .apply(&raster(8, 8, Edge::Wrap, impulse))
            .unwrap();
        let total: f32 = spread.values().iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "wrapped blur keeps mass: {total}"
        );
        assert!(spread.values()[27] < 1.0);
    }

    #[test]
    fn footprints_are_physical() {
        let blur = GaussianBlur { sigma: 0.004 };
        assert_eq!(blur.footprint(Vec2::splat(1.0 / 1024.0)), Some([13, 13]));
        assert_eq!(blur.footprint(Vec2::splat(1.0 / 4096.0)), Some([50, 50]));
        assert_eq!(
            GaussianBlur { sigma: 0.0 }.footprint(Vec2::ONE),
            Some([0, 0])
        );
    }

    #[test]
    fn rejects_bad_sigma() {
        let flat = raster(2, 2, Edge::Clamp, alloc::vec![0.0; 4]);
        assert!(GaussianBlur { sigma: -1.0 }.apply(&flat).is_err());
        assert!(GaussianBlur { sigma: f32::NAN }.apply(&flat).is_err());
    }
}
