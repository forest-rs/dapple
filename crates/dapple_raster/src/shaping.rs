// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Measured value shaping: histograms, percentiles and percentile
//! remapping.
//!
//! Pointwise shaping (tone curves, ramps, smooth thresholds) needs no
//! neighbors and lives in `dapple_field::shaping` and scoped programs.
//! These operations measure the whole raster first, so they are
//! [`OpCategory::Reduction`]s or [`OpCategory::GlobalTransform`]s: any
//! texel can change every output texel. Measurements are exact order
//! statistics of the texel values (no binning), so they do not depend on
//! tile order or thread count.

use alloc::vec;
use alloc::vec::Vec;

use glam::Vec2;

use crate::{OpCategory, Raster, RasterError, RasterOp};

/// A histogram of a scalar raster over a fixed range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Histogram {
    /// Texels per bin, from the range's low end.
    pub counts: Vec<u64>,
    /// Texels below the range.
    pub below: u64,
    /// Texels at or above the range's high end, or not finite.
    pub above: u64,
}

impl Histogram {
    /// Counts `raster`'s values into `bins` equal bins over `range`.
    ///
    /// # Errors
    ///
    /// [`RasterError::InvalidParameter`] for zero bins or a range that is
    /// not finite and increasing.
    pub fn measure(raster: &Raster, bins: u32, range: [f32; 2]) -> Result<Self, RasterError> {
        if bins == 0 || !(range[0].is_finite() && range[1].is_finite() && range[0] < range[1]) {
            return Err(RasterError::InvalidParameter { name: "range" });
        }
        let mut h = Self {
            counts: vec![0; bins as usize],
            below: 0,
            above: 0,
        };
        #[expect(clippy::cast_precision_loss, reason = "bin counts are small")]
        let scale = bins as f32 / (range[1] - range[0]);
        for &v in raster.values() {
            if v < range[0] {
                h.below += 1;
            } else if v < range[1] {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "v is inside the range, so the bin is in 0..bins"
                )]
                let bin = (((v - range[0]) * scale) as usize).min(bins as usize - 1);
                h.counts[bin] += 1;
            } else {
                h.above += 1;
            }
        }
        Ok(h)
    }
}

/// The values at `fractions` (each in `[0, 1]`) of `raster`'s sorted
/// values, interpolating linearly between neighboring order statistics:
/// 0 is the minimum, 1 the maximum, 0.5 the median.
///
/// Non-finite texels are ignored; a raster without finite texels gives
/// zeros.
///
/// # Errors
///
/// [`RasterError::InvalidParameter`] for a fraction outside `[0, 1]`.
pub fn percentiles(raster: &Raster, fractions: &[f32]) -> Result<Vec<f32>, RasterError> {
    if fractions.iter().any(|f| !(0.0..=1.0).contains(f)) {
        return Err(RasterError::InvalidParameter { name: "fraction" });
    }
    let mut sorted: Vec<f32> = raster
        .values()
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if sorted.is_empty() {
        return Ok(vec![0.0; fractions.len()]);
    }
    sorted.sort_unstable_by(f32::total_cmp);
    let last = sorted.len() - 1;
    Ok(fractions
        .iter()
        .map(|&f| {
            #[expect(clippy::cast_precision_loss, reason = "texel counts fit in f32 range")]
            let at = f * last as f32;
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "at is in [0, last]"
            )]
            let i = (libm::floorf(at) as usize).min(last);
            #[expect(clippy::cast_precision_loss, reason = "as above")]
            let t = at - i as f32;
            let j = (i + 1).min(last);
            sorted[i] + (sorted[j] - sorted[i]) * t
        })
        .collect())
}

/// Maps the values at percentiles `from` of the input to `to`, linearly,
/// optionally clamped: a controlled normalization that ignores outliers
/// beyond the chosen percentiles.
///
/// `from: [0.0, 1.0]` is a min–max normalization; `from: [0.02, 0.98]`
/// ignores the extreme 2% at each end. A threshold at percentile `1 − c`
/// (`from: [1 − c − ε, 1 − c + ε]`, clamped to `[0, 1]`) covers a fraction
/// `c` of the texels whatever the input's range, which is how a module
/// states "about 8% covered" rather than a raw threshold.
///
/// When both percentiles measure the same value, texels below it map to
/// `to[0]` and the rest to `to[1]`.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PercentileRemap {
    /// The input percentiles, fractions in `[0, 1]`, increasing.
    pub from: [f32; 2],
    /// The output values they map to.
    pub to: [f32; 2],
    /// Whether outputs are clamped to `to`'s range.
    pub clamp: bool,
}

impl PercentileRemap {
    /// The input values at [`Self::from`]'s percentiles.
    ///
    /// # Errors
    ///
    /// [`RasterError::InvalidParameter`] for percentiles outside `[0, 1]`
    /// or decreasing.
    pub fn measure(&self, input: &Raster) -> Result<[f32; 2], RasterError> {
        if self.from[0] > self.from[1] || !self.to.iter().all(|v| v.is_finite()) {
            return Err(RasterError::InvalidParameter { name: "from" });
        }
        let p = percentiles(input, &self.from)?;
        Ok([p[0], p[1]])
    }
}

impl RasterOp for PercentileRemap {
    type Output = f32;

    fn footprint(&self, _texel: Vec2) -> Option<[u32; 2]> {
        None
    }

    fn category(&self) -> OpCategory {
        OpCategory::GlobalTransform
    }

    fn apply(&self, input: &Raster) -> Result<Raster, RasterError> {
        let [lo, hi] = self.measure(input)?;
        let (a, b) = (self.to[0], self.to[1]);
        let (min, max) = (a.min(b), a.max(b));
        let values = input
            .values()
            .iter()
            .map(|&v| {
                let out = if hi > lo {
                    a + (v - lo) / (hi - lo) * (b - a)
                } else if v < lo {
                    a
                } else {
                    b
                };
                if self.clamp { out.clamp(min, max) } else { out }
            })
            .collect();
        Raster::from_values(
            input.width(),
            input.height(),
            input.origin(),
            input.texel(),
            input.edge(),
            values,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Edge;

    fn ramp(n: u32) -> Raster {
        #[expect(clippy::cast_precision_loss, reason = "small test values")]
        let values = (0..n).map(|i| i as f32).collect();
        Raster::from_values(n, 1, Vec2::ZERO, Vec2::ONE, Edge::Clamp, values).unwrap()
    }

    #[test]
    fn percentiles_are_order_statistics() {
        let r = ramp(101);
        assert_eq!(
            percentiles(&r, &[0.0, 0.5, 1.0, 0.255]).unwrap(),
            vec![0.0, 50.0, 100.0, 25.5],
            "min, median, max, interpolated"
        );
        let h = Histogram::measure(&r, 4, [0.0, 100.0]).unwrap();
        assert_eq!(h.counts, vec![25, 25, 25, 25], "equal bins");
        assert_eq!(h.above, 1, "the maximum is at the range's end");
    }

    #[test]
    fn percentile_thresholds_cover_the_stated_fraction() {
        let r = ramp(1000);
        let covered = 0.08;
        let op = PercentileRemap {
            from: [1.0 - covered - 0.001, 1.0 - covered + 0.001],
            to: [0.0, 1.0],
            clamp: true,
        };
        let mask = op.apply(&r).unwrap();
        let sum: f32 = mask.values().iter().sum();
        assert!((sum / 1000.0 - covered).abs() < 0.002, "{sum}");
        assert_eq!(op.category(), OpCategory::GlobalTransform);
    }
}
