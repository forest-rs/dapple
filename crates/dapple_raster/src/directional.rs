// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Directional processing: sampling along slopes, displacement by vector
//! fields, and advection along flows.
//!
//! Each takes a second raster that says where to look (a height whose
//! slope leads, a vector field, a flow) on the input's grid, and states its
//! reach in domain units, so its footprint is bounded and it stays a local
//! stencil at any resolution. Samples between texels are bilinear with the
//! input's edge policy, so wrapping inputs give tiling results.

use alloc::vec::Vec;

use dapple_field::image::bilinear;
use glam::Vec2;

use crate::{OpCategory, Raster, RasterError};

/// Bilinear sample of `r` at a domain position offset `d` from texel
/// `(x, y)`'s center.
fn sample_at(r: &Raster, x: u32, y: u32, d: Vec2) -> f32 {
    #[expect(clippy::cast_precision_loss, reason = "texel indices are small")]
    let t = Vec2::new(x as f32, y as f32) + d / r.texel();
    bilinear(r.values(), [r.width(), r.height()], r.edge(), t)
}

fn sample_vec(r: &Raster<[f32; 2]>, t: Vec2) -> Vec2 {
    let (fx, fy) = (libm::floorf(t.x), libm::floorf(t.y));
    #[expect(
        clippy::cast_possible_truncation,
        reason = "positions are within a few grids of the origin"
    )]
    let (x0, y0) = (fx as i64, fy as i64);
    let (tx, ty) = (t.x - fx, t.y - fy);
    let at = |x: i64, y: i64| Vec2::from_array(r.at(x, y));
    let top = at(x0, y0).lerp(at(x0 + 1, y0), tx);
    let bottom = at(x0, y0 + 1).lerp(at(x0 + 1, y0 + 1), tx);
    top.lerp(bottom, ty)
}

fn same_grid<T: Copy, U: Copy>(a: &Raster<T>, b: &Raster<U>) -> Result<(), RasterError> {
    if a.same_grid(b) {
        Ok(())
    } else {
        Err(RasterError::LengthMismatch {
            expected: a.values().len(),
            found: b.values().len(),
        })
    }
}

fn like(grid: &Raster, values: Vec<f32>) -> Result<Raster, RasterError> {
    Raster::from_values(
        grid.width(),
        grid.height(),
        grid.origin(),
        grid.texel(),
        grid.edge(),
        values,
    )
}

/// How [`SlopeSample`] combines the samples along a path.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum SlopeMode {
    /// The mean: a blur along the slope.
    Average,
    /// The largest: features dragged downhill.
    Max,
    /// The smallest.
    Min,
}

/// Slope-driven sampling: each texel reads the input along the path that
/// runs downhill on a `slope` height from it, `reach` domain units long in
/// `steps` steps, and combines what it reads by `mode`.
///
/// Where the height is flat (a gradient under `1e-6` per domain unit) the
/// path does not move. Paths turn with the slope at every step, so the
/// input is smeared along the height's fall lines, as eroded stone and
/// weathered paint are.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SlopeSample {
    /// Path length in domain units; finite and non-negative.
    pub reach: f32,
    /// Steps along the path, 1 to 256.
    pub steps: u32,
    /// How the samples combine.
    pub mode: SlopeMode,
}

impl SlopeSample {
    /// The op's category: bounded by its reach, a local stencil.
    #[must_use]
    pub const fn category(&self) -> OpCategory {
        OpCategory::LocalStencil
    }

    /// Texels read on each side of an output texel.
    #[must_use]
    pub fn footprint(&self, texel: Vec2) -> [u32; 2] {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "reach over texel is small and positive"
        )]
        let r = |t: f32| libm::ceilf(self.reach / t) as u32 + 1;
        [r(texel.x), r(texel.y)]
    }

    /// Samples `input` along the fall lines of `slope`, both on one grid.
    ///
    /// # Errors
    ///
    /// [`RasterError::InvalidParameter`] for a bad reach or step count,
    /// [`RasterError::LengthMismatch`] for rasters on different grids.
    pub fn apply(&self, input: &Raster, slope: &Raster) -> Result<Raster, RasterError> {
        if !(self.reach.is_finite() && self.reach >= 0.0) {
            return Err(RasterError::InvalidParameter { name: "reach" });
        }
        if !(1..=256).contains(&self.steps) {
            return Err(RasterError::InvalidParameter { name: "steps" });
        }
        same_grid(input, slope)?;
        #[expect(clippy::cast_precision_loss, reason = "step counts are small")]
        let step = self.reach / self.steps as f32;
        let texel = input.texel();
        let mut out = Vec::with_capacity(input.values().len());
        for y in 0..input.height() {
            for x in 0..input.width() {
                let mut d = Vec2::ZERO;
                let mut acc = sample_at(input, x, y, d);
                for _ in 0..self.steps {
                    // The slope's gradient by central differences at the
                    // path's current point, in value per domain unit.
                    let gx = (sample_at(slope, x, y, d + Vec2::new(texel.x, 0.0))
                        - sample_at(slope, x, y, d - Vec2::new(texel.x, 0.0)))
                        / (2.0 * texel.x);
                    let gy = (sample_at(slope, x, y, d + Vec2::new(0.0, texel.y))
                        - sample_at(slope, x, y, d - Vec2::new(0.0, texel.y)))
                        / (2.0 * texel.y);
                    let g = Vec2::new(gx, gy);
                    let len = g.length();
                    if len > 1e-6 {
                        d -= g / len * step;
                    }
                    let v = sample_at(input, x, y, d);
                    acc = match self.mode {
                        SlopeMode::Average => acc + v,
                        SlopeMode::Max => acc.max(v),
                        SlopeMode::Min => acc.min(v),
                    };
                }
                if self.mode == SlopeMode::Average {
                    #[expect(clippy::cast_precision_loss, reason = "step counts are small")]
                    let n = (self.steps + 1) as f32;
                    acc /= n;
                }
                out.push(acc);
            }
        }
        like(input, out)
    }
}

/// Vector displacement: each texel reads the input `amount ·
/// vectors(p)` domain units away, `out(p) = input(p − amount · v(p))`, so
/// the content moves along the vectors.
///
/// `max_length` bounds the vectors' length (longer ones are clamped), which
/// bounds the footprint.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Displace {
    /// Domain units of displacement per unit of vector; finite.
    pub amount: f32,
    /// The longest vector honored, in vector units; finite and positive.
    pub max_length: f32,
}

impl Displace {
    /// The op's category: bounded by its reach, a local stencil.
    #[must_use]
    pub const fn category(&self) -> OpCategory {
        OpCategory::LocalStencil
    }

    /// Displaces `input` by `vectors`, both on one grid.
    ///
    /// # Errors
    ///
    /// [`RasterError::InvalidParameter`] for a bad amount or length,
    /// [`RasterError::LengthMismatch`] for rasters on different grids.
    pub fn apply(&self, input: &Raster, vectors: &Raster<[f32; 2]>) -> Result<Raster, RasterError> {
        if !(self.amount.is_finite() && self.max_length.is_finite() && self.max_length > 0.0) {
            return Err(RasterError::InvalidParameter { name: "amount" });
        }
        same_grid(input, vectors)?;
        let mut out = Vec::with_capacity(input.values().len());
        for y in 0..input.height() {
            for x in 0..input.width() {
                let v = Vec2::from_array(vectors.at(i64::from(x), i64::from(y)))
                    .clamp_length_max(self.max_length);
                out.push(sample_at(input, x, y, -v * self.amount));
            }
        }
        like(input, out)
    }
}

/// Advection along a flow: each texel traces the flow's streamline
/// upstream, `steps` steps of `step` domain units, and averages the input
/// along it, so what the input holds is carried downstream and smeared
/// along the flow's own curves (rain washing grime down a wall that
/// wanders around its units).
///
/// The flow is read as a direction; its length scales the step, clamped
/// at one. Paths are semi-Lagrangian: each step reads the flow where the
/// path has reached. The reach is `steps · step`, so the footprint is bounded.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Advect {
    /// Step length in domain units; finite and positive.
    pub step: f32,
    /// Steps upstream, 1 to 512.
    pub steps: u32,
}

impl Advect {
    /// The op's category: an iterated tracer of bounded reach.
    #[must_use]
    pub const fn category(&self) -> OpCategory {
        OpCategory::IterativeSolve
    }

    /// Advects `input` along `flow`, both on one grid.
    ///
    /// # Errors
    ///
    /// [`RasterError::InvalidParameter`] for a bad step or count,
    /// [`RasterError::LengthMismatch`] for rasters on different grids.
    pub fn apply(&self, input: &Raster, flow: &Raster<[f32; 2]>) -> Result<Raster, RasterError> {
        if !(self.step.is_finite() && self.step > 0.0 && (1..=512).contains(&self.steps)) {
            return Err(RasterError::InvalidParameter { name: "step" });
        }
        same_grid(input, flow)?;
        let texel = input.texel();
        let mut out = Vec::with_capacity(input.values().len());
        for y in 0..input.height() {
            for x in 0..input.width() {
                #[expect(clippy::cast_precision_loss, reason = "texel indices are small")]
                let origin = Vec2::new(x as f32, y as f32);
                let mut d = Vec2::ZERO;
                let mut acc = sample_at(input, x, y, d);
                for _ in 0..self.steps {
                    let v = sample_vec(flow, origin + d / texel).clamp_length_max(1.0);
                    d -= v * self.step;
                    acc += sample_at(input, x, y, d);
                }
                #[expect(clippy::cast_precision_loss, reason = "step counts are small")]
                let n = (self.steps + 1) as f32;
                out.push(acc / n);
            }
        }
        like(input, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Edge;

    fn grid(n: u32, f: impl Fn(u32, u32) -> f32) -> Raster {
        let mut v = Vec::new();
        for y in 0..n {
            for x in 0..n {
                v.push(f(x, y));
            }
        }
        Raster::from_values(n, n, Vec2::ZERO, Vec2::splat(0.01), Edge::Wrap, v).unwrap()
    }

    fn field(n: u32, v: Vec2) -> Raster<[f32; 2]> {
        Raster::from_values(
            n,
            n,
            Vec2::ZERO,
            Vec2::splat(0.01),
            Edge::Wrap,
            alloc::vec![v.to_array(); (n * n) as usize],
        )
        .unwrap()
    }

    #[test]
    fn slope_sampling_reads_downhill_only() {
        // A height falling toward +x, and a spike at x = 10: texels
        // uphill of the spike (smaller x) pick it up, those past it do not.
        let slope = grid(32, |x, _| -(x as f32));
        let input = grid(32, |x, _| if x == 10 { 1.0 } else { 0.0 });
        let out = SlopeSample {
            reach: 0.05,
            steps: 5,
            mode: SlopeMode::Max,
        }
        .apply(&input, &slope)
        .unwrap();
        assert_eq!(out.at(7, 3), 1.0, "uphill reads the spike");
        assert_eq!(out.at(12, 3), 0.0, "downhill does not");
        let flat = grid(32, |_, _| 0.0);
        let same = SlopeSample {
            reach: 0.05,
            steps: 5,
            mode: SlopeMode::Average,
        }
        .apply(&input, &flat)
        .unwrap();
        assert_eq!(same, input, "flat heights do not move paths");
    }

    #[test]
    fn displacement_moves_content_and_advection_carries_it() {
        let input = grid(32, |x, y| if (x, y) == (10, 10) { 1.0 } else { 0.0 });
        let moved = Displace {
            amount: 0.03,
            max_length: 2.0,
        }
        .apply(&input, &field(32, Vec2::X))
        .unwrap();
        assert!((moved.at(13, 10) - 1.0).abs() < 1e-6, "moved 3 texels");
        let carried = Advect {
            step: 0.01,
            steps: 4,
        }
        .apply(&input, &field(32, Vec2::NEG_Y))
        .unwrap();
        assert!(carried.at(10, 8) > 0.0, "carried downstream");
        assert_eq!(carried.at(10, 12), 0.0, "never upstream");
        assert_eq!(
            Advect {
                step: 0.01,
                steps: 4
            }
            .category(),
            OpCategory::IterativeSolve
        );
    }
}
