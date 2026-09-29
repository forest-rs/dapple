// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Normals from height.

use glam::{Vec2, Vec3};

use crate::{Edge, Raster, RasterError, RasterOp, TexelRect, check_into};

/// Unit surface normals of the height field `scale * input`.
///
/// **Frame.** Normals are in the raster's domain frame: `+X` along the domain
/// x axis, `+Y` along the domain y axis (increasing row index), and `+Z` out of
/// the surface. A flat height gives `(0, 0, 1)`. Mapping to a texture's
/// tangent space, including any green-channel flip, belongs to the output
/// profile, not here.
///
/// Slopes are central differences over two texels in domain units:
/// `n = normalize(-∂h/∂x, -∂h/∂y, 1)`. `scale` converts raster values to
/// domain units of height, so the result is resolution-independent.
///
/// At the border of a clamped raster the texel outside is the edge texel
/// itself, so the difference is one-sided over the one texel it spans (the
/// slope stays a slope in domain units, not half of one). Along an axis only
/// one texel long there is nothing to differ, and the slope is 0.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HeightToNormal {
    /// Domain units of height per raster value unit; finite.
    pub scale: f32,
}

impl RasterOp for HeightToNormal {
    type Output = [f32; 3];

    fn footprint(&self, _texel: Vec2) -> Option<[u32; 2]> {
        Some([1, 1])
    }

    fn apply(&self, input: &Raster) -> Result<Raster<[f32; 3]>, RasterError> {
        self.check()?;
        Ok(input.map_texels(|x, y| self.texel(input, x, y)))
    }

    fn apply_into(
        &self,
        input: &Raster,
        rect: TexelRect,
        output: &mut Raster<[f32; 3]>,
    ) -> Result<(), RasterError> {
        self.check()?;
        check_into(input, rect, output)?;
        output.map_rect(rect, |x, y| self.texel(input, x, y));
        Ok(())
    }
}

impl HeightToNormal {
    fn check(&self) -> Result<(), RasterError> {
        if self.scale.is_finite() {
            Ok(())
        } else {
            Err(RasterError::InvalidParameter { name: "scale" })
        }
    }

    fn texel(&self, input: &Raster, x: i64, y: i64) -> [f32; 3] {
        let texel = input.texel();
        let dx = self.slope(input, [x - 1, y], [x + 1, y], texel.x);
        let dy = self.slope(input, [x, y - 1], [x, y + 1], texel.y);
        Vec3::new(-dx, -dy, 1.0).normalize().to_array()
    }

    /// The slope between the `lo` and `hi` neighbors of a texel, per domain
    /// unit.
    ///
    /// A clamped raster repeats its edge texel outside, so at the border the
    /// central difference spans one texel, not two; divide by the texels it
    /// really spans, or the border slope comes out at half strength.
    fn slope(&self, input: &Raster, lo: [i64; 2], hi: [i64; 2], texel: f32) -> f32 {
        let span = match input.edge() {
            Edge::Wrap => 2,
            Edge::Clamp => {
                let (width, height) = (i64::from(input.width()), i64::from(input.height()));
                let clamp = |[x, y]: [i64; 2]| [x.clamp(0, width - 1), y.clamp(0, height - 1)];
                let (lo, hi) = (clamp(lo), clamp(hi));
                // The neighbors differ along one axis only.
                (hi[0] - lo[0]) + (hi[1] - lo[1])
            }
        };
        if span == 0 {
            return 0.0;
        }
        #[expect(clippy::cast_precision_loss, reason = "the span is 0, 1, or 2")]
        let distance = span as f32 * texel;
        (input.at(hi[0], hi[1]) - input.at(lo[0], lo[1])) * self.scale / distance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn slopes_tilt_normals_against_the_gradient() {
        // Height rises 0.1 domain units per texel of 0.1 along x: slope 1.
        let values: Vec<f32> = (0..16).map(|i| (i % 4) as f32 * 0.1).collect();
        let ramp =
            Raster::from_values(4, 4, Vec2::ZERO, Vec2::splat(0.1), Edge::Clamp, values).unwrap();
        let normals = HeightToNormal { scale: 1.0 }.apply(&ramp).unwrap();
        let inner = normals.at(1, 1);
        let s = core::f32::consts::FRAC_1_SQRT_2;
        assert!(
            (inner[0] + s).abs() < 1e-6 && inner[1].abs() < 1e-6,
            "{inner:?}"
        );
        assert!((inner[2] - s).abs() < 1e-6, "{inner:?}");

        // A clamped border reads one texel outside as itself, but the slope
        // there is still the ramp's slope, not half of it.
        for x in 0..4 {
            let n = normals.at(x, 2);
            assert!(
                (n[0] + s).abs() < 1e-6 && n[1].abs() < 1e-6,
                "x = {x}: {n:?}"
            );
        }

        let flat =
            Raster::from_values(2, 2, Vec2::ZERO, Vec2::ONE, Edge::Wrap, alloc::vec![3.0; 4])
                .unwrap();
        let up = HeightToNormal { scale: 5.0 }.apply(&flat).unwrap();
        assert!(up.values().iter().all(|n| *n == [0.0, 0.0, 1.0]));
    }

    #[test]
    fn each_axis_divides_by_its_own_texel_at_the_border() {
        // Height rises 0.05 per row of 0.05 along y (slope 1) on texels
        // twice as wide as tall, clamped: the top and bottom rows are borders.
        let values: Vec<f32> = (0..12).map(|i| (i / 3) as f32 * 0.05).collect();
        let ramp = Raster::from_values(3, 4, Vec2::ZERO, Vec2::new(0.1, 0.05), Edge::Clamp, values)
            .unwrap();
        let normals = HeightToNormal { scale: 1.0 }.apply(&ramp).unwrap();
        let s = core::f32::consts::FRAC_1_SQRT_2;
        for y in 0..4 {
            let n = normals.at(1, y);
            assert!(
                n[0].abs() < 1e-6 && (n[1] + s).abs() < 1e-6,
                "y = {y}: {n:?}"
            );
        }
    }

    #[test]
    fn a_one_texel_axis_has_no_slope() {
        // Clamped, one column wide: every neighbor along x is the texel
        // itself. Rows still slope.
        let ramp = Raster::from_values(
            1,
            3,
            Vec2::ZERO,
            Vec2::splat(0.1),
            Edge::Clamp,
            alloc::vec![0.0, 0.1, 0.2],
        )
        .unwrap();
        let normals = HeightToNormal { scale: 1.0 }.apply(&ramp).unwrap();
        for y in 0..3 {
            let n = normals.at(0, y);
            assert_eq!(n[0], 0.0, "y = {y}");
            assert!(n[1] < 0.0);
        }
    }
}
