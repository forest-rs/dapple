// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Normals from height.

use glam::{Vec2, Vec3};

use crate::{Raster, RasterError, RasterOp};

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
#[derive(Copy, Clone, Debug, PartialEq)]
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
        if !self.scale.is_finite() {
            return Err(RasterError::InvalidParameter { name: "scale" });
        }
        let step = input.texel() * 2.0;
        Ok(input.map_texels(|x, y| {
            let dx = (input.at(x + 1, y) - input.at(x - 1, y)) * self.scale / step.x;
            let dy = (input.at(x, y + 1) - input.at(x, y - 1)) * self.scale / step.y;
            Vec3::new(-dx, -dy, 1.0).normalize().to_array()
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Edge;
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

        let flat =
            Raster::from_values(2, 2, Vec2::ZERO, Vec2::ONE, Edge::Wrap, alloc::vec![3.0; 4])
                .unwrap();
        let up = HeightToNormal { scale: 5.0 }.apply(&flat).unwrap();
        assert!(up.values().iter().all(|n| *n == [0.0, 0.0, 1.0]));
    }
}
