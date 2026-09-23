// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Horizon-based ambient occlusion from height.

use alloc::vec::Vec;

use glam::Vec2;

use crate::{Raster, RasterError, RasterOp};

/// Largest supported number of directions.
const MAX_DIRECTIONS: u32 = 64;

/// Ambient occlusion of the height field `scale * input`, in `[0, 1]` where 1
/// is fully open.
///
/// For each of `directions` evenly spaced azimuths, the op walks one-texel
/// steps out to `radius` domain units, samples the height bilinearly, and
/// keeps the steepest elevation above the texel: the horizon. Occlusion is the
/// mean of `sin(horizon)` over directions, so a flat surface is 1 and a texel
/// at the bottom of a narrow slot approaches 0. Heights below the texel do not
/// open it further.
///
/// Sample offsets are computed once, relative to the texel, so the result
/// commutes exactly with rolling a wrapping raster.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AmbientOcclusion {
    /// Search radius in domain units; finite and positive.
    pub radius: f32,
    /// Number of azimuths, 1 to 64.
    pub directions: u32,
    /// Domain units of height per raster value unit; finite.
    pub scale: f32,
}

/// One precomputed sample: integer texel offset, bilinear weights, distance.
#[derive(Copy, Clone, Debug)]
struct Tap {
    x: i64,
    y: i64,
    u: f32,
    v: f32,
    distance: f32,
}

impl AmbientOcclusion {
    fn reach(self, texel: Vec2) -> [f32; 2] {
        [self.radius / texel.x, self.radius / texel.y]
    }

    fn taps(self, texel: Vec2) -> Vec<Vec<Tap>> {
        let steps_len = texel.min_element();
        #[expect(
            clippy::cast_possible_truncation,
            reason = "radius is checked against the raster size before this"
        )]
        let steps = libm::floorf(self.radius / steps_len) as u32;
        (0..self.directions)
            .map(|k| {
                let angle = core::f32::consts::TAU * k as f32 / self.directions as f32;
                let dir = Vec2::new(libm::cosf(angle), libm::sinf(angle));
                (1..=steps)
                    .map(|s| {
                        let offset = dir * (s as f32 * steps_len);
                        let t = offset / texel;
                        let (fx, fy) = (libm::floorf(t.x), libm::floorf(t.y));
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "offsets are bounded by the radius"
                        )]
                        Tap {
                            x: fx as i64,
                            y: fy as i64,
                            u: t.x - fx,
                            v: t.y - fy,
                            distance: offset.length(),
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

impl RasterOp for AmbientOcclusion {
    type Output = f32;

    fn footprint(&self, texel: Vec2) -> Option<[u32; 2]> {
        let [rx, ry] = self.reach(texel);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "saturating conversion of a finite reach"
        )]
        let r = |t: f32| libm::ceilf(t) as u32 + 1;
        (rx.is_finite() && ry.is_finite()).then(|| [r(rx), r(ry)])
    }

    fn apply(&self, input: &Raster) -> Result<Raster, RasterError> {
        if !(self.radius.is_finite() && self.radius > 0.0) {
            return Err(RasterError::InvalidParameter { name: "radius" });
        }
        if self.directions == 0 || self.directions > MAX_DIRECTIONS {
            return Err(RasterError::InvalidParameter { name: "directions" });
        }
        if !self.scale.is_finite() {
            return Err(RasterError::InvalidParameter { name: "scale" });
        }
        let [rx, ry] = self.reach(input.texel());
        let limit = (input.width().max(input.height()) * 4) as f32;
        if rx > limit || ry > limit {
            return Err(RasterError::InvalidParameter { name: "radius" });
        }
        let taps = self.taps(input.texel());
        Ok(input.map_texels(|x, y| {
            let here = input.at(x, y) * self.scale;
            let mut occlusion = 0.0;
            for direction in &taps {
                let mut slope = 0.0_f32;
                for tap in direction {
                    let (px, py) = (x + tap.x, y + tap.y);
                    let a = input.at(px, py);
                    let b = input.at(px + 1, py);
                    let c = input.at(px, py + 1);
                    let d = input.at(px + 1, py + 1);
                    let top = a + (b - a) * tap.u;
                    let bottom = c + (d - c) * tap.u;
                    let h = (top + (bottom - top) * tap.v) * self.scale;
                    slope = slope.max((h - here) / tap.distance);
                }
                // sin(atan(slope)) without trigonometry.
                occlusion += slope / libm::sqrtf(1.0 + slope * slope);
            }
            1.0 - occlusion / self.directions as f32
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Edge;

    #[test]
    fn flat_is_open_and_pits_are_occluded() {
        let flat = Raster::from_values(
            8,
            8,
            Vec2::ZERO,
            Vec2::splat(0.1),
            Edge::Wrap,
            alloc::vec![0.5; 64],
        )
        .unwrap();
        let ao = AmbientOcclusion {
            radius: 0.3,
            directions: 8,
            scale: 1.0,
        };
        assert!(ao.apply(&flat).unwrap().values().iter().all(|v| *v == 1.0));

        let mut pit = alloc::vec![1.0; 64];
        pit[4 * 8 + 4] = 0.0;
        let pit = Raster::from_values(8, 8, Vec2::ZERO, Vec2::splat(0.1), Edge::Wrap, pit).unwrap();
        let out = ao.apply(&pit).unwrap();
        let bottom = out.at(4, 4);
        assert!(bottom < 0.2, "pit bottom is occluded: {bottom}");
        assert_eq!(out.at(0, 0), 1.0, "far texels stay open");
    }

    #[test]
    fn rejects_bad_parameters() {
        let r = Raster::from_values(
            2,
            2,
            Vec2::ZERO,
            Vec2::ONE,
            Edge::Clamp,
            alloc::vec![0.0; 4],
        )
        .unwrap();
        let ok = AmbientOcclusion {
            radius: 1.0,
            directions: 4,
            scale: 1.0,
        };
        assert!(ok.apply(&r).is_ok());
        assert!(
            AmbientOcclusion {
                directions: 0,
                ..ok
            }
            .apply(&r)
            .is_err()
        );
        assert!(AmbientOcclusion { radius: 0.0, ..ok }.apply(&r).is_err());
        assert!(AmbientOcclusion { radius: 1e6, ..ok }.apply(&r).is_err());
    }
}
