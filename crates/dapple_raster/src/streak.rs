// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! One-sided directional smearing: streaks that run from their sources.

use alloc::vec;

use glam::Vec2;

use crate::{Edge, OpCategory, Raster, RasterError, RasterOp};

/// The direction a [`Streak`] runs, along a domain axis.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Flow {
    /// Toward `+x`.
    PositiveX,
    /// Toward `−x`.
    NegativeX,
    /// Toward `+y`.
    PositiveY,
    /// Toward `−y`: down a wall whose domain `+y` is up.
    NegativeY,
}

/// A directional smear that runs one way from its sources and fades: each
/// texel keeps the larger of its own value and its upstream neighbor's
/// result attenuated by `exp(−texel / length)`.
///
/// A source of value `v` therefore leaves a trail `v · exp(−d / length)`
/// at distance `d` downstream, and never affects texels upstream of it:
/// rain carrying dirt down from a ledge, rust running from a nail.
/// `length` is in domain units, so trails keep their physical length at
/// every resolution. The result is at least the input everywhere.
///
/// It is a scan along one axis ([`OpCategory::SeparablePass`]): on a
/// wrapping raster each line is scanned twice around, so a trail crossing
/// the period's edge continues on the other side exactly as a longer
/// raster would give it up to the attenuation after one full period.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Streak {
    /// Which way trails run.
    pub flow: Flow,
    /// The distance, in domain units, over which a trail fades by `1/e`;
    /// finite and positive.
    pub length: f32,
}

impl RasterOp for Streak {
    type Output = f32;

    fn footprint(&self, _texel: Vec2) -> Option<[u32; 2]> {
        None
    }

    fn category(&self) -> OpCategory {
        OpCategory::SeparablePass
    }

    fn apply(&self, input: &Raster) -> Result<Raster, RasterError> {
        if !(self.length.is_finite() && self.length > 0.0) {
            return Err(RasterError::InvalidParameter { name: "length" });
        }
        let (w, h) = (input.width() as usize, input.height() as usize);
        let horizontal = matches!(self.flow, Flow::PositiveX | Flow::NegativeX);
        let forward = matches!(self.flow, Flow::PositiveX | Flow::PositiveY);
        let (lines, len) = if horizontal { (h, w) } else { (w, h) };
        let step = if horizontal {
            input.texel().x
        } else {
            input.texel().y
        };
        let decay = libm::expf(-step / self.length);
        let index = |line: usize, k: usize| {
            let k = if forward { k } else { len - 1 - k };
            if horizontal {
                line * w + k
            } else {
                k * w + line
            }
        };
        let values = input.values();
        let mut out = vec![0.0_f32; values.len()];
        let passes = if input.edge() == Edge::Wrap { 2 } else { 1 };
        for line in 0..lines {
            let mut c = f32::NEG_INFINITY;
            for pass in 0..passes {
                for k in 0..len {
                    let i = index(line, k);
                    c = values[i].max(c * decay);
                    if pass + 1 == passes {
                        out[i] = c;
                    }
                }
            }
        }
        Raster::from_values(
            input.width(),
            input.height(),
            input.origin(),
            input.texel(),
            input.edge(),
            out,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trails_run_downstream_and_fade_physically() {
        // One source at row 6 of a clamping 1 × 8 column, domain +y up.
        let mut values = vec![0.0; 8];
        values[6] = 1.0;
        let r =
            Raster::from_values(1, 8, Vec2::ZERO, Vec2::splat(0.01), Edge::Clamp, values).unwrap();
        let s = Streak {
            flow: Flow::NegativeY,
            length: 0.02,
        }
        .apply(&r)
        .unwrap();
        assert_eq!(s.at(0, 7), 0.0, "nothing upstream");
        assert_eq!(s.at(0, 6), 1.0, "the source");
        let expected = libm::expf(-0.5);
        assert!((s.at(0, 5) - expected).abs() < 1e-6, "one texel down");
        assert!(
            (s.at(0, 4) - expected * expected).abs() < 1e-6,
            "two texels down"
        );

        // Twice the resolution, the same physical trail.
        let mut fine = vec![0.0; 16];
        fine[13] = 1.0;
        let r2 =
            Raster::from_values(1, 16, Vec2::ZERO, Vec2::splat(0.005), Edge::Clamp, fine).unwrap();
        let s2 = Streak {
            flow: Flow::NegativeY,
            length: 0.02,
        }
        .apply(&r2)
        .unwrap();
        assert!((s2.at(0, 11) - expected).abs() < 1e-6, "same length");
    }

    #[test]
    fn wrapping_trails_cross_the_edge() {
        let mut values = vec![0.0; 8];
        values[0] = 1.0;
        let r =
            Raster::from_values(8, 1, Vec2::ZERO, Vec2::splat(0.1), Edge::Wrap, values).unwrap();
        let s = Streak {
            flow: Flow::NegativeX,
            length: 0.1,
        }
        .apply(&r)
        .unwrap();
        assert!(
            (s.at(7, 0) - libm::expf(-1.0)).abs() < 1e-6,
            "continues past the edge"
        );
        assert_eq!(
            s.clone(),
            Streak {
                flow: Flow::NegativeX,
                length: 0.1,
            }
            .apply(&r.rolled(3, 0))
            .unwrap()
            .rolled(-3, 0),
            "commutes with rolling"
        );
    }
}
