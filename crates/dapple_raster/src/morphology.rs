// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Grayscale morphology with a disk in domain units.

use alloc::vec::Vec;

use glam::Vec2;

use crate::{Edge, Raster, RasterError, RasterOp, TexelRect, check_into};

/// A morphological operation.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum MorphologyOp {
    /// The largest value within the disk: masks grow.
    Dilate,
    /// The smallest value within the disk: masks shrink.
    Erode,
    /// Erode, then dilate: removes features narrower than the disk and
    /// keeps the rest.
    Open,
    /// Dilate, then erode: fills gaps narrower than the disk and keeps the
    /// rest.
    Close,
}

/// Grayscale morphology with a flat disk of `radius` domain units: every
/// texel takes the largest ([`Dilate`](MorphologyOp::Dilate)) or smallest
/// ([`Erode`](MorphologyOp::Erode)) value among the texels whose centers lie
/// within `radius` of its own, or composes the two
/// ([`Open`](MorphologyOp::Open), [`Close`](MorphologyOp::Close)).
///
/// On a binary mask this is the familiar binary morphology; on a coverage
/// mask it keeps coverage in `[0, 1]`, and on a height it rounds peaks or
/// fills pits. The disk is exact in domain units, including for
/// non-square texels, so a 2 mm opening removes the same features at every
/// resolution. Texels past the border follow the raster's edge policy.
///
/// The footprint is the disk's radius in texels per axis, doubled for
/// openings and closings, so tiles recompute locally.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Morphology {
    /// The operation.
    pub op: MorphologyOp,
    /// The disk's radius in domain units; finite and non-negative.
    pub radius: f32,
}

/// The largest disk radius, in texels, a morphology accepts.
const MAX_RADIUS_TEXELS: f32 = 256.0;

impl Morphology {
    /// The disk's offsets in texels, row by row.
    fn disk(&self, texel: Vec2) -> Result<Vec<(i64, i64)>, RasterError> {
        let invalid = RasterError::InvalidParameter { name: "radius" };
        if !(self.radius.is_finite() && self.radius >= 0.0) {
            return Err(invalid);
        }
        let reach = self.reach(texel).ok_or(invalid)?;
        let (rx, ry) = (i64::from(reach[0]), i64::from(reach[1]));
        let r2 = f64::from(self.radius) * f64::from(self.radius);
        let mut offsets = Vec::new();
        for dy in -ry..=ry {
            for dx in -rx..=rx {
                #[expect(clippy::cast_precision_loss, reason = "offsets are small")]
                let (x, y) = (
                    dx as f64 * f64::from(texel.x),
                    dy as f64 * f64::from(texel.y),
                );
                if x * x + y * y <= r2 {
                    offsets.push((dx, dy));
                }
            }
        }
        Ok(offsets)
    }

    /// The disk's radius in whole texels per axis.
    fn reach(&self, texel: Vec2) -> Option<[u32; 2]> {
        let texels = |t: f32| {
            let n = libm::floorf(self.radius / t);
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "checked to be a small non-negative whole number"
            )]
            let n = (n.is_finite() && (0.0..=MAX_RADIUS_TEXELS).contains(&n)).then_some(n as u32);
            n
        };
        Some([texels(texel.x)?, texels(texel.y)?])
    }

    /// The two stages, in order.
    const fn stages(&self) -> (bool, Option<bool>) {
        // `true` takes the largest value.
        match self.op {
            MorphologyOp::Dilate => (true, None),
            MorphologyOp::Erode => (false, None),
            MorphologyOp::Open => (false, Some(true)),
            MorphologyOp::Close => (true, Some(false)),
        }
    }
}

/// The largest or smallest of `at` over the disk around `(x, y)`.
fn extreme(
    disk: &[(i64, i64)],
    largest: bool,
    x: i64,
    y: i64,
    at: impl Fn(i64, i64) -> f32,
) -> f32 {
    disk.iter().fold(
        if largest {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        },
        |acc, &(dx, dy)| {
            let v = at(x + dx, y + dy);
            if largest { acc.max(v) } else { acc.min(v) }
        },
    )
}

/// Texel `(x, y)` resolved into `raster` by its edge policy.
fn resolved(raster: &Raster, x: i64, y: i64) -> (i64, i64) {
    let (w, h) = (i64::from(raster.width()), i64::from(raster.height()));
    match raster.edge() {
        Edge::Wrap => (x.rem_euclid(w), y.rem_euclid(h)),
        Edge::Clamp => (x.clamp(0, w - 1), y.clamp(0, h - 1)),
    }
}

impl RasterOp for Morphology {
    type Output = f32;

    fn footprint(&self, texel: Vec2) -> Option<[u32; 2]> {
        let [x, y] = self.reach(texel)?;
        let k = if self.stages().1.is_some() { 2 } else { 1 };
        Some([x * k, y * k])
    }

    fn apply(&self, input: &Raster) -> Result<Raster, RasterError> {
        let disk = self.disk(input.texel())?;
        let (first, second) = self.stages();
        let once = input.map_texels(|x, y| extreme(&disk, first, x, y, |x, y| input.at(x, y)));
        Ok(match second {
            None => once,
            Some(largest) => {
                once.map_texels(|x, y| extreme(&disk, largest, x, y, |x, y| once.at(x, y)))
            }
        })
    }

    fn apply_into(
        &self,
        input: &Raster,
        rect: TexelRect,
        output: &mut Raster,
    ) -> Result<(), RasterError> {
        let disk = self.disk(input.texel())?;
        check_into(input, rect, output)?;
        if rect.is_empty() {
            return Ok(());
        }
        let (first, second) = self.stages();
        let Some(largest) = second else {
            output.map_rect(rect, |x, y| {
                extreme(&disk, first, x, y, |x, y| input.at(x, y))
            });
            return Ok(());
        };
        // The first stage over `rect` grown by the disk, each texel computed
        // where the whole pass computes it (its resolved position), so the
        // second stage reads the same values it would read from the pass.
        let reach = self
            .reach(input.texel())
            .ok_or(RasterError::InvalidParameter { name: "radius" })?;
        let (rx, ry) = (i64::from(reach[0]), i64::from(reach[1]));
        let (x0, y0) = (i64::from(rect.x0) - rx, i64::from(rect.y0) - ry);
        let (x1, y1) = (i64::from(rect.x1) + rx, i64::from(rect.y1) + ry);
        let columns = usize::try_from(x1 - x0).expect("columns fit usize");
        let mut stage = Vec::with_capacity(columns * usize::try_from(y1 - y0).unwrap_or(0));
        for y in y0..y1 {
            for x in x0..x1 {
                let (x, y) = resolved(input, x, y);
                stage.push(extreme(&disk, first, x, y, |x, y| input.at(x, y)));
            }
        }
        let at = |x: i64, y: i64| {
            let row = usize::try_from(y - y0).expect("row within the stage");
            let column = usize::try_from(x - x0).expect("column within the stage");
            stage[row * columns + column]
        };
        output.map_rect(rect, |x, y| {
            extreme(&disk, largest, x, y, |xx, yy| {
                // Inside the grown rect, a texel's resolved twin was
                // computed at the unresolved position with the same value.
                at(xx, yy)
            })
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    fn raster(edge: Edge, values: Vec<f32>) -> Raster {
        Raster::from_values(8, 8, Vec2::ZERO, Vec2::splat(0.125), edge, values).unwrap()
    }

    /// A 2×2 square and a lone texel.
    fn shapes() -> Vec<f32> {
        let mut v = vec![0.0; 64];
        for (x, y) in [(2, 2), (3, 2), (2, 3), (3, 3), (6, 6)] {
            v[y * 8 + x] = 1.0;
        }
        v
    }

    #[test]
    fn dilation_and_erosion_use_an_exact_disk() {
        let mask = raster(Edge::Clamp, shapes());
        // One texel: the four neighbors, not the diagonals (√2 texels away).
        let grow = Morphology {
            op: MorphologyOp::Dilate,
            radius: 0.125,
        };
        let grown = grow.apply(&mask).unwrap();
        assert_eq!(grown.at(6, 5), 1.0);
        assert_eq!(grown.at(5, 5), 0.0);
        assert_eq!(grown.values().iter().filter(|&&v| v == 1.0).count(), 12 + 5);
        let shrink = Morphology {
            op: MorphologyOp::Erode,
            radius: 0.125,
        };
        assert!(
            shrink
                .apply(&mask)
                .unwrap()
                .values()
                .iter()
                .all(|&v| v == 0.0)
        );
    }

    #[test]
    fn opening_removes_small_features_and_closing_fills_gaps() {
        let mask = raster(Edge::Wrap, shapes());
        let open = Morphology {
            op: MorphologyOp::Open,
            radius: 0.07,
        };
        // Half a texel: only the texel itself, so nothing changes.
        assert_eq!(open.apply(&mask).unwrap(), mask);
        // A 2×2 square survives a 2×2 opening; with a disk of one texel it
        // does not (the disk is a plus, which fits nowhere inside).
        let wide = Morphology {
            op: MorphologyOp::Open,
            radius: 0.125,
        };
        assert!(
            wide.apply(&mask)
                .unwrap()
                .values()
                .iter()
                .all(|&v| v == 0.0)
        );
        // Closing fills a one-texel gap between two bars.
        let mut bars = vec![0.0; 64];
        for y in 0..8 {
            bars[y * 8 + 2] = 1.0;
            bars[y * 8 + 4] = 1.0;
        }
        let close = Morphology {
            op: MorphologyOp::Close,
            radius: 0.125,
        };
        let closed = close.apply(&raster(Edge::Wrap, bars)).unwrap();
        assert!((0..8).all(|y| closed.at(3, y) == 1.0));
        assert!((0..8).all(|y| closed.at(6, y) == 0.0));
    }

    #[test]
    fn tiles_match_whole_passes() {
        let values: Vec<f32> = (0..64).map(|i| ((i * 37) % 11) as f32 / 10.0).collect();
        for edge in [Edge::Wrap, Edge::Clamp] {
            let input = raster(edge, values.clone());
            for op in [
                MorphologyOp::Dilate,
                MorphologyOp::Erode,
                MorphologyOp::Open,
                MorphologyOp::Close,
            ] {
                let m = Morphology { op, radius: 0.2 };
                let whole = m.apply(&input).unwrap();
                let mut tiled = raster(edge, vec![0.0; 64]);
                for (x0, y0) in [(0, 0), (3, 0), (0, 5), (3, 5)] {
                    let rect = TexelRect {
                        x0,
                        y0,
                        x1: (x0 + 5).min(8),
                        y1: if y0 == 0 { 5 } else { 8 },
                    };
                    m.apply_into(&input, rect, &mut tiled).unwrap();
                }
                assert_eq!(tiled, whole, "{op:?} {edge:?}");
            }
        }
    }

    #[test]
    fn footprints_are_physical() {
        let m = Morphology {
            op: MorphologyOp::Close,
            radius: 0.002,
        };
        assert_eq!(m.footprint(Vec2::splat(1.0 / 1024.0)), Some([4, 4]));
        assert_eq!(m.footprint(Vec2::splat(1.0 / 4096.0)), Some([16, 16]));
        let bad = Morphology {
            op: MorphologyOp::Dilate,
            radius: -1.0,
        };
        assert!(bad.apply(&raster(Edge::Wrap, shapes())).is_err());
    }
}
