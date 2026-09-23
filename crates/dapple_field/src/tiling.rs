// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tile layouts: bonds (grids and running bonds) and herringbone.

use glam::Vec2;

use crate::domain::{Domain, DomainError, Footprint, Lattice};
use crate::field::ScalarField;
use crate::hash::{hash, key, unit_f32};

/// Purpose tag for per-tile hashes.
const TILE_TAG: u64 = 0x7469_6c65; // "tile"
/// Purpose tag for per-tile values.
const VALUE_TAG: u64 = 0x0076_616c_7565; // "value"

/// Longest herringbone tile, in tile widths.
pub const MAX_HERRINGBONE_RATIO: u32 = 16;

/// How tiles are laid.
///
/// Every tile is an axis-aligned rectangle, so a tile's border distance, its
/// local coordinates and its orientation are exact.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "kind", rename_all = "snake_case"))]
pub enum Pattern {
    /// Rows of equal tiles, `frequency` tiles per domain unit along each
    /// axis, each row shifted along x by `shift` of a tile from the row
    /// below it: 0 is a stack bond (a grid), 0.5 a running (stretcher) bond,
    /// and 1/3 or 1/4 the raking bonds.
    ///
    /// On a periodic domain the tile counts across the period must be whole,
    /// and so must the row count times `shift`, so the rows line up again.
    Bond {
        /// Tiles per domain unit, per axis.
        frequency: [f32; 2],
        /// Row-to-row shift as a fraction of a tile, in `[0, 1)`.
        shift: f32,
    },
    /// Herringbone: tiles `ratio` widths long, laid alternately along x and
    /// along y in stairs that climb to the upper right, `frequency` tile
    /// widths per domain unit.
    ///
    /// The layout repeats every `2 · ratio` widths along both axes; on a
    /// periodic domain that repeat must divide the period.
    Herringbone {
        /// Tile widths per domain unit.
        frequency: f32,
        /// Tile length in widths, `1..=`[`MAX_HERRINGBONE_RATIO`].
        ratio: u32,
    },
}

/// One tile lookup.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TileSample {
    /// Distance to the nearest border of the containing tile, in domain
    /// units: zero on joints, half the tile's width at its center line.
    pub edge: f32,
    /// Position along the tile's long axis (x for bond tiles), in `[0, 1)`.
    pub u: f32,
    /// Position across the tile, in `[0, 1)`.
    pub v: f32,
    /// Whether the tile's long axis runs along y.
    pub vertical: bool,
    /// Stable identifier of the tile. On periodic domains it is the same for
    /// every repeat of the tile.
    pub id: u64,
    /// A uniform random value in `[0, 1)` for the tile.
    pub value: f32,
}

/// Which [`TileSample`] quantity a [`TilingField`] returns.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum TileOutput {
    /// [`TileSample::edge`].
    Edge,
    /// [`TileSample::u`].
    U,
    /// [`TileSample::v`].
    V,
    /// 1 for [`TileSample::vertical`] tiles, 0 otherwise.
    Vertical,
    /// [`TileSample::value`].
    TileValue,
}

/// A tile layout over a domain.
///
/// Randomness per tile (its [`TileSample::value`]) is keyed by the tile's
/// lattice position, so it is independent of where the tile is sampled.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Tiling {
    domain: Domain,
    pattern: Pattern,
    lattice: Lattice,
    seed: u64,
}

impl Tiling {
    /// Lays `pattern` over `domain`.
    ///
    /// # Errors
    ///
    /// [`DomainError::InvalidFrequency`] for a non-positive frequency,
    /// [`DomainError::NonIntegerLattice`] when a periodic domain's period is
    /// not a whole number of tiles (or herringbone repeats), and
    /// [`DomainError::InvalidParameter`] for a `shift` outside `[0, 1)`, a
    /// bond whose rows do not line up again after one period, or a `ratio`
    /// outside `1..=`[`MAX_HERRINGBONE_RATIO`].
    pub fn new(domain: Domain, pattern: Pattern, seed: u64) -> Result<Self, DomainError> {
        let lattice = match pattern {
            Pattern::Bond { frequency, shift } => {
                if !(shift.is_finite() && (0.0..1.0).contains(&shift)) {
                    return Err(DomainError::InvalidParameter { name: "shift" });
                }
                let lattice = Lattice::new(domain, Vec2::from(frequency))?;
                if let Some([_, rows]) = lattice.wrap {
                    let turns = row_count(rows) * f64::from(shift);
                    if libm::trunc(turns) != turns {
                        return Err(DomainError::InvalidParameter { name: "shift" });
                    }
                }
                lattice
            }
            Pattern::Herringbone { frequency, ratio } => {
                if !(1..=MAX_HERRINGBONE_RATIO).contains(&ratio) {
                    return Err(DomainError::InvalidParameter { name: "ratio" });
                }
                let lattice = Lattice::new(domain, Vec2::splat(frequency))?;
                if let Some(cells) = lattice.wrap {
                    let repeat = 2 * i64::from(ratio);
                    for (axis, cells) in cells.into_iter().enumerate() {
                        if cells % repeat != 0 {
                            return Err(DomainError::NonIntegerLattice {
                                axis,
                                cells: cell_count(cells) / cell_count(repeat),
                            });
                        }
                    }
                }
                lattice
            }
        };
        Ok(Self {
            domain,
            pattern,
            lattice,
            seed,
        })
    }

    /// The domain this layout covers.
    #[must_use]
    pub const fn domain(&self) -> Domain {
        self.domain
    }

    /// The layout.
    #[must_use]
    pub const fn pattern(&self) -> Pattern {
        self.pattern
    }

    /// Selects one quantity as a scalar field.
    ///
    /// The field's band-limiting mean (see [`TilingField`]) is computed here:
    /// 0.5 for [`TileOutput::U`], [`TileOutput::V`] and
    /// [`TileOutput::TileValue`], the share of vertical tiles for
    /// [`TileOutput::Vertical`], and for [`TileOutput::Edge`] the average of
    /// a fixed 64 × 64 stratified sample over one repeat of the layout,
    /// accumulated in `f64`, so it is deterministic.
    #[must_use]
    pub fn output(self, output: TileOutput) -> TilingField {
        let mean = match (output, self.pattern) {
            (TileOutput::U | TileOutput::V | TileOutput::TileValue, _) => 0.5,
            (TileOutput::Vertical, Pattern::Bond { .. }) => 0.0,
            (TileOutput::Vertical, Pattern::Herringbone { .. }) => 0.5,
            (TileOutput::Edge, pattern) => {
                const SIDE: u32 = 64;
                // One repeat of the layout, in lattice cells.
                let cells = match pattern {
                    Pattern::Bond { .. } => Vec2::ONE,
                    Pattern::Herringbone { ratio, .. } => Vec2::splat(2.0 * ratio as f32),
                };
                let mut sum = 0.0_f64;
                for j in 0..SIDE {
                    for i in 0..SIDE {
                        let cell = Vec2::new(
                            (i as f32 + 0.5) * cells.x / SIDE as f32,
                            (j as f32 + 0.5) * cells.y / SIDE as f32,
                        );
                        sum += f64::from(self.sample(cell / self.lattice.frequency).edge);
                    }
                }
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the mean is narrowed back to the field's f32"
                )]
                let mean = (sum / f64::from(SIDE * SIDE)) as f32;
                mean
            }
        };
        TilingField {
            tiling: self,
            output,
            mean,
        }
    }

    /// Finds the tile containing `p`.
    #[must_use]
    pub fn sample(&self, p: Vec2) -> TileSample {
        self.sample_gradient::<false>(p).0
    }

    /// [`Tiling::sample`], and when `GRADIENT` the gradients of `edge`, `u`
    /// and `v` with respect to `p`.
    fn sample_gradient<const GRADIENT: bool>(&self, p: Vec2) -> (TileSample, [Vec2; 3]) {
        let at = self.lattice.locate(p);
        let [cx, cy] = at.cell;
        let f = at.frac;
        let frequency = self.lattice.frequency;
        // Each layout finds the tile's anchor cell, the position inside the
        // tile along and across its long axis in cells, and its extent.
        let (anchor, along, across, length, vertical) = match self.pattern {
            Pattern::Bond { shift, .. } => {
                let mut x = f.x - self.row_offset(cy, shift);
                let mut column = cx;
                if x < 0.0 {
                    x += 1.0;
                    column -= 1;
                }
                ([column, cy], x, f.y, 1.0, false)
            }
            Pattern::Herringbone { ratio, .. } => {
                let k = i64::from(ratio);
                let steps = cx - cy;
                let band = steps.div_euclid(2 * k);
                let r = steps - 2 * k * band;
                if r < k {
                    // A tile along x: the stair's tread.
                    let along = cell_count(r) + f.x;
                    ([cy + 2 * k * band, cy], along, f.y, ratio as f32, false)
                } else {
                    // A tile along y: the stair's riser.
                    let along = f.y + cell_count(2 * k - 1 - r);
                    ([cx, cy + r - 2 * k + 1], along, f.x, ratio as f32, true)
                }
            }
        };
        // Cell sizes along and across the tile, in domain units.
        let (step_along, step_across) = if vertical {
            (1.0 / frequency.y, 1.0 / frequency.x)
        } else {
            (1.0 / frequency.x, 1.0 / frequency.y)
        };
        let distances = [
            along * step_along,
            (length - along) * step_along,
            across * step_across,
            (1.0 - across) * step_across,
        ];
        let mut nearest = 0;
        for (index, distance) in distances.iter().enumerate() {
            if *distance < distances[nearest] {
                nearest = index;
            }
        }
        let [wx, wy] = self.lattice.wrap_cell(anchor);
        let id = hash(
            self.seed,
            &[TILE_TAG, key(wx), key(wy), u64::from(vertical)],
        );
        let sample = TileSample {
            edge: distances[nearest].max(0.0),
            u: (along / length).clamp(0.0, 1.0 - f32::EPSILON / 2.0),
            v: across.clamp(0.0, 1.0 - f32::EPSILON / 2.0),
            vertical,
            id,
            value: unit_f32(hash(id, &[VALUE_TAG])),
        };
        if !GRADIENT {
            return (sample, [Vec2::ZERO; 3]);
        }
        // Unit directions along and across the tile.
        let (axis_along, axis_across) = if vertical {
            (Vec2::Y, Vec2::X)
        } else {
            (Vec2::X, Vec2::Y)
        };
        let edge = match nearest {
            0 => axis_along,
            1 => -axis_along,
            2 => axis_across,
            _ => -axis_across,
        };
        (
            sample,
            [
                edge,
                axis_along / (length * step_along),
                axis_across / step_across,
            ],
        )
    }

    /// How far row `row`'s joints sit to the right of the lattice, in
    /// tiles, in `[0, 1)`.
    fn row_offset(&self, row: i64, shift: f32) -> f32 {
        // Rows repeat with the period; on the plane, keep the product exact
        // by folding the row onto a long repeat first.
        let row = match self.lattice.wrap {
            Some([_, rows]) => row.rem_euclid(rows),
            None => row.rem_euclid(1 << 24),
        };
        let turns = row_count(row) * f64::from(shift);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a fraction in [0, 1) narrows to f32"
        )]
        let offset = (turns - libm::floor(turns)) as f32;
        // Rounding can land exactly on 1.
        if offset >= 1.0 { 0.0 } else { offset }
    }
}

/// A small non-negative cell count as a float.
#[expect(
    clippy::cast_precision_loss,
    reason = "cell counts stay below 2^24, exact in f32 and f64"
)]
fn row_count(cells: i64) -> f64 {
    cells as f64
}

/// A small cell count as an `f32`.
#[expect(
    clippy::cast_precision_loss,
    reason = "cell counts stay below 2^24, exact in f32"
)]
fn cell_count(cells: i64) -> f32 {
    cells as f32
}

fn select(output: TileOutput, s: &TileSample) -> f32 {
    match output {
        TileOutput::Edge => s.edge,
        TileOutput::U => s.u,
        TileOutput::V => s.v,
        TileOutput::Vertical => {
            if s.vertical {
                1.0
            } else {
                0.0
            }
        }
        TileOutput::TileValue => s.value,
    }
}

/// One [`TileOutput`] of a [`Tiling`], as a scalar field.
///
/// **Band limiting.** Like [`CellularField`](crate::CellularField), the
/// field fades toward the output's mean with [`Footprint::band_weight`] of
/// the tile frequency: full detail while tiles span at least four
/// footprints, only the mean below two. Joints within that range are not
/// filtered separately; soften them where the edge distance is shaped.
///
/// Its gradient is analytic: the edge distance moves at unit speed away
/// from its nearest joint, `u` and `v` at the inverse of the tile's length
/// and width, and the orientation and tile value not at all.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TilingField {
    tiling: Tiling,
    output: TileOutput,
    mean: f32,
}

impl TilingField {
    /// The value coarse footprints fade toward.
    #[must_use]
    pub const fn mean(&self) -> f32 {
        self.mean
    }
}

impl ScalarField for TilingField {
    fn domain(&self) -> Domain {
        self.tiling.domain
    }

    fn eval(&self, p: Vec2, footprint: Footprint) -> f32 {
        let weight = footprint.band_weight(self.tiling.lattice.max_frequency());
        if weight == 0.0 {
            return self.mean;
        }
        let value = select(self.output, &self.tiling.sample(p));
        if weight == 1.0 {
            value
        } else {
            self.mean + (value - self.mean) * weight
        }
    }

    fn eval_gradient(&self, p: Vec2, footprint: Footprint) -> (f32, Vec2) {
        let weight = footprint.band_weight(self.tiling.lattice.max_frequency());
        if weight == 0.0 {
            return (self.mean, Vec2::ZERO);
        }
        let (sample, [edge, u, v]) = self.tiling.sample_gradient::<true>(p);
        let value = select(self.output, &sample);
        let gradient = match self.output {
            TileOutput::Edge => edge,
            TileOutput::U => u,
            TileOutput::V => v,
            TileOutput::Vertical | TileOutput::TileValue => Vec2::ZERO,
        };
        if weight == 1.0 {
            (value, gradient)
        } else {
            (self.mean + (value - self.mean) * weight, gradient * weight)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bond(shift: f32) -> Tiling {
        Tiling::new(
            Domain::Plane,
            Pattern::Bond {
                frequency: [4.0, 8.0],
                shift,
            },
            3,
        )
        .unwrap()
    }

    #[test]
    fn running_bond_shifts_alternate_rows() {
        let tiles = bond(0.5);
        // Row 0 has joints at x = 0, 0.25, …; row 1 at 0.125, 0.375, ….
        let a = tiles.sample(Vec2::new(0.01, 0.06));
        let b = tiles.sample(Vec2::new(0.01, 0.19));
        assert!((a.edge - 0.01).abs() < 1e-6, "{a:?}");
        assert!((b.u - (0.01 + 0.125) / 0.25).abs() < 1e-5, "{b:?}");
        // Both sides of a row-1 joint are different tiles.
        let left = tiles.sample(Vec2::new(0.12, 0.19));
        let right = tiles.sample(Vec2::new(0.13, 0.19));
        assert_ne!(left.id, right.id);
        assert!(left.u > 0.9 && right.u < 0.1);
        // A bed joint is 1/16 unit from the row center.
        let center = tiles.sample(Vec2::new(0.125, 0.0625));
        assert!((center.edge - 0.0625).abs() < 1e-6, "{center:?}");
    }

    #[test]
    fn herringbone_tiles_the_plane_with_whole_tiles() {
        for ratio in [1, 2, 3, 4] {
            let tiles = Tiling::new(
                Domain::Plane,
                Pattern::Herringbone {
                    frequency: 10.0,
                    ratio,
                },
                9,
            )
            .unwrap();
            // Every tile covers `ratio` cells: count cell centers per id.
            let mut counts = alloc::collections::BTreeMap::new();
            let side = 12 * i32::try_from(ratio).unwrap();
            let mut vertical = 0;
            for j in -side..side {
                for i in -side..side {
                    let p = Vec2::new(i as f32 + 0.5, j as f32 + 0.5) / 10.0;
                    let s = tiles.sample(p);
                    assert!((0.0..=0.05 + 1e-6).contains(&s.edge), "{s:?}");
                    vertical += usize::from(s.vertical);
                    *counts.entry(s.id).or_insert(0_u32) += 1;
                }
            }
            // Tiles cut by the window's border hold fewer cells; every tile
            // fully inside holds exactly `ratio`.
            let whole = counts.values().filter(|c| **c == ratio).count();
            assert!(counts.values().all(|c| *c <= ratio), "ratio {ratio}");
            assert!(whole * 10 > counts.len() * 8, "ratio {ratio}");
            let total = (2 * side * 2 * side) as usize;
            assert_eq!(
                vertical * 2,
                total,
                "ratio {ratio}: half the area is vertical"
            );
        }
    }

    #[test]
    fn herringbone_neighbors_abut_at_joints() {
        let tiles = Tiling::new(
            Domain::Plane,
            Pattern::Herringbone {
                frequency: 1.0,
                ratio: 2,
            },
            1,
        )
        .unwrap();
        // The tread in row 0 spans x in [0, 2); the riser beside it spans
        // x in [2, 3), y in [-1, 1).
        let tread = tiles.sample(Vec2::new(1.9, 0.5));
        let riser = tiles.sample(Vec2::new(2.1, 0.5));
        assert!(!tread.vertical && riser.vertical);
        assert!((tread.edge - 0.1).abs() < 1e-5 && (riser.edge - 0.1).abs() < 1e-5);
        assert!((riser.u - 0.75).abs() < 1e-5, "{riser:?}");
        assert_eq!(riser.id, tiles.sample(Vec2::new(2.5, -0.9)).id);
    }

    #[test]
    fn periodic_layouts_repeat_ids() {
        let domain = Domain::periodic(1, 1).unwrap();
        for pattern in [
            Pattern::Bond {
                frequency: [4.0, 12.0],
                shift: 0.25,
            },
            Pattern::Herringbone {
                frequency: 12.0,
                ratio: 3,
            },
        ] {
            let tiles = Tiling::new(domain, pattern, 5).unwrap();
            for i in 0..64 {
                let p = Vec2::new(i as f32 * 0.0137, i as f32 * 0.0291);
                let a = tiles.sample(p);
                let b = tiles.sample(p + Vec2::new(1.0, -2.0));
                assert_eq!(a.id, b.id, "{pattern:?} at {p}");
                assert!((a.edge - b.edge).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn layouts_that_would_not_tile_are_refused() {
        let domain = Domain::periodic(1, 1).unwrap();
        let bond = |frequency, shift| Tiling::new(domain, Pattern::Bond { frequency, shift }, 0);
        assert!(bond([4.0, 4.0], 0.5).is_ok());
        assert!(bond([4.0, 3.0], 0.5).is_err(), "3 rows at a half shift");
        assert!(bond([4.5, 4.0], 0.5).is_err());
        assert!(bond([4.0, 4.0], 1.0).is_err());
        let herringbone =
            |frequency, ratio| Tiling::new(domain, Pattern::Herringbone { frequency, ratio }, 0);
        assert!(herringbone(8.0, 2).is_ok());
        assert!(herringbone(6.0, 2).is_err(), "the repeat is 4 widths");
        assert!(herringbone(8.0, 0).is_err());
        assert!(herringbone(8.0, MAX_HERRINGBONE_RATIO + 1).is_err());
    }

    #[test]
    fn gradients_match_finite_differences() {
        let fields = [
            bond(0.5).output(TileOutput::Edge),
            bond(0.5).output(TileOutput::U),
            Tiling::new(
                Domain::Plane,
                Pattern::Herringbone {
                    frequency: 3.0,
                    ratio: 3,
                },
                2,
            )
            .unwrap()
            .output(TileOutput::V),
        ];
        let h = 1e-3;
        for field in fields {
            for i in 0..32 {
                let p = Vec2::new(0.113 + i as f32 * 0.071, 0.057 + i as f32 * 0.043);
                let (_, g) = field.eval_gradient(p, Footprint::POINT);
                let dx = (field.eval(p + Vec2::X * h, Footprint::POINT)
                    - field.eval(p - Vec2::X * h, Footprint::POINT))
                    / (2.0 * h);
                let dy = (field.eval(p + Vec2::Y * h, Footprint::POINT)
                    - field.eval(p - Vec2::Y * h, Footprint::POINT))
                    / (2.0 * h);
                // Skip samples straddling a joint or the edge's ridge.
                if (dx - g.x).abs() > 0.05 * g.length().max(1.0)
                    || (dy - g.y).abs() > 0.05 * g.length().max(1.0)
                {
                    let near_joint = field.tiling.sample(p).edge < 2.0 * h
                        || field.tiling.sample(p + Vec2::X * h).id
                            != field.tiling.sample(p - Vec2::X * h).id
                        || field.tiling.sample(p + Vec2::Y * h).id
                            != field.tiling.sample(p - Vec2::Y * h).id;
                    let on_ridge = field.output == TileOutput::Edge;
                    assert!(near_joint || on_ridge, "{p}: {g} vs ({dx}, {dy})");
                }
            }
        }
    }

    #[test]
    fn coarse_footprints_fade_to_the_mean() {
        let field = bond(0.5).output(TileOutput::Edge);
        let coarse = Footprint::new(1.0).unwrap();
        assert_eq!(field.eval(Vec2::new(0.3, 0.2), coarse), field.mean());
        // The mean edge distance of a 1/4 × 1/8 tile lies inside its range.
        assert!(field.mean() > 0.0 && field.mean() < 0.0625);
    }
}
