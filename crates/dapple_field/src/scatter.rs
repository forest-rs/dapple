// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Scatter: stamps splatted at random points, with bounded overlap.

use glam::Vec2;

use crate::domain::{Domain, DomainError, Footprint, Lattice};
use crate::field::ScalarField;
use crate::hash::{hash, key, unit_f32};
use crate::image::SampleImage;

/// Purpose tag for per-splat hashes.
const SPLAT_TAG: u64 = 0x0073_706c_6174; // "splat"

/// Which shape each splat stamps.
///
/// A stamp is defined over the unit disk's bounding square, `[-1, 1]²` in
/// splat coordinates, and is zero outside it.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "kind", rename_all = "snake_case"))]
pub enum Stamp {
    /// A filled unit disk whose edge falls along a smoothstep over the
    /// outermost `softness` of its radius (widened to the footprint, so it
    /// stays antialiased, and capped at the whole radius).
    Disk {
        /// Edge band width, as a fraction of the splat's radius.
        softness: f32,
    },
    /// A hemisphere's height, `sqrt(1 − r²)`: pebbles, bumps and blisters.
    Dome,
    /// An image, its level 0 extent stretched over `[-1, 1]²`: leaves,
    /// petals, stones drawn as coverage or height. It must be a
    /// [`Domain::Plane`] image; it is read with its mips, at the footprint
    /// scaled into the stamp.
    #[cfg_attr(feature = "serde", serde(skip))]
    Image(SampleImage),
}

/// How splats are placed.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Placement {
    /// Candidate cells per domain unit: at most one splat per cell.
    pub frequency: f32,
    /// Probability that a cell holds a splat, in `[0, 1]`.
    pub density: f32,
    /// Smallest and largest splat radius, in cells, within `(0, 1]`.
    pub radius: [f32; 2],
    /// Whether each splat is turned by a random angle.
    pub rotate: bool,
}

/// Which quantity of the splats a [`ScatterField`] returns.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum ScatterOutput {
    /// The union of the stamps as coverage, `1 − Π(1 − aᵢ)` with each stamp
    /// value clamped to `[0, 1]`: independent of splat order.
    Coverage,
    /// The largest stamp value: the height of overlapping pebbles.
    Max,
    /// The random value in `[0, 1)` of the splat that gives
    /// [`ScatterOutput::Max`] (the higher value on ties), or 0 where no
    /// splat reaches: a per-splat tone.
    TopValue,
}

/// One splat around a query point.
#[derive(Copy, Clone, Debug)]
struct Splat {
    /// The query point in splat coordinates.
    local: Vec2,
    /// Splat coordinates per domain unit (the inverse radius).
    scale: f32,
    /// Rotation from domain axes to splat axes, as (cos, sin).
    turn: Vec2,
    value: f32,
}

/// Stamps scattered at jittered points of a lattice.
///
/// Each lattice cell holds a splat with probability `density`, centered at
/// a random point of the cell, with a random radius and, optionally, a
/// random turn. Radii are at most one cell, so a stamp's square reaches
/// less than two cells from its center and every point is covered only by
/// splats of the 5 × 5 cells around it: evaluation is local and its cost
/// bounded, whatever the density.
#[derive(Clone, Debug, PartialEq)]
pub struct Scatter {
    domain: Domain,
    lattice: Lattice,
    placement: Placement,
    stamp: Stamp,
    seed: u64,
}

impl Scatter {
    /// Scatters `stamp` over `domain`.
    ///
    /// # Errors
    ///
    /// [`DomainError::InvalidFrequency`] or
    /// [`DomainError::NonIntegerLattice`] for a frequency that is not
    /// positive or does not fit the period, and
    /// [`DomainError::InvalidParameter`] for a density outside `[0, 1]`,
    /// radii outside `(0, 1]` or out of order, a negative disk softness, or
    /// an image stamp on a periodic domain.
    pub fn new(
        domain: Domain,
        placement: Placement,
        stamp: Stamp,
        seed: u64,
    ) -> Result<Self, DomainError> {
        if !(0.0..=1.0).contains(&placement.density) {
            return Err(DomainError::InvalidParameter { name: "density" });
        }
        let [small, large] = placement.radius;
        if !(small > 0.0 && small <= large && large <= 1.0) {
            return Err(DomainError::InvalidParameter { name: "radius" });
        }
        match &stamp {
            Stamp::Disk { softness } if !(softness.is_finite() && *softness >= 0.0) => {
                return Err(DomainError::InvalidParameter { name: "softness" });
            }
            Stamp::Image(image) if image.domain() != Domain::Plane => {
                return Err(DomainError::InvalidParameter { name: "stamp" });
            }
            _ => {}
        }
        Ok(Self {
            domain,
            lattice: Lattice::new(domain, Vec2::splat(placement.frequency))?,
            placement,
            stamp,
            seed,
        })
    }

    /// The domain the splats cover.
    #[must_use]
    pub const fn domain(&self) -> Domain {
        self.domain
    }

    /// Selects one quantity as a scalar field.
    ///
    /// The band-limiting mean (see [`ScatterField`]) is the average of a
    /// fixed 64 × 64 stratified sample over an 8 × 8-cell block, accumulated
    /// in `f64`, so it is deterministic.
    #[must_use]
    pub fn output(self, output: ScatterOutput) -> ScatterField {
        const SIDE: u32 = 64;
        const CELLS: f32 = 8.0;
        let footprint = Footprint::POINT;
        let mut sum = 0.0_f64;
        for j in 0..SIDE {
            for i in 0..SIDE {
                let cell = Vec2::new(
                    (i as f32 + 0.5) * CELLS / SIDE as f32,
                    (j as f32 + 0.5) * CELLS / SIDE as f32,
                );
                let p = cell / self.lattice.frequency;
                sum += f64::from(self.evaluate::<false>(p, footprint, output).0);
            }
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the mean is narrowed back to the field's f32"
        )]
        let mean = (sum / f64::from(SIDE * SIDE)) as f32;
        ScatterField {
            scatter: self,
            output,
            mean,
        }
    }

    /// The splat of cell `cell + offset`, relative to the query point `q`
    /// inside `cell` (both in cells), if the cell holds one.
    fn splat(&self, cell: [i64; 2], offset: [i8; 2], q: Vec2) -> Option<Splat> {
        let [wx, wy] = self.lattice.wrap_cell([
            cell[0] + i64::from(offset[0]),
            cell[1] + i64::from(offset[1]),
        ]);
        let id = hash(self.seed, &[SPLAT_TAG, key(wx), key(wy)]);
        if unit_f32(hash(id, &[0])) >= self.placement.density {
            return None;
        }
        let center = Vec2::new(
            f32::from(offset[0]) + unit_f32(hash(id, &[1])),
            f32::from(offset[1]) + unit_f32(hash(id, &[2])),
        );
        let [small, large] = self.placement.radius;
        let radius = small + (large - small) * unit_f32(hash(id, &[3]));
        let turn = if self.placement.rotate {
            let angle = unit_f32(hash(id, &[4])) * core::f32::consts::TAU;
            Vec2::new(libm::cosf(angle), libm::sinf(angle))
        } else {
            Vec2::X
        };
        let d = (q - center) / radius;
        // Rotate by −angle: domain axes into splat axes.
        let local = Vec2::new(d.x * turn.x + d.y * turn.y, d.y * turn.x - d.x * turn.y);
        Some(Splat {
            local,
            scale: self.lattice.frequency.x / radius,
            turn,
            value: unit_f32(hash(id, &[5])),
        })
    }

    /// The stamp's value at `local`, for a footprint `width` splat units
    /// wide, and when `GRADIENT` its gradient in splat units.
    fn stamp<const GRADIENT: bool>(&self, local: Vec2, width: f32) -> (f32, Vec2) {
        if local.x.abs() >= 1.0 || local.y.abs() >= 1.0 {
            return (0.0, Vec2::ZERO);
        }
        match self.stamp {
            Stamp::Disk { softness } => {
                let r = local.length();
                let band = softness.max(width).min(1.0);
                if band <= 0.0 {
                    return (if r <= 1.0 { 1.0 } else { 0.0 }, Vec2::ZERO);
                }
                let t = ((1.0 - r) / band).clamp(0.0, 1.0);
                let value = t * t * (3.0 - 2.0 * t);
                if !GRADIENT || t <= 0.0 || t >= 1.0 || r == 0.0 {
                    return (value, Vec2::ZERO);
                }
                (value, local / r * (-6.0 * t * (1.0 - t) / band))
            }
            Stamp::Dome => {
                let rest = 1.0 - local.length_squared();
                if rest <= 0.0 {
                    return (0.0, Vec2::ZERO);
                }
                let value = libm::sqrtf(rest);
                // The rim is vertical; keep the slope finite.
                (value, -local / value.max(1e-3))
            }
            Stamp::Image(ref image) => {
                let level = &image.levels()[0];
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "image sizes are far below f32's exact integer range"
                )]
                let extent = Vec2::new(level.width() as f32, level.height() as f32) * level.texel();
                let at = image.origin() + (local + Vec2::ONE) * 0.5 * extent;
                let footprint =
                    Footprint::new(width * 0.5 * extent.max_element()).unwrap_or(Footprint::POINT);
                if GRADIENT {
                    let (value, gradient) = image.sample_gradient(at, footprint);
                    (value, gradient * extent * 0.5)
                } else {
                    (image.sample(at, footprint), Vec2::ZERO)
                }
            }
        }
    }

    /// `output` at `p` and, when `GRADIENT`, its gradient per domain unit.
    fn evaluate<const GRADIENT: bool>(
        &self,
        p: Vec2,
        footprint: Footprint,
        output: ScatterOutput,
    ) -> (f32, Vec2) {
        let at = self.lattice.locate(p);
        let mut uncovered = 1.0_f32;
        let mut coverage_gradient = Vec2::ZERO;
        let mut best: Option<(f32, f32, Vec2)> = None;
        for dy in -2..=2 {
            for dx in -2..=2 {
                let Some(splat) = self.splat(at.cell, [dx, dy], at.frac) else {
                    continue;
                };
                let width = footprint.width() * splat.scale;
                let (value, local_gradient) = self.stamp::<GRADIENT>(splat.local, width);
                if value <= 0.0 && local_gradient == Vec2::ZERO {
                    continue;
                }
                // Splat units per domain unit, rotated back to domain axes.
                let gradient = if GRADIENT {
                    let g = local_gradient * splat.scale;
                    Vec2::new(
                        g.x * splat.turn.x - g.y * splat.turn.y,
                        g.x * splat.turn.y + g.y * splat.turn.x,
                    )
                } else {
                    Vec2::ZERO
                };
                let a = value.clamp(0.0, 1.0);
                if GRADIENT && a > 0.0 && a < 1.0 {
                    // d(1 − Π(1 − aᵢ)) = Σ Π_{j≠i}(1 − aⱼ) daᵢ: scale the
                    // running sum by this splat's (1 − a) and add its term.
                    coverage_gradient = coverage_gradient * (1.0 - a) + gradient * uncovered;
                } else if GRADIENT {
                    coverage_gradient *= 1.0 - a;
                }
                uncovered *= 1.0 - a;
                let wins = match best {
                    None => value > 0.0,
                    Some((top, top_value, _)) => {
                        value > top || (value == top && splat.value > top_value)
                    }
                };
                if wins {
                    best = Some((value, splat.value, gradient));
                }
            }
        }
        match output {
            ScatterOutput::Coverage => (1.0 - uncovered, coverage_gradient),
            ScatterOutput::Max => best.map_or((0.0, Vec2::ZERO), |(v, _, g)| (v, g)),
            ScatterOutput::TopValue => (best.map_or(0.0, |(_, v, _)| v), Vec2::ZERO),
        }
    }
}

/// One [`ScatterOutput`] of a [`Scatter`], as a scalar field.
///
/// **Band limiting.** Stamps are read at the footprint scaled into splat
/// units (a disk's edge widens, an image reads coarser mips), and as cells
/// shrink below a few footprints the field fades toward its mean with
/// [`Footprint::band_weight`] of the cell frequency, like
/// [`CellularField`](crate::CellularField).
///
/// Its gradient is analytic for disks, domes and images, through each
/// splat's scale and turn; per-splat values have none.
#[derive(Clone, Debug, PartialEq)]
pub struct ScatterField {
    scatter: Scatter,
    output: ScatterOutput,
    mean: f32,
}

impl ScatterField {
    /// The value coarse footprints fade toward.
    #[must_use]
    pub const fn mean(&self) -> f32 {
        self.mean
    }
}

impl ScalarField for ScatterField {
    fn domain(&self) -> Domain {
        self.scatter.domain
    }

    fn eval(&self, p: Vec2, footprint: Footprint) -> f32 {
        let weight = footprint.band_weight(self.scatter.lattice.max_frequency());
        if weight == 0.0 {
            return self.mean;
        }
        let value = self.scatter.evaluate::<false>(p, footprint, self.output).0;
        if weight == 1.0 {
            value
        } else {
            self.mean + (value - self.mean) * weight
        }
    }

    fn eval_gradient(&self, p: Vec2, footprint: Footprint) -> (f32, Vec2) {
        let weight = footprint.band_weight(self.scatter.lattice.max_frequency());
        if weight == 0.0 {
            return (self.mean, Vec2::ZERO);
        }
        let (value, gradient) = self.scatter.evaluate::<true>(p, footprint, self.output);
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

    fn placement(density: f32) -> Placement {
        Placement {
            frequency: 8.0,
            density,
            radius: [0.3, 0.9],
            rotate: true,
        }
    }

    fn points() -> impl Iterator<Item = Vec2> {
        (0..400).map(|i| Vec2::new((i % 20) as f32 * 0.0513, (i / 20) as f32 * 0.0497))
    }

    #[test]
    fn empty_and_full_density() {
        let none = Scatter::new(Domain::Plane, placement(0.0), Stamp::Dome, 1).unwrap();
        let field = none.output(ScatterOutput::Coverage);
        assert!(points().all(|p| field.eval(p, Footprint::POINT) == 0.0));
        assert_eq!(field.mean(), 0.0);
        let full = Scatter::new(
            Domain::Plane,
            placement(1.0),
            Stamp::Disk { softness: 0.1 },
            1,
        )
        .unwrap()
        .output(ScatterOutput::Coverage);
        let covered = points()
            .filter(|p| full.eval(*p, Footprint::POINT) > 0.5)
            .count();
        assert!(covered > 200, "{covered} of 400 covered");
    }

    #[test]
    fn splats_have_their_own_values_and_domes_peak_at_one() {
        let scatter = Scatter::new(Domain::Plane, placement(0.7), Stamp::Dome, 4).unwrap();
        let top = scatter.clone().output(ScatterOutput::TopValue);
        let max = scatter.output(ScatterOutput::Max);
        let mut values: alloc::vec::Vec<f32> = points()
            .map(|p| top.eval(p, Footprint::POINT))
            .filter(|v| *v > 0.0)
            .collect();
        values.sort_by(f32::total_cmp);
        values.dedup();
        assert!(values.len() > 10, "{} distinct splat values", values.len());
        assert!(points().all(|p| (0.0..=1.0).contains(&max.eval(p, Footprint::POINT))));
    }

    #[test]
    fn periodic_scatter_repeats() {
        let domain = Domain::periodic(1, 1).unwrap();
        let field = Scatter::new(domain, placement(0.6), Stamp::Disk { softness: 0.2 }, 9)
            .unwrap()
            .output(ScatterOutput::Coverage);
        for p in points() {
            let a = field.eval(p, Footprint::POINT);
            let b = field.eval(p + Vec2::new(-1.0, 3.0), Footprint::POINT);
            assert!((a - b).abs() < 1e-4, "{p}: {a} vs {b}");
        }
    }

    #[test]
    fn invalid_placements_are_refused() {
        let bad = |placement| Scatter::new(Domain::Plane, placement, Stamp::Dome, 0).is_err();
        assert!(bad(Placement {
            density: 1.5,
            ..placement(1.0)
        }));
        assert!(bad(Placement {
            radius: [0.5, 1.2],
            ..placement(1.0)
        }));
        assert!(bad(Placement {
            radius: [0.6, 0.5],
            ..placement(1.0)
        }));
        assert!(
            Scatter::new(
                Domain::periodic(1, 1).unwrap(),
                Placement {
                    frequency: 2.5,
                    ..placement(1.0)
                },
                Stamp::Dome,
                0
            )
            .is_err()
        );
    }

    #[test]
    fn gradients_match_finite_differences() {
        for (stamp, output) in [
            (Stamp::Disk { softness: 0.5 }, ScatterOutput::Coverage),
            (Stamp::Dome, ScatterOutput::Max),
        ] {
            let field = Scatter::new(Domain::Plane, placement(0.5), stamp, 2)
                .unwrap()
                .output(output);
            let h = 1e-4;
            let mut checked = 0;
            for p in points() {
                let (v, g) = field.eval_gradient(p, Footprint::POINT);
                // Near a dome's rim the slope steepens too fast for finite
                // differences in f32.
                if v <= 0.2 || v >= 0.98 {
                    continue;
                }
                let dx = (field.eval(p + Vec2::X * h, Footprint::POINT)
                    - field.eval(p - Vec2::X * h, Footprint::POINT))
                    / (2.0 * h);
                let dy = (field.eval(p + Vec2::Y * h, Footprint::POINT)
                    - field.eval(p - Vec2::Y * h, Footprint::POINT))
                    / (2.0 * h);
                let error = (Vec2::new(dx, dy) - g).length();
                assert!(
                    error < 0.05 * g.length().max(1.0),
                    "{p}: {g} vs ({dx}, {dy})"
                );
                checked += 1;
            }
            assert!(checked > 20, "{checked} samples checked");
        }
    }
}
