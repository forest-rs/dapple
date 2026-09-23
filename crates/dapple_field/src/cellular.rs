// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Cellular (Worley) noise.

use glam::Vec2;

use crate::domain::{Domain, DomainError, Footprint, Lattice};
use crate::field::ScalarField;
use crate::hash::{hash, key, unit_f32};

/// Purpose tag for feature-point hashes.
const FEATURE_TAG: u64 = 0x0066_6561_7475_7265; // "feature"
/// Purpose tag for per-cell values.
const VALUE_TAG: u64 = 0x0076_616c_7565; // "value"

/// One cellular lookup.
///
/// Distances are in lattice-cell units: a frequency of 8 cells per unit makes
/// a distance of 1 equal to 1/8 domain unit. Unequal axis frequencies measure
/// in the stretched space, which elongates the cells.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CellSample {
    /// Distance to the nearest feature point.
    pub f1: f32,
    /// Distance to the second-nearest feature point.
    pub f2: f32,
    /// Exact distance to the border of the nearest point's Voronoi cell.
    pub border: f32,
    /// Stable identifier of the nearest point's cell. On periodic domains it
    /// is the same for every repeat of the cell.
    pub id: u64,
    /// A uniform random value in `[0, 1)` for the nearest point's cell.
    pub value: f32,
}

/// Which [`CellSample`] quantity a [`CellularField`] returns.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum CellOutput {
    /// [`CellSample::f1`].
    F1,
    /// [`CellSample::f2`].
    F2,
    /// `f2 − f1`: zero on cell borders, rising toward cell centers.
    F2MinusF1,
    /// [`CellSample::border`].
    Border,
    /// [`CellSample::value`].
    CellValue,
}

/// Worley noise with one jittered feature point per lattice cell.
///
/// With `jitter` in `[0, 1]`, every feature point lies in its own cell. For a
/// query in cell `c`, the nearest two points are then always within distance
/// √3.25 < 2, and every cell two or more steps away is at least 2 away, so
/// searching the 5 × 5 block around `c` is exact for `f1`, `f2`, and `border`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Cellular {
    domain: Domain,
    lattice: Lattice,
    jitter: f32,
    seed: u64,
}

impl Cellular {
    /// Builds cellular noise with `frequency` cells per unit.
    ///
    /// `jitter` is how far points may stray from their cell centers: 0 gives a
    /// regular grid, 1 the classic irregular pattern.
    pub fn new(
        domain: Domain,
        frequency: Vec2,
        jitter: f32,
        seed: u64,
    ) -> Result<Self, DomainError> {
        if !(0.0..=1.0).contains(&jitter) {
            return Err(DomainError::InvalidParameter { name: "jitter" });
        }
        Ok(Self {
            domain,
            lattice: Lattice::new(domain, frequency)?,
            jitter,
            seed,
        })
    }

    /// The domain this noise is defined over.
    #[must_use]
    pub const fn domain(&self) -> Domain {
        self.domain
    }

    /// Selects one quantity as a scalar field.
    ///
    /// The field's band-limiting mean (see [`CellularField`]) is computed
    /// here: 0.5 for [`CellOutput::CellValue`], and otherwise the average of
    /// a fixed 32 × 32 stratified sample over an 8 × 8-cell block of this
    /// noise, accumulated in `f64`, so it is deterministic.
    #[must_use]
    pub fn output(self, output: CellOutput) -> CellularField {
        let mean = match output {
            CellOutput::CellValue => 0.5,
            _ => {
                const SIDE: u32 = 32;
                const CELLS: f32 = 8.0;
                let frequency = self.lattice.frequency;
                let mut sum = 0.0_f64;
                for j in 0..SIDE {
                    for i in 0..SIDE {
                        let cell = Vec2::new(
                            (i as f32 + 0.5) * CELLS / SIDE as f32,
                            (j as f32 + 0.5) * CELLS / SIDE as f32,
                        );
                        sum += f64::from(select(output, &self.sample(cell / frequency)));
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
        CellularField {
            cellular: self,
            output,
            mean,
        }
    }

    /// Feature point of the cell `offset` steps from `cell`, relative to
    /// `cell`'s corner.
    ///
    /// Working relative to the query cell keeps precision independent of how
    /// far the query is from the origin, and makes periodic repeats
    /// bit-identical.
    fn feature(&self, cell: [i64; 2], offset: [i8; 2]) -> (Vec2, u64) {
        let [wx, wy] = self.lattice.wrap_cell([
            cell[0] + i64::from(offset[0]),
            cell[1] + i64::from(offset[1]),
        ]);
        let id = hash(self.seed, &[FEATURE_TAG, key(wx), key(wy)]);
        let jitter = Vec2::new(
            unit_f32(hash(id, &[0])) - 0.5,
            unit_f32(hash(id, &[1])) - 0.5,
        ) * self.jitter;
        let corner = Vec2::new(f32::from(offset[0]), f32::from(offset[1]));
        (corner + Vec2::splat(0.5) + jitter, id)
    }

    /// Looks up the nearest feature points around `p`.
    #[must_use]
    pub fn sample(&self, p: Vec2) -> CellSample {
        self.sample_gradient::<false>(p).0
    }

    /// [`Cellular::sample`], and when `GRADIENT` the gradients of `f1`,
    /// `f2` and `border` with respect to the in-cell position.
    fn sample_gradient<const GRADIENT: bool>(&self, p: Vec2) -> (CellSample, [Vec2; 3]) {
        let at = self.lattice.locate(p);
        let q = at.frac;

        let mut points = [(Vec2::ZERO, 0_u64); 25];
        let mut nearest = (f32::INFINITY, 0);
        let mut second = f32::INFINITY;
        let mut second_index = 0;
        for (slot, (dy, dx)) in points
            .iter_mut()
            .zip((-2..=2).flat_map(|dy| (-2..=2).map(move |dx| (dy, dx))))
        {
            *slot = self.feature(at.cell, [dx, dy]);
        }
        for (index, (point, _)) in points.iter().enumerate() {
            let d = (*point - q).length_squared();
            if d < nearest.0 {
                second = nearest.0;
                second_index = nearest.1;
                nearest = (d, index);
            } else if d < second {
                second = d;
                second_index = index;
            }
        }
        let (near_point, id) = points[nearest.1];
        let mut border = f32::INFINITY;
        let mut border_axis = Vec2::ZERO;
        for (index, (point, _)) in points.iter().enumerate() {
            if index == nearest.1 {
                continue;
            }
            // Distance from q to the bisector between the nearest point and this one.
            let axis = *point - near_point;
            let length = axis.length();
            if length > 0.0 {
                let midpoint = (*point + near_point) * 0.5;
                let distance = (midpoint - q).dot(axis) / length;
                if GRADIENT && distance < border {
                    border_axis = axis / length;
                }
                border = border.min(distance);
            }
        }
        let sample = CellSample {
            f1: libm::sqrtf(nearest.0),
            f2: libm::sqrtf(second),
            border,
            id,
            value: unit_f32(hash(id, &[VALUE_TAG])),
        };
        if !GRADIENT {
            return (sample, [Vec2::ZERO; 3]);
        }
        // A distance's gradient is the unit vector away from its point.
        let away = |point: Vec2, distance: f32| {
            if distance > 0.0 {
                (q - point) / distance
            } else {
                Vec2::ZERO
            }
        };
        (
            sample,
            [
                away(near_point, sample.f1),
                away(points[second_index].0, sample.f2),
                -border_axis,
            ],
        )
    }
}

fn select(output: CellOutput, s: &CellSample) -> f32 {
    match output {
        CellOutput::F1 => s.f1,
        CellOutput::F2 => s.f2,
        CellOutput::F2MinusF1 => s.f2 - s.f1,
        CellOutput::Border => s.border,
        CellOutput::CellValue => s.value,
    }
}

/// One [`CellOutput`] of a [`Cellular`] noise, as a scalar field.
///
/// **Band limiting.** The pattern repeats at the cell frequency, so cells
/// smaller than a few footprints alias. The field fades toward the output's
/// mean with [`Footprint::band_weight`] of the cell frequency: full detail
/// while cells span at least four footprints, only the mean below two. At full
/// weight the value is returned exactly, and at zero weight the noise is not
/// sampled. Detail within a cell (its edges) is not filtered separately.
///
/// Its gradient ([`ScalarField::eval_gradient`]) is analytic: F1 and F2
/// move with the unit vector away from their feature points, the border
/// distance against the normal of its nearest bisector, and cell values not
/// at all. On a cell border itself the gradient is that of the side the
/// sample picked.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CellularField {
    cellular: Cellular,
    output: CellOutput,
    mean: f32,
}

impl CellularField {
    /// The value coarse footprints fade toward.
    #[must_use]
    pub const fn mean(&self) -> f32 {
        self.mean
    }
}

impl ScalarField for CellularField {
    fn domain(&self) -> Domain {
        self.cellular.domain
    }

    fn eval(&self, p: Vec2, footprint: Footprint) -> f32 {
        let weight = footprint.band_weight(self.cellular.lattice.max_frequency());
        if weight == 0.0 {
            return self.mean;
        }
        let value = select(self.output, &self.cellular.sample(p));
        if weight == 1.0 {
            value
        } else {
            self.mean + (value - self.mean) * weight
        }
    }

    fn eval_gradient(&self, p: Vec2, footprint: Footprint) -> (f32, Vec2) {
        let weight = footprint.band_weight(self.cellular.lattice.max_frequency());
        if weight == 0.0 {
            return (self.mean, Vec2::ZERO);
        }
        let (sample, [df1, df2, dborder]) = self.cellular.sample_gradient::<true>(p);
        let value = select(self.output, &sample);
        let gradient = match self.output {
            CellOutput::F1 => df1,
            CellOutput::F2 => df2,
            CellOutput::F2MinusF1 => df2 - df1,
            CellOutput::Border => dborder,
            CellOutput::CellValue => Vec2::ZERO,
        } * self.cellular.lattice.frequency;
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

    #[test]
    fn distances_are_ordered_and_border_is_bounded() {
        let cells = Cellular::new(Domain::Plane, Vec2::splat(5.0), 1.0, 21).unwrap();
        for i in 0..2048 {
            let p = Vec2::new((i % 64) as f32 * 0.029, (i / 64) as f32 * 0.031);
            let s = cells.sample(p);
            assert!(s.f1 <= s.f2, "{s:?}");
            // Every bisector is at least (fk - f1) / 2 away and the one with
            // the second point at most (f1 + f2) / 2.
            let slack = 1e-5;
            assert!(s.border >= (s.f2 - s.f1) * 0.5 - slack, "{s:?}");
            assert!(s.border <= (s.f1 + s.f2) * 0.5 + slack, "{s:?}");
            assert!((0.0..1.0).contains(&s.value));
        }
    }

    #[test]
    fn zero_jitter_is_a_regular_grid() {
        let cells = Cellular::new(Domain::Plane, Vec2::ONE, 0.0, 1).unwrap();
        let s = cells.sample(Vec2::new(0.5, 0.5));
        assert_eq!(s.f1, 0.0);
        assert_eq!(s.f2, 1.0);
        assert_eq!(s.border, 0.5);
    }

    #[test]
    fn periodic_cells_share_ids_across_the_seam() {
        let domain = Domain::periodic(1, 1).unwrap();
        let cells = Cellular::new(domain, Vec2::splat(4.0), 1.0, 8).unwrap();
        let a = cells.sample(Vec2::new(0.0625, 0.3125));
        let b = cells.sample(Vec2::new(1.0625, 0.3125));
        assert_eq!(a, b);
    }

    #[test]
    fn rejects_out_of_range_jitter() {
        assert!(Cellular::new(Domain::Plane, Vec2::ONE, 1.5, 0).is_err());
    }

    #[test]
    fn coarse_footprints_fade_to_the_mean() {
        let cells = Cellular::new(Domain::Plane, Vec2::splat(16.0), 1.0, 3).unwrap();
        for output in [
            CellOutput::F1,
            CellOutput::F2,
            CellOutput::F2MinusF1,
            CellOutput::Border,
            CellOutput::CellValue,
        ] {
            let field = cells.output(output);
            let p = Vec2::new(0.37, 0.61);
            // Cells of 1/16 unit span four footprints of 1/64: full detail.
            let sharp = Footprint::new(1.0 / 64.0).unwrap();
            assert_eq!(
                field.eval(p, sharp).to_bits(),
                field.eval(p, Footprint::POINT).to_bits()
            );
            // Two footprints per cell or fewer: only the mean.
            let coarse = Footprint::new(1.0 / 32.0).unwrap();
            assert_eq!(field.eval(p, coarse), field.mean());
            // Between: toward the mean.
            let between = Footprint::new(1.0 / 48.0).unwrap();
            let (v, m) = (field.eval(p, Footprint::POINT), field.mean());
            let blended = field.eval(p, between);
            assert!((blended - m).abs() <= (v - m).abs() + 1e-6, "{output:?}");
        }
    }

    #[test]
    fn means_match_the_sampled_average() {
        let domain = Domain::periodic(1, 1).unwrap();
        for jitter in [0.0, 0.5, 1.0] {
            // 32 × 32 cells, so the per-cell values average out too.
            let cells = Cellular::new(domain, Vec2::splat(32.0), jitter, 11).unwrap();
            for output in [CellOutput::F1, CellOutput::Border, CellOutput::CellValue] {
                let field = cells.output(output);
                let mut sum = 0.0_f64;
                let n = 256;
                for j in 0..n {
                    for i in 0..n {
                        let p =
                            Vec2::new((i as f32 + 0.37) / n as f32, (j as f32 + 0.71) / n as f32);
                        sum += f64::from(field.eval(p, Footprint::POINT));
                    }
                }
                let measured = sum / f64::from(n * n);
                assert!(
                    (measured - f64::from(field.mean())).abs() < 0.03,
                    "jitter {jitter} {output:?}: {measured} vs {}",
                    field.mean()
                );
            }
        }
    }
}
