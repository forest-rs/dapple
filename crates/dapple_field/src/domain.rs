// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Domains, footprints, lattices, and domain errors.

use core::fmt;

use glam::Vec2;

/// Where a field is defined.
///
/// A domain is part of a field's type contract. `Periodic` tiling is
/// guaranteed by construction: fields refuse lattices and transforms that would
/// break it, rather than blending seams afterwards.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Domain {
    /// The unbounded, non-repeating plane.
    Plane,
    /// A torus: the field repeats every `period` domain units on each axis.
    Periodic {
        /// Repeat length per axis, in domain units. Each component is at least 1.
        period: [u32; 2],
    },
}

impl Domain {
    /// A periodic domain, or `None` when a period component is zero.
    #[must_use]
    pub const fn periodic(x: u32, y: u32) -> Option<Self> {
        if x == 0 || y == 0 {
            None
        } else {
            Some(Self::Periodic { period: [x, y] })
        }
    }

    /// The repeat length, when periodic.
    #[must_use]
    pub const fn period(self) -> Option<[u32; 2]> {
        match self {
            Self::Plane => None,
            Self::Periodic { period } => Some(period),
        }
    }
}

/// The size of the region a field value stands for.
///
/// A point sample uses [`Footprint::POINT`]. Realizing a texel, a mip level, or
/// a screen pixel uses that region's side length in domain units. Fields use
/// the footprint to band-limit themselves: detail above the region's Nyquist
/// limit fades to its mean instead of aliasing.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Footprint {
    width: f32,
}

impl Footprint {
    /// An infinitely small region: no band limiting.
    pub const POINT: Self = Self { width: 0.0 };

    /// A square region `width` domain units across.
    ///
    /// Returns `None` for negative or non-finite widths.
    #[must_use]
    pub fn new(width: f32) -> Option<Self> {
        (width.is_finite() && width >= 0.0).then_some(Self { width })
    }

    /// The region's side length in domain units.
    #[must_use]
    pub const fn width(self) -> f32 {
        self.width
    }

    /// The footprint after the domain is stretched by `factor`.
    #[must_use]
    pub(crate) fn scaled(self, factor: f32) -> Self {
        Self {
            width: self.width * factor,
        }
    }

    /// How much of a component with `frequency` cycles per domain unit
    /// survives this footprint.
    ///
    /// Components at or below a quarter of the sampling rate
    /// (`frequency * width <= 0.25`) keep full weight. Components at or above
    /// the Nyquist limit (`>= 0.5`) are removed. Between them the weight falls
    /// along a smoothstep.
    #[must_use]
    pub fn band_weight(self, frequency: f32) -> f32 {
        let t = frequency * self.width;
        if t <= 0.25 {
            1.0
        } else if t >= 0.5 {
            0.0
        } else {
            let s = (0.5 - t) * 4.0;
            s * s * (3.0 - 2.0 * s)
        }
    }
}

/// A rejected field or transform construction.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DomainError {
    /// A frequency is non-finite or not positive.
    InvalidFrequency {
        /// The offending frequency, per axis.
        frequency: [f32; 2],
    },
    /// On a periodic domain, `frequency * period` must be a whole number of
    /// lattice cells, or the field would not tile.
    NonIntegerLattice {
        /// Axis index, 0 for x and 1 for y.
        axis: usize,
        /// The requested cell count across one period.
        cells: f32,
    },
    /// The lattice would exceed [`MAX_LATTICE_CELLS`] cells across one period.
    LatticeTooFine {
        /// Axis index, 0 for x and 1 for y.
        axis: usize,
        /// The requested cell count across one period.
        cells: u64,
    },
    /// The transform does not map the periodic lattice onto itself, so the
    /// result would not tile. Demote the field to [`Domain::Plane`] first if
    /// a non-repeating result is intended.
    NotLatticePreserving,
    /// A parameter is outside its documented range.
    InvalidParameter {
        /// Parameter name.
        name: &'static str,
    },
}

impl fmt::Display for DomainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFrequency { frequency } => {
                write!(f, "frequency {frequency:?} must be finite and positive")
            }
            Self::NonIntegerLattice { axis, cells } => write!(
                f,
                "axis {axis}: {cells} lattice cells per period does not tile"
            ),
            Self::LatticeTooFine { axis, cells } => {
                write!(f, "axis {axis}: {cells} lattice cells per period is too many")
            }
            Self::NotLatticePreserving => {
                f.write_str("transform does not preserve the periodic lattice")
            }
            Self::InvalidParameter { name } => write!(f, "parameter {name} is out of range"),
        }
    }
}

impl core::error::Error for DomainError {}

/// Largest cell count across one period, per axis.
///
/// Lattice coordinates stay exact in `f32` below 2^24.
pub const MAX_LATTICE_CELLS: u64 = 1 << 24;

/// A scaled integer lattice, wrapped on periodic domains.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct Lattice {
    pub(crate) frequency: Vec2,
    /// Cells per period, per axis, on periodic domains.
    pub(crate) wrap: Option<[i64; 2]>,
}

/// One lattice lookup: the containing cell and the position inside it.
#[derive(Copy, Clone, Debug)]
pub(crate) struct CellPoint {
    pub(crate) cell: [i64; 2],
    pub(crate) frac: Vec2,
}

impl Lattice {
    pub(crate) fn new(domain: Domain, frequency: Vec2) -> Result<Self, DomainError> {
        if !(frequency.is_finite() && frequency.x > 0.0 && frequency.y > 0.0) {
            return Err(DomainError::InvalidFrequency {
                frequency: frequency.to_array(),
            });
        }
        let wrap = match domain {
            Domain::Plane => None,
            Domain::Periodic { period } => Some([
                cells_per_period(0, frequency.x, period[0])?,
                cells_per_period(1, frequency.y, period[1])?,
            ]),
        };
        Ok(Self { frequency, wrap })
    }

    /// The lattice `factor` times finer, for fractal octaves.
    pub(crate) fn refined(self, factor: u32) -> Result<Self, DomainError> {
        let frequency = self.frequency * factor as f32;
        let wrap = match self.wrap {
            None => None,
            Some(cells) => {
                let mut out = [0; 2];
                for (axis, (out, cells)) in out.iter_mut().zip(cells).enumerate() {
                    let refined = cells.unsigned_abs() * u64::from(factor);
                    if refined > MAX_LATTICE_CELLS {
                        return Err(DomainError::LatticeTooFine {
                            axis,
                            cells: refined,
                        });
                    }
                    *out = refined.cast_signed();
                }
                Some(out)
            }
        };
        if !frequency.is_finite() {
            return Err(DomainError::InvalidFrequency {
                frequency: frequency.to_array(),
            });
        }
        Ok(Self { frequency, wrap })
    }

    /// Highest lattice frequency across both axes.
    pub(crate) fn max_frequency(self) -> f32 {
        self.frequency.max_element()
    }

    pub(crate) fn locate(self, p: Vec2) -> CellPoint {
        let q = p * self.frequency;
        let floor = Vec2::new(libm::floorf(q.x), libm::floorf(q.y));
        CellPoint {
            cell: [to_cell(floor.x), to_cell(floor.y)],
            frac: q - floor,
        }
    }

    /// Wraps a cell index onto the period, for hashing.
    pub(crate) fn wrap_cell(self, cell: [i64; 2]) -> [i64; 2] {
        match self.wrap {
            None => cell,
            Some(n) => [cell[0].rem_euclid(n[0]), cell[1].rem_euclid(n[1])],
        }
    }
}

fn cells_per_period(axis: usize, frequency: f32, period: u32) -> Result<i64, DomainError> {
    let cells = f64::from(frequency) * f64::from(period);
    if libm::trunc(cells) != cells || cells < 1.0 {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "reported for diagnostics only"
        )]
        return Err(DomainError::NonIntegerLattice {
            axis,
            cells: cells as f32,
        });
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "cells is a positive whole number; larger values are rejected below"
    )]
    let whole = cells.min(u64::MAX as f64) as u64;
    if whole > MAX_LATTICE_CELLS {
        return Err(DomainError::LatticeTooFine { axis, cells: whole });
    }
    Ok(whole.cast_signed())
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "a floored coordinate saturates at the i64 range, far beyond usable precision"
)]
fn to_cell(floor: f32) -> i64 {
    floor as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodic_lattices_must_be_whole() {
        let domain = Domain::periodic(2, 1).unwrap();
        assert!(Lattice::new(domain, Vec2::new(1.5, 4.0)).is_ok());
        assert_eq!(
            Lattice::new(domain, Vec2::new(1.25, 4.0)),
            Err(DomainError::NonIntegerLattice {
                axis: 0,
                cells: 2.5
            })
        );
        assert!(matches!(
            Lattice::new(domain, Vec2::new(0.0, 1.0)),
            Err(DomainError::InvalidFrequency { .. })
        ));
        assert!(Lattice::new(Domain::Plane, Vec2::new(1.25, 0.3)).is_ok());
    }

    #[test]
    fn refinement_stays_bounded() {
        let domain = Domain::periodic(1, 1).unwrap();
        let lattice = Lattice::new(domain, Vec2::splat(8_388_608.0)).unwrap();
        assert!(lattice.refined(2).is_ok());
        assert_eq!(
            lattice.refined(4),
            Err(DomainError::LatticeTooFine {
                axis: 0,
                cells: 1 << 25
            })
        );
    }

    #[test]
    fn band_weight_is_monotone_and_bounded() {
        let footprint = Footprint::new(1.0).unwrap();
        assert_eq!(footprint.band_weight(0.25), 1.0);
        assert_eq!(footprint.band_weight(0.5), 0.0);
        let mid = footprint.band_weight(0.375);
        assert!(mid > 0.0 && mid < 1.0, "mid-band weight {mid}");
        assert_eq!(Footprint::POINT.band_weight(1e9), 1.0);
        assert_eq!(Footprint::new(-1.0), None);
    }
}
