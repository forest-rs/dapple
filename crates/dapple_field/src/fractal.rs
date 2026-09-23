// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Fractal sums of lattice noise: fBm and ridged.

use alloc::vec::Vec;

use glam::Vec2;

use crate::domain::{Domain, DomainError, Footprint, Lattice};
use crate::field::ScalarField;
use crate::hash::{hash, unit_f32};
use crate::noise::{Basis, lattice_noise};

/// Purpose tag for per-octave seeds.
const OCTAVE_TAG: u64 = 0x6f63_7461_7665; // "octave"
/// Purpose tag for per-octave lattice offsets.
const OFFSET_TAG: u64 = 0x6f66_6673_6574; // "offset"

/// Maximum octave count.
pub const MAX_OCTAVES: u8 = 24;

/// How octaves combine.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum FractalKind {
    /// Fractional Brownian motion: the amplitude-weighted sum of octaves, in
    /// about `[-1, 1]` with mean zero.
    Fbm,
    /// Ridged: each octave is `(1 − |n|)²`, sharp crests where the noise
    /// crosses zero. The sum is in `[0, 1]`.
    Ridged,
}

/// Octave structure of a [`Fractal`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FractalParams {
    /// How octaves combine.
    pub kind: FractalKind,
    /// Number of octaves, 1 to [`MAX_OCTAVES`].
    pub octaves: u8,
    /// Frequency multiplier between octaves, at least 2. An integer, so every
    /// octave of a periodic fractal still tiles.
    pub lacunarity: u32,
    /// Amplitude multiplier between octaves, in `(0, 1]`.
    pub gain: f32,
}

impl Default for FractalParams {
    fn default() -> Self {
        Self {
            kind: FractalKind::Fbm,
            octaves: 5,
            lacunarity: 2,
            gain: 0.5,
        }
    }
}

/// Mean of one ridged octave, `E[(1 − |n|)²]`, per basis.
///
/// Octaves removed by the footprint are replaced by this mean, so band
/// limiting does not shift the average. The values were measured over 1.28M
/// samples (8 seeds) and are checked by a test.
const fn ridged_mean(basis: Basis) -> f32 {
    match basis {
        Basis::Value => 0.447,
        Basis::Gradient => 0.595,
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct Octave {
    lattice: Lattice,
    seed: u64,
    /// Sub-cell lattice shift, so octave lattices do not share lines.
    offset: Vec2,
    amplitude: f32,
}

/// A fractal sum of lattice-noise octaves, band-limited by footprint.
///
/// Octave `i` has `lacunarity^i` times the base frequency, `gain^i` of the base
/// amplitude, its own seed `hash(seed, [OCTAVE_TAG, i])`, and a keyed
/// sub-cell lattice offset. Integer lacunarity nests the octave lattices, and
/// without offsets their shared lattice lines show as straight seams, most
/// visibly in ridged sums. The offsets shift the lattice, not the domain, so
/// periodic fractals still tile bit-exactly. The sum is
/// normalized by the total amplitude of all octaves. Octaves above the
/// footprint's Nyquist limit fade to their mean (see
/// [`Footprint::band_weight`]), so a coarse footprint yields a smooth,
/// unshifted average instead of aliasing.
#[derive(Clone, Debug, PartialEq)]
pub struct Fractal {
    basis: Basis,
    kind: FractalKind,
    domain: Domain,
    octaves: Vec<Octave>,
    total_amplitude: f32,
}

impl Fractal {
    /// Builds a fractal whose first octave has `frequency` cells per unit.
    pub fn new(
        basis: Basis,
        domain: Domain,
        frequency: Vec2,
        seed: u64,
        params: FractalParams,
    ) -> Result<Self, DomainError> {
        if params.octaves == 0 || params.octaves > MAX_OCTAVES {
            return Err(DomainError::InvalidParameter { name: "octaves" });
        }
        if params.lacunarity < 2 {
            return Err(DomainError::InvalidParameter { name: "lacunarity" });
        }
        if !(params.gain > 0.0 && params.gain <= 1.0) {
            return Err(DomainError::InvalidParameter { name: "gain" });
        }
        let mut lattice = Lattice::new(domain, frequency)?;
        let mut amplitude = 1.0;
        let mut octaves = Vec::with_capacity(usize::from(params.octaves));
        for i in 0..params.octaves {
            if i > 0 {
                lattice = lattice.refined(params.lacunarity)?;
                amplitude *= params.gain;
            }
            let octave_seed = hash(seed, &[OCTAVE_TAG, u64::from(i)]);
            octaves.push(Octave {
                lattice,
                seed: octave_seed,
                offset: Vec2::new(
                    unit_f32(hash(octave_seed, &[OFFSET_TAG, 0])),
                    unit_f32(hash(octave_seed, &[OFFSET_TAG, 1])),
                ),
                amplitude,
            });
        }
        let total_amplitude = octaves.iter().map(|o| o.amplitude).sum();
        Ok(Self {
            basis,
            kind: params.kind,
            domain,
            octaves,
            total_amplitude,
        })
    }

    /// The number of octaves that contribute detail at `footprint`.
    #[must_use]
    pub fn active_octaves(&self, footprint: Footprint) -> usize {
        self.octaves
            .iter()
            .take_while(|o| footprint.band_weight(o.lattice.max_frequency()) > 0.0)
            .count()
    }
}

impl ScalarField for Fractal {
    fn domain(&self) -> Domain {
        self.domain
    }

    fn eval(&self, p: Vec2, footprint: Footprint) -> f32 {
        let mean = match self.kind {
            FractalKind::Fbm => 0.0,
            FractalKind::Ridged => ridged_mean(self.basis),
        };
        let mut sum = 0.0;
        for octave in &self.octaves {
            let weight = footprint.band_weight(octave.lattice.max_frequency());
            let value = if weight == 0.0 {
                mean
            } else {
                let n = lattice_noise(self.basis, octave.lattice, octave.seed, octave.offset, p);
                let detail = match self.kind {
                    FractalKind::Fbm => n,
                    FractalKind::Ridged => {
                        let r = 1.0 - n.abs();
                        r * r
                    }
                };
                mean + (detail - mean) * weight
            };
            sum += value * octave.amplitude;
        }
        sum / self.total_amplitude
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_mean(field: &impl ScalarField, footprint: Footprint) -> f32 {
        let mut sum = 0.0_f64;
        let n = 256;
        for y in 0..n {
            for x in 0..n {
                let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5) / n as f32 * 16.0;
                sum += f64::from(field.eval(p, footprint));
            }
        }
        #[expect(clippy::cast_possible_truncation, reason = "test statistic")]
        let mean = (sum / f64::from(n * n)) as f32;
        mean
    }

    #[test]
    fn ridged_means_match_their_constants() {
        for basis in [Basis::Value, Basis::Gradient] {
            let one = FractalParams {
                kind: FractalKind::Ridged,
                octaves: 1,
                ..FractalParams::default()
            };
            let field = Fractal::new(basis, Domain::Plane, Vec2::splat(3.7), 9, one).unwrap();
            let measured = sample_mean(&field, Footprint::POINT);
            assert!(
                (measured - ridged_mean(basis)).abs() < 0.02,
                "{basis:?}: measured {measured}, constant {}",
                ridged_mean(basis)
            );
        }
    }

    #[test]
    fn coarse_footprints_keep_the_mean() {
        let params = FractalParams {
            kind: FractalKind::Ridged,
            octaves: 6,
            ..FractalParams::default()
        };
        let field =
            Fractal::new(Basis::Gradient, Domain::Plane, Vec2::splat(2.0), 4, params).unwrap();
        let sharp = sample_mean(&field, Footprint::POINT);
        let soft = sample_mean(&field, Footprint::new(1.0 / 16.0).unwrap());
        assert!((sharp - soft).abs() < 0.02, "means {sharp} vs {soft}");
        assert_eq!(field.active_octaves(Footprint::POINT), 6);
        assert_eq!(field.active_octaves(Footprint::new(1.0 / 16.0).unwrap()), 2);
    }

    #[test]
    fn rejects_bad_parameters() {
        let bad = |params| Fractal::new(Basis::Value, Domain::Plane, Vec2::ONE, 0, params);
        let base = FractalParams::default();
        assert!(bad(FractalParams { octaves: 0, ..base }).is_err());
        assert!(bad(FractalParams {
            lacunarity: 1,
            ..base
        })
        .is_err());
        assert!(bad(FractalParams { gain: 0.0, ..base }).is_err());
        assert!(bad(base).is_ok());
    }
}
