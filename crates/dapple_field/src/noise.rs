// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Lattice noise: value noise and gradient noise.

use glam::Vec2;

use crate::domain::{Domain, DomainError, Footprint, Lattice};
use crate::field::ScalarField;
use crate::hash::{hash, key, unit_f32};

/// Purpose tag for lattice-corner hashes.
const LATTICE_TAG: u64 = 0x006c_6174_7469_6365; // "lattice"

/// Lattice noise flavor, shared by single-octave noise and fractals.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Basis {
    /// Random values at lattice corners, blended with a quintic fade.
    Value,
    /// Random gradients at lattice corners (Perlin-style), blended with a
    /// quintic fade. Zero at every lattice corner.
    Gradient,
}

/// Sixteen unit gradients at multiples of 22.5°.
///
/// Literals rather than `sin`/`cos` calls, so every platform uses identical bits.
const GRADIENTS: [[f32; 2]; 16] = [
    [1.0, 0.0],
    [0.923_879_5, 0.382_683_43],
    [0.707_106_77, 0.707_106_77],
    [0.382_683_43, 0.923_879_5],
    [0.0, 1.0],
    [-0.382_683_43, 0.923_879_5],
    [-0.707_106_77, 0.707_106_77],
    [-0.923_879_5, 0.382_683_43],
    [-1.0, 0.0],
    [-0.923_879_5, -0.382_683_43],
    [-0.707_106_77, -0.707_106_77],
    [-0.382_683_43, -0.923_879_5],
    [0.0, -1.0],
    [0.382_683_43, -0.923_879_5],
    [0.707_106_77, -0.707_106_77],
    [0.923_879_5, -0.382_683_43],
];

/// Scales 2D gradient noise, whose extremes are ±√½, to about ±1.
const GRADIENT_SCALE: f32 = core::f32::consts::SQRT_2;

/// Quintic fade `6t⁵ − 15t⁴ + 10t³`: C² continuous at cell borders.
#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Evaluates one octave of `basis` noise at `p`, in `[-1, 1]`.
///
/// `offset` shifts the lattice by a fraction of a cell per axis, each
/// component in `[0, 1)`. It is added to the in-cell position after locating
/// the cell, so periodic repeats stay bit-identical.
pub(crate) fn lattice_noise(
    basis: Basis,
    lattice: Lattice,
    seed: u64,
    offset: Vec2,
    p: Vec2,
) -> f32 {
    let at = lattice.locate(p);
    let mut cell = at.cell;
    let mut f = at.frac + offset;
    // Both terms are below 1, so each sum is below 2 and subtracting 1 is exact.
    if f.x >= 1.0 {
        f.x -= 1.0;
        cell[0] += 1;
    }
    if f.y >= 1.0 {
        f.y -= 1.0;
        cell[1] += 1;
    }
    let corner = |dx: i64, dy: i64| -> u64 {
        let [x, y] = lattice.wrap_cell([cell[0] + dx, cell[1] + dy]);
        hash(seed, &[LATTICE_TAG, key(x), key(y)])
    };
    let [h00, h10, h01, h11] = [corner(0, 0), corner(1, 0), corner(0, 1), corner(1, 1)];
    let (u, v) = (fade(f.x), fade(f.y));
    match basis {
        Basis::Value => {
            let value = |h: u64| unit_f32(h) * 2.0 - 1.0;
            lerp(
                lerp(value(h00), value(h10), u),
                lerp(value(h01), value(h11), u),
                v,
            )
        }
        Basis::Gradient => {
            let dot = |h: u64, dx: f32, dy: f32| {
                // The top four bits pick the gradient.
                let [gx, gy] = GRADIENTS[(h >> 60) as usize];
                gx * dx + gy * dy
            };
            let n00 = dot(h00, f.x, f.y);
            let n10 = dot(h10, f.x - 1.0, f.y);
            let n01 = dot(h01, f.x, f.y - 1.0);
            let n11 = dot(h11, f.x - 1.0, f.y - 1.0);
            lerp(lerp(n00, n10, u), lerp(n01, n11, u), v) * GRADIENT_SCALE
        }
    }
}

/// One octave of lattice noise in about `[-1, 1]`, with mean zero.
///
/// `frequency` is lattice cells per domain unit on each axis; unequal axes
/// stretch the noise. On a periodic domain each axis must fit a whole number
/// of cells into the period.
///
/// ```
/// use dapple_field::{Basis, Domain, Footprint, Noise, ScalarField};
/// use glam::Vec2;
///
/// let domain = Domain::periodic(1, 1).unwrap();
/// let noise = Noise::new(Basis::Gradient, domain, Vec2::splat(8.0), 7)?;
/// let a = noise.eval(Vec2::new(0.0, 0.25), Footprint::POINT);
/// let b = noise.eval(Vec2::new(1.0, 0.25), Footprint::POINT);
/// assert_eq!(a.to_bits(), b.to_bits(), "the period wraps exactly");
/// # Ok::<(), dapple_field::DomainError>(())
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Noise {
    basis: Basis,
    domain: Domain,
    lattice: Lattice,
    seed: u64,
}

impl Noise {
    /// Builds noise of `basis` at `frequency` cells per unit.
    pub fn new(
        basis: Basis,
        domain: Domain,
        frequency: Vec2,
        seed: u64,
    ) -> Result<Self, DomainError> {
        Ok(Self {
            basis,
            domain,
            lattice: Lattice::new(domain, frequency)?,
            seed,
        })
    }

    /// The noise flavor.
    #[must_use]
    pub const fn basis(&self) -> Basis {
        self.basis
    }
}

impl ScalarField for Noise {
    fn domain(&self) -> Domain {
        self.domain
    }

    fn eval(&self, p: Vec2, footprint: Footprint) -> f32 {
        let weight = footprint.band_weight(self.lattice.max_frequency());
        if weight == 0.0 {
            return 0.0;
        }
        lattice_noise(self.basis, self.lattice, self.seed, Vec2::ZERO, p) * weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradients_are_unit_length() {
        for [x, y] in GRADIENTS {
            assert!((x * x + y * y - 1.0).abs() < 1e-6, "({x}, {y})");
        }
    }

    #[test]
    fn gradient_noise_vanishes_on_lattice_corners() {
        let noise = Noise::new(Basis::Gradient, Domain::Plane, Vec2::splat(4.0), 3).unwrap();
        for i in -4..4 {
            let p = Vec2::new(i as f32 * 0.25, 0.5);
            assert_eq!(noise.eval(p, Footprint::POINT), 0.0);
        }
    }

    #[test]
    fn noise_stays_in_range() {
        for basis in [Basis::Value, Basis::Gradient] {
            let noise = Noise::new(basis, Domain::Plane, Vec2::splat(3.0), 11).unwrap();
            for i in 0..4096 {
                let p = Vec2::new((i % 64) as f32 * 0.037, (i / 64) as f32 * 0.041);
                let v = noise.eval(p, Footprint::POINT);
                assert!((-1.0..=1.0).contains(&v), "{basis:?} value {v} at {p}");
            }
        }
    }

    #[test]
    fn footprints_fade_noise_to_its_mean() {
        let noise = Noise::new(Basis::Value, Domain::Plane, Vec2::splat(8.0), 5).unwrap();
        let coarse = Footprint::new(1.0 / 16.0).unwrap();
        assert_eq!(noise.eval(Vec2::new(0.3, 0.1), coarse), 0.0);
        let fine = Footprint::new(1.0 / 64.0).unwrap();
        assert_eq!(
            noise.eval(Vec2::new(0.3, 0.1), fine),
            noise.eval(Vec2::new(0.3, 0.1), Footprint::POINT)
        );
    }
}
