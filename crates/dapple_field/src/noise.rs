// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Lattice noise: value noise and gradient noise.

use glam::Vec2;

use crate::domain::{Domain, DomainError, Footprint, Lattice};
use crate::field::ScalarField;
use crate::hash::{key, mix, unit_f32};

/// Purpose tag for lattice-corner hashes.
const LATTICE_TAG: u64 = 0x006c_6174_7469_6365; // "lattice"

/// Lattice noise flavor, shared by single-octave noise and fractals.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Basis {
    /// Random values at lattice corners, blended with a quintic fade.
    Value,
    /// Random gradients plus a smaller random value at each lattice corner,
    /// each weighted by a radial kernel of 1.5 cells, `(1 − r²/2.25)⁴`.
    ///
    /// Unlike Perlin's separable blend, the radial kernels give no axis
    /// special treatment, and the value term fills the dip every gradient
    /// term has at its own corner. Together they remove the grid lines
    /// classic gradient noise shows along its lattice (mean |n| on lattice
    /// lines is 0.99 of elsewhere, from 0.72). The lattice stays square, so
    /// periodic domains still tile exactly on integer periods. Values stay
    /// within `[-1, 1]`, with RMS about 0.22.
    Gradient,
}

/// 256 unit gradients at angles `(k + ½) · 2π / 256`.
///
/// The half-step offset keeps every gradient off the axes and diagonals, so
/// no gradient lines up with the lattice. Literals rather than `sin`/`cos`
/// calls, so every platform uses identical bits.
#[rustfmt::skip]
const GRADIENTS: [[f32; 2]; 256] = [
    [0.9999247, 0.012271538],
    [0.99932235, 0.036807224],
    [0.9981181, 0.061320737],
    [0.9963126, 0.08579731],
    [0.993907, 0.110222206],
    [0.99090266, 0.1345807],
    [0.9873014, 0.15885815],
    [0.9831055, 0.18303989],
    [0.9783174, 0.20711137],
    [0.97293997, 0.2310581],
    [0.96697646, 0.25486565],
    [0.9604305, 0.2785197],
    [0.953306, 0.30200595],
    [0.9456073, 0.3253103],
    [0.937339, 0.34841868],
    [0.9285061, 0.3713172],
    [0.9191139, 0.39399204],
    [0.909168, 0.41642955],
    [0.8986745, 0.43861625],
    [0.88763964, 0.46053872],
    [0.8760701, 0.48218378],
    [0.86397284, 0.50353837],
    [0.8513552, 0.52458966],
    [0.8382247, 0.545325],
    [0.8245893, 0.5657318],
    [0.81045717, 0.58579785],
    [0.7958369, 0.60551107],
    [0.7807372, 0.6248595],
    [0.76516724, 0.64383155],
    [0.7491364, 0.6624158],
    [0.7326543, 0.680601],
    [0.71573085, 0.69837624],
    [0.69837624, 0.71573085],
    [0.680601, 0.7326543],
    [0.6624158, 0.7491364],
    [0.64383155, 0.76516724],
    [0.6248595, 0.7807372],
    [0.60551107, 0.7958369],
    [0.58579785, 0.81045717],
    [0.5657318, 0.8245893],
    [0.545325, 0.8382247],
    [0.52458966, 0.8513552],
    [0.50353837, 0.86397284],
    [0.48218378, 0.8760701],
    [0.46053872, 0.88763964],
    [0.43861625, 0.8986745],
    [0.41642955, 0.909168],
    [0.39399204, 0.9191139],
    [0.3713172, 0.9285061],
    [0.34841868, 0.937339],
    [0.3253103, 0.9456073],
    [0.30200595, 0.953306],
    [0.2785197, 0.9604305],
    [0.25486565, 0.96697646],
    [0.2310581, 0.97293997],
    [0.20711137, 0.9783174],
    [0.18303989, 0.9831055],
    [0.15885815, 0.9873014],
    [0.1345807, 0.99090266],
    [0.110222206, 0.993907],
    [0.08579731, 0.9963126],
    [0.061320737, 0.9981181],
    [0.036807224, 0.99932235],
    [0.012271538, 0.9999247],
    [-0.012271538, 0.9999247],
    [-0.036807224, 0.99932235],
    [-0.061320737, 0.9981181],
    [-0.08579731, 0.9963126],
    [-0.110222206, 0.993907],
    [-0.1345807, 0.99090266],
    [-0.15885815, 0.9873014],
    [-0.18303989, 0.9831055],
    [-0.20711137, 0.9783174],
    [-0.2310581, 0.97293997],
    [-0.25486565, 0.96697646],
    [-0.2785197, 0.9604305],
    [-0.30200595, 0.953306],
    [-0.3253103, 0.9456073],
    [-0.34841868, 0.937339],
    [-0.3713172, 0.9285061],
    [-0.39399204, 0.9191139],
    [-0.41642955, 0.909168],
    [-0.43861625, 0.8986745],
    [-0.46053872, 0.88763964],
    [-0.48218378, 0.8760701],
    [-0.50353837, 0.86397284],
    [-0.52458966, 0.8513552],
    [-0.545325, 0.8382247],
    [-0.5657318, 0.8245893],
    [-0.58579785, 0.81045717],
    [-0.60551107, 0.7958369],
    [-0.6248595, 0.7807372],
    [-0.64383155, 0.76516724],
    [-0.6624158, 0.7491364],
    [-0.680601, 0.7326543],
    [-0.69837624, 0.71573085],
    [-0.71573085, 0.69837624],
    [-0.7326543, 0.680601],
    [-0.7491364, 0.6624158],
    [-0.76516724, 0.64383155],
    [-0.7807372, 0.6248595],
    [-0.7958369, 0.60551107],
    [-0.81045717, 0.58579785],
    [-0.8245893, 0.5657318],
    [-0.8382247, 0.545325],
    [-0.8513552, 0.52458966],
    [-0.86397284, 0.50353837],
    [-0.8760701, 0.48218378],
    [-0.88763964, 0.46053872],
    [-0.8986745, 0.43861625],
    [-0.909168, 0.41642955],
    [-0.9191139, 0.39399204],
    [-0.9285061, 0.3713172],
    [-0.937339, 0.34841868],
    [-0.9456073, 0.3253103],
    [-0.953306, 0.30200595],
    [-0.9604305, 0.2785197],
    [-0.96697646, 0.25486565],
    [-0.97293997, 0.2310581],
    [-0.9783174, 0.20711137],
    [-0.9831055, 0.18303989],
    [-0.9873014, 0.15885815],
    [-0.99090266, 0.1345807],
    [-0.993907, 0.110222206],
    [-0.9963126, 0.08579731],
    [-0.9981181, 0.061320737],
    [-0.99932235, 0.036807224],
    [-0.9999247, 0.012271538],
    [-0.9999247, -0.012271538],
    [-0.99932235, -0.036807224],
    [-0.9981181, -0.061320737],
    [-0.9963126, -0.08579731],
    [-0.993907, -0.110222206],
    [-0.99090266, -0.1345807],
    [-0.9873014, -0.15885815],
    [-0.9831055, -0.18303989],
    [-0.9783174, -0.20711137],
    [-0.97293997, -0.2310581],
    [-0.96697646, -0.25486565],
    [-0.9604305, -0.2785197],
    [-0.953306, -0.30200595],
    [-0.9456073, -0.3253103],
    [-0.937339, -0.34841868],
    [-0.9285061, -0.3713172],
    [-0.9191139, -0.39399204],
    [-0.909168, -0.41642955],
    [-0.8986745, -0.43861625],
    [-0.88763964, -0.46053872],
    [-0.8760701, -0.48218378],
    [-0.86397284, -0.50353837],
    [-0.8513552, -0.52458966],
    [-0.8382247, -0.545325],
    [-0.8245893, -0.5657318],
    [-0.81045717, -0.58579785],
    [-0.7958369, -0.60551107],
    [-0.7807372, -0.6248595],
    [-0.76516724, -0.64383155],
    [-0.7491364, -0.6624158],
    [-0.7326543, -0.680601],
    [-0.71573085, -0.69837624],
    [-0.69837624, -0.71573085],
    [-0.680601, -0.7326543],
    [-0.6624158, -0.7491364],
    [-0.64383155, -0.76516724],
    [-0.6248595, -0.7807372],
    [-0.60551107, -0.7958369],
    [-0.58579785, -0.81045717],
    [-0.5657318, -0.8245893],
    [-0.545325, -0.8382247],
    [-0.52458966, -0.8513552],
    [-0.50353837, -0.86397284],
    [-0.48218378, -0.8760701],
    [-0.46053872, -0.88763964],
    [-0.43861625, -0.8986745],
    [-0.41642955, -0.909168],
    [-0.39399204, -0.9191139],
    [-0.3713172, -0.9285061],
    [-0.34841868, -0.937339],
    [-0.3253103, -0.9456073],
    [-0.30200595, -0.953306],
    [-0.2785197, -0.9604305],
    [-0.25486565, -0.96697646],
    [-0.2310581, -0.97293997],
    [-0.20711137, -0.9783174],
    [-0.18303989, -0.9831055],
    [-0.15885815, -0.9873014],
    [-0.1345807, -0.99090266],
    [-0.110222206, -0.993907],
    [-0.08579731, -0.9963126],
    [-0.061320737, -0.9981181],
    [-0.036807224, -0.99932235],
    [-0.012271538, -0.9999247],
    [0.012271538, -0.9999247],
    [0.036807224, -0.99932235],
    [0.061320737, -0.9981181],
    [0.08579731, -0.9963126],
    [0.110222206, -0.993907],
    [0.1345807, -0.99090266],
    [0.15885815, -0.9873014],
    [0.18303989, -0.9831055],
    [0.20711137, -0.9783174],
    [0.2310581, -0.97293997],
    [0.25486565, -0.96697646],
    [0.2785197, -0.9604305],
    [0.30200595, -0.953306],
    [0.3253103, -0.9456073],
    [0.34841868, -0.937339],
    [0.3713172, -0.9285061],
    [0.39399204, -0.9191139],
    [0.41642955, -0.909168],
    [0.43861625, -0.8986745],
    [0.46053872, -0.88763964],
    [0.48218378, -0.8760701],
    [0.50353837, -0.86397284],
    [0.52458966, -0.8513552],
    [0.545325, -0.8382247],
    [0.5657318, -0.8245893],
    [0.58579785, -0.81045717],
    [0.60551107, -0.7958369],
    [0.6248595, -0.7807372],
    [0.64383155, -0.76516724],
    [0.6624158, -0.7491364],
    [0.680601, -0.7326543],
    [0.69837624, -0.71573085],
    [0.71573085, -0.69837624],
    [0.7326543, -0.680601],
    [0.7491364, -0.6624158],
    [0.76516724, -0.64383155],
    [0.7807372, -0.6248595],
    [0.7958369, -0.60551107],
    [0.81045717, -0.58579785],
    [0.8245893, -0.5657318],
    [0.8382247, -0.545325],
    [0.8513552, -0.52458966],
    [0.86397284, -0.50353837],
    [0.8760701, -0.48218378],
    [0.88763964, -0.46053872],
    [0.8986745, -0.43861625],
    [0.909168, -0.41642955],
    [0.9191139, -0.39399204],
    [0.9285061, -0.3713172],
    [0.937339, -0.34841868],
    [0.9456073, -0.3253103],
    [0.953306, -0.30200595],
    [0.9604305, -0.2785197],
    [0.96697646, -0.25486565],
    [0.97293997, -0.2310581],
    [0.9783174, -0.20711137],
    [0.9831055, -0.18303989],
    [0.9873014, -0.15885815],
    [0.99090266, -0.1345807],
    [0.993907, -0.110222206],
    [0.9963126, -0.08579731],
    [0.9981181, -0.061320737],
    [0.99932235, -0.036807224],
    [0.9999247, -0.012271538],
];

/// Squared radius of each corner's kernel, in cells. At 1.5 cells, a point
/// sees corners on both sides of the lattice line it is nearest.
const KERNEL_RADIUS_SQ: f32 = 2.25;

/// Weight of each corner's random value relative to its gradient term.
///
/// Every gradient term vanishes at its own corner, which alone makes the
/// noise quieter along lattice lines (mean |n| 0.72 of elsewhere). This much
/// value term evens it out (0.99).
const VALUE_WEIGHT: f32 = 0.7;

/// Scales the sum into `[-1, 1]`: the largest possible `Σ K(r)·(r + 0.7)`
/// over the 4 × 4 corners around any point is below 2.07 (measured on a
/// 400 × 400 grid of the cell, which the kernel's smoothness bounds).
const GRADIENT_SCALE: f32 = 1.0 / 2.07;

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
    lattice_eval::<false>(basis, lattice, seed, offset, p).0
}

/// [`lattice_noise`] and its gradient with respect to `p`.
///
/// The value is computed exactly as [`lattice_noise`] computes it.
pub(crate) fn lattice_noise_gradient(
    basis: Basis,
    lattice: Lattice,
    seed: u64,
    offset: Vec2,
    p: Vec2,
) -> (f32, Vec2) {
    lattice_eval::<true>(basis, lattice, seed, offset, p)
}

/// Derivative of [`fade`]: `30t⁴ − 60t³ + 30t²`.
#[inline]
fn fade_derivative(t: f32) -> f32 {
    let s = t * (1.0 - t);
    30.0 * s * s
}

/// One octave, with its gradient in domain units when `GRADIENT`. The value's
/// arithmetic does not depend on `GRADIENT`.
fn lattice_eval<const GRADIENT: bool>(
    basis: Basis,
    lattice: Lattice,
    seed: u64,
    offset: Vec2,
    p: Vec2,
) -> (f32, Vec2) {
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
    // A corner's hash is `hash(seed, [LATTICE_TAG, x, y])` of its wrapped
    // cell. The hash folds its keys from the left, so every corner shares
    // the seed-and-tag prefix and every corner of a column its `x` prefix:
    // mixing those once per octave and per column leaves one mix per corner
    // instead of three, for the same bits, and wraps each row and column
    // once rather than at every corner.
    let prefix = mix(seed, LATTICE_TAG);
    let column = |dx: i64| mix(prefix, key(lattice.wrap_axis(0, cell[0] + dx)));
    let row = |dy: i64| key(lattice.wrap_axis(1, cell[1] + dy));
    // Gradients below are per cell; the in-cell position moves `frequency`
    // cells per domain unit.
    match basis {
        Basis::Value => {
            let (x0, x1, y0, y1) = (column(0), column(1), row(0), row(1));
            let [h00, h10, h01, h11] = [mix(x0, y0), mix(x1, y0), mix(x0, y1), mix(x1, y1)];
            let (u, v) = (fade(f.x), fade(f.y));
            let value = |h: u64| unit_f32(h) * 2.0 - 1.0;
            let (a, b, c, d) = (value(h00), value(h10), value(h01), value(h11));
            let bottom = lerp(a, b, u);
            let top = lerp(c, d, u);
            let result = lerp(bottom, top, v);
            if !GRADIENT {
                return (result, Vec2::ZERO);
            }
            let dfx = ((b - a) + ((d - c) - (b - a)) * v) * fade_derivative(f.x);
            let dfy = (top - bottom) * fade_derivative(f.y);
            (result, Vec2::new(dfx, dfy) * lattice.frequency)
        }
        Basis::Gradient => {
            let mut sum = 0.0;
            let mut gradient = Vec2::ZERO;
            // Written out rather than `array::map`, which is not always inlined.
            let columns = [column(-1), column(0), column(1), column(2)];
            let rows = [row(-1), row(0), row(1), row(2)];
            for (dy, row) in (-1..=2_i64).zip(rows) {
                for (dx, column) in (-1..=2_i64).zip(columns) {
                    #[expect(clippy::cast_precision_loss, reason = "offsets are small integers")]
                    let d = Vec2::new(f.x - dx as f32, f.y - dy as f32);
                    let r2 = d.length_squared();
                    // Corners outside the kernel add nothing. Which those are
                    // changes from point to point, so every corner is computed
                    // and the outside ones' terms are dropped by a select, not
                    // skipped by a branch that mispredicts. Adding zero keeps
                    // the sums' bits: they start at +0, and a float sum is −0
                    // only when both terms are. A NaN distance is not
                    // outside, as it was not when this was a `continue`.
                    let outside = r2 >= KERNEL_RADIUS_SQ;
                    let h = mix(column, row);
                    // The top byte picks the gradient; the next 24 bits give
                    // the corner's value.
                    let [gx, gy] = GRADIENTS[(h >> 56) as usize];
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "a 24-bit integer is exact in f32"
                    )]
                    let value = ((h >> 32) & 0xFF_FFFF) as f32 * (1.0 / 16_777_216.0) * 2.0 - 1.0;
                    let t = 1.0 - r2 / KERNEL_RADIUS_SQ;
                    let t2 = t * t;
                    let term = gx * d.x + gy * d.y + VALUE_WEIGHT * value;
                    sum += if outside { 0.0 } else { t2 * t2 * term };
                    if GRADIENT {
                        // d/dd of t⁴·term: 4t³·(−2d/R²)·term + t⁴·g.
                        let dk = -8.0 * t2 * t / KERNEL_RADIUS_SQ;
                        let g = d * (dk * term) + Vec2::new(gx, gy) * (t2 * t2);
                        gradient += if outside { Vec2::ZERO } else { g };
                    }
                }
            }
            (
                sum * GRADIENT_SCALE,
                gradient * GRADIENT_SCALE * lattice.frequency,
            )
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

    fn eval_gradient(&self, p: Vec2, footprint: Footprint) -> (f32, Vec2) {
        let weight = footprint.band_weight(self.lattice.max_frequency());
        if weight == 0.0 {
            return (0.0, Vec2::ZERO);
        }
        let (n, gradient) =
            lattice_noise_gradient(self.basis, self.lattice, self.seed, Vec2::ZERO, p);
        (n * weight, gradient * weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::hash;
    use alloc::vec::Vec;

    /// [`lattice_eval`] as first written, hashing every corner in full:
    /// the reference its hoisted hashing must match bit for bit.
    fn reference_eval<const GRADIENT: bool>(
        basis: Basis,
        lattice: Lattice,
        seed: u64,
        offset: Vec2,
        p: Vec2,
    ) -> (f32, Vec2) {
        let at = lattice.locate(p);
        let mut cell = at.cell;
        let mut f = at.frac + offset;
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
        match basis {
            Basis::Value => {
                let [h00, h10, h01, h11] = [corner(0, 0), corner(1, 0), corner(0, 1), corner(1, 1)];
                let (u, v) = (fade(f.x), fade(f.y));
                let value = |h: u64| unit_f32(h) * 2.0 - 1.0;
                let (a, b, c, d) = (value(h00), value(h10), value(h01), value(h11));
                let bottom = lerp(a, b, u);
                let top = lerp(c, d, u);
                let result = lerp(bottom, top, v);
                if !GRADIENT {
                    return (result, Vec2::ZERO);
                }
                let dfx = ((b - a) + ((d - c) - (b - a)) * v) * fade_derivative(f.x);
                let dfy = (top - bottom) * fade_derivative(f.y);
                (result, Vec2::new(dfx, dfy) * lattice.frequency)
            }
            Basis::Gradient => {
                let mut sum = 0.0;
                let mut gradient = Vec2::ZERO;
                for dy in -1..=2_i64 {
                    for dx in -1..=2_i64 {
                        let d = Vec2::new(f.x - dx as f32, f.y - dy as f32);
                        let r2 = d.length_squared();
                        if r2 >= KERNEL_RADIUS_SQ {
                            continue;
                        }
                        let h = corner(dx, dy);
                        let [gx, gy] = GRADIENTS[(h >> 56) as usize];
                        let value =
                            ((h >> 32) & 0xFF_FFFF) as f32 * (1.0 / 16_777_216.0) * 2.0 - 1.0;
                        let t = 1.0 - r2 / KERNEL_RADIUS_SQ;
                        let t2 = t * t;
                        let term = gx * d.x + gy * d.y + VALUE_WEIGHT * value;
                        sum += t2 * t2 * term;
                        if GRADIENT {
                            let dk = -8.0 * t2 * t / KERNEL_RADIUS_SQ;
                            gradient += d * (dk * term) + Vec2::new(gx, gy) * (t2 * t2);
                        }
                    }
                }
                (
                    sum * GRADIENT_SCALE,
                    gradient * GRADIENT_SCALE * lattice.frequency,
                )
            }
        }
    }

    #[test]
    fn hoisted_corner_hashes_match_the_reference_bit_for_bit() {
        let lattices = [
            (Domain::Plane, Vec2::new(0.37, 5.0)),
            (Domain::periodic(1, 1).unwrap(), Vec2::ONE),
            (Domain::periodic(1, 1).unwrap(), Vec2::new(8.0, 3.0)),
            (Domain::periodic(2, 3).unwrap(), Vec2::new(1.5, 2.0)),
        ]
        .map(|(domain, frequency)| Lattice::new(domain, frequency).unwrap());
        let mut points = Vec::new();
        for i in 0..512_u64 {
            let h = hash(0x6e_6f69_7365, &[i]);
            let unit = |shift: u32| unit_f32(h.rotate_left(shift)) * 2.0 - 1.0;
            let scale = [1.0, 3.0, 1e3, 1e6][(i % 4) as usize];
            points.push(Vec2::new(unit(0), unit(32)) * scale);
        }
        let below_one = 1.0 - f32::EPSILON;
        let edges = [0.0, -0.0, 1.0, -1.0, 0.5, below_one, 1e-30, f32::NAN, 1e12];
        for v in edges {
            points.extend([Vec2::splat(v), Vec2::new(v, 0.3), Vec2::new(-0.7, v)]);
        }
        let offsets = [
            Vec2::ZERO,
            Vec2::new(0.5, 0.25),
            Vec2::splat(1.0 - f32::EPSILON),
        ];
        let bits = |(v, g): (f32, Vec2)| [v.to_bits(), g.x.to_bits(), g.y.to_bits()];
        for basis in [Basis::Value, Basis::Gradient] {
            for lattice in lattices {
                for (seed, offset) in [0, 7, u64::MAX].into_iter().zip(offsets) {
                    for &p in &points {
                        assert_eq!(
                            bits(lattice_eval::<false>(basis, lattice, seed, offset, p)),
                            bits(reference_eval::<false>(basis, lattice, seed, offset, p)),
                            "{basis:?} {lattice:?} seed {seed} at {p}"
                        );
                        assert_eq!(
                            bits(lattice_eval::<true>(basis, lattice, seed, offset, p)),
                            bits(reference_eval::<true>(basis, lattice, seed, offset, p)),
                            "{basis:?} {lattice:?} seed {seed} at {p}, with gradient"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn gradients_are_unit_length() {
        for [x, y] in GRADIENTS {
            assert!((x * x + y * y - 1.0).abs() < 1e-6, "({x}, {y})");
        }
    }

    #[test]
    fn gradient_noise_is_even_across_lattice_lines() {
        // Mean |n| on lattice lines against elsewhere, over 64 × 64 cells.
        let (cells, sub) = (64_u32, 8_u32);
        let noise = Noise::new(Basis::Gradient, Domain::Plane, Vec2::ONE, 5).unwrap();
        let (mut on, mut off) = ((0.0_f64, 0_u32), (0.0_f64, 0_u32));
        for j in 0..cells * sub {
            for i in 0..cells * sub {
                let p = Vec2::new(i as f32 / sub as f32, (j as f32 + 0.37) / sub as f32);
                let v = f64::from(noise.eval(p, Footprint::POINT).abs());
                let bin = if i % sub == 0 { &mut on } else { &mut off };
                bin.0 += v;
                bin.1 += 1;
            }
        }
        let ratio = (on.0 / f64::from(on.1)) / (off.0 / f64::from(off.1));
        assert!((ratio - 1.0).abs() < 0.03, "lattice-line ratio {ratio}");
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
