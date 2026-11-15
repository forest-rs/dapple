// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Building blocks for lane kernels: kernels that evaluate one field at
//! many points at once, as structure-of-arrays loops the compiler can
//! vectorize.
//!
//! A lane kernel computes, for each point, the operations its point kernel
//! computes, in the same order, so each lane's result has the point
//! kernel's bits; only the loops are arranged differently. Its fixed-width
//! scratch arrays hold [`WIDTH`] lanes, so a kernel walks longer runs of
//! points a chunk at a time.

/// Lanes per chunk of a lane kernel: the block evaluator's block size, so
/// a block's points are one chunk.
pub(crate) const WIDTH: usize = 64;

/// `out` a chunk of at most [`WIDTH`] lanes at a time, each with the same
/// lanes of every column of `points`.
///
/// # Panics
///
/// When a column of `points` is shorter than `out`.
pub(crate) fn chunks<'a, const N: usize>(
    points: [&'a [f32]; N],
    out: &'a mut [f32],
) -> impl Iterator<Item = ([&'a [f32]; N], &'a mut [f32])> {
    out.chunks_mut(WIDTH).enumerate().map(move |(chunk, out)| {
        let start = chunk * WIDTH;
        (points.map(|column| &column[start..start + out.len()]), out)
    })
}

/// `libm::floorf(x)`, inline, so that loops calling it vectorize.
///
/// Its bits equal `libm::floorf`'s for every non-NaN `x`, checked over all
/// 2³² patterns, one at a time and in vectorized loops; a NaN stays a NaN,
/// perhaps with another payload. Below 2²³ in magnitude, truncating
/// through `i32` is exact, a truncation above `x` steps down by one, and
/// the sign is `x`'s, so `-0.0` stays `-0.0` and `-0.5` gives `-1.0`; from
/// 2²³ up every `f32` is already whole, and that branch also passes
/// infinities and NaN through.
#[inline(always)]
pub(crate) fn floor(x: f32) -> f32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "only magnitudes below 2^23 are kept, and they are exact in i32"
    )]
    let truncated = (x as i32) as f32;
    let floor = if truncated > x {
        truncated - 1.0
    } else {
        truncated
    };
    if x.abs() < 8_388_608.0 {
        floor.copysign(x)
    } else {
        x
    }
}

/// Locates each of `p`, scaled by `frequency` cells per unit, on one axis
/// of a lattice shifted by `offset` of a cell: its cell index and its
/// position in that cell, as `Lattice::locate` and the lattice kernels'
/// offset carry compute them.
///
/// The cell is the scaled point's floor, cast as `Lattice` casts it
/// (saturating, NaN to 0); the position is what remains plus `offset`,
/// moved into the next cell when it reaches 1, as the lattice kernels move
/// it.
#[inline(always)]
pub(crate) fn locate(p: &[f32], frequency: f32, offset: f32, cell: &mut [i64], frac: &mut [f32]) {
    for ((&p, cell), frac) in p.iter().zip(cell).zip(frac) {
        let q = p * frequency;
        let floor = floor(q);
        let f = (q - floor) + offset;
        let carry = f >= 1.0;
        *frac = if carry { f - 1.0 } else { f };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the cast saturates, as `Lattice::locate`'s does"
        )]
        let index = floor as i64;
        *cell = index + i64::from(carry);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use alloc::vec::Vec;

    use super::*;

    /// Whether `a` and `b` are the same bits, NaN for NaN whatever its
    /// payload: lane kernels promise no more of a NaN.
    pub(crate) fn same_bits(a: f32, b: f32) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    /// Run lengths around a lane kernel's chunk: empty, single, short, a
    /// chunk and either side of it, and several chunks with a remainder.
    pub(crate) const RUNS: [usize; 8] = [
        0,
        1,
        3,
        WIDTH - 1,
        WIDTH,
        WIDTH + 1,
        2 * WIDTH,
        2 * WIDTH + 3,
    ];

    /// Points, in cells of a lattice of `wrap` cells per period (if
    /// periodic), in the patterns a kernel that reuses a lane's cell for the
    /// next lane must get right, from the first lane on, so that the runs a
    /// test cuts them into begin at chunk boundaries:
    ///
    /// - two chunks in one cell, then a chunk in another, so the cell
    ///   changes exactly from lane 63 to lane 64 when a run starts at the
    ///   first point, and stays the same across it before;
    /// - a cell for every lane, alternating between two cells, the second
    ///   differing on one axis only;
    /// - walks along each axis, a cell every 5 lanes, crossing cell borders
    ///   in the middle of chunks;
    /// - a cell left and revisited;
    /// - NaN, signed zeros, 10¹⁵, tiny negatives (whose cell is -1 but whose
    ///   position rounds to its far border) among lanes in one cell, the
    ///   cell 0 that NaN and zero locate in and another;
    /// - on a periodic lattice, cells a period apart (one cell, wrapped)
    ///   side by side, and the cells either side of each wrap.
    pub(crate) fn coherent_cells<const N: usize>(wrap: Option<[i64; N]>) -> Vec<[f32; N]> {
        let mut out = Vec::new();
        // Positions in a cell, different from lane to lane and axis to axis.
        let at = |cell: [i64; N], lane: usize| {
            let mut p = [0.0; N];
            for (axis, p) in p.iter_mut().enumerate() {
                let step = (lane * 37 + axis * 11) % 64;
                *p = cell[axis] as f32 + (step as f32 + 0.5) / 64.0;
            }
            p
        };
        let shift = |cell: [i64; N], axis: usize, by: i64| {
            let mut cell = cell;
            cell[axis] += by;
            cell
        };
        let home = [2; N];
        for i in 0..3 * WIDTH {
            let cell = if i < 2 * WIDTH {
                home
            } else {
                shift(home, 0, 1)
            };
            out.push(at(cell, i));
        }
        for axis in 0..N {
            for i in 0..70 {
                let cell = if i % 2 == 0 {
                    home
                } else {
                    shift(home, axis, 1)
                };
                out.push(at(cell, i));
            }
            let walk = (-3_i64..).flat_map(|cell| [cell; 5]);
            for (i, by) in (0..70).zip(walk) {
                out.push(at(shift(home, axis, by), i));
            }
        }
        for (i, by) in [0, 0, 0, 1, 1, 0, 0, -1, 0, 0].into_iter().enumerate() {
            out.push(at(shift(home, N - 1, by), i));
        }
        for cell in [home, [0; N], [-1; N]] {
            let edges = [f32::NAN, 0.0, -0.0, 1e15, -1e-30, -0.0, f32::NAN];
            for (i, edge) in edges.into_iter().enumerate() {
                for axis in 0..N {
                    out.push(at(cell, 2 * i));
                    let mut p = at(cell, 2 * i + 1);
                    p[axis] = edge;
                    out.push(p);
                    out.push(at(cell, 2 * i + 2));
                }
            }
        }
        if let Some(period) = wrap {
            // One cell, unwrapped a period or two apart.
            for i in 0..40 {
                let laps = if i % 2 == 0 { 0 } else { [1, 2, 3][i % 3] };
                let cell = core::array::from_fn(|axis| home[axis] + period[axis] * laps);
                out.push(at(cell, i));
            }
            // Runs of four lanes over the last two cells of a period and
            // the first two of the next, then the same a period lower.
            let steps = [-2_i64, -1, 0, 1].into_iter().flat_map(|step| [step; 4]);
            for (i, step) in (0..32).zip(steps.cycle()) {
                let laps = if i < 16 { 1 } else { 0 };
                let cell = core::array::from_fn(|axis| period[axis] * laps + step);
                out.push(at(cell, i));
            }
        }
        out
    }

    /// The full proof walks all 2³² patterns (a scratch harness, about two
    /// seconds); this keeps the edges and a strided sweep in the suite.
    #[test]
    fn floor_matches_libm() {
        let check = |x: f32| {
            let (got, want) = (floor(x), libm::floorf(x));
            assert!(
                same_bits(got, want),
                "floor({x:e}) = {got:e}, libm gives {want:e}"
            );
        };
        for x in [
            0.0,
            -0.0,
            0.5,
            -0.5,
            1.0,
            -1.0,
            1.5,
            -1.5,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1e-45,
            -1e-45,
            8_388_607.5,
            -8_388_607.5,
            8_388_608.0,
            -8_388_608.0,
            16_777_217.0,
            2_147_483_648.0,
            -2_147_483_904.0,
            1e15,
            -1e15,
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
        ] {
            check(x);
            check(f32::from_bits(x.to_bits().wrapping_add(1)));
            check(f32::from_bits(x.to_bits().wrapping_sub(1)));
        }
        for bits in (0..=u32::MAX).step_by(4099) {
            check(f32::from_bits(bits));
        }
    }
}
