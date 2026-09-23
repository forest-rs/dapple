// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Keyed deterministic randomness.
//!
//! Every random decision in dapple is a pure function of a seed and a key:
//! `hash(seed, [purpose, x, y, …])`. There are no sequential random streams, so
//! values do not depend on evaluation order, tile size, or thread count.
//!
//! The functions are `exedra_math::keyed`, re-exported: version 1 of the
//! keyed-hash contract that sylva shares, frozen by `exedra_math`'s ADR-0001.
//! This module keeps only dapple's own key derivation, `key`, for signed
//! lattice coordinates.
//!
//! `mix(0, 0)` is `0`, so a zero seed with an all-zero key hashes to zero.
//! Dapple leads every key with a nonzero purpose tag.

pub use exedra_math::keyed::{hash, mix, splitmix64_mix, unit_f32, unit_f64};

/// Reinterprets a signed lattice coordinate as a hash key word.
#[inline]
pub(crate) const fn key(v: i64) -> u64 {
    v.cast_unsigned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_vectors() {
        assert_eq!(hash(0, &[]), 0);
        assert_eq!(hash(1, &[2, 3]), 0x614a_eb9e_d12c_cf8d);
        assert_eq!(hash(u64::MAX, &[0]), 0xb4d0_55fc_f2cb_bd7b);

        assert_eq!(unit_f32(0).to_bits(), 0);
        assert_eq!(unit_f32(0x614a_eb9e_d12c_cf8d).to_bits(), 0x3ec2_95d6);
        assert_eq!(unit_f32(0xb4d0_55fc_f2cb_bd7b).to_bits(), 0x3f34_d055);

        assert_eq!(unit_f64(0x614a_eb9e_d12c_cf8d), 0.380_049_444_459_632_4);
        assert_eq!(unit_f64(0xb4d0_55fc_f2cb_bd7b), 0.706_303_953_413_949_6);
    }

    #[test]
    fn hash_folds_left() {
        assert_eq!(hash(7, &[1, 2]), mix(mix(7, 1), 2));
        assert_ne!(hash(7, &[1, 2]), hash(7, &[2, 1]), "keys are ordered");
    }

    #[test]
    fn unit_ranges_are_half_open() {
        assert!(unit_f32(u64::MAX) < 1.0);
        assert!(unit_f64(u64::MAX) < 1.0);
    }
}
