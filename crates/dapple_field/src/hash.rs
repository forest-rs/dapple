// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Keyed deterministic randomness.
//!
//! Every random decision in dapple is a pure function of a seed and a key:
//! `hash(seed, [purpose, x, y, …])`. There are no sequential random streams, so
//! values do not depend on evaluation order, tile size, or thread count.
//!
//! The contract is shared with sylva and is intended to move to `exedra_math`.
//! It must not change:
//!
//! - [`splitmix64_mix`] is the SplitMix64 finalizer.
//! - [`mix`]`(h, k) = splitmix64_mix(h ^ k.wrapping_mul(0x9E3779B97F4A7C15))`.
//! - [`hash`] folds the keys left, starting from the seed. An empty key returns
//!   the seed unchanged.
//! - [`unit_f32`] takes the top 24 bits, and [`unit_f64`] the top 53 bits, as
//!   exact fractions in `[0, 1)`.
//!
//! Golden vectors:
//!
//! | call | result | `unit_f32` | `unit_f64` |
//! |---|---|---|---|
//! | `hash(0, [])` | `0x0` | `0.0` | `0.0` |
//! | `hash(1, [2, 3])` | `0x614aeb9ed12ccf8d` | `0.38004940748214720` | `0.3800494444596324` |
//! | `hash(u64::MAX, [0])` | `0xb4d055fcf2cbbd7b` | `0.70630389451980590` | `0.7063039534139496` |
//!
//! `mix(0, 0)` is `0`, so a zero seed with an all-zero key hashes to zero.
//! Callers should lead every key with a nonzero purpose tag, as dapple's own
//! fields do.

/// Golden-ratio increment used by [`mix`].
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// The SplitMix64 finalizer: a bijective avalanche of `z`.
#[must_use]
#[inline]
pub const fn splitmix64_mix(z: u64) -> u64 {
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Mixes one key word into a running hash.
#[must_use]
#[inline]
pub const fn mix(h: u64, k: u64) -> u64 {
    splitmix64_mix(h ^ k.wrapping_mul(GOLDEN))
}

/// Hashes `keys` under `seed`, folding left from the seed.
///
/// ```
/// use dapple_field::hash::{hash, unit_f32};
///
/// assert_eq!(hash(1, &[2, 3]), 0x614a_eb9e_d12c_cf8d);
/// assert_eq!(unit_f32(hash(1, &[2, 3])).to_bits(), 0x3ec2_95d6);
/// ```
#[must_use]
#[inline]
pub const fn hash(seed: u64, keys: &[u64]) -> u64 {
    let mut h = seed;
    let mut i = 0;
    while i < keys.len() {
        h = mix(h, keys[i]);
        i += 1;
    }
    h
}

/// The top 24 bits of `h` as an exact `f32` in `[0, 1)`.
#[must_use]
#[inline]
pub const fn unit_f32(h: u64) -> f32 {
    (h >> 40) as f32 * (1.0 / 16_777_216.0)
}

/// The top 53 bits of `h` as an exact `f64` in `[0, 1)`.
#[must_use]
#[inline]
pub const fn unit_f64(h: u64) -> f64 {
    (h >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

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
