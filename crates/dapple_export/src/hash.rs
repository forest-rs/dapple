// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Content hashes of written files.

use core::fmt;

use dapple_field::hash::mix;

/// A 128-bit content hash, from [`content_hash`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ContentHash(pub u128);

impl fmt::Display for ContentHash {
    /// 32 lowercase hexadecimal digits.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

/// The two lanes' seeds: "dapple-0" and "dapple-1", as `dapple_field`'s
/// program fingerprints use.
const LANES: [u64; 2] = [0x6461_7070_6c65_2d30, 0x6461_7070_6c65_2d31];

/// Purpose tag of file content hashes ("file").
const PURPOSE: u64 = 0x6669_6c65;

/// The content hash of `bytes`: two lanes of `dapple_field`'s keyed hash
/// (`exedra_math`'s frozen keyed-hash contract) folded over the bytes as
/// little-endian 64-bit words, the last zero-padded, then the length.
///
/// It detects changed and truncated files, as a stamp needs; it is not
/// cryptographic, and must not be trusted against deliberate collisions.
#[must_use]
pub fn content_hash(bytes: &[u8]) -> ContentHash {
    let mut lanes = LANES.map(|seed| mix(seed, PURPOSE));
    let (words, tail) = bytes.as_chunks::<8>();
    let mut fold = |word: u64| {
        for lane in &mut lanes {
            *lane = mix(*lane, word);
        }
    };
    for word in words {
        fold(u64::from_le_bytes(*word));
    }
    if !tail.is_empty() {
        let mut last = [0; 8];
        last[..tail.len()].copy_from_slice(tail);
        fold(u64::from_le_bytes(last));
    }
    fold(bytes.len() as u64);
    ContentHash(u128::from(lanes[0]) | (u128::from(lanes[1]) << 64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_tell_contents_and_lengths_apart() {
        let a = content_hash(b"dapple");
        assert_eq!(a, content_hash(b"dapple"), "deterministic");
        assert_ne!(a, content_hash(b"dapplf"));
        // Zero padding does not make a shorter file equal a longer one.
        assert_ne!(content_hash(b"ab"), content_hash(b"ab\0"));
        assert_ne!(content_hash(b""), content_hash(b"\0"));
        assert_eq!(alloc::format!("{a}").len(), 32);
    }
}
