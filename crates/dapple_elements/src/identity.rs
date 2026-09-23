// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Element identity.
//!
//! Four things stay distinct:
//!
//! - an element's **identity**, its [`ElementKey`]: which authored or
//!   generated element it is;
//! - its **content fingerprint** ([`ElementSet::fingerprint_of`]): whether
//!   its placement, size, variant or attributes changed;
//! - its **dense index**: where it is stored in one set or one realization,
//!   such as a label in an owner raster;
//! - **correspondence** ([`correspondence`]): how the elements of one
//!   version relate to those of another, reported separately.
//!
//! [`ElementSet::fingerprint_of`]: crate::ElementSet::fingerprint_of
//! [`correspondence`]: crate::correspondence

use core::fmt;

use dapple_field::hash::{hash, unit_f32};

/// Purpose tag of element keys ("elemkey").
const KEY_TAG: u64 = 0x0065_6c65_6d6b_6579;
/// Purpose tag of layout identities ("layoutid").
const LAYOUT_TAG: u64 = 0x6c61_796f_7574_6964;
/// Purpose tag of per-element random streams ("elemrand").
const RANDOM_TAG: u64 = 0x656c_656d_7261_6e64;

/// A layout's logical identity: a hash of its author-given name.
///
/// It is **not** the layout's content fingerprint. Changing a layout's
/// parameters (joint width, brick size) keeps its identity, so every element
/// keeps its key.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LayoutId(u64);

/// A hash of `name` under purpose `tag`.
pub(crate) fn name_word(tag: u64, name: &str) -> u64 {
    let mut h = hash(tag, &[name.len() as u64]);
    for chunk in name.as_bytes().chunks(8) {
        let mut word = [0_u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        h = hash(h, &[u64::from_le_bytes(word)]);
    }
    h
}

impl LayoutId {
    /// The identity of the layout named `name`.
    #[must_use]
    pub fn named(name: &str) -> Self {
        Self(name_word(LAYOUT_TAG, name))
    }

    /// The identity's hash word.
    #[must_use]
    pub const fn word(self) -> u64 {
        self.0
    }
}

/// Where a generator anchored an element: a lattice cell such as a brick's
/// course and index. Anchors are generator coordinates, not positions; a
/// moved element keeps its anchor.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Anchor(pub [i32; 2]);

/// A stable element identity: a keyed hash of its layout's identity, its
/// anchor and its slot within the anchor.
///
/// Keys order elements canonically. They are hashes, so distinct elements
/// have distinct keys with overwhelming probability; an [`ElementSet`]
/// refuses a collision rather than merging elements.
///
/// [`ElementSet`]: crate::ElementSet
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ElementKey(u64);

impl ElementKey {
    /// The key of the element `slot` at `anchor` of `layout`.
    #[must_use]
    pub fn new(layout: LayoutId, anchor: Anchor, slot: u32) -> Self {
        let [a, b] = anchor.0;
        Self(hash(
            KEY_TAG,
            &[
                layout.word(),
                u64::from(a.cast_unsigned()),
                u64::from(b.cast_unsigned()),
                u64::from(slot),
            ],
        ))
    }

    /// A key from its raw word, for keys persisted elsewhere.
    #[must_use]
    pub const fn from_word(word: u64) -> Self {
        Self(word)
    }

    /// The key's hash word.
    #[must_use]
    pub const fn word(self) -> u64 {
        self.0
    }

    /// A uniform value in `[0, 1)` for this element and `stream`: the
    /// identity-derived randomness an element keeps wherever it moves.
    #[must_use]
    pub fn unit(self, stream: u64) -> f32 {
        unit_f32(hash(RANDOM_TAG, &[self.0, stream]))
    }
}

impl fmt::Debug for ElementKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ElementKey({:016x})", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_depend_on_layout_anchor_and_slot_only() {
        let a = LayoutId::named("wall");
        let b = LayoutId::named("wall.2");
        assert_eq!(a, LayoutId::named("wall"));
        assert_ne!(a, b);
        let k = ElementKey::new(a, Anchor([3, -1]), 0);
        assert_eq!(k, ElementKey::new(a, Anchor([3, -1]), 0));
        assert_ne!(k, ElementKey::new(b, Anchor([3, -1]), 0));
        assert_ne!(k, ElementKey::new(a, Anchor([-1, 3]), 0));
        assert_ne!(k, ElementKey::new(a, Anchor([3, -1]), 1));
        assert_ne!(k.unit(1), k.unit(2));
        assert!((0.0..1.0).contains(&k.unit(1)));
    }
}
