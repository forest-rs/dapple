// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Layouts: generators of keyed elements.

use alloc::vec::Vec;

use dapple_field::{Domain, Value};
use glam::Vec2;

use crate::identity::{Anchor, ElementKey, LayoutId};
use crate::set::{AttributeDecl, Element, ElementError, ElementSet, Placement};

/// Running-bond brickwork over one period of a periodic domain: `courses`
/// rows of `per_course` bricks, alternate courses offset by half a brick,
/// with `joint`-wide mortar between bricks.
///
/// Brick `i` of course `c` has anchor `[i, c]` and slot 0, so its key
/// depends only on the layout's identity and its place in the bond:
/// changing the joint width or the course count moves and resizes bricks
/// without changing the keys of those that remain. Changing `per_course` or
/// `courses` removes or adds the bricks at anchors outside the new bond.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RunningBond {
    /// The layout's logical identity.
    pub layout: LayoutId,
    /// The periodic domain the bond tiles.
    pub domain: Domain,
    /// Rows of bricks per period; even, so the bond tiles.
    pub courses: u32,
    /// Bricks per row per period.
    pub per_course: u32,
    /// Mortar width, in domain units.
    pub joint: f32,
}

impl RunningBond {
    /// The bond's elements, with attribute values from `attributes`, in
    /// schema order, for each key and anchor.
    ///
    /// # Errors
    ///
    /// [`ElementError::InvalidGeometry`] for a non-periodic domain, an odd or
    /// zero course count, no bricks per course, or a joint that leaves no
    /// brick; otherwise as [`ElementSet::new`].
    pub fn elements(
        &self,
        schema: Vec<AttributeDecl>,
        mut attributes: impl FnMut(ElementKey, Anchor) -> Vec<Value>,
    ) -> Result<ElementSet, ElementError> {
        let invalid = ElementError::InvalidGeometry(ElementKey::from_word(0));
        let Domain::Periodic { period } = self.domain else {
            return Err(invalid);
        };
        if self.courses == 0 || !self.courses.is_multiple_of(2) || self.per_course == 0 {
            return Err(invalid);
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "periods and counts are far below f32's exact integer range"
        )]
        let cell = Vec2::new(
            period[0] as f32 / self.per_course as f32,
            period[1] as f32 / self.courses as f32,
        );
        let half = (cell - self.joint) * 0.5;
        if !(self.joint >= 0.0 && half.min_element() > 0.0) {
            return Err(invalid);
        }
        let mut elements = Vec::with_capacity((self.courses * self.per_course) as usize);
        for c in 0..self.courses {
            for i in 0..self.per_course {
                let anchor = Anchor([
                    i32::try_from(i).map_err(|_| invalid.clone())?,
                    i32::try_from(c).map_err(|_| invalid.clone())?,
                ]);
                let key = ElementKey::new(self.layout, anchor, 0);
                let offset = if c % 2 == 1 { 0.5 } else { 0.0 };
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "counts are far below f32's exact integer range"
                )]
                let center = Vec2::new(
                    (i as f32 + 0.5 + offset) * cell.x,
                    (c as f32 + 0.5) * cell.y,
                );
                elements.push(Element {
                    key,
                    placement: Placement::at(center),
                    half_size: half,
                    variant: 0,
                    attributes: attributes(key, anchor),
                });
            }
        }
        ElementSet::new(schema, elements)
    }
}
