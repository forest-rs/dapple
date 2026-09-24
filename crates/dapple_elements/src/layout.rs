// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Layouts: generators of keyed elements.

use alloc::vec::Vec;

use dapple_field::{Domain, Value};
use glam::Vec2;

use dapple_field::{Scatter, SplatPlacement};

use crate::identity::{Anchor, ElementKey, LayoutId};
use crate::set::{AttributeDecl, Element, ElementError, ElementSet, Outline, Placement};

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
                    outline: Outline::Rectangle,
                    variant: 0,
                    attributes: attributes(key, anchor),
                });
            }
        }
        ElementSet::new(schema, elements)
    }
}

/// One kind of scattered element: its share of the elements, its outline,
/// and its aspect (half height over half width).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ScatterVariant {
    /// Relative share, positive; shares need not sum to one.
    pub weight: f32,
    /// The outline.
    pub outline: Outline,
    /// Half height over half width: 1 for a disk or square, below 1 for a
    /// flat flake, above for a tall one.
    pub aspect: f32,
}

/// Elements scattered as `dapple_field::Scatter` places its splats: at
/// most one per lattice cell, present with the placement's density,
/// jittered in the cell, with a random radius and, optionally, a random
/// turn. The positions, radii and turns are exactly the field's, so
/// `Op::Scatter` is the field-level lowering of this layout.
///
/// Each element is `radius` wide and `radius · aspect` tall along its local
/// axes, in the variant its key draws from `variants` by weight (element
/// variant `i` is `variants[i]`), so a program can choose shapes and
/// sub-materials per variant (`Binding::Variant`).
///
/// Keys come from the layout's identity and the element's cell, so a
/// density or radius edit keeps the keys of the elements that remain; a
/// new frequency re-cells, and replaces, every element.
#[derive(Clone, Debug, PartialEq)]
pub struct ScatterLayout {
    /// The layout's logical identity.
    pub layout: LayoutId,
    /// Where the splats go; its domain must be periodic.
    pub scatter: Scatter,
    /// The kinds of element, at least one.
    pub variants: Vec<ScatterVariant>,
}

/// The stream variants are drawn from.
const VARIANT_STREAM: u64 = 0x0076_6172_6961_6e74; // "variant"

impl ScatterLayout {
    /// The scattered elements over one period, with attribute values from
    /// `attributes` for each key and splat.
    ///
    /// # Errors
    ///
    /// [`ElementError::InvalidGeometry`] for a scatter on the plane, no
    /// variants, a weight that is not positive and finite, or an aspect
    /// that is not; otherwise as [`ElementSet::new`].
    pub fn elements(
        &self,
        schema: Vec<AttributeDecl>,
        mut attributes: impl FnMut(ElementKey, SplatPlacement) -> Vec<Value>,
    ) -> Result<ElementSet, ElementError> {
        let invalid = ElementError::InvalidGeometry(ElementKey::from_word(0));
        let Some([nx, ny]) = self.scatter.cells() else {
            return Err(invalid);
        };
        let valid = |v: f32| v.is_finite() && v > 0.0;
        if self.variants.is_empty()
            || !self
                .variants
                .iter()
                .all(|v| valid(v.weight) && valid(v.aspect))
        {
            return Err(invalid);
        }
        let total: f32 = self.variants.iter().map(|v| v.weight).sum();
        let mut elements = Vec::new();
        for cy in 0..ny {
            for cx in 0..nx {
                let Some(splat) = self.scatter.splat_at([cx, cy]) else {
                    continue;
                };
                let anchor = Anchor([
                    i32::try_from(cx).map_err(|_| invalid.clone())?,
                    i32::try_from(cy).map_err(|_| invalid.clone())?,
                ]);
                let key = ElementKey::new(self.layout, anchor, 0);
                let mut pick = key.unit(VARIANT_STREAM) * total;
                let mut variant = self.variants.len() - 1;
                for (i, v) in self.variants.iter().enumerate() {
                    if pick < v.weight {
                        variant = i;
                        break;
                    }
                    pick -= v.weight;
                }
                let kind = self.variants[variant];
                elements.push(Element {
                    key,
                    placement: Placement {
                        center: splat.center,
                        rotation: splat.rotation,
                    },
                    half_size: Vec2::new(splat.radius, splat.radius * kind.aspect),
                    outline: kind.outline,
                    variant: u32::try_from(variant).map_err(|_| invalid.clone())?,
                    attributes: attributes(key, splat),
                });
            }
        }
        ElementSet::new(schema, elements)
    }
}
