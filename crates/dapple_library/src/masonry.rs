// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Unit layouts for masonry modules: the contract between a layout and the
//! material laid on it.
//!
//! A masonry module (ashlar, rubble, flint, brick, marble, roof tile) needs
//! three things of its units at every texel, whoever laid them out:
//!
//! - **`units`** (identifiers): the nearest unit's identity, everywhere,
//!   joints included, so a unit's color, tooling and wear follow its key
//!   rather than its position. `0` means no unit is known.
//! - **`edge`** (scalars): the signed distance to that unit's outline, in
//!   meters, negative inside: arrises round and mortar tools from it.
//! - **`local`** (two-vectors): the texel's position in the unit's own
//!   frame, over its half extent (`[-1, 1]` across the unit), so a roof
//!   tile thickens toward its lower edge and tooling runs along a stone.
//!
//! [`UnitMaps::from_elements`] rasterizes any [`ElementSet`] into them: the
//! construction layer's unit layout, or the layout a module makes for
//! itself when the host supplies none. Identities come from element keys
//! ([`unit_id`]), so a unit keeps its appearance when the layout moves it.

use alloc::vec;
use alloc::vec::Vec;

use dapple_elements::{ElementKey, ElementSet};
use dapple_field::{Edge, PortType, Value};
use dapple_material::Grid;
use dapple_raster::typed::{Storage, TypedError, TypedRaster};
use glam::Vec2;

/// A unit's identity in a `units` map: the low 32 bits of its element key,
/// never 0.
#[must_use]
pub fn unit_id(key: ElementKey) -> u32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "identity maps hold 32 bits of the key"
    )]
    let id = key.word() as u32;
    id.max(1)
}

/// The three maps a masonry module reads: see the [module docs](self).
#[derive(Clone, Debug, PartialEq)]
pub struct UnitMaps {
    /// The nearest unit's identity ([`PortType::Id`]).
    pub units: TypedRaster,
    /// Signed distance to its outline, in meters, negative inside
    /// ([`PortType::Scalar`]).
    pub edge: TypedRaster,
    /// Position in the unit's frame over its half extent
    /// ([`PortType::Vector2`]).
    pub local: TypedRaster,
}

/// The maps as plain values, for module bodies.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Units {
    pub(crate) id: Vec<u32>,
    pub(crate) edge: Vec<f32>,
    pub(crate) local: Vec<Vec2>,
}

impl UnitMaps {
    /// Rasterizes `set` onto `grid`: every texel takes the element whose
    /// outline is nearest (inside wins over outside), searching `reach`
    /// meters beyond each element's bounds. On a wrapping grid, elements
    /// repeat with the grid's extent, so a layout over one period tiles.
    ///
    /// # Errors
    ///
    /// [`TypedError`] if the maps cannot be built on `grid`.
    pub fn from_elements(set: &ElementSet, grid: Grid, reach: f32) -> Result<Self, TypedError> {
        rasterize(set, grid, reach).to_maps(grid)
    }

    pub(crate) fn values(&self) -> Option<Units> {
        let (Storage::U32(id), Storage::F32(edge), Storage::F32x2(local)) = (
            self.units.storage(),
            self.edge.storage(),
            self.local.storage(),
        ) else {
            return None;
        };
        Some(Units {
            id: id.values().to_vec(),
            edge: edge.values().to_vec(),
            local: local
                .values()
                .iter()
                .map(|v| Vec2::from_array(*v))
                .collect(),
        })
    }
}

impl Units {
    pub(crate) fn to_maps(&self, grid: Grid) -> Result<UnitMaps, TypedError> {
        Ok(UnitMaps {
            units: grid.typed(PortType::Id, self.id.iter().map(|&v| Value::Id(v)))?,
            edge: grid.typed(
                PortType::Scalar,
                self.edge.iter().map(|&v| Value::Scalar(v)),
            )?,
            local: grid.typed(
                PortType::Vector2,
                self.local.iter().map(|&v| Value::Vector2(v)),
            )?,
        })
    }
}

/// The grid's extent in meters.
pub(crate) fn extent(grid: Grid) -> Vec2 {
    #[expect(clippy::cast_precision_loss, reason = "grid sizes are small")]
    let e = grid.texel * Vec2::new(grid.width as f32, grid.height as f32);
    e
}

pub(crate) fn rasterize(set: &ElementSet, grid: Grid, reach: f32) -> Units {
    let n = grid.len();
    let mut best = vec![f32::INFINITY; n];
    let mut id = vec![0_u32; n];
    let mut local = vec![Vec2::ZERO; n];
    let wrap = grid.edge == Edge::Wrap;
    let (w, h) = (i64::from(grid.width), i64::from(grid.height));
    for i in 0..set.len() {
        let e = set.element(i);
        let b = set.bounds(i).grown(reach);
        // Texel columns and rows the grown bounds touch, unwrapped.
        let to_texel = |v: f32, o: f32, t: f32| libm::floorf((v - o) / t);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "texel indices near the grid are small"
        )]
        let (x0, x1, y0, y1) = (
            to_texel(b.min.x, grid.origin.x, grid.texel.x) as i64,
            to_texel(b.max.x, grid.origin.x, grid.texel.x) as i64,
            to_texel(b.min.y, grid.origin.y, grid.texel.y) as i64,
            to_texel(b.max.y, grid.origin.y, grid.texel.y) as i64,
        );
        for ty in y0..=y1 {
            for tx in x0..=x1 {
                let (cx, cy) = if wrap {
                    (tx.rem_euclid(w), ty.rem_euclid(h))
                } else if (0..w).contains(&tx) && (0..h).contains(&ty) {
                    (tx, ty)
                } else {
                    continue;
                };
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "wrapped texel indices are in the grid"
                )]
                let k = (cy * w + cx) as usize;
                // The texel center, in the element's repeat.
                #[expect(clippy::cast_precision_loss, reason = "texel indices are small")]
                let p = grid.origin + grid.texel * Vec2::new(tx as f32 + 0.5, ty as f32 + 0.5);
                let q = e.placement.to_local(p);
                let d = e.outline.signed_distance(q, e.half_size);
                if d < best[k] {
                    best[k] = d;
                    id[k] = unit_id(e.key);
                    local[k] = q / e.half_size;
                }
            }
        }
    }
    for b in &mut best {
        if !b.is_finite() {
            *b = reach;
        }
    }
    Units {
        id,
        edge: best,
        local,
    }
}
