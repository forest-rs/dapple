// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tile-level invalidation inside realizations.
//!
//! Every realize and raster node splits its raster into square tiles. Tile
//! keys `(node, level, tile)` are interned into an
//! [`InvalidationTracker`], whose edges run from each raster-node tile to the
//! input tiles its kernel footprint reads. A node that rewrites tiles compares
//! each recomputed tile with its previous bits and marks the direct
//! dependents of the tiles that actually changed, so a downstream node
//! recomputes only tiles whose inputs changed, and a change that leaves a
//! tile's bits alone stops there.

use alloc::vec::Vec;

use dapple_raster::{Edge, TexelRect};
use invalidation::intern::{InternId, Interner};
use invalidation::{Channel, InvalidationTracker};

/// The single invalidation channel: tile content.
const TILES: Channel = Channel::new(0);

/// Default tile side, in texels.
pub const DEFAULT_TILE_SIZE: u32 = 32;

/// One tile of one node's raster at one mip level.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
struct TileKey {
    node: u32,
    level: u8,
    tile: u32,
}

/// Tile work done by one [`MaterialGraph::run`](crate::MaterialGraph::run).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct TileReport {
    /// Tiles realized or recomputed.
    pub tiles_recomputed: u64,
    /// Tiles kept from the previous run of a node that ran.
    pub tiles_reused: u64,
    /// Recomputed tiles whose bits differ from the previous run's.
    pub tiles_changed: u64,
    /// Texels realized or recomputed.
    pub texels_recomputed: u64,
    /// Node runs that recomputed every tile: a first run, a new grid or
    /// operation, a global operation, or an unbounded field change.
    pub whole_recomputes: u64,
    /// Realizations recomputed whole only because their field changed with
    /// no stated region, although a previous raster existed.
    pub unbounded_changes: u64,
}

/// The tile layout of one raster.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct Grid {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) size: u32,
    pub(crate) wrap: bool,
}

impl Grid {
    pub(crate) fn new(width: u32, height: u32, size: u32, edge: Edge) -> Self {
        Self {
            width,
            height,
            size,
            wrap: edge == Edge::Wrap,
        }
    }

    pub(crate) const fn columns(&self) -> u32 {
        self.width.div_ceil(self.size)
    }

    pub(crate) const fn rows(&self) -> u32 {
        self.height.div_ceil(self.size)
    }

    pub(crate) const fn count(&self) -> u32 {
        self.columns() * self.rows()
    }

    pub(crate) fn rect(&self, tile: u32) -> TexelRect {
        let (column, row) = (tile % self.columns(), tile / self.columns());
        TexelRect {
            x0: column * self.size,
            y0: row * self.size,
            x1: ((column + 1) * self.size).min(self.width),
            y1: ((row + 1) * self.size).min(self.height),
        }
    }

    /// Tiles on one axis holding any texel of `lo..hi`, which may leave the
    /// grid: wrapped around on a torus, clipped otherwise.
    fn spans(&self, lo: i64, hi: i64, extent: u32) -> Vec<u32> {
        let size = i64::from(self.size);
        let extent = i64::from(extent);
        let mut ranges = Vec::with_capacity(2);
        if self.wrap {
            if hi - lo >= extent {
                ranges.push((0, extent));
            } else if lo < hi {
                let a = lo.rem_euclid(extent);
                let b = a + (hi - lo);
                if b <= extent {
                    ranges.push((a, b));
                } else {
                    ranges.push((a, extent));
                    ranges.push((0, b - extent));
                }
            }
        } else if lo.max(0) < hi.min(extent) {
            ranges.push((lo.max(0), hi.min(extent)));
        }
        let mut out = Vec::new();
        for (a, b) in ranges {
            for t in a.div_euclid(size)..=(b - 1).div_euclid(size) {
                out.push(u32::try_from(t).expect("tile index fits u32"));
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Tiles holding any texel of `x0..x1` × `y0..y1`.
    pub(crate) fn overlapping(&self, x0: i64, x1: i64, y0: i64, y1: i64) -> Vec<u32> {
        let columns = self.spans(x0, x1, self.width);
        let rows = self.spans(y0, y1, self.height);
        let mut out = Vec::with_capacity(columns.len() * rows.len());
        for row in &rows {
            for column in &columns {
                out.push(row * self.columns() + column);
            }
        }
        out.sort_unstable();
        out
    }
}

/// Tile keys and the tracker connecting them.
#[derive(Debug, Default)]
pub(crate) struct Tiles {
    size: u32,
    interner: Interner<TileKey>,
    tracker: InvalidationTracker<InternId>,
    pub(crate) report: TileReport,
}

impl Tiles {
    pub(crate) fn new(size: u32) -> Self {
        Self {
            size,
            ..Self::default()
        }
    }

    pub(crate) const fn size(&self) -> u32 {
        self.size
    }

    /// Interned keys for every tile of `node` on `grid`, in tile order.
    ///
    /// `previous` are the node's keys on its previous grid; keys no longer
    /// used leave the tracker.
    pub(crate) fn keys(&mut self, node: u32, grid: Grid, previous: &[InternId]) -> Vec<InternId> {
        let keys: Vec<InternId> = (0..grid.count())
            .map(|tile| {
                self.interner.intern(TileKey {
                    node,
                    level: 0,
                    tile,
                })
            })
            .collect();
        // Tile keys are interned by index, so the keys a smaller grid no
        // longer uses are exactly those past its count.
        for &key in previous {
            if self
                .interner
                .get(key)
                .is_some_and(|k| k.node == node && k.tile >= grid.count())
            {
                self.tracker.remove_key(key);
            }
        }
        keys
    }

    /// The key of tile `tile` of `node`.
    pub(crate) fn key(&mut self, node: u32, tile: u32) -> InternId {
        self.interner.intern(TileKey {
            node,
            level: 0,
            tile,
        })
    }

    /// Makes tile `key` depend on exactly `inputs`.
    pub(crate) fn depend(&mut self, key: InternId, inputs: impl IntoIterator<Item = InternId>) {
        self.tracker
            .replace_dependencies(key, TILES, inputs)
            .expect("tile edges point upstream, so they cannot form a cycle");
    }

    /// Removes every input edge of `keys`, for a node whose tiles all depend
    /// on its whole input.
    pub(crate) fn detach(&mut self, keys: &[InternId]) {
        for &key in keys {
            self.depend(key, []);
        }
    }

    /// Takes the tiles of `keys` that upstream changes marked, in tile order.
    pub(crate) fn take_marked(&mut self, keys: &[InternId]) -> Vec<u32> {
        let marked: Vec<InternId> = self
            .tracker
            .drain(TILES)
            .within_keys(keys)
            .deterministic()
            .run()
            .collect();
        let mut tiles: Vec<u32> = marked
            .iter()
            .filter_map(|id| self.interner.get(*id).map(|key| key.tile))
            .collect();
        tiles.sort_unstable();
        tiles
    }

    /// Marks the direct dependents of `changed`, without propagating
    /// further: each dependent marks its own dependents only if its bits
    /// change.
    pub(crate) fn mark_dependents(&mut self, changed: impl IntoIterator<Item = InternId>) {
        let mut dependents = Vec::new();
        for key in changed {
            dependents.extend(self.tracker.graph().dependents(key, TILES));
        }
        for key in dependents {
            self.tracker.mark(key, TILES);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grids_split_into_edge_tiles() {
        let grid = Grid::new(70, 40, 32, Edge::Clamp);
        assert_eq!((grid.columns(), grid.rows(), grid.count()), (3, 2, 6));
        assert_eq!(
            grid.rect(5),
            TexelRect {
                x0: 64,
                y0: 32,
                x1: 70,
                y1: 40
            }
        );
    }

    #[test]
    fn overlaps_clip_or_wrap() {
        let clamp = Grid::new(70, 40, 32, Edge::Clamp);
        assert_eq!(clamp.overlapping(-5, 3, -5, 3), [0]);
        assert_eq!(clamp.overlapping(60, 80, 30, 34), [1, 2, 4, 5]);
        let wrap = Grid::new(70, 40, 32, Edge::Wrap);
        // Columns -5..3 wrap to 65..70 (tile 2) and 0..3 (tile 0).
        assert_eq!(wrap.overlapping(-5, 3, 0, 1), [0, 2]);
        // Rows 38..42 wrap to 38..40 (row 1) and 0..2 (row 0).
        assert_eq!(wrap.overlapping(0, 1, 38, 42), [0, 3]);
        assert_eq!(wrap.overlapping(0, 200, 0, 1).len(), 3);
    }
}
