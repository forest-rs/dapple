// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! How nodes learn what changed since they last ran.
//!
//! A node keeps the output it last published and enough state to tell, on
//! its next run, what changed in its inputs since it last read them. Fields
//! say so with a change and a serial; rasters with tile marks. Whatever runs
//! in between (runs that fail, budgets that leave tiles for later, early
//! cutoff), the rules below keep every published output equal, bit for bit,
//! to computing the graph from scratch.
//!
//! # Fields: changes and serials
//!
//! A field output ([`FieldValue`]) carries `previous`, the fingerprint of
//! its producer's output before it, `change`, where its values can differ
//! from that output, and `serial`, which counts the producer's distinct
//! outputs.
//!
//! - A producer advances `serial` exactly when its output does not
//!   [repeat](repeats) the last one: a new program, or the same program
//!   reporting a change. A repeat is cut off (`values_equal` uses the same
//!   test), so no consumer ever sees it, and it keeps the serial.
//! - A consumer remembers the [version] it last consumed. It applies an
//!   input's `change` only to the output right after that one, has nothing
//!   new when it sees that output again (a node re-running to work off
//!   pending tiles), and assumes everything changed when it missed an
//!   output, since changes do not compose ([`change_since`]).
//! - An equal fingerprint alone never means "unchanged": a sampled image's
//!   fingerprint names its rasters' derivations, not their texels, which a
//!   tile budget fills in over several runs.
//!
//! # Rasters: tile marks
//!
//! Each tile of a raster node is a key in the `invalidation` tracker, wired
//! to the input tiles it reads.
//!
//! - After computing, a node compares each recomputed tile with the output
//!   it last published and marks the direct dependents of the tiles whose
//!   bits changed. A mark stays until its node takes it, so a node's marks
//!   cover every output of its inputs it has not read, however many there
//!   were.
//! - Consumers learn what changed in a raster only from their marks.
//!   [`RasterValue::changed`] says whether an output differs from its
//!   producer's previous one, which is what early cutoff needs and not what
//!   a consumer that missed outputs needs.
//! - A node rewires its tiles when its [`Layout`] or grid changes, which
//!   covers every change to what its tiles read. Tile keys a shrinking grid
//!   no longer uses stay in the tracker, with the edges dependents wired to
//!   them, until those dependents rewire (see [`Tiles::keys`]).
//! - A node computed whole wires one tile to every input tile, and any mark
//!   recomputes it.
//!
//! # One run of a tile node
//!
//! Every node that outputs a raster runs through [`run_tiles`]:
//!
//! 1. It takes its layout from its state. Until the run succeeds, the node
//!    does not know what its kept output was computed with.
//! 2. It can reuse tiles only when its layout and grid are unchanged;
//!    otherwise it rekeys and rewires them.
//! 3. It learns which tiles to recompute: from its field input's change, or
//!    from its marks, which it takes. A node computed whole recomputes
//!    everything or nothing.
//! 4. It schedules those tiles, with the ones a budget left pending,
//!    within what is left of the run's budget. The rest stay pending.
//! 5. It computes the scheduled tiles over a copy of its kept output, or
//!    everything.
//! 6. It compares the result with its kept output, reports the work, and
//!    marks the dependents of the tiles that changed.
//! 7. It records its layout and the field version it consumed, and keeps
//!    the output, noting the node for another run if tiles are pending.
//!
//! Steps 3 and 4 spend state the kept output relies on: the marks, and the
//! pending tiles. A run that fails after them, while computing, must not
//! leave the node believing its kept output current, and it does not: the
//! layout taken in step 1 is restored only in step 7, so the next run
//! recomputes whole and compares with the kept output. Checks that fail
//! before `run_tiles` spend nothing.
//!
//! Sample nodes read rasters but output a field; they take their marks
//! only after everything that can fail. Field nodes change their state only
//! when they succeed.

use alloc::sync::Arc;
use alloc::vec::Vec;

use dapple_encode::Filter;
use dapple_field::PortType;
use dapple_field::program::{Change, Fingerprint};
use dapple_field::raster::Region;
use dapple_raster::{Raster, Realization, TexelRect};
use glam::Vec2;
use invalidation::intern::InternId;

use crate::tiles::{Grid, Tiles};
use crate::{FieldValue, GraphValue, NodeError, RasterData, RasterParams, RasterValue, Shape};

/// The version of `field` a consumer remembers: its fingerprint and serial.
pub(crate) fn version(field: &FieldValue) -> (Fingerprint, u64) {
    (field.program.fingerprint(), field.serial)
}

/// Where `field` can differ from the output a consumer last consumed, whose
/// version is `seen` (`None` for a consumer that has consumed none).
pub(crate) fn change_since(field: &FieldValue, seen: Option<(Fingerprint, u64)>) -> Change {
    match seen {
        Some(seen) if seen == version(field) => Change::Nowhere,
        Some((fingerprint, serial))
            if field.previous == Some(fingerprint) && field.serial == serial + 1 =>
        {
            field.change.clone()
        }
        _ => Change::Everywhere,
    }
}

/// Whether a field output with `fingerprint` and `change` repeats its
/// producer's last output, fingerprinted `previous`: then it is cut off and
/// keeps its producer's serial.
pub(crate) fn repeats(
    previous: Option<Fingerprint>,
    fingerprint: Fingerprint,
    change: &Change,
) -> bool {
    previous == Some(fingerprint) && *change == Change::Nowhere
}

/// What a tile node's output was computed with, besides the content of its
/// input: a run reuses tiles only when this and the grid are unchanged, and
/// rewires them otherwise.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Layout {
    /// A realization of a scalar field.
    Realize { width: u32, height: u32 },
    /// The normals of a scalar field; `scale` holds the height scale's bits.
    Normals { width: u32, height: u32, scale: u32 },
    /// A raster operation on the raster in tile space `input`.
    Raster {
        params: RasterParams,
        input: u32,
        /// The input's shape: kernels and the output's texels depend on its
        /// texel size, which can change while its grid stays (a mip of 24
        /// rows or of 25 has 12).
        shape: Shape,
    },
    /// The next mip level of the raster in tile space `input`.
    Mip {
        filter: Filter,
        input: u32,
        /// The source's shape: its size sets the taps of each tile and its
        /// texel size the level's, even when the level keeps its grid.
        source: Shape,
    },
    /// A raster computed whole, from a derivation fingerprint that names
    /// its input, the input's grid included.
    Whole { derivation: u64 },
}

/// State a node that outputs a raster keeps between runs.
#[derive(Clone, Debug, Default)]
pub(crate) struct TileState {
    /// What `output` was computed with; `None` while a run is under way,
    /// and after one failed.
    layout: Option<Layout>,
    /// The version of the field input `output` was computed from, for a
    /// node that reads a field.
    consumed: Option<(Fingerprint, u64)>,
    /// The previous output, shared with the graph's output value.
    output: Option<Arc<RasterValue>>,
    keys: Vec<InternId>,
    /// Tiles still to recompute against `layout`, left by a tile budget.
    pending: Vec<u32>,
}

/// How a tile node learns which tiles an input change reaches.
pub(crate) enum Reads<'a> {
    /// A field realized over `realization`: its change, applied through the
    /// serial protocol, says which tiles to recompute.
    Field {
        value: &'a FieldValue,
        realization: &'a Realization,
    },
    /// A field read whole: every run recomputes everything.
    WholeField,
    /// The raster in tile space `space`, tile by tile: output tile `t` reads
    /// the input tiles `inputs(t)` lists, and recomputes when one of them
    /// changed.
    Tiles {
        space: u32,
        inputs: &'a dyn Fn(u32) -> Vec<u32>,
    },
    /// The raster in tile space `space`, on `grid`, read whole: a change to
    /// any of its tiles recomputes everything.
    WholeRaster { space: u32, grid: Grid },
}

/// One run of a tile node: everything that differs between realize,
/// normals, raster, mip and reduce nodes, except how tiles are computed.
pub(crate) struct TileJob<'a> {
    /// What the output is computed with.
    pub(crate) layout: Layout,
    /// The output's grid.
    pub(crate) grid: Grid,
    /// How the node learns what changed.
    pub(crate) reads: Reads<'a>,
    /// The output's type.
    pub(crate) port: PortType,
    /// The output's fingerprint.
    pub(crate) fingerprint: u64,
}

/// What a run recomputes: the listed tiles over a copy of the kept output,
/// or, with `None`, everything.
pub(crate) type Work<'a> = Option<(&'a [u32], &'a RasterData)>;

/// Runs a tile node; see the [module docs](self).
///
/// `compute` computes the output: the tiles of its [`Work`] over a copy of
/// the kept output, or everything. It is not called when there is nothing
/// to recompute.
pub(crate) fn run_tiles(
    tiles: &mut Tiles,
    node: u32,
    state: &mut TileState,
    job: TileJob<'_>,
    compute: impl FnOnce(Work<'_>) -> Result<RasterData, NodeError>,
) -> Result<GraphValue, NodeError> {
    let TileJob {
        layout,
        grid,
        reads,
        port,
        fingerprint,
    } = job;
    let size = tiles.size();
    let same = state.layout.take().as_ref() == Some(&layout)
        && state
            .output
            .as_ref()
            .is_some_and(|out| out.data.grid(size) == grid);
    let count = grid.count();
    if !same || state.keys.len() != count as usize {
        state.keys = tiles.keys(node, grid, &state.keys);
        wire(tiles, &state.keys, &reads);
    }
    let dirty = match &reads {
        Reads::Field { value, realization } => field_dirty(
            tiles,
            grid,
            realization,
            value,
            state.consumed.filter(|_| same),
        ),
        Reads::WholeField => None,
        Reads::Tiles { .. } => {
            let marked = tiles.take_marked(&state.keys);
            same.then_some(marked)
        }
        Reads::WholeRaster { .. } => {
            let marked = tiles.take_marked(&state.keys);
            (same && marked.is_empty()).then(Vec::new)
        }
    };
    let dirty = tiles.schedule(&mut state.pending, dirty, count);
    #[cfg(test)]
    tiles.injected_failure(node)?;
    let previous = state.output.clone().filter(|_| same);
    let (data, recomputed) = match (dirty.as_deref(), previous.as_deref()) {
        (Some([]), Some(previous)) => (previous.data.clone(), Vec::new()),
        (Some(dirty), Some(previous)) => (compute(Some((dirty, &previous.data)))?, dirty.to_vec()),
        _ => (compute(None)?, (0..count).collect()),
    };
    debug_assert_eq!(data.grid(size), grid, "a node computes the grid it planned");
    let changed = settle(tiles, state, &data, &recomputed, dirty.is_none());
    state.layout = Some(layout);
    if let Reads::Field { value, .. } = reads {
        state.consumed = Some(version(value));
    }
    Ok(keep(
        tiles,
        node,
        state,
        RasterValue {
            data,
            port,
            fingerprint,
            changed,
            tile_space: node,
        },
    ))
}

/// Wires `keys`, a node's tiles in order, to the input tiles `reads` says
/// they read.
pub(crate) fn wire(tiles: &mut Tiles, keys: &[InternId], reads: &Reads<'_>) {
    match *reads {
        Reads::Field { .. } | Reads::WholeField => {}
        Reads::Tiles { space, inputs } => {
            for (t, &key) in keys.iter().enumerate() {
                let t = u32::try_from(t).expect("tile counts fit u32");
                let read: Vec<InternId> =
                    inputs(t).into_iter().map(|i| tiles.key(space, i)).collect();
                tiles.depend(key, read);
            }
        }
        Reads::WholeRaster { space, grid } => tiles.depend_whole(keys, space, grid),
    }
}

/// Which tiles of a node's grid to recompute: `None` for all of them.
type Dirty = Option<Vec<u32>>;

/// Tiles holding a texel whose center lies within `region` grown by `pad`,
/// or `None` when the region is not finite.
fn region_tiles(
    grid: Grid,
    realization: &Realization,
    region: &Region,
    pad: f32,
) -> Option<Vec<u32>> {
    let texel = realization.texel();
    let origin = realization.origin();
    let lo = (region.origin - Vec2::splat(pad) - origin) / texel - Vec2::splat(0.5);
    let hi = (region.origin + region.size + Vec2::splat(pad) - origin) / texel - Vec2::splat(0.5);
    if !(lo.is_finite() && hi.is_finite()) {
        return None;
    }
    let limit = f64::from(u32::MAX);
    let to_index = |v: f32| {
        let v = f64::from(v).clamp(-limit, limit);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "clamped to the u32 range before the cast"
        )]
        let i = v as i64;
        i
    };
    // Conservative by one texel on each side of the exact center test.
    let (x0, y0) = (
        to_index(libm::floorf(lo.x)) - 1,
        to_index(libm::floorf(lo.y)) - 1,
    );
    let (x1, y1) = (
        to_index(libm::ceilf(hi.x)) + 2,
        to_index(libm::ceilf(hi.y)) + 2,
    );
    Some(grid.overlapping(x0, x1, y0, y1))
}

/// The tiles of a realization of `field` to recompute, given the version of
/// the field the kept output was realized from (`None` when there is no
/// reusable output). Counts a recomputation an unbounded change forces over
/// a kept output.
///
/// A texel's value, or its gradient, reads the field within half a footprint
/// of its center, so the change regions grow by that much.
fn field_dirty(
    tiles: &mut Tiles,
    grid: Grid,
    realization: &Realization,
    field: &FieldValue,
    seen: Option<(Fingerprint, u64)>,
) -> Dirty {
    let seen = seen?;
    let dirty = match change_since(field, Some(seen)) {
        Change::Nowhere => Some(Vec::new()),
        Change::Within {
            regions,
            footprint_scale,
        } => {
            // A warp widens the footprint its input sees, so a region grows
            // by that many half texels.
            let pad = realization.texel().max_element() * 0.5 * footprint_scale;
            let mut dirty = Vec::new();
            let mut bounded = true;
            for region in &regions {
                match region_tiles(grid, realization, region, pad) {
                    Some(t) => dirty.extend(t),
                    None => bounded = false,
                }
            }
            dirty.sort_unstable();
            dirty.dedup();
            bounded.then_some(dirty)
        }
        Change::Everywhere => None,
    };
    if dirty.is_none() {
        tiles.report.unbounded_changes += 1;
    }
    dirty
}

/// Rewrites the tiles of `work` in a copy of its raster, or computes
/// everything with `whole`.
pub(crate) fn refresh<T: Copy>(
    grid: Grid,
    work: Option<(&[u32], &Raster<T>)>,
    whole: impl FnOnce() -> Result<Raster<T>, NodeError>,
    mut tile: impl FnMut(TexelRect, &mut Raster<T>) -> Result<(), NodeError>,
) -> Result<Raster<T>, NodeError> {
    match work {
        Some((dirty, previous)) => {
            let mut raster = previous.clone();
            for &t in dirty {
                tile(grid.rect(t), &mut raster)?;
            }
            Ok(raster)
        }
        None => whole(),
    }
}

/// Keeps `value` as the node's output, shared with the graph, and notes
/// tiles a budget left pending so the node runs again.
fn keep(tiles: &mut Tiles, node: u32, state: &mut TileState, value: RasterValue) -> GraphValue {
    if !state.pending.is_empty() {
        tiles.report.pending_tiles += state.pending.len() as u64;
        tiles.pending_nodes.push(node);
    }
    let value = Arc::new(value);
    state.output = Some(Arc::clone(&value));
    GraphValue::Raster(value)
}

/// Reports the work of a run that recomputed the tiles `recomputed` of
/// `output` (all of them, if `whole`), and marks the dependents of those
/// whose bits differ from the kept output. Returns whether any differ.
fn settle(
    tiles: &mut Tiles,
    state: &TileState,
    output: &RasterData,
    recomputed: &[u32],
    whole: bool,
) -> bool {
    let grid = output.grid(tiles.size());
    let previous = state
        .output
        .as_ref()
        .map(|v| &v.data)
        .filter(|previous| previous.same_grid(output));
    let changed: Vec<u32> = recomputed
        .iter()
        .copied()
        .filter(|&t| previous.is_none_or(|p| !p.rect_bits_eq(output, grid.rect(t))))
        .collect();
    let report = &mut tiles.report;
    report.tiles_recomputed += recomputed.len() as u64;
    report.tiles_reused += u64::from(grid.count()) - recomputed.len() as u64;
    report.tiles_changed += changed.len() as u64;
    report.texels_recomputed += recomputed.iter().map(|&t| grid.rect(t).area()).sum::<u64>();
    report.whole_recomputes += u64::from(whole);
    let keys: Vec<InternId> = changed.iter().map(|&t| state.keys[t as usize]).collect();
    tiles.mark_dependents(keys);
    !changed.is_empty()
}
