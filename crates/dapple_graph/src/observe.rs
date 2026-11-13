// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! What a graph run tells [`footfall`], and nothing else.
//!
//! The descriptors live here so that the names, which tooling reads, are in
//! one place; the crate docs list them. Everything is emitted under the root
//! context. Nothing here is read back: the scopes and counters only report,
//! and the deterministic [`TileReport`] stays the source of truth for tests
//! and for any decision.
//!
//! The grain is the run and the node. Counters are added once per run, from
//! the finished report; nothing is emitted per tile, texel or sample.

use footfall::{Args, ContextHandle, CounterDesc, ScopeDesc, ScopeGuard};

use crate::{NodeKind, TileReport};

static RUN: ScopeDesc = ScopeDesc::new("dapple_graph.run");
static NODE: ScopeDesc = ScopeDesc::new("dapple_graph.node");

static TILES_RECOMPUTED: CounterDesc = CounterDesc::count("dapple_graph.tiles_recomputed");
static TILES_REUSED: CounterDesc = CounterDesc::count("dapple_graph.tiles_reused");
static TILES_CHANGED: CounterDesc = CounterDesc::count("dapple_graph.tiles_changed");
static TEXELS_RECOMPUTED: CounterDesc = CounterDesc::count("dapple_graph.texels_recomputed");
static WHOLE_RECOMPUTES: CounterDesc = CounterDesc::count("dapple_graph.whole_recomputes");
static UNBOUNDED_CHANGES: CounterDesc = CounterDesc::count("dapple_graph.unbounded_changes");
static UNBOUNDED_WARPS: CounterDesc = CounterDesc::count("dapple_graph.unbounded_warps");
static PENDING_TILES: CounterDesc = CounterDesc::count("dapple_graph.pending_tiles");

/// Opens the scope of one [`MaterialGraph::run`](crate::MaterialGraph::run).
pub(crate) fn run() -> ScopeGuard {
    footfall::scope(&RUN, ContextHandle::ROOT)
}

/// Opens the scope of one node execution, naming its kind and key.
pub(crate) fn node(kind: NodeKind, key: u32) -> ScopeGuard {
    // Build the arguments only when a backend will keep them.
    if footfall::max_level().captures_scopes() {
        let args = Args::none().with("kind", kind_name(kind)).with("key", key);
        footfall::scope_with(&NODE, ContextHandle::ROOT, &args)
    } else {
        footfall::scope(&NODE, ContextHandle::ROOT)
    }
}

/// Adds the finished report of a run to the counters, one add per field.
pub(crate) fn report(report: &TileReport) {
    // One level check for all eight, so a run that nobody listens to pays for
    // one, not eight.
    if !footfall::max_level().captures_counters() {
        return;
    }
    // Destructured so that a new field cannot be forgotten here.
    let TileReport {
        tiles_recomputed,
        tiles_reused,
        tiles_changed,
        texels_recomputed,
        whole_recomputes,
        unbounded_changes,
        unbounded_warps,
        pending_tiles,
    } = *report;
    for (desc, value) in [
        (&TILES_RECOMPUTED, tiles_recomputed),
        (&TILES_REUSED, tiles_reused),
        (&TILES_CHANGED, tiles_changed),
        (&TEXELS_RECOMPUTED, texels_recomputed),
        (&WHOLE_RECOMPUTES, whole_recomputes),
        (&UNBOUNDED_CHANGES, unbounded_changes),
        (&UNBOUNDED_WARPS, unbounded_warps),
        (&PENDING_TILES, pending_tiles),
    ] {
        footfall::counter_add(desc, ContextHandle::ROOT, value);
    }
}

const fn kind_name(kind: NodeKind) -> &'static str {
    match kind {
        NodeKind::Field => "field",
        NodeKind::Realize => "realize",
        NodeKind::Raster => "raster",
        NodeKind::Normals => "normals",
        NodeKind::Mip => "mip",
        NodeKind::Reduce => "reduce",
        NodeKind::Sample => "sample",
    }
}
