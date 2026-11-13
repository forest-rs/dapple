// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! What a graph run reports to `footfall`, and that reporting changes nothing.
//!
//! The global footfall backend is installed once per process and never
//! removed, so this is its own test binary with one test: it runs a script of
//! edits on a graph with no backend, installs a runtime, runs the same script
//! on a second graph, and compares the two bit for bit before checking what
//! the runtime saw.

use std::collections::BTreeMap;

use dapple_field::program::Op;
use dapple_field::{Basis, Domain};
use dapple_graph::{MaterialGraph, RasterParams, TileReport, operand};
use dapple_raster::{GaussianBlur, HeightToNormal};
use execution_graph::{NodeId, RunSummary};
use footfall::runtime::{Observation, Runtime, Snapshot};

fn domain() -> Domain {
    Domain::periodic(1, 1).expect("a unit period is valid")
}

fn noise(seed: u64) -> Op {
    Op::Noise {
        basis: Basis::Gradient,
        domain: domain(),
        frequency: [8.0, 8.0],
        seed,
    }
}

fn disk(x: f32) -> Op {
    Op::Disk {
        domain: domain(),
        center: [x, 0.3],
        radius: 0.1,
        softness: 0.01,
    }
}

/// Field nodes and every raster-producing node kind the recipe uses.
struct Recipe {
    graph: MaterialGraph,
    noise: NodeId,
    disk: NodeId,
    nodes: Vec<NodeId>,
}

fn recipe() -> Recipe {
    let mut g = MaterialGraph::with_tile_size(16);
    let noise_ = g.field("noise", noise(1), &[]).expect("noise");
    let disk_ = g.field("disk", disk(0.2), &[]).expect("disk");
    let height = g
        .field(
            "height",
            Op::Max {
                a: operand(0),
                b: operand(1),
            },
            &[noise_, disk_],
        )
        .expect("height");
    let map = g.realize("map", height, 64, 64).expect("map");
    let soft = g
        .raster(
            "soft",
            RasterParams::Blur(GaussianBlur { sigma: 0.01 }),
            map,
        )
        .expect("soft");
    let relief = g
        .raster(
            "relief",
            RasterParams::HeightToNormal(HeightToNormal { scale: 0.05 }),
            soft,
        )
        .expect("relief");
    Recipe {
        graph: g,
        noise: noise_,
        disk: disk_,
        nodes: vec![noise_, disk_, height, map, soft, relief],
    }
}

/// Everything a run produces that a caller can see.
#[derive(Debug, PartialEq)]
struct Observed {
    summary: RunSummary,
    report: TileReport,
    /// Per node: the debug form of the raster (which tells `-0.0` from `0.0`
    /// and prints every bit of a float) and its fingerprint; or the field
    /// program's fingerprint.
    outputs: Vec<(String, u128)>,
}

fn observe(r: &Recipe, summary: RunSummary) -> Observed {
    let outputs = r
        .nodes
        .iter()
        .map(|&node| match r.graph.raster_value(node) {
            Some(raster) => (format!("{raster:?}"), u128::from(raster.fingerprint)),
            None => (
                String::new(),
                r.graph
                    .field_value(node)
                    .expect("a node that is not a raster is a field")
                    .fingerprint()
                    .0,
            ),
        })
        .collect();
    Observed {
        summary,
        report: r.graph.tile_report(),
        outputs,
    }
}

/// A first run, a local edit, a global edit, a clean run, and a budgeted
/// edit run until it converges.
fn script() -> Vec<Observed> {
    let mut r = recipe();
    let mut seen = Vec::new();
    let run = |r: &mut Recipe, seen: &mut Vec<Observed>| {
        let summary = r.graph.run().expect("runs");
        seen.push(observe(r, summary));
    };
    run(&mut r, &mut seen);
    let disk_node = r.disk;
    r.graph.set_field_op(disk_node, disk(0.7)).expect("edit");
    run(&mut r, &mut seen);
    let noise_node = r.noise;
    r.graph.set_field_op(noise_node, noise(2)).expect("edit");
    run(&mut r, &mut seen);
    run(&mut r, &mut seen);
    r.graph.set_tile_budget(Some(3));
    r.graph.set_field_op(disk_node, disk(0.2)).expect("edit");
    for _ in 0..200 {
        run(&mut r, &mut seen);
        if seen.last().expect("a run").report.pending_tiles == 0 {
            break;
        }
    }
    assert_eq!(
        seen.last().expect("a run").report.pending_tiles,
        0,
        "the budgeted edit converges"
    );
    seen
}

fn scope_name_count(snapshot: &Snapshot, wanted: &str) -> usize {
    snapshot
        .events()
        .iter()
        .filter(|event| {
            matches!(event.observation(), Observation::ScopeBegin { name, .. } if *name == wanted)
        })
        .count()
}

fn counter(snapshot: &Snapshot, name: &str) -> u64 {
    snapshot
        .counters()
        .iter()
        .filter(|row| row.name() == name)
        .map(|row| row.value())
        .sum()
}

#[test]
fn capture_reports_the_run_and_changes_nothing() {
    // No backend yet: the baseline every captured result must equal.
    let plain = script();

    let runtime = Runtime::builder()
        .process_name("dapple_graph_test")
        .install_global()
        .expect("no other backend is installed in this process");
    let captured = script();

    // (b) Capture changes no raster, fingerprint, report or summary.
    assert_eq!(plain.len(), captured.len(), "same number of runs");
    for (step, (plain, captured)) in plain.iter().zip(&captured).enumerate() {
        assert_eq!(plain, captured, "run {step} differs under capture");
    }

    // (a) The snapshot holds the scopes and counters, with sensible values.
    let snapshot = runtime.snapshot();
    let drops = snapshot.drops();
    assert_eq!(drops.evicted_events(), 0, "the window kept every event");
    assert_eq!(drops.counter_updates(), 0, "no counter update was lost");

    let runs = captured.len();
    let attempts: usize = captured.iter().map(|o| o.summary.execution_attempts).sum();
    assert_eq!(scope_name_count(&snapshot, "dapple_graph.run"), runs);
    assert_eq!(scope_name_count(&snapshot, "dapple_graph.node"), attempts);
    assert_eq!(
        attempts,
        captured
            .iter()
            .map(|o| o.summary.executed_nodes)
            .sum::<usize>(),
        "no node suspends or fails in this script, so every attempt publishes"
    );

    let total = captured
        .iter()
        .fold(TileReport::default(), |t, o| TileReport {
            tiles_recomputed: t.tiles_recomputed + o.report.tiles_recomputed,
            tiles_reused: t.tiles_reused + o.report.tiles_reused,
            tiles_changed: t.tiles_changed + o.report.tiles_changed,
            texels_recomputed: t.texels_recomputed + o.report.texels_recomputed,
            whole_recomputes: t.whole_recomputes + o.report.whole_recomputes,
            unbounded_changes: t.unbounded_changes + o.report.unbounded_changes,
            unbounded_warps: t.unbounded_warps + o.report.unbounded_warps,
            pending_tiles: t.pending_tiles + o.report.pending_tiles,
        });
    for (name, expected) in [
        ("dapple_graph.tiles_recomputed", total.tiles_recomputed),
        ("dapple_graph.tiles_reused", total.tiles_reused),
        ("dapple_graph.tiles_changed", total.tiles_changed),
        ("dapple_graph.texels_recomputed", total.texels_recomputed),
        ("dapple_graph.whole_recomputes", total.whole_recomputes),
        ("dapple_graph.unbounded_changes", total.unbounded_changes),
        ("dapple_graph.unbounded_warps", total.unbounded_warps),
        ("dapple_graph.pending_tiles", total.pending_tiles),
    ] {
        assert_eq!(counter(&snapshot, name), expected, "{name}");
    }
    // Not vacuous: the script recomputes, reuses and leaves tiles pending.
    assert!(total.tiles_recomputed > 0 && total.tiles_reused > 0);
    assert!(total.pending_tiles > 0 && total.whole_recomputes > 0);

    // Node scopes nest in run scopes, and name their kind and key.
    let mut open = BTreeMap::new();
    let mut in_run = 0_usize;
    let mut kinds = BTreeMap::new();
    for event in snapshot.events() {
        match event.observation() {
            Observation::ScopeBegin {
                name, token, args, ..
            } => {
                open.insert(token.get(), *name);
                match *name {
                    "dapple_graph.run" => in_run += 1,
                    "dapple_graph.node" => {
                        assert_eq!(in_run, 1, "a node scope opens inside a run scope");
                        let kind = args.get("kind").and_then(|v| v.as_static_str());
                        let key = args.get("key").and_then(|v| v.as_unsigned());
                        *kinds
                            .entry((kind.expect("a kind"), key.expect("a key")))
                            .or_insert(0_usize) += 1;
                    }
                    other => panic!("unexpected scope {other}"),
                }
            }
            Observation::ScopeEnd { token } => {
                let closed = open.remove(&token.get());
                in_run -= usize::from(closed == Some("dapple_graph.run"));
            }
            _ => {}
        }
    }
    assert!(open.is_empty(), "every scope closed");
    // The first run executes each of the six nodes once.
    let kind_names: Vec<_> = kinds.keys().map(|(kind, _)| *kind).collect();
    for kind in ["field", "realize", "raster"] {
        assert!(kind_names.contains(&kind), "{kind} nodes ran");
    }
    assert_eq!(kinds.keys().map(|(_, key)| *key).max(), Some(5));
}
