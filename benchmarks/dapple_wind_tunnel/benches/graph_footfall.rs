// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The cost of capturing a graph run with a `footfall` runtime installed.
//!
//! These are the `graph` benchmark's `fresh_run`, `edit_local_disk` and
//! `clean_run` with the same recipe and sizes, under new names, in a bench
//! target of their own: the global backend is installed once per process and
//! never removed, so it cannot share a process with the uninstrumented cases.
//! Compare each `_captured` case with its namesake in `graph` from the same
//! sitting; the difference is what capturing costs, and the `graph` cases
//! alone (no backend) show what the instrumentation costs when nobody is
//! listening.
//!
//! The runtime is the default one at its most verbose level, never reset, so
//! its event window is full and every scope also evicts the oldest event: the
//! steady state of a long capture.

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use dapple_field::program::Op;
use dapple_field::{Basis, Domain};
use dapple_graph::{MaterialGraph, RasterParams, operand};
use dapple_raster::{GaussianBlur, HeightToNormal};
use dapple_wind_tunnel::period;
use execution_graph::NodeId;
use footfall::runtime::Runtime;

const SIZES: [u32; 2] = [256, 512];
const TILE: u32 = 32;

struct Recipe {
    graph: MaterialGraph,
    disk: NodeId,
}

fn disk(domain: Domain, x: f32) -> Op {
    Op::Disk {
        domain,
        center: [x, 0.3],
        radius: 0.05,
        softness: 0.01,
    }
}

/// The `graph` benchmark's recipe: noise and a disk combined by `max`,
/// realized, blurred and turned into normals.
fn recipe(size: u32) -> Recipe {
    let domain = period();
    let mut g = MaterialGraph::with_tile_size(TILE);
    let noise = g
        .field(
            "noise",
            Op::Noise {
                basis: Basis::Gradient,
                domain,
                frequency: [8.0, 8.0],
                seed: 1,
            },
            &[],
        )
        .expect("noise");
    let disk_node = g.field("disk", disk(domain, 0.2), &[]).expect("disk");
    let height = g
        .field(
            "height",
            Op::Max {
                a: operand(0),
                b: operand(1),
            },
            &[noise, disk_node],
        )
        .expect("height");
    let map = g.realize("map", height, size, size).expect("map");
    let soft = g
        .raster(
            "soft",
            RasterParams::Blur(GaussianBlur { sigma: 0.01 }),
            map,
        )
        .expect("soft");
    g.raster(
        "normals",
        RasterParams::HeightToNormal(HeightToNormal { scale: 0.05 }),
        soft,
    )
    .expect("normals");
    Recipe {
        graph: g,
        disk: disk_node,
    }
}

fn settled(size: u32) -> Recipe {
    let mut r = recipe(size);
    r.graph.run().expect("first run");
    r
}

fn bench_graph_captured(c: &mut Criterion) {
    let runtime = Runtime::builder()
        .process_name("dapple_wind_tunnel")
        .install_global()
        .expect("this bench target installs the only backend");
    let mut group = c.benchmark_group("graph");
    group.sample_size(10);
    let domain = period();
    for size in SIZES {
        group.bench_with_input(
            BenchmarkId::new("fresh_run_captured", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || recipe(size),
                    |mut r| r.graph.run().expect("runs"),
                    BatchSize::LargeInput,
                );
            },
        );
        group.bench_with_input(
            BenchmarkId::new("edit_local_disk_captured", size),
            &size,
            |b, &size| {
                let mut r = settled(size);
                let mut far = false;
                b.iter(|| {
                    far = !far;
                    let x = if far { 0.6 } else { 0.2 };
                    r.graph.set_field_op(r.disk, disk(domain, x)).expect("edit");
                    r.graph.run().expect("runs")
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("clean_run_captured", size),
            &size,
            |b, &size| {
                let mut r = settled(size);
                b.iter(|| r.graph.run().expect("runs"));
            },
        );
    }
    group.finish();
    // The capture saw the work: a benchmark of a runtime that recorded
    // nothing would measure the wrong thing.
    assert!(
        runtime
            .snapshot()
            .counters()
            .iter()
            .any(|row| row.name() == "dapple_graph.tiles_recomputed" && row.value() > 0),
        "the runtime captured the graph's counters"
    );
}

criterion_group!(benches, bench_graph_captured);
criterion_main!(benches);
