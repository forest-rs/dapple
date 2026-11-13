// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Material graph runs and edits.
//!
//! The recipe is a height map: gradient noise and a disk combined by `max`,
//! realized over one period in 32-texel tiles, blurred, and turned into
//! normals. Sizes are 256 and 512 texels a side (64 and 256 tiles), so the
//! cost of an edit can be compared with the size of the raster.
//!
//! - `fresh_run`: a first run of a freshly built graph, nothing cached.
//! - `clean_run`: a run with nothing edited.
//! - `edit_local_disk`: move the disk, an edit that states its change region;
//!   only the tiles it reaches should recompute.
//! - `edit_global_seed`: reseed the noise, which changes every texel.
//! - `edit_blur_sigma`: change the blur, which re-runs the blur and the
//!   normals but not the realization.
//! - `budgeted_slice`: the first run after moving the disk under a budget of
//!   four tiles, the unit of work a frame-budgeted caller pays per run.
//! - `budgeted_converge`: the same edit, run repeatedly until the budget
//!   leaves nothing pending.
//!
//! - `field_chain_fresh`: a first run of a chain of 8, 32 or 128 dependent
//!   field nodes with no realization, so what is timed is each node building
//!   its program from its upstream programs.
//! - `field_chain_edit`: reseed the head of that chain and run again, so every
//!   node downstream rebuilds its program once.
//!
//! Each edit benchmark alternates between two values so that every
//! iteration is a real edit, and times the edit and the run together.

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use dapple_field::program::Op;
use dapple_field::{Basis, Domain};
use dapple_graph::{MaterialGraph, RasterParams, operand};
use dapple_raster::{GaussianBlur, HeightToNormal};
use dapple_wind_tunnel::period;
use execution_graph::NodeId;

const SIZES: [u32; 2] = [256, 512];
const TILE: u32 = 32;

/// The recipe's graph and the nodes edits reach.
struct Recipe {
    graph: MaterialGraph,
    noise: NodeId,
    disk: NodeId,
    soft: NodeId,
}

fn disk(domain: Domain, x: f32) -> Op {
    Op::Disk {
        domain,
        center: [x, 0.3],
        radius: 0.05,
        softness: 0.01,
    }
}

fn noise(domain: Domain, seed: u64) -> Op {
    Op::Noise {
        basis: Basis::Gradient,
        domain,
        frequency: [8.0, 8.0],
        seed,
    }
}

fn blur(sigma: f32) -> RasterParams {
    RasterParams::Blur(GaussianBlur { sigma })
}

fn recipe(size: u32) -> Recipe {
    let domain = period();
    let mut g = MaterialGraph::with_tile_size(TILE);
    let noise_node = g.field("noise", noise(domain, 1), &[]).expect("noise");
    let disk_node = g.field("disk", disk(domain, 0.2), &[]).expect("disk");
    let height = g
        .field(
            "height",
            Op::Max {
                a: operand(0),
                b: operand(1),
            },
            &[noise_node, disk_node],
        )
        .expect("height");
    let map = g.realize("map", height, size, size).expect("map");
    let soft = g.raster("soft", blur(0.01), map).expect("soft");
    g.raster(
        "normals",
        RasterParams::HeightToNormal(HeightToNormal { scale: 0.05 }),
        soft,
    )
    .expect("normals");
    Recipe {
        graph: g,
        noise: noise_node,
        disk: disk_node,
        soft,
    }
}

/// A recipe that has run once, so edits start from a cached result.
fn settled(size: u32) -> Recipe {
    let mut r = recipe(size);
    r.graph.run().expect("first run");
    r
}

/// Moving the disk must recompute some tiles and reuse more than it
/// recomputes (the report counts the tiles of the realize, blur and normals
/// nodes together), or the benchmark would not measure a localized edit.
fn check_local(size: u32) {
    let mut r = settled(size);
    r.graph
        .set_field_op(r.disk, disk(period(), 0.6))
        .expect("edit");
    r.graph.run().expect("run");
    let report = r.graph.tile_report();
    assert!(
        report.tiles_recomputed > 0 && report.tiles_recomputed < report.tiles_reused,
        "a disk edit at {size} is no longer local: {report:?}"
    );
}

fn bench_graph(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph");
    group.sample_size(10);
    let domain = period();
    for size in SIZES {
        check_local(size);
        group.bench_with_input(BenchmarkId::new("fresh_run", size), &size, |b, &size| {
            b.iter_batched(
                || recipe(size),
                |mut r| r.graph.run().expect("runs"),
                BatchSize::LargeInput,
            );
        });
        group.bench_with_input(
            BenchmarkId::new("edit_local_disk", size),
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
            BenchmarkId::new("edit_global_seed", size),
            &size,
            |b, &size| {
                let mut r = settled(size);
                let mut seed = 1;
                b.iter(|| {
                    seed = 3 - seed;
                    r.graph
                        .set_field_op(r.noise, noise(domain, seed))
                        .expect("edit");
                    r.graph.run().expect("runs")
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("edit_blur_sigma", size),
            &size,
            |b, &size| {
                let mut r = settled(size);
                let mut wide = false;
                b.iter(|| {
                    wide = !wide;
                    let sigma = if wide { 0.02 } else { 0.01 };
                    r.graph
                        .set_raster_params(r.soft, blur(sigma))
                        .expect("edit");
                    r.graph.run().expect("runs")
                });
            },
        );
        group.bench_with_input(BenchmarkId::new("clean_run", size), &size, |b, &size| {
            let mut r = settled(size);
            b.iter(|| r.graph.run().expect("runs"));
        });
        group.bench_with_input(
            BenchmarkId::new("budgeted_slice", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || budgeted_edit(size),
                    |mut r| r.graph.run().expect("runs"),
                    BatchSize::LargeInput,
                );
            },
        );
        group.bench_with_input(
            BenchmarkId::new("budgeted_converge", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || budgeted_edit(size),
                    |mut r| {
                        let mut runs = 0_u32;
                        loop {
                            r.graph.run().expect("runs");
                            runs += 1;
                            if r.graph.tile_report().pending_tiles == 0 {
                                break runs;
                            }
                            assert!(runs < 1000, "budgeted runs must converge");
                        }
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

/// Chain lengths for the field chain benchmarks.
const CHAINS: [usize; 3] = [8, 32, 128];

/// A noise node and `len - 1` remaps after it, each with its own parameters,
/// returning the graph and the head.
fn field_chain(len: usize) -> (MaterialGraph, NodeId) {
    let mut g = MaterialGraph::with_tile_size(TILE);
    let head = g.field("head", noise(period(), 1), &[]).expect("head");
    let mut last = head;
    for i in 1..len {
        last = g
            .field(
                &format!("remap{i}"),
                Op::Remap {
                    input: operand(0),
                    from: [0.0, 1.0 + i as f32],
                    to: [0.0, 1.0],
                },
                &[last],
            )
            .expect("remap");
    }
    (g, head)
}

fn bench_field_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph");
    group.sample_size(10);
    for len in CHAINS {
        group.bench_with_input(
            BenchmarkId::new("field_chain_fresh", len),
            &len,
            |b, &len| {
                b.iter_batched(
                    || field_chain(len).0,
                    |mut g| g.run().expect("runs"),
                    BatchSize::LargeInput,
                );
            },
        );
        group.bench_with_input(
            BenchmarkId::new("field_chain_edit", len),
            &len,
            |b, &len| {
                let (mut g, head) = field_chain(len);
                g.run().expect("first run");
                let mut seed = 1;
                b.iter(|| {
                    seed = 3 - seed;
                    g.set_field_op(head, noise(period(), seed)).expect("edit");
                    g.run().expect("runs")
                });
            },
        );
    }
    group.finish();
}

/// A settled recipe with a four-tile budget and the disk moved, before the
/// run that pays for the edit.
fn budgeted_edit(size: u32) -> Recipe {
    let mut r = settled(size);
    r.graph.set_tile_budget(Some(4));
    r.graph
        .set_field_op(r.disk, disk(period(), 0.6))
        .expect("edit");
    r
}

criterion_group!(benches, bench_graph, bench_field_chain);
criterion_main!(benches);
