// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Building and importing programs.
//!
//! Each benchmark runs over chains of 256, 1024 and 4096 dependent nodes, so
//! the shape of the scaling is visible: time per node should stay flat. Every
//! node of a chain has its own parameters, so no two share a fingerprint and
//! `ProgramBuilder::import` can deduplicate none of them within one program.

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dapple_field::program::ProgramBuilder;
use dapple_wind_tunnel::{chain, chain_builder};

const LENGTHS: [usize; 3] = [256, 1024, 4096];

fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("build");
    for len in LENGTHS {
        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::new("add_chain", len), &len, |b, &len| {
            b.iter(|| chain_builder(black_box(len)));
        });
        let (builder, last) = chain_builder(len);
        group.bench_with_input(BenchmarkId::new("finish_chain", len), &len, |b, _| {
            b.iter_batched(
                || builder.clone(),
                |builder| builder.finish(last).expect("a scalar output"),
                BatchSize::LargeInput,
            );
        });
        let program = chain(len);
        // Every node is new to the builder: the fingerprint of each is
        // compared with those already imported.
        group.bench_with_input(BenchmarkId::new("import_into_empty", len), &len, |b, _| {
            b.iter_batched(
                ProgramBuilder::new,
                |mut builder| builder.import(black_box(&program)).expect("imports"),
                BatchSize::LargeInput,
            );
        });
        // Every node is already there: all are found, and none is copied.
        group.bench_with_input(BenchmarkId::new("import_again", len), &len, |b, _| {
            b.iter_batched(
                || {
                    let mut builder = ProgramBuilder::new();
                    builder.import(&program).expect("imports");
                    builder
                },
                |mut builder| builder.import(black_box(&program)).expect("imports"),
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_build);
criterion_main!(benches);
