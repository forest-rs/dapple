// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Realizing fields as rasters.
//!
//! Sizes are 256 and 1024 texels a side, reported as texels per second.
//! `small` programs are cheap enough that the realization loop itself shows;
//! `bark` is the library's oak bark, a realistic program, at 256 only to keep
//! the suite short.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dapple_field::program::{FieldProgram, ValueProgram};
use dapple_raster::typed::realize_value;
use dapple_raster::{HeightToNormal, RasterOp, Realization, realize, realize_normals};
use dapple_wind_tunnel::{deep_color, deep_scalar, period, small_color, small_scalar};

const SIZES: [u32; 2] = [256, 1024];

fn realization(size: u32) -> Realization {
    Realization::period(period(), size, size).expect("a valid realization")
}

fn texels(size: u32) -> Throughput {
    Throughput::Elements(u64::from(size) * u64::from(size))
}

fn scalar(c: &mut Criterion, name: &str, program: &FieldProgram, sizes: &[u32]) {
    let mut group = c.benchmark_group(format!("raster/realize_scalar/{name}"));
    group.sample_size(10);
    for &size in sizes {
        group.throughput(texels(size));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| realize(program, realization(size)).expect("realizes"));
        });
    }
    group.finish();
}

fn typed(c: &mut Criterion, name: &str, program: &ValueProgram, sizes: &[u32]) {
    let mut group = c.benchmark_group(format!("raster/realize_value/{name}"));
    group.sample_size(10);
    for &size in sizes {
        group.throughput(texels(size));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| realize_value(program, realization(size)).expect("realizes"));
        });
    }
    group.finish();
}

fn normals(c: &mut Criterion, name: &str, program: &FieldProgram, sizes: &[u32]) {
    let mut group = c.benchmark_group(format!("raster/realize_normals/{name}"));
    group.sample_size(10);
    for &size in sizes {
        group.throughput(texels(size));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| realize_normals(program, realization(size), 0.05).expect("realizes"));
        });
    }
    group.finish();
}

/// Normals from an already realized height raster by central differences,
/// the raster-side counterpart of `realize_normals`.
fn height_to_normal(c: &mut Criterion) {
    let mut group = c.benchmark_group("raster/height_to_normal");
    group.sample_size(10);
    let program = small_scalar();
    for size in SIZES {
        let height = realize(&program, realization(size)).expect("realizes");
        let op = HeightToNormal { scale: 0.05 };
        group.throughput(texels(size));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| op.apply(&height).expect("applies"));
        });
    }
    group.finish();
}

fn bench_raster(c: &mut Criterion) {
    scalar(c, "small", &small_scalar(), &SIZES);
    scalar(c, "bark", &deep_scalar(), &[256]);
    typed(c, "small_color", &small_color(), &SIZES);
    typed(c, "bark_color", &deep_color(), &[256]);
    normals(c, "small", &small_scalar(), &SIZES);
    normals(c, "bark", &deep_scalar(), &[256]);
    height_to_normal(c);
}

criterion_group!(benches, bench_raster);
criterion_main!(benches);
