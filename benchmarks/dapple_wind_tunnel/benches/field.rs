// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Point evaluation of fields and programs.
//!
//! Every benchmark evaluates the same 1024 points on a regular grid over the
//! unit period, at the footprint of one texel of a 256 texel raster, and
//! reports elements per second, so numbers are per sample. The `eval`
//! variants call the one-point entry points, which set up an evaluator per
//! call; the `evaluator` and `batch` variants share one across the points.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use dapple_field::program::{FieldProgram, ValueProgram};
use dapple_field::{Basis, Cellular, Footprint, Fractal, FractalParams, Noise, ScalarField, Value};
use dapple_wind_tunnel::{
    deep_color, deep_scalar, grid_points, period, small_color, small_scalar, trivial_value,
};
use glam::Vec2;

const SIDE: u32 = 32;

fn footprint() -> Footprint {
    Footprint::new(1.0 / 256.0).expect("a finite footprint")
}

fn sum_scalar(field: &impl ScalarField, points: &[Vec2]) -> f32 {
    let footprint = footprint();
    points
        .iter()
        .map(|&p| field.eval(black_box(p), footprint))
        .sum()
}

fn bench_raw_fields(c: &mut Criterion) {
    let points = grid_points(SIDE);
    let domain = period();
    let freq = Vec2::splat(8.0);
    let noise = Noise::new(Basis::Gradient, domain, freq, 1).expect("valid noise");
    let fractal = Fractal::new(Basis::Gradient, domain, freq, 2, FractalParams::default())
        .expect("valid fractal");
    let cellular = Cellular::new(domain, freq, 1.0, 3).expect("valid cellular");

    let mut group = c.benchmark_group("field/raw");
    group.throughput(Throughput::Elements(points.len() as u64));
    group.bench_function("noise_gradient", |b| b.iter(|| sum_scalar(&noise, &points)));
    group.bench_function("fractal_5_octaves", |b| {
        b.iter(|| sum_scalar(&fractal, &points));
    });
    group.bench_function("cellular_f1", |b| {
        b.iter(|| sum_scalar(&cellular.output(dapple_field::CellOutput::F1), &points));
    });
    group.finish();
}

fn bench_scalar_program(c: &mut Criterion, name: &str, program: &FieldProgram) {
    let points = grid_points(SIDE);
    let footprint = footprint();
    let mut group = c.benchmark_group(format!("field/program/{name}"));
    group.throughput(Throughput::Elements(points.len() as u64));
    group.bench_function("eval", |b| b.iter(|| sum_scalar(program, &points)));
    group.bench_function("evaluator", |b| {
        b.iter(|| {
            let mut evaluator = program.evaluator();
            points
                .iter()
                .map(|&p| evaluator.eval(black_box(p), footprint))
                .sum::<f32>()
        });
    });
    group.bench_function("batch", |b| {
        let mut out = vec![0.0; points.len()];
        b.iter(|| {
            program.eval_batch(black_box(&points), footprint, &mut out);
            out[0]
        });
    });
    group.bench_function("gradient", |b| {
        b.iter(|| {
            points
                .iter()
                .map(|&p| program.eval_gradient(black_box(p), footprint).0)
                .sum::<f32>()
        });
    });
    group.finish();
}

fn bench_value_program(c: &mut Criterion, name: &str, program: &ValueProgram) {
    let points = grid_points(SIDE);
    let footprint = footprint();
    let red = |v: Value| v.component(0).unwrap_or(0.0);
    let mut group = c.benchmark_group(format!("field/value_program/{name}"));
    group.throughput(Throughput::Elements(points.len() as u64));
    group.bench_function("eval", |b| {
        b.iter(|| {
            points
                .iter()
                .map(|&p| red(program.eval(black_box(p), footprint)))
                .sum::<f32>()
        });
    });
    group.bench_function("evaluator", |b| {
        b.iter(|| {
            let mut evaluator = program.evaluator();
            points
                .iter()
                .map(|&p| red(evaluator.eval_value(black_box(p), footprint)))
                .sum::<f32>()
        });
    });
    group.finish();
}

fn bench_field(c: &mut Criterion) {
    bench_raw_fields(c);
    let trivial = trivial_value();
    bench_scalar_program(c, "trivial", trivial.program());
    bench_value_program(c, "trivial", &trivial);
    bench_scalar_program(c, "small", &small_scalar());
    bench_scalar_program(c, "bark_height", &deep_scalar());
    bench_value_program(c, "small_color", &small_color());
    bench_value_program(c, "bark_color", &deep_color());
}

criterion_group!(benches, bench_field);
criterion_main!(benches);
