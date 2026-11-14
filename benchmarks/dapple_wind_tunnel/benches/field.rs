// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Point evaluation of fields and programs.
//!
//! Every benchmark evaluates the same 1024 points on a regular grid over the
//! unit period, at the footprint of one texel of a 256 texel raster, and
//! reports elements per second, so numbers are per sample. The `eval`
//! variants call the one-point entry points, which set up an evaluator per
//! call; the `evaluator` and `batch` variants share one across the points.
//! Solid fields take the same grid lifted onto a slanted plane through the
//! unit cube, so that every coordinate varies.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use dapple_field::program::{ChartSample, FieldProgram, ValueProgram};
use dapple_field::{
    Basis, CellOutput, Cellular, Cellular3, Footprint, Fractal, Fractal3, FractalParams, Noise,
    Noise3, ScalarField, SolidField, Value,
};
use dapple_wind_tunnel::{
    BOARD_OFFSET, deep_color, deep_scalar, grid_points, grid_points3, oak_board_color,
    oak_wood_red, period, small_color, small_scalar, solid_period, trivial_value,
};
use glam::{Vec2, Vec3};

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
        b.iter(|| sum_scalar(&cellular.output(CellOutput::F1), &points));
    });
    group.finish();
}

fn sum_solid(field: &impl SolidField, points: &[Vec3]) -> f32 {
    let footprint = footprint();
    points
        .iter()
        .map(|&p| field.eval(black_box(p), footprint))
        .sum()
}

fn bench_raw_solids(c: &mut Criterion) {
    let points = grid_points3(SIDE);
    let domain = solid_period();
    let freq = Vec3::splat(8.0);
    let gradient = Noise3::new(Basis::Gradient, domain, freq, 1).expect("valid noise");
    let value = Noise3::new(Basis::Value, domain, freq, 1).expect("valid noise");
    let fractal = Fractal3::new(Basis::Gradient, domain, freq, 2, FractalParams::default())
        .expect("valid fractal");
    let cellular = Cellular3::new(domain, freq, 1.0, 3)
        .expect("valid cellular")
        .output(CellOutput::F1);

    let mut group = c.benchmark_group("field/raw");
    group.throughput(Throughput::Elements(points.len() as u64));
    group.bench_function("noise3_gradient", |b| {
        b.iter(|| sum_solid(&gradient, &points));
    });
    group.bench_function("noise3_value", |b| b.iter(|| sum_solid(&value, &points)));
    group.bench_function("noise3_gradient_with_gradient", |b| {
        b.iter(|| {
            points
                .iter()
                .map(|&p| gradient.eval_gradient(black_box(p), footprint()).1.x)
                .sum::<f32>()
        });
    });
    group.bench_function("fractal3_5_octaves", |b| {
        b.iter(|| sum_solid(&fractal, &points));
    });
    group.bench_function("cellular3_f1", |b| b.iter(|| sum_solid(&cellular, &points)));
    group.finish();
}

/// The oak wood's red channel at chart samples on the wood module's board
/// face, the plane `x = BOARD_OFFSET` with the grid's x up the trunk.
fn bench_solid_program(c: &mut Criterion) {
    let program = oak_wood_red();
    let footprint = footprint();
    let samples: Vec<ChartSample> = grid_points(SIDE)
        .into_iter()
        .map(|p| ChartSample::isotropic(Vec3::new(BOARD_OFFSET, p.y, p.x), footprint))
        .collect();
    let mut group = c.benchmark_group("field/solid/oak_wood");
    group.throughput(Throughput::Elements(samples.len() as u64));
    group.bench_function("eval_chart", |b| {
        let mut out = vec![0.0; samples.len()];
        b.iter(|| {
            program.eval_chart(black_box(&samples), &mut out);
            out[0]
        });
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
    bench_raw_solids(c);
    bench_solid_program(c);
    let trivial = trivial_value();
    bench_scalar_program(c, "trivial", trivial.program());
    bench_value_program(c, "trivial", &trivial);
    bench_scalar_program(c, "small", &small_scalar());
    bench_scalar_program(c, "bark_height", &deep_scalar());
    bench_value_program(c, "small_color", &small_color());
    bench_value_program(c, "bark_color", &deep_color());
    bench_value_program(c, "oak_board", &oak_board_color());
}

criterion_group!(benches, bench_field);
criterion_main!(benches);
