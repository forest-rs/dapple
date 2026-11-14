// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Batches of points evaluate to the bits single points do.

use alloc::vec;
use alloc::vec::Vec;

use glam::{Mat2, Mat3, Vec2, Vec3};

use super::{Op, ProgramBuilder, ValueProgram};
use crate::cellular::CellOutput;
use crate::domain::{Domain, Domain3, Footprint};
use crate::field::{Affine2, Affine3, ScalarField};
use crate::fractal::{FractalKind, FractalParams};
use crate::hash::{hash, unit_f32};
use crate::noise::Basis;
use crate::types::Value;

/// Whether two values are the same bits, NaN for NaN whatever its payload.
pub(super) fn same_bits(a: Value, b: Value) -> bool {
    let bits = |x: f32| if x.is_nan() { u32::MAX } else { x.to_bits() };
    match (a, b) {
        (Value::Id(a), Value::Id(b)) => a == b,
        (Value::Scalar(a), Value::Scalar(b)) => bits(a) == bits(b),
        (Value::Vector2(a), Value::Vector2(b)) => a.to_array().map(bits) == b.to_array().map(bits),
        (Value::Vector3(a), Value::Vector3(b)) => a.to_array().map(bits) == b.to_array().map(bits),
        _ => false,
    }
}

/// Programs that reach every op the flat plan evaluates: shared
/// subgraphs, transforms, nested warps (gradient instances), slices of
/// solid fields, constants and their arithmetic, and outputs of every
/// storage shape.
pub(super) fn zoo() -> Vec<ValueProgram> {
    let torus = Domain::periodic(2, 1).unwrap();
    let solid = Domain3::Space;
    let mut programs = Vec::new();
    let mut b = ProgramBuilder::new();
    let k = |b: &mut ProgramBuilder, value| {
        b.add(Op::Constant {
            domain: torus,
            value,
        })
        .unwrap()
    };
    let fbm = b
        .add(Op::Fractal {
            basis: Basis::Gradient,
            domain: torus,
            frequency: [2.0, 4.0],
            seed: 5,
            params: FractalParams::default(),
        })
        .unwrap();
    let ridged = b
        .add(Op::Fractal {
            basis: Basis::Value,
            domain: torus,
            frequency: [3.0, 1.0],
            seed: 6,
            params: FractalParams {
                kind: FractalKind::Ridged,
                octaves: 3,
                ..FractalParams::default()
            },
        })
        .unwrap();
    let noise = b
        .add(Op::Noise {
            basis: Basis::Value,
            domain: torus,
            frequency: [3.0, 3.0],
            seed: 9,
        })
        .unwrap();
    let cells = b
        .add(Op::Cellular {
            domain: torus,
            frequency: [4.0, 6.0],
            jitter: 0.9,
            seed: 3,
            output: CellOutput::Border,
        })
        .unwrap();
    let inner = b
        .add(Op::Warp {
            input: noise,
            dx: ridged,
            dy: fbm,
            amount: 0.03,
        })
        .unwrap();
    let doubled = b.add(Op::Mul { a: fbm, b: fbm }).unwrap();
    let warped = b
        .add(Op::Warp {
            input: cells,
            dx: inner,
            dy: doubled,
            amount: 0.05,
        })
        .unwrap();
    let shifted = b
        .add(Op::Transform {
            input: warped,
            transform: Affine2 {
                matrix: Mat2::from_cols(Vec2::new(1.0, 0.5), Vec2::new(0.0, 2.0)),
                translation: Vec2::new(0.25, 0.0),
            },
        })
        .unwrap();
    let ridges = b.add(Op::Abs { input: fbm }).unwrap();
    let depth = b
        .add(Op::Remap {
            input: ridges,
            from: [0.0, 1.0],
            to: [0.2, 1.0],
        })
        .unwrap();
    let grooved = b
        .add(Op::Min {
            a: shifted,
            b: depth,
        })
        .unwrap();
    // Arithmetic on constants alone, and constants shared by many nodes.
    let (half, three) = (k(&mut b, 0.5), k(&mut b, 3.0));
    let six = b.add(Op::Mul { a: three, b: three }).unwrap();
    let weight = b
        .add(Op::Remap {
            input: six,
            from: [0.0, 12.0],
            to: [0.0, 1.0],
        })
        .unwrap();
    let blend = b
        .add(Op::Mix {
            a: grooved,
            b: noise,
            t: weight,
        })
        .unwrap();
    let sum = b.add(Op::Add { a: blend, b: half }).unwrap();
    let diff = b.add(Op::Sub { a: sum, b: noise }).unwrap();
    let prod = b.add(Op::Mul { a: diff, b: half }).unwrap();
    let top = b
        .add(Op::Max {
            a: prod,
            b: grooved,
        })
        .unwrap();
    let height = b
        .add(Op::Clamp {
            input: top,
            min: 0.0,
            max: 1.0,
        })
        .unwrap();
    let mask = b.add(Op::AsMask { input: diff }).unwrap();
    programs.push(b.clone().finish_value(height).unwrap());
    programs.push(b.clone().finish_value(mask).unwrap());

    // Colors from constants and a ramp, an identifier, and directions.
    let color = |b: &mut ProgramBuilder, rgb: [f32; 3]| {
        let [r, g, bl] = rgb.map(|v| k(b, v));
        b.add(Op::Color { r, g, b: bl }).unwrap()
    };
    let dark = color(&mut b, [0.05, 0.03, 0.02]);
    let light = color(&mut b, [0.6, 0.5, 0.35]);
    let tinted = b.add(Op::Mul { a: light, b: half }).unwrap();
    let painted = b
        .add(Op::Mix {
            a: dark,
            b: tinted,
            t: height,
        })
        .unwrap();
    let red = b
        .add(Op::Component {
            input: painted,
            index: 0,
        })
        .unwrap();
    let id = b
        .add(Op::ToId {
            input: red,
            levels: 7,
        })
        .unwrap();
    let direction = b.add(Op::Direction { angle: diff }).unwrap();
    let across = b.add(Op::Direction { angle: fbm }).unwrap();
    let field = b
        .add(Op::Mix {
            a: direction,
            b: across,
            t: height,
        })
        .unwrap();
    let angle = b.add(Op::Angle { input: field }).unwrap();
    let coherence = b.add(Op::Coherence { input: field }).unwrap();
    let toward = b.add(Op::Vector2 { x: fbm, y: half }).unwrap();
    let length = b.add(Op::Length { input: toward }).unwrap();
    let both = b
        .add(Op::Atan2 {
            y: angle,
            x: coherence,
        })
        .unwrap();
    let wrapped = b.add(Op::Fract { input: both }).unwrap();
    let summary = b
        .add(Op::Mix {
            a: wrapped,
            b: length,
            t: half,
        })
        .unwrap();
    programs.push(b.clone().finish_value(painted).unwrap());
    programs.push(b.clone().finish_value(id).unwrap());
    programs.push(b.clone().finish_value(field).unwrap());
    programs.push(b.clone().finish_value(toward).unwrap());
    programs.push(b.clone().finish_value(summary).unwrap());

    // A solid wood-like field, sliced and transformed.
    let mut b = ProgramBuilder::new();
    let k3 = |b: &mut ProgramBuilder, value| {
        b.add(Op::Constant3 {
            domain: solid,
            value,
        })
        .unwrap()
    };
    let position = b.add(Op::Position3).unwrap();
    let radius = b.add(Op::Length { input: position }).unwrap();
    let rings = b
        .add(Op::Remap {
            input: radius,
            from: [0.0, 0.1],
            to: [0.0, 1.0],
        })
        .unwrap();
    let grain = b
        .add(Op::Fractal3 {
            basis: Basis::Gradient,
            domain: solid,
            frequency: [2.0, 2.0, 1.0],
            seed: 5,
            params: FractalParams::default(),
        })
        .unwrap();
    let pores = b
        .add(Op::Noise3 {
            basis: Basis::Value,
            domain: solid,
            frequency: [3.0, 2.0, 4.0],
            seed: 9,
        })
        .unwrap();
    let flecks = b
        .add(Op::Cellular3 {
            domain: solid,
            frequency: [2.0, 2.0, 4.0],
            jitter: 0.8,
            seed: 2,
            output: CellOutput::F1,
        })
        .unwrap();
    let scale = k3(&mut b, 0.2);
    let wobble = b.add(Op::Mul { a: grain, b: scale }).unwrap();
    let late = b
        .add(Op::Add {
            a: rings,
            b: wobble,
        })
        .unwrap();
    let ring = b.add(Op::Fract { input: late }).unwrap();
    let x = b
        .add(Op::Component {
            input: position,
            index: 0,
        })
        .unwrap();
    let y = b
        .add(Op::Component {
            input: position,
            index: 1,
        })
        .unwrap();
    let around = b.add(Op::Atan2 { y, x }).unwrap();
    let axis = b
        .add(Op::Vector3 {
            x: ring,
            y: pores,
            z: around,
        })
        .unwrap();
    let unit = b.add(Op::Normalize { input: axis }).unwrap();
    let z = b
        .add(Op::Component {
            input: unit,
            index: 2,
        })
        .unwrap();
    let wood = b
        .add(Op::Mix {
            a: ring,
            b: flecks,
            t: z,
        })
        .unwrap();
    let turned = b
        .add(Op::Transform3 {
            input: wood,
            transform: Affine3 {
                matrix: Mat3::from_cols(
                    Vec3::new(1.0, 0.2, 0.0),
                    Vec3::new(0.0, 1.0, 0.3),
                    Vec3::new(0.1, 0.0, 2.0),
                ),
                translation: Vec3::new(0.1, -0.3, 0.7),
            },
        })
        .unwrap();
    let board = b
        .add(Op::Slice {
            input: turned,
            origin: [0.14, 0.0, 0.3],
            u: [0.0, 0.0, 1.0],
            v: [0.0, 1.0, 0.0],
            domain: Domain::Plane,
        })
        .unwrap();
    programs.push(b.finish_value(board).unwrap());
    programs
}

/// Random points near and far on both signs, then signed zeros, cell and
/// period boundaries, NaN and huge coordinates.
pub(super) fn awkward_points() -> Vec<Vec2> {
    let mut points: Vec<Vec2> = (0..300_u64)
        .map(|i| {
            let h = |axis: u64| unit_f32(hash(0x726f_7773, &[i, axis])) * 2.0 - 1.0;
            let scale = [1.0, 4.0, 1e3, 1e6][(i % 4) as usize];
            Vec2::new(h(0), h(1)) * scale
        })
        .collect();
    for v in [0.0, -0.0, 0.5, 1.0, -1.0, 2.0, 1e-30, f32::NAN, 1e15, -1e15] {
        points.push(Vec2::splat(v));
        points.push(Vec2::new(v, 0.3));
        points.push(Vec2::new(-0.7, v));
    }
    points
}

pub(super) fn footprints() -> [Footprint; 7] {
    [0.0, 1e-6, 1.0 / 256.0, 0.02, 0.07, 0.3, 1e6].map(|w| Footprint::new(w).unwrap())
}

/// Lengths that split the points into runs of every awkward size: empty,
/// single, short, and longer than any inline buffer.
const RUNS: [usize; 6] = [0, 1, 3, 64, 0, 257];

#[test]
fn batches_evaluate_each_point_as_alone() {
    let points = awkward_points();
    for program in zoo() {
        let output = program.program().output();
        let mut kept = program.evaluator();
        for footprint in footprints() {
            // The same points in runs of every length, through one kept
            // evaluator, and all at once through the program's batch.
            let mut values = vec![Value::Id(u32::MAX); points.len()];
            let mut start = 0;
            for &len in RUNS.iter().cycle() {
                if start == points.len() {
                    break;
                }
                let end = (start + len).min(points.len());
                kept.eval_value_batch(&points[start..end], footprint, &mut values[start..end]);
                start = end;
            }
            let mut whole = vec![Value::Id(u32::MAX); points.len()];
            program.eval_batch(&points, footprint, &mut whole);
            for (i, &p) in points.iter().enumerate() {
                let alone = program.eval(p, footprint);
                let recursive = program.program().eval_value(output, p, footprint);
                assert!(same_bits(alone, recursive), "{p} {footprint:?}");
                assert!(same_bits(values[i], alone), "run at {p} {footprint:?}");
                assert!(same_bits(whole[i], alone), "batch at {p} {footprint:?}");
            }
        }
    }
}

#[test]
fn scalar_batches_evaluate_each_point_as_alone() {
    let points = awkward_points();
    for program in zoo() {
        if !program.output_type().is_scalar() {
            continue;
        }
        let field = program.program();
        let mut kept = field.evaluator();
        for footprint in footprints() {
            let mut values = vec![-7.0; points.len()];
            kept.eval_batch(&points, footprint, &mut values);
            let mut through = vec![-7.0; points.len()];
            ScalarField::eval_batch(field, &points, footprint, &mut through);
            for (i, &p) in points.iter().enumerate() {
                let alone = Value::Scalar(ScalarField::eval(field, p, footprint));
                assert!(same_bits(Value::Scalar(values[i]), alone), "{p}");
                assert!(same_bits(Value::Scalar(through[i]), alone), "{p}");
            }
        }
    }
}

#[test]
fn batches_leave_the_rest_of_the_output() {
    let program = &zoo()[2];
    let points = [Vec2::new(0.1, 0.2), Vec2::new(0.3, -0.4)];
    let mut out = [Value::Id(9); 4];
    program.eval_batch(&points, Footprint::POINT, &mut out);
    assert!(same_bits(out[1], program.eval(points[1], Footprint::POINT)));
    assert_eq!(out[2..], [Value::Id(9); 2]);
    program.eval_batch(&[], Footprint::POINT, &mut []);
}

#[test]
#[should_panic(expected = "output shorter than the points")]
fn batches_need_room_for_every_point() {
    let program = &zoo()[0];
    let mut out = [Value::Scalar(0.0); 1];
    program.eval_batch(&[Vec2::ZERO; 2], Footprint::POINT, &mut out);
}

#[test]
#[should_panic(expected = "the output is a scalar or mask")]
fn scalar_batches_need_a_scalar_output() {
    let program = &zoo()[2];
    let mut out = [0.0; 1];
    program
        .evaluator()
        .eval_batch(&[Vec2::ZERO], Footprint::POINT, &mut out);
}

#[test]
fn values_independent_of_the_point_are_computed_once() {
    let domain = Domain::periodic(1, 1).unwrap();
    let mut b = ProgramBuilder::new();
    let x = b
        .add(Op::Constant {
            domain,
            value: 0.25,
        })
        .unwrap();
    let y = b.add(Op::Constant { domain, value: 0.5 }).unwrap();
    let sum = b.add(Op::Add { a: x, b: y }).unwrap();
    let square = b.add(Op::Mul { a: sum, b: sum }).unwrap();
    let total = b.add(Op::Add { a: square, b: sum }).unwrap();
    let constant = b.clone().finish(total).unwrap();
    assert_eq!(constant.plan.steps(), 0, "nothing is left per point");
    assert_eq!(constant.evaluation_stats().instances, 5);
    let p = Vec2::new(0.3, 0.6);
    assert_eq!(
        constant.evaluator().eval(p, Footprint::POINT).to_bits(),
        constant.eval_node(total, p, Footprint::POINT).to_bits()
    );
    // Arithmetic reading the point stays, constants feeding it do not.
    let noise = b
        .add(Op::Noise {
            basis: Basis::Gradient,
            domain,
            frequency: [2.0, 2.0],
            seed: 1,
        })
        .unwrap();
    let scaled = b.add(Op::Mul { a: noise, b: total }).unwrap();
    assert_eq!(b.finish(scaled).unwrap().plan.steps(), 2);
}
