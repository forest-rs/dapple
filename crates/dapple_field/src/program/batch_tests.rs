// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Batches of points evaluate to the bits single points do.

use alloc::vec;
use alloc::vec::Vec;

use glam::{Mat2, Mat3, Vec2, Vec3};

use super::block::LANES;
use super::{ChartSample, Fingerprint, NodeId, Op, ProgramBuilder, SolidProgram, ValueProgram};
use crate::cellular::CellOutput;
use crate::domain::{Domain, Domain3, Footprint};
use crate::field::{Affine2, Affine3, ScalarField};
use crate::fractal::{FractalKind, FractalParams};
use crate::hash::{hash, unit_f32};
use crate::image::{ImageLevel, SampleImage, SamplePolicy};
use crate::noise::Basis;
use crate::scatter::{Placement, ScatterOutput, Stamp};
use crate::solid::SolidField;
use crate::tiling::{Pattern, TileOutput};
use crate::types::{PortType, Primaries, Value};

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
    // The fractals themselves, without their gradients.
    let crests = b.add(Op::Mul { a: ridged, b: fbm }).unwrap();
    let crests = b.clone().finish_value(crests).unwrap();

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
    programs.push(crests);
    programs.extend(leaves());
    programs
}

/// A value in `[-1, 1)` keyed by `keys`.
fn signed(keys: &[u64]) -> f32 {
    unit_f32(hash(0x6c65_6166, keys)) * 2.0 - 1.0
}

/// An 8 × 4 image of `port` over the plane's `[0, 2] × [0, 1]`, with a
/// 4 × 2 mip, so footprints choose and blend levels.
fn image(port: PortType, policy: SamplePolicy, seed: u64) -> SampleImage {
    let channels = u8::try_from(port.components()).unwrap();
    let level = |width: u32, height: u32, key: u64| {
        let texel = Vec2::splat(2.0 / width as f32);
        let texels = u64::from(width * height);
        if port == PortType::Id {
            let ids: Vec<u32> = (0..texels)
                .map(|i| (hash(seed, &[key, i]) % 1000) as u32)
                .collect();
            ImageLevel::ids(width, height, texel, ids).unwrap()
        } else {
            let count = texels * u64::from(channels);
            let values: Vec<f32> = (0..count).map(|i| signed(&[seed, key, i])).collect();
            ImageLevel::with_channels(width, height, texel, channels, values).unwrap()
        }
    };
    let levels = vec![level(8, 4, 0), level(4, 2, 1)];
    SampleImage::typed(
        port,
        policy,
        Domain::Plane,
        Vec2::ZERO,
        levels,
        Fingerprint(0),
    )
    .unwrap()
}

/// Programs on the plane reaching every leaf kernel, and every gradient
/// rule and chain step, through warps whose displacements use them: leaf
/// gradients evaluated with their values, arithmetic gradients in closed
/// form, others (images, scatters, components, lengths) by central
/// differences, and chains through transforms, solid transforms, slices
/// and warps.
fn leaves() -> Vec<ValueProgram> {
    let plane = Domain::Plane;
    let mut b = ProgramBuilder::new();
    let mut add = |op: Op| b.add(op).unwrap();
    let k = add(Op::Constant {
        domain: plane,
        value: 3.0,
    });
    let bond = add(Op::Tiling {
        domain: plane,
        pattern: Pattern::Bond {
            frequency: [4.0, 8.0],
            shift: 0.5,
        },
        seed: 6,
        output: TileOutput::Edge,
    });
    let herringbone = add(Op::Tiling {
        domain: plane,
        pattern: Pattern::Herringbone {
            frequency: 4.0,
            ratio: 2,
        },
        seed: 7,
        output: TileOutput::U,
    });
    let placement = Placement {
        frequency: 6.0,
        density: 0.7,
        radius: [0.3, 0.9],
        rotate: true,
    };
    let pebbles = add(Op::Scatter {
        domain: plane,
        placement,
        stamp: Stamp::Dome,
        seed: 4,
        output: ScatterOutput::Max,
    });
    let cover = add(Op::Scatter {
        domain: plane,
        placement,
        stamp: Stamp::Disk { softness: 0.2 },
        seed: 5,
        output: ScatterOutput::Coverage,
    });
    let disk = add(Op::Disk {
        domain: plane,
        center: [0.5, 0.25],
        radius: 0.4,
        softness: 0.1,
    });
    let cells = add(Op::Cellular {
        domain: plane,
        frequency: [3.0, 5.0],
        jitter: 0.7,
        seed: 8,
        output: CellOutput::F2MinusF1,
    });
    let noise = add(Op::Noise {
        basis: Basis::Gradient,
        domain: plane,
        frequency: [2.0, 3.0],
        seed: 1,
    });
    let fbm = add(Op::Fractal {
        basis: Basis::Value,
        domain: plane,
        frequency: [1.5, 2.5],
        seed: 2,
        params: FractalParams::default(),
    });
    let sampled = add(Op::Sample {
        image: image(PortType::Scalar, SamplePolicy::Linear, 1),
    });
    let nearest = add(Op::Sample {
        image: image(PortType::Scalar, SamplePolicy::Nearest, 2),
    });
    let color = PortType::Color(Primaries::Rec709);
    let paint = add(Op::Sample {
        image: image(color, SamplePolicy::Linear, 3),
    });
    let labels = add(Op::Sample {
        image: image(PortType::Id, SamplePolicy::Nearest, 4),
    });
    let grain = add(Op::Sample {
        image: image(PortType::Direction, SamplePolicy::Linear, 5),
    });
    // Arithmetic of every kind on the leaves.
    let a = add(Op::Mul {
        a: bond,
        b: pebbles,
    });
    let a = add(Op::Sub { a, b: disk });
    let a = add(Op::Add { a, b: sampled });
    let a = add(Op::Min { a, b: cover });
    let a = add(Op::Max { a, b: herringbone });
    let e = add(Op::Abs { input: a });
    let e = add(Op::Clamp {
        input: e,
        min: 0.1,
        max: 0.9,
    });
    let e = add(Op::Remap {
        input: e,
        from: [0.1, 0.9],
        to: [-0.5, 2.0],
    });
    let m = add(Op::AsMask { input: e });
    let tripled = add(Op::Mul { a: e, b: k });
    let f = add(Op::Fract { input: tripled });
    let t = add(Op::Atan2 { y: m, x: f });
    let t = add(Op::Mix { a: t, b: a, t: m });
    // Gradients by central differences: of a component, a length, an
    // image and a cellular field.
    let green = add(Op::Component {
        input: paint,
        index: 1,
    });
    let toward = add(Op::Vector2 { x: f, y: cells });
    let length = add(Op::Length { input: toward });
    let g = add(Op::Mul {
        a: green,
        b: length,
    });
    let g = add(Op::Add { a: g, b: nearest });
    let dx = add(Op::Add { a: t, b: g });
    // Chains: through a transform, and a slice of a turned solid field.
    let moved = add(Op::Transform {
        input: noise,
        transform: Affine2 {
            matrix: Mat2::from_cols(Vec2::new(0.8, 0.6), Vec2::new(-0.6, 0.8)),
            translation: Vec2::new(0.3, -0.1),
        },
    });
    let solid = Domain3::Space;
    let wood = add(Op::Fractal3 {
        basis: Basis::Gradient,
        domain: solid,
        frequency: [2.0, 1.0, 3.0],
        seed: 3,
        params: FractalParams {
            octaves: 3,
            ..FractalParams::default()
        },
    });
    let position = add(Op::Position3);
    let height = add(Op::Component {
        input: position,
        index: 2,
    });
    let wood = add(Op::Add { a: wood, b: height });
    let turned = add(Op::Transform3 {
        input: wood,
        transform: Affine3 {
            matrix: Mat3::from_cols(
                Vec3::new(1.0, 0.1, 0.0),
                Vec3::new(0.0, 0.9, 0.2),
                Vec3::new(0.3, 0.0, 1.1),
            ),
            translation: Vec3::new(0.5, 0.0, -0.25),
        },
    });
    let board = add(Op::Slice {
        input: turned,
        origin: [0.2, -0.4, 0.1],
        u: [0.6, 0.0, 0.8],
        v: [0.0, 1.0, 0.0],
        domain: plane,
    });
    let dy = add(Op::Add { a: moved, b: board });
    // A warp of a warp: its gradient chains through the inner warp.
    let inner = add(Op::Warp {
        input: fbm,
        dx,
        dy,
        amount: 0.07,
    });
    let warp = |add: &mut dyn FnMut(Op) -> NodeId, input: NodeId| {
        add(Op::Warp {
            input,
            dx: inner,
            dy,
            amount: 0.05,
        })
    };
    let warped_paint = warp(&mut add, paint);
    let warped_labels = warp(&mut add, labels);
    let warped_grain = warp(&mut add, grain);
    let angle = add(Op::Angle {
        input: warped_grain,
    });
    let coherence = add(Op::Coherence {
        input: warped_grain,
    });
    let spread = add(Op::Mul {
        a: angle,
        b: coherence,
    });
    // Per-point footprints from the warp, through a transform and a slice.
    let turned_cells = add(Op::Transform {
        input: cells,
        transform: Affine2 {
            matrix: Mat2::from_cols(Vec2::new(0.0, 1.5), Vec2::new(-1.5, 0.0)),
            translation: Vec2::new(0.1, 0.2),
        },
    });
    let layered = add(Op::Add {
        a: turned_cells,
        b: board,
    });
    let warped = warp(&mut add, layered);
    let scalar = add(Op::Mix {
        a: warped,
        b: spread,
        t: m,
    });
    let blend = add(Op::Mix {
        a: warped_paint,
        b: paint,
        t: m,
    });
    let tinted = add(Op::Mul { a: blend, b: e });
    [scalar, tinted, warped_labels, warped_grain, dx, dy]
        .map(|output| b.clone().finish_value(output).unwrap())
        .into()
}

/// Solid programs, for chart evaluation.
fn solids() -> Vec<SolidProgram> {
    let mut b = ProgramBuilder::new();
    let space = Domain3::Space;
    let mut add = |op: Op| b.add(op).unwrap();
    let grain = add(Op::Fractal3 {
        basis: Basis::Gradient,
        domain: space,
        frequency: [2.0, 2.0, 1.0],
        seed: 5,
        params: FractalParams::default(),
    });
    let pores = add(Op::Noise3 {
        basis: Basis::Value,
        domain: space,
        frequency: [3.0, 2.0, 4.0],
        seed: 9,
    });
    let flecks = add(Op::Cellular3 {
        domain: space,
        frequency: [2.0, 2.0, 4.0],
        jitter: 0.8,
        seed: 2,
        output: CellOutput::F1,
    });
    let position = add(Op::Position3);
    let radius = add(Op::Length { input: position });
    let rings = add(Op::Add {
        a: radius,
        b: grain,
    });
    let rings = add(Op::Fract { input: rings });
    let wood = add(Op::Mix {
        a: rings,
        b: flecks,
        t: pores,
    });
    let turned = add(Op::Transform3 {
        input: wood,
        transform: Affine3 {
            matrix: Mat3::from_cols(
                Vec3::new(1.0, 0.2, 0.0),
                Vec3::new(0.0, 1.0, 0.3),
                Vec3::new(0.1, 0.0, 2.0),
            ),
            translation: Vec3::new(0.1, -0.3, 0.7),
        },
    });
    let periodic = Domain3::periodic(1, 1, 2).unwrap();
    let tiled = add(Op::Fractal3 {
        basis: Basis::Value,
        domain: periodic,
        frequency: [2.0, 3.0, 1.0],
        seed: 7,
        params: FractalParams {
            kind: FractalKind::Ridged,
            octaves: 4,
            ..FractalParams::default()
        },
    });
    let mut programs: Vec<SolidProgram> = [grain, wood, turned]
        .map(|output| b.clone().finish_solid(output).unwrap())
        .into();
    programs.push(b.finish_solid(tiled).unwrap());
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
/// single, short, around a block and its multiples, and longer than any
/// inline buffer.
const RUNS: [usize; 11] = [
    0,
    1,
    3,
    LANES - 1,
    LANES,
    LANES + 1,
    0,
    2 * LANES,
    2 * LANES + 3,
    257,
    5 * LANES - 1,
];

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
            for run in runs(points.len()) {
                kept.eval_value_batch(&points[run.clone()], footprint, &mut values[run]);
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
            for run in runs(points.len()) {
                kept.eval_batch(&points[run.clone()], footprint, &mut values[run]);
            }
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

/// Runs of `items` cut by [`RUNS`], cycled.
fn runs(len: usize) -> impl Iterator<Item = core::ops::Range<usize>> {
    let mut start = 0;
    RUNS.iter().cycle().map_while(move |&run| {
        (start < len).then(|| {
            let end = (start + run).min(len);
            let range = start..end;
            start = end;
            range
        })
    })
}

#[test]
fn charts_evaluate_each_sample_as_alone() {
    let points: Vec<Vec3> = awkward_points()
        .iter()
        .enumerate()
        .map(|(i, p)| p.extend(signed(&[i as u64, 9]) * [1.0, 7.0, 1e3][i % 3]))
        .collect();
    let widths = footprints();
    for program in solids() {
        let mut alone = program.program().evaluator();
        // Every sample with one footprint, each with its own, and all with
        // one but a sample in each block.
        for case in 0..widths.len() + 2 {
            let footprint = |i: usize| match case {
                case if case < widths.len() => widths[case],
                case if case == widths.len() => widths[i % widths.len()],
                _ if i % LANES == 17 => widths[3],
                _ => widths[2],
            };
            let samples: Vec<ChartSample> = points
                .iter()
                .enumerate()
                .map(|(i, &p)| ChartSample::isotropic(p, footprint(i)))
                .collect();
            let mut values = vec![-7.0; samples.len()];
            for run in runs(samples.len()) {
                program.eval_chart(&samples[run.clone()], &mut values[run]);
            }
            let mut whole = vec![-7.0; samples.len()];
            program.eval_chart(&samples, &mut whole);
            for (i, sample) in samples.iter().enumerate() {
                let (p, footprint) = (sample.position, sample.footprint);
                let point = alone.eval_value_at(p, footprint);
                let recursive = Value::Scalar(SolidField::eval(&program, p, footprint));
                assert!(same_bits(point, recursive), "{p} {footprint:?}");
                assert!(
                    same_bits(Value::Scalar(values[i]), point),
                    "run at {p} {footprint:?}"
                );
                assert!(
                    same_bits(Value::Scalar(whole[i]), point),
                    "chart at {p} {footprint:?}"
                );
            }
        }
    }
}

/// Planar programs of pointwise arithmetic alone, on the point's
/// coordinates through a slice of the solid position: defined at every
/// point, infinities included.
fn pointwise() -> Vec<ValueProgram> {
    let mut b = ProgramBuilder::new();
    let mut add = |op: Op| b.add(op).unwrap();
    let position = add(Op::Position3);
    let [x, y, z] = [0, 1, 2].map(|index| {
        add(Op::Component {
            input: position,
            index,
        })
    });
    let k = add(Op::Constant3 {
        domain: Domain3::Space,
        value: 0.75,
    });
    let sum = add(Op::Add { a: x, b: y });
    let product = add(Op::Mul { a: sum, b: k });
    let low = add(Op::Min { a: product, b: y });
    let high = add(Op::Max { a: low, b: z });
    let diff = add(Op::Sub { a: high, b: x });
    let abs = add(Op::Abs { input: diff });
    let clamped = add(Op::Clamp {
        input: abs,
        min: -1.0,
        max: 4.0,
    });
    let remapped = add(Op::Remap {
        input: clamped,
        from: [0.0, 4.0],
        to: [1.0, -1.0],
    });
    let fract = add(Op::Fract { input: product });
    let atan = add(Op::Atan2 { y: x, x: y });
    let mask = add(Op::AsMask { input: remapped });
    let mixed = add(Op::Mix {
        a: atan,
        b: fract,
        t: mask,
    });
    let id = add(Op::ToId {
        input: mixed,
        levels: 9,
    });
    let length = add(Op::Length { input: position });
    let vector = add(Op::Vector3 {
        x: mixed,
        y: length,
        z: x,
    });
    let unit = add(Op::Normalize { input: vector });
    let unit_z = add(Op::Component {
        input: unit,
        index: 2,
    });
    let tint = add(Op::Color {
        r: mixed,
        g: unit_z,
        b: mask,
    });
    let scaled = add(Op::Mul { a: tint, b: mask });
    let direction = add(Op::Direction { angle: diff });
    let pair = add(Op::Vector2 { x: atan, y });
    let blended = add(Op::Mix {
        a: direction,
        b: direction,
        t: mask,
    });
    let angle = add(Op::Angle { input: blended });
    let coherence = add(Op::Coherence { input: blended });
    let flat = add(Op::Length { input: pair });
    let summary = add(Op::Add {
        a: angle,
        b: coherence,
    });
    let summary = add(Op::Add {
        a: summary,
        b: flat,
    });
    // An identifier of constants alone, computed when the plan is built.
    let half = add(Op::Constant3 {
        domain: Domain3::Space,
        value: 0.5,
    });
    let fixed = add(Op::ToId {
        input: half,
        levels: 9,
    });
    let outputs = [mixed, id, scaled, summary, unit_z, fixed];
    outputs
        .map(|input| {
            let mut b = b.clone();
            let slice = b
                .add(Op::Slice {
                    input,
                    origin: [0.5, -0.25, 2.0],
                    u: [1.0, 0.0, 0.0],
                    v: [0.0, 0.6, 0.8],
                    domain: Domain::Plane,
                })
                .unwrap();
            b.finish_value(slice).unwrap()
        })
        .into()
}

#[test]
fn pointwise_batches_evaluate_each_point_as_alone_even_at_infinity() {
    let mut points = awkward_points();
    for v in [f32::INFINITY, f32::NEG_INFINITY, f32::MAX, -f32::MAX] {
        points.push(Vec2::splat(v));
        points.push(Vec2::new(v, 0.3));
        points.push(Vec2::new(-0.7, v));
        points.push(Vec2::new(v, -v));
    }
    for program in pointwise() {
        let output = program.program().output();
        let mut kept = program.evaluator();
        let mut values = vec![Value::Id(u32::MAX); points.len()];
        for run in runs(points.len()) {
            kept.eval_value_batch(&points[run.clone()], Footprint::POINT, &mut values[run]);
        }
        for (i, &p) in points.iter().enumerate() {
            let alone = program.eval(p, Footprint::POINT);
            let recursive = program.program().eval_value(output, p, Footprint::POINT);
            assert!(same_bits(alone, recursive), "{p}");
            assert!(same_bits(values[i], alone), "run at {p}");
        }
    }
}
