// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Solid nodes, slices and chart evaluation in field programs.

use alloc::vec::Vec;

use glam::{Mat3, Vec2, Vec3};

use super::{ChartSample, Op, ProgramBuilder, ProgramError, Space};
use crate::cellular::CellOutput;
use crate::domain::{Domain, Domain3, DomainError, Footprint};
use crate::field::{Affine3, ScalarField};
use crate::fractal::FractalParams;
use crate::hash::{hash, unit_f32};
use crate::noise::Basis;
use crate::solid::{Fractal3, Noise3, SolidField, central_difference3};

fn points3(count: usize, scale: f32) -> Vec<Vec3> {
    (0..count)
        .map(|i| {
            let h = |axis: u64| unit_f32(hash(21, &[i as u64, axis])) * scale;
            Vec3::new(h(0), h(1), h(2))
        })
        .collect()
}

fn fbm3(b: &mut ProgramBuilder, domain: Domain3) -> super::NodeId {
    b.add(Op::Fractal3 {
        basis: Basis::Gradient,
        domain,
        frequency: [2.0, 2.0, 1.0],
        seed: 5,
        params: FractalParams::default(),
    })
    .unwrap()
}

#[test]
fn solid_programs_evaluate_like_the_direct_fields() {
    let mut b = ProgramBuilder::new();
    let noise = fbm3(&mut b, Domain3::Space);
    let program = b.finish_solid(noise).unwrap();
    let direct = Fractal3::new(
        Basis::Gradient,
        Domain3::Space,
        Vec3::new(2.0, 2.0, 1.0),
        5,
        FractalParams::default(),
    )
    .unwrap();
    let footprint = Footprint::new(0.01).unwrap();
    let samples: Vec<ChartSample> = points3(64, 4.0)
        .into_iter()
        .map(|position| ChartSample {
            position,
            footprint,
        })
        .collect();
    let mut chart = alloc::vec![0.0; samples.len()];
    program.eval_chart(&samples, &mut chart);
    for (sample, charted) in samples.iter().zip(&chart) {
        let p = sample.position;
        let value = program.eval(p, footprint);
        assert_eq!(value.to_bits(), direct.eval(p, footprint).to_bits());
        assert_eq!(value.to_bits(), charted.to_bits(), "chart at {p}");
        let (v, g) = program.eval_gradient(p, footprint);
        let (dv, dg) = direct.eval_gradient(p, footprint);
        assert_eq!(v.to_bits(), dv.to_bits());
        assert_eq!(g, dg);
    }
}

/// `noise3(transform(p)) * 0.5 + position.length() * 0.1`, sliced.
fn sliced(domain: Domain) -> Result<super::FieldProgram, ProgramError> {
    let mut b = ProgramBuilder::new();
    let noise = b.add(Op::Noise3 {
        basis: Basis::Gradient,
        domain: Domain3::Space,
        frequency: [3.0, 2.0, 4.0],
        seed: 9,
    })?;
    let turned = b.add(Op::Transform3 {
        input: noise,
        transform: Affine3 {
            matrix: Mat3::from_rotation_z(0.4),
            translation: Vec3::new(0.2, 0.0, -1.0),
        },
    })?;
    let half = b.add(Op::Constant3 {
        domain: Domain3::Space,
        value: 0.5,
    })?;
    let scaled = b.add(Op::Mul { a: turned, b: half })?;
    let position = b.add(Op::Position3)?;
    let radius = b.add(Op::Length { input: position })?;
    let tenth = b.add(Op::Constant3 {
        domain: Domain3::Space,
        value: 0.1,
    })?;
    let rings = b.add(Op::Mul {
        a: radius,
        b: tenth,
    })?;
    let sum = b.add(Op::Add {
        a: scaled,
        b: rings,
    })?;
    let slice = b.add(Op::Slice {
        input: sum,
        origin: [0.3, -0.2, 0.7],
        u: [0.8, 0.6, 0.0],
        v: [0.0, 0.0, 1.0],
        domain,
    })?;
    // Planar work downstream of the slice: the slice plus itself.
    let twice = b.add(Op::Add { a: slice, b: slice })?;
    b.finish(twice)
}

#[test]
fn slices_read_the_solid_along_their_plane() {
    let program = sliced(Domain::Plane).unwrap();
    let mut evaluator = program.evaluator();
    for p in points3(64, 3.0) {
        let q = Vec2::new(p.x, p.y);
        let recursive = program.eval(q, Footprint::POINT);
        let flat = evaluator.eval(q, Footprint::POINT);
        assert_eq!(recursive.to_bits(), flat.to_bits(), "flat plan at {q}");
        // The gradient follows the chain rule through the slice and transform.
        let (value, gradient) = program.eval_gradient(q, Footprint::POINT);
        assert_eq!(value.to_bits(), recursive.to_bits());
        let numeric = crate::central_difference(&program, q, Footprint::new(2e-3).unwrap());
        assert!(
            (gradient - numeric).length() < 2e-2 * gradient.length().max(1.0),
            "{q}: {gradient} vs {numeric}"
        );
    }
}

#[test]
fn periodic_slices_tile_exactly() {
    let domain3 = Domain3::periodic(1, 1, 2).unwrap();
    let build = |u: [f32; 3], v: [f32; 3], domain| {
        let mut b = ProgramBuilder::new();
        let noise = fbm3(&mut b, domain3);
        let slice = b.add(Op::Slice {
            input: noise,
            origin: [0.25, 0.5, 0.125],
            u,
            v,
            domain,
        })?;
        b.finish(slice)
    };
    let domain = Domain::periodic(1, 2).unwrap();
    // A diagonal slice: u·1 = (1, 1, 0) and v·2 = (0, 0, 2) are lattice vectors.
    let program = build([1.0, 1.0, 0.0], [0.0, 0.0, 1.0], domain).unwrap();
    for p in points3(64, 1.0) {
        let q = (Vec2::new(p.x, p.y) * 64.0).floor() / 64.0;
        let a = program.eval(q, Footprint::POINT);
        let b = program.eval(q + Vec2::new(3.0, -4.0), Footprint::POINT);
        assert_eq!(a.to_bits(), b.to_bits(), "repeat at {q}");
    }
    assert_eq!(
        build([0.5, 1.0, 0.0], [0.0, 0.0, 1.0], domain).err(),
        Some(ProgramError::Domain(DomainError::NotLatticePreserving))
    );
    assert!(build([0.5, 1.0, 0.0], [0.0, 0.0, 1.0], Domain::Plane).is_ok());
}

#[test]
fn spaces_are_checked() {
    let mut b = ProgramBuilder::new();
    let planar = b
        .add(Op::Constant {
            domain: Domain::Plane,
            value: 1.0,
        })
        .unwrap();
    let solid = b
        .add(Op::Constant3 {
            domain: Domain3::Space,
            value: 1.0,
        })
        .unwrap();
    assert_eq!(b.space(solid), Ok(Space::Solid(Domain3::Space)));
    assert!(matches!(
        b.add(Op::Add {
            a: planar,
            b: solid
        }),
        Err(ProgramError::SpaceMismatch { op: "add", .. })
    ));
    assert!(matches!(
        b.add(Op::Transform {
            input: solid,
            transform: crate::field::Affine2::IDENTITY,
        }),
        Err(ProgramError::SpaceMismatch { .. })
    ));
    assert!(matches!(
        b.add(Op::Slice {
            input: planar,
            origin: [0.0; 3],
            u: [1.0, 0.0, 0.0],
            v: [0.0, 1.0, 0.0],
            domain: Domain::Plane,
        }),
        Err(ProgramError::SpaceMismatch { op: "slice", .. })
    ));
    assert!(b.domain(solid).is_err());
    assert!(matches!(
        b.clone().finish(solid),
        Err(ProgramError::OutputSpace { .. })
    ));
    assert!(matches!(
        b.finish_solid(planar),
        Err(ProgramError::OutputSpace { .. })
    ));
    // Periodic solids refuse transforms that break their lattice.
    let mut b = ProgramBuilder::new();
    let noise = fbm3(&mut b, Domain3::periodic(1, 1, 1).unwrap());
    assert_eq!(
        b.add(Op::Transform3 {
            input: noise,
            transform: Affine3::scale(Vec3::new(1.5, 1.0, 1.0)),
        }),
        Err(ProgramError::Domain(DomainError::NotLatticePreserving))
    );
}

#[test]
fn solid_bounds_hold() {
    let footprints = [Footprint::POINT, Footprint::new(0.05).unwrap()];
    let mut b = ProgramBuilder::new();
    let mut nodes = Vec::new();
    for basis in [Basis::Value, Basis::Gradient] {
        nodes.push(
            b.add(Op::Noise3 {
                basis,
                domain: Domain3::Space,
                frequency: [2.0, 3.0, 1.0],
                seed: 3,
            })
            .unwrap(),
        );
    }
    nodes.push(fbm3(&mut b, Domain3::Space));
    for output in [CellOutput::F1, CellOutput::F2MinusF1, CellOutput::Border] {
        nodes.push(
            b.add(Op::Cellular3 {
                domain: Domain3::Space,
                frequency: [2.0; 3],
                jitter: 1.0,
                seed: 8,
                output,
            })
            .unwrap(),
        );
    }
    let turned = b
        .add(Op::Transform3 {
            input: nodes[1],
            transform: Affine3 {
                matrix: Mat3::from_rotation_x(0.7) * 2.0,
                translation: Vec3::ZERO,
            },
        })
        .unwrap();
    nodes.push(turned);
    let program = b.finish_solid(turned).unwrap();
    for &node in &nodes {
        let bounds = program.program().node_bounds(node);
        let [lo, hi] = bounds.range.expect("a range");
        let slope = bounds.slope.expect("a slope");
        for p in points3(512, 4.0) {
            for footprint in footprints {
                let (value, gradient) = program.program().gradient_at(node, p, footprint);
                let value = value.scalar().unwrap();
                assert!(lo <= value && value <= hi, "{node:?} value {value} at {p}");
                let l1 = gradient.x.abs() + gradient.y.abs() + gradient.z.abs();
                assert!(l1 <= slope, "{node:?} slope {l1} > {slope} at {p}");
            }
        }
    }
}

#[test]
fn solid_fingerprints_are_distinct_and_stable() {
    let fingerprint = |domain| {
        let mut b = ProgramBuilder::new();
        let noise = fbm3(&mut b, domain);
        b.finish_solid(noise).unwrap().fingerprint()
    };
    assert_eq!(fingerprint(Domain3::Space), fingerprint(Domain3::Space));
    assert_ne!(
        fingerprint(Domain3::Space),
        fingerprint(Domain3::periodic(1, 1, 1).unwrap())
    );
    let a = sliced(Domain::Plane).unwrap();
    assert_eq!(
        a.fingerprint(),
        sliced(Domain::Plane).unwrap().fingerprint()
    );
    // A central difference at the solid output agrees with its gradient.
    let mut b = ProgramBuilder::new();
    let noise = b
        .add(Op::Noise3 {
            basis: Basis::Gradient,
            domain: Domain3::Space,
            frequency: [2.0; 3],
            seed: 1,
        })
        .unwrap();
    let program = b.finish_solid(noise).unwrap();
    let direct = Noise3::new(Basis::Gradient, Domain3::Space, Vec3::splat(2.0), 1).unwrap();
    let p = Vec3::new(0.3, 0.7, 0.1);
    let numeric = central_difference3(&direct, p, Footprint::new(1e-3).unwrap());
    assert!((program.eval_gradient(p, Footprint::POINT).1 - numeric).length() < 1e-2);
}

#[test]
fn angles_around_an_axis() {
    use core::f32::consts::TAU;

    let angle = |sectors: Option<f32>| {
        let mut b = ProgramBuilder::new();
        let p = b.add(Op::Position3).unwrap();
        let x = b.add(Op::Component { input: p, index: 0 }).unwrap();
        let y = b.add(Op::Component { input: p, index: 1 }).unwrap();
        let mut out = b.add(Op::Atan2 { y, x }).unwrap();
        if let Some(n) = sectors {
            let turns = b
                .add(Op::Remap {
                    input: out,
                    from: [0.0, TAU],
                    to: [0.0, n],
                })
                .unwrap();
            out = b.add(Op::Fract { input: turns }).unwrap();
        }
        b.finish_solid(out).unwrap()
    };
    let program = angle(None);
    let bounds = program.bounds();
    for p in points3(64, 2.0) {
        let p = p - Vec3::ONE;
        let value = program.eval(p, Footprint::POINT);
        assert_eq!(value.to_bits(), libm::atan2f(p.y, p.x).to_bits());
        let [lo, hi] = bounds.range.unwrap();
        assert!((lo..=hi).contains(&value));
        // Away from the jump, the analytic gradient matches differences.
        if p.x > -0.1 || p.y.abs() > 0.1 {
            let numeric = central_difference3(&program, p, Footprint::new(1e-3).unwrap());
            let analytic = program.eval_gradient(p, Footprint::POINT).1;
            assert!(
                (analytic - numeric).length() < 1e-2 * (1.0 + numeric.length()),
                "{p} {analytic} {numeric}"
            );
        }
    }
    assert_eq!(program.eval(Vec3::ZERO, Footprint::POINT), 0.0);
    // Whole sectors make the sawtooth continuous across the jump.
    let sectors = angle(Some(8.0));
    let below = sectors.eval(Vec3::new(-1.0, -1e-4, 0.0), Footprint::POINT);
    let above = sectors.eval(Vec3::new(-1.0, 1e-4, 0.0), Footprint::POINT);
    let gap = (above - below).abs();
    assert!(gap.min(1.0 - gap) < 1e-3, "{below} {above}");
}
