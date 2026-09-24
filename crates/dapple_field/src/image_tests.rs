// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sampled images and `Op::Sample`.

use alloc::vec::Vec;

use glam::Vec2;

use crate::program::{Fingerprint, Op, ProgramBuilder, sample_fingerprint};
use crate::{
    Basis, Domain, Footprint, ImageLevel, Noise, NormalFrame, PortType, Primaries, SampleImage,
    SamplePolicy, ScalarField, Value,
};

fn torus() -> Domain {
    Domain::periodic(1, 1).unwrap()
}

fn noise() -> Noise {
    Noise::new(Basis::Gradient, torus(), Vec2::splat(4.0), 11).unwrap()
}

/// `field` at the texel centers of a `size` × `size` grid over one period.
fn texels(field: &impl ScalarField, size: u32) -> Vec<f32> {
    #[expect(clippy::cast_precision_loss, reason = "tiny test sizes")]
    let texel = 1.0 / size as f32;
    let mut values = Vec::new();
    for y in 0..size {
        for x in 0..size {
            #[expect(clippy::cast_precision_loss, reason = "tiny test sizes")]
            let p = Vec2::new((x as f32 + 0.5) * texel, (y as f32 + 0.5) * texel);
            values.push(field.eval(p, Footprint::POINT));
        }
    }
    values
}

fn level(size: u32, values: Vec<f32>) -> ImageLevel {
    #[expect(clippy::cast_precision_loss, reason = "tiny test sizes")]
    let texel = Vec2::splat(1.0 / size as f32);
    ImageLevel::new(size, size, texel, values).unwrap()
}

fn image(levels: Vec<ImageLevel>, derivation: u128) -> SampleImage {
    SampleImage::new(torus(), Vec2::ZERO, levels, Fingerprint(derivation)).unwrap()
}

/// A 2×2 box-filtered level below a square level.
fn halve(size: u32, values: &[f32]) -> Vec<f32> {
    let half = size / 2;
    let at = |x: u32, y: u32| values[(y * size + x) as usize];
    let mut out = Vec::new();
    for y in 0..half {
        for x in 0..half {
            let (x0, y0) = (2 * x, 2 * y);
            out.push((at(x0, y0) + at(x0 + 1, y0) + at(x0, y0 + 1) + at(x0 + 1, y0 + 1)) * 0.25);
        }
    }
    out
}

#[test]
fn texel_centers_reproduce_the_realized_values() {
    let field = noise();
    let values = texels(&field, 32);
    let sampled = image(alloc::vec![level(32, values)], 1);
    for y in 0..32_u32 {
        for x in 0..32_u32 {
            #[expect(clippy::cast_precision_loss, reason = "tiny test sizes")]
            let p = Vec2::new((x as f32 + 0.5) / 32.0, (y as f32 + 0.5) / 32.0);
            assert_eq!(
                sampled.sample(p, Footprint::POINT).to_bits(),
                field.eval(p, Footprint::POINT).to_bits(),
            );
        }
    }
}

#[test]
fn periodic_images_repeat_exactly_and_wrap_at_the_seam() {
    let values: Vec<f32> = (0..16)
        .map(|i| f32::from(u8::try_from(i).unwrap()))
        .collect();
    let sampled = image(alloc::vec![level(4, values.clone())], 2);
    // Dyadic points, so each shifted point is exactly a repeat in f32.
    for p in [
        Vec2::new(0.109_375, 0.3125),
        Vec2::new(0.9375, 0.953_125),
        Vec2::new(0.0, 0.5),
    ] {
        let base = sampled.sample(p, Footprint::POINT).to_bits();
        for shift in [Vec2::new(1.0, 0.0), Vec2::new(-3.0, 2.0)] {
            assert_eq!(sampled.sample(p + shift, Footprint::POINT).to_bits(), base);
        }
    }
    // Halfway between the last and first texel of row 0.
    let seam = sampled.sample(Vec2::new(0.0, 0.125), Footprint::POINT);
    assert_eq!(seam, (values[3] + values[0]) * 0.5);
}

#[test]
fn gradients_match_central_differences() {
    let sampled = image(alloc::vec![level(32, texels(&noise(), 32))], 3);
    for p in [Vec2::new(0.21, 0.37), Vec2::new(0.77, 0.05)] {
        let (value, gradient) = sampled.sample_gradient(p, Footprint::POINT);
        assert_eq!(
            value.to_bits(),
            sampled.sample(p, Footprint::POINT).to_bits()
        );
        let h = 1e-4;
        let at = |q: Vec2| sampled.sample(q, Footprint::POINT);
        let numeric = Vec2::new(
            (at(p + Vec2::new(h, 0.0)) - at(p - Vec2::new(h, 0.0))) / (2.0 * h),
            (at(p + Vec2::new(0.0, h)) - at(p - Vec2::new(0.0, h))) / (2.0 * h),
        );
        assert!(
            (gradient - numeric).abs().max_element() < 1e-2,
            "{gradient} vs {numeric}"
        );
    }
}

#[test]
fn footprints_select_and_blend_mip_levels() {
    let fine = texels(&noise(), 32);
    let mid = halve(32, &fine);
    let coarse = halve(16, &mid);
    let chain = image(
        alloc::vec![
            level(32, fine.clone()),
            level(16, mid.clone()),
            level(8, coarse.clone())
        ],
        4,
    );
    let only = |size, values| image(alloc::vec![level(size, values)], 5);
    let (l0, l1, l2) = (only(32, fine), only(16, mid), only(8, coarse));
    let p = Vec2::new(0.43, 0.61);
    let at = |w: f32| chain.sample(p, Footprint::new(w).unwrap());
    let point = |img: &SampleImage| img.sample(p, Footprint::POINT);
    assert_eq!(at(0.0).to_bits(), point(&l0).to_bits());
    assert_eq!(
        at(1.0 / 64.0).to_bits(),
        point(&l0).to_bits(),
        "finer than level 0"
    );
    assert_eq!(
        at(1.0 / 16.0).to_bits(),
        point(&l1).to_bits(),
        "exactly level 1"
    );
    assert_eq!(
        at(1.0).to_bits(),
        point(&l2).to_bits(),
        "past the last level"
    );
    // Between levels 1 and 2 the value blends them.
    let between = at(3.0 / 32.0);
    let (a, b) = (point(&l1), point(&l2));
    assert!(between >= a.min(b) && between <= a.max(b));
}

#[test]
fn images_are_validated() {
    let values = alloc::vec![0.0_f32; 16];
    assert!(ImageLevel::new(4, 4, Vec2::splat(0.25), values.clone()).is_ok());
    assert!(ImageLevel::new(4, 3, Vec2::splat(0.25), values.clone()).is_err());
    assert!(ImageLevel::new(4, 4, Vec2::splat(-0.25), values.clone()).is_err());
    // A periodic image covers exactly one period from the origin.
    let short = ImageLevel::new(4, 4, Vec2::splat(0.2), values.clone()).unwrap();
    assert!(SampleImage::new(torus(), Vec2::ZERO, alloc::vec![short], Fingerprint(0)).is_err());
    let moved = level(4, values.clone());
    assert!(
        SampleImage::new(
            torus(),
            Vec2::splat(0.5),
            alloc::vec![moved],
            Fingerprint(0)
        )
        .is_err()
    );
    assert!(SampleImage::new(torus(), Vec2::ZERO, Vec::new(), Fingerprint(0)).is_err());
    // Mips cover the same extent with growing texels.
    let bad = ImageLevel::new(2, 2, Vec2::splat(0.25), alloc::vec![0.0; 4]).unwrap();
    assert!(
        SampleImage::new(
            torus(),
            Vec2::ZERO,
            alloc::vec![level(4, values), bad],
            Fingerprint(0)
        )
        .is_err()
    );
}

#[test]
fn sample_ops_fingerprint_their_derivation() {
    let values = texels(&noise(), 8);
    let a = image(alloc::vec![level(8, values.clone())], 7);
    let same = image(alloc::vec![level(8, alloc::vec![0.0; 64])], 7);
    let other = image(alloc::vec![level(8, values)], 8);
    let fp = |image: SampleImage| {
        let mut b = ProgramBuilder::new();
        let id = b.add(Op::Sample { image }).unwrap();
        b.finish(id).unwrap().fingerprint()
    };
    assert_eq!(
        fp(a.clone()),
        sample_fingerprint(Fingerprint(7), SamplePolicy::Linear)
    );
    assert_eq!(fp(a.clone()), fp(same), "texels are not fingerprinted");
    assert_ne!(fp(a), fp(other));
}

#[test]
fn programs_sample_bit_identically_through_flat_and_recursive_plans() {
    let sampled = image(alloc::vec![level(32, texels(&noise(), 32))], 9);
    let domain = torus();
    let mut b = ProgramBuilder::new();
    let s = b
        .add(Op::Sample {
            image: sampled.clone(),
        })
        .unwrap();
    let wobble = b
        .add(Op::Noise {
            basis: Basis::Value,
            domain,
            frequency: [2.0, 2.0],
            seed: 3,
        })
        .unwrap();
    let warped = b
        .add(Op::Warp {
            input: s,
            dx: wobble,
            dy: wobble,
            amount: 0.05,
        })
        .unwrap();
    let sum = b.add(Op::Add { a: warped, b: s }).unwrap();
    let program = b.finish(sum).unwrap();
    let mut flat = program.evaluator();
    for p in [Vec2::new(0.1, 0.2), Vec2::new(0.8, 0.45)] {
        for w in [0.0, 1.0 / 16.0] {
            let footprint = Footprint::new(w).unwrap();
            let recursive = program.eval_node(program.output(), p, footprint);
            assert_eq!(flat.eval(p, footprint).to_bits(), recursive.to_bits());
            assert_eq!(
                program.eval_node(s, p, footprint).to_bits(),
                sampled.sample(p, footprint).to_bits()
            );
        }
    }
}

/// A `size` × `size` level over one period with `channels` components per
/// texel from `f(x, y)`.
fn channels_level(size: u32, channels: u8, f: impl Fn(u32, u32) -> [f32; 3]) -> ImageLevel {
    let mut values = Vec::new();
    for y in 0..size {
        for x in 0..size {
            values.extend_from_slice(&f(x, y)[..usize::from(channels)]);
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "tiny test sizes")]
    let texel = Vec2::splat(1.0 / size as f32);
    ImageLevel::with_channels(size, size, texel, channels, values).unwrap()
}

fn typed(port: PortType, policy: SamplePolicy, levels: Vec<ImageLevel>) -> SampleImage {
    SampleImage::typed(port, policy, torus(), Vec2::ZERO, levels, Fingerprint(1)).unwrap()
}

#[test]
fn typed_images_check_their_levels_and_policy() {
    let color = PortType::Color(Primaries::Rec709);
    let three = || channels_level(4, 3, |x, y| [x as f32, y as f32, 1.0]);
    let make = |port, policy, level| {
        SampleImage::typed(
            port,
            policy,
            torus(),
            Vec2::ZERO,
            alloc::vec![level],
            Fingerprint(0),
        )
    };
    assert!(make(color, SamplePolicy::Linear, three()).is_ok());
    // Three channels are not a direction.
    assert!(make(PortType::Direction, SamplePolicy::Linear, three()).is_err());
    let ids = || ImageLevel::ids(4, 4, Vec2::splat(0.25), alloc::vec![7_u32; 16]).unwrap();
    assert!(make(PortType::Id, SamplePolicy::Nearest, ids()).is_ok());
    // Identifiers never interpolate, and numbers are not identifiers.
    assert_eq!(
        make(PortType::Id, SamplePolicy::Linear, ids()).err(),
        Some(crate::DomainError::InvalidParameter {
            name: "sample policy"
        })
    );
    assert!(make(PortType::Scalar, SamplePolicy::Nearest, ids()).is_err());
    assert!(ImageLevel::with_channels(4, 4, Vec2::splat(0.25), 4, alloc::vec![0.0; 64]).is_err());
    assert!(ImageLevel::with_channels(4, 4, Vec2::splat(0.25), 2, alloc::vec![0.0; 31]).is_err());
    assert!(SamplePolicy::default_for(PortType::Id).permits(PortType::Id));
    assert!(!SamplePolicy::Linear.permits(PortType::Id));
}

#[test]
fn linear_sampling_interpolates_each_component_like_a_scalar() {
    let f = |x: u32, y: u32| {
        let (x, y) = (x as f32, y as f32);
        [x * 0.25 + y, y * y - x, (x * 3.0 + y) * 0.1]
    };
    let vectors = typed(
        PortType::Vector3,
        SamplePolicy::Linear,
        alloc::vec![channels_level(8, 3, f)],
    );
    let channel = |k: usize| {
        let mut values = Vec::new();
        for y in 0..8 {
            for x in 0..8 {
                values.push(f(x, y)[k]);
            }
        }
        image(alloc::vec![level(8, values)], 1)
    };
    for p in [Vec2::new(0.13, 0.71), Vec2::new(0.99, 0.02)] {
        let Value::Vector3(v) = vectors.sample_value(p, Footprint::POINT) else {
            panic!("vector3 images sample vectors")
        };
        for k in 0..3 {
            assert_eq!(
                v[k].to_bits(),
                channel(k).sample(p, Footprint::POINT).to_bits()
            );
        }
    }
}

#[test]
fn normals_keep_their_mean_length_and_directions_stay_axial() {
    // Alternating tilts: halfway between two texels the mean normal is
    // shorter than 1, and sampling keeps that length.
    let tilt = |s: f32| glam::Vec3::new(s, 0.0, 1.0).normalize().to_array();
    let normals = typed(
        PortType::Normal(NormalFrame::Domain),
        SamplePolicy::Linear,
        alloc::vec![channels_level(4, 3, |x, _| tilt(if x % 2 == 0 {
            0.8
        } else {
            -0.8
        }))],
    );
    let Value::Vector3(mid) = normals.sample_value(Vec2::new(0.25, 0.125), Footprint::POINT) else {
        panic!("normals sample vectors")
    };
    assert!(
        (mid.length() - 1.0 / libm::sqrtf(1.64)).abs() < 1e-5,
        "{mid}"
    );
    // A direction and its opposite (θ and θ + π) share a doubled-angle
    // vector, so their blend is that direction at full agreement.
    let axis = |theta: f32| [libm::cosf(2.0 * theta), libm::sinf(2.0 * theta), 0.0];
    let pi = core::f32::consts::PI;
    let directions = typed(
        PortType::Direction,
        SamplePolicy::Linear,
        alloc::vec![channels_level(4, 2, |x, _| axis(if x % 2 == 0 {
            0.4
        } else {
            0.4 + pi
        }))],
    );
    let Value::Vector2(d) = directions.sample_value(Vec2::new(0.25, 0.125), Footprint::POINT)
    else {
        panic!("directions sample doubled-angle vectors")
    };
    assert!((d.length() - 1.0).abs() < 1e-5);
    assert!((0.5 * libm::atan2f(d.y, d.x) - 0.4).abs() < 1e-5);
}

#[test]
fn identifiers_sample_the_nearest_texel_of_the_nearest_level() {
    let ids: Vec<u32> = (0..16).collect();
    let fine = ImageLevel::ids(4, 4, Vec2::splat(0.25), ids).unwrap();
    let coarse = ImageLevel::ids(2, 2, Vec2::splat(0.5), alloc::vec![100, 101, 102, 103]).unwrap();
    let image = typed(
        PortType::Id,
        SamplePolicy::Nearest,
        alloc::vec![fine, coarse],
    );
    // Texel (2, 1) of level 0 holds 6; just past a texel border the
    // neighbor wins, never a blend.
    assert_eq!(
        image.sample_value(Vec2::new(0.6, 0.3), Footprint::POINT),
        Value::Id(6)
    );
    assert_eq!(
        image.sample_value(Vec2::new(0.49, 0.3), Footprint::POINT),
        Value::Id(5)
    );
    // Periodic: wraps.
    assert_eq!(
        image.sample_value(Vec2::new(1.6, -0.7), Footprint::POINT),
        Value::Id(6)
    );
    // A footprint of 1.6 texels rounds to level 1; of 1.3 to level 0.
    let at = |w: f32| image.sample_value(Vec2::new(0.6, 0.3), Footprint::new(w).unwrap());
    assert_eq!(at(0.4), Value::Id(101));
    assert_eq!(at(0.325), Value::Id(6));
    // A program sampling identifiers has identifier type and value.
    let mut b = ProgramBuilder::new();
    let id = b.add(Op::Sample { image }).unwrap();
    assert_eq!(b.port_type(id), Ok(PortType::Id));
    let program = b.finish_value(id).unwrap();
    assert_eq!(
        program.eval(Vec2::new(0.6, 0.3), Footprint::POINT),
        Value::Id(6)
    );
}

#[test]
fn sampling_policies_are_fingerprinted() {
    let values = texels(&noise(), 8);
    let fp = |policy| {
        let image = SampleImage::typed(
            PortType::Scalar,
            policy,
            torus(),
            Vec2::ZERO,
            alloc::vec![level(8, values.clone())],
            Fingerprint(3),
        )
        .unwrap();
        let mut b = ProgramBuilder::new();
        let id = b.add(Op::Sample { image }).unwrap();
        b.finish(id).unwrap().fingerprint()
    };
    assert_ne!(fp(SamplePolicy::Linear), fp(SamplePolicy::Nearest));
    assert_eq!(
        fp(SamplePolicy::Nearest),
        sample_fingerprint(Fingerprint(3), SamplePolicy::Nearest)
    );
}
