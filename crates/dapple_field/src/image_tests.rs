// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sampled images and `Op::Sample`.

use alloc::vec::Vec;

use glam::Vec2;

use crate::program::{Fingerprint, Op, ProgramBuilder, sample_fingerprint};
use crate::{Basis, Domain, Footprint, ImageLevel, Noise, SampleImage, ScalarField};

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
    assert_eq!(fp(a.clone()), sample_fingerprint(Fingerprint(7)));
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
