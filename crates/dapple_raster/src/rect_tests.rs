// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Rectangle-wise realization and operations equal whole passes, bit for bit.

use alloc::vec::Vec;

use dapple_field::raster::Region;
use dapple_field::{Basis, Domain, Fractal, FractalParams};
use glam::Vec2;

use crate::{
    AmbientOcclusion, DigestValue, DistanceTransform, GaussianBlur, HeightToNormal, Raster,
    RasterOp, Realization, TexelRect, realize, realize_into,
};

fn torus() -> Domain {
    Domain::periodic(2, 1).unwrap()
}

fn fbm(domain: Domain) -> Fractal {
    Fractal::new(
        Basis::Gradient,
        domain,
        Vec2::new(3.0, 5.0),
        7,
        FractalParams::default(),
    )
    .unwrap()
}

/// Uneven tiles covering a `width` × `height` grid.
fn tiles(width: u32, height: u32) -> Vec<TexelRect> {
    let mut out = Vec::new();
    let (xs, ys) = ([0, 5, 17, 18, width], [0, 3, 11, height]);
    for y in ys.windows(2) {
        for x in xs.windows(2) {
            out.push(TexelRect {
                x0: x[0],
                y0: y[0],
                x1: x[1],
                y1: y[1],
            });
        }
    }
    out
}

fn realizations() -> [Realization; 2] {
    [
        Realization::period(torus(), 40, 20).unwrap(),
        Realization::region(
            Region {
                origin: Vec2::new(-0.3, 0.2),
                size: Vec2::new(1.7, 0.9),
            },
            40,
            20,
        )
        .unwrap(),
    ]
}

#[test]
fn realizing_in_rectangles_equals_realizing_whole() {
    for (realization, domain) in realizations().into_iter().zip([torus(), Domain::Plane]) {
        let field = fbm(domain);
        let whole = realize(&field, realization).unwrap();
        let mut pieces = whole.clone();
        // Start from different values so every texel must be rewritten.
        pieces.map_rect(TexelRect::full(40, 20), |_, _| f32::NAN);
        for rect in tiles(40, 20) {
            realize_into(&field, realization, rect, &mut pieces).unwrap();
        }
        assert_eq!(pieces.digest(), whole.digest());
    }
}

fn check<Op: RasterOp>(op: &Op, input: &Raster, stale: Raster<Op::Output>)
where
    Op::Output: DigestValue,
{
    let whole = op.apply(input).unwrap();
    // Recomputing every tile from stale values reproduces the whole pass.
    let mut pieces = stale.clone();
    for rect in tiles(input.width(), input.height()) {
        op.apply_into(input, rect, &mut pieces).unwrap();
    }
    assert_eq!(pieces.digest(), whole.digest());
    // Recomputing one tile rewrites exactly that tile.
    let rect = TexelRect {
        x0: 5,
        y0: 3,
        x1: 17,
        y1: 11,
    };
    let mut one = stale.clone();
    op.apply_into(input, rect, &mut one).unwrap();
    assert!(one.rect_bits_eq(&whole, rect));
    assert!(one.rect_bits_eq(
        &stale,
        TexelRect {
            x0: 17,
            ..TexelRect::full(40, 20)
        }
    ));
    // A rectangle off the grid, or a stale raster on another grid, is refused.
    let off = TexelRect { x1: 41, ..rect };
    assert!(op.apply_into(input, off, &mut stale.clone()).is_err());
}

#[test]
fn operations_in_rectangles_equal_whole_passes() {
    for (realization, domain) in realizations().into_iter().zip([torus(), Domain::Plane]) {
        let input = realize(&fbm(domain), realization).unwrap();
        let zeros = input.map_texels(|_, _| 0.25_f32);
        let normals = input.map_texels(|_, _| [0.0_f32, 0.0, 1.0]);
        check(&GaussianBlur { sigma: 0.07 }, &input, zeros.clone());
        check(&HeightToNormal { scale: 0.3 }, &input, normals);
        check(
            &AmbientOcclusion {
                radius: 0.12,
                directions: 6,
                scale: 0.5,
            },
            &input,
            zeros.clone(),
        );
        check(&DistanceTransform { threshold: 0.1 }, &input, zeros);
    }
}

#[test]
fn analytic_normals_agree_with_the_stencil() {
    // A smooth, low-frequency height, where the two-texel stencil is
    // accurate: the two normal maps agree closely.
    let height = Fractal::new(
        Basis::Gradient,
        torus(),
        Vec2::new(1.0, 2.0),
        3,
        FractalParams {
            octaves: 2,
            ..FractalParams::default()
        },
    )
    .unwrap();
    let realization = Realization::period(torus(), 128, 64).unwrap();
    let analytic = crate::realize_normals(&height, realization, 0.2).unwrap();
    let stencil = HeightToNormal { scale: 0.2 }
        .apply(&realize(&height, realization).unwrap())
        .unwrap();
    let mut worst = 0.0_f32;
    for (a, b) in analytic.values().iter().zip(stencil.values()) {
        let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        worst = worst.max(1.0 - dot);
    }
    assert!(
        worst < 1e-3,
        "largest normal disagreement 1 - cos = {worst}"
    );
    assert!(crate::realize_normals(&height, realization, f32::NAN).is_err());
}
