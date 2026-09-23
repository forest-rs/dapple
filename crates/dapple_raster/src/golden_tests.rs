// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Golden digests and exact tiling of raster operations.
//!
//! A digest change means every realized texture changes. Update these only
//! for an intentional, documented change to an operation's definition.

use alloc::vec::Vec;

use dapple_field::raster::{Grid, Region};
use dapple_field::{Basis, CellOutput, Cellular, Domain, Fractal, FractalParams, ScalarField};
use glam::Vec2;

use crate::{
    AmbientOcclusion, DistanceTransform, GaussianBlur, HeightToNormal, Raster, RasterOp,
    Realization, SampledField, realize,
};

fn torus() -> Domain {
    Domain::periodic(2, 1).unwrap()
}

/// A periodic height field with ridges and cells.
fn height() -> Raster {
    let fbm = Fractal::new(
        Basis::Gradient,
        torus(),
        Vec2::new(3.0, 5.0),
        11,
        FractalParams::default(),
    )
    .unwrap();
    realize(&fbm, Realization::period(torus(), 48, 24).unwrap()).unwrap()
}

fn cells() -> Raster {
    let cells = Cellular::new(torus(), Vec2::new(4.0, 2.0), 1.0, 12)
        .unwrap()
        .output(CellOutput::F1);
    realize(&cells, Realization::period(torus(), 48, 24).unwrap()).unwrap()
}

const BLUR: GaussianBlur = GaussianBlur { sigma: 0.05 };
const NORMAL: HeightToNormal = HeightToNormal { scale: 0.05 };
const AO: AmbientOcclusion = AmbientOcclusion {
    radius: 0.15,
    directions: 8,
    scale: 0.1,
};
const DISTANCE: DistanceTransform = DistanceTransform { threshold: 0.9 };

#[test]
fn realization_matches_grid_sampling() {
    let fbm = Fractal::new(
        Basis::Gradient,
        torus(),
        Vec2::new(3.0, 5.0),
        11,
        FractalParams::default(),
    )
    .unwrap();
    let grid = Grid::sample(&fbm, Region::period(torus()).unwrap(), 48, 24);
    assert_eq!(height().values(), grid.values.as_slice());
}

#[test]
fn golden_digests() {
    let got = [
        height().digest(),
        BLUR.apply(&height()).unwrap().digest(),
        NORMAL.apply(&height()).unwrap().digest(),
        AO.apply(&height()).unwrap().digest(),
        DISTANCE.apply(&cells()).unwrap().digest(),
    ];
    assert_eq!(got, GOLDEN, "digests: {got:#018x?}");
}

const GOLDEN: [u64; 5] = [
    0x5774_0a72_8ecf_66f7,
    0x1e1d_92bc_2c83_5fac,
    0x272d_0ab3_1c65_1986,
    0x9b6e_2b32_0811_06fb,
    0xbf5b_7e6b_0ecb_d524,
];

/// Every op commutes exactly with rolling a wrapping raster, which is what
/// "tiles seamlessly" means for a raster.
#[test]
fn wrapping_ops_commute_with_rolling() {
    for (dx, dy) in [(1, 0), (0, 1), (17, -5), (-47, 23)] {
        let h = height();
        let rolled = h.rolled(dx, dy);
        assert_eq!(
            BLUR.apply(&rolled).unwrap(),
            BLUR.apply(&h).unwrap().rolled(dx, dy),
            "blur ({dx}, {dy})"
        );
        assert_eq!(
            NORMAL.apply(&rolled).unwrap(),
            NORMAL.apply(&h).unwrap().rolled(dx, dy),
            "normal ({dx}, {dy})"
        );
        assert_eq!(
            AO.apply(&rolled).unwrap(),
            AO.apply(&h).unwrap().rolled(dx, dy),
            "ao ({dx}, {dy})"
        );
        let c = cells();
        assert_eq!(
            DISTANCE.apply(&c.rolled(dx, dy)).unwrap(),
            DISTANCE.apply(&c).unwrap().rolled(dx, dy),
            "distance ({dx}, {dy})"
        );
    }
}

#[test]
fn sampled_rasters_are_fields_again() {
    let h = height();
    let field = SampledField::new(h.clone(), torus()).unwrap();
    assert_eq!(field.domain(), torus());
    // Texel centers reproduce the raster up to coordinate rounding.
    let back = realize(&field, Realization::period(torus(), 48, 24).unwrap()).unwrap();
    for (a, b) in back.values().iter().zip(h.values()) {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }
    // Repeats of a point sample identical bits, across the seam too.
    for i in 0..32_u8 {
        let p = Vec2::new(f32::from(i) * 0.0625, f32::from(i % 8) * 0.125);
        for shift in [Vec2::new(2.0, 0.0), Vec2::new(-4.0, 3.0)] {
            assert_eq!(
                field.eval(p, dapple_field::Footprint::POINT).to_bits(),
                field
                    .eval(p + shift, dapple_field::Footprint::POINT)
                    .to_bits(),
                "{p} + {shift}"
            );
        }
    }
    // A mismatched period is refused.
    assert!(SampledField::new(h, Domain::periodic(1, 1).unwrap()).is_err());
}

#[test]
fn realization_checks_domains() {
    let plane_noise = dapple_field::Noise::new(Basis::Value, Domain::Plane, Vec2::ONE, 1).unwrap();
    assert!(realize(&plane_noise, Realization::period(torus(), 4, 4).unwrap()).is_err());
    let region = Region {
        origin: Vec2::new(-1.0, 2.0),
        size: Vec2::new(3.0, 1.0),
    };
    let clamped = realize(&plane_noise, Realization::region(region, 6, 2).unwrap()).unwrap();
    assert_eq!(clamped.edge(), crate::Edge::Clamp);
    assert_eq!(clamped.texel(), Vec2::new(0.5, 0.5));
    assert!(Realization::period(Domain::Plane, 4, 4).is_err());
    assert!(Realization::period(torus(), 0, 4).is_err());
    let _: Vec<f32> = clamped.values().to_vec();
}
