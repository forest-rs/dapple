// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Bit-exact golden digests and exact tiling.
//!
//! A digest change means every realized texture changes. Update these only
//! for an intentional, documented change to a field's definition.

use glam::Vec2;

use crate::raster::{Grid, Region};
use crate::{
    Basis, CellOutput, Cellular, Domain, Footprint, Fractal, FractalKind, FractalParams, Noise,
    ScalarField,
};

fn periodic() -> Domain {
    Domain::periodic(2, 1).unwrap()
}

/// The golden corpus: name, field, and its expected digest over one period.
fn corpus() -> [(&'static str, alloc::boxed::Box<dyn ScalarField>, u64); 6] {
    let d = periodic();
    let f = Vec2::new(3.0, 5.0);
    let ridged = FractalParams {
        kind: FractalKind::Ridged,
        ..FractalParams::default()
    };
    [
        (
            "value",
            alloc::boxed::Box::new(Noise::new(Basis::Value, d, f, 1).unwrap()),
            GOLDEN[0],
        ),
        (
            "gradient",
            alloc::boxed::Box::new(Noise::new(Basis::Gradient, d, f, 2).unwrap()),
            GOLDEN[1],
        ),
        (
            "fbm",
            alloc::boxed::Box::new(
                Fractal::new(Basis::Gradient, d, f, 3, FractalParams::default()).unwrap(),
            ),
            GOLDEN[2],
        ),
        (
            "ridged",
            alloc::boxed::Box::new(Fractal::new(Basis::Value, d, f, 4, ridged).unwrap()),
            GOLDEN[3],
        ),
        (
            "cellular-f1",
            alloc::boxed::Box::new(Cellular::new(d, f, 1.0, 5).unwrap().output(CellOutput::F1)),
            GOLDEN[4],
        ),
        (
            "cellular-border",
            alloc::boxed::Box::new(
                Cellular::new(d, f, 0.8, 6)
                    .unwrap()
                    .output(CellOutput::Border),
            ),
            GOLDEN[5],
        ),
    ]
}

const GOLDEN: [u64; 6] = [
    0x9011_4b5e_e286_a6a7,
    0xa7a3_ba54_d740_1013,
    0xbec9_2eb7_597f_6e54,
    0xa7b3_cc81_770e_37a3,
    0xca9e_7616_6349_8cc5,
    0x032a_4f9f_e190_1862,
];

#[test]
fn golden_digests() {
    let region = Region::period(periodic()).unwrap();
    let mut failures = alloc::vec::Vec::new();
    for (name, field, expected) in corpus() {
        let digest = Grid::sample(&field, region, 32, 16).digest();
        if digest != expected {
            failures.push(alloc::format!("{name}: {digest:#018x}"));
        }
    }
    assert!(failures.is_empty(), "digest changes: {failures:?}");
}

#[test]
fn periodic_fields_repeat_bit_exactly() {
    // Dyadic sample points keep `p + period` exact in f32, so equality must
    // be exact, not approximate.
    let [px, py] = periodic().period().unwrap();
    for (name, field, _) in corpus() {
        for i in 0..64_u8 {
            let p = Vec2::new(f32::from(i % 8) * 0.25, f32::from(i / 8) * 0.125);
            let base = field.eval(p, Footprint::POINT);
            for shift in [
                Vec2::new(px as f32, 0.0),
                Vec2::new(0.0, py as f32),
                Vec2::new(-(px as f32), 3.0 * py as f32),
            ] {
                let moved = field.eval(p + shift, Footprint::POINT);
                assert_eq!(base.to_bits(), moved.to_bits(), "{name} at {p} + {shift}");
            }
        }
    }
}

#[test]
fn periodic_grids_are_continuous_across_the_seam() {
    // The wrap from the last texel to the first must look like any interior
    // step: compare the largest seam jump to the largest interior jump.
    let region = Region::period(periodic()).unwrap();
    for (name, field, _) in corpus() {
        if name.starts_with("cellular") {
            continue; // Not continuous by design at cell borders.
        }
        let grid = Grid::sample(&field, region, 128, 64);
        let at = |x: u32, y: u32| grid.values[(y * grid.width + x) as usize];
        let mut interior = 0.0_f32;
        let mut seam = 0.0_f32;
        for y in 0..grid.height {
            for x in 0..grid.width {
                let right = (x + 1) % grid.width;
                let jump = (at(right, y) - at(x, y)).abs();
                if right == 0 {
                    seam = seam.max(jump);
                } else {
                    interior = interior.max(jump);
                }
            }
        }
        assert!(
            seam <= interior,
            "{name}: seam jump {seam} > interior {interior}"
        );
    }
}
