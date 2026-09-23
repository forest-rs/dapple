// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use dapple_field::program::ValueProgram;
use dapple_field::{Domain, Footprint, ScalarField, Value};
use glam::Vec2;

use crate::{brick, oak, parquet};

fn torus() -> Domain {
    Domain::periodic(1, 1).unwrap()
}

/// Points spread over the tile, off every joint.
fn points() -> impl Iterator<Item = Vec2> {
    (0..97).map(|i| Vec2::new(i as f32 * 0.0731 % 1.0, i as f32 * 0.1379 % 1.0))
}

fn assert_tiles(name: &str, height: &impl ScalarField, color: &ValueProgram) {
    let footprint = Footprint::new(1.0 / 1024.0).unwrap();
    for p in points() {
        let q = p + Vec2::new(1.0, -1.0);
        let (a, b) = (height.eval(p, footprint), height.eval(q, footprint));
        assert!((a - b).abs() < 1e-4, "{name} height at {p}: {a} vs {b}");
        assert!((-0.01..=1.01).contains(&a), "{name} height {a} at {p}");
        let (Value::Vector3(a), Value::Vector3(b)) =
            (color.eval(p, footprint), color.eval(q, footprint))
        else {
            panic!("{name} color is not a color");
        };
        let (a, b) = (a.to_array(), b.to_array());
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-4, "{name} color at {p}: {a:?} vs {b:?}");
            assert!((0.0..=1.0).contains(x), "{name} color {a:?} at {p}");
        }
    }
}

#[test]
fn tileable_materials_repeat_across_the_period() {
    let bark = oak::bark_height(torus()).unwrap();
    assert_tiles("bark", &bark, &oak::bark_color(torus(), &bark).unwrap());
    assert_tiles(
        "brick",
        &brick::height(torus()).unwrap(),
        &brick::color(torus()).unwrap(),
    );
    assert_tiles(
        "parquet",
        &parquet::height(torus()).unwrap(),
        &parquet::color(torus()).unwrap(),
    );
}

#[test]
fn materials_are_stable_programs() {
    // Building a material twice gives the same graph.
    assert_eq!(
        brick::height(torus()).unwrap().fingerprint(),
        brick::height(torus()).unwrap().fingerprint()
    );
    assert_eq!(
        oak::wood_channels().unwrap()[0].fingerprint(),
        oak::wood_channels().unwrap()[0].fingerprint()
    );
}

#[test]
fn brick_joints_are_mortar() {
    let height = brick::height(torus()).unwrap();
    let footprint = Footprint::new(1.0 / 1024.0).unwrap();
    // A bed joint runs along y = 0 and a brick's middle sits half a course up.
    let joint = height.eval(Vec2::new(0.1, 0.0), footprint);
    let face = height.eval(Vec2::new(0.1, 0.5 / brick::BRICKS[1]), footprint);
    assert!(joint < 0.3 && face > 0.6, "joint {joint}, face {face}");
}
