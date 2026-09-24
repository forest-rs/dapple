// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use dapple_field::program::ValueProgram;
use dapple_field::{Domain, Footprint, ScalarField, Value};
use glam::Vec2;

use crate::{brick, gravel, oak, parquet};

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
        "gravel",
        &gravel::height(torus()).unwrap(),
        &gravel::color(torus()).unwrap(),
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

mod glazed {
    use alloc::sync::Arc;
    use alloc::vec::Vec;

    use dapple_elements::{Composite, Realized};
    use dapple_field::Domain;
    use dapple_raster::typed::Storage;
    use dapple_raster::{DistanceTransform, Raster, RasterOp};

    use crate::glazed_brick::{self, BODY, GLAZE, MORTAR};

    const SIZE: u32 = 512;

    fn composite() -> (Realized, glazed_brick::Finished) {
        let domain = Domain::periodic(1, 1).unwrap();
        let set = glazed_brick::layout().unwrap();
        let instance = glazed_brick::instance(Arc::new(glazed_brick::program().unwrap())).unwrap();
        let realized = Realized::composite(&Composite {
            set: &set,
            instance: &instance,
            background: &glazed_brick::BACKGROUND,
            domain,
            width: SIZE,
            height: SIZE,
            tile_size: 64,
        })
        .unwrap();
        let finished = glazed_brick::finish(&realized, domain).unwrap();
        (realized, finished)
    }

    fn ids(finished: &glazed_brick::Finished) -> Vec<u32> {
        let Storage::U32(ids) = finished.material.storage() else {
            panic!("materials are identifiers")
        };
        ids.values().to_vec()
    }

    #[test]
    fn chips_break_from_the_arris_and_expose_rough_body() {
        let (realized, finished) = composite();
        let materials = ids(&finished);
        let Some(Storage::F32(cover)) = realized
            .output("cover")
            .unwrap()
            .map(|r| r.storage().clone())
        else {
            panic!("cover is a mask")
        };
        // Distance from each texel into its brick: to the nearest mortar.
        let mortar = Raster::from_values(
            SIZE,
            SIZE,
            cover.origin(),
            cover.texel(),
            cover.edge(),
            cover.values().iter().map(|c| 1.0 - c).collect(),
        )
        .unwrap();
        let inward = DistanceTransform { threshold: 0.5 }.apply(&mortar).unwrap();
        let rough = finished.specular_roughness.values();
        let (mut body, mut near_edge) = (0_u32, 0_u32);
        let (mut body_rough, mut glaze_rough, mut glazed) = (0.0_f64, 0.0_f64, 0_u32);
        for (i, &m) in materials.iter().enumerate() {
            match m {
                BODY => {
                    body += 1;
                    near_edge += u32::from(inward.values()[i] < 0.01);
                    body_rough += f64::from(rough[i]);
                }
                GLAZE => {
                    glazed += 1;
                    glaze_rough += f64::from(rough[i]);
                }
                MORTAR => {}
                other => panic!("unknown material {other}"),
            }
        }
        // Texels mixing materials blend their roughness, so compare means.
        let (body_rough, glaze_rough) = (
            body_rough / f64::from(body),
            glaze_rough / f64::from(glazed),
        );
        assert!(body_rough > 0.65, "exposed body is rough: {body_rough}");
        assert!(glaze_rough < 0.3, "glaze is glossy: {glaze_rough}");
        // Chips are there, a small share of the bricks, and nearly all of
        // them break from an edge (the rest are pits).
        let share = f64::from(body) / f64::from(SIZE * SIZE);
        assert!(share > 0.002 && share < 0.04, "chipped share {share}");
        assert!(
            near_edge * 100 >= body * 90,
            "{near_edge} of {body} chipped texels near an edge"
        );
    }

    #[test]
    fn mortar_is_recessed_and_textured() {
        let (_, finished) = composite();
        let materials = ids(&finished);
        let height = finished.height.values();
        let Storage::F32x3(color) = finished.base_color.storage() else {
            panic!("colors have three channels")
        };
        let mut mortar = Vec::new();
        let mut glaze = Vec::new();
        for (i, &m) in materials.iter().enumerate() {
            match m {
                MORTAR => mortar.push((height[i], color.values()[i][0])),
                GLAZE => glaze.push(height[i]),
                _ => {}
            }
        }
        let highest_mortar = mortar.iter().map(|m| m.0).fold(f32::MIN, f32::max);
        #[expect(clippy::cast_precision_loss, reason = "texel counts")]
        let mean_glaze = glaze.iter().sum::<f32>() / glaze.len() as f32;
        assert!(
            mean_glaze - highest_mortar > 0.002,
            "glaze at {mean_glaze} m, mortar up to {highest_mortar} m"
        );
        // The mortar is not one flat color.
        let (lo, hi) = mortar
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), m| (a.min(m.1), b.max(m.1)));
        assert!(hi - lo > 0.1, "mortar red from {lo} to {hi}");
        assert!(mortar.iter().all(|m| m.0 < 0.004 && m.0 > -0.002));
    }

    #[test]
    fn brick_faces_are_tilted_and_bowed_apart() {
        let (realized, finished) = composite();
        // Per-brick tilt and bow put brick centers at different heights.
        let mut heights: Vec<f32> = (0..realized.set().len())
            .map(|i| {
                let p = realized
                    .set()
                    .placement(i)
                    .center
                    .rem_euclid(glam::Vec2::ONE);
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "centers lie in the unit period"
                )]
                let (x, y) = ((p.x * SIZE as f32) as i64, (p.y * SIZE as f32) as i64);
                finished.height.at(x, y)
            })
            .collect();
        heights.sort_by(f32::total_cmp);
        assert!(
            heights[heights.len() - 1] - heights[0] > 0.0003,
            "face heights {heights:?}"
        );
    }
}
