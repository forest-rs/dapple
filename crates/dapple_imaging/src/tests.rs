// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::f32::consts::PI;

use dapple_field::raster::Region;
use dapple_field::{Domain, Footprint};
use dapple_raster::{Raster, Realization};
use glam::Vec2;
use imaging::kurbo::{BezPath, Circle, Rect};
use imaging::peniko::Color;
use imaging::{Painter, record::Scene};

use super::{ImagingError, coverage_image, rasterize};

fn torus() -> Domain {
    Domain::periodic(1, 1).unwrap()
}

fn scene(draw: impl FnOnce(&mut Painter<'_, Scene>)) -> Scene {
    let mut scene = Scene::new();
    draw(&mut Painter::new(&mut scene));
    scene
}

fn disk(center: (f64, f64), radius: f64) -> Scene {
    scene(|p| p.fill(Circle::new(center, radius), Color::WHITE).draw())
}

/// Mean coverage: the covered fraction of the realized region.
fn mean(mask: &Raster) -> f32 {
    mask.values().iter().sum::<f32>() / mask.values().len() as f32
}

/// How far below `area` a curved outline of `perimeter` may cover, with
/// chords within a quarter texel of the curve, plus 8-bit rounding.
fn area_slack(perimeter: f32, texel: f32) -> f32 {
    perimeter * texel * 0.25 + 2.0 * perimeter * texel / 255.0
}

/// A pointed leaf blade: two arcs meeting at the base and the tip.
fn leaf() -> Scene {
    let mut blade = BezPath::new();
    blade.move_to((0.5, 0.08));
    blade.curve_to((0.84, 0.3), (0.8, 0.72), (0.5, 0.94));
    blade.curve_to((0.2, 0.72), (0.16, 0.3), (0.5, 0.08));
    blade.close_path();
    scene(|p| p.fill(&blade, Color::WHITE).draw())
}

#[test]
fn texel_aligned_shapes_cover_whole_texels() {
    let square = scene(|p| p.fill_rect(Rect::new(0.25, 0.25, 0.75, 0.5), Color::WHITE));
    let mask = rasterize(&square, Realization::period(torus(), 16, 16).unwrap()).unwrap();
    for y in 0..16 {
        for x in 0..16 {
            let inside = (4..12).contains(&x) && (4..8).contains(&y);
            assert_eq!(mask.at(x, y), if inside { 1.0 } else { 0.0 }, "({x}, {y})");
        }
    }
}

#[test]
fn coverage_is_area_in_steps_of_1_255() {
    // A half-texel-wide sliver covers half of each texel it crosses.
    let sliver = scene(|p| p.fill_rect(Rect::new(0.0, 0.0, 0.5 / 16.0, 1.0), Color::WHITE));
    let mask = rasterize(&sliver, Realization::period(torus(), 16, 16).unwrap()).unwrap();
    for y in 0..16 {
        assert!(
            (mask.at(0, y) - 0.5).abs() <= 1.0 / 255.0,
            "{}",
            mask.at(0, y)
        );
        assert_eq!(mask.at(1, y), 0.0);
    }
    let mask = rasterize(&leaf(), Realization::period(torus(), 64, 64).unwrap()).unwrap();
    for &v in mask.values() {
        let steps = v * 255.0;
        assert!(
            (0.0..=1.0).contains(&v) && (steps - steps.round()).abs() < 1e-4,
            "{v}"
        );
    }
}

#[test]
fn a_disk_covers_its_area() {
    let mask = rasterize(
        &disk((0.5, 0.5), 0.3),
        Realization::period(torus(), 128, 128).unwrap(),
    )
    .unwrap();
    let (area, slack) = (PI * 0.09, area_slack(2.0 * PI * 0.3, 1.0 / 128.0));
    assert!(
        mean(&mask) <= area && mean(&mask) >= area - slack,
        "{}",
        mean(&mask)
    );
}

#[test]
fn plane_regions_map_domain_units_to_texels() {
    // The region [2, 4] × [-1, 0], 64 × 32 texels; a disk inside it.
    let region = Region {
        origin: Vec2::new(2.0, -1.0),
        size: Vec2::new(2.0, 1.0),
    };
    let realization = Realization::region(region, 64, 32).unwrap();
    let mask = rasterize(&disk((3.0, -0.5), 0.25), realization).unwrap();
    assert_eq!(
        (mask.origin(), mask.texel()),
        (region.origin, Vec2::splat(1.0 / 32.0))
    );
    let area = mean(&mask) * 2.0;
    let slack = area_slack(PI * 0.5, 1.0 / 32.0);
    assert!(area <= PI / 16.0 && area >= PI / 16.0 - slack, "{area}");
    // Centered on texel (32, 16)'s corner; nothing outside its reach.
    assert_eq!(mask.at(32, 16), 1.0);
    assert_eq!(mask.at(20, 16), 0.0);
}

#[test]
fn wrapping_masks_tile() {
    // A disk at the tile's corner splits into four quarters, which together
    // are the centered disk rolled by half a tile.
    let realization = Realization::period(torus(), 64, 64).unwrap();
    let corner = rasterize(&disk((0.0, 0.0), 0.2), realization).unwrap();
    let center = rasterize(&disk((0.5, 0.5), 0.2), realization).unwrap();
    let rolled = center.rolled(32, 32);
    for (a, b) in corner.values().iter().zip(rolled.values()) {
        assert!((a - b).abs() <= 1.0 / 255.0, "{a} {b}");
    }
    // Only the wrapping realization wraps.
    let region = Region {
        origin: Vec2::ZERO,
        size: Vec2::ONE,
    };
    let clamped = rasterize(
        &disk((0.0, 0.0), 0.2),
        Realization::region(region, 64, 64).unwrap(),
    )
    .unwrap();
    assert_eq!(clamped.at(63, 63), 0.0);
    assert_eq!(corner.at(63, 63), 1.0);
}

#[test]
fn masks_are_deterministic() {
    let realization = Realization::period(torus(), 96, 64).unwrap();
    let a = rasterize(&leaf(), realization).unwrap();
    let b = rasterize(&leaf(), realization).unwrap();
    assert_eq!(a.digest(), b.digest());
}

/// Pins the rasterizer's output at the pinned `imaging` revision; see the
/// crate docs on determinism.
#[test]
fn golden_leaf_mask() {
    let mask = rasterize(&leaf(), Realization::period(torus(), 64, 64).unwrap()).unwrap();
    assert_eq!(
        mask.digest(),
        0xd4bc_bae4_f338_cf45,
        "digest {:#018x}",
        mask.digest()
    );
}

#[test]
fn coverage_images_are_area_means_of_the_mask() {
    let realization = Realization::period(torus(), 64, 32).unwrap();
    let image = coverage_image(&disk((0.5, 0.5), 0.3), realization).unwrap();
    let sizes: alloc::vec::Vec<_> = image
        .levels()
        .iter()
        .map(|l| (l.width(), l.height()))
        .collect();
    assert_eq!(
        sizes,
        [(64, 32), (32, 16), (16, 8), (8, 4), (4, 2), (2, 1), (1, 1)]
    );
    // Coarse levels are area means of level 0: the 1 × 1 level is the
    // mask's mean, and each 2 × 2 block of a level averages to the texel
    // below it.
    let base = rasterize(&disk((0.5, 0.5), 0.3), realization).unwrap();
    let levels = image.levels();
    let top = levels.last().unwrap().values()[0];
    assert!((top - mean(&base)).abs() < 1e-6, "{top}");
    let (fine, coarse) = (&levels[1], &levels[2]);
    for y in 0..coarse.height() as usize {
        for x in 0..coarse.width() as usize {
            let at = |dx: usize, dy: usize| fine.values()[(2 * y + dy) * 32 + 2 * x + dx];
            let block = (at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) / 4.0;
            let texel = coarse.values()[y * 16 + x];
            assert!((block - texel).abs() < 1e-6, "{block} {texel}");
        }
    }
    // Sampled as a field: inside, outside, and through a wide footprint.
    let point = Footprint::POINT;
    assert_eq!(image.sample(Vec2::new(0.5, 0.5), point), 1.0);
    assert_eq!(image.sample(Vec2::new(0.05, 0.05), point), 0.0);
    let wide = image.sample(Vec2::new(0.5, 0.5), Footprint::new(4.0).unwrap());
    assert!((wide - top).abs() < 1e-6, "{wide}");
    // Equal scenes derive equal images; different ones differ.
    let again = coverage_image(&disk((0.5, 0.5), 0.3), realization).unwrap();
    let other = coverage_image(&disk((0.5, 0.5), 0.2), realization).unwrap();
    assert_eq!(image.derivation(), again.derivation());
    assert_ne!(image.derivation(), other.derivation());
}

#[test]
fn errors() {
    let big = Realization::period(torus(), 70_000, 4).unwrap();
    assert!(matches!(
        rasterize(&leaf(), big),
        Err(ImagingError::TooLarge {
            width: 70_000,
            height: 4
        })
    ));
    let unbalanced = scene(|p| p.push_fill_clip(Rect::new(0.0, 0.0, 0.5, 0.5)));
    assert!(matches!(
        rasterize(&unbalanced, Realization::period(torus(), 8, 8).unwrap()),
        Err(ImagingError::Render(_))
    ));
}
