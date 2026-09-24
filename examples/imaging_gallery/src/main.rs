// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Writes preview images of `dapple_imaging` coverage masks as PNG files.
//!
//! Run with `cargo run -p imaging_gallery -- [output-dir]`; the default output
//! directory is the repository's git-ignored `.local/gallery/imaging-gallery`,
//! which survives `cargo clean`. Masks are written as coverage, 0 black and
//! 1 white.

use std::f64::consts::{PI, TAU};
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use dapple_field::hash::{hash, unit_f64};
use dapple_field::program::{Op, ProgramBuilder};
use dapple_field::raster::Region;
use dapple_field::{Basis, Domain, FractalParams};
use dapple_imaging::imaging::kurbo::{Affine, BezPath, Cap, Circle, Point, RoundedRect, Stroke};
use dapple_imaging::imaging::peniko::Color;
use dapple_imaging::imaging::{Painter, record::Scene};
use dapple_imaging::{coverage_image, rasterize};
use dapple_raster::seam::{Axis, seam};
use dapple_raster::{Raster, Realization, realize};
use glam::Vec2;

const SEED: u64 = 7;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args_os().nth(1).map_or_else(
        || repository().join(".local/gallery/imaging-gallery"),
        PathBuf::from,
    );
    std::fs::create_dir_all(&out)?;

    // A lobed leaf on a 1 × 1.5 plane region: the blade and petiole as one
    // opacity mask, the veins as a second, stroked.
    let leaf_region = Region {
        origin: Vec2::ZERO,
        size: Vec2::new(1.0, 1.5),
    };
    let leaf_grid = Realization::region(leaf_region, 256, 384)?;
    let (blade, veins) = leaf();
    write_mask(&out, "leaf-opacity", &rasterize(&blade, leaf_grid)?)?;
    write_mask(&out, "leaf-veins", &rasterize(&veins, leaf_grid)?)?;

    // Running-bond bricks on a periodic tile; the courses' offsets carry
    // bricks across the tile's edges, where they wrap.
    let torus = Domain::periodic(1, 1).expect("valid period");
    let bricks = bricks();
    let tile = rasterize(&bricks, Realization::period(torus, 256, 256)?)?;
    write_mask(&out, "brick-tiles-2x2", &tile_2x2(&tile)?)?;

    // A decal: a star inside a ring, on a plane region.
    let decal_grid = Realization::region(
        Region {
            origin: Vec2::splat(-1.0),
            size: Vec2::splat(2.0),
        },
        256,
        256,
    )?;
    write_mask(&out, "decal", &rasterize(&decal(), decal_grid)?)?;

    // The bricks as a field input: the coverage image, sampled by a program
    // that shades each brick with noise. Realized coarsely, the image's
    // mips give each texel the coverage of its whole area.
    let image = coverage_image(&bricks, Realization::period(torus, 256, 256)?)?;
    let mut b = ProgramBuilder::new();
    let mask = b.add(Op::Sample { image })?;
    let noise = b.add(Op::Fractal {
        basis: Basis::Gradient,
        domain: torus,
        frequency: [8.0, 8.0],
        seed: SEED,
        params: FractalParams::default(),
    })?;
    let shade = b.add(Op::Remap {
        input: noise,
        from: [-1.0, 1.0],
        to: [0.55, 1.0],
    })?;
    let bricks_field = b.add(Op::Mul { a: mask, b: shade })?;
    let program = b.finish(bricks_field)?;
    for size in [256, 32] {
        let raster = realize(&program, Realization::period(torus, size, size)?)?;
        write_mask(
            &out,
            &format!("brick-field-{size}-2x2"),
            &tile_2x2(&raster)?,
        )?;
    }
    Ok(())
}

/// A lobed oak-like blade with its petiole, and its veins.
fn leaf() -> (Scene, Scene) {
    // Half-width along the midrib, t from the base (0) to the tip (1): a
    // broad blade with five lobes a side.
    let half_width = |t: f64| {
        let blade = (PI * t).sin().powf(0.7) * (1.0 - 0.35 * t);
        let lobes = 0.72 + 0.28 * (TAU * 5.0 * t - 0.6).cos();
        0.36 * blade * lobes
    };
    let (base, tip) = (0.35, 1.42);
    let at = |t: f64, side: f64| Point::new(0.5 + side * half_width(t), base + (tip - base) * t);
    let steps = 240;
    let mut outline = BezPath::new();
    outline.move_to(at(0.0, 1.0));
    for i in 1..=steps {
        outline.line_to(at(f64::from(i) / f64::from(steps), 1.0));
    }
    for i in (0..steps).rev() {
        outline.line_to(at(f64::from(i) / f64::from(steps), -1.0));
    }
    outline.close_path();

    let mut petiole = BezPath::new();
    petiole.move_to((0.5, 0.06));
    petiole.quad_to((0.49, 0.2), (0.5, base + 0.02));
    let petiole_stroke = Stroke::new(0.022).with_caps(Cap::Round);

    let mut blade = Scene::new();
    {
        let mut p = Painter::new(&mut blade);
        p.fill(&outline, Color::WHITE).draw();
        p.stroke(&petiole, &petiole_stroke, Color::WHITE).draw();
    }

    // The midrib, and a thinner vein from it toward each lobe tip.
    let mut veins = Scene::new();
    {
        let mut p = Painter::new(&mut veins);
        let mut midrib = BezPath::new();
        midrib.move_to((0.5, base));
        midrib.line_to((0.5, tip - 0.03));
        p.stroke(
            &midrib,
            &Stroke::new(0.012).with_caps(Cap::Round),
            Color::WHITE,
        )
        .draw();
        let secondary = Stroke::new(0.006).with_caps(Cap::Round);
        for lobe in 0..5 {
            // Lobe tips sit where the lobe term peaks.
            let t = (f64::from(lobe) + 0.6 / TAU) / 5.0;
            if !(0.05..0.95).contains(&t) {
                continue;
            }
            for side in [-1.0, 1.0] {
                let root = Point::new(0.5, base + (tip - base) * (t - 0.06).max(0.0));
                let end = at(t, side * 0.9);
                let mut vein = BezPath::new();
                vein.move_to(root);
                vein.quad_to(
                    Point::new(0.5 + side * 0.4 * half_width(t), end.y - 0.01),
                    end,
                );
                p.stroke(&vein, &secondary, Color::WHITE).draw();
            }
        }
    }
    // Images put row 0 at the top: flip, so the tip points up.
    let upright = |scene: &Scene| {
        let mut flipped = Scene::new();
        flipped.append_transformed(scene, Affine::new([1.0, 0.0, 0.0, -1.0, 0.0, 1.5]));
        flipped
    };
    (upright(&blade), upright(&veins))
}

/// Eight courses of four bricks on a unit tile, each course offset by a
/// random fraction of a brick, with rounded corners and mortar gaps.
fn bricks() -> Scene {
    let (courses, per_course, mortar) = (8_u32, 4_u32, 0.012);
    let (length, height) = (1.0 / f64::from(per_course), 1.0 / f64::from(courses));
    let mut scene = Scene::new();
    let mut p = Painter::new(&mut scene);
    for course in 0..courses {
        let bond = if course % 2 == 0 { 0.0 } else { 0.5 };
        let jitter = 0.15 * (unit_f64(hash(SEED, &[1, u64::from(course)])) - 0.5);
        let offset = (bond + jitter) * length;
        let y0 = f64::from(course) * height;
        for brick in 0..per_course {
            let x0 = offset + f64::from(brick) * length;
            let shape = RoundedRect::new(
                x0 + mortar / 2.0,
                y0 + mortar / 2.0,
                x0 + length - mortar / 2.0,
                y0 + height - mortar / 2.0,
                0.012,
            );
            p.fill(shape, Color::WHITE).draw();
        }
    }
    scene
}

/// A five-pointed star inside a ring, centered on the origin.
fn decal() -> Scene {
    let mut star = BezPath::new();
    for i in 0..10 {
        let radius = if i % 2 == 0 { 0.62 } else { 0.25 };
        let angle = f64::from(i) * PI / 5.0 - PI / 2.0;
        let point = Point::new(radius * angle.cos(), radius * angle.sin());
        if i == 0 {
            star.move_to(point);
        } else {
            star.line_to(point);
        }
    }
    star.close_path();
    let mut scene = Scene::new();
    let mut p = Painter::new(&mut scene);
    p.fill(&star, Color::WHITE).draw();
    p.stroke(
        Circle::new((0.0, 0.0), 0.8),
        &Stroke::new(0.08),
        Color::WHITE,
    )
    .draw();
    scene
}

/// The raster repeated 2 × 2, to show that wrapping masks tile.
/// `raster` repeated 2 × 2. Anything tiled here is declared periodic, so it
/// must be seamless along both axes.
fn tile_2x2(raster: &Raster) -> Result<Raster, Box<dyn std::error::Error>> {
    for axis in [Axis::X, Axis::Y] {
        let s = seam(raster, axis);
        if !s.is_seamless() {
            return Err(format!("a declared-periodic preview has a seam: {s:?}").into());
        }
    }
    let (w, h) = (raster.width(), raster.height());
    let mut values = Vec::with_capacity(4 * (w * h) as usize);
    for y in 0..2 * i64::from(h) {
        for x in 0..2 * i64::from(w) {
            values.push(raster.at(x, y));
        }
    }
    Ok(Raster::from_values(
        2 * w,
        2 * h,
        raster.origin(),
        raster.texel(),
        raster.edge(),
        values,
    )?)
}

/// Writes coverage as 8-bit grayscale, 0 black and 1 white.
fn write_mask(dir: &Path, name: &str, mask: &Raster) -> Result<(), Box<dyn std::error::Error>> {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "coverage is clamped to [0, 1]"
    )]
    let pixels: Vec<u8> = mask
        .values()
        .iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect();
    let path = dir.join(format!("{name}.png"));
    let mut encoder = png::Encoder::new(
        BufWriter::new(File::create(&path)?),
        mask.width(),
        mask.height(),
    );
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&pixels)?;
    let covered = mask.values().iter().sum::<f32>() / mask.values().len() as f32;
    println!("{} ({:.1}% covered)", path.display(), covered * 100.0);
    Ok(())
}

/// The repository root: this crate's manifest sits two levels below it.
fn repository() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the crate lives two levels below the repository root")
}
