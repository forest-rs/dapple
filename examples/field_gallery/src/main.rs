// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Writes preview images of `dapple_field` fields as PNG files.
//!
//! Run with `cargo run -p field_gallery -- [output-dir]`; the default output
//! directory is `target/field-gallery`. Each image is normalized to its own
//! value range, which is printed.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use dapple_field::raster::{Grid, Region};
use dapple_field::{
    Basis, CellOutput, Cellular, Domain, DomainError, Footprint, Fractal, FractalKind,
    FractalParams, Noise, ScalarField,
};
use glam::Vec2;

const SIZE: u32 = 256;
const SEED: u64 = 7;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args_os()
        .nth(1)
        .map_or_else(|| PathBuf::from("target/field-gallery"), PathBuf::from);
    std::fs::create_dir_all(&out)?;

    let plane = Domain::Plane;
    let square = Region {
        origin: Vec2::ZERO,
        size: Vec2::ONE,
    };

    let gradient = Noise::new(Basis::Gradient, plane, Vec2::splat(8.0), SEED)?;
    write_grid(&out, "gradient-noise", &Grid::sample(&gradient, square, SIZE, SIZE))?;
    let value = Noise::new(Basis::Value, plane, Vec2::splat(8.0), SEED)?;
    write_grid(&out, "value-noise", &Grid::sample(&value, square, SIZE, SIZE))?;

    let fbm = fractal(plane, FractalKind::Fbm, 4.0, 6)?;
    write_grid(&out, "fbm", &Grid::sample(&fbm, square, SIZE, SIZE))?;
    let ridged = fractal(plane, FractalKind::Ridged, 4.0, 6)?;
    write_grid(&out, "ridged", &Grid::sample(&ridged, square, SIZE, SIZE))?;

    let cells = Cellular::new(plane, Vec2::splat(8.0), 1.0, SEED)?;
    write_grid(
        &out,
        "cellular-border",
        &Grid::sample(&cells.output(CellOutput::Border), square, SIZE, SIZE),
    )?;
    write_grid(
        &out,
        "cellular-values",
        &Grid::sample(&cells.output(CellOutput::CellValue), square, SIZE, SIZE),
    )?;

    // Periodic fields sampled over exactly one period, then tiled 2 × 2: any
    // seam would show as a cross through the middle.
    let torus = Domain::periodic(1, 1).expect("valid period");
    let period = Region::period(torus).expect("periodic region");
    let periodic_fbm = fractal(torus, FractalKind::Fbm, 4.0, 6)?;
    write_grid(
        &out,
        "periodic-fbm-2x2",
        &tile_2x2(&Grid::sample(&periodic_fbm, period, SIZE / 2, SIZE / 2)),
    )?;
    let periodic_cells = Cellular::new(torus, Vec2::splat(6.0), 1.0, SEED)?;
    write_grid(
        &out,
        "periodic-cells-2x2",
        &tile_2x2(&Grid::sample(
            &periodic_cells.output(CellOutput::F2MinusF1),
            period,
            SIZE / 2,
            SIZE / 2,
        )),
    )?;

    // Twelve octaves from 16 cells per unit reach far past a 256-texel grid's
    // Nyquist limit: point sampling aliases, the texel footprint does not. A
    // high gain keeps the fine octaves strong enough to see the difference.
    let fine = Fractal::new(
        Basis::Gradient,
        plane,
        Vec2::splat(16.0),
        SEED,
        FractalParams {
            octaves: 12,
            gain: 0.85,
            ..FractalParams::default()
        },
    )?;
    write_grid(&out, "fbm-point-sampled", &sample_points(&fine, square, SIZE))?;
    write_grid(
        &out,
        "fbm-footprint-filtered",
        &Grid::sample(&fine, square, SIZE, SIZE),
    )?;
    println!(
        "fine fBm: {} of 12 octaves active at a {SIZE}-texel footprint",
        fine.active_octaves(Footprint::new(1.0 / SIZE as f32).expect("finite"))
    );
    Ok(())
}

fn fractal(
    domain: Domain,
    kind: FractalKind,
    frequency: f32,
    octaves: u8,
) -> Result<Fractal, DomainError> {
    Fractal::new(
        Basis::Gradient,
        domain,
        Vec2::splat(frequency),
        SEED,
        FractalParams {
            kind,
            octaves,
            ..FractalParams::default()
        },
    )
}

/// Samples texel centers with a point footprint, for the aliasing comparison.
fn sample_points(field: &impl ScalarField, region: Region, size: u32) -> Grid {
    let texel = region.size / size as f32;
    let mut values = Vec::with_capacity((size * size) as usize);
    for y in 0..size {
        for x in 0..size {
            let center = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            values.push(field.eval(region.origin + center * texel, Footprint::POINT));
        }
    }
    Grid {
        width: size,
        height: size,
        values,
    }
}

fn tile_2x2(grid: &Grid) -> Grid {
    let (w, h) = (grid.width as usize, grid.height as usize);
    let mut values = Vec::with_capacity(4 * w * h);
    for y in 0..2 * h {
        for x in 0..2 * w {
            values.push(grid.values[(y % h) * w + x % w]);
        }
    }
    Grid {
        width: grid.width * 2,
        height: grid.height * 2,
        values,
    }
}

fn write_grid(dir: &Path, name: &str, grid: &Grid) -> Result<(), Box<dyn std::error::Error>> {
    let (lo, hi) = grid
        .values
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let scale = if hi > lo { 1.0 / (hi - lo) } else { 0.0 };
    let pixels: Vec<u8> = grid
        .values
        .iter()
        .map(|&v| {
            let t = ((v - lo) * scale).clamp(0.0, 1.0);
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "t is clamped to [0, 1]"
            )]
            let byte = (t * 255.0 + 0.5) as u8;
            byte
        })
        .collect();
    let path = dir.join(format!("{name}.png"));
    let mut encoder = png::Encoder::new(
        BufWriter::new(File::create(&path)?),
        grid.width,
        grid.height,
    );
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&pixels)?;
    println!("{} [{lo:.4}, {hi:.4}]", path.display());
    Ok(())
}
