// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Writes preview images of `dapple_field` fields and `dapple_raster`
//! operations as PNG files.
//!
//! Run with `cargo run -p field_gallery -- [output-dir]`; the default output
//! directory is the repository's git-ignored `.local/gallery/field-gallery`,
//! which survives `cargo clean`. Each grayscale image is normalized to its
//! own value range, which is printed; color images are linear colors,
//! written sRGB-encoded.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use dapple_compress::{CompressSettings, Encoding, compress};
use dapple_encode::{Filter, Image, MaterialMaps, PackSettings, Profile, ktx2, pack};
use dapple_field::program::{
    ChartSample, FieldProgram, NodeId, Op, ProgramBuilder, ProgramError, SolidProgram,
};
use dapple_field::raster::{Grid, Region};
use dapple_field::{
    Basis, CellOutput, Cellular, Domain, DomainError, Footprint, Fractal, FractalKind,
    FractalParams, ImageLevel, Noise, SampleImage, ScalarField,
};
use dapple_raster::{
    AmbientOcclusion, DistanceTransform, Edge, GaussianBlur, HeightToNormal, Raster, RasterOp,
    Realization, realize,
};
use glam::{Vec2, Vec3};

use dapple_library::oak;

const SIZE: u32 = 256;
const SEED: u64 = 7;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args_os().nth(1).map_or_else(
        || repository().join(".local/gallery/field-gallery"),
        PathBuf::from,
    );
    std::fs::create_dir_all(&out)?;

    let plane = Domain::Plane;
    let square = Region {
        origin: Vec2::ZERO,
        size: Vec2::ONE,
    };

    let gradient = Noise::new(Basis::Gradient, plane, Vec2::splat(8.0), SEED)?;
    write_grid(
        &out,
        "gradient-noise",
        &Grid::sample(&gradient, square, SIZE, SIZE),
    )?;
    let value = Noise::new(Basis::Value, plane, Vec2::splat(8.0), SEED)?;
    write_grid(
        &out,
        "value-noise",
        &Grid::sample(&value, square, SIZE, SIZE),
    )?;

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
    write_grid(
        &out,
        "fbm-point-sampled",
        &sample_points(&fine, square, SIZE),
    )?;
    write_grid(
        &out,
        "fbm-footprint-filtered",
        &Grid::sample(&fine, square, SIZE, SIZE),
    )?;
    println!(
        "fine fBm: {} of 12 octaves active at a {SIZE}-texel footprint",
        fine.active_octaves(Footprint::new(1.0 / SIZE as f32).expect("finite"))
    );

    bark(&out)?;
    wood(&out)?;
    Ok(())
}

/// A tileable bark study: a field program for height, realized once, then
/// raster operations on it. Each image shows the tile 2 × 2.
fn bark(out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let torus = Domain::periodic(1, 1).expect("valid period");
    let program = oak::bark_height(torus)?;
    println!(
        "bark program {} ({} nodes)",
        program.fingerprint(),
        program.len()
    );
    let stats = program.evaluation_stats();
    let start = std::time::Instant::now();
    let height = realize(&program, Realization::period(torus, SIZE, SIZE)?)?;
    println!(
        "bark realized in {:?}: {} node evaluations per texel ({} recursively), {} contexts",
        start.elapsed(),
        stats.instances,
        stats.tree_evaluations,
        stats.contexts
    );
    write_raster(out, "bark-height-2x2", &height)?;
    mip_previews(out, &height)?;
    warped_bark(out, torus, &program, &height)?;

    // Heights are in [0, 1]; 1 means 15 mm of relief on a 1 m tile.
    let relief = 0.015;
    let normals = HeightToNormal { scale: relief }.apply(&height)?;
    write_normals(out, "bark-normal-2x2", &normals)?;
    let ao = AmbientOcclusion {
        radius: 0.02,
        directions: 12,
        scale: relief,
    }
    .apply(&height)?;
    write_raster(out, "bark-ao-2x2", &ao)?;
    let blur = GaussianBlur { sigma: 0.006 };
    println!(
        "blur footprint at {SIZE} texels: {:?}",
        blur.footprint(height.texel())
    );
    write_raster(out, "bark-blurred-2x2", &blur.apply(&height)?)?;

    // Distance from the fissure floors, in domain units.
    let fissures = Raster::from_values(
        height.width(),
        height.height(),
        height.origin(),
        height.texel(),
        Edge::Wrap,
        height
            .values()
            .iter()
            .map(|h| if *h < 0.12 { 1.0 } else { 0.0 })
            .collect(),
    )?;
    let distance = DistanceTransform { threshold: 0.5 }.apply(&fissures)?;
    write_raster(out, "bark-fissure-distance-2x2", &distance)?;

    let color = oak::bark_color(torus, &program)?;
    let base_color: Vec<[f32; 3]> = {
        let channels = (0..3)
            .map(|c| -> Result<Raster, Box<dyn std::error::Error>> {
                Ok(realize(
                    &color.channel(c)?,
                    Realization::period(torus, SIZE, SIZE)?,
                )?)
            })
            .collect::<Result<Vec<Raster>, _>>()?;
        (0..channels[0].values().len())
            .map(|i| [0, 1, 2].map(|c| channels[c].values()[i]))
            .collect()
    };
    let tiled: Vec<[f32; 3]> = (0..4 * SIZE * SIZE)
        .map(|i| {
            let (x, y) = (i % (2 * SIZE) % SIZE, i / (2 * SIZE) % SIZE);
            base_color[(y * SIZE + x) as usize]
        })
        .collect();
    write_color(out, "bark-base-color-2x2", 2 * SIZE, 2 * SIZE, &tiled)?;

    bark_set(out, &height, &base_color, &normals, &ao)
}

/// Writes the first mip levels of the bark height, each scaled back up to the
/// base size with nearest texels, as `bark-height-mip<n>.png`.
fn mip_previews(out: &Path, height: &Raster) -> Result<(), Box<dyn std::error::Error>> {
    let image = Image::new(
        height.width(),
        height.height(),
        1,
        Edge::Wrap,
        height.values().to_vec(),
    )?;
    let chain = dapple_encode::data_mips(&image, Filter::Kaiser);
    for (n, level) in chain.levels().iter().enumerate().skip(1).take(4) {
        let (w, h) = (level.width(), level.height());
        let values = (0..SIZE * SIZE)
            .map(|i| {
                let (x, y) = (i % SIZE * w / SIZE, i / SIZE * h / SIZE);
                level.texel(x, y)[0]
            })
            .collect();
        write_grid(
            out,
            &format!("bark-height-mip{n}"),
            &Grid {
                width: SIZE,
                height: SIZE,
                values,
            },
        )?;
    }
    Ok(())
}

/// Samples the realized bark height back as a field, warps it by noise, and
/// realizes it again: `bark-warped-2x2.png`. Prints the warp's static reach.
fn warped_bark(
    out: &Path,
    torus: Domain,
    program: &FieldProgram,
    height: &Raster,
) -> Result<(), Box<dyn std::error::Error>> {
    let level = ImageLevel::new(
        height.width(),
        height.height(),
        height.texel(),
        height.values().to_vec(),
    )?;
    let image = SampleImage::new(torus, Vec2::ZERO, vec![level], program.fingerprint())?;
    let mut b = ProgramBuilder::new();
    let sampled = b.add(Op::Sample { image })?;
    let displacement = |b: &mut ProgramBuilder, seed| {
        b.add(Op::Noise {
            basis: Basis::Gradient,
            domain: torus,
            frequency: [3.0, 3.0],
            seed,
        })
    };
    let dx = displacement(&mut b, 21)?;
    let dy = displacement(&mut b, 22)?;
    let amount = 0.02;
    let warp = b.add(Op::Warp {
        input: sampled,
        dx,
        dy,
        amount,
    })?;
    let warped = b.finish(warp)?;
    let bounds = warped.node_bounds(dx);
    println!(
        "warped bark: displacement range {:?}, slope bound {:?}, so a change reaches {:.3} units",
        bounds.range,
        bounds.slope,
        bounds.max_abs().unwrap_or(f32::INFINITY) * amount
    );
    let again = realize(&warped, Realization::period(torus, SIZE, SIZE)?)?;
    write_raster(out, "bark-warped-2x2", &again)
}

/// Packs the bark study as a material and writes its textures, with full mip
/// chains, as KTX2 and level 0 as PNG, for Lightweald and glTF.
fn bark_set(
    out: &Path,
    height: &Raster,
    base_color: &[[f32; 3]],
    normals: &Raster<[f32; 3]>,
    ao: &Raster,
) -> Result<(), Box<dyn std::error::Error>> {
    let edge = height.edge();
    let base_color: Vec<f32> = base_color.iter().flatten().copied().collect();
    let roughness: Vec<f32> = height
        .values()
        .iter()
        .map(|&h| 0.55 + 0.35 * (1.0 - h.clamp(0.0, 1.0)))
        .collect();
    let image = |channels, values| Image::new(SIZE, SIZE, channels, edge, values);
    let maps = MaterialMaps {
        base_color: Some(image(3, base_color)?),
        normal: Some(Image::from(normals)),
        specular_roughness: Some(image(1, roughness)?),
        occlusion: Some(Image::from(ao)),
        ..MaterialMaps::default()
    };
    let settings = PackSettings {
        filter: Filter::Kaiser,
        ..PackSettings::default()
    };
    for (profile, dir) in [(Profile::Lightweald, "lightweald"), (Profile::Gltf, "gltf")] {
        let bundle = pack(&maps, profile, &settings)?;
        let dir = out.join("bark-set").join(dir);
        std::fs::create_dir_all(&dir)?;
        for texture in &bundle.textures {
            std::fs::write(
                dir.join(format!("{}.ktx2", texture.name)),
                ktx2::write(texture),
            )?;
            std::fs::write(
                dir.join(format!("{}.png", texture.name)),
                dapple_encode::png::write(texture)?,
            )?;
        }
        // Lightweald's pool encodings, with dapple's own mips kept.
        if profile == Profile::Lightweald {
            for (encoding, name) in [(Encoding::Bc, "bc"), (Encoding::Astc, "astc")] {
                let encoded = dir.join(name);
                std::fs::create_dir_all(&encoded)?;
                for texture in &bundle.textures {
                    let file = compress(texture, CompressSettings::new(encoding))?;
                    std::fs::write(encoded.join(format!("{}.ktx2", texture.name)), file)?;
                }
            }
        }
        let variance: Vec<String> = bundle
            .report
            .normal_variance
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect();
        println!(
            "bark set ({dir:?}): {} textures, {} levels, normal variance per level [{}]",
            bundle.textures.len(),
            bundle.textures[0].levels.len(),
            variance.join(", ")
        );
    }
    Ok(())
}

/// Solid oak-like wood around a trunk along z, in meters, previewed on three
/// sawn faces (slices) and on a log's surface (a cylinder chart).
fn wood(out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let channels =
        |finish: &dyn Fn(&mut ProgramBuilder, NodeId) -> Result<NodeId, ProgramError>| {
            (0..3)
                .map(|index| {
                    let mut b = ProgramBuilder::new();
                    let color = oak::wood_color(&mut b)?;
                    let channel = b.add(Op::Component {
                        input: color,
                        index,
                    })?;
                    let output = finish(&mut b, channel)?;
                    Ok((b, output))
                })
                .collect::<Result<Vec<_>, ProgramError>>()
        };
    let unit = Region {
        origin: Vec2::ZERO,
        size: Vec2::ONE,
    };
    // Sawn faces: planar slices through the solid, 12 cm across at about 4
    // texels per mm, since the thin rays and sharp ring ends are not
    // band-limited.
    let faces = [
        // End grain: a cross-section from near the pith outward.
        (
            "wood-end-grain",
            [-0.03, -0.03, 0.2],
            [0.12, 0.0, 0.0],
            [0.0, 0.12, 0.0],
            1,
        ),
        // Flat sawn: a plane parallel to the axis, 7 cm off it.
        (
            "wood-flat-sawn",
            [-0.06, 0.07, 0.0],
            [0.0, 0.0, 0.12],
            [0.12, 0.0, 0.0],
            3,
        ),
        // Quarter sawn: nearly through the axis, 6° off radial as real
        // boards are, so the face crosses rays at a shallow angle and they
        // show as flecks.
        (
            "wood-quarter-sawn",
            [0.02, -0.004, 0.0],
            [0.0, 0.0, 0.12],
            [0.1193, 0.0126, 0.0],
            3,
        ),
    ];
    for (name, origin, u, v, aspect) in faces {
        let rgb = channels(&|b, channel| {
            b.add(Op::Slice {
                input: channel,
                origin,
                u,
                v,
                domain: Domain::Plane,
            })
        })?
        .into_iter()
        .map(|(b, output)| {
            let program = b.finish(output)?;
            let region = Region {
                size: Vec2::new(aspect as f32, 1.0),
                ..unit
            };
            Ok(Grid::sample(&program, region, 2 * aspect * SIZE, 2 * SIZE).values)
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;
        let pixels: Vec<[f32; 3]> = (0..rgb[0].len())
            .map(|i| [rgb[0][i], rgb[1][i], rgb[2][i]])
            .collect();
        write_color(out, name, 2 * aspect * SIZE, 2 * SIZE, &pixels)?;
    }

    let solids = channels(&|_, channel| Ok(channel))?
        .into_iter()
        .map(|(b, output)| b.finish_solid(output))
        .collect::<Result<Vec<SolidProgram>, ProgramError>>()?;
    println!(
        "wood program {} ({} nodes)",
        solids[0].fingerprint(),
        solids[0].program().len()
    );
    // A log of radius 11 cm, its surface unrolled: x is the angle around
    // the axis, y runs along it at the same scale.
    let radius = 0.11;
    let (width, height) = (3 * SIZE, SIZE);
    let texel = core::f32::consts::TAU * radius / width as f32;
    let footprint = Footprint::new(texel).expect("finite");
    let samples: Vec<ChartSample> = (0..width * height)
        .map(|i| {
            let (x, y) = (i % width, i / width);
            let angle = (x as f32 + 0.5) / width as f32 * core::f32::consts::TAU;
            ChartSample {
                position: Vec3::new(
                    radius * angle.cos(),
                    radius * angle.sin(),
                    (y as f32 + 0.5) * texel,
                ),
                footprint,
            }
        })
        .collect();
    let chart = chart_color(&solids, &samples);
    write_color(out, "wood-log-chart", width, height, &chart)?;

    // The same log seen from the side, shaded: each visible point evaluated
    // with a footprint widened by the surface's slant.
    let (width, height) = (SIZE, 2 * SIZE);
    let pixel = 2.4 * radius / width as f32;
    let to_light = Vec3::new(-0.5, -1.0, 0.4).normalize();
    let mut samples = Vec::new();
    let mut shading = Vec::new();
    for i in 0..width * height {
        let (x, y) = (i % width, i / width);
        let px = (x as f32 + 0.5) * pixel - 1.2 * radius;
        let z = (height - y) as f32 * pixel;
        if px.abs() < radius {
            let normal = Vec3::new(px / radius, -(1.0 - (px / radius).powi(2)).sqrt(), 0.0);
            let slant = (-normal.y).max(0.05);
            samples.push(ChartSample {
                position: normal * radius + Vec3::Z * z,
                footprint: Footprint::new(pixel / slant).expect("finite"),
            });
            shading.push(Some(0.2 + 0.8 * normal.dot(to_light).max(0.0)));
        } else {
            shading.push(None);
        }
    }
    let mut colors = chart_color(&solids, &samples).into_iter();
    let side: Vec<[f32; 3]> = shading
        .into_iter()
        .map(|shade| match shade {
            Some(shade) => colors
                .next()
                .expect("one color per sample")
                .map(|c| c * shade),
            None => [0.8; 3],
        })
        .collect();
    write_color(out, "wood-log-side", width, height, &side)?;
    Ok(())
}

/// Evaluates three solid channel programs at chart samples, as colors.
fn chart_color(channels: &[SolidProgram], samples: &[ChartSample]) -> Vec<[f32; 3]> {
    let mut values = vec![vec![0.0; samples.len()]; 3];
    for (program, out) in channels.iter().zip(&mut values) {
        program.eval_chart(samples, out);
    }
    (0..samples.len())
        .map(|i| [values[0][i], values[1][i], values[2][i]])
        .collect()
}

/// Writes linear colors as an sRGB-encoded PNG.
fn write_color(
    dir: &Path,
    name: &str,
    width: u32,
    height: u32,
    colors: &[[f32; 3]],
) -> Result<(), Box<dyn std::error::Error>> {
    let encode = |c: f32| {
        let c = c.clamp(0.0, 1.0);
        let s = if c <= 0.003_130_8 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "s is in [0, 1]"
        )]
        let byte = (s * 255.0 + 0.5) as u8;
        byte
    };
    let pixels: Vec<u8> = colors.iter().flatten().map(|&c| encode(c)).collect();
    let path = dir.join(format!("{name}.png"));
    let mut encoder = png::Encoder::new(BufWriter::new(File::create(&path)?), width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&pixels)?;
    println!("{}", path.display());
    Ok(())
}

fn raster_grid(raster: &Raster) -> Grid {
    Grid {
        width: raster.width(),
        height: raster.height(),
        values: raster.values().to_vec(),
    }
}

fn write_raster(dir: &Path, name: &str, raster: &Raster) -> Result<(), Box<dyn std::error::Error>> {
    write_grid(dir, name, &tile_2x2(&raster_grid(raster)))
}

/// Writes normals as RGB, `0.5 + 0.5 * n` per channel, tiled 2 × 2.
fn write_normals(
    dir: &Path,
    name: &str,
    normals: &Raster<[f32; 3]>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = (normals.width(), normals.height());
    let mut pixels = Vec::with_capacity((4 * w * h * 3) as usize);
    for y in 0..2 * i64::from(h) {
        for x in 0..2 * i64::from(w) {
            for c in normals.at(x, y) {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "clamped to [0, 255]"
                )]
                pixels.push(((c * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            }
        }
    }
    let path = dir.join(format!("{name}.png"));
    let mut encoder = png::Encoder::new(BufWriter::new(File::create(&path)?), 2 * w, 2 * h);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&pixels)?;
    println!("{}", path.display());
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

/// The repository root: this crate's manifest sits two levels below it.
fn repository() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the crate lives two levels below the repository root")
}
