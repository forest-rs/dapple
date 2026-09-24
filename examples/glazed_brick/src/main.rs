// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Builds glazed-brick assets from `dapple_library`'s material modules and
//! writes their maps and review images.
//!
//! Run with `cargo run --release -p glazed_brick -- [asset] [output-dir]`,
//! where `asset` is `wall` (the default: a weathered Victorian glazed brick
//! wall, cream over a green band and a brown dado) or `sill` (a stone sill
//! bedded over green and oxblood glazed brick, streaked below). The default
//! output is the repository's git-ignored `.local/gallery/glazed-brick` or
//! `.local/gallery/stone-sill`. Images show one 1 m period with domain `+y`
//! up.
//!
//! `maps/` holds the material packed by `dapple_encode` for the glTF
//! profile: `base_color.png` (sRGB), `normal.png` (tangent-space, `+Y`
//! toward the image top), `orm.png` (occlusion, roughness, metalness),
//! `clearcoat.png` (R coat weight, G coat roughness), and, outside glTF,
//! `coat_color.png` (sRGB coat tint, which glTF cannot carry) and
//! `height.png`, 16-bit height over the range `height.txt` gives in meters.
//! Rows run from domain `y = 0` at the image top, as glTF reads textures.
//! `tools/render.py` renders them in Blender as a displaced wall.
//!
//! `diagnostics.txt` lists every module instance by path and every
//! material operation's approximation report, attributed to the instance
//! that made it; `report.json` is `dapple_lab`'s machine-readable report,
//! and the run fails when it does.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use dapple_encode::{
    Edge, Filter, Image, PackSettings, PixelFormat, Profile, data_mips, encode_data, pack,
};
use dapple_field::Value;
use dapple_library::glazed_brick::{BODY, DIRT, GLAZE, MORTAR, SALT, STONE};
use dapple_library::modules::{GlazedBrickWall, StoneSill};
use dapple_material::module::{Bind, Context, Module};
use dapple_material::resource::NoResources;
use dapple_material::{Aux, ChannelId, Grid, Material, Param, lower};
use glam::{Vec2, Vec3};

const SIZE: u32 = 2048;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let asset = args.next().unwrap_or_else(|| String::from("wall"));
    let (module, gallery): (&dyn Module, &str) = match asset.as_str() {
        "wall" => (&GlazedBrickWall, "glazed-brick"),
        "sill" => (&StoneSill, "stone-sill"),
        other => return Err(format!("unknown asset {other:?}: wall or sill").into()),
    };
    let out = args.next().map_or_else(
        || repository().join(".local/gallery").join(gallery),
        PathBuf::from,
    );
    std::fs::create_dir_all(&out)?;

    #[expect(clippy::cast_precision_loss, reason = "a small grid size")]
    let texel = Vec2::splat(1.0 / SIZE as f32);
    let grid = Grid {
        width: SIZE,
        height: SIZE,
        origin: Vec2::ZERO,
        texel,
        edge: Edge::Wrap,
    };
    let start = std::time::Instant::now();
    let mut cx = Context::new(grid, &NoResources);
    let mut outputs = cx.instantiate(module, &asset, Bind::new())?;
    let material = outputs
        .take_material("material")
        .ok_or("the asset returned no material")?;
    println!("built {asset} at {SIZE}² in {:.2?}", start.elapsed());
    let diagnostics = cx.into_diagnostics();
    std::fs::write(out.join("diagnostics.txt"), diagnostics.to_string())?;
    print!("{diagnostics}");

    write_previews(&out, &material)?;
    write_maps(&out.join("maps"), &material)?;

    // The lab's report: ranges, invalid values, seams against the tiling
    // promise (checked across the wrap when the asset was built), lowering
    // losses, and a material-wide transform that must move every channel.
    let mut report = dapple_lab::material::material_report(&asset, &material);
    report.merge(
        "relationships",
        dapple_lab::material::transform_check(&material),
    );
    std::fs::write(out.join("report.json"), report.to_json())?;
    if !report.passed() {
        for f in report.failures() {
            eprintln!("FAILED: {f:?}");
        }
        return Err("the lab report has failures".into());
    }
    Ok(())
}

/// Writes linear RGB rows (top row first) as an 8-bit PNG, sRGB-encoded
/// when `srgb`.
fn write_png(path: &Path, width: u32, height: u32, pixels: &[[f32; 3]], srgb: bool) -> Result<()> {
    let encode = |c: f32| {
        let c = c.clamp(0.0, 1.0);
        let s = if !srgb {
            c
        } else if c <= 0.003_130_8 {
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
    let bytes: Vec<u8> = pixels.iter().flatten().map(|&c| encode(c)).collect();
    let mut encoder = png::Encoder::new(BufWriter::new(File::create(path)?), width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&bytes)?;
    println!("{}", path.display());
    Ok(())
}

/// Texel values in image order: domain `+y` up, so the last row first.
fn image_rows(m: &Material, f: impl Fn(usize) -> [f32; 3]) -> Vec<[f32; 3]> {
    let (w, h) = (m.grid().width as usize, m.grid().height as usize);
    let mut rows = Vec::with_capacity(w * h);
    for row in 0..h {
        let y = h - 1 - row;
        for x in 0..w {
            rows.push(f(y * w + x));
        }
    }
    rows
}

fn vec3(v: Value) -> Vec3 {
    match v {
        Value::Vector3(v) => v,
        other => Vec3::splat(other.component(0).unwrap_or(0.0)),
    }
}

fn scalar(m: &Material, p: Param, i: usize) -> f32 {
    m.value(ChannelId::Param(p), i).component(0).unwrap_or(0.0)
}

fn write_previews(out: &Path, m: &Material) -> Result<()> {
    let (w, h) = (m.grid().width, m.grid().height);
    let color = image_rows(m, |i| {
        vec3(m.value(ChannelId::Param(Param::BaseColor), i)).to_array()
    });
    write_png(&out.join("base-color.png"), w, h, &color, true)?;
    let rough = image_rows(m, |i| [scalar(m, Param::SpecularRoughness, i); 3]);
    write_png(&out.join("roughness.png"), w, h, &rough, false)?;
    let coat = image_rows(m, |i| {
        (vec3(m.value(ChannelId::Param(Param::CoatColor), i)) * scalar(m, Param::CoatWeight, i))
            .to_array()
    });
    write_png(&out.join("coat.png"), w, h, &coat, true)?;
    let height = m.height()?;
    let (lo, hi) = height
        .values()
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let heights = image_rows(m, |i| [(height.values()[i] - lo) / (hi - lo); 3]);
    write_png(&out.join("height.png"), w, h, &heights, false)?;
    let surfaces = image_rows(m, |i| match m.value(ChannelId::Aux(Aux::Surface), i) {
        Value::Id(GLAZE) => [0.1, 0.55, 0.6],
        Value::Id(BODY) => [0.9, 0.5, 0.2],
        Value::Id(MORTAR) => [0.35, 0.35, 0.35],
        Value::Id(DIRT) => [0.1, 0.08, 0.05],
        Value::Id(SALT) => [0.95, 0.95, 0.9],
        Value::Id(STONE) => [0.7, 0.6, 0.3],
        _ => [1.0, 0.0, 1.0],
    });
    write_png(&out.join("surface.png"), w, h, &surfaces, false)?;
    Ok(())
}

/// Packs the material for the glTF profile and writes it as PNG files.
fn write_maps(dir: &Path, m: &Material) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let (maps, lowering) = lower::maps(m)?;
    println!(
        "lowered: normal from height {}, dropped {:?}",
        lowering.normal_from_height, lowering.dropped
    );
    let bundle = pack(&maps, Profile::Gltf, &PackSettings::default())?;
    println!("packing could not carry {:?}", bundle.report.unsupported);
    let mut textures: Vec<_> = bundle.textures.iter().collect();

    let grid = m.grid();
    let height = m.height()?;
    let (lo, hi) = height
        .values()
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let unit: Vec<f32> = height
        .values()
        .iter()
        .map(|&v| (v - lo) / (hi - lo))
        .collect();
    let chain = data_mips(
        &Image::new(grid.width, grid.height, 1, Edge::Wrap, unit)?,
        Filter::Box,
    );
    let height_texture = encode_data("height", &chain, PixelFormat::R16Unorm)?;
    textures.push(&height_texture);
    // The coat's tint, which glTF cannot carry, for renderers that can.
    let tint: Vec<f32> = (0..grid.len())
        .flat_map(|i| {
            let c = vec3(m.value(ChannelId::Param(Param::CoatColor), i));
            [c.x, c.y, c.z, 1.0]
        })
        .collect();
    let tint_chain = data_mips(
        &Image::new(grid.width, grid.height, 4, Edge::Wrap, tint)?,
        Filter::Box,
    );
    let tint_texture = encode_data("coat_color", &tint_chain, PixelFormat::Rgba8Srgb)?;
    textures.push(&tint_texture);
    for texture in textures {
        let path = dir.join(format!("{}.png", texture.name));
        std::fs::write(&path, dapple_encode::png::write(texture)?)?;
        println!("{}", path.display());
    }
    std::fs::write(dir.join("height.txt"), format!("{lo} {hi}\n"))?;
    Ok(())
}

fn repository() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the crate lives two levels below the repository root")
}
