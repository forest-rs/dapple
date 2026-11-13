// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Bakes the actual varnished-board JSON package, at its defaults and with
//! its `dark_and_worn` preset. See the example README for rendering commands.

use std::path::{Path, PathBuf};

use dapple_encode::{
    Edge, Filter, Image, PackSettings, PixelFormat, Profile, data_mips, encode_data, pack,
    quantize_unorm16,
};
use dapple_library::packages::{VARNISHED_BOARD, registry};
use dapple_material::module::{Bind, Context};
use dapple_material::resource::NoResources;
use dapple_material::{ChannelId, Grid, Material, Param, lower};
use dapple_package::Package;
use glam::Vec2;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    let out = std::env::args().nth(1).map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.local/gallery/varnished-board"),
        PathBuf::from,
    );
    let package = Package::from_json(VARNISHED_BOARD)?;
    let module = registry().compile(&package)?;
    let worn = module.preset("dark_and_worn").ok_or("missing preset")?;
    // 80 × 20 cm, along the trunk in +x, across the board in +y.
    let grid = Grid {
        width: 1536,
        height: 384,
        origin: Vec2::new(0.0, -0.1),
        texel: Vec2::splat(0.8 / 1536.0),
        edge: Edge::Clamp,
    };
    for (name, bind) in [("default", Bind::new()), ("dark_and_worn", worn)] {
        let start = std::time::Instant::now();
        let dir = out.join(name);
        std::fs::create_dir_all(&dir)?;
        let mut cx = Context::new(grid, &NoResources);
        // Keep the instance path identical so both variants use the same grain.
        let material = cx
            .instantiate(&module, "board", bind)?
            .take_material("material")
            .ok_or("package returned no material")?;
        let diagnostics = cx.into_diagnostics();
        std::fs::write(dir.join("diagnostics.txt"), diagnostics.to_string())?;
        let report = dapple_lab::material::material_report(name, &material);
        std::fs::write(dir.join("report.json"), report.to_json())?;
        if !report.passed() {
            return Err(format!(
                "{name}: material report failed: {:?}",
                report.failures().collect::<Vec<_>>()
            )
            .into());
        }
        write_maps(&dir, &material)?;
        println!("{name}: baked and checked in {:.2?}", start.elapsed());
    }
    Ok(())
}

fn write_maps(dir: &Path, material: &Material) -> Result<()> {
    let (maps, lowering) = lower::maps(material)?;
    let bundle = pack(&maps, Profile::Gltf, &PackSettings::default())?;
    for texture in &bundle.textures {
        std::fs::write(
            dir.join(format!("{}.png", texture.name)),
            dapple_encode::png::write(texture)?,
        )?;
    }
    // glTF has no coat tint or independent coat normal. Carry both in
    // sidecar maps, using the same color and normal conventions as above.
    let grid = material.grid();
    let tint = maps.coat_color.as_ref().ok_or("missing coat tint")?;
    let tint = Image::new(
        grid.width,
        grid.height,
        4,
        grid.edge,
        tint.values()
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|c| [c[0], c[1], c[2], 1.0])
            .collect(),
    )?;
    let tint = encode_data(
        "coat_color",
        &data_mips(&tint, Filter::Box),
        PixelFormat::Rgba8Srgb,
    )?;
    std::fs::write(
        dir.join("coat_color.png"),
        dapple_encode::png::write(&tint)?,
    )?;
    // The coat ripple is below one 8-bit normal step. Keep 16-bit
    // precision so packing does not turn it into sign-only noise.
    let normals: Vec<u8> = (0..grid.len())
        .flat_map(|i| {
            let value = material.value(ChannelId::Param(Param::GeometryCoatNormal), i);
            [
                value.component(0).unwrap_or(0.0),
                -value.component(1).unwrap_or(0.0),
                value.component(2).unwrap_or(1.0),
            ]
        })
        .flat_map(|v| quantize_unorm16(v * 0.5 + 0.5).to_be_bytes())
        .collect();
    let file = std::io::BufWriter::new(std::fs::File::create(dir.join("coat_normal.png"))?);
    let mut encoder = png::Encoder::new(file, grid.width, grid.height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Sixteen);
    encoder.write_header()?.write_image_data(&normals)?;
    std::fs::write(
        dir.join("lowering.txt"),
        format!(
            "{lowering:#?}\nPacking: {:#?}\nCoatColor and GeometryCoatNormal are carried by sidecar PNG files.\nBlender uses IOR 1.5 for both base and coat; CoatDarkening has no exact Principled equivalent.\n",
            bundle.report
        ),
    )?;
    Ok(())
}
