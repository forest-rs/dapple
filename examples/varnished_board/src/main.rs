// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Bakes the actual varnished-board JSON package, at its defaults and with
//! its `dark_and_worn` preset. See the example README for rendering commands.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use dapple_encode::{Edge, PackSettings, Profile, pack};
use dapple_library::packages::{VARNISHED_BOARD, registry};
use dapple_material::module::{Bind, Context};
use dapple_material::resource::NoResources;
use dapple_material::{Grid, Material, lower};
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

/// Packs the material for the glTF profile and writes it as PNG files, with
/// the raw profile's textures for whatever glTF reports it cannot carry.
fn write_maps(dir: &Path, material: &Material) -> Result<()> {
    let (maps, lowering) = lower::maps(material)?;
    // The coat ripple is below one 8-bit normal step. Keep 16-bit normals
    // so packing does not turn it into sign-only noise.
    let settings = PackSettings {
        sixteen_bit_normals: true,
        ..PackSettings::default()
    };
    let gltf = pack(&maps, Profile::Gltf, &settings)?;
    let raw = pack(&maps, Profile::Raw, &settings)?;
    let mut textures: Vec<_> = gltf.textures.iter().collect();
    let mut notes = String::new();
    for name in &gltf.report.unsupported {
        match raw.texture(name) {
            Some(texture) => {
                textures.push(texture);
                writeln!(notes, "{name} is carried by the raw profile's {name}.png.")?;
            }
            None => writeln!(notes, "{name} is the constant {:?}.", raw.constant(name))?,
        }
    }
    for texture in textures {
        std::fs::write(
            dir.join(format!("{}.png", texture.name)),
            dapple_encode::png::write(texture)?,
        )?;
    }
    if let Some(height) = raw.placement("height") {
        let [lo, hi] = height.range;
        std::fs::write(dir.join("height.txt"), format!("{lo} {hi}\n"))?;
    }
    std::fs::write(
        dir.join("lowering.txt"),
        format!(
            "{lowering:#?}\nglTF constants: {:#?}\nPacking: {:#?}\n{notes}Blender uses IOR 1.5 for both base and coat; CoatDarkening has no exact Principled equivalent.\n",
            gltf.constants, gltf.report
        ),
    )?;
    Ok(())
}
