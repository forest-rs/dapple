// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Bakes the actual varnished-board JSON package, at its defaults and with
//! its `dark_and_worn` preset, and exports each through `dapple_export`.
//! See the example README for rendering commands.

use std::path::{Path, PathBuf};

use dapple_encode::{Edge, PackSettings};
use dapple_export::{Settings, Source};
use dapple_library::packages::{VARNISHED_BOARD, registry};
use dapple_material::Grid;
use dapple_material::module::{Bind, Context};
use dapple_material::resource::NoResources;
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
    // The coat ripple is below one 8-bit normal step. Keep 16-bit normals
    // so packing does not turn it into sign-only noise.
    let settings = Settings {
        pack: PackSettings {
            sixteen_bit_normals: true,
            ..PackSettings::default()
        },
        ..Settings::default()
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
        let source = Source {
            diagnostics: Some(&diagnostics),
            lab: Some(&report),
            ..Source::new(name, &material)
        };
        let exported = dapple_export::export(&source, &settings)?;
        exported.write_dir(&dir)?;
        for p in exported.parameters.iter().filter(|p| !p.notes.is_empty()) {
            println!("{name}: {}: {}", p.name, p.notes.join("; "));
        }
        println!("{name}: baked and checked in {:.2?}", start.elapsed());
    }
    Ok(())
}
