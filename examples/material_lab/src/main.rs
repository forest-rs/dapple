// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Runs `dapple_lab` over `dapple_library`'s modules.
//!
//! Run with `cargo run --release -p material_lab -- [output-dir]`; the
//! default is the repository's git-ignored `.local/gallery/lab`. It writes:
//!
//! - `sweep-*.png`: contact sheets of parameter sweeps, each value a
//!   raking-light (grazing) preview over its base color;
//! - `wall-tiled.png` and `wall-raking-tiled.png`: the glazed wall repeated
//!   2 × 2, to show its seams (it promises to tile along x only: its foot
//!   is weathered);
//! - `wall-mips.png`: its base color's mip levels side by side;
//! - `report.json`: every material's report (ranges, invalid values, seams
//!   against its tiling promise, lowering losses) and the relationship
//!   checks: a material-wide transform leaves no channel behind, raising
//!   the resolution keeps feature size, an incremental composite equals a
//!   clean one.
//!
//! It exits with an error when any check fails.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use dapple_elements::{Composite, Placement, Realized};
use dapple_field::Domain;
use dapple_lab::material::{
    agreement, material_report, resolution_check, transform_check, unit_tile,
};
use dapple_lab::preview::{Picture, base_color, contact_sheet, mips, raking, tiled};
use dapple_lab::report::Report;
use dapple_library::glazed_brick::{self, Palette};
use dapple_library::modules::{GlazedBrickWall, Grime, Mortar, Moss, Stone};
use dapple_material::module::{Bind, Context, Module};
use dapple_material::resource::NoResources;
use dapple_material::{Aux, ChannelId, Grid, Material};
use glam::Vec2;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// A low light from the upper left, as a grazing view shows relief.
const AZIMUTH: f32 = 2.4;
const ELEVATION: f32 = 0.3;

fn build(module: &dyn Module, grid: Grid, bind: Bind) -> Result<Material> {
    let mut cx = Context::new(grid, &NoResources);
    let mut out = cx.instantiate(module, "lab", bind)?;
    out.take_material("material")
        .ok_or_else(|| "no material".into())
}

/// A sweep: for each value, the material's raking preview stacked over its
/// base color, and its report.
fn sweep(
    out: &Path,
    report: &mut Report,
    name: &str,
    values: &[f32],
    make: impl Fn(f32) -> Result<Material>,
) -> Result<()> {
    let mut cells = Vec::new();
    for &v in values {
        let m = make(v)?;
        let (shaded, color) = (raking(&m, AZIMUTH, ELEVATION)?, base_color(&m));
        cells.push(contact_sheet(&[shaded, color], 1, 0));
        report.merge(&format!("{name}={v}"), material_report(name, &m));
    }
    let columns = u32::try_from(values.len()).unwrap_or(1);
    write(
        out,
        &format!("sweep-{name}"),
        &contact_sheet(&cells, columns, 8),
    )
}

fn write(out: &Path, name: &str, picture: &Picture) -> Result<()> {
    let path = out.join(format!("{name}.png"));
    picture.write_png(&path)?;
    println!("{}", path.display());
    Ok(())
}

fn main() -> Result<()> {
    let out = std::env::args_os()
        .nth(1)
        .map_or_else(|| repository().join(".local/gallery/lab"), PathBuf::from);
    std::fs::create_dir_all(&out)?;
    let mut report = Report::new("dapple_library modules");
    let small = unit_tile(256);

    // Sweeps.
    sweep(&out, &mut report, "stone-bedding", &[0.0, 0.3, 0.8], |v| {
        build(&Stone, small, Bind::new().scalar("bedding_strength", v))
    })?;
    let bare_wall = build(
        &GlazedBrickWall,
        small,
        Bind::new().flag("weathered", false),
    )?;
    sweep(&out, &mut report, "grime-coverage", &[0.1, 0.3, 0.5], |v| {
        build(
            &Grime,
            small,
            Bind::new()
                .material("base", bare_wall.clone())
                .scalar("coverage", v),
        )
    })?;
    sweep(
        &out,
        &mut report,
        "moss-coverage",
        &[0.02, 0.05, 0.1],
        |v| {
            build(
                &Moss,
                small,
                Bind::new()
                    .material("base", bare_wall.clone())
                    .scalar("coverage", v),
            )
        },
    )?;

    // The weathered wall: tiled, raking, mips, report, transform.
    let wall = build(&GlazedBrickWall, unit_tile(512), Bind::new())?;
    write(&out, "wall-tiled", &tiled(&base_color(&wall), 2, 2))?;
    write(
        &out,
        "wall-raking-tiled",
        &tiled(&raking(&wall, AZIMUTH, ELEVATION)?, 2, 2),
    )?;
    write(&out, "wall-mips", &mips(&wall, 5)?)?;
    report.merge("wall", material_report("wall", &wall));
    report.merge("wall", transform_check(&wall));

    // Raising resolution keeps physical feature size, once the coarser
    // grid resolves the finest features (the stone's 3 mm tooling needs
    // texels of about a millimeter).
    for (name, module) in [("stone", &Stone as &dyn Module), ("mortar", &Mortar)] {
        let coarse = build(module, unit_tile(1024), Bind::new())?;
        let fine = build(module, unit_tile(2048), Bind::new())?;
        report.merge(
            name,
            resolution_check(name, ChannelId::Aux(Aux::Height), &coarse, &fine),
        );
    }

    // An incremental composite equals a clean one after a brick moves.
    let domain = Domain::periodic(1, 1).ok_or("a unit period")?;
    let set = glazed_brick::layout(&Palette::default())?;
    let instance = glazed_brick::instance(Arc::new(glazed_brick::program()?))?;
    let composite = |set| Composite {
        set,
        instance: &instance,
        background: &glazed_brick::BACKGROUND,
        domain,
        width: 256,
        height: 256,
        tile_size: 32,
    };
    let mut realized = Realized::composite(&composite(&set))?;
    let key = set.keys()[set.len() / 3];
    let mut moved = set.clone();
    let p = set.placement(set.index_of(key).ok_or("a brick")?);
    moved.set_placement(
        key,
        Placement {
            center: p.center + Vec2::new(0.02, 0.01),
            rotation: 0.05,
        },
    )?;
    realized.update(&composite(&moved))?;
    let clean = Realized::composite(&composite(&moved))?;
    report.merge(
        "composite",
        agreement(
            "incremental_equals_clean",
            realized.digest(),
            clean.digest(),
        ),
    );

    let path = out.join("report.json");
    std::fs::write(&path, report.to_json())?;
    println!("{}", path.display());
    let failures: Vec<_> = report.failures().collect();
    if failures.is_empty() {
        println!("all {} entries pass", report.entries.len());
        Ok(())
    } else {
        for f in &failures {
            eprintln!("FAILED: {f:?}");
        }
        Err(format!("{} lab checks failed", failures.len()).into())
    }
}

fn repository() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the crate lives two levels below the repository root")
}
