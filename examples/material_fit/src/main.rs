// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Authoring by measurement: fits `dapple_library` modules' public
//! parameters to stated targets with `dapple_lab::fit` (CMA-ES).
//!
//! Run with `cargo run --release -p material_fit -- [output-dir]`; the
//! default is the repository's git-ignored `.local/gallery/fit`.
//!
//! 1. **The glazed brick wall**, weathered, fitted for its chips, dirt and
//!    cream glaze to: about 8% of the brick face exposed substrate; chips
//!    within 15 mm of an arris; the cream field's mean albedo within ΔE 3
//!    of sRGB (223, 209, 177); and a perceived-roughness distribution
//!    (the coat's where coated, the surface's elsewhere) with stated 25th
//!    and 75th percentiles: a satin rather than a glassy glaze, and less
//!    dirt than the default.
//! 2. **Stone to an exemplar**: a host supplies an exemplar image (here a
//!    synthetic one, stone rendered with hidden parameters, standing in for
//!    a photograph); the stone module's lightness, darkness, bedding and
//!    grain are fitted to its luminance percentiles, mean chroma and band
//!    energies at four scales (a simple spectral descriptor), and the
//!    fitted values are compared with the hidden ones.
//!
//! Each fit writes its report in the lab format (`*-fit.json`),
//! before-and-after previews, and the fitted parameters as a preset
//! document (`*-preset.json`, `dapple_package`), which is read back and
//! checked to rebuild the fitted material bit for bit.

use std::path::{Path, PathBuf};

use dapple_field::Value;
use dapple_lab::fit::{Cmaes, Dimension, Space, Target, fit};
use dapple_lab::material::unit_tile;
use dapple_lab::measure::{band_energies, delta_e, lab};
use dapple_lab::preview::{base_color, contact_sheet, raking};
use dapple_library::glazed_brick::{BODY, GLAZE};
use dapple_library::modules::{GlazedBrickWall, Stone};
use dapple_material::module::{Bind, Context, Module, ParamValue};
use dapple_material::resource::NoResources;
use dapple_material::{Aux, ChannelId, Material, Param};
use dapple_package::{Preset, PresetSet};
use dapple_raster::{DistanceTransform, RasterOp, percentiles};
use glam::Vec3;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn build(module: &dyn Module, n: u32, bind: Bind) -> Result<Material> {
    let mut cx = Context::new(unit_tile(n), &NoResources);
    let mut out = cx.instantiate(module, "fit", bind)?;
    out.take_material("material")
        .ok_or_else(|| "no material".into())
}

fn id(m: &Material, a: Aux, i: usize) -> u32 {
    match m.value(ChannelId::Aux(a), i) {
        Value::Id(v) => v,
        _ => 0,
    }
}

fn scalar(m: &Material, p: Param, i: usize) -> f32 {
    m.value(ChannelId::Param(p), i).component(0).unwrap_or(0.0)
}

fn color(m: &Material, i: usize) -> Vec3 {
    match m.value(ChannelId::Param(Param::BaseColor), i) {
        Value::Vector3(c) => c,
        _ => Vec3::ZERO,
    }
}

/// The wall's measurements: exposed substrate, chips near arrises, the
/// cream field's ΔE, and perceived-roughness percentiles.
fn wall_measures(m: &Material) -> Result<Vec<f64>> {
    let g = m.grid();
    let n = g.len();
    let brick: Vec<bool> = (0..n).map(|i| id(m, Aux::Region, i) != 0).collect();
    let body: Vec<bool> = (0..n).map(|i| id(m, Aux::Surface, i) == BODY).collect();
    let bricks = brick.iter().filter(|b| **b).count().max(1);
    let exposed = (0..n).filter(|&i| brick[i] && body[i]).count();
    // Distance from each texel to the nearest mortar, in meters.
    let mortar = g.raster(brick.iter().map(|b| if *b { 0.0 } else { 1.0 }).collect())?;
    let to_mortar = DistanceTransform { threshold: 0.5 }.apply(&mortar)?;
    let near = (0..n)
        .filter(|&i| brick[i] && body[i] && to_mortar.values()[i] <= 0.015)
        .count();
    // The cream field: glazed texels above the dado and band.
    let field_from = 5.0 / 14.0;
    let (mut sum, mut count) = (Vec3::ZERO, 0_u32);
    for i in 0..n {
        let y = g.center(i).y - g.origin.y;
        if y > field_from && id(m, Aux::Surface, i) == GLAZE {
            sum += color(m, i);
            count += 1;
        }
    }
    let mean = sum / count.max(1) as f32;
    let de = delta_e(mean.to_array().map(f64::from), [223, 209, 177]);
    // Perceived roughness: the coat's where coated, the surface's elsewhere.
    let rough = g.raster(
        (0..n)
            .map(|i| {
                if scalar(m, Param::CoatWeight, i) > 0.5 {
                    scalar(m, Param::CoatRoughness, i)
                } else {
                    scalar(m, Param::SpecularRoughness, i)
                }
            })
            .collect(),
    )?;
    let p = percentiles(&rough, &[0.25, 0.75])?;
    Ok(vec![
        exposed as f64 / bricks as f64,
        near as f64 / exposed.max(1) as f64,
        de,
        f64::from(p[0]),
        f64::from(p[1]),
    ])
}

/// A stone's statistics: L* percentiles, mean a* and b*, and band energies
/// of luminance at 1, 4, 16 and 64 mm.
fn stone_measures(m: &Material) -> Result<Vec<f64>> {
    let g = m.grid();
    let labs: Vec<[f64; 3]> = (0..g.len())
        .map(|i| lab(color(m, i).to_array().map(f64::from)))
        .collect();
    #[expect(clippy::cast_possible_truncation, reason = "L* in f32")]
    let l = g.raster(labs.iter().map(|c| c[0] as f32).collect())?;
    let p = percentiles(&l, &[0.1, 0.5, 0.9])?;
    let n = labs.len() as f64;
    let (a, b) = labs
        .iter()
        .fold((0.0, 0.0), |(a, b), c| (a + c[1] / n, b + c[2] / n));
    let bands = band_energies(&l, &[0.001, 0.004, 0.016, 0.064])?;
    let mut out = vec![f64::from(p[0]), f64::from(p[1]), f64::from(p[2]), a, b];
    out.extend(bands);
    Ok(out)
}

/// The search's parameters as the modules' `f32`s.
#[expect(
    clippy::cast_possible_truncation,
    reason = "parameters are in modest ranges"
)]
fn f32s(p: &[f64]) -> Vec<f32> {
    p.iter().map(|v| *v as f32).collect()
}

/// The stone's parameters for search point `p`, as a preset.
fn stone_preset(p: &[f64]) -> Preset {
    let p = f32s(p);
    let light = Vec3::new(0.50, 0.41, 0.27) * p[0];
    let dark = Vec3::new(0.37, 0.29, 0.19) * p[1];
    Preset::new(
        "fitted",
        "fitted to an exemplar's statistics by material_fit",
    )
    .with("light", ParamValue::Color(light.min(Vec3::ONE)))
    .with("dark", ParamValue::Color(dark.min(Vec3::ONE)))
    .with("bedding_strength", ParamValue::Scalar(p[2]))
    .with("grain", ParamValue::Scalar(p[3]))
}

/// The wall's parameters for search point `p`, as a preset.
fn wall_preset(p: &[f64]) -> Preset {
    let p = f32s(p);
    Preset::new("fitted", "fitted to target measurements by material_fit")
        .with("battered", ParamValue::Scalar(p[0]))
        .with("dirt", ParamValue::Scalar(p[1]))
        .with("field", ParamValue::Color(Vec3::new(p[2], p[3], p[4])))
        .with("glaze_roughness", ParamValue::Scalar(p[5]))
}

fn stone(n: u32, p: &[f64]) -> Result<Material> {
    build(&Stone, n, stone_preset(p).bind())
}

fn wall(n: u32, p: &[f64]) -> Result<Material> {
    build(&GlazedBrickWall, n, wall_preset(p).bind())
}

/// Saves `preset` for `module` as a preset document, reads it back and
/// checks that it rebuilds `fitted` bit for bit.
fn save_preset(
    out: &Path,
    name: &str,
    module: &dyn Module,
    n: u32,
    preset: Preset,
    fitted: &Material,
) -> Result<()> {
    let interface = module.interface();
    let set = PresetSet::new(&interface.id, vec![preset]);
    set.check(&interface)?;
    let path = out.join(format!("{name}.json"));
    std::fs::write(&path, set.to_json())?;
    let again = PresetSet::from_json(&std::fs::read_to_string(&path)?)?;
    again.check(&interface)?;
    let rebuilt = build(module, n, again.get("fitted").ok_or("no preset")?.bind())?;
    if rebuilt.digest() != fitted.digest() {
        return Err("the saved preset does not rebuild the fitted material".into());
    }
    println!("{} (rebuilds the fit bit for bit)", path.display());
    Ok(())
}

/// A contact sheet: base color over raking light, one column a material.
fn preview(out: &Path, name: &str, materials: &[&Material]) -> Result<()> {
    let mut tiles: Vec<_> = materials.iter().map(|m| base_color(m)).collect();
    for m in materials {
        tiles.push(raking(m, 2.4, 0.3)?);
    }
    let columns = u32::try_from(materials.len())?;
    let sheet = contact_sheet(&tiles, columns, 6);
    let path = out.join(format!("{name}.png"));
    sheet.write_png(&path)?;
    println!("{}", path.display());
    Ok(())
}

fn show(name: &str, space: &Space, targets: &[Target], fit: &dapple_lab::fit::Fit) {
    println!(
        "{name}: loss {:.4} after {} evaluations",
        fit.loss, fit.evaluations
    );
    for (d, v) in space.dimensions.iter().zip(&fit.params) {
        println!("  {} = {v:.4}", d.name);
    }
    for (t, m) in targets.iter().zip(&fit.measurements) {
        println!(
            "  {} = {m:.4} (target {:?} {:.4} ± {:.4}){}",
            t.name,
            t.goal,
            t.value,
            t.tolerance,
            if t.met(*m) { "" } else { "  MISSED" }
        );
    }
}

fn main() -> Result<()> {
    let out = std::env::args_os()
        .nth(1)
        .map_or_else(|| repository().join(".local/gallery/fit"), PathBuf::from);
    std::fs::create_dir_all(&out)?;
    let n = 256;

    // 1. The glazed wall.
    let space = Space {
        dimensions: vec![
            Dimension::linear("battered", 0.0, 1.0),
            Dimension::linear("dirt", 0.05, 0.6),
            Dimension::linear("field.r", 0.5, 1.0),
            Dimension::linear("field.g", 0.4, 0.9),
            Dimension::linear("field.b", 0.2, 0.7),
            Dimension::linear("glaze_roughness", 0.01, 0.3),
        ],
    };
    let targets = vec![
        Target::equal("exposed_substrate", 0.08, 0.01),
        Target::at_least("chips_within_15mm_of_arris", 0.9, 0.05),
        Target::at_most("cream_delta_e", 3.0, 1.0),
        Target::equal("roughness_p25", 0.06, 0.01),
        Target::equal("roughness_p75", 0.55, 0.05),
    ];
    let start = [0.4, 0.35, 0.74, 0.64, 0.44, 0.035];
    let before = wall(n, &start)?;
    println!("wall before: {:?}", wall_measures(&before)?);
    let options = Cmaes {
        max_evaluations: 400,
        target_loss: 0.05,
        seed: 7,
        ..Cmaes::default()
    };
    let result = fit(&space, &targets, &options, Some(&start), |p| {
        wall(n, p).and_then(|m| wall_measures(&m))
    })?;
    show("wall", &space, &targets, &result);
    let report = result.report("glazed brick wall fit", &space, &targets);
    std::fs::write(out.join("wall-fit.json"), report.to_json())?;
    let fitted = wall(n, &result.params)?;
    preview(&out, "wall-fit", &[&before, &fitted])?;
    save_preset(
        &out,
        "wall-preset",
        &GlazedBrickWall,
        n,
        wall_preset(&result.params),
        &fitted,
    )?;

    // 2. Stone to an exemplar's statistics.
    let hidden = [1.12, 0.78, 0.45, 0.0011];
    let exemplar = stone(n, &hidden)?;
    let stats = stone_measures(&exemplar)?;
    let names = [
        "l_p10",
        "l_p50",
        "l_p90",
        "a_mean",
        "b_mean",
        "band_1_4mm",
        "band_4_16mm",
        "band_16_64mm",
    ];
    let tolerances = [0.5, 0.5, 0.5, 0.3, 0.3, 0.01, 0.01, 0.01];
    let targets: Vec<Target> = names
        .iter()
        .zip(&stats)
        .zip(tolerances)
        .map(|((name, v), t)| Target::equal(name, *v, t))
        .collect();
    let space = Space {
        dimensions: vec![
            Dimension::linear("light_level", 0.6, 1.5),
            Dimension::linear("dark_level", 0.5, 1.4),
            Dimension::linear("bedding_strength", 0.0, 1.0),
            Dimension::log("grain", 0.0003, 0.004),
        ],
    };
    let start = [1.0, 1.0, 0.12, 0.0007];
    let before = stone(n, &start)?;
    let options = Cmaes {
        max_evaluations: 300,
        target_loss: 0.05,
        seed: 11,
        ..Cmaes::default()
    };
    let result = fit(&space, &targets, &options, Some(&start), |p| {
        stone(n, p).and_then(|m| stone_measures(&m))
    })?;
    show("stone", &space, &targets, &result);
    for ((d, v), h) in space.dimensions.iter().zip(&result.params).zip(hidden) {
        println!("  {}: fitted {v:.4}, hidden {h:.4}", d.name);
    }
    let mut report = result.report("stone fit to an exemplar", &space, &targets);
    for (d, h) in space.dimensions.iter().zip(hidden) {
        report.measure(&format!("hidden.{}", d.name), h, "", None);
    }
    std::fs::write(out.join("stone-fit.json"), report.to_json())?;
    // Start, fitted, exemplar.
    let fitted = stone(n, &result.params)?;
    preview(&out, "stone-fit", &[&before, &fitted, &exemplar])?;
    save_preset(
        &out,
        "stone-preset",
        &Stone,
        n,
        stone_preset(&result.params),
        &fitted,
    )?;
    Ok(())
}

fn repository() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the crate lives two levels below the repository root")
}
