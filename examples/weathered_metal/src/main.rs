// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Bakes `dapple_library`'s weathered metals at every preset of their
//! packages and writes their maps and review images.
//!
//! Run with `cargo run --release -p weathered_metal -- [family] [output-dir]`,
//! where `family` is `copper` (copper cladding from bright to verdigris),
//! `steel` (riveted steel plate from mill scale to pitted rust),
//! `painted` (painted steel from new to derelict) or `all` (the default).
//! The default output is the repository's git-ignored
//! `.local/gallery/weathered-metal`. Each preset is built from its package
//! ([`dapple_library::packages`]) on a 1 m wrapping tile and written to
//! `<output-dir>/<family>/<preset>/`; images show the tile with domain `+y`
//! up.
//!
//! `maps/` holds the material as `dapple_export` writes it for glTF:
//! `material.gltf` (the material on a 1 m quad, viewable in any glTF
//! viewer), its textures (`base_color.png`, 16-bit `normal.png`, `orm.png`,
//! `anisotropy.png` where brushed, and for painted steel `clearcoat.png`
//! and `clearcoat_normal.png`), the raw profile's textures for what glTF
//! cannot carry (`coat_color.png`, the linear coat tint, and `height.png`,
//! 16-bit height), and `manifest.json`, which says what every parameter
//! is, which channels hold it, how to decode them (the height's range in
//! meters, the anisotropy axis's frame), what is a constant (a uniformly
//! brushed sheet's anisotropy, the coat's IOR) and what glTF loses. Rows
//! run from domain `y = 0` at the image top, as glTF reads textures.
//! `tools/render.py` renders them in Blender, building each material from
//! its manifest.
//!
//! `diagnostics.txt` lists every module instance by path and every
//! operation's approximation report; `report.json` is `dapple_lab`'s
//! material report with this family's relationship checks, and the run
//! fails when it does. `<family>/progression.json` checks the presets
//! against each other: roughness rises as the metal weathers, and (bare
//! metals) metalness falls. `<family>/timings.txt` records each bake and
//! `<family>/presets.txt` lists the presets from fresh to weathered.

use std::fmt::Write as _;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use dapple_encode::PackSettings;
use dapple_export::{Settings, Source};
use dapple_field::{Edge, Value};
use dapple_lab::report::Report;
use dapple_library::glazed_brick::{METAL, PAINT, PATINA, RUST};
use dapple_library::modules::calibration::COPPER_F0;
use dapple_library::packages::{self, PAINTED_STEEL, RUSTED_STEEL, WEATHERED_COPPER};
use dapple_material::module::Context;
use dapple_material::resource::NoResources;
use dapple_material::{Aux, ChannelId, Grid, Material, Param};
use dapple_package::Package;
use glam::{Vec2, Vec3};

const SIZE: u32 = 2048;

/// The grid of the lab-gate test: the smallest on which every family
/// builds and passes (at 8² copper's seams and laps no longer fit).
#[cfg(test)]
const GATE_SIZE: u32 = 12;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// A family: its name on the command line, its package, and whether its
/// bare metal's metalness should fall preset over preset.
struct Family {
    name: &'static str,
    source: &'static str,
    bare: bool,
}

const FAMILIES: [Family; 3] = [
    Family {
        name: "copper",
        source: WEATHERED_COPPER,
        bare: true,
    },
    Family {
        name: "steel",
        source: RUSTED_STEEL,
        bare: true,
    },
    Family {
        name: "painted",
        source: PAINTED_STEEL,
        bare: false,
    },
];

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let which = args.next().unwrap_or_else(|| String::from("all"));
    let out = args.next().map_or_else(
        || repository().join(".local/gallery/weathered-metal"),
        PathBuf::from,
    );
    let chosen: Vec<&Family> = FAMILIES
        .iter()
        .filter(|f| which == "all" || which == f.name)
        .collect();
    if chosen.is_empty() {
        return Err(format!("unknown family {which:?}: copper, steel, painted or all").into());
    }
    let mut failed = false;
    for family in chosen {
        failed |= !bake_family(family, &out.join(family.name))?;
    }
    if failed {
        return Err("a lab report has failures".into());
    }
    Ok(())
}

/// Bakes every preset of `family` into `dir`; whether every report passed.
fn bake_family(family: &Family, dir: &Path) -> Result<bool> {
    std::fs::create_dir_all(dir)?;
    let package = Package::from_json(family.source)?;
    let module = packages::registry().compile(&package)?;
    let grid = tile(SIZE);
    // Sixteen-bit normals: eight-bit ones band in glossy paint seen up
    // close.
    let settings = Settings {
        pack: PackSettings {
            sixteen_bit_normals: true,
            ..PackSettings::default()
        },
        ..Settings::default()
    };
    let mut passed = true;
    let mut timings = String::new();
    let mut means = Vec::new();
    for preset in &package.presets {
        let name = preset.name.as_str();
        let bind = module.preset(name).ok_or("a preset the package lists")?;
        let start = std::time::Instant::now();
        let mut cx = Context::new(grid, &NoResources);
        let material = cx
            .instantiate(&module, name, bind)?
            .take_material("material")
            .ok_or("the package returned no material")?;
        let elapsed = start.elapsed();
        println!("built {}/{name} at {SIZE}² in {elapsed:.2?}", family.name);
        writeln!(timings, "{name}: {:.2} s", elapsed.as_secs_f64())?;
        let out = dir.join(name);
        std::fs::create_dir_all(&out)?;
        let diagnostics = cx.into_diagnostics();
        std::fs::write(out.join("diagnostics.txt"), diagnostics.to_string())?;
        write_previews(&out, &material)?;

        let (report, mean) = lab_report(family, name, &material);
        means.push((name, mean));
        std::fs::write(out.join("report.json"), report.to_json())?;

        let source = Source {
            diagnostics: Some(&diagnostics),
            lab: Some(&report),
            ..Source::new(name, &material)
        };
        let exported = dapple_export::export(&source, &settings)?;
        let manifest = exported.write_dir(&out.join("maps"))?;
        let files: Vec<&str> = manifest.files.iter().map(|f| f.path.as_str()).collect();
        println!("  wrote {files:?}");
        for p in exported.parameters.iter().filter(|p| !p.notes.is_empty()) {
            println!("  {}: {}", p.name, p.notes.join("; "));
        }
        if !report.passed() {
            for f in report.failures() {
                eprintln!("FAILED {}/{name}: {f:?}", family.name);
            }
            passed = false;
        }
    }
    std::fs::write(dir.join("timings.txt"), timings)?;
    let order: Vec<&str> = package.presets.iter().map(|p| p.name.as_str()).collect();
    std::fs::write(dir.join("presets.txt"), order.join("\n") + "\n")?;
    let progression = progression(family, &means);
    std::fs::write(dir.join("progression.json"), progression.to_json())?;
    if !progression.passed() {
        for f in progression.failures() {
            eprintln!("FAILED {} progression: {f:?}", family.name);
        }
        passed = false;
    }
    Ok(passed)
}

/// A 1 m wrapping tile of `size`² texels.
fn tile(size: u32) -> Grid {
    #[expect(clippy::cast_precision_loss, reason = "a small grid size")]
    let texel = Vec2::splat(1.0 / size as f32);
    Grid {
        width: size,
        height: size,
        origin: Vec2::ZERO,
        texel,
        edge: Edge::Wrap,
    }
}

/// The lab's report on one preset: ranges, invalid values, seams against
/// the tiling promise, lowering losses, a material-wide transform that
/// must move every channel, and the metals' own relationships; with the
/// preset's means for the family's progression.
fn lab_report(family: &Family, preset: &str, material: &Material) -> (Report, Means) {
    let mut report = dapple_lab::material::material_report(preset, material);
    report.merge(
        "relationships",
        dapple_lab::material::transform_check(material),
    );
    let (checks, mean) = relationships(family, preset, material);
    report.merge("metal", checks);
    (report, mean)
}

/// A preset's mean roughness and metalness.
#[derive(Copy, Clone, Debug)]
struct Means {
    roughness: f64,
    metalness: f64,
}

fn surface(m: &Material, i: usize) -> u32 {
    match m.value(ChannelId::Aux(Aux::Surface), i) {
        Value::Id(id) => id,
        _ => 0,
    }
}

/// The metals' relationship checks on one preset:
///
/// - where a corrosion product or paint wins the surface (it covers at
///   least half the texel), metalness is at most a half, since they are
///   dielectrics laid over metal;
/// - they are rougher than the bare metal beside them;
/// - bright copper's bare metal keeps copper's calibrated F0.
fn relationships(family: &Family, preset: &str, m: &Material) -> (Report, Means) {
    let mut r = Report::new("metal");
    let (mut covered, mut covered_metal_max) = (0_u64, 0.0_f32);
    let (mut covered_rough, mut bare_rough, mut bare) = (0.0_f64, 0.0_f64, 0_u64);
    let (mut rough_sum, mut metal_sum) = (0.0_f64, 0.0_f64);
    let mut f0 = Vec3::ZERO;
    for i in 0..m.grid().len() {
        let metal = scalar(m, Param::BaseMetalness, i);
        let rough = scalar(m, Param::SpecularRoughness, i);
        rough_sum += f64::from(rough);
        metal_sum += f64::from(metal);
        match surface(m, i) {
            PATINA | RUST | PAINT => {
                covered += 1;
                covered_metal_max = covered_metal_max.max(metal);
                covered_rough += f64::from(rough);
            }
            METAL if metal >= 0.999 => {
                bare += 1;
                bare_rough += f64::from(rough);
                f0 += vec3(m.value(ChannelId::Param(Param::BaseColor), i));
            }
            _ => {}
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "texel counts are small")]
    let n = m.grid().len() as f64;
    r.measure("covered.share", count_share(covered, n), "", None);
    r.measure("bare.share", count_share(bare, n), "", None);
    if covered > 0 {
        r.measure(
            "covered.metalness.max",
            f64::from(covered_metal_max),
            "",
            Some([0.0, 0.5 + 1e-4]),
        );
    }
    // Paint is as rough as its finish says; corrosion is rough.
    if covered > 0 && bare > 0 && family.bare {
        #[expect(clippy::cast_precision_loss, reason = "texel counts are small")]
        let (c, b) = (covered_rough / covered as f64, bare_rough / bare as f64);
        r.measure("covered.roughness.mean", c, "", None);
        r.measure("bare.roughness.mean", b, "", None);
        r.check(
            "corrosion_is_rougher",
            c > b,
            format!("corrosion {c:.3} against bare metal {b:.3}"),
        );
    }
    if bare > 0 {
        #[expect(clippy::cast_precision_loss, reason = "texel counts are small")]
        let f0 = f0 / bare as f32;
        for (k, v) in f0.to_array().iter().enumerate() {
            r.measure(&format!("bare.f0[{k}]"), f64::from(*v), "", None);
        }
        if family.name == "copper" && preset == "bright" {
            let miss = (f0 - COPPER_F0).abs().max_element();
            r.check(
                "copper_f0",
                miss < 0.05,
                format!("bare copper {f0} against {COPPER_F0}"),
            );
        }
    }
    (
        r,
        Means {
            roughness: rough_sum / n,
            metalness: metal_sum / n,
        },
    )
}

fn count_share(count: u64, n: f64) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "texel counts are small")]
    let c = count as f64;
    c / n
}

/// Checks the presets against each other, in the package's order from
/// fresh to weathered.
fn progression(family: &Family, means: &[(&str, Means)]) -> Report {
    let mut r = Report::new(family.name);
    for (name, m) in means {
        r.measure(&format!("{name}.roughness.mean"), m.roughness, "", None);
        r.measure(&format!("{name}.metalness.mean"), m.metalness, "", None);
    }
    for pair in means.windows(2) {
        let ((a, ma), (b, mb)) = (pair[0], pair[1]);
        r.check(
            &format!("roughness.{a}_to_{b}"),
            mb.roughness >= ma.roughness - 1e-3,
            format!("{:.3} to {:.3}", ma.roughness, mb.roughness),
        );
        if family.bare {
            r.check(
                &format!("metalness.{a}_to_{b}"),
                mb.metalness <= ma.metalness + 1e-3,
                format!("{:.3} to {:.3}", ma.metalness, mb.metalness),
            );
        }
    }
    r
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
    let metal = image_rows(m, |i| [scalar(m, Param::BaseMetalness, i); 3]);
    write_png(&out.join("metalness.png"), w, h, &metal, false)?;
    let height = m.height()?;
    let (lo, hi) = height
        .values()
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let heights = image_rows(m, |i| [(height.values()[i] - lo) / (hi - lo).max(1e-9); 3]);
    write_png(&out.join("height.png"), w, h, &heights, false)?;
    let surfaces = image_rows(m, |i| match surface(m, i) {
        METAL => [0.75, 0.75, 0.78],
        PATINA => [0.2, 0.7, 0.55],
        RUST => [0.7, 0.3, 0.1],
        PAINT => [0.1, 0.25, 0.6],
        _ => [1.0, 0.0, 1.0],
    });
    write_png(&out.join("surface.png"), w, h, &surfaces, false)?;
    let raking = dapple_lab::preview::raking(m, 2.4, 0.35)?;
    write_png(
        &out.join("raking.png"),
        raking.width,
        raking.height,
        &raking.pixels,
        true,
    )?;
    Ok(())
}

fn repository() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the crate lives two levels below the repository root")
}

#[cfg(test)]
mod tests {
    use dapple_material::module::Bind;

    use super::*;

    /// Every family passes its lab report at its defaults on a small grid:
    /// the gate `main` applies to each preset at 2048².
    #[test]
    fn the_families_pass_the_lab_report() {
        for family in &FAMILIES {
            let package = Package::from_json(family.source).expect("a shipped package");
            let module = packages::registry()
                .compile(&package)
                .expect("the package compiles");
            let mut cx = Context::new(tile(GATE_SIZE), &NoResources);
            let material = cx
                .instantiate(&module, family.name, Bind::new())
                .expect("instantiates")
                .take_material("material")
                .expect("a material");
            let (report, _) = lab_report(family, "defaults", &material);
            let failures: Vec<_> = report.failures().collect();
            assert!(failures.is_empty(), "{}: {failures:#?}", family.name);
        }
    }
}
