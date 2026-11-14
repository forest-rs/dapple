// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Bakes every material `dapple_library` can make as a swatch for one
//! contact sheet: every preset of every package, and every library module
//! that is a complete surface by itself.
//!
//! Run with `cargo run --release -p material_wall -- [options] [output-dir]`
//! (options: `--size <texels>`, default 512; `--no-cache`; `--budget-mib
//! <MiB>`, default 512). The default output is the repository's
//! git-ignored `.local/gallery/material-wall`. Then
//! `blender --background --python examples/material_wall/tools/render.py --
//! <output-dir>` lays the swatches out as a labeled, lit contact sheet.
//!
//! **What is on the wall** comes from the library, not from a list kept
//! here:
//!
//! - every package in [`dapple_library::packages::PACKAGES`], at every
//!   preset it ships, in its order (fresh to weathered); a package's family
//!   is its name;
//! - every module [`dapple_library::packages::registry`] registers whose
//!   inputs are all optional and which returns a `material`: a surface
//!   that needs nothing from its host. Modules that need an input material
//!   (finishes, weathering, corrosion, paint) or a host resource (by
//!   example) are building blocks, seen on the wall inside the assets and
//!   packages that use them; a module that a package re-expresses
//!   (`dapple_library.X` beside `dapple_library.packages.X`) is shown as
//!   the package's presets instead.
//!
//! The registry does not say what family a module belongs to, so
//! [`FAMILIES`] groups the library's modules into rows; a module it does
//! not list still appears, in the `other` row.
//!
//! Each swatch is built at its defaults (or preset) on a 1 m wrapping tile
//! through one [`InstanceCache`], so presets that share a step (a package's
//! bare metal under every preset's weathering) build it once. Each is
//! gated on `dapple_lab`'s material report: a failing swatch is still
//! exported, marked in the index, and fails the run at the end. Each is
//! exported through `dapple_export` into `<output-dir>/<family>/<label>/`
//! with its `report.json` and `diagnostics.txt`, and `wall.json` indexes
//! them for the render script.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use dapple_export::{Exported, Settings, Source};
use dapple_field::Edge;
use dapple_lab::report::Report;
use dapple_library::packages::{self, PACKAGES};
use dapple_material::module::{
    Bind, Context, Diagnostics, InstanceCache, Interface, Module, OutputKind,
};
use dapple_material::resource::NoResources;
use dapple_material::{Grid, Material};
use dapple_package::Package;
use glam::Vec2;
use serde_json::{Value, json};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The texels across the 1 m tile, unless `--size` says otherwise: the
/// wall is an overview, not a close-up.
const SIZE: u32 = 512;

/// The instance cache's default budget, in MiB.
const BUDGET_MIB: usize = 512;

/// The prefix of a package's module name; a library module named as the
/// rest of it is the module the package re-expresses.
const PACKAGED: &str = "dapple_library.packages.";

/// The prefix of the library's own module names.
const LIBRARY: &str = "dapple_library.";

/// Rows of the wall for the library's own modules, in order, each with
/// its modules' names (without [`LIBRARY`]) in column order. Packages come
/// first, a row each; a module not listed here goes to [`OTHER`].
const FAMILIES: &[(&str, &[&str])] = &[
    ("metal", &["copper", "steel"]),
    ("assets", &["glazed_brick_wall", "stone_sill", "threshold"]),
    (
        "masonry",
        &[
            "ashlar_limestone",
            "rubble_wall",
            "flint_wall",
            "roman_brick",
            "marble",
            "terracotta_tile",
        ],
    ),
    (
        "bark",
        &["beech_bark", "birch_bark", "scots_pine_bark", "spruce_bark"],
    ),
    ("bodies", &["ceramic_body", "mortar", "stone", "wood"]),
];

/// The row of modules [`FAMILIES`] does not list.
const OTHER: &str = "other";

/// One swatch: a module (a package or a library module), how it is bound,
/// and where it goes on the wall.
struct Swatch {
    /// The wall's row.
    family: String,
    /// The column's label: the preset, or the module's short name.
    label: String,
    /// The package preset, if the swatch is one.
    preset: Option<String>,
    /// What it is.
    doc: String,
    /// The top-level instance name. A package's presets share it, so their
    /// shared steps have equal paths and the cache finds them.
    instance: String,
    module: Arc<dyn Module>,
    bind: Bind,
}

impl Swatch {
    /// `family/label`: the subject of its report and export, and its
    /// directory.
    fn subject(&self) -> String {
        format!("{}/{}", self.family, self.label)
    }
}

/// A registered module the wall does not show, and why.
struct LeftOut {
    module: String,
    reason: &'static str,
}

/// Whether a module is a complete surface: every input optional, and a
/// `material` output.
fn showable(interface: &Interface) -> bool {
    interface.inputs.iter().all(|i| !i.required)
        && interface
            .outputs
            .iter()
            .any(|o| o.name == "material" && o.kind == OutputKind::Material)
}

/// Everything the library makes, in wall order, and the registered modules
/// left out.
fn swatches() -> Result<(Vec<Swatch>, Vec<LeftOut>)> {
    let registry = packages::registry();
    let mut out = Vec::new();
    let mut packaged = Vec::new();
    for source in PACKAGES {
        let package = Package::from_json(source)?;
        let compiled = registry.compile(&package)?;
        let name = compiled.interface().id.name.into_owned();
        let family = name.strip_prefix(PACKAGED).unwrap_or(&name).to_owned();
        #[expect(
            clippy::arc_with_non_send_sync,
            reason = "the registry hands out modules as `Arc<dyn Module>`; the wall is one thread"
        )]
        let module: Arc<dyn Module> = Arc::new(compiled.clone());
        if package.presets.is_empty() {
            out.push(Swatch {
                family: family.clone(),
                label: String::from("defaults"),
                preset: None,
                doc: package.interface.doc.clone(),
                instance: family.clone(),
                module: module.clone(),
                bind: Bind::new(),
            });
        }
        for preset in &package.presets {
            out.push(Swatch {
                family: family.clone(),
                label: preset.name.clone(),
                preset: Some(preset.name.clone()),
                doc: preset.doc.clone(),
                instance: family.clone(),
                module: module.clone(),
                bind: compiled
                    .preset(&preset.name)
                    .ok_or("a preset the package lists")?,
            });
        }
        packaged.push(format!("{LIBRARY}{family}"));
    }

    let mut natives = Vec::new();
    let mut left_out = Vec::new();
    for module in registry.modules() {
        let interface = module.interface();
        let name = interface.id.name.to_string();
        if packaged.contains(&name) {
            left_out.push(LeftOut {
                module: interface.id.to_string(),
                reason: "shown as its package's presets",
            });
            continue;
        }
        if !showable(&interface) {
            left_out.push(LeftOut {
                module: interface.id.to_string(),
                reason: "a building block: needs an input material or resource",
            });
            continue;
        }
        let short = name.strip_prefix(LIBRARY).unwrap_or(&name).to_owned();
        let place = FAMILIES
            .iter()
            .enumerate()
            .find_map(|(row, (family, members))| {
                let column = members.iter().position(|m| *m == short)?;
                Some((row, column, *family))
            })
            .unwrap_or((FAMILIES.len(), natives.len(), OTHER));
        natives.push((
            place,
            Swatch {
                family: place.2.to_owned(),
                label: short.clone(),
                preset: None,
                doc: interface.doc.to_string(),
                instance: short,
                module: module.clone(),
                bind: Bind::new(),
            },
        ));
    }
    natives.sort_by_key(|((row, column, _), _)| (*row, *column));
    out.extend(natives.into_iter().map(|(_, s)| s));
    Ok((out, left_out))
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

/// A swatch built and gated.
struct Built {
    material: Material,
    diagnostics: Diagnostics,
    report: Report,
    /// Seconds spent building the material (not gating or exporting it).
    seconds: f64,
    /// Instances this build found in the cache.
    hits: u64,
    /// Instances this build ran.
    misses: u64,
}

fn instantiate(mut cx: Context<'_>, swatch: &Swatch) -> Result<(Material, Diagnostics)> {
    let material = cx
        .instantiate(
            swatch.module.as_ref(),
            &swatch.instance,
            swatch.bind.clone(),
        )?
        .take_material("material")
        .ok_or("the module returned no material")?;
    Ok((material, cx.into_diagnostics()))
}

/// Builds `swatch` on `grid`, through `cache` when there is one, and gates
/// it on the lab's material report.
fn build(swatch: &Swatch, grid: Grid, cache: Option<&mut InstanceCache>) -> Result<Built> {
    let counts = |c: &InstanceCache| {
        let s = c.stats();
        (s.hits, s.misses)
    };
    let start = Instant::now();
    let (material, diagnostics, hits, misses) = match cache {
        Some(cache) => {
            let before = counts(cache);
            let cx = Context::new(grid, &NoResources).with_cache(cache);
            let (m, d) = instantiate(cx, swatch)?;
            let after = counts(cache);
            (m, d, after.0 - before.0, after.1 - before.1)
        }
        None => {
            let (m, d) = instantiate(Context::new(grid, &NoResources), swatch)?;
            let n = d.instances().count() as u64;
            (m, d, 0, n)
        }
    };
    let seconds = start.elapsed().as_secs_f64();
    let report = gate(&swatch.subject(), &material);
    Ok(Built {
        material,
        diagnostics,
        report,
        seconds,
        hits,
        misses,
    })
}

/// The gate every swatch passes: ranges, invalid values, seams against
/// the tiling promise, lowering losses, and a material-wide transform that
/// must move every channel.
fn gate(subject: &str, material: &Material) -> Report {
    let mut report = dapple_lab::material::material_report(subject, material);
    report.merge(
        "relationships",
        dapple_lab::material::transform_check(material),
    );
    report
}

/// The export of a built swatch, with its diagnostics and lab report.
fn export(swatch: &Swatch, built: &Built) -> Result<Exported> {
    let subject = swatch.subject();
    let source = Source {
        diagnostics: Some(&built.diagnostics),
        lab: Some(&built.report),
        ..Source::new(&subject, &built.material)
    };
    Ok(dapple_export::export(&source, &Settings::default())?)
}

/// The names of the report's failing entries.
fn failures(report: &Report) -> Vec<String> {
    report
        .failures()
        .map(|e| match e {
            dapple_lab::Entry::Measurement(m) => format!("{} = {}", m.name, m.value),
            dapple_lab::Entry::Check(c) => format!("{}: {}", c.name, c.detail),
        })
        .collect()
}

/// The index entry of a swatch.
fn index_entry(swatch: &Swatch, built: &Built, exported: &Exported) -> Value {
    let grid = &exported.provenance.grid;
    let tiling = &exported.provenance.tiling;
    json!({
        "label": swatch.label,
        "family": swatch.family,
        "preset": swatch.preset,
        "module": swatch.module.interface().id.to_string(),
        "doc": swatch.doc,
        "directory": swatch.subject(),
        "size": grid.size,
        "texels": [grid.width, grid.height],
        "tiling": { "x": tiling.x, "y": tiling.y },
        "seconds": built.seconds,
        "cache": { "hits": built.hits, "misses": built.misses },
        "passed": built.report.passed(),
        "failures": failures(&built.report),
    })
}

struct Options {
    size: u32,
    cache: bool,
    budget_mib: usize,
    out: PathBuf,
}

fn options() -> Result<Options> {
    let mut o = Options {
        size: SIZE,
        cache: true,
        budget_mib: BUDGET_MIB,
        out: repository().join(".local/gallery/material-wall"),
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--size" => o.size = args.next().ok_or("--size needs a value")?.parse()?,
            "--no-cache" => o.cache = false,
            "--budget-mib" => {
                o.budget_mib = args.next().ok_or("--budget-mib needs a value")?.parse()?;
            }
            flag if flag.starts_with("--") => {
                return Err(format!(
                    "unknown option {flag}: --size <texels>, --no-cache, --budget-mib <MiB>"
                )
                .into());
            }
            dir => o.out = PathBuf::from(dir),
        }
    }
    Ok(o)
}

fn main() -> Result<()> {
    let o = options()?;
    let run = Instant::now();
    let (swatches, left_out) = swatches()?;
    for l in &left_out {
        println!("not on the wall: {} ({})", l.module, l.reason);
    }
    std::fs::create_dir_all(&o.out)?;
    let grid = tile(o.size);
    let mut cache = o.cache.then(|| InstanceCache::new(o.budget_mib << 20));
    let mut entries = Vec::new();
    let mut table = String::new();
    writeln!(
        table,
        "{:<16} {:<20} {:>9} {:>8} {:>5} {:>6}  verdict",
        "family", "swatch", "size (m)", "build", "hits", "misses"
    )?;
    let (mut total, mut failed) = (0.0, Vec::new());
    for swatch in &swatches {
        let built = build(swatch, grid, cache.as_mut())?;
        total += built.seconds;
        let exported = export(swatch, &built)?;
        let dir = o.out.join(swatch.subject());
        exported.write_dir(&dir)?;
        std::fs::write(dir.join("report.json"), built.report.to_json())?;
        std::fs::write(dir.join("diagnostics.txt"), built.diagnostics.to_string())?;
        let size = exported.provenance.grid.size;
        let passed = built.report.passed();
        println!(
            "{} in {:.2} s ({} hits, {} misses){}",
            swatch.subject(),
            built.seconds,
            built.hits,
            built.misses,
            if passed { "" } else { ": FAILED" }
        );
        writeln!(
            table,
            "{:<16} {:<20} {:>9} {:>6.2} s {:>5} {:>6}  {}",
            swatch.family,
            swatch.label,
            format!("{}×{}", size[0], size[1]),
            built.seconds,
            built.hits,
            built.misses,
            if passed { "pass" } else { "FAIL" }
        )?;
        if !passed {
            for f in failures(&built.report) {
                eprintln!("FAILED {}: {f}", swatch.subject());
            }
            failed.push(swatch.subject());
        }
        entries.push(index_entry(swatch, &built, &exported));
    }
    let stats = cache.as_ref().map(InstanceCache::stats);
    let index = json!({
        "format": "dapple-material-wall",
        "version": 1,
        "texels": o.size,
        "build_seconds": total,
        "run_seconds": run.elapsed().as_secs_f64(),
        "cache": stats.as_ref().map(|s| json!({
            "budget": s.budget,
            "hits": s.hits,
            "misses": s.misses,
            "evictions": s.evictions,
            "rejected": s.rejected,
            "entries": s.entries,
            "bytes": s.bytes,
        })),
        "passed": failed.is_empty(),
        "swatches": entries,
        "left_out": left_out
            .iter()
            .map(|l| json!({ "module": l.module, "reason": l.reason }))
            .collect::<Vec<_>>(),
    });
    std::fs::write(
        o.out.join("wall.json"),
        serde_json::to_string_pretty(&index)? + "\n",
    )?;
    print!("\n{table}");
    println!(
        "{} swatches at {}² built in {total:.2} s ({:.2} s in all)",
        swatches.len(),
        o.size,
        run.elapsed().as_secs_f64()
    );
    match &stats {
        Some(s) => print!("instance cache: {s}"),
        None => println!("instance cache: off"),
    }
    println!("{}", o.out.join("wall.json").display());
    if !failed.is_empty() {
        return Err(format!("lab reports failed: {}", failed.join(", ")).into());
    }
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
    use super::*;

    /// The grid of the test: small enough for `cargo test`, large enough
    /// for every swatch to build and pass the gate (at 16² the glazed
    /// brick wall's glaze no longer tiles).
    const GATE_SIZE: u32 = 32;

    /// Every package preset and every showable module is on the wall, and
    /// every registered module is either on it or left out with a reason:
    /// a package or module added to the library appears without an edit
    /// here.
    #[test]
    fn the_wall_shows_the_whole_library() {
        let (swatches, left_out) = swatches().expect("the library enumerates");
        for source in PACKAGES {
            let package = Package::from_json(source).expect("a shipped package");
            for preset in &package.presets {
                assert!(
                    swatches
                        .iter()
                        .any(|s| s.preset.as_deref() == Some(preset.name.as_str())
                            && s.module.interface().id.name == package.interface.module.name),
                    "{}/{} is on the wall",
                    package.interface.module.name,
                    preset.name
                );
            }
        }
        for module in packages::registry().modules() {
            let id = module.interface().id;
            let shown = swatches.iter().any(|s| s.module.interface().id == id);
            let left = left_out.iter().any(|l| l.module == id.to_string());
            assert!(shown != left, "{id} is shown or left out, not both");
        }
        assert!(
            swatches.iter().all(|s| s.family != OTHER),
            "every library module has a family row"
        );
        let mut subjects: Vec<String> = swatches.iter().map(Swatch::subject).collect();
        subjects.sort();
        subjects.dedup();
        assert_eq!(subjects.len(), swatches.len(), "directories are unique");
    }

    /// Every swatch builds on a small grid through one cache, passes the
    /// gate `main` applies, and exports in memory; presets that share
    /// steps find them in the cache.
    #[test]
    fn every_swatch_builds_passes_and_exports() {
        let (swatches, _) = swatches().expect("the library enumerates");
        let mut cache = InstanceCache::new(64 << 20);
        let mut hits = 0;
        for swatch in &swatches {
            let built = build(swatch, tile(GATE_SIZE), Some(&mut cache)).expect("builds");
            assert!(
                built.report.passed(),
                "{}: {:#?}",
                swatch.subject(),
                failures(&built.report)
            );
            let exported = export(swatch, &built).expect("exports");
            assert_eq!(exported.provenance.grid.size, [1.0, 1.0], "a 1 m tile");
            let entry = index_entry(swatch, &built, &exported);
            assert_eq!(entry["directory"], swatch.subject(), "indexed");
            hits += built.hits;
        }
        assert!(hits > 0, "presets reuse their shared steps");
    }
}
