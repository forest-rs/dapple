// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Calibrates `dapple_library`'s bark and masonry modules against measured
//! reflectance, and renders a swatch sheet of them.
//!
//! For each module with a reference in
//! `dapple_library::modules::calibration`, `dapple_lab::fit` searches the
//! module's color (and roughness, where a measured value exists) so that
//! its mean linear base color over the measured surface matches the
//! reference, and prints the fitted values beside the calibrated defaults;
//! `calibration.json` holds every fit in the lab's report format. Then each
//! module is realized at its defaults, and its base color, normal and
//! roughness maps are written for `tools/render.py`, which lays them out as
//! a lit, labeled swatch sheet in Blender. `sheet.png` is the same sheet as
//! a software render (base color over raking light), without Blender.
//!
//! Run with `cargo run --release -p library_swatches -- [output-dir]`; the
//! default is the repository's git-ignored `.local/gallery/swatches`. Then
//! `blender --background --python examples/library_swatches/tools/render.py -- <output-dir>`.

use std::path::{Path, PathBuf};

use dapple_elements::{LayoutId, RunningBond};
use dapple_field::{Domain, Value};
use dapple_lab::fit::{Cmaes, Dimension, Space, Target, fit};
use dapple_lab::material::unit_tile;
use dapple_lab::preview::{Picture, base_color, contact_sheet, raking};
use dapple_lab::report::Report;
use dapple_library::masonry::UnitMaps;
use dapple_library::modules::calibration::{REFERENCES, Reference, luminance};
use dapple_library::modules::{
    AshlarLimestone, Beech, Birch, FlintWall, Marble, RomanBrick, RubbleWall, ScotsPine, Spruce,
    TerracottaTile,
};
use dapple_material::module::{Bind, Context, Input, Module, ParamValue};
use dapple_material::resource::NoResources;
use dapple_material::{Aux, ChannelId, Material, Param};
use dapple_raster::{HeightToNormal, RasterOp};
use glam::Vec3;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn build(module: &dyn Module, n: u32, bind: Bind) -> Result<Material> {
    let mut cx = Context::new(unit_tile(n), &NoResources);
    let mut out = cx.instantiate(module, "swatch", bind)?;
    out.take_material("material")
        .ok_or_else(|| "no material".into())
}

/// The mean linear base color and roughness, over the units only when
/// `units_only`.
fn means(m: &Material, units_only: bool) -> (Vec3, f32) {
    let (mut c, mut r, mut n) = (Vec3::ZERO, 0.0_f64, 0_u32);
    for i in 0..m.grid().len() {
        if units_only && m.value(ChannelId::Aux(Aux::Region), i) == Value::Id(0) {
            continue;
        }
        if let Value::Vector3(v) = m.value(ChannelId::Param(Param::BaseColor), i) {
            c += v;
        }
        r += f64::from(
            m.value(ChannelId::Param(Param::SpecularRoughness), i)
                .component(0)
                .unwrap_or(0.0),
        );
        n += 1;
    }
    let n = n.max(1);
    #[expect(clippy::cast_precision_loss, reason = "texel counts are small")]
    #[expect(clippy::cast_possible_truncation, reason = "a mean of f32 values")]
    let out = (c / n as f32, (r / f64::from(n)) as f32);
    out
}

fn modules() -> Vec<(&'static str, Box<dyn Module>)> {
    vec![
        ("beech", Box::new(Beech)),
        ("birch", Box::new(Birch)),
        ("scots_pine", Box::new(ScotsPine)),
        ("spruce", Box::new(Spruce)),
        ("ashlar", Box::new(AshlarLimestone)),
        ("rubble", Box::new(RubbleWall)),
        ("flint", Box::new(FlintWall)),
        ("roman_brick", Box::new(RomanBrick)),
        ("marble", Box::new(Marble)),
        ("terracotta", Box::new(TerracottaTile)),
    ]
}

#[expect(clippy::cast_possible_truncation, reason = "parameters are f32")]
fn f(v: f64) -> f32 {
    v as f32
}

/// Fits `module`'s color (and roughness) to `reference` at `n` texels.
fn calibrate(module: &dyn Module, reference: &Reference, n: u32) -> Result<Report> {
    let default = match module
        .interface()
        .params
        .iter()
        .find(|p| p.name == "color")
        .map(|p| p.default)
    {
        Some(ParamValue::Color(c)) => c,
        _ => return Err("no color parameter".into()),
    };
    let mut dims = if reference.luminance_only {
        vec![Dimension::log("scale", 0.25, 4.0)]
    } else {
        vec![
            Dimension::linear("color.r", 0.005, 1.0),
            Dimension::linear("color.g", 0.005, 1.0),
            Dimension::linear("color.b", 0.005, 1.0),
        ]
    };
    let mut targets = if reference.luminance_only {
        let y = f64::from(reference.albedo.x);
        vec![if reference.at_most {
            Target::at_most("luminance", y, 0.004)
        } else {
            Target::equal("luminance", y, 0.004)
        }]
    } else {
        ["r", "g", "b"]
            .iter()
            .zip(reference.albedo.to_array())
            .map(|(c, v)| Target::equal(&format!("albedo.{c}"), f64::from(v), 0.004))
            .collect()
    };
    if let Some(r) = reference.roughness {
        dims.push(Dimension::linear("roughness", 0.5, 1.0));
        targets.push(Target::equal("roughness", f64::from(r), 0.01));
    }
    let space = Space { dimensions: dims };
    let mut start: Vec<f64> = if reference.luminance_only {
        vec![1.0]
    } else {
        default.to_array().map(f64::from).to_vec()
    };
    if reference.roughness.is_some() {
        start.push(0.85);
    }
    let bind = |p: &[f64]| {
        let mut b = if reference.luminance_only {
            Bind::new().color("color", (default * f(p[0])).min(Vec3::ONE))
        } else {
            Bind::new().color("color", Vec3::new(f(p[0]), f(p[1]), f(p[2])))
        };
        if reference.roughness.is_some() {
            b = b.scalar("roughness", f(p[p.len() - 1]));
        }
        b
    };
    let options = Cmaes {
        sigma: 0.15,
        max_evaluations: if reference.roughness.is_some() {
            480
        } else {
            300
        },
        target_loss: 0.02,
        seed: 3,
        ..Cmaes::default()
    };
    let result = fit(&space, &targets, &options, Some(&start), |p| {
        let m = build(module, n, bind(p))?;
        let (c, r) = means(&m, reference.units_only);
        let mut out: Vec<f64> = if reference.luminance_only {
            vec![f64::from(luminance(c))]
        } else {
            c.to_array().map(f64::from).to_vec()
        };
        if reference.roughness.is_some() {
            out.push(f64::from(r));
        }
        Ok::<_, Box<dyn std::error::Error>>(out)
    })?;
    // What the defaults measure, beside what the fit found.
    let at_defaults = build(module, n, Bind::new())?;
    let (c, r) = means(&at_defaults, reference.units_only);
    let fitted = if reference.luminance_only {
        default * f(result.params[0])
    } else {
        Vec3::new(
            f(result.params[0]),
            f(result.params[1]),
            f(result.params[2]),
        )
    };
    println!(
        "{}: fitted color ({:.3}, {:.3}, {:.3}){} in {} evaluations, loss {:.4}; \
         defaults measure ({:.3}, {:.3}, {:.3}), Y {:.3}, roughness {:.3}",
        reference.module,
        fitted.x,
        fitted.y,
        fitted.z,
        if reference.roughness.is_some() {
            format!(", roughness {:.3}", result.params[result.params.len() - 1])
        } else {
            String::new()
        },
        result.evaluations,
        result.loss,
        c.x,
        c.y,
        c.z,
        luminance(c),
        r
    );
    let mut report = result.report(reference.module, &space, &targets);
    for (k, v) in c.to_array().iter().enumerate() {
        report.measure(
            &format!("defaults.albedo.{}", ["r", "g", "b"][k]),
            f64::from(*v),
            "",
            None,
        );
    }
    report.measure("defaults.roughness", f64::from(r), "", None);
    Ok(report)
}

/// Non-color data, stored so the sRGB-encoded PNG holds `v` itself.
fn raw(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.04045 {
        v / 12.92
    } else {
        libm::powf((v + 0.055) / 1.055, 2.4)
    }
}

fn picture(m: &Material, f: impl Fn(usize) -> [f32; 3]) -> Picture {
    let g = m.grid();
    let mut pixels = Vec::with_capacity(g.len());
    for row in 0..g.height {
        for x in 0..g.width {
            let i = (g.height - 1 - row) as usize * g.width as usize + x as usize;
            pixels.push(f(i));
        }
    }
    Picture {
        width: g.width,
        height: g.height,
        pixels,
    }
}

/// Writes `name`'s base color, normal and roughness maps.
fn maps(out: &Path, name: &str, m: &Material) -> Result<()> {
    base_color(m).write_png(&out.join(format!("{name}-base.png")))?;
    let normals = HeightToNormal { scale: 1.0 }.apply(&m.height()?)?;
    picture(m, |i| {
        let n = normals.values()[i];
        [0, 1, 2].map(|k| raw(0.5 + 0.5 * n[k]))
    })
    .write_png(&out.join(format!("{name}-normal.png")))?;
    picture(m, |i| {
        let r = m
            .value(ChannelId::Param(Param::SpecularRoughness), i)
            .component(0)
            .unwrap_or(0.5);
        [raw(r); 3]
    })
    .write_png(&out.join(format!("{name}-roughness.png")))?;
    Ok(())
}

fn main() -> Result<()> {
    let out = std::env::args_os().nth(1).map_or_else(
        || repository().join(".local/gallery/swatches"),
        PathBuf::from,
    );
    std::fs::create_dir_all(&out)?;

    // 1. Calibration.
    let mut report = Report::new("dapple_library calibration");
    for ((_, module), reference) in modules().iter().zip(REFERENCES.iter()) {
        assert_eq!(module.interface().id.name, reference.module, "table order");
        let r = calibrate(module.as_ref(), reference, 128)?;
        report.merge(reference.module, r);
    }
    std::fs::write(out.join("calibration.json"), report.to_json())?;

    // 2. Swatches at the defaults.
    let n = 512;
    let mut swatches: Vec<(String, Material)> = Vec::new();
    for (name, module) in modules() {
        swatches.push((name.into(), build(module.as_ref(), n, Bind::new())?));
    }
    // Bark following growth: a young and an old birch at breast height,
    // and a pine high and low on the stem.
    swatches.push((
        "birch_young".into(),
        build(&Birch, n, Bind::new().scalar("girth", 0.3))?,
    ));
    swatches.push((
        "birch_old".into(),
        build(
            &Birch,
            n,
            Bind::new().scalar("girth", 2.2).scalar("height", 0.5),
        )?,
    ));
    swatches.push((
        "scots_pine_upper".into(),
        build(&ScotsPine, n, Bind::new().scalar("height", 14.0))?,
    ));
    // Ashlar on a unit layout the host supplies: a running bond from
    // dapple_elements, as the construction layer would lay it.
    let bond = RunningBond {
        layout: LayoutId::named("host.bond"),
        domain: Domain::periodic(1, 1).ok_or("a unit period")?,
        courses: 4,
        per_course: 3,
        joint: 0.006,
    }
    .elements(vec![], |_, _| vec![])?;
    let units = UnitMaps::from_elements(&bond, unit_tile(n), 0.03)?;
    swatches.push((
        "ashlar_host_bond".into(),
        build(
            &AshlarLimestone,
            n,
            Bind::new()
                .input("units", Input::Map(units.units))
                .input("edge", Input::Map(units.edge))
                .input("local", Input::Map(units.local)),
        )?,
    ));
    let mut sheet = Vec::new();
    let mut names = String::new();
    for (name, m) in &swatches {
        maps(&out, name, m)?;
        sheet.push(raking(m, 2.4, 0.5)?);
        names.push_str(name);
        names.push('\n');
    }
    std::fs::write(out.join("swatches.txt"), names)?;
    contact_sheet(&sheet, 5, 8).write_png(&out.join("sheet.png"))?;
    println!("{}", out.display());
    Ok(())
}

fn repository() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the crate lives two levels below the repository root")
}
