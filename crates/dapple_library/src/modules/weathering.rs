// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Weathering as deposits: grime in hollows, streaks below ledges, salts
//! on mortar.
//!
//! Each module measures its base material with raster passes (ambient
//! occlusion, cavities, slopes, trails) and scores every texel with a
//! scoped program whose pass inputs are declared as such. The score becomes
//! a coverage by percentile (`coverage` of the texels, whatever the
//! score's range), and [`ops::deposit`] lays the deposit: its height, its
//! surface identity, and its optics over the base's, coat included.

use alloc::vec;
use alloc::vec::Vec;

use dapple_field::scoped::{CompareOp, Node, Scope, ScopedBuilder};
use dapple_field::{PortType, Value};
use dapple_material::module::{
    Args, Context, InputDecl, InputKind, Interface, Module, ModuleError, ModuleId, Output,
    OutputDecl, OutputKind, Outputs,
};
use dapple_material::ops::{self, Deposit};
use dapple_material::program::{MapBinding, evaluate_mask};
use dapple_material::{Aux, Channel, ChannelId, Grid, Material, Param};
use dapple_raster::{
    Advect, AmbientOcclusion, Flow, GaussianBlur, HeightToNormal, Raster, RasterOp, Streak,
};
use glam::Vec3;

use super::{
    color, coverage, fail, fbm, fraction, integer, mask_map, meters, realize_scalar, scalar,
    scalar_map, seed,
};
use crate::glazed_brick::{DIRT, MORTAR, SALT};

fn base_input() -> InputDecl {
    InputDecl {
        name: "base",
        kind: InputKind::Material,
        required: true,
        doc: "the material weathered",
    }
}

fn material_output() -> Vec<OutputDecl> {
    vec![OutputDecl {
        name: "material",
        kind: OutputKind::Material,
        doc: "the weathered material",
    }]
}

/// The deposit's own material: `color` varied by slow noise, rough, with
/// surface identity `surface`.
fn deposit_material(
    grid: Grid,
    args: &Args,
    color: Vec3,
    roughness: f32,
    surface: u32,
) -> Result<Material, ModuleError> {
    let tone = realize_scalar(grid, |b, d| fbm(b, d, [14.0, 14.0], args.seed("tone"), 3))?;
    let colors: Vec<Vec3> = tone
        .values()
        .iter()
        .map(|t| color * (1.0 + 0.25 * t))
        .collect();
    let mut m = Material::new(grid);
    m.set_param(Param::BaseColor, super::color_map(grid, &colors)?)?;
    m.set_param(
        Param::SpecularRoughness,
        Channel::Constant(Value::Scalar(roughness)),
    )?;
    m.set_aux(Aux::Surface, Channel::Constant(Value::Id(surface)))?;
    Ok(m)
}

fn scaled(grid: Grid, r: &Raster, k: f32) -> Result<Raster, ModuleError> {
    Ok(grid.raster(r.values().iter().map(|v| v * k).collect())?)
}

/// Dirt settled where it is sheltered and where it is thrown: in hollows
/// that ambient occlusion finds, in cavities narrower than `radius` (chips,
/// pits, the joint's corners), in broad patches, and as splash-back speckle
/// within `splash` of the foot, with the face cleaner toward its top by
/// `clean_top`; `coverage` of the surface at most `strength` thick.
#[derive(Copy, Clone, Debug, Default)]
pub struct Grime;

impl Module for Grime {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId {
                name: "dapple_library.grime",
                version: 1,
            },
            doc: "dirt from occlusion and cavities",
            params: vec![
                color(
                    "color",
                    Vec3::new(0.12, 0.1, 0.075),
                    "the dirt's color: soil and soot, darker than cream glaze and lighter than brown",
                ),
                fraction("coverage", 0.3, "share of the surface dirtied"),
                meters("radius", [0.001, 0.05], 0.008, "the hollows' scale"),
                fraction("strength", 0.65, "coverage at its densest"),
                fraction("patchiness", 0.5, "how much broad patches decide"),
                meters(
                    "splash",
                    [0.0, 5.0],
                    0.35,
                    "height above the foot that rain splashes back onto; 0 for none",
                ),
                scalar(
                    "splash_weight",
                    dapple_material::module::Unit::None,
                    [0.0, 4.0],
                    1.6,
                    "how much the splash zone decides",
                ),
                color(
                    "splash_color",
                    Vec3::new(0.2, 0.16, 0.11),
                    "the splashed soil's color, lighter than soot",
                ),
                fraction("clean_top", 0.5, "how much cleaner the face is at its top"),
                meters("thickness", [0.0, 0.001], 0.00004, "the dirt's thickness"),
                seed(),
            ],
            inputs: vec![base_input()],
            outputs: material_output(),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let base = args.material("base").ok_or_else(|| fail("base"))?;
        let h = base.height()?;
        let radius = args.scalar("radius");
        let ao = AmbientOcclusion {
            radius,
            directions: 12,
            scale: 1.0,
        }
        .apply(&h)?;
        let blurred = GaussianBlur {
            sigma: radius * 0.5,
        }
        .apply(&h)?;
        let cavity = grid.raster(
            blurred
                .values()
                .iter()
                .zip(h.values())
                .map(|(b, h)| b - h)
                .collect(),
        )?;
        let patches = realize_scalar(grid, |b, d| fbm(b, d, [4.0, 4.0], args.seed("patches"), 4))?;
        let fine_map =
            realize_scalar(grid, |b, d| fbm(b, d, [150.0, 150.0], args.seed("fine"), 3))?;
        #[expect(clippy::cast_precision_loss, reason = "grid sizes are small")]
        let extent_m = grid.texel.y * grid.height as f32;
        let splash_m = args.scalar("splash");

        // The score: occlusion, cavity depth in millimeters, patches.
        let mut p = ScopedBuilder::new("dapple_library.grime.score");
        let ao_in = p.input("occlusion", PortType::Mask, Scope::Pass)?;
        let cav = p.input("cavity", PortType::Scalar, Scope::Pass)?;
        let patch = p.input("patches", PortType::Scalar, Scope::Sample)?;
        let patchiness = p.input("patchiness", PortType::Scalar, Scope::Material)?;
        let at = p.input("position", PortType::Vector2, Scope::Sample)?;
        let fine = p.input("fine", PortType::Scalar, Scope::Sample)?;
        let foot = p.input("foot", PortType::Scalar, Scope::Material)?;
        let splash = p.input("splash", PortType::Scalar, Scope::Material)?;
        let splash_w = p.input("splash_weight", PortType::Scalar, Scope::Material)?;
        let extent = p.input("extent", PortType::Scalar, Scope::Material)?;
        let clean_top = p.input("clean_top", PortType::Scalar, Scope::Material)?;
        let one = p.scalar(1.0);
        let closed = p.sub(one, ao_in)?;
        let closed = p.scale(0.5, closed)?;
        let cav_mm = p.scale(1000.0, cav)?;
        let cav_mm = p.saturate(cav_mm)?;
        let score = p.add_scaled(closed, 0.7, cav_mm)?;
        let broad = p.mul(patch, patchiness)?;
        let score = p.add_scaled(score, 0.6, broad)?;
        // Where it is: splash-back speckle at the foot, a cleaner top.
        let y = p.component(at, 1)?;
        let up = p.sub(y, foot)?;
        let low = p.node(Node::Div(up, splash))?;
        let low = p.sub(one, low)?;
        let low = p.saturate(low)?;
        let speckle = p.smoothstep(-0.2, 0.6, fine)?;
        let speckle = p.lerp(0.4, 1.0, speckle)?;
        let low = p.mul(low, speckle)?;
        let low = p.mul(low, splash_w)?;
        let score = p.add(score, low)?;
        let height = p.node(Node::Div(up, extent))?;
        let height = p.saturate(height)?;
        let fade = p.mul(height, clean_top)?;
        let fade = p.sub(one, fade)?;
        let score = p.mul(score, fade)?;
        p.output("score", PortType::Scalar, Scope::Pass, score)?;
        let program = p.finish();
        let score = evaluate_mask(
            &program,
            &[
                MapBinding::Pass(mask_map(grid, &ao)?),
                MapBinding::Pass(scalar_map(grid, &cavity)?),
                MapBinding::Map(scalar_map(grid, &patches)?),
                MapBinding::Constant(Value::Scalar(args.scalar("patchiness"))),
                MapBinding::Position,
                MapBinding::Map(scalar_map(grid, &fine_map)?),
                MapBinding::Constant(Value::Scalar(grid.origin.y)),
                MapBinding::Constant(Value::Scalar(if splash_m > 0.0 { splash_m } else { 1e9 })),
                MapBinding::Constant(Value::Scalar(args.scalar("splash_weight"))),
                MapBinding::Constant(Value::Scalar(extent_m)),
                MapBinding::Constant(Value::Scalar(args.scalar("clean_top"))),
            ],
            base,
        )?;
        let (score, score_tiling) = score;
        let covered = coverage(&score, args.scalar("coverage"), 0.12)?;
        let covered = scaled(grid, &covered, args.scalar("strength"))?;
        let mut dirt = deposit_material(grid, args, args.color("color"), 0.92, DIRT)?;
        if splash_m > 0.0 {
            // Soil splashed up from the ground is lighter than soot.
            let splash_color = args.color("splash_color");
            let colors: Vec<Vec3> = (0..grid.len())
                .map(|i| {
                    let c = match dirt.value(ChannelId::Param(Param::BaseColor), i) {
                        Value::Vector3(c) => c,
                        _ => args.color("color"),
                    };
                    let y = grid.center(i).y - grid.origin.y;
                    let low = (1.0 - y / splash_m).clamp(0.0, 1.0);
                    c.lerp(splash_color, low)
                })
                .collect();
            dirt.set_param(Param::BaseColor, super::color_map(grid, &colors)?)?;
        }
        let mut m = cx.record(ops::deposit(
            base,
            &Deposit {
                material: dirt,
                coverage: covered,
                thickness: args.scalar("thickness"),
                relief: None,
                matting: 3.0,
            },
        )?);
        // Splash-back and a cleaner top tie the dirt to the face's foot and
        // top; the score's evaluation across the wrap found where it tiles.
        m.set_tiling(m.tiling().and(score_tiling));
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}

/// Grime washed down a face from its ledges: upward-facing surfaces (the
/// lower lips of joints, a sill's drip) shed trails that run `length` down
/// the face and fade, some ledges more than others, in vertical runs,
/// carried along a wandering downward flow ([`Advect`]) so they bend;
/// `coverage` of the surface at most `strength` thick. A `sources` mask
/// adds ledges the height does not show.
#[derive(Copy, Clone, Debug, Default)]
pub struct Streaks;

impl Module for Streaks {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId {
                name: "dapple_library.streaks",
                version: 1,
            },
            doc: "grime streaks below ledges",
            params: vec![
                color("color", Vec3::new(0.03, 0.027, 0.022), "the grime's color"),
                fraction("coverage", 0.12, "share of the surface streaked"),
                meters(
                    "length",
                    [0.005, 2.0],
                    0.22,
                    "how far a trail runs before fading by 1/e",
                ),
                fraction("strength", 0.6, "coverage at its densest"),
                meters("thickness", [0.0, 0.001], 0.00002, "the grime's thickness"),
                seed(),
            ],
            inputs: vec![
                base_input(),
                InputDecl {
                    name: "sources",
                    kind: InputKind::Map(PortType::Mask),
                    required: false,
                    doc: "more ledges, where grime starts",
                },
            ],
            outputs: material_output(),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let base = args.material("base").ok_or_else(|| fail("base"))?;
        // Ledges at the scale of units and sills, not of tooling or grain.
        let smooth = GaussianBlur { sigma: 0.003 }.apply(&base.height()?)?;
        let normals = HeightToNormal { scale: 1.0 }.apply(&smooth)?;
        let gate = realize_scalar(grid, |b, d| fbm(b, d, [7.0, 3.0], args.seed("gate"), 3))?;
        let extra = args.map("sources");
        let mut sources = Vec::with_capacity(grid.len());
        for (i, n) in normals.values().iter().enumerate() {
            // Upward-facing: the normal's +y (the domain's up).
            let ledge = ((n[1] - 0.12) * 5.0).clamp(0.0, 1.0);
            let shed = super::smoothstep(0.15, 0.45, gate.values()[i]);
            let more = extra.map_or(0.0, |m| m.value(i).component(0).unwrap_or(0.0));
            sources.push((ledge * shed).max(more));
        }
        let trails = Streak {
            flow: Flow::NegativeY,
            length: args.scalar("length"),
        }
        .apply(&grid.raster(sources)?)?;
        let runs = realize_scalar(grid, |b, d| fbm(b, d, [70.0, 2.0], args.seed("runs"), 3))?;
        let score: Vec<f32> = trails
            .values()
            .iter()
            .zip(runs.values())
            .map(|(t, r)| t * (0.35 + 0.65 * super::smoothstep(-0.4, 0.5, *r)))
            .collect();
        // Water does not run plumb: carry the trails along a flow that runs
        // down the face and wanders sideways, so they bend and soften.
        let wander = realize_scalar(grid, |b, d| fbm(b, d, [12.0, 6.0], args.seed("wander"), 3))?;
        let flow = grid.raster(wander.values().iter().map(|w| [0.6 * w, -1.0]).collect())?;
        let score = Advect {
            step: 2.0 * grid.texel.y,
            steps: 6,
        }
        .apply(&grid.raster(score)?, &flow)?;
        let covered = coverage(&score, args.scalar("coverage"), 0.1)?;
        let covered = scaled(grid, &covered, args.scalar("strength"))?;
        let grime = deposit_material(grid, args, args.color("color"), 0.85, DIRT)?;
        let m = cx.record(ops::deposit(
            base,
            &Deposit {
                material: grime,
                coverage: covered,
                thickness: args.scalar("thickness"),
                relief: None,
                matting: 4.0,
            },
        )?);
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}

/// Efflorescence: salts carried up by damp and left where it dries, on one
/// surface material (mortar by default), bleeding `spread` onto its
/// neighbors' edges, densest at the foot and gone `rise` above it, in
/// patches; `coverage` of the surface.
#[derive(Copy, Clone, Debug, Default)]
pub struct Efflorescence;

impl Module for Efflorescence {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId {
                name: "dapple_library.efflorescence",
                version: 1,
            },
            doc: "salt bloom rising from the foot",
            params: vec![
                color("color", Vec3::new(0.50, 0.49, 0.46), "the salts' color"),
                fraction("coverage", 0.08, "share of the surface bloomed"),
                meters(
                    "rise",
                    [0.01, 5.0],
                    0.4,
                    "height above the foot where the bloom ends",
                ),
                meters(
                    "spread",
                    [0.0, 0.02],
                    0.003,
                    "how far it bleeds onto neighbors",
                ),
                integer(
                    "target",
                    [0, 1 << 16],
                    MORTAR,
                    "the surface identity it forms on",
                ),
                fraction("strength", 0.55, "coverage at its densest"),
                meters("thickness", [0.0, 0.001], 0.00008, "the salts' thickness"),
                seed(),
            ],
            inputs: vec![base_input()],
            outputs: material_output(),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let base = args.material("base").ok_or_else(|| fail("base"))?;
        // Where the target surface is: an identifier comparison.
        let mut p = ScopedBuilder::new("dapple_library.efflorescence.target");
        let surface = p.input("surface", PortType::Id, Scope::Sample)?;
        let target = p.input("target", PortType::Id, Scope::Material)?;
        let on = p.node(Node::Compare(CompareOp::Equal, surface, target))?;
        p.output("on", PortType::Mask, Scope::Sample, on)?;
        let on = evaluate_mask(
            &p.finish(),
            &[
                MapBinding::Channel(ChannelId::Aux(Aux::Surface)),
                MapBinding::Constant(Value::Id(args.integer("target"))),
            ],
            base,
        )?
        .0;
        let spread = args.scalar("spread");
        let near = if spread > 0.0 {
            GaussianBlur { sigma: spread }.apply(&on)?
        } else {
            on
        };
        let patches = realize_scalar(grid, |b, d| fbm(b, d, [7.0, 7.0], args.seed("patches"), 5))?;
        let fine = realize_scalar(grid, |b, d| fbm(b, d, [90.0, 90.0], args.seed("fine"), 3))?;

        // The score: near the target, low on the face, in patches.
        let mut p = ScopedBuilder::new("dapple_library.efflorescence.score");
        let near_in = p.input("near", PortType::Scalar, Scope::Pass)?;
        let at = p.input("position", PortType::Vector2, Scope::Sample)?;
        let patch = p.input("patches", PortType::Scalar, Scope::Sample)?;
        let fine_in = p.input("fine", PortType::Scalar, Scope::Sample)?;
        let foot = p.input("foot", PortType::Scalar, Scope::Material)?;
        let rise = p.input("rise", PortType::Scalar, Scope::Material)?;
        let y = p.component(at, 1)?;
        let up = p.sub(y, foot)?;
        let up = p.node(Node::Div(up, rise))?;
        let one = p.scalar(1.0);
        let damp = p.sub(one, up)?;
        let damp = p.saturate(damp)?;
        let damp = p.mul(damp, damp)?;
        let patchy = p.smoothstep(0.0, 0.6, patch)?;
        let k = p.mul(near_in, damp)?;
        let k = p.mul(k, patchy)?;
        let grainy = p.add_scaled(one, 0.8, fine_in)?;
        let k = p.mul(k, grainy)?;
        p.output("score", PortType::Scalar, Scope::Pass, k)?;
        let score = evaluate_mask(
            &p.finish(),
            &[
                MapBinding::Pass(scalar_map(grid, &near)?),
                MapBinding::Position,
                MapBinding::Map(scalar_map(grid, &patches)?),
                MapBinding::Map(scalar_map(grid, &fine)?),
                MapBinding::Constant(Value::Scalar(grid.origin.y)),
                MapBinding::Constant(Value::Scalar(args.scalar("rise"))),
            ],
            base,
        )?;
        let (score, score_tiling) = score;
        let covered = coverage(&score, args.scalar("coverage"), 0.08)?;
        let covered = scaled(grid, &covered, args.scalar("strength"))?;
        let salt = deposit_material(grid, args, args.color("color"), 0.97, SALT)?;
        let mut m = cx.record(ops::deposit(
            base,
            &Deposit {
                material: salt,
                coverage: covered,
                thickness: args.scalar("thickness"),
                relief: None,
                matting: 3.0,
            },
        )?);
        // Rising damp is tied to the foot: the bloom tiles along the face,
        // as evaluating its score across the wrap finds.
        m.set_tiling(m.tiling().and(score_tiling));
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}
