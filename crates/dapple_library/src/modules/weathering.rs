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
use dapple_field::{Edge, PortType, Value};
use dapple_material::module::{
    Args, Context, InputDecl, InputKind, Interface, Module, ModuleError, ModuleId, Output,
    OutputDecl, OutputKind, Outputs,
};
use dapple_material::ops::{self, Deposit};
use dapple_material::program::{MapBinding, evaluate_mask};
use dapple_material::{Aux, Channel, ChannelId, Grid, Material, Param, Tiling};
use dapple_raster::{
    Advect, AmbientOcclusion, Flow, GaussianBlur, HeightToNormal, Raster, RasterOp, Streak,
};
use glam::{Vec2, Vec3};

use super::{
    color, coverage, fail, fbm, fraction, integer, mask_map, meters, realize_scalar, scalar,
    scalar_map, seed,
};
use crate::glazed_brick::{DIRT, MORTAR, SALT};

fn base_input() -> InputDecl {
    InputDecl {
        name: "base".into(),
        kind: InputKind::Material,
        required: true,
        doc: "the material weathered".into(),
    }
}

fn material_output() -> Vec<OutputDecl> {
    vec![OutputDecl {
        name: "material".into(),
        kind: OutputKind::Material,
        doc: "the weathered material".into(),
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
            id: ModuleId::new("dapple_library.grime", 1),
            doc: "dirt from occlusion and cavities".into(),
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

/// How far [`extend`] carries a raster past each edge of `grid` so that an
/// operation reading up to `reach` texels away stops at the edges `tiling`
/// does not cross: `reach` along those axes of a wrapping grid, nothing
/// along the rest (a clamping grid stops at every edge already).
fn apron(grid: Grid, tiling: Tiling, reach: u32) -> [u32; 2] {
    if grid.edge != Edge::Wrap {
        return [0, 0];
    }
    [
        if tiling.x { 0 } else { reach },
        if tiling.y { 0 } else { reach },
    ]
}

/// `r` carried `apron` texels past each edge, each new texel a copy of the
/// nearest edge texel, still wrapping: an operation reading at most `apron`
/// texels away sees `r` clamp across those edges and wrap across the rest.
/// [`trim`] takes the original texels back.
fn extend<T: Copy>(r: &Raster<T>, apron: [u32; 2]) -> Result<Raster<T>, ModuleError> {
    let [ax, ay] = apron;
    if ax == 0 && ay == 0 {
        return Ok(r.clone());
    }
    let (w, h) = (r.width() as usize, r.height() as usize);
    let (ax, ay) = (ax as usize, ay as usize);
    let mut values = Vec::with_capacity((w + 2 * ax) * (h + 2 * ay));
    for y in 0..h + 2 * ay {
        let row = y.saturating_sub(ay).min(h - 1) * w;
        for x in 0..w + 2 * ax {
            values.push(r.values()[row + x.saturating_sub(ax).min(w - 1)]);
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "aprons are small")]
    let shift = r.texel() * Vec2::new(ax as f32, ay as f32);
    Ok(Raster::from_values(
        r.width() + 2 * apron[0],
        r.height() + 2 * apron[1],
        r.origin() - shift,
        r.texel(),
        r.edge(),
        values,
    )?)
}

/// The texels of `grid` within `r`, a raster [`extend`] carried `apron`
/// past them.
fn trim<T: Copy>(grid: Grid, r: &Raster<T>, apron: [u32; 2]) -> Result<Raster<T>, ModuleError> {
    let (w, ax, ay) = (r.width() as usize, apron[0] as usize, apron[1] as usize);
    let width = grid.width as usize;
    let values = (0..grid.height as usize)
        .flat_map(|y| {
            let start = (y + ay) * w + ax;
            r.values()[start..start + width].iter().copied()
        })
        .collect();
    Ok(grid.raster(values)?)
}

/// How strongly each texel is a ledge that sheds water: the surface faces
/// up (its normal's `+y`, the domain's up) at the scale of units, sills and
/// laps, not of tooling or grain; in `[0, 1]`.
///
/// Past an edge the base does not tile across ([`Material::tiling`]) the
/// face ends: its height is taken to run on flat there rather than wrap to
/// the far edge, so the step between a sill's top and a wall's foot that
/// only the wrap puts side by side is no ledge.
pub(crate) fn ledges(base: &Material) -> Result<Raster, ModuleError> {
    let grid = base.grid();
    let blur = GaussianBlur { sigma: 0.003 };
    let normal = HeightToNormal { scale: 1.0 };
    let reach = |footprint: Option<[u32; 2]>| {
        footprint
            .map(|[x, y]| x.max(y))
            .ok_or_else(|| fail("a ledge's footprint is unbounded"))
    };
    let apron = apron(
        grid,
        base.tiling(),
        reach(blur.footprint(grid.texel))? + reach(normal.footprint(grid.texel))?,
    );
    let smooth = blur.apply(&extend(&base.height()?, apron)?)?;
    let normals = trim(grid, &normal.apply(&smooth)?, apron)?;
    Ok(grid.raster(
        normals
            .values()
            .iter()
            .map(|n| ((n[1] - 0.12) * 5.0).clamp(0.0, 1.0))
            .collect(),
    )?)
}

/// Water's trails from `sources`, a mask: each runs `length` down the face
/// (domain `−y`) and fades by `1/e`, some more strongly than others in
/// vertical runs, carried along a downward flow that wanders sideways
/// ([`Advect`]) so trails bend and soften. A trail is as strong as its
/// source, so what runs off is what was there to run off; the seeds are
/// `args`' `runs` and `wander`.
///
/// Water crosses only the edges `tiling` (the face's [`Material::tiling`])
/// tiles across: on a face that does not tile up it, a trail leaves at the
/// foot and none comes in over the top.
pub(crate) fn runoff(
    grid: Grid,
    args: &Args,
    sources: &Raster,
    length: f32,
    tiling: Tiling,
) -> Result<Raster, ModuleError> {
    let streak = Streak {
        flow: Flow::NegativeY,
        length,
    };
    // A streak scans along y alone, so clamping is all it needs to stop at
    // the top and foot.
    let trails = if tiling.y {
        streak.apply(sources)?
    } else {
        let clamped = Raster::from_values(
            sources.width(),
            sources.height(),
            sources.origin(),
            sources.texel(),
            Edge::Clamp,
            sources.values().to_vec(),
        )?;
        streak.apply(&clamped)?
    };
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
    let advect = Advect {
        step: 2.0 * grid.texel.y,
        steps: 6,
    };
    // A path runs at most `steps · step` (the flow is clamped to unit
    // length), and its last sample reads one texel further.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "a reach of a few texels"
    )]
    let reach =
        libm::ceilf(advect.steps as f32 * advect.step / grid.texel.min_element()) as u32 + 1;
    let apron = apron(grid, tiling, reach);
    let carried = advect.apply(
        &extend(&grid.raster(score)?, apron)?,
        &extend(&flow, apron)?,
    )?;
    trim(grid, &carried, apron)
}

/// Grime washed down a face from its ledges: upward-facing surfaces (the
/// lower lips of joints, a sill's drip) shed trails that run `length` down
/// the face and fade, some ledges more than others, in vertical runs,
/// carried along a wandering downward flow ([`Advect`]) so they bend;
/// `coverage` of the surface at most `strength` thick. A `sources` mask
/// adds ledges the height does not show. On a base that tiles along the
/// face but not up it, water neither sheds from nor runs across the tile's
/// top and foot.
#[derive(Copy, Clone, Debug, Default)]
pub struct Streaks;

impl Module for Streaks {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId::new("dapple_library.streaks", 1),
            doc: "grime streaks below ledges".into(),
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
                    name: "sources".into(),
                    kind: InputKind::Map(PortType::Mask),
                    required: false,
                    doc: "more ledges, where grime starts".into(),
                },
            ],
            outputs: material_output(),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let base = args.material("base").ok_or_else(|| fail("base"))?;
        let ledges = ledges(base)?;
        let gate = realize_scalar(grid, |b, d| fbm(b, d, [7.0, 3.0], args.seed("gate"), 3))?;
        let extra = args.map("sources");
        let mut sources = Vec::with_capacity(grid.len());
        for (i, &ledge) in ledges.values().iter().enumerate() {
            let shed = super::smoothstep(0.15, 0.45, gate.values()[i]);
            let more = extra.map_or(0.0, |m| m.value(i).component(0).unwrap_or(0.0));
            sources.push((ledge * shed).max(more));
        }
        let score = runoff(
            grid,
            args,
            &grid.raster(sources)?,
            args.scalar("length"),
            base.tiling(),
        )?;
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
            id: ModuleId::new("dapple_library.efflorescence", 1),
            doc: "salt bloom rising from the foot".into(),
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

#[cfg(test)]
mod tests {
    use dapple_material::module::{Bind, Context, Output};
    use dapple_material::resource::NoResources;

    use super::*;

    /// 32² texels over a meter, wrapping.
    fn grid() -> Grid {
        Grid {
            width: 32,
            height: 32,
            origin: Vec2::ZERO,
            texel: Vec2::splat(1.0 / 32.0),
            edge: Edge::Wrap,
        }
    }

    /// The rows of `r` from `rows`, flattened.
    fn rows(r: &Raster, rows: core::ops::Range<u32>) -> Vec<f32> {
        let w = r.width() as usize;
        r.values()[rows.start as usize * w..rows.end as usize * w].to_vec()
    }

    /// Runs [`runoff`] from `sources` on a face that tiles along `tiling`.
    struct Trails {
        sources: Raster,
        tiling: Tiling,
    }

    impl Module for Trails {
        fn interface(&self) -> Interface {
            Interface {
                id: ModuleId::new("test.trails", 1),
                doc: "runoff's trails".into(),
                params: vec![seed()],
                inputs: vec![],
                outputs: vec![OutputDecl {
                    name: "trails".into(),
                    kind: OutputKind::Map(PortType::Scalar),
                    doc: "the trails".into(),
                }],
            }
        }

        fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
            let grid = cx.grid();
            let trails = runoff(grid, args, &self.sources, 0.3, self.tiling)?;
            Ok(Outputs::new().with("trails", Output::Map(scalar_map(grid, &trails)?)))
        }
    }

    fn trails(sources: &Raster, tiling: Tiling) -> Raster {
        let module = Trails {
            sources: sources.clone(),
            tiling,
        };
        let mut cx = Context::new(grid(), &NoResources);
        let out = cx.instantiate(&module, "trails", Bind::new()).unwrap();
        let Some(Output::Map(m)) = out.get("trails") else {
            panic!("a map");
        };
        super::super::scalar_of(m).unwrap()
    }

    /// Water runs on past the foot of a face that tiles up it, into the
    /// top of the next repeat; on one that tiles only along it, a trail
    /// leaves at the foot and none comes in over the top.
    #[test]
    fn runoff_crosses_only_the_edges_the_face_tiles_across() {
        let g = grid();
        let sources = g
            .raster((0..g.len()).map(|i| f32::from(u8::from(i < 64))).collect())
            .unwrap();
        let wrapped = trails(&sources, Tiling::BOTH);
        let bounded = trails(&sources, Tiling::X);
        let top = |r: &Raster| rows(r, 24..32).into_iter().fold(0.0, f32::max);
        assert!(top(&wrapped) > 0.2, "tiling up it: {}", top(&wrapped));
        assert_eq!(top(&bounded), 0.0, "tiling only along it");
        assert!(rows(&bounded, 3..32).iter().all(|&t| t == 0.0), "or above");
        assert!(rows(&bounded, 0..2).iter().all(|&t| t > 0.0), "at the foot");
    }

    /// The top of a sill meets the foot of the wall below it only across
    /// the wrap: a ledge there on a face that tiles up it, none on one
    /// that tiles only along it (the stone sill's dark drips that ran down
    /// its face from the top of the tile).
    #[test]
    fn ledges_stop_at_the_edges_the_face_does_not_tile_across() {
        let g = grid();
        let sill = |tiling: Tiling| {
            let mut m = Material::new(g);
            let h = (0..g.len()).map(|i| Value::Scalar(if i >= 24 * 32 { 0.02 } else { 0.0 }));
            m.set_aux(
                Aux::Height,
                Channel::Map(g.typed(PortType::Scalar, h).unwrap()),
            )
            .unwrap();
            m.set_tiling(tiling);
            ledges(&m).unwrap()
        };
        let most = |r: &Raster| r.values().iter().copied().fold(0.0, f32::max);
        let wrapped = sill(Tiling::BOTH);
        assert!(
            rows(&wrapped, 30..32).iter().any(|&l| l > 0.5),
            "across the wrap, the sill's top is a ledge"
        );
        // The sill's lower edge faces down: no ledge anywhere.
        assert_eq!(most(&sill(Tiling::X)), 0.0, "the face ends at its top");
    }
}
