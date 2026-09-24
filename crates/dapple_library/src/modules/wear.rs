// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Edge wear, moss, and by-example texture.

use alloc::vec;
use alloc::vec::Vec;

use dapple_field::scoped::{Scope, ScopedBuilder};
use dapple_field::{PortType, Primaries, Value};
use dapple_material::module::{
    Args, Context, InputDecl, InputKind, Interface, Module, ModuleError, ModuleId, Output,
    OutputDecl, OutputKind, Outputs, Unit,
};
use dapple_material::ops::{self, Deposit, Transition};
use dapple_material::program::{MapBinding, evaluate_mask};
use dapple_material::resource::ResourceRequest;
use dapple_material::{Aux, Channel, ChannelId, Material, Param};
use dapple_raster::synthesis::ByExample as Synthesis;
use dapple_raster::{GaussianBlur, HeightToNormal, Raster, RasterOp};
use glam::{Vec2, Vec3};

use super::{
    color, coverage, fail, fbm, fraction, integer, meters, realize_scalar, scalar, scalar_map, seed,
};
use crate::glazed_brick::{MOSS, STONE};

fn base_input() -> InputDecl {
    InputDecl {
        name: "base".into(),
        kind: InputKind::Material,
        required: true,
        doc: "the material worn or grown over".into(),
    }
}

fn material_output() -> Vec<OutputDecl> {
    vec![OutputDecl {
        name: "material".into(),
        kind: OutputKind::Material,
        doc: "the result".into(),
    }]
}

/// Convexity: how far the height stands above its blur over `radius`, in
/// meters; positive on ridges and arrises, negative in hollows.
fn convexity(base: &Material, radius: f32) -> Result<Raster, ModuleError> {
    let grid = base.grid();
    let h = base.height()?;
    let blurred = GaussianBlur { sigma: radius }.apply(&h)?;
    Ok(grid.raster(
        h.values()
            .iter()
            .zip(blurred.values())
            .map(|(h, b)| h - b)
            .collect(),
    )?)
}

/// Wear on edges: where the surface is convex (arrises, ridges, grain
/// standing proud) at the scale of `radius`, broken up by noise, `coverage`
/// of the surface wears through to `worn`, the material beneath, selected
/// over the base with one decision for every channel, on the surface
/// identity `surface` alone when it is not 0. Without `worn`, the
/// base itself wears: its coat goes, it roughens by `roughening` and
/// lightens by `lighten`.
#[derive(Copy, Clone, Debug, Default)]
pub struct EdgeWear;

impl Module for EdgeWear {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId::new("dapple_library.edge_wear", 1),
            doc: "wear on convex edges, from curvature".into(),
            params: vec![
                meters(
                    "radius",
                    [0.0005, 0.05],
                    0.003,
                    "the scale of the edges worn",
                ),
                fraction("coverage", 0.03, "share of the surface worn through"),
                fraction("strength", 1.0, "how fully it wears where it wears"),
                fraction(
                    "roughening",
                    0.25,
                    "roughness added where worn, without `worn`",
                ),
                fraction("lighten", 0.15, "lightening where worn, without `worn`"),
                integer(
                    "surface",
                    [0, 1 << 16],
                    0,
                    "the surface identity that wears; 0 for any",
                ),
                seed(),
            ],
            inputs: vec![
                base_input(),
                InputDecl {
                    name: "worn".into(),
                    kind: InputKind::Material,
                    required: false,
                    doc: "what wear exposes".into(),
                },
            ],
            outputs: material_output(),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let base = args.material("base").ok_or_else(|| fail("base"))?;
        let convex = convexity(base, args.scalar("radius"))?;
        let breakup = realize_scalar(grid, |b, d| {
            fbm(b, d, [60.0, 60.0], args.seed("breakup"), 3)
        })?;
        // The score: convexity in tenths of a millimeter, broken up.
        let mut p = ScopedBuilder::new("dapple_library.edge_wear.score");
        let convex_in = p.input("convexity", PortType::Scalar, Scope::Pass)?;
        let noise = p.input("breakup", PortType::Scalar, Scope::Sample)?;
        let k = p.scale(10_000.0, convex_in)?;
        let k = p.saturate(k)?;
        let n = p.smoothstep(-0.5, 0.5, noise)?;
        let n = p.lerp(0.3, 1.0, n)?;
        let score = p.mul(k, n)?;
        p.output("score", PortType::Scalar, Scope::Pass, score)?;
        let score = evaluate_mask(
            &p.finish(),
            &[
                MapBinding::Pass(scalar_map(grid, &convex)?),
                MapBinding::Map(scalar_map(grid, &breakup)?),
            ],
            base,
        )?;
        let (score, score_tiling) = score;
        let target = args.integer("surface");
        let score = if target == 0 {
            score
        } else {
            grid.raster(
                score
                    .values()
                    .iter()
                    .enumerate()
                    .map(
                        |(i, &s)| match base.value(ChannelId::Aux(Aux::Surface), i) {
                            Value::Id(id) if id == target => s,
                            _ => 0.0,
                        },
                    )
                    .collect(),
            )?
        };
        let worn_mask = coverage(&score, args.scalar("coverage"), 0.01)?;
        let strength = args.scalar("strength");
        let worn_mask = grid.raster(worn_mask.values().iter().map(|v| v * strength).collect())?;
        let worn = match args.material("worn") {
            Some(w) => w.clone(),
            None => {
                let mut w = base.clone();
                w.set_param(Param::CoatWeight, Channel::Constant(Value::Scalar(0.0)))?;
                let (gain, lighten) = (args.scalar("roughening"), args.scalar("lighten"));
                let rough: Vec<f32> = (0..grid.len())
                    .map(|i| {
                        let r = base.value(ChannelId::Param(Param::SpecularRoughness), i);
                        (r.component(0).unwrap_or(0.3) + gain).min(1.0)
                    })
                    .collect();
                w.set_param(
                    Param::SpecularRoughness,
                    super::scalar_channel(grid, &rough)?,
                )?;
                let colors: Vec<Vec3> = (0..grid.len())
                    .map(
                        |i| match base.value(ChannelId::Param(Param::BaseColor), i) {
                            Value::Vector3(c) => (c * (1.0 + lighten)).min(Vec3::ONE),
                            _ => Vec3::splat(0.5),
                        },
                    )
                    .collect();
                w.set_param(Param::BaseColor, super::color_map(grid, &colors)?)?;
                w
            }
        };
        let mut m = cx.record(ops::select(base, &worn, &worn_mask, Transition::Mask)?);
        m.set_tiling(m.tiling().and(score_tiling));
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}

/// Moss by orientation: it settles where the surface faces up (the normal
/// against `up`, in the domain frame: `+y` is up a wall, `+z` out of a
/// floor), in hollows and joints, low on the face where it stays damp, in
/// clumps; `coverage` of the surface, lumpy, `thickness` deep, with a
/// velvet sheen (OpenPBR fuzz) over its green.
#[derive(Copy, Clone, Debug, Default)]
pub struct Moss;

impl Module for Moss {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId::new("dapple_library.moss", 1),
            doc: "moss where the surface faces up, is hollow or damp".into(),
            params: vec![
                color("color", Vec3::new(0.05, 0.085, 0.02), "the moss's color"),
                color(
                    "sheen",
                    Vec3::new(0.16, 0.2, 0.07),
                    "its velvet sheen (fuzz color)",
                ),
                fraction("coverage", 0.04, "share of the surface grown over"),
                scalar(
                    "up_y",
                    Unit::None,
                    [-1.0, 1.0],
                    1.0,
                    "up's y in the domain frame",
                ),
                scalar(
                    "up_z",
                    Unit::None,
                    [-1.0, 1.0],
                    0.0,
                    "up's z (out of the surface)",
                ),
                meters(
                    "damp",
                    [0.0, 5.0],
                    0.3,
                    "height above the foot where it stays damp; 0 for none",
                ),
                meters("thickness", [0.0, 0.01], 0.0012, "the moss's thickness"),
                meters("lumps", [0.0, 0.01], 0.0015, "the height of its clumps"),
                seed(),
            ],
            inputs: vec![base_input()],
            outputs: material_output(),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let base = args.material("base").ok_or_else(|| fail("base"))?;
        let normals = HeightToNormal { scale: 1.0 }.apply(&base.height()?)?;
        let up = Vec3::new(0.0, args.scalar("up_y"), args.scalar("up_z")).normalize_or(Vec3::Y);
        let facing = grid.raster(
            normals
                .values()
                .iter()
                .map(|n| Vec3::from_array(*n).dot(up))
                .collect(),
        )?;
        let hollow = convexity(base, 0.004)?;
        let clumps = realize_scalar(grid, |b, d| fbm(b, d, [25.0, 25.0], args.seed("clumps"), 4))?;
        let damp = args.scalar("damp");

        let mut p = ScopedBuilder::new("dapple_library.moss.score");
        let facing_in = p.input("facing", PortType::Scalar, Scope::Pass)?;
        let hollow_in = p.input("hollow", PortType::Scalar, Scope::Pass)?;
        let clump = p.input("clumps", PortType::Scalar, Scope::Sample)?;
        let at = p.input("position", PortType::Vector2, Scope::Sample)?;
        let foot = p.input("foot", PortType::Scalar, Scope::Material)?;
        let damp_in = p.input("damp", PortType::Scalar, Scope::Material)?;
        let face = p.smoothstep(0.0, 0.3, facing_in)?;
        let hole = p.scale(-2000.0, hollow_in)?;
        let hole = p.saturate(hole)?;
        let y = p.component(at, 1)?;
        let up_from_foot = p.sub(y, foot)?;
        let rel = p.node(dapple_field::scoped::Node::Div(up_from_foot, damp_in))?;
        let one = p.scalar(1.0);
        let wet = p.sub(one, rel)?;
        let wet = p.saturate(wet)?;
        let s = p.add(face, hole)?;
        let s = p.add_scaled(s, 0.8, wet)?;
        let c = p.smoothstep(-0.2, 0.5, clump)?;
        let s = p.mul(s, c)?;
        p.output("score", PortType::Scalar, Scope::Pass, s)?;
        let score = evaluate_mask(
            &p.finish(),
            &[
                MapBinding::Pass(scalar_map(grid, &facing)?),
                MapBinding::Pass(scalar_map(grid, &hollow)?),
                MapBinding::Map(scalar_map(grid, &clumps)?),
                MapBinding::Position,
                MapBinding::Constant(Value::Scalar(grid.origin.y)),
                MapBinding::Constant(Value::Scalar(if damp > 0.0 { damp } else { 1e9 })),
            ],
            base,
        )?;
        let (score, score_tiling) = score;
        let covered = coverage(&score, args.scalar("coverage"), 0.04)?;
        let lumps = realize_scalar(grid, |b, d| {
            fbm(b, d, [300.0, 300.0], args.seed("lumps"), 3)
        })?;
        let relief = grid.raster(
            lumps
                .values()
                .iter()
                .map(|l| args.scalar("lumps") * (0.5 + 0.5 * l).clamp(0.0, 1.0))
                .collect(),
        )?;
        let tone = realize_scalar(grid, |b, d| fbm(b, d, [40.0, 40.0], args.seed("tone"), 3))?;
        let colors: Vec<Vec3> = tone
            .values()
            .iter()
            .map(|t| args.color("color") * (1.0 + 0.4 * t))
            .collect();
        let mut moss = Material::new(grid);
        moss.set_param(Param::BaseColor, super::color_map(grid, &colors)?)?;
        moss.set_param(
            Param::SpecularRoughness,
            Channel::Constant(Value::Scalar(0.95)),
        )?;
        moss.set_param(Param::FuzzWeight, Channel::Constant(Value::Scalar(1.0)))?;
        moss.set_param(
            Param::FuzzColor,
            Channel::Constant(Value::Vector3(args.color("sheen"))),
        )?;
        moss.set_param(Param::FuzzRoughness, Channel::Constant(Value::Scalar(0.7)))?;
        moss.set_aux(Aux::Surface, Channel::Constant(Value::Id(MOSS)))?;
        let mut m = cx.record(ops::deposit(
            base,
            &Deposit {
                material: moss,
                coverage: covered,
                thickness: args.scalar("thickness"),
                relief: Some(relief),
                matting: 2.0,
            },
        )?);
        // Damp ties moss to the foot; evaluating its score across the wrap
        // found where it tiles.
        m.set_tiling(m.tiling().and(score_tiling));
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}

/// A material from an exemplar image the host supplies (a photograph of
/// stone, say): its color tiled over the grid by histogram-preserving
/// blending ([`dapple_raster::synthesis::ByExample`]), keeping the
/// exemplar's physical scale and its histogram without visible repeats; a
/// relief from its luminance, `relief` deep; a roughness; and a surface
/// identity.
#[derive(Copy, Clone, Debug, Default)]
pub struct ByExample;

impl Module for ByExample {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId::new("dapple_library.by_example", 1),
            doc: "a material tiled from an exemplar image".into(),
            params: vec![
                meters("cell", [0.005, 2.0], 0.08, "the blended patches' size"),
                meters("relief", [0.0, 0.01], 0.0004, "relief from luminance"),
                fraction("roughness", 0.75, "specular roughness"),
                integer("surface", [0, 1 << 16], STONE, "surface identity"),
                seed(),
            ],
            inputs: vec![InputDecl {
                name: "exemplar".into(),
                kind: InputKind::Resource(ResourceRequest {
                    port: PortType::Color(Primaries::Rec709),
                    periodic: false,
                }),
                required: true,
                doc: "the exemplar, linear, in meters".into(),
            }],
            outputs: material_output(),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let resolved = args.resource("exemplar").ok_or_else(|| fail("exemplar"))?;
        let level = resolved
            .image
            .levels()
            .first()
            .ok_or_else(|| fail("an exemplar without levels"))?;
        let texels: Vec<[f32; 3]> = level
            .values()
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let exemplar = Raster::from_values(
            level.width(),
            level.height(),
            Vec2::ZERO,
            level.texel(),
            dapple_field::Edge::Clamp,
            texels,
        )?;
        let shape = grid.raster(vec![(); grid.len()])?;
        let out = Synthesis {
            cell: args.scalar("cell"),
            seed: args.seed("offsets"),
        }
        .apply(&exemplar, &shape)?;
        let relief = args.scalar("relief");
        let (mut colors, mut heights) = (
            Vec::with_capacity(grid.len()),
            Vec::with_capacity(grid.len()),
        );
        for c in out.values() {
            let c = Vec3::from_array(*c);
            colors.push(c);
            heights.push(relief * c.dot(Vec3::new(0.2126, 0.7152, 0.0722)));
        }
        let mut m = Material::new(grid);
        m.set_param(Param::BaseColor, super::color_map(grid, &colors)?)?;
        m.set_param(
            Param::SpecularRoughness,
            Channel::Constant(Value::Scalar(args.scalar("roughness"))),
        )?;
        m.set_aux(Aux::Height, super::scalar_channel(grid, &heights)?)?;
        m.set_aux(
            Aux::Surface,
            Channel::Constant(Value::Id(args.integer("surface"))),
        )?;
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}
