// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Paint that chips: a primer and a topcoat over a base.

use alloc::vec;
use alloc::vec::Vec;

use dapple_field::program::Op;
use dapple_field::{FractalKind, FractalParams, Value};
use dapple_material::module::{
    Args, Bind, Context, Input, InputDecl, InputKind, Interface, Module, ModuleError, ModuleId,
    Output, OutputDecl, OutputKind, Outputs,
};
use dapple_material::ops::{self, Deposit};
use dapple_material::{Aux, Channel, Material, Param};
use glam::Vec3;

use super::calibration::luminance;
use super::wear::convexity;
use super::{
    Finish, color, color_map, coverage, fail, fbm, fit, fraction, mask_map, meters, realize_scalar,
    scalar_channel, seed, smoothstep,
};
use crate::glazed_brick::PAINT;

/// Paint over a base, chipped: a `primer` and a topcoat of `color`, as
/// glossy as `gloss` (OpenPBR's coat, from [`Finish`]), `thickness` thick
/// together, the topcoat chalking paler in patches and rain-washed runs as
/// far as `fade` says.
///
/// `chipping` of the surface has lost its topcoat: on convex edges first
/// (rivet heads, laps, arrises), as much as `edges` says, otherwise in
/// ragged flakes, and along `scratches`. A chip shows the primer at its
/// rim and, over its inner 60%, the base: bare metal on steel, which
/// [`super::Rust`] then rusts. Chip edges are steps the paint's thickness
/// deep.
///
/// Paint is a dielectric: where it lies, metalness is 0.
#[derive(Copy, Clone, Debug, Default)]
pub struct Paint;

impl Module for Paint {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId::new("dapple_library.paint", 1),
            doc: "a primer and a topcoat, chipped at edges, in flakes and scratches".into(),
            params: vec![
                color(
                    "color",
                    Vec3::new(0.035, 0.085, 0.11),
                    "the topcoat's color: a deep blue-green",
                ),
                color(
                    "primer",
                    Vec3::new(0.16, 0.045, 0.026),
                    "the primer's color: red oxide",
                ),
                fraction(
                    "gloss",
                    0.5,
                    "how glossy the topcoat is: 0 flat, 1 high gloss",
                ),
                meters(
                    "thickness",
                    [0.00002, 0.002],
                    0.00025,
                    "primer and topcoat together",
                ),
                fraction(
                    "chipping",
                    0.03,
                    "share of the surface that lost its topcoat",
                ),
                fraction("edges", 0.6, "how much chips favor convex edges"),
                fraction("scratches", 0.3, "how much chips follow scratches"),
                fraction("fade", 0.3, "how far the topcoat has chalked"),
                seed(),
            ],
            inputs: vec![InputDecl {
                name: "base".into(),
                kind: InputKind::Material,
                required: true,
                doc: "what is painted".into(),
            }],
            outputs: vec![OutputDecl {
                name: "material".into(),
                kind: OutputKind::Material,
                doc: "the painted material".into(),
            }],
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let base = args.material("base").ok_or_else(|| fail("base"))?;
        let n = grid.len();
        let convex = convexity(base, 0.004)?;
        let flakes = realize_scalar(grid, |b, d| fbm(b, d, [16.0, 16.0], args.seed("flakes"), 7))?;
        let knocks = realize_scalar(grid, |b, d| fbm(b, d, [70.0, 70.0], args.seed("knocks"), 4))?;
        // Scratches: the crests of ridged noise stretched along each axis.
        let ridged =
            |b: &mut dapple_field::program::ProgramBuilder, d, frequency: [f32; 2], seed: u64| {
                b.add(Op::Fractal {
                    basis: dapple_field::Basis::Gradient,
                    domain: d,
                    frequency: fit(d, frequency),
                    seed,
                    params: FractalParams {
                        kind: FractalKind::Ridged,
                        octaves: 3,
                        ..FractalParams::default()
                    },
                })
            };
        let along = realize_scalar(grid, |b, d| ridged(b, d, [2.0, 24.0], args.seed("along")))?;
        let across = realize_scalar(grid, |b, d| ridged(b, d, [24.0, 2.0], args.seed("across")))?;
        let sparse = realize_scalar(grid, |b, d| fbm(b, d, [5.0, 5.0], args.seed("sparse"), 3))?;

        let (edges, scratches) = (args.scalar("edges"), args.scalar("scratches"));
        let mut score = Vec::with_capacity(n);
        for i in 0..n {
            // Convexity in units of 0.4 mm: a rivet head or a lap's edge.
            let edge = (convex.values()[i] * 2500.0).clamp(0.0, 1.0);
            let edge = edge * (0.6 + 0.4 * smoothstep(-0.5, 0.5, knocks.values()[i]));
            let flake = 0.5 + 0.5 * flakes.values()[i];
            let line = |r: f32| libm::powf(r.clamp(0.0, 1.0), 10.0);
            let scratch = line(along.values()[i]).max(0.7 * line(across.values()[i]))
                * smoothstep(0.0, 0.5, sparse.values()[i]);
            score.push(edges * edge + (1.0 - 0.5 * edges) * flake + scratches * scratch);
        }
        let score = grid.raster(score)?;
        let chipping = args.scalar("chipping");
        let (topcoat_off, primer_off) = (
            coverage(&score, chipping, 0.003)?,
            coverage(&score, 0.6 * chipping, 0.003)?,
        );

        // The primer: a matte deposit everywhere but the chips' middles.
        let tone = realize_scalar(grid, |b, d| fbm(b, d, [30.0, 30.0], args.seed("tone"), 3))?;
        let primer = args.color("primer");
        let mut layer = Material::new(grid);
        let primer_colors: Vec<Vec3> = tone
            .values()
            .iter()
            .map(|t| primer * (1.0 + 0.12 * t))
            .collect();
        layer.set_param(Param::BaseColor, color_map(grid, &primer_colors)?)?;
        layer.set_param(
            Param::SpecularRoughness,
            Channel::Constant(Value::Scalar(0.7)),
        )?;
        layer.set_aux(Aux::Surface, Channel::Constant(Value::Id(PAINT)))?;
        let thickness = args.scalar("thickness");
        let primed = cx.record(ops::deposit(
            base,
            &Deposit {
                material: layer,
                coverage: grid.raster(primer_off.values().iter().map(|c| 1.0 - c).collect())?,
                thickness: 0.35 * thickness,
                relief: None,
                matting: 1.0,
            },
        )?);

        // The topcoat, chalked paler and duller in patches and in runs
        // down the face.
        let patches = realize_scalar(grid, |b, d| fbm(b, d, [5.0, 5.0], args.seed("fade"), 4))?;
        let runs = realize_scalar(grid, |b, d| fbm(b, d, [14.0, 2.0], args.seed("runs"), 4))?;
        let (paint, fade) = (args.color("color"), args.scalar("fade"));
        let chalk = paint.lerp(Vec3::splat(luminance(paint)), 0.55) * 1.6 + Vec3::splat(0.015);
        let colors: Vec<Vec3> = (0..n)
            .map(|i| {
                let f = smoothstep(-0.3, 0.6, patches.values()[i])
                    .max(0.8 * smoothstep(0.1, 0.6, runs.values()[i]));
                let c = paint.lerp(chalk, fade * f);
                (c * (1.0 + 0.05 * tone.values()[i])).min(Vec3::ONE)
            })
            .collect();
        let color_map = match color_map(grid, &colors)? {
            Channel::Map(m) => m,
            Channel::Constant(_) => unreachable!("a map"),
        };
        let kept = grid.raster(topcoat_off.values().iter().map(|c| 1.0 - c).collect())?;
        let gloss = args.scalar("gloss");
        let mut out = cx.instantiate(
            &Finish,
            "topcoat",
            Bind::new()
                .material("base", primed)
                .input("coverage", Input::Map(mask_map(grid, &kept)?))
                .input("color", Input::Map(color_map))
                .scalar("opacity", 1.0)
                .scalar("full_thickness", 0.3 * thickness)
                .scalar("thickness", 0.65 * thickness)
                .scalar("tint", 0.1)
                .scalar("roughness", 0.5 + (0.05 - 0.5) * gloss)
                .scalar("roughness_variation", 0.04 + 0.08 * fade)
                .scalar("peel", 0.000_002_5 * gloss)
                .scalar("peel_size", 0.004)
                .integer("surface", PAINT),
        )?;
        let m = out
            .take_material("material")
            .ok_or_else(|| fail("the topcoat returned no material"))?;
        // Chalking dulls the pigment under the coat as well.
        let mut m = m;
        let rough: Vec<f32> = (0..n)
            .map(|i| {
                let r = m
                    .value(
                        dapple_material::ChannelId::Param(Param::SpecularRoughness),
                        i,
                    )
                    .component(0)
                    .unwrap_or(0.5);
                (r + 0.3 * fade * smoothstep(-0.3, 0.6, patches.values()[i]) * kept.values()[i])
                    .min(1.0)
            })
            .collect();
        m.set_param(Param::SpecularRoughness, scalar_channel(grid, &rough)?)?;
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}
