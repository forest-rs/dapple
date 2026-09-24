// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sanded lime mortar.

use alloc::vec;

use dapple_field::program::Op;
use dapple_field::scoped::{Node, Scope, ScopedBuilder};
use dapple_field::{CellOutput, PortType, Primaries, ScatterOutput, Stamp, Value};
use dapple_material::module::{
    Args, Context, InputDecl, InputKind, Interface, Module, ModuleError, ModuleId, Output,
    OutputDecl, OutputKind, Outputs,
};
use dapple_material::program::{MapBinding, evaluate};
use dapple_material::{Aux, Channel, Material, Param};
use glam::Vec3;

use super::{color, fbm, fit, fraction, meters, realize, seed};
use crate::glazed_brick::MORTAR;

/// Sanded lime mortar: sand grains a millimeter or two across, each its own
/// tone, coarser aggregate standing proud, sooty patches, pits, and a
/// rough, matte surface.
///
/// Given `joint`, the distance from each texel to the nearest unit (brick,
/// stone) in meters, it is tooled into a concave joint recessed `recess`
/// below the datum in the joint's middle and rising toward the units'
/// edges; without it, it is flat at `-recess`.
#[derive(Copy, Clone, Debug, Default)]
pub struct Mortar;

impl Module for Mortar {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId {
                name: "dapple_library.mortar",
                version: 1,
            },
            doc: "sanded lime mortar",
            params: vec![
                color(
                    "color",
                    Vec3::new(0.36, 0.33, 0.27),
                    "mean color: an aged buff-grey lime",
                ),
                meters("grain", [0.0005, 0.005], 0.0016, "sand grain size"),
                fraction("soot", 0.15, "how strong the sooty patches are"),
                meters(
                    "recess",
                    [0.0, 0.02],
                    0.0025,
                    "depth of the joint's middle below the datum",
                ),
                meters("joint_width", [0.002, 0.05], 0.01, "the joint's width"),
                fraction("roughness", 0.95, "specular roughness"),
                seed(),
            ],
            inputs: vec![InputDecl {
                name: "joint",
                kind: InputKind::Map(PortType::Scalar),
                required: false,
                doc: "distance to the nearest unit, in meters, for tooling",
            }],
            outputs: vec![OutputDecl {
                name: "material",
                kind: OutputKind::Material,
                doc: "the mortar, surface identity MORTAR",
            }],
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let cell = 1.0 / args.scalar("grain");
        let grains_seed = args.seed("grains");
        let grains = realize(grid, |b, d| {
            b.add(Op::Cellular {
                domain: d,
                frequency: fit(d, [cell, cell]),
                jitter: 0.9,
                seed: grains_seed,
                output: CellOutput::F1,
            })
        })?;
        let tone = realize(grid, |b, d| {
            b.add(Op::Cellular {
                domain: d,
                frequency: fit(d, [cell, cell]),
                jitter: 0.9,
                seed: grains_seed,
                output: CellOutput::CellValue,
            })
        })?;
        let coarse = realize(grid, |b, d| {
            b.add(Op::Scatter {
                domain: d,
                placement: dapple_field::Placement {
                    frequency: fit(d, [200.0, 200.0])[0],
                    density: 0.25,
                    radius: [0.35, 0.8],
                    rotate: false,
                },
                stamp: Stamp::Dome,
                seed: args.seed("coarse"),
                output: ScatterOutput::Max,
            })
        })?;
        let patches = realize(grid, |b, d| fbm(b, d, [9.0, 9.0], args.seed("soot"), 5))?;
        let fine = realize(grid, |b, d| fbm(b, d, [260.0, 260.0], args.seed("fine"), 2))?;

        // The mortar at a point, as an inspectable program.
        let mut p = ScopedBuilder::new("dapple_library.mortar.surface");
        let sample =
            |p: &mut ScopedBuilder, name: &str| p.input(name, PortType::Scalar, Scope::Sample);
        let f1 = sample(&mut p, "grains")?;
        let tone_in = sample(&mut p, "tone")?;
        let coarse_in = sample(&mut p, "coarse")?;
        let patch = sample(&mut p, "patches")?;
        let fine_in = sample(&mut p, "fine")?;
        let joint = sample(&mut p, "joint")?;
        let base = p.input("color", PortType::Color(Primaries::Rec709), Scope::Material)?;
        let soot = p.input("soot", PortType::Scalar, Scope::Material)?;
        let recess = p.input("recess", PortType::Scalar, Scope::Material)?;
        let half = p.input("half_width", PortType::Scalar, Scope::Material)?;
        let rough = p.input("roughness", PortType::Scalar, Scope::Material)?;
        // Color: grain tones 0.72 to 1.28, darker sooty patches, lighter
        // coarse aggregate, a little fine variation.
        let grain_tone = p.lerp(0.72, 1.28, tone_in)?;
        let sooty = p.smoothstep(-0.1, 0.6, patch)?;
        let sooty = p.mul(sooty, soot)?;
        let darken = p.lerp(1.0, 0.55, sooty)?;
        let pebble = p.smoothstep(0.05, 0.3, coarse_in)?;
        let light = p.lerp(1.0, 1.3, pebble)?;
        let one = p.scalar(1.0);
        let finer = p.add_scaled(one, 0.06, fine_in)?;
        let k = p.mul(grain_tone, darken)?;
        let k = p.mul(k, light)?;
        let k = p.mul(k, finer)?;
        let color_out = p.mul(base, k)?;
        // Height: the tooled joint, grains up to 0.35 mm proud, aggregate up
        // to 0.6 mm, and pits where the fine noise dips.
        let across = p.node(Node::Div(joint, half))?;
        let s = p.sub(one, across)?;
        let s = p.saturate(s)?;
        let s2 = p.mul(s, s)?;
        let tooled = p.scale(0.003, s2)?;
        let neg = p.scale(-1.0, recess)?;
        let h = p.add(neg, tooled)?;
        let f1n = p.scale(1.0 / 0.6, f1)?;
        let f1n = p.saturate(f1n)?;
        let grain_h = p.sub(one, f1n)?;
        let h = p.add_scaled(h, 0.00035, grain_h)?;
        let h = p.add_scaled(h, 0.0006, coarse_in)?;
        let pit = p.smoothstep(-0.45, -0.7, fine_in)?;
        let h = p.add_scaled(h, -0.0004, pit)?;
        // Roughness: rough everywhere, a touch smoother on aggregate.
        let r = p.add_scaled(rough, -0.08, pebble)?;
        let r = p.add_scaled(r, 0.03, fine_in)?;
        let r = p.saturate(r)?;
        p.output(
            "color",
            PortType::Color(Primaries::Rec709),
            Scope::Sample,
            color_out,
        )?;
        p.output("height", PortType::Scalar, Scope::Sample, h)?;
        p.output("roughness", PortType::Scalar, Scope::Sample, r)?;
        let program = p.finish();

        let half_width = 0.5 * args.scalar("joint_width");
        let joint_binding = match args.map("joint") {
            Some(j) => MapBinding::Map(j.clone()),
            None => MapBinding::Constant(Value::Scalar(half_width)),
        };
        let m0 = Material::new(grid);
        let out = evaluate(
            &program,
            &[
                MapBinding::Map(grains),
                MapBinding::Map(tone),
                MapBinding::Map(coarse),
                MapBinding::Map(patches),
                MapBinding::Map(fine),
                joint_binding,
                MapBinding::Constant(Value::Vector3(args.color("color"))),
                MapBinding::Constant(Value::Scalar(args.scalar("soot"))),
                MapBinding::Constant(Value::Scalar(args.scalar("recess"))),
                MapBinding::Constant(Value::Scalar(half_width)),
                MapBinding::Constant(Value::Scalar(args.scalar("roughness"))),
            ],
            &m0,
        )?;
        let out_tiling = out.tiling;
        let [color_map, height, roughness]: [_; 3] = out
            .outputs
            .try_into()
            .map_err(|_| super::fail("three outputs"))?;
        let mut m = Material::new(grid);
        m.set_tiling(out_tiling);
        m.set_param(Param::BaseColor, Channel::Map(color_map))?;
        m.set_param(Param::SpecularRoughness, Channel::Map(roughness))?;
        m.set_aux(Aux::Height, Channel::Map(height))?;
        m.set_aux(Aux::Surface, Channel::Constant(Value::Id(MORTAR)))?;
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}
