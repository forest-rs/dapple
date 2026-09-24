// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Bedded, tooled stone.

use alloc::vec;

use dapple_field::program::{Op, ProgramBuilder};
use dapple_field::scoped::{Node, Scope, ScopedBuilder};
use dapple_field::shaping::ColorRamp;
use dapple_field::{CellOutput, PortType, Primaries, Value};
use dapple_material::module::{
    Args, Context, Interface, Module, ModuleError, ModuleId, Output, OutputDecl, OutputKind,
    Outputs,
};
use dapple_material::program::{MapBinding, evaluate};
use dapple_material::{Aux, Channel, Material, Param};
use glam::Vec3;

use super::{color, fail, fbm, fit, fraction, integer, meters, realize, seed};
use crate::glazed_brick::STONE;

/// Sedimentary stone (sandstone, limestone) as a dressed face: beds of
/// lighter and darker stone running across it, wandering, a grain of its
/// own, iron staining in patches, and drag-tooled marks.
///
/// Color comes from a ramp through `dark`, the midpoint of the two, and
/// `light`, read at the bedding; the tooling is fine parallel furrows
/// `tooling` deep, across the beds.
#[derive(Copy, Clone, Debug, Default)]
pub struct Stone;

impl Module for Stone {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId {
                name: "dapple_library.stone",
                version: 1,
            },
            doc: "bedded, tooled stone",
            params: vec![
                color("light", Vec3::new(0.50, 0.41, 0.27), "the lightest beds"),
                color("dark", Vec3::new(0.37, 0.29, 0.19), "the darkest beds"),
                meters("bedding", [0.002, 0.2], 0.018, "bed spacing"),
                meters("grain", [0.0002, 0.005], 0.0007, "grain size"),
                meters("tooling", [0.0, 0.003], 0.0005, "tooling furrow depth"),
                fraction("stain", 0.3, "iron staining"),
                fraction("roughness", 0.78, "specular roughness"),
                integer("surface", [0, 1 << 16], STONE, "surface identity"),
                seed(),
            ],
            inputs: vec![],
            outputs: vec![OutputDecl {
                name: "material",
                kind: OutputKind::Material,
                doc: "the stone",
            }],
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let bed = 1.0 / args.scalar("bedding");
        // Beds: noise stretched across, warped so they wander.
        let beds = realize(grid, |b: &mut ProgramBuilder, d| {
            let layers = fbm(b, d, [1.5, bed], args.seed("beds"), 5)?;
            let dx = fbm(b, d, [3.0, 3.0], args.seed("warp_x"), 3)?;
            let dy = fbm(b, d, [3.0, 3.0], args.seed("warp_y"), 3)?;
            b.add(Op::Warp {
                input: layers,
                dx,
                dy,
                amount: 0.03,
            })
        })?;
        let cell = 1.0 / args.scalar("grain");
        let grain = realize(grid, |b, d| {
            b.add(Op::Cellular {
                domain: d,
                frequency: fit(d, [cell, cell]),
                jitter: 0.9,
                seed: args.seed("grain"),
                output: CellOutput::CellValue,
            })
        })?;
        let stain = realize(grid, |b, d| fbm(b, d, [5.0, 5.0], args.seed("stain"), 4))?;
        // Drag marks: furrows running down the face, a few millimeters
        // apart, varying in depth along their length.
        let tooling = realize(grid, |b, d| {
            fbm(b, d, [180.0, 6.0], args.seed("tooling"), 2)
        })?;
        let broad = realize(grid, |b, d| fbm(b, d, [4.0, 4.0], args.seed("broad"), 3))?;

        let (light, dark) = (args.color("light"), args.color("dark"));
        let ramp = ColorRamp::new(&[
            (-0.6, dark),
            (-0.1, (light + dark) * 0.5),
            (0.2, light),
            (0.6, (light * 0.8 + dark * 0.2)),
        ])
        .map_err(|_| fail("ramp"))?;
        let mut p = ScopedBuilder::new("dapple_library.stone.face");
        let sample =
            |p: &mut ScopedBuilder, name: &str| p.input(name, PortType::Scalar, Scope::Sample);
        let beds_in = sample(&mut p, "beds")?;
        let grain_in = sample(&mut p, "grain")?;
        let stain_in = sample(&mut p, "stain")?;
        let tool_in = sample(&mut p, "tooling")?;
        let broad_in = sample(&mut p, "broad")?;
        let stain_k = p.input("stain_amount", PortType::Scalar, Scope::Material)?;
        let depth = p.input("tooling_depth", PortType::Scalar, Scope::Material)?;
        let rough = p.input("roughness", PortType::Scalar, Scope::Material)?;
        let c = p.node(Node::Ramp(beds_in, ramp))?;
        let g = p.lerp(0.82, 1.18, grain_in)?;
        let c = p.mul(c, g)?;
        let rust = p.vector3(1.1, 0.88, 0.7);
        let s = p.smoothstep(0.15, 0.7, stain_in)?;
        let s = p.mul(s, stain_k)?;
        let one3 = p.vector3(1.0, 1.0, 1.0);
        let tinted = p.mix(one3, rust, s)?;
        let c = p.mul(c, tinted)?;
        let furrow = p.node(Node::Unary(dapple_field::scoped::UnaryOp::Abs, tool_in))?;
        let h = p.mul(furrow, depth)?;
        let h = p.scale(-1.0, h)?;
        let h = p.add_scaled(h, 0.0008, broad_in)?;
        let h = p.add_scaled(h, 0.00015, grain_in)?;
        let r = p.add_scaled(rough, 0.05, grain_in)?;
        let r = p.saturate(r)?;
        p.output(
            "color",
            PortType::Color(Primaries::Rec709),
            Scope::Sample,
            c,
        )?;
        p.output("height", PortType::Scalar, Scope::Sample, h)?;
        p.output("roughness", PortType::Scalar, Scope::Sample, r)?;
        let program = p.finish();
        let out = evaluate(
            &program,
            &[
                MapBinding::Map(beds),
                MapBinding::Map(grain),
                MapBinding::Map(stain),
                MapBinding::Map(tooling),
                MapBinding::Map(broad),
                MapBinding::Constant(Value::Scalar(args.scalar("stain"))),
                MapBinding::Constant(Value::Scalar(args.scalar("tooling"))),
                MapBinding::Constant(Value::Scalar(args.scalar("roughness"))),
            ],
            &Material::new(grid),
        )?;
        let [color_map, height, roughness]: [_; 3] =
            out.try_into().map_err(|_| fail("three outputs"))?;
        let mut m = Material::new(grid);
        m.set_param(Param::BaseColor, Channel::Map(color_map))?;
        m.set_param(Param::SpecularRoughness, Channel::Map(roughness))?;
        m.set_aux(Aux::Height, Channel::Map(height))?;
        m.set_aux(
            Aux::Surface,
            Channel::Constant(Value::Id(args.integer("surface"))),
        )?;
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}
