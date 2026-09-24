// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Bedded, tooled stone.

use alloc::vec;

use dapple_field::program::{Op, ProgramBuilder};
use dapple_field::scoped::{Node, Scope, ScopedBuilder, UnaryOp};
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

/// Sedimentary stone (sandstone, limestone) as a dressed face: an
/// isotropic, mottled body with a grain of its own, faint bedding only as
/// strong as `bedding_strength` says (0 for a massive stone), iron staining
/// in patches, and batted tooling.
///
/// Color comes from a ramp through `dark`, the midpoint of the two, and
/// `light`, read at the mottling. The tooling is a batting chisel's work:
/// bands `tooling_band` wide, each of fine parallel cuts `tooling_pitch`
/// apart at the band's own slant, `tooling` deep, fading in and out.
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
                meters("bedding", [0.002, 0.2], 0.03, "bed spacing"),
                fraction(
                    "bedding_strength",
                    0.12,
                    "how much the bedding shows; 0 for a massive stone",
                ),
                meters("grain", [0.0002, 0.005], 0.0007, "grain size"),
                meters("tooling", [0.0, 0.003], 0.0004, "tooling cut depth"),
                meters("tooling_band", [0.005, 0.2], 0.045, "the chisel's width"),
                meters("tooling_pitch", [0.0005, 0.02], 0.003, "the cuts' spacing"),
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
        // The stone's body: isotropic mottling at a few scales.
        let mottle = realize(grid, |b: &mut ProgramBuilder, d| {
            fbm(b, d, [5.0, 5.0], args.seed("mottle"), 6)
        })?;
        // Bedding, where the stone shows it: faint wandering layers.
        let bed = 1.0 / args.scalar("bedding");
        let beds = realize(grid, |b: &mut ProgramBuilder, d| {
            let layers = fbm(b, d, [2.0, bed], args.seed("beds"), 4)?;
            let dx = fbm(b, d, [3.0, 3.0], args.seed("warp_x"), 3)?;
            let dy = fbm(b, d, [3.0, 3.0], args.seed("warp_y"), 3)?;
            b.add(Op::Warp {
                input: layers,
                dx,
                dy,
                amount: 0.04,
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
        let stain = realize(grid, |b, d| fbm(b, d, [7.0, 7.0], args.seed("stain"), 4))?;
        let wander = realize(grid, |b, d| fbm(b, d, [30.0, 30.0], args.seed("wander"), 2))?;

        let (light, dark) = (args.color("light"), args.color("dark"));
        let ramp = ColorRamp::new(&[
            (-0.7, dark),
            (-0.15, (light + dark) * 0.5),
            (0.25, light),
            (0.7, (light * 0.85 + dark * 0.15)),
        ])
        .map_err(|_| fail("ramp"))?;
        let mut p = ScopedBuilder::new("dapple_library.stone.face");
        let sample =
            |p: &mut ScopedBuilder, name: &str| p.input(name, PortType::Scalar, Scope::Sample);
        let mottle_in = sample(&mut p, "mottle")?;
        let beds_in = sample(&mut p, "beds")?;
        let grain_in = sample(&mut p, "grain")?;
        let stain_in = sample(&mut p, "stain")?;
        let wander_in = sample(&mut p, "wander")?;
        let at = p.input("position", PortType::Vector2, Scope::Sample)?;
        let bedding = p.input("bedding_strength", PortType::Scalar, Scope::Material)?;
        let stain_k = p.input("stain_amount", PortType::Scalar, Scope::Material)?;
        let depth = p.input("tooling_depth", PortType::Scalar, Scope::Material)?;
        let band_w = p.input("tooling_band", PortType::Scalar, Scope::Material)?;
        let pitch = p.input("tooling_pitch", PortType::Scalar, Scope::Material)?;
        let rough = p.input("roughness", PortType::Scalar, Scope::Material)?;
        let bands = p.resource("band_random", {
            let mut b = ProgramBuilder::new();
            let n = b
                .add(Op::Noise {
                    basis: dapple_field::Basis::Value,
                    domain: dapple_field::Domain::Plane,
                    frequency: [1.0, 1.0],
                    seed: args.seed("bands"),
                })
                .map_err(super::program_error)?;
            b.finish_value(n).map_err(super::program_error)?
        });
        // Color: the mottled body, a little bedding, grains, stains.
        let beds_k = p.mul(beds_in, bedding)?;
        let t = p.add(mottle_in, beds_k)?;
        let c = p.node(Node::Ramp(t, ramp))?;
        let g = p.lerp(0.85, 1.15, grain_in)?;
        let c = p.mul(c, g)?;
        let rust = p.vector3(1.1, 0.9, 0.72);
        let s = p.smoothstep(0.25, 0.75, stain_in)?;
        let s = p.mul(s, stain_k)?;
        let one3 = p.vector3(1.0, 1.0, 1.0);
        let tinted = p.mix(one3, rust, s)?;
        let c = p.mul(c, tinted)?;
        // Tooling: the face dressed with a batting chisel in bands a
        // chisel wide, each band of fine parallel cuts at its own slant.
        let x = p.component(at, 0)?;
        let y = p.component(at, 1)?;
        let bx = p.node(Node::Div(x, band_w))?;
        let band = p.node(Node::Unary(UnaryOp::Floor, bx))?;
        let zero = p.scalar(0.0);
        let key = p.node(Node::Vector2(band, zero))?;
        let r = p.sample(bands, key)?;
        let slant = p.scale(0.25, r)?;
        let sx = p.mul(x, slant)?;
        let u = p.add(y, sx)?;
        let u = p.node(Node::Div(u, pitch))?;
        let phase = p.scale(3.7, r)?;
        let u = p.add(u, phase)?;
        let f = p.node(Node::Unary(UnaryOp::Fract, u))?;
        // A chisel cut: a steep side and a long shallow one.
        let rise = p.smoothstep(0.0, 0.15, f)?;
        let fall = p.smoothstep(1.0, 0.45, f)?;
        let cut = p.mul(rise, fall)?;
        let one = p.scalar(1.0);
        let groove = p.sub(one, cut)?;
        let w = p.smoothstep(-0.4, 0.4, wander_in)?;
        let w = p.lerp(0.4, 1.0, w)?;
        let groove = p.mul(groove, w)?;
        let h = p.mul(groove, depth)?;
        let h = p.scale(-1.0, h)?;
        let h = p.add_scaled(h, 0.0003, mottle_in)?;
        let h = p.add_scaled(h, 0.00012, grain_in)?;
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
                MapBinding::Map(mottle),
                MapBinding::Map(beds),
                MapBinding::Map(grain),
                MapBinding::Map(stain),
                MapBinding::Map(wander),
                MapBinding::Position,
                MapBinding::Constant(Value::Scalar(args.scalar("bedding_strength"))),
                MapBinding::Constant(Value::Scalar(args.scalar("stain"))),
                MapBinding::Constant(Value::Scalar(args.scalar("tooling"))),
                MapBinding::Constant(Value::Scalar(args.scalar("tooling_band"))),
                MapBinding::Constant(Value::Scalar(args.scalar("tooling_pitch"))),
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
