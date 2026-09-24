// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Glazes, varnishes and oils: pigment and an optical coat.

use alloc::vec;
use alloc::vec::Vec;

use dapple_field::{PortType, Primaries, Value};
use dapple_material::module::{
    Args, Context, InputDecl, InputKind, Interface, Module, ModuleError, ModuleId, Output,
    OutputDecl, OutputKind, Outputs,
};
use dapple_material::ops::{self, Coating, Detail, Layer, Transition};
use dapple_material::{Aux, Channel, Material, Param};
use dapple_raster::{GaussianBlur, RasterOp};
use glam::Vec3;

use super::{color, fbm, fraction, integer, meters, realize_scalar, scalar, scalar_channel, seed};

/// A finish over a base material: a glaze, a varnish or an oil.
///
/// It is two things OpenPBR keeps apart. Where `opacity` is above zero the
/// finish carries **pigment** that hides the base as it thickens (an
/// enamel glaze): a [selection](dapple_material::ops::select) of a pigment
/// material over the base, weighted by `opacity` and by how the thickness
/// compares to `full_thickness`; the pigment deepens where the finish pools
/// thicker than that. Over everything it covers it lays a clear or tinted
/// **coat** ([`ops::coat`]) of refractive index `ior`, tinted by `tint`
/// toward its color, whose roughness varies slowly, which rises by its
/// thickness, and whose own surface ripples with orange peel
/// ([`ops::apply_detail`] on the coat layer only). Given `leveling`, it
/// fills the relief it lies on, as a thick glaze fills a body's grain. A
/// varnish is opacity 0
/// and a strong tint; an enamel glaze is opaque with a faint tint.
///
/// Inputs: `base` (required), and optionally `coverage` (where the finish
/// lies), `thickness` (meters) and `color`, per texel.
#[derive(Copy, Clone, Debug, Default)]
pub struct Finish;

impl Module for Finish {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId::new("dapple_library.finish", 1),
            doc: "a glaze, varnish or oil: pigment and a coat".into(),
            params: vec![
                color(
                    "color",
                    Vec3::new(0.85, 0.62, 0.32),
                    "pigment and tint color",
                ),
                fraction("opacity", 0.0, "how fully the pigment hides the base"),
                meters(
                    "full_thickness",
                    [0.00001, 0.005],
                    0.0003,
                    "thickness at which the pigment is fully opaque",
                ),
                fraction(
                    "tint",
                    0.4,
                    "how strongly the coat is tinted toward the color",
                ),
                fraction("roughness", 0.08, "coat roughness"),
                fraction(
                    "roughness_variation",
                    0.04,
                    "slow variation of coat roughness",
                ),
                scalar(
                    "ior",
                    dapple_material::module::Unit::None,
                    [1.2, 2.0],
                    1.5,
                    "coat index of refraction",
                ),
                meters(
                    "thickness",
                    [0.0, 0.005],
                    0.0001,
                    "thickness where no map is given",
                ),
                meters(
                    "peel",
                    [0.0, 0.0005],
                    0.000008,
                    "orange-peel amplitude on the coat",
                ),
                meters("peel_size", [0.0005, 0.02], 0.003, "orange-peel wavelength"),
                meters(
                    "leveling",
                    [0.0, 0.01],
                    0.0,
                    "how far the finish levels the relief under it (a blur radius)",
                ),
                integer(
                    "surface",
                    [0, 1 << 16],
                    0,
                    "surface identity where opaque; 0 keeps the base's",
                ),
                seed(),
            ],
            inputs: vec![
                InputDecl {
                    name: "base".into(),
                    kind: InputKind::Material,
                    required: true,
                    doc: "what the finish is on".into(),
                },
                InputDecl {
                    name: "coverage".into(),
                    kind: InputKind::Map(PortType::Mask),
                    required: false,
                    doc: "where the finish lies".into(),
                },
                InputDecl {
                    name: "thickness".into(),
                    kind: InputKind::Map(PortType::Scalar),
                    required: false,
                    doc: "its thickness in meters".into(),
                },
                InputDecl {
                    name: "color".into(),
                    kind: InputKind::Map(PortType::Color(Primaries::Rec709)),
                    required: false,
                    doc: "its color per texel".into(),
                },
            ],
            outputs: vec![OutputDecl {
                name: "material".into(),
                kind: OutputKind::Material,
                doc: "the finished material".into(),
            }],
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let base = args.material("base").ok_or_else(|| super::fail("base"))?;
        let n = grid.len();
        let coverage = |i: usize| {
            args.map("coverage")
                .map_or(1.0, |m| m.value(i).component(0).unwrap_or(0.0))
        };
        let thick = args.scalar("thickness");
        let thickness = |i: usize| {
            args.map("thickness")
                .map_or(thick, |m| m.value(i).component(0).unwrap_or(0.0))
        };
        let param_color = args.color("color");
        let color_at = |i: usize| match args.map("color").map(|m| m.value(i)) {
            Some(Value::Vector3(c)) => c,
            _ => param_color,
        };
        let full = args.scalar("full_thickness");
        let opacity = args.scalar("opacity");

        let mut m = base.clone();
        if opacity > 0.0 {
            let mut weight = Vec::with_capacity(n);
            let mut pigment = Vec::with_capacity(n);
            for i in 0..n {
                let t = thickness(i);
                weight.push(opacity * super::smoothstep(0.0, full, t) * coverage(i));
                // Pools thicker than opaque deepen the color, up to 30%.
                let pool = ((t / full) - 1.0).clamp(0.0, 1.0);
                pigment.push(color_at(i) * (1.0 - 0.3 * pool));
            }
            let mut layer = Material::new(grid);
            layer.set_param(Param::BaseColor, super::color_map(grid, &pigment)?)?;
            layer.set_param(
                Param::SpecularRoughness,
                Channel::Constant(Value::Scalar(0.35)),
            )?;
            let surface = args.integer("surface");
            match (surface, base.aux(Aux::Surface)) {
                (0, Some(s)) => layer.set_aux(Aux::Surface, s.clone())?,
                (0, None) => {}
                (id, _) => layer.set_aux(Aux::Surface, Channel::Constant(Value::Id(id)))?,
            }
            if let Some(h) = base.aux(Aux::Height) {
                layer.set_aux(Aux::Height, h.clone())?;
            }
            if let Some(r) = base.aux(Aux::Region) {
                layer.set_aux(Aux::Region, r.clone())?;
            }
            let w = grid.raster(weight)?;
            m = cx.record(ops::select(&m, &layer, &w, Transition::Mask)?);
        }

        let variation = realize_scalar(grid, |b, d| fbm(b, d, [6.0, 6.0], args.seed("gloss"), 3))?;
        let rough = args.scalar("roughness");
        let spread = args.scalar("roughness_variation");
        let tint = args.scalar("tint");
        let (mut weights, mut tints, mut roughs, mut thicknesses) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for i in 0..n {
            weights.push(coverage(i));
            tints.push(Vec3::ONE + (color_at(i) - Vec3::ONE) * tint);
            roughs.push((rough + spread * variation.values()[i]).clamp(0.0, 1.0));
            thicknesses.push(thickness(i));
        }
        let coating = Coating {
            weight: Channel::Map(
                grid.typed(PortType::Mask, weights.iter().map(|&w| Value::Scalar(w)))?,
            ),
            color: super::color_map(grid, &tints)?,
            roughness: scalar_channel(grid, &roughs)?,
            ior: Channel::Constant(Value::Scalar(args.scalar("ior"))),
            darkening: Channel::Constant(Value::Scalar(1.0)),
            thickness: Some(grid.raster(thicknesses)?),
        };
        // A thick finish fills the relief under it: where it lies, the
        // surface is the relief smoothed over `leveling`.
        let leveling = args.scalar("leveling");
        if leveling > 0.0 {
            let h = m.height()?;
            let smooth = GaussianBlur { sigma: leveling }.apply(&h)?;
            let delta: Vec<f32> = (0..n)
                .map(|i| (smooth.values()[i] - h.values()[i]) * weights[i])
                .collect();
            let detail = Detail {
                height: Some(grid.raster(delta)?),
                ..Detail::identity(Layer::Base)
            };
            m = cx.record(ops::apply_detail(&m, &detail)?);
        }
        m = cx.record(ops::coat(&m, &coating)?);

        let peel = args.scalar("peel");
        if peel > 0.0 {
            let f = 1.0 / args.scalar("peel_size");
            let ripple = realize_scalar(grid, |b, d| fbm(b, d, [f, f], args.seed("peel"), 2))?;
            let ripple = grid.raster(ripple.values().iter().map(|v| v * peel).collect())?;
            let detail = Detail {
                height: Some(ripple),
                strength: Some(grid.raster(weights)?),
                ..Detail::identity(Layer::Coat)
            };
            m = cx.record(ops::apply_detail(&m, &detail)?);
        }
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}
