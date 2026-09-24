// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A fired-clay body.

use alloc::vec;
use alloc::vec::Vec;

use dapple_field::program::Op;
use dapple_field::{CellOutput, Value};
use dapple_material::module::{
    Args, Context, Interface, Module, ModuleError, ModuleId, Output, OutputDecl, OutputKind,
    Outputs,
};
use dapple_material::{Aux, Channel, Material, Param};
use glam::Vec3;

use super::{color, fbm, fit, fraction, meters, realize_scalar, scalar_channel, seed};
use crate::glazed_brick::BODY;

/// Fired clay, as the body of a brick or tile shows where it is exposed:
/// sand-sized grains of their own tone, dark iron speckle, slow mottling
/// from the kiln, a matte surface and a fine grain relief.
#[derive(Copy, Clone, Debug, Default)]
pub struct CeramicBody;

impl Module for CeramicBody {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId::new("dapple_library.ceramic_body", 1),
            doc: "a fired-clay body".into(),
            params: vec![
                color(
                    "color",
                    Vec3::new(0.36, 0.27, 0.17),
                    "the body's mean color: buff fireclay by default",
                ),
                fraction("speckle", 0.3, "how much dark iron speckle"),
                meters("grain", [0.0002, 0.005], 0.0012, "grain size"),
                fraction("roughness", 0.86, "specular roughness"),
                seed(),
            ],
            inputs: vec![],
            outputs: vec![OutputDecl {
                name: "material".into(),
                kind: OutputKind::Material,
                doc: "the body, surface identity BODY".into(),
            }],
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let cell = 1.0 / args.scalar("grain");
        let cells = |output, seed| {
            move |b: &mut dapple_field::program::ProgramBuilder, d| {
                b.add(Op::Cellular {
                    domain: d,
                    frequency: fit(d, [cell, cell]),
                    jitter: 0.9,
                    seed,
                    output,
                })
            }
        };
        let f1 = realize_scalar(grid, cells(CellOutput::F1, args.seed("grains")))?;
        let tone = realize_scalar(grid, cells(CellOutput::CellValue, args.seed("grains")))?;
        let speck_cells = 0.35 * cell;
        let speck = realize_scalar(grid, |b, d| {
            b.add(Op::Cellular {
                domain: d,
                frequency: fit(d, [speck_cells, speck_cells]),
                jitter: 1.0,
                seed: args.seed("speckle"),
                output: CellOutput::CellValue,
            })
        })?;
        let speck_f1 = realize_scalar(grid, |b, d| {
            b.add(Op::Cellular {
                domain: d,
                frequency: fit(d, [speck_cells, speck_cells]),
                jitter: 1.0,
                seed: args.seed("speckle"),
                output: CellOutput::F1,
            })
        })?;
        let mottle = realize_scalar(grid, |b, d| fbm(b, d, [12.0, 12.0], args.seed("mottle"), 4))?;
        let base = args.color("color");
        let amount = args.scalar("speckle");
        let rough = args.scalar("roughness");
        let n = grid.len();
        let (mut colors, mut heights, mut roughs) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        let iron = Vec3::new(0.09, 0.05, 0.03);
        for i in 0..n {
            let g = tone.values()[i];
            let m = mottle.values()[i];
            // Iron spots: the rarest speckle cells, small, dark.
            let is_spot =
                super::smoothstep(1.0 - 0.08 * amount, 1.0 - 0.06 * amount, speck.values()[i])
                    * super::smoothstep(0.35, 0.2, speck_f1.values()[i]);
            let c = base * (0.9 + 0.2 * g) * (1.0 + 0.12 * m);
            colors.push(c + (iron - c) * is_spot);
            // Grains stand proud by up to 0.12 mm.
            heights.push(0.00012 * (1.0 - (f1.values()[i] / 0.7).min(1.0)));
            roughs.push((rough + 0.05 * (g - 0.5)).clamp(0.0, 1.0));
        }
        let mut m = Material::new(grid);
        m.set_param(Param::BaseColor, super::color_map(grid, &colors)?)?;
        m.set_param(Param::SpecularRoughness, scalar_channel(grid, &roughs)?)?;
        m.set_aux(Aux::Height, scalar_channel(grid, &heights)?)?;
        m.set_aux(Aux::Surface, Channel::Constant(Value::Id(BODY)))?;
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}
