// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Oak boards, cut from a solid log.

use alloc::vec;
use alloc::vec::Vec;

use dapple_field::Value;
use dapple_field::program::Op;
use dapple_material::module::{
    Args, Context, Interface, Module, ModuleError, ModuleId, Output, OutputDecl, OutputKind,
    Outputs,
};
use dapple_material::{Aux, Channel, Material, Param};
use glam::Vec3;

use super::{fraction, integer, meters, realize, scalar_channel, seed};
use crate::glazed_brick::WOOD;
use crate::oak;

/// Oak, flat-sawn from a solid log ([`oak::wood_color`]): the board's face
/// is a plane `offset` from the pith, running along the trunk, so its
/// figure is real growth rings cut tangentially (cathedrals), not a
/// painted pattern. The seed moves the cut along the trunk and around it.
///
/// Latewood reads darker and, weathered, stands a little proud
/// (`relief`); roughness follows the rings.
#[derive(Copy, Clone, Debug, Default)]
pub struct Wood;

impl Module for Wood {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId {
                name: "dapple_library.wood",
                version: 1,
            },
            doc: "flat-sawn oak",
            params: vec![
                meters(
                    "offset",
                    [0.02, 0.4],
                    0.14,
                    "distance of the face from the pith",
                ),
                meters("relief", [0.0, 0.002], 0.00015, "latewood relief"),
                fraction("roughness", 0.55, "specular roughness of bare wood"),
                integer("surface", [0, 1 << 16], WOOD, "surface identity"),
                seed(),
            ],
            inputs: vec![],
            outputs: vec![OutputDecl {
                name: "material",
                kind: OutputKind::Material,
                doc: "the board",
            }],
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        #[expect(clippy::cast_precision_loss, reason = "a bounded seed fraction")]
        let along = (args.seed("along") % 1000) as f32 * 0.003;
        #[expect(clippy::cast_precision_loss, reason = "a bounded seed fraction")]
        let turn = (args.seed("around") % 1000) as f32 * 0.006_283;
        let offset = args.scalar("offset");
        let (s, c) = (libm::sinf(turn), libm::cosf(turn));
        let color = realize(grid, |b, _| {
            let solid = oak::wood_color(b)?;
            b.add(Op::Slice {
                input: solid,
                // Across the board is tangential to the rings at `offset`
                // from the pith; along it runs up the trunk.
                origin: [offset * c, offset * s, along],
                u: [0.0, 0.0, 1.0],
                v: [-s, c, 0.0],
                domain: dapple_field::Domain::Plane,
            })
        })?;
        let n = grid.len();
        let (mut heights, mut roughs, mut colors) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        let relief = args.scalar("relief");
        let rough = args.scalar("roughness");
        for i in 0..n {
            let Value::Vector3(col) = color.value(i) else {
                return Err(super::fail("wood color"));
            };
            let lum = col.dot(Vec3::new(0.2126, 0.7152, 0.0722));
            // Latewood (dark) stands proud and is smoother.
            let late = (1.0 - lum / 0.25).clamp(0.0, 1.0);
            heights.push(relief * late);
            roughs.push((rough - 0.1 * late).clamp(0.0, 1.0));
            colors.push(col);
        }
        let mut m = Material::new(grid);
        m.set_param(Param::BaseColor, super::color_map(grid, &colors)?)?;
        m.set_param(Param::SpecularRoughness, scalar_channel(grid, &roughs)?)?;
        m.set_aux(Aux::Height, scalar_channel(grid, &heights)?)?;
        m.set_aux(
            Aux::Surface,
            Channel::Constant(Value::Id(args.integer("surface"))),
        )?;
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}
