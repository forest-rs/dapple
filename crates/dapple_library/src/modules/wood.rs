// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Oak boards, cut from a solid log.

use alloc::vec;
use alloc::vec::Vec;

use dapple_field::hash::{hash, unit_f32};
use dapple_field::{Edge, Value};
use dapple_material::module::{
    Args, Context, Interface, Module, ModuleError, ModuleId, Output, OutputDecl, OutputKind,
    Outputs,
};
use dapple_material::{Aux, Channel, Grid, Material, Param, Tiling};
use glam::{Vec2, Vec3};

use super::{fraction, integer, meters, realize, scalar_channel, seed};
use crate::glazed_brick::WOOD;
use crate::oak;

/// Oak boards, flat-sawn from a solid log ([`oak::wood_face_color`]),
/// laid side by side along the grid's x, `width` apart in y from `y = 0`.
///
/// Each board's face is a plane about `offset` from the pith, running
/// along the trunk, so its figure is real growth rings cut tangentially
/// (a cathedral about its middle), not a painted pattern. The seed moves
/// the first board's cut (the one from `y = 0` to `width`) along the trunk
/// and around it; every other board is its own cut, from elsewhere in the
/// log, a little nearer the pith or further from it, its cathedral off
/// center, its tone a little lighter or darker. Rings and rays are
/// band-limited by the grid's texels as they cross the face, so a coarse
/// grid shows them faded toward their mean, not aliased.
///
/// Between boards is a narrow dark gap with eased edges, averaged over
/// each texel row. Latewood reads darker and, weathered, stands a little
/// proud (`relief`); roughness follows the rings.
#[derive(Copy, Clone, Debug, Default)]
pub struct Wood;

/// The gap between boards, in meters.
const GAP: f32 = 0.0006;
/// How far each board's edge is eased (rounded over), in meters.
const EASE: f32 = 0.0012;
/// How far an eased edge falls by the time it meets the gap, in meters.
const EDGE_DROP: f32 = 0.0003;
/// How far below the faces the gap's floor lies, in meters.
const GAP_DEPTH: f32 = 0.0008;
/// Samples across a texel row for its share of gap and eased edge.
const ROW_SAMPLES: u32 = 16;

/// One board's cut through the log: the face's solid origin and axes, and
/// a tone multiplying its color.
struct Board {
    origin: [f32; 3],
    u: [f32; 3],
    v: [f32; 3],
    tone: f32,
}

impl Board {
    /// Board `k`, covering `y` in `[k·width, (k + 1)·width)`. Board 0 is the
    /// seed's own cut, `offset` from the pith with its cathedral centered;
    /// the others are hashed from `boards`.
    fn new(k: i64, width: f32, offset: f32, args: &Args) -> Self {
        #[expect(clippy::cast_precision_loss, reason = "a bounded seed fraction")]
        let along = (args.seed("along") % 1000) as f32 * 0.003;
        #[expect(clippy::cast_precision_loss, reason = "a bounded seed fraction")]
        let turn = (args.seed("around") % 1000) as f32 * 0.006_283;
        let (along, turn, offset, shift, tone) = if k == 0 {
            (along, turn, offset, 0.0, 1.0)
        } else {
            let u = |i: u64| unit_f32(hash(args.seed("boards"), &[k.cast_unsigned(), i]));
            (
                // Elsewhere along the trunk, anywhere around it.
                along + 0.4 + 2.6 * u(0),
                core::f32::consts::TAU * u(1),
                offset * (0.7 + 0.6 * u(2)),
                (u(3) - 0.5) * 0.3 * width,
                0.93 + 0.14 * u(4),
            )
        };
        let (s, c) = (libm::sinf(turn), libm::cosf(turn));
        // Across the board is tangential to the rings at `offset` from the
        // pith, at the board's middle (shifted); along it runs up the trunk.
        #[expect(clippy::cast_precision_loss, reason = "board indices are small")]
        let middle = (k as f32 + 0.5) * width + shift;
        let v = [-s, c, 0.0];
        Self {
            origin: [
                offset * c - v[0] * middle,
                offset * s - v[1] * middle,
                along,
            ],
            u: [0.0, 0.0, 1.0],
            v,
            tone,
        }
    }
}

/// The board under the texel row centered at `y`, the boards `width` wide.
fn board_of(y: f32, width: f32) -> i64 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "board indices of a grid are small"
    )]
    let k = libm::floorf(y / width) as i64;
    k
}

/// A texel row's share of gap, and its mean height from the eased edges
/// and the gap, for the row centered at `y` and `height` tall.
fn joint(y: f32, height: f32, width: f32) -> (f32, f32) {
    let (mut gap, mut drop) = (0.0, 0.0);
    for i in 0..ROW_SAMPLES {
        #[expect(clippy::cast_precision_loss, reason = "a few samples")]
        let t = (i as f32 + 0.5) / ROW_SAMPLES as f32 - 0.5;
        let at = y + t * height;
        // The distance to the nearest joint, and from it to the board.
        let d = (at - libm::roundf(at / width) * width).abs() - 0.5 * GAP;
        if d < 0.0 {
            gap += 1.0;
            drop += GAP_DEPTH;
        } else {
            let e = (1.0 - d / EASE).max(0.0);
            drop += EDGE_DROP * e * e;
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "a few samples")]
    let n = ROW_SAMPLES as f32;
    (gap / n, drop / n)
}

impl Module for Wood {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId::new("dapple_library.wood", 3),
            doc: "flat-sawn oak boards".into(),
            params: vec![
                meters(
                    "offset",
                    [0.02, 0.4],
                    0.14,
                    "distance of the face from the pith",
                ),
                meters("width", [0.05, 1.0], 0.2, "width of each board"),
                meters("relief", [0.0, 0.002], 0.00015, "latewood relief"),
                fraction("roughness", 0.55, "specular roughness of bare wood"),
                integer("surface", [0, 1 << 16], WOOD, "surface identity"),
                seed(),
            ],
            inputs: vec![],
            outputs: vec![OutputDecl {
                name: "material".into(),
                kind: OutputKind::Material,
                doc: "the boards".into(),
            }],
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let (offset, width) = (args.scalar("offset"), args.scalar("width"));
        #[expect(clippy::cast_precision_loss, reason = "grid sizes are small")]
        let row_y = |j: u32| grid.origin.y + (j as f32 + 0.5) * grid.texel.y;
        let n = grid.len();
        let mut colors = Vec::with_capacity(n);
        // Each run of rows on one board is realized as its own cut.
        let mut j = 0;
        while j < grid.height {
            let k = board_of(row_y(j), width);
            let end = (j + 1..grid.height)
                .find(|&e| board_of(row_y(e), width) != k)
                .unwrap_or(grid.height);
            let board = Board::new(k, width, offset, args);
            #[expect(clippy::cast_precision_loss, reason = "grid sizes are small")]
            let rows = Grid {
                height: end - j,
                origin: Vec2::new(grid.origin.x, grid.origin.y + j as f32 * grid.texel.y),
                edge: Edge::Clamp,
                ..grid
            };
            let color = realize(rows, |b, _| {
                oak::wood_face_color(b, board.origin, board.u, board.v)
            })?;
            for i in 0..rows.len() {
                let Value::Vector3(col) = color.value(i) else {
                    return Err(super::fail("wood color"));
                };
                colors.push(col * board.tone);
            }
            j = end;
        }
        let (mut heights, mut roughs) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let relief = args.scalar("relief");
        let rough = args.scalar("roughness");
        let row = grid.width as usize;
        for (r, colors) in colors.chunks_mut(row).enumerate() {
            #[expect(clippy::cast_possible_truncation, reason = "grid sizes are small")]
            let (gap, drop) = joint(row_y(r as u32), grid.texel.y, width);
            for col in colors {
                let lum = col.dot(Vec3::new(0.2126, 0.7152, 0.0722));
                // Latewood (dark) stands proud and is smoother.
                let late = (1.0 - lum / 0.25).clamp(0.0, 1.0);
                heights.push(relief * late * (1.0 - gap) - drop);
                // The gap between boards is dark and rough.
                roughs.push(((rough - 0.1 * late) * (1.0 - gap) + 0.85 * gap).clamp(0.0, 1.0));
                *col *= 1.0 - 0.8 * gap;
            }
        }
        let mut m = Material::new(grid);
        // A board cut from a log does not repeat.
        m.set_tiling(Tiling::NONE);
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
