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

/// Oak boards sawn from a solid log ([`oak::wood_face_color`]), laid side
/// by side along the grid's x, `width` apart in y from `y = 0`.
///
/// Each board's face is a plane through the log, running along the trunk,
/// so its figure is real growth rings cut by it, not a painted pattern. The
/// first board (the one from `y = 0` to `width`) is flat-sawn `offset` from
/// the pith, its cathedral about its middle; the seed moves it along the
/// trunk and around it. Every other board is its own cut from elsewhere in
/// the log, as a log sawn through and through gives them: about half
/// flat-sawn, nearer the pith or further from it, with long cathedrals; a
/// quarter rift-sawn, the rings cut at 30° to 55° into straight grain; and
/// a quarter quarter-sawn, the face nearly along the rays, which show as
/// flecks. Each face leans a little along the trunk, so its cathedrals come
/// faster or slower, and each board's tone is a little lighter or darker,
/// warmer or cooler. Rings and rays are band-limited by the grid's texels
/// as they cross the face, so a coarse grid shows them faded toward their
/// mean, not aliased.
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
/// a tint multiplying its color.
struct Board {
    origin: [f32; 3],
    u: [f32; 3],
    v: [f32; 3],
    tint: Vec3,
}

impl Board {
    /// Board `k`, covering `y` in `[k·width, (k + 1)·width)`. Board 0 is the
    /// seed's own cut, flat-sawn `offset` from the pith with its cathedral
    /// centered, leaning so they come about every quarter meter; the others
    /// are hashed from `boards`.
    fn new(k: i64, width: f32, offset: f32, seed: &dyn Fn(&str) -> u64) -> Self {
        use core::f32::consts::TAU;
        #[expect(clippy::cast_precision_loss, reason = "a bounded seed fraction")]
        let along = (seed("along") % 1000) as f32 * 0.003;
        #[expect(clippy::cast_precision_loss, reason = "a bounded seed fraction")]
        let turn = (seed("around") % 1000) as f32 * 0.006_283;
        // The board's middle is `radius` from the pith; across it, the face
        // meets the rings at `slant` (0 flat-sawn, a right angle
        // quarter-sawn); along it, the face leans `lean` per meter toward
        // or away from the pith.
        let (along, turn, radius, slant, lean, tint) = if k == 0 {
            (along, turn, offset, 0.0, 0.004, Vec3::ONE)
        } else {
            let u = |i: u64| unit_f32(hash(seed("boards"), &[k.cast_unsigned(), i]));
            // Sawn through and through, a log gives flat-sawn boards from
            // its flanks and rift- and quarter-sawn ones from near its
            // middle: about half flat, a quarter rift and a quarter
            // quarter-sawn, each slanted either way.
            let kind = u(5);
            let slant = if kind < 0.5 {
                0.25 * u(3)
            } else if kind < 0.75 {
                0.5 + 0.45 * u(3)
            } else {
                1.15 + 0.35 * u(3)
            };
            // A quarter-sawn board's face passes near the pith; the board
            // keeps clear of it, its near edge at least 3 cm out.
            let radius = offset * (0.6 + 0.9 * u(2));
            let radius = if kind < 0.75 {
                radius
            } else {
                radius.max((0.03 + 0.5 * width) / libm::sinf(slant))
            };
            let slant = if u(6) < 0.5 { -slant } else { slant };
            let tone = 0.88 + 0.24 * u(4);
            let warmth = 0.08 * (u(8) - 0.5);
            (
                // Elsewhere along the trunk, anywhere around it.
                along + 0.4 + 2.6 * u(0),
                TAU * u(1),
                radius,
                slant,
                0.016 * (u(7) - 0.5),
                tone * Vec3::new(1.0 + warmth, 1.0, 1.0 - warmth),
            )
        };
        let (s, c) = (libm::sinf(turn), libm::cosf(turn));
        // The face's normal points away from the pith, across it is
        // tangential to the rings at `turn`, and along it runs up the
        // trunk, leaning.
        let n = Vec3::new(c, s, 0.0);
        let v = Vec3::new(-s, c, 0.0);
        let u = (Vec3::Z + n * lean).normalize();
        let distance = radius * libm::cosf(slant);
        let across = radius * libm::sinf(slant);
        #[expect(clippy::cast_precision_loss, reason = "board indices are small")]
        let middle = (k as f32 + 0.5) * width;
        // The face's point at the board's middle is `distance` out along the
        // normal and `across` along it, at `along` up the trunk.
        let origin = n * distance + v * (across - middle) + Vec3::Z * along;
        Self {
            origin: origin.to_array(),
            u: u.to_array(),
            v: v.to_array(),
            tint,
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
            id: ModuleId::new("dapple_library.wood", 4),
            doc: "oak boards, flat-, rift- and quarter-sawn".into(),
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
            let board = Board::new(k, width, offset, &|purpose| args.seed(purpose));
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
                colors.push(col * board.tint);
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

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::Board;

    /// The boards from one log are a mix of cuts, each by the angle its
    /// face meets the rings at mid-board, and none passes within 3 cm of
    /// the pith.
    #[test]
    fn boards_are_flat_rift_and_quarter_sawn_clear_of_the_pith() {
        let (width, offset) = (0.2, 0.14);
        let seed = |purpose: &str| purpose.len() as u64 * 0x9e37_79b9;
        let (mut flat, mut rift, mut quarter) = (0, 0, 0);
        let count = 200;
        for k in 1..=count {
            let board = Board::new(k, width, offset, &seed);
            let (origin, v) = (Vec3::from(board.origin), Vec3::from(board.v));
            let at = |y: f32| (origin + v * y).truncate();
            #[expect(clippy::cast_precision_loss, reason = "small board indices")]
            let low = k as f32 * width;
            let middle = at(low + 0.5 * width);
            // The angle between the face and the ring's tangent there.
            let slant = libm::asinf(middle.normalize().dot(v.truncate()).abs());
            if slant < 0.3 {
                flat += 1;
            } else if slant < 1.0 {
                rift += 1;
            } else {
                quarter += 1;
            }
            let nearest = (0..=100)
                .map(|i| at(low + width * i as f32 / 100.0).length())
                .fold(f32::INFINITY, f32::min);
            assert!(
                nearest > 0.029,
                "board {k} passes {nearest} m from the pith"
            );
        }
        for (name, n, share) in [
            ("flat", flat, 0.5),
            ("rift", rift, 0.25),
            ("quarter", quarter, 0.25),
        ] {
            #[expect(clippy::cast_precision_loss, reason = "small counts")]
            let found = n as f32 / count as f32;
            assert!(
                (found - share).abs() < 0.12,
                "{found} of the boards {name}-sawn"
            );
        }
    }
}
