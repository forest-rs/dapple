// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Brick: a running-bond wall.

use dapple_field::program::{FieldProgram, NodeId, Op, ProgramBuilder, ProgramError, ValueProgram};
use dapple_field::{Basis, Domain, FractalParams, Pattern, TileOutput};

use crate::ramp;

/// The seed the brick programs derive their noise seeds from.
pub const SEED: u64 = 11;

/// Bricks per domain unit along each axis: on a 1 m tile, 250 × 71 mm
/// courses including the joints.
pub const BRICKS: [f32; 2] = [4.0, 14.0];

/// Half the mortar joint's width, in domain units.
const HALF_JOINT: f32 = 0.005;

/// A running-bond layout's quantity.
fn bond(
    b: &mut ProgramBuilder,
    domain: Domain,
    output: TileOutput,
) -> Result<NodeId, ProgramError> {
    b.add(Op::Tiling {
        domain,
        pattern: Pattern::Bond {
            frequency: BRICKS,
            shift: 0.5,
        },
        seed: SEED,
        output,
    })
}

fn fractal(
    b: &mut ProgramBuilder,
    domain: Domain,
    frequency: [f32; 2],
    seed: u64,
    octaves: u8,
) -> Result<NodeId, ProgramError> {
    b.add(Op::Fractal {
        basis: Basis::Gradient,
        domain,
        frequency,
        seed,
        params: FractalParams {
            octaves,
            ..FractalParams::default()
        },
    })
}

/// How far into a brick a point is, in `[0, 1]`: 0 in the mortar, rising
/// across a worn edge to 1 on the brick's face. Brick edges are chipped
/// irregularly and every brick sits slightly out of line.
fn face(b: &mut ProgramBuilder, domain: Domain) -> Result<NodeId, ProgramError> {
    let edge = bond(b, domain, TileOutput::Edge)?;
    let wobble_x = fractal(b, domain, [8.0, 8.0], SEED + 1, 3)?;
    let wobble_y = fractal(b, domain, [8.0, 8.0], SEED + 2, 3)?;
    let edge = b.add(Op::Warp {
        input: edge,
        dx: wobble_x,
        dy: wobble_y,
        amount: 0.0015,
    })?;
    // Chips: the edge retreats by up to 4 mm where the noise is high.
    let chips = fractal(b, domain, [48.0, 48.0], SEED + 3, 4)?;
    let chips = ramp(b, chips, [0.0, 0.6], [0.0, 0.004])?;
    let edge = b.add(Op::Sub { a: edge, b: chips })?;
    ramp(b, edge, [HALF_JOINT, HALF_JOINT + 0.004], [0.0, 1.0])
}

/// Brick wall height in `[0, 1]`: recessed, sandy mortar joints and brick
/// faces with chipped edges, a slight per-brick tilt and a fired-clay
/// surface texture. Use a periodic domain of period 1 for a 1 m tile.
///
/// # Errors
///
/// Propagates [`ProgramError`]s; a periodic domain must fit whole courses.
pub fn height(domain: Domain) -> Result<FieldProgram, ProgramError> {
    let mut b = ProgramBuilder::new();
    let face = face(&mut b, domain)?;
    let mortar = fractal(&mut b, domain, [160.0, 160.0], SEED + 4, 3)?;
    let mortar = b.add(Op::Remap {
        input: mortar,
        from: [-1.0, 1.0],
        to: [0.05, 0.2],
    })?;
    let tilt = bond(&mut b, domain, TileOutput::TileValue)?;
    let tilt = b.add(Op::Remap {
        input: tilt,
        from: [0.0, 1.0],
        to: [0.72, 0.84],
    })?;
    let texture = fractal(&mut b, domain, [90.0, 90.0], SEED + 5, 4)?;
    let texture = b.add(Op::Remap {
        input: texture,
        from: [-1.0, 1.0],
        to: [-0.1, 0.1],
    })?;
    let brick = b.add(Op::Add {
        a: tilt,
        b: texture,
    })?;
    let height = b.add(Op::Mix {
        a: mortar,
        b: brick,
        t: face,
    })?;
    b.finish(height)
}

/// Linear brick wall color: each brick's own tone between two clay reds,
/// the occasional over-fired dark brick, speckle and faint soot, set in pale
/// grey mortar.
///
/// # Errors
///
/// Propagates [`ProgramError`]s; a periodic domain must fit whole courses.
pub fn color(domain: Domain) -> Result<ValueProgram, ProgramError> {
    let mut b = ProgramBuilder::new();
    let color = |b: &mut ProgramBuilder, rgb: [f32; 3]| {
        let [r, g, bl] = rgb.map(|value| b.add(Op::Constant { domain, value }));
        b.add(Op::Color {
            r: r?,
            g: g?,
            b: bl?,
        })
    };
    let face = face(&mut b, domain)?;
    let tone = bond(&mut b, domain, TileOutput::TileValue)?;
    let light = color(&mut b, [0.42, 0.14, 0.07])?;
    let dark = color(&mut b, [0.24, 0.075, 0.045])?;
    let burnt = color(&mut b, [0.09, 0.045, 0.035])?;
    let brick = b.add(Op::Mix {
        a: dark,
        b: light,
        t: tone,
    })?;
    let fired = ramp(&mut b, tone, [0.9, 0.93], [0.0, 0.85])?;
    let brick = b.add(Op::Mix {
        a: brick,
        b: burnt,
        t: fired,
    })?;
    // Speckle and broad soot, darkening multiplicatively.
    let speckle = b.add(Op::Noise {
        basis: Basis::Value,
        domain,
        frequency: [300.0, 300.0],
        seed: SEED + 6,
    })?;
    let speckle = ramp(&mut b, speckle, [0.5, 0.9], [1.0, 0.7])?;
    let soot = fractal(&mut b, domain, [3.0, 3.0], SEED + 7, 4)?;
    let soot = ramp(&mut b, soot, [-0.5, 0.6], [1.0, 0.75])?;
    let shade = b.add(Op::Mul {
        a: speckle,
        b: soot,
    })?;
    let brick = b.add(Op::Mul { a: brick, b: shade })?;
    let mortar = color(&mut b, [0.36, 0.34, 0.31])?;
    let grit = fractal(&mut b, domain, [200.0, 200.0], SEED + 8, 2)?;
    let grit = ramp(&mut b, grit, [-1.0, 1.0], [0.8, 1.1])?;
    let mortar = b.add(Op::Mul { a: mortar, b: grit })?;
    let color = b.add(Op::Mix {
        a: mortar,
        b: brick,
        t: face,
    })?;
    b.finish_value(color)
}
