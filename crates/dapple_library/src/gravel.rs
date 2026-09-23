// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Gravel: pebbles of mixed sizes and stones over sandy soil.

use dapple_field::program::{FieldProgram, NodeId, Op, ProgramBuilder, ProgramError, ValueProgram};
use dapple_field::{Basis, Domain, FractalParams, Placement, ScatterOutput, Stamp};

use crate::ramp;

/// The seed the gravel programs derive their noise seeds from.
pub const SEED: u64 = 17;

/// Pebble layers, coarse to fine: cells per unit, density, radii in cells,
/// and height scale.
const LAYERS: [(f32, f32, [f32; 2], f32); 3] = [
    (16.0, 0.35, [0.45, 0.8], 1.0),
    (40.0, 0.5, [0.4, 0.85], 0.55),
    (96.0, 0.7, [0.4, 0.9], 0.3),
];

/// One layer's pebble heights (domes, scaled and slightly squashed per
/// pebble) and each pebble's own value.
fn layer(
    b: &mut ProgramBuilder,
    domain: Domain,
    index: usize,
) -> Result<(NodeId, NodeId), ProgramError> {
    let (frequency, density, radius, scale) = LAYERS[index];
    let seed = SEED + 10 * index as u64;
    let scatter = |b: &mut ProgramBuilder, output| {
        b.add(Op::Scatter {
            domain,
            placement: Placement {
                frequency,
                density,
                radius,
                rotate: false,
            },
            stamp: Stamp::Dome,
            seed,
            output,
        })
    };
    let dome = scatter(b, ScatterOutput::Max)?;
    let value = scatter(b, ScatterOutput::TopValue)?;
    // Flattened domes: pebbles are lenses, not hemispheres.
    let height = ramp(b, dome, [0.0, 1.0], [0.0, 0.6 * scale])?;
    // Surface pitting.
    let pits = b.add(Op::Fractal {
        basis: Basis::Gradient,
        domain,
        frequency: [frequency * 6.0, frequency * 6.0],
        seed: seed + 1,
        params: FractalParams {
            octaves: 2,
            ..FractalParams::default()
        },
    })?;
    let pits = ramp(b, pits, [-1.0, 1.0], [-0.03 * scale, 0.03 * scale])?;
    let covered = ramp(b, dome, [0.0, 0.05], [0.0, 1.0])?;
    let pits = b.add(Op::Mul {
        a: pits,
        b: covered,
    })?;
    let height = b.add(Op::Add { a: height, b: pits })?;
    Ok((height, value))
}

/// The tallest pebble at each point, its value, and the ground between.
fn pebbles(b: &mut ProgramBuilder, domain: Domain) -> Result<(NodeId, NodeId), ProgramError> {
    let ground = b.add(Op::Fractal {
        basis: Basis::Gradient,
        domain,
        frequency: [8.0, 8.0],
        seed: SEED,
        params: FractalParams::default(),
    })?;
    let mut height = ramp(b, ground, [-1.0, 1.0], [0.0, 0.08])?;
    let mut value = b.add(Op::Constant {
        domain,
        value: -1.0,
    })?;
    for index in 0..LAYERS.len() {
        let (h, v) = layer(b, domain, index)?;
        // Whichever is higher shows: a steep ramp over the height
        // difference picks the pebble, with a thin blend at contacts.
        let above = b.add(Op::Sub { a: h, b: height })?;
        let above = ramp(b, above, [-0.004, 0.004], [0.0, 1.0])?;
        height = b.add(Op::Max { a: height, b: h })?;
        value = b.add(Op::Mix {
            a: value,
            b: v,
            t: above,
        })?;
    }
    Ok((height, value))
}

/// Gravel height in `[0, 1]`: three sizes of lens-shaped pebbles, larger
/// ones standing proud, over a gently rolling sandy ground. Use a periodic
/// domain of period 1 for a 1 m tile.
///
/// # Errors
///
/// Propagates [`ProgramError`]s; a periodic domain must fit whole cells.
pub fn height(domain: Domain) -> Result<FieldProgram, ProgramError> {
    let mut b = ProgramBuilder::new();
    let (height, _) = pebbles(&mut b, domain)?;
    b.finish(height)
}

/// Linear gravel color: each pebble its own stone, from pale limestone
/// through ocher flint to dark basalt, over sand.
///
/// # Errors
///
/// Propagates [`ProgramError`]s; a periodic domain must fit whole cells.
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
    let (_, value) = pebbles(&mut b, domain)?;
    let pale = color(&mut b, [0.55, 0.52, 0.46])?;
    let ocher = color(&mut b, [0.42, 0.3, 0.17])?;
    let dark = color(&mut b, [0.1, 0.1, 0.1])?;
    let sand = color(&mut b, [0.3, 0.25, 0.18])?;
    let warm = ramp(&mut b, value, [0.0, 0.6], [0.0, 1.0])?;
    let stone = b.add(Op::Mix {
        a: pale,
        b: ocher,
        t: warm,
    })?;
    let basalt = ramp(&mut b, value, [0.9, 0.92], [0.0, 1.0])?;
    let stone = b.add(Op::Mix {
        a: stone,
        b: dark,
        t: basalt,
    })?;
    // Sand where no pebble is (the value is −1 there).
    let pebble = ramp(&mut b, value, [-0.5, 0.0], [0.0, 1.0])?;
    let grains = b.add(Op::Noise {
        basis: Basis::Value,
        domain,
        frequency: [400.0, 400.0],
        seed: SEED + 5,
    })?;
    let grains = ramp(&mut b, grains, [0.0, 1.0], [0.75, 1.2])?;
    let sand = b.add(Op::Mul { a: sand, b: grains })?;
    let color = b.add(Op::Mix {
        a: sand,
        b: stone,
        t: pebble,
    })?;
    b.finish_value(color)
}
