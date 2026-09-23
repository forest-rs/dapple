// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Parquet: oak planks laid in herringbone.

use dapple_field::program::{FieldProgram, NodeId, Op, ProgramBuilder, ProgramError, ValueProgram};
use dapple_field::{Basis, Domain, FractalParams, Pattern, TileOutput};

use crate::ramp;

/// The seed the parquet programs derive their noise seeds from.
pub const SEED: u64 = 13;

/// Plank widths per domain unit: 62.5 mm planks on a 1 m tile.
pub const PLANKS: f32 = 16.0;

/// Plank length in widths.
pub const RATIO: u32 = 4;

fn layout(
    b: &mut ProgramBuilder,
    domain: Domain,
    output: TileOutput,
) -> Result<NodeId, ProgramError> {
    b.add(Op::Tiling {
        domain,
        pattern: Pattern::Herringbone {
            frequency: PLANKS,
            ratio: RATIO,
        },
        seed: SEED,
        output,
    })
}

/// A fractal stretched along each plank's length: `frequency` is along and
/// across the grain, and planks along y swap the axes.
fn along_grain(
    b: &mut ProgramBuilder,
    domain: Domain,
    vertical: NodeId,
    frequency: [f32; 2],
    seed: u64,
) -> Result<NodeId, ProgramError> {
    let mut fractal = |frequency| {
        b.add(Op::Fractal {
            basis: Basis::Gradient,
            domain,
            frequency,
            seed,
            params: FractalParams {
                octaves: 4,
                ..FractalParams::default()
            },
        })
    };
    let along_x = fractal(frequency)?;
    let along_y = fractal([frequency[1], frequency[0]])?;
    b.add(Op::Mix {
        a: along_x,
        b: along_y,
        t: vertical,
    })
}

/// Latewood lines: a plank's growth rings cut at a shallow angle, each plank
/// cut from its own place in its log.
fn figure(b: &mut ProgramBuilder, domain: Domain) -> Result<NodeId, ProgramError> {
    let vertical = layout(b, domain, TileOutput::Vertical)?;
    let offset = layout(b, domain, TileOutput::TileValue)?;
    let rings = along_grain(b, domain, vertical, [2.0, 30.0], SEED + 1)?;
    let offset = b.add(Op::Remap {
        input: offset,
        from: [0.0, 1.0],
        to: [0.0, 5.0],
    })?;
    let rings = b.add(Op::Remap {
        input: rings,
        from: [-1.0, 1.0],
        to: [-8.0, 8.0],
    })?;
    let rings = b.add(Op::Add {
        a: rings,
        b: offset,
    })?;
    let phase = b.add(Op::Fract { input: rings })?;
    ramp(b, phase, [0.62, 0.86], [0.0, 1.0])
}

/// How far into a plank a point is: 0 in the joint, 1 half a millimeter in.
fn plank(b: &mut ProgramBuilder, domain: Domain) -> Result<NodeId, ProgramError> {
    let edge = layout(b, domain, TileOutput::Edge)?;
    ramp(b, edge, [0.0002, 0.0008], [0.0, 1.0])
}

/// Parquet height in `[0, 1]`: flat planks with fine grain relief and
/// narrow joints between them. Use a periodic domain of period 1 for a 1 m
/// tile.
///
/// # Errors
///
/// Propagates [`ProgramError`]s; a periodic domain must fit whole repeats.
pub fn height(domain: Domain) -> Result<FieldProgram, ProgramError> {
    let mut b = ProgramBuilder::new();
    let plank = plank(&mut b, domain)?;
    let figure = figure(&mut b, domain)?;
    let relief = b.add(Op::Remap {
        input: figure,
        from: [0.0, 1.0],
        to: [0.9, 0.85],
    })?;
    let height = b.add(Op::Mul {
        a: plank,
        b: relief,
    })?;
    b.finish(height)
}

/// Linear parquet color: oak planks each with their own tone, latewood
/// lines and streaky grain along each plank, dark joints between them.
///
/// # Errors
///
/// Propagates [`ProgramError`]s; a periodic domain must fit whole repeats.
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
    let tone = layout(&mut b, domain, TileOutput::TileValue)?;
    let pale = color(&mut b, [0.5, 0.31, 0.14])?;
    let deep = color(&mut b, [0.34, 0.18, 0.07])?;
    let wood = b.add(Op::Mix {
        a: pale,
        b: deep,
        t: tone,
    })?;
    let figure = figure(&mut b, domain)?;
    let figure = b.add(Op::Remap {
        input: figure,
        from: [0.0, 1.0],
        to: [1.0, 0.72],
    })?;
    let vertical = layout(&mut b, domain, TileOutput::Vertical)?;
    let streaks = along_grain(&mut b, domain, vertical, [4.0, 160.0], SEED + 2)?;
    let streaks = ramp(&mut b, streaks, [-1.0, 1.0], [0.85, 1.1])?;
    let shade = b.add(Op::Mul {
        a: figure,
        b: streaks,
    })?;
    let wood = b.add(Op::Mul { a: wood, b: shade })?;
    let joint = color(&mut b, [0.05, 0.03, 0.02])?;
    let plank = plank(&mut b, domain)?;
    let color = b.add(Op::Mix {
        a: joint,
        b: wood,
        t: plank,
    })?;
    b.finish_value(color)
}
