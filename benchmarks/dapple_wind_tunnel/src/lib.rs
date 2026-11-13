// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared fixtures for dapple's wind tunnel benchmarks.
//!
//! Every fixture is deterministic: fixed seeds, fixed sizes, no clocks and no
//! randomness, so that the same benchmark on two revisions measures the same
//! work. The benchmarks themselves live in `benches/`; run them with
//! `cargo bench -p dapple_wind_tunnel`.

use dapple_field::program::{FieldProgram, NodeId, Op, ProgramBuilder, ValueProgram};
use dapple_field::{Basis, Domain, FractalParams};
use dapple_library::oak;
use glam::Vec2;

/// The unit period every fixture program is defined over.
///
/// # Panics
///
/// Never: a 1 by 1 period is valid.
#[must_use]
pub fn period() -> Domain {
    Domain::periodic(1, 1).expect("a unit period is valid")
}

/// A pointwise program over constants with a shared subgraph, so that almost
/// all of its cost is per-call setup rather than field math.
///
/// # Panics
///
/// Never: the program is valid by construction.
#[must_use]
pub fn trivial_value() -> ValueProgram {
    let domain = period();
    let mut b = ProgramBuilder::new();
    let x = b
        .add(Op::Constant {
            domain,
            value: 0.25,
        })
        .expect("valid constant");
    let y = b
        .add(Op::Constant { domain, value: 0.5 })
        .expect("valid constant");
    let sum = b.add(Op::Add { a: x, b: y }).expect("valid add");
    let square = b.add(Op::Mul { a: sum, b: sum }).expect("valid mul");
    let out = b.add(Op::Add { a: square, b: sum }).expect("valid add");
    b.finish_value(out).expect("a planar output")
}

/// A scalar program of a few nodes: a five-octave fractal, its absolute
/// value, and a remap.
///
/// # Panics
///
/// Never: the program is valid by construction.
#[must_use]
pub fn small_scalar() -> FieldProgram {
    let domain = period();
    let mut b = ProgramBuilder::new();
    let fractal = b
        .add(Op::Fractal {
            basis: Basis::Gradient,
            domain,
            frequency: [8.0, 8.0],
            seed: 7,
            params: FractalParams::default(),
        })
        .expect("valid fractal");
    let abs = b.add(Op::Abs { input: fractal }).expect("valid abs");
    let out = b
        .add(Op::Remap {
            input: abs,
            from: [0.0, 1.0],
            to: [0.2, 0.9],
        })
        .expect("valid remap");
    b.finish(out).expect("a scalar output")
}

/// A color program of about a dozen nodes: two fractals ramped into the
/// channels of a mix between two constant colors.
///
/// # Panics
///
/// Never: the program is valid by construction.
#[must_use]
pub fn small_color() -> ValueProgram {
    let domain = period();
    let mut b = ProgramBuilder::new();
    let constant = |b: &mut ProgramBuilder, value| {
        b.add(Op::Constant { domain, value })
            .expect("valid constant")
    };
    let color = |b: &mut ProgramBuilder, rgb: [f32; 3]| {
        let [r, g, bl] = rgb.map(|value| constant(b, value));
        b.add(Op::Color { r, g, b: bl }).expect("valid color")
    };
    let dark = color(&mut b, [0.05, 0.03, 0.02]);
    let light = color(&mut b, [0.6, 0.5, 0.35]);
    let fractal = b
        .add(Op::Fractal {
            basis: Basis::Gradient,
            domain,
            frequency: [6.0, 6.0],
            seed: 11,
            params: FractalParams::default(),
        })
        .expect("valid fractal");
    let t = b
        .add(Op::Remap {
            input: fractal,
            from: [-1.0, 1.0],
            to: [0.0, 1.0],
        })
        .expect("valid remap");
    let t = b
        .add(Op::Clamp {
            input: t,
            min: 0.0,
            max: 1.0,
        })
        .expect("valid clamp");
    let out = b
        .add(Op::Mix {
            a: dark,
            b: light,
            t,
        })
        .expect("valid mix");
    b.finish_value(out).expect("a planar output")
}

/// The library's oak bark height: cellular plates under two domain warps,
/// fissure cracks and grain, a real, moderately deep scalar program.
///
/// # Panics
///
/// Never: the library builds it on a valid domain.
#[must_use]
pub fn deep_scalar() -> FieldProgram {
    oak::bark_height(period()).expect("the library's bark height builds")
}

/// The library's oak bark color: [`deep_scalar`] ramped, patched with
/// weathering and lichen, and mixed between four colors.
///
/// # Panics
///
/// Never: the library builds it on a valid domain.
#[must_use]
pub fn deep_color() -> ValueProgram {
    oak::bark_color(period(), &deep_scalar()).expect("the library's bark color builds")
}

/// A chain of `len` dependent nodes over one fractal source, every node
/// with its own parameters so no two share a fingerprint.
///
/// Returns the builder and the chain's last node.
///
/// # Panics
///
/// Never: the program is valid by construction.
#[must_use]
pub fn chain_builder(len: usize) -> (ProgramBuilder, NodeId) {
    let mut b = ProgramBuilder::new();
    let mut last = b
        .add(Op::Fractal {
            basis: Basis::Gradient,
            domain: period(),
            frequency: [4.0, 4.0],
            seed: 3,
            params: FractalParams::default(),
        })
        .expect("valid fractal");
    for i in 0..len {
        #[expect(
            clippy::cast_precision_loss,
            reason = "chains are far below 2^24 nodes"
        )]
        let top = 1.0 + i as f32 * 1.0e-4;
        last = b
            .add(Op::Remap {
                input: last,
                from: [0.0, 1.0],
                to: [0.0, top],
            })
            .expect("valid remap");
    }
    (b, last)
}

/// [`chain_builder`] finished into a program.
///
/// # Panics
///
/// Never: the chain's output is a planar scalar.
#[must_use]
pub fn chain(len: usize) -> FieldProgram {
    let (b, last) = chain_builder(len);
    b.finish(last).expect("a scalar output")
}

/// `side * side` points at the centers of a regular grid over the unit
/// period, in row-major order.
#[must_use]
pub fn grid_points(side: u32) -> Vec<Vec2> {
    #[expect(clippy::cast_precision_loss, reason = "grid sides are tiny")]
    let step = 1.0 / side as f32;
    (0..side)
        .flat_map(|y| {
            (0..side).map(move |x| {
                #[expect(clippy::cast_precision_loss, reason = "grid sides are tiny")]
                let center = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                center * step
            })
        })
        .collect()
}
