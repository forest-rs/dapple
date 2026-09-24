// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Ready-made dapple materials, built as field programs.
//!
//! Each material is a function that builds a [`dapple_field`] program, so its
//! graph is inspectable, fingerprinted and evaluable like any other: realize
//! it with `dapple_raster`, bake it onto a surface with `dapple_exedra`, or
//! pack it with `dapple_encode`.
//!
//! - [`oak`]: oak bark (planar, tileable) and solid oak wood (a solid field
//!   around a trunk axis, for slices and surface charts).
//! - [`brick`]: a running-bond brick wall, tileable.
//! - [`gravel`]: scattered pebbles of three sizes over sand, tileable.
//! - [`parquet`]: oak planks in herringbone, tileable.
//! - [`glazed_brick`]: keyed glazed bricks whose identity drives shape,
//!   glaze thickness and color, and chips (`dapple_elements`).
//! - [`modules`]: reusable material modules (`dapple_material`): ceramic
//!   body, mortar, stone, wood, finishes and weathering, and assets built
//!   from them: a glazed brick wall, a stone sill, a varnished board and a
//!   threshold.

#![no_std]

extern crate alloc;

use dapple_field::program::{NodeId, Op, ProgramBuilder, ProgramError};

pub mod brick;
pub mod glazed_brick;
pub mod gravel;
pub mod modules;
pub mod oak;
pub mod parquet;

#[cfg(test)]
mod tests;

/// `input` remapped from `from` to `to`, clamped to `to`.
///
/// # Errors
///
/// Propagates [`ProgramError`]s from adding the nodes.
pub fn ramp(
    b: &mut ProgramBuilder,
    input: NodeId,
    from: [f32; 2],
    to: [f32; 2],
) -> Result<NodeId, ProgramError> {
    let t = b.add(Op::Remap { input, from, to })?;
    b.add(Op::Clamp {
        input: t,
        min: to[0].min(to[1]),
        max: to[0].max(to[1]),
    })
}
