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

#![no_std]

use dapple_field::program::{NodeId, Op, ProgramBuilder, ProgramError};

pub mod oak;

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
