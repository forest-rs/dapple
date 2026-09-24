// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Scoped programs over material maps.
//!
//! A module computes masks and channels from a material with an ordinary
//! `dapple_field` [`ScopedProgram`], inspectable and fingerprinted, whose
//! inputs are bound here ([`MapBinding`]): to constants, to material
//! channels, to rasters a pass produced, or to the texel's position.
//! Material-scope nodes run once; the rest once per texel.
//!
//! Scopes are checked against the bindings: a channel or plain map is
//! [`Scope::Sample`], a pass result (ambient occlusion, a blur, a streak)
//! is [`Scope::Pass`], so an output declared per sample never secretly
//! depends on a neighborhood.

use alloc::vec;
use alloc::vec::Vec;

use dapple_field::scoped::{ContractError, Scope, ScopedProgram, Shape};
use dapple_field::{Footprint, PortType, Value};
use dapple_raster::typed::TypedRaster;
use glam::Vec2;

use crate::material::{ChannelId, Material, MaterialError, Tiling};

/// Where a program input's values come from.
#[derive(Clone, Debug, PartialEq)]
pub enum MapBinding {
    /// One value everywhere ([`Scope::Material`]).
    Constant(Value),
    /// A channel of the material, or its default ([`Scope::Sample`]).
    Channel(ChannelId),
    /// A map on the material's grid read texel by texel
    /// ([`Scope::Sample`]).
    Map(TypedRaster),
    /// A map a raster pass produced on the material's grid
    /// ([`Scope::Pass`]).
    Pass(TypedRaster),
    /// The texel center's domain position, a `Vector2` ([`Scope::Sample`]).
    Position,
}

impl MapBinding {
    /// The scope of the values it supplies.
    #[must_use]
    pub const fn scope(&self) -> Scope {
        match self {
            Self::Constant(_) => Scope::Material,
            Self::Channel(_) | Self::Map(_) | Self::Position => Scope::Sample,
            Self::Pass(_) => Scope::Pass,
        }
    }

    fn shape(&self) -> Shape {
        match self {
            Self::Constant(v) => Shape::of_value(*v),
            Self::Channel(c) => Shape::of(c.port()),
            Self::Map(r) | Self::Pass(r) => Shape::of(r.port()),
            Self::Position => Shape::Vector2,
        }
    }
}

/// What [`evaluate`] made: one raster per output, and the axes along which
/// the program was found periodic.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluated {
    /// One raster per output, in output order.
    pub outputs: Vec<TypedRaster>,
    /// Where the program reads [`MapBinding::Position`] on a wrapping grid,
    /// the axes along which evaluating it one period away gives the same
    /// outputs (exactly, for identifiers; within `1e-5` relative, for
    /// numbers) at every probed texel; otherwise [`Tiling::of`] the grid.
    pub tiling: Tiling,
}

/// Evaluates `program` at every texel of `material`'s grid, with one
/// binding per input, and returns one raster per output.
///
/// Programs that read the position are checked **across the wrap**: at up
/// to 64 texels spread over the grid, the program is evaluated again with
/// the position one period away along each axis, which a periodic program
/// cannot tell apart. An axis where any output differs is not periodic, and
/// [`Evaluated::tiling`] says so, so a module narrows its material's tiling
/// from what its formulas do rather than from what it remembers to
/// declare.
///
/// # Errors
///
/// [`MaterialError::Contract`] when the bindings do not fit the inputs in
/// number, shape or scope; [`MaterialError::GridMismatch`] for a map off
/// the grid.
pub fn evaluate(
    program: &ScopedProgram,
    bindings: &[MapBinding],
    material: &Material,
) -> Result<Evaluated, MaterialError> {
    let grid = material.grid();
    if bindings.len() != program.inputs().len() {
        return Err(ContractError::BindingCount.into());
    }
    for (input, binding) in program.inputs().iter().zip(bindings) {
        if !binding.scope().within(input.scope) {
            return Err(ContractError::ScopeViolation {
                name: input.name.clone(),
                declared: input.scope,
                found: binding.scope(),
            }
            .into());
        }
        if binding.shape() != Shape::of(input.port) {
            return Err(ContractError::TypeMismatch(input.name.clone()).into());
        }
        if let MapBinding::Map(r) | MapBinding::Pass(r) = binding
            && !grid.holds(r)
        {
            return Err(MaterialError::GridMismatch);
        }
    }
    let read_shifted = |i: u32, texel: usize, shift: Vec2| -> Value {
        match &bindings[i as usize] {
            MapBinding::Constant(v) => *v,
            MapBinding::Channel(c) => material.value(*c, texel),
            MapBinding::Map(r) | MapBinding::Pass(r) => r.value(texel),
            MapBinding::Position => Value::Vector2(grid.center(texel) + shift),
        }
    };
    let read = |i: u32, texel: usize| read_shifted(i, texel, Vec2::ZERO);
    let mut hoisted = vec![None; program.nodes().len()];
    program.evaluate(
        &mut hoisted,
        |s| s == Scope::Material,
        &mut |i| read(i, 0),
        Footprint::POINT,
    );
    let footprint = Footprint::new(grid.texel.max_element()).unwrap_or(Footprint::POINT);
    let outputs = program.outputs();
    let mut columns: Vec<Vec<Value>> = vec![Vec::with_capacity(grid.len()); outputs.len()];
    for texel in 0..grid.len() {
        let mut values = hoisted.clone();
        program.evaluate(&mut values, |_| true, &mut |i| read(i, texel), footprint);
        for (column, o) in columns.iter_mut().zip(outputs) {
            column.push(values[o.node.index()].expect("every node evaluated"));
        }
    }
    // Across the wrap: the same texels, one period away.
    let mut tiling = Tiling::of(grid);
    let reads_position = bindings.iter().any(|b| matches!(b, MapBinding::Position));
    if reads_position && grid.edge == dapple_field::Edge::Wrap {
        #[expect(clippy::cast_precision_loss, reason = "grid sizes are small")]
        let period = grid.texel * Vec2::new(grid.width as f32, grid.height as f32);
        let stride = (grid.len() / 64).max(1) | 1;
        for (axis, shift) in [(0, Vec2::new(period.x, 0.0)), (1, Vec2::new(0.0, period.y))] {
            let periodic = (0..grid.len()).step_by(stride).all(|texel| {
                let mut values = hoisted.clone();
                program.evaluate(
                    &mut values,
                    |_| true,
                    &mut |i| read_shifted(i, texel, shift),
                    footprint,
                );
                outputs.iter().zip(&columns).all(|(o, column)| {
                    same(
                        values[o.node.index()].expect("every node evaluated"),
                        column[texel],
                    )
                })
            });
            if !periodic {
                if axis == 0 {
                    tiling.x = false;
                } else {
                    tiling.y = false;
                }
            }
        }
    }
    let outputs = outputs
        .iter()
        .zip(columns)
        .map(|(o, column)| Ok(grid.typed(o.port, column)?))
        .collect::<Result<Vec<_>, MaterialError>>()?;
    Ok(Evaluated { outputs, tiling })
}

/// Whether two values agree: identifiers exactly, numbers within `1e-5`
/// relative.
fn same(a: Value, b: Value) -> bool {
    match (a, b) {
        (Value::Id(a), Value::Id(b)) => a == b,
        _ => (0..3).all(|k| match (a.component(k), b.component(k)) {
            (Some(x), Some(y)) => (x - y).abs() <= 1e-5 * (1.0 + x.abs().max(y.abs())),
            (None, None) => true,
            _ => false,
        }),
    }
}

/// The first output of [`evaluate`] as a scalar raster, for masks, with
/// the axes the program was found periodic along.
///
/// # Errors
///
/// As [`evaluate`], and [`MaterialError::TypeMismatch`] when the first
/// output is not a scalar or mask.
pub fn evaluate_mask(
    program: &ScopedProgram,
    bindings: &[MapBinding],
    material: &Material,
) -> Result<(dapple_raster::Raster, Tiling), MaterialError> {
    let out = evaluate(program, bindings, material)?;
    let tiling = out.tiling;
    let first = out
        .outputs
        .into_iter()
        .next()
        .ok_or(MaterialError::InvalidParameter("outputs"))?;
    match (first.port(), first.storage()) {
        (PortType::Scalar | PortType::Mask, dapple_raster::typed::Storage::F32(r)) => {
            Ok((r.clone(), tiling))
        }
        _ => Err(MaterialError::InvalidParameter("mask output")),
    }
}
