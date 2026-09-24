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

use crate::material::{ChannelId, Material, MaterialError};

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

/// Evaluates `program` at every texel of `material`'s grid, with one
/// binding per input, and returns one raster per output.
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
) -> Result<Vec<TypedRaster>, MaterialError> {
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
    let read = |i: u32, texel: usize| -> Value {
        match &bindings[i as usize] {
            MapBinding::Constant(v) => *v,
            MapBinding::Channel(c) => material.value(*c, texel),
            MapBinding::Map(r) | MapBinding::Pass(r) => r.value(texel),
            MapBinding::Position => Value::Vector2(grid.center(texel)),
        }
    };
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
    outputs
        .iter()
        .zip(columns)
        .map(|(o, column)| Ok(grid.typed(o.port, column)?))
        .collect()
}

/// The first output of [`evaluate`] as a scalar raster, for masks.
///
/// # Errors
///
/// As [`evaluate`], and [`MaterialError::TypeMismatch`] when the first
/// output is not a scalar or mask.
pub fn evaluate_mask(
    program: &ScopedProgram,
    bindings: &[MapBinding],
    material: &Material,
) -> Result<dapple_raster::Raster, MaterialError> {
    let out = evaluate(program, bindings, material)?;
    let first = out
        .into_iter()
        .next()
        .ok_or(MaterialError::InvalidParameter("outputs"))?;
    match (first.port(), first.storage()) {
        (PortType::Scalar | PortType::Mask, dapple_raster::typed::Storage::F32(r)) => Ok(r.clone()),
        _ => Err(MaterialError::InvalidParameter("mask output")),
    }
}
