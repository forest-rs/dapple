// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Typed material values for dapple, the operations that combine them, and
//! parameterized modules that build them.
//!
//! - **Material values** ([`Material`]): every OpenPBR parameter (the
//!   `openpbr` crate's vocabulary) bound to a constant or a typed map, with
//!   auxiliary channels ([`Aux`]: height, occlusion, region and surface
//!   identity) kept apart, all on one [`Grid`].
//! - **Operations** ([`ops`]), each with its own contract rather than one
//!   channel-wise blend under different names: [`ops::select`] (one shared
//!   decision), [`ops::apply_detail`] (height and normal perturbation in a
//!   stated layer; reoriented normal mapping lives only here),
//!   [`ops::coat`] (OpenPBR's coat over an untouched base),
//!   [`ops::deposit`] (a covering that changes coverage, height, identity
//!   and optics together), and [`ops::transform`] (every channel moves).
//!   Each returns a [`Report`] of what it approximated, where it happened.
//! - **Programs over materials** ([`program`]): `dapple_field` scoped
//!   programs evaluated per texel with inputs bound to channels, pass
//!   results and positions, scope-checked.
//! - **Modules** ([`module`]): typed public interfaces with units, ranges
//!   and defaults, material, map and host-resolved resource inputs
//!   ([`resource`]), named outputs and versioned identity; instances derive
//!   seeds from their path and keep their boundary in diagnostics.
//! - **Lowering** ([`lower::maps`]) to `dapple_encode`'s maps, naming what
//!   the maps cannot carry.
//!
//! Colors are linear Rec. 709, dapple's working space. Heights and domain
//! units are meters.

#![no_std]

extern crate alloc;

pub mod lower;
mod material;
pub mod module;
pub mod ops;
pub mod program;
mod report;
pub mod resource;

pub use material::{
    Aux, Channel, ChannelId, Grid, Material, MaterialError, param_default, param_port,
};
pub use openpbr;
pub use openpbr::Param;
pub use report::{Approximation, ApproximationKind, Report};

#[cfg(test)]
mod tests;
