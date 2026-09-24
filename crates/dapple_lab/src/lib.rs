// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A headless material laboratory for dapple.
//!
//! - **Reports** ([`report`]): named measurements with units and bounds,
//!   and named checks, written as deterministic JSON. The model knows
//!   nothing of materials, so any harness can use it: a tree generator's
//!   reference renders as well as a material bake.
//! - **Measurements** ([`measure`]): value statistics, and a
//!   resolution-independent feature size.
//! - **Material reports and relationship checks** ([`material`]): ranges,
//!   invalid values, seams against the material's tiling promise, what
//!   lowering drops; a material-wide transform leaves no channel behind;
//!   raising resolution keeps physical feature size; an incremental result
//!   agrees with a clean one.
//! - **Previews** ([`preview`]): tiled, raking-light (grazing) and mip
//!   views, and contact sheets for parameter sweeps, as software renders;
//!   PNG output behind the `std` feature.

#![no_std]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod material;
pub mod measure;
pub mod preview;
pub mod report;

pub use report::{Check, Entry, Measurement, Report};
