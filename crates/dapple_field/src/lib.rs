// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Deterministic, band-limited procedural fields.
//!
//! `dapple_field` is the point-evaluable half of dapple's material engine. A
//! [`ScalarField`] maps a point of its [`Domain`] and a [`Footprint`] to a value:
//!
//! - **Domains are types.** [`Domain::Periodic`] fields tile by construction:
//!   lattices must fit a whole number of cells into the period, and transforms
//!   must map the period lattice onto itself. Violations are
//!   [`DomainError`]s at construction, not seams found later. [`PlaneField`]
//!   demotes a periodic field explicitly.
//! - **Footprints band-limit.** A field evaluated for a region drops detail
//!   above that region's Nyquist limit, fading it to its mean, so realized
//!   textures and their mips do not alias. Cellular noise ignores footprints
//!   for now.
//! - **Results are bit-exact.** All randomness is keyed hashing ([`hash`]), with
//!   no sequential streams. The math is plain IEEE `f32` add, multiply, and
//!   divide, `floor` and `sqrt` from the pure-Rust `libm`, and no fused
//!   multiply-add, so equal inputs give equal bits on every platform.
//!
//! The fields:
//!
//! - [`Noise`]: one octave of value or gradient lattice noise.
//! - [`Fractal`]: fBm or ridged sums of octaves, with integer lacunarity so
//!   periodic fractals tile.
//! - [`Cellular`]: Worley F1, F2, exact cell-border distance, and per-cell IDs
//!   and values, selected as a field with [`Cellular::output`].
//! - [`Transformed`]: domain-checked affine coordinate maps.
//!
//! [`raster::Grid`] samples a field onto a small grid for tests and previews.
//! Tiled, cached, and incremental realization belongs to later dapple crates.
//!
//! ```
//! use dapple_field::raster::{Grid, Region};
//! use dapple_field::{Basis, Domain, Fractal, FractalParams};
//! use glam::Vec2;
//!
//! let domain = Domain::periodic(1, 1).unwrap();
//! let fbm = Fractal::new(Basis::Gradient, domain, Vec2::splat(4.0), 42, FractalParams::default())?;
//! let grid = Grid::sample(&fbm, Region::period(domain).unwrap(), 64, 64);
//! assert_eq!(grid.values.len(), 64 * 64);
//! # Ok::<(), dapple_field::DomainError>(())
//! ```

#![no_std]

extern crate alloc;

mod cellular;
mod domain;
mod field;
mod fractal;
pub mod hash;
mod noise;
pub mod raster;

pub use cellular::{CellOutput, CellSample, Cellular, CellularField};
pub use domain::{Domain, DomainError, Footprint, MAX_LATTICE_CELLS};
pub use field::{Affine2, PlaneField, ScalarField, Transformed};
pub use fractal::{Fractal, FractalKind, FractalParams, MAX_OCTAVES};
pub use noise::{Basis, Noise};

#[cfg(test)]
mod golden_tests;
