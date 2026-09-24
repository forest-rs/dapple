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
//!   textures and their mips do not alias. Cellular noise fades to its mean
//!   as cells shrink below a few footprints.
//! - **Gradients.** [`ScalarField::eval_gradient`] returns a field's value and
//!   its gradient, analytic for noise, fractals, cellular noise, disks,
//!   transforms and programs built from them, numerical
//!   ([`central_difference`]) otherwise.
//!   Warps size their footprints from it.
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
//! - [`Tiling`]: tile layouts (stack and running bonds, herringbone) with
//!   exact joint distances, in-tile coordinates, orientation and per-tile
//!   values, selected as a field with [`Tiling::output`].
//! - [`Scatter`]: disks, domes or image stamps splatted at random points
//!   with bounded overlap: their union coverage, the highest stamp, and
//!   that splat's own random value.
//! - [`Disk`]: a filled, antialiased disk mask, zero outside its support.
//! - [`Transformed`]: domain-checked affine coordinate maps.
//!
//! **Solid fields.** [`Domain3`] fields ([`SolidField`]) exist through a
//! volume: [`Noise3`], [`Fractal3`], and [`Cellular3`], with the same
//! contracts. Programs mix both spaces in one IR: solid nodes reach the plane
//! only through a slice ([`program::Op::Slice`]), and a
//! [`SolidProgram`](program::SolidProgram) is evaluated at chart points by
//! [`SolidProgram::eval_chart`](program::SolidProgram::eval_chart).
//!
//! [`program`] describes fields as values: a DAG of operations with content
//! [`Fingerprint`](program::Fingerprint)s, evaluated bit-identically to the
//! field types above.
//!
//! [`scoped`] is the programmable core above field programs: a
//! [`ScopedProgram`](scoped::ScopedProgram) is an inspectable expression
//! with typed inputs and outputs whose execution [`Scope`](scoped::Scope)
//! (per material, region, element, sample or raster pass) is checked, that
//! samples field programs as resources and calls other programs as
//! functions. Elements, regions and materials bind its inputs. [`shaping`]
//! holds its tone curves and color ramps.
//!
//! [`SampleImage`] reads texels back as a field: bilinear, periodic-aware,
//! and footprint-filtered through its mip chain. Programs sample one with
//! [`program::Op::Sample`].
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

pub mod anisotropic;
mod cellular;
mod domain;
mod field;
mod fractal;
pub mod hash;
pub mod image;
mod noise;
pub mod program;
pub mod raster;
mod scatter;
pub mod scoped;
mod shape;
pub mod shaping;
mod solid;
mod tiling;
mod types;

pub use cellular::{CellOutput, CellSample, Cellular, CellularField};
pub use domain::{Domain, Domain3, DomainError, Footprint, MAX_LATTICE_CELLS};
pub use field::{Affine2, Affine3, PlaneField, ScalarField, Transformed, central_difference};
pub use fractal::{Fractal, FractalKind, FractalParams, MAX_OCTAVES};
pub use image::{Edge, ImageLevel, SampleImage, SamplePolicy};
pub use noise::{Basis, Noise};
pub use scatter::{Placement, Scatter, ScatterField, ScatterOutput, SplatPlacement, Stamp};
pub use shape::Disk;
pub use solid::{Cellular3, CellularField3, Fractal3, Noise3, SolidField, central_difference3};
pub use tiling::{MAX_HERRINGBONE_RATIO, Pattern, TileOutput, TileSample, Tiling, TilingField};
pub use types::{NormalFrame, PortType, Primaries, Value};

#[cfg(test)]
mod golden_tests;
#[cfg(test)]
mod image_tests;
