// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Reusable material modules (`dapple_material::module`).
//!
//! Material families are library modules, each with a public interface of
//! typed parameters (units, ranges, defaults), inputs and outputs, and a
//! versioned identity:
//!
//! | Module | Makes |
//! |---|---|
//! | [`CeramicBody`] | Fired-clay body: grain, iron speckle, mottling. |
//! | [`Mortar`] | Sanded lime mortar, tooled into its joints when given their distance map. |
//! | [`Stone`] | Bedded sandstone or limestone with drag-tooled faces. |
//! | [`Wood`] | Oak cut from a solid log: flat-sawn figure from real growth rings. |
//! | [`Finish`] | A glaze, varnish or oil: pigment where opaque, OpenPBR's coat on top, orange peel on the coat. |
//! | [`Grime`] | Dirt settled where ambient occlusion and cavities trap it. |
//! | [`Streaks`] | Grime washed down from upward-facing ledges. |
//! | [`Efflorescence`] | Salts blooming on one surface (mortar), rising from the foot. |
//! | [`EdgeWear`] | Wear through to the material beneath on convex edges, from curvature. |
//! | [`Moss`] | Moss where the surface faces up, is hollow or damp, with a fuzz sheen. |
//! | [`ByExample`] | A material tiled from a host-supplied exemplar by histogram-preserving blending. |
//!
//! Assets compose them without copying their graphs:
//! [`GlazedBrickWall`] (ceramic body, glaze finish, mortar, weathering),
//! [`StoneSill`] (a glazed wall, a stone sill on a mortar bed, weathering),
//! [`VarnishedBoard`] (wood, varnish finish, grime) and [`Threshold`]
//! (stone, wood and an oil finish).
//!
//! Modules build on whatever grid their context has; on a wrapping grid
//! their noise uses whole lattice cells per period, so they tile.

use alloc::vec::Vec;

use dapple_field::program::{NodeId, Op, ProgramBuilder, ProgramError};
use dapple_field::raster::Region;
use dapple_field::{Basis, Domain, Edge, FractalParams, PortType, Value};
use dapple_material::module::{
    ModuleError, ModuleErrorKind, ParamDecl, ParamKind, ParamValue, Unit,
};
use dapple_material::{Channel, Grid, MaterialError};
use dapple_raster::typed::{Storage, TypedRaster, realize_value};
use dapple_raster::{PercentileRemap, Raster, RasterOp, Realization};
use glam::{Vec2, Vec3};

mod assets;
mod ceramic;
mod finish;
mod mortar;
mod stone;
mod wear;
mod weathering;
mod wood;

pub use assets::{GlazedBrickWall, StoneSill, Threshold, VarnishedBoard};
pub use ceramic::CeramicBody;
pub use finish::Finish;
pub use mortar::Mortar;
pub use stone::Stone;
pub use wear::{ByExample, EdgeWear, Moss};
pub use weathering::{Efflorescence, Grime, Streaks};
pub use wood::Wood;

pub(crate) fn fail(what: &'static str) -> ModuleError {
    ModuleError::body(ModuleErrorKind::Body(what))
}

pub(crate) fn program_error(_: ProgramError) -> ModuleError {
    fail("a field program could not be built")
}

/// The domain a grid realizes: one period of a periodic domain for a
/// wrapping grid, the plane otherwise.
pub(crate) fn domain_of(grid: Grid) -> Domain {
    if grid.edge == Edge::Wrap {
        #[expect(clippy::cast_precision_loss, reason = "grid sizes are small")]
        let extent = grid.texel * Vec2::new(grid.width as f32, grid.height as f32);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "periods are small positive whole numbers"
        )]
        let period = |v: f32| (libm::roundf(v) as u32).max(1);
        Domain::periodic(period(extent.x), period(extent.y)).unwrap_or(Domain::Plane)
    } else {
        Domain::Plane
    }
}

/// Cells per unit that fit the domain: whole cells per period on a
/// periodic domain.
pub(crate) fn fit(domain: Domain, frequency: [f32; 2]) -> [f32; 2] {
    match domain {
        Domain::Periodic { .. } => frequency.map(|f| libm::roundf(f).max(1.0)),
        Domain::Plane => frequency,
    }
}

/// Realizes the value program `build` makes over `grid`'s domain.
pub(crate) fn realize(
    grid: Grid,
    build: impl FnOnce(&mut ProgramBuilder, Domain) -> Result<NodeId, ProgramError>,
) -> Result<TypedRaster, ModuleError> {
    let domain = domain_of(grid);
    let mut b = ProgramBuilder::new();
    let out = build(&mut b, domain).map_err(program_error)?;
    let program = b.finish_value(out).map_err(program_error)?;
    let realization = if program.domain() == Domain::Plane {
        #[expect(clippy::cast_precision_loss, reason = "grid sizes are small")]
        let size = grid.texel * Vec2::new(grid.width as f32, grid.height as f32);
        Realization::region(
            Region {
                origin: grid.origin,
                size,
            },
            grid.width,
            grid.height,
        )?
    } else {
        Realization::period(program.domain(), grid.width, grid.height)?
    };
    let raster = realize_value(&program, realization)?;
    // Keep the grid's own placement and edge policy.
    let values: Vec<Value> = (0..raster.len()).map(|i| raster.value(i)).collect();
    Ok(grid.typed(raster.port(), values)?)
}

/// As [`realize`], for a scalar program.
pub(crate) fn realize_scalar(
    grid: Grid,
    build: impl FnOnce(&mut ProgramBuilder, Domain) -> Result<NodeId, ProgramError>,
) -> Result<Raster, ModuleError> {
    scalar_of(&realize(grid, build)?)
}

/// A scalar typed raster's texels.
pub(crate) fn scalar_of(r: &TypedRaster) -> Result<Raster, ModuleError> {
    match r.storage() {
        Storage::F32(r) => Ok(r.clone()),
        _ => Err(fail("expected a scalar map")),
    }
}

/// Gradient fBm in `[-1, 1]`-ish over `domain`.
pub(crate) fn fbm(
    b: &mut ProgramBuilder,
    domain: Domain,
    frequency: [f32; 2],
    seed: u64,
    octaves: u8,
) -> Result<NodeId, ProgramError> {
    b.add(Op::Fractal {
        basis: Basis::Gradient,
        domain,
        frequency: fit(domain, frequency),
        seed,
        params: FractalParams {
            octaves,
            ..FractalParams::default()
        },
    })
}

/// A mask covering about `fraction` of the texels where `score` is
/// highest, with a soft edge `softness` wide in percentile: authoring by
/// coverage rather than by raw threshold.
pub(crate) fn coverage(
    score: &Raster,
    fraction: f32,
    softness: f32,
) -> Result<Raster, ModuleError> {
    let f = fraction.clamp(0.0, 1.0);
    let lo = (1.0 - f - softness).clamp(0.0, 1.0);
    let hi = (1.0 - f + softness).clamp(lo, 1.0);
    Ok(PercentileRemap {
        from: [lo, hi],
        to: [0.0, 1.0],
        clamp: true,
    }
    .apply(score)
    .map_err(MaterialError::Raster)?)
}

/// A mask as a typed raster.
pub(crate) fn mask_map(grid: Grid, values: &Raster) -> Result<TypedRaster, ModuleError> {
    Ok(grid.typed(
        PortType::Mask,
        values.values().iter().map(|&v| Value::Scalar(v)),
    )?)
}

/// A scalar as a typed raster.
pub(crate) fn scalar_map(grid: Grid, values: &Raster) -> Result<TypedRaster, ModuleError> {
    Ok(grid.typed(
        PortType::Scalar,
        values.values().iter().map(|&v| Value::Scalar(v)),
    )?)
}

/// A color map from linear colors.
pub(crate) fn color_map(grid: Grid, values: &[Vec3]) -> Result<Channel, ModuleError> {
    Ok(Channel::Map(grid.typed(
        PortType::Color(dapple_field::Primaries::Rec709),
        values.iter().map(|&v| Value::Vector3(v)),
    )?))
}

/// A scalar channel from values.
pub(crate) fn scalar_channel(grid: Grid, values: &[f32]) -> Result<Channel, ModuleError> {
    Ok(Channel::Map(grid.typed(
        PortType::Scalar,
        values.iter().map(|&v| Value::Scalar(v)),
    )?))
}

pub(crate) fn scalar(
    name: &'static str,
    unit: Unit,
    range: [f32; 2],
    default: f32,
    doc: &'static str,
) -> ParamDecl {
    ParamDecl {
        name,
        kind: ParamKind::Scalar { unit, range },
        default: ParamValue::Scalar(default),
        doc,
    }
}

pub(crate) fn meters(
    name: &'static str,
    range: [f32; 2],
    default: f32,
    doc: &'static str,
) -> ParamDecl {
    scalar(name, Unit::Meters, range, default, doc)
}

pub(crate) fn fraction(name: &'static str, default: f32, doc: &'static str) -> ParamDecl {
    scalar(name, Unit::Fraction, [0.0, 1.0], default, doc)
}

pub(crate) fn color(name: &'static str, default: Vec3, doc: &'static str) -> ParamDecl {
    ParamDecl {
        name,
        kind: ParamKind::Color,
        default: ParamValue::Color(default),
        doc,
    }
}

pub(crate) fn integer(
    name: &'static str,
    range: [u32; 2],
    default: u32,
    doc: &'static str,
) -> ParamDecl {
    ParamDecl {
        name,
        kind: ParamKind::Integer { range },
        default: ParamValue::Integer(default),
        doc,
    }
}

pub(crate) fn flag(name: &'static str, default: bool, doc: &'static str) -> ParamDecl {
    ParamDecl {
        name,
        kind: ParamKind::Flag,
        default: ParamValue::Flag(default),
        doc,
    }
}

pub(crate) const fn seed() -> ParamDecl {
    ParamDecl {
        name: "seed",
        kind: ParamKind::Seed,
        default: ParamValue::Seed(0),
        doc: "varies the instance; mixed with its path",
    }
}

pub(crate) fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
