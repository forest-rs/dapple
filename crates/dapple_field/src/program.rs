// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Field programs: fields as inspectable, fingerprinted values.
//!
//! A [`FieldProgram`] is a DAG of [`Op`]s built with a [`ProgramBuilder`]. Each
//! node declares its operation and parameters as plain data, so a program can
//! be listed, compared, diffed and cached; the Rust field types
//! ([`Noise`], [`Fractal`], [`Cellular`], [`Transformed`](crate::Transformed))
//! remain the evaluators behind it. Evaluating a program is bit-identical to
//! evaluating the same fields directly.
//!
//! **Typed ports.** Every node has a [`PortType`], derived when it is added:
//! scalars, masks, identifiers, vectors, linear colors with declared
//! primaries, normals in a declared frame, and undirected directions. Type
//! rules reject misuse at build time: normals never blend here (detail
//! composition belongs to `dapple_material`'s detail application, and only
//! there); directions only blend, sign-free,
//! through their doubled-angle vectors; identifiers never blend; masks stay masks only through operations
//! that keep them in `[0, 1]`; and a transform that rotates or scales cannot
//! move a vector-valued field, whose values it would leave unrotated.
//! [`ProgramBuilder::finish`] makes a scalar [`FieldProgram`], which is a
//! [`ScalarField`]; [`ProgramBuilder::finish_value`] makes a
//! [`ValueProgram`] of any type, evaluated to a [`Value`].
//!
//! Every node has a [`Fingerprint`]: a hash of its operation, parameters and
//! the fingerprints of its inputs, so equal subgraphs have equal fingerprints
//! wherever they appear and unused nodes never affect a result. See
//! [`Fingerprint`] for the encoding, which is stable across runs, platforms
//! and versions of this crate unless the encoding version changes.
//!
//! ```
//! use dapple_field::program::{Op, ProgramBuilder};
//! use dapple_field::{Basis, Domain, Footprint, Noise, ScalarField};
//! use glam::Vec2;
//!
//! let domain = Domain::periodic(1, 1).unwrap();
//! let mut b = ProgramBuilder::new();
//! let noise = b.add(Op::Noise { basis: Basis::Gradient, domain, frequency: [8.0, 8.0], seed: 7 })?;
//! let half = b.add(Op::Constant { domain, value: 0.5 })?;
//! let scaled = b.add(Op::Mul { a: noise, b: half })?;
//! let program = b.finish(scaled)?;
//!
//! let direct = Noise::new(Basis::Gradient, domain, Vec2::splat(8.0), 7)?;
//! let p = Vec2::new(0.3, 0.7);
//! assert_eq!(
//!     program.eval(p, Footprint::POINT).to_bits(),
//!     (direct.eval(p, Footprint::POINT) * 0.5).to_bits(),
//! );
//! # Ok::<(), Box<dyn core::error::Error>>(())
//! ```

use alloc::vec::Vec;
use core::fmt;

use glam::{Mat2, Mat3, Vec2, Vec3};

use crate::cellular::{CellOutput, Cellular, CellularField};
use crate::domain::{Domain, Domain3, DomainError, Footprint};
use crate::field::{Affine2, Affine3, ScalarField, check_transform, check_transform3};
use crate::fractal::{Fractal, FractalKind, FractalParams};
use crate::hash::hash;
use crate::image::{SampleImage, SamplePolicy};
use crate::noise::{Basis, Noise};
use crate::raster::Region;
use crate::scatter::{Placement, Scatter, ScatterField, ScatterOutput, Stamp};
use crate::shape::Disk;
use crate::solid::{Cellular3, CellularField3, Fractal3, Noise3, SolidField};
use crate::tiling::{Pattern, TileOutput, Tiling, TilingField};
use crate::types::{NormalFrame, PortType, Primaries, Value};

mod bounds;
mod flat;
#[cfg(test)]
mod solid_tests;

pub use bounds::StaticBounds;

pub use flat::{EvaluationStats, Evaluator};

/// Version of the fingerprint encoding. Changing any word the encoding emits
/// requires a new version, so persisted fingerprints never collide.
///
/// Version 2 added an [`Op::Sample`] image's [`SamplePolicy`] word.
pub const FINGERPRINT_VERSION: u64 = 2;

/// Where a program node is defined: over a planar or a solid domain.
///
/// One program IR describes both. Leaves state their space; operations that
/// combine inputs require them to share one, and [`Op::Slice`] is the only
/// way from a solid field to a planar one.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Space {
    /// A planar node over a [`Domain`], evaluated at 2D points.
    Planar(Domain),
    /// A solid node over a [`Domain3`], evaluated at 3D points.
    Solid(Domain3),
}

impl Space {
    /// The planar domain, if planar.
    #[must_use]
    pub const fn planar(self) -> Option<Domain> {
        match self {
            Self::Planar(domain) => Some(domain),
            Self::Solid(_) => None,
        }
    }

    /// The solid domain, if solid.
    #[must_use]
    pub const fn solid(self) -> Option<Domain3> {
        match self {
            Self::Planar(_) => None,
            Self::Solid(domain) => Some(domain),
        }
    }
}

/// A node in a [`FieldProgram`] or [`ProgramBuilder`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct NodeId(u32);

impl NodeId {
    /// The node at position `index` in creation order.
    ///
    /// Useful for writing an [`Op`] whose operands are placeholders, such as
    /// operand positions that [`Op::map_inputs`] later rewrites.
    #[must_use]
    pub const fn from_index(index: u32) -> Self {
        Self(index)
    }

    /// The node's position in creation order.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// One field operation and its parameters.
///
/// Sources name their domain; every other node derives its domain from its
/// inputs. Nodes combining several inputs require equal domains: demote
/// explicitly with [`Op::Demote`] to combine a periodic field with a plane one.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "op", rename_all = "snake_case"))]
pub enum Op {
    /// A constant value.
    Constant {
        /// Domain the constant is defined over.
        domain: Domain,
        /// The value; must be finite.
        value: f32,
    },
    /// One octave of lattice noise ([`Noise`]).
    Noise {
        /// Noise flavor.
        basis: Basis,
        /// Domain.
        domain: Domain,
        /// Lattice cells per domain unit, per axis.
        frequency: [f32; 2],
        /// Seed.
        seed: u64,
    },
    /// A fractal sum of lattice noise ([`Fractal`]).
    Fractal {
        /// Noise flavor.
        basis: Basis,
        /// Domain.
        domain: Domain,
        /// Base lattice cells per domain unit, per axis.
        frequency: [f32; 2],
        /// Seed.
        seed: u64,
        /// Octave structure.
        params: FractalParams,
    },
    /// One quantity of cellular noise ([`Cellular`]).
    Cellular {
        /// Domain.
        domain: Domain,
        /// Lattice cells per domain unit, per axis.
        frequency: [f32; 2],
        /// Feature-point jitter in `[0, 1]`.
        jitter: f32,
        /// Seed.
        seed: u64,
        /// The quantity returned.
        output: CellOutput,
    },
    /// One quantity of a tile layout ([`Tiling`]): bonds and herringbone.
    Tiling {
        /// Domain.
        domain: Domain,
        /// The layout.
        pattern: Pattern,
        /// Seed for per-tile values.
        seed: u64,
        /// The quantity returned.
        output: TileOutput,
    },
    /// One quantity of stamps scattered at random points ([`Scatter`]).
    Scatter {
        /// Domain.
        domain: Domain,
        /// Where splats go and how large they are.
        placement: Placement,
        /// What each splat stamps.
        stamp: Stamp,
        /// Seed.
        seed: u64,
        /// The quantity returned.
        output: ScatterOutput,
    },
    /// A filled [`Disk`] mask, zero outside its support.
    Disk {
        /// Domain.
        domain: Domain,
        /// Center, in domain units.
        center: [f32; 2],
        /// Radius, in domain units.
        radius: f32,
        /// Edge band width at a point footprint, in domain units.
        softness: f32,
    },
    /// `input(transform(p))`, lattice-checked like [`Transformed`](crate::Transformed).
    Transform {
        /// The transformed field.
        input: NodeId,
        /// The coordinate map.
        transform: Affine2,
    },
    /// The input, demoted to [`Domain::Plane`] (like [`PlaneField`](crate::PlaneField)).
    Demote {
        /// The demoted field.
        input: NodeId,
    },
    /// `a + b`.
    Add {
        /// Left operand.
        a: NodeId,
        /// Right operand.
        b: NodeId,
    },
    /// `a - b`.
    Sub {
        /// Left operand.
        a: NodeId,
        /// Right operand.
        b: NodeId,
    },
    /// `a * b`.
    Mul {
        /// Left operand.
        a: NodeId,
        /// Right operand.
        b: NodeId,
    },
    /// `min(a, b)`.
    Min {
        /// Left operand.
        a: NodeId,
        /// Right operand.
        b: NodeId,
    },
    /// `max(a, b)`.
    Max {
        /// Left operand.
        a: NodeId,
        /// Right operand.
        b: NodeId,
    },
    /// `|input|`.
    Abs {
        /// Operand.
        input: NodeId,
    },
    /// `input` clamped to `[min, max]`.
    Clamp {
        /// Operand.
        input: NodeId,
        /// Lower bound; finite and at most `max`.
        min: f32,
        /// Upper bound; finite.
        max: f32,
    },
    /// Linear remap taking `from[0]` to `to[0]` and `from[1]` to `to[1]`,
    /// without clamping.
    Remap {
        /// Operand.
        input: NodeId,
        /// Source range; finite with distinct ends.
        from: [f32; 2],
        /// Target range; finite.
        to: [f32; 2],
    },
    /// `a + (b - a) * t`.
    Mix {
        /// Value where `t` is 0.
        a: NodeId,
        /// Value where `t` is 1.
        b: NodeId,
        /// Blend weight.
        t: NodeId,
    },
    /// Domain warp: `input(p + amount * (dx(p), dy(p)))`.
    ///
    /// All three fields share a domain, so a periodic warp of a periodic
    /// field stays periodic.
    ///
    /// **Footprint.** Where the warp stretches or compresses space, one
    /// footprint of `p` covers more of `input`. The input's footprint is
    /// scaled by `1 + |amount| · max_i Σ_j |∂d_i/∂p_j|`, a bound on the warp's
    /// local stretch, with the derivatives of `dx` and `dy` from their
    /// gradients ([`FieldProgram::eval_node_gradient`]): analytic through
    /// every op with a closed form, band-limited like the displacements. A
    /// point footprint skips the scaling.
    Warp {
        /// The warped field.
        input: NodeId,
        /// Displacement along x, per unit of `amount`.
        dx: NodeId,
        /// Displacement along y, per unit of `amount`.
        dy: NodeId,
        /// Displacement scale in domain units; finite.
        amount: f32,
    },
    /// A [`PortType::Vector2`] from two scalars.
    Vector2 {
        /// First component.
        x: NodeId,
        /// Second component.
        y: NodeId,
    },
    /// A [`PortType::Vector3`] from three scalars.
    Vector3 {
        /// First component.
        x: NodeId,
        /// Second component.
        y: NodeId,
        /// Third component.
        z: NodeId,
    },
    /// A linear [`PortType::Color`] with Rec. 709 primaries from three
    /// scalars.
    Color {
        /// Red.
        r: NodeId,
        /// Green.
        g: NodeId,
        /// Blue.
        b: NodeId,
    },
    /// One component of a vector, color or normal, as a scalar.
    Component {
        /// The vector-valued input.
        input: NodeId,
        /// Component index: 0 or 1 for a `Vector2`, 0 to 2 otherwise.
        index: u8,
    },
    /// A scalar clamped to `[0, 1]`, as a [`PortType::Mask`].
    AsMask {
        /// Operand.
        input: NodeId,
    },
    /// A scalar in `[0, 1]` quantized to `levels` identifiers:
    /// `floor(clamp(v, 0, 1) * levels)`, at most `levels - 1`.
    ToId {
        /// Operand.
        input: NodeId,
        /// Number of identifiers; at least 1.
        levels: u32,
    },
    /// A `Vector3` normalized into a [`PortType::Normal`] in the domain frame;
    /// a zero vector gives `(0, 0, 1)`.
    Normalize {
        /// The vector.
        input: NodeId,
    },
    /// A [`PortType::Direction`] at angle `θ` (radians, from the domain's x
    /// axis toward its y axis), stored as `(cos 2θ, sin 2θ)`.
    Direction {
        /// The angle; any real value, taken modulo π.
        angle: NodeId,
    },
    /// The angle of a direction in `[0, π)`, from its doubled-angle vector;
    /// 0 where the vector vanishes.
    Angle {
        /// The direction.
        input: NodeId,
    },
    /// How much a direction's samples agree: the length of its doubled-angle
    /// vector, 1 for a single direction and 0 where opposite axes cancel. A
    /// [`PortType::Mask`].
    Coherence {
        /// The direction.
        input: NodeId,
    },
    /// Texels read back as a field ([`SampleImage`]): bilinear, wrapping on a
    /// periodic domain, filtered through the image's mips by the footprint.
    ///
    /// The image carries its texels, so programs holding one are values, not
    /// descriptions: the op never serializes, and its fingerprint is the
    /// image's [derivation](SampleImage::derivation), not a hash of the
    /// texels. A material graph samples its rasters through sample nodes,
    /// which recipes name by label.
    #[cfg_attr(feature = "serde", serde(skip))]
    Sample {
        /// The image and its domain.
        image: SampleImage,
    },
    /// A constant value over a solid domain.
    Constant3 {
        /// Domain the constant is defined over.
        domain: Domain3,
        /// The value; must be finite.
        value: f32,
    },
    /// One octave of solid lattice noise ([`Noise3`]).
    Noise3 {
        /// Noise flavor.
        basis: Basis,
        /// Domain.
        domain: Domain3,
        /// Lattice cells per domain unit, per axis.
        frequency: [f32; 3],
        /// Seed.
        seed: u64,
    },
    /// A fractal sum of solid lattice noise ([`Fractal3`]).
    Fractal3 {
        /// Noise flavor.
        basis: Basis,
        /// Domain.
        domain: Domain3,
        /// Base lattice cells per domain unit, per axis.
        frequency: [f32; 3],
        /// Seed.
        seed: u64,
        /// Octave structure.
        params: FractalParams,
    },
    /// One quantity of solid cellular noise ([`Cellular3`]).
    Cellular3 {
        /// Domain.
        domain: Domain3,
        /// Lattice cells per domain unit, per axis.
        frequency: [f32; 3],
        /// Feature-point jitter in `[0, 1]`.
        jitter: f32,
        /// Seed.
        seed: u64,
        /// The quantity returned.
        output: CellOutput,
    },
    /// The evaluation point of [`Domain3::Space`] as a [`PortType::Vector3`].
    ///
    /// Positions do not repeat, so there is no periodic form. With
    /// [`Op::Length`] and [`Op::Component`] it gives radial and axial
    /// coordinates, such as the distance from a trunk's axis.
    Position3,
    /// `input(transform(p))` over a solid domain, lattice-checked like
    /// [`Op::Transform`].
    Transform3 {
        /// The transformed field.
        input: NodeId,
        /// The coordinate map.
        transform: Affine3,
    },
    /// A planar field from a solid one: `input(origin + x·u + y·v)`.
    ///
    /// `domain` is the planar result's domain. [`Domain::Plane`] is always
    /// allowed. [`Domain::Periodic`] needs a periodic input whose lattice
    /// contains `u · px` and `v · py`, so the slice tiles exactly. A footprint
    /// `w` wide covers the parallelogram spanned by `w·u` and `w·v`, so the
    /// input sees it scaled by the larger singular value of `[u v]`.
    Slice {
        /// The solid field.
        input: NodeId,
        /// The solid point at the plane's origin.
        origin: [f32; 3],
        /// The solid step per unit of the plane's x axis.
        u: [f32; 3],
        /// The solid step per unit of the plane's y axis.
        v: [f32; 3],
        /// The planar result's domain.
        domain: Domain,
    },
    /// The Euclidean length of a [`PortType::Vector2`] or [`PortType::Vector3`].
    Length {
        /// The vector.
        input: NodeId,
    },
    /// The fractional part `x − ⌊x⌋`, a sawtooth in `[0, 1)`.
    ///
    /// Its jumps are not band-limited: feed it slowly varying inputs, or
    /// shape its output with ops that are smooth across the jump.
    Fract {
        /// Operand.
        input: NodeId,
    },
    /// The angle `atan2(y, x)` of the point `(x, y)`, in `[−π, π]`; 0 at the
    /// origin.
    ///
    /// With [`Op::Component`]s of [`Op::Position3`] it gives the angle
    /// around an axis, such as a trunk's. The angle jumps by 2π across the
    /// negative x half-axis; `fract(angle · n / 2π)` for an integer `n` is
    /// continuous there.
    Atan2 {
        /// The point's second coordinate.
        y: NodeId,
        /// The point's first coordinate.
        x: NodeId,
    },
}

/// Where replacing one node's operation, or an input's value, can change a
/// node's value.
///
/// Regions are stated at a point footprint. Evaluating with a footprint `w`
/// wide grows each region by `footprint_scale · w / 2` on every side:
/// `footprint_scale` is 1 unless a warp downstream of the change widens the
/// footprint its input sees (see [`Op::Warp`]). On a periodic domain the
/// regions also repeat with the period.
#[derive(Clone, Debug, PartialEq)]
pub enum Change {
    /// Nowhere: the value is unchanged.
    Nowhere,
    /// Only inside these regions.
    Within {
        /// The regions, at a point footprint.
        regions: Vec<Region>,
        /// How many half footprints each region grows by at a footprint;
        /// at least 1.
        footprint_scale: f32,
    },
    /// Possibly anywhere.
    Everywhere,
}

impl Change {
    /// Largest number of regions kept before they merge into their bounds.
    pub const MAX_REGIONS: usize = 16;

    /// A change within `regions`, growing by half a footprint.
    #[must_use]
    pub fn within(regions: Vec<Region>) -> Self {
        Self::Within {
            regions,
            footprint_scale: 1.0,
        }
    }

    /// The change a warp makes of its input's change: a warp reading its
    /// input at most `reach` away per axis, with its input's footprint
    /// widened at most `stretch` times.
    ///
    /// Each region grows by `reach`, and its footprint growth by `stretch`,
    /// since an output point reads the input `reach` away at a footprint up
    /// to `stretch` times its own. `Everywhere` when either is not finite.
    #[must_use]
    pub fn warped(self, reach: Vec2, stretch: f32) -> Self {
        match self {
            Self::Within {
                regions,
                footprint_scale,
            } if reach.is_finite() && stretch.is_finite() && stretch >= 1.0 => {
                let scale = footprint_scale * stretch;
                if !scale.is_finite() {
                    return Self::Everywhere;
                }
                Self::Within {
                    regions: regions
                        .into_iter()
                        .map(|r| Region {
                            origin: r.origin - reach,
                            size: r.size + reach * 2.0,
                        })
                        .collect(),
                    footprint_scale: scale,
                }
            }
            Self::Within { .. } => Self::Everywhere,
            change => change,
        }
    }

    /// The union of `self` and `other`.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        match (self, other) {
            (Self::Everywhere, _) | (_, Self::Everywhere) => Self::Everywhere,
            (Self::Nowhere, change) | (change, Self::Nowhere) => change,
            (
                Self::Within {
                    regions: mut a,
                    footprint_scale: sa,
                },
                Self::Within {
                    regions: b,
                    footprint_scale: sb,
                },
            ) => {
                a.extend(b);
                if a.len() > Self::MAX_REGIONS {
                    let min = a.iter().fold(Vec2::INFINITY, |m, r| m.min(r.origin));
                    let max = a
                        .iter()
                        .fold(Vec2::NEG_INFINITY, |m, r| m.max(r.origin + r.size));
                    a = alloc::vec![Region {
                        origin: min,
                        size: max - min,
                    }];
                }
                Self::Within {
                    regions: a,
                    footprint_scale: sa.max(sb),
                }
            }
        }
    }
}

impl Op {
    /// Where replacing `previous` with `self` can change this node's value,
    /// with its inputs unchanged.
    ///
    /// Equal operations change nothing. A [`Op::Disk`] replaced by a disk on
    /// the same domain changes only the two disks' supports. Any other edit
    /// may change the value anywhere.
    #[must_use]
    pub fn change_from(&self, previous: &Self) -> Change {
        if self == previous {
            return Change::Nowhere;
        }
        match (self, previous) {
            (
                Self::Disk {
                    domain,
                    center,
                    radius,
                    softness,
                },
                Self::Disk {
                    domain: d0,
                    center: c0,
                    radius: r0,
                    softness: s0,
                },
            ) if domain == d0 => {
                match (
                    Disk::new(*domain, Vec2::from(*center), *radius, *softness),
                    Disk::new(*d0, Vec2::from(*c0), *r0, *s0),
                ) {
                    (Ok(a), Ok(b)) => Change::within(alloc::vec![b.support(), a.support()]),
                    _ => Change::Everywhere,
                }
            }
            _ => Change::Everywhere,
        }
    }

    /// The [`PortType`] a node computing this op has when its operands have
    /// the types `port` gives, by the rules [`ProgramBuilder::add`] applies.
    ///
    /// Only types are checked: domains, spaces and parameters are checked
    /// when a program is built. A material graph uses this to refuse
    /// mistyped nodes when they are added rather than when they run.
    ///
    /// # Errors
    ///
    /// [`ProgramError::TypeMismatch`] when an operand's type does not fit.
    pub fn port_type_with(
        &self,
        port: impl Fn(NodeId) -> PortType,
    ) -> Result<PortType, ProgramError> {
        derive_port(self, &port)
    }

    /// This op's [`Fingerprint`] with `inputs`, the fingerprints of its
    /// operands in operand order: the fingerprint a program node computing
    /// this op over those inputs has.
    #[must_use]
    pub fn fingerprint_with(&self, inputs: &[Fingerprint]) -> Fingerprint {
        let op = self;
        let mut words = Vec::with_capacity(16);
        words.push(FINGERPRINT_VERSION);
        words.push(op_tag(op));
        let domain = |words: &mut Vec<u64>, domain: Domain| match domain {
            Domain::Plane => words.push(0),
            Domain::Periodic { period } => {
                words.extend([1, u64::from(period[0]), u64::from(period[1])]);
            }
        };
        let domain3 = |words: &mut Vec<u64>, domain: Domain3| match domain {
            Domain3::Space => words.push(0),
            Domain3::Periodic3 { period } => {
                words.extend([
                    1,
                    u64::from(period[0]),
                    u64::from(period[1]),
                    u64::from(period[2]),
                ]);
            }
        };
        let float = |v: f32| u64::from(v.to_bits());
        match *op {
            Self::Constant { domain: d, value } => {
                domain(&mut words, d);
                words.push(float(value));
            }
            Self::Noise {
                basis,
                domain: d,
                frequency,
                seed,
            } => {
                words.push(basis_tag(basis));
                domain(&mut words, d);
                words.extend([float(frequency[0]), float(frequency[1]), seed]);
            }
            Self::Fractal {
                basis,
                domain: d,
                frequency,
                seed,
                params,
            } => {
                words.push(basis_tag(basis));
                domain(&mut words, d);
                words.extend([float(frequency[0]), float(frequency[1]), seed]);
                words.extend([
                    match params.kind {
                        FractalKind::Fbm => 0,
                        FractalKind::Ridged => 1,
                    },
                    u64::from(params.octaves),
                    u64::from(params.lacunarity),
                    float(params.gain),
                ]);
            }
            Self::Cellular {
                domain: d,
                frequency,
                jitter,
                seed,
                output,
            } => {
                domain(&mut words, d);
                words.extend([
                    float(frequency[0]),
                    float(frequency[1]),
                    float(jitter),
                    seed,
                    cell_output_tag(output),
                ]);
            }
            Self::Tiling {
                domain: d,
                pattern,
                seed,
                output,
            } => {
                domain(&mut words, d);
                match pattern {
                    Pattern::Bond { frequency, shift } => {
                        words.extend([0, float(frequency[0]), float(frequency[1]), float(shift)]);
                    }
                    Pattern::Herringbone { frequency, ratio } => {
                        words.extend([1, float(frequency), u64::from(ratio)]);
                    }
                }
                words.extend([seed, tile_output_tag(output)]);
            }
            Self::Scatter {
                domain: d,
                placement,
                ref stamp,
                seed,
                output,
            } => {
                domain(&mut words, d);
                words.extend([
                    float(placement.frequency),
                    float(placement.density),
                    float(placement.radius[0]),
                    float(placement.radius[1]),
                    u64::from(placement.rotate),
                ]);
                match stamp {
                    Stamp::Disk { softness } => words.extend([0, float(*softness)]),
                    Stamp::Dome => words.push(1),
                    Stamp::Image(image) => {
                        let [lo, hi] = fingerprint_halves(image.derivation());
                        words.extend([2, lo, hi]);
                    }
                }
                words.extend([seed, scatter_output_tag(output)]);
            }
            Self::Disk {
                domain: d,
                center,
                radius,
                softness,
            } => {
                domain(&mut words, d);
                words.extend([
                    float(center[0]),
                    float(center[1]),
                    float(radius),
                    float(softness),
                ]);
            }
            Self::Transform { transform, .. } => {
                let m = transform.matrix.to_cols_array();
                let t = transform.translation.to_array();
                words.extend(m.iter().chain(&t).map(|v| float(*v)));
            }
            Self::Sample { ref image } => {
                let [lo, hi] = fingerprint_halves(image.derivation());
                words.extend([lo, hi, image.policy().word()]);
            }
            Self::Clamp { min, max, .. } => words.extend([float(min), float(max)]),
            Self::Remap { from, to, .. } => words.extend(from.iter().chain(&to).map(|v| float(*v))),
            Self::Warp { amount, .. } => words.push(float(amount)),
            Self::Component { index, .. } => words.push(u64::from(index)),
            Self::ToId { levels, .. } => words.push(u64::from(levels)),
            Self::Constant3 { domain: d, value } => {
                domain3(&mut words, d);
                words.push(float(value));
            }
            Self::Noise3 {
                basis,
                domain: d,
                frequency,
                seed,
            } => {
                words.push(basis_tag(basis));
                domain3(&mut words, d);
                words.extend(frequency.map(float));
                words.push(seed);
            }
            Self::Fractal3 {
                basis,
                domain: d,
                frequency,
                seed,
                params,
            } => {
                words.push(basis_tag(basis));
                domain3(&mut words, d);
                words.extend(frequency.map(float));
                words.push(seed);
                words.extend([
                    match params.kind {
                        FractalKind::Fbm => 0,
                        FractalKind::Ridged => 1,
                    },
                    u64::from(params.octaves),
                    u64::from(params.lacunarity),
                    float(params.gain),
                ]);
            }
            Self::Cellular3 {
                domain: d,
                frequency,
                jitter,
                seed,
                output,
            } => {
                domain3(&mut words, d);
                words.extend(frequency.map(float));
                words.extend([float(jitter), seed, cell_output_tag(output)]);
            }
            Self::Transform3 { transform, .. } => {
                let m = transform.matrix.to_cols_array();
                let t = transform.translation.to_array();
                words.extend(m.iter().chain(&t).map(|v| float(*v)));
            }
            Self::Slice {
                origin,
                u,
                v,
                domain: d,
                ..
            } => {
                words.extend(origin.iter().chain(&u).chain(&v).map(|v| float(*v)));
                domain(&mut words, d);
            }
            Self::Position3 | Self::Length { .. } | Self::Fract { .. } | Self::Atan2 { .. } => {}
            Self::Vector2 { .. }
            | Self::Vector3 { .. }
            | Self::Color { .. }
            | Self::AsMask { .. }
            | Self::Normalize { .. }
            | Self::Direction { .. }
            | Self::Angle { .. }
            | Self::Coherence { .. }
            | Self::Demote { .. }
            | Self::Add { .. }
            | Self::Sub { .. }
            | Self::Mul { .. }
            | Self::Min { .. }
            | Self::Max { .. }
            | Self::Abs { .. }
            | Self::Mix { .. } => {}
        }
        for input in inputs {
            let fp = input.0;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "splitting the 128-bit fingerprint into its halves"
            )]
            words.extend([fp as u64, (fp >> 64) as u64]);
        }
        fingerprint_words(&words)
    }

    /// Whether this node's value at a point depends on operand `index` only
    /// at the same point.
    ///
    /// Then a change of that operand within a region changes this node only
    /// within the same region. [`Op::Transform`] and the warped input of
    /// [`Op::Warp`] read their operand elsewhere. [`Op::Demote`] reads it at
    /// the same point, but a plane field does not repeat, so a periodic
    /// change's repeats would no longer be stated.
    #[must_use]
    pub const fn operand_is_pointwise(&self, index: usize) -> bool {
        match self {
            Self::Transform { .. }
            | Self::Demote { .. }
            | Self::Transform3 { .. }
            | Self::Slice { .. } => false,
            Self::Warp { .. } => index != 0,
            _ => true,
        }
    }

    /// The node's inputs, in operand order.
    #[must_use]
    pub fn inputs(&self) -> Inputs {
        let mut inputs = Inputs::default();
        match *self {
            Self::Constant { .. }
            | Self::Noise { .. }
            | Self::Fractal { .. }
            | Self::Cellular { .. }
            | Self::Tiling { .. }
            | Self::Scatter { .. }
            | Self::Disk { .. }
            | Self::Sample { .. }
            | Self::Constant3 { .. }
            | Self::Noise3 { .. }
            | Self::Fractal3 { .. }
            | Self::Cellular3 { .. }
            | Self::Position3 => {}
            Self::Transform { input, .. }
            | Self::Transform3 { input, .. }
            | Self::Slice { input, .. }
            | Self::Length { input }
            | Self::Fract { input }
            | Self::Demote { input }
            | Self::Abs { input }
            | Self::Clamp { input, .. }
            | Self::Remap { input, .. }
            | Self::Component { input, .. }
            | Self::AsMask { input }
            | Self::ToId { input, .. }
            | Self::Normalize { input }
            | Self::Direction { angle: input }
            | Self::Angle { input }
            | Self::Coherence { input } => inputs.push(input),
            Self::Add { a, b }
            | Self::Sub { a, b }
            | Self::Mul { a, b }
            | Self::Min { a, b }
            | Self::Max { a, b }
            | Self::Atan2 { y: a, x: b }
            | Self::Vector2 { x: a, y: b } => {
                inputs.push(a);
                inputs.push(b);
            }
            Self::Vector3 { x, y, z } | Self::Color { r: x, g: y, b: z } => {
                inputs.push(x);
                inputs.push(y);
                inputs.push(z);
            }
            Self::Mix { a, b, t } => {
                inputs.push(a);
                inputs.push(b);
                inputs.push(t);
            }
            Self::Warp { input, dx, dy, .. } => {
                inputs.push(input);
                inputs.push(dx);
                inputs.push(dy);
            }
        }
        inputs
    }

    /// The same operation and parameters with every operand replaced by
    /// `f(operand)`.
    #[must_use]
    pub fn map_inputs(&self, mut f: impl FnMut(NodeId) -> NodeId) -> Self {
        let mut op = self.clone();
        match &mut op {
            Self::Constant { .. }
            | Self::Noise { .. }
            | Self::Fractal { .. }
            | Self::Cellular { .. }
            | Self::Tiling { .. }
            | Self::Scatter { .. }
            | Self::Disk { .. }
            | Self::Sample { .. }
            | Self::Constant3 { .. }
            | Self::Noise3 { .. }
            | Self::Fractal3 { .. }
            | Self::Cellular3 { .. }
            | Self::Position3 => {}
            Self::Transform { input, .. }
            | Self::Transform3 { input, .. }
            | Self::Slice { input, .. }
            | Self::Length { input }
            | Self::Fract { input }
            | Self::Demote { input }
            | Self::Abs { input }
            | Self::Clamp { input, .. }
            | Self::Remap { input, .. }
            | Self::Component { input, .. }
            | Self::AsMask { input }
            | Self::ToId { input, .. }
            | Self::Normalize { input }
            | Self::Direction { angle: input }
            | Self::Angle { input }
            | Self::Coherence { input } => *input = f(*input),
            Self::Add { a, b }
            | Self::Sub { a, b }
            | Self::Mul { a, b }
            | Self::Min { a, b }
            | Self::Max { a, b }
            | Self::Atan2 { y: a, x: b }
            | Self::Vector2 { x: a, y: b } => {
                *a = f(*a);
                *b = f(*b);
            }
            Self::Vector3 { x, y, z } | Self::Color { r: x, g: y, b: z } => {
                *x = f(*x);
                *y = f(*y);
                *z = f(*z);
            }
            Self::Mix { a, b, t } => {
                *a = f(*a);
                *b = f(*b);
                *t = f(*t);
            }
            Self::Warp { input, dx, dy, .. } => {
                *input = f(*input);
                *dx = f(*dx);
                *dy = f(*dy);
            }
        }
        op
    }

    /// A short operation name, for listings and reports.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Constant { .. } => "constant",
            Self::Noise { .. } => "noise",
            Self::Fractal { .. } => "fractal",
            Self::Cellular { .. } => "cellular",
            Self::Tiling { .. } => "tiling",
            Self::Scatter { .. } => "scatter",
            Self::Disk { .. } => "disk",
            Self::Transform { .. } => "transform",
            Self::Demote { .. } => "demote",
            Self::Add { .. } => "add",
            Self::Sub { .. } => "sub",
            Self::Mul { .. } => "mul",
            Self::Min { .. } => "min",
            Self::Max { .. } => "max",
            Self::Abs { .. } => "abs",
            Self::Clamp { .. } => "clamp",
            Self::Remap { .. } => "remap",
            Self::Mix { .. } => "mix",
            Self::Warp { .. } => "warp",
            Self::Vector2 { .. } => "vector2",
            Self::Vector3 { .. } => "vector3",
            Self::Color { .. } => "color",
            Self::Component { .. } => "component",
            Self::AsMask { .. } => "as-mask",
            Self::ToId { .. } => "to-id",
            Self::Normalize { .. } => "normalize",
            Self::Direction { .. } => "direction",
            Self::Angle { .. } => "angle",
            Self::Coherence { .. } => "coherence",
            Self::Sample { .. } => "sample",
            Self::Constant3 { .. } => "constant3",
            Self::Noise3 { .. } => "noise3",
            Self::Fractal3 { .. } => "fractal3",
            Self::Cellular3 { .. } => "cellular3",
            Self::Position3 => "position3",
            Self::Transform3 { .. } => "transform3",
            Self::Slice { .. } => "slice",
            Self::Length { .. } => "length",
            Self::Fract { .. } => "fract",
            Self::Atan2 { .. } => "atan2",
        }
    }
}

/// Up to three node inputs, in operand order.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Inputs {
    ids: [Option<NodeId>; 3],
    len: u8,
}

impl Inputs {
    fn push(&mut self, id: NodeId) {
        self.ids[usize::from(self.len)] = Some(id);
        self.len += 1;
    }

    /// Iterates the inputs in operand order.
    pub fn iter(self) -> impl Iterator<Item = NodeId> {
        self.ids.into_iter().take(usize::from(self.len)).flatten()
    }
}

/// A 128-bit content fingerprint of a node and everything it depends on.
///
/// The encoding (version [`FINGERPRINT_VERSION`]) is a sequence of `u64`
/// words hashed with [`hash`] under two fixed seeds, one
/// per half:
///
/// - the version, then the op's tag (its position in [`Op`]'s declaration);
/// - a domain as `0` for `Plane` or `1, px, py` for `Periodic`;
/// - every `f32` as its bit pattern, every seed as itself, and enums
///   ([`Basis`], [`FractalKind`], [`CellOutput`]) as their declaration index;
/// - [`FractalParams`] as kind, octaves, lacunarity and gain bits;
/// - an [`Affine2`] as its matrix columns, then its translation;
/// - a component index or identifier level count as itself;
/// - an [`Op::Sample`] image as the two halves of its derivation
///   ([`SampleImage::derivation`]), then its [`SamplePolicy::word`]; its
///   texels, type and shape are not encoded (the derivation names them);
/// - each input as the two halves of its own fingerprint, in operand order.
///
/// Node identities and creation order are not part of it: equal subgraphs
/// have equal fingerprints. Port types are not encoded: they follow from the
/// operations and inputs, which are.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Fingerprint(pub u128);

impl fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

const LANE_SEEDS: [u64; 2] = [0x6461_7070_6c65_2d30, 0x6461_7070_6c65_2d31]; // "dapple-0", "dapple-1"

/// A fingerprint's low and high 64-bit halves.
fn fingerprint_halves(fp: Fingerprint) -> [u64; 2] {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "splitting the 128-bit fingerprint into its halves"
    )]
    [fp.0 as u64, (fp.0 >> 64) as u64]
}

/// The fingerprint an [`Op::Sample`] node has for an image derived as
/// `derivation` and sampled under `policy`, for predicting it without the
/// texels.
#[must_use]
pub fn sample_fingerprint(derivation: Fingerprint, policy: SamplePolicy) -> Fingerprint {
    let [lo, hi] = fingerprint_halves(derivation);
    fingerprint_words(&[FINGERPRINT_VERSION, 28, lo, hi, policy.word()])
}

fn fingerprint_words(words: &[u64]) -> Fingerprint {
    let [lo, hi] = LANE_SEEDS.map(|seed| hash(seed, words));
    Fingerprint((u128::from(hi) << 64) | u128::from(lo))
}

/// A rejected program construction.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ProgramError {
    /// A field parameter was rejected; see [`DomainError`].
    Domain(DomainError),
    /// An input refers to a node that does not exist (yet).
    UnknownNode {
        /// The unknown node.
        node: NodeId,
    },
    /// Inputs of one node have different domains.
    DomainMismatch {
        /// The first input's domain.
        first: Domain,
        /// A later input's differing domain.
        other: Domain,
    },
    /// An input's [`PortType`] does not fit the operation.
    TypeMismatch {
        /// The operation's [`Op::name`].
        op: &'static str,
        /// The offending input's type.
        found: PortType,
        /// What the operation needs, or why the input cannot be used.
        reason: &'static str,
    },
    /// [`ProgramBuilder::finish`] needs a scalar or mask output; use
    /// [`ProgramBuilder::finish_value`] for other types.
    OutputType {
        /// The output node's type.
        found: PortType,
    },
    /// Inputs of one node lie in different spaces, or an operation needs
    /// the other space: a planar op given a solid input, or the reverse.
    SpaceMismatch {
        /// The operation's [`Op::name`].
        op: &'static str,
        /// The offending space.
        found: Space,
    },
    /// The output is in the wrong space: [`ProgramBuilder::finish`] and
    /// [`ProgramBuilder::finish_value`] need a planar output,
    /// [`ProgramBuilder::finish_solid`] a solid one.
    OutputSpace {
        /// The output node's space.
        found: Space,
    },
}

impl From<DomainError> for ProgramError {
    fn from(error: DomainError) -> Self {
        Self::Domain(error)
    }
}

impl fmt::Display for ProgramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Domain(error) => error.fmt(f),
            Self::UnknownNode { node } => write!(f, "unknown node {}", node.index()),
            Self::DomainMismatch { first, other } => {
                write!(f, "inputs mix domains {first:?} and {other:?}")
            }
            Self::TypeMismatch { op, found, reason } => {
                write!(f, "{op} cannot take a {found} input: {reason}")
            }
            Self::OutputType { found } => {
                write!(f, "a scalar field program cannot output a {found}")
            }
            Self::SpaceMismatch { op, found } => {
                write!(f, "{op} cannot take an input in {found:?}")
            }
            Self::OutputSpace { found } => write!(f, "the output is in the wrong space, {found:?}"),
        }
    }
}

impl core::error::Error for ProgramError {}

/// The evaluator behind one node.
#[derive(Clone, Debug, PartialEq)]
enum Kernel {
    Constant(f32),
    Noise(Noise),
    Fractal(Fractal),
    Cellular(CellularField),
    Tiling(TilingField),
    Scatter(ScatterField),
    Disk(Disk),
    Sample(SampleImage),
    Transform {
        input: NodeId,
        transform: Affine2,
        stretch: f32,
    },
    Pass(NodeId),
    Noise3(Noise3),
    Fractal3(Fractal3),
    Cellular3(CellularField3),
    Position3,
    Transform3 {
        input: NodeId,
        transform: Affine3,
        stretch: f32,
    },
    Slice {
        input: NodeId,
        origin: Vec3,
        u: Vec3,
        v: Vec3,
        stretch: f32,
    },
    Length(NodeId),
    Fract(NodeId),
    Atan2(NodeId, NodeId),
    Binary(BinaryOp, NodeId, NodeId),
    Abs(NodeId),
    Clamp(NodeId, f32, f32),
    Remap {
        input: NodeId,
        scale: f32,
        from: f32,
        to: f32,
    },
    Mix(NodeId, NodeId, NodeId),
    Warp {
        input: NodeId,
        dx: NodeId,
        dy: NodeId,
        amount: f32,
    },
    Vector2(NodeId, NodeId),
    Vector3(NodeId, NodeId, NodeId),
    Component(NodeId, usize),
    AsMask(NodeId),
    ToId(NodeId, u32),
    Normalize(NodeId),
    Direction(NodeId),
    Angle(NodeId),
    Coherence(NodeId),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Min,
    Max,
}

#[derive(Clone, Debug, PartialEq)]
struct Node {
    op: Op,
    space: Space,
    port: PortType,
    fingerprint: Fingerprint,
    kernel: Kernel,
}

/// Builds a [`FieldProgram`] node by node.
///
/// Nodes may only refer to earlier nodes, so every program is acyclic and
/// creation order is a valid evaluation order.
#[derive(Clone, Debug, Default)]
pub struct ProgramBuilder {
    nodes: Vec<Node>,
}

impl ProgramBuilder {
    /// An empty builder.
    #[must_use]
    pub const fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Validates `op` and appends it, returning its node.
    pub fn add(&mut self, op: Op) -> Result<NodeId, ProgramError> {
        for input in op.inputs().iter() {
            self.node(input)?;
        }
        let space = self.derive_space(&op)?;
        let port = self.derive_type(&op)?;
        let kernel = self.kernel(&op)?;
        let fingerprint = self.fingerprint(&op);
        let id = NodeId(u32::try_from(self.nodes.len()).expect("fewer than 2^32 nodes"));
        self.nodes.push(Node {
            op,
            space,
            port,
            fingerprint,
            kernel,
        });
        Ok(id)
    }

    /// Copies the nodes `program`'s output depends on into this builder and
    /// returns the output's node here.
    ///
    /// A node whose fingerprint equals one already in the builder is not
    /// copied again: shared subgraphs of several imported programs become one
    /// subgraph, which the flat plan then evaluates once.
    pub fn import(&mut self, program: &FieldProgram) -> Result<NodeId, ProgramError> {
        // Nodes the output depends on; operands always precede their users.
        let mut needed = alloc::vec![false; program.nodes.len()];
        needed[program.output.0 as usize] = true;
        for index in (0..program.nodes.len()).rev() {
            if needed[index] {
                for input in program.nodes[index].op.inputs().iter() {
                    needed[input.0 as usize] = true;
                }
            }
        }
        let mut mapped = alloc::vec![None::<NodeId>; program.nodes.len()];
        for (index, node) in program.nodes.iter().enumerate() {
            if !needed[index] {
                continue;
            }
            let existing = self
                .nodes
                .iter()
                .position(|n| n.fingerprint == node.fingerprint);
            let id = match existing {
                Some(existing) => NodeId(u32::try_from(existing).expect("fewer than 2^32 nodes")),
                None => self
                    .add(node.op.map_inputs(|input| {
                        mapped[input.0 as usize].expect("operands come first")
                    }))?,
            };
            mapped[index] = Some(id);
        }
        Ok(mapped[program.output.0 as usize].expect("the output is needed"))
    }

    /// The planar domain of an existing node.
    ///
    /// A solid node has none: it fails with [`ProgramError::SpaceMismatch`];
    /// see [`Self::space`].
    pub fn domain(&self, id: NodeId) -> Result<Domain, ProgramError> {
        let node = self.node(id)?;
        node.space.planar().ok_or(ProgramError::SpaceMismatch {
            op: node.op.name(),
            found: node.space,
        })
    }

    /// The [`Space`] of an existing node.
    pub fn space(&self, id: NodeId) -> Result<Space, ProgramError> {
        Ok(self.node(id)?.space)
    }

    /// The type of an existing node.
    pub fn port_type(&self, id: NodeId) -> Result<PortType, ProgramError> {
        Ok(self.node(id)?.port)
    }

    /// Finishes a scalar-valued program with `output` as its result.
    ///
    /// `output` must be a [`PortType::Scalar`] or [`PortType::Mask`], so the
    /// program is a [`ScalarField`]; use [`Self::finish_value`] for other
    /// types. Nodes that `output` does not depend on are kept for inspection
    /// but never evaluated, and do not affect [`FieldProgram::fingerprint`].
    pub fn finish(self, output: NodeId) -> Result<FieldProgram, ProgramError> {
        let node = self.node(output)?;
        if !node.port.is_scalar() {
            return Err(ProgramError::OutputType { found: node.port });
        }
        if node.space.planar().is_none() {
            return Err(ProgramError::OutputSpace { found: node.space });
        }
        Ok(FieldProgram::new(self.nodes, output))
    }

    /// Finishes a program whose output may have any [`PortType`].
    ///
    /// The output must be planar; slice a solid field first.
    pub fn finish_value(self, output: NodeId) -> Result<ValueProgram, ProgramError> {
        let node = self.node(output)?;
        if node.space.planar().is_none() {
            return Err(ProgramError::OutputSpace { found: node.space });
        }
        Ok(ValueProgram {
            program: FieldProgram::new(self.nodes, output),
        })
    }

    /// Finishes a solid program with scalar or mask `output`, evaluated at 3D
    /// points ([`SolidProgram`]).
    pub fn finish_solid(self, output: NodeId) -> Result<SolidProgram, ProgramError> {
        let node = self.node(output)?;
        if !node.port.is_scalar() {
            return Err(ProgramError::OutputType { found: node.port });
        }
        if node.space.solid().is_none() {
            return Err(ProgramError::OutputSpace { found: node.space });
        }
        Ok(SolidProgram {
            program: FieldProgram::new(self.nodes, output),
        })
    }

    fn node(&self, id: NodeId) -> Result<&Node, ProgramError> {
        self.nodes
            .get(id.0 as usize)
            .ok_or(ProgramError::UnknownNode { node: id })
    }

    fn derive_space(&self, op: &Op) -> Result<Space, ProgramError> {
        let space = |id: NodeId| self.nodes[id.0 as usize].space;
        let needs = |id: NodeId, solid: bool| {
            let found = space(id);
            if found.solid().is_some() == solid {
                Ok(found)
            } else {
                Err(ProgramError::SpaceMismatch {
                    op: op.name(),
                    found,
                })
            }
        };
        Ok(match *op {
            Op::Constant { domain, .. }
            | Op::Noise { domain, .. }
            | Op::Fractal { domain, .. }
            | Op::Cellular { domain, .. }
            | Op::Tiling { domain, .. }
            | Op::Scatter { domain, .. }
            | Op::Disk { domain, .. } => Space::Planar(domain),
            Op::Sample { ref image } => Space::Planar(image.domain()),
            Op::Constant3 { domain, .. }
            | Op::Noise3 { domain, .. }
            | Op::Fractal3 { domain, .. }
            | Op::Cellular3 { domain, .. } => Space::Solid(domain),
            Op::Position3 => Space::Solid(Domain3::Space),
            Op::Demote { input } => match space(input) {
                Space::Planar(_) => Space::Planar(Domain::Plane),
                Space::Solid(_) => Space::Solid(Domain3::Space),
            },
            Op::Transform { input, .. } | Op::Warp { input, .. } => {
                needs(input, false)?;
                self.shared_space(op)?
            }
            Op::Transform3 { input, .. } => needs(input, true)?,
            Op::Slice { input, domain, .. } => {
                needs(input, true)?;
                Space::Planar(domain)
            }
            _ => self.shared_space(op)?,
        })
    }

    /// The one space all of `op`'s inputs share.
    fn shared_space(&self, op: &Op) -> Result<Space, ProgramError> {
        let mut inputs = op.inputs().iter().map(|id| self.nodes[id.0 as usize].space);
        let first = inputs.next().expect("derived nodes have inputs");
        if let Some(other) = inputs.find(|s| *s != first) {
            return Err(match (first, other) {
                (Space::Planar(first), Space::Planar(other)) => {
                    ProgramError::DomainMismatch { first, other }
                }
                _ => ProgramError::SpaceMismatch {
                    op: op.name(),
                    found: other,
                },
            });
        }
        Ok(first)
    }

    fn derive_type(&self, op: &Op) -> Result<PortType, ProgramError> {
        derive_port(op, &|id: NodeId| self.nodes[id.0 as usize].port)
    }

    fn kernel(&self, op: &Op) -> Result<Kernel, ProgramError> {
        let finite = |name: &'static str, values: &[f32]| {
            if values.iter().all(|v| v.is_finite()) {
                Ok(())
            } else {
                Err(DomainError::InvalidParameter { name })
            }
        };
        Ok(match *op {
            Op::Constant { value, .. } => {
                finite("value", &[value])?;
                Kernel::Constant(value)
            }
            Op::Noise {
                basis,
                domain,
                frequency,
                seed,
            } => Kernel::Noise(Noise::new(basis, domain, Vec2::from(frequency), seed)?),
            Op::Fractal {
                basis,
                domain,
                frequency,
                seed,
                params,
            } => Kernel::Fractal(Fractal::new(
                basis,
                domain,
                Vec2::from(frequency),
                seed,
                params,
            )?),
            Op::Cellular {
                domain,
                frequency,
                jitter,
                seed,
                output,
            } => Kernel::Cellular(
                Cellular::new(domain, Vec2::from(frequency), jitter, seed)?.output(output),
            ),
            Op::Tiling {
                domain,
                pattern,
                seed,
                output,
            } => Kernel::Tiling(Tiling::new(domain, pattern, seed)?.output(output)),
            Op::Scatter {
                domain,
                placement,
                ref stamp,
                seed,
                output,
            } => Kernel::Scatter(
                Scatter::new(domain, placement, stamp.clone(), seed)?.output(output),
            ),
            Op::Disk {
                domain,
                center,
                radius,
                softness,
            } => Kernel::Disk(Disk::new(domain, Vec2::from(center), radius, softness)?),
            Op::Sample { ref image } => Kernel::Sample(image.clone()),
            Op::Transform { input, transform } => {
                let domain = self.nodes[input.0 as usize]
                    .space
                    .planar()
                    .expect("the input's space was checked");
                let stretch = check_transform(domain, transform)?;
                Kernel::Transform {
                    input,
                    transform,
                    stretch,
                }
            }
            Op::Demote { input } => Kernel::Pass(input),
            Op::Add { a, b } => Kernel::Binary(BinaryOp::Add, a, b),
            Op::Sub { a, b } => Kernel::Binary(BinaryOp::Sub, a, b),
            Op::Mul { a, b } => Kernel::Binary(BinaryOp::Mul, a, b),
            Op::Min { a, b } => Kernel::Binary(BinaryOp::Min, a, b),
            Op::Max { a, b } => Kernel::Binary(BinaryOp::Max, a, b),
            Op::Abs { input } => Kernel::Abs(input),
            Op::Clamp { input, min, max } => {
                finite("clamp", &[min, max])?;
                if min > max {
                    return Err(DomainError::InvalidParameter { name: "clamp" }.into());
                }
                Kernel::Clamp(input, min, max)
            }
            Op::Remap { input, from, to } => {
                finite("remap", &[from[0], from[1], to[0], to[1]])?;
                let span = from[1] - from[0];
                if span == 0.0 || !span.is_finite() {
                    return Err(DomainError::InvalidParameter { name: "remap" }.into());
                }
                Kernel::Remap {
                    input,
                    scale: (to[1] - to[0]) / span,
                    from: from[0],
                    to: to[0],
                }
            }
            Op::Mix { a, b, t } => Kernel::Mix(a, b, t),
            Op::Warp {
                input,
                dx,
                dy,
                amount,
            } => {
                finite("amount", &[amount])?;
                Kernel::Warp {
                    input,
                    dx,
                    dy,
                    amount,
                }
            }
            Op::Vector2 { x, y } => Kernel::Vector2(x, y),
            Op::Vector3 { x, y, z } => Kernel::Vector3(x, y, z),
            Op::Color { r, g, b } => Kernel::Vector3(r, g, b),
            Op::Component { input, index } => Kernel::Component(input, usize::from(index)),
            Op::AsMask { input } => Kernel::AsMask(input),
            Op::ToId { input, levels } => {
                if levels == 0 {
                    return Err(DomainError::InvalidParameter { name: "levels" }.into());
                }
                Kernel::ToId(input, levels)
            }
            Op::Normalize { input } => Kernel::Normalize(input),
            Op::Direction { angle } => Kernel::Direction(angle),
            Op::Angle { input } => Kernel::Angle(input),
            Op::Coherence { input } => Kernel::Coherence(input),
            Op::Constant3 { value, .. } => {
                finite("value", &[value])?;
                Kernel::Constant(value)
            }
            Op::Noise3 {
                basis,
                domain,
                frequency,
                seed,
            } => Kernel::Noise3(Noise3::new(basis, domain, Vec3::from(frequency), seed)?),
            Op::Fractal3 {
                basis,
                domain,
                frequency,
                seed,
                params,
            } => Kernel::Fractal3(Fractal3::new(
                basis,
                domain,
                Vec3::from(frequency),
                seed,
                params,
            )?),
            Op::Cellular3 {
                domain,
                frequency,
                jitter,
                seed,
                output,
            } => Kernel::Cellular3(
                Cellular3::new(domain, Vec3::from(frequency), jitter, seed)?.output(output),
            ),
            Op::Position3 => Kernel::Position3,
            Op::Transform3 { input, transform } => {
                let domain = self.nodes[input.0 as usize]
                    .space
                    .solid()
                    .expect("the input's space was checked");
                let stretch = check_transform3(domain, transform)?;
                Kernel::Transform3 {
                    input,
                    transform,
                    stretch,
                }
            }
            Op::Slice {
                input,
                origin,
                u,
                v,
                domain,
            } => {
                finite("slice", &[origin, u, v].concat())?;
                let input_domain = self.nodes[input.0 as usize]
                    .space
                    .solid()
                    .expect("the input's space was checked");
                check_slice(input_domain, u, v, domain)?;
                let (u, v) = (Vec3::from(u), Vec3::from(v));
                Kernel::Slice {
                    input,
                    origin: Vec3::from(origin),
                    u,
                    v,
                    stretch: slice_stretch(u, v),
                }
            }
            Op::Length { input } => Kernel::Length(input),
            Op::Fract { input } => Kernel::Fract(input),
            Op::Atan2 { y, x } => Kernel::Atan2(y, x),
        })
    }

    fn fingerprint(&self, op: &Op) -> Fingerprint {
        let inputs: Vec<Fingerprint> = op
            .inputs()
            .iter()
            .map(|input| self.nodes[input.0 as usize].fingerprint)
            .collect();
        op.fingerprint_with(&inputs)
    }
}

fn op_tag(op: &Op) -> u64 {
    match op {
        Op::Constant { .. } => 0,
        Op::Noise { .. } => 1,
        Op::Fractal { .. } => 2,
        Op::Cellular { .. } => 3,
        Op::Transform { .. } => 4,
        Op::Demote { .. } => 5,
        Op::Add { .. } => 6,
        Op::Sub { .. } => 7,
        Op::Mul { .. } => 8,
        Op::Min { .. } => 9,
        Op::Max { .. } => 10,
        Op::Abs { .. } => 11,
        Op::Clamp { .. } => 12,
        Op::Remap { .. } => 13,
        Op::Mix { .. } => 14,
        Op::Warp { .. } => 15,
        Op::Vector2 { .. } => 16,
        Op::Vector3 { .. } => 17,
        Op::Color { .. } => 18,
        Op::Component { .. } => 19,
        Op::AsMask { .. } => 20,
        Op::ToId { .. } => 21,
        Op::Normalize { .. } => 22,
        Op::Direction { .. } => 24,
        Op::Angle { .. } => 25,
        Op::Coherence { .. } => 26,
        Op::Disk { .. } => 27,
        Op::Sample { .. } => 28,
        Op::Constant3 { .. } => 29,
        Op::Noise3 { .. } => 30,
        Op::Fractal3 { .. } => 31,
        Op::Cellular3 { .. } => 32,
        Op::Position3 => 33,
        Op::Transform3 { .. } => 34,
        Op::Slice { .. } => 35,
        Op::Length { .. } => 36,
        Op::Fract { .. } => 37,
        Op::Atan2 { .. } => 38,
        Op::Tiling { .. } => 39,
        Op::Scatter { .. } => 40,
    }
}

/// Checks a slice's requested planar domain against its solid input's.
///
/// A periodic slice repeats every `px` along `u` and `py` along `v`, which
/// must be lattice vectors of the input's period on every solid axis.
fn check_slice(
    input: Domain3,
    u: [f32; 3],
    v: [f32; 3],
    domain: Domain,
) -> Result<(), DomainError> {
    let Domain::Periodic { period } = domain else {
        return Ok(());
    };
    let Domain3::Periodic3 { period: solid } = input else {
        return Err(DomainError::NotLatticePreserving);
    };
    let tiles = |step: [f32; 3], repeat: u32| {
        step.iter().zip(solid).all(|(&m, p)| {
            let steps = f64::from(m) * f64::from(repeat) / f64::from(p);
            libm::trunc(steps) == steps
        })
    };
    if tiles(u, period[0]) && tiles(v, period[1]) {
        Ok(())
    } else {
        Err(DomainError::NotLatticePreserving)
    }
}

/// The larger singular value of the 3 × 2 matrix `[u v]`: how much a slice
/// stretches a planar footprint.
fn slice_stretch(u: Vec3, v: Vec3) -> f32 {
    let (uu, vv, uv) = (u.dot(u), v.dot(v), u.dot(v));
    let sum = uu + vv;
    let det = uu * vv - uv * uv;
    let disc = libm::sqrtf((sum * sum - 4.0 * det).max(0.0));
    libm::sqrtf((sum + disc) * 0.5)
}

const fn basis_tag(basis: Basis) -> u64 {
    match basis {
        Basis::Value => 0,
        Basis::Gradient => 1,
    }
}

const fn scatter_output_tag(output: ScatterOutput) -> u64 {
    match output {
        ScatterOutput::Coverage => 0,
        ScatterOutput::Max => 1,
        ScatterOutput::TopValue => 2,
    }
}

const fn tile_output_tag(output: TileOutput) -> u64 {
    match output {
        TileOutput::Edge => 0,
        TileOutput::U => 1,
        TileOutput::V => 2,
        TileOutput::Vertical => 3,
        TileOutput::TileValue => 4,
    }
}

const fn cell_output_tag(output: CellOutput) -> u64 {
    match output {
        CellOutput::F1 => 0,
        CellOutput::F2 => 1,
        CellOutput::F2MinusF1 => 2,
        CellOutput::Border => 3,
        CellOutput::CellValue => 4,
    }
}

/// A finished, immutable field program.
///
/// Finishing compiles a flat plan of the output (see [`Evaluator`]): each
/// node is evaluated once per point in each context it is reached in, so a
/// subgraph shared by several consumers is not repeated.
/// [`ScalarField::eval`] uses the plan when it saves work and walks the DAG
/// recursively otherwise; both give the same bits. For many points, keep one
/// [`Self::evaluator`] to reuse its buffers. [`Self::evaluation_stats`]
/// reports the saving.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldProgram {
    nodes: Vec<Node>,
    output: NodeId,
    plan: flat::Plan,
}

impl FieldProgram {
    fn new(nodes: Vec<Node>, output: NodeId) -> Self {
        let plan = flat::Plan::new(&nodes, output);
        Self {
            nodes,
            output,
            plan,
        }
    }

    /// An evaluator of the output through the flat plan.
    #[must_use]
    pub fn evaluator(&self) -> Evaluator<'_> {
        Evaluator::new(self)
    }

    /// Evaluation counts of the output, flat and recursive.
    #[must_use]
    pub fn evaluation_stats(&self) -> EvaluationStats {
        self.plan.stats()
    }

    /// The output node.
    #[must_use]
    pub const fn output(&self) -> NodeId {
        self.output
    }

    /// The number of nodes, including any the output does not use.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True when the program has no nodes; never true for a finished program.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The operation of `id`, if it exists.
    #[must_use]
    pub fn op(&self, id: NodeId) -> Option<&Op> {
        self.nodes.get(id.0 as usize).map(|node| &node.op)
    }

    /// The planar domain of `id`, if it exists and is planar.
    #[must_use]
    pub fn node_domain(&self, id: NodeId) -> Option<Domain> {
        self.nodes
            .get(id.0 as usize)
            .and_then(|node| node.space.planar())
    }

    /// The [`Space`] of `id`, if it exists.
    #[must_use]
    pub fn node_space(&self, id: NodeId) -> Option<Space> {
        self.nodes.get(id.0 as usize).map(|node| node.space)
    }

    /// The fingerprint of `id`, if it exists.
    #[must_use]
    pub fn node_fingerprint(&self, id: NodeId) -> Option<Fingerprint> {
        self.nodes.get(id.0 as usize).map(|node| node.fingerprint)
    }

    /// The program's content fingerprint: its output node's.
    #[must_use]
    pub fn fingerprint(&self) -> Fingerprint {
        self.nodes[self.output.0 as usize].fingerprint
    }

    /// Iterates all nodes in creation order.
    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, &Op)> + '_ {
        self.nodes.iter().enumerate().map(|(index, node)| {
            (
                NodeId(u32::try_from(index).expect("fewer than 2^32 nodes")),
                &node.op,
            )
        })
    }

    /// The type of `id`, if it exists.
    #[must_use]
    pub fn node_type(&self, id: NodeId) -> Option<PortType> {
        self.nodes.get(id.0 as usize).map(|node| node.port)
    }

    /// Evaluates scalar node `id` at `p`.
    ///
    /// # Panics
    ///
    /// Panics if `id` is not a scalar or mask node of this program.
    #[must_use]
    pub fn eval_node(&self, id: NodeId, p: Vec2, footprint: Footprint) -> f32 {
        self.eval_value(id, p, footprint)
            .scalar()
            .expect("eval_node needs a scalar or mask node")
    }

    /// Evaluates node `id` at `p`, whatever its type.
    ///
    /// # Panics
    ///
    /// Panics if `id` is not a planar node of this program.
    #[must_use]
    pub fn eval_value(&self, id: NodeId, p: Vec2, footprint: Footprint) -> Value {
        assert!(
            self.nodes[id.0 as usize].space.planar().is_some(),
            "eval_value needs a planar node"
        );
        self.eval_at(id, p.extend(0.0), footprint)
    }

    /// Evaluates node `id` at a point of its space: planar nodes read `x`
    /// and `y` and ignore `z`.
    pub(super) fn eval_at(&self, id: NodeId, p: Vec3, footprint: Footprint) -> Value {
        let node = &self.nodes[id.0 as usize];
        match node.kernel {
            Kernel::Transform {
                input,
                transform,
                stretch,
            } => self.eval_at(
                input,
                transform.apply(p.truncate()).extend(p.z),
                footprint.scaled(stretch),
            ),
            Kernel::Transform3 {
                input,
                transform,
                stretch,
            } => self.eval_at(input, transform.apply(p), footprint.scaled(stretch)),
            Kernel::Slice {
                input,
                origin,
                u,
                v,
                stretch,
            } => self.eval_at(
                input,
                slice_point(origin, u, v, p),
                footprint.scaled(stretch),
            ),
            Kernel::Pass(input) => self.eval_at(input, p, footprint),
            Kernel::Warp {
                input,
                dx,
                dy,
                amount,
            } => {
                let (q, footprint) = self.warp_point(dx, dy, amount, p, footprint).0;
                self.eval_at(input, q, footprint)
            }
            ref kernel => {
                let mut args = [Value::Scalar(0.0); 3];
                let mut count = 0;
                for input in node.op.inputs().iter() {
                    args[count] = self.eval_at(input, p, footprint);
                    count += 1;
                }
                kernel.combine(p, footprint, &args[..count])
            }
        }
    }

    /// Node `id`'s scalar value at `p` and its gradient, both band-limited
    /// to `footprint`.
    ///
    /// The value equals [`Self::eval_node`]'s, bit for bit. Gradients follow
    /// the chain rule through every op with a closed-form derivative, and
    /// through transforms, slices and warps; any other node's gradient, such
    /// as a vector component's, is taken by central differences of that node
    /// (see [`central_difference`](crate::central_difference)).
    ///
    /// # Panics
    ///
    /// Panics if `id` is not a planar scalar or mask node of this program.
    #[must_use]
    pub fn eval_node_gradient(&self, id: NodeId, p: Vec2, footprint: Footprint) -> (f32, Vec2) {
        assert!(
            self.nodes[id.0 as usize].space.planar().is_some(),
            "eval_node_gradient needs a planar node"
        );
        let (value, gradient) = self.gradient_at(id, p.extend(0.0), footprint);
        (
            value
                .scalar()
                .expect("eval_node_gradient needs a scalar or mask node"),
            gradient.truncate(),
        )
    }

    /// The warp's point and footprint, and its Jacobian's rows, at `p`.
    fn warp_point(
        &self,
        dx: NodeId,
        dy: NodeId,
        amount: f32,
        p: Vec3,
        footprint: Footprint,
    ) -> ((Vec3, Footprint), [Vec3; 2]) {
        if footprint.width() <= 0.0 {
            // A point footprint does not scale, so no derivatives are needed.
            let center = [
                self.eval_at(dx, p, footprint),
                self.eval_at(dy, p, footprint),
            ];
            return (
                warp_move(p, footprint, amount, center, [Vec3::ZERO; 2]),
                [Vec3::ZERO; 2],
            );
        }
        let (cx, gx) = self.gradient_at(dx, p, footprint);
        let (cy, gy) = self.gradient_at(dy, p, footprint);
        (
            warp_move(p, footprint, amount, [cx, cy], [gx, gy]),
            [gx, gy],
        )
    }

    /// A node's value and the gradient of its scalar value (zero for vector
    /// values, whose consumers differentiate their own scalars). A planar
    /// node's gradient has zero `z`.
    pub(super) fn gradient_at(&self, id: NodeId, p: Vec3, footprint: Footprint) -> (Value, Vec3) {
        let node = &self.nodes[id.0 as usize];
        match node.kernel {
            Kernel::Transform {
                input,
                transform,
                stretch,
            } => {
                let (value, gradient) = self.gradient_at(
                    input,
                    transform.apply(p.truncate()).extend(p.z),
                    footprint.scaled(stretch),
                );
                (
                    value,
                    (transform.matrix.transpose() * gradient.truncate()).extend(0.0),
                )
            }
            Kernel::Transform3 {
                input,
                transform,
                stretch,
            } => {
                let (value, gradient) =
                    self.gradient_at(input, transform.apply(p), footprint.scaled(stretch));
                (value, transform.matrix.transpose() * gradient)
            }
            Kernel::Slice {
                input,
                origin,
                u,
                v,
                stretch,
            } => {
                let (value, gradient) = self.gradient_at(
                    input,
                    slice_point(origin, u, v, p),
                    footprint.scaled(stretch),
                );
                (value, slice_chain(gradient, u, v))
            }
            Kernel::Pass(input) => self.gradient_at(input, p, footprint),
            Kernel::Warp {
                input,
                dx,
                dy,
                amount,
            } => {
                let (cx, gx) = self.gradient_at(dx, p, footprint);
                let (cy, gy) = self.gradient_at(dy, p, footprint);
                let (q, moved) = warp_move(p, footprint, amount, [cx, cy], [gx, gy]);
                let (value, gradient) = self.gradient_at(input, q, moved);
                (value, warp_chain(gradient, [gx, gy], amount))
            }
            ref kernel => {
                let mut values = [Value::Scalar(0.0); 3];
                let mut gradients = [Vec3::ZERO; 3];
                let mut count = 0;
                for input in node.op.inputs().iter() {
                    (values[count], gradients[count]) = self.gradient_at(input, p, footprint);
                    count += 1;
                }
                let value = kernel.combine(p, footprint, &values[..count]);
                let gradient = kernel
                    .gradient(p, footprint, &values[..count], &gradients[..count])
                    .unwrap_or_else(|| self.numeric_gradient(id, value, p, footprint));
                (value, gradient)
            }
        }
    }

    /// Central differences of node `id` around `p`, zero for vector values.
    ///
    /// Planar nodes step along `x` and `y` only.
    pub(super) fn numeric_gradient(
        &self,
        id: NodeId,
        value: Value,
        p: Vec3,
        footprint: Footprint,
    ) -> Vec3 {
        if value.scalar().is_none() {
            return Vec3::ZERO;
        }
        let at = |q: Vec3| {
            self.eval_at(id, q, footprint)
                .scalar()
                .expect("a node's type does not depend on the point")
        };
        if self.nodes[id.0 as usize].space.planar().is_some() {
            let q = p.truncate();
            let h = (footprint.width() * 0.5).max(1e-4 * q.abs().max_element().max(1.0));
            let at2 = |r: Vec2| at(r.extend(p.z));
            return Vec3::new(
                (at2(q + Vec2::new(h, 0.0)) - at2(q - Vec2::new(h, 0.0))) / (2.0 * h),
                (at2(q + Vec2::new(0.0, h)) - at2(q - Vec2::new(0.0, h))) / (2.0 * h),
                0.0,
            );
        }
        let h = (footprint.width() * 0.5).max(1e-4 * p.abs().max_element().max(1.0));
        let axis = |e: Vec3| (at(p + e * h) - at(p - e * h)) / (2.0 * h);
        Vec3::new(axis(Vec3::X), axis(Vec3::Y), axis(Vec3::Z))
    }
}

/// The solid point a slice reads for planar point `p`.
fn slice_point(origin: Vec3, u: Vec3, v: Vec3, p: Vec3) -> Vec3 {
    origin + u * p.x + v * p.y
}

/// A slice's gradient chain rule: the planar gradient is `(u·∇, v·∇)`.
fn slice_chain(gradient: Vec3, u: Vec3, v: Vec3) -> Vec3 {
    Vec3::new(u.dot(gradient), v.dot(gradient), 0.0)
}

/// The point and footprint a warp's input is evaluated at, from the
/// displacements `center` at `p` and their gradients `rows`, the rows of the
/// displacement's Jacobian.
///
/// The footprint grows by `1 + |amount| · max_i Σ_j |∂d_i/∂p_j|`, a bound on
/// the warp's local stretch; a point footprint stays a point.
fn warp_move(
    p: Vec3,
    footprint: Footprint,
    amount: f32,
    center: [Value; 2],
    rows: [Vec3; 2],
) -> (Vec3, Footprint) {
    let scalar = |v: Value| v.scalar().expect("warp displacements are scalars");
    let q = (p.truncate() + Vec2::new(scalar(center[0]), scalar(center[1])) * amount).extend(p.z);
    if footprint.width() <= 0.0 {
        return (q, footprint);
    }
    let row = |i: usize| rows[i].x.abs() + rows[i].y.abs();
    let stretch = 1.0 + amount.abs() * row(0).max(row(1));
    let stretch = if stretch.is_finite() { stretch } else { 1.0 };
    (q, footprint.scaled(stretch))
}

/// A warp's gradient chain rule: `input(q(p))` with `q = p + a·d(p)` has
/// gradient `(I + a·J)ᵀ ∇input`, where `rows` are the rows of `J`. Warps are
/// planar, so only `x` and `y` take part.
fn warp_chain(gradient: Vec3, rows: [Vec3; 2], amount: f32) -> Vec3 {
    let (g, rows) = (gradient.truncate(), rows.map(Vec3::truncate));
    (g + (rows[0] * g.x + rows[1] * g.y) * amount).extend(0.0)
}

impl Kernel {
    /// The gradient of a pure scalar kernel's value from its operands'
    /// values and gradients, or `None` when it has no closed-form rule here.
    fn gradient(
        &self,
        p: Vec3,
        footprint: Footprint,
        values: &[Value],
        gradients: &[Vec3],
    ) -> Option<Vec3> {
        let s = |i: usize| values[i].scalar();
        let q = p.truncate();
        Some(match *self {
            Self::Constant(_) => Vec3::ZERO,
            Self::Noise(ref noise) => noise.eval_gradient(q, footprint).1.extend(0.0),
            Self::Fractal(ref fractal) => fractal.eval_gradient(q, footprint).1.extend(0.0),
            Self::Disk(ref disk) => disk.eval_gradient(q, footprint).1.extend(0.0),
            Self::Cellular(ref cellular) => cellular.eval_gradient(q, footprint).1.extend(0.0),
            Self::Tiling(ref tiling) => tiling.eval_gradient(q, footprint).1.extend(0.0),
            Self::Scatter(ref scatter) => scatter.eval_gradient(q, footprint).1.extend(0.0),
            Self::Sample(ref image) => {
                if !image.port().is_scalar() {
                    return None;
                }
                image.sample_gradient(q, footprint).1.extend(0.0)
            }
            Self::Noise3(ref noise) => noise.eval_gradient(p, footprint).1,
            Self::Fractal3(ref fractal) => fractal.eval_gradient(p, footprint).1,
            Self::Cellular3(ref cellular) => cellular.eval_gradient(p, footprint).1,
            Self::Fract(_) => {
                s(0)?;
                gradients[0]
            }
            Self::Atan2(..) => {
                let (y, x) = (s(0)?, s(1)?);
                let r2 = x * x + y * y;
                if r2 > 0.0 {
                    (gradients[0] * x - gradients[1] * y) / r2
                } else {
                    Vec3::ZERO
                }
            }
            Self::Binary(op, ..) => {
                let (a, b) = (s(0)?, s(1)?);
                let (ga, gb) = (gradients[0], gradients[1]);
                match op {
                    BinaryOp::Add => ga + gb,
                    BinaryOp::Sub => ga - gb,
                    BinaryOp::Mul => ga * b + gb * a,
                    BinaryOp::Min => {
                        if a <= b {
                            ga
                        } else {
                            gb
                        }
                    }
                    BinaryOp::Max => {
                        if a >= b {
                            ga
                        } else {
                            gb
                        }
                    }
                }
            }
            Self::Abs(_) => {
                let a = s(0)?;
                if a == 0.0 {
                    Vec3::ZERO
                } else {
                    gradients[0] * a.signum()
                }
            }
            Self::Clamp(_, min, max) => {
                let a = s(0)?;
                if a > min && a < max {
                    gradients[0]
                } else {
                    Vec3::ZERO
                }
            }
            Self::AsMask(_) => {
                let a = s(0)?;
                if a > 0.0 && a < 1.0 {
                    gradients[0]
                } else {
                    Vec3::ZERO
                }
            }
            Self::Remap { scale, .. } => gradients[0] * scale,
            Self::Mix(..) => {
                let (a, b, t) = (s(0)?, s(1)?, s(2)?);
                gradients[0] + (gradients[1] - gradients[0]) * t + gradients[2] * (b - a)
            }
            _ => return None,
        })
    }

    /// Evaluates a pure kernel from its operands' values, in operand order.
    fn combine(&self, p: Vec3, footprint: Footprint, args: &[Value]) -> Value {
        let scalar = |i: usize| args[i].scalar().expect("operand types were checked");
        let q = p.truncate();
        match *self {
            Self::Constant(value) => Value::Scalar(value),
            Self::Noise(ref noise) => Value::Scalar(noise.eval(q, footprint)),
            Self::Fractal(ref fractal) => Value::Scalar(fractal.eval(q, footprint)),
            Self::Cellular(ref cellular) => Value::Scalar(cellular.eval(q, footprint)),
            Self::Tiling(ref tiling) => Value::Scalar(tiling.eval(q, footprint)),
            Self::Scatter(ref scatter) => Value::Scalar(scatter.eval(q, footprint)),
            Self::Disk(ref disk) => Value::Scalar(disk.eval(q, footprint)),
            Self::Sample(ref image) => image.sample_value(q, footprint),
            Self::Noise3(ref noise) => Value::Scalar(noise.eval(p, footprint)),
            Self::Fractal3(ref fractal) => Value::Scalar(fractal.eval(p, footprint)),
            Self::Cellular3(ref cellular) => Value::Scalar(cellular.eval(p, footprint)),
            Self::Position3 => Value::Vector3(p),
            Self::Length(_) => Value::Scalar(match args[0] {
                Value::Vector2(v) => v.length(),
                Value::Vector3(v) => v.length(),
                _ => unreachable!("length input was type-checked"),
            }),
            Self::Fract(_) => {
                let x = scalar(0);
                Value::Scalar(x - libm::floorf(x))
            }
            Self::Atan2(..) => Value::Scalar(libm::atan2f(scalar(0), scalar(1))),
            Self::Binary(op, ..) => componentwise(args[0], args[1], |a, b| match op {
                BinaryOp::Add => a + b,
                BinaryOp::Sub => a - b,
                BinaryOp::Mul => a * b,
                BinaryOp::Min => a.min(b),
                BinaryOp::Max => a.max(b),
            }),
            Self::Abs(_) => Value::Scalar(scalar(0).abs()),
            Self::Clamp(_, min, max) => Value::Scalar(scalar(0).clamp(min, max)),
            Self::Remap {
                scale, from, to, ..
            } => Value::Scalar(to + (scalar(0) - from) * scale),
            Self::Mix(..) => {
                let t = scalar(2);
                componentwise(args[0], args[1], |a, b| a + (b - a) * t)
            }
            Self::Vector2(..) => Value::Vector2(Vec2::new(scalar(0), scalar(1))),
            Self::Vector3(..) => Value::Vector3(Vec3::new(scalar(0), scalar(1), scalar(2))),
            Self::Component(_, index) => Value::Scalar(
                args[0]
                    .component(index)
                    .expect("component index was checked"),
            ),
            Self::AsMask(_) => Value::Scalar(scalar(0).clamp(0.0, 1.0)),
            Self::ToId(_, levels) => {
                let v = scalar(0);
                let v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "v is in [0, 1], so the product is in [0, levels]"
                )]
                let id = libm::floorf(v * levels as f32) as u32;
                Value::Id(id.min(levels - 1))
            }
            Self::Normalize(_) => {
                let Value::Vector3(v) = args[0] else {
                    unreachable!("normalize input was type-checked");
                };
                Value::Vector3(v.try_normalize().unwrap_or(Vec3::Z))
            }
            Self::Direction(_) => {
                let twice = 2.0 * scalar(0);
                Value::Vector2(Vec2::new(libm::cosf(twice), libm::sinf(twice)))
            }
            Self::Angle(_) => {
                let Value::Vector2(v) = args[0] else {
                    unreachable!("angle input was type-checked");
                };
                let mut angle = 0.5 * libm::atan2f(v.y, v.x);
                if angle < 0.0 {
                    angle += core::f32::consts::PI;
                }
                Value::Scalar(if v == Vec2::ZERO { 0.0 } else { angle })
            }
            Self::Coherence(_) => {
                let Value::Vector2(v) = args[0] else {
                    unreachable!("coherence input was type-checked");
                };
                Value::Scalar(v.length().min(1.0))
            }
            Self::Transform { .. }
            | Self::Transform3 { .. }
            | Self::Slice { .. }
            | Self::Pass(_)
            | Self::Warp { .. } => {
                unreachable!("only pure kernels combine")
            }
        }
    }
}

/// Applies `f` per component; a scalar operand broadcasts over a vector.
fn componentwise(a: Value, b: Value, f: impl Fn(f32, f32) -> f32) -> Value {
    match (a, b) {
        (Value::Scalar(a), Value::Scalar(b)) => Value::Scalar(f(a, b)),
        (Value::Vector2(a), Value::Vector2(b)) => {
            Value::Vector2(Vec2::new(f(a.x, b.x), f(a.y, b.y)))
        }
        (Value::Vector3(a), Value::Vector3(b)) => {
            Value::Vector3(Vec3::new(f(a.x, b.x), f(a.y, b.y), f(a.z, b.z)))
        }
        (Value::Scalar(s), Value::Vector2(v)) => Value::Vector2(Vec2::new(f(s, v.x), f(s, v.y))),
        (Value::Vector2(v), Value::Scalar(s)) => Value::Vector2(Vec2::new(f(v.x, s), f(v.y, s))),
        (Value::Scalar(s), Value::Vector3(v)) => {
            Value::Vector3(Vec3::new(f(s, v.x), f(s, v.y), f(s, v.z)))
        }
        (Value::Vector3(v), Value::Scalar(s)) => {
            Value::Vector3(Vec3::new(f(v.x, s), f(v.y, s), f(v.z, s)))
        }
        _ => unreachable!("operand types were checked"),
    }
}

impl ScalarField for FieldProgram {
    fn domain(&self) -> Domain {
        self.nodes[self.output.0 as usize]
            .space
            .planar()
            .expect("finish checks the output is planar")
    }

    fn eval(&self, p: Vec2, footprint: Footprint) -> f32 {
        let stats = self.plan.stats();
        if stats.instances < stats.tree_evaluations {
            self.evaluator().eval(p, footprint)
        } else {
            self.eval_node(self.output, p, footprint)
        }
    }

    /// Evaluates through one [`Evaluator`] when the plan saves work, so a
    /// batch allocates its buffers once.
    fn eval_batch(&self, points: &[Vec2], footprint: Footprint, out: &mut [f32]) {
        assert!(out.len() >= points.len(), "output shorter than the points");
        let stats = self.plan.stats();
        if stats.instances < stats.tree_evaluations {
            let mut evaluator = self.evaluator();
            for (value, &p) in out.iter_mut().zip(points) {
                *value = evaluator.eval(p, footprint);
            }
        } else {
            for (value, &p) in out.iter_mut().zip(points) {
                *value = self.eval_node(self.output, p, footprint);
            }
        }
    }
    /// Analytic where the program's ops allow; see
    /// [`FieldProgram::eval_node_gradient`].
    fn eval_gradient(&self, p: Vec2, footprint: Footprint) -> (f32, Vec2) {
        self.eval_node_gradient(self.output, p, footprint)
    }
}

/// A finished program whose output may have any [`PortType`].
///
/// Built with [`ProgramBuilder::finish_value`]. [`Self::channel`] gives one
/// component as a scalar [`FieldProgram`], for realizing colors and normals
/// channel by channel.
#[derive(Clone, Debug, PartialEq)]
pub struct ValueProgram {
    program: FieldProgram,
}

impl ValueProgram {
    /// The program's nodes, fingerprints and per-node evaluation.
    #[must_use]
    pub const fn program(&self) -> &FieldProgram {
        &self.program
    }

    /// The output's type.
    #[must_use]
    pub fn output_type(&self) -> PortType {
        self.program.nodes[self.program.output.0 as usize].port
    }

    /// The output's domain.
    #[must_use]
    pub fn domain(&self) -> Domain {
        self.program.nodes[self.program.output.0 as usize]
            .space
            .planar()
            .expect("finish_value checks the output is planar")
    }

    /// The program's content fingerprint: its output node's.
    #[must_use]
    pub fn fingerprint(&self) -> Fingerprint {
        self.program.fingerprint()
    }

    /// Evaluates the output at `p`.
    #[must_use]
    pub fn eval(&self, p: Vec2, footprint: Footprint) -> Value {
        self.program.evaluator().eval_value(p, footprint)
    }

    /// An evaluator of the output, reusing its buffers across points.
    #[must_use]
    pub fn evaluator(&self) -> Evaluator<'_> {
        self.program.evaluator()
    }

    /// Component `index` of the output as a scalar program: the output
    /// itself for a scalar or mask (index 0), or a new
    /// [`Op::Component`] node.
    pub fn channel(&self, index: u8) -> Result<FieldProgram, ProgramError> {
        let output = self.program.output;
        let found = self.output_type();
        if found.is_scalar() && index == 0 {
            return Ok(self.program.clone());
        }
        let mut builder = ProgramBuilder {
            nodes: self.program.nodes.clone(),
        };
        let component = builder.add(Op::Component {
            input: output,
            index,
        })?;
        builder.finish(component)
    }
}

/// A finished solid program: a scalar field over a [`Domain3`].
///
/// The same IR as [`FieldProgram`], finished at a solid output by
/// [`ProgramBuilder::finish_solid`]. It is a [`SolidField`], and
/// [`Self::eval_chart`] evaluates it at surface points given in its solid
/// space, the seam through which charted surfaces bake solid materials.
#[derive(Clone, Debug, PartialEq)]
pub struct SolidProgram {
    program: FieldProgram,
}

/// One surface point for [`SolidProgram::eval_chart`]: where a texel lands in
/// the solid, and the footprint the texel covers there.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ChartSample {
    /// The texel's point in the program's solid space.
    pub position: Vec3,
    /// The side of the solid region the texel stands for, in domain units.
    pub footprint: Footprint,
}

impl SolidProgram {
    /// The underlying program: its nodes, fingerprints and bounds.
    #[must_use]
    pub const fn program(&self) -> &FieldProgram {
        &self.program
    }

    /// The program's content fingerprint.
    #[must_use]
    pub fn fingerprint(&self) -> Fingerprint {
        self.program.fingerprint()
    }

    /// Static bounds of the output; `slope` bounds `|∂f/∂x| + |∂f/∂y| + |∂f/∂z|`.
    #[must_use]
    pub fn bounds(&self) -> StaticBounds {
        self.program.bounds()
    }

    /// Evaluates the program at each chart sample into `out`, through the
    /// flat plan, bit-identical to [`SolidField::eval`] at each sample.
    ///
    /// # Panics
    ///
    /// Panics if `out` is shorter than `samples`.
    pub fn eval_chart(&self, samples: &[ChartSample], out: &mut [f32]) {
        assert!(
            out.len() >= samples.len(),
            "output shorter than the samples"
        );
        let mut evaluator = self.program.evaluator();
        for (value, sample) in out.iter_mut().zip(samples) {
            *value = evaluator
                .eval_value_at(sample.position, sample.footprint)
                .scalar()
                .expect("finish_solid checks the output is scalar");
        }
    }
}

impl SolidField for SolidProgram {
    fn domain(&self) -> Domain3 {
        self.program.nodes[self.program.output.0 as usize]
            .space
            .solid()
            .expect("finish_solid checks the output is solid")
    }

    fn eval(&self, p: Vec3, footprint: Footprint) -> f32 {
        self.program
            .eval_at(self.program.output, p, footprint)
            .scalar()
            .expect("finish_solid checks the output is scalar")
    }

    fn eval_gradient(&self, p: Vec3, footprint: Footprint) -> (f32, Vec3) {
        let (value, gradient) = self.program.gradient_at(self.program.output, p, footprint);
        (
            value
                .scalar()
                .expect("finish_solid checks the output is scalar"),
            gradient,
        )
    }
}

/// The type of a node computing `op`, given its operands' types.
fn derive_port(op: &Op, port: &dyn Fn(NodeId) -> PortType) -> Result<PortType, ProgramError> {
    let name = op.name();
    let mismatch = |found: PortType, reason: &'static str| ProgramError::TypeMismatch {
        op: name,
        found,
        reason,
    };
    let scalar = |id: NodeId| {
        let found = port(id);
        if found.is_scalar() {
            Ok(found)
        } else {
            Err(mismatch(found, "needs a scalar or mask"))
        }
    };
    // Values that componentwise arithmetic treats as plain numbers.
    let arithmetic = |found: PortType| match found {
        PortType::Normal(_) => Err(mismatch(
            found,
            "normals combine only through blend-normals",
        )),
        PortType::Direction if !matches!(op, Op::Mix { .. }) => {
            Err(mismatch(found, "directions only blend, with mix"))
        }
        PortType::Id => Err(mismatch(found, "identifiers are never combined")),
        _ => Ok(found),
    };
    let both_masks = |a: PortType, b: PortType| {
        if a == PortType::Mask && b == PortType::Mask {
            PortType::Mask
        } else {
            PortType::Scalar
        }
    };
    Ok(match *op {
        Op::Constant { .. }
        | Op::Noise { .. }
        | Op::Fractal { .. }
        | Op::Cellular { .. }
        | Op::Tiling { .. }
        | Op::Scatter { .. }
        | Op::Constant3 { .. }
        | Op::Noise3 { .. }
        | Op::Fractal3 { .. }
        | Op::Cellular3 { .. } => PortType::Scalar,
        Op::Position3 => PortType::Vector3,
        Op::Transform3 { input, transform } => {
            let found = port(input);
            let directional = matches!(
                found,
                PortType::Vector2 | PortType::Vector3 | PortType::Normal(_) | PortType::Direction
            );
            if directional && transform.matrix != Mat3::IDENTITY {
                return Err(mismatch(
                    found,
                    "a rotating or scaling transform would leave directions unrotated",
                ));
            }
            found
        }
        Op::Slice { input, .. } => {
            let found = port(input);
            if matches!(
                found,
                PortType::Vector2 | PortType::Vector3 | PortType::Normal(_) | PortType::Direction
            ) {
                return Err(mismatch(
                    found,
                    "directional values keep their solid frame and cannot be sliced",
                ));
            }
            found
        }
        Op::Length { input } => {
            let found = port(input);
            if !matches!(found, PortType::Vector2 | PortType::Vector3) {
                return Err(mismatch(found, "needs a vector2 or vector3"));
            }
            PortType::Scalar
        }
        Op::Fract { input } => {
            scalar(input)?;
            PortType::Scalar
        }
        Op::Atan2 { y, x } => {
            scalar(y)?;
            scalar(x)?;
            PortType::Scalar
        }
        Op::Disk { .. } => PortType::Mask,
        Op::Sample { ref image } => image.port(),
        Op::Transform { input, transform } => {
            let found = port(input);
            let directional = matches!(
                found,
                PortType::Vector2 | PortType::Vector3 | PortType::Normal(_) | PortType::Direction
            );
            if directional && transform.matrix != Mat2::IDENTITY {
                return Err(mismatch(
                    found,
                    "a rotating or scaling transform would leave directions unrotated",
                ));
            }
            found
        }
        Op::Demote { input } => port(input),
        Op::Add { a, b } | Op::Sub { a, b } => {
            let (ta, tb) = (arithmetic(port(a))?, arithmetic(port(b))?);
            if ta.is_scalar() && tb.is_scalar() {
                PortType::Scalar
            } else if ta == tb {
                ta
            } else {
                return Err(mismatch(tb, "operands must have the same type"));
            }
        }
        Op::Mul { a, b } | Op::Min { a, b } | Op::Max { a, b } => {
            let (ta, tb) = (arithmetic(port(a))?, arithmetic(port(b))?);
            if ta.is_scalar() && tb.is_scalar() {
                both_masks(ta, tb)
            } else if ta == tb {
                ta
            } else if matches!(op, Op::Mul { .. }) && ta.is_scalar() {
                tb
            } else if matches!(op, Op::Mul { .. }) && tb.is_scalar() {
                ta
            } else {
                return Err(mismatch(tb, "operands must have the same type"));
            }
        }
        Op::Abs { input } | Op::Clamp { input, .. } | Op::Remap { input, .. } => {
            scalar(input)?;
            PortType::Scalar
        }
        Op::Mix { a, b, t } => {
            let tt = scalar(t)?;
            let (ta, tb) = (arithmetic(port(a))?, arithmetic(port(b))?);
            if ta.is_scalar() && tb.is_scalar() {
                if tt == PortType::Mask {
                    both_masks(ta, tb)
                } else {
                    PortType::Scalar
                }
            } else if ta == tb {
                ta
            } else {
                return Err(mismatch(tb, "operands must have the same type"));
            }
        }
        Op::Warp { input, dx, dy, .. } => {
            scalar(dx)?;
            scalar(dy)?;
            port(input)
        }
        Op::Vector2 { x, y } => {
            scalar(x)?;
            scalar(y)?;
            PortType::Vector2
        }
        Op::Vector3 { x, y, z } => {
            scalar(x)?;
            scalar(y)?;
            scalar(z)?;
            PortType::Vector3
        }
        Op::Color { r, g, b } => {
            scalar(r)?;
            scalar(g)?;
            scalar(b)?;
            PortType::Color(Primaries::Rec709)
        }
        Op::Component { input, index } => {
            let found = port(input);
            if found.components() < 2 {
                return Err(mismatch(found, "needs a vector, color or normal"));
            }
            if usize::from(index) >= found.components() {
                return Err(mismatch(found, "component index out of range"));
            }
            PortType::Scalar
        }
        Op::AsMask { input } => {
            scalar(input)?;
            PortType::Mask
        }
        Op::ToId { input, .. } => {
            scalar(input)?;
            PortType::Id
        }
        Op::Normalize { input } => {
            let found = port(input);
            if found != PortType::Vector3 {
                return Err(mismatch(found, "needs a vector3"));
            }
            PortType::Normal(NormalFrame::Domain)
        }
        Op::Direction { angle } => {
            scalar(angle)?;
            PortType::Direction
        }
        Op::Angle { input } | Op::Coherence { input } => {
            let found = port(input);
            if found != PortType::Direction {
                return Err(mismatch(found, "needs a direction"));
            }
            if matches!(op, Op::Angle { .. }) {
                PortType::Scalar
            } else {
                PortType::Mask
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::{Grid, Region};
    use crate::{Transformed, central_difference};
    use glam::Mat2;

    fn torus() -> Domain {
        Domain::periodic(2, 1).unwrap()
    }

    /// A bark-like program exercising every op kind once.
    fn bark(b: &mut ProgramBuilder) -> NodeId {
        let d = torus();
        let cells = b
            .add(Op::Cellular {
                domain: d,
                frequency: [4.0, 12.0],
                jitter: 0.9,
                seed: 3,
                output: CellOutput::Border,
            })
            .unwrap();
        let fbm = b
            .add(Op::Fractal {
                basis: Basis::Gradient,
                domain: d,
                frequency: [2.0, 4.0],
                seed: 5,
                params: FractalParams::default(),
            })
            .unwrap();
        let noise = b
            .add(Op::Noise {
                basis: Basis::Value,
                domain: d,
                frequency: [3.0, 3.0],
                seed: 9,
            })
            .unwrap();
        let warped = b
            .add(Op::Warp {
                input: cells,
                dx: fbm,
                dy: noise,
                amount: 0.05,
            })
            .unwrap();
        let shifted = b
            .add(Op::Transform {
                input: warped,
                transform: Affine2 {
                    matrix: Mat2::from_diagonal(Vec2::new(1.0, 2.0)),
                    translation: Vec2::new(0.25, 0.0),
                },
            })
            .unwrap();
        let ridges = b.add(Op::Abs { input: fbm }).unwrap();
        let depth = b
            .add(Op::Remap {
                input: ridges,
                from: [0.0, 1.0],
                to: [0.2, 1.0],
            })
            .unwrap();
        let grooved = b
            .add(Op::Min {
                a: shifted,
                b: depth,
            })
            .unwrap();
        let half = b
            .add(Op::Constant {
                domain: d,
                value: 0.5,
            })
            .unwrap();
        let blend = b
            .add(Op::Mix {
                a: grooved,
                b: noise,
                t: half,
            })
            .unwrap();
        let sum = b.add(Op::Add { a: blend, b: half }).unwrap();
        let diff = b.add(Op::Sub { a: sum, b: noise }).unwrap();
        let prod = b.add(Op::Mul { a: diff, b: half }).unwrap();
        let top = b
            .add(Op::Max {
                a: prod,
                b: grooved,
            })
            .unwrap();
        b.add(Op::Clamp {
            input: top,
            min: 0.0,
            max: 1.0,
        })
        .unwrap()
    }

    #[test]
    fn programs_evaluate_like_the_direct_fields() {
        let d = torus();
        let mut b = ProgramBuilder::new();
        let fbm = b
            .add(Op::Fractal {
                basis: Basis::Gradient,
                domain: d,
                frequency: [3.0, 5.0],
                seed: 3,
                params: FractalParams::default(),
            })
            .unwrap();
        let transform = Affine2 {
            matrix: Mat2::from_diagonal(Vec2::new(2.0, 3.0)),
            translation: Vec2::new(0.1, 0.2),
        };
        let moved = b
            .add(Op::Transform {
                input: fbm,
                transform,
            })
            .unwrap();
        let program = b.finish(moved).unwrap();

        let direct = Transformed::new(
            Fractal::new(
                Basis::Gradient,
                d,
                Vec2::new(3.0, 5.0),
                3,
                FractalParams::default(),
            )
            .unwrap(),
            transform,
        )
        .unwrap();
        let region = Region::period(d).unwrap();
        assert_eq!(
            Grid::sample(&program, region, 32, 16),
            Grid::sample(&direct, region, 32, 16)
        );
    }

    #[test]
    fn domains_are_checked() {
        let mut b = ProgramBuilder::new();
        let periodic = b
            .add(Op::Constant {
                domain: torus(),
                value: 1.0,
            })
            .unwrap();
        let plane = b
            .add(Op::Constant {
                domain: Domain::Plane,
                value: 1.0,
            })
            .unwrap();
        assert_eq!(
            b.add(Op::Add {
                a: periodic,
                b: plane
            }),
            Err(ProgramError::DomainMismatch {
                first: torus(),
                other: Domain::Plane
            })
        );
        let demoted = b.add(Op::Demote { input: periodic }).unwrap();
        assert!(
            b.add(Op::Add {
                a: demoted,
                b: plane
            })
            .is_ok()
        );
        assert_eq!(
            b.add(Op::Transform {
                input: periodic,
                transform: Affine2::scale(Vec2::splat(1.5)),
            }),
            Err(ProgramError::Domain(DomainError::NotLatticePreserving))
        );
        assert_eq!(
            b.add(Op::Abs { input: NodeId(99) }),
            Err(ProgramError::UnknownNode { node: NodeId(99) })
        );
        assert!(
            b.add(Op::Remap {
                input: plane,
                from: [1.0, 1.0],
                to: [0.0, 1.0]
            })
            .is_err()
        );
    }

    #[test]
    fn fingerprints_follow_content_not_identity() {
        let mut a = ProgramBuilder::new();
        let out_a = bark(&mut a);
        let a = a.finish(out_a).unwrap();

        // Unused nodes and different creation positions do not matter.
        let mut b = ProgramBuilder::new();
        b.add(Op::Constant {
            domain: Domain::Plane,
            value: 3.0,
        })
        .unwrap();
        let out_b = bark(&mut b);
        let b = b.finish(out_b).unwrap();
        assert_ne!(out_a, out_b);
        assert_eq!(a.fingerprint(), b.fingerprint());

        // Every parameter participates.
        let mut c = ProgramBuilder::new();
        let n = c
            .add(Op::Noise {
                basis: Basis::Value,
                domain: torus(),
                frequency: [3.0, 3.0],
                seed: 9,
            })
            .unwrap();
        let base = c.finish(n).unwrap().fingerprint();
        for changed in [
            Op::Noise {
                basis: Basis::Gradient,
                domain: torus(),
                frequency: [3.0, 3.0],
                seed: 9,
            },
            Op::Noise {
                basis: Basis::Value,
                domain: Domain::periodic(1, 1).unwrap(),
                frequency: [3.0, 3.0],
                seed: 9,
            },
            Op::Noise {
                basis: Basis::Value,
                domain: torus(),
                frequency: [3.0, 6.0],
                seed: 9,
            },
            Op::Noise {
                basis: Basis::Value,
                domain: torus(),
                frequency: [3.0, 3.0],
                seed: 10,
            },
        ] {
            let mut c = ProgramBuilder::new();
            let n = c.add(changed).unwrap();
            assert_ne!(c.finish(n).unwrap().fingerprint(), base);
        }

        // Operand order matters for non-commutative ops.
        let mut d = ProgramBuilder::new();
        let one = d
            .add(Op::Constant {
                domain: Domain::Plane,
                value: 1.0,
            })
            .unwrap();
        let two = d
            .add(Op::Constant {
                domain: Domain::Plane,
                value: 2.0,
            })
            .unwrap();
        let one_two = d.add(Op::Sub { a: one, b: two }).unwrap();
        let two_one = d.add(Op::Sub { a: two, b: one }).unwrap();
        let program = d.finish(one_two).unwrap();
        assert_ne!(
            program.node_fingerprint(one_two),
            program.node_fingerprint(two_one)
        );
    }

    /// Pinned fingerprint and output digest of [`bark`]. A change here means
    /// persisted fingerprints or realized textures change: bump
    /// [`FINGERPRINT_VERSION`] for encoding changes, and document field
    /// definition changes.
    #[test]
    fn golden_fingerprint_and_digest() {
        let mut b = ProgramBuilder::new();
        let out = bark(&mut b);
        let program = b.finish(out).unwrap();
        let digest = Grid::sample(&program, Region::period(torus()).unwrap(), 32, 16).digest();
        assert_eq!(
            (program.fingerprint(), digest),
            (Fingerprint(GOLDEN_FINGERPRINT), GOLDEN_DIGEST),
            "got fingerprint {} and digest {digest:#018x}",
            program.fingerprint()
        );
    }

    const GOLDEN_FINGERPRINT: u128 = 0xbdeb_9cd9_4501_cd3c_437b_7bc6_5078_0875;
    const GOLDEN_DIGEST: u64 = 0x90ff_6a69_e1da_9f83;

    #[test]
    fn listing_shows_every_node() {
        let mut b = ProgramBuilder::new();
        let out = bark(&mut b);
        let program = b.finish(out).unwrap();
        let names: Vec<_> = program.nodes().map(|(_, op)| op.name()).collect();
        assert_eq!(names.len(), program.len());
        for name in [
            "constant",
            "noise",
            "fractal",
            "cellular",
            "transform",
            "add",
            "sub",
            "mul",
            "min",
            "max",
            "abs",
            "clamp",
            "remap",
            "mix",
            "warp",
        ] {
            assert!(names.contains(&name), "{name} missing from {names:?}");
        }
        assert_eq!(program.node_domain(out), Some(torus()));
    }

    fn noise(b: &mut ProgramBuilder, seed: u64) -> NodeId {
        b.add(Op::Noise {
            basis: Basis::Gradient,
            domain: torus(),
            frequency: [4.0, 4.0],
            seed,
        })
        .unwrap()
    }

    #[test]
    fn colors_and_components_round_trip() {
        let mut b = ProgramBuilder::new();
        let (r, g, bl) = (noise(&mut b, 1), noise(&mut b, 2), noise(&mut b, 3));
        let color = b.add(Op::Color { r, g, b: bl }).unwrap();
        assert_eq!(b.port_type(color), Ok(PortType::Color(Primaries::Rec709)));
        let vector = b.add(Op::Vector3 { x: r, y: g, z: bl }).unwrap();
        assert_ne!(
            b.nodes[color.0 as usize].fingerprint, b.nodes[vector.0 as usize].fingerprint,
            "a color is not a vector"
        );
        let direct = Noise::new(Basis::Gradient, torus(), Vec2::splat(4.0), 2).unwrap();
        let program = b.finish_value(color).unwrap();
        assert_eq!(program.output_type(), PortType::Color(Primaries::Rec709));
        let green = program.channel(1).unwrap();
        for p in [Vec2::new(0.1, 0.2), Vec2::new(1.7, 0.4)] {
            assert_eq!(
                green.eval(p, Footprint::POINT).to_bits(),
                direct.eval(p, Footprint::POINT).to_bits()
            );
            let Value::Vector3(c) = program.eval(p, Footprint::POINT) else {
                panic!("colors evaluate to three components");
            };
            assert_eq!(c.y.to_bits(), direct.eval(p, Footprint::POINT).to_bits());
        }
        assert!(matches!(
            program.channel(3),
            Err(ProgramError::TypeMismatch {
                op: "component",
                ..
            })
        ));
    }

    #[test]
    fn type_rules_reject_misuse() {
        let mut b = ProgramBuilder::new();
        let n = noise(&mut b, 1);
        let half = b
            .add(Op::Constant {
                domain: torus(),
                value: 0.5,
            })
            .unwrap();
        let v = b
            .add(Op::Vector3 {
                x: n,
                y: n,
                z: half,
            })
            .unwrap();
        let normal = b.add(Op::Normalize { input: v }).unwrap();
        assert_eq!(
            b.port_type(normal),
            Ok(PortType::Normal(NormalFrame::Domain))
        );
        // Normals never lerp.
        assert!(matches!(
            b.add(Op::Mix {
                a: normal,
                b: normal,
                t: half
            }),
            Err(ProgramError::TypeMismatch { op: "mix", .. })
        ));
        // A vector is not a blend weight.
        assert!(matches!(
            b.add(Op::Mix {
                a: n,
                b: half,
                t: v
            }),
            Err(ProgramError::TypeMismatch { op: "mix", .. })
        ));
        // Rotating a directional field would leave its values unrotated.
        let rotate = Affine2 {
            matrix: Mat2::from_cols(Vec2::Y, -Vec2::X),
            translation: Vec2::ZERO,
        };
        assert!(matches!(
            b.add(Op::Transform {
                input: normal,
                transform: rotate
            }),
            Err(ProgramError::TypeMismatch {
                op: "transform",
                ..
            })
        ));
        let shift = Affine2 {
            matrix: Mat2::IDENTITY,
            translation: Vec2::new(0.5, 0.0),
        };
        assert!(
            b.add(Op::Transform {
                input: normal,
                transform: shift
            })
            .is_ok()
        );
        // Identifiers never blend.
        let id = b
            .add(Op::ToId {
                input: half,
                levels: 4,
            })
            .unwrap();
        assert_eq!(b.port_type(id), Ok(PortType::Id));
        assert!(matches!(
            b.add(Op::Add { a: id, b: id }),
            Err(ProgramError::TypeMismatch { op: "add", .. })
        ));
        assert!(
            b.add(Op::ToId {
                input: half,
                levels: 0
            })
            .is_err()
        );
        // Mismatched vectors do not add; a scalar scales a vector.
        let v2 = b.add(Op::Vector2 { x: n, y: n }).unwrap();
        assert!(b.add(Op::Add { a: v, b: v2 }).is_err());
        let scaled = b.add(Op::Mul { a: half, b: v }).unwrap();
        assert_eq!(b.port_type(scaled), Ok(PortType::Vector3));
        assert!(b.add(Op::Min { a: half, b: v }).is_err());
        // Scalar programs need a scalar output.
        assert_eq!(
            b.clone().finish(v),
            Err(ProgramError::OutputType {
                found: PortType::Vector3
            })
        );
    }

    #[test]
    fn masks_stay_masks_only_when_in_range() {
        let mut b = ProgramBuilder::new();
        let n = noise(&mut b, 1);
        let m = b.add(Op::AsMask { input: n }).unwrap();
        let product = b.add(Op::Mul { a: m, b: m }).unwrap();
        assert_eq!(b.port_type(product), Ok(PortType::Mask));
        let sum = b.add(Op::Add { a: m, b: m }).unwrap();
        assert_eq!(b.port_type(sum), Ok(PortType::Scalar));
        let mixed = b.add(Op::Mix { a: m, b: m, t: m }).unwrap();
        assert_eq!(b.port_type(mixed), Ok(PortType::Mask));
        let loose = b.add(Op::Mix { a: m, b: m, t: n }).unwrap();
        assert_eq!(b.port_type(loose), Ok(PortType::Scalar));
        let program = b.finish(product).unwrap();
        for i in 0..32 {
            let v = program.eval(Vec2::new(i as f32 * 0.13, 0.4), Footprint::POINT);
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn identifiers_quantize() {
        let mut b = ProgramBuilder::new();
        let d = torus();
        let ids: Vec<_> = [-1.0, 0.0, 0.49, 0.5, 1.0, 7.0]
            .into_iter()
            .map(|value| {
                let c = b.add(Op::Constant { domain: d, value }).unwrap();
                b.add(Op::ToId {
                    input: c,
                    levels: 2,
                })
                .unwrap()
            })
            .collect();
        let program = b.finish_value(ids[0]).unwrap();
        let values: Vec<_> = ids
            .iter()
            .map(|&id| {
                program
                    .program()
                    .eval_value(id, Vec2::ZERO, Footprint::POINT)
            })
            .collect();
        assert_eq!(values, [0, 0, 0, 1, 1, 1].map(Value::Id));
    }

    #[test]
    fn flat_plans_match_recursion_bit_for_bit() {
        let mut b = ProgramBuilder::new();
        let out = bark(&mut b);
        let program = b.finish(out).unwrap();
        let mut evaluator = program.evaluator();
        for i in 0..64 {
            let p = Vec2::new(i as f32 * 0.071, (i * 7 % 13) as f32 * 0.09);
            for footprint in [Footprint::POINT, Footprint::new(0.02).unwrap()] {
                let recursive = program.eval_node(program.output(), p, footprint);
                assert_eq!(evaluator.eval(p, footprint).to_bits(), recursive.to_bits());
                assert_eq!(program.eval(p, footprint).to_bits(), recursive.to_bits());
            }
        }
    }

    #[test]
    fn shared_subgraphs_evaluate_once_per_context() {
        let d = torus();
        let mut b = ProgramBuilder::new();
        let n = noise(&mut b, 1);
        // `n` feeds both operands of both products, and the sum.
        let square = b.add(Op::Mul { a: n, b: n }).unwrap();
        let sum = b
            .add(Op::Add {
                a: square,
                b: square,
            })
            .unwrap();
        let shift = Affine2 {
            matrix: Mat2::IDENTITY,
            translation: Vec2::new(0.25, 0.0),
        };
        let moved = b
            .add(Op::Transform {
                input: sum,
                transform: shift,
            })
            .unwrap();
        let total = b.add(Op::Add { a: sum, b: moved }).unwrap();
        let program = b.finish(total).unwrap();
        assert_eq!(
            program.evaluation_stats(),
            EvaluationStats {
                // n, square, sum at the output's point and at the shifted
                // one, plus the total.
                instances: 7,
                // total + 2 × (sum + 2 × (square + 2 × n)).
                tree_evaluations: 1 + 2 * (1 + 2 * (1 + 2)),
                contexts: 2,
            }
        );
        let direct = Noise::new(Basis::Gradient, d, Vec2::splat(4.0), 1).unwrap();
        let p = Vec2::new(0.3, 0.6);
        let at = |q: Vec2| {
            let v = direct.eval(q, Footprint::POINT);
            (v * v) + (v * v)
        };
        assert_eq!(
            program.eval(p, Footprint::POINT).to_bits(),
            (at(p) + at(p + Vec2::new(0.25, 0.0))).to_bits()
        );

        // A program without sharing keeps equal counts.
        let mut b = ProgramBuilder::new();
        let n = noise(&mut b, 2);
        let program = b.finish(n).unwrap();
        let stats = program.evaluation_stats();
        assert_eq!((stats.instances, stats.tree_evaluations), (1, 1));
    }

    #[test]
    fn value_programs_evaluate_through_the_plan() {
        let mut b = ProgramBuilder::new();
        let n = noise(&mut b, 4);
        let color = b.add(Op::Color { r: n, g: n, b: n }).unwrap();
        let program = b.finish_value(color).unwrap();
        assert_eq!(program.program().evaluation_stats().instances, 2);
        let p = Vec2::new(0.9, 0.1);
        assert_eq!(
            program.eval(p, Footprint::POINT),
            program.program().eval_value(color, p, Footprint::POINT)
        );
    }

    #[test]
    fn batches_match_single_evaluations() {
        let mut b = ProgramBuilder::new();
        let n = noise(&mut b, 6);
        let square = b.add(Op::Mul { a: n, b: n }).unwrap();
        let shared = b.add(Op::Add { a: square, b: n }).unwrap();
        let program = b.finish(shared).unwrap();
        assert!(program.evaluation_stats().instances < program.evaluation_stats().tree_evaluations);
        let points: Vec<Vec2> = (0..40).map(|i| Vec2::new(i as f32 * 0.051, 0.3)).collect();
        let footprint = Footprint::new(0.01).unwrap();
        let mut out = alloc::vec![0.0; points.len()];
        program.eval_batch(&points, footprint, &mut out);
        for (&p, v) in points.iter().zip(&out) {
            assert_eq!(v.to_bits(), program.eval(p, footprint).to_bits());
        }
    }

    #[test]
    fn directions_are_undirected_and_blend_sign_free() {
        use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};
        let d = torus();
        let mut b = ProgramBuilder::new();
        let constant =
            |b: &mut ProgramBuilder, value| b.add(Op::Constant { domain: d, value }).unwrap();
        let direction = |b: &mut ProgramBuilder, angle| {
            let a = constant(b, angle);
            b.add(Op::Direction { angle: a }).unwrap()
        };
        let east = direction(&mut b, 0.0);
        let west = direction(&mut b, PI);
        let north = direction(&mut b, FRAC_PI_2);
        let half = constant(&mut b, 0.5);
        // East and west are one axis: blending keeps it, fully coherent.
        let same = b
            .add(Op::Mix {
                a: east,
                b: west,
                t: half,
            })
            .unwrap();
        // East and north are perpendicular: the blend cancels.
        let crossed = b
            .add(Op::Mix {
                a: east,
                b: north,
                t: half,
            })
            .unwrap();
        let diagonal = direction(&mut b, FRAC_PI_4 + PI);
        let eval = |b: &ProgramBuilder, id, op: fn(NodeId) -> Op| {
            let mut b = b.clone();
            let out = b.add(op(id)).unwrap();
            b.finish(out).unwrap().eval(Vec2::ZERO, Footprint::POINT)
        };
        let angle = |input| Op::Angle { input };
        let coherence = |input| Op::Coherence { input };
        assert!(eval(&b, same, angle).abs() < 1e-6);
        assert!((eval(&b, same, coherence) - 1.0).abs() < 1e-6);
        assert!(eval(&b, crossed, coherence) < 1e-6);
        assert!((eval(&b, diagonal, angle) - FRAC_PI_4).abs() < 1e-6);
        assert_eq!(b.port_type(same), Ok(PortType::Direction));
        assert!(matches!(
            b.add(Op::Add { a: east, b: north }),
            Err(ProgramError::TypeMismatch { op: "add", .. })
        ));
        assert!(b.add(Op::Angle { input: half }).is_err());
    }

    #[test]
    fn warps_widen_the_footprint_by_their_stretch() {
        let d = torus();
        let freq = |f: f32, seed| Op::Fractal {
            basis: Basis::Gradient,
            domain: d,
            frequency: [f, f],
            seed,
            params: FractalParams::default(),
        };
        let mut b = ProgramBuilder::new();
        let input = b.add(freq(16.0, 1)).unwrap();
        let dx = b.add(freq(4.0, 2)).unwrap();
        let dy = b.add(freq(4.0, 3)).unwrap();
        let amount = 0.2;
        let warp = b
            .add(Op::Warp {
                input,
                dx,
                dy,
                amount,
            })
            .unwrap();
        let program = b.finish(warp).unwrap();

        let field = |seed, f: f32| {
            Fractal::new(
                Basis::Gradient,
                d,
                Vec2::splat(f),
                seed,
                FractalParams::default(),
            )
            .unwrap()
        };
        let (fin, fdx, fdy) = (field(1, 16.0), field(2, 4.0), field(3, 4.0));
        let footprint = Footprint::new(1.0 / 128.0).unwrap();
        let mut widened = 0;
        for i in 0..32 {
            let p = Vec2::new(i as f32 * 0.061, 0.37);
            let at = |f: &Fractal, q: Vec2| f.eval(q, footprint);
            // The analytic Jacobian's rows bound the stretch.
            let slope = |f: &Fractal| {
                let g = f.eval_gradient(p, footprint).1;
                g.x.abs() + g.y.abs()
            };
            let stretch = 1.0 + amount * slope(&fdx).max(slope(&fdy));
            widened += usize::from(stretch > 1.5);
            let q = p + Vec2::new(at(&fdx, p), at(&fdy, p)) * amount;
            assert_eq!(
                program.eval(p, footprint).to_bits(),
                fin.eval(q, footprint.scaled(stretch)).to_bits(),
                "at {p}"
            );
            // A point footprint is not scaled.
            let q0 = p + Vec2::new(fdx.eval(p, Footprint::POINT), fdy.eval(p, Footprint::POINT))
                * amount;
            assert_eq!(
                program.eval(p, Footprint::POINT).to_bits(),
                fin.eval(q0, Footprint::POINT).to_bits()
            );
        }
        assert!(widened > 0, "the fixture should stretch somewhere");
    }

    #[test]
    fn imports_share_equal_subgraphs() {
        let build = |seed| {
            let mut b = ProgramBuilder::new();
            let shared = noise(&mut b, 1);
            let own = noise(&mut b, seed);
            let sum = b.add(Op::Add { a: shared, b: own }).unwrap();
            b.finish(sum).unwrap()
        };
        let (p, q) = (build(2), build(3));
        let mut b = ProgramBuilder::new();
        let a = b.import(&p).unwrap();
        let c = b.import(&q).unwrap();
        let product = b.add(Op::Mul { a, b: c }).unwrap();
        let combined = b.finish(product).unwrap();
        // The shared noise is imported once: 1 shared + 2 own + 2 sums + 1.
        assert_eq!(combined.len(), 6);
        assert_eq!(combined.node_fingerprint(a), Some(p.fingerprint()));
        let at = Vec2::new(0.3, 0.9);
        assert_eq!(
            combined.eval(at, Footprint::POINT).to_bits(),
            (p.eval(at, Footprint::POINT) * q.eval(at, Footprint::POINT)).to_bits()
        );
        // Operand placeholders map through `map_inputs`.
        let op = Op::Add {
            a: NodeId::from_index(0),
            b: NodeId::from_index(1),
        };
        assert_eq!(
            op.map_inputs(|n| NodeId(n.0 + 5)),
            Op::Add {
                a: NodeId(5),
                b: NodeId(6)
            }
        );
    }

    #[test]
    fn disks_are_masks_matching_the_direct_field() {
        let d = torus();
        let op = Op::Disk {
            domain: d,
            center: [0.5, 0.5],
            radius: 0.2,
            softness: 0.05,
        };
        let mut b = ProgramBuilder::new();
        let disk = b.add(op.clone()).unwrap();
        assert_eq!(b.port_type(disk), Ok(PortType::Mask));
        let program = b.finish(disk).unwrap();
        let direct = Disk::new(d, Vec2::new(0.5, 0.5), 0.2, 0.05).unwrap();
        let footprint = Footprint::new(0.01).unwrap();
        for i in 0..50 {
            let p = Vec2::new(i as f32 * 0.041, i as f32 * 0.023);
            assert_eq!(
                program.eval(p, footprint).to_bits(),
                direct.eval(p, footprint).to_bits()
            );
        }

        let mut other = ProgramBuilder::new();
        let moved = other
            .add(Op::Disk {
                domain: d,
                center: [0.6, 0.5],
                radius: 0.2,
                softness: 0.05,
            })
            .unwrap();
        assert_ne!(
            program.fingerprint(),
            other.finish(moved).unwrap().fingerprint()
        );
    }

    #[test]
    fn disk_edits_change_only_their_supports() {
        let d = torus();
        let disk = |x: f32| Op::Disk {
            domain: d,
            center: [x, 0.5],
            radius: 0.1,
            softness: 0.0,
        };
        assert_eq!(disk(0.3).change_from(&disk(0.3)), Change::Nowhere);
        let Change::Within { regions, .. } = disk(0.6).change_from(&disk(0.3)) else {
            panic!("a moved disk changes locally");
        };
        assert_eq!(regions.len(), 2);
        assert!(regions[0].origin.abs_diff_eq(Vec2::new(0.2, 0.4), 1e-6));
        assert!(regions[1].origin.abs_diff_eq(Vec2::new(0.5, 0.4), 1e-6));
        let noise = Op::Noise {
            basis: Basis::Gradient,
            domain: d,
            frequency: [2.0, 2.0],
            seed: 1,
        };
        assert_eq!(disk(0.3).change_from(&noise), Change::Everywhere);
        assert!(
            !Op::Warp {
                input: NodeId(0),
                dx: NodeId(1),
                dy: NodeId(2),
                amount: 1.0
            }
            .operand_is_pointwise(0)
        );
        let many = (0..20).fold(Change::Nowhere, |c, i| {
            c.union(Change::within(alloc::vec![Region {
                origin: Vec2::splat(i as f32),
                size: Vec2::ONE,
            }]))
        });
        let Change::Within {
            regions: merged, ..
        } = many
        else {
            panic!("regions stay local");
        };
        assert!(merged.len() <= Change::MAX_REGIONS);
        let covers = |p: Vec2| {
            merged
                .iter()
                .any(|r| p.cmpge(r.origin).all() && p.cmple(r.origin + r.size).all())
        };
        assert!((0..20).all(|i| covers(Vec2::splat(i as f32 + 0.5))));
    }

    /// A scalar program using every op with a closed-form gradient.
    fn smooth(b: &mut ProgramBuilder) -> NodeId {
        let d = torus();
        let fractal = |b: &mut ProgramBuilder, kind, seed| {
            b.add(Op::Fractal {
                basis: Basis::Gradient,
                domain: d,
                frequency: [2.0, 3.0],
                seed,
                params: FractalParams {
                    kind,
                    octaves: 3,
                    ..FractalParams::default()
                },
            })
            .unwrap()
        };
        let fbm = fractal(b, FractalKind::Fbm, 1);
        let ridged = fractal(b, FractalKind::Ridged, 2);
        let value = b
            .add(Op::Noise {
                basis: Basis::Value,
                domain: d,
                frequency: [4.0, 2.0],
                seed: 3,
            })
            .unwrap();
        let disk = b
            .add(Op::Disk {
                domain: d,
                center: [1.0, 0.5],
                radius: 0.3,
                softness: 0.2,
            })
            .unwrap();
        let sum = b.add(Op::Add { a: fbm, b: ridged }).unwrap();
        let product = b.add(Op::Mul { a: sum, b: value }).unwrap();
        let mixed = b
            .add(Op::Mix {
                a: product,
                b: fbm,
                t: disk,
            })
            .unwrap();
        let remapped = b
            .add(Op::Remap {
                input: mixed,
                from: [-1.0, 1.0],
                to: [0.0, 2.0],
            })
            .unwrap();
        let shifted = b
            .add(Op::Transform {
                input: remapped,
                transform: Affine2 {
                    matrix: Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(-2.0, 0.0)),
                    translation: Vec2::new(0.3, 0.1),
                },
            })
            .unwrap_or(remapped);
        b.add(Op::Warp {
            input: shifted,
            dx: fbm,
            dy: value,
            amount: 0.05,
        })
        .unwrap()
    }

    #[test]
    fn analytic_gradients_match_central_differences() {
        let mut b = ProgramBuilder::new();
        let output = smooth(&mut b);
        let program = b.finish(output).unwrap();
        // A small footprint, so band limits barely move between probes.
        let footprint = Footprint::new(1e-3).unwrap();
        let mut evaluator = program.evaluator();
        for i in 0..40 {
            let p = Vec2::new(0.05 + i as f32 * 0.047, 0.11 + i as f32 * 0.019);
            let (value, gradient) = program.eval_gradient(p, footprint);
            assert_eq!(value.to_bits(), program.eval(p, footprint).to_bits());
            assert_eq!(value.to_bits(), evaluator.eval(p, footprint).to_bits());
            let numeric = central_difference(&program, p, footprint);
            let error = (gradient - numeric).length();
            let scale = numeric.length().max(1.0);
            assert!(
                error < 0.05 * scale,
                "at {p}: analytic {gradient}, numeric {numeric}"
            );
        }
    }

    #[test]
    fn primitive_gradients_keep_their_values() {
        let d = torus();
        let footprint = Footprint::new(0.01).unwrap();
        let cells = Cellular::new(d, Vec2::new(4.0, 2.0), 0.8, 5).unwrap();
        let (f1, edges, border) = (
            cells.output(CellOutput::F1),
            cells.output(CellOutput::F2MinusF1),
            cells.output(CellOutput::Border),
        );
        let fields: [&dyn ScalarField; 7] = [
            &f1,
            &edges,
            &border,
            &Noise::new(Basis::Gradient, d, Vec2::new(4.0, 6.0), 9).unwrap(),
            &Noise::new(Basis::Value, d, Vec2::new(4.0, 6.0), 9).unwrap(),
            &Fractal::new(
                Basis::Gradient,
                d,
                Vec2::splat(2.0),
                4,
                FractalParams {
                    kind: FractalKind::Ridged,
                    ..FractalParams::default()
                },
            )
            .unwrap(),
            &Disk::new(d, Vec2::new(1.0, 0.5), 0.25, 0.1).unwrap(),
        ];
        for field in fields {
            for i in 0..50 {
                let p = Vec2::new(i as f32 * 0.043, i as f32 * 0.017);
                let (value, gradient) = field.eval_gradient(p, footprint);
                assert_eq!(value.to_bits(), field.eval(p, footprint).to_bits());
                let numeric = central_difference(field, p, Footprint::new(1e-4).unwrap());
                let analytic = field.eval_gradient(p, Footprint::new(1e-4).unwrap()).1;
                assert!(
                    (analytic - numeric).length() < 0.02 * numeric.length().max(1.0),
                    "at {p}: {analytic} vs {numeric}"
                );
                assert!(gradient.is_finite());
            }
        }
    }
}
