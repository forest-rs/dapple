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
//! primaries, and normals in a declared frame. Type rules reject misuse at
//! build time: normals combine only through [`Op::BlendNormals`], never by
//! lerping; identifiers never blend; masks stay masks only through operations
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

use glam::{Mat2, Vec2, Vec3};

use crate::cellular::{CellOutput, Cellular, CellularField};
use crate::domain::{Domain, DomainError, Footprint};
use crate::field::{Affine2, ScalarField, check_transform};
use crate::fractal::{Fractal, FractalKind, FractalParams};
use crate::hash::hash;
use crate::noise::{Basis, Noise};
use crate::types::{NormalBlend, NormalFrame, PortType, Primaries, Value};

mod flat;

pub use flat::{EvaluationStats, Evaluator};

/// Version of the fingerprint encoding. Changing any word the encoding emits
/// requires a new version, so persisted fingerprints never collide.
pub const FINGERPRINT_VERSION: u64 = 1;

/// A node in a [`FieldProgram`] or [`ProgramBuilder`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct NodeId(u32);

impl NodeId {
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
    /// field stays periodic. The footprint passes through unchanged, which
    /// under-filters where the warp compresses the input.
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
    /// A detail normal applied to a base normal, both in the same frame.
    BlendNormals {
        /// The base normal.
        base: NodeId,
        /// The detail normal.
        detail: NodeId,
        /// The blend.
        method: NormalBlend,
    },
}

impl Op {
    /// The node's inputs, in operand order.
    #[must_use]
    pub fn inputs(&self) -> Inputs {
        let mut inputs = Inputs::default();
        match *self {
            Self::Constant { .. }
            | Self::Noise { .. }
            | Self::Fractal { .. }
            | Self::Cellular { .. } => {}
            Self::Transform { input, .. }
            | Self::Demote { input }
            | Self::Abs { input }
            | Self::Clamp { input, .. }
            | Self::Remap { input, .. }
            | Self::Component { input, .. }
            | Self::AsMask { input }
            | Self::ToId { input, .. }
            | Self::Normalize { input } => inputs.push(input),
            Self::Add { a, b }
            | Self::Sub { a, b }
            | Self::Mul { a, b }
            | Self::Min { a, b }
            | Self::Max { a, b }
            | Self::Vector2 { x: a, y: b }
            | Self::BlendNormals {
                base: a, detail: b, ..
            } => {
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

    /// A short operation name, for listings and reports.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Constant { .. } => "constant",
            Self::Noise { .. } => "noise",
            Self::Fractal { .. } => "fractal",
            Self::Cellular { .. } => "cellular",
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
            Self::BlendNormals { .. } => "blend-normals",
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
/// - a component index or identifier level count as itself, and a
///   [`NormalBlend`] as its declaration index;
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
    Transform {
        input: NodeId,
        transform: Affine2,
        stretch: f32,
    },
    Pass(NodeId),
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
    BlendNormals(NodeId, NodeId, NormalBlend),
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
    domain: Domain,
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
        let domain = self.derive_domain(&op)?;
        let port = self.derive_type(&op)?;
        let kernel = self.kernel(&op)?;
        let fingerprint = self.fingerprint(&op);
        let id = NodeId(u32::try_from(self.nodes.len()).expect("fewer than 2^32 nodes"));
        self.nodes.push(Node {
            op,
            domain,
            port,
            fingerprint,
            kernel,
        });
        Ok(id)
    }

    /// The domain of an existing node.
    pub fn domain(&self, id: NodeId) -> Result<Domain, ProgramError> {
        Ok(self.node(id)?.domain)
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
        let found = self.node(output)?.port;
        if !found.is_scalar() {
            return Err(ProgramError::OutputType { found });
        }
        Ok(FieldProgram::new(self.nodes, output))
    }

    /// Finishes a program whose output may have any [`PortType`].
    pub fn finish_value(self, output: NodeId) -> Result<ValueProgram, ProgramError> {
        self.node(output)?;
        Ok(ValueProgram {
            program: FieldProgram::new(self.nodes, output),
        })
    }

    fn node(&self, id: NodeId) -> Result<&Node, ProgramError> {
        self.nodes
            .get(id.0 as usize)
            .ok_or(ProgramError::UnknownNode { node: id })
    }

    fn derive_domain(&self, op: &Op) -> Result<Domain, ProgramError> {
        Ok(match *op {
            Op::Constant { domain, .. }
            | Op::Noise { domain, .. }
            | Op::Fractal { domain, .. }
            | Op::Cellular { domain, .. } => domain,
            Op::Demote { .. } => Domain::Plane,
            _ => {
                let mut inputs = op
                    .inputs()
                    .iter()
                    .map(|id| self.nodes[id.0 as usize].domain);
                let first = inputs.next().expect("derived nodes have inputs");
                if let Some(other) = inputs.find(|d| *d != first) {
                    return Err(ProgramError::DomainMismatch { first, other });
                }
                first
            }
        })
    }

    fn derive_type(&self, op: &Op) -> Result<PortType, ProgramError> {
        let name = op.name();
        let port = |id: NodeId| self.nodes[id.0 as usize].port;
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
            Op::Constant { .. } | Op::Noise { .. } | Op::Fractal { .. } | Op::Cellular { .. } => {
                PortType::Scalar
            }
            Op::Transform { input, transform } => {
                let found = port(input);
                let directional = matches!(
                    found,
                    PortType::Vector2 | PortType::Vector3 | PortType::Normal(_)
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
            Op::BlendNormals { base, detail, .. } => {
                let (tb, td) = (port(base), port(detail));
                if !matches!(tb, PortType::Normal(_)) {
                    return Err(mismatch(tb, "needs normals"));
                }
                if td != tb {
                    return Err(mismatch(td, "needs a normal in the base's frame"));
                }
                tb
            }
        })
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
            Op::Transform { input, transform } => {
                let stretch = check_transform(self.nodes[input.0 as usize].domain, transform)?;
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
            Op::BlendNormals {
                base,
                detail,
                method,
            } => Kernel::BlendNormals(base, detail, method),
        })
    }

    fn fingerprint(&self, op: &Op) -> Fingerprint {
        let mut words = Vec::with_capacity(16);
        words.push(FINGERPRINT_VERSION);
        words.push(op_tag(op));
        let domain = |words: &mut Vec<u64>, domain: Domain| match domain {
            Domain::Plane => words.push(0),
            Domain::Periodic { period } => {
                words.extend([1, u64::from(period[0]), u64::from(period[1])]);
            }
        };
        let float = |v: f32| u64::from(v.to_bits());
        match *op {
            Op::Constant { domain: d, value } => {
                domain(&mut words, d);
                words.push(float(value));
            }
            Op::Noise {
                basis,
                domain: d,
                frequency,
                seed,
            } => {
                words.push(basis_tag(basis));
                domain(&mut words, d);
                words.extend([float(frequency[0]), float(frequency[1]), seed]);
            }
            Op::Fractal {
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
            Op::Cellular {
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
            Op::Transform { transform, .. } => {
                let m = transform.matrix.to_cols_array();
                let t = transform.translation.to_array();
                words.extend(m.iter().chain(&t).map(|v| float(*v)));
            }
            Op::Clamp { min, max, .. } => words.extend([float(min), float(max)]),
            Op::Remap { from, to, .. } => words.extend(from.iter().chain(&to).map(|v| float(*v))),
            Op::Warp { amount, .. } => words.push(float(amount)),
            Op::Component { index, .. } => words.push(u64::from(index)),
            Op::ToId { levels, .. } => words.push(u64::from(levels)),
            Op::BlendNormals { method, .. } => words.push(match method {
                NormalBlend::Reoriented => 0,
                NormalBlend::Udn => 1,
            }),
            Op::Vector2 { .. }
            | Op::Vector3 { .. }
            | Op::Color { .. }
            | Op::AsMask { .. }
            | Op::Normalize { .. }
            | Op::Demote { .. }
            | Op::Add { .. }
            | Op::Sub { .. }
            | Op::Mul { .. }
            | Op::Min { .. }
            | Op::Max { .. }
            | Op::Abs { .. }
            | Op::Mix { .. } => {}
        }
        for input in op.inputs().iter() {
            let fp = self.nodes[input.0 as usize].fingerprint.0;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "splitting the 128-bit fingerprint into its halves"
            )]
            words.extend([fp as u64, (fp >> 64) as u64]);
        }
        fingerprint_words(&words)
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
        Op::BlendNormals { .. } => 23,
    }
}

const fn basis_tag(basis: Basis) -> u64 {
    match basis {
        Basis::Value => 0,
        Basis::Gradient => 1,
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

    /// The domain of `id`, if it exists.
    #[must_use]
    pub fn node_domain(&self, id: NodeId) -> Option<Domain> {
        self.nodes.get(id.0 as usize).map(|node| node.domain)
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
    /// Panics if `id` is not a node of this program.
    #[must_use]
    pub fn eval_value(&self, id: NodeId, p: Vec2, footprint: Footprint) -> Value {
        let node = &self.nodes[id.0 as usize];
        match node.kernel {
            Kernel::Transform {
                input,
                transform,
                stretch,
            } => self.eval_value(input, transform.apply(p), footprint.scaled(stretch)),
            Kernel::Pass(input) => self.eval_value(input, p, footprint),
            Kernel::Warp {
                input,
                dx,
                dy,
                amount,
            } => {
                let offset = warp_offset(
                    self.eval_value(dx, p, footprint),
                    self.eval_value(dy, p, footprint),
                    amount,
                );
                self.eval_value(input, p + offset, footprint)
            }
            ref kernel => {
                let mut args = [Value::Scalar(0.0); 3];
                let mut count = 0;
                for input in node.op.inputs().iter() {
                    args[count] = self.eval_value(input, p, footprint);
                    count += 1;
                }
                kernel.combine(p, footprint, &args[..count])
            }
        }
    }
}

fn warp_offset(dx: Value, dy: Value, amount: f32) -> Vec2 {
    let scalar = |v: Value| v.scalar().expect("warp displacements are scalars");
    Vec2::new(scalar(dx), scalar(dy)) * amount
}

impl Kernel {
    /// Evaluates a pure kernel from its operands' values, in operand order.
    fn combine(&self, p: Vec2, footprint: Footprint, args: &[Value]) -> Value {
        let scalar = |i: usize| args[i].scalar().expect("operand types were checked");
        match *self {
            Self::Constant(value) => Value::Scalar(value),
            Self::Noise(ref noise) => Value::Scalar(noise.eval(p, footprint)),
            Self::Fractal(ref fractal) => Value::Scalar(fractal.eval(p, footprint)),
            Self::Cellular(ref cellular) => Value::Scalar(cellular.eval(p, footprint)),
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
            Self::BlendNormals(_, _, method) => {
                let (Value::Vector3(base), Value::Vector3(detail)) = (args[0], args[1]) else {
                    unreachable!("blend inputs were type-checked");
                };
                Value::Vector3(blend_normals(base, detail, method))
            }
            Self::Transform { .. } | Self::Pass(_) | Self::Warp { .. } => {
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

/// `detail` applied to `base`; both unit normals in one frame.
fn blend_normals(base: Vec3, detail: Vec3, method: NormalBlend) -> Vec3 {
    let n = match method {
        NormalBlend::Reoriented => {
            // Barré-Brisebois and Hill: rotate `detail` from +Z onto `base`.
            let t = base + Vec3::Z;
            let u = detail * Vec3::new(-1.0, -1.0, 1.0);
            t * t.dot(u) / t.z - u
        }
        NormalBlend::Udn => Vec3::new(base.x + detail.x, base.y + detail.y, base.z),
    };
    n.try_normalize().unwrap_or(Vec3::Z)
}

impl ScalarField for FieldProgram {
    fn domain(&self) -> Domain {
        self.nodes[self.output.0 as usize].domain
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
        self.program.nodes[self.program.output.0 as usize].domain
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Transformed;
    use crate::raster::{Grid, Region};
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

    const GOLDEN_FINGERPRINT: u128 = 0x9f47_5e28_e73d_17cf_d6eb_c799_c751_d5a4;
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
    fn normal_blends_keep_detail_on_a_flat_base() {
        let d = torus();
        let mut b = ProgramBuilder::new();
        let constant =
            |b: &mut ProgramBuilder, value| b.add(Op::Constant { domain: d, value }).unwrap();
        let (zero, one, tilt) = (
            constant(&mut b, 0.0),
            constant(&mut b, 1.0),
            constant(&mut b, 0.6),
        );
        let flat_v = b
            .add(Op::Vector3 {
                x: zero,
                y: zero,
                z: one,
            })
            .unwrap();
        let flat = b.add(Op::Normalize { input: flat_v }).unwrap();
        let detail_v = b
            .add(Op::Vector3 {
                x: tilt,
                y: zero,
                z: one,
            })
            .unwrap();
        let detail = b.add(Op::Normalize { input: detail_v }).unwrap();
        let expected = Vec3::new(0.6, 0.0, 1.0).normalize();
        // Reoriented blending reproduces the detail exactly; UDN keeps the
        // base's z, so it flattens the detail slightly.
        let udn = Vec3::new(expected.x, 0.0, 1.0).normalize();
        for (method, expected) in [(NormalBlend::Reoriented, expected), (NormalBlend::Udn, udn)] {
            let blended = b
                .add(Op::BlendNormals {
                    base: flat,
                    detail,
                    method,
                })
                .unwrap();
            let Value::Vector3(n) = b
                .clone()
                .finish_value(blended)
                .unwrap()
                .eval(Vec2::ZERO, Footprint::POINT)
            else {
                panic!("normals are vectors");
            };
            assert!((n - expected).length() < 1e-6, "{method:?}: {n}");
        }
        // Reoriented blending of a tilted base and tilted detail tilts further.
        let tilted = b
            .add(Op::BlendNormals {
                base: detail,
                detail,
                method: NormalBlend::Reoriented,
            })
            .unwrap();
        let Value::Vector3(n) = b
            .clone()
            .finish_value(tilted)
            .unwrap()
            .eval(Vec2::ZERO, Footprint::POINT)
        else {
            panic!("normals are vectors");
        };
        assert!(n.x > expected.x && (n.length() - 1.0).abs() < 1e-6, "{n}");
        assert!(matches!(
            b.add(Op::BlendNormals {
                base: flat,
                detail: flat_v,
                method: NormalBlend::Udn
            }),
            Err(ProgramError::TypeMismatch {
                op: "blend-normals",
                ..
            })
        ));
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
}
