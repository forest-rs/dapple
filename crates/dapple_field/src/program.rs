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

use glam::Vec2;

use crate::cellular::{CellOutput, Cellular, CellularField};
use crate::domain::{Domain, DomainError, Footprint};
use crate::field::{Affine2, ScalarField, check_transform};
use crate::fractal::{Fractal, FractalKind, FractalParams};
use crate::hash::hash;
use crate::noise::{Basis, Noise};

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
            | Self::Remap { input, .. } => inputs.push(input),
            Self::Add { a, b }
            | Self::Sub { a, b }
            | Self::Mul { a, b }
            | Self::Min { a, b }
            | Self::Max { a, b } => {
                inputs.push(a);
                inputs.push(b);
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
/// - each input as the two halves of its own fingerprint, in operand order.
///
/// Node identities and creation order are not part of it: equal subgraphs
/// have equal fingerprints.
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
        let kernel = self.kernel(&op)?;
        let fingerprint = self.fingerprint(&op);
        let id = NodeId(u32::try_from(self.nodes.len()).expect("fewer than 2^32 nodes"));
        self.nodes.push(Node {
            op,
            domain,
            fingerprint,
            kernel,
        });
        Ok(id)
    }

    /// The domain of an existing node.
    pub fn domain(&self, id: NodeId) -> Result<Domain, ProgramError> {
        Ok(self.node(id)?.domain)
    }

    /// Finishes the program with `output` as its result.
    ///
    /// Nodes that `output` does not depend on are kept for inspection but
    /// never evaluated, and do not affect [`FieldProgram::fingerprint`].
    pub fn finish(self, output: NodeId) -> Result<FieldProgram, ProgramError> {
        self.node(output)?;
        Ok(FieldProgram {
            nodes: self.nodes,
            output,
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
            Op::Demote { .. }
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
/// Evaluation walks the DAG from the output. A node shared by several
/// consumers is evaluated once per consumer; there is no per-point cache.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldProgram {
    nodes: Vec<Node>,
    output: NodeId,
}

impl FieldProgram {
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

    /// Evaluates node `id` at `p`.
    ///
    /// # Panics
    ///
    /// Panics if `id` is not a node of this program.
    #[must_use]
    pub fn eval_node(&self, id: NodeId, p: Vec2, footprint: Footprint) -> f32 {
        match self.nodes[id.0 as usize].kernel {
            Kernel::Constant(value) => value,
            Kernel::Noise(ref noise) => noise.eval(p, footprint),
            Kernel::Fractal(ref fractal) => fractal.eval(p, footprint),
            Kernel::Cellular(ref cellular) => cellular.eval(p, footprint),
            Kernel::Transform {
                input,
                transform,
                stretch,
            } => self.eval_node(input, transform.apply(p), footprint.scaled(stretch)),
            Kernel::Pass(input) => self.eval_node(input, p, footprint),
            Kernel::Binary(op, a, b) => {
                let a = self.eval_node(a, p, footprint);
                let b = self.eval_node(b, p, footprint);
                match op {
                    BinaryOp::Add => a + b,
                    BinaryOp::Sub => a - b,
                    BinaryOp::Mul => a * b,
                    BinaryOp::Min => a.min(b),
                    BinaryOp::Max => a.max(b),
                }
            }
            Kernel::Abs(input) => self.eval_node(input, p, footprint).abs(),
            Kernel::Clamp(input, min, max) => self.eval_node(input, p, footprint).clamp(min, max),
            Kernel::Remap {
                input,
                scale,
                from,
                to,
            } => to + (self.eval_node(input, p, footprint) - from) * scale,
            Kernel::Mix(a, b, t) => {
                let a = self.eval_node(a, p, footprint);
                let b = self.eval_node(b, p, footprint);
                let t = self.eval_node(t, p, footprint);
                a + (b - a) * t
            }
            Kernel::Warp {
                input,
                dx,
                dy,
                amount,
            } => {
                let offset = Vec2::new(
                    self.eval_node(dx, p, footprint),
                    self.eval_node(dy, p, footprint),
                ) * amount;
                self.eval_node(input, p + offset, footprint)
            }
        }
    }
}

impl ScalarField for FieldProgram {
    fn domain(&self) -> Domain {
        self.nodes[self.output.0 as usize].domain
    }

    fn eval(&self, p: Vec2, footprint: Footprint) -> f32 {
        self.eval_node(self.output, p, footprint)
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
    const GOLDEN_DIGEST: u64 = 0x90f6_27f1_c5e8_d386;

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
}
