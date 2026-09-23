// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Incremental material graphs for dapple.
//!
//! A [`MaterialGraph`] is an `execution_graph::ExecutionGraph` whose nodes are
//! dapple operations:
//!
//! - **Field nodes** hold one `dapple_field::program::Op`. Its operands are
//!   placeholders ([`operand`]) for the node's upstream field inputs. Running
//!   the node imports the input programs and adds the op, so its output is
//!   the whole program so far, with shared subgraphs imported once.
//! - **Realize nodes** turn a field into a raster over one period of its
//!   periodic domain: scalars and masks tile by tile, other types
//!   (identifiers, directions, vectors, colors, normals) whole.
//! - **Raster nodes** apply a `dapple_raster` operation.
//! - **Normals nodes** realize a scalar field's normals from its gradient,
//!   analytic where the field's ops allow.
//! - **Mip nodes** filter a scalar raster down one mip level, as each level
//!   of `dapple_encode::data_mips`; chain them for a whole mip chain.
//! - **Reduce nodes** reduce a raster of any type to one mip level under an
//!   explicit, type-checked `ReductionPolicy`, computed from the level-0
//!   texels the level's texels cover.
//! - **Sample nodes** read scalar rasters back as a field
//!   ([`Op::Sample`]): a base raster, optionally with its mip chain, sampled
//!   bilinearly and filtered by footprint. Field nodes build on them like on
//!   any other field, so a realized, blurred or eroded height can be warped,
//!   combined and realized again.
//!
//! Each node's parameters (the op, the resolution, the raster operation) are
//! an input of that node, named `<label>.params`. Editing them with
//! [`MaterialGraph::set_field_op`] or its siblings rebinds and invalidates only
//! that input, so the next [`MaterialGraph::run`] re-runs exactly the edited
//! node and its dependents. `execution_graph` schedules, records dependencies
//! and reports.
//!
//! Every raster carries its semantic type ([`RasterValue::port`]): a mask
//! stays a mask, a normal a normal, an identifier an identifier. Operations
//! check it and refuse types they do not accept: a blur, mip or sample of
//! identifiers, directions or normals is [`NodeError::TypeRefused`], and an
//! average of identifiers is a refused reduction policy.
//!
//! Values on edges are cheap handles: field programs and rasters are
//! reference-counted, with fingerprints for caching and comparison.
//!
//! ## Tiles
//!
//! Realize and raster nodes split their rasters into square tiles
//! ([`MaterialGraph::with_tile_size`]) and recompute only the tiles an edit
//! reaches:
//!
//! - A field value carries [`FieldValue::change`]: where it can differ from
//!   the producing node's previous program. Ops that can state their edit's
//!   region ([`Op::change_from`], such as a moved [`Op::Disk`]) start one;
//!   pointwise ops carry their inputs' regions through; a warp grows its
//!   warped input's regions by its displacements' static reach and footprint
//!   stretch, or makes them unbounded when the displacements have no static
//!   bounds ([`TileReport::unbounded_warps`]); transforms make the change
//!   unbounded.
//! - A realize node re-realizes the tiles whose texel centers fall in the
//!   change, grown by the change's footprint growth, and recomputes whole on
//!   an unbounded change ([`TileReport::unbounded_changes`]).
//! - Each raster-node tile depends, through an `invalidation` tracker, on
//!   the input tiles its kernel footprint reads; each mip-node tile on the
//!   level-above tiles its filter taps read. A node compares every
//!   recomputed tile with its previous bits and marks the dependents of the
//!   tiles that changed, so unchanged tiles stop propagating. Global
//!   operations, such as the distance transform, recompute whole when their
//!   input changed.
//!
//! - A sample node's field changes only near the tiles of its rasters whose
//!   bits changed: each changed tile's region, grown by one texel for the
//!   bilinear taps, becomes [`FieldValue::change`], so realizing the sampled
//!   field downstream recomputes only the tiles those regions reach.
//!
//! Tile-wise results equal whole recomputation bit for bit.
//! [`MaterialGraph::tile_report`] counts the work of the last run.
//!
//! ## Early cutoff
//!
//! A node that re-runs and produces the same value as before (the same
//! program, or a raster with the same derivation and texels) stops
//! propagation: its dependents are cut off instead of re-run, and
//! [`RunSummary::cut_off_nodes`] counts them. Re-setting a node's parameters
//! to their current value, for example, re-runs only that node. An edit
//! that changes a derivation but not the texels, such as a clamp bound that
//! no value reaches, still re-runs the dependents so their fingerprints stay
//! exact, but they recompute no tiles.
//!
//! ```
//! use dapple_field::program::Op;
//! use dapple_field::{Basis, Domain};
//! use dapple_graph::{MaterialGraph, RasterParams, operand};
//! use dapple_raster::GaussianBlur;
//!
//! let domain = Domain::periodic(1, 1).unwrap();
//! let mut g = MaterialGraph::new();
//! let noise = g.field("noise", Op::Noise { basis: Basis::Gradient, domain, frequency: [8.0, 8.0], seed: 1 }, &[])?;
//! let height = g.field("height", Op::Abs { input: operand(0) }, &[noise])?;
//! let map = g.realize("map", height, 64, 64)?;
//! let soft = g.raster("soft", RasterParams::Blur(GaussianBlur { sigma: 0.01 }), map)?;
//! assert_eq!(g.run()?.executed_nodes, 4);
//! assert!(g.raster_value(soft).is_some());
//!
//! // Blurring more re-runs only the blur.
//! g.set_raster_params(soft, RasterParams::Blur(GaussianBlur { sigma: 0.02 }))?;
//! assert_eq!(g.run()?.executed_nodes, 1);
//! # Ok::<(), dapple_graph::MaterialError>(())
//! ```

#![no_std]

extern crate alloc;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use dapple_encode::{EncodeError, Filter, Image, next_level, next_level_into, source_span};
use dapple_field::hash::hash;
use dapple_field::program::{
    Change, Fingerprint, NodeId as FieldNode, Op, ProgramBuilder, ProgramError, ValueProgram,
};
use dapple_field::raster::Region;
use dapple_field::{Domain, DomainError, ImageLevel, NormalFrame, PortType, SampleImage};
use dapple_raster::typed::{ReductionPolicy, Storage, TypedError, TypedRaster, realize_value};
use dapple_raster::{
    AmbientOcclusion, DistanceTransform, Edge, GaussianBlur, HeightToNormal, Raster, RasterError,
    RasterOp, Realization, TexelRect, realize, realize_into, realize_normals, realize_normals_into,
};
use execution_graph::{ExecutionGraph, Executor, GraphError, NodeAccess, NodeId, RunSummary};
use glam::Vec2;
use invalidation::intern::InternId;

mod recipe;
mod tiles;

pub use recipe::{
    NodeFingerprint, RECIPE_VERSION, Recipe, RecipeError, RecipeNode, RecipeOutput, Step,
};
pub use tiles::{DEFAULT_TILE_SIZE, TileReport};
use tiles::{Grid, Tiles};

/// The placeholder for a field node's `index`-th upstream input, for use as an
/// operand of the node's [`Op`].
#[must_use]
pub const fn operand(index: u32) -> FieldNode {
    FieldNode::from_index(index)
}

/// A raster operation and its parameters.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "op", rename_all = "snake_case"))]
pub enum RasterParams {
    /// [`GaussianBlur`].
    Blur(GaussianBlur),
    /// [`HeightToNormal`]; outputs a three-channel raster.
    HeightToNormal(HeightToNormal),
    /// [`AmbientOcclusion`].
    AmbientOcclusion(AmbientOcclusion),
    /// [`DistanceTransform`].
    DistanceTransform(DistanceTransform),
}

/// A node's parameters, carried on its `<label>.params` input.
#[derive(Clone, Debug, PartialEq)]
pub enum Params {
    /// A field node's op.
    Field(Op),
    /// A realize node's resolution.
    Realize {
        /// Texels per row.
        width: u32,
        /// Rows.
        height: u32,
    },
    /// A raster node's operation.
    Raster(RasterParams),
    /// A mip node's filter.
    Mip(Filter),
    /// A reduce node's policy and level.
    Reduce {
        /// How texels combine.
        policy: ReductionPolicy,
        /// The mip level, at least 1.
        level: u32,
    },
    /// A sample node, which has no parameters of its own.
    Sample,
    /// A normals node's resolution and height scale.
    Normals {
        /// Texels per row.
        width: u32,
        /// Rows.
        height: u32,
        /// Domain units of height per field value unit.
        scale: f32,
    },
}

/// A realized raster's texels.
///
/// Scalar and three-channel rasters are computed tile by tile; other typed
/// rasters (identifiers, 2D vectors and directions, colors and vectors
/// realized from programs, reduced levels) are computed whole. Every raster's
/// meaning is its [`RasterValue::port`].
#[derive(Clone, Debug, PartialEq)]
pub enum RasterData {
    /// One channel, computed tile-wise.
    Scalar(Raster),
    /// Three channels (normals), computed tile-wise.
    Vector3(Raster<[f32; 3]>),
    /// Any type, computed whole.
    Typed(TypedRaster),
}

impl RasterData {
    fn same_grid(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Scalar(a), Self::Scalar(b)) => a.same_grid(b),
            (Self::Vector3(a), Self::Vector3(b)) => a.same_grid(b),
            (Self::Typed(a), Self::Typed(b)) => match (a.storage(), b.storage()) {
                (Storage::F32(a), Storage::F32(b)) => a.same_grid(b),
                (Storage::U32(a), Storage::U32(b)) => a.same_grid(b),
                (Storage::F32x2(a), Storage::F32x2(b)) => a.same_grid(b),
                (Storage::F32x3(a), Storage::F32x3(b)) => a.same_grid(b),
                _ => false,
            },
            _ => false,
        }
    }

    fn rect_bits_eq(&self, other: &Self, rect: TexelRect) -> bool {
        match (self, other) {
            (Self::Scalar(a), Self::Scalar(b)) => a.rect_bits_eq(b, rect),
            (Self::Vector3(a), Self::Vector3(b)) => a.rect_bits_eq(b, rect),
            (Self::Typed(a), Self::Typed(b)) => match (a.storage(), b.storage()) {
                (Storage::F32(a), Storage::F32(b)) => a.rect_bits_eq(b, rect),
                (Storage::U32(a), Storage::U32(b)) => a.rect_bits_eq(b, rect),
                (Storage::F32x2(a), Storage::F32x2(b)) => a.rect_bits_eq(b, rect),
                (Storage::F32x3(a), Storage::F32x3(b)) => a.rect_bits_eq(b, rect),
                _ => false,
            },
            _ => false,
        }
    }

    fn grid(&self, tile_size: u32) -> Grid {
        let (w, h, edge) = match self {
            Self::Scalar(r) => (r.width(), r.height(), r.edge()),
            Self::Vector3(r) => (r.width(), r.height(), r.edge()),
            Self::Typed(r) => (r.width(), r.height(), r.edge()),
        };
        Grid::new(w, h, tile_size, edge)
    }

    /// The texels as a [`TypedRaster`] of type `port`.
    fn to_typed(&self, port: PortType) -> Result<TypedRaster, NodeError> {
        let storage = match self {
            Self::Scalar(r) => Storage::F32(r.clone()),
            Self::Vector3(r) => Storage::F32x3(r.clone()),
            Self::Typed(r) => return Ok(r.clone()),
        };
        TypedRaster::new(port, storage).map_err(NodeError::Typed)
    }
}

/// A raster value and its content fingerprint.
#[derive(Clone, Debug, PartialEq)]
pub struct RasterValue {
    /// The texels.
    pub data: RasterData,
    /// What the texels mean: a mask stays a mask and a normal a normal
    /// through the graph, so every consumer can check what it reads.
    pub port: PortType,
    /// Hash of the producing operation, its parameters, and its input's
    /// fingerprint.
    pub fingerprint: u64,
    /// Whether any texel differs from the producing node's previous output.
    pub changed: bool,
    /// The producing node's tile key space, for tile invalidation.
    pub tile_space: u32,
}

/// A field program and how it differs from the producing node's previous
/// program.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldValue {
    /// The program.
    pub program: ValueProgram,
    /// The producing node's previous program, when there was one.
    pub previous: Option<Fingerprint>,
    /// Where this program's values can differ from `previous`'s.
    pub change: Change,
}

/// A value on a graph edge or input.
#[derive(Clone, Debug, PartialEq)]
pub enum GraphValue {
    /// Node parameters.
    Params(Arc<Params>),
    /// A field program.
    Field(Arc<FieldValue>),
    /// A raster.
    Raster(Arc<RasterValue>),
}

/// The kind of a graph node; its parameters arrive as an input.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum NodeKind {
    /// Adds one field op to its inputs' programs.
    Field,
    /// Realizes a scalar field over one period.
    Realize,
    /// Applies a raster operation.
    Raster,
    /// Realizes a scalar field's normals from its gradient.
    Normals,
    /// Filters a scalar raster down one mip level.
    Mip,
    /// Reduces a typed raster to one mip level under an explicit policy.
    Reduce,
    /// Samples scalar rasters, with their mips, as a field.
    Sample,
}

/// State a field node keeps between runs.
#[derive(Clone, Debug, Default)]
struct FieldState {
    op: Option<Op>,
    inputs: Vec<Fingerprint>,
    output: Option<Fingerprint>,
}

/// State a realize or raster node keeps between runs.
#[derive(Clone, Debug, Default)]
struct TileState {
    /// What the previous output was computed from: the input program's
    /// fingerprint, or the raster operation and its input grid.
    source: Option<Source>,
    /// The previous output, shared with the graph's output value.
    output: Option<Arc<RasterValue>>,
    keys: Vec<InternId>,
    /// Tiles still to recompute against `source`, left by a tile budget.
    pending: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
enum Source {
    Realize {
        program: Fingerprint,
        width: u32,
        height: u32,
    },
    Normals {
        program: Fingerprint,
        width: u32,
        height: u32,
        scale: u32,
    },
    Raster {
        params: RasterParams,
        input: u32,
    },
    Mip {
        filter: Filter,
        input: u32,
    },
    /// A raster computed whole, from a derivation fingerprint.
    Whole {
        derivation: u64,
    },
}

/// State a sample node keeps between runs.
#[derive(Clone, Debug, Default)]
struct SampleState {
    /// Per sampled raster: its tile space, its grid, and this node's keys
    /// mirroring its tiles one to one.
    levels: Vec<(u32, Grid, Vec<InternId>)>,
    /// The previous program's fingerprint.
    output: Option<Fingerprint>,
}

/// A graph node: its kind, its tile key space, and its state between runs.
#[derive(Clone, Debug)]
pub struct DappleNode {
    kind: NodeKind,
    key: u32,
    field: FieldState,
    tiles: TileState,
    sample: SampleState,
}

impl DappleNode {
    /// The node's kind.
    #[must_use]
    pub const fn kind(&self) -> NodeKind {
        self.kind
    }
}

/// A failure while running one node.
#[derive(Clone, Debug, PartialEq)]
pub enum NodeError {
    /// The field op was rejected; see [`ProgramError`].
    Program(ProgramError),
    /// Realization or a raster operation failed; see [`RasterError`].
    Raster(RasterError),
    /// A realized field is not a periodic scalar field.
    NotRealizable {
        /// The field's type.
        port: PortType,
        /// The field's domain.
        domain: Domain,
    },
    /// An input carried a value of the wrong kind.
    WrongValue {
        /// What the node expected.
        expected: &'static str,
    },
    /// A field op refers to an operand the node has no input for.
    MissingOperand {
        /// The operand's position.
        index: u32,
    },
    /// Filtering a mip level failed; see [`EncodeError`].
    Mip(EncodeError),
    /// A typed raster could not be built or reduced, for example a policy
    /// that is not meaningful for the raster's type; see [`TypedError`].
    Typed(TypedError),
    /// An operation does not accept rasters of this type, for example a
    /// Gaussian blur of identifiers.
    TypeRefused {
        /// The operation.
        operation: &'static str,
        /// The raster's type.
        port: PortType,
    },
}

impl fmt::Display for NodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Program(error) => error.fmt(f),
            Self::Raster(error) => error.fmt(f),
            Self::NotRealizable { port, domain } => {
                write!(f, "cannot realize a {port} field over {domain:?}")
            }
            Self::WrongValue { expected } => write!(f, "expected a {expected} input"),
            Self::MissingOperand { index } => write!(f, "operand {index} has no input"),
            Self::Mip(error) => error.fmt(f),
            Self::Typed(error) => error.fmt(f),
            Self::TypeRefused { operation, port } => {
                write!(f, "{operation} does not accept {port} rasters")
            }
        }
    }
}

impl core::error::Error for NodeError {}

/// A failure building or running a [`MaterialGraph`].
#[derive(Debug)]
pub enum MaterialError {
    /// A label is already used; labels name each node's parameter input.
    DuplicateLabel(String),
    /// A node is not part of this graph, or has the wrong kind for the call.
    UnknownNode,
    /// The graph rejected a call or a node failed.
    Graph(GraphError<NodeError>),
}

impl From<GraphError<NodeError>> for MaterialError {
    fn from(error: GraphError<NodeError>) -> Self {
        Self::Graph(error)
    }
}

impl fmt::Display for MaterialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateLabel(label) => write!(f, "label {label:?} is already used"),
            Self::UnknownNode => f.write_str("unknown node, or a node of another kind"),
            Self::Graph(error) => write!(f, "{error:?}"),
        }
    }
}

impl core::error::Error for MaterialError {}

/// Runs dapple nodes for [`execution_graph`] and tracks their tiles.
#[derive(Debug)]
pub struct DappleExecutor {
    tiles: Tiles,
}

impl DappleExecutor {
    /// An executor splitting rasters into `tile_size`-texel tiles.
    #[must_use]
    pub fn new(tile_size: u32) -> Self {
        Self {
            tiles: Tiles::new(tile_size.max(1)),
        }
    }

    /// Tile work since the current or last [`MaterialGraph::run`] began.
    #[must_use]
    pub const fn tile_report(&self) -> TileReport {
        self.tiles.report
    }

    /// Limits the tiles one run recomputes; see
    /// [`MaterialGraph::set_tile_budget`].
    pub fn set_tile_budget(&mut self, budget: Option<u64>) {
        self.tiles.budget = budget;
    }
}

impl Default for DappleExecutor {
    fn default() -> Self {
        Self::new(DEFAULT_TILE_SIZE)
    }
}

fn field_input(value: &GraphValue) -> Result<&Arc<FieldValue>, NodeError> {
    match value {
        GraphValue::Field(field) => Ok(field),
        _ => Err(NodeError::WrongValue { expected: "field" }),
    }
}

fn raster_input(value: &GraphValue) -> Result<&Arc<RasterValue>, NodeError> {
    match value {
        GraphValue::Raster(raster) => Ok(raster),
        _ => Err(NodeError::WrongValue { expected: "raster" }),
    }
}

fn run_field(
    state: &mut FieldState,
    report: &mut TileReport,
    op: &Op,
    inputs: &[GraphValue],
) -> Result<GraphValue, NodeError> {
    let mut builder = ProgramBuilder::new();
    let mut imported = Vec::with_capacity(inputs.len());
    let mut fields = Vec::with_capacity(inputs.len());
    for input in inputs {
        let field = field_input(input)?;
        imported.push(
            builder
                .import(field.program.program())
                .map_err(NodeError::Program)?,
        );
        fields.push(field);
    }
    let mut missing = None;
    let mapped = op.map_inputs(|operand| {
        imported
            .get(operand.index() as usize)
            .copied()
            .unwrap_or_else(|| {
                missing = Some(operand.index());
                operand
            })
    });
    if let Some(index) = missing {
        return Err(NodeError::MissingOperand { index });
    }
    let output = builder.add(mapped).map_err(NodeError::Program)?;
    let program = builder.finish_value(output).map_err(NodeError::Program)?;

    // Where the new program can differ from this node's previous one: the op
    // edit's own region, plus each changed input's region where the op reads
    // that input pointwise, or grown by a warp's static reach.
    let mut change = match &state.op {
        Some(previous) => op.change_from(previous),
        None => Change::Everywhere,
    };
    let consumed: Vec<Fingerprint> = fields.iter().map(|f| f.program.fingerprint()).collect();
    for (i, field) in fields.iter().enumerate() {
        let last = state.inputs.get(i).copied();
        let input_change = if last == Some(consumed[i]) && state.inputs.len() == fields.len() {
            Change::Nowhere
        } else if last.is_some() && field.previous == last {
            field.change.clone()
        } else {
            Change::Everywhere
        };
        let input_change = match (input_change, op) {
            (Change::Nowhere, _) => Change::Nowhere,
            (change, Op::Warp { input, .. }) if input.index() as usize == i => {
                let warped = warp_change(op, &fields, change.clone());
                if matches!(change, Change::Within { .. }) && warped == Change::Everywhere {
                    report.unbounded_warps += 1;
                }
                warped
            }
            (_, Op::Transform { .. } | Op::Demote { .. }) => Change::Everywhere,
            (change, _) => change,
        };
        change = change.union(input_change);
    }
    let previous = state.output;
    let fingerprint = program.fingerprint();
    *state = FieldState {
        op: Some(op.clone()),
        inputs: consumed,
        output: Some(fingerprint),
    };
    Ok(GraphValue::Field(Arc::new(FieldValue {
        program,
        previous,
        change: if previous.is_some() {
            change
        } else {
            Change::Everywhere
        },
    })))
}

/// The change a warp's output makes of its warped input's `change`.
///
/// The warp reads its input at `p + amount · (dx, dy)`, so a changed input
/// region reaches output points up to `|amount| · max|dx|` (and `dy`) away,
/// and at a footprint up to `1 + |amount| · max(slope)` times wider (see
/// [`Op::Warp`]). Both come from the displacements' static bounds
/// ([`FieldProgram::bounds`](dapple_field::program::FieldProgram::bounds));
/// without them the change is `Everywhere`.
fn warp_change(op: &Op, fields: &[&Arc<FieldValue>], change: Change) -> Change {
    let Op::Warp { dx, dy, amount, .. } = *op else {
        return Change::Everywhere;
    };
    let bounds = |id: FieldNode| {
        fields
            .get(id.index() as usize)
            .map(|field| field.program.program().bounds())
    };
    let (Some(x), Some(y)) = (bounds(dx), bounds(dy)) else {
        return Change::Everywhere;
    };
    match (x.max_abs(), y.max_abs(), x.slope, y.slope) {
        (Some(mx), Some(my), Some(sx), Some(sy)) => {
            // Rounded outward so the f32 products cannot fall short.
            let a = amount.abs();
            let reach = Vec2::new((a * mx).next_up(), (a * my).next_up());
            let stretch = (1.0 + a * sx.max(sy)).next_up();
            change.warped(reach, stretch)
        }
        _ => Change::Everywhere,
    }
}

/// Content fingerprint of a realization of the program `program`.
fn realize_fingerprint(program: Fingerprint, width: u32, height: u32) -> u64 {
    let [lo, hi] = fingerprint_words(program);
    hash(
        0x0072_6561_6c69_7a65, // "realize"
        &[lo, hi, u64::from(width), u64::from(height)],
    )
}

/// Content fingerprint of the normals of the program `program`.
fn normals_fingerprint(program: Fingerprint, width: u32, height: u32, scale: f32) -> u64 {
    let [lo, hi] = fingerprint_words(program);
    hash(
        0x006e_6f72_6d61_6c73, // "normals"
        &[
            lo,
            hi,
            u64::from(width),
            u64::from(height),
            u64::from(scale.to_bits()),
        ],
    )
}

/// Content fingerprint of `params` applied to a raster fingerprinted `input`.
fn raster_fingerprint(params: RasterParams, input: u64) -> u64 {
    let f = |v: f32| u64::from(v.to_bits());
    let (tag, words) = match params {
        RasterParams::Blur(op) => (0, vec![f(op.sigma)]),
        RasterParams::HeightToNormal(op) => (1, vec![f(op.scale)]),
        RasterParams::AmbientOcclusion(op) => {
            (2, vec![f(op.radius), u64::from(op.directions), f(op.scale)])
        }
        RasterParams::DistanceTransform(op) => (3, vec![f(op.threshold)]),
    };
    let mut key = vec![tag, input];
    key.extend(words);
    hash(0x7261_7374_6572_6f70, &key)
}

/// Content fingerprint of the next mip level, under `filter`, of a raster
/// fingerprinted `input`.
fn mip_fingerprint(filter: Filter, input: u64) -> u64 {
    let tag = match filter {
        Filter::Box => 0,
        Filter::Kaiser => 1,
    };
    hash(0x0000_006d_6970_6d61, &[tag, input]) // "mipma"
}

/// The derivation of a sample node's image: a hash of its rasters'
/// fingerprints, base first.
fn sample_derivation(rasters: &[u64]) -> Fingerprint {
    let mut words = vec![0x0073_616d_706c_6500, rasters.len() as u64]; // "sample"
    words.extend_from_slice(rasters);
    let [lo, hi] = [0x7361_6d70_6c65_2d30, 0x7361_6d70_6c65_2d31] // "sample-0", "sample-1"
        .map(|seed| hash(seed, &words));
    Fingerprint((u128::from(hi) << 64) | u128::from(lo))
}

fn fingerprint_words(fp: Fingerprint) -> [u64; 2] {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "splitting the 128-bit fingerprint into its halves"
    )]
    [fp.0 as u64, (fp.0 >> 64) as u64]
}

/// Tiles holding a texel whose center lies within `region` grown by `pad`,
/// or `None` when the region is not finite.
fn region_tiles(
    grid: Grid,
    realization: &Realization,
    region: &Region,
    pad: f32,
) -> Option<Vec<u32>> {
    let texel = realization.texel();
    let origin = realization.origin();
    let lo = (region.origin - Vec2::splat(pad) - origin) / texel - Vec2::splat(0.5);
    let hi = (region.origin + region.size + Vec2::splat(pad) - origin) / texel - Vec2::splat(0.5);
    if !(lo.is_finite() && hi.is_finite()) {
        return None;
    }
    let limit = f64::from(u32::MAX);
    let to_index = |v: f32| {
        let v = f64::from(v).clamp(-limit, limit);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "clamped to the u32 range before the cast"
        )]
        let i = v as i64;
        i
    };
    // Conservative by one texel on each side of the exact center test.
    let (x0, y0) = (
        to_index(libm::floorf(lo.x)) - 1,
        to_index(libm::floorf(lo.y)) - 1,
    );
    let (x1, y1) = (
        to_index(libm::ceilf(hi.x)) + 2,
        to_index(libm::ceilf(hi.y)) + 2,
    );
    Some(grid.overlapping(x0, x1, y0, y1))
}

/// Which tiles of a node's grid to recompute: `None` for all of them.
type Dirty = Option<Vec<u32>>;

/// The tiles of a realization of `field` to recompute, given the program
/// fingerprint the previous output was realized from (`None` when there is
/// no reusable output). Counts a recomputation an unbounded change forces
/// over a previous output.
///
/// A texel's value, or its gradient, reads the field within half a footprint
/// of its center, so the change regions grow by that much.
fn field_dirty(
    tiles: &mut Tiles,
    grid: Grid,
    realization: &Realization,
    field: &FieldValue,
    previous: Option<Fingerprint>,
    fingerprint: Fingerprint,
) -> Dirty {
    let last = previous?;
    let dirty = if last == fingerprint {
        Some(Vec::new())
    } else if field.previous == Some(last) {
        match &field.change {
            Change::Nowhere => Some(Vec::new()),
            Change::Within {
                regions,
                footprint_scale,
            } => {
                // A warp widens the footprint its input sees, so a region
                // grows by that many half texels.
                let pad = realization.texel().max_element() * 0.5 * footprint_scale;
                let mut dirty = Vec::new();
                let mut bounded = true;
                for region in regions {
                    match region_tiles(grid, realization, region, pad) {
                        Some(t) => dirty.extend(t),
                        None => bounded = false,
                    }
                }
                dirty.sort_unstable();
                dirty.dedup();
                bounded.then_some(dirty)
            }
            Change::Everywhere => None,
        }
    } else {
        None
    };
    if dirty.is_none() {
        tiles.report.unbounded_changes += 1;
    }
    dirty
}

/// Rewrites the tiles of `dirty` in a copy of `previous` (or computes
/// everything), then reports which tiles' bits changed.
fn refresh<T: Copy>(
    grid: Grid,
    dirty: &Dirty,
    previous: Option<&Raster<T>>,
    whole: impl FnOnce() -> Result<Raster<T>, NodeError>,
    mut tile: impl FnMut(TexelRect, &mut Raster<T>) -> Result<(), NodeError>,
) -> Result<(Raster<T>, Vec<u32>), NodeError> {
    match (dirty, previous) {
        (Some(tiles), Some(previous)) => {
            let mut raster = previous.clone();
            for &t in tiles {
                tile(grid.rect(t), &mut raster)?;
            }
            Ok((raster, tiles.clone()))
        }
        _ => Ok((whole()?, (0..grid.count()).collect())),
    }
}

/// Keeps `value` as the node's output, shared with the graph, and notes
/// tiles a budget left pending so the node runs again.
fn keep(tiles: &mut Tiles, node: u32, state: &mut TileState, value: RasterValue) -> GraphValue {
    if !state.pending.is_empty() {
        tiles.report.pending_tiles += state.pending.len() as u64;
        tiles.pending_nodes.push(node);
    }
    let value = Arc::new(value);
    state.output = Some(Arc::clone(&value));
    GraphValue::Raster(value)
}

/// Keeps the new output, reports the work, and marks the dependents of the
/// tiles whose bits changed. Returns whether any changed.
fn settle(
    tiles: &mut Tiles,
    node: u32,
    state: &mut TileState,
    output: &RasterData,
    recomputed: &[u32],
    whole: bool,
) -> bool {
    let grid = output.grid(tiles.size());
    let previous = state
        .output
        .as_ref()
        .map(|v| &v.data)
        .filter(|previous| previous.same_grid(output));
    if previous.is_none() || state.keys.len() != grid.count() as usize {
        state.keys = tiles.keys(node, grid, &state.keys);
    }
    let changed: Vec<u32> = recomputed
        .iter()
        .copied()
        .filter(|&t| previous.is_none_or(|p| !p.rect_bits_eq(output, grid.rect(t))))
        .collect();
    let report = &mut tiles.report;
    report.tiles_recomputed += recomputed.len() as u64;
    report.tiles_reused += u64::from(grid.count()) - recomputed.len() as u64;
    report.tiles_changed += changed.len() as u64;
    report.texels_recomputed += recomputed.iter().map(|&t| grid.rect(t).area()).sum::<u64>();
    report.whole_recomputes += u64::from(whole);
    let keys: Vec<InternId> = changed.iter().map(|&t| state.keys[t as usize]).collect();
    tiles.mark_dependents(keys);
    !changed.is_empty()
}

fn run_realize(
    tiles: &mut Tiles,
    node: u32,
    state: &mut TileState,
    width: u32,
    height: u32,
    input: &GraphValue,
) -> Result<GraphValue, NodeError> {
    let field_value = field_input(input)?;
    let program = &field_value.program;
    let (port, domain) = (program.output_type(), program.domain());
    if !matches!(domain, Domain::Periodic { .. }) {
        return Err(NodeError::NotRealizable { port, domain });
    }
    let realization = Realization::period(domain, width, height).map_err(NodeError::Raster)?;
    let fingerprint = program.fingerprint();
    if !port.is_scalar() {
        let derivation = realize_fingerprint(fingerprint, width, height);
        return run_whole(tiles, node, state, derivation, port, true, || {
            realize_value(program, realization).map_err(NodeError::Typed)
        });
    }
    let field = program.channel(0).map_err(NodeError::Program)?;
    let grid = Grid::new(width, height, tiles.size(), Edge::Wrap);
    let previous = match (&state.source, state.output.as_ref().map(|v| &v.data)) {
        (
            Some(Source::Realize {
                program: last,
                width: w,
                height: h,
            }),
            Some(RasterData::Scalar(raster)),
        ) if (*w, *h) == (width, height) => Some((*last, raster)),
        _ => None,
    };
    let dirty = field_dirty(
        tiles,
        grid,
        &realization,
        field_value,
        previous.map(|(last, _)| last),
        fingerprint,
    );
    let dirty = tiles.schedule(&mut state.pending, dirty, grid.count());
    let (raster, recomputed) = refresh(
        grid,
        &dirty,
        previous.map(|(_, raster)| raster),
        || realize(&field, realization).map_err(NodeError::Raster),
        |rect, raster| realize_into(&field, realization, rect, raster).map_err(NodeError::Raster),
    )?;
    let data = RasterData::Scalar(raster);
    let changed = settle(tiles, node, state, &data, &recomputed, dirty.is_none());
    let value = RasterValue {
        data,
        port,
        fingerprint: realize_fingerprint(fingerprint, width, height),
        changed,
        tile_space: node,
    };
    state.source = Some(Source::Realize {
        program: fingerprint,
        width,
        height,
    });
    Ok(keep(tiles, node, state, value))
}

fn run_normals(
    tiles: &mut Tiles,
    node: u32,
    state: &mut TileState,
    (width, height, scale): (u32, u32, f32),
    input: &GraphValue,
) -> Result<GraphValue, NodeError> {
    let field_value = field_input(input)?;
    let program = &field_value.program;
    let (port, domain) = (program.output_type(), program.domain());
    if !port.is_scalar() || !matches!(domain, Domain::Periodic { .. }) {
        return Err(NodeError::NotRealizable { port, domain });
    }
    let field = program.channel(0).map_err(NodeError::Program)?;
    let realization = Realization::period(domain, width, height).map_err(NodeError::Raster)?;
    let fingerprint = program.fingerprint();
    let grid = Grid::new(width, height, tiles.size(), Edge::Wrap);
    let previous = match (&state.source, state.output.as_ref().map(|v| &v.data)) {
        (
            Some(Source::Normals {
                program: last,
                width: w,
                height: h,
                scale: k,
            }),
            Some(RasterData::Vector3(raster)),
        ) if (*w, *h, *k) == (width, height, scale.to_bits()) => Some((*last, raster)),
        _ => None,
    };
    let dirty = field_dirty(
        tiles,
        grid,
        &realization,
        field_value,
        previous.map(|(last, _)| last),
        fingerprint,
    );
    let dirty = tiles.schedule(&mut state.pending, dirty, grid.count());
    let (raster, recomputed) = refresh(
        grid,
        &dirty,
        previous.map(|(_, raster)| raster),
        || realize_normals(&field, realization, scale).map_err(NodeError::Raster),
        |rect, raster| {
            realize_normals_into(&field, realization, scale, rect, raster)
                .map_err(NodeError::Raster)
        },
    )?;
    let data = RasterData::Vector3(raster);
    let changed = settle(tiles, node, state, &data, &recomputed, dirty.is_none());
    let value = RasterValue {
        data,
        port: PortType::Normal(NormalFrame::Domain),
        fingerprint: normals_fingerprint(fingerprint, width, height, scale),
        changed,
        tile_space: node,
    };
    state.source = Some(Source::Normals {
        program: fingerprint,
        width,
        height,
        scale: scale.to_bits(),
    });
    Ok(keep(tiles, node, state, value))
}

/// Runs one raster operation tile-wise: `recompute` returns the new data and
/// the recomputed tiles.
fn run_op<O>(
    op: &O,
    input: &Raster,
    grid: Grid,
    dirty: &Dirty,
    previous: Option<&Raster<O::Output>>,
) -> Result<(Raster<O::Output>, Vec<u32>), NodeError>
where
    O: RasterOp,
{
    refresh(
        grid,
        dirty,
        previous,
        || op.apply(input).map_err(NodeError::Raster),
        |rect, raster| {
            op.apply_into(input, rect, raster)
                .map_err(NodeError::Raster)
        },
    )
}

fn run_raster(
    tiles: &mut Tiles,
    node: u32,
    state: &mut TileState,
    params: RasterParams,
    input: &GraphValue,
) -> Result<GraphValue, NodeError> {
    let value = raster_input(input)?;
    let (RasterData::Scalar(raster), PortType::Scalar | PortType::Mask) = (&value.data, value.port)
    else {
        return Err(NodeError::TypeRefused {
            operation: raster_name(params),
            port: value.port,
        });
    };
    let port = raster_output(params, value.port);
    let grid = Grid::new(raster.width(), raster.height(), tiles.size(), raster.edge());
    let footprint = match params {
        RasterParams::Blur(op) => op.footprint(raster.texel()),
        RasterParams::HeightToNormal(op) => op.footprint(raster.texel()),
        RasterParams::AmbientOcclusion(op) => op.footprint(raster.texel()),
        RasterParams::DistanceTransform(op) => op.footprint(raster.texel()),
    };
    let source = Source::Raster {
        params,
        input: value.tile_space,
    };
    let same = state.source.as_ref() == Some(&source)
        && state
            .output
            .as_ref()
            .is_some_and(|out| out.data.grid(tiles.size()) == grid);
    if !same || state.keys.len() != grid.count() as usize {
        // A new operation, input, or grid: re-key and re-wire the tiles.
        state.keys = tiles.keys(node, grid, &state.keys);
        match footprint {
            Some([fx, fy]) => {
                let (fx, fy) = (i64::from(fx), i64::from(fy));
                for t in 0..grid.count() {
                    let r = grid.rect(t);
                    let inputs: Vec<InternId> = grid
                        .overlapping(
                            i64::from(r.x0) - fx,
                            i64::from(r.x1) + fx,
                            i64::from(r.y0) - fy,
                            i64::from(r.y1) + fy,
                        )
                        .into_iter()
                        .map(|i| tiles.key(value.tile_space, i))
                        .collect();
                    tiles.depend(state.keys[t as usize], inputs);
                }
            }
            None => tiles.detach(&state.keys),
        }
    }
    let marked = tiles.take_marked(&state.keys);
    let dirty: Dirty = if !same {
        None
    } else if footprint.is_none() {
        // Global: pending tiles cannot occur, as every run recomputes whole.
        if value.changed {
            None
        } else {
            Some(Vec::new())
        }
    } else {
        Some(marked)
    };
    let dirty = tiles.schedule(&mut state.pending, dirty, grid.count());
    let previous_output = state.output.clone();
    let previous_scalar = match previous_output.as_ref().map(|v| &v.data) {
        Some(RasterData::Scalar(r)) if same => Some(r),
        _ => None,
    };
    let (data, recomputed) = match params {
        RasterParams::Blur(op) => {
            let (r, t) = run_op(&op, raster, grid, &dirty, previous_scalar)?;
            (RasterData::Scalar(r), t)
        }
        RasterParams::HeightToNormal(op) => {
            let previous = match previous_output.as_ref().map(|v| &v.data) {
                Some(RasterData::Vector3(r)) if same => Some(r),
                _ => None,
            };
            let (r, t) = run_op(&op, raster, grid, &dirty, previous)?;
            (RasterData::Vector3(r), t)
        }
        RasterParams::AmbientOcclusion(op) => {
            let (r, t) = run_op(&op, raster, grid, &dirty, previous_scalar)?;
            (RasterData::Scalar(r), t)
        }
        RasterParams::DistanceTransform(op) => {
            let (r, t) = run_op(&op, raster, grid, &dirty, previous_scalar)?;
            (RasterData::Scalar(r), t)
        }
    };
    if !same {
        state.output = None;
    }
    let changed = settle(tiles, node, state, &data, &recomputed, dirty.is_none());
    let out = RasterValue {
        data,
        port,
        fingerprint: raster_fingerprint(params, value.fingerprint),
        changed,
        tile_space: node,
    };
    state.source = Some(source);
    Ok(keep(tiles, node, state, out))
}

/// The next mip level's size along one axis, as `dapple_encode` halves it.
const fn next_size(size: u32) -> u32 {
    if size > 1 { size / 2 } else { 1 }
}

fn run_mip(
    tiles: &mut Tiles,
    node: u32,
    state: &mut TileState,
    filter: Filter,
    input: &GraphValue,
) -> Result<GraphValue, NodeError> {
    let value = raster_input(input)?;
    let (RasterData::Scalar(raster), PortType::Scalar | PortType::Mask) = (&value.data, value.port)
    else {
        return Err(NodeError::TypeRefused {
            operation: "mip filtering",
            port: value.port,
        });
    };
    let (sw, sh, edge) = (raster.width(), raster.height(), raster.edge());
    let (dw, dh) = (next_size(sw), next_size(sh));
    let source_grid = Grid::new(sw, sh, tiles.size(), edge);
    let grid = Grid::new(dw, dh, tiles.size(), edge);
    let source = Source::Mip {
        filter,
        input: value.tile_space,
    };
    let same = state.source.as_ref() == Some(&source)
        && state
            .output
            .as_ref()
            .is_some_and(|out| out.data.grid(tiles.size()) == grid);
    if !same || state.keys.len() != grid.count() as usize {
        // A new filter, input, or size: re-key and re-wire the tiles to the
        // level-above tiles their filter taps read.
        state.keys = tiles.keys(node, grid, &state.keys);
        for t in 0..grid.count() {
            let r = grid.rect(t);
            let (x0, x1) = source_span(filter, sw, r.x0, r.x1);
            let (y0, y1) = source_span(filter, sh, r.y0, r.y1);
            let inputs: Vec<InternId> = source_grid
                .overlapping(x0, x1, y0, y1)
                .into_iter()
                .map(|i| tiles.key(value.tile_space, i))
                .collect();
            tiles.depend(state.keys[t as usize], inputs);
        }
    }
    let marked = tiles.take_marked(&state.keys);
    let dirty: Dirty = if same { Some(marked) } else { None };
    let dirty = tiles.schedule(&mut state.pending, dirty, grid.count());
    let image = Image::new(sw, sh, 1, edge, raster.values().to_vec()).map_err(NodeError::Mip)?;
    let previous = match state.output.as_ref().map(|v| &v.data) {
        Some(RasterData::Scalar(r)) if same => Some(r),
        _ => None,
    };
    let (next, recomputed) = match (&dirty, previous) {
        (Some(dirty), Some(previous)) => {
            let mut next =
                Image::new(dw, dh, 1, edge, previous.values().to_vec()).map_err(NodeError::Mip)?;
            for &t in dirty {
                next_level_into(&image, filter, grid.rect(t), &mut next).map_err(NodeError::Mip)?;
            }
            (next, dirty.clone())
        }
        _ => (next_level(&image, filter), (0..grid.count()).collect()),
    };
    // The level covers the same region with fewer, larger texels.
    #[expect(
        clippy::cast_precision_loss,
        reason = "mip sizes are far below f32's exact integer range"
    )]
    let scale = Vec2::new(sw as f32 / dw as f32, sh as f32 / dh as f32);
    let level = Raster::from_values(
        dw,
        dh,
        raster.origin(),
        raster.texel() * scale,
        edge,
        next.values().to_vec(),
    )
    .map_err(NodeError::Raster)?;
    if !same {
        state.output = None;
    }
    let data = RasterData::Scalar(level);
    let changed = settle(tiles, node, state, &data, &recomputed, dirty.is_none());
    let out = RasterValue {
        data,
        port: value.port,
        fingerprint: mip_fingerprint(filter, value.fingerprint),
        changed,
        tile_space: node,
    };
    state.source = Some(source);
    Ok(keep(tiles, node, state, out))
}

/// The type of a raster operation's output for an input of type `port`.
///
/// Every operation reads one scalar channel, so each accepts scalars and
/// masks (a mask's coverage is a valid height or distance feature) and
/// refuses every other type.
const fn raster_output(params: RasterParams, port: PortType) -> PortType {
    match params {
        // Blurring a mask gives a mask: the mean of values in [0, 1].
        RasterParams::Blur(_) => port,
        RasterParams::HeightToNormal(_) => PortType::Normal(NormalFrame::Domain),
        RasterParams::AmbientOcclusion(_) | RasterParams::DistanceTransform(_) => PortType::Scalar,
    }
}

const fn raster_name(params: RasterParams) -> &'static str {
    match params {
        RasterParams::Blur(_) => "Gaussian blur",
        RasterParams::HeightToNormal(_) => "height to normal",
        RasterParams::AmbientOcclusion(_) => "ambient occlusion",
        RasterParams::DistanceTransform(_) => "distance transform",
    }
}

/// Content fingerprint of level `level` of a raster fingerprinted `input`
/// under `policy`.
fn reduce_fingerprint(policy: ReductionPolicy, level: u32, input: u64) -> u64 {
    let [tag, word] = policy.fingerprint();
    hash(0x7265_6475_6365, &[tag, word, u64::from(level), input]) // "reduce"
}

/// Runs a node whose raster is computed whole: reuses the previous output
/// when `derivation` is unchanged and `stale` is false, and otherwise
/// recomputes every tile, comparing bits so unchanged tiles stop
/// propagating.
fn run_whole(
    tiles: &mut Tiles,
    node: u32,
    state: &mut TileState,
    derivation: u64,
    port: PortType,
    stale: bool,
    compute: impl FnOnce() -> Result<TypedRaster, NodeError>,
) -> Result<GraphValue, NodeError> {
    let source = Source::Whole { derivation };
    if !stale
        && state.source.as_ref() == Some(&source)
        && let Some(previous) = &state.output
    {
        let grid = previous.data.grid(tiles.size());
        tiles.report.tiles_reused += u64::from(grid.count());
        let value = RasterValue {
            changed: false,
            ..RasterValue::clone(previous)
        };
        return Ok(keep(tiles, node, state, value));
    }
    let raster = compute()?;
    debug_assert_eq!(raster.port(), port, "a whole raster keeps its node's type");
    let data = RasterData::Typed(raster);
    let all: Vec<u32> = (0..data.grid(tiles.size()).count()).collect();
    let changed = settle(tiles, node, state, &data, &all, true);
    let value = RasterValue {
        data,
        port,
        fingerprint: derivation,
        changed,
        tile_space: node,
    };
    state.source = Some(source);
    Ok(keep(tiles, node, state, value))
}

fn run_reduce(
    tiles: &mut Tiles,
    node: u32,
    state: &mut TileState,
    (policy, level): (ReductionPolicy, u32),
    input: &GraphValue,
) -> Result<GraphValue, NodeError> {
    let value = raster_input(input)?;
    policy.check(value.port).map_err(NodeError::Typed)?;
    if level == 0 {
        return Err(NodeError::WrongValue {
            expected: "a mip level of at least 1",
        });
    }
    let derivation = reduce_fingerprint(policy, level, value.fingerprint);
    run_whole(
        tiles,
        node,
        state,
        derivation,
        value.port,
        value.changed,
        || {
            let base = value.data.to_typed(value.port)?;
            dapple_raster::typed::reduce(&base, policy, level).map_err(NodeError::Typed)
        },
    )
}

/// The domain an image of `raster` samples over: the period a wrapping
/// raster covers, or the plane.
fn image_domain(raster: &Raster) -> Result<Domain, NodeError> {
    let invalid = NodeError::Program(ProgramError::Domain(DomainError::InvalidParameter {
        name: "image",
    }));
    match raster.edge() {
        Edge::Clamp => Ok(Domain::Plane),
        Edge::Wrap => {
            #[expect(
                clippy::cast_precision_loss,
                reason = "raster sizes are far below f32's exact integer range"
            )]
            let covered = Vec2::new(raster.width() as f32, raster.height() as f32) * raster.texel();
            let whole = |v: f32| {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "checked to be a small positive whole number first"
                )]
                let n = v as u32;
                ((1.0..=16_777_216.0).contains(&v) && libm::floorf(v) == v).then_some(n)
            };
            let (x, y) = (whole(covered.x), whole(covered.y));
            x.zip(y)
                .and_then(|(x, y)| Domain::periodic(x, y))
                .ok_or(invalid)
        }
    }
}

fn run_sample(
    tiles: &mut Tiles,
    node: u32,
    state: &mut SampleState,
    inputs: &[GraphValue],
) -> Result<GraphValue, NodeError> {
    let domain_error = |e: DomainError| NodeError::Program(ProgramError::Domain(e));
    let mut rasters = Vec::with_capacity(inputs.len());
    for input in inputs {
        let value = raster_input(input)?;
        let (RasterData::Scalar(raster), PortType::Scalar | PortType::Mask) =
            (&value.data, value.port)
        else {
            return Err(NodeError::TypeRefused {
                operation: "sampling",
                port: value.port,
            });
        };
        rasters.push((value, raster));
    }
    let Some(&(_, base)) = rasters.first() else {
        return Err(NodeError::WrongValue { expected: "raster" });
    };
    let domain = image_domain(base)?;
    let mut levels = Vec::with_capacity(rasters.len());
    for (_, raster) in &rasters {
        levels.push(
            ImageLevel::new(
                raster.width(),
                raster.height(),
                raster.texel(),
                raster.values().to_vec(),
            )
            .map_err(domain_error)?,
        );
    }
    let fingerprints: Vec<u64> = rasters.iter().map(|(v, _)| v.fingerprint).collect();
    let image = SampleImage::new(
        domain,
        base.origin(),
        levels,
        sample_derivation(&fingerprints),
    )
    .map_err(domain_error)?;
    let mut builder = ProgramBuilder::new();
    let output = builder
        .add(Op::Sample { image })
        .map_err(NodeError::Program)?;
    let program = builder.finish_value(output).map_err(NodeError::Program)?;

    // Mirror each sampled raster's tiles, so the marks its changed tiles
    // leave say where the image changed.
    let grids: Vec<(u32, Grid)> = rasters
        .iter()
        .map(|(value, raster)| {
            (
                value.tile_space,
                Grid::new(raster.width(), raster.height(), tiles.size(), raster.edge()),
            )
        })
        .collect();
    let same = state.output.is_some()
        && state.levels.len() == grids.len()
        && state
            .levels
            .iter()
            .zip(&grids)
            .all(|((space, grid, _), (s, g))| space == s && grid == g);
    if !same {
        let old = core::mem::take(&mut state.levels);
        for (_, _, keys) in &old[grids.len().min(old.len())..] {
            tiles.detach(keys);
        }
        for (level, &(space, grid)) in grids.iter().enumerate() {
            let previous = old
                .get(level)
                .map_or(&[][..], |(_, _, keys)| keys.as_slice());
            let level_index = u8::try_from(level).unwrap_or(u8::MAX);
            let keys = tiles.level_keys(node, level_index, grid, previous);
            for (t, &key) in keys.iter().enumerate() {
                let input = tiles.key(space, u32::try_from(t).expect("tile counts fit u32"));
                tiles.depend(key, [input]);
            }
            state.levels.push((space, grid, keys));
        }
    }
    let mut change = Change::Nowhere;
    for (level, (_, grid, keys)) in state.levels.iter().enumerate() {
        let marked = tiles.take_marked(keys);
        let raster = rasters[level].1;
        let (origin, texel) = (raster.origin(), raster.texel());
        let regions: Vec<Region> = marked
            .into_iter()
            .map(|t| {
                let r = grid.rect(t);
                // A texel reaches points within one texel of its center.
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "texel indices are far below f32's exact integer range"
                )]
                let (lo, hi) = (
                    Vec2::new(r.x0 as f32 - 1.0, r.y0 as f32 - 1.0),
                    Vec2::new(r.x1 as f32 + 1.0, r.y1 as f32 + 1.0),
                );
                Region {
                    origin: origin + lo * texel,
                    size: (hi - lo) * texel,
                }
            })
            .collect();
        if !regions.is_empty() {
            change = change.union(Change::within(regions));
        }
    }
    let previous = state.output;
    state.output = Some(program.fingerprint());
    Ok(GraphValue::Field(Arc::new(FieldValue {
        program,
        previous,
        change: if same { change } else { Change::Everywhere },
    })))
}

impl Executor for DappleExecutor {
    type Value = GraphValue;
    type Node = DappleNode;
    type Error = NodeError;

    fn execute(
        &mut self,
        node: &mut DappleNode,
        inputs: &[GraphValue],
        outputs: &mut Vec<GraphValue>,
        _access: &mut NodeAccess<'_>,
    ) -> Result<(), NodeError> {
        let GraphValue::Params(params) = &inputs[0] else {
            return Err(NodeError::WrongValue { expected: "params" });
        };
        let upstream = &inputs[1..];
        let value = match (node.kind, params.as_ref()) {
            (NodeKind::Field, Params::Field(op)) => {
                run_field(&mut node.field, &mut self.tiles.report, op, upstream)?
            }
            (NodeKind::Realize, Params::Realize { width, height }) => run_realize(
                &mut self.tiles,
                node.key,
                &mut node.tiles,
                *width,
                *height,
                &upstream[0],
            )?,
            (NodeKind::Raster, Params::Raster(params)) => run_raster(
                &mut self.tiles,
                node.key,
                &mut node.tiles,
                *params,
                &upstream[0],
            )?,
            (
                NodeKind::Normals,
                Params::Normals {
                    width,
                    height,
                    scale,
                },
            ) => run_normals(
                &mut self.tiles,
                node.key,
                &mut node.tiles,
                (*width, *height, *scale),
                &upstream[0],
            )?,
            (NodeKind::Mip, Params::Mip(filter)) => run_mip(
                &mut self.tiles,
                node.key,
                &mut node.tiles,
                *filter,
                &upstream[0],
            )?,
            (NodeKind::Reduce, Params::Reduce { policy, level }) => run_reduce(
                &mut self.tiles,
                node.key,
                &mut node.tiles,
                (*policy, *level),
                &upstream[0],
            )?,
            (NodeKind::Sample, Params::Sample) => {
                run_sample(&mut self.tiles, node.key, &mut node.sample, upstream)?
            }
            _ => {
                return Err(NodeError::WrongValue {
                    expected: "params of the node's kind",
                });
            }
        };
        outputs.push(value);
        Ok(())
    }

    /// Early cutoff: an output equal to the node's previous one stops its
    /// dependents from re-running.
    ///
    /// A field program is equal when its fingerprint is: fingerprints are
    /// structural, so equal programs evaluate identically everywhere. A
    /// raster is equal when its fingerprint is, it lies on the same grid in
    /// the same tile space, and no recomputed tile's bits changed
    /// ([`RasterValue::changed`]); tiles not recomputed are copied from the
    /// previous output, so the texels are identical.
    ///
    /// Raster fingerprints identify a raster's derivation, which
    /// [`Recipe::fingerprints`] predicts, so an edit that changes a
    /// derivation never cuts off even when the texels come out the same.
    /// Tile tracking covers that case instead: the node marks no tiles
    /// changed, so its dependents re-run but recompute no tiles.
    fn values_equal(&self, previous: &GraphValue, next: &GraphValue) -> bool {
        match (previous, next) {
            (GraphValue::Field(a), GraphValue::Field(b)) => {
                a.program.fingerprint() == b.program.fingerprint()
            }
            (GraphValue::Raster(a), GraphValue::Raster(b)) => {
                a.fingerprint == b.fingerprint
                    && a.tile_space == b.tile_space
                    && !b.changed
                    && a.data.same_grid(&b.data)
            }
            (GraphValue::Params(a), GraphValue::Params(b)) => a == b,
            _ => false,
        }
    }

    fn describe(&self, node: &DappleNode) -> Option<String> {
        Some(format!("{:?}", node.kind))
    }
}

/// Output name of every node.
const OUT: &str = "out";

#[derive(Clone, Debug)]
struct Entry {
    kind: NodeKind,
    label: String,
    params: Params,
    upstream: Vec<NodeId>,
}

/// An incremental material graph; see the [crate docs](crate).
#[derive(Debug)]
pub struct MaterialGraph {
    graph: ExecutionGraph<DappleExecutor>,
    entries: BTreeMap<NodeId, Entry>,
    order: Vec<NodeId>,
}

impl Default for MaterialGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl MaterialGraph {
    /// An empty graph with [`DEFAULT_TILE_SIZE`] tiles.
    #[must_use]
    pub fn new() -> Self {
        Self::with_tile_size(DEFAULT_TILE_SIZE)
    }

    /// An empty graph splitting rasters into `tile_size`-texel square tiles
    /// (at least 1).
    #[must_use]
    pub fn with_tile_size(tile_size: u32) -> Self {
        Self {
            graph: ExecutionGraph::new(DappleExecutor::new(tile_size)),
            entries: BTreeMap::new(),
            order: Vec::new(),
        }
    }

    fn params_name(label: &str) -> String {
        format!("{label}.params")
    }

    fn add(
        &mut self,
        kind: NodeKind,
        label: &str,
        params: Params,
        upstream: &[NodeId],
    ) -> Result<NodeId, MaterialError> {
        if self.entries.values().any(|e| e.label == label) {
            return Err(MaterialError::DuplicateLabel(label.into()));
        }
        if upstream.iter().any(|n| !self.entries.contains_key(n)) {
            return Err(MaterialError::UnknownNode);
        }
        let params_name = Self::params_name(label);
        let mut inputs: Vec<Box<str>> = vec![params_name.clone().into()];
        inputs.extend((0..upstream.len()).map(|i| Box::from(format!("{label}.in{i}"))));
        let body = DappleNode {
            kind,
            key: u32::try_from(self.entries.len()).expect("fewer than 2^32 nodes"),
            field: FieldState::default(),
            tiles: TileState::default(),
            sample: SampleState::default(),
        };
        let node = self.graph.add_node(body, inputs, vec![OUT.into()])?;
        self.graph.set_node_label(node, label)?;
        self.graph.set_input_value(
            node,
            params_name,
            GraphValue::Params(Arc::new(params.clone())),
        )?;
        for (i, &from) in upstream.iter().enumerate() {
            self.graph
                .connect(from, OUT, node, format!("{label}.in{i}"))?;
        }
        self.entries.insert(
            node,
            Entry {
                kind,
                label: label.into(),
                params,
                upstream: upstream.to_vec(),
            },
        );
        self.order.push(node);
        Ok(node)
    }

    /// Adds a field node computing `op` over the fields of `upstream`, which
    /// its operands name by position ([`operand`]).
    pub fn field(
        &mut self,
        label: &str,
        op: Op,
        upstream: &[NodeId],
    ) -> Result<NodeId, MaterialError> {
        self.add(NodeKind::Field, label, Params::Field(op), upstream)
    }

    /// Adds a node realizing the scalar periodic field of `field` over one
    /// period at `width` × `height` texels.
    pub fn realize(
        &mut self,
        label: &str,
        field: NodeId,
        width: u32,
        height: u32,
    ) -> Result<NodeId, MaterialError> {
        self.add(
            NodeKind::Realize,
            label,
            Params::Realize { width, height },
            &[field],
        )
    }

    /// Adds a node realizing the normals of the height field `scale * field`
    /// over one period at `width` × `height` texels, from the field's
    /// gradient (`dapple_raster::realize_normals`).
    pub fn normals(
        &mut self,
        label: &str,
        field: NodeId,
        (width, height): (u32, u32),
        scale: f32,
    ) -> Result<NodeId, MaterialError> {
        self.add(
            NodeKind::Normals,
            label,
            Params::Normals {
                width,
                height,
                scale,
            },
            &[field],
        )
    }

    /// Adds a node applying `params` to the scalar raster of `raster`.
    pub fn raster(
        &mut self,
        label: &str,
        params: RasterParams,
        raster: NodeId,
    ) -> Result<NodeId, MaterialError> {
        self.add(NodeKind::Raster, label, Params::Raster(params), &[raster])
    }

    /// Adds a node filtering the scalar raster of `raster` down one mip
    /// level with `filter`: half the size, rounding down, at least 1, bit
    /// for bit as each level of `dapple_encode::data_mips`. Chain mip nodes
    /// for a whole chain.
    pub fn mip(
        &mut self,
        label: &str,
        raster: NodeId,
        filter: Filter,
    ) -> Result<NodeId, MaterialError> {
        self.add(NodeKind::Mip, label, Params::Mip(filter), &[raster])
    }

    /// Adds a node reducing the raster of `raster` to mip level `level` (at
    /// least 1) under `policy`, computed from the level-0 texels each output
    /// texel covers (`dapple_raster::typed::reduce`).
    ///
    /// The policy is explicit: the node fails when it is not meaningful for
    /// the raster's type ([`NodeError::Typed`]), and it is part of the
    /// output's fingerprint. `ReductionPolicy::default_for` names the usual
    /// choice for a type. Reduce nodes compute whole.
    ///
    /// # Errors
    ///
    /// [`MaterialError::DuplicateLabel`] or [`MaterialError::UnknownNode`].
    pub fn reduce(
        &mut self,
        label: &str,
        raster: NodeId,
        policy: ReductionPolicy,
        level: u32,
    ) -> Result<NodeId, MaterialError> {
        self.add(
            NodeKind::Reduce,
            label,
            Params::Reduce { policy, level },
            &[raster],
        )
    }

    /// Changes a reduce node's policy and level.
    pub fn set_reduction(
        &mut self,
        node: NodeId,
        policy: ReductionPolicy,
        level: u32,
    ) -> Result<(), MaterialError> {
        self.set_params(node, NodeKind::Reduce, Params::Reduce { policy, level })
    }

    /// Adds a node sampling the scalar rasters of `levels` as a field: the
    /// base raster first, then its mip chain, finest to coarsest (mip nodes
    /// chained from the base, for example). With one level the field is not
    /// band-limited; with mips it is filtered by footprint (see
    /// `dapple_field::SampleImage`).
    ///
    /// A wrapping raster, such as a realization over one period, samples as
    /// a periodic field over that period; a clamping one as a plane field.
    pub fn sample(&mut self, label: &str, levels: &[NodeId]) -> Result<NodeId, MaterialError> {
        self.add(NodeKind::Sample, label, Params::Sample, levels)
    }

    fn set_params(
        &mut self,
        node: NodeId,
        kind: NodeKind,
        params: Params,
    ) -> Result<(), MaterialError> {
        let entry = self
            .entries
            .get_mut(&node)
            .ok_or(MaterialError::UnknownNode)?;
        if entry.kind != kind {
            return Err(MaterialError::UnknownNode);
        }
        let name = Self::params_name(&entry.label);
        entry.params = params.clone();
        self.graph
            .set_input_value(node, name.clone(), GraphValue::Params(Arc::new(params)))?;
        self.graph.invalidate_input(name);
        Ok(())
    }

    /// Replaces a field node's op; the next run re-runs it and its
    /// dependents.
    pub fn set_field_op(&mut self, node: NodeId, op: Op) -> Result<(), MaterialError> {
        self.set_params(node, NodeKind::Field, Params::Field(op))
    }

    /// Changes a realize node's resolution.
    pub fn set_resolution(
        &mut self,
        node: NodeId,
        width: u32,
        height: u32,
    ) -> Result<(), MaterialError> {
        self.set_params(node, NodeKind::Realize, Params::Realize { width, height })
    }

    /// Changes a normals node's resolution and height scale.
    pub fn set_normals(
        &mut self,
        node: NodeId,
        (width, height): (u32, u32),
        scale: f32,
    ) -> Result<(), MaterialError> {
        self.set_params(
            node,
            NodeKind::Normals,
            Params::Normals {
                width,
                height,
                scale,
            },
        )
    }

    /// Changes a mip node's filter.
    pub fn set_mip_filter(&mut self, node: NodeId, filter: Filter) -> Result<(), MaterialError> {
        self.set_params(node, NodeKind::Mip, Params::Mip(filter))
    }

    /// Replaces a raster node's operation.
    pub fn set_raster_params(
        &mut self,
        node: NodeId,
        params: RasterParams,
    ) -> Result<(), MaterialError> {
        self.set_params(node, NodeKind::Raster, Params::Raster(params))
    }

    /// Runs every node whose inputs changed since the last run.
    ///
    /// Realize and raster nodes recompute only the tiles their input changes
    /// reach; [`MaterialGraph::tile_report`] counts the work. Dependents of a
    /// node whose output did not change are cut off
    /// ([`RunSummary::cut_off_nodes`]); see the [crate docs](crate).
    pub fn run(&mut self) -> Result<RunSummary, MaterialError> {
        self.graph.executor_mut().tiles.begin_run();
        let summary = self.graph.run_all()?;
        // Nodes a tile budget left unfinished run again next time.
        let pending = core::mem::take(&mut self.graph.executor_mut().tiles.pending_nodes);
        for key in pending {
            let node = self.order[key as usize];
            self.graph
                .invalidate_input(Self::params_name(&self.entries[&node].label));
        }
        Ok(summary)
    }

    /// Limits each [`MaterialGraph::run`] to recomputing about `budget`
    /// tiles, or removes the limit with `None`.
    ///
    /// Tiles past the budget stay pending in their node, which runs again
    /// on the next run, so repeated runs converge to exactly the unbudgeted
    /// result. Until then those nodes' rasters mix recomputed and stale
    /// tiles; [`TileReport::pending_tiles`] counts what is left. A node with
    /// no reusable output (a first run, a new grid or operation, a global
    /// operation, or an unbounded change) still computes whole, and that
    /// work counts against the budget.
    pub fn set_tile_budget(&mut self, budget: Option<u64>) {
        self.graph.executor_mut().set_tile_budget(budget);
    }

    /// Tile work done by the last [`MaterialGraph::run`].
    #[must_use]
    pub fn tile_report(&self) -> TileReport {
        self.graph.executor().tile_report()
    }

    /// The last output of `node`, if it has run.
    #[must_use]
    pub fn value(&self, node: NodeId) -> Option<&GraphValue> {
        self.graph.node_outputs(node)?.get(OUT)
    }

    /// The last field program of a field node.
    #[must_use]
    pub fn field_value(&self, node: NodeId) -> Option<&ValueProgram> {
        match self.value(node)? {
            GraphValue::Field(field) => Some(&field.program),
            _ => None,
        }
    }

    /// The last raster of a realize or raster node.
    #[must_use]
    pub fn raster_value(&self, node: NodeId) -> Option<&Arc<RasterValue>> {
        match self.value(node)? {
            GraphValue::Raster(raster) => Some(raster),
            _ => None,
        }
    }

    /// How many times `node` has run.
    #[must_use]
    pub fn run_count(&self, node: NodeId) -> Option<u64> {
        self.graph.node_run_count(node)
    }

    /// The underlying execution graph, for reports and DOT output.
    #[must_use]
    pub const fn execution_graph(&self) -> &ExecutionGraph<DappleExecutor> {
        &self.graph
    }
}

#[cfg(test)]
mod tests;
