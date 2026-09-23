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
//! - **Realize nodes** turn a scalar field into a raster over one period of
//!   its periodic domain.
//! - **Raster nodes** apply a `dapple_raster` operation.
//!
//! Each node's parameters (the op, the resolution, the raster operation) are
//! an input of that node, named `<label>.params`. Editing them with
//! [`MaterialGraph::set_field_op`] or its siblings rebinds and invalidates only
//! that input, so the next [`MaterialGraph::run`] re-runs exactly the edited
//! node and its dependents. `execution_graph` schedules, records dependencies
//! and reports.
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
//!   pointwise ops carry their inputs' regions through; anything else makes
//!   the change unbounded.
//! - A realize node re-realizes the tiles whose texel centers fall in the
//!   change, grown by half its footprint, and recomputes whole on an
//!   unbounded change ([`TileReport::unbounded_changes`]).
//! - Each raster-node tile depends, through an `invalidation` tracker, on
//!   the input tiles its kernel footprint reads. A node compares every
//!   recomputed tile with its previous bits and marks the dependents of the
//!   tiles that changed, so unchanged tiles stop propagating. Global
//!   operations, such as the distance transform, recompute whole when their
//!   input changed.
//!
//! Tile-wise results equal whole recomputation bit for bit.
//! [`MaterialGraph::tile_report`] counts the work of the last run.
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

use dapple_field::hash::hash;
use dapple_field::program::{
    Change, Fingerprint, NodeId as FieldNode, Op, ProgramBuilder, ProgramError, ValueProgram,
};
use dapple_field::raster::Region;
use dapple_field::{Domain, PortType};
use dapple_raster::{
    AmbientOcclusion, DistanceTransform, Edge, GaussianBlur, HeightToNormal, Raster, RasterError,
    RasterOp, Realization, TexelRect, realize, realize_into,
};
use execution_graph::{ExecutionGraph, Executor, GraphError, NodeAccess, NodeId, RunSummary};
use glam::Vec2;
use invalidation::intern::InternId;

mod tiles;

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
}

/// A realized raster: scalar, or three-channel (normals).
#[derive(Clone, Debug, PartialEq)]
pub enum RasterData {
    /// One channel.
    Scalar(Raster),
    /// Three channels.
    Vector3(Raster<[f32; 3]>),
}

impl RasterData {
    fn same_grid(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Scalar(a), Self::Scalar(b)) => a.same_grid(b),
            (Self::Vector3(a), Self::Vector3(b)) => a.same_grid(b),
            _ => false,
        }
    }

    fn rect_bits_eq(&self, other: &Self, rect: TexelRect) -> bool {
        match (self, other) {
            (Self::Scalar(a), Self::Scalar(b)) => a.rect_bits_eq(b, rect),
            (Self::Vector3(a), Self::Vector3(b)) => a.rect_bits_eq(b, rect),
            _ => false,
        }
    }

    fn grid(&self, tile_size: u32) -> Grid {
        let (w, h, edge) = match self {
            Self::Scalar(r) => (r.width(), r.height(), r.edge()),
            Self::Vector3(r) => (r.width(), r.height(), r.edge()),
        };
        Grid::new(w, h, tile_size, edge)
    }
}

/// A raster value and its content fingerprint.
#[derive(Clone, Debug, PartialEq)]
pub struct RasterValue {
    /// The texels.
    pub data: RasterData,
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
    output: Option<RasterData>,
    keys: Vec<InternId>,
}

#[derive(Clone, Debug, PartialEq)]
enum Source {
    Realize {
        program: Fingerprint,
        width: u32,
        height: u32,
    },
    Raster {
        params: RasterParams,
        input: u32,
    },
}

/// A graph node: its kind, its tile key space, and its state between runs.
#[derive(Clone, Debug)]
pub struct DappleNode {
    kind: NodeKind,
    key: u32,
    field: FieldState,
    tiles: TileState,
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
    // that input pointwise.
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
        let input_change = if input_change != Change::Nowhere && !op.operand_is_pointwise(i) {
            Change::Everywhere
        } else {
            input_change
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
    if !port.is_scalar() || !matches!(domain, Domain::Periodic { .. }) {
        return Err(NodeError::NotRealizable { port, domain });
    }
    let field = program.channel(0).map_err(NodeError::Program)?;
    let realization = Realization::period(domain, width, height).map_err(NodeError::Raster)?;
    let fingerprint = program.fingerprint();
    let grid = Grid::new(width, height, tiles.size(), Edge::Wrap);
    let previous = match (&state.source, &state.output) {
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
    let dirty: Dirty = match previous {
        None => None,
        Some((last, _)) if last == fingerprint => Some(Vec::new()),
        Some((last, _)) if field_value.previous == Some(last) => match &field_value.change {
            Change::Nowhere => Some(Vec::new()),
            Change::Within(regions) => {
                let pad = realization.texel().max_element() * 0.5;
                let mut dirty = Vec::new();
                let mut bounded = true;
                for region in regions {
                    match region_tiles(grid, &realization, region, pad) {
                        Some(t) => dirty.extend(t),
                        None => bounded = false,
                    }
                }
                dirty.sort_unstable();
                dirty.dedup();
                bounded.then_some(dirty)
            }
            Change::Everywhere => None,
        },
        Some(_) => None,
    };
    if previous.is_some() && dirty.is_none() {
        tiles.report.unbounded_changes += 1;
    }
    let (raster, recomputed) = refresh(
        grid,
        &dirty,
        previous.map(|(_, raster)| raster),
        || realize(&field, realization).map_err(NodeError::Raster),
        |rect, raster| realize_into(&field, realization, rect, raster).map_err(NodeError::Raster),
    )?;
    let data = RasterData::Scalar(raster);
    let changed = settle(tiles, node, state, &data, &recomputed, dirty.is_none());
    let [lo, hi] = fingerprint_words(fingerprint);
    let value = RasterValue {
        data,
        fingerprint: hash(
            0x0072_6561_6c69_7a65, // "realize"
            &[lo, hi, u64::from(width), u64::from(height)],
        ),
        changed,
        tile_space: node,
    };
    state.source = Some(Source::Realize {
        program: fingerprint,
        width,
        height,
    });
    state.output = Some(value.data.clone());
    Ok(GraphValue::Raster(Arc::new(value)))
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
    let RasterData::Scalar(raster) = &value.data else {
        return Err(NodeError::WrongValue {
            expected: "scalar raster",
        });
    };
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
            .is_some_and(|out| out.grid(tiles.size()) == grid);
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
        if value.changed {
            None
        } else {
            Some(Vec::new())
        }
    } else {
        Some(marked)
    };
    let f = |v: f32| u64::from(v.to_bits());
    let previous_scalar = match &state.output {
        Some(RasterData::Scalar(r)) if same => Some(r),
        _ => None,
    };
    let (tag, words, data, recomputed) = match params {
        RasterParams::Blur(op) => {
            let (r, t) = run_op(&op, raster, grid, &dirty, previous_scalar)?;
            (0, vec![f(op.sigma)], RasterData::Scalar(r), t)
        }
        RasterParams::HeightToNormal(op) => {
            let previous = match &state.output {
                Some(RasterData::Vector3(r)) if same => Some(r),
                _ => None,
            };
            let (r, t) = run_op(&op, raster, grid, &dirty, previous)?;
            (1, vec![f(op.scale)], RasterData::Vector3(r), t)
        }
        RasterParams::AmbientOcclusion(op) => {
            let (r, t) = run_op(&op, raster, grid, &dirty, previous_scalar)?;
            (
                2,
                vec![f(op.radius), u64::from(op.directions), f(op.scale)],
                RasterData::Scalar(r),
                t,
            )
        }
        RasterParams::DistanceTransform(op) => {
            let (r, t) = run_op(&op, raster, grid, &dirty, previous_scalar)?;
            (3, vec![f(op.threshold)], RasterData::Scalar(r), t)
        }
    };
    if !same {
        state.output = None;
    }
    let changed = settle(tiles, node, state, &data, &recomputed, dirty.is_none());
    let mut key = vec![tag, value.fingerprint];
    key.extend(words);
    let out = RasterValue {
        data,
        fingerprint: hash(0x7261_7374_6572_6f70, &key),
        changed,
        tile_space: node,
    };
    state.source = Some(source);
    state.output = Some(out.data.clone());
    Ok(GraphValue::Raster(Arc::new(out)))
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
            (NodeKind::Field, Params::Field(op)) => run_field(&mut node.field, op, upstream)?,
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
            _ => {
                return Err(NodeError::WrongValue {
                    expected: "params of the node's kind",
                });
            }
        };
        outputs.push(value);
        Ok(())
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
}

/// An incremental material graph; see the [crate docs](crate).
#[derive(Debug)]
pub struct MaterialGraph {
    graph: ExecutionGraph<DappleExecutor>,
    entries: BTreeMap<NodeId, Entry>,
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
        };
        let node = self.graph.add_node(body, inputs, vec![OUT.into()])?;
        self.graph.set_node_label(node, label)?;
        self.graph
            .set_input_value(node, params_name, GraphValue::Params(Arc::new(params)))?;
        for (i, &from) in upstream.iter().enumerate() {
            self.graph
                .connect(from, OUT, node, format!("{label}.in{i}"))?;
        }
        self.entries.insert(
            node,
            Entry {
                kind,
                label: label.into(),
            },
        );
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

    /// Adds a node applying `params` to the scalar raster of `raster`.
    pub fn raster(
        &mut self,
        label: &str,
        params: RasterParams,
        raster: NodeId,
    ) -> Result<NodeId, MaterialError> {
        self.add(NodeKind::Raster, label, Params::Raster(params), &[raster])
    }

    fn set_params(
        &mut self,
        node: NodeId,
        kind: NodeKind,
        params: Params,
    ) -> Result<(), MaterialError> {
        let entry = self.entries.get(&node).ok_or(MaterialError::UnknownNode)?;
        if entry.kind != kind {
            return Err(MaterialError::UnknownNode);
        }
        let name = Self::params_name(&entry.label);
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
    /// reach; [`MaterialGraph::tile_report`] counts the work.
    pub fn run(&mut self) -> Result<RunSummary, MaterialError> {
        self.graph.executor_mut().tiles.report = TileReport::default();
        Ok(self.graph.run_all()?)
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
