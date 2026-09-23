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
    Fingerprint, NodeId as FieldNode, Op, ProgramBuilder, ProgramError, ValueProgram,
};
use dapple_field::{Domain, PortType};
use dapple_raster::{
    AmbientOcclusion, DistanceTransform, GaussianBlur, HeightToNormal, Raster, RasterError,
    RasterOp, Realization, realize,
};
use execution_graph::{ExecutionGraph, Executor, GraphError, NodeAccess, NodeId, RunSummary};

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

/// A raster value and its content fingerprint.
#[derive(Clone, Debug, PartialEq)]
pub struct RasterValue {
    /// The texels.
    pub data: RasterData,
    /// Hash of the producing operation, its parameters, and its input's
    /// fingerprint.
    pub fingerprint: u64,
}

/// A value on a graph edge or input.
#[derive(Clone, Debug, PartialEq)]
pub enum GraphValue {
    /// Node parameters.
    Params(Arc<Params>),
    /// A field program.
    Field(Arc<ValueProgram>),
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

/// Runs dapple nodes for [`execution_graph`].
#[derive(Copy, Clone, Debug, Default)]
pub struct DappleExecutor;

fn field_input(value: &GraphValue) -> Result<&Arc<ValueProgram>, NodeError> {
    match value {
        GraphValue::Field(program) => Ok(program),
        _ => Err(NodeError::WrongValue { expected: "field" }),
    }
}

fn run_field(op: &Op, inputs: &[GraphValue]) -> Result<GraphValue, NodeError> {
    let mut builder = ProgramBuilder::new();
    let mut imported = Vec::with_capacity(inputs.len());
    for input in inputs {
        let program = field_input(input)?;
        imported.push(
            builder
                .import(program.program())
                .map_err(NodeError::Program)?,
        );
    }
    let mut missing = None;
    let op = op.map_inputs(|operand| {
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
    let output = builder.add(op).map_err(NodeError::Program)?;
    let program = builder.finish_value(output).map_err(NodeError::Program)?;
    Ok(GraphValue::Field(Arc::new(program)))
}

fn fingerprint_words(fp: Fingerprint) -> [u64; 2] {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "splitting the 128-bit fingerprint into its halves"
    )]
    [fp.0 as u64, (fp.0 >> 64) as u64]
}

fn run_realize(width: u32, height: u32, input: &GraphValue) -> Result<GraphValue, NodeError> {
    let program = field_input(input)?;
    let (port, domain) = (program.output_type(), program.domain());
    if !port.is_scalar() || !matches!(domain, Domain::Periodic { .. }) {
        return Err(NodeError::NotRealizable { port, domain });
    }
    let field = program.channel(0).map_err(NodeError::Program)?;
    let realization = Realization::period(domain, width, height).map_err(NodeError::Raster)?;
    let raster = realize(&field, realization).map_err(NodeError::Raster)?;
    let [lo, hi] = fingerprint_words(program.fingerprint());
    let fingerprint = hash(
        0x0072_6561_6c69_7a65, // "realize"
        &[lo, hi, u64::from(width), u64::from(height)],
    );
    Ok(GraphValue::Raster(Arc::new(RasterValue {
        data: RasterData::Scalar(raster),
        fingerprint,
    })))
}

fn run_raster(params: RasterParams, input: &GraphValue) -> Result<GraphValue, NodeError> {
    let GraphValue::Raster(value) = input else {
        return Err(NodeError::WrongValue { expected: "raster" });
    };
    let RasterData::Scalar(raster) = &value.data else {
        return Err(NodeError::WrongValue {
            expected: "scalar raster",
        });
    };
    let f = |v: f32| u64::from(v.to_bits());
    let (tag, words, data) = match params {
        RasterParams::Blur(op) => (
            0,
            vec![f(op.sigma)],
            RasterData::Scalar(op.apply(raster).map_err(NodeError::Raster)?),
        ),
        RasterParams::HeightToNormal(op) => (
            1,
            vec![f(op.scale)],
            RasterData::Vector3(op.apply(raster).map_err(NodeError::Raster)?),
        ),
        RasterParams::AmbientOcclusion(op) => (
            2,
            vec![f(op.radius), u64::from(op.directions), f(op.scale)],
            RasterData::Scalar(op.apply(raster).map_err(NodeError::Raster)?),
        ),
        RasterParams::DistanceTransform(op) => (
            3,
            vec![f(op.threshold)],
            RasterData::Scalar(op.apply(raster).map_err(NodeError::Raster)?),
        ),
    };
    let mut key = vec![tag, value.fingerprint];
    key.extend(words);
    Ok(GraphValue::Raster(Arc::new(RasterValue {
        data,
        fingerprint: hash(0x7261_7374_6572_6f70, &key),
    })))
}

impl Executor for DappleExecutor {
    type Value = GraphValue;
    type Node = NodeKind;
    type Error = NodeError;

    fn execute(
        &mut self,
        node: &mut NodeKind,
        inputs: &[GraphValue],
        outputs: &mut Vec<GraphValue>,
        _access: &mut NodeAccess<'_>,
    ) -> Result<(), NodeError> {
        let GraphValue::Params(params) = &inputs[0] else {
            return Err(NodeError::WrongValue { expected: "params" });
        };
        let upstream = &inputs[1..];
        let value = match (*node, params.as_ref()) {
            (NodeKind::Field, Params::Field(op)) => run_field(op, upstream)?,
            (NodeKind::Realize, Params::Realize { width, height }) => {
                run_realize(*width, *height, &upstream[0])?
            }
            (NodeKind::Raster, Params::Raster(params)) => run_raster(*params, &upstream[0])?,
            _ => {
                return Err(NodeError::WrongValue {
                    expected: "params of the node's kind",
                });
            }
        };
        outputs.push(value);
        Ok(())
    }

    fn describe(&self, node: &NodeKind) -> Option<String> {
        Some(format!("{node:?}"))
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
    /// An empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph: ExecutionGraph::new(DappleExecutor),
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
        let node = self.graph.add_node(kind, inputs, vec![OUT.into()])?;
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
    pub fn run(&mut self) -> Result<RunSummary, MaterialError> {
        Ok(self.graph.run_all()?)
    }

    /// The last output of `node`, if it has run.
    #[must_use]
    pub fn value(&self, node: NodeId) -> Option<&GraphValue> {
        self.graph.node_outputs(node)?.get(OUT)
    }

    /// The last field program of a field node.
    #[must_use]
    pub fn field_value(&self, node: NodeId) -> Option<&Arc<ValueProgram>> {
        match self.value(node)? {
            GraphValue::Field(program) => Some(program),
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
