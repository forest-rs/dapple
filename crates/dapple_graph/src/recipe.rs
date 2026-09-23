// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Material recipes: a [`MaterialGraph`] as plain data.
//!
//! A [`Recipe`] lists labeled nodes in dependency order and names which
//! raster nodes feed which material roles. It is inspectable as it stands,
//! serializes with the `serde` feature, and computes every node's content
//! fingerprint without realizing anything: the same fingerprints the built
//! graph produces when it runs. Tools can therefore cache baked textures by
//! recipe content.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use dapple_encode::Filter;
use dapple_field::hash::hash;
use dapple_field::program::{Fingerprint, Op, sample_fingerprint};
use execution_graph::NodeId;

use crate::{
    MaterialError, MaterialGraph, NodeKind, Params, RasterParams, fingerprint_words,
    mip_fingerprint, normals_fingerprint, raster_fingerprint, realize_fingerprint,
    reduce_fingerprint, sample_derivation,
};
use dapple_raster::typed::ReductionPolicy;

/// The recipe format version this crate reads and writes.
///
/// A reader refuses other versions. The version changes when a recipe's
/// meaning or its fingerprints change.
pub const RECIPE_VERSION: u32 = 1;

/// What a node computes, and from which labeled inputs.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "kind", rename_all = "snake_case"))]
pub enum Step {
    /// A field node: `op` over the fields of `inputs`, which its operands
    /// name by position ([`operand`](crate::operand)).
    Field {
        /// The operation.
        op: Op,
        /// Labels of the upstream field nodes, in operand order.
        inputs: Vec<String>,
    },
    /// A realize node over one period of its field.
    Realize {
        /// Label of the field node.
        input: String,
        /// Texels per row.
        width: u32,
        /// Rows.
        height: u32,
    },
    /// A raster node.
    Raster {
        /// Label of the realize, normals or raster node.
        input: String,
        /// The operation.
        params: RasterParams,
    },
    /// A mip node ([`MaterialGraph::mip`]).
    Mip {
        /// Label of the realize, raster or mip node.
        input: String,
        /// The filter.
        filter: Filter,
    },
    /// A reduce node ([`MaterialGraph::reduce`]).
    Reduce {
        /// Label of the level-0 raster node.
        input: String,
        /// How texels combine.
        policy: ReductionPolicy,
        /// The mip level, at least 1.
        level: u32,
    },
    /// A sample node ([`MaterialGraph::sample`]): rasters read back as a
    /// field.
    Sample {
        /// Labels of the base raster node and its mip nodes, finest first.
        inputs: Vec<String>,
    },
    /// A normals node over one period of its field
    /// ([`MaterialGraph::normals`]).
    Normals {
        /// Label of the field node.
        input: String,
        /// Texels per row.
        width: u32,
        /// Rows.
        height: u32,
        /// Domain units of height per field value unit.
        scale: f32,
    },
}

/// One labeled node.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RecipeNode {
    /// The node's label, unique in the recipe.
    pub label: String,
    /// What it computes.
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub step: Step,
}

/// A material role and the raster nodes that fill it, channel by channel.
///
/// Roles are the consumer's names, for example `dapple_encode`'s material
/// map names (`base_color`, `normal`, `specular_roughness`); this crate does
/// not interpret them.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RecipeOutput {
    /// The role.
    pub role: String,
    /// Labels of realize or raster nodes, one per channel group, in order.
    pub channels: Vec<String>,
}

/// A material graph as data.
///
/// A recipe lists labeled nodes in dependency order and names which raster
/// nodes feed which material roles. It computes every node's content
/// fingerprint without realizing anything ([`Recipe::fingerprints`]), the
/// same fingerprints the built graph produces when it runs, so tools can
/// cache baked textures by recipe content. With the `serde` feature it
/// serializes; [`MaterialGraph::recipe`] exports a graph's current state.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Recipe {
    /// Format version; must be [`RECIPE_VERSION`].
    pub version: u32,
    /// Nodes, each after the nodes it reads.
    pub nodes: Vec<RecipeNode>,
    /// Material roles and their sources.
    #[cfg_attr(feature = "serde", serde(default))]
    pub outputs: Vec<RecipeOutput>,
}

impl Default for Recipe {
    fn default() -> Self {
        Self {
            version: RECIPE_VERSION,
            nodes: Vec::new(),
            outputs: Vec::new(),
        }
    }
}

/// A node's content fingerprint: its field program's, or its raster's.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum NodeFingerprint {
    /// A field node's program fingerprint.
    Field(Fingerprint),
    /// A realize or raster node's [`RasterValue::fingerprint`](crate::RasterValue::fingerprint).
    Raster(u64),
}

/// A recipe that cannot be read, fingerprinted, or built.
#[derive(Debug)]
pub enum RecipeError {
    /// The recipe's version is not [`RECIPE_VERSION`].
    Version {
        /// The recipe's version.
        found: u32,
    },
    /// Two nodes share a label.
    DuplicateLabel(String),
    /// A node or output names a label no earlier node has.
    UnknownLabel(String),
    /// A node reads a node of the wrong kind, for example a realize node
    /// reading a raster.
    WrongInput {
        /// The reading node or output role.
        at: String,
        /// The label it reads.
        input: String,
    },
    /// A field node's op holds an image ([`Op::Sample`]), which recipes
    /// cannot carry; sample rasters with a [`Step::Sample`] node instead.
    EmbeddedImage(String),
    /// Building the graph failed.
    Material(MaterialError),
}

impl fmt::Display for RecipeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Version { found } => write!(
                f,
                "recipe version {found} is not the supported version {RECIPE_VERSION}"
            ),
            Self::DuplicateLabel(label) => write!(f, "label {label:?} is used twice"),
            Self::UnknownLabel(label) => write!(f, "no earlier node is labeled {label:?}"),
            Self::WrongInput { at, input } => {
                write!(f, "{at:?} cannot read {input:?}, a node of another kind")
            }
            Self::EmbeddedImage(label) => write!(
                f,
                "field node {label:?} holds an image; sample rasters with a sample node"
            ),
            Self::Material(error) => error.fmt(f),
        }
    }
}

impl core::error::Error for RecipeError {}

impl From<MaterialError> for RecipeError {
    fn from(error: MaterialError) -> Self {
        Self::Material(error)
    }
}

impl Step {
    const fn kind(&self) -> NodeKind {
        match self {
            Self::Field { .. } => NodeKind::Field,
            Self::Realize { .. } => NodeKind::Realize,
            Self::Raster { .. } => NodeKind::Raster,
            Self::Normals { .. } => NodeKind::Normals,
            Self::Mip { .. } => NodeKind::Mip,
            Self::Reduce { .. } => NodeKind::Reduce,
            Self::Sample { .. } => NodeKind::Sample,
        }
    }
}

/// Whether nodes of `kind` output a field.
const fn is_field(kind: NodeKind) -> bool {
    matches!(kind, NodeKind::Field | NodeKind::Sample)
}

impl Recipe {
    /// Checks the version, label uniqueness, and that every node reads earlier
    /// nodes of a fitting kind, returning each label's kind.
    fn check(&self) -> Result<BTreeMap<&str, NodeKind>, RecipeError> {
        if self.version != RECIPE_VERSION {
            return Err(RecipeError::Version {
                found: self.version,
            });
        }
        let mut kinds: BTreeMap<&str, NodeKind> = BTreeMap::new();
        for node in &self.nodes {
            let input_kind = |label: &String| {
                kinds
                    .get(label.as_str())
                    .copied()
                    .ok_or_else(|| RecipeError::UnknownLabel(label.clone()))
            };
            let wrong = |input: &String| RecipeError::WrongInput {
                at: node.label.clone(),
                input: input.clone(),
            };
            match &node.step {
                Step::Field { op, inputs } => {
                    if matches!(op, Op::Sample { .. }) {
                        return Err(RecipeError::EmbeddedImage(node.label.clone()));
                    }
                    for input in inputs {
                        if !is_field(input_kind(input)?) {
                            return Err(wrong(input));
                        }
                    }
                }
                Step::Realize { input, .. } | Step::Normals { input, .. } => {
                    if !is_field(input_kind(input)?) {
                        return Err(wrong(input));
                    }
                }
                Step::Raster { input, .. }
                | Step::Mip { input, .. }
                | Step::Reduce { input, .. } => {
                    if is_field(input_kind(input)?) {
                        return Err(wrong(input));
                    }
                }
                Step::Sample { inputs } => {
                    if inputs.is_empty() {
                        return Err(RecipeError::WrongInput {
                            at: node.label.clone(),
                            input: String::new(),
                        });
                    }
                    for input in inputs {
                        if is_field(input_kind(input)?) {
                            return Err(wrong(input));
                        }
                    }
                }
            }
            if kinds
                .insert(node.label.as_str(), node.step.kind())
                .is_some()
            {
                return Err(RecipeError::DuplicateLabel(node.label.clone()));
            }
        }
        for output in &self.outputs {
            for channel in &output.channels {
                match kinds.get(channel.as_str()) {
                    None => return Err(RecipeError::UnknownLabel(channel.clone())),
                    Some(NodeKind::Field | NodeKind::Sample) => {
                        return Err(RecipeError::WrongInput {
                            at: output.role.clone(),
                            input: channel.clone(),
                        });
                    }
                    Some(_) => {}
                }
            }
        }
        Ok(kinds)
    }

    /// Every node's content fingerprint, by label, computed from the recipe
    /// alone.
    ///
    /// Each equals what the built graph produces for that node when it runs:
    /// the field program's fingerprint, or the raster's
    /// [`RasterValue::fingerprint`](crate::RasterValue::fingerprint).
    ///
    /// # Errors
    ///
    /// As [`Recipe::build`], except that ops are not validated.
    pub fn fingerprints(&self) -> Result<BTreeMap<String, NodeFingerprint>, RecipeError> {
        self.check()?;
        let mut out: BTreeMap<String, NodeFingerprint> = BTreeMap::new();
        for node in &self.nodes {
            let fingerprint = match &node.step {
                Step::Field { op, inputs } => {
                    let inputs: Vec<Fingerprint> = inputs
                        .iter()
                        .map(|label| match out[label] {
                            NodeFingerprint::Field(fp) => fp,
                            NodeFingerprint::Raster(_) => {
                                unreachable!("checked: field nodes read fields")
                            }
                        })
                        .collect();
                    // The op's operands are placeholders naming input
                    // positions, so its inputs are those positions' programs.
                    let operands: Vec<Fingerprint> = op
                        .inputs()
                        .iter()
                        .map(|operand| {
                            inputs
                                .get(operand.index() as usize)
                                .copied()
                                .unwrap_or(Fingerprint(0))
                        })
                        .collect();
                    NodeFingerprint::Field(op.fingerprint_with(&operands))
                }
                Step::Realize {
                    input,
                    width,
                    height,
                } => match out[input] {
                    NodeFingerprint::Field(fp) => {
                        NodeFingerprint::Raster(realize_fingerprint(fp, *width, *height))
                    }
                    NodeFingerprint::Raster(_) => unreachable!("checked: realize reads a field"),
                },
                Step::Normals {
                    input,
                    width,
                    height,
                    scale,
                } => match out[input] {
                    NodeFingerprint::Field(fp) => {
                        NodeFingerprint::Raster(normals_fingerprint(fp, *width, *height, *scale))
                    }
                    NodeFingerprint::Raster(_) => unreachable!("checked: normals read a field"),
                },
                Step::Raster { input, params } => match out[input] {
                    NodeFingerprint::Raster(fp) => {
                        NodeFingerprint::Raster(raster_fingerprint(*params, fp))
                    }
                    NodeFingerprint::Field(_) => unreachable!("checked: rasters read rasters"),
                },
                Step::Mip { input, filter } => match out[input] {
                    NodeFingerprint::Raster(fp) => {
                        NodeFingerprint::Raster(mip_fingerprint(*filter, fp))
                    }
                    NodeFingerprint::Field(_) => unreachable!("checked: mips read rasters"),
                },
                Step::Reduce {
                    input,
                    policy,
                    level,
                } => match out[input] {
                    NodeFingerprint::Raster(fp) => {
                        NodeFingerprint::Raster(reduce_fingerprint(*policy, *level, fp))
                    }
                    NodeFingerprint::Field(_) => unreachable!("checked: reductions read rasters"),
                },
                Step::Sample { inputs } => {
                    let rasters: Vec<u64> = inputs
                        .iter()
                        .map(|label| match out[label] {
                            NodeFingerprint::Raster(fp) => fp,
                            NodeFingerprint::Field(_) => {
                                unreachable!("checked: sample nodes read rasters")
                            }
                        })
                        .collect();
                    NodeFingerprint::Field(sample_fingerprint(sample_derivation(&rasters)))
                }
            };
            out.insert(node.label.clone(), fingerprint);
        }
        Ok(out)
    }

    /// A content fingerprint of what the recipe produces.
    ///
    /// With outputs, it covers each role's name and its channels' node
    /// fingerprints, so relabeling nodes or adding unused ones leaves it
    /// unchanged. Without outputs, it covers every node's fingerprint in
    /// recipe order.
    ///
    /// # Errors
    ///
    /// As [`Recipe::fingerprints`].
    pub fn fingerprint(&self) -> Result<Fingerprint, RecipeError> {
        let nodes = self.fingerprints()?;
        let mut words = alloc::vec![u64::from(RECIPE_VERSION)];
        let mut push = |fp: NodeFingerprint| match fp {
            NodeFingerprint::Field(fp) => {
                words.push(0);
                words.extend(fingerprint_words(fp));
            }
            NodeFingerprint::Raster(fp) => words.extend([1, fp]),
        };
        if self.outputs.is_empty() {
            for node in &self.nodes {
                push(nodes[&node.label]);
            }
        } else {
            for output in &self.outputs {
                let role = output.role.as_bytes();
                push(NodeFingerprint::Raster(hash(
                    role.len() as u64,
                    &role.iter().map(|&b| u64::from(b)).collect::<Vec<_>>(),
                )));
                for channel in &output.channels {
                    push(nodes[channel]);
                }
            }
        }
        let [lo, hi] = [0x7265_6369_7065_2d30, 0x7265_6369_7065_2d31] // "recipe-0", "recipe-1"
            .map(|seed| hash(seed, &words));
        Ok(Fingerprint((u128::from(hi) << 64) | u128::from(lo)))
    }

    /// Builds the recipe's graph with `tile_size`-texel tiles, returning it
    /// and each label's node.
    ///
    /// # Errors
    ///
    /// [`RecipeError::Version`], [`RecipeError::DuplicateLabel`],
    /// [`RecipeError::UnknownLabel`] or [`RecipeError::WrongInput`] for a
    /// malformed recipe; [`RecipeError::Material`] when the graph rejects a
    /// node. Invalid ops fail when the graph runs.
    pub fn build(
        &self,
        tile_size: u32,
    ) -> Result<(MaterialGraph, BTreeMap<String, NodeId>), RecipeError> {
        self.check()?;
        let mut graph = MaterialGraph::with_tile_size(tile_size);
        let mut ids: BTreeMap<String, NodeId> = BTreeMap::new();
        for node in &self.nodes {
            let id = match &node.step {
                Step::Field { op, inputs } => {
                    let upstream: Vec<NodeId> = inputs.iter().map(|l| ids[l]).collect();
                    graph.field(&node.label, op.clone(), &upstream)?
                }
                Step::Realize {
                    input,
                    width,
                    height,
                } => graph.realize(&node.label, ids[input], *width, *height)?,
                Step::Raster { input, params } => graph.raster(&node.label, *params, ids[input])?,
                Step::Mip { input, filter } => graph.mip(&node.label, ids[input], *filter)?,
                Step::Reduce {
                    input,
                    policy,
                    level,
                } => graph.reduce(&node.label, ids[input], *policy, *level)?,
                Step::Sample { inputs } => {
                    let levels: Vec<NodeId> = inputs.iter().map(|l| ids[l]).collect();
                    graph.sample(&node.label, &levels)?
                }
                Step::Normals {
                    input,
                    width,
                    height,
                    scale,
                } => graph.normals(&node.label, ids[input], (*width, *height), *scale)?,
            };
            ids.insert(node.label.clone(), id);
        }
        Ok((graph, ids))
    }
}

impl MaterialGraph {
    /// The graph's nodes and their current parameters as a [`Recipe`], in
    /// the order they were added, with no outputs.
    #[must_use]
    pub fn recipe(&self) -> Recipe {
        let label = |id: &NodeId| self.entries[id].label.clone();
        let nodes = self
            .order
            .iter()
            .map(|id| {
                let entry = &self.entries[id];
                let step = match &entry.params {
                    Params::Field(op) => Step::Field {
                        op: op.clone(),
                        inputs: entry.upstream.iter().map(label).collect(),
                    },
                    Params::Realize { width, height } => Step::Realize {
                        input: label(&entry.upstream[0]),
                        width: *width,
                        height: *height,
                    },
                    Params::Raster(params) => Step::Raster {
                        input: label(&entry.upstream[0]),
                        params: *params,
                    },
                    Params::Mip(filter) => Step::Mip {
                        input: label(&entry.upstream[0]),
                        filter: *filter,
                    },
                    Params::Reduce { policy, level } => Step::Reduce {
                        input: label(&entry.upstream[0]),
                        policy: *policy,
                        level: *level,
                    },
                    Params::Sample => Step::Sample {
                        inputs: entry.upstream.iter().map(label).collect(),
                    },
                    Params::Normals {
                        width,
                        height,
                        scale,
                    } => Step::Normals {
                        input: label(&entry.upstream[0]),
                        width: *width,
                        height: *height,
                        scale: *scale,
                    },
                };
                RecipeNode {
                    label: entry.label.clone(),
                    step,
                }
            })
            .collect();
        Recipe {
            version: RECIPE_VERSION,
            nodes,
            outputs: Vec::new(),
        }
    }
}
