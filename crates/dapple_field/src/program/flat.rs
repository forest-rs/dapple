// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Flat evaluation plans: each shared subgraph evaluated once per point.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use glam::Vec2;

use super::{FieldProgram, Kernel, Node, NodeId, probe_point, warp_move};
use crate::domain::Footprint;
use crate::field::Affine2;
use crate::types::Value;

/// How a context's point and footprint follow from its parent's.
#[derive(Clone, Debug, PartialEq)]
enum Move {
    Transform {
        transform: Affine2,
        stretch: f32,
    },
    /// Probe `k` of a warp: the parent's point one footprint away.
    Probe(usize),
    /// A warp: registers of `dx` and `dy` at the parent's point, then at each
    /// probe in [`super::PROBES`] order.
    Warp {
        center: [usize; 2],
        probes: [[usize; 2]; 4],
        amount: f32,
    },
}

#[derive(Clone, Debug, PartialEq)]
enum Step {
    /// Computes context `ctx` from `parent`.
    Context { ctx: usize, parent: usize, by: Move },
    /// Evaluates a pure node in `ctx` into register `reg`.
    Eval {
        reg: usize,
        node: NodeId,
        ctx: usize,
        args: [usize; 3],
        count: usize,
    },
}

/// A program flattened into a linear list of steps.
///
/// A *context* is the point and footprint a subgraph is evaluated at: the
/// output's, or one derived through transforms and warps. An *instance* is a
/// pure node in one context; each gets one register, so a subgraph shared by
/// several consumers in the same context is evaluated once.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Plan {
    steps: Vec<Step>,
    registers: usize,
    contexts: usize,
    output: usize,
    tree_evaluations: u64,
}

/// Evaluation counts of a [`FieldProgram`], for reports.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct EvaluationStats {
    /// Node evaluations per point with the flat plan: one per instance.
    pub instances: u64,
    /// Node evaluations per point when every consumer evaluates its inputs
    /// again, as recursive evaluation does with a nonzero footprint (a warp
    /// then also evaluates its displacements at four probes).
    pub tree_evaluations: u64,
    /// Distinct evaluation points per point: the output's, plus one per
    /// distinct transform or warp path.
    pub contexts: u64,
}

struct Compiler<'a> {
    nodes: &'a [Node],
    steps: Vec<Step>,
    registers: usize,
    instances: BTreeMap<(NodeId, usize), usize>,
    /// Contexts by parent, node, and role: 0 for a transform or warp, 1 to
    /// 4 for a warp's probes.
    contexts: BTreeMap<(usize, NodeId, usize), usize>,
    context_count: usize,
    counts: BTreeMap<(NodeId, usize), u64>,
}

impl Compiler<'_> {
    fn context(&mut self, parent: usize, node: NodeId, role: usize, by: Move) -> usize {
        if let Some(&ctx) = self.contexts.get(&(parent, node, role)) {
            return ctx;
        }
        let ctx = self.context_count;
        self.context_count += 1;
        self.contexts.insert((parent, node, role), ctx);
        self.steps.push(Step::Context { ctx, parent, by });
        ctx
    }

    /// The register holding `node`'s value in `ctx`, compiling it if needed.
    fn compile(&mut self, node: NodeId, ctx: usize) -> usize {
        if let Some(&reg) = self.instances.get(&(node, ctx)) {
            return reg;
        }
        let entry = &self.nodes[node.0 as usize];
        let reg = match entry.kernel {
            Kernel::Transform {
                input,
                transform,
                stretch,
            } => {
                let child = self.context(ctx, node, 0, Move::Transform { transform, stretch });
                self.compile(input, child)
            }
            Kernel::Pass(input) => self.compile(input, ctx),
            Kernel::Warp {
                input,
                dx,
                dy,
                amount,
            } => {
                let center = [self.compile(dx, ctx), self.compile(dy, ctx)];
                let mut probes = [[0; 2]; 4];
                for (k, probe) in probes.iter_mut().enumerate() {
                    let at = self.context(ctx, node, k + 1, Move::Probe(k));
                    *probe = [self.compile(dx, at), self.compile(dy, at)];
                }
                let child = self.context(
                    ctx,
                    node,
                    0,
                    Move::Warp {
                        center,
                        probes,
                        amount,
                    },
                );
                self.compile(input, child)
            }
            _ => {
                let mut args = [0; 3];
                let mut count = 0;
                for input in entry.op.inputs().iter() {
                    args[count] = self.compile(input, ctx);
                    count += 1;
                }
                let reg = self.registers;
                self.registers += 1;
                self.steps.push(Step::Eval {
                    reg,
                    node,
                    ctx,
                    args,
                    count,
                });
                reg
            }
        };
        self.instances.insert((node, ctx), reg);
        reg
    }

    /// Kernel evaluations recursive evaluation performs for `node` in `ctx`.
    fn tree_count(&mut self, node: NodeId, ctx: usize) -> u64 {
        if let Some(&count) = self.counts.get(&(node, ctx)) {
            return count;
        }
        let entry = &self.nodes[node.0 as usize];
        let count = match entry.kernel {
            Kernel::Transform { input, .. } | Kernel::Warp { input, .. } => {
                let child = self.contexts[&(ctx, node, 0)];
                let moved = self.tree_count(input, child);
                match entry.kernel {
                    Kernel::Warp { dx, dy, .. } => {
                        // The displacements at the point and at four probes.
                        let mut total = moved;
                        for role in 0..5 {
                            let at = if role == 0 {
                                ctx
                            } else {
                                self.contexts[&(ctx, node, role)]
                            };
                            total = total
                                .saturating_add(self.tree_count(dx, at))
                                .saturating_add(self.tree_count(dy, at));
                        }
                        total
                    }
                    _ => moved,
                }
            }
            Kernel::Pass(input) => self.tree_count(input, ctx),
            _ => entry.op.inputs().iter().fold(1_u64, |sum, input| {
                sum.saturating_add(self.tree_count(input, ctx))
            }),
        };
        self.counts.insert((node, ctx), count);
        count
    }
}

impl Plan {
    pub(super) fn new(nodes: &[Node], output: NodeId) -> Self {
        let mut compiler = Compiler {
            nodes,
            steps: Vec::new(),
            registers: 0,
            instances: BTreeMap::new(),
            contexts: BTreeMap::new(),
            context_count: 1,
            counts: BTreeMap::new(),
        };
        let register = compiler.compile(output, 0);
        let tree_evaluations = compiler.tree_count(output, 0);
        Self {
            steps: compiler.steps,
            registers: compiler.registers,
            contexts: compiler.context_count,
            output: register,
            tree_evaluations,
        }
    }

    pub(super) fn stats(&self) -> EvaluationStats {
        EvaluationStats {
            instances: self.registers as u64,
            tree_evaluations: self.tree_evaluations,
            contexts: self.contexts as u64,
        }
    }
}

/// Evaluates a program's output through its flat plan, reusing its buffers
/// across points.
///
/// Results are bit-identical to [`FieldProgram::eval_value`] on the output:
/// the same operations run on the same operands, only without repetition.
/// Obtain one from [`FieldProgram::evaluator`] and keep it for a batch of
/// points, such as a raster.
#[derive(Clone, Debug)]
pub struct Evaluator<'a> {
    program: &'a FieldProgram,
    registers: Vec<Value>,
    points: Vec<(Vec2, Footprint)>,
}

impl<'a> Evaluator<'a> {
    pub(super) fn new(program: &'a FieldProgram) -> Self {
        Self {
            program,
            registers: alloc::vec![Value::Scalar(0.0); program.plan.registers],
            points: alloc::vec![(Vec2::ZERO, Footprint::POINT); program.plan.contexts],
        }
    }

    /// The output's value at `p`.
    pub fn eval_value(&mut self, p: Vec2, footprint: Footprint) -> Value {
        let plan = &self.program.plan;
        self.points[0] = (p, footprint);
        for step in &plan.steps {
            match *step {
                Step::Context {
                    ctx,
                    parent,
                    ref by,
                } => {
                    let (p, footprint) = self.points[parent];
                    self.points[ctx] = match *by {
                        Move::Transform { transform, stretch } => {
                            (transform.apply(p), footprint.scaled(stretch))
                        }
                        Move::Probe(k) => (probe_point(p, footprint, k), footprint),
                        Move::Warp {
                            center,
                            probes,
                            amount,
                        } => {
                            let r = |reg: usize| self.registers[reg];
                            warp_move(
                                p,
                                footprint,
                                amount,
                                center.map(r),
                                probes.map(|pair| pair.map(r)),
                            )
                        }
                    };
                }
                Step::Eval {
                    reg,
                    node,
                    ctx,
                    args,
                    count,
                } => {
                    let (p, footprint) = self.points[ctx];
                    let mut values = [Value::Scalar(0.0); 3];
                    for (value, &arg) in values.iter_mut().zip(&args[..count]) {
                        *value = self.registers[arg];
                    }
                    self.registers[reg] = self.program.nodes[node.0 as usize].kernel.combine(
                        p,
                        footprint,
                        &values[..count],
                    );
                }
            }
        }
        self.registers[plan.output]
    }

    /// The output's scalar value at `p`.
    ///
    /// # Panics
    ///
    /// Panics if the output is not a scalar or mask.
    pub fn eval(&mut self, p: Vec2, footprint: Footprint) -> f32 {
        self.eval_value(p, footprint)
            .scalar()
            .expect("the output is a scalar or mask")
    }
}
