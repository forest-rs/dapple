// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Flat evaluation plans: each shared subgraph evaluated once per point.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use glam::{Mat2, Mat3, Vec2, Vec3};

use super::{FieldProgram, Kernel, Node, NodeId, slice_chain, slice_point, warp_chain, warp_move};
use crate::domain::Footprint;
use crate::field::{Affine2, Affine3};
use crate::types::Value;

/// How a context's point and footprint follow from its parent's.
#[derive(Clone, Debug, PartialEq)]
enum Move {
    Transform {
        transform: Affine2,
        stretch: f32,
    },
    Transform3 {
        transform: Affine3,
        stretch: f32,
    },
    Slice {
        origin: Vec3,
        u: Vec3,
        v: Vec3,
        stretch: f32,
    },
    /// A warp: registers of `dx` and `dy` at the parent's point, with their
    /// gradients, the rows of the displacement's Jacobian.
    Warp {
        center: [usize; 2],
        amount: f32,
    },
}

/// How a chain step turns a moved context's gradient into its parent's.
#[derive(Clone, Debug, PartialEq)]
enum Chain {
    /// Through a transform: the transposed matrix.
    Linear(Mat2),
    /// Through a solid transform: the transposed matrix.
    Linear3(Mat3),
    /// Through a slice: `(u·∇, v·∇)`.
    Slice { u: Vec3, v: Vec3 },
    /// Through a warp: `(I + a·J)ᵀ`, with the Jacobian rows in registers.
    Warp { rows: [usize; 2], amount: f32 },
}

#[derive(Clone, Debug, PartialEq)]
enum Step {
    /// Computes context `ctx` from `parent`.
    Context { ctx: usize, parent: usize, by: Move },
    /// Evaluates a pure node in `ctx` into register `reg`, with its gradient
    /// when `gradient`.
    Eval {
        reg: usize,
        node: NodeId,
        ctx: usize,
        args: [usize; 3],
        count: usize,
        gradient: bool,
    },
    /// Copies register `from`'s value into `reg`, carrying its gradient back
    /// through a transform or warp.
    Chain { reg: usize, from: usize, by: Chain },
}

/// A program flattened into a linear list of steps.
///
/// A *context* is the point and footprint a subgraph is evaluated at: the
/// output's, or one derived through transforms and warps. An *instance* is a
/// pure node in one context; each gets one register, so a subgraph shared by
/// several consumers in the same context is evaluated once. Instances whose
/// gradient a warp needs (its displacements and everything they read) carry
/// a gradient register too.
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
    /// again, as recursive evaluation does.
    pub tree_evaluations: u64,
    /// Distinct evaluation points per point: the output's, plus one per
    /// distinct transform or warp path.
    pub contexts: u64,
}

struct Compiler<'a> {
    nodes: &'a [Node],
    steps: Vec<Step>,
    registers: usize,
    /// Registers by node, context, and whether they carry a gradient.
    instances: BTreeMap<(NodeId, usize, bool), usize>,
    /// Contexts by parent and the transform or warp node that moves them.
    contexts: BTreeMap<(usize, NodeId), usize>,
    context_count: usize,
    counts: BTreeMap<(NodeId, usize), u64>,
}

impl Compiler<'_> {
    fn context(&mut self, parent: usize, node: NodeId, by: Move) -> usize {
        if let Some(&ctx) = self.contexts.get(&(parent, node)) {
            return ctx;
        }
        let ctx = self.context_count;
        self.context_count += 1;
        self.contexts.insert((parent, node), ctx);
        self.steps.push(Step::Context { ctx, parent, by });
        ctx
    }

    fn register(&mut self) -> usize {
        let reg = self.registers;
        self.registers += 1;
        reg
    }

    /// The register holding `node`'s value in `ctx`, with its gradient when
    /// `gradient`, compiling it if needed.
    fn compile(&mut self, node: NodeId, ctx: usize, gradient: bool) -> usize {
        if let Some(&reg) = self.instances.get(&(node, ctx, gradient)) {
            return reg;
        }
        let entry = &self.nodes[node.0 as usize];
        let reg = match entry.kernel {
            Kernel::Transform {
                input,
                transform,
                stretch,
            } => {
                let child = self.context(ctx, node, Move::Transform { transform, stretch });
                let from = self.compile(input, child, gradient);
                if gradient {
                    let reg = self.register();
                    self.steps.push(Step::Chain {
                        reg,
                        from,
                        by: Chain::Linear(transform.matrix.transpose()),
                    });
                    reg
                } else {
                    from
                }
            }
            Kernel::Transform3 {
                input,
                transform,
                stretch,
            } => {
                let child = self.context(ctx, node, Move::Transform3 { transform, stretch });
                self.chained(
                    input,
                    child,
                    gradient,
                    Chain::Linear3(transform.matrix.transpose()),
                )
            }
            Kernel::Slice {
                input,
                origin,
                u,
                v,
                stretch,
            } => {
                let child = self.context(
                    ctx,
                    node,
                    Move::Slice {
                        origin,
                        u,
                        v,
                        stretch,
                    },
                );
                self.chained(input, child, gradient, Chain::Slice { u, v })
            }
            Kernel::Pass(input) => self.compile(input, ctx, gradient),
            Kernel::Warp {
                input,
                dx,
                dy,
                amount,
            } => {
                // The footprint's stretch needs the displacements' gradients.
                let center = [self.compile(dx, ctx, true), self.compile(dy, ctx, true)];
                let child = self.context(ctx, node, Move::Warp { center, amount });
                let from = self.compile(input, child, gradient);
                if gradient {
                    let reg = self.register();
                    self.steps.push(Step::Chain {
                        reg,
                        from,
                        by: Chain::Warp {
                            rows: center,
                            amount,
                        },
                    });
                    reg
                } else {
                    from
                }
            }
            _ => {
                let mut args = [0; 3];
                let mut count = 0;
                for input in entry.op.inputs().iter() {
                    args[count] = self.compile(input, ctx, gradient);
                    count += 1;
                }
                let reg = self.register();
                self.steps.push(Step::Eval {
                    reg,
                    node,
                    ctx,
                    args,
                    count,
                    gradient,
                });
                reg
            }
        };
        self.instances.insert((node, ctx, gradient), reg);
        reg
    }

    /// `input` compiled in the moved context `child`, with its gradient
    /// carried back `by` a chain step when `gradient`.
    fn chained(&mut self, input: NodeId, child: usize, gradient: bool, by: Chain) -> usize {
        let from = self.compile(input, child, gradient);
        if gradient {
            let reg = self.register();
            self.steps.push(Step::Chain { reg, from, by });
            reg
        } else {
            from
        }
    }

    /// Kernel evaluations recursive evaluation performs for `node` in `ctx`.
    fn tree_count(&mut self, node: NodeId, ctx: usize) -> u64 {
        if let Some(&count) = self.counts.get(&(node, ctx)) {
            return count;
        }
        let entry = &self.nodes[node.0 as usize];
        let count = match entry.kernel {
            Kernel::Transform { input, .. }
            | Kernel::Transform3 { input, .. }
            | Kernel::Slice { input, .. } => {
                let child = self.contexts[&(ctx, node)];
                self.tree_count(input, child)
            }
            Kernel::Warp { input, dx, dy, .. } => {
                // The input where the warp moves, and the displacements here.
                let child = self.contexts[&(ctx, node)];
                self.tree_count(input, child)
                    .saturating_add(self.tree_count(dx, ctx))
                    .saturating_add(self.tree_count(dy, ctx))
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
        let register = compiler.compile(output, 0, false);
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
    gradients: Vec<Vec3>,
    points: Vec<(Vec3, Footprint)>,
}

impl<'a> Evaluator<'a> {
    pub(super) fn new(program: &'a FieldProgram) -> Self {
        Self {
            program,
            registers: alloc::vec![Value::Scalar(0.0); program.plan.registers],
            gradients: alloc::vec![Vec3::ZERO; program.plan.registers],
            points: alloc::vec![(Vec3::ZERO, Footprint::POINT); program.plan.contexts],
        }
    }

    /// The output's value at `p`.
    pub fn eval_value(&mut self, p: Vec2, footprint: Footprint) -> Value {
        self.eval_value_at(p.extend(0.0), footprint)
    }

    /// The output's value at a point of its space; a planar output ignores `z`.
    pub(crate) fn eval_value_at(&mut self, p: Vec3, footprint: Footprint) -> Value {
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
                        Move::Transform { transform, stretch } => (
                            transform.apply(p.truncate()).extend(p.z),
                            footprint.scaled(stretch),
                        ),
                        Move::Transform3 { transform, stretch } => {
                            (transform.apply(p), footprint.scaled(stretch))
                        }
                        Move::Slice {
                            origin,
                            u,
                            v,
                            stretch,
                        } => (slice_point(origin, u, v, p), footprint.scaled(stretch)),
                        Move::Warp { center, amount } => warp_move(
                            p,
                            footprint,
                            amount,
                            center.map(|reg| self.registers[reg]),
                            center.map(|reg| self.gradients[reg]),
                        ),
                    };
                }
                Step::Eval {
                    reg,
                    node,
                    ctx,
                    args,
                    count,
                    gradient,
                } => {
                    let (p, footprint) = self.points[ctx];
                    let mut values = [Value::Scalar(0.0); 3];
                    let mut gradients = [Vec3::ZERO; 3];
                    for (i, &arg) in args[..count].iter().enumerate() {
                        values[i] = self.registers[arg];
                        gradients[i] = self.gradients[arg];
                    }
                    let kernel = &self.program.nodes[node.0 as usize].kernel;
                    let value = kernel.combine(p, footprint, &values[..count]);
                    self.registers[reg] = value;
                    if gradient {
                        self.gradients[reg] = kernel
                            .gradient(p, footprint, &values[..count], &gradients[..count])
                            .unwrap_or_else(|| {
                                self.program.numeric_gradient(node, value, p, footprint)
                            });
                    }
                }
                Step::Chain { reg, from, ref by } => {
                    self.registers[reg] = self.registers[from];
                    let g = self.gradients[from];
                    self.gradients[reg] = match *by {
                        Chain::Linear(transpose) => (transpose * g.truncate()).extend(0.0),
                        Chain::Linear3(transpose) => transpose * g,
                        Chain::Slice { u, v } => slice_chain(g, u, v),
                        Chain::Warp { rows, amount } => {
                            warp_chain(g, rows.map(|r| self.gradients[r]), amount)
                        }
                    };
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
