// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Flat evaluation plans: each shared subgraph evaluated once per point.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use glam::{Mat2, Mat3, Vec2, Vec3};

use super::block::{Block, LANES, Layout};
use super::{
    ChartSample, FieldProgram, Kernel, Node, NodeId, slice_chain, slice_point, warp_chain,
    warp_move, wave_value,
};
use crate::domain::Footprint;
use crate::field::{Affine2, Affine3};
use crate::scoped::Shape;
use crate::types::Value;

/// How a context's point and footprint follow from its parent's.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Move {
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
pub(super) enum Chain {
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
pub(super) enum Step {
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
    pub(super) steps: Vec<Step>,
    /// Registers whose values do not depend on the point, with those
    /// values: constants, and arithmetic on constants alone. An evaluator
    /// holds them from its construction; no step writes them.
    pub(super) constants: Vec<(usize, Value)>,
    registers: usize,
    pub(super) contexts: usize,
    pub(super) output: usize,
    tree_evaluations: u64,
    /// Where each register and context lives in a block of points.
    pub(super) layout: Layout,
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
    /// The shape of each register's value.
    shapes: Vec<Shape>,
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

    fn register(&mut self, shape: Shape) -> usize {
        self.shapes.push(shape);
        self.shapes.len() - 1
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
                    let reg = self.register(self.shapes[from]);
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
                    let reg = self.register(self.shapes[from]);
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
                for (k, input) in entry.op.inputs().iter().enumerate() {
                    // A wave's footprint in phase needs its phase's gradient.
                    let phase = k == 0 && matches!(entry.kernel, Kernel::Wave(..));
                    args[count] = self.compile(input, ctx, gradient || phase);
                    count += 1;
                }
                let reg = self.register(Shape::of(entry.port));
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
            let reg = self.register(self.shapes[from]);
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

/// Takes out of `steps` the evaluations whose values do not depend on the
/// point: constant leaves, with or without their (zero) gradient, and
/// pointwise kernels of such values alone without a gradient, evaluated
/// here by the same [`Kernel::combine`] a step would run, so they are the
/// same bits.
fn fold(nodes: &[Node], steps: Vec<Step>, registers: usize) -> (Vec<Step>, Vec<(usize, Value)>) {
    let mut known: Vec<Option<Value>> = alloc::vec![None; registers];
    let mut constants = Vec::new();
    let mut kept = Vec::with_capacity(steps.len());
    for step in steps {
        if let Step::Eval {
            reg,
            node,
            ref args,
            count,
            gradient,
            ..
        } = step
        {
            let kernel = &nodes[node.0 as usize].kernel;
            let mut values = [Value::Scalar(0.0); 3];
            let operands = args[..count]
                .iter()
                .zip(&mut values)
                .all(|(&arg, value)| known[arg].map(|known| *value = known).is_some());
            let value = match kernel {
                // A constant's gradient is zero, as the registers start.
                Kernel::Constant(value) => Some(Value::Scalar(*value)),
                _ if !gradient && operands && kernel.is_pointwise() => {
                    Some(kernel.combine(Vec3::ZERO, Footprint::POINT, &values[..count]))
                }
                _ => None,
            };
            if let Some(value) = value {
                known[reg] = Some(value);
                constants.push((reg, value));
                continue;
            }
        }
        kept.push(step);
    }
    (kept, constants)
}

impl Plan {
    pub(super) fn new(nodes: &[Node], output: NodeId) -> Self {
        let mut compiler = Compiler {
            nodes,
            steps: Vec::new(),
            shapes: Vec::new(),
            instances: BTreeMap::new(),
            contexts: BTreeMap::new(),
            context_count: 1,
            counts: BTreeMap::new(),
        };
        let register = compiler.compile(output, 0, false);
        let tree_evaluations = compiler.tree_count(output, 0);
        let registers = compiler.shapes.len();
        let (steps, constants) = fold(nodes, compiler.steps, registers);
        let layout = Layout::new(nodes, &steps, &compiler.shapes);
        Self {
            steps,
            constants,
            registers,
            contexts: compiler.context_count,
            output: register,
            tree_evaluations,
            layout,
        }
    }

    /// Steps evaluated per point.
    #[cfg(test)]
    pub(super) fn steps(&self) -> usize {
        self.steps.len()
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
/// What does not depend on the point (constants, and arithmetic on
/// constants alone) is computed once, when the plan is built.
/// Obtain one from [`FieldProgram::evaluator`] and keep it for a batch of
/// points, such as a raster.
///
/// [`Self::eval_value`] and [`Self::eval`] run the plan at one point. The
/// batch entry points, [`Self::eval_value_batch`] and [`Self::eval_batch`],
/// run it over blocks of points one step at a time, choosing each step's
/// work once per block rather than once per point, with each point's
/// operations unchanged; their scratch is made at the first batch and kept.
#[derive(Clone, Debug)]
pub struct Evaluator<'a> {
    program: &'a FieldProgram,
    registers: Slots<Value, INLINE_REGISTERS>,
    gradients: Slots<Vec3, INLINE_REGISTERS>,
    points: Slots<(Vec3, Footprint), INLINE_CONTEXTS>,
    /// Block scratch, made at the first batch.
    block: Option<Block>,
}

/// Registers an [`Evaluator`] holds without allocating.
const INLINE_REGISTERS: usize = 16;
/// Contexts an [`Evaluator`] holds without allocating.
const INLINE_CONTEXTS: usize = 4;

/// Fixed-size scratch storage: inline when the plan fits in `N` slots, on
/// the heap otherwise, so evaluating a small program once does not allocate.
#[derive(Clone, Debug)]
enum Slots<T, const N: usize> {
    Inline([T; N]),
    Heap(Vec<T>),
}

impl<T: Copy, const N: usize> Slots<T, N> {
    fn filled(value: T, len: usize) -> Self {
        if len <= N {
            Self::Inline([value; N])
        } else {
            Self::Heap(alloc::vec![value; len])
        }
    }
}

impl<T, const N: usize> core::ops::Deref for Slots<T, N> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match self {
            Self::Inline(slots) => slots,
            Self::Heap(slots) => slots,
        }
    }
}

impl<T, const N: usize> core::ops::DerefMut for Slots<T, N> {
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            Self::Inline(slots) => slots,
            Self::Heap(slots) => slots,
        }
    }
}

impl<'a> Evaluator<'a> {
    pub(super) fn new(program: &'a FieldProgram) -> Self {
        let mut registers = Slots::filled(Value::Scalar(0.0), program.plan.registers);
        for &(reg, value) in &program.plan.constants {
            registers[reg] = value;
        }
        Self {
            program,
            registers,
            gradients: Slots::filled(Vec3::ZERO, program.plan.registers),
            points: Slots::filled((Vec3::ZERO, Footprint::POINT), program.plan.contexts),
            block: None,
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
                    for (i, &arg) in args[..count].iter().enumerate() {
                        values[i] = self.registers[arg];
                    }
                    if gradient {
                        let mut gradients = [Vec3::ZERO; 3];
                        for (i, &arg) in args[..count].iter().enumerate() {
                            gradients[i] = self.gradients[arg];
                        }
                        (self.registers[reg], self.gradients[reg]) = self.program.pure_gradient(
                            node,
                            p,
                            footprint,
                            &values[..count],
                            &gradients[..count],
                        );
                    } else {
                        let kernel = &self.program.nodes[node.0 as usize].kernel;
                        self.registers[reg] = if let Kernel::Wave(_, _, ref ramp) = *kernel {
                            let phase = self.gradients[args[0]];
                            wave_value(ramp, footprint, &values[..count], phase)
                        } else {
                            kernel.combine(p, footprint, &values[..count])
                        };
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

    /// The output's values at `points`, all with `footprint`, into the
    /// first `points.len()` slots of `out`.
    ///
    /// Each value is the one [`Self::eval_value`] gives at its point, bit
    /// for bit. This is the entry point for a run of points, such as a
    /// raster row: keep one evaluator for every run, and the work that does
    /// not depend on the point is done once per run rather than per point.
    ///
    /// # Panics
    ///
    /// Panics if `out` is shorter than `points`.
    pub fn eval_value_batch(&mut self, points: &[Vec2], footprint: Footprint, out: &mut [Value]) {
        assert!(out.len() >= points.len(), "output shorter than the points");
        let plan = &self.program.plan;
        let block = self.block.get_or_insert_with(|| Block::new(plan));
        for (points, out) in points.chunks(LANES).zip(out.chunks_mut(LANES)) {
            block.set_points(points, footprint);
            block.run(self.program, points.len());
            for (i, value) in out[..points.len()].iter_mut().enumerate() {
                *value = block.output(plan, i);
            }
        }
    }

    /// The output's scalar values at `points`, all with `footprint`, into
    /// the first `points.len()` slots of `out`: [`Self::eval_value_batch`]
    /// for a scalar or mask output.
    ///
    /// # Panics
    ///
    /// Panics if `out` is shorter than `points`, or if the output is not a
    /// scalar or mask.
    pub fn eval_batch(&mut self, points: &[Vec2], footprint: Footprint, out: &mut [f32]) {
        assert!(out.len() >= points.len(), "output shorter than the points");
        assert!(
            self.program.nodes[self.program.output.0 as usize]
                .port
                .is_scalar(),
            "the output is a scalar or mask"
        );
        let plan = &self.program.plan;
        let block = self.block.get_or_insert_with(|| Block::new(plan));
        for (points, out) in points.chunks(LANES).zip(out.chunks_mut(LANES)) {
            block.set_points(points, footprint);
            block.run(self.program, points.len());
            out[..points.len()].copy_from_slice(&block.scalars(plan)[..points.len()]);
        }
    }

    /// The scalar output at each chart sample's point and footprint, into
    /// the first `samples.len()` slots of `out`, each the bits
    /// [`Self::eval_value_at`] gives.
    pub(super) fn eval_samples(&mut self, samples: &[ChartSample], out: &mut [f32]) {
        let plan = &self.program.plan;
        let block = self.block.get_or_insert_with(|| Block::new(plan));
        for (samples, out) in samples.chunks(LANES).zip(out.chunks_mut(LANES)) {
            block.set_samples(samples);
            block.run(self.program, samples.len());
            out[..samples.len()].copy_from_slice(&block.scalars(plan)[..samples.len()]);
        }
    }
}
