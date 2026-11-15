// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Block evaluation: a flat plan run one step at a time over many points.
//!
//! [`Evaluator`](super::Evaluator)'s batch entry points split a run of
//! points into blocks of up to [`LANES`] and run the plan over each block
//! step by step, so the step and kernel are chosen once per step per block
//! rather than once per step per point.
//!
//! A block keeps every register as columns, one per component, of
//! [`LANES`] values (structure of arrays): a register's [`Shape`], decided
//! when the plan is built, says how many and of which kind, so every step
//! runs typed loops over its lanes without inspecting a [`Value`]. Folded
//! constants are broadcast into their columns once, when the block is made,
//! and no step writes them.
//!
//! Each context holds its point as three columns and its footprint either
//! as one value for the whole block (the row's, and every transform or
//! slice of it) or, once a warp has stretched it differently at each point,
//! as a column of its own.
//!
//! Every step computes, per lane, the operations the point evaluation
//! computes, in the same order, so each lane is the bits
//! [`Evaluator::eval_value`](super::Evaluator::eval_value) gives at its
//! point:
//!
//! - context steps (transforms, slices, warps) and chain steps are loops
//!   over the context's and register's columns;
//! - pointwise kernels are loops over operand columns, with a scalar
//!   operand read for every component of a vector;
//! - leaf kernels (noise, fractal, cellular, tiling, scatter, disk, image
//!   sample, and the solid ones) call their scalar kernel per lane in a
//!   loop of their own; with their gradient, the fused value-and-gradient
//!   evaluation; a fractal whose block shares one footprint goes octave by
//!   octave over the lanes, taking each octave's band weight once;
//! - lattice noise, planar and solid, alone and in fractals, runs a lane
//!   kernel when its block shares one footprint: the block's points are
//!   located on the lattice together, and value noise's corners blended
//!   together;
//! - the gradients of arithmetic on scalars are loops over the operands'
//!   gradient columns.
//!
//! Gradient instances of the other kernels (image samples, scatters,
//! vectors, components and the like, whose gradients are taken by central
//! differences of the recursive walk) are evaluated lane by lane through
//! [`FieldProgram::pure_gradient`], still chosen once per block.

use alloc::vec;
use alloc::vec::Vec;

use glam::{Vec2, Vec3};

use super::flat::{Chain, Move, Plan, Step};
use super::{
    BinaryOp, ChartSample, FieldProgram, Kernel, NodeId, abs_gradient, angle, atan2_gradient,
    binary, binary_gradient, clamp_gradient, direction, fract, identifier, mix, mix_gradient,
    remap, slice_chain, slice_point, warp_chain, warp_move,
};
use crate::domain::Footprint;
use crate::field::ScalarField;
use crate::scoped::Shape;
use crate::solid::SolidField;
use crate::types::Value;

/// Points per block.
///
/// 64 lanes keep a block's columns of the library's largest programs (the
/// oak's wood, of about 60 registers) within a few tens of kilobytes, and
/// make the per-step choice negligible beside a step's work. Blocks of 32
/// to 256 lanes measured within 1% of each other on the library's
/// programs; 64 was the cheapest for a program whose work is all setup.
pub(super) const LANES: usize = 64;

/// One component of a register, or one coordinate of a context's point,
/// over a block.
type Column = [f32; LANES];

/// Where a register's columns are in a block.
#[derive(Copy, Clone, Debug, PartialEq)]
struct Slot {
    shape: Shape,
    /// The register's first `f32` column: its value's components (none for
    /// an identifier), then, with a gradient, the gradient's three.
    at: usize,
    /// An identifier register's column among the identifier columns.
    id: usize,
}

impl Slot {
    /// `f32` columns of the register's value.
    const fn components(self) -> usize {
        match self.shape {
            Shape::Scalar => 1,
            Shape::Id => 0,
            Shape::Vector2 => 2,
            Shape::Vector3 => 3,
        }
    }

    /// The column of value component `c`; a scalar's one column serves
    /// every component, as a scalar operand broadcasts over a vector.
    const fn component(self, c: usize) -> usize {
        match self.shape {
            Shape::Scalar => self.at,
            _ => self.at + c,
        }
    }

    /// The column of gradient component `c`.
    const fn gradient(self, c: usize) -> usize {
        self.at + self.components() + c
    }
}

/// Where each register of a plan lives in a block.
///
/// Columns follow register order, and a step's operands are always
/// registers made before its result, so everything a step reads lies below
/// the columns it writes.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Layout {
    slots: Vec<Slot>,
    /// `f32` columns of all registers.
    columns: usize,
    /// Identifier columns.
    ids: usize,
}

impl Layout {
    /// The layout of a plan's `steps`, whose registers have `shapes`.
    pub(super) fn new(steps: &[Step], shapes: &[Shape]) -> Self {
        // Registers whose gradient a step reads or writes.
        let mut gradient = vec![false; shapes.len()];
        for step in steps {
            match *step {
                Step::Context {
                    by: Move::Warp { center, .. },
                    ..
                } => center.iter().for_each(|&reg| gradient[reg] = true),
                Step::Eval {
                    reg,
                    args,
                    count,
                    gradient: true,
                    ..
                } => {
                    debug_assert!(args[..count].iter().all(|&arg| arg < reg), "{step:?}");
                    gradient[reg] = true;
                    args[..count].iter().for_each(|&arg| gradient[arg] = true);
                }
                Step::Chain { reg, from, ref by } => {
                    debug_assert!(from < reg, "{step:?}");
                    gradient[reg] = true;
                    gradient[from] = true;
                    if let Chain::Warp { rows, .. } = *by {
                        debug_assert!(rows.iter().all(|&row| row < reg), "{step:?}");
                        rows.iter().for_each(|&row| gradient[row] = true);
                    }
                }
                Step::Eval {
                    reg, args, count, ..
                } => debug_assert!(args[..count].iter().all(|&arg| arg < reg), "{step:?}"),
                Step::Context { .. } => {}
            }
        }
        let (mut columns, mut ids) = (0, 0);
        let slots = shapes
            .iter()
            .zip(gradient)
            .map(|(&shape, gradient)| {
                let slot = Slot {
                    shape,
                    at: columns,
                    id: ids,
                };
                columns += slot.components() + if gradient { 3 } else { 0 };
                ids += usize::from(shape == Shape::Id);
                slot
            })
            .collect();
        Self {
            slots,
            columns,
            ids,
        }
    }
}

/// The footprints of a context over a block.
#[derive(Copy, Clone)]
enum Footprints<'a> {
    /// The same footprint at every lane.
    Uniform(Footprint),
    /// One footprint per lane.
    Lanes(&'a [Footprint]),
}

/// A context's points and footprints over the first lanes of a block.
#[derive(Copy, Clone)]
struct Points<'a> {
    x: &'a [f32],
    y: &'a [f32],
    z: &'a [f32],
    footprints: Footprints<'a>,
}

impl Points<'_> {
    /// Calls `f` with each lane's index, planar point and footprint.
    #[inline(always)]
    fn planar(self, mut f: impl FnMut(usize, Vec2, Footprint)) {
        let points = self.x.iter().zip(self.y).map(|(&x, &y)| Vec2::new(x, y));
        match self.footprints {
            Footprints::Uniform(footprint) => {
                points.enumerate().for_each(|(i, p)| f(i, p, footprint));
            }
            Footprints::Lanes(footprints) => points
                .zip(footprints)
                .enumerate()
                .for_each(|(i, (p, &footprint))| f(i, p, footprint)),
        }
    }

    /// Calls `f` with each lane's index, point and footprint.
    #[inline(always)]
    fn solid(self, mut f: impl FnMut(usize, Vec3, Footprint)) {
        let points = self
            .x
            .iter()
            .zip(self.y)
            .zip(self.z)
            .map(|((&x, &y), &z)| Vec3::new(x, y, z));
        match self.footprints {
            Footprints::Uniform(footprint) => {
                points.enumerate().for_each(|(i, p)| f(i, p, footprint));
            }
            Footprints::Lanes(footprints) => points
                .zip(footprints)
                .enumerate()
                .for_each(|(i, (p, &footprint))| f(i, p, footprint)),
        }
    }
}

/// The registers a step reads: every column below its result's.
#[derive(Copy, Clone)]
struct Operands<'a> {
    columns: &'a [Column],
    layout: &'a Layout,
    args: &'a [usize],
    len: usize,
}

impl<'a> Operands<'a> {
    fn slot(self, k: usize) -> Slot {
        self.layout.slots[self.args[k]]
    }

    /// Value component `c` of operand `k`, a scalar's for every `c`.
    fn component(self, k: usize, c: usize) -> &'a [f32] {
        &self.column(k, c)[..self.len]
    }

    /// The whole column of [`Self::component`].
    fn column(self, k: usize, c: usize) -> &'a Column {
        &self.columns[self.slot(k).component(c)]
    }

    /// Operand `k`'s gradient columns.
    fn gradient(self, k: usize) -> [&'a Column; 3] {
        let slot = self.slot(k);
        [0, 1, 2].map(|c| &self.columns[slot.gradient(c)])
    }

    /// Whether every operand is a scalar.
    fn scalars(self) -> bool {
        self.args
            .iter()
            .all(|&arg| self.layout.slots[arg].shape == Shape::Scalar)
    }

    /// Operand `k`'s value at lane `i`.
    fn value(self, k: usize, i: usize) -> Value {
        let slot = self.slot(k);
        let c = |c: usize| self.columns[slot.at + c][i];
        match slot.shape {
            Shape::Scalar => Value::Scalar(c(0)),
            Shape::Vector2 => Value::Vector2(Vec2::new(c(0), c(1))),
            Shape::Vector3 => Value::Vector3(Vec3::new(c(0), c(1), c(2))),
            Shape::Id => unreachable!("identifiers are never operands"),
        }
    }
}

/// The columns a step writes: its register's value, then its gradient.
struct Out<'a> {
    columns: &'a mut [Column],
    id: &'a mut [u32],
    len: usize,
}

impl Out<'_> {
    /// Value component `c`'s column.
    fn component(&mut self, c: usize) -> &mut [f32] {
        &mut self.columns[c][..self.len]
    }

    /// Stores `value` at lane `i`.
    fn store(&mut self, i: usize, value: Value) {
        match value {
            Value::Scalar(v) => self.columns[0][i] = v,
            Value::Id(v) => self.id[i] = v,
            Value::Vector2(v) => {
                self.columns[0][i] = v.x;
                self.columns[1][i] = v.y;
            }
            Value::Vector3(v) => {
                self.columns[0][i] = v.x;
                self.columns[1][i] = v.y;
                self.columns[2][i] = v.z;
            }
        }
    }

    /// Stores gradient `g` at lane `i`, after `components` value columns.
    fn store_gradient(&mut self, components: usize, i: usize, g: Vec3) {
        self.columns[components][i] = g.x;
        self.columns[components + 1][i] = g.y;
        self.columns[components + 2][i] = g.z;
    }

    /// A scalar's value column and gradient columns.
    fn scalar_gradient(&mut self) -> (&mut Column, [&mut Column; 3]) {
        let [v, gx, gy, gz, ..] = &mut self.columns[..] else {
            unreachable!("a scalar with its gradient has four columns");
        };
        (v, [gx, gy, gz])
    }
}

/// The lanes `0..len` of a block, provably within its columns.
fn lanes(len: usize) -> core::ops::Range<usize> {
    0..len.min(LANES)
}

/// Lane `i` of three columns, as a vector.
#[inline(always)]
fn vec3(columns: [&Column; 3], i: usize) -> Vec3 {
    Vec3::new(columns[0][i], columns[1][i], columns[2][i])
}

/// Stores `v` at lane `i` of three columns.
#[inline(always)]
fn store3(columns: &mut [&mut Column; 3], i: usize, v: Vec3) {
    (columns[0][i], columns[1][i], columns[2][i]) = v.into();
}

/// `out[i] = f(a[i])` over the lanes.
#[inline(always)]
fn map1(out: &mut [f32], a: &[f32], f: impl Fn(f32) -> f32) {
    for (out, &a) in out.iter_mut().zip(a) {
        *out = f(a);
    }
}

/// `out[i] = f(a[i], b[i])` over the lanes.
#[inline(always)]
fn map2(out: &mut [f32], a: &[f32], b: &[f32], f: impl Fn(f32, f32) -> f32) {
    for ((out, &a), &b) in out.iter_mut().zip(a).zip(b) {
        *out = f(a, b);
    }
}

/// `out[i] = f(a[i], b[i], c[i])` over the lanes.
#[inline(always)]
fn map3(out: &mut [f32], a: &[f32], b: &[f32], c: &[f32], f: impl Fn(f32, f32, f32) -> f32) {
    for (((out, &a), &b), &c) in out.iter_mut().zip(a).zip(b).zip(c) {
        *out = f(a, b, c);
    }
}

/// A block's scratch: register columns and context points, sized for one
/// plan when made and reused for every block.
#[derive(Clone, Debug)]
pub(super) struct Block {
    columns: Vec<Column>,
    ids: Vec<[u32; LANES]>,
    /// Per context: its point's x, y and z columns.
    points: Vec<Column>,
    /// Per context: its footprint at each lane, when not `uniform`.
    footprints: Vec<[Footprint; LANES]>,
    /// Per context: the footprint of every lane, when they share one.
    uniform: Vec<Option<Footprint>>,
}

impl Block {
    /// Scratch for `plan`, with its constants broadcast.
    pub(super) fn new(plan: &Plan) -> Self {
        let layout = &plan.layout;
        let mut block = Self {
            columns: vec![[0.0; LANES]; layout.columns],
            ids: vec![[0; LANES]; layout.ids],
            points: vec![[0.0; LANES]; 3 * plan.contexts],
            footprints: vec![[Footprint::POINT; LANES]; plan.contexts],
            uniform: vec![Some(Footprint::POINT); plan.contexts],
        };
        for &(reg, value) in &plan.constants {
            let slot = layout.slots[reg];
            let mut fill = |c: usize, v: f32| block.columns[slot.at + c] = [v; LANES];
            match value {
                Value::Scalar(v) => fill(0, v),
                Value::Vector2(v) => v
                    .to_array()
                    .into_iter()
                    .enumerate()
                    .for_each(|(c, v)| fill(c, v)),
                Value::Vector3(v) => v
                    .to_array()
                    .into_iter()
                    .enumerate()
                    .for_each(|(c, v)| fill(c, v)),
                Value::Id(v) => block.ids[slot.id] = [v; LANES],
            }
        }
        block
    }

    /// Sets the output's context to `points`, at most [`LANES`], all with
    /// `footprint`.
    pub(super) fn set_points(&mut self, points: &[Vec2], footprint: Footprint) {
        let [x, y, z, ..] = &mut self.points[..] else {
            unreachable!("every plan has the output's context");
        };
        for (((x, y), z), p) in x.iter_mut().zip(y).zip(z).zip(points) {
            (*x, *y, *z) = (p.x, p.y, 0.0);
        }
        self.uniform[0] = Some(footprint);
    }

    /// Sets the output's context to the chart samples' points and
    /// footprints, at most [`LANES`].
    pub(super) fn set_samples(&mut self, samples: &[ChartSample]) {
        let [x, y, z, ..] = &mut self.points[..] else {
            unreachable!("every plan has the output's context");
        };
        let footprints = &mut self.footprints[0];
        for ((((x, y), z), footprint), sample) in x
            .iter_mut()
            .zip(y)
            .zip(z)
            .zip(footprints.iter_mut())
            .zip(samples)
        {
            (*x, *y, *z) = sample.position.into();
            *footprint = sample.footprint;
        }
        let first = samples.first().map(|sample| sample.footprint);
        let shared = samples.iter().all(|sample| {
            Some(sample.footprint.width().to_bits()) == first.map(|f| f.width().to_bits())
        });
        self.uniform[0] = if shared { first } else { None };
    }

    /// Runs `program`'s plan over the first `len` lanes, whose points are
    /// set.
    pub(super) fn run(&mut self, program: &FieldProgram, len: usize) {
        let plan = &program.plan;
        for step in &plan.steps {
            match *step {
                Step::Context {
                    ctx,
                    parent,
                    ref by,
                } => self.context(&plan.layout, ctx, parent, by, len),
                Step::Eval {
                    reg,
                    node,
                    ctx,
                    ref args,
                    count,
                    gradient,
                } => {
                    let layout = &plan.layout;
                    let slot = layout.slots[reg];
                    let (below, above) = self.columns.split_at_mut(slot.at);
                    let operands = Operands {
                        columns: below,
                        layout,
                        args: &args[..count],
                        len,
                    };
                    let mut out = Out {
                        columns: above,
                        id: if slot.shape == Shape::Id {
                            &mut self.ids[slot.id][..len]
                        } else {
                            &mut []
                        },
                        len,
                    };
                    let at = Points {
                        x: &self.points[3 * ctx][..len],
                        y: &self.points[3 * ctx + 1][..len],
                        z: &self.points[3 * ctx + 2][..len],
                        footprints: match self.uniform[ctx] {
                            Some(footprint) => Footprints::Uniform(footprint),
                            None => Footprints::Lanes(&self.footprints[ctx][..len]),
                        },
                    };
                    let kernel = &program.nodes[node.0 as usize].kernel;
                    if gradient {
                        eval_gradient(program, node, kernel, slot, at, operands, &mut out);
                    } else {
                        eval_value(kernel, slot, at, operands, &mut out);
                    }
                }
                Step::Chain { reg, from, ref by } => self.chain(&plan.layout, reg, from, by, len),
            }
        }
    }

    /// The output's value at lane `i`.
    pub(super) fn output(&self, plan: &Plan, i: usize) -> Value {
        let slot = plan.layout.slots[plan.output];
        let c = |c: usize| self.columns[slot.at + c][i];
        match slot.shape {
            Shape::Scalar => Value::Scalar(c(0)),
            Shape::Id => Value::Id(self.ids[slot.id][i]),
            Shape::Vector2 => Value::Vector2(Vec2::new(c(0), c(1))),
            Shape::Vector3 => Value::Vector3(Vec3::new(c(0), c(1), c(2))),
        }
    }

    /// The scalar output's column.
    pub(super) fn scalars(&self, plan: &Plan) -> &Column {
        let slot = plan.layout.slots[plan.output];
        debug_assert_eq!(slot.shape, Shape::Scalar, "the output is a scalar");
        &self.columns[slot.at]
    }

    /// Computes context `ctx`'s points and footprints from `parent`'s.
    fn context(&mut self, layout: &Layout, ctx: usize, parent: usize, by: &Move, len: usize) {
        let (below, above) = self.points.split_at_mut(3 * ctx);
        let p = [0, 1, 2].map(|c| &below[3 * parent + c]);
        let [x, y, z, ..] = above else {
            unreachable!("every context has three columns");
        };
        let mut q = [x, y, z];
        let stretch = match *by {
            Move::Transform { transform, stretch } => {
                for i in lanes(len) {
                    let moved = transform.apply(Vec2::new(p[0][i], p[1][i]));
                    store3(&mut q, i, moved.extend(p[2][i]));
                }
                stretch
            }
            Move::Transform3 { transform, stretch } => {
                for i in lanes(len) {
                    store3(&mut q, i, transform.apply(vec3(p, i)));
                }
                stretch
            }
            Move::Slice {
                origin,
                u,
                v,
                stretch,
            } => {
                for i in lanes(len) {
                    store3(&mut q, i, slice_point(origin, u, v, vec3(p, i)));
                }
                stretch
            }
            Move::Warp { center, amount } => {
                let [dx, dy] = center.map(|reg| layout.slots[reg]);
                let (cx, cy) = (&self.columns[dx.at], &self.columns[dy.at]);
                let rows = [dx, dy].map(|slot| [0, 1, 2].map(|c| &self.columns[slot.gradient(c)]));
                let (before, after) = self.footprints.split_at_mut(ctx);
                let (from, footprints) = (&before[parent], &mut after[0]);
                let parent_uniform = self.uniform[parent];
                for i in lanes(len) {
                    let (moved, footprint) = warp_move(
                        vec3(p, i),
                        parent_uniform.unwrap_or(from[i]),
                        amount,
                        [Value::Scalar(cx[i]), Value::Scalar(cy[i])],
                        [vec3(rows[0], i), vec3(rows[1], i)],
                    );
                    store3(&mut q, i, moved);
                    footprints[i] = footprint;
                }
                // A point footprint stays a point and needs no derivatives.
                self.uniform[ctx] = parent_uniform.filter(|footprint| footprint.width() <= 0.0);
                return;
            }
        };
        self.uniform[ctx] = self.uniform[parent].map(|footprint| footprint.scaled(stretch));
        if self.uniform[ctx].is_none() {
            let (before, after) = self.footprints.split_at_mut(ctx);
            for (out, footprint) in after[0][..len].iter_mut().zip(&before[parent][..len]) {
                *out = footprint.scaled(stretch);
            }
        }
    }

    /// Copies register `from` into `reg`, carrying its gradient back `by`.
    fn chain(&mut self, layout: &Layout, reg: usize, from: usize, by: &Chain, len: usize) {
        let (slot, source) = (layout.slots[reg], layout.slots[from]);
        let (below, above) = self.columns.split_at_mut(slot.at);
        for c in 0..slot.components() {
            above[c][..len].copy_from_slice(&below[source.at + c][..len]);
        }
        if slot.shape == Shape::Id {
            let (below, above) = self.ids.split_at_mut(slot.id);
            above[0][..len].copy_from_slice(&below[source.id][..len]);
        }
        let gradient = |slot: Slot| [0, 1, 2].map(|c| &below[slot.gradient(c)]);
        let from = gradient(source);
        let [gx, gy, gz, ..] = &mut above[slot.components()..] else {
            unreachable!("a chained register has a gradient");
        };
        let out = [gx, gy, gz];
        match *by {
            Chain::Linear(transpose) => {
                chain_lanes(out, from, len, |g, _| {
                    (transpose * g.truncate()).extend(0.0)
                });
            }
            Chain::Linear3(transpose) => chain_lanes(out, from, len, |g, _| transpose * g),
            Chain::Slice { u, v } => chain_lanes(out, from, len, |g, _| slice_chain(g, u, v)),
            Chain::Warp { rows, amount } => {
                let rows = rows.map(|reg| gradient(layout.slots[reg]));
                chain_lanes(out, from, len, |g, i| {
                    warp_chain(g, [vec3(rows[0], i), vec3(rows[1], i)], amount)
                });
            }
        }
    }
}

/// Gradient columns `out = f(from, i)` over the first `len` lanes.
#[inline(always)]
fn chain_lanes(
    mut out: [&mut Column; 3],
    from: [&Column; 3],
    len: usize,
    f: impl Fn(Vec3, usize) -> Vec3,
) {
    for i in lanes(len) {
        store3(&mut out, i, f(vec3(from, i), i));
    }
}

/// `out[i] = f(p, footprint)` at each lane's planar point.
#[inline(always)]
fn planar(out: &mut Out<'_>, at: Points<'_>, f: impl Fn(Vec2, Footprint) -> f32) {
    let out = out.component(0);
    at.planar(|i, p, footprint| out[i] = f(p, footprint));
}

/// `out[i] = f(p, footprint)` at each lane's point.
#[inline(always)]
fn solid(out: &mut Out<'_>, at: Points<'_>, f: impl Fn(Vec3, Footprint) -> f32) {
    let out = out.component(0);
    at.solid(|i, p, footprint| out[i] = f(p, footprint));
}

/// A binary operation per component, a scalar operand broadcast over a
/// vector's components.
fn binary_columns(op: BinaryOp, slot: Slot, operands: Operands<'_>, out: &mut Out<'_>) {
    for c in 0..slot.components() {
        let (out, a, b) = (
            out.component(c),
            operands.component(0, c),
            operands.component(1, c),
        );
        match op {
            BinaryOp::Add => map2(out, a, b, |a, b| binary(BinaryOp::Add, a, b)),
            BinaryOp::Sub => map2(out, a, b, |a, b| binary(BinaryOp::Sub, a, b)),
            BinaryOp::Mul => map2(out, a, b, |a, b| binary(BinaryOp::Mul, a, b)),
            BinaryOp::Min => map2(out, a, b, |a, b| binary(BinaryOp::Min, a, b)),
            BinaryOp::Max => map2(out, a, b, |a, b| binary(BinaryOp::Max, a, b)),
        }
    }
}

/// Evaluates `kernel`, whose result is `slot`, over the lanes.
fn eval_value(
    kernel: &Kernel,
    slot: Slot,
    at: Points<'_>,
    operands: Operands<'_>,
    out: &mut Out<'_>,
) {
    let scalar = |k: usize| operands.component(k, 0);
    match *kernel {
        Kernel::Noise(ref noise) => match at.footprints {
            Footprints::Uniform(footprint) => {
                noise.eval_lanes([at.x, at.y], footprint, out.component(0));
            }
            Footprints::Lanes(_) => planar(out, at, |p, fp| noise.eval(p, fp)),
        },
        // A fractal of one footprint goes octave by octave over the lanes.
        Kernel::Fractal(ref fractal) => match at.footprints {
            Footprints::Uniform(footprint) => {
                fractal.eval_lanes(at.x, at.y, footprint, out.component(0));
            }
            Footprints::Lanes(_) => planar(out, at, |p, fp| fractal.eval(p, fp)),
        },
        Kernel::Cellular(ref cellular) => planar(out, at, |p, fp| cellular.eval(p, fp)),
        Kernel::Tiling(ref tiling) => planar(out, at, |p, fp| tiling.eval(p, fp)),
        Kernel::Scatter(ref scatter) => planar(out, at, |p, fp| scatter.eval(p, fp)),
        Kernel::Disk(ref disk) => planar(out, at, |p, fp| disk.eval(p, fp)),
        // An image's value is stored by its own shape.
        Kernel::Sample(ref image) => {
            at.planar(|i, p, footprint| out.store(i, image.sample_value(p, footprint)));
        }
        Kernel::Noise3(ref noise) => match at.footprints {
            Footprints::Uniform(footprint) => {
                noise.eval_lanes([at.x, at.y, at.z], footprint, out.component(0));
            }
            Footprints::Lanes(_) => solid(out, at, |p, fp| noise.eval(p, fp)),
        },
        Kernel::Fractal3(ref fractal) => match at.footprints {
            Footprints::Uniform(footprint) => {
                fractal.eval_lanes([at.x, at.y, at.z], footprint, out.component(0));
            }
            Footprints::Lanes(_) => solid(out, at, |p, fp| fractal.eval(p, fp)),
        },
        Kernel::Cellular3(ref cellular) => solid(out, at, |p, fp| cellular.eval(p, fp)),
        Kernel::Position3 => {
            out.component(0).copy_from_slice(at.x);
            out.component(1).copy_from_slice(at.y);
            out.component(2).copy_from_slice(at.z);
        }
        Kernel::Binary(op, ..) => binary_columns(op, slot, operands, out),
        Kernel::Abs(_) => map1(out.component(0), scalar(0), f32::abs),
        Kernel::Clamp(_, min, max) => map1(out.component(0), scalar(0), |a| a.clamp(min, max)),
        Kernel::AsMask(_) => map1(out.component(0), scalar(0), |a| a.clamp(0.0, 1.0)),
        Kernel::Remap {
            scale, from, to, ..
        } => map1(out.component(0), scalar(0), |a| remap(a, scale, from, to)),
        Kernel::Fract(_) => map1(out.component(0), scalar(0), fract),
        Kernel::Atan2(..) => map2(out.component(0), scalar(0), scalar(1), libm::atan2f),
        Kernel::Mix(..) => {
            for c in 0..slot.components() {
                let (a, b) = (operands.component(0, c), operands.component(1, c));
                map3(out.component(c), a, b, scalar(2), mix);
            }
        }
        Kernel::Vector2(..) | Kernel::Vector3(..) => {
            for c in 0..slot.components() {
                out.component(c).copy_from_slice(scalar(c));
            }
        }
        Kernel::Component(_, index) => {
            out.component(0)
                .copy_from_slice(operands.component(0, index));
        }
        Kernel::Length(_) => {
            let [x, y] = [0, 1].map(|c| operands.component(0, c));
            if operands.slot(0).shape == Shape::Vector2 {
                map2(out.component(0), x, y, |x, y| Vec2::new(x, y).length());
            } else {
                let z = operands.component(0, 2);
                map3(out.component(0), x, y, z, |x, y, z| {
                    Vec3::new(x, y, z).length()
                });
            }
        }
        Kernel::Coherence(_) => {
            let [x, y] = [0, 1].map(|c| operands.component(0, c));
            map2(out.component(0), x, y, |x, y| {
                Vec2::new(x, y).length().min(1.0)
            });
        }
        Kernel::Angle(_) => {
            let [x, y] = [0, 1].map(|c| operands.component(0, c));
            map2(out.component(0), x, y, |x, y| angle(Vec2::new(x, y)));
        }
        Kernel::Direction(_) => {
            for (i, &a) in scalar(0).iter().enumerate() {
                out.store(i, Value::Vector2(direction(a)));
            }
        }
        Kernel::Normalize(_) => {
            let [x, y, z] = [0, 1, 2].map(|c| operands.component(0, c));
            for i in 0..x.len() {
                let v = Vec3::new(x[i], y[i], z[i]);
                out.store(i, Value::Vector3(v.try_normalize().unwrap_or(Vec3::Z)));
            }
        }
        Kernel::ToId(_, levels) => {
            for (out, &v) in out.id.iter_mut().zip(scalar(0)) {
                *out = identifier(v, levels);
            }
        }
        Kernel::Constant(_)
        | Kernel::Transform { .. }
        | Kernel::Transform3 { .. }
        | Kernel::Slice { .. }
        | Kernel::Pass(_)
        | Kernel::Warp { .. } => {
            unreachable!("constants are folded, and moves are context steps")
        }
    }
}

/// `(value, gradient) = f(p, footprint)` at each lane's planar point; the
/// gradient's `z` is zero.
#[inline(always)]
fn planar_gradient(out: &mut Out<'_>, at: Points<'_>, f: impl Fn(Vec2, Footprint) -> (f32, Vec2)) {
    let (v, mut g) = out.scalar_gradient();
    at.planar(|i, p, footprint| {
        let (value, gradient) = f(p, footprint);
        v[i] = value;
        store3(&mut g, i, gradient.extend(0.0));
    });
}

/// `(value, gradient) = f(p, footprint)` at each lane's point.
#[inline(always)]
fn solid_gradient(out: &mut Out<'_>, at: Points<'_>, f: impl Fn(Vec3, Footprint) -> (f32, Vec3)) {
    let (v, mut g) = out.scalar_gradient();
    at.solid(|i, p, footprint| {
        let (value, gradient) = f(p, footprint);
        v[i] = value;
        store3(&mut g, i, gradient);
    });
}

/// The gradient `f(a, ga)` of a function of scalar operand `a`.
#[inline(always)]
fn gradient1(out: &mut Out<'_>, operands: Operands<'_>, f: impl Fn(f32, Vec3) -> Vec3) {
    let (a, ga) = (operands.column(0, 0), operands.gradient(0));
    let (_, mut g) = out.scalar_gradient();
    for i in lanes(operands.len) {
        store3(&mut g, i, f(a[i], vec3(ga, i)));
    }
}

/// The gradient `f([a, b], [ga, gb])` of a function of scalar operands
/// `a` and `b`.
#[inline(always)]
fn gradient2(out: &mut Out<'_>, operands: Operands<'_>, f: impl Fn([f32; 2], [Vec3; 2]) -> Vec3) {
    let (a, ga) = (operands.column(0, 0), operands.gradient(0));
    let (b, gb) = (operands.column(1, 0), operands.gradient(1));
    let (_, mut g) = out.scalar_gradient();
    for i in lanes(operands.len) {
        store3(&mut g, i, f([a[i], b[i]], [vec3(ga, i), vec3(gb, i)]));
    }
}

/// The gradient `f([a, b, t], [ga, gb, gt])` of a function of scalar
/// operands `a`, `b` and `t`.
#[inline(always)]
fn gradient3(out: &mut Out<'_>, operands: Operands<'_>, f: impl Fn([f32; 3], [Vec3; 3]) -> Vec3) {
    let [a, b, t] = [0, 1, 2].map(|k| operands.column(k, 0));
    let [ga, gb, gt] = [0, 1, 2].map(|k| operands.gradient(k));
    let (_, mut g) = out.scalar_gradient();
    for i in lanes(operands.len) {
        let gradient = f([a[i], b[i], t[i]], [vec3(ga, i), vec3(gb, i), vec3(gt, i)]);
        store3(&mut g, i, gradient);
    }
}

/// Evaluates pure node `node`, whose result is `slot`, with its gradient
/// over the lanes.
fn eval_gradient(
    program: &FieldProgram,
    node: NodeId,
    kernel: &Kernel,
    slot: Slot,
    at: Points<'_>,
    operands: Operands<'_>,
    out: &mut Out<'_>,
) {
    // Leaves evaluate their value and gradient at once, as
    // `Kernel::value_and_gradient` does.
    match *kernel {
        Kernel::Noise(ref noise) => {
            return planar_gradient(out, at, |p, fp| noise.eval_gradient(p, fp));
        }
        Kernel::Fractal(ref fractal) => {
            return planar_gradient(out, at, |p, fp| fractal.eval_gradient(p, fp));
        }
        Kernel::Cellular(ref cellular) => {
            return planar_gradient(out, at, |p, fp| cellular.eval_gradient(p, fp));
        }
        Kernel::Tiling(ref tiling) => {
            return planar_gradient(out, at, |p, fp| tiling.eval_gradient(p, fp));
        }
        Kernel::Disk(ref disk) => {
            return planar_gradient(out, at, |p, fp| disk.eval_gradient(p, fp));
        }
        Kernel::Noise3(ref noise) => {
            return solid_gradient(out, at, |p, fp| noise.eval_gradient(p, fp));
        }
        Kernel::Fractal3(ref fractal) => {
            return solid_gradient(out, at, |p, fp| fractal.eval_gradient(p, fp));
        }
        Kernel::Cellular3(ref cellular) => {
            return solid_gradient(out, at, |p, fp| cellular.eval_gradient(p, fp));
        }
        _ => {}
    }
    // Arithmetic on scalars: the value as without the gradient, then the
    // gradient's closed form, as `Kernel::gradient` gives it.
    let arithmetic = matches!(
        kernel,
        Kernel::Binary(..)
            | Kernel::Atan2(..)
            | Kernel::Mix(..)
            | Kernel::Abs(_)
            | Kernel::Clamp(..)
            | Kernel::AsMask(_)
            | Kernel::Remap { .. }
            | Kernel::Fract(_)
    );
    if arithmetic && slot.shape == Shape::Scalar && operands.scalars() {
        eval_value(kernel, slot, at, operands, out);
        match *kernel {
            Kernel::Binary(op, ..) => match op {
                BinaryOp::Add => gradient2(out, operands, |[a, b], [ga, gb]| {
                    binary_gradient(BinaryOp::Add, a, b, ga, gb)
                }),
                BinaryOp::Sub => gradient2(out, operands, |[a, b], [ga, gb]| {
                    binary_gradient(BinaryOp::Sub, a, b, ga, gb)
                }),
                BinaryOp::Mul => gradient2(out, operands, |[a, b], [ga, gb]| {
                    binary_gradient(BinaryOp::Mul, a, b, ga, gb)
                }),
                BinaryOp::Min => gradient2(out, operands, |[a, b], [ga, gb]| {
                    binary_gradient(BinaryOp::Min, a, b, ga, gb)
                }),
                BinaryOp::Max => gradient2(out, operands, |[a, b], [ga, gb]| {
                    binary_gradient(BinaryOp::Max, a, b, ga, gb)
                }),
            },
            Kernel::Atan2(..) => gradient2(out, operands, |[y, x], [gy, gx]| {
                atan2_gradient(y, x, gy, gx)
            }),
            Kernel::Mix(..) => gradient3(out, operands, mix_gradient),
            Kernel::Abs(_) => gradient1(out, operands, abs_gradient),
            Kernel::Clamp(_, min, max) => {
                gradient1(out, operands, |a, g| clamp_gradient(a, min, max, g));
            }
            Kernel::AsMask(_) => gradient1(out, operands, |a, g| clamp_gradient(a, 0.0, 1.0, g)),
            Kernel::Remap { scale, .. } => gradient1(out, operands, |_, g| g * scale),
            Kernel::Fract(_) => gradient1(out, operands, |_, g| g),
            _ => unreachable!("only arithmetic reaches here"),
        }
        return;
    }
    // Everything else, lane by lane as the point evaluation does it.
    let count = operands.args.len();
    at.solid(|i, p, footprint| {
        let mut values = [Value::Scalar(0.0); 3];
        let mut gradients = [Vec3::ZERO; 3];
        for k in 0..count {
            values[k] = operands.value(k, i);
            gradients[k] = vec3(operands.gradient(k), i);
        }
        let (value, gradient) =
            program.pure_gradient(node, p, footprint, &values[..count], &gradients[..count]);
        out.store(i, value);
        out.store_gradient(slot.components(), i, gradient);
    });
}
