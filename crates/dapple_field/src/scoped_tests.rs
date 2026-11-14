// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Scoped programs evaluated over runs give each sample's bits.

use alloc::sync::Arc;
use alloc::vec;

use glam::{Vec2, Vec3};

use crate::hash::{hash, unit_f32};
use crate::program::{Op, ProgramBuilder, ValueProgram};
use crate::scoped::{
    CompareOp, Node, NodeRef, Scope, ScopedBuilder, ScopedProgram, Shape, UnaryOp,
};
use crate::shaping::{ColorRamp, ToneCurve};
use crate::{Basis, Domain, Footprint, FractalParams, PortType, Primaries, Value};

fn same_bits(a: Value, b: Value) -> bool {
    let bits = |x: f32| if x.is_nan() { u32::MAX } else { x.to_bits() };
    match (a, b) {
        (Value::Id(a), Value::Id(b)) => a == b,
        (Value::Scalar(a), Value::Scalar(b)) => bits(a) == bits(b),
        (Value::Vector2(a), Value::Vector2(b)) => a.to_array().map(bits) == b.to_array().map(bits),
        (Value::Vector3(a), Value::Vector3(b)) => a.to_array().map(bits) == b.to_array().map(bits),
        _ => false,
    }
}

/// A color field to sample: fractal noise ramped into a color.
fn field(seed: u64) -> ValueProgram {
    let domain = Domain::periodic(1, 1).unwrap();
    let mut b = ProgramBuilder::new();
    let n = b
        .add(Op::Fractal {
            basis: Basis::Gradient,
            domain,
            frequency: [4.0, 4.0],
            seed,
            params: FractalParams::default(),
        })
        .unwrap();
    let half = b.add(Op::Constant { domain, value: 0.5 }).unwrap();
    let color = b
        .add(Op::Color {
            r: n,
            g: half,
            b: n,
        })
        .unwrap();
    b.finish_value(color).unwrap()
}

/// `f(x, k) = sample(x, k)·r + |x|`, sampling its own resource, with
/// constants of its own.
fn function() -> Arc<ScopedProgram> {
    let mut f = ScopedBuilder::new("f");
    let x = f.input("x", PortType::Scalar, Scope::Pass).unwrap();
    let k = f.input("k", PortType::Scalar, Scope::Pass).unwrap();
    let at = f.node(Node::Vector2(x, k)).unwrap();
    let resource = f.resource("grain", field(3));
    let color = f.sample(resource, at).unwrap();
    let r = f.component(color, 0).unwrap();
    let scaled = f.scale(0.25, r).unwrap();
    let size = f.node(Node::Unary(UnaryOp::Abs, x)).unwrap();
    let y = f.add(scaled, size).unwrap();
    f.output("y", PortType::Scalar, Scope::Pass, y).unwrap();
    f.output(
        "color",
        PortType::Color(Primaries::Rec709),
        Scope::Pass,
        color,
    )
    .unwrap();
    Arc::new(f.finish())
}

/// A program using every node kind, inputs of every scope and shape,
/// resources sampled per sample and once, functions called per element
/// and per sample (one calling another), and outputs of every scope.
fn zoo() -> ScopedProgram {
    let mut b = ScopedBuilder::new("zoo");
    let m = b.input("m", PortType::Scalar, Scope::Material).unwrap();
    let e = b.input("e", PortType::Scalar, Scope::Element).unwrap();
    let s = b.input("s", PortType::Scalar, Scope::Sample).unwrap();
    let pos = b.input("pos", PortType::Vector2, Scope::Sample).unwrap();
    let tint = b
        .input("tint", PortType::Color(Primaries::Rec709), Scope::Sample)
        .unwrap();
    let id = b.input("id", PortType::Id, Scope::Sample).unwrap();
    let ao = b.input("ao", PortType::Scalar, Scope::Pass).unwrap();

    let wood = b.resource("wood", field(7));
    let sampled = b.sample(wood, pos).unwrap();
    let corner = b.constant(Value::Vector2(Vec2::new(0.3, 0.7)));
    let once = b.sample(wood, corner).unwrap();
    let f = b.function("f", function());
    // g(x) = f(f(x, 1), x): a function calling a function.
    let g = {
        let mut g = ScopedBuilder::new("g");
        let x = g.input("x", PortType::Scalar, Scope::Pass).unwrap();
        let fi = g.function("f", function());
        let one = g.scalar(1.0);
        let inner = g
            .node(Node::Call {
                function: fi,
                output: 0,
                args: vec![x, one],
            })
            .unwrap();
        let outer = g
            .node(Node::Call {
                function: fi,
                output: 0,
                args: vec![inner, x],
            })
            .unwrap();
        g.output("y", PortType::Scalar, Scope::Pass, outer).unwrap();
        b.function("g", Arc::new(g.finish()))
    };
    let per_element = b
        .node(Node::Call {
            function: f,
            output: 0,
            args: vec![e, m],
        })
        .unwrap();
    let per_sample = b
        .node(Node::Call {
            function: f,
            output: 1,
            args: vec![s, e],
        })
        .unwrap();
    let nested = b
        .node(Node::Call {
            function: g,
            output: 0,
            args: vec![s],
        })
        .unwrap();

    let mut nodes = vec![sampled, once, per_element, per_sample, nested];
    let x = b.add(s, per_element).unwrap();
    for op in [
        Node::Add(x, m),
        Node::Sub(x, e),
        Node::Mul(x, ao),
        Node::Div(x, e),
        Node::Min(x, m),
        Node::Max(x, ao),
        Node::Pow(x, m),
        Node::Mul(tint, x),
        Node::Add(sampled, tint),
        Node::Sub(pos, x),
        Node::Div(x, pos),
        Node::Mix {
            a: tint,
            b: sampled,
            t: x,
        },
        Node::Clamp { x, lo: m, hi: e },
        Node::Clamp {
            x: tint,
            lo: m,
            hi: x,
        },
        Node::SmoothStep {
            edge0: m,
            edge1: e,
            x,
        },
        Node::SmoothStep {
            edge0: m,
            edge1: m,
            x,
        },
        Node::Select {
            condition: x,
            a: tint,
            b: sampled,
        },
        Node::Component {
            input: per_sample,
            index: 2,
        },
        Node::Component {
            input: pos,
            index: 1,
        },
        Node::Vector2(x, ao),
        Node::Vector3(x, e, m),
        Node::Length(x),
        Node::Length(pos),
        Node::Length(tint),
        Node::Dot(tint, sampled),
        Node::Dot(pos, pos),
        Node::Cross(tint, sampled),
        Node::Normalize(pos),
        Node::Normalize(tint),
        Node::Curve(
            x,
            ToneCurve::new(&[[0.0, 0.0], [0.5, 0.2], [1.0, 1.0]]).unwrap(),
        ),
        Node::Ramp(
            x,
            ColorRamp::new(&[(0.0, Vec3::ZERO), (0.4, Vec3::X), (1.0, Vec3::ONE)]).unwrap(),
        ),
    ] {
        nodes.push(b.node(op).unwrap());
    }
    for op in [
        UnaryOp::Negate,
        UnaryOp::Abs,
        UnaryOp::Floor,
        UnaryOp::Fract,
        UnaryOp::Sqrt,
        UnaryOp::Exp,
        UnaryOp::Ln,
        UnaryOp::Sin,
        UnaryOp::Cos,
        UnaryOp::Saturate,
    ] {
        nodes.push(b.node(Node::Unary(op, x)).unwrap());
        nodes.push(b.node(Node::Unary(op, tint)).unwrap());
    }
    for op in [
        CompareOp::Less,
        CompareOp::LessEqual,
        CompareOp::Greater,
        CompareOp::GreaterEqual,
        CompareOp::Equal,
        CompareOp::NotEqual,
    ] {
        nodes.push(b.node(Node::Compare(op, x, m)).unwrap());
    }
    let three = b.constant(Value::Id(3));
    nodes.push(b.node(Node::Compare(CompareOp::Equal, id, three)).unwrap());
    nodes.push(
        b.node(Node::Compare(CompareOp::NotEqual, id, three))
            .unwrap(),
    );
    let constant = b.scalar(2.5);
    nodes.push(constant);
    for (i, node) in nodes.into_iter().enumerate() {
        let port = match b_shape(&b, node) {
            Shape::Scalar => PortType::Scalar,
            Shape::Vector2 => PortType::Vector2,
            Shape::Vector3 => PortType::Vector3,
            Shape::Id => PortType::Id,
        };
        b.output(&alloc::format!("out{i}"), port, Scope::Pass, node)
            .unwrap();
    }
    b.finish()
}

fn b_shape(b: &ScopedBuilder, node: NodeRef) -> Shape {
    b.clone().finish().shape(node)
}

/// Input `input` of `p` at sample `lane`, fixed for the material and
/// element scopes, varied per sample otherwise: random values near and far,
/// signed zeros, NaN and huge values.
fn input(p: &ScopedProgram, input: u32, lane: usize) -> Value {
    let lane = match p.inputs()[input as usize].scope {
        Scope::Material | Scope::Region | Scope::Element => 0,
        Scope::Sample | Scope::Pass => lane,
    };
    unscoped(p, input, lane)
}

/// Input `input` of `p` at sample `lane`, whatever its scope.
fn unscoped(p: &ScopedProgram, input: u32, lane: usize) -> Value {
    let decl = &p.inputs()[input as usize];
    let h = |axis: u64| {
        let i = lane as u64;
        match i % 23 {
            0 => [0.0, -0.0, f32::NAN][(axis % 3) as usize],
            1 => [1e12, -1e-30, 0.5][(axis % 3) as usize],
            _ => {
                let scale = [1.0, 2.0, 1e3][(i % 3) as usize];
                (unit_f32(hash(0x7363_6f70, &[i, axis, u64::from(input)])) * 2.0 - 1.0) * scale
            }
        }
    };
    match Shape::of(decl.port) {
        Shape::Scalar => Value::Scalar(h(0)),
        Shape::Vector2 => Value::Vector2(Vec2::new(h(0), h(1))),
        Shape::Vector3 => Value::Vector3(Vec3::new(h(0), h(1), h(2))),
        Shape::Id => Value::Id(u32::try_from(lane % 5).unwrap()),
    }
}

#[test]
fn runs_evaluate_each_sample_as_alone() {
    let p = zoo();
    let fp = Footprint::new(0.01).unwrap();
    let hoists: [fn(Scope) -> bool; 3] = [
        |_| false,
        |s| s == Scope::Material,
        |s| s.within(Scope::Element),
    ];
    for hoist in hoists {
        let mut kept = p.evaluator();
        let mut hoisted = vec![None; p.nodes().len()];
        p.evaluate(
            &mut hoisted,
            hoist,
            &mut |i| input(&p, i, 0),
            Footprint::POINT,
        );
        kept.hoist(hoist, &mut |i| input(&p, i, 0), Footprint::POINT);
        let mut first = 0;
        // Runs that grow, shrink and are empty, through one evaluator.
        for len in [5, 1, 0, 64, 3, 130, 7] {
            kept.eval_batch(
                len,
                &mut |i, values| {
                    for (lane, value) in values.iter_mut().enumerate() {
                        *value = input(&p, i, first + lane);
                    }
                },
                fp,
            );
            for lane in 0..len {
                let mut values = hoisted.clone();
                p.evaluate(
                    &mut values,
                    |_| true,
                    &mut |i| input(&p, i, first + lane),
                    fp,
                );
                for (o, decl) in p.outputs().iter().enumerate() {
                    let alone = values[decl.node.index()].unwrap();
                    let run = kept.output(o)[lane];
                    assert!(
                        same_bits(run, alone),
                        "output {o} at sample {}: {run:?} != {alone:?}",
                        first + lane
                    );
                }
            }
            for o in 0..p.outputs().len() {
                assert_eq!(kept.output(o).len(), len);
            }
            first += len;
        }
    }
}

#[test]
fn hoisting_after_runs_fixes_every_lane() {
    let p = zoo();
    let read = |salt: usize| {
        let p = &p;
        move |i, values: &mut [Value]| {
            for (lane, value) in values.iter_mut().enumerate() {
                // Salted, material inputs vary per sample too.
                *value = if salt == 0 {
                    input(p, i, lane)
                } else {
                    unscoped(p, i, lane + salt)
                };
            }
        }
    };
    let hoist = |s: Scope| s == Scope::Material;
    let mut fresh = p.evaluator();
    fresh.hoist(hoist, &mut |i| input(&p, i, 0), Footprint::POINT);
    fresh.eval_batch(9, &mut read(0), Footprint::POINT);
    // Runs before hoisting fill every column; hoisting then fixes the
    // hoisted ones for runs short and long within the same buffers.
    let mut kept = p.evaluator();
    kept.eval_batch(40, &mut read(1000), Footprint::POINT);
    kept.hoist(hoist, &mut |i| input(&p, i, 0), Footprint::POINT);
    kept.eval_batch(3, &mut read(0), Footprint::POINT);
    kept.eval_batch(9, &mut read(0), Footprint::POINT);
    for o in 0..p.outputs().len() {
        for (a, b) in fresh.output(o).iter().zip(kept.output(o)) {
            assert!(same_bits(*a, *b), "output {o}: {a:?} != {b:?}");
        }
    }
}
