// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Element compositing in isolation.
//!
//! Four hundred jittered, partly rotated bricks and pebbles on a wrapping
//! 256 by 256 grid, realized through one scoped program with per-material,
//! per-element and per-sample nodes, one of them sampling a noise
//! resource.
//!
//! - `elements/composite`: `Realized::composite` from nothing.
//! - `elements/update_one`: `Realized::update` after moving one element
//!   (which recomposites the tiles its old and new bounds reach).

use std::sync::Arc;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use dapple_elements::{
    Anchor, AttributeDecl, Binding, Composite, Element, ElementKey, ElementSet, InstanceId,
    LayoutId, Outline, Placement, ProgramInstance, Realized,
};
use dapple_field::program::{Op, ProgramBuilder};
use dapple_field::scoped::{Node, Scope, ScopedBuilder};
use dapple_field::{Basis, Domain, PortType, Value};
use glam::{Vec2, Vec3};

const SIZE: u32 = 256;
const SIDE: i32 = 20;

fn domain() -> Domain {
    Domain::periodic(1, 1).expect("a unit period")
}

fn set() -> ElementSet {
    let layout = LayoutId::named("bench.elements");
    let schema = vec![AttributeDecl {
        name: String::from("tone"),
        port: PortType::Scalar,
    }];
    let mut elements = Vec::new();
    #[expect(clippy::cast_precision_loss, reason = "small lattice coordinates")]
    for j in 0..SIDE {
        for i in 0..SIDE {
            let key = ElementKey::new(layout, Anchor([i, j]), 0);
            let cell = 1.0 / SIDE as f32;
            let jitter = Vec2::new(key.unit(1) - 0.5, key.unit(2) - 0.5) * cell * 0.5;
            let center = (Vec2::new(i as f32, j as f32) + 0.5) * cell + jitter;
            elements.push(Element {
                key,
                placement: Placement {
                    center,
                    rotation: (key.unit(3) - 0.5) * 0.6,
                },
                half_size: Vec2::new(0.5, 0.35 + 0.15 * key.unit(4)) * cell * 1.05,
                outline: if key.unit(5) < 0.25 {
                    Outline::Ellipse
                } else {
                    Outline::Rectangle
                },
                variant: 0,
                attributes: vec![Value::Scalar(key.unit(6))],
            });
        }
    }
    ElementSet::new(schema, elements).expect("a valid set")
}

fn instance() -> ProgramInstance {
    let mut noise = ProgramBuilder::new();
    let n = noise
        .add(Op::Noise {
            basis: Basis::Gradient,
            domain: Domain::Plane,
            frequency: [40.0, 40.0],
            seed: 5,
        })
        .expect("noise");
    let noise = noise.finish_value(n).expect("a value program");

    let mut b = ScopedBuilder::new("bench.glaze");
    let tone = b.input("tone", PortType::Scalar, Scope::Element).unwrap();
    let local = b.input("local", PortType::Vector2, Scope::Sample).unwrap();
    let edge = b.input("edge", PortType::Scalar, Scope::Sample).unwrap();
    let seed = b.input("seed", PortType::Scalar, Scope::Element).unwrap();
    let chips = b.resource("chips", noise);
    let zero = b.constant(Value::Scalar(0.0));
    let bevel = b.constant(Value::Scalar(0.02));
    let red = b.constant(Value::Vector3(Vec3::new(0.8, 0.2, 0.1)));
    let blue = b.constant(Value::Vector3(Vec3::new(0.1, 0.2, 0.8)));
    let body = b.constant(Value::Vector3(Vec3::splat(0.5)));
    let tint = b
        .node(Node::Mix {
            a: red,
            b: blue,
            t: tone,
        })
        .unwrap();
    let shape = b
        .node(Node::SmoothStep {
            edge0: zero,
            edge1: bevel,
            x: edge,
        })
        .unwrap();
    let offset = b.node(Node::Vector2(seed, seed)).unwrap();
    let at = b.node(Node::Add(local, offset)).unwrap();
    let n = b
        .node(Node::Sample {
            resource: chips,
            at,
        })
        .unwrap();
    let lo = b.constant(Value::Scalar(0.3));
    let hi = b.constant(Value::Scalar(0.4));
    let chip = b
        .node(Node::SmoothStep {
            edge0: lo,
            edge1: hi,
            x: n,
        })
        .unwrap();
    let color = b
        .node(Node::Mix {
            a: tint,
            b: body,
            t: chip,
        })
        .unwrap();
    let height = b.node(Node::Sub(shape, chip)).unwrap();
    let glaze = b.constant(Value::Id(0));
    let exposed = b.constant(Value::Id(1));
    let material = b
        .node(Node::Select {
            condition: chip,
            a: exposed,
            b: glaze,
        })
        .unwrap();
    b.output("color", PortType::Vector3, Scope::Sample, color)
        .unwrap();
    b.output("height", PortType::Scalar, Scope::Sample, height)
        .unwrap();
    b.output("material", PortType::Id, Scope::Sample, material)
        .unwrap();
    b.output("tint", PortType::Vector3, Scope::Element, tint)
        .unwrap();
    ProgramInstance::new(
        InstanceId::named("bench.glaze"),
        Arc::new(b.finish()),
        vec![
            Binding::Attribute(String::from("tone")),
            Binding::LocalPosition,
            Binding::EdgeDistance,
            Binding::ElementRandom(7),
        ],
    )
    .expect("the bindings fit")
}

const BACKGROUND: [Value; 4] = [
    Value::Vector3(Vec3::splat(0.3)),
    Value::Scalar(-0.1),
    Value::Id(2),
    Value::Vector3(Vec3::ZERO),
];

fn composite<'a>(set: &'a ElementSet, instance: &'a ProgramInstance) -> Composite<'a> {
    Composite {
        set,
        instance,
        background: &BACKGROUND,
        domain: domain(),
        width: SIZE,
        height: SIZE,
        tile_size: 32,
    }
}

fn bench_elements(c: &mut Criterion) {
    let (set, instance) = (set(), instance());
    let texels = Throughput::Elements(u64::from(SIZE) * u64::from(SIZE));

    let mut group = c.benchmark_group("elements/composite");
    group.sample_size(20);
    group.throughput(texels.clone());
    group.bench_function("400", |b| {
        b.iter(|| Realized::composite(&composite(&set, &instance)).expect("composites"));
    });
    group.finish();

    let mut edited = set.clone();
    let key = set.keys()[137];
    let mut placement = set.placement(set.index_of(key).unwrap());
    placement.center += Vec2::new(0.013, -0.009);
    edited.set_placement(key, placement).expect("moves");
    let base = Realized::composite(&composite(&set, &instance)).expect("composites");
    let mut group = c.benchmark_group("elements/update_one");
    group.sample_size(20);
    group.bench_function("400", |b| {
        b.iter_batched(
            || base.clone(),
            |mut realized| {
                realized
                    .update(&composite(&edited, &instance))
                    .expect("updates")
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_elements);
criterion_main!(benches);
