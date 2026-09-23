// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Slice 1a's acceptance gates, and the program contract.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use dapple_field::program::{Op, ProgramBuilder};
use dapple_field::{Basis, Domain, PortType, Value};
use glam::{Vec2, Vec3};

use crate::*;

fn domain() -> Domain {
    Domain::periodic(1, 1).unwrap()
}

fn schema() -> Vec<AttributeDecl> {
    vec![
        AttributeDecl {
            name: String::from("tone"),
            port: PortType::Scalar,
        },
        AttributeDecl {
            name: String::from("thickness"),
            port: PortType::Scalar,
        },
    ]
}

/// A 4 × 4 running bond with `joint`-wide gaps and key-derived attributes.
fn bricks(joint: f32) -> ElementSet {
    RunningBond {
        layout: LayoutId::named("test.wall"),
        domain: domain(),
        courses: 4,
        per_course: 4,
        joint,
    }
    .elements(schema(), |key, _| {
        vec![Value::Scalar(key.unit(1)), Value::Scalar(key.unit(2))]
    })
    .unwrap()
}

/// Outputs: `color` (per sample), `height` (per sample), `material` (glaze
/// 0 or chipped body 1, per sample), `tint` (per element).
fn program() -> SurfaceProgram {
    let mut noise = ProgramBuilder::new();
    let n = noise
        .add(Op::Noise {
            basis: Basis::Gradient,
            domain: Domain::Plane,
            frequency: [40.0, 40.0],
            seed: 5,
        })
        .unwrap();
    let noise = noise.finish_value(n).unwrap();

    let mut b = SurfaceBuilder::new("test.glaze");
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
        .add(Node::Mix {
            a: red,
            b: blue,
            t: tone,
        })
        .unwrap();
    let shape = b
        .add(Node::SmoothStep {
            edge0: zero,
            edge1: bevel,
            x: edge,
        })
        .unwrap();
    let offset = b.add(Node::Vector2(seed, seed)).unwrap();
    let at = b.add(Node::Add(local, offset)).unwrap();
    let n = b
        .add(Node::Sample {
            resource: chips,
            at,
        })
        .unwrap();
    let lo = b.constant(Value::Scalar(0.3));
    let hi = b.constant(Value::Scalar(0.4));
    let chip = b
        .add(Node::SmoothStep {
            edge0: lo,
            edge1: hi,
            x: n,
        })
        .unwrap();
    let color = b
        .add(Node::Mix {
            a: tint,
            b: body,
            t: chip,
        })
        .unwrap();
    let height = b.add(Node::Sub(shape, chip)).unwrap();
    let glaze = b.constant(Value::Id(0));
    let exposed = b.constant(Value::Id(1));
    let material = b
        .add(Node::Select {
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
    b.finish()
}

fn instance() -> ProgramInstance {
    ProgramInstance::new(
        InstanceId::named("test.glaze"),
        Arc::new(program()),
        vec![
            Binding::Attribute(String::from("tone")),
            Binding::LocalPosition,
            Binding::EdgeDistance,
            Binding::ElementRandom(7),
        ],
    )
    .unwrap()
}

const BACKGROUND: [Value; 4] = [
    Value::Vector3(Vec3::splat(0.3)),
    Value::Scalar(-0.1),
    Value::Id(2),
    Value::Vector3(Vec3::ZERO),
];

fn composite<'a>(set: &'a ElementSet, instance: &'a ProgramInstance, size: u32) -> Composite<'a> {
    Composite {
        set,
        instance,
        background: &BACKGROUND,
        domain: domain(),
        width: size,
        height: size,
        tile_size: 8,
    }
}

#[expect(clippy::cast_precision_loss, reason = "test grids are small")]
fn f(v: u32) -> f32 {
    v as f32
}

/// The texel whose center is nearest to domain point `p`.
fn texel_at(p: Vec2, size: u32) -> (u32, u32) {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "test points lie inside the unit period"
    )]
    let t = |v: f32| ((v.rem_euclid(1.0) * f(size)) as u32).min(size - 1);
    (t(p.x), t(p.y))
}

// Gate 1.
#[test]
fn moving_a_brick_keeps_its_key_and_appearance_and_recomputes_both_bounds() {
    let inst = instance();
    let before = bricks(0.1);
    let mut realized = Realized::composite(&composite(&before, &inst, 64)).unwrap();
    let key = before.keys()[5];
    let i = before.index_of(key).unwrap();
    let old = before.placement(i);
    let old_bounds = before.bounds(i);
    let (x0, y0) = texel_at(old.center, 64);
    assert_eq!(realized.owner(x0, y0), Some(key));
    let color = realized.value(0, x0, y0);
    let tint = realized.value(3, x0, y0);

    // Four texels to the right, into the gap.
    let mut after = before.clone();
    let moved = Placement::at(old.center + Vec2::new(4.0 / 64.0, 0.0));
    after.set_placement(key, moved).unwrap();
    assert_eq!(after.keys(), before.keys(), "moving changes no identity");
    assert_ne!(after.fingerprint_of(i), before.fingerprint_of(i));
    let report = realized.update(&composite(&after, &inst, 64)).unwrap();
    assert!(!report.whole);
    assert_eq!(report.correspondence.changed, vec![key]);

    // The brick owns its new center and looks the same there: the
    // identity-derived tint exactly, the sampled color to rounding.
    let x1 = (x0 + 4) % 64;
    assert_eq!(realized.owner(x1, y0), Some(key));
    assert_eq!(realized.value(3, x1, y0), tint);
    let (Value::Vector3(a), Value::Vector3(b)) = (color, realized.value(0, x1, y0)) else {
        panic!("colors are vectors")
    };
    assert!((a - b).abs().max_element() < 1e-5, "{a} vs {b}");

    // The recomputed tiles cover both the old and the new bounds.
    let pad = 1.0 / 64.0;
    for bounds in [old_bounds, after.bounds(i)] {
        for tile in realized.tiles_reached(bounds.grown(pad)) {
            assert!(report.tiles.contains(&tile), "tile {tile} not recomputed");
        }
    }
    assert!(
        report.tiles.len() < 64,
        "only nearby tiles: {}",
        report.tiles.len()
    );
    let clean = Realized::composite(&composite(&after, &inst, 64)).unwrap();
    assert_eq!(realized.digest(), clean.digest());
}

// Gate 2.
#[test]
fn changing_one_glaze_leaves_other_elements_and_regions_unchanged() {
    let inst = instance();
    let before = bricks(0.02);
    let key = before.keys()[3];
    let i = before.index_of(key).unwrap();
    let mut after = before.clone();
    after
        .set_attribute(key, "tone", Value::Scalar(0.999))
        .unwrap();

    // Every other element's attributes, and every attribute but this one,
    // are untouched.
    for (name, changed) in [("tone", true), ("thickness", false)] {
        let (a, b) = (
            before.attribute(name).unwrap(),
            after.attribute(name).unwrap(),
        );
        for j in 0..before.len() {
            assert_eq!(a[j] != b[j], changed && j == i, "{name} of element {j}");
        }
    }
    let report = correspondence(&before, &after);
    assert_eq!(report.changed, vec![key]);
    assert_eq!(report.unchanged.len(), before.len() - 1);

    // Outside the brick's reach, every output and label keeps its bits.
    let a = Realized::composite(&composite(&before, &inst, 64)).unwrap();
    let b = Realized::composite(&composite(&after, &inst, 64)).unwrap();
    let reach = before.bounds(i).grown(1.0 / 64.0);
    let mut inside_changed = false;
    for y in 0..64 {
        for x in 0..64 {
            let p = (Vec2::new(f(x), f(y)) + 0.5) / 64.0;
            let near = [-1.0, 0.0, 1.0].iter().any(|&dx| {
                [-1.0, 0.0, 1.0]
                    .iter()
                    .any(|&dy| reach.contains(p + Vec2::new(dx, dy)))
            });
            for o in 0..4 {
                let same = a.value(o, x, y) == b.value(o, x, y);
                if near {
                    inside_changed |= !same;
                } else {
                    assert!(same, "output {o} changed at ({x}, {y}), outside the brick");
                }
            }
            assert_eq!(a.owner(x, y), b.owner(x, y));
        }
    }
    assert!(inside_changed, "the glaze change shows on the brick");
}

// Gate 3.
#[test]
fn identity_is_independent_of_resolution_labels_and_storage_order() {
    let inst = instance();
    let set = bricks(0.05);
    let coarse = Realized::composite(&composite(&set, &inst, 32)).unwrap();
    let fine = Realized::composite(&composite(&set, &inst, 96)).unwrap();
    assert_eq!(coarse.keys(), fine.keys());
    for i in 0..set.len() {
        let center = set.placement(i).center;
        let (cx, cy) = texel_at(center, 32);
        let (fx, fy) = texel_at(center, 96);
        assert_eq!(coarse.owner(cx, cy), Some(set.keys()[i]));
        assert_eq!(fine.owner(fx, fy), Some(set.keys()[i]));
        // The per-element output is the element's, at any resolution.
        assert_eq!(coarse.value(3, cx, cy), fine.value(3, fx, fy));
    }
    // Supplying the elements in another order gives the same set and bits.
    let mut shuffled: Vec<Element> = (0..set.len()).map(|i| set.element(i)).collect();
    shuffled.reverse();
    shuffled.swap(2, 9);
    let reordered = ElementSet::new(schema(), shuffled).unwrap();
    assert_eq!(reordered, set);
    let again = Realized::composite(&composite(&reordered, &inst, 32)).unwrap();
    assert_eq!(again.digest(), coarse.digest());
    // Labels are dense indices into the key table, not identities.
    let labels = coarse.owner_labels().unwrap();
    assert_eq!(labels.port(), PortType::Id);
}

// Gate 4.
#[test]
fn edit_histories_reach_the_same_canonical_output() {
    let inst = instance();
    let a_key = bricks(0.1).keys()[1];
    let b_key = bricks(0.1).keys()[10];
    let edit_tone = |set: &mut ElementSet| {
        set.set_attribute(a_key, "tone", Value::Scalar(0.25))
            .unwrap();
    };
    let edit_move = |set: &mut ElementSet| {
        let i = set.index_of(b_key).unwrap();
        let mut p = set.placement(i);
        p.center += Vec2::new(0.02, -0.01);
        p.rotation = 0.1;
        set.set_placement(b_key, p).unwrap();
    };

    // History A: the final joint, tone then move.
    let mut a = bricks(0.1);
    edit_tone(&mut a);
    edit_move(&mut a);
    // History B: another joint and a move first, then the layout re-run
    // with the final joint (keys survive), move then tone.
    let mut b = bricks(0.04);
    edit_move(&mut b);
    assert_ne!(b, a);
    b = bricks(0.1);
    edit_move(&mut b);
    edit_tone(&mut b);
    assert_eq!(a, b);
    assert_eq!(a.fingerprint(), b.fingerprint());

    // History C: incremental updates along the way, starting elsewhere.
    let start = bricks(0.04);
    let mut realized = Realized::composite(&composite(&start, &inst, 48)).unwrap();
    let mut c = bricks(0.1);
    realized.update(&composite(&c, &inst, 48)).unwrap();
    edit_move(&mut c);
    realized.update(&composite(&c, &inst, 48)).unwrap();
    edit_tone(&mut c);
    realized.update(&composite(&c, &inst, 48)).unwrap();

    let clean_a = Realized::composite(&composite(&a, &inst, 48)).unwrap();
    let clean_b = Realized::composite(&composite(&b, &inst, 48)).unwrap();
    assert_eq!(clean_a.digest(), clean_b.digest());
    assert_eq!(realized.digest(), clean_a.digest());
    assert_eq!(realized, clean_a);
}

// Gate 5.
#[test]
fn incremental_updates_equal_a_clean_reference() {
    let inst = instance();
    let full = bricks(0.03);
    let mut set = full.clone();
    let mut realized = Realized::composite(&composite(&set, &inst, 64)).unwrap();
    let keys: Vec<ElementKey> = full.keys().to_vec();
    let check = |realized: &Realized, set: &ElementSet| {
        let clean = Realized::composite(&composite(set, &inst, 64)).unwrap();
        assert_eq!(realized, &clean);
    };

    // Moves across the period's edge, rotations, attribute edits.
    for (step, &key) in keys.iter().enumerate().take(6) {
        let i = set.index_of(key).unwrap();
        let mut p = set.placement(i);
        let s = f(u32::try_from(step).unwrap());
        p.center += Vec2::new(0.9 - 0.13 * s, 0.07 * s);
        p.rotation = 0.2 * s;
        set.set_placement(key, p).unwrap();
        set.set_attribute(key, "thickness", Value::Scalar(0.1 * s))
            .unwrap();
        let report = realized.update(&composite(&set, &inst, 64)).unwrap();
        assert!(!report.whole);
        check(&realized, &set);
    }
    // Removing elements renumbers labels; restoring them adds them back.
    let removed = set.filter(|e| e.key != keys[7] && e.key != keys[11]);
    let report = realized.update(&composite(&removed, &inst, 64)).unwrap();
    assert!(report.relabeled);
    assert_eq!(report.correspondence.removed.len(), 2);
    check(&realized, &removed);
    let report = realized.update(&composite(&full, &inst, 64)).unwrap();
    assert_eq!(report.correspondence.added.len(), 2);
    check(&realized, &full);
    // A new program binding recomputes whole, and says so.
    let rebound = inst.rebind("seed", Binding::ElementRandom(8)).unwrap();
    let report = realized.update(&composite(&full, &rebound, 64)).unwrap();
    assert!(report.whole);
    check_rebound(&realized, &full, &rebound);
    assert_eq!(
        rebound.id(),
        inst.id(),
        "rebinding keeps the instance identity"
    );
    assert_ne!(rebound.fingerprint(), inst.fingerprint());
}

fn check_rebound(realized: &Realized, set: &ElementSet, inst: &ProgramInstance) {
    let clean = Realized::composite(&composite(set, inst, 64)).unwrap();
    assert_eq!(realized, &clean);
}

// Gate 6.
#[test]
fn mixed_texels_separate_ownership_from_contribution() {
    let inst = instance();
    // Touching bricks whose edges fall mid-texel.
    let set = bricks(0.0);
    let c = composite(&set, &inst, 60);
    let realized = Realized::composite(&c).unwrap();
    let mut shared = None;
    'search: for y in 0..60 {
        for x in 0..60 {
            let contributors = Realized::contributors(&c, x, y).unwrap();
            if contributors.elements.len() >= 2 {
                shared = Some((x, y, contributors));
                break 'search;
            }
        }
    }
    let (x, y, contributors) = shared.expect("some texel straddles two bricks");
    let owner = realized.owner(x, y).expect("an element owns it");
    // The label names one element; the contributors name every one.
    let best = contributors
        .elements
        .iter()
        .map(|e| e.coverage)
        .fold(0.0, f32::max);
    let owner_coverage = contributors
        .elements
        .iter()
        .find(|e| e.key == owner)
        .expect("the owner contributes")
        .coverage;
    assert_eq!(owner_coverage, best, "the owner is the largest contributor");
    assert!(owner_coverage < 1.0, "and not the only one");
    // The continuous output mixes every contributor's tint; the owner's
    // per-element tint alone differs from it.
    let mixed = realized.value(3, x, y);
    let owner_tint = {
        let i = set.index_of(owner).unwrap();
        let (ox, oy) = texel_at(set.placement(i).center, 60);
        realized.value(3, ox, oy)
    };
    assert_ne!(mixed, owner_tint, "contributions blend");
    // The identifier output follows the owner; it is never blended.
    assert!(matches!(realized.value(2, x, y), Value::Id(0 | 1)));

    // A brick-and-mortar texel: part element, part background, and the
    // shares add up.
    let wide = bricks(0.05);
    let c = composite(&wide, &inst, 60);
    let mut edged = None;
    'joint: for y in 0..60 {
        for x in 0..60 {
            let contributors = Realized::contributors(&c, x, y).unwrap();
            if contributors.elements.len() == 1 && contributors.background > 0.0 {
                edged = Some(contributors);
                break 'joint;
            }
        }
    }
    let edged = edged.expect("some texel straddles a joint");
    let total = edged.elements[0].coverage + edged.background;
    assert!((total - 1.0).abs() < 1e-6);
}

#[test]
fn contracts_are_checked() {
    let mut b = SurfaceBuilder::new("bad");
    let local = b.input("local", PortType::Vector2, Scope::Sample).unwrap();
    let x = b
        .add(Node::Component {
            input: local,
            index: 0,
        })
        .unwrap();
    // A per-element output cannot depend on the sample position.
    assert!(matches!(
        b.output("tone", PortType::Scalar, Scope::Element, x),
        Err(ContractError::ScopeViolation {
            declared: Scope::Element,
            found: Scope::Sample,
            ..
        })
    ));
    // Shapes are checked as nodes are added.
    let id = b.constant(Value::Id(3));
    assert!(matches!(
        b.add(Node::Add(id, x)),
        Err(ContractError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        b.add(Node::Component { input: x, index: 0 }),
        Err(ContractError::ShapeMismatch { .. })
    ));
    b.output("x", PortType::Scalar, Scope::Sample, x).unwrap();
    assert!(matches!(
        b.output("x", PortType::Scalar, Scope::Sample, x),
        Err(ContractError::DuplicateName(_))
    ));

    // A per-sample binding cannot feed a per-element input.
    let program = Arc::new(program());
    let bad = ProgramInstance::new(
        InstanceId::named("bad"),
        Arc::clone(&program),
        vec![
            Binding::EdgeDistance,
            Binding::LocalPosition,
            Binding::EdgeDistance,
            Binding::ElementRandom(7),
        ],
    );
    assert!(matches!(bad, Err(ContractError::ScopeViolation { .. })));
    // Attribute bindings are checked against the set.
    let unknown = instance()
        .rebind("tone", Binding::Attribute(String::from("missing")))
        .unwrap();
    let set = bricks(0.1);
    assert!(matches!(
        Realized::composite(&composite(&set, &unknown, 16)),
        Err(CompositeError::Contract(ContractError::UnknownAttribute(_)))
    ));

    // The body is inspectable, and its scopes are what evaluation hoists.
    let tint = program.outputs()[3].node;
    assert_eq!(program.scope(tint), Scope::Element);
    assert!(matches!(program.nodes()[tint.index()], Node::Mix { .. }));
    assert_ne!(
        program.fingerprint(),
        SurfaceBuilder::new("x").finish().fingerprint()
    );
}
