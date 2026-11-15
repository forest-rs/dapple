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
use dapple_field::scoped::{ContractError, Node, Scope, ScopedBuilder, ScopedProgram};

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
fn program() -> ScopedProgram {
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

    let mut b = ScopedBuilder::new("test.glaze");
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
    let mut b = ScopedBuilder::new("bad");
    let local = b.input("local", PortType::Vector2, Scope::Sample).unwrap();
    let x = b
        .node(Node::Component {
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
        b.node(Node::Add(id, x)),
        Err(ContractError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        b.node(Node::Component { input: x, index: 0 }),
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
        ScopedBuilder::new("x").finish().fingerprint()
    );
}

/// The texels any element owns, as a mask.
fn owned_mask(realized: &Realized) -> dapple_raster::Raster {
    let labels = realized.owner_labels().unwrap();
    let dapple_raster::typed::Storage::U32(labels) = labels.storage() else {
        panic!("owner labels are identifiers")
    };
    dapple_raster::Raster::from_values(
        labels.width(),
        labels.height(),
        labels.origin(),
        labels.texel(),
        labels.edge(),
        labels
            .values()
            .iter()
            .map(|&l| if l > 0 { 1.0 } else { 0.0 })
            .collect(),
    )
    .unwrap()
}

/// Sets `mask` to `value` over the texels whose centers lie in `[lo, hi]`.
fn paint(mask: &mut dapple_raster::Raster, lo: Vec2, hi: Vec2, value: f32) {
    let size = mask.width();
    let mut values = mask.values().to_vec();
    for y in 0..size {
        for x in 0..size {
            let p = Vec2::new((f(x) + 0.5) / f(size), (f(y) + 0.5) / f(size));
            let q = p.rem_euclid(Vec2::ONE);
            let inside = |lo: f32, hi: f32, v: f32| {
                (lo..=hi).contains(&v)
                    || (lo..=hi).contains(&(v + 1.0))
                    || (lo..=hi).contains(&(v - 1.0))
            };
            if inside(lo.x, hi.x, q.x) && inside(lo.y, hi.y, q.y) {
                values[(y * size + x) as usize] = value;
            }
        }
    }
    *mask = dapple_raster::Raster::from_values(
        size,
        mask.height(),
        mask.origin(),
        mask.texel(),
        mask.edge(),
        values,
    )
    .unwrap();
}

// Slice 1b gate: a region round-trips through composite → reconstruct with
// a correspondence that names every split and merge.
#[test]
fn regions_round_trip_and_name_every_split_and_merge() {
    let inst = instance();
    let set = bricks(0.1);
    let realized = Realized::composite(&composite(&set, &inst, 64)).unwrap();
    let composited = RegionMap::from_composite(&realized).unwrap();
    assert_eq!(composited.regions().len(), set.len());
    for (region, &key) in composited.regions().iter().zip(set.keys()) {
        assert_eq!(region.key, RegionKey::of_element(key));
        assert_eq!(region.provenance, Provenance::Element(key));
        assert!(region.neighbors.is_empty(), "joints separate the bricks");
    }

    // Reconstruction is canonical: keys come from anchors, not elements.
    let mask = owned_mask(&realized);
    let canonical = RegionMap::reconstruct(&mask, 0.5, Connectivity::Four).unwrap();
    assert_eq!(canonical.regions().len(), set.len());
    assert!(
        canonical
            .regions()
            .iter()
            .all(|r| composited.region(r.key).is_none())
    );
    // Retaining the composite's identities round-trips every region.
    let (kept, report) = canonical.retaining(&composited, 0.25).unwrap();
    assert_eq!(kept.labels(), composited.labels());
    assert_eq!(report.matched.len(), set.len());
    assert!(report.matched.iter().all(|(a, b)| a == b));
    assert!(report.splits.is_empty() && report.merges.is_empty());

    // Cut one brick in two, and bridge two others across a head joint.
    let region_at = |p: Vec2| {
        let (x, y) = texel_at(p, 64);
        composited.key_at(i64::from(x), i64::from(y)).unwrap()
    };
    let cut = set.placement(3).center;
    let bridged = set.placement(9).center;
    let (cut_key, left_key) = (region_at(cut), region_at(bridged));
    let right_key = region_at(bridged + Vec2::new(0.25, 0.0));
    let mut edited = mask.clone();
    paint(
        &mut edited,
        cut + Vec2::new(-0.01, -0.1),
        cut + Vec2::new(0.01, 0.1),
        0.0,
    );
    paint(
        &mut edited,
        bridged + Vec2::new(0.05, -0.02),
        bridged + Vec2::new(0.2, 0.02),
        1.0,
    );
    let rebuilt = RegionMap::reconstruct(&edited, 0.5, Connectivity::Four).unwrap();
    let (rebuilt, report) = rebuilt.retaining(&composited, 0.25).unwrap();
    assert_eq!(rebuilt.regions().len(), set.len());
    // The split is named, and one child keeps the brick's identity.
    assert_eq!(report.splits.len(), 1, "{report:?}");
    let split = &report.splits[0];
    assert_eq!(split.from, cut_key);
    assert_eq!(split.into.len(), 2);
    assert!(split.into.contains(&cut_key));
    // The merge is named and keeps one parent's identity.
    let mut parents = vec![left_key, right_key];
    parents.sort_unstable();
    assert_eq!(report.merges.len(), 1, "{report:?}");
    assert_eq!(report.merges[0].from, parents);
    assert!(parents.contains(&report.merges[0].into));
    // Everything else is matched to itself.
    assert_eq!(report.matched.len(), set.len() - 3);
    assert!(report.matched.iter().all(|(a, b)| a == b));
    assert!(report.regroups.is_empty() && report.appeared.is_empty());
    assert!(report.vanished.is_empty());

    // Canonical: the same mask and the same retained state give the same
    // map, whatever edits led to the mask.
    let mut other_history = mask.clone();
    paint(
        &mut other_history,
        bridged + Vec2::new(0.05, -0.02),
        bridged + Vec2::new(0.2, 0.02),
        1.0,
    );
    paint(
        &mut other_history,
        cut + Vec2::new(-0.01, -0.1),
        cut + Vec2::new(0.01, 0.1),
        0.0,
    );
    let again = RegionMap::reconstruct(&other_history, 0.5, Connectivity::Four)
        .unwrap()
        .retaining(&composited, 0.25)
        .unwrap()
        .0;
    assert_eq!(again, rebuilt);
    // Without retained state the edited map is canonical: anchors alone.
    let fresh = RegionMap::reconstruct(&edited, 0.5, Connectivity::Four).unwrap();
    let plain = composited.correspondence(&fresh, 0.25).unwrap();
    assert_eq!((plain.splits.len(), plain.merges.len()), (1, 1));
}

#[test]
fn region_tables_measure_their_regions() {
    let inst = instance();
    let set = bricks(0.1);
    let realized = Realized::composite(&composite(&set, &inst, 64)).unwrap();
    let map = RegionMap::from_composite(&realized).unwrap();
    for (i, region) in map.regions().iter().enumerate() {
        let p = set.placement(i);
        let half = set.half_size(i);
        // Area and centroid of the square brick, to a texel.
        let texel = 1.0 / 64.0;
        assert!((region.area - 4.0 * half.x * half.y).abs() < 4.0 * half.x * texel);
        let d = (region.centroid - p.center.rem_euclid(Vec2::ONE)).abs();
        let d = d.min(Vec2::ONE - d);
        assert!(
            d.max_element() < texel,
            "{:?} vs {:?}",
            region.centroid,
            p.center
        );
    }
    // Insets grow inward from the edges: a brick's center is half its
    // size in.
    let inset = map.inset().unwrap();
    let (x, y) = texel_at(set.placement(0).center, 64);
    let depth = inset.at(i64::from(x), i64::from(y));
    assert!((depth - set.half_size(0).x).abs() < 2.0 / 64.0, "{depth}");
    assert_eq!(map.boundaries().at(i64::from(x), i64::from(y)), 0.0);
    // Statistics per region: the height output's range on each brick.
    let height = realized.output("height").unwrap().unwrap();
    let dapple_raster::typed::Storage::F32(height) = height.storage() else {
        panic!("height is scalar")
    };
    let stats = map.statistics(height).unwrap();
    assert_eq!(stats.len(), map.regions().len());
    assert!(stats.iter().all(|s| s.min <= s.mean && s.mean <= s.max));
    // A region across the period's seam stays one region, its centroid in
    // the period.
    let mut seam = vec![0.0; 64 * 64];
    for y in 20..24 {
        for x in (0..4).chain(60..64) {
            seam[y * 64 + x] = 1.0;
        }
    }
    let mask = dapple_raster::Raster::from_values(
        64,
        64,
        Vec2::ZERO,
        Vec2::splat(1.0 / 64.0),
        dapple_raster::Edge::Wrap,
        seam,
    )
    .unwrap();
    let seam = RegionMap::reconstruct(&mask, 0.5, Connectivity::Four).unwrap();
    assert_eq!(seam.regions().len(), 1);
    let c = seam.regions()[0].centroid;
    assert!(c.x < 1.0 / 64.0 || c.x > 63.0 / 64.0, "{c}");
    // Wide, not tall: oriented along x.
    let o = seam.regions()[0].orientation;
    assert!(!(0.01..=core::f32::consts::PI - 0.01).contains(&o), "{o}");
}

// Slice 1b gate: curve-driven element spacing is independent of
// realization resolution.
#[test]
fn curve_stitch_spacing_is_independent_of_resolution() {
    let seam = Curve::open(
        vec![
            Vec2::new(0.1, 0.2),
            Vec2::new(0.6, 0.35),
            Vec2::new(0.8, 0.8),
        ],
        vec![0.004; 3],
    )
    .unwrap();
    let net = CurveNetwork::new(domain(), vec![seam.clone()]);
    let spacing = 0.06;
    let stitches = net
        .stitches(
            LayoutId::named("test.seam"),
            spacing,
            Vec2::new(0.018, 0.006),
            schema(),
            |key, _, _| vec![Value::Scalar(key.unit(1)), Value::Scalar(0.5)],
        )
        .unwrap();
    let inst = instance();
    let maps: Vec<RegionMap> = [128_u32, 512]
        .iter()
        .map(|&size| {
            let realized = Realized::composite(&composite(&stitches, &inst, size)).unwrap();
            RegionMap::from_composite(&realized).unwrap()
        })
        .collect();
    // The same stitches at every resolution, each where its arc length
    // puts it, to within the resolution's texel.
    for (map, size) in maps.iter().zip([128_u32, 512]) {
        assert_eq!(map.regions().len(), stitches.len());
        let texel = 1.0 / f(size);
        for (i, &key) in stitches.keys().iter().enumerate() {
            let region = map.region(RegionKey::of_element(key)).unwrap();
            let expected = stitches.placement(i).center;
            assert!(
                (region.centroid - expected).length() < texel,
                "{size}: {:?} vs {expected}",
                region.centroid
            );
        }
    }
    // Consecutive stitches along the curve are `spacing` apart in arc
    // length, whatever the resolution: centroids agree across resolutions.
    for key in stitches.keys() {
        let key = RegionKey::of_element(*key);
        let (a, b) = (maps[0].region(key).unwrap(), maps[1].region(key).unwrap());
        assert!((a.centroid - b.centroid).length() < 1.0 / 128.0);
    }
    // The seam as a field: the stroke's area is its length times its width
    // at both resolutions.
    for size in [128_u32, 512] {
        let realization = dapple_raster::Realization::period(domain(), size, size).unwrap();
        let stroke = dapple_raster::realize(&net.field(CurveOutput::Stroke), realization).unwrap();
        let area: f32 = stroke.values().iter().sum::<f32>() / f(size * size);
        let expected = seam.length() * 0.004;
        assert!(
            (area - expected).abs() < 0.08 * expected,
            "{size}: {area} vs {expected}"
        );
    }
}

fn scatter(density: f32) -> dapple_field::Scatter {
    dapple_field::Scatter::new(
        domain(),
        dapple_field::Placement {
            frequency: 16.0,
            density,
            radius: [0.25, 0.5],
            rotate: true,
        },
        dapple_field::Stamp::Disk { softness: 0.0 },
        9,
    )
    .unwrap()
}

/// A program whose outputs are a sub-material chosen by the element's
/// variant (0 or 1) and full coverage.
fn variant_instance() -> ProgramInstance {
    let mut b = ScopedBuilder::new("test.variant");
    let variant = b
        .input("variant", PortType::Scalar, Scope::Element)
        .unwrap();
    let one = b.constant(Value::Scalar(1.0));
    let (stone, flake) = (b.constant(Value::Id(0)), b.constant(Value::Id(1)));
    let material = b
        .node(Node::Select {
            condition: variant,
            a: flake,
            b: stone,
        })
        .unwrap();
    b.output("material", PortType::Id, Scope::Element, material)
        .unwrap();
    b.output("cover", PortType::Mask, Scope::Element, one)
        .unwrap();
    ProgramInstance::new(
        InstanceId::named("test.variant"),
        Arc::new(b.finish()),
        vec![Binding::Variant],
    )
    .unwrap()
}

#[test]
fn scattered_elements_follow_the_field_and_pick_variants() {
    let disks = ScatterLayout {
        layout: LayoutId::named("test.pebbles"),
        scatter: scatter(0.7),
        variants: vec![ScatterVariant {
            weight: 1.0,
            outline: Outline::Ellipse,
            aspect: 1.0,
        }],
    };
    let set = disks.elements(Vec::new(), |_, _| Vec::new()).unwrap();
    assert!(set.len() > 100, "{}", set.len());
    // Composited as disks, the elements cover what the field's disks cover.
    let inst = variant_instance();
    let background = [Value::Id(9), Value::Scalar(0.0)];
    let size = 256;
    let realized = Realized::composite(&Composite {
        set: &set,
        instance: &inst,
        background: &background,
        domain: domain(),
        width: size,
        height: size,
        tile_size: 32,
    })
    .unwrap();
    let mut b = ProgramBuilder::new();
    let field = b
        .add(Op::Scatter {
            domain: domain(),
            placement: dapple_field::Placement {
                frequency: 16.0,
                density: 0.7,
                radius: [0.25, 0.5],
                rotate: true,
            },
            stamp: dapple_field::Stamp::Disk { softness: 0.0 },
            seed: 9,
            output: dapple_field::ScatterOutput::Coverage,
        })
        .unwrap();
    let field = b.finish(field).unwrap();
    let realization = dapple_raster::Realization::period(domain(), size, size).unwrap();
    let coverage = dapple_raster::realize(&field, realization).unwrap();
    let cover = realized.output("cover").unwrap().unwrap();
    let dapple_raster::typed::Storage::F32(cover) = cover.storage() else {
        panic!("cover is a mask")
    };
    // The two antialias their edges differently (the field's disk edge is
    // a smoothstep over a footprint inside the radius, the composite's a
    // box estimate summed over overlapping elements), so compare texels
    // both are sure of: none is inside one and outside the other.
    let sure = |v: f32| v <= 0.001 || v >= 0.999;
    let (mut compared, mut disagree) = (0_u32, 0_u32);
    let mut bad = Vec::new();
    for (&a, &b) in cover.values().iter().zip(coverage.values()) {
        if sure(a) && sure(b) {
            compared += 1;
            disagree += u32::from((a >= 0.5) != (b >= 0.5));
            if (a >= 0.5) != (b >= 0.5) {
                bad.push((a, b));
            }
        }
    }
    assert!(compared > size * size * 3 / 4, "{compared}");
    assert_eq!(disagree, 0, "of {compared} texels: {bad:?}");

    // Two kinds, drawn by weight; the program reads each element's.
    let mixed = ScatterLayout {
        variants: vec![
            ScatterVariant {
                weight: 3.0,
                outline: Outline::Ellipse,
                aspect: 1.0,
            },
            ScatterVariant {
                weight: 1.0,
                outline: Outline::Rectangle,
                aspect: 0.4,
            },
        ],
        ..disks.clone()
    };
    let set = mixed.elements(Vec::new(), |_, _| Vec::new()).unwrap();
    let flakes = (0..set.len()).filter(|&i| set.variant(i) == 1).count();
    assert!(flakes > set.len() / 8 && flakes < set.len() / 2, "{flakes}");
    for i in 0..set.len() {
        let expected = if set.variant(i) == 1 {
            (Outline::Rectangle, 0.4)
        } else {
            (Outline::Ellipse, 1.0)
        };
        let h = set.half_size(i);
        assert_eq!(set.outline(i), expected.0);
        assert!((h.y / h.x - expected.1).abs() < 1e-5);
    }
    let realized = Realized::composite(&Composite {
        set: &set,
        instance: &inst,
        background: &background,
        domain: domain(),
        width: size,
        height: size,
        tile_size: 32,
    })
    .unwrap();
    for i in 0..set.len() {
        let (x, y) = texel_at(set.placement(i).center, size);
        if realized.owner(x, y) == Some(set.keys()[i]) {
            assert_eq!(realized.value(0, x, y), Value::Id(set.variant(i)));
        }
    }
    // A density edit keeps the keys of the elements that remain.
    let sparse = ScatterLayout {
        scatter: scatter(0.4),
        ..disks
    };
    let fewer = sparse.elements(Vec::new(), |_, _| Vec::new()).unwrap();
    let all = disks_set();
    assert!(fewer.len() < all.len());
    assert!(fewer.keys().iter().all(|k| all.index_of(*k).is_some()));
}

fn disks_set() -> ElementSet {
    ScatterLayout {
        layout: LayoutId::named("test.pebbles"),
        scatter: scatter(0.7),
        variants: vec![ScatterVariant {
            weight: 1.0,
            outline: Outline::Ellipse,
            aspect: 1.0,
        }],
    }
    .elements(Vec::new(), |_, _| Vec::new())
    .unwrap()
}

#[test]
fn region_scope_runs_once_per_region_and_keeps_its_identity() {
    use dapple_raster::{Edge, Raster};
    // Two blobs on an 8 × 4 grid.
    let mut values = vec![0.0_f32; 32];
    for (x, y) in [(1, 1), (2, 1), (1, 2), (5, 1), (6, 1), (6, 2)] {
        values[y * 8 + x] = 1.0;
    }
    let mask =
        Raster::from_values(8, 4, Vec2::ZERO, Vec2::splat(0.125), Edge::Clamp, values).unwrap();
    let map = RegionMap::reconstruct(&mask, 0.5, Connectivity::Four).unwrap();
    assert_eq!(map.regions().len(), 2, "two regions");

    let mut b = ScopedBuilder::new("test.region");
    let tone = b.input("tone", PortType::Scalar, Scope::Region).unwrap();
    let area = b.input("area", PortType::Scalar, Scope::Region).unwrap();
    let at = b.input("at", PortType::Vector2, Scope::Sample).unwrap();
    let x = b.component(at, 0).unwrap();
    let shade = b.add(tone, x).unwrap();
    b.output("tone", PortType::Scalar, Scope::Region, tone)
        .unwrap();
    b.output("area", PortType::Scalar, Scope::Region, area)
        .unwrap();
    b.output("shade", PortType::Scalar, Scope::Sample, shade)
        .unwrap();
    let program = Arc::new(b.finish());
    let instance = ProgramInstance::new(
        InstanceId::named("test.region"),
        Arc::clone(&program),
        vec![
            Binding::RegionRandom(7),
            Binding::RegionArea,
            Binding::Position,
        ],
    )
    .unwrap();
    let background = [Value::Scalar(-1.0), Value::Scalar(0.0), Value::Scalar(-1.0)];
    let out = map.evaluate(&instance, &background).unwrap();
    for region in map.regions() {
        let expected = ElementKey::from_word(region.key.word()).unit(7);
        for y in 0..4 {
            for x in 0..8 {
                if map.key_at(x, y) == Some(region.key) {
                    assert_eq!(out[0].value_at(x, y), Value::Scalar(expected), "tone");
                    assert_eq!(out[1].value_at(x, y), Value::Scalar(region.area), "area");
                }
            }
        }
    }
    assert_eq!(out[0].value_at(0, 0), Value::Scalar(-1.0), "background");

    assert!(
        matches!(
            ProgramInstance::new(
                InstanceId::named("test.region"),
                Arc::clone(&program),
                vec![
                    Binding::ElementRandom(7),
                    Binding::RegionArea,
                    Binding::Position
                ],
            ),
            Err(ContractError::ScopeViolation { .. })
        ),
        "an element value cannot stand for a region input"
    );
    let local = ProgramInstance::new(
        InstanceId::named("test.region"),
        program,
        vec![
            Binding::RegionRandom(7),
            Binding::RegionArea,
            Binding::LocalPosition,
        ],
    )
    .unwrap();
    assert!(
        matches!(
            map.evaluate(&local, &background),
            Err(RegionError::Contract(ContractError::UnsupportedBinding(_)))
        ),
        "regions cannot supply element bindings"
    );
}

// The batched compositor against the per-sample reference.

/// A second program: a per-sample input fed by an element attribute, a
/// per-element noise sample (at a point footprint), and a per-sample one.
fn mixed_instance() -> ProgramInstance {
    let mut noise = ProgramBuilder::new();
    let n = noise
        .add(Op::Noise {
            basis: Basis::Gradient,
            domain: Domain::Plane,
            frequency: [9.0, 9.0],
            seed: 3,
        })
        .unwrap();
    let noise = noise.finish_value(n).unwrap();
    let mut disk = ProgramBuilder::new();
    let d = disk
        .add(Op::Disk {
            domain: Domain::Plane,
            center: [0.0, 0.0],
            radius: 0.1,
            softness: 0.05,
        })
        .unwrap();
    let disk = disk.finish_value(d).unwrap();
    let mut b = ScopedBuilder::new("test.mixed");
    let tone = b.input("tone", PortType::Scalar, Scope::Sample).unwrap();
    let half = b.input("half", PortType::Vector2, Scope::Element).unwrap();
    let local = b.input("local", PortType::Vector2, Scope::Sample).unwrap();
    let variant = b
        .input("variant", PortType::Scalar, Scope::Element)
        .unwrap();
    let k = b.input("k", PortType::Scalar, Scope::Material).unwrap();
    let grain = b.resource("grain", noise);
    let rim = b.resource("rim", disk);
    let band = b
        .node(Node::Sample {
            resource: rim,
            at: half,
        })
        .unwrap();
    let per_element = b
        .node(Node::Sample {
            resource: grain,
            at: half,
        })
        .unwrap();
    let scaled = b.node(Node::Mul(per_element, k)).unwrap();
    let scaled = b.node(Node::Add(scaled, band)).unwrap();
    let at = b.node(Node::Add(local, half)).unwrap();
    let per_sample = b
        .node(Node::Sample {
            resource: grain,
            at,
        })
        .unwrap();
    let at_rim = b.node(Node::Add(local, band)).unwrap();
    let rim_sample = b
        .node(Node::Sample {
            resource: rim,
            at: at_rim,
        })
        .unwrap();
    let mixed = b.node(Node::Add(per_sample, scaled)).unwrap();
    let mixed = b.node(Node::Add(mixed, rim_sample)).unwrap();
    let blend = b.node(Node::Mul(mixed, tone)).unwrap();
    let blend = b.node(Node::Add(blend, variant)).unwrap();
    let id = b.constant(Value::Id(4));
    let other = b.constant(Value::Id(5));
    let pick = b
        .node(Node::Select {
            condition: per_sample,
            a: id,
            b: other,
        })
        .unwrap();
    b.output("blend", PortType::Scalar, Scope::Sample, blend)
        .unwrap();
    b.output("pick", PortType::Id, Scope::Sample, pick).unwrap();
    b.output("scaled", PortType::Scalar, Scope::Element, scaled)
        .unwrap();
    ProgramInstance::new(
        InstanceId::named("test.mixed"),
        Arc::new(b.finish()),
        vec![
            Binding::Attribute(String::from("tone")),
            Binding::HalfSize,
            Binding::LocalPosition,
            Binding::Variant,
            Binding::Constant(Value::Scalar(0.75)),
        ],
    )
    .unwrap()
}

const MIXED_BACKGROUND: [Value; 3] = [Value::Scalar(0.2), Value::Id(9), Value::Scalar(0.0)];

/// A deterministic stream of numbers in `[0, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        #[expect(clippy::cast_precision_loss, reason = "24 bits")]
        let v = (self.0 >> 40) as f32;
        v / 16_777_216.0
    }
}

impl Lcg {
    /// A number in `0..n`.
    fn below(&mut self, n: u32) -> u32 {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the product is below n"
        )]
        let v = (self.next() * f(n)) as u32;
        v.min(n - 1)
    }
}

/// Overlapping, wrapping, tiny, huge, rotated and elliptic elements.
fn varied(seed: u64, count: u32) -> ElementSet {
    let mut rng = Lcg(seed);
    let layout = LayoutId::named("test.varied");
    let elements = (0..count)
        .map(|n| {
            let (x, y) = (rng.next() * 1.6 - 0.3, rng.next() * 1.6 - 0.3);
            let half = match n % 7 {
                0 => Vec2::new(1.0e-30, 2.0e-30),
                1 => Vec2::new(2.5, 3.0),
                2 => Vec2::new(0.4, 0.05),
                _ => Vec2::new(0.02 + 0.2 * rng.next(), 0.02 + 0.2 * rng.next()),
            };
            Element {
                key: ElementKey::new(layout, Anchor([i32::try_from(n).unwrap(), 0]), 0),
                placement: Placement {
                    center: Vec2::new(x, y),
                    rotation: rng.next() * 6.0,
                },
                half_size: half,
                outline: if rng.next() < 0.4 {
                    Outline::Ellipse
                } else {
                    Outline::Rectangle
                },
                variant: rng.below(3),
                attributes: vec![Value::Scalar(rng.next()), Value::Scalar(rng.next() - 0.5)],
            }
        })
        .collect();
    ElementSet::new(schema(), elements).unwrap()
}

/// Every output's and owner's exact bits, texel by texel.
fn bits(realized: &Realized, outputs: usize, size: [u32; 2]) -> Vec<u64> {
    let mut words = Vec::new();
    for y in 0..size[1] {
        for x in 0..size[0] {
            for o in 0..outputs {
                set::value_words(realized.value(o, x, y), &mut words);
            }
            words.push(realized.owner(x, y).map_or(0, |k| k.word()));
        }
    }
    words
}

struct Case<'a> {
    instance: &'a ProgramInstance,
    background: &'a [Value],
    size: [u32; 2],
    tile: u32,
}

impl<'a> Case<'a> {
    fn composite<'b>(&'b self, set: &'b ElementSet) -> Composite<'b>
    where
        'a: 'b,
    {
        Composite {
            set,
            instance: self.instance,
            background: self.background,
            domain: domain(),
            width: self.size[0],
            height: self.size[1],
            tile_size: self.tile,
        }
    }

    /// The batched path, in bands of `band`, equals the per-sample one.
    fn agree(&self, set: &ElementSet, band: usize) -> Realized {
        let c = self.composite(set);
        let reference = Realized::composite_reference(&c).unwrap();
        let batched = Realized::composite_banded(&c, band).unwrap();
        let outputs = self.background.len();
        assert_eq!(
            bits(&batched, outputs, self.size),
            bits(&reference, outputs, self.size)
        );
        assert_eq!(batched.digest(), reference.digest());
        assert_eq!(
            Realized::composite(&c).unwrap().digest(),
            reference.digest()
        );
        batched
    }
}

/// Whether to run the large sweeps of the batched-compositing tests; set
/// `DAPPLE_ELEMENTS_SWEEP` to any value when building the tests to do.
/// The default is the smallest configuration found to catch every
/// mutation tried (stale element hoists, lane order, band bookkeeping,
/// footprints, summation order).
fn sweep() -> bool {
    option_env!("DAPPLE_ELEMENTS_SWEEP").is_some()
}

fn cases() -> (ProgramInstance, ProgramInstance) {
    (instance(), mixed_instance())
}

#[test]
fn batched_compositing_is_bit_identical_to_the_per_sample_path() {
    let (glaze, mixed) = cases();
    let setups = [
        Case {
            instance: &glaze,
            background: &BACKGROUND,
            size: if sweep() { [41, 29] } else { [21, 15] },
            tile: 8,
        },
        Case {
            instance: &mixed,
            background: &MIXED_BACKGROUND,
            size: if sweep() { [32, 32] } else { [16, 16] },
            tile: 7,
        },
    ];
    for case in &setups {
        let (sets, bands): (&[(u64, u32)], &[usize]) = if sweep() {
            (&[(1, 1), (2, 9), (3, 40), (4, 150)], &[1, 5, 64, 1 << 17])
        } else {
            (&[(2, 9), (3, 24)], &[1, 40, 1 << 17])
        };
        for &(seed, count) in sets {
            let set = varied(seed, count);
            // One band, many bands, and bands of a single contribution.
            for &band in bands {
                case.agree(&set, band);
            }
        }
    }
    // Whole running bonds, scatter layouts with variants, and no elements.
    let bond = bricks(0.03);
    for case in &setups {
        case.agree(&bond, 100);
        case.agree(&ElementSet::new(schema(), Vec::new()).unwrap(), 100);
    }
}

#[test]
fn batched_incremental_updates_equal_the_reference_after_edits() {
    let (glaze, mixed) = cases();
    for (instance, background) in [(&glaze, &BACKGROUND[..]), (&mixed, &MIXED_BACKGROUND[..])] {
        let case = Case {
            instance,
            background,
            size: if sweep() { [48, 40] } else { [24, 16] },
            tile: 8,
        };
        let outputs = background.len();
        let mut set = varied(11, 60);
        let steps = if sweep() { 12 } else { 5 };
        let mut batched = Realized::composite_banded(&case.composite(&set), 50).unwrap();
        let mut reference = Realized::composite_reference(&case.composite(&set)).unwrap();
        let keys: Vec<ElementKey> = set.keys().to_vec();
        let mut rng = Lcg(77);
        for step in 0..steps {
            match step % 4 {
                0 | 1 => {
                    let key = keys[rng.below(59) as usize];
                    let mut p = set.placement(set.index_of(key).unwrap());
                    p.center += Vec2::new(rng.next() - 0.5, rng.next() - 0.5);
                    p.rotation += rng.next();
                    set.set_placement(key, p).unwrap();
                }
                2 => {
                    let key = keys[rng.below(59) as usize];
                    set.set_attribute(key, "tone", Value::Scalar(rng.next()))
                        .unwrap();
                }
                _ => {
                    let drop = keys[rng.below(59) as usize];
                    set = set.filter(|e| e.key != drop);
                }
            }
            let c = case.composite(&set);
            let a = batched.update_banded(&c, 30).unwrap();
            let b = reference.update_reference(&c).unwrap();
            assert_eq!(a, b);
            assert_eq!(
                bits(&batched, outputs, case.size),
                bits(&reference, outputs, case.size)
            );
            let clean = Realized::composite_reference(&c).unwrap();
            assert_eq!(batched.digest(), clean.digest());
        }
        // Back to the full set, then a new program rebuilds whole.
        let full = varied(11, 60);
        let a = batched.update(&case.composite(&full)).unwrap();
        let b = reference.update_reference(&case.composite(&full)).unwrap();
        assert_eq!(a, b);
        assert_eq!(batched.digest(), reference.digest());
    }
}
