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
    let mut b = SurfaceBuilder::new("test.variant");
    let variant = b
        .input("variant", PortType::Scalar, Scope::Element)
        .unwrap();
    let one = b.constant(Value::Scalar(1.0));
    let (stone, flake) = (b.constant(Value::Id(0)), b.constant(Value::Id(1)));
    let material = b
        .add(Node::Select {
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
