// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use dapple_field::program::Op;
use dapple_field::{Basis, Domain, FractalParams, Value};

use super::*;

fn domain() -> Domain {
    Domain::periodic(1, 1).unwrap()
}

/// noise, fbm → mix → height → realize → normals, and a blur of the height.
struct Fixture {
    graph: MaterialGraph,
    noise: NodeId,
    fbm: NodeId,
    height: NodeId,
    map: NodeId,
    normals: NodeId,
    soft: NodeId,
}

fn fixture() -> Fixture {
    let d = domain();
    let mut g = MaterialGraph::new();
    let noise = g
        .field(
            "noise",
            Op::Noise {
                basis: Basis::Gradient,
                domain: d,
                frequency: [8.0, 8.0],
                seed: 1,
            },
            &[],
        )
        .unwrap();
    let fbm = g
        .field(
            "fbm",
            Op::Fractal {
                basis: Basis::Value,
                domain: d,
                frequency: [4.0, 4.0],
                seed: 2,
                params: FractalParams::default(),
            },
            &[],
        )
        .unwrap();
    let height = g
        .field(
            "height",
            Op::Add {
                a: operand(0),
                b: operand(1),
            },
            &[noise, fbm],
        )
        .unwrap();
    let map = g.realize("map", height, 32, 32).unwrap();
    let normals = g
        .raster(
            "normals",
            RasterParams::HeightToNormal(HeightToNormal { scale: 0.05 }),
            map,
        )
        .unwrap();
    let soft = g
        .raster(
            "soft",
            RasterParams::Blur(GaussianBlur { sigma: 0.02 }),
            map,
        )
        .unwrap();
    Fixture {
        graph: g,
        noise,
        fbm,
        height,
        map,
        normals,
        soft,
    }
}

#[test]
fn nodes_compose_programs_and_rasters() {
    let mut f = fixture();
    assert_eq!(f.graph.run().unwrap().executed_nodes, 6);
    let height = f.graph.field_value(f.height).unwrap();
    let noise = f.graph.field_value(f.noise).unwrap();
    let fbm = f.graph.field_value(f.fbm).unwrap();
    let texels = |program: &ValueProgram| {
        let field = program.channel(0).unwrap();
        realize(&field, Realization::period(domain(), 16, 16).unwrap()).unwrap()
    };
    let (h, n, b) = (texels(height), texels(noise), texels(fbm));
    for ((h, n), b) in h.values().iter().zip(n.values()).zip(b.values()) {
        assert_eq!(h.to_bits(), (n + b).to_bits());
    }
    let map = f.graph.raster_value(f.map).unwrap();
    let RasterData::Scalar(raster) = &map.data else {
        panic!("realized fields are scalar");
    };
    let direct = realize(
        &height.channel(0).unwrap(),
        Realization::period(domain(), 32, 32).unwrap(),
    )
    .unwrap();
    assert_eq!(raster, &direct);
    assert!(matches!(
        f.graph.raster_value(f.normals).unwrap().data,
        RasterData::Vector3(_)
    ));
    // Nothing changed: nothing runs.
    assert_eq!(f.graph.run().unwrap().executed_nodes, 0);
}

#[test]
fn edits_rerun_exactly_their_dependents() {
    let mut f = fixture();
    f.graph.run().unwrap();
    let before = f.graph.raster_value(f.soft).unwrap().fingerprint;

    // A new fbm seed re-runs fbm, height, map and both rasters, not noise.
    f.graph
        .set_field_op(
            f.fbm,
            Op::Fractal {
                basis: Basis::Value,
                domain: domain(),
                frequency: [4.0, 4.0],
                seed: 3,
                params: FractalParams::default(),
            },
        )
        .unwrap();
    assert_eq!(f.graph.run().unwrap().executed_nodes, 5);
    assert_eq!(f.graph.run_count(f.noise), Some(1));
    assert_eq!(f.graph.run_count(f.fbm), Some(2));
    assert_ne!(f.graph.raster_value(f.soft).unwrap().fingerprint, before);

    // A new blur re-runs only the blur.
    f.graph
        .set_raster_params(f.soft, RasterParams::Blur(GaussianBlur { sigma: 0.04 }))
        .unwrap();
    assert_eq!(f.graph.run().unwrap().executed_nodes, 1);
    assert_eq!(f.graph.run_count(f.normals), Some(2));

    // A new resolution re-runs the map and both rasters.
    f.graph.set_resolution(f.map, 16, 16).unwrap();
    assert_eq!(f.graph.run().unwrap().executed_nodes, 3);
    let RasterData::Scalar(raster) = &f.graph.raster_value(f.soft).unwrap().data else {
        panic!("blur keeps one channel");
    };
    assert_eq!(raster.width(), 16);
}

#[test]
fn misuse_is_reported() {
    let mut f = fixture();
    assert!(matches!(
        f.graph
            .field("noise", Op::Abs { input: operand(0) }, &[f.fbm]),
        Err(MaterialError::DuplicateLabel(_))
    ));
    // Setting a raster node's resolution is a kind mismatch.
    assert!(matches!(
        f.graph.set_resolution(f.soft, 8, 8),
        Err(MaterialError::UnknownNode)
    ));
    // An operand without an input is refused when the node is added.
    assert!(matches!(
        f.graph
            .field("bad", Op::Abs { input: operand(3) }, &[f.noise]),
        Err(MaterialError::Refused {
            error: NodeError::MissingOperand { index: 3 },
            ..
        })
    ));
    // So is a node reading the wrong kind of value: realizing a raster.
    assert!(matches!(
        f.graph.realize("bad", f.map, 8, 8),
        Err(MaterialError::Refused {
            error: NodeError::WrongValue { expected: "field" },
            ..
        })
    ));
    // Refused nodes are not added: the label stays free.
    f.graph
        .field("bad", Op::Abs { input: operand(0) }, &[f.noise])
        .unwrap();
    f.graph.run().unwrap();
}

/// noise max disk → realize (128² in 16-texel tiles) → blur, normals, and a
/// distance transform.
struct Stamped {
    graph: MaterialGraph,
    disk: NodeId,
    noise: NodeId,
    map: NodeId,
    soft: NodeId,
    normals: NodeId,
    distance: NodeId,
}

fn disk_op(x: f32) -> Op {
    Op::Disk {
        domain: domain(),
        center: [x, 0.3],
        radius: 0.05,
        softness: 0.01,
    }
}

fn noise_op(seed: u64) -> Op {
    Op::Noise {
        basis: Basis::Gradient,
        domain: domain(),
        frequency: [8.0, 8.0],
        seed,
    }
}

fn stamped(x: f32, seed: u64) -> Stamped {
    let mut g = MaterialGraph::with_tile_size(16);
    let noise = g.field("noise", noise_op(seed), &[]).unwrap();
    let disk = g.field("disk", disk_op(x), &[]).unwrap();
    let height = g
        .field(
            "height",
            Op::Max {
                a: operand(0),
                b: operand(1),
            },
            &[noise, disk],
        )
        .unwrap();
    let map = g.realize("map", height, 128, 128).unwrap();
    let soft = g
        .raster(
            "soft",
            RasterParams::Blur(GaussianBlur { sigma: 0.01 }),
            map,
        )
        .unwrap();
    let normals = g
        .raster(
            "normals",
            RasterParams::HeightToNormal(HeightToNormal { scale: 0.05 }),
            soft,
        )
        .unwrap();
    let distance = g
        .raster(
            "distance",
            RasterParams::DistanceTransform(DistanceTransform { threshold: 0.9 }),
            map,
        )
        .unwrap();
    Stamped {
        graph: g,
        disk,
        noise,
        map,
        soft,
        normals,
        distance,
    }
}

fn digests(s: &Stamped) -> [u64; 4] {
    let digest = |node| match &s.graph.raster_value(node).unwrap().data {
        RasterData::Scalar(r) => r.digest(),
        RasterData::Vector3(r) => r.digest(),
        RasterData::Typed(r) => r.digest(),
    };
    [
        digest(s.map),
        digest(s.soft),
        digest(s.normals),
        digest(s.distance),
    ]
}

#[test]
fn moving_a_disk_recomputes_only_nearby_tiles() {
    let mut s = stamped(0.2, 1);
    s.graph.run().unwrap();
    let first = s.graph.tile_report();
    // Four rasters of 64 tiles, all computed once.
    assert_eq!(first.tiles_recomputed, 4 * 64);
    assert_eq!(first.whole_recomputes, 4);

    s.graph.set_field_op(s.disk, disk_op(0.25)).unwrap();
    s.graph.run().unwrap();
    let edit = s.graph.tile_report();
    assert_eq!(edit.unbounded_changes, 0);
    // The distance transform is global, so it alone recomputes whole.
    assert_eq!(edit.whole_recomputes, 1);
    let local = edit.tiles_recomputed - 64;
    assert!(local > 0 && local < 3 * 64 / 4, "{edit:?}");
    assert!(edit.tiles_reused >= 3 * 64 - local, "{edit:?}");

    // Bit-identical to computing the final graph from scratch.
    let mut fresh = stamped(0.25, 1);
    fresh.graph.run().unwrap();
    assert_eq!(digests(&s), digests(&fresh));
}

#[test]
fn unbounded_edits_recompute_whole_and_say_so() {
    let mut s = stamped(0.2, 1);
    s.graph.run().unwrap();
    s.graph.set_field_op(s.noise, noise_op(2)).unwrap();
    s.graph.run().unwrap();
    let edit = s.graph.tile_report();
    assert_eq!(edit.unbounded_changes, 1);
    assert_eq!(edit.tiles_recomputed, 4 * 64);
    let mut fresh = stamped(0.2, 2);
    fresh.graph.run().unwrap();
    assert_eq!(digests(&s), digests(&fresh));
}

#[test]
fn unchanged_tiles_stop_propagating() {
    let mut s = stamped(0.2, 1);
    s.graph.run().unwrap();
    // A blur change recomputes the blur and the normals of changed tiles,
    // not the map or the distance transform.
    s.graph
        .set_raster_params(s.soft, RasterParams::Blur(GaussianBlur { sigma: 0.012 }))
        .unwrap();
    s.graph.run().unwrap();
    let edit = s.graph.tile_report();
    assert_eq!(edit.tiles_recomputed, 2 * 64, "{edit:?}");
    assert_eq!(s.graph.run_count(s.map), Some(1));
    assert_eq!(s.graph.run_count(s.distance), Some(1));

    // Moving the disk back and forth by nothing changes no bits.
    s.graph.set_field_op(s.disk, disk_op(0.2)).unwrap();
    s.graph.run().unwrap();
    assert_eq!(s.graph.tile_report().tiles_recomputed, 0);
}

#[test]
fn new_resolutions_rebuild_the_tiles() {
    let mut s = stamped(0.2, 1);
    s.graph.run().unwrap();
    s.graph.set_resolution(s.map, 96, 64).unwrap();
    s.graph.run().unwrap();
    let edit = s.graph.tile_report();
    assert_eq!(edit.whole_recomputes, 4);
    assert_eq!(edit.tiles_recomputed, 4 * 6 * 4);
    s.graph.set_field_op(s.disk, disk_op(0.21)).unwrap();
    s.graph.run().unwrap();
    assert!(s.graph.tile_report().tiles_recomputed < 4 * 24);
}

fn bark_recipe() -> Recipe {
    let node = |label: &str, step: Step| RecipeNode {
        label: label.into(),
        step,
    };
    Recipe {
        version: RECIPE_VERSION,
        nodes: vec![
            node(
                "noise",
                Step::Field {
                    op: noise_op(4),
                    inputs: vec![],
                },
            ),
            node(
                "disk",
                Step::Field {
                    op: disk_op(0.4),
                    inputs: vec![],
                },
            ),
            node(
                "height",
                Step::Field {
                    op: Op::Max {
                        a: operand(1),
                        b: operand(0),
                    },
                    inputs: vec!["noise".into(), "disk".into()],
                },
            ),
            node(
                "map",
                Step::Realize {
                    input: "height".into(),
                    width: 64,
                    height: 64,
                },
            ),
            node(
                "normal",
                Step::Raster {
                    input: "map".into(),
                    params: RasterParams::HeightToNormal(HeightToNormal { scale: 0.05 }),
                },
            ),
        ],
        outputs: vec![RecipeOutput {
            role: "normal".into(),
            channels: vec!["normal".into()],
        }],
    }
}

#[test]
fn recipes_predict_the_fingerprints_their_graphs_produce() {
    let recipe = bark_recipe();
    let predicted = recipe.fingerprints().unwrap();
    let (mut graph, ids) = recipe.build(16).unwrap();
    graph.run().unwrap();
    for (label, id) in &ids {
        let actual = match graph.value(*id).unwrap() {
            GraphValue::Field(field) => NodeFingerprint::Field(field.program.fingerprint()),
            GraphValue::Raster(raster) => NodeFingerprint::Raster(raster.fingerprint),
            GraphValue::Params(_) => unreachable!(),
        };
        assert_eq!(predicted[label], actual, "{label}");
    }
}

#[test]
fn graphs_export_recipes_that_rebuild_them() {
    let recipe = bark_recipe();
    let (mut graph, ids) = recipe.build(16).unwrap();
    graph.run().unwrap();
    graph.set_field_op(ids["disk"], disk_op(0.45)).unwrap();
    graph.run().unwrap();

    let mut exported = graph.recipe();
    assert_eq!(exported.nodes.len(), recipe.nodes.len());
    exported.outputs = recipe.outputs.clone();
    let (mut rebuilt, rebuilt_ids) = exported.build(8).unwrap();
    rebuilt.run().unwrap();
    let digest = |g: &MaterialGraph, id| match &g.raster_value(id).unwrap().data {
        RasterData::Vector3(r) => r.digest(),
        RasterData::Typed(r) => r.digest(),
        RasterData::Scalar(r) => r.digest(),
    };
    assert_eq!(
        digest(&graph, ids["normal"]),
        digest(&rebuilt, rebuilt_ids["normal"])
    );
    assert_ne!(
        exported.fingerprint().unwrap(),
        recipe.fingerprint().unwrap()
    );

    // Relabeling nodes keeps an output-based fingerprint.
    let mut relabeled = recipe.clone();
    for node in &mut relabeled.nodes {
        node.label.insert_str(0, "x.");
        match &mut node.step {
            Step::Field { inputs, .. } | Step::Sample { inputs, .. } => {
                for i in inputs {
                    i.insert_str(0, "x.");
                }
            }
            Step::Realize { input, .. }
            | Step::Raster { input, .. }
            | Step::Mip { input, .. }
            | Step::Reduce { input, .. }
            | Step::Normals { input, .. } => {
                input.insert_str(0, "x.");
            }
        }
    }
    relabeled.outputs[0].channels[0].insert_str(0, "x.");
    assert_eq!(
        relabeled.fingerprint().unwrap(),
        recipe.fingerprint().unwrap()
    );
}

#[test]
fn malformed_recipes_are_refused() {
    let recipe = bark_recipe();
    let with = |f: &dyn Fn(&mut Recipe)| {
        let mut r = recipe.clone();
        f(&mut r);
        r.fingerprint().unwrap_err()
    };
    assert!(matches!(
        with(&|r| r.version = 1),
        RecipeError::Version { found: 1 }
    ));
    assert!(matches!(
        with(&|r| r.nodes[1].label = "noise".into()),
        RecipeError::DuplicateLabel(_)
    ));
    assert!(matches!(
        with(&|r| r.outputs[0].channels[0] = "nope".into()),
        RecipeError::UnknownLabel(_)
    ));
    assert!(matches!(
        with(&|r| r.outputs[0].channels[0] = "height".into()),
        RecipeError::WrongInput { .. }
    ));
    assert!(matches!(
        with(&|r| r.nodes.swap(2, 3)),
        RecipeError::UnknownLabel(_)
    ));
}

#[test]
fn normals_nodes_recompute_locally_and_match_fresh_graphs() {
    let build = |x: f32| {
        let mut g = MaterialGraph::with_tile_size(16);
        let noise = g.field("noise", noise_op(1), &[]).unwrap();
        let disk = g.field("disk", disk_op(x), &[]).unwrap();
        let height = g
            .field(
                "height",
                Op::Max {
                    a: operand(0),
                    b: operand(1),
                },
                &[noise, disk],
            )
            .unwrap();
        let normals = g.normals("normals", height, (128, 128), 0.05).unwrap();
        (g, disk, normals)
    };
    let (mut g, disk, normals) = build(0.2);
    g.run().unwrap();
    assert!(matches!(
        g.raster_value(normals).unwrap().data,
        RasterData::Vector3(_)
    ));
    g.set_field_op(disk, disk_op(0.25)).unwrap();
    g.run().unwrap();
    let edit = g.tile_report();
    assert!(
        edit.tiles_recomputed > 0 && edit.tiles_recomputed < 64 / 2,
        "{edit:?}"
    );
    assert_eq!(edit.unbounded_changes, 0);
    let (mut fresh, _, fresh_normals) = build(0.25);
    fresh.run().unwrap();
    let digest = |g: &MaterialGraph, id| match &g.raster_value(id).unwrap().data {
        RasterData::Vector3(r) => r.digest(),
        RasterData::Typed(r) => r.digest(),
        RasterData::Scalar(r) => r.digest(),
    };
    assert_eq!(digest(&g, normals), digest(&fresh, fresh_normals));
    // The recipe round trip keeps normals nodes.
    let recipe = g.recipe();
    assert!(matches!(recipe.nodes[3].step, Step::Normals { .. }));
    assert_eq!(
        recipe.fingerprints().unwrap()["normals"],
        NodeFingerprint::Raster(g.raster_value(normals).unwrap().fingerprint)
    );
}

#[test]
fn tile_budgets_spread_work_and_converge_exactly() {
    let mut s = stamped(0.2, 1);
    s.graph.run().unwrap();
    s.graph.set_tile_budget(Some(3));
    s.graph.set_field_op(s.disk, disk_op(0.6)).unwrap();
    let mut runs = 0;
    loop {
        s.graph.run().unwrap();
        runs += 1;
        let report = s.graph.tile_report();
        // The distance transform is global and recomputes whole when its
        // input changed; everything else stays within the budget.
        let local = report.tiles_recomputed - 64 * report.whole_recomputes;
        assert!(local <= 3, "run {runs}: {report:?}");
        if report.pending_tiles == 0 {
            break;
        }
        assert!(runs < 100, "budgeted runs must converge");
    }
    assert!(runs > 1, "the edit should need several budgeted runs");
    let mut fresh = stamped(0.6, 1);
    fresh.graph.run().unwrap();
    assert_eq!(digests(&s), digests(&fresh));
    // Settled: nothing left to run.
    assert_eq!(s.graph.run().unwrap().executed_nodes, 0);
}

#[test]
fn unchanged_outputs_cut_off_their_dependents() {
    let mut f = fixture();
    f.graph.run().unwrap();
    let before = f.graph.raster_value(f.soft).unwrap().fingerprint;

    // Re-setting fbm's op to its current value re-runs fbm only; everything
    // downstream sees an equal program and is cut off.
    f.graph
        .set_field_op(
            f.fbm,
            Op::Fractal {
                basis: Basis::Value,
                domain: domain(),
                frequency: [4.0, 4.0],
                seed: 2,
                params: FractalParams::default(),
            },
        )
        .unwrap();
    let summary = f.graph.run().unwrap();
    assert_eq!(summary.executed_nodes, 1);
    assert_eq!(summary.cut_off_nodes, 4, "height, map, normals, soft");
    assert_eq!(f.graph.run_count(f.fbm), Some(2));
    for node in [f.height, f.map, f.normals, f.soft] {
        assert_eq!(f.graph.run_count(node), Some(1));
    }
    assert_eq!(f.graph.raster_value(f.soft).unwrap().fingerprint, before);

    // Re-setting a raster's parameters re-runs just that raster.
    f.graph
        .set_raster_params(f.soft, RasterParams::Blur(GaussianBlur { sigma: 0.02 }))
        .unwrap();
    let summary = f.graph.run().unwrap();
    assert_eq!((summary.executed_nodes, summary.cut_off_nodes), (1, 0));
    assert_eq!(f.graph.raster_value(f.soft).unwrap().fingerprint, before);
}

#[test]
fn derivation_changes_with_equal_texels_rerun_without_recomputing_tiles() {
    let d = domain();
    let mut g = MaterialGraph::with_tile_size(8);
    let noise = g
        .field(
            "noise",
            Op::Noise {
                basis: Basis::Gradient,
                domain: d,
                frequency: [8.0, 8.0],
                seed: 1,
            },
            &[],
        )
        .unwrap();
    let clamp = |max: f32| Op::Clamp {
        input: operand(0),
        min: -10.0,
        max,
    };
    let clamped = g.field("clamped", clamp(10.0), &[noise]).unwrap();
    let map = g.realize("map", clamped, 32, 32).unwrap();
    let soft = g
        .raster(
            "soft",
            RasterParams::Blur(GaussianBlur { sigma: 0.02 }),
            map,
        )
        .unwrap();
    g.run().unwrap();
    let texels = |g: &MaterialGraph| match &g.raster_value(soft).unwrap().data {
        RasterData::Scalar(r) => r.digest(),
        RasterData::Vector3(r) => r.digest(),
        RasterData::Typed(r) => r.digest(),
    };
    let (before, fingerprint) = (texels(&g), g.raster_value(soft).unwrap().fingerprint);

    // No noise value reaches either bound, so the texels cannot change, but
    // the derivation does: the dependents re-run to keep their fingerprints
    // exact, and recompute no tiles.
    g.set_field_op(clamped, clamp(11.0)).unwrap();
    let summary = g.run().unwrap();
    assert_eq!((summary.executed_nodes, summary.cut_off_nodes), (3, 0));
    let report = g.tile_report();
    assert_eq!(report.tiles_changed, 0);
    assert_eq!(
        report.tiles_recomputed, 16,
        "the map re-realizes, the blur reuses"
    );
    assert_eq!(texels(&g), before);
    assert_ne!(g.raster_value(soft).unwrap().fingerprint, fingerprint);
}

/// A disk over noise realized at 128², then a Box and a Kaiser mip chain
/// down three levels.
fn mipped(x: f32) -> (MaterialGraph, NodeId, NodeId, Vec<NodeId>) {
    let mut g = MaterialGraph::with_tile_size(16);
    let noise = g.field("noise", noise_op(1), &[]).unwrap();
    let disk = g.field("disk", disk_op(x), &[]).unwrap();
    let height = g
        .field(
            "height",
            Op::Max {
                a: operand(0),
                b: operand(1),
            },
            &[noise, disk],
        )
        .unwrap();
    let map = g.realize("map", height, 128, 128).unwrap();
    let mut levels = Vec::new();
    for (name, filter) in [("box", Filter::Box), ("kaiser", Filter::Kaiser)] {
        let mut above = map;
        for level in 1..=3 {
            above = g.mip(&format!("{name}{level}"), above, filter).unwrap();
            levels.push(above);
        }
    }
    (g, disk, map, levels)
}

fn scalar(g: &MaterialGraph, node: NodeId) -> &Raster {
    match &g.raster_value(node).unwrap().data {
        RasterData::Scalar(r) => r,
        RasterData::Vector3(_) | RasterData::Typed(_) => panic!("mips are scalar"),
    }
}

#[test]
fn mip_nodes_match_encode_chains_bit_for_bit() {
    let (mut g, _, map, levels) = mipped(0.2);
    g.run().unwrap();
    let base = scalar(&g, map);
    let image = Image::new(
        base.width(),
        base.height(),
        1,
        base.edge(),
        base.values().to_vec(),
    )
    .unwrap();
    for (chain, filter) in levels.chunks(3).zip([Filter::Box, Filter::Kaiser]) {
        let expected = dapple_encode::data_mips(&image, filter);
        for (level, &node) in chain.iter().enumerate() {
            let raster = scalar(&g, node);
            let want = &expected.levels()[level + 1];
            assert_eq!(
                (raster.width(), raster.height()),
                (want.width(), want.height())
            );
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(raster.values()),
                bits(want.values()),
                "{filter:?} {level}"
            );
            // Each level covers the same region with larger texels.
            assert_eq!(
                raster.texel(),
                base.texel() * f32::powi(2.0, i32::try_from(level).unwrap() + 1)
            );
        }
    }
}

#[test]
fn mip_nodes_recompute_only_reached_tiles() {
    let (mut g, disk, _, levels) = mipped(0.2);
    g.run().unwrap();
    g.set_field_op(disk, disk_op(0.25)).unwrap();
    g.run().unwrap();
    let report = g.tile_report();
    // map 64 tiles, box/kaiser levels 16 + 4 + 1 tiles each: 106 in all.
    assert_eq!(report.whole_recomputes, 0, "{report:?}");
    assert!(report.tiles_recomputed < 106 / 2, "{report:?}");

    let (mut fresh, _, _, fresh_levels) = mipped(0.25);
    fresh.run().unwrap();
    for (&a, &b) in levels.iter().zip(&fresh_levels) {
        assert_eq!(scalar(&g, a).digest(), scalar(&fresh, b).digest());
    }

    // A new filter re-keys that node and recomputes it whole.
    g.set_mip_filter(levels[0], Filter::Kaiser).unwrap();
    g.run().unwrap();
    assert_eq!(
        scalar(&g, levels[0]).digest(),
        scalar(&g, levels[3]).digest()
    );
}

#[test]
fn recipes_carry_mip_nodes() {
    let (mut g, _, _, levels) = mipped(0.2);
    g.run().unwrap();
    let recipe = g.recipe();
    let predicted = recipe.fingerprints().unwrap();
    for (node, label) in levels
        .iter()
        .zip(["box1", "box2", "box3", "kaiser1", "kaiser2", "kaiser3"])
    {
        assert_eq!(
            predicted[label],
            NodeFingerprint::Raster(g.raster_value(*node).unwrap().fingerprint),
            "{label}"
        );
    }
    let (mut rebuilt, ids) = recipe.build(16).unwrap();
    rebuilt.run().unwrap();
    assert_eq!(
        scalar(&rebuilt, ids["kaiser3"]).digest(),
        scalar(&g, levels[5]).digest()
    );
    // Mips read rasters, not fields.
    let mut wrong = recipe.clone();
    for node in &mut wrong.nodes {
        if let Step::Mip { input, .. } = &mut node.step {
            *input = "height".into();
        }
    }
    assert!(matches!(
        wrong.fingerprints(),
        Err(RecipeError::WrongInput { .. })
    ));
}

/// A disk over noise realized at 64², blurred, sampled back with its Box
/// mips, warped by noise, and realized again.
struct Resampled {
    graph: MaterialGraph,
    disk: NodeId,
    soft: NodeId,
    sample: NodeId,
    again: NodeId,
}

fn resampled(x: f32) -> Resampled {
    let mut g = MaterialGraph::with_tile_size(16);
    let noise = g.field("noise", noise_op(1), &[]).unwrap();
    let disk = g.field("disk", disk_op(x), &[]).unwrap();
    let height = g
        .field(
            "height",
            Op::Max {
                a: operand(0),
                b: operand(1),
            },
            &[noise, disk],
        )
        .unwrap();
    let map = g.realize("map", height, 64, 64).unwrap();
    let soft = g
        .raster(
            "soft",
            RasterParams::Blur(GaussianBlur { sigma: 0.01 }),
            map,
        )
        .unwrap();
    let mip1 = g.mip("soft1", soft, Filter::Box).unwrap();
    let mip2 = g.mip("soft2", mip1, Filter::Box).unwrap();
    let sample = g
        .sample("sampled", &[soft, mip1, mip2], SamplePolicy::Linear)
        .unwrap();
    let doubled = g
        .field(
            "doubled",
            Op::Add {
                a: operand(0),
                b: operand(0),
            },
            &[sample],
        )
        .unwrap();
    let again = g.realize("again", doubled, 64, 64).unwrap();
    Resampled {
        graph: g,
        disk,
        soft,
        sample,
        again,
    }
}

#[test]
fn sample_nodes_read_rasters_back_as_fields() {
    let mut r = resampled(0.2);
    r.graph.run().unwrap();
    let soft = scalar(&r.graph, r.soft);
    let program = r.graph.field_value(r.sample).unwrap();
    // At texel centers with a point footprint the field is the raster.
    for (y, x) in [(0_u32, 0_u32), (13, 40), (63, 63)] {
        #[expect(clippy::cast_precision_loss, reason = "tiny test sizes")]
        let p = Vec2::new((x as f32 + 0.5) / 64.0, (y as f32 + 0.5) / 64.0);
        let value = program.eval(p, dapple_field::Footprint::POINT);
        assert_eq!(
            value.scalar().unwrap().to_bits(),
            soft.values()[(y * 64 + x) as usize].to_bits()
        );
    }
    // Realizing the doubled field at the same resolution doubles the texels:
    // realization samples texel centers with a one-texel footprint, which
    // reads level 0 exactly.
    let again = scalar(&r.graph, r.again);
    for (a, b) in again.values().iter().zip(soft.values()) {
        assert_eq!(a.to_bits(), (b + b).to_bits());
    }
}

#[test]
fn sample_nodes_change_only_near_changed_tiles() {
    let mut r = resampled(0.2);
    r.graph.run().unwrap();
    r.graph.set_field_op(r.disk, disk_op(0.25)).unwrap();
    r.graph.run().unwrap();
    let report = r.graph.tile_report();
    assert_eq!(report.unbounded_changes, 0, "{report:?}");
    // Every node, the resampled realization included, recomputes only the
    // tiles the move reaches.
    assert_eq!(report.whole_recomputes, 0, "{report:?}");
    assert!(report.tiles_reused > 0, "{report:?}");
    let mut fresh = resampled(0.25);
    fresh.graph.run().unwrap();
    assert_eq!(
        scalar(&r.graph, r.again).digest(),
        scalar(&fresh.graph, fresh.again).digest()
    );
    assert_eq!(
        r.graph.field_value(r.sample).unwrap().fingerprint(),
        fresh.graph.field_value(fresh.sample).unwrap().fingerprint()
    );
}

#[test]
fn recipes_carry_sample_nodes() {
    let mut r = resampled(0.2);
    r.graph.run().unwrap();
    let recipe = r.graph.recipe();
    let predicted = recipe.fingerprints().unwrap();
    assert_eq!(
        predicted["sampled"],
        NodeFingerprint::Field(r.graph.field_value(r.sample).unwrap().fingerprint())
    );
    assert_eq!(
        predicted["again"],
        NodeFingerprint::Raster(r.graph.raster_value(r.again).unwrap().fingerprint)
    );
    let (mut rebuilt, ids) = recipe.build(16).unwrap();
    rebuilt.run().unwrap();
    assert_eq!(
        scalar(&rebuilt, ids["again"]).digest(),
        scalar(&r.graph, r.again).digest()
    );
    // Sample nodes read rasters, and are fields themselves.
    let mut wrong = recipe.clone();
    for node in &mut wrong.nodes {
        if let Step::Sample { inputs, .. } = &mut node.step {
            inputs[0] = "height".into();
        }
    }
    assert!(matches!(
        wrong.fingerprints(),
        Err(RecipeError::WrongInput { .. })
    ));
    // A field op holding an image cannot live in a recipe.
    let image = match r.graph.field_value(r.sample).unwrap().program().op(r
        .graph
        .field_value(r.sample)
        .unwrap()
        .program()
        .output())
    {
        Some(Op::Sample { image }) => image.clone(),
        other => panic!("expected a sample op, found {other:?}"),
    };
    let mut embedded = recipe;
    embedded.nodes.push(RecipeNode {
        label: "embedded".into(),
        step: Step::Field {
            op: Op::Sample { image },
            inputs: Vec::new(),
        },
    });
    assert!(matches!(
        embedded.fingerprints(),
        Err(RecipeError::EmbeddedImage(_))
    ));
}

/// A disk over noise, realized, sampled back, warped by `dx`/`dy`, and
/// realized again.
fn warped(x: f32, displacement: impl Fn(u64) -> Op) -> (MaterialGraph, NodeId, NodeId) {
    let mut g = MaterialGraph::with_tile_size(16);
    let noise = g.field("noise", noise_op(1), &[]).unwrap();
    let disk = g.field("disk", disk_op(x), &[]).unwrap();
    let height = g
        .field(
            "height",
            Op::Max {
                a: operand(0),
                b: operand(1),
            },
            &[noise, disk],
        )
        .unwrap();
    let map = g.realize("map", height, 64, 64).unwrap();
    let sample = g.sample("sampled", &[map], SamplePolicy::Linear).unwrap();
    let dx = g.field("dx", displacement(7), &[]).unwrap();
    let dy = g.field("dy", displacement(8), &[]).unwrap();
    let warp = g
        .field(
            "warp",
            Op::Warp {
                input: operand(0),
                dx: operand(1),
                dy: operand(2),
                amount: 0.02,
            },
            &[sample, dx, dy],
        )
        .unwrap();
    let again = g.realize("again", warp, 64, 64).unwrap();
    (g, disk, again)
}

#[test]
fn bounded_warps_keep_changes_local() {
    let smooth = |seed| Op::Noise {
        basis: Basis::Gradient,
        domain: domain(),
        frequency: [4.0, 4.0],
        seed,
    };
    let (mut g, disk, again) = warped(0.2, smooth);
    g.run().unwrap();
    g.set_field_op(disk, disk_op(0.25)).unwrap();
    g.run().unwrap();
    let report = g.tile_report();
    assert_eq!(report.unbounded_warps, 0, "{report:?}");
    assert_eq!(report.unbounded_changes, 0, "{report:?}");
    // The warped realization recomputes only tiles the move reaches, grown by
    // the warp's reach.
    assert_eq!(report.whole_recomputes, 0, "{report:?}");
    assert!(report.tiles_reused > 0, "{report:?}");
    let (mut fresh, _, fresh_again) = warped(0.25, smooth);
    fresh.run().unwrap();
    assert_eq!(
        scalar(&g, again).digest(),
        scalar(&fresh, fresh_again).digest()
    );
}

#[test]
fn unbounded_warps_recompute_whole_and_say_so() {
    // Cell values jump between cells, so the warp's stretch has no bound.
    let jumpy = |seed| Op::Cellular {
        domain: domain(),
        frequency: [4.0, 4.0],
        jitter: 1.0,
        seed,
        output: dapple_field::CellOutput::CellValue,
    };
    let (mut g, disk, again) = warped(0.2, jumpy);
    g.run().unwrap();
    g.set_field_op(disk, disk_op(0.25)).unwrap();
    g.run().unwrap();
    let report = g.tile_report();
    assert_eq!(report.unbounded_warps, 1, "{report:?}");
    assert_eq!(report.unbounded_changes, 1, "{report:?}");
    let (mut fresh, _, fresh_again) = warped(0.25, jumpy);
    fresh.run().unwrap();
    assert_eq!(
        scalar(&g, again).digest(),
        scalar(&fresh, fresh_again).digest()
    );
}

/// A noise field with a mask, identifier, direction and normals derived
/// from it, each realized with its own type.
struct Typed {
    graph: MaterialGraph,
    noise: NodeId,
    mask: NodeId,
    ids: NodeId,
    directions: NodeId,
    normals: NodeId,
}

fn typed() -> Typed {
    let mut g = MaterialGraph::with_tile_size(8);
    let noise = g.field("noise", noise_op(3), &[]).unwrap();
    let mask = g
        .field("mask", Op::AsMask { input: operand(0) }, &[noise])
        .unwrap();
    let id = g
        .field(
            "id",
            Op::ToId {
                input: operand(0),
                levels: 4,
            },
            &[mask],
        )
        .unwrap();
    let angle = g
        .field("angle", Op::Direction { angle: operand(0) }, &[noise])
        .unwrap();
    let mask = g.realize("mask.map", mask, 32, 32).unwrap();
    let ids = g.realize("id.map", id, 32, 32).unwrap();
    let directions = g.realize("angle.map", angle, 32, 32).unwrap();
    let normals = g.normals("normals", noise, (32, 32), 0.05).unwrap();
    Typed {
        graph: g,
        noise,
        mask,
        ids,
        directions,
        normals,
    }
}

#[test]
fn rasters_keep_their_semantic_type() {
    let mut t = typed();
    t.graph.run().unwrap();
    let port = |node| t.graph.raster_value(node).unwrap().port;
    assert_eq!(port(t.mask), PortType::Mask);
    assert_eq!(port(t.ids), PortType::Id);
    assert_eq!(port(t.directions), PortType::Direction);
    assert_eq!(port(t.normals), PortType::Normal(NormalFrame::Domain));
    let RasterData::Typed(ids) = &t.graph.raster_value(t.ids).unwrap().data else {
        panic!("identifiers are typed rasters")
    };
    assert!(matches!(ids.storage(), Storage::U32(_)));
    // Unchanged programs reuse the whole realization.
    let before = t.graph.raster_value(t.ids).unwrap().fingerprint;
    t.graph.set_field_op(t.noise, noise_op(3)).unwrap();
    t.graph.run().unwrap();
    assert_eq!(t.graph.raster_value(t.ids).unwrap().fingerprint, before);
}

fn refusal(result: Result<NodeId, MaterialError>) -> NodeError {
    match result {
        Err(MaterialError::Refused { error, .. }) => error,
        other => panic!("expected a refusal, found {other:?}"),
    }
}

#[test]
fn operations_refuse_types_they_do_not_accept() {
    // Blurring identifiers is refused when the node is added, not guessed;
    // blurring a mask is fine.
    let mut t = typed();
    let blur = RasterParams::Blur(GaussianBlur { sigma: 0.01 });
    assert!(matches!(
        refusal(t.graph.raster("bad", blur, t.ids)),
        NodeError::TypeRefused {
            port: PortType::Id,
            ..
        }
    ));
    let soft = t.graph.raster("soft", blur, t.mask).unwrap();
    assert_eq!(t.graph.port(soft), Some(PortType::Mask));
    t.graph.run().unwrap();
    assert_eq!(t.graph.raster_value(soft).unwrap().port, PortType::Mask);
    // Averaging identifiers, or normals (which would drop the length that
    // measures their spread), is a refused reduction policy.
    for (raster, port) in [
        (t.ids, PortType::Id),
        (t.normals, PortType::Normal(NormalFrame::Domain)),
    ] {
        assert!(matches!(
            refusal(t.graph.reduce("bad", raster, ReductionPolicy::Average, 1)),
            NodeError::Typed(TypedError::PolicyRefused { port: p, .. }) if p == port
        ));
    }
    // Mip filtering refuses directions.
    assert!(matches!(
        refusal(t.graph.mip("bad", t.directions, Filter::Box)),
        NodeError::TypeRefused {
            port: PortType::Direction,
            ..
        }
    ));
    // Identifiers are never interpolated; nearest sampling is fine.
    assert_eq!(
        refusal(t.graph.sample("bad", &[t.ids], SamplePolicy::Linear)),
        NodeError::SamplingRefused {
            port: PortType::Id,
            policy: SamplePolicy::Linear
        }
    );
    let cells = t
        .graph
        .sample("cells", &[t.ids], SamplePolicy::Nearest)
        .unwrap();
    assert_eq!(t.graph.port(cells), Some(PortType::Id));
    // A mip chain holds its base's type.
    assert!(matches!(
        refusal(
            t.graph
                .sample("bad", &[t.mask, t.ids], SamplePolicy::Nearest)
        ),
        NodeError::MixedLevels { .. }
    ));
    // Nothing refused was added, so the graph still runs.
    t.graph.run().unwrap();
}

#[test]
fn edits_that_would_mistype_a_node_are_refused() {
    let mut t = typed();
    let reduced = t
        .graph
        .reduce("mask.1", t.mask, ReductionPolicy::Average, 1)
        .unwrap();
    t.graph.run().unwrap();
    // Switching the reduction to an identifier policy is refused.
    assert!(matches!(
        t.graph.set_reduction(reduced, ReductionPolicy::IdMode, 1),
        Err(MaterialError::Refused { .. })
    ));
    // So is an upstream edit that would turn the noise into a vector, which
    // the mask node downstream cannot clamp.
    let before = t.graph.recipe();
    let refused = t.graph.set_field_op(t.noise, Op::Position3);
    assert!(
        matches!(&refused, Err(MaterialError::Refused { label, .. }) if label == "mask"),
        "{refused:?}"
    );
    // Refused edits leave the graph as it was.
    assert_eq!(t.graph.recipe(), before);
    assert_eq!(t.graph.port(t.noise), Some(PortType::Scalar));
    t.graph.run().unwrap();
    // A type-changing edit every dependent accepts goes through.
    t.graph
        .set_reduction(
            reduced,
            ReductionPolicy::ThresholdCoverage { cutoff: 0.5 },
            1,
        )
        .unwrap();
    t.graph.run().unwrap();
}

#[test]
fn reduction_policies_are_part_of_the_derivation() {
    let mut t = typed();
    let mode = t
        .graph
        .reduce("id.1", t.ids, ReductionPolicy::IdMode, 2)
        .unwrap();
    let axial = t
        .graph
        .reduce("angle.1", t.directions, ReductionPolicy::Axial, 1)
        .unwrap();
    let coverage = t
        .graph
        .reduce("mask.1", t.mask, ReductionPolicy::Average, 1)
        .unwrap();
    t.graph.run().unwrap();
    let fingerprint = |g: &MaterialGraph, node| g.raster_value(node).unwrap().fingerprint;
    let before = fingerprint(&t.graph, coverage);
    assert_eq!(
        t.graph.raster_value(mode).unwrap().port,
        PortType::Id,
        "reductions keep the type"
    );
    assert_eq!(
        t.graph.raster_value(axial).unwrap().port,
        PortType::Direction
    );

    // The recipe predicts every fingerprint, policies included, and
    // rebuilds the same graph.
    let recipe = t.graph.recipe();
    let predicted = recipe.fingerprints().unwrap();
    for (label, node) in [("id.1", mode), ("angle.1", axial), ("mask.1", coverage)] {
        assert_eq!(
            predicted[label],
            NodeFingerprint::Raster(fingerprint(&t.graph, node))
        );
    }

    // Choosing coverage preservation is a different derivation.
    t.graph
        .set_reduction(
            coverage,
            ReductionPolicy::ThresholdCoverage { cutoff: 0.5 },
            1,
        )
        .unwrap();
    t.graph.run().unwrap();
    assert_ne!(fingerprint(&t.graph, coverage), before);
    let exported = t.graph.recipe();
    assert_ne!(
        exported.fingerprint().unwrap(),
        recipe.fingerprint().unwrap()
    );
}

#[test]
fn normal_reductions_keep_the_variance_encode_moves_into_roughness() {
    let mut t = typed();
    let reduced = t
        .graph
        .reduce("normals.1", t.normals, ReductionPolicy::NormalMean, 1)
        .unwrap();
    t.graph.run().unwrap();
    let RasterData::Vector3(base) = &t.graph.raster_value(t.normals).unwrap().data else {
        panic!("normals are three-channel")
    };
    let RasterData::Typed(level) = &t.graph.raster_value(reduced).unwrap().data else {
        panic!("reductions are typed")
    };
    let image = Image::new(
        base.width(),
        base.height(),
        3,
        base.edge(),
        base.values().iter().flatten().copied().collect(),
    )
    .unwrap();
    let chain = dapple_encode::normal_mips(&image, None, 0.0, Filter::Box).unwrap();
    let roughness = &chain.roughness.levels()[1];
    // With zero base roughness, encode's level-1 roughness is Toksvig's
    // σ² = (1 − L) / L from the mean length L, as α′² = σ², α = r².
    for y in 0..level.height() {
        for x in 0..level.width() {
            let Value::Vector3(mean) = level.value_at(i64::from(x), i64::from(y)) else {
                panic!("normals are vectors")
            };
            let length = mean.length();
            let variance = ((1.0 - length) / length).min(1.0);
            let r = roughness.values()[(y * level.width() + x) as usize];
            assert!(
                (r.powi(4) - variance).abs() < 1e-4,
                "({x}, {y}): {r} vs {variance}"
            );
        }
    }
}

/// The value of texel `(x, y)` of a realized or reduced raster.
fn texel_value(data: &RasterData, x: i64, y: i64) -> Value {
    match data {
        RasterData::Scalar(r) => Value::Scalar(r.at(x, y)),
        RasterData::Vector3(r) => Value::Vector3(glam::Vec3::from_array(r.at(x, y))),
        RasterData::Typed(r) => r.value_at(x, y),
    }
}

/// A field of each semantic type over the unit torus, from noise.
fn typed_field(g: &mut MaterialGraph, port: PortType) -> NodeId {
    let n: Vec<NodeId> = (0..3)
        .map(|i| g.field(&format!("n{i}"), noise_op(20 + i), &[]).unwrap())
        .collect();
    let one = g
        .field(
            "one",
            Op::Constant {
                domain: domain(),
                value: 1.0,
            },
            &[],
        )
        .unwrap();
    let (a, b, c) = (operand(0), operand(1), operand(2));
    let mask = g.field("mask", Op::AsMask { input: a }, &[n[0]]).unwrap();
    match port {
        PortType::Scalar => n[0],
        PortType::Mask => mask,
        PortType::Id => g
            .field(
                "id",
                Op::ToId {
                    input: a,
                    levels: 5,
                },
                &[mask],
            )
            .unwrap(),
        PortType::Vector2 => g.field("v2", Op::Vector2 { x: a, y: b }, &n[..2]).unwrap(),
        PortType::Vector3 => g.field("v3", Op::Vector3 { x: a, y: b, z: c }, &n).unwrap(),
        PortType::Color(_) => g
            .field("color", Op::Color { r: a, g: b, b: c }, &n)
            .unwrap(),
        PortType::Normal(_) => {
            let v = g
                .field("tilt", Op::Vector3 { x: a, y: b, z: c }, &[n[0], n[1], one])
                .unwrap();
            g.field("normal", Op::Normalize { input: a }, &[v]).unwrap()
        }
        PortType::Direction => g
            .field("direction", Op::Direction { angle: a }, &[n[0]])
            .unwrap(),
    }
}

/// What reducing the level-0 `base` to texel `(x, y)` of level `level`
/// under `policy` retains, computed directly from the footprint's texels.
fn expected_texel(
    base: &RasterData,
    policy: ReductionPolicy,
    level: u32,
    (x, y): (i64, i64),
) -> Value {
    let step = 1_i64 << level;
    let texels: Vec<Value> = (0..step)
        .flat_map(|dy| (0..step).map(move |dx| (dx, dy)))
        .map(|(dx, dy)| texel_value(base, x * step + dx, y * step + dy))
        .collect();
    #[expect(clippy::cast_precision_loss, reason = "tiny footprints")]
    let n = texels.len() as f32;
    match policy {
        ReductionPolicy::IdPoint => {
            let c = (step - 1) / 2;
            texel_value(base, x * step + c, y * step + c)
        }
        ReductionPolicy::IdMode => {
            let mut ids: Vec<u32> = texels
                .iter()
                .map(|v| match v {
                    Value::Id(id) => *id,
                    other => panic!("expected identifiers, found {other:?}"),
                })
                .collect();
            ids.sort_unstable();
            let mut best = (0, 0);
            for &id in &ids {
                let count = ids.iter().filter(|&&i| i == id).count();
                if count > best.1 {
                    best = (id, count);
                }
            }
            Value::Id(best.0)
        }
        // The (unnormalized, per-component) mean of the footprint.
        _ => {
            let mut sum = [0.0_f32; 3];
            for v in &texels {
                for (k, s) in sum.iter_mut().enumerate() {
                    *s += v.component(k).unwrap_or(0.0);
                }
            }
            let mean = sum.map(|s| s / n);
            match texels[0] {
                Value::Scalar(_) => Value::Scalar(mean[0]),
                Value::Vector2(_) => Value::Vector2(Vec2::new(mean[0], mean[1])),
                _ => Value::Vector3(glam::Vec3::from_array(mean)),
            }
        }
    }
}

fn close(a: Value, b: Value) -> bool {
    match (a, b) {
        (Value::Id(a), Value::Id(b)) => a == b,
        _ => (0..3).all(|k| match (a.component(k), b.component(k)) {
            (Some(x), Some(y)) => (x - y).abs() <= 1e-5,
            (None, None) => true,
            _ => false,
        }),
    }
}

/// Slice 0's gate: every semantic type survives realize → reduce → sample
/// under each permitted policy, with the retention the policy states, and
/// the sampled field agrees with the reference reader.
#[test]
fn every_type_survives_realize_reduce_sample() {
    use dapple_raster::typed::sample as reference;
    let color = PortType::Color(dapple_field::Primaries::Rec709);
    let normal = PortType::Normal(NormalFrame::Domain);
    let cases = [
        (PortType::Scalar, ReductionPolicy::Average),
        (PortType::Mask, ReductionPolicy::Average),
        (
            PortType::Mask,
            ReductionPolicy::ThresholdCoverage { cutoff: 0.5 },
        ),
        (PortType::Id, ReductionPolicy::IdPoint),
        (PortType::Id, ReductionPolicy::IdMode),
        (PortType::Vector2, ReductionPolicy::Average),
        (PortType::Vector3, ReductionPolicy::Average),
        (color, ReductionPolicy::Average),
        (normal, ReductionPolicy::NormalMean),
        (PortType::Direction, ReductionPolicy::Axial),
    ];
    const SIZE: u32 = 32;
    for (port, policy) in cases {
        for sampling in [SamplePolicy::default_for(port), SamplePolicy::Nearest] {
            let mut g = MaterialGraph::with_tile_size(8);
            let field = typed_field(&mut g, port);
            let base = g.realize("base", field, SIZE, SIZE).unwrap();
            let l1 = g.reduce("l1", base, policy, 1).unwrap();
            let l2 = g.reduce("l2", base, policy, 2).unwrap();
            let sampled = g.sample("sampled", &[base, l1, l2], sampling).unwrap();
            assert_eq!(g.port(sampled), Some(port), "{port:?}: typed when built");
            g.run().unwrap();
            let program = g.field_value(sampled).unwrap();
            assert_eq!(program.output_type(), port, "{port:?}: typed when run");
            let rasters: Vec<TypedRaster> = [base, l1, l2]
                .iter()
                .map(|&n| {
                    let v = g.raster_value(n).unwrap();
                    assert_eq!(v.port, port, "{port:?}: every level keeps the type");
                    v.data.to_typed(v.port).unwrap()
                })
                .collect();
            let base_data = &g.raster_value(base).unwrap().data;
            // At each level's texel centers, with that level's footprint, the
            // field reads the reduced texel exactly, and that texel is what
            // the policy retains of its level-0 footprint.
            for level in 1..=2_u32 {
                let n = SIZE >> level;
                #[expect(clippy::cast_precision_loss, reason = "tiny sizes")]
                let texel = 1.0 / n as f32;
                let footprint = dapple_field::Footprint::new(texel).unwrap();
                for (x, y) in [(0_u32, 0_u32), (n / 2, 1), (n - 1, n - 1), (3, n / 3)] {
                    #[expect(clippy::cast_precision_loss, reason = "tiny sizes")]
                    let p = Vec2::new((x as f32 + 0.5) * texel, (y as f32 + 0.5) * texel);
                    let value = program.eval(p, footprint);
                    let reduced = rasters[level as usize].value_at(i64::from(x), i64::from(y));
                    assert_eq!(
                        value, reduced,
                        "{port:?} {policy:?} {sampling:?} level {level}"
                    );
                    if !matches!(policy, ReductionPolicy::ThresholdCoverage { .. }) {
                        let expected =
                            expected_texel(base_data, policy, level, (i64::from(x), i64::from(y)));
                        assert!(
                            close(value, expected),
                            "{port:?} {policy:?}: {value:?} retains {expected:?}"
                        );
                    }
                }
            }
            // Anywhere, at any footprint, the field is the reference reader.
            for (i, w) in [0.0, 0.02, 0.05, 0.11, 0.4].into_iter().enumerate() {
                #[expect(clippy::cast_precision_loss, reason = "a handful of points")]
                let p = Vec2::new(0.137 + 0.19 * i as f32, 0.71 - 0.13 * i as f32);
                let value = program.eval(p, dapple_field::Footprint::new(w).unwrap());
                let expected = reference(&rasters, domain(), sampling, p, w).unwrap();
                assert!(
                    close(value, expected),
                    "{port:?} {sampling:?} at {p} ({w}): {value:?} vs {expected:?}"
                );
            }
            // What the policies promise, beyond single texels.
            let level1 = &rasters[1];
            match policy {
                ReductionPolicy::ThresholdCoverage { cutoff } => {
                    let above = |r: &TypedRaster| {
                        let n = r.width() * r.height();
                        let count = (0..n)
                            .filter(|&i| {
                                let v =
                                    r.value_at(i64::from(i % r.width()), i64::from(i / r.width()));
                                v.scalar().unwrap() >= cutoff
                            })
                            .count();
                        #[expect(clippy::cast_precision_loss, reason = "tiny sizes")]
                        let fraction = count as f32 / n as f32;
                        fraction
                    };
                    let (a, b) = (above(&rasters[0]), above(level1));
                    assert!((a - b).abs() <= 1.0 / 64.0, "coverage {a} vs {b}");
                }
                ReductionPolicy::NormalMean => {
                    // Not renormalized: spread shows as a mean shorter than 1.
                    let shortest = (0..16)
                        .flat_map(|y| (0..16).map(move |x| (x, y)))
                        .map(|(x, y)| match level1.value_at(x, y) {
                            Value::Vector3(v) => v.length(),
                            other => panic!("normals are vectors, found {other:?}"),
                        })
                        .fold(f32::INFINITY, f32::min);
                    assert!(shortest < 0.999, "{shortest}");
                }
                _ => {}
            }
        }
    }
}

#[test]
fn sampling_policies_are_part_of_the_fingerprint() {
    let mut t = typed();
    let linear = t
        .graph
        .sample("linear", &[t.mask], SamplePolicy::Linear)
        .unwrap();
    let nearest = t
        .graph
        .sample("nearest", &[t.mask], SamplePolicy::Nearest)
        .unwrap();
    t.graph.run().unwrap();
    let fp = |g: &MaterialGraph, node| g.field_value(node).unwrap().fingerprint();
    assert_ne!(fp(&t.graph, linear), fp(&t.graph, nearest));
    // The recipe carries the policy and predicts both.
    let recipe = t.graph.recipe();
    let predicted = recipe.fingerprints().unwrap();
    assert_eq!(
        predicted["linear"],
        NodeFingerprint::Field(fp(&t.graph, linear))
    );
    assert_eq!(
        predicted["nearest"],
        NodeFingerprint::Field(fp(&t.graph, nearest))
    );
    // Editing the policy re-runs the node with the other fingerprint.
    t.graph
        .set_sample_policy(linear, SamplePolicy::Nearest)
        .unwrap();
    t.graph.run().unwrap();
    assert_eq!(fp(&t.graph, linear), fp(&t.graph, nearest));
}
