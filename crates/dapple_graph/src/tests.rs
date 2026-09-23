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
    // An operand without an input fails when the node runs.
    f.graph
        .field("bad", Op::Abs { input: operand(3) }, &[f.noise])
        .unwrap();
    let error = f.graph.run().unwrap_err();
    assert!(
        matches!(
            &error,
            MaterialError::Graph(GraphError::Node {
                source: NodeError::MissingOperand { index: 3 },
                ..
            })
        ),
        "{error:?}"
    );
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
            Step::Field { inputs, .. } | Step::Sample { inputs } => {
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
        with(&|r| r.version = 2),
        RecipeError::Version { found: 2 }
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
    let sample = g.sample("sampled", &[soft, mip1, mip2]).unwrap();
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
        if let Step::Sample { inputs } = &mut node.step {
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
    let sample = g.sample("sampled", &[map]).unwrap();
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

fn node_error(error: &MaterialError) -> &NodeError {
    match error {
        MaterialError::Graph(GraphError::Node { source, .. }) => source,
        other => panic!("expected a node error, found {other:?}"),
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

#[test]
fn operations_refuse_types_they_do_not_accept() {
    // Blurring identifiers is refused, not guessed; blurring a mask is fine.
    let mut t = typed();
    t.graph
        .raster(
            "bad",
            RasterParams::Blur(GaussianBlur { sigma: 0.01 }),
            t.ids,
        )
        .unwrap();
    let error = t.graph.run().unwrap_err();
    assert!(
        matches!(
            node_error(&error),
            NodeError::TypeRefused {
                port: PortType::Id,
                ..
            }
        ),
        "{error:?}"
    );
    let mut t = typed();
    let soft = t
        .graph
        .raster(
            "soft",
            RasterParams::Blur(GaussianBlur { sigma: 0.01 }),
            t.mask,
        )
        .unwrap();
    t.graph.run().unwrap();
    assert_eq!(t.graph.raster_value(soft).unwrap().port, PortType::Mask);
    // Averaging identifiers is refused.
    let mut t = typed();
    t.graph
        .reduce("bad", t.ids, ReductionPolicy::Average, 1)
        .unwrap();
    let error = t.graph.run().unwrap_err();
    assert!(
        matches!(
            node_error(&error),
            NodeError::Typed(TypedError::PolicyRefused { .. })
        ),
        "{error:?}"
    );
    // So is averaging normals, which would drop the length that measures
    // their spread.
    let mut t = typed();
    t.graph
        .reduce("bad", t.normals, ReductionPolicy::Average, 1)
        .unwrap();
    assert!(t.graph.run().is_err());
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
