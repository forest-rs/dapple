// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use dapple_field::program::Op;
use dapple_field::{Basis, Domain, FractalParams};

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
            Step::Field { inputs, .. } => {
                for i in inputs {
                    i.insert_str(0, "x.");
                }
            }
            Step::Realize { input, .. }
            | Step::Raster { input, .. }
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
