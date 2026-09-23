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
