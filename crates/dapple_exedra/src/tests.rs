// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::vec;
use alloc::vec::Vec;

use exedra_mesh::TriMesh;
use glam::{Affine3A, Vec2, Vec3};

use super::{BakeError, BakeParams, SurfaceBake, UNCOVERED};

/// A 2 m × 1 m rectangle in the z = 0 plane whose chart is 1 UV unit per
/// meter, as two triangles.
fn rectangle() -> TriMesh {
    TriMesh {
        indices: vec![0, 1, 2, 0, 2, 3],
        positions: vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        uvs: vec![[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]],
        normals: vec![[0.0, 0.0, 1.0]; 4],
        ..TriMesh::default()
    }
}

fn params(density: f32, padding: u32) -> BakeParams {
    BakeParams {
        padding,
        ..BakeParams::new(density)
    }
}

#[test]
fn covered_texels_carry_their_surface_point() {
    let mesh = rectangle();
    let bake = SurfaceBake::new(&mesh, &mesh.indices, &params(8.0, 0)).unwrap();
    assert_eq!((bake.width(), bake.height()), (16, 8));
    assert_eq!(bake.stats().covered_texels, 16 * 8);
    assert_eq!(bake.stats().overlapping_texels, 0);
    // The chart is isometric, so each texel is 1/8 m on the surface and its
    // sample sits at its own center.
    for (sample, uv) in bake.samples().iter().zip(bake.uvs()) {
        assert!((sample.footprint.width() - 0.125).abs() < 1e-5);
        assert!((sample.position - Vec3::new(uv.x, uv.y, 0.0)).length() < 1e-5);
    }
    assert!(
        bake.normals()
            .iter()
            .all(|n| (*n - Vec3::Z).length() < 1e-6)
    );
}

#[test]
fn rows_run_from_the_smallest_v_and_the_transform_maps_the_chart() {
    let mesh = rectangle();
    let bake = SurfaceBake::new(&mesh, &mesh.indices, &params(4.0, 2)).unwrap();
    let uvs: Vec<Vec2> = bake.uvs().to_vec();
    let grid = bake.scatter(&uvs, Vec2::splat(f32::NAN));
    let width = bake.width() as usize;
    let transform = bake.texture_transform();
    let map = |uv: Vec2| uv * Vec2::from(transform.scale) + Vec2::from(transform.offset);
    // A covered texel's UV maps back to its own texel center.
    let (x, y) = (5, 4);
    let uv = grid[y * width + x];
    let texel = map(uv) * Vec2::new(bake.width() as f32, bake.height() as f32);
    assert!((texel - Vec2::new(x as f32 + 0.5, y as f32 + 0.5)).length() < 1e-4);
    // Grid rows grow with V.
    assert!(grid[(y + 1) * width + x].y > uv.y);
}

#[test]
fn dilation_fills_the_padding_from_the_nearest_texel() {
    let mesh = rectangle();
    let bake = SurfaceBake::new(&mesh, &mesh.indices, &params(4.0, 3)).unwrap();
    // 8 × 4 covered texels, padded by 3 on every side.
    assert_eq!((bake.width(), bake.height()), (14, 10));
    assert_eq!(bake.stats().covered_texels, 32);
    let ids: Vec<u32> = (0..u32::try_from(bake.samples().len()).unwrap()).collect();
    let grid = bake.scatter(&ids, UNCOVERED);
    // Every texel within 3 steps of the rectangle is filled; the corners
    // beyond 3 steps (Manhattan distance) are not.
    assert_eq!(grid[0], UNCOVERED);
    let covered_corner = grid[3 * 14 + 3];
    assert_ne!(covered_corner, UNCOVERED);
    assert_eq!(
        grid[3 * 14],
        covered_corner,
        "left padding repeats the edge"
    );
    assert_eq!(grid[3], covered_corner, "top padding repeats the edge");
    assert_eq!(
        bake.stats().dilated_texels as usize,
        grid.iter().filter(|s| **s != UNCOVERED).count() - 32
    );
}

#[test]
fn placement_maps_positions_normals_and_footprints() {
    let mesh = rectangle();
    let place = Affine3A::from_scale_rotation_translation(
        Vec3::splat(2.0),
        glam::Quat::from_rotation_x(core::f32::consts::FRAC_PI_2),
        Vec3::new(0.0, 0.0, 5.0),
    );
    let bake =
        SurfaceBake::new(&mesh, &mesh.indices, &params(8.0, 0).with_placement(place)).unwrap();
    let sample = bake.samples()[0];
    // Twice the size in the material, so each texel covers twice as much.
    assert!((sample.footprint.width() - 0.25).abs() < 1e-5);
    let uv = bake.uvs()[0];
    let expected = place.transform_point3(Vec3::new(uv.x, uv.y, 0.0));
    assert!((sample.position - expected).length() < 1e-5);
    assert!((bake.normals()[0] - Vec3::NEG_Y).length() < 1e-5);
}

#[test]
fn overlapping_charts_are_counted_and_first_triangle_wins() {
    let mut mesh = rectangle();
    // A second copy of the rectangle 1 m above, on the same chart.
    let lifted: Vec<[f32; 3]> = mesh
        .positions
        .iter()
        .map(|p| [p[0], p[1], p[2] + 1.0])
        .collect();
    mesh.positions.extend(lifted);
    mesh.uvs.extend_from_within(..);
    mesh.normals.extend_from_within(..);
    mesh.indices.extend([4, 5, 6, 4, 6, 7]);
    let bake = SurfaceBake::new(&mesh, &mesh.indices, &params(4.0, 0)).unwrap();
    assert!(bake.stats().overlapping_texels > 0);
    assert!(bake.samples().iter().all(|s| s.position.z == 0.0));
}

#[test]
fn malformed_input_is_refused() {
    let mesh = rectangle();
    assert_eq!(
        SurfaceBake::new(&mesh, &[], &params(4.0, 0)).unwrap_err(),
        BakeError::NoTriangles
    );
    assert_eq!(
        SurfaceBake::new(&mesh, &[0, 1, 9], &params(4.0, 0)).unwrap_err(),
        BakeError::InvalidIndex { index: 9 }
    );
    assert_eq!(
        SurfaceBake::new(&mesh, &mesh.indices, &params(0.0, 0)).unwrap_err(),
        BakeError::InvalidParams
    );
    let mut flat = rectangle();
    flat.uvs = vec![[0.5, 0.5]; 4];
    assert_eq!(
        SurfaceBake::new(&flat, &flat.indices, &params(4.0, 0)).unwrap_err(),
        BakeError::DegenerateChart
    );
    let mut bad = rectangle();
    bad.uvs[2] = [f32::NAN, 0.0];
    assert_eq!(
        SurfaceBake::new(&bad, &bad.indices, &params(4.0, 0)).unwrap_err(),
        BakeError::NonFiniteVertex { vertex: 2 }
    );
    let huge = BakeParams {
        max_texels: 100,
        ..params(64.0, 0)
    };
    assert!(matches!(
        SurfaceBake::new(&mesh, &mesh.indices, &huge),
        Err(BakeError::TooLarge { .. })
    ));
}

#[test]
fn bakes_are_deterministic() {
    let mesh = rectangle();
    let a = SurfaceBake::new(&mesh, &mesh.indices, &params(7.0, 2)).unwrap();
    let b = SurfaceBake::new(&mesh, &mesh.indices, &params(7.0, 2)).unwrap();
    assert_eq!(a.texel_source, b.texel_source);
    assert_eq!(a.samples().len(), b.samples().len());
    for (x, y) in a.samples().iter().zip(b.samples()) {
        assert_eq!(
            x.position.to_array().map(f32::to_bits),
            y.position.to_array().map(f32::to_bits)
        );
    }
}
