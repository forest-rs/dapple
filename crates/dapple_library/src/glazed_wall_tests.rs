// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Pre-refactor bakes pin both the wall's bits and its original child keys.

use dapple_field::Edge;
use dapple_material::Grid;
use dapple_material::module::{Bind, Context};
use dapple_material::resource::NoResources;
use glam::Vec2;

use crate::modules::GlazedBrickWall;

#[test]
fn native_wall_preserves_its_pre_refactor_bakes_and_child_fingerprints() {
    // Captured before extracting the inline operations, at `wall` on
    // these unit wrapping grids, defaults except weathered/moss as below.
    let cases = [
        (
            32,
            [0x22bf81164de1ea4e, 0xbd26d5dd994c4183, 0x14983ad927fd2ea6],
            &[
                ("wall/body", 0x2cd89b5a3e2ac9cd),
                ("wall/mortar", 0x4d3e72472ca06f78),
                ("wall/glaze", 0x1a127e7059705161),
                ("wall/wear", 0x1c07a66fce1d435f),
                ("wall/dirt", 0x95fbd2575ef7ab31),
                ("wall/streaks", 0x806a7a09d9581fff),
                ("wall/salts", 0xf25b72d381b2eddc),
                ("wall/moss", 0x1c270e0588af3d9c),
            ][..],
        ),
        (
            64,
            [0x839576accedeb5cf, 0x0c3fd26c31b1098c, 0x45446fa5ee17d8f7],
            &[
                ("wall/body", 0x6d742a61f0cf877d),
                ("wall/mortar", 0x1217b3ff4e8ec174),
                ("wall/glaze", 0xacec1c9f5d211c6c),
                ("wall/wear", 0x09d31597ea64a642),
                ("wall/dirt", 0x1b333126bda2908d),
                ("wall/streaks", 0x965aaa1903986a5b),
                ("wall/salts", 0x3925807139ba97b0),
                ("wall/moss", 0xa1b0c16c6af3311c),
            ][..],
        ),
        (
            512,
            [0x339f238c7fe96c12, 0x7aa9bdb0064982a5, 0x6077876881bd52c7],
            &[
                ("wall/body", 0xbc465cf994aca923),
                ("wall/mortar", 0xbc83bde994d7fc7e),
                ("wall/glaze", 0x0348ecbf1cdb05fd),
                ("wall/wear", 0xc5e82e41d507ebfb),
                ("wall/dirt", 0xdf3591515cd3b485),
                ("wall/streaks", 0xbc7dda1b5dbd86e3),
                ("wall/salts", 0x731f914ce49c283f),
                ("wall/moss", 0x7e13362bb96d8e13),
            ][..],
        ),
    ];
    for (size, digests, fingerprints) in cases {
        let grid = Grid {
            width: size,
            height: size,
            origin: Vec2::ZERO,
            texel: Vec2::splat(1.0 / size as f32),
            edge: Edge::Wrap,
        };
        for ((weathered, moss), digest) in [(false, 0.0), (true, 0.0), (true, 0.03)]
            .into_iter()
            .zip(digests)
        {
            let mut cx = Context::new(grid, &NoResources);
            let m = cx
                .instantiate(
                    &GlazedBrickWall,
                    "wall",
                    Bind::new()
                        .flag("weathered", weathered)
                        .scalar("moss", moss),
                )
                .unwrap()
                .take_material("material")
                .unwrap();
            assert_eq!(
                m.digest(),
                digest,
                "{size} weathered={weathered} moss={moss}"
            );
            for &(path, expected) in fingerprints {
                let runs = !matches!(
                    path,
                    "wall/dirt" | "wall/streaks" | "wall/salts" | "wall/moss"
                ) || weathered && (path != "wall/moss" || moss > 0.0);
                assert_eq!(
                    cx.diagnostics().fingerprint_at(path),
                    runs.then_some(expected),
                    "{size} {path} weathered={weathered} moss={moss}"
                );
            }
        }
    }
}
