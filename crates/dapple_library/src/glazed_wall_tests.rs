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

mod package {
    use alloc::format;
    use alloc::string::String;
    use alloc::vec::Vec;

    use dapple_material::Material;
    use dapple_material::module::{Diagnostics, InstanceCache, Module, ModuleError};
    use dapple_package::Package;

    use super::*;
    use crate::packages::{GLAZED_BRICK_WALL, registry};

    fn grid(size: u32) -> Grid {
        Grid {
            width: size,
            height: size,
            origin: Vec2::ZERO,
            texel: Vec2::splat(1.0 / size as f32),
            edge: Edge::Wrap,
        }
    }

    fn build(
        module: &dyn Module,
        grid: Grid,
        path: &str,
        bind: Bind,
        cache: Option<&mut InstanceCache>,
    ) -> Result<(Material, Diagnostics), ModuleError> {
        let mut cx = Context::new(grid, &NoResources);
        if let Some(cache) = cache {
            cx = cx.with_cache(cache);
        }
        let m = cx
            .instantiate(module, path, bind)?
            .take_material("material")
            .expect("a wall");
        Ok((m, cx.into_diagnostics()))
    }

    fn parity(module: &dyn Module, grid: Grid, path: &str, bind: Bind) -> Diagnostics {
        let (packaged, package_d) =
            build(module, grid, path, bind.clone(), None).expect("package builds");
        let (native, native_d) =
            build(&GlazedBrickWall, grid, path, bind, None).expect("native builds");
        assert_eq!(packaged.digest(), native.digest(), "{path}");
        assert!(packaged == native, "all channels and metadata at {path}");
        // Only the parent has a different identity and definition. Every
        // child event includes its path, module version and fingerprint,
        // skipped steps included; operation reports agree as well.
        assert_eq!(package_d.entries[1..], native_d.entries[1..], "{path}");
        package_d
    }

    #[test]
    fn glazed_wall_package_matches_native_defaults_and_every_preset() {
        let package = Package::from_json(GLAZED_BRICK_WALL).expect("reads");
        let again = Package::from_json(&package.to_json()).expect("round trips");
        assert_eq!(package, again);
        assert_eq!(package.fingerprint(), again.fingerprint());
        let engine = registry();
        let compiled = engine.compile(&package).expect("compiles");
        assert_eq!(
            compiled.fingerprint(),
            engine.compile(&again).unwrap().fingerprint()
        );
        let params = |m: &dyn Module| {
            m.interface()
                .params
                .into_iter()
                .map(|p| (p.name, p.kind, p.default))
                .collect::<Vec<_>>()
        };
        assert_eq!(params(&compiled), params(&GlazedBrickWall));
        for size in [32, 64] {
            parity(&compiled, grid(size), "wall", Bind::new());
            for preset in &package.presets {
                parity(
                    &compiled,
                    grid(size),
                    "wall",
                    compiled.preset(&preset.name).expect("preset"),
                );
            }
        }
    }

    #[test]
    fn glazed_wall_package_guards_and_fingerprints_match_across_parameters() {
        let compiled = registry()
            .compile(&Package::from_json(GLAZED_BRICK_WALL).unwrap())
            .unwrap();
        for weathered in [false, true] {
            for moss in [0.0, 0.000001, 0.25] {
                for profile in 0..3 {
                    for path in ["wall", "scene/wall"] {
                        let mut bind = Bind::new()
                            .flag("weathered", weathered)
                            .scalar("moss", moss);
                        if profile == 1 {
                            bind = bind
                                .integer("dado_courses", 0)
                                .integer("band_courses", 0)
                                .scalar("variation", 0.0)
                                .scalar("battered", 0.0)
                                .scalar("glaze_roughness", 0.0)
                                .scalar("dirt", 0.0)
                                .scalar("streaks", 0.0)
                                .scalar("salts", 0.0)
                                .seed(5);
                        } else if profile == 2 {
                            bind = bind
                                .color("field", glam::Vec3::new(0.04, 0.078, 0.042))
                                .color("band", glam::Vec3::new(0.105, 0.017, 0.012))
                                .color("dado", glam::Vec3::new(0.74, 0.64, 0.44))
                                .integer("dado_courses", 14)
                                .integer("band_courses", 14)
                                .scalar("variation", 0.8)
                                .scalar("battered", 0.7)
                                .scalar("glaze_roughness", 0.3)
                                .scalar("dirt", 0.6)
                                .scalar("streaks", 0.2)
                                .scalar("salts", 0.1)
                                .seed(u64::MAX);
                        }
                        let d = parity(&compiled, grid(64), path, bind);
                        let mut names = alloc::vec![
                            "structure",
                            "body",
                            "mortar",
                            "bricks",
                            "bedded",
                            "region",
                            "glaze",
                            "worn",
                            "wear"
                        ];
                        let mut skipped = Vec::new();
                        if weathered {
                            names.extend(["dirt", "streaks", "salts"]);
                        } else {
                            skipped.extend(["dirt", "streaks", "salts"]);
                        }
                        if weathered && moss > 0.0 {
                            names.push("moss");
                        } else {
                            skipped.push("moss");
                        }
                        let paths = |names: Vec<&str>| {
                            names
                                .into_iter()
                                .map(|n| format!("{path}/{n}"))
                                .collect::<Vec<String>>()
                        };
                        assert_eq!(
                            d.instances().skip(1).map(|(p, _)| p).collect::<Vec<_>>(),
                            paths(names)
                        );
                        assert_eq!(
                            d.skipped().map(|(p, _)| p).collect::<Vec<_>>(),
                            paths(skipped)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn glazed_wall_package_replays_diagnostics_and_skips_without_cache_entries() {
        let compiled = registry()
            .compile(&Package::from_json(GLAZED_BRICK_WALL).unwrap())
            .unwrap();
        let mut cache = InstanceCache::new(usize::MAX);
        for (weathered, moss) in [(false, 0.03), (true, 0.0), (true, 0.03)] {
            let bind = Bind::new()
                .flag("weathered", weathered)
                .scalar("moss", moss);
            let uncached = build(&compiled, grid(32), "wall", bind.clone(), None).unwrap();
            let cached =
                build(&compiled, grid(32), "wall", bind.clone(), Some(&mut cache)).unwrap();
            assert_eq!(uncached, cached);
            let misses = cache.stats().misses;
            assert_eq!(
                cached,
                build(&compiled, grid(32), "wall", bind, Some(&mut cache)).unwrap()
            );
            assert_eq!(cache.stats().misses, misses, "the whole instance replays");
            let stats = cache.stats();
            for module in ["grime", "streaks", "efflorescence", "moss"] {
                let misses = stats
                    .modules
                    .iter()
                    .find(|s| s.module.name == format!("dapple_library.{module}"))
                    .map_or(0, |s| s.misses);
                assert_eq!(
                    misses,
                    u64::from(weathered && (module != "moss" || moss > 0.0)),
                    "{module}"
                );
            }
        }
    }

    #[test]
    fn glazed_wall_package_rejects_the_same_unsupported_grids() {
        let compiled = registry()
            .compile(&Package::from_json(GLAZED_BRICK_WALL).unwrap())
            .unwrap();
        let unit = grid(32);
        for grid in [
            Grid {
                edge: Edge::Clamp,
                ..unit
            },
            Grid {
                texel: Vec2::splat(0.0625),
                ..unit
            },
            Grid {
                origin: Vec2::ONE,
                ..unit
            },
        ] {
            let a = build(&compiled, grid, "wall", Bind::new(), None).unwrap_err();
            let b = build(&GlazedBrickWall, grid, "wall", Bind::new(), None).unwrap_err();
            assert_eq!(a, b);
            assert_eq!(a.path, "wall/structure");
        }
    }
}
