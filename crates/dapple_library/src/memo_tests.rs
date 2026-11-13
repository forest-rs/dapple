// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Seeded edit histories built with and without an instance cache, checked
//! bit for bit.
//!
//! A history edits real library assets the way an author or a fit does:
//! one parameter at a time, a preset switched on or off, another asset, a
//! new seed, a reset to defaults, another grid. After every step the asset
//! is built three ways: without a cache, with a cache kept for the whole
//! history, and with one kept as long but with a budget too small for one
//! build, so results are evicted all the time. The assets nest modules up
//! to four deep (a sill over a glazed wall over its glaze, wear and
//! weathering), and the packaged varnished board is a data-defined module
//! over the same modules as the native one. Every output must equal the
//! uncached build's, bit for bit, and so must every diagnostics entry,
//! fingerprints included; a build that fails must fail the same way.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use dapple_field::Edge;
use dapple_material::Grid;
use dapple_material::module::{
    Bind, Context, Diagnostics, InstanceCache, Module, ModuleError, Output, Outputs, ParamValue,
};
use dapple_material::resource::NoResources;
use glam::{Vec2, Vec3};

use crate::modules::{GlazedBrickWall, RomanBrick, StoneSill, VarnishedBoard};
use crate::packages::{VARNISHED_BOARD, registry};

/// Histories checked, one per seed; set `DAPPLE_MEMO_HISTORIES` when
/// building the tests to check more.
fn seeds() -> u64 {
    option_env!("DAPPLE_MEMO_HISTORIES").map_or(3, |n| n.parse().expect("a number of histories"))
}

/// Steps per history.
const STEPS: usize = 32;

/// A budget smaller than one build of any asset here, so every build
/// evicts.
const TIGHT: usize = 96 * 1024;

/// `SplitMix64`: small, deterministic, and plenty for picking edits.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % n as u64).expect("below n")
    }

    fn pick<T: Clone>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())].clone()
    }
}

/// A one-meter wrapping tile, `n` texels a side: every asset builds on it.
fn tile(n: u32) -> Grid {
    #[expect(clippy::cast_precision_loss, reason = "small grid sizes")]
    let texel = Vec2::splat(1.0 / n as f32);
    Grid {
        width: n,
        height: n,
        origin: Vec2::ZERO,
        texel,
        edge: Edge::Wrap,
    }
}

/// A clamping strip, for the assets that do not need a wrapping tile.
fn strip() -> Grid {
    Grid {
        width: 24,
        height: 16,
        origin: Vec2::new(0.25, -0.5),
        texel: Vec2::splat(1.0 / 24.0),
        edge: Edge::Clamp,
    }
}

/// Grids that differ from the one before in one field each: the edge,
/// then the origin, then the texel size. A result kept for one must not
/// be found for the next.
fn one_field_apart() -> [Grid; 3] {
    let clamped = Grid {
        edge: Edge::Clamp,
        ..tile(16)
    };
    let moved = Grid {
        origin: Vec2::new(0.5, 0.0),
        ..clamped
    };
    let finer = Grid {
        texel: Vec2::splat(1.0 / 32.0),
        ..moved
    };
    [clamped, moved, finer]
}

/// An asset, the edits it takes, and the grids it builds on.
struct Asset {
    module: Box<dyn Module>,
    /// Each parameter with a few values to choose from, so that edits come
    /// back to earlier values and find them kept.
    params: Vec<(&'static str, Vec<ParamValue>)>,
    presets: Vec<Bind>,
    grids: Vec<Grid>,
}

fn color(r: f32, g: f32, b: f32) -> ParamValue {
    ParamValue::Color(Vec3::new(r, g, b))
}

fn scalars(values: &[f32]) -> Vec<ParamValue> {
    values.iter().map(|&v| ParamValue::Scalar(v)).collect()
}

fn board_params() -> Vec<(&'static str, Vec<ParamValue>)> {
    vec![
        (
            "varnish",
            vec![color(0.78, 0.52, 0.25), color(0.55, 0.3, 0.12)],
        ),
        ("dirt", scalars(&[0.0, 0.08, 0.3])),
    ]
}

fn assets() -> Vec<Asset> {
    let compiled = registry()
        .compile(&dapple_package::Package::from_json(VARNISHED_BOARD).expect("reads"))
        .expect("compiles");
    let worn = compiled.preset("dark_and_worn").expect("a preset");
    let seeds = vec![ParamValue::Seed(0), ParamValue::Seed(5)];
    let mut board = board_params();
    board.push(("seed", seeds.clone()));
    vec![
        Asset {
            module: Box::new(VarnishedBoard),
            params: board,
            presets: vec![],
            grids: [tile(16), tile(24), strip()]
                .into_iter()
                .chain(one_field_apart())
                .collect(),
        },
        Asset {
            module: Box::new(compiled),
            params: board_params(),
            presets: vec![worn],
            grids: vec![tile(16), strip()],
        },
        Asset {
            module: Box::new(GlazedBrickWall),
            params: vec![
                ("field", vec![color(0.74, 0.64, 0.44), color(0.2, 0.5, 0.3)]),
                (
                    "dado_courses",
                    vec![ParamValue::Integer(2), ParamValue::Integer(4)],
                ),
                ("battered", scalars(&[0.0, 0.4])),
                ("glaze_roughness", scalars(&[0.035, 0.2])),
                (
                    "weathered",
                    vec![ParamValue::Flag(true), ParamValue::Flag(false)],
                ),
                ("dirt", scalars(&[0.1, 0.35])),
                ("moss", scalars(&[0.0, 0.03])),
                ("seed", seeds.clone()),
            ],
            presets: vec![],
            grids: vec![tile(24), tile(32)],
        },
        Asset {
            module: Box::new(RomanBrick),
            params: vec![
                ("joint", scalars(&[0.015, 0.025])),
                ("roughness", scalars(&[0.5, 0.8])),
                (
                    "mortar_color",
                    vec![color(0.44, 0.36, 0.3), color(0.6, 0.6, 0.55)],
                ),
                ("seed", seeds.clone()),
            ],
            presets: vec![],
            grids: vec![tile(16), strip()],
        },
        Asset {
            module: Box::new(StoneSill),
            params: vec![
                ("projection", scalars(&[0.0, 0.012])),
                ("dirt", scalars(&[0.1, 0.3])),
                ("streaks", scalars(&[0.0, 0.22])),
                ("seed", seeds),
            ],
            presets: vec![],
            grids: vec![tile(24)],
        },
    ]
}

/// What an author has set on one asset.
#[derive(Clone, Debug, Default)]
struct Setting {
    params: Vec<(&'static str, ParamValue)>,
    preset: Option<usize>,
    grid: usize,
}

/// The preset's bindings, if one is on, then the parameters set since.
fn bind(asset: &Asset, setting: &Setting) -> Bind {
    let mut b = setting
        .preset
        .map_or_else(Bind::new, |i| asset.presets[i].clone());
    for (name, value) in &setting.params {
        b = b.param(name, *value);
    }
    b
}

type Built = (Result<Outputs, ModuleError>, Diagnostics);

fn build(asset: &Asset, setting: &Setting, cache: Option<&mut InstanceCache>) -> Built {
    let mut cx = Context::new(asset.grids[setting.grid], &NoResources);
    if let Some(cache) = cache {
        cx = cx.with_cache(cache);
    }
    let out = cx.instantiate(&*asset.module, "asset", bind(asset, setting));
    (out, cx.into_diagnostics())
}

/// Every output's digest, in output order.
fn digests(outputs: &Outputs) -> Vec<u64> {
    outputs
        .iter()
        .map(|(_, o)| match o {
            Output::Material(m) => m.digest(),
            Output::Map(r) => r.digest(),
        })
        .collect()
}

fn check(context: &str, expected: &Built, got: &Built) {
    match (&expected.0, &got.0) {
        (Ok(a), Ok(b)) => {
            assert_eq!(digests(a), digests(b), "{context}: output bits");
            assert!(a == b, "{context}: outputs");
        }
        (a, b) => assert_eq!(a, b, "{context}: the same failure"),
    }
    assert_eq!(expected.1, got.1, "{context}: diagnostics");
}

#[test]
fn cached_builds_equal_uncached_builds_through_edit_histories() {
    let assets = assets();
    let (mut hits, mut evictions, mut nested_hits) = (0, 0, 0);
    let (mut built, mut succeeded) = (0, 0);
    let outer = [
        "dapple_library.varnished_board",
        "dapple_library.packages.varnished_board",
        "dapple_library.glazed_brick_wall",
        "dapple_library.roman_brick",
        "dapple_library.stone_sill",
    ];
    for seed in 0..seeds() {
        let mut rng = Rng(seed);
        let mut settings: Vec<Setting> = assets.iter().map(|_| Setting::default()).collect();
        let mut current = 0;
        let mut kept = InstanceCache::new(usize::MAX);
        let mut tight = InstanceCache::new(TIGHT);
        let mut log: Vec<String> = Vec::new();
        for step in 0..STEPS {
            let asset = &assets[current];
            let setting = &mut settings[current];
            let action = match rng.below(20) {
                0..=10 => {
                    let (name, values) = rng.pick(&asset.params);
                    let value = rng.pick(&values);
                    setting.params.retain(|(n, _)| *n != name);
                    setting.params.push((name, value));
                    format!("set {name} = {value:?}")
                }
                11..=13 => {
                    current = rng.below(assets.len());
                    format!("switch to asset {current}")
                }
                14..=15 => {
                    setting.grid = rng.below(asset.grids.len());
                    format!("grid {}", setting.grid)
                }
                16 if !asset.presets.is_empty() => {
                    setting.preset = match setting.preset {
                        Some(_) => None,
                        None => Some(rng.below(asset.presets.len())),
                    };
                    format!("preset {:?}", setting.preset)
                }
                17 => {
                    *setting = Setting::default();
                    String::from("reset")
                }
                _ => String::from("build again"),
            };
            log.push(action);
            let asset = &assets[current];
            let setting = &settings[current];
            let plain = build(asset, setting, None);
            let context = format!("seed {seed} step {step}: {log:?}");
            built += 1;
            succeeded += usize::from(plain.0.is_ok());
            check(&context, &plain, &build(asset, setting, Some(&mut kept)));
            check(&context, &plain, &build(asset, setting, Some(&mut tight)));
        }
        let (k, t) = (kept.stats(), tight.stats());
        assert!(t.bytes <= TIGHT, "{t}");
        hits += k.hits;
        evictions += t.evictions;
        nested_hits += k
            .modules
            .iter()
            .filter(|m| !outer.contains(&m.module.name.as_ref()))
            .map(|m| m.hits)
            .sum::<u64>();
    }
    // The histories exercised what they claim to. A build may fail (a
    // glazed wall can break its tiling promise on a tile too small for its
    // bricks), and must then fail the same way with a cache; most must
    // not.
    assert!(4 * succeeded >= 3 * built, "{succeeded} of {built} built");
    assert!(hits > 0, "no result was ever reused");
    assert!(nested_hits > 0, "no nested result was ever reused");
    assert!(evictions > 0, "the tight budget never evicted");
}

/// Every grid field is part of what a result is kept by: the board built in
/// turn on grids one field apart, through one cache, matches each uncached
/// build, at the top and at every nested instance.
#[test]
fn grids_one_field_apart_keep_apart() {
    let assets = assets();
    let board = &assets[0];
    let mut cache = InstanceCache::new(usize::MAX);
    // The wrapping tile, then the three grids one field apart from it.
    let n = board.grids.len();
    for _ in 0..2 {
        for grid in [0, n - 3, n - 2, n - 1] {
            let setting = Setting {
                grid,
                ..Setting::default()
            };
            let plain = build(board, &setting, None);
            assert!(plain.0.is_ok());
            let cached = build(board, &setting, Some(&mut cache));
            check(&format!("grid {grid}"), &plain, &cached);
        }
    }
    let s = cache.stats();
    assert_eq!((s.misses, s.hits), (16, 4), "{s}");
}
