// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The material layer on real library modules.
//!
//! Two modules stand for the library: `varnished_board` (oak, a varnish
//! finish and grime, an asset composing several modules) and `roman_brick`
//! (masonry on a running-bond layout). Both are built on a wrapping 256 by
//! 256 grid with their default parameters.
//!
//! - `material/instantiate/<module>`: `Context::instantiate`, building the
//!   whole material from nothing.
//! - `material/lower_maps/<module>`: `lower::maps`, the material's channels
//!   as images for packing.
//! - `material/pack_gltf/<module>`: `dapple_encode::pack` of those maps for
//!   the glTF profile, with mip chains.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use dapple_encode::{PackSettings, Profile, pack};
use dapple_field::Edge;
use dapple_library::modules::{RomanBrick, VarnishedBoard};
use dapple_material::module::{Bind, Context, Module};
use dapple_material::resource::NoResources;
use dapple_material::{Grid, Material, lower};
use glam::Vec2;

const SIZE: u32 = 256;

fn grid() -> Grid {
    #[expect(clippy::cast_precision_loss, reason = "a small grid size")]
    let texel = Vec2::splat(1.0 / SIZE as f32);
    Grid {
        width: SIZE,
        height: SIZE,
        origin: Vec2::ZERO,
        texel,
        edge: Edge::Wrap,
    }
}

fn instantiate(module: &dyn Module) -> Material {
    let mut cx = Context::new(grid(), &NoResources);
    cx.instantiate(module, "bench", Bind::new())
        .expect("the module instantiates")
        .take_material("material")
        .expect("the module returns a material")
}

fn bench_material(c: &mut Criterion) {
    let modules: [(&str, &dyn Module); 2] = [
        ("varnished_board", &VarnishedBoard),
        ("roman_brick", &RomanBrick),
    ];
    let texels = Throughput::Elements(u64::from(SIZE) * u64::from(SIZE));

    let mut group = c.benchmark_group("material/instantiate");
    group.sample_size(10);
    group.throughput(texels.clone());
    for (name, module) in modules {
        group.bench_function(name, |b| b.iter(|| instantiate(module)));
    }
    group.finish();

    let mut group = c.benchmark_group("material/lower_maps");
    group.throughput(texels.clone());
    for (name, module) in modules {
        let material = instantiate(module);
        group.bench_function(name, |b| {
            b.iter(|| lower::maps(&material).expect("lowers"));
        });
    }
    group.finish();

    let mut group = c.benchmark_group("material/pack_gltf");
    group.throughput(texels);
    for (name, module) in modules {
        let (maps, _) = lower::maps(&instantiate(module)).expect("lowers");
        let settings = PackSettings::default();
        group.bench_function(name, |b| {
            b.iter(|| pack(&maps, Profile::Gltf, &settings).expect("packs"));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_material);
criterion_main!(benches);
