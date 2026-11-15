// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Packages: compiling the library's packages, and running a package's
//! expressions.
//!
//! - `package/compile/<package>`: `Registry::compile` of each package the
//!   library ships, from its parsed source. `painted_steel` has expression
//!   bindings, so its time includes checking them and compiling and
//!   verifying their tape program; the others have none.
//! - `package/logic/painted_steel`: one run of `painted_steel`'s expression
//!   program at the package's defaults, the work every instance of the
//!   package does before its steps.
//! - `package/instantiate/varnished_board/<case>`: `Context::instantiate`
//!   of the compiled varnished-board package on a wrapping 256 by 256 grid
//!   (as `material/instantiate`), at its defaults and with its
//!   `dark_and_worn` preset.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use dapple_field::Edge;
use dapple_library::packages::{PACKAGES, PAINTED_STEEL, VARNISHED_BOARD, registry};
use dapple_logic::Value;
use dapple_material::Grid;
use dapple_material::module::{Bind, Context};
use dapple_material::resource::NoResources;
use dapple_package::Package;
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

fn bench_package(c: &mut Criterion) {
    let engine = registry();
    let mut group = c.benchmark_group("package/compile");
    for source in PACKAGES {
        let package = Package::from_json(source).expect("a shipped package reads");
        let name = package
            .interface
            .module
            .name
            .rsplit('.')
            .next()
            .unwrap_or_default()
            .to_owned();
        group.bench_function(name, |b| {
            b.iter(|| engine.compile(black_box(&package)).expect("compiles"));
        });
    }
    group.finish();

    let package = Package::from_json(PAINTED_STEEL).expect("reads");
    let compiled = engine.compile(&package).expect("compiles");
    let program = compiled.logic().expect("painted steel has expressions");
    // The defaults of the scalars the program reads, in interface order.
    let inputs: Vec<Value> = package
        .interface
        .params
        .iter()
        .filter_map(|p| match p.ty.kind().1 {
            dapple_material::module::ParamValue::Scalar(v) => Some(Value::Scalar(v)),
            _ => None,
        })
        .collect();
    let mut group = c.benchmark_group("package/logic");
    group.bench_function("painted_steel", |b| {
        b.iter(|| program.run(black_box(&inputs)).expect("runs"));
    });
    group.finish();

    let board = engine
        .compile(&Package::from_json(VARNISHED_BOARD).expect("reads"))
        .expect("compiles");
    let mut group = c.benchmark_group("package/instantiate/varnished_board");
    group.sample_size(10);
    group.throughput(Throughput::Elements(u64::from(SIZE) * u64::from(SIZE)));
    let worn = board.preset("dark_and_worn").expect("a preset");
    for (name, bind) in [("defaults", Bind::new()), ("dark_and_worn", worn)] {
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut cx = Context::new(grid(), &NoResources);
                cx.instantiate(&board, "bench", black_box(bind.clone()))
                    .expect("the package instantiates")
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_package);
criterion_main!(benches);
