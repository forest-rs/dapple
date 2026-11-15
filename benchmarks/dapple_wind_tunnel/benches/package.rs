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

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use dapple_library::packages::{PACKAGES, PAINTED_STEEL, registry};
use dapple_logic::Value;
use dapple_package::Package;

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
}

criterion_group!(benches, bench_package);
criterion_main!(benches);
