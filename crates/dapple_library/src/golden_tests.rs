// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Golden digests that give module versions a meaning.
//!
//! A [`ModuleId`] version promises behavior: packages pin modules "at exact
//! versions", and anything keyed on a version (a cache entry, a bake stamp)
//! trusts that the same `name@version` makes the same bits. Nothing in the
//! type system enforces that, so this test does: every module in
//! [`registry`] is instantiated at its declared defaults on a small fixed
//! grid, every shipped package ([`PACKAGES`]) at its defaults and at each
//! of its presets, and a digest of everything produced is compared with the
//! value recorded in [`GOLDEN`] for that `name@version`.
//!
//! The modules are enumerated from the registry and the packages from
//! [`PACKAGES`], not listed here, so a module registered without a golden
//! fails the test, as does a golden for a module that is gone. A digest
//! changes when a module's body changes; if that was intended, the
//! module's version must change with it (the version is part of the key,
//! and of every instance's seed derivation), and the new digest is recorded
//! under the new version.
//!
//! The digests are bit-exact on every platform CI covers (Linux, macOS,
//! Windows, wasm), like the field and raster goldens, because the math goes
//! through `libm`. A platform that disagrees has found a determinism bug;
//! it is not a reason to loosen this test.
//!
//! To record digests, read them from the failure message: it prints every
//! current entry in the form [`GOLDEN`] holds. `print_current_digests`
//! prints the same table on demand:
//! `cargo test -p dapple_library print_current_digests -- --ignored --nocapture`.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use dapple_field::hash::hash;
use dapple_field::{Domain, Edge, ImageLevel, PortType, Primaries, SampleImage, SamplePolicy};
use dapple_field::{Value, program::Fingerprint};
use dapple_material::module::{Bind, Context, Input, InputKind, Module, Output};
use dapple_material::resource::{
    Resolved, ResourceError, ResourceHost, ResourceRef, ResourceRequest,
};
use dapple_material::{Aux, Channel, Grid, Material, Param};
use dapple_package::Package;
use dapple_raster::typed::ReductionPolicy;
use glam::{Vec2, Vec3};

use crate::packages::{PACKAGES, registry};

/// The recorded digests: module or package name, version, case, digest.
///
/// A case is `defaults`, or `preset:<name>` for a package's preset. A
/// version bump adds new entries; the old version's entries go with it.
const GOLDEN: &[(&str, u32, &str, u64)] = &[
    (
        "dapple_library.ceramic_body",
        1,
        "defaults",
        0xb8182035ac72924a,
    ),
    ("dapple_library.mortar", 1, "defaults", 0x7d7af7ff89a382bd),
    ("dapple_library.stone", 1, "defaults", 0x2891423ad38f175c),
    ("dapple_library.wood", 1, "defaults", 0x33bde5d2b5d183bb),
    ("dapple_library.finish", 1, "defaults", 0x199c80122b1b655e),
    ("dapple_library.grime", 1, "defaults", 0x8a022ded6de17e71),
    ("dapple_library.streaks", 1, "defaults", 0x94b52e36dd387279),
    (
        "dapple_library.efflorescence",
        1,
        "defaults",
        0x1548703e5b27091f,
    ),
    (
        "dapple_library.edge_wear",
        1,
        "defaults",
        0xf7709cabebfff21a,
    ),
    ("dapple_library.moss", 1, "defaults", 0xd970f3d6124df69c),
    (
        "dapple_library.by_example",
        1,
        "defaults",
        0x4130a7c93225f720,
    ),
    (
        "dapple_library.glazed_brick_wall",
        2,
        "defaults",
        0xa4db69f4cfb93dfa,
    ),
    (
        "dapple_library.stone_sill",
        2,
        "defaults",
        0xa877a1bfb523886e,
    ),
    (
        "dapple_library.varnished_board",
        1,
        "defaults",
        0x43c3d549cb79c282,
    ),
    (
        "dapple_library.threshold",
        1,
        "defaults",
        0xa6d9d4156029ea32,
    ),
    (
        "dapple_library.beech_bark",
        1,
        "defaults",
        0x346317362ce957c2,
    ),
    (
        "dapple_library.birch_bark",
        1,
        "defaults",
        0xfcea1a15d71d907c,
    ),
    (
        "dapple_library.scots_pine_bark",
        1,
        "defaults",
        0x67bea30e55d153c1,
    ),
    (
        "dapple_library.spruce_bark",
        1,
        "defaults",
        0xcf5c79b39df4253d,
    ),
    (
        "dapple_library.ashlar_limestone",
        1,
        "defaults",
        0xa94903ea4bf45365,
    ),
    (
        "dapple_library.rubble_wall",
        1,
        "defaults",
        0x6b1a4deacb4a7ad7,
    ),
    (
        "dapple_library.flint_wall",
        1,
        "defaults",
        0x239ec636e97b33d1,
    ),
    (
        "dapple_library.roman_brick",
        1,
        "defaults",
        0xe682a0d026bdeafe,
    ),
    ("dapple_library.marble", 1, "defaults", 0x2970a8167ab7898d),
    (
        "dapple_library.terracotta_tile",
        1,
        "defaults",
        0x445412d937b81948,
    ),
    ("dapple_library.copper", 1, "defaults", 0xdf904f754a80358e),
    ("dapple_library.steel", 1, "defaults", 0xa6dbeb0c926ba2c3),
    ("dapple_library.patina", 1, "defaults", 0x3eac5c0178494337),
    ("dapple_library.rust", 1, "defaults", 0xd0f98a46b30f81f1),
    ("dapple_library.paint", 1, "defaults", 0x0e42c93bc55f180e),
    (
        "dapple_library.weathered_copper",
        1,
        "defaults",
        0x867a28acb988541a,
    ),
    (
        "dapple_library.rusted_steel",
        1,
        "defaults",
        0x352c406498500f55,
    ),
    (
        "dapple_library.painted_steel",
        1,
        "defaults",
        0xd0ee9d6099fe3027,
    ),
    (
        "dapple_library.packages.varnished_board",
        1,
        "defaults",
        0x43c3d549cb79c282,
    ),
    (
        "dapple_library.packages.varnished_board",
        1,
        "preset:dark_and_worn",
        0xfd49441ffd65982e,
    ),
    (
        "dapple_library.packages.weathered_copper",
        1,
        "defaults",
        0x867a28acb988541a,
    ),
    (
        "dapple_library.packages.weathered_copper",
        1,
        "preset:bright",
        0x3d3e7568e5e49ced,
    ),
    (
        "dapple_library.packages.weathered_copper",
        1,
        "preset:tarnished",
        0xc909e42dca7dd5fb,
    ),
    (
        "dapple_library.packages.weathered_copper",
        1,
        "preset:streaked",
        0x81f99a26a997a5cc,
    ),
    (
        "dapple_library.packages.weathered_copper",
        1,
        "preset:verdigris",
        0xc7fa2f4eb99ab0c2,
    ),
    (
        "dapple_library.packages.rusted_steel",
        1,
        "defaults",
        0x352c406498500f55,
    ),
    (
        "dapple_library.packages.rusted_steel",
        1,
        "preset:fresh",
        0x5af086edaa52cfbb,
    ),
    (
        "dapple_library.packages.rusted_steel",
        1,
        "preset:flash_rusted",
        0xa4bcf9b7702f82a0,
    ),
    (
        "dapple_library.packages.rusted_steel",
        1,
        "preset:rusting",
        0x6a967f935825f1e7,
    ),
    (
        "dapple_library.packages.rusted_steel",
        1,
        "preset:pitted",
        0x0dd4f932eeb91059,
    ),
    (
        "dapple_library.packages.painted_steel",
        1,
        "defaults",
        0xd0ee9d6099fe3027,
    ),
    (
        "dapple_library.packages.painted_steel",
        1,
        "preset:new",
        0x8b047104c5f9eda0,
    ),
    (
        "dapple_library.packages.painted_steel",
        1,
        "preset:chipped",
        0xbee5064634b028a2,
    ),
    (
        "dapple_library.packages.painted_steel",
        1,
        "preset:weeping",
        0x3083ec50a9cf3151,
    ),
    (
        "dapple_library.packages.painted_steel",
        1,
        "preset:derelict",
        0x6d4cef47910dd0e2,
    ),
];

/// A small fixed grid: 32² texels over one meter, tiling.
fn grid() -> Grid {
    Grid {
        width: 32,
        height: 32,
        origin: Vec2::ZERO,
        texel: Vec2::splat(1.0 / 32.0),
        edge: Edge::Wrap,
    }
}

/// The material given to every module that needs one: a raised square on a
/// flat ground with a base color, a coat and a roughness.
fn base() -> Material {
    let g = grid();
    let mut m = Material::new(g);
    let heights = (0..g.len()).map(|i| {
        let (x, y) = (i % 32, i / 32);
        let inside = (8..24).contains(&x) && (8..24).contains(&y);
        Value::Scalar(if inside { 0.005 } else { 0.0 })
    });
    m.set_aux(
        Aux::Height,
        Channel::Map(g.typed(PortType::Scalar, heights).unwrap()),
    )
    .unwrap();
    m.set_param(
        Param::BaseColor,
        Channel::Constant(Value::Vector3(Vec3::new(0.4, 0.3, 0.2))),
    )
    .unwrap();
    m.set_param(Param::CoatWeight, Channel::Constant(Value::Scalar(1.0)))
        .unwrap();
    m.set_param(
        Param::SpecularRoughness,
        Channel::Constant(Value::Scalar(0.5)),
    )
    .unwrap();
    m
}

/// A host serving one color exemplar under any name: 48² texels of 5 mm in
/// blocks of color.
struct Fixtures;

impl ResourceHost for Fixtures {
    fn resolve(&self, _: &ResourceRef) -> Result<Resolved, ResourceError> {
        let n = 48_u32;
        let values: Vec<f32> = (0..n * n)
            .flat_map(|i| {
                let block = (i % n) / 4 + 100 * ((i / n) / 4);
                let u = (hash(5, &[u64::from(block)]) % 1000) as f32 / 1000.0;
                [0.3 + 0.4 * u, 0.25 + 0.3 * u, 0.2 + 0.1 * u * u]
            })
            .collect();
        let level = ImageLevel::with_channels(n, n, Vec2::splat(0.005), 3, values).unwrap();
        let image = SampleImage::typed(
            PortType::Color(Primaries::Rec709),
            SamplePolicy::Linear,
            Domain::Plane,
            Vec2::ZERO,
            alloc::vec![level],
            Fingerprint(11),
        )
        .unwrap();
        Ok(Resolved {
            content: Fingerprint(11),
            image,
            mips: ReductionPolicy::Average,
        })
    }
}

/// Binds every required input of `module` to a fixture; optional inputs
/// stay unbound, so the digest is of the module's defaults.
fn bind_required(module: &dyn Module, mut bind: Bind) -> Bind {
    let id = module.interface().id;
    for input in module.interface().inputs.iter().filter(|i| i.required) {
        let fixture = match input.kind {
            InputKind::Material => Input::Material(base()),
            InputKind::Resource(ResourceRequest {
                port: PortType::Color(_),
                ..
            }) => Input::Resource(ResourceRef("fixture".into())),
            other => panic!(
                "{id} requires input {:?} of kind {other:?}, which has no golden fixture: \
                 add one to `bind_required` in golden_tests.rs",
                input.name
            ),
        };
        bind = bind.input(&input.name, fixture);
    }
    bind
}

/// A digest of everything `module` produces under `bind`, outputs in
/// their declared order.
fn digest(module: &dyn Module, bind: Bind) -> u64 {
    let mut cx = Context::new(grid(), &Fixtures);
    let outputs = cx
        .instantiate(module, "golden", bind_required(module, bind))
        .unwrap_or_else(|e| panic!("{}: {e}", module.interface().id));
    let mut words = Vec::new();
    for decl in module.interface().outputs {
        words.push(hash(1, &[decl.name.len() as u64]));
        words.push(match outputs.get(&decl.name) {
            Some(Output::Material(m)) => m.digest(),
            Some(Output::Map(r)) => r.digest(),
            None => panic!("{}: no output {:?}", module.interface().id, decl.name),
        });
    }
    hash(2, &words)
}

/// Every current entry: `(name, version, case, digest)`, modules in
/// registry order, then packages with their presets.
fn current() -> Vec<(String, u32, String, u64)> {
    let engine = registry();
    let mut out = Vec::new();
    for module in engine.modules() {
        let id = module.interface().id;
        let a = digest(module.as_ref(), Bind::new());
        // Instantiating twice must give the same bits: a golden of a
        // nondeterministic module would be noise.
        assert_eq!(
            a,
            digest(module.as_ref(), Bind::new()),
            "{id} is not deterministic"
        );
        out.push((id.name.to_string(), id.version, "defaults".to_string(), a));
    }
    for source in PACKAGES {
        let package = Package::from_json(source).expect("a shipped package reads");
        let compiled = engine
            .compile(&package)
            .expect("a shipped package compiles");
        let id = compiled.interface().id;
        out.push((
            id.name.to_string(),
            id.version,
            "defaults".to_string(),
            digest(&compiled, Bind::new()),
        ));
        for preset in &package.presets {
            let bind = compiled.preset(&preset.name).expect("a declared preset");
            out.push((
                id.name.to_string(),
                id.version,
                format!("preset:{}", preset.name),
                digest(&compiled, bind),
            ));
        }
    }
    out
}

/// The table of `entries` as [`GOLDEN`] holds it, ready to paste.
fn table(entries: &[(String, u32, String, u64)]) -> String {
    let mut s = String::new();
    for (name, version, case, d) in entries {
        s.push_str(&format!(
            "    ({name:?}, {version}, {case:?}, {d:#018x}),\n"
        ));
    }
    s
}

#[test]
fn every_module_and_package_matches_its_golden_digest() {
    let entries = current();
    let mut problems = Vec::new();
    for (name, version, case, d) in &entries {
        let golden = GOLDEN
            .iter()
            .find(|(n, v, c, _)| n == name && v == version && c == case);
        match golden {
            Some(&(.., g)) if g == *d => {}
            Some(&(.., g)) => problems.push(format!(
                "{name}@{version} ({case}) now digests to {d:#018x}, recorded {g:#018x}: \
                 its output changed without a new version. If the change was unintended, \
                 undo it. If it was intended, bump the version of {name} (its `ModuleId`, \
                 and every package that pins it) and record the new digest under the \
                 new version."
            )),
            None => problems.push(format!(
                "{name}@{version} ({case}) has no golden digest ({d:#018x}). If the module \
                 is new, or its version was just bumped, record it."
            )),
        }
    }
    for &(name, version, case, _) in GOLDEN {
        if !entries
            .iter()
            .any(|(n, v, c, _)| n == name && *v == version && c == case)
        {
            problems.push(format!(
                "golden {name}@{version} ({case}) matches no registered module, package or \
                 preset: remove it (a module was removed, or its version moved on)."
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "module goldens out of date:\n  {}\n\nThe current table, to replace `GOLDEN` in \
         golden_tests.rs:\n{}",
        problems.join("\n  "),
        table(&entries)
    );
}

/// Prints the current table of digests, to record after a deliberate
/// change or a new module: see the [module docs](self).
#[test]
#[ignore = "prints the current golden table; run on demand with --nocapture"]
fn print_current_digests() {
    extern crate std;
    std::println!("{}", table(&current()));
}

/// The golden table has no two entries for one key, and registered modules
/// have distinct identities (a registry would silently replace one).
#[test]
fn identities_are_unique() {
    let engine = registry();
    let ids: Vec<_> = engine.modules().map(|m| m.interface().id).collect();
    for (i, a) in ids.iter().enumerate() {
        assert!(
            ids[i + 1..].iter().all(|b| b.name != a.name),
            "{a}: two registered modules share a name"
        );
        assert!(a.version >= 1, "{a} is registered without a version");
    }
    for (i, a) in GOLDEN.iter().enumerate() {
        assert!(
            GOLDEN[i + 1..]
                .iter()
                .all(|b| (a.0, a.1, a.2) != (b.0, b.1, b.2)),
            "{a:?} is recorded twice"
        );
    }
}
