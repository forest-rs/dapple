// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use dapple_field::{Edge, Value};
use dapple_material::module::{
    Args, Bind, Context, InputDecl, InputKind, Interface, Module, ModuleError, ModuleErrorKind,
    ModuleId, Output, OutputDecl, OutputKind, Outputs, ParamDecl, ParamKind, ParamValue, Unit,
};
use dapple_material::resource::NoResources;
use dapple_material::{Channel, Grid, Material, Param};
use glam::{Vec2, Vec3};

use crate::source::{InterfaceSource, PresetSet};
use crate::{Capability, FORMAT_VERSION, Package, PackageError, Preset, Registry, ValueSource};

fn grid() -> Grid {
    Grid {
        width: 8,
        height: 8,
        origin: Vec2::ZERO,
        texel: Vec2::splat(0.125),
        edge: Edge::Wrap,
    }
}

fn material_out() -> Vec<OutputDecl> {
    vec![OutputDecl {
        name: "material".into(),
        kind: OutputKind::Material,
        doc: "".into(),
    }]
}

/// Paint whose roughness depends on its instance seed.
struct Paint;

impl Module for Paint {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId::new("test.paint", 1),
            doc: "paint".into(),
            params: vec![
                ParamDecl {
                    name: "color".into(),
                    kind: ParamKind::Color,
                    default: ParamValue::Color(Vec3::splat(0.5)),
                    doc: "".into(),
                },
                ParamDecl {
                    name: "seed".into(),
                    kind: ParamKind::Seed,
                    default: ParamValue::Seed(0),
                    doc: "".into(),
                },
            ],
            inputs: vec![],
            outputs: material_out(),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let mut m = Material::new(cx.grid());
        m.set_param(
            Param::BaseColor,
            Channel::Constant(Value::Vector3(args.color("color"))),
        )?;
        #[expect(clippy::cast_precision_loss, reason = "a test value")]
        let r = (args.seed("roughness") % 1000) as f32 / 1000.0;
        m.set_param(
            Param::SpecularRoughness,
            Channel::Constant(Value::Scalar(r)),
        )?;
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}

/// Darkens a material.
struct Dirt;

impl Module for Dirt {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId::new("test.dirt", 1),
            doc: "dirt".into(),
            params: vec![ParamDecl {
                name: "amount".into(),
                kind: ParamKind::Scalar {
                    unit: Unit::Fraction,
                    range: [0.0, 1.0],
                },
                default: ParamValue::Scalar(0.2),
                doc: "".into(),
            }],
            inputs: vec![InputDecl {
                name: "base".into(),
                kind: InputKind::Material,
                required: true,
                doc: "".into(),
            }],
            outputs: material_out(),
        }
    }

    fn build(&self, _cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let mut m = args
            .material("base")
            .cloned()
            .ok_or(ModuleError::body(ModuleErrorKind::Body("base")))?;
        let c = match m.param(Param::BaseColor) {
            Some(Channel::Constant(Value::Vector3(c))) => *c,
            _ => Vec3::ONE,
        };
        m.set_param(
            Param::BaseColor,
            Channel::Constant(Value::Vector3(c * (1.0 - args.scalar("amount")))),
        )?;
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}

/// The native composition the package below reproduces.
struct DirtyPaint;

impl Module for DirtyPaint {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId::new("test.dirty_paint", 1),
            doc: "dirty paint".into(),
            params: vec![],
            inputs: vec![],
            outputs: material_out(),
        }
    }

    fn build(&self, cx: &mut Context<'_>, _args: &Args) -> Result<Outputs, ModuleError> {
        let mut p = cx.instantiate(
            &Paint,
            "paint",
            Bind::new().color("color", Vec3::new(0.8, 0.6, 0.4)),
        )?;
        let m = p.take_material("material").expect("paint");
        let mut d = cx.instantiate(
            &Dirt,
            "dirt",
            Bind::new().material("base", m).scalar("amount", 0.25),
        )?;
        let m = d.take_material("material").expect("dirt");
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}

const DIRTY_PAINT: &str = r#"{
  "format": "dapple.package",
  "version": 1,
  "interface": {
    "module": { "name": "test.dirty_paint_package", "version": 1 },
    "doc": "dirty paint, as a package",
    "params": [
      {
        "name": "dirt",
        "doc": "how dirty",
        "type": { "scalar": { "unit": "fraction", "range": [0.0, 1.0], "default": 0.25 } }
      }
    ],
    "outputs": [
      {
        "name": "material",
        "type": "material",
        "semantics": { "channels": ["base_color", "specular_roughness"], "tiling": "both" }
      }
    ]
  },
  "requires": {
    "capabilities": [
      { "name": "dapple.graph", "version": 1 },
      { "name": "dapple.semantics", "version": 1 }
    ],
    "modules": [
      { "name": "test.paint", "version": 1 },
      { "name": "test.dirt", "version": 1 }
    ]
  },
  "presets": [
    { "name": "clean", "values": { "dirt": { "scalar": 0.0 } } }
  ],
  "body": {
    "steps": [
      {
        "name": "paint",
        "module": { "name": "test.paint", "version": 1 },
        "params": { "color": { "value": { "color": [0.8, 0.6, 0.4] } } }
      },
      {
        "name": "dirt",
        "module": { "name": "test.dirt", "version": 1 },
        "params": { "amount": { "param": "dirt" } },
        "inputs": { "base": { "output": { "step": "paint", "output": "material" } } }
      }
    ],
    "outputs": { "material": { "step": "dirt", "output": "material" } }
  }
}"#;

fn source() -> String {
    DIRTY_PAINT.into()
}

fn registry() -> Registry {
    Registry::new().with(Arc::new(Paint)).with(Arc::new(Dirt))
}

fn realize(module: &dyn Module, bind: Bind) -> Material {
    let mut cx = Context::new(grid(), &NoResources);
    cx.instantiate(module, "wall", bind)
        .expect("instantiates")
        .take_material("material")
        .expect("a material")
}

#[test]
fn a_package_round_trips_and_realizes_the_same_bits() {
    let package = Package::from_json(&source()).expect("reads");
    let text = package.to_json();
    let again = Package::from_json(&text).expect("reads its own output");
    assert_eq!(again, package, "serialize, deserialize");
    assert_eq!(again.fingerprint(), package.fingerprint());
    assert_eq!(again.to_json(), text, "the written form is canonical");

    let registry = registry();
    let a = registry.compile(&package).expect("compiles");
    let b = registry.compile(&again).expect("compiles");
    assert_eq!(a.fingerprint(), b.fingerprint());
    let ma = realize(&a, Bind::new());
    let mb = realize(&b, Bind::new());
    assert_eq!(ma, mb, "bit for bit");
    // The same composition written natively: the same instance paths, so
    // the same seeds, so the same bits.
    let native = realize(&DirtyPaint, Bind::new());
    assert_eq!(ma.digest(), native.digest());
    assert_eq!(ma, native);

    // A preset binds its values.
    let clean = realize(&a, a.preset("clean").expect("a preset"));
    assert_ne!(clean, ma);
    assert_eq!(
        clean.param(Param::BaseColor),
        Some(&Channel::Constant(Value::Vector3(Vec3::new(0.8, 0.6, 0.4))))
    );
}

#[test]
fn every_f32_value_round_trips() {
    let mut preset = Preset::new("many", "");
    let mut bits = 0x3f80_0000_u32;
    for i in 0..2000 {
        bits = bits.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
        let v = f32::from_bits(bits);
        if v.is_finite() {
            preset = preset.with(&i.to_string(), ValueSource::Scalar(v));
        }
    }
    let set = PresetSet {
        format: crate::source::PRESETS_FORMAT.into(),
        version: FORMAT_VERSION,
        module: crate::ModuleRef {
            name: "x".into(),
            version: 1,
        },
        presets: vec![preset],
    };
    let again = PresetSet::from_json(&set.to_json()).expect("reads");
    for (k, v) in &set.presets[0].values {
        let (ValueSource::Scalar(a), ValueSource::Scalar(b)) = (v, &again.presets[0].values[k])
        else {
            panic!("scalars");
        };
        assert_eq!(a.to_bits(), b.to_bits(), "{a}");
    }
}

#[test]
fn unknown_versions_formats_and_fields_are_refused() {
    let newer = source().replace(r#""version": 1,"#, r#""version": 2,"#);
    assert_eq!(
        Package::from_json(&newer),
        Err(PackageError::UnsupportedVersion {
            found: 2,
            supported: 1
        })
    );
    let other = source().replace("dapple.package", "dapple.presets");
    assert!(matches!(
        Package::from_json(&other),
        Err(PackageError::Format { .. })
    ));
    let extra = source().replace(
        r#""doc": "how dirty","#,
        r#""doc": "how dirty", "gain": 2,"#,
    );
    assert!(matches!(
        Package::from_json(&extra),
        Err(PackageError::Syntax(_))
    ));
}

#[test]
fn capabilities_and_modules_are_checked_against_the_engine() {
    let package = Package::from_json(&source()).expect("reads");
    // A capability the engine has never heard of.
    let mut p = package.clone();
    p.requires
        .capabilities
        .push(Capability::new("dapple.gpu", 1));
    assert_eq!(
        registry().compile(&p).map(|_| ()),
        Err(PackageError::UnknownCapability(Capability::new(
            "dapple.gpu",
            1
        )))
    );
    // A newer version than the engine offers.
    let mut p = package.clone();
    p.requires.capabilities[0].version = 3;
    assert!(matches!(
        registry().compile(&p),
        Err(PackageError::CapabilityVersion { offered: 1, .. })
    ));
    // An engine that withdrew what the package needs.
    assert!(matches!(
        registry().withdraw("dapple.semantics").compile(&package),
        Err(PackageError::UnknownCapability(_))
    ));
    // Semantics used but not required.
    let mut p = package.clone();
    p.requires
        .capabilities
        .retain(|c| c.name != "dapple.semantics");
    assert!(matches!(
        registry().compile(&p),
        Err(PackageError::UndeclaredCapability(_))
    ));
    // A module the engine lacks, and one the package does not declare.
    assert!(matches!(
        Registry::new().with(Arc::new(Paint)).compile(&package),
        Err(PackageError::UnknownModule(_))
    ));
    let mut p = package;
    p.requires.modules.pop();
    assert!(matches!(
        registry().compile(&p),
        Err(PackageError::UndeclaredDependency(_))
    ));
}

#[test]
fn bindings_are_checked_when_compiled() {
    let package = Package::from_json(&source()).expect("reads");
    let compile = |edit: &dyn Fn(&mut Package)| {
        let mut p = package.clone();
        edit(&mut p);
        registry().compile(&p).map(|_| ())
    };
    let reason = |r: Result<(), PackageError>| match r {
        Err(PackageError::Binding { reason, .. }) => reason,
        other => panic!("{other:?}"),
    };
    // A literal out of the step's range.
    assert_eq!(
        reason(compile(&|p| {
            p.body.steps[1].params.insert(
                "amount".into(),
                crate::source::ParamBinding::Value(ValueSource::Scalar(2.0)),
            );
        })),
        "out of range"
    );
    // A package parameter whose range is wider than the step's.
    assert_eq!(
        reason(compile(&|p| {
            p.interface.params[0].ty = crate::source::ParamType::Scalar {
                unit: crate::source::UnitSource::Fraction,
                range: [0.0, 2.0],
                default: 0.25,
            };
        })),
        "the package parameter's kind or range does not fit"
    );
    // A required input left unbound, and one read from a later step.
    assert_eq!(
        reason(compile(&|p| {
            p.body.steps[1].inputs.clear();
        })),
        "a required input"
    );
    assert_eq!(
        reason(compile(&|p| {
            p.body.steps.swap(0, 1);
        })),
        "not an earlier step"
    );
    // An unknown channel in the semantics.
    assert_eq!(
        compile(&|p| p.interface.outputs[0].semantics.channels[0] = "gloss".into()),
        Err(PackageError::UnknownChannel("gloss".into()))
    );
    // A preset value that does not fit.
    assert!(matches!(
        compile(&|p| {
            p.presets[0]
                .values
                .insert("dirt".into(), ValueSource::Flag(true));
        }),
        Err(PackageError::InvalidParam { .. })
    ));
}

#[test]
fn semantics_are_checked_on_every_instance() {
    let mut package = Package::from_json(&source()).expect("reads");
    package.interface.outputs[0].semantics.channels = vec!["height".into()];
    let compiled = registry().compile(&package).expect("compiles");
    let mut cx = Context::new(grid(), &NoResources);
    let e = cx
        .instantiate(&compiled, "wall", Bind::new())
        .expect_err("no height is bound");
    assert!(matches!(*e.kind, ModuleErrorKind::Output(_)), "{e}");
}

#[test]
fn interfaces_and_presets_serialize_for_native_modules() {
    let interface = Dirt.interface();
    let source = InterfaceSource::of(&interface).expect("serializable");
    assert_eq!(source.to_interface().expect("checks"), interface);
    let set = PresetSet::new(
        &interface.id,
        vec![Preset::new("fitted", "from a fit").scalar("amount", 0.4)],
    );
    let again = PresetSet::from_json(&set.to_json()).expect("reads");
    again.check(&interface).expect("fits");
    let bind = again.get("fitted").expect("a preset").bind();
    assert_eq!(bind, Bind::new().scalar("amount", 0.4));
    assert!(matches!(
        again.check(&Paint.interface()),
        Err(PackageError::WrongModule { .. })
    ));
}
