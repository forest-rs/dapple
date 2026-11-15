// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Guarded steps, selects and `otherwise`: what they build, what compiling
//! refuses, and what they do to fingerprints, cache keys and diagnostics.

use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use dapple_field::{PortType, Value};
use dapple_material::module::{
    Args, Bind, Context, Diagnostics, InputDecl, InputKind, InstanceCache, Interface, Module,
    ModuleError, ModuleErrorKind, ModuleId, Output, OutputDecl, OutputKind, Outputs, ParamDecl,
    ParamKind, ParamValue, Unit,
};
use dapple_material::resource::NoResources;
use dapple_material::{Channel, Material, Param};
use glam::Vec3;

use crate::compile::{LOGIC, LOGIC_GUARDS};
use crate::source::{BodyOutput, InputBinding, OutputRef, ParamBinding, Select};
use crate::tests::{Dirt, Paint, grid};
use crate::{Capability, Package, PackageError, Registry};

/// Greens a material: a layer like moss, with inputs to pass through.
struct Stain;

impl Module for Stain {
    fn interface(&self) -> Interface {
        let material = |name: &'static str, required| InputDecl {
            name: name.into(),
            kind: InputKind::Material,
            required,
            doc: "".into(),
        };
        Interface {
            id: ModuleId::new("test.stain", 1),
            doc: "stain".into(),
            params: vec![ParamDecl {
                name: "amount".into(),
                kind: ParamKind::Scalar {
                    unit: Unit::Fraction,
                    range: [0.0, 1.0],
                },
                default: ParamValue::Scalar(0.1),
                doc: "".into(),
            }],
            inputs: vec![
                material("base", true),
                material("under", false),
                InputDecl {
                    name: "where".into(),
                    kind: InputKind::Map(PortType::Mask),
                    required: false,
                    doc: "".into(),
                },
            ],
            outputs: vec![OutputDecl {
                name: "material".into(),
                kind: OutputKind::Material,
                doc: "".into(),
            }],
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
        let t = args.scalar("amount");
        m.set_param(
            Param::BaseColor,
            Channel::Constant(Value::Vector3(c.lerp(Vec3::new(0.2, 0.4, 0.1), t))),
        )?;
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}

fn registry() -> Registry {
    Registry::new()
        .with(Arc::new(Paint))
        .with(Arc::new(Dirt))
        .with(Arc::new(Stain))
}

/// Paint or bare, chosen by `painted`; dirt over it when `dirt > 0.0`;
/// moss over that when `moss > 0.0`. Both layers are optional: where one
/// does not run, its `otherwise` passes its base through.
const LAYERS: &str = r#"{
  "format": "dapple.package",
  "version": 1,
  "interface": {
    "module": { "name": "test.layers", "version": 1 },
    "params": [
      { "name": "painted", "type": { "flag": { "default": true } } },
      { "name": "dirt", "type": { "scalar": { "unit": "fraction", "range": [0.0, 1.0], "default": 0.25 } } },
      { "name": "moss", "type": { "scalar": { "unit": "fraction", "range": [0.0, 1.0], "default": 0.0 } } }
    ],
    "inputs": [
      { "name": "backing", "type": "material" },
      { "name": "mask", "type": { "map": "mask" } }
    ],
    "outputs": [{ "name": "material", "type": "material" }]
  },
  "requires": {
    "capabilities": [
      { "name": "dapple.graph", "version": 1 },
      { "name": "dapple.logic", "version": 2 }
    ],
    "modules": [
      { "name": "test.paint", "version": 1 },
      { "name": "test.dirt", "version": 1 },
      { "name": "test.stain", "version": 1 }
    ]
  },
  "body": {
    "steps": [
      {
        "name": "paint",
        "module": { "name": "test.paint", "version": 1 },
        "when": "painted",
        "params": { "color": { "value": { "color": [0.8, 0.2, 0.1] } } }
      },
      {
        "name": "bare",
        "module": { "name": "test.paint", "version": 1 },
        "unless": "painted"
      },
      {
        "name": "dirt",
        "module": { "name": "test.dirt", "version": 1 },
        "when": "dirt > 0.0",
        "params": { "amount": { "param": "dirt" } },
        "inputs": {
          "base": {
            "select": {
              "when": "painted",
              "then": { "output": { "step": "paint", "output": "material" } },
              "else": { "output": { "step": "bare", "output": "material" } }
            }
          }
        },
        "otherwise": { "material": "base" }
      },
      {
        "name": "moss",
        "module": { "name": "test.stain", "version": 1 },
        "when": "moss > 0.0",
        "params": { "amount": { "expr": "min(moss, 0.5)" } },
        "inputs": { "base": { "output": { "step": "dirt", "output": "material" } } },
        "otherwise": { "material": "base" }
      }
    ],
    "outputs": { "material": { "step": "moss", "output": "material" } }
  }
}"#;

fn layers() -> Package {
    Package::from_json(LAYERS).expect("reads")
}

/// [`LAYERS`] written natively: the same modules under the same names.
struct NativeLayers;

impl Module for NativeLayers {
    fn interface(&self) -> Interface {
        registry().compile(&layers()).expect("compiles").interface()
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let take = |mut o: Outputs| o.take_material("material").expect("a material");
        let base = if args.flag("painted") {
            take(cx.instantiate(
                &Paint,
                "paint",
                Bind::new().color("color", Vec3::new(0.8, 0.2, 0.1)),
            )?)
        } else {
            take(cx.instantiate(&Paint, "bare", Bind::new())?)
        };
        let dirt = args.scalar("dirt");
        let m = if dirt > 0.0 {
            take(cx.instantiate(
                &Dirt,
                "dirt",
                Bind::new().material("base", base).scalar("amount", dirt),
            )?)
        } else {
            base
        };
        let moss = args.scalar("moss");
        let m = if moss > 0.0 {
            take(
                cx.instantiate(
                    &Stain,
                    "moss",
                    Bind::new()
                        .material("base", m)
                        .scalar("amount", moss.min(0.5)),
                )?,
            )
        } else {
            m
        };
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}

fn bind(painted: bool, dirt: f32, moss: f32) -> Bind {
    Bind::new()
        .flag("painted", painted)
        .scalar("dirt", dirt)
        .scalar("moss", moss)
}

fn build(module: &dyn Module, bind: Bind) -> (Result<Material, ModuleError>, Diagnostics) {
    let mut cx = Context::new(grid(), &NoResources);
    let m = cx
        .instantiate(module, "wall", bind)
        .map(|mut o| o.take_material("material").expect("a material"));
    (m, cx.into_diagnostics())
}

fn paths<'a>(it: impl Iterator<Item = (&'a str, &'a ModuleId)>) -> Vec<String> {
    it.map(|(p, m)| alloc::format!("{p} {}@{}", m.name, m.version))
        .collect()
}

#[test]
fn guarded_steps_run_only_where_their_guards_hold() {
    let package = layers();
    let again = Package::from_json(&package.to_json()).expect("reads its own output");
    assert_eq!(again, package);
    assert_eq!(again.to_json(), package.to_json(), "canonical");
    let compiled = registry().compile(&package).expect("compiles");
    // Three conditions, each compiled once though `painted` guards two
    // steps and a select; and one parameter expression.
    let program = compiled.logic().expect("a program");
    assert_eq!(program.outputs().len(), 4);

    let mut distinct = Vec::new();
    for painted in [true, false] {
        for dirt in [0.0, 0.3] {
            for moss in [0.0, 0.4, 0.8] {
                let at = alloc::format!("painted {painted}, dirt {dirt}, moss {moss}");
                let (m, d) = build(&compiled, bind(painted, dirt, moss));
                let (n, native) = build(&NativeLayers, bind(painted, dirt, moss));
                let (m, n) = (m.expect("builds"), n.expect("builds"));
                assert_eq!(m, n, "{at}: bit for bit");
                // The same instances, by the same keys.
                let instances = paths(d.instances());
                assert_eq!(instances[1..], paths(native.instances())[1..], "{at}");
                for (path, _) in native.instances().skip(1) {
                    assert_eq!(d.fingerprint_at(path), native.fingerprint_at(path), "{at}");
                }
                // What did not run is listed, and only that.
                let mut expected = vec![if painted {
                    "wall/bare test.paint@1"
                } else {
                    "wall/paint test.paint@1"
                }];
                if dirt == 0.0 {
                    expected.push("wall/dirt test.dirt@1");
                }
                if moss == 0.0 {
                    expected.push("wall/moss test.stain@1");
                }
                assert_eq!(paths(d.skipped()), expected, "{at}");
                assert_eq!(instances.len() + expected.len(), 5, "{at}");
                if !distinct.contains(&m.digest()) {
                    distinct.push(m.digest());
                }
            }
        }
    }
    // Moss at 0.4 and 0.8 differ; nothing else collapses.
    assert_eq!(distinct.len(), 12);
    // The diagnostics say it in words.
    let (_, d) = build(&compiled, bind(true, 0.0, 0.0));
    let text = d.to_string();
    assert!(
        text.contains("wall/bare = test.paint@1 skipped\n"),
        "{text}"
    );
    assert!(
        text.contains("wall/moss = test.stain@1 skipped\n"),
        "{text}"
    );
}

/// A guard is a condition like any other: one that stops the program fails
/// the instance at its step, naming the guard and its expression.
#[test]
fn a_guard_that_fails_names_its_step_and_expression() {
    let mut package = layers();
    package.body.steps[3].when = Some("clamp(moss, 0.5, dirt) > 0.25".into());
    let compiled = registry().compile(&package).expect("compiles");
    assert!(build(&compiled, bind(true, 0.75, 0.3)).0.is_ok());
    let e = build(&compiled, bind(true, 0.25, 0.3)).0.unwrap_err();
    assert_eq!(e.path, "wall/moss");
    assert_eq!(
        *e.kind,
        ModuleErrorKind::InvalidParam {
            name: "when = clamp(moss, 0.5, dirt) > 0.25".into(),
            reason: "the expression clamps with bounds out of order",
        }
    );
    assert_eq!(
        e.to_string(),
        r#"wall/moss: "when = clamp(moss, 0.5, dirt) > 0.25": the expression clamps with bounds out of order"#
    );
    // A select's condition fails at the step whose input it chooses.
    let mut package = layers();
    package.body.steps[3].inputs.insert(
        "base".into(),
        InputBinding::Select(alloc::boxed::Box::new(Select {
            when: "clamp(dirt, 0.5, moss) > 0.0".into(),
            then: output("dirt"),
            otherwise: output("dirt"),
        })),
    );
    let compiled = registry().compile(&package).expect("compiles");
    assert!(build(&compiled, bind(true, 0.25, 0.6)).0.is_ok());
    let e = build(&compiled, bind(true, 0.25, 0.3)).0.unwrap_err();
    assert_eq!(e.path, "wall/moss");
    assert_eq!(
        *e.kind,
        ModuleErrorKind::InvalidParam {
            name: "base = select when clamp(dirt, 0.5, moss) > 0.0".into(),
            reason: "the expression clamps with bounds out of order",
        }
    );
}

/// A skipped step's parameter expressions do not run, so what they would
/// compute cannot fail the instance; where the step runs, they fail it as
/// any expression does.
#[test]
fn a_skipped_steps_expressions_neither_run_nor_fail() {
    for (expression, fails_at) in [
        ("moss - 0.1", 0.05),               // out of range below 0.1
        ("clamp(moss, 0.01, moss)", 0.005), // bounds out of order below 0.01
    ] {
        let mut package = layers();
        package.body.steps[3]
            .params
            .insert("amount".into(), ParamBinding::Expr(expression.into()));
        let compiled = registry().compile(&package).expect("compiles");
        let (m, d) = build(&compiled, bind(true, 0.3, 0.0));
        assert!(m.is_ok(), "{expression}: skipped, it is never computed");
        assert_eq!(d.skipped().count(), 2, "bare and moss");
        assert!(build(&compiled, bind(true, 0.3, 0.4)).0.is_ok());
        let e = build(&compiled, bind(true, 0.3, fails_at)).0.unwrap_err();
        assert_eq!(e.path, "wall/moss", "{expression}");
        assert!(
            matches!(&*e.kind, ModuleErrorKind::InvalidParam { name, .. }
                if *name == alloc::format!("amount = {expression}")),
            "{e}"
        );
    }
}

fn compile(package: &Package) -> Result<(), PackageError> {
    registry().compile(package).map(|_| ())
}

fn output(step: &str) -> InputBinding {
    InputBinding::Output(OutputRef {
        step: step.into(),
        output: "material".into(),
    })
}

/// Every way of reading a step where it might not have run is refused when
/// the package compiles, saying who reads what, and under which condition
/// it runs.
#[test]
fn reading_a_step_that_might_not_have_run_is_refused() {
    let skipped = |r: Result<(), PackageError>| match r {
        Err(e @ PackageError::MaybeSkipped { .. }) => e,
        other => panic!("{other:?}"),
    };

    // An unguarded step reads a guarded one that has no `otherwise`.
    let mut p = layers();
    p.body.steps[2].otherwise.clear();
    let e = skipped(compile(&p));
    assert_eq!(
        e,
        PackageError::MaybeSkipped {
            step: "moss".into(),
            name: "base".into(),
            reads: "dirt".into(),
            condition: "dirt > 0.0".into(),
            holds: true,
        }
    );
    assert_eq!(
        e.to_string(),
        "step \"moss\", \"base\": reads step \"dirt\", which runs only when \"dirt > 0.0\"; read \
         it under the same condition (from a step with the same guard, or through a select on \
         it), or give \"dirt\" an `otherwise`"
    );
    // A step under the same text may read it.
    p.body.steps[3].when = Some("dirt > 0.0".into());
    p.body.steps[3].otherwise.clear();
    p.body.outputs.insert(
        "material".into(),
        BodyOutput::Select {
            select: alloc::boxed::Box::new(Select {
                when: "dirt > 0.0".into(),
                then: BodyOutput::Output(OutputRef {
                    step: "moss".into(),
                    output: "material".into(),
                }),
                otherwise: BodyOutput::Output(OutputRef {
                    step: "paint".into(),
                    output: "material".into(),
                }),
            }),
        },
    );
    // ...but not this output's `else`, where `painted` is not known.
    let e = skipped(compile(&p));
    assert!(
        matches!(&e, PackageError::MaybeSkipped { step, name, reads, .. }
            if step.is_empty() && name == "material" && reads == "paint"),
        "{e:?}"
    );
    assert!(
        e.to_string().starts_with(
            "output \"material\": reads step \"paint\", which runs only when \"painted\";"
        ),
        "{e}"
    );
    // Texts are compared, not meanings: another spelling of the same
    // condition is another condition.
    let mut p = layers();
    p.body.steps[2].otherwise.clear();
    p.body.steps[3].when = Some("0.0 < dirt".into());
    assert!(matches!(
        skipped(compile(&p)),
        PackageError::MaybeSkipped { reads, .. } if reads == "dirt"
    ));
    // A select's branches are read under their own sides of its condition:
    // `then` may not read the `unless` step, nor `else` the `when` one.
    let mut p = layers();
    let InputBinding::Select(s) = p.body.steps[2].inputs.get_mut("base").unwrap() else {
        panic!("a select");
    };
    core::mem::swap(&mut s.then, &mut s.otherwise);
    assert!(matches!(
        skipped(compile(&p)),
        PackageError::MaybeSkipped { step, name, reads, holds: false, .. }
            if step == "dirt" && name == "base" && reads == "bare"
    ));
    // A package output reads a guarded step that has no `otherwise`.
    let mut p = layers();
    p.body.steps[3].otherwise.clear();
    assert!(matches!(
        skipped(compile(&p)),
        PackageError::MaybeSkipped { step, reads, .. } if step.is_empty() && reads == "moss"
    ));
    // An `otherwise` is read where its step does not run, so it may not
    // read what runs only where its step does.
    let mut p = layers();
    p.body.steps[3].when = Some("dirt > 0.0".into());
    p.body.steps[2].otherwise.clear();
    assert!(matches!(
        skipped(compile(&p)),
        PackageError::MaybeSkipped { step, name, reads, .. }
            if step == "moss" && name == "material" && reads == "dirt"
    ));
}

/// Each malformed guard, select or `otherwise` is refused with a reason.
#[test]
fn guards_selects_and_otherwise_are_checked_when_compiled() {
    let reason = |r: Result<(), PackageError>| match r {
        Err(PackageError::Binding { step, name, reason }) => (step, name, reason),
        Err(PackageError::Output { name, reason }) => (String::new(), name, reason),
        other => panic!("{other:?}"),
    };
    let edit = |f: &dyn Fn(&mut Package)| {
        let mut p = layers();
        f(&mut p);
        compile(&p)
    };
    assert_eq!(compile(&layers()), Ok(()));
    assert_eq!(
        reason(edit(&|p| p.body.steps[2].unless = Some("painted".into()))),
        (
            "dirt".into(),
            "when".into(),
            "a step has `when` or `unless`, not both"
        )
    );
    assert_eq!(
        reason(edit(&|p| {
            p.body.steps[0]
                .otherwise
                .insert("material".into(), "base".into());
            p.body.steps[0].when = None;
        })),
        (
            "paint".into(),
            "otherwise".into(),
            "only a step with `when` or `unless` has an `otherwise`"
        )
    );
    assert_eq!(
        reason(edit(&|p| {
            p.body.steps[3]
                .otherwise
                .insert("mask".into(), "base".into());
        })),
        ("moss".into(), "mask".into(), "no such step output")
    );
    assert_eq!(
        reason(edit(&|p| {
            p.body.steps[3]
                .otherwise
                .insert("material".into(), "bottom".into());
        })),
        (
            "moss".into(),
            "material".into(),
            "`otherwise` names no input of the step"
        )
    );
    assert_eq!(
        reason(edit(&|p| {
            p.body.steps[3]
                .otherwise
                .insert("material".into(), "under".into());
        })),
        (
            "moss".into(),
            "material".into(),
            "`otherwise` passes through an unbound input"
        )
    );
    assert_eq!(
        reason(edit(&|p| {
            p.body.steps[3]
                .inputs
                .insert("where".into(), InputBinding::Input("mask".into()));
            p.body.steps[3]
                .otherwise
                .insert("material".into(), "where".into());
        })),
        (
            "moss".into(),
            "material".into(),
            "`otherwise` passes through an input of another kind"
        )
    );
    assert_eq!(
        reason(edit(&|p| {
            p.body.steps[3]
                .inputs
                .insert("under".into(), InputBinding::Input("backing".into()));
            p.body.steps[3]
                .otherwise
                .insert("material".into(), "under".into());
        })),
        (
            "moss".into(),
            "material".into(),
            "`otherwise` passes through an input that may be absent"
        )
    );
    // A select whose branch may be absent cannot feed a required input.
    assert_eq!(
        reason(edit(&|p| {
            let InputBinding::Select(s) = p.body.steps[2].inputs.get_mut("base").unwrap() else {
                panic!("a select");
            };
            s.otherwise = InputBinding::Input("backing".into());
        })),
        (
            "dirt".into(),
            "base".into(),
            "a required input bound to an optional one"
        )
    );
    // A branch of the wrong kind.
    assert_eq!(
        reason(edit(&|p| {
            let InputBinding::Select(s) = p.body.steps[2].inputs.get_mut("base").unwrap() else {
                panic!("a select");
            };
            s.otherwise = InputBinding::Input("mask".into());
        })),
        (
            "dirt".into(),
            "base".into(),
            "the package input's kind does not fit"
        )
    );
    // A package output's select reads steps that exist.
    assert_eq!(
        reason(edit(&|p| {
            p.body.outputs.insert(
                "material".into(),
                BodyOutput::Select {
                    select: alloc::boxed::Box::new(Select {
                        when: "painted".into(),
                        then: BodyOutput::Output(OutputRef {
                            step: "moss".into(),
                            output: "material".into(),
                        }),
                        otherwise: BodyOutput::Output(OutputRef {
                            step: "lichen".into(),
                            output: "material".into(),
                        }),
                    }),
                },
            );
        })),
        (String::new(), "material".into(), "no such step")
    );

    // Conditions are flag expressions over the package's parameters.
    let expression = |r: Result<(), PackageError>| match r {
        Err(e @ PackageError::Expression { .. }) => e,
        other => panic!("{other:?}"),
    };
    let e = expression(edit(&|p| p.body.steps[3].when = Some("moss".into())));
    assert_eq!(
        e.to_string(),
        r#"step "moss", "when": in "moss", at 0..4: a scalar where a flag is needed"#
    );
    let e = expression(edit(&|p| {
        p.body.steps[1].unless = Some("painted &&".into());
    }));
    assert!(
        matches!(&e, PackageError::Expression { step, name, .. } if step == "bare" && name == "unless"),
        "{e:?}"
    );
    let e = expression(edit(&|p| {
        let InputBinding::Select(s) = p.body.steps[2].inputs.get_mut("base").unwrap() else {
            panic!("a select");
        };
        s.when = "glossy".into();
    }));
    assert_eq!(
        e.to_string(),
        r#"step "dirt", "base": in "glossy", at 0..6: no input named "glossy""#
    );
    let e = expression(edit(&|p| {
        p.body.outputs.insert(
            "material".into(),
            BodyOutput::Select {
                select: alloc::boxed::Box::new(Select {
                    when: "dirt".into(),
                    then: BodyOutput::Output(OutputRef {
                        step: "moss".into(),
                        output: "material".into(),
                    }),
                    otherwise: BodyOutput::Output(OutputRef {
                        step: "moss".into(),
                        output: "material".into(),
                    }),
                }),
            },
        );
    }));
    assert_eq!(
        e.to_string(),
        r#"output "material": in "dirt", at 0..4: a scalar where a flag is needed"#
    );

    // Guards and selects need `dapple.logic` at version 2, declared and
    // offered.
    let needs = Err(PackageError::UndeclaredCapability(Capability::new(
        LOGIC,
        LOGIC_GUARDS,
    )));
    assert_eq!(
        edit(&|p| p.requires.capabilities[1].version = 1),
        needs.clone()
    );
    assert_eq!(
        edit(&|p| {
            p.requires.capabilities.pop();
            p.body.steps[3].params.clear();
        }),
        needs
    );
    let mut select_only = layers();
    for s in &mut select_only.body.steps {
        s.when = None;
        s.unless = None;
        s.otherwise.clear();
    }
    select_only.requires.capabilities[1].version = 1;
    select_only.body.steps[3].params.clear();
    assert_eq!(compile(&select_only), needs);
    assert!(matches!(
        registry()
            .offer(Capability::new(LOGIC, 1))
            .compile(&layers()),
        Err(PackageError::CapabilityVersion { offered: 1, .. })
    ));
}

/// The new shapes read strictly: unknown fields are refused, and a package
/// output is a step's output or a select, not both.
#[test]
fn the_new_shapes_read_strictly() {
    let syntax = |text: String| match Package::from_json(&text) {
        Err(PackageError::Syntax(e)) => e,
        other => panic!("{other:?}"),
    };
    syntax(LAYERS.replace(r#""when": "painted","#, r#""when": "painted", "if": "x","#));
    syntax(LAYERS.replace(
        r#""otherwise": { "material": "base" }
      },
      {
        "name": "moss","#,
        r#""otherwise": { "material": { "input": "base" } }
      },
      {
        "name": "moss","#,
    ));
    let e = syntax(LAYERS.replace(
        r#""outputs": { "material": { "step": "moss", "output": "material" } }"#,
        r#""outputs": { "material": { "step": "moss", "output": "material", "select": { "when": "painted", "then": { "step": "moss", "output": "material" }, "else": { "step": "moss", "output": "material" } } } }"#,
    ));
    assert!(
        e.contains("a package output is a step's output (`step` and `output`) or a `select`"),
        "{e}"
    );
    syntax(LAYERS.replace(
        r#""outputs": { "material": { "step": "moss", "output": "material" } }"#,
        r#""outputs": { "material": { "step": "moss" } }"#,
    ));
    // A package output's select, written and read back.
    let text = LAYERS.replace(
        r#""outputs": { "material": { "step": "moss", "output": "material" } }"#,
        r#""outputs": { "material": { "select": { "when": "painted", "then": { "step": "moss", "output": "material" }, "else": { "step": "dirt", "output": "material" } } } }"#,
    );
    let p = Package::from_json(&text).expect("reads");
    assert!(matches!(
        p.body.outputs["material"],
        BodyOutput::Select { .. }
    ));
    assert_eq!(Package::from_json(&p.to_json()).unwrap(), p);
    assert_eq!(compile(&p), Ok(()));
}

/// The compiled fingerprint covers every guard, `otherwise` and select (the
/// source's canonical bytes do), and their conditions' program.
#[test]
fn fingerprints_cover_guards_selects_and_otherwise() {
    let fingerprint = |p: &Package| registry().compile(p).expect("compiles").fingerprint();
    let base = fingerprint(&layers());
    assert_eq!(base, fingerprint(&layers()), "stable");
    let mut seen = vec![base];
    let mut differs = |f: &dyn Fn(&mut Package)| {
        let mut p = layers();
        f(&mut p);
        let x = fingerprint(&p);
        assert!(!seen.contains(&x), "{p:?}");
        seen.push(x);
    };
    // Another condition text, the same meaning.
    differs(&|p| p.body.steps[2].when = Some("0.0 < dirt".into()));
    // `when` turned to `unless`, and a step's guard dropped.
    differs(&|p| {
        p.body.steps[3].unless = p.body.steps[3].when.take();
    });
    differs(&|p| {
        p.body.steps[3].when = None;
        p.body.steps[3].otherwise.clear();
    });
    // An `otherwise` passing another input, bound the same way.
    differs(&|p| {
        p.body.steps[3]
            .inputs
            .insert("under".into(), output("dirt"));
        p.body.steps[3]
            .otherwise
            .insert("material".into(), "under".into());
    });
    // A select's condition, and its branches swapped with their guards.
    differs(&|p| {
        let InputBinding::Select(s) = p.body.steps[2].inputs.get_mut("base").unwrap() else {
            panic!("a select");
        };
        s.when = "!(!painted)".into();
        p.body.steps[0].when = Some("!(!painted)".into());
        p.body.steps[1].unless = Some("!(!painted)".into());
    });
    differs(&|p| {
        let InputBinding::Select(s) = p.body.steps[2].inputs.get_mut("base").unwrap() else {
            panic!("a select");
        };
        core::mem::swap(&mut s.then, &mut s.otherwise);
        p.body.steps[0].unless = p.body.steps[0].when.take();
        p.body.steps[1].when = p.body.steps[1].unless.take();
    });
    // A layer made unconditional.
    differs(&|p| {
        p.body.steps[2].inputs.insert(
            "base".into(),
            InputBinding::Select(alloc::boxed::Box::new(Select {
                when: "painted".into(),
                then: output("paint"),
                otherwise: output("bare"),
            })),
        );
        p.body.steps[2].otherwise.clear();
        p.body.steps[2].when = None;
    });
}

/// Which steps run is a function of the package's arguments, and those are
/// in its instance key; each step that runs is keyed by its own arguments.
/// So: arguments that differ only in whether a step runs build different
/// instances; arguments that run the same steps with the same values share
/// every step's cached instance; and a skipped step leaves nothing in the
/// cache.
#[test]
fn cache_keys_follow_what_runs() {
    let compiled = registry().compile(&layers()).expect("compiles");
    let mut cache = InstanceCache::new(1 << 24);
    let cached = |b: Bind, cache: &mut InstanceCache| {
        let plain = build(&compiled, b.clone());
        let mut cx = Context::new(grid(), &NoResources).with_cache(cache);
        let m = cx
            .instantiate(&compiled, "wall", b)
            .map(|mut o| o.take_material("material").expect("a material"));
        let got = (m, cx.into_diagnostics());
        assert_eq!(got, plain, "cached equals uncached, diagnostics too");
        got
    };
    let count = |cache: &InstanceCache, module: &str| {
        cache
            .stats()
            .modules
            .iter()
            .find(|m| m.module.name == module)
            .map_or((0, 0), |m| (m.hits, m.misses))
    };

    // No moss: the stain module is never instantiated, nor kept.
    let (bare, d0) = cached(bind(true, 0.3, 0.0), &mut cache);
    assert_eq!(count(&cache, "test.stain"), (0, 0));
    assert_eq!(cache.stats().entries, 3, "wall, paint and dirt");

    // Moss, which differs only in the guard: the package's own key differs,
    // its result differs, and the steps under the moss hit.
    let (mossy, d1) = cached(bind(true, 0.3, 0.6), &mut cache);
    assert_ne!(bare.unwrap(), mossy.clone().unwrap());
    assert_ne!(d0.fingerprint_at("wall"), d1.fingerprint_at("wall"));
    assert_eq!(count(&cache, "test.paint"), (1, 1));
    assert_eq!(count(&cache, "test.dirt"), (1, 1));
    assert_eq!(count(&cache, "test.stain"), (0, 1));

    // Other moss, the same steps and derived amount (`min(moss, 0.5)`):
    // every step hits; only the package's own instance is new.
    let (again, d2) = cached(bind(true, 0.3, 0.9), &mut cache);
    assert_eq!(again.unwrap(), mossy.unwrap());
    assert_eq!(count(&cache, "test.stain"), (1, 1));
    assert_eq!(count(&cache, "test.layers"), (0, 3));
    assert_eq!(
        d1.fingerprint_at("wall/moss"),
        d2.fingerprint_at("wall/moss")
    );

    // No dirt: the dirt step is skipped and the moss reads the paint
    // itself, so the moss step's key (its base's digest) differs.
    let (_, d3) = cached(bind(true, 0.0, 0.9), &mut cache);
    assert_eq!(count(&cache, "test.dirt"), (2, 1), "only earlier builds");
    assert_eq!(count(&cache, "test.stain"), (1, 2));
    assert_ne!(
        d3.fingerprint_at("wall/moss"),
        d2.fingerprint_at("wall/moss")
    );
    assert_eq!(
        paths(d3.skipped()),
        ["wall/bare test.paint@1", "wall/dirt test.dirt@1"]
    );

    // The same arguments again: the package's instance hits, and replays
    // its diagnostics, skipped steps included.
    let (_, d4) = cached(bind(true, 0.0, 0.9), &mut cache);
    assert_eq!(d4, d3);
    assert_eq!(count(&cache, "test.layers"), (1, 4));
}

/// Steps under one guard read each other directly, as the varnished
/// board's wear reads the traffic mask made only for it; what follows them
/// reads through a select on that guard.
#[test]
fn a_step_reads_another_under_the_same_guard() {
    let mut p = layers();
    // A second layer that runs exactly where moss does, reading the moss
    // step's output directly: no `otherwise` needed for that read.
    let mut lichen = p.body.steps[3].clone();
    lichen.name = "lichen".into();
    lichen.params.clear();
    lichen.inputs.insert("base".into(), output("moss"));
    lichen.otherwise.clear();
    p.body.steps[3].otherwise.clear();
    p.body.steps.push(lichen);
    p.body.outputs.insert(
        "material".into(),
        BodyOutput::Select {
            select: alloc::boxed::Box::new(Select {
                when: "moss > 0.0".into(),
                then: BodyOutput::Output(OutputRef {
                    step: "lichen".into(),
                    output: "material".into(),
                }),
                otherwise: BodyOutput::Output(OutputRef {
                    step: "dirt".into(),
                    output: "material".into(),
                }),
            }),
        },
    );
    let compiled = registry().compile(&p).expect("compiles");
    let (m, d) = build(&compiled, bind(false, 0.2, 0.0));
    let (n, _) = build(&NativeLayers, bind(false, 0.2, 0.0));
    assert_eq!(m.unwrap(), n.unwrap(), "the select's else: dirt over bare");
    assert_eq!(
        paths(d.skipped()),
        [
            "wall/paint test.paint@1",
            "wall/moss test.stain@1",
            "wall/lichen test.stain@1"
        ]
    );
    let (m, d) = build(&compiled, bind(false, 0.2, 0.3));
    assert!(m.is_ok());
    assert_eq!(
        paths(d.instances()),
        [
            "wall test.layers@1",
            "wall/bare test.paint@1",
            "wall/dirt test.dirt@1",
            "wall/moss test.stain@1",
            "wall/lichen test.stain@1"
        ]
    );
}

/// A package without guards or selects needs only what it needed before.
#[test]
fn a_package_without_guards_needs_no_more() {
    let mut p = layers();
    for s in &mut p.body.steps {
        s.when = None;
        s.unless = None;
        s.otherwise.clear();
    }
    p.body.steps[2]
        .inputs
        .insert("base".into(), output("paint"));
    p.requires.capabilities[1].version = 1;
    assert_eq!(compile(&p), Ok(()));
    let compiled = registry().compile(&p).expect("compiles");
    let (_, d) = build(&compiled, bind(true, 0.2, 0.3));
    assert_eq!(d.skipped().count(), 0);
    assert_eq!(d.instances().count(), 5);
    assert_eq!(
        compiled.logic().map(|l| l.outputs().len()),
        Some(1),
        "the moss amount alone"
    );
}
