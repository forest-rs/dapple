// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Contracts of material values, operations, programs and modules.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "small test values"
)]

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use dapple_field::program::Fingerprint;
use dapple_field::scoped::{Scope, ScopedBuilder};
use dapple_field::{
    Domain, Edge, Footprint, ImageLevel, NormalFrame, PortType, SampleImage, SamplePolicy, Value,
};
use dapple_raster::Raster;
use dapple_raster::typed::ReductionPolicy;
use glam::{Vec2, Vec3};

use crate::module::{
    Args, Bind, Context, Input, InputDecl, InputKind, Interface, Module, ModuleError,
    ModuleErrorKind, ModuleId, Output, OutputDecl, OutputKind, Outputs, ParamDecl, ParamKind,
    ParamValue, Unit,
};
use crate::ops::{
    self, Coating, Deposit, Detail, Layer, MaterialTransform, Transition, reoriented,
};
use crate::program::{MapBinding, evaluate};
use crate::resource::{
    NoResources, Resolved, ResourceError, ResourceHost, ResourceRef, ResourceRequest,
};
use crate::*;

const N: u32 = 8;

fn grid() -> Grid {
    Grid {
        width: N,
        height: N,
        origin: Vec2::ZERO,
        texel: Vec2::splat(0.01),
        edge: Edge::Wrap,
    }
}

fn ramp_raster(f: impl Fn(u32, u32) -> f32) -> Raster {
    let mut v = Vec::new();
    for y in 0..N {
        for x in 0..N {
            v.push(f(x, y));
        }
    }
    grid().raster(v).unwrap()
}

fn map(port: PortType, f: impl Fn(u32, u32) -> Value) -> Channel {
    let mut v = Vec::new();
    for y in 0..N {
        for x in 0..N {
            v.push(f(x, y));
        }
    }
    Channel::Map(grid().typed(port, v).unwrap())
}

fn color() -> PortType {
    PortType::Color(dapple_field::Primaries::Rec709)
}

/// A material binding one channel of every kind, all varying.
fn rich(tag: f32) -> Material {
    let mut m = Material::new(grid());
    let s = |x: u32, y: u32| (x as f32 * 0.1 + y as f32 * 0.01 + tag).fract();
    m.set_param(
        Param::BaseColor,
        map(color(), |x, y| Value::Vector3(Vec3::new(s(x, y), 0.5, tag))),
    )
    .unwrap();
    m.set_param(
        Param::SpecularRoughness,
        map(PortType::Scalar, |x, y| Value::Scalar(s(x, y))),
    )
    .unwrap();
    m.set_param(
        Param::GeometryNormal,
        map(PortType::Normal(NormalFrame::Domain), |x, y| {
            Value::Vector3(Vec3::new(s(x, y) - 0.5, 0.2, 1.0).normalize())
        }),
    )
    .unwrap();
    m.set_aux(
        Aux::Height,
        map(PortType::Scalar, |x, y| Value::Scalar(s(x, y) * 0.001)),
    )
    .unwrap();
    m.set_aux(
        Aux::Region,
        map(PortType::Id, |x, y| {
            Value::Id(x + N * y + (tag * 100.0) as u32)
        }),
    )
    .unwrap();
    m.set_param(
        Param::SpecularIor,
        Channel::Constant(Value::Scalar(1.4 + tag)),
    )
    .unwrap();
    m
}

#[test]
fn channels_are_typed_and_uniform_parameters_stay_constant() {
    let mut m = Material::new(grid());
    assert_eq!(
        m.value(ChannelId::Param(Param::SpecularRoughness), 0),
        Value::Scalar(0.3),
        "the specification's default"
    );
    assert_eq!(
        m.set_param(Param::BaseColor, Channel::Constant(Value::Scalar(1.0))),
        Err(MaterialError::TypeMismatch(ChannelId::Param(
            Param::BaseColor
        )))
    );
    assert_eq!(
        m.set_param(
            Param::GeometryThinWalled,
            map(PortType::Scalar, |_, _| Value::Scalar(1.0))
        ),
        Err(MaterialError::Uniform(Param::GeometryThinWalled))
    );
    assert_eq!(
        param_port(Param::GeometryNormal),
        PortType::Normal(NormalFrame::Domain)
    );
    let from = Material::from_parameters(
        grid(),
        &openpbr::Parameters {
            specular_roughness: 0.7,
            ..openpbr::Parameters::DEFAULT
        },
    );
    assert_eq!(
        from.bound().map(|(c, _)| c).collect::<Vec<_>>(),
        vec![ChannelId::Param(Param::SpecularRoughness)],
        "only what differs from the defaults is bound"
    );
}

#[test]
fn selection_decides_once_for_every_channel() {
    let (a, b) = (rich(0.0), rich(0.5));
    // Exactly 0 on the left, 1 on the right, and 0.5 in column 3.
    let w = ramp_raster(|x, _| match x {
        0..3 => 0.0,
        3 => 0.5,
        _ => 1.0,
    });
    let (m, report) = ops::select(&a, &b, &w, Transition::Mask).unwrap();
    for (c, _) in a.bound() {
        for y in 0..N as usize {
            let (left, right) = (y * N as usize, y * N as usize + 5);
            assert_eq!(m.value(c, left), a.value(c, left), "{c} left");
            assert_eq!(m.value(c, right), b.value(c, right), "{c} right");
        }
    }
    assert_eq!(report.partial, u64::from(N), "one fractional column");
    for kind in [
        ApproximationKind::RoughnessInAlpha,
        ApproximationKind::NormalsAveraged,
        ApproximationKind::IorInterpolated,
        ApproximationKind::WinnerLabel,
    ] {
        assert!(report.has(kind), "{kind:?} in {report}");
    }
    // Roughness in α: sqrt((ra² + rb²) / 2).
    let i = 3;
    let rough = |m: &Material| {
        m.value(ChannelId::Param(Param::SpecularRoughness), i)
            .scalar()
            .unwrap()
    };
    let (ra, rb, r) = (rough(&a), rough(&b), rough(&m));
    assert!((r - libm::sqrtf(0.5 * (ra * ra + rb * rb))).abs() < 1e-6);

    // A selection that is never fractional is exact.
    let hard = ramp_raster(|x, _| if x < 4 { 0.0 } else { 1.0 });
    let (_, exact) = ops::select(&a, &b, &hard, Transition::Mask).unwrap();
    assert!(exact.is_exact(), "{exact}");
}

#[test]
fn detail_has_an_identity_and_composes_by_rnm() {
    let base = rich(0.25);
    for layer in [Layer::Base, Layer::Coat, Layer::Both] {
        let (same, report) = ops::apply_detail(&base, &Detail::identity(layer)).unwrap();
        assert_eq!(same.digest(), base.digest(), "identity for {layer:?}");
        assert!(report.is_exact());
    }
    // A detail normal onto a flat base is the detail itself.
    let flat = Material::new(grid());
    let n = Vec3::new(0.3, -0.2, 0.9).normalize();
    let detail = Detail {
        normal: Some(
            grid()
                .typed(
                    PortType::Normal(NormalFrame::Domain),
                    vec![Value::Vector3(n); 64],
                )
                .unwrap(),
        ),
        ..Detail::identity(Layer::Base)
    };
    let (m, report) = ops::apply_detail(&flat, &detail).unwrap();
    let got = m.value(ChannelId::Param(Param::GeometryNormal), 0);
    assert!(matches!(got, Value::Vector3(v) if (v - n).length() < 1e-6));
    assert!(report.is_exact(), "no height to derive a normal from");
    assert!((reoriented(Vec3::Z, n) - n).length() < 1e-6, "RNM onto +Z");
    // Onto a base with height and no normal, the base normal is derived.
    let mut bumpy = Material::new(grid());
    bumpy
        .set_aux(
            Aux::Height,
            map(PortType::Scalar, |x, _| Value::Scalar(x as f32 * 1e-4)),
        )
        .unwrap();
    let (_, report) = ops::apply_detail(&bumpy, &detail).unwrap();
    assert!(report.has(ApproximationKind::NormalFromHeight));
    // A coat-layer height ripples the coat, not the displacement.
    let ripple = Detail {
        height: Some(ramp_raster(|x, _| if x % 2 == 0 { 1e-4 } else { 0.0 })),
        ..Detail::identity(Layer::Coat)
    };
    let (m, _) = ops::apply_detail(&bumpy, &ripple).unwrap();
    assert_eq!(
        m.aux(Aux::Height),
        bumpy.aux(Aux::Height),
        "no displacement"
    );
    assert!(
        m.param(Param::GeometryCoatNormal).is_some(),
        "the coat ripples"
    );
    assert!(
        m.param(Param::GeometryNormal).is_none(),
        "the base does not"
    );
}

#[test]
fn coats_keep_the_base_and_collapse_when_doubled() {
    let base = rich(0.1);
    let glaze = Coating {
        color: Channel::Constant(Value::Vector3(Vec3::new(0.8, 0.9, 0.7))),
        roughness: Channel::Constant(Value::Scalar(0.05)),
        ..Coating::clear()
    };
    let (coated, report) = ops::coat(&base, &glaze).unwrap();
    assert!(report.is_exact(), "{report}");
    for (c, ch) in base.bound() {
        assert_eq!(coated.channel(c), Some(ch), "{c} untouched");
    }
    assert_eq!(
        coated.constant(ChannelId::Param(Param::CoatWeight)),
        Some(Value::Scalar(1.0))
    );
    let (twice, report) = ops::coat(&coated, &glaze).unwrap();
    assert!(report.has(ApproximationKind::CoatsCollapsed), "{report}");
    let tint = twice.value(ChannelId::Param(Param::CoatColor), 0);
    assert!(
        matches!(tint, Value::Vector3(t) if (t.x - 0.64).abs() < 1e-6),
        "tints multiply"
    );
}

#[test]
fn deposits_cover_raise_and_hide_the_coat() {
    let (coated, _) = ops::coat(&rich(0.1), &Coating::clear()).unwrap();
    let mut dirt = Material::new(grid());
    dirt.set_param(
        Param::BaseColor,
        Channel::Constant(Value::Vector3(Vec3::splat(0.05))),
    )
    .unwrap();
    dirt.set_param(
        Param::SpecularRoughness,
        Channel::Constant(Value::Scalar(0.95)),
    )
    .unwrap();
    dirt.set_aux(Aux::Surface, Channel::Constant(Value::Id(9)))
        .unwrap();
    let coverage = ramp_raster(|x, _| x as f32 / (N - 1) as f32);
    let (m, report) = ops::deposit(
        &coated,
        &Deposit {
            material: dirt,
            coverage,
            thickness: 0.0005,
            relief: None,
        },
    )
    .unwrap();
    let last = (N - 1) as usize;
    let h = |mat: &Material, i| mat.value(ChannelId::Aux(Aux::Height), i).scalar().unwrap();
    assert!(
        (h(&m, last) - h(&coated, last) - 0.0005).abs() < 1e-7,
        "raised"
    );
    assert_eq!(h(&m, 0), h(&coated, 0), "uncovered");
    assert_eq!(m.value(ChannelId::Aux(Aux::Surface), last), Value::Id(9));
    assert_eq!(
        m.value(ChannelId::Aux(Aux::Region), last),
        coated.value(ChannelId::Aux(Aux::Region), last),
        "ownership stays"
    );
    assert_eq!(
        m.value(ChannelId::Param(Param::CoatWeight), last),
        Value::Scalar(0.0),
        "the covered coat is hidden"
    );
    assert!(report.has(ApproximationKind::LayersMixed), "{report}");
}

#[test]
fn a_material_wide_transform_leaves_no_channel_behind() {
    let m = rich(0.3);
    // Two whole texels right: an exact permutation of every channel.
    let (moved, report) =
        ops::transform(&m, MaterialTransform::offset(Vec2::new(0.02, 0.0))).unwrap();
    assert!(report.is_exact());
    assert_eq!(
        moved.bound().count(),
        m.bound().count(),
        "every channel is still bound"
    );
    for (c, ch) in m.bound() {
        if let Channel::Map(r) = ch {
            for y in 0..i64::from(N) {
                for x in 0..i64::from(N) {
                    let i = usize::try_from(y * i64::from(N) + x).unwrap();
                    assert_eq!(moved.value(c, i), r.value_at(x - 2, y), "{c} moved");
                }
            }
        }
    }
    // A quarter turn turns normals with the texels.
    let (turned, _) = ops::transform(
        &m,
        MaterialTransform {
            flip_x: false,
            quarter_turns: 1,
            offset: Vec2::ZERO,
        },
    )
    .unwrap();
    let n = |mat: &Material, x: u32, y: u32| match mat.value(
        ChannelId::Param(Param::GeometryNormal),
        (y * N + x) as usize,
    ) {
        Value::Vector3(v) => v,
        _ => unreachable!(),
    };
    // Output (5, 2) reads source (y, N − 1 − x) = (2, 2).
    let (src, dst) = (n(&m, 2, N - 1 - 5), n(&turned, 5, 2));
    assert!(
        (dst - Vec3::new(-src.y, src.x, src.z)).length() < 1e-6,
        "turned"
    );
    // Half a texel resamples, and says so for every map it touched.
    let (_, report) = ops::transform(&m, MaterialTransform::offset(Vec2::new(0.005, 0.0))).unwrap();
    let maps = m
        .bound()
        .filter(|(_, c)| matches!(c, Channel::Map(_)))
        .count();
    assert_eq!(report.approximations.len(), maps);
    assert!(
        report
            .approximations
            .iter()
            .all(|a| a.kind == ApproximationKind::Resampled)
    );
}

#[test]
fn programs_over_materials_check_scopes() {
    let m = rich(0.2);
    let mut b = ScopedBuilder::new("dirt");
    let ao = b.input("ao", PortType::Scalar, Scope::Pass).unwrap();
    let rough = b.input("rough", PortType::Scalar, Scope::Sample).unwrap();
    let k = b.input("k", PortType::Scalar, Scope::Material).unwrap();
    let t = b.mul(ao, rough).unwrap();
    let t = b.mul(t, k).unwrap();
    assert!(
        b.clone()
            .output("x", PortType::Mask, Scope::Sample, t)
            .is_err(),
        "a pass value is not per sample"
    );
    b.output("x", PortType::Mask, Scope::Pass, t).unwrap();
    let p = b.finish();
    let ao = grid()
        .typed(PortType::Scalar, vec![Value::Scalar(0.5); 64])
        .unwrap();
    let out = evaluate(
        &p,
        &[
            MapBinding::Pass(ao.clone()),
            MapBinding::Channel(ChannelId::Param(Param::SpecularRoughness)),
            MapBinding::Constant(Value::Scalar(2.0)),
        ],
        &m,
    )
    .unwrap();
    let r = m
        .value(ChannelId::Param(Param::SpecularRoughness), 9)
        .scalar()
        .unwrap();
    assert_eq!(out[0].value(9), Value::Scalar(0.5 * r * 2.0));
    assert!(
        evaluate(
            &p,
            &[
                MapBinding::Pass(ao.clone()),
                MapBinding::Pass(ao),
                MapBinding::Constant(Value::Scalar(2.0)),
            ],
            &m,
        )
        .is_err(),
        "a pass result cannot stand for a per-sample input"
    );
}

/// A test module: a flat coated paint whose roughness varies with its
/// seed, tinted by an optional exemplar.
struct Paint;

impl Module for Paint {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId {
                name: "test.paint",
                version: 1,
            },
            doc: "flat paint",
            params: vec![
                ParamDecl {
                    name: "color",
                    kind: ParamKind::Color,
                    default: ParamValue::Color(Vec3::splat(0.5)),
                    doc: "its color",
                },
                ParamDecl {
                    name: "thickness",
                    kind: ParamKind::Scalar {
                        unit: Unit::Meters,
                        range: [0.0, 0.001],
                    },
                    default: ParamValue::Scalar(0.0001),
                    doc: "film thickness",
                },
                ParamDecl {
                    name: "seed",
                    kind: ParamKind::Seed,
                    default: ParamValue::Seed(0),
                    doc: "variation",
                },
            ],
            inputs: vec![InputDecl {
                name: "exemplar",
                kind: InputKind::Resource(ResourceRequest {
                    port: PortType::Scalar,
                    periodic: true,
                }),
                required: false,
                doc: "a tint image",
            }],
            outputs: vec![OutputDecl {
                name: "material",
                kind: OutputKind::Material,
                doc: "the paint",
            }],
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let mut m = Material::new(cx.grid());
        let tint = args
            .resource("exemplar")
            .map_or(1.0, |r| r.image.sample(Vec2::ZERO, Footprint::POINT));
        m.set_param(
            Param::BaseColor,
            Channel::Constant(Value::Vector3(args.color("color") * tint)),
        )?;
        #[expect(clippy::cast_precision_loss, reason = "a test value")]
        let r = (args.seed("roughness") % 1000) as f32 / 1000.0;
        m.set_param(
            Param::SpecularRoughness,
            Channel::Constant(Value::Scalar(r)),
        )?;
        let m = cx.record(ops::coat(&m, &Coating::clear())?);
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}

/// Paints twice, as a wall's two coats.
struct TwoCoats;

impl Module for TwoCoats {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId {
                name: "test.two_coats",
                version: 1,
            },
            doc: "two paints",
            params: vec![],
            inputs: vec![],
            outputs: vec![],
        }
    }

    fn build(&self, cx: &mut Context<'_>, _args: &Args) -> Result<Outputs, ModuleError> {
        let mut a = cx.instantiate(&Paint, "first", Bind::new())?;
        let mut b = cx.instantiate(&Paint, "second", Bind::new())?;
        Ok(Outputs::new()
            .with("a", Output::Material(a.take_material("material").unwrap()))
            .with("b", Output::Material(b.take_material("material").unwrap())))
    }
}

struct Host;

impl ResourceHost for Host {
    fn resolve(&self, reference: &ResourceRef) -> Result<Resolved, ResourceError> {
        if reference.0 != "tint.png" {
            return Err(ResourceError::NotFound(reference.clone()));
        }
        let level = ImageLevel::new(2, 2, Vec2::splat(0.5), vec![0.5_f32; 4]).unwrap();
        let image = SampleImage::typed(
            PortType::Scalar,
            SamplePolicy::Linear,
            Domain::periodic(1, 1).unwrap(),
            Vec2::ZERO,
            vec![level],
            Fingerprint(7),
        )
        .unwrap();
        Ok(Resolved {
            content: Fingerprint(7),
            image,
            mips: ReductionPolicy::Average,
        })
    }
}

#[test]
fn modules_check_bindings_derive_seeds_and_keep_boundaries() {
    let mut cx = Context::new(grid(), &NoResources);
    let err = cx
        .instantiate(&Paint, "p", Bind::new().scalar("thickness", 0.5))
        .unwrap_err();
    assert!(
        matches!(*err.kind, ModuleErrorKind::InvalidParam { .. }),
        "range"
    );
    let err = cx
        .instantiate(&Paint, "p", Bind::new().scalar("thinness", 0.0))
        .unwrap_err();
    assert_eq!(
        *err.kind,
        ModuleErrorKind::Unknown(String::from("thinness"))
    );
    let err = cx
        .instantiate(
            &Paint,
            "p",
            Bind::new().input("exemplar", Input::Resource(ResourceRef("tint.png".into()))),
        )
        .unwrap_err();
    assert!(matches!(
        *err.kind,
        ModuleErrorKind::Resource(ResourceError::NotFound(_))
    ));

    let mut cx = Context::new(grid(), &Host);
    let out = cx.instantiate(&TwoCoats, "wall", Bind::new()).unwrap();
    let rough = |name| {
        out.material(name)
            .unwrap()
            .value(ChannelId::Param(Param::SpecularRoughness), 0)
    };
    assert_ne!(
        rough("a"),
        rough("b"),
        "seeds derive from the instance path"
    );
    let paths: Vec<_> = cx
        .diagnostics()
        .instances()
        .map(|(p, _)| String::from(p))
        .collect();
    assert_eq!(paths, ["wall", "wall/first", "wall/second"]);
    assert_eq!(
        cx.diagnostics().reports_at("wall/first").count(),
        1,
        "the coat's report stays with its instance"
    );

    let tinted = cx
        .instantiate(
            &Paint,
            "tinted",
            Bind::new().input("exemplar", Input::Resource(ResourceRef("tint.png".into()))),
        )
        .unwrap();
    assert_eq!(
        tinted
            .material("material")
            .unwrap()
            .value(ChannelId::Param(Param::BaseColor), 0),
        Value::Vector3(Vec3::splat(0.25)),
        "the resolved exemplar tints"
    );
}
