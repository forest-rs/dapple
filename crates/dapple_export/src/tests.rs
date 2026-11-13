// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

extern crate std;

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use dapple_encode::{PackSettings, Profile};
use dapple_field::{NormalFrame, PortType, Primaries, Value};
use dapple_material::{Aux, Channel, Grid, Material, Param, param_default, param_port};

use super::*;

/// A 4 × 2 texel grid, 1 × 0.5 m, wrapping.
fn grid() -> Grid {
    Grid {
        width: 4,
        height: 2,
        ..dapple_lab::material::unit_tile(4)
    }
}

/// A three-component value. `glam` is not a dependency of this crate, so
/// the vector comes from a default and has its components replaced.
fn vec3(x: f32, y: f32, z: f32) -> Value {
    let Value::Vector3(v) = param_default(Param::BaseColor) else {
        unreachable!("colors are three-component vectors")
    };
    Value::Vector3(v.with_x(x).with_y(y).with_z(z))
}

fn scalar(v: f32) -> Channel {
    Channel::Constant(Value::Scalar(v))
}

fn map(port: PortType, f: impl Fn(usize) -> Value) -> Channel {
    let values: Vec<Value> = (0..grid().len()).map(f).collect();
    Channel::Map(grid().typed(port, values).unwrap())
}

#[expect(clippy::cast_precision_loss, reason = "small test indices")]
fn ramp(i: usize) -> f32 {
    i as f32 / 8.0
}

/// A small material that turns on every glTF extension `dapple_export`
/// writes, has something glTF cannot carry (a coat tint, a height map), a
/// constant parameter without a `dapple_encode` field (coat IOR) and a map
/// without one (thin film), which is lost.
fn showcase() -> Material {
    let color = PortType::Color(Primaries::Rec709);
    let mut m = Material::new(grid());
    let set = |m: &mut Material, p, ch| m.set_param(p, ch).unwrap();
    set(
        &mut m,
        Param::BaseColor,
        map(color, |i| vec3(0.2 + 0.5 * ramp(i), 0.3, 0.1)),
    );
    set(
        &mut m,
        Param::SpecularRoughness,
        map(PortType::Scalar, |i| Value::Scalar(0.2 + 0.4 * ramp(i))),
    );
    set(&mut m, Param::BaseMetalness, scalar(0.25));
    set(
        &mut m,
        Param::SpecularWeight,
        map(PortType::Scalar, |i| {
            Value::Scalar(if i % 2 == 0 { 0.5 } else { 1.5 })
        }),
    );
    set(
        &mut m,
        Param::SpecularColor,
        Channel::Constant(vec3(1.0, 0.9, 0.8)),
    );
    set(&mut m, Param::SpecularIor, scalar(1.4));
    set(&mut m, Param::SpecularRoughnessAnisotropy, scalar(0.5));
    set(
        &mut m,
        Param::GeometryTangent,
        Channel::Constant(vec3(0.0, 1.0, 0.0)),
    );
    set(&mut m, Param::TransmissionWeight, scalar(0.1));
    set(&mut m, Param::SubsurfaceWeight, scalar(0.3));
    set(
        &mut m,
        Param::SubsurfaceColor,
        Channel::Constant(vec3(0.9, 0.5, 0.4)),
    );
    set(
        &mut m,
        Param::CoatWeight,
        map(PortType::Scalar, |i| Value::Scalar(ramp(i))),
    );
    set(&mut m, Param::CoatRoughness, scalar(0.05));
    set(
        &mut m,
        Param::CoatColor,
        Channel::Constant(vec3(0.9, 0.8, 0.6)),
    );
    set(&mut m, Param::CoatIor, scalar(1.45));
    set(
        &mut m,
        Param::GeometryCoatNormal,
        map(PortType::Normal(NormalFrame::Domain), |i| {
            if i % 2 == 0 {
                vec3(0.0, 0.0, 1.0)
            } else {
                vec3(0.6, 0.0, 0.8)
            }
        }),
    );
    set(&mut m, Param::FuzzWeight, scalar(0.5));
    set(
        &mut m,
        Param::FuzzColor,
        Channel::Constant(vec3(0.4, 0.6, 0.2)),
    );
    set(&mut m, Param::EmissionLuminance, scalar(40.0));
    set(
        &mut m,
        Param::EmissionColor,
        map(color, |i| vec3(1.0, ramp(i), 0.0)),
    );
    set(
        &mut m,
        Param::ThinFilmWeight,
        map(PortType::Scalar, |i| Value::Scalar(ramp(i))),
    );
    m.set_aux(
        Aux::Height,
        map(PortType::Scalar, |i| Value::Scalar(0.001 * ramp(i))),
    )
    .unwrap();
    m
}

fn exported(settings: &Settings) -> Exported {
    let m = showcase();
    let mut report = dapple_lab::Report::new("showcase");
    report.check("demo.passes", true, "");
    report.check("demo.fails", false, "on purpose");
    let source = Source {
        lab: Some(&report),
        ..Source::new("showcase", &m)
    };
    export(&source, settings).unwrap()
}

/// Compares `actual` with `expected`, the golden file `name` as it was
/// compiled in; with `DAPPLE_BLESS` set, rewrites the golden instead.
///
/// The golden is compiled in rather than read when the test runs so the test
/// needs no file system, which `wasm32-wasip1` runs without. A checkout may
/// have turned its line endings into CRLF, so those compare as LF.
fn golden(name: &str, expected: &str, actual: &str) {
    if std::env::var_os("DAPPLE_BLESS").is_some() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/goldens")
            .join(name);
        std::fs::write(&path, actual).unwrap();
        return;
    }
    assert!(
        expected.replace("\r\n", "\n") == actual,
        "src/goldens/{name} differs; rerun with DAPPLE_BLESS=1 to accept:\n{actual}"
    );
}

/// The manifest of `exported` as written beside its glTF document alone:
/// the PNG files' bytes depend on the `png` crate's compressor, and are
/// covered by `writes_a_directory_that_reads_back`.
fn manifest_without_pngs(exported: &Exported) -> Manifest {
    let gltf = gltf::document(exported).unwrap();
    exported.manifest(vec![FileEntry {
        path: GLTF_FILE.to_string(),
        bytes: gltf.len() as u64,
        hash: content_hash(gltf.as_bytes()).to_string(),
    }])
}

#[test]
fn gltf_and_manifest_match_their_goldens() {
    let exported = exported(&Settings::default());
    let gltf = gltf::document(&exported).unwrap();
    golden(
        "showcase.gltf",
        include_str!("goldens/showcase.gltf"),
        &gltf,
    );
    golden(
        "showcase.manifest.json",
        include_str!("goldens/showcase.manifest.json"),
        &manifest_without_pngs(&exported).to_json(),
    );
    let fragment = gltf::material(&exported).unwrap();
    assert_eq!(
        fragment.extensions_used,
        [
            "KHR_materials_anisotropy",
            "KHR_materials_clearcoat",
            "KHR_materials_diffuse_transmission",
            "KHR_materials_emissive_strength",
            "KHR_materials_ior",
            "KHR_materials_sheen",
            "KHR_materials_specular",
            "KHR_materials_transmission",
        ],
        "every extension"
    );
}

#[test]
fn the_account_merges_lowering_packing_and_gltf() {
    let exported = exported(&Settings::default());
    let held = |name: &str| exported.parameter(name).unwrap();
    // Maps glTF carries are in its textures, constants are factors.
    let coat = held("coat_weight");
    assert!(coat.in_profile);
    assert!(
        matches!(&coat.held, Held::Texture(l) if l.texture == "clearcoat" && l.channels == (0..1))
    );
    assert_eq!(
        held("base_metalness").held,
        Held::Texture(Location {
            texture: "orm".into(),
            channels: 2..3,
            range: [0.0, 1.0],
            encoding: Encoding::Value,
        }),
        "a constant that shares a texture with a map is placed in it"
    );
    // A roughness map makes glTF's anisotropy strength vary, so the
    // constant tangent and anisotropy are placed in a texture.
    assert!(matches!(
        &held("geometry_tangent").held,
        Held::Texture(l) if l.texture == "anisotropy" && l.encoding == Encoding::AxisYUp
    ));
    assert!(matches!(
        &held("specular_roughness_anisotropy").held,
        Held::Texture(l) if l.encoding == Encoding::GltfAnisotropyStrength
    ));
    // glTF cannot carry the coat tint or height: the raw profile's
    // textures hold them.
    for (name, texture) in [("coat_color", None), ("height", Some("height"))] {
        let p = held(name);
        assert!(!p.in_profile, "{name}");
        match texture {
            Some(t) => {
                assert!(matches!(&p.held, Held::Texture(l) if l.texture == t));
                assert!(!exported.texture(t).unwrap().in_profile);
            }
            None => assert_eq!(
                p.held,
                Held::Constant {
                    value: vec![0.9, 0.8, 0.6]
                }
            ),
        }
    }
    // No field: a constant is kept, a map is lost.
    let ior = held("coat_ior");
    assert_eq!(ior.held, Held::Constant { value: vec![1.45] });
    assert!(!ior.in_profile);
    assert!(ior.notes[0].contains("1.5"), "{:?}", ior.notes);
    // The thin film ramps over 0, 1/8, ..., 7/8: its mean stands in.
    assert_eq!(
        held("thin_film_weight").held,
        Held::Lost { mean: vec![0.4375] }
    );
    // glTF fixes the coat's darkening, which matters on a coated material.
    let darkening = held("coat_darkening");
    assert_eq!(darkening.held, Held::Default { value: vec![1.0] });
    assert!(!darkening.in_profile);
    // Approximations are notes on parameters glTF still carries.
    for name in [
        "specular_weight",
        "specular_roughness_anisotropy",
        "subsurface_weight",
        "transmission_weight",
        "fuzz_weight",
    ] {
        let p = held(name);
        assert!(p.in_profile && !p.notes.is_empty(), "{name}: {p:?}");
    }
    // Sheen color is fuzz color times weight; both constants, so factors.
    assert!(matches!(held("fuzz_color").held, Held::Constant { .. }));
    // The lab's verdict is kept.
    let lab = exported.lab.as_ref().unwrap();
    assert!(!lab.passed);
    assert_eq!(lab.failures, ["demo.fails"]);
}

#[test]
fn constant_anisotropy_becomes_gltf_factors() {
    let mut m = Material::new(grid());
    m.set_param(Param::SpecularRoughness, scalar(0.4)).unwrap();
    m.set_param(Param::SpecularRoughnessAnisotropy, scalar(0.5))
        .unwrap();
    // 90° from +X in the domain frame, whose +Y runs down the image.
    m.set_param(
        Param::GeometryTangent,
        Channel::Constant(vec3(0.0, 2.0, 0.5)),
    )
    .unwrap();
    let exported = export(&Source::new("brushed", &m), &Settings::default()).unwrap();
    // A tangent is stated as the material has it, not as packing's doubled
    // angle.
    assert_eq!(
        exported.parameter("geometry_tangent").unwrap().held,
        Held::Constant {
            value: vec![0.0, 2.0, 0.5]
        }
    );
    let fragment = gltf::material(&exported).unwrap();
    let e = &fragment.material["extensions"]["KHR_materials_anisotropy"];
    let strength = dapple_encode::gltf_anisotropy_strength(0.4, 0.5);
    #[expect(clippy::cast_possible_truncation, reason = "written from an f32")]
    let written = e["anisotropyStrength"].as_f64().unwrap() as f32;
    assert_eq!(written, strength, "the f32, in its shortest form");
    let rotation = e["anisotropyRotation"].as_f64().unwrap();
    assert!(
        (rotation + core::f64::consts::FRAC_PI_2).abs() < 1e-6,
        "{rotation}"
    );
    assert!(fragment.textures.is_empty());
}

#[test]
fn a_material_that_sets_everything_is_accounted_for_once() {
    let mut m = Material::new(grid());
    for p in Param::ALL {
        let value = match param_default(p) {
            Value::Scalar(0.0) => Value::Scalar(0.5),
            Value::Scalar(v) => Value::Scalar(v * 0.9),
            Value::Vector3(v) if p == Param::GeometryTangent => {
                Value::Vector3(v.with_x(0.0).with_y(1.0))
            }
            Value::Vector3(v) if matches!(param_port(p), PortType::Normal(_)) => {
                Value::Vector3(v.with_x(0.6).with_z(0.8))
            }
            Value::Vector3(v) => Value::Vector3(v * 0.5 + 0.1),
            other => other,
        };
        m.set_param(p, Channel::Constant(value)).unwrap();
    }
    m.set_aux(Aux::Occlusion, scalar(0.5)).unwrap();
    m.set_aux(Aux::Height, scalar(0.01)).unwrap();
    for profile in [Profile::Gltf, Profile::Lightweald, Profile::Raw] {
        let settings = Settings {
            profile,
            pack: PackSettings::default(),
        };
        let exported = export(&Source::new("everything", &m), &settings).unwrap();
        let names: Vec<&str> = exported
            .parameters
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        let expected: Vec<&str> = Param::ALL
            .iter()
            .map(|p| p.identifier())
            .chain(["occlusion", "height"])
            .collect();
        assert_eq!(names, expected, "each once, in order");
        for p in &exported.parameters {
            assert!(
                !matches!(p.held, Held::Default { .. }),
                "{profile:?}: {} was set",
                p.name
            );
            if !p.in_profile {
                assert!(!p.notes.is_empty(), "{profile:?}: a loss says why: {p:?}");
            }
        }
    }
}

#[test]
fn exports_are_deterministic() {
    let settings = Settings::default();
    let (a, b) = (exported(&settings), exported(&settings));
    assert_eq!(a, b);
    assert_eq!(gltf::document(&a), gltf::document(&b));
    assert_eq!(
        manifest_without_pngs(&a).to_json(),
        manifest_without_pngs(&b).to_json()
    );
}

#[test]
fn manifests_round_trip() {
    let exported = exported(&Settings::default());
    let manifest = manifest_without_pngs(&exported);
    let text = manifest.to_json();
    let read = Manifest::from_json(&text).unwrap();
    assert_eq!(read, manifest);
    assert_eq!(read.to_json(), text, "reading and writing is the identity");

    let newer = text.replace("\"version\": 1", "\"version\": 2");
    assert!(matches!(
        Manifest::from_json(&newer),
        Err(ManifestError::Version(2))
    ));
    let other = text.replace("dapple-material", "something-else");
    assert!(matches!(
        Manifest::from_json(&other),
        Err(ManifestError::Format(_))
    ));
    let extra = text.replacen('{', "{\n  \"surprise\": true,", 1);
    assert!(matches!(
        Manifest::from_json(&extra),
        Err(ManifestError::Syntax(_))
    ));
}

#[test]
fn other_profiles_have_no_gltf() {
    for profile in [Profile::Raw, Profile::Lightweald] {
        let settings = Settings {
            profile,
            ..Settings::default()
        };
        let exported = exported(&settings);
        assert!(gltf::document(&exported).is_none());
        assert!(exported.manifest(Vec::new()).gltf.is_none());
    }
    // Raw carries every given parameter it has a field for.
    let raw = exported(&Settings {
        profile: Profile::Raw,
        ..Settings::default()
    });
    assert!(raw.textures.iter().all(|t| t.in_profile));
    let coat_color = raw.parameter("coat_color").unwrap();
    assert!(coat_color.in_profile);
}

// `wasm32-wasip1` tests run without a directory to write to.
#[cfg(all(feature = "std", not(target_os = "wasi")))]
#[test]
fn writes_a_directory_that_reads_back() {
    let exported = exported(&Settings::default());
    let dir =
        std::env::temp_dir().join(alloc::format!("dapple_export-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let written = exported.write_dir(&dir).unwrap();
    assert_eq!(read_dir(&dir).unwrap(), written);
    let files: Vec<String> = written.files.iter().map(|f| f.path.clone()).collect();
    let mut expected: Vec<String> = exported.textures.iter().map(Texture::file).collect();
    expected.push(GLTF_FILE.to_string());
    assert_eq!(files, expected);
    assert!(written.changed_files(&dir).is_empty());

    // Writing again gives the same bytes.
    let first: Vec<Vec<u8>> = files
        .iter()
        .chain([&MANIFEST_FILE.to_string()])
        .map(|f| std::fs::read(dir.join(f)).unwrap())
        .collect();
    exported.write_dir(&dir).unwrap();
    for (f, bytes) in files.iter().chain([&MANIFEST_FILE.to_string()]).zip(&first) {
        assert_eq!(&std::fs::read(dir.join(f)).unwrap(), bytes, "{f}");
    }

    // A changed or missing file is noticed.
    std::fs::write(dir.join("orm.png"), b"not a png").unwrap();
    std::fs::remove_file(dir.join("height.png")).unwrap();
    assert_eq!(written.changed_files(&dir), ["orm.png", "height.png"]);
    std::fs::remove_dir_all(&dir).unwrap();
}
