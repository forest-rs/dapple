// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::path::Path;

use super::*;

fn materials() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/materials"))
}

#[test]
fn the_example_recipe_parses_and_round_trips() {
    let recipe = read_recipe(&materials().join("recipes/oak_bark.toml")).unwrap();
    assert_eq!(recipe.outputs.len(), 4);
    let fingerprint = recipe.fingerprint().unwrap();
    // Serializing and reading back keeps the recipe and its fingerprints.
    let text = toml::to_string(&recipe).unwrap();
    let back = parse_recipe(&text).unwrap();
    assert_eq!(back, recipe);
    assert_eq!(back.fingerprint().unwrap(), fingerprint);
}

#[test]
fn manifests_are_checked() {
    let manifest = read_manifest(&materials().join("materials.toml")).unwrap();
    assert_eq!(manifest.materials[0].id, "oak_bark");
    assert_eq!(manifest.materials[0].filter, FilterName::Kaiser);
    let twice: Result<Manifest, _> =
        toml::from_str("[[material]]\nid = \"a\"\nrecipe = \"r.toml\"\ntint = 1\n");
    assert!(twice.is_err(), "unknown fields are refused");
}

/// A small recipe: a disk mask as opacity over flat color channels.
fn small() -> Recipe {
    parse_recipe(
        r#"
version = 1

[[nodes]]
label = "disk"
kind = "field"
inputs = []
op = { op = "disk", domain = { periodic = { period = [1, 1] } }, center = [0.5, 0.5], radius = 0.3, softness = 0.05 }

[[nodes]]
label = "opacity"
kind = "realize"
input = "disk"
width = 32
height = 32

[[nodes]]
label = "relief"
kind = "raster"
input = "opacity"
params = { op = "height_to_normal", scale = 0.1 }

[[outputs]]
role = "opacity"
channels = ["opacity"]

[[outputs]]
role = "normal"
channels = ["relief"]
"#,
    )
    .unwrap()
}

#[test]
fn bakes_pack_each_profile_and_check_roles() {
    let entry = MaterialEntry {
        id: "leaf".into(),
        recipe: "unused".into(),
        profiles: vec![ProfileName::Lightweald, ProfileName::Raw],
        filter: FilterName::Box,
        alpha_cutoff: Some(0.5),
    };
    let baked = bake(&entry, &small()).unwrap();
    assert_eq!(baked.fingerprint, small().fingerprint().unwrap());
    let names =
        |i: usize| -> Vec<&str> { baked.bundles[i].1.textures.iter().map(|t| t.name).collect() };
    // Normal variance widens roughness, so normals bring an ORM texture.
    assert_eq!(names(0), ["base_color", "normal", "orm"]);
    assert_eq!(
        names(1),
        ["base_color", "opacity", "normal", "specular_roughness"]
    );
    assert!(!baked.bundles[0].1.report.coverage.is_empty());

    let mut unknown = small();
    unknown.outputs[0].role = "tint".into();
    assert!(matches!(
        bake(&entry, &unknown),
        Err(BakeError::UnknownRole { .. })
    ));
    let mut wrong = small();
    wrong.outputs[1].channels = vec!["opacity".into()];
    assert!(matches!(
        bake(&entry, &wrong),
        Err(BakeError::Channels { .. })
    ));
}

#[test]
fn unchanged_materials_are_skipped() {
    let dir = std::env::temp_dir().join(format!("dapple-bake-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("recipes")).unwrap();
    std::fs::write(
        dir.join("recipes/small.toml"),
        toml::to_string(&small()).unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("materials.toml"),
        "[[material]]\nid = \"small\"\nrecipe = \"recipes/small.toml\"\nprofiles = [\"lightweald\", \"gltf\"]\n",
    )
    .unwrap();
    let out = dir.join("out");
    let settings = WriteSettings {
        encodings: vec![],
        quality: Quality::Fast,
    };
    let manifest = dir.join("materials.toml");
    let first = bake_manifest(&manifest, &out, &settings, false).unwrap();
    assert!(
        matches!(first[0].1, Outcome::Baked { files: 6, .. }),
        "{first:?}"
    );
    assert!(out.join("lightweald/small/normal.ktx2").exists());
    assert!(out.join("gltf/small/base_color.png").exists());
    let again = bake_manifest(&manifest, &out, &settings, false).unwrap();
    assert_eq!(again[0].1, Outcome::Cached);
    let forced = bake_manifest(&manifest, &out, &settings, true).unwrap();
    assert!(matches!(forced[0].1, Outcome::Baked { .. }));
    std::fs::remove_dir_all(&dir).unwrap();
}
