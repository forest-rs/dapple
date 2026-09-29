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
version = 2

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

    // A matching stamp is not enough when a texture has gone missing.
    std::fs::remove_file(out.join("gltf/small/base_color.png")).unwrap();
    let repaired = bake_manifest(&manifest, &out, &settings, false).unwrap();
    assert!(
        matches!(repaired[0].1, Outcome::Baked { .. }),
        "{repaired:?}"
    );
    assert!(out.join("gltf/small/base_color.png").exists());

    // Dropping a profile preserves its earlier outputs.
    std::fs::write(
        &manifest,
        "[[material]]\nid = \"small\"\nrecipe = \"recipes/small.toml\"\nprofiles = [\"gltf\"]\n",
    )
    .unwrap();
    bake_manifest(&manifest, &out, &settings, false).unwrap();
    assert!(out.join("lightweald/small/normal.ktx2").exists());
    assert!(out.join("gltf/small/normal.png").exists());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn incomplete_stamps_do_not_skip_missing_outputs() {
    let dir = std::env::temp_dir().join(format!("dapple-bake-incomplete-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("recipes")).unwrap();
    std::fs::write(
        dir.join("recipes/small.toml"),
        toml::to_string(&small()).unwrap(),
    )
    .unwrap();
    let manifest = dir.join("materials.toml");
    std::fs::write(
        &manifest,
        "[[material]]\nid = \"small\"\nrecipe = \"recipes/small.toml\"\nprofiles = [\"gltf\"]\n",
    )
    .unwrap();
    let out = dir.join("out");
    let settings = WriteSettings {
        encodings: vec![],
        quality: Quality::Fast,
    };
    bake_manifest(&manifest, &out, &settings, true).unwrap();
    let stamp_path = out.join(".stamps/small");
    let stamp = std::fs::read_to_string(&stamp_path).unwrap();
    // No proper prefix is a complete stamp, even at a file boundary.
    for end in 0..stamp.len() {
        std::fs::write(&stamp_path, &stamp[..end]).unwrap();
        assert!(
            read_stamp(&stamp_path).is_none(),
            "accepted prefix of {end} bytes"
        );
    }
    let end = stamp.find("\nfiles\n").unwrap() + "\nfiles\n".len();
    std::fs::write(&stamp_path, &stamp[..end]).unwrap();
    std::fs::remove_file(out.join("gltf/small/base_color.png")).unwrap();
    let repaired = bake_manifest(&manifest, &out, &settings, false).unwrap();
    assert!(
        matches!(repaired[0].1, Outcome::Baked { .. }),
        "{repaired:?}"
    );
    assert!(out.join("gltf/small/base_color.png").exists());
    // Removing a listed path while keeping the footer is incomplete too.
    let stamp = std::fs::read_to_string(&stamp_path).unwrap();
    let incomplete = stamp.replace("gltf/small/base_color.png\n", "");
    std::fs::write(&stamp_path, incomplete).unwrap();
    assert!(read_stamp(&stamp_path).is_none());
    // An intentionally empty output list still has a complete cache stamp.
    std::fs::write(
        &manifest,
        "[[material]]\nid = \"small\"\nrecipe = \"recipes/small.toml\"\nprofiles = []\n",
    )
    .unwrap();
    bake_manifest(&manifest, &out, &settings, false).unwrap();
    assert!(read_stamp(&stamp_path).unwrap().files.is_empty());
    assert_eq!(
        bake_manifest(&manifest, &out, &settings, false).unwrap()[0].1,
        Outcome::Cached
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn ids_must_be_plain_directory_names() {
    for id in [
        "",
        ".",
        "..",
        "a/b",
        "/abs",
        "../up",
        "a\nb",
        "x\n../victim",
        "a\rb",
    ] {
        let text = format!("[[material]]\nid = {id:?}\nrecipe = \"r.toml\"\n");
        let dir = std::env::temp_dir().join(format!("dapple-bake-ids-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("materials.toml");
        std::fs::write(&path, text).unwrap();
        assert!(
            matches!(read_manifest(&path), Err(BakeError::InvalidId(_))),
            "{id:?}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[test]
fn stamps_naming_paths_outside_the_output_delete_nothing() {
    let dir = std::env::temp_dir().join(format!("dapple-bake-stamp-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("recipes")).unwrap();
    std::fs::write(
        dir.join("recipes/small.toml"),
        toml::to_string(&small()).unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("materials.toml"),
        "[[material]]\nid = \"small\"\nrecipe = \"recipes/small.toml\"\nprofiles = [\"gltf\"]\n",
    )
    .unwrap();
    let out = dir.join("out");
    let manifest = dir.join("materials.toml");
    let settings = WriteSettings {
        encodings: vec![],
        quality: Quality::Fast,
    };
    bake_manifest(&manifest, &out, &settings, false).unwrap();

    // Files outside the output that a tampered stamp tries to reach.
    let victim = dir.join("victim.txt");
    std::fs::write(&victim, "keep").unwrap();
    let stamp_path = out.join(".stamps/small");
    let stamp = std::fs::read_to_string(&stamp_path).unwrap();
    assert!(stamp.starts_with(&format!("dapple_bake {STAMP_VERSION}\n")));
    let count = read_stamp(&stamp_path).unwrap().files.len();
    let (body, _) = stamp.strip_suffix('\n').unwrap().rsplit_once('\n').unwrap();
    for evil in [
        victim.to_string_lossy().into_owned(),
        "../victim.txt".to_owned(),
        "gltf/../../victim.txt".to_owned(),
    ] {
        std::fs::write(
            &stamp_path,
            format!("{body}\n{evil}\ncomplete {}\n", count + 1),
        )
        .unwrap();
        assert!(read_stamp(&stamp_path).is_none(), "accepted {evil:?}");
        let again = bake_manifest(&manifest, &out, &settings, false).unwrap();
        assert!(matches!(again[0].1, Outcome::Baked { .. }));
        assert!(
            victim.exists(),
            "{evil:?} deleted a file outside the output"
        );
    }

    #[cfg(unix)]
    {
        // Even a valid relative path through a symlink cannot authorize
        // deletion of an output or of a file outside the output directory.
        std::os::unix::fs::symlink(&dir, out.join("link")).unwrap();
        std::fs::write(
            &stamp_path,
            format!("{body}\nlink/victim.txt\ncomplete {}\n", count + 1),
        )
        .unwrap();
        assert!(read_stamp(&stamp_path).is_some());
        bake_manifest(&manifest, &out, &settings, true).unwrap();
        assert!(victim.exists());
    }

    // A stamp from another version no longer matches.
    let older = stamp.replacen(
        &format!("dapple_bake {STAMP_VERSION}"),
        &format!("dapple_bake {}", STAMP_VERSION + 1),
        1,
    );
    std::fs::write(&stamp_path, older).unwrap();
    let rebaked = bake_manifest(&manifest, &out, &settings, false).unwrap();
    assert!(matches!(rebaked[0].1, Outcome::Baked { .. }));

    std::fs::remove_dir_all(&dir).unwrap();
}
