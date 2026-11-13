// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `--profile` records a capture and changes nothing the bake writes.
//!
//! Each run is its own process (the tool installs the global backend), so the
//! test drives the binary.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use footfall::session::OwnedObservation;
use footfall::session::capture::CaptureReader;

const MANIFEST: &str = r#"
[[material]]
id = "leaf"
recipe = "leaf.toml"
profiles = ["lightweald", "gltf"]
"#;

const RECIPE: &str = r#"
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
"#;

fn bake(dir: &Path, out: &str, extra: &[&Path]) -> String {
    let mut command = Command::new(env!("CARGO_BIN_EXE_dapple_bake"));
    command
        .arg("--manifest")
        .arg(dir.join("materials.toml"))
        .arg("--out")
        .arg(dir.join(out))
        .args(["--encoding", "uncompressed,bc", "--quality", "fast"]);
    if let [profile] = extra {
        command.arg("--profile").arg(profile);
    }
    let output = command.output().expect("the tool runs");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("utf-8")
}

/// Every file under `root`, relative, with its bytes.
fn tree(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    fn walk(root: &Path, dir: &Path, files: &mut BTreeMap<PathBuf, Vec<u8>>) {
        for entry in fs::read_dir(dir).expect("a readable directory") {
            let path = entry.expect("an entry").path();
            if path.is_dir() {
                walk(root, &path, files);
            } else {
                let rel = path.strip_prefix(root).expect("under the root").to_owned();
                files.insert(rel, fs::read(&path).expect("a readable file"));
            }
        }
    }
    let mut files = BTreeMap::new();
    walk(root, root, &mut files);
    files
}

#[test]
fn a_profile_records_the_bake_and_changes_no_output() {
    let dir = std::env::temp_dir().join(format!("dapple_bake_profile_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("a scratch directory");
    fs::write(dir.join("materials.toml"), MANIFEST).expect("manifest");
    fs::write(dir.join("leaf.toml"), RECIPE).expect("recipe");
    let capture = dir.join("bake.ffcap");

    let plain_log = bake(&dir, "plain", &[]);
    let profiled_log = bake(&dir, "profiled", &[&capture]);

    // The same files, byte for byte.
    let (plain, profiled) = (tree(&dir.join("plain")), tree(&dir.join("profiled")));
    assert!(plain.len() > 4, "the bake wrote textures and a stamp");
    assert_eq!(plain, profiled, "--profile changes what is written");
    // The same report, but for the lines that name a directory or the profile.
    let stable = |log: &str| {
        log.lines()
            .filter(|l| !l.starts_with("output in") && !l.starts_with("profile in"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert_eq!(stable(&plain_log), stable(&profiled_log));
    assert!(profiled_log.contains("profile in"), "the tool says where");

    // The capture reads back and holds the bake's scopes and counters.
    let joined = CaptureReader::new(fs::File::open(&capture).expect("a capture"))
        .expect("a capture header")
        .join()
        .expect("snapshots");
    assert!(joined.complete(), "the capture is whole");
    let snapshot = joined.into_snapshot();
    let count = |wanted: &str| {
        snapshot
            .events()
            .iter()
            .filter(|e| {
                matches!(e.observation(), OwnedObservation::ScopeBegin { name, .. } if name == wanted)
            })
            .count()
    };
    assert_eq!(count("dapple_bake.manifest"), 1);
    assert_eq!(count("dapple_bake.material"), 1);
    assert_eq!(count("dapple_bake.images"), 1);
    assert_eq!(count("dapple_bake.pack"), 2, "one per profile");
    assert_eq!(count("dapple_graph.run"), 1);
    assert_eq!(count("dapple_graph.node"), 3, "the recipe has three nodes");
    // One KTX2 per lightweald texture, each encoded once per requested
    // encoding into a subdirectory; the other profile writes PNG files.
    let under = |dir: &str, ext: &str, depth: usize| {
        plain
            .keys()
            .filter(|p| {
                p.starts_with(dir)
                    && p.extension().is_some_and(|e| e == ext)
                    && p.iter().count() == depth
            })
            .count()
    };
    let textures = under("lightweald/leaf", "ktx2", 3);
    assert!(textures > 0);
    assert_eq!(count("dapple_bake.ktx2"), textures);
    assert_eq!(count("dapple_bake.compress"), 2 * textures);
    assert_eq!(count("dapple_bake.png"), under("gltf/leaf", "png", 3));
    let counter = |wanted: &str| {
        snapshot
            .counters()
            .iter()
            .filter(|c| c.name() == wanted)
            .map(|c| c.value())
            .sum::<u64>()
    };
    assert_eq!(counter("dapple_bake.materials_baked"), 1);
    assert_eq!(
        counter("dapple_bake.files_written"),
        plain.len() as u64,
        "every file, the stamp included"
    );
    assert_eq!(
        counter("dapple_bake.bytes_written"),
        plain.values().map(|b| b.len() as u64).sum::<u64>()
    );
    let _ = fs::remove_dir_all(&dir);
}
