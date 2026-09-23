// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Bakes the dapple material recipes a manifest lists; see the library docs.
//!
//! ```text
//! dapple_bake --manifest <materials.toml> [--out <dir>]
//!             [--encoding uncompressed,bc,astc] [--quality fast|balanced|best]
//!             [--force]
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use dapple_bake::{Outcome, WriteSettings, bake_manifest};
use dapple_compress::{Encoding, Quality};

const USAGE: &str = "usage: dapple_bake --manifest <materials.toml> [--out <dir>] \
[--encoding uncompressed,bc,astc] [--quality fast|balanced|best] [--force]";

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("dapple_bake: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    let mut manifest = None;
    let mut out = PathBuf::from("target/dapple-bake");
    let mut settings = WriteSettings {
        encodings: vec![Encoding::Bc],
        quality: Quality::Balanced,
    };
    let mut force = false;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let mut value = || {
            args.next()
                .ok_or_else(|| format!("{arg} needs a value\n{USAGE}"))
        };
        match arg.as_str() {
            "--manifest" => manifest = Some(PathBuf::from(value()?)),
            "--out" => out = PathBuf::from(value()?),
            "--encoding" => {
                settings.encodings = value()?
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|name| match name {
                        "uncompressed" => Ok(Encoding::Uncompressed),
                        "bc" => Ok(Encoding::Bc),
                        "astc" => Ok(Encoding::Astc),
                        other => Err(format!("unknown encoding {other}\n{USAGE}")),
                    })
                    .collect::<Result<_, _>>()?;
            }
            "--quality" => {
                settings.quality = match value()?.as_str() {
                    "fast" => Quality::Fast,
                    "balanced" => Quality::Balanced,
                    "best" => Quality::Best,
                    other => return Err(format!("unknown quality {other}\n{USAGE}")),
                };
            }
            "--force" => force = true,
            other => return Err(format!("unexpected argument {other}\n{USAGE}")),
        }
    }
    let manifest = manifest.ok_or_else(|| USAGE.to_owned())?;
    let outcomes = bake_manifest(&manifest, &out, &settings, force).map_err(|e| e.to_string())?;
    for (id, outcome) in outcomes {
        match outcome {
            Outcome::Cached => println!("{id}: unchanged, skipped"),
            Outcome::Baked {
                files,
                bytes,
                unsupported,
            } => {
                println!("{id}: {files} files, {bytes} bytes");
                for (profile, maps) in unsupported {
                    println!("  {profile:?} cannot carry: {}", maps.join(", "));
                }
            }
        }
    }
    println!("output in {}", out.display());
    Ok(())
}
