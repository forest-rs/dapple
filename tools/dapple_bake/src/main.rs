// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Bakes the dapple material recipes a manifest lists; see the library docs.
//!
//! ```text
//! dapple_bake --manifest <materials.toml> [--out <dir>]
//!             [--encoding uncompressed,bc,astc] [--quality fast|balanced|best]
//!             [--force] [--profile <capture.ffcap>]
//! ```
//!
//! `--profile` records where the bake spent its time (the scopes and
//! counters the library docs list, plus those of `dapple_graph`) to a
//! `footfall` capture, which `footfall analyze` reads. It changes nothing the
//! bake writes.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use dapple_bake::{Outcome, WriteSettings, bake_manifest};
use dapple_compress::{Encoding, Quality};
use footfall::runtime::Runtime;
use footfall::session::capture::CaptureRecorder;

const USAGE: &str = "usage: dapple_bake --manifest <materials.toml> [--out <dir>] \
[--encoding uncompressed,bc,astc] [--quality fast|balanced|best] [--force] \
[--profile <capture.ffcap>]";

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
    let mut out = repository().join(".local/gallery/dapple-bake");
    let mut settings = WriteSettings {
        encodings: vec![Encoding::Bc],
        quality: Quality::Balanced,
    };
    let mut force = false;
    let mut profile = None;
    let mut args = args.into_iter().peekable();
    while let Some(arg) = args.next() {
        let mut value = || {
            args.next_if(|v| !v.starts_with("--"))
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
            "--profile" => profile = Some(PathBuf::from(value()?)),
            other => return Err(format!("unexpected argument {other}\n{USAGE}")),
        }
    }
    let manifest = manifest.ok_or_else(|| USAGE.to_owned())?;
    let capture = profile.as_deref().map(Capture::start).transpose()?;
    let outcomes = bake_manifest(&manifest, &out, &settings, force).map_err(|e| e.to_string());
    // Keep what was captured even when the bake failed: that is when a
    // profile is most wanted.
    if let Some(capture) = capture {
        capture.finish()?;
    }
    let outcomes = outcomes?;
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

/// A `footfall` runtime installed for this process, and the capture file it
/// is written to when the bake ends.
struct Capture {
    recorder: CaptureRecorder<'static, BufWriter<File>>,
    path: PathBuf,
}

impl Capture {
    /// Events the runtime retains until the capture is written: far more than
    /// a bake emits (a few hundred per material), so none are evicted.
    const EVENTS: usize = 1 << 18;

    fn start(path: &Path) -> Result<Self, String> {
        let file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let runtime = Runtime::builder()
            .process_name("dapple_bake")
            .event_capacity(Self::EVENTS)
            .install_global()
            .map_err(|e| e.to_string())?;
        let recorder = CaptureRecorder::new(runtime, BufWriter::new(file))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self {
            recorder,
            path: path.to_owned(),
        })
    }

    fn finish(self) -> Result<(), String> {
        let Self { recorder, path } = self;
        let finished = recorder
            .finish()
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if finished.missed() > 0 {
            eprintln!(
                "dapple_bake: the profile lost {} events the runtime evicted",
                finished.missed()
            );
        }
        println!(
            "profile in {} ({} events)",
            path.display(),
            finished.events()
        );
        Ok(())
    }
}

/// The repository root: this crate's manifest sits two levels below it.
fn repository() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the crate lives two levels below the repository root")
}
