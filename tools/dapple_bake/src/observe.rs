// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! What a bake tells [`footfall`]: where time goes, per material and step.
//!
//! Nothing is recorded unless the host installs a backend, as `dapple_bake
//! --profile` does. Everything is emitted under the root context, and nothing
//! emitted here is read back, so a profile never changes what is written.
//!
//! Scopes, outermost first:
//!
//! - `dapple_bake.manifest`: one [`bake_manifest`](crate::bake_manifest).
//! - `dapple_bake.material`: one manifest entry, cached or baked, with the
//!   argument `index` (its position in the manifest). A message,
//!   `dapple_bake.material_id`, carries the entry's id as its text just
//!   before, because arguments cannot hold a string that is not `'static`.
//! - `dapple_bake.images`: building the recipe's graph, running it and
//!   interleaving the output channels into material maps.
//! - `dapple_bake.pack`: packing the maps for one profile, with the argument
//!   `profile`.
//! - `dapple_bake.ktx2`: serializing one texture as uncompressed KTX2.
//! - `dapple_bake.compress`: encoding one texture, with the argument
//!   `encoding`.
//! - `dapple_bake.png`: encoding one texture as PNG.
//! - `dapple_bake.write`: writing one file.
//!
//! Counters: `dapple_bake.materials_baked`, `dapple_bake.materials_cached`,
//! `dapple_bake.files_written` and `dapple_bake.bytes_written` (bytes), the
//! last two counting every file written, stamps included.

use footfall::{Args, ContextHandle, CounterDesc, EventDesc, ScopeDesc, ScopeGuard};

use crate::ProfileName;

static MANIFEST: ScopeDesc = ScopeDesc::new("dapple_bake.manifest");
static MATERIAL: ScopeDesc = ScopeDesc::new("dapple_bake.material");
static IMAGES: ScopeDesc = ScopeDesc::new("dapple_bake.images");
static PACK: ScopeDesc = ScopeDesc::new("dapple_bake.pack");
static KTX2: ScopeDesc = ScopeDesc::new("dapple_bake.ktx2");
static COMPRESS: ScopeDesc = ScopeDesc::new("dapple_bake.compress");
static PNG: ScopeDesc = ScopeDesc::new("dapple_bake.png");
static WRITE: ScopeDesc = ScopeDesc::new("dapple_bake.write");

static MATERIAL_ID: EventDesc = EventDesc::new("dapple_bake.material_id");

static MATERIALS_BAKED: CounterDesc = CounterDesc::count("dapple_bake.materials_baked");
static MATERIALS_CACHED: CounterDesc = CounterDesc::count("dapple_bake.materials_cached");
static FILES_WRITTEN: CounterDesc = CounterDesc::count("dapple_bake.files_written");
static BYTES_WRITTEN: CounterDesc = CounterDesc::bytes("dapple_bake.bytes_written");

const ROOT: ContextHandle = ContextHandle::ROOT;

pub(crate) fn manifest() -> ScopeGuard {
    footfall::scope(&MANIFEST, ROOT)
}

/// Names the entry (by message) and opens its scope.
pub(crate) fn material(index: usize, id: &str) -> ScopeGuard {
    let args = Args::none().with("index", index as u64);
    footfall::message_with(&MATERIAL_ID, ROOT, id, &args);
    footfall::scope_with(&MATERIAL, ROOT, &args)
}

pub(crate) fn images() -> ScopeGuard {
    footfall::scope(&IMAGES, ROOT)
}

pub(crate) fn pack(profile: ProfileName) -> ScopeGuard {
    let args = Args::none().with("profile", profile.dir());
    footfall::scope_with(&PACK, ROOT, &args)
}

pub(crate) fn ktx2() -> ScopeGuard {
    footfall::scope(&KTX2, ROOT)
}

pub(crate) fn compress(encoding: &'static str) -> ScopeGuard {
    let args = Args::none().with("encoding", encoding);
    footfall::scope_with(&COMPRESS, ROOT, &args)
}

pub(crate) fn png() -> ScopeGuard {
    footfall::scope(&PNG, ROOT)
}

pub(crate) fn write() -> ScopeGuard {
    footfall::scope(&WRITE, ROOT)
}

pub(crate) fn cached() {
    footfall::counter_add(&MATERIALS_CACHED, ROOT, 1);
}

pub(crate) fn baked() {
    footfall::counter_add(&MATERIALS_BAKED, ROOT, 1);
}

pub(crate) fn written(bytes: u64) {
    footfall::counter_add(&FILES_WRITTEN, ROOT, 1);
    footfall::counter_add(&BYTES_WRITTEN, ROOT, bytes);
}
