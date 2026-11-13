// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The instance memo: results of module instances, kept by their exact
//! arguments and handed back instead of running a body again.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use core::mem::size_of;

use dapple_raster::typed::{StorageKind, TypedRaster};

use crate::material::{Aux, Channel, Material};
use crate::module::{Entry, Event, ModuleId, Output, Outputs};
use crate::report::Approximation;
use crate::resource::ResourceRef;

/// Results of module instances, kept by everything that decides them and
/// returned by [`Context::instantiate`](crate::module::Context::instantiate)
/// in place of running the body again.
///
/// **Opt in, owned by the caller.** A context uses a cache only when one
/// is attached ([`Context::with_cache`](crate::module::Context::with_cache)),
/// and a cache holds nothing the caller did not build with it. Keep one
/// across builds that repeat work: interactive parameter edits, fits that
/// build one module hundreds of times, preset sweeps.
///
/// **What a result is kept by.** An instance's key is its path, its
/// [`ModuleId`], and the exact words its
/// [`Args::fingerprint`](crate::module::Args::fingerprint) hashes, not the
/// hash: the module's
/// [definition fingerprint](crate::module::Module::definition_fingerprint),
/// every grid field bit for bit (size, origin, texel size, edge), every
/// parameter, every material and map input's content digest, every
/// resource's content, sampling and reduction, and the seed. Nested
/// instances are kept and found the same way, so after an edit of one
/// parameter only the instances whose arguments changed run again.
///
/// Resources are resolved by the host on every instantiation, hit or not,
/// so an instance's own resources are always current. A resource a body
/// names itself and hands to a nested instance is not one of its
/// arguments: a kept result records every resource resolved beneath it,
/// and is used only while each still resolves to the same content.
///
/// **Bit-exact.** A hit returns the outputs the body returned and replays
/// the [`Diagnostics`](crate::module::Diagnostics) entries it and its
/// nested instances recorded, so a build with a cache (cold, warm, or
/// after any edits) is bit for bit the build without one. This rests on
/// the [`Module`](crate::module::Module) contract: a body's result is a
/// function of its arguments, grid and path, and one [`ModuleId`] with one
/// definition fingerprint names one behavior. Failed instances are never
/// kept.
///
/// **Budgeted.** Results are charged an estimate of the memory they keep
/// alive: their maps' texels, their diagnostics and their key. Texels
/// shared between results (a material passed through a module unchanged,
/// a nested result inside its parent's) are charged to each, so the charge
/// bounds what the cache keeps alive from above. When a new result does
/// not fit the byte budget, the least recently used results are evicted,
/// in an order that depends only on the sequence of instantiations; a
/// result larger than the whole budget is not kept.
///
/// **Introspectable.** [`InstanceCache::stats`] counts hits, misses,
/// evictions and bytes, overall and per module.
#[derive(Clone, Default)]
pub struct InstanceCache {
    budget: usize,
    slots: BTreeMap<Key, Slot>,
    /// Keys by the clock tick of their last use: least recently used first.
    order: BTreeMap<u64, Key>,
    clock: u64,
    bytes: usize,
    counts: Counts,
    modules: BTreeMap<ModuleId, ModuleCounts>,
}

/// Everything that decides an instance's result, exactly.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Key {
    path: String,
    module: ModuleId,
    words: Vec<u64>,
}

impl Key {
    pub(crate) const fn new(path: String, module: ModuleId, words: Vec<u64>) -> Self {
        Self {
            path,
            module,
            words,
        }
    }
}

/// A resource resolved somewhere beneath an instance: its name and the
/// words that describe what it resolved to.
pub(crate) type Resolution = (ResourceRef, Vec<u64>);

/// A kept result.
#[derive(Clone)]
struct Slot {
    outputs: Outputs,
    entries: Vec<Entry>,
    resources: Vec<Resolution>,
    bytes: usize,
    used: u64,
}

/// What a hit hands back.
pub(crate) struct Hit {
    pub(crate) outputs: Outputs,
    pub(crate) entries: Vec<Entry>,
    pub(crate) resources: Vec<Resolution>,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct ModuleCounts {
    hits: u64,
    misses: u64,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct Counts {
    hits: u64,
    misses: u64,
    stores: u64,
    evictions: u64,
    rejected: u64,
    invalidated: u64,
}

impl InstanceCache {
    /// An empty cache keeping at most about `budget` bytes of results.
    #[must_use]
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            ..Self::default()
        }
    }

    /// The byte budget.
    #[must_use]
    pub const fn budget(&self) -> usize {
        self.budget
    }

    /// Sets the byte budget, evicting the least recently used results
    /// until the rest fit.
    pub fn set_budget(&mut self, budget: usize) {
        self.budget = budget;
        self.evict_until(0);
    }

    /// Results kept.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether no result is kept.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Drops every kept result; the counts go on.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.order.clear();
        self.bytes = 0;
    }

    /// What the cache has done and holds.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let c = self.counts;
        CacheStats {
            hits: c.hits,
            misses: c.misses,
            stores: c.stores,
            evictions: c.evictions,
            rejected: c.rejected,
            invalidated: c.invalidated,
            entries: self.slots.len(),
            bytes: self.bytes,
            budget: self.budget,
            modules: self
                .modules
                .iter()
                .map(|(module, c)| ModuleStats {
                    module: module.clone(),
                    hits: c.hits,
                    misses: c.misses,
                })
                .collect(),
        }
    }

    /// The result kept for `key`, if any and if every resource resolved
    /// beneath it still resolves the same (`current` re-resolves one);
    /// counts a hit or a miss either way.
    pub(crate) fn lookup(
        &mut self,
        key: &Key,
        mut current: impl FnMut(&ResourceRef) -> Option<Vec<u64>>,
    ) -> Option<Hit> {
        let stale = self.slots.get(key).map(|slot| {
            slot.resources
                .iter()
                .any(|(r, words)| current(r).as_ref() != Some(words))
        });
        let module = self.modules.entry(key.module.clone()).or_default();
        match stale {
            None => {
                self.counts.misses += 1;
                module.misses += 1;
                return None;
            }
            Some(true) => {
                self.counts.misses += 1;
                self.counts.invalidated += 1;
                module.misses += 1;
                self.remove(key);
                return None;
            }
            Some(false) => {
                self.counts.hits += 1;
                module.hits += 1;
            }
        }
        let slot = self.slots.get_mut(key)?;
        self.order.remove(&slot.used);
        slot.used = self.clock;
        self.order.insert(self.clock, key.clone());
        self.clock += 1;
        Some(Hit {
            outputs: slot.outputs.clone(),
            entries: slot.entries.clone(),
            resources: slot.resources.clone(),
        })
    }

    /// Keeps `outputs`, the diagnostics `entries` that made them and the
    /// `resources` resolved beneath them, as the result for `key`.
    pub(crate) fn store(
        &mut self,
        key: Key,
        outputs: Outputs,
        entries: Vec<Entry>,
        resources: Vec<Resolution>,
    ) {
        self.remove(&key);
        let bytes = charge(&key, &outputs, &entries, &resources);
        if bytes > self.budget {
            self.counts.rejected += 1;
            return;
        }
        self.evict_until(bytes);
        self.counts.stores += 1;
        self.bytes += bytes;
        self.order.insert(self.clock, key.clone());
        self.slots.insert(
            key,
            Slot {
                outputs,
                entries,
                resources,
                bytes,
                used: self.clock,
            },
        );
        self.clock += 1;
    }

    /// Evicts the least recently used results until `incoming` more bytes
    /// fit the budget.
    fn evict_until(&mut self, incoming: usize) {
        while self.bytes.saturating_add(incoming) > self.budget {
            let Some((_, key)) = self.order.pop_first() else {
                break;
            };
            if let Some(slot) = self.slots.remove(&key) {
                self.bytes -= slot.bytes;
                self.counts.evictions += 1;
            }
        }
    }

    fn remove(&mut self, key: &Key) {
        if let Some(slot) = self.slots.remove(key) {
            self.order.remove(&slot.used);
            self.bytes -= slot.bytes;
        }
    }
}

impl fmt::Debug for InstanceCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InstanceCache")
            .field("entries", &self.slots.len())
            .field("bytes", &self.bytes)
            .field("budget", &self.budget)
            .field("counts", &self.counts)
            .finish_non_exhaustive()
    }
}

/// What an [`InstanceCache`] has done and holds, from
/// [`InstanceCache::stats`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Instantiations answered from the cache.
    pub hits: u64,
    /// Instantiations whose body ran: nothing was kept for their key, or
    /// what was kept had gone stale.
    pub misses: u64,
    /// Results kept.
    pub stores: u64,
    /// Results evicted to make room.
    pub evictions: u64,
    /// Results not kept because they alone exceed the budget.
    pub rejected: u64,
    /// Kept results dropped because a resource resolved beneath them now
    /// resolves differently; each is also a miss.
    pub invalidated: u64,
    /// Results kept now.
    pub entries: usize,
    /// The bytes charged for them: see [`InstanceCache`].
    pub bytes: usize,
    /// The byte budget.
    pub budget: usize,
    /// Hits and misses per module, in [`ModuleId`] order.
    pub modules: Vec<ModuleStats>,
}

/// One module's share of [`CacheStats`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleStats {
    /// The module.
    pub module: ModuleId,
    /// Its instantiations answered from the cache.
    pub hits: u64,
    /// Its instantiations whose body ran.
    pub misses: u64,
}

impl fmt::Display for CacheStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{} hits, {} misses; {} kept ({} bytes of {}); {} stored, {} evicted, {} too large, {} stale",
            self.hits,
            self.misses,
            self.entries,
            self.bytes,
            self.budget,
            self.stores,
            self.evictions,
            self.rejected,
            self.invalidated,
        )?;
        for m in &self.modules {
            writeln!(f, "{}: {} hits, {} misses", m.module, m.hits, m.misses)?;
        }
        Ok(())
    }
}

/// The bytes a result is charged: its key, its outputs' texels and
/// channel tables, its diagnostics and its resource records, plus the
/// bookkeeping around them.
fn charge(key: &Key, outputs: &Outputs, entries: &[Entry], resources: &[Resolution]) -> usize {
    let key_bytes = size_of::<Key>()
        + key.path.len()
        + key.module.name.len()
        + key.words.len() * size_of::<u64>();
    let outputs_bytes: usize = outputs
        .iter()
        .map(|(name, output)| {
            name.len()
                + size_of::<Output>()
                + match output {
                    Output::Material(m) => material_bytes(m),
                    Output::Map(r) => texel_bytes(r),
                }
        })
        .sum();
    let entry_bytes: usize = entries
        .iter()
        .map(|e| {
            size_of::<Entry>()
                + e.path.len()
                + e.module.name.len()
                + match &e.event {
                    Event::Instantiated { .. } => 0,
                    Event::Operation(r) => r.approximations.len() * size_of::<Approximation>(),
                }
        })
        .sum();
    let resource_bytes: usize = resources
        .iter()
        .map(|(r, words)| size_of::<Resolution>() + r.0.len() + words.len() * size_of::<u64>())
        .sum();
    // The slot and its place in the recency order.
    let bookkeeping = size_of::<Slot>() + size_of::<Key>() + 2 * size_of::<u64>();
    key_bytes + outputs_bytes + entry_bytes + resource_bytes + bookkeeping
}

fn material_bytes(m: &Material) -> usize {
    let table = (openpbr::Param::COUNT + Aux::ALL.len()) * size_of::<Option<Channel>>();
    let maps: usize = m
        .bound()
        .map(|(_, ch)| match ch {
            Channel::Map(r) => texel_bytes(r),
            Channel::Constant(_) => 0,
        })
        .sum();
    size_of::<Material>() + table + maps
}

fn texel_bytes(r: &TypedRaster) -> usize {
    let per_texel = match r.storage().kind() {
        StorageKind::F32 => size_of::<f32>(),
        StorageKind::U32 => size_of::<u32>(),
        StorageKind::F32x2 => size_of::<[f32; 2]>(),
        StorageKind::F32x3 => size_of::<[f32; 3]>(),
    };
    r.len() * per_texel
}
