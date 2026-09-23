// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Compositing element sets into rasters, with explicit ownership.
//!
//! Each texel center `p` asks every element near it for its **coverage**:
//! the fraction of a texel-wide box around `p` inside the element's
//! rectangle, estimated from the signed distance to its boundary
//! (`clamp(0.5 − d / texel, 0, 1)`). The background (mortar, grout) covers
//! what the elements leave, `1 − min(Σ cᵢ, 1)`.
//!
//! Two ownership modes are kept apart:
//!
//! - **Coverage compositing** (continuous outputs): every contributor's
//!   program output is weighted by its coverage, normalized when the
//!   coverages sum above 1, over the background. The rule is a sum, so it
//!   is independent of order; it is evaluated in key order for bit-exact
//!   results.
//! - **Partition / winner** (identifier outputs and owner labels): the
//!   contributor with the largest coverage owns the texel, ties to the
//!   smallest key; the background owns it only when its share is strictly
//!   larger.
//!
//! The owner label raster is a **summary**: it names the dominant element
//! only. [`Realized::contributors`] recomputes every contributor of a texel
//! on request, so "the owner" is never mistaken for "the only contributor".
//! Labels are dense indices into [`Realized::keys`], canonical for the
//! current set; identity is the key, not the label.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use dapple_field::hash::hash;
use dapple_field::{Domain, Edge, Footprint, PortType, Value};
use dapple_raster::typed::{Storage, TypedError, TypedRaster};
use dapple_raster::{Raster, TexelRect};
use glam::Vec2;

use crate::identity::ElementKey;
use crate::program::{
    ContractError, Prepared, ProgramInstance, SampleContext, Shape, weighted, zero_like,
};
use crate::set::{Bounds, Correspondence, ElementSet, correspondence, value_fits, value_words};

/// What to composite, and at which resolution.
#[derive(Clone, Copy, Debug)]
pub struct Composite<'a> {
    /// The elements.
    pub set: &'a ElementSet,
    /// The surface program every element invokes.
    pub instance: &'a ProgramInstance,
    /// The background's value for each program output, in output order.
    pub background: &'a [Value],
    /// The periodic domain the elements tile.
    pub domain: Domain,
    /// Texels per row.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// Tile edge, in texels, for incremental updates.
    pub tile_size: u32,
}

/// A compositing failure.
#[derive(Clone, Debug, PartialEq)]
pub enum CompositeError {
    /// The instance does not fit the element set.
    Contract(ContractError),
    /// The domain is not periodic, or the resolution or tile size is zero.
    InvalidGrid,
    /// The background has the wrong number or types of values.
    Background,
    /// Building a typed raster failed.
    Raster(TypedError),
}

impl fmt::Display for CompositeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(e) => e.fmt(f),
            Self::InvalidGrid => {
                f.write_str("composites need a periodic domain and a nonempty grid")
            }
            Self::Background => {
                f.write_str("the background must give one fitting value per output")
            }
            Self::Raster(e) => e.fmt(f),
        }
    }
}

impl core::error::Error for CompositeError {}

impl From<ContractError> for CompositeError {
    fn from(e: ContractError) -> Self {
        Self::Contract(e)
    }
}

/// One element's contribution to one texel.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Contribution {
    /// The element.
    pub key: ElementKey,
    /// Its coverage of the texel, in `(0, 1]`.
    pub coverage: f32,
}

/// Every contributor to one texel, recomputed on request.
#[derive(Clone, Debug, PartialEq)]
pub struct Contributors {
    /// Elements with nonzero coverage, in key order.
    pub elements: Vec<Contribution>,
    /// The background's share.
    pub background: f32,
}

/// What an incremental update did.
#[derive(Clone, Debug, PartialEq)]
pub struct UpdateReport {
    /// Whether everything was recomputed (a new program, background or
    /// grid).
    pub whole: bool,
    /// Tiles recomputed, ascending.
    pub tiles: Vec<u32>,
    /// Texels recomputed.
    pub texels: u64,
    /// Whether owner labels outside the recomputed tiles were renumbered,
    /// because elements were added or removed.
    pub relabeled: bool,
    /// How the new set's elements relate to the previous one's.
    pub correspondence: Correspondence,
}

/// Composited outputs and owner labels: see the [module docs](self).
#[derive(Clone, Debug, PartialEq)]
pub struct Realized {
    width: u32,
    height: u32,
    tile_size: u32,
    domain: Domain,
    period: Vec2,
    set: ElementSet,
    instance: u64,
    background: Vec<Value>,
    ports: Vec<PortType>,
    names: Vec<alloc::string::String>,
    outputs: Vec<Vec<Value>>,
    owners: Vec<u32>,
}

/// Per-composite working state: the prepared instance, hoisted element
/// values and a spatial index.
struct Context<'a> {
    prepared: Prepared<'a>,
    set: &'a ElementSet,
    elements: Vec<Vec<Option<Value>>>,
    buckets: Vec<Vec<(u32, Vec2)>>,
    counts: [u32; 2],
    bucket: Vec2,
    texel: Vec2,
    period: Vec2,
    footprint: Footprint,
    background: &'a [Value],
    ports: Vec<PortType>,
}

fn signed_distance(q: Vec2, half: Vec2) -> f32 {
    let d = q.abs() - half;
    d.max(Vec2::ZERO).length() + d.x.max(d.y).min(0.0)
}

impl<'a> Context<'a> {
    fn new(c: &Composite<'a>) -> Result<Self, CompositeError> {
        let Domain::Periodic { period } = c.domain else {
            return Err(CompositeError::InvalidGrid);
        };
        if c.width == 0 || c.height == 0 || c.tile_size == 0 {
            return Err(CompositeError::InvalidGrid);
        }
        let program = c.instance.program();
        if c.background.len() != program.outputs().len()
            || program
                .outputs()
                .iter()
                .zip(c.background)
                .any(|(o, &v)| !value_fits(v, o.port))
        {
            return Err(CompositeError::Background);
        }
        let prepared = c.instance.prepare(c.set)?;
        let elements = (0..c.set.len()).map(|i| prepared.element(i)).collect();
        #[expect(
            clippy::cast_precision_loss,
            reason = "periods and grid sizes are far below f32's exact integer range"
        )]
        let (period, texel) = {
            let period = Vec2::new(period[0] as f32, period[1] as f32);
            (period, period / Vec2::new(c.width as f32, c.height as f32))
        };
        let side = (1..=64_u32)
            .take_while(|s| (s * s) as usize <= c.set.len())
            .last()
            .unwrap_or(1);
        let counts = [side, side];
        #[expect(clippy::cast_precision_loss, reason = "at most 64 buckets")]
        let bucket = period / side as f32;
        let mut buckets = vec![Vec::new(); (side * side) as usize];
        let pad = texel.max_element();
        for i in 0..c.set.len() {
            let b = c.set.bounds(i).grown(pad);
            let lo = (b.min / bucket).floor();
            let hi = (b.max / bucket).floor();
            #[expect(
                clippy::cast_possible_truncation,
                reason = "bucket coordinates of finite bounds are small"
            )]
            let (lo, hi) = ([lo.x as i64, lo.y as i64], [hi.x as i64, hi.y as i64]);
            for by in lo[1]..=hi[1] {
                for bx in lo[0]..=hi[0] {
                    let (wx, kx) = (
                        bx.rem_euclid(i64::from(side)),
                        bx.div_euclid(i64::from(side)),
                    );
                    let (wy, ky) = (
                        by.rem_euclid(i64::from(side)),
                        by.div_euclid(i64::from(side)),
                    );
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "wrap counts are small integers"
                    )]
                    let shift = Vec2::new(kx as f32, ky as f32) * period;
                    let index = usize::try_from(wy * i64::from(side) + wx).expect("wrapped");
                    buckets[index]
                        .push((u32::try_from(i).expect("fewer than 2^32 elements"), shift));
                }
            }
        }
        Ok(Self {
            prepared,
            set: c.set,
            elements,
            buckets,
            counts,
            bucket,
            texel,
            period,
            footprint: Footprint::new(texel.max_element()).ok_or(CompositeError::InvalidGrid)?,
            background: c.background,
            ports: program.outputs().iter().map(|o| o.port).collect(),
        })
    }

    fn center(&self, x: u32, y: u32) -> Vec2 {
        #[expect(
            clippy::cast_precision_loss,
            reason = "grid sizes are far below f32's exact integer range"
        )]
        let t = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
        t * self.texel
    }

    /// Contributors at texel `(x, y)`: element index, coverage, local
    /// position and edge distance, in key order.
    fn contributors(&self, x: u32, y: u32, out: &mut Vec<(usize, f32, Vec2, f32)>) {
        out.clear();
        let p = self.center(x, y);
        let b = (p / self.bucket).floor();
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "texel centers lie inside the period"
        )]
        let (bx, by) = (
            (b.x as u32).min(self.counts[0] - 1),
            (b.y as u32).min(self.counts[1] - 1),
        );
        let width = self.texel.max_element();
        for &(i, shift) in &self.buckets[(by * self.counts[0] + bx) as usize] {
            let i = i as usize;
            let q = self.set.placement(i).to_local(p + shift);
            let d = signed_distance(q, self.set.half_size(i));
            let coverage = (0.5 - d / width).clamp(0.0, 1.0);
            if coverage > 0.0 {
                out.push((i, coverage, q, -d));
            }
        }
    }

    /// Composites texel `(x, y)`: every output value, then the owner label.
    fn texel(
        &self,
        x: u32,
        y: u32,
        scratch: &mut Vec<(usize, f32, Vec2, f32)>,
        values: &mut Vec<Value>,
        out: &mut [Value],
    ) -> u32 {
        self.contributors(x, y, scratch);
        let total: f32 = scratch.iter().map(|c| c.1).sum();
        let scale = if total > 1.0 { 1.0 / total } else { 1.0 };
        let share = 1.0 - (total * scale);
        let mut owner: Option<(usize, f32)> = None;
        for &(i, c, _, _) in scratch.iter() {
            if owner.is_none_or(|(_, best)| c > best) {
                owner = Some((i, c));
            }
        }
        let owner = owner.filter(|&(_, c)| c * scale >= share);
        for (o, slot) in out.iter_mut().enumerate() {
            *slot = if Shape::of(self.ports[o]) == Shape::Id {
                self.background[o]
            } else {
                let mut acc = zero_like(self.background[o]);
                weighted(&mut acc, self.background[o], share);
                acc
            };
        }
        for &(i, c, local, edge) in scratch.iter() {
            let context = SampleContext {
                local,
                edge,
                footprint: self.footprint,
            };
            self.prepared.sample(i, &self.elements[i], context, values);
            for (o, slot) in out.iter_mut().enumerate() {
                if Shape::of(self.ports[o]) == Shape::Id {
                    if owner.is_some_and(|(w, _)| w == i) {
                        *slot = values[o];
                    }
                } else {
                    weighted(slot, values[o], c * scale);
                }
            }
        }
        owner.map_or(0, |(i, _)| {
            u32::try_from(i + 1).expect("fewer than 2^32 elements")
        })
    }

    fn fill(&self, rect: TexelRect, width: u32, outputs: &mut [Vec<Value>], owners: &mut [u32]) {
        let (mut scratch, mut values) = (Vec::new(), Vec::new());
        let mut out = vec![Value::Scalar(0.0); outputs.len()];
        for y in rect.y0..rect.y1 {
            for x in rect.x0..rect.x1 {
                let t = (y * width + x) as usize;
                owners[t] = self.texel(x, y, &mut scratch, &mut values, &mut out);
                for (o, v) in out.iter().enumerate() {
                    outputs[o][t] = *v;
                }
            }
        }
    }
}

fn tiles_per_axis(n: u32, tile: u32) -> u32 {
    n.div_ceil(tile)
}

impl Realized {
    /// Composites `c` from scratch.
    ///
    /// # Errors
    ///
    /// [`CompositeError::InvalidGrid`], [`CompositeError::Background`], or
    /// [`CompositeError::Contract`] when the instance binds attributes the
    /// set does not have.
    pub fn composite(c: &Composite<'_>) -> Result<Self, CompositeError> {
        let context = Context::new(c)?;
        let count = (c.width * c.height) as usize;
        let program = c.instance.program();
        let mut realized = Self {
            width: c.width,
            height: c.height,
            tile_size: c.tile_size,
            domain: c.domain,
            period: context.period,
            set: c.set.clone(),
            instance: c.instance.fingerprint(),
            background: c.background.to_vec(),
            ports: context.ports.clone(),
            names: program.outputs().iter().map(|o| o.name.clone()).collect(),
            outputs: c.background.iter().map(|&v| vec![v; count]).collect(),
            owners: vec![0; count],
        };
        let rect = TexelRect {
            x0: 0,
            y0: 0,
            x1: c.width,
            y1: c.height,
        };
        context.fill(rect, c.width, &mut realized.outputs, &mut realized.owners);
        Ok(realized)
    }

    /// Brings this realization up to date with `c`, recomputing only the
    /// tiles that the changed, added and removed elements' old and new
    /// bounds reach. The result is bit-identical to [`Realized::composite`].
    ///
    /// # Errors
    ///
    /// As [`Realized::composite`].
    pub fn update(&mut self, c: &Composite<'_>) -> Result<UpdateReport, CompositeError> {
        let report = correspondence(&self.set, c.set);
        let same_setup = self.instance == c.instance.fingerprint()
            && self.background == c.background
            && (self.width, self.height, self.tile_size) == (c.width, c.height, c.tile_size)
            && self.set.schema() == c.set.schema()
            && self.domain == c.domain;
        if !same_setup {
            *self = Self::composite(c)?;
            let (tx, ty) = self.tile_counts();
            return Ok(UpdateReport {
                whole: true,
                tiles: (0..tx * ty).collect(),
                texels: u64::from(self.width) * u64::from(self.height),
                relabeled: true,
                correspondence: report,
            });
        }
        let context = Context::new(c)?;
        let mut dirty = vec![
            false;
            {
                let (tx, ty) = self.tile_counts();
                (tx * ty) as usize
            }
        ];
        let pad = context.texel.max_element();
        for key in report.changed.iter().chain(&report.removed) {
            let i = self.set.index_of(*key).expect("in the old set");
            self.mark(self.set.bounds(i).grown(pad), &mut dirty);
        }
        for key in report.changed.iter().chain(&report.added) {
            let i = c.set.index_of(*key).expect("in the new set");
            self.mark(c.set.bounds(i).grown(pad), &mut dirty);
        }
        // Labels are dense indices into the key table; when keys come or
        // go, renumber the labels the recomputed tiles will not rewrite.
        let relabeled = self.set.keys() != c.set.keys();
        if relabeled {
            let map: Vec<u32> = self
                .set
                .keys()
                .iter()
                .map(|k| {
                    c.set
                        .index_of(*k)
                        .map_or(0, |i| u32::try_from(i + 1).expect("small"))
                })
                .collect();
            for label in &mut self.owners {
                if *label != 0 {
                    *label = map[*label as usize - 1];
                }
            }
        }
        let (tx, _) = self.tile_counts();
        let mut tiles = Vec::new();
        let mut texels = 0;
        for (t, &d) in dirty.iter().enumerate() {
            if d {
                let t = u32::try_from(t).expect("tile counts fit u32");
                let rect = self.tile_rect(t % tx, t / tx);
                texels += rect.area();
                context.fill(rect, self.width, &mut self.outputs, &mut self.owners);
                tiles.push(t);
            }
        }
        self.set = c.set.clone();
        Ok(UpdateReport {
            whole: false,
            tiles,
            texels,
            relabeled,
            correspondence: report,
        })
    }

    fn tile_counts(&self) -> (u32, u32) {
        (
            tiles_per_axis(self.width, self.tile_size),
            tiles_per_axis(self.height, self.tile_size),
        )
    }

    fn tile_rect(&self, tx: u32, ty: u32) -> TexelRect {
        let t = self.tile_size;
        TexelRect {
            x0: tx * t,
            y0: ty * t,
            x1: ((tx + 1) * t).min(self.width),
            y1: ((ty + 1) * t).min(self.height),
        }
    }

    /// The tiles whose texel centers `bounds` (in unwrapped domain units)
    /// can reach, wrapped into the period.
    #[must_use]
    pub fn tiles_reached(&self, bounds: Bounds) -> Vec<u32> {
        let mut dirty = vec![
            false;
            {
                let (tx, ty) = self.tile_counts();
                (tx * ty) as usize
            }
        ];
        self.mark(bounds, &mut dirty);
        dirty
            .iter()
            .enumerate()
            .filter(|(_, d)| **d)
            .map(|(t, _)| u32::try_from(t).expect("tile counts fit u32"))
            .collect()
    }

    fn mark(&self, bounds: Bounds, dirty: &mut [bool]) {
        #[expect(
            clippy::cast_precision_loss,
            reason = "grid sizes are far below f32's exact integer range"
        )]
        let texel = self.period / Vec2::new(self.width as f32, self.height as f32);
        let lo = (bounds.min / texel).floor();
        let hi = (bounds.max / texel).ceil();
        #[expect(
            clippy::cast_possible_truncation,
            reason = "texel coordinates of finite bounds are small"
        )]
        let (lo, hi) = ([lo.x as i64, lo.y as i64], [hi.x as i64, hi.y as i64]);
        let (tx, ty) = self.tile_counts();
        let t = i64::from(self.tile_size);
        let span = |lo: i64, hi: i64, n: u32, count: u32| -> Vec<u32> {
            let n = i64::from(n);
            if hi - lo >= n {
                return (0..count).collect();
            }
            let mut tiles: Vec<u32> = (lo..=hi)
                .map(|v| u32::try_from(v.rem_euclid(n) / t).expect("wrapped"))
                .collect();
            tiles.sort_unstable();
            tiles.dedup();
            tiles
        };
        let xs = span(lo[0], hi[0], self.width, tx);
        let ys = span(lo[1], hi[1], self.height, ty);
        for &y in &ys {
            for &x in &xs {
                dirty[(y * tx + x) as usize] = true;
            }
        }
    }

    /// The element set this realization shows.
    #[must_use]
    pub fn set(&self) -> &ElementSet {
        &self.set
    }

    /// The key table owner labels index: label `n > 0` is `keys()[n − 1]`;
    /// label 0 is the background.
    #[must_use]
    pub fn keys(&self) -> &[ElementKey] {
        self.set.keys()
    }

    /// The owner-label raster: a winner summary, see the [module
    /// docs](self).
    ///
    /// # Errors
    ///
    /// Never for a valid realization.
    pub fn owner_labels(&self) -> Result<TypedRaster, CompositeError> {
        self.typed(
            PortType::Id,
            Storage::U32(self.raster(self.owners.clone())?),
        )
    }

    /// The key owning texel `(x, y)`, if an element does.
    #[must_use]
    pub fn owner(&self, x: u32, y: u32) -> Option<ElementKey> {
        let label = self.owners[(y * self.width + x) as usize];
        (label > 0).then(|| self.set.keys()[label as usize - 1])
    }

    /// Output `name` as a typed raster.
    ///
    /// # Errors
    ///
    /// Never for a valid realization.
    pub fn output(&self, name: &str) -> Result<Option<TypedRaster>, CompositeError> {
        let Some(o) = self.names.iter().position(|n| n == name) else {
            return Ok(None);
        };
        let values = &self.outputs[o];
        let port = self.ports[o];
        let storage = match Shape::of(port) {
            Shape::Scalar => Storage::F32(
                self.raster(
                    values
                        .iter()
                        .map(|v| v.component(0).unwrap_or(0.0))
                        .collect(),
                )?,
            ),
            Shape::Id => Storage::U32(
                self.raster(
                    values
                        .iter()
                        .map(|v| match v {
                            Value::Id(id) => *id,
                            _ => 0,
                        })
                        .collect(),
                )?,
            ),
            Shape::Vector2 => Storage::F32x2(
                self.raster(
                    values
                        .iter()
                        .map(|v| match v {
                            Value::Vector2(v) => v.to_array(),
                            _ => [0.0; 2],
                        })
                        .collect(),
                )?,
            ),
            Shape::Vector3 => Storage::F32x3(
                self.raster(
                    values
                        .iter()
                        .map(|v| match v {
                            Value::Vector3(v) => v.to_array(),
                            _ => [0.0; 3],
                        })
                        .collect(),
                )?,
            ),
        };
        self.typed(port, storage).map(Some)
    }

    /// The value of output `output` (by position) at texel `(x, y)`.
    #[must_use]
    pub fn value(&self, output: usize, x: u32, y: u32) -> Value {
        self.outputs[output][(y * self.width + x) as usize]
    }

    fn raster<T: Copy>(&self, values: Vec<T>) -> Result<Raster<T>, CompositeError> {
        #[expect(
            clippy::cast_precision_loss,
            reason = "grid sizes are far below f32's exact integer range"
        )]
        let texel = self.period / Vec2::new(self.width as f32, self.height as f32);
        Raster::from_values(
            self.width,
            self.height,
            Vec2::ZERO,
            texel,
            Edge::Wrap,
            values,
        )
        .map_err(|e| CompositeError::Raster(TypedError::Raster(e)))
    }

    fn typed(&self, port: PortType, storage: Storage) -> Result<TypedRaster, CompositeError> {
        TypedRaster::new(port, storage).map_err(CompositeError::Raster)
    }

    /// Every contributor to texel `(x, y)`, recomputed from the elements.
    ///
    /// # Errors
    ///
    /// As [`Realized::composite`] for `c`, which must describe this
    /// realization.
    pub fn contributors(c: &Composite<'_>, x: u32, y: u32) -> Result<Contributors, CompositeError> {
        let context = Context::new(c)?;
        let mut scratch = Vec::new();
        context.contributors(x, y, &mut scratch);
        let total: f32 = scratch.iter().map(|s| s.1).sum();
        Ok(Contributors {
            elements: scratch
                .iter()
                .map(|&(i, coverage, _, _)| Contribution {
                    key: c.set.keys()[i],
                    coverage,
                })
                .collect(),
            background: 1.0 - total.min(1.0),
        })
    }

    /// A digest of every output's and label's exact bits, with the key table.
    #[must_use]
    pub fn digest(&self) -> u64 {
        let mut words = vec![u64::from(self.width), u64::from(self.height)];
        words.extend(self.set.keys().iter().map(|k| k.word()));
        for output in &self.outputs {
            for &v in output {
                value_words(v, &mut words);
            }
        }
        words.extend(self.owners.iter().map(|&l| u64::from(l)));
        hash(0x636f_6d70_6f73_6974, &words) // the first eight bytes of "composite"
    }
}
