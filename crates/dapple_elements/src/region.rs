// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Region maps and region tables.
//!
//! A [`RegionMap`] is a label raster, one label per texel (0 for none), and
//! a table of the regions the labels name, in canonical key order: each
//! [`Region`]'s identity, provenance, area, centroid, bounds, orientation and
//! neighbors. A region's label is its dense index in this map plus one;
//! identity is its [`RegionKey`], never the label.
//!
//! Regions come from two places, kept apart by their [`Provenance`]:
//!
//! - **Composited** ([`RegionMap::from_composite`]): a composite's owner
//!   labels, each region the texels one element owns. The region keeps the
//!   element's identity: its key is the element's key.
//! - **Reconstructed** ([`RegionMap::reconstruct`]): the connected
//!   components of a mask (a flood fill), for masks that are not made of
//!   elements, such as a thresholded noise, a cracked glaze or a mask a
//!   morphology edited. Reconstruction is **canonical**: the mask alone
//!   determines the labels and keys, a component's key being a hash of its
//!   anchor, the first texel in row-major order.
//!
//! **Correspondence** is a separate report ([`RegionMap::correspondence`]):
//! regions of two maps on one grid are linked by the texels they share,
//! and every group of linked regions is named for what it is: a match, a
//! split, a merge, a regroup (several into several), an appearance or a
//! disappearance. It never changes either map.
//!
//! **Retained identity** is explicit: [`RegionMap::retaining`] rekeys a
//! canonical reconstruction from a previous map, which is then persisted
//! identity state, so a clean rebuild from the same mask and the same
//! previous map gives the same keys. Each group keeps the previous keys of
//! its largest overlaps: in a split, one child (the largest overlap) keeps
//! the parent's key and the others are new; in a merge, the merged region
//! keeps the key of its largest parent.
//!
//! On a wrapping raster, regions continue across the period's edges:
//! components connect through them, and centroids, bounds and orientations
//! are measured on the region unwrapped from its anchor.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use dapple_field::hash::hash;
use dapple_field::{Edge, PortType};
use dapple_raster::typed::{Storage, TypedRaster};
use dapple_raster::{DistanceTransform, Raster, RasterError, RasterOp};
use glam::Vec2;

use crate::composite::{CompositeError, Realized};
use crate::identity::ElementKey;
use crate::program::{Binding, ProgramInstance};
use crate::set::Bounds;
use dapple_field::scoped::{ContractError, Scope, value_fits};
use dapple_field::{Footprint, Value};
use dapple_raster::typed::TypedError;

/// Purpose tag of reconstructed region keys ("regionky").
const REGION_TAG: u64 = 0x7265_6769_6f6e_6b79;

/// A region's identity.
///
/// A composited region's key is its element's key word; a reconstructed
/// region's is a hash of its anchor, unless [`RegionMap::retaining`] kept a
/// previous key.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct RegionKey(u64);

impl RegionKey {
    /// A key from its raw word, for keys persisted elsewhere.
    #[must_use]
    pub const fn from_word(word: u64) -> Self {
        Self(word)
    }

    /// The key's hash word.
    #[must_use]
    pub const fn word(self) -> u64 {
        self.0
    }

    /// The key of the region element `key` owns.
    #[must_use]
    pub const fn of_element(key: ElementKey) -> Self {
        Self(key.word())
    }

    /// The canonical key of a component anchored at texel `anchor`.
    #[must_use]
    pub fn of_anchor(anchor: [u32; 2]) -> Self {
        Self(hash(
            REGION_TAG,
            &[u64::from(anchor[0]), u64::from(anchor[1])],
        ))
    }
}

impl fmt::Debug for RegionKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RegionKey({:016x})", self.0)
    }
}

/// Where a region came from.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Provenance {
    /// The texels a composited element owns.
    Element(ElementKey),
    /// A connected component of a mask, anchored at its first texel in
    /// row-major order.
    Component {
        /// The anchor texel.
        anchor: [u32; 2],
    },
}

/// Which texels touch.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Connectivity {
    /// Texels sharing an edge.
    Four,
    /// Texels sharing an edge or a corner.
    Eight,
}

/// One region of a [`RegionMap`].
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    /// Identity.
    pub key: RegionKey,
    /// Where it came from.
    pub provenance: Provenance,
    /// Texels labeled with it.
    pub texels: u32,
    /// Area in square domain units.
    pub area: f32,
    /// The mean of its texel centers, in domain units, inside the period
    /// on a wrapping map.
    pub centroid: Vec2,
    /// Its bounds in domain units, unwrapped from its anchor (so they may
    /// extend past the period).
    pub bounds: Bounds,
    /// The angle of its principal axis, in radians in `[0, π)`, from the
    /// domain's x axis: the direction its texels spread most.
    pub orientation: f32,
    /// Regions sharing an edge with it, by key, ascending.
    pub neighbors: Vec<RegionKey>,
}

/// A region-map failure.
#[derive(Clone, Debug, PartialEq)]
pub enum RegionError {
    /// Two maps are not on the same grid.
    GridMismatch,
    /// A threshold is not finite.
    InvalidThreshold,
    /// A retained key collides with a canonical one.
    KeyCollision(RegionKey),
    /// A value raster is not on the map's grid.
    Raster(RasterError),
    /// Reading a composite failed.
    Composite(CompositeError),
    /// A program instance does not fit region evaluation.
    Contract(ContractError),
    /// Building an output raster failed.
    Typed(TypedError),
}

impl fmt::Display for RegionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GridMismatch => f.write_str("the maps are not on one grid"),
            Self::InvalidThreshold => f.write_str("the threshold is not finite"),
            Self::KeyCollision(key) => write!(f, "two regions would share {key:?}"),
            Self::Raster(e) => e.fmt(f),
            Self::Composite(e) => e.fmt(f),
            Self::Contract(e) => e.fmt(f),
            Self::Typed(e) => e.fmt(f),
        }
    }
}

impl core::error::Error for RegionError {}

/// Per-region statistics of a scalar raster ([`RegionMap::statistics`]).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RegionStatistics {
    /// The region.
    pub key: RegionKey,
    /// The smallest value on it.
    pub min: f32,
    /// The largest value on it.
    pub max: f32,
    /// The mean of its values.
    pub mean: f32,
}

/// A connected piece of one class, unwrapped from its anchor.
#[derive(Clone, Debug)]
struct Piece {
    class: u32,
    anchor: usize,
    texels: u32,
    sum: [f64; 2],
    sum2: [f64; 3],
    lo: [i64; 2],
    hi: [i64; 2],
}

/// A label raster and its region table: see the [module docs](self).
#[derive(Clone, Debug, PartialEq)]
pub struct RegionMap {
    labels: Raster<u32>,
    regions: Vec<Region>,
    /// Per region, its pieces' texel bounds, unwrapped: `[x0, y0, x1, y1]`
    /// inclusive.
    pieces: Vec<Vec<[i64; 4]>>,
}

impl RegionMap {
    /// The regions of a composite's owner labels: one per element owning a
    /// texel, keyed by the element.
    ///
    /// # Errors
    ///
    /// [`RegionError::Composite`] for an invalid realization.
    pub fn from_composite(realized: &Realized) -> Result<Self, RegionError> {
        let owners = realized.owner_labels().map_err(RegionError::Composite)?;
        let Storage::U32(owners) = owners.storage() else {
            unreachable!("owner labels are identifiers")
        };
        let keys = realized.keys();
        Ok(Self::build(owners, Connectivity::Four, |class, _| {
            let key = keys[class as usize - 1];
            (RegionKey::of_element(key), Provenance::Element(key))
        }))
    }

    /// The connected components of `mask`'s texels at or above `threshold`,
    /// canonically keyed by their anchors.
    ///
    /// # Errors
    ///
    /// [`RegionError::InvalidThreshold`] for a threshold that is not finite.
    pub fn reconstruct(
        mask: &Raster,
        threshold: f32,
        connectivity: Connectivity,
    ) -> Result<Self, RegionError> {
        if !threshold.is_finite() {
            return Err(RegionError::InvalidThreshold);
        }
        let classes = Raster::from_values(
            mask.width(),
            mask.height(),
            mask.origin(),
            mask.texel(),
            mask.edge(),
            mask.values()
                .iter()
                .map(|&v| u32::from(v >= threshold))
                .collect(),
        )
        .map_err(RegionError::Raster)?;
        Ok(Self::build(&classes, connectivity, |_, anchor| {
            (
                RegionKey::of_anchor(anchor),
                Provenance::Component { anchor },
            )
        }))
    }

    /// Builds a map from nonzero `classes`: every connected piece of a class
    /// is its own region when `name` gives each piece its own key, and
    /// pieces given one key form one region.
    fn build(
        classes: &Raster<u32>,
        connectivity: Connectivity,
        name: impl Fn(u32, [u32; 2]) -> (RegionKey, Provenance),
    ) -> Self {
        let (w, h) = (classes.width(), classes.height());
        let wrap = classes.edge() == Edge::Wrap;
        let pieces = pieces(w, h, wrap, classes.values(), connectivity);
        // Pieces by key, in canonical key order.
        let mut by_key: BTreeMap<RegionKey, (Provenance, Vec<usize>)> = BTreeMap::new();
        let mut piece_key = Vec::with_capacity(pieces.1.len());
        for (i, piece) in pieces.1.iter().enumerate() {
            let anchor = xy(piece.anchor, w);
            let (key, provenance) = name(piece.class, anchor);
            by_key
                .entry(key)
                .or_insert_with(|| (provenance, Vec::new()))
                .1
                .push(i);
            piece_key.push(key);
        }
        let index: BTreeMap<RegionKey, u32> = by_key
            .keys()
            .enumerate()
            .map(|(i, &k)| (k, u32::try_from(i).expect("fewer than 2^32 regions")))
            .collect();
        let labels: Vec<u32> = pieces
            .0
            .iter()
            .map(|&p| {
                if p == u32::MAX {
                    0
                } else {
                    index[&piece_key[p as usize]] + 1
                }
            })
            .collect();
        let labels = Raster::from_values(
            w,
            h,
            classes.origin(),
            classes.texel(),
            classes.edge(),
            labels,
        )
        .expect("the classes' grid");
        let texel = classes.texel();
        let origin = classes.origin();
        #[expect(
            clippy::cast_precision_loss,
            reason = "grid sizes are far below f32's exact integer range"
        )]
        let period = Vec2::new(w as f32, h as f32) * texel;
        let mut regions = Vec::with_capacity(by_key.len());
        let mut region_pieces = Vec::with_capacity(by_key.len());
        for (&key, (provenance, members)) in &by_key {
            let mut n = 0_u32;
            let (mut sum, mut sum2) = ([0.0_f64; 2], [0.0_f64; 3]);
            let mut lo = [i64::MAX; 2];
            let mut hi = [i64::MIN; 2];
            let mut bounds = Vec::with_capacity(members.len());
            for &m in members {
                let p = &pieces.1[m];
                n += p.texels;
                for k in 0..2 {
                    sum[k] += p.sum[k];
                    lo[k] = lo[k].min(p.lo[k]);
                    hi[k] = hi[k].max(p.hi[k]);
                }
                for (total, part) in sum2.iter_mut().zip(p.sum2) {
                    *total += part;
                }
                bounds.push([p.lo[0], p.lo[1], p.hi[0], p.hi[1]]);
            }
            let count = f64::from(n);
            let mean = [sum[0] / count, sum[1] / count];
            let cxx = sum2[0] / count - mean[0] * mean[0];
            let cxy = sum2[1] / count - mean[0] * mean[1];
            let cyy = sum2[2] / count - mean[1] * mean[1];
            // Texel units to domain units, per axis.
            let (tx, ty) = (f64::from(texel.x), f64::from(texel.y));
            let mut angle = 0.5 * libm::atan2(2.0 * cxy * tx * ty, cxx * tx * tx - cyy * ty * ty);
            if angle < 0.0 {
                angle += core::f64::consts::PI;
            }
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_precision_loss,
                reason = "domain coordinates are f32; the sums are narrowed once"
            )]
            let (centroid, min, max, orientation) = (
                Vec2::new(((mean[0] + 0.5) * tx) as f32, ((mean[1] + 0.5) * ty) as f32),
                Vec2::new(lo[0] as f32, lo[1] as f32) * texel,
                Vec2::new((hi[0] + 1) as f32, (hi[1] + 1) as f32) * texel,
                (angle as f32).min(core::f32::consts::PI.next_down()),
            );
            let centroid = if wrap {
                centroid.rem_euclid(period)
            } else {
                centroid
            };
            #[expect(clippy::cast_precision_loss, reason = "texel counts")]
            let area = n as f32 * texel.x * texel.y;
            regions.push(Region {
                key,
                provenance: *provenance,
                texels: n,
                area,
                centroid: origin + centroid,
                bounds: Bounds {
                    min: origin + min,
                    max: origin + max,
                },
                orientation,
                neighbors: Vec::new(),
            });
            region_pieces.push(bounds);
        }
        let mut map = Self {
            labels,
            regions,
            pieces: region_pieces,
        };
        map.find_neighbors();
        map
    }

    fn find_neighbors(&mut self) {
        let (w, h) = (self.labels.width(), self.labels.height());
        let wrap = self.labels.edge() == Edge::Wrap;
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        let v = self.labels.values();
        for y in 0..h {
            for x in 0..w {
                let a = v[(y * w + x) as usize];
                if a == 0 {
                    continue;
                }
                let right = if x + 1 < w {
                    Some(x + 1)
                } else {
                    wrap.then_some(0)
                };
                let below = if y + 1 < h {
                    Some(y + 1)
                } else {
                    wrap.then_some(0)
                };
                let mut touch = |b: u32| {
                    if b != 0 && b != a {
                        pairs.push((a, b));
                        pairs.push((b, a));
                    }
                };
                if let Some(r) = right {
                    touch(v[(y * w + r) as usize]);
                }
                if let Some(b) = below {
                    touch(v[(b * w + x) as usize]);
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        for region in &mut self.regions {
            region.neighbors.clear();
        }
        for (a, b) in pairs {
            let key = self.regions[b as usize - 1].key;
            self.regions[a as usize - 1].neighbors.push(key);
        }
        for region in &mut self.regions {
            region.neighbors.sort_unstable();
        }
    }

    /// The label raster: label `n > 0` is `regions()[n − 1]`, 0 is no
    /// region.
    #[must_use]
    pub const fn labels(&self) -> &Raster<u32> {
        &self.labels
    }

    /// The labels as an identifier raster.
    ///
    /// # Panics
    ///
    /// Never: the storage fits the type.
    #[must_use]
    pub fn label_raster(&self) -> TypedRaster {
        TypedRaster::new(PortType::Id, Storage::U32(self.labels.clone()))
            .expect("identifiers are stored as u32")
    }

    /// Evaluates `instance` over the map: its region-scope nodes once per
    /// region, the rest once per texel, and `background` (one value per
    /// output) where no region is.
    ///
    /// Inputs may be bound to constants, region properties
    /// ([`Binding::RegionRandom`], [`Binding::RegionArea`],
    /// [`Binding::RegionCentroid`], [`Binding::RegionOrientation`]) and the
    /// texel center's domain position ([`Binding::Position`]); element
    /// bindings are refused. Returns one raster per output, in output order.
    ///
    /// # Errors
    ///
    /// [`RegionError::Contract`] for an element binding or a background
    /// that does not fit, [`RegionError::Typed`] when building an output
    /// fails.
    pub fn evaluate(
        &self,
        instance: &ProgramInstance,
        background: &[Value],
    ) -> Result<Vec<TypedRaster>, RegionError> {
        let program = instance.program();
        let outputs = program.outputs();
        if background.len() != outputs.len()
            || outputs
                .iter()
                .zip(background)
                .any(|(o, v)| !value_fits(*v, o.port))
        {
            return Err(RegionError::Contract(ContractError::BindingCount));
        }
        for (input, binding) in program.inputs().iter().zip(instance.bindings()) {
            if matches!(
                binding,
                Binding::Attribute(_)
                    | Binding::ElementRandom(_)
                    | Binding::HalfSize
                    | Binding::Variant
                    | Binding::LocalPosition
                    | Binding::EdgeDistance
            ) {
                return Err(RegionError::Contract(ContractError::UnsupportedBinding(
                    input.name.clone(),
                )));
            }
        }
        let input = |i: u32, region: Option<&Region>, position: Vec2| -> Value {
            let r = || region.expect("region scope is evaluated per region");
            match &instance.bindings()[i as usize] {
                Binding::Constant(v) => *v,
                Binding::RegionRandom(stream) => {
                    Value::Scalar(crate::identity::unit_of_word(r().key.word(), *stream))
                }
                Binding::RegionArea => Value::Scalar(r().area),
                Binding::RegionCentroid => Value::Vector2(r().centroid),
                Binding::RegionOrientation => Value::Scalar(r().orientation),
                Binding::Position => Value::Vector2(position),
                _ => unreachable!("refused above"),
            }
        };
        let hoisted: Vec<Vec<Option<Value>>> = self
            .regions
            .iter()
            .map(|region| {
                let mut values = vec![None; program.nodes().len()];
                program.evaluate(
                    &mut values,
                    |s| s.within(Scope::Region),
                    &mut |i| input(i, Some(region), Vec2::ZERO),
                    Footprint::POINT,
                );
                values
            })
            .collect();
        let labels = &self.labels;
        let (origin, texel) = (labels.origin(), labels.texel());
        let footprint = Footprint::new(texel.max_element()).unwrap_or(Footprint::POINT);
        let mut columns: Vec<Vec<Value>> =
            vec![Vec::with_capacity(labels.values().len()); outputs.len()];
        for y in 0..labels.height() {
            for x in 0..labels.width() {
                let label = labels.values()[(y * labels.width() + x) as usize];
                if label == 0 {
                    for (column, v) in columns.iter_mut().zip(background) {
                        column.push(*v);
                    }
                    continue;
                }
                let index = label as usize - 1;
                #[expect(clippy::cast_precision_loss, reason = "texel indices are small")]
                let position = origin + texel * Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let mut values = hoisted[index].clone();
                program.evaluate(
                    &mut values,
                    |_| true,
                    &mut |i| input(i, Some(&self.regions[index]), position),
                    footprint,
                );
                for (column, o) in columns.iter_mut().zip(outputs) {
                    column.push(values[o.node.index()].expect("every node evaluated"));
                }
            }
        }
        outputs
            .iter()
            .zip(columns)
            .map(|(o, column)| {
                TypedRaster::from_values(o.port, labels, column).map_err(RegionError::Typed)
            })
            .collect()
    }

    /// The region table, in key order.
    #[must_use]
    pub fn regions(&self) -> &[Region] {
        &self.regions
    }

    /// The region `key`, if the map has it.
    #[must_use]
    pub fn region(&self, key: RegionKey) -> Option<&Region> {
        self.index_of(key).map(|i| &self.regions[i])
    }

    /// The dense index of region `key` in this map.
    #[must_use]
    pub fn index_of(&self, key: RegionKey) -> Option<usize> {
        self.regions.binary_search_by_key(&key, |r| r.key).ok()
    }

    /// The region at texel `(x, y)`, if any.
    #[must_use]
    pub fn key_at(&self, x: i64, y: i64) -> Option<RegionKey> {
        let label = self.labels.at(x, y);
        (label > 0).then(|| self.regions[label as usize - 1].key)
    }

    /// How the regions of `self` (old) relate to those of `new`, linking
    /// two regions when they share at least `min_share` of the smaller
    /// one's texels.
    ///
    /// # Errors
    ///
    /// [`RegionError::GridMismatch`] when the maps have different sizes.
    pub fn correspondence(
        &self,
        new: &Self,
        min_share: f32,
    ) -> Result<RegionCorrespondence, RegionError> {
        let links = self.links(new, min_share)?;
        Ok(correspond(self, new, &links))
    }

    /// Texel overlaps between `self`'s and `new`'s regions (dense indices),
    /// kept when they reach `min_share` of the smaller region.
    fn links(&self, new: &Self, min_share: f32) -> Result<Vec<(usize, usize, u32)>, RegionError> {
        if (self.labels.width(), self.labels.height()) != (new.labels.width(), new.labels.height())
        {
            return Err(RegionError::GridMismatch);
        }
        let mut overlaps: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        for (&a, &b) in self.labels.values().iter().zip(new.labels.values()) {
            if a > 0 && b > 0 {
                *overlaps.entry((a, b)).or_insert(0) += 1;
            }
        }
        Ok(overlaps
            .into_iter()
            .filter_map(|((a, b), n)| {
                let (a, b) = (a as usize - 1, b as usize - 1);
                let smaller = self.regions[a].texels.min(new.regions[b].texels);
                #[expect(clippy::cast_precision_loss, reason = "texel counts")]
                let share = n as f32 / smaller as f32;
                (share >= min_share).then_some((a, b, n))
            })
            .collect())
    }

    /// This canonical map rekeyed to keep `previous`'s identities where its
    /// regions correspond (see the [module docs](self)), with the
    /// correspondence from `previous` to the result.
    ///
    /// Deterministic in `self` and `previous`: `previous` is the persisted
    /// identity state, and the result never depends on how either was
    /// reached.
    ///
    /// # Errors
    ///
    /// [`RegionError::GridMismatch`], or [`RegionError::KeyCollision`] when
    /// a kept key equals another region's canonical key.
    pub fn retaining(
        &self,
        previous: &Self,
        min_share: f32,
    ) -> Result<(Self, RegionCorrespondence), RegionError> {
        let mut links = previous.links(self, min_share)?;
        // Largest overlaps first, ties by the keys, old then new.
        links.sort_by(|a, b| {
            b.2.cmp(&a.2)
                .then(previous.regions[a.0].key.cmp(&previous.regions[b.0].key))
                .then(self.regions[a.1].key.cmp(&self.regions[b.1].key))
        });
        let mut kept: Vec<Option<RegionKey>> = vec![None; self.regions.len()];
        let mut used = vec![false; previous.regions.len()];
        for &(a, b, _) in &links {
            if !used[a] && kept[b].is_none() {
                used[a] = true;
                kept[b] = Some(previous.regions[a].key);
            }
        }
        let rekey: Vec<RegionKey> = self
            .regions
            .iter()
            .zip(&kept)
            .map(|(r, k)| k.unwrap_or(r.key))
            .collect();
        let mut sorted = rekey.clone();
        sorted.sort_unstable();
        if let Some(pair) = sorted.windows(2).find(|p| p[0] == p[1]) {
            return Err(RegionError::KeyCollision(pair[0]));
        }
        // New order and labels.
        let mut order: Vec<usize> = (0..self.regions.len()).collect();
        order.sort_by_key(|&i| rekey[i]);
        let mut relabel = vec![0_u32; self.regions.len() + 1];
        for (new_index, &old_index) in order.iter().enumerate() {
            relabel[old_index + 1] = u32::try_from(new_index + 1).expect("fewer than 2^32");
        }
        let labels = Raster::from_values(
            self.labels.width(),
            self.labels.height(),
            self.labels.origin(),
            self.labels.texel(),
            self.labels.edge(),
            self.labels
                .values()
                .iter()
                .map(|&l| relabel[l as usize])
                .collect(),
        )
        .map_err(RegionError::Raster)?;
        let key_of: BTreeMap<RegionKey, RegionKey> = self
            .regions
            .iter()
            .zip(&rekey)
            .map(|(r, &k)| (r.key, k))
            .collect();
        let regions = order
            .iter()
            .map(|&i| {
                let mut r = self.regions[i].clone();
                r.key = rekey[i];
                r.neighbors = r.neighbors.iter().map(|k| key_of[k]).collect();
                r.neighbors.sort_unstable();
                r
            })
            .collect();
        let pieces = order.iter().map(|&i| self.pieces[i].clone()).collect();
        let map = Self {
            labels,
            regions,
            pieces,
        };
        let report = previous.correspondence(&map, min_share)?;
        Ok((map, report))
    }

    /// 1 on texels with a 4-neighbor in another region (or in none), 0
    /// elsewhere: the regions' edges.
    #[must_use]
    pub fn boundaries(&self) -> Raster {
        let (w, h) = (self.labels.width(), self.labels.height());
        let values = (0..h)
            .flat_map(|y| (0..w).map(move |x| (i64::from(x), i64::from(y))))
            .map(|(x, y)| {
                let a = self.labels.at(x, y);
                let edge = a != 0
                    && [(1, 0), (-1, 0), (0, 1), (0, -1)]
                        .iter()
                        .any(|&(dx, dy)| self.neighbor(x + dx, y + dy).is_some_and(|b| b != a));
                if edge { 1.0 } else { 0.0 }
            })
            .collect();
        self.like(values)
    }

    /// The label at `(x, y)`, or 0 past a clamped border.
    fn neighbor(&self, x: i64, y: i64) -> Option<u32> {
        let (w, h) = (
            i64::from(self.labels.width()),
            i64::from(self.labels.height()),
        );
        let outside = !(0..w).contains(&x) || !(0..h).contains(&y);
        if self.labels.edge() == Edge::Clamp && outside {
            return Some(0);
        }
        Some(self.labels.at(x, y))
    }

    /// The distance, in domain units, from each labeled texel's center to
    /// the nearest texel center outside its region; 0 on unlabeled texels.
    /// Insets and bevel profiles are functions of it (a 3 mm inset is the
    /// texels farther than 3 mm in).
    ///
    /// Computed per region with an exact distance transform over the
    /// region's bounds grown by a texel, so neighboring regions never
    /// shorten each other's distances. Past a clamped border counts as
    /// outside.
    ///
    /// # Errors
    ///
    /// Never for a valid map.
    pub fn inset(&self) -> Result<Raster, RegionError> {
        let (w, h) = (self.labels.width(), self.labels.height());
        let mut out = vec![0.0_f32; (w as usize) * (h as usize)];
        let resolve = |x: i64, y: i64| {
            let (wi, hi) = (i64::from(w), i64::from(h));
            (x.rem_euclid(wi), y.rem_euclid(hi))
        };
        for (i, pieces) in self.pieces.iter().enumerate() {
            let label = u32::try_from(i + 1).expect("fewer than 2^32 regions");
            for &[x0, y0, x1, y1] in pieces {
                let (x0, y0, x1, y1) = (x0 - 1, y0 - 1, x1 + 1, y1 + 1);
                let (bw, bh) = (
                    u32::try_from(x1 - x0 + 1).expect("bounded by the grid"),
                    u32::try_from(y1 - y0 + 1).expect("bounded by the grid"),
                );
                let outside: Vec<f32> = (y0..=y1)
                    .flat_map(|y| (x0..=x1).map(move |x| (x, y)))
                    .map(|(x, y)| {
                        let inside = self.neighbor(x, y) == Some(label);
                        if inside { 0.0 } else { 1.0 }
                    })
                    .collect();
                let local = Raster::from_values(
                    bw,
                    bh,
                    Vec2::ZERO,
                    self.labels.texel(),
                    Edge::Clamp,
                    outside,
                )
                .map_err(RegionError::Raster)?;
                let distance = DistanceTransform { threshold: 0.5 }
                    .apply(&local)
                    .map_err(RegionError::Raster)?;
                let columns = (x0..=x1).count();
                for (j, &d) in distance.values().iter().enumerate() {
                    let (lx, ly) = (j % columns, j / columns);
                    let (x, y) = (
                        x0 + i64::try_from(lx).expect("small"),
                        y0 + i64::try_from(ly).expect("small"),
                    );
                    if self.neighbor(x, y) != Some(label) {
                        continue;
                    }
                    let (x, y) = resolve(x, y);
                    let t = usize::try_from(y * i64::from(w) + x).expect("in the grid");
                    out[t] = d;
                }
            }
        }
        Ok(self.like(out))
    }

    /// 1 on region `key`'s texels, 0 elsewhere, for per-region morphology.
    #[must_use]
    pub fn mask(&self, key: RegionKey) -> Raster {
        let label = self
            .index_of(key)
            .map_or(u32::MAX, |i| u32::try_from(i + 1).expect("fewer than 2^32"));
        let values = self
            .labels
            .values()
            .iter()
            .map(|&l| if l == label { 1.0 } else { 0.0 })
            .collect();
        self.like(values)
    }

    /// The smallest, largest and mean of `values` over each region, in the
    /// table's order.
    ///
    /// # Errors
    ///
    /// [`RegionError::GridMismatch`] when `values` is not on the map's grid.
    pub fn statistics(&self, values: &Raster) -> Result<Vec<RegionStatistics>, RegionError> {
        if (values.width(), values.height()) != (self.labels.width(), self.labels.height()) {
            return Err(RegionError::GridMismatch);
        }
        let mut acc = vec![(f32::INFINITY, f32::NEG_INFINITY, 0.0_f64); self.regions.len()];
        for (&l, &v) in self.labels.values().iter().zip(values.values()) {
            if l > 0 {
                let a = &mut acc[l as usize - 1];
                a.0 = a.0.min(v);
                a.1 = a.1.max(v);
                a.2 += f64::from(v);
            }
        }
        Ok(self
            .regions
            .iter()
            .zip(acc)
            .map(|(r, (min, max, sum))| {
                #[expect(clippy::cast_possible_truncation, reason = "narrowing the mean")]
                let mean = (sum / f64::from(r.texels)) as f32;
                RegionStatistics {
                    key: r.key,
                    min,
                    max,
                    mean,
                }
            })
            .collect())
    }

    fn like(&self, values: Vec<f32>) -> Raster {
        Raster::from_values(
            self.labels.width(),
            self.labels.height(),
            self.labels.origin(),
            self.labels.texel(),
            self.labels.edge(),
            values,
        )
        .expect("the map's grid")
    }
}

/// Texel index `i` of a `width`-wide grid as `[x, y]`.
fn xy(i: usize, width: u32) -> [u32; 2] {
    let w = width as usize;
    [
        u32::try_from(i % w).expect("x fits u32"),
        u32::try_from(i / w).expect("y fits u32"),
    ]
}

/// Connected pieces of equal nonzero class, in row-major order of their
/// anchors: each texel's piece (`u32::MAX` for class 0) and the pieces.
fn pieces(
    w: u32,
    h: u32,
    wrap: bool,
    classes: &[u32],
    connectivity: Connectivity,
) -> (Vec<u32>, Vec<Piece>) {
    let (wi, hi) = (i64::from(w), i64::from(h));
    let steps: &[(i64, i64)] = match connectivity {
        Connectivity::Four => &[(1, 0), (-1, 0), (0, 1), (0, -1)],
        Connectivity::Eight => &[
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ],
    };
    let mut piece_of = vec![u32::MAX; classes.len()];
    let mut out = Vec::new();
    let mut queue: Vec<(usize, i64, i64)> = Vec::new();
    for start in 0..classes.len() {
        let class = classes[start];
        if class == 0 || piece_of[start] != u32::MAX {
            continue;
        }
        let id = u32::try_from(out.len()).expect("fewer than 2^32 pieces");
        let [sx, sy] = xy(start, w);
        let mut piece = Piece {
            class,
            anchor: start,
            texels: 0,
            sum: [0.0; 2],
            sum2: [0.0; 3],
            lo: [i64::MAX; 2],
            hi: [i64::MIN; 2],
        };
        piece_of[start] = id;
        queue.clear();
        queue.push((start, i64::from(sx), i64::from(sy)));
        let mut head = 0;
        while head < queue.len() {
            let (_, ux, uy) = queue[head];
            head += 1;
            #[expect(clippy::cast_precision_loss, reason = "texel coordinates")]
            let (fx, fy) = (ux as f64, uy as f64);
            piece.texels += 1;
            piece.sum[0] += fx;
            piece.sum[1] += fy;
            piece.sum2[0] += fx * fx;
            piece.sum2[1] += fx * fy;
            piece.sum2[2] += fy * fy;
            piece.lo = [piece.lo[0].min(ux), piece.lo[1].min(uy)];
            piece.hi = [piece.hi[0].max(ux), piece.hi[1].max(uy)];
            for &(dx, dy) in steps {
                let (nx, ny) = (ux + dx, uy + dy);
                let (rx, ry) = if wrap {
                    (nx.rem_euclid(wi), ny.rem_euclid(hi))
                } else if (0..wi).contains(&nx) && (0..hi).contains(&ny) {
                    (nx, ny)
                } else {
                    continue;
                };
                let n = usize::try_from(ry * wi + rx).expect("in the grid");
                if classes[n] == class && piece_of[n] == u32::MAX {
                    piece_of[n] = id;
                    queue.push((n, nx, ny));
                }
            }
        }
        out.push(piece);
    }
    (piece_of, out)
}

/// A split: one region became several.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Split {
    /// The old region.
    pub from: RegionKey,
    /// The new regions, ascending.
    pub into: Vec<RegionKey>,
}

/// A merge: several regions became one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Merge {
    /// The old regions, ascending.
    pub from: Vec<RegionKey>,
    /// The new region.
    pub into: RegionKey,
}

/// Several regions became several others, with splits and merges tangled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Regroup {
    /// The old regions, ascending.
    pub from: Vec<RegionKey>,
    /// The new regions, ascending.
    pub into: Vec<RegionKey>,
}

/// Texels an old and a new region share.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Overlap {
    /// The old region.
    pub old: RegionKey,
    /// The new region.
    pub new: RegionKey,
    /// Shared texels.
    pub texels: u32,
}

/// How the regions of one map relate to another's: a report only.
///
/// Every old and every new region appears in exactly one group; each list
/// is in key order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RegionCorrespondence {
    /// One old region to one new, `(old, new)`: the same key when identity
    /// was kept.
    pub matched: Vec<(RegionKey, RegionKey)>,
    /// One old region into several new ones.
    pub splits: Vec<Split>,
    /// Several old regions into one new one.
    pub merges: Vec<Merge>,
    /// Several into several.
    pub regroups: Vec<Regroup>,
    /// New regions linked to no old one.
    pub appeared: Vec<RegionKey>,
    /// Old regions linked to no new one.
    pub vanished: Vec<RegionKey>,
    /// Every link, by old then new key.
    pub overlaps: Vec<Overlap>,
}

/// Groups the linked regions of `old` and `new` (dense indices).
fn correspond(
    old: &RegionMap,
    new: &RegionMap,
    links: &[(usize, usize, u32)],
) -> RegionCorrespondence {
    // Union-find over old regions then new ones.
    let n_old = old.regions.len();
    let mut parent: Vec<usize> = (0..n_old + new.regions.len()).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for &(a, b, _) in links {
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, n_old + b));
        if ra != rb {
            parent[ra.max(rb)] = ra.min(rb);
        }
    }
    let mut groups: BTreeMap<usize, (Vec<RegionKey>, Vec<RegionKey>)> = BTreeMap::new();
    for i in 0..parent.len() {
        let root = find(&mut parent, i);
        let group = groups.entry(root).or_default();
        if i < n_old {
            group.0.push(old.regions[i].key);
        } else {
            group.1.push(new.regions[i - n_old].key);
        }
    }
    let mut report = RegionCorrespondence::default();
    for (_, (mut from, mut into)) in groups {
        from.sort_unstable();
        into.sort_unstable();
        match (from.len(), into.len()) {
            (1, 1) => report.matched.push((from[0], into[0])),
            (1, 0) => report.vanished.push(from[0]),
            (0, 1) => report.appeared.push(into[0]),
            (1, _) => report.splits.push(Split {
                from: from[0],
                into,
            }),
            (_, 1) => report.merges.push(Merge {
                from,
                into: into[0],
            }),
            _ => report.regroups.push(Regroup { from, into }),
        }
    }
    report.matched.sort_unstable();
    report.splits.sort_by_key(|s| s.from);
    report.merges.sort_by_key(|m| m.into);
    report.regroups.sort_by(|a, b| a.from.cmp(&b.from));
    report.appeared.sort_unstable();
    report.vanished.sort_unstable();
    report.overlaps = links
        .iter()
        .map(|&(a, b, texels)| Overlap {
            old: old.regions[a].key,
            new: new.regions[b].key,
            texels,
        })
        .collect();
    report.overlaps.sort_by_key(|o| (o.old, o.new));
    report
}
