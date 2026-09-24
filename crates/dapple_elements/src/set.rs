// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Element sets: column tables of keyed elements in canonical key order.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use dapple_field::hash::hash;
use dapple_field::{PortType, Value};
use glam::Vec2;

use crate::identity::ElementKey;

/// An element's rigid placement in the domain: a rotation about its center,
/// then a translation to its center.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Placement {
    /// The element's center, in domain units.
    pub center: Vec2,
    /// Counterclockwise rotation, in radians.
    pub rotation: f32,
}

impl Placement {
    /// A placement at `center` without rotation.
    #[must_use]
    pub const fn at(center: Vec2) -> Self {
        Self {
            center,
            rotation: 0.0,
        }
    }

    /// The element-local position of domain point `p`.
    #[must_use]
    pub fn to_local(self, p: Vec2) -> Vec2 {
        let (s, c) = (libm::sinf(self.rotation), libm::cosf(self.rotation));
        let d = p - self.center;
        Vec2::new(c * d.x + s * d.y, -s * d.x + c * d.y)
    }

    fn words(self) -> [u64; 3] {
        [
            u64::from(self.center.x.to_bits()),
            u64::from(self.center.y.to_bits()),
            u64::from(self.rotation.to_bits()),
        ]
    }
}

/// An axis-aligned box in domain units.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Bounds {
    /// Minimum corner.
    pub min: Vec2,
    /// Maximum corner.
    pub max: Vec2,
}

impl Bounds {
    /// Whether `p` lies inside or on the box.
    #[must_use]
    pub fn contains(self, p: Vec2) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }

    /// The box grown by `pad` on every side.
    #[must_use]
    pub fn grown(self, pad: f32) -> Self {
        Self {
            min: self.min - pad,
            max: self.max + pad,
        }
    }
}

/// An element's outline within its half extent.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum Outline {
    /// The rectangle of its half extent: bricks, tiles, planks.
    #[default]
    Rectangle,
    /// The ellipse inscribed in it: pebbles, flakes, spots.
    Ellipse,
}

impl Outline {
    /// The signed distance from element-local point `q` to the outline of
    /// half extent `half`, negative inside.
    ///
    /// Exact for rectangles and circles; for other ellipses a first-order
    /// estimate, exact on the outline and good near it, which is where
    /// coverage and bevels read it.
    #[must_use]
    pub fn signed_distance(self, q: Vec2, half: Vec2) -> f32 {
        match self {
            Self::Rectangle => {
                let d = q.abs() - half;
                d.max(Vec2::ZERO).length() + d.x.max(d.y).min(0.0)
            }
            Self::Ellipse if half.x == half.y => q.length() - half.x,
            Self::Ellipse => {
                let k0 = (q / half).length();
                let k1 = (q / (half * half)).length();
                if k1 > 0.0 {
                    k0 * (k0 - 1.0) / k1
                } else {
                    -half.min_element()
                }
            }
        }
    }
}

/// A named, typed per-element attribute column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttributeDecl {
    /// The attribute's name, unique in its set.
    pub name: String,
    /// The type of its values.
    pub port: PortType,
}

/// One element, as given to [`ElementSet::new`].
#[derive(Clone, Debug, PartialEq)]
pub struct Element {
    /// Identity.
    pub key: ElementKey,
    /// Where it is.
    pub placement: Placement,
    /// Half its extent along its local axes, in domain units; positive.
    pub half_size: Vec2,
    /// Its outline within that extent.
    pub outline: Outline,
    /// Which variant of the realizing program it uses.
    pub variant: u32,
    /// Attribute values in the set's schema order.
    pub attributes: Vec<Value>,
}

/// An element-set failure.
#[derive(Clone, Debug, PartialEq)]
pub enum ElementError {
    /// Two elements share a key.
    DuplicateKey(ElementKey),
    /// No element has this key.
    UnknownKey(ElementKey),
    /// No attribute has this name.
    UnknownAttribute(String),
    /// Two attributes share a name.
    DuplicateAttribute(String),
    /// An element's attribute values do not match the schema.
    AttributeMismatch {
        /// The element.
        key: ElementKey,
        /// The attribute's position in the schema, or the schema's length
        /// when the element has the wrong number of values.
        index: usize,
    },
    /// A placement or size is not finite, or a size is not positive.
    InvalidGeometry(ElementKey),
}

impl fmt::Display for ElementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateKey(key) => write!(f, "two elements share {key:?}"),
            Self::UnknownKey(key) => write!(f, "no element has {key:?}"),
            Self::UnknownAttribute(name) => write!(f, "no attribute is named {name:?}"),
            Self::DuplicateAttribute(name) => write!(f, "two attributes are named {name:?}"),
            Self::AttributeMismatch { key, index } => {
                write!(f, "{key:?}: attribute {index} does not match the schema")
            }
            Self::InvalidGeometry(key) => write!(f, "{key:?}: invalid placement or size"),
        }
    }
}

impl core::error::Error for ElementError {}

/// Whether `value` is a value of type `port`.
pub(crate) fn value_fits(value: Value, port: PortType) -> bool {
    matches!(
        (value, port),
        (Value::Scalar(_), PortType::Scalar | PortType::Mask)
            | (Value::Id(_), PortType::Id)
            | (Value::Vector2(_), PortType::Vector2 | PortType::Direction)
            | (
                Value::Vector3(_),
                PortType::Vector3 | PortType::Color(_) | PortType::Normal(_)
            )
    )
}

pub(crate) fn value_words(value: Value, out: &mut Vec<u64>) {
    let f = |v: f32| u64::from(v.to_bits());
    match value {
        Value::Scalar(v) => out.extend([1, f(v)]),
        Value::Id(v) => out.extend([2, u64::from(v)]),
        Value::Vector2(v) => out.extend([3, f(v.x), f(v.y)]),
        Value::Vector3(v) => out.extend([4, f(v.x), f(v.y), f(v.z)]),
    }
}

/// Keyed elements as a column table, sorted by key.
///
/// The order is canonical: the same elements give the same set whatever
/// order they were supplied or edited in, and an element's position in the
/// table is its dense index in this set only.
#[derive(Clone, Debug, PartialEq)]
pub struct ElementSet {
    schema: Vec<AttributeDecl>,
    keys: Vec<ElementKey>,
    placements: Vec<Placement>,
    half_sizes: Vec<Vec2>,
    outlines: Vec<Outline>,
    variants: Vec<u32>,
    /// One column per schema attribute.
    columns: Vec<Vec<Value>>,
}

impl ElementSet {
    /// Builds a set from elements in any order.
    ///
    /// # Errors
    ///
    /// [`ElementError::DuplicateKey`], [`ElementError::DuplicateAttribute`],
    /// [`ElementError::AttributeMismatch`] or
    /// [`ElementError::InvalidGeometry`].
    pub fn new(
        schema: Vec<AttributeDecl>,
        mut elements: Vec<Element>,
    ) -> Result<Self, ElementError> {
        for (i, a) in schema.iter().enumerate() {
            if schema[..i].iter().any(|b| b.name == a.name) {
                return Err(ElementError::DuplicateAttribute(a.name.clone()));
            }
        }
        elements.sort_by_key(|e| e.key);
        for pair in elements.windows(2) {
            if pair[0].key == pair[1].key {
                return Err(ElementError::DuplicateKey(pair[0].key));
            }
        }
        let mut set = Self {
            columns: schema
                .iter()
                .map(|_| Vec::with_capacity(elements.len()))
                .collect(),
            schema,
            keys: Vec::with_capacity(elements.len()),
            placements: Vec::with_capacity(elements.len()),
            half_sizes: Vec::with_capacity(elements.len()),
            outlines: Vec::with_capacity(elements.len()),
            variants: Vec::with_capacity(elements.len()),
        };
        for e in elements {
            check_geometry(e.key, e.placement, e.half_size)?;
            if e.attributes.len() != set.schema.len() {
                return Err(ElementError::AttributeMismatch {
                    key: e.key,
                    index: set.schema.len(),
                });
            }
            for (index, (&value, decl)) in e.attributes.iter().zip(&set.schema).enumerate() {
                if !value_fits(value, decl.port) {
                    return Err(ElementError::AttributeMismatch { key: e.key, index });
                }
                set.columns[index].push(value);
            }
            set.keys.push(e.key);
            set.placements.push(e.placement);
            set.half_sizes.push(e.half_size);
            set.outlines.push(e.outline);
            set.variants.push(e.variant);
        }
        Ok(set)
    }

    /// The number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether the set has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// The attribute schema.
    #[must_use]
    pub fn schema(&self) -> &[AttributeDecl] {
        &self.schema
    }

    /// Keys in canonical order.
    #[must_use]
    pub fn keys(&self) -> &[ElementKey] {
        &self.keys
    }

    /// The dense index of `key` in this set.
    #[must_use]
    pub fn index_of(&self, key: ElementKey) -> Option<usize> {
        self.keys.binary_search(&key).ok()
    }

    /// Element `i`'s placement.
    #[must_use]
    pub fn placement(&self, i: usize) -> Placement {
        self.placements[i]
    }

    /// Element `i`'s half extent.
    #[must_use]
    pub fn half_size(&self, i: usize) -> Vec2 {
        self.half_sizes[i]
    }

    /// Element `i`'s outline.
    #[must_use]
    pub fn outline(&self, i: usize) -> Outline {
        self.outlines[i]
    }

    /// Element `i`'s variant.
    #[must_use]
    pub fn variant(&self, i: usize) -> u32 {
        self.variants[i]
    }

    /// The column of the attribute named `name`, in key order.
    #[must_use]
    pub fn attribute(&self, name: &str) -> Option<&[Value]> {
        let i = self.schema.iter().position(|a| a.name == name)?;
        Some(&self.columns[i])
    }

    /// Element `i` as an [`Element`].
    #[must_use]
    pub fn element(&self, i: usize) -> Element {
        Element {
            key: self.keys[i],
            placement: self.placements[i],
            half_size: self.half_sizes[i],
            outline: self.outlines[i],
            variant: self.variants[i],
            attributes: self.columns.iter().map(|c| c[i]).collect(),
        }
    }

    /// Element `i`'s axis-aligned bounds in the domain (before any periodic
    /// wrap).
    #[must_use]
    pub fn bounds(&self, i: usize) -> Bounds {
        let p = self.placements[i];
        let h = self.half_sizes[i];
        let (s, c) = (libm::sinf(p.rotation), libm::cosf(p.rotation));
        let extent = Vec2::new(c.abs() * h.x + s.abs() * h.y, s.abs() * h.x + c.abs() * h.y);
        Bounds {
            min: p.center - extent,
            max: p.center + extent,
        }
    }

    /// Element `i`'s content fingerprint: its key, placement, size, outline,
    /// variant and attribute values. It changes when the element changes; its
    /// identity does not.
    #[must_use]
    pub fn fingerprint_of(&self, i: usize) -> u64 {
        let mut words = Vec::with_capacity(8 + 4 * self.columns.len());
        words.push(self.keys[i].word());
        words.extend(self.placements[i].words());
        words.extend([
            u64::from(self.half_sizes[i].x.to_bits()),
            u64::from(self.half_sizes[i].y.to_bits()),
            match self.outlines[i] {
                Outline::Rectangle => 0,
                Outline::Ellipse => 1,
            },
            u64::from(self.variants[i]),
        ]);
        for column in &self.columns {
            value_words(column[i], &mut words);
        }
        hash(0x0065_6c65_6d65_6e74, &words) // "element"
    }

    /// The set's content fingerprint: its schema and every element's
    /// fingerprint, in key order.
    #[must_use]
    pub fn fingerprint(&self) -> u64 {
        let mut words = Vec::with_capacity(self.len() + 2 * self.schema.len());
        for a in &self.schema {
            words.push(a.name.len() as u64);
            words.extend(a.name.bytes().map(u64::from));
            words.push(dapple_raster::typed::port_word(a.port));
        }
        words.extend((0..self.len()).map(|i| self.fingerprint_of(i)));
        hash(0x0065_6c65_6d73_6574, &words) // "elemset"
    }

    /// The elements `keep` accepts, with their identities.
    #[must_use]
    pub fn filter(&self, mut keep: impl FnMut(&Element) -> bool) -> Self {
        let elements = (0..self.len())
            .map(|i| self.element(i))
            .filter(|e| keep(e))
            .collect();
        Self::new(self.schema.clone(), elements).expect("a subset of a valid set is valid")
    }

    /// Moves element `key`; its identity, attributes and everything derived
    /// from them are unchanged.
    ///
    /// # Errors
    ///
    /// [`ElementError::UnknownKey`] or [`ElementError::InvalidGeometry`].
    pub fn set_placement(
        &mut self,
        key: ElementKey,
        placement: Placement,
    ) -> Result<(), ElementError> {
        let i = self.index_of(key).ok_or(ElementError::UnknownKey(key))?;
        check_geometry(key, placement, self.half_sizes[i])?;
        self.placements[i] = placement;
        Ok(())
    }

    /// Sets element `key`'s attribute `name`.
    ///
    /// # Errors
    ///
    /// [`ElementError::UnknownKey`], [`ElementError::UnknownAttribute`] or
    /// [`ElementError::AttributeMismatch`].
    pub fn set_attribute(
        &mut self,
        key: ElementKey,
        name: &str,
        value: Value,
    ) -> Result<(), ElementError> {
        let i = self.index_of(key).ok_or(ElementError::UnknownKey(key))?;
        let a = self
            .schema
            .iter()
            .position(|a| a.name == name)
            .ok_or_else(|| ElementError::UnknownAttribute(name.into()))?;
        if !value_fits(value, self.schema[a].port) {
            return Err(ElementError::AttributeMismatch { key, index: a });
        }
        self.columns[a][i] = value;
        Ok(())
    }
}

fn check_geometry(key: ElementKey, p: Placement, half: Vec2) -> Result<(), ElementError> {
    if p.center.is_finite()
        && p.rotation.is_finite()
        && half.is_finite()
        && half.min_element() > 0.0
    {
        Ok(())
    } else {
        Err(ElementError::InvalidGeometry(key))
    }
}

/// How the elements of one set relate to those of another.
///
/// A report only: it never changes either set. Elements correspond by key,
/// so this is exact for sets that keep identities (layouts, explicit
/// edits); reconstructed regions need their own matching.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Correspondence {
    /// Keys in both sets with equal content.
    pub unchanged: Vec<ElementKey>,
    /// Keys in both sets whose content changed.
    pub changed: Vec<ElementKey>,
    /// Keys only in the new set.
    pub added: Vec<ElementKey>,
    /// Keys only in the old set.
    pub removed: Vec<ElementKey>,
}

/// The correspondence of `old`'s elements to `new`'s, all lists in key
/// order.
#[must_use]
pub fn correspondence(old: &ElementSet, new: &ElementSet) -> Correspondence {
    let mut report = Correspondence::default();
    let (mut i, mut j) = (0, 0);
    while i < old.len() || j < new.len() {
        match (old.keys.get(i), new.keys.get(j)) {
            (Some(&a), Some(&b)) if a == b => {
                if old.fingerprint_of(i) == new.fingerprint_of(j) {
                    report.unchanged.push(a);
                } else {
                    report.changed.push(a);
                }
                i += 1;
                j += 1;
            }
            (Some(&a), Some(&b)) if a < b => {
                report.removed.push(a);
                i += 1;
            }
            (Some(&a), None) => {
                report.removed.push(a);
                i += 1;
            }
            (_, Some(&b)) => {
                report.added.push(b);
                j += 1;
            }
            (None, None) => unreachable!("the loop runs while either remains"),
        }
    }
    report
}
