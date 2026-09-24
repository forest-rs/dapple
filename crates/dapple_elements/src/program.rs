// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Binding scoped programs to elements and regions.
//!
//! The program itself is `dapple_field`'s programmable core, a
//! [`ScopedProgram`]: an inspectable value with typed, scoped inputs and
//! outputs, resources, functions and a body. This module adds what elements
//! and regions supply to it:
//!
//! - [`Binding`]: where an input's value comes from, each with the
//!   [`Scope`] of the values it supplies: a constant, an element attribute
//!   or identity-derived random value, a region property, or the sample's
//!   position;
//! - [`ProgramInstance`]: a program with one binding per input and a stable
//!   instance identity ([`InstanceId`]), separate from its content
//!   fingerprint, as element keys are separate from element fingerprints.
//!
//! Binding checks scopes: an instance binding a per-sample value to a
//! per-element input is refused. Evaluation relies on it: nodes of material
//! and element scope run once per element when compositing
//! ([`crate::Realized::composite`]), and nodes of material and region scope
//! once per region ([`crate::RegionMap::evaluate`]).

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use dapple_field::hash::hash;
use dapple_field::scoped::{
    ContractError, Scope, ScopedProgram, Shape, name_word, value_fits, value_words,
};
use dapple_field::{Footprint, Value};
use glam::{Vec2, Vec3};

use crate::identity::ElementKey;
use crate::set::ElementSet;

/// A program instance's stable identity: a hash of its author-given name.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct InstanceId(u64);

impl InstanceId {
    /// The identity of the instance named `name`.
    #[must_use]
    pub fn named(name: &str) -> Self {
        Self(name_word(0x696e_7374_616e_6365, name)) // "instance"
    }

    /// The identity's hash word.
    #[must_use]
    pub const fn word(self) -> u64 {
        self.0
    }
}

/// Where an instance's input comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum Binding {
    /// A material parameter ([`Scope::Material`]).
    Constant(Value),
    /// A per-element attribute of the element set, by name
    /// ([`Scope::Element`]).
    Attribute(String),
    /// A uniform value in `[0, 1)` from the element's key and a stream
    /// ([`Scope::Element`]); it follows the element wherever it moves.
    ElementRandom(u64),
    /// The element's half extent, a `Vector2` ([`Scope::Element`]).
    HalfSize,
    /// The element's variant as a scalar (0, 1, 2, …), for programs that
    /// choose a shape or sub-material per variant ([`Scope::Element`]).
    Variant,
    /// The sample's element-local position, a `Vector2`
    /// ([`Scope::Sample`]).
    LocalPosition,
    /// The sample's distance inside the element's boundary, a scalar,
    /// negative outside ([`Scope::Sample`]).
    EdgeDistance,
    /// A uniform value in `[0, 1)` from the region's key and a stream
    /// ([`Scope::Region`]). A composited region's key is its element's, so
    /// it equals that element's [`Binding::ElementRandom`] for the stream.
    RegionRandom(u64),
    /// The region's area in square domain units ([`Scope::Region`]).
    RegionArea,
    /// The region's centroid, a `Vector2` in domain units
    /// ([`Scope::Region`]).
    RegionCentroid,
    /// The angle of the region's principal axis in radians
    /// ([`Scope::Region`]).
    RegionOrientation,
    /// The sample's position in the domain, a `Vector2` ([`Scope::Sample`]).
    Position,
}

impl Binding {
    /// The scope of the values this binding supplies.
    #[must_use]
    pub const fn scope(&self) -> Scope {
        match self {
            Self::Constant(_) => Scope::Material,
            Self::Attribute(_) | Self::ElementRandom(_) | Self::HalfSize | Self::Variant => {
                Scope::Element
            }
            Self::RegionRandom(_)
            | Self::RegionArea
            | Self::RegionCentroid
            | Self::RegionOrientation => Scope::Region,
            Self::LocalPosition | Self::EdgeDistance | Self::Position => Scope::Sample,
        }
    }

    fn words(&self, w: &mut Vec<u64>) {
        match self {
            Self::Constant(v) => {
                w.push(1);
                value_words(*v, w);
            }
            Self::Attribute(name) => w.extend([2, name_word(5, name)]),
            Self::ElementRandom(stream) => w.extend([3, *stream]),
            Self::HalfSize => w.push(4),
            Self::LocalPosition => w.push(5),
            Self::EdgeDistance => w.push(6),
            Self::Variant => w.push(7),
            Self::RegionRandom(stream) => w.extend([8, *stream]),
            Self::RegionArea => w.push(9),
            Self::RegionCentroid => w.push(10),
            Self::RegionOrientation => w.push(11),
            Self::Position => w.push(12),
        }
    }
}

/// A program bound for use: its identity, the program, and one binding per
/// input.
#[derive(Clone, Debug, PartialEq)]
pub struct ProgramInstance {
    id: InstanceId,
    program: Arc<ScopedProgram>,
    bindings: Vec<Binding>,
}

impl ProgramInstance {
    /// Binds `program`'s inputs, in declaration order.
    ///
    /// # Errors
    ///
    /// [`ContractError::BindingCount`]; [`ContractError::ScopeViolation`]
    /// when a binding varies more often than its input allows (a sample
    /// position bound to a per-element input); [`ContractError::TypeMismatch`]
    /// when a constant or context value does not fit.
    pub fn new(
        id: InstanceId,
        program: Arc<ScopedProgram>,
        bindings: Vec<Binding>,
    ) -> Result<Self, ContractError> {
        if bindings.len() != program.inputs().len() {
            return Err(ContractError::BindingCount);
        }
        for (input, binding) in program.inputs().iter().zip(&bindings) {
            if !binding.scope().within(input.scope) {
                return Err(ContractError::ScopeViolation {
                    name: input.name.clone(),
                    declared: input.scope,
                    found: binding.scope(),
                });
            }
            let shape = match binding {
                Binding::Constant(v) => {
                    if !value_fits(*v, input.port) {
                        return Err(ContractError::TypeMismatch(input.name.clone()));
                    }
                    continue;
                }
                Binding::Attribute(_) => continue,
                Binding::ElementRandom(_)
                | Binding::EdgeDistance
                | Binding::Variant
                | Binding::RegionRandom(_)
                | Binding::RegionArea
                | Binding::RegionOrientation => Shape::Scalar,
                Binding::HalfSize
                | Binding::LocalPosition
                | Binding::RegionCentroid
                | Binding::Position => Shape::Vector2,
            };
            if shape != Shape::of(input.port) {
                return Err(ContractError::TypeMismatch(input.name.clone()));
            }
        }
        Ok(Self {
            id,
            program,
            bindings,
        })
    }

    /// The instance's identity.
    #[must_use]
    pub const fn id(&self) -> InstanceId {
        self.id
    }

    /// The program.
    #[must_use]
    pub fn program(&self) -> &ScopedProgram {
        &self.program
    }

    /// The bindings, in input order.
    #[must_use]
    pub fn bindings(&self) -> &[Binding] {
        &self.bindings
    }

    /// Rebinds input `name`, keeping the instance's identity.
    ///
    /// # Errors
    ///
    /// As [`ProgramInstance::new`], or [`ContractError::UnknownReference`]
    /// for an unknown input.
    pub fn rebind(&self, name: &str, binding: Binding) -> Result<Self, ContractError> {
        let i = self
            .program
            .inputs()
            .iter()
            .position(|input| input.name == name)
            .ok_or(ContractError::UnknownReference)?;
        let mut bindings = self.bindings.clone();
        bindings[i] = binding;
        Self::new(self.id, Arc::clone(&self.program), bindings)
    }

    /// A content fingerprint of the program and bindings (not the
    /// identity).
    #[must_use]
    pub fn fingerprint(&self) -> u64 {
        let mut w = alloc::vec![self.program.fingerprint()];
        for b in &self.bindings {
            b.words(&mut w);
        }
        hash(0x0062_696e_6469_6e67, &w) // "binding"
    }

    /// Resolves attribute bindings against `set`'s schema.
    ///
    /// # Errors
    ///
    /// [`ContractError::UnknownAttribute`] or
    /// [`ContractError::TypeMismatch`].
    pub(crate) fn prepare<'a>(
        &'a self,
        set: &'a ElementSet,
    ) -> Result<Prepared<'a>, ContractError> {
        let mut columns = Vec::with_capacity(self.bindings.len());
        for (input, binding) in self.program.inputs().iter().zip(&self.bindings) {
            columns.push(match binding {
                Binding::Attribute(name) => {
                    let decl = set
                        .schema()
                        .iter()
                        .find(|a| &a.name == name)
                        .ok_or_else(|| ContractError::UnknownAttribute(name.clone()))?;
                    if Shape::of(decl.port) != Shape::of(input.port) {
                        return Err(ContractError::TypeMismatch(input.name.clone()));
                    }
                    set.attribute(name)
                }
                Binding::RegionRandom(_)
                | Binding::RegionArea
                | Binding::RegionCentroid
                | Binding::RegionOrientation
                | Binding::Position => {
                    return Err(ContractError::UnsupportedBinding(input.name.clone()));
                }
                _ => None,
            });
        }
        Ok(Prepared {
            instance: self,
            set,
            columns,
        })
    }
}

/// An instance resolved against one element set.
pub(crate) struct Prepared<'a> {
    instance: &'a ProgramInstance,
    set: &'a ElementSet,
    columns: Vec<Option<&'a [Value]>>,
}

/// What the compositor knows about one sample of one element.
#[derive(Copy, Clone, Debug)]
pub(crate) struct SampleContext {
    pub(crate) local: Vec2,
    pub(crate) edge: f32,
    pub(crate) footprint: Footprint,
}

impl Prepared<'_> {
    /// Evaluates every node of material or element scope for element `i`;
    /// sample-scope nodes are left empty.
    pub(crate) fn element(&self, i: usize) -> Vec<Option<Value>> {
        let p = &self.instance.program;
        let mut values = alloc::vec![None; p.nodes().len()];
        p.evaluate(
            &mut values,
            |s| s.within(Scope::Element),
            &mut |input| self.input(input, i, None),
            Footprint::POINT,
        );
        values
    }

    /// Evaluates the outputs at one sample of element `i`, reusing
    /// `element`'s hoisted values.
    pub(crate) fn sample(
        &self,
        i: usize,
        element: &[Option<Value>],
        context: SampleContext,
        out: &mut Vec<Value>,
    ) {
        let p = &self.instance.program;
        let mut values = element.to_vec();
        p.evaluate(
            &mut values,
            |_| true,
            &mut |input| self.input(input, i, Some(context)),
            context.footprint,
        );
        out.clear();
        out.extend(
            p.outputs()
                .iter()
                .map(|o| values[o.node.index()].expect("every node evaluated")),
        );
    }

    fn input(&self, i: u32, element: usize, context: Option<SampleContext>) -> Value {
        let i = i as usize;
        match &self.instance.bindings[i] {
            Binding::Constant(value) => *value,
            Binding::Attribute(_) => self.columns[i].expect("prepared")[element],
            Binding::ElementRandom(stream) => {
                Value::Scalar(key_of(self.set, element).unit(*stream))
            }
            Binding::HalfSize => Value::Vector2(self.set.half_size(element)),
            #[expect(clippy::cast_precision_loss, reason = "variant indices are small")]
            Binding::Variant => Value::Scalar(self.set.variant(element) as f32),
            Binding::LocalPosition => Value::Vector2(context.expect("sample scope").local),
            Binding::EdgeDistance => Value::Scalar(context.expect("sample scope").edge),
            Binding::RegionRandom(_)
            | Binding::RegionArea
            | Binding::RegionCentroid
            | Binding::RegionOrientation
            | Binding::Position => unreachable!("refused when prepared"),
        }
    }
}

fn key_of(set: &ElementSet, i: usize) -> ElementKey {
    set.keys()[i]
}

/// Converts a composited sum back to the output's value, for callers mixing
/// values by weight.
pub(crate) fn weighted(acc: &mut Value, value: Value, weight: f32) {
    *acc = dapple_field::scoped::zip(*acc, value, |x, y| x + y * weight);
}

/// The zero of `value`'s shape.
pub(crate) fn zero_like(value: Value) -> Value {
    match value {
        Value::Scalar(_) => Value::Scalar(0.0),
        Value::Id(_) => Value::Id(0),
        Value::Vector2(_) => Value::Vector2(Vec2::ZERO),
        Value::Vector3(_) => Value::Vector3(Vec3::ZERO),
    }
}
