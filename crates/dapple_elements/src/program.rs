// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The minimal callable-program contract: surface programs that elements
//! invoke.
//!
//! A [`SurfaceProgram`] is an inspectable value, never a closure:
//!
//! - named, typed **inputs**, each with an execution [`Scope`];
//! - named, typed **outputs**, each with the widest scope it may have;
//! - **resources** it depends on: field programs it samples, named;
//! - a **body** of [`Node`]s, in dependency order, readable with
//!   [`SurfaceProgram::nodes`].
//!
//! Every node's scope is the widest of its operands' (a constant is
//! [`Scope::Material`], an input its declared scope, a resource sample its
//! position's). The builder checks the declarations: an output declared
//! per element that depends on the sample position is refused
//! ([`ContractError::ScopeViolation`]), and so is an instance binding a
//! per-sample value to a per-element input. Evaluation relies on it: nodes
//! of material and element scope run once per element, not per texel.
//!
//! A [`ProgramInstance`] binds a program's inputs and carries a stable
//! instance identity ([`InstanceId`]), separate from its content
//! fingerprint, as element keys are separate from element fingerprints.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;

use dapple_field::hash::hash;
use dapple_field::program::ValueProgram;
use dapple_field::{Footprint, PortType, Value};
use glam::{Vec2, Vec3};

use crate::identity::{ElementKey, name_word};
use crate::set::{ElementSet, value_fits, value_words};

/// How often a value is computed.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Scope {
    /// Once per material: parameters and constants.
    Material,
    /// Once per element: attributes and identity-derived values.
    Element,
    /// Once per sample point.
    Sample,
}

/// A reference to a node of the program being built.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct NodeRef(u32);

impl NodeRef {
    /// The node's position in [`SurfaceProgram::nodes`].
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// One operation of a program body.
///
/// Arithmetic is per component; a scalar operand broadcasts over a vector.
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    /// Input `index`, as declared.
    Input(u32),
    /// A constant.
    Constant(Value),
    /// Resource `resource` sampled at the 2D position `at`, with a footprint
    /// of the sample's texel.
    Sample {
        /// The resource's position in [`SurfaceProgram::resources`].
        resource: u32,
        /// Where to sample.
        at: NodeRef,
    },
    /// `a + b`.
    Add(NodeRef, NodeRef),
    /// `a − b`.
    Sub(NodeRef, NodeRef),
    /// `a · b`.
    Mul(NodeRef, NodeRef),
    /// Per-component minimum.
    Min(NodeRef, NodeRef),
    /// Per-component maximum.
    Max(NodeRef, NodeRef),
    /// `a + (b − a) · t`, with a scalar `t`.
    Mix {
        /// At `t = 0`.
        a: NodeRef,
        /// At `t = 1`.
        b: NodeRef,
        /// The weight.
        t: NodeRef,
    },
    /// `x` clamped to `[lo, hi]`.
    Clamp {
        /// The value.
        x: NodeRef,
        /// Lower bound.
        lo: NodeRef,
        /// Upper bound.
        hi: NodeRef,
    },
    /// Hermite step of scalar `x` from `edge0` to `edge1`.
    SmoothStep {
        /// Where the result is 0.
        edge0: NodeRef,
        /// Where the result is 1.
        edge1: NodeRef,
        /// The value.
        x: NodeRef,
    },
    /// `a` where scalar `condition` is at least 0.5, else `b`.
    Select {
        /// The condition.
        condition: NodeRef,
        /// Chosen when the condition holds.
        a: NodeRef,
        /// Chosen otherwise.
        b: NodeRef,
    },
    /// Component `index` of a vector.
    Component {
        /// The vector.
        input: NodeRef,
        /// The component.
        index: u8,
    },
    /// A 2D vector from two scalars.
    Vector2(NodeRef, NodeRef),
    /// A 3D vector (or color) from three scalars.
    Vector3(NodeRef, NodeRef, NodeRef),
    /// The Euclidean length of a vector, or the absolute value of a scalar.
    Length(NodeRef),
}

/// The storage shape of a node's value.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Shape {
    /// One `f32`.
    Scalar,
    /// A `u32` identifier.
    Id,
    /// Two `f32`.
    Vector2,
    /// Three `f32`.
    Vector3,
}

impl Shape {
    /// The shape values of `port` have.
    #[must_use]
    pub const fn of(port: PortType) -> Self {
        match port {
            PortType::Scalar | PortType::Mask => Self::Scalar,
            PortType::Id => Self::Id,
            PortType::Vector2 | PortType::Direction => Self::Vector2,
            PortType::Vector3 | PortType::Color(_) | PortType::Normal(_) => Self::Vector3,
        }
    }

    const fn of_value(value: Value) -> Self {
        match value {
            Value::Scalar(_) => Self::Scalar,
            Value::Id(_) => Self::Id,
            Value::Vector2(_) => Self::Vector2,
            Value::Vector3(_) => Self::Vector3,
        }
    }
}

/// A declared input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputDecl {
    /// The input's name, unique among inputs.
    pub name: String,
    /// Its type.
    pub port: PortType,
    /// How often it may change: the widest binding it accepts.
    pub scope: Scope,
}

/// A declared output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputDecl {
    /// The output's name, unique among outputs.
    pub name: String,
    /// Its type.
    pub port: PortType,
    /// The widest scope its value may have.
    pub scope: Scope,
    /// The node computing it.
    pub node: NodeRef,
}

/// A declared resource: a field program the body samples.
#[derive(Clone, Debug, PartialEq)]
pub struct ResourceDecl {
    /// The resource's name.
    pub name: String,
    /// The field.
    pub program: ValueProgram,
}

/// A contract violation, found when a program is built or bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContractError {
    /// A node refers to a node, input or resource that does not precede it.
    UnknownReference,
    /// Operand shapes do not fit the operation.
    ShapeMismatch {
        /// The node, by position.
        node: usize,
    },
    /// A value has a wider scope than its declaration allows, such as a
    /// per-element output that depends on the sample position.
    ScopeViolation {
        /// The output or input.
        name: String,
        /// The declared scope.
        declared: Scope,
        /// The scope found.
        found: Scope,
    },
    /// A name is used twice among inputs or among outputs.
    DuplicateName(String),
    /// An output's type does not fit its node, or a binding's value does not
    /// fit its input.
    TypeMismatch(String),
    /// An instance binds a different number of inputs than declared.
    BindingCount,
    /// A binding names an attribute the element set does not have.
    UnknownAttribute(String),
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownReference => f.write_str("a node refers to a later or unknown item"),
            Self::ShapeMismatch { node } => write!(f, "node {node}: operand shapes do not fit"),
            Self::ScopeViolation {
                name,
                declared,
                found,
            } => write!(
                f,
                "{name:?} is declared {declared:?} but varies per {found:?}"
            ),
            Self::DuplicateName(name) => write!(f, "{name:?} is declared twice"),
            Self::TypeMismatch(name) => write!(f, "{name:?}: type does not fit"),
            Self::BindingCount => f.write_str("an instance must bind every input once"),
            Self::UnknownAttribute(name) => write!(f, "no element attribute {name:?}"),
        }
    }
}

impl core::error::Error for ContractError {}

/// A surface program: see the [module docs](self).
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceProgram {
    name: String,
    inputs: Vec<InputDecl>,
    outputs: Vec<OutputDecl>,
    resources: Vec<ResourceDecl>,
    nodes: Vec<Node>,
    shapes: Vec<Shape>,
    scopes: Vec<Scope>,
}

impl SurfaceProgram {
    /// The program's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Declared inputs, in binding order.
    #[must_use]
    pub fn inputs(&self) -> &[InputDecl] {
        &self.inputs
    }

    /// Declared outputs.
    #[must_use]
    pub fn outputs(&self) -> &[OutputDecl] {
        &self.outputs
    }

    /// Declared resources.
    #[must_use]
    pub fn resources(&self) -> &[ResourceDecl] {
        &self.resources
    }

    /// The body, in dependency order.
    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// The scope of node `node`.
    #[must_use]
    pub fn scope(&self, node: NodeRef) -> Scope {
        self.scopes[node.index()]
    }

    /// The position of the output named `name`.
    #[must_use]
    pub fn output_index(&self, name: &str) -> Option<usize> {
        self.outputs.iter().position(|o| o.name == name)
    }

    /// A content fingerprint of the whole program: declarations, resource
    /// fingerprints and body.
    #[must_use]
    pub fn fingerprint(&self) -> u64 {
        let mut w = Vec::new();
        let port = dapple_raster::typed::port_word;
        w.push(name_word(1, &self.name));
        for i in &self.inputs {
            w.extend([name_word(2, &i.name), port(i.port), i.scope as u64]);
        }
        for o in &self.outputs {
            w.extend([
                name_word(3, &o.name),
                port(o.port),
                o.scope as u64,
                o.node.0.into(),
            ]);
        }
        for r in &self.resources {
            let fp = r.program.fingerprint().0;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "splitting the 128-bit fingerprint into its halves"
            )]
            w.extend([name_word(4, &r.name), fp as u64, (fp >> 64) as u64]);
        }
        for node in &self.nodes {
            node_words(node, &mut w);
        }
        hash(0x0073_7572_6661_6365, &w) // "surface"
    }
}

fn node_words(node: &Node, w: &mut Vec<u64>) {
    let r = |n: &NodeRef| u64::from(n.0);
    match node {
        Node::Input(i) => w.extend([1, u64::from(*i)]),
        Node::Constant(v) => {
            w.push(2);
            value_words(*v, w);
        }
        Node::Sample { resource, at } => w.extend([3, u64::from(*resource), r(at)]),
        Node::Add(a, b) => w.extend([4, r(a), r(b)]),
        Node::Sub(a, b) => w.extend([5, r(a), r(b)]),
        Node::Mul(a, b) => w.extend([6, r(a), r(b)]),
        Node::Min(a, b) => w.extend([7, r(a), r(b)]),
        Node::Max(a, b) => w.extend([8, r(a), r(b)]),
        Node::Mix { a, b, t } => w.extend([9, r(a), r(b), r(t)]),
        Node::Clamp { x, lo, hi } => w.extend([10, r(x), r(lo), r(hi)]),
        Node::SmoothStep { edge0, edge1, x } => w.extend([11, r(edge0), r(edge1), r(x)]),
        Node::Select { condition, a, b } => w.extend([12, r(condition), r(a), r(b)]),
        Node::Component { input, index } => w.extend([13, r(input), u64::from(*index)]),
        Node::Vector2(a, b) => w.extend([14, r(a), r(b)]),
        Node::Vector3(a, b, c) => w.extend([15, r(a), r(b), r(c)]),
        Node::Length(a) => w.extend([16, r(a)]),
    }
}

/// Builds a [`SurfaceProgram`], checking shapes as nodes are added and
/// scopes as outputs are declared.
#[derive(Clone, Debug)]
pub struct SurfaceBuilder {
    program: SurfaceProgram,
}

impl SurfaceBuilder {
    /// Starts a program named `name`.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            program: SurfaceProgram {
                name: name.into(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                resources: Vec::new(),
                nodes: Vec::new(),
                shapes: Vec::new(),
                scopes: Vec::new(),
            },
        }
    }

    /// Declares an input and returns the node reading it.
    ///
    /// # Errors
    ///
    /// [`ContractError::DuplicateName`].
    pub fn input(
        &mut self,
        name: &str,
        port: PortType,
        scope: Scope,
    ) -> Result<NodeRef, ContractError> {
        let p = &mut self.program;
        if p.inputs.iter().any(|i| i.name == name) {
            return Err(ContractError::DuplicateName(name.into()));
        }
        let index = u32::try_from(p.inputs.len()).map_err(|_| ContractError::UnknownReference)?;
        p.inputs.push(InputDecl {
            name: name.into(),
            port,
            scope,
        });
        self.push(Node::Input(index), Shape::of(port), scope)
    }

    /// Declares a field resource and returns its index for
    /// [`Node::Sample`].
    pub fn resource(&mut self, name: &str, program: ValueProgram) -> u32 {
        self.program.resources.push(ResourceDecl {
            name: name.into(),
            program,
        });
        u32::try_from(self.program.resources.len() - 1).expect("fewer than 2^32 resources")
    }

    /// A constant node.
    pub fn constant(&mut self, value: Value) -> NodeRef {
        self.push(
            Node::Constant(value),
            Shape::of_value(value),
            Scope::Material,
        )
        .expect("constants always fit")
    }

    /// Adds `node`.
    ///
    /// # Errors
    ///
    /// [`ContractError::UnknownReference`] or
    /// [`ContractError::ShapeMismatch`].
    pub fn add(&mut self, node: Node) -> Result<NodeRef, ContractError> {
        let at = self.program.nodes.len();
        let get = |n: &NodeRef| -> Result<(Shape, Scope), ContractError> {
            let i = n.index();
            if i < at {
                Ok((self.program.shapes[i], self.program.scopes[i]))
            } else {
                Err(ContractError::UnknownReference)
            }
        };
        let bad = ContractError::ShapeMismatch { node: at };
        let arith = |a: Shape, b: Shape| -> Result<Shape, ContractError> {
            match (a, b) {
                (Shape::Id, _) | (_, Shape::Id) => Err(bad.clone()),
                (x, y) if x == y => Ok(x),
                (Shape::Scalar, y) => Ok(y),
                (x, Shape::Scalar) => Ok(x),
                _ => Err(bad.clone()),
            }
        };
        let (shape, scope) = match &node {
            Node::Input(_) | Node::Constant(_) => return Err(ContractError::UnknownReference),
            Node::Sample { resource, at } => {
                let (s, scope) = get(at)?;
                let r = self
                    .program
                    .resources
                    .get(*resource as usize)
                    .ok_or(ContractError::UnknownReference)?;
                if s != Shape::Vector2 {
                    return Err(bad);
                }
                (Shape::of(r.program.output_type()), scope)
            }
            Node::Add(a, b)
            | Node::Sub(a, b)
            | Node::Mul(a, b)
            | Node::Min(a, b)
            | Node::Max(a, b) => {
                let ((sa, ca), (sb, cb)) = (get(a)?, get(b)?);
                (arith(sa, sb)?, ca.max(cb))
            }
            Node::Mix { a, b, t } => {
                let ((sa, ca), (sb, cb), (st, ct)) = (get(a)?, get(b)?, get(t)?);
                if st != Shape::Scalar || sa != sb || sa == Shape::Id {
                    return Err(bad);
                }
                (sa, ca.max(cb).max(ct))
            }
            Node::Clamp { x, lo, hi } => {
                let ((sx, cx), (sl, cl), (sh, ch)) = (get(x)?, get(lo)?, get(hi)?);
                arith(sx, sl)?;
                arith(sx, sh)?;
                if sl != Shape::Scalar && sl != sx || sh != Shape::Scalar && sh != sx {
                    return Err(bad);
                }
                (sx, cx.max(cl).max(ch))
            }
            Node::SmoothStep { edge0, edge1, x } => {
                let ((s0, c0), (s1, c1), (sx, cx)) = (get(edge0)?, get(edge1)?, get(x)?);
                if [s0, s1, sx] != [Shape::Scalar; 3] {
                    return Err(bad);
                }
                (Shape::Scalar, c0.max(c1).max(cx))
            }
            Node::Select { condition, a, b } => {
                let ((sc, cc), (sa, ca), (sb, cb)) = (get(condition)?, get(a)?, get(b)?);
                if sc != Shape::Scalar || sa != sb {
                    return Err(bad);
                }
                (sa, cc.max(ca).max(cb))
            }
            Node::Component { input, index } => {
                let (s, c) = get(input)?;
                let ok = match s {
                    Shape::Vector2 => *index < 2,
                    Shape::Vector3 => *index < 3,
                    _ => false,
                };
                if !ok {
                    return Err(bad);
                }
                (Shape::Scalar, c)
            }
            Node::Vector2(a, b) => {
                let ((sa, ca), (sb, cb)) = (get(a)?, get(b)?);
                if [sa, sb] != [Shape::Scalar; 2] {
                    return Err(bad);
                }
                (Shape::Vector2, ca.max(cb))
            }
            Node::Vector3(a, b, c) => {
                let ((sa, ca), (sb, cb), (sc, cc)) = (get(a)?, get(b)?, get(c)?);
                if [sa, sb, sc] != [Shape::Scalar; 3] {
                    return Err(bad);
                }
                (Shape::Vector3, ca.max(cb).max(cc))
            }
            Node::Length(a) => {
                let (s, c) = get(a)?;
                if s == Shape::Id {
                    return Err(bad);
                }
                (Shape::Scalar, c)
            }
        };
        self.push(node, shape, scope)
    }

    fn push(&mut self, node: Node, shape: Shape, scope: Scope) -> Result<NodeRef, ContractError> {
        let p = &mut self.program;
        let r = NodeRef(u32::try_from(p.nodes.len()).map_err(|_| ContractError::UnknownReference)?);
        p.nodes.push(node);
        p.shapes.push(shape);
        p.scopes.push(scope);
        Ok(r)
    }

    /// Declares output `name` of type `port`, computed by `node`, whose
    /// value may vary at most per `scope`.
    ///
    /// # Errors
    ///
    /// [`ContractError::DuplicateName`], [`ContractError::UnknownReference`],
    /// [`ContractError::TypeMismatch`], or
    /// [`ContractError::ScopeViolation`] when `node` varies more often than
    /// `scope`.
    pub fn output(
        &mut self,
        name: &str,
        port: PortType,
        scope: Scope,
        node: NodeRef,
    ) -> Result<(), ContractError> {
        let p = &mut self.program;
        if p.outputs.iter().any(|o| o.name == name) {
            return Err(ContractError::DuplicateName(name.into()));
        }
        let i = node.index();
        if i >= p.nodes.len() {
            return Err(ContractError::UnknownReference);
        }
        if p.shapes[i] != Shape::of(port) {
            return Err(ContractError::TypeMismatch(name.into()));
        }
        if p.scopes[i] > scope {
            return Err(ContractError::ScopeViolation {
                name: name.into(),
                declared: scope,
                found: p.scopes[i],
            });
        }
        p.outputs.push(OutputDecl {
            name: name.into(),
            port,
            scope,
            node,
        });
        Ok(())
    }

    /// The finished program.
    #[must_use]
    pub fn finish(self) -> SurfaceProgram {
        self.program
    }
}

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
            Self::LocalPosition | Self::EdgeDistance => Scope::Sample,
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
        }
    }
}

/// A program bound for use: its identity, the program, and one binding per
/// input.
#[derive(Clone, Debug, PartialEq)]
pub struct ProgramInstance {
    id: InstanceId,
    program: Arc<SurfaceProgram>,
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
        program: Arc<SurfaceProgram>,
        bindings: Vec<Binding>,
    ) -> Result<Self, ContractError> {
        if bindings.len() != program.inputs.len() {
            return Err(ContractError::BindingCount);
        }
        for (input, binding) in program.inputs.iter().zip(&bindings) {
            if binding.scope() > input.scope {
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
                Binding::ElementRandom(_) | Binding::EdgeDistance | Binding::Variant => {
                    Shape::Scalar
                }
                Binding::HalfSize | Binding::LocalPosition => Shape::Vector2,
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
    pub fn program(&self) -> &SurfaceProgram {
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
            .inputs
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
        for (input, binding) in self.program.inputs.iter().zip(&self.bindings) {
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
        let mut values = alloc::vec![None; p.nodes.len()];
        for n in 0..p.nodes.len() {
            if p.scopes[n] <= Scope::Element {
                values[n] = Some(self.eval(n, i, &values, None));
            }
        }
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
        for n in 0..p.nodes.len() {
            if values[n].is_none() {
                values[n] = Some(self.eval(n, i, &values, Some(context)));
            }
        }
        out.clear();
        out.extend(
            p.outputs
                .iter()
                .map(|o| values[o.node.index()].expect("every node evaluated")),
        );
    }

    fn eval(
        &self,
        n: usize,
        element: usize,
        values: &[Option<Value>],
        context: Option<SampleContext>,
    ) -> Value {
        let p = &self.instance.program;
        let v = |r: &NodeRef| values[r.index()].expect("operands precede their nodes");
        match &p.nodes[n] {
            Node::Input(i) => {
                let i = *i as usize;
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
                }
            }
            Node::Constant(value) => *value,
            Node::Sample { resource, at } => {
                let Value::Vector2(at) = v(at) else {
                    unreachable!("checked: sample positions are 2D")
                };
                let footprint = context.map_or(Footprint::POINT, |c| c.footprint);
                p.resources[*resource as usize].program.eval(at, footprint)
            }
            Node::Add(a, b) => zip(v(a), v(b), |x, y| x + y),
            Node::Sub(a, b) => zip(v(a), v(b), |x, y| x - y),
            Node::Mul(a, b) => zip(v(a), v(b), |x, y| x * y),
            Node::Min(a, b) => zip(v(a), v(b), f32::min),
            Node::Max(a, b) => zip(v(a), v(b), f32::max),
            Node::Mix { a, b, t } => {
                let t = scalar(v(t));
                zip(v(a), v(b), |x, y| x + (y - x) * t)
            }
            Node::Clamp { x, lo, hi } => zip(zip(v(x), v(lo), f32::max), v(hi), f32::min),
            Node::SmoothStep { edge0, edge1, x } => {
                let (e0, e1, x) = (scalar(v(edge0)), scalar(v(edge1)), scalar(v(x)));
                let t = if e1 == e0 {
                    if x < e0 { 0.0 } else { 1.0 }
                } else {
                    ((x - e0) / (e1 - e0)).clamp(0.0, 1.0)
                };
                Value::Scalar(t * t * (3.0 - 2.0 * t))
            }
            Node::Select { condition, a, b } => {
                if scalar(v(condition)) >= 0.5 {
                    v(a)
                } else {
                    v(b)
                }
            }
            Node::Component { input, index } => {
                Value::Scalar(v(input).component(*index as usize).unwrap_or(0.0))
            }
            Node::Vector2(a, b) => Value::Vector2(Vec2::new(scalar(v(a)), scalar(v(b)))),
            Node::Vector3(a, b, c) => {
                Value::Vector3(Vec3::new(scalar(v(a)), scalar(v(b)), scalar(v(c))))
            }
            Node::Length(a) => Value::Scalar(match v(a) {
                Value::Scalar(x) => x.abs(),
                Value::Vector2(x) => x.length(),
                Value::Vector3(x) => x.length(),
                Value::Id(_) => 0.0,
            }),
        }
    }
}

fn key_of(set: &ElementSet, i: usize) -> ElementKey {
    set.keys()[i]
}

fn scalar(v: Value) -> f32 {
    v.component(0).unwrap_or(0.0)
}

/// Applies `f` per component, broadcasting a scalar operand.
fn zip(a: Value, b: Value, f: impl Fn(f32, f32) -> f32) -> Value {
    match (a, b) {
        (Value::Scalar(x), Value::Scalar(y)) => Value::Scalar(f(x, y)),
        (Value::Vector2(x), Value::Vector2(y)) => {
            Value::Vector2(Vec2::new(f(x.x, y.x), f(x.y, y.y)))
        }
        (Value::Vector3(x), Value::Vector3(y)) => {
            Value::Vector3(Vec3::new(f(x.x, y.x), f(x.y, y.y), f(x.z, y.z)))
        }
        (Value::Scalar(x), Value::Vector2(y)) => Value::Vector2(Vec2::new(f(x, y.x), f(x, y.y))),
        (Value::Vector2(x), Value::Scalar(y)) => Value::Vector2(Vec2::new(f(x.x, y), f(x.y, y))),
        (Value::Scalar(x), Value::Vector3(y)) => {
            Value::Vector3(Vec3::new(f(x, y.x), f(x, y.y), f(x, y.z)))
        }
        (Value::Vector3(x), Value::Scalar(y)) => {
            Value::Vector3(Vec3::new(f(x.x, y), f(x.y, y), f(x.z, y)))
        }
        (a, _) => a,
    }
}

/// Converts a composited sum back to the output's value, for callers mixing
/// values by weight.
pub(crate) fn weighted(acc: &mut Value, value: Value, weight: f32) {
    *acc = zip(*acc, value, |x, y| x + y * weight);
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
