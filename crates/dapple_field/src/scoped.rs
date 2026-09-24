// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Scoped programs: dapple's programmable core.
//!
//! A [`ScopedProgram`] is an inspectable expression value, never a closure:
//!
//! - named, typed **inputs**, each with an execution [`Scope`]: how often
//!   its value may change;
//! - named, typed **outputs**, each with the widest scope it may have;
//! - **resources** it samples: field programs ([`ValueProgram`]), named;
//! - **functions** it calls: other scoped programs, named, so a shape or
//!   profile is written once and reused;
//! - a **body** of [`Node`]s in dependency order: arithmetic, comparisons
//!   and selection, vector operations, tone curves and color ramps
//!   ([`crate::shaping`]), resource samples and calls. Coordinates are
//!   inputs, bound by whoever evaluates the program (an element's local
//!   position, a material's domain position).
//!
//! **Scopes.** Every node's scope is the join of its operands' ([`Scope::join`]):
//! a constant is [`Scope::Material`], an input has its declared scope, a
//! resource sample has its position's. Declaring an output whose node varies
//! more often than the output allows is refused
//! ([`ContractError::ScopeViolation`]), and binders check that a binding
//! does not vary more often than its input. Evaluators rely on it: nodes of
//! material, region or element scope run once per material, region or
//! element, not once per sample, and outputs declared per sample never
//! depend on a raster pass, so they stay point-evaluable (and translatable
//! to shaders).
//!
//! Binding inputs to values is the evaluator's business: element compositing
//! and region evaluation (`dapple_elements`) and material maps
//! (`dapple_material`) each supply their own bindings over this core.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use glam::{Vec2, Vec3};

use crate::hash::hash;
use crate::program::ValueProgram;
use crate::shaping::{ColorRamp, ToneCurve};
use crate::{Footprint, PortType, Value};

/// How often a value is computed: an execution frequency.
///
/// Scopes form a lattice, not a chain: a value of [`Scope::Region`] and a
/// value of [`Scope::Element`] are independent (an element need not lie in
/// one region), so a value depending on both varies per
/// [`Scope::Sample`]. [`Scope::Pass`] is the widest: a value that varies per
/// sample *and* needs a raster pass (a neighborhood, a reduction) to exist,
/// such as ambient occlusion of a height.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Scope {
    /// Once per material: parameters and constants.
    Material,
    /// Once per region of a region map.
    Region,
    /// Once per element: attributes and identity-derived values.
    Element,
    /// Once per sample point, from the point alone.
    Sample,
    /// Once per sample point, after a raster pass.
    Pass,
}

impl Scope {
    /// Whether values of this scope may stand where `other` is declared:
    /// whether this scope varies no more often than `other`.
    #[must_use]
    pub const fn within(self, other: Self) -> bool {
        matches!(
            (self, other),
            (Self::Material, _)
                | (_, Self::Pass)
                | (Self::Region, Self::Region | Self::Sample)
                | (Self::Element, Self::Element | Self::Sample)
                | (Self::Sample, Self::Sample)
        )
    }

    /// The narrowest scope both `self` and `other` are within: the scope of
    /// a value computed from both.
    #[must_use]
    pub const fn join(self, other: Self) -> Self {
        if self.within(other) {
            other
        } else if other.within(self) {
            self
        } else {
            Self::Sample
        }
    }

    const fn word(self) -> u64 {
        match self {
            Self::Material => 0,
            Self::Element => 1,
            Self::Sample => 2,
            Self::Region => 3,
            Self::Pass => 4,
        }
    }
}

/// A reference to a node of the program being built.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct NodeRef(u32);

impl NodeRef {
    /// The node's position in [`ScopedProgram::nodes`].
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// A one-operand operation, per component.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum UnaryOp {
    /// `−x`.
    Negate,
    /// `|x|`.
    Abs,
    /// The largest integer at most `x`.
    Floor,
    /// `x − floor(x)`.
    Fract,
    /// `√x`, 0 for negative `x`.
    Sqrt,
    /// `eˣ`.
    Exp,
    /// `ln x`, for positive `x`.
    Ln,
    /// `sin x`, in radians.
    Sin,
    /// `cos x`, in radians.
    Cos,
    /// `x` clamped to `[0, 1]`.
    Saturate,
}

impl UnaryOp {
    fn apply(self, x: f32) -> f32 {
        match self {
            Self::Negate => -x,
            Self::Abs => x.abs(),
            Self::Floor => libm::floorf(x),
            Self::Fract => x - libm::floorf(x),
            Self::Sqrt => libm::sqrtf(x.max(0.0)),
            Self::Exp => libm::expf(x),
            Self::Ln => libm::logf(x),
            Self::Sin => libm::sinf(x),
            Self::Cos => libm::cosf(x),
            Self::Saturate => x.clamp(0.0, 1.0),
        }
    }
}

/// A comparison, 1 where it holds and 0 elsewhere.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum CompareOp {
    /// `a < b`.
    Less,
    /// `a ≤ b`.
    LessEqual,
    /// `a > b`.
    Greater,
    /// `a ≥ b`.
    GreaterEqual,
    /// `a = b`; also for identifiers.
    Equal,
    /// `a ≠ b`; also for identifiers.
    NotEqual,
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
    /// Resource `resource` sampled at the 2D position `at`, with the
    /// evaluation's footprint.
    Sample {
        /// The resource's position in [`ScopedProgram::resources`].
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
    /// `a / b`.
    Div(NodeRef, NodeRef),
    /// `aᵇ` per component, for nonnegative `a`.
    Pow(NodeRef, NodeRef),
    /// A one-operand operation.
    Unary(UnaryOp, NodeRef),
    /// A comparison of two scalars (or identifiers, for equality), 1 or 0.
    Compare(CompareOp, NodeRef, NodeRef),
    /// The dot product of two vectors of one shape.
    Dot(NodeRef, NodeRef),
    /// The cross product of two 3D vectors.
    Cross(NodeRef, NodeRef),
    /// A vector scaled to unit length; the zero vector stays zero.
    Normalize(NodeRef),
    /// A tone curve applied to a scalar.
    Curve(NodeRef, ToneCurve),
    /// A color ramp read at a scalar.
    Ramp(NodeRef, ColorRamp),
    /// Output `output` of function `function`, called with `args` bound to
    /// its inputs in order.
    Call {
        /// The function's position in [`ScopedProgram::functions`].
        function: u32,
        /// Which of its outputs.
        output: u32,
        /// One argument per input.
        args: Vec<NodeRef>,
    },
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

    /// The shape of `value`.
    #[must_use]
    pub const fn of_value(value: Value) -> Self {
        match value {
            Value::Scalar(_) => Self::Scalar,
            Value::Id(_) => Self::Id,
            Value::Vector2(_) => Self::Vector2,
            Value::Vector3(_) => Self::Vector3,
        }
    }
}

/// Whether `value` can stand for a value of type `port`.
#[must_use]
pub fn value_fits(value: Value, port: PortType) -> bool {
    Shape::of_value(value) == Shape::of(port)
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

/// A declared function: another program the body calls.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionDecl {
    /// The function's name.
    pub name: String,
    /// Its program.
    pub program: Arc<ScopedProgram>,
}

/// A contract violation, found when a program is built or bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContractError {
    /// A node refers to a node, input, resource or function that does not
    /// precede it.
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
    /// A binder binds a different number of inputs than declared.
    BindingCount,
    /// A binding names an attribute or channel its evaluator does not have.
    UnknownAttribute(String),
    /// A binding cannot be supplied by this evaluator, such as a region
    /// property when compositing elements.
    UnsupportedBinding(String),
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
            Self::BindingCount => f.write_str("every input must be bound once"),
            Self::UnknownAttribute(name) => write!(f, "nothing named {name:?} to bind"),
            Self::UnsupportedBinding(name) => {
                write!(f, "{name:?}: this evaluator cannot supply its binding")
            }
        }
    }
}

impl core::error::Error for ContractError {}

/// A scoped program: see the [module docs](self).
#[derive(Clone, Debug, PartialEq)]
pub struct ScopedProgram {
    name: String,
    inputs: Vec<InputDecl>,
    outputs: Vec<OutputDecl>,
    resources: Vec<ResourceDecl>,
    functions: Vec<FunctionDecl>,
    nodes: Vec<Node>,
    shapes: Vec<Shape>,
    scopes: Vec<Scope>,
}

impl ScopedProgram {
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

    /// Declared functions.
    #[must_use]
    pub fn functions(&self) -> &[FunctionDecl] {
        &self.functions
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

    /// The shape of node `node`.
    #[must_use]
    pub fn shape(&self, node: NodeRef) -> Shape {
        self.shapes[node.index()]
    }

    /// The position of the input named `name`.
    #[must_use]
    pub fn input_index(&self, name: &str) -> Option<usize> {
        self.inputs.iter().position(|i| i.name == name)
    }

    /// The position of the output named `name`.
    #[must_use]
    pub fn output_index(&self, name: &str) -> Option<usize> {
        self.outputs.iter().position(|o| o.name == name)
    }

    /// A content fingerprint of the whole program: declarations, resource
    /// and function fingerprints, and body.
    #[must_use]
    pub fn fingerprint(&self) -> u64 {
        let mut w = Vec::new();
        let port = PortType::word;
        w.push(name_word(1, &self.name));
        for i in &self.inputs {
            w.extend([name_word(2, &i.name), port(i.port), i.scope.word()]);
        }
        for o in &self.outputs {
            w.extend([
                name_word(3, &o.name),
                port(o.port),
                o.scope.word(),
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
        for f in &self.functions {
            w.extend([name_word(6, &f.name), f.program.fingerprint()]);
        }
        for node in &self.nodes {
            node_words(node, &mut w);
        }
        hash(0x0073_7572_6661_6365, &w) // "surface"
    }

    /// Evaluates every node not yet in `values` whose scope `include`
    /// accepts, reading input `i` from `inputs(i)` and sampling resources
    /// with `footprint`.
    ///
    /// `values` holds one slot per node; slots already filled are kept, so
    /// an evaluator hoists work by evaluating the narrow scopes once and
    /// copying their slots into each wider evaluation. Operands of an
    /// included node must be included or already filled.
    ///
    /// # Panics
    ///
    /// When `values` is not one slot per node, or an operand is missing.
    pub fn evaluate(
        &self,
        values: &mut [Option<Value>],
        include: impl Fn(Scope) -> bool,
        inputs: &mut dyn FnMut(u32) -> Value,
        footprint: Footprint,
    ) {
        assert_eq!(values.len(), self.nodes.len(), "one slot per node");
        for n in 0..self.nodes.len() {
            if values[n].is_none() && include(self.scopes[n]) {
                values[n] = Some(self.eval_node(n, values, inputs, footprint));
            }
        }
    }

    /// Evaluates the whole program with `inputs` in declaration order and
    /// returns its outputs in declaration order.
    ///
    /// # Panics
    ///
    /// When `inputs` has the wrong length.
    #[must_use]
    pub fn eval(&self, inputs: &[Value], footprint: Footprint) -> Vec<Value> {
        assert_eq!(inputs.len(), self.inputs.len(), "one value per input");
        let mut values = vec![None; self.nodes.len()];
        self.evaluate(
            &mut values,
            |_| true,
            &mut |i| inputs[i as usize],
            footprint,
        );
        self.outputs
            .iter()
            .map(|o| values[o.node.index()].expect("every node evaluated"))
            .collect()
    }

    fn eval_node(
        &self,
        n: usize,
        values: &[Option<Value>],
        inputs: &mut dyn FnMut(u32) -> Value,
        footprint: Footprint,
    ) -> Value {
        let v = |r: &NodeRef| values[r.index()].expect("operands precede their nodes");
        match &self.nodes[n] {
            Node::Input(i) => inputs(*i),
            Node::Constant(value) => *value,
            Node::Sample { resource, at } => {
                let Value::Vector2(at) = v(at) else {
                    unreachable!("checked: sample positions are 2D")
                };
                self.resources[*resource as usize]
                    .program
                    .eval(at, footprint)
            }
            Node::Add(a, b) => zip(v(a), v(b), |x, y| x + y),
            Node::Sub(a, b) => zip(v(a), v(b), |x, y| x - y),
            Node::Mul(a, b) => zip(v(a), v(b), |x, y| x * y),
            Node::Div(a, b) => zip(v(a), v(b), |x, y| x / y),
            Node::Min(a, b) => zip(v(a), v(b), f32::min),
            Node::Max(a, b) => zip(v(a), v(b), f32::max),
            Node::Pow(a, b) => zip(v(a), v(b), |x, y| libm::powf(x.max(0.0), y)),
            Node::Unary(op, a) => map(v(a), |x| op.apply(x)),
            Node::Compare(op, a, b) => {
                let (a, b) = (v(a), v(b));
                let holds = match (a, b) {
                    (Value::Id(a), Value::Id(b)) => match op {
                        CompareOp::Equal => a == b,
                        CompareOp::NotEqual => a != b,
                        _ => unreachable!("checked: identifiers only compare for equality"),
                    },
                    _ => {
                        let (a, b) = (scalar(a), scalar(b));
                        match op {
                            CompareOp::Less => a < b,
                            CompareOp::LessEqual => a <= b,
                            CompareOp::Greater => a > b,
                            CompareOp::GreaterEqual => a >= b,
                            CompareOp::Equal => a == b,
                            CompareOp::NotEqual => a != b,
                        }
                    }
                };
                Value::Scalar(if holds { 1.0 } else { 0.0 })
            }
            Node::Mix { a, b, t } => {
                let t = scalar(v(t));
                zip(v(a), v(b), |x, y| x + (y - x) * t)
            }
            Node::Clamp { x, lo, hi } => zip(zip(v(x), v(lo), f32::max), v(hi), f32::min),
            Node::SmoothStep { edge0, edge1, x } => {
                Value::Scalar(smoothstep(scalar(v(edge0)), scalar(v(edge1)), scalar(v(x))))
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
            Node::Dot(a, b) => Value::Scalar(match (v(a), v(b)) {
                (Value::Vector2(a), Value::Vector2(b)) => a.dot(b),
                (Value::Vector3(a), Value::Vector3(b)) => a.dot(b),
                _ => unreachable!("checked: dot products take two vectors of one shape"),
            }),
            Node::Cross(a, b) => match (v(a), v(b)) {
                (Value::Vector3(a), Value::Vector3(b)) => Value::Vector3(a.cross(b)),
                _ => unreachable!("checked: cross products take 3D vectors"),
            },
            Node::Normalize(a) => match v(a) {
                Value::Vector2(x) => Value::Vector2(x.normalize_or_zero()),
                Value::Vector3(x) => Value::Vector3(x.normalize_or_zero()),
                other => other,
            },
            Node::Curve(x, curve) => Value::Scalar(curve.eval(scalar(v(x)))),
            Node::Ramp(t, ramp) => Value::Vector3(ramp.eval(scalar(v(t)))),
            Node::Call {
                function,
                output,
                args,
            } => {
                let f = &self.functions[*function as usize].program;
                let args: Vec<Value> = args.iter().map(v).collect();
                let mut slots = vec![None; f.nodes.len()];
                f.evaluate(&mut slots, |_| true, &mut |i| args[i as usize], footprint);
                slots[f.outputs[*output as usize].node.index()].expect("every node evaluated")
            }
        }
    }

    /// The scopes of every node when the inputs have scopes `inputs`.
    fn scopes_with(&self, inputs: &[Scope]) -> Vec<Scope> {
        let mut scopes: Vec<Scope> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let s = |r: &NodeRef| scopes[r.index()];
            let scope = match node {
                Node::Input(i) => inputs[*i as usize],
                Node::Constant(_) => Scope::Material,
                Node::Call { function, args, .. } => {
                    let f = &self.functions[*function as usize].program;
                    let arg_scopes: Vec<Scope> = args.iter().map(s).collect();
                    // A called function's result varies as its output does
                    // for these arguments.
                    let inner = f.scopes_with(&arg_scopes);
                    f.outputs
                        .iter()
                        .map(|o| inner[o.node.index()])
                        .fold(Scope::Material, Scope::join)
                }
                other => operands(other).fold(Scope::Material, |acc, r| acc.join(s(&r))),
            };
            scopes.push(scope);
        }
        scopes
    }
}

/// The operands of a node, in order.
fn operands(node: &Node) -> impl Iterator<Item = NodeRef> + '_ {
    let fixed: [Option<NodeRef>; 3] = match node {
        Node::Input(_) | Node::Constant(_) | Node::Call { .. } => [None; 3],
        Node::Sample { at, .. } => [Some(*at), None, None],
        Node::Add(a, b)
        | Node::Sub(a, b)
        | Node::Mul(a, b)
        | Node::Min(a, b)
        | Node::Max(a, b)
        | Node::Div(a, b)
        | Node::Pow(a, b)
        | Node::Compare(_, a, b)
        | Node::Dot(a, b)
        | Node::Cross(a, b)
        | Node::Vector2(a, b) => [Some(*a), Some(*b), None],
        Node::Mix { a, b, t } => [Some(*a), Some(*b), Some(*t)],
        Node::Clamp { x, lo, hi } => [Some(*x), Some(*lo), Some(*hi)],
        Node::SmoothStep { edge0, edge1, x } => [Some(*edge0), Some(*edge1), Some(*x)],
        Node::Select { condition, a, b } => [Some(*condition), Some(*a), Some(*b)],
        Node::Vector3(a, b, c) => [Some(*a), Some(*b), Some(*c)],
        Node::Component { input: a, .. }
        | Node::Length(a)
        | Node::Unary(_, a)
        | Node::Normalize(a)
        | Node::Curve(a, _)
        | Node::Ramp(a, _) => [Some(*a), None, None],
    };
    let args: &[NodeRef] = match node {
        Node::Call { args, .. } => args,
        _ => &[],
    };
    fixed.into_iter().flatten().chain(args.iter().copied())
}

/// A word for `name` under `tag`.
#[must_use]
pub fn name_word(tag: u64, name: &str) -> u64 {
    let mut h = hash(tag, &[name.len() as u64]);
    for chunk in name.as_bytes().chunks(8) {
        let mut word = [0_u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        h = hash(h, &[u64::from_le_bytes(word)]);
    }
    h
}

/// Appends a value's words to `w`.
pub fn value_words(value: Value, w: &mut Vec<u64>) {
    match value {
        Value::Scalar(v) => w.extend([1, u64::from(v.to_bits())]),
        Value::Id(v) => w.extend([2, u64::from(v)]),
        Value::Vector2(v) => w.extend([3, u64::from(v.x.to_bits()), u64::from(v.y.to_bits())]),
        Value::Vector3(v) => w.extend([
            4,
            u64::from(v.x.to_bits()),
            u64::from(v.y.to_bits()),
            u64::from(v.z.to_bits()),
        ]),
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
        Node::Div(a, b) => w.extend([17, r(a), r(b)]),
        Node::Pow(a, b) => w.extend([18, r(a), r(b)]),
        Node::Unary(op, a) => w.extend([19, *op as u64, r(a)]),
        Node::Compare(op, a, b) => w.extend([20, *op as u64, r(a), r(b)]),
        Node::Dot(a, b) => w.extend([21, r(a), r(b)]),
        Node::Cross(a, b) => w.extend([22, r(a), r(b)]),
        Node::Normalize(a) => w.extend([23, r(a)]),
        Node::Curve(x, curve) => {
            w.extend([24, r(x)]);
            curve.words(w);
        }
        Node::Ramp(t, ramp) => {
            w.extend([25, r(t)]);
            ramp.words(w);
        }
        Node::Call {
            function,
            output,
            args,
        } => {
            w.extend([
                26,
                u64::from(*function),
                u64::from(*output),
                args.len() as u64,
            ]);
            w.extend(args.iter().map(r));
        }
    }
}

/// Builds a [`ScopedProgram`], checking shapes and scopes as nodes are
/// added and as outputs are declared.
///
/// Besides [`ScopedBuilder::node`], it has shorthands for common nodes;
/// the ones taking `f32` add the constants they need.
#[derive(Clone, Debug)]
pub struct ScopedBuilder {
    program: ScopedProgram,
}

impl ScopedBuilder {
    /// Starts a program named `name`.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            program: ScopedProgram {
                name: name.into(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                resources: Vec::new(),
                functions: Vec::new(),
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

    /// Declares a function and returns its index for [`Node::Call`].
    pub fn function(&mut self, name: &str, program: Arc<ScopedProgram>) -> u32 {
        self.program.functions.push(FunctionDecl {
            name: name.into(),
            program,
        });
        u32::try_from(self.program.functions.len() - 1).expect("fewer than 2^32 functions")
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
    /// [`ContractError::UnknownReference`], [`ContractError::ShapeMismatch`],
    /// or [`ContractError::ScopeViolation`] for a call argument that varies
    /// more often than its parameter allows.
    pub fn node(&mut self, node: Node) -> Result<NodeRef, ContractError> {
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
        let mut scope = Scope::Material;
        for r in operands(&node) {
            scope = scope.join(get(&r)?.1);
        }
        let shape = match &node {
            Node::Input(_) | Node::Constant(_) => return Err(ContractError::UnknownReference),
            Node::Sample { resource, at } => {
                let r = self
                    .program
                    .resources
                    .get(*resource as usize)
                    .ok_or(ContractError::UnknownReference)?;
                if get(at)?.0 != Shape::Vector2 {
                    return Err(bad);
                }
                Shape::of(r.program.output_type())
            }
            Node::Add(a, b)
            | Node::Sub(a, b)
            | Node::Mul(a, b)
            | Node::Div(a, b)
            | Node::Pow(a, b)
            | Node::Min(a, b)
            | Node::Max(a, b) => arith(get(a)?.0, get(b)?.0)?,
            Node::Unary(_, a) => match get(a)?.0 {
                Shape::Id => return Err(bad),
                s => s,
            },
            Node::Compare(op, a, b) => match (get(a)?.0, get(b)?.0) {
                (Shape::Scalar, Shape::Scalar) => Shape::Scalar,
                (Shape::Id, Shape::Id) if matches!(op, CompareOp::Equal | CompareOp::NotEqual) => {
                    Shape::Scalar
                }
                _ => return Err(bad),
            },
            Node::Mix { a, b, t } => {
                let (sa, sb, st) = (get(a)?.0, get(b)?.0, get(t)?.0);
                if st != Shape::Scalar || sa != sb || sa == Shape::Id {
                    return Err(bad);
                }
                sa
            }
            Node::Clamp { x, lo, hi } => {
                let (sx, sl, sh) = (get(x)?.0, get(lo)?.0, get(hi)?.0);
                arith(sx, sl)?;
                arith(sx, sh)?;
                if sl != Shape::Scalar && sl != sx || sh != Shape::Scalar && sh != sx {
                    return Err(bad);
                }
                sx
            }
            Node::SmoothStep { edge0, edge1, x } => {
                if [get(edge0)?.0, get(edge1)?.0, get(x)?.0] != [Shape::Scalar; 3] {
                    return Err(bad);
                }
                Shape::Scalar
            }
            Node::Select { condition, a, b } => {
                let (sc, sa, sb) = (get(condition)?.0, get(a)?.0, get(b)?.0);
                if sc != Shape::Scalar || sa != sb {
                    return Err(bad);
                }
                sa
            }
            Node::Component { input, index } => {
                let ok = match get(input)?.0 {
                    Shape::Vector2 => *index < 2,
                    Shape::Vector3 => *index < 3,
                    _ => false,
                };
                if !ok {
                    return Err(bad);
                }
                Shape::Scalar
            }
            Node::Vector2(a, b) => {
                if [get(a)?.0, get(b)?.0] != [Shape::Scalar; 2] {
                    return Err(bad);
                }
                Shape::Vector2
            }
            Node::Vector3(a, b, c) => {
                if [get(a)?.0, get(b)?.0, get(c)?.0] != [Shape::Scalar; 3] {
                    return Err(bad);
                }
                Shape::Vector3
            }
            Node::Length(a) => {
                if get(a)?.0 == Shape::Id {
                    return Err(bad);
                }
                Shape::Scalar
            }
            Node::Dot(a, b) => match (get(a)?.0, get(b)?.0) {
                (Shape::Vector2, Shape::Vector2) | (Shape::Vector3, Shape::Vector3) => {
                    Shape::Scalar
                }
                _ => return Err(bad),
            },
            Node::Cross(a, b) => {
                if [get(a)?.0, get(b)?.0] != [Shape::Vector3; 2] {
                    return Err(bad);
                }
                Shape::Vector3
            }
            Node::Normalize(a) => match get(a)?.0 {
                s @ (Shape::Vector2 | Shape::Vector3) => s,
                _ => return Err(bad),
            },
            Node::Curve(x, _) => {
                if get(x)?.0 != Shape::Scalar {
                    return Err(bad);
                }
                Shape::Scalar
            }
            Node::Ramp(t, _) => {
                if get(t)?.0 != Shape::Scalar {
                    return Err(bad);
                }
                Shape::Vector3
            }
            Node::Call {
                function,
                output,
                args,
            } => {
                let f = &self
                    .program
                    .functions
                    .get(*function as usize)
                    .ok_or(ContractError::UnknownReference)?
                    .program;
                let out = f
                    .outputs
                    .get(*output as usize)
                    .ok_or(ContractError::UnknownReference)?;
                if args.len() != f.inputs.len() {
                    return Err(bad);
                }
                let mut arg_scopes = Vec::with_capacity(args.len());
                for (arg, input) in args.iter().zip(&f.inputs) {
                    let (shape, scope) = get(arg)?;
                    if shape != Shape::of(input.port) {
                        return Err(bad);
                    }
                    if !scope.within(input.scope) {
                        return Err(ContractError::ScopeViolation {
                            name: input.name.clone(),
                            declared: input.scope,
                            found: scope,
                        });
                    }
                    arg_scopes.push(scope);
                }
                scope = f.scopes_with(&arg_scopes)[out.node.index()];
                Shape::of(out.port)
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
        if !p.scopes[i].within(scope) {
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
    pub fn finish(self) -> ScopedProgram {
        self.program
    }

    // --- Shorthands ------------------------------------------------------

    /// A scalar constant.
    pub fn scalar(&mut self, v: f32) -> NodeRef {
        self.constant(Value::Scalar(v))
    }

    /// A 3D vector or color constant.
    pub fn vector3(&mut self, x: f32, y: f32, z: f32) -> NodeRef {
        self.constant(Value::Vector3(Vec3::new(x, y, z)))
    }

    /// `a + b`.
    ///
    /// # Errors
    ///
    /// As [`Self::node`].
    pub fn add(&mut self, a: NodeRef, b: NodeRef) -> Result<NodeRef, ContractError> {
        self.node(Node::Add(a, b))
    }

    /// `a − b`.
    ///
    /// # Errors
    ///
    /// As [`Self::node`].
    pub fn sub(&mut self, a: NodeRef, b: NodeRef) -> Result<NodeRef, ContractError> {
        self.node(Node::Sub(a, b))
    }

    /// `a · b`.
    ///
    /// # Errors
    ///
    /// As [`Self::node`].
    pub fn mul(&mut self, a: NodeRef, b: NodeRef) -> Result<NodeRef, ContractError> {
        self.node(Node::Mul(a, b))
    }

    /// `k · a` for a constant `k`.
    ///
    /// # Errors
    ///
    /// As [`Self::node`].
    pub fn scale(&mut self, k: f32, a: NodeRef) -> Result<NodeRef, ContractError> {
        let k = self.scalar(k);
        self.mul(k, a)
    }

    /// `a + k` for a constant `k`.
    ///
    /// # Errors
    ///
    /// As [`Self::node`].
    pub fn offset(&mut self, a: NodeRef, k: f32) -> Result<NodeRef, ContractError> {
        let k = self.scalar(k);
        self.add(a, k)
    }

    /// `a + k · b` for a constant `k`.
    ///
    /// # Errors
    ///
    /// As [`Self::node`].
    pub fn add_scaled(&mut self, a: NodeRef, k: f32, b: NodeRef) -> Result<NodeRef, ContractError> {
        let kb = self.scale(k, b)?;
        self.add(a, kb)
    }

    /// `a + (b − a) · t`.
    ///
    /// # Errors
    ///
    /// As [`Self::node`].
    pub fn mix(&mut self, a: NodeRef, b: NodeRef, t: NodeRef) -> Result<NodeRef, ContractError> {
        self.node(Node::Mix { a, b, t })
    }

    /// `lo + (hi − lo) · t` for constants `lo` and `hi`.
    ///
    /// # Errors
    ///
    /// As [`Self::node`].
    pub fn lerp(&mut self, lo: f32, hi: f32, t: NodeRef) -> Result<NodeRef, ContractError> {
        let (lo, hi) = (self.scalar(lo), self.scalar(hi));
        self.mix(lo, hi, t)
    }

    /// A Hermite step of `x` between constant edges.
    ///
    /// # Errors
    ///
    /// As [`Self::node`].
    pub fn smoothstep(
        &mut self,
        edge0: f32,
        edge1: f32,
        x: NodeRef,
    ) -> Result<NodeRef, ContractError> {
        let (edge0, edge1) = (self.scalar(edge0), self.scalar(edge1));
        self.node(Node::SmoothStep { edge0, edge1, x })
    }

    /// `x` clamped to `[0, 1]`.
    ///
    /// # Errors
    ///
    /// As [`Self::node`].
    pub fn saturate(&mut self, x: NodeRef) -> Result<NodeRef, ContractError> {
        self.node(Node::Unary(UnaryOp::Saturate, x))
    }

    /// Component `index` of a vector.
    ///
    /// # Errors
    ///
    /// As [`Self::node`].
    pub fn component(&mut self, input: NodeRef, index: u8) -> Result<NodeRef, ContractError> {
        self.node(Node::Component { input, index })
    }

    /// Resource `resource` sampled at `at`.
    ///
    /// # Errors
    ///
    /// As [`Self::node`].
    pub fn sample(&mut self, resource: u32, at: NodeRef) -> Result<NodeRef, ContractError> {
        self.node(Node::Sample { resource, at })
    }
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = if e1 == e0 {
        if x < e0 { 0.0 } else { 1.0 }
    } else {
        ((x - e0) / (e1 - e0)).clamp(0.0, 1.0)
    };
    t * t * (3.0 - 2.0 * t)
}

fn scalar(v: Value) -> f32 {
    v.component(0).unwrap_or(0.0)
}

fn map(a: Value, f: impl Fn(f32) -> f32) -> Value {
    match a {
        Value::Scalar(x) => Value::Scalar(f(x)),
        Value::Vector2(x) => Value::Vector2(Vec2::new(f(x.x), f(x.y))),
        Value::Vector3(x) => Value::Vector3(Vec3::new(f(x.x), f(x.y), f(x.z))),
        Value::Id(_) => a,
    }
}

/// Applies `f` per component, broadcasting a scalar operand.
#[must_use]
pub fn zip(a: Value, b: Value, f: impl Fn(f32, f32) -> f32) -> Value {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scopes_form_a_lattice() {
        use Scope::*;
        assert_eq!(Region.join(Element), Sample, "independent frequencies");
        assert_eq!(Material.join(Region), Region, "material is the bottom");
        assert_eq!(Sample.join(Pass), Pass, "pass is the top");
        assert!(Element.within(Sample), "element within sample");
        assert!(!Pass.within(Sample), "a pass value is not point-evaluable");
        assert!(!Region.within(Element), "incomparable");
        for a in [Material, Region, Element, Sample, Pass] {
            for b in [Material, Region, Element, Sample, Pass] {
                let j = a.join(b);
                assert!(a.within(j) && b.within(j), "{a:?} ∨ {b:?} bounds both");
                assert_eq!(j, b.join(a), "join commutes");
            }
        }
    }

    #[test]
    fn outputs_that_vary_too_often_are_refused() {
        let mut b = ScopedBuilder::new("test");
        let region = b.input("area", PortType::Scalar, Scope::Region).unwrap();
        let ao = b.input("ao", PortType::Scalar, Scope::Pass).unwrap();
        let both = b.mul(region, ao).unwrap();
        assert_eq!(b.program.scopes[both.index()], Scope::Pass, "join");
        assert_eq!(
            b.output("x", PortType::Scalar, Scope::Sample, both),
            Err(ContractError::ScopeViolation {
                name: "x".into(),
                declared: Scope::Sample,
                found: Scope::Pass,
            }),
            "a pass-scope value is not point-evaluable"
        );
        b.output("x", PortType::Scalar, Scope::Pass, both).unwrap();
        b.output("r", PortType::Scalar, Scope::Region, region)
            .unwrap();
    }

    #[test]
    fn calls_reuse_a_function_and_keep_argument_scopes() {
        // f(x, k) = x · x + k
        let mut f = ScopedBuilder::new("square_plus");
        let x = f.input("x", PortType::Scalar, Scope::Pass).unwrap();
        let k = f.input("k", PortType::Scalar, Scope::Pass).unwrap();
        let xx = f.mul(x, x).unwrap();
        let y = f.add(xx, k).unwrap();
        f.output("y", PortType::Scalar, Scope::Pass, y).unwrap();
        let f = Arc::new(f.finish());

        let mut b = ScopedBuilder::new("caller");
        let e = b.input("e", PortType::Scalar, Scope::Element).unwrap();
        let s = b.input("s", PortType::Scalar, Scope::Sample).unwrap();
        let fi = b.function("square_plus", f);
        let one = b.scalar(1.0);
        let per_element = b
            .node(Node::Call {
                function: fi,
                output: 0,
                args: vec![e, one],
            })
            .unwrap();
        let per_sample = b
            .node(Node::Call {
                function: fi,
                output: 0,
                args: vec![e, s],
            })
            .unwrap();
        assert_eq!(b.program.scopes[per_element.index()], Scope::Element);
        assert_eq!(b.program.scopes[per_sample.index()], Scope::Sample);
        b.output("a", PortType::Scalar, Scope::Element, per_element)
            .unwrap();
        b.output("b", PortType::Scalar, Scope::Sample, per_sample)
            .unwrap();
        let p = b.finish();
        let out = p.eval(&[Value::Scalar(3.0), Value::Scalar(0.5)], Footprint::POINT);
        assert_eq!(out, vec![Value::Scalar(10.0), Value::Scalar(9.5)]);
    }

    #[test]
    fn comparisons_curves_and_ramps_evaluate() {
        let mut b = ScopedBuilder::new("shaping");
        let x = b.input("x", PortType::Scalar, Scope::Sample).unwrap();
        let half = b.scalar(0.5);
        let less = b.node(Node::Compare(CompareOp::Less, x, half)).unwrap();
        let curve = ToneCurve::new(&[[0.0, 0.0], [0.5, 0.2], [1.0, 1.0]]).unwrap();
        let shaped = b.node(Node::Curve(x, curve)).unwrap();
        let ramp = ColorRamp::new(&[(0.0, Vec3::ZERO), (1.0, Vec3::ONE)]).unwrap();
        let color = b.node(Node::Ramp(x, ramp)).unwrap();
        let root = b.node(Node::Unary(UnaryOp::Sqrt, x)).unwrap();
        b.output("less", PortType::Mask, Scope::Sample, less)
            .unwrap();
        b.output("shaped", PortType::Scalar, Scope::Sample, shaped)
            .unwrap();
        b.output(
            "color",
            PortType::Color(crate::Primaries::Rec709),
            Scope::Sample,
            color,
        )
        .unwrap();
        b.output("root", PortType::Scalar, Scope::Sample, root)
            .unwrap();
        let p = b.finish();
        let out = p.eval(&[Value::Scalar(0.25)], Footprint::POINT);
        assert_eq!(out[0], Value::Scalar(1.0), "0.25 < 0.5");
        assert_eq!(out[2], Value::Vector3(Vec3::splat(0.25)), "ramp");
        assert_eq!(out[3], Value::Scalar(0.5), "sqrt");
        assert!(
            matches!(out[1], Value::Scalar(v) if v > 0.0 && v < 0.2),
            "curve"
        );
        let mut again = ScopedBuilder::new("shaping");
        let id = again.constant(Value::Id(3));
        assert!(
            again.node(Node::Compare(CompareOp::Less, id, id)).is_err(),
            "identifiers only compare for equality"
        );
    }
}
