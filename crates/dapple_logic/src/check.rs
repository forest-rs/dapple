// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Syntax tree to typed tree: names resolved, types checked, whole-number
//! literals given the type their context needs.

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::error::{Error, ErrorKind};
use crate::syntax::{Binary, Expr, ExprKind, Unary};
use crate::{Span, Type};

/// A function of the language.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Function {
    Min,
    Max,
    Clamp,
    Abs,
    Floor,
    Round,
    Sqrt,
    Pow,
    Exp,
    Ln,
    Sin,
    Cos,
    Atan2,
    Mix,
    Smoothstep,
    ToScalar,
}

/// What a function's arguments and result are.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Shape {
    /// Scalars or integers, all of one type, which is the result's.
    Numeric,
    /// Scalars, to a scalar.
    Scalar,
    /// An integer, to a scalar.
    IntegerToScalar,
}

impl Function {
    const ALL: [Self; 16] = [
        Self::Min,
        Self::Max,
        Self::Clamp,
        Self::Abs,
        Self::Floor,
        Self::Round,
        Self::Sqrt,
        Self::Pow,
        Self::Exp,
        Self::Ln,
        Self::Sin,
        Self::Cos,
        Self::Atan2,
        Self::Mix,
        Self::Smoothstep,
        Self::ToScalar,
    ];

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Min => "min",
            Self::Max => "max",
            Self::Clamp => "clamp",
            Self::Abs => "abs",
            Self::Floor => "floor",
            Self::Round => "round",
            Self::Sqrt => "sqrt",
            Self::Pow => "pow",
            Self::Exp => "exp",
            Self::Ln => "ln",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Atan2 => "atan2",
            Self::Mix => "mix",
            Self::Smoothstep => "smoothstep",
            Self::ToScalar => "scalar",
        }
    }

    const fn arity(self) -> usize {
        match self {
            Self::Abs
            | Self::Floor
            | Self::Round
            | Self::Sqrt
            | Self::Exp
            | Self::Ln
            | Self::Sin
            | Self::Cos
            | Self::ToScalar => 1,
            Self::Min | Self::Max | Self::Pow | Self::Atan2 => 2,
            Self::Clamp | Self::Mix | Self::Smoothstep => 3,
        }
    }

    const fn shape(self) -> Shape {
        match self {
            Self::Min | Self::Max | Self::Clamp => Shape::Numeric,
            Self::ToScalar => Shape::IntegerToScalar,
            _ => Shape::Scalar,
        }
    }

    fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|f| f.name() == name)
    }
}

/// A node of the typed tree: what it computes, its type, and where its
/// source is.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Typed {
    pub(crate) node: Node,
    pub(crate) ty: Type,
    pub(crate) span: Span,
}

/// What a typed node computes. Operand types are their nodes' `ty`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Node {
    Scalar(f32),
    Integer(u32),
    Flag(bool),
    /// An input, by index.
    Input(usize),
    Unary(Unary, Box<Typed>),
    /// Arithmetic or a comparison: the operands share one type.
    Binary(Binary, Box<Typed>, Box<Typed>),
    /// `cond ? then : otherwise`
    Cond(Box<Typed>, Box<Typed>, Box<Typed>),
    Call(Function, Vec<Typed>),
}

/// The inputs an expression can name.
pub(crate) struct Scope<'a> {
    pub(crate) inputs: &'a [(String, Type)],
}

fn mismatch(span: Span, expected: Type, found: Type) -> Error {
    Error::new(span, ErrorKind::Mismatch { expected, found })
}

/// Checks that `found` is what `want` asks for, if it asks.
fn expect(span: Span, want: Option<Type>, found: Type) -> Result<(), Error> {
    match want {
        Some(w) if w != found => Err(mismatch(span, w, found)),
        _ => Ok(()),
    }
}

/// The one type of two operands, if either has one.
fn join(span: Span, a: Option<Type>, b: Option<Type>) -> Result<Option<Type>, Error> {
    match (a, b) {
        (Some(l), Some(r)) if l != r => {
            Err(Error::new(span, ErrorKind::Operands { left: l, right: r }))
        }
        (Some(t), _) | (_, Some(t)) => Ok(Some(t)),
        (None, None) => Ok(None),
    }
}

impl Scope<'_> {
    fn input(&self, name: &str, span: Span) -> Result<(usize, Type), Error> {
        self.inputs
            .iter()
            .position(|(n, _)| n == name)
            .map(|i| (i, self.inputs[i].1))
            .ok_or_else(|| Error::new(span, ErrorKind::UnknownName(name.to_string())))
    }

    /// The type `e` has whatever its context, or `None` when it is a
    /// whole-number literal (or made only of them) and takes its context's.
    /// Reports only what [`Self::elaborate`] would report as well.
    fn synthesize(&self, e: &Expr<'_>) -> Result<Option<Type>, Error> {
        Ok(match &e.kind {
            ExprKind::Number(_, real) => real.then_some(Type::Scalar),
            ExprKind::Bool(_) => Some(Type::Flag),
            ExprKind::Name(n) => Some(self.input(n, e.span)?.1),
            ExprKind::Unary(Unary::Neg, _) => Some(Type::Scalar),
            ExprKind::Unary(Unary::Not, _) => Some(Type::Flag),
            ExprKind::Binary(op, _, _) if op.is_comparison() => Some(Type::Flag),
            ExprKind::Binary(Binary::And | Binary::Or, _, _) => Some(Type::Flag),
            ExprKind::Binary(_, a, b) => join(e.span, self.synthesize(a)?, self.synthesize(b)?)?,
            ExprKind::Cond(_, a, b) => join(e.span, self.synthesize(a)?, self.synthesize(b)?)?,
            ExprKind::Call(name, span, args) => {
                let f = Function::named(name).ok_or_else(|| {
                    Error::new(*span, ErrorKind::UnknownFunction((*name).to_string()))
                })?;
                match f.shape() {
                    Shape::Numeric => {
                        let mut t = None;
                        for a in args {
                            t = join(e.span, t, self.synthesize(a)?)?;
                        }
                        t
                    }
                    Shape::Scalar | Shape::IntegerToScalar => Some(Type::Scalar),
                }
            }
        })
    }

    /// The typed tree of `e`, which must be of type `want` if given.
    pub(crate) fn elaborate(&self, e: &Expr<'_>, want: Option<Type>) -> Result<Typed, Error> {
        let span = e.span;
        let typed = |node, ty| Typed { node, ty, span };
        match &e.kind {
            ExprKind::Number(text, real) => {
                let ty = if *real {
                    Type::Scalar
                } else {
                    want.unwrap_or(Type::Integer)
                };
                expect(span, want, ty)?;
                match ty {
                    Type::Scalar => {
                        // Exactly as Rust reads an `f32` literal.
                        let v: f32 = text
                            .parse()
                            .map_err(|_| Error::new(span, ErrorKind::Syntax("a number")))?;
                        if !v.is_finite() {
                            return Err(Error::new(
                                span,
                                ErrorKind::Literal("too large for a scalar"),
                            ));
                        }
                        Ok(typed(Node::Scalar(v), ty))
                    }
                    Type::Integer => {
                        let v: u32 = text.parse().map_err(|_| {
                            Error::new(span, ErrorKind::Literal("too large for an integer"))
                        })?;
                        Ok(typed(Node::Integer(v), ty))
                    }
                    Type::Flag => Err(mismatch(span, Type::Flag, Type::Integer)),
                }
            }
            ExprKind::Bool(b) => {
                expect(span, want, Type::Flag)?;
                Ok(typed(Node::Flag(*b), Type::Flag))
            }
            ExprKind::Name(n) => {
                let (i, ty) = self.input(n, span)?;
                expect(span, want, ty)?;
                Ok(typed(Node::Input(i), ty))
            }
            ExprKind::Unary(op, a) => {
                let ty = match op {
                    Unary::Neg => {
                        if let Some(t @ (Type::Integer | Type::Flag)) = want.or(self.synthesize(a)?)
                        {
                            return Err(Error::new(
                                span,
                                ErrorKind::Unsupported {
                                    operation: "`-`",
                                    ty: t,
                                },
                            ));
                        }
                        Type::Scalar
                    }
                    Unary::Not => Type::Flag,
                };
                expect(span, want, ty)?;
                let a = self.elaborate(a, Some(ty))?;
                Ok(typed(Node::Unary(*op, Box::new(a)), ty))
            }
            ExprKind::Binary(op @ (Binary::And | Binary::Or), a, b) => {
                expect(span, want, Type::Flag)?;
                let a = self.elaborate(a, Some(Type::Flag))?;
                let b = self.elaborate(b, Some(Type::Flag))?;
                Ok(typed(
                    Node::Binary(*op, Box::new(a), Box::new(b)),
                    Type::Flag,
                ))
            }
            ExprKind::Binary(op, a, b) => {
                let shared = join(span, self.synthesize(a)?, self.synthesize(b)?)?;
                let (operands, ty) = if op.is_comparison() {
                    (shared.unwrap_or(Type::Integer), Type::Flag)
                } else {
                    let t = shared.or(want).unwrap_or(Type::Integer);
                    (t, t)
                };
                let ordered = !matches!(op, Binary::Eq | Binary::Ne);
                if operands == Type::Flag && ordered {
                    return Err(Error::new(
                        span,
                        ErrorKind::Unsupported {
                            operation: op.symbol(),
                            ty: Type::Flag,
                        },
                    ));
                }
                expect(span, want, ty)?;
                let a = self.elaborate(a, Some(operands))?;
                let b = self.elaborate(b, Some(operands))?;
                Ok(typed(Node::Binary(*op, Box::new(a), Box::new(b)), ty))
            }
            ExprKind::Cond(c, a, b) => {
                let c = self.elaborate(c, Some(Type::Flag))?;
                let ty = join(span, self.synthesize(a)?, self.synthesize(b)?)?
                    .or(want)
                    .unwrap_or(Type::Integer);
                expect(span, want, ty)?;
                let a = self.elaborate(a, Some(ty))?;
                let b = self.elaborate(b, Some(ty))?;
                Ok(typed(Node::Cond(Box::new(c), Box::new(a), Box::new(b)), ty))
            }
            ExprKind::Call(name, name_span, args) => {
                let f = Function::named(name).ok_or_else(|| {
                    Error::new(*name_span, ErrorKind::UnknownFunction((*name).to_string()))
                })?;
                if args.len() != f.arity() {
                    return Err(Error::new(
                        span,
                        ErrorKind::Arity {
                            function: f.name(),
                            expected: f.arity(),
                            found: args.len(),
                        },
                    ));
                }
                let (operands, ty) = match f.shape() {
                    Shape::Numeric => {
                        let mut t = None;
                        for a in args {
                            t = join(span, t, self.synthesize(a)?)?;
                        }
                        let t = t.or(want).unwrap_or(Type::Integer);
                        if t == Type::Flag {
                            return Err(Error::new(
                                span,
                                ErrorKind::Unsupported {
                                    operation: f.name(),
                                    ty: t,
                                },
                            ));
                        }
                        (t, t)
                    }
                    Shape::Scalar => (Type::Scalar, Type::Scalar),
                    Shape::IntegerToScalar => (Type::Integer, Type::Scalar),
                };
                expect(span, want, ty)?;
                let args = args
                    .iter()
                    .map(|a| self.elaborate(a, Some(operands)))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(typed(Node::Call(f, args), ty))
            }
        }
    }
}
