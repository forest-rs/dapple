// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Why an expression does not compile, and why a run fails.

use alloc::string::String;
use core::fmt;

use crate::{Span, Type};

/// Why an expression, or a set of them, does not compile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    /// Where in the expression's source. Empty (`0..0`) for an error that
    /// is not about one expression's text: a repeated input, or a tape that
    /// does not build.
    pub span: Span,
    /// What is wrong.
    pub kind: ErrorKind,
}

impl Error {
    pub(crate) const fn new(span: Span, kind: ErrorKind) -> Self {
        Self { span, kind }
    }
}

/// What is wrong with an expression.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// The text does not parse: what was expected.
    Syntax(&'static str),
    /// Comparisons chained without parentheses, such as `a < b == c`.
    ChainedComparison,
    /// The expression nests deeper than a program may.
    TooDeep,
    /// A literal the language cannot hold: why.
    Literal(&'static str),
    /// A name that is no input.
    UnknownName(String),
    /// A call of a function the language does not have.
    UnknownFunction(String),
    /// A function called with the wrong number of arguments.
    Arity {
        /// The function.
        function: &'static str,
        /// The arguments it takes.
        expected: usize,
        /// The arguments given.
        found: usize,
    },
    /// A value of one type where another is needed.
    Mismatch {
        /// The type needed.
        expected: Type,
        /// The type found.
        found: Type,
    },
    /// Two operands, or two arms of `?:`, of different types.
    Operands {
        /// The left operand's type.
        left: Type,
        /// The right operand's type.
        right: Type,
    },
    /// An operation or function that does not take this type: which one.
    Unsupported {
        /// The operation or function.
        operation: &'static str,
        /// The type it was given.
        ty: Type,
    },
    /// Two inputs of one name.
    DuplicateInput(String),
    /// An expression conditioned on a result that is no earlier flag
    /// expression: that result's index.
    Condition(usize),
    /// The compiled tape did not build or verify: a bug in this crate, with
    /// the tape's own account of it.
    Tape(String),
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax(expected) => write!(f, "expected {expected}"),
            Self::ChainedComparison => f.write_str("comparisons do not chain; add parentheses"),
            Self::TooDeep => f.write_str("the expression nests too deeply"),
            Self::Literal(why) => f.write_str(why),
            Self::UnknownName(n) => write!(f, "no input named {n:?}"),
            Self::UnknownFunction(n) => write!(f, "no function named {n:?}"),
            Self::Arity {
                function,
                expected,
                found,
            } => write!(
                f,
                "{function} takes {expected} argument{}, not {found}",
                if *expected == 1 { "" } else { "s" }
            ),
            Self::Mismatch { expected, found } => write!(
                f,
                "{} where {} is needed",
                found.with_article(),
                expected.with_article()
            ),
            Self::Operands { left, right } => write!(
                f,
                "{} and {} do not mix",
                left.with_article(),
                right.with_article()
            ),
            Self::Unsupported { operation, ty } => {
                write!(f, "{operation} does not take {}", ty.with_article())
            }
            Self::DuplicateInput(n) => write!(f, "two inputs named {n:?}"),
            Self::Condition(i) => write!(f, "result {i} is no earlier flag expression"),
            Self::Tape(e) => write!(f, "the tape does not build: {e}"),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.span == Span::default() {
            self.kind.fmt(f)
        } else {
            write!(
                f,
                "at {}..{}: {}",
                self.span.start, self.span.end, self.kind
            )
        }
    }
}

impl core::error::Error for Error {}

/// Why a run of a [`crate::Program`] failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunError {
    /// The expression that failed, by index, when the failure is one
    /// expression's.
    pub expression: Option<usize>,
    /// What happened.
    pub kind: RunErrorKind,
}

/// What stopped a run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunErrorKind {
    /// The inputs are not the program's: a count or a type differs.
    Inputs,
    /// The run spent its fuel.
    OutOfFuel,
    /// Integer arithmetic left `u32`.
    Overflow,
    /// An integer division by zero.
    DivisionByZero,
    /// A `clamp` whose bounds are out of order (or NaN).
    ClampBounds,
    /// The VM stopped for another reason: a bug in this crate, with the
    /// VM's account of it.
    Trap(String),
}

impl fmt::Display for RunErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inputs => f.write_str("the inputs do not match the program's"),
            Self::OutOfFuel => f.write_str("the run spent its fuel"),
            Self::Overflow => f.write_str("integer arithmetic overflowed"),
            Self::DivisionByZero => f.write_str("an integer division by zero"),
            Self::ClampBounds => f.write_str("clamp bounds out of order"),
            Self::Trap(t) => write!(f, "the VM trapped: {t}"),
        }
    }
}

impl RunErrorKind {
    /// A short, fixed description, for error types that hold only a
    /// `&'static str`.
    #[must_use]
    pub const fn reason(&self) -> &'static str {
        match self {
            Self::Inputs => "the inputs do not match the program's",
            Self::OutOfFuel => "the expression ran out of fuel",
            Self::Overflow => "the expression's integer arithmetic overflowed",
            Self::DivisionByZero => "the expression divides by zero",
            Self::ClampBounds => "the expression clamps with bounds out of order",
            Self::Trap(_) => "the expression's program trapped",
        }
    }
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.expression {
            Some(i) => write!(f, "expression {i}: {}", self.kind),
            None => self.kind.fmt(f),
        }
    }
}

impl core::error::Error for RunError {}
