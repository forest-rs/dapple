// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Small typed expressions over named inputs, compiled to one verified
//! [`execution_tape`] program: "let the numbers be formulas".
//!
//! A [`Builder`] takes the inputs (each a name and a [`Type`]) and any
//! number of expressions, each checked against the type its result must
//! have when it is added. [`Builder::build`] compiles them all into one
//! pure tape function, inputs in and one result per expression out, with
//! no host calls, and verifies it. The resulting [`Program`] runs that
//! function on a parked VM under a fuel budget and returns the values.
//!
//! An expression can also be added on a condition
//! ([`Builder::expression_when`]): an earlier flag result, and the value it
//! must have for the expression to run. Where the condition does not hold,
//! none of the expression runs, so nothing in it can stop the run, and its
//! result is a placeholder zero. A package uses this for the parameters of
//! a step that runs only when a flag holds.
//!
//! ```
//! use dapple_logic::{Builder, Type, Value};
//!
//! let mut b = Builder::new();
//! b.input("chipping", Type::Scalar)?;
//! b.input("weather", Type::Scalar)?;
//! b.expression("max(chipping, 0.5 * weather * weather)", Type::Scalar)?;
//! let program = b.build()?;
//! let out = program.run(&[Value::Scalar(0.03), Value::Scalar(0.5)])?;
//! assert_eq!(out, [Value::Scalar(0.125)]);
//! # Ok::<(), Box<dyn core::error::Error>>(())
//! ```
//!
//! # Exactness
//!
//! A program computes what the same expression written in Rust computes,
//! bit for bit, on every platform: operations are emitted in source order,
//! left to right, as written (`a - b - c` is `(a - b) - c`), never
//! reassociated, never folded and never contracted into a fused
//! multiply-add. Literals are read as `str::parse::<f32>` reads them.
//! Arithmetic is IEEE 754 `f32` arithmetic, and the functions are the
//! `libm` 0.2.16 routines dapple's native modules call (through
//! `execution_tape::math`, which is defined as them). A NaN result is a
//! NaN; its sign and payload are not part of the promise.
//!
//! [`Program::content_words`] is what a fingerprint of a program's behavior
//! needs: the encoded tape (its encoding is canonical, so the same source
//! always gives the same bytes) and [`CONTRACT`], which names the tape
//! format and the math the bytes are run with.
//!
//! # The language
//!
//! Three types: `scalar` (`f32`), `integer` (`u32`) and `flag` (`bool`).
//!
//! ```text
//! expr    := or ( "?" expr ":" expr )?          right-associative
//! or      := and ( "||" and )*                  short-circuit
//! and     := cmp ( "&&" cmp )*                  short-circuit
//! cmp     := sum ( ("<" | ">" | "<=" | ">=" | "==" | "!=") sum )?
//! sum     := product ( ("+" | "-") product )*   left-associative
//! product := unary ( ("*" | "/") unary )*       left-associative
//! unary   := ("-" | "!") unary | primary
//! primary := number | "true" | "false" | name | name "(" args ")" | "(" expr ")"
//! ```
//!
//! - **Comparisons do not chain.** `a < b == c` is an error, as in Rust;
//!   write `(a < b) == c`. `<`, `>`, `<=`, `>=` compare scalars or
//!   integers; `==` and `!=` compare any two values of one type.
//! - **Unary minus binds tighter than `*`**: `-a * b` is `(-a) * b` (the
//!   same bits either way, since negation is exact). It negates scalars
//!   only.
//! - **Numbers.** A number with a fraction or an exponent (`0.5`, `2e-3`)
//!   is a scalar. A whole number (`2`) takes the type its context needs: a
//!   scalar in `0.5 * 2`, an integer in `count + 2`, and an integer when
//!   nothing decides. Numbers are unsigned; `-2.0` is a negation.
//! - **Integer arithmetic is checked**: a sum, difference or product that
//!   leaves `u32`, and a division by zero, stop the run with an error,
//!   where Rust would panic in a debug build.
//! - **`?:`, `&&` and `||` evaluate only what they need**, as Rust's
//!   `if`, `&&` and `||` do; both arms of `?:` have one type.
//!
//! Functions, each exactly the Rust over `libm` beside it:
//!
//! | Function | Types | Rust |
//! |---|---|---|
//! | `min(a, b)`, `max(a, b)` | scalars or integers | `f32::min`/`f32::max` (`-0.0` below `+0.0`), `Ord::min`/`Ord::max` |
//! | `clamp(x, lo, hi)` | scalars or integers | `f32::clamp`, `Ord::clamp`: an error unless `lo <= hi` |
//! | `abs(x)` | scalar | `libm::fabsf` |
//! | `floor(x)` | scalar | `libm::floorf` |
//! | `round(x)` | scalar | `libm::roundf`: half away from zero |
//! | `sqrt(x)` | scalar | `libm::sqrtf` |
//! | `pow(x, y)` | scalars | `libm::powf` |
//! | `exp(x)`, `ln(x)` | scalar | `libm::expf`, `libm::logf` |
//! | `sin(x)`, `cos(x)` | scalar | `libm::sinf`, `libm::cosf` |
//! | `atan2(y, x)` | scalars | `libm::atan2f` |
//! | `mix(a, b, t)` | scalars | `a + (b - a) * t` |
//! | `smoothstep(e0, e1, x)` | scalars | `dapple_field`'s: `t * t * (3.0 - 2.0 * t)` for `t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0)`, or a step at `e0` when `e1 == e0` |
//! | `scalar(n)` | integer to scalar | `n as f32` |
//!
//! `min` and `max` differ from Rust's `f32::min` and `f32::max` in one
//! respect only: those may return either zero for `-0.0` and `+0.0`, these
//! always order `-0.0` below `+0.0`.
//!
//! An expression nests at most 64 deep, counting each operator, call and
//! parenthesis on the way down (a chain `a + b + c` is two deep), so no
//! input can exhaust the stack of the passes that walk it.
//!
//! Not in the language yet, on purpose: colors and seeds (no vector type),
//! a scalar-to-integer conversion (it needs a decision between rounding
//! modes and between saturating and failing), `%`, and named constants such
//! as `pi`.

#![no_std]

extern crate alloc;

mod check;
mod emit;
mod error;
mod program;
mod syntax;

pub use error::{Error, ErrorKind, RunError, RunErrorKind};
pub use program::{Builder, CONTRACT, Program};

/// Where in an expression's source something is: a byte range.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Span {
    /// The first byte.
    pub start: usize,
    /// One past the last byte.
    pub end: usize,
}

impl Span {
    /// The bytes `start..end`.
    #[must_use]
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// From the start of `self` to the end of `other`.
    #[must_use]
    pub(crate) const fn to(self, other: Self) -> Self {
        Self::new(self.start, other.end)
    }
}

/// The type of an input, an expression or a value.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    /// A real number: `f32`.
    Scalar,
    /// A count: `u32`.
    Integer,
    /// A switch: `bool`.
    Flag,
}

impl Type {
    /// Its name in messages: `scalar`, `integer` or `flag`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Integer => "integer",
            Self::Flag => "flag",
        }
    }

    /// Its name with an article, for messages: `a scalar`, `an integer`.
    pub(crate) const fn with_article(self) -> &'static str {
        match self {
            Self::Scalar => "a scalar",
            Self::Integer => "an integer",
            Self::Flag => "a flag",
        }
    }
}

impl core::fmt::Display for Type {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

/// A value of a [`Type`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Value {
    /// A [`Type::Scalar`].
    Scalar(f32),
    /// A [`Type::Integer`].
    Integer(u32),
    /// A [`Type::Flag`].
    Flag(bool),
}

impl Value {
    /// Its type.
    #[must_use]
    pub const fn ty(self) -> Type {
        match self {
            Self::Scalar(_) => Type::Scalar,
            Self::Integer(_) => Type::Integer,
            Self::Flag(_) => Type::Flag,
        }
    }
}

#[cfg(test)]
mod reference_tests;
#[cfg(test)]
mod tests;
