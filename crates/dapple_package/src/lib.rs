// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Portable material packages for dapple.
//!
//! A package defines a material module as data, so it can be saved,
//! diffed, shipped and loaded by another program. It has two forms:
//!
//! - **The source** ([`Package`], in [`source`]): the editable form, read
//!   and written as JSON. It holds the module's **interface** (typed
//!   parameters with units, ranges and defaults; material, map and
//!   resource inputs, where a resource input is a requirement the host
//!   must meet; outputs with **semantics**, the channels a material binds
//!   and the axes it tiles along), what it **requires** of the engine
//!   (capabilities at a least version, and the modules it uses at exact
//!   versions), **presets** (named parameter sets, such as a fit's result),
//!   and its **body**: a graph of module instances, each binding its
//!   parameters to literals, to the package's own parameters or to
//!   expressions over them (`{"expr": "max(chipping, 0.5 * weather *
//!   weather)"}`, in [`dapple_logic`]'s language), and its inputs to the
//!   package's inputs or to earlier steps' outputs. A step can be
//!   **guarded** (`"when": "moss > 0.0"`): it runs only where that holds.
//! - **The execution artifact** ([`CompiledPackage`], from
//!   [`Registry::compile`]): the source checked against one engine's
//!   modules and capabilities, every name resolved and every binding's
//!   kind and range checked. It is a [`dapple_material::module::Module`]
//!   and instantiates like one, so its steps' seeds derive from their
//!   instance paths exactly as a native module's children's do: a package
//!   that composes the same modules under the same names realizes the same
//!   bits.
//!
//! Documents carry a `format` and a `version` ([`FORMAT_VERSION`]); a
//! reader refuses any other version, unknown fields, capabilities the
//! engine does not offer and modules it does not have, each with a typed
//! [`PackageError`]. Maps serialize in key order, so a package's bytes,
//! and its [`Package::fingerprint`], are canonical.
//!
//! Presets can also stand alone ([`PresetSet`]) for modules compiled into
//! the program.
//!
//! **Expressions** derive step parameters from the package's own: one knob
//! can drive several steps. Compiling checks every expression's names and
//! types and compiles all of a package's expressions into one verified
//! `execution_tape` program ([`CompiledPackage::logic`]); each instance
//! runs it once, before its steps, and binds the values like literals, so
//! a step's instance key holds the derived value's bits and two package
//! arguments that derive the same value share the step's cached instance.
//! The values are exactly what the same expressions written in Rust over
//! `libm` compute. A value that does not fit its step parameter (out of
//! range, or not finite) fails the instance with
//! [`ModuleErrorKind::InvalidParam`](dapple_material::module::ModuleErrorKind::InvalidParam)
//! at the step's path, naming the parameter and the expression. A package
//! that uses expressions requires [`compile::LOGIC`], and its fingerprint
//! covers the compiled program; a package without them pays nothing and
//! keeps the fingerprint it had before expressions existed.
//!
//! **Guards** make some steps conditional: a step with `"when": "<flag
//! expression>"` runs only where that holds, one with `"unless"` only where
//! it does not. A step that does not run is not instantiated, costs nothing
//! and leaves no cache entry; the instance's diagnostics list it as
//! skipped. The recipe is still a fixed list of steps: some are
//! conditional. Compiling proves that nothing reads a step's output where
//! the step might not have run ([`compile`] states the rule): a reader
//! under the same guard can; a select (`{"select": {"when": ..., "then":
//! ..., "else": ...}}`) reads each branch under its side of its condition;
//! and a step's `otherwise` (`{"material": "base"}`) passes its own input
//! through where it does not run, which makes it an optional layer anyone
//! can read. A package that uses guards or selects requires
//! [`compile::LOGIC`] at version [`compile::LOGIC_GUARDS`].
//!
//! ```json
//! { "name": "moss",
//!   "module": { "name": "dapple_library.moss", "version": 1 },
//!   "when": "moss > 0.0",
//!   "params": { "coverage": { "param": "moss" } },
//!   "inputs": { "base": { "output": { "step": "salts", "output": "material" } } },
//!   "otherwise": { "material": "base" } }
//! ```
//!
//! Bodies compose modules; they do not carry field programs, which stay
//! inside native modules, and they are not programs: no step runs twice,
//! and no step is made by an expression. A package reaches everything a
//! registered module exposes and nothing more.

#![no_std]

extern crate alloc;

use alloc::boxed::Box;
use alloc::string::String;
use core::fmt;

pub mod compile;
pub mod source;

pub use compile::{CompiledPackage, Registry};
pub use source::{
    Capability, FORMAT_VERSION, InterfaceSource, ModuleRef, Package, Preset, PresetSet, ValueSource,
};

/// Why a package or preset document could not be read or compiled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackageError {
    /// The text is not valid JSON of the document's shape (including a
    /// field the format does not have).
    Syntax(String),
    /// The document is not of the expected format.
    Format {
        /// The format expected.
        expected: String,
        /// The format found.
        found: String,
    },
    /// The document is of a version this crate does not read.
    UnsupportedVersion {
        /// Its version.
        found: u32,
        /// The version this crate reads.
        supported: u32,
    },
    /// The package requires a capability the engine does not offer.
    UnknownCapability(Capability),
    /// The engine offers the capability at too old a version.
    CapabilityVersion {
        /// What the package requires.
        required: Capability,
        /// The version the engine offers.
        offered: u32,
    },
    /// The package uses a capability it does not declare.
    UndeclaredCapability(Capability),
    /// A step uses a module the engine does not have at that version.
    UnknownModule(ModuleRef),
    /// A step uses a module the package does not declare.
    UndeclaredDependency(ModuleRef),
    /// Presets are for another module.
    WrongModule {
        /// The module checked against.
        expected: ModuleRef,
        /// The module the presets name.
        found: ModuleRef,
    },
    /// A name appears twice where names must be unique.
    Duplicate(String),
    /// A parameter default or preset value does not fit its parameter.
    InvalidParam {
        /// The parameter.
        name: String,
        /// What is wrong.
        reason: &'static str,
    },
    /// A semantic declaration names no channel.
    UnknownChannel(String),
    /// A step's binding does not check.
    Binding {
        /// The step.
        step: String,
        /// The parameter or input.
        name: String,
        /// What is wrong.
        reason: &'static str,
    },
    /// A package output does not check.
    Output {
        /// The output.
        name: String,
        /// What is wrong.
        reason: &'static str,
    },
    /// An expression does not check: it does not parse, names something
    /// that is no scalar, integer or flag parameter of the package, or does
    /// not make what its place needs (the step parameter's kind, or a flag
    /// for a guard or a select).
    Expression {
        /// The step; empty for a select of a package output.
        step: String,
        /// Where in the step: the parameter it binds, `when` or `unless`
        /// for a guard, or the input whose select it is; for a package
        /// output's select, the output.
        name: String,
        /// The expression.
        source: String,
        /// What is wrong, and where in `source` (boxed: it is the largest
        /// error a package has).
        error: Box<dapple_logic::Error>,
    },
    /// The package's expressions, each of which checks, do not compile
    /// into one program: a bug in `dapple_logic`.
    Logic(dapple_logic::Error),
    /// A binding reads the output of a guarded step where that step might
    /// not have run: not under its guard, and not an output its `otherwise`
    /// gives. See [`compile`] for the rule.
    MaybeSkipped {
        /// The step that reads; empty for a package output.
        step: String,
        /// Its input (or `otherwise` output) that reads, or the package
        /// output.
        name: String,
        /// The guarded step it reads.
        reads: String,
        /// That step's condition.
        condition: String,
        /// Whether that step runs where the condition holds (`when`) or
        /// where it does not (`unless`).
        holds: bool,
    },
}

impl fmt::Display for PackageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax(e) => write!(f, "not a readable document: {e}"),
            Self::Format { expected, found } => write!(f, "a {found:?} document, not {expected:?}"),
            Self::UnsupportedVersion { found, supported } => {
                write!(f, "format version {found}; this reader reads {supported}")
            }
            Self::UnknownCapability(c) => {
                write!(f, "the engine does not offer {}@{}", c.name, c.version)
            }
            Self::CapabilityVersion { required, offered } => write!(
                f,
                "{} needs version {}; the engine offers {offered}",
                required.name, required.version
            ),
            Self::UndeclaredCapability(c) => {
                write!(f, "uses {}@{} without requiring it", c.name, c.version)
            }
            Self::UnknownModule(m) => {
                write!(f, "the engine has no module {}@{}", m.name, m.version)
            }
            Self::UndeclaredDependency(m) => {
                write!(f, "uses {}@{} without requiring it", m.name, m.version)
            }
            Self::WrongModule { expected, found } => write!(
                f,
                "presets for {}@{}, not {}@{}",
                found.name, found.version, expected.name, expected.version
            ),
            Self::Duplicate(n) => write!(f, "{n:?} appears twice"),
            Self::InvalidParam { name, reason } => write!(f, "parameter {name:?}: {reason}"),
            Self::UnknownChannel(n) => write!(f, "no channel {n:?}"),
            Self::Binding { step, name, reason } => write!(f, "step {step:?}, {name:?}: {reason}"),
            Self::Output { name, reason } => write!(f, "output {name:?}: {reason}"),
            Self::Expression {
                step,
                name,
                source,
                error,
            } if step.is_empty() => write!(f, "output {name:?}: in {source:?}, {error}"),
            Self::Expression {
                step,
                name,
                source,
                error,
            } => write!(f, "step {step:?}, {name:?}: in {source:?}, {error}"),
            Self::Logic(e) => write!(f, "the package's expressions do not compile: {e}"),
            Self::MaybeSkipped {
                step,
                name,
                reads,
                condition,
                holds,
            } => {
                if step.is_empty() {
                    write!(f, "output {name:?}")?;
                } else {
                    write!(f, "step {step:?}, {name:?}")?;
                }
                write!(
                    f,
                    ": reads step {reads:?}, which runs only {} {condition:?}; read it under the same \
                     condition (from a step with the same guard, or through a select on it), or \
                     give {reads:?} an `otherwise`",
                    if *holds { "when" } else { "unless" }
                )
            }
        }
    }
}

impl core::error::Error for PackageError {}

#[cfg(test)]
mod guard_tests;
#[cfg(test)]
mod tests;
