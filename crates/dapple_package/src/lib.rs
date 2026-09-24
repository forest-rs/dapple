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
//!   parameters to literals or to the package's own parameters, and its
//!   inputs to the package's inputs or to earlier steps' outputs.
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
//! Bodies compose modules; they do not yet carry field programs, which
//! stay inside native modules. A package reaches everything a registered
//! module exposes and nothing more.

#![no_std]

extern crate alloc;

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
        }
    }
}

impl core::error::Error for PackageError {}

#[cfg(test)]
mod tests;
