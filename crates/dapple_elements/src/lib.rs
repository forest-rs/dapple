// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Keyed element sets for dapple.
//!
//! Structured materials are made of *things*: bricks, tiles, stones,
//! flakes. This crate keeps those things as data before they become texels,
//! so one layout drives every output of a material coherently.
//!
//! - **Identity** ([`identity`]): an [`ElementKey`] is a keyed hash of a
//!   layout's author-given [`LayoutId`], the element's [`Anchor`] and its
//!   slot. It never depends on position, size or the layout's parameters,
//!   so moving or re-glazing a brick keeps its key and everything derived
//!   from it. Identity, content fingerprint, dense index and
//!   [`correspondence`] are four separate things.
//! - **Element sets** ([`ElementSet`]): a column table of keys, placements,
//!   sizes, variants and typed attributes, in canonical key order. Layouts
//!   ([`RunningBond`]) produce them; [`ElementSet::filter`],
//!   [`ElementSet::set_placement`] and [`ElementSet::set_attribute`] edit
//!   them.
//! - **Surface programs** ([`program`]): the callable contract an element
//!   invokes. A [`SurfaceProgram`] is an inspectable value with typed,
//!   scoped inputs and outputs, resource dependencies and a body of
//!   [`Node`]s; a [`ProgramInstance`] binds it under a stable
//!   [`InstanceId`]. Scopes are checked, and per-element work runs once per
//!   element.
//! - **Compositing** ([`composite`]): [`Realized::composite`] writes every
//!   program output under coverage compositing plus a winner owner label,
//!   which is a summary; [`Realized::contributors`] recomputes the full
//!   contributor list of a texel. Element identity (the owner label) and
//!   surface-material identity (an identifier output such as glaze versus
//!   body) are separate rasters. [`Realized::update`] recomputes only the
//!   tiles an edit reaches and matches a clean composite bit for bit.

#![no_std]

extern crate alloc;

pub mod composite;
pub mod identity;
mod layout;
pub mod program;
mod set;

pub use composite::{
    Composite, CompositeError, Contribution, Contributors, Realized, UpdateReport,
};
pub use identity::{Anchor, ElementKey, LayoutId};
pub use layout::RunningBond;
pub use program::{
    Binding, ContractError, InstanceId, Node, NodeRef, ProgramInstance, Scope, SurfaceBuilder,
    SurfaceProgram,
};
pub use set::{
    AttributeDecl, Bounds, Correspondence, Element, ElementError, ElementSet, Placement,
    correspondence,
};

#[cfg(test)]
mod tests;
