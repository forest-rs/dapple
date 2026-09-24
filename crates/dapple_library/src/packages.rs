// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The library as a package engine: every module registered, and packages
//! written against them.
//!
//! [`registry`] offers every module in [`crate::modules`] to
//! [`dapple_package`] packages. [`VARNISHED_BOARD`] is
//! [`crate::modules::VarnishedBoard`] written as a package: the same
//! modules under the same instance names, so it realizes the same bits as
//! the native module.

use alloc::sync::Arc;

use dapple_package::Registry;

use crate::modules::{
    AshlarLimestone, Beech, Birch, ByExample, CeramicBody, EdgeWear, Efflorescence, Finish,
    FlintWall, GlazedBrickWall, Grime, Marble, Mortar, Moss, RomanBrick, RubbleWall, ScotsPine,
    Spruce, Stone, StoneSill, Streaks, TerracottaTile, Threshold, VarnishedBoard, Wood,
};

/// The varnished board's package source.
pub const VARNISHED_BOARD: &str = include_str!("../packages/varnished_board.json");

/// An engine offering this crate's modules and `dapple_package`'s own
/// capabilities.
#[must_use]
pub fn registry() -> Registry {
    Registry::new()
        .with(Arc::new(CeramicBody))
        .with(Arc::new(Mortar))
        .with(Arc::new(Stone))
        .with(Arc::new(Wood))
        .with(Arc::new(Finish))
        .with(Arc::new(Grime))
        .with(Arc::new(Streaks))
        .with(Arc::new(Efflorescence))
        .with(Arc::new(EdgeWear))
        .with(Arc::new(Moss))
        .with(Arc::new(ByExample))
        .with(Arc::new(GlazedBrickWall))
        .with(Arc::new(StoneSill))
        .with(Arc::new(VarnishedBoard))
        .with(Arc::new(Threshold))
        .with(Arc::new(Beech))
        .with(Arc::new(Birch))
        .with(Arc::new(ScotsPine))
        .with(Arc::new(Spruce))
        .with(Arc::new(AshlarLimestone))
        .with(Arc::new(RubbleWall))
        .with(Arc::new(FlintWall))
        .with(Arc::new(RomanBrick))
        .with(Arc::new(Marble))
        .with(Arc::new(TerracottaTile))
}
