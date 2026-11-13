// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The library as a package engine: every module registered, and packages
//! written against them.
//!
//! [`registry`] offers every module in [`crate::modules`] to
//! [`dapple_package`] packages. [`VARNISHED_BOARD`] is
//! [`crate::modules::VarnishedBoard`] written as a package: the same
//! modules under the same instance names, so it realizes the same bits as
//! the native module. So are the metals: [`WEATHERED_COPPER`] and
//! [`RUSTED_STEEL`], each with presets from fresh to heavily weathered.

use alloc::sync::Arc;

use dapple_package::Registry;

use crate::modules::{
    AshlarLimestone, Beech, Birch, ByExample, CeramicBody, Copper, EdgeWear, Efflorescence, Finish,
    FlintWall, GlazedBrickWall, Grime, Marble, Mortar, Moss, Patina, RomanBrick, RubbleWall, Rust,
    RustedSteel, ScotsPine, Spruce, Steel, Stone, StoneSill, Streaks, TerracottaTile, Threshold,
    VarnishedBoard, WeatheredCopper, Wood,
};

/// The varnished board's package source.
pub const VARNISHED_BOARD: &str = include_str!("../packages/varnished_board.json");
/// [`WeatheredCopper`]'s package source, with presets `bright`,
/// `tarnished`, `streaked` and `verdigris`.
pub const WEATHERED_COPPER: &str = include_str!("../packages/weathered_copper.json");
/// [`RustedSteel`]'s package source, with presets `fresh`, `flash_rusted`,
/// `rusting` and `pitted`.
pub const RUSTED_STEEL: &str = include_str!("../packages/rusted_steel.json");

/// Every package this crate ships, as source. A package added to the
/// crate belongs here, so its golden digests are checked.
pub const PACKAGES: &[&str] = &[VARNISHED_BOARD, WEATHERED_COPPER, RUSTED_STEEL];

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
        .with(Arc::new(Copper))
        .with(Arc::new(Steel))
        .with(Arc::new(Patina))
        .with(Arc::new(Rust))
        .with(Arc::new(WeatheredCopper))
        .with(Arc::new(RustedSteel))
}
