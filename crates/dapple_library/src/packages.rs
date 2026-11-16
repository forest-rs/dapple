// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The library as a package engine: every module registered, and packages
//! written against them.
//!
//! [`registry`] offers every module in [`crate::modules`] to
//! [`dapple_package`] packages. [`VARNISHED_BOARD`] is
//! [`crate::modules::VarnishedBoard`] written as a package: the same
//! modules under the same instance names, so it realizes the same bits as
//! the native module, and, like it, makes its traffic patches and wear only
//! when `wear > 0.0` (guarded steps; the wear's `otherwise` passes the
//! varnish through). So are the metals: [`WEATHERED_COPPER`],
//! [`RUSTED_STEEL`] and [`PAINTED_STEEL`], each with presets from fresh to
//! heavily weathered. [`GLAZED_BRICK_WALL`] assembles brick maps and
//! material operations before its guarded weathering layers.

use alloc::sync::Arc;

use dapple_package::Registry;

use crate::modules::{
    AshlarLimestone, Beech, Birch, ByExample, CeramicBody, Copper, EdgeWear, Efflorescence, Finish,
    FlintWall, GlazedBrickWall, GlazedBricks, Grime, HeightDetail, Marble, Mortar, Moss, Paint,
    PaintedSteel, Patches, Patina, RomanBrick, RubbleWall, Rust, RustedSteel, ScotsPine,
    SelectByMask, SetRegion, Spruce, Steel, Stone, StoneSill, Streaks, TerracottaTile, Threshold,
    VarnishedBoard, WeatheredCopper, Wood,
};

/// The varnished board's package source, with preset `dark_and_worn`. Its
/// traffic patches and edge wear are guarded on `wear > 0.0`.
pub const VARNISHED_BOARD: &str = include_str!("../packages/varnished_board.json");
/// [`WeatheredCopper`]'s package source, with presets `bright`,
/// `tarnished`, `streaked` and `verdigris`.
pub const WEATHERED_COPPER: &str = include_str!("../packages/weathered_copper.json");
/// [`RustedSteel`]'s package source, with presets `fresh`, `flash_rusted`,
/// `rusting` and `pitted`.
pub const RUSTED_STEEL: &str = include_str!("../packages/rusted_steel.json");
/// [`PaintedSteel`]'s package source, with presets `new`, `chipped`,
/// `weeping`, `weathered` and `derelict`. Its `weather` knob derives the
/// paint's gloss, chipping and fade and the rust's age through expression
/// bindings; `weathered` sets nothing else.
pub const PAINTED_STEEL: &str = include_str!("../packages/painted_steel.json");

/// [`GlazedBrickWall`]'s package source, with presets `clean`, `weathered`
/// and `mossy`. Dirt, streaks and salts run only when `weathered`; moss
/// runs only when `weathered && moss > 0.0`. Each skipped layer passes its
/// base through with `otherwise`.
pub const GLAZED_BRICK_WALL: &str = include_str!("../packages/glazed_brick_wall.json");

/// Every package this crate ships, as source. A package added to the
/// crate belongs here, so its golden digests are checked.
pub const PACKAGES: &[&str] = &[
    VARNISHED_BOARD,
    WEATHERED_COPPER,
    RUSTED_STEEL,
    PAINTED_STEEL,
    GLAZED_BRICK_WALL,
];

/// An engine offering this crate's modules and `dapple_package`'s own
/// capabilities.
#[must_use]
pub fn registry() -> Registry {
    Registry::new()
        .with(Arc::new(GlazedBricks))
        .with(Arc::new(HeightDetail))
        .with(Arc::new(SelectByMask))
        .with(Arc::new(SetRegion))
        .with(Arc::new(CeramicBody))
        .with(Arc::new(Mortar))
        .with(Arc::new(Stone))
        .with(Arc::new(Wood))
        .with(Arc::new(Finish))
        .with(Arc::new(Grime))
        .with(Arc::new(Streaks))
        .with(Arc::new(Efflorescence))
        .with(Arc::new(EdgeWear))
        .with(Arc::new(Patches))
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
        .with(Arc::new(Paint))
        .with(Arc::new(WeatheredCopper))
        .with(Arc::new(RustedSteel))
        .with(Arc::new(PaintedSteel))
}
