// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Calibration of the bark and masonry modules against measured
//! reflectance.
//!
//! Each module's default color (and, where a measured value exists, its
//! roughness) was fitted with `dapple_lab::fit` (the `library_swatches`
//! example reruns the fit) so that the module's **mean linear base color**
//! over the measured surface matches the reference below, at the stated
//! conditions. Reference spectra were integrated over 400–700 nm against
//! the CIE 1931 2° observer (Wyman, Sloan and Shirley's 2013 analytic fit)
//! under a D65-like 6504 K illuminant and converted to linear Rec. 709,
//! white-balanced so a perfect reflector is (1, 1, 1).
//!
//! | Module | Reference | Linear RGB | Conditions |
//! |---|---|---|---|
//! | [`super::Beech`] | Grey alder (*Alnus incana*) stem bark, luminance only (no beech spectrum was found; alder's is the nearest smooth grey bark measured) (ref. 1) | Y 0.235 | breast height, default girth |
//! | [`super::Birch`] | Silver birch (*Betula pendula*) stem bark (ref. 1) | (0.221, 0.188, 0.136) | 1.3 m, girth 0.9 m |
//! | [`super::ScotsPine`] | Scots pine (*Pinus sylvestris*) stem bark (ref. 1) | (0.149, 0.119, 0.088) | 1.3 m, girth 1.1 m |
//! | [`super::Spruce`] | Norway spruce (*Picea abies*) stem bark (ref. 1) | (0.160, 0.122, 0.075) | 1.3 m, girth 1.0 m |
//! | [`super::AshlarLimestone`] | Oolitic limestone, Ward's sample 72 (ref. 2) | (0.427, 0.362, 0.282) | the blocks |
//! | [`super::RubbleWall`] | Crinoidal ("ecrinal") limestone, Ward's sample 66 (ref. 2) | (0.304, 0.255, 0.188) | the stones |
//! | [`super::FlintWall`] | Black chert, "generally <10 % reflectance" (short-wave infrared; no visible spectrum of flint was found, and black chert is black to the eye, so the bound is carried over as an upper bound) (ref. 3) | Y ≤ 0.10 | the knapped faces |
//! | [`super::RomanBrick`] | Weathered and bare red brick, JHU samples 0412 and 0413, averaged (ref. 2) | (0.204, 0.069, 0.035) | the bricks |
//! | [`super::Marble`] | White weathered construction marble (Naxos), JHU sample 0722 (ref. 2) | (0.825, 0.788, 0.673) | the slabs |
//! | [`super::TerracottaTile`] | Weathered terracotta roofing tile, JHU sample 0484 (ref. 2) | (0.146, 0.110, 0.085) | the tiles |
//!
//! **Roughness.** Published microfacet roughness for these materials is
//! scarce. Fired clay (Roman brick, terracotta) is fitted to 0.9, the
//! roughness Physically Based (ref. 4) lists for brick; the other roughnesses
//! are authored, not measured: matte stone and bark around 0.8–0.9,
//! birch's papery bark and beech's smooth bark a little lower, knapped
//! flint glassy, and marble set by its `polish`.
//!
//! The bark references are field measurements at breast height (1.3 m) on
//! stems at least 15 cm across, of hemispherical-directional reflectance,
//! which includes the bark's own shadowing; they are compared with the
//! mean base color, so relief-heavy barks are, if anything, calibrated
//! slightly dark. The stone references are laboratory
//! directional-hemispherical reflectance of solid samples.
//!
//! 1. Juola, J., Hovi, A. and Rautiainen, M. (2022). A spectral analysis of
//!    stem bark for boreal and temperate tree species. *Ecology and
//!    Evolution* 12(3), e8718. <https://doi.org/10.1002/ece3.8718>. Data:
//!    Mendeley Data, V2, <https://doi.org/10.17632/pwfxgzz5fj.2> (the
//!    species' mean spectra).
//! 2. Meerdink, S. K., Hook, S. J., Roberts, D. A. and Abbott, E. A.
//!    (2019). The ECOSTRESS spectral library version 1.0. *Remote Sensing
//!    of Environment* 230, 111196; spectra from the JPL and JHU libraries
//!    it includes (Baldridge et al., 2009, the ASTER spectral library
//!    version 2.0), <https://speclib.jpl.nasa.gov>.
//! 3. Brown, A. J., Walter, M. and Cudahy, T. SWIR investigation of sites
//!    of astrobiological interest. *Astrobiology* 4(3), 359–376 (2004);
//!    <https://arxiv.org/abs/1401.4771>.
//! 4. Physically Based, a database of physically based values for CG
//!    artists: "Brick". <https://physicallybased.info>.

use glam::Vec3;

/// [`super::Beech`]'s calibrated grey.
pub const BEECH_COLOR: Vec3 = Vec3::new(0.245, 0.236, 0.21);
/// [`super::Birch`]'s calibrated white.
pub const BIRCH_COLOR: Vec3 = Vec3::new(0.233, 0.198, 0.143);
/// [`super::ScotsPine`]'s calibrated plate color.
pub const PINE_COLOR: Vec3 = Vec3::new(0.145, 0.145, 0.118);
/// [`super::Spruce`]'s calibrated scale color.
pub const SPRUCE_COLOR: Vec3 = Vec3::new(0.154, 0.129, 0.082);
/// [`super::AshlarLimestone`]'s calibrated stone color.
pub const ASHLAR_COLOR: Vec3 = Vec3::new(0.417, 0.354, 0.276);
/// [`super::RubbleWall`]'s calibrated stone color.
pub const RUBBLE_COLOR: Vec3 = Vec3::new(0.298, 0.25, 0.183);
/// [`super::FlintWall`]'s calibrated flint color.
pub const FLINT_COLOR: Vec3 = Vec3::new(0.05, 0.05, 0.055);
/// [`super::RomanBrick`]'s calibrated brick color.
pub const ROMAN_BRICK_COLOR: Vec3 = Vec3::new(0.192, 0.058, 0.026);
/// [`super::RomanBrick`]'s calibrated roughness.
pub const ROMAN_BRICK_ROUGHNESS: f32 = 0.898;
/// [`super::Marble`]'s calibrated ground color.
pub const MARBLE_COLOR: Vec3 = Vec3::new(0.908, 0.863, 0.728);
/// [`super::TerracottaTile`]'s calibrated tile color.
pub const TERRACOTTA_COLOR: Vec3 = Vec3::new(0.136, 0.116, 0.097);
/// [`super::TerracottaTile`]'s calibrated roughness.
pub const TERRACOTTA_ROUGHNESS: f32 = 0.901;

/// What a module's calibration measures and aims for.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Reference {
    /// The module's name, as in its identity.
    pub module: &'static str,
    /// The target mean linear base color, or only its luminance (as the
    /// color's gray) when `luminance_only`.
    pub albedo: Vec3,
    /// Whether only the luminance is matched.
    pub luminance_only: bool,
    /// Whether the albedo is an upper bound rather than a value.
    pub at_most: bool,
    /// The target mean roughness, where a published value exists.
    pub roughness: Option<f32>,
    /// Whether the mean is over the units only (masonry) rather than the
    /// whole tile.
    pub units_only: bool,
}

const fn reference(module: &'static str, albedo: [f32; 3]) -> Reference {
    Reference {
        module,
        albedo: Vec3::new(albedo[0], albedo[1], albedo[2]),
        luminance_only: false,
        at_most: false,
        roughness: None,
        units_only: false,
    }
}

/// The references of the table above, in its order.
pub const REFERENCES: [Reference; 10] = [
    Reference {
        luminance_only: true,
        ..reference("dapple_library.beech_bark", [0.235, 0.235, 0.235])
    },
    reference("dapple_library.birch_bark", [0.221, 0.188, 0.136]),
    reference("dapple_library.scots_pine_bark", [0.149, 0.119, 0.088]),
    reference("dapple_library.spruce_bark", [0.160, 0.122, 0.075]),
    Reference {
        units_only: true,
        ..reference("dapple_library.ashlar_limestone", [0.427, 0.362, 0.282])
    },
    Reference {
        units_only: true,
        ..reference("dapple_library.rubble_wall", [0.304, 0.255, 0.188])
    },
    Reference {
        units_only: true,
        luminance_only: true,
        at_most: true,
        ..reference("dapple_library.flint_wall", [0.10, 0.10, 0.10])
    },
    Reference {
        units_only: true,
        roughness: Some(0.9),
        ..reference("dapple_library.roman_brick", [0.204, 0.069, 0.035])
    },
    Reference {
        units_only: true,
        ..reference("dapple_library.marble", [0.825, 0.788, 0.673])
    },
    Reference {
        units_only: true,
        roughness: Some(0.9),
        ..reference("dapple_library.terracotta_tile", [0.146, 0.110, 0.085])
    },
];

/// Rec. 709 luminance of a linear color.
#[must_use]
pub fn luminance(c: Vec3) -> f32 {
    c.dot(Vec3::new(0.2126, 0.7152, 0.0722))
}
