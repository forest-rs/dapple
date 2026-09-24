// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Calibration of the bark and masonry modules against measured
//! reflectance, with a plausibility gate from a second source.
//!
//! Each module's default color (and, where a measured value exists, its
//! roughness) is fitted with `dapple_lab::fit` (the `library_swatches`
//! example reruns the fit and rederives every number here from the
//! spectra it embeds) so that the module's **mean linear base color** over
//! the measured surface, under the stated conditions, matches a
//! **reference**. Spectra are integrated from 400 to 700 nm in 10 nm steps
//! against the CIE 1931 2° color-matching functions under CIE illuminant
//! D65 and converted to linear sRGB (Rec. 709); the same code reproduces
//! the published sRGB of Macbeth color-checker patches measured by Ohta (1997) to
//! within 6/255.
//!
//! Every reference, and every calibrated default, must also fall inside a
//! **plausible range** (luminance, and for fired clay the red-to-blue
//! ratio) drawn from a second source. Where no independent second source
//! was found the range says so ([`Plausible::independent`]); where sources
//! disagree the disagreement is listed below rather than resolved
//! silently.
//!
//! | Module | Reference (conditions) | Linear RGB | Plausible range, second source |
//! |---|---|---|---|
//! | [`super::Beech`] | Grey alder (*Alnus incana*) stem bark (ref. 1): no beech spectrum was found; alder's is the nearest smooth grey bark measured. Luminance only. | Y 0.235 | Y 0.2–0.3: the smooth barks of ref. 1 (not independent) |
//! | [`super::Birch`] | The whitest quarter of the 20 silver birch trees of ref. 1 (white bark, 6 m up the stem) | (0.376, 0.319, 0.232) | Y 0.22–0.6: ref. 2's laboratory reflectance of birch stem sections from 1 to 10 m |
//! | [`super::ScotsPine`] | Scots pine stem bark, mean of 20 trees (ref. 1; 1.3 m) | (0.150, 0.118, 0.089) | Y 0.07–0.25: ref. 2 |
//! | [`super::Spruce`] | Norway spruce stem bark, mean of 20 trees (ref. 1; 1.3 m) | (0.162, 0.122, 0.075) | Y 0.07–0.25: ref. 2 |
//! | [`super::AshlarLimestone`] | Oolitic limestone, Ward's sample 72 (ref. 3; the blocks) | (0.429, 0.361, 0.282) | Y 0.24–0.64: ten USGS limestones (ref. 3, a different laboratory) |
//! | [`super::RubbleWall`] | Crinoidal limestone, Ward's sample 66 (ref. 3; the stones) | (0.306, 0.255, 0.189) | the same |
//! | [`super::FlintWall`] | Black chert, "generally <10 % reflectance" (ref. 4; the faces): an upper bound | Y ≤ 0.10 | none found (not independent) |
//! | [`super::RomanBrick`] | Weathered and bare red brick, JHU samples 0412 and 0413, averaged (ref. 3; the bricks) | (0.208, 0.068, 0.035) | Y 0.07–0.2, red/blue 3–8: Physically Based's brick (ref. 5) |
//! | [`super::Marble`] | White construction marble, Naxos, JHU sample 0722 (ref. 3; the slabs) | (0.826, 0.788, 0.674) | Y 0.6–0.95: Physically Based's marble (ref. 5) |
//! | [`super::TerracottaTile`] | Fired clay: the red bricks above (ref. 3; the tiles) | (0.208, 0.068, 0.035) | as Roman brick |
//!
//! **Disagreements and rejections.**
//!
//! - *Silver birch.* White birch bark is often quoted at 0.4–0.6 visible
//!   reflectance. The field measurements of ref. 1 (breast height, north
//!   side, trees at least 15 cm across) spread from Y 0.08 to 0.37, median
//!   0.18: at breast height many stems already show dark bark. The module
//!   therefore calibrates its white bark to the whitest quarter (Y 0.325)
//!   and checks that its breast-height mean on a default stem falls inside
//!   the measured interquartile range (0.12–0.28). Ref. 2's laboratory
//!   values span 0.22–0.55 over 400–1000 nm. Neither measurement supports
//!   0.4–0.6 in the visible; a brighter white would have to come from a
//!   source not found here.
//! - *Terracotta tile.* The only measured roofing tile (JHU sample 0484,
//!   "weathered", ref. 3) is nearly neutral (red/blue 1.7) where fired
//!   clay in two sources is strongly red (red/blue 4–6). It fails the
//!   plausibility gate and is rejected as soiled; the tiles calibrate to
//!   the measured red brick, the same fired clay.
//!
//! **Roughness.** Published microfacet roughness for these materials is
//! scarce. Fired clay (Roman brick, terracotta) is fitted to 0.9, the
//! roughness Physically Based (ref. 5) lists for brick; the other
//! roughnesses are authored, not measured: matte stone and bark around
//! 0.8–0.9, birch's papery bark and beech's smooth bark a little lower,
//! knapped flint glassy, and marble set by its `polish`.
//!
//! The bark references of ref. 1 are field measurements of
//! hemispherical-directional reflectance, which includes the bark's own
//! shadowing; they are compared with the mean base color, so relief-heavy
//! barks are, if anything, calibrated slightly dark. The stone references
//! are laboratory directional-hemispherical reflectance of solid samples.
//!
//! 1. Juola, J., Hovi, A. and Rautiainen, M. (2022). A spectral analysis of
//!    stem bark for boreal and temperate tree species. *Ecology and
//!    Evolution* 12(3), e8718. <https://doi.org/10.1002/ece3.8718>. Data:
//!    Mendeley Data, V2, <https://doi.org/10.17632/pwfxgzz5fj.2>.
//! 2. Juola, J., Hovi, A. and Rautiainen, M. (2020). Multiangular spectra
//!    of tree bark for common boreal tree species in Europe. *Silva
//!    Fennica* 54(4), 10331. <https://doi.org/10.14214/sf.10331>.
//! 3. Meerdink, S. K., Hook, S. J., Roberts, D. A. and Abbott, E. A.
//!    (2019). The ECOSTRESS spectral library version 1.0. *Remote Sensing
//!    of Environment* 230, 111196; spectra from the JPL, JHU and USGS
//!    libraries it includes (Baldridge et al., 2009, the ASTER spectral
//!    library version 2.0), <https://speclib.jpl.nasa.gov>.
//! 4. Brown, A. J., Walter, M. and Cudahy, T. SWIR investigation of sites
//!    of astrobiological interest. *Astrobiology* 4(3), 359–376 (2004);
//!    <https://arxiv.org/abs/1401.4771>.
//! 5. Physically Based, a database of physically based values for CG
//!    artists: "Brick" and "Marble". <https://physicallybased.info>.

use glam::Vec3;

/// [`super::Beech`]'s calibrated grey.
pub const BEECH_COLOR: Vec3 = Vec3::new(0.254, 0.245, 0.217);
/// [`super::Birch`]'s calibrated white.
pub const BIRCH_COLOR: Vec3 = Vec3::new(0.391, 0.331, 0.24);
/// [`super::ScotsPine`]'s calibrated plate color.
pub const PINE_COLOR: Vec3 = Vec3::new(0.167, 0.14, 0.109);
/// [`super::Spruce`]'s calibrated scale color.
pub const SPRUCE_COLOR: Vec3 = Vec3::new(0.22, 0.166, 0.101);
/// [`super::AshlarLimestone`]'s calibrated stone color.
pub const ASHLAR_COLOR: Vec3 = Vec3::new(0.42, 0.353, 0.275);
/// [`super::RubbleWall`]'s calibrated stone color.
pub const RUBBLE_COLOR: Vec3 = Vec3::new(0.3, 0.249, 0.184);
/// [`super::FlintWall`]'s calibrated flint color.
pub const FLINT_COLOR: Vec3 = Vec3::new(0.036, 0.036, 0.04);
/// [`super::RomanBrick`]'s calibrated brick color.
pub const ROMAN_BRICK_COLOR: Vec3 = Vec3::new(0.196, 0.057, 0.026);
/// [`super::RomanBrick`]'s calibrated roughness.
pub const ROMAN_BRICK_ROUGHNESS: f32 = 0.898;
/// [`super::Marble`]'s calibrated ground color.
pub const MARBLE_COLOR: Vec3 = Vec3::new(0.909, 0.863, 0.729);
/// [`super::TerracottaTile`]'s calibrated tile color.
pub const TERRACOTTA_COLOR: Vec3 = Vec3::new(0.206, 0.076, 0.042);
/// [`super::TerracottaTile`]'s calibrated roughness.
pub const TERRACOTTA_ROUGHNESS: f32 = 0.9;

/// A plausible range from a second source: see the [module docs](self).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Plausible {
    /// Rec. 709 luminance, inclusive.
    pub luminance: [f32; 2],
    /// Red over blue, inclusive, where the hue is diagnostic.
    pub red_over_blue: Option<[f32; 2]>,
    /// Whether the range comes from a source independent of the
    /// reference.
    pub independent: bool,
}

impl Plausible {
    /// Whether linear color `c` lies inside.
    #[must_use]
    pub fn holds(&self, c: Vec3) -> bool {
        let y = luminance(c);
        let hue = self
            .red_over_blue
            .is_none_or(|[lo, hi]| c.z > 0.0 && (lo..=hi).contains(&(c.x / c.z)));
        (self.luminance[0]..=self.luminance[1]).contains(&y) && hue
    }
}

/// What a module's calibration measures and aims for.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Reference {
    /// The module's name, as in its identity.
    pub module: &'static str,
    /// Parameters the measurement's conditions set, such as the height on
    /// the trunk.
    pub conditions: &'static [(&'static str, f32)],
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
    /// The plausibility gate.
    pub plausible: Plausible,
}

const fn reference(module: &'static str, albedo: [f32; 3], plausible: Plausible) -> Reference {
    Reference {
        module,
        conditions: &[],
        albedo: Vec3::new(albedo[0], albedo[1], albedo[2]),
        luminance_only: false,
        at_most: false,
        roughness: None,
        units_only: false,
        plausible,
    }
}

const BARK: Plausible = Plausible {
    luminance: [0.07, 0.25],
    red_over_blue: None,
    independent: true,
};
const LIMESTONE: Plausible = Plausible {
    luminance: [0.24, 0.64],
    red_over_blue: None,
    independent: true,
};
const FIRED_CLAY: Plausible = Plausible {
    luminance: [0.07, 0.2],
    red_over_blue: Some([3.0, 8.0]),
    independent: true,
};
const RED_BRICK: [f32; 3] = [0.208, 0.068, 0.035];

/// The references of the table above, in its order.
pub const REFERENCES: [Reference; 10] = [
    Reference {
        luminance_only: true,
        ..reference(
            "dapple_library.beech_bark",
            [0.235, 0.235, 0.235],
            Plausible {
                luminance: [0.2, 0.3],
                red_over_blue: None,
                independent: false,
            },
        )
    },
    Reference {
        conditions: &[("height", 6.0)],
        ..reference(
            "dapple_library.birch_bark",
            [0.376, 0.319, 0.232],
            Plausible {
                luminance: [0.22, 0.6],
                red_over_blue: None,
                independent: true,
            },
        )
    },
    reference(
        "dapple_library.scots_pine_bark",
        [0.150, 0.118, 0.089],
        BARK,
    ),
    reference("dapple_library.spruce_bark", [0.162, 0.122, 0.075], BARK),
    Reference {
        units_only: true,
        ..reference(
            "dapple_library.ashlar_limestone",
            [0.429, 0.361, 0.282],
            LIMESTONE,
        )
    },
    Reference {
        units_only: true,
        ..reference(
            "dapple_library.rubble_wall",
            [0.306, 0.255, 0.189],
            LIMESTONE,
        )
    },
    Reference {
        units_only: true,
        luminance_only: true,
        at_most: true,
        ..reference(
            "dapple_library.flint_wall",
            [0.10, 0.10, 0.10],
            Plausible {
                luminance: [0.0, 0.1],
                red_over_blue: None,
                independent: false,
            },
        )
    },
    Reference {
        units_only: true,
        roughness: Some(0.9),
        ..reference("dapple_library.roman_brick", RED_BRICK, FIRED_CLAY)
    },
    Reference {
        units_only: true,
        ..reference(
            "dapple_library.marble",
            [0.826, 0.788, 0.674],
            Plausible {
                luminance: [0.6, 0.95],
                red_over_blue: None,
                independent: true,
            },
        )
    },
    Reference {
        units_only: true,
        roughness: Some(0.9),
        ..reference("dapple_library.terracotta_tile", RED_BRICK, FIRED_CLAY)
    },
];

/// A measured sample the plausibility gate rejected, kept so the rejection
/// stays visible and tested.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rejected {
    /// What was measured.
    pub sample: &'static str,
    /// Its linear color.
    pub albedo: Vec3,
    /// The gate it fails.
    pub plausible: Plausible,
}

/// The rejected samples: see the [module docs](self).
pub const REJECTED: [Rejected; 1] = [Rejected {
    sample: "weathered terracotta roofing tile, JHU 0484 (ECOSTRESS)",
    albedo: Vec3::new(0.147, 0.110, 0.085),
    plausible: FIRED_CLAY,
}];

/// The interquartile range of silver birch's breast-height luminance
/// across the 20 trees of ref. 1, which a default birch stem's
/// breast-height mean must fall in.
pub const BIRCH_BREAST_HEIGHT: [f32; 2] = [0.12, 0.284];

/// Rec. 709 luminance of a linear color.
#[must_use]
pub fn luminance(c: Vec3) -> f32 {
    c.dot(Vec3::new(0.2126, 0.7152, 0.0722))
}
