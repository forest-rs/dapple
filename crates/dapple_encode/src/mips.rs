// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Per-kind mip rules.

use alloc::vec::Vec;

use dapple_field::ScalarField;
use dapple_raster::{Realization, realize};

use crate::filter::{Filter, downsample, downsample_majority, next_size};
use crate::{EncodeError, Image};

/// Mip levels of one image, largest first; each level halves the one above
/// (rounding down, at least 1) until 1×1.
#[derive(Clone, Debug, PartialEq)]
pub struct MipChain {
    levels: Vec<Image>,
}

impl MipChain {
    fn build(level0: Image, mut next: impl FnMut(&Image) -> Image) -> Self {
        let mut levels = Vec::new();
        let mut current = level0;
        loop {
            let done = current.width == 1 && current.height == 1;
            let following = (!done).then(|| next(&current));
            levels.push(current);
            match following {
                Some(image) => current = image,
                None => break,
            }
        }
        Self { levels }
    }

    /// The levels, largest first.
    #[must_use]
    pub fn levels(&self) -> &[Image] {
        &self.levels
    }

    /// Level 0.
    #[must_use]
    pub fn base(&self) -> &Image {
        &self.levels[0]
    }

    pub(crate) fn from_levels(levels: Vec<Image>) -> Self {
        Self { levels }
    }

    fn map(&self, f: impl FnMut(&Image) -> Image) -> Self {
        Self {
            levels: self.levels.iter().map(f).collect(),
        }
    }
}

/// Mips of non-color data (heights, occlusion, metalness), filtered as-is.
#[must_use]
pub fn data_mips(image: &Image, filter: Filter) -> MipChain {
    MipChain::build(image.clone(), |level| downsample(level, filter))
}

/// Mips of a single-channel identifier image (region or cell IDs stored as
/// exact integers in `f32`).
///
/// Each level takes, per texel, the identifier covering most of its source
/// area, ties going to the smaller one: identifiers are never blended, so no
/// level invents an ID that is not in the one above.
pub fn id_mips(image: &Image) -> Result<MipChain, EncodeError> {
    if image.channels != 1 {
        return Err(EncodeError::ChannelMismatch {
            role: "id",
            expected: 1,
            found: image.channels,
        });
    }
    Ok(MipChain::build(image.clone(), downsample_majority))
}

/// Mips of an undirected-direction image: two channels holding the
/// doubled-angle vector `(cos 2θ, sin 2θ)` of `dapple_field`'s
/// `PortType::Direction`.
///
/// Averaging in that space is the correct rule: it is sign-free, so `θ` and
/// `θ + π` reinforce instead of cancelling, and the averaged vector's length
/// falls where the directions disagree. Levels keep the unnormalized average
/// so packing can scale anisotropy by that agreement.
pub fn direction_mips(image: &Image, filter: Filter) -> Result<MipChain, EncodeError> {
    if image.channels != 2 {
        return Err(EncodeError::ChannelMismatch {
            role: "direction",
            expected: 2,
            found: image.channels,
        });
    }
    Ok(data_mips(image, filter))
}

/// Mips realized from fields, one channel per field, instead of filtered.
///
/// Level `k` realizes every channel on `realization` resized to the chain's
/// `k`-th size, so each level is evaluated with its own texel as the
/// footprint: band-limited fields drop exactly the detail that level cannot
/// hold, with no filter blur or ringing. Level sizes and the edge policy
/// match [`data_mips`] of level 0. Channels must number 1 to 4.
pub fn field_mips<F: ScalarField>(
    channels: &[F],
    realization: Realization,
) -> Result<MipChain, EncodeError> {
    if !(1..=4).contains(&channels.len()) {
        return Err(EncodeError::InvalidChannels(channels.len()));
    }
    let level = |realization: Realization| -> Result<Image, EncodeError> {
        let rasters = channels
            .iter()
            .map(|field| realize(field, realization))
            .collect::<Result<Vec<_>, _>>()?;
        let first = &rasters[0];
        let texels = first.values().len();
        let mut values = Vec::with_capacity(texels * rasters.len());
        for i in 0..texels {
            values.extend(rasters.iter().map(|r| r.values()[i]));
        }
        Image::new(
            first.width(),
            first.height(),
            rasters.len(),
            first.edge(),
            values,
        )
    };
    let mut levels = alloc::vec![level(realization)?];
    let (mut width, mut height) = (realization.width(), realization.height());
    while width > 1 || height > 1 {
        width = next_size(width);
        height = next_size(height);
        levels.push(level(realization.resized(width, height)?)?);
    }
    Ok(MipChain { levels })
}

/// Mips of a linear color image.
///
/// With 4 channels the fourth is straight (unpremultiplied) alpha: each level
/// is filtered premultiplied and divided back, so the color of transparent
/// texels does not bleed. A texel whose filtered alpha is zero keeps color
/// zero. Other channel counts filter every channel as-is.
#[must_use]
pub fn color_mips(image: &Image, filter: Filter) -> MipChain {
    if image.channels != 4 {
        return data_mips(image, filter);
    }
    let premultiply = |img: &Image| {
        let mut out = img.clone();
        for t in out.values.chunks_exact_mut(4) {
            for c in 0..3 {
                t[c] *= t[3];
            }
        }
        out
    };
    let unpremultiply = |img: &Image| {
        let mut out = img.clone();
        for t in out.values.chunks_exact_mut(4) {
            for c in 0..3 {
                t[c] = if t[3] > 0.0 { t[c] / t[3] } else { 0.0 };
            }
        }
        out
    };
    let chain = MipChain::build(premultiply(image), |level| downsample(level, filter));
    let mut chain = chain.map(unpremultiply);
    // Level 0 is the input exactly, not a premultiply round trip.
    chain.levels[0] = image.clone();
    chain
}

/// Coverage of one level after [`preserve_coverage`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CoverageLevel {
    /// Fraction of texels at or above the cutoff before scaling.
    pub filtered: f32,
    /// Fraction after scaling; level 0's value is the target.
    pub preserved: f32,
    /// The factor alpha was multiplied by (then clamped to 1).
    pub scale: f32,
}

/// Rescales the alpha channel of each level so the fraction of texels at or
/// above `cutoff` matches level 0 (Castaño, "Computing Alpha Mipmaps", 2010).
///
/// `channel` selects the alpha channel. Each level's scale is the smallest
/// factor, found by a fixed 40-step bisection over `[0, 1024]`, whose coverage
/// reaches level 0's; values are clamped to 1 after scaling. Returns the
/// per-level coverage for reports.
pub fn preserve_coverage(
    chain: &mut MipChain,
    channel: usize,
    cutoff: f32,
) -> Result<Vec<CoverageLevel>, EncodeError> {
    if !(cutoff > 0.0 && cutoff < 1.0) {
        return Err(EncodeError::InvalidParameter { name: "cutoff" });
    }
    let channels = chain.base().channels;
    if channel >= channels {
        return Err(EncodeError::InvalidParameter { name: "channel" });
    }
    let coverage = |img: &Image, scale: f32| {
        let above = img
            .values
            .chunks_exact(channels)
            .filter(|t| t[channel] * scale >= cutoff)
            .count();
        above as f32 / img.texels() as f32
    };
    let target = coverage(chain.base(), 1.0);
    let mut report = Vec::with_capacity(chain.levels.len());
    report.push(CoverageLevel {
        filtered: target,
        preserved: target,
        scale: 1.0,
    });
    for level in chain.levels.iter_mut().skip(1) {
        let filtered = coverage(level, 1.0);
        let (mut lo, mut hi) = (0.0_f32, 1024.0_f32);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if coverage(level, mid) >= target {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let scale = hi;
        for t in level.values.chunks_exact_mut(channels) {
            t[channel] = (t[channel] * scale).min(1.0);
        }
        report.push(CoverageLevel {
            filtered,
            preserved: coverage(level, 1.0),
            scale,
        });
    }
    Ok(report)
}

/// Normal and roughness mips from [`normal_mips`].
#[derive(Clone, Debug, PartialEq)]
pub struct NormalChain {
    /// Unit normals per level, in the input's frame.
    pub normals: MipChain,
    /// `specular_roughness` per level, widened by the normals' variance.
    pub roughness: MipChain,
    /// Largest variance added on each level.
    pub max_variance: Vec<f32>,
}

/// Normal mips with the normals' variance moved into roughness.
///
/// Levels filter the *unnormalized* averages down the chain, so each level
/// holds the average of the level-0 unit normals it covers. That average's
/// length `L ≤ 1` shrinks as the normals disagree. The stored normal is the
/// average renormalized, and the lost length becomes variance by Toksvig's
/// estimate
/// `σ² = (1 − L) / L`, added to the GGX `α²` of the roughness:
/// `α′² = min(α² + σ², 1)`, with OpenPBR's `α = specular_roughness²`.
/// Roughness itself is filtered in `α²`, the space in which widths add, so
/// each level's variance accounts for all normal detail lost since level 0.
///
/// `roughness` is a single-channel `specular_roughness` image, or `None` with
/// `constant` used everywhere. A zero-length average keeps `(0, 0, 1)` and
/// full roughness.
pub fn normal_mips(
    normals: &Image,
    roughness: Option<&Image>,
    constant: f32,
    filter: Filter,
) -> Result<NormalChain, EncodeError> {
    if normals.channels != 3 {
        return Err(EncodeError::ChannelMismatch {
            role: "normal",
            expected: 3,
            found: normals.channels,
        });
    }
    if !(0.0..=1.0).contains(&constant) {
        return Err(EncodeError::InvalidParameter {
            name: "specular_roughness",
        });
    }
    let base_roughness = match roughness {
        Some(r) if r.channels != 1 => {
            return Err(EncodeError::ChannelMismatch {
                role: "specular_roughness",
                expected: 1,
                found: r.channels,
            });
        }
        Some(r) if !r.same_grid(normals) => {
            return Err(EncodeError::Incompatible {
                role: "specular_roughness",
            });
        }
        Some(r) => r.clone(),
        None => Image {
            values: alloc::vec![constant; normals.texels()],
            channels: 1,
            ..normals.clone()
        },
    };
    let alpha_sq = |r: f32| {
        let a = r * r;
        a * a
    };
    let alpha_sq_image = Image {
        values: base_roughness.values.iter().map(|&r| alpha_sq(r)).collect(),
        ..base_roughness.clone()
    };

    // Filter the raw (unnormalized) average down the chain so each level's
    // length measures the spread of all level-0 normals it covers.
    let raw = data_mips(normals, filter);
    let alpha_sq_chain = data_mips(&alpha_sq_image, filter);
    let mut normal_levels = Vec::with_capacity(raw.levels.len());
    let mut roughness_levels = Vec::with_capacity(raw.levels.len());
    let mut max_variance = Vec::with_capacity(raw.levels.len());
    for (index, (n, a2)) in raw.levels.iter().zip(&alpha_sq_chain.levels).enumerate() {
        let mut unit = Vec::with_capacity(n.values.len());
        let mut rough = Vec::with_capacity(a2.values.len());
        let mut level_max = 0.0_f32;
        for (t, &a2) in n.values.chunks_exact(3).zip(&a2.values) {
            let len = libm::sqrtf(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
            let (normal, variance) = if index == 0 {
                // Level 0 is the input: its own spread is not measured here.
                let inv = if len > 0.0 { 1.0 / len } else { 0.0 };
                ([t[0] * inv, t[1] * inv, t[2] * inv], 0.0)
            } else if len > 1e-6 {
                let inv = 1.0 / len;
                let l = len.min(1.0);
                ([t[0] * inv, t[1] * inv, t[2] * inv], (1.0 - l) / l)
            } else {
                ([0.0, 0.0, 1.0], 1.0)
            };
            unit.extend_from_slice(&normal);
            level_max = level_max.max(variance);
            let widened = (a2 + variance).clamp(0.0, 1.0);
            rough.push(libm::sqrtf(libm::sqrtf(widened)));
        }
        max_variance.push(level_max);
        normal_levels.push(Image {
            values: unit,
            ..n.clone()
        });
        roughness_levels.push(Image {
            values: rough,
            ..a2.clone()
        });
    }
    // Level 0 roughness is the input exactly, not an α² round trip.
    roughness_levels[0] = base_roughness;
    Ok(NormalChain {
        normals: MipChain {
            levels: normal_levels,
        },
        roughness: MipChain {
            levels: roughness_levels,
        },
        max_variance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use dapple_raster::Edge;

    #[test]
    fn chains_halve_to_one_texel() {
        let image = Image::new(8, 3, 1, Edge::Clamp, vec![0.5; 24]).unwrap();
        let chain = data_mips(&image, Filter::Box);
        let sizes: Vec<_> = chain.levels().iter().map(|l| (l.width, l.height)).collect();
        assert_eq!(sizes, [(8, 3), (4, 1), (2, 1), (1, 1)]);
    }

    #[test]
    fn transparent_color_does_not_bleed() {
        // Two opaque red texels and two transparent green ones.
        let values = vec![
            1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        ];
        let image = Image::new(2, 2, 4, Edge::Wrap, values).unwrap();
        let chain = color_mips(&image, Filter::Box);
        assert_eq!(chain.levels()[1].values, [1.0, 0.0, 0.0, 0.5]);
        assert_eq!(chain.base(), &image);
    }

    #[test]
    fn coverage_survives_minification() {
        // Sparse leaves: one opaque texel in every 2×2 block, 25 % coverage.
        let values: Vec<f32> = (0..64)
            .map(|i| {
                if (i % 8) % 2 == 0 && (i / 8) % 2 == 0 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let image = Image::new(8, 8, 1, Edge::Wrap, values).unwrap();
        let mut chain = data_mips(&image, Filter::Box);
        // Plain filtering: every level-1 texel is 0.25, below a 0.5 cutoff.
        assert!(chain.levels()[1].values.iter().all(|&a| a == 0.25));
        let report = preserve_coverage(&mut chain, 0, 0.5).unwrap();
        assert_eq!(report[0].preserved, 0.25);
        assert_eq!(report[1].filtered, 0.0);
        // Uniform 0.25 cannot keep exactly a quarter: all or nothing. The
        // smallest scale reaching the target takes every texel.
        assert_eq!(report[1].preserved, 1.0);
        assert!(report.iter().all(|l| l.preserved >= report[0].preserved));
        assert!(preserve_coverage(&mut chain, 0, 1.0).is_err());
    }

    #[test]
    fn coverage_tracks_the_target_on_varied_alpha() {
        let values: Vec<f32> = (0..256).map(|i| ((i * 97) % 101) as f32 / 100.0).collect();
        let image = Image::new(16, 16, 1, Edge::Wrap, values).unwrap();
        let mut chain = data_mips(&image, Filter::Kaiser);
        let report = preserve_coverage(&mut chain, 0, 0.6).unwrap();
        let target = report[0].preserved;
        for level in &report[1..3] {
            assert!(level.preserved >= target, "{level:?}");
            assert!(level.preserved - target <= 0.25, "{level:?}");
        }
    }

    #[test]
    fn disagreeing_normals_widen_roughness() {
        let s = core::f32::consts::FRAC_1_SQRT_2;
        // Alternating ±45° normals about x.
        let values: Vec<f32> = (0..16)
            .flat_map(|i| {
                if i % 2 == 0 {
                    [s, 0.0, s]
                } else {
                    [-s, 0.0, s]
                }
            })
            .collect();
        let image = Image::new(4, 4, 3, Edge::Wrap, values).unwrap();
        let chain = normal_mips(&image, None, 0.3, Filter::Box).unwrap();
        let level1 = &chain.normals.levels()[1];
        assert!(
            level1
                .values
                .chunks_exact(3)
                .all(|n| n[0] == 0.0 && n[1] == 0.0 && (n[2] - 1.0).abs() < 1e-6)
        );
        // L = 1/√2, σ² = √2 − 1.
        let variance = core::f32::consts::SQRT_2 - 1.0;
        assert!((chain.max_variance[1] - variance).abs() < 1e-6);
        let expected = libm::sqrtf(libm::sqrtf(0.3_f32.powi(4) + variance));
        let rough = chain.roughness.levels()[1].values[0];
        assert!((rough - expected).abs() < 1e-6, "{rough} vs {expected}");
        assert_eq!(chain.roughness.base().values, vec![0.3; 16]);
    }

    #[test]
    fn flat_normals_keep_their_roughness() {
        let image = Image::new(4, 4, 3, Edge::Wrap, [0.0, 0.0, 1.0].repeat(16)).unwrap();
        let rough = Image::new(4, 4, 1, Edge::Wrap, vec![0.5; 16]).unwrap();
        let chain = normal_mips(&image, Some(&rough), 0.0, Filter::Kaiser).unwrap();
        for (level, variance) in chain.roughness.levels().iter().zip(&chain.max_variance) {
            assert_eq!(*variance, 0.0);
            assert!(level.values.iter().all(|r| (r - 0.5).abs() < 1e-6));
        }
        let wrong = Image::new(2, 2, 1, Edge::Wrap, vec![0.5; 4]).unwrap();
        assert!(normal_mips(&image, Some(&wrong), 0.0, Filter::Box).is_err());
    }

    #[test]
    fn field_levels_are_realized_at_their_own_footprint() {
        use dapple_field::program::{FieldProgram, Op, ProgramBuilder};
        use dapple_field::{Basis, Domain, FractalParams};

        let domain = Domain::periodic(1, 1).unwrap();
        let mut b = ProgramBuilder::new();
        let node = b
            .add(Op::Fractal {
                basis: Basis::Gradient,
                domain,
                frequency: [4.0, 4.0],
                seed: 3,
                params: FractalParams::default(),
            })
            .unwrap();
        let fbm = b.finish(node).unwrap();
        let realization = Realization::period(domain, 32, 16).unwrap();
        let chain = field_mips(core::slice::from_ref(&fbm), realization).unwrap();
        let sizes: Vec<_> = chain.levels().iter().map(|l| (l.width, l.height)).collect();
        assert_eq!(sizes, [(32, 16), (16, 8), (8, 4), (4, 2), (2, 1), (1, 1)]);
        for level in chain.levels() {
            let direct = realize(
                &fbm,
                realization.resized(level.width, level.height).unwrap(),
            )
            .unwrap();
            assert_eq!(level.values, direct.values());
            assert_eq!(level.edge, Edge::Wrap);
        }
        // Coarse levels hold less detail than filtering level 0 would.
        let spread = |image: &Image| {
            let mean = image.values.iter().sum::<f32>() / image.values.len() as f32;
            image
                .values
                .iter()
                .map(|v| (v - mean).abs())
                .fold(0.0, f32::max)
        };
        let filtered = data_mips(chain.base(), Filter::Box);
        assert!(spread(&chain.levels()[3]) <= spread(&filtered.levels()[3]) + 1e-6);
        assert!(field_mips::<FieldProgram>(&[], realization).is_err());
    }

    #[test]
    fn directions_average_sign_free() {
        // Angles 0 and π are one axis; 0 and π/2 cancel.
        let axis = |angle: f32| [libm::cosf(2.0 * angle), libm::sinf(2.0 * angle)];
        let pi = core::f32::consts::PI;
        let values: Vec<f32> = [axis(0.0), axis(pi), axis(0.0), axis(pi)]
            .into_iter()
            .chain([axis(0.0), axis(pi / 2.0), axis(0.0), axis(pi / 2.0)])
            .flatten()
            .collect();
        let image = Image::new(4, 2, 2, Edge::Clamp, values).unwrap();
        let chain = direction_mips(&image, Filter::Box).unwrap();
        let level = &chain.levels()[1];
        // Left 2×2 block: 0, π, 0, π/2 → three of one axis, one crossed.
        let (x, y) = (level.values[0], level.values[1]);
        assert!((x - 0.5).abs() < 1e-6 && y.abs() < 1e-6, "{x} {y}");
        assert!(
            direction_mips(
                &Image::new(1, 1, 1, Edge::Clamp, vec![0.0]).unwrap(),
                Filter::Box
            )
            .is_err()
        );
        let ids = Image::new(2, 2, 1, Edge::Wrap, vec![4.0, 4.0, 1.0, 4.0]).unwrap();
        assert_eq!(id_mips(&ids).unwrap().levels()[1].values, [4.0]);
    }
}
