// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Separable downsampling filters.

use alloc::vec;
use alloc::vec::Vec;

use dapple_raster::{Edge, TexelRect};

use crate::{EncodeError, Image};

/// Kaiser window shape parameter used by [`Filter::Kaiser`].
pub const KAISER_BETA: f64 = 4.0;

/// Radius of [`Filter::Kaiser`] in destination texels.
pub const KAISER_RADIUS: f64 = 2.0;

/// How a mip level is filtered from the one above it.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Filter {
    /// Each destination texel averages exactly the source area it covers.
    /// For even sizes this is the 2×2 average; odd sizes weight partially
    /// covered texels by their overlap.
    #[default]
    Box,
    /// A sinc windowed by a Kaiser window ([`KAISER_BETA`]) over
    /// [`KAISER_RADIUS`] destination texels. Sharper than [`Filter::Box`]; it
    /// can overshoot near steps, which quantization clamps.
    Kaiser,
}

/// Source taps for one destination texel along one axis.
#[derive(Clone, Debug)]
struct Taps {
    /// First source index, before the edge policy is applied.
    first: i64,
    /// Normalized weights for `first..first + weights.len()`.
    weights: Vec<f64>,
}

/// The size of the next mip level along one axis.
pub(crate) const fn next_size(size: u32) -> u32 {
    if size > 1 { size / 2 } else { 1 }
}

fn taps(filter: Filter, src: u32, dst: u32) -> Vec<Taps> {
    let scale = f64::from(src) / f64::from(dst);
    (0..dst)
        .map(|i| match filter {
            Filter::Box => {
                let lo = f64::from(i) * scale;
                let hi = lo + scale;
                // Exact: `lo` and `hi` are multiples of 1/dst of `src`.
                let first = libm::floor(lo);
                let last = libm::ceil(hi);
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "tap bounds are small, finite integers"
                )]
                let (first_i, last_i) = (first as i64, last as i64);
                let weights = (first_i..last_i)
                    .map(|j| {
                        let j = j as f64;
                        (hi.min(j + 1.0) - lo.max(j)).max(0.0) / scale
                    })
                    .collect();
                Taps {
                    first: first_i,
                    weights,
                }
            }
            Filter::Kaiser => {
                let center = (f64::from(i) + 0.5) * scale;
                let reach = KAISER_RADIUS * scale;
                let first = libm::floor(center - reach);
                let last = libm::ceil(center + reach);
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "tap bounds are small, finite integers"
                )]
                let (first_i, last_i) = (first as i64, last as i64);
                let mut weights: Vec<f64> = (first_i..last_i)
                    .map(|j| kaiser_sinc((j as f64 + 0.5 - center) / scale))
                    .collect();
                let sum: f64 = weights.iter().sum();
                for w in &mut weights {
                    *w /= sum;
                }
                Taps {
                    first: first_i,
                    weights,
                }
            }
        })
        .collect()
}

/// Windowed sinc at `x` destination texels from the center.
fn kaiser_sinc(x: f64) -> f64 {
    let t = x / KAISER_RADIUS;
    if t.abs() >= 1.0 {
        return 0.0;
    }
    let sinc = if x == 0.0 {
        1.0
    } else {
        let px = core::f64::consts::PI * x;
        libm::sin(px) / px
    };
    sinc * bessel_i0(KAISER_BETA * libm::sqrt(1.0 - t * t)) / bessel_i0(KAISER_BETA)
}

/// Modified Bessel function of the first kind, order zero, by its power
/// series (a fixed number of terms, so results are deterministic).
fn bessel_i0(x: f64) -> f64 {
    let half_sq = x * x / 4.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..=32 {
        let k = f64::from(k);
        term *= half_sq / (k * k);
        sum += term;
    }
    sum
}

fn resolve(index: i64, size: u32, edge: Edge) -> usize {
    let size = i64::from(size);
    let i = match edge {
        Edge::Wrap => index.rem_euclid(size),
        Edge::Clamp => index.clamp(0, size - 1),
    };
    usize::try_from(i).expect("resolved index is in range")
}

/// Filters `image` down to the next mip level size.
pub(crate) fn downsample(image: &Image, filter: Filter) -> Image {
    let (dw, dh) = (next_size(image.width), next_size(image.height));
    let mut out = Image {
        width: dw,
        height: dh,
        channels: image.channels,
        edge: image.edge,
        values: vec![0.0; dw as usize * dh as usize * image.channels],
    };
    downsample_rect(image, filter, TexelRect::full(dw, dh), &mut out);
    out
}

/// Writes the texels of `rect` of the next mip level of `image` into `out`,
/// which has the next level's size. Each destination texel's arithmetic is
/// the same as in [`downsample`], so any set of rects covering the level
/// reproduces it bit for bit.
fn downsample_rect(image: &Image, filter: Filter, rect: TexelRect, out: &mut Image) {
    let (sw, sh) = (image.width, image.height);
    let (dw, dh) = (out.width, out.height);
    let c = image.channels;
    let x_taps = taps(filter, sw, dw);
    let y_taps = taps(filter, sh, dh);
    let (x0, x1) = (rect.x0 as usize, rect.x1 as usize);
    let span = x1 - x0;

    // Horizontal pass, only for the source rows the rect's y taps read:
    // `span` destination columns per row, accumulated in f64.
    let mut rows: Vec<Option<Vec<f64>>> = vec![None; sh as usize];
    for t in &y_taps[rect.y0 as usize..rect.y1 as usize] {
        for k in 0..t.weights.len() {
            let sy = resolve(
                t.first + i64::try_from(k).expect("tap index fits i64"),
                sh,
                image.edge,
            );
            if rows[sy].is_some() {
                continue;
            }
            let src_row = &image.values[sy * sw as usize * c..(sy + 1) * sw as usize * c];
            let mut row = vec![0.0_f64; span * c];
            for (x, t) in x_taps[x0..x1].iter().enumerate() {
                let out = &mut row[x * c..(x + 1) * c];
                for (k, w) in t.weights.iter().enumerate() {
                    let sx = resolve(
                        t.first + i64::try_from(k).expect("tap index fits i64"),
                        sw,
                        image.edge,
                    );
                    for ch in 0..c {
                        out[ch] += w * f64::from(src_row[sx * c + ch]);
                    }
                }
            }
            rows[sy] = Some(row);
        }
    }

    // Vertical pass.
    for (y, t) in y_taps
        .iter()
        .enumerate()
        .take(rect.y1 as usize)
        .skip(rect.y0 as usize)
    {
        for x in 0..span {
            for ch in 0..c {
                let mut sum = 0.0_f64;
                for (k, w) in t.weights.iter().enumerate() {
                    let sy = resolve(
                        t.first + i64::try_from(k).expect("tap index fits i64"),
                        sh,
                        image.edge,
                    );
                    let row = rows[sy].as_ref().expect("row computed for its tap");
                    sum += w * row[x * c + ch];
                }
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "filtered values are narrowed back to the image's f32"
                )]
                let v = sum as f32;
                out.values[(y * dw as usize + x0 + x) * c + ch] = v;
            }
        }
    }
}

/// Recomputes the texels of `rect` in `next`, the next mip level of
/// `image` under `filter` (as [`next_level`] builds it).
///
/// Every texel's arithmetic matches [`next_level`], so rects covering the
/// level reproduce it bit for bit. Use [`source_span`] to find which source
/// texels a rect reads.
///
/// # Errors
///
/// [`EncodeError::LevelMismatch`] when `next` does not have the next
/// level's size, channels and edge, or `rect` is not inside it.
pub fn next_level_into(
    image: &Image,
    filter: Filter,
    rect: TexelRect,
    next: &mut Image,
) -> Result<(), EncodeError> {
    let fits = next.width == next_size(image.width)
        && next.height == next_size(image.height)
        && next.channels == image.channels
        && next.edge == image.edge
        && rect.x0 <= rect.x1
        && rect.y0 <= rect.y1
        && rect.x1 <= next.width
        && rect.y1 <= next.height;
    if !fits {
        return Err(EncodeError::LevelMismatch);
    }
    downsample_rect(image, filter, rect, next);
    Ok(())
}

/// The next mip level of `image` under `filter`: half the size (rounding
/// down, at least 1), as each level of [`data_mips`](crate::data_mips).
#[must_use]
pub fn next_level(image: &Image, filter: Filter) -> Image {
    downsample(image, filter)
}

/// The source texels along one axis that destination texels `lo..hi` of the
/// next level read, as a half-open index range before the edge policy
/// (indices may fall outside `0..src`, where the edge wraps or clamps).
///
/// # Panics
///
/// Panics when `lo..hi` is empty or not within the next level's size.
#[must_use]
pub fn source_span(filter: Filter, src: u32, lo: u32, hi: u32) -> (i64, i64) {
    let dst = next_size(src);
    assert!(lo < hi && hi <= dst, "destination span out of range");
    let taps = taps(filter, src, dst);
    let first = taps[lo as usize..hi as usize]
        .iter()
        .map(|t| t.first)
        .min()
        .expect("non-empty span");
    let last = taps[lo as usize..hi as usize]
        .iter()
        .map(|t| t.first + i64::try_from(t.weights.len()).expect("tap count fits i64"))
        .max()
        .expect("non-empty span");
    (first, last)
}

/// Downsamples a single-channel identifier image: each destination texel
/// takes the identifier covering most of its source area (box weights), ties
/// going to the smaller identifier. Identifiers are never averaged.
pub(crate) fn downsample_majority(image: &Image) -> Image {
    debug_assert_eq!(image.channels, 1, "identifiers are one channel");
    let (sw, sh) = (image.width, image.height);
    let (dw, dh) = (next_size(sw), next_size(sh));
    let x_taps = taps(Filter::Box, sw, dw);
    let y_taps = taps(Filter::Box, sh, dh);
    let mut values = Vec::with_capacity(dw as usize * dh as usize);
    let mut votes: Vec<(f32, f64)> = Vec::new();
    for ty in &y_taps {
        for tx in &x_taps {
            votes.clear();
            for (ky, wy) in ty.weights.iter().enumerate() {
                let sy = resolve(
                    ty.first + i64::try_from(ky).expect("tap index fits i64"),
                    sh,
                    image.edge,
                );
                for (kx, wx) in tx.weights.iter().enumerate() {
                    let sx = resolve(
                        tx.first + i64::try_from(kx).expect("tap index fits i64"),
                        sw,
                        image.edge,
                    );
                    let id = image.values[sy * sw as usize + sx];
                    let weight = wx * wy;
                    match votes.iter_mut().find(|(v, _)| v.to_bits() == id.to_bits()) {
                        Some((_, total)) => *total += weight,
                        None => votes.push((id, weight)),
                    }
                }
            }
            let winner = votes
                .iter()
                .copied()
                .reduce(|best, vote| {
                    if vote.1 > best.1 || (vote.1 == best.1 && vote.0 < best.0) {
                        vote
                    } else {
                        best
                    }
                })
                .expect("every texel has taps");
            values.push(winner.0);
        }
    }
    Image {
        width: dw,
        height: dh,
        channels: 1,
        edge: image.edge,
        values,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(width: u32, height: u32, channels: usize, edge: Edge) -> Image {
        let n = width as usize * height as usize * channels;
        let values = (0..n)
            .map(|i| {
                dapple_field::hash::unit_f32(dapple_field::hash::hash(7, &[i as u64])) * 2.0 - 0.5
            })
            .collect();
        Image::new(width, height, channels, edge, values).unwrap()
    }

    #[test]
    fn rects_reproduce_the_next_level_bit_for_bit() {
        for (w, h, edge) in [
            (16, 12, Edge::Wrap),
            (13, 7, Edge::Clamp),
            (5, 1, Edge::Wrap),
        ] {
            for filter in [Filter::Box, Filter::Kaiser] {
                let src = image(w, h, 2, edge);
                let whole = next_level(&src, filter);
                let mut tiled = next_level(&image(w, h, 2, edge), filter);
                for v in &mut tiled.values {
                    *v = f32::NAN;
                }
                let (dw, dh) = (whole.width(), whole.height());
                let step = 3_u32;
                for y0 in (0..dh).step_by(3) {
                    for x0 in (0..dw).step_by(3) {
                        let rect = TexelRect {
                            x0,
                            y0,
                            x1: (x0 + step).min(dw),
                            y1: (y0 + step).min(dh),
                        };
                        next_level_into(&src, filter, rect, &mut tiled).unwrap();
                    }
                }
                let bits = |i: &Image| i.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&whole), bits(&tiled), "{w}x{h} {edge:?} {filter:?}");
            }
        }
    }

    #[test]
    fn source_spans_cover_every_tap() {
        for filter in [Filter::Box, Filter::Kaiser] {
            for src in [1_u32, 2, 5, 16] {
                let dst = next_size(src);
                let taps = taps(filter, src, dst);
                for lo in 0..dst {
                    for hi in lo + 1..=dst {
                        let (first, last) = source_span(filter, src, lo, hi);
                        for t in &taps[lo as usize..hi as usize] {
                            assert!(first <= t.first);
                            let n = i64::try_from(t.weights.len()).unwrap();
                            assert!(t.first + n <= last);
                        }
                    }
                }
            }
        }
        let mismatched = next_level(&image(4, 4, 1, Edge::Wrap), Filter::Box);
        assert_eq!(
            next_level_into(
                &image(8, 8, 1, Edge::Wrap),
                Filter::Box,
                TexelRect::full(2, 2),
                &mut mismatched.clone()
            ),
            Err(EncodeError::LevelMismatch)
        );
    }

    #[test]
    fn box_taps_cover_the_source_exactly() {
        let even = taps(Filter::Box, 8, 4);
        assert_eq!(even[1].first, 2);
        assert_eq!(even[1].weights, [0.5, 0.5]);
        // 5 → 2: each destination texel covers 2.5 source texels.
        let odd = taps(Filter::Box, 5, 2);
        assert_eq!(odd[0].weights, [0.4, 0.4, 0.2]);
        assert_eq!(odd[1].first, 2);
        assert_eq!(odd[1].weights, [0.2, 0.4, 0.4]);
    }

    #[test]
    fn kaiser_taps_are_normalized_and_symmetric() {
        for t in taps(Filter::Kaiser, 16, 8) {
            let sum: f64 = t.weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12, "{sum}");
            let n = t.weights.len();
            for k in 0..n / 2 {
                assert!((t.weights[k] - t.weights[n - 1 - k]).abs() < 1e-12);
            }
        }
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-15);
        // I0(1) = 1.2660658777520082.
        assert!((bessel_i0(1.0) - 1.266_065_877_752_008_2).abs() < 1e-14);
    }

    #[test]
    fn constants_survive_both_filters() {
        let image = Image::new(6, 5, 2, Edge::Clamp, [0.25, 0.75].repeat(30)).unwrap();
        for filter in [Filter::Box, Filter::Kaiser] {
            let down = downsample(&image, filter);
            assert_eq!((down.width, down.height), (3, 2));
            for t in down.values.chunks_exact(2) {
                assert!((t[0] - 0.25).abs() < 1e-6 && (t[1] - 0.75).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn wrapping_mips_commute_with_rolling() {
        let values: Vec<f32> = (0..64).map(|i| ((i * 37) % 11) as f32 / 10.0).collect();
        let image = Image::new(8, 8, 1, Edge::Wrap, values).unwrap();
        let roll = |img: &Image, dx: usize, dy: usize| {
            let (w, h) = (img.width as usize, img.height as usize);
            let mut out = img.clone();
            for y in 0..h {
                for x in 0..w {
                    out.values[((y + dy) % h) * w + (x + dx) % w] = img.values[y * w + x];
                }
            }
            out
        };
        for filter in [Filter::Box, Filter::Kaiser] {
            let a = downsample(&roll(&image, 2, 4), filter);
            let b = roll(&downsample(&image, filter), 1, 2);
            assert_eq!(a.values, b.values, "{filter:?}");
        }
    }

    #[test]
    fn identifiers_take_the_majority() {
        // 4×2: left block mostly 7, right block tied between 2 and 9.
        let values = vec![7.0, 7.0, 2.0, 9.0, 7.0, 3.0, 9.0, 2.0];
        let image = Image::new(4, 2, 1, Edge::Clamp, values).unwrap();
        let down = downsample_majority(&image);
        assert_eq!(down.values, [7.0, 2.0]);
    }
}
