// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sampling fields onto small grids, for tests and previews.
//!
//! This is not the realization engine: it has no tiles, caches, or
//! invalidation. It samples texel centers with a texel-sized footprint.

use alloc::vec::Vec;

use glam::Vec2;

use crate::domain::{Domain, Footprint};
use crate::field::ScalarField;
use crate::hash::{hash, key};

/// A rectangle of the domain: `origin` to `origin + size`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Region {
    /// Minimum corner.
    pub origin: Vec2,
    /// Extent; both components positive.
    pub size: Vec2,
}

impl Region {
    /// One full period of a periodic domain, from the origin.
    ///
    /// A grid over this region tiles seamlessly: the texel after the last one
    /// is the first one again.
    #[must_use]
    pub fn period(domain: Domain) -> Option<Self> {
        let [x, y] = domain.period()?;
        // Exact for periods below 2^24.
        let size = Vec2::new(x as f32, y as f32);
        Some(Self {
            origin: Vec2::ZERO,
            size,
        })
    }
}

/// Row-major samples of a field, one per texel center.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    /// Texels per row.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// `width * height` values, row by row from the region's origin.
    pub values: Vec<f32>,
}

impl Grid {
    /// Samples `field` at the texel centers of a `width` × `height` grid over
    /// `region`, each with a footprint of one texel (the larger axis).
    #[must_use]
    pub fn sample(field: &impl ScalarField, region: Region, width: u32, height: u32) -> Self {
        let texel = region.size / Vec2::new(width as f32, height as f32);
        let footprint = Footprint::new(texel.max_element()).unwrap_or(Footprint::POINT);
        let mut values = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                let center = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                values.push(field.eval(region.origin + center * texel, footprint));
            }
        }
        Self {
            width,
            height,
            values,
        }
    }

    /// A 64-bit digest of the exact value bits, for golden tests.
    #[must_use]
    pub fn digest(&self) -> u64 {
        let mut h = hash(0x6772_6964, &[u64::from(self.width), u64::from(self.height)]);
        for value in &self.values {
            h = hash(h, &[key(i64::from(value.to_bits()))]);
        }
        h
    }
}
