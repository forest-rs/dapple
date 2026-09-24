// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Bakes dapple materials onto exedra surfaces through their texture charts.
//!
//! Exedra gives every constructed surface a texture chart: its UVs are
//! measured on the surface in recipe units, so one UV unit is about one
//! meter of surface. This crate turns a region of an extracted
//! [`TriMesh`] into a texel grid over that chart, finds the surface point
//! each texel stands for, and hands those points to a dapple evaluator: a
//! solid field (wood grain, stone) through
//! [`SolidProgram::eval_chart`](dapple_field::program::SolidProgram::eval_chart),
//! or a planar field at the chart coordinates themselves.
//!
//! **Flow.** [`SurfaceBake::new`] rasterizes the region's triangles in UV
//! space and records one [`ChartSample`] per covered texel: its position in
//! the material's solid space and a footprint that is the texel's size on
//! the surface. Evaluate the samples with any evaluator, then
//! [`SurfaceBake::scatter`] the values back into a row-major grid, with
//! `padding` texels of dilation around the covered area so filtering and
//! mip levels do not bleed the background in at chart edges.
//!
//! **Placement in the texture.** The grid covers the region's UV bounds,
//! padded, at [`BakeParams::texels_per_unit`]. The mesh keeps its UVs:
//! [`SurfaceBake::texture_transform`] gives the offset and scale that map
//! them onto the baked image, as glTF's `KHR_texture_transform` expects.
//! Row 0 of the grid is the smallest V, which is the top of the image.
//!
//! **Space.** Sample positions are the mesh's positions mapped by
//! [`BakeParams::to_solid`], which places the part inside the material,
//! such as a timber inside its log. Footprints are measured after that map.
//!
//! Baking is deterministic: triangles rasterize in index order, the first
//! triangle to cover a texel wins it, and dilation grows in row-major order.

#![no_std]

extern crate alloc;

use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use dapple_field::Footprint;
use dapple_field::program::ChartSample;
use exedra_mesh::TriMesh;
use glam::{Affine3A, Vec2, Vec3, Vec3A};

/// Marks a grid texel that no sample covers.
const UNCOVERED: u32 = u32::MAX;

/// Barycentric tolerance for texel centers on triangle edges.
const EDGE: f32 = 1e-5;

/// Settings for [`SurfaceBake::new`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BakeParams {
    /// Texels per UV unit, on both axes. Positive and finite.
    pub texels_per_unit: f32,
    /// Texels of dilation around the covered area, also added to the grid on
    /// every side.
    pub padding: u32,
    /// The largest grid, in texels, the bake may allocate.
    pub max_texels: u64,
    /// Maps mesh positions into the material's solid space.
    pub to_solid: Affine3A,
}

impl BakeParams {
    /// `texels_per_unit` texels per UV unit, 4 texels of padding, at most
    /// 4096 × 4096 texels, and the mesh's own space as the material's.
    #[must_use]
    pub fn new(texels_per_unit: f32) -> Self {
        Self {
            texels_per_unit,
            padding: 4,
            max_texels: 4096 * 4096,
            to_solid: Affine3A::IDENTITY,
        }
    }

    /// The same settings with `to_solid` as the placement in the material.
    #[must_use]
    pub fn with_placement(self, to_solid: Affine3A) -> Self {
        Self { to_solid, ..self }
    }
}

/// Why a region cannot be baked.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BakeError {
    /// The index range is empty or not a whole number of triangles.
    NoTriangles,
    /// An index names no vertex of the mesh.
    InvalidIndex {
        /// The offending index value.
        index: u32,
    },
    /// A vertex used by the region has a non-finite UV or position.
    NonFiniteVertex {
        /// The vertex.
        vertex: u32,
    },
    /// Every triangle of the region has zero UV area.
    DegenerateChart,
    /// The grid would exceed [`BakeParams::max_texels`].
    TooLarge {
        /// Grid width.
        width: u64,
        /// Grid height.
        height: u64,
    },
    /// [`BakeParams::texels_per_unit`] is not positive and finite, or
    /// [`BakeParams::to_solid`] is not finite and invertible.
    InvalidParams,
}

impl fmt::Display for BakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoTriangles => write!(f, "no whole triangles to bake"),
            Self::InvalidIndex { index } => write!(f, "index {index} names no vertex"),
            Self::NonFiniteVertex { vertex } => {
                write!(f, "vertex {vertex} has a non-finite UV or position")
            }
            Self::DegenerateChart => write!(f, "every triangle has zero UV area"),
            Self::TooLarge { width, height } => {
                write!(f, "a {width} x {height} grid exceeds the texel budget")
            }
            Self::InvalidParams => write!(f, "invalid bake parameters"),
        }
    }
}

impl core::error::Error for BakeError {}

/// Counts from one [`SurfaceBake::new`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct BakeStats {
    /// Triangles rasterized.
    pub triangles: u32,
    /// Triangles skipped for zero UV area.
    pub degenerate_triangles: u32,
    /// Texels whose center a triangle covers: one sample each.
    pub covered_texels: u32,
    /// Texels filled from a nearby covered texel by dilation.
    pub dilated_texels: u32,
    /// Texels strictly inside more than one triangle: the chart overlaps
    /// itself there, and the first triangle's surface point was baked.
    pub overlapping_texels: u32,
}

/// A region of a surface as a texel grid over its texture chart, with the
/// surface point behind each covered texel.
#[derive(Clone, Debug)]
pub struct SurfaceBake {
    width: u32,
    height: u32,
    uv_origin: Vec2,
    uv_size: Vec2,
    samples: Vec<ChartSample>,
    normals: Vec<Vec3>,
    uvs: Vec<Vec2>,
    texel_source: Vec<u32>,
    stats: BakeStats,
}

impl SurfaceBake {
    /// Rasterizes the triangles `indices` of `mesh` (a whole number of
    /// triangles, such as one region's index range) over their UV chart.
    ///
    /// Normals come from the mesh's render normals when it has them, else
    /// from each triangle's geometry.
    ///
    /// # Errors
    ///
    /// Returns a [`BakeError`] for invalid parameters, a malformed index
    /// range, non-finite vertices, a chart with no area, or a grid over the
    /// texel budget.
    pub fn new(mesh: &TriMesh, indices: &[u32], params: &BakeParams) -> Result<Self, BakeError> {
        let density = params.texels_per_unit;
        if !density.is_finite() || density <= 0.0 || !params.to_solid.is_finite() {
            return Err(BakeError::InvalidParams);
        }
        let linear = params.to_solid.matrix3;
        if linear.determinant() == 0.0 {
            return Err(BakeError::InvalidParams);
        }
        let normal_map = linear.inverse().transpose();
        if indices.is_empty() || !indices.len().is_multiple_of(3) {
            return Err(BakeError::NoTriangles);
        }
        let vertex_count = mesh.positions.len().min(mesh.uvs.len());
        let mut lo = Vec2::splat(f32::INFINITY);
        let mut hi = Vec2::splat(f32::NEG_INFINITY);
        for &index in indices {
            let vertex = index as usize;
            if vertex >= vertex_count {
                return Err(BakeError::InvalidIndex { index });
            }
            let uv = Vec2::from(mesh.uvs[vertex]);
            let position = Vec3::from(mesh.positions[vertex]);
            if !uv.is_finite() || !position.is_finite() {
                return Err(BakeError::NonFiniteVertex { vertex: index });
            }
            lo = lo.min(uv);
            hi = hi.max(uv);
        }

        let texel = 1.0 / density;
        let pad = params.padding as f32 * texel;
        let span = ((hi - lo) * density).ceil().max(Vec2::ONE);
        let width = u64::from(params.padding) * 2 + texel_count(span.x);
        let height = u64::from(params.padding) * 2 + texel_count(span.y);
        // Texel indices are `u32`, with `u32::MAX` reserved for uncovered
        // texels, so the grid must also stay below that.
        let texels = width.saturating_mul(height);
        let (Ok(width32), Ok(height32)) = (u32::try_from(width), u32::try_from(height)) else {
            return Err(BakeError::TooLarge { width, height });
        };
        if texels > params.max_texels || texels >= u64::from(UNCOVERED) {
            return Err(BakeError::TooLarge { width, height });
        }
        let (width, height) = (width32, height32);
        let uv_origin = lo - Vec2::splat(pad);
        let uv_size = Vec2::new(width as f32, height as f32) * texel;

        let texels = width as usize * height as usize;
        let mut texel_source = vec![UNCOVERED; texels];
        let mut samples = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut stats = BakeStats::default();
        let has_normals = mesh.normals.len() >= vertex_count;

        for triangle in indices.chunks_exact(3) {
            let [a, b, c] = [triangle[0], triangle[1], triangle[2]].map(|i| i as usize);
            let uv = [a, b, c].map(|v| (Vec2::from(mesh.uvs[v]) - uv_origin) * density);
            let area = cross2(uv[1] - uv[0], uv[2] - uv[0]);
            if area == 0.0 || !area.is_finite() {
                stats.degenerate_triangles += 1;
                continue;
            }
            stats.triangles += 1;
            let solid = [a, b, c].map(|v| {
                Vec3::from(
                    params
                        .to_solid
                        .transform_point3a(Vec3A::from(mesh.positions[v])),
                )
            });
            let geometric = (solid[1] - solid[0]).cross(solid[2] - solid[0]);
            // Surface length per UV unit, from the triangle's areas in solid
            // space and on the chart (in texels, so rescale by the density).
            let chart_area = area.abs() * texel * texel;
            let stretch = libm::sqrtf(geometric.length() / chart_area);
            let footprint = Footprint::new(stretch * texel).unwrap_or(Footprint::POINT);
            // The differential basis: solid-space steps per texel along the
            // chart's x and y, from the triangle's affine map.
            let (e1, e2) = (uv[1] - uv[0], uv[2] - uv[0]);
            let (d1, d2) = (solid[1] - solid[0], solid[2] - solid[0]);
            let basis = [
                (d1 * e2.y - d2 * e1.y) / area,
                (d2 * e1.x - d1 * e2.x) / area,
            ];
            let vertex_normals = [a, b, c].map(|v| {
                let n = if has_normals {
                    Vec3A::from(mesh.normals[v])
                } else {
                    Vec3A::from(geometric)
                };
                Vec3::from(normal_map * n)
            });

            let min = uv[0].min(uv[1]).min(uv[2]).floor().max(Vec2::ZERO);
            let max = uv[0].max(uv[1]).max(uv[2]).ceil();
            let (x0, y0) = (grid_index(min.x), grid_index(min.y));
            let (x1, y1) = (grid_index(max.x).min(width), grid_index(max.y).min(height));
            for y in y0..y1 {
                for x in x0..x1 {
                    let center = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                    let w0 = cross2(uv[1] - center, uv[2] - center) / area;
                    let w1 = cross2(uv[2] - center, uv[0] - center) / area;
                    let w2 = 1.0 - w0 - w1;
                    // Texel centers on a shared edge land in both triangles
                    // within rounding; the tolerance keeps them from falling
                    // into neither.
                    if w0 < -EDGE || w1 < -EDGE || w2 < -EDGE {
                        continue;
                    }
                    let slot = y as usize * width as usize + x as usize;
                    if texel_source[slot] != UNCOVERED {
                        if w0 > EDGE && w1 > EDGE && w2 > EDGE {
                            stats.overlapping_texels += 1;
                        }
                        continue;
                    }
                    let position = solid[0] * w0 + solid[1] * w1 + solid[2] * w2;
                    let normal =
                        (vertex_normals[0] * w0 + vertex_normals[1] * w1 + vertex_normals[2] * w2)
                            .normalize_or_zero();
                    texel_source[slot] = sample_index(samples.len());
                    samples.push(ChartSample {
                        position,
                        footprint,
                        basis,
                    });
                    normals.push(normal);
                    uvs.push(uv_origin + center * texel);
                }
            }
        }
        if stats.triangles == 0 {
            return Err(BakeError::DegenerateChart);
        }
        stats.covered_texels = sample_index(samples.len());
        stats.dilated_texels = dilate(&mut texel_source, width, height, params.padding);
        Ok(Self {
            width,
            height,
            uv_origin,
            uv_size,
            samples,
            normals,
            uvs,
            texel_source,
            stats,
        })
    }

    /// Grid width in texels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Grid height in texels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// One sample per covered texel: its point in the material's solid space
    /// and its footprint on the surface.
    #[must_use]
    pub fn samples(&self) -> &[ChartSample] {
        &self.samples
    }

    /// The unit surface normal at each sample, in the material's solid space
    /// (zero where the interpolated normal vanishes).
    #[must_use]
    pub fn normals(&self) -> &[Vec3] {
        &self.normals
    }

    /// The chart coordinates of each sample: the texel center's UV, for
    /// evaluating a planar field in the chart's own units.
    #[must_use]
    pub fn uvs(&self) -> &[Vec2] {
        &self.uvs
    }

    /// Counts from the bake.
    #[must_use]
    pub const fn stats(&self) -> BakeStats {
        self.stats
    }

    /// Spreads one value per sample over the grid, row-major, row 0 at the
    /// smallest V. Dilated texels repeat a nearby covered texel's value;
    /// texels beyond the padding get `background`.
    ///
    /// # Panics
    ///
    /// Panics if `values` has fewer entries than [`Self::samples`].
    #[must_use]
    pub fn scatter<T: Copy>(&self, values: &[T], background: T) -> Vec<T> {
        assert!(
            values.len() >= self.samples.len(),
            "one value per sample is needed"
        );
        self.texel_source
            .iter()
            .map(|&source| {
                if source == UNCOVERED {
                    background
                } else {
                    values[source as usize]
                }
            })
            .collect()
    }

    /// The mapping from the mesh's UVs onto the baked grid, as
    /// `KHR_texture_transform` applies it: `uv' = uv * scale + offset`.
    #[must_use]
    pub fn texture_transform(&self) -> TextureTransform {
        let scale = self.uv_size.recip();
        TextureTransform {
            offset: (-self.uv_origin * scale).to_array(),
            scale: scale.to_array(),
        }
    }
}

/// A texture coordinate mapping, `uv' = uv * scale + offset`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TextureTransform {
    /// Added after scaling.
    pub offset: [f32; 2],
    /// Per-axis scale.
    pub scale: [f32; 2],
}

/// Fills uncovered texels within `rings` steps of a covered one with the
/// nearest covered texel's sample (4-neighbor distance, row-major order on
/// ties). Returns the number of texels filled.
fn dilate(sources: &mut [u32], width: u32, height: u32, rings: u32) -> u32 {
    if rings == 0 {
        return 0;
    }
    let (w, h) = (width as usize, height as usize);
    let mut distance = vec![0_u32; sources.len()];
    let mut queue = VecDeque::new();
    for (slot, source) in sources.iter().enumerate() {
        if *source != UNCOVERED {
            queue.push_back(slot);
        }
    }
    let mut filled = 0;
    while let Some(slot) = queue.pop_front() {
        let step = distance[slot] + 1;
        if step > rings {
            continue;
        }
        let (x, y) = (slot % w, slot / w);
        let neighbors = [
            (y > 0).then(|| slot - w),
            (x > 0).then(|| slot - 1),
            (x + 1 < w).then(|| slot + 1),
            (y + 1 < h).then(|| slot + w),
        ];
        for next in neighbors.into_iter().flatten() {
            if sources[next] == UNCOVERED {
                sources[next] = sources[slot];
                distance[next] = step;
                filled += 1;
                queue.push_back(next);
            }
        }
    }
    filled
}

/// A non-negative grid coordinate as an index, saturating past the grid.
fn grid_index(coordinate: f32) -> u32 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "float-to-int casts saturate, and callers clamp to the grid"
    )]
    let index = coordinate as u32;
    index
}

/// A sample's index; the grid size check keeps every index below
/// [`UNCOVERED`].
fn sample_index(index: usize) -> u32 {
    u32::try_from(index).unwrap_or(UNCOVERED)
}

/// A whole, positive texel count as an integer, saturating far beyond any
/// budget so the caller's size check refuses it.
fn texel_count(count: f32) -> u64 {
    if count >= 1.0e18 {
        u64::MAX / 4
    } else {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a whole, positive count below 1e18 fits a u64"
        )]
        let count = count as u64;
        count
    }
}

fn cross2(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

#[cfg(test)]
mod tests;
