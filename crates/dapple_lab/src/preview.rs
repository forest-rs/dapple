// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Headless previews: tiled, raking-light and mip views, and contact
//! sheets.
//!
//! Previews are linear RGB [`Picture`]s, rows from the top, with domain
//! `+y` up. They are deterministic software renders meant for review and
//! regression, not for final looks (an external renderer does those):
//!
//! - [`base_color`] and [`tiled`]: the color map, and a repeat of any
//!   picture to show seams;
//! - [`raking`]: shading from the height's normals under one directional
//!   light, with a Blinn–Phong highlight from the coat (or the specular
//!   layer) at its roughness; a low elevation shows relief the way a
//!   grazing view does;
//! - [`mips`]: the base color's mip chain side by side, each level scaled
//!   back up by texel replication, to compare levels;
//! - [`contact_sheet`]: pictures on a grid, for parameter sweeps.

use alloc::vec;
use alloc::vec::Vec;

use dapple_encode::{Edge, Filter, Image, color_mips};
use dapple_field::Value;
use dapple_material::{ChannelId, Material, MaterialError, Param};
use dapple_raster::{HeightToNormal, RasterOp};
use glam::Vec3;

/// A linear RGB picture, rows from the top.
#[derive(Clone, Debug, PartialEq)]
pub struct Picture {
    /// Pixels per row.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// Row-major linear RGB.
    pub pixels: Vec<[f32; 3]>,
}

impl Picture {
    /// A picture of `color`.
    #[must_use]
    pub fn filled(width: u32, height: u32, color: [f32; 3]) -> Self {
        Self {
            width,
            height,
            pixels: vec![color; width as usize * height as usize],
        }
    }

    /// The pixel at `(x, y)`, from the top left.
    #[must_use]
    pub fn at(&self, x: u32, y: u32) -> [f32; 3] {
        self.pixels[y as usize * self.width as usize + x as usize]
    }

    /// The picture sRGB-encoded as 8-bit RGB bytes, row-major.
    #[must_use]
    pub fn srgb8(&self) -> Vec<u8> {
        self.pixels
            .iter()
            .flatten()
            .map(|&c| {
                dapple_encode::quantize_unorm8(dapple_encode::linear_to_srgb(c.clamp(0.0, 1.0)))
            })
            .collect()
    }

    /// Writes the picture as an sRGB 8-bit PNG.
    ///
    /// # Errors
    ///
    /// File or encoding errors.
    #[cfg(feature = "std")]
    pub fn write_png(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let file = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut encoder = png::Encoder::new(file, self.width, self.height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(std::io::Error::other)?;
        writer
            .write_image_data(&self.srgb8())
            .map_err(std::io::Error::other)
    }
}

/// Texel `i` of the material's grid in picture order: domain `+y` up.
fn texel_of(m: &Material, x: u32, row: u32) -> usize {
    let g = m.grid();
    (g.height - 1 - row) as usize * g.width as usize + x as usize
}

fn vec3(v: Value) -> Vec3 {
    match v {
        Value::Vector3(v) => v,
        other => Vec3::splat(other.component(0).unwrap_or(0.0)),
    }
}

/// The base color.
#[must_use]
pub fn base_color(m: &Material) -> Picture {
    let g = m.grid();
    let mut pixels = Vec::with_capacity(g.len());
    for row in 0..g.height {
        for x in 0..g.width {
            pixels.push(
                vec3(m.value(ChannelId::Param(Param::BaseColor), texel_of(m, x, row))).to_array(),
            );
        }
    }
    Picture {
        width: g.width,
        height: g.height,
        pixels,
    }
}

/// `picture` repeated `nx` × `ny` times.
#[must_use]
pub fn tiled(picture: &Picture, nx: u32, ny: u32) -> Picture {
    let (w, h) = (picture.width * nx, picture.height * ny);
    let mut pixels = Vec::with_capacity(w as usize * h as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push(picture.at(x % picture.width, y % picture.height));
        }
    }
    Picture {
        width: w,
        height: h,
        pixels,
    }
}

/// The material shaded under a directional light at `azimuth` and
/// `elevation` (radians; azimuth from the domain's `+x` toward `+y`), seen
/// from straight on, with a sky fill of a fifth.
///
/// # Errors
///
/// Never for a valid material.
pub fn raking(m: &Material, azimuth: f32, elevation: f32) -> Result<Picture, MaterialError> {
    let g = m.grid();
    let normals = HeightToNormal { scale: 1.0 }.apply(&m.height()?)?;
    let light = Vec3::new(
        libm::cosf(elevation) * libm::cosf(azimuth),
        libm::cosf(elevation) * libm::sinf(azimuth),
        libm::sinf(elevation),
    );
    let half = (light + Vec3::Z).normalize();
    let p = ChannelId::Param;
    let mut pixels = Vec::with_capacity(g.len());
    for row in 0..g.height {
        for x in 0..g.width {
            let i = texel_of(m, x, row);
            let n = Vec3::from_array(normals.values()[i]);
            let albedo = vec3(m.value(p(Param::BaseColor), i));
            let coat = m.value(p(Param::CoatWeight), i).component(0).unwrap_or(0.0);
            let rough = if coat > 0.5 {
                m.value(p(Param::CoatRoughness), i)
                    .component(0)
                    .unwrap_or(0.0)
            } else {
                m.value(p(Param::SpecularRoughness), i)
                    .component(0)
                    .unwrap_or(0.5)
            };
            let diffuse = n.dot(light).max(0.0);
            let alpha = (rough * rough).max(0.02);
            let shininess = 2.0 / (alpha * alpha) - 2.0;
            let spec = libm::powf(n.dot(half).max(0.0), shininess) * (1.0 - rough) * 0.5;
            pixels.push((albedo * (0.2 + 0.9 * diffuse) + Vec3::splat(spec)).to_array());
        }
    }
    Ok(Picture {
        width: g.width,
        height: g.height,
        pixels,
    })
}

/// The base color's first `levels` mip levels side by side, each scaled
/// up to the material's size by texel replication.
///
/// # Errors
///
/// Never for a valid material.
pub fn mips(m: &Material, levels: usize) -> Result<Picture, MaterialError> {
    let g = m.grid();
    let picture = base_color(m);
    let mut values = Vec::with_capacity(g.len() * 4);
    for c in &picture.pixels {
        values.extend_from_slice(c);
        values.push(1.0);
    }
    let image = Image::new(g.width, g.height, 4, Edge::Wrap, values)
        .map_err(|_| MaterialError::GridMismatch)?;
    let chain = color_mips(&image, Filter::Box);
    let levels: Vec<&Image> = chain.levels().iter().take(levels.max(1)).collect();
    let count = u32::try_from(levels.len()).unwrap_or(1);
    let mut out = Picture::filled(g.width * count, g.height, [0.0; 3]);
    for (k, level) in (0_u32..).zip(&levels) {
        for y in 0..g.height {
            for x in 0..g.width {
                let (lx, ly) = (x * level.width() / g.width, y * level.height() / g.height);
                let t = level.texel(lx, ly);
                out.pixels[(y * out.width + k * g.width + x) as usize] = [t[0], t[1], t[2]];
            }
        }
    }
    Ok(out)
}

/// `pictures` on a grid of `columns`, `gap` pixels apart on a dark ground;
/// cells take the largest picture's size.
#[must_use]
pub fn contact_sheet(pictures: &[Picture], columns: u32, gap: u32) -> Picture {
    let columns = columns.max(1);
    let cw = pictures.iter().map(|p| p.width).max().unwrap_or(1);
    let ch = pictures.iter().map(|p| p.height).max().unwrap_or(1);
    let rows = u32::try_from(pictures.len())
        .unwrap_or(1)
        .div_ceil(columns)
        .max(1);
    let (w, h) = (
        columns * cw + (columns + 1) * gap,
        rows * ch + (rows + 1) * gap,
    );
    let mut out = Picture::filled(w, h, [0.02; 3]);
    for (k, p) in (0_u32..).zip(pictures) {
        let (ox, oy) = (
            gap + (k % columns) * (cw + gap),
            gap + (k / columns) * (ch + gap),
        );
        for y in 0..p.height {
            for x in 0..p.width {
                out.pixels[((oy + y) * w + ox + x) as usize] = p.at(x, y);
            }
        }
    }
    out
}
