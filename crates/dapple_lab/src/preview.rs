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

use dapple_encode::{Filter, Image, color_mips};
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
    let image = Image::new(g.width, g.height, 4, g.edge, values)
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

#[cfg(test)]
mod tests {
    use dapple_field::PortType;
    use dapple_material::{Aux, Channel, Grid, param_port};
    use glam::Vec2;

    use super::*;
    use crate::material::unit_tile;

    /// A tile whose base color runs red with domain `x` and green with
    /// domain `y`, over a bumpy height, so every preview has something to
    /// show and orientation can be told.
    fn material(grid: Grid) -> Material {
        let mut m = Material::new(grid);
        let color: Vec<Value> = (0..grid.len())
            .map(|i| {
                let c = grid.center(i);
                Value::Vector3(Vec3::new(c.x, c.y, 0.25))
            })
            .collect();
        m.set_param(
            Param::BaseColor,
            Channel::Map(grid.typed(param_port(Param::BaseColor), color).unwrap()),
        )
        .unwrap();
        let height: Vec<Value> = (0..grid.len())
            .map(|i| {
                let c: Vec2 = grid.center(i);
                Value::Scalar(0.02 * libm::sinf(core::f32::consts::TAU * 2.0 * c.x))
            })
            .collect();
        m.set_aux(
            Aux::Height,
            Channel::Map(grid.typed(PortType::Scalar, height).unwrap()),
        )
        .unwrap();
        m
    }

    #[test]
    fn base_color_has_the_grids_size_and_domain_y_up() {
        let m = material(unit_tile(16));
        let p = base_color(&m);
        assert_eq!((p.width, p.height), (16, 16));
        assert_eq!(p.pixels.len(), 256);
        // The top row is the largest domain y; the left column the smallest x.
        let (top, bottom) = (p.at(0, 0), p.at(0, 15));
        assert!(top[1] > bottom[1], "{top:?} over {bottom:?}");
        assert!(p.at(15, 8)[0] > p.at(0, 8)[0]);
        assert_eq!(p.at(3, 4)[2], 0.25);
    }

    #[test]
    fn tiling_repeats_the_picture() {
        let p = base_color(&material(unit_tile(8)));
        let t = tiled(&p, 3, 2);
        assert_eq!((t.width, t.height), (24, 16));
        assert_eq!(t.at(8 + 3, 8 + 5), p.at(3, 5));
        assert_eq!(t.at(23, 15), p.at(7, 7));
    }

    #[test]
    fn raking_light_is_deterministic_and_shows_relief() {
        let m = material(unit_tile(32));
        let a = raking(&m, 2.4, 0.3).expect("renders");
        assert_eq!((a.width, a.height), (32, 32));
        assert_eq!(a, raking(&m, 2.4, 0.3).unwrap(), "deterministic");
        assert!(
            a.pixels
                .iter()
                .flatten()
                .all(|c| c.is_finite() && *c >= 0.0)
        );
        // Under a low light a gray ground shows its bumps; flat, it shows
        // none.
        let spread = |mut m: Material, flatten: bool| {
            m.set_param(
                Param::BaseColor,
                Channel::Constant(Value::Vector3(Vec3::splat(0.5))),
            )
            .unwrap();
            if flatten {
                m.set_aux(Aux::Height, Channel::Constant(Value::Scalar(0.0)))
                    .unwrap();
            }
            let p = raking(&m, 2.4, 0.3).unwrap();
            let (lo, hi) = p
                .pixels
                .iter()
                .fold((f32::MAX, f32::MIN), |(l, h), c| (l.min(c[0]), h.max(c[0])));
            hi - lo
        };
        assert!(spread(m.clone(), false) > 0.05, "relief varies the shade");
        assert!(spread(m.clone(), true) < 1e-6, "a flat ground is uniform");
        // Another light makes another picture.
        assert_ne!(a, raking(&m, 0.4, 0.3).unwrap());
    }

    #[test]
    fn mips_put_each_level_beside_the_last_at_full_size() {
        let m = material(unit_tile(16));
        let p = mips(&m, 3).expect("renders");
        assert_eq!((p.width, p.height), (48, 16));
        assert_eq!(p, mips(&m, 3).unwrap(), "deterministic");
        // Level 0 is the base color; the coarser levels are blockier.
        let base = base_color(&m);
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(p.at(x, y), base.at(x, y));
            }
        }
        assert_eq!(p.at(16, 0), p.at(17, 0), "level 1 repeats texels in pairs");
        // Zero levels still shows the base.
        assert_eq!(mips(&m, 0).unwrap().width, 16);
    }

    #[test]
    fn contact_sheets_lay_out_on_a_grid_with_gaps() {
        let a = Picture::filled(4, 2, [1.0; 3]);
        let b = Picture::filled(2, 3, [0.5; 3]);
        let sheet = contact_sheet(&[a.clone(), b.clone(), a.clone()], 2, 1);
        // Two columns of the widest (4), two rows of the tallest (3).
        assert_eq!((sheet.width, sheet.height), (2 * 4 + 3, 2 * 3 + 3));
        assert_eq!(sheet.at(1, 1), [1.0; 3], "the first picture");
        assert_eq!(sheet.at(1 + 5, 1), [0.5; 3], "the second, a gap over");
        assert_eq!(sheet.at(0, 0), [0.02; 3], "the ground shows in the gap");
        let empty = contact_sheet(&[], 3, 2);
        assert_eq!((empty.width, empty.height), (11, 5));
    }

    #[test]
    fn pictures_encode_as_srgb_bytes() {
        let p = Picture::filled(2, 1, [0.0, 1.0, 2.0]);
        // Out-of-range values clamp; linear 0 and 1 map to 0 and 255.
        assert_eq!(p.srgb8(), [0, 255, 255, 0, 255, 255]);
        let mid = Picture::filled(1, 1, [0.5; 3]).srgb8();
        assert!((185..=190).contains(&mid[0]), "{}", mid[0]);
    }
}
