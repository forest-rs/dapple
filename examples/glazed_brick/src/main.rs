// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Writes review images of `dapple_library::glazed_brick`: one keyed layout
//! driving color, height, roughness, surface material and owner labels.
//!
//! Run with `cargo run --release -p glazed_brick -- [output-dir]`; the
//! default is the repository's git-ignored `.local/gallery/glazed-brick`.
//! Images show one 1 m period with domain `+y` up. `lit.png` is a simple
//! shaded preview (a raking light from the upper left, a Blinn–Phong
//! highlight on the glaze), `close-up-*.png` crops one corner at full
//! resolution, and `moved-*.png` shows one brick moved by an edit and
//! updated incrementally.
//!
//! `maps/` holds the material as textures for a renderer, packed by
//! `dapple_encode` for the glTF profile: `base_color.png` (sRGB),
//! `normal.png` (tangent-space, `+Y` toward the image top), `orm.png`
//! (occlusion, roughness, metalness), and `height.png`, 16-bit height over
//! the range `height.txt` gives in meters. Rows run from domain `y = 0` at
//! the image top, as glTF reads textures. `tools/render.py` renders them in
//! Blender as a displaced wall.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use dapple_elements::{Composite, ElementKey, Placement, Realized};
use dapple_field::{Domain, Value};
use dapple_library::glazed_brick;
use dapple_raster::typed::Storage;
use dapple_raster::{HeightToNormal, Raster, RasterOp};
use glam::{Vec2, Vec3};

const SIZE: u32 = 2048;
const CROP: u32 = 512;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// An RGB image, rows top to bottom.
struct Rgb {
    width: u32,
    height: u32,
    pixels: Vec<[f32; 3]>,
}

impl Rgb {
    /// From a raster-ordered (bottom row first) function of texels.
    fn from_texels(width: u32, height: u32, f: impl Fn(u32, u32) -> [f32; 3]) -> Self {
        let mut pixels = Vec::with_capacity((width * height) as usize);
        for row in 0..height {
            let y = height - 1 - row;
            for x in 0..width {
                pixels.push(f(x, y));
            }
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    fn crop(&self, x0: u32, row0: u32, size: u32) -> Self {
        let mut pixels = Vec::with_capacity((size * size) as usize);
        for row in row0..row0 + size {
            for x in x0..x0 + size {
                pixels.push(self.pixels[(row * self.width + x) as usize]);
            }
        }
        Self {
            width: size,
            height: size,
            pixels,
        }
    }

    /// Writes linear values sRGB-encoded, or raw when `linear` is false.
    fn write(&self, dir: &Path, name: &str, srgb: bool) -> Result<()> {
        let encode = |c: f32| {
            let c = c.clamp(0.0, 1.0);
            let s = if !srgb {
                c
            } else if c <= 0.003_130_8 {
                12.92 * c
            } else {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            };
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "s is in [0, 1]"
            )]
            let byte = (s * 255.0 + 0.5) as u8;
            byte
        };
        let bytes: Vec<u8> = self.pixels.iter().flatten().map(|&c| encode(c)).collect();
        let path = dir.join(format!("{name}.png"));
        let mut encoder = png::Encoder::new(
            BufWriter::new(File::create(&path)?),
            self.width,
            self.height,
        );
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.write_header()?.write_image_data(&bytes)?;
        println!("{}", path.display());
        Ok(())
    }
}

fn vec3(v: Value) -> Vec3 {
    match v {
        Value::Vector3(v) => v,
        other => Vec3::splat(other.component(0).unwrap_or(0.0)),
    }
}

/// A stable color for a key, for label previews.
fn key_color(key: ElementKey) -> [f32; 3] {
    let w = key.word();
    let c = |shift: u32| 0.25 + 0.6 * f32::from(((w >> shift) & 0xff) as u8) / 255.0;
    [c(0), c(8), c(16)]
}

fn main() -> Result<()> {
    let out = std::env::args_os().nth(1).map_or_else(
        || repository().join(".local/gallery/glazed-brick"),
        PathBuf::from,
    );
    std::fs::create_dir_all(&out)?;

    let domain = Domain::periodic(1, 1).expect("a unit period");
    let set = glazed_brick::layout()?;
    let instance = glazed_brick::instance(Arc::new(glazed_brick::program()?))?;
    let composite = |set| Composite {
        set,
        instance: &instance,
        background: &glazed_brick::BACKGROUND,
        domain,
        width: SIZE,
        height: SIZE,
        tile_size: 64,
    };
    let start = std::time::Instant::now();
    let mut realized = Realized::composite(&composite(&set))?;
    println!(
        "composited {} bricks at {SIZE}² in {:.2?}",
        set.len(),
        start.elapsed()
    );
    let start = std::time::Instant::now();
    let finished = glazed_brick::finish(&realized, domain)?;
    println!("finished the mortar in {:.2?}", start.elapsed());
    write_all(&out, "", &realized, &finished)?;
    write_maps(&out.join("maps"), &finished)?;

    // Move one brick up and to the right, and update incrementally.
    let key = set.keys()[set.len() / 2];
    let i = set.index_of(key).expect("present");
    let mut moved = set.clone();
    let p = set.placement(i);
    moved.set_placement(
        key,
        Placement {
            center: p.center + Vec2::new(0.03, 0.012),
            rotation: 0.08,
        },
    )?;
    let start = std::time::Instant::now();
    let report = realized.update(&composite(&moved))?;
    println!(
        "moved {key:?}: {} of {} tiles, {} texels, in {:.2?}",
        report.tiles.len(),
        (SIZE / 64) * (SIZE / 64),
        report.texels,
        start.elapsed()
    );
    let clean = Realized::composite(&composite(&moved))?;
    assert_eq!(
        realized.digest(),
        clean.digest(),
        "incremental equals clean"
    );
    write_all(
        &out,
        "moved-",
        &realized,
        &glazed_brick::finish(&realized, domain)?,
    )?;
    Ok(())
}

fn write_all(
    out: &Path,
    prefix: &str,
    realized: &Realized,
    finished: &glazed_brick::Finished,
) -> Result<()> {
    let color = &finished.base_color;
    let material = &finished.material;
    let h = &finished.height;
    let at = |r: &dapple_raster::typed::TypedRaster, x: u32, y: u32| {
        r.value_at(i64::from(x), i64::from(y))
    };
    let scalar = |r: &Raster, x: u32, y: u32| r.at(i64::from(x), i64::from(y));

    let base = Rgb::from_texels(SIZE, SIZE, |x, y| vec3(at(color, x, y)).to_array());
    base.write(out, &format!("{prefix}base-color"), true)?;

    let (lo, hi) = h
        .values()
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    Rgb::from_texels(SIZE, SIZE, |x, y| [(scalar(h, x, y) - lo) / (hi - lo); 3]).write(
        out,
        &format!("{prefix}height"),
        false,
    )?;
    Rgb::from_texels(SIZE, SIZE, |x, y| {
        [scalar(&finished.specular_roughness, x, y); 3]
    })
    .write(out, &format!("{prefix}roughness"), false)?;
    Rgb::from_texels(SIZE, SIZE, |x, y| match at(material, x, y) {
        Value::Id(glazed_brick::GLAZE) => [0.1, 0.55, 0.6],
        Value::Id(glazed_brick::BODY) => [0.9, 0.5, 0.2],
        _ => [0.35, 0.35, 0.35],
    })
    .write(out, &format!("{prefix}material"), false)?;
    Rgb::from_texels(SIZE, SIZE, |x, y| {
        realized.owner(x, y).map_or([0.1; 3], key_color)
    })
    .write(out, &format!("{prefix}owners"), false)?;

    // Normals from the height (meters), and a shaded preview.
    let normals: Raster<[f32; 3]> = HeightToNormal { scale: 1.0 }.apply(h)?;
    Rgb::from_texels(SIZE, SIZE, |x, y| {
        normals
            .at(i64::from(x), i64::from(y))
            .map(|c| c * 0.5 + 0.5)
    })
    .write(out, &format!("{prefix}normal"), false)?;
    let light = Vec3::new(-0.55, 0.6, 0.58).normalize();
    let view = Vec3::Z;
    let half = (light + view).normalize();
    let lit = Rgb::from_texels(SIZE, SIZE, |x, y| {
        let n = Vec3::from_array(normals.at(i64::from(x), i64::from(y)));
        let albedo = vec3(at(color, x, y));
        let r = scalar(&finished.specular_roughness, x, y);
        let diffuse = n.dot(light).max(0.0);
        let alpha = (r * r).max(0.02);
        let shininess = 2.0 / (alpha * alpha) - 2.0;
        let spec = n.dot(half).max(0.0).powf(shininess) * (1.0 - r) * 0.6;
        (albedo * (0.25 + 0.9 * diffuse) + Vec3::splat(spec)).to_array()
    });
    lit.write(out, &format!("{prefix}lit"), true)?;
    let row0 = SIZE / 2 - CROP / 2;
    lit.crop(SIZE / 2 - CROP / 2, row0, CROP)
        .write(out, &format!("{prefix}close-up-lit"), true)?;
    base.crop(SIZE / 2 - CROP / 2, row0, CROP).write(
        out,
        &format!("{prefix}close-up-base-color"),
        true,
    )?;
    Ok(())
}

/// Packs the material for the glTF profile and writes it as PNG files.
fn write_maps(dir: &Path, finished: &glazed_brick::Finished) -> Result<()> {
    use dapple_encode::{
        Edge, Filter, Image, MaterialMaps, PackSettings, PixelFormat, Profile, data_mips,
        encode_data, pack,
    };
    std::fs::create_dir_all(dir)?;
    let Storage::F32x3(color) = finished.base_color.storage() else {
        unreachable!("colors have three channels")
    };
    let h = &finished.height;
    let normals: Raster<[f32; 3]> = HeightToNormal { scale: 1.0 }.apply(h)?;
    let maps = MaterialMaps {
        base_color: Some(Image::new(
            SIZE,
            SIZE,
            3,
            Edge::Wrap,
            color.values().iter().flatten().copied().collect(),
        )?),
        normal: Some(Image::new(
            SIZE,
            SIZE,
            3,
            Edge::Wrap,
            normals.values().iter().flatten().copied().collect(),
        )?),
        specular_roughness: Some(Image::new(
            SIZE,
            SIZE,
            1,
            Edge::Wrap,
            finished.specular_roughness.values().to_vec(),
        )?),
        ..MaterialMaps::default()
    };
    let bundle = pack(&maps, Profile::Gltf, &PackSettings::default())?;
    let mut textures: Vec<_> = bundle.textures.iter().collect();
    let (lo, hi) = h
        .values()
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let unit: Vec<f32> = h.values().iter().map(|&v| (v - lo) / (hi - lo)).collect();
    let chain = data_mips(&Image::new(SIZE, SIZE, 1, Edge::Wrap, unit)?, Filter::Box);
    let height_texture = encode_data("height", &chain, PixelFormat::R16Unorm)?;
    textures.push(&height_texture);
    for texture in textures {
        let path = dir.join(format!("{}.png", texture.name));
        std::fs::write(&path, dapple_encode::png::write(texture)?)?;
        println!("{}", path.display());
    }
    std::fs::write(dir.join("height.txt"), format!("{lo} {hi}\n"))?;
    Ok(())
}

fn repository() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the crate lives two levels below the repository root")
}
