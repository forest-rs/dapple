// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Packing material maps into 8-bit textures for a consumer.

use alloc::vec;
use alloc::vec::Vec;

use dapple_raster::Edge;

use crate::filter::Filter;
use crate::mips::{
    CoverageLevel, MipChain, color_mips, data_mips, direction_mips, normal_mips, preserve_coverage,
};
use crate::quantize::{PixelFormat, linear_to_srgb, quantize_unorm8, quantize_unorm16};
use crate::{EncodeError, Image};

/// Who the textures are for.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Profile {
    /// `lightweald_material` slots: `base_color` (sRGB RGB, linear alpha
    /// opacity), a two-channel `normal` (XY; Lightweald rebuilds Z), and an
    /// `orm` texture (R occlusion, G roughness, B metalness) that both the
    /// occlusion and metallic-roughness slots can reference.
    Lightweald,
    /// glTF 2.0: `base_color` as Lightweald, `normal` as RGB tangent-space
    /// XYZ, and the `orm` texture shared by `occlusionTexture` and
    /// `metallicRoughnessTexture`.
    Gltf,
    /// One texture per parameter, linear 8-bit, normals in the input frame;
    /// for inspection and tools rather than rendering.
    Raw,
}

/// Material parameter maps to pack. All present maps must share a size and
/// edge policy. Values are linear.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MaterialMaps {
    /// OpenPBR `base_color`, 3 channels.
    pub base_color: Option<Image>,
    /// OpenPBR `geometry_opacity`, 1 channel.
    pub opacity: Option<Image>,
    /// Unit shading normals, 3 channels, in the domain frame of
    /// `dapple_raster::HeightToNormal`: `+X` along increasing column, `+Y`
    /// along increasing row, `+Z` out of the surface.
    pub normal: Option<Image>,
    /// OpenPBR `specular_roughness`, 1 channel.
    pub specular_roughness: Option<Image>,
    /// OpenPBR `base_metalness`, 1 channel.
    pub base_metalness: Option<Image>,
    /// Ambient occlusion, 1 channel; an auxiliary output, not an OpenPBR
    /// parameter.
    pub occlusion: Option<Image>,
    /// The anisotropy axis, 2 channels: the doubled-angle vector
    /// `(cos 2θ, sin 2θ)` of `dapple_field`'s `PortType::Direction`, with `θ`
    /// measured in the normal map's domain frame.
    pub anisotropy_direction: Option<Image>,
    /// OpenPBR `specular_roughness_anisotropy`, 1 channel.
    pub specular_roughness_anisotropy: Option<Image>,
}

/// Settings for [`pack()`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PackSettings {
    /// Mip filter.
    pub filter: Filter,
    /// Alpha-test cutoff whose coverage opacity mips preserve; `None` filters
    /// opacity like any data.
    pub alpha_cutoff: Option<f32>,
    /// `specular_roughness` where no map is given; normal variance widens it.
    pub specular_roughness: f32,
    /// `base_color` where no map is given but opacity is.
    pub base_color: [f32; 3],
    /// `base_metalness` where no map is given.
    pub base_metalness: f32,
    /// `specular_roughness_anisotropy` where a direction map is given but no
    /// strength map.
    pub specular_roughness_anisotropy: f32,
    /// Widen roughness mips by the variance of the normals they cover
    /// (Toksvig). Off, roughness is filtered alone and the textures match a
    /// plain bake of the same maps.
    pub fold_normal_variance: bool,
}

impl Default for PackSettings {
    /// Box filtering, no coverage preservation, normal variance folded into
    /// roughness, and OpenPBR's defaults: roughness 0.3, base color 0.8,
    /// metalness 0, anisotropy 0.
    fn default() -> Self {
        Self {
            filter: Filter::Box,
            alpha_cutoff: None,
            specular_roughness: 0.3,
            base_color: [0.8; 3],
            base_metalness: 0.0,
            specular_roughness_anisotropy: 0.0,
            fold_normal_variance: true,
        }
    }
}

/// How a texture is sampled, which decides its pool format and mip rule in a
/// renderer; the same three kinds as `lightweald_material::TextureKind`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum TextureKind {
    /// Color, stored sRGB-encoded and decoded before filtering.
    Color,
    /// Linear data, each channel independent.
    Data,
    /// A tangent-space normal, glTF convention (+Y toward the image top);
    /// renderers may keep only X and Y and rebuild Z.
    Normal,
}

/// One encoded texture with its full mip chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedTexture {
    /// The texture's role, for example `"base_color"`, `"normal"` or `"orm"`.
    pub name: &'static str,
    /// How the texture is sampled.
    pub kind: TextureKind,
    /// The texel format of every level.
    pub format: PixelFormat,
    /// Width of level 0.
    pub width: u32,
    /// Height of level 0.
    pub height: u32,
    /// How samplers should address it: wrapping inputs make repeating
    /// textures.
    pub edge: Edge,
    /// Tightly packed levels, largest first, row 0 first.
    pub levels: Vec<Vec<u8>>,
}

/// What packing measured.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PackReport {
    /// Opacity coverage per level, when an alpha cutoff was set.
    pub coverage: Vec<CoverageLevel>,
    /// Largest normal variance moved into roughness, per level.
    pub normal_variance: Vec<f32>,
}

/// Textures for one profile.
#[derive(Clone, Debug, PartialEq)]
pub struct Bundle {
    /// The profile the textures follow.
    pub profile: Profile,
    /// The textures, in a fixed order: `base_color`, `normal`, `orm`,
    /// `anisotropy` (raw: one per parameter, in [`MaterialMaps`] field
    /// order).
    pub textures: Vec<EncodedTexture>,
    /// Measurements.
    pub report: PackReport,
}

impl Bundle {
    /// The texture named `name`.
    #[must_use]
    pub fn texture(&self, name: &str) -> Option<&EncodedTexture> {
        self.textures.iter().find(|t| t.name == name)
    }
}

fn check(image: Option<&Image>, role: &'static str, channels: usize) -> Result<(), EncodeError> {
    match image {
        Some(i) if i.channels != channels => Err(EncodeError::ChannelMismatch {
            role,
            expected: channels,
            found: i.channels,
        }),
        _ => Ok(()),
    }
}

fn filled(like: &Image, value: f32) -> Image {
    Image {
        channels: 1,
        values: vec![value; like.texels()],
        ..like.clone()
    }
}

/// Roughness mips filtered in `α² = roughness⁴`, where lobe widths add.
fn roughness_mips(image: &Image, filter: Filter) -> MipChain {
    let to = |r: f32| {
        let a = r * r;
        a * a
    };
    let squared = Image {
        values: image.values.iter().map(|&r| to(r)).collect(),
        ..image.clone()
    };
    let chain = data_mips(&squared, filter);
    let mut levels: Vec<Image> = chain
        .levels()
        .iter()
        .map(|l| Image {
            values: l
                .values
                .iter()
                .map(|&a| libm::sqrtf(libm::sqrtf(a.max(0.0))))
                .collect(),
            ..l.clone()
        })
        .collect();
    levels[0] = image.clone();
    MipChain::from_levels(levels)
}

fn encode(
    name: &'static str,
    format: PixelFormat,
    levels: usize,
    edge: Edge,
    mut texel: impl FnMut(usize, usize, &mut Vec<u8>),
    size: impl Fn(usize) -> (u32, u32),
) -> EncodedTexture {
    let (width, height) = size(0);
    let levels = (0..levels)
        .map(|level| {
            let (w, h) = size(level);
            let count = w as usize * h as usize;
            let mut bytes = Vec::with_capacity(count * format.bytes_per_texel());
            for i in 0..count {
                texel(level, i, &mut bytes);
            }
            bytes
        })
        .collect();
    EncodedTexture {
        name,
        kind: if format.is_srgb() {
            TextureKind::Color
        } else {
            TextureKind::Data
        },
        format,
        width,
        height,
        edge,
        levels,
    }
}

/// Builds mips for every present map and packs them for `profile`.
///
/// Anisotropy (Lightweald's and glTF's `KHR_materials_anisotropy` layout)
/// packs the axis in RG and the strength in B. Each level's strength is scaled
/// by how much the axes it averages agree, so regions whose axes disagree
/// fade toward isotropic instead of flickering.
///
/// Opacity mips preserve coverage at `settings.alpha_cutoff`. Normal mips
/// widen roughness by their variance, so the roughness channel is written
/// whenever normals are present, even without a roughness map. Lightweald and
/// glTF normal maps store `+Y` toward the top of the image, the glTF
/// convention: the input's `+Y` runs along increasing rows, which files store
/// top to bottom, so Y is negated.
pub fn pack(
    maps: &MaterialMaps,
    profile: Profile,
    settings: &PackSettings,
) -> Result<Bundle, EncodeError> {
    check(maps.base_color.as_ref(), "base_color", 3)?;
    check(maps.opacity.as_ref(), "opacity", 1)?;
    check(maps.normal.as_ref(), "normal", 3)?;
    check(maps.specular_roughness.as_ref(), "specular_roughness", 1)?;
    check(maps.base_metalness.as_ref(), "base_metalness", 1)?;
    check(maps.occlusion.as_ref(), "occlusion", 1)?;
    check(
        maps.anisotropy_direction.as_ref(),
        "anisotropy_direction",
        2,
    )?;
    check(
        maps.specular_roughness_anisotropy.as_ref(),
        "specular_roughness_anisotropy",
        1,
    )?;
    let present = [
        ("base_color", maps.base_color.as_ref()),
        ("opacity", maps.opacity.as_ref()),
        ("normal", maps.normal.as_ref()),
        ("specular_roughness", maps.specular_roughness.as_ref()),
        ("base_metalness", maps.base_metalness.as_ref()),
        ("occlusion", maps.occlusion.as_ref()),
        ("anisotropy_direction", maps.anisotropy_direction.as_ref()),
        (
            "specular_roughness_anisotropy",
            maps.specular_roughness_anisotropy.as_ref(),
        ),
    ];
    let Some(first) = present.iter().find_map(|(_, i)| *i) else {
        return Ok(Bundle {
            profile,
            textures: Vec::new(),
            report: PackReport::default(),
        });
    };
    for (role, image) in present {
        if image.is_some_and(|i| !i.same_grid(first)) {
            return Err(EncodeError::Incompatible { role });
        }
    }
    if !(0.0..=1.0).contains(&settings.specular_roughness) {
        return Err(EncodeError::InvalidParameter {
            name: "specular_roughness",
        });
    }
    let edge = first.edge;
    let mut report = PackReport::default();

    // Color and opacity together, so color is filtered premultiplied.
    let color = if maps.base_color.is_some() || maps.opacity.is_some() {
        let rgb = maps.base_color.clone();
        let alpha = maps.opacity.clone();
        let mut values = Vec::with_capacity(first.texels() * 4);
        for i in 0..first.texels() {
            match &rgb {
                Some(c) => values.extend_from_slice(&c.values[i * 3..i * 3 + 3]),
                None => values.extend_from_slice(&settings.base_color),
            }
            values.push(alpha.as_ref().map_or(1.0, |a| a.values[i]));
        }
        let rgba = Image {
            channels: 4,
            values,
            ..first.clone()
        };
        let mut chain = color_mips(&rgba, settings.filter);
        if let (Some(cutoff), Some(_)) = (settings.alpha_cutoff, &maps.opacity) {
            report.coverage = preserve_coverage(&mut chain, 3, cutoff)?;
        }
        Some(chain)
    } else {
        None
    };

    let (normals, roughness) = match &maps.normal {
        Some(n) => {
            let chain = normal_mips(
                n,
                maps.specular_roughness.as_ref(),
                settings.specular_roughness,
                settings.filter,
            )?;
            report.normal_variance = chain.max_variance.clone();
            let roughness = if settings.fold_normal_variance {
                Some(chain.roughness)
            } else {
                maps.specular_roughness
                    .as_ref()
                    .map(|r| roughness_mips(r, settings.filter))
            };
            (Some(chain.normals), roughness)
        }
        None => (
            None,
            maps.specular_roughness
                .as_ref()
                .map(|r| roughness_mips(r, settings.filter)),
        ),
    };
    let metalness = maps
        .base_metalness
        .as_ref()
        .map(|m| data_mips(m, settings.filter));
    let occlusion = maps
        .occlusion
        .as_ref()
        .map(|o| data_mips(o, settings.filter));
    let direction = maps
        .anisotropy_direction
        .as_ref()
        .map(|d| direction_mips(d, settings.filter))
        .transpose()?;
    let strength = maps
        .specular_roughness_anisotropy
        .as_ref()
        .map(|a| data_mips(a, settings.filter));

    let reference = data_mips(&filled(first, 0.0), Filter::Box);
    let level_count = reference.levels().len();
    let size = |level: usize| {
        let l = &reference.levels()[level];
        (l.width, l.height)
    };
    let value = |chain: &Option<MipChain>, level: usize, i: usize, default: f32| {
        chain
            .as_ref()
            .map_or(default, |c| c.levels()[level].values[i])
    };

    let mut textures = Vec::new();
    if let Some(color) = &color {
        let format = if profile == Profile::Raw {
            PixelFormat::Rgba8Unorm
        } else {
            PixelFormat::Rgba8Srgb
        };
        textures.push(encode(
            "base_color",
            format,
            level_count,
            edge,
            |level, i, out| {
                let t = &color.levels()[level].values[i * 4..i * 4 + 4];
                for &c in &t[..3] {
                    let c = if format.is_srgb() {
                        linear_to_srgb(c)
                    } else {
                        c
                    };
                    out.push(quantize_unorm8(c));
                }
                out.push(quantize_unorm8(if profile == Profile::Raw {
                    1.0
                } else {
                    t[3]
                }));
            },
            size,
        ));
        if profile == Profile::Raw && maps.opacity.is_some() {
            textures.push(encode(
                "opacity",
                PixelFormat::R8Unorm,
                level_count,
                edge,
                |level, i, out| {
                    out.push(quantize_unorm8(color.levels()[level].values[i * 4 + 3]));
                },
                size,
            ));
        }
    }
    if let Some(normals) = &normals {
        let unit = |v: f32| v * 0.5 + 0.5;
        let (format, name) = match profile {
            Profile::Lightweald => (PixelFormat::Rg8Unorm, "normal"),
            Profile::Gltf | Profile::Raw => (PixelFormat::Rgba8Unorm, "normal"),
        };
        textures.push(encode(
            name,
            format,
            level_count,
            edge,
            |level, i, out| {
                let n = &normals.levels()[level].values[i * 3..i * 3 + 3];
                let y = if profile == Profile::Raw { n[1] } else { -n[1] };
                out.push(quantize_unorm8(unit(n[0])));
                out.push(quantize_unorm8(unit(y)));
                if format == PixelFormat::Rgba8Unorm {
                    out.push(quantize_unorm8(unit(n[2])));
                    out.push(255);
                }
            },
            size,
        ));
    }
    match profile {
        Profile::Lightweald | Profile::Gltf => {
            if occlusion.is_some() || roughness.is_some() || metalness.is_some() {
                textures.push(encode(
                    "orm",
                    PixelFormat::Rgba8Unorm,
                    level_count,
                    edge,
                    |level, i, out| {
                        out.push(quantize_unorm8(value(&occlusion, level, i, 1.0)));
                        out.push(quantize_unorm8(value(
                            &roughness,
                            level,
                            i,
                            settings.specular_roughness,
                        )));
                        out.push(quantize_unorm8(value(
                            &metalness,
                            level,
                            i,
                            settings.base_metalness,
                        )));
                        out.push(255);
                    },
                    size,
                ));
            }
        }
        Profile::Raw => {
            for (name, chain) in [
                ("specular_roughness", &roughness),
                ("base_metalness", &metalness),
                ("occlusion", &occlusion),
            ] {
                if chain.is_some() {
                    textures.push(encode(
                        name,
                        PixelFormat::R8Unorm,
                        level_count,
                        edge,
                        |level, i, out| out.push(quantize_unorm8(value(chain, level, i, 0.0))),
                        size,
                    ));
                }
            }
        }
    }
    if let Some(direction) = &direction {
        // The axis θ from the averaged doubled-angle vector, and how much the
        // texels it averages agree (1 for one axis, 0 when they cancel).
        let axis = |level: usize, i: usize| {
            let v = &direction.levels()[level].values[i * 2..i * 2 + 2];
            let coherence = libm::sqrtf(v[0] * v[0] + v[1] * v[1]).min(1.0);
            let theta = 0.5 * libm::atan2f(v[1], v[0]);
            (libm::cosf(theta), libm::sinf(theta), coherence)
        };
        let unit = |v: f32| v * 0.5 + 0.5;
        match profile {
            Profile::Lightweald | Profile::Gltf => textures.push(encode(
                "anisotropy",
                PixelFormat::Rgba8Unorm,
                level_count,
                edge,
                |level, i, out| {
                    let (c, s, coherence) = axis(level, i);
                    let strength =
                        value(&strength, level, i, settings.specular_roughness_anisotropy);
                    // +Y toward the image top, as for normals. An axis and its
                    // opposite are one direction, so the sign of the pair is free.
                    out.push(quantize_unorm8(unit(c)));
                    out.push(quantize_unorm8(unit(-s)));
                    out.push(quantize_unorm8(strength * coherence));
                    out.push(255);
                },
                size,
            )),
            Profile::Raw => {
                textures.push(encode(
                    "anisotropy_direction",
                    PixelFormat::Rg8Unorm,
                    level_count,
                    edge,
                    |level, i, out| {
                        let (c, s, _) = axis(level, i);
                        out.push(quantize_unorm8(unit(c)));
                        out.push(quantize_unorm8(unit(s)));
                    },
                    size,
                ));
                textures.push(encode(
                    "specular_roughness_anisotropy",
                    PixelFormat::R8Unorm,
                    level_count,
                    edge,
                    |level, i, out| {
                        let (_, _, coherence) = axis(level, i);
                        let strength =
                            value(&strength, level, i, settings.specular_roughness_anisotropy);
                        out.push(quantize_unorm8(strength * coherence));
                    },
                    size,
                ));
            }
        }
    }
    if profile != Profile::Raw {
        for texture in &mut textures {
            if texture.name == "normal" {
                texture.kind = TextureKind::Normal;
            }
        }
    }
    Ok(Bundle {
        profile,
        textures,
        report,
    })
}

/// Encodes an arbitrary chain of data, such as height or identifiers, as one
/// texture in `format`.
///
/// Values are written as they are: clamped and quantized for unsigned
/// normalized formats (sRGB-encoded RGB for [`PixelFormat::Rgba8Srgb`]), and
/// as little-endian IEEE bits for [`PixelFormat::R32Float`]. The chain's
/// channel count must match the format's. The texture's kind is
/// [`TextureKind::Color`] for sRGB formats and [`TextureKind::Data`]
/// otherwise.
pub fn encode_data(
    name: &'static str,
    chain: &MipChain,
    format: PixelFormat,
) -> Result<EncodedTexture, EncodeError> {
    let base = chain.base();
    if base.channels != format.channels() {
        return Err(EncodeError::ChannelMismatch {
            role: name,
            expected: format.channels(),
            found: base.channels,
        });
    }
    let levels = chain
        .levels()
        .iter()
        .map(|level| {
            let mut bytes = Vec::with_capacity(level.values.len() * format.bytes_per_channel());
            for (i, &v) in level.values.iter().enumerate() {
                match format {
                    PixelFormat::R8Unorm | PixelFormat::Rg8Unorm | PixelFormat::Rgba8Unorm => {
                        bytes.push(quantize_unorm8(v));
                    }
                    PixelFormat::Rgba8Srgb => bytes.push(quantize_unorm8(if i % 4 == 3 {
                        v
                    } else {
                        linear_to_srgb(v)
                    })),
                    PixelFormat::R16Unorm => {
                        bytes.extend_from_slice(&quantize_unorm16(v).to_le_bytes());
                    }
                    PixelFormat::R32Float => bytes.extend_from_slice(&v.to_le_bytes()),
                }
            }
            bytes
        })
        .collect();
    Ok(EncodedTexture {
        name,
        kind: if format.is_srgb() {
            TextureKind::Color
        } else {
            TextureKind::Data
        },
        format,
        width: base.width,
        height: base.height,
        edge: base.edge,
        levels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(channels: usize, texel: &[f32]) -> Image {
        Image::new(4, 4, channels, Edge::Wrap, texel.repeat(16)).unwrap()
    }

    #[test]
    fn profiles_place_channels() {
        let s = core::f32::consts::FRAC_1_SQRT_2;
        let maps = MaterialMaps {
            base_color: Some(flat(3, &[0.5, 0.0, 1.0])),
            normal: Some(flat(3, &[0.0, s, s])),
            base_metalness: Some(flat(1, &[1.0])),
            ..MaterialMaps::default()
        };
        let lw = pack(&maps, Profile::Lightweald, &PackSettings::default()).unwrap();
        let names: Vec<_> = lw.textures.iter().map(|t| t.name).collect();
        assert_eq!(names, ["base_color", "normal", "orm"]);
        assert_eq!(
            &lw.texture("base_color").unwrap().levels[0][..4],
            [188, 0, 255, 255]
        );
        // +Y along rows becomes −Y (toward the image top).
        let normal = lw.texture("normal").unwrap();
        assert_eq!(normal.format, PixelFormat::Rg8Unorm);
        assert_eq!(
            &normal.levels[0][..2],
            [128, quantize_unorm8(-s * 0.5 + 0.5)]
        );
        // Occlusion 1, roughness 0.3 (flat normals add nothing), metalness 1.
        assert_eq!(
            &lw.texture("orm").unwrap().levels[0][..4],
            [255, 77, 255, 255]
        );

        let gltf = pack(&maps, Profile::Gltf, &PackSettings::default()).unwrap();
        let normal = gltf.texture("normal").unwrap();
        assert_eq!(normal.format, PixelFormat::Rgba8Unorm);
        assert_eq!(normal.levels[0][2], quantize_unorm8(s * 0.5 + 0.5));

        let raw = pack(&maps, Profile::Raw, &PackSettings::default()).unwrap();
        let names: Vec<_> = raw.textures.iter().map(|t| t.name).collect();
        assert_eq!(
            names,
            [
                "base_color",
                "normal",
                "specular_roughness",
                "base_metalness"
            ]
        );
        assert_eq!(
            raw.texture("normal").unwrap().levels[0][1],
            quantize_unorm8(s * 0.5 + 0.5)
        );
    }

    #[test]
    fn mismatched_inputs_are_refused() {
        let maps = MaterialMaps {
            base_color: Some(flat(3, &[0.5; 3])),
            occlusion: Some(Image::new(2, 2, 1, Edge::Wrap, vec![1.0; 4]).unwrap()),
            ..MaterialMaps::default()
        };
        assert_eq!(
            pack(&maps, Profile::Gltf, &PackSettings::default()),
            Err(EncodeError::Incompatible { role: "occlusion" })
        );
        let maps = MaterialMaps {
            normal: Some(flat(1, &[1.0])),
            ..MaterialMaps::default()
        };
        assert!(matches!(
            pack(&maps, Profile::Gltf, &PackSettings::default()),
            Err(EncodeError::ChannelMismatch { role: "normal", .. })
        ));
        let empty = pack(
            &MaterialMaps::default(),
            Profile::Raw,
            &PackSettings::default(),
        );
        assert!(empty.unwrap().textures.is_empty());
    }

    #[test]
    fn opacity_coverage_is_reported() {
        let values: Vec<f32> = (0..16)
            .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
            .collect();
        let maps = MaterialMaps {
            opacity: Some(Image::new(4, 4, 1, Edge::Wrap, values).unwrap()),
            ..MaterialMaps::default()
        };
        let settings = PackSettings {
            alpha_cutoff: Some(0.5),
            ..PackSettings::default()
        };
        let bundle = pack(&maps, Profile::Lightweald, &settings).unwrap();
        assert_eq!(bundle.report.coverage.len(), 3);
        let color = bundle.texture("base_color").unwrap();
        // No base color map: OpenPBR's 0.8 default fills RGB.
        assert_eq!(color.levels[0][0], quantize_unorm8(linear_to_srgb(0.8)));
    }

    #[test]
    fn anisotropy_packs_axis_and_agreement() {
        // θ = 30° everywhere, strength 0.8: RG is (cos θ, −sin θ), B is 0.8.
        let theta = core::f32::consts::FRAC_PI_6;
        let doubled = [libm::cosf(2.0 * theta), libm::sinf(2.0 * theta)];
        let maps = MaterialMaps {
            anisotropy_direction: Some(flat(2, &doubled)),
            specular_roughness_anisotropy: Some(flat(1, &[0.8])),
            ..MaterialMaps::default()
        };
        let bundle = pack(&maps, Profile::Gltf, &PackSettings::default()).unwrap();
        let aniso = bundle.texture("anisotropy").unwrap();
        let unit = |v: f32| quantize_unorm8(v * 0.5 + 0.5);
        assert_eq!(
            &aniso.levels[0][..4],
            [
                unit(libm::cosf(theta)),
                unit(-libm::sinf(theta)),
                quantize_unorm8(0.8),
                255
            ]
        );

        // Perpendicular axes in every 2×2 block cancel: no anisotropy left.
        let axis = |a: f32| [libm::cosf(2.0 * a), libm::sinf(2.0 * a)];
        let half_pi = core::f32::consts::FRAC_PI_2;
        let values: Vec<f32> = (0..16)
            .flat_map(|i| {
                if (i + i / 4) % 2 == 0 {
                    axis(0.0)
                } else {
                    axis(half_pi)
                }
            })
            .collect();
        let maps = MaterialMaps {
            anisotropy_direction: Some(Image::new(4, 4, 2, Edge::Wrap, values).unwrap()),
            ..MaterialMaps::default()
        };
        let settings = PackSettings {
            specular_roughness_anisotropy: 1.0,
            ..PackSettings::default()
        };
        let raw = pack(&maps, Profile::Raw, &settings).unwrap();
        let strength = raw.texture("specular_roughness_anisotropy").unwrap();
        assert_eq!(strength.levels[0][0], 255);
        assert!(
            strength.levels[1].iter().all(|&b| b == 0),
            "{:?}",
            strength.levels[1]
        );
    }

    #[test]
    fn data_encodes_in_wide_formats() {
        let height = Image::new(2, 2, 1, Edge::Clamp, vec![0.0, 0.5, 1.0, 2.0]).unwrap();
        let chain = data_mips(&height, Filter::Box);
        let short = encode_data("height", &chain, PixelFormat::R16Unorm).unwrap();
        assert_eq!(short.levels[0], [0, 0, 0, 128, 255, 255, 255, 255]);
        assert_eq!(
            short.levels[1],
            57_343_u16.to_le_bytes(),
            "unclamped values average: (0 + 0.5 + 1 + 2) / 4 = 0.875"
        );
        let float = encode_data("height", &chain, PixelFormat::R32Float).unwrap();
        assert_eq!(float.levels[0][12..16], 2.0_f32.to_le_bytes(), "unclamped");
        assert_eq!(float.levels[1], 0.875_f32.to_le_bytes());
        assert!(encode_data("height", &chain, PixelFormat::Rg8Unorm).is_err());
        let file = crate::ktx2::write(&short);
        assert_eq!(u32::from_le_bytes(file[12..16].try_into().unwrap()), 70);
    }

    #[cfg(feature = "std")]
    #[test]
    fn png_writes_sixteen_bits_and_refuses_floats() {
        let image = Image::new(2, 1, 1, Edge::Clamp, vec![0.25, 1.0]).unwrap();
        let chain = data_mips(&image, Filter::Box);
        let short = encode_data("h", &chain, PixelFormat::R16Unorm).unwrap();
        let bytes = crate::png::write(&short).unwrap();
        assert_eq!(&bytes[1..4], b"PNG");
        let float = encode_data("h", &chain, PixelFormat::R32Float).unwrap();
        assert!(matches!(
            crate::png::write(&float),
            Err(crate::png::PngError::UnsupportedFormat(
                PixelFormat::R32Float
            ))
        ));
    }

    #[test]
    fn variance_folding_can_be_turned_off() {
        let s = core::f32::consts::FRAC_1_SQRT_2;
        let values: Vec<f32> = (0..16)
            .flat_map(|i| {
                if i % 2 == 0 {
                    [s, 0.0, s]
                } else {
                    [-s, 0.0, s]
                }
            })
            .collect();
        let maps = MaterialMaps {
            normal: Some(Image::new(4, 4, 3, Edge::Wrap, values).unwrap()),
            specular_roughness: Some(flat(1, &[0.3])),
            ..MaterialMaps::default()
        };
        let roughness = |fold| {
            let settings = PackSettings {
                fold_normal_variance: fold,
                ..PackSettings::default()
            };
            let bundle = pack(&maps, Profile::Lightweald, &settings).unwrap();
            bundle.texture("orm").unwrap().levels[1][1]
        };
        assert_eq!(roughness(false), quantize_unorm8(0.3));
        assert!(roughness(true) > roughness(false) + 50);
    }
}
