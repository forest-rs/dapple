// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Packing material maps into textures for a consumer.

use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

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
    /// occlusion and metallic-roughness slots can reference. Lightweald's
    /// surface parameters hold every OpenPBR parameter, so any constant but
    /// occlusion is carried as one.
    Lightweald,
    /// glTF 2.0: `base_color` as Lightweald, `normal` as RGB tangent-space
    /// XYZ, and the `orm` texture shared by `occlusionTexture` and
    /// `metallicRoughnessTexture`. Constants are carried as glTF's factors,
    /// except occlusion (glTF has no occlusion factor) and coat tint (no
    /// extension has one).
    Gltf,
    /// One texture per parameter map, linear 8-bit, normals in the input
    /// frame, and every constant as one; for inspection and tools rather
    /// than rendering.
    Raw,
}

/// One parameter's values for [`pack()`]: the same everywhere, or one per
/// texel.
///
/// `N` is the parameter's component count: 1 for scalars, 2 for the
/// anisotropy direction, 3 for colors.
#[derive(Clone, Debug, PartialEq)]
pub enum Input<const N: usize> {
    /// One value everywhere. Profiles state it as a factor in
    /// [`Bundle::constants`] instead of expanding it into a texture, unless
    /// a texture is written for the other parameters it shares.
    Constant([f32; N]),
    /// One value per texel: an image of `N` channels.
    Map(Image),
}

impl<const N: usize> Input<N> {
    /// The image, when the input is a map.
    #[must_use]
    pub fn map(&self) -> Option<&Image> {
        match self {
            Self::Map(image) => Some(image),
            Self::Constant(_) => None,
        }
    }

    /// The value, when the input is a constant.
    #[must_use]
    pub const fn constant(&self) -> Option<[f32; N]> {
        match self {
            Self::Constant(value) => Some(*value),
            Self::Map(_) => None,
        }
    }
}

impl<const N: usize> From<Image> for Input<N> {
    fn from(image: Image) -> Self {
        Self::Map(image)
    }
}

impl<const N: usize> From<[f32; N]> for Input<N> {
    fn from(value: [f32; N]) -> Self {
        Self::Constant(value)
    }
}

impl From<f32> for Input<1> {
    fn from(value: f32) -> Self {
        Self::Constant([value])
    }
}

/// Material parameters to pack, each a map or a constant. All maps must
/// share a size and edge policy. Values are linear. A parameter left `None`
/// is not given: a texture that holds it anyway (because it shares the
/// texture with a parameter that is given) holds the OpenPBR
/// specification's default.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MaterialMaps {
    /// OpenPBR `base_color`.
    pub base_color: Option<Input<3>>,
    /// OpenPBR `geometry_opacity`.
    pub opacity: Option<Input<1>>,
    /// Unit shading normals, 3 channels, in the domain frame of
    /// `dapple_raster::HeightToNormal`: `+X` along increasing column, `+Y`
    /// along increasing row, `+Z` out of the surface. A map only: a
    /// constant normal is a flat surface, which is the default.
    pub normal: Option<Image>,
    /// OpenPBR `specular_roughness`.
    pub specular_roughness: Option<Input<1>>,
    /// OpenPBR `base_metalness`.
    pub base_metalness: Option<Input<1>>,
    /// Ambient occlusion; an auxiliary output, not an OpenPBR parameter.
    pub occlusion: Option<Input<1>>,
    /// The anisotropy axis: the doubled-angle vector `(cos 2θ, sin 2θ)` of
    /// `dapple_field`'s `PortType::Direction`, with `θ` measured in the
    /// normal map's domain frame. The default is `(1, 0)`, the `+X` axis.
    /// Lightweald and glTF flip Y as for normals, so a constant axis `θ` is
    /// a glTF `anisotropyRotation` of `-θ`.
    pub anisotropy_direction: Option<Input<2>>,
    /// OpenPBR `specular_roughness_anisotropy`.
    pub specular_roughness_anisotropy: Option<Input<1>>,
    /// OpenPBR `subsurface_weight`. On a thin-walled material
    /// (`geometry_thin_walled`, a material constant rather than a map), this
    /// is diffuse translucency: the share of light scattered through a thin
    /// sheet such as a leaf.
    pub subsurface_weight: Option<Input<1>>,
    /// OpenPBR `subsurface_color`: the tint of that scattered light.
    pub subsurface_color: Option<Input<3>>,
    /// OpenPBR `transmission_weight`: specular transmission, as through
    /// glass.
    pub transmission_weight: Option<Input<1>>,
    /// OpenPBR `coat_weight`: the fraction covered by the dielectric coat.
    pub coat_weight: Option<Input<1>>,
    /// OpenPBR `coat_roughness`.
    pub coat_roughness: Option<Input<1>>,
    /// OpenPBR `coat_color`: the coat's absorption tint.
    pub coat_color: Option<Input<3>>,
}

/// OpenPBR 1.1's defaults for what [`MaterialMaps`] carries.
mod default {
    pub(super) const BASE_COLOR: [f32; 3] = [0.8; 3];
    pub(super) const OPACITY: [f32; 1] = [1.0];
    pub(super) const NORMAL: [f32; 3] = [0.0, 0.0, 1.0];
    pub(super) const SPECULAR_ROUGHNESS: [f32; 1] = [0.3];
    pub(super) const BASE_METALNESS: [f32; 1] = [0.0];
    pub(super) const OCCLUSION: [f32; 1] = [1.0];
    pub(super) const ANISOTROPY_DIRECTION: [f32; 2] = [1.0, 0.0];
    pub(super) const SPECULAR_ROUGHNESS_ANISOTROPY: [f32; 1] = [0.0];
    pub(super) const SUBSURFACE_WEIGHT: [f32; 1] = [0.0];
    pub(super) const SUBSURFACE_COLOR: [f32; 3] = [0.8; 3];
    pub(super) const TRANSMISSION_WEIGHT: [f32; 1] = [0.0];
    pub(super) const COAT_WEIGHT: [f32; 1] = [0.0];
    pub(super) const COAT_ROUGHNESS: [f32; 1] = [0.0];
    pub(super) const COAT_COLOR: [f32; 3] = [1.0; 3];
}

/// One [`MaterialMaps`] field, seen without its type.
struct Entry<'a> {
    name: &'static str,
    channels: usize,
    map: Option<&'a Image>,
    constant: Option<&'a [f32]>,
    default: &'static [f32],
    /// Whether constants must lie in `[0, 1]`.
    unit: bool,
}

fn entry<'a, const N: usize>(
    name: &'static str,
    input: &'a Option<Input<N>>,
    default: &'static [f32; N],
    unit: bool,
) -> Entry<'a> {
    Entry {
        name,
        channels: N,
        map: input.as_ref().and_then(Input::map),
        constant: match input {
            Some(Input::Constant(value)) => Some(value.as_slice()),
            _ => None,
        },
        default,
        unit,
    }
}

impl MaterialMaps {
    /// Every field, in declaration order.
    fn entries(&self) -> [Entry<'_>; 14] {
        use default as d;
        [
            entry("base_color", &self.base_color, &d::BASE_COLOR, true),
            entry("opacity", &self.opacity, &d::OPACITY, true),
            Entry {
                name: "normal",
                channels: 3,
                map: self.normal.as_ref(),
                constant: None,
                default: &d::NORMAL,
                unit: false,
            },
            entry(
                "specular_roughness",
                &self.specular_roughness,
                &d::SPECULAR_ROUGHNESS,
                true,
            ),
            entry(
                "base_metalness",
                &self.base_metalness,
                &d::BASE_METALNESS,
                true,
            ),
            entry("occlusion", &self.occlusion, &d::OCCLUSION, true),
            entry(
                "anisotropy_direction",
                &self.anisotropy_direction,
                &d::ANISOTROPY_DIRECTION,
                false,
            ),
            entry(
                "specular_roughness_anisotropy",
                &self.specular_roughness_anisotropy,
                &d::SPECULAR_ROUGHNESS_ANISOTROPY,
                true,
            ),
            entry(
                "subsurface_weight",
                &self.subsurface_weight,
                &d::SUBSURFACE_WEIGHT,
                true,
            ),
            entry(
                "subsurface_color",
                &self.subsurface_color,
                &d::SUBSURFACE_COLOR,
                true,
            ),
            entry(
                "transmission_weight",
                &self.transmission_weight,
                &d::TRANSMISSION_WEIGHT,
                true,
            ),
            entry("coat_weight", &self.coat_weight, &d::COAT_WEIGHT, true),
            entry(
                "coat_roughness",
                &self.coat_roughness,
                &d::COAT_ROUGHNESS,
                true,
            ),
            entry("coat_color", &self.coat_color, &d::COAT_COLOR, true),
        ]
    }
}

/// Whether `profile` can state parameter `name` as a constant, without a
/// texture.
fn states_constant(profile: Profile, name: &str) -> bool {
    match profile {
        Profile::Raw => true,
        Profile::Lightweald => name != "occlusion",
        Profile::Gltf => !matches!(name, "occlusion" | "coat_color"),
    }
}

/// Settings for [`pack()`].
///
/// Parameter values are not settings: give a parameter that is the same
/// everywhere as an [`Input::Constant`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PackSettings {
    /// Mip filter.
    pub filter: Filter,
    /// Alpha-test cutoff whose coverage opacity mips preserve; `None` filters
    /// opacity like any data.
    pub alpha_cutoff: Option<f32>,
    /// Widen roughness mips by the variance of the normals they cover
    /// (Toksvig). Off, roughness is filtered alone and the textures match a
    /// plain bake of the same maps.
    pub fold_normal_variance: bool,
}

impl Default for PackSettings {
    /// Box filtering, no coverage preservation, and normal variance folded
    /// into roughness.
    fn default() -> Self {
        Self {
            filter: Filter::Box,
            alpha_cutoff: None,
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

/// What packing measured, and what it could not carry.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PackReport {
    /// Opacity coverage per level, when an alpha cutoff was set.
    pub coverage: Vec<CoverageLevel>,
    /// Largest normal variance moved into roughness, per level.
    pub normal_variance: Vec<f32>,
    /// Parameters given that the profile can carry neither in a texture
    /// nor as a constant, by [`MaterialMaps`] field name in field order;
    /// their values are not in the bundle. A constant equal to the
    /// specification's default loses nothing and is not listed.
    pub unsupported: Vec<&'static str>,
}

/// Where a [`Bundle`] holds one parameter: which channels of which texture,
/// and what their values stand for.
///
/// A parameter value is `range[0] + (range[1] - range[0]) * t`, where `t`
/// is the channel's texel as a `[0, 1]` fraction of full scale after the
/// format's transfer function (sRGB channels are decoded first). When two
/// placements cover the same channels, the channels hold the product of
/// their values.
#[derive(Clone, Debug, PartialEq)]
pub struct Placement {
    /// The parameter, by its [`MaterialMaps`] field name, for example
    /// `"coat_weight"` or `"normal"`.
    pub name: &'static str,
    /// The [`EncodedTexture::name`] of the texture that holds it.
    pub texture: &'static str,
    /// The texel channels it occupies, as many as the parameter has.
    pub channels: Range<usize>,
    /// The values texel 0 and full scale stand for: `[0, 1]` for most
    /// parameters, `[-1, 1]` for unit vectors.
    pub range: [f32; 2],
}

/// A parameter a [`Bundle`] carries as one value instead of a texture.
#[derive(Clone, Debug, PartialEq)]
pub struct Constant {
    /// The parameter, by its [`MaterialMaps`] field name.
    pub name: &'static str,
    /// Its value as given, one component per channel of the parameter.
    pub value: Vec<f32>,
}

/// Textures and constants for one profile.
///
/// Every parameter given to [`pack()`] ends in exactly one place: a
/// [`Placement`] in a texture, a [`Constant`], or
/// [`PackReport::unsupported`]. The one exception is a constant equal to
/// the specification's default that the profile cannot state, which loses
/// nothing. A parameter that was not given is at the specification's
/// default.
#[derive(Clone, Debug, PartialEq)]
pub struct Bundle {
    /// The profile the textures follow.
    pub profile: Profile,
    /// The textures, in a fixed order: `base_color`, `normal`, `orm`,
    /// `anisotropy`, then translucency: `diffuse_transmission` (glTF) and
    /// `transmission`, then the coat: `clearcoat` (glTF) (raw: one per
    /// parameter, in [`MaterialMaps`] field order).
    pub textures: Vec<EncodedTexture>,
    /// Where each parameter went that a texture holds, in texture order.
    ///
    /// Parameters given as maps are placed, and so are constants that
    /// share a texture with a map (an `orm` texture packed for a roughness
    /// map holds a constant metalness too) and values derived from maps
    /// (roughness widened by normal variance). A texture's other channels
    /// hold the specification's default for what the layout puts there,
    /// such as occlusion 1, or padding (alpha 1), so a consumer that
    /// samples a texture for one channel reads correct values in the others
    /// with a factor of 1.
    pub placements: Vec<Placement>,
    /// Parameters given as constants that no texture holds, in
    /// [`MaterialMaps`] field order: a consumer applies each as a factor
    /// (glTF's `roughnessFactor`, `clearcoatFactor` and so on).
    pub constants: Vec<Constant>,
    /// Measurements and losses.
    pub report: PackReport,
}

impl Bundle {
    /// The texture named `name`.
    #[must_use]
    pub fn texture(&self, name: &str) -> Option<&EncodedTexture> {
        self.textures.iter().find(|t| t.name == name)
    }

    /// Where the parameter named `name` (a [`MaterialMaps`] field name) is
    /// held, when a texture holds it.
    #[must_use]
    pub fn placement(&self, name: &str) -> Option<&Placement> {
        self.placements.iter().find(|p| p.name == name)
    }

    /// The value of the parameter named `name` (a [`MaterialMaps`] field
    /// name), when the bundle carries it as a constant.
    #[must_use]
    pub fn constant(&self, name: &str) -> Option<&[f32]> {
        self.constants
            .iter()
            .find(|c| c.name == name)
            .map(|c| c.value.as_slice())
    }
}

const UNIT: [f32; 2] = [0.0, 1.0];
const SIGNED: [f32; 2] = [-1.0, 1.0];

fn place(
    placements: &mut Vec<Placement>,
    texture: &'static str,
    layout: &[(&'static str, Range<usize>, [f32; 2])],
) {
    for (name, channels, range) in layout {
        placements.push(Placement {
            name,
            texture,
            channels: channels.clone(),
            range: *range,
        });
    }
}

/// The map of an input, if it is one.
fn map<const N: usize>(input: &Option<Input<N>>) -> Option<&Image> {
    input.as_ref().and_then(Input::map)
}

/// The value a texture holds where an input has no map: its constant, or
/// the specification's default when it was not given.
fn fill<const N: usize>(input: &Option<Input<N>>, default: [f32; N]) -> [f32; N] {
    input.as_ref().and_then(Input::constant).unwrap_or(default)
}

fn filled(like: &Image, value: f32) -> Image {
    Image {
        channels: 1,
        values: vec![value; like.texels()],
        ..like.clone()
    }
}

/// An interleaved image of `rgb` then `alpha`, each a map or a fill value.
fn rgba(
    like: &Image,
    rgb: Option<&Image>,
    rgb_fill: [f32; 3],
    alpha: Option<&Image>,
    alpha_fill: f32,
) -> Image {
    let mut values = Vec::with_capacity(like.texels() * 4);
    for i in 0..like.texels() {
        match rgb {
            Some(c) => values.extend_from_slice(&c.values[i * 3..i * 3 + 3]),
            None => values.extend_from_slice(&rgb_fill),
        }
        values.push(alpha.map_or(alpha_fill, |a| a.values[i]));
    }
    Image {
        channels: 4,
        values,
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

/// Builds mips for every map and packs them for `profile`, with constants
/// carried as constants where the profile can state them.
///
/// A texture is written when at least one parameter it holds is a map;
/// parameters it shares that are constants, or not given, fill their
/// channels. Every given parameter is placed, carried as a constant, or
/// reported unsupported; see [`Bundle`].
///
/// Anisotropy (Lightweald's and glTF's `KHR_materials_anisotropy` layout)
/// packs the axis in RG and the strength in B. Each level's strength is scaled
/// by how much the axes it averages agree, so regions whose axes disagree
/// fade toward isotropic instead of flickering. A strength map without a
/// direction uses the `+X` axis, the default.
///
/// Opacity mips preserve coverage at `settings.alpha_cutoff`. Normal mips
/// widen roughness by their variance, so the roughness channel is written
/// whenever normals are present, even without a roughness map. Lightweald and
/// glTF normal maps store `+Y` toward the top of the image, the glTF
/// convention: the input's `+Y` runs along increasing rows, which files store
/// top to bottom, so Y is negated.
///
/// Translucency:
///
/// - `transmission` (R `transmission_weight`, linear) is Lightweald's
///   `Transmission` slot and glTF's `KHR_materials_transmission`
///   `transmissionTexture`.
/// - glTF packs thin-walled subsurface as `diffuse_transmission`: sRGB RGB
///   `subsurface_color` and linear A `subsurface_weight`, the layout of
///   `KHR_materials_diffuse_transmission`, whose
///   `diffuseTransmissionColorTexture` reads RGB and
///   `diffuseTransmissionTexture` reads A of the same image. That extension
///   models OpenPBR's thin-walled subsurface; on a volumetric material the
///   projection is lossy. Color mips are weighted by the weight, like base
///   color by opacity.
/// - Lightweald has no subsurface slot yet, so its profile reports
///   subsurface maps in [`PackReport::unsupported`] instead of packing them.
///
/// Coat: glTF packs `clearcoat` (linear R `coat_weight`, G `coat_roughness`),
/// the layout `KHR_materials_clearcoat`'s `clearcoatTexture` (R) and
/// `clearcoatRoughnessTexture` (G) read from one image. The extension has no
/// tint, so `coat_color` is reported unsupported; Lightweald has no coat
/// slot and reports coat maps. Coat roughness mips are filtered in `α²`,
/// like specular roughness.
///
/// # Errors
///
/// [`EncodeError::ChannelMismatch`] for a map with the wrong channel count,
/// [`EncodeError::Incompatible`] for maps of different sizes or edge
/// policies, and [`EncodeError::InvalidParameter`] for a constant that is
/// not finite, or outside `[0, 1]` for a parameter with that range.
pub fn pack(
    maps: &MaterialMaps,
    profile: Profile,
    settings: &PackSettings,
) -> Result<Bundle, EncodeError> {
    let entries = maps.entries();
    for e in &entries {
        if let Some(image) = e.map
            && image.channels != e.channels
        {
            return Err(EncodeError::ChannelMismatch {
                role: e.name,
                expected: e.channels,
                found: image.channels,
            });
        }
        if let Some(value) = e.constant
            && value
                .iter()
                .any(|v| !v.is_finite() || (e.unit && !(0.0..=1.0).contains(v)))
        {
            return Err(EncodeError::InvalidParameter { name: e.name });
        }
    }
    let first = entries.iter().find_map(|e| e.map);
    let mut report = PackReport::default();
    let (textures, placements) = match first {
        Some(first) => {
            for e in &entries {
                if e.map.is_some_and(|i| !i.same_grid(first)) {
                    return Err(EncodeError::Incompatible { role: e.name });
                }
            }
            textures(maps, profile, settings, first, &mut report)?
        }
        None => (Vec::new(), Vec::new()),
    };
    let mut constants = Vec::new();
    for e in &entries {
        if (e.map.is_none() && e.constant.is_none())
            || placements.iter().any(|p: &Placement| p.name == e.name)
        {
            continue;
        }
        match e.constant {
            Some(value) if states_constant(profile, e.name) => constants.push(Constant {
                name: e.name,
                value: value.to_vec(),
            }),
            Some(value) if value == e.default => {}
            _ => report.unsupported.push(e.name),
        }
    }
    Ok(Bundle {
        profile,
        textures,
        placements,
        constants,
        report,
    })
}

#[expect(
    clippy::too_many_lines,
    reason = "one pass over the profiles' layouts, in bundle order"
)]
fn textures(
    maps: &MaterialMaps,
    profile: Profile,
    settings: &PackSettings,
    first: &Image,
    report: &mut PackReport,
) -> Result<(Vec<EncodedTexture>, Vec<Placement>), EncodeError> {
    use default as d;
    let edge = first.edge;
    let raw = profile == Profile::Raw;

    // Color and opacity together, so color is filtered premultiplied.
    let base_color = map(&maps.base_color);
    let opacity = map(&maps.opacity);
    let color = if base_color.is_some() || opacity.is_some() {
        let rgba = rgba(
            first,
            base_color,
            fill(&maps.base_color, d::BASE_COLOR),
            opacity,
            fill(&maps.opacity, d::OPACITY)[0],
        );
        let mut chain = color_mips(&rgba, settings.filter);
        if let (Some(cutoff), Some(_)) = (settings.alpha_cutoff, opacity) {
            report.coverage = preserve_coverage(&mut chain, 3, cutoff)?;
        }
        Some(chain)
    } else {
        None
    };

    let rough_map = map(&maps.specular_roughness);
    let rough_fill = fill(&maps.specular_roughness, d::SPECULAR_ROUGHNESS)[0];
    let (normals, roughness) = match &maps.normal {
        Some(n) => {
            let chain = normal_mips(n, rough_map, rough_fill, settings.filter)?;
            report.normal_variance = chain.max_variance.clone();
            let roughness = if settings.fold_normal_variance {
                Some(chain.roughness)
            } else {
                rough_map.map(|r| roughness_mips(r, settings.filter))
            };
            (Some(chain.normals), roughness)
        }
        None => (None, rough_map.map(|r| roughness_mips(r, settings.filter))),
    };
    let metalness = map(&maps.base_metalness).map(|m| data_mips(m, settings.filter));
    let metal_fill = fill(&maps.base_metalness, d::BASE_METALNESS)[0];
    let occlusion = map(&maps.occlusion).map(|o| data_mips(o, settings.filter));
    let occlusion_fill = fill(&maps.occlusion, d::OCCLUSION)[0];
    let direction = map(&maps.anisotropy_direction)
        .map(|d| direction_mips(d, settings.filter))
        .transpose()?;
    let direction_fill = fill(&maps.anisotropy_direction, d::ANISOTROPY_DIRECTION);
    let strength = map(&maps.specular_roughness_anisotropy).map(|a| data_mips(a, settings.filter));
    let strength_fill = fill(
        &maps.specular_roughness_anisotropy,
        d::SPECULAR_ROUGHNESS_ANISOTROPY,
    )[0];

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
    let mut placements = Vec::new();
    if let Some(color) = &color {
        if !raw {
            textures.push(encode(
                "base_color",
                PixelFormat::Rgba8Srgb,
                level_count,
                edge,
                |level, i, out| {
                    let t = &color.levels()[level].values[i * 4..i * 4 + 4];
                    for &c in &t[..3] {
                        out.push(quantize_unorm8(linear_to_srgb(c)));
                    }
                    out.push(quantize_unorm8(t[3]));
                },
                size,
            ));
            if maps.base_color.is_some() {
                place(&mut placements, "base_color", &[("base_color", 0..3, UNIT)]);
            }
            if maps.opacity.is_some() {
                place(&mut placements, "base_color", &[("opacity", 3..4, UNIT)]);
            }
        } else {
            if base_color.is_some() {
                textures.push(encode(
                    "base_color",
                    PixelFormat::Rgba8Unorm,
                    level_count,
                    edge,
                    |level, i, out| {
                        let t = &color.levels()[level].values[i * 4..i * 4 + 4];
                        for &c in &t[..3] {
                            out.push(quantize_unorm8(c));
                        }
                        out.push(255);
                    },
                    size,
                ));
                place(&mut placements, "base_color", &[("base_color", 0..3, UNIT)]);
            }
            if opacity.is_some() {
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
                place(&mut placements, "opacity", &[("opacity", 0..1, UNIT)]);
            }
        }
    }
    if let Some(normals) = &normals {
        let unit = |v: f32| v * 0.5 + 0.5;
        let format = match profile {
            Profile::Lightweald => PixelFormat::Rg8Unorm,
            Profile::Gltf | Profile::Raw => PixelFormat::Rgba8Unorm,
        };
        textures.push(encode(
            "normal",
            format,
            level_count,
            edge,
            |level, i, out| {
                let n = &normals.levels()[level].values[i * 3..i * 3 + 3];
                let y = if raw { n[1] } else { -n[1] };
                out.push(quantize_unorm8(unit(n[0])));
                out.push(quantize_unorm8(unit(y)));
                if format == PixelFormat::Rgba8Unorm {
                    out.push(quantize_unorm8(unit(n[2])));
                    out.push(255);
                }
            },
            size,
        ));
        let channels = if format == PixelFormat::Rg8Unorm {
            0..2
        } else {
            0..3
        };
        place(&mut placements, "normal", &[("normal", channels, SIGNED)]);
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
                        out.push(quantize_unorm8(value(&occlusion, level, i, occlusion_fill)));
                        out.push(quantize_unorm8(value(&roughness, level, i, rough_fill)));
                        out.push(quantize_unorm8(value(&metalness, level, i, metal_fill)));
                        out.push(255);
                    },
                    size,
                ));
                for (name, placed, channel) in [
                    ("occlusion", maps.occlusion.is_some(), 0),
                    (
                        "specular_roughness",
                        roughness.is_some() || maps.specular_roughness.is_some(),
                        1,
                    ),
                    ("base_metalness", maps.base_metalness.is_some(), 2),
                ] {
                    if placed {
                        place(
                            &mut placements,
                            "orm",
                            &[(name, channel..channel + 1, UNIT)],
                        );
                    }
                }
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
                    place(&mut placements, name, &[(name, 0..1, UNIT)]);
                }
            }
        }
    }
    if direction.is_some() || strength.is_some() {
        // The axis θ from the averaged doubled-angle vector, and how much the
        // texels it averages agree (1 for one axis, 0 when they cancel).
        let axis = |level: usize, i: usize| {
            let v = direction.as_ref().map_or(direction_fill, |d| {
                let v = &d.levels()[level].values[i * 2..i * 2 + 2];
                [v[0], v[1]]
            });
            let coherence = libm::sqrtf(v[0] * v[0] + v[1] * v[1]).min(1.0);
            let theta = 0.5 * libm::atan2f(v[1], v[0]);
            (libm::cosf(theta), libm::sinf(theta), coherence)
        };
        let unit = |v: f32| v * 0.5 + 0.5;
        match profile {
            Profile::Lightweald | Profile::Gltf => {
                textures.push(encode(
                    "anisotropy",
                    PixelFormat::Rgba8Unorm,
                    level_count,
                    edge,
                    |level, i, out| {
                        let (c, s, coherence) = axis(level, i);
                        let strength = value(&strength, level, i, strength_fill);
                        // +Y toward the image top, as for normals. An axis and its
                        // opposite are one direction, so the sign of the pair is free.
                        out.push(quantize_unorm8(unit(c)));
                        out.push(quantize_unorm8(unit(-s)));
                        out.push(quantize_unorm8(strength * coherence));
                        out.push(255);
                    },
                    size,
                ));
                if maps.anisotropy_direction.is_some() {
                    place(
                        &mut placements,
                        "anisotropy",
                        &[("anisotropy_direction", 0..2, SIGNED)],
                    );
                }
                if maps.specular_roughness_anisotropy.is_some() {
                    place(
                        &mut placements,
                        "anisotropy",
                        &[("specular_roughness_anisotropy", 2..3, UNIT)],
                    );
                }
            }
            Profile::Raw => {
                if direction.is_some() {
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
                    place(
                        &mut placements,
                        "anisotropy_direction",
                        &[("anisotropy_direction", 0..2, SIGNED)],
                    );
                }
                // A strength that was not given is 0 whatever the axes do.
                if maps.specular_roughness_anisotropy.is_some() {
                    textures.push(encode(
                        "specular_roughness_anisotropy",
                        PixelFormat::R8Unorm,
                        level_count,
                        edge,
                        |level, i, out| {
                            let (_, _, coherence) = axis(level, i);
                            let strength = value(&strength, level, i, strength_fill);
                            out.push(quantize_unorm8(strength * coherence));
                        },
                        size,
                    ));
                    place(
                        &mut placements,
                        "specular_roughness_anisotropy",
                        &[("specular_roughness_anisotropy", 0..1, UNIT)],
                    );
                }
            }
        }
    }
    let subsurface_weight = map(&maps.subsurface_weight);
    let subsurface_color = map(&maps.subsurface_color);
    match profile {
        Profile::Lightweald => {}
        Profile::Gltf if subsurface_weight.is_some() || subsurface_color.is_some() => {
            // Color weighted by the weight, as base color by opacity.
            let chain = color_mips(
                &rgba(
                    first,
                    subsurface_color,
                    fill(&maps.subsurface_color, d::SUBSURFACE_COLOR),
                    subsurface_weight,
                    fill(&maps.subsurface_weight, d::SUBSURFACE_WEIGHT)[0],
                ),
                settings.filter,
            );
            textures.push(encode(
                "diffuse_transmission",
                PixelFormat::Rgba8Srgb,
                level_count,
                edge,
                |level, i, out| {
                    let t = &chain.levels()[level].values[i * 4..i * 4 + 4];
                    for &c in &t[..3] {
                        out.push(quantize_unorm8(linear_to_srgb(c)));
                    }
                    out.push(quantize_unorm8(t[3]));
                },
                size,
            ));
            if maps.subsurface_color.is_some() {
                place(
                    &mut placements,
                    "diffuse_transmission",
                    &[("subsurface_color", 0..3, UNIT)],
                );
            }
            if maps.subsurface_weight.is_some() {
                place(
                    &mut placements,
                    "diffuse_transmission",
                    &[("subsurface_weight", 3..4, UNIT)],
                );
            }
        }
        Profile::Gltf => {}
        Profile::Raw => {
            if let Some(weight) = subsurface_weight {
                raw_map(
                    &mut textures,
                    &mut placements,
                    "subsurface_weight",
                    weight,
                    settings.filter,
                );
            }
            if let Some(color) = subsurface_color {
                raw_map(
                    &mut textures,
                    &mut placements,
                    "subsurface_color",
                    color,
                    settings.filter,
                );
            }
        }
    }
    if let Some(transmission) = map(&maps.transmission_weight) {
        let chain = Some(data_mips(transmission, settings.filter));
        let name = if raw {
            "transmission_weight"
        } else {
            "transmission"
        };
        place(
            &mut placements,
            name,
            &[("transmission_weight", 0..1, UNIT)],
        );
        textures.push(encode(
            name,
            PixelFormat::R8Unorm,
            level_count,
            edge,
            |level, i, out| out.push(quantize_unorm8(value(&chain, level, i, 0.0))),
            size,
        ));
    }
    let coat_weight = map(&maps.coat_weight).map(|w| data_mips(w, settings.filter));
    let coat_roughness = map(&maps.coat_roughness).map(|r| roughness_mips(r, settings.filter));
    match profile {
        Profile::Lightweald => {}
        Profile::Gltf => {
            if coat_weight.is_some() || coat_roughness.is_some() {
                let weight_fill = fill(&maps.coat_weight, d::COAT_WEIGHT)[0];
                let roughness_fill = fill(&maps.coat_roughness, d::COAT_ROUGHNESS)[0];
                textures.push(encode(
                    "clearcoat",
                    PixelFormat::Rgba8Unorm,
                    level_count,
                    edge,
                    |level, i, out| {
                        out.push(quantize_unorm8(value(&coat_weight, level, i, weight_fill)));
                        out.push(quantize_unorm8(value(
                            &coat_roughness,
                            level,
                            i,
                            roughness_fill,
                        )));
                        out.push(0);
                        out.push(255);
                    },
                    size,
                ));
                for (name, placed, channel) in [
                    ("coat_weight", maps.coat_weight.is_some(), 0),
                    ("coat_roughness", maps.coat_roughness.is_some(), 1),
                ] {
                    if placed {
                        place(
                            &mut placements,
                            "clearcoat",
                            &[(name, channel..channel + 1, UNIT)],
                        );
                    }
                }
            }
        }
        Profile::Raw => {
            for (name, chain) in [
                ("coat_weight", &coat_weight),
                ("coat_roughness", &coat_roughness),
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
                    place(&mut placements, name, &[(name, 0..1, UNIT)]);
                }
            }
            if let Some(color) = map(&maps.coat_color) {
                raw_map(
                    &mut textures,
                    &mut placements,
                    "coat_color",
                    color,
                    settings.filter,
                );
            }
        }
    }
    if !raw {
        for texture in &mut textures {
            if texture.name == "normal" {
                texture.kind = TextureKind::Normal;
            }
        }
    }
    Ok((textures, placements))
}

/// A raw profile's texture for one map of `[0, 1]` data: R8 for one
/// channel, RGBA8 with alpha 1 for three.
fn raw_map(
    textures: &mut Vec<EncodedTexture>,
    placements: &mut Vec<Placement>,
    name: &'static str,
    image: &Image,
    filter: Filter,
) {
    let chain = data_mips(image, filter);
    let channels = image.channels;
    let format = if channels == 1 {
        PixelFormat::R8Unorm
    } else {
        PixelFormat::Rgba8Unorm
    };
    textures.push(encode(
        name,
        format,
        chain.levels().len(),
        image.edge,
        |level, i, out| {
            let t = &chain.levels()[level].values[i * channels..(i + 1) * channels];
            for &c in t {
                out.push(quantize_unorm8(c));
            }
            if channels == 3 {
                out.push(255);
            }
        },
        |level| {
            let l = &chain.levels()[level];
            (l.width, l.height)
        },
    ));
    place(placements, name, &[(name, 0..channels, UNIT)]);
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
            base_color: Some(flat(3, &[0.5, 0.0, 1.0]).into()),
            normal: Some(flat(3, &[0.0, s, s])),
            base_metalness: Some(flat(1, &[1.0]).into()),
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

        let placed = |bundle: &Bundle| {
            bundle
                .placements
                .iter()
                .map(|p| (p.name, p.texture, p.channels.clone()))
                .collect::<Vec<_>>()
        };
        // Roughness is placed though no map was given: normal variance
        // widens it. Occlusion was not given, so it is not placed.
        assert_eq!(
            placed(&lw),
            [
                ("base_color", "base_color", 0..3),
                ("normal", "normal", 0..2),
                ("specular_roughness", "orm", 1..2),
                ("base_metalness", "orm", 2..3),
            ]
        );
        assert_eq!(lw.placement("normal").unwrap().range, [-1.0, 1.0]);
        assert_eq!(gltf.placement("normal").unwrap().channels, 0..3);

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
        for p in &raw.placements {
            assert_eq!(p.name, p.texture, "raw: one texture per parameter");
        }
        assert_eq!(
            raw.texture("normal").unwrap().levels[0][1],
            quantize_unorm8(s * 0.5 + 0.5)
        );
    }

    #[test]
    fn mismatched_inputs_are_refused() {
        let maps = MaterialMaps {
            base_color: Some(flat(3, &[0.5; 3]).into()),
            occlusion: Some(
                Image::new(2, 2, 1, Edge::Wrap, vec![1.0; 4])
                    .unwrap()
                    .into(),
            ),
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
            opacity: Some(Image::new(4, 4, 1, Edge::Wrap, values).unwrap().into()),
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
            anisotropy_direction: Some(flat(2, &doubled).into()),
            specular_roughness_anisotropy: Some(flat(1, &[0.8]).into()),
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
            anisotropy_direction: Some(Image::new(4, 4, 2, Edge::Wrap, values).unwrap().into()),
            ..MaterialMaps::default()
        };
        let maps = MaterialMaps {
            specular_roughness_anisotropy: Some(1.0.into()),
            ..maps
        };
        let raw = pack(&maps, Profile::Raw, &PackSettings::default()).unwrap();
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
            specular_roughness: Some(flat(1, &[0.3]).into()),
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

    #[test]
    fn translucency_packs_per_profile() {
        let maps = MaterialMaps {
            subsurface_weight: Some(flat(1, &[0.6]).into()),
            subsurface_color: Some(flat(3, &[0.5, 0.9, 0.2]).into()),
            transmission_weight: Some(flat(1, &[0.25]).into()),
            ..MaterialMaps::default()
        };
        let settings = PackSettings::default();

        let gltf = pack(&maps, Profile::Gltf, &settings).unwrap();
        let names: Vec<_> = gltf.textures.iter().map(|t| t.name).collect();
        assert_eq!(names, ["diffuse_transmission", "transmission"]);
        let diffuse = gltf.texture("diffuse_transmission").unwrap();
        assert_eq!(diffuse.format, PixelFormat::Rgba8Srgb);
        assert_eq!(diffuse.kind, TextureKind::Color);
        let texel = &diffuse.levels[0][..4];
        assert_eq!(texel[0], quantize_unorm8(linear_to_srgb(0.5)));
        assert_eq!(texel[1], quantize_unorm8(linear_to_srgb(0.9)));
        assert_eq!(texel[3], quantize_unorm8(0.6));
        // A uniform color survives its mips unchanged despite the weighting.
        let last = diffuse.levels.last().unwrap();
        assert_eq!(&last[..3], &texel[..3]);
        let transmission = gltf.texture("transmission").unwrap();
        assert_eq!(transmission.format, PixelFormat::R8Unorm);
        assert_eq!(transmission.levels[0][0], quantize_unorm8(0.25));
        assert!(gltf.report.unsupported.is_empty());
        let weight = gltf.placement("subsurface_weight").unwrap();
        assert_eq!(
            (weight.texture, weight.channels.clone()),
            ("diffuse_transmission", 3..4)
        );

        let lightweald = pack(&maps, Profile::Lightweald, &settings).unwrap();
        let names: Vec<_> = lightweald.textures.iter().map(|t| t.name).collect();
        assert_eq!(names, ["transmission"]);
        assert_eq!(
            lightweald.report.unsupported,
            ["subsurface_weight", "subsurface_color"]
        );

        let raw = pack(&maps, Profile::Raw, &settings).unwrap();
        let names: Vec<_> = raw.textures.iter().map(|t| t.name).collect();
        assert_eq!(
            names,
            [
                "subsurface_weight",
                "subsurface_color",
                "transmission_weight"
            ]
        );

        // A color alone uses the default weight, and a bad default is refused.
        let color_only = MaterialMaps {
            subsurface_color: Some(flat(3, &[0.5, 0.9, 0.2]).into()),
            ..MaterialMaps::default()
        };
        let packed = pack(&color_only, Profile::Gltf, &settings).unwrap();
        assert_eq!(
            packed.texture("diffuse_transmission").unwrap().levels[0][3],
            0
        );
        let bad = MaterialMaps {
            subsurface_weight: Some(2.0.into()),
            ..color_only.clone()
        };
        assert_eq!(
            pack(&bad, Profile::Gltf, &settings),
            Err(EncodeError::InvalidParameter {
                name: "subsurface_weight"
            })
        );
        let mismatched = MaterialMaps {
            subsurface_color: Some(flat(1, &[0.5]).into()),
            ..MaterialMaps::default()
        };
        assert!(pack(&mismatched, Profile::Gltf, &settings).is_err());
    }

    #[test]
    fn coats_pack_per_profile() {
        let maps = MaterialMaps {
            coat_weight: Some(flat(1, &[0.75]).into()),
            coat_roughness: Some(flat(1, &[0.2]).into()),
            coat_color: Some(flat(3, &[0.9, 0.8, 0.5]).into()),
            ..MaterialMaps::default()
        };
        let settings = PackSettings::default();
        let gltf = pack(&maps, Profile::Gltf, &settings).unwrap();
        let clearcoat = gltf.texture("clearcoat").unwrap();
        assert_eq!(clearcoat.levels[0][0], quantize_unorm8(0.75), "weight in R");
        assert_eq!(
            clearcoat.levels[0][1],
            quantize_unorm8(0.2),
            "roughness in G"
        );
        assert_eq!(gltf.report.unsupported, ["coat_color"], "no tint in glTF");
        assert_eq!(gltf.placement("coat_roughness").unwrap().channels, 1..2);
        assert!(gltf.placement("coat_color").is_none());
        let lightweald = pack(&maps, Profile::Lightweald, &settings).unwrap();
        assert!(lightweald.textures.is_empty(), "no coat slot");
        assert_eq!(lightweald.report.unsupported.len(), 3);
        let raw = pack(&maps, Profile::Raw, &settings).unwrap();
        let names: Vec<_> = raw.textures.iter().map(|t| t.name).collect();
        assert_eq!(names, ["coat_weight", "coat_roughness", "coat_color"]);
    }

    #[test]
    fn constants_are_carried_placed_or_reported() {
        let constants = MaterialMaps {
            specular_roughness: Some(0.6.into()),
            base_metalness: Some(1.0.into()),
            occlusion: Some(0.8.into()),
            coat_weight: Some(0.5.into()),
            coat_color: Some([0.9, 0.8, 0.5].into()),
            ..MaterialMaps::default()
        };
        let settings = PackSettings::default();
        let names = |b: &Bundle| b.constants.iter().map(|c| c.name).collect::<Vec<_>>();

        // No map, so no texture: glTF states factors but has no occlusion
        // factor and no coat tint.
        let gltf = pack(&constants, Profile::Gltf, &settings).unwrap();
        assert!(gltf.textures.is_empty());
        assert_eq!(
            names(&gltf),
            ["specular_roughness", "base_metalness", "coat_weight"]
        );
        assert_eq!(gltf.constant("specular_roughness"), Some([0.6].as_slice()));
        assert_eq!(gltf.report.unsupported, ["occlusion", "coat_color"]);
        let lightweald = pack(&constants, Profile::Lightweald, &settings).unwrap();
        assert_eq!(lightweald.report.unsupported, ["occlusion"]);
        assert_eq!(
            lightweald.constant("coat_color"),
            Some([0.9, 0.8, 0.5].as_slice())
        );
        let raw = pack(&constants, Profile::Raw, &settings).unwrap();
        assert_eq!(raw.constants.len(), 5);
        assert!(raw.report.unsupported.is_empty());

        // A map makes the texture; the constants it shares fill it.
        let mapped = MaterialMaps {
            specular_roughness: Some(flat(1, &[0.6]).into()),
            ..constants.clone()
        };
        let gltf = pack(&mapped, Profile::Gltf, &settings).unwrap();
        let orm = gltf.texture("orm").unwrap();
        assert_eq!(
            orm.levels[0][..4],
            [quantize_unorm8(0.8), quantize_unorm8(0.6), 255, 255]
        );
        assert_eq!(gltf.placement("base_metalness").unwrap().channels, 2..3);
        assert_eq!(gltf.placement("occlusion").unwrap().channels, 0..1);
        assert_eq!(names(&gltf), ["coat_weight"]);
        assert_eq!(gltf.report.unsupported, ["coat_color"]);
        let raw = pack(&mapped, Profile::Raw, &settings).unwrap();
        assert_eq!(raw.textures.len(), 1);
        assert_eq!(raw.constants.len(), 4);

        // A default that cannot be stated loses nothing.
        let white = MaterialMaps {
            coat_color: Some([1.0; 3].into()),
            ..MaterialMaps::default()
        };
        let gltf = pack(&white, Profile::Gltf, &settings).unwrap();
        assert!(gltf.constants.is_empty() && gltf.report.unsupported.is_empty());

        let bad = MaterialMaps {
            coat_roughness: Some(f32::NAN.into()),
            ..MaterialMaps::default()
        };
        assert_eq!(
            pack(&bad, Profile::Raw, &settings),
            Err(EncodeError::InvalidParameter {
                name: "coat_roughness"
            })
        );
    }
}
