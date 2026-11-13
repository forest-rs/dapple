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
    /// glTF 2.0 and its material extensions: `base_color` as Lightweald,
    /// `normal` as RGB tangent-space XYZ, the `orm` texture shared by
    /// `occlusionTexture` and `metallicRoughnessTexture`, and one texture
    /// per extension in the layout it reads: `anisotropy`
    /// (`KHR_materials_anisotropy`), `specular` (`KHR_materials_specular`),
    /// `diffuse_transmission`, `transmission`, `clearcoat` and
    /// `clearcoat_normal` (`KHR_materials_clearcoat`), `sheen`
    /// (`KHR_materials_sheen`) and `emissive` (with
    /// `KHR_materials_emissive_strength`); see [`pack()`]. Constants are
    /// carried as glTF's factors (`specular_ior` only as one, through
    /// `KHR_materials_ior`), except occlusion (glTF has no occlusion
    /// factor), coat tint (no extension has one) and height (glTF has no
    /// displacement).
    Gltf,
    /// One texture per parameter map, normals in the input frame, and
    /// every constant as one; for inspection and tools rather than
    /// rendering. Data in `[0, 1]` is linear 8-bit; parameters without an
    /// upper bound (`specular_weight`, `specular_ior`,
    /// `emission_luminance`, `height`) are 16-bit over the map's own
    /// `[min, max]`, which their placement states.
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
    /// OpenPBR `specular_weight`, at least 0; values above 1 strengthen
    /// the dielectric reflection.
    pub specular_weight: Option<Input<1>>,
    /// OpenPBR `specular_color`: the tint of the specular reflection.
    pub specular_color: Option<Input<3>>,
    /// OpenPBR `specular_ior`, the dielectric index of refraction.
    pub specular_ior: Option<Input<1>>,
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
    /// OpenPBR `geometry_coat_normal`: the coat's unit shading normals, in
    /// the frame of [`Self::normal`]. A map only, as for the base normal.
    pub coat_normal: Option<Image>,
    /// OpenPBR `fuzz_weight`: how much of the surface the fuzz (sheen)
    /// layer covers.
    pub fuzz_weight: Option<Input<1>>,
    /// OpenPBR `fuzz_color`.
    pub fuzz_color: Option<Input<3>>,
    /// OpenPBR `fuzz_roughness`.
    pub fuzz_roughness: Option<Input<1>>,
    /// OpenPBR `emission_luminance`, in nits (cd/m²), at least 0.
    pub emission_luminance: Option<Input<1>>,
    /// OpenPBR `emission_color`.
    pub emission_color: Option<Input<3>>,
    /// Displacement above the surface's datum in the material's length
    /// unit (meters for `dapple_material`); an auxiliary output, not an
    /// OpenPBR parameter.
    pub height: Option<Input<1>>,
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
    pub(super) const SPECULAR_WEIGHT: [f32; 1] = [1.0];
    pub(super) const SPECULAR_COLOR: [f32; 3] = [1.0; 3];
    pub(super) const SPECULAR_IOR: [f32; 1] = [1.5];
    pub(super) const SUBSURFACE_WEIGHT: [f32; 1] = [0.0];
    pub(super) const SUBSURFACE_COLOR: [f32; 3] = [0.8; 3];
    pub(super) const TRANSMISSION_WEIGHT: [f32; 1] = [0.0];
    pub(super) const COAT_WEIGHT: [f32; 1] = [0.0];
    pub(super) const COAT_ROUGHNESS: [f32; 1] = [0.0];
    pub(super) const COAT_COLOR: [f32; 3] = [1.0; 3];
    pub(super) const FUZZ_WEIGHT: [f32; 1] = [0.0];
    pub(super) const FUZZ_COLOR: [f32; 3] = [1.0; 3];
    pub(super) const FUZZ_ROUGHNESS: [f32; 1] = [0.5];
    pub(super) const EMISSION_LUMINANCE: [f32; 1] = [0.0];
    pub(super) const EMISSION_COLOR: [f32; 3] = [1.0; 3];
    pub(super) const HEIGHT: [f32; 1] = [0.0];
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

fn normal_entry<'a>(name: &'static str, map: Option<&'a Image>) -> Entry<'a> {
    Entry {
        name,
        channels: 3,
        map,
        constant: None,
        default: &default::NORMAL,
        unit: false,
    }
}

impl MaterialMaps {
    /// Every field, in declaration order.
    fn entries(&self) -> [Entry<'_>; 24] {
        use default as d;
        [
            entry("base_color", &self.base_color, &d::BASE_COLOR, true),
            entry("opacity", &self.opacity, &d::OPACITY, true),
            normal_entry("normal", self.normal.as_ref()),
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
                "specular_weight",
                &self.specular_weight,
                &d::SPECULAR_WEIGHT,
                false,
            ),
            entry(
                "specular_color",
                &self.specular_color,
                &d::SPECULAR_COLOR,
                true,
            ),
            entry("specular_ior", &self.specular_ior, &d::SPECULAR_IOR, false),
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
            normal_entry("coat_normal", self.coat_normal.as_ref()),
            entry("fuzz_weight", &self.fuzz_weight, &d::FUZZ_WEIGHT, true),
            entry("fuzz_color", &self.fuzz_color, &d::FUZZ_COLOR, true),
            entry(
                "fuzz_roughness",
                &self.fuzz_roughness,
                &d::FUZZ_ROUGHNESS,
                true,
            ),
            entry(
                "emission_luminance",
                &self.emission_luminance,
                &d::EMISSION_LUMINANCE,
                false,
            ),
            entry(
                "emission_color",
                &self.emission_color,
                &d::EMISSION_COLOR,
                true,
            ),
            entry("height", &self.height, &d::HEIGHT, false),
        ]
    }
}

/// Whether `profile` can state parameter `name` as a constant, without a
/// texture.
fn states_constant(profile: Profile, name: &str) -> bool {
    match profile {
        Profile::Raw => true,
        Profile::Lightweald => !matches!(name, "occlusion" | "height"),
        Profile::Gltf => !matches!(name, "occlusion" | "coat_color" | "height"),
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
    /// Store glTF and raw normal maps, base and coat, as 16-bit RGB
    /// ([`PixelFormat::Rgb16Unorm`]) instead of 8-bit RGBA, for close
    /// views where 8-bit normals band. Lightweald's two-channel normal stays
    /// 8-bit, as its pool formats are.
    pub sixteen_bit_normals: bool,
}

impl Default for PackSettings {
    /// Box filtering, no coverage preservation, normal variance folded
    /// into roughness, and 8-bit normals.
    fn default() -> Self {
        Self {
            filter: Filter::Box,
            alpha_cutoff: None,
            fold_normal_variance: true,
            sixteen_bit_normals: false,
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
    /// Largest coat normal variance moved into coat roughness, per level.
    pub coat_normal_variance: Vec<f32>,
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
/// format's transfer function (sRGB channels are decoded first).
///
/// When several placements cover the same channels, the channels hold the
/// product of those parameters (glTF's sheen color is `fuzz_color` times
/// `fuzz_weight`); such ranges all start at 0, and the product is `t` times
/// the product of their `range[1]`.
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
    /// parameters, `[-1, 1]` for unit vectors, and the map's own extent for
    /// a parameter without an upper bound.
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
    /// `anisotropy`, `specular`, then translucency: `diffuse_transmission`
    /// and `transmission`, then the coat: `clearcoat` and
    /// `clearcoat_normal`, then `sheen` and `emissive` (raw: one per
    /// parameter map, in [`MaterialMaps`] field order).
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
/// `clearcoatRoughnessTexture` (G) read from one image, and the coat normal
/// as `clearcoat_normal`, stored like `normal`, for
/// `clearcoatNormalTexture`. The coat normal's variance widens coat
/// roughness as the base normal's widens specular roughness. The extension
/// has no tint, so `coat_color` is reported unsupported; Lightweald has no
/// coat slot and reports coat maps. Coat roughness mips are filtered in
/// `α²`, like specular roughness.
///
/// glTF's other extensions:
///
/// - `specular`: sRGB RGB `specular_color` and linear A `specular_weight`,
///   which `KHR_materials_specular`'s `specularColorTexture` and
///   `specularTexture` read. A weight above 1 is stored as a fraction of
///   its largest value, which the placement's range states (glTF's
///   `specularFactor`).
/// - `sheen`: sRGB RGB `fuzz_color` times `fuzz_weight` (the extension has
///   no weight) and linear A `fuzz_roughness`, which
///   `KHR_materials_sheen`'s `sheenColorTexture` and
///   `sheenRoughnessTexture` read.
/// - `emissive`: sRGB RGB `emission_color` times `emission_luminance`
///   divided by the largest luminance, whose value the luminance's
///   placement range states; converting nits to a glTF emissive strength is
///   the consumer's choice.
/// - `specular_ior` has no texture in `KHR_materials_ior`: a constant is a
///   factor, a map is reported unsupported.
///
/// Lightweald has no slot for these yet and reports their maps.
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
        let texture = normal_texture("normal", normals, profile, settings);
        let channels = 0..texture.format.channels().min(3);
        place(&mut placements, "normal", &[("normal", channels, SIGNED)]);
        textures.push(texture);
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
    let specular_weight = map(&maps.specular_weight);
    let specular_color = map(&maps.specular_color);
    match profile {
        Profile::Lightweald => {}
        Profile::Gltf if specular_weight.is_some() || specular_color.is_some() => {
            // KHR_materials_specular reads the weight from A as a fraction
            // of `specularFactor`, so weights above 1 are scaled into range.
            let weight_fill = fill(&maps.specular_weight, d::SPECULAR_WEIGHT)[0];
            let scale = specular_weight.map_or(weight_fill, max_value).max(1.0);
            let mut image = rgba(
                first,
                specular_color,
                fill(&maps.specular_color, d::SPECULAR_COLOR),
                specular_weight,
                weight_fill,
            );
            for t in image.values.as_chunks_mut::<4>().0 {
                t[3] /= scale;
            }
            textures.push(srgb_rgb_linear_a(
                "specular",
                &data_mips(&image, settings.filter),
            ));
            if maps.specular_color.is_some() {
                place(
                    &mut placements,
                    "specular",
                    &[("specular_color", 0..3, UNIT)],
                );
            }
            if maps.specular_weight.is_some() {
                place(
                    &mut placements,
                    "specular",
                    &[("specular_weight", 3..4, [0.0, scale])],
                );
            }
        }
        Profile::Gltf => {}
        Profile::Raw => {
            if let Some(weight) = specular_weight {
                raw_ranged(
                    &mut textures,
                    &mut placements,
                    "specular_weight",
                    weight,
                    settings.filter,
                );
            }
            if let Some(color) = specular_color {
                raw_map(
                    &mut textures,
                    &mut placements,
                    "specular_color",
                    color,
                    settings.filter,
                );
            }
            if let Some(ior) = map(&maps.specular_ior) {
                raw_ranged(
                    &mut textures,
                    &mut placements,
                    "specular_ior",
                    ior,
                    settings.filter,
                );
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
    let coat_rough_map = map(&maps.coat_roughness);
    let coat_rough_fill = fill(&maps.coat_roughness, d::COAT_ROUGHNESS)[0];
    let (coat_normals, coat_roughness) = match &maps.coat_normal {
        Some(n) => {
            let chain = normal_mips(n, coat_rough_map, coat_rough_fill, settings.filter)?;
            report.coat_normal_variance = chain.max_variance.clone();
            let roughness = if settings.fold_normal_variance {
                Some(chain.roughness)
            } else {
                coat_rough_map.map(|r| roughness_mips(r, settings.filter))
            };
            (Some(chain.normals), roughness)
        }
        None => (
            None,
            coat_rough_map.map(|r| roughness_mips(r, settings.filter)),
        ),
    };
    match profile {
        Profile::Lightweald => {}
        Profile::Gltf => {
            if coat_weight.is_some() || coat_roughness.is_some() {
                let weight_fill = fill(&maps.coat_weight, d::COAT_WEIGHT)[0];
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
                            coat_rough_fill,
                        )));
                        out.push(0);
                        out.push(255);
                    },
                    size,
                ));
                for (name, placed, channel) in [
                    ("coat_weight", maps.coat_weight.is_some(), 0),
                    (
                        "coat_roughness",
                        coat_roughness.is_some() || maps.coat_roughness.is_some(),
                        1,
                    ),
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
            if let Some(normals) = &coat_normals {
                textures.push(normal_texture(
                    "clearcoat_normal",
                    normals,
                    profile,
                    settings,
                ));
                place(
                    &mut placements,
                    "clearcoat_normal",
                    &[("coat_normal", 0..3, SIGNED)],
                );
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
            if let Some(normals) = &coat_normals {
                textures.push(normal_texture("coat_normal", normals, profile, settings));
                place(
                    &mut placements,
                    "coat_normal",
                    &[("coat_normal", 0..3, SIGNED)],
                );
            }
        }
    }

    let fuzz_weight = map(&maps.fuzz_weight);
    let fuzz_color = map(&maps.fuzz_color);
    let fuzz_roughness = map(&maps.fuzz_roughness).map(|r| roughness_mips(r, settings.filter));
    match profile {
        Profile::Lightweald => {}
        Profile::Gltf
            if fuzz_weight.is_some() || fuzz_color.is_some() || fuzz_roughness.is_some() =>
        {
            // KHR_materials_sheen has no weight: its color is the fuzz color
            // times the fuzz weight.
            let mut tint = rgba(
                first,
                fuzz_color,
                fill(&maps.fuzz_color, d::FUZZ_COLOR),
                fuzz_weight,
                fill(&maps.fuzz_weight, d::FUZZ_WEIGHT)[0],
            );
            let roughness_fill = fill(&maps.fuzz_roughness, d::FUZZ_ROUGHNESS)[0];
            for t in tint.values.as_chunks_mut::<4>().0 {
                let w = t[3];
                for c in &mut t[..3] {
                    *c *= w;
                }
                t[3] = roughness_fill;
            }
            let mut chain = data_mips(&tint, settings.filter);
            if let Some(roughness) = &fuzz_roughness {
                chain = with_alpha(&chain, roughness);
            }
            textures.push(srgb_rgb_linear_a("sheen", &chain));
            for (name, given, channels) in [
                ("fuzz_color", maps.fuzz_color.is_some(), 0..3),
                ("fuzz_weight", maps.fuzz_weight.is_some(), 0..3),
                ("fuzz_roughness", maps.fuzz_roughness.is_some(), 3..4),
            ] {
                if given {
                    place(&mut placements, "sheen", &[(name, channels, UNIT)]);
                }
            }
        }
        Profile::Gltf => {}
        Profile::Raw => {
            if let Some(weight) = fuzz_weight {
                raw_map(
                    &mut textures,
                    &mut placements,
                    "fuzz_weight",
                    weight,
                    settings.filter,
                );
            }
            if let Some(color) = fuzz_color {
                raw_map(
                    &mut textures,
                    &mut placements,
                    "fuzz_color",
                    color,
                    settings.filter,
                );
            }
            if fuzz_roughness.is_some() {
                textures.push(encode(
                    "fuzz_roughness",
                    PixelFormat::R8Unorm,
                    level_count,
                    edge,
                    |level, i, out| {
                        out.push(quantize_unorm8(value(&fuzz_roughness, level, i, 0.0)));
                    },
                    size,
                ));
                place(
                    &mut placements,
                    "fuzz_roughness",
                    &[("fuzz_roughness", 0..1, UNIT)],
                );
            }
        }
    }

    let luminance = map(&maps.emission_luminance);
    let emission_color = map(&maps.emission_color);
    match profile {
        Profile::Lightweald => {}
        Profile::Gltf if luminance.is_some() || emission_color.is_some() => {
            // glTF's emissive texture is a color: the luminance, divided by
            // its largest value, scales it, and that largest value is the
            // luminance's range (glTF's emissive strength).
            let luminance_fill = fill(&maps.emission_luminance, d::EMISSION_LUMINANCE)[0];
            let scale = luminance.map_or(luminance_fill, max_value).max(0.0);
            let mut light = rgba(
                first,
                emission_color,
                fill(&maps.emission_color, d::EMISSION_COLOR),
                luminance,
                luminance_fill,
            );
            for t in light.values.as_chunks_mut::<4>().0 {
                let l = if scale > 0.0 { t[3] / scale } else { 0.0 };
                for c in &mut t[..3] {
                    *c *= l;
                }
                t[3] = 1.0;
            }
            textures.push(srgb_rgb_linear_a(
                "emissive",
                &data_mips(&light, settings.filter),
            ));
            if maps.emission_color.is_some() {
                place(
                    &mut placements,
                    "emissive",
                    &[("emission_color", 0..3, UNIT)],
                );
            }
            if maps.emission_luminance.is_some() {
                place(
                    &mut placements,
                    "emissive",
                    &[("emission_luminance", 0..3, [0.0, scale])],
                );
            }
        }
        Profile::Gltf => {}
        Profile::Raw => {
            if let Some(luminance) = luminance {
                raw_ranged(
                    &mut textures,
                    &mut placements,
                    "emission_luminance",
                    luminance,
                    settings.filter,
                );
            }
            if let Some(color) = emission_color {
                raw_map(
                    &mut textures,
                    &mut placements,
                    "emission_color",
                    color,
                    settings.filter,
                );
            }
        }
    }
    if let (Profile::Raw, Some(height)) = (profile, map(&maps.height)) {
        raw_ranged(
            &mut textures,
            &mut placements,
            "height",
            height,
            settings.filter,
        );
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

/// A raw profile's texture for a scalar map without an upper bound: 16-bit
/// over the map's own `[min, max]`, which its placement states.
fn raw_ranged(
    textures: &mut Vec<EncodedTexture>,
    placements: &mut Vec<Placement>,
    name: &'static str,
    image: &Image,
    filter: Filter,
) {
    let chain = data_mips(image, filter);
    let (lo, hi) = image
        .values
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let span = hi - lo;
    textures.push(encode(
        name,
        PixelFormat::R16Unorm,
        chain.levels().len(),
        image.edge,
        |level, i, out| {
            let v = chain.levels()[level].values[i];
            let t = if span > 0.0 { (v - lo) / span } else { 0.0 };
            out.extend_from_slice(&quantize_unorm16(t).to_le_bytes());
        },
        |level| {
            let l = &chain.levels()[level];
            (l.width, l.height)
        },
    ));
    place(placements, name, &[(name, 0..1, [lo, hi])]);
}

/// The largest value of a one-channel image.
fn max_value(image: &Image) -> f32 {
    image
        .values
        .iter()
        .fold(f32::NEG_INFINITY, |m, &v| m.max(v))
}

/// `chain` with each level's alpha replaced by `alpha`'s single channel.
fn with_alpha(chain: &MipChain, alpha: &MipChain) -> MipChain {
    MipChain::from_levels(
        chain
            .levels()
            .iter()
            .zip(alpha.levels())
            .map(|(rgba, a)| {
                let mut out = rgba.clone();
                for (t, &a) in out.values.as_chunks_mut::<4>().0.iter_mut().zip(&a.values) {
                    t[3] = a;
                }
                out
            })
            .collect(),
    )
}

/// An RGBA chain as an sRGB texture: RGB sRGB-encoded, alpha linear.
fn srgb_rgb_linear_a(name: &'static str, chain: &MipChain) -> EncodedTexture {
    let base = chain.base();
    encode(
        name,
        PixelFormat::Rgba8Srgb,
        chain.levels().len(),
        base.edge,
        |level, i, out| {
            let t = &chain.levels()[level].values[i * 4..i * 4 + 4];
            for &c in &t[..3] {
                out.push(quantize_unorm8(linear_to_srgb(c)));
            }
            out.push(quantize_unorm8(t[3]));
        },
        |level| {
            let l = &chain.levels()[level];
            (l.width, l.height)
        },
    )
}

/// A chain of unit normals as `profile` stores them: Lightweald keeps XY,
/// glTF and raw XYZ (with alpha 1 in 8 bits, or as 16-bit RGB); Lightweald
/// and glTF negate Y so `+Y` points to the image top, and are
/// [`TextureKind::Normal`].
fn normal_texture(
    name: &'static str,
    normals: &MipChain,
    profile: Profile,
    settings: &PackSettings,
) -> EncodedTexture {
    let unit = |v: f32| v * 0.5 + 0.5;
    let format = match profile {
        Profile::Lightweald => PixelFormat::Rg8Unorm,
        Profile::Gltf | Profile::Raw if settings.sixteen_bit_normals => PixelFormat::Rgb16Unorm,
        Profile::Gltf | Profile::Raw => PixelFormat::Rgba8Unorm,
    };
    let raw = profile == Profile::Raw;
    let mut texture = encode(
        name,
        format,
        normals.levels().len(),
        normals.base().edge,
        |level, i, out| {
            let n = &normals.levels()[level].values[i * 3..i * 3 + 3];
            let y = if raw { n[1] } else { -n[1] };
            match format {
                PixelFormat::Rgb16Unorm => {
                    for v in [n[0], y, n[2]] {
                        out.extend_from_slice(&quantize_unorm16(unit(v)).to_le_bytes());
                    }
                }
                PixelFormat::Rgba8Unorm => {
                    out.extend([unit(n[0]), unit(y), unit(n[2])].map(quantize_unorm8));
                    out.push(255);
                }
                _ => out.extend([unit(n[0]), unit(y)].map(quantize_unorm8)),
            }
        },
        |level| {
            let l = &normals.levels()[level];
            (l.width, l.height)
        },
    );
    if !raw {
        texture.kind = TextureKind::Normal;
    }
    texture
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
                    PixelFormat::R16Unorm | PixelFormat::Rgb16Unorm => {
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

    /// A 4×4 one-channel image alternating `a` and `b` along rows.
    fn alternating(a: f32, b: f32) -> Image {
        let values = (0..16).map(|i| if i % 2 == 0 { a } else { b }).collect();
        Image::new(4, 4, 1, Edge::Wrap, values).unwrap()
    }

    fn extensions() -> MaterialMaps {
        let tilted = [0.6, 0.0, 0.8];
        MaterialMaps {
            specular_weight: Some(flat(1, &[2.0]).into()),
            specular_color: Some([0.5, 0.9, 0.2].into()),
            specular_ior: Some(alternating(1.4, 1.6).into()),
            coat_weight: Some(1.0.into()),
            coat_normal: Some(flat(3, &tilted)),
            fuzz_weight: Some(flat(1, &[0.5]).into()),
            fuzz_color: Some([1.0, 0.5, 0.25].into()),
            fuzz_roughness: Some(0.7.into()),
            emission_luminance: Some(alternating(100.0, 50.0).into()),
            emission_color: Some([1.0, 0.5, 0.0].into()),
            height: Some(alternating(-0.01, 0.03).into()),
            ..MaterialMaps::default()
        }
    }

    #[test]
    fn gltf_packs_its_extension_layouts() {
        let gltf = pack(&extensions(), Profile::Gltf, &PackSettings::default()).unwrap();
        let names: Vec<_> = gltf.textures.iter().map(|t| t.name).collect();
        assert_eq!(
            names,
            [
                "specular",
                "clearcoat",
                "clearcoat_normal",
                "sheen",
                "emissive"
            ]
        );
        let srgb = |v: f32| quantize_unorm8(linear_to_srgb(v));

        // KHR_materials_specular: RGB color, A the weight over its largest
        // value, which the placement states.
        let specular = gltf.texture("specular").unwrap();
        assert_eq!(specular.format, PixelFormat::Rgba8Srgb);
        assert_eq!(
            specular.levels[0][..4],
            [srgb(0.5), srgb(0.9), srgb(0.2), 255]
        );
        assert_eq!(gltf.placement("specular_weight").unwrap().range, [0.0, 2.0]);

        // KHR_materials_clearcoat: the constant weight fills R, the coat
        // normal's variance widens G, and the coat normal has its own
        // texture with +Y toward the image top.
        let clearcoat = gltf.texture("clearcoat").unwrap();
        assert_eq!(clearcoat.levels[0][0], 255);
        assert_eq!(gltf.placement("coat_weight").unwrap().channels, 0..1);
        assert_eq!(gltf.placement("coat_roughness").unwrap().channels, 1..2);
        let coat_normal = gltf.texture("clearcoat_normal").unwrap();
        assert_eq!(coat_normal.kind, TextureKind::Normal);
        assert_eq!(
            coat_normal.levels[0][..4],
            [quantize_unorm8(0.8), 128, quantize_unorm8(0.9), 255]
        );
        assert_eq!(gltf.report.coat_normal_variance.len(), 3);

        // KHR_materials_sheen: color times weight in RGB, roughness in A.
        let sheen = gltf.texture("sheen").unwrap();
        assert_eq!(
            sheen.levels[0][..4],
            [srgb(0.5), srgb(0.25), srgb(0.125), quantize_unorm8(0.7)]
        );
        let on = |name| gltf.placement(name).unwrap().channels.clone();
        assert_eq!(on("fuzz_color"), on("fuzz_weight"), "a product");
        assert_eq!(on("fuzz_roughness"), 3..4);

        // Emission: color times luminance over the largest luminance.
        let emissive = gltf.texture("emissive").unwrap();
        assert_eq!(emissive.levels[0][..3], [255, srgb(0.5), 0]);
        assert_eq!(emissive.levels[0][4..7], [srgb(0.5), srgb(0.25), 0]);
        let luminance = gltf.placement("emission_luminance").unwrap();
        assert_eq!(
            (luminance.channels.clone(), luminance.range),
            (0..3, [0.0, 100.0])
        );

        // KHR_materials_ior has no texture, and glTF has no displacement.
        assert_eq!(gltf.report.unsupported, ["specular_ior", "height"]);
        assert!(gltf.constants.is_empty(), "every constant shares a texture");

        let constant_ior = MaterialMaps {
            specular_ior: Some(1.33.into()),
            ..MaterialMaps::default()
        };
        let gltf = pack(&constant_ior, Profile::Gltf, &PackSettings::default()).unwrap();
        assert_eq!(gltf.constant("specular_ior"), Some([1.33].as_slice()));
    }

    #[test]
    fn lightweald_and_raw_carry_or_report_the_extensions() {
        let settings = PackSettings::default();
        let lightweald = pack(&extensions(), Profile::Lightweald, &settings).unwrap();
        assert!(lightweald.textures.is_empty());
        assert_eq!(
            lightweald.report.unsupported,
            [
                "specular_weight",
                "specular_ior",
                "coat_normal",
                "fuzz_weight",
                "emission_luminance",
                "height"
            ]
        );
        let constants: Vec<_> = lightweald.constants.iter().map(|c| c.name).collect();
        assert_eq!(
            constants,
            [
                "specular_color",
                "coat_weight",
                "fuzz_color",
                "fuzz_roughness",
                "emission_color"
            ]
        );

        let raw = pack(&extensions(), Profile::Raw, &settings).unwrap();
        let names: Vec<_> = raw.textures.iter().map(|t| t.name).collect();
        assert_eq!(
            names,
            [
                "specular_weight",
                "specular_ior",
                "coat_roughness",
                "coat_normal",
                "fuzz_weight",
                "emission_luminance",
                "height"
            ]
        );
        assert!(raw.report.unsupported.is_empty());
        // Parameters without an upper bound are 16-bit over their own range.
        let ior = raw.texture("specular_ior").unwrap();
        assert_eq!(ior.format, PixelFormat::R16Unorm);
        assert_eq!(ior.levels[0][..4], [0, 0, 255, 255]);
        assert_eq!(raw.placement("specular_ior").unwrap().range, [1.4, 1.6]);
        assert_eq!(raw.placement("height").unwrap().range, [-0.01, 0.03]);
        assert_eq!(
            raw.texture("coat_normal").unwrap().kind,
            TextureKind::Data,
            "raw normals stay in the input frame"
        );
    }

    #[test]
    fn normals_can_be_sixteen_bit() {
        let tilted = [0.6, 0.0, 0.8];
        let maps = MaterialMaps {
            normal: Some(flat(3, &tilted)),
            coat_normal: Some(flat(3, &tilted)),
            ..MaterialMaps::default()
        };
        let settings = PackSettings {
            sixteen_bit_normals: true,
            ..PackSettings::default()
        };
        let gltf = pack(&maps, Profile::Gltf, &settings).unwrap();
        let unit = |v: f32| quantize_unorm16(v * 0.5 + 0.5).to_le_bytes();
        for name in ["normal", "clearcoat_normal"] {
            let texture = gltf.texture(name).unwrap();
            assert_eq!(texture.format, PixelFormat::Rgb16Unorm);
            assert_eq!(texture.kind, TextureKind::Normal);
            assert_eq!(texture.levels[0].len(), 16 * 6);
            assert_eq!(
                texture.levels[0][..6],
                [unit(0.6), unit(-0.0), unit(0.8)].concat()
            );
        }
        assert_eq!(gltf.placement("coat_normal").unwrap().channels, 0..3);
        let raw = pack(&maps, Profile::Raw, &settings).unwrap();
        assert_eq!(
            raw.texture("coat_normal").unwrap().format,
            PixelFormat::Rgb16Unorm
        );
        let lightweald = pack(&maps, Profile::Lightweald, &settings).unwrap();
        assert_eq!(
            lightweald.texture("normal").unwrap().format,
            PixelFormat::Rg8Unorm
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn png_writes_sixteen_bit_rgb() {
        let maps = MaterialMaps {
            normal: Some(flat(3, &[0.6, 0.0, 0.8])),
            ..MaterialMaps::default()
        };
        let settings = PackSettings {
            sixteen_bit_normals: true,
            ..PackSettings::default()
        };
        let bundle = pack(&maps, Profile::Gltf, &settings).unwrap();
        let normal = bundle.texture("normal").unwrap();
        let bytes = crate::png::write(normal).unwrap();
        let mut reader = png::Decoder::new(std::io::Cursor::new(bytes))
            .read_info()
            .unwrap();
        let info = reader.info();
        assert_eq!(info.color_type, png::ColorType::Rgb);
        assert_eq!(info.bit_depth, png::BitDepth::Sixteen);
        let mut samples = vec![0; reader.output_buffer_size().unwrap()];
        reader.next_frame(&mut samples).unwrap();
        // PNG samples are big-endian; the texture's levels little-endian.
        let le: Vec<u8> = samples.chunks(2).flat_map(|b| [b[1], b[0]]).collect();
        assert_eq!(le, normal.levels[0]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn png_files_carry_no_color_space_chunk() {
        let maps = MaterialMaps {
            base_color: Some(flat(3, &[0.5, 0.25, 0.0]).into()),
            ..MaterialMaps::default()
        };
        let bundle = pack(&maps, Profile::Gltf, &PackSettings::default()).unwrap();
        let color = bundle.texture("base_color").unwrap();
        assert!(color.format.is_srgb());
        let bytes = crate::png::write(color).unwrap();
        // glTF 2.0, 3.8.3.2: color-space chunks should not be present.
        for chunk in [b"sRGB", b"gAMA", b"iCCP", b"cHRM"] {
            assert!(
                !bytes.windows(4).any(|w| w == chunk),
                "{} chunk written",
                core::str::from_utf8(chunk).unwrap()
            );
        }
        let mut reader = png::Decoder::new(std::io::Cursor::new(bytes))
            .read_info()
            .unwrap();
        let mut samples = vec![0; reader.output_buffer_size().unwrap()];
        reader.next_frame(&mut samples).unwrap();
        assert_eq!(samples, color.levels[0], "the samples are the texels");
    }
}
