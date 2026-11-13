// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The dapple manifest: a versioned JSON description of an export.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;
use core::ops::Range;

use serde::{Deserialize, Serialize};

/// The manifest format this crate writes and reads. Any change to the
/// schema, however small, is a new version: readers refuse versions they
/// do not know, and unknown fields, rather than guess.
pub const VERSION: u32 = 1;

/// What the `format` field of every dapple manifest holds.
pub const FORMAT: &str = "dapple-material";

/// The file name [`crate::Exported::write_dir`] gives the manifest.
pub const MANIFEST_FILE: &str = "manifest.json";

/// The file name [`crate::Exported::write_dir`] gives the glTF document.
pub const GLTF_FILE: &str = "material.gltf";

/// A dapple manifest: what an export holds, where, and how to read it.
///
/// Written by [`crate::Exported::write_dir`] as [`MANIFEST_FILE`] with
/// [`Self::to_json`], whose output is deterministic: fields in declaration
/// order, lists in the order documented on each, and numbers in Rust's
/// shortest round-tripping form, so equal exports give equal bytes.
/// [`Self::from_json`] reads it back.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Always [`FORMAT`].
    pub format: String,
    /// The schema version, [`VERSION`].
    pub version: u32,
    /// How the textures were packed.
    pub packing: Packing,
    /// Where the material came from.
    pub provenance: Provenance,
    /// Units and frames every value and texture follows.
    pub conventions: Conventions,
    /// One entry per OpenPBR parameter, in the specification's order, then
    /// `occlusion` and `height`.
    pub parameters: Vec<Parameter>,
    /// The textures, the profile's first, in packing order.
    pub textures: Vec<TextureEntry>,
    /// The glTF document, when the profile is glTF.
    pub gltf: Option<String>,
    /// Every file written but the manifest itself, in writing order.
    pub files: Vec<FileEntry>,
    /// The material laboratory's verdict, when one was given.
    pub lab: Option<LabSummary>,
}

/// Why a manifest could not be read.
#[derive(Debug)]
pub enum ManifestError {
    /// The text is not JSON, or not a manifest of this schema.
    Syntax(serde_json::Error),
    /// The `format` field is not [`FORMAT`].
    Format(String),
    /// The manifest has a version this crate does not read.
    Version(u32),
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax(error) => write!(f, "not a dapple manifest: {error}"),
            Self::Format(format) => write!(f, "format {format:?} is not {FORMAT:?}"),
            Self::Version(version) => {
                write!(f, "manifest version {version}; this reader knows {VERSION}")
            }
        }
    }
}

impl core::error::Error for ManifestError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Syntax(error) => Some(error),
            Self::Format(_) | Self::Version(_) => None,
        }
    }
}

/// The two header fields, read before the rest so a newer version reports
/// its version rather than a schema error.
#[derive(Deserialize)]
struct Header {
    format: String,
    version: u32,
}

impl Manifest {
    /// The manifest as pretty-printed JSON, ending in a newline.
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut text = serde_json::to_string_pretty(self).expect("manifests always serialize");
        text.push('\n');
        text
    }

    /// Reads a manifest written by [`Self::to_json`].
    ///
    /// # Errors
    ///
    /// [`ManifestError::Format`] or [`ManifestError::Version`] for another
    /// kind or version of file, and [`ManifestError::Syntax`] for text that
    /// does not follow the schema.
    pub fn from_json(text: &str) -> Result<Self, ManifestError> {
        let header: Header = serde_json::from_str(text).map_err(ManifestError::Syntax)?;
        if header.format != FORMAT {
            return Err(ManifestError::Format(header.format));
        }
        if header.version != VERSION {
            return Err(ManifestError::Version(header.version));
        }
        serde_json::from_str(text).map_err(ManifestError::Syntax)
    }

    /// The parameter named `name`.
    #[must_use]
    pub fn parameter(&self, name: &str) -> Option<&Parameter> {
        self.parameters.iter().find(|p| p.name == name)
    }

    /// The texture named `name`.
    #[must_use]
    pub fn texture(&self, name: &str) -> Option<&TextureEntry> {
        self.textures.iter().find(|t| t.name == name)
    }
}

/// How an export's textures were packed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Packing {
    /// The packing profile: `"gltf"`, `"lightweald"` or `"raw"`.
    pub profile: String,
    /// The mip filter: `"box"` or `"kaiser"`.
    pub filter: String,
    /// The alpha-test cutoff opacity mips preserve coverage at.
    pub alpha_cutoff: Option<f32>,
    /// Whether normal variance widens roughness mips.
    pub fold_normal_variance: bool,
    /// Whether normal maps are 16-bit.
    pub sixteen_bit_normals: bool,
}

/// Where a material came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    /// What the export is called, such as an asset or preset name.
    pub subject: String,
    /// The module instance that built the material, when diagnostics were
    /// given.
    pub instance: Option<Instance>,
    /// The texel grid.
    pub grid: GridInfo,
    /// The axes the material promises to tile along; samplers repeat along
    /// these and clamp along the others.
    pub tiling: TilingInfo,
}

/// A module instance, from `dapple_material`'s diagnostics.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Instance {
    /// The instance path, such as `"wall"`.
    pub path: String,
    /// The module's name, such as `"dapple_library.glazed_brick_wall"`.
    pub module: String,
    /// The module's version.
    pub version: u32,
    /// The arguments' fingerprint (parameters, inputs, resources and the
    /// seed the path gives), as 16 hexadecimal digits: equal fingerprints
    /// build equal materials with the same module version.
    pub fingerprint: String,
}

/// A material's texel grid.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GridInfo {
    /// Texels per row.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// Domain position of texel `(0, 0)`'s minimum corner, in meters.
    pub origin: [f32; 2],
    /// Texel size in meters.
    pub texel: [f32; 2],
    /// The grid's physical extent, `width × texel[0]` by
    /// `height × texel[1]`, in meters.
    pub size: [f32; 2],
    /// The edge policy: `"wrap"` (one period of a repeating domain) or
    /// `"clamp"`.
    pub edge: String,
}

/// The axes a material tiles along.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TilingInfo {
    /// Across columns (the texture's u).
    pub x: bool,
    /// Across rows (the texture's v).
    pub y: bool,
}

/// The units and frames an export follows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Conventions {
    /// The unit of lengths: positions, texel sizes and height (`"m"`).
    pub length_unit: String,
    /// The unit of `emission_luminance` (`"cd/m2"`), which is also glTF's
    /// emissive unit.
    pub luminance_unit: String,
    /// The color space of color values once decoded (`"lin_rec709"`:
    /// linear Rec. 709 primaries, D65).
    pub color_space: String,
    /// Where texture rows start (`"row 0 is the grid's minimum y"`): a
    /// texture's v grows with domain y, as glTF's v grows down the image.
    pub rows: String,
}

impl Conventions {
    /// The conventions of every dapple export.
    #[must_use]
    pub fn dapple() -> Self {
        Self {
            length_unit: "m".to_string(),
            luminance_unit: "cd/m2".to_string(),
            color_space: "lin_rec709".to_string(),
            rows: "row 0 is the grid's minimum y".to_string(),
        }
    }
}

/// One parameter's account: where an export holds it, and whether the
/// profile's own material (the glTF material, for glTF) does.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parameter {
    /// The OpenPBR identifier, such as `"coat_weight"`, or `"occlusion"` or
    /// `"height"` for the auxiliary channels.
    pub name: String,
    /// Where its values are.
    pub held: Held,
    /// Whether the profile's material carries it: the textures and factors
    /// a glTF reader sees, for glTF. False when only the export's extra
    /// textures or the manifest hold it, or when the profile fixes a value
    /// that differs from the material's.
    pub in_profile: bool,
    /// What was approximated or lost, and why.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

/// Where an export holds a parameter's values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Held {
    /// Not set (or set to the specification's default): the OpenPBR
    /// specification's default, stated so readers need no table of them.
    Default {
        /// The default, one component per channel of the parameter.
        value: Vec<f32>,
    },
    /// One value everywhere, one component per channel of the parameter
    /// (colors linear Rec. 709).
    Constant {
        /// The value.
        value: Vec<f32>,
    },
    /// In texture channels.
    Texture(Location),
    /// Its map could not be carried at all.
    Lost {
        /// The map's mean, one component per channel of the parameter:
        /// a stand-in for consumers that must use some value.
        mean: Vec<f32>,
    },
}

/// Which channels of which texture hold a parameter, and how to decode
/// them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Location {
    /// The texture's name ([`TextureEntry::name`]).
    pub texture: String,
    /// The channels, `start..end`, as many as the parameter has (or the
    /// encoding stores).
    pub channels: Range<usize>,
    /// The values a channel's 0 and full scale stand for, after the
    /// texture's transfer function is undone.
    pub range: [f32; 2],
    /// What the decoded channels mean.
    pub encoding: Encoding,
}

/// What a [`Location`]'s channels mean once decoded into `range`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Encoding {
    /// The parameter's value.
    Value,
    /// The product of this parameter and every other parameter placed on
    /// the same channels (glTF's sheen color is `fuzz_color` times
    /// `fuzz_weight`).
    Product,
    /// A tangent-space unit normal, `+X` along u and `+Y` toward the image
    /// top (glTF's and OpenGL's convention).
    NormalYUp,
    /// A unit normal in the domain frame: `+X` along u, `+Y` along
    /// increasing rows (down the image).
    NormalYDown,
    /// The anisotropy axis `(cos θ, sin θ)` in tangent space, `+Y` toward
    /// the image top: glTF's anisotropy direction. Carries
    /// `geometry_tangent` as the direction it projects onto the surface.
    AxisYUp,
    /// The anisotropy axis `(cos θ, sin θ)` in the domain frame, `+Y` down
    /// the image.
    AxisYDown,
    /// glTF's `anisotropyStrength` for the texel's roughness, which
    /// `dapple_encode::gltf_anisotropy_strength` converts
    /// `specular_roughness_anisotropy` to.
    GltfAnisotropyStrength,
}

/// One texture of an export.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextureEntry {
    /// The texture's name, which [`Location::texture`] refers to.
    pub name: String,
    /// Its file, relative to the manifest.
    pub file: String,
    /// Whether the profile's material uses it (false for textures that
    /// hold what the profile cannot carry).
    pub in_profile: bool,
    /// The transfer function of its first three channels: `"srgb"` or
    /// `"linear"`. A fourth channel is always linear.
    pub color_space: String,
    /// Channels per texel in the file.
    pub channels: u32,
    /// Bits per channel in the file: 8 or 16.
    pub bits: u32,
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
}

/// One written file and its content hash.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileEntry {
    /// The path, relative to the manifest, with `/` separators.
    pub path: String,
    /// Its size in bytes.
    pub bytes: u64,
    /// Its [`crate::content_hash`], as 32 hexadecimal digits.
    pub hash: String,
}

/// A material laboratory report's verdict.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabSummary {
    /// The report's subject.
    pub subject: String,
    /// Whether every measurement and check passed.
    pub passed: bool,
    /// The names of the entries that failed, in report order.
    pub failures: Vec<String>,
}

impl From<&dapple_lab::Report> for LabSummary {
    fn from(report: &dapple_lab::Report) -> Self {
        Self {
            subject: report.subject.clone(),
            passed: report.passed(),
            failures: report
                .failures()
                .map(|e| match e {
                    dapple_lab::Entry::Measurement(m) => m.name.clone(),
                    dapple_lab::Entry::Check(c) => c.name.clone(),
                })
                .collect(),
        }
    }
}
