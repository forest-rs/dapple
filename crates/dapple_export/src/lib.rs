// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Exports dapple materials as self-describing results.
//!
//! [`export`] turns a [`Material`] into an [`Exported`] value: the packed
//! textures, and one account ([`Parameter`]) per OpenPBR parameter and
//! auxiliary channel saying whether it is the specification's default, a
//! constant, which channels of which texture hold it and how to decode
//! them, or lost, with why. It merges what `dapple_material`'s lowering
//! dropped and what `dapple_encode`'s packing could not carry or only
//! approximates into that one account, so a consumer reads one thing. What
//! the profile cannot carry comes from the raw profile instead, as extra
//! textures, or as constants in the account.
//!
//! The value is everything a writer needs; nothing exists only in file
//! form:
//!
//! - [`gltf::material`] builds the glTF material (textures, images,
//!   samplers and extensions) and [`gltf::document`] a complete glTF file:
//!   one quad the material's physical size, so the bake opens by itself in
//!   any glTF viewer.
//! - [`Exported::manifest`] describes it as a versioned [`Manifest`]:
//!   files and their content hashes, the account, conventions, provenance
//!   and the material laboratory's verdict. [`Manifest::from_json`] reads
//!   one back.
//! - With the `std` feature, [`Exported::write_dir`] writes the textures as
//!   PNG, the glTF document and the manifest into a directory.
//!
//! ```
//! use dapple_export::{Held, Settings, Source, export};
//! use dapple_field::Value;
//! use dapple_material::{Channel, Material, Param};
//!
//! // A 1 m tile of 4 × 4 texels.
//! let mut material = Material::new(dapple_lab::material::unit_tile(4));
//! material.set_param(Param::CoatWeight, Channel::Constant(Value::Scalar(1.0)))?;
//! let exported = export(&Source::new("varnish", &material), &Settings::default())?;
//! let coat = exported.parameter("coat_weight").expect("every parameter is accounted for");
//! assert_eq!(coat.held, Held::Constant { value: vec![1.0] });
//! // glTF's clearcoat fixes its IOR at 1.5; OpenPBR's default is 1.6.
//! assert!(!exported.parameter("coat_ior").unwrap().in_profile);
//! let gltf = dapple_export::gltf::document(&exported).expect("a glTF export");
//! assert!(gltf.contains("KHR_materials_clearcoat"));
//! # Ok::<(), Box<dyn core::error::Error>>(())
//! ```
//!
//! **Choices.** The value is built from `dapple_encode`'s bundles rather
//! than wrapping them: the account replaces their placements, constants and
//! loss lists, and the textures move out of them, so a consumer reads one
//! flat value instead of two bundles and two reports. JSON goes through
//! `serde_json` (already the workspace's choice for `dapple_package`, and
//! `no_std` with `alloc`): one schema definition serves the writer and the
//! reader, so the manifest round-trips by construction instead of through a
//! hand-written parser. Objects are written in a fixed order (declaration
//! order for the manifest, sorted keys for glTF) and numbers in their
//! shortest round-tripping form, so output is deterministic. Content
//! hashes reuse `dapple_field`'s keyed hash; they detect changes, not
//! tampering.
//!
//! Files are PNG: glTF core reads no other lossless format, and KTX2 needs
//! `KHR_texture_basisu`'s supercompression, which `dapple_encode` does not
//! write. PNG keeps only the first mip level; `dapple_encode::ktx2` writes
//! whole chains for engines that want them.

#![no_std]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod account;
pub mod gltf;
mod hash;
mod manifest;
#[cfg(feature = "std")]
mod write;

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use dapple_encode::{
    EncodeError, EncodedTexture, Filter, MaterialMaps, PackReport, PackSettings, PixelFormat,
    Profile, pack,
};
use dapple_material::module::Diagnostics;
use dapple_material::{Material, MaterialError, lower};

pub use hash::{ContentHash, content_hash};
pub use manifest::{
    Conventions, Encoding, FORMAT, FileEntry, GLTF_FILE, GridInfo, Held, Instance, LabSummary,
    Location, MANIFEST_FILE, Manifest, ManifestError, Packing, Parameter, Provenance, TextureEntry,
    TilingInfo, VERSION,
};
#[cfg(feature = "std")]
pub use write::read_dir;

/// What to export: a material, what it is called, and what is known about
/// how it was made.
#[derive(Copy, Clone, Debug)]
pub struct Source<'a> {
    /// What the export is called, such as an asset or preset name.
    pub subject: &'a str,
    /// The material.
    pub material: &'a Material,
    /// The diagnostics of the context that built it, for its module
    /// instance and argument fingerprint.
    pub diagnostics: Option<&'a Diagnostics>,
    /// The material laboratory's report on it, whose verdict the manifest
    /// records.
    pub lab: Option<&'a dapple_lab::Report>,
}

impl<'a> Source<'a> {
    /// A material and its name, with no diagnostics or report.
    #[must_use]
    pub const fn new(subject: &'a str, material: &'a Material) -> Self {
        Self {
            subject,
            material,
            diagnostics: None,
            lab: None,
        }
    }
}

/// How to export.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Settings {
    /// The profile whose textures and material the export is for; what it
    /// cannot carry comes from [`Profile::Raw`] instead. [`gltf`] writes
    /// only glTF exports.
    pub profile: Profile,
    /// How to pack.
    pub pack: PackSettings,
}

impl Default for Settings {
    /// glTF, with `dapple_encode`'s default packing.
    fn default() -> Self {
        Self {
            profile: Profile::Gltf,
            pack: PackSettings::default(),
        }
    }
}

/// One texture of an export.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Texture {
    /// The texture, with its whole mip chain.
    pub encoded: EncodedTexture,
    /// Whether the profile's material uses it; false for the raw profile's
    /// textures of what the profile cannot carry.
    pub in_profile: bool,
}

impl Texture {
    /// The texture's file name: its name and `.png`.
    #[must_use]
    pub fn file(&self) -> String {
        alloc::format!("{}.png", self.encoded.name)
    }
}

/// An exported material: textures, the account of every parameter, and
/// where it came from. See the [crate docs](crate).
#[derive(Clone, Debug, PartialEq)]
pub struct Exported {
    /// How it was exported.
    pub settings: Settings,
    /// Where it came from.
    pub provenance: Provenance,
    /// One account per OpenPBR parameter, in the specification's order,
    /// then `occlusion` and `height`.
    pub parameters: Vec<Parameter>,
    /// The profile's textures in packing order, then the raw profile's for
    /// what the profile cannot carry. Names are unique.
    pub textures: Vec<Texture>,
    /// The profile's packing measurements (coverage and normal variance
    /// per level); its losses are in [`Self::parameters`].
    pub report: PackReport,
    /// The material laboratory's verdict, when a report was given.
    pub lab: Option<LabSummary>,
}

/// Why an export failed.
#[derive(Debug)]
pub enum ExportError {
    /// The material could not be lowered.
    Material(MaterialError),
    /// Its maps could not be packed.
    Encode(EncodeError),
    /// A file could not be written or read.
    #[cfg(feature = "std")]
    Io {
        /// The file.
        path: std::path::PathBuf,
        /// The failure.
        source: std::io::Error,
    },
    /// A texture could not be encoded as PNG.
    #[cfg(feature = "std")]
    Png {
        /// The texture.
        texture: &'static str,
        /// The failure.
        source: dapple_encode::png::PngError,
    },
    /// A manifest could not be read.
    Manifest(ManifestError),
}

impl fmt::Display for ExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Material(_) => f.write_str("the material could not be lowered"),
            Self::Encode(_) => f.write_str("the material's maps could not be packed"),
            #[cfg(feature = "std")]
            Self::Io { path, .. } => write!(f, "{}", path.display()),
            #[cfg(feature = "std")]
            Self::Png { texture, .. } => write!(f, "texture {texture} could not be encoded"),
            Self::Manifest(_) => f.write_str("the manifest could not be read"),
        }
    }
}

impl core::error::Error for ExportError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Material(error) => Some(error),
            Self::Encode(error) => Some(error),
            #[cfg(feature = "std")]
            Self::Io { source, .. } => Some(source),
            #[cfg(feature = "std")]
            Self::Png { source, .. } => Some(source),
            Self::Manifest(error) => Some(error),
        }
    }
}

impl From<MaterialError> for ExportError {
    fn from(error: MaterialError) -> Self {
        Self::Material(error)
    }
}

impl From<EncodeError> for ExportError {
    fn from(error: EncodeError) -> Self {
        Self::Encode(error)
    }
}

impl From<ManifestError> for ExportError {
    fn from(error: ManifestError) -> Self {
        Self::Manifest(error)
    }
}

/// Lowers, packs and accounts for `source`'s material.
///
/// The material is packed for `settings.profile`; the parameters that
/// profile reports it cannot carry are packed for [`Profile::Raw`] too, and
/// their textures follow the profile's in [`Exported::textures`].
///
/// # Errors
///
/// [`ExportError::Material`] or [`ExportError::Encode`] when lowering or
/// packing fails, which they do not for a valid material.
pub fn export(source: &Source<'_>, settings: &Settings) -> Result<Exported, ExportError> {
    let (maps, lowering) = lower::maps(source.material)?;
    let mut bundle = pack(&maps, settings.profile, &settings.pack)?;
    let extra = if settings.profile == Profile::Raw || bundle.report.unsupported.is_empty() {
        None
    } else {
        let only = only(&maps, &bundle.report.unsupported);
        Some(pack(&only, Profile::Raw, &settings.pack)?)
    };
    let parameters = account::account(source.material, &maps, &lowering, &bundle, extra.as_ref());
    let mut textures: Vec<Texture> = core::mem::take(&mut bundle.textures)
        .into_iter()
        .map(|encoded| Texture {
            encoded,
            in_profile: true,
        })
        .collect();
    if let Some(extra) = extra {
        // Only the textures that hold what the profile could not carry:
        // others come along as their dependencies (normals for roughness).
        for encoded in extra.textures {
            let wanted = extra
                .placements
                .iter()
                .any(|p| p.texture == encoded.name && bundle.report.unsupported.contains(&p.name));
            if wanted {
                debug_assert!(
                    textures.iter().all(|t| t.encoded.name != encoded.name),
                    "raw texture names are parameter names, which no profile texture has"
                );
                textures.push(Texture {
                    encoded,
                    in_profile: false,
                });
            }
        }
    }
    Ok(Exported {
        settings: *settings,
        provenance: provenance(source),
        parameters,
        textures,
        report: bundle.report,
        lab: source.lab.map(LabSummary::from),
    })
}

/// `maps` with only the fields named, and what packing them needs: the
/// normals whose variance widens a roughness.
fn only(maps: &MaterialMaps, names: &[&str]) -> MaterialMaps {
    fn keep<T: Clone>(names: &[&str], name: &str, field: &Option<T>) -> Option<T> {
        if names.contains(&name) {
            field.clone()
        } else {
            None
        }
    }
    let has = |name| names.contains(&name);
    MaterialMaps {
        base_color: keep(names, "base_color", &maps.base_color),
        opacity: keep(names, "opacity", &maps.opacity),
        normal: if has("normal") || has("specular_roughness") {
            maps.normal.clone()
        } else {
            None
        },
        specular_roughness: keep(names, "specular_roughness", &maps.specular_roughness),
        base_metalness: keep(names, "base_metalness", &maps.base_metalness),
        occlusion: keep(names, "occlusion", &maps.occlusion),
        anisotropy_direction: keep(names, "anisotropy_direction", &maps.anisotropy_direction),
        specular_roughness_anisotropy: keep(
            names,
            "specular_roughness_anisotropy",
            &maps.specular_roughness_anisotropy,
        ),
        specular_weight: keep(names, "specular_weight", &maps.specular_weight),
        specular_color: keep(names, "specular_color", &maps.specular_color),
        specular_ior: keep(names, "specular_ior", &maps.specular_ior),
        subsurface_weight: keep(names, "subsurface_weight", &maps.subsurface_weight),
        subsurface_color: keep(names, "subsurface_color", &maps.subsurface_color),
        transmission_weight: keep(names, "transmission_weight", &maps.transmission_weight),
        coat_weight: keep(names, "coat_weight", &maps.coat_weight),
        coat_roughness: keep(names, "coat_roughness", &maps.coat_roughness),
        coat_color: keep(names, "coat_color", &maps.coat_color),
        coat_normal: if has("coat_normal") || has("coat_roughness") {
            maps.coat_normal.clone()
        } else {
            None
        },
        fuzz_weight: keep(names, "fuzz_weight", &maps.fuzz_weight),
        fuzz_color: keep(names, "fuzz_color", &maps.fuzz_color),
        fuzz_roughness: keep(names, "fuzz_roughness", &maps.fuzz_roughness),
        emission_luminance: keep(names, "emission_luminance", &maps.emission_luminance),
        emission_color: keep(names, "emission_color", &maps.emission_color),
        height: keep(names, "height", &maps.height),
        thin_walled: has("thin_walled") && maps.thin_walled,
    }
}

fn provenance(source: &Source<'_>) -> Provenance {
    let grid = source.material.grid();
    let tiling = source.material.tiling();
    #[expect(clippy::cast_precision_loss, reason = "grid sizes are at most 2^15")]
    let size = [
        grid.width as f32 * grid.texel.x,
        grid.height as f32 * grid.texel.y,
    ];
    let instance = source.diagnostics.and_then(|d| {
        let (path, module) = d.instances().next()?;
        Some(Instance {
            path: path.to_string(),
            module: module.name.to_string(),
            version: module.version,
            fingerprint: alloc::format!("{:016x}", d.fingerprint_at(path)?),
        })
    });
    Provenance {
        subject: source.subject.to_string(),
        instance,
        grid: GridInfo {
            width: grid.width,
            height: grid.height,
            origin: grid.origin.to_array(),
            texel: grid.texel.to_array(),
            size,
            edge: match grid.edge {
                dapple_encode::Edge::Wrap => "wrap",
                dapple_encode::Edge::Clamp => "clamp",
            }
            .to_string(),
        },
        tiling: TilingInfo {
            x: tiling.x,
            y: tiling.y,
        },
    }
}

impl Exported {
    /// The account of the parameter named `name` (an OpenPBR identifier,
    /// `"occlusion"` or `"height"`).
    #[must_use]
    pub fn parameter(&self, name: &str) -> Option<&Parameter> {
        self.parameters.iter().find(|p| p.name == name)
    }

    /// The texture named `name`.
    #[must_use]
    pub fn texture(&self, name: &str) -> Option<&Texture> {
        self.textures.iter().find(|t| t.encoded.name == name)
    }

    /// The manifest of this export, listing `files` (as written, with
    /// their hashes); [`Self::write_dir`] passes the files it wrote.
    #[must_use]
    pub fn manifest(&self, files: Vec<FileEntry>) -> Manifest {
        let pack = &self.settings.pack;
        Manifest {
            format: FORMAT.to_string(),
            version: VERSION,
            packing: Packing {
                profile: account::profile_name(self.settings.profile).to_string(),
                filter: match pack.filter {
                    Filter::Box => "box",
                    Filter::Kaiser => "kaiser",
                }
                .to_string(),
                alpha_cutoff: pack.alpha_cutoff,
                fold_normal_variance: pack.fold_normal_variance,
                sixteen_bit_normals: pack.sixteen_bit_normals,
            },
            provenance: self.provenance.clone(),
            conventions: Conventions::dapple(),
            parameters: self.parameters.clone(),
            textures: self.textures.iter().map(texture_entry).collect(),
            gltf: (self.settings.profile == Profile::Gltf).then(|| GLTF_FILE.to_string()),
            files,
            lab: self.lab.clone(),
        }
    }
}

fn texture_entry(texture: &Texture) -> TextureEntry {
    let encoded = &texture.encoded;
    // PNG has no two-channel color type: `dapple_encode::png` writes RG as
    // RGB with blue 0.
    let channels = match encoded.format {
        PixelFormat::Rg8Unorm => 3,
        format => format.channels(),
    };
    TextureEntry {
        name: encoded.name.to_string(),
        file: texture.file(),
        in_profile: texture.in_profile,
        color_space: if encoded.format.is_srgb() {
            "srgb"
        } else {
            "linear"
        }
        .to_string(),
        channels: u32::try_from(channels).expect("at most 4 channels"),
        bits: u32::try_from(encoded.format.bytes_per_channel() * 8).expect("at most 32 bits"),
        width: encoded.width,
        height: encoded.height,
    }
}

#[cfg(test)]
mod tests;
