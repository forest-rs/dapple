// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `dapple_bake`: bakes the material recipes a manifest lists into textures.
//!
//! This is dapple's counterpart to Lightweald's `texture_bake`, for
//! procedural sources: `texture_bake` bakes photographs, `dapple_bake` bakes
//! [`Recipe`]s. A manifest (TOML) names materials:
//!
//! ```toml
//! [[material]]
//! id = "oak_bark"
//! recipe = "recipes/oak_bark.toml"   # relative to the manifest
//! profiles = ["lightweald", "gltf"]  # default: both
//! filter = "kaiser"                  # "box" (default) or "kaiser"
//! alpha_cutoff = 0.5                 # optional; preserves opacity coverage
//! ```
//!
//! Each recipe's outputs name `dapple_encode` material roles (`base_color`,
//! `opacity`, `normal`, `specular_roughness`, `base_metalness`, `occlusion`,
//! `anisotropy_direction`, `specular_roughness_anisotropy`,
//! `subsurface_weight`, `subsurface_color`, `transmission_weight`) and the
//! raster nodes that fill them, a scalar raster per channel or one normal
//! raster.
//!
//! Output lands in `<out>/<profile>/<material>/`: Lightweald textures as
//! KTX2, uncompressed and in each requested pool encoding, and glTF and raw
//! textures as PNG. A stamp file per material records the recipe fingerprint,
//! the bake settings and the files written; a material whose stamp matches
//! and whose files are all present is skipped unless forced, so re-baking an
//! unchanged manifest does no work. Baking preserves earlier outputs when a
//! profile, texture or material is dropped; output cleanup is explicit.
//!
//! The stamp cannot see a change to the encoders themselves: bump
//! [`STAMP_VERSION`] when the output for the same inputs changes, or bake
//! with `--force`.

use std::fmt;
use std::path::{Path, PathBuf};

use dapple_compress::{CompressSettings, Encoding, Quality, compress};
use dapple_encode::{Bundle, Filter, Image, MaterialMaps, PackSettings, Profile, ktx2, pack, png};
use dapple_field::program::Fingerprint;
use dapple_graph::{RasterData, Recipe};
use dapple_raster::typed::Storage;
use serde::Deserialize;

/// Tile size used to realize recipes.
const TILE_SIZE: u32 = 64;

/// The version of what a stamp promises. Bump it whenever the same recipe and
/// settings would write different files (an encoder or packing change), so
/// stamps from before the change stop matching and everything rebakes.
pub const STAMP_VERSION: u32 = 1;

/// A manifest: the materials to bake.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// The materials, in bake order.
    #[serde(rename = "material", default)]
    pub materials: Vec<MaterialEntry>,
}

/// One material of a [`Manifest`].
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MaterialEntry {
    /// Output directory name; unique in the manifest.
    pub id: String,
    /// The recipe file, relative to the manifest.
    pub recipe: PathBuf,
    /// Packing profiles to write.
    #[serde(default = "default_profiles")]
    pub profiles: Vec<ProfileName>,
    /// Mip filter.
    #[serde(default)]
    pub filter: FilterName,
    /// Alpha-test cutoff whose opacity coverage the mips preserve.
    pub alpha_cutoff: Option<f32>,
}

fn default_profiles() -> Vec<ProfileName> {
    vec![ProfileName::Lightweald, ProfileName::Gltf]
}

/// A packing profile, as a manifest names it.
#[derive(Copy, Clone, Debug, Deserialize, Eq, PartialEq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ProfileName {
    /// [`Profile::Lightweald`].
    Lightweald,
    /// [`Profile::Gltf`].
    Gltf,
    /// [`Profile::Raw`].
    Raw,
}

impl ProfileName {
    const fn profile(self) -> Profile {
        match self {
            Self::Lightweald => Profile::Lightweald,
            Self::Gltf => Profile::Gltf,
            Self::Raw => Profile::Raw,
        }
    }

    const fn dir(self) -> &'static str {
        match self {
            Self::Lightweald => "lightweald",
            Self::Gltf => "gltf",
            Self::Raw => "raw",
        }
    }
}

/// A mip filter, as a manifest names it.
#[derive(Copy, Clone, Debug, Default, Deserialize, Eq, PartialEq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum FilterName {
    /// [`Filter::Box`].
    #[default]
    Box,
    /// [`Filter::Kaiser`].
    Kaiser,
}

/// Why a bake failed.
#[derive(Debug)]
pub enum BakeError {
    /// A file could not be read or written.
    Io(PathBuf, std::io::Error),
    /// A manifest or recipe is not valid TOML for its schema.
    Parse(PathBuf, String),
    /// Two materials share an id.
    DuplicateId(String),
    /// A material id is not a single plain directory name.
    InvalidId(String),
    /// A recipe is malformed, or its graph failed to run.
    Recipe(String, dapple_graph::RecipeError),
    /// A recipe output names a role no material map has.
    UnknownRole {
        /// The material.
        material: String,
        /// The role.
        role: String,
    },
    /// A role's sources have the wrong number of channels, or differ in size.
    Channels {
        /// The material.
        material: String,
        /// The role.
        role: String,
    },
    /// Packing, compression or file encoding failed.
    Encode(String, String),
}

impl fmt::Display for BakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(path, error) => write!(f, "{}: {error}", path.display()),
            Self::Parse(path, error) => write!(f, "{}: {error}", path.display()),
            Self::DuplicateId(id) => write!(f, "material id {id:?} is used twice"),
            Self::InvalidId(id) => {
                write!(f, "material id {id:?} is not a single plain directory name")
            }
            Self::Recipe(id, error) => write!(f, "{id}: {error}"),
            Self::UnknownRole { material, role } => {
                write!(f, "{material}: no material map is called {role:?}")
            }
            Self::Channels { material, role } => write!(
                f,
                "{material}: the sources of {role:?} have the wrong channel count or size"
            ),
            Self::Encode(id, error) => write!(f, "{id}: {error}"),
        }
    }
}

impl std::error::Error for BakeError {}

/// Reads a manifest.
///
/// # Errors
///
/// [`BakeError::Io`], [`BakeError::Parse`], [`BakeError::DuplicateId`] or
/// [`BakeError::InvalidId`].
pub fn read_manifest(path: &Path) -> Result<Manifest, BakeError> {
    let text = std::fs::read_to_string(path).map_err(|e| BakeError::Io(path.into(), e))?;
    let manifest: Manifest =
        toml::from_str(&text).map_err(|e| BakeError::Parse(path.into(), e.to_string()))?;
    for (i, entry) in manifest.materials.iter().enumerate() {
        if !is_plain_name(&entry.id) {
            return Err(BakeError::InvalidId(entry.id.clone()));
        }
        if manifest.materials[..i].iter().any(|m| m.id == entry.id) {
            return Err(BakeError::DuplicateId(entry.id.clone()));
        }
    }
    Ok(manifest)
}

/// Whether `id` is one ordinary path component: ids become directory names
/// under the output, so `""`, `".."`, `"a/b"` and absolute paths must not
/// get through.
fn is_plain_name(id: &str) -> bool {
    // Control characters would break the line-based stamp: an id holding a
    // newline could write a path of its own choosing into it.
    !id.chars().any(char::is_control) && is_one_normal_component(Path::new(id), Some(id))
}

/// Whether `path` is exactly one ordinary component (equal to `name` when
/// given).
fn is_one_normal_component(path: &Path, name: Option<&str>) -> bool {
    let mut components = path.components();
    matches!(
        (components.next(), components.next()),
        (Some(std::path::Component::Normal(part)), None) if name.is_none_or(|n| part == n)
    )
}

/// Reads a recipe written as TOML.
///
/// # Errors
///
/// [`BakeError::Io`] or [`BakeError::Parse`].
pub fn read_recipe(path: &Path) -> Result<Recipe, BakeError> {
    let text = std::fs::read_to_string(path).map_err(|e| BakeError::Io(path.into(), e))?;
    parse_recipe(&text).map_err(|e| BakeError::Parse(path.into(), e))
}

/// Parses a recipe from TOML text.
///
/// # Errors
///
/// The parser's message when the text is not a recipe.
pub fn parse_recipe(text: &str) -> Result<Recipe, String> {
    toml::from_str(text).map_err(|e| e.to_string())
}

/// One baked material: its recipe fingerprint and a bundle per profile.
#[derive(Clone, Debug)]
pub struct Baked {
    /// The recipe's content fingerprint.
    pub fingerprint: Fingerprint,
    /// The packed textures, per profile.
    pub bundles: Vec<(ProfileName, Bundle)>,
}

fn images(id: &str, recipe: &Recipe) -> Result<MaterialMaps, BakeError> {
    let (mut graph, nodes) = recipe
        .build(TILE_SIZE)
        .map_err(|e| BakeError::Recipe(id.into(), e))?;
    graph
        .run()
        .map_err(|e| BakeError::Recipe(id.into(), dapple_graph::RecipeError::Material(e)))?;
    let mut maps = MaterialMaps::default();
    for output in &recipe.outputs {
        let wrong = || BakeError::Channels {
            material: id.into(),
            role: output.role.clone(),
        };
        let (slot, expected) = match output.role.as_str() {
            "base_color" => (&mut maps.base_color, 3),
            "opacity" => (&mut maps.opacity, 1),
            "normal" => (&mut maps.normal, 3),
            "specular_roughness" => (&mut maps.specular_roughness, 1),
            "base_metalness" => (&mut maps.base_metalness, 1),
            "occlusion" => (&mut maps.occlusion, 1),
            "anisotropy_direction" => (&mut maps.anisotropy_direction, 2),
            "specular_roughness_anisotropy" => (&mut maps.specular_roughness_anisotropy, 1),
            "subsurface_weight" => (&mut maps.subsurface_weight, 1),
            "subsurface_color" => (&mut maps.subsurface_color, 3),
            "transmission_weight" => (&mut maps.transmission_weight, 1),
            _ => {
                return Err(BakeError::UnknownRole {
                    material: id.into(),
                    role: output.role.clone(),
                });
            }
        };
        // Interleave the sources' channels texel by texel.
        let sources: Vec<Image> = output
            .channels
            .iter()
            .map(|label| {
                let value = graph
                    .raster_value(nodes[label])
                    .expect("recipe outputs are built raster nodes that ran");
                Ok(match &value.data {
                    RasterData::Scalar(r) => Image::from(r),
                    RasterData::Vector3(r) => Image::from(r),
                    RasterData::Typed(r) => match r.storage() {
                        Storage::F32(r) => Image::from(r),
                        Storage::F32x2(r) => Image::from(r),
                        Storage::F32x3(r) => Image::from(r),
                        // Identifiers are not texture channels.
                        Storage::U32(_) => return Err(wrong()),
                    },
                })
            })
            .collect::<Result<_, _>>()?;
        let first = sources.first().ok_or_else(wrong)?;
        let grid = |i: &Image| (i.width(), i.height(), i.edge());
        if sources.iter().any(|s| grid(s) != grid(first)) {
            return Err(wrong());
        }
        let texels = first.width() as usize * first.height() as usize;
        let channels: usize = sources.iter().map(Image::channels).sum();
        if channels != expected {
            return Err(wrong());
        }
        let mut values = Vec::with_capacity(texels * channels);
        for i in 0..texels {
            for source in &sources {
                let c = source.channels();
                values.extend_from_slice(&source.values()[i * c..(i + 1) * c]);
            }
        }
        *slot = Some(
            Image::new(
                first.width(),
                first.height(),
                channels,
                first.edge(),
                values,
            )
            .map_err(|_| wrong())?,
        );
    }
    Ok(maps)
}

/// Runs `recipe` and packs its outputs for `entry`'s profiles.
///
/// # Errors
///
/// [`BakeError::Recipe`], [`BakeError::UnknownRole`],
/// [`BakeError::Channels`] or [`BakeError::Encode`].
pub fn bake(entry: &MaterialEntry, recipe: &Recipe) -> Result<Baked, BakeError> {
    let fingerprint = recipe
        .fingerprint()
        .map_err(|e| BakeError::Recipe(entry.id.clone(), e))?;
    let maps = images(&entry.id, recipe)?;
    let settings = PackSettings {
        filter: match entry.filter {
            FilterName::Box => Filter::Box,
            FilterName::Kaiser => Filter::Kaiser,
        },
        alpha_cutoff: entry.alpha_cutoff,
        ..PackSettings::default()
    };
    let bundles = entry
        .profiles
        .iter()
        .map(|&name| {
            pack(&maps, name.profile(), &settings)
                .map(|bundle| (name, bundle))
                .map_err(|e| BakeError::Encode(entry.id.clone(), e.to_string()))
        })
        .collect::<Result<_, _>>()?;
    Ok(Baked {
        fingerprint,
        bundles,
    })
}

/// How to write baked textures.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WriteSettings {
    /// Lightweald pool encodings besides uncompressed KTX2.
    pub encodings: Vec<Encoding>,
    /// Encoder effort for compressed encodings.
    pub quality: Quality,
}

impl WriteSettings {
    /// What the bake depends on other than the files it wrote: the recipe,
    /// the settings, the realization tile size, and [`STAMP_VERSION`].
    fn stamp_header(&self, entry: &MaterialEntry, fingerprint: Fingerprint) -> String {
        format!(
            "dapple_bake {STAMP_VERSION}\ntile_size {TILE_SIZE}\nrecipe {fingerprint}\nprofiles {:?}\nfilter {:?}\nalpha_cutoff {:?}\nencodings {:?}\nquality {:?}\n",
            entry.profiles, entry.filter, entry.alpha_cutoff, self.encodings, self.quality
        )
    }
}

/// Marks where a stamp's header ends and its list of written files begins:
/// a line of its own.
const FILES_MARK: &str = "files\n";

/// A material's stamp as an earlier bake left it.
struct OldStamp {
    header: String,
    /// The files it wrote, relative to the output directory.
    files: Vec<PathBuf>,
}

/// Reads a complete stamp, or `None` for one that is missing, from an older
/// format, or malformed. The final file count detects a truncated list,
/// including one ending at a file boundary. Listed paths must be relative
/// to the output directory and contain only ordinary components.
fn read_stamp(path: &Path) -> Option<OldStamp> {
    let text = std::fs::read_to_string(path).ok()?;
    let end = text.find(&format!("\n{FILES_MARK}"))? + 1;
    let (header, files) = text.split_at(end);
    let body = files[FILES_MARK.len()..].strip_suffix('\n')?;
    let (files, footer) = body.rsplit_once('\n').unwrap_or(("", body));
    let count: usize = footer.strip_prefix("complete ")?.parse().ok()?;
    let files = files
        .lines()
        .map(|line| {
            let file = PathBuf::from(line);
            let plain = file
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_)));
            (plain && !line.is_empty()).then_some(file)
        })
        .collect::<Option<Vec<_>>>()?;
    if files.len() != count {
        return None;
    }
    Some(OldStamp {
        header: header.to_owned(),
        files,
    })
}

/// What happened to one material.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Its stamp matched: nothing was baked.
    Cached,
    /// It was baked: files written and texture bytes.
    Baked {
        /// Files written.
        files: usize,
        /// Bytes written.
        bytes: u64,
        /// Maps a profile could not carry, per profile.
        unsupported: Vec<(ProfileName, Vec<&'static str>)>,
    },
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<u64, BakeError> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| BakeError::Io(dir.into(), e))?;
    }
    std::fs::write(path, bytes).map_err(|e| BakeError::Io(path.into(), e))?;
    Ok(bytes.len() as u64)
}

/// Bakes every material of the manifest at `manifest_path` into `out`.
///
/// Materials whose stamp matches, and whose files are all still there, are
/// skipped unless `force`. Earlier outputs are preserved when a profile,
/// texture or material is dropped. Returns each material's id and outcome,
/// in manifest order.
///
/// # Errors
///
/// The first material's failure; earlier materials stay written.
pub fn bake_manifest(
    manifest_path: &Path,
    out: &Path,
    settings: &WriteSettings,
    force: bool,
) -> Result<Vec<(String, Outcome)>, BakeError> {
    let manifest = read_manifest(manifest_path)?;
    let base = manifest_path.parent().unwrap_or(Path::new("."));
    let mut outcomes = Vec::new();
    for entry in &manifest.materials {
        let recipe = read_recipe(&base.join(&entry.recipe))?;
        let fingerprint = recipe
            .fingerprint()
            .map_err(|e| BakeError::Recipe(entry.id.clone(), e))?;
        let stamp_path = out.join(".stamps").join(&entry.id);
        let header = settings.stamp_header(entry, fingerprint);
        let old = read_stamp(&stamp_path);
        let intact = old.as_ref().is_some_and(|old| {
            old.header == header && old.files.iter().all(|f| out.join(f).is_file())
        });
        if !force && intact {
            outcomes.push((entry.id.clone(), Outcome::Cached));
            continue;
        }
        let baked = bake(entry, &recipe)?;
        let (mut files, mut bytes) = (0, 0);
        let mut written: Vec<PathBuf> = Vec::new();
        let mut unsupported = Vec::new();
        let encode_error =
            |e: &dyn fmt::Display| BakeError::Encode(entry.id.clone(), e.to_string());
        // Nothing from an earlier bake vouches for a half-written one.
        match std::fs::remove_file(&stamp_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(BakeError::Io(stamp_path, error)),
        }
        for (name, bundle) in &baked.bundles {
            // Paths are kept relative to `out`, as the stamp records them.
            let rel_dir = Path::new(name.dir()).join(&entry.id);
            if !bundle.report.unsupported.is_empty() {
                unsupported.push((*name, bundle.report.unsupported.clone()));
            }
            for texture in &bundle.textures {
                if *name == ProfileName::Lightweald {
                    let rel = rel_dir.join(format!("{}.ktx2", texture.name));
                    bytes += write_file(&out.join(&rel), &ktx2::write(texture))?;
                    files += 1;
                    written.push(rel);
                    for &encoding in &settings.encodings {
                        let encoded = compress(
                            texture,
                            CompressSettings {
                                encoding,
                                quality: settings.quality,
                                zstd: true,
                            },
                        )
                        .map_err(|e| encode_error(&e))?;
                        let sub = match encoding {
                            Encoding::Uncompressed => "uncompressed",
                            Encoding::Bc => "bc",
                            Encoding::Astc => "astc",
                        };
                        let rel = rel_dir.join(sub).join(format!("{}.ktx2", texture.name));
                        bytes += write_file(&out.join(&rel), &encoded)?;
                        files += 1;
                        written.push(rel);
                    }
                } else {
                    let rel = rel_dir.join(format!("{}.png", texture.name));
                    let data = png::write(texture).map_err(|e| encode_error(&e))?;
                    bytes += write_file(&out.join(&rel), &data)?;
                    files += 1;
                    written.push(rel);
                }
            }
        }
        let mut stamp = header;
        stamp.push_str(FILES_MARK);
        for file in &written {
            // Always `/`, so a stamp reads the same on every platform.
            let parts: Vec<_> = file.iter().map(|p| p.to_string_lossy()).collect();
            stamp.push_str(&parts.join("/"));
            stamp.push('\n');
        }
        // A partial write, even at a file boundary, cannot vouch for a bake.
        stamp.push_str(&format!("complete {}\n", written.len()));
        write_file(&stamp_path, stamp.as_bytes())?;
        outcomes.push((
            entry.id.clone(),
            Outcome::Baked {
                files,
                bytes,
                unsupported,
            },
        ));
    }
    Ok(outcomes)
}

#[cfg(test)]
mod tests;
