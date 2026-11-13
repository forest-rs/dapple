// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Writing an export into a directory, and reading its manifest back.

use std::path::Path;
use std::string::{String, ToString};
use std::vec::Vec;

use crate::{
    ExportError, Exported, FileEntry, GLTF_FILE, MANIFEST_FILE, Manifest, content_hash, gltf,
};

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> ExportError + '_ {
    move |source| ExportError::Io {
        path: path.to_path_buf(),
        source,
    }
}

impl Exported {
    /// Writes the export into `dir`, creating it if needed, and returns the
    /// manifest it wrote.
    ///
    /// Each texture's level 0 is written as `<name>.png`, then, for a glTF
    /// export, [`GLTF_FILE`] ([`gltf::document`]), then [`MANIFEST_FILE`],
    /// listing every other file with its size and [`content_hash`]. An
    /// existing manifest is removed first and the new one written last, so
    /// a manifest in a directory always vouches for complete files. Other
    /// files already in `dir` are left alone.
    ///
    /// # Errors
    ///
    /// [`ExportError::Io`] when a file cannot be written and
    /// [`ExportError::Png`] when a texture cannot be encoded (a float
    /// texture, which packing does not make).
    pub fn write_dir(&self, dir: &Path) -> Result<Manifest, ExportError> {
        std::fs::create_dir_all(dir).map_err(io(dir))?;
        let manifest_path = dir.join(MANIFEST_FILE);
        match std::fs::remove_file(&manifest_path) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
                return Err(io(&manifest_path)(error));
            }
            _ => {}
        }
        let mut files = Vec::new();
        let mut write = |name: String, bytes: &[u8]| {
            let path = dir.join(&name);
            std::fs::write(&path, bytes).map_err(io(&path))?;
            files.push(FileEntry {
                path: name,
                bytes: bytes.len() as u64,
                hash: content_hash(bytes).to_string(),
            });
            Ok::<_, ExportError>(())
        };
        for texture in &self.textures {
            let png =
                dapple_encode::png::write(&texture.encoded).map_err(|source| ExportError::Png {
                    texture: texture.encoded.name,
                    source,
                })?;
            write(texture.file(), &png)?;
        }
        if let Some(document) = gltf::document(self) {
            write(GLTF_FILE.to_string(), document.as_bytes())?;
        }
        let manifest = self.manifest(files);
        std::fs::write(&manifest_path, manifest.to_json()).map_err(io(&manifest_path))?;
        Ok(manifest)
    }
}

/// Reads the manifest [`Exported::write_dir`] wrote into `dir`.
///
/// # Errors
///
/// [`ExportError::Io`] when it cannot be read and
/// [`ExportError::Manifest`] when it is not a manifest this crate reads.
pub fn read_dir(dir: &Path) -> Result<Manifest, ExportError> {
    let path = dir.join(MANIFEST_FILE);
    let text = std::fs::read_to_string(&path).map_err(io(&path))?;
    Ok(Manifest::from_json(&text)?)
}

impl Manifest {
    /// The files the manifest lists, relative to `dir`, that are missing
    /// or whose content no longer has the listed hash and size, in list
    /// order. Empty when the directory still holds what was written, which
    /// is what a bake cache needs to trust it.
    #[must_use]
    pub fn changed_files(&self, dir: &Path) -> Vec<&str> {
        self.files
            .iter()
            .filter(|file| {
                std::fs::read(dir.join(&file.path)).map_or(true, |bytes| {
                    bytes.len() as u64 != file.bytes
                        || content_hash(&bytes).to_string() != file.hash
                })
            })
            .map(|file| file.path.as_str())
            .collect()
    }
}
