// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! glTF 2.0 output of a glTF export: the material, and a complete document
//! that shows it on a quad.
//!
//! [`material`] maps the account onto glTF's slots. A texture the profile
//! wrote is referenced with factors of 1, since packing filled its other
//! channels with the constants and defaults they need; without one, the
//! constants become factors. glTF's defaults are not OpenPBR's (glTF's
//! metalness and roughness default to 1, OpenPBR's to 0 and 0.3), so base
//! factors are always written. Extensions are written only when their
//! layer is on:
//!
//! - `KHR_materials_specular`: a weight above 1 goes into
//!   `specularColorFactor`, as `dapple_encode::pack` describes.
//! - `KHR_materials_ior`, when `specular_ior` is not glTF's 1.5.
//! - `KHR_materials_anisotropy`: a constant strength is converted with
//!   `dapple_encode::gltf_anisotropy_strength`, and a constant tangent at
//!   angle θ in the domain frame (`+Y` down the image) is a rotation of -θ.
//! - `KHR_materials_transmission`, `KHR_materials_diffuse_transmission`
//!   (subsurface), `KHR_materials_clearcoat` (coat), `KHR_materials_sheen`
//!   (fuzz: color times weight) and `KHR_materials_emissive_strength`
//!   (luminance in cd/m², glTF's emissive unit).
//!
//! A thin-walled material is double-sided. Samplers repeat along the axes
//! the material tiles along and clamp along the others. No
//! `KHR_texture_transform` is written: [`document`]'s quad maps one period
//! of the material onto UVs `[0, 1]²`, and a mesh with UVs in meters applies
//! the material with a scale of one over [`crate::GridInfo::size`].

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use dapple_encode::{Profile, gltf_anisotropy_strength};
use serde_json::{Map, Value, json};

use crate::{Exported, Held, Location};

/// A glTF material and what it refers to, for splicing into a document:
/// texture indices count from 0 in [`Self::textures`], which refer to
/// [`Self::images`] and [`Self::samplers`] by index from 0 too.
#[derive(Clone, Debug, PartialEq)]
pub struct Fragment {
    /// The material object.
    pub material: Value,
    /// The texture objects.
    pub textures: Vec<Value>,
    /// The image objects, with URIs relative to the export's directory.
    pub images: Vec<Value>,
    /// The sampler objects.
    pub samplers: Vec<Value>,
    /// The extensions the material uses, sorted.
    pub extensions_used: Vec<&'static str>,
}

/// A JSON number in `v`'s shortest round-tripping decimal form, rather
/// than the `f64` widening of an `f32`; non-finite values are `null`.
fn num(v: f32) -> Value {
    format!("{v}")
        .parse::<f64>()
        .ok()
        .and_then(serde_json::Number::from_f64)
        .map_or(Value::Null, Value::Number)
}

fn nums(values: &[f32]) -> Value {
    Value::Array(values.iter().map(|&v| num(v)).collect())
}

/// The account, as glTF's slots read it.
struct Slots<'a> {
    exported: &'a Exported,
    /// The names of the textures the glTF material uses, by glTF index.
    textures: Vec<&'static str>,
}

impl Slots<'_> {
    fn held(&self, name: &str) -> Option<&Held> {
        self.exported
            .parameter(name)
            .filter(|p| p.in_profile)
            .map(|p| &p.held)
    }

    /// The parameter's constant, or `default` when it is not given.
    fn value<const N: usize>(&self, name: &str, default: [f32; N]) -> [f32; N] {
        match self.held(name) {
            Some(Held::Constant { value }) => value.as_slice().try_into().unwrap_or(default),
            _ => default,
        }
    }

    /// Whether the parameter's lobe is on: a texture, or a constant not 0.
    fn on(&self, name: &str) -> bool {
        match self.held(name) {
            Some(Held::Texture(_)) => true,
            Some(Held::Constant { value }) => value.iter().any(|&v| v != 0.0),
            _ => false,
        }
    }

    fn location(&self, name: &str) -> Option<&Location> {
        match self.held(name) {
            Some(Held::Texture(location)) => Some(location),
            _ => None,
        }
    }

    /// A `textureInfo` for the texture named `texture`, if glTF uses it.
    fn texture(&self, texture: &str) -> Option<Value> {
        let index = self.textures.iter().position(|&t| t == texture)?;
        Some(json!({ "index": index }))
    }
}

/// glTF's sampler wrap mode for an axis that tiles, or does not.
const fn wrap(tiles: bool) -> u32 {
    if tiles {
        10497 // REPEAT
    } else {
        33071 // CLAMP_TO_EDGE
    }
}

/// The glTF material of `exported` and what it refers to, or `None` unless
/// it was exported for [`Profile::Gltf`]. See the [module docs](self).
#[must_use]
pub fn material(exported: &Exported) -> Option<Fragment> {
    if exported.settings.profile != Profile::Gltf {
        return None;
    }
    let used: Vec<_> = exported.textures.iter().filter(|t| t.in_profile).collect();
    let slots = Slots {
        exported,
        textures: used.iter().map(|t| t.encoded.name).collect(),
    };
    let mut m = Map::new();
    let mut extensions = Map::new();
    m.insert(
        "name".into(),
        Value::String(exported.provenance.subject.clone()),
    );

    let mut pbr = Map::new();
    if let Some(texture) = slots.texture("base_color") {
        pbr.insert("baseColorFactor".into(), nums(&[1.0; 4]));
        pbr.insert("baseColorTexture".into(), texture);
    } else {
        let [r, g, b] = slots.value("base_color", [0.8; 3]);
        let [a] = slots.value("geometry_opacity", [1.0]);
        pbr.insert("baseColorFactor".into(), nums(&[r, g, b, a]));
    }
    if let Some(texture) = slots.texture("orm") {
        pbr.insert("metallicFactor".into(), num(1.0));
        pbr.insert("roughnessFactor".into(), num(1.0));
        pbr.insert("metallicRoughnessTexture".into(), texture.clone());
        if slots.location("occlusion").is_some() {
            m.insert("occlusionTexture".into(), texture);
        }
    } else {
        let [metal] = slots.value("base_metalness", [0.0]);
        let [rough] = slots.value("specular_roughness", [0.3]);
        pbr.insert("metallicFactor".into(), num(metal));
        pbr.insert("roughnessFactor".into(), num(rough));
    }
    m.insert("pbrMetallicRoughness".into(), Value::Object(pbr));
    if let Some(texture) = slots.texture("normal") {
        m.insert("normalTexture".into(), texture);
    }
    let opacity = slots.held("geometry_opacity");
    let translucent = match opacity {
        Some(Held::Texture(_)) => true,
        Some(Held::Constant { value }) => value.first().is_some_and(|&a| a < 1.0),
        _ => false,
    };
    if translucent {
        match exported.settings.pack.alpha_cutoff {
            Some(cutoff) => {
                m.insert("alphaMode".into(), json!("MASK"));
                m.insert("alphaCutoff".into(), num(cutoff));
            }
            None => {
                m.insert("alphaMode".into(), json!("BLEND"));
            }
        }
    }
    if slots.value("geometry_thin_walled", [0.0]) != [0.0] {
        m.insert("doubleSided".into(), Value::Bool(true));
    }
    emission(&slots, &mut m, &mut extensions);
    specular(&slots, &mut extensions);
    anisotropy(&slots, &mut extensions);
    translucency(&slots, &mut extensions);
    coat_and_fuzz(&slots, &mut extensions);
    let extensions_used: Vec<&'static str> = extensions
        .keys()
        .map(|k| {
            *EXTENSIONS
                .iter()
                .find(|e| **e == k.as_str())
                .expect("only known extensions are written")
        })
        .collect();
    if !extensions.is_empty() {
        m.insert("extensions".into(), Value::Object(extensions));
    }

    let tiling = exported.provenance.tiling;
    Some(Fragment {
        material: Value::Object(m),
        textures: used
            .iter()
            .enumerate()
            .map(|(i, t)| json!({ "name": t.encoded.name, "sampler": 0, "source": i }))
            .collect(),
        images: used
            .iter()
            .map(|t| json!({ "name": t.encoded.name, "uri": t.file() }))
            .collect(),
        samplers: if used.is_empty() {
            Vec::new()
        } else {
            alloc::vec![json!({
                "magFilter": 9729, // LINEAR
                "minFilter": 9987, // LINEAR_MIPMAP_LINEAR
                "wrapS": wrap(tiling.x),
                "wrapT": wrap(tiling.y),
            })]
        },
        extensions_used,
    })
}

/// Every extension [`material`] can write, sorted.
const EXTENSIONS: [&str; 8] = [
    "KHR_materials_anisotropy",
    "KHR_materials_clearcoat",
    "KHR_materials_diffuse_transmission",
    "KHR_materials_emissive_strength",
    "KHR_materials_ior",
    "KHR_materials_sheen",
    "KHR_materials_specular",
    "KHR_materials_transmission",
];

fn emission(slots: &Slots<'_>, m: &mut Map<String, Value>, extensions: &mut Map<String, Value>) {
    let (factor, strength) = if let Some(texture) = slots.texture("emissive") {
        m.insert("emissiveTexture".into(), texture);
        // The texture holds the luminance over its largest value.
        let largest = slots
            .location("emission_luminance")
            .map_or(1.0, |l| l.range[1]);
        ([1.0; 3], largest)
    } else {
        let [luminance] = slots.value("emission_luminance", [0.0]);
        if luminance <= 0.0 {
            return;
        }
        let color = slots.value("emission_color", [1.0; 3]);
        // glTF's emissive factor is at most 1: a brighter color raises the
        // strength instead.
        let peak = color.iter().fold(1.0_f32, |m, &c| m.max(c));
        (color.map(|c| c / peak), luminance * peak)
    };
    m.insert("emissiveFactor".into(), nums(&factor));
    if strength != 1.0 {
        extensions.insert(
            "KHR_materials_emissive_strength".into(),
            json!({ "emissiveStrength": num(strength) }),
        );
    }
}

fn specular(slots: &Slots<'_>, extensions: &mut Map<String, Value>) {
    let mut e = Map::new();
    if let Some(texture) = slots.texture("specular") {
        e.insert("specularTexture".into(), texture.clone());
        e.insert("specularColorTexture".into(), texture);
        // A weight above 1 is stored over its largest value, which glTF's
        // color factor (unbounded, unlike `specularFactor`) restores.
        let scale = slots
            .location("specular_weight")
            .map_or(1.0, |l| l.range[1].max(1.0));
        e.insert("specularFactor".into(), num(1.0));
        e.insert("specularColorFactor".into(), nums(&[scale; 3]));
    } else {
        let given = |name| matches!(slots.held(name), Some(Held::Constant { .. }));
        if !given("specular_weight") && !given("specular_color") {
            return;
        }
        let [weight] = slots.value("specular_weight", [1.0]);
        let color = slots.value("specular_color", [1.0; 3]);
        let (factor, color) = if weight <= 1.0 {
            (weight, color)
        } else {
            (1.0, color.map(|c| c * weight))
        };
        e.insert("specularFactor".into(), num(factor));
        e.insert("specularColorFactor".into(), nums(&color));
    }
    extensions.insert("KHR_materials_specular".into(), Value::Object(e));
    let [ior] = slots.value("specular_ior", [1.5]);
    if ior != 1.5 {
        extensions.insert("KHR_materials_ior".into(), json!({ "ior": num(ior) }));
    }
}

fn anisotropy(slots: &Slots<'_>, extensions: &mut Map<String, Value>) {
    let e = if let Some(texture) = slots.texture("anisotropy") {
        json!({ "anisotropyStrength": num(1.0), "anisotropyTexture": texture })
    } else {
        let [anisotropy] = slots.value("specular_roughness_anisotropy", [0.0]);
        if anisotropy <= 0.0 {
            return;
        }
        // The tangent's projection onto the surface; along the normal it
        // has no axis, and the lobe is isotropic.
        let [x, y, _] = slots.value("geometry_tangent", [1.0, 0.0, 0.0]);
        if x == 0.0 && y == 0.0 {
            return;
        }
        let [roughness] = slots.value("specular_roughness", [0.3]);
        // glTF's +Y is toward the image top, the domain's -Y.
        json!({
            "anisotropyStrength": num(gltf_anisotropy_strength(roughness, anisotropy)),
            "anisotropyRotation": num(-libm::atan2f(y, x)),
        })
    };
    extensions.insert("KHR_materials_anisotropy".into(), e);
}

fn translucency(slots: &Slots<'_>, extensions: &mut Map<String, Value>) {
    if let Some(texture) = slots.texture("transmission") {
        extensions.insert(
            "KHR_materials_transmission".into(),
            json!({ "transmissionFactor": num(1.0), "transmissionTexture": texture }),
        );
    } else if slots.on("transmission_weight") {
        let [weight] = slots.value("transmission_weight", [0.0]);
        extensions.insert(
            "KHR_materials_transmission".into(),
            json!({ "transmissionFactor": num(weight) }),
        );
    }
    let e = if let Some(texture) = slots.texture("diffuse_transmission") {
        json!({
            "diffuseTransmissionFactor": num(1.0),
            "diffuseTransmissionTexture": texture,
            "diffuseTransmissionColorFactor": nums(&[1.0; 3]),
            "diffuseTransmissionColorTexture": texture,
        })
    } else if slots.on("subsurface_weight") {
        let [weight] = slots.value("subsurface_weight", [0.0]);
        json!({
            "diffuseTransmissionFactor": num(weight),
            "diffuseTransmissionColorFactor": nums(&slots.value("subsurface_color", [0.8; 3])),
        })
    } else {
        return;
    };
    extensions.insert("KHR_materials_diffuse_transmission".into(), e);
}

fn coat_and_fuzz(slots: &Slots<'_>, extensions: &mut Map<String, Value>) {
    let mut coat = if let Some(texture) = slots.texture("clearcoat") {
        json!({
            "clearcoatFactor": num(1.0),
            "clearcoatTexture": texture,
            "clearcoatRoughnessFactor": num(1.0),
            "clearcoatRoughnessTexture": texture,
        })
    } else if slots.on("coat_weight") {
        json!({
            "clearcoatFactor": num(slots.value("coat_weight", [0.0])[0]),
            "clearcoatRoughnessFactor": num(slots.value("coat_roughness", [0.0])[0]),
        })
    } else {
        Value::Null
    };
    if let (Value::Object(coat), Some(normal)) = (&mut coat, slots.texture("clearcoat_normal")) {
        coat.insert("clearcoatNormalTexture".into(), normal);
    }
    if !coat.is_null() {
        extensions.insert("KHR_materials_clearcoat".into(), coat);
    }
    let sheen = if let Some(texture) = slots.texture("sheen") {
        json!({
            "sheenColorFactor": nums(&[1.0; 3]),
            "sheenColorTexture": texture,
            "sheenRoughnessFactor": num(1.0),
            "sheenRoughnessTexture": texture,
        })
    } else if slots.on("fuzz_weight") {
        let [weight] = slots.value("fuzz_weight", [0.0]);
        let color = slots.value("fuzz_color", [1.0; 3]).map(|c| c * weight);
        json!({
            "sheenColorFactor": nums(&color),
            "sheenRoughnessFactor": num(slots.value("fuzz_roughness", [0.5])[0]),
        })
    } else {
        return;
    };
    extensions.insert("KHR_materials_sheen".into(), sheen);
}

/// Standard base64 with padding, for the quad's data URI.
fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for k in 0..4 {
            if k <= chunk.len() {
                out.push(char::from(ALPHABET[(n >> (18 - 6 * k)) as usize & 63]));
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// A complete glTF 2.0 document showing `exported`'s material on one quad,
/// or `None` unless it was exported for [`Profile::Gltf`].
///
/// The quad is the material's physical size ([`crate::GridInfo::size`], in
/// meters, glTF's unit), centered on the origin and facing `+Z` with `+Y`
/// up: domain `+x` along `+X` and domain `+y` up `+Y`. Its UVs cover one
/// period, `v` growing with domain y as texture rows do, so its tangent
/// frame is mirrored (tangent `w` = -1): the bitangent points down, toward
/// the image top, where glTF normal maps put `+Y`. Geometry, normals,
/// tangents and UVs are embedded as a data URI; images are the export's
/// PNG files beside the document.
#[must_use]
pub fn document(exported: &Exported) -> Option<String> {
    let fragment = material(exported)?;
    let [w, h] = exported.provenance.grid.size.map(|s| s * 0.5);
    let positions = [[-w, -h, 0.0], [w, -h, 0.0], [w, h, 0.0], [-w, h, 0.0]];
    let normals = [[0.0_f32, 0.0, 1.0]; 4];
    let tangents = [[1.0_f32, 0.0, 0.0, -1.0]; 4];
    let uvs = [[0.0_f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
    let mut bytes = Vec::new();
    for v in positions.iter().flatten().chain(normals.iter().flatten()) {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in tangents.iter().flatten().chain(uvs.iter().flatten()) {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let vertex_bytes = bytes.len();
    for i in indices {
        bytes.extend_from_slice(&i.to_le_bytes());
    }
    let name = &exported.provenance.subject;
    let mut doc = json!({
        "asset": { "version": "2.0", "generator": "dapple_export" },
        "scene": 0,
        "scenes": [{ "name": name, "nodes": [0] }],
        "nodes": [{ "name": name, "mesh": 0 }],
        "meshes": [{
            "name": name,
            "primitives": [{
                "attributes": { "POSITION": 0, "NORMAL": 1, "TANGENT": 2, "TEXCOORD_0": 3 },
                "indices": 4,
                "material": 0,
            }],
        }],
        "accessors": [
            {
                "bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 4,
                "type": "VEC3", "min": nums(&[-w, -h, 0.0]), "max": nums(&[w, h, 0.0]),
            },
            { "bufferView": 0, "byteOffset": 48, "componentType": 5126, "count": 4, "type": "VEC3" },
            { "bufferView": 0, "byteOffset": 96, "componentType": 5126, "count": 4, "type": "VEC4" },
            { "bufferView": 0, "byteOffset": 160, "componentType": 5126, "count": 4, "type": "VEC2" },
            { "bufferView": 1, "byteOffset": 0, "componentType": 5123, "count": 6, "type": "SCALAR" },
        ],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": vertex_bytes, "target": 34962 },
            {
                "buffer": 0, "byteOffset": vertex_bytes,
                "byteLength": bytes.len() - vertex_bytes, "target": 34963,
            },
        ],
        "buffers": [{
            "byteLength": bytes.len(),
            "uri": format!("data:application/octet-stream;base64,{}", base64(&bytes)),
        }],
        "materials": [fragment.material],
    });
    let object = doc.as_object_mut().expect("a JSON object");
    for (key, list) in [
        ("textures", fragment.textures),
        ("images", fragment.images),
        ("samplers", fragment.samplers),
    ] {
        if !list.is_empty() {
            object.insert(key.to_string(), Value::Array(list));
        }
    }
    if !fragment.extensions_used.is_empty() {
        object.insert("extensionsUsed".into(), json!(fragment.extensions_used));
    }
    let mut text = serde_json::to_string_pretty(&doc).expect("JSON values always serialize");
    text.push('\n');
    Some(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_pads_like_rfc_4648() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"M"), "TQ==");
        assert_eq!(base64(b"Ma"), "TWE=");
        assert_eq!(base64(b"Man"), "TWFu");
        assert_eq!(base64(&[0xfb, 0xff, 0xbf]), "+/+/");
    }

    #[test]
    fn numbers_print_shortest() {
        assert_eq!(num(0.3).to_string(), "0.3");
        assert_eq!(num(1.0).to_string(), "1.0");
        assert_eq!(num(f32::NAN), Value::Null);
    }
}
