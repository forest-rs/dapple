// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Golden digests of packed textures and KTX2 files.
//!
//! A change to any digest means encoded output changed. Update the pinned
//! values only for an intended change, and say why in the commit message.

use alloc::vec::Vec;

use dapple_raster::Edge;

use crate::{Filter, Image, MaterialMaps, PackSettings, Profile, ktx2, pack};

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// A deterministic, bumpy test material on a 16×16 torus.
fn material() -> MaterialMaps {
    let n = 16_u32;
    let wave = |i: u32, k: u32| ((i * k + 3) % 17) as f32 / 16.0;
    let mut color = Vec::new();
    let mut opacity = Vec::new();
    let mut normal = Vec::new();
    let mut rough = Vec::new();
    let mut occlusion = Vec::new();
    for y in 0..n {
        for x in 0..n {
            color.extend_from_slice(&[wave(x, 3), wave(y, 5), wave(x + y, 7)]);
            opacity.push(if (x * 7 + y * 3) % 5 < 2 { 1.0 } else { 0.1 });
            let (dx, dy) = (wave(x, 11) - 0.5, wave(y, 13) - 0.5);
            let len = libm::sqrtf(dx * dx + dy * dy + 1.0);
            normal.extend_from_slice(&[dx / len, dy / len, 1.0 / len]);
            rough.push(0.2 + 0.6 * wave(x ^ y, 9));
            occlusion.push(0.5 + 0.5 * wave(x + 2 * y, 5));
        }
    }
    let image = |c, v| Image::new(n, n, c, Edge::Wrap, v).unwrap();
    MaterialMaps {
        base_color: Some(image(3, color)),
        opacity: Some(image(1, opacity)),
        normal: Some(image(3, normal)),
        specular_roughness: Some(image(1, rough)),
        base_metalness: None,
        occlusion: Some(image(1, occlusion)),
        anisotropy_direction: None,
        specular_roughness_anisotropy: None,
        subsurface_weight: None,
        subsurface_color: None,
        transmission_weight: None,
    }
}

fn digests(profile: Profile, filter: Filter) -> Vec<(&'static str, u64)> {
    let settings = PackSettings {
        filter,
        alpha_cutoff: Some(0.5),
        ..PackSettings::default()
    };
    let bundle = pack(&material(), profile, &settings).unwrap();
    bundle
        .textures
        .iter()
        .map(|t| (t.name, fnv1a(&ktx2::write(t))))
        .collect()
}

#[test]
fn lightweald_box_ktx2_is_pinned() {
    assert_eq!(
        digests(Profile::Lightweald, Filter::Box),
        [
            ("base_color", 13_986_416_310_451_596_289),
            ("normal", 14_446_010_263_683_254_402),
            ("orm", 8_832_060_144_503_095_627),
        ]
    );
}

#[test]
fn gltf_kaiser_ktx2_is_pinned() {
    assert_eq!(
        digests(Profile::Gltf, Filter::Kaiser),
        [
            ("base_color", 16_338_841_297_191_050_071),
            ("normal", 6_860_539_956_111_657_120),
            ("orm", 12_241_470_466_670_387_981),
        ]
    );
}

#[test]
fn encoding_is_deterministic() {
    assert_eq!(
        digests(Profile::Raw, Filter::Kaiser),
        digests(Profile::Raw, Filter::Kaiser)
    );
}
