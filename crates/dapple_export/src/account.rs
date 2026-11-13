// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! One account of every parameter, from lowering's and packing's reports.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use dapple_encode::{Bundle, Input, MaterialMaps, Placement, Profile};
use dapple_material::lower::Lowering;
use dapple_material::{Aux, ChannelId, Material, Param};

use crate::manifest::{Encoding, Held, Location, Parameter};

/// A parameter of the account: an OpenPBR parameter or an auxiliary
/// channel that textures carry.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum Subject {
    Param(Param),
    Aux(Aux),
}

impl Subject {
    /// Every subject, in account order.
    pub(crate) fn all() -> impl Iterator<Item = Self> {
        Param::ALL
            .iter()
            .map(|&p| Self::Param(p))
            .chain([Self::Aux(Aux::Occlusion), Self::Aux(Aux::Height)])
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Param(p) => p.identifier(),
            Self::Aux(Aux::Occlusion) => "occlusion",
            Self::Aux(_) => "height",
        }
    }

    fn channel(self) -> ChannelId {
        match self {
            Self::Param(p) => ChannelId::Param(p),
            Self::Aux(a) => ChannelId::Aux(a),
        }
    }

    /// The [`MaterialMaps`] field that carries it, if any.
    fn field(self) -> Option<&'static str> {
        Some(match self {
            Self::Param(p) => match p {
                Param::GeometryOpacity => "opacity",
                Param::GeometryNormal => "normal",
                Param::GeometryTangent => "anisotropy_direction",
                Param::GeometryCoatNormal => "coat_normal",
                Param::GeometryThinWalled => "thin_walled",
                Param::BaseColor
                | Param::BaseMetalness
                | Param::SpecularWeight
                | Param::SpecularColor
                | Param::SpecularRoughness
                | Param::SpecularRoughnessAnisotropy
                | Param::SpecularIor
                | Param::TransmissionWeight
                | Param::SubsurfaceWeight
                | Param::SubsurfaceColor
                | Param::CoatWeight
                | Param::CoatColor
                | Param::CoatRoughness
                | Param::FuzzWeight
                | Param::FuzzColor
                | Param::FuzzRoughness
                | Param::EmissionLuminance
                | Param::EmissionColor => p.identifier(),
                _ => return None,
            },
            Self::Aux(Aux::Occlusion) => "occlusion",
            Self::Aux(Aux::Height) => "height",
            Self::Aux(_) => return None,
        })
    }
}

/// Parameters [`MaterialMaps`] has no field for whose value glTF fixes,
/// that value, and why it matters. They matter only on a coated material.
const GLTF_COAT_FIXED: [(Param, f32, &str); 2] = [
    (
        Param::CoatIor,
        1.5,
        "glTF's clearcoat has an index of refraction of 1.5",
    ),
    (
        Param::CoatDarkening,
        0.0,
        "glTF's clearcoat does not darken the base below it",
    ),
];

pub(crate) const fn profile_name(profile: Profile) -> &'static str {
    match profile {
        Profile::Gltf => "gltf",
        Profile::Lightweald => "lightweald",
        Profile::Raw => "raw",
    }
}

/// The material's value of `subject` when it is the same everywhere, one
/// component per channel.
fn constant(material: &Material, subject: Subject) -> Option<Vec<f32>> {
    let c = subject.channel();
    let value = material.constant(c)?;
    Some(
        (0..c.port().components())
            .map(|k| value.component(k).unwrap_or(0.0))
            .collect(),
    )
}

/// The mean of `subject` over the material's texels, one component per
/// channel.
fn mean(material: &Material, subject: Subject) -> Vec<f32> {
    let c = subject.channel();
    let n = material.grid().len();
    let mut sums = alloc::vec![0.0_f64; c.port().components()];
    for i in 0..n {
        let value = material.value(c, i);
        for (k, sum) in sums.iter_mut().enumerate() {
            *sum += f64::from(value.component(k).unwrap_or(0.0));
        }
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "a mean of f32 values, over at most 2^30 texels"
    )]
    sums.iter().map(|&s| (s / n as f64) as f32).collect()
}

/// The specification's default of `subject`.
fn default(subject: Subject) -> Held {
    let c = subject.channel();
    let value = c.default_value();
    Held::Default {
        value: (0..c.port().components())
            .map(|k| value.component(k).unwrap_or(0.0))
            .collect(),
    }
}

/// Whether a weight is given and not the constant 0.
fn active(weight: &Option<Input<1>>) -> bool {
    match weight {
        None => false,
        Some(Input::Constant([w])) => *w != 0.0,
        Some(Input::Map(_)) => true,
    }
}

fn location(bundle: &Bundle, placement: &Placement) -> Location {
    let shared = bundle.placements.iter().any(|q| {
        q.name != placement.name
            && q.texture == placement.texture
            && q.channels.start < placement.channels.end
            && placement.channels.start < q.channels.end
    });
    let y_up = bundle.profile != Profile::Raw;
    let encoding = match placement.name {
        _ if shared => Encoding::Product,
        "normal" | "coat_normal" if y_up => Encoding::NormalYUp,
        "normal" | "coat_normal" => Encoding::NormalYDown,
        "anisotropy_direction" if y_up => Encoding::AxisYUp,
        "anisotropy_direction" => Encoding::AxisYDown,
        "specular_roughness_anisotropy" if bundle.profile == Profile::Gltf => {
            Encoding::GltfAnisotropyStrength
        }
        _ => Encoding::Value,
    };
    Location {
        texture: placement.texture.to_string(),
        channels: placement.channels.clone(),
        range: placement.range,
        encoding,
    }
}

/// Where `field` is in `bundle`, if it holds it.
fn held_in(bundle: &Bundle, field: &str) -> Option<Held> {
    if let Some(p) = bundle.placement(field) {
        return Some(Held::Texture(location(bundle, p)));
    }
    bundle.constant(field).map(|value| Held::Constant {
        value: value.to_vec(),
    })
}

/// The account of every parameter: what lowering dropped, where the
/// profile's `bundle` holds the rest, what `extra` (the raw profile's
/// bundle of what `bundle` could not carry) holds instead, and what the
/// profile only approximates or fixes at another value.
pub(crate) fn account(
    material: &Material,
    maps: &MaterialMaps,
    lowering: &Lowering,
    bundle: &Bundle,
    extra: Option<&Bundle>,
) -> Vec<Parameter> {
    Subject::all()
        .map(|subject| {
            let mut parameter = carried(material, lowering, bundle, extra, subject);
            if let Subject::Param(p) = subject {
                for approximation in &bundle.report.approximated {
                    if Some(approximation.name) == subject.field() {
                        parameter.notes.push(approximation.why.to_string());
                    }
                }
                if p == Param::GeometryNormal && lowering.normal_from_height {
                    parameter
                        .notes
                        .push(String::from("derived from the height"));
                }
                if bundle.profile == Profile::Gltf && active(&maps.coat_weight) {
                    gltf_coat_fixed(material, p, &mut parameter);
                }
            }
            parameter
        })
        .collect()
}

/// Where the export holds `subject`, before approximations.
fn carried(
    material: &Material,
    lowering: &Lowering,
    bundle: &Bundle,
    extra: Option<&Bundle>,
    subject: Subject,
) -> Parameter {
    let profile = match bundle.profile {
        Profile::Gltf => "glTF",
        Profile::Lightweald => "Lightweald",
        Profile::Raw => "raw",
    };
    let mut notes = Vec::new();
    let dropped = matches!(subject, Subject::Param(p) if lowering.dropped.contains(&p));
    let (held, in_profile) = match subject.field() {
        _ if dropped => match constant(material, subject) {
            Some(value) => {
                notes.push(format!(
                    "dapple_encode's maps have no field for it, so the {profile} material does \
                     not carry it; the manifest keeps its value"
                ));
                (Held::Constant { value }, false)
            }
            None => {
                notes.push(String::from(
                    "dapple_encode's maps have no field for it: its map is lost",
                ));
                (
                    Held::Lost {
                        mean: mean(material, subject),
                    },
                    false,
                )
            }
        },
        Some(field) => {
            // A constant is stated as the material has it: packing's form
            // of a tangent is the doubled angle of its direction.
            let held_in = |bundle: &Bundle| {
                held_in(bundle, field).map(|held| match held {
                    Held::Constant { value } => Held::Constant {
                        value: constant(material, subject).unwrap_or(value),
                    },
                    held => held,
                })
            };
            if let Some(held) = held_in(bundle) {
                (held, true)
            } else if bundle.report.unsupported.contains(&field) {
                let held = extra.and_then(held_in);
                notes.push(match held {
                    Some(Held::Texture(_)) => format!(
                        "the {profile} profile cannot carry it; the raw profile's texture holds it"
                    ),
                    _ => format!(
                        "the {profile} profile cannot carry it; the manifest keeps its value"
                    ),
                });
                let held = held.unwrap_or_else(|| Held::Lost {
                    mean: mean(material, subject),
                });
                (held, false)
            } else {
                // Not given, or a constant at the specification's default
                // that the profile did not need to state.
                (default(subject), true)
            }
        }
        None => (default(subject), true),
    };
    Parameter {
        name: subject.name().to_string(),
        held,
        in_profile,
        notes,
    }
}

/// Accounts for glTF's clearcoat fixing `p` on a coated material: the
/// glTF material carries it only when the material's value is glTF's.
fn gltf_coat_fixed(material: &Material, p: Param, parameter: &mut Parameter) {
    let Some(&(_, fixed, why)) = GLTF_COAT_FIXED.iter().find(|(q, ..)| *q == p) else {
        return;
    };
    // Unbound, this is the specification's default.
    match constant(material, Subject::Param(p)) {
        Some(value) if value == [fixed] => {
            parameter.in_profile = true;
            parameter.notes.clear();
        }
        Some(_) => {
            parameter.in_profile = false;
            parameter.notes.clear();
            parameter
                .notes
                .push(format!("{why}; the manifest keeps the material's value"));
        }
        None => {
            parameter.in_profile = false;
            parameter.notes.push(why.to_string());
        }
    }
}
