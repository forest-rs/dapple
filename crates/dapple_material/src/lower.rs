// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Lowering a material to `dapple_encode`'s maps for packing.

use alloc::vec::Vec;

use dapple_encode::{Image, Input, MaterialMaps};
use dapple_raster::{HeightToNormal, RasterOp};
use openpbr::Param;

use crate::material::{Aux, Channel, ChannelId, Material, MaterialError, param_default};

/// What [`maps`] could not carry.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Lowering {
    /// Parameters that differ from the specification's defaults (as a
    /// constant or a map) but have no field in [`MaterialMaps`]; their
    /// values are lost unless the consumer sets them another way.
    pub dropped: Vec<Param>,
    /// Whether the normal map was derived from [`Aux::Height`] because no
    /// `geometry_normal` was bound.
    pub normal_from_height: bool,
}

/// The material's maps for `dapple_encode::pack`: every bound channel
/// [`MaterialMaps`] has a field for, maps as images on the material's grid
/// and constants as [`Input::Constant`], with the normal map derived from
/// a height map when no `geometry_normal` is bound.
///
/// - A constant normal other than the unperturbed `+Z` (base or coat) is
///   expanded into a map, since normals have no constant form.
/// - `geometry_tangent` becomes [`MaterialMaps::anisotropy_direction`]: the
///   doubled angle of its projection onto the surface, so an axis and its
///   opposite agree. A tangent along the normal has no axis and lowers to
///   `(0, 0)`, which anisotropy reads as isotropic.
/// - [`Aux::Height`] becomes [`MaterialMaps::height`] and
///   [`Aux::Occlusion`] [`MaterialMaps::occlusion`]; the other auxiliary
///   channels are identifiers, not texture data.
///
/// The walk follows the `openpbr` parameter enumeration ([`Param::ALL`]):
/// each parameter's type gives its channel count and its default decides
/// whether leaving it out loses anything, so one match names which
/// parameters have a field.
///
/// # Errors
///
/// Never for a valid material.
pub fn maps(material: &Material) -> Result<(MaterialMaps, Lowering), MaterialError> {
    let mut maps = MaterialMaps {
        occlusion: input(material, ChannelId::Aux(Aux::Occlusion))?,
        height: input(material, ChannelId::Aux(Aux::Height))?,
        ..MaterialMaps::default()
    };
    let mut lowering = Lowering::default();
    for param in Param::ALL {
        if lower_param(material, param, &mut maps)? {
            continue;
        }
        let differs = match material.param(param) {
            None => false,
            Some(ch) => ch.constant() != Some(param_default(param)),
        };
        if differs {
            lowering.dropped.push(param);
        }
    }
    if maps.normal.is_none() && matches!(material.aux(Aux::Height), Some(Channel::Map(_))) {
        let normals = HeightToNormal { scale: 1.0 }.apply(&material.height()?)?;
        maps.normal = Some(Image::from(&normals));
        lowering.normal_from_height = true;
    }
    Ok((maps, lowering))
}

/// Lowers `param` into its field of `maps` and returns true, or returns
/// false when [`MaterialMaps`] has no field for it. Adding a field is one
/// arm here.
fn lower_param(
    material: &Material,
    param: Param,
    maps: &mut MaterialMaps,
) -> Result<bool, MaterialError> {
    let m = material;
    let c = ChannelId::Param(param);
    match param {
        Param::BaseColor => maps.base_color = input(m, c)?,
        Param::BaseMetalness => maps.base_metalness = input(m, c)?,
        Param::SpecularWeight => maps.specular_weight = input(m, c)?,
        Param::SpecularColor => maps.specular_color = input(m, c)?,
        Param::SpecularRoughness => maps.specular_roughness = input(m, c)?,
        Param::SpecularRoughnessAnisotropy => {
            maps.specular_roughness_anisotropy = input(m, c)?;
        }
        Param::SpecularIor => maps.specular_ior = input(m, c)?,
        Param::TransmissionWeight => maps.transmission_weight = input(m, c)?,
        Param::SubsurfaceWeight => maps.subsurface_weight = input(m, c)?,
        Param::SubsurfaceColor => maps.subsurface_color = input(m, c)?,
        Param::CoatWeight => maps.coat_weight = input(m, c)?,
        Param::CoatColor => maps.coat_color = input(m, c)?,
        Param::CoatRoughness => maps.coat_roughness = input(m, c)?,
        Param::FuzzWeight => maps.fuzz_weight = input(m, c)?,
        Param::FuzzColor => maps.fuzz_color = input(m, c)?,
        Param::FuzzRoughness => maps.fuzz_roughness = input(m, c)?,
        Param::EmissionLuminance => maps.emission_luminance = input(m, c)?,
        Param::EmissionColor => maps.emission_color = input(m, c)?,
        Param::GeometryOpacity => maps.opacity = input(m, c)?,
        Param::GeometryNormal => maps.normal = normal(m, c)?,
        Param::GeometryTangent => maps.anisotropy_direction = direction(m, c)?,
        Param::GeometryCoatNormal => maps.coat_normal = normal(m, c)?,
        _ => return Ok(false),
    }
    Ok(true)
}

/// Channel `c` as an input of `N` components, or `None` when it is unbound.
fn input<const N: usize>(
    material: &Material,
    c: ChannelId,
) -> Result<Option<Input<N>>, MaterialError> {
    debug_assert_eq!(c.port().components(), N, "{c} has {N} components");
    Ok(match material.channel(c) {
        None => None,
        Some(Channel::Constant(v)) => Some(Input::Constant(core::array::from_fn(|k| {
            v.component(k).unwrap_or(0.0)
        }))),
        Some(Channel::Map(_)) => image(material, c)?.map(Input::Map),
    })
}

/// Tangent channel `c` as the doubled-angle direction of its projection
/// onto the surface, `(cos 2θ, sin 2θ)`, or `None` when it is unbound.
fn direction(material: &Material, c: ChannelId) -> Result<Option<Input<2>>, MaterialError> {
    let doubled = |v: dapple_field::Value| {
        let x = v.component(0).unwrap_or(0.0);
        let y = v.component(1).unwrap_or(0.0);
        let r2 = x * x + y * y;
        if r2 > 0.0 {
            [(x * x - y * y) / r2, 2.0 * x * y / r2]
        } else {
            [0.0, 0.0]
        }
    };
    Ok(match material.channel(c) {
        None => None,
        Some(Channel::Constant(v)) => Some(Input::Constant(doubled(*v))),
        Some(Channel::Map(_)) => {
            let grid = material.grid();
            let values = (0..grid.len())
                .flat_map(|i| doubled(material.value(c, i)))
                .collect();
            let image = Image::new(grid.width, grid.height, 2, grid.edge, values)
                .map_err(|_| MaterialError::GridMismatch)?;
            Some(Input::Map(image))
        }
    })
}

/// Normal channel `c` as a map, or `None` when it is unbound or the
/// unperturbed normal.
fn normal(material: &Material, c: ChannelId) -> Result<Option<Image>, MaterialError> {
    match material.channel(c) {
        Some(ch) if ch.constant() == Some(c.default_value()) => Ok(None),
        _ => image(material, c),
    }
}

/// Channel `c` as an image on the material's grid with as many channels as
/// its type has components, or `None` when it is unbound.
fn image(material: &Material, c: ChannelId) -> Result<Option<Image>, MaterialError> {
    if material.channel(c).is_none() {
        return Ok(None);
    }
    let grid = material.grid();
    let channels = c.port().components();
    let mut values = Vec::with_capacity(grid.len() * channels);
    for i in 0..grid.len() {
        let v = material.value(c, i);
        for k in 0..channels {
            values.push(v.component(k).unwrap_or(0.0));
        }
    }
    Image::new(grid.width, grid.height, channels, grid.edge, values)
        .map(Some)
        .map_err(|_| MaterialError::GridMismatch)
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use dapple_field::{Edge, NormalFrame, PortType, Primaries, Value};
    use glam::{Vec2, Vec3};

    use super::*;
    use crate::Grid;

    const N: u32 = 4;

    fn grid() -> Grid {
        Grid {
            width: N,
            height: N,
            origin: Vec2::ZERO,
            texel: Vec2::splat(0.25),
            edge: Edge::Wrap,
        }
    }

    fn map(port: PortType, f: impl Fn(usize) -> Value) -> Channel {
        let values: Vec<Value> = (0..grid().len()).map(f).collect();
        Channel::Map(grid().typed(port, values).unwrap())
    }

    fn scalar(v: f32) -> Channel {
        Channel::Constant(Value::Scalar(v))
    }

    #[test]
    fn an_unbound_material_lowers_to_nothing() {
        let (maps, lowering) = maps(&Material::new(grid())).unwrap();
        assert_eq!(maps, MaterialMaps::default());
        assert_eq!(lowering, Lowering::default());
    }

    #[test]
    fn maps_become_images_on_the_grid_and_constants_stay_constant() {
        #[expect(clippy::cast_precision_loss, reason = "small test indices")]
        let ramp = |i: usize| i as f32 / 16.0;
        let mut m = Material::new(grid());
        m.set_param(
            Param::BaseColor,
            map(PortType::Color(Primaries::Rec709), |i| {
                Value::Vector3(Vec3::new(ramp(i), 0.5, 0.25))
            }),
        )
        .unwrap();
        m.set_param(Param::SpecularRoughness, scalar(0.6)).unwrap();
        m.set_param(
            Param::CoatWeight,
            map(PortType::Scalar, |i| Value::Scalar(ramp(i))),
        )
        .unwrap();
        m.set_aux(Aux::Occlusion, scalar(0.75)).unwrap();
        let (maps, lowering) = maps(&m).unwrap();

        let color = maps.base_color.unwrap();
        let color = color.map().unwrap();
        assert_eq!((color.width(), color.height(), color.channels()), (N, N, 3));
        assert_eq!(color.edge(), Edge::Wrap);
        assert_eq!(color.texel(1, 2), [ramp(9), 0.5, 0.25], "row-major");
        // A constant stays a constant.
        assert_eq!(maps.specular_roughness, Some(Input::Constant([0.6])));
        let coat = maps.coat_weight.unwrap();
        assert_eq!(coat.map().unwrap().texel(3, 3), [ramp(15)]);
        assert_eq!(maps.occlusion, Some(Input::Constant([0.75])));
        // Unbound parameters have no image, and nothing else was bound.
        assert!(maps.base_metalness.is_none());
        assert!(maps.coat_color.is_none());
        assert!(maps.normal.is_none());
        assert_eq!(lowering, Lowering::default());
    }

    #[test]
    fn the_normal_comes_from_the_height_only_when_unbound() {
        let mut m = Material::new(grid());
        #[expect(clippy::cast_precision_loss, reason = "small test indices")]
        m.set_aux(
            Aux::Height,
            map(PortType::Scalar, |i| Value::Scalar((i % 4) as f32 * 0.01)),
        )
        .unwrap();
        let (derived, lowering) = maps(&m).unwrap();
        assert!(lowering.normal_from_height);
        let expected = HeightToNormal { scale: 1.0 }
            .apply(&m.height().unwrap())
            .unwrap();
        assert_eq!(derived.normal, Some(Image::from(&expected)));

        let tilted = Vec3::new(0.6, 0.0, 0.8);
        m.set_param(
            Param::GeometryNormal,
            map(PortType::Normal(NormalFrame::Domain), |_| {
                Value::Vector3(tilted)
            }),
        )
        .unwrap();
        let (bound, lowering) = maps(&m).unwrap();
        assert!(!lowering.normal_from_height);
        assert_eq!(bound.normal.unwrap().texel(0, 0), tilted.to_array());

        // A constant normal has no constant form: flat is no map at all, and
        // anything else is expanded.
        let mut flat = Material::new(grid());
        let constant = |v: Vec3| Channel::Constant(Value::Vector3(v));
        flat.set_param(Param::GeometryNormal, constant(Vec3::Z))
            .unwrap();
        assert_eq!(maps(&flat).unwrap().0.normal, None);
        flat.set_param(Param::GeometryNormal, constant(tilted))
            .unwrap();
        let expanded = maps(&flat).unwrap().0.normal.unwrap();
        assert!(expanded.values().chunks(3).all(|n| n == tilted.to_array()));
    }

    #[test]
    fn parameters_without_a_map_are_dropped_unless_at_their_defaults() {
        let mut m = Material::new(grid());
        // At the specification's default, nothing is lost.
        m.set_param(Param::BaseWeight, scalar(1.0)).unwrap();
        // Away from it, or as a map, a parameter without a field is lost.
        m.set_param(Param::CoatIor, scalar(1.4)).unwrap();
        m.set_param(
            Param::ThinFilmWeight,
            map(PortType::Scalar, |_| Value::Scalar(0.0)),
        )
        .unwrap();
        m.set_param(Param::GeometryThinWalled, scalar(1.0)).unwrap();
        let (maps, lowering) = maps(&m).unwrap();
        assert_eq!(
            lowering.dropped,
            [
                Param::CoatIor,
                Param::ThinFilmWeight,
                Param::GeometryThinWalled,
            ],
            "in specification order; a map is dropped even if it holds the default"
        );
        assert_eq!(maps, MaterialMaps::default());
    }

    #[test]
    fn specular_fuzz_emission_coat_normals_and_height_are_carried() {
        let mut m = Material::new(grid());
        let color = |c: [f32; 3]| Channel::Constant(Value::Vector3(Vec3::from_array(c)));
        m.set_param(Param::SpecularWeight, scalar(0.8)).unwrap();
        m.set_param(Param::SpecularColor, color([1.0, 0.9, 0.8]))
            .unwrap();
        m.set_param(Param::SpecularIor, scalar(1.33)).unwrap();
        m.set_param(Param::SpecularRoughnessAnisotropy, scalar(0.4))
            .unwrap();
        m.set_param(Param::FuzzWeight, scalar(1.0)).unwrap();
        m.set_param(Param::FuzzColor, color([0.3, 0.4, 0.1]))
            .unwrap();
        m.set_param(
            Param::FuzzRoughness,
            map(PortType::Scalar, |_| Value::Scalar(0.7)),
        )
        .unwrap();
        m.set_param(Param::EmissionLuminance, scalar(40.0)).unwrap();
        m.set_param(Param::EmissionColor, color([1.0, 0.5, 0.0]))
            .unwrap();
        let tilted = Vec3::new(0.0, 0.6, 0.8);
        m.set_param(
            Param::GeometryCoatNormal,
            map(PortType::Normal(NormalFrame::Domain), |_| {
                Value::Vector3(tilted)
            }),
        )
        .unwrap();
        m.set_aux(Aux::Height, scalar(0.002)).unwrap();
        let (maps, lowering) = maps(&m).unwrap();
        assert_eq!(lowering, Lowering::default(), "nothing lost");

        assert_eq!(maps.specular_weight, Some(Input::Constant([0.8])));
        assert_eq!(maps.specular_color, Some(Input::Constant([1.0, 0.9, 0.8])));
        assert_eq!(maps.specular_ior, Some(Input::Constant([1.33])));
        assert_eq!(
            maps.specular_roughness_anisotropy,
            Some(Input::Constant([0.4]))
        );
        assert_eq!(maps.fuzz_weight, Some(Input::Constant([1.0])));
        assert_eq!(maps.fuzz_color, Some(Input::Constant([0.3, 0.4, 0.1])));
        let fuzz_roughness = maps.fuzz_roughness.unwrap();
        assert_eq!(fuzz_roughness.map().unwrap().texel(2, 1), [0.7]);
        assert_eq!(maps.emission_luminance, Some(Input::Constant([40.0])));
        assert_eq!(maps.emission_color, Some(Input::Constant([1.0, 0.5, 0.0])));
        assert_eq!(maps.coat_normal.unwrap().texel(0, 0), tilted.to_array());
        // A constant height is carried, and derives no normal map: it is
        // flat.
        assert_eq!(maps.height, Some(Input::Constant([0.002])));
        assert_eq!(maps.normal, None);
    }

    #[test]
    fn tangents_lower_to_doubled_angle_directions() {
        let mut m = Material::new(grid());
        let tangent = |v: Vec3| Channel::Constant(Value::Vector3(v));
        let direction = |m: &Material| maps(m).unwrap().0.anisotropy_direction;
        // 45°: the doubled angle is 90°.
        m.set_param(Param::GeometryTangent, tangent(Vec3::new(1.0, 1.0, 0.0)))
            .unwrap();
        assert_eq!(direction(&m), Some(Input::Constant([0.0, 1.0])));
        // An axis and its opposite are one direction; only the projection
        // onto the surface counts.
        m.set_param(Param::GeometryTangent, tangent(Vec3::new(-2.0, 0.0, 5.0)))
            .unwrap();
        assert_eq!(direction(&m), Some(Input::Constant([1.0, 0.0])));
        // Along the normal there is no axis.
        m.set_param(Param::GeometryTangent, tangent(Vec3::Z))
            .unwrap();
        assert_eq!(direction(&m), Some(Input::Constant([0.0, 0.0])));
        // Maps lower texel by texel.
        m.set_param(
            Param::GeometryTangent,
            map(PortType::Vector3, |i| {
                Value::Vector3(if i % 2 == 0 { Vec3::X } else { Vec3::Y })
            }),
        )
        .unwrap();
        let image = direction(&m).unwrap();
        let image = image.map().unwrap();
        assert_eq!(image.channels(), 2);
        assert_eq!(image.texel(0, 0), [1.0, 0.0]);
        assert_eq!(image.texel(1, 0), [-1.0, 0.0]);
    }
}
