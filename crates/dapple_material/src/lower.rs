// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Lowering a material to `dapple_encode`'s maps for packing.

use alloc::vec::Vec;

use dapple_encode::{Image, MaterialMaps};
use dapple_raster::{HeightToNormal, RasterOp};
use openpbr::Param;

use crate::material::{Aux, ChannelId, Material, MaterialError, param_default};

/// What [`maps`] could not carry.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Lowering {
    /// Parameters that differ from the specification's defaults (as a
    /// constant or a map) but have no map in [`MaterialMaps`]; their values
    /// are lost unless the consumer sets them another way.
    pub dropped: Vec<Param>,
    /// Whether the normal map was derived from [`Aux::Height`] because no
    /// `geometry_normal` was bound.
    pub normal_from_height: bool,
}

/// The material's maps for `dapple_encode::pack`: every channel
/// [`MaterialMaps`] has a field for, as an image on the material's grid
/// (constants that differ from the specification's defaults included),
/// with the normal map derived from the height when no `geometry_normal`
/// is bound. [`Aux::Height`] stays with the material: consumers that
/// displace read it directly.
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
        occlusion: image(material, ChannelId::Aux(Aux::Occlusion))?,
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
    if maps.normal.is_none() && material.aux(Aux::Height).is_some() {
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
        Param::BaseColor => maps.base_color = image(m, c)?,
        Param::BaseMetalness => maps.base_metalness = image(m, c)?,
        Param::SpecularRoughness => maps.specular_roughness = image(m, c)?,
        Param::TransmissionWeight => maps.transmission_weight = image(m, c)?,
        Param::SubsurfaceWeight => maps.subsurface_weight = image(m, c)?,
        Param::SubsurfaceColor => maps.subsurface_color = image(m, c)?,
        Param::CoatWeight => maps.coat_weight = image(m, c)?,
        Param::CoatColor => maps.coat_color = image(m, c)?,
        Param::CoatRoughness => maps.coat_roughness = image(m, c)?,
        Param::GeometryOpacity => maps.opacity = image(m, c)?,
        Param::GeometryNormal => maps.normal = image(m, c)?,
        _ => return Ok(false),
    }
    Ok(true)
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
    use crate::{Channel, Grid};

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
    fn maps_and_constants_become_images_on_the_grid() {
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
        assert_eq!((color.width(), color.height(), color.channels()), (N, N, 3));
        assert_eq!(color.edge(), Edge::Wrap);
        assert_eq!(color.texel(1, 2), [ramp(9), 0.5, 0.25], "row-major");
        // A constant is expanded to a full image.
        let roughness = maps.specular_roughness.unwrap();
        assert_eq!(roughness.channels(), 1);
        assert!(roughness.values().iter().all(|&r| r == 0.6));
        assert_eq!(maps.coat_weight.unwrap().texel(3, 3), [ramp(15)]);
        assert!(maps.occlusion.unwrap().values().iter().all(|&o| o == 0.75));
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
    }

    #[test]
    fn parameters_without_a_map_are_dropped_unless_at_their_defaults() {
        let mut m = Material::new(grid());
        // At the specification's default, nothing is lost.
        m.set_param(Param::SpecularIor, scalar(1.5)).unwrap();
        // Away from it, or as a map, a parameter without a field is lost.
        m.set_param(Param::FuzzWeight, scalar(1.0)).unwrap();
        m.set_param(
            Param::FuzzRoughness,
            map(PortType::Scalar, |_| Value::Scalar(0.7)),
        )
        .unwrap();
        m.set_param(Param::SpecularRoughnessAnisotropy, scalar(0.4))
            .unwrap();
        m.set_param(Param::GeometryThinWalled, scalar(1.0)).unwrap();
        let (maps, lowering) = maps(&m).unwrap();
        assert_eq!(
            lowering.dropped,
            [
                Param::SpecularRoughnessAnisotropy,
                Param::FuzzWeight,
                Param::FuzzRoughness,
                Param::GeometryThinWalled,
            ],
            "in specification order; a map is dropped even if it holds the default"
        );
        assert!(maps.specular_roughness_anisotropy.is_none());
        assert!(maps.anisotropy_direction.is_none());
    }
}
