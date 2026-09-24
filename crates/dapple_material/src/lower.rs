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
/// # Errors
///
/// Never for a valid material.
pub fn maps(material: &Material) -> Result<(MaterialMaps, Lowering), MaterialError> {
    let grid = material.grid();
    let mut lowering = Lowering::default();
    let image = |c: ChannelId, channels: usize| -> Result<Option<Image>, MaterialError> {
        if material.channel(c).is_none() {
            return Ok(None);
        }
        let mut values = Vec::with_capacity(grid.len() * channels);
        for i in 0..grid.len() {
            let v = material.value(c, i);
            for k in 0..channels {
                values.push(v.component(k).unwrap_or(0.0));
            }
        }
        Ok(Some(
            Image::new(grid.width, grid.height, channels, grid.edge, values)
                .map_err(|_| MaterialError::GridMismatch)?,
        ))
    };
    let p = ChannelId::Param;
    let mut maps = MaterialMaps {
        base_color: image(p(Param::BaseColor), 3)?,
        opacity: image(p(Param::GeometryOpacity), 1)?,
        normal: image(p(Param::GeometryNormal), 3)?,
        specular_roughness: image(p(Param::SpecularRoughness), 1)?,
        base_metalness: image(p(Param::BaseMetalness), 1)?,
        occlusion: image(ChannelId::Aux(Aux::Occlusion), 1)?,
        subsurface_weight: image(p(Param::SubsurfaceWeight), 1)?,
        subsurface_color: image(p(Param::SubsurfaceColor), 3)?,
        transmission_weight: image(p(Param::TransmissionWeight), 1)?,
        coat_weight: image(p(Param::CoatWeight), 1)?,
        coat_roughness: image(p(Param::CoatRoughness), 1)?,
        coat_color: image(p(Param::CoatColor), 3)?,
        ..MaterialMaps::default()
    };
    if maps.normal.is_none() && material.aux(Aux::Height).is_some() {
        let normals = HeightToNormal { scale: 1.0 }.apply(&material.height()?)?;
        maps.normal = Some(Image::from(&normals));
        lowering.normal_from_height = true;
    }
    let carried = [
        Param::BaseColor,
        Param::GeometryOpacity,
        Param::GeometryNormal,
        Param::SpecularRoughness,
        Param::BaseMetalness,
        Param::SubsurfaceWeight,
        Param::SubsurfaceColor,
        Param::TransmissionWeight,
        Param::CoatWeight,
        Param::CoatRoughness,
        Param::CoatColor,
    ];
    for param in Param::ALL {
        if carried.contains(&param) {
            continue;
        }
        let c = p(param);
        let differs = match material.channel(c) {
            None => false,
            Some(ch) => ch.constant() != Some(param_default(param)),
        };
        if differs {
            lowering.dropped.push(param);
        }
    }
    Ok((maps, lowering))
}
