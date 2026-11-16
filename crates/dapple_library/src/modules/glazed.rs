// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Brick geometry and glaze maps, independent of material composition.

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use dapple_elements::{Composite, Realized};
use dapple_field::{Domain, Edge, PortType, Primaries};
use dapple_material::Channel;
use dapple_material::module::{
    Args, Context, Interface, Module, ModuleError, ModuleId, Output, OutputDecl, OutputKind,
    Outputs,
};
use dapple_raster::typed::Storage;
use dapple_raster::{DistanceTransform, RasterOp};
use glam::{Vec2, Vec3};

use super::{color, fail, fraction, integer, mask_map, scalar_map};
use crate::glazed_brick::{self, Palette, Structure};

fn composite_error<E>(_: E) -> ModuleError {
    fail("the brick composite failed")
}

/// Keyed glazed bricks on a 1 m wrapping tile, at the composite's origin.
///
/// Owns the brick layout, geometric detail, owner labels and glaze maps;
/// material bodies, finishing and weathering belong to its caller.
/// `color` includes crazing darkening and `coverage` includes brick cover.
/// The layout uses fixed element keys, independent of the instance seed.
#[derive(Copy, Clone, Debug, Default)]
pub struct GlazedBricks;

impl Module for GlazedBricks {
    fn interface(&self) -> Interface {
        let d = Palette::default();
        Interface {
            id: ModuleId::new("dapple_library.glazed_bricks", 1),
            doc: "glazed brick geometry and glaze maps on a 1 m wrapping tile".into(),
            params: vec![
                color("field", d.field, "the field's glaze, above the band"),
                color("band", d.band, "the band's glaze"),
                color("dado", d.dado, "the dado's glaze, at the foot"),
                integer(
                    "dado_courses",
                    [0, glazed_brick::BOND[1]],
                    d.dado_courses,
                    "courses of dado",
                ),
                integer(
                    "band_courses",
                    [0, glazed_brick::BOND[1]],
                    d.band_courses,
                    "courses of band",
                ),
                fraction("variation", d.variation, "brick-to-brick glaze variation"),
                fraction("battered", d.battered, "how chipped the bricks are"),
            ],
            inputs: vec![],
            outputs: [
                ("cover", PortType::Mask, "the bricks' share of each texel"),
                (
                    "height",
                    PortType::Scalar,
                    "brick height above the datum, in meters",
                ),
                ("owners", PortType::Id, "dense brick labels, 0 for mortar"),
                (
                    "color",
                    PortType::Color(Primaries::Rec709),
                    "glaze color darkened in crazing cracks",
                ),
                (
                    "coverage",
                    PortType::Mask,
                    "surviving glaze multiplied by brick cover",
                ),
                ("thickness", PortType::Scalar, "glaze thickness in meters"),
                (
                    "joint",
                    PortType::Scalar,
                    "distance to the nearest brick, in meters",
                ),
            ]
            .into_iter()
            .map(|(name, port, doc)| OutputDecl {
                name: name.into(),
                kind: OutputKind::Map(port),
                doc: doc.into(),
            })
            .collect(),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        #[expect(clippy::cast_precision_loss, reason = "grid sizes are small")]
        let extent = grid.texel * Vec2::new(grid.width as f32, grid.height as f32);
        if grid.edge != Edge::Wrap || (extent - Vec2::ONE).abs().max_element() > 1e-4 {
            return Err(fail("glazed brick walls need a 1 m wrapping grid"));
        }
        let palette = Palette {
            field: args.color("field"),
            band: args.color("band"),
            dado: args.color("dado"),
            dado_courses: args.integer("dado_courses"),
            band_courses: args.integer("band_courses"),
            variation: args.scalar("variation"),
            battered: args.scalar("battered"),
        };
        let set = glazed_brick::layout(&palette).map_err(composite_error)?;
        let program = Arc::new(glazed_brick::program()?);
        let instance = glazed_brick::instance(program)?;
        let realized = Realized::composite(&Composite {
            set: &set,
            instance: &instance,
            background: &glazed_brick::BACKGROUND,
            domain: Domain::periodic(1, 1).expect("a unit period"),
            width: grid.width,
            height: grid.height,
            tile_size: 64,
        })
        .map_err(composite_error)?;
        let st = Structure::of(&realized).map_err(composite_error)?;
        if !grid.holds_raster(&st.cover) {
            return Err(fail("the composite is not on the context's grid"));
        }
        // Distance to the nearest brick, for tooling the joints.
        let joint = DistanceTransform { threshold: 0.5 }.apply(&st.cover)?;

        // The glaze: each brick's color, darkened in crazing cracks, laid
        // where it survives, as thick as the bricks say.
        let Storage::F32x3(colors) = st.glaze_color.storage() else {
            return Err(fail("glaze colors"));
        };
        let colors: Vec<Vec3> = colors
            .values()
            .iter()
            .zip(st.craze.values())
            .map(|(c, k)| Vec3::from_array(*c) * (1.0 - 0.35 * k))
            .collect();
        let glaze_cover = grid.raster(
            st.glazed
                .values()
                .iter()
                .zip(st.cover.values())
                .map(|(g, c)| g * c)
                .collect(),
        )?;
        let color_map = match super::color_map(grid, &colors)? {
            Channel::Map(m) => m,
            Channel::Constant(_) => unreachable!("a map"),
        };
        Ok(Outputs::new()
            .with("cover", Output::Map(mask_map(grid, &st.cover)?))
            .with("height", Output::Map(scalar_map(grid, &st.height)?))
            .with("owners", Output::Map(st.owners))
            .with("color", Output::Map(color_map))
            .with("coverage", Output::Map(mask_map(grid, &glaze_cover)?))
            .with("thickness", Output::Map(scalar_map(grid, &st.thickness)?))
            .with("joint", Output::Map(scalar_map(grid, &joint)?)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dapple_material::Grid;
    use dapple_material::module::Bind;
    use dapple_material::resource::NoResources;

    #[test]
    fn rejects_grids_the_fixed_composite_cannot_represent() {
        let unit = Grid {
            width: 32,
            height: 32,
            origin: Vec2::ZERO,
            texel: Vec2::splat(1.0 / 32.0),
            edge: Edge::Wrap,
        };
        for grid in [
            Grid {
                edge: Edge::Clamp,
                ..unit
            },
            Grid {
                texel: Vec2::splat(0.0625),
                ..unit
            },
            Grid {
                origin: Vec2::ONE,
                ..unit
            },
        ] {
            let mut cx = Context::new(grid, &NoResources);
            assert!(
                cx.instantiate(&GlazedBricks, "bricks", Bind::new())
                    .is_err()
            );
        }
    }
}
