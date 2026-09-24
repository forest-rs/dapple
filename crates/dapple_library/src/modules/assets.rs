// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Assets: modules composed of modules.

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use dapple_elements::{Composite, Realized};
use dapple_field::{Domain, Edge};
use dapple_material::module::{
    Args, Bind, Context, Input, Interface, Module, ModuleError, ModuleId, Output, OutputDecl,
    OutputKind, Outputs,
};
use dapple_material::ops::{self, Detail, Layer, Transition};
use dapple_material::{Aux, Channel, Grid, Material, Tiling};
use dapple_raster::typed::Storage;
use dapple_raster::{DistanceTransform, Raster, RasterOp};
use glam::{Vec2, Vec3};

use super::{
    CeramicBody, EdgeWear, Efflorescence, Finish, Grime, Mortar, Moss, Stone, Streaks, Wood, color,
    fail, flag, fraction, integer, mask_map, meters, scalar_map, seed, smoothstep,
};
use crate::glazed_brick::{self, GLAZE, JOINT, Palette, Structure, glazes};

fn material_output() -> Vec<OutputDecl> {
    vec![OutputDecl {
        name: "material",
        kind: OutputKind::Material,
        doc: "the asset's material",
    }]
}

fn take(mut outputs: Outputs) -> Result<Material, ModuleError> {
    outputs
        .take_material("material")
        .ok_or_else(|| fail("a module did not return its material"))
}

fn composite_error<E>(_: E) -> ModuleError {
    fail("the brick composite failed")
}

/// A mask of texels whose center's `y` lies in `[lo, hi)`, softened over
/// one texel.
fn band(grid: Grid, lo: f32, hi: f32) -> Result<Raster, ModuleError> {
    let t = grid.texel.y;
    Ok(grid.raster(
        (0..grid.len())
            .map(|i| {
                let y = grid.center(i).y;
                smoothstep(lo - 0.5 * t, lo + 0.5 * t, y)
                    * smoothstep(hi + 0.5 * t, hi - 0.5 * t, y)
            })
            .collect(),
    )?)
}

/// A glazed brick wall on a 1 m wrapping tile: keyed bricks
/// ([`glazed_brick`]) of a [`CeramicBody`], glazed with a [`Finish`] whose
/// color, coverage and thickness the bricks decide, in [`Mortar`] tooled
/// into their joints, and, when `weathered`, dirtied by [`Grime`],
/// [`Streaks`] and [`Efflorescence`].
#[derive(Copy, Clone, Debug, Default)]
pub struct GlazedBrickWall;

impl Module for GlazedBrickWall {
    fn interface(&self) -> Interface {
        let d = Palette::default();
        Interface {
            id: ModuleId {
                name: "dapple_library.glazed_brick_wall",
                version: 1,
            },
            doc: "Victorian glazed brickwork on a 1 m tile",
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
                fraction("glaze_roughness", 0.035, "the glaze coat's roughness"),
                flag("weathered", true, "whether to add dirt, streaks and salts"),
                fraction("dirt", 0.35, "share of the surface dirtied"),
                fraction("streaks", 0.06, "share of the surface streaked"),
                fraction("salts", 0.05, "share of the surface bloomed with salts"),
                fraction("moss", 0.03, "share of the surface grown with moss"),
                seed(),
            ],
            inputs: vec![],
            outputs: material_output(),
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

        let body = take(cx.instantiate(&CeramicBody, "body", Bind::new())?)?;
        let mortar = take(cx.instantiate(
            &Mortar,
            "mortar",
            Bind::new().input("joint", Input::Map(scalar_map(grid, &joint)?)),
        )?)?;
        // The bricks' shape is geometric detail on the body.
        let bricks = cx.record(ops::apply_detail(
            &body,
            &Detail {
                height: Some(st.height.clone()),
                ..Detail::identity(Layer::Base)
            },
        )?);
        let mut wall = cx.record(ops::select(&mortar, &bricks, &st.cover, Transition::Mask)?);
        wall.set_aux(Aux::Region, Channel::Map(st.owners.clone()))?;

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
        let mut wall = take(
            cx.instantiate(
                &Finish,
                "glaze",
                Bind::new()
                    .material("base", wall)
                    .input("coverage", Input::Map(mask_map(grid, &glaze_cover)?))
                    .input("thickness", Input::Map(scalar_map(grid, &st.thickness)?))
                    .input("color", Input::Map(color_map))
                    .scalar("opacity", 0.94)
                    .scalar("full_thickness", 0.0003)
                    .scalar("tint", 0.15)
                    .scalar("roughness", args.scalar("glaze_roughness"))
                    .scalar("roughness_variation", 0.03)
                    .scalar("peel", 0.000006)
                    .scalar("peel_size", 0.004)
                    .scalar("leveling", 0.0015)
                    .integer("surface", GLAZE),
            )?,
        )?;
        // Glaze worn off the sharpest arrises, down to the body.
        let mut worn = bricks;
        worn.set_aux(Aux::Region, Channel::Map(st.owners.clone()))?;
        wall = take(
            cx.instantiate(
                &EdgeWear,
                "wear",
                Bind::new()
                    .material("base", wall)
                    .material("worn", worn)
                    .scalar("radius", 0.002)
                    .scalar("coverage", 0.006)
                    .integer("surface", GLAZE),
            )?,
        )?;
        if args.flag("weathered") {
            wall = weather(
                cx,
                wall,
                &Weathering {
                    dirt: args.scalar("dirt"),
                    streaks: args.scalar("streaks"),
                    salts: args.scalar("salts"),
                    moss: args.scalar("moss"),
                },
                None,
            )?;
        }
        Ok(Outputs::new().with("material", Output::Material(wall)))
    }
}

/// How much of each weathering to apply.
struct Weathering {
    dirt: f32,
    streaks: f32,
    salts: f32,
    moss: f32,
}

/// Dirt, streaks, salts and moss over `m`, heaviest low on the face.
fn weather(
    cx: &mut Context<'_>,
    m: Material,
    w: &Weathering,
    sources: Option<&Raster>,
) -> Result<Material, ModuleError> {
    let grid = cx.grid();
    let m = take(cx.instantiate(
        &Grime,
        "dirt",
        Bind::new().material("base", m).scalar("coverage", w.dirt),
    )?)?;
    let mut bind = Bind::new()
        .material("base", m)
        .scalar("coverage", w.streaks);
    if let Some(s) = sources {
        bind = bind
            .input("sources", Input::Map(mask_map(grid, s)?))
            .scalar("length", 0.35)
            .scalar("strength", 0.85);
    }
    let m = take(cx.instantiate(&Streaks, "streaks", bind)?)?;
    let m = take(
        cx.instantiate(
            &Efflorescence,
            "salts",
            Bind::new()
                .material("base", m)
                .scalar("coverage", w.salts)
                .scalar("rise", 0.35),
        )?,
    )?;
    if w.moss > 0.0 {
        take(
            cx.instantiate(
                &Moss,
                "moss",
                Bind::new()
                    .material("base", m)
                    .scalar("coverage", w.moss)
                    .scalar("damp", 0.25),
            )?,
        )
    } else {
        Ok(m)
    }
}

/// A stone sill over glazed brickwork, on a 1 m wrapping tile: the top
/// `sill` of the tile is [`Stone`] standing `projection` proud with a
/// rounded lower arris, bedded on a [`Mortar`] joint over a
/// [`GlazedBrickWall`]; weathering runs streaks down from under the sill.
#[derive(Copy, Clone, Debug, Default)]
pub struct StoneSill;

impl Module for StoneSill {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId {
                name: "dapple_library.stone_sill",
                version: 1,
            },
            doc: "a stone sill bedded over glazed brickwork",
            params: vec![
                meters("sill", [0.05, 0.4], 0.15, "the sill's height on the face"),
                meters(
                    "projection",
                    [0.0, 0.03],
                    0.012,
                    "how far the sill stands proud",
                ),
                color("field", glazes::GREEN, "the wall's field glaze"),
                color("dado", glazes::OXBLOOD, "the wall's dado glaze"),
                fraction("dirt", 0.3, "share of the surface dirtied"),
                fraction("streaks", 0.22, "share of the surface streaked"),
                fraction("salts", 0.06, "share of the surface bloomed with salts"),
                fraction("moss", 0.03, "share of the surface grown with moss"),
                seed(),
            ],
            inputs: vec![],
            outputs: material_output(),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let wall = take(
            cx.instantiate(
                &GlazedBrickWall,
                "wall",
                Bind::new()
                    .color("field", args.color("field"))
                    .color("band", glazes::CREAM)
                    .color("dado", args.color("dado"))
                    .integer("dado_courses", 3)
                    .flag("weathered", false),
            )?,
        )?;
        let stone = take(cx.instantiate(&Stone, "sill", Bind::new())?)?;
        let bed = take(cx.instantiate(&Mortar, "bed", Bind::new().scalar("recess", 0.001))?)?;
        #[expect(clippy::cast_precision_loss, reason = "grid sizes are small")]
        let top = grid.origin.y + grid.texel.y * grid.height as f32;
        let bottom = top - args.scalar("sill");
        let sill_mask = band(grid, bottom, top + grid.texel.y)?;
        let bed_mask = band(grid, bottom - JOINT, bottom)?;
        // The sill's face: proud, rounding down over its lower 8 mm.
        let projection = args.scalar("projection");
        let lift = grid.raster(
            (0..grid.len())
                .map(|i| projection * smoothstep(bottom, bottom + 0.008, grid.center(i).y))
                .collect(),
        )?;
        let raised = cx.record(ops::apply_detail(
            &stone,
            &Detail {
                height: Some(lift),
                ..Detail::identity(Layer::Base)
            },
        )?);
        let bedded = cx.record(ops::select(&wall, &bed, &bed_mask, Transition::Mask)?);
        let mut m = cx.record(ops::select(&bedded, &raised, &sill_mask, Transition::Mask)?);
        if let Some(r) = wall.aux(Aux::Region) {
            m.set_aux(Aux::Region, r.clone())?;
        }
        // A sill sits at the top of the wall: the asset tiles along it.
        m.set_tiling(m.tiling().and(Tiling::X));
        // Rain runs off the sill and down the wall from just under it.
        let drip = band(grid, bottom - JOINT - 0.004, bottom - JOINT)?;
        let m = weather(
            cx,
            m,
            &Weathering {
                dirt: args.scalar("dirt"),
                streaks: args.scalar("streaks"),
                salts: args.scalar("salts"),
                moss: args.scalar("moss"),
            },
            Some(&drip),
        )?;
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}

/// An oak board under an amber varnish, lightly dirtied: [`Wood`],
/// [`Finish`] and [`Grime`].
#[derive(Copy, Clone, Debug, Default)]
pub struct VarnishedBoard;

impl Module for VarnishedBoard {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId {
                name: "dapple_library.varnished_board",
                version: 1,
            },
            doc: "varnished oak",
            params: vec![
                color("varnish", Vec3::new(0.78, 0.52, 0.25), "the varnish's tint"),
                fraction("dirt", 0.08, "share of the surface dirtied"),
                seed(),
            ],
            inputs: vec![],
            outputs: material_output(),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let wood = take(cx.instantiate(&Wood, "oak", Bind::new())?)?;
        let varnished = take(
            cx.instantiate(
                &Finish,
                "varnish",
                Bind::new()
                    .material("base", wood)
                    .color("color", args.color("varnish"))
                    .scalar("tint", 0.55)
                    .scalar("roughness", 0.1)
                    .scalar("thickness", 0.00006)
                    .scalar("peel", 0.000004)
                    .scalar("peel_size", 0.006),
            )?,
        )?;
        let m = take(
            cx.instantiate(
                &Grime,
                "dirt",
                Bind::new()
                    .material("base", varnished)
                    .scalar("coverage", args.scalar("dirt"))
                    .scalar("strength", 0.5),
            )?,
        )?;
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}

/// A door threshold: a limestone step below an oiled oak tread, split at
/// the middle of the grid: [`Stone`], [`Wood`] and an oil [`Finish`] on
/// the wood alone.
#[derive(Copy, Clone, Debug, Default)]
pub struct Threshold;

impl Module for Threshold {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId {
                name: "dapple_library.threshold",
                version: 1,
            },
            doc: "a limestone step and an oiled oak tread",
            params: vec![seed()],
            inputs: vec![],
            outputs: material_output(),
        }
    }

    fn build(&self, cx: &mut Context<'_>, _args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let stone = take(
            cx.instantiate(
                &Stone,
                "step",
                Bind::new()
                    .color("light", Vec3::new(0.52, 0.50, 0.44))
                    .color("dark", Vec3::new(0.36, 0.34, 0.30))
                    .scalar("bedding", 0.05),
            )?,
        )?;
        let wood = take(cx.instantiate(&Wood, "tread", Bind::new())?)?;
        #[expect(clippy::cast_precision_loss, reason = "grid sizes are small")]
        let (middle, top) = (
            grid.origin.y + 0.5 * grid.texel.y * grid.height as f32,
            grid.origin.y + grid.texel.y * (grid.height + 1) as f32,
        );
        let tread = band(grid, middle, top)?;
        let joined = cx.record(ops::select(&stone, &wood, &tread, Transition::Mask)?);
        let m = take(
            cx.instantiate(
                &Finish,
                "oil",
                Bind::new()
                    .material("base", joined)
                    .input("coverage", Input::Map(mask_map(grid, &tread)?))
                    .scalar("tint", 0.3)
                    .scalar("roughness", 0.3)
                    .scalar("thickness", 0.00002)
                    .scalar("peel", 0.0),
            )?,
        )?;
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}
