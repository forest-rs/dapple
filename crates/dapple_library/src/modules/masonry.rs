// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Stone and masonry on unit layouts: ashlar, rubble, knapped flint, Roman
//! brick, marble and plain roof tiles.
//!
//! Every module here reads its units through [`crate::masonry`]'s contract
//! (`units`, `edge` and `local` maps). A host that has a unit layout, such
//! as the construction layer's, binds all three
//! ([`crate::masonry::UnitMaps::from_elements`] rasterizes an element set
//! into them); otherwise the module lays out its own element set and
//! rasterizes it the same way. A unit's color, tooling and wear derive from
//! its identity, never its position, so a unit keeps its look when the
//! layout moves it.

use alloc::vec;
use alloc::vec::Vec;

use dapple_elements::{
    Anchor, Element, ElementError, ElementKey, ElementSet, LayoutId, Outline, Placement,
};
use dapple_field::hash::{hash, unit_f32};
use dapple_field::scoped::name_word;
use dapple_field::{PortType, Value};
use dapple_material::module::{
    Args, Bind, Context, Input, InputDecl, InputKind, Interface, Module, ModuleError, ModuleId,
    Output, OutputDecl, OutputKind, Outputs, ParamDecl,
};
use dapple_material::ops::{self, Transition};
use dapple_material::{Aux, Channel, Grid, Material, Param};
use dapple_raster::Raster;
use glam::{Vec2, Vec3};

use super::{
    color, color_map, fail, fbm, fraction, meters, realize_scalar, scalar_channel, scalar_map,
    seed, smoothstep,
};
use crate::glazed_brick::{BODY, FLINT, MARBLE, STONE};
use crate::masonry::{Units, extent, rasterize};

/// A uniform value in `[0, 1)` for unit `id`, `purpose` and the instance
/// `seed`.
fn unit_random(seed: u64, id: u32, purpose: &str) -> f32 {
    unit_f32(hash(
        seed,
        &[u64::from(id), name_word(0x756e_6974, purpose)],
    ))
}

/// The optional unit-layout inputs every masonry module declares.
fn unit_inputs() -> Vec<InputDecl> {
    let input = |name: &'static str, port, doc: &'static str| InputDecl {
        name: name.into(),
        kind: InputKind::Map(port),
        required: false,
        doc: doc.into(),
    };
    vec![
        input(
            "units",
            PortType::Id,
            "the nearest unit's identity (bind with edge and local)",
        ),
        input(
            "edge",
            PortType::Scalar,
            "signed distance to the unit's outline, meters, negative inside",
        ),
        input(
            "local",
            PortType::Vector2,
            "position in the unit's frame over its half extent",
        ),
    ]
}

fn material_output(doc: &'static str) -> Vec<OutputDecl> {
    vec![OutputDecl {
        name: "material".into(),
        kind: OutputKind::Material,
        doc: doc.into(),
    }]
}

/// The units: the host's, when it binds all three maps, or `own`'s layout
/// over the grid's extent, rasterized `reach` meters around each unit.
fn units_for(
    args: &Args,
    grid: Grid,
    reach: f32,
    own: impl FnOnce(Vec2) -> Result<ElementSet, ElementError>,
) -> Result<Units, ModuleError> {
    match (args.map("units"), args.map("edge"), args.map("local")) {
        (None, None, None) => {
            let set = own(extent(grid)).map_err(|_| fail("the unit layout failed"))?;
            Ok(rasterize(&set, grid, reach))
        }
        (Some(units), Some(edge), Some(local)) => crate::masonry::UnitMaps {
            units: units.clone(),
            edge: edge.clone(),
            local: local.clone(),
        }
        .values()
        .ok_or_else(|| fail("unit maps of the wrong storage")),
        _ => Err(fail("bind units, edge and local together")),
    }
}

/// Coursed units over `size`: rows `course` high (a range: each course
/// draws its height, then all scale to fill the extent), of units `length`
/// long (likewise per course), `joint` apart. With `stagger`, each course
/// starts half its first unit along from the one below (a running bond);
/// otherwise at a random offset (rubble).
struct Coursing {
    layout: &'static str,
    course: [f32; 2],
    length: [f32; 2],
    joint: f32,
    stagger: bool,
    seed: u64,
    /// Rounded units (knapped nodules) rather than squared ones: ellipses
    /// that touch their neighbors at mid-sides, leaving mortar in the
    /// corners.
    rounded: bool,
}

impl Coursing {
    fn elements(&self, size: Vec2) -> Result<ElementSet, ElementError> {
        let layout = LayoutId::named(self.layout);
        let r = |a: u64, b: u64, k: u64| unit_f32(hash(self.seed, &[a, b, k]));
        let draw = |range: [f32; 2], t: f32| range[0] + (range[1] - range[0]) * t;
        let count = |total: f32, range: [f32; 2]| {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "unit counts are small and positive"
            )]
            let n = libm::roundf(total / (0.5 * (range[0] + range[1]))).max(1.0) as u64;
            n
        };
        let rows = count(size.y, self.course);
        let heights: Vec<f32> = (0..rows).map(|c| draw(self.course, r(c, 0, 1))).collect();
        let scale_y = size.y / heights.iter().sum::<f32>();
        let mut elements = Vec::new();
        // Start a little into the first course and unit, so no joint lies
        // on the wrap: a sub-texel joint on the wrap would be a hard step
        // there that interior joints spread over two texels.
        let mut y = 0.37 * heights[0] * scale_y;
        for (c, h) in heights.iter().enumerate() {
            let h = h * scale_y;
            let c64 = c as u64;
            let n = count(size.x, self.length);
            let lengths: Vec<f32> = (0..n).map(|i| draw(self.length, r(c64, i, 2))).collect();
            let scale_x = size.x / lengths.iter().sum::<f32>();
            let mut x = if self.stagger {
                let first = lengths[0] * scale_x;
                if c % 2 == 1 { 0.8 * first } else { 0.3 * first }
            } else {
                r(c64, 0, 3) * size.x
            };
            for (i, l) in lengths.iter().enumerate() {
                let l = l * scale_x;
                let anchor = Anchor([
                    i32::try_from(i).unwrap_or(i32::MAX),
                    i32::try_from(c).unwrap_or(i32::MAX),
                ]);
                elements.push(Element {
                    key: ElementKey::new(layout, anchor, 0),
                    placement: Placement::at(Vec2::new(x + 0.5 * l, y + 0.5 * h)),
                    half_size: if self.rounded {
                        Vec2::new(0.58 * l - 0.5 * self.joint, 0.58 * h - 0.5 * self.joint)
                    } else {
                        Vec2::new(
                            (0.5 * (l - self.joint)).max(1e-4),
                            (0.5 * (h - self.joint)).max(1e-4),
                        )
                    },
                    outline: if self.rounded {
                        Outline::Ellipse
                    } else {
                        Outline::Rectangle
                    },
                    variant: 0,
                    attributes: vec![],
                });
                x += l;
            }
            y += h;
        }
        ElementSet::new(vec![], elements)
    }
}

/// What a unit material is made of, per texel.
struct UnitSurface {
    color: Vec<Vec3>,
    roughness: Vec<f32>,
    height: Vec<f32>,
}

/// The mortar a unit layout is laid in: [`super::Mortar`], tooled into the
/// joints `edge` describes.
fn mortar(
    cx: &mut Context<'_>,
    edge: &[f32],
    color: Vec3,
    recess: f32,
    joint_width: f32,
) -> Result<Material, ModuleError> {
    let grid = cx.grid();
    let joint = grid.raster(edge.iter().map(|e| e.max(0.0)).collect())?;
    cx.instantiate(
        &super::Mortar,
        "mortar",
        Bind::new()
            .input("joint", Input::Map(scalar_map(grid, &joint)?))
            .color("color", color)
            .scalar("recess", recess)
            .scalar("joint_width", joint_width.clamp(0.002, 0.05)),
    )?
    .take_material("material")
    .ok_or_else(|| fail("the mortar did not return its material"))
}

/// Lays `surface` on the units where `edge` is negative (softened over a
/// texel) and `joints` elsewhere, labeling each texel's region with its
/// unit.
fn lay(
    cx: &mut Context<'_>,
    units: &Units,
    edge: &[f32],
    surface: &UnitSurface,
    surface_id: u32,
    joints: &Material,
) -> Result<Material, ModuleError> {
    let grid = cx.grid();
    let t = grid.texel.min_element();
    let mut m = Material::new(grid);
    m.set_param(Param::BaseColor, color_map(grid, &surface.color)?)?;
    m.set_param(
        Param::SpecularRoughness,
        scalar_channel(grid, &surface.roughness)?,
    )?;
    m.set_aux(Aux::Height, scalar_channel(grid, &surface.height)?)?;
    m.set_aux(Aux::Surface, Channel::Constant(Value::Id(surface_id)))?;
    let cover: Raster = grid.raster(
        edge.iter()
            .map(|&e| smoothstep(0.5 * t, -0.5 * t, e))
            .collect(),
    )?;
    let mut laid = cx.record(ops::select(joints, &m, &cover, Transition::Mask)?);
    let regions: Vec<Value> = units
        .id
        .iter()
        .zip(cover.values())
        .map(|(&id, &c)| Value::Id(if c > 0.5 { id } else { 0 }))
        .collect();
    laid.set_aux(
        Aux::Region,
        Channel::Map(grid.typed(PortType::Id, regions)?),
    )?;
    Ok(laid)
}

/// A rounded arris: 0 in the unit's face, falling by `depth` over the
/// `radius` nearest its outline.
fn arris(edge: f32, radius: f32, depth: f32) -> f32 {
    let t = smoothstep(0.0, radius.max(1e-5), -edge);
    -depth * (1.0 - t * (2.0 - t))
}

fn common_params(
    color_default: Vec3,
    color_doc: &'static str,
    roughness: f32,
    mortar_color: Vec3,
) -> Vec<ParamDecl> {
    vec![
        color("color", color_default, color_doc),
        fraction("roughness", roughness, "specular roughness of the units"),
        color("mortar_color", mortar_color, "the mortar's mean color"),
        seed(),
    ]
}

/// Tooled ashlar limestone: coursed blocks of an oolitic freestone with
/// fine joints, each block its own tone and batted with parallel tool
/// marks at its own slant, faint bedding along the courses, and slightly
/// eased arrises.
///
/// Calibrated ([`super::calibration`]): the blocks' mean albedo matches an
/// oolitic limestone's measured reflectance.
#[derive(Copy, Clone, Debug, Default)]
pub struct AshlarLimestone;

impl Module for AshlarLimestone {
    fn interface(&self) -> Interface {
        let mut params = common_params(
            super::calibration::ASHLAR_COLOR,
            "the stone's mean color",
            0.82,
            Vec3::new(0.42, 0.39, 0.33),
        );
        params.extend([
            meters("course", [0.1, 0.6], 0.3, "course height"),
            meters("block", [0.2, 1.2], 0.6, "mean block length"),
            meters("joint", [0.002, 0.02], 0.005, "joint width"),
            fraction("bedding", 0.3, "how much the bedding shows"),
            meters("tooling", [0.0, 0.002], 0.0004, "tool-mark depth"),
        ]);
        Interface {
            id: ModuleId::new("dapple_library.ashlar_limestone", 1),
            doc: "tooled ashlar limestone".into(),
            params,
            inputs: unit_inputs(),
            outputs: material_output("the wall, surface identity STONE on the blocks"),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let joint = args.scalar("joint");
        let course = args.scalar("course");
        let block = args.scalar("block");
        let units = units_for(args, grid, 0.03, |size| {
            Coursing {
                layout: "dapple_library.ashlar",
                course: [course, course],
                length: [0.7 * block, 1.3 * block],
                joint,
                stagger: true,
                seed: args.seed("layout"),
                rounded: false,
            }
            .elements(size)
        })?;
        let grain = realize_scalar(grid, |b, d| {
            fbm(b, d, [300.0, 300.0], args.seed("grain"), 2)
        })?;
        let beds = realize_scalar(grid, |b, d| fbm(b, d, [3.0, 40.0], args.seed("beds"), 4))?;
        let mottle = realize_scalar(grid, |b, d| fbm(b, d, [6.0, 6.0], args.seed("mottle"), 4))?;
        let base = args.color("color");
        let (rough, bedding, depth) = (
            args.scalar("roughness"),
            args.scalar("bedding"),
            args.scalar("tooling"),
        );
        let seed = args.seed("units");
        let mut s = UnitSurface {
            color: Vec::with_capacity(grid.len()),
            roughness: Vec::with_capacity(grid.len()),
            height: Vec::with_capacity(grid.len()),
        };
        for i in 0..grid.len() {
            let id = units.id[i];
            let tone = 0.93 + 0.14 * unit_random(seed, id, "tone");
            let g = grain.values()[i];
            let c = base
                * tone
                * (1.0 + 0.06 * g + 0.05 * mottle.values()[i] + 0.12 * bedding * beds.values()[i]);
            // Batted tooling: parallel cuts across the block at its slant.
            let l = units.local[i];
            let slant = 0.25 * (unit_random(seed, id, "slant") - 0.5);
            let cuts = libm::fabsf(fract(l.x * 24.0 + l.y * slant * 6.0) - 0.5) * 2.0;
            let tool = -depth * smoothstep(0.35, 0.0, cuts);
            s.color.push(c * (1.0 + 0.05 * tool / depth.max(1e-6)));
            s.roughness.push((rough + 0.04 * g).clamp(0.0, 1.0));
            s.height
                .push(arris(units.edge[i], 0.004, 0.002) + tool + 0.0002 * g);
        }
        let joints = mortar(cx, &units.edge, args.color("mortar_color"), 0.0015, joint)?;
        let m = lay(cx, &units, &units.edge, &s, STONE, &joints)?;
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}

fn fract(x: f32) -> f32 {
    x - libm::floorf(x)
}

/// Rubble walling: coursed rubble of a crinoidal limestone, stones of
/// many sizes with irregular, heavily rounded outlines standing proud of a
/// wide, recessed lime joint.
///
/// Calibrated ([`super::calibration`]): the stones' mean albedo matches a
/// crinoidal limestone's measured reflectance.
#[derive(Copy, Clone, Debug, Default)]
pub struct RubbleWall;

impl Module for RubbleWall {
    fn interface(&self) -> Interface {
        let mut params = common_params(
            super::calibration::RUBBLE_COLOR,
            "the stones' mean color",
            0.88,
            Vec3::new(0.40, 0.37, 0.31),
        );
        params.extend([
            meters("stone", [0.05, 0.5], 0.22, "mean stone length"),
            meters("joint", [0.005, 0.05], 0.02, "joint width"),
            meters("irregularity", [0.0, 0.08], 0.04, "how ragged outlines are"),
            fraction("variety", 0.5, "how much stones differ in tone"),
        ]);
        Interface {
            id: ModuleId::new("dapple_library.rubble_wall", 1),
            doc: "coursed limestone rubble".into(),
            params,
            inputs: unit_inputs(),
            outputs: material_output("the wall, surface identity STONE on the stones"),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let stone = args.scalar("stone");
        let joint = args.scalar("joint");
        let units = units_for(args, grid, 0.05, |size| {
            Coursing {
                layout: "dapple_library.rubble",
                course: [0.35 * stone, 0.8 * stone],
                length: [0.5 * stone, 1.5 * stone],
                joint,
                stagger: false,
                seed: args.seed("layout"),
                rounded: false,
            }
            .elements(size)
        })?;
        let ragged = realize_scalar(grid, |b, d| fbm(b, d, [14.0, 14.0], args.seed("ragged"), 5))?;
        let face = realize_scalar(grid, |b, d| fbm(b, d, [40.0, 40.0], args.seed("face"), 4))?;
        let amp = args.scalar("irregularity");
        let edge: Vec<f32> = units
            .edge
            .iter()
            .zip(ragged.values())
            .map(|(e, n)| e + amp * n)
            .collect();
        let base = args.color("color");
        let (rough, variety) = (args.scalar("roughness"), args.scalar("variety"));
        let seed = args.seed("units");
        let mut s = UnitSurface {
            color: Vec::with_capacity(grid.len()),
            roughness: Vec::with_capacity(grid.len()),
            height: Vec::with_capacity(grid.len()),
        };
        #[expect(
            clippy::needless_range_loop,
            reason = "several per-texel arrays in step"
        )]
        for i in 0..grid.len() {
            let id = units.id[i];
            let tone = 1.0 + variety * 0.35 * (unit_random(seed, id, "tone") - 0.5);
            let warm = 1.0 + variety * 0.12 * (unit_random(seed, id, "warm") - 0.5);
            let f = face.values()[i];
            let c = base * Vec3::new(tone * warm, tone, tone / warm) * (1.0 + 0.1 * f);
            s.color.push(c);
            s.roughness.push((rough + 0.05 * f).clamp(0.0, 1.0));
            // Pillowed faces: proud of the joint, rounding away over 3 cm.
            let lift = 0.012 * smoothstep(0.0, 0.03, -edge[i]);
            s.height.push(lift - 0.006 + 0.003 * f);
        }
        let joints = mortar(cx, &edge, args.color("mortar_color"), 0.008, joint)?;
        let m = lay(cx, &units, &edge, &s, STONE, &joints)?;
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}

/// Knapped flint walling: flint nodules split to show their glassy,
/// near-black faces, squared roughly and laid in rough courses in lime
/// mortar, the joints kept narrow. Each face is its own shade from
/// blue-black to grey-brown, clouded with milky translucent patches and
/// rippled around its point of percussion; some keep a white, chalky rim
/// of cortex where the knapping left the nodule's skin.
///
/// Calibrated ([`super::calibration`]): the faces stay under the upper
/// bound on dark chert's reflectance.
#[derive(Copy, Clone, Debug, Default)]
pub struct FlintWall;

impl Module for FlintWall {
    fn interface(&self) -> Interface {
        let mut params = common_params(
            super::calibration::FLINT_COLOR,
            "the knapped faces' mean color",
            0.25,
            Vec3::new(0.40, 0.38, 0.33),
        );
        params.extend([
            meters("course", [0.04, 0.2], 0.08, "mean course height"),
            meters("length", [0.04, 0.25], 0.09, "mean flint length"),
            meters("joint", [0.004, 0.04], 0.008, "joint width"),
            fraction("cortex", 0.3, "share of flints showing their cortex"),
        ]);
        Interface {
            id: ModuleId::new("dapple_library.flint_wall", 1),
            doc: "knapped, coursed flint in lime mortar".into(),
            params,
            inputs: unit_inputs(),
            outputs: material_output("the wall, surface identity FLINT on the flints"),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let (course, length, joint) = (
            args.scalar("course"),
            args.scalar("length"),
            args.scalar("joint"),
        );
        let units = units_for(args, grid, 0.03, |extent| {
            Coursing {
                layout: "dapple_library.flint",
                course: [0.75 * course, 1.25 * course],
                length: [0.55 * length, 1.45 * length],
                joint,
                stagger: false,
                seed: args.seed("layout"),
                rounded: true,
            }
            .elements(extent)
        })?;
        let ragged = realize_scalar(grid, |b, d| fbm(b, d, [20.0, 20.0], args.seed("ragged"), 4))?;
        let cloud = realize_scalar(grid, |b, d| fbm(b, d, [45.0, 45.0], args.seed("cloud"), 4))?;
        let rind_n = realize_scalar(grid, |b, d| fbm(b, d, [70.0, 70.0], args.seed("rind"), 3))?;
        // Knapped nodules are lumpy and their corners broken: ragged
        // outlines, rounded where corners would be.
        let edge: Vec<f32> = units
            .edge
            .iter()
            .zip(ragged.values())
            .map(|(e, n)| e + 0.01 * n)
            .collect();
        let (base, rough, cortex) = (
            args.color("color"),
            args.scalar("roughness"),
            args.scalar("cortex"),
        );
        let white = Vec3::new(0.55, 0.53, 0.47);
        let milky = Vec3::new(0.2, 0.21, 0.22);
        let seed = args.seed("units");
        let mut s = UnitSurface {
            color: Vec::with_capacity(grid.len()),
            roughness: Vec::with_capacity(grid.len()),
            height: Vec::with_capacity(grid.len()),
        };
        #[expect(
            clippy::needless_range_loop,
            reason = "several per-texel arrays in step"
        )]
        for i in 0..grid.len() {
            let id = units.id[i];
            // Flints range from blue-black to grey-brown.
            let hue = unit_random(seed, id, "hue");
            let tint = Vec3::new(0.9 + 0.3 * hue, 1.0, 1.12 - 0.3 * hue);
            let tone = 0.7 + 0.6 * unit_random(seed, id, "tone");
            let c = base * tint * tone;
            // Milky, translucent clouds in the flint.
            let cl = smoothstep(0.15, 0.55, cloud.values()[i]) * unit_random(seed, id, "milk");
            let c = c + (milky - c) * (0.5 * cl);
            // A white cortex rind, of varying width, on some flints.
            let has = if unit_random(seed, id, "cortex") < cortex {
                1.0
            } else {
                0.0
            };
            let width =
                (0.003 + 0.007 * unit_random(seed, id, "rind")) * (1.0 + 0.6 * rind_n.values()[i]);
            let rind = has * smoothstep(width, 0.6 * width, -edge[i]);
            // Conchoidal ripples around the point of percussion, off center.
            let strike = Vec2::new(
                unit_random(seed, id, "sx") - 0.5,
                unit_random(seed, id, "sy") - 0.5,
            );
            let r = (units.local[i] - strike).length();
            let ripple = 0.00015 * libm::sinf(r * 28.0 + 6.0 * unit_random(seed, id, "phase"));
            s.color.push(c + (white - c) * rind);
            s.roughness.push(rough + 0.1 * cl + (0.92 - rough) * rind);
            s.height
                .push(arris(edge[i], 0.004, 0.003) + ripple * (1.0 - rind));
        }
        let joints = mortar(cx, &edge, args.color("mortar_color"), 0.003, joint)?;
        let m = lay(cx, &units, &edge, &s, FLINT, &joints)?;
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}

/// Roman brickwork: long, thin fired bricks (a sesquipedalis and its
/// half) in thick lime-and-brick-dust mortar, each brick its own firing
/// tone, sandy and pitted, with worn arrises.
///
/// Calibrated ([`super::calibration`]): the bricks' mean albedo and
/// roughness match a measured red brick's reflectance and a fired-clay
/// roughness.
#[derive(Copy, Clone, Debug, Default)]
pub struct RomanBrick;

impl Module for RomanBrick {
    fn interface(&self) -> Interface {
        let mut params = common_params(
            super::calibration::ROMAN_BRICK_COLOR,
            "the bricks' mean color",
            super::calibration::ROMAN_BRICK_ROUGHNESS,
            Vec3::new(0.44, 0.36, 0.30),
        );
        params.extend([
            meters("brick", [0.02, 0.1], 0.045, "brick height on the face"),
            meters("length", [0.15, 0.6], 0.44, "a whole brick's length"),
            meters("joint", [0.005, 0.05], 0.025, "joint width"),
        ]);
        Interface {
            id: ModuleId::new("dapple_library.roman_brick", 1),
            doc: "Roman brickwork in thick mortar".into(),
            params,
            inputs: unit_inputs(),
            outputs: material_output("the wall, surface identity BODY on the bricks"),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let joint = args.scalar("joint");
        let course = args.scalar("brick") + joint;
        let length = args.scalar("length");
        let units = units_for(args, grid, 0.03, |size| {
            Coursing {
                layout: "dapple_library.roman_brick",
                course: [course, course],
                length: [0.5 * length, length],
                joint,
                stagger: false,
                seed: args.seed("layout"),
                rounded: false,
            }
            .elements(size)
        })?;
        let sand = realize_scalar(grid, |b, d| fbm(b, d, [250.0, 250.0], args.seed("sand"), 2))?;
        let wear = realize_scalar(grid, |b, d| fbm(b, d, [30.0, 30.0], args.seed("wear"), 4))?;
        let (base, rough) = (args.color("color"), args.scalar("roughness"));
        let seed = args.seed("units");
        let mut s = UnitSurface {
            color: Vec::with_capacity(grid.len()),
            roughness: Vec::with_capacity(grid.len()),
            height: Vec::with_capacity(grid.len()),
        };
        let yellow = Vec3::new(1.25, 1.35, 1.2);
        for i in 0..grid.len() {
            let id = units.id[i];
            let tone = 0.85 + 0.3 * unit_random(seed, id, "tone");
            // A few bricks fired paler and yellower.
            let pale = smoothstep(0.8, 0.9, unit_random(seed, id, "pale"));
            let g = sand.values()[i];
            let c = base * tone * (Vec3::ONE + (yellow - Vec3::ONE) * pale) * (1.0 + 0.1 * g);
            s.color.push(c);
            s.roughness.push((rough + 0.03 * g).clamp(0.0, 1.0));
            let worn = 0.004 + 0.004 * wear.values()[i].max(0.0);
            s.height
                .push(arris(units.edge[i], worn, 0.003) + 0.00025 * g);
        }
        let joints = mortar(cx, &units.edge, args.color("mortar_color"), 0.003, joint)?;
        let m = lay(cx, &units, &units.edge, &s, BODY, &joints)?;
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}

/// Veined white marble in square slabs with hairline joints: a milky
/// ground clouded in grey, crossed by a few long veins (a curve network,
/// continuous across slabs, as if the slabs were cut from one block) and
/// many fine ones, polished as far as `polish` says.
///
/// Calibrated ([`super::calibration`]): the mean albedo matches a white
/// construction marble's measured reflectance.
#[derive(Copy, Clone, Debug, Default)]
pub struct Marble;

impl Module for Marble {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId::new("dapple_library.marble", 1),
            doc: "veined marble slabs".into(),
            params: vec![
                color(
                    "color",
                    super::calibration::MARBLE_COLOR,
                    "the ground's color",
                ),
                color("vein", Vec3::new(0.26, 0.28, 0.30), "the veins' color"),
                fraction("veining", 0.5, "how strongly veined"),
                fraction("polish", 0.7, "0 honed, 1 mirror polished"),
                meters("slab", [0.2, 2.0], 0.5, "slab size"),
                seed(),
            ],
            inputs: unit_inputs(),
            outputs: material_output("the slabs, surface identity MARBLE"),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let slab = args.scalar("slab");
        let units = units_for(args, grid, 0.01, |size| {
            Coursing {
                layout: "dapple_library.marble",
                course: [slab, slab],
                length: [slab, slab],
                joint: 0.0015,
                stagger: false,
                seed: args.seed("layout"),
                rounded: false,
            }
            .elements(size)
        })?;
        let size = extent(grid);
        let veins = vein_network(size, args.seed("veins"))?;
        let cloud = realize_scalar(grid, |b, d| fbm(b, d, [4.0, 4.0], args.seed("cloud"), 5))?;
        let fine = realize_scalar(grid, |b, d| fbm(b, d, [9.0, 9.0], args.seed("fine"), 6))?;
        let (ground, vein, veining, polish) = (
            args.color("color"),
            args.color("vein"),
            args.scalar("veining"),
            args.scalar("polish"),
        );
        let rough = 0.35 + (0.04 - 0.35) * polish;
        let t = grid.texel.min_element();
        let mut s = UnitSurface {
            color: Vec::with_capacity(grid.len()),
            roughness: Vec::with_capacity(grid.len()),
            height: Vec::with_capacity(grid.len()),
        };
        for i in 0..grid.len() {
            let p = grid.center(i) - grid.origin;
            let v = veins.nearest(p).map_or(0.0, |n| {
                let w = n.width.max(t);
                // A sharp core and a soft halo around it.
                smoothstep(w, 0.25 * w, n.distance) + 0.3 * smoothstep(6.0 * w, w, n.distance)
            });
            // Fine veins: the thin ridges of a warped fractal.
            let ridge = 1.0 - libm::fabsf(fine.values()[i]);
            let thin = smoothstep(0.93, 0.99, ridge) * 0.6;
            let k = (veining * (v.min(1.0) + thin)).min(1.0);
            let c = ground * (1.0 - 0.07 * cloud.values()[i]);
            s.color.push(c + (vein - c) * k);
            s.roughness.push(rough);
            s.height.push(arris(units.edge[i], 0.0008, 0.0003));
        }
        // Hairline joints: marble dust in a pale mortar, nearly invisible.
        let joints = mortar(cx, &units.edge, ground * 0.8, 0.0003, 0.002)?;
        let m = lay(cx, &units, &units.edge, &s, MARBLE, &joints)?;
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}

/// A few long veins across the tile: curves that wander diagonally and
/// close on themselves across the period, so the network tiles.
fn vein_network(size: Vec2, seed: u64) -> Result<dapple_elements::CurveNetwork, ModuleError> {
    let r = |c: u64, k: u64| unit_f32(hash(seed, &[c, k]));
    let mut curves = Vec::new();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "tile extents are a few meters"
    )]
    let period = [
        libm::roundf(size.x).max(1.0) as u32,
        libm::roundf(size.y).max(1.0) as u32,
    ];
    let domain = dapple_field::Domain::periodic(period[0], period[1])
        .ok_or_else(|| fail("marble needs a whole-meter tile"))?;
    for c in 0..4_u64 {
        let y0 = r(c, 0) * size.y;
        let turns = if r(c, 1) < 0.5 { 0.0 } else { 1.0 };
        let (a1, a2) = (0.08 * r(c, 2), 0.04 * r(c, 3));
        let (p1, p2) = (
            core::f32::consts::TAU * r(c, 4),
            core::f32::consts::TAU * r(c, 5),
        );
        let width = 0.0015 + 0.003 * r(c, 6);
        let steps = 96;
        let mut points = Vec::with_capacity(steps + 1);
        let mut widths = Vec::with_capacity(steps + 1);
        for k in 0..=steps {
            #[expect(clippy::cast_precision_loss, reason = "small step counts")]
            let u = k as f32 / steps as f32 * 1.4 - 0.2;
            let tau = core::f32::consts::TAU;
            // Whole harmonics of the period, so the curve closes on itself.
            let y = y0
                + turns * u * size.y
                + size.y
                    * (a1 * libm::sinf(tau * u + p1)
                        + a2 * libm::sinf(2.0 * tau * u + p2)
                        + 0.02 * libm::sinf(5.0 * tau * u + p1 + p2)
                        + 0.008 * libm::sinf(11.0 * tau * u + 2.0 * p2));
            points.push(Vec2::new(u * size.x, y));
            widths.push(width * (0.6 + 0.4 * libm::sinf(3.0 * tau * u + p1).abs()));
        }
        curves
            .push(dapple_elements::Curve::open(points, widths).map_err(|_| fail("a vein curve"))?);
    }
    Ok(dapple_elements::CurveNetwork::new(domain, curves))
}

/// Plain clay roof tiles in broken bond: each course laps the one below,
/// so a tile is thickest at its exposed lower edge and cambered across,
/// its own firing tone, weathered darker toward the tail, with the tile
/// beneath showing, in shadow, through the gaps.
///
/// Calibrated ([`super::calibration`]): the tiles' mean albedo and
/// roughness match a weathered terracotta roofing tile's measured
/// reflectance and a fired-clay roughness.
#[derive(Copy, Clone, Debug, Default)]
pub struct TerracottaTile;

impl Module for TerracottaTile {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId::new("dapple_library.terracotta_tile", 1),
            doc: "plain clay roof tiles".into(),
            params: vec![
                color(
                    "color",
                    super::calibration::TERRACOTTA_COLOR,
                    "the tiles' mean color",
                ),
                fraction(
                    "roughness",
                    super::calibration::TERRACOTTA_ROUGHNESS,
                    "specular roughness",
                ),
                meters("gauge", [0.05, 0.3], 0.1, "exposed height of a course"),
                meters("width", [0.1, 0.4], 0.165, "tile width"),
                meters("thickness", [0.005, 0.03], 0.012, "tile thickness"),
                fraction("weathering", 0.4, "how weathered the tiles are"),
                seed(),
            ],
            inputs: unit_inputs(),
            outputs: material_output("the roof, surface identity BODY"),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let (gauge, width) = (args.scalar("gauge"), args.scalar("width"));
        let units = units_for(args, grid, 0.02, |size| {
            Coursing {
                layout: "dapple_library.plain_tile",
                course: [gauge, gauge],
                length: [width, width],
                joint: 0.003,
                stagger: true,
                seed: args.seed("layout"),
                rounded: false,
            }
            .elements(size)
        })?;
        let grain = realize_scalar(grid, |b, d| {
            fbm(b, d, [200.0, 200.0], args.seed("grain"), 2)
        })?;
        let lichen = realize_scalar(grid, |b, d| fbm(b, d, [12.0, 12.0], args.seed("lichen"), 5))?;
        let (base, rough, thick, weathering) = (
            args.color("color"),
            args.scalar("roughness"),
            args.scalar("thickness"),
            args.scalar("weathering"),
        );
        let orange = Vec3::new(1.5, 1.0, 0.7);
        let seed = args.seed("units");
        let mut s = UnitSurface {
            color: Vec::with_capacity(grid.len()),
            roughness: Vec::with_capacity(grid.len()),
            height: Vec::with_capacity(grid.len()),
        };
        let gap_color = base * 0.35;
        for i in 0..grid.len() {
            let id = units.id[i];
            let l = units.local[i];
            let tone = 0.85 + 0.3 * unit_random(seed, id, "tone");
            let fresh = smoothstep(0.75, 0.95, unit_random(seed, id, "fresh"));
            let g = grain.values()[i];
            // Dirt gathers toward the tile's tail, under the course above.
            let tail = smoothstep(-0.2, 1.0, l.y) * weathering;
            // The course above lies on this tile's tail: its butt casts a
            // shadow down onto the exposed face just below it.
            let lap_shadow = smoothstep(0.35, 1.0, l.y);
            let c = base * tone * (Vec3::ONE + (orange - Vec3::ONE) * fresh) * (1.0 + 0.08 * g);
            let c = c * (1.0 - 0.3 * tail);
            let spots = smoothstep(0.35, 0.55, lichen.values()[i]) * weathering;
            let c = c + (Vec3::new(0.30, 0.30, 0.24) - c) * (0.5 * spots);
            s.color.push(c * (1.0 - 0.55 * lap_shadow * lap_shadow));
            s.roughness.push((rough + 0.03 * g).clamp(0.0, 1.0));
            // Thickest at the lower edge (local y = -1), cambered across.
            let lap = thick * (0.5 - 0.5 * l.y);
            let camber = 0.0015 * (1.0 - l.x * l.x);
            s.height
                .push(lap + camber + arris(units.edge[i], 0.002, 0.001) + 0.0002 * g);
        }
        // The gaps show the tile beneath, in shadow.
        let mut gaps = Material::new(grid);
        gaps.set_param(
            Param::BaseColor,
            Channel::Constant(Value::Vector3(gap_color)),
        )?;
        gaps.set_param(
            Param::SpecularRoughness,
            Channel::Constant(Value::Scalar(rough)),
        )?;
        gaps.set_aux(Aux::Height, Channel::Constant(Value::Scalar(-0.004)))?;
        gaps.set_aux(Aux::Surface, Channel::Constant(Value::Id(BODY)))?;
        let m = lay(cx, &units, &units.edge, &s, BODY, &gaps)?;
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}
