// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Glazed brickwork's structure: keyed bricks whose identity drives their
//! shape, glaze thickness, glaze color and chips.
//!
//! This is the structural half of the glazed brick wall; the materials
//! (ceramic body, glaze, mortar, weathering) are modules
//! ([`crate::modules`]), and [`crate::modules::GlazedBrickWall`] puts them
//! together. Here:
//!
//! - [`layout`]: running-bond bricks on a 1 m tile, laid a little off true,
//!   each with a glaze color from a [`Palette`] (a dado, a band and a
//!   field, each brick varied from its course's color), a glaze thickness
//!   and how battered it is, all derived from its key;
//! - [`program`]: what one brick is at a point: its surface height, where
//!   its glaze remains and how thick, its glaze color, and crazing;
//! - [`instance`] binds the program to the layout's attributes and to
//!   key-derived randomness; [`BACKGROUND`] is what lies between bricks;
//! - [`Structure::of`] reads a composite into maps.
//!
//! **One brick** is shaped from its own key: a face a few millimeters proud
//! of the mortar, a rounded arris whose radius and profile vary per brick,
//! a slightly wandering outline, a face tilted and bowed by fractions of a
//! millimeter; a glaze that pools into a bead above the lower arris and
//! breaks thin over the edges; conchoidal chips broken from the arrises,
//! deepest at corners, and a few pits; and a crazing network on some
//! bricks. Moving a brick moves all of it.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use dapple_elements::{
    AttributeDecl, Binding, CompositeError, Element, ElementError, ElementSet, InstanceId,
    LayoutId, Placement, ProgramInstance, Realized, RunningBond,
};
use dapple_field::program::{NodeId, Op, ProgramBuilder, ProgramError, ValueProgram};
use dapple_field::scoped::{ContractError, Node, Scope, ScopedBuilder, ScopedProgram};
use dapple_field::{
    Basis, CellOutput, Domain, FractalParams, PortType, Primaries, ScatterOutput, Stamp, Value,
};
use dapple_raster::Raster;
use dapple_raster::typed::{Storage, TypedRaster};
use glam::{Vec2, Vec3};

/// Surface-material identifier of the glaze.
pub const GLAZE: u32 = 1;
/// Surface-material identifier of the exposed ceramic body.
pub const BODY: u32 = 2;
/// Surface-material identifier of the mortar.
pub const MORTAR: u32 = 3;
/// Surface-material identifier of dirt and grime deposits.
pub const DIRT: u32 = 4;
/// Surface-material identifier of salt deposits (efflorescence).
pub const SALT: u32 = 5;
/// Surface-material identifier of stone.
pub const STONE: u32 = 6;
/// Surface-material identifier of wood.
pub const WOOD: u32 = 7;
/// Surface-material identifier of moss.
pub const MOSS: u32 = 8;

/// Bricks per course and courses per 1 m tile: 250 × 71 mm courses
/// including the joints.
pub const BOND: [u32; 2] = [4, 14];

/// Mortar joint width, in meters.
pub const JOINT: f32 = 0.01;

/// A brick's nominal half length, in meters.
const HALF_LENGTH: f32 = 0.5 / 4.0 - 0.5 * JOINT;

/// The brick face's height above the mortar's datum, in meters.
const FACE: f32 = 0.006;

/// Victorian glazed-brick glazes, as linear Rec. 709 colors: muted, as
/// fired glazes are, after references of London light wells and pub
/// fronts.
pub mod glazes {
    use glam::Vec3;

    /// Cream (ivory) glaze, the commonest light-well brick: about sRGB
    /// (223, 209, 177), as photographed cream glazed brick reads in
    /// daylight.
    pub const CREAM: Vec3 = Vec3::new(0.74, 0.64, 0.44);
    /// Honey-brown (salt-glaze brown).
    pub const BROWN: Vec3 = Vec3::new(0.072, 0.033, 0.014);
    /// Bottle green.
    pub const GREEN: Vec3 = Vec3::new(0.040, 0.078, 0.042);
    /// Oxblood.
    pub const OXBLOOD: Vec3 = Vec3::new(0.105, 0.017, 0.012);
}

/// Which glaze each course gets: a dado of `dado_courses` from the foot of
/// the tile, then `band_courses` of the band, then the field.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Palette {
    /// The glaze above the band.
    pub field: Vec3,
    /// The band's glaze.
    pub band: Vec3,
    /// The dado's glaze.
    pub dado: Vec3,
    /// Courses of dado, from the tile's foot.
    pub dado_courses: u32,
    /// Courses of band above the dado.
    pub band_courses: u32,
    /// How much bricks vary from their course's glaze, a fraction: kiln
    /// variation in depth of color and, less, in hue.
    pub variation: f32,
    /// How battered the bricks are, a fraction: scales chips.
    pub battered: f32,
}

impl Default for Palette {
    /// A cream field over a green band and a brown dado.
    fn default() -> Self {
        Self {
            field: glazes::CREAM,
            band: glazes::GREEN,
            dado: glazes::BROWN,
            dado_courses: 4,
            band_courses: 1,
            variation: 0.5,
            battered: 0.4,
        }
    }
}

/// The layout's logical identity.
#[must_use]
pub fn layout_id() -> LayoutId {
    LayoutId::named("dapple_library.glazed_brick")
}

/// The attribute schema: `glaze_color` (a color), `glaze_thickness` (in
/// meters), `knocks` (how battered, 0 to 1) and `replaced` (1 for a later
/// replacement brick).
#[must_use]
pub fn schema() -> Vec<AttributeDecl> {
    vec![
        AttributeDecl {
            name: String::from("glaze_color"),
            port: PortType::Color(Primaries::Rec709),
        },
        AttributeDecl {
            name: String::from("glaze_thickness"),
            port: PortType::Scalar,
        },
        AttributeDecl {
            name: String::from("knocks"),
            port: PortType::Scalar,
        },
        AttributeDecl {
            name: String::from("replaced"),
            port: PortType::Scalar,
        },
    ]
}

/// The running bond over one period of `domain`.
#[must_use]
pub fn bond(domain: Domain) -> RunningBond {
    RunningBond {
        layout: layout_id(),
        domain,
        courses: BOND[1],
        per_course: BOND[0],
        joint: JOINT,
    }
}

/// The bricks of a 1 m tile glazed by `palette`, as a bricklayer lays them:
/// each up to 1.5 mm off its bond position, turned up to 0.25°, and up to
/// 2 mm shorter and 1.2 mm lower than nominal (fired bricks shrink
/// unevenly), so joints vary in width.
///
/// A brick's glaze is its course's, darkened or lightened by up to
/// `variation · 25%` and shifted in hue by a fifth of that, from its key;
/// one brick in ten is a kiln outlier twice as far off. One in twenty-five
/// is a later **replacement**: a near match, cleaner and a little cooler
/// or redder, crisp and unchipped. Glaze thickness runs from 0.2 to
/// 0.45 mm.
///
/// # Errors
///
/// Never for the fixed bond.
pub fn layout(palette: &Palette) -> Result<ElementSet, ElementError> {
    let domain = Domain::periodic(1, 1).expect("a unit period");
    let mut set = bond(domain).elements(schema(), |key, anchor| {
        let course = u32::try_from(anchor.0[1]).unwrap_or(0);
        let base = if course < palette.dado_courses {
            palette.dado
        } else if course < palette.dado_courses + palette.band_courses {
            palette.band
        } else {
            palette.field
        };
        let outlier = if key.unit(20) < 0.1 { 2.0 } else { 1.0 };
        let depth = 1.0 + (key.unit(21) - 0.5) * 0.5 * palette.variation * outlier;
        let hue = Vec3::new(
            1.0 + (key.unit(22) - 0.5) * 0.1 * palette.variation,
            1.0,
            1.0 + (key.unit(23) - 0.5) * 0.1 * palette.variation,
        );
        let replaced = key.unit(25) < 0.04;
        let color = if replaced {
            // A near match from another maker: cleaner, a shade off.
            base * Vec3::new(1.08, 1.1, 1.18) * (0.95 + 0.1 * key.unit(26))
        } else {
            base * depth * hue
        }
        .clamp(Vec3::ZERO, Vec3::ONE);
        vec![
            Value::Vector3(color),
            Value::Scalar(0.0002 + 0.00025 * key.unit(2)),
            Value::Scalar(if replaced {
                0.0
            } else {
                palette.battered * (0.3 + 0.7 * key.unit(24))
            }),
            Value::Scalar(f32::from(u8::from(replaced))),
        ]
    })?;
    let elements: Vec<Element> = (0..set.len())
        .map(|i| {
            let mut e = set.element(i);
            let key = e.key;
            e.placement = Placement {
                center: e.placement.center
                    + Vec2::new(key.unit(11) - 0.5, key.unit(12) - 0.5) * 0.003,
                rotation: (key.unit(13) - 0.5) * 0.009,
            };
            e.half_size -= Vec2::new(key.unit(14) * 0.001, key.unit(15) * 0.0006);
            e
        })
        .collect();
    set = ElementSet::new(schema(), elements)?;
    Ok(set)
}

/// A finished planar field program of `op` alone.
fn leaf(op: Op) -> Result<ValueProgram, ProgramError> {
    let mut b = ProgramBuilder::new();
    let n = b.add(op)?;
    b.finish_value(n)
}

/// Gradient fBm over the plane.
fn fbm(frequency: [f32; 2], seed: u64, octaves: u8) -> Op {
    Op::Fractal {
        basis: Basis::Gradient,
        domain: Domain::Plane,
        frequency,
        seed,
        params: FractalParams {
            octaves,
            ..FractalParams::default()
        },
    }
}

/// Dome splats over the plane: `frequency` candidate cells per meter.
fn domes(
    b: &mut ProgramBuilder,
    frequency: f32,
    density: f32,
    radius: [f32; 2],
    seed: u64,
) -> Result<NodeId, ProgramError> {
    b.add(Op::Scatter {
        domain: Domain::Plane,
        placement: dapple_field::Placement {
            frequency,
            density,
            radius,
            rotate: false,
        },
        stamp: Stamp::Dome,
        seed,
        output: ScatterOutput::Max,
    })
}

/// `input` displaced by gradient noise of `frequency` by up to `amount`.
fn scalloped(
    b: &mut ProgramBuilder,
    input: NodeId,
    frequency: f32,
    amount: f32,
    seed: u64,
) -> Result<NodeId, ProgramError> {
    let noise = |seed| Op::Noise {
        basis: Basis::Gradient,
        domain: Domain::Plane,
        frequency: [frequency; 2],
        seed,
    };
    let dx = b.add(noise(seed))?;
    let dy = b.add(noise(seed + 1))?;
    b.add(Op::Warp {
        input,
        dx,
        dy,
        amount,
    })
}

/// Chips along an edge: sampled on the arris, how far (in meters) the
/// glaze has spalled away from it there. Impacts of a spread of sizes each
/// bite a scalloped, roughly semicircular piece: up to 9 mm deep into the
/// face for large ones, 3 mm for nicks.
fn chip_field() -> Result<ValueProgram, ProgramError> {
    let mut b = ProgramBuilder::new();
    let scaled = |b: &mut ProgramBuilder, input, value| -> Result<NodeId, ProgramError> {
        let k = b.add(Op::Constant {
            domain: Domain::Plane,
            value,
        })?;
        b.add(Op::Mul { a: input, b: k })
    };
    let large = domes(&mut b, 100.0, 0.08, [0.3, 0.95], 51)?;
    let large = scaled(&mut b, large, 0.009)?;
    let small = domes(&mut b, 320.0, 0.06, [0.25, 0.85], 52)?;
    let small = scaled(&mut b, small, 0.003)?;
    let bite = b.add(Op::Max { a: large, b: small })?;
    let bite = scalloped(&mut b, bite, 450.0, 0.0009, 53)?;
    b.finish_value(bite)
}

/// Small pits anywhere on a face: domes 0.6 to 1.6 mm across.
fn pit_field() -> Result<ValueProgram, ProgramError> {
    let mut b = ProgramBuilder::new();
    let pits = domes(&mut b, 140.0, 0.01, [0.07, 0.17], 55)?;
    b.finish_value(pits)
}

/// Crazing: the distance, in cells of about 15 mm, to the border of a
/// wandering Voronoi network.
fn crazing_field() -> Result<ValueProgram, ProgramError> {
    let mut b = ProgramBuilder::new();
    let cells = b.add(Op::Cellular {
        domain: Domain::Plane,
        frequency: [70.0, 62.0],
        jitter: 0.95,
        seed: 61,
        output: CellOutput::Border,
    })?;
    let out = scalloped(&mut b, cells, 160.0, 0.0018, 62)?;
    b.finish_value(out)
}

/// The glazed-brick structure program.
///
/// Inputs, in binding order: `glaze_color`, `thickness`, `knocks` and
/// `replaced` (the attributes), `seed`, `arris`, `profile`, `tilt_x`,
/// `tilt_y`, `bow`, `crazed` and `lip` (uniform per-element randomness),
/// `half_size` per element, and `local` and `edge` per sample.
///
/// Outputs: `height` (meters above the mortar's datum, before the glaze),
/// `cover` (1 on a brick), `glazed` (1 where glaze remains), `thickness`
/// (the glaze's, in meters), `glaze_color` (per element), `surface`
/// ([`GLAZE`] or [`BODY`]) and `craze` (1 on a crazing crack).
///
/// # Errors
///
/// Never: the program is fixed and satisfies its contract.
pub fn program() -> Result<ScopedProgram, ContractError> {
    let fail = |_| ContractError::UnknownReference;
    let color = PortType::Color(Primaries::Rec709);
    let mut p = ScopedBuilder::new("dapple_library.glazed_brick");
    let glaze_color = p.input("glaze_color", color, Scope::Element)?;
    let element =
        |p: &mut ScopedBuilder, name: &str| p.input(name, PortType::Scalar, Scope::Element);
    let thickness = element(&mut p, "thickness")?;
    let knocks = element(&mut p, "knocks")?;
    let replaced = element(&mut p, "replaced")?;
    let seed = element(&mut p, "seed")?;
    let arris = element(&mut p, "arris")?;
    let profile = element(&mut p, "profile")?;
    let tilt_x = element(&mut p, "tilt_x")?;
    let tilt_y = element(&mut p, "tilt_y")?;
    let bow = element(&mut p, "bow")?;
    let crazed = element(&mut p, "crazed")?;
    let lip = element(&mut p, "lip")?;
    let half = p.input("half_size", PortType::Vector2, Scope::Element)?;
    let local = p.input("local", PortType::Vector2, Scope::Sample)?;
    let edge = p.input("edge", PortType::Scalar, Scope::Sample)?;

    let wobble = p.resource("wobble", leaf(fbm([18.0, 18.0], 31, 3)).map_err(fail)?);
    let mottle = p.resource("mottle", leaf(fbm([25.0, 25.0], 91, 3)).map_err(fail)?);
    let chips = p.resource("chips", chip_field().map_err(fail)?);
    let pits = p.resource("pits", pit_field().map_err(fail)?);
    let crazing = p.resource("crazing", crazing_field().map_err(fail)?);

    // Every resource is read at this brick's own place in its noise.
    let sx = p.scalar(37.0);
    let sy = p.scalar(53.0);
    let ox = p.mul(seed, sx)?;
    let oy = p.mul(seed, sy)?;
    let offset = p.node(Node::Vector2(ox, oy))?;
    let at = p.add(local, offset)?;

    let hx = p.component(half, 0)?;
    let hy = p.component(half, 1)?;
    let lx = p.component(local, 0)?;
    let ly = p.component(local, 1)?;
    let zero = p.scalar(0.0);
    let one = p.scalar(1.0);

    // --- Shape -----------------------------------------------------------
    // A slightly wandering outline: the edge distance, off by up to 0.6 mm.
    let w = p.sample(wobble, at)?;
    let e = p.add_scaled(edge, 0.0006, w)?;
    // The arris: radius r from 1.5 to 4.5 mm, rounded (t(2 − t)) or
    // S-shaped (smoothstep) per brick.
    let r = p.lerp(0.0015, 0.0045, arris)?;
    let inv_r = p.lerp(1.0 / 0.0015, 1.0 / 0.0045, arris)?;
    let t = p.mul(e, inv_r)?;
    let t = p.saturate(t)?;
    let two = p.scalar(2.0);
    let two_minus = p.sub(two, t)?;
    let round = p.mul(t, two_minus)?;
    let s_curve = p.node(Node::SmoothStep {
        edge0: zero,
        edge1: r,
        x: e,
    })?;
    let shape = p.mix(round, s_curve, profile)?;
    // The face: lipping, set up to 1.2 mm proud of or back from its
    // neighbors; tilted up to ±1.1 mm along the brick and ±0.6 mm across
    // it; and bowed by up to 0.9 mm at its middle.
    let tx = p.lerp(-0.009, 0.009, tilt_x)?;
    let ty = p.lerp(-0.02, 0.02, tilt_y)?;
    let along = p.mul(tx, lx)?;
    let across = p.mul(ty, ly)?;
    let tilt = p.add(along, across)?;
    let u = p.scale(1.0 / HALF_LENGTH, lx)?;
    let u2 = p.mul(u, u)?;
    let bulge = p.sub(one, u2)?;
    let b = p.lerp(-0.0003, 0.0009, bow)?;
    let bowed = p.mul(b, bulge)?;
    let face = p.add(tilt, bowed)?;
    let lipping = p.lerp(-0.0012, 0.0012, lip)?;
    let face = p.add(face, lipping)?;
    let face = p.offset(face, FACE)?;
    let drop = p.sub(one, shape)?;
    let drop = p.mul(r, drop)?;
    let brick_h = p.sub(face, drop)?;

    // --- Glaze thickness --------------------------------------------------
    // A bead pooled just above the lower arris (glazes run down in the
    // kiln), 3 to 12 mm up; slow, isotropic unevenness; thin over the
    // arris, where the glaze breaks.
    let neg_hy = p.sub(zero, hy)?;
    let b0 = p.offset(neg_hy, 0.003)?;
    let b1 = p.offset(neg_hy, 0.006)?;
    let b2 = p.offset(neg_hy, 0.012)?;
    let rise = p.node(Node::SmoothStep {
        edge0: b0,
        edge1: b1,
        x: ly,
    })?;
    let fall = p.node(Node::SmoothStep {
        edge0: b2,
        edge1: b1,
        x: ly,
    })?;
    let bead = p.mul(rise, fall)?;
    let m = p.sample(mottle, at)?;
    let depth = p.add_scaled(one, 0.2, m)?;
    let depth = p.add_scaled(depth, 0.5, bead)?;
    let covered = p.smoothstep(0.0, 0.004, e)?;
    let breaking = p.lerp(0.3, 1.0, covered)?;
    let depth = p.mul(depth, breaking)?;
    let glaze_t = p.mul(thickness, depth)?;
    let glaze_t = p.node(Node::Max(glaze_t, zero))?;

    // Crazing on about a third of the bricks.
    let border = p.sample(crazing, at)?;
    let crack = p.smoothstep(0.012, 0.003, border)?;
    let crazes = p.smoothstep(0.65, 0.8, crazed)?;
    let crack = p.mul(crack, crazes)?;
    let original = p.sub(one, replaced)?;
    let crack = p.mul(crack, original)?;

    // --- Chips -----------------------------------------------------------
    // Chips break from the arris: the chip field is read at the point of
    // the nearest edge opposite this sample, and bites that far into the
    // face. Corners, exposed on two sides, bite deeper; how battered a
    // brick is scales every bite.
    let ax = p.node(Node::Length(lx))?;
    let ay = p.node(Node::Length(ly))?;
    let dx = p.sub(hx, ax)?;
    let dy = p.sub(hy, ay)?;
    let right = p.smoothstep(-1.0e-6, 1.0e-6, lx)?;
    let up = p.smoothstep(-1.0e-6, 1.0e-6, ly)?;
    let neg_hx = p.sub(zero, hx)?;
    let ex = p.mix(neg_hx, hx, right)?;
    let ey = p.mix(neg_hy, hy, up)?;
    let on_horizontal = p.node(Node::Vector2(lx, ey))?;
    let on_vertical = p.node(Node::Vector2(ex, ly))?;
    let gap = p.sub(dx, dy)?;
    let horizontal = p.smoothstep(-1.0e-6, 1.0e-6, gap)?;
    let foot = p.node(Node::Select {
        condition: horizontal,
        a: on_horizontal,
        b: on_vertical,
    })?;
    let foot = p.add(foot, offset)?;
    let bite = p.sample(chips, foot)?;
    let knocks2 = p.mul(knocks, knocks)?;
    let battered = p.lerp(0.2, 1.4, knocks2)?;
    let bite = p.mul(bite, battered)?;
    let cx = p.smoothstep(0.014, 0.0, dx)?;
    let cy = p.smoothstep(0.014, 0.0, dy)?;
    let corner = p.mul(cx, cy)?;
    let corner = p.mul(corner, knocks)?;
    let bite = p.add_scaled(bite, 0.003, corner)?;
    let inside = p.sub(bite, e)?;
    let spalled = p.smoothstep(0.0, 0.0002, inside)?;
    let pit = p.sample(pits, at)?;
    let pitted = p.smoothstep(0.45, 0.6, pit)?;
    let chip = p.node(Node::Max(spalled, pitted))?;
    let glazed = p.sub(one, chip)?;
    // A conchoidal dish, deepest at the arris where it broke away: up to
    // 60% of the bite's reach, plus 0.5 mm where the glaze itself went.
    let inside = p.node(Node::Max(inside, zero))?;
    let dish = p.scale(0.6, inside)?;
    let dish = p.offset(dish, 0.0005)?;
    let pit_depth = p.scale(0.0008, pit)?;
    let dish = p.mix(pit_depth, dish, spalled)?;
    let dent = p.mul(chip, dish)?;
    let height = p.sub(brick_h, dent)?;
    let glaze_t = p.mul(glaze_t, glazed)?;

    let glaze_id = p.constant(Value::Id(GLAZE));
    let body_id = p.constant(Value::Id(BODY));
    let surface = p.node(Node::Select {
        condition: glazed,
        a: glaze_id,
        b: body_id,
    })?;

    p.output("height", PortType::Scalar, Scope::Sample, height)?;
    p.output("cover", PortType::Mask, Scope::Sample, one)?;
    p.output("glazed", PortType::Mask, Scope::Sample, glazed)?;
    p.output("thickness", PortType::Scalar, Scope::Sample, glaze_t)?;
    p.output("glaze_color", color, Scope::Element, glaze_color)?;
    p.output("surface", PortType::Id, Scope::Sample, surface)?;
    p.output("craze", PortType::Mask, Scope::Sample, crack)?;
    Ok(p.finish())
}

/// The program bound to [`layout`]'s attributes and to key-derived
/// randomness (streams 3 to 10).
///
/// # Errors
///
/// Never for [`program`]'s contract.
pub fn instance(program: Arc<ScopedProgram>) -> Result<ProgramInstance, ContractError> {
    let mut bindings = vec![
        Binding::Attribute(String::from("glaze_color")),
        Binding::Attribute(String::from("glaze_thickness")),
        Binding::Attribute(String::from("knocks")),
        Binding::Attribute(String::from("replaced")),
    ];
    bindings.extend((3..=10).map(Binding::ElementRandom));
    bindings.extend([
        Binding::HalfSize,
        Binding::LocalPosition,
        Binding::EdgeDistance,
    ]);
    ProgramInstance::new(
        InstanceId::named("dapple_library.glazed_brick"),
        program,
        bindings,
    )
}

/// What lies between bricks, in [`program`]'s output order.
pub const BACKGROUND: [Value; 7] = [
    Value::Scalar(0.0),
    Value::Scalar(0.0),
    Value::Scalar(0.0),
    Value::Scalar(0.0),
    Value::Vector3(Vec3::ZERO),
    Value::Id(MORTAR),
    Value::Scalar(0.0),
];

/// A glazed-brick composite read into maps.
#[derive(Clone, Debug, PartialEq)]
pub struct Structure {
    /// Brick surface height in meters above the mortar's datum, before the
    /// glaze; 0 between bricks.
    pub height: Raster,
    /// The bricks' share of each texel.
    pub cover: Raster,
    /// Where glaze remains.
    pub glazed: Raster,
    /// Glaze thickness in meters.
    pub thickness: Raster,
    /// Each brick's glaze color.
    pub glaze_color: TypedRaster,
    /// [`GLAZE`], [`BODY`] or [`MORTAR`], by the dominant contributor.
    pub surface: TypedRaster,
    /// Crazing cracks.
    pub craze: Raster,
    /// Which brick owns each texel: dense labels, 0 for mortar.
    pub owners: TypedRaster,
}

impl Structure {
    /// Reads `realized`, a composite of [`program`] under [`instance`].
    ///
    /// # Errors
    ///
    /// [`CompositeError::Background`] when an output is missing or has the
    /// wrong storage.
    pub fn of(realized: &Realized) -> Result<Self, CompositeError> {
        let get = |name: &str| -> Result<TypedRaster, CompositeError> {
            realized.output(name)?.ok_or(CompositeError::Background)
        };
        let scalar = |name: &str| -> Result<Raster, CompositeError> {
            match get(name)?.storage() {
                Storage::F32(r) => Ok(r.clone()),
                _ => Err(CompositeError::Background),
            }
        };
        Ok(Self {
            height: scalar("height")?,
            cover: scalar("cover")?,
            glazed: scalar("glazed")?,
            thickness: scalar("thickness")?,
            glaze_color: get("glaze_color")?,
            surface: get("surface")?,
            craze: scalar("craze")?,
            owners: realized.owner_labels()?,
        })
    }
}
