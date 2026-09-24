// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Glazed brick: keyed bricks whose identity drives every channel.
//!
//! Unlike [`brick`](crate::brick), which is one field program, this material
//! is an element set plus a surface program (`dapple_elements`):
//!
//! - [`layout`]: running-bond bricks on a 1 m tile, each with a glaze tone
//!   and glaze thickness derived from its key, and laid a little off true
//!   (under a millimeter and a tenth of a degree);
//! - [`program`]: what one brick looks like at a point (see below);
//! - [`instance`] binds the program to the layout's attributes and to
//!   key-derived randomness, and [`BACKGROUND`] is flat mortar;
//! - [`finish`] replaces the flat mortar of a composite with textured,
//!   tooled mortar recessed between the bricks.
//!
//! **One brick** is shaped and glazed from its own key:
//!
//! - *Shape:* a face a few millimeters proud of the mortar, with a rounded
//!   arris whose radius (1.5 to 4.5 mm) and profile (a round or an S-shaped
//!   edge) vary per brick, a slightly wandering outline, and a face that is
//!   tilted and bowed by fractions of a millimeter, so each brick catches
//!   the sky at its own angle.
//! - *Glaze:* a thickness that runs down the face in streaks, beads just
//!   above the lower arris and breaks thin over the edges. Its color follows
//!   the thickness: thin glaze lets the pale body through, thick glaze is
//!   deep and saturated. Its roughness varies across the face and rises where
//!   it is thin; an orange-peel ripple and crazing (a network of fine,
//!   dirt-darkened cracks, heavier on thickly glazed bricks) complete it.
//! - *Chips:* impacts of a spread of sizes break scalloped, shell-like
//!   pieces away from the arrises, deeper at the corners, which are exposed
//!   on two sides; a few small pits land anywhere on the face. How battered
//!   a brick is varies per brick. Chips expose the ceramic body, rough and
//!   grainy, dished deepest where they broke away.
//!
//! One brick's key decides all of it, so moving a brick moves its
//! appearance with it, and the chips in the color, height, roughness and
//! material outputs always agree. The `material` output separates the glaze
//! ([`GLAZE`]) from the exposed body ([`BODY`]) and the mortar ([`MORTAR`]);
//! which brick owns a texel is a separate raster.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use dapple_elements::{
    AttributeDecl, Binding, CompositeError, ContractError, ElementError, ElementSet, InstanceId,
    LayoutId, Node, NodeRef, Placement, ProgramInstance, Realized, RunningBond, Scope,
    SurfaceBuilder, SurfaceProgram,
};
use dapple_field::program::{NodeId, Op, ProgramBuilder, ProgramError, ValueProgram};
use dapple_field::{
    Basis, CellOutput, Domain, FractalParams, PortType, Primaries, ScatterOutput, Stamp, Value,
};
use dapple_raster::typed::{Storage, TypedError, TypedRaster, realize_value};
use dapple_raster::{DistanceTransform, Raster, RasterError, RasterOp, Realization};
use glam::{Vec2, Vec3};

/// Surface-material identifier of the glaze.
pub const GLAZE: u32 = 0;
/// Surface-material identifier of the exposed ceramic body.
pub const BODY: u32 = 1;
/// Surface-material identifier of the mortar.
pub const MORTAR: u32 = 2;

/// Bricks per course and courses per 1 m tile: 250 × 71 mm courses
/// including the joints.
pub const BOND: [u32; 2] = [4, 14];

/// Mortar joint width, in meters.
pub const JOINT: f32 = 0.01;

/// A brick's nominal half length, in meters.
const HALF_LENGTH: f32 = 0.5 / 4.0 - 0.5 * JOINT;

/// The brick face's height above the mortar's datum, in meters.
const FACE: f32 = 0.006;

/// The layout's logical identity.
#[must_use]
pub fn layout_id() -> LayoutId {
    LayoutId::named("dapple_library.glazed_brick")
}

/// The attribute schema: `glaze_tone` and `glaze_thickness`, both scalars.
#[must_use]
pub fn schema() -> Vec<AttributeDecl> {
    vec![
        AttributeDecl {
            name: String::from("glaze_tone"),
            port: PortType::Scalar,
        },
        AttributeDecl {
            name: String::from("glaze_thickness"),
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

/// The bricks of a 1 m tile, with key-derived glaze tones and thicknesses,
/// each laid up to 0.6 mm off its bond position and turned up to 0.1°.
///
/// # Errors
///
/// Never for the fixed bond.
pub fn layout() -> Result<ElementSet, ElementError> {
    let domain = Domain::periodic(1, 1).expect("a unit period");
    let mut set = bond(domain).elements(schema(), |key, _| {
        vec![
            Value::Scalar(key.unit(1)),
            Value::Scalar(0.4 + 0.6 * key.unit(2)),
        ]
    })?;
    for i in 0..set.len() {
        let key = set.keys()[i];
        let p = set.placement(i);
        let shift = Vec2::new(key.unit(11) - 0.5, key.unit(12) - 0.5) * 0.0012;
        set.set_placement(
            key,
            Placement {
                center: p.center + shift,
                rotation: (key.unit(13) - 0.5) * 0.0035,
            },
        )?;
    }
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
/// bite a scalloped, roughly semicircular piece whose depth into the face
/// follows the impact's profile along the edge: up to 7.5 mm for large ones,
/// 2.4 mm for nicks.
fn chip_field() -> Result<ValueProgram, ProgramError> {
    let mut b = ProgramBuilder::new();
    let scaled = |b: &mut ProgramBuilder, input, value| -> Result<NodeId, ProgramError> {
        let k = b.add(Op::Constant {
            domain: Domain::Plane,
            value,
        })?;
        b.add(Op::Mul { a: input, b: k })
    };
    let large = domes(&mut b, 110.0, 0.13, [0.25, 0.9], 51)?;
    let large = scaled(&mut b, large, 0.0075)?;
    let small = domes(&mut b, 330.0, 0.1, [0.2, 0.8], 52)?;
    let small = scaled(&mut b, small, 0.0024)?;
    let bite = b.add(Op::Max { a: large, b: small })?;
    let bite = scalloped(&mut b, bite, 450.0, 0.0008, 53)?;
    b.finish_value(bite)
}

/// Small pits anywhere on a face: domes 0.5 to 1.4 mm across, dark with
/// dirt.
fn pit_field() -> Result<ValueProgram, ProgramError> {
    let mut b = ProgramBuilder::new();
    let pits = domes(&mut b, 140.0, 0.007, [0.06, 0.15], 55)?;
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

/// A small expression helper over a [`SurfaceBuilder`].
struct Body {
    b: SurfaceBuilder,
}

impl Body {
    fn c(&mut self, v: f32) -> NodeRef {
        self.b.constant(Value::Scalar(v))
    }

    fn rgb(&mut self, r: f32, g: f32, b: f32) -> NodeRef {
        self.b.constant(Value::Vector3(Vec3::new(r, g, b)))
    }

    fn n(&mut self, node: Node) -> Result<NodeRef, ContractError> {
        self.b.add(node)
    }

    fn add(&mut self, a: NodeRef, b: NodeRef) -> Result<NodeRef, ContractError> {
        self.n(Node::Add(a, b))
    }

    fn sub(&mut self, a: NodeRef, b: NodeRef) -> Result<NodeRef, ContractError> {
        self.n(Node::Sub(a, b))
    }

    fn mul(&mut self, a: NodeRef, b: NodeRef) -> Result<NodeRef, ContractError> {
        self.n(Node::Mul(a, b))
    }

    /// `a + k · b` for a constant `k`.
    fn add_k(&mut self, a: NodeRef, k: f32, b: NodeRef) -> Result<NodeRef, ContractError> {
        let k = self.c(k);
        let kb = self.mul(k, b)?;
        self.add(a, kb)
    }

    /// `a + k` for a constant `k`.
    fn plus(&mut self, a: NodeRef, k: f32) -> Result<NodeRef, ContractError> {
        let k = self.c(k);
        self.add(a, k)
    }

    /// `k · a` for a constant `k`.
    fn k(&mut self, k: f32, a: NodeRef) -> Result<NodeRef, ContractError> {
        let k = self.c(k);
        self.mul(k, a)
    }

    fn mix(&mut self, a: NodeRef, b: NodeRef, t: NodeRef) -> Result<NodeRef, ContractError> {
        self.n(Node::Mix { a, b, t })
    }

    /// `lo + (hi − lo) · t` for constants.
    fn lerp(&mut self, lo: f32, hi: f32, t: NodeRef) -> Result<NodeRef, ContractError> {
        let (lo, hi) = (self.c(lo), self.c(hi));
        self.mix(lo, hi, t)
    }

    fn step(
        &mut self,
        edge0: NodeRef,
        edge1: NodeRef,
        x: NodeRef,
    ) -> Result<NodeRef, ContractError> {
        self.n(Node::SmoothStep { edge0, edge1, x })
    }

    /// A smooth step of `x` between constant edges.
    fn step_k(&mut self, edge0: f32, edge1: f32, x: NodeRef) -> Result<NodeRef, ContractError> {
        let (e0, e1) = (self.c(edge0), self.c(edge1));
        self.step(e0, e1, x)
    }

    fn clamp01(&mut self, x: NodeRef) -> Result<NodeRef, ContractError> {
        let (lo, hi) = (self.c(0.0), self.c(1.0));
        self.n(Node::Clamp { x, lo, hi })
    }

    fn component(&mut self, input: NodeRef, index: u8) -> Result<NodeRef, ContractError> {
        self.n(Node::Component { input, index })
    }

    fn sample(&mut self, resource: u32, at: NodeRef) -> Result<NodeRef, ContractError> {
        self.n(Node::Sample { resource, at })
    }
}

/// The glazed-brick surface program.
///
/// Inputs, in binding order: `tone` and `thickness` (the glaze attributes),
/// `seed`, `arris`, `profile`, `tilt_x`, `tilt_y`, `bow`, `gloss` and
/// `knocks` (uniform per-element randomness), `half_size` per element, and
/// `local` and `edge` per sample. Outputs: `base_color` (linear Rec. 709),
/// `height` (meters above the mortar's datum), `specular_roughness`,
/// `material` ([`GLAZE`] or [`BODY`]) and `cover` (1 on a brick, so a
/// composite's `cover` is the bricks' share of each texel) per sample, and
/// `glaze_tint` per element.
///
/// # Errors
///
/// Never: the program is fixed and satisfies its contract.
pub fn program() -> Result<SurfaceProgram, ContractError> {
    let fail = |_| ContractError::UnknownReference;
    let mut p = Body {
        b: SurfaceBuilder::new("dapple_library.glazed_brick"),
    };
    let element = |p: &mut Body, name: &str| p.b.input(name, PortType::Scalar, Scope::Element);
    let tone = element(&mut p, "tone")?;
    let thickness = element(&mut p, "thickness")?;
    let seed = element(&mut p, "seed")?;
    let arris = element(&mut p, "arris")?;
    let profile = element(&mut p, "profile")?;
    let tilt_x = element(&mut p, "tilt_x")?;
    let tilt_y = element(&mut p, "tilt_y")?;
    let bow = element(&mut p, "bow")?;
    let gloss = element(&mut p, "gloss")?;
    let knocks = element(&mut p, "knocks")?;
    let half = p.b.input("half_size", PortType::Vector2, Scope::Element)?;
    let local = p.b.input("local", PortType::Vector2, Scope::Sample)?;
    let edge = p.b.input("edge", PortType::Scalar, Scope::Sample)?;

    let wobble =
        p.b.resource("wobble", leaf(fbm([18.0, 18.0], 31, 3)).map_err(fail)?);
    let runs =
        p.b.resource("runs", leaf(fbm([90.0, 7.0], 41, 2)).map_err(fail)?);
    let mottle =
        p.b.resource("mottle", leaf(fbm([30.0, 30.0], 91, 3)).map_err(fail)?);
    let peel = p.b.resource(
        "orange_peel",
        leaf(Op::Noise {
            basis: Basis::Gradient,
            domain: Domain::Plane,
            frequency: [380.0, 380.0],
            seed: 81,
        })
        .map_err(fail)?,
    );
    let grain = p.b.resource(
        "grain",
        leaf(Op::Fractal {
            basis: Basis::Value,
            domain: Domain::Plane,
            frequency: [900.0, 900.0],
            seed: 71,
            params: FractalParams {
                octaves: 2,
                ..FractalParams::default()
            },
        })
        .map_err(fail)?,
    );
    let chips = p.b.resource("chips", chip_field().map_err(fail)?);
    let pits = p.b.resource("pits", pit_field().map_err(fail)?);
    let crazing = p.b.resource("crazing", crazing_field().map_err(fail)?);

    // Every resource is read at this brick's own place in its noise.
    let sx = p.c(37.0);
    let sy = p.c(53.0);
    let ox = p.mul(seed, sx)?;
    let oy = p.mul(seed, sy)?;
    let offset = p.n(Node::Vector2(ox, oy))?;
    let at = p.add(local, offset)?;

    let hx = p.component(half, 0)?;
    let hy = p.component(half, 1)?;
    let lx = p.component(local, 0)?;
    let ly = p.component(local, 1)?;
    let zero = p.c(0.0);
    let one = p.c(1.0);

    // --- Shape -----------------------------------------------------------
    // A slightly wandering outline: the edge distance, off by up to 0.6 mm.
    let w = p.sample(wobble, at)?;
    let e = p.add_k(edge, 0.0006, w)?;
    // The arris: radius r from 1.5 to 4.5 mm, `inv_r` its reciprocal
    // (interpolated between the ends' reciprocals, close enough for a
    // profile), rounded (t(2 − t)) or S-shaped (smoothstep) per brick.
    let r = p.lerp(0.0015, 0.0045, arris)?;
    let inv_r = p.lerp(1.0 / 0.0015, 1.0 / 0.0045, arris)?;
    let t = p.mul(e, inv_r)?;
    let t = p.clamp01(t)?;
    let two = p.c(2.0);
    let two_minus = p.sub(two, t)?;
    let round = p.mul(t, two_minus)?;
    let s_curve = p.step(zero, r, e)?;
    let shape = p.mix(round, s_curve, profile)?;
    // The face: tilted up to ±0.7 mm along the brick, ±0.3 mm across it,
    // and bowed by up to 0.9 mm at its middle.
    let tx = p.lerp(-0.006, 0.006, tilt_x)?;
    let ty = p.lerp(-0.01, 0.01, tilt_y)?;
    let along = p.mul(tx, lx)?;
    let across = p.mul(ty, ly)?;
    let tilt = p.add(along, across)?;
    let u = p.k(1.0 / HALF_LENGTH, lx)?;
    let u2 = p.mul(u, u)?;
    let bulge = p.sub(one, u2)?;
    let b = p.lerp(-0.0003, 0.0009, bow)?;
    let bowed = p.mul(b, bulge)?;
    let face = p.add(tilt, bowed)?;
    let face = p.plus(face, FACE)?;
    let drop = p.sub(one, shape)?;
    let drop = p.mul(r, drop)?;
    let brick_h = p.sub(face, drop)?;

    // --- Glaze -----------------------------------------------------------
    // How far down the face: 0 at the top edge, 1 at the bottom.
    let neg_hy = p.sub(zero, hy)?;
    let lower = p.step(hy, neg_hy, ly)?;
    // A bead just above the lower arris, 4 to 18 mm up.
    let b0 = p.plus(neg_hy, 0.004)?;
    let b1 = p.plus(neg_hy, 0.009)?;
    let b2 = p.plus(neg_hy, 0.018)?;
    let rise = p.step(b0, b1, ly)?;
    let fall = p.step(b2, b1, ly)?;
    let bead = p.mul(rise, fall)?;
    // Streaks running down, stronger lower on the face.
    let streak = p.sample(runs, at)?;
    let streak = p.mul(streak, lower)?;
    let depth = p.lerp(0.7, 0.9, lower)?;
    let beading = p.add_k(one, 0.6, streak)?;
    let bead = p.mul(bead, beading)?;
    let depth = p.add_k(depth, 0.3, bead)?;
    let depth = p.add_k(depth, 0.15, streak)?;
    // The glaze breaks thin over the arris.
    let covered = p.step_k(0.0, 0.004, e)?;
    let breaking = p.lerp(0.2, 1.0, covered)?;
    let depth = p.mul(depth, breaking)?;
    let glaze_t = p.mul(thickness, depth)?;
    let glaze_t = p.n(Node::Max(glaze_t, zero))?;

    // Color by thickness: thin glaze shows the pale body through a light
    // tint, thick glaze is deep and saturated.
    let green = p.rgb(0.035, 0.20, 0.14);
    let blue = p.rgb(0.030, 0.12, 0.28);
    let tint = p.mix(green, blue, tone)?;
    let m = p.sample(mottle, at)?;
    let mottling = p.add_k(one, 0.18, m)?;
    let body = p.rgb(0.47, 0.29, 0.19);
    let thin_body = p.k(0.25, body)?;
    let thin_tint = p.k(1.5, tint)?;
    let thin = p.add(thin_body, thin_tint)?;
    let deep = p.k(0.75, tint)?;
    let opacity = p.step_k(0.05, 0.5, glaze_t)?;
    let glaze_color = p.mix(thin, deep, opacity)?;
    let glaze_color = p.mul(glaze_color, mottling)?;

    // Crazing: fine cracks, heavier on thickly glazed bricks, darkened by
    // dirt, slightly rough and sunk.
    let border = p.sample(crazing, at)?;
    let crack = p.step_k(0.016, 0.003, border)?;
    let crazed = p.step_k(0.7, 0.95, thickness)?;
    let crack = p.mul(crack, crazed)?;
    let darken = p.k(-0.14, crack)?;
    let darken = p.add(one, darken)?;
    let glaze_color = p.mul(glaze_color, darken)?;

    // Roughness: per brick, mottled, rougher where thin and along cracks.
    let rough = p.lerp(0.05, 0.16, gloss)?;
    let rough = p.add_k(rough, 0.035, m)?;
    let thinness = p.sub(one, covered)?;
    let rough = p.add_k(rough, 0.12, thinness)?;
    let rough = p.add_k(rough, 0.08, crack)?;
    let op = p.sample(peel, at)?;
    let glaze_rough = p.add_k(rough, 0.015, op)?;

    // --- Chips -----------------------------------------------------------
    // Chips break from the arris: the chip field is read at the point of
    // the nearest edge opposite this sample, and bites that far into the
    // face. Corners, exposed on two sides, bite deeper; how battered a brick
    // is scales every bite.
    let ax = p.n(Node::Length(lx))?;
    let ay = p.n(Node::Length(ly))?;
    let dx = p.sub(hx, ax)?;
    let dy = p.sub(hy, ay)?;
    let right = p.step_k(-1.0e-6, 1.0e-6, lx)?;
    let up = p.step_k(-1.0e-6, 1.0e-6, ly)?;
    let neg_hx = p.sub(zero, hx)?;
    let ex = p.mix(neg_hx, hx, right)?;
    let ey = p.mix(neg_hy, hy, up)?;
    let on_horizontal = p.n(Node::Vector2(lx, ey))?;
    let on_vertical = p.n(Node::Vector2(ex, ly))?;
    let gap = p.sub(dx, dy)?;
    let horizontal = p.step_k(-1.0e-6, 1.0e-6, gap)?;
    let foot = p.n(Node::Select {
        condition: horizontal,
        a: on_horizontal,
        b: on_vertical,
    })?;
    let foot = p.add(foot, offset)?;
    let bite = p.sample(chips, foot)?;
    let knocks2 = p.mul(knocks, knocks)?;
    let battered = p.lerp(0.35, 1.0, knocks2)?;
    let bite = p.mul(bite, battered)?;
    let cx = p.step_k(0.012, 0.0, dx)?;
    let cy = p.step_k(0.012, 0.0, dy)?;
    let corner = p.mul(cx, cy)?;
    let corner = p.mul(corner, knocks)?;
    let bite = p.add_k(bite, 0.002, corner)?;
    let inside = p.sub(bite, e)?;
    let spalled = p.step_k(0.0, 0.00025, inside)?;
    let pit = p.sample(pits, at)?;
    let pitted = p.step_k(0.45, 0.6, pit)?;
    let chip = p.n(Node::Max(spalled, pitted))?;
    let glazed = p.sub(one, chip)?;
    // A shell-like dish, deepest at the arris where it broke away.
    let inside = p.n(Node::Max(inside, zero))?;
    let dish = p.k(0.5, inside)?;
    let dish = p.plus(dish, 0.0003)?;
    let pit_depth = p.k(0.0004, pit)?;
    let dish = p.mix(pit_depth, dish, spalled)?;
    let dent = p.mul(chip, dish)?;

    // The body: grainy, dulled by dirt in the chips and dark in the pits.
    let g = p.sample(grain, at)?;
    let grainy = p.add_k(one, 0.22, g)?;
    let body_color = p.mul(body, grainy)?;
    let dirt = p.lerp(0.8, 0.35, pitted)?;
    let body_color = p.mul(body_color, dirt)?;
    let base_color = p.mix(body_color, glaze_color, glazed)?;

    // Height: the brick, the glaze on it (up to 0.4 mm, with orange peel and
    // sunk cracks), less the chips.
    let coat = p.k(0.0003, glaze_t)?;
    let coat = p.add_k(coat, 0.000_012, op)?;
    let coat = p.add_k(coat, -0.000_04, crack)?;
    let coat = p.mul(coat, glazed)?;
    let raised = p.add(brick_h, coat)?;
    let height = p.sub(raised, dent)?;

    let body_rough = p.add_k(one, 0.06, g)?;
    let body_rough = p.k(0.85, body_rough)?;
    let roughness = p.mix(body_rough, glaze_rough, glazed)?;
    let roughness = p.clamp01(roughness)?;

    let glaze_id = p.b.constant(Value::Id(GLAZE));
    let body_id = p.b.constant(Value::Id(BODY));
    let material = p.n(Node::Select {
        condition: chip,
        a: body_id,
        b: glaze_id,
    })?;

    let color = PortType::Color(Primaries::Rec709);
    let mut b = p.b;
    b.output("base_color", color, Scope::Sample, base_color)?;
    b.output("height", PortType::Scalar, Scope::Sample, height)?;
    b.output(
        "specular_roughness",
        PortType::Scalar,
        Scope::Sample,
        roughness,
    )?;
    b.output("material", PortType::Id, Scope::Sample, material)?;
    b.output("glaze_tint", color, Scope::Element, tint)?;
    b.output("cover", PortType::Mask, Scope::Sample, one)?;
    Ok(b.finish())
}

/// The program bound to [`layout`]'s attributes and to key-derived
/// randomness (streams 3 to 10).
///
/// # Errors
///
/// Never for [`program`]'s contract.
pub fn instance(program: Arc<SurfaceProgram>) -> Result<ProgramInstance, ContractError> {
    let mut bindings = vec![
        Binding::Attribute(String::from("glaze_tone")),
        Binding::Attribute(String::from("glaze_thickness")),
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

/// The flat mortar, in [`program`]'s output order; [`finish`] textures it.
pub const BACKGROUND: [Value; 6] = [
    Value::Vector3(Vec3::new(0.42, 0.40, 0.37)),
    Value::Scalar(0.0),
    Value::Scalar(0.9),
    Value::Id(MORTAR),
    Value::Vector3(Vec3::ZERO),
    Value::Scalar(0.0),
];

/// The mortar's texture over `domain`, as two `Vector3` fields: its color,
/// and its grains' height in meters with its roughness (and a zero).
fn mortar_fields(domain: Domain) -> Result<[ValueProgram; 2], ProgramError> {
    let mut b = ProgramBuilder::new();
    // Sand grains about a millimeter across, each its own tone, in patches
    // of a few centimeters, lighter and darker.
    let cells = |b: &mut ProgramBuilder, output| {
        b.add(Op::Cellular {
            domain,
            frequency: [900.0, 900.0],
            jitter: 0.9,
            seed: 101,
            output,
        })
    };
    let grains = cells(&mut b, CellOutput::F1)?;
    let tone = cells(&mut b, CellOutput::CellValue)?;
    let patches = b.add(Op::Fractal {
        basis: Basis::Gradient,
        domain,
        frequency: [24.0, 24.0],
        seed: 102,
        params: FractalParams {
            octaves: 4,
            ..FractalParams::default()
        },
    })?;
    let fine = b.add(Op::Noise {
        basis: Basis::Value,
        domain,
        frequency: [300.0, 300.0],
        seed: 103,
    })?;
    let bright = crate::ramp(&mut b, patches, [-0.6, 0.6], [0.8, 1.1])?;
    let grain_tone = crate::ramp(&mut b, tone, [0.0, 1.0], [0.7, 1.3])?;
    let fine_tone = crate::ramp(&mut b, fine, [-1.0, 1.0], [0.95, 1.05])?;
    let bright = b.add(Op::Mul {
        a: bright,
        b: grain_tone,
    })?;
    let bright = b.add(Op::Mul {
        a: bright,
        b: fine_tone,
    })?;
    let channel = |b: &mut ProgramBuilder, base: f32| -> Result<NodeId, ProgramError> {
        let c = b.add(Op::Constant {
            domain,
            value: base,
        })?;
        b.add(Op::Mul { a: c, b: bright })
    };
    let (r, g, bl) = (
        channel(&mut b, 0.34)?,
        channel(&mut b, 0.315)?,
        channel(&mut b, 0.28)?,
    );
    let color = b.add(Op::Color { r, g, b: bl })?;
    // Grains stand up to 0.3 mm proud; roughness 0.88 to 0.98.
    let bump = crate::ramp(&mut b, grains, [0.0, 0.6], [0.0003, 0.0])?;
    let rough = crate::ramp(&mut b, fine, [-1.0, 1.0], [0.88, 0.98])?;
    let zero = b.add(Op::Constant { domain, value: 0.0 })?;
    let detail = b.add(Op::Vector3 {
        x: bump,
        y: rough,
        z: zero,
    })?;
    Ok([b.clone().finish_value(color)?, b.finish_value(detail)?])
}

/// A glazed-brick composite with its mortar finished.
#[derive(Clone, Debug, PartialEq)]
pub struct Finished {
    /// Linear Rec. 709 base color.
    pub base_color: TypedRaster,
    /// Height in meters above the mortar's datum.
    pub height: Raster,
    /// Specular roughness.
    pub specular_roughness: Raster,
    /// Surface material: [`GLAZE`], [`BODY`] or [`MORTAR`], by the texel's
    /// dominant contributor.
    pub material: TypedRaster,
}

/// Why [`finish`] failed.
#[derive(Clone, Debug, PartialEq)]
pub enum FinishError {
    /// The composite lacks an output of [`program`], or holds the wrong type.
    Outputs,
    /// Reading the composite failed.
    Composite(CompositeError),
    /// Building the mortar fields failed.
    Program(ProgramError),
    /// Realizing the mortar failed.
    Typed(TypedError),
    /// The distance transform failed.
    Raster(RasterError),
}

impl fmt::Display for FinishError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Outputs => f.write_str("the composite is not of the glazed-brick program"),
            Self::Composite(e) => e.fmt(f),
            Self::Program(e) => e.fmt(f),
            Self::Typed(e) => e.fmt(f),
            Self::Raster(e) => e.fmt(f),
        }
    }
}

impl core::error::Error for FinishError {}

/// Tools and textures the mortar of `realized`, composited over `domain`
/// with [`program`], [`instance`] and [`BACKGROUND`].
///
/// Where the bricks leave a texel uncovered (`1 − cover`), the flat
/// background gives way to sanded mortar: grains about a millimeter across,
/// each its own tone, in lighter and darker patches, rough, and tooled into
/// a concave joint whose surface rises from 1 mm below the datum in the
/// joint's middle to 1.5 mm above it where it meets the bricks, a few
/// millimeters under their faces. The distance across the joint comes from
/// an exact distance transform of the bricks' cover.
///
/// # Errors
///
/// [`FinishError`] when `realized` is not a glazed-brick composite over
/// `domain`.
pub fn finish(realized: &Realized, domain: Domain) -> Result<Finished, FinishError> {
    let get = |name: &str| {
        realized
            .output(name)
            .map_err(FinishError::Composite)?
            .ok_or(FinishError::Outputs)
    };
    let (color, height, rough, material, cover) = (
        get("base_color")?,
        get("height")?,
        get("specular_roughness")?,
        get("material")?,
        get("cover")?,
    );
    let (Storage::F32x3(color), Storage::F32(height), Storage::F32(rough), Storage::F32(cover)) = (
        color.storage(),
        height.storage(),
        rough.storage(),
        cover.storage(),
    ) else {
        return Err(FinishError::Outputs);
    };
    let (w, h) = (cover.width(), cover.height());
    let realization = Realization::period(domain, w, h).map_err(FinishError::Raster)?;
    let texels = |program: &ValueProgram| -> Result<Vec<[f32; 3]>, FinishError> {
        let raster = realize_value(program, realization).map_err(FinishError::Typed)?;
        match raster.storage() {
            Storage::F32x3(r) => Ok(r.values().to_vec()),
            _ => Err(FinishError::Outputs),
        }
    };
    let [mortar_color, mortar_detail] = mortar_fields(domain).map_err(FinishError::Program)?;
    let (mortar_color, mortar_detail) = (texels(&mortar_color)?, texels(&mortar_detail)?);
    // Distance from each texel to the nearest brick, in meters.
    let across = DistanceTransform { threshold: 0.5 }
        .apply(cover)
        .map_err(FinishError::Raster)?;

    let flat_color = match BACKGROUND[0] {
        Value::Vector3(v) => v,
        _ => Vec3::ZERO,
    };
    let flat_height = BACKGROUND[1].component(0).unwrap_or(0.0);
    let flat_rough = BACKGROUND[2].component(0).unwrap_or(0.0);
    let half_joint = 0.5 * JOINT;
    let n = mortar_color.len();
    let mut out_color = Vec::with_capacity(n);
    let mut out_height = Vec::with_capacity(n);
    let mut out_rough = Vec::with_capacity(n);
    for i in 0..n {
        let share = 1.0 - cover.values()[i].clamp(0.0, 1.0);
        // The tooled joint: concave, 1.5 mm up at the bricks, 1 mm down in
        // the middle.
        let s = (1.0 - across.values()[i] / half_joint).clamp(0.0, 1.0);
        let [bump, grit, _] = mortar_detail[i];
        let joint = -0.001 + 0.0025 * s * s + bump;
        let mortar = Vec3::from_array(mortar_color[i]);
        let c = Vec3::from_array(color.values()[i]) + (mortar - flat_color) * share;
        out_color.push(c.to_array());
        out_height.push(height.values()[i] + (joint - flat_height) * share);
        out_rough.push(rough.values()[i] + (grit - flat_rough) * share);
    }
    Ok(Finished {
        base_color: TypedRaster::new(
            PortType::Color(Primaries::Rec709),
            Storage::F32x3(like(cover, out_color)?),
        )
        .map_err(FinishError::Typed)?,
        height: like(cover, out_height)?,
        specular_roughness: like(cover, out_rough)?,
        material,
    })
}

/// `values` on `grid`'s texels.
fn like<T: Copy>(grid: &Raster, values: Vec<T>) -> Result<Raster<T>, FinishError> {
    Raster::from_values(
        grid.width(),
        grid.height(),
        grid.origin(),
        grid.texel(),
        grid.edge(),
        values,
    )
    .map_err(FinishError::Raster)
}
