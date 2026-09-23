// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Glazed brick: keyed bricks whose identity drives every channel.
//!
//! Unlike [`brick`](crate::brick), which is one field program, this material
//! is an element set plus a surface program (`dapple_elements`):
//!
//! - [`layout`]: running-bond bricks on a 1 m tile, each with a glaze tone
//!   and glaze thickness derived from its key;
//! - [`program`]: what one brick looks like at a point: its beveled shape,
//!   its glaze (tone, thickness, pooling toward the lower edge) and chips
//!   near its edges that expose the ceramic body;
//! - [`instance`] binds the program to the layout's attributes, and
//!   [`BACKGROUND`] is the mortar.
//!
//! One brick's key decides its tone, its thickness and where its chips
//! fall, so moving a brick moves its appearance with it, and the chips in
//! the color, height, roughness and material outputs always agree. The
//! `material` output separates the glaze ([`GLAZE`]) from the exposed body
//! ([`BODY`]) and the mortar ([`MORTAR`]); which brick owns a texel is a
//! separate raster.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use dapple_elements::{
    AttributeDecl, Binding, ContractError, ElementError, ElementSet, InstanceId, LayoutId, Node,
    ProgramInstance, RunningBond, Scope, SurfaceBuilder, SurfaceProgram,
};
use dapple_field::program::{Op, ProgramBuilder, ProgramError, ValueProgram};
use dapple_field::{Basis, Domain, FractalParams, PortType, Primaries, Value};
use glam::Vec3;

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

/// The bricks of a 1 m tile, with key-derived glaze tones and thicknesses.
///
/// # Errors
///
/// Never for the fixed bond.
pub fn layout() -> Result<ElementSet, ElementError> {
    let domain = Domain::periodic(1, 1).expect("a unit period");
    bond(domain).elements(schema(), |key, _| {
        vec![
            Value::Scalar(key.unit(1)),
            Value::Scalar(0.4 + 0.6 * key.unit(2)),
        ]
    })
}

/// The chip noise: a planar fractal sampled in brick-local meters.
fn chip_noise() -> Result<ValueProgram, ProgramError> {
    let mut b = ProgramBuilder::new();
    let n = b.add(Op::Fractal {
        basis: Basis::Gradient,
        domain: Domain::Plane,
        frequency: [90.0, 90.0],
        seed: 23,
        params: FractalParams {
            octaves: 3,
            ..FractalParams::default()
        },
    })?;
    b.finish_value(n)
}

/// The glazed-brick surface program.
///
/// Inputs, in binding order: `tone`, `thickness` and `chip_seed` per
/// element, `half_size` per element, `local` and `edge` per sample, and
/// `bevel` per material. Outputs: `base_color` (linear Rec. 709),
/// `height` (meters above the mortar), `specular_roughness`, `material`
/// ([`GLAZE`] or [`BODY`]) per sample, and `glaze_tint` per element.
///
/// # Errors
///
/// Never: the program is fixed and satisfies its contract.
pub fn program() -> Result<SurfaceProgram, ContractError> {
    let mut b = SurfaceBuilder::new("dapple_library.glazed_brick");
    let tone = b.input("tone", PortType::Scalar, Scope::Element)?;
    let thickness = b.input("thickness", PortType::Scalar, Scope::Element)?;
    let seed = b.input("chip_seed", PortType::Scalar, Scope::Element)?;
    let half = b.input("half_size", PortType::Vector2, Scope::Element)?;
    let local = b.input("local", PortType::Vector2, Scope::Sample)?;
    let edge = b.input("edge", PortType::Scalar, Scope::Sample)?;
    let bevel = b.input("bevel", PortType::Scalar, Scope::Material)?;
    let noise = b.resource(
        "chip_noise",
        chip_noise().map_err(|_| ContractError::UnknownReference)?,
    );

    let c = |b: &mut SurfaceBuilder, v: f32| b.constant(Value::Scalar(v));
    let rgb = |b: &mut SurfaceBuilder, r: f32, g: f32, bl: f32| {
        b.constant(Value::Vector3(Vec3::new(r, g, bl)))
    };
    let zero = c(&mut b, 0.0);
    let one = c(&mut b, 1.0);

    // The brick's beveled shape: 0 at its boundary, 1 once `bevel` inside.
    let shape = b.add(Node::SmoothStep {
        edge0: zero,
        edge1: bevel,
        x: edge,
    })?;

    // Glaze pools toward the lower edge (−y): its depth runs from half to
    // one and a half times the brick's thickness.
    let hy = b.add(Node::Component {
        input: half,
        index: 1,
    })?;
    let ly = b.add(Node::Component {
        input: local,
        index: 1,
    })?;
    let neg_hy = b.add(Node::Sub(zero, hy))?;
    let lower = b.add(Node::SmoothStep {
        edge0: hy,
        edge1: neg_hy,
        x: ly,
    })?;
    let half_c = c(&mut b, 0.5);
    let pooling = b.add(Node::Add(half_c, lower))?;
    let pool = b.add(Node::Mul(thickness, pooling))?;

    // The glaze's tone is the brick's own, darkening where it pools.
    let green = rgb(&mut b, 0.035, 0.20, 0.14);
    let blue = rgb(&mut b, 0.030, 0.12, 0.28);
    let tint = b.add(Node::Mix {
        a: green,
        b: blue,
        t: tone,
    })?;
    let dim = c(&mut b, 0.45);
    let pooled = b.add(Node::Mul(dim, pool))?;
    let lighten = c(&mut b, 1.25);
    let shade = b.add(Node::Sub(lighten, pooled))?;
    let glaze = b.add(Node::Mul(tint, shade))?;

    // Chips: a noise sampled where this brick's seed puts it, pushed up
    // near the brick's edges so they gather along edges and at corners.
    let sx = c(&mut b, 37.0);
    let sy = c(&mut b, 53.0);
    let ox = b.add(Node::Mul(seed, sx))?;
    let oy = b.add(Node::Mul(seed, sy))?;
    let offset = b.add(Node::Vector2(ox, oy))?;
    let at = b.add(Node::Add(local, offset))?;
    let n = b.add(Node::Sample {
        resource: noise,
        at,
    })?;
    let reach = c(&mut b, 0.012);
    let near = b.add(Node::SmoothStep {
        edge0: reach,
        edge1: zero,
        x: edge,
    })?;
    let push = c(&mut b, 0.55);
    let pushed = b.add(Node::Mul(near, push))?;
    let chance = b.add(Node::Add(n, pushed))?;
    let lo = c(&mut b, 0.68);
    let hi = c(&mut b, 0.74);
    let chip = b.add(Node::SmoothStep {
        edge0: lo,
        edge1: hi,
        x: chance,
    })?;
    let glazed = b.add(Node::Sub(one, chip))?;

    let body = rgb(&mut b, 0.56, 0.36, 0.24);
    let base_color = b.add(Node::Mix {
        a: body,
        b: glaze,
        t: glazed,
    })?;

    // Height in meters: a 20 mm brick face over the mortar, glaze up to
    // 1.5 mm on top, chips 3 mm into the body.
    let face = c(&mut b, 0.02);
    let brick = b.add(Node::Mul(shape, face))?;
    let glaze_depth = c(&mut b, 0.001);
    let coat = b.add(Node::Mul(pool, glaze_depth))?;
    let coat = b.add(Node::Mul(coat, glazed))?;
    let chip_depth = c(&mut b, 0.003);
    let dent = b.add(Node::Mul(chip, chip_depth))?;
    let raised = b.add(Node::Add(brick, coat))?;
    let height = b.add(Node::Sub(raised, dent))?;

    let rough_body = c(&mut b, 0.8);
    let rough_glaze = c(&mut b, 0.12);
    let roughness = b.add(Node::Mix {
        a: rough_body,
        b: rough_glaze,
        t: glazed,
    })?;

    let glaze_id = b.constant(Value::Id(GLAZE));
    let body_id = b.constant(Value::Id(BODY));
    let material = b.add(Node::Select {
        condition: chip,
        a: body_id,
        b: glaze_id,
    })?;

    let color = PortType::Color(Primaries::Rec709);
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
    Ok(b.finish())
}

/// The program bound to [`layout`]'s attributes with a 6 mm bevel.
///
/// # Errors
///
/// Never for [`program`]'s contract.
pub fn instance(program: Arc<SurfaceProgram>) -> Result<ProgramInstance, ContractError> {
    ProgramInstance::new(
        InstanceId::named("dapple_library.glazed_brick"),
        program,
        vec![
            Binding::Attribute(String::from("glaze_tone")),
            Binding::Attribute(String::from("glaze_thickness")),
            Binding::ElementRandom(3),
            Binding::HalfSize,
            Binding::LocalPosition,
            Binding::EdgeDistance,
            Binding::Constant(Value::Scalar(0.006)),
        ],
    )
}

/// The mortar, in [`program`]'s output order.
pub const BACKGROUND: [Value; 5] = [
    Value::Vector3(Vec3::new(0.42, 0.40, 0.37)),
    Value::Scalar(0.0),
    Value::Scalar(0.9),
    Value::Id(MORTAR),
    Value::Vector3(Vec3::ZERO),
];
