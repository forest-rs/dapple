// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Oak: bark and wood.

use dapple_field::program::{
    FieldProgram, NodeId, Op, ProgramBuilder, ProgramError, SolidProgram, ValueProgram,
};
use dapple_field::{Affine3, Basis, CellOutput, Domain, Domain3, FractalParams, WaveShape};
use glam::{Mat3, Vec3};

use crate::ramp;

/// The seed the oak programs derive their noise seeds from.
pub const SEED: u64 = 7;

/// Oak bark height over a planar domain, in `[0, 1]`: plates separated by
/// wavy vertical fissures, their size and height varied across the tile, with
/// shallow secondary cracks in patches and fine grain on top. On a periodic
/// domain of period 1 the tile repeats; `dapple_bake`'s `oak_bark` recipe
/// carries the same graph.
///
/// # Errors
///
/// Propagates [`ProgramError`]s, which a valid `domain` does not produce.
pub fn bark_height(domain: Domain) -> Result<FieldProgram, ProgramError> {
    let mut b = ProgramBuilder::new();
    let fractal = |b: &mut ProgramBuilder, frequency, seed| {
        b.add(Op::Fractal {
            basis: Basis::Gradient,
            domain,
            frequency,
            seed,
            params: FractalParams::default(),
        })
    };
    // Plates stretched along y, their sizes varied by a broad warp that
    // compresses some columns and widens others. Border distance is in cell
    // units.
    let cells = b.add(Op::Cellular {
        domain,
        frequency: [7.0, 2.0],
        jitter: 1.0,
        seed: SEED,
        output: CellOutput::Border,
    })?;
    let stretch_x = fractal(&mut b, [1.0, 2.0], SEED + 22)?;
    let stretch_y = fractal(&mut b, [1.0, 2.0], SEED + 23)?;
    let cells = b.add(Op::Warp {
        input: cells,
        dx: stretch_x,
        dy: stretch_y,
        amount: 0.08,
    })?;
    let wobble_x = fractal(&mut b, [3.0, 6.0], SEED + 1)?;
    let wobble_y = fractal(&mut b, [3.0, 6.0], SEED + 2)?;
    let wavy = b.add(Op::Warp {
        input: cells,
        dx: wobble_x,
        dy: wobble_y,
        amount: 0.03,
    })?;
    // Plates flatten between 0.14 and 0.32 cells in from their fissures, so
    // fissure width and plate height vary across the tile.
    let flat = fractal(&mut b, [3.0, 2.0], SEED + 24)?;
    let flat = b.add(Op::Remap {
        input: flat,
        from: [-1.0, 1.0],
        to: [0.14, 0.32],
    })?;
    let plates = b.add(Op::Min { a: wavy, b: flat })?;
    let plates = b.add(Op::Remap {
        input: plates,
        from: [0.0, 0.32],
        to: [0.0, 0.85],
    })?;
    // Shallow secondary cracks split some plates, in patches.
    let cracks = b.add(Op::Cellular {
        domain,
        frequency: [16.0, 5.0],
        jitter: 1.0,
        seed: SEED + 20,
        output: CellOutput::Border,
    })?;
    let cracks = b.add(Op::Warp {
        input: cracks,
        dx: wobble_x,
        dy: wobble_y,
        amount: 0.02,
    })?;
    let cracks = ramp(&mut b, cracks, [0.0, 0.1], [0.4, 1.0])?;
    let patches = fractal(&mut b, [2.0, 2.0], SEED + 21)?;
    let patches = ramp(&mut b, patches, [0.1, 0.4], [1.0, 0.0])?;
    let cracks = b.add(Op::Max {
        a: cracks,
        b: patches,
    })?;
    let plates = b.add(Op::Min {
        a: plates,
        b: cracks,
    })?;
    let grain = b.add(Op::Fractal {
        basis: Basis::Gradient,
        domain,
        frequency: [24.0, 6.0],
        seed: SEED + 3,
        params: FractalParams {
            octaves: 4,
            ..FractalParams::default()
        },
    })?;
    let grain = b.add(Op::Remap {
        input: grain,
        from: [-1.0, 1.0],
        to: [0.0, 0.15],
    })?;
    let height = b.add(Op::Add {
        a: plates,
        b: grain,
    })?;
    b.finish(height)
}

/// Linear bark color over the bark height: damp dark fissures, brown bark
/// that weathers grey on exposed plate tops, lichen on some plates, and
/// large-scale mottling. The base color of `dapple_bake`'s `oak_bark` recipe
/// is the same graph.
///
/// # Errors
///
/// Propagates [`ProgramError`]s, for example a `height` on another domain.
pub fn bark_color(domain: Domain, height: &FieldProgram) -> Result<ValueProgram, ProgramError> {
    let mut b = ProgramBuilder::new();
    let height = b.import(height)?;
    let color = |b: &mut ProgramBuilder, rgb: [f32; 3]| {
        let [r, g, bl] = rgb.map(|value| b.add(Op::Constant { domain, value }));
        b.add(Op::Color {
            r: r?,
            g: g?,
            b: bl?,
        })
    };
    let noise = |b: &mut ProgramBuilder, frequency, seed, octaves| {
        b.add(Op::Fractal {
            basis: Basis::Gradient,
            domain,
            frequency,
            seed,
            params: FractalParams {
                octaves,
                ..FractalParams::default()
            },
        })
    };
    let fissure = color(&mut b, [0.016, 0.012, 0.01])?;
    let bark = color(&mut b, [0.11, 0.08, 0.058])?;
    let weathered = color(&mut b, [0.27, 0.24, 0.2])?;
    let lichen_color = color(&mut b, [0.2, 0.24, 0.12])?;

    // Damp fissure depths darken toward the bark's brown on the plates.
    let depth = ramp(&mut b, height, [0.0, 0.5], [0.0, 1.0])?;
    let color = b.add(Op::Mix {
        a: fissure,
        b: bark,
        t: depth,
    })?;
    // Exposed plate tops weather grey, in patches.
    let top = ramp(&mut b, height, [0.6, 0.95], [0.0, 1.0])?;
    let patches = noise(&mut b, [4.0, 3.0], SEED + 4, 4)?;
    let patches = ramp(&mut b, patches, [0.0, 0.5], [0.0, 0.9])?;
    let weather = b.add(Op::Mul { a: top, b: patches })?;
    let color = b.add(Op::Mix {
        a: color,
        b: weathered,
        t: weather,
    })?;
    // Sparse lichen on the plates, never in the fissures.
    let plate = ramp(&mut b, height, [0.45, 0.8], [0.0, 1.0])?;
    let lichen = noise(&mut b, [3.0, 2.0], SEED + 5, 5)?;
    let lichen = ramp(&mut b, lichen, [0.1, 0.35], [0.0, 0.8])?;
    let lichen = b.add(Op::Mul {
        a: plate,
        b: lichen,
    })?;
    let color = b.add(Op::Mix {
        a: color,
        b: lichen_color,
        t: lichen,
    })?;
    // Large-scale mottling of the whole tile.
    let mottle = noise(&mut b, [2.0, 2.0], SEED + 6, 3)?;
    let mottle = b.add(Op::Remap {
        input: mottle,
        from: [-1.0, 1.0],
        to: [0.7, 1.3],
    })?;
    let color = b.add(Op::Mul {
        a: color,
        b: mottle,
    })?;
    b.finish_value(color)
}

/// The wood's linear color: growth rings with noise-perturbed radii, dark
/// latewood closing each ring, earlywood pores, medullary rays and slow
/// mottling.
///
/// The trunk runs along the solid domain's z axis through the origin, in
/// meters. Returns the color node, a [`dapple_field::PortType::Color`].
///
/// Rings and rays repeat through [`Op::Wave`]s, so they are band-limited by
/// the footprint: as they near and pass its size they spread and fade to
/// their mean rather than alias. In the solid, a wave filters along the
/// phase's three-dimensional gradient, so a face that cuts the rings at a
/// grazing angle (a cathedral's crown) is filtered as if it crossed them;
/// [`wood_face_color`] filters a sawn face by its own gradients instead.
///
/// # Errors
///
/// Propagates [`ProgramError`]s from building the graph.
pub fn wood_color(b: &mut ProgramBuilder) -> Result<NodeId, ProgramError> {
    wood(b, Cut::Solid)
}

/// The wood on a sawn face: [`wood_color`] on the plane through `origin`
/// spanned by `u` and `v`, the solid steps per unit of the face's x and y,
/// as [`Op::Slice`] would cut it. Returns a planar color node over
/// [`Domain::Plane`].
///
/// Each solid quantity (the ring radius, the angle around the trunk, the
/// noises) is sliced onto the face before it is shaped, so the rings and
/// rays are filtered by their gradients across the face itself: a
/// cathedral's crown, where the face runs along the rings, keeps its broad
/// bands at a footprint that blends the tight rings at its flanks. At a
/// point footprint it is [`wood_color`] sliced, up to rounding.
///
/// # Errors
///
/// Propagates [`ProgramError`]s from building the graph, such as a
/// non-finite plane.
pub fn wood_face_color(
    b: &mut ProgramBuilder,
    origin: [f32; 3],
    u: [f32; 3],
    v: [f32; 3],
) -> Result<NodeId, ProgramError> {
    wood(b, Cut::Face { origin, u, v })
}

/// Where the wood is shaped: in the solid, or on a sawn face.
#[derive(Copy, Clone, Debug)]
enum Cut {
    Solid,
    Face {
        origin: [f32; 3],
        u: [f32; 3],
        v: [f32; 3],
    },
}

impl Cut {
    /// Solid quantity `solid` where the wood is shaped.
    fn lift(self, b: &mut ProgramBuilder, solid: NodeId) -> Result<NodeId, ProgramError> {
        match self {
            Self::Solid => Ok(solid),
            Self::Face { origin, u, v } => b.add(Op::Slice {
                input: solid,
                origin,
                u,
                v,
                domain: Domain::Plane,
            }),
        }
    }

    /// A constant where the wood is shaped.
    fn constant(self, b: &mut ProgramBuilder, value: f32) -> Result<NodeId, ProgramError> {
        match self {
            Self::Solid => b.add(Op::Constant3 {
                domain: Domain3::Space,
                value,
            }),
            Self::Face { .. } => b.add(Op::Constant {
                domain: Domain::Plane,
                value,
            }),
        }
    }
}

fn wood(b: &mut ProgramBuilder, cut: Cut) -> Result<NodeId, ProgramError> {
    let domain = Domain3::Space;
    let color = |b: &mut ProgramBuilder, rgb: [f32; 3]| {
        let [r, g, bl] = rgb.map(|value| cut.constant(b, value));
        b.add(Op::Color {
            r: r?,
            g: g?,
            b: bl?,
        })
    };
    let fractal = |b: &mut ProgramBuilder, basis, frequency, seed, octaves| {
        b.add(Op::Fractal3 {
            basis,
            domain,
            frequency,
            seed,
            params: FractalParams {
                octaves,
                ..FractalParams::default()
            },
        })
    };
    // One octave of solid noise, where the wood is shaped.
    let noise = |b: &mut ProgramBuilder, basis, frequency, seed| {
        let n = b.add(Op::Noise3 {
            basis,
            domain,
            frequency,
            seed,
        })?;
        cut.lift(b, n)
    };
    // One octave of solid noise on an oblique lattice, where the wood is
    // shaped: `frequency` along the solid's axes as `noise` takes it, but
    // its lattice turned off them, so features that end on lattice planes
    // (a ray's fleck, cut where its extent runs out) do not end together
    // along a plane across the trunk.
    let oblique = |b: &mut ProgramBuilder, basis, frequency: [f32; 3], seed| {
        let n = b.add(Op::Noise3 {
            basis,
            domain,
            frequency: [1.0, 1.0, 1.0],
            seed,
        })?;
        let n = b.add(Op::Transform3 {
            input: n,
            transform: Affine3 {
                matrix: Mat3::from_axis_angle(Vec3::new(0.36, 0.8, 0.48).normalize(), 0.9)
                    * Mat3::from_diagonal(Vec3::from(frequency)),
                translation: Vec3::ZERO,
            },
        })?;
        cut.lift(b, n)
    };
    let position = b.add(Op::Position3)?;
    let x = b.add(Op::Component {
        input: position,
        index: 0,
    })?;
    let y = b.add(Op::Component {
        input: position,
        index: 1,
    })?;
    // The distance from the trunk's axis: the position's length with its z
    // flattened onto the cross-section, which keeps the gradient the rings'
    // footprint needs in closed form.
    let length = b.add(Op::Length { input: position })?;
    let radius = b.add(Op::Transform3 {
        input: length,
        transform: Affine3 {
            matrix: Mat3::from_diagonal(Vec3::new(1.0, 1.0, 0.0)),
            translation: Vec3::ZERO,
        },
    })?;
    // The trunk tapers: each ring is a cone, 1.6 cm wider per meter lower.
    let z = b.add(Op::Component {
        input: position,
        index: 2,
    })?;
    let taper = b.add(Op::Remap {
        input: z,
        from: [0.0, 1.0],
        to: [0.0, 0.016],
    })?;
    let radius = b.add(Op::Add {
        a: radius,
        b: taper,
    })?;
    // No trunk tapers evenly: over a meter or two its rings swell and
    // pinch by up to ±4 mm, so a face along it crosses them now faster,
    // now slower, now not at all, and its cathedrals come irregularly.
    let sweep = b.add(Op::Noise3 {
        basis: Basis::Gradient,
        domain,
        frequency: [0.6, 0.6, 0.8],
        seed: SEED + 20,
    })?;
    let sweep = b.add(Op::Remap {
        input: sweep,
        from: [-1.0, 1.0],
        to: [-0.004, 0.004],
    })?;
    let radius = b.add(Op::Add {
        a: radius,
        b: sweep,
    })?;

    // Rings wander by up to ±8 mm, around the trunk more than along it,
    // so a face along the trunk crosses them steadily enough to draw
    // cathedrals rather than eyes.
    let wobble = fractal(b, Basis::Gradient, [6.0, 6.0, 1.2], SEED + 10, 4)?;
    let wobble = b.add(Op::Remap {
        input: wobble,
        from: [-1.0, 1.0],
        to: [-0.008, 0.008],
    })?;
    let radius = b.add(Op::Add {
        a: radius,
        b: wobble,
    })?;
    // The tree grew fast when young and slower as it aged: rings 6.8 mm
    // apart at the pith, 5.5 mm at 14 cm and 4.5 mm at 30 cm (the phase is
    // 148·r + 120·r²).
    let linear = b.add(Op::Remap {
        input: radius,
        from: [0.0, 1.0],
        to: [0.0, 148.0],
    })?;
    let square = b.add(Op::Mul {
        a: radius,
        b: radius,
    })?;
    let square = b.add(Op::Remap {
        input: square,
        from: [0.0, 1.0],
        to: [0.0, 120.0],
    })?;
    let rings = b.add(Op::Add {
        a: linear,
        b: square,
    })?;
    let rings = cut.lift(b, rings)?;
    let zero = cut.constant(b, 0.0)?;
    // Each year opens with a band of large pores and darkens toward its
    // latewood, which ends sharply at the next year's pores.
    let late = b.add(Op::Wave {
        input: rings,
        shift: zero,
        shape: WaveShape::Sawtooth,
        from: [0.4, 0.95],
        to: [0.0, 1.0],
    })?;
    let early = b.add(Op::Wave {
        input: rings,
        shift: zero,
        shape: WaveShape::Sawtooth,
        from: [0.12, 0.3],
        to: [1.0, 0.0],
    })?;

    // Large earlywood pores, drawn out along the grain. The noise is
    // symmetric about 0 and fades to 0 as the pores pass the footprint, so
    // a ramp symmetric about 0 fades them to their mean (0.45), not to a
    // floor.
    let pores = noise(b, Basis::Value, [700.0, 700.0, 60.0], SEED + 11)?;
    let pores = ramp(b, pores, [-0.5, 0.5], [0.15, 0.75])?;
    let pores = b.add(Op::Mul { a: pores, b: early })?;

    // Rays: thin radial ribbons around the trunk. Each set is a stripe
    // around every whole sector of the (gently swaying) angle; a noise
    // field along the ribbon varies its width, so rays pinch off into
    // flecks of varying length and thickness, and a sparse extent field
    // keeps only some of them, fading in and out. Broad rays are few and
    // pale; fine rays are many and faint. Extent varies faster around the
    // trunk than rays are spaced, so neighboring rays start and stop
    // independently, and both fields are on oblique lattices, so flecks do
    // not end together across a face. A face nearly along the rays
    // (quarter-sawn) shows them as long tapering flakes.
    let angle = b.add(Op::Atan2 { y, x })?;
    let sway = fractal(b, Basis::Gradient, [12.0, 12.0, 10.0], SEED + 12, 3)?;
    let sway = b.add(Op::Remap {
        input: sway,
        from: [-1.0, 1.0],
        to: [-0.06, 0.06],
    })?;
    let angle = b.add(Op::Add { a: angle, b: sway })?;
    let angle = cut.lift(b, angle)?;
    let rays = |b: &mut ProgramBuilder,
                count: f32,
                edge: f32,
                width_frequency: [f32; 3],
                extent_frequency: [f32; 3],
                extent_cut: [f32; 2],
                each: Option<[f32; 3]>,
                seed: u64|
     -> Result<NodeId, ProgramError> {
        // A phase keeps the sawn test planes, which pass through angle 0,
        // off a ray.
        let sectors = b.add(Op::Remap {
            input: angle,
            from: [0.0, core::f32::consts::TAU],
            to: [0.37, count + 0.37],
        })?;
        // Width varies along each ribbon: within `edge` of a whole sector
        // at most, thinning to nothing where the noise is low.
        let width = oblique(b, Basis::Gradient, width_frequency, seed)?;
        let width = b.add(Op::Remap {
            input: width,
            from: [-1.0, 1.0],
            to: [-edge, edge],
        })?;
        let ray = b.add(Op::Wave {
            input: sectors,
            shift: width,
            shape: WaveShape::Triangle,
            from: [edge, edge * 0.6],
            to: [0.0, 1.0],
        })?;
        let extent = oblique(b, Basis::Value, extent_frequency, seed + 1)?;
        let extent = ramp(b, extent, extent_cut, [0.0, 1.0])?;
        let ray = b.add(Op::Mul { a: ray, b: extent })?;
        // Within the extent, each ray comes and goes by itself: a field
        // that varies about as fast as rays are spaced, ramped
        // symmetrically about its mean so that, faded under a coarse
        // footprint, it leaves every ray at half strength rather than none.
        let Some(each) = each else {
            return Ok(ray);
        };
        let each = oblique(b, Basis::Value, each, seed + 5)?;
        let each = ramp(b, each, [-0.15, 0.15], [0.0, 1.0])?;
        b.add(Op::Mul { a: ray, b: each })
    };
    // Broad rays run centimeters along the trunk, so their extent varies
    // slowly along it (25 cells a meter) and around it (60 a meter), and
    // each ray within it comes and goes on 150 cells a meter around, about
    // one per ray. They are about a millimeter wide at a 15 cm radius.
    let broad = rays(
        b,
        131.0,
        0.12,
        [60.0, 60.0, 30.0],
        [60.0, 60.0, 25.0],
        [0.05, 0.3],
        Some([150.0, 150.0, 25.0]),
        SEED + 16,
    )?;
    let fine = rays(
        b,
        409.0,
        0.06,
        [160.0, 160.0, 90.0],
        [900.0, 900.0, 80.0],
        [0.0, 0.2],
        None,
        SEED + 18,
    )?;
    let faint = cut.constant(b, 0.45)?;
    let fine = b.add(Op::Mul { a: fine, b: faint })?;
    let ray = b.add(Op::Max { a: broad, b: fine })?;

    let early_color = color(b, [0.42, 0.27, 0.14])?;
    let late_color = color(b, [0.27, 0.16, 0.075])?;
    let pore_color = color(b, [0.1, 0.055, 0.025])?;
    let ray_color = color(b, [0.54, 0.39, 0.225])?;
    let wood = b.add(Op::Mix {
        a: early_color,
        b: late_color,
        t: late,
    })?;
    let wood = b.add(Op::Mix {
        a: wood,
        b: pore_color,
        t: pores,
    })?;
    let wood = b.add(Op::Mix {
        a: wood,
        b: ray_color,
        t: ray,
    })?;
    let mottle = fractal(b, Basis::Gradient, [4.0, 4.0, 1.0], SEED + 14, 3)?;
    let mottle = cut.lift(b, mottle)?;
    let mottle = b.add(Op::Remap {
        input: mottle,
        from: [-1.0, 1.0],
        to: [0.85, 1.15],
    })?;
    b.add(Op::Mul { a: wood, b: mottle })
}

/// The wood's color as three solid channel programs (linear red, green and
/// blue), for evaluation at chart samples with [`SolidProgram::eval_chart`].
///
/// # Errors
///
/// Propagates [`ProgramError`]s from building the graph.
pub fn wood_channels() -> Result<[SolidProgram; 3], ProgramError> {
    let channel = |index| {
        let mut b = ProgramBuilder::new();
        let color = wood_color(&mut b)?;
        let channel = b.add(Op::Component {
            input: color,
            index,
        })?;
        b.finish_solid(channel)
    };
    Ok([channel(0)?, channel(1)?, channel(2)?])
}
