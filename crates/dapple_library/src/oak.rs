// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Oak: bark and wood.

use dapple_field::program::{
    FieldProgram, NodeId, Op, ProgramBuilder, ProgramError, SolidProgram, ValueProgram,
};
use dapple_field::{Basis, CellOutput, Domain, Domain3, FractalParams};

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
/// # Errors
///
/// Propagates [`ProgramError`]s from building the graph.
pub fn wood_color(b: &mut ProgramBuilder) -> Result<NodeId, ProgramError> {
    let domain = Domain3::Space;
    let color = |b: &mut ProgramBuilder, rgb: [f32; 3]| {
        let [r, g, bl] = rgb.map(|value| b.add(Op::Constant3 { domain, value }));
        b.add(Op::Color {
            r: r?,
            g: g?,
            b: bl?,
        })
    };
    let noise = |b: &mut ProgramBuilder, basis, frequency, seed, octaves| {
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
    let position = b.add(Op::Position3)?;
    let x = b.add(Op::Component {
        input: position,
        index: 0,
    })?;
    let y = b.add(Op::Component {
        input: position,
        index: 1,
    })?;
    let xy = b.add(Op::Vector2 { x, y })?;
    let radius = b.add(Op::Length { input: xy })?;
    // The trunk tapers: each ring is a cone, 3 cm wider per meter lower.
    let z = b.add(Op::Component {
        input: position,
        index: 2,
    })?;
    let taper = b.add(Op::Remap {
        input: z,
        from: [0.0, 1.0],
        to: [0.0, 0.03],
    })?;
    let radius = b.add(Op::Add {
        a: radius,
        b: taper,
    })?;

    // Rings 5.5 mm apart, their radii wandering by up to ±8 mm.
    let wobble = noise(b, Basis::Gradient, [6.0, 6.0, 4.0], SEED + 10, 4)?;
    let wobble = b.add(Op::Remap {
        input: wobble,
        from: [-1.0, 1.0],
        to: [-0.008, 0.008],
    })?;
    let radius = b.add(Op::Add {
        a: radius,
        b: wobble,
    })?;
    let rings = b.add(Op::Remap {
        input: radius,
        from: [0.0, 1.0],
        to: [0.0, 180.0],
    })?;
    let phase = b.add(Op::Fract { input: rings })?;
    // Each year opens with a band of large pores and darkens toward its
    // latewood, which ends sharply at the next year's pores.
    let late = ramp(b, phase, [0.4, 0.95], [0.0, 1.0])?;
    let early = ramp(b, phase, [0.12, 0.3], [1.0, 0.0])?;

    // Large earlywood pores, drawn out along the grain.
    let pores = b.add(Op::Noise3 {
        basis: Basis::Value,
        domain,
        frequency: [700.0, 700.0, 60.0],
        seed: SEED + 11,
    })?;
    let pores = ramp(b, pores, [0.1, 0.5], [0.3, 0.9])?;
    let pores = b.add(Op::Mul { a: pores, b: early })?;

    // Rays: thin radial ribbons around the trunk. Each set is a sawtooth in
    // the (swaying) angle with a ribbon at every tooth edge; a noise field
    // along the ribbon varies its width, so rays pinch off into flecks of
    // varying length and thickness, and a sparse extent field keeps only
    // some of them. Broad rays are few and pale; fine rays are many and
    // faint. Extent varies faster around the trunk than rays are spaced,
    // so neighboring rays start and stop independently.
    let angle = b.add(Op::Atan2 { y, x })?;
    let sway = noise(b, Basis::Gradient, [12.0, 12.0, 10.0], SEED + 12, 3)?;
    let sway = b.add(Op::Remap {
        input: sway,
        from: [-1.0, 1.0],
        to: [-0.12, 0.12],
    })?;
    let angle = b.add(Op::Add { a: angle, b: sway })?;
    let half = b.add(Op::Constant3 { domain, value: 0.5 })?;
    let rays = |b: &mut ProgramBuilder,
                count: f32,
                edge: f32,
                width_frequency: [f32; 3],
                extent_frequency: [f32; 3],
                extent_cut: [f32; 2],
                seed: u64|
     -> Result<NodeId, ProgramError> {
        // A phase keeps the sawn test planes, which pass through angle 0,
        // off a ray edge.
        let sectors = b.add(Op::Remap {
            input: angle,
            from: [0.0, core::f32::consts::TAU],
            to: [0.37, count + 0.37],
        })?;
        let sector = b.add(Op::Fract { input: sectors })?;
        let offset = b.add(Op::Sub { a: sector, b: half })?;
        let offset = b.add(Op::Abs { input: offset })?;
        // Width varies along each ribbon: the ribbon is `edge` wide at
        // most, thinning to nothing where the noise is low.
        let width = b.add(Op::Noise3 {
            basis: Basis::Gradient,
            domain,
            frequency: width_frequency,
            seed,
        })?;
        let width = b.add(Op::Remap {
            input: width,
            from: [-1.0, 1.0],
            to: [-edge, edge],
        })?;
        let offset = b.add(Op::Add {
            a: offset,
            b: width,
        })?;
        let ray = ramp(b, offset, [0.5 - edge, 0.5 - edge * 0.6], [0.0, 1.0])?;
        let extent = b.add(Op::Noise3 {
            basis: Basis::Value,
            domain,
            frequency: extent_frequency,
            seed: seed + 1,
        })?;
        let extent = ramp(b, extent, extent_cut, [0.0, 1.0])?;
        b.add(Op::Mul { a: ray, b: extent })
    };
    let broad = rays(
        b,
        131.0,
        0.1,
        [90.0, 90.0, 45.0],
        [300.0, 300.0, 45.0],
        [0.1, 0.3],
        SEED + 16,
    )?;
    let fine = rays(
        b,
        409.0,
        0.06,
        [160.0, 160.0, 90.0],
        [900.0, 900.0, 80.0],
        [0.0, 0.2],
        SEED + 18,
    )?;
    let faint = b.add(Op::Constant3 {
        domain,
        value: 0.45,
    })?;
    let fine = b.add(Op::Mul { a: fine, b: faint })?;
    let ray = b.add(Op::Max { a: broad, b: fine })?;

    let early_color = color(b, [0.42, 0.27, 0.14])?;
    let late_color = color(b, [0.27, 0.16, 0.075])?;
    let pore_color = color(b, [0.1, 0.055, 0.025])?;
    let ray_color = color(b, [0.52, 0.37, 0.21])?;
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
    let mottle = noise(b, Basis::Gradient, [4.0, 4.0, 1.0], SEED + 14, 3)?;
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
