// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Worked metal: copper cladding and riveted mild-steel plate.
//!
//! Both are metals in OpenPBR's sense: `base_metalness` 1, `base_color`
//! their reflectance at normal incidence (F0), calibrated against
//! tabulated values ([`super::calibration::METALS`]), and no diffuse lobe.
//! `specular_color` (OpenPBR's edge tint) stays white: the tabulated F0
//! alone is what the sources give. A brushed finish is anisotropic
//! roughness: brushing scores fine grooves along the domain's `x`, which
//! spread reflections across them, so the rougher axis, OpenPBR's
//! `geometry_tangent`, is `+y`, and highlights stretch across the grooves.
//!
//! The sheet's construction is relief the weathering modules read: a
//! standing seam is a ridge water runs down beside, a lap is a ledge it
//! drips from, a rivet head is a boss it runs around. On a wrapping grid
//! every spacing snaps to the nearest that divides the period, and no seam
//! lies on the wrap, so the sheet tiles.

use alloc::vec;
use alloc::vec::Vec;

use dapple_field::{CellOutput, Value};
use dapple_material::module::{
    Args, Context, Interface, Module, ModuleError, ModuleId, Output, OutputDecl, OutputKind,
    Outputs,
};
use dapple_material::{Aux, Channel, Grid, Material, Param};
use dapple_raster::Raster;
use glam::{Vec2, Vec3};

use super::calibration::{COPPER_F0, IRON_F0, MILL_SCALE_F0};
use super::{
    color, color_map, coverage, fbm, fit, fraction, integer, meters, realize_scalar,
    scalar_channel, seed, smoothstep,
};
use crate::glazed_brick::METAL;

fn material_output(doc: &'static str) -> Vec<OutputDecl> {
    vec![OutputDecl {
        name: "material".into(),
        kind: OutputKind::Material,
        doc: doc.into(),
    }]
}

/// `spacing` snapped to divide `extent` a whole number of times on a
/// wrapping grid; unchanged otherwise, and 0 (none) stays 0.
fn fit_spacing(grid: Grid, extent: f32, spacing: f32) -> f32 {
    if spacing <= 0.0 || grid.edge != dapple_field::Edge::Wrap {
        return spacing;
    }
    extent / libm::roundf(extent / spacing).max(1.0)
}

/// The grid's extent in meters.
fn extent(grid: Grid) -> Vec2 {
    #[expect(clippy::cast_precision_loss, reason = "grid sizes are small")]
    let e = grid.texel * Vec2::new(grid.width as f32, grid.height as f32);
    e
}

/// Signed offset of `v` from the nearest of the lines at `(k + 0.5) ·
/// spacing` (relative to `origin`), and that line's index: lines sit
/// between the wrap's, never on it.
fn nearest_line(v: f32, origin: f32, spacing: f32) -> (f32, i64) {
    let u = (v - origin) / spacing - 0.5;
    let k = libm::roundf(u);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "line indices are small whole numbers"
    )]
    let index = k as i64;
    ((u - k) * spacing, index)
}

/// A worked surface's finish, shared by the metals: fine grooves along `x`
/// as deep as `brushing`, which tint and roughen the metal along their
/// length, and a slow mottle.
struct Worked {
    /// Per texel: groove strength in about `[-1, 1]`.
    grooves: Raster,
    /// Per texel: slow tonal variation in about `[-1, 1]`.
    mottle: Raster,
}

impl Worked {
    fn of(grid: Grid, args: &Args) -> Result<Self, ModuleError> {
        // Grooves: long along x, a few millimeters across.
        let grooves = realize_scalar(grid, |b, d| {
            fbm(b, d, [3.0, 260.0], args.seed("grooves"), 4)
        })?;
        let mottle = realize_scalar(grid, |b, d| fbm(b, d, [5.0, 5.0], args.seed("mottle"), 4))?;
        Ok(Self { grooves, mottle })
    }
}

/// What a metal module binds, per texel or constant.
struct Metal {
    color: Vec<Vec3>,
    roughness: Vec<f32>,
    height: Vec<f32>,
    brushing: f32,
    surface: u32,
}

impl Metal {
    /// The material: metallic, its F0 as `base_color`, brushed where
    /// `brushing` is above zero.
    fn material(&self, grid: Grid) -> Result<Material, ModuleError> {
        let mut m = Material::new(grid);
        m.set_param(Param::BaseColor, color_map(grid, &self.color)?)?;
        m.set_param(Param::BaseMetalness, Channel::Constant(Value::Scalar(1.0)))?;
        m.set_param(
            Param::SpecularRoughness,
            scalar_channel(grid, &self.roughness)?,
        )?;
        if self.brushing > 0.0 {
            // Grooves along x roughen across them: the rough axis is +y.
            m.set_param(
                Param::SpecularRoughnessAnisotropy,
                Channel::Constant(Value::Scalar(0.8 * self.brushing)),
            )?;
            m.set_param(
                Param::GeometryTangent,
                Channel::Constant(Value::Vector3(Vec3::Y)),
            )?;
        }
        m.set_aux(Aux::Height, scalar_channel(grid, &self.height)?)?;
        m.set_aux(Aux::Surface, Channel::Constant(Value::Id(self.surface)))?;
        Ok(m)
    }
}

/// Copper cladding: sheets of bright copper (F0 from
/// [`COPPER_F0`]) joined by standing seams `seams` apart that stand
/// `relief` proud, and by cross laps `courses` apart where each sheet's
/// folded lower edge laps over the sheet below, each sheet a little wavy
/// between its seams (oil canning, `waviness`). `brushing` scores it along
/// `x`, from a rolled mill finish (0) to a heavy brush (1).
///
/// Copper tarnishes and weathers to verdigris: see [`super::Patina`].
#[derive(Copy, Clone, Debug, Default)]
pub struct Copper;

impl Module for Copper {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId::new("dapple_library.copper", 1),
            doc: "copper cladding: standing seams and cross laps".into(),
            params: vec![
                color(
                    "color",
                    COPPER_F0,
                    "the metal's reflectance at normal incidence",
                ),
                fraction("roughness", 0.2, "specular roughness of the sheet"),
                fraction(
                    "brushing",
                    0.3,
                    "how deeply it is brushed along x; 0 for a rolled finish",
                ),
                meters(
                    "seams",
                    [0.0, 2.0],
                    0.5,
                    "spacing of the standing seams; 0 for none",
                ),
                meters(
                    "courses",
                    [0.0, 2.0],
                    0.5,
                    "spacing of the cross laps; 0 for none",
                ),
                meters(
                    "relief",
                    [0.0, 0.03],
                    0.008,
                    "how far a standing seam stands proud",
                ),
                meters(
                    "waviness",
                    [0.0, 0.003],
                    0.0004,
                    "the sheets' oil canning between seams",
                ),
                integer("surface", [0, 1 << 16], METAL, "surface identity"),
                seed(),
            ],
            inputs: vec![],
            outputs: material_output("the copper"),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let size = extent(grid);
        let seams = fit_spacing(grid, size.x, args.scalar("seams"));
        let courses = fit_spacing(grid, size.y, args.scalar("courses"));
        let relief = args.scalar("relief");
        let brushing = args.scalar("brushing");
        let worked = Worked::of(grid, args)?;
        let wave = realize_scalar(grid, |b, d| fbm(b, d, [3.0, 2.0], args.seed("wave"), 3))?;
        let f0 = args.color("color");
        let rough = args.scalar("roughness");
        let waviness = args.scalar("waviness");
        // Edges a texel wide, so steps are band-limited.
        let soft = grid.texel.max_element();

        let n = grid.len();
        let mut metal = Metal {
            color: Vec::with_capacity(n),
            roughness: Vec::with_capacity(n),
            height: Vec::with_capacity(n),
            brushing,
            surface: args.integer("surface"),
        };
        for i in 0..n {
            let p = grid.center(i);
            // The standing seam: a double lock with rounded shoulders,
            // 12 mm across, the pan turning up into it.
            let (seam, near_seam) = if seams > 0.0 {
                let (dx, _) = nearest_line(p.x, grid.origin.x, seams);
                let d = dx.abs();
                let a = 0.006;
                let s = (d / a).min(1.0);
                let ridge = libm::sqrtf((1.0 - s * s * s * s).max(0.0));
                let ridge = ridge * smoothstep(a + soft, a - soft, d);
                let upturn = smoothstep(a + 0.01, a, d);
                (relief * (ridge + 0.06 * upturn), smoothstep(0.06, 0.01, d))
            } else {
                (0.0, 0.0)
            };
            // The cross lap: the sheet above laps over the one below by
            // its thickness, and its folded lower edge is a 6 mm bead.
            let lap = if courses > 0.0 {
                // The sheet above laps over the one below by its thickness
                // and falls away toward the next lap up, where it tucks
                // under; its folded lower edge is a 6 mm bead.
                let (dy, _) = nearest_line(p.y, grid.origin.y, courses);
                let bead_t = (dy / 0.006).clamp(0.0, 1.0);
                let bead = libm::sqrtf(libm::sinf(core::f32::consts::PI * bead_t).max(0.0));
                lap_step(dy, courses, 0.0012, soft) + 0.0018 * bead
            } else {
                0.0
            };
            let bow = waviness * canning(wave.values()[i]) * (1.0 - near_seam);
            let g = worked.grooves.values()[i];
            let m = worked.mottle.values()[i];
            let tone = 1.0 + 0.035 * m + 0.05 * brushing * g;
            metal.color.push((f0 * tone).min(Vec3::ONE));
            metal
                .roughness
                .push((rough + 0.05 * m + 0.08 * brushing * g.abs()).clamp(0.02, 1.0));
            metal.height.push(seam + lap + bow);
        }
        let m = metal.material(grid)?;
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}

/// A lap's sawtooth, given the signed offset `d` from the nearest lap
/// line (as [`nearest_line`] gives it) and the laps' `spacing`: the piece
/// past the line stands `rise` proud there and falls away to nothing at the
/// next line, where it tucks under the next piece; the step is eased over
/// `ease` either side of the line.
fn lap_step(d: f32, spacing: f32, rise: f32, ease: f32) -> f32 {
    rise * (smoothstep(-ease, ease, d) - d / spacing)
}

/// Oil canning: the sheet bows a little one way or the other.
fn canning(w: f32) -> f32 {
    (1.6 * w).clamp(-1.0, 1.0)
}

/// Mild steel plate, riveted: courses of plate `plates` high, each lapping
/// over the course below by its thickness, with a vertical lap per course,
/// staggered course to course, and a row of domed rivet heads
/// `rivet_pitch` apart along every lap; 0 for a plain plate. The steel's
/// F0 is iron's ([`IRON_F0`]); `brushing` grinds it along `x`, and
/// `mill_scale` leaves that share of it under the hot-rolled oxide
/// (magnetite, F0 [`MILL_SCALE_F0`]), flaking at its edges.
///
/// Steel rusts: see [`super::Rust`]; and takes paint: [`super::Paint`].
#[derive(Copy, Clone, Debug, Default)]
pub struct Steel;

impl Module for Steel {
    fn interface(&self) -> Interface {
        Interface {
            id: ModuleId::new("dapple_library.steel", 1),
            doc: "riveted mild-steel plate".into(),
            params: vec![
                color(
                    "color",
                    IRON_F0,
                    "the metal's reflectance at normal incidence",
                ),
                fraction("roughness", 0.3, "specular roughness of bare steel"),
                fraction(
                    "brushing",
                    0.0,
                    "how deeply it is ground along x; 0 for none",
                ),
                fraction(
                    "mill_scale",
                    0.6,
                    "share of the plate still under mill scale",
                ),
                meters(
                    "plates",
                    [0.0, 2.0],
                    0.5,
                    "the height of a course of plate; 0 for none",
                ),
                meters(
                    "rivet_pitch",
                    [0.0, 0.5],
                    0.0625,
                    "rivet spacing along the laps; 0 for none",
                ),
                integer("surface", [0, 1 << 16], METAL, "surface identity"),
                seed(),
            ],
            inputs: vec![],
            outputs: material_output("the steel"),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let size = extent(grid);
        let plates = fit_spacing(grid, size.y, args.scalar("plates"));
        let pitch = fit_spacing(grid, size.x.min(size.y), args.scalar("rivet_pitch"));
        // A course is two plates wide, so its vertical lap staggers.
        let width = fit_spacing(grid, size.x, 2.0 * plates);
        let brushing = args.scalar("brushing");
        let worked = Worked::of(grid, args)?;
        let soft = grid.texel.max_element();
        let rough = args.scalar("roughness");
        let f0 = args.color("color");

        // Mill scale: broad blotches, broken up at every scale down to the
        // flakes it sheds, so its edges are ragged and feathered.
        let scale_score = realize_scalar(grid, |b, d| {
            let blotch = fbm(b, d, [9.0, 9.0], args.seed("scale"), 7)?;
            let flake = b.add(dapple_field::program::Op::Cellular {
                domain: d,
                frequency: fit(d, [110.0, 110.0]),
                jitter: 0.9,
                seed: args.seed("flakes"),
                output: CellOutput::CellValue,
            })?;
            let flake = b.add(dapple_field::program::Op::Remap {
                input: flake,
                from: [0.0, 1.0],
                to: [-0.14, 0.14],
            })?;
            b.add(dapple_field::program::Op::Add {
                a: blotch,
                b: flake,
            })
        })?;
        let scale = coverage(&scale_score, args.scalar("mill_scale"), 0.03)?;

        let n = grid.len();
        let mut metal = Metal {
            color: Vec::with_capacity(n),
            roughness: Vec::with_capacity(n),
            height: Vec::with_capacity(n),
            brushing,
            surface: args.integer("surface"),
        };
        // A rivet head: a 19 mm dome standing 6 mm proud.
        let (head, crown) = (0.0095, 0.006);
        let dome = |d: f32| {
            let s = (d / head).min(1.0);
            crown * libm::sqrtf((1.0 - s * s).max(0.0)) * smoothstep(head + soft, head - soft, d)
        };
        for i in 0..n {
            let p = grid.center(i);
            let mut h = 0.0;
            if plates > 0.0 {
                // The course above laps over the one below by the plate's
                // 3 mm, its edge eased by a millimeter.
                let (dy, course) = nearest_line(p.y, grid.origin.y, plates);
                let up = if dy >= 0.0 { dy } else { dy + plates };
                let course = if dy >= 0.0 { course } else { course - 1 };
                h += lap_step(dy, plates, 0.003, 0.001);
                // The vertical lap, staggered: the plate to its right laps
                // over the one to its left.
                // Half a plate apart, and neither on the wrap.
                let shift = if course.rem_euclid(2) == 0 { -0.2 } else { 0.3 };
                let (dx, _) = nearest_line(p.x, grid.origin.x + shift * width, width);
                h += lap_step(dx, width, 0.002, 0.001);
                if pitch > 0.0 {
                    // Rivets along the lap, 22 mm above its edge, and up the
                    // vertical lap, 22 mm to its right.
                    let (rx, _) = nearest_line(p.x, grid.origin.x, pitch);
                    let row = Vec2::new(rx, dy - 0.022).length();
                    let (ry, _) = nearest_line(p.y, grid.origin.y + 0.5 * pitch, pitch);
                    let clear = up > 0.05 && up < plates - 0.03;
                    let column = if clear {
                        Vec2::new(dx - 0.022, ry).length()
                    } else {
                        f32::MAX
                    };
                    h += dome(row.min(column));
                }
            }
            let g = worked.grooves.values()[i];
            let m = worked.mottle.values()[i];
            let s = scale.values()[i];
            let tone = 1.0 + 0.04 * m + 0.06 * brushing * g;
            let bare = (f0 * tone).min(Vec3::ONE);
            let oxide = MILL_SCALE_F0 * (1.0 + 0.12 * m);
            metal.color.push(bare.lerp(oxide, s));
            let r = (rough + 0.06 * m + 0.1 * brushing * g.abs()).clamp(0.02, 1.0);
            metal.roughness.push(r + (0.45 - r) * s);
            // Mill scale is some 50 µm thick.
            metal.height.push(h + 0.00005 * s);
        }
        let m = metal.material(grid)?;
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}
