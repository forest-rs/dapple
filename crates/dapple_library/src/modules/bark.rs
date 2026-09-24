// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Bark of European beech, silver birch, Scots pine and Norway spruce.
//!
//! Each bark is one tile of bark (`x` around the stem, `y` up it) and takes
//! the stem's `girth` (circumference, meters) and the tile's `height` on
//! the trunk (meters above the ground) as parameters, so bark can follow a
//! tree's growth: where a real bark changes with age or height (birch's
//! dark, fissured base; pine's thick lower plates giving way to thin
//! orange flakes; plates that grow with girth), a parameter drives the
//! change continuously rather than a separate recipe per stage. A tile
//! stands for one height, so it tiles along and up the stem; a stem is
//! textured by instances at several heights, blended by the host.

use alloc::vec;
use alloc::vec::Vec;

use dapple_field::hash::{hash, unit_f32};
use dapple_field::program::Op;
use dapple_field::{CellOutput, Value};
use dapple_material::module::{
    Args, Context, Interface, Module, ModuleError, ModuleId, Output, OutputDecl, OutputKind,
    Outputs, ParamDecl,
};
use dapple_material::{Aux, Channel, Grid, Material, Param};
use glam::{Vec2, Vec3};

use super::{
    color, color_map, fbm, fit, fraction, meters, realize_scalar, scalar_channel, seed, smoothstep,
};
use crate::glazed_brick::BARK;
use crate::masonry::extent;

/// The girth and height parameters every bark takes.
fn stem(girth: f32) -> [ParamDecl; 2] {
    [
        meters("girth", [0.05, 8.0], girth, "the stem's circumference"),
        meters("height", [0.0, 50.0], 1.3, "the tile's height on the trunk"),
    ]
}

fn bark_output() -> Vec<OutputDecl> {
    vec![OutputDecl {
        name: "material".into(),
        kind: OutputKind::Material,
        doc: "the bark, surface identity BARK".into(),
    }]
}

/// Per-texel bark, before it becomes a material.
struct Bark {
    color: Vec<Vec3>,
    roughness: Vec<f32>,
    height: Vec<f32>,
}

impl Bark {
    fn with_capacity(n: usize) -> Self {
        Self {
            color: Vec::with_capacity(n),
            roughness: Vec::with_capacity(n),
            height: Vec::with_capacity(n),
        }
    }

    fn push(&mut self, color: Vec3, roughness: f32, height: f32) {
        self.color.push(color.max(Vec3::ZERO).min(Vec3::ONE));
        self.roughness.push(roughness.clamp(0.0, 1.0));
        self.height.push(height);
    }

    fn material(&self, grid: Grid) -> Result<Outputs, ModuleError> {
        let mut m = Material::new(grid);
        m.set_param(Param::BaseColor, color_map(grid, &self.color)?)?;
        m.set_param(
            Param::SpecularRoughness,
            scalar_channel(grid, &self.roughness)?,
        )?;
        m.set_aux(Aux::Height, scalar_channel(grid, &self.height)?)?;
        m.set_aux(Aux::Surface, Channel::Constant(Value::Id(BARK)))?;
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}

/// Features scattered one per lattice cell, periodic over the grid:
/// calls `f` with each of the nine cells around `p` (its hash) and `p`'s
/// offset from that cell's jittered center.
struct Lattice {
    cells: [i64; 2],
    size: Vec2,
    seed: u64,
    jitter: f32,
}

impl Lattice {
    /// Cells about `spacing` apart over `grid`, a whole number per period.
    fn new(grid: Grid, spacing: Vec2, seed: u64, jitter: f32) -> Self {
        let e = extent(grid);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "cell counts are small and positive"
        )]
        let cells = [
            libm::roundf(e.x / spacing.x).max(1.0) as i64,
            libm::roundf(e.y / spacing.y).max(1.0) as i64,
        ];
        #[expect(clippy::cast_precision_loss, reason = "cell counts are small")]
        let size = e / Vec2::new(cells[0] as f32, cells[1] as f32);
        Self {
            cells,
            size,
            seed,
            jitter,
        }
    }

    fn around(&self, p: Vec2, mut f: impl FnMut(u64, Vec2)) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "cell indices near the grid are small"
        )]
        let (col, row) = (
            libm::floorf(p.x / self.size.x) as i64,
            libm::floorf(p.y / self.size.y) as i64,
        );
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (cx, cy) = (col + dx, row + dy);
                let h = hash(
                    self.seed,
                    &[
                        cx.rem_euclid(self.cells[0]).cast_unsigned(),
                        cy.rem_euclid(self.cells[1]).cast_unsigned(),
                    ],
                );
                let j = |k: u64| self.jitter * (unit_f32(hash(h, &[k])) - 0.5);
                #[expect(clippy::cast_precision_loss, reason = "cell indices are small")]
                let center = self.size * Vec2::new(cx as f32 + 0.5 + j(1), cy as f32 + 0.5 + j(2));
                f(h, p - center);
            }
        }
    }
}

/// A uniform value in `[0, 1)` for the feature hashed `h` and `k`.
fn draw(h: u64, k: u64) -> f32 {
    unit_f32(hash(h, &[k]))
}

/// Coverage of an ellipse of half size `half` at offset `q`, antialiased
/// over a texel `t` wide and area-preserving below a texel: a lenticel
/// thinner than a texel fades rather than vanishing or widening.
fn blob(q: Vec2, half: Vec2, t: f32) -> f32 {
    let drawn = half.max(Vec2::splat(0.5 * t));
    let keep = (half.x * half.y) / (drawn.x * drawn.y);
    let k = (q / drawn).length();
    let sd = (k - 1.0) * drawn.min_element();
    keep * smoothstep(0.5 * t, -0.5 * t, sd)
}

/// A diamond (rhombus) of half diagonals `half`, antialiased over `t`.
fn diamond(q: Vec2, half: Vec2, t: f32) -> f32 {
    let sd = (libm::fabsf(q.x) / half.x + libm::fabsf(q.y) / half.y - 1.0) * half.min_element();
    smoothstep(0.5 * t, -0.5 * t, sd)
}

/// A cellular field over `grid`: cells `size` meters (along, up the stem).
fn cells(
    grid: Grid,
    size: Vec2,
    seed: u64,
    output: CellOutput,
) -> Result<dapple_raster::Raster, ModuleError> {
    let f = [1.0 / size.x, 1.0 / size.y];
    realize_scalar(grid, |b, d| {
        b.add(Op::Cellular {
            domain: d,
            frequency: fit(d, f),
            jitter: 1.0,
            seed,
            output,
        })
    })
}

/// Cells `size` meters (around, up the stem), their borders bent by a
/// displacement `warp` meters strong varying over `wander` meters, so plates
/// have wavy, irregular outlines rather than straight Voronoi edges.
fn warped_cells(
    grid: Grid,
    size: Vec2,
    seed: u64,
    output: CellOutput,
    warp: f32,
    wander: f32,
) -> Result<dapple_raster::Raster, ModuleError> {
    let f = [1.0 / size.x, 1.0 / size.y];
    let w = [1.0 / wander, 1.0 / wander];
    realize_scalar(grid, |b, d| {
        let cells = b.add(Op::Cellular {
            domain: d,
            frequency: fit(d, f),
            jitter: 1.0,
            seed,
            output,
        })?;
        let dx = fbm(b, d, w, seed ^ 0x5157, 3)?;
        let dy = fbm(b, d, w, seed ^ 0x5158, 3)?;
        b.add(Op::Warp {
            input: cells,
            dx,
            dy,
            amount: warp,
        })
    })
}

fn position(grid: Grid, i: usize) -> Vec2 {
    grid.center(i) - grid.origin
}

/// European beech (*Fagus sylvatica*): smooth, thin, silver-grey bark,
/// finely mottled in lighter and darker greys, with short horizontal
/// lenticels, occasional pale crustose lichen patches, faint horizontal
/// wrinkles that deepen with girth, and green algae that gathers low on
/// the stem.
///
/// Calibrated ([`super::calibration`]): the mean albedo at breast height
/// matches measured smooth grey bark.
#[derive(Copy, Clone, Debug, Default)]
pub struct Beech;

impl Module for Beech {
    fn interface(&self) -> Interface {
        let [girth, height] = stem(1.2);
        Interface {
            id: ModuleId::new("dapple_library.beech_bark", 1),
            doc: "European beech bark".into(),
            params: vec![
                girth,
                height,
                color("color", super::calibration::BEECH_COLOR, "the bark's grey"),
                fraction("lenticels", 0.55, "how many lenticels"),
                fraction("lichen", 0.35, "how many lichen patches"),
                fraction("algae", 0.3, "green algae low on the stem"),
                fraction("roughness", 0.62, "specular roughness"),
                seed(),
            ],
            inputs: vec![],
            outputs: bark_output(),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let (girth, height) = (args.scalar("girth"), args.scalar("height"));
        let mottle = realize_scalar(grid, |b, d| fbm(b, d, [7.0, 5.0], args.seed("mottle"), 5))?;
        let fine = realize_scalar(grid, |b, d| fbm(b, d, [45.0, 30.0], args.seed("fine"), 3))?;
        let fleck = realize_scalar(grid, |b, d| {
            fbm(b, d, [140.0, 110.0], args.seed("fleck"), 2)
        })?;
        let wrinkle = realize_scalar(grid, |b, d| fbm(b, d, [4.0, 60.0], args.seed("wrinkle"), 4))?;
        let algae_n = realize_scalar(grid, |b, d| fbm(b, d, [3.0, 4.0], args.seed("algae"), 5))?;
        let ragged = realize_scalar(grid, |b, d| fbm(b, d, [22.0, 22.0], args.seed("ragged"), 4))?;
        let lenticels = Lattice::new(grid, Vec2::new(0.03, 0.022), args.seed("lenticels"), 0.9);
        let lichens = Lattice::new(grid, Vec2::new(0.16, 0.16), args.seed("lichens"), 0.8);
        let base = args.color("color");
        let (lent, lichen, rough) = (
            args.scalar("lenticels"),
            args.scalar("lichen"),
            args.scalar("roughness"),
        );
        // Algae reaches further on old, low, damp stems.
        let algae = args.scalar("algae") * smoothstep(4.0, 0.0, height);
        // Wrinkles deepen with girth.
        let wrinkling = smoothstep(0.3, 2.5, girth);
        let green = Vec3::new(0.13, 0.16, 0.08);
        let crust = Vec3::new(0.34, 0.35, 0.3);
        let t = grid.texel.min_element();
        let mut bark = Bark::with_capacity(grid.len());
        for i in 0..grid.len() {
            let p = position(grid, i);
            let mut dots = 0.0_f32;
            lenticels.around(p, |h, q| {
                if draw(h, 3) < lent {
                    let half = Vec2::new(0.003 + 0.004 * draw(h, 4), 0.0008 + 0.0004 * draw(h, 5));
                    dots = dots.max(blob(q, half, t));
                }
            });
            // Crustose lichen: round patches with ragged edges and a pale
            // rim.
            let mut patch = 0.0_f32;
            let mut rim = 0.0_f32;
            lichens.around(p, |h, q| {
                if draw(h, 3) < lichen {
                    let r = 0.012 + 0.035 * draw(h, 4);
                    let d = q.length() - r * (1.0 + 0.6 * ragged.values()[i]);
                    patch = patch.max(smoothstep(0.5 * t, -0.5 * t, d));
                    rim = rim.max(smoothstep(0.004, 0.0, d.abs()) * smoothstep(t, -t, d));
                }
            });
            let m = mottle.values()[i];
            let blotch = smoothstep(-0.35, 0.35, m);
            let c =
                base * (0.84 + 0.26 * blotch + 0.08 * fine.values()[i] + 0.04 * fleck.values()[i]);
            let a = smoothstep(0.1, 0.6, algae_n.values()[i] + 0.4 * algae) * algae;
            let c = c + (green - c) * a;
            let c = c + (crust * (1.0 + 0.15 * rim) - c) * (0.6 * patch);
            let c = c + (Vec3::new(0.09, 0.075, 0.06) - c) * (0.75 * dots);
            let w = wrinkle.values()[i];
            bark.push(
                c,
                rough + 0.2 * a + 0.2 * patch,
                0.0005 * w * wrinkling + 0.0001 * m + 0.00015 * dots + 0.0002 * patch,
            );
        }
        bark.material(grid)
    }
}

/// Silver birch (*Betula pendula*): white, papery bark peeling in thin
/// horizontal strips, dotted with dark horizontal lenticels; lower down,
/// black diamond-shaped fissured scars, and at the base of an old stem
/// thick, blackish bark of long vertical plates between deep, irregular
/// fissures. The dark base reaches higher as the stem's girth grows, so
/// one parameter carries a birch from sapling to veteran.
///
/// Calibrated ([`super::calibration`]): the white bark (high on the stem,
/// clear of the base) matches the whitest measured silver birch bark, and
/// the breast-height mean falls inside the measured range.
#[derive(Copy, Clone, Debug, Default)]
pub struct Birch;

impl Birch {
    /// How high the rugged base reaches on a stem of `girth`, meters.
    #[must_use]
    pub fn rugged_height(girth: f32) -> f32 {
        (2.2 * (girth - 0.35)).max(0.0)
    }
}

impl Module for Birch {
    fn interface(&self) -> Interface {
        let [girth, height] = stem(0.9);
        Interface {
            id: ModuleId::new("dapple_library.birch_bark", 1),
            doc: "silver birch bark".into(),
            params: vec![
                girth,
                height,
                color("color", super::calibration::BIRCH_COLOR, "the white bark"),
                color(
                    "dark",
                    Vec3::new(0.035, 0.033, 0.03),
                    "lenticels, diamonds and the base",
                ),
                fraction("lenticels", 0.6, "how many lenticels"),
                fraction("roughness", 0.55, "specular roughness of the white bark"),
                seed(),
            ],
            inputs: vec![],
            outputs: bark_output(),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let (girth, height) = (args.scalar("girth"), args.scalar("height"));
        let top = Self::rugged_height(girth);
        // 1 at the base of the rugged zone, 0 above it; diamonds reach
        // about twice as high.
        let rugged = smoothstep(top + 0.4, top - 0.4, height);
        let diamonds = smoothstep(2.0 * top + 1.0, top, height);
        let strips = realize_scalar(grid, |b, d| fbm(b, d, [2.0, 28.0], args.seed("strips"), 4))?;
        let peel = realize_scalar(grid, |b, d| fbm(b, d, [3.0, 60.0], args.seed("peel"), 3))?;
        let patch = realize_scalar(grid, |b, d| fbm(b, d, [5.0, 3.0], args.seed("patch"), 4))?;
        let depth_n = realize_scalar(grid, |b, d| fbm(b, d, [6.0, 2.5], args.seed("depth"), 4))?;
        let grain = realize_scalar(grid, |b, d| fbm(b, d, [90.0, 40.0], args.seed("grain"), 3))?;
        let s = (girth / 1.5).clamp(0.4, 2.0);
        // Long vertical plates with wavy outlines.
        let plates = warped_cells(
            grid,
            Vec2::new(0.05 * s, 0.2 * s),
            args.seed("fissures"),
            CellOutput::Border,
            0.02 * s,
            0.12,
        )?;
        let lenticels = Lattice::new(grid, Vec2::new(0.06, 0.022), args.seed("lenticels"), 0.9);
        let diamond_cells = Lattice::new(grid, Vec2::new(0.14, 0.2), args.seed("diamonds"), 0.5);
        let (white, dark) = (args.color("color"), args.color("dark"));
        let (lent, rough) = (args.scalar("lenticels"), args.scalar("roughness"));
        let t = grid.texel.min_element();
        let mut bark = Bark::with_capacity(grid.len());
        let orange = Vec3::new(0.45, 0.25, 0.14);
        let depth = 0.006 + 0.018 * smoothstep(0.5, 3.0, girth);
        for i in 0..grid.len() {
            let p = position(grid, i);
            let mut dash = 0.0_f32;
            lenticels.around(p, |h, q| {
                if draw(h, 3) < lent {
                    let half = Vec2::new(0.006 + 0.02 * draw(h, 4), 0.0012 + 0.0012 * draw(h, 5));
                    dash = dash.max(blob(q, half, t));
                }
            });
            let mut dia = 0.0_f32;
            diamond_cells.around(p, |h, q| {
                if draw(h, 3) < 0.6 * diamonds {
                    let half = Vec2::new(0.025 + 0.04 * draw(h, 4), 0.02 + 0.03 * draw(h, 5));
                    dia = dia.max(diamond(q, half, t));
                }
            });
            // Papery strips of slightly different whites, some peeling to
            // the orange inner bark at their edges.
            let st = strips.values()[i];
            let c = white * (1.0 + 0.06 * st);
            let lift = smoothstep(0.55, 0.7, peel.values()[i]);
            let c = c + (orange - c) * (0.5 * lift);
            let dia = smoothstep(
                0.35,
                0.75,
                dia * (0.75 + 0.5 * peel.values()[i].abs() + 0.3 * st),
            );
            // The rugged base: blackish plates between fissures that deepen
            // and fade along their length; white still showing on some plate
            // tops toward the top of the zone.
            let f = plates.values()[i];
            let fissure_depth = 0.35 + 0.65 * smoothstep(-0.4, 0.4, depth_n.values()[i]);
            let fissure = smoothstep(0.14, 0.0, f) * fissure_depth;
            // In the transition, whole patches are dark or white rather than
            // everything half dark: the zone's edge is ragged.
            let dark_here = smoothstep(-0.08, 0.08, 1.7 * rugged - 0.6 + 0.7 * patch.values()[i]);
            let base = dark_here * (1.0 - 0.6 * smoothstep(0.2, 0.35, f) * (1.0 - rugged));
            let g = grain.values()[i];
            let plate_c = dark * (1.4 + 0.8 * g) * (1.0 - 0.6 * fissure);
            let c = c + (plate_c - c) * base;
            let c = c + (dark - c) * dia.max(0.85 * dash);
            // Plates are cracked across and rough on top.
            let plate_h = 0.0015 * g + 0.002 * smoothstep(0.0, 0.3, f);
            bark.push(
                c,
                rough + (0.9 - rough) * base.max(dia),
                base * (plate_h - depth * fissure) - 0.0015 * dia - 0.0002 * dash + 0.0002 * lift,
            );
        }
        bark.material(grid)
    }
}

/// Scots pine (*Pinus sylvestris*): high on the stem, thin orange bark
/// flaking in papery scales; low on the stem, thick grey-brown bark in
/// large, irregular, vertically elongated plates, built of layers that
/// scale off at their edges to show redder bark, between deep fissures
/// whose depth varies along them. The plated zone climbs with girth, and
/// plates and fissures grow with it.
///
/// Calibrated ([`super::calibration`]): the plated bark's mean albedo at
/// breast height matches measured Scots pine bark. The upper orange bark
/// is authored; no measurement of it was found.
#[derive(Copy, Clone, Debug, Default)]
pub struct ScotsPine;

impl ScotsPine {
    /// How high the plated lower bark reaches on a stem of `girth`,
    /// meters.
    #[must_use]
    pub fn plated_height(girth: f32) -> f32 {
        1.0 + 5.0 * girth
    }
}

impl Module for ScotsPine {
    fn interface(&self) -> Interface {
        let [girth, height] = stem(1.1);
        Interface {
            id: ModuleId::new("dapple_library.scots_pine_bark", 1),
            doc: "Scots pine bark".into(),
            params: vec![
                girth,
                height,
                color(
                    "color",
                    super::calibration::PINE_COLOR,
                    "the lower plates' grey-brown",
                ),
                color(
                    "upper",
                    Vec3::new(0.52, 0.2, 0.07),
                    "the upper bark's orange",
                ),
                fraction("roughness", 0.85, "specular roughness"),
                seed(),
            ],
            inputs: vec![],
            outputs: bark_output(),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let (girth, height) = (args.scalar("girth"), args.scalar("height"));
        let top = Self::plated_height(girth);
        let lower = smoothstep(top + 1.5, top - 1.5, height);
        let s = (girth / 1.2).clamp(0.3, 2.5);
        // Big plates, about three times as tall as wide, with wandering
        // outlines; a second, larger set of cells merges some neighbors
        // (their shared fissure fades), so plate sizes vary.
        let plates = warped_cells(
            grid,
            Vec2::new(0.08 * s, 0.24 * s),
            args.seed("plates"),
            CellOutput::Border,
            0.03 * s,
            0.2,
        )?;
        let merge = realize_scalar(grid, |b, d| fbm(b, d, [5.0, 2.0], args.seed("merge"), 3))?;
        let depth_n = realize_scalar(grid, |b, d| fbm(b, d, [9.0, 3.0], args.seed("depth"), 4))?;
        // The layers a plate is built of, flaking away at their edges.
        let layers = warped_cells(
            grid,
            Vec2::new(0.035, 0.05),
            args.seed("layers"),
            CellOutput::Border,
            0.008,
            0.05,
        )?;
        let scaled = realize_scalar(grid, |b, d| fbm(b, d, [14.0, 9.0], args.seed("scaled"), 3))?;
        let flakes = warped_cells(
            grid,
            Vec2::new(0.03, 0.02),
            args.seed("flakes"),
            CellOutput::Border,
            0.006,
            0.05,
        )?;
        let flake_tone = cells(
            grid,
            Vec2::new(0.03, 0.02),
            args.seed("flakes"),
            CellOutput::CellValue,
        )?;
        let patch = realize_scalar(grid, |b, d| fbm(b, d, [4.0, 3.0], args.seed("patch"), 4))?;
        let mottle = realize_scalar(grid, |b, d| fbm(b, d, [9.0, 9.0], args.seed("mottle"), 4))?;
        let grain = realize_scalar(grid, |b, d| fbm(b, d, [80.0, 50.0], args.seed("grain"), 3))?;
        let (grey, orange, rough) = (
            args.color("color"),
            args.color("upper"),
            args.scalar("roughness"),
        );
        let fissure_c = Vec3::new(0.045, 0.025, 0.016);
        let fresh = Vec3::new(0.34, 0.14, 0.07);
        let depth = 0.008 + 0.025 * smoothstep(0.4, 3.0, girth);
        let mut bark = Bark::with_capacity(grid.len());
        for i in 0..grid.len() {
            // Where this texel is plated: the zone, with a ragged boundary.
            let k = (lower + 0.35 * patch.values()[i]).clamp(0.0, 1.0);
            let k = smoothstep(0.3, 0.7, k);
            let m = mottle.values()[i];
            let g = grain.values()[i];
            // Lower bark: fissures fade where plates merge, and deepen and
            // shallow along their length.
            let f = plates.values()[i];
            let open = smoothstep(-0.2, 0.25, merge.values()[i]);
            let fdepth = open * (0.4 + 0.6 * smoothstep(-0.4, 0.4, depth_n.values()[i]));
            let fissure = smoothstep(0.15, 0.02, f) * fdepth;
            let wall = smoothstep(0.24, 0.1, f) * (1.0 - smoothstep(0.15, 0.02, f)) * fdepth;
            // Layers: where the top layer has scaled off, a step down to
            // the redder layer beneath, its broken edge lifted.
            let off = smoothstep(0.15, 0.35, scaled.values()[i]);
            // Layer edges show only around where the top layer is lifting.
            let lifting = smoothstep(0.0, 0.15, scaled.values()[i]) - off;
            let edge = smoothstep(0.08, 0.0, layers.values()[i]) * lifting.max(0.0);
            let lc = grey * (0.9 + 0.2 * m + 0.08 * g);
            let lc = lc + (fresh - lc) * (0.55 * off);
            let lc = lc + (grey * 1.25 - lc) * (0.5 * edge * (1.0 - off));
            let lc = lc + (fresh * 0.8 - lc) * (0.6 * wall);
            let lc = lc + (fissure_c - lc) * fissure;
            let lh = 0.003 * smoothstep(0.0, 0.3, f) - depth * fissure - 0.0015 * off
                + 0.0008 * edge
                + 0.0003 * g;
            // Upper bark: thin papery flakes, lighter where they lift.
            let fl = flakes.values()[i];
            let flake_edge = smoothstep(0.08, 0.0, fl);
            let uc = orange * (0.85 + 0.3 * flake_tone.values()[i]) * (1.0 + 0.1 * m);
            let uc = uc + (Vec3::new(0.6, 0.45, 0.35) - uc) * (0.35 * flake_edge);
            let uh = 0.0008 * smoothstep(0.0, 0.1, fl);
            bark.push(
                uc + (lc - uc) * k,
                rough + 0.1 * (1.0 - k) * flake_edge,
                uh + (lh - uh) * k,
            );
        }
        bark.material(grid)
    }
}

/// Norway spruce (*Picea abies*): thin, reddish-brown bark of small,
/// round-ish scales that overlap like shingles, each curling up at its
/// lower edge and shadowing the scale beneath, with the fresh redder bark
/// showing between; on old stems the scales grow and their tops weather
/// grey, and high on the stem the bark is smoother and redder.
///
/// This revises the spruce bark sylva ships (a recipe in its species
/// gallery): that one's linear base color, about (0.06, 0.04, 0.03), is
/// less than half the reflectance measured for spruce bark and lacks its
/// red cast; this module is calibrated to the measurement
/// ([`super::calibration`]).
#[derive(Copy, Clone, Debug, Default)]
pub struct Spruce;

impl Module for Spruce {
    fn interface(&self) -> Interface {
        let [girth, height] = stem(1.0);
        Interface {
            id: ModuleId::new("dapple_library.spruce_bark", 1),
            doc: "Norway spruce bark".into(),
            params: vec![
                girth,
                height,
                color(
                    "color",
                    super::calibration::SPRUCE_COLOR,
                    "the scales' reddish brown",
                ),
                fraction("roughness", 0.8, "specular roughness"),
                seed(),
            ],
            inputs: vec![],
            outputs: bark_output(),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let (girth, height) = (args.scalar("girth"), args.scalar("height"));
        let age = smoothstep(0.4, 2.5, girth);
        // Smooth, thin, red bark high on the stem.
        let crown = smoothstep(8.0, 20.0, height);
        let s = 1.0 + 0.8 * age - 0.4 * crown;
        let spacing = Vec2::new(0.028 * s, 0.03 * s);
        let scales = Lattice::new(grid, spacing, args.seed("scales"), 0.8);
        let weather_n =
            realize_scalar(grid, |b, d| fbm(b, d, [4.0, 3.0], args.seed("weather"), 4))?;
        let ragged = realize_scalar(grid, |b, d| fbm(b, d, [60.0, 60.0], args.seed("ragged"), 3))?;
        let grain = realize_scalar(grid, |b, d| fbm(b, d, [110.0, 70.0], args.seed("grain"), 3))?;
        let (base, rough) = (args.color("color"), args.scalar("roughness"));
        // The bark between scales: the scales' brown in shadow.
        let fissure = base * 0.5;
        // Weathered scale tops: the scales' own lightness, grey.
        let grey = Vec3::splat(base.dot(Vec3::new(0.2126, 0.7152, 0.0722)) * 1.15);
        let weathering = 0.45 * age * (1.0 - crown);
        let relief = 0.0015 + 0.003 * age * (1.0 - crown);
        let mut bark = Bark::with_capacity(grid.len());
        let t = grid.texel.min_element();
        for i in 0..grid.len() {
            let p = position(grid, i);
            let wobble = 1.0 + 0.12 * ragged.values()[i];
            // The topmost scale here: scales higher on the stem lie over
            // those below, so of the scales covering a texel the one whose
            // center is highest wins.
            let mut best: Option<(f32, u64, Vec2, Vec2)> = None;
            let mut shade = 0.0_f32;
            let mut covers = [(0_u64, Vec2::ZERO, 0.0_f32); 9];
            let mut count = 0;
            scales.around(p, |h, q| {
                let half =
                    spacing * Vec2::new(0.62 + 0.3 * draw(h, 4), 0.6 + 0.3 * draw(h, 5)) * wobble;
                let k = (q / half).length();
                covers[count] = (h, q, k);
                count += 1;
                if k < 1.0 && best.is_none_or(|b| -q.y > b.0) {
                    best = Some((-q.y, h, q, half));
                }
            });
            let (inside, curl, tone, hash_of) = match best {
                Some((_, h, q, half)) => {
                    // Lift toward the lower edge (local y = -1).
                    let v = (q.y / half.y).clamp(-1.0, 1.0);
                    let k = (q / half).length();
                    let rim = smoothstep(0.7, 1.0, k);
                    (
                        smoothstep(1.0, 1.0 - 2.0 * t / half.min_element(), k),
                        (0.5 - 0.5 * v) * (0.4 + 0.6 * rim),
                        draw(h, 6),
                        h,
                    )
                }
                None => (0.0, 0.0, 0.5, 0),
            };
            // The shadow of an overlapping scale's curled lower edge, cast
            // just below it onto this one.
            for &(h, q, k) in &covers[..count] {
                if h != hash_of && q.y > 0.0 && (1.0..1.35).contains(&k) {
                    let over = best.is_none_or(|b| -q.y < b.0);
                    if !over {
                        shade = shade.max(smoothstep(1.35, 1.0, k));
                    }
                }
            }
            let g = grain.values()[i];
            let c = base * (0.8 + 0.4 * tone) * (1.0 + 0.08 * g);
            let w = smoothstep(0.2, 0.7, weather_n.values()[i]) * weathering * curl;
            let c = c + (grey - c) * w;
            let c = c * (1.0 - 0.45 * shade) * (1.0 + 0.1 * crown);
            let c = fissure + (c - fissure) * inside;
            bark.push(
                c,
                rough + 0.08 * (1.0 - inside),
                relief * (inside * (0.4 + 0.6 * curl) - 1.0) + 0.0002 * g,
            );
        }
        bark.material(grid)
    }
}
