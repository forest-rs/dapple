// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Corrosion: copper's tarnish and verdigris, steel's rust.
//!
//! Metals corrode where water stays longest. Both modules here read their
//! base the same way ([`Wetness`]): crevices that ambient occlusion and
//! cavities find (the foot of a standing seam, the underside of a lap, the
//! ring around a rivet head), trails run down from ledges ([`runoff`]),
//! and broad blotches for what the base's shape does not explain. `wetness`
//! says how much the water decides: 0 corrodes in blotches alone, 1 along
//! the water's paths. `age` says how far it has gone, from untouched (0)
//! to as far as it goes (1).
//!
//! Corrosion follows the metal: what it grows on is the base's
//! `base_metalness`, so paint, glaze or dirt over a metal keeps it from
//! corroding, and bare metal where they chip corrodes. The products are
//! dielectrics laid by [`ops::deposit`]: where they cover, metalness falls
//! to 0, roughness rises, and the surface stands proud by their thickness.
//! Their colors are authored, within plausible ranges
//! ([`super::calibration::OXIDES`]).

use alloc::vec;
use alloc::vec::Vec;

use dapple_field::program::Op;
use dapple_field::{CellOutput, Value};
use dapple_material::module::{
    Args, Context, InputDecl, InputKind, Interface, Module, ModuleError, ModuleId, Output,
    OutputDecl, OutputKind, Outputs,
};
use dapple_material::ops::{self, Deposit};
use dapple_material::{Aux, Channel, ChannelId, Grid, Material, Param};
use dapple_raster::{
    AmbientOcclusion, Flow, GaussianBlur, HeightToNormal, PercentileRemap, Raster, RasterOp, Streak,
};
use glam::Vec3;

use super::calibration::{
    RUST_COLOR, RUST_DARK, TARNISH_DARK_F0, TARNISH_F0, TARNISH_ROSE_F0, VERDIGRIS_COLOR,
};
use super::weathering::{ledges, runoff};
use super::{
    color, color_map, coverage, fail, fbm, fit, fraction, meters, realize_scalar, scalar_channel,
    seed, smoothstep,
};
use crate::glazed_brick::{PATINA, RUST};

fn base_input() -> InputDecl {
    InputDecl {
        name: "base".into(),
        kind: InputKind::Material,
        required: true,
        doc: "the metal corroded".into(),
    }
}

fn material_output() -> Vec<OutputDecl> {
    vec![OutputDecl {
        name: "material".into(),
        kind: OutputKind::Material,
        doc: "the corroded material".into(),
    }]
}

/// Where water stays on a base, per texel: see the [module docs](self).
struct Wetness {
    /// Where water stays, normalized to about `[0, 1]` between the 2nd and
    /// 98th percentiles.
    score: Raster,
    /// Water's trails down the face, in `[0, 1]`: where it runs rather
    /// than stays.
    trails: Raster,
    /// Vertical runs in about `[-1, 1]`, a few centimeters across, that
    /// break what water leaves into streaks.
    runs: Raster,
    /// Finer runs, millimeters across, that striate a trail.
    rivulets: Raster,
    /// Each texel's metalness: what can corrode.
    metal: Raster,
}

impl Wetness {
    fn of(grid: Grid, args: &Args, base: &Material) -> Result<Self, ModuleError> {
        let h = base.height()?;
        let ao = AmbientOcclusion {
            radius: 0.012,
            directions: 12,
            scale: 1.0,
        }
        .apply(&h)?;
        let blurred = GaussianBlur { sigma: 0.004 }.apply(&h)?;
        // Not every ledge sheds alike: a rivet head or a stretch of lap
        // sheds where water happens to gather along it.
        let gate = realize_scalar(grid, |b, d| fbm(b, d, [19.0, 4.0], args.seed("gate"), 3))?;
        let shed = grid.raster(
            ledges(base)?
                .values()
                .iter()
                .zip(gate.values())
                .map(|(l, g)| l * smoothstep(-0.15, 0.45, *g))
                .collect(),
        )?;
        let trails = runoff(grid, args, &shed, args.scalar("runoff"))?;
        // Water gathers as it runs down a sheet and hangs above a drip
        // edge before it falls: wettest just above a downward-facing edge,
        // drying up the face from it.
        let smooth = GaussianBlur { sigma: 0.003 }.apply(&h)?;
        let normals = HeightToNormal { scale: 1.0 }.apply(&smooth)?;
        let drips = grid.raster(
            normals
                .values()
                .iter()
                .map(|n| ((-n[1] - 0.12) * 5.0).clamp(0.0, 1.0))
                .collect(),
        )?;
        let pooled = Streak {
            flow: Flow::PositiveY,
            length: 0.06,
        }
        .apply(&drips)?;
        let blotches =
            realize_scalar(grid, |b, d| fbm(b, d, [4.0, 4.0], args.seed("blotches"), 5))?;
        let runs = realize_scalar(grid, |b, d| fbm(b, d, [45.0, 3.0], args.seed("streaks"), 4))?;
        let rivulets = realize_scalar(grid, |b, d| {
            fbm(b, d, [160.0, 6.0], args.seed("rivulets"), 3)
        })?;
        let wetness = args.scalar("wetness");
        let mut score = Vec::with_capacity(grid.len());
        for i in 0..grid.len() {
            let closed = 1.0 - ao.values()[i];
            // Cavity depth in millimeters.
            let cavity = ((blurred.values()[i] - h.values()[i]) * 1000.0).clamp(0.0, 1.0);
            let run = smoothstep(-0.5, 0.6, runs.values()[i]);
            let water = 1.2 * closed
                + 0.6 * cavity
                + (0.4 + 0.8 * run) * (trails.values()[i].min(1.0) + 0.45 * pooled.values()[i]);
            let blotch = 0.5 + 0.5 * blotches.values()[i];
            score.push(wetness * water + (0.35 + 0.65 * (1.0 - wetness)) * blotch);
        }
        let score = PercentileRemap {
            from: [0.02, 0.98],
            to: [0.0, 1.0],
            clamp: true,
        }
        .apply(&grid.raster(score)?)
        .map_err(dapple_material::MaterialError::Raster)?;
        let trails = grid.raster(trails.values().iter().map(|t| t.min(1.0)).collect())?;
        let metal = base.scalar(ChannelId::Param(Param::BaseMetalness))?;
        Ok(Self {
            score,
            trails,
            runs,
            rivulets,
            metal,
        })
    }
}

impl Wetness {
    /// How strongly a trail of `t` (from [`runoff`]) stains texel `i`:
    /// against a threshold the rivulets vary, so a trail is striated, and
    /// narrows and breaks up as it fades down the face, as a stain run
    /// tapers below its source.
    fn stain(&self, i: usize, t: f32) -> f32 {
        let rivulet = smoothstep(-0.5, 0.5, self.rivulets.values()[i]);
        let run = smoothstep(-0.4, 0.5, self.runs.values()[i]);
        let threshold = 0.05 + 0.3 * (1.0 - rivulet) + 0.2 * (1.0 - run);
        smoothstep(threshold, threshold + 0.5, t)
    }
}

fn wetness_params() -> [dapple_material::module::ParamDecl; 3] {
    [
        fraction("age", 0.5, "how far it has corroded: 0 untouched, 1 fully"),
        fraction(
            "wetness",
            0.6,
            "how much water's paths decide where, against blotches",
        ),
        meters(
            "runoff",
            [0.005, 2.0],
            0.25,
            "how far water runs down from a ledge before its trail fades by 1/e",
        ),
    ]
}

/// The base's value of channel `c` at texel `i` as a color.
fn base_color(base: &Material, i: usize) -> Vec3 {
    match base.value(ChannelId::Param(Param::BaseColor), i) {
        Value::Vector3(c) => c,
        other => Vec3::splat(other.component(0).unwrap_or(0.0)),
    }
}

/// A deposit's material from per-texel colors, roughness and surface
/// identity, keeping the base's tangent so the two do not mix.
fn product(
    base: &Material,
    colors: &[Vec3],
    roughness: &[f32],
    surface: u32,
) -> Result<Material, ModuleError> {
    let grid = base.grid();
    let mut m = Material::new(grid);
    m.set_param(Param::BaseColor, color_map(grid, colors)?)?;
    m.set_param(Param::SpecularRoughness, scalar_channel(grid, roughness)?)?;
    if let Some(t) = base.param(Param::GeometryTangent) {
        m.set_param(Param::GeometryTangent, t.clone())?;
    }
    m.set_aux(Aux::Surface, Channel::Constant(Value::Id(surface)))?;
    Ok(m)
}

/// Copper's patina, by `age`: the metal tarnishes first, its F0 darkening
/// unevenly toward a brown cuprite film (`tarnish`) and then near black
/// (`dark`), soonest where it stays wet; then pale blue-green verdigris
/// (`color`) crusts over it where water stays, `thickness` thick and
/// chalky, from crevices and below laps outward, until at `age` 1 it
/// covers some 85% of the surface; and where it has formed, rain washes
/// it down in thin green trails over the metal below, as strongly as
/// `wetness` says.
///
/// Verdigris is a dielectric: where it covers, metalness is 0 and
/// roughness near 0.9. The tarnish is a thin film the metal shows
/// through, so it stays a metal.
#[derive(Copy, Clone, Debug, Default)]
pub struct Patina;

impl Module for Patina {
    fn interface(&self) -> Interface {
        let mut params = wetness_params().to_vec();
        params.extend([
            color("tarnish", TARNISH_F0, "copper's F0 under a brown tarnish"),
            color("dark", TARNISH_DARK_F0, "its F0 under an old, dark tarnish"),
            color("color", VERDIGRIS_COLOR, "the verdigris's color"),
            meters(
                "thickness",
                [0.0, 0.002],
                0.00008,
                "the verdigris crust's thickness",
            ),
            seed(),
        ]);
        Interface {
            id: ModuleId::new("dapple_library.patina", 1),
            doc: "copper's tarnish, then verdigris where water stays and runs".into(),
            params,
            inputs: vec![base_input()],
            outputs: material_output(),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let base = args.material("base").ok_or_else(|| fail("base"))?;
        let age = args.scalar("age");
        if age <= 0.0 {
            return Ok(Outputs::new().with("material", Output::Material(base.clone())));
        }
        let wet = Wetness::of(grid, args, base)?;
        let stain = realize_scalar(grid, |b, d| fbm(b, d, [9.0, 9.0], args.seed("stain"), 5))?;
        let grain = realize_scalar(grid, |b, d| {
            fbm(b, d, [140.0, 140.0], args.seed("grain"), 3)
        })?;
        let fine = realize_scalar(grid, |b, d| fbm(b, d, [450.0, 450.0], args.seed("fine"), 2))?;
        let hue = realize_scalar(grid, |b, d| fbm(b, d, [3.0, 3.0], args.seed("hue"), 3))?;
        let patches = realize_scalar(grid, |b, d| {
            fbm(b, d, [28.0, 28.0], args.seed("patches"), 4)
        })?;
        let sheen = realize_scalar(grid, |b, d| fbm(b, d, [7.0, 5.0], args.seed("sheen"), 4))?;
        // Water spots: where drops dried, a dark ring round a darker disc,
        // in a few cells of a 3 cm lattice.
        let spot_cells = |output| {
            realize_scalar(grid, |b, d| {
                b.add(Op::Cellular {
                    domain: d,
                    frequency: fit(d, [32.0, 32.0]),
                    jitter: 0.9,
                    seed: args.seed("spots"),
                    output,
                })
            })
        };
        let (spot_value, spot_distance) = (
            spot_cells(CellOutput::CellValue)?,
            spot_cells(CellOutput::F1)?,
        );
        let (tarnish, dark) = (args.color("tarnish"), args.color("dark"));
        let n = grid.len();

        // Tarnish: the metal's own F0 darkened by the film, as a ratio to
        // fresh copper's, so the brushing and mottle under it still show;
        // soonest and darkest where water runs and stays, in its runs and
        // where drops dried. A young film is rose to purplish brown in
        // patches, where its thickness tints it, before it browns over.
        let fresh = super::calibration::COPPER_F0;
        let (to_brown, to_rose, to_dark) = (tarnish / fresh, TARNISH_ROSE_F0 / fresh, dark / fresh);
        let early = smoothstep(0.0, 0.35, age);
        let late = smoothstep(0.2, 0.85, age);
        let spotting = early * (1.0 - late);
        let mut m = base.clone();
        let mut colors = Vec::with_capacity(n);
        let mut rough = Vec::with_capacity(n);
        for i in 0..n {
            let q = wet.score.values()[i];
            let s = stain.values()[i];
            let k = wet.metal.values()[i];
            let run = wet.trails.values()[i] * smoothstep(-0.3, 0.5, wet.runs.values()[i]);
            let spot = if spot_value.values()[i] < 0.06 {
                let d = spot_distance.values()[i];
                let ring = smoothstep(0.17, 0.23, d) * smoothstep(0.31, 0.25, d);
                (0.35 * ring + 0.15 * smoothstep(0.26, 0.18, d)) * spotting
            } else {
                0.0
            };
            let brown = (early * (0.8 + 0.5 * q + 0.6 * s + 0.4 * run) + spot).clamp(0.0, 1.0);
            let black = (late * (0.8 + 0.4 * q + 0.5 * s + 0.4 * run)).clamp(0.0, 1.0);
            let rose = smoothstep(-0.1, 0.5, sheen.values()[i]) * (1.0 - late);
            let film = Vec3::ONE
                .lerp(to_brown, brown)
                .lerp(Vec3::ONE.lerp(to_rose, brown), 0.7 * rose)
                .lerp(to_dark, black);
            let c = base_color(base, i);
            colors.push(c.lerp((c * film).min(Vec3::ONE), k));
            let r = base
                .value(ChannelId::Param(Param::SpecularRoughness), i)
                .component(0)
                .unwrap_or(0.3);
            rough.push((r + k * (0.08 * brown + 0.1 * black)).min(1.0));
        }
        m.set_param(Param::BaseColor, color_map(grid, &colors)?)?;
        m.set_param(Param::SpecularRoughness, scalar_channel(grid, &rough)?)?;

        // Verdigris: where water stays, drawn down into streaks by its
        // runs and broken up into a granular crust at its edges.
        let share = 0.85 * libm::powf(smoothstep(0.2, 1.0, age), 1.2);
        let score: Vec<f32> = (0..n)
            .map(|i| {
                wet.metal.values()[i]
                    * (wet.score.values()[i]
                        + 0.22 * wet.runs.values()[i]
                        + 0.1 * grain.values()[i]
                        + 0.06 * fine.values()[i]
                        + 0.1 * stain.values()[i])
            })
            .collect();
        // A crust, crisp-edged, inside a thinner film it grows from,
        // through which the metal still shows. A percentile threshold can
        // fall on the zeros of bare metal's absence; only metal takes
        // verdigris.
        let score = grid.raster(score)?;
        let crust = coverage(&score, share, 0.012)?;
        let film = coverage(&score, (1.4 * share).min(1.0), 0.06)?;
        let covered = grid.raster(
            (0..n)
                .map(|i| crust.values()[i].max(0.45 * film.values()[i]) * wet.metal.values()[i])
                .collect(),
        )?;
        let verdigris = args.color("color");
        let thickness = args.scalar("thickness");
        // Its range: a pale, chalky bloom where it is thick and dry, a
        // deeper blue-green where it stays wet, greener or bluer in broad
        // patches as the salts vary.
        let grey = Vec3::splat(super::calibration::luminance(verdigris));
        let bloom = (verdigris.lerp(grey, 0.4) * 1.5 + Vec3::splat(0.03)).min(Vec3::ONE);
        let deep = verdigris * Vec3::new(0.6, 0.76, 0.76);
        let (mut crust_colors, mut crust_rough, mut relief) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for i in 0..n {
            let (q, g, f) = (wet.score.values()[i], grain.values()[i], fine.values()[i]);
            let run = smoothstep(-0.2, 0.7, wet.runs.values()[i]) * wet.trails.values()[i];
            let patch = 0.6 * patches.values()[i] + 0.4 * stain.values()[i];
            let pale = (smoothstep(-0.3, 0.45, patch) * (1.0 - 0.6 * q) + 0.25 * g + 0.12 * f)
                .clamp(0.0, 1.0);
            let c = verdigris
                .lerp(bloom, pale)
                .lerp(deep, (0.7 * q + 0.5 * run).min(0.85));
            // Greener or bluer, broadly, as the salts vary.
            let h = hue.values()[i];
            let c = c * Vec3::new(1.0 - 0.15 * h, 1.0 + 0.05 * h, 1.0 + 0.18 * h);
            crust_colors.push(c.clamp(Vec3::ZERO, Vec3::ONE));
            crust_rough.push((0.82 + 0.12 * pale).min(1.0));
            // Thicker in patches, granular at every scale.
            let build = 0.6 + 0.6 * patch + 0.45 * g + 0.25 * f;
            relief.push(thickness * build.clamp(0.0, 1.8));
        }
        let crust = product(&m, &crust_colors, &crust_rough, PATINA)?;
        let mut m = cx.record(ops::deposit(
            &m,
            &Deposit {
                material: crust,
                coverage: covered.clone(),
                thickness: 0.5 * thickness,
                relief: Some(grid.raster(relief)?),
                matting: 2.0,
            },
        )?);

        // Wash: verdigris carried down by rain in thin trails, once there
        // is enough of it to carry.
        let wetness = args.scalar("wetness");
        let sources = grid.raster(
            (0..n)
                .map(|i| covered.values()[i] * (0.4 + 0.6 * wet.score.values()[i]))
                .collect(),
        )?;
        let sources = GaussianBlur { sigma: 0.002 }.apply(&sources)?;
        let trails = runoff(grid, args, &sources, args.scalar("runoff"))?;
        let carried = smoothstep(0.3, 0.75, age);
        let wash: Vec<f32> = (0..n)
            .map(|i| {
                let t = wet.stain(i, wetness * trails.values()[i]);
                0.6 * carried * t * (1.0 - covered.values()[i]) * wet.metal.values()[i]
            })
            .collect();
        if wash.iter().any(|&w| w > 0.0) {
            let wash_colors: Vec<Vec3> = (0..n)
                .map(|i| verdigris.lerp(bloom, 0.4) * (0.92 + 0.12 * grain.values()[i]))
                .collect();
            let washed = product(&m, &wash_colors, &vec![0.6; n], PATINA)?;
            m = cx.record(ops::deposit(
                &m,
                &Deposit {
                    material: washed,
                    coverage: grid.raster(wash)?,
                    thickness: 0.000_005,
                    relief: None,
                    matting: 1.2,
                },
            )?);
        }
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}

/// Steel's rust, by `age`: where the base is bare metal it flashes over
/// first, a thin orange film in the wettest places that spreads as it ages,
/// then rusts through to a deep, rough crust, `color` mottled with dark
/// scale (`dark`), that builds up `scale` thick in flaking lumps and eats
/// pits `pitting` deep into the metal. Where paint has chipped it creeps
/// under the paint around the chip, further as it ages (`creep`), lifting
/// it; and rain carries it down in orange trails over whatever lies below,
/// as strongly as `wetness` says.
///
/// Rust is a dielectric: where it covers fully, metalness is 0; flash rust
/// is a partial cover the metal shows through.
#[derive(Copy, Clone, Debug, Default)]
pub struct Rust;

impl Module for Rust {
    fn interface(&self) -> Interface {
        let mut params = wetness_params().to_vec();
        params.extend([
            color("color", RUST_COLOR, "orange-brown rust"),
            color("dark", RUST_DARK, "the dark scale in it"),
            meters(
                "scale",
                [0.0, 0.005],
                0.0007,
                "how thick deep rust builds up",
            ),
            meters("pitting", [0.0, 0.005], 0.0008, "how deep its pits go"),
            meters(
                "creep",
                [0.0, 0.05],
                0.004,
                "how far it creeps under paint from bare metal",
            ),
            seed(),
        ]);
        Interface {
            id: ModuleId::new("dapple_library.rust", 1),
            doc: "rust on bare steel: flash rust, deep pitted rust, and its runoff".into(),
            params,
            inputs: vec![base_input()],
            outputs: material_output(),
        }
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let grid = cx.grid();
        let base = args.material("base").ok_or_else(|| fail("base"))?;
        let age = args.scalar("age");
        if age <= 0.0 {
            return Ok(Outputs::new().with("material", Output::Material(base.clone())));
        }
        let wet = Wetness::of(grid, args, base)?;
        let n = grid.len();

        // What is exposed: bare metal, and the paint around it rust has
        // crept under, further as it ages.
        let creep = args.scalar("creep");
        let reach = if creep > 0.0 {
            GaussianBlur {
                sigma: creep * (0.3 + age),
            }
            .apply(&wet.metal)?
        } else {
            wet.metal.clone()
        };
        let lift = 0.45 - 0.3 * age;
        let exposed: Vec<f32> = (0..n)
            .map(|i| {
                let undercut = smoothstep(lift, lift + 0.12, reach.values()[i]);
                wet.metal.values()[i].max(undercut * smoothstep(0.05, 0.3, age))
            })
            .collect();

        let speckle = realize_scalar(grid, |b, d| {
            fbm(b, d, [220.0, 220.0], args.seed("speckle"), 3)
        })?;
        let blooms = realize_scalar(grid, |b, d| fbm(b, d, [11.0, 11.0], args.seed("blooms"), 5))?;
        let broad = realize_scalar(grid, |b, d| fbm(b, d, [3.0, 3.0], args.seed("broad"), 4))?;
        let mottle = realize_scalar(grid, |b, d| fbm(b, d, [18.0, 18.0], args.seed("mottle"), 5))?;
        let lumps = realize_scalar(grid, |b, d| {
            fbm(b, d, [160.0, 160.0], args.seed("lumps"), 4)
        })?;
        let cells = |frequency: f32, purpose: &str, output: CellOutput| {
            realize_scalar(grid, |b, d| {
                b.add(Op::Cellular {
                    domain: d,
                    frequency: fit(d, [frequency, frequency]),
                    jitter: 0.9,
                    seed: args.seed(purpose),
                    output,
                })
            })
        };
        // Pits: some cells of a 4 mm lattice, deepest at their centers.
        let pit_cells = cells(240.0, "pits", CellOutput::CellValue)?;
        let pit_distance = cells(240.0, "pits", CellOutput::F1)?;
        // Scale: deep rust laminates into plates a few centimeters across,
        // cracked apart, some lifting at their edges.
        // Warped, so their edges wander rather than run straight.
        let warped = |output: CellOutput| {
            realize_scalar(grid, |b, d| {
                let cells = b.add(Op::Cellular {
                    domain: d,
                    frequency: fit(d, [36.0, 36.0]),
                    jitter: 0.9,
                    seed: args.seed("plates"),
                    output,
                })?;
                let dx = fbm(b, d, [22.0, 22.0], args.seed("plates_x"), 3)?;
                let dy = fbm(b, d, [22.0, 22.0], args.seed("plates_y"), 3)?;
                b.add(Op::Warp {
                    input: cells,
                    dx,
                    dy,
                    amount: 0.012,
                })
            })
        };
        let plate = warped(CellOutput::CellValue)?;
        let crack = warped(CellOutput::F2MinusF1)?;

        // Flash rust first, in blooms that start where it is wettest; then
        // rust through, in the wettest places first. A score of 0 where
        // nothing is exposed keeps both there.
        let flash_share = (3.0 * age).min(1.0);
        let deep_share = 0.94 * smoothstep(0.12, 1.0, age);
        let flash_score: Vec<f32> = (0..n)
            .map(|i| {
                exposed[i]
                    * (1.0
                        + 0.6 * wet.score.values()[i]
                        + 0.5 * blooms.values()[i]
                        + 0.25 * speckle.values()[i])
            })
            .collect();
        let deep_score: Vec<f32> = (0..n)
            .map(|i| {
                exposed[i]
                    * (1.0
                        + wet.score.values()[i]
                        + 0.25 * mottle.values()[i]
                        + 0.12 * speckle.values()[i])
            })
            .collect();
        let flash = coverage(&grid.raster(flash_score)?, flash_share, 0.06)?;
        let deep = coverage(&grid.raster(deep_score)?, deep_share, 0.01)?;

        let (orange, dark) = (args.color("color"), args.color("dark"));
        let brown = orange * Vec3::new(0.55, 0.5, 0.5);
        let (scale, pitting) = (args.scalar("scale"), args.scalar("pitting"));
        let pit_density = 0.25 + 0.35 * age;
        let (mut cover, mut colors, mut rough, mut relief) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for (i, &e) in exposed.iter().enumerate() {
            // A percentile threshold can fall on the zeros where nothing
            // is exposed; only what is exposed rusts.
            let (f, d) = (flash.values()[i] * e, deep.values()[i] * e);
            // Flash rust is a speckled haze the steel shows through.
            let haze = 0.4 + 0.35 * smoothstep(-0.4, 0.4, speckle.values()[i]);
            let c = (haze * f).max(d);
            cover.push(c);
            let through = if c > 0.0 { d / c } else { 0.0 };
            // Pits cluster where it is wettest and in patches, and each is
            // its own size: the cell's value scales its radius.
            let q = wet.score.values()[i];
            let clustered = pit_density * smoothstep(-0.3, 0.5, mottle.values()[i] + 0.8 * q - 0.3);
            let v = pit_cells.values()[i];
            let pit = if v < clustered {
                let radius = 0.18 + 0.32 * v / clustered.max(1e-3);
                let p = (1.0 - pit_distance.values()[i] / radius).max(0.0);
                p * p
            } else {
                0.0
            };
            // Only some plate edges have cracked open, in patches.
            let cracked = (1.0 - smoothstep(0.0, 0.035, crack.values()[i]))
                * smoothstep(0.25, 0.7, broad.values()[i] + mottle.values()[i]);
            let lifted = smoothstep(0.7, 1.0, plate.values()[i])
                * smoothstep(0.0, 0.5, broad.values()[i] + 0.5);
            let lump = (0.5 + 0.6 * lumps.values()[i]).clamp(0.0, 1.0);
            // Deep rust ranges from orange through brown to dark scale:
            // darker where it stays wet, in pits and cracks; brighter on
            // lifted plates and in fresh blooms.
            let t = (0.36
                + 0.4 * broad.values()[i]
                + 0.35 * mottle.values()[i]
                + 0.2 * lump
                + 0.12 * lifted
                - 0.3 * q
                - 0.6 * pit
                - 0.35 * cracked)
                .clamp(0.0, 1.0);
            let crust = if t < 0.45 {
                dark.lerp(brown, smoothstep(0.1, 0.45, t))
            } else {
                brown.lerp(orange * 1.1, smoothstep(0.45, 0.9, t))
            };
            let film = orange * (1.15 + 0.15 * speckle.values()[i]);
            colors.push(film.lerp(crust, through).min(Vec3::ONE));
            rough.push(0.62 + (0.9 + 0.06 * cracked - 0.62) * through);
            let build = scale * (0.4 + age) * (0.6 * lump + 0.4 * lifted - 0.25 * cracked);
            relief.push(d * (build - pitting * pit));
        }
        let rust = product(base, &colors, &rough, RUST)?;
        let cover = grid.raster(cover)?;
        let mut m = cx.record(ops::deposit(
            base,
            &Deposit {
                material: rust,
                coverage: cover.clone(),
                thickness: 0.000_02,
                relief: Some(grid.raster(relief)?),
                matting: 2.5,
            },
        )?);

        // Runoff: fresh rust carried down in orange trails over whatever
        // lies below, the crust included.
        let wetness = args.scalar("wetness");
        let sources = grid.raster(
            (0..n)
                .map(|i| (deep.values()[i] + 0.3 * flash.values()[i]) * exposed[i])
                .collect(),
        )?;
        // A chip's worth of rust stains less than a rivet head's.
        let sources = GaussianBlur { sigma: 0.003 }.apply(&sources)?;
        let length = args.scalar("runoff") * (0.4 + 0.8 * wetness);
        let trails = runoff(grid, args, &sources, length)?;
        let bleed: Vec<f32> = (0..n)
            .map(|i| {
                let t = trails.values()[i].min(1.0);
                let stain = wet.stain(i, 1.2 * wetness * t) * (0.35 + 0.65 * t);
                0.6 * stain * (1.0 - 0.7 * cover.values()[i]) * smoothstep(0.0, 0.4, age)
            })
            .collect();
        if bleed.iter().any(|&b| b > 0.0) {
            let stain_colors: Vec<Vec3> = (0..n)
                .map(|i| (orange * (1.1 + 0.25 * mottle.values()[i])).min(Vec3::ONE))
                .collect();
            let stain = product(&m, &stain_colors, &vec![0.55; n], RUST)?;
            m = cx.record(ops::deposit(
                &m,
                &Deposit {
                    material: stain,
                    coverage: grid.raster(bleed)?,
                    thickness: 0.000_005,
                    relief: None,
                    matting: 1.5,
                },
            )?);
        }
        Ok(Outputs::new().with("material", Output::Material(m)))
    }
}
