// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use dapple_field::program::ValueProgram;
use dapple_field::{Domain, Footprint, ScalarField, Value};
use glam::Vec2;

use crate::{brick, gravel, oak, parquet};

fn torus() -> Domain {
    Domain::periodic(1, 1).unwrap()
}

/// Points spread over the tile, off every joint.
fn points() -> impl Iterator<Item = Vec2> {
    (0..97).map(|i| Vec2::new(i as f32 * 0.0731 % 1.0, i as f32 * 0.1379 % 1.0))
}

fn assert_tiles(name: &str, height: &impl ScalarField, color: &ValueProgram) {
    let footprint = Footprint::new(1.0 / 1024.0).unwrap();
    for p in points() {
        let q = p + Vec2::new(1.0, -1.0);
        let (a, b) = (height.eval(p, footprint), height.eval(q, footprint));
        assert!((a - b).abs() < 1e-4, "{name} height at {p}: {a} vs {b}");
        assert!((-0.01..=1.01).contains(&a), "{name} height {a} at {p}");
        let (Value::Vector3(a), Value::Vector3(b)) =
            (color.eval(p, footprint), color.eval(q, footprint))
        else {
            panic!("{name} color is not a color");
        };
        let (a, b) = (a.to_array(), b.to_array());
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-4, "{name} color at {p}: {a:?} vs {b:?}");
            assert!((0.0..=1.0).contains(x), "{name} color {a:?} at {p}");
        }
    }
}

#[test]
fn tileable_materials_repeat_across_the_period() {
    let bark = oak::bark_height(torus()).unwrap();
    assert_tiles("bark", &bark, &oak::bark_color(torus(), &bark).unwrap());
    assert_tiles(
        "brick",
        &brick::height(torus()).unwrap(),
        &brick::color(torus()).unwrap(),
    );
    assert_tiles(
        "gravel",
        &gravel::height(torus()).unwrap(),
        &gravel::color(torus()).unwrap(),
    );
    assert_tiles(
        "parquet",
        &parquet::height(torus()).unwrap(),
        &parquet::color(torus()).unwrap(),
    );
}

#[test]
fn materials_are_stable_programs() {
    // Building a material twice gives the same graph.
    assert_eq!(
        brick::height(torus()).unwrap().fingerprint(),
        brick::height(torus()).unwrap().fingerprint()
    );
    assert_eq!(
        oak::wood_channels().unwrap()[0].fingerprint(),
        oak::wood_channels().unwrap()[0].fingerprint()
    );
}

#[test]
fn brick_joints_are_mortar() {
    let height = brick::height(torus()).unwrap();
    let footprint = Footprint::new(1.0 / 1024.0).unwrap();
    // A bed joint runs along y = 0 and a brick's middle sits half a course up.
    let joint = height.eval(Vec2::new(0.1, 0.0), footprint);
    let face = height.eval(Vec2::new(0.1, 0.5 / brick::BRICKS[1]), footprint);
    assert!(joint < 0.3 && face > 0.6, "joint {joint}, face {face}");
}

mod glazed {
    use alloc::vec::Vec;

    use dapple_field::{Edge, Value};
    use dapple_material::module::{Bind, Context};
    use dapple_material::resource::NoResources;
    use dapple_material::{Aux, ChannelId, Grid, Material, Param};
    use dapple_raster::{DistanceTransform, RasterOp};
    use glam::Vec2;

    use crate::glazed_brick::{BODY, GLAZE, MORTAR};
    use crate::modules::GlazedBrickWall;

    const SIZE: u32 = 256;

    pub(super) fn tile(size: u32) -> Grid {
        Grid {
            width: size,
            height: size,
            origin: Vec2::ZERO,
            texel: Vec2::splat(1.0 / size as f32),
            edge: Edge::Wrap,
        }
    }

    fn wall(weathered: bool) -> Material {
        let mut cx = Context::new(tile(SIZE), &NoResources);
        let mut out = cx
            .instantiate(
                &GlazedBrickWall,
                "wall",
                Bind::new().flag("weathered", weathered),
            )
            .unwrap();
        out.take_material("material").unwrap()
    }

    fn surfaces(m: &Material) -> Vec<u32> {
        (0..m.grid().len())
            .map(|i| match m.value(ChannelId::Aux(Aux::Surface), i) {
                Value::Id(id) => id,
                other => panic!("surface {other:?}"),
            })
            .collect()
    }

    fn scalar(m: &Material, c: ChannelId, i: usize) -> f32 {
        m.value(c, i).scalar().unwrap()
    }

    #[test]
    fn chips_break_from_the_arris_and_expose_rough_body() {
        let m = wall(false);
        let ids = surfaces(&m);
        // Distance from each texel into its brick: to the nearest mortar.
        let mortar = m
            .grid()
            .raster(
                ids.iter()
                    .map(|&s| f32::from(u8::from(s == MORTAR)))
                    .collect(),
            )
            .unwrap();
        let inward = DistanceTransform { threshold: 0.5 }.apply(&mortar).unwrap();
        let (mut body, mut near_edge, mut glazed, mut coated) = (0_u32, 0_u32, 0_u32, 0_u32);
        let (mut body_rough, mut coat_rough) = (0.0_f64, 0.0_f64);
        for (i, &s) in ids.iter().enumerate() {
            match s {
                BODY => {
                    body += 1;
                    near_edge += u32::from(inward.values()[i] < 0.012);
                    body_rough +=
                        f64::from(scalar(&m, ChannelId::Param(Param::SpecularRoughness), i));
                }
                GLAZE => {
                    glazed += 1;
                    coat_rough += f64::from(scalar(&m, ChannelId::Param(Param::CoatRoughness), i));
                    coated += u32::from(scalar(&m, ChannelId::Param(Param::CoatWeight), i) > 0.5);
                }
                MORTAR => {}
                other => panic!("unknown surface {other}"),
            }
        }
        let (body_rough, coat_rough) =
            (body_rough / f64::from(body), coat_rough / f64::from(glazed));
        assert!(body_rough > 0.65, "exposed body is rough: {body_rough}");
        assert!(
            coated * 100 >= glazed * 97,
            "glaze is coated: {coated} of {glazed}"
        );
        assert!(coat_rough < 0.15, "glaze is glossy: {coat_rough}");
        let share = f64::from(body) / f64::from(SIZE * SIZE);
        assert!(share > 0.002 && share < 0.1, "chipped share {share}");
        assert!(
            near_edge * 100 >= body * 85,
            "{near_edge} of {body} chipped texels near an edge"
        );
    }

    #[test]
    fn mortar_is_recessed_and_textured_and_weathering_darkens_it() {
        let clean = wall(false);
        let ids = surfaces(&clean);
        let h = ChannelId::Aux(Aux::Height);
        let color = ChannelId::Param(Param::BaseColor);
        let (mut mortar, mut glaze) = (Vec::new(), Vec::new());
        for (i, &s) in ids.iter().enumerate() {
            match s {
                MORTAR => mortar.push((
                    scalar(&clean, h, i),
                    clean.value(color, i).component(0).unwrap(),
                )),
                GLAZE => glaze.push(scalar(&clean, h, i)),
                _ => {}
            }
        }
        let highest = mortar.iter().map(|m| m.0).fold(f32::MIN, f32::max);
        let mean_glaze = glaze.iter().sum::<f32>() / glaze.len() as f32;
        assert!(
            mean_glaze - highest > 0.002,
            "glaze at {mean_glaze} m, mortar up to {highest} m"
        );
        let (lo, hi) = mortar
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), m| (a.min(m.1), b.max(m.1)));
        assert!(hi - lo > 0.03, "mortar red from {lo} to {hi}");

        let weathered = wall(true);
        let mean = |m: &Material| {
            (0..m.grid().len())
                .map(|i| f64::from(m.value(color, i).component(1).unwrap()))
                .sum::<f64>()
                / m.grid().len() as f64
        };
        assert!(mean(&weathered) < mean(&clean), "dirt darkens");
        let salts = surfaces(&weathered)
            .iter()
            .filter(|&&s| s == crate::glazed_brick::SALT)
            .count();
        assert!(salts > 0, "salts bloom");
    }
}

mod sill {
    use dapple_material::module::{Bind, Context};
    use dapple_material::resource::NoResources;
    use dapple_material::{ChannelId, Material, Param};

    use crate::modules::StoneSill;

    /// Rain runs off a stone sill and down the wall from under it, and
    /// nothing runs down the sill's own face: the tile's top, where the
    /// sill's top meets the wall's foot only across a wrap the sill does
    /// not tile across, sheds no streaks (it did, as dark drips from the
    /// top of the tile).
    #[test]
    fn streaks_run_from_under_the_sill_not_down_its_face() {
        let grid = super::glazed::tile(128);
        let build = |streaks: f32| -> Material {
            let mut cx = Context::new(grid, &NoResources);
            cx.instantiate(&StoneSill, "sill", Bind::new().scalar("streaks", streaks))
                .unwrap()
                .take_material("material")
                .unwrap()
        };
        let (streaked, clean) = (build(0.22), build(0.0));
        let color = ChannelId::Param(Param::BaseColor);
        // The sill's default height on the face is 0.15 m.
        let bottom = 1.0 - 0.15;
        let (mut face, mut below) = (0, 0);
        for i in 0..grid.len() {
            let y = grid.center(i).y;
            let streak = streaked.value(color, i) != clean.value(color, i);
            if y > bottom + 0.01 {
                face += usize::from(streak);
            } else if y > bottom - 0.2 {
                below += usize::from(streak);
            }
        }
        assert_eq!(face, 0, "texels streaked on the sill's face");
        assert!(below > 500, "{below} texels streaked under the sill");
    }
}

mod gate {
    use alloc::string::String;
    use alloc::vec::Vec;

    use dapple_field::Edge;
    use dapple_material::module::{Bind, Context, Diagnostics, Event, Module};
    use dapple_material::ops::{self, MaterialTransform};
    use dapple_material::resource::NoResources;
    use dapple_material::{Channel, Grid, Material};
    use glam::Vec2;

    use crate::modules::{StoneSill, Threshold, VarnishedBoard};

    fn build(asset: &dyn Module, grid: Grid) -> (Material, Diagnostics) {
        let mut cx = Context::new(grid, &NoResources);
        let mut out = cx.instantiate(asset, "asset", Bind::new()).unwrap();
        let m = out.take_material("material").unwrap();
        (m, cx.into_diagnostics())
    }

    fn strip() -> Grid {
        Grid {
            width: 128,
            height: 32,
            origin: Vec2::ZERO,
            texel: Vec2::splat(1.0 / 128.0),
            edge: Edge::Clamp,
        }
    }

    fn modules(d: &Diagnostics) -> Vec<&str> {
        let mut v: Vec<&str> = d.instances().map(|(_, m)| m.name.as_ref()).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Slice 2's gate: the same stone, wood and finish modules reused in
    /// several assets without duplicating their graphs; each operation
    /// reports its approximations; a material-wide transform moves every
    /// channel.
    #[test]
    fn modules_are_reused_across_assets_and_report_approximations() {
        let (sill, sill_d) = build(&StoneSill, super::glazed::tile(128));
        let (_, board_d) = build(&VarnishedBoard, strip());
        let (_, threshold_d) = build(&Threshold, strip());
        let used: [Vec<&str>; 3] = [modules(&sill_d), modules(&board_d), modules(&threshold_d)];
        let count = |name: &str| used.iter().filter(|u| u.contains(&name)).count();
        assert!(count("dapple_library.stone") >= 2, "{used:?}");
        assert!(count("dapple_library.wood") >= 2, "{used:?}");
        assert!(count("dapple_library.finish") >= 3, "{used:?}");
        assert!(count("dapple_library.mortar") >= 1, "{used:?}");
        // One module, one definition: every instance of a module names the
        // same versioned identity, whichever asset instantiated it: the one
        // the library registers.
        let registry = crate::packages::registry();
        for d in [&sill_d, &board_d, &threshold_d] {
            for (path, id) in d.instances() {
                let registered = registry
                    .modules()
                    .map(|m| m.interface().id)
                    .find(|r| r.name == id.name);
                assert_eq!(registered.as_ref(), Some(id), "{path}");
            }
        }
        // Reports stay with the instances that made them, and partial
        // selections and deposits say what they approximated.
        let mut kinds = Vec::new();
        for e in &sill_d.entries {
            if let Event::Operation(r) = &e.event {
                assert!(!e.path.is_empty(), "attributed");
                for a in &r.approximations {
                    kinds.push((String::from(r.operation), a.kind));
                }
            }
        }
        assert!(
            kinds.iter().any(|(op, _)| op == "deposit"),
            "deposits report: {kinds:?}"
        );
        assert!(
            kinds.iter().any(|(op, _)| op == "select"),
            "selections report: {kinds:?}"
        );
        let dirt = sill_d.reports_at("asset/dirt").count();
        assert_eq!(dirt, 1, "the grime's deposit is reported at its instance");

        // A material-wide transform moves every channel.
        let (moved, report) = ops::transform(
            &sill,
            MaterialTransform::offset(Vec2::new(3.0 / 128.0, 0.0)),
        )
        .unwrap();
        assert!(report.is_exact());
        let mut maps = 0;
        for (c, ch) in sill.bound() {
            if let Channel::Map(r) = ch {
                maps += 1;
                assert_eq!(moved.value(c, 3), r.value_at(0, 0), "{c} moved");
            }
        }
        assert!(maps >= 6, "a rich material: {maps} maps");
    }
}

mod wear {
    use alloc::vec::Vec;

    use dapple_field::hash::{hash, unit_f32};
    use dapple_field::program::Fingerprint;
    use dapple_field::{Domain, ImageLevel, PortType, Primaries, SampleImage, SamplePolicy, Value};
    use dapple_material::module::{Bind, Context, Input, Output};
    use dapple_material::resource::{
        NoResources, Resolved, ResourceError, ResourceHost, ResourceRef,
    };
    use dapple_material::{Aux, Channel, ChannelId, Material, Param};
    use dapple_raster::typed::ReductionPolicy;
    use glam::{Vec2, Vec3};

    use crate::modules::{ByExample, EdgeWear, Moss, Patches};

    /// A raised square on a flat ground, 64² over one meter.
    fn block() -> Material {
        let grid = super::glazed::tile(64);
        let mut m = Material::new(grid);
        let h: Vec<Value> = (0..grid.len())
            .map(|i| {
                let (x, y) = (i % 64, i / 64);
                let inside = (16..48).contains(&x) && (16..48).contains(&y);
                Value::Scalar(if inside { 0.005 } else { 0.0 })
            })
            .collect();
        m.set_aux(
            Aux::Height,
            Channel::Map(grid.typed(PortType::Scalar, h).unwrap()),
        )
        .unwrap();
        m.set_param(
            Param::BaseColor,
            Channel::Constant(Value::Vector3(Vec3::splat(0.2))),
        )
        .unwrap();
        m.set_param(Param::CoatWeight, Channel::Constant(Value::Scalar(1.0)))
            .unwrap();
        m
    }

    fn value(m: &Material, p: Param, x: usize, y: usize) -> Value {
        m.value(ChannelId::Param(p), y * 64 + x)
    }

    #[test]
    fn edge_wear_takes_the_arrises_and_moss_the_ledges() {
        let mut cx = Context::new(super::glazed::tile(64), &NoResources);
        let mut out = cx
            .instantiate(
                &EdgeWear,
                "wear",
                Bind::new()
                    .material("base", block())
                    .scalar("coverage", 0.05)
                    .scalar("radius", 0.02),
            )
            .unwrap();
        let worn = out.take_material("material").unwrap();
        // The block's top edge, just inside, is worn; its middle is not.
        assert_eq!(value(&worn, Param::CoatWeight, 32, 32), Value::Scalar(1.0));
        let edge = value(&worn, Param::CoatWeight, 32, 47).scalar().unwrap();
        assert!(edge < 0.5, "the arris lost its coat: {edge}");

        let mut out = cx
            .instantiate(
                &Moss,
                "moss",
                Bind::new()
                    .material("base", block())
                    .scalar("coverage", 0.05)
                    .scalar("damp", 0.0),
            )
            .unwrap();
        let moss = out.take_material("material").unwrap();
        let fuzz = |x, y| value(&moss, Param::FuzzWeight, x, y).scalar().unwrap();
        // Up-facing: the step up at the block's foot faces down, its top
        // edge faces up (+y), and moss takes the ledge above.
        let top: f32 = (16..48).map(|x| fuzz(x, 48)).sum();
        let bottom: f32 = (16..48).map(|x| fuzz(x, 15)).sum();
        assert!(
            top > bottom,
            "moss on the ledge {top} over the overhang {bottom}"
        );
        assert!(
            cx.diagnostics().reports_at("moss").count() == 1,
            "the deposit is reported"
        );
    }

    /// Wear comes only where its `where` mask allows, and broad patches
    /// cover about their share of the surface.
    #[test]
    fn edge_wear_comes_where_it_is_allowed() {
        let grid = super::glazed::tile(64);
        let mut cx = Context::new(grid, &NoResources);
        // The block's left half may wear, its right half may not.
        let left: Vec<Value> = (0..grid.len())
            .map(|i| Value::Scalar(if i % 64 < 32 { 1.0 } else { 0.0 }))
            .collect();
        let mut out = cx
            .instantiate(
                &EdgeWear,
                "wear",
                Bind::new()
                    .material("base", block())
                    .input(
                        "where",
                        Input::Map(grid.typed(PortType::Mask, left).unwrap()),
                    )
                    .scalar("coverage", 0.05)
                    .scalar("radius", 0.02),
            )
            .unwrap();
        let worn = out.take_material("material").unwrap();
        let coat = |x, y| value(&worn, Param::CoatWeight, x, y).scalar().unwrap();
        assert!(coat(16, 32) < 0.5, "the left arris wore: {}", coat(16, 32));
        assert!(coat(47, 32) > 0.99, "the right arris kept its coat");

        let out = cx
            .instantiate(&Patches, "patches", Bind::new().scalar("coverage", 0.3))
            .unwrap();
        let Some(Output::Map(mask)) = out.get("mask").cloned() else {
            panic!("a mask");
        };
        let share = (0..grid.len())
            .map(|i| mask.value(i).component(0).unwrap())
            .sum::<f32>()
            / grid.len() as f32;
        assert!((share - 0.3).abs() < 0.05, "patches cover {share}");
    }

    /// Wearing nothing gives the base as it is.
    #[test]
    fn edge_wear_of_nothing_is_the_base() {
        let mut cx = Context::new(super::glazed::tile(64), &NoResources);
        for (coverage, strength) in [(0.0, 1.0), (0.05, 0.0)] {
            let mut out = cx
                .instantiate(
                    &EdgeWear,
                    "wear",
                    Bind::new()
                        .material("base", block())
                        .scalar("coverage", coverage)
                        .scalar("strength", strength),
                )
                .unwrap();
            let worn = out.take_material("material").unwrap();
            assert_eq!(worn, block(), "coverage {coverage}, strength {strength}");
        }
    }

    struct Exemplar;

    impl ResourceHost for Exemplar {
        fn resolve(&self, reference: &ResourceRef) -> Result<Resolved, ResourceError> {
            if reference.0 != "stone.exr" {
                return Err(ResourceError::NotFound(reference.clone()));
            }
            // 96² texels of 5 mm: blotches of color in blocks of 6 texels.
            let n = 96_u32;
            let values: Vec<f32> = (0..n * n)
                .flat_map(|i| {
                    let block = (i % n) / 6 + 1000 * ((i / n) / 6);
                    let u = unit_f32(hash(5, &[u64::from(block)]));
                    [0.3 + 0.4 * u, 0.25 + 0.3 * u, 0.2 + 0.1 * u * u]
                })
                .collect();
            let level = ImageLevel::with_channels(n, n, Vec2::splat(0.005), 3, values).unwrap();
            let image = SampleImage::typed(
                PortType::Color(Primaries::Rec709),
                SamplePolicy::Linear,
                Domain::Plane,
                Vec2::ZERO,
                alloc::vec![level],
                Fingerprint(11),
            )
            .unwrap();
            Ok(Resolved {
                content: Fingerprint(11),
                image,
                mips: ReductionPolicy::Average,
            })
        }
    }

    #[test]
    fn by_example_tiles_a_host_exemplar_keeping_its_colors() {
        let mut cx = Context::new(super::glazed::tile(64), &Exemplar);
        let mut out = cx
            .instantiate(
                &ByExample,
                "stone",
                Bind::new()
                    .input("exemplar", Input::Resource(ResourceRef("stone.exr".into())))
                    .scalar("cell", 0.1),
            )
            .unwrap();
        let m = out.take_material("material").unwrap();
        let reds: Vec<f32> = (0..m.grid().len())
            .map(|i| {
                m.value(ChannelId::Param(Param::BaseColor), i)
                    .component(0)
                    .unwrap()
            })
            .collect();
        let (lo, hi) = reds
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        assert!(
            lo >= 0.29 && hi <= 0.71,
            "within the exemplar's range: {lo}..{hi}"
        );
        assert!(hi - lo > 0.2, "keeps its contrast: {lo}..{hi}");
    }
}

mod tiling {
    use dapple_material::Tiling;
    use dapple_material::module::{Bind, Context, Module};
    use dapple_material::resource::NoResources;

    use crate::modules::{CeramicBody, Mortar, Stone};

    /// Every material family promises to tile on a wrapping grid, and
    /// instantiation holds it to that: features whose frequency does not
    /// divide the period (a tooling pitch, a band width) would leave a
    /// seam and fail.
    #[test]
    fn families_tile_on_a_wrapping_grid() {
        let modules: [&dyn Module; 3] = [&Stone, &CeramicBody, &Mortar];
        for module in modules {
            let mut cx = Context::new(super::glazed::tile(512), &NoResources);
            let name = module.interface().id.name;
            let mut out = cx
                .instantiate(module, "m", Bind::new())
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let m = out.take_material("material").unwrap();
            assert_eq!(m.tiling(), Tiling::BOTH, "{name}");
        }
    }
}

mod packages {
    use dapple_field::Edge;
    use dapple_material::module::{Bind, Context, Module};
    use dapple_material::resource::NoResources;
    use dapple_material::{Grid, Material};
    use dapple_package::{Package, PackageError};
    use glam::{Vec2, Vec3};

    use crate::modules::VarnishedBoard;
    use crate::packages::{VARNISHED_BOARD, registry};

    fn grid() -> Grid {
        Grid {
            width: 64,
            height: 32,
            origin: Vec2::ZERO,
            texel: Vec2::splat(1.0 / 64.0),
            edge: Edge::Clamp,
        }
    }

    fn realize(module: &dyn Module, bind: Bind) -> Material {
        let mut cx = Context::new(grid(), &NoResources);
        cx.instantiate(module, "board", bind)
            .unwrap_or_else(|e| panic!("{e}"))
            .take_material("material")
            .expect("a material")
    }

    /// The package form of the varnished board survives a round trip
    /// through its serialized form, and realizes the native module's bits,
    /// with and without its preset.
    #[test]
    fn the_packaged_board_matches_the_native_one_bit_for_bit() {
        let package = Package::from_json(VARNISHED_BOARD).expect("reads");
        let again = Package::from_json(&package.to_json()).expect("reads its own output");
        assert_eq!(again, package);
        assert_eq!(again.fingerprint(), package.fingerprint());
        let engine = registry();
        let (a, b) = (
            engine.compile(&package).expect("compiles"),
            engine.compile(&again).expect("compiles"),
        );
        assert_eq!(a.fingerprint(), b.fingerprint());
        let packaged = realize(&a, Bind::new());
        assert_eq!(packaged, realize(&b, Bind::new()));
        let native = realize(&VarnishedBoard, Bind::new());
        assert_eq!(packaged.digest(), native.digest());

        let preset = a.preset("dark_and_worn").expect("a preset");
        let worn = realize(&a, preset);
        let native_worn = realize(
            &VarnishedBoard,
            Bind::new()
                .color("varnish", Vec3::new(0.3, 0.16, 0.04))
                .scalar("tint", 0.85)
                .scalar("roughness", 0.16)
                .scalar("dullness", 0.12)
                .scalar("wear", 0.05)
                .scalar("dirt", 0.08),
        );
        assert_eq!(worn.digest(), native_worn.digest());
        assert_ne!(worn.digest(), packaged.digest());
    }

    /// An engine without a module the package needs refuses it by name.
    #[test]
    fn a_missing_module_is_refused() {
        let package = Package::from_json(VARNISHED_BOARD).expect("reads");
        let engine = dapple_package::Registry::new();
        assert!(matches!(
            engine.compile(&package),
            Err(PackageError::UnknownModule(m)) if m.name == "dapple_library.wood"
        ));
    }
}

mod breadth {
    use alloc::boxed::Box;
    use alloc::vec::Vec;

    use dapple_elements::{LayoutId, RunningBond};
    use dapple_field::{Domain, Value};
    use dapple_material::module::{Bind, Context, Input, Module};
    use dapple_material::resource::NoResources;
    use dapple_material::{Aux, ChannelId, Material, Param, Tiling};
    use glam::Vec3;

    use crate::masonry::{UnitMaps, unit_id};
    use crate::modules::calibration::{BIRCH_BREAST_HEIGHT, REFERENCES, REJECTED, luminance};
    use crate::modules::{
        AshlarLimestone, Beech, Birch, FlintWall, Marble, RomanBrick, RubbleWall, ScotsPine,
        Spruce, TerracottaTile,
    };

    fn build(module: &dyn Module, n: u32, bind: Bind) -> Material {
        let mut cx = Context::new(super::glazed::tile(n), &NoResources);
        cx.instantiate(module, "m", bind)
            .unwrap_or_else(|e| panic!("{e}"))
            .take_material("material")
            .unwrap()
    }

    fn mean(m: &Material, units_only: bool) -> (Vec3, f32) {
        let (mut c, mut r, mut n) = (Vec3::ZERO, 0.0, 0.0);
        for i in 0..m.grid().len() {
            if units_only && m.value(ChannelId::Aux(Aux::Region), i) == Value::Id(0) {
                continue;
            }
            if let Value::Vector3(v) = m.value(ChannelId::Param(Param::BaseColor), i) {
                c += v;
            }
            r += m
                .value(ChannelId::Param(Param::SpecularRoughness), i)
                .component(0)
                .unwrap();
            n += 1.0;
        }
        (c / n, r / n)
    }

    fn modules() -> Vec<Box<dyn Module>> {
        alloc::vec![
            Box::new(Beech),
            Box::new(Birch),
            Box::new(ScotsPine),
            Box::new(Spruce),
            Box::new(AshlarLimestone),
            Box::new(RubbleWall),
            Box::new(FlintWall),
            Box::new(RomanBrick),
            Box::new(Marble),
            Box::new(TerracottaTile),
        ]
    }

    /// The calibrated defaults meet their measured references under the
    /// references' conditions (the `library_swatches` example refits them
    /// and rederives the references from spectra), pass the plausibility
    /// gate of a second source, and tile; the rejected sample fails its
    /// gate; a default birch's breast-height mean lies in the measured
    /// spread.
    #[test]
    fn defaults_match_measured_references_and_tile() {
        for (module, reference) in modules().iter().zip(REFERENCES) {
            assert_eq!(module.interface().id.name, reference.module);
            let bind = reference
                .conditions
                .iter()
                .fold(Bind::new(), |b, (n, v)| b.scalar(n, *v));
            let m = build(module.as_ref(), 128, bind);
            assert_eq!(m.tiling(), Tiling::BOTH, "{}", reference.module);
            let (c, r) = mean(&m, reference.units_only);
            let plausible = reference.plausible;
            if reference.luminance_only {
                let (y, target) = (luminance(c), reference.albedo.x);
                if reference.at_most {
                    assert!(y <= target, "{}: {y}", reference.module);
                } else {
                    assert!((y - target).abs() < 0.01, "{}: {y}", reference.module);
                    assert!(plausible.holds(c), "{}: {c}", reference.module);
                }
            } else {
                let miss = (c - reference.albedo).abs().max_element();
                assert!(
                    miss < 0.01,
                    "{}: {c} vs {}",
                    reference.module,
                    reference.albedo
                );
                assert!(plausible.holds(c), "{}: {c} implausible", reference.module);
                assert!(plausible.holds(reference.albedo), "{}", reference.module);
            }
            if let Some(target) = reference.roughness {
                assert!((r - target).abs() < 0.01, "{}: {r}", reference.module);
            }
        }
        for rejected in REJECTED {
            assert!(
                !rejected.plausible.holds(rejected.albedo),
                "{}",
                rejected.sample
            );
        }
        let y = luminance(mean(&build(&Birch, 128, Bind::new()), false).0);
        assert!(
            (BIRCH_BREAST_HEIGHT[0]..=BIRCH_BREAST_HEIGHT[1]).contains(&y),
            "birch at breast height: {y}"
        );
    }

    /// Bark follows growth: a birch's dark base climbs with girth, and a
    /// pine's upper bark is its orange, not its grey plates.
    #[test]
    fn bark_follows_girth_and_height() {
        let birch = |girth| {
            luminance(
                mean(
                    &build(&Birch, 64, Bind::new().scalar("girth", girth)),
                    false,
                )
                .0,
            )
        };
        assert!(
            birch(2.5) < 0.5 * birch(0.4),
            "{} {}",
            birch(2.5),
            birch(0.4)
        );
        let pine = |height| {
            mean(
                &build(&ScotsPine, 64, Bind::new().scalar("height", height)),
                false,
            )
            .0
        };
        let (low, high) = (pine(1.0), pine(15.0));
        assert!(high.x / high.z > 2.0 * low.x / low.z, "{low} {high}");
    }

    /// Masonry takes its units from the host: every unit texel carries the
    /// host element's identity.
    #[test]
    fn masonry_lays_on_the_host_units() {
        let grid = super::glazed::tile(128);
        let bond = RunningBond {
            layout: LayoutId::named("host"),
            domain: Domain::periodic(1, 1).unwrap(),
            courses: 4,
            per_course: 3,
            joint: 0.01,
        }
        .elements(alloc::vec![], |_, _| alloc::vec![])
        .unwrap();
        let ids: Vec<u32> = bond.keys().iter().map(|k| unit_id(*k)).collect();
        let maps = UnitMaps::from_elements(&bond, grid, 0.03).unwrap();
        let m = build(
            &AshlarLimestone,
            128,
            Bind::new()
                .input("units", Input::Map(maps.units))
                .input("edge", Input::Map(maps.edge))
                .input("local", Input::Map(maps.local)),
        );
        let mut seen = alloc::collections::BTreeSet::new();
        for i in 0..grid.len() {
            if let Value::Id(id) = m.value(ChannelId::Aux(Aux::Region), i)
                && id != 0
            {
                assert!(ids.contains(&id));
                seen.insert(id);
            }
        }
        assert_eq!(seen.len(), ids.len(), "every host unit is laid");
        // Binding only some of the maps is an error, not a silent fallback.
        let mut cx = Context::new(grid, &NoResources);
        let partial = UnitMaps::from_elements(&bond, grid, 0.03).unwrap();
        assert!(
            cx.instantiate(
                &AshlarLimestone,
                "m",
                Bind::new().input("units", Input::Map(partial.units))
            )
            .is_err()
        );
    }
}

mod wood {
    use alloc::vec::Vec;

    use dapple_field::{Edge, Value};
    use dapple_material::module::{Bind, Context};
    use dapple_material::resource::NoResources;
    use dapple_material::{Aux, Channel, Grid, Param};
    use glam::{Vec2, Vec3};

    use crate::modules::Wood;

    /// The board's linear base color over 10 × 4 cm about its middle at
    /// `texel` meters.
    fn board(texel: f32) -> (u32, Vec<Vec3>) {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "small whole texel counts"
        )]
        let (w, h) = (
            libm::roundf(0.1 / texel) as u32,
            libm::roundf(0.04 / texel) as u32,
        );
        let grid = Grid {
            width: w,
            height: h,
            origin: Vec2::new(0.3, 0.08),
            texel: Vec2::splat(texel),
            edge: Edge::Clamp,
        };
        let mut cx = Context::new(grid, &NoResources);
        let m = cx
            .instantiate(&Wood, "board", Bind::new())
            .unwrap()
            .take_material("material")
            .unwrap();
        let Some(Channel::Map(colors)) = m.param(Param::BaseColor) else {
            panic!("a base color map");
        };
        let colors = (0..colors.len())
            .map(|i| match colors.value(i) {
                Value::Vector3(c) => c,
                other => panic!("a color, not {other:?}"),
            })
            .collect();
        (w, colors)
    }

    /// The wood over a grid of `w` × `h` texels of 2 mm from `origin`.
    fn boards(origin: Vec2, w: u32, h: u32) -> dapple_material::Material {
        let grid = Grid {
            width: w,
            height: h,
            origin,
            texel: Vec2::splat(0.002),
            edge: Edge::Clamp,
        };
        let mut cx = Context::new(grid, &NoResources);
        cx.instantiate(&Wood, "boards", Bind::new())
            .unwrap()
            .take_material("material")
            .unwrap()
    }

    /// Texel `i`'s height.
    fn height(m: &dapple_material::Material, i: usize) -> f32 {
        let Some(Channel::Map(map)) = m.aux(Aux::Height) else {
            panic!("a height map");
        };
        map.value(i).scalar().expect("a scalar height")
    }

    /// Texel `i`'s base color, its components summed.
    fn color(m: &dapple_material::Material, i: usize) -> f32 {
        let Some(Channel::Map(map)) = m.param(Param::BaseColor) else {
            panic!("a color map");
        };
        let Value::Vector3(c) = map.value(i) else {
            panic!("a color");
        };
        c.x + c.y + c.z
    }

    /// Boards are laid 20 cm apart from `y = 0`, a sunken gap between them,
    /// and each board is the same cut wherever the grid starts.
    #[test]
    fn boards_are_laid_side_by_side_each_its_own_cut() {
        // 4 cm by 40 cm: two boards, the joint at row 100.
        let both = boards(Vec2::ZERO, 20, 200);
        let row = |r: usize| (0..20).map(|x| height(&both, r * 20 + x)).sum::<f32>() / 20.0;
        let joint = row(100).min(row(99));
        let lowest = (5..95)
            .chain(105..195)
            .map(row)
            .fold(f32::INFINITY, f32::min);
        assert!(
            joint < lowest - 1e-4,
            "the joint, {joint}, is sunken below {lowest}"
        );
        // The second board alone.
        let second = boards(Vec2::new(0.0, 0.2), 20, 100);
        for i in 20 * 5..20 * 95 {
            let (a, b) = (color(&both, 20 * 100 + i), color(&second, i));
            assert!((a - b).abs() < 1e-3, "texel {i}: {a} against {b}");
        }
        // The two boards are different cuts.
        let first: f32 = (20 * 5..20 * 95).map(|i| color(&both, i)).sum();
        let other: f32 = (20 * 5..20 * 95).map(|i| color(&second, i)).sum();
        assert!(
            (first - other).abs() > 1e-2 * first,
            "{first} against {other}"
        );
    }

    /// Baked at 4 mm, the board is close to its 0.5 mm bake averaged over
    /// each 4 mm texel: rings (5.5 mm apart) and rays (about a millimeter
    /// wide) fade to their mean rather than alias.
    #[test]
    fn a_coarse_board_is_a_downsample_of_a_fine_one() {
        let (fine_width, fine) = board(0.0005);
        let (width, coarse) = board(0.004);
        let k = 8;
        let mut squared = 0.0;
        for (i, &c) in coarse.iter().enumerate() {
            let (x, y) = (i % width as usize, i / width as usize);
            let mut sum = Vec3::ZERO;
            for j in 0..k {
                for l in 0..k {
                    sum += fine[(y * k + j) * fine_width as usize + x * k + l];
                }
            }
            let d = c - sum / (k * k) as f32;
            squared += d.length_squared() / 3.0;
        }
        let rms = libm::sqrtf(squared / coarse.len() as f32);
        // About 0.016, most of it the earlywood pores (0.004 without
        // them): a 4 mm texel's average of them still varies, where the
        // coarse bake has faded them to their mean. Pores that faded to
        // their ramp's floor gave 0.019; point-sampled rings and rays (a
        // `fract` of the radius and angle) gave 0.026.
        assert!(rms < 0.019, "rms {rms} against the downsampled bake");
        // Sub-texel pores fade to their mean, so the coarse bake is as
        // light as the fine one (within 0.4%); fading to the floor made it
        // 4.1% lighter.
        let mean = |c: &[Vec3]| c.iter().copied().sum::<Vec3>() / c.len() as f32;
        let (a, b) = (mean(&coarse), mean(&fine));
        let lighter = (a.x + a.y + a.z) / (b.x + b.y + b.z) - 1.0;
        assert!(
            lighter.abs() < 0.006,
            "the coarse bake is {lighter} lighter"
        );
    }
}

mod coverage {
    use alloc::vec::Vec;

    use dapple_field::Edge;
    use dapple_raster::Raster;
    use glam::Vec2;

    use crate::modules::coverage;

    /// A ramp of `n` distinct scores, below it a run of `zeros` zeros.
    fn scores(n: u32, zeros: u32) -> Raster {
        let values = (0..zeros + n)
            .map(|i| {
                if i < zeros {
                    0.0
                } else {
                    (i - zeros + 1) as f32
                }
            })
            .collect();
        Raster::from_values(zeros + n, 1, Vec2::ZERO, Vec2::ONE, Edge::Wrap, values).unwrap()
    }

    fn covered(r: &Raster) -> f32 {
        r.values().iter().sum::<f32>() / r.values().len() as f32
    }

    #[test]
    fn the_extremes_cover_nothing_and_everything() {
        for (score, softness) in [scores(200, 0), scores(100, 100)]
            .iter()
            .flat_map(|s| [0.0, 0.01, 0.12].map(|soft| (s, soft)))
        {
            let none = coverage(score, 0.0, softness).unwrap();
            assert!(
                none.values().iter().all(|&v| v == 0.0),
                "a share of 0 covers nothing (softness {softness})"
            );
            let all = coverage(score, 1.0, softness).unwrap();
            assert!(
                all.values().iter().all(|&v| v == 1.0),
                "a share of 1 covers everything (softness {softness})"
            );
        }
        let score = scores(100, 100);
        // Out-of-range shares clamp to the extremes.
        assert_eq!(covered(&coverage(&score, -0.5, 0.05).unwrap()), 0.0);
        assert_eq!(covered(&coverage(&score, 1.5, 0.05).unwrap()), 1.0);
        // In between, the share is about what was asked.
        let some: Vec<f32> = [0.1, 0.3]
            .iter()
            .map(|&f| covered(&coverage(&score, f, 0.01).unwrap()))
            .collect();
        assert!(
            (some[0] - 0.1).abs() < 0.02 && (some[1] - 0.3).abs() < 0.02,
            "{some:?}"
        );
    }
}

mod metals {
    use alloc::vec::Vec;

    use dapple_field::Value;
    use dapple_material::module::{Bind, Context, Module};
    use dapple_material::resource::NoResources;
    use dapple_material::{Aux, ChannelId, Material, Param, Tiling};
    use dapple_package::Package;
    use dapple_raster::{DistanceTransform, RasterOp};
    use glam::Vec3;

    use crate::glazed_brick::{METAL, PAINT, PATINA, RUST};
    use crate::modules::calibration::{METALS, OXIDES, fresnel_f0};
    use crate::modules::{
        Copper, Paint, PaintedSteel, Patina, Rust, RustedSteel, Steel, WeatheredCopper,
    };
    use crate::packages::{PAINTED_STEEL, RUSTED_STEEL, WEATHERED_COPPER, registry};

    fn build(module: &dyn Module, n: u32, bind: Bind) -> Material {
        let mut cx = Context::new(super::glazed::tile(n), &NoResources);
        cx.instantiate(module, "m", bind)
            .unwrap_or_else(|e| panic!("{e}"))
            .take_material("material")
            .unwrap()
    }

    fn scalar(m: &Material, p: Param, i: usize) -> f32 {
        m.value(ChannelId::Param(p), i).component(0).unwrap()
    }

    fn surface(m: &Material, i: usize) -> u32 {
        match m.value(ChannelId::Aux(Aux::Surface), i) {
            Value::Id(id) => id,
            other => panic!("surface {other:?}"),
        }
    }

    /// Means of roughness and metalness over the texels `keep` admits.
    fn means(m: &Material, keep: impl Fn(usize) -> bool) -> (f32, f32, usize) {
        let (mut r, mut k, mut n) = (0.0, 0.0, 0);
        for i in (0..m.grid().len()).filter(|&i| keep(i)) {
            r += scalar(m, Param::SpecularRoughness, i);
            k += scalar(m, Param::BaseMetalness, i);
            n += 1;
        }
        (r / n.max(1) as f32, k / n.max(1) as f32, n)
    }

    /// The metals' F0 is the tabulated one and passes the gate computed
    /// from optical constants; the authored oxides pass theirs; and the
    /// bare metals are metals, without diffuse, and tile.
    #[test]
    fn bare_metals_match_tabulated_f0_and_tile() {
        assert!(
            (fresnel_f0(1.5, 0.0) - 0.04).abs() < 1e-6,
            "a dielectric's F0"
        );
        let bare = Bind::new().scalar("brushing", 0.0);
        let metals: [(&dyn Module, Bind); 2] = [
            (&Copper, bare.clone()),
            (&Steel, bare.scalar("mill_scale", 0.0)),
        ];
        for ((module, bind), reference) in metals.into_iter().zip(METALS) {
            assert_eq!(module.interface().id.name, reference.module);
            assert!(reference.holds(reference.f0), "{}", reference.module);
            let m = build(module, 64, bind);
            assert_eq!(m.tiling(), Tiling::BOTH, "{}", reference.module);
            assert_eq!(
                m.constant(ChannelId::Param(Param::BaseMetalness)),
                Some(Value::Scalar(1.0))
            );
            let mut f0 = Vec3::ZERO;
            for i in 0..m.grid().len() {
                if let Value::Vector3(c) = m.value(ChannelId::Param(Param::BaseColor), i) {
                    f0 += c;
                }
            }
            let f0 = f0 / m.grid().len() as f32;
            let miss = (f0 - reference.f0).abs().max_element();
            assert!(
                miss < 0.02,
                "{}: {f0} vs {}",
                reference.module,
                reference.f0
            );
        }
        for oxide in OXIDES {
            assert!(oxide.plausible.holds(oxide.color), "{}", oxide.name);
        }
    }

    /// A brushed finish is anisotropic about `+y`, across its grooves.
    #[test]
    fn brushing_roughens_across_the_grooves() {
        let m = build(&Copper, 64, Bind::new().scalar("brushing", 0.5));
        assert_eq!(
            m.constant(ChannelId::Param(Param::GeometryTangent)),
            Some(Value::Vector3(Vec3::Y))
        );
        assert_eq!(
            m.constant(ChannelId::Param(Param::SpecularRoughnessAnisotropy)),
            Some(Value::Scalar(0.4))
        );
        let rolled = build(&Copper, 64, Bind::new().scalar("brushing", 0.0));
        assert!(rolled.param(Param::SpecularRoughnessAnisotropy).is_none());
    }

    /// Corrosion is a dielectric laid over the metal: where it wins the
    /// surface, metalness is at most a half, and mostly nearly 0; it is
    /// rougher than the fresh metal. Age 0 is the metal itself.
    #[test]
    fn corrosion_replaces_metal_with_a_rough_dielectric() {
        // Copper well into verdigris with bare metal left, steel rusted
        // through rather than flashed.
        for (asset, product, age) in [
            (&WeatheredCopper as &dyn Module, PATINA, 0.6),
            (&RustedSteel as &dyn Module, RUST, 0.9),
        ] {
            let m = build(asset, 128, Bind::new().scalar("age", age));
            let name = asset.interface().id.name;
            let (covered_rough, covered_metal, covered) = means(&m, |i| surface(&m, i) == product);
            let fresh = build(asset, 128, Bind::new().scalar("age", 0.0));
            let (bare_rough, bare_metal, _) = means(&fresh, |_| true);
            assert!(bare_metal > 0.999, "{name}: fresh metal {bare_metal}");
            assert!(covered > 1000, "{name}: {covered} covered");
            assert!(covered_metal < 0.15, "{name}: metalness {covered_metal}");
            for i in 0..m.grid().len() {
                if surface(&m, i) == product {
                    assert!(scalar(&m, Param::BaseMetalness, i) <= 0.5 + 1e-5);
                }
            }
            assert!(
                covered_rough > bare_rough + 0.2,
                "{name}: {covered_rough} vs {bare_rough}"
            );
        }
        let copper = build(&Copper, 64, Bind::new());
        let mut cx = Context::new(super::glazed::tile(64), &NoResources);
        let unaged = cx
            .instantiate(
                &Patina,
                "m",
                Bind::new()
                    .material("base", copper.clone())
                    .scalar("age", 0.0),
            )
            .unwrap()
            .take_material("material")
            .unwrap();
        assert_eq!(unaged.digest(), copper.digest());
    }

    /// Aging is monotone: as `age` rises, the surface roughens and (bare
    /// metal) loses metalness, and copper darkens before it greens.
    #[test]
    fn weathering_roughens_and_dulls_with_age() {
        for asset in [&WeatheredCopper as &dyn Module, &RustedSteel] {
            let name = asset.interface().id.name;
            let stats: Vec<(f32, f32)> = [0.0, 0.3, 0.6, 1.0]
                .iter()
                .map(|&age| {
                    let m = build(asset, 64, Bind::new().scalar("age", age));
                    let (r, k, _) = means(&m, |_| true);
                    (r, k)
                })
                .collect();
            for pair in stats.windows(2) {
                assert!(pair[1].0 > pair[0].0, "{name} roughness {stats:?}");
                assert!(pair[1].1 < pair[0].1 + 1e-4, "{name} metalness {stats:?}");
            }
        }
        let luminance = |age: f32| {
            let m = build(&WeatheredCopper, 64, Bind::new().scalar("age", age));
            let (mut y, mut n) = (0.0, 0);
            for i in (0..m.grid().len()).filter(|&i| surface(&m, i) == METAL) {
                if let Value::Vector3(c) = m.value(ChannelId::Param(Param::BaseColor), i) {
                    y += crate::modules::calibration::luminance(c);
                    n += 1;
                }
            }
            y / n as f32
        };
        assert!(luminance(0.5) < 0.5 * luminance(0.0), "tarnish darkens");
    }

    /// Paint hides the metal; its chips bare it; and rust grows only on
    /// what is bare and the paint it creeps under, not under sound paint.
    #[test]
    fn paint_chips_bare_the_steel_and_rust_follows_the_chips() {
        let steel = build(&Steel, 128, Bind::new());
        let mut cx = Context::new(super::glazed::tile(128), &NoResources);
        let painted = cx
            .instantiate(
                &Paint,
                "m",
                Bind::new().material("base", steel).scalar("chipping", 0.05),
            )
            .unwrap()
            .take_material("material")
            .unwrap();
        let n = painted.grid().len();
        let (_, paint_metal, paint) = means(&painted, |i| surface(&painted, i) == PAINT);
        assert!(
            paint_metal < 1e-3 && paint > n / 2,
            "{paint_metal} over {paint}"
        );
        let bared = (0..n)
            .filter(|&i| scalar(&painted, Param::BaseMetalness, i) > 0.5)
            .count();
        assert!(bared > 0 && bared < n / 20, "{bared} bare texels");

        let rusted = cx
            .instantiate(
                &Rust,
                "rust",
                Bind::new()
                    .material("base", painted.clone())
                    .scalar("age", 0.5)
                    .scalar("wetness", 0.0),
            )
            .unwrap()
            .take_material("material")
            .unwrap();
        let metal = painted
            .scalar(ChannelId::Param(Param::BaseMetalness))
            .unwrap();
        let from_bare = DistanceTransform { threshold: 0.5 }.apply(&metal).unwrap();
        let mut rust = 0;
        for i in 0..n {
            if surface(&rusted, i) == RUST {
                rust += 1;
                assert!(
                    from_bare.values()[i] < 0.02,
                    "rust under sound paint at {i}"
                );
            }
        }
        assert!(rust > bared / 4, "{rust} rusted of {bared} bared");
    }

    /// Each metal's package survives a round trip, realizes its native
    /// asset's bits, and rebuilds every preset as the native asset does
    /// with the preset's values bound; every preset tiles.
    #[test]
    fn metal_packages_match_their_native_assets_at_every_preset() {
        let engine = registry();
        let packages: [(&str, &dyn Module); 3] = [
            (WEATHERED_COPPER, &WeatheredCopper),
            (RUSTED_STEEL, &RustedSteel),
            (PAINTED_STEEL, &PaintedSteel),
        ];
        for (source, native) in packages {
            let package = Package::from_json(source).expect("reads");
            assert_eq!(Package::from_json(&package.to_json()).unwrap(), package);
            let compiled = engine.compile(&package).expect("compiles");
            // The package exposes the native asset's parameters, but its seed.
            let names = |m: &dyn Module| -> Vec<(alloc::string::String, _)> {
                m.interface()
                    .params
                    .into_iter()
                    .filter(|p| p.name != "seed")
                    .map(|p| (p.name.into_owned(), p.default))
                    .collect()
            };
            assert_eq!(names(&compiled), names(native));
            assert!(package.presets.len() >= 3);
            let presets = package.presets.iter().map(|p| Some(p.name.as_str()));
            for preset in core::iter::once(None).chain(presets) {
                let bind = preset.map_or_else(Bind::new, |p| compiled.preset(p).unwrap());
                let packaged = build(&compiled, 64, bind.clone());
                let built = build(native, 64, bind);
                assert_eq!(packaged.digest(), built.digest(), "{preset:?}");
                assert_eq!(packaged.tiling(), Tiling::BOTH, "{preset:?}");
                assert!(packaged.range_violations().is_empty(), "{preset:?}");
            }
        }
    }

    /// Painted steel's `weather` derives four step parameters: through the
    /// package's expressions (one tape program) and through the native
    /// asset's Rust, the same bits, alone and over every preset; and at 0
    /// it changes nothing.
    #[test]
    fn painted_steels_weather_derives_as_the_native_asset_does() {
        let package = Package::from_json(PAINTED_STEEL).expect("reads");
        let compiled = registry().compile(&package).expect("compiles");
        let program = compiled.logic().expect("expressions");
        assert_eq!(program.outputs().len(), 4);
        let presets = package.presets.iter().map(|p| Some(p.name.as_str()));
        for preset in core::iter::once(None).chain(presets) {
            let base = preset.map_or_else(Bind::new, |p| compiled.preset(p).unwrap());
            let unweathered = build(&compiled, 16, base.clone());
            for weather in [0.0, 0.15, 0.5, 0.999, 1.0] {
                let bind = base.clone().scalar("weather", weather);
                let packaged = build(&compiled, 16, bind.clone());
                let built = build(&PaintedSteel, 16, bind);
                assert_eq!(packaged.digest(), built.digest(), "{preset:?} at {weather}");
                if weather == 0.0 && preset != Some("weathered") {
                    assert_eq!(packaged.digest(), unweathered.digest(), "{preset:?}");
                }
            }
        }
    }
}

/// The library's programs evaluated in runs, through the batch entry
/// points rasters and materials use, give each point's bits.
mod rows {
    use alloc::format;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;

    use dapple_field::program::{ChartSample, Op, ProgramBuilder, ValueProgram};
    use dapple_field::scoped::{Scope, Shape};
    use dapple_field::{Domain, Footprint, ScalarField, SolidField, Value};
    use glam::{Vec2, Vec3};

    use crate::{brick, glazed_brick, gravel, oak, parquet};

    fn bits(v: Value) -> String {
        // NaN is NaN, whatever its payload.
        let f = |x: f32| if x.is_nan() { u32::MAX } else { x.to_bits() };
        match v {
            Value::Scalar(x) => format!("s{}", f(x)),
            Value::Id(x) => format!("i{x}"),
            Value::Vector2(x) => format!("v{:?}", x.to_array().map(f)),
            Value::Vector3(x) => format!("c{:?}", x.to_array().map(f)),
        }
    }

    /// Random points near and far on both signs, then signed zeros, NaN
    /// and huge coordinates.
    fn points() -> Vec<Vec2> {
        let mut state = 0x726f_7773_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 40) as f32 / (1_u64 << 24) as f32 * 2.0 - 1.0
        };
        let mut points: Vec<Vec2> = (0..90)
            .map(|i| Vec2::new(next(), next()) * [1.0, 3.0, 1e4][i % 3])
            .collect();
        for v in [0.0, -0.0, f32::NAN, 1e12, -1e12] {
            points.push(Vec2::new(v, 0.3));
            points.push(Vec2::new(0.7, v));
        }
        points
    }

    fn footprints() -> [Footprint; 4] {
        [0.0, 1.0 / 256.0, 0.03, 1e6].map(|w| Footprint::new(w).unwrap())
    }

    fn assert_rows(name: &str, program: &ValueProgram) {
        let points = points();
        let output = program.program().output();
        let mut kept = program.evaluator();
        for footprint in footprints() {
            let mut runs = vec![Value::Id(u32::MAX); points.len()];
            for (run, out) in points.chunks(37).zip(runs.chunks_mut(37)) {
                kept.eval_value_batch(run, footprint, out);
            }
            let mut whole = vec![Value::Id(u32::MAX); points.len()];
            program.eval_batch(&points, footprint, &mut whole);
            for (i, &p) in points.iter().enumerate() {
                let alone = bits(program.eval(p, footprint));
                let recursive = bits(program.program().eval_value(output, p, footprint));
                assert_eq!(alone, recursive, "{name} at {p}");
                assert_eq!(bits(runs[i]), alone, "{name} run at {p}");
                assert_eq!(bits(whole[i]), alone, "{name} batch at {p}");
            }
            if program.output_type().is_scalar() {
                let mut scalars = vec![0.0; points.len()];
                program
                    .program()
                    .eval_batch(&points, footprint, &mut scalars);
                for (&v, &p) in scalars.iter().zip(&points) {
                    let alone = bits(Value::Scalar(program.program().eval(p, footprint)));
                    assert_eq!(bits(Value::Scalar(v)), alone, "{name} scalar at {p}");
                }
            }
        }
    }

    #[test]
    fn library_programs_evaluate_in_runs_as_alone() {
        for domain in [Domain::periodic(1, 1).unwrap(), Domain::Plane] {
            let height = oak::bark_height(domain).unwrap();
            let as_value = |b: &mut ProgramBuilder, program| b.import(program).unwrap();
            let mut b = ProgramBuilder::new();
            let out = as_value(&mut b, &height);
            assert_rows("bark height", &b.finish_value(out).unwrap());
            assert_rows("bark color", &oak::bark_color(domain, &height).unwrap());
            assert_rows("brick color", &brick::color(domain).unwrap());
            assert_rows("gravel color", &gravel::color(domain).unwrap());
            assert_rows("parquet color", &parquet::color(domain).unwrap());
        }
        // The solid oak as the wood module cuts its board.
        let mut b = ProgramBuilder::new();
        let solid = oak::wood_color(&mut b).unwrap();
        let board = b
            .add(Op::Slice {
                input: solid,
                origin: [0.14, 0.0, 0.0],
                u: [0.0, 0.0, 1.0],
                v: [0.0, 1.0, 0.0],
                domain: Domain::Plane,
            })
            .unwrap();
        assert_rows("oak board", &b.finish_value(board).unwrap());
        // And cut on the face, as the wood module cuts it.
        let mut b = ProgramBuilder::new();
        let face = oak::wood_face_color(&mut b, [0.14, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
            .unwrap();
        assert_rows("oak face", &b.finish_value(face).unwrap());
    }

    /// At a point footprint, the face cut is the solid sliced, bit for bit:
    /// slicing commutes with everything after it.
    #[test]
    fn the_oak_face_is_the_sliced_solid_at_a_point() {
        let (origin, u, v) = ([0.11, 0.07, 0.3], [0.0, 0.0, 1.0], [-0.54, 0.84, 0.0]);
        let mut b = ProgramBuilder::new();
        let solid = oak::wood_color(&mut b).unwrap();
        let sliced = b
            .add(Op::Slice {
                input: solid,
                origin,
                u,
                v,
                domain: Domain::Plane,
            })
            .unwrap();
        let sliced = b.finish_value(sliced).unwrap();
        let mut b = ProgramBuilder::new();
        let face = oak::wood_face_color(&mut b, origin, u, v).unwrap();
        let face = b.finish_value(face).unwrap();
        for p in points() {
            assert_eq!(
                bits(face.eval(p, Footprint::POINT)),
                bits(sliced.eval(p, Footprint::POINT)),
                "at {p}"
            );
        }
    }

    #[test]
    fn oak_charts_evaluate_each_sample_as_alone() {
        let points: Vec<Vec3> = points()
            .into_iter()
            .map(|p| Vec3::new(0.14, p.y, p.x))
            .collect();
        for channel in oak::wood_channels().unwrap() {
            for footprint in footprints() {
                let samples: Vec<ChartSample> = points
                    .iter()
                    .map(|&p| ChartSample::isotropic(p, footprint))
                    .collect();
                let mut out = vec![0.0; samples.len()];
                channel.eval_chart(&samples, &mut out);
                for (&v, &p) in out.iter().zip(&points) {
                    assert_eq!(
                        bits(Value::Scalar(v)),
                        bits(Value::Scalar(channel.eval(p, footprint))),
                        "oak at {p}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_glazed_brick_program_evaluates_in_runs_as_alone() {
        let program = glazed_brick::program().unwrap();
        let input = |i: u32, lane: usize| {
            let decl = &program.inputs()[i as usize];
            let lane = if decl.scope.within(Scope::Element) {
                0
            } else {
                lane
            };
            let h = |axis: usize| {
                let x = ((lane * 7 + axis * 13 + i as usize * 29) % 101) as f32 / 50.0 - 1.0;
                if lane % 17 == 5 { x * 1e4 } else { x }
            };
            match Shape::of(decl.port) {
                Shape::Scalar => Value::Scalar(h(0)),
                Shape::Id => Value::Id(u32::try_from(lane % 4).unwrap()),
                Shape::Vector2 => Value::Vector2(Vec2::new(h(0), h(1))),
                Shape::Vector3 => Value::Vector3(Vec3::new(h(0), h(1), h(2))),
            }
        };
        let footprint = Footprint::new(1.0 / 512.0).unwrap();
        let hoist = |s: Scope| s.within(Scope::Element);
        let mut hoisted = vec![None; program.nodes().len()];
        program.evaluate(&mut hoisted, hoist, &mut |i| input(i, 0), Footprint::POINT);
        let mut kept = program.evaluator();
        kept.hoist(hoist, &mut |i| input(i, 0), Footprint::POINT);
        let mut first = 0;
        for len in [1, 64, 0, 200] {
            kept.eval_batch(
                len,
                &mut |i, values| {
                    for (lane, value) in values.iter_mut().enumerate() {
                        *value = input(i, first + lane);
                    }
                },
                footprint,
            );
            for lane in 0..len {
                let mut values = hoisted.clone();
                program.evaluate(
                    &mut values,
                    |_| true,
                    &mut |i| input(i, first + lane),
                    footprint,
                );
                for (o, decl) in program.outputs().iter().enumerate() {
                    let alone = values[decl.node.index()].unwrap();
                    assert_eq!(bits(kept.output(o)[lane]), bits(alone), "{}", decl.name);
                }
            }
            first += len;
        }
    }
}
