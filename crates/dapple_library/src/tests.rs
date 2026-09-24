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

    fn modules(d: &Diagnostics) -> Vec<&'static str> {
        let mut v: Vec<_> = d.instances().map(|(_, m)| m.name).collect();
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
        // same versioned identity, whichever asset instantiated it.
        for d in [&sill_d, &board_d, &threshold_d] {
            for (path, id) in d.instances() {
                assert_eq!(id.version, 1, "{path}");
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
    use dapple_material::module::{Bind, Context, Input};
    use dapple_material::resource::{
        NoResources, Resolved, ResourceError, ResourceHost, ResourceRef,
    };
    use dapple_material::{Aux, Channel, ChannelId, Material, Param};
    use dapple_raster::typed::ReductionPolicy;
    use glam::{Vec2, Vec3};

    use crate::modules::{ByExample, EdgeWear, Moss};

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
