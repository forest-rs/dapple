// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Seeded edit histories checked bit for bit against graphs built fresh.
//!
//! A history starts from one graph holding every node kind and applies
//! random actions: local edits (a moved disk) and global ones (a new noise
//! seed, a new resolution), edits that change what sample nodes read, edits
//! the graph must refuse, tile budgets, and failures, both an invalid
//! parameter and a failure injected where a node's computation can fail,
//! each followed later by recovery. After every run that settles without
//! error, every node's output must equal, bit for bit, the output of a graph
//! built fresh from the same recipe and run once, and the graph's recipe
//! must be the one the accepted edits describe. A failing history is shrunk
//! to a short list of actions before it is reported.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use dapple_field::{Basis, CellOutput};

use super::*;

/// Histories checked per tile size, one per seed; set
/// `DAPPLE_GRAPH_HISTORIES` when building the tests to check more.
fn seeds() -> u64 {
    option_env!("DAPPLE_GRAPH_HISTORIES").map_or(8, |n| n.parse().expect("a number of histories"))
}

/// Settled runs per history.
const STEPS: usize = 40;

/// `SplitMix64`: small, deterministic, and plenty for picking edits.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % n as u64).expect("below n")
    }

    fn pick<T: Clone>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())].clone()
    }

    /// A point of the unit period, on a 1/1024 lattice.
    fn unit(&mut self) -> f32 {
        f32::from(u16::try_from(self.below(1024)).expect("below 1024")) / 1024.0
    }
}

fn domain() -> Domain {
    Domain::periodic(1, 1).unwrap()
}

fn noise(seed: u64, frequency: f32) -> Op {
    Op::Noise {
        basis: Basis::Gradient,
        domain: domain(),
        frequency: [frequency; 2],
        seed,
    }
}

/// Cell values jump between cells, so warping by them has no static bound.
fn cells(seed: u64) -> Op {
    Op::Cellular {
        domain: domain(),
        frequency: [4.0, 4.0],
        jitter: 1.0,
        seed,
        output: CellOutput::CellValue,
    }
}

fn disk(center: [f32; 2], radius: f32) -> Op {
    Op::Disk {
        domain: domain(),
        center,
        radius,
        softness: 0.01,
    }
}

fn warp(amount: f32) -> Op {
    Op::Warp {
        input: operand(0),
        dx: operand(1),
        dy: operand(2),
        amount,
    }
}

const fn blur(sigma: f32) -> RasterParams {
    RasterParams::Blur(GaussianBlur { sigma })
}

/// Every node kind, wired so edits reach each through the others:
///
/// - `noise` and `disk` → `height` → `map`, blurred into `soft`, whose box
///   mips `mip1` and `mip2` are sampled back as `sampled`, warped by `dx`
///   and `dy`, and realized again as `again` and as `warp.normals`;
/// - `soft` → `relief` (height to normal), `map` → `dist` (a global
///   distance transform), sampled as `dist.sample` and added to `noise` in
///   `blend` → `blend.map`;
/// - `height` → `normals` (analytic normals);
/// - `noise` → `mask` → `mask.map` → `mask.red` (a reduction), and
///   `mask` → `id` → `id.map` → `id.red`, both sampled nearest as
///   `id.sample` and realized again as `id.again` (typed, computed whole).
fn initial_recipe() -> Recipe {
    let (a, b) = (operand(0), operand(1));
    let mut g = MaterialGraph::new();
    let noise_ = g.field("noise", noise(1, 8.0), &[]).unwrap();
    let disk_ = g.field("disk", disk([0.2, 0.3], 0.05), &[]).unwrap();
    let height = g
        .field("height", Op::Max { a, b }, &[noise_, disk_])
        .unwrap();
    let dx = g.field("dx", noise(7, 4.0), &[]).unwrap();
    let dy = g.field("dy", noise(8, 4.0), &[]).unwrap();
    let map = g.realize("map", height, 24, 24).unwrap();
    let soft = g.raster("soft", blur(0.02), map).unwrap();
    let dist = g
        .raster(
            "dist",
            RasterParams::DistanceTransform(DistanceTransform { threshold: 0.6 }),
            map,
        )
        .unwrap();
    g.raster(
        "relief",
        RasterParams::HeightToNormal(HeightToNormal { scale: 0.05 }),
        soft,
    )
    .unwrap();
    let mip1 = g.mip("mip1", soft, Filter::Box).unwrap();
    let mip2 = g.mip("mip2", mip1, Filter::Box).unwrap();
    let sampled = g
        .sample("sampled", &[soft, mip1, mip2], SamplePolicy::Linear)
        .unwrap();
    let warped = g.field("warp", warp(0.02), &[sampled, dx, dy]).unwrap();
    g.realize("again", warped, 24, 24).unwrap();
    g.normals("warp.normals", warped, (24, 24), 0.05).unwrap();
    g.normals("normals", height, (24, 24), 0.05).unwrap();
    let dist_sample = g
        .sample("dist.sample", &[dist], SamplePolicy::Linear)
        .unwrap();
    let blend = g
        .field("blend", Op::Add { a, b }, &[dist_sample, noise_])
        .unwrap();
    g.realize("blend.map", blend, 24, 24).unwrap();
    let mask = g.field("mask", Op::AsMask { input: a }, &[noise_]).unwrap();
    let mask_map = g.realize("mask.map", mask, 24, 24).unwrap();
    g.reduce("mask.red", mask_map, ReductionPolicy::Average, 1)
        .unwrap();
    let id = g
        .field(
            "id",
            Op::ToId {
                input: a,
                levels: 4,
            },
            &[mask],
        )
        .unwrap();
    let id_map = g.realize("id.map", id, 16, 16).unwrap();
    let id_red = g
        .reduce("id.red", id_map, ReductionPolicy::IdMode, 1)
        .unwrap();
    let id_sample = g
        .sample("id.sample", &[id_map, id_red], SamplePolicy::Nearest)
        .unwrap();
    g.realize("id.again", id_sample, 16, 16).unwrap();
    g.recipe()
}

/// Every node, for picking where to inject a failure.
const LABELS: [&str; 27] = [
    "noise",
    "disk",
    "height",
    "dx",
    "dy",
    "map",
    "soft",
    "dist",
    "relief",
    "mip1",
    "mip2",
    "sampled",
    "warp",
    "again",
    "warp.normals",
    "normals",
    "dist.sample",
    "blend",
    "blend.map",
    "mask",
    "mask.map",
    "mask.red",
    "id",
    "id.map",
    "id.red",
    "id.sample",
    "id.again",
];

#[derive(Clone, Debug)]
enum Action {
    /// An edit the graph accepts.
    Edit(&'static str, Params),
    /// An edit the graph refuses, leaving everything as it was.
    Refuse(&'static str, Params),
    /// A new tile budget.
    Budget(Option<u64>),
    /// Make a node's computation fail until [`Action::Heal`].
    Fail(&'static str),
    /// Stop injecting failures.
    Heal,
    /// Run until no tiles are pending, then check against a fresh graph.
    Settle,
}

/// A valid operation for `soft`.
fn soft_params(rng: &mut Rng) -> Params {
    Params::Raster(match rng.below(3) {
        0 => blur(rng.pick(&[0.01, 0.02, 0.03])),
        1 => RasterParams::Morphology(Morphology {
            op: rng.pick(&[
                MorphologyOp::Dilate,
                MorphologyOp::Erode,
                MorphologyOp::Open,
                MorphologyOp::Close,
            ]),
            radius: rng.pick(&[0.02, 0.04]),
        }),
        _ => RasterParams::AmbientOcclusion(AmbientOcclusion {
            radius: 0.05,
            directions: 4,
            scale: rng.pick(&[0.5, 1.0]),
        }),
    })
}

fn random_edit(rng: &mut Rng) -> Action {
    let (a, b) = (operand(0), operand(1));
    let edit = Action::Edit;
    match rng.below(18) {
        0..=3 => edit(
            "disk",
            Params::Field(disk(
                [rng.unit(), rng.unit()],
                rng.pick(&[0.03, 0.05, 0.08]),
            )),
        ),
        4 => edit(
            "noise",
            Params::Field(noise(rng.pick(&[1, 2, 3]), rng.pick(&[4.0, 8.0]))),
        ),
        5 => edit(
            "height",
            Params::Field(rng.pick(&[
                Op::Max { a, b },
                Op::Min { a, b },
                Op::Add { a, b },
                Op::Sub { a, b },
            ])),
        ),
        6 => {
            let label = rng.pick(&["dx", "dy"]);
            let seed = rng.pick(&[7, 8, 9]);
            let op = if rng.below(4) == 0 {
                cells(seed)
            } else {
                noise(seed, 4.0)
            };
            edit(label, Params::Field(op))
        }
        7 => edit(
            "warp",
            Params::Field(warp(rng.pick(&[0.0, 0.01, 0.02, 0.04]))),
        ),
        8 => edit("soft", soft_params(rng)),
        9 => edit(
            "dist",
            Params::Raster(RasterParams::DistanceTransform(DistanceTransform {
                threshold: rng.pick(&[0.3, 0.6, 0.9]),
            })),
        ),
        10 => edit(
            "relief",
            Params::Raster(RasterParams::HeightToNormal(HeightToNormal {
                scale: rng.pick(&[0.02, 0.05]),
            })),
        ),
        11 | 12 => {
            let label = rng.pick(&["map", "map", "again", "blend.map", "mask.map"]);
            // 24 and 25 texels mip to the same 12.
            let (width, height) = rng.pick(&[(24, 24), (25, 24), (24, 25), (20, 12), (16, 16)]);
            edit(label, Params::Realize { width, height })
        }
        13 => {
            let (width, height) = rng.pick(&[(24, 24), (16, 12)]);
            edit(
                rng.pick(&["normals", "warp.normals"]),
                Params::Normals {
                    width,
                    height,
                    scale: rng.pick(&[0.02, 0.05]),
                },
            )
        }
        14 => edit(
            rng.pick(&["mip1", "mip2"]),
            Params::Mip(rng.pick(&[Filter::Box, Filter::Kaiser])),
        ),
        15 => edit(
            rng.pick(&["sampled", "dist.sample"]),
            Params::Sample(rng.pick(&[SamplePolicy::Linear, SamplePolicy::Nearest])),
        ),
        16 => edit(
            "mask.red",
            Params::Reduce {
                policy: rng.pick(&[
                    ReductionPolicy::Average,
                    ReductionPolicy::ThresholdCoverage { cutoff: 0.5 },
                ]),
                level: rng.pick(&[1, 2]),
            },
        ),
        _ => {
            if rng.below(2) == 0 {
                edit(
                    "id.red",
                    Params::Reduce {
                        policy: rng.pick(&[ReductionPolicy::IdMode, ReductionPolicy::IdPoint]),
                        level: 1,
                    },
                )
            } else {
                edit(
                    "id",
                    Params::Field(Op::ToId {
                        input: a,
                        levels: rng.pick(&[3, 4, 6]),
                    }),
                )
            }
        }
    }
}

/// An edit every state of the graph refuses.
fn refused_edit(rng: &mut Rng) -> Action {
    let refuse = Action::Refuse;
    match rng.below(5) {
        // `soft` cannot blur a vector.
        0 => refuse("height", Params::Field(Op::Position3)),
        // A mask has no mode.
        1 => refuse(
            "mask.red",
            Params::Reduce {
                policy: ReductionPolicy::IdMode,
                level: 1,
            },
        ),
        2 => refuse(
            "mask.red",
            Params::Reduce {
                policy: ReductionPolicy::Average,
                level: 0,
            },
        ),
        // Identifiers are never interpolated.
        3 => refuse("id.sample", Params::Sample(SamplePolicy::Linear)),
        // `mip1` cannot filter normals.
        _ => refuse(
            "soft",
            Params::Raster(RasterParams::HeightToNormal(HeightToNormal { scale: 0.05 })),
        ),
    }
}

/// A seeded history of `steps` settles.
fn history(seed: u64, steps: usize) -> Vec<Action> {
    let mut rng = Rng(seed);
    let mut actions = vec![Action::Settle];
    // Settles left before a failure, injected or a broken `soft`, is fixed.
    let mut heal_in = None;
    let mut fix_in = None;
    for _ in 0..steps {
        for _ in 0..=rng.below(3) {
            actions.push(match rng.below(20) {
                0..=13 => random_edit(&mut rng),
                14 => refused_edit(&mut rng),
                15 | 16 => {
                    Action::Budget(rng.pick(&[None, None, Some(1), Some(3), Some(8), Some(20)]))
                }
                17 | 18 => {
                    heal_in.get_or_insert(rng.below(3));
                    Action::Fail(rng.pick(&LABELS))
                }
                _ => {
                    fix_in.get_or_insert(rng.below(3));
                    Action::Edit("soft", Params::Raster(blur(-1.0)))
                }
            });
        }
        actions.push(Action::Settle);
        if let Some(n) = &mut heal_in {
            if *n == 0 {
                actions.push(Action::Heal);
                heal_in = None;
            } else {
                *n -= 1;
            }
        }
        if let Some(n) = &mut fix_in {
            if *n == 0 {
                actions.push(Action::Edit("soft", soft_params(&mut rng)));
                fix_in = None;
            } else {
                *n -= 1;
            }
        }
    }
    actions.extend([
        Action::Heal,
        Action::Edit("soft", Params::Raster(blur(0.02))),
        Action::Settle,
    ]);
    actions
}

/// Applies `params` to `node` through the public setter for its kind.
fn set(graph: &mut MaterialGraph, node: NodeId, params: &Params) -> Result<(), MaterialError> {
    match params.clone() {
        Params::Field(op) => graph.set_field_op(node, op),
        Params::Realize { width, height } => graph.set_resolution(node, width, height),
        Params::Raster(params) => graph.set_raster_params(node, params),
        Params::Mip(filter) => graph.set_mip_filter(node, filter),
        Params::Reduce { policy, level } => graph.set_reduction(node, policy, level),
        Params::Sample(policy) => graph.set_sample_policy(node, policy),
        Params::Normals {
            width,
            height,
            scale,
        } => graph.set_normals(node, (width, height), scale),
    }
}

/// Writes `params` into a recipe step of the same kind.
fn assign(step: &mut Step, params: &Params) {
    match (step, params.clone()) {
        (Step::Field { op, .. }, Params::Field(new)) => *op = new,
        (
            Step::Realize { width, height, .. },
            Params::Realize {
                width: w,
                height: h,
            },
        ) => {
            (*width, *height) = (w, h);
        }
        (Step::Raster { params, .. }, Params::Raster(new)) => *params = new,
        (Step::Mip { filter, .. }, Params::Mip(new)) => *filter = new,
        (
            Step::Reduce { policy, level, .. },
            Params::Reduce {
                policy: p,
                level: l,
            },
        ) => {
            (*policy, *level) = (p, l);
        }
        (Step::Sample { policy, .. }, Params::Sample(new)) => *policy = new,
        (
            Step::Normals {
                width,
                height,
                scale,
                ..
            },
            Params::Normals {
                width: w,
                height: h,
                scale: s,
            },
        ) => (*width, *height, *scale) = (w, h, s),
        (step, params) => panic!("{params:?} does not fit {step:?}"),
    }
}

/// A raster's grid and exact texel bits.
fn raster_bits(data: &RasterData) -> Vec<u32> {
    let (width, height, origin, texel, edge) = data.shape();
    let mut bits = vec![
        width,
        height,
        origin.x.to_bits(),
        origin.y.to_bits(),
        texel.x.to_bits(),
        texel.y.to_bits(),
        u32::from(edge == Edge::Wrap),
    ];
    let floats = |values: &mut Vec<u32>, v: &mut dyn Iterator<Item = f32>| {
        values.extend(v.map(f32::to_bits));
    };
    match data {
        RasterData::Scalar(r) => floats(&mut bits, &mut r.values().iter().copied()),
        RasterData::Vector3(r) => floats(&mut bits, &mut r.values().iter().flatten().copied()),
        RasterData::Typed(r) => match r.storage() {
            Storage::F32(r) => floats(&mut bits, &mut r.values().iter().copied()),
            Storage::U32(r) => bits.extend(r.values()),
            Storage::F32x2(r) => floats(&mut bits, &mut r.values().iter().flatten().copied()),
            Storage::F32x3(r) => floats(&mut bits, &mut r.values().iter().flatten().copied()),
        },
    }
    bits
}

/// The texel bits of every image a program samples, which its fingerprint
/// does not cover.
fn image_bits(program: &ValueProgram) -> Vec<u32> {
    let mut bits = Vec::new();
    for (_, op) in program.program().nodes() {
        if let Op::Sample { image } = op {
            for level in image.levels() {
                bits.extend(level.values().iter().map(|v| v.to_bits()));
                bits.extend(level.id_values());
            }
        }
    }
    bits
}

/// Where two graphs' values for one node differ, if they do.
fn difference(ours: Option<&GraphValue>, fresh: Option<&GraphValue>) -> Option<String> {
    match (ours, fresh) {
        (Some(GraphValue::Raster(a)), Some(GraphValue::Raster(b))) => {
            if a.port != b.port {
                Some(format!("port {:?}, fresh {:?}", a.port, b.port))
            } else if a.fingerprint != b.fingerprint {
                Some("raster fingerprint".into())
            } else if raster_bits(&a.data) != raster_bits(&b.data) {
                Some("raster texels".into())
            } else {
                None
            }
        }
        (Some(GraphValue::Field(a)), Some(GraphValue::Field(b))) => {
            if a.program != b.program {
                Some("field program".into())
            } else if image_bits(&a.program) != image_bits(&b.program) {
                Some("sampled image texels".into())
            } else {
                None
            }
        }
        (ours, fresh) => Some(format!(
            "value {:?}, fresh {:?}",
            ours.map(core::mem::discriminant),
            fresh.map(core::mem::discriminant)
        )),
    }
}

/// A graph under edit, and the recipe its accepted edits describe.
struct Replay {
    tile_size: u32,
    graph: MaterialGraph,
    ids: BTreeMap<String, NodeId>,
    recipe: Recipe,
    /// Settles that compared against a fresh graph.
    checked: usize,
}

impl Replay {
    fn new(tile_size: u32) -> Self {
        let recipe = initial_recipe();
        let (graph, ids) = recipe.build(tile_size).unwrap();
        Self {
            tile_size,
            graph,
            ids,
            recipe,
            checked: 0,
        }
    }

    fn node(&self, label: &str) -> NodeId {
        self.ids[label]
    }

    /// The tile space of `label`'s node, which keys injected failures.
    fn key(&self, label: &str) -> u32 {
        let node = self.node(label);
        let index = self.graph.order.iter().position(|&n| n == node).unwrap();
        u32::try_from(index).unwrap()
    }

    fn failing(&mut self) -> &mut alloc::collections::BTreeSet<u32> {
        &mut self.graph.graph.executor_mut().tiles.failing
    }

    fn apply(&mut self, action: &Action) -> Result<(), String> {
        match action {
            Action::Edit(label, params) => {
                let node = self.node(label);
                set(&mut self.graph, node, params).map_err(|e| format!("edit refused: {e}"))?;
                let step = self.recipe.nodes.iter_mut().find(|n| n.label == *label);
                assign(&mut step.unwrap().step, params);
            }
            Action::Refuse(label, params) => {
                let node = self.node(label);
                match set(&mut self.graph, node, params) {
                    Err(MaterialError::Refused { .. }) => {}
                    other => return Err(format!("expected a refusal, found {other:?}")),
                }
                if self.graph.recipe() != self.recipe {
                    return Err("a refused edit changed the graph".into());
                }
            }
            Action::Budget(budget) => self.graph.set_tile_budget(*budget),
            Action::Fail(label) => {
                let key = self.key(label);
                self.failing().insert(key);
            }
            Action::Heal => self.failing().clear(),
            Action::Settle => self.settle()?,
        }
        Ok(())
    }

    /// Runs until no tiles are pending, then compares every node with a
    /// fresh graph. A run may fail only while a failure is injected or when
    /// the fresh graph fails too.
    fn settle(&mut self) -> Result<(), String> {
        let ran = self.run_until_settled()?;
        let injected = !self.graph.graph.executor().tiles.failing.is_empty();
        if ran.is_err() && injected {
            return Ok(());
        }
        let (mut fresh, fresh_ids) = self
            .recipe
            .build(self.tile_size)
            .map_err(|e| format!("fresh build: {e}"))?;
        match (ran, fresh.run()) {
            (Ok(()), Ok(_)) => {}
            (Err(_), Err(_)) => return Ok(()),
            (Err(e), Ok(_)) => return Err(format!("failed where a fresh graph runs: {e}")),
            (Ok(()), Err(e)) => return Err(format!("ran where a fresh graph fails: {e}")),
        }
        if self.graph.recipe() != self.recipe {
            return Err("the graph's recipe is not the edited one".into());
        }
        for node in &self.recipe.nodes {
            let ours = self.graph.value(self.ids[&node.label]);
            let theirs = fresh.value(fresh_ids[&node.label]);
            if let Some(difference) = difference(ours, theirs) {
                return Err(format!("{}: {difference}", node.label));
            }
        }
        self.checked += 1;
        Ok(())
    }

    /// Runs until a tile budget leaves nothing pending. The outer error is
    /// a broken promise; the inner one a run that failed.
    fn run_until_settled(&mut self) -> Result<Result<(), MaterialError>, String> {
        for _ in 0..2000 {
            if let Err(error) = self.graph.run() {
                return Ok(Err(error));
            }
            if self.graph.tile_report().pending_tiles == 0 {
                return match self.graph.run() {
                    Ok(summary) if summary.executed_nodes == 0 => Ok(Ok(())),
                    Ok(summary) => Err(format!("not settled: {summary:?}")),
                    Err(error) => Err(format!("a settled graph failed: {error}")),
                };
            }
        }
        Err("budgeted runs did not converge".into())
    }
}

/// Replays `actions`, returning how many settles compared equal, or the
/// first difference.
fn replay(tile_size: u32, actions: &[Action]) -> Result<usize, String> {
    let mut r = Replay::new(tile_size);
    for (i, action) in actions.iter().enumerate() {
        r.apply(action)
            .map_err(|e| format!("action {i} ({action:?}): {e}"))?;
    }
    Ok(r.checked)
}

/// Deletes actions from a failing history while it still fails.
fn shrink(mut actions: Vec<Action>, fails: impl Fn(&[Action]) -> bool) -> Vec<Action> {
    let mut chunk = actions.len() / 2;
    while chunk > 0 {
        let mut start = 0;
        while start < actions.len() {
            let mut candidate = actions.clone();
            candidate.drain(start..(start + chunk).min(candidate.len()));
            if fails(&candidate) {
                actions = candidate;
            } else {
                start += chunk;
            }
        }
        chunk /= 2;
    }
    actions
}

/// Checks [`seeds`] histories on graphs with `tile_size`-texel tiles.
fn check_histories(tile_size: u32) {
    let mut checked = 0;
    for seed in 0..seeds() {
        let actions = history(seed, STEPS);
        match replay(tile_size, &actions) {
            Ok(n) => checked += n,
            Err(error) => {
                let minimal = shrink(actions, |a| replay(tile_size, a).is_err());
                panic!(
                    "seed {seed}, tiles of {tile_size}: {error}\n\
                     reduced: {:?}\n{minimal:#?}",
                    replay(tile_size, &minimal)
                );
            }
        }
    }
    // Failures and budgets must not crowd out the comparisons.
    let total = usize::try_from(seeds()).unwrap() * STEPS;
    assert!(
        checked > total / 2,
        "only {checked} of {total} settles compared"
    );
}

// Tiles of 4 and 8 divide the even resolutions; tiles of 5 divide none, so
// every grid has partial tiles.

#[test]
fn edit_histories_match_fresh_graphs_in_tiles_of_4() {
    check_histories(4);
}

#[test]
fn edit_histories_match_fresh_graphs_in_tiles_of_5() {
    check_histories(5);
}

#[test]
fn edit_histories_match_fresh_graphs_in_tiles_of_8() {
    check_histories(8);
}
