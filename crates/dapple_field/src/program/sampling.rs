// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sampling correctness: what a footprint evaluation promises, and a
//! reference to hold it to.
//!
//! Evaluating a node with a footprint is meant to stand for the node
//! averaged over that footprint. How well it does is a property of the
//! whole expression, not of each op: filtering each input of a nonlinear
//! op does not filter its output (the mean of `f²` is not the square of
//! the mean of `f`). So every node gets a [`Sampling`] guarantee, from its
//! op and its inputs' guarantees:
//!
//! | Guarantee | Meaning | Ops |
//! |---|---|---|
//! | [`Exact`](Sampling::Exact) | exact integration under a box filter of the footprint's width | constants, the solid position, and linear combinations of exact nodes |
//! | [`Attenuated`](Sampling::Attenuated) | components above the footprint's Nyquist limit fade, the rest pass: frequency attenuation, not integration | noise and fractal octaves, image samples through their mips, and linear combinations of them |
//! | [`Heuristic`](Sampling::Heuristic) | detail fades toward a mean by a rule that is not an integral | cellular noise, scatter stamps, the antialiased disk, waves (a box mean along the phase, exact only where the phase is linear), and every continuous nonlinear op of varying inputs (products, minima and maxima, clamps, lengths, warps) |
//! | [`PointOnly`](Sampling::PointOnly) | the footprint is not honored: values are point samples and alias | discontinuous ops of varying inputs (`fract`, quantization to identifiers) and tile layouts |
//!
//! Linear ops (sums, differences, scaling by a constant, affine remaps,
//! mixes by a constant weight, component moves and domain transforms) keep
//! the weakest guarantee of their inputs, because averaging commutes with
//! them. A nonlinear op of inputs that are all constant is exact.
//!
//! [`reference_box`] integrates the **complete expression**: it evaluates
//! the node at point footprints on a stratified grid over the box and
//! averages. [`measure`] compares a footprint evaluation with it over a set
//! of points, so tests hold whole expressions, not single ops, to their
//! guarantees.

use alloc::vec::Vec;

use glam::Vec2;

use super::{FieldProgram, NodeId, Op};
use crate::domain::Footprint;
use crate::field::ScalarField;

/// What a footprint evaluation of a node promises: see the [module
/// docs](self). Ordered from strongest to weakest.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Sampling {
    /// Exact integration under a box filter of the footprint's width.
    Exact,
    /// Frequency attenuation: detail above the footprint's Nyquist limit
    /// fades out.
    Attenuated,
    /// Detail fades toward a mean by a rule that is not an integral.
    Heuristic,
    /// Point evaluation only: the footprint is not honored.
    PointOnly,
}

impl Sampling {
    /// A one-line description.
    #[must_use]
    pub const fn describe(self) -> &'static str {
        match self {
            Self::Exact => "exact integration under a box filter",
            Self::Attenuated => "frequency attenuation",
            Self::Heuristic => "heuristic fading to a mean",
            Self::PointOnly => "point evaluation only",
        }
    }
}

/// How an op treats its inputs.
enum Class {
    /// A leaf with its own guarantee.
    Leaf(Sampling),
    /// Linear in its inputs: the weakest of theirs.
    Linear,
    /// Linear when every input but the first is constant, else continuous
    /// nonlinear (a product, a mix by a varying weight).
    LinearIfOthersConstant,
    /// Continuous and nonlinear.
    Nonlinear,
    /// Discontinuous.
    Discontinuous,
}

fn class(op: &Op) -> Class {
    match op {
        Op::Constant { .. } | Op::Constant3 { .. } | Op::Position3 => Class::Leaf(Sampling::Exact),
        Op::Noise { .. } | Op::Fractal { .. } | Op::Noise3 { .. } | Op::Fractal3 { .. } => {
            Class::Leaf(Sampling::Attenuated)
        }
        Op::Sample { .. } => Class::Leaf(Sampling::Attenuated),
        Op::Cellular { .. } | Op::Cellular3 { .. } | Op::Scatter { .. } | Op::Disk { .. } => {
            Class::Leaf(Sampling::Heuristic)
        }
        Op::Tiling { .. } => Class::Leaf(Sampling::PointOnly),
        Op::Add { .. }
        | Op::Sub { .. }
        | Op::Remap { .. }
        | Op::Vector2 { .. }
        | Op::Vector3 { .. }
        | Op::Color { .. }
        | Op::Component { .. }
        | Op::Transform { .. }
        | Op::Transform3 { .. }
        | Op::Demote { .. }
        | Op::Slice { .. } => Class::Linear,
        Op::Mul { .. } => Class::LinearIfOthersConstant,
        Op::Mix { .. } => Class::LinearIfOthersConstant,
        Op::Min { .. }
        | Op::Max { .. }
        | Op::Abs { .. }
        | Op::Clamp { .. }
        | Op::AsMask { .. }
        | Op::Warp { .. }
        | Op::Normalize { .. }
        | Op::Direction { .. }
        | Op::Angle { .. }
        | Op::Coherence { .. }
        | Op::Length { .. }
        | Op::Atan2 { .. }
        | Op::Wave { .. } => Class::Nonlinear,
        Op::Fract { .. } | Op::ToId { .. } => Class::Discontinuous,
    }
}

impl FieldProgram {
    /// The sampling guarantee of node `id`: see the [module docs](self).
    ///
    /// # Panics
    ///
    /// Panics if `id` is not a node of this program.
    #[must_use]
    pub fn node_sampling(&self, id: NodeId) -> Sampling {
        let end = id.0 as usize;
        assert!(end < self.nodes.len(), "node {end} is not in the program");
        // Per node: its guarantee, and whether it is constant.
        let mut all: Vec<(Sampling, bool)> = Vec::with_capacity(end + 1);
        for node in &self.nodes[..=end] {
            let inputs: Vec<(Sampling, bool)> =
                node.op.inputs().iter().map(|i| all[i.0 as usize]).collect();
            let weakest = inputs.iter().map(|i| i.0).max().unwrap_or(Sampling::Exact);
            let constant = !inputs.is_empty() && inputs.iter().all(|i| i.1);
            let entry = match class(&node.op) {
                Class::Leaf(s) => (
                    s,
                    matches!(node.op, Op::Constant { .. } | Op::Constant3 { .. }),
                ),
                _ if constant => (Sampling::Exact, true),
                Class::Linear => (weakest, false),
                Class::LinearIfOthersConstant => {
                    // `Mul { a, b }` is linear when either side is constant;
                    // `Mix { a, b, t }` when its weight is.
                    let linear = match node.op {
                        Op::Mul { .. } => inputs.iter().any(|i| i.1),
                        _ => inputs.last().is_some_and(|t| t.1),
                    };
                    if linear {
                        (weakest, false)
                    } else {
                        (weakest.max(Sampling::Heuristic), false)
                    }
                }
                Class::Nonlinear => (weakest.max(Sampling::Heuristic), false),
                Class::Discontinuous => (Sampling::PointOnly, false),
            };
            all.push(entry);
        }
        all[end].0
    }

    /// The sampling guarantee of the program's output.
    #[must_use]
    pub fn sampling(&self) -> Sampling {
        self.node_sampling(self.output)
    }
}

/// The complete expression `field` integrated over the box `width` wide
/// around `p`: the mean of point evaluations at the centers of an `n` × `n`
/// grid of subcells. This is the reference a footprint evaluation stands
/// for.
#[must_use]
pub fn reference_box(field: &impl ScalarField, p: Vec2, width: f32, n: u32) -> f32 {
    let n = n.max(1);
    #[expect(clippy::cast_precision_loss, reason = "sample counts are small")]
    let step = width / n as f32;
    let mut sum = 0.0_f64;
    for j in 0..n {
        for i in 0..n {
            #[expect(clippy::cast_precision_loss, reason = "sample counts are small")]
            let q = p + Vec2::new(
                (i as f32 + 0.5) * step - 0.5 * width,
                (j as f32 + 0.5) * step - 0.5 * width,
            );
            sum += f64::from(field.eval(q, Footprint::POINT));
        }
    }
    #[expect(clippy::cast_possible_truncation, reason = "a mean of f32 values")]
    let mean = (sum / f64::from(n * n)) as f32;
    mean
}

/// How a footprint evaluation compares with [`reference_box`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SamplingError {
    /// Root-mean-square error.
    pub rms: f32,
    /// Largest absolute error.
    pub max: f32,
    /// Mean signed error: the evaluation's bias.
    pub bias: f32,
    /// Root-mean-square deviation of the reference itself over the points,
    /// the scale the errors compare with.
    pub spread: f32,
}

/// `field` evaluated with `footprint` at each of `points`, against its
/// complete expression integrated over the footprint's box with `n` × `n`
/// point samples.
#[must_use]
pub fn measure(
    field: &impl ScalarField,
    points: &[Vec2],
    footprint: Footprint,
    n: u32,
) -> SamplingError {
    let (mut sq, mut max, mut bias) = (0.0_f64, 0.0_f32, 0.0_f64);
    let mut references = Vec::with_capacity(points.len());
    for &p in points {
        let reference = reference_box(field, p, footprint.width(), n);
        let e = field.eval(p, footprint) - reference;
        sq += f64::from(e * e);
        bias += f64::from(e);
        max = max.max(e.abs());
        references.push(f64::from(reference));
    }
    #[expect(clippy::cast_precision_loss, reason = "point counts are small")]
    let count = points.len().max(1) as f64;
    let mean = references.iter().sum::<f64>() / count;
    let spread = references
        .iter()
        .map(|r| (r - mean) * (r - mean))
        .sum::<f64>()
        / count;
    #[expect(clippy::cast_possible_truncation, reason = "summaries of f32 errors")]
    SamplingError {
        rms: libm::sqrt(sq / count) as f32,
        max,
        bias: (bias / count) as f32,
        spread: libm::sqrt(spread) as f32,
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::program::{FieldProgram, ProgramBuilder};
    use crate::{Basis, Domain};

    fn noise(b: &mut ProgramBuilder, frequency: f32) -> NodeId {
        b.add(Op::Noise {
            basis: Basis::Gradient,
            domain: Domain::Plane,
            frequency: [frequency; 2],
            seed: 5,
        })
        .unwrap()
    }

    fn constant(b: &mut ProgramBuilder, value: f32) -> NodeId {
        b.add(Op::Constant {
            domain: Domain::Plane,
            value,
        })
        .unwrap()
    }

    fn points() -> Vec<Vec2> {
        (0..64)
            .map(|i| Vec2::new(i as f32 * 0.137 % 3.0, i as f32 * 0.291 % 3.0))
            .collect()
    }

    #[test]
    fn guarantees_follow_the_whole_expression() {
        let mut b = ProgramBuilder::new();
        let n = noise(&mut b, 8.0);
        let two = constant(&mut b, 2.0);
        let one = constant(&mut b, 1.0);
        let scaled = b.add(Op::Mul { a: n, b: two }).unwrap();
        let linear = b.add(Op::Add { a: scaled, b: one }).unwrap();
        let squared = b.add(Op::Mul { a: n, b: n }).unwrap();
        let fract = b.add(Op::Fract { input: scaled }).unwrap();
        let constant_product = b.add(Op::Mul { a: two, b: one }).unwrap();
        let p = b.finish(linear).unwrap();
        assert_eq!(p.node_sampling(one), Sampling::Exact);
        assert_eq!(p.node_sampling(constant_product), Sampling::Exact);
        assert_eq!(p.node_sampling(n), Sampling::Attenuated);
        assert_eq!(p.sampling(), Sampling::Attenuated, "linear keeps it");
        assert_eq!(p.node_sampling(squared), Sampling::Heuristic);
        assert_eq!(p.node_sampling(fract), Sampling::PointOnly);
    }

    #[test]
    fn reference_integration_holds_whole_expressions_to_their_guarantees() {
        let wide = Footprint::new(0.5).unwrap(); // four noise cells across
        // Exact: a constant integrates to itself.
        let mut b = ProgramBuilder::new();
        let c = constant(&mut b, 0.25);
        let exact = measure(&b.finish(c).unwrap(), &points(), wide, 16);
        assert!(exact.max < 1e-6, "{exact:?}");

        // Attenuated and linear: 2n + 1 errs exactly twice as much as n.
        let mut b = ProgramBuilder::new();
        let n = noise(&mut b, 8.0);
        let alone = b.clone().finish(n).unwrap();
        let two = constant(&mut b, 2.0);
        let one = constant(&mut b, 1.0);
        let scaled = b.add(Op::Mul { a: n, b: two }).unwrap();
        let linear = b.add(Op::Add { a: scaled, b: one }).unwrap();
        let (e1, e2) = (
            measure(&alone, &points(), wide, 16),
            measure(&b.finish(linear).unwrap(), &points(), wide, 16),
        );
        assert!((e2.rms - 2.0 * e1.rms).abs() < 1e-3, "{e1:?} {e2:?}");

        // Nonlinear: the mean of n² is not the square of the mean of n. The
        // footprint evaluation fades n and squares what is left, so it
        // loses the energy the reference keeps: a clear negative bias.
        let mut b = ProgramBuilder::new();
        let n = noise(&mut b, 8.0);
        let squared = b.add(Op::Mul { a: n, b: n }).unwrap();
        let program = b.finish(squared).unwrap();
        assert_eq!(program.sampling(), Sampling::Heuristic);
        let e = measure(&program, &points(), wide, 24);
        assert!(e.bias < -0.02, "average(f²) ≠ average(f)²: {e:?}");
        // A point footprint is exact for any expression.
        let point = measure(&program, &points(), Footprint::POINT, 1);
        assert!(point.max < 1e-6, "{point:?}");
    }

    /// Rings ending in a sharp edge, 11 a unit along `(3, 1)`, as a ramp of
    /// a `fract` and as a wave: planar, through a slice of the solid
    /// position.
    fn rings(wave: bool) -> FieldProgram {
        let mut b = ProgramBuilder::new();
        let position = b.add(Op::Position3).unwrap();
        let [x, y] = [0, 1].map(|index| {
            b.add(Op::Component {
                input: position,
                index,
            })
            .unwrap()
        });
        let x = b
            .add(Op::Remap {
                input: x,
                from: [0.0, 1.0],
                to: [0.0, 33.0],
            })
            .unwrap();
        let y = b
            .add(Op::Remap {
                input: y,
                from: [0.0, 1.0],
                to: [0.0, 11.0],
            })
            .unwrap();
        let phase = b.add(Op::Add { a: x, b: y }).unwrap();
        let ring = if wave {
            let zero = b
                .add(Op::Constant3 {
                    domain: crate::Domain3::Space,
                    value: 0.0,
                })
                .unwrap();
            b.add(Op::Wave {
                input: phase,
                shift: zero,
                shape: crate::WaveShape::Sawtooth,
                from: [0.4, 0.95],
                to: [0.0, 1.0],
            })
            .unwrap()
        } else {
            let t = b.add(Op::Fract { input: phase }).unwrap();
            let t = b
                .add(Op::Remap {
                    input: t,
                    from: [0.4, 0.95],
                    to: [0.0, 1.0],
                })
                .unwrap();
            b.add(Op::Clamp {
                input: t,
                min: 0.0,
                max: 1.0,
            })
            .unwrap()
        };
        let planar = b
            .add(Op::Slice {
                input: ring,
                origin: [0.0; 3],
                u: [1.0, 0.0, 0.0],
                v: [0.0, 1.0, 0.0],
                domain: Domain::Plane,
            })
            .unwrap();
        b.finish(planar).unwrap()
    }

    #[test]
    fn waves_fade_rings_to_their_mean_where_fract_aliases() {
        let (wave, fract) = (rings(true), rings(false));
        assert_eq!(wave.sampling(), Sampling::Heuristic);
        assert_eq!(fract.sampling(), Sampling::PointOnly);
        // From a fifth of a ring to many rings per footprint (the phase
        // runs √(33² + 11²) ≈ 34.8 rings a unit).
        for width in [0.006, 0.015, 0.03, 0.06, 0.2] {
            let footprint = Footprint::new(width).unwrap();
            let w = measure(&wave, &points(), footprint, 48);
            let f = measure(&fract, &points(), footprint, 48);
            assert!(w.rms < 0.03, "wave at {width}: {w:?}");
            assert!(w.bias.abs() < 0.01, "wave at {width}: {w:?}");
            assert!(
                w.rms * 4.0 < f.rms,
                "wave {w:?} against fract {f:?} at {width}"
            );
        }
        // A point footprint is the ramp itself.
        let point = measure(&wave, &points(), Footprint::POINT, 1);
        assert!(point.max < 1e-6, "{point:?}");
    }
}
