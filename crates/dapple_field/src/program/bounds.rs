// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Static bounds of scalar nodes: a value range and a slope bound that hold
//! at every point and every footprint.

use alloc::vec::Vec;

use super::{FieldProgram, NodeId, Op};
use crate::cellular::CellOutput;
use crate::fractal::FractalKind;
use crate::noise::Basis;
use crate::types::PortType;

/// Bounds of a scalar or mask node that hold at every point and footprint.
///
/// `range` bounds the node's values; `slope` bounds `|∂f/∂x| + |∂f/∂y|`, the
/// row sum a warp's footprint scaling reads, or `|∂f/∂x| + |∂f/∂y| + |∂f/∂z|`
/// for a solid node. Either is `None` where no sound
/// bound is known: a discontinuous field (cell values, identifiers, a hard
/// disk) has no slope bound, and vector-valued nodes have no bounds at all.
/// Bounds are conservative, never tight: they may exceed what the node
/// reaches, but the node never exceeds them.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct StaticBounds {
    /// Every value lies in `[range[0], range[1]]`.
    pub range: Option<[f32; 2]>,
    /// `|∂f/∂x| + |∂f/∂y|` never exceeds this, wherever the gradient exists.
    pub slope: Option<f32>,
}

/// Provable bound on `|n|` for one octave of [`Basis::Gradient`] noise.
///
/// At most 9 corners lie within the kernel's 1.5-cell radius of any point
/// (their offsets span under 3 cells per axis), and one corner's term is at
/// most `max_r (1 − r²/2.25)⁴ (r + 0.7) < 0.8504`, so `|n| ≤ 9 · 0.8504 /
/// 2.07 < 3.7`. The documented `[-1, 1]` is measured, not proven.
const GRADIENT_NOISE_BOUND: f64 = 3.7;

/// Provable bound on one axis of one octave's gradient, per lattice cell.
///
/// One corner's term contributes `|d_x|·|K'(r)|·|term| + |g_x|·K(r)` to the
/// x derivative, with `|d_x| ≤ r`, `|term| ≤ r + 0.7`, `|K'| = 8t³r/2.25`:
/// at most `max_r r(r + 0.7)·8t³/2.25 + t⁴ < 2.149`, over at most 9 corners,
/// scaled by `1 / 2.07`: `9 · 2.149 / 2.07 < 9.35`.
const GRADIENT_NOISE_AXIS_SLOPE: f64 = 9.35;

/// [`Basis::Value`] noise per axis, per cell: the fade's derivative peaks at
/// `15/8`, times the largest value difference, 2.
const VALUE_NOISE_AXIS_SLOPE: f64 = 3.75;

/// Cellular distances stay below √3.25 < 2 cells; see [`crate::Cellular`].
const CELL_DISTANCE_BOUND: f64 = 2.0;

/// Provable bound on `|n|` for one octave of solid [`Basis::Gradient`] noise:
/// at most 27 corners lie within 1.5 cells of any point, each term is at most
/// 0.8504 as in the plane, and the sum is scaled by `1 / 2.49`:
/// `27 · 0.8504 / 2.49 < 9.3`.
const GRADIENT_NOISE3_BOUND: f64 = 9.3;

/// Per-axis, per-cell slope bound of one solid gradient-noise octave:
/// `27 · 2.149 / 2.49 < 23.4`, by the planar argument over 27 corners.
const GRADIENT_NOISE3_AXIS_SLOPE: f64 = 23.4;

/// Solid cellular distances stay below √4.25 < 2.1 cells; see [`crate::Cellular3`].
const CELL3_DISTANCE_BOUND: f64 = 2.1;

/// Relative widening absorbing `f32` rounding, per node.
const ROUNDING: f64 = 1e-5;

#[derive(Copy, Clone, Debug)]
struct Bounds {
    range: Option<[f64; 2]>,
    slope: Option<f64>,
}

const UNKNOWN: Bounds = Bounds {
    range: None,
    slope: None,
};

impl FieldProgram {
    /// Static [`StaticBounds`] of node `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` is not a node of this program.
    #[must_use]
    pub fn node_bounds(&self, id: NodeId) -> StaticBounds {
        let end = id.0 as usize;
        assert!(end < self.nodes.len(), "node {end} is not in the program");
        let mut all: Vec<Bounds> = Vec::with_capacity(end + 1);
        for node in &self.nodes[..=end] {
            let bounds = if matches!(node.port, PortType::Scalar | PortType::Mask) {
                node_bounds(&node.op, &all)
            } else {
                UNKNOWN
            };
            all.push(widen(bounds));
        }
        let b = all[end];
        StaticBounds {
            range: b.range.map(|[lo, hi]| [down(lo), up(hi)]),
            slope: b.slope.map(up),
        }
        .finite()
    }

    /// Static bounds of the program's output.
    #[must_use]
    pub fn bounds(&self) -> StaticBounds {
        self.node_bounds(self.output)
    }
}

impl StaticBounds {
    fn finite(self) -> Self {
        Self {
            range: self
                .range
                .filter(|[lo, hi]| lo.is_finite() && hi.is_finite() && lo <= hi),
            slope: self.slope.filter(|s| s.is_finite() && *s >= 0.0),
        }
    }

    /// The largest `|v|` over the range.
    #[must_use]
    pub fn max_abs(&self) -> Option<f32> {
        self.range.map(|[lo, hi]| lo.abs().max(hi.abs()))
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "rounds toward -inf after the f64 widening"
)]
fn down(v: f64) -> f32 {
    let f = v as f32;
    if f64::from(f) > v { f.next_down() } else { f }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "rounds toward +inf after the f64 widening"
)]
fn up(v: f64) -> f32 {
    let f = v as f32;
    if f64::from(f) < v { f.next_up() } else { f }
}

fn widen(b: Bounds) -> Bounds {
    Bounds {
        range: b.range.map(|[lo, hi]| {
            let pad = (lo.abs().max(hi.abs()) + 1e-30) * ROUNDING;
            [lo - pad, hi + pad]
        }),
        slope: b.slope.map(|s| s * (1.0 + ROUNDING) + 1e-30),
    }
}

fn max_abs(r: [f64; 2]) -> f64 {
    r[0].abs().max(r[1].abs())
}

fn add(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[1], a[1] - b[0]]
}

fn mul(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    let p = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]];
    [
        p.iter().copied().fold(f64::INFINITY, f64::min),
        p.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    ]
}

fn clamp_range(r: [f64; 2], min: f64, max: f64) -> [f64; 2] {
    [r[0].clamp(min, max), r[1].clamp(min, max)]
}

/// Largest absolute row sum of a 2 × 2 matrix given column-major.
fn max_row_sum(m: glam::Mat2) -> f64 {
    let [a, b, c, d] = m.to_cols_array().map(f64::from);
    // Columns (a, b) and (c, d): row 0 is (a, c), row 1 is (b, d).
    (a.abs() + c.abs()).max(b.abs() + d.abs())
}

fn noise_bounds(basis: Basis, frequency: [f64; 2]) -> (f64, f64) {
    let (bound, axis) = match basis {
        Basis::Value => (1.0, VALUE_NOISE_AXIS_SLOPE),
        Basis::Gradient => (GRADIENT_NOISE_BOUND, GRADIENT_NOISE_AXIS_SLOPE),
    };
    (bound, axis * (frequency[0] + frequency[1]))
}

fn noise3_bounds(basis: Basis, frequency: [f64; 3]) -> (f64, f64) {
    let (bound, axis) = match basis {
        Basis::Value => (1.0, VALUE_NOISE_AXIS_SLOPE),
        Basis::Gradient => (GRADIENT_NOISE3_BOUND, GRADIENT_NOISE3_AXIS_SLOPE),
    };
    (bound, axis * (frequency[0] + frequency[1] + frequency[2]))
}

/// Largest absolute row sum of a 3 × 3 matrix given column-major.
fn max_row_sum3(m: glam::Mat3) -> f64 {
    let c = m.to_cols_array().map(f64::from);
    (0..3)
        .map(|row| c[row].abs() + c[3 + row].abs() + c[6 + row].abs())
        .fold(0.0, f64::max)
}

/// Fractal bounds from one octave's `(bound, slope)` at a frequency scale.
fn fractal_bounds(
    params: crate::fractal::FractalParams,
    bound: f64,
    octave_slope: impl Fn(f64) -> f64,
) -> Bounds {
    let lacunarity = f64::from(params.lacunarity);
    let gain = f64::from(params.gain);
    let (mut amplitude, mut scale) = (1.0, 1.0);
    let (mut total, mut slope) = (0.0, 0.0);
    for _ in 0..params.octaves {
        total += amplitude;
        slope += amplitude * octave_slope(scale);
        amplitude *= gain;
        scale *= lacunarity;
    }
    let slope = slope / total;
    match params.kind {
        FractalKind::Fbm => Bounds {
            range: Some([-bound, bound]),
            slope: Some(slope),
        },
        FractalKind::Ridged => Bounds {
            range: Some([0.0, (1.0 + bound) * (1.0 + bound)]),
            slope: Some(slope * 2.0 * (bound - 1.0).max(1.0)),
        },
    }
}

fn node_bounds(op: &Op, all: &[Bounds]) -> Bounds {
    let at = |id: NodeId| all[id.0 as usize];
    match *op {
        Op::Constant { value, .. } => Bounds {
            range: Some([f64::from(value); 2]),
            slope: Some(0.0),
        },
        // Band limiting scales a single octave toward 0, inside the range,
        // and its slope by a weight in [0, 1].
        Op::Noise {
            basis, frequency, ..
        } => {
            let (bound, slope) = noise_bounds(basis, frequency.map(f64::from));
            Bounds {
                range: Some([-bound, bound]),
                slope: Some(slope),
            }
        }
        Op::Fractal {
            basis,
            frequency,
            params,
            ..
        } => {
            // The sum is normalized by the total amplitude, so the range is
            // one octave's; the slope is the amplitude-weighted average of
            // the octaves' slopes. Faded octaves move toward a constant mean.
            let lacunarity = f64::from(params.lacunarity);
            let gain = f64::from(params.gain);
            let (bound, _) = noise_bounds(basis, [1.0, 1.0]);
            let (mut amplitude, mut scale) = (1.0, 1.0);
            let (mut total, mut slope) = (0.0, 0.0);
            for _ in 0..params.octaves {
                let (_, octave) = noise_bounds(
                    basis,
                    [
                        f64::from(frequency[0]) * scale,
                        f64::from(frequency[1]) * scale,
                    ],
                );
                total += amplitude;
                slope += amplitude * octave;
                amplitude *= gain;
                scale *= lacunarity;
            }
            let slope = slope / total;
            match params.kind {
                FractalKind::Fbm => Bounds {
                    range: Some([-bound, bound]),
                    slope: Some(slope),
                },
                // (1 − |n|)², with |1 − |n|| ≤ max(1, bound − 1).
                FractalKind::Ridged => Bounds {
                    range: Some([0.0, (1.0 + bound) * (1.0 + bound)]),
                    slope: Some(slope * 2.0 * (bound - 1.0).max(1.0)),
                },
            }
        }
        Op::Cellular {
            frequency, output, ..
        } => {
            let cells = f64::from(frequency[0]) + f64::from(frequency[1]);
            // Distances move at unit speed per cell; the mean each output
            // fades toward lies inside its range.
            let (range, slope) = match output {
                CellOutput::F1 | CellOutput::F2 | CellOutput::Border => {
                    ([0.0, CELL_DISTANCE_BOUND], Some(cells))
                }
                CellOutput::F2MinusF1 => ([0.0, CELL_DISTANCE_BOUND], Some(2.0 * cells)),
                CellOutput::CellValue => ([0.0, 1.0], None),
            };
            Bounds {
                range: Some(range),
                slope,
            }
        }
        // A smoothstep over a band at least `softness` wide: slope at most
        // 1.5 / softness along the radius, √2 times that as a row sum.
        Op::Disk { softness, .. } => Bounds {
            range: Some([0.0, 1.0]),
            slope: (softness > 0.0).then(|| core::f64::consts::SQRT_2 * 1.5 / f64::from(softness)),
        },
        Op::Sample { ref image } => {
            let b = image.static_bounds();
            Bounds {
                range: b.range.map(|r| r.map(f64::from)),
                slope: b.slope.map(f64::from),
            }
        }
        // ∇(input ∘ T) = Mᵀ ∇input; the row-sum norm of Mᵀv is at most M's
        // largest row sum times v's.
        Op::Transform { input, transform } => {
            let b = at(input);
            Bounds {
                range: b.range,
                slope: b.slope.map(|s| s * max_row_sum(transform.matrix)),
            }
        }
        Op::Demote { input } => at(input),
        // ∇(input ∘ q) = (I + aJ)ᵀ ∇input, whose row-sum norm is at most
        // 1 + |a| · max(slope(dx), slope(dy)) times the input's.
        Op::Warp {
            input,
            dx,
            dy,
            amount,
        } => {
            let b = at(input);
            let stretch = at(dx)
                .slope
                .zip(at(dy).slope)
                .map(|(x, y)| 1.0 + f64::from(amount).abs() * x.max(y));
            Bounds {
                range: b.range,
                slope: b.slope.zip(stretch).map(|(s, k)| s * k),
            }
        }
        Op::Add { a, b } | Op::Sub { a, b } => {
            let (x, y) = (at(a), at(b));
            let range = x.range.zip(y.range).map(|(x, y)| {
                if matches!(op, Op::Add { .. }) {
                    add(x, y)
                } else {
                    sub(x, y)
                }
            });
            Bounds {
                range,
                slope: x.slope.zip(y.slope).map(|(x, y)| x + y),
            }
        }
        // ∇(ab) = a∇b + b∇a.
        Op::Mul { a, b } => {
            let (x, y) = (at(a), at(b));
            let slope = match (x.range, x.slope, y.range, y.slope) {
                (Some(xr), Some(xs), Some(yr), Some(ys)) => {
                    Some(max_abs(xr) * ys + max_abs(yr) * xs)
                }
                _ => None,
            };
            Bounds {
                range: x.range.zip(y.range).map(|(x, y)| mul(x, y)),
                slope,
            }
        }
        // Min and max are as steep as the steeper operand.
        Op::Min { a, b } | Op::Max { a, b } => {
            let (x, y) = (at(a), at(b));
            let range = x.range.zip(y.range).map(|(x, y)| {
                if matches!(op, Op::Min { .. }) {
                    [x[0].min(y[0]), x[1].min(y[1])]
                } else {
                    [x[0].max(y[0]), x[1].max(y[1])]
                }
            });
            Bounds {
                range,
                slope: x.slope.zip(y.slope).map(|(x, y)| x.max(y)),
            }
        }
        Op::Abs { input } => {
            let b = at(input);
            Bounds {
                range: b.range.map(|[lo, hi]| {
                    let top = lo.abs().max(hi.abs());
                    if lo <= 0.0 && hi >= 0.0 {
                        [0.0, top]
                    } else {
                        [lo.abs().min(hi.abs()), top]
                    }
                }),
                slope: b.slope,
            }
        }
        Op::Clamp { input, min, max } => {
            let b = at(input);
            Bounds {
                range: Some(b.range.map_or([f64::from(min), f64::from(max)], |r| {
                    clamp_range(r, f64::from(min), f64::from(max))
                })),
                slope: b.slope,
            }
        }
        Op::AsMask { input } => {
            let b = at(input);
            Bounds {
                range: Some(b.range.map_or([0.0, 1.0], |r| clamp_range(r, 0.0, 1.0))),
                slope: b.slope,
            }
        }
        Op::Remap { input, from, to } => {
            let b = at(input);
            let k =
                (f64::from(to[1]) - f64::from(to[0])) / (f64::from(from[1]) - f64::from(from[0]));
            let map = |v: f64| f64::from(to[0]) + (v - f64::from(from[0])) * k;
            Bounds {
                range: b.range.map(|[lo, hi]| {
                    let (x, y) = (map(lo), map(hi));
                    [x.min(y), x.max(y)]
                }),
                slope: b.slope.map(|s| s * k.abs()),
            }
        }
        // a + (b − a)t, with ∇ = (1 − t)∇a + t∇b + (b − a)∇t.
        Op::Mix { a, b, t } => {
            let (x, y, w) = (at(a), at(b), at(t));
            let range = match (x.range, y.range, w.range) {
                (Some(xr), Some(yr), Some(wr)) => Some(add(xr, mul(sub(yr, xr), wr))),
                _ => None,
            };
            let slope = match (x.slope, y.slope, w.slope, x.range, y.range, w.range) {
                (Some(xs), Some(ys), Some(ws), Some(xr), Some(yr), Some(wr)) => Some(
                    max_abs(sub([1.0, 1.0], wr)) * xs
                        + max_abs(wr) * ys
                        + max_abs(sub(yr, xr)) * ws,
                ),
                _ => None,
            };
            Bounds { range, slope }
        }
        // An angle in [0, π) and an agreement in [0, 1] jump where their
        // direction vanishes, so neither has a slope bound.
        Op::Angle { .. } => Bounds {
            range: Some([0.0, core::f64::consts::PI]),
            slope: None,
        },
        Op::Coherence { .. } => Bounds {
            range: Some([0.0, 1.0]),
            slope: None,
        },
        Op::Constant3 { value, .. } => Bounds {
            range: Some([f64::from(value); 2]),
            slope: Some(0.0),
        },
        Op::Noise3 {
            basis, frequency, ..
        } => {
            let (bound, slope) = noise3_bounds(basis, frequency.map(f64::from));
            Bounds {
                range: Some([-bound, bound]),
                slope: Some(slope),
            }
        }
        // As for planar fractals: one octave's range, the amplitude-weighted
        // average of the octaves' slopes.
        Op::Fractal3 {
            basis,
            frequency,
            params,
            ..
        } => {
            let (bound, _) = noise3_bounds(basis, [1.0; 3]);
            fractal_bounds(params, bound, |scale| {
                noise3_bounds(basis, frequency.map(|f| f64::from(f) * scale)).1
            })
        }
        Op::Cellular3 {
            frequency, output, ..
        } => {
            let cells: f64 = frequency.iter().map(|f| f64::from(*f)).sum();
            let (range, slope) = match output {
                CellOutput::F1 | CellOutput::F2 | CellOutput::Border => {
                    ([0.0, CELL3_DISTANCE_BOUND], Some(cells))
                }
                CellOutput::F2MinusF1 => ([0.0, CELL3_DISTANCE_BOUND], Some(2.0 * cells)),
                CellOutput::CellValue => ([0.0, 1.0], None),
            };
            Bounds {
                range: Some(range),
                slope,
            }
        }
        Op::Transform3 { input, transform } => {
            let b = at(input);
            Bounds {
                range: b.range,
                slope: b.slope.map(|s| s * max_row_sum3(transform.matrix)),
            }
        }
        // The planar gradient is (u·∇, v·∇), and |u·g| ≤ max|u_i| · Σ|g_i|.
        Op::Slice { input, u, v, .. } => {
            let b = at(input);
            let norm = |w: [f32; 3]| w.iter().map(|c| f64::from(c.abs())).fold(0.0, f64::max);
            Bounds {
                range: b.range,
                slope: b.slope.map(|s| s * (norm(u) + norm(v))),
            }
        }
        // A sawtooth jumps, so it has no slope bound.
        Op::Fract { .. } => Bounds {
            range: Some([0.0, 1.0]),
            slope: None,
        },
        // The angle jumps across the negative x half-axis.
        Op::Atan2 { .. } => Bounds {
            range: Some([-core::f64::consts::PI, core::f64::consts::PI]),
            slope: None,
        },
        Op::Length { .. } => Bounds {
            range: None,
            slope: None,
        },
        // Vector-valued nodes and identifiers: no scalar bounds.
        Op::Position3
        | Op::Vector2 { .. }
        | Op::Vector3 { .. }
        | Op::Color { .. }
        | Op::Component { .. }
        | Op::ToId { .. }
        | Op::Normalize { .. }
        | Op::Direction { .. }
        | Op::BlendNormals { .. } => UNKNOWN,
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use glam::{Mat2, Vec2};

    use super::StaticBounds;
    use crate::cellular::CellOutput;
    use crate::domain::{Domain, Footprint};
    use crate::field::Affine2;
    use crate::fractal::{FractalKind, FractalParams};
    use crate::image::{ImageLevel, SampleImage};
    use crate::noise::Basis;
    use crate::program::{FieldProgram, Fingerprint, NodeId, Op, ProgramBuilder};

    fn torus() -> Domain {
        Domain::periodic(1, 1).unwrap()
    }

    fn noise(b: &mut ProgramBuilder, basis: Basis, seed: u64) -> NodeId {
        b.add(Op::Noise {
            basis,
            domain: torus(),
            frequency: [3.0, 5.0],
            seed,
        })
        .unwrap()
    }

    /// One program per bounded op kind, with its output node.
    fn programs() -> Vec<FieldProgram> {
        let mut out = Vec::new();
        let mut single = |make: &dyn Fn(&mut ProgramBuilder) -> NodeId| {
            let mut b = ProgramBuilder::new();
            let id = make(&mut b);
            out.push(b.finish(id).unwrap());
        };
        for basis in [Basis::Value, Basis::Gradient] {
            single(&|b| noise(b, basis, 1));
            for kind in [FractalKind::Fbm, FractalKind::Ridged] {
                single(&|b| {
                    b.add(Op::Fractal {
                        basis,
                        domain: torus(),
                        frequency: [2.0, 2.0],
                        seed: 3,
                        params: FractalParams {
                            kind,
                            octaves: 4,
                            lacunarity: 2,
                            gain: 0.5,
                        },
                    })
                    .unwrap()
                });
            }
        }
        for output in [
            CellOutput::F1,
            CellOutput::F2,
            CellOutput::F2MinusF1,
            CellOutput::Border,
        ] {
            single(&|b| {
                b.add(Op::Cellular {
                    domain: torus(),
                    frequency: [4.0, 4.0],
                    jitter: 1.0,
                    seed: 5,
                    output,
                })
                .unwrap()
            });
        }
        single(&|b| {
            b.add(Op::Disk {
                domain: torus(),
                center: [0.5, 0.5],
                radius: 0.2,
                softness: 0.05,
            })
            .unwrap()
        });
        single(&|b| {
            let values: Vec<f32> = (0..64_u16).map(|i| f32::from(i % 7) * 0.3 - 1.0).collect();
            let level = ImageLevel::new(8, 8, Vec2::splat(0.125), values).unwrap();
            let image =
                SampleImage::new(torus(), Vec2::ZERO, alloc::vec![level], Fingerprint(9)).unwrap();
            b.add(Op::Sample { image }).unwrap()
        });
        single(&|b| {
            // Arithmetic, a transform and a warp over two noises.
            let n = noise(b, Basis::Gradient, 11);
            let m = noise(b, Basis::Value, 12);
            let t = b
                .add(Op::Transform {
                    input: n,
                    transform: Affine2 {
                        matrix: Mat2::from_cols(Vec2::new(2.0, 1.0), Vec2::new(-1.0, 3.0)),
                        translation: Vec2::ZERO,
                    },
                })
                .unwrap();
            let w = b
                .add(Op::Warp {
                    input: t,
                    dx: n,
                    dy: m,
                    amount: 0.05,
                })
                .unwrap();
            let product = b.add(Op::Mul { a: w, b: m }).unwrap();
            let remapped = b
                .add(Op::Remap {
                    input: product,
                    from: [-1.0, 1.0],
                    to: [3.0, -2.0],
                })
                .unwrap();
            let clamped = b
                .add(Op::Clamp {
                    input: remapped,
                    min: -1.0,
                    max: 2.0,
                })
                .unwrap();
            let t01 = b.add(Op::AsMask { input: m }).unwrap();
            b.add(Op::Mix {
                a: clamped,
                b: n,
                t: t01,
            })
            .unwrap()
        });
        out
    }

    #[test]
    fn values_and_slopes_stay_within_their_bounds() {
        for (index, program) in programs().iter().enumerate() {
            let StaticBounds { range, slope } = program.bounds();
            let [lo, hi] = range.unwrap_or_else(|| panic!("program {index} has a range"));
            let slope = slope.unwrap_or_else(|| panic!("program {index} has a slope"));
            let output = program.nodes().last().unwrap().0;
            let mut seen_slope = 0.0_f32;
            for footprint in [Footprint::POINT, Footprint::new(0.01).unwrap()] {
                for i in 0..97_u16 {
                    for j in 0..89_u16 {
                        let p = Vec2::new(f32::from(i) / 97.0, f32::from(j) / 89.0);
                        let (v, g) = program.eval_node_gradient(output, p, footprint);
                        assert!(
                            lo <= v && v <= hi,
                            "program {index}: {v} outside [{lo}, {hi}]"
                        );
                        seen_slope = seen_slope.max(g.x.abs() + g.y.abs());
                    }
                }
            }
            assert!(
                seen_slope <= slope,
                "program {index}: slope {seen_slope} above bound {slope}"
            );
        }
    }

    #[test]
    fn jumps_and_vectors_have_no_slope() {
        let mut b = ProgramBuilder::new();
        let cells = b
            .add(Op::Cellular {
                domain: torus(),
                frequency: [4.0, 4.0],
                jitter: 1.0,
                seed: 5,
                output: CellOutput::CellValue,
            })
            .unwrap();
        let hard = b
            .add(Op::Disk {
                domain: torus(),
                center: [0.5, 0.5],
                radius: 0.2,
                softness: 0.0,
            })
            .unwrap();
        let sum = b.add(Op::Add { a: cells, b: hard }).unwrap();
        let program = b.finish(sum).unwrap();
        let [lo, hi] = program.node_bounds(cells).range.unwrap();
        assert!(lo <= 0.0 && (1.0..1.001).contains(&hi));
        let bounds = program.bounds();
        assert!(bounds.range.is_some());
        assert_eq!(bounds.slope, None);
    }
}
