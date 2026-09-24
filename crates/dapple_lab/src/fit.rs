// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Authoring by measurement: fitting parameters to target measurements.
//!
//! Instead of turning twenty knobs, state what the result should measure
//! (about 8% exposed substrate, a cream within ΔE 3 of a reference, a
//! roughness distribution) and search the parameters for it. Nothing here
//! knows about materials:
//!
//! - a [`Space`] of named parameters, each a closed range, searched
//!   linearly or logarithmically;
//! - [`Target`]s: named measurements with a value and a tolerance, to equal,
//!   reach or stay under;
//! - an objective, any function from parameters to one measurement per
//!   target (build a material and measure it, grow a tree and measure it);
//! - [`fit`], which minimizes the [`loss`] with CMA-ES (Hansen's covariance
//!   matrix adaptation evolution strategy): derivative-free, since the
//!   gradients available (spatial ones) are not gradients with respect to
//!   authoring parameters, and deterministic under its seed, since every
//!   random draw is a keyed hash of the seed, generation and sample.
//!
//! [`Fit::report`] writes the result in the lab's report format: fitted
//! parameters with their ranges, each target's measurement with its
//! tolerance band, the loss and its history.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use dapple_field::hash::{hash, unit_f64};

use crate::report::Report;

/// One searched parameter.
#[derive(Clone, Debug, PartialEq)]
pub struct Dimension {
    /// Its name.
    pub name: String,
    /// The smallest value searched.
    pub lo: f64,
    /// The largest value searched.
    pub hi: f64,
    /// Search the range logarithmically (for scales and lengths spanning
    /// orders of magnitude); both ends must then be positive.
    pub log: bool,
}

impl Dimension {
    /// A linear range.
    #[must_use]
    pub fn linear(name: &str, lo: f64, hi: f64) -> Self {
        Self {
            name: name.into(),
            lo,
            hi,
            log: false,
        }
    }

    /// A logarithmic range.
    #[must_use]
    pub fn log(name: &str, lo: f64, hi: f64) -> Self {
        Self {
            name: name.into(),
            lo,
            hi,
            log: true,
        }
    }

    /// The value at `t` in `[0, 1]` of the range.
    #[must_use]
    pub fn at(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        if self.log {
            libm::exp(libm::log(self.lo) + (libm::log(self.hi) - libm::log(self.lo)) * t)
        } else {
            self.lo + (self.hi - self.lo) * t
        }
    }

    /// Where `v` lies in the range, in `[0, 1]`.
    #[must_use]
    pub fn position(&self, v: f64) -> f64 {
        let t = if self.log {
            (libm::log(v) - libm::log(self.lo)) / (libm::log(self.hi) - libm::log(self.lo))
        } else {
            (v - self.lo) / (self.hi - self.lo)
        };
        t.clamp(0.0, 1.0)
    }
}

/// The searched parameters.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Space {
    /// The dimensions, in the order values are passed.
    pub dimensions: Vec<Dimension>,
}

/// What a target asks of its measurement.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Goal {
    /// Equal the value.
    Equal,
    /// Be at least the value.
    AtLeast,
    /// Be at most the value.
    AtMost,
}

/// A target measurement.
#[derive(Clone, Debug, PartialEq)]
pub struct Target {
    /// Its name.
    pub name: String,
    /// The value.
    pub value: f64,
    /// How far off counts as one unit of loss (squared).
    pub tolerance: f64,
    /// What is asked.
    pub goal: Goal,
}

impl Target {
    /// A target to equal `value` within `tolerance`.
    #[must_use]
    pub fn equal(name: &str, value: f64, tolerance: f64) -> Self {
        Self {
            name: name.into(),
            value,
            tolerance,
            goal: Goal::Equal,
        }
    }

    /// A target to reach at least `value`, with `tolerance` for shortfall.
    #[must_use]
    pub fn at_least(name: &str, value: f64, tolerance: f64) -> Self {
        Self {
            name: name.into(),
            value,
            tolerance,
            goal: Goal::AtLeast,
        }
    }

    /// A target to stay at most `value`, with `tolerance` for excess.
    #[must_use]
    pub fn at_most(name: &str, value: f64, tolerance: f64) -> Self {
        Self {
            name: name.into(),
            value,
            tolerance,
            goal: Goal::AtMost,
        }
    }

    /// This target's miss in tolerances: 0 when met.
    #[must_use]
    pub fn miss(&self, measured: f64) -> f64 {
        if !measured.is_finite() {
            return 1e3;
        }
        let off = match self.goal {
            Goal::Equal => measured - self.value,
            Goal::AtLeast => (self.value - measured).max(0.0),
            Goal::AtMost => (measured - self.value).max(0.0),
        };
        off / self.tolerance
    }

    /// Whether `measured` is within one tolerance of the goal.
    #[must_use]
    pub fn met(&self, measured: f64) -> bool {
        self.miss(measured).abs() <= 1.0
    }
}

/// The loss of `measurements` against `targets`: the sum of squared misses
/// in tolerances. Missing measurements count as far off.
#[must_use]
pub fn loss(targets: &[Target], measurements: &[f64]) -> f64 {
    targets
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let m = t.miss(measurements.get(i).copied().unwrap_or(f64::NAN));
            m * m
        })
        .sum()
}

/// Settings for [`fit`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Cmaes {
    /// The initial step size, in units of each dimension's range.
    pub sigma: f64,
    /// Samples per generation; `None` for `4 + ⌊3 ln n⌋`.
    pub population: Option<usize>,
    /// Stop after this many evaluations.
    pub max_evaluations: usize,
    /// Stop once the best loss is at most this.
    pub target_loss: f64,
    /// Keys every random draw.
    pub seed: u64,
}

impl Default for Cmaes {
    fn default() -> Self {
        Self {
            sigma: 0.3,
            population: None,
            max_evaluations: 400,
            target_loss: 1e-3,
            seed: 1,
        }
    }
}

/// The result of [`fit`].
#[derive(Clone, Debug, PartialEq)]
pub struct Fit {
    /// The best parameters found, in the space's units and order.
    pub params: Vec<f64>,
    /// Their loss.
    pub loss: f64,
    /// Their measurements, in target order.
    pub measurements: Vec<f64>,
    /// Objective evaluations made.
    pub evaluations: usize,
    /// The best loss after each generation.
    pub history: Vec<f64>,
}

impl Fit {
    /// The fit in the lab's report format: parameters (bounded by their
    /// ranges), target measurements (bounded by their tolerance bands),
    /// the loss, and a check per target that it was met.
    #[must_use]
    pub fn report(&self, subject: &str, space: &Space, targets: &[Target]) -> Report {
        let mut r = Report::new(subject);
        for (d, v) in space.dimensions.iter().zip(&self.params) {
            r.measure(&format!("param.{}", d.name), *v, "", Some([d.lo, d.hi]));
        }
        for (t, m) in targets.iter().zip(&self.measurements) {
            let band = match t.goal {
                Goal::Equal => [t.value - t.tolerance, t.value + t.tolerance],
                Goal::AtLeast => [t.value - t.tolerance, f64::MAX],
                Goal::AtMost => [f64::MIN, t.value + t.tolerance],
            };
            r.measure(&format!("target.{}", t.name), *m, "", Some(band));
        }
        r.measure("loss", self.loss, "", None);
        r.measure(
            "evaluations",
            f64::from(u32::try_from(self.evaluations).unwrap_or(u32::MAX)),
            "",
            None,
        );
        r.check(
            "history",
            true,
            format!("best loss per generation: {:?}", self.history),
        );
        r
    }
}

/// A standard normal draw keyed by `words`.
fn normal(seed: u64, words: &[u64]) -> f64 {
    let h = hash(seed, words);
    let u1 = unit_f64(h).max(1e-300);
    let u2 = unit_f64(hash(h, &[1]));
    libm::sqrt(-2.0 * libm::log(u1)) * libm::cos(core::f64::consts::TAU * u2)
}

/// The eigen-decomposition of a symmetric matrix (Jacobi rotations):
/// eigenvalues, and eigenvectors as the columns of the returned matrix.
fn eigen(a: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut m: Vec<Vec<f64>> = a.to_vec();
    let mut v = vec![vec![0.0; n]; n];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..100 {
        let off: f64 = m
            .iter()
            .enumerate()
            .map(|(i, row)| row[i + 1..].iter().map(|x| x * x).sum::<f64>())
            .sum();
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if m[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
                let t = theta.signum() / (theta.abs() + libm::sqrt(theta * theta + 1.0));
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / libm::sqrt(t * t + 1.0);
                let s = t * c;
                for row in &mut m {
                    let (mkp, mkq) = (row[p], row[q]);
                    row[p] = c * mkp - s * mkq;
                    row[q] = s * mkp + c * mkq;
                }
                let (upper, lower) = m.split_at_mut(q);
                for (mpk, mqk) in upper[p].iter_mut().zip(lower[0].iter_mut()) {
                    let (a, b) = (*mpk, *mqk);
                    *mpk = c * a - s * b;
                    *mqk = s * a + c * b;
                }
                for row in &mut v {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    ((0..n).map(|i| m[i][i]).collect(), v)
}

/// Minimizes the [`loss`] of `objective`'s measurements against `targets`
/// over `space` with CMA-ES: see the [module docs](self).
///
/// The search runs in each dimension's normalized `[0, 1]` range from
/// `start` (in the space's units; the middle of every range when `None`).
/// Samples outside the range are evaluated at the nearest point inside it
/// and penalized by their squared distance, so the search stays inside.
///
/// # Errors
///
/// The objective's first error.
pub fn fit<E>(
    space: &Space,
    targets: &[Target],
    options: &Cmaes,
    start: Option<&[f64]>,
    mut objective: impl FnMut(&[f64]) -> Result<Vec<f64>, E>,
) -> Result<Fit, E> {
    let n = space.dimensions.len().max(1);
    #[expect(clippy::cast_precision_loss, reason = "small dimension counts")]
    let nf = n as f64;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "small population sizes"
    )]
    let lambda = options
        .population
        .unwrap_or(4 + (3.0 * libm::log(nf)) as usize)
        .max(2);
    let mu = lambda / 2;
    #[expect(clippy::cast_precision_loss, reason = "small population sizes")]
    let raw: Vec<f64> = (0..mu)
        .map(|i| libm::log(mu as f64 + 0.5) - libm::log(i as f64 + 1.0))
        .collect();
    let total: f64 = raw.iter().sum();
    let weights: Vec<f64> = raw.iter().map(|w| w / total).collect();
    let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
    let cc = (4.0 + mu_eff / nf) / (nf + 4.0 + 2.0 * mu_eff / nf);
    let cs = (mu_eff + 2.0) / (nf + mu_eff + 5.0);
    let c1 = 2.0 / ((nf + 1.3) * (nf + 1.3) + mu_eff);
    let cmu =
        (1.0 - c1).min(2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((nf + 2.0) * (nf + 2.0) + mu_eff));
    let damps = 1.0 + 2.0 * (libm::sqrt((mu_eff - 1.0) / (nf + 1.0)) - 1.0).max(0.0) + cs;
    let chi_n = libm::sqrt(nf) * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));

    let mut mean: Vec<f64> = match start {
        Some(s) => space
            .dimensions
            .iter()
            .zip(s)
            .map(|(d, v)| d.position(*v))
            .collect(),
        None => vec![0.5; n],
    };
    let mut sigma = options.sigma;
    let mut c = vec![vec![0.0; n]; n];
    for (i, row) in c.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    let (mut ps, mut pc) = (vec![0.0; n], vec![0.0; n]);
    let to_params = |x: &[f64]| -> Vec<f64> {
        space
            .dimensions
            .iter()
            .zip(x)
            .map(|(d, t)| d.at(*t))
            .collect()
    };
    let mut evaluate = |x: &[f64]| -> Result<(f64, Vec<f64>), E> {
        let inside: Vec<f64> = x.iter().map(|t| t.clamp(0.0, 1.0)).collect();
        let penalty: f64 = x.iter().zip(&inside).map(|(a, b)| (a - b) * (a - b)).sum();
        let measured = objective(&to_params(&inside))?;
        Ok((loss(targets, &measured) + 1e3 * penalty, measured))
    };

    let (first_loss, first_measured) = evaluate(&mean)?;
    let mut best = Fit {
        params: to_params(&mean.iter().map(|t| t.clamp(0.0, 1.0)).collect::<Vec<_>>()),
        loss: first_loss,
        measurements: first_measured,
        evaluations: 1,
        history: Vec::new(),
    };
    let mut generation = 0_u64;
    while best.evaluations + lambda <= options.max_evaluations && best.loss > options.target_loss {
        let (d2, b) = eigen(&c);
        let d: Vec<f64> = d2.iter().map(|v| libm::sqrt(v.max(1e-20))).collect();
        let mut samples: Vec<(f64, Vec<f64>, Vec<f64>)> = Vec::with_capacity(lambda);
        for k in 0..lambda {
            let z: Vec<f64> = (0..n)
                .map(|i| normal(options.seed, &[generation, k as u64, i as u64]))
                .collect();
            // y = B D z
            let y: Vec<f64> = (0..n)
                .map(|i| (0..n).map(|j| b[i][j] * d[j] * z[j]).sum())
                .collect();
            let x: Vec<f64> = (0..n).map(|i| mean[i] + sigma * y[i]).collect();
            let (l, measured) = evaluate(&x)?;
            best.evaluations += 1;
            if l < best.loss {
                best.loss = l;
                best.params = to_params(&x.iter().map(|t| t.clamp(0.0, 1.0)).collect::<Vec<_>>());
                best.measurements = measured;
            }
            samples.push((l, x, y));
        }
        // Rank by loss; ties keep sample order, so the run is deterministic.
        samples.sort_by(|a, b| a.0.total_cmp(&b.0));
        let old = mean.clone();
        mean = (0..n)
            .map(|i| (0..mu).map(|k| weights[k] * samples[k].1[i]).sum())
            .collect();
        let yw: Vec<f64> = (0..n).map(|i| (mean[i] - old[i]) / sigma).collect();
        // C^(-1/2) y_w = B D⁻¹ Bᵀ y_w.
        let bty: Vec<f64> = (0..n)
            .map(|j| (0..n).map(|i| b[i][j] * yw[i]).sum())
            .collect();
        let inv: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| b[i][j] * bty[j] / d[j]).sum())
            .collect();
        let k_s = libm::sqrt(cs * (2.0 - cs) * mu_eff);
        for i in 0..n {
            ps[i] = (1.0 - cs) * ps[i] + k_s * inv[i];
        }
        let ps_norm = libm::sqrt(ps.iter().map(|v| v * v).sum::<f64>());
        #[expect(clippy::cast_precision_loss, reason = "small generation counts")]
        let gen_f = (generation + 1) as f64;
        let hsig = ps_norm / libm::sqrt(1.0 - libm::pow(1.0 - cs, 2.0 * gen_f)) / chi_n
            < 1.4 + 2.0 / (nf + 1.0);
        let k_c = libm::sqrt(cc * (2.0 - cc) * mu_eff);
        for i in 0..n {
            pc[i] = (1.0 - cc) * pc[i] + if hsig { k_c * yw[i] } else { 0.0 };
        }
        let delta = if hsig { 0.0 } else { cc * (2.0 - cc) };
        for i in 0..n {
            for j in 0..n {
                let rank_mu: f64 = (0..mu)
                    .map(|k| weights[k] * samples[k].2[i] * samples[k].2[j])
                    .sum();
                c[i][j] = (1.0 - c1 - cmu) * c[i][j]
                    + c1 * (pc[i] * pc[j] + delta * c[i][j])
                    + cmu * rank_mu;
            }
        }
        sigma *= libm::exp((cs / damps) * (ps_norm / chi_n - 1.0));
        sigma = sigma.min(1.0);
        best.history.push(best.loss);
        generation += 1;
        if sigma < 1e-6 {
            break;
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmaes_recovers_hidden_parameters_deterministically() {
        // Three parameters, measured through a nonlinear model with
        // coupling; the targets are the model at hidden values.
        let space = Space {
            dimensions: vec![
                Dimension::linear("a", 0.0, 1.0),
                Dimension::log("b", 0.01, 10.0),
                Dimension::linear("c", -5.0, 5.0),
            ],
        };
        let model = |p: &[f64]| -> Result<Vec<f64>, ()> {
            Ok(vec![
                p[0] + 0.1 * p[2],
                libm::log(p[1]) * (1.0 + p[0]),
                p[2] * p[2] + p[0],
            ])
        };
        let hidden = [0.3, 0.5, 1.5];
        let truth = model(&hidden).unwrap();
        let targets: Vec<Target> = truth
            .iter()
            .enumerate()
            .map(|(i, v)| Target::equal(&format!("m{i}"), *v, 1e-3))
            .collect();
        let options = Cmaes {
            max_evaluations: 3000,
            target_loss: 1e-2,
            ..Cmaes::default()
        };
        let a = fit(&space, &targets, &options, None, model).unwrap();
        assert!(a.loss <= 1e-2, "{a:?}");
        for (p, h) in a.params.iter().zip(hidden) {
            assert!((p - h).abs() < 0.01 * (1.0 + h.abs()), "{p} vs {h}");
        }
        let b = fit(&space, &targets, &options, None, model).unwrap();
        assert_eq!(a, b, "deterministic under a seed");
        let report = a.report("toy", &space, &targets);
        assert!(report.passed(), "{}", report.to_json());
    }

    #[test]
    fn eigen_decomposes_symmetric_matrices() {
        let m = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (values, vectors) = eigen(&m);
        let mut sorted = values.clone();
        sorted.sort_by(f64::total_cmp);
        assert!((sorted[0] - 1.0).abs() < 1e-9 && (sorted[1] - 3.0).abs() < 1e-9);
        let v0 = [vectors[0][0], vectors[1][0]];
        let mv = [2.0 * v0[0] + v0[1], v0[0] + 2.0 * v0[1]];
        assert!((mv[0] - values[0] * v0[0]).abs() < 1e-9);
    }
}
