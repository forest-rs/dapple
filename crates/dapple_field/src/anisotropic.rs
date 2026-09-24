// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Anisotropic footprints, as a local linear approximation.
//!
//! A [`Footprint`] is one width: the side of a square region. A texel seen
//! at a grazing angle, or a chart stretched over a surface, covers an
//! elongated region instead, and one width either blurs it along its short
//! axis (the long width) or aliases along its long one (the short width).
//!
//! - [`Covariance2`] is the region's second moment in a 2D parameter space:
//!   a box of width `w` has variance `w²/12` per axis, and a linear map `A`
//!   takes a covariance `Σ` to `A Σ Aᵀ`. Its eigenvectors are the region's
//!   axes, and [`Covariance2::taps`] covers it with taps along its major axis,
//!   each as wide as the minor one: the footprint-assembly scheme of
//!   anisotropic texture filtering.
//! - [`SurfaceFootprint`] keeps a footprint on a surface: the 2D covariance
//!   in the chart's parameter space plus the differential basis `J`, the
//!   3 × 2 Jacobian from chart parameters to material space. The material
//!   space covariance `J Σ Jᵀ` (3 × 3, rank at most 2) keeps the footprint's
//!   orientation in the solid, which a single width cannot:
//!   [`SurfaceFootprint::taps`] lays the taps along the major axis there.
//!
//! These are local linear approximations: through a nonlinear warp the
//! region is the image of the ellipse under the warp's Jacobian at its
//! center, not its exact image.

use alloc::vec::Vec;

use glam::{Mat2, Vec2, Vec3};

use crate::domain::Footprint;
use crate::field::ScalarField;
use crate::solid::SolidField;

/// The variance of a unit box along one axis.
const BOX_VARIANCE: f32 = 1.0 / 12.0;

/// A footprint's covariance in a 2D parameter space: see the [module
/// docs](self).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Covariance2 {
    /// The symmetric matrix `[[xx, xy], [xy, yy]]`.
    pub matrix: Mat2,
}

/// The axes of a covariance: directions and box-equivalent widths.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Axes2 {
    /// Unit direction of the major axis.
    pub major: Vec2,
    /// Width along the major axis: the side of the box of equal variance.
    pub major_width: f32,
    /// Width along the minor axis, perpendicular to the major.
    pub minor_width: f32,
}

impl Covariance2 {
    /// The covariance of a square box `width` across.
    #[must_use]
    pub fn isotropic(width: f32) -> Self {
        Self {
            matrix: Mat2::from_diagonal(Vec2::splat(width * width * BOX_VARIANCE)),
        }
    }

    /// The covariance of a texel box `texel` wide per axis carried through
    /// the linear map `jacobian` (columns: the image of a unit step along
    /// x and along y): `J diag(t²/12) Jᵀ`.
    #[must_use]
    pub fn from_jacobian(jacobian: Mat2, texel: Vec2) -> Self {
        let d = Mat2::from_diagonal(texel * texel * BOX_VARIANCE);
        Self {
            matrix: jacobian * d * jacobian.transpose(),
        }
    }

    /// The covariance after the linear map `a`: `A Σ Aᵀ`.
    #[must_use]
    pub fn transformed(self, a: Mat2) -> Self {
        Self {
            matrix: a * self.matrix * a.transpose(),
        }
    }

    /// The eigen-axes, with box-equivalent widths `√(12 λ)`.
    #[must_use]
    pub fn axes(self) -> Axes2 {
        let m = self.matrix;
        let (a, b, c) = (m.x_axis.x, 0.5 * (m.x_axis.y + m.y_axis.x), m.y_axis.y);
        let mid = 0.5 * (a + c);
        let r = libm::sqrtf((0.5 * (a - c)) * (0.5 * (a - c)) + b * b);
        let (l1, l2) = ((mid + r).max(0.0), (mid - r).max(0.0));
        let major = if b.abs() > 1e-20 {
            Vec2::new(b, l1 - a).normalize_or(Vec2::X)
        } else if a >= c {
            Vec2::X
        } else {
            Vec2::Y
        };
        let width = |l: f32| libm::sqrtf(l / BOX_VARIANCE);
        Axes2 {
            major,
            major_width: width(l1),
            minor_width: width(l2),
        }
    }

    /// One width for evaluators that take one: the major axis's
    /// (conservative, blurs the minor axis) or the geometric mean (keeps
    /// the area).
    #[must_use]
    pub fn isotropic_width(self, conservative: bool) -> f32 {
        let axes = self.axes();
        if conservative {
            axes.major_width
        } else {
            libm::sqrtf(axes.major_width * axes.minor_width)
        }
    }

    /// Taps covering the footprint: `n = min(max_taps, ⌈major / minor⌉)`
    /// offsets evenly spread along the major axis over its width, each
    /// with an isotropic footprint as wide as the major width over `n` (at
    /// least the minor width). Averaging a field over them approximates
    /// its average over the footprint without blurring the minor axis.
    #[must_use]
    pub fn taps(self, max_taps: u32) -> Vec<(Vec2, Footprint)> {
        let axes = self.axes();
        let (offsets, width) = tap_layout(axes.major_width, axes.minor_width, max_taps);
        offsets
            .into_iter()
            .map(|t| {
                (
                    axes.major * t,
                    Footprint::new(width).unwrap_or(Footprint::POINT),
                )
            })
            .collect()
    }
}

/// Offsets along an axis of width `major`, and each tap's width. The
/// count rounds a ratio within a thousandth of a whole number down to it,
/// so rounding in the axes does not add a tap.
fn tap_layout(major: f32, minor: f32, max_taps: u32) -> (Vec<f32>, f32) {
    let ratio = if minor > 0.0 {
        major / minor
    } else {
        f32::INFINITY
    };
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the tap count is small and positive"
    )]
    let n = if ratio.is_finite() {
        (libm::ceilf(ratio - 1e-3) as u32).clamp(1, max_taps.max(1))
    } else {
        max_taps.max(1)
    };
    #[expect(clippy::cast_precision_loss, reason = "tap counts are small")]
    let nf = n as f32;
    let width = (major / nf).max(minor);
    #[expect(clippy::cast_precision_loss, reason = "tap counts are small")]
    let offsets = (0..n)
        .map(|i| ((i as f32 + 0.5) / nf - 0.5) * major)
        .collect();
    (offsets, width)
}

/// A footprint on a surface: see the [module docs](self).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SurfaceFootprint {
    /// The footprint's covariance in chart parameters (texels, say).
    pub covariance: Covariance2,
    /// The differential basis: material-space steps per unit of each chart
    /// parameter, the columns of the 3 × 2 Jacobian `J`.
    pub basis: [Vec3; 2],
}

/// The axes of a surface footprint in material space.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Axes3 {
    /// Unit direction of the major axis, in material space.
    pub major: Vec3,
    /// Width along it.
    pub major_width: f32,
    /// Unit direction of the minor axis, in material space.
    pub minor: Vec3,
    /// Width along it.
    pub minor_width: f32,
}

impl SurfaceFootprint {
    /// A texel of a chart whose one-texel steps along x and y land `basis`
    /// apart in material space.
    #[must_use]
    pub fn texel(basis: [Vec3; 2]) -> Self {
        Self {
            covariance: Covariance2::isotropic(1.0),
            basis,
        }
    }

    /// The material-space covariance `J Σ Jᵀ`, as three rows; its rank is
    /// at most 2.
    #[must_use]
    pub fn material_covariance(&self) -> [Vec3; 3] {
        let [u, v] = self.basis;
        let m = self.covariance.matrix;
        let (a, b, c) = (m.x_axis.x, 0.5 * (m.x_axis.y + m.y_axis.x), m.y_axis.y);
        // J Σ Jᵀ = a u uᵀ + b (u vᵀ + v uᵀ) + c v vᵀ.
        let row = |i: usize| {
            Vec3::new(
                a * u[i] * u.x + b * (u[i] * v.x + v[i] * u.x) + c * v[i] * v.x,
                a * u[i] * u.y + b * (u[i] * v.y + v[i] * u.y) + c * v[i] * v.y,
                a * u[i] * u.z + b * (u[i] * v.z + v[i] * u.z) + c * v[i] * v.z,
            )
        };
        [row(0), row(1), row(2)]
    }

    /// The material-space axes: with `Σ = L Lᵀ` and `A = J L` (3 × 2), the
    /// eigenpairs of the 2 × 2 `Aᵀ A` are those of `J Σ Jᵀ`'s nonzero part,
    /// and `A w / √λ` its axes.
    #[must_use]
    pub fn axes(&self) -> Axes3 {
        let m = self.covariance.matrix;
        let (a, b, c) = (m.x_axis.x, 0.5 * (m.x_axis.y + m.y_axis.x), m.y_axis.y);
        // Cholesky of Σ.
        let l11 = libm::sqrtf(a.max(0.0));
        let l21 = if l11 > 0.0 { b / l11 } else { 0.0 };
        let l22 = libm::sqrtf((c - l21 * l21).max(0.0));
        let [u, v] = self.basis;
        let a1 = u * l11 + v * l21;
        let a2 = v * l22;
        let gram = Covariance2 {
            matrix: Mat2::from_cols(
                Vec2::new(a1.dot(a1), a1.dot(a2)),
                Vec2::new(a1.dot(a2), a2.dot(a2)),
            ),
        };
        let axes = gram.axes();
        let w1 = axes.major;
        let w2 = Vec2::new(-w1.y, w1.x);
        let dir = |w: Vec2| (a1 * w.x + a2 * w.y).normalize_or_zero();
        // Box-equivalent widths from the eigenvalues, as in 2D.
        Axes3 {
            major: dir(w1),
            major_width: axes.major_width,
            minor: dir(w2),
            minor_width: axes.minor_width,
        }
    }

    /// Taps covering the footprint in material space, as
    /// [`Covariance2::taps`] does in 2D.
    #[must_use]
    pub fn taps(&self, max_taps: u32) -> Vec<(Vec3, Footprint)> {
        let axes = self.axes();
        let (offsets, width) = tap_layout(axes.major_width, axes.minor_width, max_taps);
        offsets
            .into_iter()
            .map(|t| {
                (
                    axes.major * t,
                    Footprint::new(width).unwrap_or(Footprint::POINT),
                )
            })
            .collect()
    }
}

/// `field` averaged over the taps of `covariance` around `p`: an
/// anisotropic footprint evaluation of any scalar field.
#[must_use]
pub fn eval_anisotropic(
    field: &impl ScalarField,
    p: Vec2,
    covariance: Covariance2,
    max_taps: u32,
) -> f32 {
    let taps = covariance.taps(max_taps);
    let sum: f32 = taps.iter().map(|&(d, fp)| field.eval(p + d, fp)).sum();
    #[expect(clippy::cast_precision_loss, reason = "tap counts are small")]
    let n = taps.len() as f32;
    sum / n
}

/// `field` averaged over the taps of `footprint` around `p` in material
/// space.
#[must_use]
pub fn eval_anisotropic3(
    field: &impl SolidField,
    p: Vec3,
    footprint: &SurfaceFootprint,
    max_taps: u32,
) -> f32 {
    let taps = footprint.taps(max_taps);
    let sum: f32 = taps.iter().map(|&(d, fp)| field.eval(p + d, fp)).sum();
    #[expect(clippy::cast_precision_loss, reason = "tap counts are small")]
    let n = taps.len() as f32;
    sum / n
}

/// The complete expression `field` integrated over the parallelogram a
/// texel box maps to under `jacobian` (columns: the images of the box's
/// unit edges, in domain units) around `p`, with `n` × `n` point samples:
/// the reference an anisotropic evaluation stands for.
#[must_use]
pub fn reference_parallelogram(field: &impl ScalarField, p: Vec2, jacobian: Mat2, n: u32) -> f32 {
    let n = n.max(1);
    let mut sum = 0.0_f64;
    for j in 0..n {
        for i in 0..n {
            #[expect(clippy::cast_precision_loss, reason = "sample counts are small")]
            let uv = Vec2::new(
                (i as f32 + 0.5) / n as f32 - 0.5,
                (j as f32 + 0.5) / n as f32 - 0.5,
            );
            sum += f64::from(field.eval(p + jacobian * uv, Footprint::POINT));
        }
    }
    #[expect(clippy::cast_possible_truncation, reason = "a mean of f32 values")]
    let mean = (sum / f64::from(n * n)) as f32;
    mean
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axes_follow_the_map() {
        // A texel stretched 8:1 along a direction 30° off x.
        let (s, c) = (
            libm::sinf(core::f32::consts::FRAC_PI_6),
            libm::cosf(core::f32::consts::FRAC_PI_6),
        );
        let rot = Mat2::from_cols(Vec2::new(c, s), Vec2::new(-s, c));
        let j = rot * Mat2::from_diagonal(Vec2::new(8.0, 1.0));
        let cov = Covariance2::from_jacobian(j, Vec2::splat(0.01));
        let axes = cov.axes();
        assert!(
            (axes.major.dot(Vec2::new(c, s)).abs() - 1.0).abs() < 1e-4,
            "{axes:?}"
        );
        assert!((axes.major_width - 0.08).abs() < 1e-4, "{axes:?}");
        assert!((axes.minor_width - 0.01).abs() < 1e-4, "{axes:?}");
        let taps = cov.taps(16);
        assert_eq!(taps.len(), 8, "one tap per minor width");
        assert!((taps[0].1.width() - 0.01).abs() < 1e-4);
    }

    #[test]
    fn surface_footprints_keep_their_orientation_in_material_space() {
        // A chart texel whose x step runs 5 mm along z and y step 1 mm
        // along x, as a grazing face of a timber.
        let basis = [Vec3::new(0.0, 0.0, 0.005), Vec3::new(0.001, 0.0, 0.0)];
        let f = SurfaceFootprint::texel(basis);
        let m = f.material_covariance();
        let det = m[0].dot(m[1].cross(m[2]));
        assert!(det.abs() < 1e-20, "rank at most 2: {det}");
        let axes = f.axes();
        assert!((axes.major.z.abs() - 1.0).abs() < 1e-5, "{axes:?}");
        assert!((axes.minor.x.abs() - 1.0).abs() < 1e-5, "{axes:?}");
        assert!((axes.major_width - 0.005).abs() < 1e-6);
        assert!((axes.minor_width - 0.001).abs() < 1e-6);
        assert_eq!(f.taps(8).len(), 5);
    }

    #[test]
    fn anisotropic_taps_track_the_reference_better_than_one_width() {
        use crate::{Basis, Domain, Noise};
        // Noise of 12 cells per unit under texels stretched 8:1 at 30°:
        // the long axis spans several cells, the short one a fraction.
        let noise = Noise::new(Basis::Gradient, Domain::Plane, Vec2::splat(12.0), 3).unwrap();
        let (s, c) = (
            libm::sinf(core::f32::consts::FRAC_PI_6),
            libm::cosf(core::f32::consts::FRAC_PI_6),
        );
        let rot = Mat2::from_cols(Vec2::new(c, s), Vec2::new(-s, c));
        let j = rot * Mat2::from_diagonal(Vec2::new(0.16, 0.02));
        let cov = Covariance2::from_jacobian(j, Vec2::ONE);
        let (mut aniso, mut major, mut minor) = (0.0_f64, 0.0_f64, 0.0_f64);
        for k in 0..64 {
            let p = Vec2::new(k as f32 * 0.173, k as f32 * 0.311);
            let reference = reference_parallelogram(&noise, p, j, 32);
            let e = |v: f32| f64::from((v - reference) * (v - reference));
            aniso += e(eval_anisotropic(&noise, p, cov, 16));
            let wide = Footprint::new(cov.isotropic_width(true)).unwrap();
            let narrow = Footprint::new(cov.axes().minor_width).unwrap();
            major += e(noise.eval(p, wide));
            minor += e(noise.eval(p, narrow));
        }
        assert!(aniso < major, "taps {aniso} beat the long width {major}");
        assert!(aniso < minor, "taps {aniso} beat the short width {minor}");
    }
}
