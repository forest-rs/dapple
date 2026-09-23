// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The scalar field trait and domain-checked coordinate transforms.

use alloc::boxed::Box;

use glam::{Mat2, Vec2};

use crate::domain::{Domain, DomainError, Footprint};

/// A point-evaluable scalar field over a [`Domain`].
///
/// `eval` is a pure function of its arguments: equal inputs give bit-identical
/// results on every platform. `footprint` is the size of the region the value
/// stands for; fields drop detail finer than it (see [`Footprint`]).
pub trait ScalarField {
    /// The domain this field is defined over.
    fn domain(&self) -> Domain;

    /// Evaluates the field at `p`, band-limited to `footprint`.
    fn eval(&self, p: Vec2, footprint: Footprint) -> f32;
}

impl<F: ScalarField + ?Sized> ScalarField for &F {
    fn domain(&self) -> Domain {
        (**self).domain()
    }

    fn eval(&self, p: Vec2, footprint: Footprint) -> f32 {
        (**self).eval(p, footprint)
    }
}

impl<F: ScalarField + ?Sized> ScalarField for Box<F> {
    fn domain(&self) -> Domain {
        (**self).domain()
    }

    fn eval(&self, p: Vec2, footprint: Footprint) -> f32 {
        (**self).eval(p, footprint)
    }
}

/// An affine map of domain coordinates: `p' = matrix * p + translation`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Affine2 {
    /// Linear part, column-major.
    pub matrix: Mat2,
    /// Translation, applied after `matrix`.
    pub translation: Vec2,
}

impl Affine2 {
    /// The identity map.
    pub const IDENTITY: Self = Self {
        matrix: Mat2::IDENTITY,
        translation: Vec2::ZERO,
    };

    /// A pure translation.
    #[must_use]
    pub const fn translation(offset: Vec2) -> Self {
        Self {
            matrix: Mat2::IDENTITY,
            translation: offset,
        }
    }

    /// A per-axis scale.
    #[must_use]
    pub const fn scale(scale: Vec2) -> Self {
        Self {
            matrix: Mat2::from_diagonal(scale),
            translation: Vec2::ZERO,
        }
    }

    /// Applies the map to `p`.
    #[must_use]
    pub fn apply(&self, p: Vec2) -> Vec2 {
        self.matrix * p + self.translation
    }

    /// The largest factor by which the map stretches any direction (the
    /// spectral norm of `matrix`).
    #[must_use]
    pub fn max_stretch(&self) -> f32 {
        let [a, b] = self.matrix.x_axis.to_array();
        let [c, d] = self.matrix.y_axis.to_array();
        let sum = a * a + b * b + c * c + d * d;
        let det = a * d - b * c;
        let disc = libm::sqrtf((sum * sum - 4.0 * det * det).max(0.0));
        libm::sqrtf((sum + disc) * 0.5)
    }
}

/// A field evaluated through a coordinate transform: `inner(transform(p))`.
///
/// On a periodic inner domain the transform must map the period lattice onto
/// itself, so the result tiles with the same period. Integer scales, lattice
/// shears, quarter turns of square periods, and any translation qualify.
/// Other maps fail with [`DomainError::NotLatticePreserving`]; wrap the inner
/// field in [`PlaneField`] first to accept a non-repeating result.
#[derive(Clone, Debug)]
pub struct Transformed<F> {
    inner: F,
    transform: Affine2,
    stretch: f32,
}

impl<F: ScalarField> Transformed<F> {
    /// Checks and builds the transformed field.
    pub fn new(inner: F, transform: Affine2) -> Result<Self, DomainError> {
        if !(transform.matrix.is_finite() && transform.translation.is_finite()) {
            return Err(DomainError::InvalidParameter { name: "transform" });
        }
        if let Domain::Periodic { period } = inner.domain()
            && !preserves_lattice(transform.matrix, period)
        {
            return Err(DomainError::NotLatticePreserving);
        }
        Ok(Self {
            inner,
            transform,
            stretch: transform.max_stretch(),
        })
    }

    /// The wrapped field.
    #[must_use]
    pub fn inner(&self) -> &F {
        &self.inner
    }
}

impl<F: ScalarField> ScalarField for Transformed<F> {
    fn domain(&self) -> Domain {
        self.inner.domain()
    }

    fn eval(&self, p: Vec2, footprint: Footprint) -> f32 {
        self.inner
            .eval(self.transform.apply(p), footprint.scaled(self.stretch))
    }
}

/// Column `j` of `matrix` times `period[j]` must be a whole multiple of the
/// period on every axis, so lattice translations map to lattice translations.
fn preserves_lattice(matrix: Mat2, period: [u32; 2]) -> bool {
    let columns = [matrix.x_axis, matrix.y_axis];
    columns.iter().enumerate().all(|(j, column)| {
        column.to_array().iter().enumerate().all(|(i, &m)| {
            let image = f64::from(m) * f64::from(period[j]);
            let steps = image / f64::from(period[i]);
            libm::trunc(steps) == steps
        })
    })
}

/// A field explicitly demoted to [`Domain::Plane`].
///
/// Demotion is how a periodic field enters a non-repeating context, such as a
/// transform that does not preserve its lattice.
#[derive(Clone, Debug)]
pub struct PlaneField<F>(pub F);

impl<F: ScalarField> ScalarField for PlaneField<F> {
    fn domain(&self) -> Domain {
        Domain::Plane
    }

    fn eval(&self, p: Vec2, footprint: Footprint) -> f32 {
        self.0.eval(p, footprint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Constant(Domain);

    impl ScalarField for Constant {
        fn domain(&self) -> Domain {
            self.0
        }

        fn eval(&self, _: Vec2, _: Footprint) -> f32 {
            1.0
        }
    }

    #[test]
    fn periodic_transforms_must_preserve_the_lattice() {
        let square = Domain::periodic(2, 2).unwrap();
        let wide = Domain::periodic(4, 1).unwrap();
        let quarter_turn = Affine2 {
            matrix: Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(-1.0, 0.0)),
            translation: Vec2::new(0.3, 0.7),
        };
        assert!(Transformed::new(Constant(square), quarter_turn).is_ok());
        assert!(matches!(
            Transformed::new(Constant(wide), quarter_turn),
            Err(DomainError::NotLatticePreserving)
        ));
        assert!(Transformed::new(Constant(square), Affine2::scale(Vec2::new(3.0, 2.0))).is_ok());
        assert!(matches!(
            Transformed::new(Constant(square), Affine2::scale(Vec2::splat(1.5))),
            Err(DomainError::NotLatticePreserving)
        ));
        // Demoting accepts any map.
        assert!(
            Transformed::new(
                PlaneField(Constant(square)),
                Affine2::scale(Vec2::splat(1.5))
            )
            .is_ok()
        );
    }

    #[test]
    fn stretch_is_the_spectral_norm() {
        let t = Affine2::scale(Vec2::new(3.0, 0.5));
        assert_eq!(t.max_stretch(), 3.0);
        let rotation = Affine2 {
            matrix: Mat2::from_cols(Vec2::new(0.6, 0.8), Vec2::new(-0.8, 0.6)),
            translation: Vec2::ZERO,
        };
        assert!((rotation.max_stretch() - 1.0).abs() < 1e-6);
    }
}
