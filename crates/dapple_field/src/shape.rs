// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Local shapes: fields that are zero outside a bounded region.

use glam::Vec2;

use crate::domain::{Domain, DomainError, Footprint};
use crate::field::ScalarField;
use crate::raster::Region;

/// A filled disk: coverage 1 inside, 0 outside, with a soft edge.
///
/// The edge ramps from 1 to 0 across a band centered on the rim, `w` domain
/// units wide, where `w` is the larger of `softness` and the evaluation
/// footprint. The ramp is a smoothstep, so a disk realized at any resolution
/// is antialiased, and a disk with zero softness is band-limited by the
/// footprint alone.
///
/// On a periodic domain, distance is measured to the nearest repeat of the
/// center, so the disk tiles; the disk and its soft edge must fit in half a
/// period on each axis.
///
/// Outside [`Disk::support`] grown by half the evaluation footprint, the
/// value is exactly zero.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Disk {
    domain: Domain,
    center: Vec2,
    radius: f32,
    softness: f32,
}

impl Disk {
    /// A disk of `radius` around `center` with an edge band `softness` wide.
    ///
    /// # Errors
    ///
    /// [`DomainError::InvalidParameter`] when `center` is not finite,
    /// `radius` is not positive and finite, `softness` is negative or not
    /// finite, or on a periodic domain the disk and its edge do not fit in
    /// half a period.
    pub fn new(
        domain: Domain,
        center: Vec2,
        radius: f32,
        softness: f32,
    ) -> Result<Self, DomainError> {
        if !center.is_finite() {
            return Err(DomainError::InvalidParameter { name: "center" });
        }
        if !(radius.is_finite() && radius > 0.0) {
            return Err(DomainError::InvalidParameter { name: "radius" });
        }
        if !(softness.is_finite() && softness >= 0.0) {
            return Err(DomainError::InvalidParameter { name: "softness" });
        }
        if let Some(period) = domain.period() {
            let reach = radius + softness * 0.5;
            let half = period[0].min(period[1]) as f32 * 0.5;
            if reach >= half {
                return Err(DomainError::InvalidParameter { name: "radius" });
            }
        }
        Ok(Self {
            domain,
            center,
            radius,
            softness,
        })
    }

    /// The disk's center.
    #[must_use]
    pub const fn center(&self) -> Vec2 {
        self.center
    }

    /// The disk's radius.
    #[must_use]
    pub const fn radius(&self) -> f32 {
        self.radius
    }

    /// The edge band's width at a point footprint.
    #[must_use]
    pub const fn softness(&self) -> f32 {
        self.softness
    }

    /// The square outside which the disk is zero at a point footprint.
    ///
    /// On a periodic domain the square may extend past the period; the field
    /// is also zero outside every repeat of it.
    #[must_use]
    pub fn support(&self) -> Region {
        let reach = self.radius + self.softness * 0.5;
        Region {
            origin: self.center - Vec2::splat(reach),
            size: Vec2::splat(reach * 2.0),
        }
    }

    fn distance(&self, p: Vec2) -> f32 {
        let mut d = p - self.center;
        if let Some([px, py]) = self.domain.period() {
            let period = Vec2::new(px as f32, py as f32);
            // Offset to the nearest repeat of the center.
            d -= period * (d / period).round();
        }
        d.length()
    }
}

impl ScalarField for Disk {
    fn domain(&self) -> Domain {
        self.domain
    }

    fn eval(&self, p: Vec2, footprint: Footprint) -> f32 {
        let d = self.distance(p);
        let width = self.softness.max(footprint.width());
        if width <= 0.0 {
            return if d <= self.radius { 1.0 } else { 0.0 };
        }
        let t = ((self.radius + width * 0.5 - d) / width).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inside_edge_and_outside() {
        let disk = Disk::new(Domain::Plane, Vec2::new(1.0, 2.0), 0.5, 0.2).unwrap();
        assert_eq!(disk.eval(Vec2::new(1.0, 2.0), Footprint::POINT), 1.0);
        assert!((disk.eval(Vec2::new(1.5, 2.0), Footprint::POINT) - 0.5).abs() < 1e-5);
        assert_eq!(disk.eval(Vec2::new(1.61, 2.0), Footprint::POINT), 0.0);
        let hard = Disk::new(Domain::Plane, Vec2::ZERO, 0.5, 0.0).unwrap();
        assert_eq!(hard.eval(Vec2::new(0.5, 0.0), Footprint::POINT), 1.0);
        assert_eq!(hard.eval(Vec2::new(0.51, 0.0), Footprint::POINT), 0.0);
        let wide = Footprint::new(0.2).unwrap();
        assert!((hard.eval(Vec2::new(0.5, 0.0), wide) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn zero_outside_the_support_grown_by_half_the_footprint() {
        let disk = Disk::new(Domain::Plane, Vec2::ZERO, 0.3, 0.1).unwrap();
        let support = disk.support();
        let footprint = Footprint::new(0.4).unwrap();
        let pad = footprint.width() * 0.5;
        for i in 0..64 {
            let a = i as f32 * core::f32::consts::TAU / 64.0;
            let r = support.size.x * 0.5 + pad + 1e-4;
            let p = Vec2::new(libm::cosf(a), libm::sinf(a)) * r * core::f32::consts::SQRT_2;
            assert_eq!(disk.eval(p, footprint), 0.0, "{p}");
        }
    }

    #[test]
    fn periodic_disks_wrap_and_must_fit() {
        let domain = Domain::periodic(2, 2).unwrap();
        let disk = Disk::new(domain, Vec2::new(0.1, 0.1), 0.3, 0.0).unwrap();
        assert_eq!(disk.eval(Vec2::new(1.95, 1.95), Footprint::POINT), 1.0);
        assert_eq!(
            disk.eval(Vec2::new(0.3, 0.1), Footprint::POINT),
            disk.eval(Vec2::new(2.3, 2.1), Footprint::POINT)
        );
        assert!(Disk::new(domain, Vec2::ZERO, 1.0, 0.0).is_err());
        assert!(Disk::new(Domain::Plane, Vec2::ZERO, 0.0, 0.0).is_err());
        assert!(Disk::new(Domain::Plane, Vec2::ZERO, 1.0, -1.0).is_err());
    }
}
