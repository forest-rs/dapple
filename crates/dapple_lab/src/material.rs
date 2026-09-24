// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Reports and relationship checks for materials.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use dapple_field::Value;
use dapple_material::ops::{self, MaterialTransform};
use dapple_material::{Channel, ChannelId, Grid, Material, lower, typed_seam};
use dapple_raster::seam::{Axis, SEAM_TOLERANCE};
use glam::Vec2;

use crate::measure::{Stats, feature_size};
use crate::report::Report;

/// A report on `material`:
///
/// - for every bound channel, per component: minimum, maximum and mean,
///   and a check that no value is NaN or infinite;
/// - a check that every parameter lies in the specification's allowed
///   range;
/// - for every map and every axis, the seam measurement: bounded by
///   [`SEAM_TOLERANCE`] along the axes the material promises to tile,
///   informational along the others;
/// - what lowering to texture maps drops.
#[must_use]
pub fn material_report(subject: &str, material: &Material) -> Report {
    let mut r = Report::new(subject);
    let grid = material.grid();
    r.measure("grid.width", f64::from(grid.width), "texels", None);
    r.measure("grid.height", f64::from(grid.height), "texels", None);
    let tiling = material.tiling();
    r.check(
        "tiling",
        true,
        format!("promises x: {}, y: {}", tiling.x, tiling.y),
    );
    for (c, ch) in material.bound() {
        let components = c.port().components();
        let mut non_finite = 0;
        for k in 0..components {
            let stats = Stats::of((0..grid.len()).map(|i| match ch {
                Channel::Constant(v) => component(*v, k),
                Channel::Map(m) => component(m.value(i), k),
            }));
            non_finite += stats.non_finite;
            let name = if components == 1 {
                String::from(c.name())
            } else {
                format!("{}[{k}]", c.name())
            };
            r.measure(&format!("{name}.min"), stats.min, "", None);
            r.measure(&format!("{name}.max"), stats.max, "", None);
            r.measure(&format!("{name}.mean"), stats.mean, "", None);
        }
        r.check(
            &format!("finite.{}", c.name()),
            non_finite == 0,
            format!("{non_finite} values NaN or infinite"),
        );
        if let Channel::Map(m) = ch {
            for (axis, promised) in [(Axis::X, tiling.x), (Axis::Y, tiling.y)] {
                let seam = typed_seam(m, axis);
                let axis_name = if axis == Axis::X { "x" } else { "y" };
                r.measure(
                    &format!("seam.{}.{axis_name}", c.name()),
                    f64::from(seam.ratio),
                    "ratio",
                    promised.then_some([0.0, f64::from(SEAM_TOLERANCE)]),
                );
            }
        }
    }
    let violations = material.range_violations();
    r.check("ranges", violations.is_empty(), format!("{violations:?}"));
    match lower::maps(material) {
        Ok((_, lowering)) => r.check(
            "lowering",
            true,
            format!(
                "normal from height: {}; dropped: {:?}",
                lowering.normal_from_height, lowering.dropped
            ),
        ),
        Err(e) => r.check("lowering", false, format!("{e}")),
    }
    r
}

fn component(v: Value, k: usize) -> f32 {
    match v {
        #[expect(
            clippy::cast_precision_loss,
            reason = "identifiers reported as numbers"
        )]
        Value::Id(id) => id as f32,
        other => other.component(k).unwrap_or(0.0),
    }
}

/// Relationship check: a material-wide transform by whole texels moves
/// every bound map exactly, leaving no channel behind.
#[must_use]
pub fn transform_check(material: &Material) -> Report {
    let mut r = Report::new("transform");
    let grid = material.grid();
    let shift = [3_i64, 2];
    #[expect(clippy::cast_precision_loss, reason = "small texel offsets")]
    let offset = grid.texel * Vec2::new(shift[0] as f32, shift[1] as f32);
    let moved = match ops::transform(material, MaterialTransform::offset(offset)) {
        Ok((m, _)) => m,
        Err(e) => {
            r.check("transform", false, format!("{e}"));
            return r;
        }
    };
    let (w, h) = (i64::from(grid.width), i64::from(grid.height));
    let mut left_behind: Vec<&'static str> = Vec::new();
    for (c, ch) in material.bound() {
        let Channel::Map(m) = ch else { continue };
        let ok = (0..h).step_by(7).all(|y| {
            (0..w).step_by(5).all(|x| {
                #[expect(
                    clippy::cast_sign_loss,
                    clippy::cast_possible_truncation,
                    reason = "in range"
                )]
                let i = (y * w + x) as usize;
                moved.value(c, i) == m.value_at(x - shift[0], y - shift[1])
            })
        });
        if !ok {
            left_behind.push(c.name());
        }
    }
    r.check(
        "no_channel_left_behind",
        left_behind.is_empty() && moved.bound().count() == material.bound().count(),
        format!("channels not moved: {left_behind:?}"),
    );
    r
}

/// Relationship check: realizing at a finer resolution does not change
/// the physical size of features. The fine realization is box-filtered
/// down to the coarse grid and both are measured with the gradient-based
/// [`feature_size`]; they agree (within 15%) when the coarse realization
/// is the band-limited version of the same content. A coarse realization
/// that rescaled features, or that point-sampled detail it should have
/// filtered (aliasing, which reads as smaller features), fails.
///
/// The fine grid must be a whole multiple of the coarse one per axis.
#[must_use]
pub fn resolution_check(
    name: &str,
    channel: ChannelId,
    coarse: &Material,
    fine: &Material,
) -> Report {
    let mut r = Report::new("resolution");
    let (cg, fg) = (coarse.grid(), fine.grid());
    let (Ok(c), Ok(f)) = (coarse.scalar(channel), fine.scalar(channel)) else {
        r.check(
            &format!("{name}.feature_size"),
            false,
            "not a scalar channel",
        );
        return r;
    };
    let (kx, ky) = (fg.width / cg.width.max(1), fg.height / cg.height.max(1));
    if kx == 0 || ky == 0 || kx * cg.width != fg.width || ky * cg.height != fg.height {
        r.check(
            &format!("{name}.feature_size"),
            false,
            "grids are not multiples",
        );
        return r;
    }
    let mut down = Vec::with_capacity(cg.len());
    for y in 0..cg.height {
        for x in 0..cg.width {
            let mut sum = 0.0;
            for j in 0..ky {
                for i in 0..kx {
                    sum += f.values()[((y * ky + j) * fg.width + x * kx + i) as usize];
                }
            }
            #[expect(clippy::cast_precision_loss, reason = "small factors")]
            let n = (kx * ky) as f32;
            down.push(sum / n);
        }
    }
    let Ok(down) = cg.raster(down) else {
        r.check(&format!("{name}.feature_size"), false, "grid");
        return r;
    };
    let (a, b) = (feature_size(&c), feature_size(&down));
    r.measure(&format!("{name}.feature_size.coarse"), a, "m", None);
    r.measure(&format!("{name}.feature_size.fine_filtered"), b, "m", None);
    let ratio = if b > 0.0 { a / b } else { f64::NAN };
    r.measure(
        &format!("{name}.feature_size.ratio"),
        ratio,
        "",
        Some([0.85, 1.15]),
    );
    r
}

/// Relationship check: two ways to the same result agree bit for bit, as
/// an incremental update must with a clean rebuild.
#[must_use]
pub fn agreement(name: &str, incremental: u64, clean: u64) -> Report {
    let mut r = Report::new("agreement");
    r.check(
        name,
        incremental == clean,
        format!("{incremental:016x} vs {clean:016x}"),
    );
    r
}

/// The grid of a unit tile with `n` texels a side, wrapping.
#[must_use]
pub fn unit_tile(n: u32) -> Grid {
    #[expect(clippy::cast_precision_loss, reason = "small grid sizes")]
    let texel = Vec2::splat(1.0 / n as f32);
    Grid {
        width: n,
        height: n,
        origin: Vec2::ZERO,
        texel,
        edge: dapple_field::Edge::Wrap,
    }
}
