// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Approximation reports, made where a richer combination collapses into
//! one OpenPBR parameter set.

use alloc::vec::Vec;
use core::fmt;

use crate::material::ChannelId;

/// What an operation approximated.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ApproximationKind {
    /// Roughness interpolated in `α = r²` rather than as a mixture of two
    /// microfacet distributions.
    RoughnessInAlpha,
    /// Two normal (or tangent) fields averaged and renormalized: a mixture
    /// of two orientations shaded as one.
    NormalsAveraged,
    /// An index of refraction interpolated linearly, though Fresnel
    /// reflectance is not linear in it.
    IorInterpolated,
    /// Two layer stacks (one coated, one not, or with different coats)
    /// mixed parameter by parameter: the coat is laid over the mixed base,
    /// so the uncoated part is shaded partly under the coat.
    LayersMixed,
    /// An identifier chosen by the larger weight: a summary of a texel
    /// several regions or materials share.
    WinnerLabel,
    /// Two coats collapsed into OpenPBR's one: weights combined as
    /// coverage, tints multiplied, lobe widths added in `α²`.
    CoatsCollapsed,
    /// A base normal derived from height by finite differences, to compose
    /// a detail normal onto.
    NormalFromHeight,
    /// Values resampled bilinearly (identifiers by nearest texel) by a
    /// transform that does not map texels onto texels.
    Resampled,
}

impl ApproximationKind {
    /// A one-line description.
    #[must_use]
    pub const fn describe(self) -> &'static str {
        match self {
            Self::RoughnessInAlpha => "roughness interpolated in alpha, not as a lobe mixture",
            Self::NormalsAveraged => "orientations averaged and renormalized",
            Self::IorInterpolated => "index of refraction interpolated linearly",
            Self::LayersMixed => "layer stacks mixed parameter by parameter",
            Self::WinnerLabel => "identifier chosen by the larger weight",
            Self::CoatsCollapsed => "two coats collapsed into one",
            Self::NormalFromHeight => "base normal derived from height",
            Self::Resampled => "values resampled between texels",
        }
    }
}

/// One approximation: what, in which channel, over how many texels.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Approximation {
    /// The channel it affects.
    pub channel: ChannelId,
    /// What was approximated.
    pub kind: ApproximationKind,
    /// Texels where it made a difference.
    pub texels: u64,
}

/// What one material operation did, and what it approximated.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// The operation: `"select"`, `"detail"`, `"coat"`, `"deposit"` or
    /// `"transform"`.
    pub operation: &'static str,
    /// Texels the operation's decision was fractional or partial on: a
    /// selection weight strictly between 0 and 1, a partial coat or
    /// deposit.
    pub partial: u64,
    /// Every approximation made, one per channel and kind, in channel
    /// order. Empty when the result is exact.
    pub approximations: Vec<Approximation>,
}

impl Report {
    pub(crate) fn new(operation: &'static str) -> Self {
        Self {
            operation,
            partial: 0,
            approximations: Vec::new(),
        }
    }

    pub(crate) fn add(&mut self, channel: ChannelId, kind: ApproximationKind, texels: u64) {
        if texels == 0 {
            return;
        }
        if let Some(a) = self
            .approximations
            .iter_mut()
            .find(|a| a.channel == channel && a.kind == kind)
        {
            a.texels += texels;
        } else {
            self.approximations.push(Approximation {
                channel,
                kind,
                texels,
            });
        }
    }

    /// Whether the operation approximated nothing.
    #[must_use]
    pub fn is_exact(&self) -> bool {
        self.approximations.is_empty()
    }

    /// Whether any approximation of `kind` was made.
    #[must_use]
    pub fn has(&self, kind: ApproximationKind) -> bool {
        self.approximations.iter().any(|a| a.kind == kind)
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} partial texels", self.operation, self.partial)?;
        if self.is_exact() {
            return f.write_str(", exact");
        }
        for a in &self.approximations {
            write!(
                f,
                "; {}: {} ({} texels)",
                a.channel,
                a.kind.describe(),
                a.texels
            )?;
        }
        Ok(())
    }
}
