// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Port types and values of field programs.

use core::fmt;

use glam::{Vec2, Vec3};

/// Color primaries of a [`PortType::Color`] value.
///
/// Colors are always linear; a transfer function such as sRGB exists only
/// when a texture is encoded, so gamma-space arithmetic cannot be written.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Primaries {
    /// ITU-R BT.709 (sRGB) primaries with a D65 white point, which OpenPBR
    /// parameters use by default.
    Rec709,
}

/// The frame a [`PortType::Normal`] is expressed in.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum NormalFrame {
    /// The field domain's frame: `+X` along the domain x axis, `+Y` along the
    /// domain y axis, `+Z` out of the surface; right-handed. This is the
    /// frame of `dapple_raster::HeightToNormal`. Texture conventions such as
    /// glTF's `+Y` toward the image top are applied when packing.
    Domain,
}

/// How [`Op::BlendNormals`](crate::program::Op::BlendNormals) combines a
/// detail normal with a base normal.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum NormalBlend {
    /// Reoriented normal mapping (Barré-Brisebois and Hill, 2012): the detail
    /// is rotated onto the base, preserving both.
    Reoriented,
    /// "Unreal Developer Network" blending: the detail's slopes are added to
    /// the base's. Cheaper, flatter at strong angles.
    Udn,
}

/// The type of a program node's value.
///
/// Types are derived when a node is added, so misuse is a build error rather
/// than wrong-looking output: normals blend only with
/// [`Op::BlendNormals`](crate::program::Op::BlendNormals), identifiers are
/// never interpolated, and directional values are never rotated by a domain
/// transform they cannot follow.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum PortType {
    /// A real number.
    Scalar,
    /// Coverage in `[0, 1]`. A mask is accepted wherever a scalar is; only
    /// operations that keep values in `[0, 1]` produce one.
    Mask,
    /// An integer region or cell identifier. It can move with the domain but
    /// never blends.
    Id,
    /// Two components.
    Vector2,
    /// Three components.
    Vector3,
    /// A linear color with declared primaries.
    Color(Primaries),
    /// A unit shading normal in a declared frame.
    Normal(NormalFrame),
}

impl PortType {
    /// True for [`Self::Scalar`] and [`Self::Mask`], which scalar operations
    /// accept.
    #[must_use]
    pub const fn is_scalar(self) -> bool {
        matches!(self, Self::Scalar | Self::Mask)
    }

    /// Components per value: 1 for scalars, masks and identifiers.
    #[must_use]
    pub const fn components(self) -> usize {
        match self {
            Self::Scalar | Self::Mask | Self::Id => 1,
            Self::Vector2 => 2,
            Self::Vector3 | Self::Color(_) | Self::Normal(_) => 3,
        }
    }
}

impl fmt::Display for PortType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Scalar => "scalar",
            Self::Mask => "mask",
            Self::Id => "id",
            Self::Vector2 => "vector2",
            Self::Vector3 => "vector3",
            Self::Color(Primaries::Rec709) => "color (Rec. 709, linear)",
            Self::Normal(NormalFrame::Domain) => "normal (domain frame)",
        };
        f.write_str(name)
    }
}

/// A program node's value at one point.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Value {
    /// A [`PortType::Scalar`] or [`PortType::Mask`] value.
    Scalar(f32),
    /// A [`PortType::Id`] value.
    Id(u32),
    /// A [`PortType::Vector2`] value.
    Vector2(Vec2),
    /// A [`PortType::Vector3`], [`PortType::Color`] or [`PortType::Normal`]
    /// value.
    Vector3(Vec3),
}

impl Value {
    /// The scalar, for scalar and mask values.
    #[must_use]
    pub const fn scalar(self) -> Option<f32> {
        match self {
            Self::Scalar(v) => Some(v),
            _ => None,
        }
    }

    /// Component `index`: the scalar or identifier itself for index 0, or a
    /// vector component.
    #[must_use]
    pub fn component(self, index: usize) -> Option<f32> {
        match self {
            Self::Scalar(v) if index == 0 => Some(v),
            #[expect(
                clippy::cast_precision_loss,
                reason = "identifiers are reported as floats only for previews"
            )]
            Self::Id(v) if index == 0 => Some(v as f32),
            Self::Vector2(v) if index < 2 => Some(v[index]),
            Self::Vector3(v) if index < 3 => Some(v[index]),
            _ => None,
        }
    }
}
