// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Material values: OpenPBR parameters and auxiliary channels on one grid.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use dapple_field::hash::hash;
use dapple_field::scoped::{ContractError, value_fits, value_words};
use dapple_field::{Edge, NormalFrame, PortType, Primaries, Value};
use dapple_raster::typed::{TypedError, TypedRaster};
use dapple_raster::{Raster, RasterError, Realization};
use glam::{Vec2, Vec3};
use openpbr::color::LinearSrgb;
use openpbr::{Kind, Param, ParamDefault, Parameters};

/// The texel grid every channel of a material shares.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Grid {
    /// Texels per row.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// Domain position of texel `(0, 0)`'s minimum corner.
    pub origin: Vec2,
    /// Texel size in domain units (meters for dapple's materials).
    pub texel: Vec2,
    /// The edge policy: wrapping for one period of a periodic domain.
    pub edge: Edge,
}

impl Grid {
    /// The grid of `realization`.
    #[must_use]
    pub fn of(realization: &Realization) -> Self {
        Self {
            width: realization.width(),
            height: realization.height(),
            origin: realization.origin(),
            texel: realization.texel(),
            edge: realization.edge(),
        }
    }

    /// The grid of `raster`.
    #[must_use]
    pub fn of_raster<T: Copy>(raster: &Raster<T>) -> Self {
        Self {
            width: raster.width(),
            height: raster.height(),
            origin: raster.origin(),
            texel: raster.texel(),
            edge: raster.edge(),
        }
    }

    /// The grid of a typed raster.
    #[must_use]
    pub fn of_typed(raster: &TypedRaster) -> Self {
        Self {
            width: raster.width(),
            height: raster.height(),
            origin: raster.origin(),
            texel: raster.texel(),
            edge: raster.edge(),
        }
    }

    /// Texels on the grid.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Whether the grid has no texels; never for a valid grid.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The domain position of texel `i`'s center, row-major.
    #[must_use]
    pub fn center(&self, i: usize) -> Vec2 {
        let w = self.width as usize;
        #[expect(clippy::cast_precision_loss, reason = "texel indices are small")]
        let (x, y) = ((i % w) as f32, (i / w) as f32);
        self.origin + self.texel * Vec2::new(x + 0.5, y + 0.5)
    }

    /// `values` on this grid.
    ///
    /// # Errors
    ///
    /// [`RasterError::LengthMismatch`] when `values` does not fill it.
    pub fn raster<T: Copy>(&self, values: Vec<T>) -> Result<Raster<T>, RasterError> {
        Raster::from_values(
            self.width,
            self.height,
            self.origin,
            self.texel,
            self.edge,
            values,
        )
    }

    /// A typed raster of `port` from one value per texel.
    ///
    /// # Errors
    ///
    /// [`TypedError`] when `values` does not fill the grid.
    pub fn typed(
        &self,
        port: PortType,
        values: impl IntoIterator<Item = Value>,
    ) -> Result<TypedRaster, TypedError> {
        let shape = self.raster(vec![(); self.len()])?;
        TypedRaster::from_values(port, &shape, values)
    }

    /// Whether `raster` lies on this grid.
    #[must_use]
    pub fn holds(&self, raster: &TypedRaster) -> bool {
        Self::of_typed(raster) == *self
    }

    /// Whether a scalar `raster` lies on this grid.
    #[must_use]
    pub fn holds_raster<T: Copy>(&self, raster: &Raster<T>) -> bool {
        Self::of_raster(raster) == *self
    }
}

/// An auxiliary channel: data consumers need that is not an OpenPBR
/// parameter, kept apart with its own declared meaning.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Aux {
    /// Displacement above the surface's datum, in meters (a scalar).
    Height,
    /// Ambient occlusion, 1 where fully open (a mask).
    Occlusion,
    /// Which element or region owns a texel (an identifier). A summary
    /// label wherever several contribute.
    Region,
    /// Which surface material a texel shows (an identifier), such as glaze,
    /// exposed body or mortar: separate from [`Aux::Region`], since the
    /// glaze and the body of one brick are one element and two materials.
    Surface,
}

impl Aux {
    /// Every auxiliary channel.
    pub const ALL: [Self; 4] = [Self::Height, Self::Occlusion, Self::Region, Self::Surface];

    /// The channel's name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Height => "height",
            Self::Occlusion => "occlusion",
            Self::Region => "region",
            Self::Surface => "surface",
        }
    }

    /// The type of its values.
    #[must_use]
    pub const fn port(self) -> PortType {
        match self {
            Self::Height => PortType::Scalar,
            Self::Occlusion => PortType::Mask,
            Self::Region | Self::Surface => PortType::Id,
        }
    }

    /// Its value where it is not bound: flat, open, unowned.
    #[must_use]
    pub const fn default_value(self) -> Value {
        match self {
            Self::Height => Value::Scalar(0.0),
            Self::Occlusion => Value::Scalar(1.0),
            Self::Region | Self::Surface => Value::Id(0),
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

/// A channel of a material: an OpenPBR parameter or an auxiliary channel.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ChannelId {
    /// An OpenPBR parameter.
    Param(Param),
    /// An auxiliary channel.
    Aux(Aux),
}

impl ChannelId {
    /// The channel's name: the specification's identifier for parameters.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Param(p) => p.identifier(),
            Self::Aux(a) => a.name(),
        }
    }

    /// The type of its values.
    #[must_use]
    pub fn port(self) -> PortType {
        match self {
            Self::Param(p) => param_port(p),
            Self::Aux(a) => a.port(),
        }
    }

    /// Its value where it is not bound.
    #[must_use]
    pub fn default_value(self) -> Value {
        match self {
            Self::Param(p) => param_default(p),
            Self::Aux(a) => a.default_value(),
        }
    }
}

impl fmt::Display for ChannelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The type dapple gives parameter `p`'s values: scalars for floats and
/// booleans, linear Rec. 709 colors for colors, three-vectors for
/// per-channel factors and tangents, and domain-frame normals for normals.
#[must_use]
pub fn param_port(p: Param) -> PortType {
    let info = p.info();
    match info.kind {
        Kind::Color3 if info.color => PortType::Color(Primaries::Rec709),
        Kind::Vector3 if matches!(info.default, ParamDefault::UnperturbedNormal) => {
            PortType::Normal(NormalFrame::Domain)
        }
        Kind::Color3 | Kind::Vector3 => PortType::Vector3,
        _ => PortType::Scalar,
    }
}

/// The specification's default for parameter `p` as a dapple value: the
/// unperturbed normal is `+Z` and the unperturbed tangent `+X` of the
/// domain frame; booleans are 0 or 1.
#[must_use]
pub fn param_default(p: Param) -> Value {
    match p.info().default {
        ParamDefault::Float(v) => Value::Scalar(v),
        ParamDefault::Boolean(b) => Value::Scalar(if b { 1.0 } else { 0.0 }),
        ParamDefault::Color3(c) => Value::Vector3(Vec3::from_array(c)),
        ParamDefault::UnperturbedNormal => Value::Vector3(Vec3::Z),
        ParamDefault::UnperturbedTangent => Value::Vector3(Vec3::X),
        _ => Value::Scalar(0.0),
    }
}

/// Where a channel's values come from.
#[derive(Clone, Debug, PartialEq)]
pub enum Channel {
    /// One value everywhere.
    Constant(Value),
    /// One value per texel of the material's grid, typed.
    Map(TypedRaster),
}

impl Channel {
    /// The value at texel `i`, row-major.
    #[must_use]
    pub fn value(&self, i: usize) -> Value {
        match self {
            Self::Constant(v) => *v,
            Self::Map(r) => r.value(i),
        }
    }

    /// The constant, when the channel is one.
    #[must_use]
    pub const fn constant(&self) -> Option<Value> {
        match self {
            Self::Constant(v) => Some(*v),
            Self::Map(_) => None,
        }
    }

    fn words(&self, w: &mut Vec<u64>) {
        match self {
            Self::Constant(v) => {
                w.push(1);
                value_words(*v, w);
            }
            Self::Map(r) => w.extend([2, r.digest()]),
        }
    }
}

/// A material-level failure.
#[derive(Clone, Debug, PartialEq)]
pub enum MaterialError {
    /// A map or mask is not on the material's grid, or two materials are
    /// not on one grid.
    GridMismatch,
    /// A value or map does not have the channel's type.
    TypeMismatch(ChannelId),
    /// A parameter that cannot vary over a surface was given a map.
    Uniform(Param),
    /// Two materials combined disagree on a uniform parameter.
    UniformConflict(Param),
    /// An operation parameter is out of range.
    InvalidParameter(&'static str),
    /// A raster operation failed.
    Raster(RasterError),
    /// Building a typed raster failed.
    Typed(TypedError),
    /// A program does not fit its bindings.
    Contract(ContractError),
}

impl fmt::Display for MaterialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GridMismatch => f.write_str("channels are not on one grid"),
            Self::TypeMismatch(c) => write!(f, "{c}: the value does not have the channel's type"),
            Self::Uniform(p) => write!(f, "{p} cannot vary over a surface"),
            Self::UniformConflict(p) => write!(f, "the materials disagree on uniform {p}"),
            Self::InvalidParameter(name) => write!(f, "{name} is out of range"),
            Self::Raster(e) => e.fmt(f),
            Self::Typed(e) => e.fmt(f),
            Self::Contract(e) => e.fmt(f),
        }
    }
}

impl core::error::Error for MaterialError {}

impl From<RasterError> for MaterialError {
    fn from(e: RasterError) -> Self {
        Self::Raster(e)
    }
}

impl From<TypedError> for MaterialError {
    fn from(e: TypedError) -> Self {
        Self::Typed(e)
    }
}

impl From<ContractError> for MaterialError {
    fn from(e: ContractError) -> Self {
        Self::Contract(e)
    }
}

/// A material: every OpenPBR parameter bound to a constant or a typed map,
/// and auxiliary channels kept separate, all on one [`Grid`].
///
/// Unbound parameters take the specification's defaults, and unbound
/// auxiliary channels their [`Aux::default_value`]. Because every map shares
/// the grid, an operation that decides something per texel (a selection
/// weight, a coordinate transform) decides it once for every channel.
#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    grid: Grid,
    params: Vec<Option<Channel>>,
    aux: Vec<Option<Channel>>,
}

fn param_index(p: Param) -> usize {
    Param::ALL
        .iter()
        .position(|q| *q == p)
        .expect("every parameter is in ALL")
}

impl Material {
    /// A material at the specification's defaults on `grid`.
    #[must_use]
    pub fn new(grid: Grid) -> Self {
        Self {
            grid,
            params: vec![None; Param::COUNT],
            aux: vec![None; Aux::ALL.len()],
        }
    }

    /// Every parameter of `parameters` as a constant on `grid`; colors are
    /// linear Rec. 709, dapple's working space. Parameters equal to the
    /// specification's defaults stay unbound.
    #[must_use]
    pub fn from_parameters(grid: Grid, parameters: &Parameters<LinearSrgb>) -> Self {
        let mut m = Self::new(grid);
        for p in Param::ALL {
            let value = match parameters.get(p) {
                Some(openpbr::Value::Float(v)) => Value::Scalar(v),
                Some(openpbr::Value::Boolean(b)) => Value::Scalar(if b { 1.0 } else { 0.0 }),
                Some(openpbr::Value::Color(c)) => Value::Vector3(Vec3::from_array(c.components)),
                Some(openpbr::Value::Channels(c)) => Value::Vector3(Vec3::from_array(c)),
                _ => continue,
            };
            if value != param_default(p) {
                m.params[param_index(p)] = Some(Channel::Constant(value));
            }
        }
        m
    }

    /// The grid.
    #[must_use]
    pub const fn grid(&self) -> Grid {
        self.grid
    }

    /// Parameter `p`'s binding, if bound.
    #[must_use]
    pub fn param(&self, p: Param) -> Option<&Channel> {
        self.params[param_index(p)].as_ref()
    }

    /// Auxiliary channel `a`'s binding, if bound.
    #[must_use]
    pub fn aux(&self, a: Aux) -> Option<&Channel> {
        self.aux[a.index()].as_ref()
    }

    /// Channel `c`'s binding, if bound.
    #[must_use]
    pub fn channel(&self, c: ChannelId) -> Option<&Channel> {
        match c {
            ChannelId::Param(p) => self.param(p),
            ChannelId::Aux(a) => self.aux(a),
        }
    }

    /// Channel `c`'s value at texel `i`, or its default when unbound.
    #[must_use]
    pub fn value(&self, c: ChannelId, i: usize) -> Value {
        self.channel(c)
            .map_or_else(|| c.default_value(), |ch| ch.value(i))
    }

    /// Channel `c`'s constant value when it has one everywhere (unbound
    /// channels have their default).
    #[must_use]
    pub fn constant(&self, c: ChannelId) -> Option<Value> {
        match self.channel(c) {
            None => Some(c.default_value()),
            Some(ch) => ch.constant(),
        }
    }

    /// Binds channel `c`.
    ///
    /// # Errors
    ///
    /// [`MaterialError::TypeMismatch`] for a value or map of another type,
    /// [`MaterialError::GridMismatch`] for a map off the grid, and
    /// [`MaterialError::Uniform`] for a map of a uniform parameter.
    pub fn set(&mut self, c: ChannelId, channel: Channel) -> Result<(), MaterialError> {
        let port = c.port();
        match &channel {
            Channel::Constant(v) => {
                if !value_fits(*v, port) {
                    return Err(MaterialError::TypeMismatch(c));
                }
            }
            Channel::Map(r) => {
                if r.port() != port
                    && !(port == PortType::Scalar && r.port() == PortType::Mask)
                    && !(port == PortType::Mask && r.port() == PortType::Scalar)
                {
                    return Err(MaterialError::TypeMismatch(c));
                }
                if !self.grid.holds(r) {
                    return Err(MaterialError::GridMismatch);
                }
                if let ChannelId::Param(p) = c
                    && p.info().uniform
                {
                    return Err(MaterialError::Uniform(p));
                }
            }
        }
        match c {
            ChannelId::Param(p) => self.params[param_index(p)] = Some(channel),
            ChannelId::Aux(a) => self.aux[a.index()] = Some(channel),
        }
        Ok(())
    }

    /// Binds parameter `p`; see [`Self::set`].
    ///
    /// # Errors
    ///
    /// As [`Self::set`].
    pub fn set_param(&mut self, p: Param, channel: Channel) -> Result<(), MaterialError> {
        self.set(ChannelId::Param(p), channel)
    }

    /// Binds auxiliary channel `a`; see [`Self::set`].
    ///
    /// # Errors
    ///
    /// As [`Self::set`].
    pub fn set_aux(&mut self, a: Aux, channel: Channel) -> Result<(), MaterialError> {
        self.set(ChannelId::Aux(a), channel)
    }

    /// Unbinds channel `c`, returning it to its default.
    pub fn clear(&mut self, c: ChannelId) {
        match c {
            ChannelId::Param(p) => self.params[param_index(p)] = None,
            ChannelId::Aux(a) => self.aux[a.index()] = None,
        }
    }

    /// Every bound channel, parameters in specification order, then
    /// auxiliary channels. Operations that must leave no channel behind
    /// iterate this.
    pub fn bound(&self) -> impl Iterator<Item = (ChannelId, &Channel)> + '_ {
        let params = Param::ALL
            .iter()
            .zip(&self.params)
            .filter_map(|(p, c)| c.as_ref().map(|c| (ChannelId::Param(*p), c)));
        let aux = Aux::ALL
            .iter()
            .zip(&self.aux)
            .filter_map(|(a, c)| c.as_ref().map(|c| (ChannelId::Aux(*a), c)));
        params.chain(aux)
    }

    /// The height in meters as a raster on the grid: zeros when unbound.
    ///
    /// # Errors
    ///
    /// Never for a valid material.
    pub fn height(&self) -> Result<Raster, MaterialError> {
        let c = ChannelId::Aux(Aux::Height);
        Ok(self.grid.raster(
            (0..self.grid.len())
                .map(|i| self.value(c, i).component(0).unwrap_or(0.0))
                .collect(),
        )?)
    }

    /// A scalar channel as a raster on the grid.
    ///
    /// # Errors
    ///
    /// Never for a valid material.
    pub fn scalar(&self, c: ChannelId) -> Result<Raster, MaterialError> {
        Ok(self.grid.raster(
            (0..self.grid.len())
                .map(|i| self.value(c, i).component(0).unwrap_or(0.0))
                .collect(),
        )?)
    }

    /// A hash of the grid and every bound channel's values.
    #[must_use]
    pub fn digest(&self) -> u64 {
        let g = &self.grid;
        let mut w = vec![
            u64::from(g.width),
            u64::from(g.height),
            u64::from(g.origin.x.to_bits()),
            u64::from(g.origin.y.to_bits()),
            u64::from(g.texel.x.to_bits()),
            u64::from(g.texel.y.to_bits()),
        ];
        for (c, ch) in self.bound() {
            w.push(match c {
                ChannelId::Param(p) => param_index(p) as u64,
                ChannelId::Aux(a) => 1000 + a.index() as u64,
            });
            ch.words(&mut w);
        }
        hash(0x6d61_7465_7269_616c, &w) // "material"
    }

    /// Texels outside each parameter's allowed range (or not finite), for
    /// parameters that have any.
    #[must_use]
    pub fn range_violations(&self) -> Vec<(Param, u64)> {
        let mut out = Vec::new();
        for p in Param::ALL {
            let (Some(range), Some(ch)) = (p.info().range, self.param(p)) else {
                continue;
            };
            let count = match ch {
                Channel::Constant(v) => {
                    let bad = (0..3)
                        .filter_map(|k| v.component(k))
                        .any(|x| !(x.is_finite() && range.contains(x)));
                    u64::from(bad) * self.grid.len() as u64
                }
                Channel::Map(r) => (0..r.len())
                    .filter(|&i| {
                        let v = r.value(i);
                        (0..3)
                            .filter_map(|k| v.component(k))
                            .any(|x| !(x.is_finite() && range.contains(x)))
                    })
                    .count() as u64,
            };
            if count > 0 {
                out.push((p, count));
            }
        }
        out
    }
}
