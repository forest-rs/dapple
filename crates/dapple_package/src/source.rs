// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The editable source form: what a package file holds, as plain data.
//!
//! Every type here is a serialized shape, with no engine behind it: names
//! are strings, modules are `name@version` references, and nothing is
//! resolved. Maps are ordered by key, so the same package always writes
//! the same bytes, and a [`Package::fingerprint`] is stable.

use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use dapple_field::hash::hash;
use dapple_field::{NormalFrame, PortType, Primaries};
use dapple_material::module::{
    Bind, InputDecl, InputKind, Interface, ModuleId, OutputDecl, OutputKind, ParamDecl, ParamKind,
    ParamValue, Unit,
};
use dapple_material::resource::ResourceRequest;
use dapple_material::{Aux, ChannelId, Param, Tiling};
use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::PackageError;

/// The `format` of a package document.
pub const PACKAGE_FORMAT: &str = "dapple.package";
/// The `format` of a preset document.
pub const PRESETS_FORMAT: &str = "dapple.presets";
/// The one format version this crate reads and writes. A document of any
/// other version is refused rather than guessed at.
pub const FORMAT_VERSION: u32 = 1;

/// A module named by identity: `name` at `version`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModuleRef {
    /// The module's stable name.
    pub name: String,
    /// Its interface and behavior version.
    pub version: u32,
}

impl ModuleRef {
    /// The reference to `id`.
    #[must_use]
    pub fn of(id: &ModuleId) -> Self {
        Self {
            name: id.name.to_string(),
            version: id.version,
        }
    }

    /// The identity it names.
    #[must_use]
    pub fn id(&self) -> ModuleId {
        ModuleId {
            name: Cow::Owned(self.name.clone()),
            version: self.version,
        }
    }
}

/// An engine feature a package expects, at a version: see
/// [`crate::Registry`].
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capability {
    /// The feature's name, such as `"dapple.graph"`.
    pub name: String,
    /// The least version that serves.
    pub version: u32,
}

impl Capability {
    /// `name` at `version`.
    #[must_use]
    pub fn new(name: &str, version: u32) -> Self {
        Self {
            name: name.into(),
            version,
        }
    }
}

/// A parameter's unit.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnitSource {
    /// Dimensionless.
    None,
    /// Meters.
    Meters,
    /// A fraction in `[0, 1]`.
    Fraction,
    /// Radians.
    Radians,
}

impl From<Unit> for UnitSource {
    fn from(u: Unit) -> Self {
        match u {
            Unit::None => Self::None,
            Unit::Meters => Self::Meters,
            Unit::Fraction => Self::Fraction,
            Unit::Radians => Self::Radians,
        }
    }
}

impl From<UnitSource> for Unit {
    fn from(u: UnitSource) -> Self {
        match u {
            UnitSource::None => Self::None,
            UnitSource::Meters => Self::Meters,
            UnitSource::Fraction => Self::Fraction,
            UnitSource::Radians => Self::Radians,
        }
    }
}

/// A parameter's type, range and default.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ParamType {
    /// A real number.
    Scalar {
        /// The unit.
        unit: UnitSource,
        /// The allowed values, inclusive.
        range: [f32; 2],
        /// The default.
        default: f32,
    },
    /// A linear Rec. 709 color.
    Color {
        /// The default.
        default: [f32; 3],
    },
    /// A count.
    Integer {
        /// The allowed values, inclusive.
        range: [u32; 2],
        /// The default.
        default: u32,
    },
    /// A seed, mixed with the instance path.
    Seed {
        /// The default.
        default: u64,
    },
    /// A switch.
    Flag {
        /// The default.
        default: bool,
    },
}

impl ParamType {
    fn of(kind: ParamKind, default: ParamValue) -> Option<Self> {
        Some(match (kind, default) {
            (ParamKind::Scalar { unit, range }, ParamValue::Scalar(d)) => Self::Scalar {
                unit: unit.into(),
                range,
                default: d,
            },
            (ParamKind::Color, ParamValue::Color(d)) => Self::Color {
                default: d.to_array(),
            },
            (ParamKind::Integer { range }, ParamValue::Integer(d)) => {
                Self::Integer { range, default: d }
            }
            (ParamKind::Seed, ParamValue::Seed(d)) => Self::Seed { default: d },
            (ParamKind::Flag, ParamValue::Flag(d)) => Self::Flag { default: d },
            _ => return None,
        })
    }

    /// The kind and default it declares.
    #[must_use]
    pub fn kind(&self) -> (ParamKind, ParamValue) {
        match *self {
            Self::Scalar {
                unit,
                range,
                default,
            } => (
                ParamKind::Scalar {
                    unit: unit.into(),
                    range,
                },
                ParamValue::Scalar(default),
            ),
            Self::Color { default } => (ParamKind::Color, ParamValue::Color(Vec3::from(default))),
            Self::Integer { range, default } => {
                (ParamKind::Integer { range }, ParamValue::Integer(default))
            }
            Self::Seed { default } => (ParamKind::Seed, ParamValue::Seed(default)),
            Self::Flag { default } => (ParamKind::Flag, ParamValue::Flag(default)),
        }
    }
}

/// A declared parameter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParamSource {
    /// Its name.
    pub name: String,
    /// What it means.
    #[serde(default)]
    pub doc: String,
    /// Its type, range and default.
    #[serde(rename = "type")]
    pub ty: ParamType,
}

/// A parameter value.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueSource {
    /// A real number.
    Scalar(f32),
    /// A linear Rec. 709 color.
    Color([f32; 3]),
    /// A count.
    Integer(u32),
    /// A seed.
    Seed(u64),
    /// A switch.
    Flag(bool),
}

impl From<ParamValue> for ValueSource {
    fn from(v: ParamValue) -> Self {
        match v {
            ParamValue::Scalar(v) => Self::Scalar(v),
            ParamValue::Color(v) => Self::Color(v.to_array()),
            ParamValue::Integer(v) => Self::Integer(v),
            ParamValue::Seed(v) => Self::Seed(v),
            ParamValue::Flag(v) => Self::Flag(v),
        }
    }
}

impl From<ValueSource> for ParamValue {
    fn from(v: ValueSource) -> Self {
        match v {
            ValueSource::Scalar(v) => Self::Scalar(v),
            ValueSource::Color(v) => Self::Color(Vec3::from(v)),
            ValueSource::Integer(v) => Self::Integer(v),
            ValueSource::Seed(v) => Self::Seed(v),
            ValueSource::Flag(v) => Self::Flag(v),
        }
    }
}

/// A port type, by name.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortSource {
    /// [`PortType::Scalar`].
    Scalar,
    /// [`PortType::Mask`].
    Mask,
    /// [`PortType::Id`].
    Id,
    /// [`PortType::Vector2`].
    Vector2,
    /// [`PortType::Vector3`].
    Vector3,
    /// A linear color with Rec. 709 primaries.
    ColorRec709,
    /// A normal in the domain frame.
    NormalDomain,
    /// [`PortType::Direction`].
    Direction,
}

impl From<PortType> for PortSource {
    fn from(p: PortType) -> Self {
        match p {
            PortType::Scalar => Self::Scalar,
            PortType::Mask => Self::Mask,
            PortType::Id => Self::Id,
            PortType::Vector2 => Self::Vector2,
            PortType::Vector3 => Self::Vector3,
            PortType::Color(Primaries::Rec709) => Self::ColorRec709,
            PortType::Normal(NormalFrame::Domain) => Self::NormalDomain,
            PortType::Direction => Self::Direction,
        }
    }
}

impl From<PortSource> for PortType {
    fn from(p: PortSource) -> Self {
        match p {
            PortSource::Scalar => Self::Scalar,
            PortSource::Mask => Self::Mask,
            PortSource::Id => Self::Id,
            PortSource::Vector2 => Self::Vector2,
            PortSource::Vector3 => Self::Vector3,
            PortSource::ColorRec709 => Self::Color(Primaries::Rec709),
            PortSource::NormalDomain => Self::Normal(NormalFrame::Domain),
            PortSource::Direction => Self::Direction,
        }
    }
}

/// What an input holds.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum InputType {
    /// A material.
    Material,
    /// A typed map.
    Map(PortSource),
    /// A host-resolved resource: a requirement the host must meet.
    Resource {
        /// The semantic type of its texels.
        port: PortSource,
        /// Whether it must tile.
        periodic: bool,
    },
}

impl From<InputKind> for InputType {
    fn from(k: InputKind) -> Self {
        match k {
            InputKind::Material => Self::Material,
            InputKind::Map(p) => Self::Map(p.into()),
            InputKind::Resource(r) => Self::Resource {
                port: r.port.into(),
                periodic: r.periodic,
            },
        }
    }
}

impl From<InputType> for InputKind {
    fn from(k: InputType) -> Self {
        match k {
            InputType::Material => Self::Material,
            InputType::Map(p) => Self::Map(p.into()),
            InputType::Resource { port, periodic } => Self::Resource(ResourceRequest {
                port: port.into(),
                periodic,
            }),
        }
    }
}

/// A declared input.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputSource {
    /// Its name.
    pub name: String,
    /// What it means.
    #[serde(default)]
    pub doc: String,
    /// Whether an instance must bind it.
    #[serde(default)]
    pub required: bool,
    /// What it holds.
    #[serde(rename = "type")]
    pub ty: InputType,
}

/// What an output holds.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum OutputType {
    /// A material.
    Material,
    /// A typed map.
    Map(PortSource),
}

impl From<OutputKind> for OutputType {
    fn from(k: OutputKind) -> Self {
        match k {
            OutputKind::Material => Self::Material,
            OutputKind::Map(p) => Self::Map(p.into()),
        }
    }
}

impl From<OutputType> for OutputKind {
    fn from(k: OutputType) -> Self {
        match k {
            OutputType::Material => Self::Material,
            OutputType::Map(p) => Self::Map(p.into()),
        }
    }
}

/// The axes a material output promises to tile along.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TilingSource {
    /// Both axes.
    Both,
    /// Along x only.
    X,
    /// Along y only.
    Y,
}

impl From<TilingSource> for Tiling {
    fn from(t: TilingSource) -> Self {
        match t {
            TilingSource::Both => Self::BOTH,
            TilingSource::X => Self::X,
            TilingSource::Y => Self { x: false, y: true },
        }
    }
}

/// What an output means, beyond its type, checked on every instance: the
/// channels a material output binds, and the axes it tiles along.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Semantics {
    /// Channels a material output binds, by name: OpenPBR identifiers
    /// (`"base_color"`) or auxiliary channels (`"height"`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub channels: Vec<String>,
    /// The axes a material output promises to tile along, when the grid
    /// wraps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tiling: Option<TilingSource>,
}

impl Semantics {
    /// The declared channels, resolved.
    ///
    /// # Errors
    ///
    /// [`PackageError::UnknownChannel`] for a name that is no channel.
    pub fn channel_ids(&self) -> Result<Vec<ChannelId>, PackageError> {
        self.channels
            .iter()
            .map(|n| {
                Param::from_identifier(n)
                    .map(ChannelId::Param)
                    .or_else(|| {
                        Aux::ALL
                            .into_iter()
                            .find(|a| a.name() == n)
                            .map(ChannelId::Aux)
                    })
                    .ok_or_else(|| PackageError::UnknownChannel(n.clone()))
            })
            .collect()
    }

    fn is_empty(&self) -> bool {
        self.channels.is_empty() && self.tiling.is_none()
    }
}

/// A declared output.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputSource {
    /// Its name.
    pub name: String,
    /// What it means.
    #[serde(default)]
    pub doc: String,
    /// What it holds.
    #[serde(rename = "type")]
    pub ty: OutputType,
    /// What it promises.
    #[serde(default, skip_serializing_if = "Semantics::is_empty")]
    pub semantics: Semantics,
}

/// A module interface as data: the serialized form of
/// [`dapple_material::module::Interface`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InterfaceSource {
    /// The module's identity.
    pub module: ModuleRef,
    /// What it makes.
    #[serde(default)]
    pub doc: String,
    /// Parameters.
    #[serde(default)]
    pub params: Vec<ParamSource>,
    /// Inputs.
    #[serde(default)]
    pub inputs: Vec<InputSource>,
    /// Outputs.
    pub outputs: Vec<OutputSource>,
}

impl InterfaceSource {
    /// The source form of `interface`, with no output semantics.
    ///
    /// # Errors
    ///
    /// [`PackageError::InvalidParam`] for a parameter whose default is not
    /// of its kind.
    pub fn of(interface: &Interface) -> Result<Self, PackageError> {
        Ok(Self {
            module: ModuleRef::of(&interface.id),
            doc: interface.doc.to_string(),
            params: interface
                .params
                .iter()
                .map(|p| {
                    Ok(ParamSource {
                        name: p.name.to_string(),
                        doc: p.doc.to_string(),
                        ty: ParamType::of(p.kind, p.default).ok_or_else(|| {
                            PackageError::InvalidParam {
                                name: p.name.to_string(),
                                reason: "wrong kind",
                            }
                        })?,
                    })
                })
                .collect::<Result<_, PackageError>>()?,
            inputs: interface
                .inputs
                .iter()
                .map(|i| InputSource {
                    name: i.name.to_string(),
                    doc: i.doc.to_string(),
                    required: i.required,
                    ty: i.kind.into(),
                })
                .collect(),
            outputs: interface
                .outputs
                .iter()
                .map(|o| OutputSource {
                    name: o.name.to_string(),
                    doc: o.doc.to_string(),
                    ty: o.kind.into(),
                    semantics: Semantics::default(),
                })
                .collect(),
        })
    }

    /// The interface it describes, checked: names unique, defaults of
    /// their kind and in range, channels known.
    ///
    /// # Errors
    ///
    /// [`PackageError::Duplicate`], [`PackageError::InvalidParam`] or
    /// [`PackageError::UnknownChannel`].
    pub fn to_interface(&self) -> Result<Interface, PackageError> {
        unique(self.params.iter().map(|p| p.name.as_str()))?;
        unique(self.inputs.iter().map(|p| p.name.as_str()))?;
        unique(self.outputs.iter().map(|p| p.name.as_str()))?;
        let mut params = Vec::with_capacity(self.params.len());
        for p in &self.params {
            let (kind, default) = p.ty.kind();
            default
                .fits(kind)
                .map_err(|reason| PackageError::InvalidParam {
                    name: p.name.clone(),
                    reason,
                })?;
            params.push(ParamDecl {
                name: Cow::Owned(p.name.clone()),
                kind,
                default,
                doc: Cow::Owned(p.doc.clone()),
            });
        }
        for o in &self.outputs {
            o.semantics.channel_ids()?;
        }
        Ok(Interface {
            id: self.module.id(),
            doc: Cow::Owned(self.doc.clone()),
            params,
            inputs: self
                .inputs
                .iter()
                .map(|i| InputDecl {
                    name: Cow::Owned(i.name.clone()),
                    kind: i.ty.into(),
                    required: i.required,
                    doc: Cow::Owned(i.doc.clone()),
                })
                .collect(),
            outputs: self
                .outputs
                .iter()
                .map(|o| OutputDecl {
                    name: Cow::Owned(o.name.clone()),
                    kind: o.ty.into(),
                    doc: Cow::Owned(o.doc.clone()),
                })
                .collect(),
        })
    }
}

/// Fails on the first name that repeats.
pub(crate) fn unique<'a>(names: impl Iterator<Item = &'a str>) -> Result<(), PackageError> {
    let mut seen: Vec<&str> = Vec::new();
    for n in names {
        if seen.contains(&n) {
            return Err(PackageError::Duplicate(n.into()));
        }
        seen.push(n);
    }
    Ok(())
}

/// A named set of parameter values: a starting point an author saved,
/// such as the result of a fit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Preset {
    /// Its name, unique among its module's presets.
    pub name: String,
    /// What it is for, or where it came from.
    #[serde(default)]
    pub doc: String,
    /// Values by parameter name; the rest keep their defaults.
    pub values: BTreeMap<String, ValueSource>,
}

impl Preset {
    /// An empty preset.
    #[must_use]
    pub fn new(name: &str, doc: &str) -> Self {
        Self {
            name: name.into(),
            doc: doc.into(),
            values: BTreeMap::new(),
        }
    }

    /// Sets parameter `name` to `value`.
    #[must_use]
    pub fn with(mut self, name: &str, value: impl Into<ValueSource>) -> Self {
        self.values.insert(name.into(), value.into());
        self
    }

    /// Sets scalar parameter `name`: for fitted values, which search in
    /// `f64`.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "parameters are stored as f32"
    )]
    pub fn scalar(self, name: &str, value: f64) -> Self {
        self.with(name, ValueSource::Scalar(value as f32))
    }

    /// Checks every value against `interface`: each names a parameter and
    /// fits its kind and range.
    ///
    /// # Errors
    ///
    /// [`PackageError::InvalidParam`].
    pub fn check(&self, interface: &Interface) -> Result<(), PackageError> {
        for (name, value) in &self.values {
            let decl = interface
                .params
                .iter()
                .find(|p| p.name == name.as_str())
                .ok_or_else(|| PackageError::InvalidParam {
                    name: name.clone(),
                    reason: "no such parameter",
                })?;
            ParamValue::from(*value).fits(decl.kind).map_err(|reason| {
                PackageError::InvalidParam {
                    name: name.clone(),
                    reason,
                }
            })?;
        }
        Ok(())
    }

    /// The bindings it makes.
    #[must_use]
    pub fn bind(&self) -> Bind {
        self.values
            .iter()
            .fold(Bind::new(), |b, (n, v)| b.param(n, (*v).into()))
    }
}

/// Presets for any module, native or packaged, as a document of their own.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PresetSet {
    /// [`PRESETS_FORMAT`].
    pub format: String,
    /// [`FORMAT_VERSION`].
    pub version: u32,
    /// The module they are for.
    pub module: ModuleRef,
    /// The presets.
    pub presets: Vec<Preset>,
}

impl PresetSet {
    /// Presets for `module`.
    #[must_use]
    pub fn new(module: &ModuleId, presets: Vec<Preset>) -> Self {
        Self {
            format: PRESETS_FORMAT.into(),
            version: FORMAT_VERSION,
            module: ModuleRef::of(module),
            presets,
        }
    }

    /// Checks the set against `interface`: the same module, unique names,
    /// and values that fit.
    ///
    /// # Errors
    ///
    /// [`PackageError::WrongModule`], [`PackageError::Duplicate`] or
    /// [`PackageError::InvalidParam`].
    pub fn check(&self, interface: &Interface) -> Result<(), PackageError> {
        if self.module != ModuleRef::of(&interface.id) {
            return Err(PackageError::WrongModule {
                expected: ModuleRef::of(&interface.id),
                found: self.module.clone(),
            });
        }
        unique(self.presets.iter().map(|p| p.name.as_str()))?;
        self.presets.iter().try_for_each(|p| p.check(interface))
    }

    /// Preset `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Preset> {
        self.presets.iter().find(|p| p.name == name)
    }

    /// Reads a preset document, refusing other formats and versions.
    ///
    /// # Errors
    ///
    /// [`PackageError::Format`], [`PackageError::UnsupportedVersion`] or
    /// [`PackageError::Syntax`].
    pub fn from_json(text: &str) -> Result<Self, PackageError> {
        header(text, PRESETS_FORMAT)?;
        serde_json::from_str(text).map_err(syntax)
    }

    /// The document, pretty-printed.
    #[must_use]
    pub fn to_json(&self) -> String {
        pretty(self)
    }
}

/// Where a step's parameter comes from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ParamBinding {
    /// A literal value.
    Value(ValueSource),
    /// The package's own parameter of that name.
    Param(String),
}

/// A named output of an earlier step.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputRef {
    /// The step.
    pub step: String,
    /// Its output.
    pub output: String,
}

/// Where a step's input comes from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum InputBinding {
    /// An earlier step's output.
    Output(OutputRef),
    /// The package's own input of that name.
    Input(String),
}

/// One module instance in a package body.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Step {
    /// The instance name, unique in the body: the last segment of its
    /// instance path, so its seeds derive from it.
    pub name: String,
    /// The module it instantiates.
    pub module: ModuleRef,
    /// Parameter bindings; the rest keep the module's defaults.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub params: BTreeMap<String, ParamBinding>,
    /// Input bindings.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub inputs: BTreeMap<String, InputBinding>,
}

/// A package body: a graph of module instances, in order (a step reads
/// only earlier steps, so the order is the evaluation order and there are
/// no cycles), and the step outputs that are the package's outputs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Body {
    /// The steps.
    pub steps: Vec<Step>,
    /// Each package output's source.
    pub outputs: BTreeMap<String, OutputRef>,
}

/// What a package needs from the engine that runs it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Requires {
    /// Engine features, at least at these versions.
    #[serde(default)]
    pub capabilities: Vec<Capability>,
    /// The modules its steps instantiate, at exactly these versions.
    #[serde(default)]
    pub modules: Vec<ModuleRef>,
}

/// A material package's editable source: interface, requirements,
/// presets and body. See the [crate docs](crate).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Package {
    /// [`PACKAGE_FORMAT`].
    pub format: String,
    /// [`FORMAT_VERSION`].
    pub version: u32,
    /// The module the package defines.
    pub interface: InterfaceSource,
    /// What it needs from the engine.
    #[serde(default)]
    pub requires: Requires,
    /// Saved parameter sets.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub presets: Vec<Preset>,
    /// How it is built.
    pub body: Body,
}

#[derive(Deserialize)]
struct Header {
    format: String,
    version: u32,
}

fn syntax(e: serde_json::Error) -> PackageError {
    PackageError::Syntax(e.to_string())
}

fn pretty<T: Serialize>(value: &T) -> String {
    let mut s = serde_json::to_string_pretty(value).expect("package data always serializes");
    s.push('\n');
    s
}

/// Checks a document's format and version before reading the rest, so an
/// unknown version is refused as such rather than as a shape error.
fn header(text: &str, format: &str) -> Result<(), PackageError> {
    let h: Header = serde_json::from_str(text).map_err(syntax)?;
    if h.format != format {
        return Err(PackageError::Format {
            expected: format.into(),
            found: h.format,
        });
    }
    if h.version != FORMAT_VERSION {
        return Err(PackageError::UnsupportedVersion {
            found: h.version,
            supported: FORMAT_VERSION,
        });
    }
    Ok(())
}

impl Package {
    /// An empty package defining `interface`.
    #[must_use]
    pub fn new(interface: InterfaceSource) -> Self {
        Self {
            format: PACKAGE_FORMAT.into(),
            version: FORMAT_VERSION,
            interface,
            requires: Requires::default(),
            presets: Vec::new(),
            body: Body {
                steps: Vec::new(),
                outputs: BTreeMap::new(),
            },
        }
    }

    /// Reads a package, refusing other formats and versions.
    ///
    /// # Errors
    ///
    /// [`PackageError::Format`], [`PackageError::UnsupportedVersion`] or
    /// [`PackageError::Syntax`] (which includes unknown fields).
    pub fn from_json(text: &str) -> Result<Self, PackageError> {
        header(text, PACKAGE_FORMAT)?;
        serde_json::from_str(text).map_err(syntax)
    }

    /// The package, pretty-printed: the editable form.
    #[must_use]
    pub fn to_json(&self) -> String {
        pretty(self)
    }

    /// A fingerprint of the package's canonical (compact) serialization:
    /// equal packages have equal fingerprints, however they were written.
    #[must_use]
    pub fn fingerprint(&self) -> u64 {
        let bytes = serde_json::to_vec(self).expect("package data always serializes");
        let mut words: Vec<u64> = bytes
            .chunks(8)
            .map(|c| {
                let mut w = [0_u8; 8];
                w[..c.len()].copy_from_slice(c);
                u64::from_le_bytes(w)
            })
            .collect();
        words.push(bytes.len() as u64);
        hash(0x7061_636b, &words) // "pack"
    }
}
