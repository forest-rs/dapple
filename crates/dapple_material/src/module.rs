// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Parameterized material modules.
//!
//! A [`Module`] publishes an [`Interface`] independent of how it is built:
//!
//! - a versioned identity ([`ModuleId`]);
//! - typed **parameters** with units, ranges and defaults ([`ParamDecl`]),
//!   so an oak board exposes board width, grain scale and seed, not
//!   `warp_17.amount`;
//! - **inputs**: materials, maps and host-resolved resources
//!   ([`InputDecl`]);
//! - named, typed **outputs** ([`OutputDecl`]).
//!
//! **Instantiation** ([`Context::instantiate`]) binds parameters and inputs
//! explicitly ([`Bind`]), checks them against the interface (unknown names,
//! wrong kinds and out-of-range values are errors, not clamps), fills
//! defaults, resolves resources through the host, and checks the outputs
//! the body returns. Modules instantiate modules the same way, so an
//! instance has a **path** (`wall/glaze`), and its seeds derive from that
//! path and its `seed` parameter ([`Args::seed`]): two instances of one
//! module differ, and one instance keeps its randomness however its
//! siblings change.
//!
//! **Diagnostics keep the module boundary.** Material operations run inside
//! a module record their reports with [`Context::record`]; each
//! [`Entry`] names the instance path and module that made it, even though
//! the result is one flat material. The module body is Rust (a builder,
//! identified by its [`ModuleId`] as a registered function is by its name);
//! what it builds, and every instance's arguments, are inspectable values.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use dapple_field::PortType;
use dapple_field::hash::hash;
use dapple_field::scoped::name_word;
use dapple_raster::typed::TypedRaster;
use glam::Vec3;

use crate::material::{Grid, Material, MaterialError};
use crate::report::Report;
use crate::resource::{self, Resolved, ResourceError, ResourceHost, ResourceRef, ResourceRequest};

/// A module's versioned identity. A new version is a new module: instances
/// name the version they were written against.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ModuleId {
    /// The module's stable name, such as `"dapple_library.mortar"`.
    pub name: &'static str,
    /// Its interface and behavior version.
    pub version: u32,
}

impl fmt::Display for ModuleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.name, self.version)
    }
}

/// A parameter's physical unit.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Unit {
    /// Dimensionless.
    None,
    /// Meters.
    Meters,
    /// A fraction in `[0, 1]`: of coverage, of an area.
    Fraction,
    /// Radians.
    Radians,
}

/// A parameter's type.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ParamKind {
    /// A real number in `range`, in `unit`.
    Scalar {
        /// The unit.
        unit: Unit,
        /// The allowed values, inclusive.
        range: [f32; 2],
    },
    /// A linear Rec. 709 color, each component in `[0, 1]`.
    Color,
    /// A count in `range`, inclusive.
    Integer {
        /// The allowed values.
        range: [u32; 2],
    },
    /// A seed, mixed with the instance path.
    Seed,
    /// A switch.
    Flag,
}

/// A parameter value.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ParamValue {
    /// For [`ParamKind::Scalar`].
    Scalar(f32),
    /// For [`ParamKind::Color`].
    Color(Vec3),
    /// For [`ParamKind::Integer`].
    Integer(u32),
    /// For [`ParamKind::Seed`].
    Seed(u64),
    /// For [`ParamKind::Flag`].
    Flag(bool),
}

impl ParamValue {
    fn fits(self, kind: ParamKind) -> Result<(), &'static str> {
        match (self, kind) {
            (Self::Scalar(v), ParamKind::Scalar { range, .. }) => {
                if v.is_finite() && v >= range[0] && v <= range[1] {
                    Ok(())
                } else {
                    Err("out of range")
                }
            }
            (Self::Color(c), ParamKind::Color) => {
                if c.is_finite() && c.min_element() >= 0.0 && c.max_element() <= 1.0 {
                    Ok(())
                } else {
                    Err("out of range")
                }
            }
            (Self::Integer(v), ParamKind::Integer { range }) => {
                if v >= range[0] && v <= range[1] {
                    Ok(())
                } else {
                    Err("out of range")
                }
            }
            (Self::Seed(_), ParamKind::Seed) | (Self::Flag(_), ParamKind::Flag) => Ok(()),
            _ => Err("wrong kind"),
        }
    }

    fn words(self, w: &mut Vec<u64>) {
        match self {
            Self::Scalar(v) => w.extend([1, u64::from(v.to_bits())]),
            Self::Color(c) => w.extend([
                2,
                u64::from(c.x.to_bits()),
                u64::from(c.y.to_bits()),
                u64::from(c.z.to_bits()),
            ]),
            Self::Integer(v) => w.extend([3, u64::from(v)]),
            Self::Seed(v) => w.extend([4, v]),
            Self::Flag(v) => w.extend([5, u64::from(v)]),
        }
    }
}

/// A declared parameter.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ParamDecl {
    /// Its name, unique in the interface.
    pub name: &'static str,
    /// Its type, unit and range.
    pub kind: ParamKind,
    /// Its value when an instance does not bind it.
    pub default: ParamValue,
    /// What it means.
    pub doc: &'static str,
}

/// What an input holds.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InputKind {
    /// A material on the context's grid.
    Material,
    /// A typed map on the context's grid.
    Map(PortType),
    /// A host-resolved resource.
    Resource(ResourceRequest),
}

/// A declared input.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct InputDecl {
    /// Its name, unique in the interface.
    pub name: &'static str,
    /// What it holds.
    pub kind: InputKind,
    /// Whether an instance must bind it.
    pub required: bool,
    /// What it means.
    pub doc: &'static str,
}

/// What an output holds.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OutputKind {
    /// A material.
    Material,
    /// A typed map.
    Map(PortType),
}

/// A declared output.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct OutputDecl {
    /// Its name, unique in the interface.
    pub name: &'static str,
    /// What it holds.
    pub kind: OutputKind,
    /// What it means.
    pub doc: &'static str,
}

/// A module's public interface.
#[derive(Clone, Debug, PartialEq)]
pub struct Interface {
    /// The module's identity.
    pub id: ModuleId,
    /// What the module makes.
    pub doc: &'static str,
    /// Parameters.
    pub params: Vec<ParamDecl>,
    /// Inputs.
    pub inputs: Vec<InputDecl>,
    /// Outputs.
    pub outputs: Vec<OutputDecl>,
}

/// A value bound to an input.
#[derive(Clone, Debug, PartialEq)]
pub enum Input {
    /// A material.
    Material(Material),
    /// A typed map.
    Map(TypedRaster),
    /// A logical resource, for the host to resolve.
    Resource(ResourceRef),
}

/// A value a module produced.
#[derive(Clone, Debug, PartialEq)]
pub enum Output {
    /// A material.
    Material(Material),
    /// A typed map.
    Map(TypedRaster),
}

/// A module's outputs, by name.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Outputs {
    entries: Vec<(&'static str, Output)>,
}

impl Outputs {
    /// No outputs yet.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Adds output `name`.
    #[must_use]
    pub fn with(mut self, name: &'static str, output: Output) -> Self {
        self.entries.push((name, output));
        self
    }

    /// Output `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Output> {
        self.entries
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, o)| o)
    }

    /// Material output `name`.
    #[must_use]
    pub fn material(&self, name: &str) -> Option<&Material> {
        match self.get(name) {
            Some(Output::Material(m)) => Some(m),
            _ => None,
        }
    }

    /// Takes material output `name`.
    #[must_use]
    pub fn take_material(&mut self, name: &str) -> Option<Material> {
        let i = self
            .entries
            .iter()
            .position(|(n, o)| *n == name && matches!(o, Output::Material(_)))?;
        match self.entries.remove(i).1 {
            Output::Material(m) => Some(m),
            Output::Map(_) => unreachable!("matched above"),
        }
    }

    /// Map output `name`.
    #[must_use]
    pub fn map(&self, name: &str) -> Option<&TypedRaster> {
        match self.get(name) {
            Some(Output::Map(m)) => Some(m),
            _ => None,
        }
    }
}

/// An instance's explicit bindings.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Bind {
    params: Vec<(String, ParamValue)>,
    inputs: Vec<(String, Input)>,
}

impl Bind {
    /// No bindings: every parameter at its default.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Binds parameter `name`.
    #[must_use]
    pub fn param(mut self, name: &str, value: ParamValue) -> Self {
        self.params.push((name.into(), value));
        self
    }

    /// Binds scalar parameter `name`.
    #[must_use]
    pub fn scalar(self, name: &str, value: f32) -> Self {
        self.param(name, ParamValue::Scalar(value))
    }

    /// Binds color parameter `name`.
    #[must_use]
    pub fn color(self, name: &str, value: Vec3) -> Self {
        self.param(name, ParamValue::Color(value))
    }

    /// Binds integer parameter `name`.
    #[must_use]
    pub fn integer(self, name: &str, value: u32) -> Self {
        self.param(name, ParamValue::Integer(value))
    }

    /// Binds the `seed` parameter.
    #[must_use]
    pub fn seed(self, value: u64) -> Self {
        self.param("seed", ParamValue::Seed(value))
    }

    /// Binds flag parameter `name`.
    #[must_use]
    pub fn flag(self, name: &str, value: bool) -> Self {
        self.param(name, ParamValue::Flag(value))
    }

    /// Binds input `name`.
    #[must_use]
    pub fn input(mut self, name: &str, input: Input) -> Self {
        self.inputs.push((name.into(), input));
        self
    }

    /// Binds material input `name`.
    #[must_use]
    pub fn material(self, name: &str, material: Material) -> Self {
        self.input(name, Input::Material(material))
    }
}

/// A resolved input.
#[derive(Clone, Debug, PartialEq)]
enum Bound {
    Material(Material),
    Map(TypedRaster),
    Resource(Resolved),
}

/// An instance's checked arguments, as its body sees them.
#[derive(Clone, Debug, PartialEq)]
pub struct Args {
    params: Vec<(&'static str, ParamValue)>,
    inputs: Vec<(&'static str, Option<Bound>)>,
    seed: u64,
    fingerprint: u64,
}

impl Args {
    fn param(&self, name: &str) -> ParamValue {
        self.params
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| *v)
            .unwrap_or_else(|| panic!("the interface declares no parameter {name:?}"))
    }

    /// Scalar parameter `name`.
    ///
    /// # Panics
    ///
    /// When the interface declares no scalar parameter `name`: a bug in the
    /// module, not in the instance.
    #[must_use]
    pub fn scalar(&self, name: &str) -> f32 {
        match self.param(name) {
            ParamValue::Scalar(v) => v,
            other => panic!("{name:?} is {other:?}, not a scalar"),
        }
    }

    /// Color parameter `name`.
    ///
    /// # Panics
    ///
    /// As [`Self::scalar`].
    #[must_use]
    pub fn color(&self, name: &str) -> Vec3 {
        match self.param(name) {
            ParamValue::Color(v) => v,
            other => panic!("{name:?} is {other:?}, not a color"),
        }
    }

    /// Integer parameter `name`.
    ///
    /// # Panics
    ///
    /// As [`Self::scalar`].
    #[must_use]
    pub fn integer(&self, name: &str) -> u32 {
        match self.param(name) {
            ParamValue::Integer(v) => v,
            other => panic!("{name:?} is {other:?}, not an integer"),
        }
    }

    /// Flag parameter `name`.
    ///
    /// # Panics
    ///
    /// As [`Self::scalar`].
    #[must_use]
    pub fn flag(&self, name: &str) -> bool {
        match self.param(name) {
            ParamValue::Flag(v) => v,
            other => panic!("{name:?} is {other:?}, not a flag"),
        }
    }

    /// A seed for `purpose`, derived from the instance path and the
    /// instance's `seed` parameter.
    #[must_use]
    pub fn seed(&self, purpose: &str) -> u64 {
        hash(self.seed, &[name_word(0x7365_6564, purpose)]) // "seed"
    }

    /// The same seed reduced to 32 bits, for field ops that take one.
    #[must_use]
    pub fn seed32(&self, purpose: &str) -> u64 {
        self.seed(purpose) & 0xffff_ffff
    }

    fn input(&self, name: &str) -> Option<&Bound> {
        self.inputs
            .iter()
            .find(|(n, _)| *n == name)
            .and_then(|(_, b)| b.as_ref())
    }

    /// Material input `name`, when bound.
    #[must_use]
    pub fn material(&self, name: &str) -> Option<&Material> {
        match self.input(name) {
            Some(Bound::Material(m)) => Some(m),
            _ => None,
        }
    }

    /// Map input `name`, when bound.
    #[must_use]
    pub fn map(&self, name: &str) -> Option<&TypedRaster> {
        match self.input(name) {
            Some(Bound::Map(m)) => Some(m),
            _ => None,
        }
    }

    /// Resource input `name`, resolved, when bound.
    #[must_use]
    pub fn resource(&self, name: &str) -> Option<&Resolved> {
        match self.input(name) {
            Some(Bound::Resource(r)) => Some(r),
            _ => None,
        }
    }

    /// A content fingerprint of the module identity and every argument.
    #[must_use]
    pub const fn fingerprint(&self) -> u64 {
        self.fingerprint
    }
}

/// Why an instantiation failed.
#[derive(Clone, Debug, PartialEq)]
pub enum ModuleErrorKind {
    /// A binding names no parameter or input of the interface.
    Unknown(String),
    /// A parameter value has the wrong kind or is out of range.
    InvalidParam {
        /// The parameter.
        name: String,
        /// What is wrong.
        reason: &'static str,
    },
    /// A required input is not bound.
    MissingInput(&'static str),
    /// An input holds the wrong kind of value or is off the grid.
    InvalidInput(&'static str),
    /// A resource could not be resolved or does not fit.
    Resource(ResourceError),
    /// The body did not produce a declared output, or produced one of the
    /// wrong kind.
    Output(&'static str),
    /// A material operation in the body failed.
    Material(MaterialError),
    /// Something else in the body failed.
    Body(&'static str),
}

/// An instantiation failure, at an instance path.
#[derive(Clone, Debug, PartialEq)]
pub struct ModuleError {
    /// Where: the instance path.
    pub path: String,
    /// What.
    pub kind: Box<ModuleErrorKind>,
}

impl ModuleError {
    /// A failure of kind `kind` in the body, at no path yet; the context
    /// adds the path.
    #[must_use]
    pub fn body(kind: ModuleErrorKind) -> Self {
        Self {
            path: String::new(),
            kind: Box::new(kind),
        }
    }
}

impl From<MaterialError> for ModuleError {
    fn from(e: MaterialError) -> Self {
        Self::body(ModuleErrorKind::Material(e))
    }
}

impl From<dapple_raster::RasterError> for ModuleError {
    fn from(e: dapple_raster::RasterError) -> Self {
        Self::body(ModuleErrorKind::Material(MaterialError::Raster(e)))
    }
}

impl From<dapple_raster::typed::TypedError> for ModuleError {
    fn from(e: dapple_raster::typed::TypedError) -> Self {
        Self::body(ModuleErrorKind::Material(MaterialError::Typed(e)))
    }
}

impl From<dapple_field::scoped::ContractError> for ModuleError {
    fn from(e: dapple_field::scoped::ContractError) -> Self {
        Self::body(ModuleErrorKind::Material(MaterialError::Contract(e)))
    }
}

impl fmt::Display for ModuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: ", self.path)?;
        match &*self.kind {
            ModuleErrorKind::Unknown(n) => write!(f, "nothing named {n:?}"),
            ModuleErrorKind::InvalidParam { name, reason } => write!(f, "{name:?}: {reason}"),
            ModuleErrorKind::MissingInput(n) => write!(f, "input {n:?} is required"),
            ModuleErrorKind::InvalidInput(n) => write!(f, "input {n:?} does not fit"),
            ModuleErrorKind::Resource(e) => e.fmt(f),
            ModuleErrorKind::Output(n) => write!(f, "output {n:?} missing or mistyped"),
            ModuleErrorKind::Material(e) => e.fmt(f),
            ModuleErrorKind::Body(what) => f.write_str(what),
        }
    }
}

impl core::error::Error for ModuleError {}

/// A material module: see the [module docs](self).
pub trait Module {
    /// The public interface.
    fn interface(&self) -> Interface;

    /// Builds the outputs from checked arguments, in `cx`.
    ///
    /// # Errors
    ///
    /// [`ModuleError`] when a nested instantiation or an operation fails.
    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError>;
}

/// What happened in an instance.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// The instance was built with arguments of this fingerprint.
    Instantiated {
        /// [`Args::fingerprint`].
        fingerprint: u64,
    },
    /// A material operation ran in the instance's body.
    Operation(Report),
}

/// One diagnostic, attributed to the instance that made it.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    /// The instance path, such as `"wall/glaze"`.
    pub path: String,
    /// The module of that instance.
    pub module: ModuleId,
    /// What happened.
    pub event: Event,
}

/// Every diagnostic of a build, in order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Diagnostics {
    /// The entries.
    pub entries: Vec<Entry>,
}

impl Diagnostics {
    /// The instances built, as `(path, module)`, in order.
    pub fn instances(&self) -> impl Iterator<Item = (&str, ModuleId)> + '_ {
        self.entries.iter().filter_map(|e| match e.event {
            Event::Instantiated { .. } => Some((e.path.as_str(), e.module)),
            Event::Operation(_) => None,
        })
    }

    /// The reports of operations run in the instance at `path` itself.
    pub fn reports_at<'a>(&'a self, path: &'a str) -> impl Iterator<Item = &'a Report> + 'a {
        self.entries.iter().filter_map(move |e| match &e.event {
            Event::Operation(r) if e.path == path => Some(r),
            _ => None,
        })
    }
}

impl fmt::Display for Diagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for e in &self.entries {
            match &e.event {
                Event::Instantiated { fingerprint } => {
                    writeln!(f, "{} = {} ({fingerprint:016x})", e.path, e.module)?;
                }
                Event::Operation(r) => writeln!(f, "{}: {r}", e.path)?,
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Frame {
    path: String,
    module: ModuleId,
}

/// Where modules are instantiated: the grid they build on, the resource
/// host, the instance path and the diagnostics.
pub struct Context<'a> {
    grid: Grid,
    host: &'a dyn ResourceHost,
    stack: Vec<Frame>,
    diagnostics: Diagnostics,
}

impl fmt::Debug for Context<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context")
            .field("grid", &self.grid)
            .field("path", &self.path())
            .field("diagnostics", &self.diagnostics.entries.len())
            .finish_non_exhaustive()
    }
}

impl<'a> Context<'a> {
    /// A context building on `grid`, resolving resources with `host`.
    #[must_use]
    pub fn new(grid: Grid, host: &'a dyn ResourceHost) -> Self {
        Self {
            grid,
            host,
            stack: Vec::new(),
            diagnostics: Diagnostics::default(),
        }
    }

    /// The grid every material and map is on.
    #[must_use]
    pub const fn grid(&self) -> Grid {
        self.grid
    }

    /// The current instance path; empty at the top.
    #[must_use]
    pub fn path(&self) -> &str {
        self.stack.last().map_or("", |f| f.path.as_str())
    }

    /// The diagnostics so far.
    #[must_use]
    pub const fn diagnostics(&self) -> &Diagnostics {
        &self.diagnostics
    }

    /// Records `report` against the current instance, and returns the
    /// value it came with, so operations chain:
    /// `let m = cx.record(ops::coat(&m, &c)?);`.
    pub fn record<T>(&mut self, (value, report): (T, Report)) -> T {
        let module = self.stack.last().map_or(
            ModuleId {
                name: "",
                version: 0,
            },
            |f| f.module,
        );
        self.diagnostics.entries.push(Entry {
            path: self.path().into(),
            module,
            event: Event::Operation(report),
        });
        value
    }

    /// Instantiates `module` as `name` under the current instance, with
    /// `bind`'s arguments.
    ///
    /// # Errors
    ///
    /// [`ModuleError`] at the instance's path for a binding that does not
    /// fit the interface, a resource that does not resolve, a body that
    /// fails, or outputs that do not match the interface.
    pub fn instantiate(
        &mut self,
        module: &dyn Module,
        name: &str,
        bind: Bind,
    ) -> Result<Outputs, ModuleError> {
        let interface = module.interface();
        let path = if self.path().is_empty() {
            String::from(name)
        } else {
            format!("{}/{name}", self.path())
        };
        let fail = |kind| ModuleError {
            path: path.clone(),
            kind: Box::new(kind),
        };
        for (n, _) in &bind.params {
            if !interface.params.iter().any(|p| p.name == n) {
                return Err(fail(ModuleErrorKind::Unknown(n.clone())));
            }
        }
        for (n, _) in &bind.inputs {
            if !interface.inputs.iter().any(|p| p.name == n) {
                return Err(fail(ModuleErrorKind::Unknown(n.clone())));
            }
        }
        let mut words = alloc::vec![
            name_word(1, interface.id.name),
            u64::from(interface.id.version)
        ];
        let mut params = Vec::with_capacity(interface.params.len());
        let mut user_seed = 0;
        for decl in &interface.params {
            let value = bind
                .params
                .iter()
                .rev()
                .find(|(n, _)| n == decl.name)
                .map_or(decl.default, |(_, v)| *v);
            value.fits(decl.kind).map_err(|reason| {
                fail(ModuleErrorKind::InvalidParam {
                    name: decl.name.into(),
                    reason,
                })
            })?;
            if let ParamValue::Seed(s) = value {
                user_seed = s;
            }
            words.push(name_word(2, decl.name));
            value.words(&mut words);
            params.push((decl.name, value));
        }
        let mut inputs = Vec::with_capacity(interface.inputs.len());
        for decl in &interface.inputs {
            let given = bind.inputs.iter().rev().find(|(n, _)| n == decl.name);
            let bound = match (given.map(|(_, i)| i), decl.kind) {
                (None, _) if decl.required => {
                    return Err(fail(ModuleErrorKind::MissingInput(decl.name)));
                }
                (None, _) => None,
                (Some(Input::Material(m)), InputKind::Material) => {
                    if m.grid() != self.grid {
                        return Err(fail(ModuleErrorKind::InvalidInput(decl.name)));
                    }
                    words.extend([name_word(3, decl.name), m.digest()]);
                    Some(Bound::Material(m.clone()))
                }
                (Some(Input::Map(r)), InputKind::Map(port)) => {
                    if r.port() != port || !self.grid.holds(r) {
                        return Err(fail(ModuleErrorKind::InvalidInput(decl.name)));
                    }
                    words.extend([name_word(3, decl.name), r.digest()]);
                    Some(Bound::Map(r.clone()))
                }
                (Some(Input::Resource(reference)), InputKind::Resource(request)) => {
                    let resolved = self
                        .host
                        .resolve(reference)
                        .map_err(|e| fail(ModuleErrorKind::Resource(e)))?;
                    resource::check(reference, request, &resolved)
                        .map_err(|e| fail(ModuleErrorKind::Resource(e)))?;
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "splitting the 128-bit fingerprint into its halves"
                    )]
                    words.extend([
                        name_word(3, decl.name),
                        resolved.content.0 as u64,
                        (resolved.content.0 >> 64) as u64,
                    ]);
                    Some(Bound::Resource(resolved))
                }
                _ => return Err(fail(ModuleErrorKind::InvalidInput(decl.name))),
            };
            inputs.push((decl.name, bound));
        }
        let mut seed_words: Vec<u64> = path.split('/').map(|s| name_word(4, s)).collect();
        seed_words.push(user_seed);
        let args = Args {
            params,
            inputs,
            seed: hash(0x696e_7374, &seed_words),   // "inst"
            fingerprint: hash(0x6172_6773, &words), // "args"
        };
        self.diagnostics.entries.push(Entry {
            path: path.clone(),
            module: interface.id,
            event: Event::Instantiated {
                fingerprint: args.fingerprint,
            },
        });
        self.stack.push(Frame {
            path: path.clone(),
            module: interface.id,
        });
        let built = module.build(self, &args);
        self.stack.pop();
        let outputs = built.map_err(|mut e| {
            if e.path.is_empty() {
                e.path.clone_from(&path);
            }
            e
        })?;
        for decl in &interface.outputs {
            let ok = match (outputs.get(decl.name), decl.kind) {
                (Some(Output::Material(m)), OutputKind::Material) => m.grid() == self.grid,
                (Some(Output::Map(r)), OutputKind::Map(port)) => {
                    r.port() == port && self.grid.holds(r)
                }
                _ => false,
            };
            if !ok {
                return Err(fail(ModuleErrorKind::Output(decl.name)));
            }
        }
        Ok(outputs)
    }

    /// The diagnostics, ending the context.
    #[must_use]
    pub fn into_diagnostics(self) -> Diagnostics {
        self.diagnostics
    }
}
