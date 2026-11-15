// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The execution artifact: a package checked against an engine and
//! resolved into a [`Module`].
//!
//! # Guarded steps
//!
//! A step with a `when` runs only where that flag expression holds; one
//! with an `unless`, only where it does not ([`crate::source::Step`]). Each
//! is a **guard**: a condition's text and the value it must have. Two
//! guards of the same text are the same condition (the text is compiled
//! once, and every instance computes it once, before its steps), so a
//! `when` and an `unless` of the same text are its two sides.
//!
//! Compiling proves that nothing reads an output that might not exist. A
//! binding is read under the guards known to hold where it is read:
//!
//! - a step's inputs under the step's own guard, if it has one;
//! - a select's `then` under its condition, and its `else` under the
//!   condition's other side, besides what holds where the select is read;
//! - a step's `otherwise` under the other side of its guard (it is read
//!   exactly where the step does not run);
//! - the package's outputs under nothing.
//!
//! A guarded step's output can be read where its guard is among those, or
//! anywhere if the step's `otherwise` gives that output; anything else is
//! [`PackageError::MaybeSkipped`]. That is the whole rule: it compares
//! texts, never what they mean, so `moss > 0.0` and `0.0 < moss` are two
//! conditions, and an author who wants one writes one. It is sound because
//! a condition is a pure function of the package's arguments, computed once
//! per instance.
//!
//! Conditions themselves always run. A guarded step's parameter
//! expressions run only where it does
//! ([`dapple_logic::Builder::expression_when`]), so a value that would not
//! fit, or would stop the program, is no error where the step is skipped.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;

use dapple_field::hash::hash;
use dapple_field::scoped::name_word;
use dapple_logic::{Builder as LogicBuilder, Program, Type, Value};
use dapple_material::module::{
    Args, Bind, Context, Input, InputKind, Interface, Module, ModuleError, ModuleErrorKind,
    ModuleId, Output, OutputDecl, OutputKind, Outputs, ParamKind, ParamValue,
};
use dapple_material::{ChannelId, Tiling};

use crate::PackageError;
use crate::source::{
    BodyOutput, Capability, InputBinding, ModuleRef, OutputRef, Package, ParamBinding, Preset,
    Select, Step, unique,
};

/// Module-graph bodies ([`crate::source::Body`]).
pub const GRAPH: &str = "dapple.graph";
/// Host-resolved resource inputs.
pub const RESOURCES: &str = "dapple.resources";
/// Output semantics checked on every instance.
pub const SEMANTICS: &str = "dapple.semantics";
/// Expression bindings ([`ParamBinding::Expr`]), in the language of
/// [`dapple_logic`], at version 1; and, at version 2, guarded steps
/// ([`crate::source::Step::when`], [`crate::source::Step::unless`],
/// [`crate::source::Step::otherwise`]) and selects
/// ([`crate::source::Select`]).
pub const LOGIC: &str = "dapple.logic";
/// The version of [`LOGIC`] that guards and selects need.
pub const LOGIC_GUARDS: u32 = 2;

/// What an engine offers packages: modules by identity, and capabilities
/// by name and version.
///
/// [`Registry::new`] offers this crate's own capabilities ([`GRAPH`],
/// [`RESOURCES`] and [`SEMANTICS`] at version 1, [`LOGIC`] at
/// [`LOGIC_GUARDS`]) and no modules; a host registers the modules it links,
/// and may withdraw capabilities it cannot serve (an engine without an
/// image host has no resources).
#[derive(Clone)]
pub struct Registry {
    modules: Vec<Arc<dyn Module>>,
    capabilities: Vec<Capability>,
}

impl fmt::Debug for Registry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Registry")
            .field(
                "modules",
                &self
                    .modules
                    .iter()
                    .map(|m| m.interface().id)
                    .collect::<Vec<_>>(),
            )
            .field("capabilities", &self.capabilities)
            .finish()
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    /// This crate's capabilities and no modules.
    #[must_use]
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            capabilities: alloc::vec![
                Capability::new(GRAPH, 1),
                Capability::new(RESOURCES, 1),
                Capability::new(SEMANTICS, 1),
                Capability::new(LOGIC, LOGIC_GUARDS),
            ],
        }
    }

    /// Registers `module`, replacing one of the same identity.
    #[must_use]
    pub fn with(mut self, module: Arc<dyn Module>) -> Self {
        let id = module.interface().id;
        self.modules.retain(|m| m.interface().id != id);
        self.modules.push(module);
        self
    }

    /// Offers `capability`, replacing another version of it.
    #[must_use]
    pub fn offer(mut self, capability: Capability) -> Self {
        self.capabilities.retain(|c| c.name != capability.name);
        self.capabilities.push(capability);
        self
    }

    /// Withdraws capability `name`.
    #[must_use]
    pub fn withdraw(mut self, name: &str) -> Self {
        self.capabilities.retain(|c| c.name != name);
        self
    }

    /// Every registered module, in registration order.
    pub fn modules(&self) -> impl Iterator<Item = &Arc<dyn Module>> + '_ {
        self.modules.iter()
    }

    /// The module `id` names, at exactly its version.
    #[must_use]
    pub fn module(&self, id: &ModuleRef) -> Option<&Arc<dyn Module>> {
        self.modules
            .iter()
            .find(|m| ModuleRef::of(&m.interface().id) == *id)
    }

    /// Checks that the engine offers `capability` at its version or later.
    ///
    /// # Errors
    ///
    /// [`PackageError::UnknownCapability`] or
    /// [`PackageError::CapabilityVersion`].
    pub fn supports(&self, capability: &Capability) -> Result<(), PackageError> {
        match self.capabilities.iter().find(|c| c.name == capability.name) {
            None => Err(PackageError::UnknownCapability(capability.clone())),
            Some(c) if c.version < capability.version => Err(PackageError::CapabilityVersion {
                required: capability.clone(),
                offered: c.version,
            }),
            Some(_) => Ok(()),
        }
    }

    /// Compiles `package` against this engine: see [`CompiledPackage`].
    ///
    /// # Errors
    ///
    /// A [`PackageError`] naming the first thing that does not check.
    pub fn compile(&self, package: &Package) -> Result<CompiledPackage, PackageError> {
        compile(self, package)
    }
}

#[derive(Clone, Debug)]
enum Arg {
    Value(ParamValue),
    Param(String),
    /// The program's result of this index.
    Derived(usize),
}

/// The type an expression has where it binds or reads a parameter of
/// `kind`, if expressions handle that kind.
const fn logic_type(kind: ParamKind) -> Option<Type> {
    match kind {
        ParamKind::Scalar { .. } => Some(Type::Scalar),
        ParamKind::Integer { .. } => Some(Type::Integer),
        ParamKind::Flag => Some(Type::Flag),
        ParamKind::Color | ParamKind::Seed => None,
    }
}

/// One of the program's expressions: where it is used, and what for.
#[derive(Clone, Debug)]
struct Expression {
    /// The step it belongs to; empty for a package output's select.
    step: String,
    /// What it is named by in a failure: `"<parameter> = <source>"`,
    /// `"when = <source>"`, `"<input> = select when <source>"`.
    label: String,
    /// For a parameter binding, the parameter's kind and the step's index;
    /// `None` for a condition.
    binds: Option<(ParamKind, usize)>,
}

/// A package's expressions, compiled into one program.
#[derive(Clone, Debug)]
struct Logic {
    program: Program,
    /// The package parameters the program reads, in argument order.
    inputs: Vec<String>,
    /// Each result's expression, in the program's result order.
    expressions: Vec<Expression>,
}

impl Logic {
    fn failure(&self, cx: &Context<'_>, i: usize, reason: &'static str) -> ModuleError {
        let e = &self.expressions[i];
        ModuleError {
            path: match (cx.path().is_empty(), e.step.is_empty()) {
                (_, true) => cx.path().into(),
                (true, false) => e.step.clone(),
                (false, false) => format!("{}/{}", cx.path(), e.step),
            },
            kind: Box::new(ModuleErrorKind::InvalidParam {
                name: e.label.clone(),
                reason,
            }),
        }
    }

    /// Every result under `args`.
    fn run(&self, cx: &Context<'_>, args: &Args) -> Result<Vec<Value>, ModuleError> {
        // The inputs are the interface's scalars, integers and flags, and
        // the context has checked `args` against the interface.
        let inputs: Vec<Value> = self
            .inputs
            .iter()
            .filter_map(|n| match args.param(n) {
                ParamValue::Scalar(v) => Some(Value::Scalar(v)),
                ParamValue::Integer(v) => Some(Value::Integer(v)),
                ParamValue::Flag(v) => Some(Value::Flag(v)),
                ParamValue::Color(_) | ParamValue::Seed(_) => None,
            })
            .collect();
        self.program.run(&inputs).map_err(|e| match e.expression {
            Some(i) => self.failure(cx, i, e.kind.reason()),
            None => ModuleError::body(ModuleErrorKind::Body(e.kind.reason())),
        })
    }

    /// The results as parameter values, each that binds a parameter of a
    /// step that runs checked against that parameter. The others are
    /// placeholders nothing reads.
    fn derive(
        &self,
        cx: &Context<'_>,
        values: &[Value],
        runs: &[bool],
    ) -> Result<Vec<ParamValue>, ModuleError> {
        values
            .iter()
            .zip(&self.expressions)
            .enumerate()
            .map(|(i, (v, e))| {
                let v = match *v {
                    Value::Scalar(x) => ParamValue::Scalar(x),
                    Value::Integer(x) => ParamValue::Integer(x),
                    Value::Flag(x) => ParamValue::Flag(x),
                };
                if let Some((kind, step)) = e.binds
                    && runs[step]
                {
                    v.fits(kind).map_err(|reason| self.failure(cx, i, reason))?;
                }
                Ok(v)
            })
            .collect()
    }
}

/// A package's expressions as they are added, compiled into one program at
/// the end. Begun at the first: a package without expressions does no work
/// for them.
struct Draft<'a> {
    interface: &'a Interface,
    builder: Option<(LogicBuilder, Vec<String>)>,
    expressions: Vec<Expression>,
    /// Each condition's text and result, so a text is compiled once.
    conditions: Vec<(String, usize)>,
}

impl Draft<'_> {
    fn builder(&mut self) -> Result<&mut LogicBuilder, PackageError> {
        if self.builder.is_none() {
            // Expressions read the package's scalars, integers and flags,
            // in the interface's order.
            let mut b = LogicBuilder::new();
            let mut inputs = Vec::new();
            for p in &self.interface.params {
                if let Some(t) = logic_type(p.kind) {
                    b.input(&p.name, t).map_err(PackageError::Logic)?;
                    inputs.push(p.name.to_string());
                }
            }
            self.builder = Some((b, inputs));
        }
        Ok(&mut self.builder.as_mut().expect("begun").0)
    }

    /// The result of condition `source`, compiled on its first use, by the
    /// binding `name` of `step` (empty for a package output).
    fn condition(
        &mut self,
        step: &str,
        name: &str,
        label: String,
        source: &str,
    ) -> Result<usize, PackageError> {
        if let Some((_, i)) = self.conditions.iter().find(|(s, _)| s == source) {
            return Ok(*i);
        }
        let i = self
            .builder()?
            .expression(source, Type::Flag)
            .map_err(|error| expression_error(step, name, source, error))?;
        self.expressions.push(Expression {
            step: step.into(),
            label,
            binds: None,
        });
        self.conditions.push((source.into(), i));
        Ok(i)
    }

    fn finish(self) -> Result<Option<Logic>, PackageError> {
        let Some((builder, inputs)) = self.builder else {
            return Ok(None);
        };
        Ok(Some(Logic {
            program: builder.build().map_err(PackageError::Logic)?,
            inputs,
            expressions: self.expressions,
        }))
    }
}

fn expression_error(
    step: &str,
    name: &str,
    source: &str,
    error: dapple_logic::Error,
) -> PackageError {
    PackageError::Expression {
        step: step.into(),
        name: name.into(),
        source: source.into(),
        error: Box::new(error),
    }
}

/// A step's guard: a condition's result, the value it must have, and its
/// text, for messages.
#[derive(Clone, Debug)]
struct Guard {
    result: usize,
    holds: bool,
    source: String,
}

/// Where a value comes from, resolved.
#[derive(Clone, Debug)]
enum Source {
    /// An earlier step's output (or, where the step did not run, what its
    /// `otherwise` gives).
    Step(usize, String),
    /// The package's input.
    Input(String),
    /// `then` where the condition result is `true`, `otherwise` where not.
    Select {
        condition: usize,
        then: Box<Self>,
        otherwise: Box<Self>,
    },
}

#[derive(Clone)]
struct CompiledStep {
    name: String,
    id: ModuleId,
    module: Arc<dyn Module>,
    guard: Option<Guard>,
    params: Vec<(String, Arg)>,
    inputs: Vec<(String, Source)>,
    /// Where the step does not run: its outputs that pass an input through.
    otherwise: Vec<(String, Source)>,
}

#[derive(Clone, Debug)]
struct CompiledOutput {
    name: String,
    source: Source,
    channels: Vec<ChannelId>,
    tiling: Option<Tiling>,
}

/// A package compiled against a [`Registry`]: the execution artifact, kept
/// apart from the editable [`Package`] it came from.
///
/// Compiling resolves every name the source uses (steps to registered
/// modules, bindings to parameters, inputs and earlier outputs), checks
/// every binding's kind and range against the module it feeds, checks that
/// nothing reads the output of a guarded step where it might not have run
/// (see the [module docs](self)), and orders nothing it does not have to:
/// the steps run in their written order. The result is a [`Module`],
/// instantiated like any other, whose outputs are checked against their
/// declared semantics on every instance. A step whose guard does not hold
/// is not instantiated; the instance's diagnostics record it as skipped
/// ([`dapple_material::module::Diagnostics::skipped`]).
#[derive(Clone)]
pub struct CompiledPackage {
    interface: Interface,
    steps: Vec<CompiledStep>,
    outputs: Vec<CompiledOutput>,
    presets: Vec<Preset>,
    logic: Option<Logic>,
    fingerprint: u64,
}

impl fmt::Debug for CompiledPackage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledPackage")
            .field("id", &self.interface.id)
            .field(
                "steps",
                &self
                    .steps
                    .iter()
                    .map(|s| (s.name.as_str(), &s.id, s.guard.as_ref()))
                    .collect::<Vec<_>>(),
            )
            .field("outputs", &self.outputs)
            .field(
                "expressions",
                &self.logic.as_ref().map_or(0, |l| l.expressions.len()),
            )
            .field("fingerprint", &self.fingerprint)
            .finish_non_exhaustive()
    }
}

impl CompiledPackage {
    /// A fingerprint of the source and of the identities and
    /// [definitions](Module::definition_fingerprint) of the modules it was
    /// compiled against: the package's own definition fingerprint.
    #[must_use]
    pub const fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// The bindings of preset `name`, checked when compiled.
    #[must_use]
    pub fn preset(&self, name: &str) -> Option<Bind> {
        self.presets
            .iter()
            .find(|p| p.name == name)
            .map(Preset::bind)
    }

    /// The package's expressions (expression bindings, and the conditions
    /// of guards and selects) compiled into one program, if it has any: run
    /// once per instance, before the steps. For inspection: its size, its
    /// bytes, its cost.
    #[must_use]
    pub fn logic(&self) -> Option<&Program> {
        self.logic.as_ref().map(|l| &l.program)
    }
}

fn binding_error(step: &str, name: &str, reason: &'static str) -> PackageError {
    PackageError::Binding {
        step: step.into(),
        name: name.into(),
        reason,
    }
}

/// Whether a package parameter of `outer` kind can feed one of `inner`
/// kind: the same kind, unit and a range inside the inner one.
fn feeds(outer: ParamKind, inner: ParamKind) -> bool {
    match (outer, inner) {
        (ParamKind::Scalar { unit: a, range: r }, ParamKind::Scalar { unit: b, range: s }) => {
            a == b && r[0] >= s[0] && r[1] <= s[1]
        }
        (ParamKind::Integer { range: r }, ParamKind::Integer { range: s }) => {
            r[0] >= s[0] && r[1] <= s[1]
        }
        (ParamKind::Color, ParamKind::Color)
        | (ParamKind::Seed, ParamKind::Seed)
        | (ParamKind::Flag, ParamKind::Flag) => true,
        _ => false,
    }
}

fn input_fits(outer: InputKind, inner: InputKind) -> bool {
    outer == inner
}

fn output_fits(outer: OutputKind, inner: InputKind) -> bool {
    matches!((outer, inner), (OutputKind::Material, InputKind::Material))
        || matches!((outer, inner), (OutputKind::Map(a), InputKind::Map(b)) if a == b)
}

/// Who reads a binding, for errors: a step's input (or `otherwise`
/// output), or a package output (`step` empty).
struct Reader<'a> {
    step: &'a str,
    name: &'a str,
}

impl Reader<'_> {
    fn fail(&self, reason: &'static str) -> PackageError {
        if self.step.is_empty() {
            PackageError::Output {
                name: self.name.into(),
                reason,
            }
        } else {
            binding_error(self.step, self.name, reason)
        }
    }
}

/// What compiling the steps so far knows.
struct Compiler<'a> {
    package: &'a Package,
    interface: &'a Interface,
    steps: Vec<CompiledStep>,
    step_interfaces: Vec<Interface>,
    logic: Draft<'a>,
    /// Whether the package may use guards and selects.
    guards: bool,
}

impl Compiler<'_> {
    fn require_guards(&self) -> Result<(), PackageError> {
        if self.guards {
            Ok(())
        } else {
            Err(PackageError::UndeclaredCapability(Capability::new(
                LOGIC,
                LOGIC_GUARDS,
            )))
        }
    }

    /// Output `r` of an earlier step, read where `holds` hold: its step's
    /// index and declaration.
    fn step_output(
        &self,
        reader: &Reader<'_>,
        r: &OutputRef,
        holds: &[(usize, bool)],
    ) -> Result<(usize, OutputDecl), PackageError> {
        let i = self
            .steps
            .iter()
            .position(|s| s.name == r.step)
            .ok_or_else(|| {
                reader.fail(if reader.step.is_empty() {
                    "no such step"
                } else {
                    "not an earlier step"
                })
            })?;
        let decl = self.step_interfaces[i]
            .outputs
            .iter()
            .find(|o| o.name == r.output.as_str())
            .ok_or_else(|| reader.fail("no such step output"))?;
        let step = &self.steps[i];
        if let Some(g) = &step.guard
            && !step.otherwise.iter().any(|(o, _)| *o == r.output)
            && !holds.contains(&(g.result, g.holds))
        {
            return Err(PackageError::MaybeSkipped {
                step: reader.step.into(),
                name: reader.name.into(),
                reads: step.name.clone(),
                condition: g.source.clone(),
                holds: g.holds,
            });
        }
        Ok((i, decl.clone()))
    }

    /// `binding`, read for an input of `kind` where `holds` hold: its
    /// source, and whether it is always there (an optional package input
    /// may not be).
    fn input(
        &mut self,
        reader: &Reader<'_>,
        binding: &InputBinding,
        kind: InputKind,
        holds: &mut Vec<(usize, bool)>,
    ) -> Result<(Source, bool), PackageError> {
        Ok(match binding {
            InputBinding::Output(r) => {
                let (i, out) = self.step_output(reader, r, holds)?;
                if !output_fits(out.kind, kind) {
                    return Err(reader.fail("the output's kind does not fit"));
                }
                (Source::Step(i, r.output.clone()), true)
            }
            InputBinding::Input(n) => {
                let outer = self
                    .interface
                    .inputs
                    .iter()
                    .find(|d| d.name == n.as_str())
                    .ok_or_else(|| reader.fail("no such package input"))?;
                if !input_fits(outer.kind, kind) {
                    return Err(reader.fail("the package input's kind does not fit"));
                }
                (Source::Input(n.clone()), outer.required)
            }
            InputBinding::Select(select) => {
                let label = format!("{} = select when {}", reader.name, select.when);
                self.select(reader, select, label, holds, |c, b, holds| {
                    c.input(reader, b, kind, holds)
                })?
            }
        })
    }

    /// A select: its condition, compiled, and each branch read under its
    /// side of it.
    fn select<T>(
        &mut self,
        reader: &Reader<'_>,
        select: &Select<T>,
        label: String,
        holds: &mut Vec<(usize, bool)>,
        mut branch: impl FnMut(
            &mut Self,
            &T,
            &mut Vec<(usize, bool)>,
        ) -> Result<(Source, bool), PackageError>,
    ) -> Result<(Source, bool), PackageError> {
        self.require_guards()?;
        let condition = self
            .logic
            .condition(reader.step, reader.name, label, &select.when)?;
        holds.push((condition, true));
        let then = branch(self, &select.then, holds);
        holds.pop();
        let (then, a) = then?;
        holds.push((condition, false));
        let otherwise = branch(self, &select.otherwise, holds);
        holds.pop();
        let (otherwise, b) = otherwise?;
        Ok((
            Source::Select {
                condition,
                then: Box::new(then),
                otherwise: Box::new(otherwise),
            },
            a && b,
        ))
    }

    /// A package output of `kind`, read where `holds` hold (at the top,
    /// nothing).
    fn output(
        &mut self,
        reader: &Reader<'_>,
        binding: &BodyOutput,
        kind: OutputKind,
        holds: &mut Vec<(usize, bool)>,
    ) -> Result<(Source, bool), PackageError> {
        match binding {
            BodyOutput::Output(r) => {
                let (i, out) = self.step_output(reader, r, holds)?;
                if out.kind != kind {
                    return Err(reader.fail("the step output's kind does not fit"));
                }
                Ok((Source::Step(i, r.output.clone()), true))
            }
            BodyOutput::Select { select } => {
                let label = format!("{} = select when {}", reader.name, select.when);
                self.select(reader, select, label, holds, |c, b, holds| {
                    c.output(reader, b, kind, holds)
                })
            }
        }
    }

    fn step(&mut self, registry: &Registry, step: &Step) -> Result<(), PackageError> {
        let package = self.package;
        let requires = &package.requires;
        if step.name.is_empty() || step.name.contains('/') {
            return Err(binding_error(
                &step.name,
                "",
                "a step name is one path segment",
            ));
        }
        if !requires.modules.contains(&step.module) {
            return Err(PackageError::UndeclaredDependency(step.module.clone()));
        }
        let module = registry
            .module(&step.module)
            .ok_or_else(|| PackageError::UnknownModule(step.module.clone()))?;
        let inner = module.interface();
        let index = self.steps.len();

        // The guard first: the step's parameters are computed under it.
        let guard = match (&step.when, &step.unless) {
            (None, None) => None,
            (Some(_), Some(_)) => {
                return Err(binding_error(
                    &step.name,
                    "when",
                    "a step has `when` or `unless`, not both",
                ));
            }
            (Some(source), None) | (None, Some(source)) => {
                self.require_guards()?;
                let (name, holds) = if step.when.is_some() {
                    ("when", true)
                } else {
                    ("unless", false)
                };
                let result =
                    self.logic
                        .condition(&step.name, name, format!("{name} = {source}"), source)?;
                Some(Guard {
                    result,
                    holds,
                    source: source.clone(),
                })
            }
        };
        let declares_logic = requires.capabilities.iter().any(|c| c.name == LOGIC);
        let mut params = Vec::with_capacity(step.params.len());
        for (name, binding) in &step.params {
            let decl = inner
                .params
                .iter()
                .find(|p| p.name == name.as_str())
                .ok_or_else(|| binding_error(&step.name, name, "no such parameter"))?;
            let arg = match binding {
                ParamBinding::Value(v) => {
                    let v = ParamValue::from(*v);
                    v.fits(decl.kind)
                        .map_err(|reason| binding_error(&step.name, name, reason))?;
                    Arg::Value(v)
                }
                ParamBinding::Param(p) => {
                    let outer = self
                        .interface
                        .params
                        .iter()
                        .find(|d| d.name == p.as_str())
                        .ok_or_else(|| {
                            binding_error(&step.name, name, "no such package parameter")
                        })?;
                    if !feeds(outer.kind, decl.kind) {
                        return Err(binding_error(
                            &step.name,
                            name,
                            "the package parameter's kind or range does not fit",
                        ));
                    }
                    Arg::Param(p.clone())
                }
                ParamBinding::Expr(source) => {
                    if !declares_logic {
                        return Err(PackageError::UndeclaredCapability(Capability::new(
                            LOGIC, 1,
                        )));
                    }
                    let ty = logic_type(decl.kind).ok_or_else(|| {
                        binding_error(
                            &step.name,
                            name,
                            "an expression makes only scalars, integers and flags",
                        )
                    })?;
                    let builder = self.logic.builder()?;
                    let added = match &guard {
                        None => builder.expression(source, ty),
                        // Computed only where the step runs.
                        Some(g) => builder.expression_when(source, ty, g.result, g.holds),
                    };
                    let i =
                        added.map_err(|error| expression_error(&step.name, name, source, error))?;
                    self.logic.expressions.push(Expression {
                        step: step.name.clone(),
                        label: format!("{name} = {source}"),
                        binds: Some((decl.kind, index)),
                    });
                    Arg::Derived(i)
                }
            };
            params.push((name.clone(), arg));
        }

        // The inputs, read where the step runs.
        let own: Vec<(usize, bool)> = guard.iter().map(|g| (g.result, g.holds)).collect();
        let mut inputs = Vec::with_capacity(step.inputs.len());
        for decl in &inner.inputs {
            let Some(binding) = step.inputs.get(decl.name.as_ref()) else {
                if decl.required {
                    return Err(binding_error(&step.name, &decl.name, "a required input"));
                }
                continue;
            };
            let reader = Reader {
                step: &step.name,
                name: &decl.name,
            };
            let (source, always) = self.input(&reader, binding, decl.kind, &mut own.clone())?;
            if decl.required && !always {
                return Err(reader.fail("a required input bound to an optional one"));
            }
            inputs.push((decl.name.to_string(), source));
        }
        if let Some(extra) = step
            .inputs
            .keys()
            .find(|k| !inner.inputs.iter().any(|d| d.name == k.as_str()))
        {
            return Err(binding_error(&step.name, extra, "no such input"));
        }

        // What its outputs are where it does not run: inputs passed
        // through, read under the other side of its guard.
        let mut otherwise = Vec::with_capacity(step.otherwise.len());
        if !step.otherwise.is_empty() {
            let Some(g) = &guard else {
                return Err(binding_error(
                    &step.name,
                    "otherwise",
                    "only a step with `when` or `unless` has an `otherwise`",
                ));
            };
            for (output, input) in &step.otherwise {
                let reader = Reader {
                    step: &step.name,
                    name: output,
                };
                let out = inner
                    .outputs
                    .iter()
                    .find(|o| o.name == output.as_str())
                    .ok_or_else(|| reader.fail("no such step output"))?;
                let decl = inner
                    .inputs
                    .iter()
                    .find(|d| d.name == input.as_str())
                    .ok_or_else(|| reader.fail("`otherwise` names no input of the step"))?;
                let binding = step
                    .inputs
                    .get(input)
                    .ok_or_else(|| reader.fail("`otherwise` passes through an unbound input"))?;
                if !output_fits(out.kind, decl.kind) {
                    return Err(reader.fail("`otherwise` passes through an input of another kind"));
                }
                let (source, always) = self.input(
                    &reader,
                    binding,
                    decl.kind,
                    &mut alloc::vec![(g.result, !g.holds)],
                )?;
                if !always {
                    return Err(
                        reader.fail("`otherwise` passes through an input that may be absent")
                    );
                }
                otherwise.push((output.clone(), source));
            }
        }

        self.steps.push(CompiledStep {
            name: step.name.clone(),
            id: inner.id.clone(),
            module: Arc::clone(module),
            guard,
            params,
            inputs,
            otherwise,
        });
        self.step_interfaces.push(inner);
        Ok(())
    }
}

fn compile(registry: &Registry, package: &Package) -> Result<CompiledPackage, PackageError> {
    let interface = package.interface.to_interface()?;
    let requires = &package.requires;
    // The engine must offer what the package expects, and the package must
    // expect what it uses.
    for c in &requires.capabilities {
        registry.supports(c)?;
    }
    let declares = |name: &str| requires.capabilities.iter().any(|c| c.name == name);
    if !declares(GRAPH) {
        return Err(PackageError::UndeclaredCapability(Capability::new(
            GRAPH, 1,
        )));
    }
    if interface
        .inputs
        .iter()
        .any(|i| matches!(i.kind, InputKind::Resource(_)))
        && !declares(RESOURCES)
    {
        return Err(PackageError::UndeclaredCapability(Capability::new(
            RESOURCES, 1,
        )));
    }
    if package
        .interface
        .outputs
        .iter()
        .any(|o| !o.semantics.channels.is_empty() || o.semantics.tiling.is_some())
        && !declares(SEMANTICS)
    {
        return Err(PackageError::UndeclaredCapability(Capability::new(
            SEMANTICS, 1,
        )));
    }
    unique(requires.modules.iter().map(|m| m.name.as_str()))?;
    unique(package.body.steps.iter().map(|s| s.name.as_str()))?;
    unique(package.presets.iter().map(|p| p.name.as_str()))?;
    for p in &package.presets {
        p.check(&interface)?;
    }

    let mut compiler = Compiler {
        package,
        interface: &interface,
        steps: Vec::with_capacity(package.body.steps.len()),
        step_interfaces: Vec::with_capacity(package.body.steps.len()),
        logic: Draft {
            interface: &interface,
            builder: None,
            expressions: Vec::new(),
            conditions: Vec::new(),
        },
        guards: requires
            .capabilities
            .iter()
            .any(|c| c.name == LOGIC && c.version >= LOGIC_GUARDS),
    };
    for step in &package.body.steps {
        compiler.step(registry, step)?;
    }

    let mut outputs = Vec::with_capacity(interface.outputs.len());
    for (decl, source) in interface.outputs.iter().zip(&package.interface.outputs) {
        let reader = Reader {
            step: "",
            name: &decl.name,
        };
        let binding = package
            .body
            .outputs
            .get(decl.name.as_ref())
            .ok_or_else(|| reader.fail("the body does not produce it"))?;
        let (from, _) = compiler.output(&reader, binding, decl.kind, &mut Vec::new())?;
        let channels = source.semantics.channel_ids()?;
        if decl.kind != OutputKind::Material
            && (!channels.is_empty() || source.semantics.tiling.is_some())
        {
            return Err(reader.fail("only material outputs have channels and tiling"));
        }
        outputs.push(CompiledOutput {
            name: decl.name.to_string(),
            source: from,
            channels,
            tiling: source.semantics.tiling.map(Tiling::from),
        });
    }
    if let Some(extra) = package
        .body
        .outputs
        .keys()
        .find(|k| !interface.outputs.iter().any(|o| o.name == k.as_str()))
    {
        return Err(PackageError::Output {
            name: extra.clone(),
            reason: "the interface does not declare it",
        });
    }
    let Compiler { steps, logic, .. } = compiler;
    let logic = logic.finish()?;

    // The source's canonical bytes cover its guards, `otherwise`s and
    // selects as they cover everything else in it.
    let mut words = alloc::vec![package.fingerprint()];
    for m in &requires.modules {
        words.extend([name_word(1, &m.name), u64::from(m.version)]);
    }
    // What the steps' identities do not name: a step that is itself a
    // compiled package can be redefined under its name and version.
    words.extend(steps.iter().map(|s| s.module.definition_fingerprint()));
    // What the expressions' text does not pin down: the program they
    // compiled to and the contract it runs under (tape format, `libm`).
    // Only a package with expressions adds these words, so every other
    // package keeps its fingerprint.
    if let Some(l) = &logic {
        words.extend(l.program.content_words());
    }
    Ok(CompiledPackage {
        interface,
        steps,
        outputs,
        presets: package.presets.clone(),
        logic,
        fingerprint: hash(0x636f_6d70, &words), // "comp"
    })
}

fn output_error(name: &str) -> ModuleError {
    ModuleError::body(ModuleErrorKind::Output(name.into()))
}

/// What `source` holds in an instance: `produced` holds every earlier
/// step's outputs, and `values` the program's results.
fn fetch(
    source: &Source,
    produced: &[Outputs],
    args: &Args,
    values: &[Value],
) -> Result<Option<Input>, ModuleError> {
    Ok(match source {
        Source::Step(i, output) => match produced[*i].get(output) {
            Some(Output::Material(m)) => Some(Input::Material(m.clone())),
            Some(Output::Map(m)) => Some(Input::Map(m.clone())),
            None => return Err(output_error(output)),
        },
        Source::Input(n) => args.input(n),
        Source::Select {
            condition,
            then,
            otherwise,
        } => {
            let taken = if matches!(values[*condition], Value::Flag(true)) {
                then
            } else {
                otherwise
            };
            return fetch(taken, produced, args, values);
        }
    })
}

/// An input passed on as an output: a material or a map.
fn pass(name: &str, input: Option<Input>) -> Result<Output, ModuleError> {
    match input {
        Some(Input::Material(m)) => Ok(Output::Material(m)),
        Some(Input::Map(m)) => Ok(Output::Map(m)),
        _ => Err(output_error(name)),
    }
}

impl Module for CompiledPackage {
    fn interface(&self) -> Interface {
        self.interface.clone()
    }

    /// [`CompiledPackage::fingerprint`]: the package is its data, which its
    /// name and version do not pin down while it is being written.
    fn definition_fingerprint(&self) -> u64 {
        self.fingerprint
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        // Every condition and expression, once, before any step: a value
        // that does not fit a step that runs fails the instance before
        // anything is built.
        let (values, derived, runs) = match &self.logic {
            Some(l) => {
                let values = l.run(cx, args)?;
                let runs: Vec<bool> = self
                    .steps
                    .iter()
                    .map(|s| {
                        s.guard.as_ref().is_none_or(
                            |g| matches!(values[g.result], Value::Flag(v) if v == g.holds),
                        )
                    })
                    .collect();
                let derived = l.derive(cx, &values, &runs)?;
                (values, derived, runs)
            }
            None => (Vec::new(), Vec::new(), alloc::vec![true; self.steps.len()]),
        };
        let mut produced: Vec<Outputs> = Vec::with_capacity(self.steps.len());
        for (step, runs) in self.steps.iter().zip(runs) {
            if !runs {
                cx.skip(&step.name, step.id.clone());
                let mut outputs = Outputs::new();
                for (name, source) in &step.otherwise {
                    let output = pass(name, fetch(source, &produced, args, &values)?)?;
                    outputs = outputs.with(name.clone(), output);
                }
                produced.push(outputs);
                continue;
            }
            let mut bind = Bind::new();
            for (name, arg) in &step.params {
                let value = match arg {
                    Arg::Value(v) => *v,
                    Arg::Param(p) => args.param(p),
                    // Bound like a literal: the step's instance key holds
                    // the value's bits, not the expression.
                    Arg::Derived(i) => derived[*i],
                };
                bind = bind.param(name, value);
            }
            for (name, source) in &step.inputs {
                if let Some(input) = fetch(source, &produced, args, &values)? {
                    bind = bind.input(name, input);
                }
            }
            produced.push(cx.instantiate(&*step.module, &step.name, bind)?);
        }
        let mut outputs = Outputs::new();
        for o in &self.outputs {
            let value = pass(&o.name, fetch(&o.source, &produced, args, &values)?)?;
            if let Output::Material(m) = &value {
                let bound = |c: ChannelId| match c {
                    ChannelId::Param(p) => m.param(p).is_some(),
                    ChannelId::Aux(a) => m.aux(a).is_some(),
                };
                if !o.channels.iter().all(|c| bound(*c)) {
                    return Err(output_error(&o.name));
                }
                // A tiling promise holds where the grid wraps.
                if let Some(t) = o.tiling {
                    let allowed = t.and(Tiling::of(cx.grid()));
                    if m.tiling().and(allowed) != allowed {
                        return Err(output_error(&o.name));
                    }
                }
            }
            outputs = outputs.with(o.name.clone(), value);
        }
        Ok(outputs)
    }
}
