// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The execution artifact: a package checked against an engine and
//! resolved into a [`Module`].

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
    Args, Bind, Context, Input, InputKind, Interface, Module, ModuleError, ModuleErrorKind, Output,
    OutputKind, Outputs, ParamKind, ParamValue,
};
use dapple_material::{ChannelId, Tiling};

use crate::PackageError;
use crate::source::{Capability, InputBinding, ModuleRef, Package, ParamBinding, Preset, unique};

/// Module-graph bodies ([`crate::source::Body`]).
pub const GRAPH: &str = "dapple.graph";
/// Host-resolved resource inputs.
pub const RESOURCES: &str = "dapple.resources";
/// Output semantics checked on every instance.
pub const SEMANTICS: &str = "dapple.semantics";
/// Expression bindings ([`ParamBinding::Expr`]), in the language of
/// [`dapple_logic`] at its first version.
pub const LOGIC: &str = "dapple.logic";

/// What an engine offers packages: modules by identity, and capabilities
/// by name and version.
///
/// [`Registry::new`] offers this crate's own capabilities ([`GRAPH`],
/// [`RESOURCES`], [`SEMANTICS`] and [`LOGIC`], each at version 1) and no
/// modules; a
/// host registers the modules it links, and may withdraw capabilities it
/// cannot serve (an engine without an image host has no resources).
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
            capabilities: [GRAPH, RESOURCES, SEMANTICS, LOGIC]
                .map(|n| Capability::new(n, 1))
                .into(),
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
    /// The package's expression of this index.
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

/// One expression binding: where it binds, and what.
#[derive(Clone, Debug)]
struct Binding {
    step: String,
    name: String,
    source: String,
    kind: ParamKind,
}

/// A package's expressions, compiled into one program.
#[derive(Clone, Debug)]
struct Logic {
    program: Program,
    /// The package parameters the program reads, in argument order.
    inputs: Vec<String>,
    /// Each expression's binding, in the program's result order.
    bindings: Vec<Binding>,
}

impl Logic {
    /// Every expression's value under `args`, each checked against the
    /// parameter it binds.
    fn derive(&self, cx: &Context<'_>, args: &Args) -> Result<Vec<ParamValue>, ModuleError> {
        let fail = |i: usize, reason: &'static str| {
            let b = &self.bindings[i];
            ModuleError {
                path: if cx.path().is_empty() {
                    b.step.clone()
                } else {
                    format!("{}/{}", cx.path(), b.step)
                },
                kind: Box::new(ModuleErrorKind::InvalidParam {
                    name: format!("{} = {}", b.name, b.source),
                    reason,
                }),
            }
        };
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
        let values = self.program.run(&inputs).map_err(|e| match e.expression {
            Some(i) => fail(i, e.kind.reason()),
            None => ModuleError::body(ModuleErrorKind::Body(e.kind.reason())),
        })?;
        values
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                let v = match v {
                    Value::Scalar(x) => ParamValue::Scalar(x),
                    Value::Integer(x) => ParamValue::Integer(x),
                    Value::Flag(x) => ParamValue::Flag(x),
                };
                v.fits(self.bindings[i].kind)
                    .map_err(|reason| fail(i, reason))?;
                Ok(v)
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
enum Source {
    Step(usize, String),
    Input(String),
}

#[derive(Clone)]
struct CompiledStep {
    name: String,
    module: Arc<dyn Module>,
    params: Vec<(String, Arg)>,
    inputs: Vec<(String, Source)>,
}

#[derive(Clone, Debug)]
struct CompiledOutput {
    name: String,
    step: usize,
    output: String,
    channels: Vec<ChannelId>,
    tiling: Option<Tiling>,
}

/// A package compiled against a [`Registry`]: the execution artifact, kept
/// apart from the editable [`Package`] it came from.
///
/// Compiling resolves every name the source uses (steps to registered
/// modules, bindings to parameters, inputs and earlier outputs), checks
/// every binding's kind and range against the module it feeds, and orders
/// nothing it does not have to: the steps run in their written order. The
/// result is a [`Module`], instantiated like any other, whose outputs are
/// checked against their declared semantics on every instance.
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
                    .map(|s| (s.name.as_str(), s.module.interface().id))
                    .collect::<Vec<_>>(),
            )
            .field("outputs", &self.outputs)
            .field(
                "expressions",
                &self.logic.as_ref().map_or(0, |l| l.bindings.len()),
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

    /// The package's expression bindings compiled into one program, if it
    /// has any: run once per instance, before the steps. For inspection:
    /// its size, its bytes, its cost.
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

    let mut steps: Vec<CompiledStep> = Vec::with_capacity(package.body.steps.len());
    let mut step_interfaces: Vec<Interface> = Vec::new();
    // A package's expressions compile together into one program, begun at
    // the first: a package without them does no work for them.
    let mut logic: Option<(LogicBuilder, Vec<String>, Vec<Binding>)> = None;
    for step in &package.body.steps {
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
                    let outer = interface
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
                    if !declares(LOGIC) {
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
                    let (builder, _, bindings) = match &mut logic {
                        Some(l) => l,
                        None => {
                            // Expressions read the package's scalars,
                            // integers and flags, in the interface's order.
                            let mut b = LogicBuilder::new();
                            let mut inputs = Vec::new();
                            for p in &interface.params {
                                if let Some(t) = logic_type(p.kind) {
                                    b.input(&p.name, t).map_err(PackageError::Logic)?;
                                    inputs.push(p.name.to_string());
                                }
                            }
                            logic.insert((b, inputs, Vec::new()))
                        }
                    };
                    let index = builder.expression(source, ty).map_err(|error| {
                        PackageError::Expression {
                            step: step.name.clone(),
                            name: name.clone(),
                            source: source.clone(),
                            error: Box::new(error),
                        }
                    })?;
                    bindings.push(Binding {
                        step: step.name.clone(),
                        name: name.clone(),
                        source: source.clone(),
                        kind: decl.kind,
                    });
                    Arg::Derived(index)
                }
            };
            params.push((name.clone(), arg));
        }
        let mut inputs = Vec::with_capacity(step.inputs.len());
        for decl in &inner.inputs {
            let Some(binding) = step.inputs.get(decl.name.as_ref()) else {
                if decl.required {
                    return Err(binding_error(&step.name, &decl.name, "a required input"));
                }
                continue;
            };
            let source = match binding {
                InputBinding::Output(r) => {
                    let i = package
                        .body
                        .steps
                        .iter()
                        .take(steps.len())
                        .position(|s| s.name == r.step)
                        .ok_or_else(|| {
                            binding_error(&step.name, &decl.name, "not an earlier step")
                        })?;
                    let out = step_interfaces[i]
                        .outputs
                        .iter()
                        .find(|o| o.name == r.output.as_str())
                        .ok_or_else(|| {
                            binding_error(&step.name, &decl.name, "no such step output")
                        })?;
                    if !output_fits(out.kind, decl.kind) {
                        return Err(binding_error(
                            &step.name,
                            &decl.name,
                            "the output's kind does not fit",
                        ));
                    }
                    Source::Step(i, r.output.clone())
                }
                InputBinding::Input(n) => {
                    let outer = interface
                        .inputs
                        .iter()
                        .find(|d| d.name == n.as_str())
                        .ok_or_else(|| {
                            binding_error(&step.name, &decl.name, "no such package input")
                        })?;
                    if !input_fits(outer.kind, decl.kind) {
                        return Err(binding_error(
                            &step.name,
                            &decl.name,
                            "the package input's kind does not fit",
                        ));
                    }
                    if decl.required && !outer.required {
                        return Err(binding_error(
                            &step.name,
                            &decl.name,
                            "a required input bound to an optional one",
                        ));
                    }
                    Source::Input(n.clone())
                }
            };
            inputs.push((decl.name.to_string(), source));
        }
        if let Some(extra) = step
            .inputs
            .keys()
            .find(|k| !inner.inputs.iter().any(|d| d.name == k.as_str()))
        {
            return Err(binding_error(&step.name, extra, "no such input"));
        }
        steps.push(CompiledStep {
            name: step.name.clone(),
            module: Arc::clone(module),
            params,
            inputs,
        });
        step_interfaces.push(inner);
    }

    let mut outputs = Vec::with_capacity(interface.outputs.len());
    for (decl, source) in interface.outputs.iter().zip(&package.interface.outputs) {
        let fail = |reason| PackageError::Output {
            name: decl.name.to_string(),
            reason,
        };
        let r = package
            .body
            .outputs
            .get(decl.name.as_ref())
            .ok_or_else(|| fail("the body does not produce it"))?;
        let step = package
            .body
            .steps
            .iter()
            .position(|s| s.name == r.step)
            .ok_or_else(|| fail("no such step"))?;
        let inner = step_interfaces[step]
            .outputs
            .iter()
            .find(|o| o.name == r.output.as_str())
            .ok_or_else(|| fail("no such step output"))?;
        if inner.kind != decl.kind {
            return Err(fail("the step output's kind does not fit"));
        }
        let channels = source.semantics.channel_ids()?;
        if decl.kind != OutputKind::Material
            && (!channels.is_empty() || source.semantics.tiling.is_some())
        {
            return Err(fail("only material outputs have channels and tiling"));
        }
        outputs.push(CompiledOutput {
            name: decl.name.to_string(),
            step,
            output: r.output.clone(),
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

    let logic = match logic {
        Some((builder, inputs, bindings)) => Some(Logic {
            program: builder.build().map_err(PackageError::Logic)?,
            inputs,
            bindings,
        }),
        None => None,
    };

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
        // Every expression, once, before any step: a value that does not
        // fit fails the instance before anything is built.
        let derived = match &self.logic {
            Some(l) => l.derive(cx, args)?,
            None => Vec::new(),
        };
        let mut produced: Vec<Outputs> = Vec::with_capacity(self.steps.len());
        for step in &self.steps {
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
                let input = match source {
                    Source::Step(i, output) => match produced[*i].get(output) {
                        Some(Output::Material(m)) => Some(Input::Material(m.clone())),
                        Some(Output::Map(m)) => Some(Input::Map(m.clone())),
                        None => return Err(output_error(output)),
                    },
                    Source::Input(n) => args.input(n),
                };
                if let Some(input) = input {
                    bind = bind.input(name, input);
                }
            }
            produced.push(cx.instantiate(&*step.module, &step.name, bind)?);
        }
        let mut outputs = Outputs::new();
        for o in &self.outputs {
            let value = produced[o.step]
                .get(&o.output)
                .cloned()
                .ok_or_else(|| output_error(&o.name))?;
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
