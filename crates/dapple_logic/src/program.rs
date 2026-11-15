// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Building a program from expressions, and running it.

use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt;

use execution_tape::host::{Host, HostContext, HostError, SigHash, ValueRef};
use execution_tape::program::{VERSION_MAJOR, VERSION_MINOR};
use execution_tape::trace::TraceMask;
use execution_tape::value::{FuncId, FuncRef, UnitId, Value as TapeValue};
use execution_tape::verifier::VerifiedProgram;
use execution_tape::vm::{ExecutionContext, Limits, Trap, Vm};

use crate::check::{Scope, Typed};
use crate::emit::{TRAP_CLAMP, TRAP_OVERFLOW, emit};
use crate::error::{Error, ErrorKind, RunError, RunErrorKind};
use crate::syntax::parse;
use crate::{Span, Type, Value};

/// The evaluation contract: what a program's results depend on besides
/// its encoded bytes.
///
/// From the high bits down: `execution_tape`'s wire format version (major,
/// minor; 16 bits each), the `libm` release its math is defined as (0.2.16,
/// 8 bits per part), and this crate's version of what the language means
/// (8 bits). A fingerprint of a program's behavior is this word and the
/// bytes ([`Program::content_words`]).
///
/// `execution_tape` pins `libm` to exactly 0.2.16 and does not export the
/// version, so it is written here: moving the `execution_tape` revision
/// means checking its pin against this word.
pub const CONTRACT: u64 =
    ((VERSION_MAJOR as u64) << 48) | ((VERSION_MINOR as u64) << 32) | (LIBM << 8) | LANGUAGE;

/// `libm` 0.2.16, a byte per part.
const LIBM: u64 = 0x00_02_10;
/// The meaning of the language, versioned: bump when the same source
/// should compute something else.
const LANGUAGE: u64 = 1;

/// Inputs, then expressions, checked as they are added: see the
/// [crate docs](crate).
#[derive(Clone, Debug, Default)]
pub struct Builder {
    inputs: Vec<(String, Type)>,
    expressions: Vec<Typed>,
}

impl Builder {
    /// No inputs and no expressions.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares input `name` of type `ty`: the program's next argument.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::DuplicateInput`] for a name already declared.
    pub fn input(&mut self, name: &str, ty: Type) -> Result<usize, Error> {
        if self.inputs.iter().any(|(n, _)| n == name) {
            return Err(Error::new(
                Span::default(),
                ErrorKind::DuplicateInput(name.to_string()),
            ));
        }
        self.inputs.push((name.to_string(), ty));
        Ok(self.inputs.len() - 1)
    }

    /// Adds `source`, an expression over the inputs declared so far whose
    /// result is a `ty`: the program's next result. Returns its index.
    ///
    /// # Errors
    ///
    /// The first thing wrong with it, with where it is in `source`.
    pub fn expression(&mut self, source: &str, ty: Type) -> Result<usize, Error> {
        let tree = parse(source)?;
        let typed = Scope {
            inputs: &self.inputs,
        }
        .elaborate(&tree, Some(ty))?;
        self.expressions.push(typed);
        Ok(self.expressions.len() - 1)
    }

    /// Compiles every expression into one verified tape program.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Tape`] if the tape does not build or verify, which is a
    /// bug in this crate.
    pub fn build(self) -> Result<Program, Error> {
        let emitted = emit(&self.inputs, &self.expressions)?;
        Ok(Program {
            program: Arc::new(emitted.program),
            bytes: Arc::from(emitted.bytes),
            inputs: self.inputs.iter().map(|(_, t)| *t).collect(),
            outputs: self.expressions.iter().map(|e| e.ty).collect(),
            ranges: emitted.ranges.into(),
            instructions: emitted.instructions,
            fuel: emitted.instructions,
            parked: RefCell::new(None),
        })
    }
}

/// A host for a program that calls none: the VM needs one.
#[derive(Debug)]
struct NoHost;

impl Host for NoHost {
    fn call(
        &mut self,
        _symbol: &str,
        _sig_hash: SigHash,
        _args: &[ValueRef<'_>],
        _rets: &mut [TapeValue],
        _ctx: HostContext<'_, '_>,
    ) -> Result<u64, HostError> {
        Err(HostError::UnknownSymbol)
    }
}

/// A VM with the program installed, and buffers, kept between runs.
struct Parked {
    vm: Vm<NoHost>,
    unit: UnitId,
    scratch: ExecutionContext,
    args: Vec<TapeValue>,
    results: Vec<TapeValue>,
}

/// Compiled expressions: one verified tape function from the inputs to
/// one result per expression. Built by a [`Builder`].
///
/// [`Self::run`] runs it on a VM kept between runs, under a fuel budget.
pub struct Program {
    program: Arc<VerifiedProgram>,
    bytes: Arc<[u8]>,
    inputs: Vec<Type>,
    outputs: Vec<Type>,
    ranges: Arc<[(u32, u32)]>,
    instructions: u64,
    fuel: u64,
    parked: RefCell<Option<Parked>>,
}

impl Clone for Program {
    /// The same program, sharing its code; it parks a VM of its own.
    fn clone(&self) -> Self {
        Self {
            program: Arc::clone(&self.program),
            bytes: Arc::clone(&self.bytes),
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            ranges: Arc::clone(&self.ranges),
            instructions: self.instructions,
            fuel: self.fuel,
            parked: RefCell::new(None),
        }
    }
}

impl fmt::Debug for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Program")
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .field("bytes", &self.bytes.len())
            .field("instructions", &self.instructions)
            .field("fuel", &self.fuel)
            .finish_non_exhaustive()
    }
}

impl Program {
    /// The inputs' types, in argument order.
    #[must_use]
    pub fn inputs(&self) -> &[Type] {
        &self.inputs
    }

    /// The results' types, one per expression in the order added.
    #[must_use]
    pub fn outputs(&self) -> &[Type] {
        &self.outputs
    }

    /// The encoded tape program. The same expressions over the same inputs
    /// always encode to the same bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The verified tape program.
    #[must_use]
    pub fn tape(&self) -> &VerifiedProgram {
        &self.program
    }

    /// What a fingerprint of the program's behavior takes: [`CONTRACT`],
    /// the bytes' length, then the bytes as little-endian words, the last
    /// padded with zeros.
    pub fn content_words(&self) -> impl Iterator<Item = u64> + '_ {
        let words = self.bytes.chunks(8).map(|c| {
            let mut w = [0_u8; 8];
            w[..c.len()].copy_from_slice(c);
            u64::from_le_bytes(w)
        });
        [CONTRACT, self.bytes.len() as u64].into_iter().chain(words)
    }

    /// Instructions in the program. Nothing branches backward, so no run
    /// executes more; a run executes fewer where `?:`, `&&` or `||` skip
    /// code.
    #[must_use]
    pub const fn instructions(&self) -> u64 {
        self.instructions
    }

    /// The fuel a run may spend, one per instruction executed. At first
    /// [`Self::instructions`], which every run fits in.
    #[must_use]
    pub const fn fuel(&self) -> u64 {
        self.fuel
    }

    /// Sets the fuel a run may spend.
    pub fn set_fuel(&mut self, fuel: u64) {
        self.fuel = fuel;
        let limits = self.limits();
        if let Some(parked) = self.parked.get_mut() {
            parked.vm.set_limits(limits);
        }
    }

    fn limits(&self) -> Limits {
        Limits {
            fuel: self.fuel,
            max_call_depth: 1,
            max_host_calls: 0,
            ..Limits::default()
        }
    }

    fn fresh(&self) -> Result<Parked, RunError> {
        let mut vm = Vm::new(NoHost, self.limits());
        let unit = vm
            .install(Arc::clone(&self.program))
            .map_err(|e| RunError {
                expression: None,
                kind: RunErrorKind::Trap(alloc::format!("{e:?}")),
            })?;
        Ok(Parked {
            vm,
            unit,
            scratch: ExecutionContext::new(),
            args: Vec::with_capacity(self.inputs.len()),
            results: Vec::with_capacity(self.outputs.len()),
        })
    }

    /// Runs the program on `inputs`, one value per input in order, and
    /// returns one value per expression.
    ///
    /// # Errors
    ///
    /// [`RunErrorKind::Inputs`] for inputs of the wrong count or types;
    /// otherwise what stopped the run, naming the expression it was in.
    pub fn run(&self, inputs: &[Value]) -> Result<Vec<Value>, RunError> {
        if inputs.len() != self.inputs.len()
            || inputs.iter().zip(&self.inputs).any(|(v, t)| v.ty() != *t)
        {
            return Err(RunError {
                expression: None,
                kind: RunErrorKind::Inputs,
            });
        }
        // A run cannot re-enter (the program calls nothing), so the parked
        // VM is free unless a caller shares this program across a run of
        // its own; then the run takes a fresh one.
        let taken = self.parked.try_borrow_mut().ok().and_then(|mut p| p.take());
        let mut parked = match taken {
            Some(p) => p,
            None => self.fresh()?,
        };
        parked.args.clear();
        parked.args.extend(inputs.iter().map(|v| match *v {
            Value::Scalar(x) => TapeValue::F32(x),
            Value::Integer(n) => TapeValue::U32(n),
            Value::Flag(b) => TapeValue::Bool(b),
        }));
        parked.results.clear();
        let Parked {
            vm,
            unit,
            scratch,
            args,
            results,
        } = &mut parked;
        let ran = vm.run_with_ctx(
            scratch,
            FuncRef::new(*unit, FuncId(0)),
            args,
            results,
            TraceMask::NONE,
            None,
            None,
        );
        if let Err(trap) = ran {
            // The VM is dropped rather than trusted after a trap.
            let expression = self
                .ranges
                .iter()
                .position(|&(start, end)| trap.pc.0 >= start && trap.pc.0 < end);
            let kind = match trap.trap {
                Trap::FuelExceeded => RunErrorKind::OutOfFuel,
                Trap::TrapCode(TRAP_OVERFLOW) => RunErrorKind::Overflow,
                Trap::TrapCode(TRAP_CLAMP) | Trap::ClampBoundsInvalid => RunErrorKind::ClampBounds,
                Trap::DivByZero => RunErrorKind::DivisionByZero,
                _ => RunErrorKind::Trap(trap.to_string()),
            };
            return Err(RunError { expression, kind });
        }
        if results.len() != self.outputs.len() {
            return Err(RunError {
                expression: None,
                kind: RunErrorKind::Trap(alloc::format!("returned {} values", results.len())),
            });
        }
        let mut out = Vec::with_capacity(results.len());
        for (v, t) in results.iter().zip(&self.outputs) {
            out.push(match (v, t) {
                (TapeValue::F32(x), Type::Scalar) => Value::Scalar(*x),
                (TapeValue::U32(n), Type::Integer) => Value::Integer(*n),
                (TapeValue::Bool(b), Type::Flag) => Value::Flag(*b),
                (other, _) => {
                    return Err(RunError {
                        expression: None,
                        kind: RunErrorKind::Trap(alloc::format!("returned {other:?}")),
                    });
                }
            });
        }
        if let Ok(mut slot) = self.parked.try_borrow_mut() {
            *slot = Some(parked);
        }
        Ok(out)
    }
}
