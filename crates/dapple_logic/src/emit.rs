// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Typed trees to one verified tape function.
//!
//! Every operation is emitted where the source has it, operands left to
//! right, with nothing folded, reordered or fused: the tape computes what
//! the Rust that reads like the source computes. Literals are loaded once
//! each, at the top of the function, which changes no result.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use execution_tape::asm::{Asm, FunctionSig, Label, ProgramBuilder, Reg};
use execution_tape::codec::decode_all;
use execution_tape::program::ValueType;
use execution_tape::value::FuncId;
use execution_tape::verifier::VerifiedProgram;

use crate::check::{Function, Node, Typed};
use crate::error::{Error, ErrorKind};
use crate::syntax::{Binary, Unary};
use crate::{Span, Type};

/// The guest trap code for integer arithmetic that leaves `u32`.
pub(crate) const TRAP_OVERFLOW: u32 = 1;
/// The guest trap code for an integer `clamp` with bounds out of order.
pub(crate) const TRAP_CLAMP: u32 = 2;

/// The compiled function and what running it needs to know.
pub(crate) struct Emitted {
    pub(crate) program: VerifiedProgram,
    pub(crate) bytes: Vec<u8>,
    /// Each expression's code, as a range of program counters.
    pub(crate) ranges: Vec<(u32, u32)>,
    /// Instructions in the function: no run executes more, as nothing
    /// branches backward.
    pub(crate) instructions: u64,
}

pub(crate) const fn value_type(t: Type) -> ValueType {
    match t {
        Type::Scalar => ValueType::F32,
        Type::Integer => ValueType::U32,
        Type::Flag => ValueType::Bool,
    }
}

fn tape_error(e: impl core::fmt::Display) -> Error {
    Error::new(Span::default(), ErrorKind::Tape(e.to_string()))
}

/// A literal, by its bits.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Literal {
    Scalar(u32),
    Integer(u32),
    Flag(bool),
}

fn add(out: &mut Vec<Literal>, l: Literal) {
    if !out.contains(&l) {
        out.push(l);
    }
}

/// Every literal `t` loads, in first-use order, each once.
fn literals(t: &Typed, out: &mut Vec<Literal>) {
    match &t.node {
        Node::Scalar(v) => add(out, Literal::Scalar(v.to_bits())),
        Node::Integer(v) => add(out, Literal::Integer(*v)),
        Node::Flag(b) => add(out, Literal::Flag(*b)),
        Node::Input(_) => {}
        Node::Unary(_, a) => literals(a, out),
        Node::Binary(_, a, b) => {
            literals(a, out);
            literals(b, out);
        }
        Node::Cond(c, a, b) => {
            literals(c, out);
            literals(a, out);
            literals(b, out);
        }
        Node::Call(f, args) => {
            for a in args {
                literals(a, out);
            }
            if *f == Function::Smoothstep {
                for v in [0.0_f32, 1.0, 2.0, 3.0] {
                    add(out, Literal::Scalar(v.to_bits()));
                }
            }
        }
    }
}

struct Emitter {
    asm: Asm,
    inputs: Vec<Reg>,
    literals: Vec<(Literal, Reg)>,
}

impl Emitter {
    fn reg(&mut self, t: ValueType) -> Reg {
        self.asm.alloc_register(t)
    }

    fn place(&mut self, l: Label) -> Result<(), Error> {
        self.asm.place(l).map_err(|e| tape_error(format!("{e:?}")))
    }

    fn literal(&self, l: Literal) -> Reg {
        self.literals
            .iter()
            .find(|(k, _)| *k == l)
            .map(|(_, r)| *r)
            .expect("every literal is loaded at the top")
    }

    fn scalar(&self, v: f32) -> Reg {
        self.literal(Literal::Scalar(v.to_bits()))
    }

    /// Stops the run with guest trap `code` when `cond` holds.
    fn trap_if(&mut self, cond: Reg, code: u32) -> Result<(), Error> {
        let (bad, good) = (self.asm.label(), self.asm.label());
        self.asm.br(cond, bad, good);
        self.place(bad)?;
        self.asm.trap(code);
        self.place(good)
    }

    fn emit(&mut self, t: &Typed) -> Result<Reg, Error> {
        Ok(match &t.node {
            Node::Scalar(v) => self.literal(Literal::Scalar(v.to_bits())),
            Node::Integer(v) => self.literal(Literal::Integer(*v)),
            Node::Flag(b) => self.literal(Literal::Flag(*b)),
            Node::Input(i) => self.inputs[*i],
            Node::Unary(op, a) => {
                let a = self.emit(a)?;
                let d = self.reg(value_type(t.ty));
                match op {
                    Unary::Neg => self.asm.f32_neg(d, a),
                    Unary::Not => self.asm.bool_not(d, a),
                };
                d
            }
            Node::Binary(op @ (Binary::And | Binary::Or), a, b) => {
                // Evaluate `b` only when `a` does not decide.
                let a = self.emit(a)?;
                let d = self.reg(ValueType::Bool);
                self.asm.mov(d, a);
                let (rhs, join) = (self.asm.label(), self.asm.label());
                if *op == Binary::And {
                    self.asm.br(a, rhs, join);
                } else {
                    self.asm.br(a, join, rhs);
                }
                self.place(rhs)?;
                let b = self.emit(b)?;
                self.asm.mov(d, b).jmp(join);
                self.place(join)?;
                d
            }
            Node::Binary(op, a, b) => {
                let operands = a.ty;
                let (a, b) = (self.emit(a)?, self.emit(b)?);
                self.binary(*op, operands, a, b, value_type(t.ty))?
            }
            Node::Cond(c, a, b) => {
                let c = self.emit(c)?;
                let d = self.reg(value_type(t.ty));
                let (then, otherwise, join) =
                    (self.asm.label(), self.asm.label(), self.asm.label());
                self.asm.br(c, then, otherwise);
                self.place(then)?;
                let a = self.emit(a)?;
                self.asm.mov(d, a).jmp(join);
                self.place(otherwise)?;
                let b = self.emit(b)?;
                self.asm.mov(d, b).jmp(join);
                self.place(join)?;
                d
            }
            Node::Call(f, args) => {
                let operands = args[0].ty;
                let mut regs = Vec::with_capacity(args.len());
                for a in args {
                    regs.push(self.emit(a)?);
                }
                self.call(*f, operands, &regs)?
            }
        })
    }

    fn binary(
        &mut self,
        op: Binary,
        operands: Type,
        a: Reg,
        b: Reg,
        result: ValueType,
    ) -> Result<Reg, Error> {
        let d = self.reg(result);
        let asm = &mut self.asm;
        match (operands, op) {
            (Type::Scalar, Binary::Add) => asm.f32_add(d, a, b),
            (Type::Scalar, Binary::Sub) => asm.f32_sub(d, a, b),
            (Type::Scalar, Binary::Mul) => asm.f32_mul(d, a, b),
            (Type::Scalar, Binary::Div) => asm.f32_div(d, a, b),
            (Type::Scalar, Binary::Lt) => asm.f32_lt(d, a, b),
            (Type::Scalar, Binary::Gt) => asm.f32_gt(d, a, b),
            (Type::Scalar, Binary::Le) => asm.f32_le(d, a, b),
            (Type::Scalar, Binary::Ge) => asm.f32_ge(d, a, b),
            (Type::Scalar, Binary::Eq) => asm.f32_eq(d, a, b),
            (Type::Scalar, Binary::Ne) => asm.f32_ne(d, a, b),
            (Type::Integer, Binary::Add | Binary::Sub | Binary::Mul) => {
                let overflow = self.reg(ValueType::Bool);
                match op {
                    Binary::Add => self.asm.u32_add_ovf(d, overflow, a, b),
                    Binary::Sub => self.asm.u32_sub_ovf(d, overflow, a, b),
                    _ => self.asm.u32_mul_ovf(d, overflow, a, b),
                };
                self.trap_if(overflow, TRAP_OVERFLOW)?;
                return Ok(d);
            }
            // Traps on a zero divisor.
            (Type::Integer, Binary::Div) => asm.u32_div(d, a, b),
            (Type::Integer, Binary::Lt) => asm.u32_lt(d, a, b),
            (Type::Integer, Binary::Gt) => asm.u32_gt(d, a, b),
            (Type::Integer, Binary::Le) => asm.u32_le(d, a, b),
            (Type::Integer, Binary::Ge) => asm.u32_ge(d, a, b),
            (Type::Integer, Binary::Eq) => asm.u32_eq(d, a, b),
            (Type::Integer, Binary::Ne) => asm.u32_eq(d, a, b).bool_not(d, d),
            (Type::Flag, Binary::Eq) => asm.bool_xor(d, a, b).bool_not(d, d),
            (Type::Flag, Binary::Ne) => asm.bool_xor(d, a, b),
            (_, op) => {
                return Err(tape_error(format!(
                    "{} of {} reached the emitter",
                    op.symbol(),
                    operands.name()
                )));
            }
        };
        Ok(d)
    }

    fn call(&mut self, f: Function, operands: Type, x: &[Reg]) -> Result<Reg, Error> {
        let ty = if f == Function::ToScalar {
            ValueType::F32
        } else {
            value_type(operands)
        };
        let d = self.reg(ty);
        match (f, operands) {
            (Function::Min, Type::Scalar) => {
                self.asm.f32_min_num(d, x[0], x[1]);
            }
            (Function::Max, Type::Scalar) => {
                self.asm.f32_max_num(d, x[0], x[1]);
            }
            (Function::Min | Function::Max, Type::Integer) => {
                let first = self.reg(ValueType::Bool);
                if f == Function::Min {
                    self.asm.u32_le(first, x[0], x[1]);
                } else {
                    self.asm.u32_gt(first, x[0], x[1]);
                }
                self.asm.select(d, first, x[0], x[1]);
            }
            // Traps unless `lo <= hi`.
            (Function::Clamp, Type::Scalar) => {
                self.asm.f32_clamp(d, x[0], x[1], x[2]);
            }
            (Function::Clamp, Type::Integer) => {
                // `Ord::clamp`: bounds in order, then `lo` below, `hi` above.
                let bad = self.reg(ValueType::Bool);
                self.asm.u32_gt(bad, x[1], x[2]);
                self.trap_if(bad, TRAP_CLAMP)?;
                let (below, above, kept) = (
                    self.reg(ValueType::Bool),
                    self.reg(ValueType::Bool),
                    self.reg(ValueType::U32),
                );
                self.asm
                    .u32_lt(below, x[0], x[1])
                    .u32_gt(above, x[0], x[2])
                    .select(kept, above, x[2], x[0])
                    .select(d, below, x[1], kept);
            }
            (Function::Abs, _) => {
                self.asm.f32_abs(d, x[0]);
            }
            (Function::Floor, _) => {
                self.asm.f32_floor(d, x[0]);
            }
            (Function::Round, _) => {
                self.asm.f32_round_half_away(d, x[0]);
            }
            (Function::Sqrt, _) => {
                self.asm.f32_sqrt(d, x[0]);
            }
            (Function::Pow, _) => {
                self.asm.f32_pow(d, x[0], x[1]);
            }
            (Function::Exp, _) => {
                self.asm.f32_exp(d, x[0]);
            }
            (Function::Ln, _) => {
                self.asm.f32_ln(d, x[0]);
            }
            (Function::Sin, _) => {
                self.asm.f32_sin(d, x[0]);
            }
            (Function::Cos, _) => {
                self.asm.f32_cos(d, x[0]);
            }
            (Function::Atan2, _) => {
                self.asm.f32_atan2(d, x[0], x[1]);
            }
            (Function::Mix, _) => {
                // a + (b - a) * t
                let (span, step) = (self.reg(ValueType::F32), self.reg(ValueType::F32));
                self.asm
                    .f32_sub(span, x[1], x[0])
                    .f32_mul(step, span, x[2])
                    .f32_add(d, x[0], step);
            }
            (Function::Smoothstep, _) => self.smoothstep(d, x[0], x[1], x[2])?,
            (Function::ToScalar, _) => {
                self.asm.u32_to_f32(d, x[0]);
            }
            (f, t) => {
                return Err(tape_error(format!(
                    "{} of {} reached the emitter",
                    f.name(),
                    t.name()
                )));
            }
        }
        Ok(d)
    }

    /// `dapple_field`'s smoothstep:
    ///
    /// ```text
    /// let t = if e1 == e0 {
    ///     if x < e0 { 0.0 } else { 1.0 }
    /// } else {
    ///     ((x - e0) / (e1 - e0)).clamp(0.0, 1.0)
    /// };
    /// t * t * (3.0 - 2.0 * t)
    /// ```
    fn smoothstep(&mut self, d: Reg, e0: Reg, e1: Reg, x: Reg) -> Result<(), Error> {
        let (zero, one, two, three) = (
            self.scalar(0.0),
            self.scalar(1.0),
            self.scalar(2.0),
            self.scalar(3.0),
        );
        let t = self.reg(ValueType::F32);
        let same = self.reg(ValueType::Bool);
        let (step, ramp, join) = (self.asm.label(), self.asm.label(), self.asm.label());
        self.asm.f32_eq(same, e1, e0).br(same, step, ramp);

        self.place(step)?;
        let below = self.reg(ValueType::Bool);
        self.asm
            .f32_lt(below, x, e0)
            .select(t, below, zero, one)
            .jmp(join);

        self.place(ramp)?;
        let (rise, run, q) = (
            self.reg(ValueType::F32),
            self.reg(ValueType::F32),
            self.reg(ValueType::F32),
        );
        self.asm
            .f32_sub(rise, x, e0)
            .f32_sub(run, e1, e0)
            .f32_div(q, rise, run)
            .f32_clamp(t, q, zero, one)
            .jmp(join);

        self.place(join)?;
        let (tt, twice, rest) = (
            self.reg(ValueType::F32),
            self.reg(ValueType::F32),
            self.reg(ValueType::F32),
        );
        self.asm
            .f32_mul(tt, t, t)
            .f32_mul(twice, two, t)
            .f32_sub(rest, three, twice)
            .f32_mul(d, tt, rest);
        Ok(())
    }
}

/// Compiles `expressions` over `inputs` into one function: the inputs in
/// order as arguments, one result per expression, each run only where its
/// condition (an earlier flag result and the value it must have), if it
/// has one, holds, and its type's zero where not.
pub(crate) fn emit(
    inputs: &[(String, Type)],
    expressions: &[Typed],
    conditions: &[Option<(usize, bool)>],
) -> Result<Emitted, Error> {
    let mut pb = ProgramBuilder::new();
    pb.set_program_name("dapple_logic");
    let f = pb
        .declare_function(FunctionSig {
            arg_types: inputs.iter().map(|(_, t)| value_type(*t)).collect(),
            ret_types: expressions.iter().map(|e| value_type(e.ty)).collect(),
        })
        .map_err(|e| tape_error(format!("{e:?}")))?;
    let mut regs = Vec::with_capacity(inputs.len());
    for (i, (name, _)) in inputs.iter().enumerate() {
        let i = u32::try_from(i).map_err(|_| tape_error("too many inputs"))?;
        regs.push(Reg(i + 1));
        pb.set_function_input_name(f, i, name).map_err(tape_error)?;
    }
    let mut e = Emitter {
        asm: pb.function_asm(f).map_err(tape_error)?,
        inputs: regs,
        literals: Vec::new(),
    };

    let mut all = Vec::new();
    for x in expressions {
        literals(x, &mut all);
    }
    for l in all {
        let r = match l {
            Literal::Scalar(bits) => {
                let r = e.reg(ValueType::F32);
                e.asm.const_f32_bits(r, bits);
                r
            }
            Literal::Integer(v) => {
                let r = e.reg(ValueType::U32);
                e.asm.const_u32(r, v);
                r
            }
            Literal::Flag(b) => {
                let r = e.reg(ValueType::Bool);
                e.asm.const_bool(r, b);
                r
            }
        };
        e.literals.push((l, r));
    }

    let mut rets: Vec<Reg> = vec![];
    let mut ranges = Vec::with_capacity(expressions.len());
    for (x, condition) in expressions.iter().zip(conditions) {
        let start = e.asm.pc().0;
        let r = match *condition {
            None => e.emit(x)?,
            Some((c, holds)) => {
                let c = *rets
                    .get(c)
                    .ok_or_else(|| tape_error("a condition after its expression"))?;
                let d = e.reg(value_type(x.ty));
                let (run, skip, join) = (e.asm.label(), e.asm.label(), e.asm.label());
                if holds {
                    e.asm.br(c, run, skip);
                } else {
                    e.asm.br(c, skip, run);
                }
                e.place(run)?;
                let r = e.emit(x)?;
                e.asm.mov(d, r).jmp(join);
                e.place(skip)?;
                match x.ty {
                    Type::Scalar => e.asm.const_f32_bits(d, 0),
                    Type::Integer => e.asm.const_u32(d, 0),
                    Type::Flag => e.asm.const_bool(d, false),
                };
                e.asm.jmp(join);
                e.place(join)?;
                d
            }
        };
        rets.push(r);
        ranges.push((start, e.asm.pc().0));
    }
    e.asm.ret(Reg::EFFECT, &rets);
    pb.define_function(f, e.asm).map_err(tape_error)?;
    let program = pb.build_verified().map_err(tape_error)?;
    let bytes = program.program().encode();
    let p = program.program();
    let code = p
        .function(FuncId(0))
        .and_then(|func| func.bytecode(p).ok())
        .ok_or_else(|| tape_error("the function has no code"))?;
    let instructions = decode_all(code).map_err(tape_error)?.len() as u64;
    Ok(Emitted {
        program,
        bytes,
        ranges,
        instructions,
    })
}
