// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The reference check: every operator and function, run on the tape,
//! against the same expression written in Rust over `libm` (as dapple's
//! native modules compute), bit for bit, over sweeps that include signed
//! zeros, subnormals, huge values, negative and odd arguments, infinities
//! and NaN-producing inputs. A NaN must be a NaN; its bits are not
//! compared.

extern crate std;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use std::println;

use crate::{Builder, Program, RunErrorKind, Type, Value};

/// The inputs every case's program declares, whichever it reads.
#[derive(Copy, Clone, Debug, Default)]
struct In {
    a: f32,
    b: f32,
    c: f32,
    n: u32,
    m: u32,
    k: u32,
    p: bool,
    q: bool,
}

impl In {
    fn values(self) -> [Value; 8] {
        [
            Value::Scalar(self.a),
            Value::Scalar(self.b),
            Value::Scalar(self.c),
            Value::Integer(self.n),
            Value::Integer(self.m),
            Value::Integer(self.k),
            Value::Flag(self.p),
            Value::Flag(self.q),
        ]
    }
}

/// Which inputs a case sweeps.
#[derive(Copy, Clone, Debug)]
enum Sweep {
    /// `a`.
    One,
    /// `a`, `b`.
    Two,
    /// `a`, `b`, `c`.
    Three,
    /// `n`, `m`, `k`.
    Integers,
    /// `p`, `q`, with `a`, `b`.
    Flags,
}

type Reference = fn(In) -> Result<Value, RunErrorKind>;

struct Case {
    source: &'static str,
    ty: Type,
    sweep: Sweep,
    rust: Reference,
}

fn s(v: f32) -> Result<Value, RunErrorKind> {
    Ok(Value::Scalar(v))
}

fn i(v: Option<u32>, e: RunErrorKind) -> Result<Value, RunErrorKind> {
    v.map(Value::Integer).ok_or(e)
}

fn f(v: bool) -> Result<Value, RunErrorKind> {
    Ok(Value::Flag(v))
}

/// `f32::min`/`f32::max`, with the order of zeros the language promises
/// (`-0.0` below `+0.0`) where Rust leaves it open.
fn zeros(a: f32, b: f32, below: bool) -> Option<f32> {
    (a == 0.0 && b == 0.0).then(|| {
        let negative = if below {
            a.is_sign_negative() || b.is_sign_negative()
        } else {
            a.is_sign_negative() && b.is_sign_negative()
        };
        if negative { -0.0 } else { 0.0 }
    })
}

/// `dapple_field`'s smoothstep, as `dapple_field/src/scoped.rs` writes it.
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = if e1 == e0 {
        if x < e0 { 0.0 } else { 1.0 }
    } else {
        ((x - e0) / (e1 - e0)).clamp(0.0, 1.0)
    };
    t * t * (3.0 - 2.0 * t)
}

/// `f32::clamp`, which panics where the tape stops with an error.
fn clamp(x: f32, lo: f32, hi: f32) -> Result<Value, RunErrorKind> {
    if lo <= hi {
        s(x.clamp(lo, hi))
    } else {
        Err(RunErrorKind::ClampBounds)
    }
}

#[expect(clippy::too_many_lines, reason = "one line per case")]
fn cases() -> Vec<Case> {
    use RunErrorKind::{ClampBounds, DivisionByZero, Overflow};
    use Sweep::{Flags, Integers, One, Three, Two};
    use Type::{Flag, Integer, Scalar};
    let c = |source, ty, sweep, rust: Reference| Case {
        source,
        ty,
        sweep,
        rust,
    };
    alloc::vec![
        // Arithmetic and its grouping.
        c("a + b", Scalar, Two, |x| s(x.a + x.b)),
        c("a - b", Scalar, Two, |x| s(x.a - x.b)),
        c("a * b", Scalar, Two, |x| s(x.a * x.b)),
        c("a / b", Scalar, Two, |x| s(x.a / x.b)),
        c("-a", Scalar, One, |x| s(-x.a)),
        c("a - b - c", Scalar, Three, |x| s(x.a - x.b - x.c)),
        c("a - (b - c)", Scalar, Three, |x| s(x.a - (x.b - x.c))),
        c("a / b * c", Scalar, Three, |x| s(x.a / x.b * x.c)),
        c("-a * b", Scalar, Two, |x| s(-x.a * x.b)),
        c("a + b * c", Scalar, Three, |x| s(x.a + x.b * x.c)),
        c("a * b + c", Scalar, Three, |x| s(x.a * x.b + x.c)),
        c("a * b - a * c", Scalar, Three, |x| s(x.a * x.b - x.a * x.c)),
        // Literals, read as Rust reads them.
        c("a * 0.1", Scalar, One, |x| s(x.a * 0.1)),
        c("a + 1e-3", Scalar, One, |x| s(x.a + 1e-3)),
        c("a * 16777217", Scalar, One, |x| s(x.a * 16_777_216.0)),
        c("a * 3.4028235e38", Scalar, One, |x| s(x.a * f32::MAX)),
        c("a * 1e-45", Scalar, One, |x| s(x.a * 1e-45)),
        // Comparisons.
        c("a < b", Flag, Two, |x| f(x.a < x.b)),
        c("a > b", Flag, Two, |x| f(x.a > x.b)),
        c("a <= b", Flag, Two, |x| f(x.a <= x.b)),
        c("a >= b", Flag, Two, |x| f(x.a >= x.b)),
        c("a == b", Flag, Two, |x| f(x.a == x.b)),
        c("a != b", Flag, Two, |x| f(x.a != x.b)),
        // Functions.
        c("min(a, b)", Scalar, Two, |x| {
            s(zeros(x.a, x.b, true).unwrap_or(x.a.min(x.b)))
        }),
        c("max(a, b)", Scalar, Two, |x| {
            s(zeros(x.a, x.b, false).unwrap_or(x.a.max(x.b)))
        }),
        c("clamp(a, b, c)", Scalar, Three, |x| clamp(x.a, x.b, x.c)),
        c("clamp(a, 0.0, 1.0)", Scalar, One, |x| clamp(x.a, 0.0, 1.0)),
        c("abs(a)", Scalar, One, |x| s(libm::fabsf(x.a))),
        c("floor(a)", Scalar, One, |x| s(libm::floorf(x.a))),
        c("round(a)", Scalar, One, |x| s(libm::roundf(x.a))),
        c("sqrt(a)", Scalar, One, |x| s(libm::sqrtf(x.a))),
        c("pow(a, b)", Scalar, Two, |x| s(libm::powf(x.a, x.b))),
        c("exp(a)", Scalar, One, |x| s(libm::expf(x.a))),
        c("ln(a)", Scalar, One, |x| s(libm::logf(x.a))),
        c("sin(a)", Scalar, One, |x| s(libm::sinf(x.a))),
        c("cos(a)", Scalar, One, |x| s(libm::cosf(x.a))),
        c("atan2(a, b)", Scalar, Two, |x| s(libm::atan2f(x.a, x.b))),
        c(
            "mix(a, b, c)",
            Scalar,
            Three,
            |x| s(x.a + (x.b - x.a) * x.c)
        ),
        c("smoothstep(a, b, c)", Scalar, Three, |x| {
            s(smoothstep(x.a, x.b, x.c))
        }),
        c("smoothstep(a, a, b)", Scalar, Two, |x| {
            s(smoothstep(x.a, x.a, x.b))
        }),
        // Formulas from dapple's modules, as they write them.
        c("0.5 + (0.05 - 0.5) * a", Scalar, One, |x| {
            s(0.5 + (0.05 - 0.5) * x.a)
        }),
        c("pow(clamp(a, 0.0, 1.0), 10.0)", Scalar, One, |x| {
            s(libm::powf(x.a.clamp(0.0, 1.0), 10.0))
        }),
        c(
            "0.85 * pow(smoothstep(0.2, 1.0, a), 1.2)",
            Scalar,
            One,
            |x| { s(0.85 * libm::powf(smoothstep(0.2, 1.0, x.a), 1.2)) }
        ),
        c("max(a, 0.6 * b * b)", Scalar, Two, |x| {
            let t = 0.6 * x.b * x.b;
            s(zeros(x.a, t, false).unwrap_or(x.a.max(t)))
        }),
        c("min(a + 0.9 * b, 1.0)", Scalar, Two, |x| {
            s((x.a + 0.9 * x.b).min(1.0))
        }),
        c("sqrt(max(sin(3.1415927 * a), 0.0))", Scalar, One, |x| {
            s(libm::sqrtf(
                libm::sinf(core::f32::consts::PI * x.a).max(0.0),
            ))
        }),
        // Integers: checked arithmetic, as Rust's `checked_*`.
        c("n + m", Integer, Integers, |x| i(
            x.n.checked_add(x.m),
            Overflow
        )),
        c("n - m", Integer, Integers, |x| i(
            x.n.checked_sub(x.m),
            Overflow
        )),
        c("n * m", Integer, Integers, |x| i(
            x.n.checked_mul(x.m),
            Overflow
        )),
        c("n / m", Integer, Integers, |x| {
            i(x.n.checked_div(x.m), DivisionByZero)
        }),
        c("min(n, m)", Integer, Integers, |x| Ok(Value::Integer(
            x.n.min(x.m)
        ))),
        c("max(n, m)", Integer, Integers, |x| Ok(Value::Integer(
            x.n.max(x.m)
        ))),
        c("clamp(n, m, k)", Integer, Integers, |x| {
            if x.m <= x.k {
                Ok(Value::Integer(x.n.clamp(x.m, x.k)))
            } else {
                Err(ClampBounds)
            }
        }),
        c("n + 2", Integer, Integers, |x| i(
            x.n.checked_add(2),
            Overflow
        )),
        #[expect(clippy::cast_precision_loss, reason = "the conversion under test")]
        c("scalar(n)", Scalar, Integers, |x| s(x.n as f32)),
        #[expect(clippy::cast_precision_loss, reason = "the conversion under test")]
        c("scalar(n) * 0.5 + a", Scalar, Integers, |x| s(x.n as f32
            * 0.5
            + x.a)),
        c("n < m", Flag, Integers, |x| f(x.n < x.m)),
        c("n > m", Flag, Integers, |x| f(x.n > x.m)),
        c("n <= m", Flag, Integers, |x| f(x.n <= x.m)),
        c("n >= m", Flag, Integers, |x| f(x.n >= x.m)),
        c("n == m", Flag, Integers, |x| f(x.n == x.m)),
        c("n != m", Flag, Integers, |x| f(x.n != x.m)),
        // Flags, `?:` and short circuits.
        c("p && q", Flag, Flags, |x| f(x.p && x.q)),
        c("p || q", Flag, Flags, |x| f(x.p || x.q)),
        c("!p", Flag, Flags, |x| f(!x.p)),
        c("p == q", Flag, Flags, |x| f(x.p == x.q)),
        c("p != q", Flag, Flags, |x| f(x.p != x.q)),
        c("p ? a : b", Scalar, Flags, |x| s(if x.p {
            x.a
        } else {
            x.b
        })),
        c("a < b ? a - b : b - a", Scalar, Two, |x| {
            s(if x.a < x.b { x.a - x.b } else { x.b - x.a })
        }),
        c("p || clamp(a, b, 0.0) > 0.5", Flag, Flags, |x| {
            if x.p {
                f(true)
            } else if x.b <= 0.0 {
                f(x.a.clamp(x.b, 0.0) > 0.5)
            } else {
                Err(ClampBounds)
            }
        }),
        c("p ? clamp(a, 1.0, 0.0) : a", Scalar, Flags, |x| {
            if x.p { Err(ClampBounds) } else { s(x.a) }
        }),
        c("!p && a == b || q", Flag, Flags, |x| f(
            !x.p && x.a == x.b || x.q
        )),
    ]
}

/// Values worth hitting: signed zeros, subnormals, the normal range's
/// edges, halves and their neighbors, odd and even integers, 2^23 and
/// its neighbors, constants, huge values, infinities and NaN.
fn specials() -> Vec<f32> {
    let mut v = alloc::vec![
        0.0,
        f32::from_bits(1),
        f32::from_bits(0x0000_1234),
        f32::from_bits(0x007f_ffff),
        f32::MIN_POSITIVE,
        1e-30,
        1e-7,
        0.078_125,
        0.1,
        0.2,
        0.25,
        0.499_999_97,
        0.5,
        0.500_000_06,
        0.6,
        0.999_999_94,
        1.0,
        1.000_000_1,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        core::f32::consts::PI,
        7.0,
        10.0,
        100.5,
        8_388_607.5,
        8_388_608.0,
        8_388_609.0,
        16_777_217.0,
        1e10,
        1e30,
        3.0e38,
        f32::MAX,
        f32::INFINITY,
    ];
    let negatives: Vec<f32> = v.iter().map(|x| -x).collect();
    v.extend(negatives);
    v.push(f32::NAN);
    v.push(-f32::NAN);
    v
}

/// A deterministic stream of bit patterns, and of values in `[-8, 8]`.
struct Draw(u64);

impl Draw {
    fn bits(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }

    /// Alternately any bit pattern, and a value of moderate size.
    #[expect(clippy::cast_precision_loss, reason = "a draw")]
    fn scalar(&mut self) -> f32 {
        let bits = self.bits();
        if bits & 1 == 0 {
            f32::from_bits(self.bits())
        } else {
            (self.bits() as f32 / u32::MAX as f32) * 16.0 - 8.0
        }
    }

    fn integer(&mut self) -> u32 {
        const SPECIAL: [u32; 10] = [
            0,
            1,
            2,
            3,
            7,
            0xffff,
            0x0100_0001,
            0x8000_0000,
            u32::MAX - 1,
            u32::MAX,
        ];
        let bits = self.bits();
        match bits % 4 {
            0 => SPECIAL[(bits as usize / 4) % SPECIAL.len()],
            1 => bits >> 16,
            _ => self.bits(),
        }
    }
}

/// Every input set a case is checked on.
fn sweep(sweep: Sweep) -> Vec<In> {
    let sp = specials();
    let mut draw = Draw(0x6461_7070_6c65);
    let mut out = Vec::new();
    match sweep {
        Sweep::One => {
            for &a in &sp {
                out.push(In { a, ..In::default() });
            }
            for _ in 0..100_000 {
                out.push(In {
                    a: draw.scalar(),
                    ..In::default()
                });
            }
        }
        Sweep::Two => {
            for &a in &sp {
                for &b in &sp {
                    out.push(In {
                        a,
                        b,
                        ..In::default()
                    });
                }
            }
            for _ in 0..50_000 {
                out.push(In {
                    a: draw.scalar(),
                    b: draw.scalar(),
                    ..In::default()
                });
            }
        }
        Sweep::Three => {
            // Every third special, cubed: 25 values.
            let few: Vec<f32> = sp.iter().copied().step_by(3).collect();
            for &a in &few {
                for &b in &few {
                    for &c in &few {
                        out.push(In {
                            a,
                            b,
                            c,
                            ..In::default()
                        });
                    }
                }
            }
            for _ in 0..30_000 {
                out.push(In {
                    a: draw.scalar(),
                    b: draw.scalar(),
                    c: draw.scalar(),
                    ..In::default()
                });
            }
        }
        Sweep::Integers => {
            for _ in 0..30_000 {
                out.push(In {
                    a: draw.scalar(),
                    n: draw.integer(),
                    m: draw.integer(),
                    k: draw.integer(),
                    ..In::default()
                });
            }
        }
        Sweep::Flags => {
            for &a in &sp {
                for (p, q) in [(false, false), (false, true), (true, false), (true, true)] {
                    out.push(In {
                        a,
                        b: -a,
                        p,
                        q,
                        ..In::default()
                    });
                    out.push(In {
                        a,
                        b: 0.25,
                        p,
                        q,
                        ..In::default()
                    });
                }
            }
        }
    }
    out
}

fn program(source: &str, ty: Type) -> Program {
    let mut b = Builder::new();
    for (name, t) in [
        ("a", Type::Scalar),
        ("b", Type::Scalar),
        ("c", Type::Scalar),
        ("n", Type::Integer),
        ("m", Type::Integer),
        ("k", Type::Integer),
        ("p", Type::Flag),
        ("q", Type::Flag),
    ] {
        b.input(name, t).expect("distinct names");
    }
    b.expression(source, ty)
        .unwrap_or_else(|e| panic!("{source}: {e}"));
    b.build().unwrap_or_else(|e| panic!("{source}: {e}"))
}

/// Bits, but every NaN alike.
fn same(a: Value, b: Value) -> bool {
    match (a, b) {
        (Value::Scalar(x), Value::Scalar(y)) => {
            (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits()
        }
        _ => a == b,
    }
}

#[test]
fn every_operation_matches_rust_over_libm_bit_for_bit() {
    let mut failures: Vec<String> = Vec::new();
    let mut total = 0_usize;
    for case in cases() {
        let program = program(case.source, case.ty);
        let points = sweep(case.sweep);
        let mut mismatches = 0_usize;
        let mut nans = 0_usize;
        for x in &points {
            let tape = program.run(&x.values()).map(|v| v[0]).map_err(|e| e.kind);
            let rust = (case.rust)(*x);
            let ok = match (&tape, &rust) {
                (Ok(t), Ok(r)) => same(*t, *r),
                (Err(t), Err(r)) => t == r,
                _ => false,
            };
            if matches!(rust, Ok(Value::Scalar(v)) if v.is_nan()) {
                nans += 1;
            }
            if !ok {
                if mismatches < 3 {
                    failures.push(format!(
                        "{}: at {x:?} the tape gives {tape:?}, Rust {rust:?}",
                        case.source
                    ));
                }
                mismatches += 1;
            }
        }
        println!(
            "{:<45} {:>7} points ({nans} NaN), {mismatches} mismatches",
            case.source,
            points.len()
        );
        total += points.len();
    }
    println!("{total} points in all");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
