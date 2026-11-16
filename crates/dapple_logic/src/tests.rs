// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use crate::program::version_word;
use crate::{Builder, CONTRACT, Error, ErrorKind, Program, RunErrorKind, Span, Type, Value};

fn builder() -> Builder {
    let mut b = Builder::new();
    b.input("age", Type::Scalar).unwrap();
    b.input("gloss", Type::Scalar).unwrap();
    b.input("count", Type::Integer).unwrap();
    b.input("weather", Type::Flag).unwrap();
    b
}

fn inputs(age: f32, gloss: f32, count: u32, weather: bool) -> [Value; 4] {
    [
        Value::Scalar(age),
        Value::Scalar(gloss),
        Value::Integer(count),
        Value::Flag(weather),
    ]
}

fn one(source: &str, ty: Type) -> Program {
    let mut b = builder();
    b.expression(source, ty).unwrap();
    b.build().unwrap()
}

fn eval(source: &str, ty: Type, at: [Value; 4]) -> Value {
    one(source, ty).run(&at).unwrap()[0]
}

fn error(source: &str, ty: Type) -> Error {
    builder().expression(source, ty).unwrap_err()
}

#[test]
fn expressions_evaluate_in_one_program() {
    let mut b = builder();
    assert_eq!(b.expression("0.6 * age * age", Type::Scalar), Ok(0));
    assert_eq!(
        b.expression("0.5 + (0.05 - 0.5) * gloss", Type::Scalar),
        Ok(1)
    );
    assert_eq!(b.expression("count * 2 + 1", Type::Integer), Ok(2));
    assert_eq!(b.expression("weather && age > 0.5", Type::Flag), Ok(3));
    let p = b.build().unwrap();
    assert_eq!(
        p.inputs(),
        [Type::Scalar, Type::Scalar, Type::Integer, Type::Flag]
    );
    assert_eq!(
        p.outputs(),
        [Type::Scalar, Type::Scalar, Type::Integer, Type::Flag]
    );
    let (age, gloss) = (0.078_125_f32, 0.85_f32);
    let out = p.run(&inputs(age, gloss, 7, true)).unwrap();
    assert_eq!(
        out,
        vec![
            Value::Scalar(0.6 * age * age),
            Value::Scalar(0.5 + (0.05 - 0.5) * gloss),
            Value::Integer(15),
            Value::Flag(false),
        ]
    );
    // The value the spike found f64 evaluation gets wrong.
    assert_eq!(0.6_f32 * age * age, f32::from_bits(0x3b70_0000));
    // Runs again on the parked VM, and on a clone's own.
    assert_eq!(p.run(&inputs(age, gloss, 7, true)).unwrap(), out);
    assert_eq!(p.clone().run(&inputs(age, gloss, 7, true)).unwrap(), out);
}

/// Operators group as Rust groups them; the tape keeps the grouping, so
/// regrouping that changes the bits is visible.
#[test]
fn precedence_and_associativity_are_rusts() {
    let at = |a: f32, g: f32| inputs(a, g, 0, false);
    // a - b - c is (a - b) - c.
    let (a, b) = (1e8_f32, 1.0_f32);
    assert_eq!(
        eval("age - gloss - age", Type::Scalar, at(a, b)),
        Value::Scalar(a - b - a)
    );
    assert_ne!(a - b - a, a - (b - a));
    assert_eq!(
        eval("age - (gloss - age)", Type::Scalar, at(a, b)),
        Value::Scalar(a - (b - a))
    );
    // a / b * c is (a / b) * c.
    let (a, b) = (1.0_f32, 3.0_f32);
    assert_eq!(
        eval("age / gloss * gloss", Type::Scalar, at(a, b)),
        Value::Scalar(a / b * b)
    );
    // -a * b is (-a) * b; * binds tighter than +, + tighter than <.
    assert_eq!(
        eval("-age * gloss", Type::Scalar, at(2.0, 3.0)),
        Value::Scalar(-6.0)
    );
    assert_eq!(
        eval("1.0 + age * gloss", Type::Scalar, at(2.0, 3.0)),
        Value::Scalar(7.0)
    );
    assert_eq!(
        eval("age + 1.0 < gloss", Type::Flag, at(2.0, 3.0)),
        Value::Flag(false)
    );
    // && binds tighter than ||, and ?: is right-associative.
    assert_eq!(
        eval(
            "weather || age > 1.0 && false",
            Type::Flag,
            inputs(2.0, 0.0, 0, true)
        ),
        Value::Flag(true)
    );
    assert_eq!(
        eval(
            "count < 2 ? 1.0 : count < 4 ? 2.0 : 3.0",
            Type::Scalar,
            inputs(0.0, 0.0, 3, false)
        ),
        Value::Scalar(2.0)
    );
    // Comparisons do not chain; parenthesized, they compare flags.
    let e = error("age < gloss == weather", Type::Flag);
    assert_eq!(e.kind, ErrorKind::ChainedComparison);
    assert_eq!(e.span, Span::new(12, 14));
    assert_eq!(
        error("1 < 2 < 3", Type::Flag).kind,
        ErrorKind::ChainedComparison
    );
    assert_eq!(
        eval(
            "(age < gloss) == weather",
            Type::Flag,
            inputs(1.0, 2.0, 0, true)
        ),
        Value::Flag(true)
    );
}

#[test]
fn whole_numbers_take_their_contexts_type() {
    let at = inputs(1.5, 0.0, 7, false);
    assert_eq!(eval("2 * age", Type::Scalar, at), Value::Scalar(3.0));
    assert_eq!(eval("count + 2", Type::Integer, at), Value::Integer(9));
    assert_eq!(eval("3", Type::Scalar, at), Value::Scalar(3.0));
    assert_eq!(eval("3", Type::Integer, at), Value::Integer(3));
    assert_eq!(eval("2 < 3", Type::Flag, at), Value::Flag(true));
    assert_eq!(eval("min(count, 5)", Type::Integer, at), Value::Integer(5));
    assert_eq!(eval("max(age, 2)", Type::Scalar, at), Value::Scalar(2.0));
    assert_eq!(
        eval("scalar(count) * 0.5", Type::Scalar, at),
        Value::Scalar(3.5)
    );
    // Read as `str::parse::<f32>` reads them.
    assert_eq!(
        eval("16777217 * 1.0", Type::Scalar, at),
        Value::Scalar(16_777_216.0)
    );
    assert_eq!(eval("0.1", Type::Scalar, at), Value::Scalar(0.1));
    assert_eq!(eval("1e-45", Type::Scalar, at), Value::Scalar(1e-45));
}

#[test]
fn bad_expressions_say_what_and_where() {
    let kind = |source: &str, ty| error(source, ty).kind;
    let e = error("max(chipping, 0.6 * age)", Type::Scalar);
    assert_eq!(e.kind, ErrorKind::UnknownName("chipping".into()));
    assert_eq!(e.span, Span::new(4, 12));
    assert_eq!(e.to_string(), "at 4..12: no input named \"chipping\"");
    assert_eq!(
        kind("lerp(age, 1.0, 0.5)", Type::Scalar),
        ErrorKind::UnknownFunction("lerp".into())
    );
    assert_eq!(
        kind("max(age)", Type::Scalar),
        ErrorKind::Arity {
            function: "max",
            expected: 2,
            found: 1
        }
    );
    assert_eq!(
        kind("age", Type::Integer),
        ErrorKind::Mismatch {
            expected: Type::Integer,
            found: Type::Scalar
        }
    );
    assert_eq!(
        kind("age + count", Type::Scalar),
        ErrorKind::Operands {
            left: Type::Scalar,
            right: Type::Integer
        }
    );
    assert_eq!(
        kind("weather ? age : count", Type::Scalar),
        ErrorKind::Operands {
            left: Type::Scalar,
            right: Type::Integer
        }
    );
    assert_eq!(
        kind("-count", Type::Integer),
        ErrorKind::Unsupported {
            operation: "`-`",
            ty: Type::Integer
        }
    );
    assert_eq!(
        kind("weather < true", Type::Flag),
        ErrorKind::Unsupported {
            operation: "`<`",
            ty: Type::Flag
        }
    );
    assert_eq!(
        kind("sqrt(count)", Type::Scalar),
        ErrorKind::Mismatch {
            expected: Type::Scalar,
            found: Type::Integer
        }
    );
    assert_eq!(
        kind("age ? 1.0 : 2.0", Type::Scalar),
        ErrorKind::Mismatch {
            expected: Type::Flag,
            found: Type::Scalar
        }
    );
    assert_eq!(
        kind("1e39", Type::Scalar),
        ErrorKind::Literal("too large for a scalar")
    );
    assert_eq!(
        kind("4294967296", Type::Integer),
        ErrorKind::Literal("too large for an integer")
    );
    // Syntax, with where.
    for (source, span) in [
        ("", Span::new(0, 0)),
        ("age +", Span::new(5, 5)),
        ("(age", Span::new(4, 4)),
        ("age gloss", Span::new(4, 9)),
        ("1.", Span::new(0, 2)),
        ("1e", Span::new(0, 2)),
        ("1.5x", Span::new(0, 4)),
        ("age @ 2", Span::new(4, 5)),
        ("age \u{b2}", Span::new(4, 6)),
        ("max(age,)", Span::new(8, 9)),
        ("weather ? 1.0", Span::new(13, 13)),
    ] {
        let e = error(source, Type::Scalar);
        assert!(matches!(e.kind, ErrorKind::Syntax(_)), "{source:?}: {e}");
        assert_eq!(e.span, span, "{source:?}: {e}");
    }
    // Nesting is bounded, whatever the input.
    let deep = "(".repeat(100) + "age" + &")".repeat(100);
    assert_eq!(kind(&deep, Type::Scalar), ErrorKind::TooDeep);
    let long = "age".to_string() + &" + age".repeat(100);
    assert_eq!(kind(&long, Type::Scalar), ErrorKind::TooDeep);
    let negations = "-".repeat(100) + "age";
    assert_eq!(kind(&negations, Type::Scalar), ErrorKind::TooDeep);
    let shallow = "age".to_string() + &" + age".repeat(40);
    assert!(builder().expression(&shallow, Type::Scalar).is_ok());
    // Inputs are named once.
    let mut b = builder();
    assert_eq!(
        b.input("age", Type::Scalar).unwrap_err().kind,
        ErrorKind::DuplicateInput("age".into())
    );
}

#[test]
fn the_same_source_compiles_to_the_same_bytes() {
    let build = |sources: &[&str]| {
        let mut b = builder();
        for s in sources {
            b.expression(s, Type::Scalar).unwrap();
        }
        b.build().unwrap()
    };
    let sources = ["max(age, 0.6 * gloss * gloss)", "smoothstep(0.2, 1.0, age)"];
    let (a, b) = (build(&sources), build(&sources));
    assert_eq!(a.bytes(), b.bytes());
    assert_eq!(
        a.content_words().collect::<Vec<_>>(),
        b.content_words().collect::<Vec<_>>()
    );
    let words: Vec<u64> = a.content_words().collect();
    assert_eq!(words[0], CONTRACT);
    assert_eq!(words[1], a.bytes().len() as u64);
    // Whitespace and redundant parentheses do not change the program.
    let spaced = build(&["max( age ,(0.6*gloss)*gloss )", "smoothstep(0.2,1.0,age)"]);
    assert_eq!(spaced.bytes(), a.bytes());
    // Regrouping does, as it changes what is computed.
    let regrouped = build(&[
        "max(age, 0.6 * (gloss * gloss))",
        "smoothstep(0.2, 1.0, age)",
    ]);
    assert_ne!(regrouped.bytes(), a.bytes());
    // So does another order of expressions.
    let swapped = build(&["smoothstep(0.2, 1.0, age)", "max(age, 0.6 * gloss * gloss)"]);
    assert_ne!(swapped.bytes(), a.bytes());
}

#[test]
fn a_run_stops_with_an_error_and_names_its_expression() {
    let mut b = builder();
    b.expression("age * 2.0", Type::Scalar).unwrap();
    b.expression("count * 1000000", Type::Integer).unwrap();
    b.expression("10 / count", Type::Integer).unwrap();
    b.expression("clamp(age, gloss, 1.0)", Type::Scalar)
        .unwrap();
    let p = b.build().unwrap();
    let fail = |at: [Value; 4]| p.run(&at).map(|_| ()).unwrap_err();
    let e = fail(inputs(0.5, 0.0, 5000, false));
    assert_eq!((e.expression, e.kind), (Some(1), RunErrorKind::Overflow));
    let e = fail(inputs(0.5, 0.0, 0, false));
    assert_eq!(
        (e.expression, e.kind),
        (Some(2), RunErrorKind::DivisionByZero)
    );
    let e = fail(inputs(0.5, 2.0, 1, false));
    assert_eq!((e.expression, e.kind), (Some(3), RunErrorKind::ClampBounds));
    let e = fail(inputs(0.5, f32::NAN, 1, false));
    assert_eq!((e.expression, e.kind), (Some(3), RunErrorKind::ClampBounds));
    // And recovers: a failed run leaves nothing behind.
    assert_eq!(
        p.run(&inputs(0.5, 0.0, 2, false)).unwrap(),
        vec![
            Value::Scalar(1.0),
            Value::Integer(2_000_000),
            Value::Integer(5),
            Value::Scalar(0.5),
        ]
    );
    // Inputs must match.
    let e = p.run(&[Value::Scalar(1.0)]).unwrap_err();
    assert_eq!((e.expression, e.kind), (None, RunErrorKind::Inputs));
    let mut wrong = inputs(0.5, 0.0, 2, false);
    wrong[2] = Value::Scalar(2.0);
    assert_eq!(p.run(&wrong).unwrap_err().kind, RunErrorKind::Inputs);
}

#[test]
fn only_the_arm_taken_runs() {
    // An arm that would fail does not, unless it is taken.
    let p = one("weather ? clamp(age, 1.0, 0.0) : age", Type::Scalar);
    assert_eq!(
        p.run(&inputs(0.5, 0.0, 0, false)).unwrap(),
        vec![Value::Scalar(0.5)]
    );
    assert_eq!(
        p.run(&inputs(0.5, 0.0, 0, true)).unwrap_err().kind,
        RunErrorKind::ClampBounds
    );
    let p = one("count == 0 || 10 / count > 2", Type::Flag);
    assert_eq!(
        p.run(&inputs(0.0, 0.0, 0, false)).unwrap(),
        vec![Value::Flag(true)]
    );
    let p = one("count != 0 && 10 / count > 2", Type::Flag);
    assert_eq!(
        p.run(&inputs(0.0, 0.0, 0, false)).unwrap(),
        vec![Value::Flag(false)]
    );
}

#[test]
fn fuel_bounds_a_run() {
    let p = one("0.6 * age * age + gloss", Type::Scalar);
    // Straight-line code: every instruction runs, so the default budget is
    // exactly enough.
    assert_eq!(p.fuel(), p.instructions());
    let at = inputs(0.5, 0.25, 0, false);
    assert_eq!(p.run(&at).unwrap(), vec![Value::Scalar(0.4)]);
    let mut short = p.clone();
    short.set_fuel(p.instructions() - 1);
    let e = short.run(&at).unwrap_err();
    assert_eq!(e.kind, RunErrorKind::OutOfFuel);
    // A budget set after a run reaches the parked VM too.
    let mut parked = p.clone();
    assert!(parked.run(&at).is_ok());
    parked.set_fuel(2);
    assert_eq!(parked.run(&at).unwrap_err().kind, RunErrorKind::OutOfFuel);
    parked.set_fuel(p.instructions());
    assert!(parked.run(&at).is_ok());
}

#[test]
fn a_program_without_expressions_or_inputs_runs() {
    let p = Builder::new().build().unwrap();
    assert_eq!(p.run(&[]).unwrap(), Vec::<Value>::new());
}

#[test]
fn a_conditioned_expression_runs_only_where_its_condition_holds() {
    let mut b = builder();
    let wet = b.expression("weather && age > 0.0", Type::Flag).unwrap();
    let when_wet = b
        .expression_when("clamp(age, gloss, 1.0)", Type::Scalar, wet, true)
        .unwrap();
    let when_dry = b
        .expression_when("count * 1000000", Type::Integer, wet, false)
        .unwrap();
    // A condition can itself be conditioned.
    let deeper = b
        .expression_when("count > 2", Type::Flag, wet, true)
        .unwrap();
    let deepest = b
        .expression_when("10 / count", Type::Integer, deeper, true)
        .unwrap();
    assert_eq!((when_wet, when_dry, deeper, deepest), (1, 2, 3, 4));
    let p = b.build().unwrap();
    // Wet: the first runs, the second does not, though it would overflow.
    assert_eq!(
        p.run(&inputs(0.5, 0.25, 5000, true)).unwrap(),
        vec![
            Value::Flag(true),
            Value::Scalar(0.5),
            Value::Integer(0),
            Value::Flag(true),
            Value::Integer(0),
        ]
    );
    // Dry: the first does not run, though its bounds are out of order, and
    // gives its type's zero; nor does what is conditioned on a result that
    // did not run (`deeper` is `false`, its placeholder).
    assert_eq!(
        p.run(&inputs(0.5, 2.0, 0, false)).unwrap(),
        vec![
            Value::Flag(false),
            Value::Scalar(0.0),
            Value::Integer(0),
            Value::Flag(false),
            Value::Integer(0),
        ]
    );
    // Where an expression runs, what stops it names it.
    let e = p.run(&inputs(0.5, 2.0, 3, true)).unwrap_err();
    assert_eq!((e.expression, e.kind), (Some(1), RunErrorKind::ClampBounds));
    let e = p.run(&inputs(0.5, 0.0, 5000, false)).unwrap_err();
    assert_eq!((e.expression, e.kind), (Some(2), RunErrorKind::Overflow));
    // Straight-line or not, no run needs more than the default fuel.
    assert_eq!(p.fuel(), p.instructions());

    // A condition is an earlier flag expression.
    let mut b = builder();
    let age = b.expression("age", Type::Scalar).unwrap();
    for condition in [age, 1] {
        let e = b
            .expression_when("gloss", Type::Scalar, condition, true)
            .unwrap_err();
        assert_eq!(e.kind, ErrorKind::Condition(condition));
        assert_eq!(e.span, Span::default());
    }
    assert_eq!(
        ErrorKind::Condition(1).to_string(),
        "result 1 is no earlier flag expression"
    );

    // The condition, and which way it must go, are in the bytes.
    let build = |holds: bool| {
        let mut b = builder();
        let c = b.expression("weather", Type::Flag).unwrap();
        b.expression_when("age", Type::Scalar, c, holds).unwrap();
        b.build().unwrap()
    };
    let mut plain = builder();
    plain.expression("weather", Type::Flag).unwrap();
    plain.expression("age", Type::Scalar).unwrap();
    let plain = plain.build().unwrap();
    assert_eq!(build(true).bytes(), build(true).bytes());
    assert_ne!(build(true).bytes(), build(false).bytes());
    assert_ne!(build(true).bytes(), plain.bytes());
}

#[test]
fn the_contract_names_the_pinned_libm() {
    // A byte per part, as the word has always had it.
    assert_eq!(version_word("0.2.16"), 0x00_02_10);
    assert_eq!(version_word("1.0.255"), 0x01_00_ff);
    assert_eq!(
        (CONTRACT >> 8) & 0xff_ff_ff,
        version_word(execution_tape::math::LIBM_VERSION)
    );
}
