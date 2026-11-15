# dapple_logic

Small typed expressions over named inputs, compiled to one verified
[`execution_tape`](https://github.com/forest-rs/execution) program: the
logic layer of [dapple](../../docs/design.md) packages ("let the numbers be
formulas").

```rust
use dapple_logic::{Builder, Type, Value};

let mut b = Builder::new();
b.input("chipping", Type::Scalar)?;
b.input("weather", Type::Scalar)?;
b.expression("max(chipping, 0.5 * weather * weather)", Type::Scalar)?;
let program = b.build()?;
let out = program.run(&[Value::Scalar(0.03), Value::Scalar(0.5)])?;
assert_eq!(out, [Value::Scalar(0.125)]);
# Ok::<(), Box<dyn core::error::Error>>(())
```

- **Three types**: `scalar` (`f32`), `integer` (`u32`) and `flag` (`bool`).
  Arithmetic, comparisons (which do not chain), `&&`, `||`, `!`, `?:`, and
  `min`, `max`, `clamp`, `abs`, `floor`, `round`, `sqrt`, `pow`, `exp`,
  `ln`, `sin`, `cos`, `atan2`, `mix`, `smoothstep` and `scalar`.
- **Exact**: a program computes what the same expression written in Rust
  over `libm` 0.2.16 computes, bit for bit, on every platform. Operations
  run in source order, nothing is folded, reassociated or fused, and
  literals read as `str::parse::<f32>` reads them. A test holds every
  operation and function to that over millions of inputs.
- **Checked when built**: unknown names, wrong types and arities, and
  syntax errors are reported with their place in the source. A run returns
  a typed error (integer overflow, division by zero, `clamp` bounds out of
  order, fuel) naming its expression, never a panic.
- **Fingerprintable**: the same source always encodes to the same bytes;
  `Program::content_words` gives those bytes and the evaluation contract
  (tape format and `libm` version) for a fingerprint.

The crate docs give the grammar and every function's Rust counterpart.

`#![no_std]` with `alloc`.
