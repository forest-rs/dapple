# Dapple wind tunnel

Criterion benchmarks for the paths dapple optimizes: field evaluation,
raster realization, graph runs and edits, program building, and the material
layer. The crate is unpublished and std-only; the core crates carry no
benchmark dependencies.

Everything is deterministic: fixed seeds, fixed sizes, no clocks. Benchmark
names are stable so that numbers from two revisions can be compared with
Criterion's baselines:

```sh
cargo bench -p dapple_wind_tunnel -- --save-baseline before
# change something
cargo bench -p dapple_wind_tunnel -- --save-baseline after
```

## Running

```sh
# Everything.
cargo bench -p dapple_wind_tunnel

# One bench target.
cargo bench -p dapple_wind_tunnel --bench field

# Benchmarks whose name contains a filter, with a shorter measurement.
cargo bench -p dapple_wind_tunnel --bench field -- trivial --measurement-time 2
```

## `field`

Point evaluation. Every benchmark evaluates the same 1024 points over the
unit period, at the footprint of one texel of a 256 texel raster, and reports
elements per second, so the numbers are per sample.

- `field/raw/*`: `Noise`, five-octave `Fractal` and `Cellular` called
  directly, the floor the programs are built on.
- `field/program/<program>/*`: scalar programs. `eval` is
  `ScalarField::eval`, one call per point; `evaluator` keeps one
  `Evaluator` across the points; `batch` is `ScalarField::eval_batch`;
  `gradient` is `eval_gradient`, the path normals take.
- `field/value_program/<program>/*`: `ValueProgram::eval` per point against a
  kept `evaluator`.

The programs: `trivial` is a few constants with a shared subgraph, so nearly
all of its cost is per-call setup, which isolates what a per-call evaluator
costs; `small` is a fractal with an abs and a remap; `small_color` mixes two
colors by a fractal; `bark_height` and `bark_color` are the library's oak
bark, cellular plates under warps, a realistically deep program.
