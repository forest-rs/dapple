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

## `build`

Building programs, over chains of 256, 1024 and 4096 dependent nodes whose
parameters all differ. Time per node should stay flat as the chain grows;
`build/<name>/<length>` reports elements, so compare per-node throughput
across lengths.

- `add_chain`: `ProgramBuilder::add` once per node.
- `finish_chain`: `ProgramBuilder::finish`, which derives the evaluation
  plan.
- `import_into_empty`: `ProgramBuilder::import` of a finished chain into an
  empty builder, comparing each node's fingerprint with those already
  imported.
- `import_again`: the same import into a builder that already holds every
  node, so each is found and none copied.

## `raster`

Realizing fields over one period, reported as texels per second. The `small`
fixtures are cheap enough that the realization loop itself shows; `bark` and
`bark_color` are the library's oak bark at 256 texels a side only, to keep the
suite short.

- `raster/realize_scalar/<program>/<size>`: `dapple_raster::realize`.
- `raster/realize_value/<program>/<size>`: `typed::realize_value` of a color
  program.
- `raster/realize_normals/<program>/<size>`: `realize_normals`, normals from
  the field's gradient.
- `raster/height_to_normal/<size>`: `HeightToNormal` on an already realized
  height raster, by central differences.

Sizes are 256 and 1024 texels a side.

## `graph`

`MaterialGraph` runs and edits on one recipe: gradient noise and a disk
combined by `max`, realized over one period in 32-texel tiles, blurred, and
turned into normals by `HeightToNormal`. Sizes are 256 and 512 texels a side
(64 and 256 tiles), so the cost of an edit can be compared with the size of
the raster. Each edit benchmark alternates between two values so every
iteration is a real edit, and times the edit and the run together.

- `graph/fresh_run/<size>`: a first run of a freshly built graph. Building is
  not timed.
- `graph/clean_run/<size>`: a run with nothing edited.
- `graph/edit_local_disk/<size>`: move the disk (`Op::Disk`), an edit that
  states its change region, so only the tiles it reaches recompute. The
  benchmark checks that the edit stays local before it measures.
- `graph/edit_global_seed/<size>`: reseed the noise, which changes every
  texel.
- `graph/edit_blur_sigma/<size>`: change the blur, which re-runs the blur and
  the normals but not the realization.
- `graph/budgeted_slice/<size>`: the first run after moving the disk under a
  budget of four tiles, the work a caller with a frame budget pays per run.
- `graph/budgeted_converge/<size>`: the same edit, run until the budget leaves
  nothing pending.

## `material`

The material layer on two real library modules: `varnished_board` (oak, a
varnish finish and grime, an asset composing several modules) and
`roman_brick` (masonry on a running-bond layout), on a wrapping 256 by 256
grid with default parameters. Reported as texels per second.

- `material/instantiate/<module>`: `Context::instantiate`, building the
  material from nothing.
- `material/lower_maps/<module>`: `dapple_material::lower::maps`, the
  material's channels as images for packing.
- `material/pack_gltf/<module>`: `dapple_encode::pack` of those maps for the
  glTF profile, with mip chains.
