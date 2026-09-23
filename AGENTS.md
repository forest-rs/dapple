# Dapple

Dapple is a procedural material engine: deterministic, incremental texture and
material synthesis for PBR assets.

## Forest Engineering Tenets

These tenets govern all forest-rs projects. They are non-negotiable.

1. **We Build to Endure.** Systems that are difficult to outgrow, difficult to entangle, easy to reason about, easy to measure. Optimize for structural strength, not short-term applause.
2. **Modularity Is Power.** Every subsystem: narrow responsibility, minimal dependency surface, replaceable internals, stable API. Monoliths are a last resort.
3. **Incrementalism Everywhere.** Full rebuilds are failure modes. Deltas over rewrites. Patches over full uploads. Caches over recomputation. Budgeted work over spikes.
4. **Introspection Is Non-Optional.** If we cannot measure it, we cannot improve it. Every system exposes: time (CPU + GPU), memory (live + fragmentation), work units, bandwidth. Diagnostics are architecture.
5. **Explicit Over Implicit.** No hidden state. No invisible scheduling. No accidental lifetime behavior. No magical performance characteristics. Predictability is a feature.
6. **Long-Term > Short-Term.** Clean structure over clever shortcuts. Extensibility over demo velocity. Architectural leverage over temporary wins.
7. **Replaceability Is a Constraint.** Major subsystems tolerate different backends, techniques, allocators, platforms. If something cannot be replaced, it must be small and contained.
8. **Calm Interfaces.** Internal complexity may be aggressive. Public APIs must be calm: boring, obvious, stable, intentional.
9. **No Sacred Subsystems.** Refactor without attachment. Remove complexity when possible. Evolve forward.

## North Star

- Keep core crates small, predictable, and long-lived.
- Prefer simple, explicit designs over clever ones.
- Avoid dependency creep; keep compile times and surface area under control.
- Optimize for long-term architecture over short-term compatibility; it’s OK to break callers to get the right core shape.

## Non-negotiables (Definition of Done)

- `typos` passes.
- `taplo fmt` passes.
- `cargo fmt` passes.
- `cargo clippy` passes (`-D warnings`).
- `cargo doc` passes.
- Public APIs are documented (types/functions; public fields/variants where it matters).
- Tests updated/added when behavior changes.
- When an issue owns the work, its close note captures the implementation
  summary, key decisions/tradeoffs, and validation.
- Architectural/invariant/public-semantic decisions are captured in a crate-local ADR (`crates/<crate>/docs/adr-XXXX-<slug>.md`) or an existing ADR is updated.
- Examples/benchmarks live in separate top-level workspace crates (no extra dev-deps in core crates).

Suggested commands:

```sh
typos
cargo fmt --all
taplo fmt
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo doc --no-deps
```

## `no_std` policy (core crates)

- Default assumption for foundational crates: `#![no_std]` whenever practical (use `extern crate alloc` when needed).
- Keep `std` behind an explicit `std` feature flag when required.
- Avoid `std` collections in `no_std` crates; use `hashbrown` (and `alloc` types) instead.

## Tests, examples, benchmarks

- Unit tests live next to code; keep them deterministic.

## Documentation and ADR workflow

- Implementation detail should live in the commit message, issue note, or PR
  summary appropriate to the size and lifetime of the change.
- Crate-local plan files in `crates/<crate>/docs/plans/` are optional and should
  be used only when additional durable design context is needed beyond those
  records.
- When a change crosses crate boundaries, document the primary decision in one owning crate ADR and link from other affected crate docs.
- If no ADR is needed, state why in the issue notes/PR summary so decision intent is still explicit.

# Issues / Tracking / Plans

This project uses [Beads](https://github.com/gastownhall/beads) (`bd`) for
shared issue tracking. Issue IDs use the `dapple-*` prefix. The authoritative
issue history is synchronized through this repository's Dolt remote, not
ordinary Git commits.

- After a fresh clone, run `bd bootstrap`.
- At the start of a work session, run `bd prime` and `bd dolt pull`.
- Use `bd ready`, `bd show`, `bd create`, `bd update --claim`, and `bd close`
  to manage work.
- At the end of a work session, run `bd dolt push` explicitly. Automatic
  Beads pushes remain disabled.

Issues are not required for every change. Create one when work spans sessions
or contributors, remains blocked or unresolved, has meaningful dependencies,
or establishes a durable public contract. Link dependencies with `bd dep` so
the issue graph states execution order. Longer prose plans live in the owning
crate's `docs/plans/` directory and reference the owning issue ID.

Beads history and source history are published independently: `git push`
publishes source commits, while `bd dolt push` publishes issue changes.
