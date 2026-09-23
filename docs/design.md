# Dapple: a procedural material engine

Design draft, 2026-09-23.

## Purpose

Dapple generates materials, not just textures. A dapple graph describes a
surface's appearance as functions over a domain and realizes them as OpenPBR
parameter maps, ready for a renderer or an exporter. It sits in the same class
as Substance Designer, but it is code-first, deterministic, incremental and
measurable. It is a library that other forest-rs projects build on:

- **sylva:** bark, leaves, twig and cluster cards, impostor post-processing;
- **exedra and joiner:** wood grain on timbers (solid, so end grain is right),
  stone, brick and mortar for `joiner_masonry`, roof tiles, plaster;
- **terrain:** ground layers, splat masks, and weathering driven by slope and
  curvature;
- **lightweald:** its material gallery and baked texture sets, and, later,
  procedural detail evaluated at runtime.

The engineering properties are part of the goal:
- a given graph, seed and realization produce identical bytes on every
  platform;
- editing one node re-realizes only the tiles it affects;
- every realization reports its work.

## Non-goals (for now)

- **An editor UI.** Authoring is Rust builders first and serde data later. A
  node editor can come later on the same graph value.
- **A GPU backend.** The design keeps the field IR translatable to shaders, but
  the first backend is CPU and exact.
- **Painting and projection.** Dapple synthesizes; it does not paint onto
  meshes interactively.
- **Physically simulated appearance** (spectral rendering, measured BRDF
  fitting). Dapple outputs OpenPBR parameters; the BSDF is the renderer's job.

## OpenPBR first

A material's output is an **OpenPBR 1.1.1 parameter set**. Each parameter is
either a constant or bound to a graph output. The names, meanings, units and
defaults are the specification's, matching `lightweald_material::OpenPbrParams`
(lightweald ADR-0002):

- **base:** `base_weight`, `base_color`, `base_diffuse_roughness`,
  `base_metalness`;
- **specular:** `specular_weight`, `specular_color`, `specular_roughness`,
  `specular_roughness_anisotropy`, `specular_ior`;
- **transmission:** `transmission_weight`, `transmission_color`,
  `transmission_depth`, `transmission_scatter`, `…_scatter_anisotropy`,
  `…_dispersion_*`;
- **subsurface:** `subsurface_weight`, `subsurface_color`, `subsurface_radius`,
  `subsurface_radius_scale`, `subsurface_scatter_anisotropy`;
- **coat:** `coat_weight`, `coat_color`, `coat_roughness`,
  `coat_roughness_anisotropy`, `coat_ior`, `coat_darkening`;
- **fuzz:** `fuzz_weight`, `fuzz_color`, `fuzz_roughness`;
- **emission:** `emission_luminance`, `emission_color`;
- **thin film:** `thin_film_weight`, `thin_film_thickness`, `thin_film_ior`;
- **geometry:** `geometry_opacity`, `geometry_thin_walled`, `geometry_normal`,
  `geometry_tangent`, `geometry_coat_normal`, `geometry_coat_tangent`.

Lightweald omits the geometry group because the mesh supplies the shading
basis. Dapple needs it: normals, tangents (anisotropy direction) and opacity
are exactly what it synthesizes. Rotations such as
`specular_anisotropy_rotation` follow the adobe/openpbr-bsdf extension, as
lightweald does.

Some channels are not OpenPBR parameters but consumers need them. Dapple emits
these as **auxiliary outputs** with their own declared semantics, never smuggled
into OpenPBR channels:
- height/displacement (in world units);
- ambient occlusion;
- curvature;
- region/ID masks;
- wind or growth masks for sylva.

### Projections are explicit and lossy

**Packing profiles** turn a realized material into textures. A profile names
which parameters go in which channels, each channel's encoding (sRGB or
linear), and what happens to parameters the target cannot represent.

- **`lightweald`** matches `MaterialTextureSlot`. For example, base-color RGB
  plus `geometry_opacity` in A, `specular_roughness` in G and `base_metalness`
  in B, and tangent-space normals as RG. Parameters that vary spatially but
  have no lightweald slot are reported (e.g. `subsurface_color`,
  `transmission_color`, `emission_luminance`). They are not dropped silently.
  The report is also the evidence for which slots lightweald should add next.
- **`gltf`** is core glTF PBR plus `KHR_materials_{specular, clearcoat,
  sheen, transmission, diffuse_transmission, volume, iridescence, anisotropy,
  emissive_strength}`. It is lossy by construction: F82 edge tint, EON diffuse
  roughness, coat darkening, and subsurface as distinct from diffuse
  transmission do not survive intact. Each approximation is named in the
  profile docs and counted in the report. This mirrors lightweald's
  `gltf.rs`, which maps glTF into OpenPBR, not the other way.
- **Raw** writes one parameter per image, as floats or 16-bit, for offline
  renderers, debugging and golden tests.

### Where the OpenPBR types live

Lightweald, dapple, exedra (materials and its glTF export) and sylva all need
the same parameter vocabulary. Three options:

1. Dapple depends on `lightweald_material`. This couples a material generator
   to a renderer's crate, including renderer concerns: pipeline keys, texture
   handles, pools.
2. Dapple defines its own set. That gives two copies of the spec that will
   drift.
3. **A small shared `openpbr` crate** (the name is free on crates.io). It
   would be `no_std` with no dependencies, holding:
   - `OpenPbrParams`, including the geometry group;
   - the parameter enumeration: name, kind (weight, color, roughness, IOR,
     distance, angle, bool), unit, range, default and whether it is color;
   - specification-version constants.

   Renderer state (alpha mode, texture slots, pipeline keys) stays in
   `lightweald_material`, which then depends on `openpbr`.

**Recommendation: option 3.** Extract it from `lightweald_material::openpbr`,
since that is already the reviewed transcription of the spec. It can live in
its own small repo, or in lightweald as an independently published crate; the
latter is less overhead while lightweald is the main consumer. Dapple's
channel types and packing profiles are generated from the parameter
enumeration, so a new OpenPBR version is one table edit.

## The graph

A dapple **material graph** is a value: typed nodes, typed ports and edges. A
dapple node is not an execution-graph node. The material graph *compiles
onto* the execution machinery; it is not the machinery.

### Two kinds of node

**Field nodes** are point-evaluable. For each input point `p` in the domain
(and optional footprint), a field node computes the value there. It needs no
neighbors. Examples:
- noise, cellular, gradients, shapes;
- arithmetic, blends, levels and curves;
- warps, coordinate transforms;
- sampling of a realized raster.

A field node can additionally provide:
- **Footprint-aware evaluation.** Given the size of the region being
  integrated (a texel, a mip level, a screen pixel later), a node returns a
  band-limited value: noise drops octaves above the footprint's Nyquist limit,
  shapes return exact area coverage instead of a point test, and step
  functions become smooth steps of the right width. This is the difference
  between clean mips and shimmering ones, and it is dapple's main quality
  lever over naive raster graphs.
- **Intervals (optional).** A conservative value range over a region lets
  realization skip constant tiles (masks that are 0 or 1 across a tile),
  bound dynamic range for encoding, and prune downstream evaluation.
- **Derivatives (optional).** Analytic gradients, where cheap, give exact
  normals from height fields without finite differences.

**Raster nodes** need neighborhoods, so they run on a realized grid:
- blur (Gaussian, directional);
- height → normal, where a field has no analytic gradient;
- AO from height, curvature;
- distance transforms, flood fill (connected-component IDs and random per-cell
  values for tile layouts);
- morphology;
- later, erosion and weathering, and histogram operations.

Each raster node declares its **kernel footprint in texels at a stated
resolution**. That footprint drives tile dependencies and the invalidation of
dirty tiles.

### Realization is an explicit boundary

A field has no resolution; a raster does. The graph makes the conversion
explicit with a `Realize { resolution, domain, filter }` step. Any raster node
downstream sees that resolution and nothing else.

A raster can go back into a field through `Sample { filter }` (bilinear,
bicubic, or exact box over a footprint). The graph can therefore mix freely,
but every resolution-dependent choice is visible in the graph and in reports.
A resolution is never inferred from the output.

One consequence: realizing the same graph at 1K and at 4K gives the same field
parts band-limited to each resolution. Raster parts are scaled by their
declared physical footprint (a kernel in world units, converted to texels at
realization). A 4 mm blur stays 4 mm at any resolution.

### Typed ports

Port types carry their meaning, so errors surface when the graph is built, not
as wrong-looking output:

- `Scalar`, optionally with declared range and units: `Weight` (0–1),
  `Roughness`, `Height { units }`, `Distance`, `Angle`.
- `Color` is **always linear, with declared primaries**: Rec. 709/sRGB
  primaries by default, which is what OpenPBR expects. sRGB transfer encoding
  exists **only** at the output-encoding stage. There is no "sRGB color" port
  type, so gamma-space blending is impossible to write by accident.
- `Vector2`, `Vector3`.
- `Normal` in a declared frame: tangent space with a stated handedness and
  green-channel convention (+Y, OpenGL-style, matching glTF; DirectX-style
  appears only as an output-profile flip). Normals blend with a normal-aware
  operator (reoriented normal mapping or UDN, chosen explicitly), never by
  lerping components.
- `Direction` for tangent/anisotropy directions: a 2D angle field in tangent
  space. It needs a π-periodic, sign-free representation when averaged or
  blurred; this matters for mips.
- `Mask` is scalar coverage in 0–1, with coverage-preserving mips.
- `Id` is integer region or cell identifiers. Its interpolation is nearest
  only; you cannot blur an ID.

### Domains

A field is defined over a **domain**, and domains are types too:

- **`Plane`:** unbounded 2D and non-repeating. Used for decals and one-off
  cards such as sylva's leaf textures.
- **`Periodic { period }`:** 2D on a torus with an integer lattice. **Tiling is
  guaranteed by construction, not repaired afterwards:**
  - Noise uses lattice periods that divide the domain period at every octave.
  - Cellular noise wraps its feature points.
  - A domain warp is allowed because displacing by a periodic field keeps the
    result periodic.
  - Scaling by a non-integer factor, or a rotation that does not preserve the
    lattice, is a **type error** unless the result is explicitly demoted to
    `Plane`. This rules out most "almost tiles" bugs.
- **`Solid`:** 3D. Wood grain is growth rings around a pith axis with
  radial/axial noise, so a cut face shows correct end grain, quarter-sawn
  figure, or knots wherever the timber is cut. Marble veins and stone strata
  work the same way.
- **`Chart`:** a surface domain from an exedra mesh's construction charts.
  Realization rasterizes the mesh's chart layout into a texel → surface-point
  map (position, normal, tangent frame, chart ID). Any `Solid` field is then
  evaluated there, producing a texture baked for that asset. This is how
  timbers get end grain and how sylva bark can follow real branch girth.

  **This is hard.** Raster nodes in a chart domain must respect chart seams:
  a blur must not bleed across unrelated charts, but should continue across a
  seam where the charts are adjacent on the surface. The first version pads
  each chart with gutters by dilation and treats every seam as a boundary.
  Seam-adjacent neighborhoods, which follow chart adjacency from exedra's
  source maps, are a later, measured step.

Converting between domains is explicit. A `Solid` field projected onto a plane
(`Slice { plane }`) is a `Plane` field. A `Periodic` field used as a `Plane` is
simply demoted.

## Incremental evaluation

### On `execution_graph`

A compiled material becomes an `ExecutionGraph<DappleExecutor>`, using the
executor generalization (`Executor`, `NodeAccess`) that is landing in the
execution repo:

- **Nodes:** each material node is one execution node. Its body is a dapple op
  plus its parameters.
- **Values** are cheap handles, as the `Executor::Value` docs ask: a field is
  an `Arc` over its compiled field program; a raster is an `Arc` over a tiled
  image with its fingerprint.
- **Parameters** are node inputs, so editing a parameter dirties exactly that
  node. The graph's dependency recording, targeted draining and cause-path
  reports come for free.
- **External reads** go through `NodeAccess::read_opaque_host` or a keyed host
  read. Examples are an exedra chart layout, an imaging scene, or a sample
  image; changing one of them dirties exactly its readers.

**Early cutoff** works at two levels:

- **Graph level.** `execution_graph`'s `Executor::values_equal` hook
  (execution #98) stops propagation when a re-run node's output equals its
  previous one. `DappleExecutor` compares field programs by fingerprint,
  and rasters by fingerprint plus "no recomputed tile changed", so
  re-setting parameters to their current value re-runs only that node.
- **Tile level.** A raster's fingerprint names its derivation, which
  recipes predict, so an edit that changes the derivation but not the
  texels (a clamp bound no value reaches, a mask whose covered region did
  not change) re-runs the dependents to keep their fingerprints exact. They
  recompute no tiles, because the node marks no tile changed.

### Tiles with `invalidation`

Node-level dirtiness is too coarse for rasters. Changing one brick's color
should not re-realize a 4K map. Inside a realization, dapple tracks work at
tile granularity with an `invalidation` tracker:

- **Keys** are `(node, level, tile)`.
- **Dependency edges come from footprints.** A raster node's tile depends on
  its input tiles expanded by its kernel footprint. A field node's tile
  depends on the same tile of its inputs, plus, for `Sample`, on the source
  tiles that its (bounded) warp can reach. When a warp's bound is unknown, the
  dependency is conservatively the whole input; the report counts this, so it
  is visible rather than silently slow.
- **Change regions:** node edits that can state their region (a moved shape,
  one repainted cell) mark only those tiles, and marks propagate. Drains use
  `deterministic()` ordering, so parallel tile work is scheduled and merged in
  a fixed order.

The first slice (`dapple_graph`) implements this for realize and raster
nodes at mip level 0:

- Field values carry a `Change` (`Nowhere`, `Within(regions)`, `Everywhere`)
  relative to the producing node's previous program. `Op::change_from`
  states an edit's region; the first op that can is `Op::Disk`. Pointwise
  ops carry their inputs' regions; `Transform`, `Demote` and the warped
  input of `Warp` make a change unbounded.
- Realize nodes re-realize the tiles whose texel centers fall in the change
  grown by half the footprint (`realize_into`); raster ops recompute tiles
  with `RasterOp::apply_into`. Both equal whole passes bit for bit.
- Instead of fingerprinting tiles, a node compares each recomputed tile's
  bits with its previous output and marks only the direct dependents of
  tiles that changed, so an edit that leaves a tile unchanged stops there.
- Global ops (no footprint) have no tile edges and recompute whole when
  their input changed. `TileReport` counts recomputed, reused and changed
  tiles, whole recomputes, and unbounded changes.

- Tile budgets (`MaterialGraph::set_tile_budget`) cap the tiles one run
  recomputes. The rest stay pending in their node, which runs again, so
  repeated runs converge exactly to the unbudgeted result;
  `TileReport::pending_tiles` says how much is left.
- Each node shares its retained output with the graph's output value
  rather than copying it, so retention costs nothing beyond the outputs
  themselves. A byte-budgeted cache would only pay off once outputs
  themselves can be dropped and recomputed on demand. That waits for a
  consumer that needs it.

Mip levels are graph nodes, one level each, whose tiles depend on the
level-above tiles their filter taps read. Still to come: `Sample` as a graph
node, with bounded-warp tile dependencies.

### Fingerprints and caches

Every node output has a fingerprint:
- field outputs: `hash(op, parameters, input fingerprints)`;
- raster outputs: additionally hashed with the resolution, domain and tile.

A **tile cache** is keyed by fingerprint and bounded by a byte budget, with
explicit eviction and reported hits, misses and evictions. The cache is shared
across realizations, so a 1K preview and a 4K bake of the same graph share any
tiles that are resolution-independent. Few are, honestly; mostly they share
field compilation.

**Budgets:** a realization can take a work budget (tiles or evaluated samples
per call). It returns partial progress with a report, and resuming continues
from the dirty set. An editor or a streaming world then never has to block on
a full realization.

## Determinism

The promise is: same graph, same seed, same realization parameters, same
bytes. That holds on every platform and thread count, up to the encoded
(pre-compression) texels. Keeping it requires:

- **f32 everywhere in synthesis, and no fused multiply-add** unless written
  explicitly. Rust does not contract `a * b + c` on its own. SIMD paths may
  only use operations whose results match the scalar path exactly: lane-wise
  IEEE add, mul, div, sqrt and min/max.
- **No platform `libm`.** `sin`, `exp`, `pow` and friends from std call the
  platform's math library, which differs between macOS, glibc and MSVC. Dapple
  uses the `libm` crate (pure Rust) or its own polynomial approximations, in
  `no_std` and std builds alike.
- **Keyed hash randomness,** the same idea as sylva: every random decision is
  `hash(seed, node ID, lattice cell or element, purpose)`. There are no
  sequential RNG streams. Noise lattices, scatter jitter and per-cell colors
  are then independent of evaluation order and tile size, which is what makes
  tile-parallel and incremental realization identical to a clean one.
  The hash function should be shared with sylva (see open decisions).
- **Fixed-order reductions:** histograms, min/max normalization and mean
  color are reduced tile by tile in tile order, never with parallel float
  reductions in arbitrary order.
- **Compression is outside the promise.** BC7, BC5 and ASTC encoders are
  deterministic for a *pinned encoder version*, but not across versions.
  Reports fingerprint the pre-compression texels, and the encoder version is
  recorded next to them.

Golden tests hash realized outputs at small resolutions across a fixed corpus
of graphs, on every CI platform.

### A later GPU backend

The field IR is a **closed op set** with no arbitrary Rust closures on the hot
path. It can therefore be translated to Slang or WGSL later:
- as a realization backend, for fast previews that are explicitly *not*
  bit-identical;
- as **runtime procedural detail**, evaluated in lightweald's shaders:
  micro-detail noise, macro variation that breaks repetition, and weathering
  masks evaluated at shading time.

Raster nodes stay CPU in the first design. The backend trait takes a realized
tile request and returns tiles, so a GPU implementation is a replacement, not
a fork. Arbitrary closures are allowed only as explicitly opaque nodes
(`Custom`), which cannot be translated and say so.

## Node library

### First slice (enough for sylva milestone 2: bark and leaf sets)

**Coordinates and domains:**
- UV and position inputs;
- scale, rotate and translate, each lattice-checked on `Periodic`;
- `Slice` from solid to plane;
- `Realize` and `Sample`.

**Noise:**
- value noise and gradient (Perlin-style) noise, periodic-capable;
- simplex-like gradient noise on a skewed lattice. The periodic version needs
  the lattice period to be compatible with the skew; that is doable but fiddly.
  Where it cannot be made periodic, it is `Plane`/`Solid` only, and the type
  system says so.
- fBm, ridged and turbulence, band-limited by footprint;
- domain warp, with a declared bound so tile dependencies stay local.

**Cellular:**
- Worley F1, F2 and F2−F1;
- cell-edge distance and exact cell borders;
- per-cell ID and random value;
- anisotropic cells, stretched along an axis (for bark fissures).

**Shapes** via forest-rs `imaging`:
- A shape is recorded as an `imaging` scene (kurbo paths: leaf contours, vein
  strokes, tile outlines) and exposed as a **coverage field** with exact area
  coverage per footprint.
- The first version rasterizes coverage through an imaging CPU backend
  (tiny-skia or vello_cpu) at the realization resolution, with
  analytic-coverage anti-aliasing.
- Later, an exact analytic-coverage evaluator of the windfoil kind, which
  answers a coverage query for any footprint without a raster.

**Patterns:**
- tile and brick samplers (running, stack, herringbone later), with per-tile
  IDs, random offsets and bevel profiles;
- scatter (Poisson-disc splats of a sub-field) with bounded overlap, so tile
  dependencies stay local.

**Tone:**
- levels, curves (monotone cubic), clamp, remap, gradient map (a color ramp
  evaluated in linear space);
- blend modes defined on linear values, including height blend (max-height
  selection with a transition width);
- normal blending with RNM or UDN.

**Rasters:**
- Gaussian blur, separable and in physical units;
- height → normal (Sobel or central difference, scaled in world units);
- AO from height (horizon-based);
- curvature;
- distance transform (exact Euclidean, Felzenszwalb–Huttenlocher);
- flood fill to IDs.

**Material outputs:** bind nodes to OpenPBR parameters and auxiliaries.

### Later

- **Erosion and weathering:** hydraulic/thermal erosion on height, edge wear
  from curvature, dirt from AO, moss and lichen growth by
  exposure/up-facing.
- **Example-based synthesis:** Heitz–Neyret histogram-preserving blending.
  It tiles a small exemplar with no visible repetition, and the same technique
  can run at runtime in a shader. Texture-by-numbers and patch-based synthesis
  come later still.
- Reaction–diffusion (patterns for lichen, bark lenticels and animal
  markings); venation growth (Runions) as a leaf-specific raster op in sylva
  rather than here.
- **Multi-scale material layering:** a macro/meso/micro split where micro
  becomes runtime detail and macro becomes a low-resolution variation map.
- **Material-level ops:** height-blended layering of two whole OpenPBR
  materials (moss over stone, snow over roof tiles), with each parameter
  blended by its own rule (colors linear, roughness in α space, normals by
  RNM).

## Output and encoding

**Mip chains** are built by dapple, not by a generic downsampler. Each port type
has its own rule:
- **Color:** box or Kaiser filtering in linear light, with premultiplied
  alpha where there is opacity.
- **Masks and opacity:** *coverage-preserving* (Castaño 2010). For each level,
  the threshold is solved so that the fraction of texels above the consumer's
  alpha cutoff matches level 0. The cutoff is part of the packing profile.
  Without this, alpha-tested foliage thins away at distance.
- **Normals:** filtering averages unit normals and records the resulting
  length loss. Rather than simply renormalizing, **the lost variance moves into
  roughness** (Toksvig; LEAN/Kaplanyan–Hill style). Each level's
  `specular_roughness` (and `coat_roughness` for the coat normal) is widened
  so distant surfaces keep their energy instead of turning mirror-like and
  sparkly. Anisotropic variance can feed `specular_roughness_anisotropy`
  later.
- **Directions** use π-periodic averaging.
- **IDs** use nearest or majority.
- **Footprint-aware fields can skip filtering entirely:** a level can be
  *re-evaluated* at its own footprint instead of downsampled. That gives
  exact, alias-free mips for pure field graphs, and is often cheaper than it
  sounds.

**Files:**
- PNG (8/16-bit), and EXR or raw float for data;
- **KTX2** with dapple's own mip chain, in the formats `lightweald_texture_io`
  reads: RGBA8 (sRGB/linear), RG8, BC7, BC5, ASTC 4×4, zstd supercompression.
  Lightweald's `texture_bake` currently builds its own mips inside `ctt` from
  one RGBA8 image. Dapple must hand over **pre-built mips**, so it writes KTX2
  itself and uses `ctt` only as a block compressor behind a std feature. A
  later step is for `texture_bake` to reuse dapple's mip builders, so both
  paths share one definition of correct mips.

**Relation to Lightweald's `texture_bake`.** The two tools cover different
sources. `texture_bake` bakes photographic sources (downloaded images listed
in a manifest) into pool textures, letting `ctt` build color and data mips and
building normal mips itself. Dapple produces procedural sources and builds
every mip chain itself from what each texture means: coverage, normal
variance folded into roughness, footprint re-evaluation of fields. It then
hands finished chains to `ctt` only for block compression
(`dapple_compress`). The output conventions are shared: the same
color/data/normal kinds, pool formats per encoding, glTF normal convention,
`Encoding`/`Quality`/Zstandard settings, and normal chains that match
`texture_bake`'s 2 × 2 renormalized average when variance folding is off. They
could later share code in both directions. `texture_bake` could reuse
dapple's mip builders for photo sources, which would give it coverage
preservation and variance folding. Both could share one pool-format and
normal-chain definition in a small common crate, instead of each keeping its
own table.

**Packing profiles** (`lightweald`, `gltf`, `raw`) decide channels, encodings
and the alpha cutoff, and declare which parameters they cannot carry. Output
is a `MaterialBundle`: textures and their encodings, constant parameters,
profile name, fingerprints and the report.

## Crates

The crate list is a target; crates appear when the milestone that needs them
starts.

| Crate | Owns | `no_std` |
|---|---|---|
| `dapple_field` | Field IR, domains, footprints, intervals, the evaluator, the noise/cellular/pattern op set | yes |
| `dapple_raster` | Tiled rasters, raster ops, kernels and footprints | yes |
| `dapple_graph` | Material graph value, typed ports, validation, compilation onto `execution_graph`, tile invalidation, caches, budgets, reports | yes |
| `dapple_material` | OpenPBR bindings, auxiliary outputs, material-level layering | yes |
| `dapple_encode` | Per-type mip builders, specular AA, packing profiles; PNG/EXR/KTX2 writing and the `ctt` compressor behind `std` | core yes |
| `dapple_imaging` | Imaging scenes → coverage fields | std at first |
| `dapple_exedra` | `Chart` domain: rasterized chart layouts, seam gutters | yes |
| `dapple` | Leaf-only facade | yes |
| `examples/*` | Material gallery (plane, sphere and draped-cloth previews through lightweald and Blender), wood/stone/brick/bark studies | std |
| `benchmarks/*` | Wind tunnels: samples per second per op, tile realize and cache, incremental edit latency | std |

Plus the shared `openpbr` crate outside dapple (see above).

**Dependencies:**
- `glam`, for math;
- `hashbrown`;
- `libm`, for deterministic transcendentals;
- `execution_graph` and `invalidation` (forest-rs);
- `imaging` and `kurbo`, in `dapple_imaging` only;
- `ctt`, `png` and similar, only in `dapple_encode`'s std features.

No image-processing framework dependency: the raster ops are ours, because
determinism and footprints are the point.

## Authoring

A Rust builder comes first:

```rust
let mut g = MaterialGraph::new(Domain::periodic([1, 1]));
let cells = g.cellular(Cellular { scale: [6, 18], metric: F2MinusF1, jitter: 0.8 });
let fissure = g.curve(cells, Curve::smoothstep(0.02, 0.15));
let height = g.fbm_warp(fissure, Fbm { octaves: 6, ..Fbm::default() }, WarpBound(0.03));
let normal = g.height_to_normal(g.realize(height, Resolution::texels_per_meter(1024)), Height::mm(6.0));
g.bind(OpenPbr::BaseColor, g.gradient_map(height, bark_ramp));
g.bind(OpenPbr::SpecularRoughness, g.remap(height, 0.9..0.6));
g.bind(OpenPbr::GeometryNormal, normal);
g.aux(Aux::Height, height);
```

The graph is a value with stable node IDs. An optional `serde` feature
serializes graphs as data, which is what presets, sylva species files and a
future editor share. Parameters are named and typed, so a preset exposes a
small public surface ("fissure depth", "moss coverage") over a large internal
graph. That is the `Substance` "exposed parameters" idea, done as graph
inputs.

Today the data form is `dapple_graph::Recipe`: labeled nodes plus outputs
naming the material maps they fill, versioned by `RECIPE_VERSION`, with
content fingerprints computed from the recipe alone. `tools/dapple_bake`
bakes the recipes a TOML manifest lists, as Lightweald's `texture_bake`
bakes photographs, and skips materials whose recipe fingerprint and settings
are unchanged. Named exposed parameters are still to come.

## Introspection

Every realization returns a report with:
- per-node timings;
- samples and tiles evaluated, cache hits/misses/evictions, bytes live;
- tiles skipped by intervals, and conservative-dependency fallbacks;
- per-output value ranges and NaN/Inf counts, which are hard errors in debug;
- for encoding: mip levels, compression, the encoder version, and the
  parameters each packing profile could not carry.

**Debug views:**
- realize any intermediate port as an image;
- tile-dirtiness overlays after an edit;
- footprint heat maps, showing where band-limiting dropped octaves;
- a DOT dump of the compiled execution graph, with dapple op descriptions.

## Milestones

Each milestone is a vertical slice, finished to the quality bar rather than
stubbed.

1. **Foundations.**
   - `openpbr` extracted from lightweald: agree it with lightweald, and
     lightweald switches to it.
   - `dapple_field`: plane and periodic domains, value/gradient noise, fBm,
     cellular, footprints.
   - Deterministic hashing and golden hashes across platforms.
2. **The graph and incremental realization.**
   - `dapple_graph` on `execution_graph` (the executor branch, once landed),
     tile invalidation, fingerprints, the cache and reports.
   - Early cutoff: `values_equal`, upstream in execution #98.
3. **Sylva's first sets (unblocks sylva milestone 2).**
   - Rasters: blur, height → normal, AO, distance transform.
   - `dapple_imaging` coverage for leaf contours.
   - `dapple_encode` with coverage-preserving and normal/roughness mips, PNG
     and KTX2, and the `lightweald` and `gltf` profiles.
   - Deliverables: an oak bark material (tileable, periodic) and an oak leaf
     set (plane domain: opacity, base color, normal, translucency, i.e.
     `subsurface_color` with thin-walled).
4. **Solid and chart domains.**
   - Solid wood grain and stone.
   - `dapple_exedra` chart baking on exedra's construction charts.
   - Timbers from the pavilion as the first consumer.
5. **Library breadth.**
   - Brick/tile/herringbone, scatter, material layering, weathering (edge
     wear, dirt, moss).
   - The Heitz–Neyret by-example blend.
   - A material gallery rendered in lightweald.
6. **Runtime procedural detail.**
   - Field IR → Slang for lightweald shaders (micro-detail, anti-repetition),
     plus a GPU preview backend.

## Open decisions

1. **The `openpbr` crate's home:** decided: its own forest-rs repo,
   published on crates.io.
2. **Shared hash/random crate:** decided: `exedra_math::keyed`, version 1
   of the keyed-hash contract, frozen by exedra_math's ADR-0001. Dapple
   re-exports it from `dapple_field::hash`; sylva uses the same contract.
3. **Upstream early cutoff in `execution_graph`:** decided: an
   executor-supplied `values_equal` (execution #98); dapple compares by
   fingerprint and, for rasters, unchanged tiles.
4. **How far shapes go through `imaging`:** raster coverage through a CPU
   backend first; whether an exact analytic coverage evaluator belongs in
   imaging, windfoil or dapple.
5. **Lightweald slots:** which of the parameters that vary spatially and lack a
   slot today (`subsurface_color`, `transmission_color`, `emission_luminance`,
   geometry tangent) lightweald adds. Leaves want `subsurface_color` first.

## References

- OpenPBR Surface specification 1.1.1, Academy Software Foundation;
  adobe/openpbr-bsdf.
- Perlin, *Improving Noise*, SIGGRAPH 2002; Worley, *A Cellular Texture Basis
  Function*, SIGGRAPH 1996.
- Lagae et al., *A Survey of Procedural Noise Functions*, 2010 (band-limiting
  and filtering).
- Castaño, *Computing Alpha Mipmaps*, 2010.
- Toksvig, *Mipmapping Normal Maps*, 2005; Olano & Baker, *LEAN Mapping*,
  2010; Kaplanyan & Hill et al., *Filtering Distributions of Normals for
  Shading Antialiasing*, 2016.
- Barré-Brisebois & Hill, *Blending in Detail* (RNM), 2012.
- Heitz & Neyret, *High-Performance By-Example Noise using a
  Histogram-Preserving Blending Operator*, HPG 2018.
- Felzenszwalb & Huttenlocher, *Distance Transforms of Sampled Functions*,
  2012.
