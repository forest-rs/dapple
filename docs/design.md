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

A raster can go back into a field through `Op::Sample` (bilinear, with the
raster's mip chain filtering by footprint, trilinearly between levels). The
op holds its texels, fingerprinted by the producing nodes' derivation rather
than by content, and never serializes: a recipe names the sampled raster
nodes instead. The graph can therefore mix freely, but every
resolution-dependent choice is visible in the graph and in reports.
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
- `Color` is **always linear, with declared primaries**. Dapple's working
  space is linear Rec. 709/sRGB primaries by default; that is dapple's
  declared choice, not OpenPBR's (OpenPBR takes ACEScg when color-space
  metadata is absent), so the primaries travel with every color across
  every boundary. sRGB transfer encoding
  exists **only** at the output-encoding stage. There is no "sRGB color" port
  type, so gamma-space blending is impossible to write by accident.
- `Vector2`, `Vector3`.
- `Normal` in a declared frame: tangent space with a stated handedness and
  green-channel convention (+Y, OpenGL-style, matching glTF; DirectX-style
  appears only as an output-profile flip). Normals never blend by lerping
  components: detail composes by reoriented normal mapping in
  `dapple_material`'s detail application, and selections that must mix
  them average, renormalize and report it.
- `Direction` for tangent/anisotropy directions: a 2D angle field in tangent
  space. It needs a π-periodic, sign-free representation when averaged or
  blurred; this matters for mips.
- `Mask` is scalar coverage in 0–1. Its reductions are an explicit policy:
  area average by default, or threshold-coverage preservation for
  alpha-tested cutouts.
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

**As built.** One program IR covers both dimensions. Every node has a
`Space`: `Planar(Domain)` or `Solid(Domain3)`, where `Domain3` is `Space` or
`Periodic3`.

- **3D leaves** (`Constant3`, `Noise3`, `Fractal3`, `Cellular3`, `Position3`)
  and `Transform3` are their own ops. The existing 2D ops and their
  fingerprints are unchanged.
- **The other ops** (arithmetic, clamp, remap, mix, fract, length, vectors,
  colors, and `atan2` for angles around an axis) work in either space, but
  refuse inputs from both.
- **`Slice { origin, u, v, domain }`** is the only way from a solid field to a
  planar one. It can be periodic when `u·px` and `v·py` are lattice vectors of
  the solid period, and it scales footprints by the larger singular value of
  `[u v]`.
- **Evaluation** carries a 3D point and gradient through every context.
  Planar kernels read `x` and `y` only, so planar results are bit-identical to
  a purely 2D evaluator. Slices and solid transforms are context moves in the
  flat plan, with their own gradient chain rules.
- **Programs:** `finish_solid` yields a `SolidProgram` (a `SolidField`).
  `SolidProgram::eval_chart` evaluates it at texel points in its solid space.
  That is the seam `dapple_exedra` chart baking fills in: the chart supplies
  each texel's point and footprint. Normals and tangent frames join the chart
  sample when a node needs them.
- **Static bounds** extend to solid nodes, with the slope bounding
  `|∂x| + |∂y| + |∂z|`.

**Chart baking as built.** `dapple_exedra::SurfaceBake` takes one region of
an extracted `exedra_mesh::TriMesh` (an extrusion face, say, which exedra
charts as a single UV island in recipe units) and rasterizes it at a texel
density.

- **Samples:** every covered texel records its surface point, mapped through
  a placement into the material's solid space (a timber inside its log), the
  interpolated normal, and a footprint of `sqrt(surface area / chart area)`
  texels. The first triangle to cover a texel wins; later overlaps are
  counted in the bake's stats rather than hidden.
- **Gutters:** padding texels copy their nearest covered texel (breadth
  first, a fixed number of steps), so bilinear and mip filtering near the
  island's edge never reach the background.
- **Placement on the mesh:** the bake reports a `KHR_texture_transform`
  offset and scale that map the mesh's own UVs onto the image, so the mesh
  is exported unchanged. Each region binds its own material slot through
  `exedra_assembly::Assembly::bind_region_slot`.
- **Evaluator-agnostic:** the bake only produces sample points; a solid
  program evaluates them through `eval_chart`, and its values are scattered
  back into the texel grid.

`examples/timber_bake` is the first consumer: a post, a beam and a pitched
rafter with plumb cuts, cut from different places in one oak log, exported
through `exedra_gltf` and rendered in Blender. Atlas packing of several
regions into one image, and raster ops that cross chart seams, are still to
come.

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

- Field values carry a `Change` (`Nowhere`, `Within { regions,
  footprint_scale }`, `Everywhere`) relative to the producing node's
  previous program. `Op::change_from` states an edit's region; the first op
  that can is `Op::Disk`. Pointwise ops carry their inputs' regions;
  `Transform` and `Demote` make a change unbounded.
- A warp keeps its warped input's change local when its displacements have
  static bounds. Every scalar node reports `StaticBounds`
  (`FieldProgram::bounds`): a value range and a slope bound on
  `|∂f/∂x| + |∂f/∂y|` that hold at every point and footprint. Noise, fractals
  and cellular distances are bounded by construction (with provable
  constants, not the measured ranges their docs quote); arithmetic, clamps,
  remaps, mixes, transforms and warps propagate bounds; a sample image
  bounds its texels and their steps once when built. A warp then reads its
  input at most `|amount| · max|d|` away per axis and at a footprint at most
  `1 + |amount| · max(slope)` times wider, so the input's regions grow by
  that reach and their footprint growth (`footprint_scale`) by that factor.
  Displacements without bounds (cell values, a hard disk, vector
  components) leave the change unbounded, and `TileReport::unbounded_warps`
  counts it.
- Realize nodes re-realize the tiles whose texel centers fall in the change
  grown by `footprint_scale` half footprints (`realize_into`); raster ops recompute tiles
  with `RasterOp::apply_into`. Both equal whole passes bit for bit.
- Instead of fingerprinting tiles, a node compares each recomputed tile's
  bits with its previous output and marks only the direct dependents of
  tiles that changed, so an edit that leaves a tile unchanged stops there.
- Global ops (no footprint) have no tile edges and recompute whole when
  their input changed. `TileReport` counts recomputed, reused and changed
  tiles, whole recomputes, unbounded changes and unbounded warps.

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
level-above tiles their filter taps read. Sample nodes mirror the tiles of
the rasters they sample; the tiles whose bits changed, grown by one texel for
the bilinear taps, become the sampled field's change regions, so realizing a
resampled field recomputes only what an edit reaches. Still to come:
bounded-warp dependencies, so a warp of a sampled field can stay local too.

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

**As built** (`dapple_imaging`): scenes are drawn in domain units and
rasterized over a `dapple_raster::Realization` by `imaging_vello_cpu`'s 8-bit
pipeline, at a pinned `forest-rs/imaging` revision. Coverage is the
composited alpha in steps of 1/255; curves become chords within a quarter
texel, the renderer's fixed tolerance. A wrapping realization draws the scene
one period away in every direction, so shapes wrap. `coverage_image` builds
the mip chain from exact area means of level 0, not by re-rasterizing each
level, whose chords would be a quarter of a much larger texel. A golden
digest pins the output.

**Patterns:**
- tile and brick samplers (running, stack, herringbone later), with per-tile
  IDs, random offsets and bevel profiles;
- scatter (Poisson-disc splats of a sub-field) with bounded overlap, so tile
  dependencies stay local.

**As built** (`Op::Tiling`, `dapple_field::Tiling`): two layouts, both of
axis-aligned rectangles. A bond is rows of equal tiles with a row-to-row
shift (0 for a stack bond, 0.5 running, 1/3 or 1/4 raking); herringbone lays
tiles `ratio` widths long alternately along x and y in stairs, repeating
every `2 · ratio` widths. Each lookup gives the exact distance to the tile's
nearest joint in domain units, the position along and across the tile, its
orientation, and a per-tile value keyed by the tile's wrapped lattice
anchor, so periodic layouts repeat their tiles exactly. Periodic domains
refuse layouts that would not tile: whole tiles per period, whole row turns
of the shift, whole herringbone repeats. Bevels, mortar and per-tile tone
are built from these in the graph (`dapple_library`'s `brick` and
`parquet`), not baked into the op; per-orientation grain mixes two
anisotropic fields by the orientation output.

**As built** (`Op::Scatter`, `dapple_field::Scatter`): at most one splat
per lattice cell, present with a given density, centered anywhere in its
cell, with a random radius of at most one cell and optionally a random
turn. A point is therefore covered only by splats of the 5 × 5 cells around
it, whatever the density, so cost and tile dependencies stay bounded. A
splat stamps a soft disk, a dome, or a `SampleImage` (a leaf drawn with
`dapple_imaging`, say) read with its mips at the footprint scaled into the
stamp. Outputs are order-independent: the union coverage `1 − Π(1 − aᵢ)`,
the highest stamp (overlapping pebbles), and that splat's own random value
for per-splat tone. Splatting a whole sub-graph, rather than a leaf stamp,
needs evaluation contexts per splat and is left for later.

**Tone:**
- levels, curves (monotone cubic), clamp, remap, gradient map (a color ramp
  evaluated in linear space);
- blend modes defined on linear values, including height blend (max-height
  selection with a transition width);
- normal detail by RNM, in material detail application (slice 2).

**Rasters:**
- Gaussian blur, separable and in physical units;
- height → normal (Sobel or central difference, scaled in world units);
- AO from height (horizon-based);
- curvature;
- distance transform (exact Euclidean, Felzenszwalb–Huttenlocher);
- flood fill to IDs.

**Material outputs:** bind nodes to OpenPBR parameters and auxiliaries.

### Later

The [milestones](#milestones) now order this list: weathering, the
by-example blend and material layering wait for structured surfaces and
reusable materials, and come back as modules.

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
  blended by its own rule (colors linear, roughness in α space). Slice 2
  replaced this with four operations with separate contracts.

## Output and encoding

**Mip chains** are built by dapple, not by a generic downsampler. Each port type
has its own rule:
- **Color:** box or Kaiser filtering in linear light, with premultiplied
  alpha where there is opacity.
- **Masks and opacity:** area average by default; for alpha-tested cutouts,
  *coverage-preserving* (Castaño 2010), where each level's threshold is
  solved so that the fraction of texels above the consumer's alpha cutoff
  matches level 0. The cutoff is part of the packing profile and the policy
  is opt-in. Without it, alpha-tested foliage thins away at distance.
- **Normals:** filtering averages unit normals and records the resulting
  length loss. Rather than simply renormalizing, **the lost variance moves into
  roughness** (Toksvig; LEAN/Kaplanyan–Hill style). Each level's
  `specular_roughness` (and `coat_roughness` for the coat normal) is widened
  so distant surfaces keep their energy instead of turning mirror-like and
  sparkly. Anisotropic variance can feed `specular_roughness_anisotropy`
  later.
- **Directions** use π-periodic averaging.
- **IDs** use point or mode over the level-0 footprint, both lossy
  summaries (see slice 0).
- **Footprint-aware fields can skip filtering entirely:** a level can be
  *re-evaluated* at its own footprint instead of downsampled. How good that
  level is depends on the compiled expression's sampling contract (see
  *Sampling correctness* in the milestones): exact where every op integrates
  exactly under the stated filter, approximate or heuristic otherwise. It is
  often cheaper than it sounds.

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
| `dapple_material` | Material values (OpenPBR bindings, auxiliary channels), the four material operations and their reports, programs over materials, modules, resources | yes |
| `dapple_encode` | Per-type mip builders, specular AA, packing profiles; PNG/EXR/KTX2 writing and the `ctt` compressor behind `std` | core yes |
| `dapple_imaging` | Imaging scenes → coverage fields | yes |
| `dapple_exedra` | `Chart` domain: rasterized chart layouts, seam gutters | yes |
| `dapple_library` | Ready-made materials (oak bark, solid oak) as field programs | yes |
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

### Done

1. **Foundations:** `openpbr` extracted from lightweald; `dapple_field`
   with plane and periodic domains, value/gradient noise, fBm, cellular
   noise and footprints; keyed hashing with golden values across platforms.
2. **The graph and incremental realization:** `dapple_graph` on
   `execution_graph`, tile invalidation, fingerprints, the cache, reports,
   and early cutoff through `values_equal` (execution #98).
3. **Sylva's first sets:** blur, height → normal, AO and distance
   transforms; `dapple_imaging` coverage; `dapple_encode` with per-type
   mips, PNG and KTX2, and the `lightweald` and `gltf` profiles; oak bark and
   an oak leaf set.
4. **Solid and chart domains:** solid wood grain, `dapple_exedra` chart
   baking, and the pavilion timbers (`examples/timber_bake`) as the first
   consumer.

Milestone 5 began as library breadth. Tile layouts (bonds, herringbone)
and scatter landed with brick, parquet and gravel, and then it stopped on
purpose, for the reason below.

### Structured content before breadth

Two external reviews (September 2026) found that dapple's gap is not more
noises or effects but **data models for structured material content**. Brick,
parquet and gravel each work, but each is a scalar field that forgets what
it is drawing: a brick's identity exists only as a hash inside
`Op::Tiling`, so the bevel, the glaze, the chips and the exposed body
cannot agree about which brick they belong to except by rebuilding the same
layout in every program. New material families then mean kernel edits
(another op, another output enum) instead of library content.

The success criterion for the whole sequence: **a new material family is
usually a new recipe library, not an edit to the kernel's operation enum.**
Material families (moss, weathered brick, varnished oak) are library
modules. New kernel primitives are introduced when they provide reusable
computational capabilities that the existing program model cannot express
adequately: histogram transforms, global reductions, neighborhood
operations and iterative solvers can be legitimate primitives.

The next slices therefore build the data models first. Each slice carries
its own **acceptance gates**: a small, executable piece of the sampling,
execution and laboratory work, so those never drift into a backlog "after
the interesting features". The gallery accompanies the gates; it never
substitutes for them.

#### Identity, change and correspondence

Four concepts stay distinct everywhere they appear:

| Concept | Meaning |
|---|---|
| **Identity** | Which authored or generated element this is. |
| **Content fingerprint** | Whether its current geometry, attributes or dependencies changed. |
| **Dense index** | Where it happens to be stored in this realization. |
| **Correspondence** | How elements or regions in one version relate to those of another. |

- **Identity does not depend on position.** An explicit transform of an
  existing element preserves its identity and everything derived from it:
  moving a brick keeps its glaze variation and its chips.
- **A generator's identity is its logical identity plus the element's
  anchor and slot**, never the generator's content fingerprint. Changing the
  mortar width changes every brick's fingerprint and none of their
  identities. Position, jitter, rotation and dimensions are properties of an
  element, not its identity.
- **Topology changes may replace elements.** Which edits replace which
  elements is documented per generator, with a correspondence where one is
  known, rather than being an accident of hashing.
- **Canonical reconstruction is the default.** The current inputs alone
  determine labels, keys and bytes. Matching a result against a previous one
  produces a separate *correspondence report* and never changes the
  canonical result, so opening a final recipe and baking it agrees with
  reaching the same recipe through a history of edits. Authored element
  identities are preserved whenever they exist. Retained identity is
  possible only as *persisted identity state* that is serialized in the
  material document and consumed by a clean rebuild; an incremental cache
  never becomes an authoring database.

#### Slice 0: semantic types and sampling policies through the whole graph

Field programs already type their ports (`Scalar`, `Mask`, `Id`,
`Vector2`, `Vector3`, `Color(primaries)`, `Normal(frame)`, `Direction`);
realization forgets them. `RasterData` is a scalar or three-channel raster,
`Sample` reads scalars, and mips, caches and recipes know nothing of
meaning.

- **Realized values preserve their semantic type.** A raster carries its
  `PortType`, with storage to match: `u32` for IDs, two and three channels
  for vectors, colors, normals and directions.
- **Sampling and reduction policies are explicit, validated against the
  type, and part of derivation identity.** A type says which operations are
  meaningful; it does not say which meaningful operation a consumer needs.
  A continuous selection mask keeps its fractional weights under averaging,
  while an alpha-test cutout may instead preserve the fraction above a
  threshold, and applying the second rule to a blend mask changes its
  meaning. So every type has *permitted* reduction policies, each stating
  what it retains and what it loses, with a documented default:
  - scalars and masks: area average (default) or, for masks,
    threshold-coverage preservation at a stated cutoff;
  - colors: area average in linear light with their primaries;
  - normals: area average **retaining the mean length** alongside the
    direction, so a later variance-to-roughness transfer still has the
    information it needs; renormalization happens where a consumer asks;
  - directions: axial (doubled-angle) averaging;
  - vectors: per-component average;
  - IDs: never filtered. Their only reductions are *point* (the texel at a
    level's sample point) and *mode over the level-0 footprint* (the most
    frequent label among the level-0 texels the texel covers, ties to the
    smallest key). Either is a **lossy summary**: it does not describe
    every region a texel covers, and repeated majority is not majority
    over the original footprint, which is why the promise is stated over
    level 0.
- Raster ops declare the types and policies they accept, so blurring an ID
  is a graph-build error, not a wrong image. `dapple_encode`'s per-type
  rules become the defaults the graph also uses.
- **Gates:** every port type survives realize → reduce → sample under each
  permitted policy with its stated retention; an unpermitted policy is
  refused when the graph is built; policies change fingerprints; a normal
  chain keeps its mean length through the graph and the packer's roughness
  adjustment matches a direct computation.
- *As built:* `TypedRaster` storage and `ReductionPolicy` in
  `dapple_raster::typed`; `SampleImage` keeps a `PortType` and samples
  under a `SamplePolicy` (`Linear` per component and never renormalized,
  or `Nearest`, the only policy identifiers permit), which `Op::Sample`
  fingerprints. Material graphs derive every node's type when it is added
  (`Op::port_type_with` for field ops), so a refused type or policy is a
  `MaterialError::Refused` from the builder, and an edit that would
  mistype any node is refused and leaves the graph unchanged. The gates are
  `dapple_graph` tests (`every_type_survives_realize_reduce_sample` among
  them).

#### Slice 1a: keyed elements and one coherent material

- **Element sets** (`dapple_elements`): elements with identity (above), a
  transform, bounds, a variant and typed attributes, stored as a column
  table. The *layout* (which elements exist and where) is separate from
  *realization* (how they are drawn), so one layout drives every output
  coherently instead of rerunning similar random choices per channel.
  First operations: layout (bonds from today's `Tiling`), filter,
  per-element transform, and composite.
- **Compositing ownership is explicit** from the first implementation:
  - *partition / winner*: one element owns a point, with a deterministic
    tie-break;
  - *coverage compositing*: several elements contribute, under a declared
    order or an order-independent rule;
  - *summary labels*: an exported owner label is identified as a summary.
    Contributors can be recomputed on request; the representation never
    equates "the dominant element" with "the only contributor".
- **Element identity and surface-material identity are separate.** The
  glaze and the exposed ceramic belong to the same brick while being
  different materials.
- **A minimal callable-program contract** comes forward from slice 2 so an
  element can invoke a reusable surface program: named typed inputs, named
  typed outputs, explicit resource dependencies, a stable instance identity
  and an inspectable body. Rust builders are fine, but they produce
  *inspectable values*, never opaque closures, so serialization, dependency
  tracking and later shader translation need no rewrite. Each input
  declares its **execution scope** (per material, per element, per
  sample), and the dependencies are checked: a per-element value may depend
  on material parameters and element attributes, never on the sample
  position. The compiler may hoist further work when it can prove it.
  *Implemented* as `dapple_elements::program`: a small expression body
  (`Node`) that samples field programs as declared resources, rather than
  new ops in the field IR. Slice 2's programmable IR core absorbs it; its
  `SurfaceProgram` value and scope rules are the contract that carries
  over.
- **Demo: glazed brickwork.** One brick's identity drives its shape, bevel,
  glaze (tone, thickness, pooling toward the lower edge), chips, and the
  ceramic body the chips expose.
- **Gates** (executable tests, not gallery images):
  - moving one brick keeps its key and identity-derived appearance, and the
    affected work covers both its old and new bounds;
  - changing one brick's glaze leaves unrelated element attributes and
    unrelated output regions unchanged;
  - changing realization resolution leaves element identity independent of
    raster labels and storage order;
  - reaching the same document through different edit histories gives the
    same canonical output;
  - tile scheduling and incremental evaluation agree with a clean reference;
  - a mixed boundary texel reports ownership labels and contribution
    semantics without confusing the two.

#### Slice 1b: the structural vocabulary

- **Region maps and region tables:** an integer label raster (an `Id`) and
  a table per label: identity, centroid, bounds, area, orientation,
  neighbors and provenance. Regions composited from element sets keep the
  elements' identity; regions reconstructed from rasters (a flood fill of a
  mask) are canonical, with split/merge correspondence reported separately.
- **Curve networks:** polylines and curves with arc length, width profiles,
  tangents and intersections, exposed as fields (distance, along and across
  coordinates) and as element layouts (stitches every 4 mm along a seam,
  fibers following a curve): joints, cracks, veins, grooves.
- **Scatter and instance** on element sets (from today's `Scatter`), with
  variants choosing shapes or sub-materials.
- **Morphology and shape processing:** dilate, erode, open, close and
  connected components on masks and per region; insets and bevel profiles
  from distance fields; edge extraction; per-region statistics.
- **Gates:** a region round-trips through composite → reconstruct with a
  correspondence that names every split and merge; curve-driven element
  spacing is independent of realization resolution.
- *As built* (`dapple_elements`, `dapple_raster`):
  - `RegionMap`: a label raster and a region table in key order. Composited
    regions (`from_composite`) keep their elements' keys; reconstructed
    ones (`reconstruct`, a flood fill that continues across a wrapping
    raster's edges) are keyed by their anchors, the first texel in
    row-major order. `correspondence` groups regions linked by shared
    texels into matches, splits, merges, regroups, appearances and
    disappearances. `retaining` rekeys a canonical map from a previous one,
    the persisted identity state: the largest overlap keeps its parent's
    key, so one child of a split and the result of a merge keep identity.
    Per-region `inset` (exact distance transforms per region),
    `boundaries`, `mask` and `statistics`.
  - `Morphology` (dilate, erode, open, close by an exact disk in domain
    units) is a tiled raster op, and a graph raster node that keeps masks
    masks.
  - `CurveNetwork`: polylines with arc length and width profiles, the
    nearest curve's frame (distance, along, across, tangent, width) as
    `ScalarField`s, a stroke box-filtered across the curve, intersections,
    and `stitches` every `spacing` of arc length keyed by curve and index.
  - `ScatterLayout` places elements exactly where `dapple_field::Scatter`
    places splats (`Scatter::splat_at`), so `Op::Scatter` is its
    field-level lowering; variants choose an `Outline` (rectangle or
    ellipse) and aspect, and `Binding::Variant` lets a program choose
    sub-materials.
  - The gates are tests: `regions_round_trip_and_name_every_split_and_merge`
    and `curve_stitch_spacing_is_independent_of_resolution`.

#### Slice 2: reusable materials

- **Typed multichannel material values:** a material is its OpenPBR
  parameters, each bound to a constant or a typed output, with auxiliary
  channels (height, region) kept separate.
- **Four material operations, each with its own contract**, rather than one
  channel-wise blend under different names:

  | Operation | Contract |
  |---|---|
  | **Spatial selection** | Select or transition between materials with one shared selection decision across every channel; parameter-space approximations (for example roughness interpolated in α) are documented as approximations. |
  | **Detail application** | Apply height, slope or normal perturbation in a specified frame, with a defined identity operation. Reoriented normal mapping belongs here, and only here. |
  | **Optical coating** | Keep base and coat parameters separate where the target represents them (OpenPBR's coat is a dielectric layer that transmits without scattering). |
  | **Deposit or covering** | A higher-level recipe (moss, dirt, snow) that may change coverage, geometry, material selection and optical properties together. |

  Approximation is **reported where it happens**, in the material
  operation that collapsed a richer combination, not only when the packer
  lowers the result. Dapple stays a material compiler; it does not grow a
  BSDF evaluator.
- **Parameterized modules:** a public interface of typed parameters with
  units, ranges and defaults, resource inputs and named outputs, with a
  versioned identity (an oak board exposes board width, grain scale,
  finish, weathering and seed, not `warp_17.amount`). Instantiation binds
  parameters explicitly and derives seeds from the instance path;
  diagnostics keep the module boundary even when compilation inlines it.
- **A programmable field-IR core:** coordinate access, a fuller arithmetic
  and vector vocabulary, comparisons and selection, reusable function
  calls and bounded iteration where a workload justifies it, plus
  registered operations with declared contracts (output type,
  dependencies, sampling behavior, deterministic implementation, supported
  backends) instead of opaque closures.
- **Execution scopes** extended to per region and per raster pass.
- **Host-resolved resource inputs:** images and exemplars a host supplies,
  with content identity, semantic type, color information, physical scale
  and mip policy; decoding and file access stay outside the `no_std`
  kernel. The Heitz–Neyret blend and weathering (edge wear from curvature,
  dirt from AO, moss by orientation) are modules built here.
- **Image processing by workflow:** value shaping (curves, ramps, smooth
  thresholds, histogram measurement, percentile remapping, controlled
  normalization) and directional processing (directional blur,
  slope-driven sampling, vector displacement, later advection). The
  scheduler knows each op's category (local stencil, separable pass,
  reduction, global transform, iterative solve) so locality, parallelism,
  work reporting and cancellation stay coherent.
- **Gates:** the same stone, wood and finish modules reused in several
  assets without duplicating graphs; a material-wide transform moves every
  channel; each material operation reports its approximations.
- *As built* (`dapple_field::scoped`, `dapple_raster`, `dapple_material`,
  `dapple_library::modules`). The decisions, in the order they were made:
  - **The programmable core is `dapple_field::scoped`.** Slice 1a's
    `SurfaceProgram` moved there as `ScopedProgram`/`ScopedBuilder`;
    `dapple_elements` keeps only bindings and instances. Its vocabulary
    grew by division, powers, unary math, comparisons (identifiers compare
    for equality only), dot and cross products, normalization, monotone
    tone curves and linear-light color ramps (`dapple_field::shaping`), and
    calls of other scoped programs as functions, whose results take the
    scope their arguments give them. Coordinates stay inputs bound by the
    evaluator. Bounded iteration waits for a workload that needs it; the
    registered operations with declared contracts are, for now, the field
    programs a scoped program samples as resources.
  - **Scopes are a lattice**, not a chain: material, region, element,
    sample, pass. Region and element values are independent, so their
    join is per sample. *Pass* is a value that needs a raster pass (a
    neighborhood or reduction) to exist; an output declared per sample
    can never depend on one, so per-sample outputs stay point-evaluable
    and shader-translatable. `RegionMap::evaluate` runs region-scope nodes
    once per region; a composited region's random stream equals its
    element's.
  - **Material values are realized.** A `Material` binds each OpenPBR
    parameter (the `openpbr` crate, pinned by git revision) to a constant
    or a `TypedRaster`, and keeps auxiliary channels (height in meters,
    occlusion, region label, surface identity) apart, all on one `Grid`.
    One grid is what makes "one decision for every channel" literal, and
    weathering needs raster passes anyway; modules stay resolution
    independent until instantiated on a grid. Material operations run
    whole-raster and do not yet join `dapple_graph`'s tile invalidation.
  - **Four operations, four contracts** (`dapple_material::ops`), each
    returning a `Report` of approximations counted on the texels where
    they made a difference:
    - `select`: one weight per texel for every channel (a mask, or a
      height-based transition). Colors, weights and heights mix linearly;
      roughness in α (`RoughnessInAlpha`); normals and tangents averaged
      and renormalized (`NormalsAveraged`); indices of refraction
      interpolated (`IorInterpolated`); identifiers by the larger weight
      (`WinnerLabel`). Coat parameters mix weighted by each side's coat
      weight, so a side without a coat leaves the other's coat exact, and
      differing layer stacks report `LayersMixed`.
    - `apply_detail`: height and normal perturbation in a stated layer
      (base, coat, both) in the domain frame, with a bit-exact identity.
      Reoriented normal mapping lives here and only here:
      `Op::BlendNormals` left the field IR. A coat-layer height ripples
      the coat normal and never displaces; a bound base normal follows
      added height so normal and displacement stay in step.
    - `coat`: sets OpenPBR's coat over an untouched base, with an
      optional thickness added to the height. A second coat over a first
      collapses into OpenPBR's one (`CoatsCollapsed`).
    - `deposit`: a covering's own material selected over the base by
      coverage, coat included (what it covers it hides), its thickness
      added to the height, its surface identity where it covers at least
      half a texel; region ownership stays the base's.
    - `transform`: mirror, quarter turns and offset for every bound
      channel, vector channels turned with the texels; whole-texel moves
      are exact permutations, fractional ones resample and report
      `Resampled`.
  - **`geometry_normal` is the shading normal of the undisplaced frame.**
    When it is unbound, lowering derives it from the height; a consumer
    that displaces uses the height and not both.
  - **Modules are Rust types with data interfaces.** A `Module` publishes
    an `Interface` (versioned `ModuleId`, parameters with units, ranges
    and defaults, material, map and resource inputs, named outputs) and
    builds from checked `Args`. The body is Rust, identified by its
    `ModuleId` as a registered function is by its name; its inputs,
    arguments and results are inspectable values, and an instance
    (module, version, arguments) is data. A serialized form for module
    *bodies* is not decided yet (see open decision 8).
    Instances have paths (`wall/glaze`); seeds hash the path and the
    `seed` parameter, and `Context::record` files every operation's report
    under the instance that ran it, so diagnostics keep module boundaries
    though the material is flat.
  - **Host-resolved resources** (`dapple_material::resource`): a module
    declares a `ResourceRequest` (semantic type, whether it must tile),
    an instance names a `ResourceRef`, the host resolves it into a
    `SampleImage` in meters (physical scale), linear with its primaries
    (color information), with the host's content fingerprint as its
    derivation (content identity) and the `ReductionPolicy` of its levels
    (mip policy), checked against the request. Decoding stays in the
    host.
  - **Value shaping and scheduling categories** (`dapple_raster`):
    `Histogram`, exact `percentiles` and `PercentileRemap` measure by
    order statistics, so a module states "8% covered" rather than a raw
    threshold; `Streak` is a one-sided directional smear of physical
    length. `RasterOp::category` names each op a local stencil, separable
    pass, reduction, global transform or iterative solve. Slope-driven
    sampling and advection are still to come.
  - **Lowering** (`lower::maps`) names what `dapple_encode`'s maps cannot
    carry, and the packer now packs coats (glTF `clearcoat`; coat tint is
    reported unsupported there).
  - **The gate** is `dapple_library`'s
    `modules_are_reused_across_assets_and_report_approximations`: stone,
    wood and finish modules appear in several of the assets
    (`GlazedBrickWall`, `StoneSill`, `VarnishedBoard`, `Threshold`), each
    module one definition instantiated by path; deposits and selections
    report their approximations at their instances; a material-wide
    transform moves every channel. `examples/glazed_brick` builds the wall
    and the sill at 2048² and renders them in Blender.

#### Slice 3: materials on objects

- **Surface evaluation context** through a host interface, with explicitly
  requested inputs: world, part and **stock-local** coordinates kept
  distinct (a timber moved into another assembly keeps its grain, and a
  fresh cut reveals the same stock-local material); tangent frames and
  derivatives; region identity; host-supplied fields such as thickness,
  curvature, exposure and distance to a boundary. Dapple owns no scene
  graph.
- **A chart-aware baking adapter** generalizing `dapple_exedra`: coverage
  per chart, seam padding, normal-frame conversion between chart and
  material, consistent sampling across charts, atlas packing, and reported
  errors or missing inputs.
- **Gates:** a cut timber or chipped plaster object whose new surfaces
  agree with the retained construction and material intent.

#### Cross-cutting, delivered as gates inside the slices

- **Sampling correctness.** Each op, and each compiled subgraph, states its
  guarantee in precise terms: *exact integration under a specified filter*,
  *frequency attenuation*, *heuristic fading to a mean*, or *point
  evaluation only*. A reference integration path evaluates the **complete
  underlying expression** at reference sample points and then integrates,
  because filtering each input does not filter a nonlinear graph
  (average(f²) ≠ average(f)²); tests compare whole expressions, not only
  single ops. Footprints become anisotropic as a local linear
  approximation: a 2 × 2 covariance in a 2D parameter space, and on a
  surface the 2D footprint plus its differential basis, so that for a
  surface map with 3 × 2 Jacobian `J` the material-space footprint
  `J Σ Jᵀ` (3 × 3, rank ≤ 2) keeps its orientation. Subpixel variation can
  survive as coverage, normal variation, roughness or a distributional
  approximation rather than always fading to a constant.
- **Execution scale:** demand-driven outputs (a consumer requests an
  output, region, resolution or mip); tile storage with eviction and
  reconstruction, later disk caches; multi-output compilation so related
  channels share one program; deterministic parallel CPU evaluation;
  explicit backend policies (a reproducible reference with pinned numerics,
  accelerated backends with declared tolerance, a shader-compatible subset)
  in cache keys; metrics for peak resident bytes, bytes copied, primitive
  evaluations, tiles recomputed, compilation cost and work avoided.
- **A headless material laboratory:** parameter sweeps, contact sheets,
  close-ups, grazing-angle views and mip/LOD comparisons through an
  external renderer, with machine-readable reports (ranges, invalid values,
  seam errors, unsupported export channels, incremental-versus-clean
  agreement) and relationship tests: a seed change never violates
  constraints; **raising resolution never changes physical feature size**;
  moving a shape across a periodic boundary stays consistent; an
  incremental edit equals a clean recompute; a material-wide transform
  leaves no channel behind.
- **A portable material package:** module interface, graph, dependencies,
  resources or resource requirements, presets, semantic output
  declarations and expected engine capabilities, with the editable source
  representation kept distinct from an optimized execution artifact.

**Beyond parity: authoring by constraints and measurements.** Instead of
exposing twenty unrelated knobs, a material can target measurable
statements: roughly 8% exposed substrate, chips concentrated near
boundaries, grain spacing within a range, average appearance retained as
detail goes subpixel. This starts with measurable outputs and parameter
sweeps; optimization comes later (spatial gradients are not gradients with
respect to authoring parameters).

**Afterwards:** library breadth as modules; runtime procedural detail (field
IR → Slang for lightweald shaders, micro-detail and anti-repetition, and a
GPU preview backend).

## Open decisions

1. **The `openpbr` crate's home:** decided: its own forest-rs repo,
   published on crates.io.
2. **Shared hash/random crate:** decided: `exedra_math::keyed`, version 1
   of the keyed-hash contract, frozen by exedra_math's ADR-0001. Dapple
   re-exports it from `dapple_field::hash`; sylva uses the same contract.
3. **Upstream early cutoff in `execution_graph`:** decided: an
   executor-supplied `values_equal` (execution #98); dapple compares by
   fingerprint and, for rasters, unchanged tiles.
4. **How far shapes go through `imaging`:** decided for now: raster
   coverage through `imaging_vello_cpu`, pinned by git revision. Still open:
   whether an exact analytic coverage evaluator belongs in imaging, windfoil
   or dapple.
5. **Lightweald slots:** which of the parameters that vary spatially and lack a
   slot today (`subsurface_color`, `transmission_color`, `emission_luminance`,
   geometry tangent) lightweald adds. Leaves want `subsurface_color` first.
6. **Where structured surfaces live:** decided: a new `no_std` crate,
   `dapple_elements`, for element sets, region maps and tables, and curve
   networks, depending on `dapple_field` and `dapple_raster`; `Op::Tiling`
   and `Op::Scatter` stay as the field-level lowering of layouts without
   attributes.
7. **Element keys:** decided: 64-bit keys from the keyed hash of the
   layout's *logical* identity (a name given by its author, never its
   content fingerprint) and the element's anchor and slot. Keys are
   independent of position and transform, so moving an element keeps its
   key; parameter edits that keep topology keep every key; topology changes
   replace elements as each generator documents. Collisions are refused per
   set.
8. **The module format:** decided in slice 2: modules are Rust types with
   a data `Interface` and a versioned `ModuleId`; instances (module,
   version, arguments, path) are data, and bodies are registered Rust
   builders that produce inspectable values, never opaque closures. A
   serialized form for module bodies beside `dapple_graph::Recipe` stays
   open until the portable material package needs it.

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
