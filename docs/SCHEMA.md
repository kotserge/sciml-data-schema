# The HDF5 dataset format

The layout of one `sciml-data-schema` file, the field model behind it, and the contracts
producers and consumers follow. The package is the reference implementation:
`src/sciml_data_schema/schema.py` holds the in-memory objects and the canonical names,
`src/sciml_data_schema/io.py` the serialization.

The format derives from the data format of
[The Well](https://github.com/PolymathicAI/the_well) and restructures it around two
principles:

1. **One core, several profiles.** A single *core convention* governs metadata, fields,
   normalization, boundary conditions and provenance. Each *spatial representation* (grid,
   mesh, …) is a **profile** that differs from the others *only* in how the spatial domain
   and the field arrays are encoded.
2. **A unified field model.** Every physical quantity — solution, forcing, coefficient,
   parameter — is a *field*. There is no separate "scalars" group. A quantity carries two
   orthogonal pieces of metadata: its **tensor order** (t0/t1/t2) and its **extent flags**
   (`dim_varying` / `time_varying` / `sample_varying`). A "scalar parameter" is a rank-0
   field that does not vary in space or time.

Scope ceilings, inherited from The Well: spatial dimensions ≤ 3 (and `0` for problems
without a spatial domain, e.g. ODEs: a time axis only); tensor rank ≤ 2.
Cross-representation training is not a goal — grid and mesh files are never mixed in one
batch — so the profiles agree on the core, not on the spatial layout.

---

## 1. The unified field model

A *field* is the single storage primitive. It is described by:

| Attribute        | Meaning                                                                 |
|------------------|-------------------------------------------------------------------------|
| `name`           | Canonical name, drawn from the **field registry** (§2).                  |
| `tensor_order`   | 0 (scalar), 1 (vector), 2 (rank-2 tensor). Carries rotation semantics.  |
| `role`           | `state` \| `forcing` \| `coefficient` \| `parameter` \| `boundary`.      |
| `unit`           | Physical unit string (e.g. `m/s`, `Pa`, `1/s`), or `dimensionless`.     |
| `dim_varying`    | `True` if the field varies across space; `False` if spatially uniform.   |
| `time_varying`   | `True` if it varies across the time axis.                               |
| `sample_varying` | `True` if it varies across trajectories.                                |
| `symmetric` / `antisymmetric` | For `tensor_order=2` only.                                 |

`name`, `tensor_order`, `role` and `unit` are the field's *identity*, fixed by the registry
and shared across files. The three `*_varying` flags are its *extent* in one file.

**Why this dissolves the scalar-vs-field jump.** The same physical quantity keeps one home
regardless of how much it varies. Refining a constant `κ` into a heterogeneous `κ(x)` is a
single flag flip (`dim_varying: False → True`), not a move between groups, so the schema is
stable under refinement.

**Storage stays cheap.** A field with `dim_varying=False` is stored *without* spatial axes
(one value, broadcast on read); a field with `time_varying=False` drops the time axis; one
with `sample_varying=False` drops the trajectory axis. Nothing materializes a redundant
array of identical numbers.

Fields are grouped **by tensor order** (`t0_fields`, `t1_fields`, `t2_fields`) because
tensor order governs how a quantity transforms. Tensor order is the only distinction kept
from The Well's group structure; the scalars group is gone.

### Resulting array shapes

The stored shape of a field is

```
(n_trajectories?, n_timesteps?, *spatial_shape?, *(n_spatial_dims,) * tensor_order)
```

with each leading axis present only if its flag is set. For a 2D grid profile, a `t0`
(scalar) field, depending on its flags:

| dim_varying | time_varying | sample_varying | Stored shape              | Example                        |
|-------------|--------------|----------------|---------------------------|--------------------------------|
| yes         | yes          | yes            | `(n_traj, n_steps, X, Y)` | solution `u(x,y,t)`            |
| yes         | no           | yes            | `(n_traj, X, Y)`          | static coefficient `κ(x,y)`    |
| no          | no           | yes            | `(n_traj,)`               | global parameter (e.g. `Re`)   |
| no          | no           | no             | `()` or `(1,)`            | a true global constant         |

A `t1` (vector) field appends a components axis `D = n_spatial_dims`; a `t2` field appends
`D, D`. The mesh profile replaces the spatial axes `(X, Y[, Z])` with a single DOF axis
`(N_dof)` — see §6. The validator (`Dataset.validate`) checks every field's array against
this rule.

---

## 2. The field registry (canonical vocabulary)

A per-file (but conventionally identical across all files of a representation) registry
that fixes the *identity* of every quantity: its tensor order, role and unit. It is what
makes **within-representation multi-physics** possible: `velocity` in a Rayleigh–Bénard file
and `velocity` in an Euler file are the *same* registered field, so a model can share
parameters across them.

Stored as a root JSON attribute `field_registry`, versioned by the root attribute
`registry_version` (independent of `schema_version`, §10):

```json
{
  "velocity":     {"tensor_order": 1, "role": "state",       "unit": "m/s"},
  "pressure":     {"tensor_order": 0, "role": "state",       "unit": "Pa"},
  "buoyancy":     {"tensor_order": 0, "role": "state",       "unit": "m/s^2"},
  "body_force":   {"tensor_order": 1, "role": "forcing",     "unit": "m/s^2"},
  "diffusivity":  {"tensor_order": 0, "role": "coefficient", "unit": "m^2/s"},
  "reynolds":     {"tensor_order": 0, "role": "parameter",   "unit": "dimensionless"}
}
```

A file declares *which* registry entries it holds in the root attribute `present_fields`,
so a one-field Poisson file and a five-field Euler file share the registry without storing
absent fields. The registry written to a file is the explicit one the producer supplies
(possibly a superset) completed by the identity of every present field
(`Dataset.effective_registry`); a present field whose identity disagrees with its explicit
entry fails validation.

---

## 3. How PDE concepts map to the schema

A generic PDE,

```
∂u/∂t = N(u; θ) + f      on Ω,    + boundary conditions,    + u(t=0) = u₀
```

maps as follows. **Role** is the bridge between PDE meaning and storage: it is what a loader
uses to split conditioning inputs from prediction targets.

| PDE concept | Symbol | Representation | role | typical flags |
|-------------|--------|----------------|------|----------------|
| **Solution / state** | `u` | field, tensor order by physical type (scalar→t0, velocity→t1, stress→t2) | `state` | `time_varying=True` (steady → `False`), `sample_varying=True` |
| **Initial condition** | `u₀` | *not a separate object*: the `t=0` slice of the state field | — | — |
| **IC-generation knobs** | seed, amplitude | rank-0 field | `parameter` | `dim_varying=False`, `time_varying=False`, `sample_varying=True` |
| **Volumetric forcing / source** | `f(x,t)` | field, tensor order by type | `forcing` | flags per its actual variation |
| **Boundary forcing** | inflow/flux profile | `boundary_conditions/<bc>/values` | — | lives in the BC group, see §4 |
| **Global parameter** | `Re`, `Ra`, `Pr` | rank-0 field, broadcast | `parameter` | `dim_varying=False`, `time_varying=False`, `sample_varying=True` |
| **Spatially-varying coefficient** | `κ(x)`, anisotropic `K(x)` | t0/t1/t2 field | `coefficient` | `dim_varying=True`, usually `time_varying=False` |
| **Domain geometry** | `Ω` | profile-specific (axes vs mesh) | — | see §5 / §6 |
| **Time axis** | `t` | `dimensions/time` | — | a coordinate vector |
| **Boundary conditions** | — | `boundary_conditions` group | — | type + mask + values (§4) |

The `boundary` role marks a field holding boundary data that is stored as a regular field;
the prescribed values attached to a named condition live in
`boundary_conditions/<name>/values`.

Beyond The Well's layout, the format adds the role tag, the field registry and the
provenance stamp. The Well folds `state`, `forcing` and `coefficient` into the same groups
and its loader windows everything as state; with the role tag, a loader routes `state` to
the prediction targets and `forcing` / `coefficient` / `parameter` to the conditioning
inputs, with no schema surgery.

**Steady-state problems** (e.g. Poisson) set `time_varying=False` on the state field and
omit the time axis. The same schema covers steady → time-dependent with one flag.

---

## 4. Shared core layout

Common to every profile. `@` marks an HDF5 attribute.

```
/                                   (root)
├── @schema_version        : "0.3.0"
├── @registry_version      : "0.1.0"
├── @dataset_name          : str
├── @representation        : "grid" | "mesh" | "particle"
├── @grid_type             : "cartesian" | "spherical"   # written for every file; the grid profile reads it
├── @n_spatial_dims        : 0 | 1 | 2 | 3                # 0 = no spatial domain (ODE); time axis only
├── @n_trajectories        : int
├── @field_registry        : JSON (see §2)
├── @present_fields        : list[str]                   # which registry entries are here
├── @provenance            : JSON object of strings      # optional; how the file was produced
│   # NB: train/validation/test splits are a downstream concern: one file per split, nothing stored here.
│
├── dimensions/
│   ├── @spatial_dims      : ["x", "y"]                  # axis names; their coordinates are profile-specific
│   └── time               : (n_steps,) float            # optional; absent for a steady dataset
│
├── boundary_conditions/
│   └── <bc_name>/
│       ├── @bc_type            : "periodic" | "dirichlet" | "neumann" | "wall" | "open"
│       ├── @associated_fields  : [str]                  # present fields
│       ├── @associated_dims    : [str]                  # entries of spatial_dims
│       ├── mask                : bool array             # optional; where the BC applies
│       └── values              : array                  # optional; prescribed values (= boundary forcing)
│
├── normalization/
│   └── <field_name>/  @mean, @std                       # per component or one scalar; ship stats, don't recompute
│                      @min, @max                        # optional (schema >= 0.3.0); for min-max scaling
│
├── t0_fields/  <field datasets>     # rank-0 quantities (state/forcing/coefficient/parameter)
├── t1_fields/  <field datasets>     # rank-1
└── t2_fields/  <field datasets>     # rank-2; the three groups always exist, empty or not
```

Each field dataset is named after its field and carries the **extent flags** as attributes
(`@dim_varying`, `@time_varying`, `@sample_varying`) plus, at rank 2, `@symmetric` and
`@antisymmetric`. Its tensor order, role and unit are *not* repeated on the dataset: they
come from the registry entry of its name. The `provenance` attribute is a flat JSON object
of string values (e.g. the generator's resolved config as a JSON string, the git revision,
the creation time); the schema writes and reads it verbatim and never interprets it.

The reader (`read_dataset`) loads every field whole, reads a missing `grid_type` as
`"cartesian"` and a missing `provenance` as empty, and validates the result.

---

## 5. Grid profile

Spatial backend = separable, logically-Cartesian axes.

```
├── @grid_type             : "cartesian" | "spherical"   (root attribute)
├── dimensions/
│   ├── x                  : (X,) float          # 1D coordinate vector per axis
│   ├── y                  : (Y,) float
│   └── (z)                : (Z,) float
```

Field arrays carry explicit spatial axes (shapes per §1): scalar state →
`(n_traj, n_steps, X, Y)`, vector state → `(…, X, Y, D)`. Boundary masks are grid-shaped.
This is the profile The Well's loaders understand; a grid file in this format is readable by
their tooling.

---

## 6. Mesh profile

Spatial backend = unstructured topology. The spatial axes collapse to a single DOF axis, and
geometry and topology are stored explicitly.

```
├── geometry/
│   ├── vertices           : (n_traj?, N_vert, n_spatial_dims) float
│   ├── cells              : (N_cell, nodes_per_cell) int        # connectivity
│   ├── @cell_type         : "triangle" | "tetra" | "quad" | ...
│   └── dof_map/
│       ├── @function_space : "P1" | "P2" | "DG0" | ...           # where DOFs live
│       └── coords          : (N_dof, n_spatial_dims) float       # DOF coordinates
```

Field arrays are flat over DOFs: scalar state → `(n_traj, n_steps, N_dof)`, vector state →
`(…, N_dof, D)`. `vertices` may be `sample_varying` (a mesh per trajectory) or shared (one
mesh, trajectory axis dropped).

**DOFs vs vertices.** For P1, DOFs coincide with vertices and `coords` mirrors `vertices`.
For P2+ or DG, DOFs are *not* the vertices, so `dof_map` carries the authoritative DOF
coordinates: the points a point/token model consumes. Grid and mesh files are never mixed in
a batch, so a mesh loader never reconciles with the grid one.

**Status.** Specified, not implemented: the package validates and writes
`representation="mesh"` (and `"particle"`) like any other file but has no `geometry/` group
yet. A mesh producer today stores the DOF axis as the single spatial axis of a 1D grid
profile.

---

## 7. Worked examples

### A. Poisson, steady, complex geometry (mesh profile)

`−∇·(κ∇u) = f` on an irregular domain.

```
representation = "mesh",  n_spatial_dims = 2
present_fields = ["u", "source", "diffusivity"]
geometry/        : vertices, cells (triangle), dof_map (P2)
t0_fields/
  u           : (n_traj, N_dof)        role=state,       time_varying=False
  source      : (n_traj, N_dof)        role=forcing,     time_varying=False
  diffusivity : (n_traj, N_dof)        role=coefficient, time_varying=False
boundary_conditions/
  outer_wall  : bc_type=dirichlet, values=(…), mask=(…)
```

Steady ⇒ no time axis; `u` is the prediction target, `source` + `diffusivity` are
conditioning inputs (by role).

### B. Forced incompressible Navier–Stokes, time-dependent (grid profile)

`∂u/∂t + (u·∇)u = −∇p + ν∇²u + f`, periodic box, global `Re`.

```
representation = "grid",  grid_type = "cartesian",  n_spatial_dims = 2
present_fields = ["velocity", "pressure", "body_force", "reynolds"]
dimensions/      : x (X,), y (Y,), time (n_steps,)
t0_fields/
  pressure   : (n_traj, n_steps, X, Y)        role=state
  reynolds   : (n_traj,)                       role=parameter,  dim_varying=False, time_varying=False
t1_fields/
  velocity   : (n_traj, n_steps, X, Y, 2)      role=state
  body_force : (n_traj, n_steps, X, Y, 2)      role=forcing
boundary_conditions/
  all        : bc_type=periodic
```

`velocity` + `pressure` are targets; `body_force` + `reynolds` are conditioning inputs.
`reynolds` is stored as one number per trajectory and broadcast: the "scalar parameter"
case, with no special group.

### C. Controlled ODE, no spatial domain (grid profile, `n_spatial_dims = 0`)

`du/dt = a·c(t) + b` on `t ∈ [0, T]`. The operator maps the control `c(·)` (+ the initial
value `u(0)`) to the solution `u(·)`. There is no spatial coordinate — the only independent
variable, `t`, is the time axis — so `n_spatial_dims = 0` and `spatial_dims = []`.

```
representation = "grid",  n_spatial_dims = 0,  spatial_dims = []
present_fields = ["u", "control", "coeff_a", "coeff_b"]
dimensions/      : time (L,)
t0_fields/
  u        : (n_traj, L)     role=state,     dim_varying=False, time_varying=True
  control  : (n_traj, L)     role=forcing,   dim_varying=False, time_varying=True
  coeff_a  : (n_traj,)       role=parameter, dim_varying=False, time_varying=False
  coeff_b  : (n_traj,)       role=parameter, dim_varying=False, time_varying=False
```

`u` is the prediction target; `control` + `coeff_a` / `coeff_b` are conditioning inputs (by
role). The initial condition `u(0)` is `u[:, 0]` (§3: the IC is the `t=0` slice of the
state), so it is not stored separately. With zero spatial dims, **every** field is
`dim_varying=False` (there is no spatial axis to vary over); the validator enforces this. A
spatial PDE is the same encoding with `dim_varying=True` and a non-empty `spatial_dims`; the
ODE is its zero-space limit.

### D. Advection-diffusion, 1+1D (grid profile)

`examples/advection_diffusion.py` in this repository: `u` `(n_traj, n_steps, X)` as the
state, `advection` and `diffusion` `(n_traj,)` as parameters, all in `t0_fields`, a periodic
boundary condition on `u` along `x`, and z-score statistics per field.

---

## 8. Contracts

### Writer

`write_dataset(dataset, path)` validates the dataset and writes it in one call. A dataset
whose bulk fields exceed memory goes through the incremental `DatasetWriter`
(schema ≥ 0.3.0): the metadata and the small fields are written on entering the context,
each streamed field's dataset is created at full shape (chunked by trajectory, and by
timestep when the field has a time axis) and filled trajectory by trajectory through
`append`; `update_normalization` sets the statistics that are only known after the pass;
leaving the context verifies completeness and writes the `normalization` group. On an
exception inside the block, or a missing trajectory, the file is deleted: a partial write
never leaves a valid-looking file. `write_dataset` is the writer with no streamed fields.

Storage filters are a `Compression` spec (filter name, gzip level, byte-shuffle) applied to
every field and boundary-condition dataset. For smooth float fields,
`Compression("gzip", level=1, shuffle=True)` is the recommended default: smaller files and
faster reads and writes than plain gzip.

### Loader (consumer)

A shared base class handles everything profile-independent; each profile subclass overrides
only the spatial decoding.

- **Shared base:** trajectory windowing (`T_in → T_out`), normalization (apply the shipped
  statistics), role-based routing (`state` → targets; everything else → conditioning),
  broadcasting of `dim_varying=False` / `time_varying=False` fields, presence masking, and
  the HDF5 IO mechanics (lazy per-worker file open, trajectory-aligned chunking).
- **Grid subclass:** decodes `(…, X, Y[, Z])` arrays; coordinates from the axis vectors.
- **Mesh subclass:** decodes `(…, N_dof[, D])` arrays; attaches `geometry` / `dof_map`.

The `sciml-template` repository's `ml/` workspace implements the steady grid case: whole-field
`forcing → state` samples, spatially uniform `parameter` / `coefficient` fields as a flat
conditioning vector, every split standardised with the training file's statistics.

**Within-representation multi-physics** uses a scenario-interleaved sampler above the
loaders: each batch is drawn from one dataset (homogeneous spatial shape, trivial collate),
datasets interleaved across batches. The field registry keeps shared fields
identity-consistent so the model can share parameters; the presence list + a channel mask
let it ingest variable field sets.

---

## 9. Open points

- Whether `vertices` is per-trajectory or shared (affects mesh storage cost); the mesh
  profile's `geometry/` group is not implemented yet (§6).
- Conditioning interface: a loader broadcasts `parameter` fields into channels or passes them
  as a separate conditioning vector. Storage is identical; the template's loader passes a
  flat vector.
- Chunking of non-streamed fields (h5py's default today); streamed fields chunk by
  trajectory and timestep.

Settled: the registry is versioned separately from the layout (`registry_version`, §10);
range statistics are optional attributes next to mean and std; large datasets stream through
`DatasetWriter`.

---

## 10. Versioning

Three independent version numbers:

| Number | Where | Changes when |
|---|---|---|
| `schema_version` (`SCHEMA_VERSION`, `0.3.0`) | root attribute | the layout changes: a group, an attribute or a shape rule |
| `registry_version` (`REGISTRY_VERSION`, `0.1.0`) | root attribute | the canonical vocabulary changes |
| the package version (`__version__`, `pyproject.toml`) | the distribution | any release of the code |

Schema `0.3.0` added the optional `min` / `max` normalization attributes and the incremental
writer contract. A reader of `0.3.0` reads earlier files: the attributes it added are
optional.
