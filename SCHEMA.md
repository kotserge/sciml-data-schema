# PDE Data Schema — Draft

A HDF5 schema for storing PDE solution data for ML, derived from The Well's
conventions but restructured around two principles it lacks:

1. **One core, several profiles.** A single shared *core convention* governs
   metadata, fields, normalization, boundary conditions, and provenance. Each
   *spatial representation* (grid, mesh, …) is a **profile** that differs from
   the others *only* in how the spatial domain and field arrays are encoded.
2. **A unified field model.** Every physical quantity — solution, forcing,
   coefficient, parameter — is a *field*. There is no separate "scalars" group.
   A quantity carries two orthogonal pieces of metadata: its **tensor order**
   (t0/t1/t2) and its **extent flags** (`dim_varying` / `time_varying` /
   `sample_varying`). A "scalar parameter" is just a rank-0 field that doesn't
   vary in space or time.

Scope ceilings (inherited, deliberate): spatial dimensions ≤ 3; tensor rank ≤ 2.
Cross-representation training is **not** a goal — grid and mesh files are never
mixed in one batch — so the profiles need only agree on the core, not on spatial
layout.

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

**Why this dissolves the scalar-vs-field jump.** The same physical quantity
keeps one home regardless of how much it varies. Refining a constant `κ` into a
heterogeneous `κ(x)` is a single flag flip (`dim_varying: False → True`), not a
move between groups — so the schema is stable under refinement.

**Storage stays cheap.** A field with `dim_varying=False` is stored *without*
spatial axes (one value, broadcast on read); a field with `time_varying=False`
drops the time axis; one with `sample_varying=False` drops the trajectory axis.
Nothing materializes a redundant array of identical numbers.

Fields are still **grouped by tensor order** (`t0_fields`, `t1_fields`,
`t2_fields`) because tensor order is load-bearing (it governs how a quantity
transforms). Tensor order is the *only* distinction kept from The Well's group
structure; the scalars group is gone.

### Resulting array shapes

For a 2D grid profile, a `t0` (scalar) field, depending on its flags:

| dim_varying | time_varying | sample_varying | Stored shape              | Example                        |
|-------------|--------------|----------------|---------------------------|--------------------------------|
| yes         | yes          | yes            | `(n_traj, n_steps, X, Y)` | solution `u(x,y,t)`            |
| yes         | no           | yes            | `(n_traj, X, Y)`          | static coefficient `κ(x,y)`    |
| no          | no           | yes            | `(n_traj,)`               | global parameter (e.g. `Re`)   |
| no          | no           | no             | `(1,)`                    | a true global constant         |

A `t1` (vector) field appends a components axis `D`; a `t2` field appends
`D, D`. The mesh profile replaces the spatial axes `(X, Y[, Z])` with a single
DOF axis `(N_dof)` — see §6.

---

## 2. The field registry (canonical vocabulary)

A per-file (but conventionally identical across all files of a representation)
registry that fixes the *identity* of every quantity. This is the linchpin for
**within-representation multi-physics**: it guarantees that `velocity` in a
Rayleigh–Bénard file and `velocity` in a Euler file are the *same* registered
field, so a model can share parameters across them.

Stored as a root JSON attribute `field_registry`:

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

Individual files declare *which* registry entries are present (a
field-presence list), so a 1-field Poisson file and a 5-field Euler file share
the registry without storing absent fields.

---

## 3. How PDE concepts map to the schema

A generic PDE,

```
∂u/∂t = N(u; θ) + f      on Ω,    + boundary conditions,    + u(t=0) = u₀
```

maps as follows. **Role** is the bridge between PDE meaning and storage, and
it's what the loader uses to split conditioning inputs from prediction targets.

| PDE concept | Symbol | Representation | role | typical flags |
|-------------|--------|----------------|------|----------------|
| **Solution / state** | `u` | field, tensor order by physical type (scalar→t0, velocity→t1, stress→t2) | `state` | `time_varying=True` (steady → `False`), `sample_varying=True` |
| **Initial condition** | `u₀` | *not a separate object* — it is the `t=0` slice of the state field | — | — |
| **IC-generation knobs** | seed, amplitude | rank-0 field | `parameter` | `dim_varying=False`, `time_varying=False`, `sample_varying=True` |
| **Volumetric forcing / source** | `f(x,t)` | field, tensor order by type | `forcing` | flags per its actual variation |
| **Boundary forcing** | inflow/flux profile | `boundary_conditions/<bc>/values` | `boundary` | (lives in BC group, see §4) |
| **Global parameter** | `Re`, `Ra`, `Pr` | rank-0 field, broadcast | `parameter` | `dim_varying=False`, `time_varying=False`, `sample_varying=True` |
| **Spatially-varying coefficient** | `κ(x)`, anisotropic `K(x)` | t0/t1/t2 field | `coefficient` | `dim_varying=True`, usually `time_varying=False` |
| **Domain geometry** | `Ω` | profile-specific (axes vs mesh) | — | see §5 / §6 |
| **Time axis** | `t` | `dimensions/time` | — | constant `dt` or explicit vector |
| **Boundary conditions** | — | `boundary_conditions` group | — | type + mask + values (§4) |

**The role tag is the only addition beyond The Well**, and it earns its place
here: The Well folds `state`, `forcing`, and `coefficient` into the same groups
and its loader windows everything as state. By tagging role, your loader routes
`state` → prediction targets and `forcing`/`coefficient`/`parameter` →
conditioning inputs, with no schema surgery.

**Steady-state problems** (e.g. Poisson) are handled by `time_varying=False` on
the state field and `n_steps = 1` (or omit the time axis entirely). The same
schema covers steady → time-dependent with one flag.

---

## 4. Shared core layout

Common to every profile.

```
/                                   (root)
├── @schema_version        : "0.1"
├── @dataset_name          : str
├── @representation        : "grid" | "mesh" | "particle"
├── @n_spatial_dims        : 1 | 2 | 3
├── @n_trajectories        : int
├── @field_registry        : JSON (see §2)
├── @present_fields        : list[str]         # which registry entries are here
│   # NB: train/val/test splits are a downstream-consumer concern, not stored here.
│
├── dimensions/
│   ├── time               : (n_steps,) float   # or @dt scalar if uniform
│   └── @spatial_dims      : ["x","y"]           # axis names (semantics; coords in profile)
│
├── boundary_conditions/
│   └── <bc_name>/
│       ├── @bc_type          : "periodic" | "dirichlet" | "neumann" | "wall" | "open"
│       ├── @associated_fields : [str]
│       ├── @associated_dims    : [str]
│       ├── mask              : bool array        # where the BC applies
│       └── values            : array             # prescribed values (= boundary forcing)
│
├── normalization/
│   └── <field_name>/  @mean, @std (per component)   # ship stats, don't recompute
│
├── t0_fields/  <field datasets>     # rank-0 quantities (state/forcing/coeff/param)
├── t1_fields/  <field datasets>     # rank-1
└── t2_fields/  <field datasets>     # rank-2
```

Each field dataset carries the §1 attributes (`name`, `role`, `unit`,
`dim_varying`, `time_varying`, `sample_varying`, …). A validator checks every
file against this core plus its profile.

---

## 5. Grid profile

Spatial backend = separable, logically-Cartesian axes.

```
├── @grid_type             : "cartesian" | "spherical"
├── dimensions/
│   ├── x                  : (X,) float          # 1D coordinate vector per axis
│   ├── y                  : (Y,) float
│   └── (z)                : (Z,) float
```

Field arrays carry explicit spatial axes (shapes per §1): scalar state →
`(n_traj, n_steps, X, Y)`, vector state → `(…, X, Y, D)`. Boundary masks are
grid-shaped. This is the profile The Well's loaders already understand; conform
to it and you inherit their tooling for free on grid data.

---

## 6. Mesh profile

Spatial backend = unstructured topology. The spatial axes collapse to a single
DOF axis, and geometry/topology are stored explicitly.

```
├── geometry/
│   ├── vertices           : (n_traj?, N_vert, n_spatial_dims) float
│   ├── cells              : (N_cell, nodes_per_cell) int        # connectivity
│   ├── @cell_type         : "triangle" | "tetra" | "quad" | ...
│   └── dof_map/
│       ├── @function_space : "P1" | "P2" | "DG0" | ...           # where DOFs live
│       └── coords          : (N_dof, n_spatial_dims) float       # DOF coordinates
```

Field arrays are flat over DOFs: scalar state → `(n_traj, n_steps, N_dof)`,
vector state → `(…, N_dof, D)`. `vertices` may be `sample_varying` (per-trajectory
mesh) or shared (one mesh, drop the trajectory axis).

**Note on DOFs vs vertices.** For P1, DOFs coincide with vertices and `coords`
mirrors `vertices`. For P2+ or DG, DOFs are *not* the vertices, so `dof_map`
carries the authoritative DOF coordinates — this is what a point/token model
would consume. (You confirmed grid+mesh aren't mixed, so this profile's loader
never has to reconcile with the grid one.)

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

Steady => no time axis; `u` is the prediction target, `source` + `diffusivity`
are conditioning inputs (by role).

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

`velocity` + `pressure` are targets; `body_force` + `reynolds` are conditioning
inputs. `reynolds` is stored as one number per trajectory and broadcast — the
"scalar parameter" case, with no special group.

---

## 8. Loader contract (per profile)

A shared base class handles everything profile-independent; each profile
subclass overrides only spatial decoding.

- **Shared base:** trajectory windowing (`T_in → T_out`), normalization (apply
  shipped stats), role-based routing (`state` → targets; everything else →
  conditioning), broadcasting of `dim_varying=False` / `time_varying=False`
  fields, presence-masking, and the HDF5 IO mechanics (lazy per-worker file
  open, trajectory-aligned chunking).
- **Grid subclass:** decodes `(…, X, Y[, Z])` arrays; coordinates from axis vectors.
- **Mesh subclass:** decodes `(…, N_dof[, D])` arrays; attaches `geometry`/`dof_map`.

**Within-representation multi-physics** uses a scenario-interleaved sampler
above the loaders: each batch is drawn from one dataset (homogeneous spatial
shape, trivial collate), datasets interleaved across batches. The field registry
keeps shared fields identity-consistent so the model can share parameters; the
presence list + a channel mask let it ingest variable field sets.

---

## 9. Open points to settle next

- Whether `vertices` is per-trajectory or shared (affects mesh storage cost).
- Conditioning interface: do you broadcast `parameter` fields into channels, or
  pass them as a separate conditioning vector? (Storage is identical; this is a
  loader choice.)
- Whether to version the `field_registry` separately from `schema_version`.
- Compression/chunking defaults per profile (grid: chunk by trajectory; mesh:
  chunk by trajectory × DOF blocks).
