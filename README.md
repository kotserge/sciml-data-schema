# sciml-data-schema

An HDF5 dataset format and metadata schema for PDE / physics simulation data, designed for scientific machine learning, based on the work of [The Well](https://github.com/PolymathicAI/the_well).
A single file holds one dataset of `n_trajectories` independent runs over a shared spatial domain.
The full design is in [`SCHEMA.md`](SCHEMA.md).

- `sciml_data_schema.schema` — the in-memory dataset representation (`Dataset`, `Field`, `FieldSpec`, `Dimension`, `BoundaryCondition`, `Normalization`) and the canonical HDF5 group/attribute names.
- `sciml_data_schema.io` — `read_dataset` / `write_dataset` (the only module that touches HDF5).

## The unified field model

Every physical quantity — solution, forcing, coefficient, parameter — is a `Field`.
There is no separate "scalars" group: a "scalar parameter" is just a rank-0 field that doesn't vary in space or time.
A field carries two orthogonal pieces of metadata:

- **tensor order** (0/1/2) — selects the `t{order}_fields` group and fixes the trailing tensor axes; together with **role** (`state` / `forcing` / `coefficient` / `parameter` / `boundary`) and **unit** it forms the field's registry identity (`FieldSpec`).
- **extent flags** (`dim_varying` / `time_varying` / `sample_varying`) — an axis along which a field does not vary is dropped, so the stored shape is `(n_trajectories?, n_timesteps?, *spatial_shape?, *(n_spatial_dims,) * tensor_order)` with each leading axis present only if its flag is set.

A file also ships a `field_registry` (the canonical vocabulary, shared by convention across files of a representation), the `present_fields` it actually contains, and per-field `normalization` stats.
The grid profile stores spatial coordinate vectors under `dimensions`.
Train/val/test splits are left to downstream consumers rather than baked into the stored format.

## Installation

```bash
pip install sciml-data-schema
```

Or, from a checkout (uses [uv](https://docs.astral.sh/uv/)):

```bash
uv sync --extra dev
```

Requires Python 3.11+.
Runtime dependencies are `h5py` and `numpy`.

## Usage

A dataset of 4 trajectories of a 2D scalar field `u(t, x, y)`, conditioned on one Reynolds number per trajectory:

```python
import numpy as np
import sciml_data_schema as sds

n_traj, n_steps, nx, ny = 4, 10, 32, 32
rng = np.random.default_rng(0)
u = rng.standard_normal((n_traj, n_steps, nx, ny)).astype(np.float32)
reynolds = np.array([100.0, 200.0, 400.0, 800.0], dtype=np.float32)

dataset = sds.Dataset(
    dataset_name="flow_2d",
    representation="grid",
    n_spatial_dims=2,
    n_trajectories=n_traj,
    spatial_dims=["x", "y"],
    dimensions=[
        sds.Dimension("x", np.linspace(0, 1, nx, dtype=np.float32)),
        sds.Dimension("y", np.linspace(0, 1, ny, dtype=np.float32)),
        sds.Dimension("time", np.linspace(0, 1, n_steps, dtype=np.float32)),
    ],
    fields=[
        # State: shape (n_traj, n_steps, nx, ny)
        sds.Field("u", u, role="state", unit="m/s"),
        # Parameter: one value per trajectory, shape (n_traj,)
        sds.Field("reynolds", reynolds, role="parameter", dim_varying=False, time_varying=False),
    ],
    boundary_conditions=[
        sds.BoundaryCondition("periodic", bc_type="periodic", associated_dims=["x", "y"]),
    ],
    normalization={"u": sds.Normalization(mean=np.asarray(u.mean()), std=np.asarray(u.std()))},
    provenance={"generator": "readme-example"},
)

sds.write_dataset(dataset, "flow_2d.h5", compression=sds.Compression("gzip", level=1, shuffle=True))

# Route fields by role: state fields are targets, everything else conditions
loaded = sds.read_dataset("flow_2d.h5")
targets = [f for f in loaded.fields if f.role == "state"]
conditioning = [f for f in loaded.fields if f.role != "state"]
dx = loaded.coords["x"][1] - loaded.coords["x"][0]
```

`write_dataset` and `read_dataset` validate the dataset; a malformed dataset raises `ValueError` and leaves no file behind.

When the bulk data does not fit in memory, stream it one trajectory at a time.
The streamed field carries a `field_placeholder` of its full shape, and statistics computed during the pass are set before the file closes:

```python
u_field = next(f for f in dataset.fields if f.name == "u")
u_field.data = sds.field_placeholder((n_traj, n_steps, nx, ny), np.float32)

with sds.DatasetWriter(dataset, "flow_2d_streamed.h5", streamed=["u"]) as writer:
    for i in range(n_traj):
        writer.append("u", i, u[i])  # e.g. the output of one simulation run
    stats = sds.Normalization(mean=np.asarray(u.mean()), std=np.asarray(u.std()))
    writer.update_normalization("u", stats)
```

If the block raises, or exits before every trajectory of every streamed field has been appended, the writer deletes the file; the error propagates or a `ValueError` is raised.
See [`examples/advection_diffusion.py`](examples/advection_diffusion.py) for a complete, runnable scenario (build → write → read → role-routed window).

## Development

```bash
uv sync --extra dev
uv run pytest               # tests
uv run ruff check           # lint
uv run ruff format          # format
uv run basedpyright         # type-check
```
