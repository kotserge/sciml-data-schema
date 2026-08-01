# sciml-data-schema

An HDF5 dataset format and metadata schema for PDE / physics simulation data,
designed for scientific machine learning, based on the work of [The Well](https://github.com/PolymathicAI/the_well). 
A single file holds one dataset of`n_trajectories` independent runs over a shared 
spatial domain. The full design is in [`SCHEMA.md`](SCHEMA.md).

- `sciml_data_schema.schema` — the in-memory dataset representation (`Dataset`,
  `Field`, `FieldSpec`, `Dimension`, `BoundaryCondition`, `Normalization`) and
  the canonical HDF5 group/attribute names.
- `sciml_data_schema.io` — `read_dataset` / `write_dataset` (the only module
  that touches HDF5).

## The unified field model

Every physical quantity — solution, forcing, coefficient, parameter — is a
`Field`. There is no separate "scalars" group: a "scalar parameter" is just a
rank-0 field that doesn't vary in space or time. A field carries two orthogonal
pieces of metadata:

- **tensor order** (0/1/2) — selects the `t{order}_fields` group and fixes the
  trailing tensor axes; together with **role** (`state` / `forcing` /
  `coefficient` / `parameter` / `boundary`) and **unit** it forms the field's
  registry identity (`FieldSpec`).
- **extent flags** (`dim_varying` / `time_varying` / `sample_varying`) — an axis
  along which a field does not vary is dropped, so the stored shape is
  `(n_trajectories?, n_timesteps?, *spatial_shape?, *(n_spatial_dims,) * tensor_order)`
  with each leading axis present only if its flag is set.

A file also ships a `field_registry` (the canonical vocabulary, shared by
convention across files of a representation), the `present_fields` it actually
contains, and per-field `normalization` stats. The grid profile stores spatial
coordinate vectors under `dimensions`. Train/val/test splits are left to
downstream consumers rather than baked into the stored format.

## Installation

```bash
pip install sciml-data-schema
```

Or, from a checkout (uses [uv](https://docs.astral.sh/uv/)):

```bash
uv sync --extra dev
```

Requires Python 3.11+. Runtime dependencies are `h5py` and `numpy`.

## Usage

```python
import numpy as np
import sciml_data_schema as sds

dataset = sds.Dataset(
    dataset_name="poisson_2d",
    representation="grid",
    n_spatial_dims=2,
    n_trajectories=1,
    spatial_dims=["x", "y"],
    dimensions=[
        sds.Dimension("x", np.linspace(0, 1, 32)),
        sds.Dimension("y", np.linspace(0, 1, 32)),
    ],
    fields=[
        sds.Field("u", np.zeros((1, 32, 32)), role="state", time_varying=False),
    ],
)

sds.write_dataset(dataset, "run.h5", compression="gzip")

ds = sds.read_dataset("run.h5")
u = next(f for f in ds.fields if f.role == "state")
```

`write_dataset` validates the dataset before touching disk, so a malformed
dataset raises `ValueError` and leaves no file behind. See
[`examples/advection_diffusion.py`](examples/advection_diffusion.py) for a
complete, runnable scenario (build → write → read → role-routed window).

## Development

```bash
uv sync --extra dev
uv run pytest        # tests
uv run ruff check    # lint
uv run ruff format   # format
uv run basedpyright  # type-check
```
