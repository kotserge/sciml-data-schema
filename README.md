# sciml-data-schema

An HDF5 dataset format and metadata schema for PDE and physics simulation data in
scientific machine learning, after the data format of
[The Well](https://github.com/PolymathicAI/the_well). One file holds one dataset of
`n_trajectories` independent runs over a shared spatial domain. Every physical quantity
(solution, forcing, coefficient, parameter) is a `Field` with a tensor order, a role and a
unit, grouped on disk by tensor order; three extent flags say which of the trajectory, time
and spatial axes its array carries, and an axis a field does not vary along is not stored.
A field registry fixes the identity of every name across the files of a representation;
each file lists the fields it holds and ships per-field normalization statistics and an
optional provenance stamp.

![Overview: a producer builds a Dataset and writes it with write_dataset, or streams it with DatasetWriter; the file holds the root attributes, the dimensions, the boundary conditions, the normalization statistics and the t0/t1/t2 field groups; a consumer reads it with read_dataset and routes the fields by role; a field is an identity shared across files plus an extent per file](docs/figures/overview.png)

The package is the interface between a data-generation workspace and a training workspace
(the layout of `sciml-template`, which installs it as an editable sibling): a producer
builds a `Dataset` and writes it, a consumer reads it and routes the fields by role.
`io.py` is the only module that touches HDF5. The format is specified in
[`docs/SCHEMA.md`](docs/SCHEMA.md); `docs/figures/overview.py` redraws the figure.

## Installation

```bash
pip install sciml-data-schema
```

Or, from a checkout, with [uv](https://docs.astral.sh/uv/) and
[just](https://github.com/casey/just) (every recipe is a thin wrapper over `uv run`):

```bash
just sync            # uv sync --extra dev
just test            # pytest, offline
just check           # ruff check, ruff format --check, basedpyright
```

Requires Python 3.11+; the runtime dependencies are `h5py` and `numpy`. A project that
consumes the format from a checkout installs it as an editable path dependency
(`sciml-data-schema = { path = "../sciml-data-schema", editable = true }` under
`[tool.uv.sources]`).

## Usage

Build a `Dataset`, write it, read it back:

```python
import numpy as np
import sciml_data_schema as sds

rng = np.random.default_rng(0)
n_trajectories, nx, ny = 16, 64, 64
source = rng.standard_normal((n_trajectories, nx, ny)).astype(np.float32)
u = rng.standard_normal((n_trajectories, nx, ny)).astype(np.float32)
diffusivity = rng.uniform(0.5, 2.0, n_trajectories).astype(np.float32)

dataset = sds.Dataset(
    dataset_name="poisson",
    representation="grid",
    n_spatial_dims=2,
    n_trajectories=n_trajectories,
    spatial_dims=["x", "y"],
    dimensions=[
        sds.Dimension("x", np.linspace(0, 1, nx, dtype=np.float32)),
        sds.Dimension("y", np.linspace(0, 1, ny, dtype=np.float32)),
    ],
    fields=[
        sds.Field("source", source, role="forcing", time_varying=False),   # (16, 64, 64)
        sds.Field("u", u, role="state", time_varying=False),               # (16, 64, 64)
        sds.Field(                                                         # (16,)
            "diffusivity", diffusivity, role="parameter", unit="m^2/s",
            dim_varying=False, time_varying=False,
        ),
    ],
    boundary_conditions=[
        sds.BoundaryCondition(
            "walls", "dirichlet", associated_fields=["u"], associated_dims=["x", "y"]
        ),
    ],
    normalization={"u": sds.Normalization(mean=np.asarray(u.mean()), std=np.asarray(u.std()))},
    provenance={"git_revision": sds.git_revision(".")},
)

sds.write_dataset(dataset, "poisson.h5", compression=sds.Compression("gzip", level=1, shuffle=True))

loaded = sds.read_dataset("poisson.h5")
targets = [f for f in loaded.fields if f.role == "state"]
conditioning = [f for f in loaded.fields if f.role != "state"]
dx = loaded.coords["x"][1] - loaded.coords["x"][0]
```

`write_dataset` validates before touching the disk: a malformed dataset raises `ValueError`
and leaves no file. `read_dataset` loads every field into memory and validates the result.

A dataset whose bulk fields do not fit in memory streams through `DatasetWriter`: give each
streamed field `field_placeholder` data of its full shape, fill it one trajectory at a time,
and set the statistics that are only known after the pass before the context closes:

```python
u = sds.Field("u", sds.field_placeholder((n_trajectories, nx, ny)), role="state", time_varying=False)
dataset = sds.Dataset(..., fields=[u, ...])

with sds.DatasetWriter(dataset, "poisson.h5", streamed=("u",)) as writer:
    for trajectory in range(n_trajectories):
        writer.append("u", trajectory, solve(trajectory))      # (nx, ny), float32
    writer.update_normalization("u", sds.Normalization(mean=mean, std=std))
```

Leaving the context verifies that every trajectory was written; on an exception, or a
missing trajectory, the file is deleted.
[`examples/advection_diffusion.py`](examples/advection_diffusion.py) is a complete scenario
(build → write → read → role-routed window); `just example` runs it.

### The public API

| Name | What it is |
|------|------------|
| `Dataset` | one dataset: the root metadata, `dimensions`, `fields`, `boundary_conditions`, `normalization`, `field_registry`, `provenance`; `validate()`, `coords`, `present_fields` |
| `Field` | a quantity: `name`, `data`, `tensor_order`, `role`, `unit`, the `dim_varying` / `time_varying` / `sample_varying` flags, `symmetric` / `antisymmetric` |
| `FieldSpec` | a registry entry: `tensor_order`, `role`, `unit` |
| `Dimension` | a coordinate axis: `name`, `coordinates` |
| `BoundaryCondition` | `name`, `bc_type`, `associated_fields`, `associated_dims`, optional `mask` and `values` |
| `Normalization` | per-field statistics: `mean`, `std`, optional `min`, `max` |
| `Role`, `Representation`, `GridType`, `BCType` | the controlled vocabularies (`Literal` types) |
| `write_dataset(dataset, path, *, mode, compression)` | validate and write in one call; returns the `Path` |
| `DatasetWriter(dataset, path, *, streamed, mode, compression)` | the incremental writer: `append`, `update_normalization` |
| `field_placeholder(shape, dtype)` | a zero-copy stand-in for a streamed field's data |
| `read_dataset(path)` | load and validate a file |
| `Compression(name, level, shuffle)` | the HDF5 filters; a bare filter name such as `"gzip"` is accepted too |
| `git_revision(anchor)` | the short SHA of the checkout containing `anchor`, for the provenance stamp |
| `SCHEMA_VERSION`, `REGISTRY_VERSION` | the layout and vocabulary versions a new `Dataset` carries |

## Structure

```
src/sciml_data_schema/
  schema.py        # Dataset, Field, FieldSpec, Dimension, BoundaryCondition, Normalization; the HDF5 names; validate()
  io.py            # write_dataset, DatasetWriter, field_placeholder, read_dataset, Compression (the only h5py importer)
  provenance.py    # git_revision
  __init__.py      # the public API, re-exported from the package root
tests/             # conftest.py builds a fixture with every tensor order and extent; round trips, validation, the writer, the example
examples/          # advection_diffusion.py: build -> write -> read -> role-routed window
docs/
  SCHEMA.md        # the format: field model, registry, layout, profiles, contracts, versioning
  figures/         # overview.py draws overview.png for this README
  refactor/        # the planning trail (docs/<category>/<N>-<slug>/)
justfile           # sync, test, check, format, figure, example
```

`CLAUDE.md` holds the conventions and the gates. The code is MIT-licensed (`LICENSE`).
