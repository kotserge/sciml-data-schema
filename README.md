# core

Shared dataset format and metadata schema — the neutral middle that `datagen`
and `ml` both depend on. Data is exchanged exclusively as HDF5 files in this
format. The full design is in [`SCHEMA.md`](SCHEMA.md).

- `core.schema` — in-memory dataset representation (`Dataset`, `Field`,
  `FieldSpec`, `Dimension`, `BoundaryCondition`, `Normalization`) and the
  canonical HDF5 group/attribute names.
- `core.io` — `read_dataset` / `write_dataset` (the only module that touches HDF5).

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

```python
import core

ds = core.read_dataset("run.h5")
u = next(f for f in ds.fields if f.role == "state")
core.write_dataset(ds, "copy.h5", compression="gzip")
```
