"""HDF5 read/write for :class:`sciml_data_schema.schema.Dataset`.

This is the only module that imports ``h5py``. Everything else works with the
plain-Python objects from :mod:`sciml_data_schema.schema`, so the on-disk layout
can change here without rippling through the codebase.

The file mirrors the schema's shared core layout: root attributes (versions,
``representation``, the ``field_registry`` JSON, and the ``present_fields``
list), a ``dimensions`` group (coordinate vectors + the ``spatial_dims``
attribute), ``boundary_conditions``, ``normalization``, and the
``t{0,1,2}_fields`` groups. Each field dataset carries its extent flags as
attributes; its tensor order, role, and unit come from the registry.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from os import PathLike
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from . import schema as s

__all__ = ["write_dataset", "read_dataset"]

StrPath = str | PathLike[str]

# Variable-length UTF-8 string dtype for HDF5 string-array attributes.
_STR = h5py.string_dtype("utf-8")


def write_dataset(
    dataset: s.Dataset,
    path: StrPath,
    *,
    mode: str = "w",
    compression: str | None = None,
) -> Path:
    """Validate ``dataset`` and write it to ``path`` as a single HDF5 file.

    ``compression`` is forwarded to every field dataset (e.g. ``"gzip"``);
    ``None`` writes uncompressed.
    """
    dataset.validate()
    registry = dataset.effective_registry()
    path = Path(path)
    with h5py.File(path, mode) as f:
        f.attrs[s.ATTR_SCHEMA_VERSION] = dataset.schema_version
        f.attrs[s.ATTR_REGISTRY_VERSION] = dataset.registry_version
        f.attrs[s.ATTR_DATASET_NAME] = dataset.dataset_name
        f.attrs[s.ATTR_REPRESENTATION] = dataset.representation
        f.attrs[s.ATTR_GRID_TYPE] = dataset.grid_type
        f.attrs[s.ATTR_N_SPATIAL_DIMS] = dataset.n_spatial_dims
        f.attrs[s.ATTR_N_TRAJECTORIES] = dataset.n_trajectories
        f.attrs[s.ATTR_FIELD_REGISTRY] = s.registry_to_json(registry)
        f.attrs[s.ATTR_PRESENT_FIELDS] = _str_array(dataset.present_fields)
        if dataset.provenance:
            f.attrs[s.ATTR_PROVENANCE] = json.dumps(dataset.provenance)

        _write_dimensions(f, dataset.dimensions, dataset.spatial_dims)
        _write_boundary_conditions(f, dataset.boundary_conditions, compression)
        _write_normalization(f, dataset.normalization)
        _write_fields(f, dataset.fields, compression)
    return path


def read_dataset(path: StrPath) -> s.Dataset:
    """Load an HDF5 file written by :func:`write_dataset` back into a :class:`Dataset`."""
    with h5py.File(Path(path), "r") as f:
        registry = s.registry_from_json(_as_str(f.attrs[s.ATTR_FIELD_REGISTRY]))
        present = _as_str_list(f.attrs[s.ATTR_PRESENT_FIELDS])
        dataset = s.Dataset(
            dataset_name=_as_str(f.attrs[s.ATTR_DATASET_NAME]),
            representation=cast(s.Representation, _as_str(f.attrs[s.ATTR_REPRESENTATION])),
            grid_type=cast(s.GridType, _as_str(f.attrs.get(s.ATTR_GRID_TYPE, "cartesian"))),
            n_spatial_dims=_as_int(f.attrs[s.ATTR_N_SPATIAL_DIMS]),
            n_trajectories=_as_int(f.attrs[s.ATTR_N_TRAJECTORIES]),
            spatial_dims=_read_spatial_dims(f),
            schema_version=_as_str(f.attrs[s.ATTR_SCHEMA_VERSION]),
            registry_version=_as_str(f.attrs[s.ATTR_REGISTRY_VERSION]),
            dimensions=_read_dimensions(f),
            fields=_read_fields(f, registry, present),
            boundary_conditions=_read_boundary_conditions(f),
            normalization=_read_normalization(f),
            field_registry=registry,
            provenance=_read_provenance(f),
        )
    dataset.validate()
    return dataset


def _read_provenance(f: h5py.File) -> dict[str, str]:
    """Read the optional free-form provenance object (absent in older files)."""
    if s.ATTR_PROVENANCE not in f.attrs:
        return {}
    return json.loads(_as_str(f.attrs[s.ATTR_PROVENANCE]))


# --- group writers ------------------------------------------------------------


def _write_dimensions(f: h5py.File, dimensions: list[s.Dimension], spatial_dims: list[str]) -> None:
    grp = f.create_group(s.GROUP_DIMENSIONS)
    grp.attrs[s.ATTR_SPATIAL_DIMS] = _str_array(spatial_dims)
    for dim in dimensions:
        grp.create_dataset(dim.name, data=np.asarray(dim.coordinates))


def _write_boundary_conditions(
    f: h5py.File, bcs: list[s.BoundaryCondition], compression: str | None
) -> None:
    grp = f.create_group(s.GROUP_BOUNDARY_CONDITIONS)
    for bc in bcs:
        sub = grp.create_group(bc.name)
        sub.attrs[s.ATTR_BC_TYPE] = bc.bc_type
        sub.attrs[s.ATTR_ASSOCIATED_FIELDS] = _str_array(bc.associated_fields)
        sub.attrs[s.ATTR_ASSOCIATED_DIMS] = _str_array(bc.associated_dims)
        if bc.mask is not None:
            sub.create_dataset("mask", data=np.asarray(bc.mask), compression=compression)
        if bc.values is not None:
            sub.create_dataset("values", data=np.asarray(bc.values), compression=compression)


def _write_normalization(f: h5py.File, normalization: dict[str, s.Normalization]) -> None:
    grp = f.create_group(s.GROUP_NORMALIZATION)
    for name, norm in normalization.items():
        sub = grp.create_group(name)
        sub.attrs[s.ATTR_MEAN] = np.asarray(norm.mean)
        sub.attrs[s.ATTR_STD] = np.asarray(norm.std)
        # Range stats are optional (schema >= 0.3.0); absent attrs read back as None.
        if norm.min is not None:
            sub.attrs[s.ATTR_MIN] = np.asarray(norm.min)
        if norm.max is not None:
            sub.attrs[s.ATTR_MAX] = np.asarray(norm.max)


def _write_fields(f: h5py.File, fields: list[s.Field], compression: str | None) -> None:
    # Create every field group up front so readers can rely on their presence.
    groups = {order: f.create_group(name) for order, name in s.FIELD_GROUPS.items()}
    for fld in fields:
        ds = groups[fld.tensor_order].create_dataset(
            fld.name, data=np.asarray(fld.data), compression=compression
        )
        ds.attrs[s.ATTR_DIM_VARYING] = bool(fld.dim_varying)
        ds.attrs[s.ATTR_TIME_VARYING] = bool(fld.time_varying)
        ds.attrs[s.ATTR_SAMPLE_VARYING] = bool(fld.sample_varying)
        if fld.tensor_order == 2:
            ds.attrs[s.ATTR_SYMMETRIC] = bool(fld.symmetric)
            ds.attrs[s.ATTR_ANTISYMMETRIC] = bool(fld.antisymmetric)


# --- group readers ------------------------------------------------------------
# h5py's get/__getitem__ are typed as unions over groups, datasets, datatypes and
# links. Narrow them explicitly so the type checker is satisfied and malformed
# files (a name present but of the wrong kind) fail with a clear error.


def _open_group(parent: h5py.Group, name: str) -> h5py.Group | None:
    obj = parent.get(name)
    if obj is None:
        return None
    if not isinstance(obj, h5py.Group):
        raise TypeError(f"expected {name!r} to be an HDF5 group, got {type(obj).__name__}")
    return obj


def _require_dataset(grp: h5py.Group, name: str) -> h5py.Dataset:
    obj = grp.get(name)
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"expected {name!r} to be an HDF5 dataset, got {type(obj).__name__}")
    return obj


def _datasets(grp: h5py.Group) -> Iterator[tuple[str, h5py.Dataset]]:
    for name, obj in grp.items():
        if not isinstance(obj, h5py.Dataset):
            raise TypeError(f"expected {name!r} to be an HDF5 dataset, got {type(obj).__name__}")
        yield name, obj


def _read_spatial_dims(f: h5py.File) -> list[str]:
    grp = _open_group(f, s.GROUP_DIMENSIONS)
    if grp is None:
        return []
    return _as_str_list(grp.attrs.get(s.ATTR_SPATIAL_DIMS, []))


def _read_dimensions(f: h5py.File) -> list[s.Dimension]:
    grp = _open_group(f, s.GROUP_DIMENSIONS)
    if grp is None:
        return []
    return [s.Dimension(name=name, coordinates=np.asarray(ds[()])) for name, ds in _datasets(grp)]


def _read_boundary_conditions(f: h5py.File) -> list[s.BoundaryCondition]:
    grp = _open_group(f, s.GROUP_BOUNDARY_CONDITIONS)
    if grp is None:
        return []
    bcs: list[s.BoundaryCondition] = []
    for name, obj in grp.items():
        if not isinstance(obj, h5py.Group):
            raise TypeError(f"boundary condition {name!r} must be an HDF5 group")
        mask = obj.get("mask")
        values = obj.get("values")
        bcs.append(
            s.BoundaryCondition(
                name=name,
                bc_type=cast(s.BCType, _as_str(obj.attrs[s.ATTR_BC_TYPE])),
                associated_fields=_as_str_list(obj.attrs.get(s.ATTR_ASSOCIATED_FIELDS, [])),
                associated_dims=_as_str_list(obj.attrs.get(s.ATTR_ASSOCIATED_DIMS, [])),
                mask=np.asarray(mask[()]) if isinstance(mask, h5py.Dataset) else None,
                values=np.asarray(values[()]) if isinstance(values, h5py.Dataset) else None,
            )
        )
    return bcs


def _read_normalization(f: h5py.File) -> dict[str, s.Normalization]:
    grp = _open_group(f, s.GROUP_NORMALIZATION)
    if grp is None:
        return {}
    out: dict[str, s.Normalization] = {}
    for name, obj in grp.items():
        if not isinstance(obj, h5py.Group):
            raise TypeError(f"normalization entry {name!r} must be an HDF5 group")
        norm_min = obj.attrs.get(s.ATTR_MIN)
        norm_max = obj.attrs.get(s.ATTR_MAX)
        out[name] = s.Normalization(
            mean=np.asarray(obj.attrs[s.ATTR_MEAN]),
            std=np.asarray(obj.attrs[s.ATTR_STD]),
            min=np.asarray(norm_min) if norm_min is not None else None,
            max=np.asarray(norm_max) if norm_max is not None else None,
        )
    return out


def _read_fields(
    f: h5py.File, registry: dict[str, s.FieldSpec], present: list[str]
) -> list[s.Field]:
    fields: list[s.Field] = []
    for name in present:
        spec = registry.get(name)
        if spec is None:
            raise ValueError(f"present field {name!r} has no field_registry entry")
        grp = _open_group(f, s.FIELD_GROUPS[spec.tensor_order])
        if grp is None:
            raise ValueError(f"missing group {s.FIELD_GROUPS[spec.tensor_order]!r} for {name!r}")
        ds = _require_dataset(grp, name)
        fields.append(
            s.Field(
                name=name,
                data=np.asarray(ds[()]),
                tensor_order=spec.tensor_order,
                role=spec.role,
                unit=spec.unit,
                dim_varying=_as_bool(ds.attrs[s.ATTR_DIM_VARYING]),
                time_varying=_as_bool(ds.attrs[s.ATTR_TIME_VARYING]),
                sample_varying=_as_bool(ds.attrs[s.ATTR_SAMPLE_VARYING]),
                symmetric=_as_bool(ds.attrs.get(s.ATTR_SYMMETRIC, False)),
                antisymmetric=_as_bool(ds.attrs.get(s.ATTR_ANTISYMMETRIC, False)),
            )
        )
    return fields


# --- attribute encoding/decoding ----------------------------------------------
# h5py returns bytes for legacy string attrs and numpy scalars for bool/int;
# normalize them to plain Python types on the way out.


def _str_array(values: list[str]) -> np.ndarray:
    return np.array(list(values), dtype=_STR)


def _as_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _as_str_list(value) -> list[str]:
    return [_as_str(v) for v in np.atleast_1d(value)]


def _as_bool(value) -> bool:
    return bool(value)


def _as_int(value) -> int:
    return int(value)
