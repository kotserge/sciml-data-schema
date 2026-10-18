"""HDF5 read and write of a :class:`~sciml_data_schema.schema.Dataset`.

The only module that imports ``h5py``. A file mirrors the schema's core layout: the root
attributes (the versions, the name, ``representation`` and ``grid_type``, the counts, the
``field_registry`` JSON, the ``present_fields`` list and the optional ``provenance`` JSON),
the ``dimensions`` group (one coordinate dataset per axis and the ``spatial_dims``
attribute), one group per boundary condition under ``boundary_conditions``, one group per
field under ``normalization``, and the ``t0_fields``, ``t1_fields`` and ``t2_fields``
groups holding the field datasets. A field dataset carries its extent flags (and, at rank
2, its symmetry flags) as attributes; its tensor order, role and unit come from the
registry.

:func:`write_dataset` writes an in-memory dataset in one call; :class:`DatasetWriter`
writes one whose bulk fields do not fit in memory, trajectory by trajectory;
:func:`read_dataset` loads a file whole and validates it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np

from . import schema as s

__all__ = ["write_dataset", "read_dataset", "Compression", "DatasetWriter", "field_placeholder"]

StrPath = str | PathLike[str]

# The variable-length UTF-8 string dtype of the string-array attributes
_STR = h5py.string_dtype("utf-8")

# The dataset names inside a boundary condition's group
_BC_MASK = "mask"
_BC_VALUES = "values"


@dataclass(frozen=True)
class Compression:
    """The HDF5 storage filters applied to every field and boundary-condition dataset.

    Attributes
    ----------
    name : str | None
        The compression filter (``"gzip"``, ``"lzf"``, ...), or ``None`` for none.
    level : int | None
        The filter's level; gzip only, 0 to 9, ``None`` for h5py's default.
    shuffle : bool
        Whether the byte-shuffle filter runs before the compression filter.
    """

    name: str | None = None
    level: int | None = None
    shuffle: bool = False

    def dataset_kwargs(self) -> dict[str, object]:
        """The ``h5py.Group.create_dataset`` keyword arguments of this spec.

        Returns
        -------
        dict[str, object]
            ``shuffle`` always; ``compression`` when ``name`` is set; ``compression_opts``
            when ``level`` is set too.
        """
        kwargs: dict[str, object] = {"shuffle": self.shuffle}
        if self.name is not None:
            kwargs["compression"] = self.name
            if self.level is not None:
                kwargs["compression_opts"] = self.level
        return kwargs


# The accepted spellings of a compression setting: nothing, a filter name, or a full spec
CompressionLike = str | Compression | None


def _as_compression(value: CompressionLike) -> Compression:
    """The :class:`Compression` of a ``CompressionLike`` value."""
    if value is None:
        return Compression()
    if isinstance(value, str):
        return Compression(name=value)
    return value


def write_dataset(
    dataset: s.Dataset,
    path: StrPath,
    *,
    mode: str = "w",
    compression: CompressionLike = None,
) -> Path:
    """Validate ``dataset`` and write it to ``path`` as one HDF5 file.

    Parameters
    ----------
    dataset : s.Dataset
        The dataset, with every field's data in memory.
    path : StrPath
        The file to write.
    mode : str
        The ``h5py.File`` mode; ``"w"`` truncates an existing file.
    compression : CompressionLike
        A filter name (``"gzip"``), a :class:`Compression` spec, or ``None`` for none.

    Returns
    -------
    Path
        ``path`` as a :class:`~pathlib.Path`.

    Raises
    ------
    ValueError
        The dataset fails :meth:`~sciml_data_schema.schema.Dataset.validate`; no file is
        created.

    Notes
    -----
    Equivalent to a :class:`DatasetWriter` with no streamed fields.
    """
    with DatasetWriter(dataset, path, streamed=(), mode=mode, compression=compression):
        pass
    return Path(path)


def field_placeholder(shape: tuple[int, ...], dtype: np.dtype | type = np.float32) -> np.ndarray:
    """A zero-copy stand-in for the ``data`` of a field :class:`DatasetWriter` streams.

    Parameters
    ----------
    shape : tuple[int, ...]
        The field's full stored shape, leading trajectory axis included.
    dtype : np.dtype | type
        The field's dtype.

    Returns
    -------
    np.ndarray
        A read-only broadcast view of one zero with the given shape and dtype; nothing of
        size ``shape`` is allocated, and the values are never written.
    """
    return np.broadcast_to(np.zeros((), dtype=dtype), shape)


class DatasetWriter:
    """Write a dataset whose bulk fields exceed memory, one trajectory at a time.

    Entering the context writes the root metadata, the dimensions, the boundary conditions
    and every non-streamed field, and creates each streamed field's dataset at full shape,
    chunked by trajectory (and by timestep when the field has a time axis).
    :meth:`append` fills one trajectory of one streamed field; :meth:`update_normalization`
    sets a field's statistics. Leaving the context checks that every trajectory of every
    streamed field was written, writes the normalization group and closes the file. On an
    exception inside the block, or on a missing trajectory, the file is deleted.

    Parameters
    ----------
    dataset : s.Dataset
        The complete dataset; each field named in ``streamed`` carries
        :func:`field_placeholder` data of its full stored shape and dtype.
    path : StrPath
        The file to write.
    streamed : tuple[str, ...] | list[str]
        The fields filled through :meth:`append`; each must be ``sample_varying``.
    mode : str
        The ``h5py.File`` mode; ``"w"`` truncates an existing file.
    compression : CompressionLike
        A filter name, a :class:`Compression` spec, or ``None`` for none.

    Raises
    ------
    ValueError
        The dataset fails :meth:`~sciml_data_schema.schema.Dataset.validate`; a streamed name
        is not a field of the dataset; a streamed field is not ``sample_varying``. Raised by
        the constructor, before any file is created.

    Examples
    --------
    ::

        with DatasetWriter(dataset, "run.h5", streamed=("u",)) as writer:
            for trajectory in range(dataset.n_trajectories):
                writer.append("u", trajectory, solve(trajectory))
            writer.update_normalization("u", Normalization(mean=mean, std=std))
    """

    def __init__(
        self,
        dataset: s.Dataset,
        path: StrPath,
        *,
        streamed: tuple[str, ...] | list[str],
        mode: str = "w",
        compression: CompressionLike = None,
    ) -> None:
        # Validate before any file exists
        dataset.validate()

        # Every streamed name is a field with a leading trajectory axis to fill along
        fields_by_name = {fld.name: fld for fld in dataset.fields}
        unknown = set(streamed) - set(fields_by_name)
        if unknown:
            raise ValueError(f"streamed fields not in dataset: {sorted(unknown)}")
        for name in streamed:
            if not fields_by_name[name].sample_varying:
                raise ValueError(f"streamed field {name!r} must be sample_varying")

        self._dataset = dataset
        self._path = Path(path)
        self._streamed = tuple(streamed)
        self._mode = mode
        self._compression = _as_compression(compression)
        self._file: h5py.File | None = None
        self._targets: dict[str, h5py.Dataset] = {}
        self._written: dict[str, np.ndarray] = {}
        # The statistics written on close; `update_normalization` replaces entries until then
        self._normalization = dict(dataset.normalization)

    def __enter__(self) -> DatasetWriter:
        dataset = self._dataset
        registry = dataset.effective_registry()

        # Write everything but the streamed bulk and the normalization; on a failure, delete the
        # partial file and re-raise
        self._file = f = h5py.File(self._path, self._mode)
        try:
            _write_root_attrs(f, dataset, registry)
            _write_dimensions(f, dataset.dimensions, dataset.spatial_dims)
            _write_boundary_conditions(f, dataset.boundary_conditions, self._compression)
            self._targets = _write_fields(
                f, dataset.fields, self._compression, streamed=frozenset(self._streamed)
            )
        except Exception:
            self._abort()
            raise

        # One flag per trajectory of each streamed field, set by `append`
        self._written = {
            name: np.zeros(self._targets[name].shape[0], dtype=bool) for name in self._streamed
        }
        return self

    def append(self, name: str, trajectory: int, data: np.ndarray) -> None:
        """Write one trajectory of a streamed field.

        Parameters
        ----------
        name : str
            A field named in ``streamed``.
        trajectory : int
            The trajectory index, ``0 <= trajectory < n_trajectories``.
        data : np.ndarray
            The trajectory's values: the field's stored shape without the leading axis, in
            the field's dtype.

        Raises
        ------
        RuntimeError
            The writer is not inside its context.
        KeyError
            ``name`` is not a streamed field.
        IndexError
            ``trajectory`` is outside the field's trajectory axis.
        ValueError
            ``data`` has another shape or dtype than the field.
        """
        if self._file is None:
            raise RuntimeError("DatasetWriter used outside its context")
        if name not in self._targets:
            raise KeyError(f"not a streamed field: {name!r}")

        # The slice fits the target exactly: index, shape and dtype
        target = self._targets[name]
        if not 0 <= trajectory < target.shape[0]:
            raise IndexError(
                f"field {name!r}: trajectory {trajectory} outside [0, {target.shape[0]})"
            )
        data = np.asarray(data)
        if data.shape != target.shape[1:]:
            raise ValueError(
                f"field {name!r}, trajectory {trajectory}: expected shape "
                f"{target.shape[1:]}, got {data.shape}"
            )
        if data.dtype != target.dtype:
            raise ValueError(
                f"field {name!r}, trajectory {trajectory}: expected dtype "
                f"{target.dtype}, got {data.dtype}"
            )

        target[trajectory] = data
        self._written[name][trajectory] = True

    def update_normalization(self, name: str, normalization: s.Normalization) -> None:
        """Set or replace a field's statistics before the file is closed.

        Parameters
        ----------
        name : str
            A present field of the dataset.
        normalization : s.Normalization
            The statistics written for it on close.

        Raises
        ------
        RuntimeError
            The writer is not inside its context.
        ValueError
            ``name`` is not a field of the dataset.
        """
        if self._file is None:
            raise RuntimeError("DatasetWriter used outside its context")
        if name not in self._dataset.present_fields:
            raise ValueError(f"normalization references unknown field {name!r}")
        self._normalization[name] = normalization

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        # An exception inside the block: delete the partial file and let it propagate
        if exc_type is not None:
            self._abort()
            return False

        # Every trajectory of every streamed field was written
        missing = {
            name: int((~mask).sum()) for name, mask in self._written.items() if not mask.all()
        }
        if missing:
            self._abort()
            raise ValueError(f"incomplete streamed fields (missing trajectories): {missing}")

        # The statistics go last, then the file is complete
        assert self._file is not None
        _write_normalization(self._file, self._normalization)
        self._file.close()
        self._file = None
        return False

    def _abort(self) -> None:
        """Close the file handle and delete the file at ``path``."""
        if self._file is not None:
            self._file.close()
            self._file = None
        self._path.unlink(missing_ok=True)


# === Group writers ===


def _write_root_attrs(f: h5py.File, dataset: s.Dataset, registry: dict[str, s.FieldSpec]) -> None:
    """Write the root attributes; ``provenance`` only when the dataset carries one."""
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


def _write_dimensions(f: h5py.File, dimensions: list[s.Dimension], spatial_dims: list[str]) -> None:
    """Write the ``dimensions`` group: the ``spatial_dims`` attribute and one dataset per axis."""
    grp = f.create_group(s.GROUP_DIMENSIONS)
    grp.attrs[s.ATTR_SPATIAL_DIMS] = _str_array(spatial_dims)
    for dim in dimensions:
        grp.create_dataset(dim.name, data=np.asarray(dim.coordinates))


def _write_boundary_conditions(
    f: h5py.File, bcs: list[s.BoundaryCondition], compression: Compression
) -> None:
    """Write the ``boundary_conditions`` group, one subgroup per condition.

    A subgroup holds the condition's attributes and its optional ``mask`` and ``values``
    datasets.
    """
    grp = f.create_group(s.GROUP_BOUNDARY_CONDITIONS)
    for bc in bcs:
        sub = grp.create_group(bc.name)
        sub.attrs[s.ATTR_BC_TYPE] = bc.bc_type
        sub.attrs[s.ATTR_ASSOCIATED_FIELDS] = _str_array(bc.associated_fields)
        sub.attrs[s.ATTR_ASSOCIATED_DIMS] = _str_array(bc.associated_dims)
        if bc.mask is not None:
            sub.create_dataset(_BC_MASK, data=np.asarray(bc.mask), **compression.dataset_kwargs())
        if bc.values is not None:
            sub.create_dataset(
                _BC_VALUES, data=np.asarray(bc.values), **compression.dataset_kwargs()
            )


def _write_normalization(f: h5py.File, normalization: dict[str, s.Normalization]) -> None:
    """Write the ``normalization`` group, one subgroup per field holding its statistics.

    ``mean`` and ``std`` are always written, ``min`` and ``max`` when the entry has them.
    """
    grp = f.create_group(s.GROUP_NORMALIZATION)
    for name, norm in normalization.items():
        sub = grp.create_group(name)
        sub.attrs[s.ATTR_MEAN] = np.asarray(norm.mean)
        sub.attrs[s.ATTR_STD] = np.asarray(norm.std)
        if norm.min is not None:
            sub.attrs[s.ATTR_MIN] = np.asarray(norm.min)
        if norm.max is not None:
            sub.attrs[s.ATTR_MAX] = np.asarray(norm.max)


def _stream_chunks(fld: s.Field, shape: tuple[int, ...]) -> tuple[int, ...]:
    """The chunk shape of a streamed field.

    One (trajectory, timestep) slab per chunk when the field has a time axis, one trajectory
    otherwise.
    """
    if fld.time_varying and len(shape) >= 2:
        return (1, 1, *shape[2:])
    return (1, *shape[1:])


def _write_fields(
    f: h5py.File,
    fields: list[s.Field],
    compression: Compression,
    *,
    streamed: frozenset[str] = frozenset(),
) -> dict[str, h5py.Dataset]:
    """Write the three field groups and every field dataset with its flag attributes.

    A streamed field's dataset is created at full shape without data (its placeholder is not
    read) and returned for trajectory-wise filling. Every field group exists afterwards, empty
    or not.
    """
    groups = {order: f.create_group(name) for order, name in s.FIELD_GROUPS.items()}

    targets: dict[str, h5py.Dataset] = {}
    for fld in fields:
        if fld.name in streamed:
            ds = groups[fld.tensor_order].create_dataset(
                fld.name,
                shape=fld.data.shape,
                dtype=fld.data.dtype,
                chunks=_stream_chunks(fld, fld.data.shape),
                **compression.dataset_kwargs(),
            )
            targets[fld.name] = ds
        else:
            ds = groups[fld.tensor_order].create_dataset(
                fld.name, data=np.asarray(fld.data), **compression.dataset_kwargs()
            )
        ds.attrs[s.ATTR_DIM_VARYING] = bool(fld.dim_varying)
        ds.attrs[s.ATTR_TIME_VARYING] = bool(fld.time_varying)
        ds.attrs[s.ATTR_SAMPLE_VARYING] = bool(fld.sample_varying)
        if fld.tensor_order == 2:
            ds.attrs[s.ATTR_SYMMETRIC] = bool(fld.symmetric)
            ds.attrs[s.ATTR_ANTISYMMETRIC] = bool(fld.antisymmetric)
    return targets


# === Reader ===


def read_dataset(path: StrPath) -> s.Dataset:
    """Load an HDF5 file written by :func:`write_dataset` or :class:`DatasetWriter`.

    Every field's data is read into memory. A file without a ``grid_type`` attribute reads
    as ``"cartesian"``; one without ``provenance`` reads as an empty mapping.

    Parameters
    ----------
    path : StrPath
        The file to read.

    Returns
    -------
    s.Dataset
        The dataset, validated.

    Raises
    ------
    ValueError
        A present field has no registry entry or no dataset in its tensor order's group, or
        the loaded dataset fails :meth:`~sciml_data_schema.schema.Dataset.validate`.
    TypeError
        A name in the file is an HDF5 object of another kind than the layout prescribes.
    """
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


# === Group readers ===
# h5py's `get` returns a union over groups, datasets, datatypes and links; these narrow it to the
# kind the layout prescribes and raise TypeError on any other.


def _open_group(parent: h5py.Group, name: str) -> h5py.Group | None:
    """The group ``name`` under ``parent``, or ``None`` when absent."""
    obj = parent.get(name)
    if obj is None:
        return None
    if not isinstance(obj, h5py.Group):
        raise TypeError(f"expected {name!r} to be an HDF5 group, got {type(obj).__name__}")
    return obj


def _require_dataset(grp: h5py.Group, name: str) -> h5py.Dataset:
    """The dataset ``name`` under ``grp``."""
    obj = grp.get(name)
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"expected {name!r} to be an HDF5 dataset, got {type(obj).__name__}")
    return obj


def _datasets(grp: h5py.Group) -> Iterator[tuple[str, h5py.Dataset]]:
    """Every member of ``grp`` as ``(name, dataset)``."""
    for name, obj in grp.items():
        if not isinstance(obj, h5py.Dataset):
            raise TypeError(f"expected {name!r} to be an HDF5 dataset, got {type(obj).__name__}")
        yield name, obj


def _read_provenance(f: h5py.File) -> dict[str, str]:
    """The ``provenance`` attribute as a mapping; empty when the attribute is absent."""
    if s.ATTR_PROVENANCE not in f.attrs:
        return {}
    return json.loads(_as_str(f.attrs[s.ATTR_PROVENANCE]))


def _read_spatial_dims(f: h5py.File) -> list[str]:
    """The ``spatial_dims`` attribute of the ``dimensions`` group; empty when absent."""
    grp = _open_group(f, s.GROUP_DIMENSIONS)
    if grp is None:
        return []
    return _as_str_list(grp.attrs.get(s.ATTR_SPATIAL_DIMS, []))


def _read_dimensions(f: h5py.File) -> list[s.Dimension]:
    """One :class:`~sciml_data_schema.schema.Dimension` per dataset of the ``dimensions`` group."""
    grp = _open_group(f, s.GROUP_DIMENSIONS)
    if grp is None:
        return []
    return [s.Dimension(name=name, coordinates=np.asarray(ds[()])) for name, ds in _datasets(grp)]


def _read_boundary_conditions(f: h5py.File) -> list[s.BoundaryCondition]:
    """One ``BoundaryCondition`` per subgroup of ``boundary_conditions``."""
    grp = _open_group(f, s.GROUP_BOUNDARY_CONDITIONS)
    if grp is None:
        return []
    bcs: list[s.BoundaryCondition] = []
    for name, obj in grp.items():
        if not isinstance(obj, h5py.Group):
            raise TypeError(f"boundary condition {name!r} must be an HDF5 group")
        mask = obj.get(_BC_MASK)
        values = obj.get(_BC_VALUES)
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
    """One ``Normalization`` per subgroup of ``normalization``.

    ``min`` and ``max`` are ``None`` when their attributes are absent.
    """
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
    """One ``Field`` per present name.

    The identity comes from the registry entry, the flags from the dataset's attributes.
    """
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


# === Attribute encoding and decoding ===
# h5py returns bytes for legacy string attributes and numpy scalars for bool and int ones; these
# convert them to plain Python values.


def _str_array(values: list[str]) -> np.ndarray:
    """``values`` as a variable-length UTF-8 string array, the form of a string-list attribute."""
    return np.array(list(values), dtype=_STR)


def _as_str(value: Any) -> str:
    """A string attribute value (``str`` or ``bytes``) as ``str``."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _as_str_list(value: Any) -> list[str]:
    """A string-array attribute value as a list of ``str``; a scalar becomes a one-item list."""
    return [_as_str(v) for v in np.atleast_1d(value)]


def _as_bool(value: Any) -> bool:
    """A bool attribute value (a numpy scalar) as ``bool``."""
    return bool(value)


def _as_int(value: Any) -> int:
    """An int attribute value (a numpy scalar) as ``int``."""
    return int(value)
