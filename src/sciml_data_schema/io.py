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
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal, cast

import h5py
import numpy as np

from . import schema as s

__all__ = [
    "write_dataset",
    "read_dataset",
    "Compression",
    "DatasetWriter",
    "WriteMode",
    "WRITE_MODES",
    "field_placeholder",
]

StrPath = str | PathLike[str]

# h5py.File modes that create a file; modes that open an existing file for editing are rejected
WriteMode = Literal["w", "w-", "x"]
WRITE_MODES: tuple[WriteMode, ...] = ("w", "w-", "x")

# Variable-length UTF-8 string dtype for HDF5 string-array attributes
_STR = h5py.string_dtype("utf-8")


@dataclass(frozen=True)
class Compression:
    """HDF5 storage filters applied to every field and boundary-condition dataset.

    Attributes
    ----------
    name : str or None
        HDF5 compression filter (e.g. ``"gzip"``, ``"lzf"``); ``None`` for
        uncompressed.
    level : int or None
        Filter option passed as ``compression_opts`` (the gzip level, 0-9);
        ``None`` for the filter's default. Ignored when ``name`` is ``None``.
    shuffle : bool
        Whether to apply the byte-shuffle filter before compression.

    Examples
    --------
    Byte-shuffle with fast gzip, which usually beats plain gzip on smooth float fields:

    >>> Compression("gzip", level=1, shuffle=True)
    Compression(name='gzip', level=1, shuffle=True)
    """

    name: str | None = None
    level: int | None = None
    shuffle: bool = False

    def dataset_kwargs(self) -> dict[str, object]:
        """Translate this spec into ``h5py.Group.create_dataset`` keyword arguments.

        Returns
        -------
        dict[str, object]
            ``shuffle``, plus ``compression`` and ``compression_opts`` when set.
        """
        kwargs: dict[str, object] = {"shuffle": self.shuffle}
        if self.name is not None:
            kwargs["compression"] = self.name
            if self.level is not None:
                kwargs["compression_opts"] = self.level
        return kwargs


CompressionLike = str | Compression | None


def _as_compression(value: CompressionLike) -> Compression:
    """Normalize ``None`` (uncompressed), a filter name, or a full spec to a spec."""
    if value is None:
        return Compression()
    if isinstance(value, str):
        return Compression(name=value)
    return value


def write_dataset(
    dataset: s.Dataset,
    path: StrPath,
    *,
    mode: WriteMode = "w",
    compression: CompressionLike = None,
) -> Path:
    """Validate ``dataset`` and write it to ``path`` as a single HDF5 file.

    This is :class:`DatasetWriter` with nothing streamed; use the writer
    directly when a dataset's bulk fields do not fit in memory.

    Parameters
    ----------
    dataset : Dataset
        Dataset to write, with all field data in memory.
    path : str or PathLike
        Destination file.
    mode : WriteMode
        ``h5py.File`` mode; ``"w"`` replaces an existing file, ``"w-"``/``"x"``
        fail if the file exists.
    compression : str, Compression or None
        Filter name (e.g. ``"gzip"``), full spec, or ``None`` for uncompressed.

    Returns
    -------
    Path
        ``path`` as a :class:`~pathlib.Path`.

    Raises
    ------
    ValueError
        If ``mode`` is not in ``WRITE_MODES`` or :meth:`Dataset.validate`
        fails; nothing is written.
    FileExistsError
        If ``mode`` is ``"w-"``/``"x"`` and the file exists; it is left untouched.
    """
    with DatasetWriter(dataset, path, streamed=(), mode=mode, compression=compression):
        pass
    return Path(path)


def field_placeholder(shape: tuple[int, ...], dtype: np.dtype | type = np.float32) -> np.ndarray:
    """Build a zero-copy stand-in for a streamed field's ``data``.

    Gives a :class:`~sciml_data_schema.schema.Field` the shape and dtype the
    validator needs without allocating the array; the values are never written.
    Use it for the fields named in :class:`DatasetWriter`'s ``streamed``.

    Parameters
    ----------
    shape : tuple[int, ...]
        Full stored shape of the field, trajectory axis first.
    dtype : np.dtype or type
        Element type the streamed slices will have.

    Returns
    -------
    np.ndarray
        Read-only broadcast view of a single zero.
    """
    return np.broadcast_to(np.zeros((), dtype=dtype), shape)


class DatasetWriter:
    """Write a dataset whose bulk fields exceed memory, one trajectory at a time.

    On entering the context, the root metadata, dimensions, boundary
    conditions and every non-streamed field are written, and each streamed
    field's HDF5 dataset is created at full shape, chunked by trajectory (and
    timestep, if present). Fill it with :meth:`append`. Leaving the context
    checks that every trajectory of every streamed field was written and
    writes the normalization. On any failure -- an exception inside the
    block, or missing trajectories -- the file is deleted. The writer only
    creates files; it never edits an existing one.

    Parameters
    ----------
    dataset : Dataset
        Complete dataset; each field named in ``streamed`` carries
        :func:`field_placeholder` data.
    path : str or PathLike
        Destination file.
    streamed : tuple[str, ...] or list[str]
        Fields written trajectory by trajectory; each must be ``sample_varying``.
    mode : WriteMode
        ``h5py.File`` mode; ``"w"`` replaces an existing file (which is then
        deleted if the write fails), ``"w-"``/``"x"`` fail if the file exists.
    compression : str, Compression or None
        Filter name, full spec, or ``None`` for uncompressed.

    Raises
    ------
    ValueError
        If ``mode`` is not in ``WRITE_MODES``, :meth:`Dataset.validate` fails,
        a ``streamed`` name is not a field of ``dataset``, or a streamed field
        is not ``sample_varying``.

    Examples
    --------
    >>> with DatasetWriter(dataset, "out.h5", streamed=["u"]) as writer:  # doctest: +SKIP
    ...     for i in range(dataset.n_trajectories):
    ...         writer.append("u", i, simulate(i))
    """

    def __init__(
        self,
        dataset: s.Dataset,
        path: StrPath,
        *,
        streamed: tuple[str, ...] | list[str],
        mode: WriteMode = "w",
        compression: CompressionLike = None,
    ) -> None:
        # Editing modes ("a", "r+") would overwrite an existing file's attributes, then delete it
        if mode not in WRITE_MODES:
            raise ValueError(f"unsupported mode {mode!r}: expected one of {list(WRITE_MODES)}")

        # Validate before touching the disk, mirroring write_dataset's contract
        dataset.validate()

        # Streamed names must exist and have a leading trajectory axis to append along
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
        # Written on close, so producers can set stats known only after streaming
        self._normalization = dict(dataset.normalization)

    def __enter__(self) -> DatasetWriter:
        """Open the file and write everything except streamed data and normalization.

        Returns
        -------
        DatasetWriter
            This writer.

        Raises
        ------
        FileExistsError
            If ``mode`` is ``"w-"``/``"x"`` and the file exists; it is left untouched.
        OSError
            If the file cannot be created; any error while writing deletes the
            file and propagates.
        """
        dataset = self._dataset
        registry = dataset.effective_registry()

        # Write all but the streamed data and normalization; delete the file on any failure
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

        self._written = {
            name: np.zeros(self._targets[name].shape[0], dtype=bool) for name in self._streamed
        }
        return self

    def append(self, name: str, trajectory: int, data: np.ndarray) -> None:
        """Write one trajectory's slice of a streamed field.

        Writing the same trajectory again overwrites it.

        Parameters
        ----------
        name : str
            Streamed field to write.
        trajectory : int
            Trajectory index, in ``[0, n_trajectories)``.
        data : np.ndarray
            Slice of the field's shape without the trajectory axis, and of
            exactly its dtype (no casting).

        Raises
        ------
        RuntimeError
            If called outside the context.
        KeyError
            If ``name`` is not a streamed field.
        IndexError
            If ``trajectory`` is out of range.
        ValueError
            If ``data`` has the wrong shape or dtype.
        """
        if self._file is None:
            raise RuntimeError("DatasetWriter used outside its context")
        if name not in self._targets:
            raise KeyError(f"not a streamed field: {name!r}")

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

        This is where stats known only after streaming go (e.g. an online
        mean/std over the pass).

        Parameters
        ----------
        name : str
            Field the statistics belong to; any present field, streamed or not.
        normalization : Normalization
            Statistics to write on close.

        Raises
        ------
        RuntimeError
            If called outside the context.
        ValueError
            If ``name`` is not a present field.
        """
        if self._file is None:
            raise RuntimeError("DatasetWriter used outside its context")
        if name not in self._dataset.present_fields:
            raise ValueError(f"normalization references unknown field {name!r}")
        self._normalization[name] = normalization

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        """Write the normalization and close the file, or delete it on failure.

        Returns
        -------
        bool
            ``False``; an exception raised inside the block propagates after
            the file is deleted.

        Raises
        ------
        ValueError
            If a streamed field has unwritten trajectories; the file is deleted.
        """
        # An exception inside the block wins; drop the partial file and re-raise
        if exc_type is not None:
            self._abort()
            return False

        missing = {
            name: int((~mask).sum()) for name, mask in self._written.items() if not mask.all()
        }
        if missing:
            self._abort()
            raise ValueError(f"incomplete streamed fields (missing trajectories): {missing}")

        assert self._file is not None
        _write_normalization(self._file, self._normalization)
        self._file.close()
        self._file = None
        return False

    def _abort(self) -> None:
        """Close the handle and remove the partial file."""
        if self._file is not None:
            self._file.close()
            self._file = None
        self._path.unlink(missing_ok=True)


def _write_root_attrs(f: h5py.File, dataset: s.Dataset, registry: dict[str, s.FieldSpec]) -> None:
    """Write the root attributes; ``provenance`` only when non-empty."""
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


def read_dataset(path: StrPath) -> s.Dataset:
    """Load an HDF5 file written by :func:`write_dataset` into a :class:`Dataset`.

    All field data is read into memory. A missing ``grid_type`` reads as
    ``"cartesian"`` and a missing ``provenance`` as ``{}``.

    Parameters
    ----------
    path : str or PathLike
        File to read.

    Returns
    -------
    Dataset
        The validated dataset.

    Raises
    ------
    KeyError
        If a required attribute is missing.
    TypeError
        If a name refers to the wrong kind of HDF5 object.
    ValueError
        If a present field has no registry entry or group, or the loaded
        dataset fails :meth:`Dataset.validate`.
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


def _read_provenance(f: h5py.File) -> dict[str, str]:
    """Read the optional free-form provenance object (absent in older files)."""
    if s.ATTR_PROVENANCE not in f.attrs:
        return {}
    return json.loads(_as_str(f.attrs[s.ATTR_PROVENANCE]))


# __ Group writers _______


def _write_dimensions(f: h5py.File, dimensions: list[s.Dimension], spatial_dims: list[str]) -> None:
    """Create the dimensions group: spatial axis names and one coordinate dataset per axis."""
    grp = f.create_group(s.GROUP_DIMENSIONS)
    grp.attrs[s.ATTR_SPATIAL_DIMS] = _str_array(spatial_dims)
    for dim in dimensions:
        grp.create_dataset(dim.name, data=np.asarray(dim.coordinates))


def _write_boundary_conditions(
    f: h5py.File, bcs: list[s.BoundaryCondition], compression: Compression
) -> None:
    """Create the boundary-conditions group, one subgroup per condition."""
    grp = f.create_group(s.GROUP_BOUNDARY_CONDITIONS)
    for bc in bcs:
        sub = grp.create_group(bc.name)
        sub.attrs[s.ATTR_BC_TYPE] = bc.bc_type
        sub.attrs[s.ATTR_ASSOCIATED_FIELDS] = _str_array(bc.associated_fields)
        sub.attrs[s.ATTR_ASSOCIATED_DIMS] = _str_array(bc.associated_dims)
        if bc.mask is not None:
            sub.create_dataset("mask", data=np.asarray(bc.mask), **compression.dataset_kwargs())
        if bc.values is not None:
            sub.create_dataset("values", data=np.asarray(bc.values), **compression.dataset_kwargs())


def _write_normalization(f: h5py.File, normalization: dict[str, s.Normalization]) -> None:
    """Create the normalization group, one subgroup of statistic attributes per field."""
    grp = f.create_group(s.GROUP_NORMALIZATION)
    for name, norm in normalization.items():
        sub = grp.create_group(name)
        sub.attrs[s.ATTR_MEAN] = np.asarray(norm.mean)
        sub.attrs[s.ATTR_STD] = np.asarray(norm.std)
        # Range stats are optional (schema >= 0.3.0); absent attrs read back as None
        if norm.min is not None:
            sub.attrs[s.ATTR_MIN] = np.asarray(norm.min)
        if norm.max is not None:
            sub.attrs[s.ATTR_MAX] = np.asarray(norm.max)


def _stream_chunks(fld: s.Field, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Chunk shape for a streamed field, matching how loaders read it.

    One (trajectory, timestep) slab per chunk when the field has a time axis,
    one trajectory otherwise.
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
    """Create the field groups and write every field dataset with its attributes.

    Parameters
    ----------
    f : h5py.File
        Open file to write into.
    fields : list[Field]
        Fields to write.
    compression : Compression
        Filters for every field dataset.
    streamed : frozenset[str]
        Fields created empty at full shape; their placeholder data is ignored.

    Returns
    -------
    dict[str, h5py.Dataset]
        The streamed fields' datasets, to be filled trajectory by trajectory.
    """
    # Create every field group up front so readers can rely on their presence
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


# __ Group readers _______

# h5py lookups return a union of object kinds; these helpers narrow it or raise TypeError


def _open_group(parent: h5py.Group, name: str) -> h5py.Group | None:
    """Return the group ``name`` of ``parent``, or ``None`` if absent; TypeError if not a group."""
    obj = parent.get(name)
    if obj is None:
        return None
    if not isinstance(obj, h5py.Group):
        raise TypeError(f"expected {name!r} to be an HDF5 group, got {type(obj).__name__}")
    return obj


def _require_dataset(grp: h5py.Group, name: str) -> h5py.Dataset:
    """Return the dataset ``name`` of ``grp``; TypeError if absent or not a dataset."""
    obj = grp.get(name)
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"expected {name!r} to be an HDF5 dataset, got {type(obj).__name__}")
    return obj


def _datasets(grp: h5py.Group) -> Iterator[tuple[str, h5py.Dataset]]:
    """Yield ``(name, dataset)`` for each member of ``grp``; TypeError on a non-dataset."""
    for name, obj in grp.items():
        if not isinstance(obj, h5py.Dataset):
            raise TypeError(f"expected {name!r} to be an HDF5 dataset, got {type(obj).__name__}")
        yield name, obj


def _read_spatial_dims(f: h5py.File) -> list[str]:
    """Read the spatial axis names; ``[]`` if the dimensions group is absent."""
    grp = _open_group(f, s.GROUP_DIMENSIONS)
    if grp is None:
        return []
    return _as_str_list(grp.attrs.get(s.ATTR_SPATIAL_DIMS, []))


def _read_dimensions(f: h5py.File) -> list[s.Dimension]:
    """Read every coordinate dataset; ``[]`` if the dimensions group is absent."""
    grp = _open_group(f, s.GROUP_DIMENSIONS)
    if grp is None:
        return []
    return [s.Dimension(name=name, coordinates=np.asarray(ds[()])) for name, ds in _datasets(grp)]


def _read_boundary_conditions(f: h5py.File) -> list[s.BoundaryCondition]:
    """Read every boundary condition; ``[]`` if the group is absent.

    Raises
    ------
    TypeError
        If a member of the group is not a group.
    KeyError
        If a condition lacks ``bc_type``.
    """
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
    """Read the statistics of every field; ``{}`` if the group is absent.

    Raises
    ------
    TypeError
        If a member of the group is not a group.
    KeyError
        If an entry lacks ``mean`` or ``std``.
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
    """Read the present fields, taking their identity from the registry.

    Parameters
    ----------
    f : h5py.File
        Open file to read from.
    registry : dict[str, FieldSpec]
        Decoded ``field_registry`` attribute.
    present : list[str]
        Decoded ``present_fields`` attribute.

    Returns
    -------
    list[Field]
        Fields in ``present`` order.

    Raises
    ------
    ValueError
        If a present field has no registry entry or its tensor-order group is missing.
    TypeError
        If a present field is not a dataset.
    KeyError
        If a field lacks an extent-flag attribute.
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


# __ Attribute encoding _______

# h5py returns bytes for legacy string attrs and numpy scalars for bool/int


def _str_array(values: list[str]) -> np.ndarray:
    """Encode strings as a variable-length UTF-8 array for an attribute."""
    return np.array(list(values), dtype=_STR)


def _as_str(value) -> str:
    """Decode a string attribute stored as ``str`` or legacy ``bytes``."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _as_str_list(value) -> list[str]:
    """Decode a scalar or array string attribute into a list."""
    return [_as_str(v) for v in np.atleast_1d(value)]


def _as_bool(value) -> bool:
    """Convert a numpy bool attribute to ``bool``."""
    return bool(value)


def _as_int(value) -> int:
    """Convert a numpy integer attribute to ``int``."""
    return int(value)
