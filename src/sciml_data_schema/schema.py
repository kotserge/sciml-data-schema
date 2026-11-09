"""In-memory dataset schema for the project's HDF5 format.

The layout follows ``SCHEMA.md``: a *unified field model* on top of the Well's
tensor-rank grouping. A single file holds one dataset of ``n_trajectories``
independent runs over a shared spatial domain. Every physical quantity --
solution, forcing, coefficient, parameter -- is a :class:`Field`. There is no
separate "scalars" group: a "scalar parameter" is just a rank-0 field that does
not vary in space or time.

A field carries two orthogonal pieces of metadata: its **tensor order** (0/1/2,
which selects the ``t{order}_fields`` group and fixes the trailing tensor axes)
and its **extent flags** (``dim_varying`` / ``time_varying`` /
``sample_varying``). An axis along which a field does not vary is dropped, so
the stored shape is::

    (n_trajectories?, n_timesteps?, *spatial_shape?, *(n_spatial_dims,) * tensor_order)

where each leading axis is present only if its flag is ``True``. A true global
constant (all flags ``False``, rank 0) may be stored as a 0-d scalar or as a
single length-1 axis ``(1,)``.

The *identity* of every quantity (its tensor order, role, and unit) is fixed by
the **field registry** -- a name-keyed table shared by convention across all
files of a representation, so ``velocity`` means the same thing everywhere. Each
file declares which registry entries are actually present.

This module defines the pure-Python representation (:class:`Dataset` and its
parts) plus the canonical HDF5 group/attribute names.
:mod:`sciml_data_schema.io` is the only place that touches HDF5; it serializes
the objects defined here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# __ Versions _______

SCHEMA_VERSION = "0.3.0"
# The registry version moves independently of SCHEMA_VERSION
REGISTRY_VERSION = "0.1.0"

# __ Canonical HDF5 names _______

GROUP_DIMENSIONS = "dimensions"
GROUP_BOUNDARY_CONDITIONS = "boundary_conditions"
GROUP_NORMALIZATION = "normalization"

# Field groups keyed by tensor order (rank)
FIELD_GROUPS: dict[int, str] = {0: "t0_fields", 1: "t1_fields", 2: "t2_fields"}

# Root-level attributes
ATTR_SCHEMA_VERSION = "schema_version"
ATTR_REGISTRY_VERSION = "registry_version"
ATTR_DATASET_NAME = "dataset_name"
ATTR_REPRESENTATION = "representation"
ATTR_GRID_TYPE = "grid_type"
ATTR_N_SPATIAL_DIMS = "n_spatial_dims"
ATTR_N_TRAJECTORIES = "n_trajectories"
ATTR_FIELD_REGISTRY = "field_registry"
ATTR_PRESENT_FIELDS = "present_fields"
# Optional provenance: a JSON object of string values
ATTR_PROVENANCE = "provenance"

# Attribute of the dimensions group holding the spatial axis names
ATTR_SPATIAL_DIMS = "spatial_dims"
# Reserved name of the time coordinate in the dimensions group
TIME_DIMENSION = "time"

# Per-field attributes: extent flags and tensor symmetry
ATTR_DIM_VARYING = "dim_varying"
ATTR_TIME_VARYING = "time_varying"
ATTR_SAMPLE_VARYING = "sample_varying"
ATTR_SYMMETRIC = "symmetric"
ATTR_ANTISYMMETRIC = "antisymmetric"

# Boundary-condition attributes
ATTR_BC_TYPE = "bc_type"
ATTR_ASSOCIATED_FIELDS = "associated_fields"
ATTR_ASSOCIATED_DIMS = "associated_dims"

# Normalization attributes; min/max are optional (schema >= 0.3.0)
ATTR_MEAN = "mean"
ATTR_STD = "std"
ATTR_MIN = "min"
ATTR_MAX = "max"

# Field-registry JSON keys
REG_TENSOR_ORDER = "tensor_order"
REG_ROLE = "role"
REG_UNIT = "unit"

# __ Controlled vocabularies _______

Representation = Literal["grid", "mesh", "particle"]
GridType = Literal["cartesian", "spherical"]
BCType = Literal["periodic", "dirichlet", "neumann", "wall", "open"]
# The loader routes `state` to prediction targets and every other role to conditioning inputs
Role = Literal["state", "forcing", "coefficient", "parameter", "boundary"]

REPRESENTATIONS: tuple[Representation, ...] = ("grid", "mesh", "particle")


# __ Schema objects _______


@dataclass
class FieldSpec:
    """Field-registry entry: the identity of a quantity.

    The identity is independent of how much the quantity varies in a given
    file; that extent lives on :class:`Field`.

    Attributes
    ----------
    tensor_order : int
        Tensor rank, a key of ``FIELD_GROUPS`` (0, 1 or 2).
    role : Role
        What the quantity is in the PDE.
    unit : str
        Physical unit.
    """

    tensor_order: int
    role: Role
    unit: str = "dimensionless"

    def as_json(self) -> dict[str, object]:
        """Encode this entry as a JSON-serializable mapping.

        Returns
        -------
        dict[str, object]
            ``tensor_order``, ``role`` and ``unit`` under the ``REG_*`` keys.
        """
        return {REG_TENSOR_ORDER: self.tensor_order, REG_ROLE: self.role, REG_UNIT: self.unit}


@dataclass
class Dimension:
    """A named coordinate axis: the time axis or a spatial axis like ``x``.

    Attributes
    ----------
    name : str
        Axis name; ``TIME_DIMENSION`` for the time axis, otherwise an entry of
        ``Dataset.spatial_dims``.
    coordinates : np.ndarray
        Coordinate values along the axis.
    """

    name: str
    coordinates: np.ndarray


@dataclass
class Field:
    """A physical quantity stored over (a subset of) trajectories, time, and space.

    ``tensor_order``, ``role`` and ``unit`` are the field's registry identity
    (see :class:`FieldSpec`); the three ``*_varying`` flags are its per-file
    extent, and an axis whose flag is ``False`` is dropped from the stored array.

    Attributes
    ----------
    name : str
        Field name, unique across all tensor ranks of a dataset.
    data : np.ndarray
        Stored array of shape
        ``(n_trajectories?, n_timesteps?, *spatial_shape?, *(n_spatial_dims,) * tensor_order)``.
    tensor_order : int
        Tensor rank; selects the HDF5 group (0 -> ``t0_fields``, ...) and the
        number of trailing axes of length ``n_spatial_dims``.
    role : Role
        What the quantity is in the PDE.
    unit : str
        Physical unit.
    dim_varying : bool
        Whether ``data`` has the spatial axes.
    time_varying : bool
        Whether ``data`` has the time axis.
    sample_varying : bool
        Whether ``data`` has the trajectory axis.
    symmetric : bool
        Whether the tensor is symmetric; only valid for ``tensor_order == 2``.
    antisymmetric : bool
        Whether the tensor is antisymmetric; only valid for ``tensor_order == 2``.
    """

    name: str
    data: np.ndarray
    tensor_order: int = 0
    role: Role = "state"
    unit: str = "dimensionless"
    dim_varying: bool = True
    time_varying: bool = True
    sample_varying: bool = True
    symmetric: bool = False
    antisymmetric: bool = False

    @property
    def group_name(self) -> str:
        """HDF5 group of this field's tensor order (e.g. ``"t1_fields"``)."""
        return FIELD_GROUPS[self.tensor_order]

    @property
    def spec(self) -> FieldSpec:
        """Registry identity of the field."""
        return FieldSpec(self.tensor_order, self.role, self.unit)

    def expected_ndim(self, n_spatial_dims: int) -> int:
        """Count the stored array axes implied by the extent flags and tensor order.

        Parameters
        ----------
        n_spatial_dims : int
            Spatial rank of the owning dataset.

        Returns
        -------
        int
            Number of axes ``data`` must have.
        """
        return (
            int(self.sample_varying)
            + int(self.time_varying)
            + (n_spatial_dims if self.dim_varying else 0)
            + self.tensor_order
        )

    def validate(self, n_spatial_dims: int) -> None:
        """Check the field's metadata and array shape against the dataset's spatial rank.

        Parameters
        ----------
        n_spatial_dims : int
            Spatial rank of the owning dataset.

        Raises
        ------
        ValueError
            If ``tensor_order`` is not a key of ``FIELD_GROUPS``; if both
            symmetry flags are set, or either is set with ``tensor_order != 2``;
            if ``data.ndim`` differs from :meth:`expected_ndim` (a global
            constant may also have shape ``(1,)``); or if a trailing tensor
            axis does not have length ``n_spatial_dims``.
        """
        if self.tensor_order not in FIELD_GROUPS:
            raise ValueError(
                f"field {self.name!r}: tensor_order {self.tensor_order} is not "
                f"one of {sorted(FIELD_GROUPS)}"
            )
        if self.symmetric and self.antisymmetric:
            raise ValueError(f"field {self.name!r}: cannot be both symmetric and antisymmetric")
        if (self.symmetric or self.antisymmetric) and self.tensor_order != 2:
            raise ValueError(
                f"field {self.name!r}: symmetric/antisymmetric only apply to tensor_order 2"
            )

        expected = self.expected_ndim(n_spatial_dims)
        actual = self.data.ndim
        # A global constant (0 expected axes) may also be stored with shape (1,)
        degenerate = expected == 0 and actual == 1 and self.data.shape == (1,)
        if actual != expected and not degenerate:
            raise ValueError(
                f"field {self.name!r}: expected {expected} array axes "
                f"(sample_varying={self.sample_varying}, time_varying={self.time_varying}, "
                f"dim_varying={self.dim_varying}, n_spatial_dims={n_spatial_dims}, "
                f"tensor_order={self.tensor_order}) but data has shape {self.data.shape}"
            )
        # The trailing `tensor_order` axes must each have length n_spatial_dims
        for axis in range(actual - self.tensor_order, actual):
            if self.data.shape[axis] != n_spatial_dims:
                raise ValueError(
                    f"field {self.name!r}: tensor axis {axis} has length "
                    f"{self.data.shape[axis]}, expected n_spatial_dims={n_spatial_dims}"
                )


@dataclass
class BoundaryCondition:
    """A boundary condition applied to one or more fields and spatial dimensions.

    Attributes
    ----------
    name : str
        Condition name; its HDF5 subgroup under ``boundary_conditions``.
    bc_type : BCType
        Kind of boundary condition.
    associated_fields : list[str]
        Fields the condition applies to; each must be present in the dataset.
    associated_dims : list[str]
        Spatial dimensions the condition applies to; each must be in
        ``Dataset.spatial_dims``.
    mask : np.ndarray or None
        Where ``values`` applies; ``None`` if not stored.
    values : np.ndarray or None
        Prescribed boundary data (e.g. inflow or flux profiles); ``None`` if
        not stored.
    """

    name: str
    bc_type: BCType
    associated_fields: list[str] = field(default_factory=list)
    associated_dims: list[str] = field(default_factory=list)
    mask: np.ndarray | None = None
    values: np.ndarray | None = None


@dataclass
class Normalization:
    """Per-field statistics shipped with the dataset, one value per tensor component.

    The loader applies these rather than recomputing them, and picks z-score
    (``mean``/``std``) or min-max (``min``/``max``) scaling explicitly.

    Attributes
    ----------
    mean : np.ndarray
        Per-component mean.
    std : np.ndarray
        Per-component standard deviation.
    min : np.ndarray or None
        Per-component minimum (schema >= 0.3.0); ``None`` if not shipped.
    max : np.ndarray or None
        Per-component maximum (schema >= 0.3.0); ``None`` if not shipped.
    """

    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray | None = None
    max: np.ndarray | None = None


@dataclass
class Dataset:
    """A complete dataset: root metadata plus dimensions, fields, BCs, and stats.

    Attributes
    ----------
    dataset_name : str
        Name of the dataset.
    representation : Representation
        Discretization of the spatial domain; one of ``REPRESENTATIONS``.
    n_spatial_dims : int
        Spatial rank, 0 to 3; 0 is the no-spatial-domain case (e.g. an ODE).
    n_trajectories : int
        Number of independent runs in the file.
    spatial_dims : list[str]
        Spatial axis names; ``n_spatial_dims`` entries.
    grid_type : GridType
        Coordinate system of the grid.
    schema_version : str
        Version of the file layout.
    registry_version : str
        Version of the field registry.
    dimensions : list[Dimension]
        Coordinate axes (time and spatial).
    fields : list[Field]
        Stored fields; names must be unique.
    boundary_conditions : list[BoundaryCondition]
        Boundary conditions over the stored fields.
    normalization : dict[str, Normalization]
        Statistics keyed by field name; every key must be a stored field.
    field_registry : dict[str, FieldSpec]
        Explicit registry; may be a superset of the stored fields (the shared
        cross-file vocabulary). Empty means derived from ``fields``, see
        :meth:`effective_registry`.
    provenance : dict[str, str]
        How the file was generated (resolved config, git revision,
        timestamp, ...); written verbatim and never interpreted.
    """

    dataset_name: str
    representation: Representation
    n_spatial_dims: int
    n_trajectories: int
    spatial_dims: list[str]
    grid_type: GridType = "cartesian"
    schema_version: str = SCHEMA_VERSION
    registry_version: str = REGISTRY_VERSION
    dimensions: list[Dimension] = field(default_factory=list)
    fields: list[Field] = field(default_factory=list)
    boundary_conditions: list[BoundaryCondition] = field(default_factory=list)
    normalization: dict[str, Normalization] = field(default_factory=dict)
    field_registry: dict[str, FieldSpec] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)

    @property
    def present_fields(self) -> list[str]:
        """Names of the fields actually stored in this file, in insertion order."""
        return [f.name for f in self.fields]

    @property
    def coords(self) -> dict[str, np.ndarray]:
        """Coordinate vector per dimension name (e.g. ``{"time": ..., "x": ...}``).

        Physical spacings follow directly, e.g. ``coords["x"][1] - coords["x"][0]``.
        """
        return {d.name: np.asarray(d.coordinates) for d in self.dimensions}

    def effective_registry(self) -> dict[str, FieldSpec]:
        """Build the registry written to disk.

        Returns
        -------
        dict[str, FieldSpec]
            The explicit ``field_registry`` entries, plus the spec of every
            stored field that has no explicit entry.
        """
        registry = dict(self.field_registry)
        for f in self.fields:
            registry.setdefault(f.name, f.spec)
        return registry

    def fields_by_order(self) -> dict[int, list[Field]]:
        """Group the fields by tensor order.

        Returns
        -------
        dict[int, list[Field]]
            Fields per tensor order, in insertion order; orders without
            fields are absent.
        """
        grouped: dict[int, list[Field]] = {}
        for f in self.fields:
            grouped.setdefault(f.tensor_order, []).append(f)
        return grouped

    def validate(self) -> None:
        """Check internal consistency, stopping at the first problem.

        Raises
        ------
        ValueError
            If ``representation`` is not in ``REPRESENTATIONS``;
            ``n_spatial_dims`` is outside 0 to 3 or disagrees with
            ``spatial_dims``; field names repeat; a field fails
            :meth:`Field.validate`, or is ``dim_varying`` with
            ``n_spatial_dims == 0``; a field disagrees with its registry entry
            on tensor order, role or unit; or a normalization or boundary
            condition references an unknown field or dimension.
        """
        if self.representation not in REPRESENTATIONS:
            raise ValueError(
                f"representation {self.representation!r} is not one of {list(REPRESENTATIONS)}"
            )
        # n_spatial_dims=0 is the no-spatial-domain case (e.g. an ODE); 3 is the rank ceiling
        if not 0 <= self.n_spatial_dims <= 3:
            raise ValueError(f"n_spatial_dims must be between 0 and 3, got {self.n_spatial_dims}")
        if len(self.spatial_dims) != self.n_spatial_dims:
            raise ValueError(
                f"spatial_dims {self.spatial_dims} has {len(self.spatial_dims)} entries "
                f"but n_spatial_dims={self.n_spatial_dims}"
            )

        names = self.present_fields
        if len(names) != len(set(names)):
            raise ValueError("field names must be unique across all tensor ranks")

        registry = self.effective_registry()
        for f in self.fields:
            f.validate(self.n_spatial_dims)
            # With no spatial axes, nothing can vary across space
            if self.n_spatial_dims == 0 and f.dim_varying:
                raise ValueError(
                    f"field {f.name!r}: dim_varying=True is invalid when n_spatial_dims=0 "
                    f"(there is no spatial axis to vary over)"
                )
            spec = registry[f.name]
            if (spec.tensor_order, spec.role, spec.unit) != (f.tensor_order, f.role, f.unit):
                raise ValueError(
                    f"field {f.name!r} disagrees with its registry entry: "
                    f"field is {(f.tensor_order, f.role, f.unit)}, "
                    f"registry is {(spec.tensor_order, spec.role, spec.unit)}"
                )

        known = set(names)
        for name in self.normalization:
            if name not in known:
                raise ValueError(f"normalization references unknown field {name!r}")
        for bc in self.boundary_conditions:
            unknown_dims = set(bc.associated_dims) - set(self.spatial_dims)
            if unknown_dims:
                raise ValueError(
                    f"boundary condition {bc.name!r} references unknown dims {sorted(unknown_dims)}"
                )
            unknown_fields = set(bc.associated_fields) - known
            if unknown_fields:
                raise ValueError(
                    f"boundary condition {bc.name!r} references unknown fields "
                    f"{sorted(unknown_fields)}"
                )


# __ Registry (de)serialization _______


def registry_to_json(registry: dict[str, FieldSpec]) -> str:
    """Encode a field registry as the JSON root attribute.

    Parameters
    ----------
    registry : dict[str, FieldSpec]
        Registry keyed by field name.

    Returns
    -------
    str
        JSON object mapping each field name to :meth:`FieldSpec.as_json`.
    """
    return json.dumps({name: spec.as_json() for name, spec in registry.items()})


def registry_from_json(text: str) -> dict[str, FieldSpec]:
    """Decode a field registry written by :func:`registry_to_json`.

    Parameters
    ----------
    text : str
        JSON object mapping field names to registry entries; ``unit`` may be
        missing and defaults to ``"dimensionless"``.

    Returns
    -------
    dict[str, FieldSpec]
        Registry keyed by field name.

    Raises
    ------
    json.JSONDecodeError
        If ``text`` is not valid JSON.
    KeyError
        If an entry lacks ``tensor_order`` or ``role``.
    """
    raw = json.loads(text)
    return {
        name: FieldSpec(
            tensor_order=int(spec[REG_TENSOR_ORDER]),
            role=spec[REG_ROLE],
            unit=spec.get(REG_UNIT, "dimensionless"),
        )
        for name, spec in raw.items()
    }
