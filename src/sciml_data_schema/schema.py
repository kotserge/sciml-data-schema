"""The in-memory dataset schema and the canonical HDF5 names.

A :class:`Dataset` is one dataset of ``n_trajectories`` independent runs over a shared
spatial domain: the root metadata, the coordinate axes (:class:`Dimension`), the fields
(:class:`Field`), the boundary conditions (:class:`BoundaryCondition`) and the per-field
statistics (:class:`Normalization`). Every physical quantity -- solution, forcing,
coefficient, parameter -- is a :class:`Field`; a spatially uniform scalar parameter is a
rank-0 field whose spatial and time flags are off.

A field carries two kinds of metadata. Its *identity* (:class:`FieldSpec`: tensor order,
role, unit) is fixed by the **field registry**, a name-keyed table shared by convention
across the files of a representation; the tensor order also selects the ``t{order}_fields``
group and the number of trailing tensor axes. Its *extent* (the ``dim_varying``,
``time_varying`` and ``sample_varying`` flags) names the axes its array carries; an axis
whose flag is ``False`` is absent, so the stored shape is::

    (n_trajectories?, n_timesteps?, *spatial_shape?, *(n_spatial_dims,) * tensor_order)

with each leading axis present only when its flag is ``True``. A rank-0 field with every
flag ``False`` is a 0-d scalar or a single length-1 axis ``(1,)``.

:meth:`Dataset.validate` checks a dataset against these rules. The ``GROUP_*``, ``ATTR_*``
and ``REG_*`` constants are the HDF5 names :mod:`sciml_data_schema.io` writes and reads;
the format is specified in ``docs/SCHEMA.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# === Versions ===

# The on-disk layout version (root attribute `schema_version`)
SCHEMA_VERSION = "0.3.0"
# The field-registry vocabulary version (root attribute `registry_version`); independent of the
# layout version
REGISTRY_VERSION = "0.1.0"

# === Canonical HDF5 names (shared with sciml_data_schema.io) ===

GROUP_DIMENSIONS = "dimensions"
GROUP_BOUNDARY_CONDITIONS = "boundary_conditions"
GROUP_NORMALIZATION = "normalization"

# The field groups, keyed by tensor order
FIELD_GROUPS: dict[int, str] = {0: "t0_fields", 1: "t1_fields", 2: "t2_fields"}

# Root attributes
ATTR_SCHEMA_VERSION = "schema_version"
ATTR_REGISTRY_VERSION = "registry_version"
ATTR_DATASET_NAME = "dataset_name"
ATTR_REPRESENTATION = "representation"
ATTR_GRID_TYPE = "grid_type"
ATTR_N_SPATIAL_DIMS = "n_spatial_dims"
ATTR_N_TRAJECTORIES = "n_trajectories"
ATTR_FIELD_REGISTRY = "field_registry"
ATTR_PRESENT_FIELDS = "present_fields"
# Free-form provenance, a JSON object of string values (the generator's resolved config, git
# revision, creation time, ...); absent when the dataset carries none
ATTR_PROVENANCE = "provenance"

# Attribute of the `dimensions` group: the spatial axis names, whose coordinate vectors are
# datasets of the same group
ATTR_SPATIAL_DIMS = "spatial_dims"
# The name of the time coordinate dataset in the `dimensions` group
TIME_DIMENSION = "time"

# Per-field attributes: the extent flags and, on rank-2 fields, the symmetry flags
ATTR_DIM_VARYING = "dim_varying"
ATTR_TIME_VARYING = "time_varying"
ATTR_SAMPLE_VARYING = "sample_varying"
ATTR_SYMMETRIC = "symmetric"
ATTR_ANTISYMMETRIC = "antisymmetric"

# Boundary-condition attributes
ATTR_BC_TYPE = "bc_type"
ATTR_ASSOCIATED_FIELDS = "associated_fields"
ATTR_ASSOCIATED_DIMS = "associated_dims"

# Normalization attributes: mean and std always, min and max optional (schema >= 0.3.0)
ATTR_MEAN = "mean"
ATTR_STD = "std"
ATTR_MIN = "min"
ATTR_MAX = "max"

# Field-registry JSON keys
REG_TENSOR_ORDER = "tensor_order"
REG_ROLE = "role"
REG_UNIT = "unit"

# === Controlled vocabularies ===

Representation = Literal["grid", "mesh", "particle"]
GridType = Literal["cartesian", "spherical"]
BCType = Literal["periodic", "dirichlet", "neumann", "wall", "open"]
# A field's role in the PDE; a loader routes `state` fields to the prediction targets and every
# other role to the conditioning inputs
Role = Literal["state", "forcing", "coefficient", "parameter", "boundary"]

REPRESENTATIONS: tuple[Representation, ...] = ("grid", "mesh", "particle")


# === Schema objects ===


@dataclass
class FieldSpec:
    """A field-registry entry: the identity of a quantity, independent of its extent in a file.

    Attributes
    ----------
    tensor_order : int
        0 (scalar), 1 (vector) or 2 (rank-2 tensor).
    role : Role
        The quantity's role in the PDE.
    unit : str
        The physical unit, e.g. ``"m/s"``; ``"dimensionless"`` for a pure number.
    """

    tensor_order: int
    role: Role
    unit: str = "dimensionless"

    def as_json(self) -> dict[str, object]:
        """The entry as the JSON object stored under its name in the ``field_registry``.

        Returns
        -------
        dict[str, object]
            ``{"tensor_order": ..., "role": ..., "unit": ...}``.
        """
        return {REG_TENSOR_ORDER: self.tensor_order, REG_ROLE: self.role, REG_UNIT: self.unit}


@dataclass
class Dimension:
    """A named coordinate axis: the time axis or a spatial axis such as ``x``.

    Attributes
    ----------
    name : str
        The axis name: ``"time"`` for the time axis, else one of the dataset's ``spatial_dims``.
    coordinates : np.ndarray
        ``(n,)``, the coordinate of each point along the axis.
    """

    name: str
    coordinates: np.ndarray


@dataclass
class Field:
    """A physical quantity stored over the trajectory, time and spatial axes it varies along.

    Attributes
    ----------
    name : str
        The field name: its key in the field registry and its dataset name on disk.
    data : np.ndarray
        The values, in the stored shape of the module docstring.
    tensor_order : int
        0, 1 or 2; selects the ``t{order}_fields`` group and the number of trailing axes of
        length ``n_spatial_dims``.
    role : Role
        The quantity's role in the PDE.
    unit : str
        The physical unit.
    dim_varying : bool
        Whether the array carries the spatial axes.
    time_varying : bool
        Whether the array carries the time axis.
    sample_varying : bool
        Whether the array carries the leading trajectory axis.
    symmetric : bool
        Whether a rank-2 field is symmetric; rank 2 only.
    antisymmetric : bool
        Whether a rank-2 field is antisymmetric; rank 2 only.
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
        """The HDF5 group of this field's tensor order (``t0_fields``, ...)."""
        return FIELD_GROUPS[self.tensor_order]

    @property
    def spec(self) -> FieldSpec:
        """The field's registry identity: tensor order, role and unit."""
        return FieldSpec(self.tensor_order, self.role, self.unit)

    def expected_ndim(self, n_spatial_dims: int) -> int:
        """The number of array axes the extent flags and the tensor order imply.

        Parameters
        ----------
        n_spatial_dims : int
            The dataset's number of spatial dimensions.

        Returns
        -------
        int
            One axis per set leading flag (``sample_varying``, ``time_varying``),
            ``n_spatial_dims`` axes when ``dim_varying``, plus ``tensor_order`` trailing axes.
        """
        return (
            int(self.sample_varying)
            + int(self.time_varying)
            + (n_spatial_dims if self.dim_varying else 0)
            + self.tensor_order
        )

    def validate(self, n_spatial_dims: int) -> None:
        """Check the field's flags against its data shape.

        Parameters
        ----------
        n_spatial_dims : int
            The dataset's number of spatial dimensions.

        Raises
        ------
        ValueError
            ``tensor_order`` is not 0, 1 or 2; both symmetry flags are set; a symmetry flag is
            set on a field of rank other than 2; ``data.ndim`` differs from
            :meth:`expected_ndim` (a rank-0 field with every flag off may also be ``(1,)``);
            or a trailing tensor axis has a length other than ``n_spatial_dims``.
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

        # The axis count, with the (1,) spelling of a global constant accepted
        expected = self.expected_ndim(n_spatial_dims)
        actual = self.data.ndim
        degenerate = expected == 0 and actual == 1 and self.data.shape == (1,)
        if actual != expected and not degenerate:
            raise ValueError(
                f"field {self.name!r}: expected {expected} array axes "
                f"(sample_varying={self.sample_varying}, time_varying={self.time_varying}, "
                f"dim_varying={self.dim_varying}, n_spatial_dims={n_spatial_dims}, "
                f"tensor_order={self.tensor_order}) but data has shape {self.data.shape}"
            )

        # Each of the trailing `tensor_order` axes has length n_spatial_dims
        for axis in range(actual - self.tensor_order, actual):
            if self.data.shape[axis] != n_spatial_dims:
                raise ValueError(
                    f"field {self.name!r}: tensor axis {axis} has length "
                    f"{self.data.shape[axis]}, expected n_spatial_dims={n_spatial_dims}"
                )


@dataclass
class BoundaryCondition:
    """A boundary condition on one or more fields along one or more spatial dimensions.

    Attributes
    ----------
    name : str
        The condition's name: its group name under ``boundary_conditions``.
    bc_type : BCType
        The kind of condition.
    associated_fields : list[str]
        The fields it applies to; each a present field.
    associated_dims : list[str]
        The spatial dimensions it applies along; each one of the dataset's ``spatial_dims``.
    mask : np.ndarray | None
        Where the condition applies (a boolean array over the domain), or ``None``.
    values : np.ndarray | None
        The prescribed values (the boundary forcing, e.g. an inflow or flux profile), or
        ``None``.
    """

    name: str
    bc_type: BCType
    associated_fields: list[str] = field(default_factory=list)
    associated_dims: list[str] = field(default_factory=list)
    mask: np.ndarray | None = None
    values: np.ndarray | None = None


@dataclass
class Normalization:
    """Shipped statistics of one field, per tensor component or as one scalar.

    Attributes
    ----------
    mean : np.ndarray
        The mean (z-score normalization).
    std : np.ndarray
        The standard deviation (z-score normalization).
    min : np.ndarray | None
        The minimum (min-max scaling); optional, schema >= 0.3.0.
    max : np.ndarray | None
        The maximum (min-max scaling); optional, schema >= 0.3.0.
    """

    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray | None = None
    max: np.ndarray | None = None


@dataclass
class Dataset:
    """One dataset: root metadata, dimensions, fields, boundary conditions and statistics.

    Attributes
    ----------
    dataset_name : str
        The dataset's name.
    representation : Representation
        The spatial representation (the profile): ``"grid"``, ``"mesh"`` or ``"particle"``.
    n_spatial_dims : int
        The number of spatial dimensions, 0 to 3; 0 is a time axis only (an ODE).
    n_trajectories : int
        The number of independent runs in the file.
    spatial_dims : list[str]
        The spatial axis names, ``n_spatial_dims`` of them.
    grid_type : GridType
        The grid profile's coordinate system.
    schema_version : str
        The layout version the dataset follows.
    registry_version : str
        The field-registry vocabulary version.
    dimensions : list[Dimension]
        The coordinate axes: the spatial axes and, for a time-dependent dataset, ``"time"``.
    fields : list[Field]
        The fields, in storage order; names are unique across tensor orders.
    boundary_conditions : list[BoundaryCondition]
        The boundary conditions.
    normalization : dict[str, Normalization]
        The statistics, keyed by field name; every key is a present field.
    field_registry : dict[str, FieldSpec]
        The explicit registry entries, possibly a superset of the present fields (the shared
        cross-file vocabulary); the entries of present fields missing here are derived from
        the fields by :meth:`effective_registry`.
    provenance : dict[str, str]
        Free-form string values recording how the dataset was produced (resolved config, git
        revision, creation time, ...); written verbatim, never interpreted.
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
        """The names of the fields in the dataset, in storage order."""
        return [f.name for f in self.fields]

    @property
    def coords(self) -> dict[str, np.ndarray]:
        """The coordinate array of each dimension, keyed by name (``"time"``, ``"x"``, ...)."""
        return {d.name: np.asarray(d.coordinates) for d in self.dimensions}

    def effective_registry(self) -> dict[str, FieldSpec]:
        """The registry written to disk: the explicit entries plus every present field's spec.

        Returns
        -------
        dict[str, FieldSpec]
            ``field_registry`` with an entry for each present field missing from it, taken from
            the field; an explicit entry is kept over the field's own spec.
        """
        registry = dict(self.field_registry)
        for f in self.fields:
            registry.setdefault(f.name, f.spec)
        return registry

    def fields_by_order(self) -> dict[int, list[Field]]:
        """The fields grouped by tensor order.

        Returns
        -------
        dict[int, list[Field]]
            Tensor order -> its fields in storage order; orders without a field are absent.
        """
        grouped: dict[int, list[Field]] = {}
        for f in self.fields:
            grouped.setdefault(f.tensor_order, []).append(f)
        return grouped

    def validate(self) -> None:
        """Check the dataset's internal consistency, stopping at the first problem.

        Raises
        ------
        ValueError
            ``representation`` is not a known one; ``n_spatial_dims`` is outside 0 to 3;
            ``spatial_dims`` does not have ``n_spatial_dims`` entries; a field name repeats;
            a field fails :meth:`Field.validate`; a field is ``dim_varying`` with
            ``n_spatial_dims == 0``; a field's identity differs from its registry entry; a
            normalization key or a boundary condition names a field or a dimension the
            dataset does not have.
        """
        if self.representation not in REPRESENTATIONS:
            raise ValueError(
                f"representation {self.representation!r} is not one of {list(REPRESENTATIONS)}"
            )
        # 0 spatial dims is a time axis only (an ODE); 3 is the ceiling
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

        # Each field against its own flags and against its registry entry
        registry = self.effective_registry()
        for f in self.fields:
            f.validate(self.n_spatial_dims)
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

        # Every name the statistics and the boundary conditions refer to exists
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


# === Registry (de)serialization ===
# The registry travels as one JSON root attribute; these two functions are its encoding.


def registry_to_json(registry: dict[str, FieldSpec]) -> str:
    """Encode a registry as the JSON text of the ``field_registry`` attribute.

    Parameters
    ----------
    registry : dict[str, FieldSpec]
        Field name -> its registry entry.

    Returns
    -------
    str
        A JSON object keyed by field name, each value a :meth:`FieldSpec.as_json` object.
    """
    return json.dumps({name: spec.as_json() for name, spec in registry.items()})


def registry_from_json(text: str) -> dict[str, FieldSpec]:
    """Decode the ``field_registry`` attribute's JSON text.

    Parameters
    ----------
    text : str
        The JSON object written by :func:`registry_to_json`.

    Returns
    -------
    dict[str, FieldSpec]
        Field name -> its registry entry; an entry without a ``unit`` reads as
        ``"dimensionless"``.
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
