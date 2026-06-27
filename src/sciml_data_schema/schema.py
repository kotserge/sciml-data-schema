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
``sample_varying``). Storage stays cheap: an axis along which a field does not
vary is dropped, so the stored shape is::

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

# --- Versions -----------------------------------------------------------------

SCHEMA_VERSION = "0.1.0"
# Versioned separately from the schema: the field registry (the canonical
# vocabulary) can evolve without a layout change, and vice versa.
REGISTRY_VERSION = "0.1.0"

# --- Canonical HDF5 names (single source of truth, shared with sciml_data_schema.io) -

GROUP_DIMENSIONS = "dimensions"
GROUP_BOUNDARY_CONDITIONS = "boundary_conditions"
GROUP_NORMALIZATION = "normalization"

# Field groups keyed by tensor order (rank).
FIELD_GROUPS: dict[int, str] = {0: "t0_fields", 1: "t1_fields", 2: "t2_fields"}

# Root-level attributes.
ATTR_SCHEMA_VERSION = "schema_version"
ATTR_REGISTRY_VERSION = "registry_version"
ATTR_DATASET_NAME = "dataset_name"
ATTR_REPRESENTATION = "representation"
ATTR_GRID_TYPE = "grid_type"
ATTR_N_SPATIAL_DIMS = "n_spatial_dims"
ATTR_N_TRAJECTORIES = "n_trajectories"
ATTR_FIELD_REGISTRY = "field_registry"
ATTR_PRESENT_FIELDS = "present_fields"
# Optional free-form provenance (JSON object of string values), e.g. the
# generator's resolved config, git revision, and creation time.
ATTR_PROVENANCE = "provenance"

# `spatial_dims` is an attribute of the `dimensions` group (the spatial axis
# names; their coordinates are profile-specific datasets in the same group).
ATTR_SPATIAL_DIMS = "spatial_dims"
# The time coordinate lives in the dimensions group under this reserved name.
TIME_DIMENSION = "time"

# Per-field attributes (extent flags + tensor symmetry).
ATTR_DIM_VARYING = "dim_varying"
ATTR_TIME_VARYING = "time_varying"
ATTR_SAMPLE_VARYING = "sample_varying"
ATTR_SYMMETRIC = "symmetric"
ATTR_ANTISYMMETRIC = "antisymmetric"

# Boundary-condition attributes.
ATTR_BC_TYPE = "bc_type"
ATTR_ASSOCIATED_FIELDS = "associated_fields"
ATTR_ASSOCIATED_DIMS = "associated_dims"

# Normalization attributes.
ATTR_MEAN = "mean"
ATTR_STD = "std"

# Field-registry JSON keys.
REG_TENSOR_ORDER = "tensor_order"
REG_ROLE = "role"
REG_UNIT = "unit"

# --- Controlled vocabularies --------------------------------------------------

Representation = Literal["grid", "mesh", "particle"]
GridType = Literal["cartesian", "spherical"]
BCType = Literal["periodic", "dirichlet", "neumann", "wall", "open"]
# The bridge between PDE meaning and storage; the loader routes `state` to
# prediction targets and everything else to conditioning inputs.
Role = Literal["state", "forcing", "coefficient", "parameter", "boundary"]

REPRESENTATIONS: tuple[Representation, ...] = ("grid", "mesh", "particle")


# --- Schema objects -----------------------------------------------------------


@dataclass
class FieldSpec:
    """A field-registry entry: the *identity* of a quantity, independent of how
    much it varies in a given file.
    """

    tensor_order: int
    role: Role
    unit: str = "dimensionless"

    def as_json(self) -> dict[str, object]:
        return {REG_TENSOR_ORDER: self.tensor_order, REG_ROLE: self.role, REG_UNIT: self.unit}


@dataclass
class Dimension:
    """A named coordinate axis: the time axis or a spatial axis like ``x``."""

    name: str
    coordinates: np.ndarray


@dataclass
class Field:
    """A physical quantity stored over (a subset of) trajectories, time, and space.

    ``tensor_order`` selects the HDF5 group (0 -> ``t0_fields``, ...) and fixes
    the number of trailing axes of length ``n_spatial_dims``. ``role`` and
    ``unit`` are the field's registry identity (see :class:`FieldSpec`); the
    three ``*_varying`` flags are its per-file *extent*, and an axis whose flag
    is ``False`` is dropped from the stored array.
    """

    name: str
    data: np.ndarray
    tensor_order: int = 0
    role: Role = "state"
    unit: str = "dimensionless"
    dim_varying: bool = True
    time_varying: bool = True
    sample_varying: bool = True
    # Only meaningful for ``tensor_order == 2``.
    symmetric: bool = False
    antisymmetric: bool = False

    @property
    def group_name(self) -> str:
        return FIELD_GROUPS[self.tensor_order]

    @property
    def spec(self) -> FieldSpec:
        """This field's registry identity."""
        return FieldSpec(self.tensor_order, self.role, self.unit)

    def expected_ndim(self, n_spatial_dims: int) -> int:
        """Number of stored array axes implied by the extent flags and tensor order."""
        return (
            int(self.sample_varying)
            + int(self.time_varying)
            + (n_spatial_dims if self.dim_varying else 0)
            + self.tensor_order
        )

    def validate(self, n_spatial_dims: int) -> None:
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
        # A field that varies along no axis (a true global constant) may be
        # stored as a 0-d scalar or as a single length-1 axis (1,) per SCHEMA.md.
        degenerate = expected == 0 and actual == 1 and self.data.shape == (1,)
        if actual != expected and not degenerate:
            raise ValueError(
                f"field {self.name!r}: expected {expected} array axes "
                f"(sample_varying={self.sample_varying}, time_varying={self.time_varying}, "
                f"dim_varying={self.dim_varying}, n_spatial_dims={n_spatial_dims}, "
                f"tensor_order={self.tensor_order}) but data has shape {self.data.shape}"
            )
        # The trailing `tensor_order` axes must each have length n_spatial_dims.
        for axis in range(actual - self.tensor_order, actual):
            if self.data.shape[axis] != n_spatial_dims:
                raise ValueError(
                    f"field {self.name!r}: tensor axis {axis} has length "
                    f"{self.data.shape[axis]}, expected n_spatial_dims={n_spatial_dims}"
                )


@dataclass
class BoundaryCondition:
    """A boundary condition applied to one or more fields/spatial dimensions.

    ``values`` (with an optional ``mask`` of where it applies) carries any
    prescribed boundary forcing -- the data home for inflow/flux profiles.
    """

    name: str
    bc_type: BCType
    associated_fields: list[str] = field(default_factory=list)
    associated_dims: list[str] = field(default_factory=list)
    mask: np.ndarray | None = None
    values: np.ndarray | None = None


@dataclass
class Normalization:
    """Shipped per-field statistics (per tensor component). The loader applies
    these rather than recomputing them.
    """

    mean: np.ndarray
    std: np.ndarray


@dataclass
class Dataset:
    """A complete dataset: root metadata plus dimensions, fields, BCs, and stats."""

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
    # Optional explicit registry. May be a *superset* of the present fields (the
    # shared cross-file vocabulary); if left empty it is derived from `fields`.
    field_registry: dict[str, FieldSpec] = field(default_factory=dict)
    # Optional free-form provenance: how this file was generated (resolved config,
    # git revision, timestamp, ...). String values; written verbatim, never
    # interpreted by the schema. Empty by default.
    provenance: dict[str, str] = field(default_factory=dict)

    @property
    def present_fields(self) -> list[str]:
        """Names of the fields actually stored in this file, in insertion order."""
        return [f.name for f in self.fields]

    def effective_registry(self) -> dict[str, FieldSpec]:
        """The registry written to disk: explicit entries, plus every present
        field (whose spec fills any gap). Explicit entries take precedence.
        """
        registry = dict(self.field_registry)
        for f in self.fields:
            registry.setdefault(f.name, f.spec)
        return registry

    def fields_by_order(self) -> dict[int, list[Field]]:
        """Group fields by tensor order, preserving insertion order within a rank."""
        grouped: dict[int, list[Field]] = {}
        for f in self.fields:
            grouped.setdefault(f.tensor_order, []).append(f)
        return grouped

    def validate(self) -> None:
        """Check internal consistency; raise :class:`ValueError` on the first problem."""
        if self.representation not in REPRESENTATIONS:
            raise ValueError(
                f"representation {self.representation!r} is not one of {list(REPRESENTATIONS)}"
            )
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


# --- Registry (de)serialization ----------------------------------------------
# The registry travels as a single JSON root attribute; keep its encoding next
# to the schema so sciml_data_schema.io stays a thin HDF5 shim.


def registry_to_json(registry: dict[str, FieldSpec]) -> str:
    return json.dumps({name: spec.as_json() for name, spec in registry.items()})


def registry_from_json(text: str) -> dict[str, FieldSpec]:
    raw = json.loads(text)
    return {
        name: FieldSpec(
            tensor_order=int(spec[REG_TENSOR_ORDER]),
            role=spec[REG_ROLE],
            unit=spec.get(REG_UNIT, "dimensionless"),
        )
        for name, spec in raw.items()
    }
