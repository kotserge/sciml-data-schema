"""An HDF5 dataset format and metadata schema for PDE / physics simulation data.

One file holds one dataset of ``n_trajectories`` independent runs over a shared spatial
domain. Every physical quantity -- solution, forcing, coefficient, parameter -- is a
:class:`Field` with a tensor order, a role and a unit, and three extent flags naming the
axes its array carries. :mod:`sciml_data_schema.schema` defines the in-memory objects
(:class:`Dataset` and its parts) and the canonical HDF5 names;
:mod:`sciml_data_schema.io` writes and reads them (:func:`write_dataset`,
:class:`DatasetWriter`, :func:`read_dataset`); :mod:`sciml_data_schema.provenance` holds
the helpers that stamp how a dataset was produced. The format is specified in
``docs/SCHEMA.md``.

Everything a producer or a consumer needs is importable from this package root.
"""

from .io import Compression, DatasetWriter, field_placeholder, read_dataset, write_dataset
from .provenance import git_revision
from .schema import (
    REGISTRY_VERSION,
    SCHEMA_VERSION,
    BCType,
    BoundaryCondition,
    Dataset,
    Dimension,
    Field,
    FieldSpec,
    GridType,
    Normalization,
    Representation,
    Role,
)

__version__ = "0.1.0"

__all__ = [
    "REGISTRY_VERSION",
    "SCHEMA_VERSION",
    # The controlled vocabularies: the types of `Field.role`, `Dataset.representation`,
    # `Dataset.grid_type` and `BoundaryCondition.bc_type`
    "BCType",
    "BoundaryCondition",
    "Dataset",
    "Dimension",
    "Field",
    "FieldSpec",
    "GridType",
    "Normalization",
    "Representation",
    "Role",
    "Compression",
    "DatasetWriter",
    "__version__",
    "field_placeholder",
    "git_revision",
    "read_dataset",
    "write_dataset",
]
