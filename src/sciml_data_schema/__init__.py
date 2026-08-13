"""HDF5 dataset format and metadata schema for PDE / physics simulation data.

:mod:`sciml_data_schema.schema` defines the in-memory dataset representation
(:class:`~sciml_data_schema.schema.Dataset` and its parts);
:mod:`sciml_data_schema.io` reads and writes it as HDF5. Producers and consumers
of simulation data exchange it exclusively through this format.

The design is documented in ``SCHEMA.md``. In brief: a single file holds one
dataset of ``n_trajectories`` independent runs over a shared spatial domain, and
every physical quantity -- solution, forcing, coefficient, parameter -- is a
:class:`~sciml_data_schema.schema.Field` tagged with a tensor order, a role, and
per-file extent flags.
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
    # The controlled vocabularies (§1). Exported because they appear in the public
    # signatures above -- `Field.role`, `Dataset.representation`/`grid_type`,
    # `BoundaryCondition.bc_type` -- so a producer annotating a helper that builds one
    # needs the type, not just the literal.
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
