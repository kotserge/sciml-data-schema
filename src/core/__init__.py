"""Shared dataset format and metadata schema (the neutral middle).

``core.schema`` defines the in-memory dataset representation; ``core.io`` reads
and writes it as HDF5. Downstream ``datagen`` and ``ml`` depend on this package
and exchange data exclusively through the ``core`` format.
"""

from .io import read_dataset, write_dataset
from .schema import (
    BoundaryCondition,
    Dataset,
    Dimension,
    Field,
    FieldSpec,
    Normalization,
)

__all__ = [
    "BoundaryCondition",
    "Dataset",
    "Dimension",
    "Field",
    "FieldSpec",
    "Normalization",
    "read_dataset",
    "write_dataset",
]
