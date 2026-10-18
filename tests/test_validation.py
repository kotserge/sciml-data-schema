"""Validation tests: each malformed dataset raises ValueError naming the problem."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import sciml_data_schema as sds


def _field(dataset: sds.Dataset, name: str) -> sds.Field:
    """The field ``name`` of ``dataset``.

    Parameters
    ----------
    dataset : sds.Dataset
        The dataset to search.
    name : str
        The field name.

    Returns
    -------
    sds.Field
        The field object itself (not a copy), so a test may mutate it.
    """
    return next(f for f in dataset.fields if f.name == name)


def test_valid_dataset_passes(make_dataset: Callable[[], sds.Dataset]) -> None:
    make_dataset().validate()  # does not raise


def test_bad_representation(make_dataset: Callable[[], sds.Dataset]) -> None:
    ds = make_dataset()
    ds.representation = "blob"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="representation"):
        ds.validate()


def test_spatial_dims_length_mismatch(make_dataset: Callable[[], sds.Dataset]) -> None:
    ds = make_dataset()
    ds.spatial_dims = ["x"]  # n_spatial_dims is 2
    with pytest.raises(ValueError, match="spatial_dims"):
        ds.validate()


def test_duplicate_field_names(make_dataset: Callable[[], sds.Dataset]) -> None:
    ds = make_dataset()
    ds.fields.append(_field(ds, "reynolds"))
    with pytest.raises(ValueError, match="unique"):
        ds.validate()


def test_normalization_unknown_field(make_dataset: Callable[[], sds.Dataset]) -> None:
    ds = make_dataset()
    ds.normalization["nonexistent"] = sds.Normalization(np.asarray(0.0), np.asarray(1.0))
    with pytest.raises(ValueError, match="normalization references unknown field"):
        ds.validate()


def test_bc_unknown_dim(make_dataset: Callable[[], sds.Dataset]) -> None:
    ds = make_dataset()
    ds.boundary_conditions[0].associated_dims = ["z"]
    with pytest.raises(ValueError, match="unknown dims"):
        ds.validate()


def test_bc_unknown_field(make_dataset: Callable[[], sds.Dataset]) -> None:
    ds = make_dataset()
    ds.boundary_conditions[0].associated_fields = ["ghost"]
    with pytest.raises(ValueError, match="unknown fields"):
        ds.validate()


def test_field_disagrees_with_explicit_registry(make_dataset: Callable[[], sds.Dataset]) -> None:
    ds = make_dataset()
    # pressure is tensor_order 0; the registry claims 1
    ds.field_registry = {"pressure": sds.FieldSpec(tensor_order=1, role="state", unit="Pa")}
    with pytest.raises(ValueError, match="disagrees with its registry entry"):
        ds.validate()


def test_tensor_order_out_of_range(make_dataset: Callable[[], sds.Dataset]) -> None:
    ds = make_dataset()
    _field(ds, "pressure").tensor_order = 3
    with pytest.raises(ValueError, match="tensor_order"):
        ds.validate()


def test_symmetric_and_antisymmetric(make_dataset: Callable[[], sds.Dataset]) -> None:
    ds = make_dataset()
    stress = _field(ds, "stress")
    stress.antisymmetric = True  # already symmetric=True
    with pytest.raises(ValueError, match="both symmetric and antisymmetric"):
        ds.validate()


def test_symmetry_requires_tensor_order_2(make_dataset: Callable[[], sds.Dataset]) -> None:
    ds = make_dataset()
    _field(ds, "pressure").symmetric = True  # pressure is tensor_order 0
    with pytest.raises(ValueError, match="tensor_order 2"):
        ds.validate()


def test_ndim_mismatch(make_dataset: Callable[[], sds.Dataset]) -> None:
    ds = make_dataset()
    pressure = _field(ds, "pressure")
    pressure.data = pressure.data[0]  # the trajectory axis dropped: one axis short
    with pytest.raises(ValueError, match="array axes"):
        ds.validate()


def test_n_spatial_dims_out_of_range(make_dataset: Callable[[], sds.Dataset]) -> None:
    ds = make_dataset()
    ds.n_spatial_dims = 4  # above the ceiling of 3
    with pytest.raises(ValueError, match="between 0 and 3"):
        ds.validate()


def test_zero_spatial_dims_rejects_dim_varying() -> None:
    """With no spatial axis, a field cannot be dim_varying."""
    u = np.zeros((2, 5), dtype=np.float32)
    ds = sds.Dataset(
        dataset_name="ode",
        representation="grid",
        n_spatial_dims=0,
        n_trajectories=2,
        spatial_dims=[],
        dimensions=[sds.Dimension("time", np.arange(5, dtype=np.float32))],
        fields=[sds.Field("u", u, role="state", dim_varying=True, time_varying=True)],
    )
    with pytest.raises(ValueError, match="dim_varying"):
        ds.validate()


def test_tensor_axis_wrong_length(make_dataset: Callable[[], sds.Dataset]) -> None:
    ds = make_dataset()
    # A vector field whose component axis has length 3 in a 2D domain
    bad = sds.Field(
        name="flux",
        data=np.zeros((3, 4, 5, 6, 3), dtype=np.float32),
        tensor_order=1,
        role="forcing",
    )
    ds.fields.append(bad)
    with pytest.raises(ValueError, match="tensor axis"):
        ds.validate()
