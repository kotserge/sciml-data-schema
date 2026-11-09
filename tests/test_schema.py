"""Unit tests for the pure-Python schema objects (no HDF5)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import sciml_data_schema as sds
from sciml_data_schema.schema import registry_from_json, registry_to_json


def test_group_name_follows_tensor_order() -> None:
    """group_name is the t{order}_fields group of the tensor order."""
    assert sds.Field("a", np.zeros(()), tensor_order=0).group_name == "t0_fields"
    assert sds.Field("a", np.zeros((2,)), tensor_order=1).group_name == "t1_fields"
    assert sds.Field("a", np.zeros((2, 2)), tensor_order=2).group_name == "t2_fields"


def test_spec_is_registry_identity() -> None:
    """Field.spec carries tensor order, role and unit."""
    f = sds.Field("velocity", np.zeros((2,)), tensor_order=1, role="state", unit="m/s")
    assert f.spec == sds.FieldSpec(tensor_order=1, role="state", unit="m/s")


@pytest.mark.parametrize(
    ("sample", "time", "dim", "order", "expected"),
    [
        (True, True, True, 0, 4),  # (traj, t, x, y)
        (True, False, True, 0, 3),  # (traj, x, y)
        (True, False, False, 0, 1),  # (traj,)
        (False, False, False, 0, 0),  # scalar
        (True, True, True, 1, 5),  # (traj, t, x, y, D)
        (True, False, True, 2, 5),  # (traj, x, y, D, D)
    ],
)
def test_expected_ndim(sample: bool, time: bool, dim: bool, order: int, expected: int) -> None:
    """expected_ndim counts the axes kept by the extent flags plus the tensor axes."""
    f = sds.Field(
        "f",
        np.zeros(()),
        tensor_order=order,
        sample_varying=sample,
        time_varying=time,
        dim_varying=dim,
    )
    assert f.expected_ndim(n_spatial_dims=2) == expected


@pytest.mark.parametrize("data", [np.array(9.81), np.array([9.81])])
def test_global_constant_accepts_scalar_or_length1(data: np.ndarray) -> None:
    """A global constant validates as a 0-d scalar or with shape (1,)."""
    f = sds.Field(
        "gravity",
        data,
        tensor_order=0,
        role="parameter",
        dim_varying=False,
        time_varying=False,
        sample_varying=False,
    )
    f.validate(n_spatial_dims=2)  # does not raise


def test_global_constant_rejects_longer_axis() -> None:
    """A global constant with more than one value fails validation."""
    f = sds.Field(
        "g",
        np.array([1.0, 2.0]),
        dim_varying=False,
        time_varying=False,
        sample_varying=False,
    )
    with pytest.raises(ValueError, match="array axes"):
        f.validate(n_spatial_dims=2)


def test_registry_json_roundtrip() -> None:
    """registry_from_json inverts registry_to_json."""
    registry = {
        "u": sds.FieldSpec(tensor_order=0, role="state", unit="Pa"),
        "velocity": sds.FieldSpec(tensor_order=1, role="state", unit="m/s"),
        "re": sds.FieldSpec(tensor_order=0, role="parameter"),  # default unit
    }
    assert registry_from_json(registry_to_json(registry)) == registry


def test_effective_registry_merges_present_and_explicit(
    make_dataset: Callable[[], sds.Dataset],
) -> None:
    """The effective registry holds explicit entries and every present field."""
    ds = make_dataset()
    # An explicit entry for a field that is not present
    ds.field_registry = {"absent": sds.FieldSpec(tensor_order=0, role="parameter")}
    effective = ds.effective_registry()

    assert "absent" in effective  # explicit superset entry kept
    assert set(ds.present_fields) <= set(effective)  # present fields all derived in


def test_explicit_registry_takes_precedence_over_derived(
    make_dataset: Callable[[], sds.Dataset],
) -> None:
    """An explicit registry entry wins over the field's own spec."""
    ds = make_dataset()
    # Same identity as the field, but a different unit string
    ds.field_registry = {"reynolds": sds.FieldSpec(tensor_order=0, role="parameter", unit="custom")}
    assert ds.effective_registry()["reynolds"].unit == "custom"


def test_present_fields_preserves_insertion_order(make_dataset: Callable[[], sds.Dataset]) -> None:
    """present_fields lists the fields in insertion order."""
    ds = make_dataset()
    assert ds.present_fields == ["pressure", "velocity", "stress", "reynolds", "gravity"]


def test_controlled_vocabularies_are_exported() -> None:
    """The controlled vocabularies are importable from the package root."""
    import sciml_data_schema as sds

    for name in ("Role", "Representation", "GridType", "BCType"):
        assert name in sds.__all__
        assert getattr(sds, name) is getattr(sds.schema, name)
