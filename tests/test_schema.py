"""Unit tests of the pure-Python schema objects (no HDF5)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import sciml_data_schema as sds
from sciml_data_schema.schema import registry_from_json, registry_to_json


def test_group_name_follows_tensor_order() -> None:
    assert sds.Field("a", np.zeros(()), tensor_order=0).group_name == "t0_fields"
    assert sds.Field("a", np.zeros((2,)), tensor_order=1).group_name == "t1_fields"
    assert sds.Field("a", np.zeros((2, 2)), tensor_order=2).group_name == "t2_fields"


def test_spec_is_registry_identity() -> None:
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
    registry = {
        "u": sds.FieldSpec(tensor_order=0, role="state", unit="Pa"),
        "velocity": sds.FieldSpec(tensor_order=1, role="state", unit="m/s"),
        "re": sds.FieldSpec(tensor_order=0, role="parameter"),  # default unit
    }
    assert registry_from_json(registry_to_json(registry)) == registry


def test_effective_registry_merges_present_and_explicit(
    make_dataset: Callable[[], sds.Dataset],
) -> None:
    ds = make_dataset()
    # An explicit entry that is not a present field
    ds.field_registry = {"absent": sds.FieldSpec(tensor_order=0, role="parameter")}
    effective = ds.effective_registry()

    assert "absent" in effective  # the explicit superset entry is kept
    assert set(ds.present_fields) <= set(effective)  # every present field is derived in


def test_explicit_registry_takes_precedence_over_derived(
    make_dataset: Callable[[], sds.Dataset],
) -> None:
    ds = make_dataset()
    # The field's identity with another unit string; the explicit entry wins
    ds.field_registry = {"reynolds": sds.FieldSpec(tensor_order=0, role="parameter", unit="custom")}
    assert ds.effective_registry()["reynolds"].unit == "custom"


def test_present_fields_preserves_insertion_order(
    make_dataset: Callable[[], sds.Dataset],
) -> None:
    ds = make_dataset()
    assert ds.present_fields == ["pressure", "velocity", "stress", "reynolds", "gravity"]


def test_controlled_vocabularies_are_exported() -> None:
    """The four vocabularies are exported from the package root, the same objects as in schema."""
    for name in ("Role", "Representation", "GridType", "BCType"):
        assert name in sds.__all__
        assert getattr(sds, name) is getattr(sds.schema, name)
