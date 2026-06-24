"""Round-trip tests: a dataset written to HDF5 reads back identically."""

from __future__ import annotations

import numpy as np
import pytest

import sciml_data_schema as sds


@pytest.mark.parametrize("compression", [None, "gzip"])
def test_roundtrip_preserves_everything(make_dataset, tmp_path, compression):
    original = make_dataset()
    path = sds.write_dataset(original, tmp_path / "data.h5", compression=compression)
    assert path.exists()

    loaded = sds.read_dataset(path)

    # Root metadata.
    assert loaded.dataset_name == original.dataset_name
    assert loaded.representation == original.representation
    assert loaded.grid_type == original.grid_type
    assert loaded.n_spatial_dims == original.n_spatial_dims
    assert loaded.n_trajectories == original.n_trajectories
    assert loaded.spatial_dims == original.spatial_dims
    assert loaded.schema_version == original.schema_version
    assert loaded.registry_version == original.registry_version
    assert loaded.present_fields == original.present_fields


def test_roundtrip_fields(make_dataset, tmp_path):
    original = make_dataset()
    loaded = sds.read_dataset(sds.write_dataset(original, tmp_path / "data.h5"))

    by_name = {f.name: f for f in loaded.fields}
    for orig in original.fields:
        got = by_name[orig.name]
        np.testing.assert_array_equal(got.data, orig.data)
        assert got.tensor_order == orig.tensor_order
        assert got.role == orig.role
        assert got.unit == orig.unit
        assert got.dim_varying == orig.dim_varying
        assert got.time_varying == orig.time_varying
        assert got.sample_varying == orig.sample_varying
        assert got.symmetric == orig.symmetric
        assert got.antisymmetric == orig.antisymmetric


def test_roundtrip_dimensions(make_dataset, tmp_path):
    original = make_dataset()
    loaded = sds.read_dataset(sds.write_dataset(original, tmp_path / "data.h5"))

    orig_dims = {d.name: d.coordinates for d in original.dimensions}
    loaded_dims = {d.name: d.coordinates for d in loaded.dimensions}
    assert orig_dims.keys() == loaded_dims.keys()
    for name, coords in orig_dims.items():
        np.testing.assert_array_equal(loaded_dims[name], coords)


def test_roundtrip_normalization(make_dataset, tmp_path):
    original = make_dataset()
    loaded = sds.read_dataset(sds.write_dataset(original, tmp_path / "data.h5"))

    assert loaded.normalization.keys() == original.normalization.keys()
    for name, norm in original.normalization.items():
        np.testing.assert_allclose(loaded.normalization[name].mean, norm.mean)
        np.testing.assert_allclose(loaded.normalization[name].std, norm.std)


def test_roundtrip_boundary_conditions(make_dataset, tmp_path):
    original = make_dataset()
    loaded = sds.read_dataset(sds.write_dataset(original, tmp_path / "data.h5"))

    orig_bcs = {bc.name: bc for bc in original.boundary_conditions}
    loaded_bcs = {bc.name: bc for bc in loaded.boundary_conditions}
    assert orig_bcs.keys() == loaded_bcs.keys()

    for name, bc in orig_bcs.items():
        got = loaded_bcs[name]
        assert got.bc_type == bc.bc_type
        assert got.associated_fields == bc.associated_fields
        assert got.associated_dims == bc.associated_dims
        if bc.mask is None:
            assert got.mask is None
        else:
            np.testing.assert_array_equal(got.mask, bc.mask)
        if bc.values is None:
            assert got.values is None
        else:
            np.testing.assert_array_equal(got.values, bc.values)


def test_registry_written_for_all_present_fields(make_dataset, tmp_path):
    original = make_dataset()
    loaded = sds.read_dataset(sds.write_dataset(original, tmp_path / "data.h5"))

    assert set(loaded.field_registry) == set(original.present_fields)
    for f in original.fields:
        spec = loaded.field_registry[f.name]
        assert (spec.tensor_order, spec.role, spec.unit) == (f.tensor_order, f.role, f.unit)


def test_string_path_accepted(make_dataset, tmp_path):
    original = make_dataset()
    path = str(tmp_path / "data.h5")
    sds.write_dataset(original, path)
    loaded = sds.read_dataset(path)
    assert loaded.dataset_name == original.dataset_name


def test_invalid_dataset_is_not_written(tmp_path):
    """write_dataset validates first; a bad dataset leaves no file behind."""
    bad = sds.Dataset(
        dataset_name="bad",
        representation="grid",
        n_spatial_dims=2,
        n_trajectories=1,
        spatial_dims=["x"],  # length 1 != n_spatial_dims 2
    )
    path = tmp_path / "bad.h5"
    with pytest.raises(ValueError):
        sds.write_dataset(bad, path)
    assert not path.exists()
