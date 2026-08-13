"""The incremental DatasetWriter: streamed output equals inline output, and no
failure mode leaves a valid-looking partial file behind."""

from __future__ import annotations

import numpy as np
import pytest

import sciml_data_schema as sds


def _streamed_copy(dataset: sds.Dataset, name: str) -> tuple[sds.Dataset, np.ndarray]:
    """Swap field ``name``'s data for a placeholder; return the dataset + original data."""
    fld = next(f for f in dataset.fields if f.name == name)
    original = np.asarray(fld.data)
    fld.data = sds.field_placeholder(original.shape, original.dtype)
    return dataset, original


def test_streamed_write_equals_inline_write(make_dataset, tmp_path):
    inline_path = sds.write_dataset(make_dataset(), tmp_path / "inline.h5")

    # Stream the two sample-varying state fields trajectory by trajectory.
    streamed_ds, pressure = _streamed_copy(make_dataset(), "pressure")
    streamed_ds, velocity = _streamed_copy(streamed_ds, "velocity")
    streamed_path = tmp_path / "streamed.h5"
    with sds.DatasetWriter(streamed_ds, streamed_path, streamed=("pressure", "velocity")) as w:
        for i in range(pressure.shape[0]):
            w.append("pressure", i, pressure[i])
            w.append("velocity", i, velocity[i])

    inline, streamed = sds.read_dataset(inline_path), sds.read_dataset(streamed_path)
    assert [f.name for f in inline.fields] == [f.name for f in streamed.fields]
    for a, b in zip(inline.fields, streamed.fields, strict=True):
        np.testing.assert_array_equal(np.asarray(a.data), np.asarray(b.data))
        assert (a.role, a.unit, a.dim_varying, a.time_varying, a.sample_varying) == (
            b.role, b.unit, b.dim_varying, b.time_varying, b.sample_varying
        )


def test_missing_trajectory_fails_and_removes_file(make_dataset, tmp_path):
    dataset, pressure = _streamed_copy(make_dataset(), "pressure")
    path = tmp_path / "partial.h5"

    with pytest.raises(ValueError, match="incomplete streamed fields"):
        with sds.DatasetWriter(dataset, path, streamed=("pressure",)) as w:
            w.append("pressure", 0, pressure[0])  # trajectories 1.. never written
    assert not path.exists()


def test_exception_inside_context_removes_file(make_dataset, tmp_path):
    dataset, _ = _streamed_copy(make_dataset(), "pressure")
    path = tmp_path / "aborted.h5"

    with pytest.raises(RuntimeError, match="boom"):
        with sds.DatasetWriter(dataset, path, streamed=("pressure",)):
            raise RuntimeError("boom")
    assert not path.exists()


def test_append_rejects_bad_shape_and_dtype(make_dataset, tmp_path):
    dataset, pressure = _streamed_copy(make_dataset(), "pressure")
    path = tmp_path / "checked.h5"

    with pytest.raises(ValueError, match="expected shape"):
        with sds.DatasetWriter(dataset, path, streamed=("pressure",)) as w:
            w.append("pressure", 0, pressure[0, :-1])
    assert not path.exists()

    dataset, pressure = _streamed_copy(make_dataset(), "pressure")
    with pytest.raises(ValueError, match="expected dtype"):
        with sds.DatasetWriter(dataset, path, streamed=("pressure",)) as w:
            w.append("pressure", 0, pressure[0].astype(np.float64))
    assert not path.exists()


def test_streamed_field_must_exist_and_be_sample_varying(make_dataset, tmp_path):
    with pytest.raises(ValueError, match="not in dataset"):
        sds.DatasetWriter(make_dataset(), tmp_path / "x.h5", streamed=("nope",))

    # "gravity" is the all-flags-False global constant in the fixture.
    with pytest.raises(ValueError, match="must be sample_varying"):
        sds.DatasetWriter(make_dataset(), tmp_path / "x.h5", streamed=("gravity",))


def test_update_normalization_lands_in_file(make_dataset, tmp_path):
    dataset, pressure = _streamed_copy(make_dataset(), "pressure")
    path = tmp_path / "stats.h5"

    with sds.DatasetWriter(dataset, path, streamed=("pressure",)) as w:
        for i in range(pressure.shape[0]):
            w.append("pressure", i, pressure[i])
        # Stats computed during the pass replace the placeholder entry on close.
        w.update_normalization(
            "pressure",
            sds.Normalization(
                mean=np.asarray(pressure.mean()),
                std=np.asarray(pressure.std()),
                min=np.asarray(pressure.min()),
                max=np.asarray(pressure.max()),
            ),
        )
        with pytest.raises(ValueError, match="unknown field"):
            w.update_normalization("nope", sds.Normalization(np.zeros(()), np.ones(())))

    loaded = sds.read_dataset(path).normalization["pressure"]
    np.testing.assert_allclose(loaded.mean, pressure.mean())
    assert loaded.min is not None
    np.testing.assert_allclose(loaded.min, pressure.min())


def test_write_dataset_still_roundtrips_via_writer(make_dataset, tmp_path):
    # write_dataset is now the degenerate streamed=() case; a plain roundtrip stays intact.
    original = make_dataset()
    loaded = sds.read_dataset(sds.write_dataset(original, tmp_path / "plain.h5"))
    assert loaded.present_fields == original.present_fields
