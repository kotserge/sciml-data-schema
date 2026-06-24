"""Shared fixtures for the test suite."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import sciml_data_schema as sds

# Small, fixed dimensions used across the round-trip and validation tests.
N_TRAJ = 3
N_STEPS = 4
NX, NY = 5, 6
N_DIMS = 2


def _build() -> sds.Dataset:
    """A valid 2D grid dataset exercising every schema feature."""
    rng = np.random.default_rng(0)

    pressure = rng.standard_normal((N_TRAJ, N_STEPS, NX, NY)).astype(np.float32)
    velocity = rng.standard_normal((N_TRAJ, N_STEPS, NX, NY, N_DIMS)).astype(np.float32)
    # A static, symmetric rank-2 coefficient (no time axis).
    stress = rng.standard_normal((N_TRAJ, NX, NY, N_DIMS, N_DIMS)).astype(np.float32)
    reynolds = rng.standard_normal((N_TRAJ,)).astype(np.float32)
    # A true global constant: all extent flags False, stored as a length-1 axis.
    gravity = np.array([9.81], dtype=np.float32)

    fields = [
        sds.Field(name="pressure", data=pressure, tensor_order=0, role="state", unit="Pa"),
        sds.Field(name="velocity", data=velocity, tensor_order=1, role="state", unit="m/s"),
        sds.Field(
            name="stress",
            data=stress,
            tensor_order=2,
            role="coefficient",
            unit="Pa",
            time_varying=False,
            symmetric=True,
        ),
        sds.Field(
            name="reynolds",
            data=reynolds,
            tensor_order=0,
            role="parameter",
            dim_varying=False,
            time_varying=False,
        ),
        sds.Field(
            name="gravity",
            data=gravity,
            tensor_order=0,
            role="parameter",
            unit="m/s^2",
            dim_varying=False,
            time_varying=False,
            sample_varying=False,
        ),
    ]

    normalization = {
        "pressure": sds.Normalization(
            mean=np.asarray(pressure.mean()), std=np.asarray(pressure.std())
        ),
        "velocity": sds.Normalization(
            mean=velocity.mean(axis=(0, 1, 2, 3)), std=velocity.std(axis=(0, 1, 2, 3))
        ),
    }

    boundary_conditions = [
        sds.BoundaryCondition(name="periodic", bc_type="periodic", associated_dims=["x"]),
        sds.BoundaryCondition(
            name="wall",
            bc_type="dirichlet",
            associated_fields=["pressure"],
            associated_dims=["y"],
            mask=np.ones((NX, NY), dtype=bool),
            values=np.zeros((NX, NY), dtype=np.float32),
        ),
    ]

    return sds.Dataset(
        dataset_name="fixture",
        representation="grid",
        grid_type="cartesian",
        n_spatial_dims=N_DIMS,
        n_trajectories=N_TRAJ,
        spatial_dims=["x", "y"],
        dimensions=[
            sds.Dimension(name="x", coordinates=np.linspace(0, 1, NX, dtype=np.float32)),
            sds.Dimension(name="y", coordinates=np.linspace(0, 1, NY, dtype=np.float32)),
            sds.Dimension(name="time", coordinates=np.arange(N_STEPS, dtype=np.float32)),
        ],
        fields=fields,
        boundary_conditions=boundary_conditions,
        normalization=normalization,
    )


@pytest.fixture
def make_dataset() -> Callable[[], sds.Dataset]:
    """Factory returning a fresh, valid dataset on each call (mutation-safe)."""
    return _build
