"""Shared fixtures: a small 2D grid dataset with a field of every tensor order and extent."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import sciml_data_schema as sds

# The fixture's dimensions
N_TRAJ = 3
N_STEPS = 4
NX, NY = 5, 6
N_DIMS = 2


def _build() -> sds.Dataset:
    """A valid 2D grid dataset exercising every schema feature.

    Returns
    -------
    sds.Dataset
        Five fields: ``pressure`` (t0 state), ``velocity`` (t1 state), ``stress`` (t2 static
        symmetric coefficient), ``reynolds`` (t0 per-trajectory parameter) and ``gravity``
        (t0 global constant stored as ``(1,)``); statistics for ``pressure`` (with the range
        statistics) and ``velocity``; a periodic and a Dirichlet boundary condition.
    """
    rng = np.random.default_rng(0)

    pressure = rng.standard_normal((N_TRAJ, N_STEPS, NX, NY)).astype(np.float32)
    velocity = rng.standard_normal((N_TRAJ, N_STEPS, NX, NY, N_DIMS)).astype(np.float32)
    # A static, symmetric rank-2 coefficient (no time axis)
    stress = rng.standard_normal((N_TRAJ, NX, NY, N_DIMS, N_DIMS)).astype(np.float32)
    reynolds = rng.standard_normal((N_TRAJ,)).astype(np.float32)
    # A global constant: every extent flag off, stored as a length-1 axis
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

    # Pressure also ships the optional range statistics (schema >= 0.3.0)
    normalization = {
        "pressure": sds.Normalization(
            mean=np.asarray(pressure.mean()),
            std=np.asarray(pressure.std()),
            min=np.asarray(pressure.min()),
            max=np.asarray(pressure.max()),
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
    """A factory returning a fresh fixture dataset on each call.

    Returns
    -------
    Callable[[], sds.Dataset]
        :func:`_build`; each call builds anew, so a test may mutate its copy.
    """
    return _build
