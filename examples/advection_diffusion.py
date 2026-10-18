"""Build, write and read a 1+1D advection-diffusion dataset through the public API.

The PDE, periodic on ``[0, L)``::

    u_t + a u_x = D u_xx

    u : the solution              -> a state field,     t0, varying in sample, time and space
    a : the advection speed       -> a parameter field, t0, one value per trajectory
    D : the diffusion coefficient -> a parameter field, t0, one value per trajectory

All three live in ``t0_fields`` and differ by role and extent flags: ``advection`` and
``diffusion`` have ``dim_varying=False`` and ``time_varying=False``, so they are stored as
``(n_trajectories,)``. :func:`build_dataset` assembles the dataset in memory,
:func:`windowed_training_view` routes its fields by role into one training window, and
:func:`main` writes the file with :func:`sciml_data_schema.write_dataset`, reads it back with
:func:`sciml_data_schema.read_dataset` and prints both.

Examples
--------
::

    python examples/advection_diffusion.py    # writes advection_diffusion_1d.h5 here
"""

from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np

import sciml_data_schema as sds

# The file `main` writes, in the current directory
OUTPUT_PATH = "advection_diffusion_1d.h5"

# The field names, and the order of the parameters in a window's conditioning vector
FIELD_STATE = "u"
FIELD_ADVECTION = "advection"
FIELD_DIFFUSION = "diffusion"
PARAMETER_FIELDS = (FIELD_ADVECTION, FIELD_DIFFUSION)


def build_dataset(
    *,
    length: float = 2.0 * np.pi,
    n_points: int = 128,
    n_steps: int = 51,
    t_final: float = 2.0,
    advections: tuple[float, ...] = (0.5, 1.0, 1.5),
    diffusions: tuple[float, ...] = (1e-3, 1e-2, 1e-1),
    n_seeds: int = 8,
    n_modes: int = 6,
    seed0: int = 0,
) -> sds.Dataset:
    """Build the advection-diffusion dataset in memory.

    One trajectory per ``(a, D, seed)`` combination. The seed draws a smooth initial condition
    of ``n_modes`` random Fourier modes, scaled to unit peak; the same initial condition is
    evolved under every ``(a, D)`` pair. Each mode evolves exactly,
    ``u_hat_k(t) = u_hat_k(0) * exp((-i a k - D k^2) t)``, so the data carries no
    time-discretization error.

    Parameters
    ----------
    length : float
        The domain length ``L``.
    n_points : int
        Grid points along ``x``.
    n_steps : int
        Time steps, ``t = 0`` included.
    t_final : float
        The final time.
    advections : tuple[float, ...]
        The advection speeds ``a`` swept.
    diffusions : tuple[float, ...]
        The diffusion coefficients ``D`` swept.
    n_seeds : int
        Initial conditions per ``(a, D)`` pair.
    n_modes : int
        Fourier modes in an initial condition.
    seed0 : int
        The seed of the first initial condition; seed ``s`` uses ``seed0 + s``.

    Returns
    -------
    sds.Dataset
        A 1D grid dataset of ``len(advections) * len(diffusions) * n_seeds`` trajectories:
        ``u`` ``(n_trajectories, n_steps, n_points)``, ``advection`` and ``diffusion``
        ``(n_trajectories,)``, all float32, with z-score statistics per field and a periodic
        boundary condition on ``u`` along ``x``.
    """
    combos = [(a, d, s) for a in advections for d in diffusions for s in range(n_seeds)]
    n_traj = len(combos)

    # The grid, the times and the angular wavenumbers of the periodic modes
    dx = length / n_points
    x = np.arange(n_points) * dx
    times = np.linspace(0.0, t_final, n_steps)
    k = np.fft.fftfreq(n_points, d=dx) * 2.0 * np.pi

    u = np.empty((n_traj, n_steps, n_points), dtype=np.float32)
    a_arr = np.empty((n_traj,), dtype=np.float32)
    d_arr = np.empty((n_traj,), dtype=np.float32)

    # Each trajectory: a smooth low-mode initial condition, evolved exactly in Fourier space
    for i, (a, d, s) in enumerate(combos):
        rng = np.random.default_rng(seed0 + s)
        u0 = np.zeros(n_points)
        for n in range(1, n_modes + 1):
            u0 += rng.normal() * np.cos(2 * np.pi * n * x / length)
            u0 += rng.normal() * np.sin(2 * np.pi * n * x / length)
        u0 /= np.max(np.abs(u0)) + 1e-12

        u0_hat = np.fft.fft(u0)
        for j, t in enumerate(times):
            propagator = np.exp((-1j * a * k - d * k * k) * t)
            u[i, j] = np.fft.ifft(u0_hat * propagator).real
        a_arr[i], d_arr[i] = a, d

    # The solution varies along every axis; the parameters only across trajectories
    fields = [
        sds.Field(name=FIELD_STATE, data=u, tensor_order=0, role="state"),
        sds.Field(
            name=FIELD_ADVECTION,
            data=a_arr,
            tensor_order=0,
            role="parameter",
            unit="m/s",
            dim_varying=False,
            time_varying=False,
        ),
        sds.Field(
            name=FIELD_DIFFUSION,
            data=d_arr,
            tensor_order=0,
            role="parameter",
            unit="m^2/s",
            dim_varying=False,
            time_varying=False,
        ),
    ]

    normalization = {
        f.name: sds.Normalization(mean=np.asarray(f.data.mean()), std=np.asarray(f.data.std()))
        for f in fields
    }

    return sds.Dataset(
        dataset_name="advection_diffusion_1d",
        representation="grid",
        grid_type="cartesian",
        n_spatial_dims=1,
        n_trajectories=n_traj,
        spatial_dims=["x"],
        dimensions=[
            sds.Dimension(name="x", coordinates=x.astype(np.float32)),
            sds.Dimension(name="time", coordinates=times.astype(np.float32)),
        ],
        fields=fields,
        boundary_conditions=[
            sds.BoundaryCondition(
                name="periodic",
                bc_type="periodic",
                associated_fields=[FIELD_STATE],
                associated_dims=["x"],
            )
        ],
        normalization=normalization,
    )


class TrainingWindow(NamedTuple):
    """One model input/target window of a trajectory plus its conditioning parameters.

    Attributes
    ----------
    input : np.ndarray
        ``(n_in, n_points)``, the past states.
    target : np.ndarray
        ``(n_out, n_points)``, the future states (the prediction target).
    params : np.ndarray
        ``(n_params,)``, the trajectory's scalar parameters.
    param_names : list[str]
        The field name of each entry of ``params``.
    """

    input: np.ndarray
    target: np.ndarray
    params: np.ndarray
    param_names: list[str]


def windowed_training_view(
    dataset: sds.Dataset, *, n_in: int = 4, n_out: int = 1
) -> TrainingWindow:
    """Route the first trajectory's fields by role into one training window.

    The ``state`` field is windowed into ``n_in`` input frames followed by ``n_out`` target
    frames; the ``parameter`` fields become the conditioning vector.

    Parameters
    ----------
    dataset : sds.Dataset
        A dataset built by :func:`build_dataset` or read back from its file.
    n_in : int
        Input frames.
    n_out : int
        Target frames, taken right after the input frames.

    Returns
    -------
    TrainingWindow
        The window of trajectory 0.

    Raises
    ------
    KeyError
        The dataset lacks ``u``, ``advection`` or ``diffusion``.
    """
    by_name = {f.name: f for f in dataset.fields}

    # The state of trajectory 0, (n_steps, n_points), split into past and future frames
    u = by_name[FIELD_STATE].data[0]
    state_in = u[:n_in]
    state_out = u[n_in : n_in + n_out]

    params = np.array([float(by_name[name].data[0]) for name in PARAMETER_FIELDS], dtype=np.float32)
    return TrainingWindow(
        input=state_in,
        target=state_out,
        params=params,
        param_names=list(PARAMETER_FIELDS),
    )


def main() -> None:
    """Write the dataset to ``OUTPUT_PATH``, read it back and print its fields and a window."""
    dataset = build_dataset(n_points=128, n_steps=51, n_seeds=8)
    sds.write_dataset(dataset, OUTPUT_PATH, compression="gzip")
    print(f"wrote {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH) / 1e6:.1f} MB)")

    loaded = sds.read_dataset(OUTPUT_PATH)
    print(
        f"dataset: {loaded.dataset_name} | n_traj: {loaded.n_trajectories} | "
        f"present: {loaded.present_fields}"
    )
    for f in loaded.fields:
        print(
            f"  {f.name:10s} order={f.tensor_order} role={f.role:9s} "
            f"shape={f.data.shape} dim={f.dim_varying} time={f.time_varying}"
        )

    window = windowed_training_view(loaded, n_in=4, n_out=1)
    print(
        f"\nwindow: input {window.input.shape} -> target {window.target.shape} "
        f"| params {window.param_names} = {window.params}"
    )


if __name__ == "__main__":
    main()
