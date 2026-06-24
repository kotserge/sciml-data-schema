"""Build, write, and read a 1+1D advection-diffusion dataset via the library.

PDE (periodic on ``[0, L)``)::

    u_t + a u_x = D u_xx

    u : scalar solution      -> state field      (t0)
    a : advection speed      -> scalar parameter  (t0, degenerate)
    D : diffusion coefficient -> scalar parameter  (t0, degenerate)

This is the simplest case that exercises the *unified field model*: ``u``, ``a``,
and ``D`` all live in ``t0_fields`` (tensor order 0) and are distinguished only
by ``role`` plus extent flags -- there is no separate "scalars" group. ``a`` and
``D`` are *degenerate* fields (``dim_varying=False``, ``time_varying=False``),
stored as one value per trajectory.

Every byte goes through the public API: :func:`sciml_data_schema.write_dataset`
and :func:`sciml_data_schema.read_dataset`. Run it directly to generate a file,
read it back, and print a role-routed training window::

    python examples/advection_diffusion.py
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

import sciml_data_schema as sds


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
    """Build an advection-diffusion :class:`~sciml_data_schema.Dataset` in memory.

    Trajectories = ``advections x diffusions x seeds``. The seed sets a smooth
    low-mode initial condition; the same IC is evolved under every ``(a, D)``
    pair. Each periodic mode evolves exactly,
    ``u_hat_k(t) = u_hat_k(0) * exp((-i a k - D k^2) t)``, so the data carries no
    time-discretization error.
    """
    combos = [(a, d, s) for a in advections for d in diffusions for s in range(n_seeds)]
    n_traj = len(combos)

    dx = length / n_points
    x = np.arange(n_points) * dx
    times = np.linspace(0.0, t_final, n_steps)
    k = np.fft.fftfreq(n_points, d=dx) * 2.0 * np.pi  # angular wavenumbers

    u = np.empty((n_traj, n_steps, n_points), dtype=np.float32)
    a_arr = np.empty((n_traj,), dtype=np.float32)
    d_arr = np.empty((n_traj,), dtype=np.float32)

    for i, (a, d, s) in enumerate(combos):
        rng = np.random.default_rng(seed0 + s)
        u0 = np.zeros(n_points)
        for n in range(1, n_modes + 1):  # smooth low-mode IC
            u0 += rng.normal() * np.cos(2 * np.pi * n * x / length)
            u0 += rng.normal() * np.sin(2 * np.pi * n * x / length)
        u0 /= np.max(np.abs(u0)) + 1e-12

        u0_hat = np.fft.fft(u0)
        for j, t in enumerate(times):
            propagator = np.exp((-1j * a * k - d * k * k) * t)
            u[i, j] = np.fft.ifft(u0_hat * propagator).real
        a_arr[i], d_arr[i] = a, d

    fields = [
        # Solution: varies across trajectories, time, and space.
        sds.Field(name="u", data=u, tensor_order=0, role="state"),
        # Scalar parameters: one value per trajectory, broadcast on read.
        sds.Field(
            name="advection",
            data=a_arr,
            tensor_order=0,
            role="parameter",
            unit="m/s",
            dim_varying=False,
            time_varying=False,
        ),
        sds.Field(
            name="diffusion",
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
                associated_fields=["u"],
                associated_dims=["x"],
            )
        ],
        normalization=normalization,
    )


class TrainingWindow(NamedTuple):
    """One model input/target window plus the trajectory's conditioning params."""

    input: np.ndarray  # (n_in, n_points)  past states
    target: np.ndarray  # (n_out, n_points) future states (prediction target)
    params: np.ndarray  # (n_params,)       scalar conditioning values
    param_names: list[str]


def windowed_training_view(
    dataset: sds.Dataset, *, n_in: int = 4, n_out: int = 1
) -> TrainingWindow:
    """Routing demo: split one trajectory into a model input/target window.

    ``role`` is the bridge between PDE meaning and storage: ``state`` fields are
    prediction targets, everything else is conditioning. Here we take the first
    trajectory, window ``u`` into past (input) and future (target) frames, and
    collect the scalar parameters as a conditioning vector -- the role-based
    routing a downstream loader would perform, in plain NumPy.
    """
    by_name = {f.name: f for f in dataset.fields}

    u = by_name["u"].data[0]  # (n_steps, n_points) for trajectory 0
    state_in = u[:n_in]
    state_out = u[n_in : n_in + n_out]

    params = np.array(
        [float(by_name[name].data[0]) for name in ("advection", "diffusion")],
        dtype=np.float32,
    )
    return TrainingWindow(
        input=state_in,
        target=state_out,
        params=params,
        param_names=["advection", "diffusion"],
    )


def main() -> None:
    import os

    path = "advection_diffusion_1d.h5"
    dataset = build_dataset(n_points=128, n_steps=51, n_seeds=8)
    sds.write_dataset(dataset, path, compression="gzip")
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

    loaded = sds.read_dataset(path)
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
