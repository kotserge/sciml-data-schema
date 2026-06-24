"""
advection_diffusion.py — concrete 1+1D advection-diffusion scenario.

PDE (periodic on [0, L)):
        u_t + a u_x = D u_xx
    u : scalar solution (state field)
    a : advection speed   (SCALAR parameter)
    D : diffusion coeff.  (SCALAR parameter)

Demonstrates the unified field model on the simplest meaningful case:
`u`, `a`, and `D` all live in `t0_fields` (all tensor_order 0). They are
distinguished only by `role` + extent flags — there is no separate "scalars"
group. `a` and `D` are degenerate fields: dim_varying=False, time_varying=False,
sample_varying=True, stored as one value per trajectory.

Layering:
  * generate_advection_diffusion(...) — writer (exact spectral solution).
  * GridDatasetReader (imported)       — BASIC, shared, representation-level.
  * AdvectionDiffusionDataset          — SPECIALIZED, concrete to this PDE.
"""

from __future__ import annotations

import json
from typing import List, Optional, Tuple

import h5py
import numpy as np

from grid_loader import GridDatasetReader   # shared basic reader (grid profile)

SCHEMA_VERSION = "0.1.0"
REGISTRY_VERSION = "0.1.0"

FIELD_REGISTRY = {
    "u":         {"tensor_order": 0, "role": "state",     "unit": "dimensionless"},
    "advection": {"tensor_order": 0, "role": "parameter", "unit": "m/s"},
    "diffusion": {"tensor_order": 0, "role": "parameter", "unit": "m^2/s"},
}


# --------------------------------------------------------------------------- #
# Writer — exact spectral generation
# --------------------------------------------------------------------------- #
def generate_advection_diffusion(
    path: str,
    *,
    L: float = 2.0 * np.pi,
    X: int = 128,
    n_steps: int = 51,
    t_final: float = 2.0,
    advections: Tuple[float, ...] = (0.5, 1.0, 1.5),
    diffusions: Tuple[float, ...] = (1e-3, 1e-2, 1e-1),
    n_seeds: int = 8,
    n_modes: int = 6,
    seed0: int = 0,
) -> None:
    """Write a schema-conforming advection-diffusion dataset.

    Trajectories = advections x diffusions x seeds. The seed sets the initial
    condition (a smooth low-mode field); the same IC is evolved under every
    (a, D) pair, so a surrogate must learn the coefficient dependence.

    Each periodic mode evolves exactly:  u_hat_k(t) = u_hat_k(0) * exp((-i a k - D k^2) t),
    so there is no time-discretization error in the data.
    """
    combos = [(a, D, s) for a in advections for D in diffusions for s in range(n_seeds)]
    n_traj = len(combos)

    dx = L / X
    x = np.arange(X) * dx
    times = np.linspace(0.0, t_final, n_steps)
    k = np.fft.fftfreq(X, d=dx) * 2.0 * np.pi          # angular wavenumbers

    u = np.empty((n_traj, n_steps, X), dtype=np.float32)
    a_arr = np.empty((n_traj,), dtype=np.float32)
    d_arr = np.empty((n_traj,), dtype=np.float32)

    for i, (a, D, s) in enumerate(combos):
        rng = np.random.default_rng(seed0 + s)
        u0 = np.zeros(X)
        for n in range(1, n_modes + 1):                # smooth low-mode IC
            u0 += rng.normal() * np.cos(2 * np.pi * n * x / L)
            u0 += rng.normal() * np.sin(2 * np.pi * n * x / L)
        u0 /= np.max(np.abs(u0)) + 1e-12

        u0_hat = np.fft.fft(u0)
        for j, t in enumerate(times):
            propagator = np.exp((-1j * a * k - D * k * k) * t)
            u[i, j] = np.fft.ifft(u0_hat * propagator).real
        a_arr[i], d_arr[i] = a, D

    # 0.8 / 0.1 / 0.1 split over trajectories
    perm = np.random.default_rng(1234).permutation(n_traj)
    n_tr, n_va = int(0.8 * n_traj), int(0.1 * n_traj)
    split = {
        "train": perm[:n_tr].tolist(),
        "val":   perm[n_tr:n_tr + n_va].tolist(),
        "test":  perm[n_tr + n_va:].tolist(),
    }

    sstr = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["registry_version"] = REGISTRY_VERSION
        f.attrs["dataset_name"] = "advection_diffusion_1d"
        f.attrs["representation"] = "grid"
        f.attrs["grid_type"] = "cartesian"
        f.attrs["n_spatial_dims"] = 1
        f.attrs["n_trajectories"] = n_traj
        f.attrs["field_registry"] = json.dumps(FIELD_REGISTRY)
        f.attrs["present_fields"] = np.array(["u", "advection", "diffusion"], dtype=sstr)
        f.attrs["split"] = json.dumps(split)

        dims = f.create_group("dimensions")
        dims.attrs["spatial_dims"] = np.array(["x"], dtype=sstr)
        dims.create_dataset("x", data=x.astype(np.float32))
        dims.create_dataset("time", data=times.astype(np.float32))

        bc = f.create_group("boundary_conditions").create_group("periodic")
        bc.attrs["bc_type"] = "periodic"
        bc.attrs["associated_fields"] = np.array(["u"], dtype=sstr)
        bc.attrs["associated_dims"] = np.array(["x"], dtype=sstr)

        norm = f.create_group("normalization")
        for name, arr in (("u", u), ("advection", a_arr), ("diffusion", d_arr)):
            g = norm.create_group(name)
            g.attrs["mean"] = float(arr.mean())
            g.attrs["std"] = float(arr.std())

        t0 = f.create_group("t0_fields")
        # state: chunk by trajectory (recommended grid chunking), compressed.
        du = t0.create_dataset("u", data=u, chunks=(1, n_steps, X), compression="gzip")
        du.attrs.update(dim_varying=True, time_varying=True, sample_varying=True)
        # scalar parameters: degenerate fields, one value per trajectory.
        for name, arr in (("advection", a_arr), ("diffusion", d_arr)):
            d = t0.create_dataset(name, data=arr)
            d.attrs.update(dim_varying=False, time_varying=False, sample_varying=True)


# --------------------------------------------------------------------------- #
# Specialized loader
# --------------------------------------------------------------------------- #
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    Dataset = object  # type: ignore


class AdvectionDiffusionDataset(Dataset):
    """Windowed training view of the 1+1D advection-diffusion scenario.

    Per item (default, coeffs_as_channels=False):
        input  : (T_in,  X)         past states u
        target : (T_out, X)         future states u  (prediction target)
        coeffs : (2,)               normalized [advection, diffusion]
    With coeffs_as_channels=True the scalar coeffs are broadcast across space
    and stacked onto the input instead:
        input  : (T_in, X, 3)       [u, advection, diffusion]
        target : (T_out, X)
    """

    def __init__(
        self,
        path: str,
        split: str = "train",
        n_in: int = 4,
        n_out: int = 1,
        stride: int = 1,
        normalize: bool = True,
        coeffs_as_channels: bool = False,
    ):
        self.reader = GridDatasetReader(path)
        m = self.reader.metadata
        if m.n_spatial_dims != 1:
            raise ValueError("AdvectionDiffusionDataset expects a 1+1D (1 spatial dim) file.")
        for needed in ("u", "advection", "diffusion"):
            if needed not in m.present_fields:
                raise ValueError(f"missing required field '{needed}'")

        self.n_in, self.n_out, self.stride = n_in, n_out, stride
        self.normalize = normalize
        self.coeffs_as_channels = coeffs_as_channels
        (self.X,) = m.spatial_shape

        win = n_in + n_out
        self.windows: List[Tuple[int, int]] = [
            (traj, start)
            for traj in m.split[split]
            for start in range(0, m.n_steps - win + 1, stride)
        ]

        # Pre-fetch normalization stats once.
        self._nu = self.reader.normalization("u")
        self._na = self.reader.normalization("advection")
        self._nd = self.reader.normalization("diffusion")

    def __len__(self) -> int:
        return len(self.windows)

    @staticmethod
    def _z(arr, stats: Optional[Tuple[np.ndarray, np.ndarray]]):
        if stats is None:
            return arr
        mean, std = stats
        return (arr - mean) / (std if np.all(std != 0) else 1.0)

    def __getitem__(self, idx: int) -> dict:
        traj, start = self.windows[idx]
        fields = self.reader.read_trajectory(traj)

        u = fields["u"].data                       # (n_steps, X), as stored
        a = float(fields["advection"].data)        # degenerate field -> scalar
        d = float(fields["diffusion"].data)

        if self.normalize:
            u = self._z(u, self._nu)
            a = float(self._z(np.asarray(a), self._na))
            d = float(self._z(np.asarray(d), self._nd))

        i0, i1 = start, start + self.n_in
        u_in = np.ascontiguousarray(u[i0:i1], dtype=np.float32)               # (T_in, X)
        u_out = np.ascontiguousarray(u[i1:i1 + self.n_out], dtype=np.float32)  # (T_out, X)
        coeffs = np.array([a, d], dtype=np.float32)                            # (2,)

        if self.coeffs_as_channels:
            ui = u_in[..., None]                                       # (T_in, X, 1)
            cc = np.broadcast_to(coeffs, (self.n_in, self.X, 2))       # (T_in, X, 2)
            inp = np.ascontiguousarray(np.concatenate([ui, cc], -1), dtype=np.float32)
            input_t = torch.from_numpy(inp)                            # (T_in, X, 3)
        else:
            input_t = torch.from_numpy(u_in)                           # (T_in, X)

        return {
            "input": input_t,
            "target": torch.from_numpy(u_out),
            "coeffs": torch.from_numpy(coeffs),
            "coeff_names": ["advection", "diffusion"],
        }


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import os
    path = "advection_diffusion_1d.h5"
    generate_advection_diffusion(path, X=128, n_steps=51, n_seeds=8)
    print("wrote", path, f"({os.path.getsize(path) / 1e6:.1f} MB)")

    # Basic layer
    r = GridDatasetReader(path)
    print("dataset:", r.metadata.dataset_name, "| n_traj:", r.metadata.n_trajectories,
          "| spatial:", r.metadata.spatial_shape, "| n_steps:", r.metadata.n_steps)
    tr = r.read_trajectory(0)
    for name, fa in tr.items():
        print(f"  {name:10s} order={fa.tensor_order} role={fa.role:9s} "
              f"shape={fa.data.shape} dim={fa.dim_varying} t={fa.time_varying}")

    # Specialized layer — vector conditioning (default)
    ds = AdvectionDiffusionDataset(path, split="train", n_in=4, n_out=1)
    s = ds[0]
    print("\n[vector ] len:", len(ds),
          "| input", tuple(s["input"].shape),
          "| target", tuple(s["target"].shape),
          "| coeffs", tuple(s["coeffs"].shape), s["coeff_names"])

    # Specialized layer — coeffs broadcast as channels
    dsc = AdvectionDiffusionDataset(path, split="train", n_in=4, n_out=1, coeffs_as_channels=True)
    sc = dsc[0]
    print("[channel] input", tuple(sc["input"].shape), "| target", tuple(sc["target"].shape))

    # DataLoader smoke test (fork-safe lazy handle)
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=8, num_workers=2)
    b = next(iter(dl))
    print("[batch  ] input", tuple(b["input"].shape),
          "| target", tuple(b["target"].shape),
          "| coeffs", tuple(b["coeffs"].shape))
