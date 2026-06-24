"""
grid_loader.py — reference sketch for the grid profile of the PDE schema.

Two layers, matching the agreed architecture:

  GridDatasetReader   — BASIC loader. Schema-aware, framework-neutral.
                        Reads data *as stored*, annotated with metadata.
                        No windowing, no normalization, no broadcasting,
                        no torch. Fork-safe lazy HDF5 handle.

  GridWindowDataset   — SPECIALIZED wrapper (torch.utils.data.Dataset).
                        Adds windowing (T_in -> T_out), normalization,
                        role-based routing (state -> targets; forcing /
                        parameter / coefficient -> conditioning), the
                        conditioning-interface choice, and tensor conversion.

This is a sketch: single-file datasets, time-dependent grid only (steady-state
would skip windowing), and forcing is aligned to the input window (a modeling
choice flagged inline).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

# Loader is written against these MAJOR versions; bump deliberately.
SUPPORTED_SCHEMA_MAJOR = 0
SUPPORTED_REGISTRY_MAJOR = 0


def _major(semver: str) -> int:
    return int(semver.split(".")[0])


# --------------------------------------------------------------------------- #
# Metadata + field containers
# --------------------------------------------------------------------------- #
@dataclass
class FieldArray:
    """A single field for one trajectory, exactly as stored, plus its metadata.

    `data` keeps the storage shape: a time_varying t0 field is (n_steps, X, Y);
    a static scalar parameter is a 0-d array. Interpretation (windowing,
    broadcasting) is the caller's job — this layer does not presume a use.
    """
    name: str
    tensor_order: int
    role: str               # state | forcing | coefficient | parameter | boundary
    unit: str
    dim_varying: bool
    time_varying: bool
    sample_varying: bool
    data: np.ndarray


@dataclass
class DatasetMetadata:
    schema_version: str
    registry_version: str
    dataset_name: str
    representation: str
    grid_type: str
    n_spatial_dims: int
    n_trajectories: int
    n_steps: int
    field_registry: Dict[str, dict]      # name -> {tensor_order, role, unit}
    present_fields: List[str]
    split: Dict[str, List[int]]          # "train"/"val"/"test" -> traj indices
    spatial_dims: List[str]              # e.g. ["x", "y"]
    spatial_shape: Tuple[int, ...]       # e.g. (X, Y)


# --------------------------------------------------------------------------- #
# BASIC loader
# --------------------------------------------------------------------------- #
class GridDatasetReader:
    """Framework-neutral, schema-aware reader for one grid-profile HDF5 file."""

    def __init__(self, path: str):
        self.path = path
        self._file: Optional[h5py.File] = None
        self._pid: Optional[int] = None
        self.metadata = self._read_metadata()
        self._check_versions()

    # -- fork-safe lazy handle ---------------------------------------------- #
    def _handle(self) -> h5py.File:
        # Re-open per process: h5py handles do not survive fork(), so a handle
        # cached in the parent must not be reused inside a DataLoader worker.
        pid = os.getpid()
        if self._file is None or self._pid != pid:
            self._file = h5py.File(self.path, "r", swmr=True)
            self._pid = pid
        return self._file

    # -- metadata ----------------------------------------------------------- #
    def _read_metadata(self) -> DatasetMetadata:
        with h5py.File(self.path, "r") as f:
            a = f.attrs
            registry = json.loads(a["field_registry"])
            present = list(a["present_fields"])
            split = json.loads(a["split"])
            spatial_dims = list(f["dimensions"].attrs["spatial_dims"])
            coords = [f["dimensions"][d] for d in spatial_dims]
            spatial_shape = tuple(int(c.shape[0]) for c in coords)
            time = f["dimensions"]["time"]
            n_steps = int(time.shape[0]) if "time" in f["dimensions"] else 1
            return DatasetMetadata(
                schema_version=str(a["schema_version"]),
                registry_version=str(a["registry_version"]),
                dataset_name=str(a["dataset_name"]),
                representation=str(a["representation"]),
                grid_type=str(a.get("grid_type", "cartesian")),
                n_spatial_dims=int(a["n_spatial_dims"]),
                n_trajectories=int(a["n_trajectories"]),
                n_steps=n_steps,
                field_registry=registry,
                present_fields=present,
                split=split,
                spatial_dims=spatial_dims,
                spatial_shape=spatial_shape,
            )

    def _check_versions(self) -> None:
        m = self.metadata
        if m.representation != "grid":
            raise ValueError(f"{self.path}: not a grid-profile file ({m.representation}).")
        if _major(m.schema_version) != SUPPORTED_SCHEMA_MAJOR:
            raise ValueError(f"Schema major {m.schema_version} unsupported.")
        if _major(m.registry_version) != SUPPORTED_REGISTRY_MAJOR:
            raise ValueError(f"Registry major {m.registry_version} unsupported.")

    # -- coordinates / stats ------------------------------------------------ #
    def coords(self) -> Dict[str, np.ndarray]:
        f = self._handle()
        return {d: np.asarray(f["dimensions"][d]) for d in self.metadata.spatial_dims}

    def normalization(self, name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        f = self._handle()
        if "normalization" in f and name in f["normalization"]:
            g = f["normalization"][name]
            return np.asarray(g.attrs["mean"]), np.asarray(g.attrs["std"])
        return None

    # -- the core read ------------------------------------------------------ #
    def read_trajectory(self, traj_idx: int) -> Dict[str, FieldArray]:
        """All present fields for one trajectory, as stored + annotated."""
        f = self._handle()
        out: Dict[str, FieldArray] = {}
        for name in self.metadata.present_fields:
            spec = self.metadata.field_registry[name]
            order = int(spec["tensor_order"])
            dset = f[f"t{order}_fields"][name]
            sample_varying = bool(dset.attrs["sample_varying"])
            # Index the trajectory axis only if it exists in storage.
            arr = dset[traj_idx] if sample_varying else dset[...]
            out[name] = FieldArray(
                name=name,
                tensor_order=order,
                role=spec["role"],
                unit=spec.get("unit", "dimensionless"),
                dim_varying=bool(dset.attrs["dim_varying"]),
                time_varying=bool(dset.attrs["time_varying"]),
                sample_varying=sample_varying,
                data=np.asarray(arr),
            )
        return out


# --------------------------------------------------------------------------- #
# SPECIALIZED wrapper
# --------------------------------------------------------------------------- #
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:                       # keep the basic layer importable w/o torch
    Dataset = object                      # type: ignore


def _to_channels(arr: np.ndarray, fa: FieldArray) -> np.ndarray:
    """Move tensor components into a trailing channel axis.

    t0 (T,X,Y)      -> (T,X,Y,1)
    t1 (T,X,Y,D)    -> (T,X,Y,D)
    t2 (T,X,Y,D,D)  -> (T,X,Y,D*D)
    """
    if fa.tensor_order == 0:
        return arr[..., None]
    if fa.tensor_order == 1:
        return arr
    return arr.reshape(*arr.shape[:-2], -1)


class GridWindowDataset(Dataset):
    """Windowed, normalized, role-routed view for training.

    Returns per item:
        input_fields  : (T_in,  X, Y, C_in)    state[+forcing][+params if channels]
        output_fields : (T_out, X, Y, C_state)  state only (prediction target)
        params        : (C_param,)              present only if conditioning="vector"
        *_names        : channel provenance (for multi-physics consumers)
    """

    def __init__(
        self,
        path: str,
        split: str = "train",
        n_in: int = 4,
        n_out: int = 1,
        stride: int = 1,
        normalize: bool = True,
        conditioning: str = "channels",      # "channels" | "vector"
        include_forcing: bool = True,
    ):
        self.reader = GridDatasetReader(path)
        self.n_in, self.n_out, self.stride = n_in, n_out, stride
        self.normalize = normalize
        self.conditioning = conditioning
        self.include_forcing = include_forcing

        m = self.reader.metadata
        self.spatial_shape = m.spatial_shape
        self._roles = {n: m.field_registry[n]["role"] for n in m.present_fields}

        # Build the (trajectory, start_step) window index over this split.
        win = n_in + n_out
        self.windows: List[Tuple[int, int]] = [
            (traj, start)
            for traj in m.split[split]
            for start in range(0, m.n_steps - win + 1, stride)
        ]

    def __len__(self) -> int:
        return len(self.windows)

    def _norm(self, name: str, arr: np.ndarray, fa: FieldArray) -> np.ndarray:
        if not self.normalize:
            return arr
        stats = self.reader.normalization(name)
        if stats is None:
            return arr
        mean, std = stats
        return (arr - mean) / np.where(std == 0, 1.0, std)

    def __getitem__(self, idx: int) -> dict:
        traj, start = self.windows[idx]
        fields = self.reader.read_trajectory(traj)

        in_sl = slice(start, start + self.n_in)
        out_sl = slice(start + self.n_in, start + self.n_in + self.n_out)

        state_in, state_out, state_names = [], [], []
        forcing_in, forcing_names = [], []
        param_chunks, param_names = [], []

        for name, fa in fields.items():
            data = self._norm(name, fa.data, fa)
            role = fa.role

            if role == "state":
                ci = _to_channels(data[in_sl], fa)
                co = _to_channels(data[out_sl], fa)
                state_in.append(ci); state_out.append(co)
                state_names += [f"{name}[{k}]" for k in range(ci.shape[-1])]

            elif role == "forcing" and self.include_forcing:
                # Forcing aligned to the INPUT window. If your model conditions
                # on forcing over the prediction interval instead, slice out_sl.
                cf = _to_channels(data[in_sl], fa)
                forcing_in.append(cf)
                forcing_names += [f"{name}[{k}]" for k in range(cf.shape[-1])]

            elif role in ("parameter", "coefficient"):
                # dim_varying=False => scalar/short vector; broadcast handled below.
                param_chunks.append((name, fa, np.atleast_1d(data).ravel()))

        X, Y = self.spatial_shape
        input_fields = np.concatenate(state_in, axis=-1)        # (T_in, X, Y, C_state)
        output_fields = np.concatenate(state_out, axis=-1)      # (T_out,X, Y, C_state)
        input_names = list(state_names)

        if forcing_in:
            input_fields = np.concatenate([input_fields, *forcing_in], axis=-1)
            input_names += forcing_names

        out = {}
        if self.conditioning == "channels" and param_chunks:
            # Broadcast each scalar/vector param across (T_in, X, Y) as channels.
            for name, fa, vec in param_chunks:
                bc = np.broadcast_to(vec, (self.n_in, X, Y, vec.shape[0]))
                input_fields = np.concatenate([input_fields, bc], axis=-1)
                input_names += [f"{name}[{k}]" for k in range(vec.shape[0])]
        elif self.conditioning == "vector" and param_chunks:
            pvec = np.concatenate([vec for _, _, vec in param_chunks])
            out["params"] = torch.from_numpy(pvec).float()
            out["param_names"] = [n for n, _, _ in param_chunks]

        out["input_fields"] = torch.from_numpy(np.ascontiguousarray(input_fields)).float()
        out["output_fields"] = torch.from_numpy(np.ascontiguousarray(output_fields)).float()
        out["input_field_names"] = input_names
        out["output_field_names"] = state_names
        return out


# --------------------------------------------------------------------------- #
# Usage
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Basic layer: inspect what's in the file, framework-neutral.
    reader = GridDatasetReader("ns_forced.h5")
    print("dataset :", reader.metadata.dataset_name)
    print("fields  :", reader.metadata.present_fields)
    traj0 = reader.read_trajectory(0)
    for name, fa in traj0.items():
        print(f"  {name:11s} order={fa.tensor_order} role={fa.role:11s} "
              f"shape={fa.data.shape} t={fa.time_varying} x={fa.dim_varying}")

    # Specialized layer: training-ready windows.
    ds = GridWindowDataset("ns_forced.h5", split="train",
                           n_in=4, n_out=1, conditioning="channels")
    sample = ds[0]
    print("input ", tuple(sample["input_fields"].shape), sample["input_field_names"])
    print("output", tuple(sample["output_fields"].shape), sample["output_field_names"])

    # from torch.utils.data import DataLoader
    # dl = DataLoader(ds, batch_size=8, num_workers=4)  # fork-safe: handle reopens per worker
