"""The advection-diffusion example as an integration test: build, validate, write, read, window."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import sciml_data_schema as sds

_EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "advection_diffusion.py"


def _load_example() -> ModuleType:
    """The example script, imported as a module from its file path.

    Returns
    -------
    ModuleType
        ``examples/advection_diffusion.py``, executed.
    """
    spec = importlib.util.spec_from_file_location("advection_diffusion_example", _EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_builds_valid_dataset_and_roundtrips(tmp_path: Path) -> None:
    example = _load_example()

    # Tiny: 1 advection x 1 diffusion x 2 seeds = 2 trajectories
    dataset = example.build_dataset(
        n_points=16,
        n_steps=5,
        advections=(1.0,),
        diffusions=(1e-2,),
        n_seeds=2,
    )
    dataset.validate()
    assert dataset.n_trajectories == 2
    assert dataset.present_fields == ["u", "advection", "diffusion"]

    path = sds.write_dataset(dataset, tmp_path / "adv_diff.h5")
    loaded = sds.read_dataset(path)
    assert loaded.dataset_name == "advection_diffusion_1d"

    window = example.windowed_training_view(loaded, n_in=2, n_out=1)
    assert window.input.shape == (2, 16)
    assert window.target.shape == (1, 16)
    assert window.params.shape == (2,)
