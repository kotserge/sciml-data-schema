"""Provenance helpers: recording how a dataset (or a run) was produced.

Small utilities shared by anything that stamps provenance — the HDF5
:attr:`~sciml_data_schema.schema.Dataset.provenance` field, a training run's
metrics file. Kept here because provenance is part of what this package owns.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

__all__ = ["git_revision"]


def git_revision(anchor: str | Path) -> str:
    """Short git SHA of the repo containing ``anchor`` (``-dirty`` if uncommitted).

    ``anchor`` is any path inside the checkout — typically the caller's
    ``__file__``; ``git`` walks up to the enclosing repo, so the exact depth does
    not matter. A best-effort stamp: returns ``"unknown"`` outside a git checkout
    (or if ``git`` is unavailable), never raising.
    """
    root = Path(anchor).resolve()
    root = root if root.is_dir() else root.parent
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"
