"""Provenance helpers: recording how a dataset (or a run) was produced.

Small utilities shared by anything that stamps provenance, such as the HDF5
:attr:`~sciml_data_schema.schema.Dataset.provenance` field or a training run's
metrics file.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

__all__ = ["git_revision"]


def git_revision(anchor: str | Path) -> str:
    """Return the short git SHA of the repository containing ``anchor``.

    Never raises: a best-effort stamp for provenance records.

    Parameters
    ----------
    anchor : str or Path
        Any file or directory inside the checkout, typically the caller's
        ``__file__``; ``git`` walks up to the enclosing repository.

    Returns
    -------
    str
        The short SHA, suffixed ``-dirty`` when ``git status`` reports any
        change (untracked files included); ``"unknown"`` outside a git
        checkout or when ``git`` is unavailable.
    """
    root = Path(anchor).resolve()
    root = root if root.is_dir() else root.parent
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"
