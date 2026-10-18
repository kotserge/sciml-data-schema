"""Provenance helpers: values that record how a dataset was produced.

The values go into :attr:`~sciml_data_schema.schema.Dataset.provenance`, the free-form
string-valued stamp the file carries verbatim.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

__all__ = ["git_revision"]


def git_revision(anchor: str | Path) -> str:
    """The short git SHA of the checkout containing ``anchor``, ``-dirty`` if it has changes.

    Parameters
    ----------
    anchor : str | Path
        Any path inside the checkout, typically the caller's ``__file__``; ``git`` walks up to
        the enclosing repository from its directory.

    Returns
    -------
    str
        ``<sha>`` or ``<sha>-dirty``; ``"unknown"`` outside a git checkout or when ``git`` is
        not installed. Never raises.
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
