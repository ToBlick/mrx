"""Shared helpers for the DESC figure and sweep scripts.

``_log``, ``examples_dir`` and ``locate`` were identical in both
``scripts/desc_figures.py`` and ``scripts/desc_example_sweep.py``.
"""
from __future__ import annotations

import os
import time


def _log(msg: str) -> None:
    """Print a timestamped progress line.

    Args:
        msg: the message.
    """
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def examples_dir(given: str | None) -> str:
    """Where the DESC example files live.

    Args:
        given: an explicit directory, or ``None`` to discover one.

    Returns:
        ``given`` if set, else the installed DESC package's ``examples/``
        if importable, else ``data/``.
    """
    if given:
        return given
    try:
        import desc  # noqa: PLC0415  (optional dependency)
        return os.path.join(os.path.dirname(desc.__file__), "examples")
    except ImportError:
        return "data"


def locate(root: str, name: str) -> str | None:
    """The file of one case, under either naming convention.

    Args:
        root: the directory to look in.
        name: the case name, e.g. ``"HELIOTRON"``.

    Returns:
        The path, or ``None`` if the case is not there.
    """
    for pattern in (f"{name}_output.h5", f"desc_{name}.h5",
                    f"desc_{name}_lowres.h5", f"{name}.h5"):
        hit = os.path.join(root, pattern)
        if os.path.isfile(hit):
            return hit
    return None
