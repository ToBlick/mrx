"""Replay archived relaxation traces through the energy-floor criterion.

Numpy only: ``scripts/relax.py`` is loaded by path and imports mrx only
inside ``main``, so this runs on a login node as
``python test/test_relax_floor.py`` and under pytest on a GPU node.

The archived traces live outside the repository (``MRX_RELAX_ARCHIVE``,
default ``/kfs3/scratch/tblickhan/mrx/out/relax_prelim``). S10 (eta=1e-2)
reached a stationary energy and is the positive control; S07 (13018 steps)
and C1 (3000 steps) were still descending at their last step and must not
trigger the criterion.
"""
import importlib.util
import json
import os
import sys

import numpy as np

ARCHIVE = os.environ.get("MRX_RELAX_ARCHIVE",
                         "/kfs3/scratch/tblickhan/mrx/out/relax_prelim")
FLOOR_TOL_F64 = 1e3 * np.finfo(np.float64).eps
WINDOW = 100


def _relax_module():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts", "relax.py")
    spec = importlib.util.spec_from_file_location("relax_driver", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _energy_trace(tag):
    path = os.path.join(ARCHIVE, tag, f"{tag}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as fh:
        d = json.load(fh)
    return np.asarray(d["arms"]["cg"]["trace"]["E"])


def _first_floor_step(E, window, tol, floor_reached):
    for i in range(1, len(E) + 1):
        if floor_reached(E[:i], window, tol):
            return i
    return None


def test_floor_fires_on_the_stationary_arm():
    relax = _relax_module()
    E = _energy_trace("S10_eta2")
    step = _first_floor_step(E, WINDOW, FLOOR_TOL_F64, relax.energy_floor_reached)
    assert step is not None
    assert 1000 < step < 2500, step
    # and stays fired: the energy is constant to 16 digits from there on
    assert relax.energy_floor_reached(E, WINDOW, FLOOR_TOL_F64)


def test_floor_silent_on_descending_arms():
    relax = _relax_module()
    for tag in ("S07_long", "C1_ls"):
        E = _energy_trace(tag)
        assert _first_floor_step(E, WINDOW, FLOOR_TOL_F64,
                                 relax.energy_floor_reached) is None, tag


def test_floor_needs_a_full_window():
    relax = _relax_module()
    assert not relax.energy_floor_reached([1.0] * WINDOW, WINDOW, 1.0)
    assert relax.energy_floor_reached([1.0] * (WINDOW + 1), WINDOW, 1.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    sys.exit(0)
