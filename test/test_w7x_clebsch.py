"""The W7-X Clebsch initial condition (``needs_data``: reads ``MRX_W7X_FILE``).

``mrx.geometries.build_sequence('w7x-fmm002', (8, 16, 8), 3)`` fits the
finite-beta W7-X map, ``load_clebsch`` reads GVEC's ``dPhi_dr``, ``dchi_dr``
and ``lambda`` from the same file and ``clebsch_form`` rebuilds
``sqrt(g) B^i`` from them. The projected field must be divergence-free
after ``leray_clean`` and close to force balance: this is the equilibrium
the converging relaxation references start from. The GVEC export is named
by the environment variable ``MRX_W7X_FILE`` (the finite-beta
``w7x_fmm002_clebsch_mrx.h5``); unset, the test skips (CI has no data).
On a GPU node pass it through ``EXTRA_ENV`` (slurm/README.md).
"""

import os
import time

import jax.numpy as jnp
import pytest

import mrx.gvec
from mrx.geometries import build_sequence
from mrx.gvec import load_clebsch
from mrx.initial_conditions import (
    clebsch_form,
    divergence_norm,
    leray_clean,
    project_reference_two_form,
)
from mrx.nullspace import compute_nullspaces
from mrx.relaxation import compute_force

pytestmark = pytest.mark.needs_data

GEOMETRY = "w7x-fmm002"
NS, P = (8, 16, 8), 3


@pytest.fixture(scope="module")
def w7x_file():
    path = os.environ.get("MRX_W7X_FILE")
    if path is None:
        pytest.skip("MRX_W7X_FILE is unset; point it at w7x_fmm002_clebsch_mrx.h5")
    if not os.path.exists(path):
        pytest.skip(f"MRX_W7X_FILE={path} does not exist")
    return path


@pytest.fixture(scope="module")
def w7x_seq(w7x_file):
    # build_sequence resolves the name through mrx.gvec's table; the test
    # is keyed on the file path, so the table entry is pointed at it.
    mp = pytest.MonkeyPatch()
    mp.setattr(mrx.gvec, "DATA_DIR", os.path.dirname(w7x_file))
    mp.setitem(mrx.gvec.GVEC_GEOMETRIES, GEOMETRY, os.path.basename(w7x_file))
    t0 = time.perf_counter()
    seq, ops = build_sequence(GEOMETRY, NS, P)
    seq.set_operators(compute_nullspaces(seq, ops))
    mp.undo()
    print(f"\n  {GEOMETRY} {NS} p={P} build + nullspaces: {time.perf_counter() - t0:.0f} s")
    return seq


def test_clebsch_ic_is_divergence_free_and_near_force_balance(w7x_seq, w7x_file):
    seq = w7x_seq
    cb = load_clebsch(w7x_file, seq.basis_0.types)
    assert cb["nfp"] == 5
    assert cb["closed_axes"] == []          # angles sampled half-open in this file
    assert cb["iota_spread"] < 1e-2         # dchi/dPhi is a flux function

    B_raw, norm = project_reference_two_form(seq, clebsch_form(cb))
    div_raw = divergence_norm(seq, B_raw)
    B, moved = leray_clean(seq, B_raw)
    div = divergence_norm(seq, B)
    F, _, _, _, _ = compute_force(B, seq)
    F_norm = float(seq.l2_norm(F, 2))
    print(f"\n  clebsch IC: ||B||_M raw {norm:.4e}, ||div B|| {div_raw:.2e} -> {div:.2e}, "
          f"moved {moved:.2e}, ||F||_M {F_norm:.3e} at ||B||_M = 1")
    assert jnp.isfinite(B).all()
    assert div <= 10 * seq.tol
    # Measured 2026-08-26 at (8, 16, 8) p=3 (see the print): the L2
    # projection carries ||div B|| 6.5e-2, leray_clean moves the field by
    # 2.90e-3 and leaves ||F||_M 1.888e-2; bands at ~1.6x.
    assert moved < 5e-3
    assert F_norm < 3e-2
