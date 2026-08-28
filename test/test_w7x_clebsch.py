"""The W7-X Clebsch initial condition on the GVEC state (``needs_data``:
reads ``data/GVEC_State_final.dat``, W7-X FMM002).

``mrx.geometry.build_sequence(<path>, (8, 16, 8), 3)`` collocates the
finite-beta W7-X map from the state's series, ``load_clebsch`` hands over
the profile splines and ``lambda`` in closed form, and the production
initial condition of ``scripts/relax.py --ic clebsch`` -- ``B = dA'`` from
the histopolated Clebsch potential -- must be divergence-free to round-off
and close to force balance: this is the equilibrium the converging
relaxation references start from. The file is not in the repository (it
sits in ``data/``, a symlink on the cluster); the test skips without it.
"""

import os
import time

import jax.numpy as jnp
import pytest

from mrx.geometry import build_sequence
from mrx.gvec import load_clebsch
from mrx.initial_conditions import (
    clebsch_potential_form,
    divergence_norm,
    potential_two_form,
)
from mrx.nullspace import compute_nullspaces
from mrx.relaxation import compute_force

W7X = "data/GVEC_State_final.dat"
NS, P = (8, 16, 8), 3

pytestmark = [pytest.mark.needs_data,
              pytest.mark.skipif(not os.path.isfile(W7X), reason=f"{W7X} is absent")]


@pytest.fixture(scope="module")
def w7x_seq():
    t0 = time.perf_counter()
    seq, ops = build_sequence(W7X, NS, P)
    seq.set_operators(compute_nullspaces(seq, ops))
    print(f"\n  {os.path.basename(W7X)} {NS} p={P} build + nullspaces: "
          f"{time.perf_counter() - t0:.0f} s")
    return seq


def test_clebsch_ic_is_divergence_free_and_near_force_balance(w7x_seq):
    seq = w7x_seq
    cb = load_clebsch(W7X)
    assert cb["nfp"] == 5
    iota = cb["dchi"][1:] / cb["dPhi"][1:]
    assert -1.1 < iota.min() and iota.max() < -0.9      # W7-X: -0.915 to -1.07

    B, norm, wall = potential_two_form(seq, clebsch_potential_form(cb))
    div = divergence_norm(seq, B)
    F, _, _, _, _ = compute_force(B, seq)
    F_norm = float(seq.l2_norm(F, 2))
    print(f"\n  clebsch IC: ||dA'||_M {norm:.4e}, ||div B|| {div:.2e}, wall-normal "
          f"discarded {wall:.2e}, ||F||_M {F_norm:.3e} at ||B||_M = 1")
    assert jnp.isfinite(B).all()
    assert div <= 10 * seq.tol
    assert wall <= 1e2 * seq.tol
    # Measured 2026-08-28 at (8, 16, 8) p=3 on the state (see the print):
    # ||div B|| 2.9e-16, wall part 0, ||F||_M 1.624e-2 (the gridded export
    # gave 1.888e-2 through the projection route); band at ~1.6x.
    assert F_norm < 2.6e-2
