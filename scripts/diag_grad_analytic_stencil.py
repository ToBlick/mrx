"""Validate the analytic inverse-free polar grad stencil G_0 against the oracle.

The shipped ``apply_incidence_matrix(., 0)`` computes the true polar grad as
``G_0 = Gram_1^{-1} (E_1 sp_0 E_0^T)`` -- correct, but with a small dense
``S x S`` axis-block inverse built at ASSEMBLY. This script builds ``G_0`` purely
from the incidence pattern and the polar mapping coefficients ``xi`` (coefficient
differences + ``xi`` weights, NO inverse) and checks it is BIT-EXACT against that
oracle for all four (dirichlet_in, dirichlet_out) BC combinations, plus the
transpose and the ``curl . grad = 0`` nilpotency.

CPU, no mass assembly. Run: python scripts/diag_grad_analytic_stencil.py
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
jax.config.update("jax_enable_x64", True)

from mrx.derham_sequence import DeRhamSequence
from mrx.extraction_operators import get_xi
from mrx.mappings import rotating_ellipse_map
from mrx.operators import (
    assemble_incidence_operators,
    _ensure_extraction_operators,
    apply_incidence_matrix,
    build_grad_stencil_g0,   # the analytic grad builder (operators.py)
    build_curl_stencil_g1,   # the analytic curl builder (operators.py)
    _incidence_components,
    _derivative_extraction,
    _mass_extraction,
    _build_inc_gram_inv,     # independent Gram oracle (no longer precomputed in core)
)


def gram_oracle_apply(seq, ops, v, k, din, dout, transpose, gram_inv):
    """Independent OLD Gram-path derivative: Gram_{k+1}^{-1} (E_out sp_k E_in^T),
    bypassing the (now analytic) apply_incidence short-circuit -- ground truth.
    ``gram_inv`` is built directly via ``_build_inc_gram_inv`` (the core no longer
    precomputes/stores it)."""
    sp, sp_T = _incidence_components(ops, k)
    e_in, e_in_T, e_out, e_out_T = _derivative_extraction(ops, k, din, dout)
    if transpose:
        w = gram_inv @ v if gram_inv is not None else v
        return e_in @ (sp_T @ (e_out_T @ w))
    y = e_out @ (sp @ (e_in_T @ v))
    return gram_inv @ y if gram_inv is not None else y

TYPES = ("clamped", "periodic", "periodic")
BETTI = (1, 1, 0, 0)


def build_dense(fn, n_in):
    return np.stack(
        [np.asarray(jax.device_get(fn(jnp.zeros((n_in,), jnp.float64).at[j].set(1.0))))
         for j in range(n_in)], axis=1)


def _gram_dev(ops, space, dbc):
    """||E_space E_space^T - I|| (0 -> unitary extraction)."""
    e, e_T = _mass_extraction(ops, space, dbc)
    if e is None:
        return float("nan")
    n = int(e.shape[0]) if e.shape[0] < e.shape[1] else int(e.shape[1])
    G = build_dense(lambda u: e @ (e_T @ u), n)
    return float(np.max(np.abs(G - np.eye(n))))


def _check(seq, ops, xi, k, builder, nin_attr, nout_attr, name):
    """Bit-exact check of an analytic stencil builder vs the Gram oracle + the
    shipped apply, for all 4 BC pairs (fwd+transpose). Returns worst diff."""
    worst = 0.0
    for din in (False, True):
        for dout in (False, True):
            nin = int(getattr(seq, nin_attr + "_dbc") if din else getattr(seq, nin_attr))
            gi = _build_inc_gram_inv(seq, ops, k + 1, dout)   # independent oracle
            g = np.asarray(builder(seq, xi, din, dout).todense())
            oracle = build_dense(
                lambda v: gram_oracle_apply(seq, ops, v, k, din, dout, False, gi), nin)
            shipped = build_dense(
                lambda v: apply_incidence_matrix(
                    seq, ops, v, k, dirichlet_in=din, dirichlet_out=dout), nin)
            err = float(np.max(np.abs(g - oracle))) if g.shape == oracle.shape else 9e9
            err_ship = float(np.max(np.abs(shipped - oracle)))
            nout = g.shape[0]
            oracle_T = build_dense(
                lambda v: gram_oracle_apply(seq, ops, v, k, din, dout, True, gi), nout)
            shipped_T = build_dense(
                lambda v: apply_incidence_matrix(
                    seq, ops, v, k, dirichlet_in=din, dirichlet_out=dout,
                    transpose=True), nout)
            errT = float(np.max(np.abs(g.T - oracle_T)))
            errT_ship = float(np.max(np.abs(shipped_T - oracle_T)))
            worst = max(worst, err, errT, err_ship, errT_ship)
            print(f"  [{name}] din={int(din)} dout={int(dout)} shape={g.shape}  "
                  f"max|G-oracle|={err:.2e} |G^T-oracle^T|={errT:.2e}  "
                  f"shipped fwd={err_ship:.2e} T={errT_ship:.2e}")
    return worst


def run(NS, P, polar=True):
    seq = DeRhamSequence(NS, (P, P, P), 2 * P, TYPES, polar=polar, betti_numbers=BETTI)
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()
    seq.set_map(rotating_ellipse_map(eps=1.0 / 3.0, kappa=1.2, R0=1.0, nfp=3))
    ops = seq.get_operators()
    ops = assemble_incidence_operators(seq, operators=ops, ks=(0, 1, 2))
    ops = _ensure_extraction_operators(seq, ops)
    xi = np.asarray(get_xi(seq.ns[1]))

    tag = "POLAR" if polar else "non-polar"
    print(f"\n========== {tag}  NS={NS} p={P} ==========")
    print(f"  Gram dev ||E E^T - I||: V1={_gram_dev(ops,1,False):.2e} "
          f"V2={_gram_dev(ops,2,False):.2e} V3={_gram_dev(ops,3,False):.2e} "
          f"(V3~0 -> div already matrix-free)")
    worst = 0.0
    worst = max(worst, _check(seq, ops, xi, 0, build_grad_stencil_g0, "n0", "n1", "grad G0"))
    worst = max(worst, _check(seq, ops, xi, 1, build_curl_stencil_g1, "n1", "n2", "curl G1"))
    # nilpotency curl.grad and div.curl on the analytic stencils (matched BCs)
    for dbc in (False, True):
        g0 = np.asarray(build_grad_stencil_g0(seq, xi, dbc, dbc).todense())
        g1 = np.asarray(build_curl_stencil_g1(seq, xi, dbc, dbc).todense())
        key = jax.random.PRNGKey(0)
        w_cg = w_dc = 0.0
        for _ in range(4):
            key, ka, kb = jax.random.split(key, 3)
            v0 = np.asarray(jax.random.normal(ka, (g0.shape[1],), jnp.float64))
            gv = g0 @ v0
            cgv = g1 @ gv
            w_cg = max(w_cg, float(np.linalg.norm(cgv)) / max(np.linalg.norm(gv), 1e-300))
            v1 = np.asarray(jax.random.normal(kb, (g1.shape[1],), jnp.float64))
            cv = g1 @ v1
            dcv = apply_incidence_matrix(seq, ops, jnp.asarray(cv), 2,
                                         dirichlet_in=dbc, dirichlet_out=dbc)
            w_dc = max(w_dc, float(jnp.linalg.norm(dcv)) / max(np.linalg.norm(cv), 1e-300))
        print(f"  nilpotency ({'dbc' if dbc else 'free'}): curl.grad={w_cg:.2e}  div.curl={w_dc:.2e}")
    return worst


def main():
    t0 = time.perf_counter()
    w1 = run((6, 8, 4), 3, polar=True)
    w2 = run((4, 4, 3), 2, polar=True)
    print(f"\n[done] worst bit-exact diff = {max(w1, w2):.2e} "
          f"(target < 1e-12)   [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
