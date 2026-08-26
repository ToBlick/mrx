"""How much does plain Jacobi cost on the k=0 Laplacian?

If Jacobi is acceptable there, the entire k=0 tensor-Hodge machinery -- core
Schur, core_coupling, the FD/modal atoms, the bundled-profile helpers -- can go.

The diagonal is FULLY closed-form here: L_0 = S_0 (there is no lower term at
k=0), so diag(L_0) = diag(S_0) comes from one sum-factorized contraction
(``build_stiffness_diagonal``, verified to 5.4e-16). Only the O(n_z) coupled
polar rows still need an apply, exactly as for the mass Jacobi.

dbc only: the free/Neumann K_0 is singular (constants) and plain PCG diverges.
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.local_assembly import build_stiffness_diagonal
from mrx.mappings import rotating_ellipse_map, toroid_map
from mrx.operators import (_apply_k0_tensor_hodge_preconditioner, _core_size,
                           apply_stiffness, assemble_incidence_operators,
                           assemble_tensor_laplacian_preconditioner)

ap = argparse.ArgumentParser()
ap.add_argument("--ns", type=int, nargs=3, default=(8, 16, 16))
ap.add_argument("--tol", type=float, default=1e-10)
ap.add_argument("--maxit", type=int, default=20000)
a = ap.parse_args()
NS, P = tuple(a.ns), 3
mrx.MAP_BATCH_SIZE_INNER = 256
TYPES = ("clamped", "periodic", "periodic")
print(f"k=0 Laplacian: Jacobi vs tensor-Hodge   ns={NS}\n", flush=True)


def pcg(A, b, M, tol, maxit):
    x = jnp.zeros_like(b)
    r = b - A(x)
    z = M(r)
    q = z
    rz = float(r @ z)
    nb = float(jnp.linalg.norm(b))
    jax.block_until_ready(z)
    t0 = time.perf_counter()
    for i in range(1, maxit + 1):
        Aq = A(q)
        den = float(q @ Aq)
        if den <= 0:
            return -i, time.perf_counter() - t0
        al = rz / den
        x = x + al * q
        r = r - al * Aq
        if float(jnp.linalg.norm(r)) / nb < tol:
            jax.block_until_ready(x)
            return i, time.perf_counter() - t0
        z = M(r)
        rzn = float(r @ z)
        q = z + (rzn / rz) * q
        rz = rzn
    return maxit, time.perf_counter() - t0


def extracted_stiffness_diag(seq, ops, e, n):  # dbc throughout
    """diag(E S E^T): closed form on bulk rows, O(n_z) applies on coupled rows."""
    d_raw = np.asarray(build_stiffness_diagonal(seq, 0))
    rows = np.asarray(e.rows)
    cols = np.asarray(e.cols)
    vals = np.asarray(e.vals)
    counts = np.bincount(rows, minlength=n)
    diag = np.zeros(n)
    single = counts[rows] == 1
    diag[rows[single]] = (vals[single] ** 2) * d_raw[cols[single]]
    coupled = np.flatnonzero(counts > 1)
    for r0 in coupled:
        v = np.zeros(n)
        v[r0] = 1.0
        diag[r0] = float(jnp.asarray(v) @ apply_stiffness(
            seq, ops, jnp.asarray(v), 0, dirichlet=True))
    return diag, coupled.size


for gname, mk in (("toroid", lambda: toroid_map(epsilon=1 / 3, R0=1.0)),
                  ("rot-ellipse", lambda: rotating_ellipse_map(
                      eps=0.33, kappa=1.5, nfp=3))):
    seq = DeRhamSequence(NS, (P,) * 3, 2 * P, TYPES, polar=True, tol=1e-12,
                         maxiter=1000, betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    seq.set_map(mk())
    ops = assemble_incidence_operators(seq)
    seq.set_operators(ops)
    size = int(seq.n0_dbc)
    cs = _core_size(seq)

    def K(x):
        return apply_stiffness(seq, ops, x, 0, dirichlet=True)
    K(jnp.zeros(size))

    t0 = time.perf_counter()
    d_full, n_coupled = extracted_stiffness_diag(seq, ops, seq.e0_dbc, size)
    t_jac = time.perf_counter() - t0
    dinv = jnp.asarray(1.0 / np.where(np.abs(d_full) > 0, d_full, 1.0))

    t0 = time.perf_counter()
    ops2 = assemble_tensor_laplacian_preconditioner(
        seq, ops, ks=(0,), rank=1,
        cp_kwargs={"maxiter": 100, "tol": 1e-9, "ridge": 1e-12})
    t_tensor = time.perf_counter() - t0

    rng = np.random.default_rng(0)
    b = jnp.asarray(rng.standard_normal(size))
    it_j, tt_j = pcg(K, b, lambda r: dinv * r, a.tol, a.maxit)
    it_t, tt_t = pcg(K, b, lambda r: _apply_k0_tensor_hodge_preconditioner(
        seq, ops2, r, dirichlet=True), a.tol, a.maxit)
    print(f"  {gname:12s} n={size:>6} coupled_rows={n_coupled}", flush=True)
    print(f"    jacobi (closed form) its={it_j:>6}  solve={tt_j * 1e3:9.1f} ms  "
          f"setup={t_jac:6.2f} s", flush=True)
    print(f"    tensor-Hodge         its={it_t:>6}  solve={tt_t * 1e3:9.1f} ms  "
          f"setup={t_tensor:6.2f} s", flush=True)
    print(f"    ratio  its={it_j / max(it_t, 1):6.1f}x  "
          f"solve={tt_j / max(tt_t, 1e-9):6.1f}x\n", flush=True)
