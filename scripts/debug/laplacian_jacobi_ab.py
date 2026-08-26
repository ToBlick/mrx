"""Three-arm A/B for the k>=1 Laplacian Jacobi diagonal.

Arms, all feeding the SAME shifted-Jacobi preconditioner and the same CG:

  none   -- no preconditioner (context: is Jacobi doing anything at all?)
  probe  -- diag(L_k) by n_ext operator applies. Exact, O(N) applies, the
            oracle; unusable at production resolution.
  stiff  -- diag(E S_k E^T) only, i.e. the weak term DROPPED. The cheap
            fallback. Undefined at k=3, where S_3 = 0 and the Laplacian IS the
            weak term.
  closed -- the closed-form diagonal (stiffness closed form + tensorized weak
            term + exact applies on the coupled rows).
  rescale      -- closed, plus the leading-order repair of the INNER Lam split
                  (MRX_LAPLACIAN_DIAG_RESCALE=upper). Free; restores the
                  Kronecker mass model's diag(M~) = diag(M) upper level.
  rescale_both -- also repairs the two split Sig inside the lower mass inverse,
                  resampled onto the upper grid (=both).
  exact   -- ORACLE: the same expansion with the inner scalings kept exact
             (MRX_LAPLACIAN_DIAG_SPLIT=exact). Dense, so A/B sizes only. Splits
             the closed-vs-probe gap in two: probe-exact is the mass-model
             error, exact-closed is the rank-1 split error.
  taylor  -- the inner Lam expanded LOCALLY to first order about the row
             instead of fitted globally (=taylor1). Same cost class as closed.
  codiff  -- no expansion at all: diag(W)_i = ||delta phi_i||^2 by quadrature,
             i.e. the k=0 stiffness energy of star(phi_i) = phi_i/J. k=3 only.
             Differentiates the k=3 basis directly, so it needs a second
             derivative and a dJ pass.
  transfer     -- the same energy, but star(phi_i) is represented in V_0 FIRST
                  (Greville collocation) and the gradient taken there: no
                  derivative of the k=3 basis, no dJ, just a k=0 stiffness
                  energy. k=3 only. Exact collocation inverse.
  transfer_b2/ -- as transfer, with the collocation inverse truncated to
  transfer_b4     bandwidth 2 / 4, i.e. a genuinely LOCAL transfer.

Usage:
    python scripts/debug/laplacian_jacobi_ab.py --geometry toroid --ns 8,16,8
    python scripts/debug/laplacian_jacobi_ab.py --geometry w7x    --ns 8,16,8
    python scripts/debug/laplacian_jacobi_ab.py --geometry w7x \
        --arms probe,closed,rescale,rescale_both
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax


import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx  # noqa: E402
import mrx.operators as op  # noqa: E402
from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.local_assembly import build_stiffness_diagonal  # noqa: E402
from mrx.mappings import toroid_map  # noqa: E402

# Shift used by the A/B. eps > 0 regularizes the Laplacian, which matters where
# it is singular (k=0 free has constants in the nullspace; k=1,2 carry harmonic
# fields for betti (1,1,0,0)). At k=3 the weak term is SPD on its own -- D_2 is
# surjective onto V_3, so D_2^T M_3 w = 0 forces w = 0 -- so eps=0 is the honest
# operator there. Set with --eps.
EPS = 1e-4

# set_map is the memory hot spot on W7-X: the default (0 = one vmap over every
# quadrature point) OOMs an 80 GB H100 from 12x24x12 up, inside the spline map
# evaluation, long before any preconditioner is built.
mrx.MAP_BATCH_SIZE_INNER = int(os.environ.get("W7X_MAP_BATCH", "256"))

# Arm -> MRX_LAPLACIAN_DIAG_RESCALE. Every other arm runs with the repair off.
RESCALE_ARMS = {"rescale": "upper", "rescale_both": "both"}
# Arm -> MRX_LAPLACIAN_DIAG_SPLIT. Every other arm uses the geometric rank-1
# split. 'exact' is the oracle that isolates the split error from the mass-model
# error; it is dense, so it is A/B-resolution only.
SPLIT_ARMS = {"exact": "exact", "taylor": "taylor1", "codiff": "codiff",
              "transfer": "transfer", "transfer_b2": "transfer_2",
              "transfer_b4": "transfer_4",
              "star": "transfer_star",
              "tfree": "transfer_free", "tfree_b2": "transfer_free_2"}


def build_sequence(geometry, ns, p):
    seq = DeRhamSequence(ns, (p,) * 3, 2 * p, ("clamped", "periodic", "periodic"),
                         polar=True, tol=1e-12, maxiter=1000,
                         betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    if geometry == "toroid":
        seq.set_map(toroid_map(epsilon=1 / 3, R0=1.0))
    elif geometry == "w7x":
        from w7x_geometry import build_w7x_map  # noqa: PLC0415
        map_func, _ = build_w7x_map(map_ns=ns, p=p)
        seq.set_map(map_func)
        jac = np.asarray(seq.geometry.jacobian_j)
        if not np.isfinite(jac).all() or jac.min() <= 0:
            raise RuntimeError(
                f"W7-X geometry is degenerate: finite={np.isfinite(jac).all()} "
                f"min(jac)={jac.min():.3e}")
    else:
        raise ValueError(geometry)
    ops = op.assemble_incidence_operators(seq)
    ops = op.assemble_mass_jacobi_preconditioner(seq, ops, ks=(0, 1, 2, 3))
    seq.set_operators(ops)
    return seq, ops


def stiffness_only_diagonal(seq, ops, k, dirichlet):
    """diag(E S_k E^T): bulk rows closed form, coupled rows by exact apply."""
    raw = np.asarray(build_stiffness_diagonal(seq, k))
    e = getattr(seq, f"e{k}_dbc" if dirichlet else f"e{k}")
    rows, cols, vals = (np.asarray(e.rows), np.asarray(e.cols), np.asarray(e.vals))
    n_ext = int(e.forward_shape[0])
    counts = np.bincount(rows, minlength=n_ext)
    diag = np.zeros(n_ext)
    single = counts[rows] == 1
    diag[rows[single]] = (vals[single] ** 2) * raw[cols[single]]
    coupled = np.flatnonzero(counts > 1)
    if coupled.size:
        size = int(getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}"))
        op.apply_stiffness(seq, ops, jnp.zeros(size), k, dirichlet=dirichlet)
        diag[coupled] = np.asarray(jax.lax.map(
            lambda i: op.apply_stiffness(
                seq, ops, jnp.zeros(size).at[i].set(1.0), k, dirichlet=dirichlet)[i],
            jnp.asarray(coupled)))
    return diag


def radial_error_profile(seq, k, dbc, rel):
    """Where the diagonal error sits, radially.

    The free/dbc asymmetry in the W7-X max error points at the OUTER radial
    boundary rather than the axis: det(DF) -> 0 at the last knot, so the metric
    weight degenerates exactly on the rows a Dirichlet BC would drop.  Returns
    the median relerr on the first two and last two radial indices against the
    interior, plus where the worst row actually is.
    """
    e = getattr(seq, f"e{k}_dbc" if dbc else f"e{k}")
    rows, cols = np.asarray(e.rows), np.asarray(e.cols)
    n_ext = int(e.forward_shape[0])
    counts = np.bincount(rows, minlength=n_ext)
    shapes = [tuple(int(s) for s in sh) for sh in getattr(seq, f"basis_{k}").shape]
    sizes = [int(np.prod(sh)) for sh in shapes]
    starts = np.cumsum([0] + sizes)

    single = counts[rows] == 1
    r_s, c_s = rows[single], cols[single]
    comp = np.searchsorted(starts[1:], c_s, side="right")
    loc = c_s - starts[comp]
    nt = np.array([sh[1] for sh in shapes])[comp]
    nz = np.array([sh[2] for sh in shapes])[comp]
    nr = np.array([sh[0] for sh in shapes])[comp]
    i_r = loc // (nt * nz)

    ir_full = np.full(n_ext, -1)          # -1 marks the coupled (polar) rows
    nr_full = np.full(n_ext, -1)
    ir_full[r_s] = i_r
    nr_full[r_s] = nr

    def med(mask):
        return float(np.median(rel[mask])) if mask.any() else float("nan")

    depth = np.where(ir_full >= 0, np.minimum(ir_full, nr_full - 1 - ir_full), -2)
    out = {
        "coupled": med(ir_full < 0),
        "outer0": med(ir_full == nr_full - 1),
        "outer1": med(ir_full == nr_full - 2),
        "inner0": med(ir_full == 0),
        "inner1": med(ir_full == 1),
        "interior": med(depth >= 2),
    }
    worst = int(np.argmax(rel))
    out["worst"] = {"rel": float(rel[worst]),
                    "i_r": int(ir_full[worst]), "n_r": int(nr_full[worst])}
    return out


def _histopolation_1d(seq, axis, nq=8):
    """``H[i, j] = int_{cell_i} Lam0_j`` over the Greville cells of one axis.

    The V_0 -> V_3 direction of the transfer, and the easy one: a V_0 function's
    k=3 DOFs ARE its integrals over the Greville cells, so this is histopolation
    -- banded, fixed by the knot vector, metric-free, and with no inverse
    anywhere (the V_3 -> V_0 direction needed a collocation solve).
    """
    lam, dlam = seq.basis_0.Λ[axis], seq.basis_0.dΛ[axis]
    grev = np.asarray(lam.greville_points())
    edges = (np.concatenate([grev, [grev[0] + 1.0]])
             if lam.type == "periodic" else grev)
    xg, wg = np.polynomial.legendre.leggauss(nq)
    H = np.zeros((int(dlam.n), int(lam.n)))
    for i in range(int(dlam.n)):
        a, b = edges[i], edges[i + 1]
        pts = np.mod(0.5 * (b - a) * xg + 0.5 * (a + b), 1.0)
        pts = np.clip(pts, 1e-8, 1 - 1e-8) if lam.type != "periodic" else pts
        vals = np.asarray(lam.collocation_matrix(jnp.asarray(pts)))  # (nq, n0)
        H[i] = 0.5 * (b - a) * (wg @ vals)
    return H


def auxiliary_space_apply(seq, ops, k, dbc, mass_jacobi):
    """``P_3 = M_3^-1 Q P_0 Q^T M_3^-1`` -- restrict to V_0, apply the k=0
    preconditioner, prolong back.

    ``Q = (x)_a H_a`` is the metric-free histopolation transfer; ``P_0`` is the
    k=0 Jacobi inverse diagonal, which is EXACT in closed form here (4.3e-16);
    ``M_3^-1`` is the k=3 mass Jacobi, also exact.  The ``M_3^-1`` flanks are
    what the Rayleigh-quotient argument requires -- ``W_3 = M_3 (D M_2^-1 D^T)
    M_3``, so the partner of ``S_0`` is the bracket, not ``W_3`` itself.

    BC flip: the partner of k=3 dirichlet is the FREE k=0 preconditioner.
    """
    q_raw = np.kron(np.kron(_histopolation_1d(seq, 0), _histopolation_1d(seq, 1)),
                    _histopolation_1d(seq, 2))
    e0 = seq.e0 if dbc else seq.e0_dbc          # <- the flip
    e0_mat = np.zeros((int(e0.forward_shape[0]), q_raw.shape[1]))
    e0_mat[np.asarray(e0.rows), np.asarray(e0.cols)] = np.asarray(e0.vals)
    # V_3's extraction is NOT the identity (896 raw vs 768 extracted here), so
    # the transfer has to land in extracted coordinates on both sides.
    e3 = getattr(seq, "e3_dbc" if dbc else "e3")
    e3_mat = np.zeros((int(e3.forward_shape[0]), q_raw.shape[0]))
    e3_mat[np.asarray(e3.rows), np.asarray(e3.cols)] = np.asarray(e3.vals)
    q = jnp.asarray(e3_mat @ q_raw @ e0_mat.T)            # (n_3ext, n_0ext)

    p0 = jnp.asarray(op._laplacian_diaginv(seq, ops, 0, dirichlet=(not dbc)))
    d3 = jnp.asarray(mass_jacobi)                         # 1/diag(M_3)

    def minv(v):
        w = d3 * v
        return d3 * (q @ (p0 * (q.T @ w)))
    return minv


def pcg(a_apply, b, minv, tol=1e-8, maxiter=3000):
    x = jnp.zeros_like(b)
    r = b
    z = minv(r)
    p = z
    rz = float(r @ z)
    nb = float(jnp.linalg.norm(b))
    for it in range(1, maxiter + 1):
        ap = a_apply(p)
        alpha = rz / float(p @ ap)
        x = x + alpha * p
        r = r - alpha * ap
        if float(jnp.linalg.norm(r)) <= tol * nb:
            return it
        z = minv(r)
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return maxiter


def main():
    global EPS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="toroid", choices=("toroid", "w7x"))
    ap.add_argument("--ns", default="8,16,8")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--arms", default="none,probe,stiff,closed")
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--out", default=None,
                    help="write the results to this JSON file")
    ap.add_argument("--eps", type=float, default=1e-4,
                    help="mass shift; the operator is L_k + eps M_k and the "
                         "Jacobi arms use 1/(diag L + eps/diag M). eps=0 is the "
                         "true Laplacian (safe at k=3, singular at k=0 free).")
    ap.add_argument("--exact-rings", type=int, default=0,
                    help="take the n innermost radial rings exactly, by one "
                         "apply each (MRX_LAPLACIAN_DIAG_EXACT_RINGS)")
    ap.add_argument("--build-only", action="store_true",
                    help="time the diagonal builds and compare them; skip CG. "
                         "This is the scaling question: the probe costs one "
                         "apply per extracted row, the closed form does not.")
    cli = ap.parse_args()

    ns = tuple(int(v) for v in cli.ns.split(","))
    os.environ["MRX_LAPLACIAN_DIAG_EXACT_RINGS"] = str(cli.exact_rings)
    EPS = cli.eps
    arms = cli.arms.split(",")
    seq, ops = build_sequence(cli.geometry, ns, cli.p)
    print(f"geometry={cli.geometry} ns={ns} p={cli.p} eps={EPS}", flush=True)
    print(f"{'k':>2} {'dbc':>5} {'n':>7} " +
          " ".join(f"{a:>20}" for a in arms), flush=True)

    results = {"geometry": cli.geometry, "ns": list(ns), "p": cli.p,
               "eps": EPS, "arms": arms, "build_only": cli.build_only,
               "rows": []}
    mass_jacobi = {}
    for k in (int(v) for v in cli.ks.split(",")):
        for dbc in (False, True):
            n = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
            rhs = jax.random.normal(jax.random.PRNGKey(97 * k + dbc), (n,))

            def a_apply(x, k=k, dbc=dbc):
                return (op.apply_hodge_laplacian_approx(seq, ops, x, k, dirichlet=dbc)
                        + EPS * op.apply_mass_matrix(seq, ops, x, k, dirichlet=dbc))

            mass_jacobi[(k, dbc)] = 1.0 / np.asarray(
                op._mass_diaginv(seq, ops, k, dbc))
            cells = []
            diags = {}
            record = {"k": k, "dbc": dbc, "n": n}
            for arm in arms:
                if arm == "none" and cli.build_only:
                    continue
                if arm == "stiff" and k == 3:
                    cells.append(f"{'n/a (S_3 = 0)':>20}")
                    record[arm] = None
                    continue
                t0 = time.perf_counter()
                if arm == "aux":
                    minv = auxiliary_space_apply(
                        seq, ops, k, dbc, mass_jacobi[(k, dbc)])
                    t_build = time.perf_counter() - t0
                    it = pcg(a_apply, rhs, minv)
                    cells.append(f"{t_build:8.1f}s {it:5d} it")
                    record[arm] = {"build_s": t_build, "iters": it}
                    continue
                if arm == "none":
                    shifted = jnp.ones(n)
                else:
                    if arm == "stiff":
                        diag = stiffness_only_diagonal(seq, ops, k, dbc)
                    else:
                        os.environ["MRX_LAPLACIAN_DIAG_PROBE"] = (
                            "1" if arm == "probe" else "0")
                        os.environ["MRX_LAPLACIAN_DIAG_RESCALE"] = (
                            RESCALE_ARMS.get(arm, "none"))
                        os.environ["MRX_LAPLACIAN_DIAG_SPLIT"] = (
                            SPLIT_ARMS.get(arm, "geometric"))
                        diag = 1.0 / np.asarray(op._hodge_diaginv(seq, ops, k, dbc))
                    shifted = jnp.asarray(1.0 / (diag + EPS * mass_jacobi[(k, dbc)]))

                def minv(v, d=shifted):
                    return d * v

                t_build = time.perf_counter() - t0
                if cli.build_only:
                    diags[arm] = np.asarray(shifted)
                    cells.append(f"{t_build:8.1f}s")
                    record[arm] = {"build_s": t_build}
                    continue
                it = pcg(a_apply, rhs, minv)
                cells.append(f"{t_build:8.1f}s {it:5d} it")
                record[arm] = {"build_s": t_build, "iters": it}
            if cli.build_only and "probe" in diags:
                for arm, d in diags.items():
                    if arm == "probe":
                        continue
                    rel = np.abs(d - diags["probe"]) / np.abs(diags["probe"])
                    record[f"{arm}_vs_probe"] = {
                        "median": float(np.median(rel)),
                        "p90": float(np.percentile(rel, 90)),
                        "max": float(rel.max()),
                        "radial": radial_error_profile(seq, k, dbc, rel)}
                    prof = record[f"{arm}_vs_probe"]["radial"]
                    cells.append(f"  {arm}: med={np.median(rel):.2e} "
                                 f"p90={np.percentile(rel, 90):.2e} "
                                 f"max={rel.max():.2e} | med by ring: "
                                 f"in0={prof['inner0']:.1e} in1={prof['inner1']:.1e} "
                                 f"mid={prof['interior']:.1e} "
                                 f"out1={prof['outer1']:.1e} out0={prof['outer0']:.1e} "
                                 f"cpl={prof['coupled']:.1e} "
                                 f"worst@i_r={prof['worst']['i_r']}/{prof['worst']['n_r']}")
            print(f"{k:>2} {dbc!s:>5} {n:>7} " + " ".join(cells), flush=True)
            results["rows"].append(record)
            if cli.out:
                os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
                with open(cli.out, "w") as fh:
                    json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
