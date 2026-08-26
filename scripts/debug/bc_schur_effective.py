"""Where does the ~0.1 boundary scale come from?  The DtN test.

§17.3 of `docs/research/natural_bc_coefficient_handoff.md` claims the atom's
boundary row should carry not `L`'s surface term `alpha` but the SCHUR
COMPLEMENT of `L` onto that row -- `alpha` minus what the interior DtN takes
back.  That is a testable statement, and this script tests it directly.

For the outer `d` radial rings `R` (interior `I`):

    B_raw   = L[R, R]                                   -- the diagonal block
    B_schur = L[R,R] - L[R,I] inv(L[I,I]) L[I,R]        -- the DtN-reduced one
    A(s)    = inv(P(s))[R, R]                           -- what the atom asserts

`P` is the preconditioner's action, so `inv(P)` is exactly the operator the
preconditioner inverts.  With `outer_rings = extra_rings = 0` that is the atom
itself (plus the polar core block, which is always dense).

PREDICTIONS if §17.3 is right:
  * matching A(s) to B_raw   picks  s ~ 1     (alpha IS L's diagonal surface
    term, so the diagonals agree by construction at s = 1);
  * matching A(s) to B_schur picks  s ~ 0.1   (the measured optimum);
  * and the best `s` for B_schur FALLS as the slab `d` deepens, because more of
    the DtN is then inside `R` rather than folded into the complement.
If instead both pick the same `s`, §17.3 is wrong and the factor is not a
Schur/DtN reduction.

Small meshes only -- this forms dense `L` (one apply per DOF).
"""
from __future__ import annotations

import argparse
import os
import sys


import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import mrx.operators as op  # noqa: E402
from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.metric_lumping_laplacian import (  # noqa: E402
    MetricLumpingLaplacian, core_rows,
)
from mrx.mappings import cylinder_map, rotating_ellipse_map, toroid_map  # noqa: E402


def build_sequence(geometry, ns, p):
    seq = DeRhamSequence(ns, (p,) * 3, 2 * p, ("clamped", "periodic", "periodic"),
                         polar=True, tol=1e-12, maxiter=1000,
                         betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    if geometry == "toroid":
        seq.set_map(toroid_map(epsilon=1 / 3, R0=1.0))
    elif geometry == "cylinder":
        # Periodic cylinder F(r,chi,z) = (a r cos2pi chi, a r sin2pi chi, h z).
        # a = 0.33 keeps the minor radius comparable to the toroid's; h = 1.0.
        # ZERO angular metric variation -- the least-coupled geometry there is,
        # and so the zero-coupling end of the s_opt trend (see handoff 17.6).
        seq.set_map(cylinder_map(a=0.33, h=1.0))
    elif geometry == "rot-ellipse":
        seq.set_map(rotating_ellipse_map(eps=0.33, kappa=1.5, nfp=3))
    else:
        from w7x_geometry import build_w7x_map  # noqa: PLC0415
        map_func, _ = build_w7x_map(map_ns=ns, p=p)
        seq.set_map(map_func)
    ops = op.assemble_incidence_operators(seq)
    ops = op.assemble_mass_jacobi_preconditioner(seq, ops, ks=(0, 1, 2, 3))
    seq.set_operators(ops)
    return seq, ops


def dense_from_apply(apply, n):
    return np.stack([np.asarray(apply(jnp.zeros(n).at[i].set(1.0)))
                     for i in range(n)], axis=1)


def sym(a):
    return 0.5 * (a + a.T)


def pencil_cond(B, A):
    """Generalised spectrum of (B, A): how well A models B, scale-free."""
    import scipy.linalg as sla
    w = sla.eigh(sym(B), sym(A), eigvals_only=True)
    w = w[w > 1e-13 * max(abs(w).max(), 1e-300)]
    if w.size == 0:
        return np.inf, np.nan, np.nan
    return w.max() / w.min(), w.min(), w.max()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="rot-ellipse",
                    choices=("toroid", "rot-ellipse", "w7x", "cylinder"))
    ap.add_argument("--ns", default="6,12,6")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--ks", default="1,3")
    ap.add_argument("--depths", default="1,2,3")
    ap.add_argument("--scales", default="0.03,0.06,0.10,0.15,0.22,0.30,0.55,1.00,2.00")
    cli = ap.parse_args()

    ns = tuple(int(v) for v in cli.ns.split(","))
    scales = [float(v) for v in cli.scales.split(",")]
    depths = [int(v) for v in cli.depths.split(",")]
    seq, ops = build_sequence(cli.geometry, ns, cli.p)
    print(f"geometry={cli.geometry} ns={ns} p={cli.p}  free BC\n", flush=True)

    for k in (int(v) for v in cli.ks.split(",")):
        n = int(getattr(seq, f"n{k}"))
        print(f"===== k={k} free  n={n}", flush=True)

        def L_apply(x, k=k):
            return op.apply_hodge_laplacian_approx(seq, ops, x, k, dirichlet=False)

        Ld = sym(dense_from_apply(L_apply, n))

        # inv(P(s)) for each scale: the operator the preconditioner inverts.
        A_of_s = {}
        for s in scales:
            os.environ["MRX_BJ_BC_SCALE"] = str(s)
            pre = MetricLumpingLaplacian(seq, ops, k, False, ktilde_mode="honest",
                                       lumped="diag", bc_entry="ibpd",
                                       extra_rings=0, outer_rings=0)
            Pd = sym(dense_from_apply(pre.apply, n))
            A_of_s[s] = sym(np.linalg.inv(Pd))
        os.environ["MRX_BJ_BC_SCALE"] = "1.0"

        for d in depths:
            _, _, _, _, _, outer = core_rows(seq, k, False, outer_rings=d)
            R = np.sort(np.asarray(outer))
            if R.size == 0:
                print(f"  depth {d}: no outer rows")
                continue
            interior = np.setdiff1d(np.arange(n), R)
            B_raw = sym(Ld[np.ix_(R, R)])
            # Schur complement onto R: the exact DtN-augmented boundary operator
            B_schur = sym(B_raw - Ld[np.ix_(R, interior)]
                          @ np.linalg.solve(Ld[np.ix_(interior, interior)], Ld[np.ix_(interior, R)]))
            dtn = np.trace(B_raw - B_schur) / np.trace(B_raw)
            print(f"\n  --- depth {d}: {R.size} outer rows, "
                  f"DtN removes {dtn*100:.1f}% of tr(L[R,R])", flush=True)
            print(f"      {'s':>6s}  {'cond(B_raw,A)':>14s}  {'cond(B_schur,A)':>16s}"
                  f"  {'|A-B_raw|/|B_raw|':>18s}  {'|A-B_sch|/|B_sch|':>18s}")
            best = {"raw": (np.inf, None), "schur": (np.inf, None),
                    "nraw": (np.inf, None), "nsch": (np.inf, None)}
            for s in scales:
                A = sym(A_of_s[s][np.ix_(R, R)])
                c_raw, _, _ = pencil_cond(B_raw, A)
                c_sch, _, _ = pencil_cond(B_schur, A)
                e_raw = np.linalg.norm(A - B_raw) / np.linalg.norm(B_raw)
                e_sch = np.linalg.norm(A - B_schur) / np.linalg.norm(B_schur)
                for key, val in (("raw", c_raw), ("schur", c_sch),
                                 ("nraw", e_raw), ("nsch", e_sch)):
                    if val < best[key][0]:
                        best[key] = (val, s)
                print(f"      {s:6.2f}  {c_raw:14.4g}  {c_sch:16.4g}"
                      f"  {e_raw:18.4g}  {e_sch:18.4g}", flush=True)
            print(f"      ARGMIN  cond raw {best['raw'][1]}   cond schur "
                  f"{best['schur'][1]}   ||.|| raw {best['nraw'][1]}   "
                  f"||.|| schur {best['nsch'][1]}", flush=True)


if __name__ == "__main__":
    main()
