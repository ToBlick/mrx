"""Tutorial 2 (convergence): convergence to an analytic vacuum (coil) field on the QA domain.

The magnetic field of external coils, in the plasma region, is a *vacuum*
field: curl-free (no current) and div-free (no monopoles). Those two facts pick
out the two ends of the de Rham complex:

  * curl-free  =>  a gradient  =>  the H-field (a **1-form**) has a SCALAR
    potential,  ``H = grad f``  (+ a harmonic form for the net linking flux);
  * div-free   =>  a curl      =>  the B-field (a **2-form**, a flux) has a
    VECTOR potential,  ``B = curl A``.

This tutorial solves for both and watches each converge to a field known in
closed form, so there is no discretisation floor -- the error falls at O(h^p).

The analytic field is a model coil field on the QA stellarator,

    B* = e_phi / R  +  lambda * grad(R^2 cos 2phi),

the axisymmetric toroidal-field part (``1/R``, the net coil current linking the
torus) plus an ``n = 2`` shaping ripple (QA has two field periods). Its toroidal
modes are multiples of ``nfp``, so it is single-valued on the modelled field
period.

  Route A (scalar potential):  G0^T M1 G0 f = G0^T (load of B*),  H = grad f +
      alpha h1                                              -- a k=0 solve.
  Route C (vector potential):  <curl A, curl W> = <B*, curl W>, B = curl A
      -- a k=1 curl-curl solve. On a solid torus b2 = 0, so B* = curl A fully
      (the flux is carried by A's boundary circulation); no harmonic term.

Both reuse the preconditioned Hodge-Laplacian solvers -- nothing new to build.

    python -u scripts/tutorials/qa_vacuum_convergence.py
"""
from __future__ import annotations

import argparse
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--geometry", default="data/wout_LandremanPaul2021_QA_lowres.nc",
                    help="a VMEC wout (.nc) or a GVEC state file (.dat)")
    ap.add_argument("--ns", default="6,12,6:9,18,9:12,24,12:15,30,15",
                    help="colon-separated n_r,n_theta,n_zeta rungs")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--lam", type=float, default=1.0, help="ripple amplitude")
    ap.add_argument("--out", default="outputs/tutorials/qa_vacuum_convergence")
    cli = ap.parse_args()
    os.makedirs(cli.out, exist_ok=True)

    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mrx.geometry import build_sequence
    from mrx.nullspace import compute_nullspaces

    # The analytic vacuum field, a lab-frame (3,) vector at the logical point.
    # B* = e_phi/R + lam grad(R^2 cos 2phi); both pieces are curl- and div-free.
    def b_star(seq):
        grad_ripple = jax.grad(lambda X: X[0] ** 2 - X[1] ** 2)   # grad(R^2 cos 2phi)

        def f(xi):
            X = seq.map(xi)
            tf = jnp.array([-X[1], X[0], 0.0]) / (X[0] ** 2 + X[1] ** 2)   # e_phi / R
            return tf + cli.lam * grad_ripple(X)
        return f

    # ||B*||_M^2 = int |B*_phys|^2 dV, by quadrature -- the exact-field norm.
    def bstar_norm_sq(seq, Bphys):
        Bq = np.asarray(jax.vmap(Bphys)(seq.quad.x))
        return float(np.sum(np.asarray(seq.quad.w) * np.asarray(seq.jacobian_j)
                            * np.sum(Bq ** 2, axis=1)))

    rungs = [tuple(int(v) for v in c.split(",")) for c in cli.ns.split(":")]
    rows = []   # (h, relerr_A, relerr_C)
    for ns in rungs:
        seq, ops = build_sequence(cli.geometry, ns, cli.p)
        ops = seq.set_operators(compute_nullspaces(seq, ops))
        Bphys = b_star(seq)
        bsq = bstar_norm_sq(seq, Bphys)

        # -- Route A: scalar potential, H = grad f + alpha h1 (all free / no-BC).
        #    The Neumann load of a div-free field is a volume integral:
        #    int_wall (B*.n) v  =  int grad v . B*  =  G0^T (load1 of B*).
        load1 = seq.load(Bphys, 1, dirichlet=False)
        f = seq.apply_inverse_laplacian(
            seq.apply_incidence_matrix(load1, 0, dirichlet_in=False,
                                       dirichlet_out=False, transpose=True),
            0, dirichlet=False, operators=ops)
        h1 = seq.nullspace(1, False)[0]                        # the flux generator
        alpha = float(load1 @ h1) / float(h1 @ seq.apply_mass_matrix(h1, 1, False))
        H = seq.apply_strong_grad(f, dirichlet_in=False, dirichlet_out=False) + alpha * h1
        err_A = np.sqrt(max(float(H @ seq.apply_mass_matrix(H, 1, False))
                            - 2.0 * float(H @ load1) + bsq, 0.0)) / np.sqrt(bsq)

        # -- Route C: vector potential, B = curl A (all free / no-BC).
        #    <curl A, curl W> = <B*, curl W>  =>  L1 A = C^T (load2 of B*).
        load2 = seq.load(Bphys, 2, dirichlet=False)
        A = seq.apply_inverse_laplacian(
            seq.apply_incidence_matrix(load2, 1, dirichlet_in=False,
                                       dirichlet_out=False, transpose=True),
            1, dirichlet=False, operators=ops)
        B = seq.apply_strong_curl(A, dirichlet_in=False, dirichlet_out=False)
        err_C = np.sqrt(max(float(B @ seq.apply_mass_matrix(B, 2, False))
                            - 2.0 * float(B @ load2) + bsq, 0.0)) / np.sqrt(bsq)

        h = 1.0 / (ns[0] - cli.p)
        rows.append((h, err_A, err_C))
        print(f"[conv] n_el {ns[0] - cli.p:2d}  h {h:.4f}  "
              f"relerr  A {err_A:.3e}  C {err_C:.3e}", flush=True)

    hs = np.array([r[0] for r in rows])
    for j, name in ((1, "A (scalar pot., H)"), (2, "C (vector pot., B)")):
        es = np.array([r[j] for r in rows])
        order = np.polyfit(np.log(hs), np.log(es), 1)[0]
        print(f"[order] Route {name}: fitted p_eff = {order:.2f}  (expected {cli.p})")

    # log-log convergence plot with an O(h^p) guide.
    fig, ax = plt.subplots(figsize=(5, 4))
    for j, mk, name in ((1, "o-", "Route A: $H=\\nabla f+\\alpha h_1$"),
                        (2, "s-", "Route C: $B=\\operatorname{curl}A$")):
        ax.loglog(hs, [r[j] for r in rows], mk, label=name)
    guide = rows[0][1] * (hs / hs[0]) ** cli.p
    ax.loglog(hs, guide, "k--", lw=0.8, label=fr"$O(h^{cli.p})$")
    ax.set_xlabel("$h = 1/n_{\\mathrm{el}}$")
    ax.set_ylabel(r"$\|B_h - B^\ast\|_M / \|B^\ast\|_M$")
    ax.set_title(f"Vacuum-field convergence on QA, $p={cli.p}$")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(cli.out, "convergence.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
