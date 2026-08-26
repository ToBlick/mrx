"""k=0 Poisson convergence A/B: C^1 (polar_order=1) vs C^2 (polar_order=2).

Toroid (epsilon=1/3), dbc, p=3. Three manufactured solutions with poloidal
mode content m = 0, 1, 2 (all smooth in PHYSICAL coordinates and vanishing
at r=1); the m=2 case is the discriminating test for the C^2 pole
conditions (its representation near the axis needs the ring-2 quadratic
DOFs). The source f = -div(grad u) is computed by AD through the metric,
so the same machinery serves all cases (and cross-checks the repo's
analytic m=0 source).

Solver: Jacobi-preconditioned CG on K0 (the production FD preconditioner
hardcodes the C^1 core layout). Error metric: relative L2 at quadrature
points. Expected: both orders converge at O(h^{p+1}); a wrong C^2
construction would collapse the m=2 (and generic) order at the axis.

Run: python scripts/debug/poisson_k0_c2_convergence.py [--ns 4 6 8 12]
"""
import argparse
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp


from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.mappings import toroid_map  # noqa: E402
from mrx.geometry import compute_geometry_terms  # noqa: E402
from mrx.operators import (  # noqa: E402
    assemble_incidence_operators,
    apply_stiffness,
    apply_mass_matrix,
    _diagonal_from_matvec,
)
from mrx.quadrature import evaluate_at_xq  # noqa: E402
from mrx.solvers import solve_singular_cg  # noqa: E402

TYPES = ("clamped", "periodic", "periodic")
PI = jnp.pi
EPS = 1.0 / 3.0


def u_m0(x):
    r, c, z = x
    return 0.25 * (r ** 2 - r ** 4) * jnp.cos(2 * PI * z)


def u_m1(x):
    r, c, z = x
    return r * jnp.sin(2 * PI * c) * (1 - r ** 2) * jnp.cos(2 * PI * z)


def u_m2(x):
    r, c, z = x
    return r ** 2 * jnp.sin(4 * PI * c) * (1 - r ** 2) * jnp.cos(2 * PI * z)


CASES = {"m0": u_m0, "m1": u_m1, "m2": u_m2}


def make_source(u_fn, F):
    """f = -(1/J) d_i (J g^{ij} d_j u) via AD through the metric."""
    du = jax.grad(u_fn)

    def flux(xp):
        _, minv, jac = compute_geometry_terms(F, xp.reshape(1, 3))
        return jac[0] * (minv[0] @ du(xp))

    def f(xp):
        Jf = jax.jacfwd(flux)(xp)
        _, _, jac = compute_geometry_terms(F, xp.reshape(1, 3))
        return (-(Jf[0, 0] + Jf[1, 1] + Jf[2, 2]) / jac[0]) * jnp.ones(1)

    return f


def exact_at_quad(seq, u_fn):
    """Exact u at the quadrature grid, layout (ny, nx, nz) -> (theta, r, zeta)."""
    xy, xx, xz = jnp.meshgrid(seq.quad.x_y, seq.quad.x_x, seq.quad.x_z,
                              indexing="ij")
    pts = jnp.stack([xx.ravel(), xy.ravel(), xz.ravel()], axis=-1)
    return jax.vmap(u_fn)(pts).reshape(-1, 1)


def run_case(n, p, polar_order, u_fn, tol, maxiter):
    ns = (n, 2 * n, n)
    F = toroid_map(epsilon=EPS)
    seq = DeRhamSequence(ns, (p, p, p), 2 * p, TYPES, polar=True,
                         tol=tol, maxiter=maxiter, polar_order=polar_order)
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()
    seq.set_map(F)
    # NOTE: assemble_laplacian_operators is NOT called -- it eagerly warms
    # the production k=0 FD preconditioner, which hardcodes the C^1 polar
    # layout. apply_stiffness/apply_mass_matrix need only the tensor
    # incidence + the matrix-free mass + the extraction operators.
    ops = seq.get_operators()
    ops = assemble_incidence_operators(seq, operators=ops, ks=(0,))

    rhs = seq.load(make_source(u_fn, F), 0, dirichlet=True)
    n_dbc = int(seq.n0_dbc)

    def A(v):
        return apply_stiffness(seq, ops, v, 0, dirichlet=True)

    def M(v):
        return apply_mass_matrix(seq, ops, v, 0, dirichlet=True)

    diag = _diagonal_from_matvec(A, n_dbc)
    dinv = jnp.where(diag > 0, 1.0 / diag, 0.0)
    vs = []  # dbc k=0: no nullspace
    u_hat, info = solve_singular_cg(A, rhs, mass_matvec=M,
                                    precond_matvec=lambda v: dinv * v,
                                    vs=vs, tol=tol, maxiter=maxiter)
    it = abs(int(info))
    res = float(jnp.linalg.norm(A(u_hat) - rhs) / jnp.linalg.norm(rhs))

    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    ci, cs = seq._form_comp_info(0)
    u_h = evaluate_at_xq(seq.e0_dbc_T @ u_hat, ci, cs, quad_shape, 1)
    u_i = exact_at_quad(seq, u_fn)
    df = u_i - u_h
    L2d = jnp.einsum("ik,ik,i,i->", df, df, seq.jacobian_j, seq.quad.w)
    L2f = jnp.einsum("ik,ik,i,i->", u_i, u_i, seq.jacobian_j, seq.quad.w)
    return float((L2d / L2f) ** 0.5), it, res, n_dbc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[4, 6, 8, 12])
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--cases", default="m0,m1,m2")
    ap.add_argument("--orders", default="1,2",
                    help="comma-separated polar orders to compare (0, 1, 2)")
    ap.add_argument("--tol", type=float, default=1e-11)
    ap.add_argument("--maxiter", type=int, default=50000)
    args = ap.parse_args()

    bad = False
    for case in [c.strip() for c in args.cases.split(",")]:
        u_fn = CASES[case]
        print(f"=== case {case}  p={args.p}  toroid eps=1/3 dbc ===", flush=True)
        for order in [int(o) for o in args.orders.split(",")]:
            errs = []
            for n in args.ns:
                t0 = time.perf_counter()
                err, it, res, ndof = run_case(n, args.p, order, u_fn,
                                              args.tol, args.maxiter)
                dt = time.perf_counter() - t0
                rate = (np.log(errs[-1] / err) / np.log(n / args.ns[len(errs) - 1])
                        if errs else float("nan"))
                errs.append(err)
                print(f"  C{order} n={n:<3d} ndof={ndof:<6d} L2={err:.3e} "
                      f"rate={rate:5.2f}  cg_it={it:<6d} rel_res={res:.1e} "
                      f"({dt:.0f}s)", flush=True)
            if len(errs) >= 2:
                final_rate = np.log(errs[-2] / errs[-1]) / np.log(args.ns[-1] / args.ns[-2])
                ok = final_rate > args.p + 0.5
                print(f"  C{order} {case}: final rate {final_rate:.2f} "
                      f"({'OK' if ok else 'LOW'}; want ~{args.p + 1})", flush=True)
                bad |= not ok
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
