"""The magnetic axis of a fixed-boundary equilibrium with nested surfaces, from a VMEC input namelist alone.

A small nested-surface solver whose one job is the finite-beta axis that a relaxation builds its mesh around
(``relax.py --mesh.nested``). It reads the boundary and the target profiles of the namelist
(:mod:`mrx.equilibria.indata`), pressure ``p*(s)`` and either the transform ``iota*(s)`` (``NCURR = 0``) or the
enclosed toroidal current ``I*(s)`` (``NCURR = 1``), and nothing else: no axis, no other code's equilibrium.

The map is an MRX map on a coarse stellarator-symmetric sequence: a :class:`mrx.equilibria.FrameMap` in the
cylindrical frame, ``X1 = R`` on the even and ``X2 = Z`` on the odd polar 0-forms (C1 at the axis), ``X3 = 0``. The
unknowns are their DoFs off the wall. The wall's DoFs hold the boundary, projected onto the 0-forms with the start,
the namelist's fill between :func:`mrx.equilibria.fill.guess_axis` and the boundary (:func:`mrx.equilibria.build_map`).
The radial coordinate is the map's ``r``, read as the label of the nested surfaces, ``s = r^2``. The axis is the
image of ``r = 0``.

The field is the nested field of the toroidal flux ``Phi_t = PHIEDGE r^2`` in the logical angles (turns, ``zeta``
one field period),

    J B^theta = chi' / nfp - Phi_t' lambda_zeta,   J B^zeta = Phi_t' (1 + lambda_theta),

with ``J = det DPhi``, ``lambda`` an odd 0-form (free on the wall) and ``chi`` the poloidal flux, ``chi' = iota*
Phi_t'`` given, or ``chi'`` a radial derivative spline that vanishes at the axis. The energy is

    W = int B^2 / 2 - mu0 p*(s) dV - int mu0 I*(s) chi'(r) dr        (in units of mu0, the last term only with I*)

whose minimum over ``chi'`` puts the current ``I*`` inside every surface in the weak sense of ``chi'``'s splines.
For a fixed map ``W`` is quadratic in ``lambda`` and ``chi'``: one linear system gives them exactly, and the reduced
energy of the map is minimised by BFGS. Its gradient is the partial gradient at the solved ``lambda`` (the solve is a
minimum). The metric and ``det DPhi`` are the sequence's (:class:`mrx.geometry.SequenceGeometry`) at its quadrature
points, ``q_r`` Gauss points per radial span: with too few the rule is blind between its points, and the minimiser
lowers the energy by folding the map there (2026-11-21). The line search keeps ``det DPhi > 0`` on a finer check grid,
which ``1 / J`` in the energy turns into a barrier. The first inverse Hessian is that of the exact Hessian at the
start with its eigenvalues replaced by their moduli held above ``1e-6`` of the largest (the start need not be
convex, and the poloidal reparametrisation is a near null space), made again whenever a step needs cutting below
1/64. BFGS stops once the axis has moved less than ``axis_tol`` over ``window`` iterations, or the energy stops
decreasing.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.equilibria import _left_handed, _reverse_theta, build_map
from mrx.equilibria.fill import fill_state, guess_axis
from mrx.equilibria.indata import read_indata
from mrx.geometry import SequenceGeometry, map_jacobian_at


def setup(path, nr=6, nt=12, nz=10, p=3, q_r=10):
    """The coarse sequence of the namelist ``path``, its start map and the constant arrays of the energy."""
    nl = read_indata(path)
    nfp = nl.nfp
    rows = np.asarray(nl.boundary_rows()[0], dtype=np.float64)
    seq = DeRhamSequence((nr + p, nt, nz), p, nfp=nfp, symmetry="stellarator", quad_points=(q_r, p, p))
    # the start: the fill between the guessed axis (from the boundary's modes the mesh resolves) and the boundary
    st = dict(fill_state(rows, guess_axis(rows[np.abs(rows[:, 1]) <= nfp * nz // 2], nfp), nfp), path=str(path))
    st = _reverse_theta(st) if _left_handed(st) else st     # MRX's logical coordinates are right-handed
    Phi, _ = build_map(st, seq, stellarator_symmetric=True)
    ev, od = seq.even.free, seq.odd.free
    v0 = (ev.conforming(Phi.raw_X1.reshape(-1), 0), od.conforming(Phi.raw_X2.reshape(-1), 0))
    inner = (seq.even._dirichlet_dofs[0], seq.odd._dirichlet_dofs[0])

    x, w = seq.quad.x, seq.quad.w
    r = x[:, 0]
    dPhi_t = 2.0 * nl.phiedge * r                               # d Phi_t / dr
    # lambda: the theta and zeta derivatives of every odd free 0-form at the quadrature points
    dlam = jax.vmap(lambda e: od.evaluate_at_quadrature(od.G[0] @ e, 1))(jnp.eye(od.n(0)))
    Ut, Uz = -dPhi_t * dlam[:, :, 2], dPhi_t * dlam[:, :, 1]
    u0z = dPhi_t
    ir = jnp.arange(x.shape[0]) // (seq.quad.ny * seq.quad.nz)
    if nl.ncurr == 1:
        # chi' on the radial derivative splines but the first, the one nonzero at the axis
        dchi = seq.d_basis_jk[0][1:]                            # (n, n_q_r)
        Ut = jnp.concatenate([Ut, dchi[:, ir] / nfp])
        Uz = jnp.concatenate([Uz, jnp.zeros((dchi.shape[0], x.shape[0]))])
        ell = jnp.concatenate([jnp.zeros(od.n(0)), dchi @ (seq.quad.w_x * nl.current(np.asarray(seq.quad.x_x) ** 2))])
        u0t = jnp.zeros_like(r)
    else:
        ell = jnp.zeros(od.n(0))
        u0t = dPhi_t * jnp.asarray(nl.iota(np.asarray(r) ** 2)) / nfp
    r_check = jnp.linspace(0.0, 1.0, 4 * nr * q_r + 1)[1:]
    check = jnp.stack(jnp.meshgrid(r_check, seq.quad.x_y, seq.quad.x_z, indexing="ij"), -1).reshape(-1, 3)
    return dict(nl=nl, seq=seq, Phi=Phi, v0=v0, inner=inner, nfp=nfp, check=check, n_lambda=od.n(0),
                arrays=dict(Ut=Ut.T, Uz=Uz.T, u0t=u0t, u0z=u0z, ell=ell, w=w,
                            mu0p=jnp.asarray(nl.pressure(np.asarray(r) ** 2))))


def map_of(prob, x):
    """The map of the free unknowns ``x``: the DoFs of ``R`` and ``Z`` off the wall."""
    seq, (vR, vZ), (iR, iZ) = prob["seq"], prob["v0"], prob["inner"]
    vR = vR.at[iR].set(x[:iR.size])
    vZ = vZ.at[iZ].set(x[iR.size:])
    shape = seq.basis_0.shape[0]
    raw_R, raw_Z = (v.E(0).T @ c for v, c in ((seq.even.free, vR), (seq.odd.free, vZ)))
    return prob["Phi"].with_coordinates(raw_R.reshape(shape), raw_Z.reshape(shape), jnp.zeros(shape))


def make_functions(prob):
    """``(F, ok, lam, hessian)``: the reduced energy of the free unknowns (partial gradient at the solved lambda),
    the smallest ``det DPhi`` on the check grid, the solved lambda (and chi'), and the exact Hessian of the reduced
    energy (Hessian-vector products in batches of 8, through the lambda solve)."""
    a, nfp, xq = prob["arrays"], prob["nfp"], prob["seq"].quad.x

    def quadratic(x):
        geo = SequenceGeometry.from_map(map_of(prob, x), xq)
        g, c = geo.metric_jkl, nfp * a["w"] / geo.jacobian_j
        gtt, gtz, gzz = c * g[:, 1, 1], c * g[:, 1, 2], c * g[:, 2, 2]
        Ut, Uz, u0t, u0z = a["Ut"], a["Uz"], a["u0t"], a["u0z"]
        A = (Ut.T @ (gtt[:, None] * Ut) + Ut.T @ (gtz[:, None] * Uz) + Uz.T @ (gtz[:, None] * Ut)
             + Uz.T @ (gzz[:, None] * Uz))
        g0 = Ut.T @ (gtt * u0t + gtz * u0z) + Uz.T @ (gtz * u0t + gzz * u0z)
        c0 = 0.5 * jnp.sum(gtt * u0t ** 2 + 2.0 * gtz * u0t * u0z + gzz * u0z ** 2)
        return A, a["ell"] - g0, c0 - nfp * jnp.sum(a["w"] * a["mu0p"] * geo.jacobian_j)

    def solve_y(A, rhs):
        d = 1.0 / jnp.sqrt(jnp.diag(A))
        return d * jnp.linalg.solve(d[:, None] * A * d[None], d * rhs)

    def F(x):
        A, rhs, c = quadratic(x)
        y = jax.lax.stop_gradient(solve_y(A, rhs))
        return 0.5 * y @ A @ y - rhs @ y + c

    def F_full(x):
        A, rhs, c = quadratic(x)
        return -0.5 * rhs @ solve_y(A, rhs) + c

    def hessian(x):
        grad = jax.grad(F_full)
        return jax.lax.map(lambda v: jax.jvp(grad, (x,), (v,))[1], jnp.eye(x.size), batch_size=8)

    def jac_min(x):
        return jnp.min(jnp.linalg.det(map_jacobian_at(map_of(prob, x), prob["check"])))

    return (jax.jit(jax.value_and_grad(F)), jax.jit(jac_min), jax.jit(lambda x: solve_y(*quadratic(x)[:2])),
            jax.jit(hessian))


def axis_points(Phi, samples=144):
    """``(zetas, points)``: the axis ``Phi(0, 0, zeta)`` at ``samples`` logical ``zetas`` of one field period as
    Cartesian points (the format of :func:`mrx.relaxation.mesh_motion.axis_curve`)."""
    zetas = np.arange(samples) / samples
    pts = jnp.stack([jnp.zeros(samples), jnp.zeros(samples), jnp.asarray(zetas)], 1)
    return zetas, np.asarray(jax.vmap(Phi)(pts))


def solve(path, max_iter=3000, ftol=1e-14, axis_tol=1e-6, window=25, verbose=True, **kw):
    """Minimise the reduced energy of the namelist ``path`` (module docstring) by BFGS. Returns ``(Phi, record)``:
    the converged map and a record with the energy, the gradient, the iterations, ``iota`` of the solution and the
    wall times."""
    t0 = time.perf_counter()
    prob = setup(path, **kw)
    vg, jac_min, lam, hess = make_functions(prob)
    iR, iZ = prob["inner"]
    x = jnp.concatenate([prob["v0"][0][iR], prob["v0"][1][iZ]])
    if not float(jac_min(x)) > 0.0:
        raise ValueError(f"{path}: the start map folds")
    f, g = vg(x)
    t_setup = time.perf_counter() - t0
    th = [0.0, 0]

    def metric(x):
        t1 = time.perf_counter()
        ev, Q = np.linalg.eigh(np.asarray(hess(x)))
        ev = np.maximum(np.abs(ev), 1e-6 * np.abs(ev).max())
        th[0] += time.perf_counter() - t1
        th[1] += 1
        return (Q / ev) @ Q.T

    zeta16 = jnp.stack([jnp.zeros(16), jnp.zeros(16), jnp.arange(16) / 16.0], 1)
    axis = jax.jit(lambda x: jax.vmap(map_of(prob, x))(zeta16))
    Hinv = metric(x)
    t_compile = time.perf_counter() - t0
    f, g = float(f), np.asarray(g)
    g0 = np.linalg.norm(g)
    small, it, fresh, axes = 0, 0, True, []
    while it < max_iter:
        it += 1
        d = -Hinv @ g
        t, gd = 1.0, float(g @ d)
        xn = None
        if gd < 0.0:
            for _ in range(30):
                xn = x + t * d
                if float(jac_min(xn)) > 0.0:
                    fn, gn = vg(xn)
                    fn = float(fn)
                    if fn <= f + 1e-4 * t * gd:
                        break
                t *= 0.5
            else:
                xn = None
        if xn is None or t < 1.0 / 64:
            # BFGS lost its metric: the exact one again, at the last point, unless it was just made
            if fresh and xn is None:
                if verbose:
                    print(f"[nested] iteration {it}: no descent with a fresh Hessian, stop", flush=True)
                break
            if not fresh:
                Hinv, fresh = metric(x), True
                if xn is None:
                    continue
        else:
            fresh = False
        gn = np.asarray(gn)
        s_, y_ = np.asarray(xn - x), gn - g
        sy = float(s_ @ y_)
        if sy > 1e-12 * np.linalg.norm(s_) * np.linalg.norm(y_):
            Hy = Hinv @ y_
            Hinv = Hinv + ((sy + y_ @ Hy) / sy ** 2) * np.outer(s_, s_) - (np.outer(Hy, s_) + np.outer(s_, Hy)) / sy
        df = f - fn
        x, f, g = xn, fn, gn
        if verbose and it % 50 == 0:
            print(f"[nested] {it}: W {f:.12e}, |grad| {np.linalg.norm(g) / g0:.3e} of the start's, step {t:g}",
                  flush=True)
        small = small + 1 if df <= ftol * abs(f) else 0
        axes.append(np.asarray(axis(x)))
        if small >= 5 or (len(axes) > window and np.abs(axes[-1] - axes[-1 - window]).max() < axis_tol):
            break
    t_total = time.perf_counter() - t0
    y = np.asarray(lam(x))
    seq = prob["seq"]
    rec = dict(path=str(path), nfp=prob["nfp"], ns=seq.ns, p=seq.p, quad_points=seq.quad_points,
               unknowns=int(x.size), lambda_unknowns=int(y.size), iterations=it, energy=f,
               grad_rel=float(np.linalg.norm(g) / g0), min_jacobian=float(jac_min(x)), t_setup=t_setup,
               t_compile_hessian=t_compile, hessians=th[1], t_hessians=th[0], t_total=t_total)
    if prob["nl"].ncurr == 1:
        r = np.asarray(seq.quad.x_x)
        rec["iota_r"] = r.tolist()
        rec["iota"] = ((np.asarray(seq.d_basis_jk[0][1:]).T @ y[prob["n_lambda"]:])
                       / (2.0 * prob["nl"].phiedge * r)).tolist()
    if verbose:
        print(f"[nested] {it} iterations, W {f:.12e}, |grad| {rec['grad_rel']:.2e} of the start's, "
              f"{t_total:.1f} s ({t_compile:.1f} s setup, compile and the first Hessian)", flush=True)
    return map_of(prob, x), rec

