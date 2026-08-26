"""Modal-radial k=0 Laplacian atom vs the production fd atom, on stellarators.

Closes the handoff item in docs/research/mass_preconditioner_pivot.md section 7.5:
the one approximation modal-radial still makes is averaging the metric over the
angular directions, and both geometries measured so far are axisymmetric, where
that averaging is nearly free. A stellarator metric varies strongly in zeta.

**Atom variants**

``fd`` (production, ``_assemble_k0_greville_bulk_factors``): each weight
``g^aa J`` is averaged over the *other two* axes to give a profile along its own
axis, then all three directions are FD-diagonalized and combined with the
additive denominator ``lam_r + lam_t + lam_z``.

``modal`` (this script): diagonalize only the theta and zeta pencils and keep the
radial direction exact::

    K ~ K_r[a] (x) M_t (x) M_z + M_r[b] (x) K_t[beta] (x) M_z
                               + M_r[c] (x) M_t (x) K_z[gamma]

    (M_t, K_t[beta]) -> mu_j ,   (M_z, K_z[gamma]) -> nu_k
    =>  block diagonal over (j,k) with A_jk = K_r[a] + mu_j M_r[b] + nu_k M_r[c]
        (n_r x n_r, solved exactly -- no radial averaging anywhere)

Only the *shared* mass factors must stay unweighted for the two pencils to
diagonalize simultaneously across the three terms. That constrains each term
differently, and only ``g^rr J`` loses both angular directions:

    term 1: theta and zeta averaged (M_t, M_z both shared)
    term 2: zeta averaged only -- the theta dependence goes into K_t[beta]
    term 3: theta averaged only -- the zeta dependence goes into K_z[gamma]

Profiles use the harmonic mean, which section 7.2 measured as never worse than
arithmetic and which removes the need for the ``wx_cut`` polar-element skip.

**Bulk-block only.** Section 7.1 established that the bulk numbers track the
full-grid ones (production 25 free / 22 dbc -> modal 13 / 13), so the atom is
isolated without rebuilding the core Schur -- and the stored ``schur_inv`` is
built from exact bulk solves, hence atom-independent anyway.

Usage:
    python -u scripts/debug/modal_radial_gate.py --ns 12 24 24
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map, toroid_map
from mrx.operators import (_assemble_1d_fd_eigendecomp,
                           _assemble_k0_greville_bulk_factors,
                           _assemble_unweighted_1d_mass,
                           _assemble_weighted_1d_stiffness,
                           _apply_k0_tensor_hodge_bulk_inverse,
                           _build_k0_tensor_hodge_preconditioner_factors,
                           _dense_incidence_1d, _core_size,
                           _reshape_quadrature_matrix_field,
                           _reshape_quadrature_scalar_field,
                           _restrict_radial_window, apply_stiffness,
                           assemble_incidence_operators)
from mrx.preconditioners import _bulk_tensor_shape

p = argparse.ArgumentParser()
p.add_argument("--ns", type=int, nargs=3, default=(12, 24, 24))
p.add_argument("--p", type=int, default=3)
p.add_argument("--h5", default="data/W7X-vacuum.h5")
p.add_argument("--stride", type=int, default=2)
p.add_argument("--tol", type=float, default=1e-10)
p.add_argument("--maxit", type=int, default=4000)
args = p.parse_args()
NS, P = tuple(args.ns), args.p
TYPES = ("clamped", "periodic", "periodic")

print(f"modal-radial gate: ns={NS} p={P}", flush=True)
print(f"backend={jax.default_backend()}\n", flush=True)


# --------------------------------------------------------------------------- #
# profiles
# --------------------------------------------------------------------------- #
def _harmonic(vals, w, axes):
    """Quadrature-weighted harmonic mean of ``vals`` over ``axes``.

    Harmonic rather than arithmetic: section 7.2 measured it as never worse and
    finite for the ``1/r``-type ``g^tt J`` weight whose arithmetic mean diverges
    at the axis -- which is what forced the ``wx_cut`` polar-element skip.
    """
    x = 1.0 / jnp.clip(vals, 1e-30)
    denom = 1.0
    for ax in sorted(axes, reverse=True):
        wa = w[ax]
        x = jnp.tensordot(x, wa, axes=([ax], [0]))
        denom = denom * jnp.sum(wa)
    return denom / jnp.clip(x, 1e-30)


def modal_radial_bulk_data(seq, *, dirichlet, angular_profiles=True):
    """Bulk factors for the modal-radial atom (see module docstring)."""
    bulk_shape = _bulk_tensor_shape(seq, dirichlet)
    nr_bulk = int(bulk_shape[0])
    types = seq.basis_0.types

    minv = jnp.transpose(
        _reshape_quadrature_matrix_field(seq, seq.geometry.metric_inv_jkl),
        (1, 0, 2, 3, 4))
    jacq = jnp.transpose(
        _reshape_quadrature_scalar_field(seq, seq.geometry.jacobian_j), (1, 0, 2))
    w00, w11, w22 = (minv[..., a, a] * jacq for a in range(3))
    W = (seq.quad.w_x, seq.quad.w_y, seq.quad.w_z)

    a_r = _harmonic(w00, W, (1, 2))               # g^rr J : both angles averaged
    b_r = _harmonic(w11, W, (1, 2))               # g^tt J : radial profile
    beta_t = _harmonic(w11, W, (0, 2))            #          theta profile (kept)
    c_r = _harmonic(w22, W, (1, 2))               # g^zz J : radial profile
    gamma_z = _harmonic(w22, W, (0, 1))           #          zeta profile (kept)
    # rank-1 split: b(r)*beta(t) must reproduce the mean, so divide out one copy
    mean_11 = _harmonic(b_r, W, (0,))
    mean_22 = _harmonic(c_r, W, (0,))
    beta_t = beta_t / jnp.clip(mean_11, 1e-30)
    gamma_z = gamma_z / jnp.clip(mean_22, 1e-30)
    if not angular_profiles:
        # 'flat' arm: average g^tt J over theta and g^zz J over zeta too, i.e.
        # the original section 7.1 form. Isolates what keeping the angular
        # dependence inside K_t / K_z is actually worth.
        beta_t = jnp.ones_like(beta_t)
        gamma_z = jnp.ones_like(gamma_z)

    g_r = _dense_incidence_1d(seq.basis_0.nr, types[0])
    g_t = _dense_incidence_1d(seq.basis_0.nt, types[1])
    g_z = _dense_incidence_1d(seq.basis_0.nz, types[2])

    K_r = _restrict_radial_window(_assemble_weighted_1d_stiffness(
        seq.basis_r_jk, seq.d_basis_r_jk, seq.quad.w_x * a_r, g_r), 2, nr_bulk)
    M_r_b = _restrict_radial_window(_assemble_unweighted_1d_mass(
        seq.basis_r_jk, seq.quad.w_x * b_r), 2, nr_bulk)
    M_r_c = _restrict_radial_window(_assemble_unweighted_1d_mass(
        seq.basis_r_jk, seq.quad.w_x * c_r), 2, nr_bulk)
    M_t = _assemble_unweighted_1d_mass(seq.basis_t_jk, seq.quad.w_y)
    M_z = _assemble_unweighted_1d_mass(seq.basis_z_jk, seq.quad.w_z)
    K_t = _assemble_weighted_1d_stiffness(
        seq.basis_t_jk, seq.d_basis_t_jk, seq.quad.w_y * beta_t, g_t)
    K_z = _assemble_weighted_1d_stiffness(
        seq.basis_z_jk, seq.d_basis_z_jk, seq.quad.w_z * gamma_z, g_z)

    V_t, mu = _assemble_1d_fd_eigendecomp(M_t, K_t)
    V_z, nu = _assemble_1d_fd_eigendecomp(M_z, K_z)

    # A_jk = K_r + mu_j M_r_b + nu_k M_r_c, pseudo-inverted per mode.
    A = (K_r[None, None] + mu[:, None, None, None] * M_r_b[None, None]
         + nu[None, :, None, None] * M_r_c[None, None])
    A_pinv = jnp.linalg.pinv(A, rcond=1e-12, hermitian=True)
    return {"bulk_shape": bulk_shape, "V_t": V_t, "V_z": V_z, "A_pinv": A_pinv}


def modal_perk_bulk_data(seq, *, dirichlet):
    """Modal-radial with the per-k pencil reduction.

    Measured (2026-08-17, toroid / rot-ellipse / W7-X): the radial profiles of
    ``g^rr J`` and ``g^zz J`` are proportional -- log-log slopes both +1, ratio
    spread <= 1.07 -- and at the ASSEMBLED MATRIX level
    ``||M_r[c] - kappa M_r[a]|| / ||M_r[c]|| = 0.014``. So there are really only
    two distinct radial operators, and the mode dependence separates::

        A_jk = (K_r[a] + kappa nu_k M_r[a])  +  mu_j M_r[b]  =  P_k + mu_j Q

    ``P_k`` depends only on k and ``Q`` is fixed, so ONE pencil (Q, P_k) per k
    serves every j::

        W_k^T Q W_k = I,  W_k^T P_k W_k = diag(d_k)
        =>  A_jk^-1 = W_k diag( 1 / (d_k + mu_j) ) W_k^T

    Storage falls from ``n_t n_z n_r^2`` to ``n_z n_r^2`` -- independent of
    ``n_t``, 252 MB -> 2.0 MB at 64x128x64 -- and the per-mode radial solves
    become a diagonal scale.
    """
    bulk_shape = _bulk_tensor_shape(seq, dirichlet)
    nr_bulk = int(bulk_shape[0])
    types = seq.basis_0.types

    minv = jnp.transpose(_reshape_quadrature_matrix_field(
        seq, seq.geometry.metric_inv_jkl), (1, 0, 2, 3, 4))
    jacq = jnp.transpose(_reshape_quadrature_scalar_field(
        seq, seq.geometry.jacobian_j), (1, 0, 2))
    w00, w11, w22 = (minv[..., a, a] * jacq for a in range(3))
    W = (seq.quad.w_x, seq.quad.w_y, seq.quad.w_z)

    a_r = _harmonic(w00, W, (1, 2))
    b_r = _harmonic(w11, W, (1, 2))
    c_r = _harmonic(w22, W, (1, 2))
    beta_t = _harmonic(w11, W, (0, 2)) / jnp.clip(_harmonic(b_r, W, (0,)), 1e-30)
    gamma_z = _harmonic(w22, W, (0, 1)) / jnp.clip(_harmonic(c_r, W, (0,)), 1e-30)

    g_r = _dense_incidence_1d(seq.basis_0.nr, types[0])
    g_t = _dense_incidence_1d(seq.basis_0.nt, types[1])
    g_z = _dense_incidence_1d(seq.basis_0.nz, types[2])

    K_r = _restrict_radial_window(_assemble_weighted_1d_stiffness(
        seq.basis_r_jk, seq.d_basis_r_jk, seq.quad.w_x * a_r, g_r), 2, nr_bulk)
    M_a = _restrict_radial_window(_assemble_unweighted_1d_mass(
        seq.basis_r_jk, seq.quad.w_x * a_r), 2, nr_bulk)
    M_b = _restrict_radial_window(_assemble_unweighted_1d_mass(
        seq.basis_r_jk, seq.quad.w_x * b_r), 2, nr_bulk)
    M_c = _restrict_radial_window(_assemble_unweighted_1d_mass(
        seq.basis_r_jk, seq.quad.w_x * c_r), 2, nr_bulk)
    M_t = _assemble_unweighted_1d_mass(seq.basis_t_jk, seq.quad.w_y)
    M_z = _assemble_unweighted_1d_mass(seq.basis_z_jk, seq.quad.w_z)
    K_t = _assemble_weighted_1d_stiffness(
        seq.basis_t_jk, seq.d_basis_t_jk, seq.quad.w_y * beta_t, g_t)
    K_z = _assemble_weighted_1d_stiffness(
        seq.basis_z_jk, seq.d_basis_z_jk, seq.quad.w_z * gamma_z, g_z)

    V_t, mu = _assemble_1d_fd_eigendecomp(M_t, K_t)
    V_z, nu = _assemble_1d_fd_eigendecomp(M_z, K_z)

    # kappa from the assembled matrices, and the residual it leaves.
    kappa = float(jnp.sum(M_c * M_a) / jnp.sum(M_a * M_a))
    resid = float(jnp.linalg.norm(M_c - kappa * M_a) / jnp.linalg.norm(M_c))

    Ws, ds = [], []
    for k in range(int(nu.shape[0])):
        P_k = K_r + (kappa * nu[k]) * M_a
        W_k, d_k = _assemble_1d_fd_eigendecomp(M_b, P_k)   # W^T M_b W = I
        Ws.append(W_k)
        ds.append(d_k)
    return {"bulk_shape": bulk_shape, "V_t": V_t, "V_z": V_z, "mu": mu,
            "W": jnp.stack(Ws), "d": jnp.stack(ds), "kappa_resid": resid}


def modal_perk_apply(data, rhs_b):
    x = rhs_b.reshape(data["bulk_shape"])
    x = jnp.einsum('tj,rtz->rjz', data["V_t"], x)
    x = jnp.einsum('zk,rjz->rjk', data["V_z"], x)
    x = jnp.einsum('krs,rjk->sjk', data["W"], x)
    den = data["d"].T[:, None, :] + data["mu"][None, :, None]   # (s, j, k)
    x = jnp.where(jnp.abs(den) > 1e-12, x / jnp.where(den == 0, 1.0, den), 0.0)
    x = jnp.einsum('krs,sjk->rjk', data["W"], x)
    x = jnp.einsum('zk,rjk->rjz', data["V_z"], x)
    x = jnp.einsum('tj,rjz->rtz', data["V_t"], x)
    return x.reshape(-1)


def modal_radial_apply(data, rhs_b):
    x = rhs_b.reshape(data["bulk_shape"])                    # (r, t, z)
    x = jnp.einsum('tj,rtz->rjz', data["V_t"], x)
    x = jnp.einsum('zk,rjz->rjk', data["V_z"], x)
    x = jnp.einsum('jkrs,sjk->rjk', data["A_pinv"], x)
    x = jnp.einsum('zk,rjk->rjz', data["V_z"], x)
    x = jnp.einsum('tj,rjz->rtz', data["V_t"], x)
    return x.reshape(-1)


# --------------------------------------------------------------------------- #
# geometries
# --------------------------------------------------------------------------- #
def w7x_map(path, stride):
    """W7-X map via the recipe in scripts/debug/w7x_vacuum_bfield_project.py.

    Three details matter, all of them learned the hard way:

    * **Interpolatory collocation at data resolution** on a NON-polar fit
      sequence (``n_basis = n_data``), then an analytic ``map_func`` passed to
      ``set_map``. L2-projecting the Cartesian map onto the coarse polar solve
      basis instead produces a degenerate geometry.
    * **The toroidal sign is auto-picked so det(DF) > 0.** This is not cosmetic:
      ``J < 0`` makes M2 indefinite and CG returns NaN. W7-X resolves to
      sign = -1, so hardcoding ``+sin`` silently yields a NaN geometry.
    * ``stride`` keeps the fit sequence small. Building it at full data
      resolution is what stalled the stride-1 runs for hours.
    """
    import h5py  # noqa: PLC0415
    from mrx.differential_forms import DiscreteFunction  # noqa: PLC0415
    from mrx.projectors import _solve_tensor_collocation_axis  # noqa: PLC0415

    with h5py.File(path, "r") as h:
        at = dict(h.attrs)
        nfp = int(at["nfp"])
        nr, nt, nz = (int(at[k]) for k in ("n_rho", "n_theta", "n_zeta"))
        R = np.asarray(h["R"]).reshape(nr, nt, nz)
        Z = np.asarray(h["Z"]).reshape(nr, nt, nz)
        ep = np.asarray(h["eval_points"]).reshape(nr, nt, nz, 3)
    sl = (slice(None, None, stride),) * 3
    R, Z, ep = R[sl], Z[sl], ep[sl]
    rho, theta, zeta = ep[:, 0, 0, 0], ep[0, :, 0, 1], ep[0, 0, :, 2]
    fit_ns = (len(rho), len(theta), len(zeta))
    R0 = float(np.mean(R))
    R, Z = R / R0, Z / R0

    fit_seq = DeRhamSequence(fit_ns, (P,) * 3, 2 * P, TYPES, polar=False)
    fit_seq.evaluate_1d()
    br, bt, bz = fit_seq.basis_0.Λ
    colls = (jnp.asarray(br.collocation_matrix(jnp.asarray(rho))),
             jnp.asarray(bt.collocation_matrix(jnp.asarray(theta))),
             jnp.asarray(bz.collocation_matrix(jnp.asarray(zeta))))

    def fit(vals):
        c = jnp.asarray(vals)
        for axis, coll in enumerate(colls):
            c = _solve_tensor_collocation_axis(coll, c, axis=axis)
        return c

    cR, cZ = fit(R), fit(Z)
    R_h = DiscreteFunction(fit_seq.e0 @ cR.reshape(-1), fit_seq.basis_0, fit_seq.e0)
    Z_h = DiscreteFunction(fit_seq.e0 @ cZ.reshape(-1), fit_seq.basis_0, fit_seq.e0)
    a_nfp = 2 * jnp.pi / nfp

    def make(sign):
        def f(x):
            ang = a_nfp * x[2]
            r = R_h(x)[0]
            return jnp.array([r * jnp.cos(ang), sign * r * jnp.sin(ang), Z_h(x)[0]])
        return f

    # Orientation: median det(DF) over interior samples decides the sign.
    rng = np.random.default_rng(0)
    pts = jnp.asarray(rng.uniform(0.05, 0.95, size=(256, 3)))
    dets = jax.vmap(lambda x: jnp.linalg.det(jax.jacfwd(make(1.0))(x)))(pts)
    sign = 1.0 if float(jnp.median(dets)) > 0 else -1.0
    print(f"    [w7x] fit_ns={fit_ns} nfp={nfp} toroidal sign={sign:+.0f}", flush=True)
    return make(sign)


def build(map_fn):
    seq = DeRhamSequence(NS, (P,) * 3, 2 * P, TYPES, polar=True,
                         tol=1e-12, maxiter=1000, betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    seq.set_map(map_fn)
    return seq


def pcg(A, b, Minv, tol, maxit):
    x = jnp.zeros_like(b)
    r = b - A(x)
    z = Minv(r)
    q = z
    rz = float(r @ z)
    nb = float(jnp.linalg.norm(b))
    for i in range(1, maxit + 1):
        Aq = A(q)
        den = float(q @ Aq)
        if den <= 0.0:
            return -i
        al = rz / den
        x = x + al * q
        r = r - al * Aq
        if float(jnp.linalg.norm(r)) / nb < tol:
            return i
        z = Minv(r)
        rz_n = float(r @ z)
        q = z + (rz_n / rz) * q
        rz = rz_n
    return maxit


GEOMS = [
    ("toroid (control)", lambda: build(toroid_map(epsilon=1 / 3, R0=1.0))),
    ("rot-ellipse nfp3", lambda: build(rotating_ellipse_map(
        eps=0.33, kappa=1.5, nfp=3))),
    ("W7-X (h5 spline)", lambda: build(w7x_map(args.h5, args.stride))),
]

print(f"{'geometry':<20} {'BC':>5} {'n_bulk':>7} {'fd':>7} {'flat':>7} "
      f"{'modal':>7} {'per-k':>7} {'gain':>6}", flush=True)
for name, mk in GEOMS:
    try:
        seq = mk()
    except Exception as exc:                                   # noqa: BLE001
        print(f"{name:<20}  BUILD FAILED: {type(exc).__name__}: {exc}", flush=True)
        continue
    jac = np.asarray(seq.geometry.jacobian_j)
    # Section 7.5: gate on the PROJECTED Jacobian sign. A folded projection
    # returns nan from production and the measurement would mean nothing.
    # NaN must be caught explicitly: `jac.min() <= 0` is False for NaN, so a
    # degenerate projection would sail past the gate and burn maxit iterations
    # on a meaningless operator (observed on W7-X, 2026-08-17).
    if not np.all(np.isfinite(jac)) or jac.min() <= 0.0:
        bad = "non-finite" if not np.all(np.isfinite(jac)) else "folded"
        print(f"{name:<20}  SKIP ({bad}): projected det(J) in "
              f"[{np.nanmin(jac):.3e}, {np.nanmax(jac):.3e}]; "
              f"{int(np.sum(~np.isfinite(jac)))} non-finite quad points", flush=True)
        continue
    print(f"{name:<20}  det(J) in [{jac.min():.3e}, {jac.max():.3e}]", flush=True)
    ops = assemble_incidence_operators(seq)
    seq.set_operators(ops)
    cs = _core_size(seq)

    for dbc in (False, True):
        size = int(seq.n0_dbc if dbc else seq.n0)

        def A(xb, dbc=dbc, size=size):
            full = jnp.zeros((size,)).at[cs:].set(xb)
            return apply_stiffness(seq, ops, full, 0, dirichlet=dbc)[cs:]

        n_b = size - cs
        A(jnp.zeros(n_b))            # eager warmup before any tracing

        fd = _assemble_k0_greville_bulk_factors(seq, dirichlet=dbc)
        f_fd = _build_k0_tensor_hodge_preconditioner_factors(
            core_size=cs, schur_inv=jnp.eye(cs), bulk_data=fd)
        md = modal_radial_bulk_data(seq, dirichlet=dbc)
        mf = modal_radial_bulk_data(seq, dirichlet=dbc, angular_profiles=False)
        mk_ = modal_perk_bulk_data(seq, dirichlet=dbc)

        rng = np.random.default_rng(0)
        b = jnp.asarray(rng.standard_normal(n_b))
        it_fd = pcg(A, b, lambda r: _apply_k0_tensor_hodge_bulk_inverse(f_fd, r),
                    args.tol, args.maxit)
        it_mf = pcg(A, b, lambda r: modal_radial_apply(mf, r),
                    args.tol, args.maxit)
        it_md = pcg(A, b, lambda r: modal_radial_apply(md, r),
                    args.tol, args.maxit)
        it_pk = pcg(A, b, lambda r: modal_perk_apply(mk_, r),
                    args.tol, args.maxit)
        gain = (it_fd / it_pk) if it_pk > 0 else float("nan")
        print(f"{'':<20} {'dbc' if dbc else 'free':>5} {n_b:>7} {it_fd:>7} "
              f"{it_mf:>7} {it_md:>7} {it_pk:>7} {gain:>5.2f}x  "
              f"(kappa resid {mk_['kappa_resid']:.3f})", flush=True)
    print(flush=True)

print("negative counts = CG breakdown (preconditioned operator not SPD)")
