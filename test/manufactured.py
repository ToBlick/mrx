"""Manufactured solutions of the eight Hodge Laplacians on the toroid.

One source of truth for ``test/test_poisson.py`` and
``scripts/poisson_study.py``: the study sets ``MRX_DTYPE`` on import, so the
generators live here and the study imports them.

The map is ``toroid_map(epsilon=a, R0=1)``; ``R = 1 + a r cos(2 pi chi)``.
Cases pair up under the Hodge star ``k <-> 3-k, free <-> Dirichlet`` into
four generators (``s = sin(pi r^2/2)``, ``c = cos(pi r^2/2)``):

  cos(2 pi zeta)               k0 free, k3 dbc    u = cos(2 pi zeta), f = u / R^2
  cos(pi r^2/2)                k0 dbc,  k3 free   u = c, f = (2 pi s + pi^2 r^2 c)/a^2
                                                      + pi r s cos(2 pi chi)/(a R)
  cos(2 pi zeta) dzeta         k1 free, k2 dbc    closed 1-form, f = grad sigma
  c cos(2 pi zeta) dzeta       k1 dbc,  k2 free   f = d sigma + curl curl omega

Harmonic dimensions (Betti numbers (1, 1, 0, 0)): k0 free 1 (the constant),
k1 free 1 (the toroidal 1-form), k2 dbc 1 (the toroidal 2-form), k3 dbc 1
(the constant 3-form); the other four pairs are non-singular. Every source
is orthogonal to its harmonic form (``cos`` has zero mean over zeta).
See ``docs/source/concepts/manufactured_solutions.md`` for the derivations.
"""

import jax
import jax.numpy as jnp

from mrx.quadrature import evaluate_at_xq

PI = jnp.pi

#: The eight ``(k, dirichlet)`` cases, in the order the study reports them.
CASES = [(0, False), (0, True), (1, False), (1, True),
         (2, False), (2, True), (3, False), (3, True)]


def case_tag(k, dirichlet):
    return f"k{k}_{'dbc' if dirichlet else 'nbc'}"


def _R(a, x):
    return 1.0 + a * x[0] * jnp.cos(2 * PI * x[1])


# --- generator A: u = cos(2 pi zeta)  (k0 free, k3 dbc) ---------------------
def u_cos(x):
    return jnp.cos(2 * PI * x[2]) * jnp.ones(1)


def make_f0_cos(a):
    """``-Delta cos(2 pi zeta) = cos(2 pi zeta) / R^2``."""
    def f(x):
        return jnp.cos(2 * PI * x[2]) / _R(a, x) ** 2 * jnp.ones(1)
    return f


# --- generator B: u = cos(pi r^2/2)  (k0 dbc, k3 free) ----------------------
def u_par(x):
    """Vanishes at r = 1 and is smooth at r = 0; not a spline."""
    return jnp.cos(0.5 * PI * x[0] ** 2) * jnp.ones(1)


def make_f0_par(a):
    def f(x):
        r = x[0]
        s = jnp.sin(0.5 * PI * r ** 2)
        c = jnp.cos(0.5 * PI * r ** 2)
        return ((2 * PI * s + PI ** 2 * r ** 2 * c) / a ** 2
                + PI * r * s * jnp.cos(2 * PI * x[1]) / (a * _R(a, x))) * jnp.ones(1)
    return f


# --- generator C: omega_1 = cos(2 pi zeta) dzeta  (k1 free, k2 dbc) ----------
# Closed, so f_1 = L_1 omega_1 = grad sigma with sigma = -div omega_1.
def _sigma1(a):
    def s(x):
        return jnp.sin(2 * PI * x[2]) / (2 * PI * _R(a, x) ** 2)
    return s


def _f1_cov_nbc(a):
    """Covariant source ``grad sigma``."""
    return jax.jacfwd(_sigma1(a))


# --- generator D: omega = cos(pi r^2/2) cos(2 pi zeta) dzeta  (k1 dbc, k2 free)
def _f1_cov_dbc(a):
    """Covariant source ``d sigma + delta d omega`` (docs/source/concepts/manufactured_solutions.md)."""
    def f(x):
        r, chi, z = x
        R = _R(a, x)
        s = jnp.sin(0.5 * PI * r ** 2)
        c = jnp.cos(0.5 * PI * r ** 2)
        z_cos, z_sin = jnp.cos(2 * PI * z), jnp.sin(2 * PI * z)
        chi_cos, chi_sin = jnp.cos(2 * PI * chi), jnp.sin(2 * PI * chi)
        fr = -a * chi_cos * c * z_sin / (PI * R ** 3)
        fchi = 2.0 * a * r * chi_sin * c * z_sin / R ** 3
        fz = z_cos * (c / R ** 2
                      + (2.0 * PI * s + PI ** 2 * r ** 2 * c) / a ** 2
                      - PI * r * s * chi_cos / (a * R))
        return jnp.array([fr, fchi, fz])
    return f


def _hodge_star_1to2_ref(a, alpha_cov, x):
    """``*: Omega^1 -> Omega^2`` in the ref proxy slots (chi zeta, r zeta, r chi):
    ``(J/g_rr, J/g_chichi, J/g_zetazeta) = (4 pi^2 r R, R/r, a^2 r/R)``."""
    r = x[0]
    R = _R(a, x)
    return jnp.array([4.0 * PI ** 2 * r * R * alpha_cov[0],
                      R / r * alpha_cov[1],
                      a ** 2 * r / R * alpha_cov[2]])


# --- load arguments ----------------------------------------------------------
# ``load(frame='ref')`` pairs the basis with the field by a plain dot product,
# so a k=1 source is passed raised (``G^-1 f_cov``) and a k=2 source as
# ``(G/J) (* f_1)``; k=3 takes the density ``f_0 J``.
def make_f1_ref(f_cov, F):
    DF = jax.jacfwd(F)

    def f(x):
        J_DF = DF(x)
        return jnp.linalg.solve(J_DF.T @ J_DF, f_cov(x))
    return f


def make_f1_phys(f_cov, F):
    DF = jax.jacfwd(F)
    f1r = make_f1_ref(f_cov, F)

    def f(x):
        return DF(x) @ f1r(x)
    return f


def make_f2_ref(a, f_cov, F):
    DF = jax.jacfwd(F)

    def f(x):
        proxy = _hodge_star_1to2_ref(a, f_cov(x), x)
        J_DF = DF(x)
        return (J_DF.T @ J_DF @ proxy) / jnp.linalg.det(J_DF)
    return f


def make_f2_phys(a, f_cov, F):
    DF = jax.jacfwd(F)
    f2r = make_f2_ref(a, f_cov, F)

    def f(x):
        return jnp.linalg.solve(DF(x).T, f2r(x))
    return f


def make_f3_ref(f0, F):
    DF = jax.jacfwd(F)

    def f(x):
        return f0(x) * jnp.linalg.det(DF(x))
    return f


# --- exact fields ------------------------------------------------------------
def v1_exact_ref(x):
    """``omega_1`` covariant: ``(0, 0, cos 2 pi zeta)``."""
    return jnp.array([0.0, 0.0, jnp.cos(2 * PI * x[2])])


def v1_dbc_exact_ref(x):
    return jnp.array([0.0, 0.0, jnp.cos(0.5 * PI * x[0] ** 2) * jnp.cos(2 * PI * x[2])])


def make_w2_exact_ref(a):
    """``* omega_1`` proxy: only the r chi slot, ``(a^2 r / R) cos 2 pi zeta``."""
    def w(x):
        return jnp.array([0.0, 0.0, a ** 2 * x[0] * jnp.cos(2 * PI * x[2]) / _R(a, x)])
    return w


def make_w2_nbc_exact_ref(a):
    def w(x):
        return jnp.array([0.0, 0.0, a ** 2 * x[0] * jnp.cos(0.5 * PI * x[0] ** 2)
                          * jnp.cos(2 * PI * x[2]) / _R(a, x)])
    return w


def case_specs(a, F):
    """``{(k, dirichlet): dict(src_ref, src_phys, exact)}`` for the eight cases.

    ``src_ref`` / ``src_phys`` are the ``frame='ref'`` / ``'phys'`` load
    callables; ``exact`` is the analytic field the error is measured against
    (physical scalar at k=0, 3; ref-frame covariant at k=1; ref proxy at k=2).
    """
    f0_cos, f0_par = make_f0_cos(a), make_f0_par(a)
    c_nbc, c_dbc = _f1_cov_nbc(a), _f1_cov_dbc(a)
    return {
        (0, False): dict(src_ref=f0_cos, src_phys=f0_cos, exact=u_cos),
        (0, True): dict(src_ref=f0_par, src_phys=f0_par, exact=u_par),
        (1, False): dict(src_ref=make_f1_ref(c_nbc, F), src_phys=make_f1_phys(c_nbc, F),
                         exact=v1_exact_ref),
        (1, True): dict(src_ref=make_f1_ref(c_dbc, F), src_phys=make_f1_phys(c_dbc, F),
                        exact=v1_dbc_exact_ref),
        (2, False): dict(src_ref=make_f2_ref(a, c_dbc, F), src_phys=make_f2_phys(a, c_dbc, F),
                         exact=make_w2_nbc_exact_ref(a)),
        (2, True): dict(src_ref=make_f2_ref(a, c_nbc, F), src_phys=make_f2_phys(a, c_nbc, F),
                        exact=make_w2_exact_ref(a)),
        (3, False): dict(src_ref=make_f3_ref(f0_par, F), src_phys=f0_par, exact=u_par),
        (3, True): dict(src_ref=make_f3_ref(f0_cos, F), src_phys=f0_cos, exact=u_cos),
    }


def relative_l2_error(seq, k, dirichlet, u_hat, exact):
    """Relative L2 error of the discrete k-form against the manufactured field.

    The norms are the sequence's own: ``int u^2 J`` at k=0, ``v^T G^-1 v J``
    for a covariant 1-form, ``w^T G w / J`` for a 2-form proxy and
    ``(u/J)^2 J`` for a 3-form density -- so only the stored metric is read,
    never ``DF``.
    """
    comp_info, comp_shapes = seq._form_comp_info(k)
    eT = seq.E(k, dirichlet).T
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    ncomp = 1 if k in (0, 3) else 3
    u_h = evaluate_at_xq(eT @ u_hat, comp_info, comp_shapes, quad_shape, ncomp)
    u_ex = jax.vmap(exact)(seq.quad.x)
    wJ = seq.quad.w * seq.jacobian_j
    if k == 3:
        u_h = u_h / seq.jacobian_j[:, None]
    if k == 1:
        weight = seq.metric_inv_jkl
    elif k == 2:
        weight = seq.metric_jkl / seq.jacobian_j[:, None, None] ** 2
    else:
        weight = jnp.ones((wJ.shape[0], 1, 1))
    diff = u_h - u_ex
    num = jnp.einsum("qi,qij,qj,q->", diff, weight, diff, wJ)
    den = jnp.einsum("qi,qij,qj,q->", u_ex, weight, u_ex, wJ)
    return float(jnp.sqrt(num / den))
