"""Relaxation initial conditions as reference 2-forms.

Every initial condition here is a k=2 field given in the reference 2-form
frame (component order ``dchi^dzeta, dr^dzeta, dr^dchi``: the rho-, chi- and
zeta-directed logical fluxes), which is the GVEC/VMEC representation since the
reference components are exactly ``sqrt(g) B^i`` (``B_phys = DF B_hat / J``):

    B_hat^rho  = 0
    B_hat^chi  = Phi'(rho) (iota(rho) - dlambda/dzeta)
    B_hat^zeta = Phi'(rho) (1 + dlambda/dchi)

Three properties hold for any ``lambda`` and any geometry before a solve:
``B^rho = 0`` (nested surfaces), ``div B = 0`` (the mixed partials cancel) and
``B.n = 0`` (what the k=2 Dirichlet space enforces). ``lambda`` redistributes
the field within a surface: the fluxes, iota and the helicity do not depend on
it; the force and the energy do.

Helicity is metric-free. With ``A = Phi dchi - X dzeta`` and ``X' = iota Phi'``,

    H = int_0^1 Phi^2 (X/Phi)' drho,                                       (1)

zero for constant iota, and for the power-law profiles of
:func:`make_profiles` it closes as :func:`analytic_helicity`. On a torus the
helicity is gauge-ambiguous by the harmonic 1-form, so (1) and
:func:`mrx.relaxation.compute_helicity` do not have to agree; the difference is
the harmonic contribution.

Three sources of the profiles:

* :func:`analytic_profile_form`: prescribed power laws, no external data;
  the ``analytic`` initial condition of ``scripts/relax.py``.
* :func:`clebsch_form`: GVEC's own ``dPhi_dr``, ``dchi_dr`` and ``lambda``
  from a file read by :func:`mrx.gvec.load_clebsch`; the equilibrium field
  rebuilt from three scalars instead of resampled as a vector.
* :func:`dzeta_form`: the constant 2-form ``(0, 0, 1)``, whose relaxation has
  an exactly known target, the harmonic field.

:func:`project_reference_two_form` turns any of them into DoFs. It pushes the
form forward and uses ``load(frame='phys')``: ``load(frame='ref')`` wants
``g omega / J`` and fails silently on ``omega``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Prescribed profiles
# ---------------------------------------------------------------------------

def make_profiles(iota0, iota1, iota_exp, flux_exp):
    """Return ``(iota, dPhi)`` with ``iota = iota0 + (iota1-iota0) rho^iota_exp``
    and ``dPhi = rho^flux_exp``.

    ``flux_exp = 1`` is the uniform-toroidal-field choice: ``B^zeta`` is a flux
    density in logical coordinates, so a uniform physical ``B_z`` has
    ``dPhi/drho`` proportional to ``rho``.
    """
    def iota(r):
        return iota0 + (iota1 - iota0) * r ** iota_exp

    def dPhi(r):
        return r ** flux_exp

    return iota, dPhi


def parse_lambda(spec):
    """``"m,n,amp;m,n,amp;..."`` -> list of ``(m, n, amp)``. Empty string -> []."""
    if not spec:
        return []
    modes = []
    for term in spec.split(";"):
        m, n, amp = term.split(",")
        modes.append((int(m), int(n), float(amp)))
    return modes


def make_lambda(modes):
    """``lambda = sum_k amp_k rho^|m_k| sin(2 pi (m_k chi - n_k zeta))``.

    ``rho^|m|`` is the regularity condition at the polar axis. Returns the two
    angular derivatives ``(d_chi, d_zeta)`` at a point; lambda itself never
    appears in the field.
    """
    def dlam(x):
        rho, chi, zeta = x[0], x[1], x[2]
        d_chi = 0.0
        d_zeta = 0.0
        for m, n, amp in modes:
            phase = 2 * jnp.pi * (m * chi - n * zeta)
            radial = amp * rho ** abs(m) if m != 0 else amp
            d_chi = d_chi + radial * 2 * jnp.pi * m * jnp.cos(phase)
            d_zeta = d_zeta - radial * 2 * jnp.pi * n * jnp.cos(phase)
        return d_chi, d_zeta

    return dlam


def analytic_helicity(iota0, iota1, iota_exp, flux_exp):
    """Eq. (1) for the power-law profiles, un-normalised field, natural gauge.

    Linear in the shear ``iota1 - iota0`` and zero without it.
    """
    q, e = flux_exp, iota_exp
    return (iota1 - iota0) * e / ((q + 1) * (q + e + 1) * (2 * q + e + 2))


def metric_coefficients(seq, rhos, n_ang):
    """Surface averages ``a = <g_cc/J>``, ``b = <g_cz/J>``, ``c = <g_zz/J>``,
    ``V' = <J>`` on ``rhos``.

    The surface-averaged energy density of the field is
    ``u = f_zeta^2 (a iota^2 + 2 b iota + c) / (2 V')``, a quadratic in iota
    with the energy-minimising transform ``-b/a``; ``b`` is zero on the
    cylinder and is the 3-D term. chi and zeta are periodic on [0, 1), so a
    uniform grid mean is the surface integral for smooth integrands.
    """
    DF = jax.jacfwd(seq.map)

    def coeffs_at(x):
        dF = DF(x)
        g = dF.T @ dF
        jac = jnp.linalg.det(dF)
        return jnp.array([g[1, 1] / jac, g[1, 2] / jac, g[2, 2] / jac, jac])

    ang = (jnp.arange(n_ang) + 0.5) / n_ang
    cc, zz = jnp.meshgrid(ang, ang, indexing='ij')

    def surface(rho):
        pts = jnp.stack([jnp.full(cc.size, rho), cc.ravel(), zz.ravel()],
                        axis=-1)
        return jnp.mean(jax.vmap(coeffs_at)(pts), axis=0)

    out = np.asarray(jax.vmap(surface)(jnp.asarray(rhos)))
    return out[:, 0], out[:, 1], out[:, 2], out[:, 3]


# ---------------------------------------------------------------------------
# Reference 2-forms
# ---------------------------------------------------------------------------

def analytic_profile_form(iota, dPhi, dlam):
    """Reference 2-form of prescribed ``iota(rho)``, ``dPhi(rho)`` and lambda."""
    def omega_ref(x):
        r = x[0]
        f = dPhi(r)
        d_chi, d_zeta = dlam(x)
        return jnp.array([0.0, f * (iota(r) - d_zeta), f * (1.0 + d_chi)])

    return omega_ref


def clebsch_form(cb):
    """Reference 2-form of the Clebsch data ``cb`` from
    :func:`mrx.gvec.load_clebsch`.

    Units: the file's derivatives are with respect to radian angles and MRX's
    zeta spans one field period, so ``Phi' = 2 pi dPhi_dr``,
    ``iota = dchi_dr / (nfp dPhi_dr)`` and ``lambda = LA / 2 pi``. The 2 pi on
    ``Phi'`` divides out of the normalised field.
    """
    rho_g = jnp.asarray(cb["rho"])
    dPhi_g = jnp.asarray(cb["dPhi"])
    dchi_g = jnp.asarray(cb["dchi"])
    grad_lam = jax.grad(cb["lam_h"])
    nfp = cb["nfp"]
    two_pi = 2.0 * jnp.pi

    def omega_ref(x):
        r = jnp.clip(x[0], rho_g[0], rho_g[-1])
        f_phi = jnp.interp(r, rho_g, dPhi_g)
        f_chi = jnp.interp(r, rho_g, dchi_g) / nfp
        g = grad_lam(jnp.array([r, x[1] % 1.0, x[2] % 1.0])) / two_pi
        lam_t, lam_z = g[1], g[2]
        return jnp.array([0.0, f_chi - f_phi * lam_z, f_phi * (1.0 + lam_t)])

    return omega_ref


def dzeta_form():
    """The constant reference 2-form ``(0, 0, 1)``.

    Zero shear, hence zero helicity by eq. (1); minimising the energy at fixed
    toroidal flux lands on the harmonic 2-form of the Dirichlet complex, which
    is curl-free, so the target field, J = 0 and a flat pressure are all known
    in advance.
    """
    def omega_ref(x):
        return jnp.array([0.0, 0.0, 1.0])

    return omega_ref


def project_reference_two_form(seq, omega_ref):
    """L2-project a reference 2-form onto the Dirichlet k=2 space.

    Returns ``(B, norm)``: the DoFs normalised to ``||B||_M = 1`` and the norm
    before normalisation. The projection goes through ``M_2``, which carries
    the metric and couples the components, so it can reintroduce a small
    ``rho`` component and a divergence; :func:`leray_clean` removes the
    latter.
    """
    DF_map = jax.jacfwd(seq.map)

    def B_phys(x):
        dF = DF_map(x)
        return dF @ omega_ref(x) / jnp.linalg.det(dF)

    B_raw = seq.apply_inverse_mass_matrix(
        seq.load(B_phys, 2, dirichlet=True), 2, dirichlet=True)
    norm = float(seq.l2_norm(B_raw, 2))
    return B_raw / norm, norm


def divergence_norm(seq, B):
    """``||div B||_L2`` through the incidence operator (exact ``d``, no solve)."""
    return float(seq.l2_norm(seq.apply_incidence_matrix(
        B, 2, dirichlet_in=True, dirichlet_out=True), 3))


def leray_clean(seq, B):
    """Leray-project ``B`` and renormalise to ``||B||_M = 1``.

    The evolution ``dB = curl E`` preserves ``div B`` exactly, so whatever
    divergence the initial condition carries it carries for the whole run;
    this removes it once, up front. Returns ``(B, moved)`` with ``moved`` the
    M-norm of the removed part.
    """
    B_leray, _ = seq.apply_leray_projection(B, k=2)
    moved = float(seq.l2_norm(B_leray - B, 2))
    return B_leray / float(seq.l2_norm(B_leray, 2)), moved
