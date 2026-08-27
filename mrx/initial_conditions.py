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

The helicity is the one the relaxation conserves,
:func:`mrx.relaxation.compute_helicity`: ``H = <A, B + B_harm>`` with ``A``
the vector potential solved in the Dirichlet 1-form space (``curl A = B -
B_harm``, ``B_harm`` the harmonic remainder that carries the toroidal flux).
``H`` is gauge-invariant and is what the flow
conserves: ``dH/dt = <E, B + B_harm> + <curl A_D, E> = 2 <v x B, B> = 0``,
the ``+ B_harm`` being what cancels the harmonic terms.

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


# This function can be used to compute lambda from the other parameters. 
# Currently not pursued.
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


def lambda_dirichlet_energy(lam_h, seq) -> tuple[float, float]:
    """``(||lam||_M^2, <lam, L_0 lam>)`` of the Greville interpolant of ``lam_h``
    on the 0-form space (natural BC): the smoothness gauge of a Clebsch
    stream function. The current sees lambda's mixed second derivatives, so a
    coarsely sampled export shows up in the ratio of the two before it shows
    up in ``||J||/||B||``."""
    dof = seq.interpolate(lam_h, 0)
    return (float(seq.l2_norm_sq(dof, 0, False)),
            float(dof @ seq.apply_laplacian(dof, 0, dirichlet=False)))


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


def clebsch_potential_form(cb, seed=None):
    """Reference 1-form ``A'`` whose exterior derivative is the Clebsch 2-form
    of :func:`clebsch_form`, from the same ``cb``.

    With the GVEC potential ``A = Phi dtheta_G - chi dzeta_G + Phi dLA`` and
    the gauge term ``d(Phi LA)`` dropped,
    ``A' = (-LA dPhi_dr, 2 pi Phi, -(2 pi / nfp) chi)`` in MRX's logical
    ``(rho, theta, zeta)`` -- ``dA'`` is exactly ``clebsch_form``'s
    ``(0, (chi' - Phi' LA_zeta) / nfp, Phi' (1 + LA_theta))`` up to the
    common ``2 pi``. It needs only VALUES of lambda: the derivatives that
    make the field, and the second derivatives that make the current, are
    taken by the discrete ``d`` in :func:`potential_two_form`, so a coarsely
    sampled lambda cannot inject grid-scale current through its
    interpolant. ``Phi`` and ``chi`` are the antiderivatives of the file's
    profiles with ``Phi(0) = chi(0) = 0`` and a vanishing slope on the axis
    (both are ``rho^2`` there).

    ``seed = (m, n, rho0, width, eps)`` adds the resonant term
    ``eps |Phi'(rho0)| / m  g(rho) cos(2 pi (m theta - s n zeta))`` to
    ``A'_zeta``, with ``g = exp(-((rho - rho0) / width)^2) (1 - rho^2) /
    (1 - rho0^2)`` and ``s`` the sign of the file's ``iota``: ``eps`` is the
    resonant normal field ``|dB^rho| / |B^zeta|`` at ``rho0``, the chain
    sits where ``|iota| = nfp n / m`` (:func:`resonant_rho`), the wall trace
    stays a function of ``rho`` alone (``B . n = 0`` exactly), and the
    island the seed opens has full width about
    ``1.6 sqrt(eps nfp / (m |iota'|))`` in ``rho`` (pendulum estimate; the
    seed's non-resonant part is ``O(eps / (m width))``).
    """
    from scipy.interpolate import CubicSpline

    rho = np.asarray(cb["rho"], dtype=np.float64)
    r_fine = np.linspace(0.0, 1.0, 2001)
    # Both profiles vanish on the axis; add that point unless the grid has it.
    lead = rho[0] > 0.0
    r_prof = np.r_[0.0, rho] if lead else rho

    def profile(v):
        return np.r_[0.0, v] if lead else np.asarray(v, dtype=np.float64)

    Phi_t = jnp.asarray(CubicSpline(r_prof, profile(cb["dPhi"])).antiderivative()(r_fine))
    chi_t = jnp.asarray(CubicSpline(r_prof, profile(cb["dchi"])).antiderivative()(r_fine))
    r_t = jnp.asarray(r_fine)
    rho_g, dPhi_g = jnp.asarray(rho), jnp.asarray(cb["dPhi"])
    lam = cb["lam_h"]
    nfp = cb["nfp"]
    two_pi = 2.0 * jnp.pi

    if seed is None:
        def seed_zeta(x):
            return 0.0
    else:
        m, n, rho0, width, eps = seed
        s = float(np.sign(np.mean(np.asarray(cb["dchi"]) / np.asarray(cb["dPhi"]))))
        amp = eps * abs(float(np.interp(rho0, rho, cb["dPhi"]))) / m

        def seed_zeta(x):
            r = x[0]
            g = jnp.exp(-((r - rho0) / width) ** 2) * (1.0 - r ** 2) / (1.0 - rho0 ** 2)
            return amp * g * jnp.cos(two_pi * (m * x[1] - s * n * x[2]))

    def A_ref(x):
        r = jnp.clip(x[0], 0.0, 1.0)
        r_lam = jnp.clip(r, rho_g[0], rho_g[-1])
        la = lam(jnp.array([r_lam, x[1] % 1.0, x[2] % 1.0]))
        return jnp.array([-la * jnp.interp(r, rho_g, dPhi_g),
                          two_pi * jnp.interp(r, r_t, Phi_t),
                          -two_pi / nfp * jnp.interp(r, r_t, chi_t) + seed_zeta(x)])
    return A_ref


def resonant_rho(cb, m, n):
    """``rho`` where the file's ``|iota|`` equals ``nfp n / m`` (linear
    interpolation between the profile's samples; ``nan`` if it never does)."""
    rho = np.asarray(cb["rho"], dtype=np.float64)
    iota = np.abs(np.asarray(cb["dchi"], dtype=np.float64) / np.asarray(cb["dPhi"], dtype=np.float64))
    target = cb["nfp"] * n / m
    k = np.nonzero(np.diff(np.sign(iota - target)))[0]
    if len(k) == 0:
        return float("nan")
    i = k[0]
    return float(rho[i] + (target - iota[i]) * (rho[i + 1] - rho[i]) / (iota[i + 1] - iota[i]))


def potential_two_form(seq, A_ref):
    """``B = d A'`` in the complex: histopolate the reference 1-form on the
    FREE 1-form space (its wall-tangential part is the toroidal flux, which
    no gauge removes) and apply the exact incidence curl into the Dirichlet
    2-form space.

    ``div B = 0`` to round-off (``d d = 0``) and ``B . n = 0`` on the wall
    exactly -- the tangential components of ``A'`` there are functions of
    ``rho`` alone, so every wall face has zero circulation -- so nothing is
    projected and no Leray step is needed. The only fit is the commuting
    histopolation, whose resolution is the mesh's: this is the route that
    keeps a coarse export's interpolation error out of the current. Returns
    ``(B, norm, wall)``: the DoFs normalised to ``||B||_M = 1``, the norm
    before normalisation, and the relative wall-normal part discarded by the
    Dirichlet restriction (a check, not a correction).
    """
    A = seq.interpolate(A_ref, 1, dirichlet=False, frame='ref')
    B_full = seq.apply_incidence_matrix(A, 1, dirichlet_in=False, dirichlet_out=False)
    B = seq.apply_incidence_matrix(A, 1, dirichlet_in=False, dirichlet_out=True)
    n_full, norm = float(seq.l2_norm(B_full, 2, False)), float(seq.l2_norm(B, 2))
    wall = abs(n_full ** 2 - norm ** 2) ** 0.5 / norm
    return B / norm, norm, wall


def dzeta_form():
    """The constant reference 2-form ``(0, 0, 1)``.

    Zero shear; minimising the energy at fixed
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
    this removes it once, up front. Returns ``(B, diff_B)`` with ``diff_B`` the
    M-norm of the removed part.
    """
    B_leray, _ = seq.apply_leray_projection(B, k=2)
    diff_B = float(seq.l2_norm(B_leray - B, 2))
    return B_leray / float(seq.l2_norm(B_leray, 2)), diff_B
