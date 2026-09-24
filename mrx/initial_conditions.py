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
  the initial condition of an analytic geometry file (``scripts/relax.py``,
  ``mrx.geometry.read_analytic``).
* :func:`clebsch_potential_form` with :func:`potential_two_form`: an
  equilibrium file's own field, ``B = dA'`` from its Clebsch data
  (:func:`mrx.gvec.load_clebsch`), the production route.

:func:`project_reference_two_form` turns the analytic form into DoFs. It
pushes the form forward and uses ``load(frame='phys')``: ``load(frame='ref')``
wants ``g omega / J`` and fails silently on ``omega``.
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
    seq = seq.odd                              # lambda is a sine series: odd
    dof = seq.interpolate(lam_h, 0)
    return (float(seq.l2_norm_sq(dof, 0, False)),
            float(dof @ seq.apply_laplacian(dof, 0, dirichlet=False)))


def clebsch_potential_form(cb, seed=None):
    """Reference 1-form ``A'`` whose exterior derivative is the Clebsch 2-form
    of the data ``cb`` (:func:`mrx.gvec.load_clebsch`).

    With the GVEC potential ``A = Phi dtheta_G - chi dzeta_G + Phi dLA`` and
    the gauge term ``d(Phi LA)`` dropped,
    ``A' = (-LA dPhi_dr, 2 pi Phi, -(2 pi / nfp) chi)`` in MRX's logical
    ``(rho, theta, zeta)`` -- ``dA'`` is exactly the pointwise Clebsch 2-form
    ``(0, (chi' - Phi' LA_zeta) / nfp, Phi' (1 + LA_theta))`` up to the
    common ``2 pi``. It needs only VALUES of lambda: the derivatives that
    make the field, and the second derivatives that make the current, are
    taken by the discrete ``d`` in :func:`potential_two_form`, so a coarsely
    sampled lambda cannot inject grid-scale current through its
    interpolant. ``Phi`` and ``chi`` are the antiderivatives of the file's
    profiles with ``Phi(0) = chi(0) = 0`` and a vanishing slope on the axis
    (both are ``rho^2`` there).

    ``seed = (m, n, rho0, width, eps, phase)`` adds the resonant term
    ``eps |Phi'(rho0)| / m  g(rho) cos(2 pi (m theta - s n zeta - phase))``
    to ``A'_zeta``, with ``g = exp(-((rho - rho0) / width)^2) (1 - rho^2) /
    (1 - rho0^2)`` and ``s`` the sign of the file's ``iota``: ``eps`` is the
    resonant normal field ``|dB^rho| / |B^zeta|`` at ``rho0``, the chain
    sits where ``|iota| = nfp n / m`` (:func:`resonant_rho`), the wall trace
    stays a function of ``rho`` alone (``B . n = 0`` exactly), and the
    island the seed opens has full width about
    ``1.6 sqrt(eps nfp / (m |iota'|))`` in ``rho`` (pendulum estimate; the
    seed's non-resonant part is ``O(eps / (m width))``). ``phase`` shifts the
    resonant angle in TURNS: the seed enters as ``cos(2 pi phase) A_c + sin(2
    pi phase) A_s``, so every linear functional of it is sinusoidal in
    ``phase`` and two evaluations determine the whole scan. It moves the
    chain's O-points along the angle; the energy and helicity a seed costs
    depend on it separately (``dE/da = <J, dA>`` against the resonant
    harmonic of the CURRENT, ``dH/da = 2 <B, dA>`` against that of the
    FIELD), which is what makes a helicity-neutral seed possible.
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
        m, n, rho0, width, eps, phase = seed
        s = float(np.sign(np.mean(np.asarray(cb["dchi"]) / np.asarray(cb["dPhi"]))))
        amp = eps * abs(float(np.interp(rho0, rho, cb["dPhi"]))) / m

        def seed_zeta(x):
            r = x[0]
            g = jnp.exp(-((r - rho0) / width) ** 2) * (1.0 - r ** 2) / (1.0 - rho0 ** 2)
            return amp * g * jnp.cos(two_pi * (m * x[1] - s * n * x[2] - phase))

    def A_ref(x):
        r = jnp.clip(x[0], 0.0, 1.0)
        r_lam = jnp.clip(r, rho_g[0], rho_g[-1])
        la = lam(jnp.array([r_lam, x[1] % 1.0, x[2] % 1.0]))
        return jnp.array([-la * jnp.interp(r, rho_g, dPhi_g),
                          two_pi * jnp.interp(r, r_t, Phi_t),
                          -two_pi / nfp * jnp.interp(r, r_t, chi_t) + seed_zeta(x)])
    return A_ref


def parse_seed(spec, eps, phase=0.0):
    """The ``seed`` of :func:`clebsch_potential_form` from a command line's
    ``"m,n,rho0,width"`` with its amplitude ``eps`` and ``phase`` (turns):
    ``(m, n, rho0, width, eps, phase)``."""
    m, n, rho0, width = (float(v) for v in spec.split(","))
    return int(m), int(n), rho0, width, eps, phase


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
    seq = seq.odd
    A = seq.interpolate(A_ref, 1, dirichlet=False, frame='ref')
    B_full = seq.apply_incidence_matrix(A, 1, dirichlet_in=False, dirichlet_out=False)
    B = seq.apply_incidence_matrix(A, 1, dirichlet_in=False, dirichlet_out=True)
    n_full, norm = float(seq.l2_norm(B_full, 2, False)), float(seq.l2_norm(B, 2))
    wall = abs(n_full ** 2 - norm ** 2) ** 0.5 / norm
    return B / norm, norm, wall


def parallel_seed(seq, B, m, n, profile, r_grid):
    r"""SIESTA's island seed: ``dB = curl(A_par B / |B|)`` on an EXISTING field.

    ``A_par = profile(rho) cos(2 pi (m theta - s n zeta))`` with ``s`` the sign of the field's iota, as
    :func:`clebsch_potential_form`. ``profile`` is the radial function sampled on ``r_grid``, amplitude included:
    a Gaussian envelope, or one radial basis function when the profile is being solved for rather than prescribed
    (the seed-selection solve expands it in the sequence's own radial basis). It needs no ``(1 - rho^2)`` factor --
    the load and the solve are on the DIRICHLET 1-form space, whose tangential trace already vanishes on the wall.
    The profile's SIGN is the seed's only phase freedom, because a phase shift multiplies the seed by
    ``cos(2 pi phase)`` and nothing else -- the quadrature part of the perturbation is ODD under ``(theta, zeta) ->
    (-theta, -zeta)`` and the stellarator parity projector removes it exactly (measured: ``||dB||^2`` at phase 1/4
    is 1e-25 of its phase-0 value).

    Why parallel rather than one covariant component (:func:`clebsch_potential_form`'s ``A'_zeta`` seed): the energy
    a seed costs is ``dE/da = <B, dB> = <J, dA>``, so with ``dA = A_par B / |B|`` it is ``int (J . B / |B|) A_par``,
    the resonant harmonic of the PARALLEL CURRENT -- the quantity the shielding sheets carry and the one tearing
    responds to. A single covariant component instead weights a metric-dependent projection of ``J`` with no
    physical meaning, and its sign is not interpretable.

    The construction is a projection, not an interpolation: ``B / |B|`` is the discrete field, known at the
    quadrature points and not in closed form. The covariant components ``dA_i = A_par g_ij B^j / sqrt(g_kl B^k B^l)``
    (the Jacobian cancels between the 2-form's density components and ``|B|``) are loaded onto the DIRICHLET 1-form
    space and the exact incidence curl takes them to the Dirichlet 2-form space. Dirichlet on the 1-form matters:
    ``d d = 0`` only holds along a sub-complex, so projecting onto the FREE space and restricting the curl afterwards
    discards a wall-normal part whose divergence does not vanish -- measured, that gave ``||div dB|| / ||dB||`` up to
    0.8 for the 3/5 chain at ``rho0 = 0.79``, where a narrow envelope sits close to the wall (2026-09-19). The
    interpolated Clebsch seed escapes this only because its tangential ``A'`` is a function of ``rho`` alone, so the
    discarded piece is exactly zero. With the tangential trace constrained, nothing is discarded and ``div dB`` is
    round-off.

    The profile's scale is NOT the Clebsch seed's ``eps`` (there ``eps`` is the resonant ``|dB^rho| / |B^zeta|`` at
    ``rho0``, so the pendulum width follows from it in closed form; here it scales ``A_par`` itself). The two seed
    families are therefore compared by the WELL DEPTH ``-<B, dB>^2 / (2 ||dB||^2)``, which is the best energy a
    family can buy and does not depend on how the family is parametrised.

    Returns ``(dB, div)``: the 2-form increment and ``||d dB|| / ||dB||``, which is round-off when the construction
    is right (it reached 0.8 when the 1-form went onto the FREE space instead).
    """
    seq = seq.odd
    Bq = seq.evaluate_at_quadrature(B, 2, dirichlet=True)                  # contravariant density, (n_q, 3)
    g_B = jnp.einsum('qij,qj->qi', seq.metric_jkl, Bq)                     # g_ij B^j, up to the common 1 / J
    Bhat_cov = g_B / jnp.sqrt(jnp.einsum('qi,qi->q', g_B, Bq))[:, None]    # B_i / |B|: the Jacobian cancels
    x, w = seq.quad.x, seq.quad.w
    s = float(jnp.sign(jnp.sum(w * Bq[:, 1]) / jnp.sum(w * Bq[:, 2])))     # sign of the flux-ratio iota
    A_par = (jnp.interp(x[:, 0], jnp.asarray(r_grid), jnp.asarray(profile))
             * jnp.cos(2.0 * jnp.pi * (m * x[:, 1] - s * n * x[:, 2])))
    load = seq.vector_load_values(A_par[:, None] * Bhat_cov, 1, 1, dirichlet_n=True)
    dA = seq.apply_inverse_mass_matrix(load, 1, dirichlet=True)
    dB = seq.apply_incidence_matrix(dA, 1, dirichlet_in=True, dirichlet_out=True)
    div = float(seq.l2_norm(seq.apply_incidence_matrix(dB, 2, dirichlet_in=True, dirichlet_out=True), 3)
                / seq.l2_norm(dB, 2))
    return dB, div


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

    seq = seq.odd
    B_raw = seq.apply_inverse_mass_matrix(
        seq.load(B_phys, 2, dirichlet=True), 2, dirichlet=True)
    norm = float(seq.l2_norm(B_raw, 2))
    return B_raw / norm, norm


def leray_clean(seq, B):
    """Leray-project ``B`` and renormalise to ``||B||_M = 1``.

    The evolution ``dB = curl E`` preserves ``div B`` exactly, so whatever
    divergence the initial condition carries it carries for the whole run;
    this removes it once, up front. Returns ``(B, diff_B)`` with ``diff_B`` the
    M-norm of the removed part.
    """
    seq = seq.odd
    B_leray, _ = seq.apply_leray_projection(B, k=2)
    diff_B = float(seq.l2_norm(B_leray - B, 2))
    return B_leray / float(seq.l2_norm(B_leray, 2)), diff_B


# ---------------------------------------------------------------------------
# The initial field of a run
# ---------------------------------------------------------------------------

def initial_field(seq, seed=None):
    """The initial field of a relaxation run and what was measured on the way.

    The sequence's geometry file decides (``seq.equilibrium``, parsed once
    by :func:`mrx.geometry.build_sequence`): an
    equilibrium file (VMEC wout, GVEC state) gives its own field ``B = dA'``
    through the histopolated Clebsch potential, exactly divergence-free,
    optionally with a resonant ``seed = (m, n, rho0, width, eps, phase)``; an
    analytic geometry file gives the logical-grid field of its ``profile``
    block, L2-projected and Leray-cleaned. ``||B||_M = 1`` in both cases.

    Returns ``(B, info)`` with ``B`` the Dirichlet 2-form DoFs and ``info``
    the numbers a driver prints and records: ``kind``, ``B_norm_raw``,
    ``div_raw``, ``div``, ``leray_moved``, and for an equilibrium file
    ``nfp``, ``iota_axis``, ``iota_edge`` (per full turn), ``wall_discarded``,
    ``lambda_norm_sq``, ``lambda_dirichlet_energy`` and, with a seed,
    ``seed_rho`` (the file's resonant surface).
    """
    from mrx.gvec import load_clebsch  # noqa: PLC0415
    from mrx.relaxation import compute_divergence_norm  # noqa: PLC0415

    eq = seq.equilibrium
    if eq is None:
        raise ValueError("the sequence has no geometry file: build it with mrx.geometry.build_sequence")
    kind = eq["kind"]
    if kind in ("gvec", "vmec"):
        cb = load_clebsch(eq, nfp=seq.nfp)
        lam_norm, lam_energy = lambda_dirichlet_energy(cb["lam_h"], seq)
        info = dict(kind=kind, nfp=int(cb["nfp"]),
                    iota_axis=float(cb["dchi"][1] / cb["dPhi"][1]),
                    iota_edge=float(cb["dchi"][-1] / cb["dPhi"][-1]),
                    lambda_norm_sq=float(lam_norm), lambda_dirichlet_energy=float(lam_energy))
        if seed is not None:
            info["seed_rho"] = float(resonant_rho(cb, seed[0], seed[1]))
        B, norm, wall = potential_two_form(seq, clebsch_potential_form(cb, seed))
        div = float(compute_divergence_norm(B, seq))
        info.update(B_norm_raw=float(norm), wall_discarded=float(wall),
                    div_raw=div, div=div, leray_moved=0.0)
        return B, info
    if seed is not None:
        raise ValueError("a resonant seed needs an equilibrium file")
    prof = eq["profile"]
    iota, dPhi = make_profiles(prof["iota"][0], prof["iota"][1], prof["iota_exp"], prof["flux_exp"])
    modes = [(int(m), int(n), float(a)) for m, n, a in prof.get("lambda", [])]
    B, norm = project_reference_two_form(seq, analytic_profile_form(iota, dPhi, make_lambda(modes)))
    div_raw = float(compute_divergence_norm(B, seq))
    B, moved = leray_clean(seq, B)
    return B, dict(kind=kind, iota=[float(v) for v in prof["iota"]], iota_exp=float(prof["iota_exp"]),
                   flux_exp=float(prof["flux_exp"]), lambda_modes=len(modes),
                   B_norm_raw=float(norm), div_raw=div_raw, div=float(compute_divergence_norm(B, seq)),
                   leray_moved=float(moved))
