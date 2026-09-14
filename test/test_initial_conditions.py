"""Prescribed profiles, the resonant-surface interpolant, and Leray cleaning."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from mrx.initial_conditions import (
    analytic_profile_form,
    clebsch_potential_form,
    initial_field,
    lambda_dirichlet_energy,
    leray_clean,
    make_lambda,
    make_profiles,
    project_reference_two_form,
    potential_two_form,
    pulled_back_clebsch_form,
    resonant_rho,
)
from mrx.precision import eps


def test_make_profiles_hits_the_endpoints() -> None:
    iota, dphi = make_profiles(0.4, 0.9, 2.0, 1.0)
    assert float(iota(0.0)) == 0.4
    assert float(iota(1.0)) == 0.9
    assert float(dphi(0.5)) == 0.5


def test_make_lambda_matches_autodiff_including_m_zero() -> None:
    """Closed-form angular derivatives against ``jax.grad`` of the explicit sum."""
    modes = [(2, 1, 0.03), (3, -2, 0.01), (0, 1, 0.02)]

    def lam(x: jnp.ndarray) -> jnp.ndarray:
        rho, chi, zeta = x
        total = 0.0
        for m, n, amp in modes:
            radial = amp * rho ** abs(m) if m != 0 else amp
            total = total + radial * jnp.sin(2.0 * jnp.pi * (m * chi - n * zeta))
        return total

    x = jnp.array([0.37, 0.21, 0.63])
    grad = jax.grad(lam)(x)
    d_chi, d_zeta = make_lambda(modes)(x)
    assert abs(float(d_chi - grad[1])) < 1e-10
    assert abs(float(d_zeta - grad[2])) < 1e-10


def test_analytic_profile_form_ratio_is_iota() -> None:
    iota, dphi = make_profiles(0.4, 0.9, 2.0, 1.0)
    omega = analytic_profile_form(iota, dphi, make_lambda([]))
    value = omega(jnp.array([0.6, 0.3, 0.7]))
    assert abs(float(value[1] / value[2]) - float(iota(0.6))) < 1e-12
    assert float(value[0]) == 0.0


def test_resonant_rho_interpolates_and_returns_nan() -> None:
    """``iota = 2 rho``: ``nfp n / m = 0.5`` sits at ``rho = 0.25``."""
    cb = {"rho": np.linspace(0.0, 1.0, 11),
          "dchi": np.linspace(0.0, 1.0, 11) * 2.0,
          "dPhi": np.ones(11),
          "nfp": 1}
    assert abs(resonant_rho(cb, 2, 1) - 0.25) < 1e-12
    assert np.isnan(resonant_rho(cb, 1, 9))


def test_leray_clean_is_divergence_free_and_unit_norm(seq, b0) -> None:
    rng = np.random.default_rng(4)
    dirty = jnp.asarray(rng.standard_normal(seq.n(2, True)))
    cleaned, moved = leray_clean(seq, dirty)
    div = seq.apply_incidence_matrix(cleaned, 2, dirichlet_in=True, dirichlet_out=True)
    assert float(seq.l2_norm(div, 3)) < 1e2 * seq.tol
    assert abs(float(seq.l2_norm(cleaned, 2)) - 1.0) < eps(100)
    assert moved >= 0.0
    energy, dirichlet = lambda_dirichlet_energy(
        lambda x: jnp.sin(2.0 * jnp.pi * x[1]), seq)
    assert energy >= 0.0 and dirichlet >= 0.0


def test_project_reference_two_form_and_initial_field(seq) -> None:
    iota, dphi = make_profiles(0.4, 0.9, 2.0, 1.0)
    b, norm = project_reference_two_form(
        seq, analytic_profile_form(iota, dphi, make_lambda([])))
    assert norm > 0.0
    assert abs(float(seq.l2_norm(b, 2)) - 1.0) < eps(100)
    field, info = initial_field(seq)
    assert info["kind"] == "vmec"
    assert abs(float(seq.l2_norm(field, 2)) - 1.0) < eps(100)
    assert info["nfp"] == 3
    assert seq.map_source == "equilibrium"
    assert info["wall_discarded"] < 1e-10
    assert info["div"] < 1e2 * seq.tol


def test_pullback_on_the_equilibrium_map_is_a_noop(seq) -> None:
    """Inverting the sequence's own map recovers its logical coordinates.

    Then ``A_phys = (DF^T)^{-1} A_ref`` and ``interpolate(..., frame='phys')``
    applies ``DF^T``, so the pulled-back histopolation is the native one.
    This is the wall-flux gate: if ``wall_discarded`` is not small, the
    pullback is wrong.
    """
    from mrx.gvec import load_clebsch

    A_ref = clebsch_potential_form(load_clebsch(seq.equilibrium))
    B_ref, _, wall_ref = potential_two_form(seq, A_ref)
    B_phys, _, wall_phys = potential_two_form(
        seq, pulled_back_clebsch_form(seq, A_ref), frame="phys")
    assert wall_phys < 1e-3
    rel = float(seq.l2_norm(B_phys - B_ref, 2))
    assert rel < 1e-5, f"pullback disagreed with the native field by {rel:.2e}"


def test_both_maps_start_from_the_same_physical_field(seq, seq_map2disc) -> None:
    """map2disc of li383, pulled back, matches the equilibrium-map IC.

    Helicity and ``||F||`` are physical, so they must agree;
    ``wall_discarded`` and ``div`` must stay small. The map2disc path
    Leray-cleans the small wall remainder.
    """
    from mrx.relaxation import compute_force, compute_helicity

    B_eq, ic_eq = initial_field(seq)
    B_m2d, ic_m2d = initial_field(seq_map2disc)
    assert ic_eq["wall_discarded"] < 1e-8
    assert ic_m2d["wall_discarded"] < 1e-3
    assert ic_eq["div"] < 1e2 * seq.tol
    assert ic_m2d["div"] < 1e2 * seq_map2disc.tol

    H_eq = float(compute_helicity(B_eq, seq, jnp.zeros(seq.n(1, True)))[0])
    H_m2d = float(compute_helicity(B_m2d, seq_map2disc, jnp.zeros(seq_map2disc.n(1, True)))[0])
    F_eq = float(seq.l2_norm(compute_force(B_eq, seq)[0], 2))
    F_m2d = float(seq_map2disc.l2_norm(compute_force(B_m2d, seq_map2disc)[0], 2))
    assert abs(H_eq - H_m2d) < 0.05 * abs(H_eq), (
        f"helicity {H_eq:+.4e} vs {H_m2d:+.4e}")
    assert abs(F_eq - F_m2d) < 0.5 * F_eq, (
        f"||F|| {F_eq:.3e} vs {F_m2d:.3e}")
