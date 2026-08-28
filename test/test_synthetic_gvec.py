"""The GVEC route end to end on a synthetic state file, and THE relaxation run.

``test/synthetic_gvec.py`` writes a ``GVEC_State_*.dat`` in the layout
``mrx.gvec.read_state`` parses, from closed formulas on a circular torus:
nfp = 5, a W7-X-like transform ``iota = -0.9 - 0.15 rho^2`` per turn, a
small stellarator-symmetric lambda and a beta = 1e-3 pressure profile.
Everything that a real state goes through -- ``read_state`` (the parser),
``build_sequence`` (the map's spline coefficients from the series
coefficients, the handedness measurement), ``load_clebsch`` (the profile splines, lambda in closed
form), ``clebsch_form`` (the radian-to-normalised unit conversions) and the
projection -- is then checked against the formulas the file was written
from, which no data file allows. The module fixture is a (4, 8, 4) p=2
sequence on that file, with the production operators and the harmonic
forms; the relaxation run of ``mrx.relaxation`` lives here on its
Clebsch initial condition, so the one relaxation test in the suite runs on
the production geometry route.

Tests:

* the parsed state reproduces the formulas (``R``, ``Z``, ``LA`` and the
  profiles to round-off: the writer is the inverse of the parser);
* the installed map reproduces the analytic torus to the map fit's error
  (stated band), and ``det DF > 0`` selected ``Y = -R sin(2 pi zeta/nfp)``;
* the projected Clebsch field matches the contract
  ``sqrt(g) B = (0, dchi_dr - dPhi_dr dLA_dz, dPhi_dr (1 + dLA_dt))``
  evaluated from the formulas (relative L2, the projection error), and is
  divergence-free after ``leray_clean``;
* without lambda, ``B^theta / B^zeta = chi' / Phi' = iota(rho) / nfp``
  pointwise, and with lambda the surface means of the components are
  unchanged (lambda redistributes within a surface);
* the relaxation run: the most general stepper -- CG descent with the
  analytic linesearch, the CFL cap, the ``(1 + mu L)^-1``
  hyperregularisation (``gamma = 1``) and the backward-Euler resistive
  solve from the first step -- compiled ONCE and driven for ``STEPS`` steps.
  Along the trajectory the energy falls at every step and the measured
  drop agrees with the linesearch prediction
  ``dE = -dt (F, u)_M (1 - dt / 2 dt*)`` for the ideal part of the step
  (exact for the quadratic energy, up to the mass solves at ``seq.tol``),
  the resistive part lowers it further; ``div B`` stays at the initial
  condition's; ``dt = min(dt*, cfl / cfl_max)``; ``eta = 0`` reproduces the
  ideal step ``B_n + dt curl E`` and skips the solve; at ``eta > 0`` the
  solve satisfies its equation; and the helicity rate of the resistive
  step. With ``E = -u x B + eta J`` the identity is ``dH/dt = -2 eta <J, B>``;
  for the backward-Euler step ``delta = -eps curl J_{n+1}`` (``eps = eta dt``)
  the polarised form
  ``H(B_{n+1}) - H(B_ideal) = -eps (<J_{n+1}, B_ideal> + <J_{n+1}, B_{n+1}>)``
  is exact for the quadratic helicity, and either single-time form
  ``-2 eps <J, B>`` is off by ``O(eps^2)`` -- the one at ``B_{n+1}`` with a
  far smaller constant, since the implicit step's current IS ``J_{n+1}``.
  Halving ``eta`` at the same state halves ``eps`` with the ideal part of
  the step unchanged, which is the ``dt -> dt/2`` refinement of the
  resistive substep without a second compiled stepper; the single-time
  error must fall by ~4x.
"""

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mrx
from mrx.geometry import build_sequence
from mrx.gvec import evaluate, load_clebsch, profile_spline, read_state
from mrx.initial_conditions import (
    clebsch_form,
    clebsch_potential_form,
    divergence_norm,
    resonant_rho,
    leray_clean,
    potential_two_form,
    project_reference_two_form,
)
from mrx.nullspace import compute_nullspaces
from mrx.relaxation import (
    DescentMethod,
    TimeStepper,
    compute_helicity,
    initial_state,
    resistive_step,
)
from test.synthetic_gvec import TWO_PI, write_synthetic_state
from test.manufactured import relative_l2_error

# The torus of the session fixture (R0 = 1, a = 1/3) with W7-X's nfp and a
# W7-X-like transform; Phi_edge = pi a^2 makes the mean toroidal field 1.
R0, A, NFP = 1.0, 1.0 / 3.0, 5
IOTA = (-0.9, -0.15)
PHI_EDGE = np.pi * A ** 2
LAM_AMPLITUDE, BETA = 0.05, 1e-3
NS, P = (4, 8, 4), 2

# eta = 1e-2 puts eps = eta dt at ~3e-4 (dt* is ~3e-2 on this field), where
# the helicity solves resolve dH in float32 as well.
ETA, MU, CFL = 1e-2, 1e-2, 0.5
STEPS = 12
CHECK = 6           # the step at which the resistive identities are measured


def _write(path, lam_amplitude):
    return write_synthetic_state(path, R0=R0, a=A, nfp=NFP, iota=IOTA,
                                 Phi_edge=PHI_EDGE, lam_amplitude=lam_amplitude,
                                 beta=BETA)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """``(path, torus)`` of the state with lambda and of the one without."""
    d = tmp_path_factory.mktemp("synthetic_gvec")
    path = str(d / "GVEC_State_torus.dat")
    path0 = str(d / "GVEC_State_torus_nolambda.dat")
    return (path, _write(path, LAM_AMPLITUDE)), (path0, _write(path0, 0.0))


@pytest.fixture(scope="module")
def synthetic_seq(synthetic):
    (path, _), _ = synthetic
    t0 = time.perf_counter()
    seq, ops = build_sequence(path, NS, P)
    t1 = time.perf_counter()
    seq.set_operators(compute_nullspaces(seq, ops))
    print(f"\n  synthetic torus {NS} p={P}: build_sequence {t1 - t0:.0f} s, "
          f"nullspaces {time.perf_counter() - t1:.0f} s")
    return seq


def contract_two_form(torus):
    """The reference 2-form of the contract in docs/source/concepts/gvec_mrx_interface.md
    section 1, from the closed formulas: ``sqrt(g) B^theta = dchi_dr -
    dPhi_dr dLA/dzeta_G``, ``sqrt(g) B^zeta = dPhi_dr (1 + dLA/dtheta_G)``
    with the radian angles ``theta_G = 2 pi theta``, ``zeta_G = 2 pi zeta /
    nfp``, then the theta component divided by nfp because MRX's zeta spans
    one field period (the 2 pi common to both divides out)."""
    grad_LA = jax.grad(lambda x: torus.LA(x[0], x[1], x[2]))

    def omega(x):
        g = grad_LA(x)
        dLA_dt, dLA_dz = g[1] / TWO_PI, g[2] * torus.nfp / TWO_PI
        f, c = torus.dPhi_dr(x[0]), torus.dchi_dr(x[0])
        return jnp.array([0.0, (c - f * dLA_dz) / torus.nfp, f * (1.0 + dLA_dt)])

    return omega


@pytest.fixture(scope="module")
def clebsch_ic(synthetic, synthetic_seq):
    """``(B_raw, norm, B, moved)``: the projected Clebsch field and its
    Leray-cleaned, unit-norm version -- the relaxation's initial condition."""
    (path, _), _ = synthetic
    seq = synthetic_seq
    cb = load_clebsch(path)
    B_raw, norm = project_reference_two_form(seq, clebsch_form(cb))
    B, moved = leray_clean(seq, B_raw)
    return B_raw, norm, B, moved


@pytest.fixture(scope="module")
def potential_ic(synthetic, synthetic_seq):
    """``B = dA'`` from the histopolated Clebsch potential, unit norm -- the
    production initial condition (``scripts/relax.py --ic clebsch``)."""
    (path, _), _ = synthetic
    seq = synthetic_seq
    cb = load_clebsch(path)
    B, _, _ = potential_two_form(seq, clebsch_potential_form(cb))
    return B


def test_state_file_reproduces_the_formulas(synthetic):
    """``read_state`` on the written file gives back the torus: the series
    of ``X1``, ``X2``, ``LA`` on a tensor grid and the profile splines at
    arbitrary radii agree with the formulas to round-off (the writer is the
    parser's inverse, and every radial function is in the spline space)."""
    (path, torus), _ = synthetic
    st = read_state(path)
    assert st["nfp"] == NFP and st["deg"] == 5 and st["X1"]["sin_cos"] == 2
    assert st["X2"]["sin_cos"] == 1 and st["LA"]["sin_cos"] == 1
    assert abs(st["a_minor"] - A) <= 1e-15 and st["r_major"] == R0
    rho = np.array([0.0, 0.13, 0.5, 0.87, 1.0])
    th, ze = np.array([0.0, 0.2, 0.45, 0.7]), np.array([0.0, 0.3, 0.8])
    RHO, TH, ZE = np.meshgrid(rho, th, ze, indexing="ij")
    for blk, want in (("X1", torus.R(RHO, TH)), ("X2", torus.Z(RHO, TH)),
                      ("LA", torus.LA(RHO, TH, ZE))):
        got = evaluate(st[blk], st["sp"], rho, TWO_PI * th, TWO_PI * ze / NFP)
        assert np.abs(got - np.asarray(want)).max() <= 1e-13, blk
    r = np.linspace(0.0, 1.0, 37)
    for name, want in (("phi", torus.Phi(r)), ("chi", torus.chi(r)),
                       ("iota", torus.iota(r)), ("pressure", torus.pressure(r))):
        got = profile_spline(st, name)(r)
        assert np.abs(got - np.asarray(want)).max() <= 1e-12 * max(1.0, np.abs(want).max()), name
    dPhi = profile_spline(st, "phi").derivative()(r)
    assert np.abs(dPhi - np.asarray(torus.dPhi_dr(r))).max() <= 1e-12


def test_map_reproduces_the_torus(synthetic, synthetic_seq):
    """``build_sequence`` on the file installs ``F = (R cos phi, -R sin phi, Z)``
    with ``phi = 2 pi zeta / nfp`` (the sign ``det DF > 0`` selects for a
    theta running counter-clockwise in the (R, Z) plane) and R, Z within
    the map fit's error of the circle: the L2 projection of the series onto
    8 periodic p=2 splines in theta (the radial dependence is linear and
    exact)."""
    (_, torus), _ = synthetic
    seq = synthetic_seq
    x = seq.quad.x
    F = jax.vmap(seq.map)(x)
    R_h = jnp.hypot(F[:, 0], F[:, 1])
    phi_h = jnp.arctan2(-F[:, 1], F[:, 0]) % TWO_PI
    err_R = float(jnp.max(jnp.abs(R_h - torus.R(x[:, 0], x[:, 1])))) / A
    err_Z = float(jnp.max(jnp.abs(F[:, 2] - torus.Z(x[:, 0], x[:, 1])))) / A
    err_phi = float(jnp.max(jnp.abs(phi_h - TWO_PI * x[:, 2] / NFP)))
    print(f"\n  map at the quadrature points: max |R_h - R| / a {err_R:.2e}, "
          f"max |Z_h - Z| / a {err_Z:.2e}, max |phi_h - 2 pi zeta / nfp| {err_phi:.1e}")
    # The toroidal angle is exact: the map applies cos/sin to 2 pi zeta/nfp
    # itself, so only round-off (1e2 eps) separates it from the formula.
    assert err_phi <= mrx.eps(1e2)
    # Measured 2026-08-28 at NS (4, 8, 4) p=2 on the closed form (see the
    # print): 4.04e-3 of a in both R and Z (the gridded route's linear
    # bridge on 24 theta points gave 1.08e-2); bands at 1.25x.
    assert err_R <= 1.25 * 4.04e-3, err_R
    assert err_Z <= 1.25 * 4.04e-3, err_Z


def test_clebsch_field_matches_the_contract(synthetic, synthetic_seq, clebsch_ic):
    """``load_clebsch`` + ``clebsch_form`` + the projection reproduce the
    contract's ``sqrt(g) B^i`` from the state's profiles and ``LA`` to the
    projection error of the (4, 8, 4) p=2 space, and ``leray_clean`` takes
    the projection's divergence to solver tolerance."""
    (path, torus), _ = synthetic
    seq = synthetic_seq
    B_raw, norm, B, moved = clebsch_ic
    cb = load_clebsch(path)
    assert cb["nfp"] == NFP

    omega = contract_two_form(torus)
    err_raw = relative_l2_error(seq, 2, True, B_raw * norm, omega)
    err = relative_l2_error(seq, 2, True, B * norm, omega)
    div_raw = divergence_norm(seq, B_raw)
    div = divergence_norm(seq, B)
    print(f"\n  clebsch field: ||B||_M raw {norm:.4e}, relative L2 error vs the "
          f"contract {err_raw:.3e} raw / {err:.3e} cleaned, ||div B|| {div_raw:.2e} "
          f"-> {div:.2e}, moved {moved:.2e}")
    assert jnp.isfinite(B).all()
    assert abs(float(seq.l2_norm(B, 2)) - 1.0) <= mrx.eps(1e2)
    assert div <= 10 * seq.tol
    # Measured 2026-08-28 in float64 on the closed form (see the print):
    # 3.274e-3 raw and cleaned (the divergence the projection carries is
    # 2.5e-4, and the cleaning moves the field by 2.0e-5); bands at 1.25x
    # on the errors.
    # The moved norm is what the Leray solve resolves: its band is the
    # measured value plus the solve tolerance.
    assert err_raw <= 1.25 * 3.28e-3, err_raw
    assert err <= 1.25 * 3.28e-3, err
    assert moved <= 2 * 2.0e-5 + 10 * seq.tol, moved


def test_rotational_transform_and_lambda_invariance(synthetic, synthetic_seq):
    """Without lambda the field is ``(0, dchi_dr / nfp, dPhi_dr)``, so
    ``B^theta / B^zeta = iota(rho) / nfp`` pointwise: exactly at the radii
    where ``load_clebsch`` tabulates the profile splines (401 uniform
    points) and to the linear-interpolation error of the cubic ``dchi_dr``
    between them. With lambda the surface means of both components are
    unchanged."""
    (path, torus), (path0, torus0) = synthetic
    cb0 = load_clebsch(path0)
    omega0 = jax.jit(jax.vmap(clebsch_form(cb0)))
    rho_nodes = jnp.asarray(cb0["rho"])[1:]          # both fluxes vanish on the axis
    rho_mid = 0.5 * (rho_nodes[1:] + rho_nodes[:-1])
    ang = jnp.array([0.3, 0.7])
    for label, rho, band in (("nodes", rho_nodes, mrx.eps(1e2)),
                             ("midpoints", rho_mid, 1.25 * 7.9e-7)):
        w = omega0(jnp.column_stack([rho, jnp.full_like(rho, ang[0]),
                                     jnp.full_like(rho, ang[1])]))
        assert float(jnp.max(jnp.abs(w[:, 0]))) == 0.0
        ratio = w[:, 1] / w[:, 2]
        err = float(jnp.max(jnp.abs(ratio - torus0.iota(rho) / NFP)
                            / jnp.abs(torus0.iota(rho) / NFP)))
        print(f"\n  iota / nfp at the {label}: max relative error {err:.2e}")
        # Nodes: round-off (6.8e-16 measured). Midpoints: 7.81e-7 measured
        # 2026-08-28 on the 401-point tabulation (see the print), band 1.25x.
        assert err <= band, (label, err)

    cb = load_clebsch(path)
    omega = jax.jit(jax.vmap(clebsch_form(cb)))
    n = 48
    t, z = jnp.meshgrid((jnp.arange(n) + 0.5) / n, (jnp.arange(n) + 0.5) / n,
                        indexing="ij")
    for rho in (0.25, 0.75):
        pts = jnp.column_stack([jnp.full(n * n, rho), t.ravel(), z.ravel()])
        mean_lam = jnp.mean(omega(pts), axis=0)
        ref = omega0(jnp.array([[rho, 0.0, 0.0]]))[0]
        err = float(jnp.max(jnp.abs(mean_lam - ref)[1:] / jnp.abs(ref[1:])))
        print(f"  surface means at rho = {rho}: with lambda {np.asarray(mean_lam)[1:]}, "
              f"without {np.asarray(ref)[1:]}, relative difference {err:.2e}")
        # The angular derivatives of a periodic spline average to zero over
        # a period; the uniform-grid mean resolves that to round-off in
        # float64 (measured 2026-08-26, see the print) -- 1e3 eps.
        assert err <= mrx.eps(1e3), err


def test_relaxation(synthetic_seq, potential_ic):
    seq = synthetic_seq
    B0 = potential_ic
    ts = TimeStepper(seq=seq, descent_method=DescentMethod.LBFGS,
                     resistive=True, velocity_smoothing_order=1,
                     velocity_smoothing_scale=MU, cfl=CFL)

    @jax.jit
    def step(state):
        s = ts.relaxation_step(state)
        return eqx.tree_at(lambda t: t.B_n, s, s.B_nplus1)

    def energy(B):
        return 0.5 * seq.l2_norm_sq(B, 2)

    @jax.jit
    def probe(B_prev, state):
        """The ideal step reconstructed from the post-step state, its energy
        prediction, and the diagnostics of the full step."""
        curl_E = seq.apply_incidence_matrix(state.E, 1, dirichlet_in=True, dirichlet_out=True)
        B_ideal = B_prev + state.dt * curl_E
        Fu = state.F_prev @ seq.apply_mass_matrix(state.v, 2)
        dE_pred = -state.dt * Fu * (1.0 - 0.5 * state.dt / state.dt_star)
        div = seq.l2_norm(seq.apply_incidence_matrix(
            state.B_n, 2, dirichlet_in=True, dirichlet_out=True), 3)
        return B_ideal, energy(B_ideal), energy(state.B_n), dE_pred, div

    @jax.jit
    def current_pairing(B_J, B):
        """``<J, B>`` with ``J = curl B_J``: the L2 pairing of the Dirichlet
        1-form with the 2-form through the mixed mass."""
        J = seq.apply_weak_curl(B_J, dirichlet_in=True, dirichlet_out=True)
        return J @ seq.apply_projection_matrix(B, 2, 1, True, dirichlet_out=True)

    get_helicity = jax.jit(compute_helicity, static_argnames=["seq"])

    state = eqx.tree_at(lambda s: s.eta, initial_state(B0, ts), ETA)
    A = jnp.zeros(seq.n(1, True))
    E0 = float(energy(B0))
    H0, A = get_helicity(B0, seq, A)
    E_prev, F_prev = E0, float(state.F_norm)
    scale = float(jnp.max(jnp.abs(B0)))
    ratios_ideal, ratios_full, divs = [], [], []
    for n in range(1, STEPS + 1):
        if n == CHECK:
            pre = state          # the pre-step state the resistive checks restart from
        B_prev = state.B_n
        state = step(state)
        B_ideal, E_ideal, E_new, dE_pred, div = (probe(B_prev, state))
        E_ideal, E_new, dE_pred, div = (float(x) for x in (E_ideal, E_new, dE_pred, div))
        dt, dt_star, cfl_max = float(state.dt), float(state.dt_star), float(state.cfl_max)
        assert int(state.resistive_info) < 0, (
            f"step {n}: resistive MINRES did not converge ({int(state.resistive_info)})")
        assert E_ideal < E_prev and E_new < E_ideal, (n, E_prev, E_ideal, E_new)
        assert cfl_max > 0 and dt_star > 0
        assert abs(dt - min(dt_star, CFL / cfl_max)) <= 100 * mrx.eps() * dt
        ratios_ideal.append((E_ideal - E_prev) / dE_pred)
        ratios_full.append((E_new - E_prev) / dE_pred)
        divs.append(div)
        if n == CHECK:
            check_state, check_B_ideal = state, B_ideal
        E_prev = E_new
    print(f"\n  {STEPS} steps: E {E0:.6f} -> {E_prev:.6f}, |F| {F_prev:.3e} -> "
          f"{float(state.F_norm):.3e}, dt {dt:.3e} dt* {dt_star:.3e} "
          f"cap {'binds' if dt < dt_star else 'inactive'}, div B max {max(divs):.2e}, "
          f"dE_meas/dE_pred ideal [{min(ratios_ideal):.6f}, {max(ratios_ideal):.6f}] "
          f"full [{min(ratios_full):.4f}, {max(ratios_full):.4f}]")

    # --- the resistive step at CHECK, from the same pre-step state --------
    # eta = 0: the ideal step alone, the solve skipped.
    ideal = step(eqx.tree_at(lambda s: s.eta, pre, 0.0))
    assert int(ideal.resistive_info) == 0 and float(ideal.resistive_delta) == 0.0
    # Two executables (the step and the probe) reach B_ideal from the same
    # pre-step state. On the GPU the scatter-adds of the extraction and mass
    # applies are not deterministic, and the descent solves amplify that
    # round-off: measured 1.7e-12 .. 1.6e-11 absolute at max |B| = 3.6e-2
    # across four runs on two commits (2026-08-27; one earlier run gave
    # 2.4e-14). On the CPU the difference is below 32 eps.
    assert float(jnp.max(jnp.abs(ideal.B_n - check_B_ideal))) <= mrx.eps(1e7) * scale
    # eta > 0: (M_2 + eps L_2) B_{n+1} = M_2 B_ideal.
    B1 = check_state.B_n
    eps = float(check_state.dt) * ETA
    lhs = seq.apply_mass_matrix(B1, 2) + eps * seq.apply_laplacian(B1, 2)
    rhs = seq.apply_mass_matrix(check_B_ideal, 2)
    rel = float(jnp.linalg.norm(lhs - rhs) / jnp.linalg.norm(rhs))
    assert rel <= 10 * mrx.sqrt_eps(), rel

    # The helicity rate, at eps and eps/2.
    half = step(eqx.tree_at(lambda s: s.eta, pre, 0.5 * ETA))
    # The same executable on the same state. The line-search dt is a ratio
    # of quantities that pass through Krylov solves stopped at seq.tol, and
    # two runs whose scatter-add reduction order differs (the GPU) land
    # anywhere within that tolerance, so the band is tol-scaled, not
    # eps-scaled (and so meaningful in float32): the 10 sqrt_eps of the
    # resistive-solve check below. Measured on the H100 2026-08-28 across
    # five runs, three commits, CG and L-BFGS: 1.6e-10 .. 3.4e-9 relative
    # (3.3e-12 .. 1.05e-11 absolute at dt = 0.0211, 4.96e-8 at dt = 14.56);
    # on the CPU the difference is below 32 eps.
    assert abs(float(half.dt) - float(check_state.dt)) <= 10 * mrx.sqrt_eps() * float(check_state.dt)
    H_ideal, A = get_helicity(check_B_ideal, seq, A)
    results = {}
    for label, B_new, e in (("eps", B1, eps), ("eps/2", half.B_n, 0.5 * eps)):
        H_new, A = get_helicity(B_new, seq, A)
        dH = float(H_new - H_ideal)
        pred_n = -2.0 * e * float(current_pairing(check_B_ideal, check_B_ideal))
        pred_n1 = -2.0 * e * float(current_pairing(B_new, B_new))
        pred_mid = -e * float(current_pairing(B_new, check_B_ideal)
                              + current_pairing(B_new, B_new))
        results[label] = (dH, pred_n, pred_n1, pred_mid)
        print(f"  helicity rate at {label} = {e:.3e}: dH {dH:+.6e}, "
              f"-2 eps <J,B> at B_n {pred_n:+.6e} (err {dH - pred_n:+.2e}), "
              f"at B_n+1 {pred_n1:+.6e} (err {dH - pred_n1:+.2e}), "
              f"polarised {pred_mid:+.6e} (err {dH - pred_mid:+.2e})")
    (dH, pn, pn1, pm), (dHh, pnh, _, _) = results["eps"], results["eps/2"]
    print(f"  H {float(H0):+.5e} -> {float(get_helicity(state.B_n, seq, A)[0]):+.5e} "
          f"over {STEPS} resistive steps")

    # The ideal step's energy is a quadratic polynomial in dt, so the
    # prediction is exact up to the mass solves behind F, u and E at
    # seq.tol: measured 2026-08-26 (see the print) 1 +- 1e-6 in float64,
    # 1 +- 3e-3 in float32. The full step adds the resistive drop
    # -eps ||J||^2 on top, which does not shrink with |F|: the full ratio
    # ran from 1.49 (step 1) to 38.6 (step 12) at eta = 1e-2, so only its
    # lower bound is a statement.
    assert all(abs(r - 1.0) <= 1e3 * seq.tol for r in ratios_ideal), ratios_ideal
    assert all(r >= 1.0 - 1e3 * seq.tol for r in ratios_full), ratios_full
    assert max(divs) <= 10 * seq.tol
    # Measured 2026-08-28 in float64 at eps = 2.113e-4 (dt = 3.31e-2, see
    # the print): dH = +5.325938e-4; the polarised form is off by 2.1e-11
    # (3.8e-8 relative, the helicity solves); -2 eps <J, B> at B_{n+1} by
    # +1.27e-7 and at B_n by -9.64e-6 (ratio 0.0132), and at eps/2 by
    # +3.2e-8 and -2.45e-6: both 0.25x, the O(eps^2) quartering. So the
    # backward-Euler step makes the rate at B_{n+1} the accurate one (78x
    # smaller constant; the torus IC measured 175x), and the exact
    # statement is the polarised one. Bands: 1.25x on the 0.0129 ratio,
    # 1.4x on the quartering. The 1e2 tol |dH| terms are the helicity
    # solves' resolution of dH (2.8e-6 measured in float32, where they
    # carry the B_{n+1} assertion).
    floor = 1e2 * seq.tol * abs(dH)
    assert abs(dH - pm) <= floor
    assert abs(dH - pn1) <= 0.016 * abs(dH - pn) + floor
    assert abs(dHh - pnh) <= 0.35 * abs(dH - pn) + floor


def test_resistive_step_diffuses_the_field(synthetic_seq, potential_ic):
    """One backward-Euler step of ``dB/dt = -curl curl B`` lowers the energy,
    keeps the field divergence-free, and satisfies its own defect equation
    ``(M + eps L)(B1 - B) = -eps L B`` to the solver tolerance."""
    seq = synthetic_seq
    B = potential_ic
    eps = 1e-3
    B1, info, rel = jax.jit(lambda b: resistive_step(b, seq, eps))(B)
    assert int(info) < 0 and 0.0 < float(rel) < 0.1      # negative = converged in |info| iterations
    E0, E1 = (0.5 * float(seq.l2_norm_sq(b, 2)) for b in (B, B1))
    assert E1 < E0, (E0, E1)
    assert divergence_norm(seq, B1) <= 10 * seq.tol
    d = B1 - B
    lhs = seq.apply_mass_matrix(d, 2, True) + eps * seq.apply_laplacian(d, 2, dirichlet=True)
    rhs = -eps * seq.apply_laplacian(B, 2, dirichlet=True)
    resid = float(jnp.linalg.norm(lhs - rhs) / jnp.linalg.norm(rhs))
    assert resid <= 1e2 * seq.tol, resid


def test_potential_route_matches_the_projection(synthetic, synthetic_seq, clebsch_ic):
    """``B = d A'`` from the histopolated Clebsch potential is exactly
    divergence-free, drops nothing at the wall, and agrees with the
    projected-and-Leray-cleaned field of ``clebsch_form`` to the two routes'
    discretisation error."""
    (path, _), _ = synthetic
    seq = synthetic_seq
    cb = load_clebsch(path)
    B_pot, norm, wall = potential_two_form(seq, clebsch_potential_form(cb))
    _, _, B_proj, _ = clebsch_ic
    div = divergence_norm(seq, B_pot)
    diff = float(seq.l2_norm(B_pot - B_proj, 2))
    print(f"\n  potential route: ||dA'|| {norm:.4e}, div {div:.2e}, wall part {wall:.2e}, "
          f"||B_pot - B_proj||_M {diff:.3e}")
    assert div <= 10 * seq.tol, div
    assert wall <= 1e2 * mrx.eps(), wall
    assert abs(float(seq.l2_norm(B_pot, 2)) - 1.0) <= mrx.eps(1e2)
    assert diff <= 5e-2, diff


def test_seed_is_resonant_and_wall_tangential(synthetic, synthetic_seq):
    """The seed's ``d_theta A'_zeta / B^zeta`` at ``rho0`` is ``eps``, its
    phase is constant along the file's field lines at the surface
    ``resonant_rho`` finds, and it vanishes on the wall."""
    import jax
    (path, _), _ = synthetic
    cb = load_clebsch(path)
    m, n = 5, 1                       # |iota| = nfp n / m = 1 at rho = sqrt(2/3) of iota = -0.9 - 0.15 rho^2
    rho0 = resonant_rho(cb, m, n)
    assert rho0 == pytest.approx((2.0 / 3.0) ** 0.5, abs=2e-3), rho0
    eps = 1e-3
    A0 = clebsch_potential_form(cb)
    A1 = clebsch_potential_form(cb, (m, n, rho0, 0.1, eps))

    def dA_zeta(x):
        return A1(x)[2] - A0(x)[2]

    pts = jnp.array([[rho0, t, z] for t in (0.05, 0.3, 0.7) for z in (0.0, 0.4)])
    # amplitude: eps * |Phi'| / m, and d_theta of it is eps |B^zeta| = eps 2 pi |Phi'|
    b = jax.vmap(jax.grad(dA_zeta))(pts)[:, 1] / jax.vmap(jax.grad(lambda x: A0(x)[1]))(pts)[:, 0]
    # the amplitude reads Phi' from the profile's linear interpolant, the
    # denominator differentiates the tabulated antiderivative
    assert float(jnp.max(jnp.abs(b))) == pytest.approx(eps, rel=1e-3)
    # resonance: the phase 2 pi (m theta - s n zeta) is constant along dtheta/dzeta = iota / nfp
    iota = jnp.asarray(cb["dchi"]) / jnp.asarray(cb["dPhi"])
    q = float(jnp.interp(rho0, jnp.asarray(cb["rho"]), iota)) / cb["nfp"]
    x0 = jnp.array([rho0, 0.11, 0.0])
    x1 = jnp.array([rho0, 0.11 + q * 0.37, 0.37])
    assert float(dA_zeta(x0)) == pytest.approx(float(dA_zeta(x1)), rel=1e-6)
    # the wall trace of the seed is zero
    assert float(jnp.max(jnp.abs(jax.vmap(dA_zeta)(pts.at[:, 0].set(1.0))))) == 0.0


def test_section_figure_renders(synthetic_seq, potential_ic):
    """``mrx.poincare.section_figure`` traces the potential IC of the synthetic
    torus and renders: every line regular, iota inside the profile's range."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mrx.poincare import section_figure
    fig, res = section_figure(synthetic_seq, potential_ic, NFP, n_seeds=6, n_periods=20,
                              steps_per_period=16, saves_per_period=4, n_rays=1,
                              title="synthetic")
    plt.close(fig)
    keep = ~(res["escaped"] | ~res["ok"] | res["chaotic"])
    assert keep.all(), res["chaotic"]
    assert 0.85 < float(np.abs(res["iota"]).min()) and float(np.abs(res["iota"]).max()) < 1.1

