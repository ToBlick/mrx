"""The poloidal flux slips and the two targets they serve, the current and the transform.

As unknowns of the Newton step (:class:`mrx.relaxation.twist.Slips`) the twists of the radial spline steps change the
field by a divergence-free form without a ``zeta`` component, their map's transpose is its transpose, the energy's
derivative along each is ``-Z`` times the spline mean of the enclosed current, the same as the weak current's to the
discretisation, and the slip block is the energy's second derivative, positive definite. A user's run with
``--current.target`` and ``--pressure.target`` from one namelist with ``NCURR = 1`` ends at both targets, and so does a
user's run with ``--iota.target`` and ``--pressure.target`` from one namelist with ``NCURR = 0``, whose seed waits for
the transform and sits at the target's resonances.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from mrx.flux_label import ANISOTROPY
from mrx.precision import DTYPE, eps
from mrx.relaxation.loop import TimeStepper, initial_state, relax
from mrx.relaxation.physics import compute_divergence_norm, current, weak_pressure
from mrx.relaxation.seeding import energy_seed
from mrx.relaxation.shells import ERROR_BINS, PROFILE_CELLS, build_shells, label, load_target, pressure_error
from mrx.relaxation.twist import (AT_TARGET, build_slips, current_error, current_on_label, current_target,
                                  enclosed_current, iota_fit, iota_slips, load_current_target, load_iota_target)

#: The cells of the profile space on the fixture. Its 14 radial cells resolve 8 cells of the profile space, not the
#: default 16 of finer meshes: with 16 the slip block's condition is 1.1e7 and the moments miss the weak current by
#: 7.4e-3, with 8 the condition is 59 and the miss 1.55e-3 (2026-11-18).
CELLS = 8
#: The Newton steps of the user-facing tests, in chunks of relax.py's default length with the label and
#: the shells rebuilt before each.
STEPS, CHUNK = 20, 10
#: The interior of the label where the user-facing test compares transforms.
S_TEST = np.linspace(0.05, 0.95, 19)
#: The transform's miss after the steps and the pressure's error: measured values in the test's docstring.
MISS_END, PRESSURE_ERROR = 8e-3, 1.5e-2
#: The moments of the slips against the profile-space means of the weak current: 1.55e-3 of the largest in float64
#: and in float32, on CELLS cells (2026-11-18).
SLIP_WEAK_J = 2e-3
#: The current's and the pressure's error after the twenty steps with both targets: 8.50e-4 and 4.68e-3 in float64
#: (2026-11-18).
CURRENT_ERROR, PC_PRESSURE_ERROR = 3e-3, 1e-2

def test_iota_and_pressure_targets_from_one_namelist(seq, b0, tmp_path):
    """The run of ``scripts/relax.py --iota.target input.x --pressure.target input.x --seed`` with ONE VMEC input
    namelist: the transform 0.95 times the file's and the pressure the file's reshaped by ``1 + 0.05 (1 - 2 s)``.

    Twenty Newton steps in two chunks from the file's field, with the label, the shells and the slips at the current
    target the transform asks for (:func:`mrx.relaxation.twist.iota_slips`) rebuilt before each, take positive steps
    and lower ``W`` at every step. The seed waits for the transform: it is applied at step 10, where the label fit
    misses the target by less than AT_TARGET of its largest value. At the end the transform misses by 4.60e-3 (0.14 of
    the change of 3.29e-2) and the pressure by 6.98e-3, against 2.81e-2 from the file's pressure, in float64. At the
    surfaces of the 6 chains seeded inside the label's interior the target's transform is within 2.2e-3 of the
    chain's, the file's at least 2.1e-2 off. The chain next to the axis, (15,2) at x 0.12, is 9.4e-3 off, where the
    low shear makes a small error in the transform a large one in the radius. The bounds are MISS_END, PRESSURE_ERROR, a third of the transform's change, half the pressure's error
    against the file's profile and AT_TARGET of the transform for the seeds, so a run that ignored either target or
    seeded at the file's transform fails. 411 s on four CPU cores after the fixture (2026-11-18). Skipped in float32,
    where a Newton step costs three times as much on the CPU.
    """
    if DTYPE != np.float64:
        pytest.skip("a float32 Newton step costs three times a float64 one on the CPU, the float64 run covers the claim")
    iota_file, p_file = seq.equilibrium["profiles"]["iota"], seq.equilibrium["profiles"]["pressure"]
    s = np.linspace(0.0, 1.0, 101)
    row = lambda x: " ".join(f"{v:.17g}" for v in x)  # noqa: E731
    path = tmp_path / "input.target"
    path.write_text(f"&INDATA\n  NFP = {seq.nfp}, NCURR = 0, PIOTA_TYPE = 'cubic_spline'\n"
                    f"  AI_AUX_S = {row(s)}\n  AI_AUX_F = {row(0.95 * np.abs(iota_file(np.sqrt(s))))}\n"
                    f"  PMASS_TYPE = 'cubic_spline'\n  AM_AUX_S = {row(s)}\n"
                    f"  AM_AUX_F = {row(p_file(np.sqrt(s)) * (1.0 + 0.05 * (1.0 - 2.0 * s)))}\n/\n")
    iota_t = load_iota_target(str(path))
    p_t, p_f = load_target(seq, str(path)).profile, load_target(seq, "file").profile
    want = np.abs(np.asarray(iota_t(S_TEST)))
    change_iota = np.abs(np.abs(iota_file(np.sqrt(S_TEST))) - want).max()

    def miss(B, s_q):
        return np.abs(np.abs(iota_fit(seq, B, T, np.asarray(s_q), 6)(S_TEST)) - want).max()

    ts = TimeStepper(seq=seq, newton=True)
    state, dW, dt_star, E0, T, rows, seed_step = initial_state(b0, ts), [], [], None, None, None, None
    for k in range(STEPS // CHUNK):
        s_q, T, _ = label(seq, state.B_n, ANISOTROPY, T)
        if rows is None and miss(state.B_n, s_q) < AT_TARGET * want.max():
            # relax.py's seed with the energy form: at the first chunk boundary where the transform is at its target
            B_seeded, rows = energy_seed(seq, state.B_n, T=T, verbose=False)
            state, seed_step = initial_state(B_seeded, ts), k * CHUNK
            s_q, T, _ = label(seq, state.B_n, ANISOTROPY, T)
        ts = ts.with_shells(build_shells(seq, state.B_n, s_q, p_t(s_q), ts.newton_penalty, ts.newton_mass_solve_tol,
                                         CELLS))
        ts = ts.with_slips(iota_slips(seq, state.B_n, T, np.asarray(s_q), iota_t, ts, CELLS)[0])
        res = relax(state, ts, steps=CHUNK, chunk=CHUNK, verbose=False)
        state, E0 = res.state, res.E0 if E0 is None else E0
        dW += list(np.asarray(res.trace["dW"], dtype=float))
        dt_star += list(np.asarray(res.trace["dt_star"], dtype=float))
    s_q, T, _ = label(seq, state.B_n, ANISOTROPY, T)
    miss_end = miss(state.B_n, s_q)
    p_w = weak_pressure(current(state.B_n, seq), state.B_n, seq)
    err, _ = pressure_error(seq, p_w, s_q, p_t(s_q), ERROR_BINS)
    err_file, _ = pressure_error(seq, p_w, s_q, p_f(s_q), ERROR_BINS)

    # each chain where the target's transform is within AT_TARGET of the chain's, the seed's tolerance on the
    # transform, and where the file's is not: a radius error is large where the shear is low, a transform error not
    tol, off_new, off_file = AT_TARGET * want.max(), [], []
    for r in rows or []:
        res_ = seq.nfp * r["n"] / r["m"]
        off_new.append(abs(abs(float(iota_t(r["x"] ** 2))) - res_))
        off_file.append(abs(abs(float(iota_file(r["x"]))) - res_))
        print(f"  seed ({r['m']},{r['n']}) x {r['x']:.4f}: the target's transform there off by {off_new[-1]:.2e}, the "
              f"file's by {off_file[-1]:.2e}")
    print(f"\n  iota miss {miss_end:.3e} at the end, change {change_iota:.3e}. |p - p*| / |p*| {err:.3e}, against the "
          f"file's profile {err_file:.3e}. max dW / E0 {max(dW) / E0:+.1e}. Seeded at step {seed_step}")
    assert min(dt_star) > 0.0
    assert max(dW) < eps() * E0, f"W not monotone: {dW}"
    assert miss_end < MISS_END and miss_end < change_iota / 3
    assert err < PRESSURE_ERROR and err < 0.5 * err_file
    # on the interior of the label where the transform is compared, next to the axis the slips do not reach
    inner = [k for k, r in enumerate(rows or []) if S_TEST[0] < r["x"] ** 2 < S_TEST[-1]]
    assert inner and seed_step is not None
    assert max(off_new[k] for k in inner) < tol < min(off_file[k] for k in inner)


def test_slip_move_and_derivative(seq, b0):
    """The slips on the fixture's field and label, with the target 0.9 times its current: ``T d`` for a random ``d``
    is divergence-free with no ``zeta`` component to round-off and ``move_t`` is the transpose of ``move`` to
    round-off. ``W(d) = E(B + T d) - d . g`` is quadratic in ``d``, so its central difference along ``d`` is the
    assembled ``d . ((B, T_j)_M - g_j)`` and its second difference ``d . D d`` with the block ``D`` up to the
    round-off of the energy over the step. The moments ``-(B, T_j)_M / Z`` are the spline means of the weak
    current's enclosed flux to SLIP_WEAK_J of the largest. The block is positive definite, its eigenvalues 1.24 to
    73.2, the slip at the first spline included. 7 s on four CPU cores after the fixture, 16 s in float32
    (2026-11-18)."""
    _, T, _ = label(seq, b0, ANISOTROPY)
    mu_file = current_on_label(seq, b0, T, CELLS)
    slips, mu = build_slips(seq, b0, T, 0.9 * mu_file)
    move, move_t = slips.maps(seq)
    odd = seq.odd
    M2 = odd.M[2]
    rng = np.random.default_rng(0)
    d = jnp.asarray(rng.standard_normal(mu.size))
    y = jnp.asarray(rng.standard_normal(b0.size))
    Td = move(d)
    adjoint = abs(float(y @ Td - d @ move_t(y))) / float(jnp.linalg.norm(y) * jnp.linalg.norm(Td))
    div = float(compute_divergence_norm(Td, seq) / odd.l2_norm(Td, 2))
    Tdq = odd.evaluate_at_quadrature(Td, 2)
    zeta = float(jnp.abs(Tdq[:, 2]).max() / jnp.abs(Tdq).max())

    def W(t):
        Bt = b0 + t * Td
        return 0.5 * float(Bt @ (M2 @ Bt)) - t * float(d @ slips.g)
    E0 = W(0.0)
    D = np.linalg.inv(np.asarray(slips.D_inv, dtype=float))
    dDd = float(np.asarray(d) @ D @ np.asarray(d))
    h = 1e-2 * (E0 / dDd) ** 0.5                       # the step changes E by about 1e-4 E0
    fd = (W(h) - W(-h)) / (2 * h)
    sd = (W(h) - 2 * E0 + W(-h)) / h ** 2
    grad = float(d @ (move_t(M2 @ b0) - slips.g))
    # the spline means of the weak current inside the label's level sets
    x = np.linspace(0.0, 1.0, 1001)
    s_q, _, _ = label(seq, b0, ANISOTROPY, T)
    I_enc = np.concatenate([[0.0], enclosed_current(seq, b0, np.asarray(s_q), x[1:] ** 2)])
    mu_enc = np.asarray(current_target(lambda s: np.interp(s, x ** 2, I_enc), CELLS))
    weak = float(np.abs(np.asarray(mu) - mu_enc).max() / np.abs(mu_enc).max())
    ev = np.linalg.eigvalsh(D)
    print(f"\n  adjoint {adjoint:.1e}, div {div:.1e}, zeta {zeta:.1e}, first difference {fd:+.10e} vs {grad:+.10e}, "
          f"second {sd:.8e} vs {dDd:.8e}, moments vs weak current {weak:.2e}, block eigenvalues {ev.min():.2e} to "
          f"{ev.max():.2e}")
    assert adjoint < 1e3 * eps()
    assert div < 1e3 * eps()
    assert zeta < 1e3 * eps()
    assert abs(fd - grad) < 1e2 * eps() * E0 / h
    assert abs(sd - dDd) < 1e2 * eps() * E0 / h ** 2
    assert weak < SLIP_WEAK_J
    assert ev.min() > 0.0


def test_current_and_pressure_targets_from_one_namelist(seq, b0, tmp_path):
    """The run of ``scripts/relax.py --current.target input.x --pressure.target input.x`` with ONE VMEC input namelist
    with ``NCURR = 1``: the current the file's (its weak current on the label, as a ``cubic_spline_i`` profile) and
    the pressure the file's, both reshaped by ``1 + 0.05 (1 - 2 s)``. Twenty Newton steps in two chunks, with the label,
    the shells and the slips rebuilt before each, take positive steps, lower ``W`` at every step and end at both
    targets. The current's error (:func:`mrx.relaxation.twist.current_error`) falls from 3.60e-2 to 8.50e-4, against
    3.42e-2 from the file's current at the end (0.025), and the pressure's to 4.68e-3, against 2.63e-2 from the file's
    profile (0.18). The bounds are CURRENT_ERROR and PC_PRESSURE_ERROR, a tenth of the current's error against the
    file's and a quarter of the pressure's, so a run that ignored either target fails. dt is 1 from the first step.
    283 s on four CPU cores after the fixture (2026-11-18). Skipped in float32, where a Newton step costs three times
    as much on the CPU."""
    if DTYPE != np.float64:
        pytest.skip("a float32 Newton step costs three times a float64 one on the CPU, the float64 run covers the claim")
    p_file = seq.equilibrium["profiles"]["pressure"]
    _, T, _ = label(seq, b0, ANISOTROPY)
    s_q, _, _ = label(seq, b0, ANISOTROPY, T)
    edges = np.linspace(0.0, 1.0, 41)
    I_z = np.concatenate([[0.0], enclosed_current(seq, b0, np.asarray(s_q), edges[1:])])
    flux = float(jnp.sum(seq.quad.w * seq.odd.evaluate_at_quadrature(b0, 2)[:, 2]))
    I_phi = I_z * np.sign(flux) * (1.0 + 0.05 * (1.0 - 2.0 * edges))      # PHIEDGE > 0: along phi is along the field
    s = np.linspace(0.0, 1.0, 101)
    row = lambda x: " ".join(f"{v:.17g}" for v in x)  # noqa: E731
    path = tmp_path / "input.target"
    path.write_text(f"&INDATA\n  NFP = {seq.nfp}, PHIEDGE = 1.0, NCURR = 1, PCURR_TYPE = 'cubic_spline_i'\n"
                    f"  CURTOR = {I_phi[-1] / (4e-7 * np.pi):.17g}\n  AC_AUX_S = {row(edges)}\n"
                    f"  AC_AUX_F = {row(I_phi / I_phi[-1])}\n  PMASS_TYPE = 'cubic_spline'\n  AM_AUX_S = {row(s)}\n"
                    f"  AM_AUX_F = {row(p_file(np.sqrt(s)) * (1.0 + 0.05 * (1.0 - 2.0 * s)))}\n/\n")
    mu_t, mu_f = load_current_target(seq, str(path), b0, cells=CELLS), current_on_label(seq, b0, T, CELLS)
    p_t, p_f = load_target(seq, str(path)).profile, load_target(seq, "file").profile

    ts = TimeStepper(seq=seq, newton=True)
    state, dW, dt_star, E0 = initial_state(b0, ts), [], [], None
    for _ in range(STEPS // CHUNK):
        s_q, T, _ = label(seq, state.B_n, ANISOTROPY, T)
        ts = ts.with_shells(build_shells(seq, state.B_n, s_q, p_t(s_q), ts.newton_penalty, ts.newton_mass_solve_tol,
                                         CELLS))
        ts = ts.with_slips(build_slips(seq, state.B_n, T, mu_t)[0])
        res = relax(state, ts, steps=CHUNK, chunk=CHUNK, verbose=False)
        state, E0 = res.state, res.E0 if E0 is None else E0
        dW += list(np.asarray(res.trace["dW"], dtype=float))
        dt_star += list(np.asarray(res.trace["dt_star"], dtype=float))
    s_q, T, _ = label(seq, state.B_n, ANISOTROPY, T)
    mu = current_on_label(seq, state.B_n, T, CELLS)
    err_I, err_I_file, err_I_start = (current_error(mu, mu_t), current_error(mu, mu_f),
                                      current_error(mu_f, mu_t))
    p_w = weak_pressure(current(state.B_n, seq), state.B_n, seq)
    err, _ = pressure_error(seq, p_w, s_q, p_t(s_q), ERROR_BINS)
    err_file, _ = pressure_error(seq, p_w, s_q, p_f(s_q), ERROR_BINS)
    print(f"\n  current {err_I_start:.3e} at the start, {err_I:.3e} at the end, against the file's {err_I_file:.3e}. "
          f"|p - p*| / |p*| {err:.3e}, against the file's profile {err_file:.3e}. max dW / E0 {max(dW) / E0:+.1e}")
    assert min(dt_star) > 0.0
    assert max(dW) < eps() * E0, f"W not monotone: {dW}"
    assert err_I < CURRENT_ERROR and err_I < 0.1 * err_I_file
    assert err < PC_PRESSURE_ERROR and err < 0.25 * err_file



def test_current_error_of_a_zero_target():
    """:func:`mrx.relaxation.twist.current_error` is relative to the target, and absolute for a zero target, the
    closure of no net current on any surface: then it is the root mean square of the moments in T m, weighted by the
    widths of the splines. A target twice the moments gives 1/2, a zero target and constant moments their value, both
    to round-off. Pure numpy, milliseconds."""
    mu = np.full(PROFILE_CELLS + 2, 0.25)
    assert abs(current_error(mu, 2.0 * mu) - 0.5) < 10 * np.finfo(float).eps
    assert abs(current_error(mu, np.zeros_like(mu)) - 0.25) < 10 * np.finfo(float).eps
