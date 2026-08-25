"""Preliminary relaxation study: logical-profile IC, then min-B^2 descent.

Three questions, in this order, and the order matters:

  0. Are the OPERATORS healthy on this commit?  Every "does it converge"
     question downstream is meaningless if they are not, so this script opens
     with identities that hold for the discrete operators alone -- no descent
     method involved, nothing fitted, nothing "small enough".
  1. Does relaxation toward min B^2 work from a logical-profile IC?
  2. What is actually wrong with L-BFGS?

THE EXACT-LINESEARCH IDENTITY, which is the spine of this script
---------------------------------------------------------------
Along the ray ``B + t dB`` with ``dB`` FROZEN, the energy is exactly quadratic:

    E(B + t dB) = E(B) + t <B, dB>_M + t^2/2 ||dB||_M^2

so the minimising t is ``-<B, dB>_M / ||dB||_M^2``.  ``ANALYTIC_LINESEARCH``
instead uses ``dt = (F, u)_M / ||dB||_M^2``.  The two agree iff

    <B, dB>_M = -(F, u)_M                                                  (*)

and (*) is an operator identity, provable in three steps that each name a
different piece of the code:

    <B, curl E>_M2 = (weak_curl B, E)_M1 = (J, u x H)         [curl adjointness]
                   = -(u, J x H)_L2                           [triple product]
                   = -(u, F)_M2                               [Leray M-orthog.,
                                                               since div u = 0]

So checking (*) numerically tests the curl adjoint pair, the sign and argument
order of ``cross_product_load``, and the M-orthogonality of the Leray
projection, all at once, against ZERO.  That is gate G1 below.

TWO CONSEQUENCES, one of which is a trap
----------------------------------------
  * The energy decrease per step is available in closed form,
    ``dE = -(F, u)_M^2 / (2 ||dB||_M^2) = -dt (F, u)_M / 2``.  Comparing the
    MEASURED decrease against it is the same identity, evaluated along the
    trajectory.
  * TRAP: because dt is the exact line minimiser, and t is free to be
    NEGATIVE, the energy decreases monotonically for ANY direction u --
    including an ascent direction, including pure noise.  **Monotone energy
    therefore does not validate a descent method here.**  What separates a
    working optimiser from a broken one is the SIZE of the decrease, i.e.
    ``cos_M(F, u) = (F,u)_M / (|F|_M |u|_M)``: it is 1 for steepest descent and
    goes to 0 for a direction orthogonal to the force.  Every arm reports it.

WHAT THE ARMS ARE
-----------------
  gradient          u = F.
  cg / cg-legacy    Polak-Ribiere with the previous GRADIENT / the previous
                    DIRECTION in the beta formula (``cg_beta``).
  lbfgs-legacy      s = B_{k+1} - B_k, y lagging one step  (the shipped code).
  lbfgs-paired      s = B_{k+1} - B_k, y aligned.          (isolates the lag)
  lbfgs             s = dt u_k, y aligned.                 (isolates the space)

The IC is the logical-profile one, B_hat = (0, Phi'(iota - lam_z),
Phi'(1 + lam_c)), built exactly as ``logical_profile_ic.py`` builds it and
imported from there rather than copied.  The GEOMETRY is loaded from an HDF5
file through ``build_sequence``.

    python scripts/debug/relax_prelim.py --geometry quasr44970 --ns 8,16,8 \
        --arms gradient,cg,lbfgs-legacy,lbfgs --steps 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx.operators as op  # noqa: E402
from mrx.differential_forms import DiscreteFunction  # noqa: E402
from mrx.nullspace import compute_nullspaces  # noqa: E402
from mrx.relaxation import (DescentMethod, IntegrationScheme,  # noqa: E402
                            TimeStepChoice, TimeStepper, compute_force,
                            compute_helicity, initial_state)
from logical_profile_ic import (analytic_helicity, make_lambda,  # noqa: E402
                                make_profiles, parse_lambda)
from verify_block_jacobi import build_sequence  # noqa: E402


ARMS = {
    # name           descent method                   lbfgs_pairing  cg_beta
    "gradient":     (DescentMethod.GRADIENT,           "velocity", "gradient"),
    "cg":           (DescentMethod.CONJUGATE_GRADIENT, "velocity", "gradient"),
    "cg-legacy":    (DescentMethod.CONJUGATE_GRADIENT, "velocity", "legacy"),
    "lbfgs-legacy": (DescentMethod.LBFGS,              "legacy",   "gradient"),
    "lbfgs-paired": (DescentMethod.LBFGS,              "paired",   "gradient"),
    "lbfgs":        (DescentMethod.LBFGS,              "velocity", "gradient"),
}


# ---------------------------------------------------------------------------
# Gate 0: operator identities.  No descent method, no IC quality, no fitting.
# ---------------------------------------------------------------------------

def operator_gates(seq, key):
    """Identities the discrete operators must satisfy on random inputs.

    Each returns a number that is zero in exact arithmetic and is reported
    relative to the size of the terms being cancelled, so "round-off" means
    round-off and not "small compared to something big".
    """
    out = {}
    k_b, k_e, k_w = jax.random.split(key, 3)
    B = jax.random.normal(k_b, (seq.n2_dbc,))
    E = jax.random.normal(k_e, (seq.n1_dbc,))

    # (a) curl adjointness: <B, strong_curl E>_M2 == <weak_curl B, E>_M1.
    #     This is the first equality of (*) and the one the linesearch rests on.
    lhs = float(B @ seq.apply_mass_matrix(seq.apply_strong_curl(E), 2))
    rhs = float(seq.apply_mass_matrix(seq.apply_weak_curl(B), 1) @ E)
    out["curl_adjoint_rel"] = abs(lhs - rhs) / max(abs(lhs), abs(rhs))

    # (b) Leray output is exactly divergence free (strong div of the result).
    v = jax.random.normal(k_w, (seq.n2_dbc,))
    v_l, _ = seq.apply_leray_projection(v, k=2)
    out["leray_div_rel"] = float(
        seq.l2_norm(seq.apply_strong_div(v_l), 3)
        / seq.l2_norm(seq.apply_strong_div(v), 3))

    # (c) Leray is M2-orthogonal: the removed part is M-orthogonal to any
    #     divergence-free field.  This is the third equality of (*).
    w_l, _ = seq.apply_leray_projection(
        jax.random.normal(k_b, (seq.n2_dbc,)) + 0.5 * v, k=2)
    sigma = v_l - v
    num = abs(float(w_l @ seq.apply_mass_matrix(sigma, 2)))
    den = float(seq.l2_norm(w_l, 2) * seq.l2_norm(sigma, 2))
    out["leray_orth_rel"] = num / den

    # (d) strong div o strong curl == 0 exactly (the topological identity that
    #     makes dB = curl E divergence free for free).
    cE = seq.apply_strong_curl(E)
    out["div_curl_rel"] = float(
        seq.l2_norm(seq.apply_strong_div(cE), 3) / seq.l2_norm(cE, 2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="quasr44970")
    ap.add_argument("--ns", default="8,16,8")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--iota", default="0.4,0.9")
    ap.add_argument("--iota-exp", type=float, default=2.0)
    ap.add_argument("--flux-exp", type=float, default=1.0)
    ap.add_argument("--lam", default="")
    ap.add_argument("--arms", default="gradient,cg,cg-legacy,"
                                      "lbfgs-legacy,lbfgs-paired,lbfgs")
    ap.add_argument("--history", type=int, default=1)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--helicity-every", type=int, default=25)
    ap.add_argument("--seconds-per-arm", type=float, default=1800.0)
    ap.add_argument("--gamma", type=int, default=0)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--maxiter", type=int, default=10000)
    ap.add_argument("--gates-only", action="store_true")
    ap.add_argument("--ic-only", action="store_true")
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()

    import mrx
    print(f"[env] mrx from {mrx.__file__}", flush=True)
    ns = tuple(int(v) for v in cli.ns.split(","))
    iota0, iota1 = (float(v) for v in cli.iota.split(","))
    iota, dPhi = make_profiles(iota0, iota1, cli.iota_exp, cli.flux_exp)
    dlam = make_lambda(parse_lambda(cli.lam))

    results = {"geometry": cli.geometry, "ns": list(ns), "p": cli.p,
               "iota": [iota0, iota1], "iota_exp": cli.iota_exp,
               "flux_exp": cli.flux_exp, "lam": cli.lam,
               "history_size": cli.history, "gamma": cli.gamma, "mu": cli.mu,
               "steps": cli.steps}

    t0 = time.perf_counter()
    seq, ops = build_sequence(cli.geometry, ns, cli.p, cli.maxiter)
    ops = seq.assemble_all_sparse(include_preconditioners=False)
    ops = op.assemble_mass_jacobi_preconditioner(seq, ops, ks=(0, 1, 2, 3))
    seq.set_operators(ops)
    ops = op.assemble_block_jacobi_laplacian_preconditioner(
        seq, ops, ks=(0, 1, 2, 3), dirichlets=(False, True))
    seq.set_operators(ops)
    ops = compute_nullspaces(seq, ops)
    seq.set_operators(ops)
    print(f"[setup] {cli.geometry} ns={ns} p={cli.p}  n2_dbc={seq.n2_dbc}  "
          f"operators+nullspaces {time.perf_counter() - t0:.1f}s", flush=True)

    # --- gate 0 -----------------------------------------------------------
    tg = time.perf_counter()
    gates = operator_gates(seq, jax.random.PRNGKey(11))
    results["operator_gates"] = gates
    print(f"\n[G0] operator identities ({time.perf_counter() - tg:.1f}s) -- "
          f"all must be at round-off, independent of any descent method")
    for k, v in gates.items():
        print(f"[G0]   {k:<20s} {v:.3e}")
    if cli.gates_only:
        if cli.out:
            json.dump(results, open(cli.out, "w"), indent=1)
        return

    # --- the IC -----------------------------------------------------------
    # frame='phys': load(frame='ref') wants g omega / J, not omega, and fails
    # silently on omega.  Push forward explicitly instead.
    DF_map = jax.jacfwd(seq.map)

    def omega_ref(x):
        r = x[0]
        f = dPhi(r)
        d_chi, d_zeta = dlam(x)
        return jnp.array([0.0, f * (iota(r) - d_zeta), f * (1.0 + d_chi)])

    def B_phys(x):
        dF = DF_map(x)
        return dF @ omega_ref(x) / jnp.linalg.det(dF)

    t1 = time.perf_counter()
    B_raw = seq.apply_inverse_mass_matrix(
        seq.load(B_phys, 2, dirichlet=True), 2, dirichlet=True)
    B_norm = float(seq.l2_norm(B_raw, 2))
    B0 = B_raw / B_norm
    print(f"\n[ic] logical-profile IC in {time.perf_counter() - t1:.1f}s   "
          f"||B||_M raw = {B_norm:.6e}", flush=True)

    div0 = float(seq.l2_norm(seq.apply_strong_div(B0), 3))
    B_leray, _ = seq.apply_leray_projection(B0, k=2)
    leray0 = float(seq.l2_norm(B_leray - B0, 2))
    # --- is compute_helicity even a FUNCTION of B here? --------------------
    # It solves a k=1 Hodge Laplacian which on a torus (b1 = 1) is SINGULAR,
    # so the answer depends on the deflation and on how far the solve got --
    # and it is warm-started from state.A along a trajectory.  Before any
    # "helicity drifted by X%" statement can mean anything, the SAME B has to
    # give the SAME number from different starting guesses.  Three guesses,
    # one field: the spread is the noise floor of every drift number below.
    H0h, A_conv = compute_helicity(B0, seq, jnp.zeros(seq.n1_dbc))
    H_warm, _ = compute_helicity(B0, seq, A_conv)
    H_half, _ = compute_helicity(B0, seq, 0.5 * A_conv)
    h_spread = (max(map(abs, (float(H0h) - float(H_warm),
                              float(H0h) - float(H_half))))
                / abs(float(H0h)))
    B_harm_rel = float(seq.l2_norm(B0 - seq.apply_strong_curl(A_conv), 2)
                       / seq.l2_norm(B0, 2))
    print(f"[ic] compute_helicity reproducibility: zeros {float(H0h):+.6e}  "
          f"warm {float(H_warm):+.6e}  half {float(H_half):+.6e}   "
          f"relative spread {h_spread:.3e}")
    print(f"[ic]   ||B - curl A||/||B|| (harmonic remainder) = "
          f"{B_harm_rel:.3e}  -- if the spread is not small, every helicity "
          f"drift number in this run is noise and must not be read")
    results.update(H_repro_spread=h_spread, B_harm_rel=B_harm_rel,
                   H_zeros=float(H0h), H_warm=float(H_warm),
                   H_half=float(H_half))

    # --- and is it reproducible because it CONVERGED, or because it fails
    #     the same way every time?  A deterministic non-convergence passes a
    #     spread test cleanly, so ask the solver directly and vary its budget:
    #     a converged solve does not move when the budget is raised.
    rhs_A = seq.apply_weak_curl(B0)
    probe_rows = []
    for tol_A, mi_A in ((1e-08, 10_000), (1e-12, 10_000), (1e-12, 40_000)):
        A_i, info_i = seq.apply_inverse_hodge_laplacian(
            rhs_A, 1, guess=jnp.zeros(seq.n1_dbc), tol=tol_A, maxiter=mi_A,
            return_info=True)
        rem_i = float(seq.l2_norm(B0 - seq.apply_strong_curl(A_i), 2)
                      / seq.l2_norm(B0, 2))
        H_i = float(A_i @ seq.apply_projection_matrix(
            B0 + (B0 - seq.apply_strong_curl(A_i)), 2, 1, True,
            dirichlet_out=True))
        probe_rows.append(dict(tol=tol_A, maxiter=mi_A, info=int(info_i),
                               harm_rel=rem_i, H=H_i,
                               A_norm=float(seq.l2_norm(A_i, 1))))
        print(f"[ic] k=1 Hodge solve  tol={tol_A:.0e} maxiter={mi_A:>6d}  "
              f"info={int(info_i)} (0 = converged)  ||A||_M={probe_rows[-1]['A_norm']:.4e}"
              f"  ||B-curl A||/||B||={rem_i:.4e}  H={H_i:+.6e}", flush=True)
    results["hodge_k1_probe"] = probe_rows

    # --- IS THE RHS EVEN IN THE RIGHT SPACE? -------------------------------
    # apply_inverse_hodge_laplacian solves the saddle form
    #     | S  D | |u|   |f|
    #     | D^T -M| |s| = |0|
    # so f is a DUAL k-form.  compute_helicity feeds it apply_weak_curl(B),
    # which is M1^-1 D1^T B -- a PRIMAL 1-form, one mass-inverse too many.
    # apply_leray_projection, solving the same kind of system, passes
    # apply_derivative_matrix(...) (dual) and not apply_strong_div(...)
    # (primal), so the convention is not in doubt.  If the spurious M1^-1 is
    # the whole story, dropping it should bring the harmonic remainder from
    # 85x||B|| down to at most ||B||, which is the bound a genuine Hodge
    # decomposition obeys.
    rhs_dual = seq.apply_derivative_matrix(
        B0, 1, dirichlet_in=True, dirichlet_out=True, transpose=True)
    A_d, info_d = seq.apply_inverse_hodge_laplacian(
        rhs_dual, 1, guess=jnp.zeros(seq.n1_dbc), return_info=True)
    harm_d = B0 - seq.apply_strong_curl(A_d)
    rem_d = float(seq.l2_norm(harm_d, 2) / seq.l2_norm(B0, 2))
    H_d = float(A_d @ seq.apply_projection_matrix(
        B0 + harm_d, 2, 1, True, dirichlet_out=True))
    print(f"\n[ic] SAME solve, DUAL rhs (D1^T B, no M1^-1):  "
          f"info={int(info_d)}  ||A||_M={float(seq.l2_norm(A_d, 1)):.4e}  "
          f"||B-curl A||/||B||={rem_d:.4e}  H={H_d:+.6e}")
    print(f"[ic]   shipped (primal rhs) gives harmonic remainder "
          f"{B_harm_rel:.4e} and H={float(H0h):+.6e}   "
          f"(eq.(2) natural gauge is printed below)")
    results.update(H_dual_rhs=H_d, B_harm_rel_dual=rem_d,
                   A_norm_dual=float(seq.l2_norm(A_d, 1)))
    H_an = analytic_helicity(iota0, iota1, cli.iota_exp,
                             cli.flux_exp) / B_norm ** 2
    # B^rho leak, per surface (the IC's own structural gate)
    B_h = DiscreteFunction(B0, seq.basis_2, seq.e2_dbc)
    ang = (np.arange(8) + 0.5) / 8
    brho = []
    for r in np.linspace(0.05, 0.95, 19):
        pts = jnp.asarray([[r, c, z] for c in ang for z in ang])
        vals = np.abs(np.asarray(jax.vmap(B_h)(pts))).max(axis=0)
        brho.append(vals[0] / vals[2])
    print(f"[ic] ||div B||_L2 = {div0:.3e}   ||P_Leray B - B|| = {leray0:.3e}"
          f"   max|B^rho|/max|B^zeta| = {max(brho):.3e}")
    print(f"[ic] helicity: code {float(H0h):+.6e}   eq.(2) natural gauge "
          f"{H_an:+.6e}   difference (harmonic gauge) {float(H0h) - H_an:+.3e}")
    E0 = 0.5 * float(seq.l2_norm_sq(B0, 2))
    F0v, _, _, _, _ = compute_force(B0, seq)
    print(f"[ic] E = {E0:.6e}   ||F||_M = {float(seq.l2_norm(F0v, 2)):.4e}",
          flush=True)
    results.update(B_norm_raw=B_norm, div_ic=div0, leray_ic=leray0,
                   Brho_max=float(max(brho)), H_code_ic=float(H0h),
                   H_analytic_ic=H_an, E_ic=E0,
                   F_ic=float(seq.l2_norm(F0v, 2)))

    if cli.ic_only:
        if cli.out:
            json.dump(results, open(cli.out, "w"), indent=1)
        return

    # --- the arms ---------------------------------------------------------
    apply_M2 = jax.jit(lambda v: seq.apply_mass_matrix(v, 2))
    get_helicity = jax.jit(compute_helicity, static_argnames=["seq"])

    results["arms"] = {}
    for name in cli.arms.split(","):
        name = name.strip()
        method, pairing, cgbeta = ARMS[name]
        ts = TimeStepper(
            seq=seq, descent_method=method,
            dt_mode=TimeStepChoice.ANALYTIC_LINESEARCH,
            timestep_mode=IntegrationScheme.EXPLICIT,
            history_size=cli.history, gamma=cli.gamma, mu=cli.mu,
            lbfgs_pairing=pairing, cg_beta=cgbeta)

        @jax.jit
        def step(state, ts=ts):
            state = ts.relaxation_step(state, state.key)
            return eqx.tree_at(lambda s: s.B_n, state, state.B_nplus1)

        @jax.jit
        def probe(state):
            """Everything the trace needs, from the POST-step state.

            F_prev and v are the F and u this step actually used, so the
            linesearch identity is reconstructible without re-solving.
            """
            Fu = state.F_prev @ apply_M2(state.v)
            # ||dB||_M^2 = Fu/dt EXACTLY, by the definition of the linesearch
            # dt -- so the size of the update the direction actually generates
            # is free.  gain = ||dB||_M / ||u||_M is the amplification of
            # C_B: u -> curl(u x H) on the chosen direction.  When it collapses
            # dB is solver noise, dt explodes to compensate, and the quadratic
            # line model stops meaning anything.
            return (0.5 * seq.l2_norm_sq(state.B_n, 2),
                    seq.l2_norm(seq.apply_strong_div(state.B_n), 3),
                    Fu,
                    Fu / (state.F_norm * state.v_norm),
                    state.lbfgs_sy,
                    (Fu / state.dt) ** 0.5 / state.v_norm)

        state = initial_state(B0, ts, dt=1.0)
        F0, p0, _, H0f, JxH0 = compute_force(
            B0, seq, dirichlet_H=ts.dirichlet_H)
        state = eqx.tree_at(
            lambda s: (s.F_norm, s.F_prev, s.p, s.H, s.JxH), state,
            (seq.l2_norm(F0, 2), F0, p0, H0f, JxH0))

        tr = {k: [] for k in ("E", "F", "dt", "div", "Fu", "cos", "sy",
                              "gain", "dE_meas", "dE_pred", "helicity",
                              "hel_it")}
        E_prev = E0
        t_arm = time.perf_counter()
        n_done = 0
        print(f"\n=== arm {name}  (method={method.name} pairing={pairing} "
              f"cg_beta={cgbeta} m={cli.history}) ===", flush=True)
        for it in range(1, cli.steps + 1):
            state = step(state)
            E, div, Fu, cos, sy, gain = (float(v) for v in probe(state))
            dE_meas = E - E_prev
            dE_pred = -0.5 * float(state.dt) * Fu
            tr["E"].append(E)
            tr["F"].append(float(state.F_norm))
            tr["dt"].append(float(state.dt))
            tr["div"].append(div)
            tr["Fu"].append(Fu)
            tr["cos"].append(cos)
            tr["sy"].append(sy)
            tr["gain"].append(gain)
            tr["dE_meas"].append(dE_meas)
            tr["dE_pred"].append(dE_pred)
            E_prev = E
            n_done = it
            if it % cli.helicity_every == 0 or it == 1:
                h, A_new = get_helicity(state.B_n, seq, state.A)
                state = eqx.tree_at(lambda s: s.A, state, A_new)
                tr["helicity"].append(float(h))
                tr["hel_it"].append(it)
            if it <= 5 or it % 20 == 0:
                print(f"  it {it:>5d}  E={E:.8e}  |F|={state.F_norm:.4e}  "
                      f"dt={float(state.dt):+.3e}  cos={cos:+.4f}  "
                      f"sy={sy:+.3e}  gain={gain:.2e}  divB={div:.2e}  "
                      f"dE_meas={dE_meas:+.3e}  dE_pred={dE_pred:+.3e}",
                      flush=True)
            if time.perf_counter() - t_arm > cli.seconds_per_arm:
                print(f"  [budget] stopping arm at it={it} after "
                      f"{time.perf_counter() - t_arm:.0f}s", flush=True)
                break

        wall = time.perf_counter() - t_arm
        dEm = np.array(tr["dE_meas"])
        dEp = np.array(tr["dE_pred"])
        ident = np.abs(dEm - dEp) / np.abs(dEp)
        # A direction that generates almost no dB makes dE_pred ~ 0, and the
        # RELATIVE identity then divides by nothing -- which is a statement
        # about the direction, not about the operators.  The same discrepancy
        # measured against the ENERGY SCALE stays meaningful there, so both
        # are reported and the absolute one is the operator test.
        ident_abs = np.abs(dEm - dEp) / E0
        n_up = int((dEm > 0).sum())
        print(f"--- arm {name}: {n_done} steps in {wall:.1f}s "
              f"({wall / max(n_done, 1):.2f} s/step)")
        print(f"    E {E0:.8e} -> {tr['E'][-1]:.8e}   "
              f"({(E0 - tr['E'][-1]) / E0:.4%} of the initial energy removed)")
        print(f"    |F| {results['F_ic']:.4e} -> {tr['F'][-1]:.4e}")
        print(f"    G1 linesearch identity |dE_meas - dE_pred|/|dE_pred|: "
              f"median {np.median(ident):.3e}  max {ident.max():.3e}")
        print(f"    G1 same, against the energy scale /E0: "
              f"median {np.median(ident_abs):.3e}  max {ident_abs.max():.3e}")
        print(f"    gain ||dB||_M/||u||_M: median {np.median(tr['gain']):.3e}"
              f"  min {min(tr['gain']):.3e}  max {max(tr['gain']):.3e}")
        print(f"    energy INCREASES on {n_up}/{n_done} steps")
        print(f"    cos_M(F,u): median {np.median(tr['cos']):+.4f}  "
              f"min {min(tr['cos']):+.4f}  max {max(tr['cos']):+.4f}")
        print(f"    dt < 0 on {int((np.array(tr['dt']) < 0).sum())}/{n_done} "
              f"steps;  s.My < 0 on "
              f"{int((np.array(tr['sy']) < 0).sum())}/{n_done} steps")
        print(f"    ||div B|| max {max(tr['div']):.3e}")
        if tr["helicity"]:
            h = np.array(tr["helicity"])
            print(f"    helicity {h[0]:+.6e} -> {h[-1]:+.6e}  "
                  f"relative drift {(h[-1] - h[0]) / abs(h[0]):+.3e}",
                  flush=True)
        results["arms"][name] = dict(
            steps=n_done, wall=wall, trace=tr,
            E_final=tr["E"][-1], F_final=tr["F"][-1],
            identity_median=float(np.median(ident)),
            identity_max=float(ident.max()),
            identity_abs_median=float(np.median(ident_abs)),
            identity_abs_max=float(ident_abs.max()),
            gain_median=float(np.median(tr["gain"])), n_energy_up=n_up,
            cos_median=float(np.median(tr["cos"])),
            n_dt_negative=int((np.array(tr["dt"]) < 0).sum()),
            n_sy_negative=int((np.array(tr["sy"]) < 0).sum()),
            div_max=float(max(tr["div"])))
        if cli.out:
            json.dump(results, open(cli.out, "w"), indent=1)

    if cli.out:
        json.dump(results, open(cli.out, "w"), indent=1)
        print(f"\nwrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
