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
  cg                Polak-Ribiere, previous GRADIENT in the beta formula.
  lbfgs             two-loop recursion, s = dt u_k and y aligned with it.

The three broken variants that produced the L-BFGS diagnosis (B-increment s;
y lagging one step; the previous DIRECTION in CG's beta) were deliberately
DELETED once they had done their job rather than left behind as knobs -- see
the handoff.  Commit ecfa3ef is the one that still carries them if the
factorial ever needs re-running.

The IC is the logical-profile one, B_hat = (0, Phi'(iota - lam_z),
Phi'(1 + lam_c)), built exactly as ``logical_profile_ic.py`` builds it and
imported from there rather than copied.  The GEOMETRY is loaded from an HDF5
file through ``build_sequence``.

    python scripts/debug/relax_prelim.py --geometry quasr44970 --ns 8,16,8 \
        --arms gradient,cg,lbfgs --steps 250
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx.operators as op  # noqa: E402
from mrx.differential_forms import DiscreteFunction, Pushforward  # noqa: E402, E501
from mrx.nullspace import compute_nullspaces  # noqa: E402
from mrx.relaxation import (DescentMethod, IntegrationScheme,  # noqa: E402
                            TimeStepChoice, TimeStepper, compute_force,
                            compute_helicity, initial_state)
from logical_profile_ic import (analytic_helicity, make_lambda,  # noqa: E402
                                make_profiles, parse_lambda)
from gvec_clebsch_ic import load_clebsch  # noqa: E402
from gvec_geometry import GVEC_GEOMETRIES  # noqa: E402
from verify_block_jacobi import build_sequence  # noqa: E402


ARMS = {
    "gradient": DescentMethod.GRADIENT,
    "cg": DescentMethod.CONJUGATE_GRADIENT,
    "lbfgs": DescentMethod.LBFGS,
}


# ---------------------------------------------------------------------------
# Gate 0: operator identities.  No descent method, no IC quality, no fitting.
# ---------------------------------------------------------------------------

def geometry_nfp(geometry):
    """Field periods spanned by logical zeta in [0, 1], read not assumed."""
    if geometry in GVEC_GEOMETRIES:
        with h5py.File(GVEC_GEOMETRIES[geometry], "r") as h:
            return int(h.attrs["nfp"])
    return {"toroid": 1, "cylinder": 1, "rot-ellipse": 3, "w7x": 5}[geometry]


def render_poincare(seq, B_dof, nfp, tag, outdir, cli):
    """Poincare section of one field.  Returns the summary dict, or raises.

    Deliberately run AFTER every arm has finished and its traces are on disk:
    ``require_zeta_parameterisation`` RAISES when B^zeta changes sign, which is
    a real finding about a relaxed field and not something to swallow, but it
    must not be able to destroy the run it is reporting on.
    """
    from types import SimpleNamespace  # noqa: PLC0415

    import poincare_vacuum as pv  # noqa: PLC0415
    from mrx.poincare import logical_field, require_zeta_parameterisation  # noqa: PLC0415, E501

    field = logical_field(seq, B_dof, 2, True)
    info = require_zeta_parameterisation(field, name=tag)
    print(f"[poincare] {tag}: B^zeta/|B| in "
          f"[{info['bz_over_b_min']:+.3e}, {info['bz_over_b_max']:+.3e}]",
          flush=True)

    # steps_per_period must be a multiple of saves_per_period, so that every
    # save time is a step endpoint and no dense interpolation enters the
    # section (mrx.poincare.trace's own requirement).
    if cli.pc_steps % cli.pc_saves:
        raise ValueError(f"--pc-steps {cli.pc_steps} must be a multiple of "
                         f"--pc-saves {cli.pc_saves}")
    pc = SimpleNamespace(
        seeds=cli.pc_seeds, saves=cli.pc_saves, steps=cli.pc_steps,
        periods=cli.pc_periods, r_min=0.03, r_max=0.97, seed_from="axis",
        batch_size=None, drift_periods=min(cli.pc_periods, 32))

    t0 = time.perf_counter()
    res = pv.run_field(seq, B_dof, 2, True, nfp, pc)
    res["saves_per_period"] = pc.saves
    print(f"[poincare] {tag}: traced {cli.pc_seeds} seeds x "
          f"{cli.pc_periods} periods in {time.perf_counter() - t0:.0f}s   "
          f"iota {np.nanmin(res['iota']):+.4f} .. "
          f"{np.nanmax(res['iota']):+.4f}   drift {res['drift']:.2e}",
          flush=True)

    # EVERY PLANE COMES FROM THE SAME TRACE.  section_RZ just picks the save
    # offset round(plane*saves), so additional zeta planes cost a plot each
    # and no integration at all -- which is why --pc-saves is worth raising.
    planes = [float(v) for v in str(cli.pc_zeta).split(",")]
    out = {"iota_min": float(np.nanmin(res["iota"])),
           "iota_max": float(np.nanmax(res["iota"])),
           "escaped": int(res["escaped"].sum()), "drift": float(res["drift"]),
           "crossings_per_line": int(res["ys"].shape[1] // pc.saves),
           "bz_min": info["bz_over_b_min"], "bz_max": info["bz_over_b_max"],
           "planes": {}}
    for plane in planes:
        k = round(plane * pc.saves)
        if abs(k - plane * pc.saves) > 1e-9:
            raise ValueError(
                f"zeta plane {plane} is not a save time: plane*saves must be "
                f"an integer, got {plane * pc.saves} with saves={pc.saves}")
        RZ = pv.section_RZ(seq, res, plane)
        a_eff = np.asarray(pv.effective_radius(
            RZ[0], RZ[1], RZ[2].mean(), RZ[3].mean()))
        suffix = "" if len(planes) == 1 else f"_z{plane:g}".replace(".", "p")
        path = os.path.join(outdir, f"poincare_{tag}{suffix}.png")
        offset = pv.plot(res, cli.geometry, tag, plane, nfp, RZ, a_eff,
                         "a_eff [m]", path)
        out["planes"][f"{plane:g}"] = {"path": path, "axis_offset": offset}
        print(f"[poincare] {tag}: zeta={plane:g} -> {path}   "
              f"axis offset {offset:.3e}", flush=True)
    return out


def make_pressure_profiler(seq, rhos, n_ang=8):
    """Surface-averaged pressure profile from the Leray multiplier.

    The scheme minimises magnetic energy under an INCOMPRESSIBLE flow, so the
    pressure is the Lagrange multiplier enforcing div v = 0 and the part of
    J x B the Leray projection removes IS grad p.  The fixed point is therefore
    J x B = grad p -- a finite-beta equilibrium, not a force-free state -- and
    ``compute_force`` has been returning that multiplier all along.

    Which makes this a real convergence test rather than a diagnostic: on a
    geometry whose file carries GVEC's own ``pressure``, the multiplier along
    the trajectory should stay CONSISTENT with it instead of wandering off.
    Compared as a SHAPE with one fitted scale, since B is normalised to
    ||B||_M = 1 and the multiplier inherits that arbitrary scale, and with the
    edge offset removed since p's gauge is an additive constant.
    """
    ang = (jnp.arange(n_ang) + 0.5) / n_ang
    pts = jnp.asarray([[[r, c, z] for c in ang for z in ang] for r in rhos])

    # Volume weight per surface, V'(rho) = <J>.  Pure geometry, so computed
    # once: it is what turns a surface-grid mean into a volume average.
    DF = jax.jacfwd(seq.map)
    Vp = jax.vmap(lambda P: jnp.mean(
        jax.vmap(lambda x: jnp.linalg.det(DF(x)))(P)))(pts)

    def profile(p_dof, B_dof=None):
        p_h = Pushforward(
            DiscreteFunction(p_dof, seq.basis_3, seq.e3_dbc), seq.map, 3)
        p_prof = jax.vmap(lambda P: jnp.mean(jax.vmap(p_h)(P)[:, 0]))(pts)
        if B_dof is None:
            return p_prof
        B_phys = Pushforward(
            DiscreteFunction(B_dof, seq.basis_2, seq.e2_dbc), seq.map, 2)

        def bsq_at(x):
            v = B_phys(x)
            return v @ v

        bsq_prof = jax.vmap(lambda P: jnp.mean(jax.vmap(bsq_at)(P)))(pts)
        return p_prof, bsq_prof, Vp

    return profile


def beta_from_profiles(p_prof, bsq_prof, Vp):
    """``beta(rho) = 2 (p - p_edge) / <|B|^2>``, and its volume average.

    ``mu0 = 1`` in these units -- the force operator uses ``J = curl B`` with
    no ``mu0`` -- so beta is just ``2 p / B^2``.

    THE GAUGE MATTERS.  The Leray multiplier is defined only up to an additive
    constant, so a raw ``2p/B^2`` is not a physical beta and would change if
    the solver returned a different constant.  The physical convention is
    ``p = 0`` at the edge, so the outermost sampled surface is subtracted
    first.  This is the same gauge fix the pressure-shape comparison makes.

    Beta is otherwise scale-invariant: p is quadratic in B (it comes from
    J x B), so normalising ``||B||_M = 1`` does not change the ratio.
    """
    p0 = p_prof - p_prof[-1]
    beta = 2.0 * p0 / bsq_prof
    beta_vol = float(np.sum(Vp * beta) / np.sum(Vp))
    return beta, beta_vol


def pressure_shape_residual(p_ours, p_file):
    """One fitted scale, edge offset removed; returns (scale, residual)."""
    a_ours = p_ours - p_ours[-1]
    a_file = p_file - p_file[-1]
    denom = float(a_ours @ a_ours)
    if denom == 0.0:
        return 0.0, float("nan")
    k = float(a_ours @ a_file / denom)
    return k, float(np.linalg.norm(a_file - k * a_ours)
                    / np.linalg.norm(a_file))


def harmonic_alignment(seq, ops, B_dof):
    """How close ``B`` is to the harmonic 2-form, in the M2 inner product.

    Returns ``(cos, rel_err)``: the alignment ``|<B,h>_M| / (|B|_M |h|_M)``
    and the relative M-norm of what is left after projecting B onto span(h).
    A field that has relaxed to the harmonic one has cos -> 1 and
    rel_err -> 0.  The harmonic vector comes from compute_nullspaces, i.e.
    from a completely different solve chain than the relaxation, so this is a
    genuine cross-check and not an internal consistency relation.
    """
    from mrx.nullspace import get_nullspace  # noqa: PLC0415

    h = get_nullspace(ops, 2, True)
    if h is None or np.asarray(h).size == 0:
        return None
    h = jnp.asarray(h)[0]
    Mh = seq.apply_mass_matrix(h, 2)
    hh = float(h @ Mh)
    bh = float(B_dof @ Mh)
    bb = float(seq.l2_norm_sq(B_dof, 2))
    cos = abs(bh) / (bb * hh) ** 0.5
    resid = B_dof - (bh / hh) * h
    # The harmonic AMPLITUDE is an EXACT invariant of this scheme, and it is
    # why the field cannot relax to B = 0.  The update is dB = curl E, i.e. an
    # exact form, and a harmonic 2-form of the Dirichlet complex satisfies
    # D_1^T h = 0; since D_1 = M_2 G_1,
    #     <h, dB>_M2 = h^T M_2 G_1 E = h^T D_1 E = (D_1^T h)^T E = 0
    # identically.  That component IS the net toroidal flux: B carries a
    # frozen flux it cannot shed, and by the M-orthogonal Hodge split
    # ||B||^2 = ||B_harm||^2 + ||B_exact||^2 the descent can only remove the
    # exact part.  Returned so the run MEASURES the invariant rather than
    # asserting it.
    return cos, float(seq.l2_norm(resid, 2) / bb ** 0.5), bh / hh


def make_force_normaliser(seq):
    """The magnetic-pressure-gradient scale the force residual is measured against.

    WHY NOT grad p.  NOT because the pressure is absent -- it is not.  The
    velocity is Leray-projected, so incompressibility is a CONSTRAINT and the
    pressure is its Lagrange multiplier; the part of J x B the projection
    removes IS grad p, and this scheme converges to J x B = grad p, a genuine
    finite-beta equilibrium.  ``compute_force`` already returns that
    multiplier.  grad p is therefore a real and gauge-independent scale (p
    itself is defined only up to a constant, its gradient is not).

    The reason to prefer ``grad(B^2/2)`` is CONDITIONING across cases.  grad p
    goes to zero in the low-beta / near-force-free limit, which is where the
    logical-profile arms sit, so a residual normalised by it blows up exactly
    when the run is doing well.  ``grad(B^2/2)`` has the same units, is built
    from the field itself, and stays O(1) in every case here -- finite-beta
    W7-X and near-force-free alike.

    HOW.  Not by autodiffing a DiscreteFunction at every quadrature point --
    that is one basis evaluation plus one reverse-mode pass per point, ~2e5
    points, and it dominated everything else in the step.  Instead go through
    the sequence, which already has fast tensor-product machinery for exactly
    this shape of object (it is what ``cross_product_load`` does for J x B):

        1. project B^2/2 onto 0-forms   -- one load + one M_0 solve
        2. take the DISCRETE gradient   -- strong_grad = M_1^-1 D_0, a 1-form
        3. measure it

    Step 1 is the operator we do not otherwise have.  The integrand: for a
    2-form, ``B_phys = DF B_hat / J``, so ``|B_phys|^2 = B_hat^T g B_hat / J^2``,
    and the 0-form dual pairing carries a J from the volume element, leaving
    ``q_i = int Lambda0_i (B_hat^T g B_hat) / (2 J)``.

    Returns both measures, because they are not the same number:
      * ``||grad(B^2/2)||_{L2}`` -- pairs with ``||F||_{L2}``, so their ratio is
        dimensionless with the volume factored out of both. This is the one
        the residual uses.
      * ``<|grad(B^2/2)|>_vol``  -- the literal volume average of the
        magnitude, ``sum w J |v| / sum w J`` with ``|v|^2 = g1^T g^-1 g1``.
    """
    from mrx.quadrature import evaluate_at_xq, integrate_against  # noqa: PLC0415, E501

    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    ci0, cs0 = seq._form_comp_info(0)
    ci1, cs1 = seq._form_comp_info(1)
    ci2, cs2 = seq._form_comp_info(2)
    wJ = seq.quad.w * seq.jacobian_j
    vol = jnp.sum(wJ)

    def normaliser(B_dof):
        B_jk = evaluate_at_xq(seq.e2_dbc_T @ B_dof, ci2, cs2, quad_shape, 3)
        bsq = jnp.einsum('qi,qij,qj->q', B_jk, seq.metric_jkl, B_jk)
        f_jk = (0.5 * bsq * seq.quad.w / seq.jacobian_j)[:, None]
        q = seq.e0 @ integrate_against(f_jk, ci0, cs0, quad_shape)

        w0 = seq.apply_inverse_mass_matrix(q, 0, dirichlet=False)
        g1 = seq.apply_strong_grad(w0, dirichlet_in=False, dirichlet_out=False)

        l2 = seq.l2_norm(g1, 1, dirichlet=False)
        g1_jk = evaluate_at_xq(seq.e1_T @ g1, ci1, cs1, quad_shape, 3)
        mag = jnp.sqrt(jnp.einsum('qi,qij,qj->q', g1_jk,
                                  seq.metric_inv_jkl, g1_jk))
        return l2, jnp.sum(wJ * mag) / vol

    return normaliser


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

    # (d) d.d == 0, by BOTH routes, because they are not the same operator and
    #     only one of them is topological.
    #
    #   strong_curl = M_2^-1 D_1 -- a Krylov solve, so d.d inherits the solver
    #     tolerance rather than vanishing.
    #   apply_incidence_matrix = Gram^-1 (E^T sp E) -- the TRUE strong
    #     derivative on extracted DoFs, with the Gram correction localised to
    #     the polar axis.  Matrix-free, and d.d = 0 identically.
    #
    # DeRhamSequence.apply_incidence_matrix's own docstring still claims the
    # opposite (that the mass-projected form "should be preferred when exact
    # d.d = 0 on extracted DoFs is required"); mrx/operators.py:2039, which is
    # what actually runs, says the correction makes d.d exact.  These gates
    # settle which is true here rather than trusting either docstring.
    cE = seq.apply_strong_curl(E)
    out["div_curl_rel_massproj"] = float(
        seq.l2_norm(seq.apply_strong_div(cE), 3) / seq.l2_norm(cE, 2))

    gE = seq.apply_incidence_matrix(E, 1, dirichlet_in=True,
                                    dirichlet_out=True)
    dgE = seq.apply_incidence_matrix(gE, 2, dirichlet_in=True,
                                     dirichlet_out=True)
    out["div_curl_rel_incidence"] = float(
        seq.l2_norm(dgE, 3) / seq.l2_norm(gE, 2))

    # and how far apart are the two curls?  If this is at round-off the swap
    # is free; if not, it changes the trajectory and has to be justified.
    out["curl_massproj_vs_incidence"] = float(
        seq.l2_norm(cE - gE, 2) / seq.l2_norm(cE, 2))
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
    ap.add_argument("--arms", default="gradient,cg,lbfgs")
    ap.add_argument("--history", type=int, default=1)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--helicity-every", type=int, default=25)
    ap.add_argument("--seconds-per-arm", type=float, default=1800.0)
    ap.add_argument("--gamma", type=int, default=0)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--maxiter", type=int, default=10000)
    ap.add_argument("--ic", default="logical",
                    choices=("logical", "clebsch", "dzeta"),
                    help="'logical' builds the IC from prescribed profiles; "
                         "'clebsch' rebuilds GVEC's own equilibrium from the "
                         "clebsch/* ingredients in the geometry file; "
                         "'dzeta' is the bare logical 2-form (0,0,1), whose "
                         "relaxation has an EXACTLY known target -- see "
                         "harmonic_alignment.")
    ap.add_argument("--no-lambda", action="store_true",
                    help="clebsch only: zero lambda out. The fluxes, iota and "
                         "the helicity must NOT move; the force must.")
    ap.add_argument("--poincare", action="store_true",
                    help="render a Poincare section of the IC and of each "
                         "arm's final field, AFTER all arms have finished")
    ap.add_argument("--pc-seeds", type=int, default=40)
    ap.add_argument("--pc-periods", type=int, default=150)
    ap.add_argument("--pc-zeta", default="0.0",
                    help="comma-separated zeta planes, e.g. '0,0.25,0.5'. "
                         "Each must be a save time (plane*--pc-saves integral). "
                         "Extra planes are nearly free: they reuse one trace.")
    ap.add_argument("--pc-saves", type=int, default=8,
                    help="samples kept per period. Raise it to get more zeta "
                         "planes; must exceed twice the poloidal turns per "
                         "period or the angle unwrapping aliases.")
    ap.add_argument("--pc-steps", type=int, default=24,
                    help="integration steps per period; must be a multiple "
                         "of --pc-saves.")
    ap.add_argument("--save-b", default=None,
                    help="HDF5 path for the IC and per-arm final B DoFs")
    ap.add_argument("--dt-mode", default="linesearch",
                    choices=("linesearch", "fixed"),
                    help="'linesearch' takes the exact energy-minimising step, "
                         "which is the LARGEST step that still reduces E. "
                         "Frozen-in flux is only preserved to O(dt^2) by "
                         "explicit Euler, so a big step can destroy field-line "
                         "topology while div B and energy monotonicity stay "
                         "exact -- nested surfaces are protected by NO "
                         "invariant this scheme enforces. 'fixed' with a small "
                         "--dt0 is the control that tests exactly that.")
    ap.add_argument("--dt0", type=float, default=1.0,
                    help="the fixed step, when --dt-mode fixed")
    ap.add_argument("--eta-max", type=float, default=0.0,
                    help="peak resistivity. eta > 0 RELAXES the topological "
                         "constraint: helicity is no longer conserved, the "
                         "field can reconnect, and it can reach states the "
                         "ideal flow cannot. Expect helicity to fall -- that "
                         "is the mechanism, not a failure.")
    ap.add_argument("--eta-schedule", default="tanh",
                    choices=("tanh", "constant", "linear"),
                    help="tanh drops eta_max -> ~0 over the middle third, so "
                         "the run ends ideal and the final state is a genuine "
                         "ideal equilibrium rather than a resistively "
                         "supported one.")
    ap.add_argument("--cold-start", action="store_true",
                    help="zero the solver warm-start slots (p, p_v, H, JxH, E) "
                         "between steps, so every Krylov solve starts from "
                         "zero. Measures what the warm starts are worth: the "
                         "guesses only set the starting vector, so the "
                         "converged answers -- and hence the trajectory -- "
                         "are the same to solver tolerance and ONLY the cost "
                         "differs.")
    ap.add_argument("--no-leray-ic", action="store_true",
                    help="skip the Leray clean-up of the initial condition")
    ap.add_argument("--beta-from", default=None,
                    help="path to a B.h5 written by --save-b. Reports beta, "
                         "the force residual and the harmonic split for EVERY "
                         "field in it and exits -- so the relaxed state is "
                         "measured, not just the IC. No relaxation, no "
                         "tracing; the sequence build is the only cost.")
    ap.add_argument("--poincare-from", default=None,
                    help="path to a B.h5 written by --save-b. Renders a "
                         "Poincare section of EVERY field in it and exits, "
                         "with no relaxation: the geometry build is ~5 min "
                         "against hours to re-relax, so presentation can be "
                         "changed without paying for the run again. Same "
                         "reasoning as scripts/debug/poincare_replot.py.")
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
    # Renamed on greville-prod (block_jacobi -> metric_lumping) and picked up
    # in the 2026-08-26 merge; same signature, same object.  The rename is
    # followed here rather than aliased -- the campaign's numbers were all
    # produced by this call under its old name, and a shim would hide that the
    # production atom has been renamed twice in a month.
    ops = op.assemble_metric_lumping_laplacian_preconditioner(
        seq, ops, ks=(0, 1, 2, 3), dirichlets=(False, True))
    seq.set_operators(ops)
    ops = compute_nullspaces(seq, ops)
    seq.set_operators(ops)
    print(f"[setup] {cli.geometry} ns={ns} p={cli.p}  n2_dbc={seq.n2_dbc}  "
          f"operators+nullspaces {time.perf_counter() - t0:.1f}s", flush=True)

    # --- beta (and friends) from saved fields, then stop -------------------
    if cli.beta_from:
        with h5py.File(cli.beta_from, "r") as f:
            fields = {k: jnp.asarray(f[k][:]) for k in f.keys()}
            attrs = {k: f.attrs[k] for k in f.attrs}
        saved = (str(attrs.get("geometry")), list(attrs.get("ns", [])),
                 int(attrs.get("p", -1)))
        asked = (cli.geometry, list(ns), cli.p)
        if saved != asked:
            raise ValueError(
                f"saved field is {saved} but this run built {asked}; "
                f"re-measure with matching --geometry/--ns/--p")
        print(f"[beta] {cli.beta_from}: {list(fields)}   attrs {attrs}",
              flush=True)

        p_rhos = np.linspace(0.05, 0.95, 19)
        prof = jax.jit(make_pressure_profiler(seq, jnp.asarray(p_rhos)))
        out = {}
        for tag, dof in fields.items():
            Fv, pv, _, _, _ = compute_force(dof, seq)
            pp, bb, vv = prof(pv, dof)
            beta_prof, beta_vol = beta_from_profiles(
                np.asarray(pp), np.asarray(bb), np.asarray(vv))
            # Hodge split: ||B||^2 = ||B_harm||^2 + ||curl A||^2, so the
            # current-driven fraction is what is left over from the harmonic
            # part.  That is the quantity that should track beta.
            _, A_c = compute_helicity(dof, seq, jnp.zeros(seq.n1_dbc))
            harm = float(seq.l2_norm(dof - seq.apply_incidence_matrix(
                A_c, 1, dirichlet_in=True, dirichlet_out=True), 2)
                / seq.l2_norm(dof, 2))
            cur = float(np.sqrt(max(0.0, 1.0 - harm ** 2)))
            fnorm = float(seq.l2_norm(Fv, 2))
            out[tag] = dict(beta_vol=beta_vol,
                            beta_max=float(np.max(np.abs(beta_prof))),
                            beta_axis=float(beta_prof[0]),
                            harmonic=harm, current_driven=cur, F=fnorm,
                            E=0.5 * float(seq.l2_norm_sq(dof, 2)))
            print(f"[beta] {tag:16s} beta_vol={beta_vol:+.4e}  "
                  f"beta_max={out[tag]['beta_max']:.4e}  "
                  f"beta_axis={out[tag]['beta_axis']:+.4e}  "
                  f"harmonic={harm:.6f}  current-driven={100 * cur:.2f}%  "
                  f"||F||={fnorm:.4e}", flush=True)
        if cli.out:
            json.dump(out, open(cli.out, "w"), indent=1)
        return

    # --- render from a saved field, then stop ------------------------------
    if cli.poincare_from:
        nfp = geometry_nfp(cli.geometry)
        outdir = os.path.dirname(cli.out) if cli.out else os.path.dirname(
            cli.poincare_from)
        with h5py.File(cli.poincare_from, "r") as f:
            fields = {k: jnp.asarray(f[k][:]) for k in f.keys()}
            attrs = {k: f.attrs[k] for k in f.attrs}
        print(f"[replot] {cli.poincare_from}: {list(fields)}   attrs {attrs}",
              flush=True)
        # The DoF vectors only mean anything against the sequence they were
        # built on.  A mismatch would fail on shape anyway, but say so here
        # rather than in a einsum traceback.
        saved = (str(attrs.get("geometry")), list(attrs.get("ns", [])),
                 int(attrs.get("p", -1)))
        asked = (cli.geometry, list(ns), cli.p)
        if saved != asked:
            raise ValueError(
                f"saved field is {saved} but this run built {asked}; "
                f"re-render with matching --geometry/--ns/--p")
        out = {}
        for tag, dof in fields.items():
            out[tag] = render_poincare(seq, dof, nfp, tag, outdir, cli)
            if cli.out:
                json.dump(out, open(cli.out, "w"), indent=1)
        return

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

    if cli.ic == "clebsch":
        # GVEC's own ingredients.  Its reference 2-form components ARE
        # sqrt(g) B^i, so the field is rebuilt from three scalars rather than
        # resampled as a vector: div B = 0 and B^rho = 0 then hold by
        # construction and not by fit quality.  Units, measured against these
        # files' own stored derivatives rather than assumed: the derivatives
        # are w.r.t. RADIAN angles, so Phi' = 2 pi dPhi_dr, iota = (1/nfp)
        # dchi_dr/dPhi_dr and lambda = LA / 2 pi.
        cb = load_clebsch(GVEC_GEOMETRIES[cli.geometry], seq)
        clebsch_data = cb
        rho_g = jnp.asarray(cb["rho"])
        dPhi_g = jnp.asarray(cb["dPhi"])
        dchi_g = jnp.asarray(cb["dchi"])
        grad_lam = jax.grad(cb["lam_h"])
        nfp_c = cb["nfp"]
        two_pi = 2.0 * jnp.pi
        use_lam = 0.0 if cli.no_lambda else 1.0
        print(f"[ic] clebsch from {GVEC_GEOMETRIES[cli.geometry]}  "
              f"nfp={nfp_c}  lambda={'OFF' if cli.no_lambda else 'on'}")
        print(f"[ic]   iota = dchi/dPhi (full turn) "
              f"{cb['dchi'][1] / cb['dPhi'][1]:+.5f} (axis) -> "
              f"{cb['dchi'][-1] / cb['dPhi'][-1]:+.5f} (edge);  per MRX field "
              f"period divide by nfp={nfp_c}")
        print(f"[ic]   max angular departure of dchi/dPhi from a flux "
              f"function, mid-radius: {cb['iota_spread']:.3e}")

        def omega_ref(x):
            r = jnp.clip(x[0], rho_g[0], rho_g[-1])
            f_phi = jnp.interp(r, rho_g, dPhi_g)
            f_chi = jnp.interp(r, rho_g, dchi_g) / nfp_c
            g = grad_lam(jnp.array([r, x[1] % 1.0, x[2] % 1.0])) / two_pi
            lam_t, lam_z = g[1] * use_lam, g[2] * use_lam
            return jnp.array([0.0,
                              f_chi - f_phi * lam_z,
                              f_phi * (1.0 + lam_t)])
    elif cli.ic == "dzeta":
        clebsch_data = None
        # B_hat = dzeta, i.e. the constant reference 2-form (0, 0, 1).
        #
        # This IC has an EXACTLY KNOWN relaxation target, which nothing else
        # in this study does.  B^rho = 0 and the components are constant, so
        # div B = 0 identically and iota = B^chi/B^zeta = 0 -- zero shear,
        # hence zero helicity by eq.(2).  Minimising B^2 at fixed toroidal
        # flux with zero helicity lands on the HARMONIC field: the unique
        # (b_2^rel = 1) harmonic 2-form of the Dirichlet complex, which
        # compute_nullspaces has already produced.  So the test is not "did
        # the residual fall" but "did it converge to that vector".
        # Doubly falsifiable: the harmonic field is curl-free, so J = 0 and
        # the pressure spread must go to zero too.
        def omega_ref(x):
            return jnp.array([0.0, 0.0, 1.0])
    else:
        clebsch_data = None

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
    print(f"\n[ic] {cli.ic} IC in {time.perf_counter() - t1:.1f}s   "
          f"||B||_M raw = {B_norm:.6e}", flush=True)

    div0 = float(seq.l2_norm(seq.apply_incidence_matrix(
        B0, 2, dirichlet_in=True, dirichlet_out=True), 3))
    B_leray, _ = seq.apply_leray_projection(B0, k=2)
    leray0 = float(seq.l2_norm(B_leray - B0, 2))
    if not cli.no_leray_ic:
        # The evolution is dB = curl E, which preserves div B EXACTLY, so
        # whatever divergence the IC carries it carries for the whole run.
        # The Clebsch IC is div-free by construction in the reference frame
        # but the L2 projection through M_2 reintroduces a component
        # (measured 2.7e-02 at ns=(8,16,8) on w7x-ini-clebsch, against
        # 3.7e-04 for the logical IC), so clean it once, up front.
        B0 = B_leray / float(seq.l2_norm(B_leray, 2))
        div_after = float(seq.l2_norm(seq.apply_incidence_matrix(
            B0, 2, dirichlet_in=True, dirichlet_out=True), 3))
        print(f"[ic] Leray-projected the IC: ||div B|| {div0:.3e} -> "
              f"{div_after:.3e}   (moved the field by {leray0:.3e})")
        results["div_ic_before_leray"] = div0
        div0 = div_after
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
    # eq.(2) is the closed form for the PRESCRIBED power-law profiles, so it
    # exists only for the logical IC; the Clebsch IC's profiles come from the
    # file and have no such closed form.
    H_an = (analytic_helicity(iota0, iota1, cli.iota_exp, cli.flux_exp)
            / B_norm ** 2) if cli.ic == "logical" else float("nan")
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
    if cli.ic == "logical":
        print(f"[ic] helicity: code {float(H0h):+.6e}   eq.(2) natural gauge "
              f"{H_an:+.6e}   difference (harmonic gauge) "
              f"{float(H0h) - H_an:+.3e}")
    elif cli.ic == "dzeta":
        # Zero shear, so eq.(2) gives exactly zero: dzeta carries toroidal
        # flux and no linked poloidal flux.  The measured value is the
        # discretisation's answer to a question with an exact answer.
        print(f"[ic] helicity: code {float(H0h):+.6e}   eq.(2) = 0 exactly "
              f"(zero shear)")
    else:
        print(f"[ic] helicity: code {float(H0h):+.6e}   (eq.(2) does not "
              f"apply -- the Clebsch profiles come from the file, not from a "
              f"power law)")
    E0 = 0.5 * float(seq.l2_norm_sq(B0, 2))
    normaliser = jax.jit(make_force_normaliser(seq))
    # Pressure tracking is only meaningful where the file carries GVEC's own
    # profile to compare against.
    # ALWAYS profile the multiplier, even with no file to compare against.
    # The scheme's fixed point is J x B = grad p, so the SHAPE of p answers a
    # question the force residual cannot: p -> constant means J x B -> 0, i.e.
    # the run is heading for a force-free (current-free, vacuum-like) state;
    # p staying structured means it settles on a pressure-balanced equilibrium.
    p_rhos = np.linspace(0.05, 0.95, 19)
    p_profiler = jax.jit(make_pressure_profiler(seq, jnp.asarray(p_rhos)))
    p_file_ref = (np.interp(p_rhos, clebsch_data["rho"], clebsch_data["p"])
                  if clebsch_data is not None else None)
    tn = time.perf_counter()
    gp_l2_0, gp_avg_0 = (float(v) for v in normaliser(B0))
    print(f"[ic] grad(B^2/2):  ||.||_L2 = {gp_l2_0:.6e}   "
          f"<|.|>_vol = {gp_avg_0:.6e}   ({time.perf_counter() - tn:.1f}s)")
    print("[ic]   force-residual denominator.  grad p is also a real scale "
          "here -- the scheme converges to JxB = grad p -- but it vanishes "
          "in the low-beta limit, and this one stays O(1) in every case.")
    results["gradp_l2_ic"] = gp_l2_0
    results["gradp_avg_ic"] = gp_avg_0
    gradp_mag0 = gp_l2_0
    F0v, p0v, _, _, _ = compute_force(B0, seq)
    # Beta at the IC is a free cross-check: it is scale-invariant (p is
    # quadratic in B) and mu0 = 1 here, so 2*p_ours/B^2 IS 2*mu0*p_SI/B^2 --
    # the same quantity the export's own `beta_mean` attribute reports.  If
    # the two disagree, either the force operator or the units are wrong.
    _pp, _bb, _vv = p_profiler(p0v, B0)
    beta_ic_prof, beta_ic_vol = beta_from_profiles(
        np.asarray(_pp), np.asarray(_bb), np.asarray(_vv))
    file_beta = None
    if cli.geometry in GVEC_GEOMETRIES:
        with h5py.File(GVEC_GEOMETRIES[cli.geometry], "r") as _h:
            file_beta = (float(_h.attrs["beta_mean"])
                         if "beta_mean" in _h.attrs else None)
    print(f"[ic] BETA (mu0=1, gauge p_edge=0): volume-avg {beta_ic_vol:+.4e}"
          f"   max |beta| {float(np.max(np.abs(beta_ic_prof))):.4e}"
          f"   axis {float(beta_ic_prof[0]):+.4e}"
          + (f"   file beta_mean {file_beta:.4e}" if file_beta is not None
             else "   (file carries no beta_mean)"))
    results.update(beta_ic_vol=beta_ic_vol,
                   beta_ic_max=float(np.max(np.abs(beta_ic_prof))),
                   beta_file_mean=file_beta)
    print(f"[ic] E = {E0:.6e}   ||F||_M = {float(seq.l2_norm(F0v, 2)):.4e}   "
          f"residual ||F||_L2/||grad(B^2/2)||_L2 = "
          f"{float(seq.l2_norm(F0v, 2)) / gradp_mag0:.4e}",
          flush=True)
    harm0 = harmonic_alignment(seq, ops, B0)
    if harm0 is not None:
        print(f"[ic] harmonic 2-form: cos = {harm0[0]:.6f}   residual off "
              f"span(h) = {harm0[1]:.4e}   amplitude = {harm0[2]:+.8e}")
        print("[ic]   the amplitude is an EXACT invariant of dB = curl E "
              "(D_1^T h = 0), which is why the field cannot relax to B = 0")
        results["harmonic_ic"] = list(harm0)
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
    final_B = {}
    for name in cli.arms.split(","):
        name = name.strip()
        method = ARMS[name]
        ts = TimeStepper(
            seq=seq, descent_method=method,
            dt_mode=(TimeStepChoice.ANALYTIC_LINESEARCH
                     if cli.dt_mode == "linesearch"
                     else TimeStepChoice.FIXED),
            timestep_mode=IntegrationScheme.EXPLICIT,
            history_size=cli.history, gamma=cli.gamma, mu=cli.mu)

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
                    seq.l2_norm(seq.apply_incidence_matrix(
                        state.B_n, 2, dirichlet_in=True,
                        dirichlet_out=True), 3),
                    Fu,
                    Fu / (state.F_norm * state.v_norm),
                    state.lbfgs_sy,
                    (Fu / state.dt) ** 0.5 / state.v_norm)

        state = initial_state(B0, ts, dt=cli.dt0)
        F0, p0, _, H0f, JxH0 = compute_force(
            B0, seq, dirichlet_H=ts.dirichlet_H)
        state = eqx.tree_at(
            lambda s: (s.F_norm, s.F_prev, s.p, s.H, s.JxH), state,
            (seq.l2_norm(F0, 2), F0, p0, H0f, JxH0))

        tr = {k: [] for k in ("E", "F", "dt", "div", "Fu", "cos", "sy", "eta",
                              "gain", "dE_meas", "dE_pred", "helicity",
                              "hel_it", "gradp_mag", "gradp_avg", "resid",
                              "p_scale", "p_resid", "p_spread", "JoverB",
                              "beta_vol", "beta_max", "beta_axis", "wall")}
        E_prev = E0
        t_arm = time.perf_counter()
        # Wall clock is sampled at the helicity iterations, NOT every step: a
        # per-step timer would need a device sync to mean anything, and that
        # sync is itself a cost the method does not otherwise pay.
        #
        # t_diag accumulates the time spent inside the diagnostic block, and
        # the recorded wall EXCLUDES it.  The diagnostics are expensive --
        # helicity alone is a k=1 Hodge solve -- and they are a measurement
        # apparatus, not part of the method.  Leaving them in would make an
        # arm look slower purely for having been watched more closely, and
        # --helicity-every would silently become a performance knob.
        t_diag = 0.0
        n_done = 0
        print(f"\n=== arm {name}  (method={method.name} m={cli.history} "
              f"gamma={cli.gamma} mu={cli.mu} "
              f"{'COLD' if cli.cold_start else 'warm'}-start) ===", flush=True)
        zero_guesses = jax.jit(lambda st: eqx.tree_at(
            lambda t: (t.p, t.p_v, t.H, t.JxH, t.E), st,
            (jnp.zeros_like(st.p), jnp.zeros_like(st.p_v),
             jnp.zeros_like(st.H), jnp.zeros_like(st.JxH),
             jnp.zeros_like(st.E))))

        def eta_at(it):
            if cli.eta_max == 0.0:
                return 0.0
            frac = it / max(cli.steps, 1)
            if cli.eta_schedule == "tanh":
                return cli.eta_max * 0.5 * (
                    1.0 - np.tanh(4.0 * np.pi * (frac - 0.5)))
            if cli.eta_schedule == "linear":
                return cli.eta_max * (1.0 - frac)
            return cli.eta_max

        for it in range(1, cli.steps + 1):
            if cli.cold_start:
                state = zero_guesses(state)
            if cli.eta_max > 0.0:
                state = eqx.tree_at(lambda t: t.eta, state, eta_at(it))
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
            tr["eta"].append(float(state.eta))
            tr["dE_meas"].append(dE_meas)
            tr["dE_pred"].append(dE_pred)
            E_prev = E
            n_done = it
            if it % cli.helicity_every == 0 or it == 1:
                t_diag0 = time.perf_counter()
                # Sampled BEFORE the diagnostics run and already net of every
                # earlier one, so this is solve-only wall clock at this step.
                tr["wall"].append(t_diag0 - t_arm - t_diag)
                # The denominator moves as B does, so it is re-measured here
                # rather than frozen at the IC.
                # ||J||/||B|| with J = weak_curl(B) = the codifferential.
                # For a smooth field this is O(1/a); if the descent is
                # shredding B at the grid scale it GROWS, which separates
                # "physically chaotic but smooth" from "numerically rough".
                tr["JoverB"].append(float(
                    seq.l2_norm(seq.apply_weak_curl(state.B_n), 1)
                    / seq.l2_norm(state.B_n, 2)))
                gp_l2, gp_avg = (float(v) for v in normaliser(state.B_n))
                tr["gradp_mag"].append(gp_l2)
                tr["gradp_avg"].append(gp_avg)
                tr["resid"].append(float(state.F_norm) / gp_l2)
                # state.p is the multiplier from compute_force(B_n), in the
                # physical-pressure convention apply_leray_projection returns.
                p_prof, bsq_prof, Vp = p_profiler(state.p, state.B_n)
                p_prof = np.asarray(p_prof)
                beta_prof, beta_vol = beta_from_profiles(
                    p_prof, np.asarray(bsq_prof), np.asarray(Vp))
                tr["beta_vol"].append(beta_vol)
                tr["beta_max"].append(float(np.max(np.abs(beta_prof))))
                tr["beta_axis"].append(float(beta_prof[0]))
                tr["p_spread"].append(float(p_prof.max() - p_prof.min()) / E)
                if p_file_ref is not None:
                    # If the scheme really converges to JxB = grad p, this
                    # shape must not drift away from the file's own pressure.
                    k_p, r_p = pressure_shape_residual(p_prof, p_file_ref)
                    tr["p_scale"].append(k_p)
                    tr["p_resid"].append(r_p)
                h, A_new = get_helicity(state.B_n, seq, state.A)
                state = eqx.tree_at(lambda s: s.A, state, A_new)
                tr["helicity"].append(float(h))
                tr["hel_it"].append(it)
                # PRINT it, do not merely record it.  Helicity used to appear
                # only in the end-of-arm summary, which makes a running job
                # impossible to judge: energy and ||F|| alone cannot tell a
                # healthy descent from one that is dissolving the topology.
                # Both forms are shown -- the ABSOLUTE change is what
                # correlates with surface destruction (handoff s19.2), the
                # relative one is what is conventionally quoted, and they
                # disagree badly when H itself is near zero.
                h0 = tr["helicity"][0]
                print(f"  it {it:>5d}  E={E:.8e}  |F|={float(state.F_norm):.4e}"
                      f"  H={float(h):+.6e}  dH={float(h) - h0:+.3e}"
                      f"  dH/H={(float(h) - h0) / abs(h0):+.3e}"
                      f"  beta_vol={beta_vol:+.3e}"
                      f"  [{t_diag0 - t_arm - t_diag:.0f}s solve"
                      f" +{t_diag:.0f}s diag]", flush=True)
                t_diag += time.perf_counter() - t_diag0
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
        print(f"    RESIDUAL ||F||_L2 / ||grad(B^2/2)||_L2  "
              f"{tr['resid'][0]:.4e} -> "
              f"{tr['resid'][-1]:.4e}   ({tr['resid'][0] / tr['resid'][-1]:.2f}x"
              f" reduction);  ||grad(B^2/2)||_L2 {tr['gradp_mag'][0]:.4e} -> "
              f"{tr['gradp_mag'][-1]:.4e}"
              f"   [volume average, recorded only: "
              f"{tr['gradp_avg'][0]:.4e} -> {tr['gradp_avg'][-1]:.4e}]")
        if cli.eta_max > 0.0:
            print(f"    eta schedule '{cli.eta_schedule}' peak {cli.eta_max:.3e}"
                  f":  {tr['eta'][0]:.3e} -> {max(tr['eta']):.3e} -> "
                  f"{tr['eta'][-1]:.3e}")
            print("    NOTE the G1 identity below is EXPECTED to break here. "
                  "With eta > 0 the step is dB = curl(u x H - eta J), so "
                  "<B,dB>_M = -(F,u)_M - eta||J||^2_M1, but ANALYTIC_LINESEARCH "
                  "computes dt = (F,u)/||dB||^2 and omits the resistive term. "
                  "dt is then no longer the line minimiser -- it UNDER-steps. "
                  "The discrepancy is the size of what it omits.")
        if cli.dt_mode == "fixed":
            print("    NOTE G1 below does NOT apply: dE_pred = -dt(F,u)/2 is "
                  "derived assuming dt is the exact line MINIMISER. Under a "
                  "fixed dt the true drop is dE = -dt(F,u) + dt^2/2||dB||^2, "
                  "so for dt well below the minimiser dE_meas -> 2*dE_pred "
                  "and the 'identity' reads as broken by a factor 2 BY "
                  "CONSTRUCTION. A ratio near 2 is the healthy signature here.")
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
        harm = harmonic_alignment(seq, ops, state.B_n)
        if harm is not None:
            print(f"    HARMONIC cos {harm0[0]:.6f} -> {harm[0]:.6f}   "
                  f"residual off span(h) {harm0[1]:.4e} -> {harm[1]:.4e}")
            print(f"    HARMONIC AMPLITUDE {harm0[2]:+.8e} -> {harm[2]:+.8e}"
                  f"   relative drift "
                  f"{(harm[2] - harm0[2]) / abs(harm0[2]):+.3e}"
                  f"   -- EXACT invariant, must be at round-off")
        print(f"    ROUGHNESS ||J||/||B||: {tr['JoverB'][0]:.4e} -> "
              f"{tr['JoverB'][-1]:.4e}   ({tr['JoverB'][-1] / tr['JoverB'][0]:.2f}x)"
              f"   -- growth means grid-scale structure, i.e. numerically "
              f"shredded rather than merely chaotic")
        print(f"    BETA (mu0=1, gauge p_edge=0): volume-avg "
              f"{tr['beta_vol'][0]:+.4e} -> {tr['beta_vol'][-1]:+.4e}   "
              f"max |beta| {tr['beta_max'][0]:.4e} -> {tr['beta_max'][-1]:.4e}"
              f"   axis {tr['beta_axis'][0]:+.4e} -> {tr['beta_axis'][-1]:+.4e}")
        print(f"    PRESSURE profile spread (max-min)/E: "
              f"{tr['p_spread'][0]:.4e} -> {tr['p_spread'][-1]:.4e}   -- the "
              f"fixed point is JxB = grad p, so this going to ZERO means "
              f"force-free / vacuum-like, and staying finite means a "
              f"pressure-balanced equilibrium")
        if tr["p_resid"]:
            print(f"    PRESSURE vs GVEC (shape, one fitted scale): "
                  f"{tr['p_resid'][0]:.4e} -> {tr['p_resid'][-1]:.4e}"
                  f"   -- the scheme's fixed point is JxB = grad p, so this "
                  f"should NOT drift away from the file")
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
            div_max=float(max(tr["div"])),
            resid_first=tr["resid"][0], resid_final=tr["resid"][-1],
            p_resid_first=(tr["p_resid"][0] if tr["p_resid"] else None),
            p_resid_final=(tr["p_resid"][-1] if tr["p_resid"] else None),
            gradp_l2_final=tr["gradp_mag"][-1],
            gradp_avg_final=tr["gradp_avg"][-1],
            beta_vol_first=tr["beta_vol"][0],
            beta_vol_final=tr["beta_vol"][-1],
            beta_max_final=tr["beta_max"][-1],
            harmonic_final=(list(harm) if harm is not None else None))
        final_B[name] = np.asarray(state.B_n)
        # Write after EVERY arm, not once at the end.  Deferring the whole
        # save until all arms finish means a job that hits its time limit --
        # which is the normal outcome of a --seconds-per-arm budget -- loses
        # every field it computed.  Rewriting the file per arm costs
        # milliseconds and the last complete arm always survives.
        if cli.save_b:
            with h5py.File(cli.save_b, "w") as f:
                f.create_dataset("B_ic", data=np.asarray(B0))
                for nm, arr in final_B.items():
                    f.create_dataset(f"B_final_{nm}", data=arr)
                f.attrs["geometry"] = cli.geometry
                f.attrs["ns"] = list(ns)
                f.attrs["p"] = cli.p
                f.attrs["ic"] = cli.ic
                f.attrs["steps"] = cli.steps
                f.attrs["gamma"] = cli.gamma
                f.attrs["mu"] = cli.mu
            print(f"    wrote {cli.save_b} (through arm '{name}')", flush=True)
        if cli.out:
            json.dump(results, open(cli.out, "w"), indent=1)

    # --- persist the fields BEFORE any tracing, so a raised zeta gate cannot
    #     cost the run its data --------------------------------------------
    if cli.save_b:
        with h5py.File(cli.save_b, "w") as f:
            f.create_dataset("B_ic", data=np.asarray(B0))
            for name, arr in final_B.items():
                f.create_dataset(f"B_final_{name}", data=arr)
            f.attrs["geometry"] = cli.geometry
            f.attrs["ns"] = list(ns)
            f.attrs["p"] = cli.p
            f.attrs["ic"] = cli.ic
            f.attrs["steps"] = cli.steps
            f.attrs["gamma"] = cli.gamma
            f.attrs["mu"] = cli.mu
        print(f"\nwrote {cli.save_b}", flush=True)

    if cli.out:
        json.dump(results, open(cli.out, "w"), indent=1)
        print(f"wrote {cli.out}", flush=True)

    # --- Poincare, last ---------------------------------------------------
    if cli.poincare:
        nfp = geometry_nfp(cli.geometry)
        outdir = os.path.dirname(cli.out) if cli.out else "."
        results["poincare"] = {}
        print(f"\n[poincare] nfp={nfp}  seeds={cli.pc_seeds}  "
              f"periods={cli.pc_periods}  zeta={cli.pc_zeta}", flush=True)
        for tag, dof in [("ic", jnp.asarray(B0))] + [
                (f"final_{k}", jnp.asarray(v)) for k, v in final_B.items()]:
            results["poincare"][tag] = render_poincare(
                seq, dof, nfp, tag, outdir, cli)
            if cli.out:
                json.dump(results, open(cli.out, "w"), indent=1)


if __name__ == "__main__":
    main()
