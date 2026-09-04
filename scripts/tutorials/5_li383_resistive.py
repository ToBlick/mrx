"""Tutorial 5: reconnection with finite resistivity on li383.

Tutorials 3 and 4 stayed ideal (eta = 0): a frozen-in flow that lowers the
energy without ever changing the field's topology. Turn on resistivity and that
constraint breaks -- a step becomes the ideal move followed by a backward-Euler
diffusion of ``B`` (an implicit resistive solve), and field lines can
**reconnect**: nested surfaces merge, a seeded island heals or grows, and
helicity is no longer conserved, it decays at the resistive rate.

This tutorial is arranged to be cheap. It **warm-starts from Tutorial 3's
relaxed field** if its checkpoint ``outputs/tutorials/li383_relaxation/B.h5`` is
present (same ``(10, 16, 16) p = 2`` mesh), so the initial descent is not
repeated; otherwise it builds the equilibrium initial condition itself. It then
takes a **single resistive step** at ``--eta-max`` -- one reconnection event --
and relaxes ideally for another 500 steps to a clean floor. Finally it draws
Poincare sections of the field before and after, so the magnetic islands the
reconnection opens or heals are visible.

Pass ``--seed`` (the Tutorial 4 syntax) when it falls back to building the IC,
to start from a seeded island and watch it reconnect. Runs in the default
float32 with velocity smoothing of order 1 (gamma = 1).

    python -u scripts/tutorials/5_li383_resistive.py
"""

# %%
# Now we parse the run's options. Everything has a default, so in a notebook
# the cell just uses them; from the command line the flags below still apply.
from __future__ import annotations

import argparse
import os
import sys

_INTERACTIVE = "ipykernel" in sys.modules

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--geometry", default="data/wout_li383_low_res_reference.nc",
                help="a VMEC wout (.nc) or a GVEC state file (.dat); match Tutorial 3")
ap.add_argument("--ns", default="10,16,16")
ap.add_argument("--p", type=int, default=2)
ap.add_argument("--warm-start", default="outputs/tutorials/li383_relaxation/B.h5",
                help="Tutorial 3's B.h5; warm-start from its relaxed field if present")
ap.add_argument("--eta-max", type=float, default=1e-4,
                help="resistivity of the single reconnection step")
ap.add_argument("--seed", default="",
                help='optional resonant seed "m,n,rho0,width" (used only when building the IC)')
ap.add_argument("--seed-eps", type=float, default=0.0)
ap.add_argument("--outer", type=int, default=10, help="outer (recorded) iterations of the ideal tail")
ap.add_argument("--inner", type=int, default=50, help="compiled steps per outer iteration")
ap.add_argument("--floor-tol", type=float, default=1e-4, help="stop the ideal tail below this")
ap.add_argument("--seeds", type=int, default=24, help="Poincare field lines")
ap.add_argument("--periods", type=int, default=200, help="field periods per traced line")
ap.add_argument("--cuts", type=int, default=6)
ap.add_argument("--out", default="outputs/tutorials/li383_resistive")
cli = ap.parse_args([] if _INTERACTIVE else None)
ns = tuple(int(v) for v in cli.ns.split(","))
os.makedirs(cli.out, exist_ok=True)

# %%
# Now we import MRX and its relaxation and Poincare machinery. Precision is the
# package default (float32, the production precision); nothing is set here.
import h5py
import jax.numpy as jnp
import matplotlib
if not _INTERACTIVE:
    matplotlib.use("Agg")  # headless as a script; a notebook keeps its inline backend
import matplotlib.pyplot as plt
import numpy as np
import mrx
from mrx.differential_forms import DiscreteFunction
from mrx.geometry import build_sequence, geometry_nfp
from mrx.gvec import load_clebsch
from mrx.initial_conditions import clebsch_potential_form, divergence_norm, potential_two_form
from mrx.nullspace import compute_nullspaces
from mrx.plotting import get_2d_grids, plot_torus, plot_twin_axis, render_section
from mrx.poincare import (logical_field, require_zeta_parameterisation, seed_from_axis,
                          trace_and_classify, section_RZ, surface_label)
from mrx.relaxation import (DescentMethod, TimeStepChoice, TimeStepper, compute_force,
                            relaxation_loop, weak_pressure)

print(f"[env] mrx precision {mrx.DTYPE}")

# %%
# Now we build the de Rham sequence on li383's geometry and its harmonic forms,
# the operators every solve and the Poincare tracing lean on.
nfp = geometry_nfp(cli.geometry)
seq, ops = build_sequence(cli.geometry, ns, cli.p)
seq.set_operators(compute_nullspaces(seq, ops))

# %%
# Now we get the starting field: warm-start from Tutorial 3's relaxed B if its
# checkpoint is on disk and matches this mesh, otherwise build the equilibrium
# initial condition ourselves (optionally with a resonant seed).
B0 = None
if os.path.exists(cli.warm_start):
    with h5py.File(cli.warm_start, "r") as fh:
        ws_ns = [int(v) for v in fh.attrs["ns"]]
        ws_p = int(fh.attrs["p"])
        B_final = np.asarray(fh["B_final"][:])
    if tuple(ws_ns) == ns and ws_p == cli.p and B_final.shape[0] == seq.n(2):
        B0 = jnp.asarray(B_final)
        print(f"[ic] warm-started from Tutorial 3: {cli.warm_start} (ns={ws_ns} p={ws_p})")
    else:
        print(f"[ic] checkpoint {cli.warm_start} is ns={ws_ns} p={ws_p} "
              f"(need {list(ns)} p={cli.p}); building the IC instead")
if B0 is None:
    cb = load_clebsch(cli.geometry)
    seed = None
    if cli.seed:
        m, n, rho0, width = (float(v) for v in cli.seed.split(","))
        seed = (int(m), int(n), rho0, width, cli.seed_eps)
        print(f"[ic] seed (m, n) = ({int(m)}, {int(n)}) at rho0 {rho0:g}, eps {cli.seed_eps:.2e}")
    B0, norm, wall = potential_two_form(seq, clebsch_potential_form(cb, seed))
    print(f"[ic] built the equilibrium IC: ||B||_M {norm:.4e}, "
          f"||div B|| {divergence_norm(seq, B0):.2e}, wall-normal {wall:.1e}")

# %%
# Now we do the single reconnection step: one resistive substep at eta = eta-max.
# Unlike the ideal descent it can change the topology, so helicity drops here.
smoothing_scale = 0.064 / ns[0] ** 2
ts_reconnect = TimeStepper(seq=seq, descent_method=DescentMethod.LBFGS,
                           dt_mode=TimeStepChoice.ANALYTIC_LINESEARCH, cfl=0.5,
                           eta_every=1, resistive=True, history_size=1,
                           velocity_smoothing_order=1, velocity_smoothing_scale=smoothing_scale)
print(f"[reconnect] one resistive step at eta = {cli.eta_max:.1e}")
state, tr_r = relaxation_loop(B0, ts_reconnect, num_iters_outer=1, num_iters_inner=1,
                              dt0=1.0, force_tolerance=0.0,
                              resistivity_schedule=lambda i: cli.eta_max)
Hr = np.asarray(tr_r["helicity"], dtype=float)
print(f"[reconnect] helicity {Hr[0]:+.3e} -> {Hr[-1]:+.3e} (dH = {Hr[-1] - Hr[0]:+.2e}): "
      "the resistive step dissipates helicity, so the topology can change.")
B_reconnected = state.B_n

# %%
# Now we relax ideally for another 500 steps (eta = 0) to a clean floor. The
# ideal tail conserves helicity and just settles the reconnected field.
ts_ideal = TimeStepper(seq=seq, descent_method=DescentMethod.LBFGS,
                       dt_mode=TimeStepChoice.ANALYTIC_LINESEARCH, cfl=0.5,
                       eta_every=1, resistive=False, history_size=1,
                       velocity_smoothing_order=1, velocity_smoothing_scale=smoothing_scale)
print(f"[relax] {cli.outer * cli.inner} ideal steps to a clean floor")
state, traces = relaxation_loop(B_reconnected, ts_ideal, num_iters_outer=cli.outer,
                                num_iters_inner=cli.inner, dt0=1.0,
                                force_tolerance=cli.floor_tol)
F = np.asarray(traces["force_norm"], dtype=float)
E = np.asarray(traces["energy"], dtype=float)
H = np.asarray(traces["helicity"], dtype=float)
steps = np.asarray(traces["iteration"])
print(f"[relax] {steps[-1]} steps: ||F|| {F[0]:.3e} -> {F[-1]:.3e}, E_0 - E = {E[0] - E[-1]:.3e}, "
      f"H {H[0]:+.3e} -> {H[-1]:+.3e} (ideal tail conserves it), "
      f"||div B|| {float(traces['divergence_B'][-1]):.1e}")
B = state.B_n

# %%
# Now we plot the force residual against the energy over the ideal tail.
fig, _ = plot_twin_axis(F, E, left_label=r"$\|F\|_M$", right_label=r"$E$",
                        num_iters_inner=cli.inner, left_marker="o", right_marker="s")
path = os.path.join(cli.out, "trace.png")
fig.savefig(path, dpi=200)
if _INTERACTIVE:
    plt.show()
else:
    plt.close(fig)
print(f"  -> {path}")

# %%
# Now we take Poincare sections of the field BEFORE the reconnection step and
# AFTER the ideal tail, at five toroidal planes. This is where the magnetic
# islands show: the resistive step can open or heal a chain the ideal descent
# would have frozen. Each field is traced once and cut at all five planes.
def sections(B_dof, tag, title):
    field = logical_field(seq, jnp.asarray(B_dof), 2, True)
    require_zeta_parameterisation(field, name=tag)
    seeds = seed_from_axis(field, cli.seeds, 8, n_rays=4, steps_per_period=32)
    res = trace_and_classify(field, seeds, nfp, n_periods=cli.periods,
                             steps_per_period=32, saves_per_period=8)
    render_keep = ~(res["escaped"] | ~res["ok"])
    for plane in (0.0, 0.125, 0.25, 0.375, 0.5):
        R, Z, aR, aZ, _, _, lr, lth = section_RZ(seq, res["ys"], res["axis"], 8, plane)
        a_eff, xlabel = surface_label(R, Z, aR, aZ)
        fig, _ = render_section(
            R, Z, res["iota"], res["iota_err"], res["seeds"][:, 0], render_keep,
            title=f"{title}  |  $\\zeta = {plane:g}$",
            subtitle=f"nfp = {nfp}   |   h/2 drift {res['drift']:.1e}",
            axis_RZ=(aR, aZ), profile_x=a_eff, profile_xlabel=xlabel, nfp=nfp,
            logical=(lr, lth), iota_scatter=res["iota_scatter"])
        path = os.path.join(cli.out, f"poincare_{tag}_zeta{plane:g}.png")
        fig.savefig(path, dpi=200)
        if _INTERACTIVE:
            plt.show()
        else:
            plt.close(fig)
        print(f"  -> {path}")

sections(B0, "before", f"before reconnection {ns} p={cli.p}")
sections(B, "after", f"after reconnection + relax {ns} p={cli.p}")

# %%
# Now we draw the weak pressure of the reconnected, relaxed field on the torus.
def weak_p(field):
    _, _, J, Hf, _ = compute_force(field, seq)
    p_w, _, _ = weak_pressure(J, Hf, seq)
    return np.asarray(p_w)

pw_ic = weak_p(B0)
pw_final = weak_p(B)
pw = DiscreteFunction(jnp.asarray(pw_final), seq.basis_0, seq.E(0, True))

def p_h(x):
    return pw(x)[0]

zetas = np.arange(cli.cuts) / cli.cuts
npt = 48
grids_pol = [get_2d_grids(seq.map, cut_axis=2, cut_value=float(z), nx=npt, ny=npt, nz=1)
             for z in zetas]
grid_surface = get_2d_grids(seq.map, cut_axis=0, cut_value=1.0 - 1e-6,
                            ny=4 * npt, nz=4 * npt, invert_z=True)
fig, _ = plot_torus(p_h, grids_pol, grid_surface, cstride=8, gridlinewidth=0.3,
                    elev=25, azim=40, cbar_label=r"$p_w$")
path = os.path.join(cli.out, "torus_pw.png")
fig.savefig(path, dpi=200)
if _INTERACTIVE:
    plt.show()
else:
    plt.close(fig)
print(f"  -> {path}")

# %%
# Now we archive B (initial and final) for scripts/poincare_relax.py, which can
# redraw the sections at any planes from this file.
h5_path = os.path.join(cli.out, "B.h5")
attrs = dict(geometry_path=os.path.abspath(cli.geometry), ns=list(ns), p=cli.p,
             nfp="", maxiter=10_000, precision=str(mrx.DTYPE), steps=int(steps[-1]),
             method="lbfgs", eta_max=cli.eta_max, ic="warmstart",
             seed=cli.seed, seed_eps=cli.seed_eps)
with h5py.File(h5_path, "w") as fh:
    fh.create_dataset("B_ic", data=np.asarray(B0))
    fh.create_dataset("B_final", data=np.asarray(B))
    fh.create_dataset("pw_ic", data=pw_ic)
    fh.create_dataset("pw_final", data=pw_final)
    for k, v in attrs.items():
        fh.attrs[k] = v
print(f"  -> {h5_path}")
