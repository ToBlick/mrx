"""Tutorial 6: reconnection with finite resistivity on li383.

Tutorials 3 to 5 stayed ideal (eta = 0): a frozen-in flow that lowers the
energy without ever changing the field's topology. Turn on resistivity and that
constraint breaks -- a step becomes the ideal move followed by a backward-Euler
diffusion of ``B`` (an implicit resistive solve), and field lines can
**reconnect**: nested surfaces merge, a seeded island heals or grows, and
helicity is no longer conserved, it decays at the resistive rate.

This tutorial is arranged to be cheap. It **warm-starts from Tutorial 4's
Newton floor** (``outputs/tutorials/li383_newton``) or, failing that, from
Tutorial 3's relaxed field, the user's run or the shipped state in
``data/tutorials/``, on the same ``(10, 16, 16) p = 2`` mesh; otherwise it
builds the equilibrium initial condition itself. It then takes a **single resistive step** at ``--eps`` --
one reconnection event -- and relaxes ideally to a clean floor again with
Newton (Tutorial 4): the reconnected field is near its floor already, so the
direction of the second variation is the right tool. Finally it draws
Poincare sections of the field before and after, so the magnetic islands the
reconnection opens or heals are visible.

Pass ``--seed`` (the Tutorial 5 syntax) when it falls back to building the IC,
to start from a seeded island and watch it reconnect. Runs in the default
float32.

    python -u scripts/tutorials/6_li383_resistive.py
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
                help="a VMEC wout (.nc) or a GVEC state file (.dat); match Tutorials 3 and 4")
ap.add_argument("--ns", default="10,16,16")
ap.add_argument("--p", type=int, default=2)
ap.add_argument("--warm-start",
                default="outputs/tutorials/li383_newton,outputs/tutorials/li383_relaxation,"
                        "data/tutorials/li383_newton,data/tutorials/li383_relaxation",
                help="run directories, first present wins: Tutorials 4 and 3, the user's runs then the shipped states")
ap.add_argument("--eps", type=float, default=1e-4,
                help="resistive dose eps = eta*dt of the single reconnection step")
ap.add_argument("--seed", default="",
                help='optional resonant seed "m,n,rho0,width" (used only when building the IC)')
ap.add_argument("--seed-eps", type=float, default=0.0)
ap.add_argument("--newton-steps", type=int, default=5,
                help="Newton steps of the ideal tail after the resistive step (a multiple of 5)")
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
import glob
import json

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
from mrx.initial_conditions import initial_field
from mrx.nullspace import compute_nullspaces
from mrx.plotting import get_2d_grids, plot_torus, plot_twin_axis, render_section
from mrx.poincare import (logical_field, require_zeta_parameterisation, seed_from_axis,
                          trace_and_classify, section_RZ, surface_label)
from mrx.relaxation import (TimeStepper, compute_force, initial_state, relax, resistive_step,
                            weak_pressure, write_checkpoint)

print(f"[env] mrx precision {mrx.DTYPE}")

# %%
# Now we build the de Rham sequence on li383's geometry and its harmonic forms,
# the operators every solve and the Poincare tracing lean on.
nfp = geometry_nfp(cli.geometry)
seq, ops = build_sequence(cli.geometry, ns, cli.p)
compute_nullspaces(seq)

# %%
# Now we get the starting field: warm-start from Tutorial 4's Newton floor or
# Tutorial 3's relaxed B -- the user's runs, then the states shipped in
# data/tutorials/ -- whichever checkpoint is on disk first and matches this
# mesh, otherwise build the equilibrium initial condition ourselves (optionally
# with a resonant seed).
B0 = None
for run in cli.warm_start.split(","):
    ws_json = os.path.join(run, "relax.json")
    if not os.path.exists(ws_json):
        continue
    with open(ws_json) as fh:
        ws = json.load(fh)["params"]
    ckpts = sorted(glob.glob(os.path.join(run, "checkpoints", "state_*.h5")))
    if tuple(ws["ns"]) == ns and int(ws["p"]) == cli.p and ckpts:
        with h5py.File(ckpts[-1], "r") as fh:
            B0 = jnp.asarray(np.asarray(fh["B_n"]))
        print(f"[ic] warm-started from {ckpts[-1]} (ns={ws['ns']} p={ws['p']})")
        break
    print(f"[ic] run {run} is ns={ws['ns']} p={ws['p']} (need {list(ns)} p={cli.p}); skipped")
if B0 is None:
    seed = None
    if cli.seed:
        m, n, rho0, width = (float(v) for v in cli.seed.split(","))
        seed = (int(m), int(n), rho0, width, cli.seed_eps)
        print(f"[ic] seed (m, n) = ({int(m)}, {int(n)}) at rho0 {rho0:g}, eps {cli.seed_eps:.2e}")
    B0, ic = initial_field(seq, seed)
    print(f"[ic] built the equilibrium IC: ||B||_M {ic['B_norm_raw']:.4e}, "
          f"||div B|| {ic['div']:.2e}, wall-normal {ic['wall_discarded']:.1e}")

# %%
# Now we do the single reconnection step with mrx.relaxation.resistive_step: one
# backward-Euler resistive substep, eps = eta*dt. Unlike the ideal descent it can
# change the topology (it dissipates helicity), so field lines reconnect.
B_reconnected, _, rel = resistive_step(B0, seq, cli.eps)
print(f"[reconnect] one resistive step at eps = {cli.eps:.1e}: "
      f"||dB||/||B|| = {rel:.2e} (the reconnection; the ideal descent could not do this)")

# %%
# Now we relax ideally back to a clean floor with Newton (Tutorial 4's stepper:
# the truncated MINRES solve of the second variation, the line search capped at
# the Newton step). The ideal tail conserves helicity and just settles the
# reconnected field.
ts_newton = TimeStepper(seq=seq, cfl=0.5, history_size=0, velocity_smoothing_order=1,
                        newton=True, newton_tol=0.1, newton_maxiter=100,
                        newton_precond="harmonic", newton_dt_cap=1.0)
print(f"[relax] {cli.newton_steps} Newton steps to a clean floor")
res = relax(initial_state(B_reconnected, ts_newton), ts_newton, steps=cli.newton_steps,
            chunk=5, floor_tol=0.0)
F = np.asarray(res.trace["F"], dtype=float)
dE = np.asarray(res.trace["dE"], dtype=float)
H = np.asarray(res.qoi["helicity"], dtype=float)
it_n = np.asarray(res.trace["newton_it"])
print(f"[relax] {res.steps} steps ({res.stop}): ||F|| {F[0]:.3e} -> {F[-1]:.3e} "
      f"(lowest {F.min():.3e} at step {F.argmin() + 1}), "
      f"E_0 - E = {-dE.sum():.3e}, H {H[0]:+.3e} -> {H[-1]:+.3e} (ideal tail conserves it), "
      f"||div B|| {float(res.trace['div'][-1]):.1e}; MINRES iterations mean {np.abs(it_n).mean():.0f}, "
      f"fallbacks {int(np.asarray(res.trace['newton_fallback']).sum())}")
B = res.state.B_n

# %%
# Now we plot the force residual against the energy removed over the ideal tail.
fig, _ = plot_twin_axis(F, np.cumsum(-dE), left_label=r"$\|F\|_M$", right_label=r"$E_0 - E$",
                        left_plot_kwargs=dict(marker=""), right_plot_kwargs=dict(marker=""))
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
sections(B, "after", f"after reconnection + Newton {ns} p={cli.p}")

# %%
# Now we draw the weak pressure of the reconnected, relaxed field on the torus.
def weak_p(field):
    _, _, J, Hf, _ = compute_force(field, seq)
    p_w, _, _ = weak_pressure(J, Hf, seq)
    return np.asarray(p_w)

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
# Now we archive the run the way scripts/relax.py does -- relax.json and the
# checkpoints of the field before the reconnection and at the end -- so
# scripts/poincare_trace.py can trace the sections at any planes from it.
os.makedirs(os.path.join(cli.out, "checkpoints"), exist_ok=True)
write_checkpoint(os.path.join(cli.out, "checkpoints", "state_000000.h5"), initial_state(B0, ts_newton), 0)
write_checkpoint(os.path.join(cli.out, "checkpoints", f"state_{res.steps:06d}.h5"), res.state, res.steps)
params = dict(geometry_path=os.path.abspath(cli.geometry), ns=list(ns), p=cli.p, nfp=None,
              knots=None, precision=str(mrx.DTYPE), steps=res.steps, scheme="explicit",
              auxiliary_B_field=False, ic="warmstart", eps=cli.eps, seed=cli.seed, seed_eps=cli.seed_eps,
              newton=True)
with open(os.path.join(cli.out, "relax.json"), "w") as fh:
    json.dump(dict(params=params, trace=res.trace, qoi=res.qoi, reconnect=[]), fh, indent=1)
print(f"  -> {cli.out}/relax.json and checkpoints/")
