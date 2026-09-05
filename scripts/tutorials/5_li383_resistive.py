"""Tutorial 5: reconnection with finite resistivity on li383.

Tutorials 3 and 4 stayed ideal (eta = 0): a frozen-in flow that lowers the
energy without ever changing the field's topology. Turn on resistivity and that
constraint breaks -- a step becomes the ideal move followed by a backward-Euler
diffusion of ``B`` (an implicit resistive solve), and field lines can
**reconnect**: nested surfaces merge, a seeded island heals or grows, and
helicity is no longer conserved, it decays at the resistive rate.

This tutorial is arranged to be cheap. It **warm-starts from Tutorial 3's
relaxed field** if its run ``outputs/tutorials/li383_relaxation`` is
present (same ``(10, 16, 16) p = 2`` mesh), so the initial descent is not
repeated; otherwise it builds the equilibrium initial condition itself. It then
takes a **single resistive step** at ``--eps`` -- one reconnection event --
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
ap.add_argument("--warm-start", default="outputs/tutorials/li383_relaxation",
                help="Tutorial 3's run directory; warm-start from its last checkpoint if present")
ap.add_argument("--eps", type=float, default=1e-4,
                help="resistive dose eps = eta*dt of the single reconnection step")
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
# Now we get the starting field: warm-start from Tutorial 3's relaxed B if its
# checkpoint is on disk and matches this mesh, otherwise build the equilibrium
# initial condition ourselves (optionally with a resonant seed).
B0 = None
ws_json = os.path.join(cli.warm_start, "relax.json")
if os.path.exists(ws_json):
    with open(ws_json) as fh:
        ws = json.load(fh)["params"]
    ckpts = sorted(glob.glob(os.path.join(cli.warm_start, "checkpoints", "state_*.h5")))
    if tuple(ws["ns"]) == ns and int(ws["p"]) == cli.p and ckpts:
        with h5py.File(ckpts[-1], "r") as fh:
            B0 = jnp.asarray(np.asarray(fh["B_n"]))
        print(f"[ic] warm-started from Tutorial 3: {ckpts[-1]} (ns={ws['ns']} p={ws['p']})")
    else:
        print(f"[ic] run {cli.warm_start} is ns={ws['ns']} p={ws['p']} "
              f"(need {list(ns)} p={cli.p}); building the IC instead")
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
# Now we relax ideally for another 500 steps to a clean floor. The ideal tail
# conserves helicity and just settles the reconnected field.
ts_ideal = TimeStepper(seq=seq, cfl=0.5, history_size=1, velocity_smoothing_order=1)
print(f"[relax] {cli.outer * cli.inner} ideal steps to a clean floor")
res = relax(initial_state(B_reconnected, ts_ideal), ts_ideal, steps=cli.outer * cli.inner,
            chunk=cli.inner, floor_tol=cli.floor_tol)
F = np.asarray(res.trace["F"], dtype=float)
dE = np.asarray(res.trace["dE"], dtype=float)
H = np.asarray(res.qoi["helicity"], dtype=float)
print(f"[relax] {res.steps} steps ({res.stop}): ||F|| {F[0]:.3e} -> {F[-1]:.3e}, "
      f"E_0 - E = {-dE.sum():.3e}, H {H[0]:+.3e} -> {H[-1]:+.3e} (ideal tail conserves it), "
      f"||div B|| {float(res.trace['div'][-1]):.1e}")
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
# Now we archive the run the way scripts/relax.py does -- relax.json and the
# checkpoints of the field before the reconnection and at the end -- so
# scripts/poincare_relax.py can redraw the sections at any planes from it.
os.makedirs(os.path.join(cli.out, "checkpoints"), exist_ok=True)
write_checkpoint(os.path.join(cli.out, "checkpoints", "state_000000.h5"), initial_state(B0, ts_ideal), 0)
write_checkpoint(os.path.join(cli.out, "checkpoints", f"state_{res.steps:06d}.h5"), res.state, res.steps)
params = dict(geometry_path=os.path.abspath(cli.geometry), ns=list(ns), p=cli.p, nfp=None,
              r_refine="", precision=str(mrx.DTYPE), steps=res.steps, scheme="explicit",
              auxiliary_B_field=False, ic="warmstart", eps=cli.eps, seed=cli.seed, seed_eps=cli.seed_eps)
with open(os.path.join(cli.out, "relax.json"), "w") as fh:
    json.dump(dict(params=params, trace=res.trace, qoi=res.qoi, reconnect=[]), fh, indent=1)
print(f"  -> {cli.out}/relax.json and checkpoints/")
