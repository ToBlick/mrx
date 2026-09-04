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
and relaxes ideally for another 500 steps to a clean floor. The helicity drop
across the resistive step is the reconnection; the ideal tail conserves it.

Pass ``--seed`` (the Tutorial 4 syntax) when it falls back to building the IC,
to watch a seeded island reconnect. Runs in the default float32 with velocity
smoothing of order 1 (gamma = 1).

    python -u scripts/tutorials/5_li383_resistive.py
"""

# %%
from __future__ import annotations

import argparse
import os
import sys

# Run the cells top to bottom in a notebook / VS Code interactive window,
# or the whole file as a script (the CLI flags below still apply then).
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
ap.add_argument("--cuts", type=int, default=6)
ap.add_argument("--out", default="outputs/tutorials/li383_resistive")
cli = ap.parse_args([] if _INTERACTIVE else None)
ns = tuple(int(v) for v in cli.ns.split(","))
os.makedirs(cli.out, exist_ok=True)

# %%
import h5py
import jax.numpy as jnp
import matplotlib
if not _INTERACTIVE:
    matplotlib.use("Agg")  # headless as a script; a notebook keeps its inline backend
import matplotlib.pyplot as plt
import numpy as np
import mrx
from mrx.differential_forms import DiscreteFunction
from mrx.geometry import build_sequence
from mrx.gvec import load_clebsch
from mrx.initial_conditions import clebsch_potential_form, divergence_norm, potential_two_form
from mrx.nullspace import compute_nullspaces
from mrx.plotting import get_2d_grids, plot_torus, plot_twin_axis
from mrx.relaxation import (DescentMethod, TimeStepChoice, TimeStepper, compute_force,
                            relaxation_loop, weak_pressure)

print(f"[env] mrx precision {mrx.DTYPE}")

seq, ops = build_sequence(cli.geometry, ns, cli.p)
seq.set_operators(compute_nullspaces(seq, ops))

# %%
# --- the starting field: warm-start from Tutorial 3, or build the IC -----------
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
# --- one resistive reconnection step -------------------------------------------
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
# --- relax ideally for another 500 steps ---------------------------------------
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
# --- the weak pressure of the reconnected, relaxed field -----------------------
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
# --- the archive scripts/poincare_relax.py reads --------------------------
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
print(f"  -> {h5_path}  (trace the sections with:")
print(f"     python -u scripts/poincare_relax.py {h5_path} "
      f"--planes 0,0.125,0.25,0.375,0.5 --out {cli.out})")
