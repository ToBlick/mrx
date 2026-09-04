"""Tutorial 5: relaxation with finite resistivity on li383.

Tutorials 3 and 4 ran the *ideal* (eta = 0) descent: a frozen-in flow that
lowers the energy without ever changing the field's topology. Turn on a small
resistivity and that constraint breaks. Each step is now the ideal move
followed by a backward-Euler diffusion of ``B`` (an implicit resistive solve),
first order in ``dt``. Resistivity lets field lines **reconnect**: nested
surfaces can merge, seeded islands can heal or grow past their frozen-in width,
and -- unlike the ideal descent -- helicity is no longer exactly conserved, it
decays at the resistive rate. The relaxed state is a genuine finite-beta
equilibrium reached through reconnection, not a topological rearrangement of
the initial field.

The resistivity follows a **tanh schedule**: ``eta`` rises to ``--eta-max`` over
the first third of the run, holds, then drops back to ~0 over the last third,
so the tail relaxes ideally to a clean floor once reconnection has done its
work. Combine with ``--seed`` (the Tutorial 4 syntax) to watch a seeded island
reconnect instead of merely breathing: an ideal run freezes the chain, a
resistive one lets the resonant surface tear or heal.

The run uses the high-resolution reference ``wout_li383_1.4m.nc`` (``ns = 49``),
``(16, 32, 32) p = 2`` in float32, velocity smoothing of order 1 (gamma = 1),
and ``eta-max = 1e-4`` -- the settings of the ``li383_eta`` resistive sweep.

    MRX_DTYPE=float32 python -u scripts/tutorials/li383_resistive.py
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
ap.add_argument("--geometry", default="data/wout_li383_1.4m.nc",
                help="a VMEC wout (.nc) or a GVEC state file (.dat)")
ap.add_argument("--ns", default="16,32,32")
ap.add_argument("--p", type=int, default=2)
ap.add_argument("--precision", default="float32", choices=("float32", "float64"))
ap.add_argument("--eta-max", type=float, default=1e-4, help="peak resistivity (tanh schedule)")
ap.add_argument("--eta-every", type=int, default=1,
                help="resistive solve every K inner steps (raise it at larger eta)")
ap.add_argument("--seed", default="",
                help='optional resonant seed "m,n,rho0,width" (as in Tutorial 4)')
ap.add_argument("--seed-eps", type=float, default=0.0)
ap.add_argument("--outer", type=int, default=100, help="outer (recorded) iterations")
ap.add_argument("--inner", type=int, default=50, help="compiled steps per outer iteration")
ap.add_argument("--floor-tol", type=float, default=1e-4,
                help="stop once ||F|| falls below this (after eta has ramped down)")
ap.add_argument("--cuts", type=int, default=6)
ap.add_argument("--out", default="outputs/tutorials/li383_resistive")
cli = ap.parse_args([] if _INTERACTIVE else None)
ns = tuple(int(v) for v in cli.ns.split(","))
os.makedirs(cli.out, exist_ok=True)

# Precision is chosen from the environment before mrx is imported.
os.environ.setdefault("MRX_DTYPE", cli.precision)

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

if cli.precision != str(mrx.DTYPE):
    raise ValueError(f"--precision {cli.precision} but mrx runs in {mrx.DTYPE} "
                     "(MRX_DTYPE was already set)")
print(f"[env] mrx precision {mrx.DTYPE}")

seq, ops = build_sequence(cli.geometry, ns, cli.p)
seq.set_operators(compute_nullspaces(seq, ops))

# %%
# --- the initial condition: B = dA', optionally with a resonant seed -----------
cb = load_clebsch(cli.geometry)
seed = None
if cli.seed:
    m, n, rho0, width = (float(v) for v in cli.seed.split(","))
    seed = (int(m), int(n), rho0, width, cli.seed_eps)
    print(f"[ic] seed (m, n) = ({int(m)}, {int(n)}) at rho0 {rho0:g}, eps {cli.seed_eps:.2e}")
B0, norm, wall = potential_two_form(seq, clebsch_potential_form(cb, seed))
print(f"[ic] ||B||_M before normalisation {norm:.4e}, ||div B|| {divergence_norm(seq, B0):.2e}, "
      f"wall-normal part {wall:.1e}")

# %%
# --- the resistive descent -----------------------------------------------------
# tanh schedule: eta up over the first third, down to ~0 over the last third.
# relaxation_loop sets eta once per outer block; evaluate it at the block midpoint.
def eta_schedule(i):
    frac = (i - 0.5) / cli.outer
    return cli.eta_max * 0.5 * (1.0 - np.tanh(4.0 * np.pi * (frac - 0.5)))

smoothing_scale = 0.064 / ns[0] ** 2
print(f"[relax] velocity smoothing order 1, scale {smoothing_scale:.3e}; "
      f"eta-max {cli.eta_max:.1e} (tanh), eta-every {cli.eta_every}")
ts = TimeStepper(seq=seq, descent_method=DescentMethod.LBFGS,
                 dt_mode=TimeStepChoice.ANALYTIC_LINESEARCH, cfl=0.5,
                 eta_every=cli.eta_every, resistive=cli.eta_max > 0.0, history_size=1,
                 velocity_smoothing_order=1, velocity_smoothing_scale=smoothing_scale)
state, traces = relaxation_loop(B0, ts, num_iters_outer=cli.outer,
                                num_iters_inner=cli.inner, dt0=1.0,
                                force_tolerance=cli.floor_tol,
                                resistivity_schedule=eta_schedule)
F = np.asarray(traces["force_norm"], dtype=float)
E = np.asarray(traces["energy"], dtype=float)
H = np.asarray(traces["helicity"], dtype=float)
eta_tr = np.asarray(traces["eta"], dtype=float)
steps = np.asarray(traces["iteration"])
dH_rel = (H[-1] - H[0]) / (abs(H[0]) + 1e-30)
print(f"[relax] {steps[-1]} steps: ||F|| {F[0]:.3e} -> {F[-1]:.3e}, E_0 - E = {E[0] - E[-1]:.3e}, "
      f"peak eta {eta_tr.max():.2e}, ||div B|| {float(traces['divergence_B'][-1]):.1e}")
print(f"[relax] helicity H {H[0]:+.3e} -> {H[-1]:+.3e} (dH/H = {dH_rel:+.2e}): "
      "resistivity dissipates it -- this is reconnection, not the ideal descent's round-off.")
B = state.B_n

fig, _ = plot_twin_axis(F, E, left_label=r"$\|F\|_M$", right_label=r"$E$",
                        num_iters_inner=cli.inner, left_marker="o", right_marker="s")
path = os.path.join(cli.out, "trace.png")
fig.savefig(path, dpi=200)
if not _INTERACTIVE:
    plt.close(fig)  # keep the figure open in a notebook so it renders inline
print(f"  -> {path}")

# helicity against eta on twin axes: the drop tracks where eta is on.
fig, _ = plot_twin_axis(H, eta_tr, left_label=r"$H$", right_label=r"$\eta$",
                        num_iters_inner=cli.inner, left_marker="o", right_marker="s")
path = os.path.join(cli.out, "helicity_eta.png")
fig.savefig(path, dpi=200)
if not _INTERACTIVE:
    plt.close(fig)  # keep the figure open in a notebook so it renders inline
print(f"  -> {path}")

# %%
# --- the weak pressure of the relaxed field ------------------------------
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
if not _INTERACTIVE:
    plt.close(fig)  # keep the figure open in a notebook so it renders inline
print(f"  -> {path}")

# %%
# --- the archive scripts/poincare_relax.py reads --------------------------
h5_path = os.path.join(cli.out, "B.h5")
attrs = dict(geometry_path=os.path.abspath(cli.geometry), ns=list(ns), p=cli.p,
             nfp="", maxiter=10_000, precision=str(mrx.DTYPE), steps=int(steps[-1]),
             method="lbfgs", eta_max=cli.eta_max, ic="clebsch",
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
      f"--planes 0,0.25,0.5 --out {cli.out})")

