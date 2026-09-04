"""Tutorial 3: a relaxation run on the li383 (NCSX) equilibrium.

li383 is the project's fruit-fly stellarator: a three-field-period
(``nfp=3``) NCSX configuration with a rotational transform sweeping
``iota ~ 0.40 -> 0.66`` and a genuine pressure. The initial condition is the
state's own equilibrium field, ``B = dA'`` from the histopolated Clebsch
potential (exactly divergence-free, tangential to the wall, nested
surfaces). The relaxation is the energy descent of ``mrx.relaxation`` with
``scripts/relax.py``'s production defaults: L-BFGS direction with history 1
(equivalent to conjugate gradient), analytic line search under a CFL cap of
0.5, no resistivity. It conserves helicity and lowers the magnetic energy
until ``J x B = grad p`` in the weak sense; ``p`` is not prescribed, it is
the Lagrange multiplier the descent finds.

This run turns on **velocity smoothing** of order 1 (gamma = 1): the descent
direction is ``(I - scale L)^-1 F`` with ``scale ~ 0.064 / n_r^2``. On li383
this reaches a clean nested floor in ~1000 steps where the unsmoothed descent
(gamma = 0) grinds for ~6000; the force residual need not fall monotonically,
what matters is the floor it settles at. It runs in float32 -- the descent is
robust there and it is the production precision.

The script prints the traces, draws ``||F||`` against the energy and the weak
pressure on the torus, and writes a ``B.h5`` in ``scripts/relax.py``'s format;
``scripts/poincare_relax.py`` then draws the Poincare sections of the initial
and relaxed fields at planes 0, 0.25, 0.5 (see the tutorials page).
``scripts/relax.py`` is the production driver (archives, QoIs, snapshots,
resistivity, seeds).

    MRX_DTYPE=float32 python -u scripts/tutorials/li383_relaxation.py
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
                help="a VMEC wout (.nc) or a GVEC state file (.dat)")
ap.add_argument("--ns", default="12,24,12")
ap.add_argument("--p", type=int, default=3)
ap.add_argument("--precision", default="float32", choices=("float32", "float64"))
ap.add_argument("--outer", type=int, default=20, help="outer (recorded) iterations")
ap.add_argument("--inner", type=int, default=50, help="compiled steps per outer iteration")
ap.add_argument("--floor-tol", type=float, default=1e-3,
                help="stop once ||F|| falls below this")
ap.add_argument("--cuts", type=int, default=6)
ap.add_argument("--out", default="outputs/tutorials/li383_relaxation")
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
# --- the initial condition: the equilibrium field as B = dA' -------------------
cb = load_clebsch(cli.geometry)
B0, norm, wall = potential_two_form(seq, clebsch_potential_form(cb))
print(f"[ic] ||B||_M before normalisation {norm:.4e}, ||div B|| {divergence_norm(seq, B0):.2e}, "
      f"wall-normal part {wall:.1e}")

# %%
# --- the descent with scripts/relax.py's defaults + velocity smoothing -----
# gamma = 1 velocity smoothing: v = (I - scale L)^-1 F, scale ~ 0.064/n_r^2.
smoothing_scale = 0.064 / ns[0] ** 2
print(f"[relax] velocity smoothing order 1, scale {smoothing_scale:.3e}")
ts = TimeStepper(seq=seq, descent_method=DescentMethod.LBFGS,
                 dt_mode=TimeStepChoice.ANALYTIC_LINESEARCH, cfl=0.5,
                 eta_every=1, resistive=False, history_size=1,
                 velocity_smoothing_order=1, velocity_smoothing_scale=smoothing_scale)
state, traces = relaxation_loop(B0, ts, num_iters_outer=cli.outer,
                                num_iters_inner=cli.inner, dt0=1.0,
                                force_tolerance=cli.floor_tol)
F = np.asarray(traces["force_norm"], dtype=float)
E = np.asarray(traces["energy"], dtype=float)
H = np.asarray(traces["helicity"], dtype=float)
steps = np.asarray(traces["iteration"])
print(f"[relax] {steps[-1]} steps: ||F|| {F[0]:.3e} -> {F[-1]:.3e}, E_0 - E = {E[0] - E[-1]:.3e}, "
      f"H {H[0]:+.3e} -> {H[-1]:+.3e} (dH = {H[-1] - H[0]:+.1e}), "
      f"||div B|| {float(traces['divergence_B'][-1]):.1e}")
B = state.B_n

fig, _ = plot_twin_axis(F, E, left_label=r"$\|F\|_M$", right_label=r"$E$",
                        num_iters_inner=cli.inner, left_marker="o", right_marker="s")
path = os.path.join(cli.out, "trace.png")
fig.savefig(path, dpi=200)
if not _INTERACTIVE:
    plt.close(fig)  # keep the figure open in a notebook so it renders inline
print(f"  -> {path}")

# %%
# --- the weak pressure of the initial and relaxed fields -----------------
def weak_p(field):
    _, _, J, Hf, _ = compute_force(field, seq)
    p_w, _, _ = weak_pressure(J, Hf, seq)
    return np.asarray(p_w)

pw_ic = weak_p(B0)
pw_final = weak_p(B)
pw = DiscreteFunction(jnp.asarray(pw_final), seq.basis_0, seq.E(0, True))

def p_h(x):
    return pw(x)[0]

print(f"[relax] weak pressure on the axis {float(p_h(jnp.array([0.0, 0.0, 0.0]))):.4e} "
      f"(||B||_M = 1 units)")
zetas = np.arange(cli.cuts) / cli.cuts
n = 48
grids_pol = [get_2d_grids(seq.map, cut_axis=2, cut_value=float(z), nx=n, ny=n, nz=1)
             for z in zetas]
grid_surface = get_2d_grids(seq.map, cut_axis=0, cut_value=1.0 - 1e-6,
                            ny=4 * n, nz=4 * n, invert_z=True)
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
             method="lbfgs", eta_max=0.0, ic="clebsch")
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

