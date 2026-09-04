"""Tutorial 4: seeding a magnetic island in an ideal relaxation on li383.

Tutorial 3 relaxed li383's equilibrium field to a nested state. Here we add a
small **resonant perturbation** to the initial condition and watch what the
*ideal* (eta = 0) descent does with it. The energy descent is a frozen-in flow:
it moves the field along its own streamlines, so it cannot change the field's
topology. A seeded island can therefore only breathe -- shrink back if the
resonant surface is tearing-stable, or settle at an ``eps``-independent width
if it is tearing-unstable -- it cannot be reconnected away. Tutorial 5 turns on
resistivity and lets it reconnect.

The seed rides on the Clebsch potential (so ``B = dA'`` stays exactly
divergence-free and wall-tangent): a term
``eps |Phi'(rho0)| / m  g(rho) cos(2 pi (m theta - s n zeta))`` added to
``A'_zeta``, a Gaussian ``g`` of the given width centred on ``rho0`` and
tapered to zero at the wall. ``eps`` is the resonant normal field
``|dB^rho| / |B^zeta|`` at ``rho0``; the chain sits where
``|iota| = nfp n / m`` and the island it opens has full width about
``1.6 sqrt(eps nfp / (m |iota'|))`` in ``rho`` (a pendulum estimate).

The default seed ``(m, n) = (6, 1)`` lands on li383's ``iota = nfp n / m = 1/2``
surface (``rho ~ 0.54``); ``(5, 1)`` would take the ``3/5`` surface near the
edge. **The run uses the high-resolution reference** ``wout_li383_1.4m.nc``
(``ns = 49``): the coarse reference reconstructs the field to a relative
residual of 0.054, which sits on top of the seeded-island signal, so the seed
cannot be told from the reconstruction noise. The high-res file starts at 0.013
and the island stands clear of it. On ``(12, 24, 24) p = 3`` the ``eps = 1e-2``
chain reaches a full width near 0.15 in ``rho`` and is unmistakable in the
Poincare section. Vary ``--seed-eps`` (1e-3, 3e-3, 1e-2) to watch the width
track ``sqrt(eps)``.

It runs in float32, the production precision, with velocity smoothing of order
1 (gamma = 1) as in Tutorial 3.

    MRX_DTYPE=float32 python -u scripts/tutorials/li383_island_seed.py
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
                help="a VMEC wout (.nc) or a GVEC state file (.dat); "
                     "use the high-res reference so the seed clears the IC residual")
ap.add_argument("--ns", default="12,24,24")
ap.add_argument("--p", type=int, default=3)
ap.add_argument("--precision", default="float32", choices=("float32", "float64"))
ap.add_argument("--seed", default="6,1,0.544,0.1",
                help='resonant seed "m,n,rho0,width"; (6,1) is the iota=1/2 surface')
ap.add_argument("--seed-eps", type=float, default=1e-2,
                help="resonant normal field |dB^rho|/|B^zeta| at rho0; width ~ sqrt(eps)")
ap.add_argument("--outer", type=int, default=40, help="outer (recorded) iterations")
ap.add_argument("--inner", type=int, default=50, help="compiled steps per outer iteration")
ap.add_argument("--floor-tol", type=float, default=1e-3,
                help="stop once ||F|| falls below this")
ap.add_argument("--cuts", type=int, default=6)
ap.add_argument("--out", default="outputs/tutorials/li383_island_seed")
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
from mrx.initial_conditions import (clebsch_potential_form, divergence_norm,
                                    potential_two_form, resonant_rho)
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
# --- the initial condition: the equilibrium field with a resonant seed ---------
m, n, rho0, width = (float(v) for v in cli.seed.split(","))
seed = (int(m), int(n), rho0, width, cli.seed_eps)
cb = load_clebsch(cli.geometry)
rho_res = resonant_rho(cb, int(m), int(n))
print(f"[ic] seed (m, n) = ({int(m)}, {int(n)}) at rho0 {rho0:g}, width {width:g}, "
      f"eps {cli.seed_eps:.2e}")
print(f"[ic] the file's |iota| = nfp n / m = {cb['nfp'] * n / m:.4f} chain sits at "
      f"rho = {rho_res:.3f} (seed rho0 {rho0:g})")
B0, norm, wall = potential_two_form(seq, clebsch_potential_form(cb, seed))
print(f"[ic] ||B||_M before normalisation {norm:.4e}, ||div B|| {divergence_norm(seq, B0):.2e}, "
      f"wall-normal part {wall:.1e}")

# %%
# --- the descent with scripts/relax.py's defaults + velocity smoothing -----
# gamma = 1 velocity smoothing: v = (I - scale L)^-1 F, scale ~ 0.064/n_r^2.
smoothing_scale = 0.064 / ns[0] ** 2
print(f"[relax] velocity smoothing order 1, scale {smoothing_scale:.3e}, eta = 0 (ideal)")
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
print("[relax] the ideal descent conserves helicity (dH ~ round-off) and freezes the "
      "topology: the seeded island can only breathe, not reconnect.")
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
             method="lbfgs", eta_max=0.0, ic="clebsch",
             seed=cli.seed, seed_eps=cli.seed_eps)
with h5py.File(h5_path, "w") as fh:
    fh.create_dataset("B_ic", data=np.asarray(B0))
    fh.create_dataset("B_final", data=np.asarray(B))
    fh.create_dataset("pw_ic", data=pw_ic)
    fh.create_dataset("pw_final", data=pw_final)
    for k, v in attrs.items():
        fh.attrs[k] = v
print(f"  -> {h5_path}  (trace the sections -- the island shows at the {int(m)}/{int(n)} chain:")
print(f"     python -u scripts/poincare_relax.py {h5_path} "
      f"--planes 0,0.25,0.5 --out {cli.out})")

