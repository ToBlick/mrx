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
pressure on the torus, and writes the run in ``scripts/relax.py``'s layout;
``scripts/poincare_relax.py`` then draws the Poincare sections of the initial
and relaxed fields at planes 0, 0.25, 0.5 (see the tutorials page).
``scripts/relax.py`` is the production driver (archives, QoIs, snapshots,
resistivity, seeds).

    python -u scripts/tutorials/3_li383_relaxation.py
"""

# %%
# Now we read the run's options and make the output folder. The defaults are
# li383 at (10, 16, 16) p=2, 500 relaxation steps.
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
ap.add_argument("--ns", default="10,16,16")
ap.add_argument("--p", type=int, default=2)
ap.add_argument("--outer", type=int, default=10, help="outer (recorded) iterations")
ap.add_argument("--inner", type=int, default=50, help="compiled steps per outer iteration")
ap.add_argument("--floor-tol", type=float, default=1e-3,
                help="stop once ||F|| falls below this")
ap.add_argument("--cuts", type=int, default=6)
ap.add_argument("--out", default="outputs/tutorials/li383_relaxation")
cli = ap.parse_args([] if _INTERACTIVE else None)
ns = tuple(int(v) for v in cli.ns.split(","))
os.makedirs(cli.out, exist_ok=True)


# %%
# Now we import MRX -- the sequence, the Clebsch initial condition, and the
# relaxation time-stepper and loop.
import json

import jax.numpy as jnp
import matplotlib
if not _INTERACTIVE:
    matplotlib.use("Agg")  # headless as a script; a notebook keeps its inline backend
import matplotlib.pyplot as plt
import numpy as np
import mrx
from mrx.differential_forms import DiscreteFunction
from mrx.geometry import build_sequence
from mrx.initial_conditions import initial_field
from mrx.nullspace import compute_nullspaces
from mrx.plotting import get_2d_grids, plot_torus, plot_twin_axis
from mrx.relaxation import (TimeStepper, compute_force, initial_state, relax, weak_pressure,
                            write_checkpoint)

print(f"[env] mrx precision {mrx.DTYPE}")

seq, ops = build_sequence(cli.geometry, ns, cli.p)
seq.set_operators(compute_nullspaces(seq, ops))

# %%
# Now we set the initial condition: li383's own equilibrium field as B = dA'
# from the histopolated Clebsch potential (divergence-free, wall-tangent, nested).
B0, ic = initial_field(seq, cli.geometry)
print(f"[ic] ||B||_M before normalisation {ic['B_norm_raw']:.4e}, ||div B|| {ic['div']:.2e}, "
      f"wall-normal part {ic['wall_discarded']:.1e}")

# %%
# Now we relax: the energy descent of mrx.relaxation with scripts/relax.py's
# defaults plus velocity smoothing, run toward a nested floor (~500 steps).
# gamma = 1 velocity smoothing: v = (I - scale L)^-1 F, scale ~ 0.064/n_r^2.
smoothing_scale = 0.064 / ns[0] ** 2
print(f"[relax] velocity smoothing order 1, scale {smoothing_scale:.3e}")
ts = TimeStepper(seq=seq, cfl=0.5, history_size=1,
                 velocity_smoothing_order=1, velocity_smoothing_scale=smoothing_scale)
res = relax(initial_state(B0, ts), ts, steps=cli.outer * cli.inner, chunk=cli.inner,
            floor_tol=cli.floor_tol)
F = np.asarray(res.trace["F"], dtype=float)
E = np.asarray(res.trace["E"], dtype=float)
H = np.asarray(res.qoi["helicity"], dtype=float)
print(f"[relax] {res.steps} steps ({res.stop}): ||F|| {F[0]:.3e} -> {F[-1]:.3e}, "
      f"E_0 - E = {E[0] - E[-1]:.3e}, H {H[0]:+.3e} -> {H[-1]:+.3e} (dH = {H[-1] - H[0]:+.1e}), "
      f"||div B|| {float(res.trace['div'][-1]):.1e}")
B = res.state.B_n

fig, _ = plot_twin_axis(F, E, left_label=r"$\|F\|_M$", right_label=r"$E$",
                        left_plot_kwargs=dict(marker=""), right_plot_kwargs=dict(marker=""))
path = os.path.join(cli.out, "trace.png")
fig.savefig(path, dpi=200)
if _INTERACTIVE:
    plt.show()
else:
    plt.close(fig)
print(f"  -> {path}")

# %%
# Now we compute the weak pressure -- the Lagrange multiplier the descent finds
# -- and draw it on the torus.
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
if _INTERACTIVE:
    plt.show()
else:
    plt.close(fig)
print(f"  -> {path}")

# %%
# Now we archive the run the way scripts/relax.py does -- relax.json with the
# parameters and the traces, and the initial and final state as checkpoints --
# so scripts/poincare_relax.py can draw the Poincare sections of both states.
os.makedirs(os.path.join(cli.out, "checkpoints"), exist_ok=True)
write_checkpoint(os.path.join(cli.out, "checkpoints", "state_000000.h5"), initial_state(B0, ts), 0)
write_checkpoint(os.path.join(cli.out, "checkpoints", f"state_{res.steps:06d}.h5"), res.state, res.steps)
params = dict(geometry_path=os.path.abspath(cli.geometry), ns=list(ns), p=cli.p, nfp=None,
              r_refine="", precision=str(mrx.DTYPE), steps=res.steps, scheme="explicit",
              auxiliary_B_field=False, ic=ic["kind"])
with open(os.path.join(cli.out, "relax.json"), "w") as fh:
    json.dump(dict(params=params, ic=ic, trace=res.trace, qoi=res.qoi, reconnect=[]), fh, indent=1)
print(f"  -> {cli.out}/relax.json and checkpoints/  (trace the sections with:")
print(f"     python -u scripts/poincare_relax.py {cli.out} --planes 0,0.25,0.5)")

