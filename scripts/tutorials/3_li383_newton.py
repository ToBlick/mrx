"""Tutorial 3: Magnetic relaxation of the li383 (NCSX) equilibrium.

We take a VMEC equilibrium, load it into MRX, and relax it to a chosen tolerance.

The relaxation is done with Newton-MR, a preconditioned MINRES solve of Newton's equations.

The tutorial takes ``--budget.steps`` Newton steps (20 by default) from the Clebsch initial field on the
``(12, 16, 16) p = 2`` mesh in mixed precision. It prints the traces, draws ``||F||`` and the energy
released against the step, and writes the run in the layout of ``scripts/relax.py``. Tutorial 4 can re-use
the result.

This tutorial mimicks ``scripts/relax.py`` and hence takes the same CLI arguments, see the documentation 
therein if you want to play with the hyperparameters.

    python -u scripts/tutorials/3_li383_newton.py
    python -u scripts/tutorials/3_li383_newton.py --budget.steps 5 --budget.chunk 5 --newton.maxiter 400
"""

# %%
# 1) Setup. The defaults are li383 at (12, 16, 16) p=2 and 20 Newton steps in chunks of 5.
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

print("[tutorial 3] Newton relaxation of li383: importing JAX and MRX", flush=True)

import tyro

from mrx.precision import current_precision
from mrx.relaxation.config import Budget, Geometry, Newton, Precision

_INTERACTIVE = "ipykernel" in sys.modules

@tyro.conf.configure(tyro.conf.EnumChoicesFromValues)
@dataclass(frozen=True)
class Options:
    """Tutorial 3: Newton steps from the equilibrium field of li383."""
    geometry: Geometry = Geometry(path="data/wout_li383_low_res_reference.nc", resolution=(12, 16, 16),
                                  spline_degree=2, precision=Precision(current_precision()))
    newton: Newton = Newton()
    budget: Budget = Budget(steps=20, chunk=5, floor_tol=0.0)
    """The step budget of the Newton run."""
    out: str = "outputs/tutorials/3_li383_newton"
    """The directory of the Newton run."""


cli = tyro.cli(Options, args=[] if _INTERACTIVE else None)
# mrx fixes its precision when it is imported, from MRX_DTYPE and MRX_RESIDUAL_DTYPE
if cli.geometry.precision != current_precision():
    sys.exit(f"--geometry.precision {cli.geometry.precision} needs MRX_DTYPE and MRX_RESIDUAL_DTYPE set before "
             f"mrx is imported (this run is {current_precision()})")
os.makedirs(cli.out, exist_ok=True)

# %%
# 2) Import MRX
import json

import matplotlib
if not _INTERACTIVE:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mrx.relaxation.initial_conditions import initial_field
from mrx.nullspace import compute_nullspaces
from mrx.diagnostics.plotting import plot_twin_axis
from mrx.relaxation.config import RelaxConfig
from mrx.relaxation.loop import initial_state, radial_cell_sq, relax, write_checkpoint

print(f"[env] mrx precision {current_precision()}")

# The configuration is the same object that scripts/relax.py builds from its command line
# (mrx.relaxation.config).
cfg = RelaxConfig(geometry=cli.geometry, newton=cli.newton, budget=cli.budget)
geometry = cfg.geometry
print("[seq] building the de Rham sequence: the map, the operators and the preconditioners", flush=True)
seq, ops = geometry.build()
compute_nullspaces(seq)
h_r_sq = radial_cell_sq(seq)

# %%
# 3) Set the initial condition: the equilibrium field of li383 as B = dA' from the Clebsch potential.
B0, ic = initial_field(seq)
print(f"[ic] ||B||_M before normalisation {ic['B_norm_raw']:.4e}, ||div B|| {ic['div']:.2e}, "
      f"wall-normal part {ic['wall_discarded']:.1e}")

# %%
# 4) Run Newton.
time_stepper = cfg.stepper(seq)
res = relax(initial_state(B0, time_stepper), time_stepper, **cfg.relax_kwargs()) 

# Get the diagnostics:
F = np.asarray(res.trace["F"], dtype=float)
dE = np.asarray(res.trace["dE"], dtype=float)
H = np.asarray(res.qoi["helicity"], dtype=float)
it_n = np.asarray(res.trace["newton_it"])
dt_n = np.asarray(res.trace["dt_star"], dtype=float)
print(f"[newton] {res.steps} steps in {res.wall:.0f} s ({res.wall / res.steps:.1f} s/step): "
      f"||F|| {F[0]:.3e} -> {F[-1]:.3e} (lowest {F.min():.3e} at step {F.argmin() + 1}), "
      f"E_0 - E = {-dE.sum():.3e}, dH/H_0 = {(H[-1] - H[0]) / H[0]:+.1e}, "
      f"||div B|| {float(res.trace['div'][-1]):.1e}")
print(f"[newton] MINRES iterations mean {np.abs(it_n).mean():.0f}, at the budget on "
      f"{int((it_n < 0).sum())}/{res.steps} steps, dt* mean {dt_n.mean():.2f} "
      f"(1 is the Newton step)")

# %%
# 5) Plots. Draw ||F|| and the energy released against the step.
fig, _ = plot_twin_axis(F, np.cumsum(-dE), left_label=r"$\|F\|_M$", right_label=r"$E_0 - E$",
                        left_plot_kwargs=dict(marker="o"), right_plot_kwargs=dict(marker=""))
path = os.path.join(cli.out, "trace.png")
fig.savefig(path, dpi=200)
if _INTERACTIVE:
    plt.show()
else:
    plt.close(fig)
print(f"  -> {path}")

# %%
# 6) Write a checkpoint of the final field the way scripts/relax.py does, with relax.json and
# the checkpoints of the start and the end. Tutorial 4 can then start from the Newton state, and 
# scripts/poincare_trace.py can trace it.
os.makedirs(os.path.join(cli.out, "checkpoints"), exist_ok=True)
write_checkpoint(os.path.join(cli.out, "checkpoints", "state_000000.h5"), initial_state(B0, time_stepper), 0, seq)
write_checkpoint(os.path.join(cli.out, "checkpoints", f"state_{res.steps:06d}.h5"), res.state, res.steps, seq)
params = dict(cfg.params, geometry_path=os.path.abspath(geometry.path), knots=geometry.knots, ic=ic["kind"],
              h_r_sq=float(h_r_sq), start_step=0)
with open(os.path.join(cli.out, "relax.json"), "w") as fh:
    json.dump(dict(params=params, ic=ic, trace=res.trace, qoi=res.qoi), fh, indent=1)
print(f"  -> {cli.out}/relax.json and checkpoints/  (trace and draw the sections with:")
print(f"     python -u scripts/poincare_trace.py --geometry {geometry.path} {cli.out}/checkpoints/state_*.h5")
print(f"     python scripts/poincare_plot.py {cli.out})")
