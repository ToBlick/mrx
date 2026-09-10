"""Tutorial 4: Newton's method on the second variation, on li383.

Tutorial 3's descent goes to its floor on a power law: the force residual
falls as a power of the step, never a plateau, and the tail is slow. The
directions the energy is flat along -- surfaces sliding past each other,
current sheets thinning -- crawl, because a gradient method scales every mode
by the same time step and the stiffest modes set it. Newton scales each mode
by the inverse of its own curvature.

The energy along the flow of a divergence-free velocity ``u`` expands to
second order: its gradient is minus the Lorentz force, its Hessian ``H`` is
the second variation (``2 delta W`` of ideal MHD at ``p = 0``; at an
equilibrium, minus the linearised force operator). Newton's equation
``H u = J x B`` is solved in the **potential form**, ``u = curl a`` with
``curl^T H curl a = curl^T (J x B)`` -- divergence-free by construction, no
Leray solve -- by MINRES with the k=1 Laplacian atom as preconditioner,
**truncated**: 300 iterations toward a relative residual of 0.1, warm-started
from the previous step's potential. The truncation is the trust region; a
fully converged direction overshoots. The rest of the step is Tutorial 3's:
the analytic line search along the direction (``dt* ~ 1`` for a Newton
direction; the cap ``newton_dt_cap = 1`` is the Newton step), the CFL cap,
and the update is a curl, so ``div B`` and the helicity stay exact. A
direction that is not a descent direction is replaced by the smoothed force
for that step (``newton_fallback`` in the trace).

Newton is the floor finder. From a state the descent has taken through its
fast phase it reaches the mesh's residual floor in tens of steps where the
descent needs thousands, at 15-30 descent steps per Newton step. The floor it
finds depends on the route -- which corner of the orbit the descent left it
in -- so the rule is: descent through its fast phase, then Newton. The
paper's floors are at sixteen radial cells and more; on the tutorial's ten the
direction is good for a handful of steps -- the residual drops fivefold in the
first five, then turns around and the helicity starts to leak (under-resolved
radial structure at the surfaces, the study's section 10e) -- and that is the
budget here: ten steps, a minute or two on a GPU.

This tutorial warm-starts from Tutorial 3's floor -- the run in
``outputs/tutorials/li383_relaxation`` or the shipped state in
``data/tutorials/li383_relaxation`` -- or, without either, runs that descent
itself. From that state it runs both continuations: the smoothed
descent for another ``--descent-steps`` and Newton for ``--newton-steps``,
and draws ``||F||`` against the step and against the wall time for both. It
writes the Newton run in ``scripts/relax.py``'s layout, so Tutorial 6 can
warm-start from it and ``scripts/poincare_trace.py`` can section it.
``scripts/relax.py`` runs the same from the command line: Newton is its
default method.

    python -u scripts/tutorials/4_li383_newton.py
"""

# %%
# Now we read the run's options. The defaults are Tutorial 3's mesh, li383 at
# (10, 16, 16) p=2, 200 more descent steps against 10 Newton steps.
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
ap.add_argument("--warm-start", default="outputs/tutorials/li383_relaxation,data/tutorials/li383_relaxation",
                help="run directories, first present wins: Tutorial 3's run, then its shipped state")
ap.add_argument("--descent-steps", type=int, default=200,
                help="steps of the smoothed descent continued from the warm start, for comparison")
ap.add_argument("--newton-steps", type=int, default=10, help="Newton steps from the warm start")
ap.add_argument("--newton-chunk", type=int, default=5, help="compiled Newton steps per chunk")
ap.add_argument("--newton-tol", type=float, default=0.1,
                help="relative residual the truncated MINRES solve aims at")
ap.add_argument("--newton-maxiter", type=int, default=300, help="its iteration budget per step")
ap.add_argument("--out", default="outputs/tutorials/li383_newton")
cli = ap.parse_args([] if _INTERACTIVE else None)
ns = tuple(int(v) for v in cli.ns.split(","))
os.makedirs(cli.out, exist_ok=True)

# %%
# Now we import MRX -- the sequence, the Clebsch initial condition, and the
# relaxation time-stepper and loop; the Newton direction is a stepper option.
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
from mrx.geometry import build_sequence
from mrx.initial_conditions import initial_field
from mrx.nullspace import compute_nullspaces
from mrx.relaxation import TimeStepper, initial_state, relax, write_checkpoint

print(f"[env] mrx precision {mrx.DTYPE}")

seq, ops = build_sequence(cli.geometry, ns, cli.p)
compute_nullspaces(seq)

# %%
# Now we get the starting field: Tutorial 3's relaxed B from the first run
# directory whose checkpoint is on disk and matches this mesh (the user's run,
# then the shipped state), otherwise the equilibrium initial condition taken
# through the descent's fast phase here (Tutorial 3's run, 500 steps).
ts_descent = TimeStepper(seq=seq, cfl=0.5, history_size=1, velocity_smoothing_order=1)
B_start = None
for run in cli.warm_start.split(","):
    ws_json = os.path.join(run, "relax.json")
    if not os.path.exists(ws_json):
        continue
    with open(ws_json) as fh:
        ws = json.load(fh)["params"]
    ckpts = sorted(glob.glob(os.path.join(run, "checkpoints", "state_*.h5")))
    if tuple(ws["ns"]) == ns and int(ws["p"]) == cli.p and ckpts:
        with h5py.File(ckpts[-1], "r") as fh:
            B_start = jnp.asarray(np.asarray(fh["B_n"]))
        print(f"[ic] warm-started from Tutorial 3: {ckpts[-1]} (ns={ws['ns']} p={ws['p']})")
        break
    print(f"[ic] run {run} is ns={ws['ns']} p={ws['p']} (need {list(ns)} p={cli.p}); skipped")
if B_start is None:
    B0, ic = initial_field(seq)
    print(f"[ic] built the equilibrium IC: ||B||_M {ic['B_norm_raw']:.4e}, "
          f"||div B|| {ic['div']:.2e}, wall-normal {ic['wall_discarded']:.1e}")
    fast = relax(initial_state(B0, ts_descent), ts_descent, steps=500, chunk=50, floor_tol=1e-3)
    B_start = fast.state.B_n
    print(f"[ic] the descent's fast phase: {fast.steps} steps ({fast.stop}), "
          f"||F|| {fast.trace['F'][0]:.3e} -> {fast.trace['F'][-1]:.3e}")

# %%
# Now we continue the smoothed descent from that state, for comparison: the
# power-law tail of Tutorial 3, another 200 steps.
res_d = relax(initial_state(B_start, ts_descent), ts_descent, steps=cli.descent_steps,
              chunk=50, floor_tol=0.0)
F_d = np.asarray(res_d.trace["F"], dtype=float)
H_d = np.asarray(res_d.qoi["helicity"], dtype=float)
print(f"[descent] {res_d.steps} steps in {res_d.wall:.0f} s ({res_d.wall / res_d.steps:.2f} s/step): "
      f"||F|| {F_d[0]:.3e} -> {F_d[-1]:.3e} (lowest {F_d.min():.3e}), "
      f"dH/H_0 = {(H_d[-1] - H_d[0]) / H_d[0]:+.1e}")

# %%
# Now we run Newton from the same state: history 0 (the direction replaces
# L-BFGS), the truncated MINRES solve with the Laplacian atom, the line search
# capped at the Newton step. The smoothing stays on for the fallback direction.
ts_newton = TimeStepper(seq=seq, cfl=0.5, history_size=0, velocity_smoothing_order=1,
                        newton=True, newton_tol=cli.newton_tol, newton_maxiter=cli.newton_maxiter,
                        newton_precond="laplacian", newton_dt_cap=1.0)
res_n = relax(initial_state(B_start, ts_newton), ts_newton, steps=cli.newton_steps,
              chunk=cli.newton_chunk, floor_tol=0.0)
F_n = np.asarray(res_n.trace["F"], dtype=float)
H_n = np.asarray(res_n.qoi["helicity"], dtype=float)
it_n = np.asarray(res_n.trace["newton_it"])
dt_n = np.asarray(res_n.trace["dt_star"], dtype=float)
print(f"[newton] {res_n.steps} steps in {res_n.wall:.0f} s ({res_n.wall / res_n.steps:.1f} s/step): "
      f"||F|| {F_n[0]:.3e} -> {F_n[-1]:.3e} (lowest {F_n.min():.3e} at step {F_n.argmin() + 1}), "
      f"dH/H_0 = {(H_n[-1] - H_n[0]) / H_n[0]:+.1e}")
print(f"[newton] MINRES iterations mean {np.abs(it_n).mean():.0f}, at the budget on "
      f"{int((it_n > 0).sum())}/{res_n.steps} steps; fallbacks to the smoothed force "
      f"{int(np.asarray(res_n.trace['newton_fallback']).sum())}; dt* mean {dt_n.mean():.2f} "
      f"(1 is the Newton step)")
print(f"[newton] one Newton step costs {res_n.wall / res_n.steps / (res_d.wall / res_d.steps):.0f} "
      f"descent steps; the descent's lowest ||F|| in {res_d.steps} steps against Newton's in "
      f"{res_n.steps}: {F_d.min():.2e} vs {F_n.min():.2e}")

# %%
# Now we draw ||F|| against the step and against the wall time for both
# continuations (the wall time per step taken as the run's mean).
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
for ax, x_d, x_n, xlabel in ((axes[0], np.arange(1, res_d.steps + 1), np.arange(1, res_n.steps + 1), "step"),
                             (axes[1], np.arange(1, res_d.steps + 1) * res_d.wall / res_d.steps,
                              np.arange(1, res_n.steps + 1) * res_n.wall / res_n.steps, "wall time [s]")):
    ax.semilogy(x_d, F_d, color="0.5", label=f"L-BFGS(1), smoothed ({res_d.steps} steps)")
    ax.semilogy(x_n, F_n, color="black", label=f"Newton ({res_n.steps} steps)")
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.3)
axes[0].set_ylabel(r"$\|F\|_M$")
axes[0].legend(frameon=False)
path = os.path.join(cli.out, "newton_vs_descent.png")
fig.savefig(path, dpi=200)
if _INTERACTIVE:
    plt.show()
else:
    plt.close(fig)
print(f"  -> {path}")

# %%
# Now we archive the Newton run the way scripts/relax.py does -- relax.json and
# the checkpoints of the start and the end -- so Tutorial 6 can warm-start from
# the Newton floor and scripts/poincare_trace.py can section it.
os.makedirs(os.path.join(cli.out, "checkpoints"), exist_ok=True)
write_checkpoint(os.path.join(cli.out, "checkpoints", "state_000000.h5"), initial_state(B_start, ts_newton), 0)
write_checkpoint(os.path.join(cli.out, "checkpoints", f"state_{res_n.steps:06d}.h5"), res_n.state, res_n.steps)
params = dict(geometry_path=os.path.abspath(cli.geometry), ns=list(ns), p=cli.p, nfp=None,
              knots=None, precision=str(mrx.DTYPE), steps=res_n.steps, scheme="explicit",
              auxiliary_B_field=False, ic="warmstart", newton=True, newton_tol=cli.newton_tol,
              newton_maxiter=cli.newton_maxiter, newton_precond="laplacian", newton_dt_cap=1.0)
with open(os.path.join(cli.out, "relax.json"), "w") as fh:
    json.dump(dict(params=params, trace=res_n.trace, qoi=res_n.qoi, reconnect=[]), fh, indent=1)
print(f"  -> {cli.out}/relax.json and checkpoints/  (trace and draw the sections with:")
print(f"     python -u scripts/poincare_trace.py --run {cli.out}")
print(f"     python scripts/poincare_plot.py {cli.out})")
