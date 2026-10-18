"""Tutorial 5: drive field with an external current

Tutorials 3 and 4 stayed ideal (eta = 0). The flow is frozen into the field and lowers the
energy without ever changing the topology of the field, so the islands that Tutorial 4
opened survive the relaxation. 

A drive breaks that constraint. After every ideal step, a backward-Euler resistive step 
``dB/dt = -eta curl (J - J*)`` of size ``eps = C h_r^2`` (``h_r`` the radial cell size)
pulls the current towards ``J*``, the current of a reference field ``B*``.

Here the reference is the unseeded nested state of Tutorial 3. Field lines can now reconnect and the
helicity is no longer conserved. A field with the current ``J*`` differs from ``B*`` by a harmonic field
of zero flux, which is zero, so the drive pulls the seeded field towards the reference. On this coarse
mesh the reference has chains of its own (a wide one at iota = 1/2), and the driven field takes on the
reference's chains instead of the seeded ones. This is
``scripts/relax.py --drive.resistivity C --drive.reference REF --drive.reference-smoothing 0``.

The run starts from the seeded, relaxed state of Tutorial 4 (``outputs/tutorials/4_li383_island_seed``,
whose ``reference.h5`` is the unseeded floor), so run Tutorials 3 and 4 first. 
It then takes ``--budget.steps`` Newton steps with the drive on. The script
prints the step size, and after every chunk the force residual, the helicity and the distance to the 
reference. Before and after the drive it prints the island width of every seeded chain from the sections (the
radial extent of the lines locked to its rotational transform). The sections trace ``--lines`` field lines
(160 by default, the paper's li383 figures trace 320 for 600 periods) for ``--periods`` field periods (400 by default).

    python -u scripts/tutorials/5_li383_drive.py
    python -u scripts/tutorials/5_li383_drive.py --resistivity 0.128 --budget.steps 20
"""

# %%
# 1) Setup.
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

print("[tutorial 5] the resistive drive on li383: importing JAX and MRX", flush=True)

import tyro

from mrx.relaxation.config import Budget, Geometry

_INTERACTIVE = "ipykernel" in sys.modules

@tyro.conf.configure(tyro.conf.EnumChoicesFromValues)
@dataclass(frozen=True)
class Options:
    """Tutorial 5: drive the seeded field of Tutorial 4 towards the current of the unseeded floor, on li383."""
    geometry: Geometry = Geometry(path="data/wout_li383_low_res_reference.nc", resolution=(12, 16, 16),
                                  spline_degree=2)
    seeded: str = "outputs/tutorials/4_li383_island_seed"
    """The run directory of Tutorial 4 (with checkpoints/ and reference.h5)."""
    resistivity: float = 0.064
    """The factor C of the resistive step size eps = C h_r^2 (0.064 as in the paper)."""
    budget: Budget = Budget(steps=10, chunk=5, floor_tol=0.0)
    """The step budget of the driven Newton run."""
    lines: int = 160
    """The number of field lines in the Poincare sections."""
    periods: int = 400
    """The number of field periods each line is traced."""
    out: str = "outputs/tutorials/5_li383_drive"
    """The directory of the run."""


cli = tyro.cli(Options, args=[] if _INTERACTIVE else None)
os.makedirs(cli.out, exist_ok=True)

# %%
# 2) Import MRX.
import glob
import json

import numpy as np
from mrx.nullspace import compute_nullspaces
from mrx.diagnostics.plotting import plot_twin_axis, trace_and_plot
from mrx.diagnostics.plotstyle import finish_figure
from mrx.diagnostics.poincare import locked_width
from mrx.relaxation.config import Drive, RelaxConfig
from mrx.relaxation.loop import initial_state, read_field, relax, save_run, write_checkpoint


geometry = cli.geometry
mesh = f"{geometry.resolution} p={geometry.spline_degree}"
print("[seq] building the de Rham sequence: the map, the operators and the preconditioners", flush=True)
seq = geometry.build()
compute_nullspaces(seq)
nfp = seq.nfp

# %%
# 3) Get the seeded start and the unseeded reference from the run of Tutorial 4: its last checkpoint and
# reference.h5, on the same mesh.
ref = os.path.join(cli.seeded, "reference.h5")
ckpts = sorted(glob.glob(os.path.join(cli.seeded, "checkpoints", "state_*.h5")))
if not (ckpts and os.path.exists(ref)):
    sys.exit(f"no run of Tutorial 4 in {cli.seeded}: run Tutorial 4 first, or point --seeded at its run")
B_seeded, _ = read_field(ckpts[-1], seq)
B_floor, _ = read_field(ref, seq)
with open(os.path.join(cli.seeded, "relax.json")) as fh:
    rows = json.load(fh)["seed"]
print(f"[start] the seeded state {ckpts[-1]} and the reference {ref}")
for r in rows:
    print(f"[start] seeded chain ({r['m']},{r['n']}) iota {nfp * r['n'] / r['m']:.4f} at r {r['r']:.3f}: "
          f"amplitude {r['dBr']:+.3e}, pendulum width {r['w']:.4f}")
print(f"[start] ||B_seeded - B*|| / ||B*|| = "
      f"{float(seq.odd.l2_norm(B_seeded - B_floor, 2) / seq.odd.l2_norm(B_floor, 2)):.3e}")

# %%
# 4) Set up the drive as scripts/relax.py does: the Drive group of the
# configuration, with the unseeded floor written as a checkpoint for its reference,
# and the stepper with the resistivity.
ref_path = os.path.join(cli.out, "reference.h5")
cfg = RelaxConfig(geometry=geometry, budget=cli.budget,
                  drive=Drive(resistivity=cli.resistivity, reference=ref_path,
                              reference_smoothing=0.0))
ts = cfg.stepper(seq)
write_checkpoint(ref_path, initial_state(B_floor, ts), 0, seq)
B_star = B_floor
ts = ts.with_reference(B_star)
print(f"[drive] eps = {cfg.drive.resistivity:g} h_r^2 = {float(ts.resistivity):.3e} per step towards the current of "
      f"B* = the unseeded floor")

def progress(res):
    q = res.qoi
    dist = float(seq.odd.l2_norm(res.state.B_n - B_star, 2) / seq.odd.l2_norm(B_star, 2))
    print(f"[drive] step {res.steps:4d}: ||F|| {res.trace['F'][-1]:.3e}  resid {res.trace['resid'][-1]:.3e}  "
          f"helicity {q['helicity'][-1]:+.6e} ({(q['helicity'][-1] - q['helicity'][0]) / q['helicity'][0]:+.2e} of "
          f"the start)  ||B - B*|| / ||B*|| {dist:.3e}")


# %%
# 5) Run the resistive relaxation with external drive. The distance to the reference shrinks as
# the field heads for the reference and takes on its chains.
res = relax(initial_state(B_seeded, ts), ts, on_chunk=progress, **cfg.relax_kwargs())
B = res.state.B_n
F = np.asarray(res.trace["F"], dtype=float)
H = np.asarray(res.qoi["helicity"], dtype=float)
print(f"[drive] {res.steps} steps ({res.stop}): ||F|| {F[0]:.3e} -> {F[-1]:.3e}, "
      f"H {H[0]:+.6e} -> {H[-1]:+.6e} ({(H[-1] - H[0]) / H[0]:+.2e}), "
      f"||B - B*|| / ||B*|| {float(seq.odd.l2_norm(B_seeded - B_star, 2) / seq.odd.l2_norm(B_star, 2)):.3e} -> "
      f"{float(seq.odd.l2_norm(B - B_star, 2) / seq.odd.l2_norm(B_star, 2)):.3e}")

# %%
# 6) Plot the force residual and the helicity over the run.
fig, _ = plot_twin_axis(F, H, x_right=np.asarray(res.qoi["it"], dtype=float),
                        left_label=r"$\|F\|_M$", right_label=r"$H$",
                        left_plot_kwargs=dict(marker=""), right_plot_kwargs=dict(marker="o"))
finish_figure(fig, os.path.join(cli.out, "trace.png"))

# %%
# 7) Draw Poincare sections of the field before and after, as scripts/poincare_plot.py does.
def chain_widths(r, what):
    for row in rows:
        target = nfp * row["n"] / row["m"]
        width, n_locked = locked_width(r, target)
        print(f"[{what}] chain ({row['m']},{row['n']}) iota {target:.4f}: {n_locked} locked line(s), "
              f"width {width:.4f}")


results = trace_and_plot(seq, {"before": (B_seeded, 0), "after": (B, res.steps)}, cli.out, lines=cli.lines,
                         periods=cli.periods, source=f"{geometry.path} {mesh}")
chain_widths(results["before"], "before")
chain_widths(results["after"], "after")

# %%
# 8) Save the run the way scripts/relax.py does: relax.json with the configuration of the drive and 
# the checkpoints of the start and the end. 
# Then scripts/poincare_trace.py --geometry ... checkpoints/state_*.h5 can trace it if needed.
save_run(cli.out, cfg, seq, initial_state(B_seeded, ts), res, "seeded", seed=rows)
print("[done] the drive takes the seeded field towards the reference. The helicity changes, the distance to "
      "the reference shrinks and the chains become those of the reference.")
