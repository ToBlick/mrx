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
whose ``reference.h5`` is the unseeded floor), or builds the same here when that run is absent. 
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

from mrx.precision import current_precision
from mrx.relaxation.config import Budget, Geometry, Precision

_INTERACTIVE = "ipykernel" in sys.modules

@tyro.conf.configure(tyro.conf.EnumChoicesFromValues)
@dataclass(frozen=True)
class Options:
    """Tutorial 5: drive the seeded field of Tutorial 4 towards the current of the unseeded floor, on li383."""
    geometry: Geometry = Geometry(path="data/wout_li383_low_res_reference.nc", resolution=(12, 16, 16),
                                  spline_degree=2, precision=Precision(current_precision()))
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
# mrx fixes its precision when it is imported, from MRX_DTYPE and MRX_RESIDUAL_DTYPE
if cli.geometry.precision != current_precision():
    sys.exit(f"--geometry.precision {cli.geometry.precision} needs MRX_DTYPE and MRX_RESIDUAL_DTYPE set before "
             f"mrx is imported (this run is {current_precision()})")
os.makedirs(cli.out, exist_ok=True)

# %%
# 2) Import MRX.
import glob
import json

import equinox as eqx
import h5py
import jax.numpy as jnp
import matplotlib
if not _INTERACTIVE:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mrx.relaxation.initial_conditions import initial_field
from mrx.nullspace import compute_nullspaces
from mrx.diagnostics.plotting import plot_archive, plot_twin_axis
from mrx.diagnostics.poincare import locked_width, trace_archive
from mrx.relaxation.config import Drive, RelaxConfig
from mrx.relaxation.loop import check_checkpoint, initial_state, radial_cell_sq, relax, write_checkpoint
from mrx.relaxation.seeding import energy_seed

print(f"[env] mrx precision {current_precision()}")

geometry = cli.geometry
mesh = f"{geometry.resolution} p={geometry.spline_degree}"
print("[seq] building the de Rham sequence: the map, the operators and the preconditioners", flush=True)
seq, ops = geometry.build()
compute_nullspaces(seq)
h_r_sq = radial_cell_sq(seq)
nfp = seq.nfp

# %%
# 3) Get the seeded start and the unseeded reference from the run of Tutorial 4
# (its last checkpoint and reference.h5). Otherwise the same is made here: the
# equilibrium is relaxed by 10 Newton steps as in Tutorial 3, seeded by the energy
# criterion and relaxed by 5 more Newton steps as in Tutorial 4.
B_seeded = B_floor = rows = None
run = cli.seeded
ref = os.path.join(run, "reference.h5")
ckpts = sorted(glob.glob(os.path.join(run, "checkpoints", "state_*.h5")))
if os.path.exists(ref) and ckpts:
    with h5py.File(ckpts[-1], "r") as fh:
        same_mesh = (tuple(int(v) for v in fh.attrs["resolution"]) == geometry.resolution
                     and int(fh.attrs["degree"]) == geometry.spline_degree)
        if same_mesh:
            B_seeded = jnp.asarray(np.asarray(fh["B_n"]))
        else:
            print(f"[start] {run} is resolution {list(fh.attrs['resolution'])} p={fh.attrs['degree']}, skipped")
    if same_mesh:
        check_checkpoint(ckpts[-1], seq)
        check_checkpoint(ref, seq)
        with h5py.File(ref, "r") as fh:
            B_floor = jnp.asarray(np.asarray(fh["B_n"]))
        rows = json.load(open(os.path.join(run, "relax.json")))["seed"]
        print(f"[start] the seeded state {ckpts[-1]} and the reference {ref}")
if B_seeded is None:
    B0, ic = initial_field(seq)
    newton = RelaxConfig(geometry=geometry, budget=Budget(steps=10, chunk=5, floor_tol=0.0))
    ts = newton.stepper(seq)
    print("[start] no Tutorial 4 run: 10 Newton steps from the initial field, then the seed and 5 more",
          flush=True)
    B_floor = relax(initial_state(B0, ts), ts, **newton.relax_kwargs(), verbose=False).state.B_n
    B_seeded, rows = energy_seed(seq, B_floor)
    seeded = RelaxConfig(geometry=geometry, budget=Budget(steps=5, chunk=5, floor_tol=0.0))
    B_seeded = relax(initial_state(B_seeded, ts), ts, **seeded.relax_kwargs(), verbose=False).state.B_n
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
ts = eqx.tree_at(lambda t: t.resistive_reference, ts, B_star, is_leaf=lambda x: x is None)
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
path = os.path.join(cli.out, "trace.png")
fig.savefig(path, dpi=200)
if _INTERACTIVE:
    plt.show()
else:
    plt.close(fig)
print(f"  -> {path}")

# %%
# 7) Draw Poincare sections of the field before and after.
def sections(fields):
    """Trace the fields into one archive (the trace.npz of scripts/poincare_trace.py) and draw it on one iota
    and one pressure scale in the layout of the paper, as scripts/poincare_plot.py does."""
    print(f"[poincare] {', '.join(fields)}: tracing {cli.lines} field lines for {cli.periods} field periods each",
          flush=True)
    archive, results = trace_archive(seq, fields, lines=cli.lines, periods=cli.periods,
                                     source=f"{geometry.path} {mesh}")
    np.savez_compressed(os.path.join(cli.out, "trace.npz"), **archive)
    for fig in plot_archive(archive, os.path.join(cli.out, "poincare")).values():
        if _INTERACTIVE:
            plt.show()
        else:
            plt.close(fig)
    return results


def chain_widths(r, what):
    for row in rows:
        target = nfp * row["n"] / row["m"]
        width, n_locked = locked_width(r, target)
        print(f"[{what}] chain ({row['m']},{row['n']}) iota {target:.4f}: {n_locked} locked line(s), "
              f"width {width:.4f}")


results = sections({"before": (B_seeded, 0), "after": (B, res.steps)})
chain_widths(results["before"], "before")
chain_widths(results["after"], "after")

# %%
# 8) Save the run the way scripts/relax.py does: relax.json with the configuration of the drive and 
# the checkpoints of the start and the end. 
# Then scripts/poincare_trace.py --geometry ... checkpoints/state_*.h5 can trace it if needed.
os.makedirs(os.path.join(cli.out, "checkpoints"), exist_ok=True)
write_checkpoint(os.path.join(cli.out, "checkpoints", "state_000000.h5"), initial_state(B_seeded, ts), 0, seq)
write_checkpoint(os.path.join(cli.out, "checkpoints", f"state_{res.steps:06d}.h5"), res.state, res.steps, seq)
params = dict(cfg.params, geometry_path=os.path.abspath(geometry.path), knots=geometry.knots, ic="seeded",
              h_r_sq=float(h_r_sq), start_step=0)
with open(os.path.join(cli.out, "relax.json"), "w") as fh:
    json.dump(dict(params=params, seed=rows, trace=res.trace, qoi=res.qoi), fh, indent=1)
print(f"  -> {cli.out}/relax.json and checkpoints/")
print("[done] the drive takes the seeded field towards the reference. The helicity changes, the distance to "
      "the reference shrinks and the chains become those of the reference.")
