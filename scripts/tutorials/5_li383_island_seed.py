"""Tutorial 5: seed magnetic islands by the energy criterion, on li383.

Tutorial 3 relaxed li383's equilibrium field to a nested state and Tutorial 4
took it to the floor with Newton. Here we open islands in that floor the way
the paper does (Sec. 6.2): every resonance ``iota = nfp n / m`` inside the
field's iota range gets SIESTA's parallel seed ``dB = curl(A B / |B|)``,
``A = a(r) cos(2 pi (m theta - n zeta))``, with its radial profile free in the
mesh's own spline basis near the resonant radius, and all chains are solved
together for the amplitudes of least energy (``mrx.seeding.energy_seed``).
There is no phase to choose: a phase shift only scales the seed, because the
stellarator parity projector removes the odd part exactly, and the profile's
sign is set by the criterion. What gets printed is the whole story of the
seed: every chain in range, its rotational transform and radius, the
amplitude the criterion chose (as the resonant normal field
``|dB^r| / |B^zeta|`` at the chain), the pendulum width it implies, and how
much energy the seed removes.

The same criterion can be steered: name the chains by their rotational
transforms (``--iotas 0.5``) and, optionally, their amplitudes
(``--amplitudes 3e-3``), which then replace the criterion's. The tutorial
seeds once automatically and once by hand, sections the unseeded floor and
both seeded fields at five toroidal planes, then relaxes the automatic seed
ideally with Newton (Tutorial 4's stepper): the ideal flow is frozen-in, it
can move the islands and change their shape, it cannot close them, so the
chains are still there at the floor. Tutorial 6 turns the drive on.

It **warm-starts from Tutorial 4's Newton floor** (``outputs/tutorials/li383_newton``
or the shipped state in ``data/tutorials/``) on the same ``(10, 16, 16) p = 2``
mesh; otherwise it takes the equilibrium initial condition through the descent's
fast phase and 5 Newton steps itself. Runs in the default float32.

    python -u scripts/tutorials/5_li383_island_seed.py
"""

# %%
# Now we read the run's options. The defaults seed every chain in range at
# (10, 16, 16) p=2, then hand-seed the iota = 1/2 chain, and relax the
# automatic seed for 5 Newton steps.
from __future__ import annotations

import argparse
import os
import sys

# Run the cells top to bottom in a notebook / VS Code interactive window,
# or the whole file as a script (the CLI flags below still apply then).
_INTERACTIVE = "ipykernel" in sys.modules

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--geometry", default="data/wout_li383_low_res_reference.nc",
                help="a VMEC wout (.nc) or a GVEC state file (.dat); match Tutorials 3 and 4")
ap.add_argument("--ns", default="10,16,16")
ap.add_argument("--p", type=int, default=2)
ap.add_argument("--warm-start", default="outputs/tutorials/li383_newton,data/tutorials/li383_newton",
                help="run directories, first present wins: Tutorial 4's run, then its shipped state")
ap.add_argument("--iotas", default="0.5",
                help="the hand-seeded chains, by their rotational transforms nfp n / m (comma-separated)")
ap.add_argument("--amplitudes", default="3e-3",
                help="their amplitudes, the resonant normal field |dB^r| / |B^zeta| at the chain, signed; "
                     "empty = the energy criterion's, restricted to those chains")
ap.add_argument("--scale", type=float, default=1.0, help="multiply the automatic seed")
ap.add_argument("--newton-steps", type=int, default=5, help="Newton steps on the seeded field (a multiple of 5)")
ap.add_argument("--lines", type=int, default=24, help="Poincare field lines")
ap.add_argument("--periods", type=int, default=200, help="field periods per traced line")
ap.add_argument("--out", default="outputs/tutorials/li383_island_seed")
cli = ap.parse_args([] if _INTERACTIVE else None)
ns = tuple(int(v) for v in cli.ns.split(","))
os.makedirs(cli.out, exist_ok=True)

# %%
# Now we import MRX -- the sequence, the relaxation, the seeding and the
# Poincare tracer.
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
from mrx.initial_conditions import initial_field
from mrx.nullspace import compute_nullspaces
from mrx.plotting import render_section
from mrx.poincare import poincare, surface_label
from mrx.relax_config import Budget, Descent, Geometry, RelaxConfig, Seed, current_precision
from mrx.relaxation import initial_state, radial_cell_sq, relax, write_checkpoint
from mrx.seeding import energy_seed

print(f"[env] mrx precision {mrx.DTYPE}")

# The configuration objects of scripts/relax.py (mrx.relax_config): the geometry builds the sequence, a
# Seed group records what was seeded, a Budget and a Descent make the steppers.
geometry = Geometry(path=cli.geometry, resolution=ns, spline_degree=cli.p, precision=current_precision())
seq, ops = geometry.build()
compute_nullspaces(seq)
h_r_sq = radial_cell_sq(seq)
nfp = seq.nfp

# %%
# Now we get the nested floor to seed: Tutorial 4's Newton floor from the first
# run directory whose checkpoint is on disk and matches this mesh, otherwise
# the equilibrium initial condition through the descent's fast phase and a
# few Newton steps here.
B_floor = None
for run in cli.warm_start.split(","):
    ws_json = os.path.join(run, "relax.json")
    if not os.path.exists(ws_json):
        continue
    with open(ws_json) as fh:
        ws = json.load(fh)["params"]
    ckpts = sorted(glob.glob(os.path.join(run, "checkpoints", "state_*.h5")))
    if tuple(ws["resolution"]) == ns and int(ws["spline_degree"]) == cli.p and ckpts:
        with h5py.File(ckpts[-1], "r") as fh:
            B_floor = jnp.asarray(np.asarray(fh["B_n"]))
        print(f"[floor] warm-started from {ckpts[-1]} (resolution {ws['resolution']} p={ws['spline_degree']})")
        break
    print(f"[floor] run {run} is resolution {ws['resolution']} p={ws['spline_degree']} "
          f"(need {list(ns)} p={cli.p}); skipped")
if B_floor is None:
    B0, ic = initial_field(seq)
    print(f"[floor] built the equilibrium IC: ||B||_M {ic['B_norm_raw']:.4e}, ||div B|| {ic['div']:.2e}")
    descent = RelaxConfig(geometry=geometry, descent=Descent(method="gradient"),
                          budget=Budget(steps=500, chunk=50, floor_tol=1e-6))
    ts = descent.stepper(seq, h_r_sq)
    fast = relax(initial_state(B0, ts), ts, **descent.relax_kwargs(), verbose=False)
    newton = RelaxConfig(geometry=geometry, budget=Budget(steps=5, chunk=5, floor_tol=0.0))
    ts = newton.stepper(seq, h_r_sq)
    res = relax(initial_state(fast.state.B_n, ts), ts, **newton.relax_kwargs(), verbose=False)
    B_floor = res.state.B_n
    print(f"[floor] the descent's fast phase ({fast.steps} steps) and {res.steps} Newton steps: "
          f"||F|| {fast.trace['F'][0]:.3e} -> {res.trace['F'][-1]:.3e}")

# %%
# Now we seed the floor by the energy criterion: every resonance in range, the
# amplitudes of least energy. energy_seed prints the chains it finds and the
# joint optimum; the table below is what a user wants to know about each one.
def report(rows, what):
    h_r = float(np.sqrt(h_r_sq))
    print(f"[{what}] {len(rows)} chain(s) seeded:")
    print(f"  {'chain':>7}  {'iota':>7}  {'r_mn':>6}  {'|diota/dr|':>10}  {'amplitude dBr':>13}  "
          f"{'width w':>8}  {'w / h_r':>7}")
    for r in rows:
        print(f"  ({r['m']:>2},{r['n']:>2})  {nfp * r['n'] / r['m']:7.4f}  {r['r']:6.3f}  {r['diota']:10.3e}  "
              f"{r['dBr']:+13.3e}  {r['w']:8.4f}  {r['w'] / h_r:7.2f}")

B_auto, rows_auto = energy_seed(seq, B_floor, scale=cli.scale)
report(rows_auto, "seed, energy criterion")

# %%
# Now we seed by hand: the chains named by their rotational transforms, at the
# amplitudes given -- the same machinery, the criterion's amplitudes replaced.
# (Amplitudes without chains would be ignored with a warning: the criterion
# then decides.) The sign of an amplitude is the profile's sign, the one phase
# freedom the seed has.
iotas = [float(v) for v in cli.iotas.split(",")]
amplitudes = [float(v) for v in cli.amplitudes.split(",")] if cli.amplitudes else None
B_hand, rows_hand = energy_seed(seq, B_floor, iotas=iotas, amplitudes=amplitudes)
report(rows_hand, f"seed by hand, iotas {iotas}" + (f" at {amplitudes}" if amplitudes else ", the criterion's amplitudes"))

# %%
# Now we section the floor and both seeded fields at five planes. Trace once
# per field, cut five planes over half a field period; the chains show as
# island chains at their rotational transforms.
def sections(B_dof, tag, title):
    res = poincare(seq, B_dof, lines=cli.lines, periods=cli.periods, name=tag)
    for plane, sec in res["sections"].items():
        R, Z, aR, aZ = sec["R"], sec["Z"], sec["axisR"], sec["axisZ"]
        a_eff, xlabel = surface_label(R, Z, aR, aZ)
        fig, _ = render_section(
            R, Z, res["iota"], res["iota_err"], res["seed_r"], res["keep"],
            title=f"{title}  |  $\\zeta = {plane:g}$",
            subtitle=f"nfp = {nfp}   |   h/2 drift {res['drift']:.1e}",
            axis_RZ=(aR, aZ), profile_x=a_eff, profile_xlabel=xlabel, nfp=nfp,
            logical=(sec["logr"], sec["logth"]), iota_scatter=res["iota_scatter"])
        path = os.path.join(cli.out, f"poincare_{tag}_zeta{plane:g}.png")
        fig.savefig(path, dpi=200)
        if _INTERACTIVE:
            plt.show()
        else:
            plt.close(fig)
        print(f"  -> {path}")
    return res


def chain_widths(res, rows, what):
    """The island width of each seeded chain as the sections measure it: the radial extent of
    the lines locked to its rotational transform (|iota - nfp n / m| < 2e-3)."""
    iota, r, keep = np.asarray(res["iota"]), np.asarray(res["seed_r"]), np.asarray(res["keep"])
    for row in rows:
        target = nfp * row["n"] / row["m"]
        locked = keep & (np.abs(np.abs(iota) - target) < 2e-3)
        width = float(r[locked].max() - r[locked].min()) if locked.sum() > 1 else 0.0
        print(f"[{what}] chain ({row['m']},{row['n']}) iota {target:.4f}: {int(locked.sum())} locked line(s), "
              f"width {width:.4f} (pendulum estimate {row['w']:.4f})")

res_floor = sections(B_floor, "floor", f"the nested floor {ns} p={cli.p}")
res_auto = sections(B_auto, "seeded", f"seeded by the energy criterion {ns} p={cli.p}")
chain_widths(res_auto, rows_auto, "seeded")
res_hand = sections(B_hand, "seeded_by_hand", f"seeded by hand, iota {cli.iotas} {ns} p={cli.p}")
chain_widths(res_hand, rows_hand, "seeded by hand")

# %%
# Now we relax the automatically seeded field ideally with Newton (Tutorial
# 4's stepper). The flow is frozen-in -- helicity and the topology are kept
# -- so the islands can only move and change shape.
cfg = RelaxConfig(geometry=geometry, seed=Seed(seed=True, scale=cli.scale),
                  budget=Budget(steps=cli.newton_steps, chunk=5, floor_tol=0.0))
ts_newton = cfg.stepper(seq, h_r_sq)
res = relax(initial_state(B_auto, ts_newton), ts_newton, **cfg.relax_kwargs())
F = np.asarray(res.trace["F"], dtype=float)
H = np.asarray(res.qoi["helicity"], dtype=float)
it_n = np.asarray(res.trace["newton_it"])
print(f"[newton] {res.steps} steps: ||F|| {F[0]:.3e} -> {F[-1]:.3e} (lowest {F.min():.3e} at step {F.argmin() + 1}), "
      f"dH/H_0 = {(H[-1] - H[0]) / H[0]:+.1e}; MINRES iterations mean {np.abs(it_n).mean():.0f}")
B = res.state.B_n
res_relaxed = sections(B, "seeded_relaxed", f"seeded and relaxed {ns} p={cli.p}")
chain_widths(res_relaxed, rows_auto, "seeded, relaxed")

# %%
# Now we archive the run the way scripts/relax.py does: relax.json with the
# seeded chains, the seeded start and the relaxed state as checkpoints, and
# the unseeded floor as reference.h5 -- Tutorial 6 drives the seeded field
# back towards it.
os.makedirs(os.path.join(cli.out, "checkpoints"), exist_ok=True)
write_checkpoint(os.path.join(cli.out, "checkpoints", "state_000000.h5"), initial_state(B_auto, ts_newton), 0, seq)
write_checkpoint(os.path.join(cli.out, "checkpoints", f"state_{res.steps:06d}.h5"), res.state, res.steps, seq)
write_checkpoint(os.path.join(cli.out, "reference.h5"), initial_state(B_floor, ts_newton), 0, seq)
params = dict(cfg.params, geometry_path=os.path.abspath(cli.geometry), knots=geometry.knots, ic="warmstart",
              h_r_sq=h_r_sq, start_step=0)
with open(os.path.join(cli.out, "relax.json"), "w") as fh:
    json.dump(dict(params=params, seed=rows_auto, seed_by_hand=rows_hand, trace=res.trace, qoi=res.qoi), fh, indent=1)
print(f"  -> {cli.out}/relax.json, checkpoints/ and reference.h5 (the unseeded floor)")
print("[done] the chains the criterion found are island chains in the seeded sections and survive the ideal "
      "relaxation; Tutorial 6 drives the field back towards the unseeded floor.")
