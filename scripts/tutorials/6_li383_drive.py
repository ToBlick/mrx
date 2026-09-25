"""Tutorial 6: drive a seeded field back towards the unseeded equilibrium, on li383.

Tutorials 3 to 5 stayed ideal (eta = 0): a frozen-in flow that lowers the
energy without ever changing the field's topology, so the islands Tutorial 5
opened survive the relaxation. The paper's last experiment (Sec. 7) breaks
that constraint with a **drive**: after every ideal step a backward-Euler
resistive step ``dB/dt = -eta curl (J - J*)`` with the dose ``eps = C h_r^2``
pulls the current towards ``J*``, the current of a reference field ``B*``
-- here the unseeded nested floor of Tutorial 4, after one heat step that
removes its rational-surface sheets (sustained, they would make the start
a fixed point). Field lines can now reconnect, helicity is no longer
conserved, and the field goes to a resistive steady state: the islands close
(or, with a resonant drive of the reference, open and saturate). This is
``scripts/relax.py --drive-resistivity C --drive-reference REF``.

The run: the seeded, relaxed state of Tutorial 5 (``outputs/tutorials/li383_island_seed``,
its ``reference.h5`` the unseeded floor), or the same built here when that
run is absent, then ``--steps`` Newton steps with the drive on. What gets
printed: the reference and its smoothing, the dose, and after every chunk
the force residual, the helicity and the distance to the reference; before
and after, the island width of every seeded chain from the sections (the
radial extent of the lines locked to its rotational transform). Runs in the
default float32 at ``(10, 16, 16) p = 2``; ~2x Tutorial 4.

    python -u scripts/tutorials/6_li383_drive.py
"""

# %%
# Now we parse the run's options. Everything has a default, so in a notebook
# the cell just uses them; from the command line the flags below still apply.
from __future__ import annotations

import argparse
import os
import sys

_INTERACTIVE = "ipykernel" in sys.modules

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--geometry", default="data/wout_li383_low_res_reference.nc",
                help="a VMEC wout (.nc) or a GVEC state file (.dat); match Tutorials 3 to 5")
ap.add_argument("--ns", default="10,16,16")
ap.add_argument("--p", type=int, default=2)
ap.add_argument("--seeded", default="outputs/tutorials/li383_island_seed,data/tutorials/li383_island_seed",
                help="Tutorial 5's run directories (checkpoints/ and reference.h5), first present wins")
ap.add_argument("--resistivity", type=float, default=0.064,
                help="the dose C of every step, eps = C h_r^2 (the paper's 0.064)")
ap.add_argument("--reference-smoothing", type=float, default=0.1,
                help="the heat step c h_r^2 applied to the reference")
ap.add_argument("--steps", type=int, default=20, help="Newton steps with the drive on (a multiple of --chunk)")
ap.add_argument("--chunk", type=int, default=5)
ap.add_argument("--lines", type=int, default=24, help="Poincare field lines")
ap.add_argument("--periods", type=int, default=200, help="field periods per traced line")
ap.add_argument("--out", default="outputs/tutorials/li383_drive")
cli = ap.parse_args([] if _INTERACTIVE else None)
ns = tuple(int(v) for v in cli.ns.split(","))
os.makedirs(cli.out, exist_ok=True)

# %%
# Now we import MRX and its relaxation, seeding and Poincare machinery.
import glob
import json

import equinox as eqx
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
from mrx.plotting import plot_twin_axis, render_section
from mrx.poincare import poincare, surface_label
from mrx.relax_config import Budget, Descent, Drive, Geometry, RelaxConfig, current_precision
from mrx.relaxation import initial_state, radial_cell_sq, relax, resistive_step, write_checkpoint
from mrx.seeding import energy_seed

print(f"[env] mrx precision {mrx.DTYPE}")

geometry = Geometry(path=cli.geometry, resolution=ns, spline_degree=cli.p, precision=current_precision())
seq, ops = geometry.build()
compute_nullspaces(seq)
h_r_sq = radial_cell_sq(seq)
nfp = seq.nfp

# %%
# Now we get the seeded start and the unseeded reference: Tutorial 5's run
# (its last checkpoint and reference.h5), or the same made here -- the
# equilibrium relaxed through the descent's fast phase and Newton, then seeded
# by the energy criterion.
B_seeded = B_floor = rows = None
for run in cli.seeded.split(","):
    ref = os.path.join(run, "reference.h5")
    ckpts = sorted(glob.glob(os.path.join(run, "checkpoints", "state_*.h5")))
    if not (os.path.exists(ref) and ckpts):
        continue
    with h5py.File(ckpts[-1], "r") as fh:
        if tuple(int(v) for v in fh.attrs["resolution"]) != ns or int(fh.attrs["degree"]) != cli.p:
            print(f"[start] {run} is resolution {list(fh.attrs['resolution'])} p={fh.attrs['degree']}; skipped")
            continue
        B_seeded = jnp.asarray(np.asarray(fh["B_n"]))
    with h5py.File(ref, "r") as fh:
        B_floor = jnp.asarray(np.asarray(fh["B_n"]))
    rows = json.load(open(os.path.join(run, "relax.json")))["seed"]
    print(f"[start] the seeded state {ckpts[-1]} and the reference {ref}")
    break
if B_seeded is None:
    B0, ic = initial_field(seq)
    descent = RelaxConfig(geometry=geometry, descent=Descent(method="gradient"),
                          budget=Budget(steps=500, chunk=50, floor_tol=1e-6))
    ts = descent.stepper(seq, h_r_sq)
    fast = relax(initial_state(B0, ts), ts, **descent.relax_kwargs(), verbose=False)
    newton = RelaxConfig(geometry=geometry, budget=Budget(steps=5, chunk=5, floor_tol=0.0))
    ts = newton.stepper(seq, h_r_sq)
    B_floor = relax(initial_state(fast.state.B_n, ts), ts, **newton.relax_kwargs(), verbose=False).state.B_n
    print(f"[start] no Tutorial 5 run: the equilibrium relaxed here ({fast.steps} descent + 5 Newton steps), "
          f"then seeded by the energy criterion")
    B_seeded, rows = energy_seed(seq, B_floor)
    B_seeded = relax(initial_state(B_seeded, ts), ts, **newton.relax_kwargs(), verbose=False).state.B_n
for r in rows:
    print(f"[start] seeded chain ({r['m']},{r['n']}) iota {nfp * r['n'] / r['m']:.4f} at r {r['r']:.3f}: "
          f"amplitude {r['dBr']:+.3e}, pendulum width {r['w']:.4f}")
print(f"[start] ||B_seeded - B*|| / ||B*|| = "
      f"{float(seq.odd.l2_norm(B_seeded - B_floor, 2) / seq.odd.l2_norm(B_floor, 2)):.3e}")

# %%
# Now we set the drive up as scripts/relax.py does: the Drive group of the
# configuration, its reference the unseeded floor written as a checkpoint,
# the stepper with the resistivity, the reference smoothed by one heat step.
ref_path = os.path.join(cli.out, "reference.h5")
cfg = RelaxConfig(geometry=geometry, budget=Budget(steps=cli.steps, chunk=cli.chunk, floor_tol=0.0),
                  drive=Drive(resistivity=cli.resistivity, reference=ref_path,
                              reference_smoothing=cli.reference_smoothing))
ts = cfg.stepper(seq, h_r_sq)
write_checkpoint(ref_path, initial_state(B_floor, ts), 0, seq)
B_star = resistive_step(B_floor, seq, cfg.drive.reference_smoothing * h_r_sq)[0]
ts = eqx.tree_at(lambda t: t.resistive_reference, ts, B_star, is_leaf=lambda x: x is None)
print(f"[drive] eps = {cfg.drive.resistivity:g} h_r^2 = {float(ts.resistivity):.3e} per step towards the current of "
      f"B* = the floor after a heat step of {cfg.drive.reference_smoothing:g} h_r^2 "
      f"(||B* - floor|| / ||floor|| = {float(seq.odd.l2_norm(B_star - B_floor, 2) / seq.odd.l2_norm(B_floor, 2)):.3e})")


def progress(res):
    q = res.qoi
    dist = float(seq.odd.l2_norm(res.state.B_n - B_star, 2) / seq.odd.l2_norm(B_star, 2))
    print(f"[drive] step {res.steps:4d}: ||F|| {res.trace['F'][-1]:.3e}  resid {res.trace['resid'][-1]:.3e}  "
          f"helicity {q['helicity'][-1]:+.6e} ({(q['helicity'][-1] - q['helicity'][0]) / q['helicity'][0]:+.2e} of "
          f"the start)  ||B - B*|| / ||B*|| {dist:.3e}")


# %%
# Now we run: Newton steps, each followed by the resistive dose. The helicity
# decays at the resistive rate, the distance to the reference shrinks, and
# the islands close as the field heads for the resistive steady state.
res = relax(initial_state(B_seeded, ts), ts, on_chunk=progress, **cfg.relax_kwargs())
B = res.state.B_n
F = np.asarray(res.trace["F"], dtype=float)
H = np.asarray(res.qoi["helicity"], dtype=float)
print(f"[drive] {res.steps} steps ({res.stop}): ||F|| {F[0]:.3e} -> {F[-1]:.3e}, "
      f"H {H[0]:+.6e} -> {H[-1]:+.6e} ({(H[-1] - H[0]) / H[0]:+.2e}), "
      f"||B - B*|| / ||B*|| {float(seq.odd.l2_norm(B_seeded - B_star, 2) / seq.odd.l2_norm(B_star, 2)):.3e} -> "
      f"{float(seq.odd.l2_norm(B - B_star, 2) / seq.odd.l2_norm(B_star, 2)):.3e}")

# %%
# Now we plot the force residual and the helicity over the run.
fig, _ = plot_twin_axis(F, H, x_right=np.asarray(res.qoi["it"], dtype=float), right_log=False,
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
# Now we section the field before and after the drive at five planes and
# measure every seeded chain's island width both times: the radial extent of
# the lines locked to its rotational transform.
def sections(B_dof, tag, title):
    r = poincare(seq, B_dof, lines=cli.lines, periods=cli.periods, name=tag)
    for plane, sec in r["sections"].items():
        R, Z, aR, aZ = sec["R"], sec["Z"], sec["axisR"], sec["axisZ"]
        a_eff, xlabel = surface_label(R, Z, aR, aZ)
        fig, _ = render_section(
            R, Z, r["iota"], r["iota_err"], r["seed_r"], r["keep"],
            title=f"{title}  |  $\\zeta = {plane:g}$",
            subtitle=f"nfp = {nfp}   |   h/2 drift {r['drift']:.1e}",
            axis_RZ=(aR, aZ), profile_x=a_eff, profile_xlabel=xlabel, nfp=nfp,
            logical=(sec["logr"], sec["logth"]), iota_scatter=r["iota_scatter"])
        path = os.path.join(cli.out, f"poincare_{tag}_zeta{plane:g}.png")
        fig.savefig(path, dpi=200)
        if _INTERACTIVE:
            plt.show()
        else:
            plt.close(fig)
        print(f"  -> {path}")
    return r


def chain_widths(r, what):
    iota, seed_r, keep = np.asarray(r["iota"]), np.asarray(r["seed_r"]), np.asarray(r["keep"])
    for row in rows:
        target = nfp * row["n"] / row["m"]
        locked = keep & (np.abs(np.abs(iota) - target) < 2e-3)
        width = float(seed_r[locked].max() - seed_r[locked].min()) if locked.sum() > 1 else 0.0
        print(f"[{what}] chain ({row['m']},{row['n']}) iota {target:.4f}: {int(locked.sum())} locked line(s), "
              f"width {width:.4f}")


chain_widths(sections(B_seeded, "before", f"seeded, before the drive {ns} p={cli.p}"), "before")
chain_widths(sections(B, "after", f"after {res.steps} driven steps {ns} p={cli.p}"), "after")

# %%
# Now we archive the run the way scripts/relax.py does: relax.json with the
# drive's configuration and the checkpoints of the start and the end, so
# scripts/poincare_trace.py --geometry ... checkpoints/state_*.h5 traces it.
os.makedirs(os.path.join(cli.out, "checkpoints"), exist_ok=True)
write_checkpoint(os.path.join(cli.out, "checkpoints", "state_000000.h5"), initial_state(B_seeded, ts), 0, seq)
write_checkpoint(os.path.join(cli.out, "checkpoints", f"state_{res.steps:06d}.h5"), res.state, res.steps, seq)
params = dict(cfg.params, geometry_path=os.path.abspath(cli.geometry), knots=geometry.knots, ic="seeded",
              h_r_sq=h_r_sq, start_step=0)
with open(os.path.join(cli.out, "relax.json"), "w") as fh:
    json.dump(dict(params=params, seed=rows, trace=res.trace, qoi=res.qoi), fh, indent=1)
print(f"  -> {cli.out}/relax.json and checkpoints/")
print("[done] the drive closes the seeded chains: the locked lines and their width go, the helicity decays, "
      "the field approaches the reference current.")
