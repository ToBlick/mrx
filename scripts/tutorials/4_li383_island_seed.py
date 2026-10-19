"""Tutorial 4: Seeding magnetic islands.

Tutorial 3 relaxed the equilibrium field of li383 with Newton to a state with nested surfaces up to grid-scale
effects. We now open islands in that field. 

Every resonance ``iota = nfp n / m`` inside the iota range of the field gets an island seed
``dB = curl(A B / |B|)`` with ``A = a(r) cos(2 pi (m theta - n zeta))``. The radial profile ``a(r)`` is free
in the spline basis of the mesh near the resonant radius, and the amplitudes of all chains are determined by
least energy (``mrx.relaxation.seeding.energy_seed``).

The criterion can also be overriden. Name the chains by their rotational transforms (``--iotas 0.5``) and, 
optionally, give their amplitudes (``--amplitudes 3e-3``), which then replace those of the criterion 
(``--amplitudes None`` keeps the criterion's amplitudes for the named chains). 

The tutorial draws Poincare sections of the unseeded minimal force state and both seeded fields, with ``--lines`` 
field lines (160 by default, the paper's li383 figures trace 320 for 600 periods) each traced for ``--periods`` field periods (400 by default). It then relaxes the 
automatic seed.

The tutorial starts from the Newton state of Tutorial 3 (``outputs/tutorials/3_li383_newton``) on the same
``(12, 16, 16) p = 2`` mesh, so run Tutorial 3 first.

    python -u scripts/tutorials/4_li383_island_seed.py
    python -u scripts/tutorials/4_li383_island_seed.py --iotas 0.5 0.6 --amplitudes 3e-3 -2e-3
"""

# %%
# 1) Setup.
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

print("[tutorial 4] island seeds on li383: importing JAX and MRX", flush=True)
from typing import Optional

import tyro

from mrx.relaxation.config import Budget, Geometry

_INTERACTIVE = "ipykernel" in sys.modules


@tyro.conf.configure(tyro.conf.EnumChoicesFromValues)
@dataclass(frozen=True)
class Options:
    """Tutorial 4: seed island chains by the energy criterion and by hand, on li383."""
    geometry: Geometry = Geometry(path="data/wout_li383_low_res_reference.nc", resolution=(12, 16, 16),
                                  spline_degree=2)
    warm_start: str = "outputs/tutorials/3_li383_newton"
    """The run directory of Tutorial 3 to start from."""
    iotas: tuple[float, ...] = (0.5,)
    """The chains seeded by hand, by their rotational transforms nfp n / m."""
    amplitudes: Optional[tuple[float, ...]] = (3e-3,)
    """One signed amplitude per --iotas value, the resonant normal field |dB^r| / |B^zeta| at the chain. None takes the amplitudes of the energy criterion, restricted to those chains."""
    scale: float = 1.0
    """The factor applied to the automatic seed."""
    budget: Budget = Budget(steps=5, chunk=5, floor_tol=0.0)
    """The step budget of the Newton run on the seeded field."""
    lines: int = 160
    """The number of field lines in the Poincare sections."""
    periods: int = 400
    """The number of field periods each line is traced."""
    out: str = "outputs/tutorials/4_li383_island_seed"
    """The directory of the run."""


cli = tyro.cli(Options, args=[] if _INTERACTIVE else None)
os.makedirs(cli.out, exist_ok=True)

# %%
# 2) Import MRX.
import glob

import numpy as np
from mrx.nullspace import compute_nullspaces
from mrx.diagnostics.plotting import trace_and_plot
from mrx.diagnostics.poincare import locked_width
from mrx.relaxation.config import RelaxConfig, Seed
from mrx.relaxation.loop import initial_state, radial_cell_sq, read_field, relax, save_run, write_checkpoint
from mrx.relaxation.seeding import energy_seed


# These are the configuration objects of scripts/relax.py (mrx.relaxation.config). The geometry builds the
# sequence, a Seed group records what was seeded, and a Budget makes the Newton stepper.
geometry = cli.geometry
mesh = f"{geometry.resolution} p={geometry.spline_degree}"
print("[seq] building the de Rham sequence: the map, the operators and the preconditioners", flush=True)
seq = geometry.build()
compute_nullspaces(seq)
h_r_sq = radial_cell_sq(seq)
nfp = seq.nfp

# %%
# 3) Get the nested state: the last checkpoint of the Newton run of Tutorial 3, on the same mesh.
ckpts = sorted(glob.glob(os.path.join(cli.warm_start, "checkpoints", "state_*.h5")))
if not ckpts:
    sys.exit(f"no checkpoints in {cli.warm_start}: run Tutorial 3 first, or point --warm-start at its run")
B_floor, _ = read_field(ckpts[-1], seq)
print(f"[floor] warm-started from {ckpts[-1]}")

# %%
# 4) Seed the floor by the energy criterion: every resonance in range, at the
# amplitudes of least energy. energy_seed prints the chains it finds and the
# joint optimum.
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
# (Optional) Seed by hand
iotas = list(cli.iotas)
amplitudes = list(cli.amplitudes) if cli.amplitudes else None
B_hand, rows_hand = energy_seed(seq, B_floor, iotas=iotas, amplitudes=amplitudes)
report(rows_hand, f"seed by hand, iotas {iotas}" + (f" at {amplitudes}" if amplitudes else ", the criterion's amplitudes"))

# %%
# 5) A helper that measures the seeded chains in the Poincare sections of step 7.
def chain_widths(res, rows, what):
    for row in rows:
        target = nfp * row["n"] / row["m"]
        width, n_locked = locked_width(res, target)
        print(f"[{what}] chain ({row['m']},{row['n']}) iota {target:.4f}: {n_locked} locked line(s), "
              f"width {width:.4f} (pendulum estimate {row['w']:.4f})")


# %%
# 6) Relax the automatically seeded field.
cfg = RelaxConfig(geometry=geometry, seed=Seed(seed=True, scale=cli.scale), budget=cli.budget)
ts_newton = cfg.stepper(seq)
res = relax(initial_state(B_auto, ts_newton), ts_newton, **cfg.relax_kwargs())
F = np.asarray(res.trace["F"], dtype=float)
H = np.asarray(res.qoi["helicity"], dtype=float)
it_n = np.asarray(res.trace["newton_it"])
print(f"[newton] {res.steps} steps: ||F|| {F[0]:.3e} -> {F[-1]:.3e} (lowest {F.min():.3e} at step {F.argmin() + 1}), "
      f"dH/H_0 = {(H[-1] - H[0]) / H[0]:+.1e}, MINRES iterations mean {np.abs(it_n).mean():.0f}")
B = res.state.B_n

# %%
# 7) Draw the Poincare sections of all four fields on one iota and one pressure scale (the trace.npz of
# scripts/poincare_trace.py, drawn as scripts/poincare_plot.py does), and measure the chains in them.
results = trace_and_plot(seq, {"floor": (B_floor, 0), "seeded": (B_auto, 0), "seeded_by_hand": (B_hand, 0),
                               "seeded_relaxed": (B, res.steps)}, cli.out, lines=cli.lines, periods=cli.periods,
                         source=f"{geometry.path} {mesh}")
chain_widths(results["seeded"], rows_auto, "seeded")
chain_widths(results["seeded_by_hand"], rows_hand, "seeded by hand")
chain_widths(results["seeded_relaxed"], rows_auto, "seeded, relaxed")

# %%
# 8) Save the run the way scripts/relax.py does. Tutorial 5 drives the seeded field
# back towards the nested state.
save_run(cli.out, cfg, seq, initial_state(B_auto, ts_newton), res, "warmstart", seed=rows_auto,
         seed_by_hand=rows_hand)
write_checkpoint(os.path.join(cli.out, "reference.h5"), initial_state(B_floor, ts_newton), 0, seq)
print(f"  -> {cli.out}/reference.h5 (the unseeded floor)")
print("[done] the chains the criterion found are island chains in the seeded sections and survive the ideal "
      "relaxation. Tutorial 5 drives the field back towards the unseeded floor.")
