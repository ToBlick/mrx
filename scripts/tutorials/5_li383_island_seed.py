"""Tutorial 5: a seeded island under ideal relaxation, on li383.

Tutorial 3 relaxed li383's equilibrium field to a nested state and Tutorial 4
took it to the floor with Newton. Here we add a small **resonant
perturbation** to the initial condition, look at the field it produces next
to the unseeded one -- so the effect of the seed is unmistakable -- and then
relax the seeded field ideally the same way, the descent through its fast
phase and Newton to the floor. The ideal flow is frozen-in: it can move the
island and change its shape, it cannot close it, so the chain is still there
at the floor. Tutorial 6 turns on resistivity and lets it reconnect.

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
edge. The run uses the high-resolution reference ``wout_li383_1.4m.nc``: on the
coarse reference the field's reconstruction residual sits on top of the seeded
signal, so the seed cannot be told from the noise. Vary ``--seed-eps``
(1e-3, 3e-3, 1e-2) to watch the island width track ``sqrt(eps)``.

The mesh is Tutorial 3's ``(10, 16, 16) p = 2`` in the default float32. The
relaxation costs what Tutorials 3 and 4 cost together; ``--descent-steps 0
--newton-steps 0`` skips it and just sections the two initial fields.

    python -u scripts/tutorials/5_li383_island_seed.py
"""

# %%
# Now we read the run's options. The defaults seed the (6,1) chain on li383's
# iota=1/2 surface at (10, 16, 16) p=2 and relax it: 200 descent steps, then
# 5 Newton steps.
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
ap.add_argument("--ns", default="10,16,16")
ap.add_argument("--p", type=int, default=2)
ap.add_argument("--seed", default="6,1,0.544,0.1",
                help='resonant seed "m,n,rho0,width"; (6,1) is the iota=1/2 surface')
ap.add_argument("--seed-eps", type=float, default=1e-2,
                help="resonant normal field |dB^rho|/|B^zeta| at rho0; width ~ sqrt(eps)")
ap.add_argument("--descent-steps", type=int, default=200,
                help="the descent's fast phase on the seeded field (a multiple of 50)")
ap.add_argument("--newton-steps", type=int, default=5,
                help="Newton steps after it (a multiple of 5)")
ap.add_argument("--lines", type=int, default=24, help="Poincare field lines")
ap.add_argument("--periods", type=int, default=200, help="field periods per traced line")
ap.add_argument("--out", default="outputs/tutorials/li383_island_seed")
cli = ap.parse_args([] if _INTERACTIVE else None)
ns = tuple(int(v) for v in cli.ns.split(","))
os.makedirs(cli.out, exist_ok=True)

# %%
# Now we import MRX -- the sequence, the seeded Clebsch initial condition, the
# relaxation time-stepper and loop, and the Poincare tracer.
import json

import matplotlib
if not _INTERACTIVE:
    matplotlib.use("Agg")  # headless as a script; a notebook keeps its inline backend
import matplotlib.pyplot as plt
import numpy as np
import mrx
from mrx.gvec import load_clebsch
from mrx.initial_conditions import clebsch_potential_form, potential_two_form, resonant_rho
from mrx.nullspace import compute_nullspaces
from mrx.plotting import render_section
from mrx.poincare import poincare, surface_label
from mrx.relax_config import Budget, Descent, Geometry, RelaxConfig, Seed, current_precision
from mrx.relaxation import (compute_divergence_norm, initial_state, radial_cell_sq, relax,
                            write_checkpoint)

print(f"[env] mrx precision {mrx.DTYPE}")

# scripts/relax.py's configuration objects (mrx.relax_config): the geometry builds the sequence, the
# seed group parses the seed, and below a descent and a Newton configuration make the steppers.
geometry = Geometry(path=cli.geometry, ns=ns, p=cli.p, precision=current_precision())
seq, ops = geometry.build()
compute_nullspaces(seq)
h_r_sq = radial_cell_sq(seq)

# %%
# Now we build two initial fields: the plain equilibrium, and the same field
# with a resonant seed added on the Clebsch potential.
seed_cfg = Seed(spec=cli.seed, eps=cli.seed_eps)
seed = seed_cfg.parsed()
m, n, rho0, width = seed[:4]
cb = load_clebsch(seq.equilibrium, nfp=seq.nfp)
nfp = int(cb["nfp"])
rho_res = resonant_rho(cb, int(m), int(n))
print(f"[ic] seed (m, n) = ({int(m)}, {int(n)}) at rho0 {rho0:g}, width {width:g}, "
      f"eps {cli.seed_eps:.2e}")
print(f"[ic] the file's |iota| = nfp n / m = {nfp * n / m:.4f} chain sits at "
      f"rho = {rho_res:.3f} (seed rho0 {rho0:g})")
B_unseeded, _, _ = potential_two_form(seq, clebsch_potential_form(cb))
B_seeded, norm, wall = potential_two_form(seq, clebsch_potential_form(cb, seed))
print(f"[ic] seeded field: ||B||_M {norm:.4e}, ||div B|| {compute_divergence_norm(B_seeded, seq):.2e}, "
      f"wall-normal part {wall:.1e}")

# %%
# Now we take Poincare sections of BOTH initial fields at five planes: the
# island at the resonant chain shows in the seeded sections, not the unseeded.
# Trace once per field, cut five planes over half a field period.
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

sections(B_unseeded, "unseeded", f"unseeded IC {ns} p={cli.p}")
sections(B_seeded, f"seeded_eps{cli.seed_eps:g}",
         f"seeded ({int(m)},{int(n)}) eps={cli.seed_eps:g} {ns} p={cli.p}")

# %%
# Now we relax the seeded field ideally: the descent through its fast phase
# (Tutorial 3's stepper), then Newton to the floor (Tutorial 4's). Both are
# frozen-in flows -- helicity and the topology are kept -- so the island can
# only move and change shape.
B = B_seeded
if cli.descent_steps:
    descent = RelaxConfig(geometry=geometry, seed=seed_cfg, descent=Descent(method="gradient"),
                          budget=Budget(steps=cli.descent_steps, chunk=50, floor_tol=1e-6))
    ts_descent = descent.stepper(seq, h_r_sq)
    res_d = relax(initial_state(B, ts_descent), ts_descent, **descent.relax_kwargs(h_r_sq))
    F = np.asarray(res_d.trace["F"], dtype=float)
    H = np.asarray(res_d.qoi["helicity"], dtype=float)
    print(f"[descent] {res_d.steps} steps ({res_d.stop}): ||F|| {F[0]:.3e} -> {F[-1]:.3e}, "
          f"dH/H_0 = {(H[-1] - H[0]) / H[0]:+.1e}")
    B = res_d.state.B_n
if cli.newton_steps:
    newton = RelaxConfig(geometry=geometry, seed=seed_cfg,
                         budget=Budget(steps=cli.newton_steps, chunk=5, floor_tol=0.0))
    ts_newton = newton.stepper(seq, h_r_sq)
    res_n = relax(initial_state(B, ts_newton), ts_newton, **newton.relax_kwargs(h_r_sq))
    F = np.asarray(res_n.trace["F"], dtype=float)
    H = np.asarray(res_n.qoi["helicity"], dtype=float)
    it_n = np.asarray(res_n.trace["newton_it"])
    print(f"[newton] {res_n.steps} steps: ||F|| {F[0]:.3e} -> {F[-1]:.3e} "
          f"(lowest {F.min():.3e} at step {F.argmin() + 1}), dH/H_0 = {(H[-1] - H[0]) / H[0]:+.1e}; "
          f"MINRES iterations mean {np.abs(it_n).mean():.0f}")
    B = res_n.state.B_n

# %%
# Now we section the relaxed seeded field: the chain is still there.
if cli.descent_steps or cli.newton_steps:
    sections(B, f"relaxed_eps{cli.seed_eps:g}",
             f"seeded ({int(m)},{int(n)}) eps={cli.seed_eps:g}, relaxed {ns} p={cli.p}")
    os.makedirs(os.path.join(cli.out, "checkpoints"), exist_ok=True)
    last, ts_any = (newton, ts_newton) if cli.newton_steps else (descent, ts_descent)
    steps = (res_d.steps if cli.descent_steps else 0) + (res_n.steps if cli.newton_steps else 0)
    write_checkpoint(os.path.join(cli.out, "checkpoints", "state_000000.h5"),
                     initial_state(B_seeded, ts_any), 0)
    write_checkpoint(os.path.join(cli.out, "checkpoints", f"state_{steps:06d}.h5"),
                     initial_state(B, ts_any), steps)
    params = dict(last.params, geometry_path=os.path.abspath(cli.geometry), knots=geometry.knots, ic="clebsch",
                  h_r_sq=h_r_sq, start_step=0)
    with open(os.path.join(cli.out, "relax.json"), "w") as fh:
        json.dump(dict(params=params, reconnect=[]), fh, indent=1)
    print(f"  -> {cli.out}/relax.json and checkpoints/")
print("[done] the island at the resonant chain appears in the seeded sections and survives the "
      "ideal relaxation; Tutorial 6 (with resistivity) can reconnect it.")
