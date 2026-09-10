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
ap.add_argument("--seeds", type=int, default=24, help="Poincare field lines")
ap.add_argument("--periods", type=int, default=200, help="field periods per traced line")
ap.add_argument("--out", default="outputs/tutorials/li383_island_seed")
cli = ap.parse_args([] if _INTERACTIVE else None)
ns = tuple(int(v) for v in cli.ns.split(","))
os.makedirs(cli.out, exist_ok=True)

# %%
# Now we import MRX -- the sequence, the seeded Clebsch initial condition, the
# relaxation time-stepper and loop, and the Poincare tracer.
import json

import jax.numpy as jnp
import matplotlib
if not _INTERACTIVE:
    matplotlib.use("Agg")  # headless as a script; a notebook keeps its inline backend
import matplotlib.pyplot as plt
import numpy as np
import mrx
from mrx.geometry import build_sequence
from mrx.gvec import load_clebsch
from mrx.initial_conditions import (clebsch_potential_form, potential_two_form, resonant_rho)
from mrx.nullspace import compute_nullspaces
from mrx.plotting import render_section
from mrx.poincare import (logical_field, require_zeta_parameterisation, seed_from_axis,
                          trace_and_classify, section_RZ, surface_label)
from mrx.relaxation import (TimeStepper, compute_divergence_norm, initial_state, relax,
                            write_checkpoint)

print(f"[env] mrx precision {mrx.DTYPE}")

seq, ops = build_sequence(cli.geometry, ns, cli.p)
compute_nullspaces(seq)

# %%
# Now we build two initial fields: the plain equilibrium, and the same field
# with a resonant seed added on the Clebsch potential.
m, n, rho0, width = (float(v) for v in cli.seed.split(","))
seed = (int(m), int(n), rho0, width, cli.seed_eps)
cb = load_clebsch(seq.equilibrium)
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
    field = logical_field(seq, jnp.asarray(B_dof), 2, True)
    require_zeta_parameterisation(field, name=tag)
    seeds = seed_from_axis(field, cli.seeds, 8, n_rays=4, steps_per_period=32)
    res = trace_and_classify(field, seeds, nfp, n_periods=cli.periods,
                             steps_per_period=32, saves_per_period=8)
    render_keep = ~(res["escaped"] | ~res["ok"])
    for plane in (0.0, 0.125, 0.25, 0.375, 0.5):
        R, Z, aR, aZ, _, _, lr, lth = section_RZ(seq, res["ys"], res["axis"], 8, plane)
        a_eff, xlabel = surface_label(R, Z, aR, aZ)
        fig, _ = render_section(
            R, Z, res["iota"], res["iota_err"], res["seeds"][:, 0], render_keep,
            title=f"{title}  |  $\\zeta = {plane:g}$",
            subtitle=f"nfp = {nfp}   |   h/2 drift {res['drift']:.1e}",
            axis_RZ=(aR, aZ), profile_x=a_eff, profile_xlabel=xlabel, nfp=nfp,
            logical=(lr, lth), iota_scatter=res["iota_scatter"])
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
    ts_descent = TimeStepper(seq=seq, cfl=0.5, history_size=1, velocity_smoothing_order=1)
    res_d = relax(initial_state(B, ts_descent), ts_descent, steps=cli.descent_steps, chunk=50,
                  floor_tol=1e-3)
    F = np.asarray(res_d.trace["F"], dtype=float)
    H = np.asarray(res_d.qoi["helicity"], dtype=float)
    print(f"[descent] {res_d.steps} steps ({res_d.stop}): ||F|| {F[0]:.3e} -> {F[-1]:.3e}, "
          f"dH/H_0 = {(H[-1] - H[0]) / H[0]:+.1e}")
    B = res_d.state.B_n
if cli.newton_steps:
    ts_newton = TimeStepper(seq=seq, cfl=0.5, history_size=0, velocity_smoothing_order=1,
                            newton=True, newton_tol=0.1, newton_maxiter=300,
                            newton_precond="laplacian", newton_dt_cap=1.0)
    res_n = relax(initial_state(B, ts_newton), ts_newton, steps=cli.newton_steps, chunk=5,
                  floor_tol=0.0)
    F = np.asarray(res_n.trace["F"], dtype=float)
    H = np.asarray(res_n.qoi["helicity"], dtype=float)
    it_n = np.asarray(res_n.trace["newton_it"])
    print(f"[newton] {res_n.steps} steps: ||F|| {F[0]:.3e} -> {F[-1]:.3e} "
          f"(lowest {F.min():.3e} at step {F.argmin() + 1}), dH/H_0 = {(H[-1] - H[0]) / H[0]:+.1e}; "
          f"MINRES iterations mean {np.abs(it_n).mean():.0f}, fallbacks "
          f"{int(np.asarray(res_n.trace['newton_fallback']).sum())}")
    B = res_n.state.B_n

# %%
# Now we section the relaxed seeded field: the chain is still there.
if cli.descent_steps or cli.newton_steps:
    sections(B, f"relaxed_eps{cli.seed_eps:g}",
             f"seeded ({int(m)},{int(n)}) eps={cli.seed_eps:g}, relaxed {ns} p={cli.p}")
    os.makedirs(os.path.join(cli.out, "checkpoints"), exist_ok=True)
    ts_any = ts_newton if cli.newton_steps else ts_descent
    steps = (res_d.steps if cli.descent_steps else 0) + (res_n.steps if cli.newton_steps else 0)
    write_checkpoint(os.path.join(cli.out, "checkpoints", "state_000000.h5"),
                     initial_state(B_seeded, ts_any), 0)
    write_checkpoint(os.path.join(cli.out, "checkpoints", f"state_{steps:06d}.h5"),
                     initial_state(B, ts_any), steps)
    params = dict(geometry_path=os.path.abspath(cli.geometry), ns=list(ns), p=cli.p, nfp=None,
                  r_refine="", precision=str(mrx.DTYPE), steps=steps, scheme="explicit",
                  auxiliary_B_field=False, ic="clebsch", seed=cli.seed, seed_eps=cli.seed_eps)
    with open(os.path.join(cli.out, "relax.json"), "w") as fh:
        json.dump(dict(params=params, reconnect=[]), fh, indent=1)
    print(f"  -> {cli.out}/relax.json and checkpoints/")
print("[done] the island at the resonant chain appears in the seeded sections and survives the "
      "ideal relaxation; Tutorial 6 (with resistivity) can reconnect it.")
