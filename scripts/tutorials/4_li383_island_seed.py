"""Tutorial 4: what a resonant seed does to the initial field on li383.

Tutorial 3 relaxed li383's equilibrium field to a nested state. Here we add a
small **resonant perturbation** to the initial condition and look at the field
it produces -- no relaxation, just the seeded initial field next to the
unseeded one, so the effect of the seed is unmistakable. The seed opens a
magnetic island at the resonant surface; Tutorial 5 turns on resistivity and
lets it reconnect.

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

The mesh is Tutorial 3's ``(10, 16, 16) p = 2`` in the default float32. This is
the cheapest tutorial: no descent, just the two initial fields sectioned at
five toroidal planes.

    python -u scripts/tutorials/4_li383_island_seed.py
"""

# %%
# Now we read the run's options. The defaults seed the (6,1) chain on li383's
# iota=1/2 surface at (10, 16, 16) p=2.
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
ap.add_argument("--seeds", type=int, default=24, help="Poincare field lines")
ap.add_argument("--periods", type=int, default=200, help="field periods per traced line")
ap.add_argument("--out", default="outputs/tutorials/li383_island_seed")
cli = ap.parse_args([] if _INTERACTIVE else None)
ns = tuple(int(v) for v in cli.ns.split(","))
os.makedirs(cli.out, exist_ok=True)

# %%
# Now we import MRX -- the sequence, the seeded Clebsch initial condition, and
# the Poincare tracer. There is no relaxation in this tutorial.
import jax.numpy as jnp
import matplotlib
if not _INTERACTIVE:
    matplotlib.use("Agg")  # headless as a script; a notebook keeps its inline backend
import matplotlib.pyplot as plt
import mrx
from mrx.geometry import build_sequence
from mrx.gvec import load_clebsch
from mrx.relaxation import compute_divergence_norm
from mrx.initial_conditions import (clebsch_potential_form, potential_two_form, resonant_rho)
from mrx.nullspace import compute_nullspaces
from mrx.plotting import render_section
from mrx.poincare import (logical_field, require_zeta_parameterisation, seed_from_axis,
                          trace_and_classify, section_RZ, surface_label)

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
# Trace once per field, cut five planes over half a field period. The island at
# the resonant chain shows in the seeded section, not the unseeded one.
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
print("[done] the island at the resonant chain appears in the seeded section only; "
      "the ideal descent of Tutorial 5 (with resistivity) can reconnect it.")
