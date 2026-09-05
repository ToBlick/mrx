"""Tutorial 2: the vacuum field of the QA domain -- its own equilibrium.

Inside a perfectly conducting wall the current-free, divergence-free field
with ``B . n = 0`` and one unit of toroidal flux is the harmonic 2-form of
the Dirichlet de Rham complex: ``curl B = 0``, ``div B = 0``, tangential to
the wall. QA (``LandremanPaul2021_QA``) is a **vacuum** equilibrium -- its
pressure is zero and its current vanishes -- so this harmonic 2-form *is*
the equilibrium field of the loaded state, reconstructed here from the
bounded geometry alone, with no reference to the field the file stored.

MRX builds it by a direct Hodge decomposition of a seed field
(``mrx.nullspace.compute_nullspaces``) -- two Hodge solves, no eigenvalue
iteration -- and stores it on the sequence's operators, where the Leray
projection and the helicity of the relaxation are deflated against it.

It runs in the package default, **float32**. The residuals it prints are near
total cancellations -- ``curl B`` and ``L B`` of a nearly harmonic ``B`` -- so
float32 cannot resolve them: ``||curl B|| / ||B||`` floors around 1e-2 and the
Rayleigh quotient (which equals ``(||curl B|| / ||B||)^2``) around 1e-3, set by
``eps_float32`` times the largest Laplacian eigenvalue -- NOT by any error in
the field. In float64 the SAME discrete form gives ``||curl B|| / ||B|| ~ 1e-6``
and Rayleigh ~1e-12: the field is essentially exactly harmonic. Run
``MRX_DTYPE=float64`` to see that; the Poincare sections are unaffected either
way.

This script builds the field, verifies ``div``, ``curl`` and the Rayleigh
quotient, draws ``|B|`` on the torus (the 2-form pushed forward by Piola),
and takes Poincare sections at five toroidal planes with the rotational
transform.

    python -u scripts/tutorials/2_qa_vacuum_field.py
"""

# %%
# Now we read the run's options and make the output folder.
from __future__ import annotations

import argparse
import os
import sys

# Run the cells top to bottom in a notebook / VS Code interactive window,
# or the whole file as a script (the CLI flags below still apply then).
_INTERACTIVE = "ipykernel" in sys.modules

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--geometry", default="data/wout_LandremanPaul2021_QA_lowres.nc",
                help="a VMEC wout (.nc) or a GVEC state file (.dat)")
ap.add_argument("--ns", default="12,24,12")
ap.add_argument("--p", type=int, default=3)
ap.add_argument("--cuts", type=int, default=6)
ap.add_argument("--periods", type=int, default=200, help="field periods per traced line")
ap.add_argument("--seeds", type=int, default=24)
ap.add_argument("--out", default="outputs/tutorials/qa_vacuum_field")
cli = ap.parse_args([] if _INTERACTIVE else None)
ns = tuple(int(v) for v in cli.ns.split(","))
os.makedirs(cli.out, exist_ok=True)

# %%
# Now we import MRX -- the sequence builder, the Hodge-decomposition
# nullspace tools, and the Poincare tracer.
import jax.numpy as jnp
import matplotlib
if not _INTERACTIVE:
    matplotlib.use("Agg")  # headless as a script; a notebook keeps its inline backend
import matplotlib.pyplot as plt
import numpy as np
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.geometry import build_sequence
from mrx.nullspace import compute_nullspaces, get_nullspace, harmonic_rayleigh
from mrx.plotting import plot_torus, section_figure, torus_grids
from mrx.poincare import trace_sections
from mrx.relaxation import compute_divergence_norm


def save(fig, name):
    """Write the figure to the output folder; show it too in a notebook."""
    path = os.path.join(cli.out, name)
    fig.savefig(path, dpi=200)
    plt.show() if _INTERACTIVE else plt.close(fig)
    print(f"  -> {path}")

seq, ops = build_sequence(cli.geometry, ns, cli.p)
nfp = seq.equilibrium["nfp"]
compute_nullspaces(seq)

# %%
# Now we build the vacuum field: the harmonic 2-form of the Dirichlet complex,
# from a direct Hodge decomposition of a seed field. We report its divergence,
# curl and Rayleigh quotient (the docstring explains why these read large in
# float32).
B = get_nullspace(seq.get_operators(), 2, True)[0]
B = B / float(seq.l2_norm(B, 2))
ratio = float(seq.l2_norm(seq.apply_weak_curl(B), 1))
rayleigh = float(harmonic_rayleigh(seq, B, 2))
print(f"[vacuum] ||div B|| = {compute_divergence_norm(B, seq):.2e}, "
      f"||curl B|| / ||B|| = {ratio:.2e}, "
      f"Rayleigh quotient of the Hodge Laplacian = {rayleigh:.2e}")
if ratio > 1e-4:
    print("[vacuum] float32 floors these near-zero cancellations; the field is fine "
          "-- float64 gives ||curl B||/||B|| ~ 1e-6, Rayleigh ~ 1e-12 (see the docstring).")

# %%
# Now we push the 2-form forward by the Piola map and draw |B| on the torus.
B_phys = Pushforward(DiscreteFunction(B, seq.basis_2, seq.E(2, True)), seq.map, 2)

def B_mag(x):
    return jnp.linalg.norm(B_phys(x))

# Probe next to the axis, not on it: the map is singular at rho = 0 and
# the Piola factor DF / det DF is 0 / 0 there.
print(f"[vacuum] |B| near the axis {float(B_mag(jnp.array([0.02, 0.0, 0.0]))):.4f}, "
      f"inboard/outboard midplane at zeta = 0: "
      f"{float(B_mag(jnp.array([0.99, 0.5, 0.0]))):.4f} / {float(B_mag(jnp.array([0.99, 0.0, 0.0]))):.4f} "
      f"(||B||_M = 1)")
zetas, grids_pol, grid_surface = torus_grids(seq.map, cli.cuts)
fig, _ = plot_torus(B_mag, grids_pol, grid_surface, cstride=8, gridlinewidth=0.3,
                    elev=25, azim=40, cbar_label=r"$|B|$")
save(fig, "torus_Bmag.png")

# %%
# Now we trace the field lines once and take Poincare sections at five toroidal
# planes over half a field period -- one integration of the trajectories, cut
# at each plane.
res, info = trace_sections(seq, B, nfp, n_seeds=cli.seeds, n_periods=cli.periods)
for plane in (0.0, 0.125, 0.25, 0.375, 0.5):
    fig, _ = section_figure(
        seq, res, plane, title=f"vacuum field {ns} p={cli.p}", nfp=nfp,
        subtitle=(f"nfp = {nfp}   |   h/2 drift {res['drift']:.1e}   |   "
                  f"$B^\\zeta/|B|$ in [{info['bz_over_b_min']:+.2e}, {info['bz_over_b_max']:+.2e}]"))
    save(fig, f"poincare_zeta{plane:g}.png")

regular = ~(res["escaped"] | ~res["ok"] | res["chaotic"])
r_reg, iota_reg = res["seeds"][:, 0][regular], res["iota"][regular]
print(f"[vacuum] {int(regular.sum())}/{regular.size} regular lines, iota from "
      f"{float(iota_reg[np.argmin(r_reg)]):.4f} (r = {float(r_reg.min()):.2f}) to "
      f"{float(iota_reg[np.argmax(r_reg)]):.4f} (r = {float(r_reg.max()):.2f}); "
      f"h/2 drift {res['drift']:.1e}")
