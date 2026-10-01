"""Tutorial 2: Compute a vacuum equilibrium state.

There is exactly one field in a solid toroidal domain with ``curl B = div B = 0``
and ``B . n = 0`` on the boundary, up to a scale. We compute this vacuum/harmonic
field for the ``LandremanPaul2021_QA`` stellarator.

The vacuum field can be defined either in the 1-form space without essential
boundary conditions or in the 2-form space with homogeneous Dirichlet boundary
conditions. We do the latter.

The domain is stellarator-symmetric, so the sequence integrates over half a field
period. This requires knowledge of the parity. ``B``, ``J`` and the harmonic
2-form are odd and hence live on ``seq.odd``.

This script computes the field, verifies ``div``, ``curl`` and the Rayleigh
quotient, draws ``|B|`` on the torus, and plots Poincare sections at five toroidal planes.
The sections trace ``--lines`` field lines (160 by default, as in the paper's vacuum figure) for ``--periods`` field periods
(400 by default).

    python -u scripts/tutorials/2_qa_vacuum_field.py
    python -u scripts/tutorials/2_qa_vacuum_field.py --lines 48 --periods 200
"""

# %%
# 1) Setup. The defaults are the QA stellarator at (12, 16, 16) p=2, as in Tutorial 1.
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

print("[tutorial 2] the vacuum field of the QA stellarator: importing JAX and MRX", flush=True)

import tyro

from mrx.precision import current_precision
from mrx.relaxation.config import Geometry, Precision

_INTERACTIVE = "ipykernel" in sys.modules


@tyro.conf.configure(tyro.conf.EnumChoicesFromValues)
@dataclass(frozen=True)
class Options:
    """Tutorial 2: the vacuum field of the QA domain as a harmonic 2-form, and its Poincare sections."""
    geometry: Geometry = Geometry(path="data/wout_LandremanPaul2021_QA_lowres.nc", resolution=(12, 16, 16),
                                  spline_degree=2, precision=Precision(current_precision()))
    cuts: int = 6
    """The number of poloidal cuts per field period in the torus figure."""
    periods: int = 400
    """The number of field periods each line is traced."""
    lines: int = 160
    """The number of field lines in the Poincare sections."""
    out: str = "outputs/tutorials/2_qa_vacuum_field"
    """The directory for the figures."""


cli = tyro.cli(Options, args=[] if _INTERACTIVE else None)
g = cli.geometry
# mrx fixes its precision when it is imported, from MRX_DTYPE and MRX_RESIDUAL_DTYPE
if g.precision != current_precision():
    sys.exit(f"--geometry.precision {g.precision} needs MRX_DTYPE and MRX_RESIDUAL_DTYPE set before mrx is "
             f"imported (this run is {current_precision()})")
os.makedirs(cli.out, exist_ok=True)

# %%
# 2) Import MRX.
import jax.numpy as jnp
import matplotlib
if not _INTERACTIVE:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.nullspace import compute_nullspaces, harmonic_rayleigh
from mrx.diagnostics.plotting import get_2d_grids, plot_archive, plot_torus
from mrx.diagnostics.poincare import trace_archive
from mrx.relaxation.physics import compute_divergence_norm, compute_force

# Geometry.build calls mrx.geometry.build_sequence with these options, as in Tutorial 1.
print("[seq] building the de Rham sequence: the map, the operators and the preconditioners", flush=True)
seq, _ = g.build()
nfp = seq.nfp

# %%
# 3) Compute the vacuum field: the harmonic 2-form of the Dirichlet complex. Since this field
# is in the kernel of the Hodge Laplacian, it needs to be computed in any case and there
# is a convenience function for it: ``compute_nullspaces(seq)``. There are four spaces with harmonic
# forms: the constant function (k=0 free, k=3 Dirichlet) and the vacuum field (k=1 free, k=2 Dirichlet).
print("[vacuum] computing the harmonic forms", flush=True)
compute_nullspaces(seq)
B = seq.odd.nullspace(2)[0]
B = B / float(seq.odd.l2_norm(B, 2))
_, _, J, _ = compute_force(B, seq)
ratio = float(seq.odd.l2_norm(J, 1))
rayleigh = float(harmonic_rayleigh(seq.odd, B, 2))
print(f"[vacuum] {seq.odd.n(2)} odd Dirichlet 2-form DoFs, ||div B|| = {compute_divergence_norm(B, seq):.2e}, "
      f"||curl B|| / ||B|| = {ratio:.2e}, "
      f"Rayleigh quotient of the Hodge Laplacian = {rayleigh:.2e}")

# %%
# 4) Push the 2-form forward to the physical space and draw |B| on the torus. Plot using ``plot_torus``.
B_phys = Pushforward(DiscreteFunction(B, seq.odd.basis_2, seq.odd.E(2)), seq.map, 2)


def B_mag(x):
    return jnp.linalg.norm(B_phys(x))


print(f"[vacuum] |B| near the axis {float(B_mag(jnp.array([0.01, 0.0, 0.0]))):.4f}, "
      f"inboard/outboard midplane at zeta = 0: "
      f"{float(B_mag(jnp.array([0.99, 0.5, 0.0]))):.4f} / {float(B_mag(jnp.array([0.99, 0.0, 0.0]))):.4f} "
      f"(||B||_M = 1)")
zetas = np.arange(cli.cuts) / cli.cuts
n = 48
grids_pol = [get_2d_grids(seq.map, cut_axis=2, cut_value=float(z), nx=n, ny=n, nz=1)
             for z in zetas]
grid_surface = get_2d_grids(seq.map, cut_axis=0, cut_value=1.0 - 1e-6,
                            ny=4 * n, nz=4 * n, invert_z=True)
print("[plot] |B| on the torus", flush=True)
fig, _ = plot_torus(B_mag, grids_pol, grid_surface, cbar_label=r"$|B|$")
path = os.path.join(cli.out, "torus_Bmag.png")
fig.savefig(path, dpi=200)
if _INTERACTIVE:
    plt.show()
else:
    plt.close(fig)
print(f"  -> {path}")

# %%
# 5) Trace the field lines once and take Poincare sections at five planes with ``trace_archive``, which
# writes the same trace.npz as scripts/poincare_trace.py. ``plot_archive`` draws it as scripts/poincare_plot.py
# does, in the layout of the paper (the vacuum field has no pressure to show).
print(f"[poincare] tracing {cli.lines} field lines for {cli.periods} field periods", flush=True)
archive, results = trace_archive(seq, {"vacuum": (B, 0)}, lines=cli.lines, periods=cli.periods,
                                 source=f"{g.path} {g.resolution} p={g.spline_degree}, vacuum field")
np.savez_compressed(os.path.join(cli.out, "trace.npz"), **archive)
for fig in plot_archive(archive, os.path.join(cli.out, "poincare"), pressure=False).values():
    if _INTERACTIVE:
        plt.show()
    else:
        plt.close(fig)
res = results["vacuum"]

# the lines near the axis circle the magnetic axis (off r = 0), so iota is read from r >= 0.1
regular = res["shown"]
print(f"[vacuum] {int(regular.sum())}/{regular.size} regular lines")
regular = regular & (res["seed_r"] >= 0.1)
r_reg, iota_reg = res["seed_r"][regular], res["iota"][regular]
print(f"[vacuum] iota from "
      f"{float(iota_reg[np.argmin(r_reg)]):.4f} (r = {float(r_reg.min()):.2f}) to "
      f"{float(iota_reg[np.argmax(r_reg)]):.4f} (r = {float(r_reg.max()):.2f}), "
      f"h/2 drift {res['drift']:.1e}")
