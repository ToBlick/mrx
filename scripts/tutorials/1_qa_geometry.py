"""Tutorial 1: load the equilibrium of a stellarator.

``LandremanPaul2021_QA`` is a vacuum equilibrium with two field periods.
The defaults load the stellarator at resolution (12, 16, 16) p=2. --geometry.path can
point at any VMEC .nc, GVEC .dat or DESC .h5 file instead.

This tutorial can be run either interactively in VS Code (cell by cell), or as a script

    python -u scripts/tutorials/1_qa_geometry.py
    python -u scripts/tutorials/1_qa_geometry.py --geometry.path data/desc_LandremanPaul2021_QA.h5
    python -u scripts/tutorials/1_qa_geometry.py --geometry.resolution 12 24 24

The DESC file is the same QA stellarator (``precise_QA_output.h5`` from the DESC examples).
It will visualize the geometry and the Jacobian of the map defining it.
"""

# %%
# 1) Setup: CLI and input handling.
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

print("[tutorial 1] the geometry of the QA stellarator: importing JAX and MRX", flush=True)

import tyro

from mrx.precision import current_precision
from mrx.relaxation.config import Geometry, Precision

_INTERACTIVE = "ipykernel" in sys.modules


@tyro.conf.configure(tyro.conf.EnumChoicesFromValues)
@dataclass(frozen=True)
class Options:
    """Tutorial 1: load the equilibrium of a stellarator and draw the Jacobian of its map."""
    geometry: Geometry = Geometry(path="data/wout_LandremanPaul2021_QA_lowres.nc", resolution=(12, 16, 16),
                                  spline_degree=2, precision=Precision(current_precision()))
    cuts: int = 6
    """The number of poloidal cuts per field period."""
    out: str = "outputs/tutorials/1_qa_geometry"
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
import jax
import jax.numpy as jnp
import matplotlib
if not _INTERACTIVE:
    matplotlib.use("Agg")  # headless as a script. A notebook keeps its inline backend.
import matplotlib.pyplot as plt
import numpy as np
from mrx.geometry import build_sequence
from mrx.equilibria import read_equilibrium
from mrx.diagnostics.plotting import get_2d_grids, plot_crossections_separate, plot_torus

# %%
# 3) Inspect the input file using ``read_equilibrium``. The file stores: R, Z and lambda, the flux,
# iota and pressure profiles. MRX's angles are right-handed. A VMEC file's are not, so its poloidal angle is
# reversed on reading and its iota changes sign.
st = read_equilibrium(g.path)
nfp = st["nfp"]
X1 = st["X1"]
print(f"[file] {g.path}: nfp = {nfp}, {len(X1['m'])} Fourier modes with m <= {X1['m'].max()}, "
      f"|n| <= {abs(X1['n']).max()}, each read in as a radial spline of the reader "
      f"({X1['cos'].shape[1]} functions of degree {X1['deg']}, independent of the mesh)")
prof = st["profiles"]
print(f"[file] profiles: Psi_edge = {2 * np.pi * prof['phi'](1.0):.4f} Wb, "
      f"iota {prof['iota'](0.0):+.4f} -> {prof['iota'](1.0):+.4f} (per full turn"
      f"{', theta reversed' if st['theta_reversed'] else ''}), "
      f"p_axis = {prof['pressure'](0.0):.4g} Pa, p_edge = {prof['pressure'](1.0):.4g} Pa")
if "a_minor" in st:
    print(f"[file] a = {st['a_minor']:.3f} m, R0 = {st['r_major']:.3f} m")

# %%
# 4) Build the de Rham sequence on that geometry: the spline spaces, operators and preconditioners that
# every solve uses.
print("[seq] building the de Rham sequence: the map, the operators and the preconditioners", flush=True)
seq, _ = build_sequence(g.path, g.resolution, g.spline_degree, knots=g.knots, symmetry=str(g.symmetry))
jac = np.asarray(seq.geometry.jacobian_j)
free = seq.free
print(f"[seq] resolution {g.resolution}, p = {g.spline_degree}: {free.n(0)} 0-form, {free.n(1)} 1-form, "
      f"{free.n(2)} 2-form, {free.n(3)} 3-form DoFs (free spaces), det DPhi at the quadrature points in "
      f"[{jac.min():.3e}, {jac.max():.3e}]")
# seq.map takes a logical point (r, theta, zeta) to the Cartesian point (x, y, z) in m. At zeta = 0 we evaluate
# it on the axis (r = 0) and on the boundary (r = 1) at the outboard (theta = 0) and inboard (theta = 1/2)
# midplane, and print the major radius R = sqrt(x^2 + y^2) of each.
R_axis, R_out, R_in = (float(jnp.hypot(*seq.map(jnp.array(x))[:2])) for x in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                                                             (1.0, 0.5, 0.0)))
print(f"[seq] the map at zeta = 0: magnetic axis at R = {R_axis:.4f} m, boundary from R = {R_in:.4f} m (inboard) "
      f"to {R_out:.4f} m (outboard)")

# %%
# 5) Draw the Jacobian det DPhi of the map on the torus using ``plot_torus``.
def detDPhi(x):
    return jnp.linalg.det(jax.jacfwd(seq.map)(x))

zetas = np.arange(cli.cuts) / cli.cuts
n = 48
grids_pol = [get_2d_grids(seq.map, cut_axis=2, cut_value=float(z), nx=n, ny=n, nz=1)
             for z in zetas]
grid_surface = get_2d_grids(seq.map, cut_axis=0, cut_value=1.0 - 1e-6,
                            ny=4 * n, nz=4 * n, invert_z=True)
print(f"[plot] det DPhi on the torus and on {cli.cuts} poloidal cuts", flush=True)
fig, _ = plot_torus(detDPhi, grids_pol, grid_surface, cbar_label=r"$\det D\Phi$")
path = os.path.join(cli.out, "torus_jacobian.png")
fig.savefig(path, dpi=200)
if _INTERACTIVE:
    plt.show()
else:
    plt.close(fig)
print(f"  -> {path}")
fig, _ = plot_crossections_separate(detDPhi, grids_pol, zetas)
path = os.path.join(cli.out, "crossections_jacobian.png")
fig.savefig(path, dpi=200)
if _INTERACTIVE:
    plt.show()
else:
    plt.close(fig)
print(f"  -> {path}")
