"""Tutorial 1: load a VMEC state of the QA stellarator and look at it.

The file is a VMEC ``wout_*.nc``: the boundary and interior flux surfaces
as ``R`` and ``Z`` Fourier series in the angles ``(theta, zeta)`` -- ``zeta``
spans ONE field period, ``nfp`` completes the torus -- with the stream
function ``lambda`` and the profiles ``Phi``, ``chi``, ``iota``, ``p`` on
the radial grid. ``mrx.vmec`` refits each Fourier mode into a clamped
B-spline in ``rho = sqrt(s)`` so the wout arrives in the same radial-splines
x Fourier-series blocks a GVEC state (``GVEC_State_*.dat``) is read into, and
everything downstream is closed form: MRX evaluates the series wherever it
needs a value, there is no grid in between. The very same
``build_sequence`` call reads a GVEC ``.dat`` state instead by just pointing
it at the ``.dat`` file (see ``docs/source/concepts/gvec_mrx_interface.md``).

``LandremanPaul2021_QA`` is a quasi-axisymmetric two-field-period (``nfp=2``)
**vacuum** equilibrium: the pressure is zero, so the field it carries is a
current-free vacuum field (tutorial 3 rebuilds exactly that field from the
geometry alone). Because ``p = 0`` there is nothing to colour a pressure plot
with, so this tutorial draws the map's Jacobian ``det DF`` -- the volume
element of the mapped domain, larger on the outboard side of the torus and
squeezed on the inboard side -- which is a property of the geometry itself.

``build_sequence`` turns the file into an MRX domain: it builds the spline
coefficients of ``R`` and ``Z`` on the map's own spline space (resolution
``ns``, degree ``p``, polar at the axis, periodic in both angles) from the
series coefficients, mode by mode, measures the toroidal handedness so that
``det DF > 0``, installs the map
``F(rho, theta, zeta) = (R cos phi, +-R sin phi, Z)`` with
``phi = 2 pi zeta / nfp``, and assembles the operators and preconditioners
of the de Rham sequence on it. Everything else in MRX -- the Poisson solve,
the vacuum field, the relaxation -- starts from this ``seq``.

    python -u scripts/tutorials/qa_geometry.py
    python -u scripts/tutorials/qa_geometry.py --geometry data/GVEC_State_final.dat --ns 12,24,24 --p 3
"""

# %%
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
ap.add_argument("--ns", default="12,24,12", help="map and sequence resolution")
ap.add_argument("--p", type=int, default=3)
ap.add_argument("--cuts", type=int, default=6, help="poloidal cuts per field period")
ap.add_argument("--out", default="outputs/tutorials/qa_geometry")
cli = ap.parse_args([] if _INTERACTIVE else None)
ns = tuple(int(v) for v in cli.ns.split(","))
os.makedirs(cli.out, exist_ok=True)

# %%
import jax
import jax.numpy as jnp
import matplotlib
if not _INTERACTIVE:
    matplotlib.use("Agg")  # headless as a script; a notebook keeps its inline backend
import matplotlib.pyplot as plt
import numpy as np
from mrx.geometry import build_sequence
from mrx.gvec import load_clebsch, read_equilibrium
from mrx.plotting import get_2d_grids, plot_crossections_separate, plot_torus

# %%
# --- 1. what the file holds --------------------------------------------
# R, Z, lambda as radial B-splines x Fourier series; the profiles at the
# radial interpolation points. A GVEC .dat lands in the same blocks.
st = read_equilibrium(cli.geometry)
nfp = st["nfp"]
X1 = st["X1"]
print(f"[file] {cli.geometry}: nfp = {nfp}, radial B-splines of degree {st['deg']} "
      f"({X1['coef'].shape[1]} functions), {len(X1['m'])} Fourier modes with "
      f"m <= {X1['m'].max()}, |n| <= {abs(X1['n']).max()}")
# load_clebsch tabulates the profile splines on 401 uniform radii: the
# flux derivative dPhi, chi' = iota Phi' and the pressure. The same dict
# feeds the relaxation's initial condition (tutorial 4, on li383).
cb = load_clebsch(cli.geometry)
iota = cb["dchi"][1:] / cb["dPhi"][1:]
print(f"[file] profiles: Phi_edge = {2 * np.pi * st['profiles']['phi'][-1]:.4f} Wb, "
      f"iota {iota[0]:+.4f} -> {iota[-1]:+.4f} (per full turn), "
      f"p_axis = {cb['p'][0]:.4g} Pa, p_edge = {cb['p'][-1]:.4g} Pa (a vacuum field)")
if "a_minor" in st:
    print(f"[file] a = {st['a_minor']:.3f} m, R0 = {st['r_major']:.3f} m")

# %%
# --- 2. the sequence on the file's geometry -----------------------------
seq, _ = build_sequence(cli.geometry, ns, cli.p)
jac = np.asarray(seq.geometry.jacobian_j)
print(f"[seq] ns = {ns}, p = {cli.p}: {seq.n(0)} 0-form, {seq.n(1)} 1-form, {seq.n(2)} 2-form, "
      f"{seq.n(3)} 3-form DoFs (free spaces); det DF at the quadrature points in "
      f"[{jac.min():.3e}, {jac.max():.3e}]")
x = jnp.array([0.5, 0.25, 0.0])
print(f"[seq] the map at logical {np.asarray(x)}: physical {np.asarray(seq.map(x)).round(4)} m")

# %%
# --- 3. the Jacobian of the map on the torus -----------------------------
# det DF is the volume element the whole de Rham complex is weighted by;
# for a vacuum equilibrium with no pressure to draw it is the cleanest
# look at the geometry itself.
def detDF(x):
    return jnp.linalg.det(jax.jacfwd(seq.map)(x))

zetas = np.arange(cli.cuts) / cli.cuts
n = 48
grids_pol = [get_2d_grids(seq.map, cut_axis=2, cut_value=float(z), nx=n, ny=n, nz=1)
             for z in zetas]
grid_surface = get_2d_grids(seq.map, cut_axis=0, cut_value=1.0 - 1e-6,
                            ny=4 * n, nz=4 * n, invert_z=True)
fig, _ = plot_torus(detDF, grids_pol, grid_surface, cstride=8, gridlinewidth=0.3,
                    elev=25, azim=40, cbar_label=r"$\det DF$")
path = os.path.join(cli.out, "torus_jacobian.png")
fig.savefig(path, dpi=200)
if not _INTERACTIVE:
    plt.close(fig)  # keep the figure open in a notebook so it renders inline
print(f"  -> {path}")
fig, _ = plot_crossections_separate(detDF, grids_pol, zetas)
path = os.path.join(cli.out, "crossections_jacobian.png")
fig.savefig(path, dpi=200)
if not _INTERACTIVE:
    plt.close(fig)  # keep the figure open in a notebook so it renders inline
print(f"  -> {path}")

