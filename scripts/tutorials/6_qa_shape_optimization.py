"""Tutorial 6: Shape optimization for quasi-axisymmetry in vacuum.

Tutorial 2 computed the vacuum field of a given domain. Here we change the domain to find a specific 
vacuum field.

The problem is

    min <Q_QA^2>_{r >= h_r}   s.t.   <iotabar>_s = iotabar*,   P <= 1e-8,

at the volume and aspect ratio of the device, which are held exactly by rescaling the map after every
optimization step. 

- ``Q_QA`` is the quasi-axisymmetry residual of the field, averaged outside the first radial knot
span ``h_r``, where the polar map is close to singular. 
- ``<iotabar>_s`` is the mean rotational transform from the flux ratio and ``iotabar*`` its value on the device. 
- ``P`` is the fraction of the field normal to the logical surfaces. Keeping it small keeps them close to flux surfaces.
   MRX does not mind non-flux-aligned grids, but ``Q_QA`` and ``iotabar`` need them. 
   
An augmented Lagrangian over L-BFGS-B imposes both constraints.

The start is the device plus a smooth random normal displacement of its boundary (``--perturb-mm`` RMS), brought
back to the device's volume and aspect ratio like every other shape. Restoring the two moves the boundary as well,
so the start's distance to the device is not ``--perturb-mm``. It depends on the draw and on the planes the aspect
ratio is measured in, and is printed. The run stops when the criterion is back at the device's own value with both constraints met.
Quasi-axisymmetry comes back, but the end boundary still differs from the device by a few percent of the
minor radius, since these three scalars alone do not determine the shape.

    python -u scripts/tutorials/6_qa_shape_optimization.py
    python -u scripts/tutorials/6_qa_shape_optimization.py --perturb-mm 5 --seed 1
"""

# %%
# 1) Setup: CLI and input handling.
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

print("[tutorial 6] shape optimization of the QA stellarator: importing JAX and MRX", flush=True)

import tyro

from mrx.paths import runs_root
from mrx.relaxation.config import Geometry

_INTERACTIVE = "ipykernel" in sys.modules


@tyro.conf.configure(tyro.conf.EnumChoicesFromValues)
@dataclass(frozen=True)
class Options:
    """Tutorial 6: optimize the shape of the QA stellarator for quasi-axisymmetry."""
    geometry: Geometry = Geometry(path="data/wout_LandremanPaul2021_QA_lowres.nc", resolution=(12, 16, 16),
                                  spline_degree=3)
    perturb_mm: float = 10.0
    """The RMS of the random normal displacement of the device's boundary, in mm, before the volume and the aspect
    ratio are restored."""
    seed: int = 0
    """The random draw of the displacement."""
    maxiter: int = 1000
    """The number of L-BFGS-B iterations in all."""
    al_inner: int = 100
    """The number of L-BFGS-B iterations between updates of the multipliers."""
    stop_qa: float = 1.0
    """Stop once the criterion is at most this times the device's value with both constraints met. 0 runs --maxiter iterations."""
    out: str = os.path.join(runs_root(), "tutorials", "6_qa_shape_optimization")
    """The directory for the figures."""


cli = tyro.cli(Options, args=[] if _INTERACTIVE else None)
g = cli.geometry
os.makedirs(cli.out, exist_ok=True)

# %%
# 2) Import MRX.
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from mrx.diagnostics.plotting import plot_twin_axis
from mrx.diagnostics.plotstyle import finish_figure
from mrx.optimization.shape_ad import (BoundaryShape, flux_seed, frame_geometry, mean_iota,
                                       normal_field_fraction, quasisymmetry_residual, section_moments,
                                       vacuum_two_form)
from mrx.spline_bases import basis_derivative_table, basis_table

BETA_SCALE = 0.01    # metres of coefficient change per unit of the variables
C_UNIT = 1e-3        # the unit of the iota constraint, c = (<iotabar>_s - iotabar*) / C_UNIT
P_MAX = 1e-8         # the bound on P
IOTA_TOL, P_TOL = 1e-5, 1.01e-8   # met when |<iotabar>_s - iotabar*| <= IOTA_TOL and P <= P_TOL

# %%
# 3) Build the sequence on the device's VMEC file (``seq.map``, a frame map in the file's cylindrical frame: the
# coordinates X1 = R and X2 = Z in the planes phi = const have (n_r, n_theta, n_zeta) spline coefficients each).
# X3, out of the planes, is zero and stays zero here.
t0 = time.perf_counter()
print("[seq] building the de Rham sequence: the map, the operators and the preconditioners", flush=True)
seq = g.build()
nfp = seq.nfp
lp = seq.map
V_lp, _, S_lp = section_moments(seq, lp)
a_lp = float(np.sqrt(S_lp / np.pi))
# The aspect ratio in the cross-sections of the map, the planes phi = const: 6 for this device.
aspect_lp = float(nfp * V_lp / (2.0 * np.sqrt(np.pi) * S_lp ** 1.5))
T = np.asarray(seq.basis_0.Lambda[0].T)
r_min = float(np.min(T[T > 0.0]))   # h_r, the end of the first radial knot span

# %%
# 4) Set up the problem. The variables x are the change of every coefficient, beta = BETA_SCALE x. The
# change of the boundary ring is extended harmonically into the interior, and every inner ring changes on
# its own too. map_at rescales the map to the device's volume and aspect ratio, so that x = 0 is the device. The
# field is h = seed - curl A on the changed map, and from it come the criterion, the mean iota and P.
shape = BoundaryShape.from_map(seq, lp, aspect=aspect_lp)
seed = flux_seed(seq)
n_var = 2 * lp.raw_X1.size


def mapped(x, seq, shape):
    """The map at the variables x."""
    return shape.map_at(seq, BETA_SCALE * jnp.reshape(x, (2,) + tuple(shape.start.raw_X1.shape)))


def terms(x, seq, shape, seed):
    """<Q_QA^2>_{r >= h_r}, <iotabar>_s (signed), P and min det DPhi at x."""
    Phi = mapped(x, seq, shape)
    sq = seq.with_geometry(frame_geometry(seq, Phi))
    h, _ = vacuum_two_form(sq, seed)
    F_qs, _ = quasisymmetry_residual(sq, h, Phi, r_min)
    return dict(F_qs=F_qs, iota=mean_iota(sq, h), P=normal_field_fraction(sq, h), jmin=jnp.min(sq.jacobian_j))


def lagrangian(x, seq, shape, seed, al):
    """The augmented Lagrangian f + nu_i c + mu c^2 / 2 + (max(0, nu_P + mu g)^2 - nu_P^2) / (2 mu), in the
    scaled f = <Q_QA^2> / F_lp (the device's value), c = (<iotabar>_s - iotabar*) / C_UNIT and g = (P - P_MAX) / P_MAX."""
    aux = terms(x, seq, shape, seed)
    f = aux["F_qs"] / al["F_lp"]
    c = (al["orient"] * aux["iota"] - al["target"]) / C_UNIT
    g = (aux["P"] - P_MAX) / P_MAX
    mu = al["mu"]
    return (f + al["nu_i"] * c + 0.5 * mu * c ** 2
            + (jnp.maximum(0.0, al["nu_P"] + mu * g) ** 2 - al["nu_P"] ** 2) / (2.0 * mu)), aux


forward = eqx.filter_jit(terms)
value_and_grad = eqx.filter_jit(jax.value_and_grad(lagrangian, has_aux=True))


@eqx.filter_jit
def min_det_dphi(x, seq, shape):
    """min det DPhi of the map at x on the quadrature points. The map folds when det DPhi <= 0."""
    return jnp.min(frame_geometry(seq, mapped(x, seq, shape)).jacobian_j)


# %%
# 5) Evaluate the device itself (x = 0). Its criterion F_lp is the baseline and its mean iota the
# target iotabar*.
print("[setup] compiling the vacuum field and the criterion", flush=True)
lp_terms = forward(jnp.zeros(n_var), seq, shape, seed)
orient = float(jnp.sign(lp_terms["iota"]))
F_lp, target = float(lp_terms["F_qs"]), orient * float(lp_terms["iota"])
print(f"[setup] QA {g.resolution} p={g.spline_degree}, {n_var} variables, {time.perf_counter() - t0:.0f} s. "
      f"Device: <Q_QA^2>_(r >= {r_min:.3f}) = {F_lp:.3e}, <iotabar>_s = {target:.5f}, "
      f"P = {float(lp_terms['P']):.1e}, a = {a_lp:.4f} m")


# %%
# 6) Perturb the device's boundary: a random smooth displacement along the normal, as a change of the
# boundary ring collocated at the Greville points. boundary_points evaluates the boundary surface and its
# angular derivatives at any logical angles.
def boundary_points(Phi, theta, zeta):
    """(F, F_theta, F_zeta) of the boundary surface r = 1 of the map Phi at the logical angles: X1, X2, X3 from the
    boundary ring of coefficients, placed in the frame, P = a + X1 N + X2 B + X3 T, and turned to the field period,
    F = R_z(phi) P."""
    lt, lz = seq.basis_0.Lambda[1], seq.basis_0.Lambda[2]
    t, z = jnp.asarray(np.mod(theta, 1.0)), jnp.asarray(np.mod(zeta, 1.0))
    Bt, Bz = basis_table(lt, t), basis_table(lz, z)
    Dt, Dz = basis_derivative_table(lt, t), basis_derivative_table(lz, z)

    def ev(C, A, B):
        return jnp.einsum("jp,jk,kp->p", A, C[-1], B)[:, None]
    raws = (Phi.raw_X1, Phi.raw_X2, Phi.raw_X3)
    (a, N, B), (da, dN, dB), _ = (tuple(f[:, i] for i in range(3)) for f in jax.vmap(Phi.frame_derivatives)(z))
    E, dE = (N, B, jnp.cross(N, B)), (dN, dB, jnp.cross(dN, B) + jnp.cross(N, dB))
    c = 2.0 * np.pi / nfp
    P = a + sum(ev(C, Bt, Bz) * e for C, e in zip(raws, E))
    Pt = sum(ev(C, Dt, Bz) * e for C, e in zip(raws, E))
    Pz = (da + sum(ev(C, Bt, Bz) * de + ev(C, Bt, Dz) * e for C, e, de in zip(raws, E, dE))
          + c * jnp.stack([-P[:, 1], P[:, 0], jnp.zeros_like(P[:, 0])], -1))
    # the turn by the angle itself, not its remainder: the search in distance() may leave the field period
    co, si = jnp.cos(c * jnp.asarray(zeta)), jnp.sin(c * jnp.asarray(zeta))

    def turn(v):
        return np.asarray(jnp.stack([co * v[:, 0] - si * v[:, 1], si * v[:, 0] + co * v[:, 1], v[:, 2]], -1))
    return turn(P), turn(Pt), turn(Pz)


def perturbation(amplitude, draw, m_max=4, n_max=4):
    """delta = sum a_mn cos 2 pi (m theta - n zeta), a_mn ~ N(0, 1) / (1 + m^2 + n^2), scaled to the
    area-weighted RMS amplitude (m) and carried by (dX1, dX2) along the part of the normal in the plane of the
    cross-section. Returns the change of the boundary ring."""
    gt, gz = (np.asarray(seq.greville[a].point_rule[0][:, 0]) for a in (1, 2))
    th, ze = (v.reshape(-1) for v in np.meshgrid(gt, gz, indexing="ij"))
    rng = np.random.default_rng(draw)
    delta = np.zeros_like(th)
    for m in range(m_max + 1):
        for n in range(-n_max, n_max + 1):
            delta += rng.standard_normal() / (1.0 + m ** 2 + n ** 2) * np.cos(2.0 * np.pi * (m * th - n * ze))
    _, Ft, Fz = boundary_points(lp, th, ze)
    normal = np.cross(Ft, Fz)
    area = np.linalg.norm(normal, axis=-1)
    _, e1, e2 = (np.asarray(v) for v in jax.vmap(lp.plane)(jnp.asarray(ze)))
    n_R, n_Z = np.sum(normal * e1, -1), np.sum(normal * e2, -1)
    delta *= amplitude / np.sqrt(np.sum(area * delta ** 2) / np.sum(area))
    t = delta * area / (n_R ** 2 + n_Z ** 2)
    ct, cz = np.asarray(seq.greville[1].coll), np.asarray(seq.greville[2].coll)

    def collocate(values):
        return np.linalg.solve(ct, np.linalg.solve(cz, values.reshape(gt.size, gz.size).T).T)
    return np.stack([collocate(t * n_R), collocate(t * n_Z)])


def distance(Phi, grid=(128, 64), iterations=8):
    """d_RMS of the boundary of the map Phi to the device's. For every point x of it on a uniform
    grid of logical angles, the normal offset (x - x_dev(u*)) . n_dev(u*) at the closest point u* of the
    device's boundary (Gauss-Newton from the same logical angles), and its RMS weighted by the area."""
    th, ze = (v.reshape(-1) for v in np.meshgrid(np.arange(grid[0]) / grid[0], np.arange(grid[1]) / grid[1],
                                                indexing="ij"))
    X, Xt, Xz = boundary_points(Phi, th, ze)
    area = np.linalg.norm(np.cross(Xt, Xz), axis=-1)
    u = np.stack([th, ze], -1)
    for _ in range(iterations):
        F, Ft, Fz = boundary_points(lp, u[:, 0], u[:, 1])
        Jac = np.stack([Ft, Fz], -1)
        u = u + np.linalg.solve(np.einsum("pki,pkj->pij", Jac, Jac),
                                np.einsum("pki,pk->pi", Jac, X - F)[..., None])[..., 0]
    F, Ft, Fz = boundary_points(lp, u[:, 0], u[:, 1])
    nrm = np.cross(Ft, Fz)
    d = np.sum((X - F) * nrm / np.linalg.norm(nrm, axis=-1, keepdims=True), -1)
    return float(np.sqrt(np.sum(area * d ** 2) / np.sum(area)))


full = np.zeros((2,) + tuple(lp.raw_X1.shape))
full[:, -1] = perturbation(1e-3 * cli.perturb_mm, cli.seed)
x = full.reshape(-1) / BETA_SCALE
Phi0 = mapped(jnp.asarray(x), seq, shape)
start = forward(jnp.asarray(x), seq, shape, seed)
if float(start["jmin"]) <= 0.0:
    sys.exit(f"the start map folds (min det DPhi {float(start['jmin']):.2e}): take another --seed")
d_start = distance(Phi0)
print(f"[start] displacement {cli.perturb_mm:g} mm RMS (draw {cli.seed}), at the device's volume and aspect ratio "
      f"d_RMS = {1e3 * d_start:.2f} mm, d_RMS / a = {d_start / a_lp:.3e}, "
      f"<Q_QA^2> = {float(start['F_qs']):.3e} ({float(start['F_qs']) / F_lp:.3g} x the device's), <iotabar>_s = {orient * float(start['iota']):.5f}, "
      f"P = {float(start['P']):.1e}, min det DPhi = {float(start['jmin']):.2e}")

# %%
# 7) Optimize. Each outer step runs --al-inner L-BFGS-B iterations on the augmented Lagrangian, then
# updates the multipliers, nu_i += mu c and nu_P = max(0, nu_P + mu g), and multiplies mu by 10 if the
# violation is above its tolerance and fell less than 4x. A trial point whose map folds returns ten
# times the last value with the last gradient, so the line search backtracks.
al = dict(F_lp=F_lp, target=target, orient=orient, nu_i=0.0, nu_P=0.0, mu=1.0)
history, cache, last = [], {}, {}
violation_tol = min(IOTA_TOL / C_UNIT, (P_TOL - P_MAX) / P_MAX)


def fun(x):
    key = x.tobytes()
    if key not in cache:
        if float(min_det_dphi(jnp.asarray(x), seq, shape)) <= 0.0:
            return last["big"], last["grad"]
        (L, aux), grad = value_and_grad(jnp.asarray(x), seq, shape, seed, {k: jnp.asarray(v) for k, v in al.items()})
        cache[key] = dict(L=float(L), grad=np.asarray(grad), F=float(aux["F_qs"]) / F_lp,
                          iota=orient * float(aux["iota"]), P=float(aux["P"]), jmin=float(aux["jmin"]))
        while len(cache) > 4:
            cache.pop(next(iter(cache)))
    return cache[key]["L"], cache[key]["grad"]


def met(r):
    return r["P"] <= P_TOL and abs(r["iota"] - target) <= IOTA_TOL


def callback(intermediate_result):
    x = np.array(intermediate_result.x)
    fun(x)
    r = cache[x.tobytes()]
    history.append(dict(r, x=x))
    it = len(history)
    if it % 10 == 0:
        print(f"[iter {it:4d}] <Q_QA^2> {r['F'] * F_lp:.3e}  <iotabar>_s {r['iota']:.6f}  P {r['P']:.2e}  "
              f"min det DPhi {r['jmin']:.2e}")
    if (cli.stop_qa and r["F"] <= cli.stop_qa and met(r)) or it >= cli.maxiter:
        raise StopIteration


print(f"[opt] compiling the value and gradient, then up to {cli.maxiter} L-BFGS-B iterations "
      "(a line every 10)", flush=True)
t0 = time.perf_counter()
fun(x)
history.append(dict(cache[x.tobytes()], x=x))
V_prev, outer = None, 0
while True:
    last.update(big=10.0 * abs(cache[x.tobytes()]["L"]) + 1.0, grad=cache[x.tobytes()]["grad"])
    res = scipy.optimize.minimize(fun, x, jac=True, method="L-BFGS-B", callback=callback,
                                  options=dict(maxiter=cli.al_inner, ftol=1e-15, gtol=1e-15, maxcor=100))
    r = history[-1]
    x = r["x"]
    if (cli.stop_qa and r["F"] <= cli.stop_qa and met(r)) or len(history) > cli.maxiter or res.nit == 0:
        break
    c, g_P = (r["iota"] - target) / C_UNIT, (r["P"] - P_MAX) / P_MAX
    V = max(abs(c), abs(max(g_P, -al["nu_P"] / al["mu"])))
    al["nu_i"] += al["mu"] * c
    al["nu_P"] = max(0.0, al["nu_P"] + al["mu"] * g_P)
    if V_prev is not None and V > violation_tol and V > 0.25 * V_prev:
        al["mu"] *= 10.0
    V_prev, outer = V, outer + 1
    cache.clear()
    fun(x)
    print(f"[outer {outer}] iteration {len(history) - 1}: c {c:+.2e}, g {g_P:+.2e}, violation {V:.2e}, "
          f"nu_i {al['nu_i']:+.3e}, nu_P {al['nu_P']:.3e}, mu {al['mu']:g}")
wall = time.perf_counter() - t0
Phi1 = mapped(jnp.asarray(x), seq, shape)
print(f"[end] {len(history) - 1} iterations, {wall:.0f} s: <Q_QA^2> {history[0]['F'] * F_lp:.3e} -> {r['F'] * F_lp:.3e} "
      f"({r['F']:.3f} x the device's {F_lp:.3e}), "
      f"<iotabar>_s {r['iota']:.6f} (iotabar* {target:.6f}), P {r['P']:.2e}, d_RMS / a "
      f"{d_start / a_lp:.3e} -> {distance(Phi1) / a_lp:.3e}")
np.savez(os.path.join(cli.out, "shapes.npz"), **{f"raw_X{i}_{name}": np.asarray(getattr(m, f"raw_X{i}"))
                                                 for name, m in (("lp", lp), ("start", Phi0), ("end", Phi1))
                                                 for i in (1, 2)})

# %%
# 8) Plot the criterion (the device's value dotted) and the mean iota (iotabar*, dotted) against the
# iteration using ``plot_twin_axis``.
F_path = F_lp * np.array([h["F"] for h in history])
iota_path = np.array([h["iota"] for h in history])
fig, (ax_F, ax_i) = plot_twin_axis(F_path, iota_path,
                                   left_label=r"$\langle Q_{\mathrm{QA}}^2 \rangle_{r \geq h_r}$",
                                   right_label=r"$\langle \bar\iota \rangle_s$",
                                   left_plot_kwargs=dict(marker=""), right_plot_kwargs=dict(marker=""))
ax_F.axhline(F_lp, color=ax_F.get_lines()[0].get_color(), ls=":", lw=1)
ax_i.axhline(target, color=ax_i.get_lines()[0].get_color(), ls=":", lw=1)
finish_figure(fig, os.path.join(cli.out, "trace.png"), bbox_inches="tight")

# %%
# 9) Draw the boundary of the device, the start and the end in three cross sections of a field period, in the
# coordinates (X1, X2) = (R, Z) of the frame's plane.
planes = (0.0, 0.25, 0.5)
theta = np.linspace(0.0, 1.0, 257)
fig, axes = plt.subplots(1, len(planes), figsize=(4 * len(planes), 4), constrained_layout=True)
for ax, zeta in zip(axes, planes):
    for m, label, style in ((lp, "device", dict(color="k", lw=1.5)),
                            (Phi0, "start", dict(color="C1", ls="--", lw=1)),
                            (Phi1, "end", dict(color="C0", lw=1))):
        F, _, _ = boundary_points(m, theta, np.full_like(theta, zeta))
        o, e1, e2 = (np.asarray(v) for v in lp.plane(zeta))
        ax.plot((F - o) @ e1, (F - o) @ e2, label=label, **style)
    ax.set_aspect("equal")
    ax.set_title(rf"$\zeta = {zeta:g}$")
    ax.set_xlabel(f"{lp.section_labels[0]} [m]")
axes[0].set_ylabel(f"{lp.section_labels[1]} [m]")
axes[1].legend(frameon=False, loc="upper right")
finish_figure(fig, os.path.join(cli.out, "sections.png"))
