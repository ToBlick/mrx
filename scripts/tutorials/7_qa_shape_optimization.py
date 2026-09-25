"""Tutorial 7: optimize the QA boundary for quasi-axisymmetry, by reverse-mode AD of the vacuum field.

Tutorial 2 solved the vacuum field of a given domain. Here the domain is the
unknown: the paper's shape optimization (Sec. 3.4, App. D) in small. The
vacuum field is one linear solve on the map, ``h = s - curl A``
(``mrx.shape_ad.vacuum_two_form``), wrapped in ``jax.lax.custom_linear_solve``,
so ``jax.value_and_grad`` of any function of ``h`` costs one forward and one
adjoint solve, however many shape variables there are -- here every spline
coefficient of the map, ``2 n_r n_theta n_zeta`` of them.

The problem, as in the paper:

    min <Q_QA^2>_{r >= h_r}   s.t.   <iotabar>_s = iotabar*,   P <= 1e-8,

at the volume and aspect ratio (6) of the device, held exactly by rescaling
the map after every change. ``Q_QA`` is the quasi-axisymmetry residual of the
field (``quasisymmetry_residual``), averaged outside the first radial knot
span ``h_r``, where the polar map is close to singular. ``<iotabar>_s`` is
the mean flux-ratio rotational transform (``mean_iota``), and ``iotabar*`` is
its value on the device. ``P`` is the fraction of the field normal to the
logical surfaces (``normal_field_fraction``), and small ``P`` keeps them close
to flux surfaces, on which ``Q_QA`` and ``iotabar`` are defined. An augmented
Lagrangian over L-BFGS-B imposes both constraints. Every ``--al-inner``
iterations the multipliers are updated, and the penalty grows tenfold when the
violation stalls. A trial map that folds (``det DF <= 0``) is infeasible.

The device is LP (``LandremanPaul2021_QA``, Tutorials 1-2), its VMEC map
interpolated on this mesh. Its own criterion ``F_LP`` is the baseline and the
unit. The start is LP plus a smooth random normal displacement of its
boundary, ``--perturb-mm`` RMS in modes ``m, |n| <= 4`` per field period with
amplitudes ``~ 1 / (1 + m^2 + n^2)``. The run stops when the criterion is back
at the baseline with both constraints met (``--stop-qa 1``). The distance of
a boundary to LP's is ``d_RMS``, the area-weighted RMS over the boundary of the
distance to the closest point of LP's boundary, and ``a`` is LP's minor radius.
As in the paper, quasi-axisymmetry and the mean iota come back, but the device
does not: the end boundary still differs from LP's by a few percent of ``a``,
since quasi-axisymmetry with these three scalars does not determine the shape.

What gets printed: LP's baseline and the start, every 10th iteration, every
multiplier update and the end. Two figures: the criterion and the mean iota
against the iteration, and the boundary of LP, the start and the end in three
toroidal planes. ``shapes.npz`` keeps the three maps' raw coefficients.

Unlike Tutorials 1-6 it runs in **float64** (the script sets
``MRX_DTYPE=float64``, as the paper's runs): in float32 the criterion, a
residual near ``1e-5``, and its adjoint gradient carry the solves' round-off:
L-BFGS-B stops after one to six iterations per multiplier update, the penalty
runs away and the run stalls at ~50 ``F_LP`` with the constraints unmet
(measured 2026-09-25 on the defaults), where float64 reaches the baseline in
86 iterations. At ``(12, 16, 16) p = 3``, 6144 variables, that is ~5 min on a
GPU and ~11 min on four CPU cores, setup included.

    python -u scripts/tutorials/7_qa_shape_optimization.py
"""

# %%
# Now we read the run's options and make the output folder, and ask for
# double precision before MRX is imported (it fixes its precision on import).
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MRX_DTYPE", "float64")

# Run the cells top to bottom in a notebook / VS Code interactive window,
# or the whole file as a script (the CLI flags below still apply then).
_INTERACTIVE = "ipykernel" in sys.modules

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--geometry", default="data/wout_LandremanPaul2021_QA_lowres.nc",
                help="a VMEC wout (.nc) or a GVEC state file (.dat)")
ap.add_argument("--ns", default="12,16,16")
ap.add_argument("--p", type=int, default=3)
ap.add_argument("--perturb-mm", type=float, default=10.0, help="RMS of the start's boundary displacement (mm)")
ap.add_argument("--seed", type=int, default=0, help="the random draw of the displacement")
ap.add_argument("--maxiter", type=int, default=1000, help="L-BFGS-B iterations in all")
ap.add_argument("--al-inner", type=int, default=100, help="L-BFGS-B iterations per multiplier update")
ap.add_argument("--stop-qa", type=float, default=1.0,
                help="stop once <Q_QA^2> <= this x LP's with both constraints met [0: run --maxiter]")
ap.add_argument("--out", default="outputs/tutorials/qa_shape_optimization")
cli = ap.parse_args([] if _INTERACTIVE else None)
ns = tuple(int(v) for v in cli.ns.split(","))
os.makedirs(cli.out, exist_ok=True)

# %%
# Now we import MRX -- the sequence builder, the VMEC map, and the
# differentiable vacuum field with its criteria.
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
if not _INTERACTIVE:
    matplotlib.use("Agg")  # headless as a script; a notebook keeps its inline backend
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import mrx
from mrx.geometry import build_sequence, grad_1d
from mrx.gvec import build_gvec_map
from mrx.nullspace import compute_nullspaces
from mrx.plotting import plot_twin_axis
from mrx.shape_ad import (BoundaryShape, cylindrical_geometry, flux_seed, mean_iota, normal_field_fraction,
                          quasisymmetry_residual, section_moments, vacuum_two_form, with_geometry)
from mrx.spline_bases import basis_table

print(f"[env] mrx precision {mrx.DTYPE}")
ASPECT = 6.0         # the aspect ratio held exactly
BETA_SCALE = 0.01    # metres of coefficient change per unit of the variables
C_UNIT = 1e-3        # the unit of the iota constraint, c = (<iotabar>_s - iotabar*) / C_UNIT
P_MAX = 1e-8         # the bound on P
IOTA_TOL, P_TOL = 1e-5, 1.01e-8   # met when |<iotabar>_s - iotabar*| <= IOTA_TOL and P <= P_TOL

# %%
# Now we build the sequence on LP's VMEC map and take LP's map coefficients:
# every ring of raw R and Z coefficients, (n_r, n_theta, n_zeta) each.
t0 = time.perf_counter()
seq, _ = build_sequence(cli.geometry, ns, cli.p)
compute_nullspaces(seq)
nfp = seq.nfp
_, info = build_gvec_map(seq.equilibrium, seq, stellarator_symmetric=seq.half_period)
lp = BoundaryShape.from_coefficients(seq, info["raw_R"], info["raw_Z"], nfp, info["sign"])
sign = lp.sign
_, _, S_lp = section_moments(seq, lp.raw_R, lp.raw_Z, nfp, sign)
a_lp = float(np.sqrt(S_lp / np.pi))
T = np.asarray(seq.basis_0.Λ[0].T)
r_min = float(np.min(T[T > 0.0]))   # h_r, the end of the first radial knot span

# %%
# Now we set up the problem. The variables x are the change of every
# coefficient, beta = BETA_SCALE x: the boundary ring's change is extended
# harmonically into the interior, every inner ring changes on its own too,
# and map_coefficients rescales the map to LP's volume and to ASPECT. The
# field is h = seed - curl A on the changed map, and from it the criterion,
# the mean iota and P.
shape = BoundaryShape.from_coefficients(seq, lp.raw_R, lp.raw_Z, nfp, sign, aspect=ASPECT,
                                        extension="harmonic", free="all")
seed = flux_seed(seq)
n_var = 2 * lp.raw_R.size


def mapped(x, seq, shape):
    """The raw coefficients (R, Z) of the map at the variables x."""
    R, Z, _, _ = shape.map_coefficients(seq, BETA_SCALE * jnp.reshape(x, (2,) + tuple(shape.raw_R.shape)))
    return R, Z


def terms(x, seq, shape, seed):
    """<Q_QA^2>_{r >= h_r}, <iotabar>_s (in the sign of the map's orientation), P and min det DF at x."""
    R, Z = mapped(x, seq, shape)
    sq = with_geometry(seq, cylindrical_geometry(seq, R, Z, nfp, sign))
    h, _ = vacuum_two_form(sq, seed)
    F_qs, _ = quasisymmetry_residual(sq, h, R, Z, nfp, sign, r_min)
    return dict(F_qs=F_qs, iota=mean_iota(sq, h), P=normal_field_fraction(sq, h), jmin=jnp.min(sq.jacobian_j))


def lagrangian(x, seq, shape, seed, al):
    """The augmented Lagrangian f + nu_i c + mu c^2 / 2 + (max(0, nu_P + mu g)^2 - nu_P^2) / (2 mu), in the
    scaled f = <Q_QA^2> / F_LP, c = (<iotabar>_s - iotabar*) / C_UNIT and g = (P - P_MAX) / P_MAX."""
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
def min_det_df(x, seq, shape):
    """min det DF of the map at x on the quadrature points, without a solve: the map folds where it is not
    positive."""
    R, Z = mapped(x, seq, shape)
    return jnp.min(cylindrical_geometry(seq, R, Z, nfp, sign).jacobian_j)


# %%
# Now we evaluate LP itself (x = 0): its criterion is the baseline F_LP and
# its mean iota the target iotabar*.
lp_terms = forward(jnp.zeros(n_var), seq, shape, seed)
orient = float(jnp.sign(lp_terms["iota"]))
F_lp, target = float(lp_terms["F_qs"]), orient * float(lp_terms["iota"])
print(f"[setup] QA {ns} p={cli.p}, {n_var} variables, {time.perf_counter() - t0:.0f} s; LP: <Q_QA^2>_(r >= "
      f"{r_min:.3f}) = F_LP = {F_lp:.3e}, <iotabar>_s = {target:.5f}, P = {float(lp_terms['P']):.1e}, "
      f"a = {a_lp:.4f} m")

# %%
# Now we perturb LP's boundary. On LP's boundary surface F(theta, zeta) a
# random smooth displacement delta along the normal, as a change of the
# boundary ring, collocated at the Greville points. boundary_points evaluates
# the surface and its angular derivatives at any logical angles.
def boundary_points(raw_R, raw_Z, theta, zeta):
    """(F, F_theta, F_zeta) of the boundary surface r = 1 of the raw coefficients at the logical angles."""
    lt, lz = seq.basis_0.Λ[1], seq.basis_0.Λ[2]
    t, z = jnp.asarray(np.mod(theta, 1.0)), jnp.asarray(np.mod(zeta, 1.0))
    Bt, Bz = basis_table(lt, t), basis_table(lz, z)
    Dt = grad_1d(basis_table(seq.basis_0.dΛ[1], t), lt.type)
    Dz = grad_1d(basis_table(seq.basis_0.dΛ[2], z), lz.type)

    def ev(C, A, B):
        return np.asarray(jnp.einsum("jp,jk,kp->p", A, jnp.asarray(C)[-1], B))
    R, Z = ev(raw_R, Bt, Bz), ev(raw_Z, Bt, Bz)
    Rt, Zt, Rz, Zz = ev(raw_R, Dt, Bz), ev(raw_Z, Dt, Bz), ev(raw_R, Bt, Dz), ev(raw_Z, Bt, Dz)
    phi = 2.0 * np.pi / nfp
    c, s = np.cos(phi * zeta), np.sin(phi * zeta)
    F = np.stack([R * c, sign * R * s, Z], -1)
    Ft = np.stack([Rt * c, sign * Rt * s, Zt], -1)
    Fz = np.stack([Rz * c - phi * R * s, sign * (Rz * s + phi * R * c), Zz], -1)
    return F, Ft, Fz


def perturbation(amplitude, draw, m_max=4, n_max=4):
    """delta = sum a_mn cos 2 pi (m theta - n zeta), a_mn ~ N(0, 1) / (1 + m^2 + n^2), scaled to the area-weighted
    RMS amplitude (m), carried by (dR, dZ) along the (R, Z) part of the normal: the change of the boundary ring."""
    gt, gz = (np.asarray(seq.greville[a].point_rule[0][:, 0]) for a in (1, 2))
    th, ze = (g.reshape(-1) for g in np.meshgrid(gt, gz, indexing="ij"))
    rng = np.random.default_rng(draw)
    delta = np.zeros_like(th)
    for m in range(m_max + 1):
        for n in range(-n_max, n_max + 1):
            delta += rng.standard_normal() / (1.0 + m ** 2 + n ** 2) * np.cos(2.0 * np.pi * (m * th - n * ze))
    _, Ft, Fz = boundary_points(lp.raw_R, lp.raw_Z, th, ze)
    normal = np.cross(Ft, Fz)
    area = np.linalg.norm(normal, axis=-1)
    phi = 2.0 * np.pi * ze / nfp
    n_R, n_Z = normal[:, 0] * np.cos(phi) + sign * normal[:, 1] * np.sin(phi), normal[:, 2]
    delta *= amplitude / np.sqrt(np.sum(area * delta ** 2) / np.sum(area))
    t = delta * area / (n_R ** 2 + n_Z ** 2)
    ct, cz = np.asarray(seq.greville[1].coll), np.asarray(seq.greville[2].coll)

    def collocate(values):
        return np.linalg.solve(ct, np.linalg.solve(cz, values.reshape(gt.size, gz.size).T).T)
    return np.stack([collocate(t * n_R), collocate(t * n_Z)])


def distance(raw_R, raw_Z, grid=(128, 64), iterations=8):
    """d_RMS of the boundary of the raw coefficients to LP's: for every point x of it on a uniform grid of
    logical angles, (x - F_LP(u*)) . n_LP(u*) at the closest point u* of LP's boundary (Gauss-Newton from the
    same logical angles), its RMS weighted by the area."""
    th, ze = (g.reshape(-1) for g in np.meshgrid(np.arange(grid[0]) / grid[0], np.arange(grid[1]) / grid[1],
                                                indexing="ij"))
    X, Xt, Xz = boundary_points(raw_R, raw_Z, th, ze)
    area = np.linalg.norm(np.cross(Xt, Xz), axis=-1)
    u = np.stack([th, ze], -1)
    for _ in range(iterations):
        F, Ft, Fz = boundary_points(lp.raw_R, lp.raw_Z, u[:, 0], u[:, 1])
        Jac = np.stack([Ft, Fz], -1)
        u = u + np.linalg.solve(np.einsum("pki,pkj->pij", Jac, Jac), np.einsum("pki,pk->pi", Jac, X - F)[..., None])[..., 0]
    F, Ft, Fz = boundary_points(lp.raw_R, lp.raw_Z, u[:, 0], u[:, 1])
    nrm = np.cross(Ft, Fz)
    d = np.sum((X - F) * nrm / np.linalg.norm(nrm, axis=-1, keepdims=True), -1)
    return float(np.sqrt(np.sum(area * d ** 2) / np.sum(area)))


full = np.zeros((2,) + tuple(lp.raw_R.shape))
full[:, -1] = perturbation(1e-3 * cli.perturb_mm, cli.seed)
x = full.reshape(-1) / BETA_SCALE
R0, Z0 = mapped(jnp.asarray(x), seq, shape)
start = forward(jnp.asarray(x), seq, shape, seed)
print(f"[start] {cli.perturb_mm:g} mm RMS (draw {cli.seed}): d_RMS / a = {distance(R0, Z0) / a_lp:.3e}, "
      f"<Q_QA^2> = {float(start['F_qs']) / F_lp:.3g} F_LP, <iotabar>_s = {orient * float(start['iota']):.5f}, "
      f"P = {float(start['P']):.1e}, min det DF = {float(start['jmin']):.2e}")

# %%
# Now we optimize. Each outer step runs --al-inner L-BFGS-B iterations on the
# augmented Lagrangian, then updates the multipliers, nu_i += mu c and
# nu_P = max(0, nu_P + mu g), and multiplies mu by 10 if the violation is
# above its tolerance and fell less than 4x. A folding trial point returns ten
# times the last value with the last gradient, so the line search backtracks.
al = dict(F_lp=F_lp, target=target, orient=orient, nu_i=0.0, nu_P=0.0, mu=1.0)
history, cache, last = [], {}, {}
violation_tol = min(IOTA_TOL / C_UNIT, (P_TOL - P_MAX) / P_MAX)


def fun(x):
    key = x.tobytes()
    if key not in cache:
        if float(min_det_df(jnp.asarray(x), seq, shape)) <= 0.0:
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
        print(f"[iter {it:4d}] <Q_QA^2> {r['F']:.3f} F_LP  <iotabar>_s {r['iota']:.6f}  P {r['P']:.2e}  "
              f"min det DF {r['jmin']:.2e}")
    if (cli.stop_qa and r["F"] <= cli.stop_qa and met(r)) or it >= cli.maxiter:
        raise StopIteration


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
    c, g = (r["iota"] - target) / C_UNIT, (r["P"] - P_MAX) / P_MAX
    V = max(abs(c), abs(max(g, -al["nu_P"] / al["mu"])))
    al["nu_i"] += al["mu"] * c
    al["nu_P"] = max(0.0, al["nu_P"] + al["mu"] * g)
    if V_prev is not None and V > violation_tol and V > 0.25 * V_prev:
        al["mu"] *= 10.0
    V_prev, outer = V, outer + 1
    cache.clear()
    fun(x)
    print(f"[outer {outer}] iteration {len(history) - 1}: c {c:+.2e}, g {g:+.2e}, violation {V:.2e}; "
          f"nu_i {al['nu_i']:+.3e}, nu_P {al['nu_P']:.3e}, mu {al['mu']:g}")
wall = time.perf_counter() - t0
R1, Z1 = mapped(jnp.asarray(x), seq, shape)
print(f"[end] {len(history) - 1} iterations, {wall:.0f} s: <Q_QA^2> {history[0]['F']:.3g} -> {r['F']:.3f} F_LP, "
      f"<iotabar>_s {r['iota']:.6f} (iotabar* {target:.6f}), P {r['P']:.2e}; d_RMS / a {distance(R0, Z0) / a_lp:.3e} "
      f"-> {distance(R1, Z1) / a_lp:.3e}")
np.savez(os.path.join(cli.out, "shapes.npz"), raw_R_lp=np.asarray(lp.raw_R), raw_Z_lp=np.asarray(lp.raw_Z),
         raw_R_start=np.asarray(R0), raw_Z_start=np.asarray(Z0), raw_R_end=np.asarray(R1), raw_Z_end=np.asarray(Z1))

# %%
# Now we plot the criterion (in units of the baseline F_LP, dotted) and the
# mean iota (iotabar*, dotted) against the iteration.
F_path = np.array([h["F"] for h in history])
iota_path = np.array([h["iota"] for h in history])
fig, (ax_F, ax_i) = plot_twin_axis(F_path, iota_path, right_log=False,
                                   left_label=r"$\langle Q_{\mathrm{QA}}^2 \rangle_{r \geq h_r} / F_{\mathrm{LP}}$",
                                   right_label=r"$\langle \bar\iota \rangle_s$",
                                   left_plot_kwargs=dict(marker=""), right_plot_kwargs=dict(marker=""))
ax_F.axhline(1.0, color=ax_F.get_lines()[0].get_color(), ls=":", lw=1)
ax_i.axhline(target, color=ax_i.get_lines()[0].get_color(), ls=":", lw=1)
path = os.path.join(cli.out, "trace.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
if _INTERACTIVE:
    plt.show()
else:
    plt.close(fig)
print(f"  -> {path}")

# %%
# Now we draw the boundary of LP, the start and the end in three toroidal
# planes of a field period.
planes = (0.0, 0.25, 0.5)
theta = np.linspace(0.0, 1.0, 257)
fig, axes = plt.subplots(1, len(planes), figsize=(4 * len(planes), 4), constrained_layout=True)
for ax, zeta in zip(axes, planes):
    for (R, Z), label, style in (((lp.raw_R, lp.raw_Z), "LP", dict(color="k", lw=1.5)),
                                 ((R0, Z0), "start", dict(color="C1", ls="--", lw=1)),
                                 ((R1, Z1), "end", dict(color="C0", lw=1))):
        F, _, _ = boundary_points(R, Z, theta, np.full_like(theta, zeta))
        ax.plot(np.hypot(F[:, 0], F[:, 1]), F[:, 2], label=label, **style)
    ax.set_aspect("equal")
    ax.set_title(rf"$\zeta = {zeta:g}$")
    ax.set_xlabel(r"$R$ [m]")
axes[0].set_ylabel(r"$Z$ [m]")
axes[1].legend(frameon=False, loc="upper right")
path = os.path.join(cli.out, "sections.png")
fig.savefig(path, dpi=200)
if _INTERACTIVE:
    plt.show()
else:
    plt.close(fig)
print(f"  -> {path}")
