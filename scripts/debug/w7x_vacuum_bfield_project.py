# %% [markdown]
# # W7-X vacuum B-field: project B onto V2 (single resolution, verbose)
#
# Loads `data/W7X-vacuum.h5` (a 32^3 logical grid over one field period, nfp=5):
#   * `eval_points` (N,3) -- logical coords (rho, theta, zeta) in [0,1)
#   * `R`, `Z`        (N,) -- cylindrical position of each logical point
#   * `B`            (N,3) -- Cartesian magnetic field (Bx, By, Bz) [T]
#
# R, Z and each B component are built by the SAME routine: an interpolatory
# tensor B-spline that collocates the data at the 32^3 grid nodes and solves the
# three separable per-axis systems (n_basis = n_data per axis => square). The
# spline interpolates the data at the nodes (O(h^{p+1}), no linear resample), so
# R,Z give a high-accuracy stellarator map and B_fn is a smooth field.
#
# Then, at ONE projection resolution, project B onto V2 via the interpolatory
# spline (default) and report rich diagnostics. The stock linear-resample path
# (project_sampled_field) is kept only as a commented-out baseline -- it carries
# an O(h^2) bias, so prefer the spline for both R/Z and B.
#
# FRAME: the stored B is already in MRX's Cartesian frame -- established by the
# flux-surface tangency |B^rho|/|B^tan| ~ 0 (vacuum field), NOT by projection
# error (frame-blind). No conversion.
#
# The rho=1 wall is evaluated at rho = 1 - EVAL_EPS because the spline map's DF
# (hence det DF and the k=2 pushforward DF*f/det) is singular at the outer knot.
#
# Run:  W7X_MAP_BATCH=2048 XLA_PYTHON_CLIENT_PREALLOCATE=false \
#         .venv/bin/python scripts/debug/w7x_vacuum_bfield_project.py

# %%
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import jax

jax.config.update("jax_enable_x64", True)

import h5py
import jax.numpy as jnp
import numpy as np

import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.io import project_sampled_field
from mrx.mappings import stellarator_map
from mrx.operators import (
    assemble_mass_surgery_preconditioner,
    assemble_tensor_mass_preconditioner,
)
from mrx.projectors import _solve_tensor_collocation_axis

mrx.MAP_BATCH_SIZE_INNER = int(os.environ.get("W7X_MAP_BATCH", "2048"))

NFP = 5
H5 = "data/W7X-vacuum.h5"
TYPES = ("clamped", "periodic", "periodic")

# --- knobs ------------------------------------------------------------------
NS = (8, 16, 16)                 # projection resolution (nr, nt, nz)
P = 3                             # projection spline degree
FIT_P = 3                         # degree of the interpolatory R/Z/B data splines
EVAL_EPS = 1e-6                   # nudge off the singular rho=1 knot
CG_TOL, CG_MAXITER = 1e-7, 3000

# %% [markdown]
# ## 1. Load the h5 and recover the logical grid

# %%
with h5py.File(H5, "r") as f:
    ep = np.asarray(f["eval_points"], dtype=np.float64)   # (N,3) logical
    B = np.asarray(f["B"], dtype=np.float64)              # (N,3) MRX-Cartesian [T]
    Rv = np.asarray(f["R"], dtype=np.float64)             # (N,)
    Zv = np.asarray(f["Z"], dtype=np.float64)             # (N,)
    nr = int(f.attrs["n_rho"]); nt = int(f.attrs["n_theta"]); nz = int(f.attrs["n_zeta"])

N = ep.shape[0]
assert N == nr * nt * nz, (N, nr, nt, nz)

rho = ep[:, 0].reshape(nr, nt, nz)[:, 0, 0]
theta = ep[:, 1].reshape(nr, nt, nz)[0, :, 0]
zeta = ep[:, 2].reshape(nr, nt, nz)[0, 0, :]

R_grid = Rv.reshape(nr, nt, nz)
Z_grid = Zv.reshape(nr, nt, nz)
B_grid = B.reshape(nr, nt, nz, 3)
bnorm = np.linalg.norm(B, axis=1)

pts = jnp.asarray(ep)                                  # data nodes
ep_eval = ep.copy()
ep_eval[:, 0] = np.minimum(ep_eval[:, 0], 1.0 - EVAL_EPS)
pts_eval = jnp.asarray(ep_eval)                        # pushforward eval (off wall knot)

print(f"[load] N={N}  grid={nr}x{nt}x{nz}  nfp={NFP}")
print(f"[load] rho in [{rho.min():.4f},{rho.max():.4f}]  "
      f"theta in [{theta.min():.3f},{theta.max():.3f}]  "
      f"zeta in [{zeta.min():.3f},{zeta.max():.3f}]")
print(f"[load] R in [{Rv.min():.3f},{Rv.max():.3f}]  Z in [{Zv.min():.3f},{Zv.max():.3f}]"
      f"  |B| in [{bnorm.min():.3f},{bnorm.max():.3f}] T")


def batched(f, xs):
    return np.asarray(jax.lax.map(f, xs, batch_size=mrx.MAP_BATCH_SIZE_INNER))


# %% [markdown]
# ## 2. Interpolatory data-grid tensor splines for R, Z, B
#
# n_basis = n_data per axis => square per-axis collocation at the data nodes.

# %%
def interp_grid_spline(bs, colls, values_grid):
    """DiscreteFunction interpolating a scalar (nr,nt,nz) grid at the data nodes."""
    c = jnp.asarray(values_grid)
    for axis, coll in enumerate(colls):
        c = _solve_tensor_collocation_axis(coll, c, axis=axis)
    return DiscreteFunction(bs.e0 @ c.reshape(-1), bs.basis_0, bs.e0)


# ns IS the number of basis functions per axis (passed straight to SplineBasis),
# so n_basis = n_data = (nr,nt,nz) makes each per-axis collocation square.
FIT_NS = (nr, nt, nz)
fit_seq = DeRhamSequence(FIT_NS, (FIT_P,) * 3, 2 * FIT_P, TYPES, polar=False)
fit_seq.evaluate_1d()
_br, _bt, _bz = fit_seq.basis_0.Λ
colls = (_br.collocation_matrix(jnp.asarray(rho)),
         _bt.collocation_matrix(jnp.asarray(theta)),
         _bz.collocation_matrix(jnp.asarray(zeta)))
print(f"[fit ] FIT_NS={FIT_NS} p={FIT_P}  collocation conds: "
      + "  ".join(f"{ax}={np.linalg.cond(np.asarray(c)):.1e}"
                  for ax, c in zip("rtz", colls)))

R_h = interp_grid_spline(fit_seq, colls, R_grid)
Z_h = interp_grid_spline(fit_seq, colls, Z_grid)
map_func = stellarator_map(R_h, Z_h, nfp=NFP)

Bx_h = interp_grid_spline(fit_seq, colls, B_grid[..., 0])
By_h = interp_grid_spline(fit_seq, colls, B_grid[..., 1])
Bz_h = interp_grid_spline(fit_seq, colls, B_grid[..., 2])


def B_fn(x):
    return jnp.array([Bx_h(x)[0], By_h(x)[0], Bz_h(x)[0]])


# node interpolation accuracy (interpolatory => ~machine zero)
F = batched(map_func, pts)
R_err = np.abs(np.hypot(F[:, 0], F[:, 1]) - Rv)
Z_err = np.abs(F[:, 2] - Zv)
Bfit = batched(B_fn, pts)
b_err = np.linalg.norm(Bfit - B, axis=1)
print(f"[fit ] R node interp: max={R_err.max():.2e} mean={R_err.mean():.2e}")
print(f"[fit ] Z node interp: max={Z_err.max():.2e} mean={Z_err.mean():.2e}")
print(f"[fit ] B node interp: max rel={(b_err / bnorm).max():.2e} "
      f"mean rel={b_err.mean() / bnorm.mean():.2e}")

# flux-surface tangency of the map: B^rho = (DF^-1 B)_0 should vanish (vacuum)
DFinv = batched(lambda x: jnp.linalg.inv(jax.jacfwd(map_func)(x)), pts_eval)
Bc = np.einsum("nij,nj->ni", DFinv, B)
print(f"[fit ] tangency |B^rho|/|B^tan| = "
      f"{np.abs(Bc[:, 0]).mean() / np.abs(Bc[:, 1:]).mean():.3e}")

# %% [markdown]
# ## 3. Build the projection sequence at NS

# %%
print(f"[seq ] building polar de Rham seq ns={NS} p={P} ...", flush=True)
seq = DeRhamSequence(NS, (P,) * 3, 2 * P, TYPES, polar=True,
                     tol=CG_TOL, maxiter=CG_MAXITER, betti_numbers=(1, 1, 0, 0))
seq.evaluate_1d()
seq.set_map(map_func)
ops = seq.get_operators()
ops = assemble_mass_surgery_preconditioner(seq, operators=ops, ks=(0, 1, 2))
ops = assemble_tensor_mass_preconditioner(
    seq, operators=ops, ks=(0, 1, 2, 3), cp_kwargs={"greville": True})
seq.set_operators(ops)

jac = np.asarray(seq.jacobian_j)
print(f"[seq ] n_quad={jac.size}  Jacobian min={jac.min():.3e} max={jac.max():.3e} "
      f"(#<=0: {int((jac <= 0).sum())})")
print(f"[seq ] V2 dofs (free) = {int(seq.n2)}")

# %% [markdown]
# ## 4. Project B onto V2 via the interpolatory spline (the default path)
#
# The interpolatory spline B_fn (exact at the data nodes, O(h^{p+1})) fed through
# the 2-form load is the production path -- it has no O(h^2) linear-resample bias.
# The stock linear resample is kept only as a commented-out baseline below.

# %%
def report(label, dof):
    """Pushforward the discrete 2-form and print rich error diagnostics."""
    B_h = DiscreteFunction(dof, seq.basis_2, seq.e2)
    B_rec = batched(Pushforward(B_h, seq.map, 2), pts_eval)
    e = np.linalg.norm(B_rec - B, axis=1)
    print(f"\n[{label}] mean rel={e.mean() / bnorm.mean():.4e}  "
          f"max rel={(e / bnorm).max():.4e}  rms={np.sqrt((e**2).mean()):.4e} T")
    for i, nm in enumerate("xyz"):
        c = np.corrcoef(B[:, i], B_rec[:, i])[0, 1]
        ea = np.abs(B_rec[:, i] - B[:, i])
        print(f"[{label}]   B{nm}: corr={c:.4f}  mean|e|={ea.mean():.3e}  max|e|={ea.max():.3e}")
    # error profile along each logical axis (spread => which axis limits us)
    for ax, nm in [(0, "rho"), (1, "theta"), (2, "zeta")]:
        vals = np.unique(ep[:, ax])
        prof = np.array([e[ep[:, ax] == v].mean() / bnorm[ep[:, ax] == v].mean()
                         for v in vals])
        print(f"[{label}]   {nm:5s} ({len(vals)}): min={prof.min():.3e} "
              f"max={prof.max():.3e} spread={(prof.max()-prof.min())/prof.mean():.0%}  "
              + " ".join(f"{p:.2e}" for p in prof[::8]))
    return B_rec


# DEFAULT: interpolatory spline B -> 2-form load -> M2^-1
print("\n[proj] interpolatory spline B (seq.load + M2^-1)", flush=True)
dual = seq.load(B_fn, 2, frame='phys')
dof_sp, info = seq.apply_inverse_mass_matrix(
    dual, 2, dirichlet=False, return_info=True)
print(f"[spline] k=2 greville PCG: iters={abs(int(info))} "
      f"converged={int(info) <= 0}")
report("spline", dof_sp)

# --- BASELINE (optional): stock linear resample of the grid ----------------
# Carries an O(h^2) bias => the ~3.4% floor that won't converge with resolution.
# Uncomment to compare against the spline path above.
# t_w = np.concatenate([theta, [1.0]]); z_w = np.concatenate([zeta, [1.0]])
# B_w = np.concatenate([B_grid, B_grid[:, :1, :, :]], axis=1)
# B_w = np.concatenate([B_w, B_w[:, :, :1, :]], axis=2)
# B_p = np.concatenate([B_w[:1], B_w], axis=0)
# axes_B = (np.concatenate([[0.0], rho]), t_w, z_w)
# print("\n[proj] BASELINE linear resample (project_sampled_field)", flush=True)
# dof_lin = project_sampled_field(axes_B, B_p, seq, k=2, dirichlet=False)
# report("linear", dof_lin)

# %% [markdown]
# ## 5. Vacuum field: convert logical dzeta (0,0,1) to a 2-form and solve the
#    harmonic (nullvector) 2-form seeded with it.  Reuses `seq` (mass precond
#    already built); only adds the incidence + Schur-Jacobi operators the k=2
#    DBC saddle nullspace solve needs.

# %%
from mrx.operators import (  # noqa: E402
    assemble_incidence_operators, assemble_schur_jacobi_preconditioner)
from mrx.nullspace import find_nullspace_vectors  # noqa: E402

print("\n[vac ] assembling incidence + Schur-Jacobi (k=2 DBC) ...", flush=True)
_ops = seq.get_operators()
_ops = assemble_incidence_operators(seq, operators=_ops)
_ops = assemble_schur_jacobi_preconditioner(
    seq, _ops, ks=(2,), dirichlet_variants=(True,))
seq.set_operators(_ops)

# Vacuum 2-form IC: logical dzeta=(0,0,1) projected to V2, DBC (B.n=0 at the wall).
# M2^-1 supplies the Hodge/metric, so the covector is the same (0,0,1) as the k=1
# harmonic guess -- geometry-robust (no 1/R assumption).
dof2_ic = seq.apply_inverse_mass_matrix(
    seq.load(lambda x: jnp.array([0.0, 0.0, 1.0]), 2, dirichlet=True, frame='ref'),
    2, dirichlet=True)
print(f"[vac ] IC built; L2 norm = {float(seq.l2_norm(dof2_ic, 2, dirichlet=True)):.4e}")

# Refine to the true harmonic vacuum 2-form via inverse iteration.
eps = 1e-3 / (NS[0] ** 2)
vs, iters = find_nullspace_vectors(seq, seq.get_operators(), 2, 1, eps,
                                   dirichlet=True, x0s=[dof2_ic], abs_tol=1e-8)
n_it, res = iters[0]
print(f"[vac ] nullvector: iters={n_it}  ||L2 v||={res:.3e}  "
      f"({'IC already harmonic' if n_it == 0 else 'refined by inverse iteration'})")
B_vac = vs[0]

# Compare the computed vacuum field to the stored B (it is defined up to a scale).
B_vac_h = DiscreteFunction(B_vac, seq.basis_2, seq.e2_dbc)
Bvp = batched(Pushforward(B_vac_h, seq.map, 2), pts_eval)
s = float(np.sum(Bvp * B) / np.sum(Bvp * Bvp))     # best-fit scale
e = np.linalg.norm(s * Bvp - B, axis=1)
print(f"[vac ] best-fit scale={s:.3e}  |s*B_vac - B| mean rel={e.mean() / bnorm.mean():.3e} "
      f"max rel={(e / bnorm).max():.3e}")
for i, nm in enumerate("xyz"):
    print(f"[vac ]   B{nm} corr={np.corrcoef(Bvp[:, i], B[:, i])[0, 1]:.4f}")

# %%