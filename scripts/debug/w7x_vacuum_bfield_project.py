# %% [markdown]
# # Vacuum B-field: harmonic 1-form / 2-form vs reference Cartesian B
#
# Loads ONE self-contained h5 (logical tensor grid rho,theta,zeta in [0,1); per
# node the cylindrical R,Z and a Cartesian reference field B).  Everything --
# geometry AND field -- comes from that single file:
#   * `eval_points` (N,3)  logical coords, C-order with zeta fastest
#   * `R`, `Z`      (N,)   cylindrical position
#   * `B`           (N,3)  Cartesian magnetic field [T]
#   * attrs: n_rho/n_theta/n_zeta (or precomputed_nr/ntheta/nzeta), nfp
# Works with both the old GVEC files (quasr_XXXX.h5, 50^3, attrs n_rho/...) and
# the new simsopt files (quasrXXXX_simsopt_B.h5, coarse, attrs precomputed_nr/...).
#
# Because R,Z and B are read from the SAME file, they are self-consistent by
# construction (for a correctly-generated file).  A frame/field-period bug shows
# up as a large `radial` fraction in the [ref] line -- see the handoff doc for the
# quasr0044970 case, where the file's B was rotated one field period off its R,Z.
#
# What it computes:
#   1. (optional) L2 projection of B onto V2, with error -- skip via --no-projection.
#   2. The harmonic 2-form (DBC nullvector of the k=2 Hodge Laplacian).
#   3. The harmonic 1-form (natural/no-DBC nullvector of the k=1 Hodge Laplacian).
#   For each: pushforward to the Cartesian (xyz) frame, best-fit-scaled, error vs
#   the reference B, and the (rho,theta,zeta) where the relative error is maximal.
#
# Run:  XLA_PYTHON_CLIENT_PREALLOCATE=false W7X_MAP_BATCH=2048 \
#         .venv/bin/python scripts/debug/w7x_vacuum_bfield_project.py \
#             --h5 data/quasr_0009983.h5

# %%
import argparse
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
from mrx.io import load_grid_field
from mrx.nullspace import find_nullspace_vectors
from mrx.operators import (
    assemble_incidence_operators,
    assemble_mass_surgery_preconditioner,
    assemble_schur_jacobi_preconditioner,
    assemble_tensor_mass_preconditioner,
)
from mrx.projectors import _solve_tensor_collocation_axis

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--h5", default="data/W7X-vacuum.h5",
               help="self-contained h5: geometry (R,Z,nfp) AND field B, old or new format")
p.add_argument("--nfp", type=int, default=None, help="override field periods (else from attrs)")
p.add_argument("--ns", type=int, nargs=3, default=(8, 16, 16), help="projection resolution nr nt nz")
p.add_argument("--p", type=int, default=3, help="projection spline degree")
p.add_argument("--fit-degree", type=int, default=3, help="interpolatory R/Z spline degree")
p.add_argument("--stride", type=int, default=1, help="per-axis subsample of the GEOMETRY grid (speed)")
p.add_argument("--eval-eps", type=float, default=1e-6, help="nudge off the singular rho=1 knot")
p.add_argument("--r0", type=float, default=None,
               help="major radius to normalize lengths to R0=1 (default: 0.5*(Rmin+Rmax))")
p.add_argument("--no-normalize", action="store_true",
               help="do NOT rescale geometry to major radius 1 (keep physical lengths)")
p.add_argument("--no-projection", action="store_true",
               help="skip the L2 projection of each reference B onto V2 (harmonics only)")
p.add_argument("--harmonic-tol", type=float, default=1e-6,
               help="absolute residual tolerance ||L v|| for the harmonic-form solves")
args = p.parse_args()

mrx.MAP_BATCH_SIZE_INNER = int(os.environ.get("W7X_MAP_BATCH", "2048"))
NS = tuple(args.ns); P = args.p; FIT_P = args.fit_degree
EVAL_EPS = args.eval_eps
TYPES = ("clamped", "periodic", "periodic")
FIT_Q = 1                          # fit_seq quadrature is unused (only its 1D bases/e0)
CG_TOL, CG_MAXITER = 1e-7, 3000


def batched(f, xs):
    return np.asarray(jax.lax.map(f, xs, batch_size=mrx.MAP_BATCH_SIZE_INNER))


# ---- factorized tensor-grid helpers (both fit + eval are per-axis) ----------
def coll_val(b, pts):
    return jnp.asarray(b.collocation_matrix(jnp.asarray(pts)))               # (npt, nbasis)


def coll_der(b, pts):
    ns = b.ns
    g = lambda x: jax.vmap(lambda i: jax.grad(lambda xx: b(xx, i))(x))(ns)
    return jax.vmap(g)(jnp.asarray(pts))


def grid_eval(c, Mr, Mt, Mz):
    t = jnp.einsum('ai,ijk->ajk', Mr, c)
    t = jnp.einsum('bj,ajk->abk', Mt, t)
    return jnp.einsum('ck,abk->abc', Mz, t)


def fit_coeffs(colls_solve, values_grid):
    c = jnp.asarray(values_grid)
    for axis, coll in enumerate(colls_solve):
        c = _solve_tensor_collocation_axis(coll, c, axis=axis)
    return c


def map_and_DF_on_grid(cR, cZ, V, Der, zeta_axis, nfp, sign):
    """Analytic F and DF for F = (R cos a, sign*R sin a, Z), a = 2*pi*zeta/nfp,
    on the tensor grid (three 1D contractions; reshape(-1,...) matches ep order)."""
    Vr, Vt, Vz = V; Dr, Dt, Dz = Der
    R = grid_eval(cR, Vr, Vt, Vz)
    Rr = grid_eval(cR, Dr, Vt, Vz); Rt = grid_eval(cR, Vr, Dt, Vz); Rz = grid_eval(cR, Vr, Vt, Dz)
    Zr = grid_eval(cZ, Dr, Vt, Vz); Zt = grid_eval(cZ, Vr, Dt, Vz); Zz = grid_eval(cZ, Vr, Vt, Dz)
    a = 2 * jnp.pi / nfp
    g = a * jnp.asarray(zeta_axis)
    cos = jnp.cos(g)[None, None, :]; sin = jnp.sin(g)[None, None, :]
    DF = jnp.stack([
        jnp.stack([Rr * cos, Rt * cos, Rz * cos - R * a * sin], axis=-1),
        jnp.stack([sign * Rr * sin, sign * Rt * sin, sign * (Rz * sin + R * a * cos)], axis=-1),
        jnp.stack([Zr, Zt, Zz], axis=-1),
    ], axis=-2)
    return DF


# ---- reference-field loading ------------------------------------------------
def _attr(f, *names):
    for n in names:
        if n in f.attrs:
            return f.attrs[n]
    raise KeyError(f"none of {names} in attrs")


def load_ref(path, stride=1):
    """Load a reference-B h5 into a dict with logical tensor axes + flat fields."""
    with h5py.File(path, "r") as f:
        ep = np.asarray(f["eval_points"], dtype=np.float64)
        B = np.asarray(f["B"], dtype=np.float64)
        Rv = np.asarray(f["R"], dtype=np.float64)
        Zv = np.asarray(f["Z"], dtype=np.float64)
        nr = int(_attr(f, "n_rho", "precomputed_nr"))
        nt = int(_attr(f, "n_theta", "precomputed_ntheta"))
        nz = int(_attr(f, "n_zeta", "precomputed_nzeta"))
        nfp = int(_attr(f, "nfp"))
    s = stride
    rho = ep[:, 0].reshape(nr, nt, nz)[::s, 0, 0]
    theta = ep[:, 1].reshape(nr, nt, nz)[0, ::s, 0]
    zeta = ep[:, 2].reshape(nr, nt, nz)[0, 0, ::s]
    R_grid = Rv.reshape(nr, nt, nz)[::s, ::s, ::s]
    Z_grid = Zv.reshape(nr, nt, nz)[::s, ::s, ::s]
    B_grid = B.reshape(nr, nt, nz, 3)[::s, ::s, ::s]
    nr, nt, nz = R_grid.shape
    rr, tt, zz = np.meshgrid(rho, theta, zeta, indexing="ij")
    ep = np.stack([rr.ravel(), tt.ravel(), zz.ravel()], axis=1)
    B = B_grid.reshape(-1, 3)
    return dict(path=path, axes=(rho, theta, zeta), shape=(nr, nt, nz), nfp=nfp,
                ep=ep, B=B, R=R_grid.reshape(-1), Z=Z_grid.reshape(-1),
                bnorm=np.linalg.norm(B, axis=1))


def eval_pts(fld):
    """Logical eval points with rho nudged off the singular rho=1 knot."""
    e = fld["ep"].copy()
    e[:, 0] = np.minimum(e[:, 0], 1.0 - EVAL_EPS)
    return jnp.asarray(e)


# Every err_line call appends {tag, path, abs, rel, diff} so the full per-point
# error record of every comparison is available (e.g. when run interactively).
B_ERRORS = []


def err_line(tag, B_rec, fld, extra=""):
    """Report error stats (xyz frame) vs a reference field: mean, a set of
    percentiles of the pointwise relative error, and the (rho,theta,zeta) where
    it is maximal.  Stores the full per-point error arrays in B_ERRORS."""
    B = fld["B"]; bn = fld["bnorm"]; ep = fld["ep"]
    diff = B_rec - B
    e = np.linalg.norm(diff, axis=1)
    rel = e / bn
    B_ERRORS.append({"tag": tag, "path": fld["path"], "abs": e, "rel": rel, "diff": diff})
    i = int(np.argmax(rel))
    qs = [50, 75, 90, 95, 99]
    pr = np.percentile(rel, qs)
    zvals = np.unique(ep[:, 2])
    prof = np.array([e[ep[:, 2] == v].mean() / bn[ep[:, 2] == v].mean() for v in zvals])
    print(f"    {tag}:{extra}")
    print(f"        rel err  mean={e.mean() / bn.mean():.4e}  median(rel)={np.median(rel):.4e}  "
          f"zeta-spread={(prof.max() - prof.min()) / prof.mean():.0%}")
    print("        rel pct  " + "  ".join(f"p{q}={v:.3e}" for q, v in zip(qs, pr))
          + f"  max={rel.max():.3e}")
    print(f"        abs err [T]  mean={e.mean():.4e}  median={np.median(e):.4e}  max={e.max():.4e}")
    print(f"        max rel at (r={ep[i, 0]:.3f},t={ep[i, 1]:.3f},z={ep[i, 2]:.3f})")


# %% [markdown]
# ## 1. Load geometry file, fit R/Z, build the stellarator map

# %%
geo = load_ref(args.h5, stride=args.stride)
nr, nt, nz = geo["shape"]
rho, theta, zeta = geo["axes"]
nfp = args.nfp if args.nfp is not None else geo["nfp"]
R_grid = geo["R"].reshape(nr, nt, nz)
Z_grid = geo["Z"].reshape(nr, nt, nz)

# Non-dimensionalize lengths to major radius R0 = 1 (scale-invariant physics,
# O(1) Hodge-Laplacian spectrum so fixed harmonic-solve tolerances transfer).
R0 = 1.0
if not args.no_normalize:
    R0 = args.r0 if args.r0 is not None else 0.5 * (float(R_grid.min()) + float(R_grid.max()))
    R_grid = R_grid / R0
    Z_grid = Z_grid / R0
    print(f"[geo ] normalized major radius R0={R0:.4f} -> "
          f"R in [{R_grid.min():.3f},{R_grid.max():.3f}]")

print(f"[geo ] {args.h5}  grid={nr}x{nt}x{nz} (stride={args.stride})  nfp={nfp}")

rho_eval = np.minimum(rho, 1.0 - EVAL_EPS)
FIT_NS = (nr, nt, nz)
fit_seq = DeRhamSequence(FIT_NS, (FIT_P,) * 3, FIT_Q, TYPES, polar=False)
fit_seq.evaluate_1d()
_br, _bt, _bz = fit_seq.basis_0.Λ
colls_solve = (coll_val(_br, rho), coll_val(_bt, theta), coll_val(_bz, zeta))
Vgrid = (coll_val(_br, rho_eval), coll_val(_bt, theta), coll_val(_bz, zeta))
Dgrid = (coll_der(_br, rho_eval), coll_der(_bt, theta), coll_der(_bz, zeta))

cR = fit_coeffs(colls_solve, R_grid); cZ = fit_coeffs(colls_solve, Z_grid)

# auto-orient: flipping the toroidal sign flips det(DF); pick the sign with J>0.
J_pos = np.linalg.det(np.asarray(
    map_and_DF_on_grid(cR, cZ, Vgrid, Dgrid, zeta, nfp, sign=1.0)).reshape(-1, 3, 3))
sign = 1.0 if np.median(J_pos) > 0 else -1.0
DF = np.asarray(map_and_DF_on_grid(cR, cZ, Vgrid, Dgrid, zeta, nfp, sign)).reshape(-1, 3, 3)
J = np.linalg.det(DF)
print(f"[map ] toroidal sign={sign:+.0f}  ->  det(DF) in [{J.min():.3e},{J.max():.3e}] "
      f"(#<=0: {int((J <= 0).sum())})")

R_h = DiscreteFunction(fit_seq.e0 @ cR.reshape(-1), fit_seq.basis_0, fit_seq.e0)
Z_h = DiscreteFunction(fit_seq.e0 @ cZ.reshape(-1), fit_seq.basis_0, fit_seq.e0)
_a_nfp = 2 * jnp.pi / nfp


def map_func(x):
    ang = _a_nfp * x[2]
    r = R_h(x)[0]
    return jnp.array([r * jnp.cos(ang), sign * r * jnp.sin(ang), Z_h(x)[0]])


# %% [markdown]
# ## 2. Field diagnostics (geometry and B both come from the one loaded file)

# %%
def check_RZ(label, fld):
    """Sanity check that the interpolatory R,Z splines reproduce the file's stored
    R,Z at its own nodes (physical units).  For an interpolatory fit this is
    ~machine precision; a large residual would mean the fit or grid is broken."""
    pts = eval_pts(fld)
    R_rec = np.asarray(batched(R_h, pts))[:, 0] * R0
    Z_rec = np.asarray(batched(Z_h, pts))[:, 0] * R0
    dR = np.abs(R_rec - fld["R"]); dZ = np.abs(Z_rec - fld["Z"])
    fld["dR"], fld["dZ"] = dR, dZ
    span = max(np.ptp(fld["R"]), np.ptp(fld["Z"]))
    print(f"[RZ  ] {label}: |R_rec-R_file| max={dR.max():.3e} mean={dR.mean():.3e} | "
          f"|Z_rec-Z_file| max={dZ.max():.3e} mean={dZ.mean():.3e} | "
          f"rel-to-span={max(dR.max(), dZ.max()) / span:.2%}")


geo["DF"], geo["J"] = DF, J
fields = [("B:" + os.path.basename(args.h5), geo)]

for label, fld in fields:
    print(f"[ref ] {label}: N={fld['ep'].shape[0]}  grid={fld['shape']}  "
          f"|B| in [{fld['bnorm'].min():.3f},{fld['bnorm'].max():.3f}] T")
    # Vacuum toroidal field scales as |B| ~ 1/R, so |B|*R should be ~constant and
    # |B| tightly (anti-)correlated with R.  A broken correlation means the B array
    # does not track its own R (e.g. rows scrambled / mismatched source).
    bR = fld["bnorm"] * fld["R"]
    cov = float(np.std(bR) / np.mean(bR))
    corr = float(np.corrcoef(fld["bnorm"], 1.0 / fld["R"])[0, 1])
    print(f"       |B|*R  CoV={cov:.3f}   corr(|B|,1/R)={corr:+.3f}")
    # Cylindrical decomposition in the geometry's toroidal frame (Phi = sign*a).
    # A vacuum field is ~99% toroidal; a large radial fraction flags that the B
    # vectors are rotated about z relative to the map frame (e.g. a field-period
    # toroidal-angle offset), which R,Z checks cannot see (rotation-invariant).
    phi = sign * 2 * np.pi * fld["ep"][:, 2] / nfp
    c, s = np.cos(phi), np.sin(phi)
    B_R = fld["B"][:, 0] * c + fld["B"][:, 1] * s
    B_phi = -fld["B"][:, 0] * s + fld["B"][:, 1] * c
    print(f"       frac  toroidal={np.mean(np.abs(B_phi) / fld['bnorm']):.3f}  "
          f"radial={np.mean(np.abs(B_R) / fld['bnorm']):.3f}  "
          f"vertical={np.mean(np.abs(fld['B'][:, 2]) / fld['bnorm']):.3f}")
    check_RZ(label, fld)


# %% [markdown]
# ## 3. Build the projection/harmonic de Rham sequence

# %%
print(f"[seq ] building polar de Rham seq ns={NS} p={P} ...", flush=True)
seq = DeRhamSequence(NS, (P,) * 3, 2 * P, TYPES, polar=True,
                     tol=CG_TOL, maxiter=CG_MAXITER, betti_numbers=(1, 1, 0, 0))
seq.evaluate_1d(); seq.set_map(map_func)
ops = seq.get_operators()
ops = assemble_mass_surgery_preconditioner(seq, operators=ops, ks=(0, 1, 2))
ops = assemble_tensor_mass_preconditioner(
    seq, operators=ops, ks=(0, 1, 2, 3), cp_kwargs={"greville": True})
ops = assemble_incidence_operators(seq, operators=ops)
ops = assemble_schur_jacobi_preconditioner(
    seq, ops, ks=(1, 2), dirichlet_variants=(False, True))
seq.set_operators(ops)
print(f"[seq ] V1 dofs={int(seq.n1)}  V2 dofs (free)={int(seq.n2)}")


# %% [markdown]
# ## 4. (optional) L2 projection of each reference B onto V2, with error
#    Covariant proxy omega = DF^T B (periodic -> seam-free) projected via
#    load_grid_field(frame='ref') then M2^-1; reconstructed to xyz with DF/J.

# %%
def project_report(label, fld):
    DFf, Jf = fld["DF"], fld["J"]
    proxy_grid = np.einsum("nij,nj->ni", np.transpose(DFf, (0, 2, 1)), fld["B"]
                           ).reshape(*fld["shape"], 3)
    dual = load_grid_field(fld["axes"], proxy_grid, seq, 2, frame='ref')
    dof, info = seq.apply_inverse_mass_matrix(dual, 2, dirichlet=False, return_info=True)
    B_h = DiscreteFunction(dof, seq.basis_2, seq.e2)
    rec_proxy = batched(B_h, eval_pts(fld))                       # ref proxy at data nodes
    B_rec = np.einsum("nij,nj->ni", DFf, rec_proxy) / Jf[:, None]  # pushforward with analytic DF/J
    err_line(f"proj {label}", B_rec, fld,
             extra=f"  iters={abs(int(info))} conv={int(info) <= 0}")


if not args.no_projection:
    print("\n[proj] L2 projection of each reference B onto V2 ...", flush=True)
    for label, fld in fields:
        project_report(label, fld)


# %% [markdown]
# ## 5. Harmonic solves + error (xyz frame) at every reference point

# %%
def compare(label, fh, k, fld):
    """Pushforward the harmonic k-form to xyz, best-fit-scale it, and report the
    error against reference field `fld` at the points where it is given."""
    Bp = batched(Pushforward(fh, seq.map, k), eval_pts(fld))
    sc = float(np.sum(Bp * fld["B"]) / np.sum(Bp * Bp))   # best-fit scale
    err_line(f"vs {label}", sc * Bp, fld, extra=f"  scale={sc:+.3e}")


eps_null = 1e-3 / (NS[0] ** 2)

# --- harmonic 2-form (DBC): k=2 nullvector, seeded by logical dzeta ----------
print("\n[vac2] solving harmonic 2-form (k=2, DBC) ...", flush=True)
dof2_ic = seq.apply_inverse_mass_matrix(
    seq.load(lambda x: jnp.array([0.0, 0.0, 1.0]), 2, dirichlet=True, frame='ref'),
    2, dirichlet=True)
vs2, it2 = find_nullspace_vectors(seq, seq.get_operators(), 2, 1, eps_null,
                                  dirichlet=True, x0s=[dof2_ic], abs_tol=args.harmonic_tol)
print(f"[vac2] nullvector: iters={it2[0][0]}  ||L2 v||={it2[0][1]:.3e}")
B_vac2 = DiscreteFunction(vs2[0], seq.basis_2, seq.e2_dbc)
for label, fld in fields:
    compare(label, B_vac2, 2, fld)

# --- harmonic 1-form (no-DBC / natural): k=1 nullvector, seeded by logical dzeta
print("\n[vac1] solving harmonic 1-form (k=1, no-DBC) ...", flush=True)
dof1_ic = seq.apply_inverse_mass_matrix(
    seq.load(lambda x: jnp.array([0.0, 0.0, 1.0]), 1, dirichlet=False, frame='ref'),
    1, dirichlet=False)
vs1, it1 = find_nullspace_vectors(seq, seq.get_operators(), 1, 1, eps_null,
                                  dirichlet=False, x0s=[dof1_ic], abs_tol=args.harmonic_tol)
print(f"[vac1] nullvector: iters={it1[0][0]}  ||L1 v||={it1[0][1]:.3e}")
B_vac1 = DiscreteFunction(vs1[0], seq.basis_1, seq.e1)
for label, fld in fields:
    compare(label, B_vac1, 1, fld)
