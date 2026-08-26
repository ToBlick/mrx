"""Verify co/contravariant -> MRX 2-form projection on quasr_0009983.h5.

Settles two things:
  (A) MRX spline map DF matches GVEC's basis:  DF^T B ~ B_cov,  DF^-1 B ~ B_contra.
  (B) load(frame='ref', k=2) with f_ref = B_cov reproduces the physical B, and
      does so WITHOUT the Cartesian zeta=0 seam (periodic components).
Controls: also try feeding B_contra and the old Cartesian frame='phys' path.

Background: MRX's k=2 pushforward is Piola with the density division,
F_*w = DF*w/J, so a discrete 2-form with reference coeffs b_ref pushes forward
to DF*b_ref/J  =>  b_ref = J * DF^-1 * B = J * B_contra (the extra Jacobian).
load(frame='ref') is a dual (paired through M2 = int L^T (g/J) L), so requiring
the recovered DOFs equal J*B_contra means you FEED f_ref = B_cov for k=2
(and f_ref = B_contra for k=1) -- the co/contra roles swap.

Performance: eval_points is a full tensor grid (rho (x) theta (x) zeta) and every
interpolation basis is a tensor product b_i(r) b_j(t) b_k(z).  So spline values
and their partials on the grid are three sequential 1D contractions with the
per-axis (derivative) collocation matrices -- O(n^4) instead of 125k scattered
3D evals -- and the map Jacobian DF is assembled analytically from R,Z and their
partials (no jacfwd over 125k points).  See grid_eval / map_and_DF_on_grid.

Run (GPU):
  XLA_PYTHON_CLIENT_PREALLOCATE=false W7X_MAP_BATCH=2048 \
    .venv/bin/python scripts/debug/quasr_covcontra_verify.py
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import jax
import h5py
import jax.numpy as jnp
import numpy as np
import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.io import load_grid_field
from mrx.operators import (assemble_mass_surgery_preconditioner,
                           assemble_tensor_mass_preconditioner)
from mrx.projectors import _solve_tensor_collocation_axis

mrx.MAP_BATCH_SIZE_INNER = int(os.environ.get("W7X_MAP_BATCH", "2048"))
H5 = "data/quasr_0009983.h5"
NFP = 2
TYPES = ("clamped", "periodic", "periodic")
NS = (8, 16, 16); P = 3; FIT_P = 3
EVAL_EPS = 1e-6; CG_TOL, CG_MAXITER = 1e-7, 3000
# fit_seq's quadrature is never used here (we only use its 1D bases / e0 /
# collocation, all independent of q), so keep its quad order minimal -- a full
# 2*FIT_P rule at 50^3 builds a ~2.7e7-point 3D grid in the constructor.
FIT_Q = 1
# First-step speed knob: subsample the rho/theta/zeta grid by this stride
# (per-axis, so it stays a tensor grid).  1 = full data.
STRIDE = int(os.environ.get("EVAL_STRIDE", "1"))
# Toroidal-angle handedness: Y = TOR_SIGN * R sin(2pi zeta/nfp).  GVEC/standard
# cylindrical is +1 (det DF > 0); MRX's stellarator_map hardcodes -1, which
# mirrors this raw GVEC data (det DF < 0) and misaligns B's Cartesian frame.
TOR_SIGN = float(os.environ.get("TOR_SIGN", "1.0"))

with h5py.File(H5, "r") as f:
    ep = np.asarray(f["eval_points"]); B = np.asarray(f["B"])
    Bco = np.asarray(f["B_cov"]); Bct = np.asarray(f["B_contra"])
    Rv = np.asarray(f["R"]); Zv = np.asarray(f["Z"])
    nr, nt, nz = [int(f.attrs[k]) for k in ("n_rho", "n_theta", "n_zeta")]

# reshape to the full tensor grid, then subsample per-axis (keeps tensor form)
s = STRIDE
rho = ep[:, 0].reshape(nr, nt, nz)[::s, 0, 0]
theta = ep[:, 1].reshape(nr, nt, nz)[0, ::s, 0]
zeta = ep[:, 2].reshape(nr, nt, nz)[0, 0, ::s]
R_grid = Rv.reshape(nr, nt, nz)[::s, ::s, ::s]
Z_grid = Zv.reshape(nr, nt, nz)[::s, ::s, ::s]
Bco_grid = Bco.reshape(nr, nt, nz, 3)[::s, ::s, ::s]
Bct_grid = Bct.reshape(nr, nt, nz, 3)[::s, ::s, ::s]
Bca_grid = B.reshape(nr, nt, nz, 3)[::s, ::s, ::s]
nr, nt, nz = R_grid.shape

# rebuild flat arrays from the (sub)grid in matching C-order (rho outer, zeta inner)
rr, tt, zz = np.meshgrid(rho, theta, zeta, indexing="ij")
ep = np.stack([rr.ravel(), tt.ravel(), zz.ravel()], axis=1)
B = Bca_grid.reshape(-1, 3); Bco = Bco_grid.reshape(-1, 3); Bct = Bct_grid.reshape(-1, 3)
N = ep.shape[0]
bnorm = np.linalg.norm(B, axis=1)
rho_eval = np.minimum(rho, 1.0 - EVAL_EPS)      # nudge off the singular rho=1 knot

ep_eval = ep.copy(); ep_eval[:, 0] = np.minimum(ep_eval[:, 0], 1.0 - EVAL_EPS)
pts_eval = jnp.asarray(ep_eval)
print(f"[load] grid={nr}x{nt}x{nz} (stride={s})  N={N}")

def batched(fn, xs):
    return np.asarray(jax.lax.map(fn, xs, batch_size=mrx.MAP_BATCH_SIZE_INNER))

# ---------------------------------------------------------------------------
# Separable tensor-grid evaluation.  eval_points = rho (x) theta (x) zeta, and
# each fit basis is a tensor product, so a discrete function and its partials on
# the grid are three sequential 1D contractions with the per-axis (derivative)
# collocation matrices M[a,i] = b_i(pt_a),  D[a,i] = b_i'(pt_a).
# ---------------------------------------------------------------------------
def coll_val(b, pts):
    return jnp.asarray(b.collocation_matrix(jnp.asarray(pts)))          # (npt, nbasis)

def coll_der(b, pts):
    ns = b.ns
    g = lambda x: jax.vmap(lambda i: jax.grad(lambda xx: b(xx, i))(x))(ns)
    return jax.vmap(g)(jnp.asarray(pts))                                # (npt, nbasis)

def grid_eval(c, Mr, Mt, Mz):
    """Evaluate tensor-coeff scalar field c(nr,nt,nz) on the axis grids."""
    t = jnp.einsum('ai,ijk->ajk', Mr, c)
    t = jnp.einsum('bj,ajk->abk', Mt, t)
    t = jnp.einsum('ck,abk->abc', Mz, t)
    return t                                                            # (na,nb,nc)

def fit_coeffs(colls_solve, values_grid):
    """Interpolatory tensor-spline coefficients (per-axis square collocation)."""
    c = jnp.asarray(values_grid)
    for axis, coll in enumerate(colls_solve):
        c = _solve_tensor_collocation_axis(coll, c, axis=axis)
    return c

def map_and_DF_on_grid(cR, cZ, V, Der, zeta_axis, nfp, sign=TOR_SIGN):
    """F and DF for the stellarator map on the tensor grid, analytically.

    ``sign`` is the toroidal-angle handedness: Y = sign * R sin(2pi zeta/nfp).
    GVEC/standard cylindrical is +1 (right-handed, det DF > 0); MRX's
    stellarator_map hardcodes -1.

    V=(Vr,Vt,Vz) value colloc mats, Der=(Dr,Dt,Dz) derivative colloc mats.
    Returns F (na,nb,nc,3) and DF (na,nb,nc,3,3), reshape(-1,...) matches ep order.
    """
    Vr, Vt, Vz = V; Dr, Dt, Dz = Der
    R  = grid_eval(cR, Vr, Vt, Vz)
    Rr = grid_eval(cR, Dr, Vt, Vz); Rt = grid_eval(cR, Vr, Dt, Vz); Rz = grid_eval(cR, Vr, Vt, Dz)
    Z  = grid_eval(cZ, Vr, Vt, Vz)
    Zr = grid_eval(cZ, Dr, Vt, Vz); Zt = grid_eval(cZ, Vr, Dt, Vz); Zz = grid_eval(cZ, Vr, Vt, Dz)
    a = 2 * jnp.pi / nfp
    g = a * jnp.asarray(zeta_axis)                     # (nc,)
    cos = jnp.cos(g)[None, None, :]; sin = jnp.sin(g)[None, None, :]
    F = jnp.stack([R * cos, sign * R * sin, Z], axis=-1)
    DF = jnp.stack([
        jnp.stack([Rr * cos, Rt * cos, Rz * cos - R * a * sin], axis=-1),
        jnp.stack([sign * Rr * sin, sign * Rt * sin, sign * (Rz * sin + R * a * cos)], axis=-1),
        jnp.stack([Zr, Zt, Zz], axis=-1),
    ], axis=-2)
    return F, DF

# ---- build fit bases + per-axis value/derivative collocation matrices -------
FIT_NS = (nr, nt, nz)
fit_seq = DeRhamSequence(FIT_NS, (FIT_P,) * 3, FIT_Q, TYPES, polar=False)
fit_seq.evaluate_1d()
_br, _bt, _bz = fit_seq.basis_0.Λ
# square collocation at data nodes (for the coefficient solve)
colls_solve = (coll_val(_br, rho), coll_val(_bt, theta), coll_val(_bz, zeta))
# value + derivative collocation at the (rho nudged) eval grid
Vgrid = (coll_val(_br, rho_eval), coll_val(_bt, theta), coll_val(_bz, zeta))
Dgrid = (coll_der(_br, rho_eval), coll_der(_bt, theta), coll_der(_bz, zeta))

# coefficient tensors (interpolatory)
cR = fit_coeffs(colls_solve, R_grid); cZ = fit_coeffs(colls_solve, Z_grid)

# map (still needed as a callable for seq.set_map / seq.load quadrature).
# Built with the GVEC/standard +sin convention (TOR_SIGN) rather than MRX's
# stellarator_map (-sin), so det DF > 0 and B's Cartesian frame aligns.
R_h = DiscreteFunction(fit_seq.e0 @ cR.reshape(-1), fit_seq.basis_0, fit_seq.e0)
Z_h = DiscreteFunction(fit_seq.e0 @ cZ.reshape(-1), fit_seq.basis_0, fit_seq.e0)
_a_nfp = 2 * jnp.pi / NFP
def map_func(x):
    ang = _a_nfp * x[2]
    r = R_h(x)[0]
    return jnp.array([r * jnp.cos(ang), TOR_SIGN * r * jnp.sin(ang), Z_h(x)[0]])

# ---- (A) map/parametrization check (separable DF, no jacfwd) ----------------
_, DF_g = map_and_DF_on_grid(cR, cZ, Vgrid, Dgrid, zeta, NFP)
DF = np.asarray(DF_g).reshape(-1, 3, 3)
DFT_B = np.einsum("nij,nj->ni", np.transpose(DF, (0, 2, 1)), B)
DFinv = np.linalg.inv(DF)
DFinv_B = np.einsum("nij,nj->ni", DFinv, B)
J = np.linalg.det(DF)
rel = lambda a, b: np.linalg.norm(a - b, axis=1).mean() / np.linalg.norm(b, axis=1).mean()
print("=== (A) map/parametrization check ===")
print(f"  DF^T  B vs B_cov     rel={rel(DFT_B, Bco):.4e}")
print(f"  DF^-1 B vs B_contra  rel={rel(DFinv_B, Bct):.4e}")
print(f"  J: min={J.min():.3e} max={J.max():.3e}")

# Per-logical-component best-fit scale between MRX's (normalized [0,1]) DF
# pullbacks and GVEC's co/contravariant components.  Reveals the coordinate-
# normalization / metric factors: expect ~1 on rho and 2*pi (or 2*pi/nfp) on the
# angular axes if GVEC uses radians while MRX uses [0,1].
def per_comp_scale(a, b):
    return [float(np.sum(a[:, i] * b[:, i]) / np.sum(b[:, i] * b[:, i])) for i in range(3)]
sc = per_comp_scale(DFT_B, Bco)
si = per_comp_scale(DFinv_B, Bct)
print(f"  scale DF^T B / B_cov      : r={sc[0]:.4f} t={sc[1]:.4f} z={sc[2]:.4f}")
print(f"  scale DF^-1 B / B_contra  : r={si[0]:.4f} t={si[1]:.4f} z={si[2]:.4f}")
print(f"  (refs: 1={1.0:.4f}  2pi={2*np.pi:.4f}  2pi/nfp={2*np.pi/NFP:.4f}  "
      f"1/2pi={1/(2*np.pi):.4f}  nfp/2pi={NFP/(2*np.pi):.4f})")

# Coordinate-Jacobian between GVEC (theta in [0,2pi], zeta in [0,2pi/nfp]) and
# MRX (both normalized [0,1]).  Diagonal, so it's a per-axis metric factor:
#   MRX covariant     = S_cov     * B_cov_gvec     (lower index: d u_gvec / d u_mrx)
#   MRX contravariant = S_contra  * B_contra_gvec  (upper index: d u_mrx / d u_gvec)
SCOV = np.array([1.0, 2 * np.pi, 2 * np.pi / NFP])
SCON = 1.0 / SCOV
print(f"  rel DF^T B  vs S_cov*B_cov     = {rel(DFT_B, Bco * SCOV):.4e}")
print(f"  rel DF^-1 B vs S_con*B_contra  = {rel(DFinv_B, Bct * SCON):.4e}")

# sanity: analytic DF vs autodiff on a small subsample
sub_idx = np.arange(0, N, 1503)
DF_ad = batched(jax.jacfwd(map_func), jnp.asarray(ep_eval[sub_idx]))
d = np.abs(np.asarray(DF_ad) - DF[sub_idx]).max()
print(f"  analytic DF vs jacfwd (subsample) max|diff|={d:.2e}")

def make_ref_fn(grid):
    hs = [DiscreteFunction(fit_seq.e0 @ fit_coeffs(colls_solve, grid[..., i]).reshape(-1),
                           fit_seq.basis_0, fit_seq.e0) for i in range(3)]
    return lambda x: jnp.array([hs[0](x)[0], hs[1](x)[0], hs[2](x)[0]])

# ---- build projection sequence --------------------------------------------
print("\n=== (B) project onto V2 ===", flush=True)
seq = DeRhamSequence(NS, (P,) * 3, 2 * P, TYPES, polar=True,
                     tol=CG_TOL, maxiter=CG_MAXITER, betti_numbers=(1, 1, 0, 0))
seq.evaluate_1d(); seq.set_map(map_func)
ops = seq.get_operators()
ops = assemble_mass_surgery_preconditioner(seq, operators=ops, ks=(0, 1, 2))
ops = assemble_tensor_mass_preconditioner(
    seq, operators=ops, ks=(0, 1, 2, 3), cp_kwargs={"greville": True})
seq.set_operators(ops)

def report_dual(label, dual):
    dof, info = seq.apply_inverse_mass_matrix(dual, 2, dirichlet=False, return_info=True)
    # pushforward = DF * (ref proxy) / J, using the separable analytic DF/J
    proxy = batched(DiscreteFunction(dof, seq.basis_2, seq.e2), pts_eval)   # spline eval only
    B_rec = np.einsum("nij,nj->ni", DF, proxy) / J[:, None]
    e = np.linalg.norm(B_rec - B, axis=1)
    prof = np.array([e[ep[:, 2] == v].mean() / bnorm[ep[:, 2] == v].mean()
                     for v in np.unique(ep[:, 2])])
    print(f"[{label}] iters={abs(int(info))}  mean rel={e.mean()/bnorm.mean():.4e}  "
          f"max rel={(e/bnorm).max():.4e}  zeta-spread={(prof.max()-prof.min())/prof.mean():.0%}")
    return B_rec

def project_and_report(label, f_ref, frame):     # pointwise seq.load(callable)
    return report_dual(label, seq.load(f_ref, 2, frame=frame))

# PRIMARY route: build the MRX covariant proxy ourselves as DF^T B on the data
# grid.  Exact (uses our verified DF + raw Cartesian B, no normalization guess)
# and periodic (DF's zeta-rotation cancels B's: (R DF0)^T (R B0) = DF0^T B0), so
# interpolating it periodically is seam-free -- unlike phys, which splines the
# quasi-periodic B *before* applying DF.
DFT_B_grid = DFT_B.reshape(nr, nt, nz, 3)
project_and_report("ref :DF^T B       ", make_ref_fn(DFT_B_grid), 'ref')

# Same field, but via the factorized library load (no pointwise eval): fits the
# interpolatory spline and evaluates it at the quad grid by 3 1D contractions.
# Should match "ref :DF^T B" to ~fp; this is the production path.
dual_lgf = load_grid_field((rho, theta, zeta), DFT_B_grid, seq, 2, frame='ref')
report_dual("ref :DF^T B (lgf) ", dual_lgf)
print(f"  |lgf - pointwise dual| / |dual| = "
      f"{float(np.linalg.norm(dual_lgf - seq.load(make_ref_fn(DFT_B_grid), 2, frame='ref')) / np.linalg.norm(dual_lgf)):.2e}")

# file's covariant rescaled by the (theoretical) diagonal -- theta factor is wrong
project_and_report("ref :S_cov*B_cov  ", make_ref_fn(Bco_grid * SCOV), 'ref')
# file's covariant raw -- seam-free but off by the metric/normalization factor
project_and_report("ref :B_cov (raw)  ", make_ref_fn(Bco_grid), 'ref')
# baseline: old Cartesian frame='phys' (accurate but has the zeta=0 seam)
project_and_report("phys:B_cart (seam)", make_ref_fn(Bca_grid), 'phys')
