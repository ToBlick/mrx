"""Diagnose the W7-X Greville-point Jacobian (why greville mass D^{-1/2} -> NaN).

Compares J at QUADRATURE points (what CP uses, finite/positive) vs at the bulk
0-form GREVILLE abscissae (what the Greville collocation diagonal D uses).
"""
import os, sys
from types import SimpleNamespace
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "benchmark"))
from benchmark_graddiv_k1_preconditioner import build_sequence
from mrx.geometry import compute_geometry_terms
from mrx.preconditioners import _bulk_tensor_shape

geo = sys.argv[1] if len(sys.argv) > 1 else "w7x"
cfg = SimpleNamespace(ns=(12, 24, 12), p=3, geometry=geo, cg_tol=1e-10,
                      cg_maxiter=10000, epsilon=1.0/3.0, kappa=1.0, r0=1.0, nfp=3)
seq = build_sequence(cfg)

jq = np.asarray(jax.device_get(seq.geometry.jacobian_j))
print(f"[{geo}] J at QUAD points: min={jq.min():.4e} max={jq.max():.4e} "
      f"nan={np.isnan(jq).any()} npos<=0={(jq<=0).sum()}", flush=True)

nr_bulk, nt, nz = _bulk_tensor_shape(seq, False)
gr = seq.basis_0.Λ[0].greville_points()[2:2+nr_bulk]
gt = seq.basis_0.Λ[1].greville_points()
gz = seq.basis_0.Λ[2].greville_points()
print(f"greville r range [{float(gr.min()):.4f},{float(gr.max()):.4f}] "
      f"t [{float(gt.min()):.4f},{float(gt.max()):.4f}] "
      f"z [{float(gz.min()):.4f},{float(gz.max()):.4f}]", flush=True)
rr, tt, zz = jnp.meshgrid(gr, gt, gz, indexing="ij")
pts = jnp.stack([rr.ravel(), tt.ravel(), zz.ravel()], axis=-1)
_, _, jg = compute_geometry_terms(seq.map, pts)
jg = np.asarray(jax.device_get(jg))
bad = ~np.isfinite(jg) | (jg <= 0)
print(f"[{geo}] J at GREVILLE points: min={np.nanmin(jg):.4e} max={np.nanmax(jg):.4e} "
      f"nan={np.isnan(jg).any()} n_bad(<=0 or nan)={bad.sum()}/{jg.size}", flush=True)
if bad.any():
    bp = np.asarray(pts)[bad]
    print("  example bad points (r,t,z) and J:", flush=True)
    for i in np.where(bad)[0][:8]:
        print(f"    {np.asarray(pts)[i]}  J={jg[i]:.4e}", flush=True)
