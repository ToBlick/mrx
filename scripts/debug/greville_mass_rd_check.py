"""Quick sanity check: radial-dense vs single-D Greville on the MASS bulk.

The mass is a single weighted Kronecker term M[W] (one channel), so "radial-dense"
(keep r exact, FD-unweighted theta,zeta) reduces to a separable inverse with the
radial direction carrying the EXACT angular-mean radial weight profile and the
angular directions UNWEIGHTED:
    rd:     M_r[rho]^{-1} x M_t^{-1} x M_z^{-1},  rho(r)=<W>_{theta,zeta}
    single: D^{-1/2} (M0_r^{-1} x M0_t^{-1} x M0_z^{-1}) D^{-1/2}
Unlike single-D, rd drops W's angular variation entirely. Compare CG iters.
"""
from __future__ import annotations
import argparse, os, sys
from types import SimpleNamespace
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "benchmark"))
sys.path.insert(0, HERE)

from benchmark_graddiv_k1_preconditioner import build_sequence
from greville_bulk_precond import component_specs, build_greville_component, true_applies, pcg
from mrx.operators import assemble_mass_surgery_preconditioner
from mrx.preconditioners import (
    _assemble_weighted_1d_mass, _restrict_radial_mass,
    _k0_bulk_weight_tensor, _k3_weight_tensor,
    _k1_diagonal_metric_tensors, _k2_diagonal_metric_tensors,
)


def weight_tensor_for(seq, k, name):
    if k == 0:
        return _k0_bulk_weight_tensor(seq)
    if k == 3:
        return _k3_weight_tensor(seq)
    if k == 1:
        return _k1_diagonal_metric_tensors(seq)[{"arr": "alpha_rr", "theta": "alpha_thetatheta", "zeta": "alpha_zetazeta"}[name]]
    return _k2_diagonal_metric_tensors(seq)[{"r": "beta_rr", "theta": "beta_thetatheta", "zeta": "beta_zetazeta"}[name]]


def build_rd_component(seq, spec, k):
    nr, ntc, nzc = (int(s) for s in spec["shape"])
    diff = spec["diff"]
    radial_start = 1 if diff[0] else 2
    rbas = seq.d_basis_r_jk if diff[0] else seq.basis_r_jk
    tbas = seq.d_basis_t_jk if diff[1] else seq.basis_t_jk
    zbas = seq.d_basis_z_jk if diff[2] else seq.basis_z_jk
    W = weight_tensor_for(seq, k, spec["name"])           # (ny,nx,nz) = (theta,r,zeta)
    rho = jnp.mean(W, axis=(0, 2))                        # (nx,) radial-quad profile
    M_r = _restrict_radial_mass(_assemble_weighted_1d_mass(rbas, seq.quad.w_x * rho), radial_start, nr)
    M_t = _assemble_weighted_1d_mass(tbas, seq.quad.w_y)
    M_z = _assemble_weighted_1d_mass(zbas, seq.quad.w_z)
    inv_r, inv_t, inv_z = jnp.linalg.inv(M_r), jnp.linalg.inv(M_t), jnp.linalg.inv(M_z)

    def apply(v):
        f = jnp.asarray(v).reshape(nr, ntc, nzc)
        f = jnp.einsum("ij,jkl->ikl", inv_r, f)
        f = jnp.einsum("ij,kjl->kil", inv_t, f)
        f = jnp.einsum("ij,klj->kli", inv_z, f)
        return f.reshape(-1)
    return apply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="toroid")
    ap.add_argument("--ns", type=int, nargs=3, default=[12, 24, 12])
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--k", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--epsilon", type=float, default=1.0/3.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--nfp", type=int, default=3)
    ap.add_argument("--dirichlet", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-10)
    args = ap.parse_args()

    cfg = SimpleNamespace(ns=tuple(args.ns), p=args.p, geometry=args.geometry, cg_tol=args.tol,
                          cg_maxiter=10000, epsilon=args.epsilon, kappa=args.kappa, r0=args.r0, nfp=args.nfp)
    print(f"=== MASS rd-vs-single  {args.geometry} ns={tuple(args.ns)} p={args.p} dbc={args.dirichlet} ===", flush=True)
    seq = build_sequence(cfg)
    ops = seq.get_operators()
    surg_ks = tuple(k for k in args.k if k in (0, 1, 2))
    surg = assemble_mass_surgery_preconditioner(seq, operators=ops, ks=surg_ks) if surg_ks else ops
    print(f"\n{'k':>2} {'comp':6} {'n':>6} {'single_it':>10} {'rd_it':>7}")
    for k in args.k:
        truths = true_applies(seq, surg, k, args.dirichlet)
        for spec in component_specs(seq, k, args.dirichlet):
            nm = spec["name"]; A, n = truths[nm]
            single, _, _ = build_greville_component(seq, spec)
            rd = build_rd_component(seq, spec, k)
            it_s, _ = pcg(A, single, n, tol=args.tol)
            it_r, _ = pcg(A, rd, n, tol=args.tol)
            print(f"{k:>2} {nm:6} {n:>6d} {it_s:>10d} {it_r:>7d}", flush=True)


if __name__ == "__main__":
    main()
