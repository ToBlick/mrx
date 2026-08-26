"""W7-X map fit: plain tensor-product collocation vs. tensor collocation
restricted onto the C1 polar space, in the working precision.

Run once with MRX_DTYPE=float64 (saves the f64 raw coefficients) and once
with MRX_DTYPE=float32 (compares against them).  Reports per-ring
coefficient differences, fit wall times, and the quadrature Jacobian of both
maps on the p+1 and 2p Gauss rules.
"""
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.gvec import _det_DF, _map_with_sign, _rgi_fn, gvec_path, load_gvec_grids
from mrx.geometries import GVEC_NFP_OVERRIDE

ns, p = (8, 16, 8), 3
REF = os.path.join(os.environ["MRX_ROOT"], "outputs", "mapfit_polar_vs_tensor_f64.npz")
print("devices", jax.devices(), "dtype", mrx.DTYPE, flush=True)

axes, R_grid, Z_grid, nfp, _ = load_gvec_grids(
    gvec_path("w7x-fmm002"), nfp=GVEC_NFP_OVERRIDE.get("w7x-fmm002"))
R_fn, Z_fn = _rgi_fn(axes, R_grid), _rgi_fn(axes, Z_grid)


def fit(polar):
    kw = dict(betti_numbers=(1, 1, 0, 0), maxiter=1000) if polar else {}
    seq = DeRhamSequence(ns, (p,) * 3, p + 1, ("clamped", "periodic", "periodic"),
                         polar=polar, **kw)
    seq.evaluate_1d()
    times = []
    for _ in range(2):                      # first call includes compilation
        t0 = time.perf_counter()
        R_dof = seq.interpolate(R_fn, 0).block_until_ready()
        Z_dof = seq.interpolate(Z_fn, 0).block_until_ready()
        times.append(time.perf_counter() - t0)
    R_h = DiscreteFunction(R_dof, seq.basis_0, seq.e0)
    Z_h = DiscreteFunction(Z_dof, seq.basis_0, seq.e0)
    print(f"[fit polar={polar}] n_dof={R_dof.size} wall cold={times[0]:.2f}s warm={times[1]:.3f}s",
          flush=True)
    return R_h, Z_h


maps = {}
for polar in (False, True):
    R_h, Z_h = fit(polar)
    F = _map_with_sign(R_h, Z_h, nfp, -1.0)
    d = _det_DF(F)
    print(f"[polar={polar}] sampled det DF in [{d.min():.4e}, {d.max():.4e}]", flush=True)
    maps[polar] = (np.asarray(R_h.raw[0]), np.asarray(Z_h.raw[0]), F)

Rt, Zt, _ = maps[False]
Rp, Zp, _ = maps[True]
print("tensor vs polar-restricted raw coefficients, max |diff| per radial ring:", flush=True)
for i in range(ns[0]):
    print(f"  ring {i}: R {np.abs(Rt[i] - Rp[i]).max():.3e}  Z {np.abs(Zt[i] - Zp[i]).max():.3e}"
          f"   (theta-spread of tensor ring: R {np.ptp(Rt[i], axis=0).max():.3e} "
          f"Z {np.ptp(Zt[i], axis=0).max():.3e})", flush=True)

if mrx.DTYPE == jnp.float64:
    np.savez(REF, Rt=Rt, Zt=Zt, Rp=Rp, Zp=Zp)
else:
    for _ in range(60):
        if os.path.exists(REF):
            break
        time.sleep(10)
    ref = np.load(REF)
    print("f32 vs f64 raw coefficients: "
          f"tensor R {np.abs(Rt - ref['Rt']).max():.2e} Z {np.abs(Zt - ref['Zt']).max():.2e}; "
          f"polar R {np.abs(Rp - ref['Rp']).max():.2e} Z {np.abs(Zp - ref['Zp']).max():.2e}", flush=True)

for rule, label in ((p + 1, "p+1"), (2 * p, "2p")):
    seq = DeRhamSequence(ns, (p,) * 3, rule, ("clamped", "periodic", "periodic"),
                         polar=True, maxiter=1000, betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    x = np.asarray(seq.quad.x)
    rho0 = x[:, 0].min()
    inner = x[:, 0] == rho0
    for polar in (False, True):
        seq.set_map(maps[polar][2])
        jac = np.asarray(seq.geometry.jacobian_j)
        print(f"[{label} rule, polar={polar}] n={jac.size} det min={jac.min():.6e} "
              f"max={jac.max():.6e} n_bad={(jac <= 0).sum()} "
              f"innermost ring rho={rho0:.4e}: [{jac[inner].min():.6e}, {jac[inner].max():.6e}]",
              flush=True)
