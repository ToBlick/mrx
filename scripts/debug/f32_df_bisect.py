"""Bisect the f32 loss of accuracy in DF of the W7-X spline map.

Mode 'ref' (run with MRX_DTYPE=float64): fit the map, save the R,Z dofs and
the reference values / DF / det at a set of probe points (the innermost
quadrature ring of the p+1 rule plus random outer points).

Mode 'f32' (run with MRX_DTYPE=float32): build the map from the f64 dofs cast
to float32 and evaluate the same probes twice -- with JAX's default GPU
matmul precision (TF32 on Ampere+) and with 'highest' -- and report the
errors against the reference, component by component.  Then rebuild the
full quadrature Jacobian both ways and count the non-positive points.
"""
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.gvec import build_gvec_map, gvec_path, _map_with_sign
from mrx.geometries import GVEC_NFP_OVERRIDE
from mrx.spline_bases import _nonzero_bsplines

mode = sys.argv[1]
assert (mode == "ref") == (mrx.DTYPE == jnp.float64), (mode, mrx.DTYPE)
ns, p = (8, 16, 8), 3
REF = os.path.join(os.environ["MRX_ROOT"], "outputs", "f32_df_bisect_ref.npz")
print("devices", jax.devices(), "dtype", mrx.DTYPE, flush=True)


def make_seq():
    seq = DeRhamSequence(ns, (p,) * 3, p + 1, ("clamped", "periodic", "periodic"),
                         polar=True, maxiter=1000, betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    return seq


def probes(seq):
    x = np.asarray(seq.quad.x, dtype=np.float64)
    inner = np.where(x[:, 0] < 0.02)[0]
    rng = np.random.default_rng(0)
    outer = rng.choice(np.where(x[:, 0] > 0.1)[0], 256, replace=False)
    return x[inner], x[outer]


def evaluate(R_h, Z_h, F, pts):
    pts = jnp.asarray(pts, dtype=mrx.DTYPE)
    R = jax.vmap(lambda x: R_h(x)[0])(pts)
    Z = jax.vmap(lambda x: Z_h(x)[0])(pts)
    dR = jax.vmap(jax.jacfwd(lambda x: R_h(x)[0]))(pts)
    dZ = jax.vmap(jax.jacfwd(lambda x: Z_h(x)[0]))(pts)
    DF = jax.vmap(jax.jacfwd(F))(pts)
    det = jnp.linalg.det(DF)
    return {k: np.asarray(v) for k, v in
            dict(R=R, Z=Z, dR=dR, dZ=dZ, DF=DF, det=det).items()}


def basis_1d(bases, pts):
    """1-D basis values and their jacfwd derivatives on the p+1 nonzero functions."""
    out = {}
    for d, b in enumerate(bases):
        xs = jnp.asarray(pts[:, d], dtype=mrx.DTYPE)
        if b.type == 'periodic':
            xs = jnp.mod(xs, 1.0)
        val = jax.vmap(lambda x: _nonzero_bsplines(b.T, b.p, x)[0])(xs)
        der = jax.vmap(jax.jacfwd(lambda x: _nonzero_bsplines(b.T, b.p, x)[0]))(xs)
        out[f"val{d}"], out[f"der{d}"] = np.asarray(val), np.asarray(der)
    return out


def rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


seq = make_seq()
inner, outer = probes(seq)
print(f"probes: {len(inner)} innermost-ring points (rho={inner[0,0]:.4e}), {len(outer)} outer", flush=True)

if mode == "ref":
    map_func, info = build_gvec_map(gvec_path("w7x-fmm002"), map_ns=ns, p=p,
                                    nfp=GVEC_NFP_OVERRIDE.get("w7x-fmm002"))
    R_h, Z_h = info["R_h"], info["Z_h"]
    bases = R_h.Λ.bases[0].bases
    ref = {"R": np.asarray(R_h.dof), "Z": np.asarray(Z_h.dof),
           "sign": info["sign"], "nfp": info["nfp"], "inner": inner, "outer": outer}
    for tag, pts in (("in", inner), ("out", outer)):
        for k, v in evaluate(R_h, Z_h, map_func, pts).items():
            ref[f"{tag}_{k}"] = v
        for k, v in basis_1d(bases, pts).items():
            ref[f"{tag}_{k}"] = v
    np.savez(REF, **ref)
    seq.set_map(map_func)
    jac = np.asarray(seq.geometry.jacobian_j)
    print(f"[f64] quad det min={jac.min():.6e} max={jac.max():.6e} "
          f"inner ring [{ref['in_det'].min():.6e}, {ref['in_det'].max():.6e}]", flush=True)
    sys.exit(0)

for _ in range(60):
    if os.path.exists(REF):
        break
    time.sleep(10)
d = np.load(REF)
inner, outer = d["inner"], d["outer"]
# the reference dofs, cast to f32 -- the split diagnostic showed the f32 fit is the same to 1e-6
map_seq = DeRhamSequence(ns, (p,) * 3, p + 1, ("clamped", "periodic", "periodic"), polar=False)
map_seq.evaluate_1d()
R_h = DiscreteFunction(jnp.asarray(d["R"], dtype=mrx.DTYPE), map_seq.basis_0, map_seq.e0)
Z_h = DiscreteFunction(jnp.asarray(d["Z"], dtype=mrx.DTYPE), map_seq.basis_0, map_seq.e0)
F = _map_with_sign(R_h, Z_h, int(d["nfp"]), float(d["sign"]))
bases = R_h.Λ.bases[0].bases

for prec in ("default", "highest"):
    with jax.default_matmul_precision(prec):
        print(f"\n===== matmul precision: {prec}", flush=True)
        for tag, pts in (("in", inner), ("out", outer)):
            got = evaluate(R_h, Z_h, F, pts)
            b1 = basis_1d(bases, pts)
            print(f"[{tag}] 1-D basis: " + " ".join(
                f"{k}={rel(b1[k], d[f'{tag}_{k}']):.1e}" for k in sorted(b1)), flush=True)
            print(f"[{tag}] values: R {rel(got['R'], d[f'{tag}_R']):.2e}  Z {rel(got['Z'], d[f'{tag}_Z']):.2e}",
                  flush=True)
            for name in ("dR", "dZ"):
                comps = [rel(got[name][:, i], d[f'{tag}_{name}'][:, i]) for i in range(3)]
                print(f"[{tag}] {name}: rel err per (rho,theta,zeta) component = "
                      + " ".join(f"{c:.2e}" for c in comps)
                      + "   |ref| max per comp = "
                      + " ".join(f"{np.abs(d[f'{tag}_{name}'][:, i]).max():.2e}" for i in range(3)), flush=True)
            print(f"[{tag}] DF rel err {rel(got['DF'], d[f'{tag}_DF']):.2e}; "
                  f"det: got [{got['det'].min():.4e}, {got['det'].max():.4e}] "
                  f"ref [{d[f'{tag}_det'].min():.4e}, {d[f'{tag}_det'].max():.4e}] "
                  f"max|diff| {np.abs(got['det'] - d[f'{tag}_det']).max():.2e}", flush=True)
        seq.set_map(F)
        jac = np.asarray(seq.geometry.jacobian_j)
        x = np.asarray(seq.quad.x)
        ring = jac[x[:, 0] < 0.02]
        print(f"[quad] n={jac.size} det min={jac.min():.6e} max={jac.max():.6e} "
              f"n_bad={(jac <= 0).sum()} inner ring [{ring.min():.6e}, {ring.max():.6e}]", flush=True)
