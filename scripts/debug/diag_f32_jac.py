"""Where does the f32 W7-X quadrature Jacobian go non-positive?"""
import numpy as np  # run with EXTRA_ENV="MRX_DTYPE=float32" or float64
import mrx
from mrx.geometries import build_sequence
print("dtype", mrx.DTYPE, flush=True)
try:
    seq, ops = build_sequence("w7x-fmm002", (8, 16, 8), 3, 10000)
except RuntimeError as e:
    print("build_sequence raised:", e, flush=True)
    # rebuild without the check to inspect
    from mrx.derham_sequence import DeRhamSequence
    from mrx.gvec import build_gvec_map, gvec_path
    from mrx.geometries import GVEC_NFP_OVERRIDE
    seq = DeRhamSequence((8, 16, 8), (3, 3, 3), 6, ("clamped", "periodic", "periodic"),
                         polar=True, tol=None, maxiter=10000, betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    map_func, info = build_gvec_map(gvec_path("w7x-fmm002"), map_ns=(8, 16, 8), p=3,
                                    nfp=GVEC_NFP_OVERRIDE.get("w7x-fmm002"))
    seq.set_map(map_func)
jac = np.asarray(seq.geometry.jacobian_j)
x = np.asarray(seq.quad.x)
print("n_quad", jac.size, "dtype", jac.dtype)
print("finite", np.isfinite(jac).all(), "min", np.nanmin(jac), "max", np.nanmax(jac))
bad = np.where(~np.isfinite(jac) | (jac <= 0))[0]
print("n_bad", bad.size)
if bad.size:
    print("bad rho range", x[bad, 0].min(), x[bad, 0].max())
    print("bad theta range", x[bad, 1].min(), x[bad, 1].max())
    print("bad zeta range", x[bad, 2].min(), x[bad, 2].max())
    print("first 5 bad:", [(float(x[i,0]), float(x[i,1]), float(x[i,2]), float(jac[i])) for i in bad[:5]])
# where are the smallest positive ones
order = np.argsort(jac)[:5]
print("5 smallest:", [(float(x[i,0]), float(x[i,1]), float(x[i,2]), float(jac[i])) for i in order])
print("rho of smallest-|jac| decile: min rho", x[np.argsort(np.abs(jac))[:jac.size//10], 0].min())
