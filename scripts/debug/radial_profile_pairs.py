"""Radial metric profiles and their pairwise proportionality.

Decides whether the modal-radial k=0 atom needs per-mode radial solves at all.

In ``A_jk = K_r[a] + mu_j M_r[b] + nu_k M_r[c]`` the ``(j,k)`` dependence enters
through TWO scalars, which is what forces one radial solve per mode
(``O(N p)`` storage banded, ``O(N n_r)`` dense). If the two profiles multiplying
the MASS terms are proportional, ``c(r) = kappa b(r)``, it becomes ONE scalar::

    A_jk = K_r[a] + (mu_j + kappa nu_k) M_r[b]
    =>  A_jk^-1 = V_r diag(1 / (lam_r + mu_j + kappa nu_k)) V_r^T

i.e. full 3D fast diagonalization at ``O(n_r^2 + n_t^2 + n_z^2)`` storage --
production's cost, but keeping the radial weighting production throws away.

So the number that matters is the **spread of c/b**, not of a/b or a/c.

Weight families reported (all as harmonic means over theta and zeta):
    Jginv  w_i = g^{ii} J     -- the k=0 Laplacian, and the k=1 mass weight
    ginvJ  v_i = g_{ii} / J   -- the k=2 mass weight
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np

import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map, toroid_map
from mrx.operators import (_reshape_quadrature_matrix_field,
                           _reshape_quadrature_scalar_field)

p = argparse.ArgumentParser()
p.add_argument("--ns", type=int, nargs=3, default=(8, 16, 16))
p.add_argument("--p", type=int, default=3)
p.add_argument("--h5", default="data/W7X-vacuum.h5")
p.add_argument("--stride", type=int, default=4)
p.add_argument("--map-batch", type=int, default=256)
args = p.parse_args()
mrx.MAP_BATCH_SIZE_INNER = args.map_batch      # set_map is the memory hot spot
NS, P = tuple(args.ns), args.p
TYPES = ("clamped", "periodic", "periodic")
print(f"radial profile pairs: ns={NS} p={P} map_batch={args.map_batch}\n", flush=True)


def harmonic_over_angles(vals, seq):
    """Quadrature-weighted harmonic mean over theta and zeta -> profile in r."""
    x = 1.0 / jnp.clip(vals, 1e-30)
    x = jnp.tensordot(x, seq.quad.w_z, axes=([2], [0]))
    x = jnp.tensordot(x, seq.quad.w_y, axes=([1], [0]))
    return (jnp.sum(seq.quad.w_y) * jnp.sum(seq.quad.w_z)) / jnp.clip(x, 1e-30)


def report(name, seq):
    jac = np.asarray(seq.geometry.jacobian_j)
    if not np.all(np.isfinite(jac)) or jac.min() <= 0.0:
        print(f"{name}: SKIP (det(J) not positive/finite)\n", flush=True)
        return
    minv = jnp.transpose(_reshape_quadrature_matrix_field(
        seq, seq.geometry.metric_inv_jkl), (1, 0, 2, 3, 4))
    met = jnp.transpose(_reshape_quadrature_matrix_field(
        seq, seq.geometry.metric_jkl), (1, 0, 2, 3, 4))
    jq = jnp.transpose(_reshape_quadrature_scalar_field(
        seq, seq.geometry.jacobian_j), (1, 0, 2))
    r = np.asarray(seq.quad.x_x)

    for fam, w in (("Jginv  (k=0 Lap / k=1 mass)",
                    [minv[..., i, i] * jq for i in range(3)]),
                   ("ginvJ  (k=2 mass)",
                    [met[..., i, i] / jq for i in range(3)])):
        prof = [np.asarray(harmonic_over_angles(wi, seq)) for wi in w]
        a, b, c = prof
        print(f"  {fam}", flush=True)
        # log-log slope against r, away from the axis, to name the shape
        m = (r > 0.15) & (r < 0.95)
        for lbl, v in (("rr(a)", a), ("tt(b)", b), ("zz(c)", c)):
            sl = np.polyfit(np.log(r[m]), np.log(np.abs(v[m])), 1)[0]
            print(f"    {lbl}: range [{v.min():.3e}, {v.max():.3e}]  "
                  f"d log w / d log r = {sl:+.2f}", flush=True)
        for lbl, x, y in (("c/b  (THE one that matters)", c, b),
                          ("a/b", a, b), ("a/c", a, c)):
            ratio = x[m] / y[m]
            spread = ratio.max() / ratio.min()
            print(f"    {lbl:28s} mean={ratio.mean():.4g}  "
                  f"max/min={spread:8.2f}  "
                  f"{'PROPORTIONAL' if spread < 1.2 else ''}", flush=True)
    print(flush=True)


def w7x_map(path, stride):
    import h5py  # noqa: PLC0415
    from mrx.differential_forms import DiscreteFunction  # noqa: PLC0415
    from mrx.projectors import _solve_tensor_collocation_axis  # noqa: PLC0415
    with h5py.File(path, "r") as h:
        at = dict(h.attrs)
        nfp = int(at["nfp"])
        nr, nt, nz = (int(at[k]) for k in ("n_rho", "n_theta", "n_zeta"))
        R = np.asarray(h["R"]).reshape(nr, nt, nz)
        Z = np.asarray(h["Z"]).reshape(nr, nt, nz)
        ep = np.asarray(h["eval_points"]).reshape(nr, nt, nz, 3)
    sl = (slice(None, None, stride),) * 3
    R, Z, ep = R[sl], Z[sl], ep[sl]
    rho, theta, zeta = ep[:, 0, 0, 0], ep[0, :, 0, 1], ep[0, 0, :, 2]
    R0 = float(np.mean(R))
    R, Z = R / R0, Z / R0
    fs = DeRhamSequence((len(rho), len(theta), len(zeta)), (P,) * 3, 2 * P,
                        TYPES, polar=False)
    fs.evaluate_1d()
    br, bt, bz = fs.basis_0.Λ
    colls = (jnp.asarray(br.collocation_matrix(jnp.asarray(rho))),
             jnp.asarray(bt.collocation_matrix(jnp.asarray(theta))),
             jnp.asarray(bz.collocation_matrix(jnp.asarray(zeta))))

    def fit(v):
        c = jnp.asarray(v)
        for ax, cm in enumerate(colls):
            c = _solve_tensor_collocation_axis(cm, c, axis=ax)
        return c
    Rh = DiscreteFunction(fs.e0 @ fit(R).reshape(-1), fs.basis_0, fs.e0)
    Zh = DiscreteFunction(fs.e0 @ fit(Z).reshape(-1), fs.basis_0, fs.e0)
    an = 2 * jnp.pi / nfp

    def make(sg):
        def f(x):
            ang = an * x[2]
            rr = Rh(x)[0]
            return jnp.array([rr * jnp.cos(ang), sg * rr * jnp.sin(ang), Zh(x)[0]])
        return f
    rng = np.random.default_rng(0)
    pts = jnp.asarray(rng.uniform(0.05, 0.95, size=(256, 3)))
    d = jax.vmap(lambda x: jnp.linalg.det(jax.jacfwd(make(1.0))(x)))(pts)
    sg = 1.0 if float(jnp.median(d)) > 0 else -1.0
    print(f"  [w7x] nfp={nfp} sign={sg:+.0f}", flush=True)
    return make(sg)


for name, mk in (("toroid", lambda: toroid_map(epsilon=1 / 3, R0=1.0)),
                 ("rot-ellipse nfp3", lambda: rotating_ellipse_map(
                     eps=0.33, kappa=1.5, nfp=3)),
                 ("W7-X", lambda: w7x_map(args.h5, args.stride))):
    print(f"===== {name} =====", flush=True)
    try:
        seq = DeRhamSequence(NS, (P,) * 3, 2 * P, TYPES, polar=True,
                             tol=1e-12, maxiter=1000, betti_numbers=(1, 1, 0, 0))
        seq.evaluate_1d()
        seq.set_map(mk())
    except Exception as exc:                                    # noqa: BLE001
        print(f"  BUILD FAILED: {type(exc).__name__}: {exc}\n", flush=True)
        continue
    report(name, seq)
