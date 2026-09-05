"""Independent check that the quadrature-order migration changed no number.

``dump`` builds the W7-X (12, 24, 12) p=3 sequence and writes to an ``.npz``
everything that touches the quadrature order: the geometry at the quadrature
points, ``M_k x``, ``L_k x``, the metric-lumping preconditioner applies, the
harmonic forms with their Rayleigh quotients, ``load`` / ``interpolate`` of
smooth analytic forms, and one cross-product load. Run it once on the
theta-major base commit and once on the r-major branch; ``compare`` permutes
the base's quadrature-ordered fields ``(theta, r, zeta) -> (r, theta, zeta)``
and prints the max relative difference per item. DOF-space vectors need no
permutation; the harmonic forms are compared up to sign.

    python scripts/quad_order_equivalence.py dump OUT.npz [--geometry ...]
    python scripts/quad_order_equivalence.py compare BASE.npz NEW.npz

Uses only API present on both sides (no ``seq.quad.shape``).
"""

import argparse
import sys

import numpy as np

W7X = "data/wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
KS = ((0, False), (0, True), (1, False), (1, True),
      (2, False), (2, True), (3, False), (3, True))


def dump(out, geometry, ns, p):
    import jax
    import jax.numpy as jnp

    import mrx
    from mrx.geometry import build_sequence, map_jacobian_at
    from mrx.nullspace import compute_nullspaces, get_nullspace, harmonic_rayleigh

    print(f"[env] mrx from {mrx.__file__}  precision {mrx.DTYPE}", flush=True)
    seq, ops = build_sequence(geometry, ns, p)
    d = {"s_nq": np.array([seq.quad.nx, seq.quad.ny, seq.quad.nz])}

    # (1) geometry at the quadrature points
    d["q_x"] = seq.quad.x
    d["q_w"] = seq.quad.w
    d["q_jacobian_j"] = seq.jacobian_j
    d["q_metric_jkl"] = seq.metric_jkl
    d["q_metric_inv_jkl"] = seq.metric_inv_jkl
    d["q_DF"] = map_jacobian_at(seq.map, seq.quad.x)

    # (2) + (4) operator and preconditioner applies on seeded random vectors
    rng = np.random.default_rng(0)
    for k, dbc in KS:
        tag = f"k{k}_{'dbc' if dbc else 'free'}"
        x = jnp.asarray(rng.standard_normal(seq.n(k, dbc)), dtype=mrx.DTYPE)
        d[f"d_x_{tag}"] = x
        d[f"d_M_{tag}"] = seq.apply_mass_matrix(x, k, dbc)
        d[f"d_L_{tag}"] = seq.apply_laplacian(x, k, dbc)
        d[f"d_PM_{tag}"] = seq.apply_mass_matrix_preconditioner(x, k, dbc)
        d[f"d_PL_{tag}"] = seq.apply_laplacian_preconditioner(x, k, dbc)
        if k in (1, 2):
            d[f"q_eval_{tag}"] = seq.evaluate_at_quadrature(x, k, dbc)
        print(f"[apply] {tag} done", flush=True)

    # (3) harmonic forms and their Rayleigh quotients / lambda_1 lines
    ops = compute_nullspaces(seq, gap_sweeps=5, verbose=True)
    for k, dbc in ((3, True), (2, True), (0, False), (1, False)):
        v = get_nullspace(ops, k, dbc)
        tag = f"k{k}_{'dbc' if dbc else 'free'}"
        d[f"n_{tag}"] = v
        d[f"s_rq_{tag}"] = np.array([float(harmonic_rayleigh(seq, vi, k, dbc, ops)) for vi in v])

    # (5) load / interpolate of smooth analytic forms (physical frame)
    def scalar(x):
        y = seq.map(x)
        return jnp.array([jnp.sin(y[0]) * jnp.cos(2 * y[1]) + 0.3 * y[2]])

    def vector(x):
        y = seq.map(x)
        return jnp.array([jnp.sin(y[1]) + 0.2 * y[2], jnp.cos(y[0]) * y[2],
                          jnp.sin(y[0] + y[1])])

    for k, dbc in KS:
        tag = f"k{k}_{'dbc' if dbc else 'free'}"
        f = scalar if k in (0, 3) else vector
        d[f"d_load_{tag}"] = seq.load(f, k, dirichlet=dbc)
        d[f"d_interp_{tag}"] = seq.interpolate(f, k, dirichlet=dbc)
        print(f"[proj] {tag} done", flush=True)

    # cross product of the two interpolated 2-forms, as a 2-form load
    w = d["d_interp_k2_dbc"]
    u = d["d_interp_k1_free"]
    d["d_cross_221"] = seq.cross_product_load(w, u, 2, 2, 1, True, True, False)
    d["d_cross_222"] = seq.cross_product_load(w, w, 2, 2, 2, True, True, True)
    d["d_magsq"] = seq.magnitude_squared_load(w, True)

    np.savez(out, **{key: np.asarray(jax.device_get(val)) for key, val in d.items()})
    print(f"[dump] {out}: {len(d)} arrays", flush=True)


def _permute_base(a, nq):
    """(theta, r, zeta)-major flat -> (r, theta, zeta)-major flat."""
    nx, ny, nz = (int(v) for v in nq)
    return a.reshape(ny, nx, nz, *a.shape[1:]).transpose(1, 0, 2, *range(3, a.ndim + 2)).reshape(a.shape)


def _rel(a, b):
    den = np.max(np.abs(a))
    return float(np.max(np.abs(a - b)) / den) if den > 0 else float(np.max(np.abs(a - b)))


def compare(base, new):
    A, B = np.load(base), np.load(new)
    nq = A["s_nq"]
    assert np.array_equal(nq, B["s_nq"]), (nq, B["s_nq"])
    worst = 0.0
    for key in sorted(A.files):
        a, b = A[key], B[key]
        if key.startswith("q_"):
            a = _permute_base(a, nq)
        if key == "s_nq":
            continue
        if key.startswith("n_"):
            # up to sign, row by row
            r = max(min(_rel(ai, bi), _rel(ai, -bi)) for ai, bi in zip(a, b))
        else:
            r = _rel(a, b)
        worst = max(worst, r)
        print(f"{key:24s} shape {str(a.shape):16s} max rel diff {r:.3e}")
    print(f"WORST {worst:.3e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    dp = sub.add_parser("dump")
    dp.add_argument("out")
    dp.add_argument("--geometry", default=W7X)
    dp.add_argument("--ns", default="12,24,12")
    dp.add_argument("--p", type=int, default=3)
    cp = sub.add_parser("compare")
    cp.add_argument("base")
    cp.add_argument("new")
    cli = ap.parse_args()
    if cli.cmd == "dump":
        dump(cli.out, cli.geometry, tuple(int(v) for v in cli.ns.split(",")), cli.p)
    else:
        compare(cli.base, cli.new)
        sys.exit(0)
