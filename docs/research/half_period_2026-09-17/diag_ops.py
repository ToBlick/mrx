"""Which operator differs between the half-period li383 (8,12,12) sequence and
its full-period twin, on vectors of definite parity."""
import os
os.environ.setdefault("MRX_DTYPE", "float64")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.geometry import build_sequence  # noqa: E402
from mrx.initial_conditions import initial_field  # noqa: E402
from mrx.nullspace import compute_nullspaces  # noqa: E402

NS, P = (8, 12, 12), 2
half, _ = build_sequence("data/wout_li383_low_res_reference.nc", NS, P, symmetry="stellarator")
full = DeRhamSequence(NS, (P, P, P), P + 1, ("clamped", "periodic", "periodic"), polar=True,
                      betti_numbers=(1, 1, 0, 0), half_period=False)
full.set_map(half.map)
full.build_preconditioners()
print("built", flush=True)


def rel(a, b):
    return float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))


rng = np.random.default_rng(0)
vecs = {}
for k in range(4):
    for d in (True, False):
        for parity in (1, -1):
            x = jnp.asarray(rng.standard_normal(half.n(k, d)))
            vecs[(k, d, parity)] = half.project_parity(x, k, parity, d)
            xr = half.E(k, d).T @ vecs[(k, d, parity)]
            from mrx.symmetry import reflect
            mix = float(jnp.vdot(xr, reflect(xr, half.reflection_plan[k])) / jnp.vdot(xr, xr))
            print(f"vector k={k} d={d} parity={parity:+d}: x.Rx/|x|^2 = {mix:+.6f}", flush=True)

for (k, d, parity), x in vecs.items():
    print(f"M_{k} d={d} parity={parity:+d}: {rel(half.apply_mass_matrix(x, k, d), full.apply_mass_matrix(x, k, d)):.2e}", flush=True)
for (kin, kout) in ((2, 1), (1, 2), (0, 3), (3, 0)):
    for parity in (1, -1):
        x = vecs[(kin, True, parity)]
        print(f"P_{kin}{kout} parity={parity:+d}: {rel(half.apply_projection_matrix(x, kin, kout, True, True), full.apply_projection_matrix(x, kin, kout, True, True)):.2e}", flush=True)
for k in (0, 1, 2):
    for d in (True, False):
        for parity in (1, -1):
            x = vecs[(k, d, parity)]
            print(f"D_{k} d={d} parity={parity:+d}: {rel(half.apply_derivative_matrix(x, k, d, d), full.apply_derivative_matrix(x, k, d, d)):.2e}  "
                  f"D_{k}^T: {rel(half.apply_derivative_matrix(vecs[(k + 1, d, parity)], k, d, d, transpose=True), full.apply_derivative_matrix(vecs[(k + 1, d, parity)], k, d, d, transpose=True)):.2e}  "
                  f"S_{k}: {rel(half.apply_stiffness(x, k, d), full.apply_stiffness(x, k, d)):.2e}", flush=True)
for k in range(4):
    for d in (True, False):
        for parity in (1, -1):
            x = vecs[(k, d, parity)]
            print(f"L_{k} d={d} parity={parity:+d}: {rel(half.apply_laplacian(x, k, d), full.apply_laplacian(x, k, d)):.2e}  "
                  f"precond: {rel(half.apply_laplacian_preconditioner(x, k, d), full.apply_laplacian_preconditioner(x, k, d)):.2e}  "
                  f"mass precond: {rel(half.apply_mass_matrix_preconditioner(x, k, d), full.apply_mass_matrix_preconditioner(x, k, d)):.2e}", flush=True)
from mrx.symmetry import reflect as _reflect
for k in range(4):
    for d in (True, False):
        # a DUAL right-hand side of definite parity: what a load or an apply produces
        x = full.apply_mass_matrix(vecs[(k, d, -1)], k, d)
        for name, z in (("mass atom", half.apply_mass_matrix_preconditioner(x, k, d)),
                        ("laplacian atom", half.apply_laplacian_preconditioner(x, k, d))):
            zr = half.E(k, d).T @ z
            print(f"{name} k={k} d={d}: output parity {float(jnp.vdot(zr, _reflect(zr, half.reflection_plan[k])) / jnp.vdot(zr, zr)):+.6f}", flush=True)
        yh, ih = half.apply_inverse_mass_matrix(x, k, d, return_info=True)
        yf, jf = full.apply_inverse_mass_matrix(x, k, d, return_info=True)
        print(f"M_{k}^-1 d={d}: {rel(yh, yf):.2e}  its {int(ih)} vs {int(jf)}", flush=True)
for k, d in ((1, True), (1, False), (2, True), (0, False), (0, True)):
    x = half.apply_laplacian(vecs[(k, d, -1)], k, d)          # in the range of L_k
    yh, ih = half.apply_inverse_laplacian(x, k, d, return_info=True)
    yf, jf = full.apply_inverse_laplacian(x, k, d, return_info=True)
    print(f"L_{k}^-1 d={d}: {rel(yh, yf):.2e}  its {ih} vs {jf}", flush=True)

compute_nullspaces(half)
compute_nullspaces(full)
for k, d in ((0, False), (1, False), (2, True), (3, True)):
    hh, hf = half.nullspace(k, d)[0], full.nullspace(k, d)[0]
    print(f"harmonic k={k} d={d}: {rel(hh, hf):.2e} (sign-blind {min(rel(hh, hf), rel(-hh, hf)):.2e})", flush=True)
B, info = initial_field(half)
print(f"initial field parity discarded {info['parity_discarded']:.2e}", flush=True)
