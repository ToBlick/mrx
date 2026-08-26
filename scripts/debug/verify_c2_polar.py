"""Verification of the C^2 polar extraction (get_xi2 / polar_order=2).

Checks, in order:
  1. AD radial end-derivatives N_1'(0), N_2''(0) against central finite
     differences (the rho constant in the ring-2 condition).
  2. Partition of unity of xi2 on every ring, and constant reproduction
     through the assembled E0 (E0^T applied to the polar-coefficient
     "all ones" must give the tensor constant vector).
  3. Subspace property: the rings-0/1 block of every C^2 basis function
     lies in the span of the C^1 polar functions' rings-0/1 block
     (V^0_{C^2} is a subspace of V^0_{C^1}).
  4. Taylor-remainder scaling at the pole: evaluate a random C^2 element
     on shrinking circles s in {eps, eps/2, eps/4} around the axis of the
     logical disk, fit a full quadratic in (x, y) = (s cos, s sin), and
     check the fit residual decays ~ eps^3 (third order). A generic C^1
     element (random ring-2 coefficients) must stall at ~ eps^2.

Run: python scripts/debug/verify_c2_polar.py
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp


from mrx.spline_bases import SplineBasis  # noqa: E402
from mrx.extraction_operators import get_xi, get_xi2  # noqa: E402

P = 3
NR, NT = 8, 16
FAILED = []


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILED.append(name)


# --- bases (r_scale=0.5 grading, the production default) -------------------
bp = jnp.linspace(0, 1, NR - P + 1) ** 0.5
Tr = jnp.concatenate([jnp.zeros(P), bp, jnp.ones(P)])
br = SplineBasis(NR, P, "clamped", Tr)
bt = SplineBasis(NT, P, "periodic", None)

# --- 1. AD end-derivatives vs finite differences ----------------------------
n1p_ad = float(jax.grad(lambda x: br.evaluate(x, 1))(0.0))
n2pp_ad = float(jax.grad(jax.grad(lambda x: br.evaluate(x, 2)))(0.0))
h = 1e-6
n1p_fd = float((br.evaluate(h, 1) - br.evaluate(0.0, 1)) / h)
n2pp_fd = float((br.evaluate(2 * h, 2) - 2 * br.evaluate(h, 2)
                 + br.evaluate(0.0, 2)) / h ** 2)
check("N1'(0) AD vs FD", abs(n1p_ad - n1p_fd) < 1e-4 * abs(n1p_ad),
      f"AD {n1p_ad:.6g} FD {n1p_fd:.6g}")
check("N2''(0) AD vs FD", abs(n2pp_ad - n2pp_fd) < 1e-3 * abs(n2pp_ad),
      f"AD {n2pp_ad:.6g} FD {n2pp_fd:.6g}")
check("N2''(0) > 0", n2pp_ad > 0, f"{n2pp_ad:.6g}")

# --- 2. PoU + constant reproduction -----------------------------------------
xi1 = get_xi(NT)
xi2 = get_xi2(NT, br)
check("xi2 shape", xi2.shape == (6, 3, NT), f"{xi2.shape}")
pou = float(jnp.abs(xi2.sum(axis=0) - 1.0).max())
check("xi2 partition of unity (all rings)", pou < 1e-14, f"max dev {pou:.2e}")

# constant reproduction through the assembled E0 (small 3D sequence)
from mrx.differential_forms import DifferentialForm  # noqa: E402
from mrx.extraction_operators import PolarExtractionOperator  # noqa: E402
NZ = 4
Lam0 = DifferentialForm(0, (NR, NT, NZ), (P, P, P),
                        ("clamped", "periodic", "periodic"), [Tr, None, None])
e0_c2 = PolarExtractionOperator(Lam0, xi2, False).build_extraction()
e0_c1 = PolarExtractionOperator(Lam0, xi1, False).build_extraction()
n_c2, n_tensor = e0_c2.forward_shape
ones_polar = jnp.ones((n_c2,))
cerr = float(jnp.abs(e0_c2.T @ ones_polar - 1.0).max())
check("E0_C2^T 1 = tensor constant", cerr < 1e-14, f"max dev {cerr:.2e}")
check("dims: n_C2 = n_C1 - nt*nz + 3*nz",
      n_c2 == e0_c1.forward_shape[0] - NT * NZ + 3 * NZ,
      f"n_C2={n_c2} n_C1={e0_c1.forward_shape[0]}")

# --- 3. subspace of C^1 ------------------------------------------------------
# rings-0/1 block of each C^2 polar function must be in span of the C^1 ones
A = np.asarray(xi1[:, :2, :]).reshape(3, -1).T      # (2*NT, 3)
maxres = 0.0
for a in range(6):
    b = np.asarray(xi2[a, :2, :]).reshape(-1)        # (2*NT,)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    maxres = max(maxres, float(np.abs(A @ coef - b).max()))
check("V_C2 subset V_C1 (rings 0-1 in C1 span)", maxres < 1e-13,
      f"max lstsq residual {maxres:.2e}")

# --- 4. Taylor-remainder scaling at the pole --------------------------------
rng = np.random.default_rng(0)


def eval_2d(coeffs_rt, s_vals, t_vals):
    """Evaluate sum_ij c_ij N_i(s) N_j(t) on a (s, t) grid."""
    Cs = jnp.stack([jnp.stack([br.evaluate(float(s), i) for i in range(NR)])
                    for s in s_vals])                       # (ns, NR)
    Ct = jnp.stack([jnp.stack([bt.evaluate(float(t), j) for j in range(NT)])
                    for t in t_vals])                       # (nt_pts, NT)
    return Cs @ coeffs_rt @ Ct.T                            # (ns, nt_pts)


def quad_fit_residual(coeffs_rt, eps):
    """Max residual of a full-quadratic fit in (x, y) on circles s<=eps."""
    s_vals = np.array([eps, 0.75 * eps, 0.5 * eps, 0.25 * eps])
    t_vals = np.linspace(0.0, 1.0, 24, endpoint=False)
    f = np.asarray(eval_2d(coeffs_rt, s_vals, t_vals)).reshape(-1)
    ss, tt = np.meshgrid(s_vals, t_vals, indexing="ij")
    x = (ss * np.cos(2 * np.pi * tt)).reshape(-1)
    y = (ss * np.sin(2 * np.pi * tt)).reshape(-1)
    V = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)
    coef, *_ = np.linalg.lstsq(V, f, rcond=None)
    return float(np.abs(V @ coef - f).max())


def taylor_orders(coeffs_rt):
    epss = [0.32, 0.16, 0.08, 0.04]
    res = [quad_fit_residual(coeffs_rt, e) for e in epss]
    orders = [np.log2(res[i] / res[i + 1]) for i in range(len(res) - 1)]
    return res, orders


# random C^2 element: random polar + random outer ring coefficients
c2_rt = np.zeros((NR, NT))
fbar = rng.standard_normal(6)
for i in range(3):
    c2_rt[i] = np.asarray(jnp.einsum("l,lj->j", jnp.asarray(fbar), xi2[:, i, :]))
c2_rt[3:] = rng.standard_normal((NR - 3, NT))
res2, ord2 = taylor_orders(jnp.asarray(c2_rt))

# generic C^1 element: C^1 rings 0-1, RANDOM ring 2
c1_rt = np.zeros((NR, NT))
gbar = rng.standard_normal(3)
for i in range(2):
    c1_rt[i] = np.asarray(jnp.einsum("l,lj->j", jnp.asarray(gbar), xi1[:, i, :]))
c1_rt[2:] = rng.standard_normal((NR - 2, NT))
res1, ord1 = taylor_orders(jnp.asarray(c1_rt))

print(f"  C2 element: residuals {['%.2e' % r for r in res2]} orders "
      f"{['%.2f' % o for o in ord2]}")
print(f"  C1 element: residuals {['%.2e' % r for r in res1]} orders "
      f"{['%.2f' % o for o in ord1]}")
check("C2 Taylor remainder ~ eps^3", min(ord2) > 2.5,
      f"min order {min(ord2):.2f} (want > 2.5)")
check("C1 (random ring 2) stalls at ~ eps^2", max(ord1) < 2.5,
      f"max order {max(ord1):.2f} (want < 2.5)")

print()
if FAILED:
    print(f"FAILED: {FAILED}")
    sys.exit(1)
print("ALL CHECKS PASS")
