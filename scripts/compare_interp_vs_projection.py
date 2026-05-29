"""Compare Greville interpolation vs L2 projection on random Besov scalar functions.

Builds a DeRhamSequence with identity map, then for each of N_FUNCS random Besov
k=0 functions computes:
  - Greville interpolation DOFs via seq.p0.zeroform_interpolation
  - L2-projection DOFs via seq.p0(f) → seq.apply_inverse_mass_matrix (sparse CG)

L2 errors are evaluated on an independent fine Gauss quadrature grid.
Prints mean, max, min, std of L2 errors for both methods.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.projectors import Projector
from mrx.utils import build_random_besov_function

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N = 10        # DOFs per direction
P = 2         # polynomial degree
Q = P + 2     # quadrature order per element
N_FUNCS = 8   # number of random Besov functions
SEED = 42     # base seed; individual keys split from this

EVAL_QUAD_N = 20  # independent Gauss points per direction for error evaluation

# ---------------------------------------------------------------------------
# Build sequence with identity map
# ---------------------------------------------------------------------------
print(f"Building DeRhamSequence: n={N}, p={P}, q={Q}")
seq = DeRhamSequence(
    (N, N, N), (P, P, P), Q,
    ('clamped', 'clamped', 'clamped'),
    polar=False, betti_numbers=(1, 1, 0, 0),
)
seq.set_map(lambda x: x)
seq.evaluate_1d()
seq.assemble_mass_matrix(0)
print(f"  k=0 DOFs (free): {seq.n0}")

# ---------------------------------------------------------------------------
# Independent fine quadrature for error evaluation
# ---------------------------------------------------------------------------
xi_1d_np, w_1d_np = np.polynomial.legendre.leggauss(EVAL_QUAD_N)
xi_1d = jnp.asarray((xi_1d_np + 1.0) / 2.0)
w_1d  = jnp.asarray(w_1d_np / 2.0)
r, t, z = jnp.meshgrid(xi_1d, xi_1d, xi_1d, indexing='ij')
wr, wt, wz = jnp.meshgrid(w_1d, w_1d, w_1d, indexing='ij')
quad_pts = jnp.stack([r.ravel(), t.ravel(), z.ravel()], axis=-1)
quad_w   = (wr * wt * wz).ravel()

# ---------------------------------------------------------------------------
# Loop over random functions
# ---------------------------------------------------------------------------
keys = jax.random.split(jax.random.PRNGKey(SEED), N_FUNCS)

errors_interp = []
errors_proj   = []

for i, key in enumerate(keys):
    f = build_random_besov_function(0, key=key)

    # Exact values at evaluation points (shape (n_pts,))
    f_exact = jax.lax.map(lambda x: f(x)[0], quad_pts)

    # --- Greville interpolation ---
    dofs_interp = seq.p0.zeroform_interpolation(f)
    disc_interp = DiscreteFunction(dofs_interp, seq.basis_0, seq.e0)
    f_h_interp  = jax.lax.map(lambda x: disc_interp(x)[0], quad_pts)
    err_interp  = float(jnp.sqrt(jnp.sum(quad_w * (f_h_interp - f_exact) ** 2)))

    # --- L2 projection ---
    # seq.p0(f) returns e0 @ (L2 RHS integral), i.e. the extracted dual vector.
    # apply_inverse_mass_matrix solves M0 u = rhs via sparse CG.
    rhs_l2   = seq.p0(f)
    dofs_l2  = seq.apply_inverse_mass_matrix(rhs_l2, 0, dirichlet=False)
    disc_l2  = DiscreteFunction(dofs_l2, seq.basis_0, seq.e0)
    f_h_l2   = jax.lax.map(lambda x: disc_l2(x)[0], quad_pts)
    err_l2   = float(jnp.sqrt(jnp.sum(quad_w * (f_h_l2 - f_exact) ** 2)))

    errors_interp.append(err_interp)
    errors_proj.append(err_l2)
    print(f"  [{i+1}/{N_FUNCS}]  interp={err_interp:.4e}  l2proj={err_l2:.4e}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
ei = np.array(errors_interp)
ep = np.array(errors_proj)

print()
print(f"{'Method':<14}  {'mean':>10}  {'max':>10}  {'min':>10}  {'std':>10}")
print("-" * 58)
print(f"{'Greville':<14}  {ei.mean():>10.4e}  {ei.max():>10.4e}  {ei.min():>10.4e}  {ei.std():>10.4e}")
print(f"{'L2 proj':<14}  {ep.mean():>10.4e}  {ep.max():>10.4e}  {ep.min():>10.4e}  {ep.std():>10.4e}")
