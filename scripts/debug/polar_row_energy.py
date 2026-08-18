"""Can the POLAR rows of diag(E S E^T) be had without an operator apply?

(E S E^T)_ii is the energy of the EXTRACTED basis function

    psi_i = sum_a E_ia phi_a ,     (E S E^T)_ii = int <grad psi_i, W grad psi_i>

so it needs no raw off-diagonal entries -- only psi_i and its derivatives at the
quadrature points, which is the same contraction the bulk rows already use.

For a k=0 polar row the columns run over a ring: ring_depth radial indices x all
theta, at a single zeta index m, so

    psi(r,t,z) = [ sum_ij xi_ij R_i(r) T_j(t) ] Z_m(z)

The (r,t) factor is a genuine 2D combination (not a tensor product), but there
are only n_polar distinct shapes and they are independent of m, so the cost is
O(n_polar * n_q^{rt} * n_q^z) with ZERO applies.

This checks the construction against the probe, which is an exact oracle.
"""
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.geometry import grad_1d  # noqa: E402
from mrx.mappings import toroid_map  # noqa: E402
from mrx.operators import (_reshape_quadrature_matrix_field,  # noqa: E402
                           _reshape_quadrature_scalar_field,
                           apply_stiffness, assemble_incidence_operators)

NS, P = (6, 8, 6), 3
seq = DeRhamSequence(NS, (P,) * 3, 2 * P, ("clamped", "periodic", "periodic"),
                     polar=True, tol=1e-12, maxiter=1000, betti_numbers=(1, 1, 0, 0))
seq.evaluate_1d(); seq.set_map(toroid_map(epsilon=1 / 3, R0=1.0))
ops = assemble_incidence_operators(seq); seq.set_operators(ops)

e = seq.e0
n_ext = int(e.shape[0])
rows = np.asarray(e.rows); cols = np.asarray(e.cols); vals = np.asarray(e.vals)
counts = np.bincount(rows, minlength=n_ext)
polar = np.flatnonzero(counts > 1)
print(f"extracted rows={n_ext}  polar rows={polar.size}\n")

# 1D tables on the quadrature grid, and the derivative of the PRIMAL basis.
types = seq.basis_0.types
Rt = np.asarray(seq.basis_r_jk); Tt = np.asarray(seq.basis_t_jk); Zt = np.asarray(seq.basis_z_jk)
Rd = np.asarray(grad_1d(seq.d_basis_r_jk, types[0]))
Td = np.asarray(grad_1d(seq.d_basis_t_jk, types[1]))
Zd = np.asarray(grad_1d(seq.d_basis_z_jk, types[2]))

nx, ny, nz = seq.quad.nx, seq.quad.ny, seq.quad.nz
minv = np.asarray(jnp.transpose(_reshape_quadrature_matrix_field(
    seq, seq.geometry.metric_inv_jkl), (1, 0, 2, 3, 4)))
jq = np.asarray(jnp.transpose(_reshape_quadrature_scalar_field(
    seq, seq.geometry.jacobian_j), (1, 0, 2)))
W = np.einsum('qrsij,qrs->qrsij', minv, jq)
wq = (np.asarray(seq.quad.w_x)[:, None, None]
      * np.asarray(seq.quad.w_y)[None, :, None]
      * np.asarray(seq.quad.w_z)[None, None, :])

nr, nt, nzb = seq.basis_0.nr, seq.basis_0.nt, seq.basis_0.nz
print(f"{'row':>5} {'energy (no apply)':>19} {'probe':>14} {'rel err':>11}")
bad = 0.0
for i in polar[:8]:
    sel = rows == i
    c, v = cols[sel], vals[sel]
    ir, it, iz = c // (nt * nzb), (c // nzb) % nt, c % nzb
    assert len(set(iz.tolist())) == 1, "polar row should sit at one zeta index"
    m = int(iz[0])
    # 2D (r,theta) shapes and their partials; zeta is a plain 1D factor.
    A   = np.einsum('a,aq,ar->qr', v, Rt[ir], Tt[it])
    A_r = np.einsum('a,aq,ar->qr', v, Rd[ir], Tt[it])
    A_t = np.einsum('a,aq,ar->qr', v, Rt[ir], Td[it])
    g = [A_r[:, :, None] * Zt[m][None, None, :],
         A_t[:, :, None] * Zt[m][None, None, :],
         A[:, :, None] * Zd[m][None, None, :]]
    energy = sum(float(np.sum(wq * W[..., p, q] * g[p] * g[q]))
                 for p in range(3) for q in range(3))
    ei = jnp.zeros(n_ext).at[int(i)].set(1.0)
    probe = float(ei @ apply_stiffness(seq, ops, ei, 0, dirichlet=False))
    rel = abs(energy - probe) / abs(probe)
    bad = max(bad, rel)
    print(f"{i:>5} {energy:>19.10e} {probe:>14.6e} {rel:>11.2e}")
print(f"\nmax rel err over polar rows: {bad:.3e}  "
      f"{'PASS -- core needs no probe' if bad < 1e-10 else 'FAIL'}")
