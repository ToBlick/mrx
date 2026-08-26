"""Dense oracles for the tests: densify a matrix-free apply, or build a mass
matrix directly from the 1-D basis tables by quadrature.

The production code never materialises an operator; these helpers exist so
tests can check symmetry, spectra and equivalences on tiny sequences.
"""

import jax
import jax.numpy as jnp
import numpy as np

import mrx


def dense_from_apply(apply, n, batch_size=16):
    """Return the ``(m, n)`` matrix of a linear ``apply`` by probing unit vectors.

    The columns are produced with ``jax.lax.map`` over one-hot vectors in
    batches of ``batch_size`` (a full ``vmap`` over ``jnp.eye`` compiles to a
    kernel that has crashed ptxas on the larger probes).
    """
    apply(jnp.zeros(n, dtype=mrx.DTYPE))  # build any host-side plan eagerly

    def column(j):
        return apply(jnp.zeros(n, dtype=mrx.DTYPE).at[j].set(1.0))

    return np.asarray(jax.lax.map(column, jnp.arange(n), batch_size=batch_size)).T


def basis_at_quadrature(seq, k):
    """Return the raw k-form basis sampled on the quadrature grid.

    Result: list over components ``c`` of ``(n_c, N_q)`` arrays, the value of
    component ``c`` of each raw basis function at every quadrature point of
    ``seq``, in the ``seq.quad.x`` ordering (meshgrid 'xy': theta, r, zeta).
    """
    comp_info, _ = seq._form_comp_info(k)
    out = []
    for _, R, T, Z in comp_info:
        tab = jnp.einsum("ai,bj,ck->abcjik", R, T, Z)
        out.append(np.asarray(tab.reshape(-1, seq.quad.ny * seq.quad.nx * seq.quad.nz)))
    return out


def dense_mixed_mass(seq, k_row, k_col, weight):
    """Quadrature oracle for ``int Lambda^{k_row} . W . Lambda^{k_col}`` in raw DOF space.

    ``weight`` is ``(N_q,)`` for scalar pairs or ``(N_q, 3, 3)`` for vector
    pairs (without the quadrature weights, which are applied here).
    """
    rows = basis_at_quadrature(seq, k_row)
    cols = basis_at_quadrature(seq, k_col)
    w = np.asarray(seq.quad.w)
    weight = np.asarray(weight)
    blocks = [[None] * len(cols) for _ in rows]
    for cr, Lr in enumerate(rows):
        for cc, Lc in enumerate(cols):
            wq = weight if weight.ndim == 1 else weight[:, cr, cc]
            blocks[cr][cc] = (Lr * (w * wq)[None, :]) @ Lc.T
    return np.block(blocks)
