"""Modal-radial k=0 Laplacian bulk atom, with the per-k pencil reduction.

Research code (see docs/research/mass_preconditioner_pivot.md section 7). The
production atom is ``_assemble_k0_greville_bulk_factors``; this is the candidate
replacement, kept here rather than in the production path until the full-grid
gate lands.

Measured 2026-08-17/18, CG to 1e-10, bulk block, p=3, fd -> per-k:

    toroid        8x16x16  24/22 -> 13/13     12x24x24  36/32 -> 14/13
    rot-ellipse   8x16x16  59/48 -> 45/36     12x24x24  83/71 -> 49/42
    W7-X          8x16x16  61/45 -> 47/34     12x24x24  83/66 -> 50/40

Mesh-independent where production drifts, and the gain grows with resolution.
"""
import jax.numpy as jnp
import numpy as np  # noqa: F401

from mrx.operators import (_assemble_1d_fd_eigendecomp,
                           _assemble_unweighted_1d_mass,
                           _assemble_weighted_1d_stiffness,
                           _dense_incidence_1d,
                           _reshape_quadrature_matrix_field,
                           _reshape_quadrature_scalar_field,
                           _restrict_radial_window)
from mrx.preconditioners import _bulk_tensor_shape

__all__ = ["modal_perk_bulk_data", "modal_perk_apply", "fd_harmonic_bulk_data"]


def _harmonic(vals, w, axes):
    """Quadrature-weighted harmonic mean of ``vals`` over ``axes``.

    Harmonic rather than arithmetic: section 7.2 measured it as never worse and
    finite for the ``1/r``-type ``g^tt J`` weight whose arithmetic mean diverges
    at the axis -- which is what forced the ``wx_cut`` polar-element skip.
    """
    x = 1.0 / jnp.clip(vals, 1e-30)
    denom = 1.0
    for ax in sorted(axes, reverse=True):
        wa = w[ax]
        x = jnp.tensordot(x, wa, axes=([ax], [0]))
        denom = denom * jnp.sum(wa)
    return denom / jnp.clip(x, 1e-30)


def modal_perk_bulk_data(seq, *, dirichlet):
    """Modal-radial with the per-k pencil reduction.

    Measured (2026-08-17, toroid / rot-ellipse / W7-X): the radial profiles of
    ``g^rr J`` and ``g^zz J`` are proportional -- log-log slopes both +1, ratio
    spread <= 1.07 -- and at the ASSEMBLED MATRIX level
    ``||M_r[c] - kappa M_r[a]|| / ||M_r[c]|| = 0.014``. So there are really only
    two distinct radial operators, and the mode dependence separates::

        A_jk = (K_r[a] + kappa nu_k M_r[a])  +  mu_j M_r[b]  =  P_k + mu_j Q

    ``P_k`` depends only on k and ``Q`` is fixed, so ONE pencil (Q, P_k) per k
    serves every j::

        W_k^T Q W_k = I,  W_k^T P_k W_k = diag(d_k)
        =>  A_jk^-1 = W_k diag( 1 / (d_k + mu_j) ) W_k^T

    Storage falls from ``n_t n_z n_r^2`` to ``n_z n_r^2`` -- independent of
    ``n_t``, 252 MB -> 2.0 MB at 64x128x64 -- and the per-mode radial solves
    become a diagonal scale.
    """
    bulk_shape = _bulk_tensor_shape(seq, dirichlet)
    nr_bulk = int(bulk_shape[0])
    types = seq.basis_0.types

    minv = jnp.transpose(_reshape_quadrature_matrix_field(
        seq, seq.geometry.metric_inv_jkl), (1, 0, 2, 3, 4))
    jacq = jnp.transpose(_reshape_quadrature_scalar_field(
        seq, seq.geometry.jacobian_j), (1, 0, 2))
    w00, w11, w22 = (minv[..., a, a] * jacq for a in range(3))
    W = (seq.quad.w_x, seq.quad.w_y, seq.quad.w_z)

    a_r = _harmonic(w00, W, (1, 2))
    b_r = _harmonic(w11, W, (1, 2))
    c_r = _harmonic(w22, W, (1, 2))
    beta_t = _harmonic(w11, W, (0, 2)) / jnp.clip(_harmonic(b_r, W, (0,)), 1e-30)
    gamma_z = _harmonic(w22, W, (0, 1)) / jnp.clip(_harmonic(c_r, W, (0,)), 1e-30)

    g_r = _dense_incidence_1d(seq.basis_0.nr, types[0])
    g_t = _dense_incidence_1d(seq.basis_0.nt, types[1])
    g_z = _dense_incidence_1d(seq.basis_0.nz, types[2])

    K_r = _restrict_radial_window(_assemble_weighted_1d_stiffness(
        seq.basis_r_jk, seq.d_basis_r_jk, seq.quad.w_x * a_r, g_r), 2, nr_bulk)
    M_a = _restrict_radial_window(_assemble_unweighted_1d_mass(
        seq.basis_r_jk, seq.quad.w_x * a_r), 2, nr_bulk)
    M_b = _restrict_radial_window(_assemble_unweighted_1d_mass(
        seq.basis_r_jk, seq.quad.w_x * b_r), 2, nr_bulk)
    M_c = _restrict_radial_window(_assemble_unweighted_1d_mass(
        seq.basis_r_jk, seq.quad.w_x * c_r), 2, nr_bulk)
    M_t = _assemble_unweighted_1d_mass(seq.basis_t_jk, seq.quad.w_y)
    M_z = _assemble_unweighted_1d_mass(seq.basis_z_jk, seq.quad.w_z)
    K_t = _assemble_weighted_1d_stiffness(
        seq.basis_t_jk, seq.d_basis_t_jk, seq.quad.w_y * beta_t, g_t)
    K_z = _assemble_weighted_1d_stiffness(
        seq.basis_z_jk, seq.d_basis_z_jk, seq.quad.w_z * gamma_z, g_z)

    V_t, mu = _assemble_1d_fd_eigendecomp(M_t, K_t)
    V_z, nu = _assemble_1d_fd_eigendecomp(M_z, K_z)

    # kappa from the assembled matrices, and the residual it leaves.
    kappa = float(jnp.sum(M_c * M_a) / jnp.sum(M_a * M_a))
    resid = float(jnp.linalg.norm(M_c - kappa * M_a) / jnp.linalg.norm(M_c))

    Ws, ds = [], []
    for k in range(int(nu.shape[0])):
        P_k = K_r + (kappa * nu[k]) * M_a
        W_k, d_k = _assemble_1d_fd_eigendecomp(M_b, P_k)   # W^T M_b W = I
        Ws.append(W_k)
        ds.append(d_k)
    return {"bulk_shape": bulk_shape, "V_t": V_t, "V_z": V_z, "mu": mu,
            "W": jnp.stack(Ws), "d": jnp.stack(ds), "kappa_resid": resid}


def modal_perk_apply(data, rhs_b):
    x = rhs_b.reshape(data["bulk_shape"])
    x = jnp.einsum('tj,rtz->rjz', data["V_t"], x)
    x = jnp.einsum('zk,rjz->rjk', data["V_z"], x)
    x = jnp.einsum('krs,rjk->sjk', data["W"], x)
    den = data["d"].T[:, None, :] + data["mu"][None, :, None]   # (s, j, k)
    den_max = jnp.max(jnp.abs(den))                 # RELATIVE, as _fd_apply_3d
    null_mask = jnp.abs(den) < 1e-10 * den_max
    x = jnp.where(null_mask, 0.0, x / jnp.where(null_mask, 1.0, den))
    x = jnp.einsum('krs,sjk->rjk', data["W"], x)
    x = jnp.einsum('zk,rjk->rjz', data["V_z"], x)
    x = jnp.einsum('tj,rjz->rtz', data["V_t"], x)
    return x.reshape(-1)


def fd_harmonic_bulk_data(seq, *, dirichlet):
    """Pure fast diagonalization with HARMONIC own-axis profiles, no wx_cut.

    Same structural form as the production ``fd`` atom -- all three masses
    unweighted, each 1D stiffness carrying a profile along its OWN axis, and the
    additive denominator ``lam_r + lam_t + lam_z``, so no ``n_r x n_r`` inverses
    and ``O(n_r^2 + n_t^2 + n_z^2)`` storage::

        K_r[a(r)] (x) M_t (x) M_z  +  M_r (x) K_t[b(t)] (x) M_z
                                   +  M_r (x) M_t (x) K_z[c(z)]

    It differs from production in the averaging rule only. Production takes a
    quadrature-weighted ARITHMETIC mean, which diverges for the ``1/r``-type
    ``g^tt J`` weight and therefore has to skip the polar element (``wx_cut``).
    The harmonic mean is finite there, so no cut is needed -- section 7.2
    measured this as never worse on the bulk and better on the full grid
    (73 vs 82), and it retires the documented dependence on "core DOFs are
    handled exactly by the Schur envelope".
    """
    bulk_shape = _bulk_tensor_shape(seq, dirichlet)
    nr_bulk = int(bulk_shape[0])
    types = seq.basis_0.types

    minv = jnp.transpose(_reshape_quadrature_matrix_field(
        seq, seq.geometry.metric_inv_jkl), (1, 0, 2, 3, 4))
    jacq = jnp.transpose(_reshape_quadrature_scalar_field(
        seq, seq.geometry.jacobian_j), (1, 0, 2))
    w00, w11, w22 = (minv[..., a, a] * jacq for a in range(3))
    W = (seq.quad.w_x, seq.quad.w_y, seq.quad.w_z)

    pr = _harmonic(w00, W, (1, 2))        # profile along r
    pt = _harmonic(w11, W, (0, 2))        # profile along theta
    pz = _harmonic(w22, W, (0, 1))        # profile along zeta

    g_r = _dense_incidence_1d(seq.basis_0.nr, types[0])
    g_t = _dense_incidence_1d(seq.basis_0.nt, types[1])
    g_z = _dense_incidence_1d(seq.basis_0.nz, types[2])

    M0_r = _restrict_radial_window(_assemble_unweighted_1d_mass(
        seq.basis_r_jk, seq.quad.w_x), 2, nr_bulk)
    M0_t = _assemble_unweighted_1d_mass(seq.basis_t_jk, seq.quad.w_y)
    M0_z = _assemble_unweighted_1d_mass(seq.basis_z_jk, seq.quad.w_z)
    K0_r = _restrict_radial_window(_assemble_weighted_1d_stiffness(
        seq.basis_r_jk, seq.d_basis_r_jk, seq.quad.w_x * pr, g_r), 2, nr_bulk)
    K0_t = _assemble_weighted_1d_stiffness(
        seq.basis_t_jk, seq.d_basis_t_jk, seq.quad.w_y * pt, g_t)
    K0_z = _assemble_weighted_1d_stiffness(
        seq.basis_z_jk, seq.d_basis_z_jk, seq.quad.w_z * pz, g_z)

    V_r, lam_r = _assemble_1d_fd_eigendecomp(M0_r, K0_r)
    V_t, lam_t = _assemble_1d_fd_eigendecomp(M0_t, K0_t)
    V_z, lam_z = _assemble_1d_fd_eigendecomp(M0_z, K0_z)
    return {
        "bulk_shape": bulk_shape,
        "bulk_V_r": V_r, "bulk_V_t": V_t, "bulk_V_z": V_z,
        "bulk_lam_r": lam_r, "bulk_lam_t": lam_t, "bulk_lam_z": lam_z,
        "bulk_alpha": jnp.ones((3,), dtype=jnp.float64),
        "bulk_greville_inv_sqrt_D": jnp.ones(bulk_shape, dtype=jnp.float64),
    }
