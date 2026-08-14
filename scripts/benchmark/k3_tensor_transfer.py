"""k=3 transfer through the UNEXTRACTED tensor V0-dbc: square, full-rank, no surgery.

P_3 = T B0 T^T with:
- T: V0-tensor(dbc) -> V3(free), Greville-collocated q3 = J * q0:
  diag(J at the V3 Greville grid) . (E_r (x) E_t (x) E_z), E = 1D collocation
  of the (windowed) N bases at the (windowed) D-basis Greville points. The
  dbc radial N-window [1 : nr-1] has EXACTLY the V3 D-count (nr-1 ... after
  dropping first+last: nr-2 = dr-1? both windows chosen size dr) -- square.
- B0: exact Lynch inverse of the rank-1-fitted tensor L0 on that window
  (pencils (M^N, K[p_a]) with fdbund own-axis profiles of g^{aa}J; the
  unextracted axis modes are STIFF (1/r-weighted), not near-null).
"""
import numpy as np
import jax.numpy as jnp
from mrx.operators import (
    _assemble_unweighted_1d_mass, _assemble_weighted_1d_stiffness,
    _dense_incidence_1d, _reshape_quadrature_matrix_field,
    _reshape_quadrature_scalar_field, _symmetrize)
from mrx.geometry import compute_geometry_terms
from mrx.preconditioners import _simultaneous_diagonalize_pair


def build_k3_tensor_transfer(seq, ops):
    nr, nt, nz = int(seq.basis_0.nr), int(seq.basis_0.nt), int(seq.basis_0.nz)
    types = seq.basis_0.types
    # V3 drops its innermost radial D-fn (f(0)=0 baked into the space):
    # radial count from n3 directly. The SQUARE V0 partner enforces zero at
    # BOTH axis and wall (drop first + last N fns) -- Tobias's constraint,
    # realized as the exact dual pairing.
    dr = int(seq.n3) // (nt * nz)
    # V0-dbc tensor window: drop first and last radial N fn -> nr-2 ... use
    # the size that matches V3 radial count dr = nr-1: keep [1 : nr] minus
    # outer dbc drop -> [1 : nr-1] has nr-2; to be SQUARE with dr use
    # [0 : nr-1] (drop only the outer fn; keep the axis fn -- legal, the
    # space is unextracted).
    slN = slice(1, 1 + dr)
    N1 = [seq.basis_0.Λ[a] for a in range(3)]
    D1 = [seq.basis_0.dΛ[a].s for a in range(3)]
    # V3 greville grid (D,D,D windows: radial full dr)
    gr = np.asarray(D1[0].greville_points())[1:1 + dr]
    gt = np.asarray(D1[1].greville_points())
    gz = np.asarray(D1[2].greville_points())
    if types[0] == "clamped":
        gr = np.clip(gr, 1e-7, 1 - 1e-7)
    E = []
    for a, (basis, pts, sl) in enumerate(((N1[0], gr, slN), (N1[1], gt, None), (N1[2], gz, None))):
        C = np.asarray(basis.collocation_matrix(jnp.asarray(pts)))
        if sl is not None:
            C = C[:, sl]
        E.append(jnp.asarray(C))
    rr, tt, zz = np.meshgrid(gr, gt, gz, indexing="ij")
    pts = jnp.asarray(np.stack([rr.ravel(), tt.ravel(), zz.ravel()], axis=-1))
    _g, _mi, jac = compute_geometry_terms(seq.map, pts)
    jac = np.asarray(jac).reshape(dr, nt, nz)
    jac = np.where(np.isfinite(jac) & (jac > 0), jac,
                   np.median(jac[np.isfinite(jac) & (jac > 0)]))
    J = jnp.asarray(jac)

    def kron(x, transpose=False):
        if transpose:
            y = jnp.einsum("ji,jkl->ikl", E[0], x)
            y = jnp.einsum("ji,kjl->kil", E[1], y)
            return jnp.einsum("ji,klj->kli", E[2], y)
        y = jnp.einsum("ij,jkl->ikl", E[0], x)
        y = jnp.einsum("ij,kjl->kil", E[1], y)
        return jnp.einsum("ij,klj->kli", E[2], y)

    # fdbund profiles of g^{aa} J on the quad grid (t, r, z layout)
    minv = jnp.transpose(_reshape_quadrature_matrix_field(
        seq, seq.geometry.metric_inv_jkl), (1, 0, 2, 3, 4))
    jq = jnp.transpose(_reshape_quadrature_scalar_field(
        seq, seq.geometry.jacobian_j), (1, 0, 2))
    w = [minv[..., a, a] * jq for a in range(3)]
    wx, wy, wz = seq.quad.w_x, seq.quad.w_y, seq.quad.w_z
    # xi_1 cutoff on the radial averaging (fdbund rule): the r-mean of
    # g^tt J ~ 1/r^2 is quadrature-divergent without it.
    xi1 = jnp.asarray(seq.basis_0.Λ[0].T)[seq.ps[0] + 1]
    wx_c = wx * (jnp.asarray(seq.quad.x_x) >= xi1)
    p_r = jnp.einsum("qrs,r,s->q", w[0], wy, wz) / (jnp.sum(wy) * jnp.sum(wz))
    p_t = jnp.einsum("qrs,q,s->r", w[1], wx_c, wz) / (jnp.sum(wx_c) * jnp.sum(wz))
    p_z = jnp.einsum("qrs,q,r->s", w[2], wx_c, wy) / (jnp.sum(wx_c) * jnp.sum(wy))
    prof = [jnp.maximum(p, 1e-8 * jnp.abs(jnp.median(p))) for p in (p_r, p_t, p_z)]
    n_bas = (seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk)
    d_bas = (seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk)
    qw = (wx, wy, wz)
    V, lam = [None] * 3, [None] * 3
    for a in range(3):
        Mn = np.asarray(_assemble_unweighted_1d_mass(n_bas[a], qw[a]))
        K = np.asarray(_assemble_weighted_1d_stiffness(
            n_bas[a], d_bas[a], qw[a] * prof[a],
            _dense_incidence_1d((nr, nt, nz)[a], types[a])))
        if a == 0:
            Mn, K = Mn[slN, slN], K[slN, slN]
        V[a], lam[a] = _simultaneous_diagonalize_pair(
            _symmetrize(jnp.asarray(Mn)), _symmetrize(jnp.asarray(K)))
    lr = np.asarray(lam[0])[:, None, None]
    lt = np.asarray(lam[1])[None, :, None]
    lz = np.asarray(lam[2])[None, None, :]
    den = lr + lt + lz
    dmax = float(den.max())
    inv_den = jnp.asarray(np.where(den > 1e-10 * dmax, 1.0 / np.maximum(den, 1e-300), 1.0 / dmax))

    def b0(x):
        y = jnp.einsum("ji,jkl->ikl", V[0], x)
        y = jnp.einsum("ji,kjl->kil", V[1], y)
        y = jnp.einsum("ji,klj->kli", V[2], y)
        y = y * inv_den
        y = jnp.einsum("ij,jkl->ikl", V[0], y)
        y = jnp.einsum("ij,kjl->kil", V[1], y)
        return jnp.einsum("ij,klj->kli", V[2], y)

    shp3 = (dr, nt, nz)

    def p3(r):
        x = J * r.reshape(shp3)          # T^T leg (diag J then E^T)
        x = kron(x, transpose=True)
        x = b0(x)
        x = kron(x)                      # T leg
        return (J * x).reshape(-1)

    return p3
