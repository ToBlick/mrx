"""P12 div-div gradient machinery for k=1: P = P_A(coupled, raw) + P12 B_div P12^T.

NO L0 solves, NO projection (live with the overlap). The k=1 weak grad-div
term is the same physical form int (div u)(div v) computed through the space
ABOVE: transfer the 1-form proxy to 2-form space (where div is the strong,
exact operator) and apply a div-div atom there.

- P12 here denotes Pi_21: V2 -> V1 primal (prolongation), so the
  preconditioner term is Pi B Pi^T (dual -> dual -> primal -> primal).
  Pointwise proxy identification v1 = (g/J) v2, Greville-collocated at each
  V1 component's (windowed) grid: 9 blocks, each
  diag((g_ab/J) at grid_a) . (E_r (x) E_t (x) E_z), E = 1D collocation
  matrices of V2-component-b's windowed bases at V1-component-a's windowed
  Greville abscissae. Adjoint = transpose.
- B_div: the k=2 div-div atom by the coupled recipe. Single channel, weight
  1/J with a full rank-1 mean-field fit m_r m_t m_z (a single channel has
  no sharing conflicts). Paired ladders per axis (pencil (M^N, K[m_a]),
  V^D = g V^N Lambda^{-1/2} against M^D[m_a]); per tensor mode the div-div
  symbol is RANK-1: B = t t^T, t = (s_r, s_t, s_z) -> the capped pinv
  t (t^T c)/|t|^4 IS the cap (zero on the curls = t-perp; P_A owns them).
- BC FLIP (Tobias): the aux div-div uses the OPPOSITE BC (aux_dbc = not
  dirichlet) -- integration by parts swaps essential/natural at the wall.

Bulk-only (the polar core is the surgery Schur's job, as for every atom).
"""
import numpy as np
import jax.numpy as jnp

from mrx.geometry import compute_geometry_terms
from mrx.operators import (
    _assemble_unweighted_1d_mass,
    _assemble_weighted_1d_mass,
    _assemble_weighted_1d_stiffness,
    _dense_incidence_1d,
    _bundled_rank1_mass_factors,
    _k2_divdiv_weight_tensor,
    _symmetrize,
)
from mrx.preconditioners import (
    _arr_shape_k1,
    _theta_bulk_shape_k1,
    _zeta_bulk_shape_k1,
)
from k1_coupled_atom import _axis_ladder


def _grev(basis, sl=None, clamped=False, eps=1e-7):
    g = np.asarray(basis.greville_points())
    if sl is not None:
        g = g[sl]
    if clamped:
        g = np.clip(g, eps, 1.0 - eps)
    return g


def _colloc(basis, pts, sl=None):
    C = np.asarray(basis.collocation_matrix(jnp.asarray(pts)))
    if sl is not None:
        C = C[:, sl]
    return jnp.asarray(C)


def build_p12_divdiv(seq, ops, dirichlet: bool):
    aux_dbc = not dirichlet
    nr, nt, nz = int(seq.basis_0.nr), int(seq.basis_0.nt), int(seq.basis_0.nz)
    types = seq.basis_0.types
    N1 = [seq.basis_0.Λ[a] for a in range(3)]
    D1 = [seq.basis_0.dΛ[a].s for a in range(3)]

    # radial windows: N starts at 2 (drop outer fn under the relevant dbc),
    # D starts at 1; angular full.
    v1_shapes = (tuple(int(v) for v in _arr_shape_k1(seq, dirichlet)),
                 tuple(int(v) for v in _theta_bulk_shape_k1(seq, dirichlet)),
                 tuple(int(v) for v in _zeta_bulk_shape_k1(seq, dirichlet)))
    nrD1 = v1_shapes[0][0]
    nrN1 = v1_shapes[1][0]
    slN1 = slice(2, 2 + nrN1)
    slD1 = slice(1, 1 + nrD1)
    nrN2 = nr - 2 - (1 if aux_dbc else 0)
    nrD2 = nr - 2
    slN2 = slice(2, 2 + nrN2)
    slD2 = slice(1, 1 + nrD2)
    # V2 component shapes: a-component = N along a, D transverse.
    v2_shapes = ((nrN2, nt, nz), (nrD2, nt, nz), (nrD2, nt, nz))
    v2_ax = ((slN2, None, None), (slD2, None, None), (slD2, None, None))
    v2_bas = ((N1[0], D1[1], D1[2]), (D1[0], N1[1], D1[2]), (D1[0], D1[1], N1[2]))
    v1_bas = ((D1[0], N1[1], N1[2]), (N1[0], D1[1], N1[2]), (N1[0], N1[1], D1[2]))
    v1_ax = ((slD1, None, None), (slN1, None, None), (slN1, None, None))

    # V1 component Greville grids + pointwise weights W_ab = g_ab / J
    W = [[None] * 3 for _ in range(3)]
    E = [[None] * 3 for _ in range(3)]  # E[a][b] = (Cr, Ct, Cz)
    for a in range(3):
        gr = _grev(v1_bas[a][0], v1_ax[a][0], clamped=(types[0] == "clamped"))
        gt = _grev(v1_bas[a][1], None, clamped=(types[1] == "clamped"))
        gz = _grev(v1_bas[a][2], None, clamped=(types[2] == "clamped"))
        rr, tt, zz = np.meshgrid(gr, gt, gz, indexing="ij")
        pts = jnp.asarray(np.stack([rr.ravel(), tt.ravel(), zz.ravel()], axis=-1))
        gmat, _minv, jac = compute_geometry_terms(seq.map, pts)
        gmat = np.asarray(gmat).reshape(len(gr), len(gt), len(gz), 3, 3)
        jac = np.asarray(jac).reshape(len(gr), len(gt), len(gz))
        jac = np.where(np.isfinite(jac) & (jac > 0), jac, np.median(jac[np.isfinite(jac) & (jac > 0)]))
        for b in range(3):
            wab = gmat[..., a, b] / jac
            wab = np.where(np.isfinite(wab), wab, 0.0)
            W[a][b] = jnp.asarray(wab)
            E[a][b] = (_colloc(v2_bas[b][0], gr, v2_ax[b][0]),
                       _colloc(v2_bas[b][1], gt, None),
                       _colloc(v2_bas[b][2], gz, None))

    def _kron(Cs, x, transpose=False):
        cr, ct, cz = Cs
        if transpose:
            y = jnp.einsum('ji,jkl->ikl', cr, x)
            y = jnp.einsum('ji,kjl->kil', ct, y)
            return jnp.einsum('ji,klj->kli', cz, y)
        y = jnp.einsum('ij,jkl->ikl', cr, x)
        y = jnp.einsum('ij,kjl->kil', ct, y)
        return jnp.einsum('ij,klj->kli', cz, y)

    # ---- B_div: coupled-recipe k=2 div-div atom ---------------------------
    wj = _k2_divdiv_weight_tensor(seq)  # 1/J on the quad grid, (t, r, z)
    _sc, f_t, f_r, f_z, _err = _bundled_rank1_mass_factors(seq, wj)
    m = [f_r, f_t, f_z]
    g1d = [_dense_incidence_1d((nr, nt, nz)[a], types[a]) for a in range(3)]
    q_w = (seq.quad.w_x, seq.quad.w_y, seq.quad.w_z)
    n_bas = (seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk)
    d_bas = (seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk)
    Vn, Vd, ss = [None] * 3, [None] * 3, [None] * 3
    for a in range(3):
        Mn = np.asarray(_assemble_unweighted_1d_mass(n_bas[a], q_w[a]))
        K = np.asarray(_assemble_weighted_1d_stiffness(
            n_bas[a], d_bas[a], q_w[a] * m[a], g1d[a]))
        Md = np.asarray(_assemble_weighted_1d_mass(d_bas[a], q_w[a] * m[a]))
        gs = np.asarray(g1d[a])
        if a == 0:
            Mn = Mn[slN2, slN2]
            K = K[slN2, slN2]
            Md = Md[slD2, slD2]
            gs = gs[slD2, slN2]
        Vn[a], Vd[a], ss[a], _lam = _axis_ladder(
            _symmetrize(jnp.asarray(Mn)), _symmetrize(jnp.asarray(K)),
            gs, jnp.asarray(Md))
    t_r = np.asarray(ss[0])[:, None, None]
    t_t = np.asarray(ss[1])[None, :, None]
    t_z = np.asarray(ss[2])[None, None, :]
    tnorm2 = t_r**2 + t_t**2 + t_z**2 + 0.0 * (t_r + t_t + t_z)
    live = tnorm2 > 1e-10 * float(tnorm2.max())
    inv_t4 = jnp.asarray(np.where(live, 1.0 / np.maximum(tnorm2, 1e-300) ** 2, 0.0))
    tvec = (jnp.asarray(t_r + 0 * tnorm2), jnp.asarray(t_t + 0 * tnorm2),
            jnp.asarray(t_z + 0 * tnorm2))
    # V2 comp a transforms: own axis V^N, transverse V^D
    v2_V = ((Vn[0], Vd[1], Vd[2]), (Vd[0], Vn[1], Vd[2]), (Vd[0], Vd[1], Vn[2]))

    def b_div(u2):  # list of 3 dual tensors -> list of 3 primal tensors
        c = [None] * 3
        for a in range(3):
            c[a] = _kron(v2_V[a], u2[a], transpose=True)
        tc = tvec[0] * c[0] + tvec[1] * c[1] + tvec[2] * c[2]
        s = tc * inv_t4
        return [_kron(v2_V[a], tvec[a] * s) for a in range(3)]

    n_sizes = [int(np.prod(s)) for s in v1_shapes]

    def p_div_bulk(rb):
        r1 = [rb[:n_sizes[0]].reshape(v1_shapes[0]),
              rb[n_sizes[0]:n_sizes[0] + n_sizes[1]].reshape(v1_shapes[1]),
              rb[n_sizes[0] + n_sizes[1]:].reshape(v1_shapes[2])]
        # Pi^T: V1-dual -> V2-dual
        u2 = []
        for b in range(3):
            acc = jnp.zeros(v2_shapes[b])
            for a in range(3):
                acc = acc + _kron(E[a][b], W[a][b] * r1[a], transpose=True)
            u2.append(acc)
        x2 = b_div(u2)
        # Pi: V2-primal -> V1-primal
        out = []
        for a in range(3):
            acc = jnp.zeros(v1_shapes[a])
            for b in range(3):
                acc = acc + W[a][b] * _kron(E[a][b], x2[b])
            out.append(acc)
        return jnp.concatenate([o.reshape(-1) for o in out])

    return p_div_bulk
