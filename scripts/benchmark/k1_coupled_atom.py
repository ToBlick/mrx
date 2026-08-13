"""No-mixing coupled k=1 curl-curl atom: per-tensor-mode 3x3 exact inverse.

Keeps the inter-component C-terms. Weight class (the no-mixing policy):
beta_cc ~ m_c(x_c) -- each channel keeps ONLY its own-axis marginal (with
magnitude), derivative-axis ladders UNWEIGHTED, nothing mixed across
channels. Per axis a there is then ONE pencil (M^N_a[m_a], K_a); the
ladder W = g V^N drags the D-mass and cross factor along exactly
(W^T M^D W = V^T K V = Lambda -> V^D = W Lambda^{-1/2}, cross -> sqrt(L)).
The de Rham-compatible bulk windows (radial N-window start 2, D-window
start 1; banded incidence) make the ladder identity exact even restricted,
with equal radial N/D mode counts. Periodic axes have one lambda=0 mode
whose ladder image vanishes; its V^D column slot is REPLACED by the
M^D-orthonormal complement ("mean" D-mode), which lands exactly where the
per-mode formula needs it: coupling s=0 decouples the slot and its
diagonal (the formula never uses the own-axis lambda) is already correct.

Per-mode 3x3 (s_a = sqrt(lambda_a)):
  B = [[l_t+l_z, -s_r s_t, -s_r s_z],
       [-s_r s_t, l_r+l_z, -s_t s_z],
       [-s_r s_z, -s_t s_z, l_r+l_t]]
with analytic null n = (s_r, s_t, s_z) (the mode's gradient), regularised
by the grad-div surrogate + sigma * nn^T / |n|^2, sigma = l_r+l_t+l_z ->
SPD, plain batched 3x3 inverse. Physics regularisation, no floors.
"""
import numpy as np
import jax.numpy as jnp

from mrx.operators import (
    _assemble_unweighted_1d_mass,
    _assemble_weighted_1d_mass,
    _assemble_weighted_1d_stiffness,
    _dense_incidence_1d,
    _bundled_rank1_mass_factors,
    _k2_diagonal_metric_tensors,
    _symmetrize,
)
from mrx.preconditioners import (
    _arr_shape_k1,
    _theta_bulk_shape_k1,
    _zeta_bulk_shape_k1,
    _simultaneous_diagonalize_pair,
)


def _axis_ladder(Mn, K, g_slice, Md, null_tol=1e-10):
    """Pencil (Mn, K) -> V^N, lam; ladder V^D = g V^N Lambda^{-1/2}. The
    M^D-orthonormal complement of the ladder fills the lambda~0 slots first
    and any remainder is APPENDED as extra modes (radial D-window can be one
    larger than the N-window); appended slots get zero-padded V^N columns so
    partner-less D-modes enter the joint grid with zero coupling (s=0) --
    the per-mode formula then gives them their correct decoupled diagonals."""
    Vn, lam = _simultaneous_diagonalize_pair(_symmetrize(Mn), _symmetrize(K))
    lam = np.asarray(lam).copy()
    Vn_np = np.asarray(Vn)
    W = np.asarray(g_slice) @ Vn_np
    Md_np = np.asarray(Md)
    nD = Md_np.shape[0]
    lam_max = max(float(lam.max()), 1e-300)
    live = lam > null_tol * lam_max
    s_arr = np.where(live, np.sqrt(np.maximum(lam, 0.0)), 0.0)
    nN = Vn_np.shape[1]
    n_ladder = int(live.sum())
    n_comp = nD - n_ladder
    n_dead = nN - n_ladder
    n_extra = max(0, n_comp - n_dead)
    n_tot = nN + n_extra

    Vd = np.zeros((nD, n_tot))
    Vd[:, np.where(live)[0]] = W[:, live] / s_arr[live][None, :]
    if n_comp > 0:
        # slots for complements: dead lambda-slots first, then appended ones
        slots = list(np.where(~live)[0]) + list(range(nN, n_tot))
        slots = slots[:n_comp]
        w_md, v_md = np.linalg.eigh(0.5 * (Md_np + Md_np.T))
        got = 0
        for c in v_md[:, ::-1].T:
            r = c - Vd @ (Vd.T @ (Md_np @ c))
            nrm2 = float(r @ (Md_np @ r))
            if nrm2 > 1e-12:
                Vd[:, slots[got]] = r / np.sqrt(nrm2)
                got += 1
                if got == n_comp:
                    break
        if got < n_comp:
            raise RuntimeError("could not complete D-complement basis")

    Vn_out = np.zeros((Vn_np.shape[0], n_tot))
    Vn_out[:, :nN] = Vn_np
    lam_out = np.zeros(n_tot)
    lam_out[:nN] = lam
    s_out = np.zeros(n_tot)
    s_out[:nN] = s_arr
    return (jnp.asarray(Vn_out), jnp.asarray(Vd),
            jnp.asarray(s_out), jnp.asarray(lam_out))


def _np_sym(a):
    return 0.5 * (a + a.T)


def build_k1_coupled_bulk_state(seq, dirichlet: bool):
    mt = _k2_diagonal_metric_tensors(seq)
    # own-axis marginals WITH magnitude (mean-field factors, xi_1 radial cutoff)
    _, f_t_tt, _, _, _ = _bundled_rank1_mass_factors(seq, mt["beta_thetatheta"])
    _, _, f_r_rr, _, _ = _bundled_rank1_mass_factors(seq, mt["beta_rr"])
    _, _, _, f_z_zz, _ = _bundled_rank1_mass_factors(seq, mt["beta_zetazeta"])

    arr_shape = tuple(int(v) for v in _arr_shape_k1(seq, dirichlet))
    th_shape = tuple(int(v) for v in _theta_bulk_shape_k1(seq, dirichlet))
    ze_shape = tuple(int(v) for v in _zeta_bulk_shape_k1(seq, dirichlet))
    nrN, nt, nz = th_shape[0], arr_shape[1], arr_shape[2]
    nrD = arr_shape[0]
    types = seq.basis_0.types
    g_r = np.asarray(_dense_incidence_1d(seq.basis_0.nr, types[0]))
    g_t = np.asarray(_dense_incidence_1d(seq.basis_0.nt, types[1]))
    g_z = np.asarray(_dense_incidence_1d(seq.basis_0.nz, types[2]))

    # radial axis: N-window starts at 2, D-window at 1 (de Rham-compatible)
    MnR = np.asarray(_assemble_weighted_1d_mass(seq.basis_r_jk, seq.quad.w_x * f_r_rr))[2:2 + nrN, 2:2 + nrN]
    KR = np.asarray(_assemble_weighted_1d_stiffness(
        seq.basis_r_jk, seq.d_basis_r_jk, seq.quad.w_x, jnp.asarray(g_r)))[2:2 + nrN, 2:2 + nrN]
    MdR = np.asarray(_assemble_unweighted_1d_mass(seq.d_basis_r_jk, seq.quad.w_x))[1:1 + nrD, 1:1 + nrD]
    gR = g_r[1:1 + nrD, 2:2 + nrN]
    VnR, VdR, sR, lamR = _axis_ladder(MnR, KR, gR, MdR)

    MnT = np.asarray(_assemble_weighted_1d_mass(seq.basis_t_jk, seq.quad.w_y * f_t_tt))
    KT = np.asarray(_assemble_weighted_1d_stiffness(
        seq.basis_t_jk, seq.d_basis_t_jk, seq.quad.w_y, jnp.asarray(g_t)))
    MdT = np.asarray(_assemble_unweighted_1d_mass(seq.d_basis_t_jk, seq.quad.w_y))
    VnT, VdT, sT, lamT = _axis_ladder(MnT, KT, g_t, MdT)

    MnZ = np.asarray(_assemble_weighted_1d_mass(seq.basis_z_jk, seq.quad.w_z * f_z_zz))
    KZ = np.asarray(_assemble_weighted_1d_stiffness(
        seq.basis_z_jk, seq.d_basis_z_jk, seq.quad.w_z, jnp.asarray(g_z)))
    MdZ = np.asarray(_assemble_unweighted_1d_mass(seq.d_basis_z_jk, seq.quad.w_z))
    VnZ, VdZ, sZ, lamZ = _axis_ladder(MnZ, KZ, g_z, MdZ)

    lr = np.asarray(lamR)[:, None, None]
    lt = np.asarray(lamT)[None, :, None]
    lz = np.asarray(lamZ)[None, None, :]
    sr = np.asarray(sR)[:, None, None]
    st = np.asarray(sT)[None, :, None]
    sz = np.asarray(sZ)[None, None, :]
    shp = (int(np.asarray(lamR).size), int(np.asarray(lamT).size), int(np.asarray(lamZ).size))
    B = np.zeros(shp + (3, 3))
    B[..., 0, 0] = lt + lz
    B[..., 1, 1] = lr + lz
    B[..., 2, 2] = lr + lt
    B[..., 0, 1] = B[..., 1, 0] = -sr * st + 0 * lz
    B[..., 0, 2] = B[..., 2, 0] = -sr * sz + 0 * lt
    B[..., 1, 2] = B[..., 2, 1] = -st * sz + 0 * lr
    # grad-div surrogate on the analytic per-mode null n = (sr, st, sz)
    n2 = sr**2 + st**2 + sz**2 + 0 * B[..., 0, 0]
    sigma = lr + lt + lz + 0 * n2
    nvec = np.stack([sr + 0 * n2, st + 0 * n2, sz + 0 * n2], axis=-1)
    safe = np.maximum(n2, 1e-300)
    B = B + (sigma / safe)[..., None, None] * (nvec[..., :, None] * nvec[..., None, :])
    # degenerate corner (all lambdas ~ 0): identity fallback
    dead = sigma < 1e-12 * max(float(sigma.max()), 1e-300)
    B[dead] = np.eye(3)
    Binv = jnp.asarray(np.linalg.inv(B))

    return dict(
        VnR=jnp.asarray(VnR), VdR=jnp.asarray(VdR),
        VnT=jnp.asarray(VnT), VdT=jnp.asarray(VdT),
        VnZ=jnp.asarray(VnZ), VdZ=jnp.asarray(VdZ),
        Binv=Binv,
    )


def apply_k1_coupled_bulk(state, rhs_bulk):
    # shapes derived from the (static) transform matrix dims -- the state
    # dict is a traced pytree inside the jitted solve, so no int leaves.
    nrD, nrN = state["VdR"].shape[0], state["VnR"].shape[0]
    ntN, ntD = state["VnT"].shape[0], state["VdT"].shape[0]
    nzN, nzD = state["VnZ"].shape[0], state["VdZ"].shape[0]
    arr_shape = (nrD, ntN, nzN)
    th_shape = (nrN, ntD, nzN)
    ze_shape = (nrN, ntN, nzD)
    n_r = nrD * ntN * nzN
    n_t = nrN * ntD * nzN
    u_r = rhs_bulk[:n_r].reshape(arr_shape)
    u_t = rhs_bulk[n_r:n_r + n_t].reshape(th_shape)
    u_z = rhs_bulk[n_r + n_t:].reshape(ze_shape)

    def fwd(x, Vr, Vt, Vz):
        y = jnp.einsum('ji,jkl->ikl', Vr, x)
        y = jnp.einsum('ji,kjl->kil', Vt, y)
        return jnp.einsum('ji,klj->kli', Vz, y)

    def bwd(x, Vr, Vt, Vz):
        y = jnp.einsum('ij,jkl->ikl', Vr, x)
        y = jnp.einsum('ij,kjl->kil', Vt, y)
        return jnp.einsum('ij,klj->kli', Vz, y)

    cr = fwd(u_r, state["VdR"], state["VnT"], state["VnZ"])
    ct = fwd(u_t, state["VnR"], state["VdT"], state["VnZ"])
    cz = fwd(u_z, state["VnR"], state["VnT"], state["VdZ"])
    c = jnp.stack([cr, ct, cz], axis=-1)
    c = jnp.einsum('...ij,...j->...i', state["Binv"], c)
    out_r = bwd(c[..., 0], state["VdR"], state["VnT"], state["VnZ"])
    out_t = bwd(c[..., 1], state["VnR"], state["VdT"], state["VnZ"])
    out_z = bwd(c[..., 2], state["VnR"], state["VnT"], state["VdZ"])
    return jnp.concatenate([out_r.reshape(-1), out_t.reshape(-1), out_z.reshape(-1)])
