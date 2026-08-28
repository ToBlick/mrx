"""``mrx.gvec`` state files: the radial basis, and (``needs_data``) the evaluation
of a GVEC state against the pyGVEC export of the same equilibrium --
``MRX_GVEC_STATE`` and ``MRX_W7X_FILE`` name the two files."""
import os

import numpy as np
import pytest

from mrx.gvec import StateField, evaluate, profile_spline, radial_design, read_state


def test_radial_basis_is_a_clamped_partition_of_unity():
    sp = np.linspace(0.0, 1.0, 11)
    s = np.linspace(0.0, 1.0, 97)
    D = radial_design(sp, 5, s)
    assert D.shape == (97, len(sp) - 1 + 5)
    assert np.allclose(D.sum(1), 1.0)
    assert D[0, 0] == 1.0 and D[-1, -1] == 1.0            # clamped ends


def test_state_field_matches_the_numpy_series():
    """The JAX evaluator and the tensor-grid series agree on a made-up state."""
    import jax
    import jax.numpy as jnp
    rng = np.random.default_rng(3)
    sp = np.linspace(0.0, 1.0, 7)
    m = np.array([0, 1, 2, 3, 1, 2])
    n = np.array([0, 0, 5, -5, 10, -10])
    block = dict(m=m, n=n, coef=rng.normal(size=(6, len(sp) - 1 + 3)), sin_cos=2, deg=3)
    field = StateField(block, sp, 5)
    s_, th, ze = np.array([0.15, 0.6, 0.97]), np.array([0.1, 0.45]), np.array([0.0, 0.8])
    grid = evaluate(block, sp, s_, 2 * np.pi * th, 2 * np.pi * ze / 5)
    pts = jnp.array([[a, b, c] for a in s_ for b in th for c in ze])
    mine = np.asarray(jax.vmap(field)(pts)).reshape(grid.shape)
    assert np.abs(mine - grid).max() < 1e-12
    block["sin_cos"] = 1
    grid = evaluate(block, sp, s_, 2 * np.pi * th, 2 * np.pi * ze / 5)
    mine = np.asarray(jax.vmap(StateField(block, sp, 5))(pts)).reshape(grid.shape)
    assert np.abs(mine - grid).max() < 1e-12


@pytest.mark.needs_data
def test_state_reproduces_the_export():
    state, export = os.environ.get("MRX_GVEC_STATE"), os.environ.get("MRX_W7X_FILE")
    if not (state and export and os.path.isfile(state) and os.path.isfile(export)):
        pytest.skip("set MRX_GVEC_STATE and MRX_W7X_FILE to the state and its export")
    import h5py
    st = read_state(state)
    with h5py.File(export, "r") as f:
        n = tuple(int(f.attrs[k]) for k in ("n_rho", "n_theta", "n_zeta"))
        assert int(f.attrs["nfp"]) == st["nfp"]
        ep = np.asarray(f["eval_points"])
        ref = {k: np.asarray(f[k]).reshape(n) for k in
               ("R", "Z", "pressure", "clebsch/dPhi_dr", "clebsch/dchi_dr", "clebsch/LA")}
    rho, th, ze = (np.unique(ep[:, i]) for i in range(3))
    theta, zeta = 2 * np.pi * th, 2 * np.pi * ze / st["nfp"]
    for blk, key in (("X1", "R"), ("X2", "Z"), ("LA", "clebsch/LA")):
        assert np.abs(evaluate(st[blk], st["sp"], rho, theta, zeta) - ref[key]).max() < 1e-10
    dphi = profile_spline(st, "phi").derivative()(rho)
    assert np.abs(dphi - ref["clebsch/dPhi_dr"][:, 0, 0]).max() < 1e-10
    dchi = profile_spline(st, "iota")(rho) * dphi
    scale = np.abs(ref["clebsch/dchi_dr"]).max()
    assert np.abs(dchi - ref["clebsch/dchi_dr"][:, 0, 0]).max() < 1e-4 * scale
    p = profile_spline(st, "pressure")(rho)
    assert np.abs(p - ref["pressure"][:, 0, 0]).max() < 1e-4 * ref["pressure"].max()
    # the closed-form lambda the initial condition uses, at the export's points
    import jax
    import jax.numpy as jnp
    from mrx.gvec import load_clebsch
    cb = load_clebsch(state)
    pts = jnp.array([[rho[i], th[j], ze[k]] for i in (1, 20, 49) for j in (0, 7) for k in (0, 31)])
    mine = np.asarray(jax.vmap(cb["lam_h"])(pts))
    want = np.array([ref["clebsch/LA"][i, j, k] for i in (1, 20, 49) for j in (0, 7) for k in (0, 31)])
    assert np.abs(mine - want).max() < 1e-10


# ---------------------------------------------------------------------------
# the map: spline coefficients from the series coefficients, no grid
# ---------------------------------------------------------------------------

def _made_up_block(rng, sp, deg, sin_cos):
    """A state block whose modes include ones beyond the Nyquist frequency
    of the small test sequences (m = 5, n/nfp = 3) so the damping of an
    unresolved mode is exercised."""
    m = np.array([0, 1, 2, 1, 5, 2])
    n = np.array([0, 0, 5, -5, 10, 15])
    coef = rng.normal(size=(len(m), len(sp) - 1 + deg))
    coef[m > 0, 0] = 0.0                                  # the axis rows of a state
    return dict(m=m, n=n, coef=coef, sin_cos=sin_cos, deg=deg)


@pytest.fixture(scope="module")
def odd_p_seq():
    from mrx.derham_sequence import DeRhamSequence
    return DeRhamSequence((5, 8, 4), (3, 3, 3), 4, ("clamped", "periodic", "periodic"),
                          polar=True, betti_numbers=(1, 1, 0, 0))


def _gauss_on(breakpoints, n):
    xi, wi = np.polynomial.legendre.leggauss(n)
    lo, hi = breakpoints[:-1], breakpoints[1:]
    pts = (0.5 * (lo + hi)[:, None] + 0.5 * (hi - lo)[:, None] * xi[None, :]).ravel()
    return pts, (0.5 * (hi - lo)[:, None] * wi[None, :]).ravel()


@pytest.mark.parametrize("which", ["tiny_seq", "odd_p_seq"])
@pytest.mark.parametrize("sin_cos", [2, 1])
def test_series_tensor_coefficients_are_the_l2_projection(request, which, sin_cos):
    """The per-mode closed form satisfies the tensor-product normal
    equations ``(M_r x M_t x M_z) C = int Lambda_ijk f`` with the mass
    matrices and the moments of the series assembled by an independent
    Gauss rule (12 points per element, radially on the union of the
    state's and the map's knots) -- including the mode beyond Nyquist, at
    both Greville layouts (on the knots at odd p, between them at even p)."""
    import jax
    import jax.numpy as jnp
    from mrx.gvec import block_knots, series_tensor_coefficients
    seq = request.getfixturevalue(which)
    rng = np.random.default_rng(11)
    sp = np.linspace(0.0, 1.0, 5)
    block = _made_up_block(rng, sp, 3, sin_cos)
    C = series_tensor_coefficients(block, sp, 5, seq)
    br, bt, bz = seq.basis_0.Λ
    bp_r = np.unique(np.concatenate([block_knots(block, sp), np.asarray(br.T)]))
    rules = [_gauss_on(bp_r, 12)] + [_gauss_on(np.linspace(0.0, 1.0, b.n + 1), 12) for b in (bt, bz)]
    B = [np.asarray(b.collocation_matrix(jnp.asarray(r[0])), dtype=np.float64)
         for b, r in zip((br, bt, bz), rules)]
    M = [Bi.T @ (r[1][:, None] * Bi) for Bi, r in zip(B, rules)]
    grid = np.stack(np.meshgrid(*(r[0] for r in rules), indexing="ij"), axis=-1).reshape(-1, 3)
    f = np.asarray(jax.vmap(StateField(block, sp, 5))(jnp.asarray(grid))).reshape(
        tuple(len(r[0]) for r in rules))
    wf = np.einsum("q,r,s,qrs->qrs", rules[0][1], rules[1][1], rules[2][1], f)
    b = np.einsum("qi,rj,sk,qrs->ijk", B[0], B[1], B[2], wf)
    MC = np.einsum("ai,bj,ck,ijk->abc", M[0], M[1], M[2], C)
    assert np.abs(MC - b).max() < 1e-11 * np.abs(b).max()


def test_radial_coefficients_are_exact_on_the_states_knots(odd_p_seq):
    """When the map's radial space contains the state's (same degree, same
    knots) the projection returns the state's coefficients themselves."""
    from mrx.gvec import _radial_coefficients
    seq = odd_p_seq
    br = seq.basis_0.Λ[0]
    T = np.asarray(br.T)
    rng = np.random.default_rng(2)
    block = dict(m=np.array([0, 1]), n=np.array([0, 5]),
                 coef=rng.normal(size=(2, br.n)), sin_cos=2, deg=br.p, T=T)
    c = _radial_coefficients(block, None, br)
    assert np.abs(c - block["coef"].T).max() < 1e-12


@pytest.mark.parametrize("which", ["tiny_seq", "odd_p_seq"])
def test_angular_l2_symbol_solves_the_normal_equations(request, which):
    """``gamma(m) cos(2 pi m x_j)`` is the L2 projection of ``cos(2 pi m
    theta)`` onto the periodic basis: it satisfies ``M c = b`` with the mass
    matrix and the moments assembled by an independent quadrature -- so the
    closed-form moment ``h sinc(m h)^(p+1)`` and the circulant symbol are
    right, at every frequency including beyond Nyquist."""
    import jax.numpy as jnp
    from mrx.gvec import _angular_symbol
    seq = request.getfixturevalue(which)
    bt = seq.basis_0.Λ[1]
    N, p = bt.n, bt.p
    xi, wi = np.polynomial.legendre.leggauss(3 * p + 4)          # a different rule
    pts = ((np.arange(N)[:, None] + 0.5 * (xi[None, :] + 1.0)) / N).ravel()
    w = np.tile(0.5 * wi / N, N)
    B = np.asarray(bt.collocation_matrix(jnp.asarray(pts)), dtype=np.float64)
    M = B.T @ (w[:, None] * B)
    x = np.asarray(bt.greville_points(), dtype=np.float64)
    freqs = np.array([0, 1, 2, N // 2, N // 2 + 1, N + 1])
    gamma = _angular_symbol(bt, freqs)
    for m, g in zip(freqs, gamma):
        b = B.T @ (w * np.cos(2 * np.pi * m * pts))
        c = g * np.cos(2 * np.pi * m * x)
        assert np.abs(M @ c - b).max() < 1e-13


def test_state_field_wall_derivative_is_the_left_limit():
    """No clip on rho: at rho = 1 exactly the autodiff radial derivative of
    the series is the polynomial's own (the clip halved it)."""
    import jax
    import jax.numpy as jnp
    rng = np.random.default_rng(5)
    sp = np.linspace(0.0, 1.0, 5)
    f = StateField(_made_up_block(rng, sp, 3, 2), sp, 5)
    d = jax.grad(f)
    x1, x0 = jnp.array([1.0, 0.3, 0.2]), jnp.array([1.0 - 1e-9, 0.3, 0.2])
    assert abs(float(f(x1)) - float(f(x0))) < 1e-7
    assert abs(float(d(x1)[0]) - float(d(x0)[0])) < 1e-6 * abs(float(d(x0)[0]))
