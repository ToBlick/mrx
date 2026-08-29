"""``mrx.gvec``: the radial basis, the JAX evaluator against the numpy
series, the map's spline coefficients from the series coefficients, and
the data-placed knots of ``mrx.vmec``'s refit. The parser is checked
against a written state in ``test_synthetic_gvec.py``; the W7-X state in
``data/`` is read by ``test_w7x_clebsch.py``."""
import numpy as np
import pytest
from scipy.interpolate import BSpline

from mrx.gvec import StateField, evaluate, knots_at_data, radial_design


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


def test_knots_at_data_make_the_refined_sample_interpolable():
    """``knots_at_data`` places the knots from the sample, so an interpolant
    through a radial grid refined toward the edge is as well posed as one
    through a uniform grid: the collocation matrix is square and far from
    singular, the fit is exact at the nodes and off the nodes to the cubic
    spline's own accuracy. (On uniform knots the refined sample violates
    Schoenberg-Whitney and the collocation solve is singular or nearly so.)"""
    n, p = 17, 3
    u = np.arange(n, dtype=np.float64) / (n - 1)
    samples = {"uniform": u, "refined": 1.0 - (1.0 - u) ** 2}
    probe = np.linspace(0.02, 0.98, 400)

    def f(x):
        return np.exp(2.0 * x) * np.sin(3.0 * x)

    conds, errs = {}, {}
    for label, x in samples.items():
        T = np.asarray(knots_at_data(x, p, "clamped"))
        assert T.shape == (n + p + 1,) and T[0] == 0.0 and T[-1] == 1.0
        assert np.all(np.diff(T) >= 0)
        A = BSpline.design_matrix(x, T, p).toarray()
        conds[label] = np.linalg.cond(A)
        fit = BSpline(T, np.linalg.solve(A, f(x)), p)
        assert np.abs(fit(x) - f(x)).max() <= 1e-12 * np.abs(f(x)).max()
        errs[label] = np.abs(fit(probe) - f(probe)).max() / np.abs(f(probe)).max()
    print(f"\n  collocation condition numbers {conds}, off-node errors {errs}")
    assert max(conds.values()) < 1e2
    # Both interpolate the same smooth function with a cubic spline, O(h^4)
    # in the largest cell; the refined sample's largest cell (at the axis)
    # is ~1.9x the uniform one. Measured 2026-08-28 (see the print):
    # condition numbers 3.9 / 5.3, off-node errors 5.9e-5 / 3.1e-4.
    h = np.max(np.diff(samples["refined"])) / np.max(np.diff(samples["uniform"]))
    assert errs["refined"] <= 1.25 * h ** 4 * errs["uniform"] + 1e-12
    assert errs["uniform"] <= 1e-3
    # uniform knots on the refined sample: the collocation matrix is singular
    # or nearly so
    T_uni = np.concatenate([np.zeros(p + 1), np.linspace(0, 1, n - p + 1)[1:-1], np.ones(p + 1)])
    A_uni = BSpline.design_matrix(samples["refined"], T_uni, p).toarray()
    assert np.linalg.cond(A_uni) > 1e3 * max(conds.values())


def test_periodic_symbol_symmetrises_roundoff_and_rejects_a_nonuniform_row():
    """The mass row of a uniform periodic B-spline is symmetric-circulant, so
    its Fourier symbol is real; the assembly leaves the symmetry exact only to
    round-off. `_periodic_symbol` must accept the round-off-asymmetric row for
    ANY frequency (the li383 nfp=3 (16,32,16) regression: integer per-period
    modes in [-3, 3] tripped the old exp+imag check that QA's larger modes on
    the identical row did not) and reject a genuinely non-circulant row."""
    import numpy as np
    from mrx.gvec import _periodic_symbol
    n = 16
    row = np.zeros(n)
    row[0] = 1.0
    row[1] = row[-1] = 0.3
    row[2] = row[-2] = 0.05
    rng = np.random.default_rng(0)
    noisy = row + 1e-13 * rng.standard_normal(n)          # round-off asymmetry
    for freqs in ([-3, -2, -1, 0, 1, 2, 3], [8], list(range(-n // 2, n // 2))):
        sym = _periodic_symbol(noisy, np.asarray(freqs, float))
        assert np.isreal(sym).all() and np.isfinite(sym).all()
    # a symmetric row's symbol is the exact real cosine transform
    exact = np.cos(2 * np.pi * np.outer([1, 2], np.arange(n)) / n) @ row
    assert np.allclose(_periodic_symbol(row, [1, 2]), exact, atol=1e-14)
    # a genuinely non-uniform (non-circulant) row is rejected
    bad = row.copy()
    bad[1] += 0.2
    with pytest.raises(ValueError, match="not symmetric"):
        _periodic_symbol(bad, [1])
