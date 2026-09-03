"""The separable ``(M + eps L)`` atom, and the two limits that define it.

:func:`mrx.preconditioners.build_shifted_mass_laplace_atom` factorises the
whole family ``M_k + eps L_k`` once, so that a solve whose shift changes every
step and again at every point of a line search does not rebuild anything. It
is not wired into any solve -- ``tpu/shifted_atom_measure.py`` measured it
losing on iteration count for want of a polar-core block, and the builder's
docstring carries those numbers -- but the construction is kept as the
measured answer to ``docs/research/OPEN.md`` section 3.9, so it has to keep
working.

What is pinned here is the part that would silently rot: the claim that both
existing atoms already build the SAME 1-D masses, which is the only reason a
shared eigenbasis exists at all. If either builder changes its axis weighting,
the eigenvectors stop diagonalising both operators and the atom degrades into
something plausible and wrong rather than failing. So that identity is checked
directly, and so are the two limits it implies: ``eps = 0`` must reproduce the
mass model's Kronecker bulk exactly, and the eigen-decomposition must satisfy
its own defining relations.
"""

import numpy as np
import numpy.testing as npt
import pytest

import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
from mrx.metric_lumping_laplacian import component_factors
from mrx.preconditioners import (_kron_mass_model_1d,
                                 _simultaneous_diagonalize_pair,
                                 apply_shifted_mass_laplace_atom,
                                 build_shifted_mass_laplace_atom)

ATOL = mrx.eps(1e3)


@pytest.fixture(scope="module")
def seq():
    """(4, 6, 4) p=2 polar rotating ellipse, as in the other atom tests."""
    s = DeRhamSequence((4, 6, 4), (2, 2, 2), 3,
                       ("clamped", "periodic", "periodic"), polar=True)
    s.set_map(rotating_ellipse_map(eps=0.33, kappa=1.2, R0=1.0, nfp=3))
    return s


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_both_atoms_build_the_same_one_dimensional_masses(seq, k):
    """The load-bearing identity: one eigenbasis can serve both operators.

    ``_kron_mass_model_1d`` and ``component_factors`` arrive at the axis mass
    by different routes and for different purposes, and both deliberately
    leave it unweighted with the metric carried outside as a diagonal. That
    they agree is what makes ``M + eps L`` diagonal in one basis. It is not
    documented as a contract anywhere else, so it is asserted here.
    """
    shapes, mass_1d, _ = _kron_mass_model_1d(seq, k)
    for c in range(len(shapes)):
        masses, _, _ = component_factors(seq, k, c)
        for a in range(3):
            npt.assert_allclose(
                np.asarray(masses[a]), np.asarray(mass_1d[c][a]),
                atol=ATOL, rtol=0,
                err_msg=f"k={k} component {c} axis {a}: the mass atom and the "
                        f"Laplacian atom no longer share a 1-D mass, so the "
                        f"shifted atom's eigenbasis diagonalises only one of "
                        f"them")


@pytest.mark.parametrize("k", [1, 2])
def test_the_eigen_decomposition_satisfies_its_defining_relations(seq, k):
    """``V.T M V = I`` and ``V.T L V = diag(mu)``, per axis and component.

    Everything downstream is the claim that ``M + eps L`` is
    ``1 + eps * mu`` in this basis, and that claim is exactly these two
    identities. Checked on the assembled 1-D factors rather than through the
    atom, so a failure localises to the axis.
    """
    shapes, mass_1d, _ = _kron_mass_model_1d(seq, k)
    for c in range(len(shapes)):
        _, stiffs, _ = component_factors(seq, k, c)
        for a in range(3):
            m = np.asarray(mass_1d[c][a])
            s = np.asarray(stiffs[a])
            v, mu = _simultaneous_diagonalize_pair(mass_1d[c][a], stiffs[a])
            v = np.asarray(v)
            npt.assert_allclose(v.T @ m @ v, np.eye(m.shape[0]),
                                atol=mrx.eps(1e5), rtol=0,
                                err_msg=f"k={k} c={c} a={a}: not M-orthonormal")
            npt.assert_allclose(v.T @ s @ v, np.diag(np.asarray(mu)),
                                atol=mrx.eps(1e5), rtol=0,
                                err_msg=f"k={k} c={c} a={a}: L not diagonal")


@pytest.mark.parametrize("k", [1, 2])
def test_at_zero_shift_the_atom_is_the_mass_model_inverse(seq, k):
    """``eps = 0`` must give back the Kronecker mass bulk, exactly.

    This is the limit that says the atom is built on the mass model and not on
    something adjacent to it: with ``V`` M-orthonormal, ``V V.T`` is the
    inverse of the axis mass, so the atom at zero shift applies
    ``Lam^-1 (A_r (x) A_t (x) A_z)^-1 Lam^-1``. Compared against that product
    formed directly per axis.
    """
    import jax.numpy as jnp

    atom = build_shifted_mass_laplace_atom(seq, k)
    shapes, _, _, _, scales = atom
    _, mass_1d, _ = _kron_mass_model_1d(seq, k)

    rng = np.random.default_rng(0)
    n_raw = int(sum(np.prod(s) for s in shapes))
    x = jnp.asarray(rng.standard_normal(n_raw), dtype=mrx.DTYPE)

    got = np.asarray(apply_shifted_mass_laplace_atom(atom, x, 0.0))

    want, start = [], 0
    for c, shape in enumerate(shapes):
        size = int(np.prod(shape))
        block = np.asarray(x[start:start + size]).reshape(shape)
        block = block / np.asarray(scales[c])
        for a in range(3):
            inv = np.linalg.inv(np.asarray(mass_1d[c][a]))
            block = np.moveaxis(
                np.tensordot(inv, np.moveaxis(block, a, 0), axes=(1, 0)), 0, a)
        want.append((block / np.asarray(scales[c])).reshape(-1))
        start += size

    npt.assert_allclose(got, np.concatenate(want), atol=mrx.eps(1e5), rtol=0)


@pytest.mark.parametrize("k", [1, 2])
def test_the_shift_only_changes_a_diagonal(seq, k):
    """One build serves every shift, which is the atom's whole reason to exist.

    ``eps`` must reach the apply through the reciprocal alone, so the same
    factors give a different and correct answer for a different shift without
    rebuilding. Pinned by requiring the atom to actually respond to ``eps``
    (it would be an easy bug to drop it) while staying finite and monotone in
    magnitude, since ``1 / (1 + eps * mu)`` shrinks as ``eps`` grows and the
    ``mu`` are non-negative.
    """
    import jax.numpy as jnp

    atom = build_shifted_mass_laplace_atom(seq, k)
    shapes = atom[0]
    rng = np.random.default_rng(1)
    n_raw = int(sum(np.prod(s) for s in shapes))
    x = jnp.asarray(rng.standard_normal(n_raw), dtype=mrx.DTYPE)

    norms = [float(np.linalg.norm(np.asarray(
        apply_shifted_mass_laplace_atom(atom, x, eps))))
        for eps in (0.0, 1e-3, 1e-1, 1.0, 10.0)]

    assert all(np.isfinite(norms)), f"non-finite atom output: {norms}"
    assert norms[0] > norms[-1], "eps did not reach the apply"
    for lo, hi in zip(norms, norms[1:]):
        assert hi <= lo * (1 + 1e-6), f"not monotone in eps: {norms}"


def test_the_shift_is_traceable(seq):
    """``eps`` is a runtime value, not a rebuild trigger.

    The relaxation varies it inside a jitted step and again along a line
    search. If it were static the step would retrace on every change, which is
    the objection this atom exists to answer, so it has to survive being a
    tracer.
    """
    import jax
    import jax.numpy as jnp

    atom = build_shifted_mass_laplace_atom(seq, 2)
    n_raw = int(sum(np.prod(s) for s in atom[0]))
    x = jnp.ones(n_raw, dtype=mrx.DTYPE)

    f = jax.jit(lambda v, eps: apply_shifted_mass_laplace_atom(atom, v, eps))
    a = np.asarray(f(x, 1e-3))
    b = np.asarray(f(x, 1e-1))
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    assert not np.allclose(a, b)


def test_a_mismatched_mass_factor_is_rejected(seq, monkeypatch):
    """A shape disagreement must fail loudly, not precondition wrongly.

    The builder's guard exists because the failure it catches is silent: wrong
    eigenvectors still produce a symmetric positive operator that looks like a
    preconditioner and merely converges badly.
    """
    import mrx.preconditioners as pc

    real = pc._kron_mass_model_1d

    def truncated(seq_, k_, d_raw=None):
        shapes, mass_1d, scales = real(seq_, k_, d_raw)
        bad = tuple((m[0][:-1, :-1], m[1], m[2]) for m in mass_1d)
        return shapes, bad, scales

    monkeypatch.setattr(pc, "_kron_mass_model_1d", truncated)
    with pytest.raises(ValueError, match="same matrix"):
        pc.build_shifted_mass_laplace_atom(seq, 2)
