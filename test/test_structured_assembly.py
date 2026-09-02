"""The index-free element assembly against the ``segment_sum`` it replaces.

:func:`mrx.mass._structured_accumulate` performs the element-to-DoF sum of the
sum-factorised kernel as shifted dense adds rather than a scatter, which is
taken whenever every axis' element-to-DoF map is a pure shift
(:func:`mrx.mass._shift_plan`). Both paths are live: the shift plan is checked
rather than assumed, and anything it rejects still assembles through
``segment_sum``.

So the two have to agree. Here the structured path is compared against the
scatter on the same sequence, for the mass apply and for the closed-form
diagonal, by forcing the fallback with a patched ``_shift_plan``. The plan
predicate itself is tested directly on hand-built maps, because a basis whose
map is not a shift would otherwise assemble silently wrong rather than fail.

The motivation is a TPU: indexed writes are the one thing it has no fast path
for. Measured on a v5e at (12,24,12) p=3, the assembly went from 2.011 ms to
0.061 ms. Nothing about the result should change, and that is what this file
pins.
"""

import numpy as np
import numpy.testing as npt
import pytest

import mrx
import mrx.mass as mass_mod
from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
from mrx.mass import (_shift_plan, _shift_plan_axis, _structured_accumulate,
                      build_mass_diagonal, build_matrixfree_mass_apply)

# The two paths sum the same terms in a different order, so they differ only by
# round-off: 1e3 eps = 2.2e-13 f64 / 1.2e-4 f32, as elsewhere in the suite.
ATOL = mrx.eps(1e3)


@pytest.fixture(scope="module")
def mf_seq():
    """(4, 6, 4) p=2 polar rotating ellipse, matching test_matrixfree_masses."""
    seq = DeRhamSequence((4, 6, 4), (2, 2, 2), 3,
                         ("clamped", "periodic", "periodic"), polar=True)
    seq.set_map(rotating_ellipse_map(eps=0.33, kappa=1.2, R0=1.0, nfp=3))
    return seq


def _n_raw(seq, k):
    """Raw (unextracted) DoF count of the ``k``-form space."""
    return sum(int(np.prod(s)) for s in getattr(seq, f"basis_{k}").shape)


def _probe(seq, k, seed=0):
    """A fixed random vector in the raw ``k``-form DoF space."""
    rng = np.random.default_rng(seed)
    return np.asarray(rng.standard_normal(_n_raw(seq, k)), dtype=mrx.DTYPE)


# ------------------------------------------------------- the plan predicate ---

def test_shift_plan_axis_accepts_a_periodic_axis():
    """A periodic axis wraps, and ``e + l`` mod ``S`` is exactly the map."""
    ne, nloc, S = 6, 3, 6
    g = (np.arange(ne)[:, None] + np.arange(nloc)[None, :]) % S
    assert _shift_plan_axis(g, S) == (ne, nloc, S)


def test_shift_plan_axis_accepts_a_clamped_axis():
    """A clamped axis has ``ne = S - p`` elements and never wraps.

    ``e + l`` stays below ``S``, so the modulo is the identity and the map is
    still a shift -- just one that leaves the last DoFs untouched by the
    lowest elements. The accumulator pads for that, so this must be accepted;
    on the real sequence the radial axis is this case.
    """
    S, p = 4, 2
    ne, nloc = S - p, p + 1
    g = np.arange(ne)[:, None] + np.arange(nloc)[None, :]
    assert g.max() < S
    assert _shift_plan_axis(g, S) == (ne, nloc, S)


def test_shift_plan_axis_rejects_a_permuted_map():
    """A relabelled basis is not a shift, and must fall back rather than assemble."""
    ne, nloc, S = 6, 3, 6
    g = (np.arange(ne)[:, None] + np.arange(nloc)[None, :]) % S
    g = g.copy()
    g[2, 1] = (g[2, 1] + 1) % S
    assert _shift_plan_axis(g, S) is None


def test_shift_plan_axis_rejects_a_reversed_map():
    """Descending local DoFs are a shift by ``-l``, which is a different sum."""
    ne, nloc, S = 6, 3, 6
    g = (np.arange(ne)[:, None] - np.arange(nloc)[None, :]) % S
    assert _shift_plan_axis(g, S) is None


def test_shift_plan_axis_rejects_a_non_matrix():
    """Only a 2-D ``(ne, nloc)`` map is a plan; anything else is not a shape it knows."""
    assert _shift_plan_axis(np.arange(6), 6) is None


def test_shift_plan_rejects_when_a_single_axis_fails():
    """One bad axis disqualifies the component, because the sum is separable."""
    ok = (np.arange(6)[:, None] + np.arange(3)[None, :]) % 6
    bad = ok.copy()
    bad[0, 0] = 5
    assert _shift_plan(ok, ok, ok, (6, 6, 6)) is not None
    assert _shift_plan(ok, bad, ok, (6, 6, 6)) is None
    assert _shift_plan(ok, ok, bad, (6, 6, 6)) is None


# --------------------------------------------------------- the accumulator ---

@pytest.mark.parametrize("plan", (
    ((4, 3, 4), (6, 3, 6), (5, 2, 5)),      # every axis wraps
    ((2, 3, 4), (6, 3, 6), (4, 2, 4)),      # first axis clamped: ne < S
))
def test_structured_accumulate_matches_the_index_sum(plan):
    """The shifted adds equal the scatter written out as an explicit loop."""
    (ne_x, nl_x, S_x), (ne_y, nl_y, S_y), (ne_z, nl_z, S_z) = plan
    rng = np.random.default_rng(1)
    y = np.asarray(
        rng.standard_normal((ne_x, ne_y, ne_z, nl_x, nl_y, nl_z)),
        dtype=mrx.DTYPE)

    want = np.zeros((S_x, S_y, S_z), dtype=mrx.DTYPE)
    for ex in range(ne_x):
        for ey in range(ne_y):
            for ez in range(ne_z):
                for lx in range(nl_x):
                    for ly in range(nl_y):
                        for lz in range(nl_z):
                            want[(ex + lx) % S_x,
                                 (ey + ly) % S_y,
                                 (ez + lz) % S_z] += y[ex, ey, ez, lx, ly, lz]

    got = np.asarray(_structured_accumulate(y, plan))
    assert got.shape == want.shape
    npt.assert_allclose(got, want, rtol=0.0, atol=ATOL * np.abs(want).max())


# ------------------------------------------------ the two paths on a sequence ---

@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_real_sequence_takes_the_structured_path(mf_seq, k):
    """Guards the comparisons below: without this they could be scatter vs scatter."""
    form, comp, n_comp = mass_mod._form_bases(mf_seq, k)
    for c in range(n_comp):
        plan = _shift_plan(comp[c][1], comp[c][3], comp[c][5], form.shape[c])
        assert plan is not None, f"k={k} component {c} fell back to segment_sum"


@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_structured_mass_apply_matches_segment_sum(mf_seq, k, monkeypatch):
    """``M_k x`` is the same vector whichever assembly the kernel chose."""
    seq = mf_seq
    x = _probe(seq, k)
    structured = np.asarray(build_matrixfree_mass_apply(seq, k)(x))

    # Disqualify every plan, which is the only switch between the two paths.
    monkeypatch.setattr(mass_mod, "_shift_plan", lambda *a, **kw: None)
    scattered = np.asarray(build_matrixfree_mass_apply(seq, k)(x))

    assert np.isfinite(structured).all()
    npt.assert_allclose(structured, scattered,
                        rtol=0.0, atol=ATOL * np.abs(scattered).max())


@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_structured_mass_diagonal_matches_segment_sum(mf_seq, k, monkeypatch):
    """``diag(M_k)`` agrees too, and is positive as a mass diagonal must be."""
    seq = mf_seq
    structured = np.asarray(build_mass_diagonal(seq, k))

    monkeypatch.setattr(mass_mod, "_shift_plan", lambda *a, **kw: None)
    scattered = np.asarray(build_mass_diagonal(seq, k))

    assert structured.shape == (_n_raw(seq, k),)
    assert (structured > 0).all()
    npt.assert_allclose(structured, scattered,
                        rtol=0.0, atol=ATOL * np.abs(scattered).max())


def test_fallback_apply_still_agrees_with_the_diagonal(mf_seq, monkeypatch):
    """With the plan disabled end to end, the scatter path is self-consistent.

    ``diag(M)_i = e_i . M e_i``, so probing the fallback apply with unit vectors
    has to reproduce the fallback diagonal. This is what says the ``segment_sum``
    branch is still a working assembly and not merely dead code that compiles.
    """
    seq = mf_seq
    monkeypatch.setattr(mass_mod, "_shift_plan", lambda *a, **kw: None)
    k = 0
    apply = build_matrixfree_mass_apply(seq, k)
    diag = np.asarray(build_mass_diagonal(seq, k))

    n = _n_raw(seq, k)
    probed = np.empty(n, dtype=np.float64)
    for i in range(n):
        e = np.zeros(n, dtype=mrx.DTYPE)
        e[i] = 1.0
        probed[i] = np.asarray(apply(e))[i]

    npt.assert_allclose(probed, diag, rtol=0.0, atol=ATOL * np.abs(diag).max())
