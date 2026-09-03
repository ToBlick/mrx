"""The index-free mass kernel against the indexed one it replaces.

Both ends of :func:`mrx.mass._sumfact_kernel` used an index tensor: a gather
``x[gather_idx]`` to read the element-local input, and a ``segment_sum`` to
write the element contributions back. Both index plans are separable tensor
products of per-axis element-to-DoF maps, and when every axis' map is a pure
shift (:func:`mrx.mass._shift_plan`) both are algebraically pure data
movement, done as rolled slices (:func:`mrx.mass._structured_gather`) and
shifted dense adds (:func:`mrx.mass._structured_accumulate`).

Both paths stay live: the shift is checked rather than assumed, and anything
it rejects still runs through the index tensors. So the two have to agree.
Here they are compared on the same sequence, for the mass apply and for the
closed-form diagonal, by forcing the fallback with a patched ``_shift_plan``,
which disqualifies the gather and the assembly together. The plan predicate
itself is tested on hand-built maps, because a basis whose map is not a shift
would otherwise read and write silently wrong rather than fail.

The motivation is a TPU, which has no fast path for indexed access. Measured
on a v5e at (12,24,12) p=3: the gather 1.624 ms -> 0.049 ms, the assembly
2.011 ms -> 0.061 ms. Nothing about the result should change, and that is what
this file pins.
"""

import numpy as np
import numpy.testing as npt
import pytest

import mrx
import mrx.mass as mass_mod
from mrx.mass import (_flat_dof_plan, _shift_plan, _shift_plan_axis,
                      _structured_accumulate, _structured_gather,
                      build_mass_diagonal, build_matrixfree_mass_apply)

# The two paths sum the same terms in a different order, so they differ only by
# round-off: 1e3 eps = 2.2e-13 f64 / 1.2e-4 f32, as elsewhere in the suite.
ATOL = mrx.eps(1e3)


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


@pytest.mark.parametrize("plan", (
    ((4, 3, 4), (6, 3, 6), (5, 2, 5)),      # every axis wraps
    ((2, 3, 4), (6, 3, 6), (4, 2, 4)),      # first axis clamped: ne < S
))
def test_structured_gather_matches_the_index_read(plan):
    """The rolled slices equal the gather the index plan performs.

    This is exact, not approximate: both are permuted reads of the same
    values, so a single differing element is a wrong answer rather than
    round-off. Compared against ``_flat_dof_plan`` itself, which is the plan
    the fallback actually indexes with.
    """
    (ne_x, nl_x, S_x), (ne_y, nl_y, S_y), (ne_z, nl_z, S_z) = plan
    rng = np.random.default_rng(2)
    x = np.asarray(rng.standard_normal(S_x * S_y * S_z), dtype=mrx.DTYPE)

    gx = (np.arange(ne_x)[:, None] + np.arange(nl_x)[None, :]) % S_x
    gy = (np.arange(ne_y)[:, None] + np.arange(nl_y)[None, :]) % S_y
    gz = (np.arange(ne_z)[:, None] + np.arange(nl_z)[None, :]) % S_z
    idx = np.asarray(_flat_dof_plan(gx, gy, gz, (S_x, S_y, S_z)))

    want = x[idx]
    got = np.asarray(_structured_gather(x, plan))
    assert got.shape == want.shape == (ne_x, ne_y, ne_z, nl_x, nl_y, nl_z)
    npt.assert_array_equal(got, want)


# ------------------------------------------------ the two paths on a sequence ---

@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_real_sequence_takes_the_structured_path(seq, k):
    """Guards the comparisons below: without this they could be indexed vs indexed.

    A mass operator has the same form on both sides, so these components are
    the gather plan and the assembly plan at once.
    """
    form, comp, n_comp = mass_mod._form_bases(seq, k)
    for c in range(n_comp):
        plan = _shift_plan(comp[c][1], comp[c][3], comp[c][5], form.shape[c])
        assert plan is not None, f"k={k} component {c} fell back to segment_sum"


@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_structured_mass_apply_matches_segment_sum(seq, k, monkeypatch):
    """``M_k x`` is the same vector whichever assembly the kernel chose."""
    seq = seq
    x = _probe(seq, k)
    structured = np.asarray(build_matrixfree_mass_apply(seq, k)(x))

    # Disqualify every plan, which is the only switch between the two paths.
    monkeypatch.setattr(mass_mod, "_shift_plan", lambda *a, **kw: None)
    scattered = np.asarray(build_matrixfree_mass_apply(seq, k)(x))

    assert np.isfinite(structured).all()
    npt.assert_allclose(structured, scattered,
                        rtol=0.0, atol=ATOL * np.abs(scattered).max())


@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_structured_mass_diagonal_matches_segment_sum(seq, k, monkeypatch):
    """``diag(M_k)`` agrees too, and is positive as a mass diagonal must be."""
    seq = seq
    structured = np.asarray(build_mass_diagonal(seq, k))

    monkeypatch.setattr(mass_mod, "_shift_plan", lambda *a, **kw: None)
    scattered = np.asarray(build_mass_diagonal(seq, k))

    assert structured.shape == (_n_raw(seq, k),)
    assert (structured > 0).all()
    npt.assert_allclose(structured, scattered,
                        rtol=0.0, atol=ATOL * np.abs(scattered).max())


def test_fallback_apply_still_agrees_with_the_diagonal(seq, monkeypatch):
    """With the plan disabled end to end, the scatter path is self-consistent.

    ``diag(M)_i = e_i . M e_i``, so probing the fallback apply with unit vectors
    has to reproduce the fallback diagonal. This is what says the ``segment_sum``
    branch is still a working assembly and not merely dead code that compiles.
    """
    seq = seq
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
