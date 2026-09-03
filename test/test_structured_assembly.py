"""The index-free mass kernel, and the shift plan it requires.

Both ends of :func:`mrx.mass._sumfact_kernel` once used an index tensor: a
gather ``x[gather_idx]`` to read the element-local input, and a ``segment_sum``
to write the element contributions back. Both index plans are separable tensor
products of per-axis element-to-DoF maps, and when every axis' map is a pure
shift (:func:`mrx.mass._shift_plan`) both are algebraically pure data
movement, done as rolled slices (:func:`mrx.mass._structured_gather`) and
shifted dense adds (:func:`mrx.mass._structured_accumulate`).

There is no second path. A tensor-product B-spline basis always gives that
shift, so the plan is a precondition of the kernel rather than an optimisation
it can decline, and :func:`mrx.mass._shift_plan` raises on anything else --
a basis numbered differently would otherwise read and write silently wrong.

So what is pinned here is the predicate on hand-built maps (which it accepts,
which it rejects), the two movements against index arithmetic written out
independently of the production plan, and the fact that a real sequence
satisfies the precondition for every component of every degree.

The motivation is a TPU, which has no fast path for indexed access. Measured
on a v5e at (12,24,12) p=3: the gather 1.624 ms -> 0.049 ms, the assembly
2.011 ms -> 0.061 ms.
"""

import numpy as np
import numpy.testing as npt
import pytest

import mrx
import mrx.mass as mass_mod
from mrx.mass import (_shift_plan, _shift_plan_axis, _structured_accumulate,
                      _structured_gather, build_mass_diagonal,
                      build_matrixfree_mass_apply)

# Reassociating a sum changes only the rounding: 1e3 eps = 2.2e-13 f64 /
# 1.2e-4 f32, as elsewhere in the suite.
ATOL = mrx.eps(1e3)


def _n_raw(seq, k):
    """Raw (unextracted) DoF count of the ``k``-form space."""
    return sum(int(np.prod(s)) for s in getattr(seq, f"basis_{k}").shape)


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
    """A relabelled basis is not a shift, and the kernel cannot assemble it."""
    ne, nloc, S = 6, 3, 6
    g = (np.arange(ne)[:, None] + np.arange(nloc)[None, :]) % S
    g = g.copy()
    g[2, 1] = (g[2, 1] + 1) % S
    with pytest.raises(ValueError, match="not the shift"):
        _shift_plan_axis(g, S)


def test_shift_plan_axis_rejects_a_reversed_map():
    """Descending local DoFs are a shift by ``-l``, which is a different sum."""
    ne, nloc, S = 6, 3, 6
    g = (np.arange(ne)[:, None] - np.arange(nloc)[None, :]) % S
    with pytest.raises(ValueError, match="not the shift"):
        _shift_plan_axis(g, S)


def test_shift_plan_axis_rejects_a_non_matrix():
    """Only a 2-D ``(ne, nloc)`` map is a plan; anything else is not a shape it knows."""
    with pytest.raises(ValueError, match="2-D"):
        _shift_plan_axis(np.arange(6), 6)


def test_shift_plan_rejects_when_a_single_axis_fails():
    """One bad axis disqualifies the component, because the sum is separable.

    The message names the axis, since the caller sees only a component index.
    """
    ok = (np.arange(6)[:, None] + np.arange(3)[None, :]) % 6
    bad = ok.copy()
    bad[0, 0] = 5
    assert len(_shift_plan(ok, ok, ok, (6, 6, 6))) == 3
    with pytest.raises(ValueError, match="axis y"):
        _shift_plan(ok, bad, ok, (6, 6, 6))
    with pytest.raises(ValueError, match="axis z"):
        _shift_plan(ok, ok, bad, (6, 6, 6))


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
    """The rolled slices equal the gather an explicit index plan performs.

    This is exact, not approximate: both are permuted reads of the same
    values, so a single differing element is a wrong answer rather than
    round-off. The flat index tensor is built here rather than imported, so
    the oracle does not share code with the thing under test.
    """
    (ne_x, nl_x, S_x), (ne_y, nl_y, S_y), (ne_z, nl_z, S_z) = plan
    rng = np.random.default_rng(2)
    x = np.asarray(rng.standard_normal(S_x * S_y * S_z), dtype=mrx.DTYPE)

    gx = (np.arange(ne_x)[:, None] + np.arange(nl_x)[None, :]) % S_x
    gy = (np.arange(ne_y)[:, None] + np.arange(nl_y)[None, :]) % S_y
    gz = (np.arange(ne_z)[:, None] + np.arange(nl_z)[None, :]) % S_z
    idx = (gx[:, None, None, :, None, None] * (S_y * S_z)
           + gy[None, :, None, None, :, None] * S_z
           + gz[None, None, :, None, None, :])

    want = x[idx]
    got = np.asarray(_structured_gather(x, plan))
    assert got.shape == want.shape == (ne_x, ne_y, ne_z, nl_x, nl_y, nl_z)
    npt.assert_array_equal(got, want)


# ------------------------------------------------- the kernel on a sequence ---

@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_real_sequence_satisfies_the_shift_precondition(seq, k):
    """The kernel's precondition holds on a real sequence, so it never raises.

    A mass operator has the same form on both sides, so these components are
    the read plan and the assembly plan at once. If this ever fails, the mass
    apply fails with it -- there is no other assembly to fall back to.
    """
    form, comp, n_comp = mass_mod._form_bases(seq, k)
    for c in range(n_comp):
        plan = _shift_plan(comp[c][1], comp[c][3], comp[c][5], form.shape[c])
        assert len(plan) == 3, f"k={k} component {c}"


def test_the_mass_diagonal_agrees_with_probing_the_apply(seq):
    """``diag(M)_i = e_i . M e_i``, which ties the closed form to the matvec.

    :func:`build_mass_diagonal` collapses the two halves of the kernel against
    squared basis tables instead of running the matvec, so it is a separate
    derivation that happens to share the element tables. Probing is O(n)
    applies and is why this is one degree rather than four.
    """
    k = 0
    apply = build_matrixfree_mass_apply(seq, k)
    diag = np.asarray(build_mass_diagonal(seq, k))

    n = _n_raw(seq, k)
    assert diag.shape == (n,)
    assert (diag > 0).all(), "a mass diagonal is positive"

    probed = np.empty(n, dtype=np.float64)
    for i in range(n):
        e = np.zeros(n, dtype=mrx.DTYPE)
        e[i] = 1.0
        probed[i] = np.asarray(apply(e))[i]

    npt.assert_allclose(probed, diag, rtol=0.0, atol=ATOL * np.abs(diag).max())
