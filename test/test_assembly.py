"""The assembly's two structural checks on the session geometry: the
projection pairs are transposes of each other, and the sum-factorised
kernel's shift plan accepts both axis kinds. The mass applies themselves are
covered by the manufactured solutions (``test_vacuum.py``): a wrong weight
moves their error by a factor.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import mrx

# Roundoff identity relative to the size of the result: 1e3 eps
# (2.2e-13 f64 / 1.2e-4 f32).
IDENT = mrx.eps(1e3)


@pytest.mark.parametrize(("pair", "partner"), (((2, 1), (1, 2)), ((0, 3), (3, 0))))
def test_projection_pairs_are_transposes(seq, pair, partner):
    """``<P_12 x, y> = <x, P_21 y>`` on the extracted Dirichlet spaces."""
    k_in, k_out = pair
    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.standard_normal(seq.n(k_in, True)), dtype=mrx.DTYPE)
    y = jnp.asarray(rng.standard_normal(seq.n(k_out, True)), dtype=mrx.DTYPE)
    x, y = seq.project_parity(x, k_in, 1, True), seq.project_parity(y, k_out, 1, True)
    lhs = float(y @ seq.apply_projection_matrix(x, *pair, True, dirichlet_out=True))
    rhs = float(x @ seq.apply_projection_matrix(y, *partner, True, dirichlet_out=True))
    assert abs(lhs - rhs) < IDENT * abs(lhs), f"{pair}: {lhs:.6e} vs {rhs:.6e}"


def test_shift_plan_accepts_both_axis_kinds_and_names_a_bad_one():
    """``_shift_plan`` is the kernel's one precondition, and it has no oracle.

    Everything else in ``mrx.mass`` is covered by the apply above. This is the
    gate that decides whether the indexless gather and assembly are usable at
    all, so the two shapes it must accept -- a periodic axis, which wraps, and
    a clamped one, which does not -- and the rejection are checked directly.
    Pure index arithmetic, no sequence and no solves.
    """
    from mrx.mass import _shift_plan

    ne, nloc = 4, 3
    e = np.arange(ne)[:, None]
    lo = np.arange(nloc)[None, :]

    S_per = ne                      # periodic: e + l wraps
    S_clamp = ne + nloc - 1         # clamped: e + l never reaches S
    g_per = (e + lo) % S_per
    g_clamp = e + lo

    plan = _shift_plan(g_clamp, g_per, g_per, (S_clamp, S_per, S_per))
    assert plan == ((ne, nloc, S_clamp), (ne, nloc, S_per), (ne, nloc, S_per))

    # Permuting two DoFs is still a bijection, so nothing but the shift itself
    # can catch it. The message has to name the axis, since the caller passes
    # three and a plan is rebuilt per component.
    g_bad = g_per.copy()
    g_bad[0, 0], g_bad[0, 1] = g_bad[0, 1], g_bad[0, 0]
    with pytest.raises(ValueError, match="axis y"):
        _shift_plan(g_clamp, g_bad, g_per, (S_clamp, S_per, S_per))
