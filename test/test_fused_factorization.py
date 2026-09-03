"""The two-stage mass factorization against the three-stage one it replaces.

:func:`mrx.mass._to_quadrature` and :func:`mrx.mass._from_quadrature` used to
run the textbook sum factorization: three sequential contractions, one per
axis, each of width ``nloc`` (3 or 4 at p=3). That minimises FLOPs, which is
the right objective on a machine that charges per FLOP.

Neither a TPU nor a GPU does, at these widths. Measured on a k=2 component of
li383 at (12,24,12) p=3, folding the y and z stages into one contraction of
width ``nly * nlz`` costs 1.5x the arithmetic and returns 1.48-1.70x on a v5e,
1.23-1.49x on an H200 and 1.62x on a VM CPU (``tpu/factorization_ab.py``).
Folding all three loses on every backend.

The fold is an einsum reassociation, so it is exact up to summation order and
there is no basis it cannot handle -- unlike the shift plan of
``test_structured_assembly.py``, it needs no fallback. What it does need is a
guard against the reshapes silently transposing something: an axis order that
is wrong but self-consistent would produce a plausible wrong answer rather
than an error, and the mass apply is symmetric enough to hide it. So the fold
is compared here against the explicit three-stage chain it replaces, on real
bases from a real sequence, for every component of every degree.
"""

import numpy as np
import numpy.testing as npt
import pytest

import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
from mrx.mass import _form_bases, _fuse_yz, _from_quadrature, _to_quadrature

# Reassociating a sum changes only the rounding, so this is the round-off
# bound used elsewhere in the suite: 2.2e-13 at f64, 1.2e-4 at f32.
ATOL = mrx.eps(1e3)


@pytest.fixture(scope="module")
def seq():
    """(4, 6, 4) p=2 polar rotating ellipse, as in the other mass tests."""
    s = DeRhamSequence((4, 6, 4), (2, 2, 2), 3,
                       ("clamped", "periodic", "periodic"), polar=True)
    s.set_map(rotating_ellipse_map(eps=0.33, kappa=1.2, R0=1.0, nfp=3))
    return s


def _chain3_to_quadrature(Bx, By, Bz, x_local):
    """The three-stage column half, written out as the reference."""
    import jax.numpy as jnp

    t1 = jnp.einsum('xqb,xyzbdf->xyzqdf', Bx, x_local)
    t2 = jnp.einsum('yrd,xyzqdf->xyzqrf', By, t1)
    return jnp.einsum('zsf,xyzqrf->xyzqrs', Bz, t2)


def _chain3_from_quadrature(Bx, By, Bz, u):
    """The three-stage row half, written out as the reference."""
    import jax.numpy as jnp

    s1 = jnp.einsum('xqa,xyzqrs->xyzars', Bx, u)
    s2 = jnp.einsum('yrc,xyzars->xyzacs', By, s1)
    return jnp.einsum('zse,xyzacs->xyzace', Bz, s2)


def _bases(seq, k, c):
    """1-D basis values of component ``c`` of the ``k``-form."""
    _, comp, n_comp = _form_bases(seq, k)
    assert c < n_comp
    Bx, _, By, _, Bz, _ = comp[c]
    return Bx, By, Bz


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_fused_column_half_matches_the_three_stage_chain(seq, k):
    """Two contractions must give what three gave, for every component."""
    import jax

    _, comp, n_comp = _form_bases(seq, k)
    for c in range(n_comp):
        Bx, By, Bz = _bases(seq, k, c)
        ne_x, _, nlx = Bx.shape
        ne_y, _, nly = By.shape
        ne_z, _, nlz = Bz.shape
        x_local = jax.random.normal(
            jax.random.PRNGKey(c), (ne_x, ne_y, ne_z, nlx, nly, nlz),
            dtype=mrx.DTYPE)

        want = np.asarray(_chain3_to_quadrature(Bx, By, Bz, x_local))
        # gather_plan=None with gather_idx as an identity would re-read the
        # vector, so the element-local input is fed in directly instead: the
        # gather is tested in test_structured_assembly.py and is not what is
        # under test here.
        got = np.asarray(_to_quadrature(
            (Bx, By, Bz, _fuse_yz(By, Bz)), x_local,
            gather_idx=Ellipsis, gather_plan=None))
        npt.assert_allclose(got, want, atol=ATOL, rtol=0,
                            err_msg=f"k={k} component {c}")


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_fused_row_half_matches_the_three_stage_chain(seq, k):
    """The transpose half, which reuses the same fused table the other way."""
    import jax

    _, comp, n_comp = _form_bases(seq, k)
    for c in range(n_comp):
        Bx, By, Bz = _bases(seq, k, c)
        ne_x, qx, _ = Bx.shape
        ne_y, qy, _ = By.shape
        ne_z, qz, _ = Bz.shape
        u = jax.random.normal(jax.random.PRNGKey(100 + c),
                              (ne_x, ne_y, ne_z, qx, qy, qz), dtype=mrx.DTYPE)

        want = np.asarray(_chain3_from_quadrature(Bx, By, Bz, u))
        got = np.asarray(_from_quadrature((Bx, By, Bz, _fuse_yz(By, Bz)), u))
        npt.assert_allclose(got, want, atol=ATOL, rtol=0,
                            err_msg=f"k={k} component {c}")


def test_the_two_halves_stay_transposes_of_each_other(seq):
    """``<to(x), u> == <x, from(u)>``, which is what makes the mass symmetric.

    The fold touches both halves, and a matching error in the two would cancel
    in the mass apply and pass the oracle tests. This pins them separately
    against each other rather than against the assembled operator.
    """
    import jax

    Bx, By, Bz = _bases(seq, 2, 0)
    Byz = _fuse_yz(By, Bz)
    ne_x, qx, nlx = Bx.shape
    ne_y, qy, nly = By.shape
    ne_z, qz, nlz = Bz.shape

    x = jax.random.normal(jax.random.PRNGKey(1),
                          (ne_x, ne_y, ne_z, nlx, nly, nlz), dtype=mrx.DTYPE)
    u = jax.random.normal(jax.random.PRNGKey(2),
                          (ne_x, ne_y, ne_z, qx, qy, qz), dtype=mrx.DTYPE)

    lhs = float(np.sum(np.asarray(_to_quadrature(
        (Bx, By, Bz, Byz), x, gather_idx=Ellipsis, gather_plan=None)) *
        np.asarray(u)))
    rhs = float(np.sum(np.asarray(x) * np.asarray(
        _from_quadrature((Bx, By, Bz, Byz), u))))
    npt.assert_allclose(lhs, rhs, rtol=mrx.eps(1e4), atol=0)


def test_the_fused_table_is_small_enough_to_precompute(seq):
    """The fold is only free because the table it needs is negligible.

    ``(ne_y, ne_z, qy*qz, nly*nlz)`` carries no x extent at all, which is what
    separates this fold from the three-axis one: the three-axis table is the
    same thing times ``ne_x * qx * nlx``, so it grows with the radial
    refinement while this does not. That factor is 36 on this small fixture
    and 144 on li383 at (12,24,12) p=3, and it is why folding two axes costs
    166 KB and folding three costs 24 MB per component.
    """
    Bx, By, Bz = _bases(seq, 2, 0)
    Byz = _fuse_yz(By, Bz)

    ne_y, qy, nly = By.shape
    ne_z, qz, nlz = Bz.shape
    assert Byz.shape == (ne_y, ne_z, qy * qz, nly * nlz)

    ne_x, qx, nlx = Bx.shape
    three_axis = ne_x * ne_y * ne_z * (qx * qy * qz) * (nlx * nly * nlz)
    assert Byz.size * (ne_x * qx * nlx) == three_axis
    assert ne_x * qx * nlx > 1
