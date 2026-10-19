"""The banded basis tables (:mod:`mrx.banded`) reproduce the dense sum-factorised contractions.

- on single 1-D bases, clamped and periodic, primal and derivative, over the whole axis and over half of it, the
  banded forward and backward contractions along each axis of a grid equal the dense ``einsum`` to round-off.
- on the li383 sequence, every mass and projection plan has a band, and the banded :func:`mrx.mass.sumfact_apply`,
  :func:`mrx.quadrature.evaluate_at_xq` and :func:`mrx.quadrature.integrate_against` equal the dense ones for
  every k-form.
- the compiled banded apply contains no gather or scatter.
"""
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mrx.mass as mass
from mrx import banded, quadrature
from mrx.banded import Band
from mrx.precision import DTYPE, eps
from mrx.quadrature import composite_quad
from mrx.spline_bases import DerivativeSpline, SplineBasis, basis_table

# The banded and the dense contraction add the same products in a different order.
ROUNDOFF = eps(50)


def _rel(a, b):
    return float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))


def _random(shape, seed):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape), DTYPE)


@pytest.mark.parametrize("kind", ("clamped", "periodic", "periodic half"))
@pytest.mark.parametrize("derivative", (False, True))
def test_band_contractions_on_one_axis(kind, derivative):
    """Forward and backward along each axis of a two-component grid, against the dense ``einsum``."""
    n, p = 9, 3
    base = SplineBasis(n, p, kind.split()[0])
    knots = base.T[p:-p]
    if kind == "periodic half":
        knots = knots[:n // 2 + 1]
    x_q, _ = composite_quad(knots, p + 1)
    basis = DerivativeSpline(base) if derivative else base
    table = jnp.stack([basis_table(basis, x_q)] * 2).astype(DTYPE) * jnp.asarray([1.0, -0.5], DTYPE)[:, None, None]
    band = Band.from_dense(table, knots.size - 1)
    assert band is not None and band.values.shape[2] == basis.p + 1
    for axis in (1, 2, 3):
        shape = [2, 4, 5, 3]
        shape[axis] = basis.n
        x = _random(shape, axis)
        dense = jnp.moveaxis(jnp.einsum('c...i,cia->c...a', jnp.moveaxis(x, axis, -1), table), -1, axis)
        assert _rel(banded.forward(x, band, axis), dense) < ROUNDOFF
        v = _random(dense.shape, axis + 3)
        dense = jnp.moveaxis(jnp.einsum('c...a,cia->c...i', jnp.moveaxis(v, axis, -1), table), -1, axis)
        assert _rel(banded.backward(v, band, axis), dense) < ROUNDOFF


def test_band_rejects_tables_it_cannot_reproduce():
    """A table whose nonzeros do not follow the spans has no band."""
    table = np.zeros((1, 6, 12))
    table[0, 0, 11] = 1.0       # function 0 on the last span: (0 - 5) mod 6 = 1, fine
    table[0, 4, 0] = 1.0        # function 4 on the first span: width 5
    table[0, 5, 0] = 1.0        # function 5 on the first span: width 6 = n, no saving
    assert Band.from_dense(table, 6) is None


def _plans(seq):
    plans = {f"M{k}": (seq.mass_plan[k], seq.geometry.mass_weights[k]) for k in range(4)}
    plans.update({f"P{r}{c}": (seq.projection_plan[(r, c)], seq.geometry.reference_weights[3 if r in (1, 2) else 1])
                  for r, c in seq.projection_plan})
    return plans


def test_banded_sumfact_apply(seq):
    """Every mass and projection plan of the sequence has a band, and its banded apply is the dense one."""
    for i, (name, (plan, weights)) in enumerate(_plans(seq).items()):
        assert plan.bands_r is not None and plan.bands_c is not None, name
        x = _random(sum(int(np.prod(s)) for s in plan.shapes_c), i)
        rel = _rel(mass._apply_banded(plan, weights, x), mass._apply_dense(plan, weights, x))
        assert rel < ROUNDOFF, f"{name}: {rel:.2e}"


@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_banded_evaluate_and_integrate(seq, k):
    """``evaluate_at_xq`` and ``integrate_against`` with the banded tables are the dense ones."""
    assert seq.bands_jk is not None
    shapes = [tuple(int(n) for n in s) for s in (seq.basis_0, seq.basis_1, seq.basis_2, seq.basis_3)[k].shape]
    d = 3 if k in (1, 2) else 1
    dense, band = seq._form_comp_info(k, banded=False), seq._form_comp_info(k, banded=True)
    x = _random(sum(int(np.prod(s)) for s in shapes), k)
    f_dense = quadrature.evaluate_at_xq(x, dense, shapes, seq.quad.shape, d)
    assert _rel(quadrature.evaluate_at_xq(x, band, shapes, seq.quad.shape, d), f_dense) < ROUNDOFF
    f = _random(f_dense.shape, k + 4)
    assert _rel(quadrature.integrate_against(f, band), quadrature.integrate_against(f, dense)) < ROUNDOFF


def test_banded_apply_has_no_gather(seq):
    """The compiled banded apply of ``M_1`` is made of slices, pads and products: no gather, no scatter."""
    plan, weights = seq.mass_plan[1], seq.geometry.mass_weights[1]
    x = jnp.zeros(sum(int(np.prod(s)) for s in plan.shapes_c), DTYPE)
    lowered = jax.jit(mass._apply_banded).lower(plan, weights, x)
    assert not re.search(r"stablehlo\.(gather|scatter)", lowered.as_text())
    assert not re.search(r"\s(gather|scatter)\(", lowered.compile().as_text())


def test_dense_block_traces_the_dense_tables(seq, monkeypatch):
    """Inside :func:`mrx.banded.dense` the sum-factorised apply is traced with the dense tables, also after the
    banded apply was compiled: the choice is made outside the compiled applies. The preconditioner probes rely on
    it (:func:`mrx.metric_lumping._probe_rows`)."""
    monkeypatch.setattr(banded, "MODE", "1")
    plan, weights = seq.mass_plan[1], seq.geometry.mass_weights[1]
    x = _random(sum(int(np.prod(s)) for s in plan.shapes_c), 7)
    mass.sumfact_apply(plan, weights, x)
    used = []
    for name in ("_apply_banded", "_apply_dense"):
        monkeypatch.setattr(mass, name, lambda *a, f=getattr(mass, name), n=name: used.append(n) or f(*a))
    jax.jit(lambda v: mass.sumfact_apply(plan, weights, v)).lower(x)
    with banded.dense():
        jax.jit(lambda v: mass.sumfact_apply(plan, weights, v)).lower(x)
    assert used == ["_apply_banded", "_apply_dense"]
