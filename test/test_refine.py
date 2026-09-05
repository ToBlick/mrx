"""The refined solves meet the residual tolerance in the residual precision,
and nothing leaks out of the working dtype.

A float32 working dtype runs every Krylov solve as iterative refinement
against the float64 view of the sequence (``mrx.precision``,
``mrx.solvers.refine``). Checked on the session field: a mass solve's
float64 residual is below the tolerance, the Leray projection's result is
divergence-free to it (the saddle solve's upper residual), and the results
come back in the working dtype unless asked for otherwise. At a float64
working dtype the solves are plain and the same statements hold.
"""
import jax.numpy as jnp
import numpy as np

from mrx.precision import DTYPE, RESIDUAL_DTYPE, SOLVE_TOL
from mrx.relaxation import compute_force


def test_refined_mass_solve_residual(seq, b0):
    b = seq.apply_mass_matrix(b0, 2, True)
    x = seq.apply_inverse_mass_matrix(b, 2, dtype=RESIDUAL_DTYPE)
    assert x.dtype == RESIDUAL_DTYPE
    assert seq.apply_inverse_mass_matrix(b, 2).dtype == DTYPE
    on = seq.residual if seq.residual is not None else seq
    r = b.astype(RESIDUAL_DTYPE) - seq.__class__.apply_mass_matrix(on, x, 2, True)

    def norm(v):   # the criterion's norm: the mass atom of the dual 2-forms
        return float(jnp.sqrt(v @ seq.apply_mass_matrix_preconditioner(v, 2, True)))

    rel = norm(r) / norm(b.astype(RESIDUAL_DTYPE))
    print(f"\n  refined M_2 solve: residual {rel:.2e} in the mass-atom norm (tol {SOLVE_TOL:.0e})")
    assert rel <= 10 * SOLVE_TOL, rel


def test_leray_projection_is_divergence_free_to_tolerance(seq, b0):
    F, p, J, X, JxX = compute_force(b0, seq)
    assert F.dtype == DTYPE and p.dtype == DTYPE and JxX.dtype == DTYPE
    div_F = seq.apply_derivative_matrix(F, 2, dirichlet_in=True, dirichlet_out=True)
    div_v = seq.apply_derivative_matrix(JxX, 2, dirichlet_in=True, dirichlet_out=True)

    def norm(v):   # the criterion's norm: the mass atom of the dual 3-forms
        return float(jnp.sqrt(v @ seq.apply_mass_matrix_preconditioner(v, 3, True)))

    rel = norm(div_F) / norm(div_v)
    print(f"  Leray: |div F| / |div JxB| = {rel:.2e}, |F| / |JxB| = "
          f"{float(seq.l2_norm(F, 2) / seq.l2_norm(JxX, 2)):.2e}")
    # F is stored in the working dtype: its rounding alone leaves eps |JxB| / h.
    assert rel <= 100 * max(SOLVE_TOL, float(np.finfo(DTYPE).eps)), rel


def test_refine_discards_a_correction_that_raises_the_residual():
    """A pass that increases the residual is dropped and the loop stops.

    The operator is the identity; the inner 'solve' returns ``100 r``, which
    overshoots. Without the stall check, ``x`` would walk to 100 and the
    residual would grow; with it, ``x`` stays at the initial zero.
    """
    from mrx.solvers import refine

    def apply_res(x: jnp.ndarray) -> jnp.ndarray:
        return x

    def solve(r: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return 100.0 * r, jnp.int32(-1)

    b = jnp.ones(4, dtype=RESIDUAL_DTYPE)
    x, info = refine(apply_res, solve, b, tol=1e-8, max_passes=6)
    assert jnp.allclose(x, 0.0)
    assert int(info) > 0
