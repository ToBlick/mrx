# %%
# test_fixed_point_solvers.py
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from mrx.iterative_solvers import picard_solver, newton_solver

jax.config.update("jax_enable_x64", True)
TOL = 1e-12


@pytest.mark.parametrize("solver", [picard_solver, newton_solver])
@pytest.mark.parametrize("case", ["linear", "dottie", "multidim"])
def test_fixed_point_solvers(solver, case):
    """Unified fixed-point solver tests for Picard and Newton."""

    # ------------------------------------------------------------
    # Define test functions
    # ------------------------------------------------------------
    if case == "linear":
        a, b = 0.5, 1.0
        def f(z): return (a * z[0] + b, None)
        z_init = (jnp.array([0.0]), None)
        true_val = b / (1 - a)

    elif case == "dottie":
        def f(z):
            _z, _aux = z
            return jnp.cos(_z), _aux
        z_init = (jnp.array([1.0]), None)
        true_val = 0.739085133215160641655312087673

    elif case == "multidim":
        # 2D linear contraction: f(v) = Av + b
        A = jnp.array([[0.5, 0.2],
                       [0.1, 0.7]])
        b = jnp.array([1.0, -1.0])
        def f(z):
            x, aux = z
            return A @ x + b, aux
        z_init = (jnp.array([1.0, -3.0]), None)
        # Solve (I - A)x = b
        true_val = jnp.linalg.solve(jnp.eye(2) - A, b)

    # ------------------------------------------------------------
    # Run solver
    # ------------------------------------------------------------
    (z_star, _aux), res, iters = solver(f, z_init, tol=TOL)

    # ------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------
    npt.assert_allclose(z_star, true_val, atol=1e-10)
    assert res < TOL

    # convergence speed expectations
    if solver is newton_solver:
        assert iters < 10
    else:
        assert iters < 200
