# test_relaxation.py
# Tests for the sparse-branch relaxation module (mrx.relaxation_sparse).
# Based on the z-pinch equilibrium from Freidberg.

# %%
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from jax.numpy import cos, pi, sin

from mrx.derham_sequence import DeRhamSequence
from mrx.relaxation import (TimeStepper, State, compute_divergence_norm,
                            compute_force, compute_helicity, initial_state,
                            relaxation_loop)
from mrx.utils import evaluate_at_xq

jax.config.update("jax_enable_x64", True)

# --- Z-pinch helpers ---


def F(p):
    r, θ, z = p
    return jnp.array([r * cos(2 * pi * θ), r * sin(2 * pi * θ), z])


def B0(p):
    r, θ, _ = p
    Bθ = 2 * r * (1 - r**2 / 2)
    return jnp.array([-sin(2 * pi * θ), cos(2 * pi * θ), 0.0]) * Bθ


def p_exact(x):
    r, _, _ = x
    return 1 / 3 * (5 - 2 * r**2) * (1 - r**2)**2 * jnp.ones(1)


# --- fixtures ---


@pytest.fixture(scope="module")
def seq() -> DeRhamSequence:
    s = DeRhamSequence(
        (6, 6, 1),
        (3, 3, 0),
        6,
        ("clamped", "periodic", "constant"),
        F,
        polar=True
    )
    s.evaluate_1d()
    s.assemble_all_sparse()
    s.compute_nullspaces()
    return s


@pytest.fixture(scope="module")
def B_hat(seq: DeRhamSequence) -> jnp.ndarray:
    rhs = seq.p2_dbc(B0)
    B = seq.apply_inverse_mass_matrix(rhs, 2)
    B, _ = seq.apply_leray_projection(B, k=2)
    return B


@pytest.fixture(scope="module")
def relaxed(seq, B_hat):
    """Perturbed B_hat after 5 relaxation steps (computed once per module)."""
    ts = TimeStepper(seq=seq)
    key = jax.random.PRNGKey(42)
    noise = 1e-3 * jax.random.normal(key, shape=B_hat.shape)
    B_noisy = B_hat + seq.apply_inverse_mass_matrix(noise, 2)
    B_noisy, _ = seq.apply_leray_projection(B_noisy, k=2)
    state, traces = relaxation_loop(
        B_noisy, ts,
        num_iters_outer=1,
        num_iters_inner=5,
        dt0=1.0,
    )
    return state, traces


def compute_pressure_0form(B_hat, seq):
    """Compute the 0-form pressure via k=1 Leray projection of J x H.

    Computes J = curl(B), H = P_{2->1}(B), projects J x H onto
    1-forms (no DBC), then applies the k=1 Leray projection which
    solves (grad p, grad w) = (JxH, grad w) with Neumann BC.
    """
    J = seq.apply_weak_curl(B_hat)
    H_dual = seq.apply_projection_matrix(
        B_hat, 2, 1, True, dirichlet_out=False)
    H = seq.apply_inverse_mass_matrix(H_dual, 1, dirichlet=False)
    JxH_dual = seq.cross_product_projection(J, H, 1, 1, 1, False, True, False)
    JxH = seq.apply_inverse_mass_matrix(JxH_dual, 1, dirichlet=False)
    _, p_hat = seq.apply_leray_projection(JxH, k=1)
    return p_hat


def compute_pressure_3form(B_hat, seq):
    """Compute the 3-form pressure from the k=2 Leray projection in compute_force."""
    _, p3, _, _, _ = compute_force(B_hat, seq)
    return p3


# --- tests ---


class TestZPinchPressure:
    def test_pressures(self, seq, B_hat):
        """Compute 0-form and 3-form pressures once, then check:
        1. Both approximate the exact z-pinch pressure (up to a constant).
        2. Both integrate to zero (null spaces projected out by CG solver).
        3. The exact pressure integrates to π/2.
        4. The 0-form and 3-form pressures agree pointwise.
        """
        wJ = seq.jacobian_j * seq.quad.w  # (n_q,)
        vol = jnp.sum(wJ)

        # --- evaluate pressures at quad points (tensor product) ---
        # The Leray projection decomposes v = v_div_free - grad(p),
        # so p_physical = -p_leray.
        quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
        comp_info_0, comp_shapes_0 = seq._form_comp_info(0)
        comp_info_3, comp_shapes_3 = seq._form_comp_info(3)

        p0 = compute_pressure_0form(B_hat, seq)
        p0_jk = -evaluate_at_xq(
            seq.e0_T @ p0, comp_info_0, comp_shapes_0, quad_shape, 1)

        p3 = compute_pressure_3form(B_hat, seq)
        p3_jk = -evaluate_at_xq(
            seq.e3_dbc_T @ p3, comp_info_3, comp_shapes_3, quad_shape, 1)
        # Pushforward k=3 divides by det(DF)
        p3_phys_jk = p3_jk / seq.jacobian_j[:, None]

        # exact pressure at quad points
        pe_jk = jax.vmap(p_exact)(seq.quad.x)  # (n_q, 1)

        # --- means ---
        mean_p0 = jnp.einsum("ik,i->", p0_jk, wJ) / vol
        mean_p3 = jnp.einsum("ik,i->", p3_phys_jk, wJ) / vol
        mean_pe = jnp.einsum("ik,i->", pe_jk, wJ) / vol

        pe_shifted = pe_jk - mean_pe

        # --- 0-form pressure matches exact (up to constant) ---
        d0 = (p0_jk - mean_p0) - pe_shifted
        err_p0 = (jnp.einsum("ik,ik,i->", d0, d0, wJ)
                  / jnp.einsum("ik,ik,i->", pe_shifted, pe_shifted, wJ)) ** 0.5
        npt.assert_allclose(err_p0, 0.0, atol=5e-2,
                            err_msg=f"0-form pressure L2 error (up to const): {err_p0}")

        # --- 3-form pressure matches exact (up to constant) ---
        d3 = (p3_phys_jk - mean_p3) - pe_shifted
        err_p3 = (jnp.einsum("ik,ik,i->", d3, d3, wJ)
                  / jnp.einsum("ik,ik,i->", pe_shifted, pe_shifted, wJ)) ** 0.5
        npt.assert_allclose(err_p3, 0.0, atol=5e-2,
                            err_msg=f"3-form pressure L2 error (up to const): {err_p3}")

        # --- both pressures integrate to zero ---
        int_p0 = jnp.einsum("ik,i->", p0_jk, wJ)
        npt.assert_allclose(int_p0, 0.0, atol=1e-10,
                            err_msg=f"0-form pressure integral: {int_p0}")

        int_p3 = jnp.einsum("ik,i->", p3_phys_jk, wJ)
        npt.assert_allclose(int_p3, 0.0, atol=1e-8,
                            err_msg=f"3-form pressure integral: {int_p3}")

        # --- 0-form and 3-form agree pointwise ---
        d03 = p0_jk - p3_phys_jk
        err_agree = (jnp.einsum("ik,ik,i->", d03, d03, wJ)
                     / jnp.einsum("ik,ik,i->", p0_jk, p0_jk, wJ)) ** 0.5
        npt.assert_allclose(err_agree, 0.0, atol=5e-2,
                            err_msg=f"0-form vs 3-form mismatch: {err_agree}")


class TestRelaxation:
    def test_force_decreases(self, relaxed):
        """Relaxation should reduce the Lorentz force norm."""
        state, traces = relaxed
        assert traces["force_norm"][-1] < traces["force_norm"][0], (
            f"Force did not decrease: "
            f"{traces['force_norm'][0]:.4e} -> {traces['force_norm'][-1]:.4e}")

    def test_energy_decreases(self, relaxed):
        """Relaxation should reduce the magnetic energy."""
        state, traces = relaxed
        assert traces["energy"][-1] < traces["energy"][0], (
            f"Energy did not decrease: "
            f"{traces['energy'][0]:.4e} -> {traces['energy'][-1]:.4e}")

    def test_traces_structure(self, relaxed):
        """traces dict should have all expected keys, each with 2 entries."""
        _, traces = relaxed
        expected_keys = {"force_norm", "helicity", "timestep", "energy",
                         "picard_residua", "picard_iterations",
                         "velocity_norm", "divergence_B", "eta", "iteration"}
        assert set(traces.keys()) == expected_keys
        # num_iters_outer=1: record at iteration 0 (initial) and 1 => 2 entries
        for k, v in traces.items():
            assert len(v) == 2, f"traces['{k}'] has {len(v)} entries, expected 2"

    def test_cg_descent_runs(self, seq, B_hat):
        """CG descent should run without error and produce a finite force norm."""
        from mrx.relaxation import DescentMethod
        ts = TimeStepper(seq=seq, descent_method=DescentMethod.CONJUGATE_GRADIENT)
        state, _ = relaxation_loop(B_hat, ts, num_iters_outer=1, num_iters_inner=2,
                                   dt0=1.0)
        assert jnp.isfinite(state.F_norm)


class TestHelicityAndDivergence:
    def test_divergence_of_projected_B(self, seq, B_hat):
        """Leray-projected B_hat should be divergence-free."""
        div_norm = compute_divergence_norm(B_hat, seq)
        npt.assert_allclose(div_norm, 0.0, atol=1e-10,
                            err_msg=f"div B norm: {div_norm:.2e}")

    def test_helicity_shape(self, seq, B_hat):
        """compute_helicity returns a scalar and a 1-form DOF array."""
        A_guess = jnp.zeros(seq.n1_dbc)
        h, A = compute_helicity(B_hat, seq, A_guess)
        assert A.shape == (seq.n1_dbc,), f"A shape {A.shape} != ({seq.n1_dbc},)"
        assert jnp.isfinite(h)

    def test_helicity_zero(self, seq, B_hat):
        """Z-pinch has purely toroidal B_θ with no B_z, so A·B = 0 and helicity is zero."""
        A_guess = jnp.zeros(seq.n1_dbc)
        h, _ = compute_helicity(B_hat, seq, A_guess)
        npt.assert_allclose(float(h), 0.0, atol=1e-10,
                            err_msg=f"Z-pinch helicity should be zero, got {h:.2e}")


class TestInitialState:
    def test_shapes(self, seq, B_hat):
        """initial_state should produce arrays with the correct DOF sizes."""
        ts = TimeStepper(seq=seq)
        state = initial_state(B_hat, ts, dt=2.0)
        n = seq.n2_dbc
        assert state.B_n.shape == (n,)
        assert state.v.shape == (n,)
        assert state.F_prev.shape == (n,)
        assert state.s_history.shape == (ts.history_size, n)
        assert state.y_history.shape == (ts.history_size, n)
        npt.assert_allclose(state.dt, 2.0)
