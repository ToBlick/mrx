# %%
# test_utils.py
import copy
import os
import tempfile

import h5py
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
from mrx.relaxation import MRXDiagnostics, State
from mrx.utils import (
    norm_2,
    jacobian_determinant,
    inv33,
    div,
    curl,
    grad,
    l2_product,
    assemble,
    evaluate_at_xq,
    integrate_against,
    append_to_trace_dict,
    save_trace_dict_to_hdf5,
    run_relaxation_loop,
    update_config,
    DEVICE_PRESETS,
    DEFAULT_CONFIG,
    default_trace_dict,
)

jax.config.update("jax_enable_x64", True)


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def seq():
    """Create a DeRhamSequence for testing."""
    F = rotating_ellipse_map(eps=0.2, kappa=1.7, nfp=3)
    return DeRhamSequence(
        (4, 4, 4),
        (2, 2, 2),
        2,
        ("clamped", "periodic", "periodic"),
        F,
        polar=True,
        dirichlet=True
    )


@pytest.fixture
def seq_simple():
    """Create a simple DeRhamSequence with identity mapping."""
    def F_identity(x):
        return x
    return DeRhamSequence(
        (4, 4, 4),
        (2, 2, 2),
        2,
        ("clamped", "periodic", "periodic"),
        F_identity,
        polar=False,
        dirichlet=True
    )


# ============================================================================
# Tests for norm_2
# ============================================================================
def test_norm_2_basic(seq):
    """Test norm_2 with a simple vector field."""
    seq.evaluate_1d()
    seq.assemble_M2()
    
    # Create a simple test vector field
    u = jnp.ones(seq.E2.shape[0])
    
    norm = norm_2(u, seq)
    
    assert isinstance(norm, (float, jnp.ndarray))
    assert norm > 0, "Norm should be positive"
    assert jnp.isfinite(norm), "Norm should be finite"


def test_norm_2_zero_vector(seq):
    """Test norm_2 with zero vector."""
    seq.evaluate_1d()
    seq.assemble_M2()
    
    u = jnp.zeros(seq.E2.shape[0])
    
    norm = norm_2(u, seq)
    
    npt.assert_allclose(norm, 0.0, atol=1e-10)


def test_norm_2_scaling(seq):
    """Test that norm_2 scales correctly."""
    seq.evaluate_1d()
    seq.assemble_M2()
    
    u = jnp.ones(seq.E2.shape[0])
    norm1 = norm_2(u, seq)
    norm2 = norm_2(2.0 * u, seq)
    
    npt.assert_allclose(norm2, 2.0 * norm1, rtol=1e-10)


# ============================================================================
# Tests for jacobian_determinant
# ============================================================================
def test_jacobian_determinant_identity():
    """Test jacobian_determinant with identity function."""
    def F(x):
        return x
    
    det_func = jacobian_determinant(F)
    
    x = jnp.array([0.5, 0.5, 0.5])
    det = det_func(x)
    
    npt.assert_allclose(det, 1.0, atol=1e-10)


def test_jacobian_determinant_scaling():
    """Test jacobian_determinant with scaling function."""
    def F(x):
        return 2.0 * x
    
    det_func = jacobian_determinant(F)
    
    x = jnp.array([0.5, 0.5, 0.5])
    det = det_func(x)
    
    # Determinant of 2*I is 2^3 = 8
    npt.assert_allclose(det, 8.0, atol=1e-10)


def test_jacobian_determinant_rotation():
    """Test jacobian_determinant with rotation (determinant should be 1)."""
    def F(x):
        # 90 degree rotation around z-axis
        return jnp.array([-x[1], x[0], x[2]])
    
    det_func = jacobian_determinant(F)
    
    x = jnp.array([1.0, 0.0, 0.5])
    det = det_func(x)
    
    npt.assert_allclose(det, 1.0, atol=1e-10)


# ============================================================================
# Tests for inv33
# ============================================================================
def test_inv33_identity():
    """Test inv33 with identity matrix."""
    Id = jnp.eye(3)
    I_inv = inv33(Id)
    
    npt.assert_allclose(I_inv, Id, atol=1e-10)


def test_inv33_diagonal():
    """Test inv33 with diagonal matrix."""
    D = jnp.diag(jnp.array([2.0, 3.0, 4.0]))
    D_inv = inv33(D)
    D_inv_expected = jnp.diag(jnp.array([0.5, 1.0/3.0, 0.25]))
    
    npt.assert_allclose(D_inv, D_inv_expected, atol=1e-10)


def test_inv33_general():
    """Test inv33 with general 3x3 matrix."""
    M = jnp.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 10.0]
    ])
    M_inv = inv33(M)
    M_inv_expected = jnp.linalg.inv(M)
    
    npt.assert_allclose(M_inv, M_inv_expected, atol=1e-10)


def test_inv33_singular():
    """Test inv33 with singular matrix (should return zero matrix)."""
    # Create a singular matrix (rank 2)
    M = jnp.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0]
    ])
    M_inv = inv33(M)
    
    npt.assert_allclose(M_inv, jnp.zeros((3, 3)), atol=1e-10)


def test_inv33_inverse_property():
    """Test that inv33 produces actual inverse (M @ M_inv = I)."""
    M = jnp.array([
        [1.0, 0.5, 0.2],
        [0.3, 2.0, 0.1],
        [0.4, 0.6, 3.0]
    ])
    M_inv = inv33(M)
    product = M @ M_inv
    
    npt.assert_allclose(product, jnp.eye(3), atol=1e-8)


# ============================================================================
# Tests for div
# ============================================================================
def test_div_constant_field():
    """Test div with constant vector field (should be zero)."""
    def F(x):
        return jnp.array([1.0, 2.0, 3.0])
    
    div_F = div(F)
    
    x = jnp.array([0.5, 0.5, 0.5])
    result = div_F(x)
    
    npt.assert_allclose(result, 0.0, atol=1e-10)


def test_div_linear_field():
    """Test div with linear vector field."""
    def F(x):
        # F = (x, y, z) has divergence 3
        return x
    
    div_F = div(F)
    
    x = jnp.array([0.5, 0.5, 0.5])
    result = div_F(x)
    
    npt.assert_allclose(result, 3.0, atol=1e-10)


def test_div_quadratic_field():
    """Test div with quadratic vector field."""
    def F(x):
        # F = (x^2, y^2, z^2) has divergence 2*(x + y + z)
        return jnp.array([x[0]**2, x[1]**2, x[2]**2])
    
    div_F = div(F)
    
    x = jnp.array([1.0, 2.0, 3.0])
    result = div_F(x)
    expected = 2.0 * (x[0] + x[1] + x[2])
    
    npt.assert_allclose(result, expected, atol=1e-10)


# ============================================================================
# Tests for curl
# ============================================================================
def test_curl_constant_field():
    """Test curl with constant vector field (should be zero)."""
    def F(x):
        return jnp.array([1.0, 2.0, 3.0])
    
    curl_F = curl(F)
    
    x = jnp.array([0.5, 0.5, 0.5])
    result = curl_F(x)
    
    npt.assert_allclose(result, jnp.zeros(3), atol=1e-10)


def test_curl_linear_field():
    """Test curl with linear vector field."""
    def F(x):
        # F = (y, z, x) has curl = (-1, -1, -1)
        # curl = (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy)
        # = (0 - 1, 0 - 1, 0 - 1) = (-1, -1, -1)
        return jnp.array([x[1], x[2], x[0]])
    
    curl_F = curl(F)
    
    x = jnp.array([0.5, 0.5, 0.5])
    result = curl_F(x)
    expected = jnp.array([-1.0, -1.0, -1.0])
    
    npt.assert_allclose(result, expected, atol=1e-10)


def test_curl_gradient_field():
    """Test curl of gradient field (should be zero)."""
    def phi(x):
        """Scalar potential."""
        return x[0]**2 + x[1]**2 + x[2]**2
    
    def F(x):
        """Gradient of phi."""
        return jnp.array([2*x[0], 2*x[1], 2*x[2]])
    
    curl_F = curl(F)
    
    x = jnp.array([0.5, 0.5, 0.5])
    result = curl_F(x)
    
    npt.assert_allclose(result, jnp.zeros(3), atol=1e-10)


# ============================================================================
# Tests for grad
# ============================================================================
def test_grad_constant():
    """Test grad with constant function (should be zero)."""
    def F(x):
        return jnp.ones(1) * 5.0
    
    grad_F = grad(F)
    
    x = jnp.array([0.5, 0.5, 0.5])
    result = grad_F(x)
    
    npt.assert_allclose(result, jnp.zeros(3), atol=1e-10)


def test_grad_linear():
    """Test grad with linear function."""
    def F(x):
        return jnp.ones(1) * (2*x[0] + 3*x[1] + 4*x[2])
    
    grad_F = grad(F)
    
    x = jnp.array([0.5, 0.5, 0.5])
    result = grad_F(x)
    expected = jnp.array([2.0, 3.0, 4.0])
    
    npt.assert_allclose(result, expected, atol=1e-10)


def test_grad_quadratic():
    """Test grad with quadratic function."""
    def F(x):
        return jnp.ones(1) * (x[0]**2 + x[1]**2 + x[2]**2)
    
    grad_F = grad(F)
    
    x = jnp.array([1.0, 2.0, 3.0])
    result = grad_F(x)
    expected = jnp.array([2*x[0], 2*x[1], 2*x[2]])
    
    npt.assert_allclose(result, expected, atol=1e-10)


# ============================================================================
# Tests for l2_product
# ============================================================================
def test_l2_product_constant_functions(seq):
    """Test l2_product with constant functions."""
    seq.evaluate_1d()
    
    def f(x):
        return jnp.ones(3) * 2.0
    
    def g(x):
        return jnp.ones(3) * 3.0
    
    def F(x):
        return x
    
    result = l2_product(f, g, seq.Q, F)
    
    # For constant functions, result should be proportional to domain volume
    assert jnp.isfinite(result), "Result should be finite"
    assert result > 0, "Result should be positive"


def test_l2_product_orthogonal_functions(seq):
    """Test l2_product with orthogonal functions."""
    seq.evaluate_1d()
    
    def f(x):
        return jnp.array([jnp.sin(2*jnp.pi*x[0]), 0.0, 0.0])
    
    def g(x):
        return jnp.array([jnp.cos(2*jnp.pi*x[0]), 0.0, 0.0])
    
    def F(x):
        return x
    
    result = l2_product(f, g, seq.Q, F)
    
    # For orthogonal sin and cos, result should be approximately zero
    assert abs(result) < 1.0, "Orthogonal functions should have small inner product"


# ============================================================================
# Tests for assemble
# ============================================================================
def test_assemble_simple():
    """Test assemble with simple getters."""
    n_q = 5
    d = 3
    n1 = 4
    n2 = 4
    
    # Create simple getters (must return scalars, not arrays)
    def getter_1(a, j, k):
        return jnp.array(a + j + k, dtype=float)
    
    def getter_2(b, j, k):
        return jnp.array(b + j + k, dtype=float)
    
    # Create weight tensor
    W = jnp.ones((n_q, d, d))
    
    M = assemble(getter_1, getter_2, W, n1, n2)
    
    assert M.shape == (n1, n2), f"Matrix should have shape ({n1}, {n2})"
    assert jnp.all(jnp.isfinite(M)), "Matrix should have finite values"


def test_assemble_symmetric():
    """Test that assemble produces symmetric matrix for symmetric getters."""
    n_q = 5
    d = 3
    n = 4
    
    # Create symmetric getter (must return scalars)
    def getter_1(a, j, k):
        return jnp.array(a + j + k, dtype=float)
    
    getter_2 = getter_1  # Same getter
    
    W = jnp.eye(d)[None, :, :] * jnp.ones((n_q, 1, 1))
    
    M = assemble(getter_1, getter_2, W, n, n)
    
    # Check symmetry
    npt.assert_allclose(M, M.T, atol=1e-10)


# ============================================================================
# Tests for evaluate_at_xq
# ============================================================================
def test_evaluate_at_xq_constant():
    """Test evaluate_at_xq with constant DOFs."""
    n_q = 5
    d = 3
    n = 4
    
    def getter(i, j, k):
        # Return scalar, not array
        return jnp.array(i + j + k, dtype=float)
    
    dofs = jnp.ones(n) * 2.0
    
    result = evaluate_at_xq(getter, dofs, n_q, d)
    
    # Result shape is (n_q, d) but getter returns scalar, so it's broadcast
    assert result.shape == (n_q, d), f"Result should have shape ({n_q}, {d})"
    assert jnp.all(jnp.isfinite(result)), "Result should have finite values"


def test_evaluate_at_xq_linear():
    """Test evaluate_at_xq with linear DOFs."""
    n_q = 5
    d = 3
    n = 4
    
    def getter(i, j, k):
        # Return scalar, not array, and use JAX-compatible conversion
        return jnp.array(i, dtype=float)
    
    dofs = jnp.arange(n, dtype=float)
    
    result = evaluate_at_xq(getter, dofs, n_q, d)
    
    assert result.shape == (n_q, d), f"Result should have shape ({n_q}, {d})"
    assert jnp.all(jnp.isfinite(result)), "Result should have finite values"


# ============================================================================
# Tests for integrate_against
# ============================================================================
def test_integrate_against_constant():
    """Test integrate_against with constant function values."""
    n_q = 5
    d = 3
    n = 4
    
    def getter(i, j, k):
        # Return scalar, use JAX-compatible conversion
        return jnp.array(i, dtype=float)
    
    w_jk = jnp.ones((n_q, d)) * 2.0
    
    result = integrate_against(getter, w_jk, n)
    
    assert result.shape == (n,), f"Result should have shape ({n},)"
    assert jnp.all(jnp.isfinite(result)), "Result should have finite values"


def test_integrate_against_linear():
    """Test integrate_against with linear function values."""
    n_q = 5
    d = 3
    n = 4
    
    def getter(i, j, k):
        # Return scalar, use JAX-compatible conversion
        return jnp.array(i * j, dtype=float)
    
    w_jk = jnp.ones((n_q, d))
    
    result = integrate_against(getter, w_jk, n)
    
    assert result.shape == (n,), f"Result should have shape ({n},)"
    assert jnp.all(jnp.isfinite(result)), "Result should have finite values"


# ============================================================================
# Tests for append_to_trace_dict
# ============================================================================
def test_append_to_trace_dict_basic():
    """Test append_to_trace_dict with basic values."""
    trace_dict = copy.deepcopy(default_trace_dict)
    trace_dict["start_time"] = 0.0
    
    result = append_to_trace_dict(
        trace_dict, i=1, f=0.5, E=1.0, H=2.0, dvg=0.01,
        v=0.1, p_i=5, e=1e-10, dt=1e-6, end_time=1.0, B=None
    )
    
    assert result["iterations"][-1] == 1
    assert result["force_trace"][-1] == 0.5
    assert result["energy_trace"][-1] == 1.0
    assert result["helicity_trace"][-1] == 2.0
    assert result["divergence_trace"][-1] == 0.01
    assert result["velocity_trace"][-1] == 0.1
    assert result["picard_iterations"][-1] == 5
    assert result["picard_errors"][-1] == 1e-10
    assert result["timesteps"][-1] == 1e-6
    assert result["wall_time_trace"][-1] == 1.0
    assert len(result["B_fields"]) == 0, "B_fields should be empty when B=None"


def test_append_to_trace_dict_with_B():
    """Test append_to_trace_dict with B field."""
    trace_dict = copy.deepcopy(default_trace_dict)
    trace_dict["start_time"] = 0.0
    
    B = jnp.array([1.0, 2.0, 3.0])
    
    result = append_to_trace_dict(
        trace_dict, i=1, f=0.5, E=1.0, H=2.0, dvg=0.01,
        v=0.1, p_i=5, e=1e-10, dt=1e-6, end_time=1.0, B=B
    )
    
    assert len(result["B_fields"]) == 1, "B_fields should contain one element"
    npt.assert_allclose(result["B_fields"][0], B)


def test_append_to_trace_dict_multiple():
    """Test append_to_trace_dict with multiple appends."""
    # Create a fresh trace dict with empty lists
    trace_dict = {
        "iterations": [],
        "force_trace": [],
        "energy_trace": [],
        "helicity_trace": [],
        "divergence_trace": [],
        "picard_iterations": [],
        "picard_errors": [],
        "timesteps": [],
        "velocity_trace": [],
        "wall_time_trace": [],
        "B_fields": [],
        "p_fields": [],
        "start_time": 0.0,
        "end_time": None,
    }
    
    # Start with empty lists
    for i in range(3):
        trace_dict = append_to_trace_dict(
            trace_dict, i=i, f=0.5*i, E=1.0*i, H=2.0*i, dvg=0.01*i,
            v=0.1*i, p_i=5, e=1e-10, dt=1e-6, end_time=float(i), B=None
        )
    
    assert len(trace_dict["iterations"]) == 3
    assert trace_dict["iterations"] == [0, 1, 2]
    npt.assert_allclose(trace_dict["force_trace"], [0.0, 0.5, 1.0])


# ============================================================================
# Tests for save_trace_dict_to_hdf5
# ============================================================================
def test_save_trace_dict_to_hdf5_basic(seq):
    """Test save_trace_dict_to_hdf5 with basic trace dict."""
    seq.evaluate_1d()
    seq.assemble_all()
    seq.assemble_leray_projection()
    
    diagnostics = MRXDiagnostics(seq, force_free=False)
    
    trace_dict = copy.deepcopy(default_trace_dict)
    trace_dict["start_time"] = 0.0
    trace_dict["end_time"] = 10.0
    trace_dict["setup_done_time"] = 1.0
    trace_dict["iterations"] = [0, 1, 2]
    trace_dict["force_trace"] = [1.0, 0.5, 0.25]
    trace_dict["energy_trace"] = [1.0, 0.9, 0.8]
    trace_dict["helicity_trace"] = [1.0, 1.0, 1.0]
    trace_dict["divergence_trace"] = [0.01, 0.005, 0.002]
    trace_dict["velocity_trace"] = [1.0, 0.8, 0.6]
    trace_dict["picard_iterations"] = [5, 4, 3]
    trace_dict["picard_errors"] = [1e-10, 1e-11, 1e-12]
    trace_dict["timesteps"] = [1e-6, 1e-5, 1e-4]
    trace_dict["wall_time_trace"] = [0.0, 1.0, 2.0]
    trace_dict["B_final"] = jnp.ones(seq.E2.shape[0])
    trace_dict["p_final"] = jnp.ones(seq.E0.shape[0])
    
    CONFIG = DEFAULT_CONFIG.copy()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "test_trace")
        save_trace_dict_to_hdf5(trace_dict, diagnostics, filename, CONFIG)
        
        # Verify file was created
        h5_file = filename + ".h5"
        assert os.path.exists(h5_file), "HDF5 file should be created"
        
        # Verify contents
        with h5py.File(h5_file, "r") as f:
            assert "iterations" in f
            assert "force_trace" in f
            assert "B_final" in f
            assert "p_final" in f
            assert "config" in f
            
            # Check data
            npt.assert_allclose(f["iterations"][:], [0, 1, 2])
            npt.assert_allclose(f["force_trace"][:], [1.0, 0.5, 0.25])
            
            # Check config
            assert f["config"].attrs["maxit"] == CONFIG["maxit"]


def test_save_trace_dict_to_hdf5_with_save_B(seq):
    """Test save_trace_dict_to_hdf5 with save_B=True."""
    seq.evaluate_1d()
    seq.assemble_all()
    seq.assemble_leray_projection()
    
    diagnostics = MRXDiagnostics(seq, force_free=False)
    
    trace_dict = copy.deepcopy(default_trace_dict)
    trace_dict["start_time"] = 0.0
    trace_dict["end_time"] = 10.0
    trace_dict["setup_done_time"] = 1.0
    trace_dict["iterations"] = [0, 1]
    trace_dict["force_trace"] = [1.0, 0.5]
    trace_dict["energy_trace"] = [1.0, 0.9]
    trace_dict["helicity_trace"] = [1.0, 1.0]
    trace_dict["divergence_trace"] = [0.01, 0.005]
    trace_dict["velocity_trace"] = [1.0, 0.8]
    trace_dict["picard_iterations"] = [5, 4]
    trace_dict["picard_errors"] = [1e-10, 1e-11]
    trace_dict["timesteps"] = [1e-6, 1e-5]
    trace_dict["wall_time_trace"] = [0.0, 1.0]
    trace_dict["B_fields"] = [jnp.ones(seq.E2.shape[0]), jnp.ones(seq.E2.shape[0]) * 2]
    trace_dict["p_fields"] = [jnp.ones(seq.E0.shape[0]), jnp.ones(seq.E0.shape[0]) * 2]
    trace_dict["B_final"] = jnp.ones(seq.E2.shape[0])
    trace_dict["p_final"] = jnp.ones(seq.E0.shape[0])
    
    CONFIG = DEFAULT_CONFIG.copy()
    CONFIG["save_B"] = True
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "test_trace")
        save_trace_dict_to_hdf5(trace_dict, diagnostics, filename, CONFIG)
        
        h5_file = filename + ".h5"
        with h5py.File(h5_file, "r") as f:
            assert "B_fields" in f
            assert "p_fields" in f
            assert f["B_fields"].shape[0] == 2
            assert f["p_fields"].shape[0] == 2


def test_save_trace_dict_to_hdf5_skips_callable(seq):
    """Test that save_trace_dict_to_hdf5 skips callable values in CONFIG."""
    seq.evaluate_1d()
    seq.assemble_all()
    seq.assemble_leray_projection()
    
    diagnostics = MRXDiagnostics(seq, force_free=False)
    
    trace_dict = copy.deepcopy(default_trace_dict)
    trace_dict["start_time"] = 0.0
    trace_dict["end_time"] = 10.0
    trace_dict["setup_done_time"] = 1.0
    trace_dict["iterations"] = [0]
    trace_dict["force_trace"] = [1.0]
    trace_dict["energy_trace"] = [1.0]
    trace_dict["helicity_trace"] = [1.0]
    trace_dict["divergence_trace"] = [0.01]
    trace_dict["velocity_trace"] = [1.0]
    trace_dict["picard_iterations"] = [5]
    trace_dict["picard_errors"] = [1e-10]
    trace_dict["timesteps"] = [1e-6]
    trace_dict["wall_time_trace"] = [0.0]
    trace_dict["B_final"] = jnp.ones(seq.E2.shape[0])
    trace_dict["p_final"] = jnp.ones(seq.E0.shape[0])
    
    CONFIG = DEFAULT_CONFIG.copy()
    CONFIG["test_function"] = lambda x: x  # Add a callable
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "test_trace")
        save_trace_dict_to_hdf5(trace_dict, diagnostics, filename, CONFIG)
        
        h5_file = filename + ".h5"
        with h5py.File(h5_file, "r") as f:
            # test_function should not be in config
            assert "test_function" not in f["config"].attrs


# ============================================================================
# Tests for update_config
# ============================================================================
def test_update_config_basic():
    """Test update_config with basic parameters."""
    CONFIG = DEFAULT_CONFIG.copy()
    params = {"n_r": 16, "n_theta": 16, "dt": 1e-5}
    
    result = update_config(params, CONFIG)
    
    assert result["n_r"] == 16
    assert result["n_theta"] == 16
    assert result["dt"] == 1e-5
    assert result is CONFIG, "Should modify CONFIG in place"


def test_update_config_with_device():
    """Test update_config with device preset."""
    CONFIG = DEFAULT_CONFIG.copy()
    params = {"device": "ITER"}
    
    result = update_config(params, CONFIG)
    
    assert result["eps"] == DEVICE_PRESETS["ITER"]["eps"]
    assert result["kappa"] == DEVICE_PRESETS["ITER"]["kappa"]
    assert result["delta"] == DEVICE_PRESETS["ITER"]["delta"]


def test_update_config_device_override():
    """Test that user parameters override device presets."""
    CONFIG = DEFAULT_CONFIG.copy()
    params = {"device": "ITER", "eps": 0.5}
    
    result = update_config(params, CONFIG)
    
    assert result["eps"] == 0.5, "User parameter should override device preset"
    assert result["kappa"] == DEVICE_PRESETS["ITER"]["kappa"], "Other device params should remain"


def test_update_config_unknown_device():
    """Test update_config with unknown device."""
    CONFIG = DEFAULT_CONFIG.copy()
    params = {"device": "UNKNOWN"}
    
    # Should not raise error, just print warning
    result = update_config(params, CONFIG)
    
    assert result is CONFIG


def test_update_config_unknown_parameter():
    """Test update_config with unknown parameter."""
    CONFIG = DEFAULT_CONFIG.copy()
    params = {"unknown_param": 123}
    
    # Should not raise error, just print warning
    result = update_config(params, CONFIG)
    
    assert "unknown_param" not in result


# ============================================================================
# Tests for constants
# ============================================================================
def test_default_trace_dict_structure():
    """Test that default_trace_dict has correct structure."""
    assert isinstance(default_trace_dict, dict)
    assert "iterations" in default_trace_dict
    assert "force_trace" in default_trace_dict
    assert "energy_trace" in default_trace_dict
    assert "helicity_trace" in default_trace_dict
    assert "divergence_trace" in default_trace_dict
    assert "velocity_trace" in default_trace_dict
    assert "picard_iterations" in default_trace_dict
    assert "picard_errors" in default_trace_dict
    assert "timesteps" in default_trace_dict
    assert "wall_time_trace" in default_trace_dict
    assert "B_fields" in default_trace_dict
    assert "p_fields" in default_trace_dict
    assert "start_time" in default_trace_dict
    assert "end_time" in default_trace_dict
    
    # Check that all values are lists or None
    for key, value in default_trace_dict.items():
        assert isinstance(value, (list, type(None))), f"{key} should be list or None"


def test_default_config_structure():
    """Test that DEFAULT_CONFIG has correct structure."""
    assert isinstance(DEFAULT_CONFIG, dict)
    
    # Check required keys
    required_keys = [
        "run_name", "boundary_type", "eps", "kappa", "q_star", "delta",
        "n_r", "n_theta", "n_zeta", "p_r", "p_theta", "p_zeta",
        "maxit", "dt", "force_tol", "solver_maxit", "solver_tol"
    ]
    
    for key in required_keys:
        assert key in DEFAULT_CONFIG, f"{key} should be in DEFAULT_CONFIG"


def test_device_presets_structure():
    """Test that DEVICE_PRESETS has correct structure."""
    assert isinstance(DEVICE_PRESETS, dict)
    
    # Check known devices
    assert "ITER" in DEVICE_PRESETS
    assert "NSTX" in DEVICE_PRESETS
    assert "SPHERO" in DEVICE_PRESETS
    
    # Check that each preset is a dict
    for device, preset in DEVICE_PRESETS.items():
        assert isinstance(preset, dict), f"{device} preset should be a dict"
        assert "type" in preset, f"{device} preset should have 'type' key"


# ============================================================================
# Tests for run_relaxation_loop
# ============================================================================
def test_run_relaxation_loop_basic(seq):
    """Test run_relaxation_loop with basic configuration."""
    seq.evaluate_1d()
    seq.assemble_all()
    seq.build_crossproduct_projections()
    seq.assemble_leray_projection()
    
    # Create initial magnetic field (harmonic component)
    B_harm = jnp.linalg.eigh(seq.M2 @ seq.dd2)[1][:, 0]
    B_hat = B_harm / norm_2(B_harm, seq)
    
    # Create CONFIG with small maxit for testing
    CONFIG = DEFAULT_CONFIG.copy()
    CONFIG["maxit"] = 5
    CONFIG["save_every"] = 2
    CONFIG["print_every"] = 10  # Won't print during test
    CONFIG["save_B"] = False
    CONFIG["force_tol"] = 1e-15  # Very low tolerance, won't converge early
    
    # Create trace dict (use deepcopy to avoid shared list references)
    import time
    trace_dict = copy.deepcopy(default_trace_dict)
    trace_dict["start_time"] = time.time()
    
    # Create diagnostics and state
    diagnostics = MRXDiagnostics(seq, CONFIG["force_free"])
    state = State(B_hat, B_hat, CONFIG["dt"], CONFIG["eta"], seq.M2, 0, 0, 0, 0)
    
    # Run relaxation loop
    run_relaxation_loop(CONFIG, trace_dict, state, diagnostics)
    
    # Check that trace_dict was updated
    assert len(trace_dict["iterations"]) > 0, "Should have at least initial iteration"
    assert "setup_done_time" in trace_dict, "Should have setup_done_time"
    assert trace_dict["setup_done_time"] > trace_dict["start_time"], "Setup time should be after start"
    
    # Check that we have saved iterations (at save_every intervals)
    saved_iterations = [i for i in trace_dict["iterations"] if i % CONFIG["save_every"] == 0 or i == CONFIG["maxit"]]
    assert len(saved_iterations) > 0, "Should have saved some iterations"


def test_run_relaxation_loop_with_save_B(seq):
    """Test run_relaxation_loop with save_B=True."""
    seq.evaluate_1d()
    seq.assemble_all()
    seq.build_crossproduct_projections()
    seq.assemble_leray_projection()
    
    B_harm = jnp.linalg.eigh(seq.M2 @ seq.dd2)[1][:, 0]
    B_hat = B_harm / norm_2(B_harm, seq)
    
    CONFIG = DEFAULT_CONFIG.copy()
    CONFIG["maxit"] = 5
    CONFIG["save_every"] = 2
    CONFIG["print_every"] = 10
    CONFIG["save_B"] = True
    CONFIG["force_tol"] = 1e-15
    
    import time
    trace_dict = copy.deepcopy(default_trace_dict)
    trace_dict["start_time"] = time.time()
    
    diagnostics = MRXDiagnostics(seq, CONFIG["force_free"])
    state = State(B_hat, B_hat, CONFIG["dt"], CONFIG["eta"], seq.M2, 0, 0, 0, 0)
    
    # Store initial B shape for later comparison
    initial_B_shape = B_hat.shape
    
    run_relaxation_loop(CONFIG, trace_dict, state, diagnostics)
    
    # Check that B_fields were saved
    assert len(trace_dict["B_fields"]) > 0, "Should have saved B fields"
    
    # When save_B=True, B_fields should be saved at the same iterations as other trace data
    # trace_dict["iterations"] contains all saved iterations (initial 0, then at save_every intervals, and at maxit)
    # So B_fields count should match iterations count when save_B=True
    assert len(trace_dict["B_fields"]) == len(trace_dict["iterations"]), \
        (f"Number of B_fields ({len(trace_dict['B_fields'])}) should match "
         f"number of saved iterations ({len(trace_dict['iterations'])}). "
         f"Iterations: {trace_dict['iterations']}")
    
    # Each B_field should have the correct shape (use initial shape, not state.B_n which may have changed)
    for i, B in enumerate(trace_dict["B_fields"]):
        assert B.shape == initial_B_shape, \
            f"B_fields[{i}] should have shape {initial_B_shape}, got {B.shape}"


def test_run_relaxation_loop_early_convergence(seq):
    """Test run_relaxation_loop with early convergence (high force_tol)."""
    seq.evaluate_1d()
    seq.assemble_all()
    seq.build_crossproduct_projections()
    seq.assemble_leray_projection()
    
    # Use harmonic field which should have low force
    B_harm = jnp.linalg.eigh(seq.M2 @ seq.dd2)[1][:, 0]
    B_hat = B_harm / norm_2(B_harm, seq)
    
    CONFIG = DEFAULT_CONFIG.copy()
    CONFIG["maxit"] = 100  # High maxit
    CONFIG["save_every"] = 10
    CONFIG["print_every"] = 10
    CONFIG["save_B"] = False
    CONFIG["force_tol"] = 1e-3  # High tolerance - should converge early
    
    import time
    trace_dict = copy.deepcopy(default_trace_dict)
    trace_dict["start_time"] = time.time()
    
    diagnostics = MRXDiagnostics(seq, CONFIG["force_free"])
    state = State(B_hat, B_hat, CONFIG["dt"], CONFIG["eta"], seq.M2, 0, 0, 0, 0)
    
    run_relaxation_loop(CONFIG, trace_dict, state, diagnostics)
    
    # Check that we converged early (before maxit)
    final_iteration = trace_dict["iterations"][-1]
    assert final_iteration < CONFIG["maxit"], "Should converge before maxit"
    assert trace_dict["force_trace"][-1] < CONFIG["force_tol"], "Final force should be below tolerance"


def test_run_relaxation_loop_with_preconditioner(seq):
    """Test run_relaxation_loop with preconditioner enabled."""
    seq.evaluate_1d()
    seq.assemble_all()
    seq.build_crossproduct_projections()
    seq.assemble_leray_projection()
    
    B_harm = jnp.linalg.eigh(seq.M2 @ seq.dd2)[1][:, 0]
    B_hat = B_harm / norm_2(B_harm, seq)
    
    CONFIG = DEFAULT_CONFIG.copy()
    CONFIG["maxit"] = 5
    CONFIG["save_every"] = 5
    CONFIG["print_every"] = 10
    CONFIG["save_B"] = False
    CONFIG["precond"] = True
    CONFIG["precond_compute_every"] = 2
    CONFIG["force_tol"] = 1e-15
    
    import time
    trace_dict = copy.deepcopy(default_trace_dict)
    trace_dict["start_time"] = time.time()
    
    diagnostics = MRXDiagnostics(seq, CONFIG["force_free"])
    state = State(B_hat, B_hat, CONFIG["dt"], CONFIG["eta"], seq.M2, 0, 0, 0, 0)
    
    run_relaxation_loop(CONFIG, trace_dict, state, diagnostics)
    
    # Should complete without errors
    assert len(trace_dict["iterations"]) > 0, "Should have iterations"


def test_run_relaxation_loop_with_perturbation(seq):
    """Test run_relaxation_loop with perturbation applied."""
    seq.evaluate_1d()
    seq.assemble_all()
    seq.build_crossproduct_projections()
    seq.assemble_leray_projection()
    
    B_harm = jnp.linalg.eigh(seq.M2 @ seq.dd2)[1][:, 0]
    B_hat = B_harm / norm_2(B_harm, seq)
    
    # Create a simple perturbation function
    def dB_xyz(p):
        r, theta, zeta = p
        return jnp.array([r * 0.1, 0.0, 0.0])
    
    CONFIG = DEFAULT_CONFIG.copy()
    CONFIG["maxit"] = 10
    CONFIG["save_every"] = 5
    CONFIG["print_every"] = 10
    CONFIG["save_B"] = False
    CONFIG["pert_strength"] = 0.1
    CONFIG["apply_pert_after"] = 5
    CONFIG["dB_xyz"] = dB_xyz  # Add perturbation function
    CONFIG["force_tol"] = 1e-15
    
    import time
    trace_dict = copy.deepcopy(default_trace_dict)
    trace_dict["start_time"] = time.time()
    
    diagnostics = MRXDiagnostics(seq, CONFIG["force_free"])
    state = State(B_hat, B_hat, CONFIG["dt"], CONFIG["eta"], seq.M2, 0, 0, 0, 0)
    
    run_relaxation_loop(CONFIG, trace_dict, state, diagnostics)
    
    # Should complete without errors
    assert len(trace_dict["iterations"]) > 0, "Should have iterations"
    # Check that we ran at least past the perturbation point
    assert max(trace_dict["iterations"]) >= CONFIG["apply_pert_after"], \
        "Should have run past perturbation point"


def test_run_relaxation_loop_trace_dict_updates(seq):
    """Test that run_relaxation_loop properly updates trace_dict."""
    seq.evaluate_1d()
    seq.assemble_all()
    seq.build_crossproduct_projections()
    seq.assemble_leray_projection()
    
    B_harm = jnp.linalg.eigh(seq.M2 @ seq.dd2)[1][:, 0]
    B_hat = B_harm / norm_2(B_harm, seq)
    
    CONFIG = DEFAULT_CONFIG.copy()
    CONFIG["maxit"] = 10
    CONFIG["save_every"] = 3
    CONFIG["print_every"] = 10
    CONFIG["save_B"] = False
    CONFIG["force_tol"] = 1e-15
    
    import time
    trace_dict = copy.deepcopy(default_trace_dict)
    trace_dict["start_time"] = time.time()
    
    diagnostics = MRXDiagnostics(seq, CONFIG["force_free"])
    state = State(B_hat, B_hat, CONFIG["dt"], CONFIG["eta"], seq.M2, 0, 0, 0, 0)
    
    run_relaxation_loop(CONFIG, trace_dict, state, diagnostics)
    
    # Check all trace arrays have same length
    trace_lengths = {
        "iterations": len(trace_dict["iterations"]),
        "force_trace": len(trace_dict["force_trace"]),
        "energy_trace": len(trace_dict["energy_trace"]),
        "helicity_trace": len(trace_dict["helicity_trace"]),
        "divergence_trace": len(trace_dict["divergence_trace"]),
        "velocity_trace": len(trace_dict["velocity_trace"]),
        "picard_iterations": len(trace_dict["picard_iterations"]),
        "picard_errors": len(trace_dict["picard_errors"]),
        "timesteps": len(trace_dict["timesteps"]),
        "wall_time_trace": len(trace_dict["wall_time_trace"]),
    }
    
    # All should have the same length
    lengths = set(trace_lengths.values())
    assert len(lengths) == 1, f"All trace arrays should have same length, got {trace_lengths}"
    
    # Check that values are finite (convert lists to arrays first)
    assert jnp.all(jnp.isfinite(jnp.array(trace_dict["force_trace"]))), "Force trace should be finite"
    assert jnp.all(jnp.isfinite(jnp.array(trace_dict["energy_trace"]))), "Energy trace should be finite"
    assert jnp.all(jnp.isfinite(jnp.array(trace_dict["helicity_trace"]))), "Helicity trace should be finite"


def test_run_relaxation_loop_state_modification(seq):
    """Test that run_relaxation_loop modifies the state correctly."""
    seq.evaluate_1d()
    seq.assemble_all()
    seq.build_crossproduct_projections()
    seq.assemble_leray_projection()
    
    B_harm = jnp.linalg.eigh(seq.M2 @ seq.dd2)[1][:, 0]
    B_hat = B_harm / norm_2(B_harm, seq)
    
    CONFIG = DEFAULT_CONFIG.copy()
    CONFIG["maxit"] = 5
    CONFIG["save_every"] = 5
    CONFIG["print_every"] = 10
    CONFIG["save_B"] = False
    CONFIG["force_tol"] = 1e-15
    
    import time
    trace_dict = copy.deepcopy(default_trace_dict)
    trace_dict["start_time"] = time.time()
    
    diagnostics = MRXDiagnostics(seq, CONFIG["force_free"])
    initial_B = B_hat.copy()
    state = State(B_hat, B_hat, CONFIG["dt"], CONFIG["eta"], seq.M2, 0, 0, 0, 0)
    
    run_relaxation_loop(CONFIG, trace_dict, state, diagnostics)
    
    # State should be modified (B_n might have changed)
    # At minimum, the state should still be valid
    assert state.B_n.shape == initial_B.shape, "State B_n should have same shape"
    assert jnp.all(jnp.isfinite(state.B_n)), "State B_n should be finite"
    assert state.dt > 0, "Time step should be positive"

