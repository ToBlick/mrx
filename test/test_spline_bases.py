# %%
# test_spline_bases.py
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from mrx.spline_bases import SplineBasis, TensorBasis, DerivativeSpline
from mrx.utils import is_running_in_github_actions

jax.config.update("jax_enable_x64", True)

if is_running_in_github_actions():
    n, p = 6, 3
else:
    n, p = 10, 3

# Helper fixture and parametrization
@pytest.fixture(params=["clamped", "periodic", "constant", "fourier"])
def basis(request):
    """Return a spline basis of given type."""
    return SplineBasis(n, p, request.param), request.param

def test_partition_of_unity(basis):
    """Test the partition of unity property of the spline basis."""
    spl, typ = basis
    if typ == "fourier" or typ == "constant":
        return
    xs = jnp.linspace(0.0, 1.0, 200)
    i = jnp.arange(spl.n)

    def sum_basis(x):
        vals = jax.vmap(lambda j: spl(x, j))(i)
        return jnp.sum(vals)

    sums = jax.vmap(sum_basis)(xs)
    npt.assert_allclose(
        sums, 1.0, atol=1e-12, err_msg=f"Partition of unity fails ({typ})"
    )

    # Initialize new spline basis with the knots from the previous basis
    spl_new = SplineBasis(spl.n, spl.p, spl.type, spl.T)
    
    def eval_basis_new(x):
        return jax.vmap(lambda j: spl_new(x, j))(i)
    
    def eval_basis_old(x):
        return jax.vmap(lambda j: spl(x, j))(i)
    
    vals_new = jax.vmap(eval_basis_new)(xs)  # (nq, n)
    vals_old = jax.vmap(eval_basis_old)(xs)  # (nq, n)
    
    npt.assert_allclose(
        vals_new, vals_old, atol=1e-12, err_msg=f"Spline basis fails ({typ})"
    )

def test_bad_spline_init(basis):
    """Test the initialization of a spline basis with bad parameters."""
    spl, typ = basis
    with pytest.raises(ValueError):
        SplineBasis(spl.n, spl.p, "bad", spl.T)
    with pytest.raises(ValueError):
        SplineBasis(spl.n, spl.n, "clamped")
    with pytest.raises(ValueError):
        SplineBasis(spl.n, spl.n * 2, "fourier")

def test_spline_getitem(basis):
    """Test the __getitem__ method of the spline basis."""
    spl, typ = basis
    xs = jnp.linspace(0.0, 1.0, 200)
    
    # Test that __getitem__ returns a callable
    for i in range(min(5, spl.n)):  # Test first few indices
        spline_func = spl[i]
        assert callable(spline_func), f"__getitem__ should return a callable ({typ})"
        
        # Test that spl[i](x) gives the same result as spl(x, i)
        vals_getitem = jax.vmap(spline_func)(xs)
        vals_direct = jax.vmap(lambda x: spl(x, i))(xs)
        
        npt.assert_allclose(
            vals_getitem, vals_direct, atol=1e-12,
            err_msg=f"__getitem__ evaluation fails for i={i} ({typ})"
        )

# Positivity
def test_basis_positivity(basis):
    """Test the positivity property of the spline basis."""
    spl, typ = basis
    xs = jnp.linspace(0.0, 1.0, 200)
    i = jnp.arange(spl.n)

    def eval_basis(x):
        return jax.vmap(lambda j: spl(x, j))(i)

    vals = jax.vmap(eval_basis)(xs)  # (nq, n)
    assert jnp.all(vals >= -1e-14), f"Negative basis value detected ({typ})"


# L2 projection of sin(2πx)
def test_spline_projection_sin(basis):
    """Test the L2 projection of the sine function using the spline basis.

    Parameters
    ----------
    basis : tuple
        A tuple containing the spline basis and the type of spline.
    """
    spl, typ = basis
    # Skip constant and fourier splines - they all evaluate to 1.0 regardless of index,
    # making the mass matrix singular and the projection meaningless
    if typ == "fourier" or typ == "constant":
        return
    n = spl.n
    i = jnp.arange(n)

    # Midpoint quadrature on [0,1]
    nq = 256
    xq = (jnp.arange(nq) + 0.5) / nq
    w = jnp.ones(nq) / nq

    # Build basis matrix N_qi
    def basis_at_x(x):
        return jax.vmap(lambda j: spl(x, j))(i)
    N_qi = jax.vmap(basis_at_x)(xq)  # (nq, n)

    # Mass matrix and RHS
    NW = N_qi * w[:, None]
    M = N_qi.T @ NW
    f_q = jnp.sin(2 * jnp.pi * xq)
    b = (f_q * w) @ N_qi

    coeffs = jnp.linalg.solve(M, b)

    # Evaluate projection
    xs = jnp.linspace(0.0, 1.0, 400)
    
    def spline_eval(x):
        Ni = jax.vmap(lambda j: spl(x, j))(i)
        return jnp.dot(coeffs, Ni)

    f_approx = jax.vmap(spline_eval)(xs)
    f_true = jnp.sin(2 * jnp.pi * xs)

    err = jnp.max(jnp.abs(f_true - f_approx))
    print(err)
    npt.assert_allclose(
        err, 0.0, atol=0.01,
        err_msg=f"L_inf error too large ({typ}): {err}"
    )


# Tests for TensorBasis
@pytest.fixture
def tensor_basis():
    """Create a TensorBasis for testing."""
    bases = [
        SplineBasis(5, 2, "clamped"),
        SplineBasis(4, 2, "periodic"),
        SplineBasis(6, 2, "clamped")
    ]
    return TensorBasis(bases)

def test_tensor_basis_init():
    """Test TensorBasis initialization."""
    bases = [
        SplineBasis(5, 2, "clamped"),
        SplineBasis(4, 2, "periodic"),
        SplineBasis(6, 2, "clamped")
    ]
    tb = TensorBasis(bases)
    
    assert tb.n == 5 * 4 * 6, "Total number of basis functions should be product of dimensions"
    assert jnp.all(tb.shape == jnp.array([5, 4, 6])), "Shape should match individual basis sizes"
    assert len(tb.bases) == 3, "Should have exactly 3 bases"
    
    # Test error case: wrong number of bases
    with pytest.raises(ValueError, match="exactly 3 bases"):
        TensorBasis([SplineBasis(5, 2, "clamped")])
    with pytest.raises(ValueError, match="exactly 3 bases"):
        TensorBasis([
            SplineBasis(5, 2, "clamped"),
            SplineBasis(4, 2, "periodic"),
            SplineBasis(6, 2, "clamped"),
            SplineBasis(3, 2, "clamped")
        ])

def test_tensor_basis_evaluate(tensor_basis):
    """Test TensorBasis evaluate method."""
    tb = tensor_basis
    
    # Test evaluation at a point
    x = jnp.array([0.5, 0.5, 0.5])
    
    # Test first few indices
    for i in range(min(5, tb.n)):
        val = tb.evaluate(x, i)
        # Get the multi-index
        ijk = jnp.unravel_index(i, tb.shape)
        # Compute manually as product
        val_manual = (tb.bases[0](x[0], ijk[0]) * 
                     tb.bases[1](x[1], ijk[1]) * 
                     tb.bases[2](x[2], ijk[2]))
        npt.assert_allclose(
            val, val_manual, atol=1e-12,
            err_msg=f"TensorBasis evaluation fails for i={i}"
        )
    
    # Test error case: wrong dimension
    with pytest.raises(ValueError, match="Input point dimension"):
        tb.evaluate(jnp.array([0.5, 0.5]), 0)

def test_tensor_basis_call(tensor_basis):
    """Test TensorBasis __call__ method."""
    tb = tensor_basis
    x = jnp.array([0.5, 0.5, 0.5])
    
    # Test that __call__ matches evaluate
    for i in range(min(5, tb.n)):
        val_call = tb(x, i)
        val_eval = tb.evaluate(x, i)
        npt.assert_allclose(
            val_call, val_eval, atol=1e-12,
            err_msg=f"TensorBasis __call__ fails for i={i}"
        )

def test_tensor_basis_getitem(tensor_basis):
    """Test TensorBasis __getitem__ method."""
    tb = tensor_basis
    x = jnp.array([0.5, 0.5, 0.5])
    
    # Test that __getitem__ returns a callable
    for i in range(min(5, tb.n)):
        basis_func = tb[i]
        assert callable(basis_func), f"__getitem__ should return a callable for i={i}"
        
        # Test that tb[i](x) gives the same result as tb(x, i)
        val_getitem = basis_func(x)
        val_direct = tb(x, i)
        npt.assert_allclose(
            val_getitem, val_direct, atol=1e-12,
            err_msg=f"TensorBasis __getitem__ evaluation fails for i={i}"
        )


# Tests for DerivativeSpline
@pytest.fixture(params=["clamped", "periodic", "constant"])
def derivative_spline(request):
    """Create a DerivativeSpline for testing."""
    s = SplineBasis(n, p, request.param)
    return DerivativeSpline(s), request.param

def test_derivative_spline_init(derivative_spline):
    """Test DerivativeSpline initialization."""
    ds, typ = derivative_spline
    s = SplineBasis(n, p, typ)
    
    # Check that n is correct
    if typ == "clamped":
        assert ds.n == s.n - 1, "Clamped derivative should have n-1 basis functions"
    else:
        assert ds.n == s.n, "Periodic/constant derivative should have n basis functions"
    
    # Check that p is correct
    if typ == "constant":
        assert ds.p == s.p, "Constant derivative should have same degree"
    else:
        assert ds.p == s.p - 1, "Non-constant derivative should have degree p-1"
    
    assert ds.type == typ, "Type should match original spline"

def test_derivative_spline_evaluate(derivative_spline):
    """Test DerivativeSpline evaluate method."""
    ds, typ = derivative_spline
    xs = jnp.linspace(0.0, 1.0, 50)
    
    # Test evaluation for first few indices
    for i in range(min(5, ds.n)):
        # Test that evaluate and __call__ match
        vals_eval = jax.vmap(lambda x: ds.evaluate(x, i))(xs)
        vals_call = jax.vmap(lambda x: ds(x, i))(xs)
        
        npt.assert_allclose(
            vals_eval, vals_call, atol=1e-12,
            err_msg=f"DerivativeSpline evaluate/__call__ mismatch for i={i} ({typ})"
        )
        
        # For constant splines, derivative should be 1.0
        if typ == "constant":
            npt.assert_allclose(
                vals_eval, 1.0, atol=1e-12,
                err_msg=f"Constant derivative should be 1.0 ({typ})"
            )

def test_derivative_spline_getitem(derivative_spline):
    """Test DerivativeSpline __getitem__ method."""
    ds, typ = derivative_spline
    xs = jnp.linspace(0.0, 1.0, 50)
    
    # Test that __getitem__ returns a callable
    for i in range(min(5, ds.n)):
        deriv_func = ds[i]
        assert callable(deriv_func), f"__getitem__ should return a callable for i={i} ({typ})"
        
        # Test that ds[i](x) gives the same result as ds(x, i)
        vals_getitem = jax.vmap(deriv_func)(xs)
        vals_direct = jax.vmap(lambda x: ds(x, i))(xs)
        
        npt.assert_allclose(
            vals_getitem, vals_direct, atol=1e-12,
            err_msg=f"DerivativeSpline __getitem__ evaluation fails for i={i} ({typ})"
        )

def test_derivative_spline_properties(derivative_spline):
    """Test DerivativeSpline properties for different spline types."""
    ds, typ = derivative_spline
    
    if typ == "constant":
        # Constant spline derivative should be 1.0 everywhere
        xs = jnp.linspace(0.0, 1.0, 20)
        for i in range(min(3, ds.n)):
            vals = jax.vmap(lambda x: ds(x, i))(xs)
            npt.assert_allclose(
                vals, 1.0, atol=1e-12,
                err_msg="Constant derivative should be 1.0 everywhere"
            )
