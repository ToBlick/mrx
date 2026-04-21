# %%
# test_utils.py

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from mrx.utils import (curl, det33, diag_schur_complement, div, evaluate_at_xq,
                       extract_diag_vector, grad, integrate_against, inv33,
                       jacobian_determinant, l2_product, square_sparse)

jax.config.update("jax_enable_x64", True)



# ============================================================================
# jacobian_determinant
# ============================================================================
def test_jacobian_determinant_identity():
    det_f = jacobian_determinant(lambda x: x)
    npt.assert_allclose(det_f(jnp.array([0.5, 0.3, 0.1])), 1.0, atol=1e-10)


def test_jacobian_determinant_scaling():
    det_f = jacobian_determinant(lambda x: 2.0 * x)
    npt.assert_allclose(det_f(jnp.array([0.5, 0.3, 0.1])), 8.0, atol=1e-10)


def test_jacobian_determinant_rotation():
    # 90-degree rotation around z-axis has det = 1
    def rot(x):
        return jnp.array([-x[1], x[0], x[2]])
    det_f = jacobian_determinant(rot)
    npt.assert_allclose(det_f(jnp.array([1.0, 0.0, 0.5])), 1.0, atol=1e-10)


# ============================================================================
# det33
# ============================================================================
def test_det33_identity():
    npt.assert_allclose(det33(jnp.eye(3)), 1.0, atol=1e-12)


def test_det33_diagonal():
    D = jnp.diag(jnp.array([2.0, 3.0, 4.0]))
    npt.assert_allclose(det33(D), 24.0, atol=1e-12)


def test_det33_singular():
    M = jnp.array([[1.0, 2.0, 3.0],
                   [2.0, 4.0, 6.0],
                   [3.0, 6.0, 9.0]])
    npt.assert_allclose(det33(M), 0.0, atol=1e-12)


def test_det33_matches_linalg():
    M = jnp.array([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 10.0]])
    npt.assert_allclose(det33(M), jnp.linalg.det(M), rtol=1e-10)


# ============================================================================
# inv33
# ============================================================================
def test_inv33_identity():
    npt.assert_allclose(inv33(jnp.eye(3)), jnp.eye(3), atol=1e-12)


def test_inv33_diagonal():
    D = jnp.diag(jnp.array([2.0, 4.0, 5.0]))
    expected = jnp.diag(jnp.array([0.5, 0.25, 0.2]))
    npt.assert_allclose(inv33(D), expected, atol=1e-12)


def test_inv33_inverse_property():
    M = jnp.array([[1.0, 0.5, 0.2],
                   [0.3, 2.0, 0.1],
                   [0.4, 0.6, 3.0]])
    npt.assert_allclose(M @ inv33(M), jnp.eye(3), atol=1e-8)


def test_inv33_singular_returns_zeros():
    M = jnp.array([[1.0, 2.0, 3.0],
                   [2.0, 4.0, 6.0],
                   [3.0, 6.0, 9.0]])
    npt.assert_allclose(inv33(M), jnp.zeros((3, 3)), atol=1e-10)


def test_inv33_matches_linalg():
    M = jnp.array([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 10.0]])
    npt.assert_allclose(inv33(M), jnp.linalg.inv(M), atol=1e-10)


# ============================================================================
# div
# ============================================================================
def test_div_constant():
    div_f = div(lambda x: jnp.array([1.0, 2.0, 3.0]))
    result = div_f(jnp.array([0.5, 0.5, 0.5]))
    npt.assert_allclose(result, jnp.zeros(1), atol=1e-10)


def test_div_linear():
    # F = (x, y, z) -> div = 3
    div_f = div(lambda x: x)
    result = div_f(jnp.array([0.5, 0.5, 0.5]))
    npt.assert_allclose(result, jnp.ones(1) * 3.0, atol=1e-10)


def test_div_quadratic():
    # F = (x^2, y^2, z^2) -> div = 2(x+y+z)
    def F(x):
        return jnp.array([x[0]**2, x[1]**2, x[2]**2])
    x = jnp.array([1.0, 2.0, 3.0])
    npt.assert_allclose(div(F)(x), jnp.array([2.0 * (1 + 2 + 3)]), atol=1e-10)


# ============================================================================
# curl
# ============================================================================
def test_curl_constant():
    curl_f = curl(lambda x: jnp.array([1.0, 2.0, 3.0]))
    npt.assert_allclose(curl_f(jnp.array([0.5, 0.5, 0.5])), jnp.zeros(3), atol=1e-10)


def test_curl_linear():
    # F = (y, z, x) -> curl = (-1, -1, -1)
    curl_f = curl(lambda x: jnp.array([x[1], x[2], x[0]]))
    npt.assert_allclose(curl_f(jnp.array([0.5, 0.5, 0.5])),
                        jnp.array([-1.0, -1.0, -1.0]), atol=1e-10)


def test_curl_of_gradient_is_zero():
    # grad(x^2 + y^2 + z^2) = (2x, 2y, 2z), curl = 0
    curl_f = curl(lambda x: jnp.array([2*x[0], 2*x[1], 2*x[2]]))
    npt.assert_allclose(curl_f(jnp.array([1.0, 2.0, 3.0])), jnp.zeros(3), atol=1e-10)


# ============================================================================
# grad
# ============================================================================
def test_grad_constant():
    grad_f = grad(lambda x: jnp.ones(1) * 5.0)
    npt.assert_allclose(grad_f(jnp.array([0.5, 0.5, 0.5])), jnp.zeros(3), atol=1e-10)


def test_grad_linear():
    # grad(2x + 3y + 4z) = (2, 3, 4)
    grad_f = grad(lambda x: jnp.ones(1) * (2*x[0] + 3*x[1] + 4*x[2]))
    npt.assert_allclose(grad_f(jnp.array([0.5, 0.5, 0.5])),
                        jnp.array([2.0, 3.0, 4.0]), atol=1e-10)


def test_grad_quadratic():
    # grad(x^2 + y^2 + z^2) = (2x, 2y, 2z)
    grad_f = grad(lambda x: jnp.ones(1) * (x[0]**2 + x[1]**2 + x[2]**2))
    x = jnp.array([1.0, 2.0, 3.0])
    npt.assert_allclose(grad_f(x), 2 * x, atol=1e-10)


# ============================================================================
# l2_product
# ============================================================================
class _SimpleQuad:
    """Minimal uniform quadrature on [0,1]^3."""
    def __init__(self, n=4):
        pts = jnp.linspace(0, 1, n)
        w1 = jnp.ones(n) / n
        grid = jnp.stack(jnp.meshgrid(pts, pts, pts, indexing="ij"), axis=-1)
        self.x = grid.reshape(-1, 3)
        w3 = jnp.einsum("i,j,k->ijk", w1, w1, w1).ravel()
        self.w = w3


def test_l2_product_constant_identity_map():
    Q = _SimpleQuad(n=5)
    # f·g = 2*3 = 6 everywhere, integral over unit cube = 6
    result = l2_product(lambda x: jnp.array([2.0, 0.0, 0.0]),
                        lambda x: jnp.array([3.0, 0.0, 0.0]), Q)
    npt.assert_allclose(result, 6.0, rtol=1e-5)


def test_l2_product_orthogonal():
    Q = _SimpleQuad(n=20)
    # ∫ sin(2πx)·cos(2πx) dx = 0 over [0,1]
    result = l2_product(lambda x: jnp.array([jnp.sin(2*jnp.pi*x[0]), 0.0, 0.0]),
                        lambda x: jnp.array([jnp.cos(2*jnp.pi*x[0]), 0.0, 0.0]), Q)
    assert abs(result) < 0.01


# ============================================================================
# evaluate_at_xq / integrate_against  (TP-structured forms)
# ============================================================================
def _const_comp_info(s1, s2, s3, nq_r, nq_t, nq_z, out_dim=0):
    R = jnp.ones((s1, nq_r))
    T = jnp.ones((s2, nq_t))
    Z = jnp.ones((s3, nq_z))
    return [(out_dim, R, T, Z)], [(s1, s2, s3)]


def test_evaluate_at_xq_ones():
    s1, s2, s3 = 1, 1, 1
    nq_r, nq_t, nq_z = 3, 4, 5
    comp_info, comp_shapes = _const_comp_info(s1, s2, s3, nq_r, nq_t, nq_z)
    dofs = jnp.ones(s1 * s2 * s3)
    result = evaluate_at_xq(dofs, comp_info, comp_shapes, (nq_t, nq_r, nq_z), d=1)
    assert result.shape == (nq_r * nq_t * nq_z, 1)
    npt.assert_allclose(result, jnp.ones((nq_r * nq_t * nq_z, 1)), atol=1e-12)


def test_evaluate_at_xq_zero_dofs():
    s1, s2, s3 = 2, 2, 2
    nq_r, nq_t, nq_z = 3, 3, 3
    comp_info, comp_shapes = _const_comp_info(s1, s2, s3, nq_r, nq_t, nq_z)
    result = evaluate_at_xq(jnp.zeros(s1*s2*s3), comp_info, comp_shapes,
                             (nq_t, nq_r, nq_z), d=1)
    npt.assert_allclose(result, 0.0, atol=1e-12)


def test_evaluate_integrate_adjoint():
    """integrate_against is the transpose of evaluate_at_xq."""
    s1, s2, s3 = 2, 3, 4
    nq_r, nq_t, nq_z = 3, 4, 5
    n_q = nq_r * nq_t * nq_z
    rng = np.random.default_rng(42)
    R = jnp.array(rng.standard_normal((s1, nq_r)))
    T = jnp.array(rng.standard_normal((s2, nq_t)))
    Z = jnp.array(rng.standard_normal((s3, nq_z)))
    comp_info = [(0, R, T, Z)]
    comp_shapes = [(s1, s2, s3)]
    quad_shape = (nq_t, nq_r, nq_z)

    dofs = jnp.array(rng.standard_normal(s1 * s2 * s3))
    f_jk = jnp.array(rng.standard_normal((n_q, 1)))

    E_dofs = evaluate_at_xq(dofs, comp_info, comp_shapes, quad_shape, d=1)
    ET_f = integrate_against(f_jk, comp_info, comp_shapes, quad_shape)

    lhs = float(jnp.sum(f_jk * E_dofs))
    rhs = float(jnp.dot(ET_f, dofs))
    npt.assert_allclose(lhs, rhs, rtol=1e-8)


# ============================================================================
# Sparse matrix utilities
# ============================================================================
def _make_bcsr(dense):
    return jsparse.BCSR.from_bcoo(jsparse.BCOO.fromdense(dense))


def _make_bcoo(dense):
    return jsparse.BCOO.fromdense(dense)


def test_extract_diag_vector_diagonal():
    D = jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0]))
    for make in [_make_bcsr, _make_bcoo]:
        result = extract_diag_vector(make(D))
        npt.assert_allclose(result, jnp.array([1.0, 2.0, 3.0, 4.0]), atol=1e-12)


def test_extract_diag_vector_off_diagonal():
    M = jnp.array([[1.0, 5.0, 0.0],
                   [0.0, 2.0, 6.0],
                   [0.0, 0.0, 3.0]])
    for make in [_make_bcsr, _make_bcoo]:
        result = extract_diag_vector(make(M))
        npt.assert_allclose(result, jnp.array([1.0, 2.0, 3.0]), atol=1e-12)


def test_square_sparse_values():
    M = jnp.array([[2.0, 0.0, 3.0],
                   [0.0, 4.0, 0.0],
                   [5.0, 0.0, 6.0]])
    expected = jnp.array([[4.0, 0.0, 9.0],
                           [0.0, 16.0, 0.0],
                           [25.0, 0.0, 36.0]])
    for make in [_make_bcsr, _make_bcoo]:
        npt.assert_allclose(square_sparse(make(M)).todense(), expected, atol=1e-12)


# ============================================================================
# diag_schur_complement
# ============================================================================
def test_diag_schur_complement_identity():
    # D = I, diag_inv = 1 -> diag(D diag_inv D^T) = 1
    n = 5
    result = diag_schur_complement(lambda v: v, jnp.ones(n), n)
    npt.assert_allclose(result, jnp.ones(n), atol=1e-10)


def test_diag_schur_complement_scaling():
    # D = I, diag_inv[i] = i+1 -> entry i = diag_inv[i]
    n = 4
    diag_inv = jnp.array([1.0, 2.0, 3.0, 4.0])
    result = diag_schur_complement(lambda v: v, diag_inv, n)
    npt.assert_allclose(result, diag_inv, atol=1e-10)


def test_diag_schur_complement_zero():
    n = 3
    result = diag_schur_complement(lambda v: jnp.zeros_like(v), jnp.ones(n), n)
    npt.assert_allclose(result, jnp.zeros(n), atol=1e-12)
