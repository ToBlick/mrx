from typing import Any, Callable

import jax
import jax.numpy as jnp

__all__ = ['jacobian_determinant', 'inv33',
           'div', 'curl', 'grad', 'l2_product']


def jacobian_determinant(f: Callable[[jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compute the determinant of the Jacobian matrix for a given function.

    Args:
        f: Function mapping from R^n to R^n for which to compute the Jacobian determinant

    Returns:
        Function that computes the Jacobian determinant at a given point
    """
    return lambda x: jnp.linalg.det(jax.jacfwd(f)(x))


def inv33(mat: jnp.ndarray) -> jnp.ndarray:
    """Compute the inverse of a 3x3 matrix using explicit formula.

    This function computes the inverse using the adjugate matrix formula,
    which is more efficient than general matrix inversion for 3x3 matrices.

    Args:
        mat: 3x3 matrix to invert

    Returns:
        The inverse of the input matrix
    """
    m1, m2, m3 = mat[0]
    m4, m5, m6 = mat[1]
    m7, m8, m9 = mat[2]
    det = m1 * (m5 * m9 - m6 * m8) + m4 * \
        (m8 * m3 - m2 * m9) + m7 * (m2 * m6 - m3 * m5)
    # Return zero matrix if determinant is zero
    return jnp.where(
        jnp.abs(det) < 1e-10,  # Careful with this tolerance
        jnp.zeros((3, 3)),
        jnp.array([
            [m5 * m9 - m6 * m8, m3 * m8 - m2 * m9, m2 * m6 - m3 * m5],
            [m6 * m7 - m4 * m9, m1 * m9 - m3 * m7, m3 * m4 - m1 * m6],
            [m4 * m8 - m5 * m7, m2 * m7 - m1 * m8, m1 * m5 - m2 * m4],
        ]) / det
    )


def div(F: Callable[[jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compute the divergence of a vector field.

    Args:
        F: Vector field function for which to compute the divergence

    Returns:
        Function that computes the divergence at a given point
    """
    def div_F(x: jnp.ndarray) -> jnp.ndarray:
        DF = jax.jacfwd(F)(x)
        return jnp.trace(DF) * jnp.ones(1)
    return div_F


def curl(F: Callable[[jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compute the curl of a vector field in 3D.

    Args:
        F: Vector field function for which to compute the curl

    Returns:
        Function that computes the curl at a given point
    """
    def curl_F(x: jnp.ndarray) -> jnp.ndarray:
        DF = jax.jacfwd(F)(x)
        return jnp.array([DF[2, 1] - DF[1, 2],
                         DF[0, 2] - DF[2, 0],
                         DF[1, 0] - DF[0, 1]])
    return curl_F


def grad(F: Callable[[jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compute the gradient of a scalar field.

    Args:
        F: Scalar field function for which to compute the gradient

    Returns:
        Function that computes the gradient at a given point
    """
    def grad_F(x: jnp.ndarray) -> jnp.ndarray:
        DF = jax.jacfwd(F)(x)
        return jnp.ravel(DF)
    return grad_F


def l2_product(f: Callable[[jnp.ndarray], jnp.ndarray],
               g: Callable[[jnp.ndarray], jnp.ndarray],
               Q: Any,
               F: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x) -> jnp.ndarray:
    """Compute the L2 inner product of two functions over a domain.

    Computes the integral of f·g over the domain defined by the quadrature rule Q,
    with optional coordinate transformation F.

    Args:
        f: First function in the inner product
        g: Second function in the inner product
        Q: Quadrature rule object with attributes x (points) and w (weights)
        F: Optional coordinate transformation function (default is identity)

    Returns:
        The L2 inner product value
    """
    Jj = jax.vmap(jacobian_determinant(F))(Q.x)
    return jnp.einsum("ij,ij,i,i->", jax.vmap(f)(Q.x), jax.vmap(g)(Q.x), Jj, Q.w)


def assemble(
    getter_1,
    getter_2,
    W,
    n1,
    n2,
):
    """
    Assemble a matrix M[a, b] = Σ_{a,j,k} Λ1[a,j,i] * W[j,i,k] * Λ2[b,j,k]

    Parameters
    ----------
    getter_1 : callable
        Function (a, j, k) -> scalar. kth component of form a evaluated at quadrature point j.
    getter_2 : callable
        Function (b, j, k) -> scalar. kth component of form b evaluated at quadrature point j.
    W : jnp.ndarray, shape (n_q, 3, 3)
        Weight tensor combining metric, Jacobian, and quadrature weights.
        (For example: G_inv[q, ...] * J[q] * w[q])
    n1 : int
        Number of row basis functions.
    n2 : int
        Number of column basis functions.

    Returns
    -------
    M : jnp.ndarray, shape (n1, n2)
        The assembled matrix.
    """

    n_q, d, _ = W.shape

    get_A_jk = jax.vmap(
        jax.vmap(getter_1, in_axes=(None, None, 0)),  # over k (dimensions)
        # over j (quadrature points)
        in_axes=(None, 0, None)
    )

    get_B_jk = jax.vmap(
        jax.vmap(getter_2, in_axes=(None, None, 0)),
        in_axes=(None, 0, None)
    )

    def body_fun(carry, i):
        ΛA_i = get_A_jk(i, jnp.arange(n_q), jnp.arange(d))

        def compute_row(m):
            ΛB_m = get_B_jk(m, jnp.arange(n_q), jnp.arange(d))
            return jnp.einsum("jk,jkm,jm->", ΛA_i, W, ΛB_m)

        M_row = jax.vmap(compute_row)(jnp.arange(n2))
        return None, M_row

    _, M = jax.lax.scan(body_fun, None, jnp.arange(n1))
    return M
# %%


def evaluate_at_xq(getter, dofs, n_q, d):
    """
    Evaluate a finite element function at quadrature points.

    Parameters
    ----------
    getter : callable
        Function (i, j, k) -> scalar. kth component of form i evaluated at quadrature point j.
    dofs : jnp.ndarray, shape (m,)
        Degrees of freedom of the finite element function, already contracted with extraction matrices

    Returns
    -------
    f_h_jk : jnp.ndarray, shape (n_q, d)
        Function values at quadrature points.
    """
    # Evaluate the finite element function at quadrature points
    get_f_jk = jax.vmap(
        jax.vmap(getter, in_axes=(None, None, 0)),  # over k (dimensions)
        # over j (quadrature points)
        in_axes=(None, 0, None)
    )

    v_i = dofs  # shape (n_i,)

    def body_fun(carry, i):
        L_i = get_f_jk(i, jnp.arange(n_q), jnp.arange(d))  # shape (n_q, d)
        # broadcast scalar over last axis (dimesions)
        carry += L_i * v_i[i]
        return carry, None

    R_init = jnp.zeros_like(
        get_f_jk(0, jnp.arange(n_q), jnp.arange(d)))  # shape (n_q, d)
    R, _ = jax.lax.scan(body_fun, R_init, jnp.arange(v_i.shape[0]))
    return R


def integrate_against(getter, w_jk, n):
    """
    Integrate a function represented at quadrature points against a set of basis functions.

    Args:
        getter (callable): Function (i, j, k) -> scalar. kth component of form i evaluated at quadrature point j.
        w_jk (jnp.ndarray): Function values at quadrature points, shape (n_q, d).
        n (int): Number of basis functions.

    Returns:
        jnp.ndarray: Integrated values, shape (n,). Entries are given by
        ∑_{j,k} Λ[i,j,k] * w[j,k]
    """
    n_q, d = w_jk.shape
    get_f_jk = jax.vmap(
        jax.vmap(getter, in_axes=(None, None, 0)),  # over k (dimensions)
        # over j (quadrature points)
        in_axes=(None, 0, None)
    )

    def body_fun(carry, i):
        L_i = get_f_jk(i, jnp.arange(n_q), jnp.arange(d))  # shape (n_q, d)
        return None, jnp.sum(L_i * w_jk)

    _, R = jax.lax.scan(body_fun, None, jnp.arange(n))
    return R
