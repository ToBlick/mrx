"""
Quadrature rules for numerical integration in finite element analysis.

This module provides various quadrature rules for numerical integration in the
context of finite element analysis. It includes:
- Composite Gauss quadrature for clamped and periodic bases
- Trapezoidal rule for Fourier bases
- Spectral quadrature for constant bases
- Pre-computed Gauss quadrature nodes and weights up to order 10

The module supports different types of basis functions and provides efficient
implementations using JAX for automatic differentiation and GPU acceleration.
"""

import jax
import jax.numpy as jnp

__all__ = ['QuadratureRule']


class QuadratureRule:
    """
    A class for handling quadrature rules in finite element analysis.

    This class implements various quadrature rules for numerical integration
    in three-dimensional space. It supports different types of basis functions
    and provides efficient computation of quadrature points and weights.

    Attributes:
        x_x (array): Quadrature points in x-direction
        x_y (array): Quadrature points in y-direction
        x_z (array): Quadrature points in z-direction
        w_x (array): Quadrature weights in x-direction
        w_y (array): Quadrature weights in y-direction
        w_z (array): Quadrature weights in z-direction
        x (array): Combined quadrature points in 3D space
        w (array): Combined quadrature weights
    """

    def __init__(self, form, p):
        """
        Initialize the quadrature rule.

        Args:
            form: The differential form defining the basis functions
            p (int): Number of quadrature points per direction
        """
        # Select appropriate quadrature rules for each direction
        (x_x, w_x), (x_y, w_y), (x_z, w_z) = [
            select_quadrature(b, p) for b in form.bases[0].bases]

        # Combine quadrature points and weights in 3D
        x_s = [x_x, x_y, x_z]
        w_s = [w_x, w_y, w_z]
        d = 3
        n = w_x.size * w_y.size * w_z.size

        # Create 3D grid of quadrature points and weights
        x_q = jnp.array(jnp.meshgrid(*x_s))  # shape d, n1, n2, n3, ...
        x_q = x_q.transpose(*range(1, d+1), 0).reshape(n, d)
        w_q = jnp.array(
            jnp.meshgrid(*w_s)).transpose(*range(1, d+1), 0).reshape(n, d)
        w_q = jnp.prod(w_q, 1)

        # Store quadrature points and weights
        self.x_x = x_x
        self.x_y = x_y
        self.x_z = x_z
        self.w_x = w_x
        self.w_y = w_y
        self.w_z = w_z
        self.x = x_q
        self.w = w_q
        self.nx = x_x.size
        self.ny = x_y.size
        self.nz = x_z.size
        self.n = n
        self.ns = jnp.arange(n)


def trapezoidal_quad(n):
    """
    Generate trapezoidal quadrature rule.

    The trapezoidal rule on [0, 1] with n points:
    \int_0^1 f(x) dx ≈ \sum_{i=0}^{n-1} f(x_i) w_i
    where x_i are evenly spaced points from 0 to 1 (spacing h = 1/(n-1)),
    and weights are w_0 = w_{n-1} = h/2 for endpoints and w_i = h for interior points.

    Args:
        n (int): Number of quadrature points

    Returns:
        tuple: (x_q, w_q) where:
            - x_q: Quadrature points (shape: (n,))
            - w_q: Quadrature weights (shape: (n,))
    """
    if n == 1:
        # Special case: single point with weight 1
        x_q = jnp.array([0.0])
        w_q = jnp.array([1.0])
    else:
        h = 1 / (n - 1)  # Spacing between points
        x_q = jnp.linspace(0, 1, n)
        w_q = h * jnp.ones(n)
        w_q = w_q.at[0].set(w_q[0] / 2).at[-1].set(w_q[-1] / 2)
    return x_q, w_q


def composite_quad(T, p):
    """
    Generate composite Gauss quadrature rule.

    Args:
        T (array): Knot vector without multiplicity
        p (int): Number of quadrature points per interval

    Returns:
        tuple: (x_q, w_q) where:
            - x_q: Quadrature points
            - w_q: Quadrature weights
    """
    # Get Gauss quadrature nodes and weights
    _x_q, _w_q = nodes_and_weights(p)

    def _rescale(a, b):
        """Rescale quadrature points and weights from [-1,1] to [a,b].

        Args:
            a (float): Lower endpoint of the interval
            b (float): Upper endpoint of the interval

        Returns:
            tuple: (x, w) where:
                - x: Quadrature points on [a,b]
                - w: Quadrature weights on [a,b]
        """
        x = (_x_q + 1) / 2 * (b - a) + a
        w = _w_q * (b - a) / 2
        return x, w

    # Get interval endpoints
    a_s = T[:-1]
    b_s = T[1:]

    # Apply rescaling to each interval
    x_q, w_q = jax.vmap(_rescale)(a_s, b_s)

    # Flatten arrays to get final points and weights
    return jnp.ravel(x_q), jnp.ravel(w_q)


def spectral_quad(p):
    """
    Generate spectral quadrature rule.

    Args:
        p (int): Number of quadrature points

    Returns:
        tuple: (x_q, w_q) where:
            - x_q: Quadrature points
            - w_q: Quadrature weights
    """
    # Using p points, we can integrate degree 2p-1 polynomials exactly
    if p == 1:
        _x_q, _w_q = nodes_and_weights(1)
    elif p == 2:
        _x_q, _w_q = nodes_and_weights(2)
    elif p == 3:
        _x_q, _w_q = nodes_and_weights(3)
    elif p == 4:
        _x_q, _w_q = nodes_and_weights(4)
    elif p == 5:
        _x_q, _w_q = nodes_and_weights(5)
    elif p == 6:
        _x_q, _w_q = nodes_and_weights(6)
    elif p == 7:
        _x_q, _w_q = nodes_and_weights(7)
    elif p == 8:
        _x_q, _w_q = nodes_and_weights(8)
    elif p == 9:
        _x_q, _w_q = nodes_and_weights(9)
    elif p == 10:
        _x_q, _w_q = nodes_and_weights(10)
    return (_x_q + 1) / 2, _w_q / 2


def select_quadrature(basis, n):
    """
    Select appropriate quadrature rule based on basis type.

    Args:
        basis: The basis function object
        n (int): Number of quadrature points

    Returns:
        tuple: (x_q, w_q) where:
            - x_q: Quadrature points
            - w_q: Quadrature weights
    """
    if basis.type == 'clamped':
        return composite_quad(basis.T[basis.p:-basis.p], n)
    elif basis.type == 'periodic':
        return composite_quad(basis.T[basis.p:-basis.p], n)
        # return trapezoidal_quad(2*n)
    elif basis.type == 'fourier':
        return trapezoidal_quad(2*n)
    elif basis.type == 'constant':
        return spectral_quad(1)

def nodes_and_weights(n):
    """
    Get pre-computed Gauss quadrature nodes and weights.

    Args:
        n (int): Number of quadrature points (1-10)

    Returns:
        tuple: (points, weights) where:
            - points: Gauss quadrature nodes
            - weights: Gauss quadrature weights
    """
    if n == 1:
        points = gauss_1_nodes
        weights = gauss_1_weights
    elif n == 2:
        points = gauss_2_nodes
        weights = gauss_2_weights
    elif n == 3:
        points = gauss_3_nodes
        weights = gauss_3_weights
    elif n == 4:
        points = gauss_4_nodes
        weights = gauss_4_weights
    elif n == 5:
        points = gauss_5_nodes
        weights = gauss_5_weights
    elif n == 6:
        points = gauss_6_nodes
        weights = gauss_6_weights
    elif n == 7:
        points = gauss_7_nodes
        weights = gauss_7_weights
    elif n == 8:
        points = gauss_8_nodes
        weights = gauss_8_weights
    elif n == 9:
        points = gauss_9_nodes
        weights = gauss_9_weights
    # n=10
    else:
        points = gauss_10_nodes
        weights = gauss_10_weights
    return points, weights


# Pre-computed Gauss quadrature nodes and weights
# These values are exact and used for efficient computation
gauss_1_nodes = jnp.array([
    0.0
])

gauss_1_weights = jnp.array([
    2.0
])

gauss_2_nodes = jnp.array([
    -0.5773502691896258,
    0.5773502691896258
])

gauss_2_weights = jnp.array([
    1.0,
    1.0
])

gauss_3_nodes = jnp.array([
    -0.7745966692414834,
    0.0,
    0.7745966692414834
])

gauss_3_weights = jnp.array([
    0.5555555555555556,
    0.8888888888888888,
    0.5555555555555556
])

gauss_4_nodes = jnp.array([
    -0.8611363115940526,
    -0.3399810435848563,
    0.3399810435848563,
    0.8611363115940526
])

gauss_4_weights = jnp.array([
    0.3478548451374538,
    0.6521451548625461,
    0.6521451548625461,
    0.3478548451374538
])

gauss_5_nodes = jnp.array([
    -0.9061798459386640,
    -0.5384693101056831,
    0.0,
    0.5384693101056831,
    0.9061798459386640
])

gauss_5_weights = jnp.array([
    0.2369268850561891,
    0.4786286704993665,
    0.5688888888888889,
    0.4786286704993665,
    0.2369268850561891
])

gauss_6_nodes = jnp.array([
    -0.9324695142031521,
    -0.6612093864662645,
    -0.2386191860831969,
    0.2386191860831969,
    0.6612093864662645,
    0.9324695142031521
])

gauss_6_weights = jnp.array([
    0.1713244923791704,
    0.3607615730481386,
    0.4679139345726910,
    0.4679139345726910,
    0.3607615730481386,
    0.1713244923791704
])

gauss_7_nodes = jnp.array([
    -0.9491079123427585,
    -0.7415311855993945,
    -0.4058451513773972,
    0.0,
    0.4058451513773972,
    0.7415311855993945,
    0.9491079123427585
])

gauss_7_weights = jnp.array([
    0.1294849661688697,
    0.2797053914892766,
    0.3818300505051189,
    0.4179591836734694,
    0.3818300505051189,
    0.2797053914892766,
    0.1294849661688697
])

gauss_8_nodes = jnp.array([
    -0.9602898564975363,
    -0.7966664774136267,
    -0.5255324099163290,
    -0.1834346424956498,
    0.1834346424956498,
    0.5255324099163290,
    0.7966664774136267,
    0.9602898564975363
])

gauss_8_weights = jnp.array([
    0.1012285362903763,
    0.2223810344533745,
    0.3137066458778873,
    0.3626837833783620,
    0.3626837833783620,
    0.3137066458778873,
    0.2223810344533745,
    0.1012285362903763
])

gauss_9_nodes = jnp.array([
    -0.9681602395076261,
    -0.8360311073266358,
    -0.6133714327005904,
    -0.3242534234038089,
    0.0,
    0.3242534234038089,
    0.6133714327005904,
    0.8360311073266358,
    0.9681602395076261
])

gauss_9_weights = jnp.array([
    0.0812743883615744,
    0.1806481606948574,
    0.2606106964029354,
    0.3123470770400029,
    0.3302393550012598,
    0.3123470770400029,
    0.2606106964029354,
    0.1806481606948574,
    0.0812743883615744
])

gauss_10_nodes = jnp.array([
    -0.9739065285171717,
    -0.8650633666889845,
    -0.6794095682990244,
    -0.4333953941292472,
    -0.1488743389816312,
    0.1488743389816312,
    0.4333953941292472,
    0.6794095682990244,
    0.8650633666889845,
    0.9739065285171717
])

gauss_10_weights = jnp.array([
    0.0666713443086881,
    0.1494513491505806,
    0.2190863625159820,
    0.2692667193099963,
    0.2955242247147529,
    0.2955242247147529,
    0.2692667193099963,
    0.2190863625159820,
    0.1494513491505806,
    0.0666713443086881
])
