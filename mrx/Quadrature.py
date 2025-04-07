import jax.numpy as jnp
import jax


class QuadratureRule:

    def __init__(self, form, p):
        # Quadratures:
        # 'clamped' - composite Gauss quadrature
        # 'periodic' - composite Gauss quadrature
        # 'fourier' - trapezoidal rule
        # 'constant' - single evaluation
        (x_x, w_x), (x_y, w_y), (x_z, w_z) = [select_quadrature(b, p) for b in form.bases[0].bases]

        x_s = [x_x, x_y, x_z]
        w_s = [w_x, w_y, w_z]
        d = 3
        N = w_x.size * w_y.size * w_z.size

        x_q = jnp.array(jnp.meshgrid(*x_s))  # shape d, n1, n2, n3, ...
        x_q = x_q.transpose(*range(1, d+1), 0).reshape(N, d)
        w_q = jnp.array(jnp.meshgrid(*w_s)).transpose(*range(1, d+1), 0).reshape(N, d)
        w_q = jnp.prod(w_q, 1)

        self.x_x = x_x
        self.x_y = x_y
        self.x_z = x_z
        self.w_x = w_x
        self.w_y = w_y
        self.w_z = w_z
        self.x = x_q
        self.w = w_q


def trapezoidal_quad(n):
    h = 1 / n
    x_q = jnp.linspace(0, 1 - h, n)
    w_q = h * jnp.ones(n)
    return x_q, w_q


def composite_quad(T, p):
    # T is the knot vector without multiplicity
    # using p points, we can integrate degree 2p-1 polynomials exactly
    if p == 1:
        _x_q, _w_q = gauss_1_nodes, gauss_1_weights
    elif p == 2:
        _x_q, _w_q = gauss_2_nodes, gauss_2_weights
    elif p == 3:
        _x_q, _w_q = gauss_3_nodes, gauss_3_weights
    elif p == 4:
        _x_q, _w_q = gauss_4_nodes, gauss_4_weights
    elif p == 5:
        _x_q, _w_q = gauss_5_nodes, gauss_5_weights
    elif p == 6:
        _x_q, _w_q = gauss_6_nodes, gauss_6_weights
    elif p == 7:
        _x_q, _w_q = gauss_7_nodes, gauss_7_weights
    elif p == 8:
        _x_q, _w_q = gauss_8_nodes, gauss_8_weights
    elif p == 9:
        _x_q, _w_q = gauss_9_nodes, gauss_9_weights
    elif p == 10:
        _x_q, _w_q = gauss_10_nodes, gauss_10_weights

    def _rescale(a, b):
        return (_x_q + 1) / 2 * (b - a) + a, _w_q * (b - a) / 2

    a_s = T[:-1]
    b_s = T[1:]
    x_q, w_q = jax.vmap(_rescale)(a_s, b_s)
    return jnp.ravel(x_q), jnp.ravel(w_q)


def spectral_quad(p):
    # using p points, we can integrate degree 2p-1 polynomials exactly
    if p == 1:
        _x_q, _w_q = gauss_1_nodes, gauss_1_weights
    elif p == 2:
        _x_q, _w_q = gauss_2_nodes, gauss_2_weights
    elif p == 3:
        _x_q, _w_q = gauss_3_nodes, gauss_3_weights
    elif p == 4:
        _x_q, _w_q = gauss_4_nodes, gauss_4_weights
    elif p == 5:
        _x_q, _w_q = gauss_5_nodes, gauss_5_weights
    elif p == 6:
        _x_q, _w_q = gauss_6_nodes, gauss_6_weights
    elif p == 7:
        _x_q, _w_q = gauss_7_nodes, gauss_7_weights
    elif p == 8:
        _x_q, _w_q = gauss_8_nodes, gauss_8_weights
    elif p == 9:
        _x_q, _w_q = gauss_9_nodes, gauss_9_weights
    elif p == 10:
        _x_q, _w_q = gauss_10_nodes, gauss_10_weights
    return (_x_q + 1) / 2, _w_q / 2


def select_quadrature(basis, n):
    if basis.type == 'clamped':
        return composite_quad(basis.T[basis.p:-basis.p], n)
    elif basis.type == 'periodic':
        return composite_quad(basis.T[basis.p:-basis.p], n)
    elif basis.type == 'fourier':
        return trapezoidal_quad(n)
    elif basis.type == 'constant':
        return spectral_quad(1)


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
