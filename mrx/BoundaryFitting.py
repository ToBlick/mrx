import jax
import jax.numpy as jnp
import optimistix

from mrx.DifferentialForms import DifferentialForm, DiscreteFunction
from mrx.Quadrature import QuadratureRule


def solovev_lcfs_fit(n_map,
                     p_map,
                     q_map,
                     R0=3,
                     a=0.6,
                     k0=1.5,
                     q0=1.5,
                     atol=1e-6,
                     rtol=1e-6,
                     maxiter=20_000):

    π = jnp.pi

    def psi(R, Z):
        def _psi(R, Z):
            return (k0**2/4*(R**2 - R0**2)**2 + R**2*Z**2) / (2 * R0**2 * k0 * q0)
        return _psi(R, Z) - _psi(R0 + a, 0)

    Λ_map = DifferentialForm(0, (n_map, 1, 1), (p_map, 0, 0),
                             ("periodic", "constant", "constant"))
    Q_map = QuadratureRule(Λ_map, q_map)

    def loss(a_hat, args):
        a_h = DiscreteFunction(a_hat, Λ_map)

        def _psi(x):
            a = a_h(x)[0]
            χ = x[0]
            R = R0 + a * jnp.cos(2 * π * χ)
            Z = a * jnp.sin(2 * π * χ)
            return (psi(R, Z) - 0)**2
        return Q_map.w @ jax.vmap(_psi)(Q_map.x)

    solver = optimistix.LevenbergMarquardt(rtol=rtol, atol=atol)
    sol = optimistix.least_squares(fn=loss,
                                   y0=jnp.ones(n_map),
                                   solver=solver,
                                   max_steps=maxiter
                                   )
    a_hat = sol.value

    return a_hat
