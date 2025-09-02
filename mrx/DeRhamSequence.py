import jax
import jax.experimental
import jax.experimental.sparse
import jax.numpy as jnp

from mrx.BoundaryConditions import LazyBoundaryOperator
from mrx.DifferentialForms import DifferentialForm
from mrx.PolarMapping import LazyExtractionOperator, get_xi
from mrx.Projectors import Projector
from mrx.Quadrature import QuadratureRule
from mrx.Utils import curl, div, grad, inv33, jacobian_determinant
from typing import NamedTuple, Any, Callable

class DeRhamSequence():
    """
    A class to represent a de Rham sequence.
    """
    Λ0: DifferentialForm
    Λ1: DifferentialForm
    Λ2: DifferentialForm
    Λ3: DifferentialForm
    Q: QuadratureRule
    F: Callable
    E0: LazyBoundaryOperator
    E1: LazyBoundaryOperator
    E2: LazyBoundaryOperator
    E3: LazyBoundaryOperator
    # kth component of 0form i evaluated at quadrature point j. shape: n x n_q x 1
    Λ0_ijk: jnp.ndarray
    # kth component of 1form i evaluated at quadrature point j. shape: n x n_q x 3
    Λ1_ijk: jnp.ndarray
    # kth component of 2form i evaluated at quadrature point j. shape: n x n_q x 3
    Λ2_ijk: jnp.ndarray
    # kth component of 3form i evaluated at quadrature point j. shape: n x n_q x 1
    Λ3_ijk: jnp.ndarray
    # kth component of grad of 0form i evaluated at quadrature point j. shape: n x n_q x 3
    dΛ0_ijk: jnp.ndarray
    # kth component of curl of 1form i evaluated at quadrature point j. shape: n x n_q x 3
    dΛ1_ijk: jnp.ndarray
    # kth component of div of 2form i evaluated at quadrature point j. shape: n x n_q x 1
    dΛ2_ijk: jnp.ndarray

    # Jacobian determinant evaluated at quadrature points. shape: n_q x 1
    J_j = jnp.ndarray
    # (k,l)th element of metric at quadrature point j. shape: n_q x 3 x 3
    G_jkl = jnp.ndarray
    G_inv_jkl = jnp.ndarray

    def __init__(self, ns, ps, q, types, bcs, F, polar):
        """
        Initialize the de Rham sequence.    

        """
        self.Λ0, self.Λ1, self.Λ2, self.Λ3 = [
            DifferentialForm(i, ns, ps, types) for i in range(0, 4)
        ]
        self.Q = QuadratureRule(self.Λ0, q)
        self.F = F

        def G(x):
            return jax.jacfwd(self.F)(x).T @ jax.jacfwd(self.F)(x)

        self.G_jkl = jax.vmap(G)(self.Q.x)
        self.G_inv_jkl = jax.vmap(inv33)(self.G_jkl)
        self.J_j = jax.vmap(jacobian_determinant(self.F))(self.Q.x)

        if polar:
            def _R(r, χ):
                return self.F(jnp.array([r, χ, 0.0]))[0] * jnp.ones(1)

            def _Z(r, χ):
                return self.F(jnp.array([r, χ, 0.0]))[2] * jnp.ones(1)
            ξ = get_xi(_R, _Z, self.Λ0, self.Q)[0]
            self.E0, self.E1, self.E2, self.E3 = [
                LazyExtractionOperator(Λ, ξ, bcs[0] == 'dirichlet')
                for Λ in [self.Λ0, self.Λ1, self.Λ2, self.Λ3]
            ]
        else:
            self.E0, self.E1, self.E2, self.E3 = [
                LazyBoundaryOperator(Λ, bcs)
                for Λ in [self.Λ0, self.Λ1, self.Λ2, self.Λ3]
            ]

        self.Λ0_ijk = jax.vmap(jax.vmap(self.Λ0, (0, None)),
                               (None, 0))(self.Q.x, self.Λ0.ns)
        self.Λ1_ijk = jax.vmap(jax.vmap(self.Λ1, (0, None)),
                               (None, 0))(self.Q.x, self.Λ1.ns)
        self.Λ2_ijk = jax.vmap(jax.vmap(self.Λ2, (0, None)),
                               (None, 0))(self.Q.x, self.Λ2.ns)
        self.Λ3_ijk = jax.vmap(jax.vmap(self.Λ3, (0, None)),
                               (None, 0))(self.Q.x, self.Λ3.ns)

        def dΛ0(x, i):
            return grad(self.Λ0[i])(x)

        def dΛ1(x, i):
            return curl(self.Λ1[i])(x)

        def dΛ2(x, i):
            return div(self.Λ2[i])(x)

        self.dΛ0_ijk = jax.vmap(jax.vmap(dΛ0, (0, None)), (None, 0))(
            self.Q.x, self.Λ0.ns)
        self.dΛ1_ijk = jax.vmap(jax.vmap(dΛ1, (0, None)), (None, 0))(
            self.Q.x, self.Λ1.ns)
        self.dΛ2_ijk = jax.vmap(jax.vmap(dΛ2, (0, None)), (None, 0))(
            self.Q.x, self.Λ2.ns)

        self.P0, self.P1, self.P2, self.P3 = [
            Projector(Λ, self.Q, self.F, E=E)
            for Λ, E in zip([self.Λ0, self.Λ1, self.Λ2, self.Λ3], [self.E0, self.E1, self.E2, self.E3])
        ]

    def assemble_M0(self):
        M0 = jnp.einsum("ijk,ljk,j,j->il", self.Λ0_ijk,
                        self.Λ0_ijk, self.J_j, self.Q.w)
        return self.E0.matrix() @ M0 @ self.E0.matrix().T

    def assemble_M1(self):
        M1 = jnp.einsum("ijk,jkl,qjl,j,j->iq", self.Λ1_ijk,
                        self.G_inv_jkl, self.Λ1_ijk, self.J_j, self.Q.w)
        return self.E1.matrix() @ M1 @ self.E1.matrix().T

    def assemble_M2(self):
        M2 = jnp.einsum("ijk,jkl,qjl,j,j->iq", self.Λ2_ijk,
                        self.G_jkl, self.Λ2_ijk, 1/self.J_j, self.Q.w)
        return self.E2.matrix() @ M2 @ self.E2.matrix().T

    def assemble_M3(self):
        M3 = jnp.einsum("ijk,ljk,j,j->il", self.Λ3_ijk,
                        self.Λ3_ijk, 1/self.J_j, self.Q.w)
        return self.E3.matrix() @ M3 @ self.E3.matrix().T

    def assemble_grad(self):
        D0 = jnp.einsum("ijk,jkl,qjl,j,j->iq", self.Λ1_ijk,
                        self.G_inv_jkl, self.dΛ0_ijk, self.J_j, self.Q.w)
        return self.E1.matrix() @ D0 @ self.E0.matrix().T

    def assemble_curl(self):
        D1 = jnp.einsum("ijk,jkl,qjl,j,j->iq", self.Λ2_ijk,
                        self.G_jkl, self.dΛ1_ijk, 1/self.J_j, self.Q.w)
        return self.E2.matrix() @ D1 @ self.E1.matrix().T

    def assemble_dvg(self):
        D2 = jnp.einsum("ijk,ljk,j,j->il", self.Λ3_ijk,
                        self.dΛ2_ijk, 1/self.J_j, self.Q.w)
        return self.E3.matrix() @ D2 @ self.E2.matrix().T

    def assemble_gradgrad(self):
        GG = jnp.einsum("ijk,jkl,qjl,j,j->iq", self.dΛ0_ijk,
                        self.G_inv_jkl, self.dΛ0_ijk, self.J_j, self.Q.w)
        return self.E0.matrix() @ GG @ self.E0.matrix().T

    def assemble_curlcurl(self):
        CC = jnp.einsum("ijk,jkl,qjl,j,j->iq", self.dΛ1_ijk,
                        self.G_inv_jkl, self.dΛ1_ijk, 1/self.J_j, self.Q.w)
        return self.E1.matrix() @ CC @ self.E1.matrix().T

    def assemble_divdiv(self):
        DD = jnp.einsum("ijk,ljk,j,j->il", self.dΛ2_ijk,
                        self.dΛ2_ijk, 1/self.J_j, self.Q.w)
        return self.E2.matrix() @ DD @ self.E2.matrix().T
