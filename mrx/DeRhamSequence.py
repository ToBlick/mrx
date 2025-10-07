from typing import Callable

import jax
import jax.experimental
import jax.experimental.sparse
import jax.numpy as jnp

from mrx.BoundaryConditions import LazyBoundaryOperator
from mrx.DifferentialForms import DifferentialForm
from mrx.Nonlinearities import CrossProductProjection
from mrx.PolarMapping import LazyExtractionOperator, get_xi
from mrx.Projectors import Projector
from mrx.Quadrature import QuadratureRule
from mrx.Utils import curl, div, grad, inv33, jacobian_determinant


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

    def __init__(self, ns, ps, q, types, F, polar, dirichlet=True):
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
            if dirichlet:
                self.E0, self.E1, self.E2, self.E3 = [
                    LazyExtractionOperator(Λ, ξ, True).matrix()
                    for Λ in [self.Λ0, self.Λ1, self.Λ2, self.Λ3]
                ]
            else:
                self.E0, self.E1, self.E2, self.E3 = [
                    LazyExtractionOperator(Λ, ξ, False).matrix()
                    for Λ in [self.Λ0, self.Λ1, self.Λ2, self.Λ3]
                ]

        else:
            if dirichlet:
                bc = ['dirichlet' if n > 1 else 'none' for n in ns]
                self.E0, self.E1, self.E2, self.E3 = [
                    LazyBoundaryOperator(
                        Λ, bc).matrix()
                    for Λ in [self.Λ0, self.Λ1, self.Λ2, self.Λ3]
                ]
            else:
                self.E0, self.E1, self.E2, self.E3 = [
                    LazyBoundaryOperator(Λ, ('none', 'none', 'none')).matrix()
                    for Λ in [self.Λ0, self.Λ1, self.Λ2, self.Λ3]
                ]

        self.P0, self.P1, self.P2, self.P3 = [
            Projector(Λ, self.Q, self.F, E=E)
            for Λ, E in zip([self.Λ0, self.Λ1, self.Λ2, self.Λ3], [self.E0, self.E1, self.E2, self.E3])
        ]

    def evaluate_0(self):
        self.Λ0_ijk = jax.vmap(jax.vmap(self.Λ0, (0, None)),
                               (None, 0))(self.Q.x, self.Λ0.ns)

    def evaluate_1(self):
        self.Λ1_ijk = jax.vmap(jax.vmap(self.Λ1, (0, None)),
                               (None, 0))(self.Q.x, self.Λ1.ns)

    def evaluate_2(self):
        self.Λ2_ijk = jax.vmap(jax.vmap(self.Λ2, (0, None)),
                               (None, 0))(self.Q.x, self.Λ2.ns)

    def evaluate_3(self):
        self.Λ3_ijk = jax.vmap(jax.vmap(self.Λ3, (0, None)),
                               (None, 0))(self.Q.x, self.Λ3.ns)

    def evaluate_d0(self):
        def dΛ0(x, i):
            return grad(self.Λ0[i])(x)
        self.dΛ0_ijk = jax.vmap(jax.vmap(dΛ0, (0, None)), (None, 0))(
            self.Q.x, self.Λ0.ns)

    def evaluate_d1(self):
        def dΛ1(x, i):
            return curl(self.Λ1[i])(x)
        self.dΛ1_ijk = jax.vmap(jax.vmap(dΛ1, (0, None)), (None, 0))(
            self.Q.x, self.Λ1.ns)

    def evaluate_d2(self):
        def dΛ2(x, i):
            return div(self.Λ2[i])(x)
        self.dΛ2_ijk = jax.vmap(jax.vmap(dΛ2, (0, None)), (None, 0))(
            self.Q.x, self.Λ2.ns)

    def evaluate_all(self):
        self.evaluate_0()
        self.evaluate_1()
        self.evaluate_2()
        self.evaluate_3()
        self.evaluate_d0()
        self.evaluate_d1()
        self.evaluate_d2()

    def assemble_M0(self):
        M0 = jnp.einsum("ijk,ljk,j,j->il", self.Λ0_ijk,
                        self.Λ0_ijk, self.J_j, self.Q.w)
        self.M0 = self.E0 @ M0 @ self.E0.T

    def assemble_M1(self):
        M1 = jnp.einsum("ijk,jkl,qjl,j,j->iq", self.Λ1_ijk,
                        self.G_inv_jkl, self.Λ1_ijk, self.J_j, self.Q.w)
        self.M1 = self.E1 @ M1 @ self.E1.T

    def assemble_M2(self):
        M2 = jnp.einsum("ijk,jkl,qjl,j,j->iq", self.Λ2_ijk,
                        self.G_jkl, self.Λ2_ijk, 1/self.J_j, self.Q.w)
        self.M2 = self.E2 @ M2 @ self.E2.T

    def assemble_M3(self):
        M3 = jnp.einsum("ijk,ljk,j,j->il", self.Λ3_ijk,
                        self.Λ3_ijk, 1/self.J_j, self.Q.w)
        self.M3 = self.E3 @ M3 @ self.E3.T

    def assemble_d0(self):
        D0 = jnp.einsum("ijk,jkl,qjl,j,j->iq", self.Λ1_ijk,
                        self.G_inv_jkl, self.dΛ0_ijk, self.J_j, self.Q.w)
        self.D0 = self.E1 @ D0 @ self.E0.T
        self.strong_grad = jnp.linalg.solve(self.M1, self.D0)
        self.weak_div = -jnp.linalg.solve(self.M0.T, self.D0.T)

    def assemble_d1(self):
        D1 = jnp.einsum("ijk,jkl,qjl,j,j->iq", self.Λ2_ijk,
                        self.G_jkl, self.dΛ1_ijk, 1/self.J_j, self.Q.w)
        self.D1 = self.E2 @ D1 @ self.E1.T
        self.strong_curl = jnp.linalg.solve(self.M2, self.D1)
        self.weak_curl = jnp.linalg.solve(self.M1.T, self.D1.T)

    def assemble_d2(self):
        D2 = jnp.einsum("ijk,ljk,j,j->il", self.Λ3_ijk,
                        self.dΛ2_ijk, 1/self.J_j, self.Q.w)
        self.D2 = self.E3 @ D2 @ self.E2.T
        self.strong_div = jnp.linalg.solve(self.M3, self.D2)
        self.weak_grad = -jnp.linalg.solve(self.M2.T, self.D2.T)

    def assemble_dd0(self):
        GG = jnp.einsum("ijk,jkl,qjl,j,j->iq", self.dΛ0_ijk,
                        self.G_inv_jkl, self.dΛ0_ijk, self.J_j, self.Q.w)
        self.dd0 = jnp.linalg.solve(self.M0, self.E0 @ GG @ self.E0.T)

    def assemble_dd1(self):
        CC = jnp.einsum("ijk,jkl,qjl,j,j->iq", self.dΛ1_ijk,
                        self.G_jkl, self.dΛ1_ijk, 1/self.J_j, self.Q.w)
        self.dd1 = jnp.linalg.solve(
            self.M1, self.E1 @ CC @ self.E1.T) - self.strong_grad @ self.weak_div

    def assemble_dd2(self):
        DD = jnp.einsum("ijk,ljk,j,j->il", self.dΛ2_ijk,
                        self.dΛ2_ijk, 1/self.J_j, self.Q.w)
        self.dd2 = jnp.linalg.solve(
            self.M2, self.E2 @ DD @ self.E2.T) + self.strong_curl @ self.weak_curl

    def assemble_dd3(self):
        self.dd3 = -self.strong_div @ self.weak_grad

    def assemble_P12(self):
        P = jnp.einsum("ijk,ljk,j->il", self.Λ1_ijk,
                       self.Λ2_ijk, self.Q.w)
        M12 = self.E1 @ P @ self.E2.T
        self.P12 = jnp.linalg.solve(self.M1, M12)

    def assemble_P03(self):
        P = jnp.einsum("ijk,ljk,j->il", self.Λ0_ijk,
                       self.Λ3_ijk, self.Q.w)
        M03 = self.E0 @ P @ self.E3.T
        self.P03 = jnp.linalg.solve(self.M0, M03)

    def build_crossproduct_projections(self):
        self.P1x1_to_1 = CrossProductProjection(1, 1, 1, self)
        self.P1x2_to_1 = CrossProductProjection(1, 1, 2, self)
        self.P2x1_to_1 = CrossProductProjection(1, 2, 1, self)
        self.P2x2_to_1 = CrossProductProjection(1, 2, 2, self)

        self.P1x1_to_2 = CrossProductProjection(2, 1, 1, self)
        self.P1x2_to_2 = CrossProductProjection(2, 1, 2, self)
        self.P2x1_to_2 = CrossProductProjection(2, 2, 1, self)
        self.P2x1_to_2 = CrossProductProjection(2, 2, 1, self)

    def assemble_all(self):
        self.assemble_M0()
        self.assemble_M1()
        self.assemble_M2()
        self.assemble_M3()
        self.assemble_d0()
        self.assemble_d1()
        self.assemble_d2()
        self.assemble_dd0()
        self.assemble_dd1()
        self.assemble_dd2()
        self.assemble_dd3()
        self.assemble_P12()
        self.assemble_P03()

    def assemble_leray_projection(self):
        self.P_Leray = jnp.eye(self.M2.shape[0]) + \
            self.weak_grad @ jnp.linalg.pinv(self.dd3) @ self.strong_div
