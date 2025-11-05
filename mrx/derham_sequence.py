from typing import Callable

import jax
import jax.experimental
import jax.experimental.sparse
import jax.numpy as jnp

from mrx.boundary import LazyBoundaryOperator
from mrx.differential_forms import DifferentialForm
from mrx.nonlinearities import CrossProductProjection
from mrx.polar import LazyExtractionOperator, get_xi
from mrx.projectors import Projector
from mrx.quadrature import QuadratureRule
from mrx.utils import assemble, curl, div, grad, inv33, jacobian_determinant


class DeRhamSequence():
    """
    A class to represent a de Rham sequence.

    Attributes:

    TODO: Tobi please add a description of the attributes.
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
    r: jnp.ndarray
    theta: jnp.ndarray
    z: jnp.ndarray
    dr: jnp.ndarray
    dtheta: jnp.ndarray
    dz: jnp.ndarray
    J_j: jnp.ndarray # Jacobian determinant evaluated at quadrature points. shape: n_q x 1
    G_jkl: jnp.ndarray # (k,l)th element of metric at quadrature point j. shape: n_q x 3 x 3
    G_inv_jkl: jnp.ndarray

    def __init__(self, ns, ps, q, types, F, polar, dirichlet=True):
        """
        Initialize the de Rham sequence.    

        Args:
            ns (list): List of integers representing the number of basis functions for each differential form.
            ps (list): List of integers representing the order of the basis functions for each differential form.
            q (int): The order of the quadrature rule.
            types (list): List of strings representing the type of boundary condition for each differential form.
            F (callable): The mapping function from logical to physical domain.
            polar (bool): Whether to use polar coordinates.
            dirichlet (bool): Whether to use Dirichlet boundary conditions.
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
            ξ = get_xi(ns[1])
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
            # TODO: right now, we only support dirichlet BCs in r
            if dirichlet:
                self.E0, self.E1, self.E2, self.E3 = [
                    LazyBoundaryOperator(
                        Λ, ('dirichlet', 'none', 'none')).matrix()
                    for Λ in [self.Λ0, self.Λ1, self.Λ2, self.Λ3]
                ]
            else:
                self.E0, self.E1, self.E2, self.E3 = [
                    LazyBoundaryOperator(Λ, ('none', 'none', 'none')).matrix()
                    for Λ in [self.Λ0, self.Λ1, self.Λ2, self.Λ3]
                ]

        self.P0, self.P1, self.P2, self.P3 = [
            Projector(self, k) for k in range(4)
        ]

    def evaluate_1d(self):
        """
        Evaluate the 1-dimensional basis functions at the quadrature points.
        """
        self.r = jax.vmap(jax.vmap(self.Λ0.Λ[0], (0, None)),
                          (None, 0))(self.Q.x_x, self.Λ0.Λ[0].ns)
        self.theta = jax.vmap(jax.vmap(self.Λ0.Λ[1], (0, None)),
                              (None, 0))(self.Q.x_y, self.Λ0.Λ[1].ns)
        self.z = jax.vmap(jax.vmap(self.Λ0.Λ[2], (0, None)),
                          (None, 0))(self.Q.x_z, self.Λ0.Λ[2].ns)
        self.dr = jax.vmap(jax.vmap(self.Λ0.dΛ[0], (0, None)),
                           (None, 0))(self.Q.x_x, self.Λ0.dΛ[0].ns)
        self.dtheta = jax.vmap(jax.vmap(self.Λ0.dΛ[1], (0, None)),
                               (None, 0))(self.Q.x_y, self.Λ0.dΛ[1].ns)
        self.dz = jax.vmap(jax.vmap(self.Λ0.dΛ[2], (0, None)),
                           (None, 0))(self.Q.x_z, self.Λ0.dΛ[2].ns)

    def get_Λ0_ijk(self, i, j, k):
        """
        Get the kth component of the ith 0-form evaluated at quadrature point j.

        Args:
            i (int): The index of the basis function.
            j (int): The index of the quadrature point.
            k (int): The index of the component.

        Returns:
            float: The value of the kth component of the ith 0-form evaluated at quadrature point j.
        """
        # get 1d quadrature points
        # weird order here is due to meshgrid's indexing
        j2, j1, j3 = jnp.unravel_index(j, (self.Q.ny, self.Q.nx, self.Q.nz))

        # get the 1d basis functions
        _, i1, i2, i3 = self.Λ0._unravel_index(i)
        # k is always 0
        return self.r[i1, j1] * self.theta[i2, j2] * self.z[i3, j3]

    def get_dΛ0_ijk(self, i, j, k):
        """
        Get the kth component of the gradient of the ith 0-form evaluated at quadrature point j.

        Args:
            i (int): The index of the basis function.
            j (int): The index of the quadrature point.
            k (int): The index of the component.

        Returns:
            float: The value of the kth component of the gradient of the ith 0-form evaluated at quadrature point j.
        """
        # kth component of gradient of 0 form i evaluated at quadrature point j.
        j2, j1, j3 = jnp.unravel_index(j, (self.Q.ny, self.Q.nx, self.Q.nz))
        _, i1, i2, i3 = self.Λ0._unravel_index(i)
        # get i-1
        dr = jnp.where(i1 == self.Λ0.nχ-1, 0.0, self.dr[i1, j1])
        dr_m1 = jnp.where(i1 > 0, self.dr[i1-1, j1], 0.0)
        dtheta_m1 = jnp.where(
            i2 > 0, self.dtheta[i2-1, j2], self.dtheta[self.Λ0.nχ-1, j2])
        dtheta = self.dtheta[i2, j2]
        dz_m1 = jnp.where(i3 > 0, self.dz[i3-1, j3], self.dz[self.Λ0.nχ-1, j3])
        dz = self.dz[i3, j3]
        return jnp.where(k == 0,
                         (dr_m1 - dr) * self.theta[i2, j2] * self.z[i3, j3],
                         jnp.where(k == 1,
                                   self.r[i1, j1] *
                                   (dtheta_m1 - dtheta) * self.z[i3, j3],
                                   self.r[i1, j1] * self.theta[i2, j2] * (dz_m1 - dz)))

    def get_Λ1_ijk(self, i, j, k):
        """
        Get the kth component of the ith 1-form evaluated at quadrature point j.

        Args:
            i (int): The index of the basis function.
            j (int): The index of the quadrature point.
            k (int): The index of the component.

        Returns:
            float: The value of the kth component of the ith 1-form evaluated at quadrature point j.
        """
        # kth component of 1 form i evaluated at quadrature point j.
        j2, j1, j3 = jnp.unravel_index(j, (self.Q.ny, self.Q.nx, self.Q.nz))
        c, i1, i2, i3 = self.Λ1._unravel_index(i)
        return jnp.where(k == c,
                         jnp.where(k == 0,
                                   self.dr[i1, j1] *
                                   self.theta[i2, j2] * self.z[i3, j3],
                                   jnp.where(k == 1,
                                             self.r[i1, j1] *
                                             self.dtheta[i2, j2] *
                                             self.z[i3, j3],
                                             self.r[i1, j1] * self.theta[i2, j2] * self.dz[i3, j3])),
                         0.0)

    def get_dΛ1_ijk(self, i, j, k):
        """
        Get the kth component of the curl of the ith 1-form evaluated at quadrature point j.

        Args:
            i (int): The index of the basis function.
            j (int): The index of the quadrature point.
            k (int): The index of the component.

        Returns:
            float: The value of the kth component of the curl of the ith 1-form evaluated at quadrature point j.
        """
        j2, j1, j3 = jnp.unravel_index(j, (self.Q.ny, self.Q.nx, self.Q.nz))
        c, i1, i2, i3 = self.Λ1._unravel_index(i)
        # get i-1
        dr = jnp.where(i1 == self.Λ1.nχ-1, 0.0, self.dr[i1, j1])
        dr_m1 = jnp.where(i1 > 0, self.dr[i1-1, j1], 0.0)
        dtheta_m1 = jnp.where(
            i2 > 0, self.dtheta[i2-1, j2], self.dtheta[self.Λ1.nχ-1, j2])
        dtheta = self.dtheta[i2, j2]
        dz_m1 = jnp.where(i3 > 0, self.dz[i3-1, j3], self.dz[self.Λ1.nχ-1, j3])
        dz = self.dz[i3, j3]
        # d3/dy - d2/dz
        d3dy = self.r[i1, j1] * (dtheta_m1 - dtheta) * self.dz[i3, j3]
        d2dz = self.r[i1, j1] * self.dtheta[i2, j2] * (dz_m1 - dz)
        d1dz = self.dr[i1, j1] * self.theta[i2, j2] * (dz_m1 - dz)
        d3dx = (dr_m1 - dr) * self.theta[i2, j2] * self.dz[i3, j3]
        d2dx = (dr_m1 - dr) * self.dtheta[i2, j2] * self.z[i3, j3]
        d1dy = self.dr[i1, j1] * (dtheta_m1 - dtheta) * self.z[i3, j3]

        # c is not defined below and this is quite a complicated chain of np.where
        return jnp.where(c == 0,
                         jnp.where(k == 0,
                                   0.0,
                                   jnp.where(k == 1,
                                             d1dz,
                                             -d1dy)
                                   ),
                         jnp.where(c == 1,
                                   jnp.where(k == 0,
                                             -d2dz,
                                             jnp.where(k == 1,
                                                       0.0,
                                                       d2dx)
                                             ),  # c==2
                                   jnp.where(k == 0,
                                             d3dy,
                                             jnp.where(k == 1,
                                                       -d3dx,
                                                       0.0)
                                             )
                                   )
                         )

    def get_Λ2_ijk(self, i, j, k):
        """
        Get the kth component of the ith 2-form evaluated at quadrature point j.

        Args:
            i (int): The index of the basis function.
            j (int): The index of the quadrature point.
            k (int): The index of the component.

        Returns:
            float: The value of the kth component of the ith 2-form evaluated at quadrature point j.
        """
        j2, j1, j3 = jnp.unravel_index(j, (self.Q.ny, self.Q.nx, self.Q.nz))
        c, i1, i2, i3 = self.Λ2._unravel_index(i)
        return jnp.where(k == c,
                         jnp.where(k == 0,
                                   self.r[i1, j1] *
                                   self.dtheta[i2, j2] * self.dz[i3, j3],
                                   jnp.where(k == 1,
                                             self.dr[i1, j1] *
                                             self.theta[i2, j2] *
                                             self.dz[i3, j3],
                                             self.dr[i1, j1] * self.dtheta[i2, j2] * self.z[i3, j3])),
                         0.0)

    def get_dΛ2_ijk(self, i, j, k):
        """
        Get the kth component of the divergence of the ith 2-form evaluated at quadrature point j.

        Args:
            i (int): The index of the basis function.
            j (int): The index of the quadrature point.
            k (int): The index of the component.

        Returns:
            float: The value of the kth component of the divergence of the ith 2-form evaluated at quadrature point j.
        """
        j2, j1, j3 = jnp.unravel_index(j, (self.Q.ny, self.Q.nx, self.Q.nz))
        c, i1, i2, i3 = self.Λ2._unravel_index(i)
        # get i-1
        dr = jnp.where(i1 == self.Λ2.nχ-1, 0.0, self.dr[i1, j1])
        dr_m1 = jnp.where(i1 > 0, self.dr[i1-1, j1], 0.0)
        dtheta_m1 = jnp.where(
            i2 > 0, self.dtheta[i2-1, j2], self.dtheta[self.Λ2.nχ-1, j2])
        dtheta = self.dtheta[i2, j2]
        dz_m1 = jnp.where(i3 > 0, self.dz[i3-1, j3], self.dz[self.Λ2.nχ-1, j3])
        dz = self.dz[i3, j3]

        return jnp.where(c == 0,
                         (dr_m1 - dr) * self.dtheta[i2, j2] * self.dz[i3, j3],
                         jnp.where(c == 1,
                                   self.dr[i1, j1] *
                                   (dtheta_m1 - dtheta) * self.dz[i3, j3],
                                   self.dr[i1, j1] *
                                   self.dtheta[i2, j2] * (dz_m1 - dz)
                                   )
                         )

    def get_Λ3_ijk(self, i, j, k):
        """
        Get the kth component of the ith 3-form evaluated at quadrature point j.

        Args:
            i (int): The index of the basis function.
            j (int): The index of the quadrature point.
            k (int): The index of the component.

        Returns:
            float: The value of the kth component of the ith 3-form evaluated at quadrature point j.
        """
        j2, j1, j3 = jnp.unravel_index(j, (self.Q.ny, self.Q.nx, self.Q.nz))
        _, i1, i2, i3 = self.Λ3._unravel_index(i)
        return self.dr[i1, j1] * self.dtheta[i2, j2] * self.dz[i3, j3]  # k is always 0

    def evaluate_0(self):
        """
        Evaluate the 0-forms at the quadrature points and store them in self.Λ0_ijk.
        """
        self.Λ0_ijk = jax.vmap(jax.vmap(self.Λ0, (0, None)),
                               (None, 0))(self.Q.x, self.Λ0.ns)

    def evaluate_1(self):
        """
        Evaluate the 1-forms at the quadrature points and store them in self.Λ1_ijk.
        """
        self.Λ1_ijk = jax.vmap(jax.vmap(self.Λ1, (0, None)),
                               (None, 0))(self.Q.x, self.Λ1.ns)

    def evaluate_2(self):
        """
        Evaluate the 2-forms at the quadrature points and store them in self.Λ2_ijk.
        """
        self.Λ2_ijk = jax.vmap(jax.vmap(self.Λ2, (0, None)),
                               (None, 0))(self.Q.x, self.Λ2.ns)

    def evaluate_3(self):
        """
        Evaluate the 3-forms at the quadrature points and store them in self.Λ3_ijk.
        """
        self.Λ3_ijk = jax.vmap(jax.vmap(self.Λ3, (0, None)),
                               (None, 0))(self.Q.x, self.Λ3.ns)

    def evaluate_d0(self):
        """
        Evaluate the gradient of the 0-forms at the quadrature points and store them in self.dΛ0_ijk.
        """
        def dΛ0(x, i):
            return grad(self.Λ0[i])(x)
        self.dΛ0_ijk = jax.vmap(jax.vmap(dΛ0, (0, None)), (None, 0))(
            self.Q.x, self.Λ0.ns)

    def evaluate_d1(self):
        """
        Evaluate the curl of the 1-forms at the quadrature points and store them in self.dΛ1_ijk.
        """
        def dΛ1(x, i):
            return curl(self.Λ1[i])(x)
        self.dΛ1_ijk = jax.vmap(jax.vmap(dΛ1, (0, None)), (None, 0))(
            self.Q.x, self.Λ1.ns)

    def evaluate_d2(self):
        """
        Evaluate the divergence of the 2-forms at the quadrature points and store them in self.dΛ2_ijk.
        """
        def dΛ2(x, i):
            return div(self.Λ2[i])(x)
        self.dΛ2_ijk = jax.vmap(jax.vmap(dΛ2, (0, None)), (None, 0))(
            self.Q.x, self.Λ2.ns)

    def evaluate_all(self):
        """
        Evaluate all the forms and their gradients at the quadrature points.
        """
        self.evaluate_0()
        self.evaluate_1()
        self.evaluate_2()
        self.evaluate_3()
        self.evaluate_d0()
        self.evaluate_d1()
        self.evaluate_d2()

    def assemble_M0(self):
        """
        Assemble the mass matrix for the 0-forms. Formula:
            M0 = E0 @ (∫ Λ0_ijk Λ0_ijk^T detD) @ E0.T
        """
        W = (self.J_j * self.Q.w)[:, None, None]  # shape (n_q, 1, 1)
        M = assemble(self.get_Λ0_ijk, self.get_Λ0_ijk, W, self.Λ0.n, self.Λ0.n)
        self.M0 = self.E0 @ M @ self.E0.T

    def assemble_M1(self):
        """
        Assemble the mass matrix for the 1-forms. Formula:
            M1 = E1 @ (∫ Λ1_ijk Λ1_ijk^T detD) @ E1.T
        """
        W = self.G_inv_jkl * (self.J_j * self.Q.w)[:, None, None] # shape (n_q, 3, 3)
        M = assemble(self.get_Λ1_ijk, self.get_Λ1_ijk, W, self.Λ1.n, self.Λ1.n)
        self.M1 = self.E1 @ M @ self.E1.T

    def assemble_M2(self):
        """
        Assemble the mass matrix for the 2-forms. Formula:
            M2 = E2 @ (∫ Λ2_ijk Λ2_ijk^T detD) @ E2.T
        """
        W = self.G_jkl * (1/self.J_j * self.Q.w)[:, None, None]
        M = assemble(self.get_Λ2_ijk, self.get_Λ2_ijk, W, self.Λ2.n, self.Λ2.n)
        self.M2 = self.E2 @ M @ self.E2.T

    def assemble_M3(self):
        """
        Assemble the mass matrix for the 3-forms. Formula:
            M3 = E3 @ (∫ Λ3_ijk Λ3_ijk^T detD) @ E3.T
        """
        W = (1/self.J_j * self.Q.w)[:, None, None]
        M = assemble(self.get_Λ3_ijk, self.get_Λ3_ijk, W, self.Λ3.n, self.Λ3.n)
        self.M3 = self.E3 @ M @ self.E3.T

    def assemble_d0(self):
        """
        Assemble the derivative matrix for the 0-forms. Formula:
            D0 = E1 @ (∫ Λ1_ijk Λ0_ijk^T detD) @ E0.T
        """
        W = self.G_inv_jkl * (self.J_j * self.Q.w)[:, None, None]
        M = assemble(self.get_Λ1_ijk, self.get_dΛ0_ijk, W, self.Λ1.n, self.Λ0.n)
        self.D0 = self.E1 @ M @ self.E0.T
        self.strong_grad = jnp.linalg.solve(self.M1, self.D0)
        self.weak_div = -jnp.linalg.solve(self.M0.T, self.D0.T)

    def assemble_d1(self):
        """
        Assemble the derivative matrix for the 1-forms. Formula:
            D1 = E2 @ (∫ Λ2_ijk Λ1_ijk^T detD) @ E1.T
        """
        W = self.G_jkl * (1/self.J_j * self.Q.w)[:, None, None]
        M = assemble(self.get_Λ2_ijk, self.get_dΛ1_ijk, W, self.Λ2.n, self.Λ1.n)
        self.D1 = self.E2 @ M @ self.E1.T
        self.strong_curl = jnp.linalg.solve(self.M2, self.D1)
        self.weak_curl = jnp.linalg.solve(self.M1.T, self.D1.T)

    def assemble_d2(self):
        """
        Assemble the derivative matrix for the 2-forms. Formula:
            D2 = E3 @ (∫ Λ3_ijk Λ2_ijk^T detD) @ E2.T
        """
        W = (1/self.J_j * self.Q.w)[:, None, None]
        M = assemble(self.get_Λ3_ijk, self.get_dΛ2_ijk, W, self.Λ3.n, self.Λ2.n)
        self.D2 = self.E3 @ M @ self.E2.T
        self.strong_div = jnp.linalg.solve(self.M3, self.D2)
        self.weak_grad = -jnp.linalg.solve(self.M2.T, self.D2.T)

    def assemble_dd0(self):
        """
        Assemble the second derivative matrix for the 0-forms. Formula:
            M0 @dd0 = E0 @ (∫ dΛ0_ijk dΛ0_ijk^T detD) @ E0.T
        """
        W = self.G_inv_jkl * (self.J_j * self.Q.w)[:, None, None]
        M = assemble(self.get_dΛ0_ijk, self.get_dΛ0_ijk, W, self.Λ0.n, self.Λ0.n)
        self.dd0 = jnp.linalg.solve(self.M0, self.E0 @ M @ self.E0.T)

    def assemble_dd1(self):
        """ 
        Assemble the second derivative matrix for the 1-forms. Formula:
            M1 @dd1 = E1 @ (∫ dΛ1_ijk dΛ1_ijk^T detD) @ E1.T - strong_grad @ weak_div
        """
        W = self.G_jkl * (1/self.J_j * self.Q.w)[:, None, None]
        M = assemble(self.get_dΛ1_ijk, self.get_dΛ1_ijk, W, self.Λ1.n, self.Λ1.n)
        self.dd1 = jnp.linalg.solve(
            self.M1, self.E1 @ M @ self.E1.T) - self.strong_grad @ self.weak_div

    def assemble_dd2(self):
        """
        Assemble the second derivative matrix for the 2-forms. Formula:
            M2 @dd2 = E2 @ (∫ dΛ2_ijk dΛ2_ijk^T detD) @ E2.T + strong_curl @ weak_curl
        """
        W = (1/self.J_j * self.Q.w)[:, None, None]
        M = assemble(self.get_dΛ2_ijk, self.get_dΛ2_ijk, W, self.Λ2.n, self.Λ2.n)
        self.dd2 = jnp.linalg.solve(self.M2, self.E2 @ M @ self.E2.T) + self.strong_curl @ self.weak_curl

    def assemble_dd3(self):
        """
        Assemble the second derivative matrix for the 3-forms. Formula:
            dd3 = - strong_div @ weak_grad
        """
        self.dd3 = -self.strong_div @ self.weak_grad

    def assemble_P12(self):
        """
        Assemble the projection matrix for the 1-forms to the 2-forms. Formula:
            M1 @ P12 = E1 @ (∫ Λ1_ijk Λ2_ijk^T detD) @ E2.T
        """
        W = self.Q.w[:, None, None] * jnp.eye(3)  # shape (n_q, 1, 1)
        M = assemble(self.get_Λ1_ijk, self.get_Λ2_ijk, W, self.Λ1.n, self.Λ2.n)
        M12 = self.E1 @ M @ self.E2.T
        self.P12 = jnp.linalg.solve(self.M1, M12)

    def assemble_P03(self):
        """
        Assemble the projection matrix for the 0-forms to the 3-forms. Formula:
            M0 @ P03 = E0 @ (∫ Λ0_ijk Λ3_ijk^T detD) @ E3.T
        """
        W = self.Q.w[:, None, None]  # shape (n_q, 1, 1)
        M = assemble(self.get_Λ0_ijk, self.get_Λ3_ijk, W, self.Λ0.n, self.Λ3.n)
        M03 = self.E0 @ M @ self.E3.T
        self.P03 = jnp.linalg.solve(self.M0, M03)

    def build_crossproduct_projections(self):
        """
        Assemble the cross product projections for the 1-forms to the 1-forms. 
        """
        # self.P1x1_to_1 = CrossProductProjection(1, 1, 1, self)
        # self.P1x2_to_1 = CrossProductProjection(1, 1, 2, self)
        self.P2x1_to_1 = CrossProductProjection(1, 2, 1, self)
        # self.P2x2_to_1 = CrossProductProjection(1, 2, 2, self)

        self.P1x1_to_2 = CrossProductProjection(2, 1, 1, self)
        # self.P1x2_to_2 = CrossProductProjection(2, 1, 2, self)
        # self.P2x1_to_2 = CrossProductProjection(2, 2, 1, self)
        # self.P2x1_to_2 = CrossProductProjection(2, 2, 1, self)

    def assemble_all(self):
        """
        Assemble all the matrices and operators.
        """
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
        """
        Assemble the Leray projection matrix. Formula:
            P_Leray = I + weak_grad @ (dd3)^-1 @ strong_div
        """
        self.P_Leray = jnp.eye(self.M2.shape[0]) + \
            self.weak_grad @ jnp.linalg.pinv(self.dd3) @ self.strong_div
