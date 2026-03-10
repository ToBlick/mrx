from typing import Callable

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

import mrx
from mrx.boundary import BoundaryOperator
from mrx.differential_forms import DifferentialForm
from mrx.nonlinearities import CrossProductProjection
from mrx.polar import ExtractionOperator, get_xi
from mrx.projectors import Projector
from mrx.quadrature import QuadratureRule
from mrx.utils import (
    assemble,
    assemble_sparse,
    build_neighbors,
    extract_diag_vector,
    inv33,
    jacobian_determinant,
    solve_singular_cg,
    square_bcoo,
)


class DeRhamSequence():
    """
    A class to represent a de Rham sequence.

    Attributes:
        Lambda_0, Lambda_1, Lambda_2, Lambda_3: DifferentialForm for 0-forms in the sequence
        Q: QuadratureRule for numerical integration
        F: Callable mapping from logical to physical coordinates
        E0, E1, E2, E3: jnp.ndarray representing the assembled constraint/extraction operators
        lambda_r_jk: jnp.ndarray of radial basis splines evaluated at radial quadrature points. Shape: n_r x n_qr.
        lambda_t_jk: jnp.ndarray of poloidal basis splines evaluated at poloidal quadrature points. Shape: n_θ x n_qθ.
        lambda_z_jk: jnp.ndarray of toroidal basis splines evaluated at toroidal quadrature points. Shape: n_ζ x n_qζ.
        d_lambda_r_jk: jnp.ndarray of radial derivative splines evaluated at radial quadrature points. Shape: n_r x n_qr.
        d_lambda_t_jk: jnp.ndarray of poloidal derivative splines evaluated at poloidal quadrature points. Shape: n_θ x n_qθ.
        d_lambda_z_jk: jnp.ndarray of toroidal derivative splines evaluated at toroidal quadrature points. Shape: n_ζ x n_qζ.
        J_j: jnp.ndarray of mapping Jacobian determinant at quad. pts. det DF(x_j). Shape: n_q.
        G_jkl, inv_G_jkl: jnp.ndarray of mapping metric at quad. pts. [ DF(x_j).T DF(x_j) ]_kl and its inverse. Shape: n_q x 3 x 3.
    """
    basis_0: DifferentialForm
    basis_1: DifferentialForm
    basis_2: DifferentialForm
    basis_3: DifferentialForm
    quad: QuadratureRule
    map: Callable
    e0: jsparse.BCOO
    e1: jsparse.BCOO
    e2: jsparse.BCOO
    e3: jsparse.BCOO
    e0_dbc: jsparse.BCOO
    e1_dbc: jsparse.BCOO
    e2_dbc: jsparse.BCOO
    e3_dbc: jsparse.BCOO
    basis_r_jk: jnp.ndarray
    basis_t_jk: jnp.ndarray
    basis_z_jk: jnp.ndarray
    d_basis_r_jk: jnp.ndarray
    d_basis_t_jk: jnp.ndarray
    d_basis_z_jk: jnp.ndarray

    # Jacobian determinant evaluated at quadrature points: det DF(x_j). Shape: n_q x 1.
    jacobian_j: jnp.ndarray
    # (k,l)th element of metric at quadrature point j: G(x_j)_kl. Shape: n_q x 3 x 3. G = DF^T DF.
    metric_jkl: jnp.ndarray
    # (k,l)th element of inverse metric at quadrature point j: G(x_j)^{-1}_kl. Shape: n_q x 3 x 3.
    metric_inv_jkl: jnp.ndarray

    def __init__(self, ns, ps, q, types, map, polar, tol=1e-9, maxiter=100, r_scale=1.0):
        """
        Initialize the de Rham sequence.

        Args:
            ns (list): List of integers representing the number of basis functions for each differential form.
            ps (list): List of integers representing the order of the basis functions for each differential form.
            q (int): The order of the quadrature rule.
            types (list): List of strings representing the type of boundary condition for each differential form.
            map (callable): The mapping function from logical to physical domain.
            polar (bool): Whether to use polar coordinates.
            tol (float): Tolerance for sparse linear solvers.
            maxiter (int): Maximum number of iterations for sparse linear solvers.
            r_scale (float): Scale factor for the radial coordinate.
        """
        self.tol = tol
        self.maxiter = maxiter
        if not polar:
            Ts = [None] * 3
        else:
            Tr = jnp.concatenate([
                jnp.zeros(ps[0]),
                jnp.linspace(0, 1, ns[0]-ps[0]+1)**r_scale,
                jnp.ones(ps[0])
            ])
            Ts = [Tr, None, None]

        self.basis_0, self.basis_1, self.basis_2, self.basis_3 = [
            DifferentialForm(i, ns, ps, types, Ts) for i in range(0, 4)
        ]
        self.quad = QuadratureRule(self.basis_0, q)
        # Mapping from logical to physical coordinates
        self.map = map

        def G(x):
            return jax.jacfwd(self.map)(x).T @ jax.jacfwd(self.map)(x)

        self.metric_jkl = jax.lax.map(
            G, self.quad.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
        self.metric_inv_jkl = jax.lax.map(
            inv33, self.metric_jkl, batch_size=mrx.MAP_BATCH_SIZE_INNER)
        self.jacobian_j = jax.lax.map(jacobian_determinant(
            self.map), self.quad.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)

        if polar:
            xi = get_xi(ns[1])
            e0, e1, e2, e3 = [
                ExtractionOperator(Λ, xi, False)
                for Λ in [self.basis_0, self.basis_1, self.basis_2, self.basis_3]
            ]
            e0_dbc, e1_dbc, e2_dbc, e3_dbc = [
                ExtractionOperator(Λ, xi, True)
                for Λ in [self.basis_0, self.basis_1, self.basis_2, self.basis_3]
            ]

        else:
            # TODO: right now, we only support dirichlet BCs in r
            e0, e1, e2, e3 = [
                BoundaryOperator(
                    Λ, ('none', 'none', 'none'))
                for Λ in [self.basis_0, self.basis_1, self.basis_2, self.basis_3]
                ]
            e0_dbc, e1_dbc, e2_dbc, e3_dbc = [
                BoundaryOperator(
                    Λ, ('dirichlet', 'none', 'none'))
                for Λ in [self.basis_0, self.basis_1, self.basis_2, self.basis_3]
                ]
        self.e0 = e0.assemble_sparse()
        self.e0_dbc = e0_dbc.assemble_sparse()
        self.n0 = e0.n
        self.n0_dbc = e0_dbc.n
        self.n0_1, self.n0_2, self.n0_3 = e0.n, 0, 0
        self.n0_1_dbc, self.n0_2_dbc, self.n0_3_dbc = e0_dbc.n, 0, 0
        self.e1 = e1.assemble_sparse()
        self.e1_dbc = e1_dbc.assemble_sparse()
        self.n1 = e1.n
        self.n1_dbc = e1_dbc.n
        self.n1_1, self.n1_2, self.n1_3 = e1.n1, e1.n2, e1.n2
        self.n1_1_dbc, self.n1_2_dbc, self.n1_3_dbc = e1_dbc.n1, e1_dbc.n2, e1_dbc.n2
        self.e2 = e2.assemble_sparse()
        self.e2_dbc = e2_dbc.assemble_sparse()
        self.n2 = e2.n
        self.n2_dbc = e2_dbc.n
        self.n2_1, self.n2_2, self.n2_3 = e2.n1, e2.n2, e2.n3
        self.n2_1_dbc, self.n2_2_dbc, self.n2_3_dbc = e2_dbc.n1, e2_dbc.n2, e2_dbc.n3
        self.e3 = e3.assemble_sparse()
        self.e3_dbc = e3_dbc.assemble_sparse()
        self.n3 = e3.n
        self.n3_dbc = e3_dbc.n
        self.n3_1, self.n3_2, self.n3_3 = e3.n1, e3.n2, e3.n3
        self.n3_1_dbc, self.n3_2_dbc, self.n3_3_dbc = e3_dbc.n1, e3_dbc.n2, e3_dbc.n3

        self.p0, self.p1, self.p2, self.p3 = [
            Projector(self, k, False) for k in range(4)
        ]
        
        self.p0_dbc, self.p1_dbc, self.p2_dbc, self.p3_dbc = [
            Projector(self, k, True) for k in range(4)
        ]

    def evaluate_1d(self):
        """
        Evaluate the 1-dimensional basis functions at the quadrature points.
        """
        # TODO: This should really be fine as double vmap since it is all 1D.
        # Consider replacing with jax.lax.map if we ever run into memory issues.
        self.basis_r_jk = jax.vmap(jax.vmap(self.basis_0.Λ[0], (0, None)),
                                   (None, 0))(self.quad.x_x, self.basis_0.Λ[0].ns)
        self.basis_t_jk = jax.vmap(jax.vmap(self.basis_0.Λ[1], (0, None)),
                                   (None, 0))(self.quad.x_y, self.basis_0.Λ[1].ns)
        self.basis_z_jk = jax.vmap(jax.vmap(self.basis_0.Λ[2], (0, None)),
                                   (None, 0))(self.quad.x_z, self.basis_0.Λ[2].ns)
        self.d_basis_r_jk = jax.vmap(jax.vmap(self.basis_0.dΛ[0], (0, None)),
                                     (None, 0))(self.quad.x_x, self.basis_0.dΛ[0].ns)
        self.d_basis_t_jk = jax.vmap(jax.vmap(self.basis_0.dΛ[1], (0, None)),
                                     (None, 0))(self.quad.x_y, self.basis_0.dΛ[1].ns)
        self.d_basis_z_jk = jax.vmap(jax.vmap(self.basis_0.dΛ[2], (0, None)),
                                     (None, 0))(self.quad.x_z, self.basis_0.dΛ[2].ns)

    def eval_basis_0_ijk(self, i, j, k):
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
        j2, j1, j3 = jnp.unravel_index(
            j, (self.quad.ny, self.quad.nx, self.quad.nz))

        # get the 1d basis functions
        _, i1, i2, i3 = self.basis_0._unravel_index(i)
        # k is always 0
        return self.basis_r_jk[i1, j1] * self.basis_t_jk[i2, j2] * self.basis_z_jk[i3, j3]

    def eval_d_basis_0_ijk(self, i, j, k):
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
        j2, j1, j3 = jnp.unravel_index(
            j, (self.quad.ny, self.quad.nx, self.quad.nz))
        _, i1, i2, i3 = self.basis_0._unravel_index(i)
        # get i-1
        dr = jnp.where(i1 == self.basis_0.nt-1, 0.0,
                       self.d_basis_r_jk[i1, j1])
        dr_m1 = jnp.where(i1 > 0, self.d_basis_r_jk[i1-1, j1], 0.0)
        dtheta_m1 = jnp.where(
            i2 > 0, self.d_basis_t_jk[i2-1, j2], self.d_basis_t_jk[self.basis_0.nt-1, j2])
        dtheta = self.d_basis_t_jk[i2, j2]
        dz_m1 = jnp.where(
            i3 > 0, self.d_basis_z_jk[i3-1, j3], self.d_basis_z_jk[self.basis_0.nt-1, j3])
        dz = self.d_basis_z_jk[i3, j3]
        return jnp.where(k == 0,
                         (dr_m1 - dr) *
                         self.basis_t_jk[i2, j2] * self.basis_z_jk[i3, j3],
                         jnp.where(k == 1,
                                   self.basis_r_jk[i1, j1] *
                                   (dtheta_m1 - dtheta) *
                                   self.basis_z_jk[i3, j3],
                                   self.basis_r_jk[i1, j1] * self.basis_t_jk[i2, j2] * (dz_m1 - dz)))

    def eval_basis_1_ijk(self, i, j, k):
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
        j2, j1, j3 = jnp.unravel_index(
            j, (self.quad.ny, self.quad.nx, self.quad.nz))
        c, i1, i2, i3 = self.basis_1._unravel_index(i)
        return jnp.where(k == c,
                         jnp.where(k == 0,
                                   self.d_basis_r_jk[i1, j1] *
                                   self.basis_t_jk[i2, j2] *
                                   self.basis_z_jk[i3, j3],
                                   jnp.where(k == 1,
                                             self.basis_r_jk[i1, j1] *
                                             self.d_basis_t_jk[i2, j2] *
                                             self.basis_z_jk[i3, j3],
                                             self.basis_r_jk[i1, j1] * self.basis_t_jk[i2, j2] * self.d_basis_z_jk[i3, j3])),
                         0.0)

    def eval_d_basis_1_ijk(self, i, j, k):
        """
        Get the kth component of the curl of the ith 1-form evaluated at quadrature point j.

        Args:
            i (int): The index of the basis function.
            j (int): The index of the quadrature point.
            k (int): The index of the component.

        Returns:
            float: The value of the kth component of the curl of the ith 1-form evaluated at quadrature point j.
        """
        j2, j1, j3 = jnp.unravel_index(
            j, (self.quad.ny, self.quad.nx, self.quad.nz))
        c, i1, i2, i3 = self.basis_1._unravel_index(i)
        # get i-1
        dr = jnp.where(i1 == self.basis_1.nt-1, 0.0,
                       self.d_basis_r_jk[i1, j1])
        dr_m1 = jnp.where(i1 > 0, self.d_basis_r_jk[i1-1, j1], 0.0)
        dtheta_m1 = jnp.where(
            i2 > 0, self.d_basis_t_jk[i2-1, j2], self.d_basis_t_jk[self.basis_1.nt-1, j2])
        dtheta = self.d_basis_t_jk[i2, j2]
        dz_m1 = jnp.where(
            i3 > 0, self.d_basis_z_jk[i3-1, j3], self.d_basis_z_jk[self.basis_1.nt-1, j3])
        dz = self.d_basis_z_jk[i3, j3]
        d3dy = self.basis_r_jk[i1, j1] * \
            (dtheta_m1 - dtheta) * self.d_basis_z_jk[i3, j3]
        d2dz = self.basis_r_jk[i1, j1] * \
            self.d_basis_t_jk[i2, j2] * (dz_m1 - dz)
        d1dz = self.d_basis_r_jk[i1, j1] * \
            self.basis_t_jk[i2, j2] * (dz_m1 - dz)
        d3dx = (dr_m1 - dr) * \
            self.basis_t_jk[i2, j2] * self.d_basis_z_jk[i3, j3]
        d2dx = (dr_m1 - dr) * \
            self.d_basis_t_jk[i2, j2] * self.basis_z_jk[i3, j3]
        d1dy = self.d_basis_r_jk[i1, j1] * \
            (dtheta_m1 - dtheta) * self.basis_z_jk[i3, j3]

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
                                             ),
                                   # c==2
                                   jnp.where(k == 0,
                                             d3dy,
                                             jnp.where(k == 1,
                                                       -d3dx,
                                                       0.0)
                                             )
                                   )
                         )

    def eval_basis_2_ijk(self, i, j, k):
        """
        Get the kth component of the ith 2-form evaluated at quadrature point j.

        Args:
            i (int): The index of the basis function.
            j (int): The index of the quadrature point.
            k (int): The index of the component.

        Returns:
            float: The value of the kth component of the ith 2-form evaluated at quadrature point j.
        """
        j2, j1, j3 = jnp.unravel_index(
            j, (self.quad.ny, self.quad.nx, self.quad.nz))
        c, i1, i2, i3 = self.basis_2._unravel_index(i)
        return jnp.where(k == c,
                         jnp.where(k == 0,
                                   self.basis_r_jk[i1, j1] *
                                   self.d_basis_t_jk[i2, j2] *
                                   self.d_basis_z_jk[i3, j3],
                                   jnp.where(k == 1,
                                             self.d_basis_r_jk[i1, j1] *
                                             self.basis_t_jk[i2, j2] *
                                             self.d_basis_z_jk[i3, j3],
                                             self.d_basis_r_jk[i1, j1] * self.d_basis_t_jk[i2, j2] * self.basis_z_jk[i3, j3])),
                         0.0)

    def eval_d_basis_2_ijk(self, i, j, k):
        """
        Get the kth component of the divergence of the ith 2-form evaluated at quadrature point j.

        Args:
            i (int): The index of the basis function.
            j (int): The index of the quadrature point.
            k (int): The index of the component.

        Returns:
            float: The value of the kth component of the divergence of the ith 2-form evaluated at quadrature point j.
        """
        j2, j1, j3 = jnp.unravel_index(
            j, (self.quad.ny, self.quad.nx, self.quad.nz))
        c, i1, i2, i3 = self.basis_2._unravel_index(i)
        # get i-1
        dr = jnp.where(i1 == self.basis_2.nt-1, 0.0,
                       self.d_basis_r_jk[i1, j1])
        dr_m1 = jnp.where(i1 > 0, self.d_basis_r_jk[i1-1, j1], 0.0)
        dtheta_m1 = jnp.where(
            i2 > 0, self.d_basis_t_jk[i2-1, j2], self.d_basis_t_jk[self.basis_2.nt-1, j2])
        dtheta = self.d_basis_t_jk[i2, j2]
        dz_m1 = jnp.where(
            i3 > 0, self.d_basis_z_jk[i3-1, j3], self.d_basis_z_jk[self.basis_2.nt-1, j3])
        dz = self.d_basis_z_jk[i3, j3]

        return jnp.where(c == 0,
                         (dr_m1 - dr) *
                         self.d_basis_t_jk[i2, j2] *
                         self.d_basis_z_jk[i3, j3],
                         jnp.where(c == 1,
                                   self.d_basis_r_jk[i1, j1] *
                                   (dtheta_m1 - dtheta) *
                                   self.d_basis_z_jk[i3, j3],
                                   self.d_basis_r_jk[i1, j1] *
                                   self.d_basis_t_jk[i2, j2] * (dz_m1 - dz)
                                   )
                         )

    def eval_basis_3_ijk(self, i, j, k):
        """
        Get the kth component of the ith 3-form evaluated at quadrature point j.

        Args:
            i (int): The index of the basis function.
            j (int): The index of the quadrature point.
            k (int): The index of the component.

        Returns:
            float: The value of the kth component of the ith 3-form evaluated at quadrature point j.
        """
        j2, j1, j3 = jnp.unravel_index(
            j, (self.quad.ny, self.quad.nx, self.quad.nz))
        _, i1, i2, i3 = self.basis_3._unravel_index(i)
        # k is always 0
        return self.d_basis_r_jk[i1, j1] * self.d_basis_t_jk[i2, j2] * self.d_basis_z_jk[i3, j3]

    def assemble_m0(self):
        """
        Assemble mass matrix for 0-forms.
            M0_ij = ∫ Λ0_i Λ0_j det DF dx
        """
        W = (self.jacobian_j * self.quad.w)[:, None, None]  # shape (n_q, 1, 1)
        M = assemble(self.eval_basis_0_ijk, self.eval_basis_0_ijk,
                     W, self.basis_0.n, self.basis_0.n)
        self.m0 = self.e0 @ M @ self.e0.T
        self.m0_dbc = self.e0_dbc @ M @ self.e0_dbc.T

    def assemble_m0_sparse(self):
        """
        Assemble mass matrix for 0-forms in sparse format.
            M0_ij = ∫ Λ0_i Λ0_j det DF dx
        """
        W = (self.jacobian_j * self.quad.w)[:, None, None]  # shape (n_q, 1, 1)
        neighbors, nnz = build_neighbors(self.basis_0)
        self.m0_sp = assemble_sparse(self.eval_basis_0_ijk, self.eval_basis_0_ijk,
                                     W, self.basis_0.n, self.basis_0.n, nnz, neighbors)
        self.m0_sp_diaginv = 1 / (square_bcoo(
            self.e0) @ extract_diag_vector(self.m0_sp))
        self.m0_sp_diaginv_dbc = 1 / (square_bcoo(
            self.e0_dbc) @ extract_diag_vector(self.m0_sp))

    def apply_m0_sparse(self, v, dirichlet=True):
        """
        Apply the sparse mass matrix for 0-forms to a vector v.
        """
        if dirichlet:
            return self.e0_dbc @ (self.m0_sp @ (self.e0_dbc.T @ v))
        else:
            return self.e0 @ (self.m0_sp @ (self.e0.T @ v))

    def apply_m0_precond(self, v, dirichlet=True):
        """
        Apply the diagonal preconditioner for the mass matrix of 0-forms to a vector v.
        """
        if dirichlet:
            return self.m0_sp_diaginv_dbc * v
        else:
            return self.m0_sp_diaginv * v

    def assemble_m1(self):
        """
        Assemble mass matrix for 1-forms.
            M1_ij = ∫ Λ1_i · G⁻¹ Λ1_j det DF dx
        """
        W = self.metric_inv_jkl * \
            (self.jacobian_j * self.quad.w)[:, None, None]
        M = assemble(self.eval_basis_1_ijk, self.eval_basis_1_ijk,
                     W, self.basis_1.n, self.basis_1.n)
        self.m1 = self.e1 @ M @ self.e1.T
        self.m1_dbc = self.e1_dbc @ M @ self.e1_dbc.T

    def assemble_m1_sparse(self):
        """
        Assemble mass matrix for 1-forms in sparse format.
            M1_ij = ∫ Λ1_i · G⁻¹ Λ1_j det DF dx
        """
        W = self.metric_inv_jkl * \
            (self.jacobian_j * self.quad.w)[:, None, None]
        neighbors, nnz = build_neighbors(self.basis_1)
        self.m1_sp = assemble_sparse(self.eval_basis_1_ijk, self.eval_basis_1_ijk,
                                     W, self.basis_1.n, self.basis_1.n, nnz, neighbors)
        self.m1_sp_diaginv = 1 / (square_bcoo(
            self.e1) @ extract_diag_vector(self.m1_sp))
        self.m1_sp_diaginv_dbc = 1 / (square_bcoo(
            self.e1_dbc) @ extract_diag_vector(self.m1_sp))
    
    def apply_m1_sparse(self, v, dirichlet=True):
        """
        Apply the sparse mass matrix for 1-forms to a vector v.
        """
        if dirichlet:
            return self.e1_dbc @ (self.m1_sp @ (self.e1_dbc.T @ v))
        else:
            return self.e1 @ (self.m1_sp @ (self.e1.T @ v))
    
    def apply_m1_precond(self, v, dirichlet=True):
        """
        Apply the diagonal preconditioner for the mass matrix of 1-forms to a vector v.
        """
        if dirichlet:
            return self.m1_sp_diaginv_dbc * v
        else:
            return self.m1_sp_diaginv * v

    def assemble_m2(self):
        """
        Assemble mass matrix for 2-forms.
            M2_ij = ∫ Λ2_i · G Λ2_j (det DF)⁻¹ dx
        """
        W = self.metric_jkl * (1/self.jacobian_j * self.quad.w)[:, None, None]
        M = assemble(self.eval_basis_2_ijk, self.eval_basis_2_ijk,
                     W, self.basis_2.n, self.basis_2.n)
        self.m2 = self.e2 @ M @ self.e2.T
        self.m2_dbc = self.e2_dbc @ M @ self.e2_dbc.T
        
    def assemble_m2_sparse(self):
        """
        Assemble mass matrix for 2-forms in sparse format.
            M2_ij = ∫ Λ2_i · G Λ2_j (det DF)⁻¹ dx
        """
        W = self.metric_jkl * (1/self.jacobian_j * self.quad.w)[:, None, None]
        neighbors, nnz = build_neighbors(self.basis_2)
        self.m2_sp = assemble_sparse(self.eval_basis_2_ijk, self.eval_basis_2_ijk,
                                     W, self.basis_2.n, self.basis_2.n, nnz, neighbors)
        self.m2_sp_diaginv = 1 / (square_bcoo(
            self.e2) @ extract_diag_vector(self.m2_sp))
        self.m2_sp_diaginv_dbc = 1 / (square_bcoo(
            self.e2_dbc) @ extract_diag_vector(self.m2_sp))

    def apply_m2_sparse(self, v, dirichlet=True):
        """
        Apply the sparse mass matrix for 2-forms to a vector v.
        """
        if dirichlet:
            return self.e2_dbc @ (self.m2_sp @ (self.e2_dbc.T @ v))
        else:
            return self.e2 @ (self.m2_sp @ (self.e2.T @ v))

    def apply_m2_precond(self, v, dirichlet=True):
        """
        Apply the diagonal preconditioner for the mass matrix of 2-forms to a vector v.
        """
        if dirichlet:
            return self.m2_sp_diaginv_dbc * v
        else:
            return self.m2_sp_diaginv * v

    def assemble_m3(self):
        """
        Assemble mass matrix for 3-forms.
            M3_ij = ∫ Λ3_i Λ3_j (det DF)⁻¹ dx
        """
        W = (1/self.jacobian_j * self.quad.w)[:, None, None]
        M = assemble(self.eval_basis_3_ijk, self.eval_basis_3_ijk,
                     W, self.basis_3.n, self.basis_3.n)
        self.m3 = self.e3 @ M @ self.e3.T
        self.m3_dbc = self.e3_dbc @ M @ self.e3_dbc.T

    def assemble_m3_sparse(self):
        """
        Assemble mass matrix for 3-forms in sparse format.
            M3_ij = ∫ Λ3_i Λ3_j (det DF)⁻¹ dx
        """
        W = (1/self.jacobian_j * self.quad.w)[:, None, None]
        neighbors, nnz = build_neighbors(self.basis_3)
        self.m3_sp = assemble_sparse(self.eval_basis_3_ijk, self.eval_basis_3_ijk,
                                     W, self.basis_3.n, self.basis_3.n, nnz, neighbors)
        self.m3_sp_diaginv = 1 / (square_bcoo(
            self.e3) @ extract_diag_vector(self.m3_sp))
        self.m3_sp_diaginv_dbc = 1 / (square_bcoo(
            self.e3_dbc) @ extract_diag_vector(self.m3_sp))

    def apply_m3_sparse(self, v, dirichlet=True):
        """
        Apply the sparse mass matrix for 3-forms to a vector v.
        """
        if dirichlet:
            return self.e3_dbc @ (self.m3_sp @ (self.e3_dbc.T @ v))
        else:
            return self.e3 @ (self.m3_sp @ (self.e3.T @ v))

    def apply_m3_precond(self, v, dirichlet=True):
        """
        Apply the diagonal preconditioner for the mass matrix of 3-forms to a vector v.
        """
        if dirichlet:
            return self.m3_sp_diaginv_dbc * v
        else:
            return self.m3_sp_diaginv * v

    def assemble_d0(self):
        """
        Assemble derivative matrices for 0-form dofs.
            D0_ij = ∫ Λ1_i · G⁻¹ grad Λ0_j det DF dx
        from this, get strong grad and weak div operators:
            v.T D0 f = (v, grad f) =: v.T M1 strong_grad f => strong_grad = M1⁻¹ D0
                    = -(div v, f) =: -(weak_div v).T M0 f => weak_div = -M0⁻¹ D0.T
        """
        W = self.metric_inv_jkl * \
            (self.jacobian_j * self.quad.w)[:, None, None]
        M = assemble(self.eval_basis_1_ijk, self.eval_d_basis_0_ijk,
                     W, self.basis_1.n, self.basis_0.n)
        self.d0 = self.e1 @ M @ self.e0.T
        self.strong_grad = jnp.linalg.solve(self.m1, self.d0)
        self.weak_div = -jnp.linalg.solve(self.m0.T, self.d0.T)

    # TODO: nnz might not be optimal for the derivative matrices
    def assemble_d0_sparse(self):
        """
        Assemble derivative matrix for 0-form dofs in sparse format.
            D0_ij = ∫ Λ1_i · G⁻¹ grad Λ0_j det DF dx
        """
        W = self.metric_inv_jkl * \
            (self.jacobian_j * self.quad.w)[:, None, None]
        neighbors, nnz = build_neighbors(self.basis_1, self.basis_0)
        self.d0_sp = assemble_sparse(self.eval_basis_1_ijk, self.eval_d_basis_0_ijk,
                                     W, self.basis_1.n, self.basis_0.n, nnz, neighbors)

    def apply_d0_sparse(self, v, dirichlet_in=True, dirichlet_out=True):
        """
        Apply the sparse derivative matrix D0 to a vector v.
        """
        if dirichlet_out and dirichlet_in:
            return self.e1_dbc @ (self.d0_sp @ (self.e0_dbc.T @ v))
        elif dirichlet_out and not dirichlet_in:
            return self.e1_dbc @ (self.d0_sp @ (self.e0.T @ v))
        elif not dirichlet_out and dirichlet_in:
            return self.e1 @ (self.d0_sp @ (self.e0_dbc.T @ v))
        else:
            return self.e1 @ (self.d0_sp @ (self.e0.T @ v))

    def apply_d0t_sparse(self, v, dirichlet_in=True, dirichlet_out=True):
        """
        Apply the transpose of the sparse derivative matrix D0 to a vector v.
        """
        if dirichlet_out and dirichlet_in:
            return self.e0_dbc @ (self.d0_sp.T @ (self.e1_dbc.T @ v))
        elif dirichlet_out and not dirichlet_in:
            return self.e0_dbc @ (self.d0_sp.T @ (self.e1.T @ v))
        elif not dirichlet_out and dirichlet_in:
            return self.e0 @ (self.d0_sp.T @ (self.e1_dbc.T @ v))
        else:
            return self.e0 @ (self.d0_sp.T @ (self.e1.T @ v))

    def assemble_d1(self):
        """
        Assemble derivative matrices for 1-form dofs.
            D1_ij = ∫ Λ2_i · G curl Λ1_j (det DF)⁻¹ dx
        from this, get strong curl and weak curl operators:
            ω.T D1 v = (ω, curl v) =: ω.T M2 strong_curl v => strong_curl = M2⁻¹ D1
                     = (curl ω, v) =: (weak_curl ω).T M1 v => weak_curl = M1⁻¹ D1.T
        """
        W = self.metric_jkl * (1/self.jacobian_j * self.quad.w)[:, None, None]
        M = assemble(self.eval_basis_2_ijk, self.eval_d_basis_1_ijk,
                     W, self.basis_2.n, self.basis_1.n)
        self.d1 = self.e2 @ M @ self.e1.T
        self.strong_curl = jnp.linalg.solve(self.m2, self.d1)
        self.weak_curl = jnp.linalg.solve(self.m1.T, self.d1.T)

    def assemble_d1_sparse(self):
        """
        Assemble derivative matrix for 1-form dofs in sparse format.
            D1_ij = ∫ Λ2_i · G curl Λ1_j (det DF)⁻¹ dx
        """
        W = self.metric_jkl * (1/self.jacobian_j * self.quad.w)[:, None, None]
        neighbors, nnz = build_neighbors(self.basis_2, self.basis_1)
        self.d1_sp = assemble_sparse(self.eval_basis_2_ijk, self.eval_d_basis_1_ijk,
                                     W, self.basis_2.n, self.basis_1.n, nnz, neighbors)

    def apply_d1_sparse(self, v, dirichlet_in=True, dirichlet_out=True):
        """
        Apply the sparse derivative matrix D1 to a vector v.
        """
        if dirichlet_out and dirichlet_in:
            return self.e2_dbc @ (self.d1_sp @ (self.e1_dbc.T @ v))
        elif dirichlet_out and not dirichlet_in:
            return self.e2_dbc @ (self.d1_sp @ (self.e1.T @ v))
        elif not dirichlet_out and dirichlet_in:
            return self.e2 @ (self.d1_sp @ (self.e1_dbc.T @ v))
        else:
            return self.e2 @ (self.d1_sp @ (self.e1.T @ v))

    def apply_d1t_sparse(self, v, dirichlet_in=True, dirichlet_out=True):
        """
        Apply the transpose of the sparse derivative matrix D1 to a vector v.
        """
        if dirichlet_out and dirichlet_in:
            return self.e1_dbc @ (self.d1_sp.T @ (self.e2_dbc.T @ v))
        elif dirichlet_out and not dirichlet_in:
            return self.e1_dbc @ (self.d1_sp.T @ (self.e2.T @ v))
        elif not dirichlet_out and dirichlet_in:
            return self.e1 @ (self.d1_sp.T @ (self.e2_dbc.T @ v))
        else:
            return self.e1 @ (self.d1_sp.T @ (self.e2.T @ v))

    def assemble_d2(self):
        """
        Assemble derivative matrices for 1-form dofs.
            D2_ij = ∫ Λ2_i div Λ1_j (det DF)⁻¹ dx
        from this, get strong div and weak grad operators:
            ρ.T D2 ω = (ρ, div ω) =: ρ.T M3 strong_div ω => strong_div = M3⁻¹ D2
                     = -(grad ρ, ω) =: -(weak_grad ρ).T M2 ω => weak_grad = -M2⁻¹ D2.T
        """
        W = (1/self.jacobian_j * self.quad.w)[:, None, None]
        M = assemble(self.eval_basis_3_ijk, self.eval_d_basis_2_ijk,
                     W, self.basis_3.n, self.basis_2.n)
        self.d2 = self.e3 @ M @ self.e2.T
        self.strong_div = jnp.linalg.solve(self.m3, self.d2)
        self.weak_grad = -jnp.linalg.solve(self.m2.T, self.d2.T)

    def assemble_d2_sparse(self):
        """
        Assemble derivative matrix for 2-form dofs in sparse format.
            D2_ij = ∫ Λ3_i div Λ2_j (det DF)⁻¹ dx
        """
        W = (1/self.jacobian_j * self.quad.w)[:, None, None]
        neighbors, nnz = build_neighbors(self.basis_3, self.basis_2)
        self.d2_sp = assemble_sparse(self.eval_basis_3_ijk, self.eval_d_basis_2_ijk,
                                     W, self.basis_3.n, self.basis_2.n, nnz, neighbors)

    def apply_d2_sparse(self, v, dirichlet_in=True, dirichlet_out=True):
        """
        Apply the sparse derivative matrix D2 to a vector v.
        """
        if dirichlet_out and dirichlet_in:
            return self.e3_dbc @ (self.d2_sp @ (self.e2_dbc.T @ v))
        elif dirichlet_out and not dirichlet_in:
            return self.e3_dbc @ (self.d2_sp @ (self.e2.T @ v))
        elif not dirichlet_out and dirichlet_in:
            return self.e3 @ (self.d2_sp @ (self.e2_dbc.T @ v))
        else:
            return self.e3 @ (self.d2_sp @ (self.e2.T @ v))

    def apply_d2t_sparse(self, v, dirichlet_in=True, dirichlet_out=True):
        """
        Apply the sparse derivative matrix D2 to a vector v.
        """
        if dirichlet_out and dirichlet_in:
            return self.e2_dbc @ (self.d2_sp.T @ (self.e3_dbc.T @ v))
        elif dirichlet_out and not dirichlet_in:
            return self.e2_dbc @ (self.d2_sp.T @ (self.e3.T @ v))
        elif not dirichlet_out and dirichlet_in:
            return self.e2 @ (self.d2_sp.T @ (self.e3_dbc.T @ v))
        else:
            return self.e2 @ (self.d2_sp.T @ (self.e3.T @ v))

    def assemble_dd0(self):
        """
        Assemble Hodge-Laplacian for 0-form dofs:
            (grad f, grad g) = (f, δdg) ∀g
            => δd = M0⁻¹ grad_grad
        where
            grad_grad_ij = ∫ grad Λ0_i · G⁻¹ grad Λ0_j det DF dx
        """
        W = self.metric_inv_jkl * \
            (self.jacobian_j * self.quad.w)[:, None, None]
        grad_grad = assemble(self.eval_d_basis_0_ijk, self.eval_d_basis_0_ijk,
                             W, self.basis_0.n, self.basis_0.n)
        self.dd0 = jnp.linalg.solve(self.m0, self.e0 @ grad_grad @ self.e0.T)

    def assemble_dd0_sparse(self):
        """
        Assemble grad-grad matrix for 0-form dofs in sparse format.
            grad_grad_ij = ∫ grad Λ0_i · G⁻¹ grad Λ0_j det DF dx
        """
        W = self.metric_inv_jkl * \
            (self.jacobian_j * self.quad.w)[:, None, None]
        neighbors, nnz = build_neighbors(self.basis_0)
        self.grad_grad_sp = assemble_sparse(self.eval_d_basis_0_ijk, self.eval_d_basis_0_ijk,
                                            W, self.basis_0.n, self.basis_0.n, nnz, neighbors)

        self.dd0_sp_diaginv = 1 / (square_bcoo(
            self.e0) @ extract_diag_vector(self.grad_grad_sp))
        self.dd0_sp_diaginv_dbc = 1 / (square_bcoo(
            self.e0_dbc) @ extract_diag_vector(self.grad_grad_sp))

    def apply_grad_grad_sparse(self, v, dirichlet=True):
        """
        Apply the sparse grad-grad matrix to a vector v.
        """
        if dirichlet:
            return self.e0_dbc @ (self.grad_grad_sp @ (self.e0_dbc.T @ v))
        else:
            return self.e0 @ (self.grad_grad_sp @ (self.e0.T @ v))

    def apply_dd0_sparse(self, v, dirichlet=True):
        """
        Forward application of the k=0 Hodge Laplacian
        """
        return self.apply_grad_grad_sparse(v, dirichlet=dirichlet)

    def apply_dd0_precond(self, v, dirichlet=True):
        if dirichlet:
            return self.dd0_sp_diaginv_dbc * v
        else:
            return self.dd0_sp_diaginv * v

    def assemble_dd1(self):
        """
        Assemble Hodge-Laplacian for 1-form dofs:
            (curl v, curl u) - (grad ω, u) = (δdv, u)   ∀u
                                    (ω, f) = (div v, f) ∀f
            => ω = weak_div v
            => δd = M1⁻¹ curl_curl - strong_grad @ weak_div
        where
            curl_curl_ij = ∫ curl Λ1_i · G curl Λ1_j (det DF)⁻¹ dx
        """
        W = self.metric_jkl * (1/self.jacobian_j * self.quad.w)[:, None, None]
        curl_curl = assemble(self.eval_d_basis_1_ijk, self.eval_d_basis_1_ijk,
                             W, self.basis_1.n, self.basis_1.n)
        self.dd1 = jnp.linalg.solve(
            self.m1, self.e1 @ curl_curl @ self.e1.T) - self.strong_grad @ self.weak_div

    def assemble_dd1_sparse(self):
        """
        Assemble curl-curl matrix for 1-form dofs in sparse format.
            curl_curl_ij = ∫ curl Λ1_i · G curl Λ1_j (det DF)⁻¹ dx
        """
        W = self.metric_jkl * (1/self.jacobian_j * self.quad.w)[:, None, None]
        neighbors, nnz = build_neighbors(self.basis_1)
        self.curl_curl_sp = assemble_sparse(self.eval_d_basis_1_ijk, self.eval_d_basis_1_ijk,
                                            W, self.basis_1.n, self.basis_1.n, nnz, neighbors)

        diag = self.m0_sp_diaginv @ square_bcoo(self.e0)
        diag = square_bcoo(self.d0_sp) @ diag
        diag = square_bcoo(self.e1) @ diag
        diag = diag + (square_bcoo(self.e1) @
                       extract_diag_vector(self.curl_curl_sp))
        self.dd1_sp_diaginv = 1 / diag
        
        diag_dbc = self.m0_sp_diaginv_dbc @ square_bcoo(self.e0_dbc)
        diag_dbc = square_bcoo(self.d0_sp) @ diag_dbc
        diag_dbc = square_bcoo(self.e1_dbc) @ diag_dbc
        diag_dbc = diag_dbc + (square_bcoo(self.e1_dbc) @
                       extract_diag_vector(self.curl_curl_sp))
        self.dd1_sp_diaginv_dbc = 1 / diag_dbc

    def apply_curl_curl_sparse(self, v, dirichlet=True):
        """
        Apply the sparse curl-curl matrix to a vector v.
        """
        if dirichlet:
            return self.e1_dbc @ (self.curl_curl_sp @ (self.e1_dbc.T @ v))
        else:
            return self.e1 @ (self.curl_curl_sp @ (self.e1.T @ v))

    def apply_dd1_sparse(self, u, dirichlet=True):
        """
        Forward application of the saddle point operator for the k=1 Hodge Laplacian:

        | curl_curl  grad |  | u (1 form) |
        | grad.T     -M0  |  | s (0 form) |

        We form the Schur complement:
        (curl_curl + grad @ M0⁻¹ @ grad.T) u

        To compute the inverse of M0, we do an inner cg solve with Jacobi preconditioning.
        """
        minus_div_u = self.apply_d0t_sparse(u, dirichlet_out=dirichlet, dirichlet_in=dirichlet)
        # inner solve with Jacobi preconditioning for M0
        m0_inv_div_u = cg(lambda x: self.apply_m0_sparse(x, dirichlet=dirichlet), 
                          minus_div_u,
                          M=lambda x: self.apply_m0_precond(x, dirichlet=dirichlet), 
                          tol=self.tol, maxiter=self.maxiter)[0]
        return self.apply_curl_curl_sparse(u, dirichlet=dirichlet) \
            + self.apply_d0_sparse(m0_inv_div_u, dirichlet_out=dirichlet, dirichlet_in=dirichlet)


    def apply_dd1_precond(self, v, dirichlet=True):
        """
        Apply the diagonal preconditioner for the k=1 Hodge Laplacian to a vector v.
        """
        if dirichlet:
            return self.dd1_sp_diaginv_dbc * v
        else:
            return self.dd1_sp_diaginv * v

    def assemble_dd2(self):
        """
        Assemble Hodge-Laplacian for 2-form dofs:
            (div ω, div ξ) + (curl v, ξ) = (δdω, ξ)     ∀ξ
                                  (v, u) = (curl ω, u)  ∀u
            => v = weak_curl ω
            => δd = M2⁻¹ div_div + strong_curl @ weak_curl
        where
            div_div_ij = ∫ div Λ2_i div Λ2_j (det DF)⁻¹ dx
        """
        W = (1/self.jacobian_j * self.quad.w)[:, None, None]
        M = assemble(self.eval_d_basis_2_ijk, self.eval_d_basis_2_ijk,
                     W, self.basis_2.n, self.basis_2.n)
        self.dd2 = jnp.linalg.solve(
            self.m2, self.e2 @ M @ self.e2.T) + self.strong_curl @ self.weak_curl

    def assemble_dd2_sparse(self):
        """
        Assemble div-div matrix for 2-form dofs in sparse format.
            div_div_ij = ∫ div Λ2_i div Λ2_j (det DF)⁻¹ dx
        """
        W = (1/self.jacobian_j * self.quad.w)[:, None, None]
        neighbors, nnz = build_neighbors(self.basis_2)
        self.div_div_sp = assemble_sparse(self.eval_d_basis_2_ijk, self.eval_d_basis_2_ijk,
                                          W, self.basis_2.n, self.basis_2.n, nnz, neighbors)

        diag = self.m1_sp_diaginv @ square_bcoo(self.e1)
        diag = square_bcoo(self.d1_sp) @ diag
        diag = square_bcoo(self.e2) @ diag
        diag = diag + (square_bcoo(self.e2) @
                       extract_diag_vector(self.div_div_sp))
        self.dd2_sp_diaginv = 1 / diag
        
        diag_dbc = self.m1_sp_diaginv_dbc @ square_bcoo(self.e1_dbc)
        diag_dbc = square_bcoo(self.d1_sp) @ diag_dbc
        diag_dbc = square_bcoo(self.e2_dbc) @ diag_dbc
        diag_dbc = diag_dbc + (square_bcoo(self.e2_dbc) @
                       extract_diag_vector(self.div_div_sp))
        self.dd2_sp_diaginv_dbc = 1 / diag_dbc

    def apply_div_div_sparse(self, v, dirichlet=True):
        """
        Apply the sparse div-div matrix to a vector v.
        """
        if dirichlet:
            return self.e2_dbc @ (self.div_div_sp @ (self.e2_dbc.T @ v))
        else:
            return self.e2 @ (self.div_div_sp @ (self.e2.T @ v))

    def apply_dd2_sparse(self, u, dirichlet=True):
        """
        Forward application of the saddle point operator for the k=2 Hodge Laplacian:

        | div_div  curl  |  | u (2 form) |
        | curl.T     -M1 |  | s (1 form) |

        We form the Schur complement:
        (div_div + curl @ M1⁻¹ @ curl.T) u

        To compute the inverse of M1, we do an inner cg solve with Jacobi preconditioning.
        """
        curl_u = self.apply_d1t_sparse(u, dirichlet_out=dirichlet, dirichlet_in=dirichlet)
        # inner solve with Jacobi preconditioning for M1
        m1_inv_curl_u = cg(lambda x: self.apply_m1_sparse(x, dirichlet=dirichlet),
                           curl_u,
                           M=lambda x: self.apply_m1_precond(x, dirichlet=dirichlet),
                           maxiter=self.maxiter, tol=self.tol)[0]
        return self.apply_div_div_sparse(u, dirichlet=dirichlet) \
            + self.apply_d1_sparse(m1_inv_curl_u, dirichlet_out=dirichlet, dirichlet_in=dirichlet)

    def apply_dd2_precond(self, v, dirichlet=True):
        """
        Apply the diagonal preconditioner for the k=2 Hodge Laplacian to a vector v.
        """
        if dirichlet:
            return self.dd2_sp_diaginv_dbc * v
        else:
            return self.dd2_sp_diaginv * v

    def assemble_dd3(self):
        """
        Assemble Hodge-Laplacian for 2-form dofs:
            -(div ξ, μ) = (δdρ, μ)       ∀μ
                 (ξ, ω) = (grad ρ, ω)    ∀ω
            => ξ = weak_grad ρ
            => δd = - strong_div @ weak_grad
        where
            div_div_ij = ∫ div Λ2_i div Λ2_j (det DF)⁻¹ dx
        """
        self.dd3 = -self.strong_div @ self.weak_grad

    def assemble_dd3_sparse(self):
        diag = self.m2_sp_diaginv @ square_bcoo(self.e2)
        diag = square_bcoo(self.d2_sp) @ diag
        diag = square_bcoo(self.e3) @ diag
        self.dd3_sp_diaginv = 1 / diag
        
        diag_dbc = self.m2_sp_diaginv_dbc @ square_bcoo(self.e2_dbc)
        diag_dbc = square_bcoo(self.d2_sp) @ diag_dbc
        diag_dbc = square_bcoo(self.e3_dbc) @ diag_dbc
        self.dd3_sp_diaginv_dbc = 1 / diag_dbc

    def apply_dd3_sparse(self, u, dirichlet=True):
        """
        Forward application of the saddle point operator for the k=3 Hodge Laplacian:

        | 0       div |  | u (3 form) |
        | div.T   -M2 |  | s (2 form) |

        We form the Schur complement:
        div @ M2⁻¹ @ div.T u

        To compute the inverse of M2, we do an inner cg solve with Jacobi preconditioning.
        """
        minus_grad_u = self.apply_d2t_sparse(u, dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        # inner solve with Jacobi preconditioning for M2
        m2_inv_minus_grad_u = cg(lambda x: self.apply_m2_sparse(x, dirichlet=dirichlet),
                                 minus_grad_u,
                                 M=lambda x: self.apply_m2_precond(x, dirichlet=dirichlet),
                                 tol=self.tol, maxiter=self.maxiter)[0]
        return self.apply_d2_sparse(m2_inv_minus_grad_u, dirichlet_out=dirichlet, dirichlet_in=dirichlet)

    def apply_dd3_precond(self, v, dirichlet=True):
        """
        Apply the diagonal preconditioner for the k=3 Hodge Laplacian to a vector v.
        """
        if dirichlet:
            return self.dd3_sp_diaginv_dbc * v
        else:
            return self.dd3_sp_diaginv * v

    def assemble_p12(self):
        """
        Projection matrix from 2- to 1-form dofs:
            (v, ω) = (v, u)       ∀v
            M12_ij = ∫ Λ1_i · Λ2_j dx
        and
            P12 = M1⁻¹ M12
        """
        w = self.quad.w[:, None, None] * jnp.eye(3)  # shape (n_q, 1, 1)
        m = assemble(self.eval_basis_1_ijk, self.eval_basis_2_ijk,
                     w, self.basis_1.n, self.basis_2.n)
        self.m12 = self.e1 @ m @ self.e2.T
        self.p12 = jnp.linalg.solve(self.m1, self.m12)

    def assemble_m12_sparse(self):
        """
        Assemble 1-2 cross mass matrix in sparse format.
            M12_ij = ∫ Λ1_i · Λ2_j dx
        """
        W = self.quad.w[:, None, None] * jnp.eye(3)
        neighbors, nnz = build_neighbors(self.basis_1, self.basis_2)
        self.m12_sp = assemble_sparse(self.eval_basis_1_ijk, self.eval_basis_2_ijk,
                                      W, self.basis_1.n, self.basis_2.n, nnz, neighbors)

    def apply_m12_sparse(self, v, dirichlet_in=True, dirichlet_out=True):
        """
        Apply the sparse 1-2 cross mass matrix to a vector v.
        """
        if dirichlet_out and dirichlet_in:
            return self.e1_dbc @ (self.m12_sp @ (self.e2_dbc.T @ v))
        elif dirichlet_out and not dirichlet_in:
            return self.e1_dbc @ (self.m12_sp @ (self.e2.T @ v))
        elif not dirichlet_out and dirichlet_in:
            return self.e1 @ (self.m12_sp @ (self.e2_dbc.T @ v))
        else:
            return self.e1 @ (self.m12_sp @ (self.e2.T @ v))

    def assemble_p03(self):
        """
        Projection matrix from 3- to 0-form dofs:
            (f, ρ) = (f, g)       ∀f
            M03_ij = ∫ Λ0_i · Λ3_j dx
        and
            P03 = M0⁻¹ M03
        """
        W = self.quad.w[:, None, None]  # shape (n_q, 1, 1)
        M = assemble(self.eval_basis_0_ijk, self.eval_basis_3_ijk,
                     W, self.basis_0.n, self.basis_3.n)
        self.m03 = self.e0 @ M @ self.e3.T
        self.p03 = jnp.linalg.solve(self.m0, self.m03)

    def assemble_m03_sparse(self):
        """
        Assemble 0-3 cross mass matrix in sparse format.
            M03_ij = ∫ Λ0_i · Λ3_j dx
        """
        W = self.quad.w[:, None, None]
        neighbors, nnz = build_neighbors(self.basis_0, self.basis_3)
        self.m03_sp = assemble_sparse(self.eval_basis_0_ijk, self.eval_basis_3_ijk,
                                      W, self.basis_0.n, self.basis_3.n, nnz, neighbors)

    def apply_m03_sparse(self, v, dirichlet_in=True, dirichlet_out=True):
        """
        Apply the sparse 0-3 cross mass matrix to a vector v.
        """
        if dirichlet_in and dirichlet_out:
            return self.e0_dbc @ (self.m03_sp @ (self.e3_dbc.T @ v))
        elif dirichlet_in and not dirichlet_out:
            return self.e0_dbc @ (self.m03_sp @ (self.e3.T @ v))
        elif not dirichlet_in and dirichlet_out:
            return self.e0 @ (self.m03_sp @ (self.e3_dbc.T @ v))
        else:
            return self.e0 @ (self.m03_sp @ (self.e3.T @ v))

    def build_crossproduct_projections(self):
        """
        Returns projections to evaluate (u, v) -> u x v
        """
        self.P1x1_to_1 = CrossProductProjection(1, 1, 1, self)
        # Not yet implemented
        # self.P1x2_to_1 = CrossProductProjection(1, 1, 2, self)
        self.P2x1_to_1 = CrossProductProjection(1, 2, 1, self)
        self.P2x2_to_1 = CrossProductProjection(1, 2, 2, self)
        self.P1x1_to_2 = CrossProductProjection(2, 1, 1, self)
        # Not yet implemented
        # self.P1x2_to_2 = CrossProductProjection(2, 1, 2, self)
        self.P2x1_to_2 = CrossProductProjection(2, 2, 1, self)
        self.P2x2_to_2 = CrossProductProjection(2, 2, 2, self)

    def assemble_all(self):
        """
        Assemble all the matrices and operators.
        """
        self.assemble_m0()
        self.assemble_m1()
        self.assemble_m2()
        self.assemble_m3()
        self.assemble_d0()
        self.assemble_d1()
        self.assemble_d2()
        self.assemble_dd0()
        self.assemble_dd1()
        self.assemble_dd2()
        self.assemble_dd3()
        self.assemble_p12()
        self.assemble_p03()

    def assemble_all_sparse(self):
        """
        Assemble all the matrices and operators in sparse format.
        """
        self.assemble_m0_sparse()
        self.assemble_m1_sparse()
        self.assemble_m2_sparse()
        self.assemble_m3_sparse()
        self.assemble_d0_sparse()
        self.assemble_d1_sparse()
        self.assemble_d2_sparse()
        self.assemble_dd0_sparse()
        self.assemble_dd1_sparse()
        self.assemble_dd2_sparse()
        self.assemble_dd3_sparse()
        self.assemble_m12_sparse()
        self.assemble_m03_sparse()

    def assemble_leray_projection(self):
        """
        Assemble the Leray projection matrix. Formula:
            P_Leray = I + weak_grad @ (dd3)^-1 @ strong_div
        """
        self.P_Leray = jnp.eye(self.m2.shape[0]) + \
            self.weak_grad @ jnp.linalg.pinv(self.dd3) @ self.strong_div

    def apply_leray_projection(self, v, k=2):
        """
        Apply the Leray projection to a 2-form v.

        When k = 2:
            Solves the system (k=3 Hodge Laplacian):
            div v = div σ
            (σ, ω) = -(p, div ω) ∀ω 2-forms
            -> div(v - weak_grad p) = div(v - σ) = 0 and σ.n = 0 on the boundary.
        When k = 1:
            Solves the k=0 Hodge Laplacian:
            (grad p, grad ω) = (v, grad ω) ∀ω 0-forms
            -> div(v - grad p) = 0 and p = 0 on the boundary.
        """
        if k == 2:
            # Assumes dirichlet == True on all spaces.
            div_v = self.apply_d2_sparse(v, True, True)
            p = solve_singular_cg(lambda x: self.apply_dd3_sparse(x, True), 
                                  div_v, 
                                  mass_matvec=lambda x: self.apply_m3_sparse(x, True),
                                  precond_matvec=lambda x: self.apply_dd3_precond(x, True), 
                                  vs=self.null_3_dbc, 
                                  tol=self.tol, maxiter=self.maxiter)[0]
            σ = -self.apply_weak_grad(p, True, True)
            return v - σ, p
        elif k == 1:
            # Assumes dirichlet == False on all spaces.
            div_v = -self.apply_d0t_sparse(v, False, False)
            p = solve_singular_cg(lambda x: self.apply_dd0_sparse(x, False), 
                                  div_v, 
                                  mass_matvec=lambda x: self.apply_m0_sparse(x, False),
                                  precond_matvec=lambda x: self.apply_dd0_precond(x, False), 
                                  vs=self.null_0, 
                                  tol=self.tol, maxiter=self.maxiter)[0]
            σ = -self.apply_strong_grad(p, False, False)
            return v - σ, p

    # TODO: We can pre-compute strong operators, they are sparse
    def apply_strong_grad(self, v, dirichlet_in=True, dirichlet_out=True):
        """
        Apply the strong gradient operator to a vector v.
        """
        return cg(lambda x: self.apply_m1_sparse(x, dirichlet_out), 
                  self.apply_d0_sparse(v, dirichlet_in=dirichlet_in, dirichlet_out=dirichlet_out), 
                  M=lambda x: self.apply_m1_precond(x, dirichlet_out), 
                  tol=self.tol, maxiter=self.maxiter)[0]

    def apply_strong_curl(self, v, dirichlet_in=True, dirichlet_out=True):
        """
        Apply the strong curl operator to a vector v.
        """
        return cg(lambda x: self.apply_m2_sparse(x, dirichlet_out), 
                  self.apply_d1_sparse(v, dirichlet_in=dirichlet_in, dirichlet_out=dirichlet_out), 
                  M=lambda x: self.apply_m2_precond(x, dirichlet_out), 
                  tol=self.tol, maxiter=self.maxiter)[0]

    def apply_strong_div(self, v, dirichlet_in=True, dirichlet_out=True):
        """
        Apply the strong divergence operator to a vector v.
        """
        return cg(lambda x: self.apply_m3_sparse(x, dirichlet_out), 
                  self.apply_d2_sparse(v, dirichlet_in=dirichlet_in, dirichlet_out=dirichlet_out), 
                  M=lambda x: self.apply_m3_precond(x, dirichlet_out),
                  tol=self.tol, maxiter=self.maxiter)[0]

    def apply_weak_grad(self, v, dirichlet_in=True, dirichlet_out=True):
        """
        Apply the weak gradient operator to a vector v.
        """
        return -cg(lambda x: self.apply_m2_sparse(x, dirichlet_out), 
                   self.apply_d2t_sparse(v, dirichlet_in=dirichlet_in, dirichlet_out=dirichlet_out), 
                   M=lambda x: self.apply_m2_precond(x, dirichlet_out), 
                   tol=self.tol, maxiter=self.maxiter)[0]

    def apply_weak_curl(self, v, dirichlet_in=True, dirichlet_out=True):
        """
        Apply the weak curl operator to a vector v.
        """
        return cg(lambda x: self.apply_m1_sparse(x, dirichlet_out), 
                  self.apply_d1t_sparse(v, dirichlet_in=dirichlet_in, dirichlet_out=dirichlet_out), 
                  M=lambda x: self.apply_m1_precond(x, dirichlet_out),
                  tol=self.tol, maxiter=self.maxiter)[0]

    def apply_weak_div(self, v, dirichlet_in=True, dirichlet_out=True):
        """
        Apply the weak divergence operator to a vector v.
        """
        return -cg(lambda x: self.apply_m0_sparse(x, dirichlet_out), 
                   self.apply_d0t_sparse(v, dirichlet_in=dirichlet_in, dirichlet_out=dirichlet_out), 
                   M=lambda x: self.apply_m0_precond(x, dirichlet_out),
                   tol=self.tol, maxiter=self.maxiter)[0]
# %%

    def compute_nullspaces(self):
        """
        Compute the nullspace of the k-th Hodge Laplacian using randomized SVD.
        TODO: For now this only handles the case where the nullspace is 1-dim.
        """
        self.null_0_dbc = []
        self.null_1_dbc = []
        v3 = cg(lambda x: self.apply_m3_sparse(x, True), 
                jnp.ones(self.n3_dbc), 
                M=lambda x: self.apply_m3_precond(x, True), 
                tol=self.tol, maxiter=self.maxiter)[0]
        v3 /= (v3 @ self.apply_m3_sparse(v3, True)) ** 0.5
        self.null_3_dbc = [v3]
        v, _ = self.apply_leray_projection(
            jnp.ones(self.n2_dbc), k=2)
        a = cg(lambda x: self.apply_dd1_sparse(x, True), 
               self.apply_d1t_sparse(v, True, True), 
               M=lambda x: self.apply_dd1_precond(x, True),
               tol=self.tol, maxiter=self.maxiter)[0]
        curl_a = self.apply_strong_curl(a, True, True) 
        v2 = v - curl_a
        v2 /= (v2 @ self.apply_m2_sparse(v2, True))**0.5
        self.null_2_dbc = [v2]
        
        # no Dirichlet BCs (all defaults to False)
        v0 = jnp.ones(self.n0)
        v0 /= (v0 @ self.apply_m0_sparse(v0, False)) ** 0.5
        self.null_0 = [v0]
        v, _ = self.apply_leray_projection(
            jnp.ones(self.n1), k=1)
        a = cg(lambda x: self.apply_dd2_sparse(x, False), 
               self.apply_d1_sparse(v, False, False), 
               M=lambda x: self.apply_dd2_precond(x, False),
               tol=self.tol, maxiter=self.maxiter)[0]
        curl_a = self.apply_weak_curl(a, False, False)
        v1 = v - curl_a
        v1 /= (v1 @ self.apply_m1_sparse(v1, False))**0.5
        self.null_1 = [v1]
        self.null_2 = []
        self.null_3 = []
