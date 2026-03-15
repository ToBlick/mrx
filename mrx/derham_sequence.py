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
from mrx.utils import (assemble, assemble_sparse, build_neighbors,
                       evaluate_at_xq, evaluate_at_xq_deprecated,
                       extract_diag_vector, integrate_against,
                       integrate_against_deprecated, inv33,
                       jacobian_determinant, solve_singular_cg, square_sparse)


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

    def __init__(self, ns, ps, q, types, map, polar, tol=1e-12, maxiter=10_000, r_scale=1.0):
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

        def _to_bcsr_pair(bcoo):
            return jsparse.BCSR.from_bcoo(bcoo), jsparse.BCSR.from_bcoo(bcoo.T)

        self.e0, self.e0_T = _to_bcsr_pair(e0.assemble_sparse())
        self.e0_dbc, self.e0_dbc_T = _to_bcsr_pair(e0_dbc.assemble_sparse())
        self.n0 = e0.n
        self.n0_dbc = e0_dbc.n
        self.n0_1, self.n0_2, self.n0_3 = e0.n, 0, 0
        self.n0_1_dbc, self.n0_2_dbc, self.n0_3_dbc = e0_dbc.n, 0, 0
        self.e1, self.e1_T = _to_bcsr_pair(e1.assemble_sparse())
        self.e1_dbc, self.e1_dbc_T = _to_bcsr_pair(e1_dbc.assemble_sparse())
        self.n1 = e1.n
        self.n1_dbc = e1_dbc.n
        self.n1_1, self.n1_2, self.n1_3 = e1.n1, e1.n2, e1.n2
        self.n1_1_dbc, self.n1_2_dbc, self.n1_3_dbc = e1_dbc.n1, e1_dbc.n2, e1_dbc.n2
        self.e2, self.e2_T = _to_bcsr_pair(e2.assemble_sparse())
        self.e2_dbc, self.e2_dbc_T = _to_bcsr_pair(e2_dbc.assemble_sparse())
        self.n2 = e2.n
        self.n2_dbc = e2_dbc.n
        self.n2_1, self.n2_2, self.n2_3 = e2.n1, e2.n2, e2.n3
        self.n2_1_dbc, self.n2_2_dbc, self.n2_3_dbc = e2_dbc.n1, e2_dbc.n2, e2_dbc.n3
        self.e3, self.e3_T = _to_bcsr_pair(e3.assemble_sparse())
        self.e3_dbc, self.e3_dbc_T = _to_bcsr_pair(e3_dbc.assemble_sparse())
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

    def _form_comp_info(self, k):
        """Return (comp_info, comp_shapes) for the k-th form.

        comp_info[c] = (output_dim, R, T, Z) for each component c.
        comp_shapes[c] = (s1, s2, s3) DOF grid shape per component.
        """
        match k:
            case 0:
                return (
                    [(0, self.basis_r_jk, self.basis_t_jk, self.basis_z_jk)],
                    list(self.basis_0.shape),
                )
            case 1:
                return (
                    [(0, self.d_basis_r_jk, self.basis_t_jk, self.basis_z_jk),
                     (1, self.basis_r_jk, self.d_basis_t_jk, self.basis_z_jk),
                     (2, self.basis_r_jk, self.basis_t_jk, self.d_basis_z_jk)],
                    list(self.basis_1.shape),
                )
            case 2:
                return (
                    [(0, self.basis_r_jk, self.d_basis_t_jk, self.d_basis_z_jk),
                     (1, self.d_basis_r_jk, self.basis_t_jk, self.d_basis_z_jk),
                     (2, self.d_basis_r_jk, self.d_basis_t_jk, self.basis_z_jk)],
                    list(self.basis_2.shape),
                )
            case 3:
                return (
                    [(0, self.d_basis_r_jk, self.d_basis_t_jk, self.d_basis_z_jk)],
                    list(self.basis_3.shape),
                )
            case _:
                raise ValueError("k must be 0, 1, 2, or 3")

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
        return jnp.array([
            (dr_m1 - dr) * self.basis_t_jk[i2, j2] * self.basis_z_jk[i3, j3],
            self.basis_r_jk[i1, j1] *
            (dtheta_m1 - dtheta) * self.basis_z_jk[i3, j3],
            self.basis_r_jk[i1, j1] * self.basis_t_jk[i2, j2] * (dz_m1 - dz)
        ])[k]

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
        components = jnp.array([
            self.d_basis_r_jk[i1, j1] *
            self.basis_t_jk[i2, j2] * self.basis_z_jk[i3, j3],
            self.basis_r_jk[i1, j1] * self.d_basis_t_jk[i2,
                                                        j2] * self.basis_z_jk[i3, j3],
            self.basis_r_jk[i1, j1] *
            self.basis_t_jk[i2, j2] * self.d_basis_z_jk[i3, j3]
        ])
        return jnp.where(k == c, components[c], 0.0)

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

        curl_matrix = jnp.array([
            [0.0,    d1dz,  -d1dy],
            [-d2dz,  0.0,    d2dx],
            [d3dy,  -d3dx,   0.0]
        ])
        return curl_matrix[c, k]

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
        components = jnp.array([
            self.basis_r_jk[i1, j1] * self.d_basis_t_jk[i2,
                                                        j2] * self.d_basis_z_jk[i3, j3],
            self.d_basis_r_jk[i1, j1] * self.basis_t_jk[i2,
                                                        j2] * self.d_basis_z_jk[i3, j3],
            self.d_basis_r_jk[i1, j1] *
            self.d_basis_t_jk[i2, j2] * self.basis_z_jk[i3, j3]
        ])
        return jnp.where(k == c, components[c], 0.0)

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

        return jnp.array([
            (dr_m1 - dr) * self.d_basis_t_jk[i2,
                                             j2] * self.d_basis_z_jk[i3, j3],
            self.d_basis_r_jk[i1, j1] *
            (dtheta_m1 - dtheta) * self.d_basis_z_jk[i3, j3],
            self.d_basis_r_jk[i1, j1] *
            self.d_basis_t_jk[i2, j2] * (dz_m1 - dz)
        ])[c]

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

    def l2_norm_sq(self, v, k, dirichlet=True):
        return v @ self.apply_mass_matrix(v, k, dirichlet=dirichlet)

    def l2_norm(self, v, k, dirichlet=True):
        return jnp.sqrt(self.l2_norm_sq(v, k, dirichlet=dirichlet))

    def assemble_all_sparse(self):
        """
        Assemble all the matrices and operators in sparse format.
        """
        for k in range(4):
            self.assemble_mass_matrix(k)
        for k in range(3):
            self.assemble_derivative_matrix(k)
        for k in range(4):
            self.assemble_hodge_laplacian(k)
        for k_from, k_to in [(2, 1), (3, 0)]:
            self.assemble_projection_matrix(k_from, k_to)

    def assemble_mass_matrix_deprecated(self, k):
        """
        Assemble the sparse mass matrix Mk for k-forms (k = 0, 1, 2, 3).
            k=0: M0_ij = ∫ Λ0_i Λ0_j det DF dx
            k=1: M1_ij = ∫ Λ1_i · G⁻¹ Λ1_j det DF dx
            k=2: M2_ij = ∫ Λ2_i · G Λ2_j (det DF)⁻¹ dx
            k=3: M3_ij = ∫ Λ3_i Λ3_j (det DF)⁻¹ dx
        Also assembles the diagonal Jacobi preconditioner (diaginv and diaginv_dbc).
        """
        match k:
            case 0:
                W = (self.jacobian_j * self.quad.w)[:, None, None]
                neighbors, nnz = build_neighbors(self.basis_0)
                sp = assemble_sparse(
                    self.eval_basis_0_ijk, self.eval_basis_0_ijk, W, self.basis_0.n, self.basis_0.n, nnz, neighbors)
                self.m0_sp_diaginv = 1 / \
                    (square_sparse(self.e0) @ extract_diag_vector(sp))
                self.m0_sp_diaginv_dbc = 1 / \
                    (square_sparse(self.e0_dbc) @ extract_diag_vector(sp))
                self.m0_sp = jsparse.BCSR.from_bcoo(sp)
            case 1:
                W = self.metric_inv_jkl * \
                    (self.jacobian_j * self.quad.w)[:, None, None]
                neighbors, nnz = build_neighbors(self.basis_1)
                sp = assemble_sparse(
                    self.eval_basis_1_ijk, self.eval_basis_1_ijk, W, self.basis_1.n, self.basis_1.n, nnz, neighbors)
                self.m1_sp_diaginv = 1 / \
                    (square_sparse(self.e1) @ extract_diag_vector(sp))
                self.m1_sp_diaginv_dbc = 1 / \
                    (square_sparse(self.e1_dbc) @ extract_diag_vector(sp))
                self.m1_sp = jsparse.BCSR.from_bcoo(sp)
            case 2:
                W = self.metric_jkl * \
                    (1/self.jacobian_j * self.quad.w)[:, None, None]
                neighbors, nnz = build_neighbors(self.basis_2)
                sp = assemble_sparse(
                    self.eval_basis_2_ijk, self.eval_basis_2_ijk, W, self.basis_2.n, self.basis_2.n, nnz, neighbors)
                self.m2_sp_diaginv = 1 / \
                    (square_sparse(self.e2) @ extract_diag_vector(sp))
                self.m2_sp_diaginv_dbc = 1 / \
                    (square_sparse(self.e2_dbc) @ extract_diag_vector(sp))
                self.m2_sp = jsparse.BCSR.from_bcoo(sp)
            case 3:
                W = (1/self.jacobian_j * self.quad.w)[:, None, None]
                neighbors, nnz = build_neighbors(self.basis_3)
                sp = assemble_sparse(
                    self.eval_basis_3_ijk, self.eval_basis_3_ijk, W, self.basis_3.n, self.basis_3.n, nnz, neighbors)
                self.m3_sp_diaginv = 1 / \
                    (square_sparse(self.e3) @ extract_diag_vector(sp))
                self.m3_sp_diaginv_dbc = 1 / \
                    (square_sparse(self.e3_dbc) @ extract_diag_vector(sp))
                self.m3_sp = jsparse.BCSR.from_bcoo(sp)
            case _:
                raise ValueError("k must be 0, 1, 2 or 3")

    def assemble_mass_matrix(self, k):
        """Assemble the mass matrix using tensor-product contraction.

        """
        from mrx.utils import assemble_scalar_tp, assemble_vectorial_tp
        quad_shape = (self.quad.ny, self.quad.nx, self.quad.nz)
        match k:
            case 0:
                W_flat = self.jacobian_j * self.quad.w
                sp = assemble_scalar_tp(
                    self.basis_r_jk, self.basis_t_jk, self.basis_z_jk,
                    self.basis_r_jk, self.basis_t_jk, self.basis_z_jk,
                    W_flat, quad_shape, self.basis_0.shape[0],
                    self.basis_0.pr, self.basis_0.pt, self.basis_0.pz)
                self.m0_sp_diaginv = 1 / \
                    (square_sparse(self.e0) @ extract_diag_vector(sp))
                self.m0_sp_diaginv_dbc = 1 / \
                    (square_sparse(self.e0_dbc) @ extract_diag_vector(sp))
                self.m0_sp = jsparse.BCSR.from_bcoo(sp)
            case 1:
                W_3x3 = self.metric_inv_jkl * \
                    (self.jacobian_j * self.quad.w)[:, None, None]
                terms = [
                    [(0, self.d_basis_r_jk, self.basis_t_jk, self.basis_z_jk, +1)],
                    [(1, self.basis_r_jk, self.d_basis_t_jk, self.basis_z_jk, +1)],
                    [(2, self.basis_r_jk, self.basis_t_jk, self.d_basis_z_jk, +1)],
                ]
                sp = assemble_vectorial_tp(
                    terms, terms, W_3x3, quad_shape,
                    list(self.basis_1.shape),
                    self.basis_1.pr)
                self.m1_sp_diaginv = 1 / \
                    (square_sparse(self.e1) @ extract_diag_vector(sp))
                self.m1_sp_diaginv_dbc = 1 / \
                    (square_sparse(self.e1_dbc) @ extract_diag_vector(sp))
                self.m1_sp = jsparse.BCSR.from_bcoo(sp)
            case 2:
                W_3x3 = self.metric_jkl * \
                    (1 / self.jacobian_j * self.quad.w)[:, None, None]
                terms = [
                    [(0, self.basis_r_jk, self.d_basis_t_jk, self.d_basis_z_jk, +1)],
                    [(1, self.d_basis_r_jk, self.basis_t_jk, self.d_basis_z_jk, +1)],
                    [(2, self.d_basis_r_jk, self.d_basis_t_jk, self.basis_z_jk, +1)],
                ]
                sp = assemble_vectorial_tp(
                    terms, terms, W_3x3, quad_shape,
                    list(self.basis_2.shape),
                    self.basis_2.pr)
                self.m2_sp_diaginv = 1 / \
                    (square_sparse(self.e2) @ extract_diag_vector(sp))
                self.m2_sp_diaginv_dbc = 1 / \
                    (square_sparse(self.e2_dbc) @ extract_diag_vector(sp))
                self.m2_sp = jsparse.BCSR.from_bcoo(sp)
            case 3:
                W_flat = (1 / self.jacobian_j) * self.quad.w
                sp = assemble_scalar_tp(
                    self.d_basis_r_jk, self.d_basis_t_jk, self.d_basis_z_jk,
                    self.d_basis_r_jk, self.d_basis_t_jk, self.d_basis_z_jk,
                    W_flat, quad_shape, self.basis_3.shape[0],
                    self.basis_3.pr, self.basis_3.pt, self.basis_3.pz)
                self.m3_sp_diaginv = 1 / \
                    (square_sparse(self.e3) @ extract_diag_vector(sp))
                self.m3_sp_diaginv_dbc = 1 / \
                    (square_sparse(self.e3_dbc) @ extract_diag_vector(sp))
                self.m3_sp = jsparse.BCSR.from_bcoo(sp)
            case _:
                raise ValueError(
                    "Tensor-product assembly supports k=0, 1, 2, 3")

    def assemble_projection_matrix_deprecated(self, k_from, k_to):
        """
        Assemble the sparse projection matrix Pk_from_k_to mapping k_from-form dofs to k_to-form dofs.
            k_from=2, k_to=1: M12_ij = ∫ Λ1_i · Λ2_j dx
            k_from=3, k_to=0: M03_ij = ∫ Λ0_i · Λ3_j dx
        and
            Pk_from_k_to = Mk_to⁻¹ M_k_from_k_to
        """
        match (k_from, k_to):
            case (2, 1) | (1, 2):
                W = self.quad.w[:, None, None] * jnp.eye(3)
                neighbors, nnz = build_neighbors(self.basis_1, self.basis_2)
                self.m12_sp = jsparse.BCSR.from_bcoo(assemble_sparse(
                    self.eval_basis_1_ijk, self.eval_basis_2_ijk, W, self.basis_1.n, self.basis_2.n, nnz, neighbors))
            case (3, 0) | (0, 3):
                W = self.quad.w[:, None, None]
                neighbors, nnz = build_neighbors(self.basis_0, self.basis_3)
                self.m03_sp = jsparse.BCSR.from_bcoo(assemble_sparse(
                    self.eval_basis_0_ijk, self.eval_basis_3_ijk, W, self.basis_0.n, self.basis_3.n, nnz, neighbors))
            case _:
                raise ValueError(
                    "Only (k_from, k_to) = (1, 2), (2, 1), (0, 3), or (3, 0) supported")

    def apply_projection_matrix(self, v, k_from, k_to, dirichlet_in=True, dirichlet_out=True):
        """
        Apply the sparse projection matrix Pk_from_k_to to a vector v.
        """
        match (k_from, k_to):
            case (2, 1):
                if dirichlet_out and dirichlet_in:
                    return self.e1_dbc @ (self.m12_sp @ (self.e2_dbc_T @ v))
                elif dirichlet_out and not dirichlet_in:
                    return self.e1_dbc @ (self.m12_sp @ (self.e2_T @ v))
                elif not dirichlet_out and dirichlet_in:
                    return self.e1 @ (self.m12_sp @ (self.e2_dbc_T @ v))
                else:
                    return self.e1 @ (self.m12_sp @ (self.e2_T @ v))
            case (0, 3):
                if dirichlet_out and dirichlet_in:
                    return self.e0_dbc @ (self.m03_sp @ (self.e3_dbc_T @ v))
                elif dirichlet_out and not dirichlet_in:
                    return self.e0_dbc @ (self.m03_sp @ (self.e3_T @ v))
                elif not dirichlet_out and dirichlet_in:
                    return self.e0 @ (self.m03_sp @ (self.e3_dbc_T @ v))
                else:
                    return self.e0 @ (self.m03_sp @ (self.e3_T @ v))
            case _:
                raise ValueError(
                    "Only (k_from, k_to) = (1, 2), (2, 1), (0, 3), or (3, 0) supported")

    def assemble_projection_matrix(self, k_from, k_to):
        """Assemble the projection matrix using tensor-product contraction.

        Supports (k_from, k_to) = (2, 1), (1, 2), (3, 0), (0, 3).
        """
        from mrx.utils import assemble_vectorial_tp
        quad_shape = (self.quad.ny, self.quad.nx, self.quad.nz)
        dR = self.d_basis_r_jk
        dT = self.d_basis_t_jk
        dZ = self.d_basis_z_jk
        R = self.basis_r_jk
        T = self.basis_t_jk
        Z = self.basis_z_jk
        match (k_from, k_to):
            case (2, 1) | (1, 2):
                W_3x3 = self.quad.w[:, None, None] * jnp.eye(3)
                # Row: 1-form identity
                row_terms = [
                    [(0, dR, T, Z, +1)],
                    [(1, R, dT, Z, +1)],
                    [(2, R, T, dZ, +1)],
                ]
                # Col: 2-form identity
                col_terms = [
                    [(0, R, dT, dZ, +1)],
                    [(1, dR, T, dZ, +1)],
                    [(2, dR, dT, Z, +1)],
                ]
                sp = assemble_vectorial_tp(
                    row_terms, col_terms, W_3x3, quad_shape,
                    list(self.basis_1.shape), self.basis_1.pr,
                    col_comp_shapes=list(self.basis_2.shape))
                self.m12_sp = jsparse.BCSR.from_bcoo(sp)
            case (3, 0) | (0, 3):
                W_1x1 = self.quad.w.reshape(-1, 1, 1)
                # Row: 0-form identity
                row_terms = [
                    [(0, R, T, Z, +1)],
                ]
                # Col: 3-form identity
                col_terms = [
                    [(0, dR, dT, dZ, +1)],
                ]
                sp = assemble_vectorial_tp(
                    row_terms, col_terms, W_1x1, quad_shape,
                    list(self.basis_0.shape), self.basis_0.pr,
                    col_comp_shapes=list(self.basis_3.shape))
                self.m03_sp = jsparse.BCSR.from_bcoo(sp)
            case _:
                raise ValueError(
                    "Only (k_from, k_to) = (1, 2), (2, 1), (0, 3), or (3, 0) supported")

    def assemble_derivative_matrix_deprecated(self, k):
        """
        Assemble the sparse exterior derivative matrix Dk mapping k-forms to (k+1)-forms (k = 0, 1, 2).
            k=0: D0_ij = ∫ Λ1_i · G⁻¹ grad Λ0_j det DF dx   (grad)
            k=1: D1_ij = ∫ Λ2_i · G curl Λ1_j (det DF)⁻¹ dx  (curl)
            k=2: D2_ij = ∫ Λ3_i div  Λ2_j (det DF)⁻¹ dx       (div)
        """
        match k:
            case 0:
                W = self.metric_inv_jkl * \
                    (self.jacobian_j * self.quad.w)[:, None, None]
                neighbors, nnz = build_neighbors(self.basis_1, self.basis_0)
                sp = assemble_sparse(
                    self.eval_basis_1_ijk, self.eval_d_basis_0_ijk, W, self.basis_1.n, self.basis_0.n, nnz, neighbors)
                self.d0_sp = jsparse.BCSR.from_bcoo(sp)
                self.d0_sp_T = jsparse.BCSR.from_bcoo(sp.T)
            case 1:
                W = self.metric_jkl * \
                    (1/self.jacobian_j * self.quad.w)[:, None, None]
                neighbors, nnz = build_neighbors(self.basis_2, self.basis_1)
                sp = assemble_sparse(
                    self.eval_basis_2_ijk, self.eval_d_basis_1_ijk, W, self.basis_2.n, self.basis_1.n, nnz, neighbors)
                self.d1_sp = jsparse.BCSR.from_bcoo(sp)
                self.d1_sp_T = jsparse.BCSR.from_bcoo(sp.T)
            case 2:
                W = (1/self.jacobian_j * self.quad.w)[:, None, None]
                neighbors, nnz = build_neighbors(self.basis_3, self.basis_2)
                sp = assemble_sparse(
                    self.eval_basis_3_ijk, self.eval_d_basis_2_ijk, W, self.basis_3.n, self.basis_2.n, nnz, neighbors)
                self.d2_sp = jsparse.BCSR.from_bcoo(sp)
                self.d2_sp_T = jsparse.BCSR.from_bcoo(sp.T)
            case _:
                raise ValueError("k must be 0, 1 or 2")

    def assemble_hodge_laplacian_deprecated(self, k):
        """
        Assemble the stiffness matrix and Jacobi preconditioner for the k-th Hodge Laplacian (δd).
            k=0: grad_grad_ij = ∫ ∇Λ0_i · G⁻¹ ∇Λ0_j det DF dx
            k=1: curl_curl_ij = ∫ curl Λ1_i · G curl Λ1_j (det DF)⁻¹ dx
            k=2: div_div_ij   = ∫ div  Λ2_i div  Λ2_j (det DF)⁻¹ dx
            k=3: (no stiffness matrix; preconditioner only uses d2_sp and m2_sp_diaginv)
        """
        match k:
            case 0:
                W = self.metric_inv_jkl * \
                    (self.jacobian_j * self.quad.w)[:, None, None]
                neighbors, nnz = build_neighbors(self.basis_0)
                sp = assemble_sparse(
                    self.eval_d_basis_0_ijk, self.eval_d_basis_0_ijk,
                    W, self.basis_0.n, self.basis_0.n, nnz, neighbors)
                self.dd0_sp_diaginv = 1 / \
                    (square_sparse(self.e0) @ extract_diag_vector(sp))
                self.dd0_sp_diaginv_dbc = 1 / \
                    (square_sparse(self.e0_dbc) @
                     extract_diag_vector(sp))
                self.grad_grad_sp = jsparse.BCSR.from_bcoo(sp)
            case 1:
                W = self.metric_jkl * \
                    (1/self.jacobian_j * self.quad.w)[:, None, None]
                neighbors, nnz = build_neighbors(self.basis_1)
                sp = assemble_sparse(
                    self.eval_d_basis_1_ijk, self.eval_d_basis_1_ijk,
                    W, self.basis_1.n, self.basis_1.n, nnz, neighbors)
                diag = self.m0_sp_diaginv @ square_sparse(self.e0)
                diag = square_sparse(self.d0_sp) @ diag
                diag = square_sparse(self.e1) @ diag
                self.dd1_sp_diaginv = 1 / \
                    (diag + square_sparse(self.e1) @
                     extract_diag_vector(sp))
                diag_dbc = self.m0_sp_diaginv_dbc @ square_sparse(self.e0_dbc)
                diag_dbc = square_sparse(self.d0_sp) @ diag_dbc
                diag_dbc = square_sparse(self.e1_dbc) @ diag_dbc
                self.dd1_sp_diaginv_dbc = 1 / \
                    (diag_dbc + square_sparse(self.e1_dbc) @
                     extract_diag_vector(sp))
                self.curl_curl_sp = jsparse.BCSR.from_bcoo(sp)
            case 2:
                W = (1/self.jacobian_j * self.quad.w)[:, None, None]
                neighbors, nnz = build_neighbors(self.basis_2)
                sp = assemble_sparse(
                    self.eval_d_basis_2_ijk, self.eval_d_basis_2_ijk,
                    W, self.basis_2.n, self.basis_2.n, nnz, neighbors)
                diag = self.m1_sp_diaginv @ square_sparse(self.e1)
                diag = square_sparse(self.d1_sp) @ diag
                diag = square_sparse(self.e2) @ diag
                self.dd2_sp_diaginv = 1 / \
                    (diag + square_sparse(self.e2) @
                     extract_diag_vector(sp))
                diag_dbc = self.m1_sp_diaginv_dbc @ square_sparse(self.e1_dbc)
                diag_dbc = square_sparse(self.d1_sp) @ diag_dbc
                diag_dbc = square_sparse(self.e2_dbc) @ diag_dbc
                self.dd2_sp_diaginv_dbc = 1 / \
                    (diag_dbc + square_sparse(self.e2_dbc) @
                     extract_diag_vector(sp))
                self.div_div_sp = jsparse.BCSR.from_bcoo(sp)
            case 3:
                diag = self.m2_sp_diaginv @ square_sparse(self.e2)
                diag = square_sparse(self.d2_sp) @ diag
                self.dd3_sp_diaginv = 1 / (square_sparse(self.e3) @ diag)
                diag_dbc = self.m2_sp_diaginv_dbc @ square_sparse(self.e2_dbc)
                diag_dbc = square_sparse(self.d2_sp) @ diag_dbc
                self.dd3_sp_diaginv_dbc = 1 / \
                    (square_sparse(self.e3_dbc) @ diag_dbc)
            case _:
                raise ValueError("k must be 0, 1, 2 or 3")

    def assemble_derivative_matrix(self, k):
        """Assemble the exterior derivative matrix using tensor-product contraction.

        Supports k=0 (grad), k=1 (curl), k=2 (div).
        """
        from mrx.utils import assemble_vectorial_tp
        quad_shape = (self.quad.ny, self.quad.nx, self.quad.nz)
        types = self.basis_0.types
        grad_r = self._grad_1d(self.d_basis_r_jk, types[0])
        grad_t = self._grad_1d(self.d_basis_t_jk, types[1])
        grad_z = self._grad_1d(self.d_basis_z_jk, types[2])
        match k:
            case 0:
                W_3x3 = self.metric_inv_jkl * \
                    (self.jacobian_j * self.quad.w)[:, None, None]
                # Row: 1-form identity (3 components)
                row_terms = [
                    [(0, self.d_basis_r_jk, self.basis_t_jk, self.basis_z_jk, +1)],
                    [(1, self.basis_r_jk, self.d_basis_t_jk, self.basis_z_jk, +1)],
                    [(2, self.basis_r_jk, self.basis_t_jk, self.d_basis_z_jk, +1)],
                ]
                # Col: grad of 0-form (1 component, 3 output terms)
                col_terms = [
                    [(0, grad_r, self.basis_t_jk, self.basis_z_jk, +1),
                     (1, self.basis_r_jk, grad_t, self.basis_z_jk, +1),
                     (2, self.basis_r_jk, self.basis_t_jk, grad_z, +1)],
                ]
                sp = assemble_vectorial_tp(
                    row_terms, col_terms, W_3x3, quad_shape,
                    list(self.basis_1.shape), self.basis_1.pr,
                    col_comp_shapes=list(self.basis_0.shape))
                self.d0_sp = jsparse.BCSR.from_bcoo(sp)
                self.d0_sp_T = jsparse.BCSR.from_bcoo(sp.T)
            case 1:
                W_3x3 = self.metric_jkl * \
                    (1 / self.jacobian_j * self.quad.w)[:, None, None]
                dR = self.d_basis_r_jk
                dT = self.d_basis_t_jk
                dZ = self.d_basis_z_jk
                R = self.basis_r_jk
                T = self.basis_t_jk
                Z = self.basis_z_jk
                # Row: 2-form identity (3 components)
                row_terms = [
                    [(0, R, dT, dZ, +1)],
                    [(1, dR, T, dZ, +1)],
                    [(2, dR, dT, Z, +1)],
                ]
                # Col: curl of 1-form (3 components, each with 2 signed terms)
                col_terms = [
                    [(1, dR, T, grad_z, +1),
                     (2, dR, grad_t, Z, -1)],
                    [(0, R, dT, grad_z, -1),
                     (2, grad_r, dT, Z, +1)],
                    [(0, R, grad_t, dZ, +1),
                     (1, grad_r, T, dZ, -1)],
                ]
                sp = assemble_vectorial_tp(
                    row_terms, col_terms, W_3x3, quad_shape,
                    list(self.basis_2.shape), self.basis_2.pr,
                    col_comp_shapes=list(self.basis_1.shape))
                self.d1_sp = jsparse.BCSR.from_bcoo(sp)
                self.d1_sp_T = jsparse.BCSR.from_bcoo(sp.T)
            case 2:
                W_scalar = (1 / self.jacobian_j) * self.quad.w
                W_1x1 = W_scalar.reshape(-1, 1, 1)
                dR = self.d_basis_r_jk
                dT = self.d_basis_t_jk
                dZ = self.d_basis_z_jk
                # Row: 3-form identity (1 component)
                row_terms = [
                    [(0, dR, dT, dZ, +1)],
                ]
                # Col: div of 2-form (3 components, each → output 0)
                col_terms = [
                    [(0, grad_r, dT, dZ, +1)],
                    [(0, dR, grad_t, dZ, +1)],
                    [(0, dR, dT, grad_z, +1)],
                ]
                sp = assemble_vectorial_tp(
                    row_terms, col_terms, W_1x1, quad_shape,
                    list(self.basis_3.shape), self.basis_3.pr,
                    col_comp_shapes=list(self.basis_2.shape))
                self.d2_sp = jsparse.BCSR.from_bcoo(sp)
                self.d2_sp_T = jsparse.BCSR.from_bcoo(sp.T)
            case _:
                raise ValueError(
                    "Tensor-product derivative assembly supports k=0, 1, 2")

    def _grad_1d(self, d_basis, boundary_type):
        """Compute gradient basis from derivative spline: dΛ(i-1) - dΛ(i)."""
        if boundary_type == 'clamped':
            padded = jnp.pad(d_basis, ((1, 1), (0, 0)))
            return padded[:-1] - padded[1:]
        else:  # periodic
            return jnp.roll(d_basis, 1, axis=0) - d_basis

    def assemble_hodge_laplacian(self, k):
        """Assemble the stiffness matrix (δd) using tensor-product contraction.

        Supports k=0 (grad-grad), k=1 (curl-curl), k=2 (div-div), k=3 (preconditioner only).
        """
        from mrx.utils import (assemble_stiffness_scalar_tp,
                               assemble_vectorial_tp)
        quad_shape = (self.quad.ny, self.quad.nx, self.quad.nz)
        types = self.basis_0.types
        grad_r = self._grad_1d(self.d_basis_r_jk, types[0])
        grad_t = self._grad_1d(self.d_basis_t_jk, types[1])
        grad_z = self._grad_1d(self.d_basis_z_jk, types[2])
        match k:
            case 0:
                W_3x3 = self.metric_inv_jkl * \
                    (self.jacobian_j * self.quad.w)[:, None, None]
                grad_basis_1d = [
                    (grad_r, self.basis_t_jk, self.basis_z_jk),
                    (self.basis_r_jk, grad_t, self.basis_z_jk),
                    (self.basis_r_jk, self.basis_t_jk, grad_z),
                ]
                sp = assemble_stiffness_scalar_tp(
                    grad_basis_1d, grad_basis_1d, W_3x3, quad_shape,
                    self.basis_0.shape[0],
                    self.basis_0.pr, self.basis_0.pt, self.basis_0.pz)
                self.dd0_sp_diaginv = 1 / \
                    (square_sparse(self.e0) @ extract_diag_vector(sp))
                self.dd0_sp_diaginv_dbc = 1 / \
                    (square_sparse(self.e0_dbc) @
                     extract_diag_vector(sp))
                self.grad_grad_sp = jsparse.BCSR.from_bcoo(sp)
            case 1:
                W_3x3 = self.metric_jkl * \
                    (1 / self.jacobian_j * self.quad.w)[:, None, None]
                dR = self.d_basis_r_jk
                dT = self.d_basis_t_jk
                dZ = self.d_basis_z_jk
                R = self.basis_r_jk
                T = self.basis_t_jk
                Z = self.basis_z_jk
                curl_terms = [
                    # c=0 (dR, T, Z) → curl k=1: +dR·T·grad_z, k=2: -dR·grad_t·Z
                    [(1, dR, T, grad_z, +1),
                     (2, dR, grad_t, Z, -1)],
                    # c=1 (R, dT, Z) → curl k=0: -R·dT·grad_z, k=2: +grad_r·dT·Z
                    [(0, R, dT, grad_z, -1),
                     (2, grad_r, dT, Z, +1)],
                    # c=2 (R, T, dZ) → curl k=0: +R·grad_t·dZ, k=1: -grad_r·T·dZ
                    [(0, R, grad_t, dZ, +1),
                     (1, grad_r, T, dZ, -1)],
                ]
                sp = assemble_vectorial_tp(
                    curl_terms, curl_terms, W_3x3, quad_shape,
                    list(self.basis_1.shape), self.basis_1.pr)
                diag = self.m0_sp_diaginv @ square_sparse(self.e0)
                diag = square_sparse(self.d0_sp) @ diag
                diag = square_sparse(self.e1) @ diag
                self.dd1_sp_diaginv = 1 / \
                    (diag + square_sparse(self.e1) @
                     extract_diag_vector(sp))
                diag_dbc = self.m0_sp_diaginv_dbc @ square_sparse(self.e0_dbc)
                diag_dbc = square_sparse(self.d0_sp) @ diag_dbc
                diag_dbc = square_sparse(self.e1_dbc) @ diag_dbc
                self.dd1_sp_diaginv_dbc = 1 / \
                    (diag_dbc + square_sparse(self.e1_dbc) @
                     extract_diag_vector(sp))
                self.curl_curl_sp = jsparse.BCSR.from_bcoo(sp)
            case 2:
                W_scalar = (1 / self.jacobian_j) * self.quad.w
                W_3x3 = W_scalar[:, None, None] * jnp.ones((1, 3, 3))
                div_terms = [
                    [(0, grad_r, self.d_basis_t_jk, self.d_basis_z_jk, +1)],
                    [(1, self.d_basis_r_jk, grad_t, self.d_basis_z_jk, +1)],
                    [(2, self.d_basis_r_jk, self.d_basis_t_jk, grad_z, +1)],
                ]
                sp = assemble_vectorial_tp(
                    div_terms, div_terms, W_3x3, quad_shape,
                    list(self.basis_2.shape), self.basis_2.pr)
                diag = self.m1_sp_diaginv @ square_sparse(self.e1)
                diag = square_sparse(self.d1_sp) @ diag
                diag = square_sparse(self.e2) @ diag
                self.dd2_sp_diaginv = 1 / \
                    (diag + square_sparse(self.e2) @
                     extract_diag_vector(sp))
                diag_dbc = self.m1_sp_diaginv_dbc @ square_sparse(self.e1_dbc)
                diag_dbc = square_sparse(self.d1_sp) @ diag_dbc
                diag_dbc = square_sparse(self.e2_dbc) @ diag_dbc
                self.dd2_sp_diaginv_dbc = 1 / \
                    (diag_dbc + square_sparse(self.e2_dbc) @
                     extract_diag_vector(sp))
                self.div_div_sp = jsparse.BCSR.from_bcoo(sp)
            case 3:
                diag = self.m2_sp_diaginv @ square_sparse(self.e2)
                diag = square_sparse(self.d2_sp) @ diag
                self.dd3_sp_diaginv = 1 / (square_sparse(self.e3) @ diag)
                diag_dbc = self.m2_sp_diaginv_dbc @ square_sparse(self.e2_dbc)
                diag_dbc = square_sparse(self.d2_sp) @ diag_dbc
                self.dd3_sp_diaginv_dbc = 1 / \
                    (square_sparse(self.e3_dbc) @ diag_dbc)
            case _:
                raise ValueError("k must be 0, 1, 2, or 3")

    def assemble_leray_projection(self):
        """
        Assemble the Leray projection matrix. Formula:
            P_Leray = I + weak_grad @ (dd3)^-1 @ strong_div
        """
        self.P_Leray = jnp.eye(self.m2.shape[0]) + \
            self.weak_grad @ jnp.linalg.pinv(self.dd3) @ self.strong_div

    # TODO: We can pre-compute strong operators, they are sparse
    def apply_strong_grad(self, v, dirichlet_in=True, dirichlet_out=True):
        """
        Apply the strong gradient operator to a vector v.
        """
        dv_dual = self.apply_derivative_matrix(
            v, 0, dirichlet_in=dirichlet_in, dirichlet_out=dirichlet_out)
        return self.apply_inverse_mass_matrix(dv_dual, 1, dirichlet=dirichlet_out)

    def apply_strong_curl(self, v, dirichlet_in=True, dirichlet_out=True):
        """
        Apply the strong curl operator to a vector v.
        """
        dv_dual = self.apply_derivative_matrix(
            v, 1, dirichlet_in=dirichlet_in, dirichlet_out=dirichlet_out)
        return self.apply_inverse_mass_matrix(dv_dual, 2, dirichlet=dirichlet_out)

    def apply_strong_div(self, v, dirichlet_in=True, dirichlet_out=True):
        """
        Apply the strong divergence operator to a vector v.
        """
        dv_dual = self.apply_derivative_matrix(
            v, 2, dirichlet_in=dirichlet_in, dirichlet_out=dirichlet_out)
        return self.apply_inverse_mass_matrix(dv_dual, 3, dirichlet=dirichlet_out)

    def apply_weak_grad(self, v, dirichlet_in=True, dirichlet_out=True):
        """
        Apply the weak gradient operator to a vector v.
        """
        dv_dual = -self.apply_derivative_matrix(
            v, 2, dirichlet_in=dirichlet_in, dirichlet_out=dirichlet_out, transpose=True)
        return self.apply_inverse_mass_matrix(dv_dual, 2, dirichlet=dirichlet_out)

    def apply_weak_curl(self, v, dirichlet_in=True, dirichlet_out=True):
        """
        Apply the weak curl operator to a vector v.
        """
        dv_dual = self.apply_derivative_matrix(
            v, 1, dirichlet_in=dirichlet_in, dirichlet_out=dirichlet_out, transpose=True)
        return self.apply_inverse_mass_matrix(dv_dual, 1, dirichlet=dirichlet_out)

    def apply_weak_div(self, v, dirichlet_in=True, dirichlet_out=True):
        """
        Apply the weak divergence operator to a vector v.
        """
        dv_dual = -self.apply_derivative_matrix(
            v, 0, dirichlet_in=dirichlet_in, dirichlet_out=dirichlet_out, transpose=True)
        return self.apply_inverse_mass_matrix(dv_dual, 0, dirichlet=dirichlet_out)

    def apply_mass_matrix_preconditioner(self, v, k, dirichlet=True):
        """
        Apply the diagonal Jacobi preconditioner for the mass matrix Mk to a vector v.
        """
        match k:
            case 0:
                if dirichlet:
                    return self.m0_sp_diaginv_dbc * v
                else:
                    return self.m0_sp_diaginv * v
            case 1:
                if dirichlet:
                    return self.m1_sp_diaginv_dbc * v
                else:
                    return self.m1_sp_diaginv * v
            case 2:
                if dirichlet:
                    return self.m2_sp_diaginv_dbc * v
                else:
                    return self.m2_sp_diaginv * v
            case 3:
                if dirichlet:
                    return self.m3_sp_diaginv_dbc * v
                else:
                    return self.m3_sp_diaginv * v
            case _:
                raise ValueError("k must be 0, 1, 2 or 3")

    def apply_inverse_mass_matrix(self, v, k, dirichlet=True, guess=None):
        """
        Apply the inverse of the sparse mass matrix Mk⁻¹ for k-forms to a vector v,
        solved via CG with Jacobi preconditioning. An optional initial guess can be
        provided to warm-start the solver.
        """
        return solve_singular_cg(
            lambda x: self.apply_mass_matrix(x, k, dirichlet=dirichlet),
            v,
            mass_matvec=lambda x: self.apply_mass_matrix(
                x, k, dirichlet=dirichlet),
            precond_matvec=lambda x: self.apply_mass_matrix_preconditioner(
                x, k, dirichlet=dirichlet),
            x0=guess,
            tol=self.tol, maxiter=self.maxiter)[0]

    def apply_mass_matrix(self, v, k, dirichlet=True):
        """
        Apply the sparse mass matrix Mk for k-forms to a vector v:
            k=0: M0_ij = ∫ Λ0_i Λ0_j det DF dx
            k=1: M1_ij = ∫ Λ1_i · G⁻¹ Λ1_j det DF dx
            k=2: M2_ij = ∫ Λ2_i · G Λ2_j (det DF)⁻¹ dx
            k=3: M3_ij = ∫ Λ3_i Λ3_j (det DF)⁻¹ dx
        """
        match k:
            case 0:
                e = self.e0_dbc if dirichlet else self.e0
                e_T = self.e0_dbc_T if dirichlet else self.e0_T
                sp = self.m0_sp
            case 1:
                e = self.e1_dbc if dirichlet else self.e1
                e_T = self.e1_dbc_T if dirichlet else self.e1_T
                sp = self.m1_sp
            case 2:
                e = self.e2_dbc if dirichlet else self.e2
                e_T = self.e2_dbc_T if dirichlet else self.e2_T
                sp = self.m2_sp
            case 3:
                e = self.e3_dbc if dirichlet else self.e3
                e_T = self.e3_dbc_T if dirichlet else self.e3_T
                sp = self.m3_sp
            case _:
                raise ValueError("k must be 0, 1, 2 or 3")
        return e @ (sp @ (e_T @ v))

    def apply_projection_matrix(self, v, k_in, k_out, dirichlet_in=True, dirichlet_out=True):
        """
        Apply the sparse projection matrix Pk_in_k_out to a vector v.
        """
        match (k_in, k_out):
            case (2, 1):
                if dirichlet_out and dirichlet_in:
                    return self.e1_dbc @ (self.m12_sp @ (self.e2_dbc_T @ v))
                elif dirichlet_out and not dirichlet_in:
                    return self.e1_dbc @ (self.m12_sp @ (self.e2_T @ v))
                elif not dirichlet_out and dirichlet_in:
                    return self.e1 @ (self.m12_sp @ (self.e2_dbc_T @ v))
                else:
                    return self.e1 @ (self.m12_sp @ (self.e2_T @ v))
            case (1, 2):
                if dirichlet_out and dirichlet_in:
                    return self.e2_dbc @ (self.m21_sp @ (self.e1_dbc_T @ v))
                elif dirichlet_out and not dirichlet_in:
                    return self.e2_dbc @ (self.m21_sp @ (self.e1_T @ v))
                elif not dirichlet_out and dirichlet_in:
                    return self.e2 @ (self.m21_sp @ (self.e1_dbc_T @ v))
                else:
                    return self.e2 @ (self.m21_sp @ (self.e1_T @ v))
            case (0, 3):
                if dirichlet_out and dirichlet_in:
                    return self.e0_dbc @ (self.m03_sp @ (self.e3_dbc_T @ v))
                elif dirichlet_out and not dirichlet_in:
                    return self.e0_dbc @ (self.m03_sp @ (self.e3_T @ v))
                elif not dirichlet_out and dirichlet_in:
                    return self.e0 @ (self.m03_sp @ (self.e3_dbc_T @ v))
                else:
                    return self.e0 @ (self.m03_sp @ (self.e3_T @ v))
            case (3, 0):
                if dirichlet_out and dirichlet_in:
                    return self.e3_dbc @ (self.m30_sp @ (self.e0_dbc_T @ v))
                elif dirichlet_out and not dirichlet_in:
                    return self.e3_dbc @ (self.m30_sp @ (self.e0_T @ v))
                elif not dirichlet_out and dirichlet_in:
                    return self.e3 @ (self.m30_sp @ (self.e0_dbc_T @ v))
                else:
                    return self.e3 @ (self.m30_sp @ (self.e0_T @ v))
            case _:
                raise ValueError(
                    "Only (k_in, k_out) = (1, 2), (2, 1), (0, 3), or (3, 0) supported")

    def apply_derivative_matrix(self, v, k, dirichlet_in=True, dirichlet_out=True, transpose=False):
        """
        Apply the derivative matrix Dk (mapping k-forms to (k+1)-forms) to a vector v:
            k=0: D0_ij = ∫ Λ1_i · G⁻¹ grad Λ0_j det DF dx  (grad)
            k=1: D1_ij = ∫ Λ2_i · G curl Λ1_j (det DF)⁻¹ dx  (curl)
            k=2: D2_ij = ∫ Λ3_i div Λ2_j (det DF)⁻¹ dx  (div)
        If transpose=True, apply Dk.T instead (mapping (k+1)-forms to k-forms).
        """
        match k:
            case 0:
                e_in = self.e0_dbc if dirichlet_in else self.e0
                e_in_T = self.e0_dbc_T if dirichlet_in else self.e0_T
                e_out = self.e1_dbc if dirichlet_out else self.e1
                e_out_T = self.e1_dbc_T if dirichlet_out else self.e1_T
                sp = self.d0_sp
                sp_T = self.d0_sp_T
            case 1:
                e_in = self.e1_dbc if dirichlet_in else self.e1
                e_in_T = self.e1_dbc_T if dirichlet_in else self.e1_T
                e_out = self.e2_dbc if dirichlet_out else self.e2
                e_out_T = self.e2_dbc_T if dirichlet_out else self.e2_T
                sp = self.d1_sp
                sp_T = self.d1_sp_T
            case 2:
                e_in = self.e2_dbc if dirichlet_in else self.e2
                e_in_T = self.e2_dbc_T if dirichlet_in else self.e2_T
                e_out = self.e3_dbc if dirichlet_out else self.e3
                e_out_T = self.e3_dbc_T if dirichlet_out else self.e3_T
                sp = self.d2_sp
                sp_T = self.d2_sp_T
            case _:
                raise ValueError("k must be 0, 1 or 2")
        if transpose:
            return e_in @ (sp_T @ (e_out_T @ v))
        return e_out @ (sp @ (e_in_T @ v))

    def apply_hodge_laplacian(self, v, k, dirichlet=True):
        """
        Apply the k-th Hodge Laplacian (δd) to a vector v.

        For k ≥ 1, applied via the saddle-point Schur complement:

        | stiffness_k   d_{k-1}  |   | u (k-form)      |
        | d_{k-1}^T    -M_{k-1}  |   | s ((k-1)-form)  |

        where stiffness_k_ij = ∫ d Λ^k_i · d Λ^k_j dx (assembled directly),
        and d_{k-1} is the discrete exterior derivative from (k-1)- to k-forms.
        Eliminating s = M_{k-1}^{-1} d_{k-1}^T u gives the Schur complement:

            (stiffness_k + d_{k-1} M_{k-1}^{-1} d_{k-1}^T) u

        Concretely:
            k=0: stiffness_0 = grad_grad_ij = ∫ ∇Λ0_i · G⁻¹ ∇Λ0_j det DF dx
            k=1: stiffness_1 = curl_curl_ij = ∫ curl Λ1_i · G curl Λ1_j (det DF)⁻¹ dx,  d_0 = grad
            k=2: stiffness_2 = div_div_ij   = ∫ div  Λ2_i div  Λ2_j (det DF)⁻¹ dx,  d_1 = curl
            k=3: stiffness_3 = 0,  d_2 = div

        The inner M_{k-1}^{-1} solves use CG with Jacobi preconditioning.
        """
        match k:
            case 0:
                e = self.e0_dbc if dirichlet else self.e0
                e_T = self.e0_dbc_T if dirichlet else self.e0_T
                return e @ (self.grad_grad_sp @ (e_T @ v))
            case 1:
                e1 = self.e1_dbc if dirichlet else self.e1
                e1_T = self.e1_dbc_T if dirichlet else self.e1_T
                minus_div_v = self.apply_derivative_matrix(
                    v, 0, dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
                m0_inv_div_v = solve_singular_cg(
                    lambda x: self.apply_mass_matrix(
                        x, 0, dirichlet=dirichlet),
                    minus_div_v,
                    mass_matvec=lambda x: self.apply_mass_matrix(
                        x, 0, dirichlet=dirichlet),
                    precond_matvec=lambda x: self.apply_mass_matrix_preconditioner(
                        x, 0, dirichlet=dirichlet),
                    tol=self.tol, maxiter=self.maxiter)[0]
                return e1 @ (self.curl_curl_sp @ (e1_T @ v)) \
                    + self.apply_derivative_matrix(m0_inv_div_v, 0,
                                                   dirichlet_in=dirichlet, dirichlet_out=dirichlet)
            case 2:
                e2 = self.e2_dbc if dirichlet else self.e2
                e2_T = self.e2_dbc_T if dirichlet else self.e2_T
                curl_v = self.apply_derivative_matrix(
                    v, 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
                m1_inv_curl_v = solve_singular_cg(
                    lambda x: self.apply_mass_matrix(
                        x, 1, dirichlet=dirichlet),
                    curl_v,
                    mass_matvec=lambda x: self.apply_mass_matrix(
                        x, 1, dirichlet=dirichlet),
                    precond_matvec=lambda x: self.apply_mass_matrix_preconditioner(
                        x, 1, dirichlet=dirichlet),
                    tol=self.tol, maxiter=self.maxiter)[0]
                return e2 @ (self.div_div_sp @ (e2_T @ v)) \
                    + self.apply_derivative_matrix(m1_inv_curl_v, 1,
                                                   dirichlet_in=dirichlet, dirichlet_out=dirichlet)
            case 3:
                minus_grad_v = self.apply_derivative_matrix(
                    v, 2, dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
                m2_inv_minus_grad_v = solve_singular_cg(
                    lambda x: self.apply_mass_matrix(
                        x, 2, dirichlet=dirichlet),
                    minus_grad_v,
                    mass_matvec=lambda x: self.apply_mass_matrix(
                        x, 2, dirichlet=dirichlet),
                    precond_matvec=lambda x: self.apply_mass_matrix_preconditioner(
                        x, 2, dirichlet=dirichlet),
                    tol=self.tol, maxiter=self.maxiter)[0]
                return self.apply_derivative_matrix(m2_inv_minus_grad_v, 2, dirichlet_in=dirichlet, dirichlet_out=dirichlet)
            case _:
                raise ValueError("k must be 0, 1, 2 or 3")

    def apply_inverse_hodge_laplacian(self, v, k, dirichlet=True, guess=None):
        """
        Apply the inverse of the k-th Hodge Laplacian (δd)⁻¹ to a vector v,
        solved via CG with Jacobi preconditioning. An optional initial guess can be
        provided to warm-start the solver.
        """
        if k == 0:
            if dirichlet:
                vs = self.null_0_dbc
            else:
                vs = self.null_0
        elif k == 1:
            if dirichlet:
                vs = self.null_1_dbc
            else:
                vs = self.null_1
        elif k == 2:
            if dirichlet:
                vs = self.null_2_dbc
            else:
                vs = self.null_2
        elif k == 3:
            if dirichlet:
                vs = self.null_3_dbc
            else:
                vs = self.null_3

        return solve_singular_cg(
            lambda x: self.apply_hodge_laplacian(x, k, dirichlet=dirichlet),
            v,
            mass_matvec=lambda x: self.apply_mass_matrix(
                x, k, dirichlet=dirichlet),
            precond_matvec=lambda x: self.apply_hodge_laplacian_preconditioner(
                x, k, dirichlet=dirichlet),
            x0=guess,
            vs=vs,
            tol=self.tol, maxiter=self.maxiter)[0]

    def apply_hodge_laplacian_preconditioner(self, v, k, dirichlet=True):
        """
        Apply the Jacobi preconditioner for the k-th Hodge Laplacian to a vector v.
        """
        match k:
            case 0:
                diaginv = self.dd0_sp_diaginv_dbc if dirichlet else self.dd0_sp_diaginv
            case 1:
                diaginv = self.dd1_sp_diaginv_dbc if dirichlet else self.dd1_sp_diaginv
            case 2:
                diaginv = self.dd2_sp_diaginv_dbc if dirichlet else self.dd2_sp_diaginv
            case 3:
                diaginv = self.dd3_sp_diaginv_dbc if dirichlet else self.dd3_sp_diaginv
            case _:
                raise ValueError("k must be 0, 1, 2 or 3")
        return diaginv * v

# %%

    def compute_nullspaces(self):
        """
        Compute the nullspace of the k-th Hodge Laplacian using randomized SVD.
        TODO: For now this only handles the case where the nullspace is 1-dim.
        """
        self.null_0_dbc = []
        self.null_1_dbc = []
        v3 = self.apply_inverse_mass_matrix(
            jnp.ones(self.n3_dbc), 3, dirichlet=True)
        v3 /= self.l2_norm(v3, 3, dirichlet=True)
        self.null_3_dbc = [v3]
        v, _ = self.apply_leray_projection(
            jnp.ones(self.n2_dbc), k=2)
        curl_v_dual = self.apply_derivative_matrix(
            v, 1, dirichlet_in=True, dirichlet_out=True, transpose=True)
        a = self.apply_inverse_hodge_laplacian(curl_v_dual, 1, dirichlet=True)
        curl_a = self.apply_strong_curl(a, True, True)
        v2 = v - curl_a
        v2 /= self.l2_norm(v2, 2, dirichlet=True)
        self.null_2_dbc = [v2]

        # no Dirichlet BCs (all defaults to False)
        v0 = jnp.ones(self.n0)
        v0 /= self.l2_norm(v0, 0, False)
        self.null_0 = [v0]
        self.null_2 = []
        v, _ = self.apply_leray_projection(
            jnp.ones(self.n1), k=1)
        curl_v_dual = self.apply_derivative_matrix(
            v, 1, dirichlet_in=False, dirichlet_out=False)
        a = self.apply_inverse_hodge_laplacian(curl_v_dual, 2, dirichlet=False)
        curl_a = self.apply_weak_curl(a, False, False)
        v1 = v - curl_a
        v1 /= self.l2_norm(v1, 1, False)
        self.null_1 = [v1]
        self.null_3 = []

    def cross_product_projection_deprecated(
        self, w, u, n, m, k,
        dirichlet_n=True,
        dirichlet_m=True,
        dirichlet_k=True
    ):
        """
        Evaluate the projection of the cross product of the m-form w 
        and the k-form u onto an n-form.
        TODO: Tobi please add a description of these projections.

        Args:
            w (array): m-form dofs
            u (array): k-form dofs
            n, m, k (ints): degree of the forms
            dirichlet_n: boundary condtions on the n-form (default True)
            dirichlet_m: boundary condtions on the m-form (default True)
            dirichlet_k: boundary condtions on the k-form (default True)

        Returns:
            array: ∫ (wₕ × uₕ) · Λn[i] dx for all i

        """
        match n:
            case 1:
                if dirichlet_n:
                    en = self.e1_dbc
                else:
                    en = self.e1
                eval_basis_n_ijk = self.eval_basis_1_ijk
                nn = self.basis_1.n
            case 2:
                if dirichlet_n:
                    en = self.e2_dbc
                else:
                    en = self.e2
                eval_basis_n_ijk = self.eval_basis_2_ijk
                nn = self.basis_2.n
            case _:
                raise ValueError("n must be 1 or 2")
        match m:
            case 1:
                if dirichlet_m:
                    em_T = self.e1_dbc_T
                else:
                    em_T = self.e1_T
                eval_basis_m_ijk = self.eval_basis_1_ijk
            case 2:
                if dirichlet_m:
                    em_T = self.e2_dbc_T
                else:
                    em_T = self.e2_T
                eval_basis_m_ijk = self.eval_basis_2_ijk
            case _:
                raise ValueError("m must be 1 or 2")
        match k:
            case 1:
                if dirichlet_k:
                    ek_T = self.e1_dbc_T
                else:
                    ek_T = self.e1_T
                eval_basis_k_ijk = self.eval_basis_1_ijk
            case 2:
                if dirichlet_k:
                    ek_T = self.e2_dbc_T
                else:
                    ek_T = self.e2_T
                eval_basis_k_ijk = self.eval_basis_2_ijk
            case _:
                raise ValueError("k must be 1 or 2")

        # w and u evaluated at quadrature points: shape: n_q x 3
        w_jk = evaluate_at_xq_deprecated(eval_basis_m_ijk,
                                         em_T @ w, self.quad.n, 3)
        u_jk = evaluate_at_xq_deprecated(eval_basis_k_ijk,
                                         ek_T @ u, self.quad.n, 3)

        # now, we compute
        # ∑ Λn[i](x_j)_a w(x_j)_b u(x_j)_c ) t(x_j)_abc
        # where t is some transformation depending on n,m,k and the metric
        # and we sum over j (quadrature points) and b,c (dimensions)
        # To avoid assembling the huge Λn[i](x_j)_a tensor, we scan over i.
        if n == 1 and m == 2 and k == 1:
            # ∫ Λ[i] (Gw x u) / J dx
            Gw_jk = jnp.einsum('jkl,jk->jl', self.metric_jkl, w_jk)
            Gw_x_u_jk = jnp.cross(Gw_jk, u_jk, axis=1)
            f_jk = Gw_x_u_jk * (self.quad.w / self.jacobian_j)[:, None]
        elif n == 1 and m == 1 and k == 1:
            # ∫ Λ[i] (w x u) dx
            w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
            f_jk = w_x_u_jk * (self.quad.w)[:, None]
        elif n == 2 and m == 1 and k == 1:
            # ∫ Λ[i] G(w x u) / J dx
            w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
            G_wxu_jk = jnp.einsum('jkl,jk->jl', self.metric_jkl, w_x_u_jk)
            f_jk = G_wxu_jk * (self.quad.w / self.jacobian_j)[:, None]
        elif n == 2 and m == 2 and k == 1:
            # ∫ Λ[i] (w x G_inv u) dx
            Ginvu_jk = jnp.einsum('jkl,jk->jl', self.metric_inv_jkl, u_jk)
            w_x_Ginvu_jk = jnp.cross(w_jk, Ginvu_jk, axis=1)
            f_jk = w_x_Ginvu_jk * (self.quad.w)[:, None]
        elif n == 1 and m == 2 and k == 2:
            # ∫ Λ[i] G_inv(w x u) dx
            w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
            Ginv_wxu_jk = jnp.einsum(
                'jkl,jk->jl', self.metric_inv_jkl, w_x_u_jk)
            f_jk = Ginv_wxu_jk * (self.quad.w)[:, None]
        elif n == 2 and m == 1 and k == 2:
            # ∫ Λ[i] (G_inv w x u) dx
            Ginvw_jk = jnp.einsum('jkl,jk->jl', self.metric_inv_jkl, w_jk)
            Ginvw_x_u_jk = jnp.cross(Ginvw_jk, u_jk, axis=1)
            f_jk = Ginvw_x_u_jk * (self.quad.w)[:, None]
        elif n == 2 and m == 2 and k == 2:
            # ∫ Λ[i] (w x u) / J dx
            w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
            f_jk = w_x_u_jk * (self.quad.w / self.jacobian_j)[:, None]
        else:
            raise ValueError("Not yet implemented")
        return en @ integrate_against_deprecated(eval_basis_n_ijk, f_jk, nn)

    def cross_product_projection(
        self, w, u, n, m, k,
        dirichlet_n=True,
        dirichlet_m=True,
        dirichlet_k=True
    ):
        """Evaluate the cross-product projection using tensor-product eval/integrate.

        Same interface as cross_product_projection but uses TP structure
        for evaluate_at_xq and integrate_against.
        """
        from mrx.utils import evaluate_at_xq, integrate_against
        quad_shape = (self.quad.ny, self.quad.nx, self.quad.nz)

        # Extraction matrices and comp_info for output form n
        match n:
            case 1:
                en = self.e1_dbc if dirichlet_n else self.e1
                comp_info_n, comp_shapes_n = self._form_comp_info(1)
                nn = self.basis_1.n
            case 2:
                en = self.e2_dbc if dirichlet_n else self.e2
                comp_info_n, comp_shapes_n = self._form_comp_info(2)
                nn = self.basis_2.n
            case _:
                raise ValueError("n must be 1 or 2")

        # Extraction matrices and comp_info for input form m
        match m:
            case 1:
                em_T = self.e1_dbc_T if dirichlet_m else self.e1_T
                comp_info_m, comp_shapes_m = self._form_comp_info(1)
            case 2:
                em_T = self.e2_dbc_T if dirichlet_m else self.e2_T
                comp_info_m, comp_shapes_m = self._form_comp_info(2)
            case _:
                raise ValueError("m must be 1 or 2")

        # Extraction matrices and comp_info for input form k
        match k:
            case 1:
                ek_T = self.e1_dbc_T if dirichlet_k else self.e1_T
                comp_info_k, comp_shapes_k = self._form_comp_info(1)
            case 2:
                ek_T = self.e2_dbc_T if dirichlet_k else self.e2_T
                comp_info_k, comp_shapes_k = self._form_comp_info(2)
            case _:
                raise ValueError("k must be 1 or 2")

        # TP evaluation at quadrature points
        w_jk = evaluate_at_xq(em_T @ w, comp_info_m, comp_shapes_m,
                              quad_shape, 3)
        u_jk = evaluate_at_xq(ek_T @ u, comp_info_k, comp_shapes_k,
                              quad_shape, 3)

        # Nonlinear part (same as original)
        if n == 1 and m == 2 and k == 1:
            Gw_jk = jnp.einsum('jkl,jk->jl', self.metric_jkl, w_jk)
            Gw_x_u_jk = jnp.cross(Gw_jk, u_jk, axis=1)
            f_jk = Gw_x_u_jk * (self.quad.w / self.jacobian_j)[:, None]
        elif n == 1 and m == 1 and k == 1:
            w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
            f_jk = w_x_u_jk * (self.quad.w)[:, None]
        elif n == 2 and m == 1 and k == 1:
            w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
            G_wxu_jk = jnp.einsum('jkl,jk->jl', self.metric_jkl, w_x_u_jk)
            f_jk = G_wxu_jk * (self.quad.w / self.jacobian_j)[:, None]
        elif n == 2 and m == 2 and k == 1:
            Ginvu_jk = jnp.einsum('jkl,jk->jl', self.metric_inv_jkl, u_jk)
            w_x_Ginvu_jk = jnp.cross(w_jk, Ginvu_jk, axis=1)
            f_jk = w_x_Ginvu_jk * (self.quad.w)[:, None]
        elif n == 1 and m == 2 and k == 2:
            w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
            Ginv_wxu_jk = jnp.einsum(
                'jkl,jk->jl', self.metric_inv_jkl, w_x_u_jk)
            f_jk = Ginv_wxu_jk * (self.quad.w)[:, None]
        elif n == 2 and m == 1 and k == 2:
            Ginvw_jk = jnp.einsum('jkl,jk->jl', self.metric_inv_jkl, w_jk)
            Ginvw_x_u_jk = jnp.cross(Ginvw_jk, u_jk, axis=1)
            f_jk = Ginvw_x_u_jk * (self.quad.w)[:, None]
        elif n == 2 and m == 2 and k == 2:
            w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
            f_jk = w_x_u_jk * (self.quad.w / self.jacobian_j)[:, None]
        else:
            raise ValueError("Not yet implemented")

        # TP integration
        return en @ integrate_against(
            f_jk, comp_info_n, comp_shapes_n, quad_shape)

    def apply_leray_projection(self, v, k=2, p_guess=None):
        """
        Apply the Leray projection to a 1 or 2-form v.

        When k = 2:
            Solves the system (k=3 Hodge Laplacian):
            div v = div σ
            (σ, ω) = -(p, div ω) ∀ω 2-forms
            -> div(v - σ) = 0 and σ.n = 0 on the boundary.
        When k = 1:
            Solves the k=0 Hodge Laplacian:
            (grad p, grad ω) = (v, grad ω) ∀ω 0-forms
            -> div(v - grad p) = 0 and p = 0 on the boundary.

        Parameters
        ----------
        v : jnp.ndarray 
            The vector form DoFs
        k : int
            The degree of the vector form
        p_guess : jnp.ndarray 
            Guess for pressure form DoFs

        Returns
        -------
        v_out : jnp.ndarray 
            divergence-cleaned v
        p : jnp.ndarray 
            The pressure form DoFs

        """
        if k == 2:
            p_guess = jnp.zeros(self.n3_dbc) if p_guess is None else p_guess
            # Assumes dirichlet == True on all spaces.
            div_v = self.apply_derivative_matrix(
                v, 2, dirichlet_in=True, dirichlet_out=True)
            p = self.apply_inverse_hodge_laplacian(
                div_v, 3, dirichlet=True, guess=p_guess)
            σ = -self.apply_weak_grad(p, True, True)
            return v - σ, p
        elif k == 1:
            # Assumes dirichlet == False on all spaces.
            p_guess = jnp.zeros(self.n0) if p_guess is None else p_guess
            div_v = -self.apply_derivative_matrix(
                v, 0, dirichlet_in=False, dirichlet_out=False, transpose=True)
            p = self.apply_inverse_hodge_laplacian(
                div_v, 0, dirichlet=False, guess=p_guess)
            σ = -self.apply_strong_grad(p, False, False)
            return v - σ, p
