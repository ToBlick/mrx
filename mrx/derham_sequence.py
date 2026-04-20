from typing import Callable

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

import mrx
from mrx.assembly import (assemble_all_sparse, assemble_derivative_matrix,
                          assemble_derivative_matrix_deprecated,
                          assemble_hodge_laplacian,
                          assemble_hodge_laplacian_deprecated,
                          assemble_leray_projection, assemble_mass_matrix,
                          assemble_mass_matrix_deprecated,
                          assemble_projection_matrix,
                          assemble_projection_matrix_deprecated,
                          eval_basis_0_ijk, eval_basis_1_ijk, eval_basis_2_ijk,
                          eval_basis_3_ijk, eval_d_basis_0_ijk,
                          eval_d_basis_1_ijk, eval_d_basis_2_ijk, grad_1d)
from mrx.differential_forms import DifferentialForm
from mrx.extraction_operators import (BoundaryOperator,
                                      PolarExtractionOperator, get_xi)
from mrx.nullspace import (compute_nullspaces, compute_nullspaces_iterative,
                           find_nullspace_vectors, get_nullspace,
                           get_saddle_point_nullspaces)
from mrx.projectors import Projector
from mrx.quadrature import QuadratureRule
from mrx.solvers import solve_saddle_point_minres, solve_singular_cg
from mrx.utils import (evaluate_at_xq_deprecated, extract_diag_vector,
                       integrate_against_deprecated, inv33,
                       jacobian_determinant, square_sparse)


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

    def __init__(self, ns, ps, q, types, map, polar, tol=1e-12, maxiter=10_000, r_scale=1.0, n_inner=5):
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
            n_inner (int): Number of CG iterations for block preconditioner solves.
        """
        self.tol = tol
        self.maxiter = maxiter
        self.n_inner = n_inner
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
                PolarExtractionOperator(Λ, xi, False)
                for Λ in [self.basis_0, self.basis_1, self.basis_2, self.basis_3]
            ]
            e0_dbc, e1_dbc, e2_dbc, e3_dbc = [
                PolarExtractionOperator(Λ, xi, True)
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
        return eval_basis_0_ijk(self, i, j, k)

    def eval_d_basis_0_ijk(self, i, j, k):
        return eval_d_basis_0_ijk(self, i, j, k)

    def eval_basis_1_ijk(self, i, j, k):
        return eval_basis_1_ijk(self, i, j, k)

    def eval_d_basis_1_ijk(self, i, j, k):
        return eval_d_basis_1_ijk(self, i, j, k)

    def eval_basis_2_ijk(self, i, j, k):
        return eval_basis_2_ijk(self, i, j, k)

    def eval_d_basis_2_ijk(self, i, j, k):
        return eval_d_basis_2_ijk(self, i, j, k)

    def eval_basis_3_ijk(self, i, j, k):
        return eval_basis_3_ijk(self, i, j, k)

    def l2_norm_sq(self, v, k, dirichlet=True):
        return v @ self.apply_mass_matrix(v, k, dirichlet=dirichlet)

    def l2_norm(self, v, k, dirichlet=True):
        return jnp.sqrt(self.l2_norm_sq(v, k, dirichlet=dirichlet))

    def assemble_all_sparse(self):
        assemble_all_sparse(self)

    def assemble_mass_matrix_deprecated(self, k):
        assemble_mass_matrix_deprecated(self, k)

    def assemble_mass_matrix(self, k):
        assemble_mass_matrix(self, k)

    def assemble_projection_matrix_deprecated(self, k_from, k_to):
        assemble_projection_matrix_deprecated(self, k_from, k_to)

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
        assemble_projection_matrix(self, k_from, k_to)

    def assemble_derivative_matrix_deprecated(self, k):
        assemble_derivative_matrix_deprecated(self, k)

    def assemble_hodge_laplacian_deprecated(self, k):
        assemble_hodge_laplacian_deprecated(self, k)

    def assemble_derivative_matrix(self, k):
        assemble_derivative_matrix(self, k)

    def _grad_1d(self, d_basis, boundary_type):
        return grad_1d(d_basis, boundary_type)

    def assemble_hodge_laplacian(self, k):
        assemble_hodge_laplacian(self, k)

    def assemble_leray_projection(self):
        assemble_leray_projection(self)

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

        The inner M_{k-1}^{-1} solves use CG with Jacobi preconditioning
        to full solver tolerance.
        """
        match k:
            case 0:
                e = self.e0_dbc if dirichlet else self.e0
                e_T = self.e0_dbc_T if dirichlet else self.e0_T
                return e @ (self.grad_grad_sp @ (e_T @ v))
            case 1:
                e1 = self.e1_dbc if dirichlet else self.e1
                e1_T = self.e1_dbc_T if dirichlet else self.e1_T
                Dt_v = self.apply_derivative_matrix(
                    v, 0, dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
                Minv_Dt_v = self.apply_inverse_mass_matrix(
                    Dt_v, 0, dirichlet=dirichlet)
                return e1 @ (self.curl_curl_sp @ (e1_T @ v)) \
                    + self.apply_derivative_matrix(Minv_Dt_v, 0,
                                                   dirichlet_in=dirichlet, dirichlet_out=dirichlet)
            case 2:
                e2 = self.e2_dbc if dirichlet else self.e2
                e2_T = self.e2_dbc_T if dirichlet else self.e2_T
                Dt_v = self.apply_derivative_matrix(
                    v, 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
                Minv_Dt_v = self.apply_inverse_mass_matrix(
                    Dt_v, 1, dirichlet=dirichlet)
                return e2 @ (self.div_div_sp @ (e2_T @ v)) \
                    + self.apply_derivative_matrix(Minv_Dt_v, 1,
                                                   dirichlet_in=dirichlet, dirichlet_out=dirichlet)
            case 3:
                Dt_v = self.apply_derivative_matrix(
                    v, 2, dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
                Minv_Dt_v = self.apply_inverse_mass_matrix(
                    Dt_v, 2, dirichlet=dirichlet)
                return self.apply_derivative_matrix(Minv_Dt_v, 2, dirichlet_in=dirichlet, dirichlet_out=dirichlet)
            case _:
                raise ValueError("k must be 0, 1, 2 or 3")

    def apply_diffusion(self, v, k, alpha, dirichlet=True):
        """Apply (M_k + alpha * L_k) to a k-form vector v."""
        return self.apply_mass_matrix(v, k, dirichlet=dirichlet) \
            + alpha * self.apply_hodge_laplacian(v, k, dirichlet=dirichlet)

    def apply_stiffness(self, v, k, dirichlet=True):
        """
        Apply the stiffness matrix S_k to a k-form vector v.

            k=0: grad_grad
            k=1: curl_curl
            k=2: div_div
            k=3: 0 (no stiffness)
        """
        match k:
            case 0:
                e = self.e0_dbc if dirichlet else self.e0
                e_T = self.e0_dbc_T if dirichlet else self.e0_T
                return e @ (self.grad_grad_sp @ (e_T @ v))
            case 1:
                e = self.e1_dbc if dirichlet else self.e1
                e_T = self.e1_dbc_T if dirichlet else self.e1_T
                return e @ (self.curl_curl_sp @ (e_T @ v))
            case 2:
                e = self.e2_dbc if dirichlet else self.e2
                e_T = self.e2_dbc_T if dirichlet else self.e2_T
                return e @ (self.div_div_sp @ (e_T @ v))
            case 3:
                return jnp.zeros_like(v)
            case _:
                raise ValueError("k must be 0, 1, 2 or 3")

    def _get_nullspace(self, k, dirichlet):
        return get_nullspace(self, k, dirichlet)

    def _get_saddle_point_nullspaces(self, k, dirichlet):
        return get_saddle_point_nullspaces(self, k, dirichlet)

    def apply_inverse_hodge_laplacian(self, v, k, dirichlet=True, guess=None):
        """Apply the inverse of the k-th Hodge Laplacian (δd)⁻¹ to a vector v."""
        return self.apply_inverse_shifted_stiffness(
            v, k, 0.0, dirichlet=dirichlet, guess=guess)

    def apply_inverse_shifted_stiffness(self, v, k, eps, dirichlet=True, guess=None):
        """
        Solve (S_k + eps * M_k) x = v for the k-form x.

        For eps=0 this reduces to the Hodge Laplacian solve; the system may be
        singular and nullspace deflation is applied automatically.
        For eps > 0 the system is nonsingular (shift-invert for S_k u = λ M_k u).

        For k=0: solved with CG.
        For k>=1: MINRES on the symmetric saddle-point system:

            | S_k + eps*M_k    D_{k-1}   | | u |   | v |
            | D_{k-1}^T       -M_{k-1}   | | σ | = | 0 |
        """
        suffix = "_dbc" if dirichlet else ""
        stiffness_diaginv = getattr(self, f"dd{k}_sp_diaginv{suffix}")
        mass_diaginv = getattr(self, f"m{k}_sp_diaginv{suffix}")
        shifted_diaginv = 1.0 / (1.0 / stiffness_diaginv + eps / mass_diaginv)

        if k == 0:
            vs = self._get_nullspace(0, dirichlet) if eps == 0 else []
            return solve_singular_cg(
                lambda x: self.apply_stiffness(x, 0, dirichlet=dirichlet)
                + eps * self.apply_mass_matrix(x, 0, dirichlet=dirichlet),
                v,
                mass_matvec=lambda x: self.apply_mass_matrix(
                    x, 0, dirichlet=dirichlet) if eps == 0 else None,
                precond_matvec=lambda x: shifted_diaginv * x,
                x0=guess,
                vs=vs,
                tol=self.tol, maxiter=self.maxiter)[0]

        # k >= 1: saddle-point MINRES
        vs_upper, vs_lower = self._get_saddle_point_nullspaces(
            k, dirichlet) if eps == 0 else ([], [])
        mass_lower_diaginv = getattr(self, f"m{k-1}_sp_diaginv{suffix}")
        n_upper = getattr(self, f"n{k}{suffix}")
        n_lower = getattr(self, f"n{k-1}{suffix}")

        def precond_upper(x):
            return shifted_diaginv * x

        def precond_lower(x):
            return mass_lower_diaginv * x

        u, sigma, info = solve_saddle_point_minres(
            stiffness_matvec=lambda x: self.apply_stiffness(
                x, k, dirichlet=dirichlet)
            + eps * self.apply_mass_matrix(x, k, dirichlet=dirichlet),
            derivative_matvec=lambda s: self.apply_derivative_matrix(
                s, k - 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet),
            derivative_T_matvec=lambda u: self.apply_derivative_matrix(
                u, k - 1, dirichlet_in=dirichlet,
                dirichlet_out=dirichlet, transpose=True),
            mass_lower_matvec=lambda s: self.apply_mass_matrix(
                s, k - 1, dirichlet=dirichlet),
            b_upper=v,
            n_upper=n_upper,
            n_lower=n_lower,
            precond_upper=precond_upper,
            precond_lower=precond_lower,
            mass_upper_matvec=lambda x: self.apply_mass_matrix(
                x, k, dirichlet=dirichlet),
            vs_upper=vs_upper,
            vs_lower=vs_lower,
            x0_upper=guess,
            tol=self.tol, maxiter=self.maxiter,
        )
        return u

    def apply_inverse_diffusion(self, v, k, alpha, dirichlet=True, guess=None):
        """
        Solve (M_k + alpha * L_k) x = v for the k-form x.

        For k=0: (M_0 + alpha * S_0) is SPD, solved with CG.
        For k>=1: uses MINRES on the symmetric saddle-point system:

            | M_k + alpha*S_k    alpha*D_{k-1}   | | u |   | v |
            | alpha*D_{k-1}^T   -alpha*M_{k-1}   | | σ | = | 0 |

        The system is nonsingular (no nullspace) since M_k + alpha*L_k is SPD.
        For small alpha, M_k^{-1} is a good upper block preconditioner.
        """
        if k == 0:
            mass_diaginv = self.m0_sp_diaginv_dbc if dirichlet else self.m0_sp_diaginv
            stiffness_diaginv = self.dd0_sp_diaginv_dbc if dirichlet else self.dd0_sp_diaginv
            diaginv = 1.0 / (1.0 / mass_diaginv + alpha / stiffness_diaginv)
            return solve_singular_cg(
                lambda x: self.apply_mass_matrix(x, 0, dirichlet=dirichlet)
                + alpha * self.apply_stiffness(x, 0, dirichlet=dirichlet),
                v,
                precond_matvec=lambda x: diaginv * x,
                x0=guess,
                tol=self.tol, maxiter=self.maxiter)[0]

        # k >= 1: saddle-point MINRES
        suffix = "_dbc" if dirichlet else ""
        mass_diaginv = getattr(self, f"m{k}_sp_diaginv{suffix}")
        mass_lower_diaginv = getattr(self, f"m{k-1}_sp_diaginv{suffix}")
        n_upper = getattr(self, f"n{k}{suffix}")
        n_lower = getattr(self, f"n{k-1}{suffix}")

        # Block-diagonal Jacobi preconditioners
        def precond_lower(x):
            return (1.0 / alpha) * mass_lower_diaginv * x

        def precond_upper(x):
            return mass_diaginv * x

        u, sigma, info = solve_saddle_point_minres(
            stiffness_matvec=lambda x: self.apply_mass_matrix(
                x, k, dirichlet=dirichlet)
            + alpha * self.apply_stiffness(x, k, dirichlet=dirichlet),
            derivative_matvec=lambda s: alpha * self.apply_derivative_matrix(
                s, k - 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet),
            derivative_T_matvec=lambda u: alpha * self.apply_derivative_matrix(
                u, k - 1, dirichlet_in=dirichlet,
                dirichlet_out=dirichlet, transpose=True),
            mass_lower_matvec=lambda s: alpha * self.apply_mass_matrix(
                s, k - 1, dirichlet=dirichlet),
            b_upper=v,
            n_upper=n_upper,
            n_lower=n_lower,
            precond_upper=precond_upper,
            precond_lower=precond_lower,
            x0_upper=guess,
            tol=self.tol, maxiter=self.maxiter,
        )
        return u

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

    def _compute_nullspaces(self, betti_numbers, eps=1e-6):
        return compute_nullspaces_iterative(self, betti_numbers, eps)

    def _find_nullspace_vectors(self, k, n_vectors, eps, dirichlet=True):
        return find_nullspace_vectors(self, k, n_vectors, eps, dirichlet)

    def compute_nullspaces(self):
        return compute_nullspaces(self)

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

    def pressure_projection(
        self, p, u, gamma,
        dirichlet_p=True,
        dirichlet_u=True,
    ):
        """Evaluate the pressure projection -(grad p · u + γ p div u).

        Computes the 0-form dual DOF vector:

            q_i = ∫ Λ⁰_i (−∇p · u − γ p div u) w dx

        The 0-form mass matrix weight J cancels with the 1/J from the
        wedge product (1-form · 2-form) and from div = (1/J) div_logical,
        so the integrand has no metric or Jacobian — only quad weights.

        Parameters
        ----------
        p : array  –  0-form DOFs
        u : array  –  2-form DOFs
        gamma : float  –  adiabatic exponent
        dirichlet_p : bool  –  Dirichlet BCs on p
        dirichlet_u : bool  –  Dirichlet BCs on u

        Returns
        -------
        q_dual : array  –  0-form dual DOFs (apply M0⁻¹ to get primal DOFs)
        """
        from mrx.utils import evaluate_at_xq, integrate_against
        quad_shape = (self.quad.ny, self.quad.nx, self.quad.nz)

        types = self.basis_0.types
        grad_r = self._grad_1d(self.d_basis_r_jk, types[0])
        grad_t = self._grad_1d(self.d_basis_t_jk, types[1])
        grad_z = self._grad_1d(self.d_basis_z_jk, types[2])

        # --- evaluate p at quad points (0-form, 1 component) ---
        ep_T = self.e0_dbc_T if dirichlet_p else self.e0_T
        comp_info_0, comp_shapes_0 = self._form_comp_info(0)
        p_jk = evaluate_at_xq(ep_T @ p, comp_info_0, comp_shapes_0,
                              quad_shape, 1)  # (n_q, 1)

        # --- evaluate grad(p) at quad points (3 components) ---
        s0 = list(self.basis_0.shape)[0]
        d0_comp_info = [
            (0, grad_r, self.basis_t_jk, self.basis_z_jk),
            (1, self.basis_r_jk, grad_t, self.basis_z_jk),
            (2, self.basis_r_jk, self.basis_t_jk, grad_z),
        ]
        d0_comp_shapes = [s0, s0, s0]
        grad_p_jk = evaluate_at_xq(
            jnp.tile(ep_T @ p, 3), d0_comp_info, d0_comp_shapes,
            quad_shape, 3)  # (n_q, 3)

        # --- evaluate u at quad points (2-form, 3 components) ---
        eu_T = self.e2_dbc_T if dirichlet_u else self.e2_T
        comp_info_2, comp_shapes_2 = self._form_comp_info(2)
        u_jk = evaluate_at_xq(eu_T @ u, comp_info_2, comp_shapes_2,
                              quad_shape, 3)  # (n_q, 3)

        # --- evaluate div_logical(u) at quad points (scalar) ---
        s2 = list(self.basis_2.shape)
        div_comp_info = [
            (0, grad_r, self.d_basis_t_jk, self.d_basis_z_jk),
            (0, self.d_basis_r_jk, grad_t, self.d_basis_z_jk),
            (0, self.d_basis_r_jk, self.d_basis_t_jk, grad_z),
        ]
        div_comp_shapes = [s2[0], s2[1], s2[2]]
        div_u_jk = evaluate_at_xq(eu_T @ u, div_comp_info, div_comp_shapes,
                                  quad_shape, 1)  # (n_q, 1)

        # --- combine: q = -(grad_p · u) - γ p div_logical(u) ---
        grad_p_dot_u = jnp.sum(grad_p_jk * u_jk, axis=1, keepdims=True)
        q_jk = -(grad_p_dot_u + gamma * p_jk * div_u_jk)  # (n_q, 1)

        # Weight by quadrature weights only (J from M0 cancels 1/J in formula)
        f_jk = q_jk * self.quad.w[:, None]

        # Integrate against 0-form basis
        e0 = self.e0_dbc if dirichlet_p else self.e0
        return e0 @ integrate_against(f_jk, comp_info_0, comp_shapes_0,
                                      quad_shape)

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
