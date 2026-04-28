"""
Projector classes for finite element differential forms.

This module provides classes for projecting functions onto finite element spaces
in the context of differential forms. It supports projections of k-forms
(k = 0, 1, 2, 3) and includes functionality for handling coordinate transformations
and curl projections.

The module implements two main classes:
- Projector: For standard projections of k-forms
- CurlProjection: For projecting curl operations on differential forms
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

import jax
import jax.numpy as jnp
from jax import Array

import mrx
from mrx.utils import integrate_against, inv33

if TYPE_CHECKING:
    from mrx.derham_sequence import DeRhamSequence


# Type aliases for callable functions used in projections
ScalarFunction = Callable[[Array], Array]  # ξ -> scalar (with trailing dim)
VectorFunction = Callable[[Array], Array]  # ξ -> 3D vector


def _as_single_component(values: Array) -> Array:
    """Normalize a scalar or length-1 array to shape (1,)."""
    return jnp.reshape(jnp.asarray(values), (1,))


class Projector:
    """
    A class for projecting functions onto finite element spaces.

    Functions are represented as functions of the logical coordinate ξ in the 
    physical (x,y,z) frame, for example:
    v(ξ) = v_x(ξ) e_x + v_y(ξ) e_y + v_z(ξ) e_z

    This class implements projection operators for differential forms of various
    degrees (k = 0, 1, 2, 3). It supports coordinate transformations through
    the mapping F and can handle extraction operators through E.

    Attributes:
        k (int): Degree of the differential form (0, 1, 2, or 3)
        seq : DeRham sequence object
        dirichlet (bool): Whether to use dirichlet boundary conditions
    """

    k: Literal[0, 1, 2, 3]
    seq: DeRhamSequence
    dirichlet: bool = True
    bc: bool = False

    def __init__(self, seq: DeRhamSequence, k: Literal[0, 1, 2, 3], dirichlet: bool = True, bc: bool = False) -> None:
        """
        Initialize the projector.

        Args:
            seq : DeRham sequence object
            k : Degree of the differential form
            dirichlet : Whether to use dirichlet boundary conditions
            bc : If True, project onto the Dirichlet boundary DOFs only
                 (uses e_k_bc instead of e_k or e_k_dbc).
                 Takes precedence over `dirichlet`.
        """
        self.k = k
        self.seq = seq
        self.dirichlet = dirichlet
        self.bc = bc

    def __call__(self, f: ScalarFunction | VectorFunction) -> Array:
        """
        Project a function onto the finite element space.

        Args:
            f (callable): Function to project

        Returns:
            array: Projection coefficients
        """

        if self.k == 0:
            if self.bc:
                e = self.seq.e0_bc
            elif self.dirichlet:
                e = self.seq.e0_dbc
            else:
                e = self.seq.e0
            return e @ self.zeroform_projection(f)
        elif self.k == 1:
            if self.bc:
                e = self.seq.e1_bc
            elif self.dirichlet:
                e = self.seq.e1_dbc
            else:
                e = self.seq.e1
            return e @ self.oneform_projection(f)
        elif self.k == 2:
            if self.bc:
                e = self.seq.e2_bc
            elif self.dirichlet:
                e = self.seq.e2_dbc
            else:
                e = self.seq.e2
            return e @ self.twoform_projection(f)
        elif self.k == 3:
            if self.bc:
                e = self.seq.e3_bc
            elif self.dirichlet:
                e = self.seq.e3_dbc
            else:
                e = self.seq.e3
            return e @ self.threeform_projection(f)
        # TODO: Consider raising an error for invalid k values
        raise ValueError(f"Invalid k value: {self.k}. Must be 0, 1, 2, or 3.")

    def zeroform_projection(self, f: ScalarFunction) -> Array:
        """
        Project a scalar function (0-form).

        Args:
            f (callable): Scalar function to project

        Returns:
            array: Projection coefficients for the 0-form
        """
        # Evaluate the given function at quadrature points
        f_jk: Array = jax.lax.map(
            lambda x: _as_single_component(f(x)),
            self.seq.quad.x,
            batch_size=mrx.MAP_BATCH_SIZE_INNER,
        )
        w_jk: Array = f_jk * (self.seq.quad.w * self.seq.jacobian_j)[:, None]
        comp_info, comp_shapes = self.seq._form_comp_info(0)
        quad_shape = (self.seq.quad.ny, self.seq.quad.nx, self.seq.quad.nz)
        return integrate_against(w_jk, comp_info, comp_shapes, quad_shape)

    def oneform_projection(self, v: VectorFunction) -> Array:
        """
        Project a vector-valued function to a 1-form.

        Args:
            A (callable): Vector field to project

        Returns:
            array: Projection coefficients for the 1-form
        """
        DF = jax.jacfwd(self.seq.map)

        def _v(x: Array) -> Array:
            return inv33(DF(x)) @ v(x)

        # Evaluate the given function at quadrature points
        A_jk: Array = jax.lax.map(
            _v, self.seq.quad.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)  # n_q x d
        w_jk: Array = A_jk * (self.seq.quad.w * self.seq.jacobian_j)[:, None]

        comp_info, comp_shapes = self.seq._form_comp_info(1)
        quad_shape = (self.seq.quad.ny, self.seq.quad.nx, self.seq.quad.nz)
        return integrate_against(w_jk, comp_info, comp_shapes, quad_shape)

    def twoform_projection(self, v: VectorFunction) -> Array:
        """
        Project to a 2-form.

        Args:
            v (callable): vector field to project - in physical coordinates

        Returns:
            array: Projection coefficients for the 2-form
        """
        DF = jax.jacfwd(self.seq.map)

        def _v(x: Array) -> Array:
            return DF(x).T @ v(x)

        # Evaluate the given function at quadrature points
        B_jk: Array = jax.lax.map(
            _v, self.seq.quad.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)  # n_q x d

        w_jk: Array = B_jk * (self.seq.quad.w)[:, None]

        comp_info, comp_shapes = self.seq._form_comp_info(2)
        quad_shape = (self.seq.quad.ny, self.seq.quad.nx, self.seq.quad.nz)
        return integrate_against(w_jk, comp_info, comp_shapes, quad_shape)

    def threeform_projection(self, f: ScalarFunction) -> Array:
        """
        Project a volume form (3-form).

        Args:
            f (callable): function

        Returns:
            array: Projection coefficients for the 3-form
        """
        # Evaluate the given function at quadrature points
        f_jk: Array = jax.lax.map(
            lambda x: _as_single_component(f(x)),
            self.seq.quad.x,
            batch_size=mrx.MAP_BATCH_SIZE_INNER,
        )
        w_jk: Array = f_jk * (self.seq.quad.w)[:, None]
        comp_info, comp_shapes = self.seq._form_comp_info(3)
        quad_shape = (self.seq.quad.ny, self.seq.quad.nx, self.seq.quad.nz)
        return integrate_against(w_jk, comp_info, comp_shapes, quad_shape)

# TODO: requires testing still
def surface_integral(f: ScalarFunction, seq: "DeRhamSequence") -> Array:
    """Integrate a scalar function over the outer boundary r = 1.

    The surface element is  dS = ‖∂_θ F × ∂_ζ F‖ dθ dζ  evaluated at r = 1.
    Quadrature in (θ, ζ) is reused from ``seq.quad``.

    Parameters
    ----------
    f : callable  ξ → array of shape (1,)
        Function of logical coordinates, called at ξ = (1, θ_q, ζ_q).
    seq : DeRhamSequence

    Returns
    -------
    scalar Array
    """
    nt, nz = seq.quad.ny, seq.quad.nz
    X_t, X_z = jnp.meshgrid(seq.quad.x_y, seq.quad.x_z, indexing='ij')
    xi_bdy = jnp.stack(
        [jnp.ones(nt * nz), X_t.ravel(), X_z.ravel()], axis=-1
    )  # (nt*nz, 3)

    DF = jax.jacfwd(seq.map)

    def _integrand(xi: Array) -> Array:
        dF = DF(xi)
        surf_jac = jnp.linalg.norm(jnp.cross(dF[:, 1], dF[:, 2]))
        return jnp.squeeze(f(xi)) * surf_jac

    vals = jax.lax.map(
        _integrand, xi_bdy, batch_size=mrx.MAP_BATCH_SIZE_INNER
    )  # (nt*nz,)
    w_bdy = jnp.outer(seq.quad.w_y, seq.quad.w_z).ravel()
    return jnp.dot(vals, w_bdy)

# TODO: requires testing still
class BoundaryProjector:
    """Project a k-form onto the Dirichlet boundary DOFs via a surface integral.

    Computes the boundary load vector

        b_i = ∫_{r=1} g(ξ) · trace(φ_i)(ξ) dS,

    then selects the BC DOF values via the ``e_k_bc`` extraction operator.

    ``g`` follows the same convention as :class:`Projector`: for k = 0 and 3,
    a scalar function ξ → (1,); for k = 1 and 2, a vector function
    ξ → (3,) in the physical (x, y, z) frame.

    All quadrature-dependent quantities (surface Jacobian, boundary quad
    points, r-spline values at r = 1) are computed once in ``__init__`` and
    reused across calls.
    """

    def __init__(self, seq: "DeRhamSequence", k: Literal[0, 1, 2, 3]) -> None:
        self.seq = seq
        self.k = k

        # r-spline values at r = 1, shapes (n_r,) and (n_dr,)
        lam_r  = seq.basis_0.Λ[0]
        dlam_r = seq.basis_0.dΛ[0]
        self._basis_r_1   = jax.vmap(lam_r,  (None, 0))(1.0, lam_r.ns)
        self._d_basis_r_1 = jax.vmap(dlam_r, (None, 0))(1.0, dlam_r.ns)

        # 2D boundary quadrature grid (θ, ζ) at r = 1
        nt, nz = seq.quad.ny, seq.quad.nz
        X_t, X_z = jnp.meshgrid(seq.quad.x_y, seq.quad.x_z, indexing='ij')
        xi_bdy = jnp.stack(
            [jnp.ones(nt * nz), X_t.ravel(), X_z.ravel()], axis=-1
        )  # (nt*nz, 3)

        # DF at all boundary quad points, shape (nt, nz, 3, 3)
        # DF[t,z,i,j] = ∂F_i/∂ξ_j
        DF = jax.jacfwd(seq.map)
        DF_bdy = jax.lax.map(
            DF, xi_bdy, batch_size=mrx.MAP_BATCH_SIZE_INNER
        ).reshape(nt, nz, 3, 3)  # (nt, nz, 3, 3)

        # Surface Jacobian ‖∂_θ F × ∂_ζ F‖ and the unnormalized surface normal
        surf_normal = jnp.cross(DF_bdy[:, :, :, 1], DF_bdy[:, :, :, 2])  # (nt, nz, 3)
        surf_jac = jnp.linalg.norm(surf_normal, axis=-1)                  # (nt, nz)

        w_bdy = jnp.outer(seq.quad.w_y, seq.quad.w_z)               # (nt, nz)
        J_bdy = jnp.linalg.det(DF_bdy)                                  # (nt, nz)
        self._xi_bdy       = xi_bdy                                  # (nt*nz, 3)
        self._DF_bdy       = DF_bdy                                  # (nt, nz, 3, 3)
        self._DF_inv_bdy   = jax.vmap(inv33)(DF_bdy.reshape(-1, 3, 3)).reshape(nt, nz, 3, 3)
        self._J_bdy        = J_bdy                                   # (nt, nz)
        self._w_surf       = w_bdy * surf_jac                        # (nt, nz)
        self._nt = nt
        self._nz = nz

    def __call__(self, g: ScalarFunction | VectorFunction | Array) -> Array:
        """Compute the boundary load vector for prescribed boundary data g.

        Parameters
        ----------
        g : callable or array
            If callable: ξ → (1,) for k = 0 or 3; ξ → (3,) in physical frame
            for k = 1 or 2.  Evaluated at the boundary quad points.

            If array of shape (ny*nx*nz, d): precomputed values at the full 3D
            quad grid (e.g. from ``oneform_projection``).  The θ,ζ quad points
            are the same as for the boundary; the r-dimension is irrelevant for
            boundary data, so slice ``[:, 0, :, :]`` is used.

        Returns
        -------
        Array of shape (n_k_bc,)
        """
        seq = self.seq
        nt, nz = self._nt, self._nz

        if callable(g):
            g_jk = jax.lax.map(
                g, self._xi_bdy, batch_size=mrx.MAP_BATCH_SIZE_INNER
            ).reshape(nt, nz, -1)  # (nt, nz, d)
        else:
            # Precomputed 3D values; any r-slice gives the same (θ,ζ) grid
            nx = seq.quad.nx
            g_jk = jnp.asarray(g).reshape(nt, nx, nz, -1)[:, 0, :, :]  # (nt, nz, d)

        if self.k == 0:
            return self._project_0form(g_jk)
        elif self.k == 1:
            return self._project_1form(g_jk)
        elif self.k == 2:
            return self._project_2form(g_jk)
        else:
            raise NotImplementedError("BoundaryProjector: k = 3 not implemented")

    def _project_0form(self, g_jk: Array) -> Array:
        seq = self.seq
        wg = g_jk[:, :, 0] * self._w_surf                          # (nt, nz)
        part = jnp.einsum('jk,bj,ck->bc',
                          wg, seq.basis_t_jk, seq.basis_z_jk)      # (n_t, n_z)
        b_full = jnp.einsum('a,bc->abc',
                            self._basis_r_1, part).ravel()
        return seq.e0_bc @ b_full

    def _project_1form(self, g_jk: Array) -> Array:
        """Transform physical → logical covariant (DF^{-1}) then integrate."""
        seq = self.seq
        nt, nz = self._nt, self._nz

        g_log = jnp.einsum('tzij,tzj->tzi', self._DF_inv_bdy, g_jk)  # (nt, nz, 3)

        # r-component: dΛ_r^a(1) ⊗ Λ_t^b ⊗ Λ_z^c
        wg0 = g_log[:, :, 0] * self._w_surf
        part_r = jnp.einsum('jk,bj,ck->bc', wg0,
                            seq.basis_t_jk, seq.basis_z_jk)
        b_r = jnp.einsum('a,bc->abc', self._d_basis_r_1, part_r).ravel()

        # θ-component: Λ_r^a(1) ⊗ dΛ_t^b ⊗ Λ_z^c
        wg1 = g_log[:, :, 1] * self._w_surf
        part_t = jnp.einsum('jk,bj,ck->bc', wg1,
                            seq.d_basis_t_jk, seq.basis_z_jk)
        b_t = jnp.einsum('a,bc->abc', self._basis_r_1, part_t).ravel()

        # ζ-component: Λ_r^a(1) ⊗ Λ_t^b ⊗ dΛ_z^c
        wg2 = g_log[:, :, 2] * self._w_surf
        part_z = jnp.einsum('jk,bj,ck->bc', wg2,
                            seq.basis_t_jk, seq.d_basis_z_jk)
        b_z = jnp.einsum('a,bc->abc', self._basis_r_1, part_z).ravel()

        return seq.e1_bc @ jnp.concatenate([b_r, b_t, b_z])

    def _project_2form(self, g_jk: Array) -> Array:
        """Pull back g to logical covariant 2-form (DF^T g / J) and integrate
        against each reference basis group weighted by surf_jac."""
        seq = self.seq

        # Pullback: (DF^T g / J)[tz, j] = Σ_i DF[tz,i,j] g[tz,i] / J[tz]
        g_log = jnp.einsum('tzij,tzi->tzj', self._DF_bdy, g_jk) / self._J_bdy[:, :, None]  # (nt, nz, 3)

        # r-component: Λ_r^a(1) ⊗ dΛ_θ^b ⊗ dΛ_ζ^c
        wg0 = g_log[:, :, 0] * self._w_surf
        part_r = jnp.einsum('tz,bz,ct->bc', wg0,
                            seq.d_basis_z_jk, seq.d_basis_t_jk)
        b_r = jnp.einsum('a,bc->abc', self._basis_r_1, part_r).ravel()

        # θ-component: dΛ_r^a(1) ⊗ Λ_θ^b ⊗ dΛ_ζ^c
        wg1 = g_log[:, :, 1] * self._w_surf
        part_t = jnp.einsum('tz,bz,ct->bc', wg1,
                            seq.d_basis_z_jk, seq.basis_t_jk)
        b_t = jnp.einsum('a,bc->abc', self._d_basis_r_1, part_t).ravel()

        # ζ-component: dΛ_r^a(1) ⊗ dΛ_θ^b ⊗ Λ_ζ^c
        wg2 = g_log[:, :, 2] * self._w_surf
        part_z = jnp.einsum('tz,bz,ct->bc', wg2,
                            seq.basis_z_jk, seq.d_basis_t_jk)
        b_z = jnp.einsum('a,bc->abc', self._d_basis_r_1, part_z).ravel()

        return seq.e2_bc @ jnp.concatenate([b_r, b_t, b_z])

    def evaluate_trace(self, u: Array) -> Array:
        """Evaluate the trace of a discrete k-form at the boundary quad points.

        Given the full (unreduced) DOF vector ``u`` of shape ``(n_k,)``,
        reconstruct the field values at the ``(nt, nz)`` boundary quad points.

        No coordinate map evaluation is needed:

        * k = 0: scalar ``f(1, θ, ζ)``, shape ``(nt, nz)``.
        * k = 1: logical components ``E_log = (E_r, E_θ, E_ζ)`` at r = 1,
          shape ``(nt, nz, 3)``.  The physical tangential vector is
          ``DF^{-T} E_log`` using the precomputed ``self._DF_inv_bdy``.
        * k = 2: normal flux ``B_log_r = B_phys · (∂_θF × ∂_ζF)`` at r = 1,
          shape ``(nt, nz)``.  The Jacobian J cancels exactly because
          ``B_phys = (1/J) DF B_log``, so no DF evaluation is needed.

        Parameters
        ----------
        u : Array, shape ``(n_k,)``
            Full DOF vector in the unreduced space (i.e. *not* BC-extracted).

        Returns
        -------
        Array of shape ``(nt, nz)`` for k = 0 or 2, ``(nt, nz, 3)`` for k = 1.
        """
        if self.k == 0:
            return self._eval_trace_0form(u)
        elif self.k == 1:
            return self._eval_trace_1form(u)
        elif self.k == 2:
            return self._eval_trace_2form(u)
        else:
            raise NotImplementedError("evaluate_trace: k = 3 not implemented")

    def _eval_trace_0form(self, u: Array) -> Array:
        seq = self.seq
        n_r = self._basis_r_1.shape[0]
        n_t = seq.basis_t_jk.shape[0]
        n_z = seq.basis_z_jk.shape[0]
        u_3d = u.reshape(n_r, n_t, n_z)
        # f(1, θ_q, ζ_q) = Σ_{a,b,c} u[a,b,c] Λ_r^a(1) Λ_t^b(θ_q) Λ_z^c(ζ_q)
        return jnp.einsum('abc,a,bt,cz->tz', u_3d,
                          self._basis_r_1, seq.basis_t_jk, seq.basis_z_jk)

    def _eval_trace_1form(self, u: Array) -> Array:
        """Return logical components E_log at r = 1, shape (nt, nz, 3).

        No DF is applied here; physical E_phys = DF^{-T} E_log via
        ``einsum('tzji,tzj->tzi', self._DF_inv_bdy, E_log)`` if needed.
        """
        seq = self.seq
        n_dr = self._d_basis_r_1.shape[0]
        n_r  = self._basis_r_1.shape[0]
        n_t  = seq.basis_t_jk.shape[0]
        n_dt = seq.d_basis_t_jk.shape[0]
        n_z  = seq.basis_z_jk.shape[0]
        n_dz = seq.d_basis_z_jk.shape[0]
        n1_r = n_dr * n_t * n_z
        n1_t = n_r  * n_dt * n_z
        u_r = u[:n1_r].reshape(n_dr, n_t, n_z)
        u_t = u[n1_r:n1_r + n1_t].reshape(n_r, n_dt, n_z)
        u_z = u[n1_r + n1_t:].reshape(n_r, n_t, n_dz)
        # E_log_r = Σ u_r[a,b,c] dΛ_r^a(1) Λ_t^b Λ_z^c
        E_r = jnp.einsum('abc,a,bt,cz->tz', u_r,
                         self._d_basis_r_1, seq.basis_t_jk, seq.basis_z_jk)
        # E_log_t = Σ u_t[a,b,c] Λ_r^a(1) dΛ_t^b Λ_z^c
        E_t = jnp.einsum('abc,a,bt,cz->tz', u_t,
                         self._basis_r_1, seq.d_basis_t_jk, seq.basis_z_jk)
        # E_log_z = Σ u_z[a,b,c] Λ_r^a(1) Λ_t^b dΛ_z^c
        E_z = jnp.einsum('abc,a,bt,cz->tz', u_z,
                         self._basis_r_1, seq.basis_t_jk, seq.d_basis_z_jk)
        return jnp.stack([E_r, E_t, E_z], axis=-1)  # (nt, nz, 3)

    def _eval_trace_2form(self, u: Array) -> Array:
        """Return B_phys · surf_normal = B_log_r at r = 1, shape (nt, nz).

        J cancels: B_phys · surf_normal = (1/J)(DF B_log) · surf_normal
                                        = (1/J) J B_log_r = B_log_r.

        This is the *unscaled* normal flux (integrated against the surface
        element).  To get the pointwise normal component B_phys · n̂ divide
        by the surface Jacobian ‖∂_θF × ∂_ζF‖, accessible as
        ``bp.surf_jac()``.
        """
        seq = self.seq
        n_r  = self._basis_r_1.shape[0]
        n_dt = seq.d_basis_t_jk.shape[0]
        n_dz = seq.d_basis_z_jk.shape[0]
        n2_r = n_r * n_dt * n_dz
        u_r = u[:n2_r].reshape(n_r, n_dt, n_dz)
        # B_log_r = Σ u_r[a,b,c] Λ_r^a(1) dΛ_t^b dΛ_z^c
        return jnp.einsum('abc,a,bt,cz->tz', u_r,
                          self._basis_r_1, seq.d_basis_t_jk, seq.d_basis_z_jk)
