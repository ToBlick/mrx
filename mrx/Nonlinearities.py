
import jax
import jax.numpy as jnp


class CrossProductProjection:
    """
    Given bases Λn, Λm, Λk, constructs an operator to evaluate
    (w, u) -> ∫ (wₕ × uₕ) · Λn[i] dx for all i, where Λn[i] is the i-th basis function of Λn
    and wₕ = ∑ w[i] Λm[i], uₕ = ∑ u[i] Λk[i] are discrete functions
    with coordinate transformation F.
    """

    def __init__(self, n, m, k, Seq):
        """
        Given bases n, m, k, constructs an operator to evaluate
        (w, u) -> ∫ (wₕ × uₕ) · Λn[i] dx for all i, where Λn[i] is the i-th basis function of Λn
        and wₕ = ∑ w[i] Λm[i], uₕ = ∑ u[i] Λk[i] are discrete functions
        with coordinate transformation F.

        Args:
            Λn: Basis for n-forms (n can be 1 or 2)
            Λm: Basis for m-forms (m can be 1 or 2)
            Λk: Basis for k-forms (k can be 1 or 2)
            Seq: DeRham sequence containing the bases and quadrature rule
        """
        self.n = n
        self.m = m
        self.k = k
        self.Seq = Seq

        match self.n:
            case 1:
                self.En = Seq.E1
                self.Λn_ijk = Seq.Λ1_ijk
            case 2:
                self.En = Seq.E2
                self.Λn_ijk = Seq.Λ2_ijk
            case _:
                raise ValueError("n must be 1 or 2")
        match self.m:
            case 1:
                self.Em = Seq.E1
                self.Λm_ijk = Seq.Λ1_ijk
            case 2:
                self.Em = Seq.E2
                self.Λm_ijk = Seq.Λ2_ijk
            case _:
                raise ValueError("m must be 1 or 2")
        match self.k:
            case 1:
                self.Ek = Seq.E1
                self.Λk_ijk = Seq.Λ1_ijk
            case 2:
                self.Ek = Seq.E2
                self.Λk_ijk = Seq.Λ2_ijk
            case _:
                raise ValueError("k must be 1 or 2")

    def __call__(self, w, u):
        """
        evaluates ∫ (wₕ × uₕ) · Λn[i] dx for all i
        and collects the values in a vector.

        Args:
            w (array): m-form dofs
            u (array): k-form dofs

        Returns:
            array: ∫ (wₕ × uₕ) · Λn[i] dx for all i
        """
        return self.En @ self.projection(w, u)

    def projection(self, w, u):

        # w_h evaluated at quadrature points: shape: n_q x 3
        w_h_jk = jnp.einsum("ijk,mi,m->jk", self.Λm_ijk, self.Em, w)
        u_h_jk = jnp.einsum("ijk,mi,m->jk", self.Λk_ijk, self.Ek, u)

        if self.n == 1 and self.m == 2 and self.k == 1:
            # ∫ Λ[i] (Gw x u) / J dx
            Gw_jk = jnp.einsum('jkl,jk->jl', self.Seq.G_jkl, w_h_jk)
            Gw_x_u_jk = jnp.cross(Gw_jk, u_h_jk, axis=1)
            return jnp.einsum("ijk,jk,j->i", self.Λn_ijk, Gw_x_u_jk, self.Seq.Q.w/self.Seq.J_j)
        elif self.n == 1 and self.m == 1 and self.k == 1:
            # ∫ Λ[i] (w x u) dx
            w_x_u_jk = jnp.cross(w_h_jk, u_h_jk, axis=1)
            return jnp.einsum("ijk,jk,j->i", self.Λn_ijk, w_x_u_jk, self.Seq.Q.w)
        elif self.n == 2 and self.m == 1 and self.k == 1:
            # ∫ Λ[i] G(w x u) / J dx
            w_x_u_jk = jnp.cross(w_h_jk, u_h_jk, axis=1)
            G_wxu_jk = jnp.einsum('jkl,jk->jl', self.Seq.G_jkl, w_x_u_jk)
            return jnp.einsum("ijk,jk,j->i", self.Λn_ijk, G_wxu_jk, self.Seq.Q.w/self.Seq.J_j)
        elif self.n == 2 and self.m == 2 and self.k == 1:
            # ∫ Λ[i] (w x G_inv u) dx
            Ginvu_jk = jnp.einsum('jkl,jk->jl', self.Seq.G_inv_jkl, u_h_jk)
            w_x_Ginvu_jk = jnp.cross(w_h_jk, Ginvu_jk, axis=1)
            return jnp.einsum("ijk,jk,j->i", self.Λn_ijk, w_x_Ginvu_jk, self.Seq.Q.w)
        elif self.n == 1 and self.m == 2 and self.k == 2:
            # ∫ Λ[i] G_inv(w x u) dx
            w_x_u_jk = jnp.cross(w_h_jk, u_h_jk, axis=1)
            G_wxu_jk = jnp.einsum('jkl,jk->jl', self.Seq.G_jkl, w_x_u_jk)
            return jnp.einsum("ijk,jk,j->i", self.Λn_ijk, G_wxu_jk, self.Seq.Q.w)
        elif self.n == 2 and self.m == 1 and self.k == 2:
            # ∫ Λ[i] (G_inv w x u) dx
            Ginvw_jk = jnp.einsum('jkl,jk->jl', self.Seq.G_inv_jkl, w_h_jk)
            Ginvw_x_u_jk = jnp.cross(Ginvw_jk, u_h_jk, axis=1)
            return jnp.einsum("ijk,jk,j->i", self.Λn_ijk, Ginvw_x_u_jk, self.Seq.Q.w)
        elif self.n == 2 and self.m == 2 and self.k == 2:
            # ∫ Λ[i] (w x u) / J dx
            w_x_u_jk = jnp.cross(w_h_jk, u_h_jk, axis=1)
            return jnp.einsum("ijk,jk,j->i", self.Λn_ijk, w_x_u_jk, self.Seq.Q.w/self.Seq.J_j)
        else:
            raise ValueError("Not yet implemented")


# class InnerProductProjection:
#     """
#     Given bases Λn, Λm, Λk, constructs an operator to evaluate
#     (w, u) -> ∫ (wₕ · uₕ) · Λn[i] dx for all i, where Λn[i] is the i-th basis function of Λn
#     and wₕ = ∑ w[i] Λm[i], uₕ = ∑ u[i] Λk[i] are discrete functions
#     with coordinate transformation F.
#     """

#     def __init__(self, Λn, Λm, Λk, Q, F=None, En=None, Em=None, Ek=None, Λn_ijk=None, Λm_ijk=None, Λk_ijk=None, J_j=None, G_jkl=None, G_inv_jkl=None):
#         """
#         Given bases Λn, Λm, Λk, constructs an operator to evaluate
#         (w, u) -> ∫ (wₕ × uₕ) · Λn[i] dx for all i, where Λn[i] is the i-th basis function of Λn
#         and wₕ = ∑ w[i] Λm[i], uₕ = ∑ u[i] Λk[i] are discrete functions
#         with coordinate transformation F.

#         Args:
#             Λn: Basis for n-forms (n can be 0 or 3)
#             Λm: Basis for m-forms (m can be 0-3)
#             Λk: Basis for k-forms (k can be 0-3)
#             Q: Quadrature rule for numerical integration
#             F (callable, optional): Coordinate transformation function.
#                                     Defaults to identity mapping.
#             Ek, Ev, En (array, optional): Extraction operator matrix for Λn, Λm, Λk.
#                                 Defaults to identity matrix.
#             Λn_ijk (array, optional): kth component of Λn[i] evaluated at quadrature point j.
#             Λm_ijk (array, optional): kth component of Λm[i] evaluated at quadrature point j.
#             Λk_ijk (array, optional): kth component of Λk[i] evaluated at quadrature point j.
#             J_j (array, optional): Jacobian determinant evaluated at quadrature points.
#             G_jkl (array, optional): (k,l)th element of metric at quadrature point j.
#             G_inv_jkl (array, optional): (k,l)th element of inverse metric at quadrature point j.
#         """
#         self.Λn = Λn
#         self.Λm = Λm
#         self.Λk = Λk
#         self.Q = Q
#         if F is None:
#             self.F = lambda x: x
#         else:
#             self.F = F
#         self.En = En.matrix() if En is not None else None
#         self.Em = Em.matrix() if Em is not None else None
#         self.Ek = Ek.matrix() if Ek is not None else None
#         self.E = self.En

#         # kth component of Λn[i] evaluated at quadrature point j. shape: n x n_q x 3
#         self.Λn_ijk = jax.vmap(jax.vmap(self.Λn, (0, None)), (None, 0))(
#             self.Q.x, self.Λn.ns) if Λn_ijk is None else Λn_ijk
#         # kth component of Λm[i] evaluated at quadrature point j. shape: n x n_q x 3
#         self.Λm_ijk = jax.vmap(jax.vmap(self.Λm, (0, None)), (None, 0))(
#             self.Q.x, self.Λm.ns) if Λm_ijk is None else Λm_ijk
#         # kth component of Λk[i] evaluated at quadrature point j. shape: n x n_q x 3
#         self.Λk_ijk = jax.vmap(jax.vmap(self.Λk, (0, None)), (None, 0))(
#             self.Q.x, self.Λk.ns) if Λk_ijk is None else Λk_ijk

#         # Jacobian determinant evaluated at quadrature points. shape: n_q x 1
#         self.J_j = jax.vmap(jacobian_determinant(self.F))(
#             self.Q.x) if J_j is None else J_j

#         def G(x):
#             return jax.jacfwd(self.F)(x).T @ jax.jacfwd(self.F)(x)

#         # (k,l)th element of metric at quadrature point j. shape: n_q x 3 x 3
#         self.G_jkl = jax.vmap(G)(self.Q.x) if G_jkl is None else G_jkl
#         self.G_inv_jkl = jax.vmap(inv33)(
#             self.G_jkl) if G_inv_jkl is None else G_inv_jkl

#     def __call__(self, w, u):
#         """
#         evaluates ∫ (wₕ × uₕ) · Λn[i] dx for all i
#         and collects the values in a vector.

#         Args:
#             w (array): m-form dofs
#             u (array): k-form dofs

#         Returns:
#             array: ∫ (wₕ × uₕ) · Λn[i] dx for all i
#         """
#         return self.E @ self.projection(w, u)

#     def projection(self, w, u):

#         # w_h evaluated at quadrature points: shape: n_q x 3
#         w_h_jk = jnp.einsum("ijk,mi,m->jk", self.Λm_ijk, self.Em, w)
#         u_h_jk = jnp.einsum("ijk,mi,m->jk", self.Λk_ijk, self.Ek, u)

#         if self.Λn.k == 3 and self.Λm.k == 2 and self.Λk.k == 1:
#             # ∫ Λ[i] (w . u) / J dx
#             return jnp.einsum("ijl,jk,jk,j->i", self.Λn_ijk, w_h_jk, u_h_jk, self.Q.w/self.J_j)
#         elif self.Λn.k == 0 and self.Λm.k == 2 and self.Λk.k == 1:
#             # ∫ Λ[i] (w . u) dx
#             return jnp.einsum("ijl,jk,jk,j->i", self.Λn_ijk, w_h_jk, u_h_jk, self.Q.w)
#         else:
#             raise ValueError("Not yet implemented")
