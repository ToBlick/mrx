
import jax.numpy as jnp

from mrx.utils import evaluate_at_xq, integrate_against


# %%
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
                self.get_Λn_ijk = Seq.get_Λ1_ijk
                self.nn = Seq.Λ1.n
            case 2:
                self.En = Seq.E2
                self.get_Λn_ijk = Seq.get_Λ2_ijk
                self.nn = Seq.Λ2.n
            case _:
                raise ValueError("n must be 1 or 2")
        match self.m:
            case 1:
                self.Em = Seq.E1
                self.get_Λm_ijk = Seq.get_Λ1_ijk
            case 2:
                self.Em = Seq.E2
                self.get_Λm_ijk = Seq.get_Λ2_ijk
            case _:
                raise ValueError("m must be 1 or 2")
        match self.k:
            case 1:
                self.Ek = Seq.E1
                self.get_Λk_ijk = Seq.get_Λ1_ijk
            case 2:
                self.Ek = Seq.E2
                self.get_Λk_ijk = Seq.get_Λ2_ijk
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

        # w and u evaluated at quadrature points: shape: n_q x 3
        w_jk = evaluate_at_xq(
            self.get_Λm_ijk, self.Em.T @ w, self.Seq.Q.n, 3)
        u_jk = evaluate_at_xq(
            self.get_Λk_ijk, self.Ek.T @ u, self.Seq.Q.n, 3)
        # shapes of this: n_q x 3

        # now, we compute
        # ∑ Λn[i](x_j)_a w(x_j)_b u(x_j)_c ) t(x_j)_abc
        # where t is some transformation depending on n,m,k and the metric
        # and we sum over j (quadrature points) and b,c (dimensions)
        # To avoid assembling the huge Λn[i](x_j)_a tensor, we scan over i.

        if self.n == 1 and self.m == 2 and self.k == 1:
            # ∫ Λ[i] (Gw x u) / J dx
            Gw_jk = jnp.einsum('jkl,jk->jl', self.Seq.G_jkl, w_jk)
            Gw_x_u_jk = jnp.cross(Gw_jk, u_jk, axis=1)
            f_jk = Gw_x_u_jk * (self.Seq.Q.w / self.Seq.J_j)[:, None]
        elif self.n == 1 and self.m == 1 and self.k == 1:
            # ∫ Λ[i] (w x u) dx
            w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
            f_jk = w_x_u_jk * (self.Seq.Q.w)[:, None]
        elif self.n == 2 and self.m == 1 and self.k == 1:
            # ∫ Λ[i] G(w x u) / J dx
            w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
            G_wxu_jk = jnp.einsum('jkl,jk->jl', self.Seq.G_jkl, w_x_u_jk)
            f_jk = G_wxu_jk * (self.Seq.Q.w / self.Seq.J_j)[:, None]
        elif self.n == 2 and self.m == 2 and self.k == 1:
            # ∫ Λ[i] (w x G_inv u) dx
            Ginvu_jk = jnp.einsum('jkl,jk->jl', self.Seq.G_inv_jkl, u_jk)
            w_x_Ginvu_jk = jnp.cross(w_jk, Ginvu_jk, axis=1)
            f_jk = w_x_Ginvu_jk * (self.Seq.Q.w)[:, None]
        elif self.n == 1 and self.m == 2 and self.k == 2:
            # ∫ Λ[i] G_inv(w x u) dx
            w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
            Ginv_wxu_jk = jnp.einsum(
                'jkl,jk->jl', self.Seq.G_inv_jkl, w_x_u_jk)
            f_jk = Ginv_wxu_jk * (self.Seq.Q.w)[:, None]
        elif self.n == 2 and self.m == 1 and self.k == 2:
            # ∫ Λ[i] (G_inv w x u) dx
            Ginvw_jk = jnp.einsum('jkl,jk->jl', self.Seq.G_inv_jkl, w_jk)
            Ginvw_x_u_jk = jnp.cross(Ginvw_jk, u_jk, axis=1)
            f_jk = Ginvw_x_u_jk * (self.Seq.Q.w)[:, None]
        elif self.n == 2 and self.m == 2 and self.k == 2:
            # ∫ Λ[i] (w x u) / J dx
            w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
            f_jk = w_x_u_jk * (self.Seq.Q.w / self.Seq.J_j)[:, None]
        else:
            raise ValueError("Not yet implemented")
        return integrate_against(self.get_Λn_ijk, f_jk, self.nn)
