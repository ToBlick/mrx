import jax.numpy as jnp

from mrx.utils import evaluate_at_xq, integrate_against


class CrossProductProjection:
    """
    Given bases Λn, Λm, Λk, constructs an operator to evaluate
    (w, u) -> ∫ (wₕ × uₕ) · Λn[i] dx for all i, where Λn[i] is the i-th basis function of Λn
    and wₕ = ∑ w[i] Λm[i], uₕ = ∑ u[i] Λk[i] are discrete functions
    with coordinate transformation F.
    """

    def __init__(self, n: int, m: int, k: int, seq):  # Seq: DeRhamSequence
        """
        Given bases n, m, k, constructs an operator to evaluate
        (w, u) -> ∫ (wₕ × uₕ) · Λn[i] dx for all i, where Λn[i] is the i-th basis function of Λn
        and wₕ = ∑ w[i] Λm[i], uₕ = ∑ u[i] Λk[i] are discrete functions
        with coordinate transformation F.

        Args:
            n: Degree of the n-form (n can be 1 or 2)
            m: Degree of the m-form (m can be 1 or 2)
            k: Degree of the k-form (k can be 1 or 2)
            seq: DeRham sequence containing the bases and quadrature rule
        """
        self.n = n
        self.m = m
        self.k = k
        self.seq = seq

        match self.n:
            case 1:
                self.en = seq.e1
                self.eval_basis_n_ijk = seq.eval_basis_1_ijk
                self.nn = seq.basis_1.n
            case 2:
                self.en = seq.e2
                self.eval_basis_n_ijk = seq.eval_basis_2_ijk
                self.nn = seq.basis_2.n
            case _:
                raise ValueError("n must be 1 or 2")
        match self.m:
            case 1:
                self.em = seq.e1
                self.eval_basis_m_ijk = seq.eval_basis_1_ijk
            case 2:
                self.em = seq.e2
                self.eval_basis_m_ijk = seq.eval_basis_2_ijk
            case _:
                raise ValueError("m must be 1 or 2")
        match self.k:
            case 1:
                self.Ek = seq.e1
                self.eval_basis_k_ijk = seq.eval_basis_1_ijk
            case 2:
                self.Ek = seq.e2
                self.eval_basis_k_ijk = seq.eval_basis_2_ijk
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
        if self.n == 1:
            en_current = self.seq.e1
        else:
            en_current = self.seq.e2
        
        result = en_current @ self.projection(w, u)
        return result

    def projection(self, w, u):
        """
        Evaluate the projection of the cross product of the m-form and the k-form onto the n-form.
        TODO: Tobi please add a description of these projections.

        Args:
            w (array): m-form dofs
            u (array): k-form dofs

        Returns:
            array: ∫ (wₕ × uₕ) · Λn[i] dx for all i

        """

        # w and u evaluated at quadrature points: shape: n_q x 3
        w_jk = evaluate_at_xq(self.eval_basis_m_ijk,
                              self.em.T @ w, self.seq.quad.n, 3)
        u_jk = evaluate_at_xq(self.eval_basis_k_ijk,
                              self.Ek.T @ u, self.seq.quad.n, 3)

        # now, we compute
        # ∑ Λn[i](x_j)_a w(x_j)_b u(x_j)_c ) t(x_j)_abc
        # where t is some transformation depending on n,m,k and the metric
        # and we sum over j (quadrature points) and b,c (dimensions)
        # To avoid assembling the huge Λn[i](x_j)_a tensor, we scan over i.
        if self.n == 1 and self.m == 2 and self.k == 1:
            # ∫ Λ[i] (Gw x u) / J dx
            Gw_jk = jnp.einsum('jkl,jk->jl', self.seq.metric_jkl, w_jk)
            Gw_x_u_jk = jnp.cross(Gw_jk, u_jk, axis=1)
            f_jk = Gw_x_u_jk * (self.seq.quad.w / self.seq.jacobian_j)[:, None]
        elif self.n == 1 and self.m == 1 and self.k == 1:
            # ∫ Λ[i] (w x u) dx
            w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
            f_jk = w_x_u_jk * (self.seq.quad.w)[:, None]
        elif self.n == 2 and self.m == 1 and self.k == 1:
            # ∫ Λ[i] G(w x u) / J dx
            w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
            G_wxu_jk = jnp.einsum('jkl,jk->jl', self.seq.metric_jkl, w_x_u_jk)
            f_jk = G_wxu_jk * (self.seq.quad.w / self.seq.jacobian_j)[:, None]
        elif self.n == 2 and self.m == 2 and self.k == 1:
            # ∫ Λ[i] (w x G_inv u) dx
            Ginvu_jk = jnp.einsum('jkl,jk->jl', self.seq.metric_inv_jkl, u_jk)
            w_x_Ginvu_jk = jnp.cross(w_jk, Ginvu_jk, axis=1)
            f_jk = w_x_Ginvu_jk * (self.seq.quad.w)[:, None]
        elif self.n == 1 and self.m == 2 and self.k == 2:
            # ∫ Λ[i] G_inv(w x u) dx
            w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
            Ginv_wxu_jk = jnp.einsum(
                'jkl,jk->jl', self.seq.metric_inv_jkl, w_x_u_jk)
            f_jk = Ginv_wxu_jk * (self.seq.quad.w)[:, None]
        elif self.n == 2 and self.m == 1 and self.k == 2:
            # ∫ Λ[i] (G_inv w x u) dx
            Ginvw_jk = jnp.einsum('jkl,jk->jl', self.seq.metric_inv_jkl, w_jk)
            Ginvw_x_u_jk = jnp.cross(Ginvw_jk, u_jk, axis=1)
            f_jk = Ginvw_x_u_jk * (self.seq.quad.w)[:, None]
        elif self.n == 2 and self.m == 2 and self.k == 2:
            # ∫ Λ[i] (w x u) / J dx
            w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
            f_jk = w_x_u_jk * (self.seq.quad.w / self.seq.jacobian_j)[:, None]
        else:
            raise ValueError("Not yet implemented")
        return integrate_against(self.eval_basis_n_ijk, f_jk, self.nn)
