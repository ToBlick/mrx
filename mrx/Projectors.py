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

import jax
import jax.numpy as jnp

from mrx.DifferentialForms import DiscreteFunction
from mrx.Utils import div, grad, inv33, jacobian_determinant

__all__ = ['Projector', 'CrossProductProjection']


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
        Λ: The domain operator defining the finite element space
        Q: Quadrature rule for numerical integration
        n (int): Total size of the operator
        ns (array): Array of indices for the finite element space
        F (callable): Coordinate transformation function, defaults to identity
        M (array): Extraction operator matrix, defaults to identity
    """

    def __init__(self, Λ, Q, F=None, E=None):
        """
        Initialize the projector.

        Args:
            Λ: Domain operator defining the finite element space
            Q: Quadrature rule for numerical integration
            F (callable, optional): Coordinate transformation function.
                                 Defaults to identity mapping.
            E (array, optional): Extraction operator matrix.
                              Defaults to identity matrix.
        """
        self.Λ = Λ
        self.Q = Q
        self.n = Λ.n
        self.ns = Λ.ns
        if F is None:
            self.F = lambda x: x
        else:
            self.F = F
        if E is None:
            self.E = jnp.eye(self.n)
        else:
            self.E = E.matrix()

    def __call__(self, f):
        """
        Project a function onto the finite element space.

        Args:
            f (callable): Function to project

        Returns:
            array: Projection coefficients
        """
        if self.Λ.k == 0:
            return self.E @ self.zeroform_projection(f)
        elif self.Λ.k == 1:
            return self.E @ self.oneform_projection(f)
        elif self.Λ.k == 2:
            return self.E @ self.twoform_projection(f)
        elif self.Λ.k == 3:
            return self.E @ self.threeform_projection(f)
        elif self.Λ.k == -1:
            return self.E @ self.vectorfield_projection(f)

    def zeroform_projection(self, f):
        """
        Project a scalar function (0-form).

        Args:
            f (callable): Scalar function to project

        Returns:
            array: Projection coefficients for the 0-form
        """
        # Evaluate all basis functions at quadrature points
        Λijk = jax.vmap(jax.vmap(self.Λ, (0, None)), (None, 0))(
            self.Q.x, self.ns)  # n x n_q x 1
        # Evaluate the given function at quadrature points
        fjk = jax.vmap(f)(self.Q.x)  # n_q x 1
        # Evaluate the jacobian of F at quadrature points
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,jk,j,j->i", Λijk, fjk, Jj, wj)

    def oneform_projection(self, v):
        """
        Project a vector-valued function to a 1-form.

        Args:
            A (callable): Vector field to project

        Returns:
            array: Projection coefficients for the 1-form
        """
        DF = jax.jacfwd(self.F)

        def _v(x):
            return inv33(DF(x)) @ v(x)

        Ajk = jax.vmap(_v)(self.Q.x)  # n_q x d
        Λijk = jax.vmap(jax.vmap(self.Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x d
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)
        wj = self.Q.w
        return jnp.einsum("ijk,jk,j,j->i", Λijk, Ajk, Jj, wj)

    def twoform_projection(self, v):
        """
        Project to a 2-form.

        Args:
            v (callable): vector field to project - in physical coordinates

        Returns:
            array: Projection coefficients for the 2-form
        """
        DF = jax.jacfwd(self.F)

        def _v(x):
            return DF(x).T @ v(x)

        def _Λ(x, i):
            return self.Λ(x, i)
        Bjk = jax.vmap(_v)(self.Q.x)  # n_q x d
        Λijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x d
        wj = self.Q.w
        return jnp.einsum("ijk,jk,j->i", Λijk, Bjk, wj)

    def threeform_projection(self, f):
        """
        Project a volume form (3-form).

        Args:
            f (callable): function

        Returns:
            array: Projection coefficients for the 3-form
        """
        # Evaluate all basis functions at quadrature points
        Λijk = jax.vmap(jax.vmap(self.Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x 1
        fjk = jax.vmap(f)(self.Q.x)  # n_q x 1
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,jk,j->i", Λijk, fjk, wj)

    def vectorfield_projection(self, v):
        """
        Project to a vector field.

        Args:
            v (callable): vector field to project - in physical coordinates

        Returns:
            array: Projection coefficients for the vector field
        """
        DF = jax.jacfwd(self.F)

        def _v(x):
            return DF(x).T @ v(x)

        def _Λ(x, i):
            return self.Λ(x, i)
        Bjk = jax.vmap(_v)(self.Q.x)  # n_q x d
        Λijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x d
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w
        return jnp.einsum("ijk,jk,j,j->i", Λijk, Bjk, Jj, wj)


class CrossProductProjection:
    """
    Given bases Λn, Λm, Λk, constructs an operator to evaluate
    (w, u) -> ∫ (wₕ × uₕ) · Λn[i] dx for all i, where Λn[i] is the i-th basis function of Λn
    and wₕ = ∑ w[i] Λm[i], uₕ = ∑ u[i] Λk[i] are discrete functions
    with coordinate transformation F.
    """

    def __init__(self, Λn, Λm, Λk, Q, F=None, En=None, Em=None, Ek=None):
        """
        Given bases Λn, Λm, Λk, constructs an operator to evaluate
        (w, u) -> ∫ (wₕ × uₕ) · Λn[i] dx for all i, where Λn[i] is the i-th basis function of Λn
        and wₕ = ∑ w[i] Λm[i], uₕ = ∑ u[i] Λk[i] are discrete functions
        with coordinate transformation F.

        Args:
            Λn: Basis for n-forms (n can be 1 or 2)
            Λm: Basis for m-forms (m can be 1 or 2)
            Λk: Basis for k-forms (k can be 1 or 2)
            Q: Quadrature rule for numerical integration
            F (callable, optional): Coordinate transformation function.
                                    Defaults to identity mapping.
            Ek, Ev, En (array, optional): Extraction operator matrix for Λn, Λm, Λk.
                                Defaults to identity matrix.
        """
        self.Λn = Λn
        self.Λm = Λm
        self.Λk = Λk
        self.Q = Q
        if F is None:
            self.F = lambda x: x
        else:
            self.F = F
        self.En = En.matrix() if En is not None else None
        self.Em = Em.matrix() if Em is not None else None
        self.Ek = Ek.matrix() if Ek is not None else None
        self.E = self.En

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
        return self.E @ self.projection(w, u)

    def projection(self, w, u):

        DF = jax.jacfwd(self.F)
        w_h = DiscreteFunction(w, self.Λm, self.Em)
        u_h = DiscreteFunction(u, self.Λk, self.Ek)

        if self.Λn.k == 1 and self.Λm.k == 2 and self.Λk.k == 1:
            def v(x):
                G = DF(x).T @ DF(x) / jnp.linalg.det(DF(x))
                return jnp.cross(G @ w_h(x), u_h(x))
        elif self.Λn.k == 1 and self.Λm.k == 1 and self.Λk.k == 1:
            def v(x):
                return jnp.cross(w_h(x), u_h(x))
        elif self.Λn.k == 2 and self.Λm.k == 1 and self.Λk.k == 1:
            def v(x):
                G = DF(x).T @ DF(x) / jnp.linalg.det(DF(x))
                return G @ jnp.cross(w_h(x), u_h(x))
        elif self.Λn.k == 2 and self.Λm.k == 2 and self.Λk.k == 1:
            def v(x):
                G = DF(x).T @ DF(x) / jnp.linalg.det(DF(x))
                return G @ jnp.cross(G @ w_h(x), u_h(x))
        elif self.Λn.k == 1 and self.Λm.k == 2 and self.Λk.k == 2:
            def v(x):
                G = DF(x).T @ DF(x)
                return inv33(G) @ jnp.cross(w_h(x), u_h(x))
        elif self.Λn.k == 2 and self.Λm.k == 1 and self.Λk.k == 2:
            def v(x):
                G = DF(x).T @ DF(x) / jnp.linalg.det(DF(x))
                return G @ jnp.cross(w_h(x), G @ u_h(x))
        elif self.Λn.k == 2 and self.Λm.k == 2 and self.Λk.k == 2:
            def v(x):
                G = DF(x).T @ DF(x)
                return jnp.cross(w_h(x), u_h(x)) / jnp.linalg.det(DF(x))
        else:
            raise ValueError("Not yet implemented")

        # Compute projections
        vjk = jax.vmap(v)(self.Q.x)  # n_q x d
        Λijk = jax.vmap(jax.vmap(self.Λn, (0, None)), (None, 0))(
            self.Q.x, self.Λn.ns)  # n x n_q x d
        wj = self.Q.w
        return jnp.einsum("ijk,jk,j->i", Λijk, vjk, wj)

# class CrossProductProjection:
#     """
#     Given bases Λn, Λv, Λk, constructs an operator to evaluate
#     1) (strong):
#         (w, u) -> ∫ (wₕ × uₕ) · Λn[i] dx for all i, where Λn[i] is the i-th basis function of Λn
#         and wₕ = ∑ w[i] Λk[i], uₕ = ∑ u[i] Λv[i] are discrete functions
#     2) (weak):
#         (w, u) -> - ∫ (wₕ × uₕ) · Λv[i] dx for all i, where Λv[i] is the i-th basis function of Λv
#         and wₕ = ∑ w[i] Λk[i], uₕ = ∑ u[i] Λn[i] are discrete functions
#     with coordinate transformation F.
#     """

#     def __init__(self, Λn, Λv, Λk, Q, F=None, En=None, Ev=None, Ek=None, mode='strong'):
#         """
#         Given bases Λn, Λv, Λk, constructs an operator to evaluate
#         1) (strong):
#             (w, u) -> ∫ (wₕ × uₕ) · Λn[i] dx for all i, where Λn[i] is the i-th basis function of Λn
#             and wₕ = ∑ w[i] Λk[i], uₕ = ∑ u[i] Λv[i] are discrete functions
#         2) (weak):
#             (w, u) -> - ∫ (wₕ × uₕ) · Λv[i] dx for all i, where Λv[i] is the i-th basis function of Λv
#             and wₕ = ∑ w[i] Λk[i], uₕ = ∑ u[i] Λn[i] are discrete functions
#         with coordinate transformation F.

#         Args:
#             Λn: Basis for n-forms (n can be 1 or 2)
#             Λk: Basis for k-forms (k can be 1 or 2)
#             Λv: Basis for 2-forms
#             Q: Quadrature rule for numerical integration
#             F (callable, optional): Coordinate transformation function.
#                                     Defaults to identity mapping.
#             Ek, Ev, En (array, optional): Extraction operator matrix for Λk, Λv, Λ1.
#                                 Defaults to identity matrix.
#         """
#         self.Λn = Λn
#         self.Λk = Λk
#         self.Λv = Λv
#         self.Q = Q
#         self.mode = mode  # 'strong' or 'weak'
#         if mode not in ['strong', 'weak']:
#             raise ValueError("mode must be 'strong' or 'weak'")
#         if F is None:
#             self.F = lambda x: x
#         else:
#             self.F = F
#         self.En = En.matrix() if En is not None else None
#         self.Ev = Ev.matrix() if Ev is not None else None
#         self.Ek = Ek.matrix() if Ek is not None else None
#         if mode == 'strong':
#             self.n = Λn.n
#             self.ns = Λn.ns
#             self.E = jnp.eye(Λn.n) if En is None else En
#         elif mode == 'weak':
#             self.n = Λv.n
#             self.ns = Λv.ns
#             self.E = jnp.eye(Λv.n) if Ev is None else Ev
#         if self.Λv.k != 2:
#             raise ValueError("Λv must be a 2-form")
#         if self.Λk.k not in [1, 2]:
#             raise ValueError("Λk must be a 1-form or 2-form")
#         if self.Λn.k not in [1, 2]:
#             raise ValueError("Λn must be a 1-form or 2-form")

#     def __call__(self, w, u):
#         """
#         evaluates
#         1) (strong) ∫ (wₕ × uₕ) · Λn[i] dx for all i
#         2) (weak) - ∫ (wₕ × uₕ) · Λv[i] dx for all i
#         and collects the values in a vector.

#         Args:
#             w (array): k-form dofs
#             u (array): vector field dofs

#         Returns:
#             array: ∫ (wₕ × uₕ) · Λn[i] dx for all i
#         """
#         return self.E @ self.projection(w, u)

#     def projection(self, w, u):

#         DF = jax.jacfwd(self.F)

#         w_h = DiscreteFunction(w, self.Λk, self.Ek)
#         if self.mode == 'strong':
#             u_h = DiscreteFunction(u, self.Λv, self.Ev)
#         elif self.mode == 'weak':
#             u_h = DiscreteFunction(u, self.Λn, self.En)

#         if self.mode == 'strong':
#             if self.Λk.k == 1 and self.Λn.k == 1:
#                 def v(x):
#                     return jnp.cross(w_h(x), DF(x).T @ DF(x) @ u_h(x)) / jnp.linalg.det(DF(x))
#             elif self.Λk.k == 2 and self.Λn.k == 1:
#                 def v(x):
#                     return inv33(DF(x).T @ DF(x)) @ jnp.cross(w_h(x), u_h(x))
#             elif self.Λk.k == 1 and self.Λn.k == 2:
#                 def v(x):
#                     return jnp.cross(inv33(DF(x).T @ DF(x)) @ w_h(x), u_h(x))
#             elif self.Λk.k == 2 and self.Λn.k == 2:
#                 def v(x):
#                     return jnp.cross(w_h(x), u_h(x)) / jnp.linalg.det(DF(x))
#             else:
#                 raise ValueError("Λk and Λn must be a 1-form or 2-form")

#             def Λ(x, i):
#                 return self.Λn(x, i)

#         elif self.mode == 'weak':
#             if self.Λk.k == 1 and self.Λn.k == 1:
#                 def v(x):
#                     return - DF(x).T @ DF(x) @ jnp.cross(w_h(x), u_h(x)) / jnp.linalg.det(DF(x))
#             elif self.Λk.k == 2 and self.Λn.k == 1:
#                 def v(x):
#                     return - jnp.cross(w_h(x), inv33(DF(x).T @ DF(x)) @ u_h(x))
#             elif self.Λk.k == 1 and self.Λn.k == 2:
#                 def v(x):
#                     return - jnp.cross(inv33(DF(x).T @ DF(x)) @ w_h(x), u_h(x))
#             elif self.Λk.k == 2 and self.Λn.k == 2:
#                 def v(x):
#                     return - jnp.cross(w_h(x), u_h(x)) / jnp.linalg.det(DF(x))
#             else:
#                 raise ValueError("Λk and Λn must be a 1-form or 2-form")

#             def Λ(x, i):
#                 return self.Λv(x, i)

#         # Compute projections
#         vjk = jax.vmap(v)(self.Q.x)  # n_q x d
#         Λijk = jax.vmap(jax.vmap(Λ, (0, None)), (None, 0))(
#             self.Q.x, self.ns)  # n x n_q x d
#         wj = self.Q.w
#         return jnp.einsum("ijk,jk,j->i", Λijk, vjk, wj)

# class ProductProjection:
#     """
#     Given bases Λn, Λv, Λk, constructs an operator to evaluate
#     1) (strong): (w, u) -> ∫ wₕ uₕ · Λn[i] dx for all i, where Λn[i] is the i-th basis function of Λn
#         and wₕ = ∑ w[i] Λk[i], uₕ = ∑ u[i] Λv[i] are the discrete functions
#     2) (weak): (w, u) -> ∫ wₕ uₕ · Λv[i] dx for all i, where Λv[i] is the i-th basis function of Λn
#         and wₕ = ∑ w[i] Λk[i], uₕ = ∑ u[i] Λn[i] are the discrete functions
#     """

#     def __init__(self, Λn, Λv, Λk, Q, F=None, En=None, Ev=None, Ek=None, mode='strong'):
#         """
#         Initialize the operator.
#         Args:
#             Λn: Basis for n-forms (n can be 1 or 2)
#             Λk: Basis for k-forms (k can be 0 or 3)
#             Λv: Basis for 2-forms
#             Q: Quadrature rule for numerical integration
#             F (callable, optional): Coordinate transformation function.
#                                     Defaults to identity mapping.
#             En, Ev, Ek (array, optional): Extraction operator matrix for Λk, Λv, Λ1.
#                                 Defaults to identity matrix.
#         """
#         self.Λk = Λk
#         self.Λv = Λv
#         self.Λn = Λn
#         self.Q = Q
#         if F is None:
#             self.F = lambda x: x
#         else:
#             self.F = F
#         self.mode = mode  # 'strong' or 'weak'
#         if mode not in ['strong', 'weak']:
#             raise ValueError("mode must be 'strong' or 'weak'")
#         self.En = En if En is not None else None
#         self.Ev = Ev if Ev is not None else None
#         self.Ek = Ek if Ek is not None else None
#         if mode == 'strong':
#             self.n = Λn.n
#             self.ns = Λn.ns
#             self.M = jnp.eye(Λn.n) if En is None else En
#         elif mode == 'weak':
#             self.n = Λv.n
#             self.ns = Λv.ns
#             self.M = jnp.eye(Λv.n) if Ev is None else Ev
#         if self.Λv.k != 2:
#             raise ValueError("Λv must be a 2-form")
#         if self.Λk.k not in [0, 3]:
#             raise ValueError("Λk must be a 0-form or 3-form")
#         if self.Λn.k not in [1, 2]:
#             raise ValueError("Λn must be a 1-form or 2-form")

#     def __call__(self, w, u):
#         """
#         evaluates ∫ wₕ uₕ · Λn[i] dx for all i and collects the values in a vector.

#         Args:
#             w (array): k-form dofs
#             u (array): vector field dofs

#         Returns:
#             array: ∫ wₕ uₕ Λn[i] dx for all i
#         """
#         return self.M @ self.projection(w, u)

#     def projection(self, w, u):

#         w_h = DiscreteFunction(w, self.Λk, self.Ek)
#         if self.mode == 'strong':
#             u_h = DiscreteFunction(u, self.Λv, self.Ev)
#         elif self.mode == 'weak':
#             u_h = DiscreteFunction(u, self.Λn, self.En)

#         DF = jax.jacfwd(self.F)

#         if self.Λk.k == 3 and self.Λn.k == 1:
#             def v(x):
#                 return w_h(x) * u_h(x) / jnp.linalg.det(DF(x))
#         elif self.Λk.k == 3 and self.Λn.k == 2:
#             def v(x):
#                 return w_h(x) * (DF(x).T @ DF(x) @ u_h(x)) / jnp.linalg.det(DF(x))**2
#         elif self.Λk.k == 0 and self.Λn.k == 1:
#             def v(x):
#                 return w_h(x) * u_h(x)
#         elif self.Λk.k == 0 and self.Λn.k == 2:
#             def v(x):
#                 return w_h(x) * (DF(x).T @ DF(x) @ u_h(x)) / jnp.linalg.det(DF(x))
#         else:
#             raise ValueError(
#                 "Λn must be a 0/3-form and Λk must be a 1/2-form")

#         if self.mode == 'strong':
#             def Λ(x, i):
#                 return self.Λn(x, i)
#         elif self.mode == 'weak':
#             def Λ(x, i):
#                 return self.Λv(x, i)

#         # Compute projections
#         vjk = jax.vmap(v)(self.Q.x)  # n_q x d
#         Λijk = jax.vmap(jax.vmap(Λ, (0, None)), (None, 0))(
#             self.Q.x, self.ns)  # n x n_q x d
#         wj = self.Q.w
#         return jnp.einsum("ijk,jk,j->i", Λijk, vjk, wj)


# class CurlProjection:
#     """
#     Given one-form A and two-form B, computes ∫ B·(A × Λ[i]) dx for all i,
#     where Λ[i] is the i-th basis function of the one-form space.
#     """

#     def __init__(self, Λ, Q, F=None, E=None):
#         """
#         Initialize the curl projector.

#         Args:
#             Λ: Domain operator defining the finite element space
#             Q: Quadrature rule for numerical integration
#             F (callable, optional): Coordinate transformation function.
#                                  Defaults to identity mapping.
#             E (array, optional): Extraction operator matrix.
#                               Defaults to identity matrix.
#         """
#         self.Λ = Λ
#         self.Q = Q
#         self.n = Λ.n
#         self.ns = Λ.ns
#         if F is None:
#             self.F = lambda x: x
#         else:
#             self.F = F
#         if E is None:
#             self.M = jnp.eye(self.n)
#         else:
#             self.M = E

#     def __call__(self, A, B):
#         """
#         Project the curl operation between forms A and B.

#         Args:
#             A (callable): One-form field
#             B (callable): Two-form field

#         Returns:
#             array: Projection coefficients
#         """
#         return self.M @ self.projection(A, B)

#     def projection(self, A, B):
#         """
#         Compute the projection of (B, A × Λ[i]).

#         Given a one-form A and two-form B, computes the projection of their
#         cross product with the basis functions.

#         Args:
#             A (callable): One-form field
#             B (callable): Two-form field

#         Returns:
#             array: Projection coefficients
#         """
#         DF = jax.jacfwd(self.F)

#         def _B(x):
#             return DF(x) @ B(x)

#         def _Λ(x, i):
#             # Note: cross products of one-forms transform like two-forms
#             return DF(x) @ jnp.cross(A(x), self.Λ(x, i))

#         # Compute projections
#         Bjk = jax.vmap(_B)(self.Q.x)  # n_q x d
#         Λijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(
#             self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x d
#         Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
#         wj = self.Q.w
#         return jnp.einsum("ijk,jk,j,j->i", Λijk, Bjk, 1/Jj, wj)


# class GradientProjection:
#     """
#     Given zero-form p and two-form u, computes
#     ∫ ( grad(p)·u Λ[i] + Ɣ p div(u) Λ[i] ) dx
#     for all i, where Λ[i] is the i-th basis function of the zero-form space.
#     """

#     def __init__(self, Λ, Q, F=None, E=None, Ɣ=5/3):
#         self.Λ = Λ
#         self.Q = Q
#         self.n = Λ.n
#         self.ns = Λ.ns
#         self.Ɣ = Ɣ
#         if F is None:
#             self.F = lambda x: x
#         else:
#             self.F = F
#         if E is None:
#             self.M = jnp.eye(self.n)
#         else:
#             self.M = E

#     def __call__(self, p, u):
#         return self.M @ self.projection(p, u)

#     def projection(self, p, u):
#         def q(x):
#             return grad(p)(x) @ u(x) + self.Ɣ * p(x) * div(u)(x)

#         # Compute projections
#         qjk = jax.vmap(q)(self.Q.x)  # n_q x 1
#         Λijk = jax.vmap(jax.vmap(self.Λ, (0, None)), (None, 0))(
#             self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x 1
#         wj = self.Q.w
#         return jnp.einsum("ijk,jk,j->i", Λijk, qjk, wj)


# class EFieldProjector:
#     """
#     Given four one-forms A, B, C and D,
#     computes ∫ ((A × B - D) × C) · Λ[i] dx for all i,
#     where Λ[i] is the i-th basis function of the one-form space.
#     """

#     def __init__(self, Λ, Q, F=None, E=None):
#         """
#         Initialize the curl projector.

#         Args:
#             Λ: Domain operator defining the finite element space
#             Q: Quadrature rule for numerical integration
#             F (callable, optional): Coordinate transformation function.
#                                  Defaults to identity mapping.
#             E (array, optional): Extraction operator matrix.
#                               Defaults to identity matrix.
#         """
#         self.Λ = Λ
#         self.Q = Q
#         self.n = Λ.n
#         self.ns = Λ.ns
#         if F is None:
#             self.F = lambda x: x
#         else:
#             self.F = F
#         if E is None:
#             self.M = jnp.eye(self.n)
#         else:
#             self.M = E

#     def __call__(self, A, B, C, D):
#         """
#         Project the curl operation between forms A and B.

#         Args:
#             A (callable): One-form field
#             B (callable): One-form field
#             C (callable): One-form field

#         Returns:
#             array: Projection coefficients
#         """
#         return self.M @ self.projection(A, B, C, D)

#     def projection(self, A, B, C, D):
#         """
#         Compute the projection of ∫ ((A × B - D) × C) · Λ[i] dx.

#         Args:
#             A (callable): One-form field
#             B (callable): One-form field
#             C (callable): One-form field
#             D (callable): Zero-form field

#         Returns:
#             array: Projection coefficients
#         """
#         DF = jax.jacfwd(self.F)

#         def u(x):
#             DFx = DF(x)
#             Jx = jnp.linalg.det(DFx)
#             return jnp.cross(1/Jx * DFx.T @ DFx @ jnp.cross(A(x), B(x)) - D(x), C(x))

#         # Compute projections
#         ujk = jax.vmap(u)(self.Q.x)  # n_q x d
#         Λijk = jax.vmap(jax.vmap(self.Λ, (0, None)), (None, 0))(
#             self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x d
#         Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
#         wj = self.Q.w
#         return jnp.einsum("ijk,jk,j,j->i", Λijk, ujk, 1/Jj, wj)


# class ForceProjector:
#     """
#     Given three one-forms A, B and C,
#     computes ∫ (A × B - C) · Λ[i] dx for all i,
#     where Λ[i] is the i-th basis function of the one-form space.
#     """

#     def __init__(self, Λ, Q, F=None, E=None):
#         """
#         Initialize the curl projector.

#         Args:
#             Λ: Domain operator defining the finite element space
#             Q: Quadrature rule for numerical integration
#             F (callable, optional): Coordinate transformation function.
#                                  Defaults to identity mapping.
#             E (array, optional): Extraction operator matrix.
#                               Defaults to identity matrix.
#         """
#         self.Λ = Λ
#         self.Q = Q
#         self.n = Λ.n
#         self.ns = Λ.ns
#         if F is None:
#             self.F = lambda x: x
#         else:
#             self.F = F
#         if E is None:
#             self.M = jnp.eye(self.n)
#         else:
#             self.M = E

#     def __call__(self, A, B, C):
#         """
#         Project the curl operation between forms A and B.

#         Args:
#             A (callable): One-form field
#             B (callable): One-form field
#             C (callable): One-form field

#         Returns:
#             array: Projection coefficients
#         """
#         return self.M @ self.projection(A, B, C)

#     def projection(self, A, B, C):
#         """
#         Compute the projection of ∫ (A × B - C) · Λ[i] dx.

#         Args:
#             A (callable): One-form field
#             B (callable): One-form field
#             C (callable): One-form field

#         Returns:
#             array: Projection coefficients
#         """
#         DF = jax.jacfwd(self.F)

#         def u(x):
#             DFx = DF(x)
#             Jx = jnp.linalg.det(DFx)
#             return jnp.cross(1/Jx * DFx.T @ DFx @ jnp.cross(A(x), B(x)) - C(x), C(x))

#         # Compute projections
#         ujk = jax.vmap(u)(self.Q.x)  # n_q x d
#         Λijk = jax.vmap(jax.vmap(self.Λ, (0, None)), (None, 0))(
#             self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x d
#         Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
#         wj = self.Q.w
#         return jnp.einsum("ijk,jk,j,j->i", Λijk, ujk, 1/Jj, wj)


# class ForceProjector2:
#     """
#     Given three one-forms A, B and C,
#     computes ∫ (A × B - C) · Λ[i] dx for all i,
#     where Λ[i] is the i-th basis function of the two-form space.
#     """

#     def __init__(self, Λ, Q, F=None, E=None):
#         """
#         Initialize the curl projector.

#         Args:
#             Λ: Domain operator defining the finite element space
#             Q: Quadrature rule for numerical integration
#             F (callable, optional): Coordinate transformation function.
#                                  Defaults to identity mapping.
#             E (array, optional): Extraction operator matrix.
#                               Defaults to identity matrix.
#         """
#         self.Λ = Λ
#         self.Q = Q
#         self.n = Λ.n
#         self.ns = Λ.ns
#         if F is None:
#             self.F = lambda x: x
#         else:
#             self.F = F
#         if E is None:
#             self.M = jnp.eye(self.n)
#         else:
#             self.M = E

#     def __call__(self, A, B, C):
#         """
#         Project the curl operation between forms A and B.

#         Args:
#             A (callable): One-form field
#             B (callable): One-form field
#             C (callable): One-form field

#         Returns:
#             array: Projection coefficients
#         """
#         return self.M @ self.projection(A, B, C)

#     def projection(self, A, B, C):
#         """
#         Compute the projection of ∫ (A × B - C) · Λ[i] dx.

#         Args:
#             A (callable): One-form field
#             B (callable): One-form field
#             C (callable): One-form field

#         Returns:
#             array: Projection coefficients
#         """
#         DF = jax.jacfwd(self.F)

#         def u(x):
#             DFx = DF(x)
#             Jx = jnp.linalg.det(DFx)
#             return jnp.cross(1/Jx * DFx.T @ DFx @ jnp.cross(A(x), B(x)) - C(x),  C(x))

#         # Compute projections
#         ujk = jax.vmap(u)(self.Q.x)  # n_q x d
#         Λijk = jax.vmap(jax.vmap(self.Λ, (0, None)), (None, 0))(
#             self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x d
#         Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
#         wj = self.Q.w
#         return jnp.einsum("ijk,jk,j,j->i", Λijk, ujk, 1/Jj, wj)


# class EFieldProjector:
#     """
#     Given four one-forms A, B, C and D,
#     computes ∫ ((A × B - D) × C) · Λ[i] dx for all i,
#     where Λ[i] is the i-th basis function of the one-form space.
#     """

#     def __init__(self, Λ, Q, F=None, E=None):
#         """
#         Initialize the curl projector.

#         Args:
#             Λ: Domain operator defining the finite element space
#             Q: Quadrature rule for numerical integration
#             F (callable, optional): Coordinate transformation function.
#                                  Defaults to identity mapping.
#             E (array, optional): Extraction operator matrix.
#                               Defaults to identity matrix.
#         """
#         self.Λ = Λ
#         self.Q = Q
#         self.n = Λ.n
#         self.ns = Λ.ns
#         if F is None:
#             self.F = lambda x: x
#         else:
#             self.F = F
#         if E is None:
#             self.M = jnp.eye(self.n)
#         else:
#             self.M = E

#     def __call__(self, A, B, C, D):
#         """
#         Project the curl operation between forms A and B.

#         Args:
#             A (callable): One-form field
#             B (callable): One-form field
#             C (callable): One-form field

#         Returns:
#             array: Projection coefficients
#         """
#         return self.M @ self.projection(A, B, C, D)

#     def projection(self, A, B, C, D):
#         """
#         Compute the projection of ∫ ((A × B - D) × C) · Λ[i] dx.

#         Args:
#             A (callable): One-form field
#             B (callable): One-form field
#             C (callable): One-form field
#             D (callable): Zero-form field

#         Returns:
#             array: Projection coefficients
#         """
#         DF = jax.jacfwd(self.F)

#         def u(x):
#             DFx = DF(x)
#             Jx = jnp.linalg.det(DFx)
#             return jnp.cross(1/Jx * DFx.T @ DFx @ jnp.cross(A(x), B(x)) - D(x), C(x))

#         # Compute projections
#         ujk = jax.vmap(u)(self.Q.x)  # n_q x d
#         Λijk = jax.vmap(jax.vmap(self.Λ, (0, None)), (None, 0))(
#             self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x d
#         Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
#         wj = self.Q.w
#         return jnp.einsum("ijk,jk,j,j->i", Λijk, ujk, 1/Jj, wj)
