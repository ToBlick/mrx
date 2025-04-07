import jax
import jax.numpy as jnp
import numpy as np

from mrx.Utils import jacobian, inv33, curl, div, grad
from mrx.DifferentialForms import DifferentialForm
from mrx.Projectors import Projector
from mrx.Quadrature import QuadratureRule


class LazyMassMatrix:
    def __init__(self, Λ, Q, F=None, E=None):
        self.Λ = Λ
        self.Q = Q
        self.n = Λ.n
        self.ns = Λ.ns
        self.F = F if F is not None else lambda x: x
        self.E = E if E is not None else jnp.eye(self.n)

        match self.Λ.k:
            case 0:
                self.M = self.E @ self.zeroform_assemble() @ self.E.T
            case 1:
                self.M = self.E @ self.oneform_assemble() @ self.E.T
            case 2:
                self.M = self.E @ self.twoform_assemble() @ self.E.T
            case 3:
                self.M = self.E @ self.threeform_assemble() @ self.E.T

    def __getitem__(self, i):
        return self.M[i]

    def __array__(self):
        return np.array(self.M)

    def zeroform_assemble(self):
        Λijk = jax.vmap(jax.vmap(self.Λ, (0, None)), (None, 0))(self.Q.x, self.ns)  # n x n_q x 1
        # evaluate the jacobian of F at all quadrature points
        Jj = jax.vmap(jacobian(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,ljk,j,j->li", Λijk, Λijk, Jj, wj)

    def oneform_assemble(self):
        DF = jax.jacfwd(self.F)

        def _Λ(x, i):
            return inv33(DF(x)).T @ self.Λ(x, i)
        Λijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(self.Q.x, jnp.arange(self.n))  # n x n_q x d
        Jj = jax.vmap(jacobian(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,ljk,j,j->li", Λijk, Λijk, Jj, wj)

    def twoform_assemble(self):
        DF = jax.jacfwd(self.F)

        def _Λ(x, i):
            return DF(x) @ self.Λ(x, i)
        Λijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x d
        Jj = jax.vmap(jacobian(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w
        return jnp.einsum("ijk,ljk,j,j->li", Λijk, Λijk, 1/Jj, wj)

    def threeform_assemble(self):
        # evaluate all basis functions at all quadrature points
        Λijk = jax.vmap(jax.vmap(self.Λ, (0, None)), (None, 0))(self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x 1
        Jj = jax.vmap(jacobian(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,ljk,j,j->li", Λijk, Λijk, 1/Jj, wj)


class LazyDerivativeMatrix:
    def __init__(self, Λ0, Λ1, Q, F=None, E0=None, E1=None):
        self.Λ0 = Λ0
        self.Λ1 = Λ1
        self.Q = Q
        self.n0 = Λ0.n
        self.ns0 = Λ0.ns
        self.n1 = Λ1.n
        self.ns1 = Λ1.ns
        self.F = F if F is not None else lambda x: x
        self.E0 = jnp.eye(self.n0) if E0 is None else E0
        self.E1 = jnp.eye(self.n1) if E1 is None else E1

        match self.Λ0.k:
            case 0:
                self.M = self.E1 @ self.gradient_assemble() @ self.E0.T
            case 1:
                self.M = self.E1 @ self.curl_assemble() @ self.E0.T
            case 2:
                self.M = self.E1 @ self.div_assemble() @ self.E0.T
            case 3:
                print("Warning: No derivative operator for 3-forms")
                self.M = jnp.zeros((self.n0, self.n1))

    def __getitem__(self, i):
        return self.M[i]

    def __array__(self):
        return np.array(self.M)

    def gradient_assemble(self):
        # evaluate the jacobian of F at all quadrature points
        DF = jax.jacfwd(self.F)

        def _Λ0(x, i):
            return inv33(DF(x)).T @ grad(lambda y: self.Λ0(y, i))(x)

        def _Λ1(x, i):
            return inv33(DF(x)).T @ self.Λ1(x, i)
        Λ0_ijk = jax.vmap(jax.vmap(_Λ0, (0, None)), (None, 0))(self.Q.x, jnp.arange(self.n0))  # n0 x n_q x d
        Λ1_ijk = jax.vmap(jax.vmap(_Λ1, (0, None)), (None, 0))(self.Q.x, jnp.arange(self.n1))  # n1 x n_q x d
        Jj = jax.vmap(jacobian(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,ljk,j,j->li", Λ0_ijk, Λ1_ijk, Jj, wj)

    def curl_assemble(self):
        DF = jax.jacfwd(self.F)

        def _Λ0(x, i):
            return DF(x) @ curl(lambda y: self.Λ0(y, i))(x)

        def _Λ1(x, i):
            return DF(x) @ self.Λ1(x, i)
        Λ0_ijk = jax.vmap(jax.vmap(_Λ0, (0, None)), (None, 0))(self.Q.x, jnp.arange(self.n0))  # n0 x n_q x d
        Λ1_ijk = jax.vmap(jax.vmap(_Λ1, (0, None)), (None, 0))(self.Q.x, jnp.arange(self.n1))  # n1 x n_q x d
        Jj = jax.vmap(jacobian(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,ljk,j,j->li", Λ0_ijk, Λ1_ijk, 1/Jj, wj)

    def div_assemble(self):
        def _Λ0(x, i):
            return div(lambda y: self.Λ0(y, i))(x)
        Λ0_ijk = jax.vmap(jax.vmap(_Λ0, (0, None)), (None, 0))(self.Q.x, jnp.arange(self.n0))  # n0 x n_q x 1
        Λ1_ijk = jax.vmap(jax.vmap(self.Λ1, (0, None)), (None, 0))(self.Q.x, jnp.arange(self.n1))  # n1 x n_q x 1
        Jj = jax.vmap(jacobian(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,ljk,j,j->li", Λ0_ijk, Λ1_ijk, 1/Jj, wj)


class LazyProjectionMatrix:
    def __init__(self, Λ0, Λ1, Q, F=None, E0=None, E1=None):
        self.Λ0 = Λ0
        self.Λ1 = Λ1
        self.Q = Q
        self.n0 = Λ0.n
        self.ns0 = Λ0.ns
        self.n1 = Λ1.n
        self.ns1 = Λ1.ns
        self.F = F if F is not None else lambda x: x
        self.E0 = E0 if E0 is not None else jnp.eye(self.n0)
        self.E1 = E1 if E1 is not None else jnp.eye(self.n1)
        self.M = self.E0 @ self.assemble() @ self.E1.T

    def __getitem__(self, i):
        return self.M[i]

    def __array__(self):
        return np.array(self.M)

    def assemble(self):
        Λ0_ijk = jax.vmap(jax.vmap(self.Λ0, (0, None)), (None, 0))(self.Q.x, self.ns0)  # n0 x n_q x d
        Λ1_ijk = jax.vmap(jax.vmap(self.Λ1, (0, None)), (None, 0))(self.Q.x, self.ns1)  # n0 x n_q x d
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,ljk,j->il", Λ0_ijk, Λ1_ijk, wj)


class LazyDoubleCurlMatrix:
    def __init__(self, Λ, Q, F=None, E=None):
        self.Λ = Λ
        self.Q = Q
        self.n = Λ.n
        self.ns = Λ.ns

        self.F = F if F is not None else lambda x: x
        self.E = E if E is not None else jnp.eye(self.n)
        self.M = self.E @ self.assemble() @ self.E.T

    def __getitem__(self, i):
        return self.M[i]

    def __array__(self):
        return np.array(self.M)

    def assemble(self):
        # evaluate the jacobian of F at all quadrature points
        DF = jax.jacfwd(self.F)

        def _Λ(x, i):
            return DF(x) @ curl(lambda y: self.Λ(y, i))(x)
        Λ_ijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(self.Q.x, jnp.arange(self.n))  # n x n_q x d
        Jj = jax.vmap(jacobian(self.F))(self.Q.x)
        wj = self.Q.w
        return jnp.einsum("ijk,ljk,j,j->li", Λ_ijk, Λ_ijk, 1/Jj, wj)


class LazyStiffnessMatrix:
    def __init__(self, Λ, Q, F=None, E=None):
        self.Λ = Λ
        self.Q = Q
        self.n = Λ.n
        self.ns = Λ.ns

        self.F = F if F is not None else lambda x: x
        self.E = E if E is not None else jnp.eye(self.n)
        self.M = self.E @ self.assemble() @ self.E.T

    def __getitem__(self, i):
        return self.M[i]

    def __array__(self):
        return np.array(self.M)

    def assemble(self):
        # evaluate the jacobian of F at all quadrature points
        DF = jax.jacfwd(self.F)

        def _Λ(x, i):
            return inv33(DF(x)).T @ grad(lambda y: self.Λ(y, i))(x)
        Λ_ijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(self.Q.x, jnp.arange(self.n))  # n x n_q x d
        Jj = jax.vmap(jacobian(self.F))(self.Q.x)
        wj = self.Q.w
        return jnp.einsum("ijk,ljk,j,j->li", Λ_ijk, Λ_ijk, Jj, wj)
