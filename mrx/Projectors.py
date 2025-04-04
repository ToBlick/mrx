import jax
import jax.numpy as jnp

from mrx.Utils import jacobian, inv33

class Projector:
    def __init__(self, Λ, Q, F=None, E=None):
        self.Λ = Λ
        self.Q = Q
        self.n = Λ.n
        self.ns = Λ.ns
        if F is None:
            self.F = lambda x: x
        else:
            self.F = F
        if E is None:
            self.M = jnp.eye(self.n)
        else:
            self.M = E
            
    # def __call__(self, f):
    #     return self.M @ self.projection(f)
            
    # def projection(self, f):
    #     # evaluate all basis functions at all quadrature points
    #     Λijk = jax.vmap(jax.vmap(self.Λ, (0, None)), (None, 0))(self.Q.x, self.ns) # n x n_q x d
    #     # evalute the given function at all quadrature points
    #     fjk = jax.vmap(f)(self.Q.x) # n_q x d
    #     # evaluate the jacobian of F at all quadrature points
    #     wj = self.Q.w # n_q
    #     return jnp.einsum("ijk,jk,j->i", Λijk, fjk, wj)

    def __call__(self, f):
        if self.Λ.k == 0:
            return self.M @ self.zeroform_projection(f)
        elif self.Λ.k == 1:
            return self.M @ self.oneform_projection(f)
        elif self.Λ.k == 2:
            return self.M @ self.twoform_projection(f)
        elif self.Λ.k == 3:
            return self.M @ self.threeform_projection(f)

    def zeroform_projection(self, f):
        # evaluate all basis functions at all quadrature points
        Λijk = jax.vmap(jax.vmap(self.Λ, (0, None)), (None, 0))(self.Q.x, self.ns) 
        # n x n_q x 1 - evalute the given function at all quadrature points
        fjk = jax.vmap(f)(self.Q.x) # n_q x 1
        # evaluate the jacobian of F at all quadrature points
        Jj = jax.vmap(jacobian(self.F))(self.Q.x) # n_q x 1
        wj = self.Q.w # n_q
        return jnp.einsum("ijk,jk,j,j->i", Λijk, fjk, Jj, wj)

    def oneform_projection(self, A):
        DF = jax.jacfwd(self.F)
        def _A(x):
            return inv33(DF(x)).T @ A(x)
        def _Λ(x, i):
            return inv33(DF(x)).T @ self.Λ(x, i)
        Ajk = jax.vmap(_A)(self.Q.x) # n_q x d    
        Λijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(self.Q.x, jnp.arange(self.Λ.n)) #  n x n_q x d
        Jj = jax.vmap(jacobian(self.F))(self.Q.x)
        wj = self.Q.w
        return jnp.einsum("ijk,jk,j,j->i", Λijk, Ajk, Jj, wj)

    def twoform_projection(self, B):
        DF = jax.jacfwd(self.F)
        def _B(x):
            return DF(x) @ B(x)
        def _Λ(x, i):
            return DF(x) @ self.Λ(x, i)
        Bjk = jax.vmap(_B)(self.Q.x) # n_q x d    
        Λijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(self.Q.x, jnp.arange(self.Λ.n)) #  n x n_q x d
        Jj = jax.vmap(jacobian(self.F))(self.Q.x) # n_q x 1
        wj = self.Q.w
        return jnp.einsum("ijk,jk,j,j->i", Λijk, Bjk, 1/Jj, wj)
    
    def threeform_projection(self, f):
        # evaluate all basis functions at all quadrature points
        Λijk = jax.vmap(jax.vmap(self.Λ, (0, None)), (None, 0))(self.Q.x, jnp.arange(self.Λ.n)) # n x n_q x 1 
        fjk = jax.vmap(f)(self.Q.x) # n_q x 1
        Jj = jax.vmap(jacobian(self.F))(self.Q.x) # n_q x 1
        wj = self.Q.w # n_q
        return jnp.einsum("ijk,jk,j,j->i", Λijk, fjk, 1/Jj, wj)
    
class CurlProjection:
    def __init__(self, Λ, Q, F=None, E=None):
        self.Λ = Λ
        self.Q = Q
        self.n = Λ.n
        self.ns = Λ.ns
        if F is None:
            self.F = lambda x: x
        else:
            self.F = F
        if E is None:
            self.M = jnp.eye(self.n)
        else:
            self.M = E
            
    def __call__(self, A, B):
        return self.M @ self.projection(A, B)
            
    # Given a one-form A and two-form B, computes (B, A x Λ[i])
    def projection(self, A, B):
        DF = jax.jacfwd(self.F)
        def _B(x):
            return DF(x) @ B(x)
        def _Λ(x, i):
            # note that cross products of one-forms transform like two-froms
            return DF(x) @ jnp.cross(A(x), self.Λ(x, i))
        Bjk = jax.vmap(_B)(self.Q.x) # n_q x d    
        Λijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(self.Q.x, jnp.arange(self.Λ.n)) #  n x n_q x d
        Jj = jax.vmap(jacobian(self.F))(self.Q.x) # n_q x 1
        wj = self.Q.w
        return jnp.einsum("ijk,jk,j,j->i", Λijk, Bjk, 1/Jj, wj)