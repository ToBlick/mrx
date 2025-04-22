from abc import abstractmethod
import jax
import jax.numpy as jnp
import numpy as np

from mrx.DifferentialForms import DifferentialForm
from mrx.Quadrature import QuadratureRule

from mrx.Utils import jacobian, inv33, curl, div, grad

class LazyMatrix:
    """
    A class to represent a lazy matrix assembly for finite element computations.
    
    In all generality, i,j-th entries of this class will be something like: ∫ L(Λ0[i])·K(Λ1[j]) dx, where L and K are (differential) operators that usually depend on F.

    Attributes that can be passed to the constructor:
        Λ0 (DifferentialForm): The input differential form.
        Λ1 (DifferentialForm): The output differential form.
        Q (QuadratureRule): The quadrature rule used for numerical integration.
        F (callable, optional): Map from logical to physical domain. Defaults to the identity function.
        E0 (jnp.ndarray, optional): Transformation matrix for Λ0 - either for extraction of basis functions in polar domains or to apply boundary conditions. Defaults to the identity matrix.
        E1 (jnp.ndarray): Transformation matrix for Λ1.
    Other attributes:
        n0 (int): The number of basis functions for Λ0.
        n1 (int): The number of basis functions for Λ1.
        ns0 (jnp.ndarray): jnp.arange(n0)
        ns1 (jnp.ndarray): jnp.arange(n1)
        M (jnp.ndarray): The assembled derivative matrix.
    Methods:
        __getitem__(i):
            Access a specific row/element of the assembled mass matrix.
        __array__():
            Convert the assembled mass matrix to a NumPy array.
        assemble():
            Assemble the matrix.       
    Notes:
        - Any subclass must implement the assemble method.
    """
    Λ0: DifferentialForm
    Λ1: DifferentialForm
    Q: QuadratureRule
    F: callable
    E0: jnp.ndarray
    E1: jnp.ndarray
    n0: int
    n1: int  
    ns0: jnp.ndarray
    ns1: jnp.ndarray
    M: jnp.ndarray
    
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
        self.M = self.E1 @ self.assemble() @ self.E0.T

    def __getitem__(self, i):
        return self.M[i]

    def __array__(self):
        return np.array(self.M)
    
    @abstractmethod
    def assemble(self):
        pass

class LazyMassMatrix(LazyMatrix):
    """
    This class supports the assembly of mass matrices for different forms (0-form, 1-form, 2-form, and 3-form).
    
    More precisely:
        - for 0-forms the i,j-th element is ∫ Λ0[i] Λ1[j] detDF dx.
        - for 1-forms the i,j-th element is ∫ DF.-T Λ0[i] · DF.-T Λ1[j] detDF dx.
        - for 2-forms the i,j-th element is ∫ DF Λ0[i] · DF Λ1[j] 1/detDF dx.
        - for 3-forms the i,j-th element is ∫ Λ0[i] Λ1[j] 1/detDF dx.
        
    Notes:
        - provides a convenience constructor that takes fewer arguments since Λ0 = Λ1.
    """
    
    def __init__(self, Λ, Q, F=None, E=None):
        super().__init__(Λ, Λ, Q, F, E, E)

    def assemble(self):
        match self.Λ0.k:
            case 0:
                return self.zeroform_assemble()
            case 1:
                return self.oneform_assemble()
            case 2:
                return self.twoform_assemble()
            case 3:
                return self.threeform_assemble()

    def zeroform_assemble(self):
        Λijk = jax.vmap(jax.vmap(self.Λ0, (0, None)), (None, 0))(self.Q.x, self.ns0)  # n x n_q x 1
        # evaluate the jacobian of F at all quadrature points
        Jj = jax.vmap(jacobian(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,ljk,j,j->li", Λijk, Λijk, Jj, wj)

    def oneform_assemble(self):
        DF = jax.jacfwd(self.F)

        def _Λ(x, i):
            return inv33(DF(x)).T @ self.Λ0(x, i)
        Λijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(self.Q.x, jnp.arange(self.n0))  # n x n_q x d
        Jj = jax.vmap(jacobian(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,ljk,j,j->li", Λijk, Λijk, Jj, wj)

    def twoform_assemble(self):
        DF = jax.jacfwd(self.F)

        def _Λ(x, i):
            return DF(x) @ self.Λ0(x, i)
        Λijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(self.Q.x, jnp.arange(self.Λ0.n))  # n x n_q x d
        Jj = jax.vmap(jacobian(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w
        return jnp.einsum("ijk,ljk,j,j->li", Λijk, Λijk, 1/Jj, wj)

    def threeform_assemble(self):
        # evaluate all basis functions at all quadrature points
        Λijk = jax.vmap(jax.vmap(self.Λ0, (0, None)), (None, 0))(self.Q.x, jnp.arange(self.Λ0.n))  # n x n_q x 1
        Jj = jax.vmap(jacobian(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,ljk,j,j->li", Λijk, Λijk, 1/Jj, wj)


class LazyDerivativeMatrix(LazyMatrix):
    """
    Class representing a matrix for computing derivatives of differential forms.
    
    It represents gradient, curl, and divergence operations depending on the degree of the input differential form.
    
    More precisely:
        - for (Λ0, Λ1) = (0-form, 1-form), the i,j-th element is ∫ DF.-T grad Λ0[i] · DF.-T Λ1[j] detDF dx.
        - for (Λ0, Λ1) = (1-form, 2-form), the i,j-th element is ∫ DF curl Λ0[i] · DF Λ1[j] 1/detDF dx.
        - for (Λ0, Λ1) = (2-form, 3-form), the i,j-th element is ∫ div Λ0[i] Λ1[j] 1/detDF dx.
    """
    
    def assemble(self):
        match self.Λ0.k:
            case 0:
                return self.gradient_assemble()
            case 1:
                return self.curl_assemble()
            case 2:
                return self.div_assemble()
            case 3:
                print("Warning: No derivative operator for 3-forms")
                return jnp.zeros((self.n0, self.n1))

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


class LazyProjectionMatrix(LazyMatrix):

    def assemble(self):
        Λ0_ijk = jax.vmap(jax.vmap(self.Λ0, (0, None)), (None, 0))(self.Q.x, self.ns0)  # n0 x n_q x d
        Λ1_ijk = jax.vmap(jax.vmap(self.Λ1, (0, None)), (None, 0))(self.Q.x, self.ns1)  # n0 x n_q x d
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,ljk,j->li", Λ0_ijk, Λ1_ijk, wj)


class LazyDoubleCurlMatrix(LazyMatrix):
    """
    Class representing a matrix that is half a vector Laplace operator.
    
    More precisely, the i,j-th element is ∫ DF curl Λ0[i] · DF curl Λ1[j] 1/detDF dx.
    """
    def __init__(self, Λ, Q, F=None, E=None):
        super().__init__(Λ, Λ, Q, F, E, E)


    def assemble(self):
        # evaluate the jacobian of F at all quadrature points
        DF = jax.jacfwd(self.F)

        def _Λ(x, i):
            return DF(x) @ curl(lambda y: self.Λ0(y, i))(x)
        Λ_ijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(self.Q.x, jnp.arange(self.n0))  # n x n_q x d
        Jj = jax.vmap(jacobian(self.F))(self.Q.x)
        wj = self.Q.w
        return jnp.einsum("ijk,ljk,j,j->li", Λ_ijk, Λ_ijk, 1/Jj, wj)

class LazyStiffnessMatrix(LazyMatrix):
    """
    Class representing a Laplace operator matrix.
    
    More precisely, the i,j-th element is ∫ DF.-T grad Λ0[i] · DF.-T grad Λ1[j] detDF dx.
    """
    def __init__(self, Λ, Q, F=None, E=None):
        super().__init__(Λ, Λ, Q, F, E, E)

    def assemble(self):
        # evaluate the jacobian of F at all quadrature points
        DF = jax.jacfwd(self.F)

        def _Λ(x, i):
            return inv33(DF(x)).T @ grad(lambda y: self.Λ0(y, i))(x)
        Λ_ijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(self.Q.x, jnp.arange(self.n0))  # n x n_q x d
        Jj = jax.vmap(jacobian(self.F))(self.Q.x)
        wj = self.Q.w
        return jnp.einsum("ijk,ljk,j,j->li", Λ_ijk, Λ_ijk, Jj, wj)
