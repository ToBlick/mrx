from abc import abstractmethod

import jax
import jax.numpy as jnp

import numpy as np

from mrx.DifferentialForms import DifferentialForm
from mrx.Quadrature import QuadratureRule
from mrx.Utils import curl, div, grad, inv33, jacobian_determinant



__all__ = ['LazyMatrix', 'LazyMassMatrix', 'LazyDerivativeMatrix',
           'LazyProjectionMatrix', 'LazyDoubleCurlMatrix', 'LazyDoubleDivergenceMatrix', 'LazyStiffnessMatrix', 'LazyWeightedDoubleDivergenceMatrix', 'LazyMagneticTensionMatrix', 'LazyPressureGradientForceMatrix', 'LazyCurrentDensityMatrix']


class LazyMatrix:
    """
    A class to represent a lazy matrix assembly for finite element computations.

    This class provides a framework for assembling matrices in finite element methods
    where the matrix entries are computed on-demand rather than all at once. The matrix
    entries typically represent integrals of the form ∫ L(Λ0[i])·K(Λ1[j]) dx, where
    L and K are differential operators that may depend on a mapping function F.

    Attributes:
        Λ0 (DifferentialForm): The input differential form.
        Λ1 (DifferentialForm): The output differential form.
        Q (QuadratureRule): The quadrature rule used for numerical integration.
        F (callable): Map from logical to physical domain. Defaults to identity.
        E0 (jnp.ndarray): Transformation matrix for Λ0. Defaults to identity matrix.
        E1 (jnp.ndarray): Transformation matrix for Λ1. Defaults to identity matrix.
        n0 (int): Number of basis functions for Λ0.
        n1 (int): Number of basis functions for Λ1.
        ns0 (jnp.ndarray): Array of indices for Λ0 basis functions.
        ns1 (jnp.ndarray): Array of indices for Λ1 basis functions.
        M (jnp.ndarray): The assembled matrix.

    Methods:
        __init__(Λ0, Λ1, Q, F=None, E0=None, E1=None):
            Initialize the lazy matrix with given differential forms and parameters.
        __getitem__(i):
            Access a specific row/element of the assembled matrix.
        __array__():
            Convert the assembled matrix to a NumPy array.
        assemble():
            Abstract method to assemble the matrix. Must be implemented by subclasses.

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
        """
        Initialize the lazy matrix.

        Args:
            Λ0 (DifferentialForm): The input differential form.
            Λ1 (DifferentialForm): The output differential form.
            Q (QuadratureRule): The quadrature rule for numerical integration.
            F (callable, optional): Map from logical to physical domain. Defaults to identity.
            E0 (jnp.ndarray, optional): Transformation matrix for Λ0. Defaults to identity.
            E1 (jnp.ndarray, optional): Transformation matrix for Λ1. Defaults to identity.
        """
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
        """Access a specific row/element of the assembled matrix."""
        return self.M[i]

    def __array__(self, dtype=None, copy=True):
        """Convert the assembled matrix to a NumPy array."""
        result = np.array(self.M, dtype=dtype)
        if not copy:
            # If copy=False is requested but we can't avoid copying, 
            # we still need to return a copy since self.M is a JAX array
            pass
        return result

    @abstractmethod
    def assemble(self):
        """Assemble the matrix. Must be implemented by subclasses."""
        pass


class LazyMassMatrix(LazyMatrix):
    """
    A class for assembling mass matrices for different differential forms.

    This class supports the assembly of mass matrices for 0-forms, 1-forms, 2-forms,
    and 3-forms. The matrix entries are computed as follows:

    - For 0-forms: ∫ Λ0[i] Λ1[j] detDF dx
    - For 1-forms: ∫ DF.-T Λ0[i] · DF.-T Λ1[j] detDF dx
    - For 2-forms: ∫ DF Λ0[i] · DF Λ1[j] 1/detDF dx
    - For 3-forms: ∫ Λ0[i] Λ1[j] 1/detDF dx

    Attributes:
        Inherits all attributes from LazyMatrix.

    Methods:
        __init__(Λ, Q, F=None, E=None):
            Initialize the mass matrix with a single differential form.
        assemble():
            Assemble the mass matrix based on the form degree.
        zeroform_assemble():
            Assemble the mass matrix for 0-forms.
        oneform_assemble():
            Assemble the mass matrix for 1-forms.
        twoform_assemble():
            Assemble the mass matrix for 2-forms.
        threeform_assemble():
            Assemble the mass matrix for 3-forms.
    """

    def __init__(self, Λ, Q, F=None, E=None):
        """
        Initialize the mass matrix with a single differential form.

        Args:
            Λ (DifferentialForm): The differential form.
            Q (QuadratureRule): The quadrature rule.
            F (callable, optional): Map from logical to physical domain. Defaults to identity.
            E (jnp.ndarray, optional): Transformation matrix. Defaults to identity.
        """
        super().__init__(Λ, Λ, Q, F, E, E)

    def assemble(self):
        """Assemble the mass matrix based on the form degree."""
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
        """Assemble the mass matrix for 0-forms."""
        Λijk = jax.vmap(jax.vmap(self.Λ0, (0, None)), (None, 0))(
            self.Q.x, self.ns0)  # n x n_q x 1
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,ljk,j,j->li", Λijk, Λijk, Jj, wj)

    def oneform_assemble(self):
        """Assemble the mass matrix for 1-forms."""
        DF = jax.jacfwd(self.F)

        def _Λ(x, i):
            return inv33(DF(x)).T @ self.Λ0(x, i)
        Λijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.n0))  # n x n_q x d
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,ljk,j,j->li", Λijk, Λijk, Jj, wj)

    def twoform_assemble(self):
        """Assemble the mass matrix for 2-forms."""
        DF = jax.jacfwd(self.F)

        def _Λ(x, i):
            return DF(x) @ self.Λ0(x, i)
        Λijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.Λ0.n))  # n x n_q x d
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w
        return jnp.einsum("ijk,ljk,j,j->li", Λijk, Λijk, 1/Jj, wj)

    def threeform_assemble(self):
        """Assemble the mass matrix for 3-forms."""
        Λijk = jax.vmap(jax.vmap(self.Λ0, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.Λ0.n))  # n x n_q x 1
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,ljk,j,j->li", Λijk, Λijk, 1/Jj, wj)


class LazyDerivativeMatrix(LazyMatrix):
    """
    A class for computing derivative matrices of differential forms.

    This class represents gradient, curl, and divergence operations depending on the
    degree of the input differential form. The matrix entries are computed as follows:

    - For (Λ0, Λ1) = (0-form, 1-form): ∫ DF.-T grad Λ0[i] · DF.-T Λ1[j] detDF dx
    - For (Λ0, Λ1) = (1-form, 2-form): ∫ DF curl Λ0[i] · DF Λ1[j] 1/detDF dx
    - For (Λ0, Λ1) = (2-form, 3-form): ∫ div Λ0[i] Λ1[j] 1/detDF dx

    Attributes:
        Inherits all attributes from LazyMatrix.

    Methods:
        assemble():
            Assemble the derivative matrix based on the form degree.
        gradient_assemble():
            Assemble the gradient matrix for 0-forms.
        curl_assemble():
            Assemble the curl matrix for 1-forms.
        div_assemble():
            Assemble the divergence matrix for 2-forms.
    """

    def assemble(self):
        """Assemble the derivative matrix based on the form degree."""
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
        """Assemble the gradient matrix for 0-forms."""
        DF = jax.jacfwd(self.F)

        def _Λ0(x, i):
            return inv33(DF(x)).T @ grad(lambda y: self.Λ0(y, i))(x)

        def _Λ1(x, i):
            return inv33(DF(x)).T @ self.Λ1(x, i)
        Λ0_ijk = jax.vmap(jax.vmap(_Λ0, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.n0))  # n0 x n_q x d
        Λ1_ijk = jax.vmap(jax.vmap(_Λ1, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.n1))  # n1 x n_q x d
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,ljk,j,j->li", Λ0_ijk, Λ1_ijk, Jj, wj)

    def curl_assemble(self):
        """Assemble the curl matrix for 1-forms."""
        DF = jax.jacfwd(self.F)

        def _Λ0(x, i):
            return DF(x) @ curl(lambda y: self.Λ0(y, i))(x)

        def _Λ1(x, i):
            return DF(x) @ self.Λ1(x, i)
        Λ0_ijk = jax.vmap(jax.vmap(_Λ0, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.n0))  # n0 x n_q x d
        Λ1_ijk = jax.vmap(jax.vmap(_Λ1, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.n1))  # n1 x n_q x d
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,ljk,j,j->li", Λ0_ijk, Λ1_ijk, 1/Jj, wj)

    def div_assemble(self):
        """Assemble the divergence matrix for 2-forms."""
        def _Λ0(x, i):
            return div(lambda y: self.Λ0(y, i))(x)
        Λ0_ijk = jax.vmap(jax.vmap(_Λ0, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.n0))  # n0 x n_q x 1
        Λ1_ijk = jax.vmap(jax.vmap(self.Λ1, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.n1))  # n1 x n_q x 1
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,ljk,j,j->li", Λ0_ijk, Λ1_ijk, 1/Jj, wj)


class LazyProjectionMatrix(LazyMatrix):
    """
    A class for assembling projection matrices between differential forms.

    The matrix entries are computed as ∫ Λ0[i] · Λ1[j] dx, where Λ0 and Λ1
    are the input and output differential forms, respectively.

    Attributes:
        Inherits all attributes from LazyMatrix.

    Methods:
        assemble():
            Assemble the projection matrix.
    """

    def assemble(self):
        """Assemble the projection matrix."""
        Λ0_ijk = jax.vmap(jax.vmap(self.Λ0, (0, None)), (None, 0))(
            self.Q.x, self.ns0)  # n0 x n_q x d
        Λ1_ijk = jax.vmap(jax.vmap(self.Λ1, (0, None)), (None, 0))(
            self.Q.x, self.ns1)  # n0 x n_q x d
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,ljk,j->li", Λ0_ijk, Λ1_ijk, wj)


class LazyDoubleCurlMatrix(LazyMatrix):
    """
    A class representing a matrix that is half a vector Laplace operator.

    The matrix entries are computed as ∫ DF curl Λ0[i] · DF curl Λ1[j] 1/detDF dx.

    Attributes:
        Inherits all attributes from LazyMatrix.

    Methods:
        __init__(Λ, Q, F=None, E=None):
            Initialize the double curl matrix with a single differential form.
        assemble():
            Assemble the double curl matrix.
    """

    def __init__(self, Λ, Q, F=None, E=None):
        """
        Initialize the double curl matrix with a single differential form.

        Args:
            Λ (DifferentialForm): The differential form.
            Q (QuadratureRule): The quadrature rule.
            F (callable, optional): Map from logical to physical domain. Defaults to identity.
            E (jnp.ndarray, optional): Transformation matrix. Defaults to identity.
        """
        super().__init__(Λ, Λ, Q, F, E, E)

    def assemble(self):
        """Assemble the double curl matrix."""
        DF = jax.jacfwd(self.F)

        def _Λ(x, i):
            return DF(x) @ curl(lambda y: self.Λ0(y, i))(x)
        Λ_ijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.n0))  # n x n_q x d
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)
        wj = self.Q.w
        return jnp.einsum("ijk,ljk,j,j->li", Λ_ijk, Λ_ijk, 1/Jj, wj)


class LazyStiffnessMatrix(LazyMatrix):
    """
    A class representing a Laplace operator matrix.

    The matrix entries are computed as ∫ DF.-T grad Λ0[i] · DF.-T grad Λ1[j] detDF dx.

    Attributes:
        Inherits all attributes from LazyMatrix.

    Methods:
        __init__(Λ, Q, F=None, E=None):
            Initialize the stiffness matrix with a single differential form.
        assemble():
            Assemble the stiffness matrix.
    """

    def __init__(self, Λ, Q, F=None, E=None):
        """
        Initialize the stiffness matrix with a single differential form.

        Args:
            Λ (DifferentialForm): The differential form.
            Q (QuadratureRule): The quadrature rule.
            F (callable, optional): Map from logical to physical domain. Defaults to identity.
            E (jnp.ndarray, optional): Transformation matrix. Defaults to identity.
        """
        super().__init__(Λ, Λ, Q, F, E, E)

    def assemble(self):
        """Assemble the stiffness matrix."""
        DF = jax.jacfwd(self.F)

        def _Λ(x, i):
            return inv33(DF(x)).T @ grad(lambda y: self.Λ0(y, i))(x)
        Λ_ijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.n0))  # n x n_q x d
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,ljk,j,j->li", Λ_ijk, Λ_ijk, Jj, wj)


class LazyDoubleDivergenceMatrix(LazyMatrix):
    """
    A class representing a matrix that is half a vector Laplace operator.

    The matrix entries are computed as ∫ div Λ0[i] ·div Λ1[j] 1/detDF dx.

    Attributes:
        Inherits all attributes from LazyMatrix.

    Methods:
        __init__(Λ, Q, F=None, E=None):
            Initialize the double divergence matrix with a single differential form.
        assemble():
            Assemble the double divergence matrix.
    """

    def __init__(self, Λ, Q, F=None, E=None):
        """
        Initialize the double divergence matrix with a single differential form.

        Args:
            Λ (DifferentialForm): The differential form.
            Q (QuadratureRule): The quadrature rule.
            F (callable, optional): Map from logical to physical domain. Defaults to identity.
            E (jnp.ndarray, optional): Transformation matrix. Defaults to identity.
        """
        super().__init__(Λ, Λ, Q, F, E, E)

    def assemble(self):
        """Assemble the double divergence matrix."""
        def _Λ(x, i):
            return div(lambda y: self.Λ0(y, i))(x)
        Λ_ijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.n0))  # n x n_q x 1
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)
        wj = self.Q.w
        return jnp.einsum("ijk,ljk,j,j->li", Λ_ijk, Λ_ijk, 1/Jj, wj)




class LazyWeightedDoubleDivergenceMatrix(LazyMatrix):
    """
    A class representing a weighted double divergence matrix.
    The matrix entries are computed as ∫ w(x) div Λ0[i] · div Λ1[j] 1/detDF dx,
    where w(x) is a weight function.

    Methods:
        __init__(Λ, Q, weight_func, F=None, E=None):
            Initialize the weighted double divergence matrix.
        assemble():
            Assemble the weighted double divergence matrix.
    """

    def __init__(self, Λ, Q, weight_func, F=None, E=None):
        """
        Initialize the weighted double divergence matrix.

        Args:
            Λ (DifferentialForm): The differential form.
            Q (QuadratureRule): The quadrature rule.
            weight_func (callable): The weight function w(x).
            F (callable, optional): Map from logical to physical domain. Defaults to identity.
            E (jnp.ndarray, optional): Transformation matrix. Defaults to identity.
        """
        # For force operator problem, the weight function is p(x)
        self.weight_func = weight_func
        super().__init__(Λ, Λ, Q, F, E, E)

    def assemble(self):
        """Assemble the weighted double divergence matrix."""
        def _Λ(x, i):
            return div(lambda y: self.Λ0(y, i))(x)
        
        Λ_ijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.n0)) 
        
        # Evaluate weight function at quadrature points
        wj = jax.vmap(self.weight_func)(self.Q.x)  
        
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)
        qw = self.Q.w
        
        # Use 1/Jj (inverse Jacobian) 
        return jnp.einsum("ijk,ljk,j,j,j->li", Λ_ijk, Λ_ijk, wj, 1/Jj, qw)


class LazyMagneticTensionMatrix(LazyMatrix):
    """
    A class representing Term 1 of the Hessian: magnetic tension matrix.

    The matrix entries are computed as ∫ (∇ × (φ_i × B₀)) · (∇ × (φ_j × B₀)) dx,
    which represents the magnetic tension contribution to the force operator.

    Methods:
        __init__(Λ, Q, B0_func, F=None, E=None):
            Initialize the magnetic tension matrix.
        assemble():
            Assemble the magnetic tension matrix using simple, direct implementation.
    """

    def __init__(self, Λ, Q, B0_func, F=None, E=None):
        """
        Initialize the magnetic tension matrix.

        Args:
            Λ (DifferentialForm): The differential form.
            Q (QuadratureRule): The quadrature rule.
            B0_func (callable): The equilibrium B-field function B₀(x).
            F (callable, optional): Map from logical to physical domain. Defaults to identity.
            E (jnp.ndarray, optional): Transformation matrix. Defaults to identity.
        """
        self.B0_func = B0_func
        super().__init__(Λ, Λ, Q, F, E, E)

    def assemble(self):
        """Assemble the magnetic tension matrix."""
        # Compute curl(φ_i × B₀) for all basis functions at all quadrature points
        def curl_phi_cross_B(x, i):
            # Define the cross product as a vector field function
            def cross_product_field(y):
                phi_y = self.Λ0(y, i)  # Basis function at point y
                B_y = self.B0_func(y)  # B field at point y
                return jnp.cross(phi_y, B_y)
            
            return curl(cross_product_field)(x)
        
        # Vectorize over quadrature points and basis functions
        curl_vals = jax.vmap(jax.vmap(curl_phi_cross_B, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.n0))
        
        # Compute integrand: curl(φ_i × B₀) · curl(φ_j × B₀)
        integrand = jnp.einsum('ijk,ljk->ijl', curl_vals, curl_vals)
        
        # Integrate with Jacobian and weights
        J = jax.vmap(jacobian_determinant(self.F))(self.Q.x)
        result = jnp.einsum('ijl,j,j->il', integrand, J, self.Q.w)
        
        return result


class LazyCurrentDensityMatrix(LazyMatrix):
    """
    A class representing Term 3 of the Hessian: current density matrix.

    The matrix entries are computed as ∫ φ_i · [(∇ × B₀) × (∇ × (φ_j × B₀))] dx,
    which represents the current density contribution to the force operator.

    Methods:
        __init__(Λ, Q, B0_func, F=None, E=None):
            Initialize the current density matrix.
        assemble():
            Assemble the current density matrix using simple implementation.
    """

    def __init__(self, Λ, Q, B0_func, F=None, E=None):
        """
        Initialize the current density matrix.

        Args:
            Λ (DifferentialForm): The differential form.
            Q (QuadratureRule): The quadrature rule.
            B0_func (callable): The equilibrium B-field function B₀(x).
            F (callable, optional): Map from logical to physical domain. Defaults to identity.
            E (jnp.ndarray, optional): Transformation matrix. Defaults to identity.
        """
        self.B0_func = B0_func
        super().__init__(Λ, Λ, Q, F, E, E)

    def assemble(self):
        """Assemble the current density matrix using simple implementation."""
        # Step 1: Pre-compute ∇ × B₀ at all quadrature points
        curl_B0 = jax.vmap(curl(self.B0_func))(self.Q.x)
        
        # Step 2: Compute φ_i at all quadrature points
        phi_vals = jax.vmap(jax.vmap(self.Λ0, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.n0))
        
        # Step 3: Compute ∇ × (φ_j × B₀) at all quadrature points
        def compute_curl_phi_cross_B(x, j):
            # Define the cross product as a vector field function
            def cross_product_field(y):
                phi_y = self.Λ0(y, j)  # Basis function at point y
                B_y = self.B0_func(y)  # B field at point y
                return jnp.cross(phi_y, B_y)
            
            return curl(cross_product_field)(x)
        
        curl_phi_cross_B_vals = jax.vmap(jax.vmap(compute_curl_phi_cross_B, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.n0))
        
        # Step 4: Compute integrand φ_i · [(∇ × B₀) × (∇ × (φ_j × B₀))]
        def compute_integrand(k, i, j):
            phi_i = phi_vals[k, i]
            curl_B0_k = curl_B0[k]
            curl_phi_j_cross_B = curl_phi_cross_B_vals[k, j]
            cross_product = jnp.cross(curl_B0_k, curl_phi_j_cross_B)
            return jnp.dot(phi_i, cross_product)
        
        # Step 5: Vectorize integrand computation
        integrand_vals = jax.vmap(jax.vmap(jax.vmap(compute_integrand, (0, None, None)), (None, 0, None)), (None, None, 0))(
            jnp.arange(len(self.Q.x)), jnp.arange(self.n0), jnp.arange(self.n0))
        
        # Step 6: Integrate with Jacobian and weights
        J = jax.vmap(jacobian_determinant(self.F))(self.Q.x)
        result = jnp.einsum('ijk,k,k->ij', integrand_vals, J, self.Q.w)
        
        return result


class LazyPressureGradientMatrix(LazyMatrix):
    """
    A class representing the pressure gradient matrix.

    The matrix entries are computed as ∫ p(x) grad Λ0[i] · grad Λ1[j] detDF dx,
    where p(x) is a pressure function.

    Attributes:
        Inherits all attributes from LazyMatrix.
        pressure_func (callable): The pressure function p(x).

    Methods:
        __init__(Λ, Q, pressure_func, F=None, E=None):
            Initialize the pressure gradient matrix.
        assemble():
            Assemble the pressure gradient matrix.
    """

    def __init__(self, Λ, Q, pressure_func, F=None, E=None):
        """
        Initialize the pressure gradient matrix.

        Args:
            Λ (DifferentialForm): The differential form.
            Q (QuadratureRule): The quadrature rule.
            pressure_func (callable): The pressure function p(x).
            F (callable, optional): Map from logical to physical domain. Defaults to identity.
            E (jnp.ndarray, optional): Transformation matrix. Defaults to identity.
        """
        # Set pressure_func before calling super().__init__() so it's available in assemble()
        self.pressure_func = pressure_func
        super().__init__(Λ, Λ, Q, F, E, E)

    def assemble(self):
        """Assemble the pressure gradient matrix."""
        def _Λ(x, i):
            return grad(lambda y: self.Λ0(y, i))(x)
        
        Λ_ijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.n0))  
        
        # Evaluate pressure function at quadrature points
        pj = jax.vmap(self.pressure_func)(self.Q.x)
        
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)
        qw = self.Q.w
        
        return jnp.einsum("ijk,ljk,j,j,j->li", Λ_ijk, Λ_ijk, pj, Jj, qw)


class LazyPressureGradientForceMatrix(LazyMatrix):
    """
    A class representing the pressure gradient force matrix.

    The matrix entries are computed as ∫ φ_i · ∇(φ_j · ∇p) dx,
    which represents the pressure gradient force term in the Alfvén wave equation.

    Attributes:
        Inherits all attributes from LazyMatrix.
        pressure_func (callable): The pressure function p(x).

    Methods:
        __init__(Λ, Q, pressure_func, F=None, E=None):
            Initialize the pressure gradient force matrix.
        assemble():
            Assemble the pressure gradient force matrix with proper coordinate transformations.
    """

    def __init__(self, Λ, Q, pressure_func, F=None, E=None):
        """
        Initialize the pressure gradient force matrix.

        Args:
            Λ (DifferentialForm): The differential form.
            Q (QuadratureRule): The quadrature rule.
            pressure_func (callable): The pressure function p(x).
            F (callable, optional): Map from logical to physical domain. Defaults to identity.
            E (jnp.ndarray, optional): Transformation matrix. Defaults to identity.
        """
        # Set pressure_func before calling super().__init__() so it's available in assemble()
        self.pressure_func = pressure_func
        super().__init__(Λ, Λ, Q, F, E, E)

    def assemble(self):
        """Assemble the pressure gradient force matrix using simple implementation."""
        # Step 1: Compute φ_i at all quadrature points (in logical coordinates)
        phi_vals = jax.vmap(jax.vmap(self.Λ0, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.n0))
        
        # Step 2: Compute ∇(φ_j · ∇p) at all quadrature points (in logical coordinates)
        def compute_grad_phi_dot_grad_p(x, j):
            # Define the dot product as a scalar field function
            def dot_product_field(y):
                phi_y = self.Λ0(y, j)  # Basis function at point y (logical)
                grad_p_y = grad(self.pressure_func)(y)  # Gradient of pressure at point y (logical)
                return jnp.dot(phi_y, grad_p_y)
            
            # Compute gradient in logical coordinates
            return grad(dot_product_field)(x)
        
        grad_phi_dot_grad_p_vals = jax.vmap(jax.vmap(compute_grad_phi_dot_grad_p, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.n0))
        
        # Step 3: Compute integrand φ_i · ∇(φ_j · ∇p) (all in logical coordinates)
        def compute_integrand(k, i, j):
            phi_i = phi_vals[k, i]  # Basis function in logical coordinates
            grad_phi_j_dot_grad_p = grad_phi_dot_grad_p_vals[k, j]  # Gradient in logical coordinates
            return jnp.dot(phi_i, grad_phi_j_dot_grad_p)
        
        # Step 4: Vectorize integrand computation
        integrand_vals = jax.vmap(jax.vmap(jax.vmap(compute_integrand, (0, None, None)), (None, 0, None)), (None, None, 0))(
            jnp.arange(len(self.Q.x)), jnp.arange(self.n0), jnp.arange(self.n0))
        
        # Step 5: Integrate with Jacobian determinant and weights
        J = jax.vmap(jacobian_determinant(self.F))(self.Q.x)
        result = jnp.einsum('ijk,k,k->ij', integrand_vals, J, self.Q.w)
        
        return result

