from abc import abstractmethod

import jax
import jax.experimental
import jax.experimental.sparse
import jax.numpy as jnp

import numpy as np

from .DifferentialForms import DifferentialForm, DiscreteFunction
from .Quadrature import QuadratureRule
from .Utils import curl, div, grad, inv33, jacobian_determinant
from .Projectors import Projector, CrossProductProjection


__all__ = ['LazyMatrix', 'LazyMassMatrix', 'LazyDerivativeMatrix',
           'LazyProjectionMatrix', 'LazyDoubleCurlMatrix', 'LazyStiffnessMatrix', 'LazyWeightedDoubleDivergenceMatrix', 'LazyMagneticTensionMatrix', 'LazyPressureGradientForceMatrix', 'LazyCurrentDensityMatrix']


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
        self.E0 = E0
        self.E1 = E1

    def matrix(self):
        E0 = self.E0.matrix() if self.E0 is not None else jnp.eye(self.n0)
        E1 = self.E1.matrix() if self.E1 is not None else jnp.eye(self.n1)
        return E1 @ self.assemble() @ E0.T

    def __array__(self, dtype=None, copy=True):
        """Convert the assembled matrix to a NumPy array."""
        return np.array(self.matrix())

    def sparse(self, M):
        return jax.experimental.sparse.bcsr_fromdense(M, )

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
            case -1:
                return self.vector_assemble()

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
            self.Q.x, jnp.arange(self.n0))  # n x n_q x 3
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

    def vector_assemble(self):
        """Assemble the mass matrix for vector fields."""
        DF = jax.jacfwd(self.F)

        def _Λ(x, i):
            return DF(x) @ self.Λ0(x, i)
        Λijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.Λ0.n))  # n x n_q x d
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w
        return jnp.einsum("ijk,ljk,j,j->li", Λijk, Λijk, Jj, wj)


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




#------------------------------------------------------------------------------------------------
""" Force operator related matrices """
#------------------------------------------------------------------------------------------------

class LazyWeightedDoubleDivergenceMatrix(LazyMatrix):
     """
    Computes ∫ p(x) (div φ_i) (div φ_j) dx.
    """

     def __init__(self, Λ, Q, pressure_func, F=None, E=None):
        self.pressure_func = pressure_func

        # 3-form for divergence output
        ns_3form = [Λ.nr, Λ.nχ, Λ.nζ]
        ps_3form = [Λ.Λ[0].p, Λ.Λ[1].p, Λ.Λ[2].p]
        self.differential_3form = DifferentialForm(
            k=3,
            ns=ns_3form,
            ps=ps_3form,
            types=Λ.types
        )

        # Divergence matrix: 2-form → 3-form
        self.div_matrix = LazyDerivativeMatrix(
            Λ, self.differential_3form, Q, F=F, E0=E, E1=E
        )

        # Quadrature rule for 3-form space 
        self.Q3 = Q  

        super().__init__(Λ, Λ, Q, F, E, E)

     def assemble(self):
        # Compute divergence matrix 
        D = self.div_matrix.M 
        Λ3 = self.differential_3form
        Q = self.Q3
       

        # Evaluate basis at quadrature points
        Λ3_ijk = jax.vmap(jax.vmap(Λ3, (0, None)), (None, 0))(Q.x, jnp.arange(Λ3.n)) 
        wj = Q.w
        pj = jax.vmap(self.pressure_func)(Q.x).reshape(-1)  

        # Weighted mass matrix 
        M3 = jnp.einsum("ijk,ljk,j,j->il", Λ3_ijk, Λ3_ijk, pj, wj)

        # Final matrix: D^T M3 D
        M =  D.T @ M3 @ D 

        # Ensure conjugate symmetry
        return 0.5 * (M + M.conj().T) 


class LazyMagneticTensionMatrix(LazyMatrix):
    """
    Computes ∫ (∇ × (φ_i × B₀)) · (∇ × (φ_j × B₀)) dx.
    """

    def __init__(self, Λ, Q, B0_func, F=None, E=None):
    
        self.B0_func = B0_func
        
        # Create a 1-form for curl operations
        ns_1form = [Λ.nr, Λ.nχ, Λ.nζ]
        ps_1form = [Λ.Λ[0].p, Λ.Λ[1].p, Λ.Λ[2].p]
        
        self.differential_1form = DifferentialForm(
            k=1,
            ns=ns_1form,
            ps=ps_1form,
            types=Λ.types
        )

        # Create a 2-form for the curl output
        self.differential_2form = DifferentialForm(
            k=2,
            ns=ns_1form,
            ps=ps_1form,
            types=Λ.types
        )

        # Use CrossProductProjection: compute φ_i × B₀ and project onto 1-form space
        self.cross_projection = CrossProductProjection(
            self.differential_1form,  # 1-form basis space
            Λ,                        # 2-form φ basis space  
            Λ,                        # 2-form B₀ field space
            Q, F=F, En=E, Em=E, Ek=E
        )
        
        # Create curl matrix: 1-form → 2-form
        self.curl_matrix = LazyDerivativeMatrix(
            self.differential_1form,  
            self.differential_2form,  
            Q, F=F, E0=E, E1=E
        )
        
        super().__init__(Λ, Λ, Q, F, E, E)

    def get_dof(self, i):
        """
        Get the degrees of freedom for the ith basis function.
        
        Args:
            i (int): Index of the basis function.
            
        Returns:
            jnp.ndarray: Coefficient vector with 1 at position i, 0 elsewhere.
        """
        dof = jnp.zeros(self.n0)
        return dof.at[i].set(1.0)

    def compute_phi_cross_B(self, i):
        """
        Compute φ_i × B₀ using cross product projection.
        
        Args:
            i (int): Index of the basis function.
            
        Returns:
            jnp.ndarray: Coefficients of φ_i × B₀ as a 1-form.
        """
        # Get degrees of freedom for the ith basis function
        phi_dof = self.get_dof(i)
        
        # Project B₀ onto 2-form space (just in case)
        temp_projector = Projector(self.Λ0, self.Q, F=self.F, E=self.E1)
        B0_coeffs = temp_projector(self.B0_func)
        
        # Use CrossProductProjection to compute φ_i × B₀
        phi_cross_B_coeffs = self.cross_projection(phi_dof, B0_coeffs)
        
        return phi_cross_B_coeffs

    def compute_curl_phi_cross_B(self, i):
        """
        Compute ∇ × (φ_i × B₀) using LazyDerivativeMatrix.
        
        Args:
            i (int): Index of the basis function.
            
        Returns:
            jnp.ndarray: Coefficients of ∇ × (φ_i × B₀) as a 2-form.
        """
        # Get φ_i × B₀ coefficients 
        phi_cross_B_coeffs = self.compute_phi_cross_B(i)
        
        # Apply curl matrix to get ∇ × (φ_i × B₀) coefficients 
        curl_coeffs = self.curl_matrix.M @ phi_cross_B_coeffs
        
        return curl_coeffs

    def assemble(self):
      
        
        # For each pair of basis functions, compute the integral
        def compute_matrix_element(i, j):
            
            # Get curl coefficients for both basis functions
            curl_E_i_coeffs = self.compute_curl_phi_cross_B(i)
            curl_E_j_coeffs = self.compute_curl_phi_cross_B(j)
            
            # Create discrete functions for the curls
            curl_E_i_func = DiscreteFunction(curl_E_i_coeffs, self.differential_2form)
            curl_E_j_func = DiscreteFunction(curl_E_j_coeffs, self.differential_2form)
            
            # Evaluate at quadrature points
            curl_E_i_vals = jax.vmap(curl_E_i_func)(self.Q.x)  
            curl_E_j_vals = jax.vmap(curl_E_j_func)(self.Q.x) 
            
            # Compute dot product at quadrature points
            integrand = jnp.sum(curl_E_i_vals * curl_E_j_vals, axis=1) 
            
            # Integrate with appropriate weights
            Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  
            wj = self.Q.w 
            
            element = jnp.sum(integrand * (1/Jj) * wj)
            
            return element
        
        # Vectorize the computation 
        i_indices, j_indices = jnp.meshgrid(jnp.arange(self.n0), jnp.arange(self.n0), indexing='ij')
        i_indices = i_indices.flatten()
        j_indices = j_indices.flatten()
        
        # Compute all matrix elements
        matrix_elements = jax.vmap(compute_matrix_element)(i_indices, j_indices)
        
        # Reshape 
        M = matrix_elements.reshape(self.n0, self.n0)
        
        # Ensure conjugate symmetry
        return 0.5 * (M + M.conj().T) 


class LazyCurrentDensityMatrix(LazyMatrix):
    """
    Computes ∫ φ_i · [(∇ × B₀) × (∇ × (φ_j × B₀))] dx.

    """

    def __init__(self, Λ, Q, B0_func, F=None, E=None):
        self.B0_func = B0_func
        
        # Create a 1-form for curl operations
        ns_1form = [Λ.nr, Λ.nχ, Λ.nζ]
        ps_1form = [Λ.Λ[0].p, Λ.Λ[1].p, Λ.Λ[2].p]
        
        self.differential_1form = DifferentialForm(
            k=1,
            ns=ns_1form,
            ps=ps_1form,
            types=Λ.types
        )

        # Create a 2-form for the curl output
        self.differential_2form = DifferentialForm(
            k=2,
            ns=ns_1form,
            ps=ps_1form,
            types=Λ.types
        )

        # Use CrossProductProjection: compute φ_i × B₀ and project onto 1-form space
        self.one_cross_projection = CrossProductProjection(
            self.differential_1form,  # 1-form basis space
            Λ,                        # 2-form φ basis space  
            Λ,                        # 2-form B₀ field space
            Q, F=F, En=E, Em=E, Ek=E
        )
        
        # Create curl matrix: 1-form → 2-form
        self.curl_matrix = LazyDerivativeMatrix(
            self.differential_1form,  
            self.differential_2form,  
            Q, F=F, E0=E, E1=E
        )
        
        # Create cross product projection for (∇ × B₀) × (∇ × (φⱼ × B₀))
   
        self.curl_cross_projection = CrossProductProjection(
            self.differential_2form, # 2-form result space
            self.differential_2form,  # 2-form space (∇ × B₀)
            self.differential_2form,  #  2-form space (∇ × (φⱼ × B₀))
            Q, F=F, En=E, Em=E, Ek=E
        )
        
        super().__init__(Λ, Λ, Q, F, E, E)

    def get_dof(self, i):
        """
        Get the degrees of freedom for the ith basis function.
        
        Args:
            i (int): Index of the basis function.
            
        Returns:
            jnp.ndarray: Coefficient vector with 1 at position i, 0 elsewhere.
        """
        dof = jnp.zeros(self.n0)
        return dof.at[i].set(1.0)

    def compute_phi_cross_B(self, i):
        """
        Compute φ_i × B₀ using cross product projection.
        
        Args:
            i (int): Index of the basis function.
            
        Returns:
            jnp.ndarray: Coefficients of φ_i × B₀ as a 1-form.
        """
        # Get degrees of freedom for the ith basis function
        phi_dof = self.get_dof(i)
        
        # Project B₀ onto the 2-form space
        temp_projector = Projector(self.Λ0, self.Q, F=self.F, E=self.E1)
        B0_coeffs = temp_projector(self.B0_func)
        
        # Use CrossProductProjection to compute φ_i × B₀
        phi_cross_B_coeffs = self.one_cross_projection(phi_dof, B0_coeffs)
        
        return phi_cross_B_coeffs

    def compute_curl_phi_cross_B(self, i):
        """
        Compute ∇ × (φ_i × B₀) using LazyDerivativeMatrix.
        
        Args:
            i (int): Index of the basis function.
            
        Returns:
            jnp.ndarray: Coefficients of ∇ × (φ_i × B₀) as a 2-form.
        """
        # Get φ_i × B₀ coefficients 
        phi_cross_B_coeffs = self.compute_phi_cross_B(i)
        
        # Apply curl matrix to get ∇ × (φ_i × B₀) coefficients 
        curl_coeffs = self.curl_matrix.M @ phi_cross_B_coeffs
        
        return curl_coeffs

    def compute_curl_B0(self):
        """
        Compute ∇ × B₀ using LazyDerivativeMatrix.
        
        Returns:
            jnp.ndarray: Coefficients of ∇ × B₀ as a 2-form.
        """
        # Project B₀ onto the 1-form space for curl
        temp_projector_1form = Projector(self.differential_1form, self.Q, F=self.F, E=None)
        B0_1form_coeffs = temp_projector_1form(self.B0_func)
        
        # Apply curl matrix to get ∇ × B₀ coefficients
        curl_B0_coeffs = self.curl_matrix.M @ B0_1form_coeffs
        
        return curl_B0_coeffs

    def compute_cross_product_term(self, j):
        """
        Compute (∇ × B₀) × (∇ × (φ_j × B₀)) .
        
        Args:
            j (int): Index of the basis function.
            
        Returns:
            jnp.ndarray: Coefficients of the cross product as a 2-form.
        """
        # Get curl coefficients
        curl_B0_coeffs = self.compute_curl_B0()
        curl_phi_cross_B_coeffs = self.compute_curl_phi_cross_B(j)
        
        # Compute cross product projection
        cross_product_coeffs = self.curl_cross_projection(curl_B0_coeffs, curl_phi_cross_B_coeffs)
        
        return cross_product_coeffs

    def assemble(self):
        """Assemble the current density matrix using LazyDerivativeMatrix."""
        
        def compute_matrix_element(i, j):
       
            
            # Get the cross product coefficients for basis function j
            cross_product_coeffs = self.compute_cross_product_term(j)
            
            # Create discrete function for the cross product
            cross_product_func = DiscreteFunction(cross_product_coeffs, self.differential_2form)
            
            # Evaluate cross product at quadrature points
            cross_product_vals = jax.vmap(cross_product_func)(self.Q.x) 
            
            # Evaluate φ_i at quadrature points
            phi_i_vals = jax.vmap(self.Λ0, (0, None))(self.Q.x, i)  
            
            # Compute dot product φ_i · [cross_product] at quadrature points
            integrand = jnp.sum(phi_i_vals * cross_product_vals, axis=1)  
            
            # Integrate 
            Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x) 
            wj = self.Q.w  
            
            element = jnp.sum(integrand * Jj * wj)
            
            return element
        
        # Vectorize the computation
        i_indices, j_indices = jnp.meshgrid(jnp.arange(self.n0), jnp.arange(self.n0), indexing='ij')
        i_indices = i_indices.flatten()
        j_indices = j_indices.flatten()
        
        # Compute all matrix elements
        matrix_elements = jax.vmap(compute_matrix_element)(i_indices, j_indices)
        
        # Reshape 
        M = matrix_elements.reshape(self.n0, self.n0)
        
        # Ensure conjugate symmetry
        return 0.5 * (M + M.conj().T) 



class LazyPressureGradientForceMatrix(LazyMatrix):
    """
    Computes Term 4: ∫ φ_i · ∇(φ_j · ∇p) dx via integration by parts: -∫ (∇·φ_i)(φ_j · ∇p) dx.
    """

    def __init__(self, Λ, Q, pressure_func, F=None, E=None):
        self.pressure_func = pressure_func

        # 3-form for divergence output
        ns_3form = [Λ.nr, Λ.nχ, Λ.nζ]
        ps_3form = [Λ.Λ[0].p, Λ.Λ[1].p, Λ.Λ[2].p]
        self.differential_3form = DifferentialForm(
            k=3,
            ns=ns_3form,
            ps=ps_3form,
            types=Λ.types
        )

        # 1-form for gradient output
        self.differential_1form = DifferentialForm(
            k=1,
            ns=ns_3form,
            ps=ps_3form,
            types=Λ.types
        )

        # 0-form for gradient input
        ns_0form = [Λ.nr, Λ.nχ, Λ.nζ]
        ps_0form = [Λ.Λ[0].p, Λ.Λ[1].p, Λ.Λ[2].p]

        self.differential_0form = DifferentialForm(
            k=0,
            ns=ns_0form,
            ps=ps_0form,
            types=Λ.types
        )

        # Divergence matrix
        self.div_matrix = LazyDerivativeMatrix(
            Λ, self.differential_3form, Q, 
            F=F, E0=E, E1=E
        )

        # Gradient matrix
        self.grad_matrix = LazyDerivativeMatrix(
            self.differential_0form, self.differential_1form, 
            Q, F=F, E0=E, E1=E
        )
        
        # Convert 2-form φ to 1-form 
        self.projection_2to1 = Projector(self.differential_1form, self.Q, F=self.F, E=None)

        super().__init__(Λ, Λ, Q, F, E, E)

    def assemble(self):
        
        # Project pressure onto 0-form space 
        temp_projector_0form = Projector(self.differential_0form, self.Q, F=self.F, E=None)
        p_coeffs = temp_projector_0form(self.pressure_func) 
        
        # Compute ∇p 
        grad_p_coeffs = self.grad_matrix.M @ p_coeffs  
        
        # Create discrete function for ∇p
        grad_p_func = DiscreteFunction(grad_p_coeffs, self.differential_1form)
        
        # For each pair of basis functions, compute the integral
        def compute_matrix_element(i, j):
            
            # Get divergence of φ_i 
            div_phi_i_coeffs = self.div_matrix.M @ self.get_dof(i) 
            div_phi_i_func = DiscreteFunction(div_phi_i_coeffs, self.differential_3form)
            
            # Project φ_j from 2-form to 1-form for dot product with ∇p 
            def phi_j_2form_func(x):
                return self.Λ0(x, j)
            phi_j_1form_coeffs = self.projection_2to1(phi_j_2form_func)  
            phi_j_1form_func = DiscreteFunction(phi_j_1form_coeffs, self.differential_1form)
            phi_j_vals = jax.vmap(phi_j_1form_func)(self.Q.x)  
            
            # Evaluate ∇p at quadrature points
            grad_p_vals = jax.vmap(grad_p_func)(self.Q.x) 
            
            # Compute φ_j · ∇p at quadrature points 
            phi_dot_grad_p = jnp.sum(phi_j_vals * grad_p_vals, axis=1)  
            
            # Evaluate ∇·φ_i at quadrature points 
            div_phi_i_vals = jax.vmap(div_phi_i_func)(self.Q.x)  
            div_phi_i_vals = div_phi_i_vals.reshape(-1) 
            
            # Compute integrand: -(∇·φ_i)(φ_j · ∇p)
            integrand = -div_phi_i_vals * phi_dot_grad_p  
            
            # Integrate with appropriate weights
            Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)
            wj = self.Q.w  
            
            # ∫ -(∇·φ_i)(φ_j · ∇p) * (1/J) dx
            element = jnp.sum(integrand * (1/Jj) * wj)
            
            return element
        
        # Vectorize the computation 
        i_indices, j_indices = jnp.meshgrid(jnp.arange(self.n0), jnp.arange(self.n0), indexing='ij')
        i_indices = i_indices.flatten()
        j_indices = j_indices.flatten()
        
        # Compute all matrix elements
        matrix_elements = jax.vmap(compute_matrix_element)(i_indices, j_indices)
        
        # Reshape to matrix form
        M = matrix_elements.reshape(self.n0, self.n0)
        
        # Ensure conjugate symmetry
        return 0.5 * (M + M.conj().T)
    
    def get_dof(self, i):
        """
        Get the degrees of freedom for the ith basis function.
        
        Args:
            i (int): Index of the basis function.
            
        Returns:
            jnp.ndarray: Coefficient vector with 1 at position i, 0 elsewhere.
        """
        dof = jnp.zeros(self.n0)
        return dof.at[i].set(1.0) 