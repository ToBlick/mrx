from abc import abstractmethod

import jax
import jax.numpy as jnp

import numpy as np

from .DifferentialForms import DifferentialForm, DiscreteFunction
from .Quadrature import QuadratureRule
from .Utils import curl, div, grad, inv33, jacobian_determinant
from .Projectors import Projector, CrossProductProjection


__all__ = ['LazyMatrix', 'LazyMassMatrix', 'LazyDerivativeMatrix',
           'LazyProjectionMatrix', 'LazyDoubleCurlMatrix', 'LazyStiffnessMatrix', 'LazyWeightedDoubleDivergenceMatrix', 'LazyWeightedDoubleDivergenceMatrixV2', 'LazyMagneticTensionMatrix', 'LazyMagneticTensionMatrixV2', 'LazyMagneticTensionMatrixV3', 'LazyPressureGradientForceMatrix', 'LazyPressureGradientForceMatrixV2', 'LazyCurrentDensityMatrix', 'LazyCurrentDensityMatrixV2']


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




#------------------------------------------------------------------------------------------------
""" Hessian-related matrices respecting: 

1. GRADIENT TRANSFORMATION:
   ∇f = DF^(-T) ∇̂f₀
   where f₀(x̂) = f(F(x̂)) and DF is the Jacobian 

2. CURL TRANSFORMATION:
   ∇×V = (1/|DF|) DF ∇̂×V₁
   where V₁(x̂) = DF^(T) V(F(x̂))

3. DIVERGENCE TRANSFORMATION:
   ∇·V = (1/√g) ∇̂·V₂
   where V₂(x̂) = √g DF^{-1}V(F(x̂)) and √g = |DF| """

#------------------------------------------------------------------------------------------------


class LazyWeightedDoubleDivergenceMatrix(LazyMatrix):
    """
    A class representing a weighted double divergence matrix.
    The matrix entries are computed as ∫ w(x) div Λ0[i] · div Λ1[j] dx,
    where w(x) is a weight function. The basis functions are now 2-forms, so we can compute divergence directly. 

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
            Λ (DifferentialForm): The 1-form differential form (input basis).
            Q (QuadratureRule): The quadrature rule.
            weight_func (callable): The weight function w(x).
            F (callable, optional): Map from logical to physical domain. Defaults to identity.
            E (jnp.ndarray, optional): Transformation matrix. Defaults to identity.
        """
        self.weight_func = weight_func
        super().__init__(Λ, Λ, Q, F, E, E)

    def assemble(self):
        """Assemble the weighted double divergence matrix."""
        
        def _Λ(x, i):
            return div(lambda y: self.Λ0(y, i))(x)
        
        Λ_ijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.n0))
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)
        wj = jax.vmap(self.weight_func)(self.Q.x)
        qw = self.Q.w

        return jnp.einsum("ijk,ljk,j,j,j->il", Λ_ijk, Λ_ijk, wj, 1/Jj, qw)


class LazyWeightedDoubleDivergenceMatrixV2(LazyMatrix):
     """
    V2: Uses LazyDerivativeMatrix for divergence and incorporates p(x) in the 3-form mass matrix.
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

        # Quadrature rule for 3-form space (for mass matrix assembly)
        self.Q3 = Q  # Use the same quadrature as input

        super().__init__(Λ, Λ, Q, F, E, E)

     def assemble(self):
        # Compute divergence matrix (3-form coefficients for each 2-form basis)
        D = self.div_matrix.M  # shape: (n3, n2)
        # Project pressure onto 3-form space
        temp_projector = Projector(self.differential_3form, self.Q3, F=self.F, E=None)
        p_coeffs = temp_projector(self.pressure_func)  # shape: (n3,)

        # Weighted mass matrix in 3-form space: M_ij = ∫ p(x) Λ3[i](x) Λ3[j](x) dx
        Λ3 = self.differential_3form
        Q = self.Q3
        ns = Λ3.ns

        # Evaluate basis at quadrature points
        Λ3_ijk = jax.vmap(jax.vmap(Λ3, (0, None)), (None, 0))(Q.x, jnp.arange(Λ3.n))  # n x n_q x 1
        wj = Q.w
        pj = jax.vmap(self.pressure_func)(Q.x).reshape(-1)  # (n_q,)

        # Weighted mass matrix in 3-form space
        M3 = jnp.einsum("ijk,ljk,j,j->il", Λ3_ijk, Λ3_ijk, pj, wj)

        # Final matrix: D^T M3 D
        return D.T @ M3 @ D 

class LazyMagneticTensionMatrix(LazyMatrix):
    """
    A class representing Term 1 of the Hessian: magnetic tension matrix.

    The matrix entries are computed as ∫ (∇ × (φ_i × B₀)) · (∇ × (φ_j × B₀)) dx,
    which represents the magnetic tension contribution to the force operator. We use CrossProductProjection.

    Methods:
        __init__(Λ, Q, B0_func, F=None, E=None, k_y=None, k_z=None):
            Initialize the magnetic tension matrix.
        assemble():
            Assemble the magnetic tension matrix using projection.
    """

    def __init__(self, Λ, Q, B0_func, F=None, E=None, k_y=None, k_z=None):
        """
        Initialize the magnetic tension matrix.

        Args:
            Λ (DifferentialForm): The differential form (2-form) representing φ basis functions.
            Q (QuadratureRule): The quadrature rule.
            B0_func (callable): The equilibrium B-field function B₀(x) (2-form).
            F (callable, optional): Map from logical to physical domain. Defaults to identity.
            E (jnp.ndarray, optional): Transformation matrix. Defaults to identity.
            k_y (float, optional): Wave number in y-direction. If None, will be inferred from context.
            k_z (float, optional): Wave number in z-direction. If None, will be inferred from context.
        """
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

        # Use CrossProductProjection: compute φ_i × B₀ and project onto 1-form space
        self.cross_projection = CrossProductProjection(
            self.differential_1form,  # 1-form basis space
            Λ,                        # 2-form φ basis space  
            Λ,                        # 2-form B₀ field space
            Q, F=F, En=E, Em=E, Ek=E
        )
        
        super().__init__(Λ, Λ, Q, F, E, E)

    def assemble(self):
        """Assemble the magnetic tension matrix using CrossProductProjection."""
        
        def DF(x):
            return jax.jacfwd(self.F)(x)
    
        # Project B₀ onto the 2-form space
        temporary_projector = Projector(self.Λ0, self.Q, F=self.F, E=self.E1)
        B0_coeffs = temporary_projector(self.B0_func)
        
        def _curl_E(j):
            """Compute ∇ × (φ_j × B₀) using CrossProductProjection"""
            phi_j = jnp.zeros(self.n0)
            phi_j = phi_j.at[j].set(1.0) # Coefficient
            
            # Use CrossProductProjection to compute φ_j × B₀
            phi_cross_B_coeffs = self.cross_projection(phi_j, B0_coeffs)
            
            phi_cross_B_func = DiscreteFunction(phi_cross_B_coeffs, self.differential_1form)
            
            def curl_phi_cross_B(x):
                return DF(x) @ curl(lambda y: phi_cross_B_func(y))(x)
            
            return curl_phi_cross_B

        curl_E_vals = jax.vmap(jax.vmap(lambda x, j: _curl_E(j)(x), (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.n0))
        
        curl_E_vals = curl_E_vals.transpose(1, 0, 2)
        
        Integrand = jnp.einsum("qia,qja->qij", curl_E_vals, curl_E_vals)
        
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  
        wj = self.Q.w 
        
        # Integrate over quadrature points
        M = jnp.einsum("qij,q,q->ij", Integrand, 1/Jj, wj)
        return 0.5*(M+M.T)


class LazyCurrentDensityMatrix(LazyMatrix):
    """
    A class representing Term 3 of the Hessian: current density matrix.

    The matrix entries are computed as ∫ φ_i · [(∇ × B₀) × (∇ × (φ_j × B₀))] dx,
    which represents the current density contribution to the force operator. We use CrossProductProjection.
    """

    def __init__(self, Λ, Q, B0_func, F=None, E=None, k_y=None, k_z=None):
        """
        Initialize the current density matrix.

        Args:
            Λ (DifferentialForm): The differential form (2-form).
            Q (QuadratureRule): The quadrature rule.
            B0_func (callable): The equilibrium B-field function B₀(x) (2-form).
            F (callable, optional): Map from logical to physical domain. Defaults to identity.
            E (jnp.ndarray, optional): Transformation matrix. Defaults to identity.
            k_y (float, optional): Wave number in y-direction. If None, will be inferred from context.
            k_z (float, optional): Wave number in z-direction. If None, will be inferred from context.
        """
        self.B0_func = B0_func

        ns_1form = [Λ.nr, Λ.nχ, Λ.nζ]
        ps_1form = [Λ.Λ[0].p, Λ.Λ[1].p, Λ.Λ[2].p]
        self.differential_1form = DifferentialForm(
            k=1,
            ns=ns_1form,
            ps=ps_1form,
            types=Λ.types
        )

        # Cross product projector
        self.cross_projection = CrossProductProjection(
            self.differential_1form,  # 1-form basis space
            Λ,                        # 2-form φ basis space  
            Λ,                        # 2-form B₀ field space
            Q, F=F, En=E, Em=E, Ek=E
       )

        super().__init__(Λ, Λ, Q, F, E, E)

    def assemble(self):
        """Assemble the current density matrix using CrossProductProjection."""
        
        def DF(x):
            return jax.jacfwd(self.F)(x)

        # Project B0 onto the 2-form space
        temp_projector = Projector(self.Λ0, self.Q, F=self.F, E=self.E1)
        B0_coeffs = temp_projector(self.B0_func)

        # Project B0 onto the 1-form space for curl
        temp_projector_1form = Projector(self.differential_1form, self.Q, F=self.F, E=None)
        B0_1form_coeffs = temp_projector_1form(self.B0_func)
        B0_1form_func = DiscreteFunction(B0_1form_coeffs, self.differential_1form)

        def curl_B0(x):
            return DF(x) @ curl(lambda y: B0_1form_func(y))(x)

        # For each basis function, compute curl(phi_j x B0)
        def _curl_E(j):
            phi_j = jnp.zeros(self.n0)
            phi_j = phi_j.at[j].set(1.0)
            phi_cross_B_coeffs = self.cross_projection(phi_j, B0_coeffs)
            phi_cross_B_func = DiscreteFunction(phi_cross_B_coeffs, self.differential_1form)
            def curl_phi_cross_B(x):
                return DF(x) @ curl(lambda y: phi_cross_B_func(y))(x)
            return curl_phi_cross_B

        # Evaluate at quadrature points
        curl_B0_vals = jax.vmap(curl_B0)(self.Q.x) 
        curl_E_vals = jax.vmap(jax.vmap(lambda x, j: _curl_E(j)(x), (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.n0)) 

        curl_E_vals = curl_E_vals.transpose(1, 0, 2)
        
        cross_vals = jax.vmap(lambda cb, ce: jnp.cross(cb, ce))(curl_B0_vals[:, None, :], curl_E_vals)
        Λ_i = jax.vmap(jax.vmap(self.Λ0, (0, None)), (None, 0))(self.Q.x, jnp.arange(self.n0))

        Λ_i = Λ_i.transpose(1, 0, 2)
        Integrand = jnp.einsum("qik,qjk->qij", Λ_i, cross_vals) # Final dot product



        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)
        wj = self.Q.w

        # Integrate over quadrature points
        M = jnp.einsum("qij,q,q->ij", Integrand, wj, Jj)
        return 0.5 * (M + M.T)
      
    

class LazyPressureGradientForceMatrix(LazyMatrix):
    """
    A class representing Term 4 of the Hessian: pressure gradient force matrix.

    The matrix entries are computed as ∫ φ_i · ∇(φ_j · ∇p) dx,
    which represents the pressure gradient force contribution to the force operator. We simplify via integration by parts (ignoring boundary for now) and get 
    -∫ (∇·φ_i)(φ_j · ∇p) dx
    
    Methods:
        __init__(Λ, Q, pressure_func, F=None, E=None):
            Initialize the pressure gradient force matrix.
        assemble():
            Assemble the pressure gradient force matrix.
    """

    def __init__(self, Λ, Q, pressure_func, F=None, E=None):
        """
        Initialize the pressure gradient force matrix.

        Args:
            Λ (DifferentialForm): The differential form (2-form).
            Q (QuadratureRule): The quadrature rule.
            pressure_func (callable): The pressure function p(x).
            F (callable, optional): Map from logical to physical domain. Defaults to identity.
            E (jnp.ndarray, optional): Transformation matrix. Defaults to identity.
        """
        self.pressure_func = pressure_func
        super().__init__(Λ, Λ, Q, F, E, E)

    def assemble(self):
        """Assemble the pressure gradient force matrix."""
        
        def DF(x):
            return jax.jacfwd(self.F)(x)
        
        def _Λ_i(x, i):
            return div(lambda y: self.Λ0(y, i))(x)
        
        def _Λ_j(x, j):
            return jnp.dot(self.Λ0(x, j), inv33(DF(x)).T @ grad(lambda y: self.pressure_func(y))(x))

        def _Integrand(x, i, j):
            return -_Λ_i(x, i) * _Λ_j(x, j)
        
        integrand_vals = jax.vmap(jax.vmap(jax.vmap(_Integrand, (0, None, None)), (None, 0, None)), (None, None, 0))(
            self.Q.x, jnp.arange(self.n0), jnp.arange(self.n0))
        
        # Transpose to get quadrature dimension first: (n_q, n0, n0)
        integrand_vals = jnp.squeeze(integrand_vals).transpose(2, 0, 1)
        
        wj = self.Q.w
        
        # Integrate over quadrature points using einsum
        M = jnp.einsum("qij,q->ij", integrand_vals, wj)
        
        return 0.5*(M + M.T)
    

class LazyMagneticTensionMatrixV2(LazyMatrix):
    """
    A cleaner implementation of the magnetic tension matrix using LazyDerivativeMatrix.
    
    This class computes ∫ (∇ × (φ_i × B₀)) · (∇ × (φ_j × B₀)) dx
    using the following approach:
    1. For each basis function φ_i, compute E_i = φ_i × B₀ using cross product projection
    2. Use LazyDerivativeMatrix to compute curl(E_i) 
    3. Integrate curl(E_i) · curl(E_j)
    
    Attributes:
        Inherits all attributes from LazyMatrix.
        B0_func (callable): The equilibrium B-field function.
        cross_projection (CrossProductProjection): For computing φ × B₀.
        differential_1form (DifferentialForm): 1-form space for curl operations.
        curl_matrix (LazyDerivativeMatrix): Matrix for computing curls of 1-forms.
    """

    def __init__(self, Λ, Q, B0_func, F=None, E=None):
        """
        Initialize the magnetic tension matrix.

        Args:
            Λ (DifferentialForm): The 2-form differential form (φ basis functions).
            Q (QuadratureRule): The quadrature rule.
            B0_func (callable): The equilibrium B-field function B₀(x) (2-form).
            F (callable, optional): Map from logical to physical domain. Defaults to identity.
            E (jnp.ndarray, optional): Transformation matrix. Defaults to identity.
        """
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
        
        # Project B₀ onto  2-form space
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
        """Assemble the magnetic tension matrix using LazyDerivativeMatrix."""
        
        # For each pair of basis functions, compute the integral
        def compute_matrix_element(i, j):
            """Compute the (i,j) element of the magnetic tension matrix."""
            
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
        
        # Vectorize the computation over all pairs
        i_indices, j_indices = jnp.meshgrid(jnp.arange(self.n0), jnp.arange(self.n0), indexing='ij')
        i_indices = i_indices.flatten()
        j_indices = j_indices.flatten()
        
        # Compute all matrix elements
        matrix_elements = jax.vmap(compute_matrix_element)(i_indices, j_indices)
        
        # Reshape 
        M = matrix_elements.reshape(self.n0, self.n0)
        
        # Ensure symmetry
        return 0.5 * (M + M.conj().T) 


class LazyCurrentDensityMatrixV2(LazyMatrix):
    """
    A cleaner implementation of the current density matrix using LazyDerivativeMatrix.
    
    This class computes ∫ φ_i · [(∇ × B₀) × (∇ × (φ_j × B₀))] dx
    using the following approach:
    1. For each basis function φ_j, compute E_j = φ_j × B₀ using cross product projection
    2. Use LazyDerivativeMatrix to compute curl(E_j)
    3. Use LazyDerivativeMatrix to compute curl(B₀)
    4. Take cross product: curl(B₀) × curl(E_j)
    5. Integrate φ_i · [curl(B₀) × curl(E_j)]
    
    Attributes:
        Inherits all attributes from LazyMatrix.
        B0_func (callable): The equilibrium B-field function.
        cross_projection (CrossProductProjection): For computing φ × B₀.
        differential_1form (DifferentialForm): 1-form space for curl operations.
        curl_matrix (LazyDerivativeMatrix): Matrix for computing curls of 1-forms.
    """

    def __init__(self, Λ, Q, B0_func, F=None, E=None):
        """
        Initialize the current density matrix.

        Args:
            Λ (DifferentialForm): The 2-form differential form (φ basis functions).
            Q (QuadratureRule): The quadrature rule.
            B0_func (callable): The equilibrium B-field function B₀(x) (2-form).
            F (callable, optional): Map from logical to physical domain. Defaults to identity.
            E (jnp.ndarray, optional): Transformation matrix. Defaults to identity.
        """
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
            self.differential_1form,  # Input: 1-form
            self.differential_2form,  # Output: 2-form
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
        
        # Project B₀ onto the 2-form space
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
        # Get φ_i × B₀ coefficients (1-form)
        phi_cross_B_coeffs = self.compute_phi_cross_B(i)
        
        # Apply curl matrix to get ∇ × (φ_i × B₀) coefficients (2-form)
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
        Compute (∇ × B₀) × (∇ × (φ_j × B₀)).
        
        Args:
            j (int): Index of the basis function.
            
        Returns:
            callable: Function that computes the cross product at any point x.
        """
        # Get curl coefficients
        curl_B0_coeffs = self.compute_curl_B0()
        curl_phi_cross_B_coeffs = self.compute_curl_phi_cross_B(j)
        
        # Create discrete functions for the curls
        curl_B0_func = DiscreteFunction(curl_B0_coeffs, self.differential_2form)
        curl_phi_cross_B_func = DiscreteFunction(curl_phi_cross_B_coeffs, self.differential_2form)
        
        def cross_product_term(x):
            """Compute (∇ × B₀) × (∇ × (φ_j × B₀)) at point x."""
            curl_B0_val = curl_B0_func(x)
            curl_phi_cross_B_val = curl_phi_cross_B_func(x)
            return jnp.cross(curl_B0_val, curl_phi_cross_B_val)
        
        return cross_product_term

    def assemble(self):
        """Assemble the current density matrix using LazyDerivativeMatrix."""
        
        def compute_matrix_element(i, j):
            """Compute the (i,j) element of the current density matrix."""
            
            # Get the cross product term for basis function j
            cross_product_func = self.compute_cross_product_term(j)
            
            # Evaluate cross product at quadrature points
            cross_product_vals = jax.vmap(cross_product_func)(self.Q.x)  # shape: (n_q, 3)
            
            # Evaluate φ_i at quadrature points
            phi_i_vals = jax.vmap(self.Λ0, (0, None))(self.Q.x, i)  # shape: (n_q, 3)
            
            # Compute dot product φ_i · [cross_product] at quadrature points
            integrand = jnp.sum(phi_i_vals * cross_product_vals, axis=1)  
            
            # Integrate 
            Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x) 
            wj = self.Q.w  
            
       
            element = jnp.sum(integrand * Jj * wj)
            
            return element
        
        # Vectorize the computation over all pairs
        i_indices, j_indices = jnp.meshgrid(jnp.arange(self.n0), jnp.arange(self.n0), indexing='ij')
        i_indices = i_indices.flatten()
        j_indices = j_indices.flatten()
        
        # Compute all matrix elements
        matrix_elements = jax.vmap(compute_matrix_element)(i_indices, j_indices)
        
        # Reshape 
        M = matrix_elements.reshape(self.n0, self.n0)
        
        # Ensure symmetry
        return 0.5 * (M + M.conj().T) 


class LazyMagneticTensionMatrixV3(LazyMatrix):
    """
    A cleaner implementation of the magnetic tension matrix using LazyDerivativeMatrix.
    
    This class computes ∫ (∇ × (φ_i × B₀)) · (∇ × (φ_j × B₀)) dx
    using the following approach:
    1. For each basis function φ_i, compute E_i = φ_i × B₀ using cross product projection
    2. Use LazyDerivativeMatrix to compute curl(E_i) 
    3. Integrate curl(E_i) · curl(E_j)
    
    Attributes:
        Inherits all attributes from LazyMatrix.
        B0_func (callable): The equilibrium B-field function.
        cross_projection (CrossProductProjection): For computing φ × B₀.
        differential_1form (DifferentialForm): 1-form space for curl operations.
        curl_matrix (LazyDerivativeMatrix): Matrix for computing curls of 1-forms.
    """

    def __init__(self, Λ, Q, B0_func, F=None, E=None):
        """
        Initialize the magnetic tension matrix.

        Args:
            Λ (DifferentialForm): The 2-form differential form (φ basis functions).
            Q (QuadratureRule): The quadrature rule.
            B0_func (callable): The equilibrium B-field function B₀(x) (2-form).
            F (callable, optional): Map from logical to physical domain. Defaults to identity.
            E (jnp.ndarray, optional): Transformation matrix. Defaults to identity.
        """
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

        # Compute φ_i × B₀ with CrossProduct projection and project onto 1-form space
        self.cross_projection = CrossProductProjection(
            self.differential_1form,  # 1-form basis space
            Λ,                        # 2-form φ basis space  
            Λ,                        # 2-form B₀ field space
            Q, F=F, En=E, Em=E, Ek=E
        )
        
        # Create curl matrix
        self.curl_matrix = LazyDerivativeMatrix(
            self.differential_1form,  # Input: 1-form
            self.differential_2form,  # Output: 2-form
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
        # Get dof for the ith basis function
        phi_dof = self.get_dof(i)
        
        # Project B₀ onto  2-form space
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
        
        # Multiply by curl matrix to get ∇ × (φ_i × B₀) coefficients 
        curl_coeffs = self.curl_matrix.M @ phi_cross_B_coeffs
        
        return curl_coeffs

    def assemble(self):
        """Assemble the magnetic tension matrix using LazyDerivativeMatrix."""
        
        # For each pair of basis functions, compute the integral
        def compute_matrix_element(i, j):
            """Compute the (i,j) element of the magnetic tension matrix."""
            
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
            
            # Integrate
            Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  
            wj = self.Q.w  
            
            # ∫ curl(E_i) · curl(E_j) * (1/J) * w dx
            element = jnp.sum(integrand * (1/Jj) * wj)
            
            return element
        
        # Vectorize the computation over all pairs
        i_indices, j_indices = jnp.meshgrid(jnp.arange(self.n0), jnp.arange(self.n0), indexing='ij')
        i_indices = i_indices.flatten()
        j_indices = j_indices.flatten()
        
        # Compute all matrix elements
        matrix_elements = jax.vmap(compute_matrix_element)(i_indices, j_indices)
        
        # Reshape to matrix form
        M = matrix_elements.reshape(self.n0, self.n0)
        
        # Ensure symmetry
        return 0.5 * (M + M.conj().T) 


class LazyPressureGradientForceMatrixV2(LazyMatrix):
    """
    V2: Uses LazyDerivativeMatrix for divergence and gradient operations.
    Computes Term 4: ∫ φ_i · ∇(φ_j · ∇p) dx
    Via integration by parts: -∫ (∇·φ_i)(φ_j · ∇p) dx
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

        super().__init__(Λ, Λ, Q, F, E, E)

    def assemble(self):
        """
        Assemble Term 4: -∫ (∇·φ_i)(φ_j · ∇p) dx (simplified with integration by parts)
        
        This requires:
        1. Computing ∇·φ_i for each basis function (divergence)
        2. Computing ∇p (gradient of pressure)
        3. Computing φ_j · ∇p (dot product)
        4. Integrating the product
        """
        
        # Project pressure onto 0-form space 
        temp_projector_0form = Projector(self.differential_0form, self.Q, F=self.F, E=None)
        p_coeffs = temp_projector_0form(self.pressure_func) 
        
        # Compute ∇p 
        grad_p_coeffs = self.grad_matrix.M @ p_coeffs  
        
        # Create discrete function for ∇p
        grad_p_func = DiscreteFunction(grad_p_coeffs, self.differential_1form)
        
        # For each pair of basis functions, compute the integral
        def compute_matrix_element(i, j):
            """Compute the (i,j) element of the pressure gradient force matrix."""
            
            # Get divergence of φ_i 
            div_phi_i_coeffs = self.div_matrix.M @ self.get_dof(i) 
            div_phi_i_func = DiscreteFunction(div_phi_i_coeffs, self.differential_3form)
            
            # Evaluate φ_j at quadrature points 
            phi_j_vals = jax.vmap(self.Λ0, (0, None))(self.Q.x, j) 
            
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
        
        # Vectorize the computation over all pairs
        i_indices, j_indices = jnp.meshgrid(jnp.arange(self.n0), jnp.arange(self.n0), indexing='ij')
        i_indices = i_indices.flatten()
        j_indices = j_indices.flatten()
        
        # Compute all matrix elements
        matrix_elements = jax.vmap(compute_matrix_element)(i_indices, j_indices)
        
        # Reshape to matrix form
        M = matrix_elements.reshape(self.n0, self.n0)
        
        # Ensure symmetry
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