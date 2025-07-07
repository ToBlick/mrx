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

    Attributes:d
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
        def compute_div_for_basis_i(i):
            """Compute divergence for a specific basis function i."""
            def phi_i_func(y):
                return self.Λ0(y, i)
            
            def div_at_point(x):
                logical_div = div(phi_i_func)(x)
                J = jacobian_determinant(self.F)(x)
                # Ensure we return a scalar by taking the sum if it's a vector
                if jnp.ndim(logical_div) > 0:
                    logical_div = jnp.sum(logical_div)
                return (1/J) * logical_div
            
            return jax.vmap(div_at_point)(self.Q.x)
        
        # Use explicit loop to avoid tracer issues with basis function indices
        Λ_vals = []
        for i in range(self.n0):
            Λ_vals.append(compute_div_for_basis_i(i))
        Λ_vals = jnp.stack(Λ_vals)
        
        # Step 3: Evaluate weight function at quadrature points
        wj = jax.vmap(self.weight_func)(self.Q.x)  
        
        # Step 4: Jacobian determinant and quadrature weights  
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)
        qw = self.Q.w
        
        # Step 5: Use J for integration
        return jnp.einsum("ij,lj,j,j,j->il", Λ_vals, Λ_vals, wj, Jj, qw)

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
    The matrix entries are computed as ∫ w(x) div Λ0[i] · div Λ1[j]  dx,
    where w(x) is a weight function.
    
    This optimized implementation computes divergence directly on 1-form basis functions
    with proper coordinate transformations.

    Methods:
        __init__(Λ, Q, weight_func, F=None, E=None):
            Initialize the weighted double divergence matrix.
        assemble():
            Assemble the weighted double divergence matrix efficiently.
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
        """Assemble the weighted double divergence matrix with ultra-fast implementation."""
        
        def compute_div_for_basis_i(i):
            """Compute divergence for a specific basis function i."""
            def div_at_point(x):
                # Compute divergence explicitly for 1-form
                # For 1-form φ = [φₓ, φᵧ, φᵤ], div(φ) = ∂φₓ/∂x + ∂φᵧ/∂y + ∂φᵤ/∂z
                
                def phi_i_component(x_in, comp):
                    """Get component comp of basis function i"""
                    phi_val = self.Λ0(x_in, i)
                    return phi_val[comp]
                
                # Compute partial derivatives
                dphi_dx = jax.grad(lambda x_in: phi_i_component(x_in, 0))(x)
                dphi_dy = jax.grad(lambda x_in: phi_i_component(x_in, 1))(x) 
                dphi_dz = jax.grad(lambda x_in: phi_i_component(x_in, 2))(x)
                
                # Divergence in logical coordinates
                logical_div = dphi_dx + dphi_dy + dphi_dz
                
                # Apply coordinate transformation
                J = jacobian_determinant(self.F)(x)
                return (1/J) * logical_div
            
            return jax.vmap(div_at_point)(self.Q.x)
        
        print("Computing divergences with explicit implementation...")
        
        # Use explicit vectorization to avoid tracer issues  
        div_vals = jax.vmap(compute_div_for_basis_i)(jnp.arange(self.n0))
        
        # Evaluate weight function at quadrature points
        weight_vals = jax.vmap(self.weight_func)(self.Q.x)  
        
        # Jacobian determinant and quadrature weights  
        J_vals = jax.vmap(jacobian_determinant(self.F))(self.Q.x)
        quad_weights = self.Q.w
        
        # Assemble matrix efficiently
        return jnp.einsum("ij,kj,j,j,j->ik", div_vals, div_vals, weight_vals, (1/J_vals**2), quad_weights)


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

        DF = jax.jacfwd(self.F)
        
        def curl_phi_cross_B(k, i):
            # k is the quadrature point index, i is the basis function index
            x_log = self.Q.x[k]
            
            # Phi and B0 both in logical coordinates, take cross product
            def phi_cross_B_func(x_log):
                phi_val = self.Λ0(x_log, i)
                B0_val = self.B0_func(x_log)
                return jnp.cross(phi_val, B0_val)
            
            # Get curl, and apply correct transformation
            DF_x = DF(x_log)  # Evaluate DF at the specific point
            return DF_x @ curl(phi_cross_B_func)(x_log)
        
        # Vectorize using quadrature point indices
        curl_vals = jax.vmap(jax.vmap(curl_phi_cross_B, (0, None)), (None, 0))(
            jnp.arange(len(self.Q.x)), jnp.arange(self.n0))
        
        # Compute integrand: curl(φ_i × B₀) · curl(φ_j × B₀)
        integrand = jnp.einsum('ijk,ljk->ijl', curl_vals, curl_vals)
        
        # Integrate
        J = jax.vmap(jacobian_determinant(self.F))(self.Q.x)
        result = jnp.einsum('ijl,j,j->il', integrand, 1/(J**2), self.Q.w)
        
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
            Assemble the current density matrix using logical coordinates.
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
        """Assemble the current density matrix using logical coordinates."""
        
        # Step 1: Pre-compute ∇ × B₀ at all logical quadrature points
        DF = jax.jacfwd(self.F)
        J = jax.vmap(jacobian_determinant(self.F))(self.Q.x)
        
        # Fix: Evaluate DF at each quadrature point
        def compute_curl_B0(k):
            x_log = self.Q.x[k]
            DF_x = DF(x_log)
            J_x = J[k]
            return (1/J_x) * DF_x @ curl(self.B0_func)(x_log)
        
        curl_B0 = jax.vmap(compute_curl_B0)(jnp.arange(len(self.Q.x)))
        
        # Step 2: Compute φ_i values at all quadrature points 
        def _phi(k, i):
            # k is quadrature point index, i is basis function index
            return self.Λ0(self.Q.x[k], i)
        
        phi_vals = jax.vmap(jax.vmap(_phi, (0, None)), (None, 0))(
            jnp.arange(len(self.Q.x)), jnp.arange(self.n0))
        
        # Step 3: Compute ∇ × (φ_j × B₀) at all quadrature points in logical coordinates
        def compute_curl_phi_cross_B(k, j):
            # k is quadrature point index, j is basis function index
            x_log = self.Q.x[k]
            DF_x = DF(x_log)
            J_x = J[k]
            
            def phi_cross_B_func(x_log):
                phi_val = self.Λ0(x_log, j)
                B0_val = self.B0_func(x_log)
                return jnp.cross(phi_val, B0_val)
            return (1/J_x) * DF_x @ curl(phi_cross_B_func)(x_log)
        
        curl_phi_cross_B_vals = jax.vmap(jax.vmap(compute_curl_phi_cross_B, (0, None)), (None, 0))(
            jnp.arange(len(self.Q.x)), jnp.arange(self.n0))
        
        # Step 4: Compute integrand φ_i · [(∇ × B₀) × (∇ × (φ_j × B₀))]
        def compute_integrand(k, i, j):
            phi_i = phi_vals[k, i]
            curl_B0_k = curl_B0[k]
            curl_phi_j_cross_B = curl_phi_cross_B_vals[k, j]
            cross_product = jnp.cross(curl_B0_k, curl_phi_j_cross_B)
            return jnp.dot(phi_i, cross_product)
        
        # Step 5: Vectorize 
        integrand_vals = jax.vmap(jax.vmap(jax.vmap(compute_integrand, (0, None, None)), (None, 0, None)), (None, None, 0))(
            jnp.arange(len(self.Q.x)), jnp.arange(self.n0), jnp.arange(self.n0))
        
        # Step 6: Integrate 
        result = jnp.einsum('ijk,k->ij', integrand_vals, self.Q.w)
        
        return result




class LazyPressureGradientForceMatrix(LazyMatrix):
    """
    A class representing the pressure gradient force matrix.

    The matrix entries are computed as ∫ φ_i · ∇(φ_j · ∇p) dx using the 
    complete vector calculus formula in logical coordinates:
    ∇(φ_j · ∇p) = (∇ ·φ_j) ∇p + ∇²pφ_j + ∇p×(∇×φ_j) + φ_j×(∇×∇p) = (∇ ·φ_j) ∇p + ∇²pφ_j  + ∇p×(∇×φ_j), since the 
    curl of a gradient vanishes.

    Attributes:
        Inherits all attributes from LazyMatrix.
        pressure_func (callable): The pressure function p(x).

    Methods:
        __init__(Λ, Q, pressure_func, F=None, E=None):
            Initialize the pressure gradient force matrix.
        assemble():
            Assemble the pressure gradient force matrix using logical coordinates.
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
        self.pressure_func = pressure_func
        super().__init__(Λ, Λ, Q, F, E, E)

    def assemble(self):
        """Assemble the pressure gradient force matrix in logical coordinates.
        
        ∫ φ_i · ∇(φ_j · ∇p) dx = (∇ ·φ_j) ∇p + ∇²pφ_j  + ∇p×(∇×φ_j)
        """
        DF = jax.jacfwd(self.F)
        
        def _phi(k, i):
            """Get basis function value at logical coordinates."""
            return self.Λ0(self.Q.x[k], i)
        
        def _div_phi(k, j):
            """Compute divergence of basis function φ_j in logical coordinates."""
            def phi_j_func(x_log):
                return self.Λ0(x_log, j)
            return div(phi_j_func)(self.Q.x[k])
        
        def _curl_phi(k, j):
            """Compute curl of basis function φ_j in logical coordinates."""
            def phi_j_func(x_log):
                return self.Λ0(x_log, j)
            return curl(phi_j_func)(self.Q.x[k])
        
        # Step 1: Precompute values at all quadrature points
        phi_vals = jax.vmap(jax.vmap(_phi, (0, None)), (None, 0))(
            jnp.arange(len(self.Q.x)), jnp.arange(self.n0))
        
        # Step 2: Get divergence of basis functions ∇ · φ_j in logical coordinates
        div_phi_vals = jax.vmap(jax.vmap(_div_phi, (0, None)), (None, 0))(
            jnp.arange(len(self.Q.x)), jnp.arange(self.n0))
        
        # Step 3: Get pressure gradient ∇p at logical points
        grad_p_vals = jax.vmap(grad(self.pressure_func))(self.Q.x)
        
        # Step 4: Get pressure Hessian: (1/J) * div(inv33(DF).T @ grad(p))
        def compute_hess_p(x_log):
            DF_x = DF(x_log)
            J_x = jacobian_determinant(self.F)(x_log)  # Evaluate J at the specific point
            
            # Transform gradient: inv33(DF).T @ grad(p)
            def transformed_grad_p(y_log):
                DF_y = DF(y_log)
                grad_p_y = grad(self.pressure_func)(y_log)
                return inv33(DF_y).T @ grad_p_y
            
            # Take divergence and scale by 1/J
            div_transformed_grad_p = div(transformed_grad_p)(x_log)
            return (1/J_x) * div_transformed_grad_p
        
        hess_p_vals = jax.vmap(compute_hess_p)(self.Q.x)
        
        # Step 5: Curl of basis functions ∇ × φ_j in logical coordinates
        curl_phi_vals = jax.vmap(jax.vmap(_curl_phi, (0, None)), (None, 0))(
            jnp.arange(len(self.Q.x)), jnp.arange(self.n0))

        # Step 6: Now we have all of the relevant pieces. Compute the integrand for each (i,j) pair at each quadrature point
        def compute_integrand(L, i, j):
            """
            Compute φ_i · ∇(φ_j · ∇p) at quadrature point L for basis functions i,j.
            """
            phi_i = phi_vals[L, i]          # φ_i at point L
            phi_j = phi_vals[L, j]          # φ_j at point L 
            div_phi_j = div_phi_vals[L, j]  # ∇ · φ_j at point L (scalar)
            grad_p = grad_p_vals[L]         # ∇p at point L
            hess_p = hess_p_vals[L]         # Transformed Hessian at point L
            curl_phi_j = curl_phi_vals[L, j] # ∇×φ_j at point L
            
            # Get DF and J at this quadrature point
            x_log = self.Q.x[L]
            DF_x = DF(x_log)
            J_x = jacobian_determinant(self.F)(x_log)
            
            # Formula: ∇(φ_j · ∇p) = (∇ · φ_j) ∇p + ∇²p φ_j + ∇p×(∇×φ_j)
            term1 = (1/J_x) * div_phi_j * (inv33(DF_x).T @ grad_p)       # (∇ · φ_j) ∇p  
            term2 = hess_p * phi_j          # ∇²p φ_j  (hess_p is already transformed )
            term3 = jnp.cross(inv33(DF_x).T @ grad_p, (1/J_x) * DF_x @ curl_phi_j)  # ∇p×(∇×φ_j)
            
            grad_phi_j_dot_grad_p = term1 + term2 + term3
            
            # Compute φ_i · ∇(φ_j · ∇p)
            result = jnp.dot(phi_i, grad_phi_j_dot_grad_p)
            
            return result
        
        # Step 7: Vectorize
        integrand_vals = jax.vmap(jax.vmap(jax.vmap(compute_integrand, (0, None, None)), (None, 0, None)), (None, None, 0))(
            jnp.arange(len(self.Q.x)), jnp.arange(self.n0), jnp.arange(self.n0))
        
        # Step 8: Integrate 
        result = jnp.einsum('ijk,k->ij', integrand_vals, self.Q.w)
        
        return result

