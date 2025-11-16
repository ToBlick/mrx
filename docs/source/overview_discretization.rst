Finite Element Discretization
==============================

**Notation Conventions**

Throughout this documentation, the following notation conventions are used:

- **Hats** (:math:`\hat{\cdot}`): Quantities in the logical domain :math:`[0,1]^3`. For example:
  - :math:`\hat{x} = (r, \theta, \zeta) \in [0,1]^3` denotes logical coordinates
  - :math:`\hat{\mathbf{u}}` denotes degrees of freedom (coefficients) in logical coordinates
  - :math:`\hat{f}_i` denotes the :math:`i`-th coefficient of a function expanded in basis functions
  
- **Subscript h** (:math:`\cdot_h`): Discretized (finite element) quantities. For example:
  - :math:`u_h` denotes the discrete approximation of a continuous function :math:`u`
  - :math:`\mathbf{B}_h` denotes the discrete magnetic field
  - :math:`\nabla_h` denotes the discrete gradient operator (matrix representation)
  - :math:`M_h` denotes a discrete mass matrix
  
- **Boldface** (:math:`\mathbf{\cdot}`): Vector quantities in physical space. For example:
  - :math:`\mathbf{B}` denotes the continuous magnetic field vector
  - :math:`\mathbf{J}` denotes the current density vector
  - :math:`\mathbf{v}` denotes a velocity field
  
- **Boldface with hat** (:math:`\hat{\mathbf{\cdot}}`): Vector quantities in logical domain or coefficient vectors. For example:
  - :math:`\hat{\mathbf{u}}` denotes the vector of degrees of freedom
  - :math:`\hat{\mathbf{B}}` denotes the magnetic field coefficients
  
- **Basis functions**: :math:`\Lambda_k^i` denotes the :math:`i`-th basis function for k-forms, evaluated in logical coordinates
  
- **Mapping**: :math:`\Phi: [0,1]^3 \to \Omega` (or :math:`F`) maps logical coordinates to physical coordinates
  - :math:`D\Phi(\hat{x})` denotes the Jacobian matrix of the mapping
  - :math:`G(\hat{x}) = D\Phi(\hat{x})^T D\Phi(\hat{x})` denotes the metric tensor
  - :math:`J(\hat{x}) = \det(D\Phi(\hat{x}))` denotes the Jacobian determinant

**General Setup**

MRX uses a **DeRham sequence** for finite element discretization. The domain is parameterized by logical
coordinates :math:`(r, \theta, \zeta) \in [0,1]^3` mapped to physical coordinates via a mapping function :math:`F: [0,1]^3 \to \Omega`.

The DeRham sequence provides a natural framework for enforcing the divergence-free constraint on the magnetic field.
The magnetic field :math:`\mathbf{B}` is represented as a **2-form** (area form) in the finite element space, which
automatically ensures :math:`\operatorname{div} \mathbf{B} = 0` when using appropriate boundary conditions.

The current density :math:`\mathbf{J} = \operatorname{curl}(\mathbf{B})` is computed as a **1-form** using the weak curl operator
from the 2-form space to the 1-form space.

**Mesh Parameters and Polynomial Degrees**

- **Mesh parameters**: :math:`n_r, n_\theta, n_\zeta` are the number of B-spline basis functions in each direction
- **Polynomial degrees**: :math:`p_r, p_\theta, p_\zeta` are the B-spline polynomial degrees in each direction
- **Quadrature order**: :math:`q` is typically chosen as :math:`q = \max(p_r, p_\theta, p_\zeta) + 2` for accurate integration

**Boundary Conditions**

Boundary conditions affect the number of degrees of freedom:

- **Clamped**: Reduces DOFs by 1 in that direction (enforces zero boundary conditions)
- **Periodic**: Keeps all :math:`n` basis functions (periodic boundary conditions)
- **Constant**: Keeps all :math:`n` basis functions (constant in that direction, used for 2D problems)

**Number of Basis Functions**

The number of degrees of freedom :math:`N_k` for k-forms depends on mesh parameters :math:`n_r, n_\theta, n_\zeta` and boundary conditions:

For **0-forms** (scalar functions):
- Total 0-form DOFs: :math:`N_0 = n_r \cdot n_\theta \cdot n_\zeta`
- Basis functions are tensor products: :math:`\Lambda_0^i(r,\theta,\zeta) = \Lambda_r^{i_r}(r) \Lambda_\theta^{i_\theta}(\theta) \Lambda_\zeta^{i_\zeta}(\zeta)`

For **1-forms** (vector fields):
- Define :math:`d_r = n_r - 1` if clamped in radial direction, else :math:`d_r = n_r`
- Define :math:`d_\theta = n_\theta - 1` if clamped in poloidal direction, else :math:`d_\theta = n_\theta`
- Define :math:`d_\zeta = n_\zeta - 1` if clamped in toroidal direction, else :math:`d_\zeta = n_\zeta`
- Total 1-form DOFs: :math:`N_1 = d_r \cdot n_\theta \cdot n_\zeta + n_r \cdot d_\theta \cdot n_\zeta + n_r \cdot n_\theta \cdot d_\zeta`
- Components: :math:`(d_r, n_\theta, n_\zeta)`, :math:`(n_r, d_\theta, n_\zeta)`, :math:`(n_r, n_\theta, d_\zeta)`

For **2-forms** (area forms):
- Total 2-form DOFs: :math:`N_2 = n_r \cdot d_\theta \cdot d_\zeta + d_r \cdot n_\theta \cdot d_\zeta + d_r \cdot d_\theta \cdot n_\zeta`
- Components: :math:`(n_r, d_\theta, d_\zeta)`, :math:`(d_r, n_\theta, d_\zeta)`, :math:`(d_r, d_\theta, n_\zeta)`

For **3-forms** (volume forms):
- Total 3-form DOFs: :math:`N_3 = d_r \cdot d_\theta \cdot d_\zeta`

**Matrix and Operator Dimensions**

All matrices and operators have explicit dimensions based on the number of DOFs:

**Mass Matrices**

Mass matrices define the :math:`L^2` inner product in the finite element spaces. They are used for:
- Projecting functions onto finite element spaces
- Computing norms and inner products
- Solving linear systems via Galerkin projection

**0-form mass matrix** (:math:`M_0 \in \mathbb{R}^{N_0 \times N_0}`):
Computes the :math:`L^2` inner product of scalar functions (0-forms):

.. math::

    (M_0)_{ij} = \int_{[0,1]^3} \Lambda_0^i(\hat{x}) \Lambda_0^j(\hat{x}) \det(D\Phi(\hat{x})) \, \mathrm{d}\hat{x}

where :math:`\{\Lambda_0^i\}` are the 0-form basis functions in logical coordinates.
For scalar functions :math:`u_h = \sum_i \hat{u}_i \Lambda_0^i` (where :math:`u_h` is the discrete function and :math:`\hat{u}_i` are the coefficients in logical coordinates), the :math:`L^2` norm is :math:`\|u_h\|_{L^2}^2 = \hat{\mathbf{u}}^T M_0 \hat{\mathbf{u}}`.

**1-form mass matrix** (:math:`M_1 \in \mathbb{R}^{N_1 \times N_1}`):
Computes the :math:`L^2` inner product of vector fields (1-forms):

.. math::

    (M_1)_{ij} = \int_{[0,1]^3} \Lambda_1^i(\hat{x}) \cdot G^{-1}(\hat{x}) \Lambda_1^j(\hat{x}) \det(D\Phi(\hat{x})) \, \mathrm{d}\hat{x}

where :math:`G^{-1}(\hat{x}) = (D\Phi(\hat{x})^T D\Phi(\hat{x}))^{-1}` is the inverse metric tensor.
For 1-forms :math:`\mathbf{v}_h = \sum_i \hat{v}_i \Lambda_1^i` (where :math:`\mathbf{v}_h` is the discrete vector field and :math:`\hat{v}_i` are the coefficients in logical coordinates), the :math:`L^2` norm is :math:`\|\mathbf{v}_h\|_{L^2}^2 = \hat{\mathbf{v}}^T M_1 \hat{\mathbf{v}}`.

**2-form mass matrix** (:math:`M_2 \in \mathbb{R}^{N_2 \times N_2}`):
Computes the :math:`L^2` inner product of area forms (2-forms, e.g., magnetic field):

.. math::

    (M_2)_{ij} = \int_{[0,1]^3} \Lambda_2^i(\hat{x}) \cdot G(\hat{x}) \Lambda_2^j(\hat{x}) \frac{1}{\det(D\Phi(\hat{x}))} \, \mathrm{d}\hat{x}

where :math:`G(\hat{x}) = D\Phi(\hat{x})^T D\Phi(\hat{x})` is the metric tensor.
For 2-forms :math:`\mathbf{B}_h = \sum_i \hat{B}_i \Lambda_2^i` (where :math:`\mathbf{B}_h` is the discrete magnetic field and :math:`\hat{B}_i` are the coefficients in logical coordinates), the :math:`L^2` norm is :math:`\|\mathbf{B}_h\|_{L^2}^2 = \hat{\mathbf{B}}^T M_2 \hat{\mathbf{B}}`.
This is the primary mass matrix used for magnetic fields in MRX.

**3-form mass matrix** (:math:`M_3 \in \mathbb{R}^{N_3 \times N_3}`):
Computes the :math:`L^2` inner product of volume forms (3-forms):

.. math::

    (M_3)_{ij} = \int_{[0,1]^3} \Lambda_3^i(\hat{x}) \Lambda_3^j(\hat{x}) \frac{1}{\det(D\Phi(\hat{x}))} \, \mathrm{d}\hat{x}

For 3-forms :math:`\rho_h = \sum_i \hat{\rho}_i \Lambda_3^i` (where :math:`\rho_h` is the discrete density and :math:`\hat{\rho}_i` are the coefficients in logical coordinates), the :math:`L^2` norm is :math:`\|\rho_h\|_{L^2}^2 = \hat{\boldsymbol{\rho}}^T M_3 \hat{\boldsymbol{\rho}}`.

**Derivative Operators**

Derivative operators map between different form spaces in the DeRham sequence. The "strong" and "weak" terminology refers to how the derivative is computed:
- **Strong** operators compute derivatives directly on the basis functions
- **Weak** operators use integration by parts (weak formulation)

**Strong gradient** (:math:`\nabla: V_0 \to V_1`, matrix representation :math:`\nabla_h \in \mathbb{R}^{N_1 \times N_0}`):
Computes the gradient of a 0-form (scalar function) to obtain a 1-form (vector field):

.. math::

    (\nabla_h)_{ij} = \int_{[0,1]^3} \Lambda_1^i(\hat{x}) \cdot G^{-1}(\hat{x}) \nabla \Lambda_0^j(\hat{x}) \det(D\Phi(\hat{x})) \, \mathrm{d}\hat{x}

This corresponds to the continuous operator :math:`\nabla: V_0 \to V_1`.

**Weak divergence** (:math:`\nabla \cdot: V_1 \to V_0`, matrix representation :math:`(\nabla \cdot)_h \in \mathbb{R}^{N_0 \times N_1}`):
Computes the divergence of a 1-form using integration by parts:

.. math::

    ((\nabla \cdot)_h)_{ij} = -\int_{[0,1]^3} \Lambda_0^i(\hat{x}) \nabla \cdot \Lambda_1^j(\hat{x}) \det(D\Phi(\hat{x})) \, \mathrm{d}\hat{x}

This corresponds to the continuous operator :math:`\nabla \cdot: V_1 \to V_0`.

**Weak curl** (:math:`\nabla \times: V_2 \to V_1`, matrix representation :math:`(\nabla \times)_h \in \mathbb{R}^{N_1 \times N_2}`):
Computes the curl of a 2-form (magnetic field) to obtain a 1-form (current density):

.. math::

    ((\nabla \times)_h)_{ij} = \int_{[0,1]^3} \Lambda_1^i(\hat{x}) \cdot \nabla \times \Lambda_2^j(\hat{x}) \det(D\Phi(\hat{x})) \, \mathrm{d}\hat{x}

This corresponds to the continuous operator :math:`\nabla \times: V_2 \to V_1` and is used to compute :math:`\mathbf{J} = \nabla \times \mathbf{B}`.

**Strong curl** (:math:`\nabla \times: V_1 \to V_2`, matrix representation :math:`(\nabla \times)_h \in \mathbb{R}^{N_2 \times N_1}`):
Computes the curl of a 1-form to obtain a 2-form:

.. math::

    ((\nabla \times)_h)_{ij} = \int_{[0,1]^3} \Lambda_2^i(\hat{x}) \cdot G(\hat{x}) \nabla \times \Lambda_1^j(\hat{x}) \frac{1}{\det(D\Phi(\hat{x}))} \, \mathrm{d}\hat{x}

This corresponds to the continuous operator :math:`\nabla \times: V_1 \to V_2` and is used in the time-stepping scheme.

**Strong divergence** (:math:`\nabla \cdot: V_2 \to V_3`, matrix representation :math:`(\nabla \cdot)_h \in \mathbb{R}^{N_3 \times N_2}`):
Computes the divergence of a 2-form:

.. math::

    ((\nabla \cdot)_h)_{ij} = \int_{[0,1]^3} \Lambda_3^i(\hat{x}) \nabla \cdot \Lambda_2^j(\hat{x}) \frac{1}{\det(D\Phi(\hat{x}))} \, \mathrm{d}\hat{x}

This corresponds to the continuous operator :math:`\nabla \cdot: V_2 \to V_3`.

**Weak gradient** (:math:`\nabla: V_3 \to V_2`, matrix representation :math:`\nabla_h \in \mathbb{R}^{N_2 \times N_3}`):
Computes the gradient of a 3-form using integration by parts:

.. math::

    (\nabla_h)_{ij} = -\int_{[0,1]^3} \Lambda_2^i(\hat{x}) \cdot G(\hat{x}) \nabla \Lambda_3^j(\hat{x}) \frac{1}{\det(D\Phi(\hat{x}))} \, \mathrm{d}\hat{x}

This corresponds to the continuous operator :math:`\nabla: V_3 \to V_2`.

**Laplacian Operators**

Laplacian operators are constructed from the derivative operators and mass matrices. They represent the second-order differential operators :math:`\Delta = \nabla^2` in the finite element spaces.

**0-form Laplacian** (:math:`\Delta_0 \in \mathbb{R}^{N_0 \times N_0}`):
The scalar Laplacian :math:`\Delta u = \nabla \cdot \nabla u`:

.. math::

    \Delta_0 = M_0^{-1} \nabla_h^T M_1 \nabla_h

The matrix :math:`\nabla_h^T M_1 \nabla_h` represents the gradient-gradient operator :math:`\nabla \cdot \nabla`:

.. math::

    (\nabla_h^T M_1 \nabla_h)_{ij} = \int_{[0,1]^3} \nabla \Lambda_0^i(\hat{x}) \cdot G^{-1}(\hat{x}) \nabla \Lambda_0^j(\hat{x}) \det(D\Phi(\hat{x})) \, \mathrm{d}\hat{x}

**1-form Laplacian** (:math:`\Delta_1 \in \mathbb{R}^{N_1 \times N_1}`):
The vector Laplacian :math:`\Delta \mathbf{v} = \nabla \times (\nabla \times \mathbf{v}) - \nabla (\nabla \cdot \mathbf{v})`:

.. math::

    \Delta_1 = M_1^{-1} (\nabla \times)_h^T M_2 (\nabla \times)_h - \nabla_h \circ (\nabla \cdot)_h

The curl-curl term :math:`(\nabla \times)_h^T M_2 (\nabla \times)_h` represents :math:`\nabla \times (\nabla \times \mathbf{v})`:

.. math::

    ((\nabla \times)_h^T M_2 (\nabla \times)_h)_{ij} = \int_{[0,1]^3} (\nabla \times \Lambda_1^i(\hat{x})) \cdot G(\hat{x}) (\nabla \times \Lambda_1^j(\hat{x})) \frac{1}{\det(D\Phi(\hat{x}))} \, \mathrm{d}\hat{x}

The gradient-divergence term :math:`\nabla_h \circ (\nabla \cdot)_h` represents :math:`\nabla (\nabla \cdot \mathbf{v})`.

**2-form Laplacian** (:math:`\Delta_2 \in \mathbb{R}^{N_2 \times N_2}`):
The 2-form Laplacian :math:`\Delta \mathbf{B} = -\nabla (\nabla \cdot \mathbf{B}) + \nabla \times (\nabla \times \mathbf{B})`:

.. math::

    \Delta_2 = -M_2^{-1} (\nabla \cdot)_h^T M_3 (\nabla \cdot)_h + M_2^{-1} (\nabla \times)_h^T M_1 (\nabla \times)_h

The divergence-divergence term :math:`(\nabla \cdot)_h^T M_3 (\nabla \cdot)_h` represents :math:`\nabla \cdot (\nabla \cdot \mathbf{B})`:

.. math::

    ((\nabla \cdot)_h^T M_3 (\nabla \cdot)_h)_{ij} = \int_{[0,1]^3} (\nabla \cdot \Lambda_2^i(\hat{x})) (\nabla \cdot \Lambda_2^j(\hat{x})) \frac{1}{\det(D\Phi(\hat{x}))} \, \mathrm{d}\hat{x}

The curl-curl term :math:`(\nabla \times)_h^T M_1 (\nabla \times)_h` represents :math:`\nabla \times (\nabla \times \mathbf{B})`:

.. math::

    ((\nabla \times)_h^T M_1 (\nabla \times)_h)_{ij} = \int_{[0,1]^3} (\nabla \times \Lambda_2^i(\hat{x})) \cdot G^{-1}(\hat{x}) (\nabla \times \Lambda_2^j(\hat{x})) \det(D\Phi(\hat{x})) \, \mathrm{d}\hat{x}

**3-form Laplacian** (:math:`\Delta_3 \in \mathbb{R}^{N_3 \times N_3}`):
The 3-form Laplacian :math:`\Delta \rho = -\nabla \cdot (\nabla \rho)`:

.. math::

    \Delta_3 = -(\nabla \cdot)_h \circ \nabla_h

This represents the negative divergence of the gradient operator.

**Projection Operators**

Projection operators :math:`P_k: L^2([0,1]^3) \to V_k` project functions onto the finite element spaces using Galerkin projection.
Given a function :math:`f: [0,1]^3 \to \mathbb{R}^d` (where :math:`d=1` for scalars, :math:`d=3` for vectors), the projection finds coefficients :math:`\hat{\mathbf{f}} \in \mathbb{R}^{N_k}` such that:

.. math::

    \int_\Omega (P_k f)(x) \cdot \Lambda_k^j(x) \, \mathrm{d}x = \int_\Omega f(x) \cdot \Lambda_k^j(x) \, \mathrm{d}x \quad \forall j

where :math:`(P_k f)(x) = \sum_i \hat{f}_i \Lambda_k^i(x)` is the projected function (subscript :math:`h` is implicit in :math:`P_k f`, and :math:`\hat{f}_i` are the coefficients in logical coordinates).

**0-form projection** (:math:`P_0: L^2([0,1]^3) \to V_0`, :math:`P_0(f) \in \mathbb{R}^{N_0^{\mathrm{extracted}}}`):
Projects a scalar function onto the 0-form space. The projection coefficients are computed as:

.. math::

    \hat{f}_i = \int_{[0,1]^3} f(\hat{x}) \Lambda_0^i(\hat{x}) \det(D\Phi(\hat{x})) \, \mathrm{d}\hat{x}

The projection operator applies the extraction operator :math:`E_0` to enforce boundary conditions:

.. math::

    P_0(f) = E_0 \, \hat{\mathbf{f}}

**1-form projection** (:math:`P_1: L^2([0,1]^3; \mathbb{R}^3) \to V_1`, :math:`P_1(\mathbf{v}) \in \mathbb{R}^{N_1^{\mathrm{extracted}}}`):
Projects a vector field onto the 1-form space. The function is first pulled back to logical coordinates using the inverse Jacobian:

.. math::

    \hat{v}_i = \int_{[0,1]^3} (D\Phi(\hat{x})^{-1} \mathbf{v}(\hat{x})) \cdot \Lambda_1^i(\hat{x}) \det(D\Phi(\hat{x})) \, \mathrm{d}\hat{x}

The projection operator applies the extraction operator :math:`E_1`:

.. math::

    P_1(\mathbf{v}) = E_1 \, \hat{\mathbf{v}}

**2-form projection** (:math:`P_2: L^2([0,1]^3; \mathbb{R}^3) \to V_2`, :math:`P_2(\mathbf{B}) \in \mathbb{R}^{N_2^{\mathrm{extracted}}}`):
Projects a vector field (e.g., magnetic field) onto the 2-form space. The function is transformed using the Jacobian transpose:

.. math::

    \hat{B}_i = \int_{[0,1]^3} (D\Phi(\hat{x})^T \mathbf{B}(\hat{x})) \cdot \Lambda_2^i(\hat{x}) \, \mathrm{d}\hat{x}

Note that the Jacobian determinant factor is not included here (unlike 0-forms and 1-forms) because 2-forms transform differently under coordinate changes.
The projection operator applies the extraction operator :math:`E_2`:

.. math::

    P_2(\mathbf{B}) = E_2 \, \hat{\mathbf{B}}

**3-form projection** (:math:`P_3: L^2([0,1]^3) \to V_3`, :math:`P_3(\rho) \in \mathbb{R}^{N_3^{\mathrm{extracted}}}`):
Projects a scalar function (density) onto the 3-form space:

.. math::

    \hat{\rho}_i = \int_{[0,1]^3} \rho(\hat{x}) \Lambda_3^i(\hat{x}) \frac{1}{\det(D\Phi(\hat{x}))} \, \mathrm{d}\hat{x}

The projection operator applies the extraction operator :math:`E_3`:

.. math::

    P_3(\rho) = E_3 \, \hat{\boldsymbol{\rho}}

**Use Cases:**
- **Initial conditions**: Projecting analytical magnetic fields onto the discrete 2-form space
- **Source terms**: Projecting source functions (e.g., pressure gradients) onto appropriate form spaces
- **Data interpolation**: Projecting data from external sources (e.g., GVEC) onto finite element spaces
- **Boundary conditions**: Enforcing boundary conditions through the extraction operators

**Leray Projection**

The Leray projection :math:`P_{\mathrm{Leray}}: V_2 \to V_2^{\mathrm{div-free}}` projects onto the divergence-free subspace:

.. math::

    P_{\mathrm{Leray}} = I - \nabla_h \circ ((\nabla \cdot)_h \circ M_1^{-1} \circ \nabla_h)^{-1} \circ (\nabla \cdot)_h

Dimensions: :math:`P_{\mathrm{Leray}} \in \mathbb{R}^{N_2 \times N_2}`.

This projection ensures that :math:`\nabla \cdot (P_{\mathrm{Leray}} \mathbf{u}) = 0` for all :math:`\mathbf{u} \in V_2`.
In MRX, the magnetic field is represented directly in the 2-form space, which naturally satisfies the divergence-free
constraint, making the Leray projection particularly important for maintaining this property during relaxation.

**Cross-Product Projections**

Cross-product projections compute the Lorentz force term :math:`\mathbf{J} \times \mathbf{B}` in finite element space.
This is essential for the force balance equation :math:`\mathbf{J} \times \mathbf{B} = \operatorname{grad} p`:

- :math:`P_{2 \times 1 \to 2}: V_2 \times V_1 \to V_2`, computes :math:`\mathbf{B} \times \mathbf{J}` where :math:`\mathbf{B} \in V_2`, :math:`\mathbf{J} \in V_1`, maps to :math:`\mathbb{R}^{N_2}`
- :math:`P_{1 \times 1 \to 2}: V_1 \times V_1 \to V_2`, maps to :math:`\mathbb{R}^{N_2}`

The cross-product projection :math:`P_{2 \times 1 \to 2}` is computed as:

.. math::

    (P_{2 \times 1 \to 2}(\mathbf{B}, \mathbf{J}))_i = \int_\Omega (\mathbf{B} \times \mathbf{J}) \cdot \Lambda_2^i(x) \frac{1}{\det(DF(x))} \, dx

where :math:`\{\Lambda_2^i\}` are the 2-form basis functions.

**Extraction Operators**

Extraction operators :math:`E_0, E_1, E_2, E_3` enforce boundary conditions and reduce the number of degrees of freedom by eliminating basis functions that violate boundary conditions.
They map from the full (unconstrained) space to the extracted (constrained) space:

- :math:`E_0 \in \mathbb{R}^{N_0^{\mathrm{extracted}} \times N_0^{\mathrm{full}}}`
- :math:`E_1 \in \mathbb{R}^{N_1^{\mathrm{extracted}} \times N_1^{\mathrm{full}}}`
- :math:`E_2 \in \mathbb{R}^{N_2^{\mathrm{extracted}} \times N_2^{\mathrm{full}}}`
- :math:`E_3 \in \mathbb{R}^{N_3^{\mathrm{extracted}} \times N_3^{\mathrm{full}}}`

where :math:`N_k^{\mathrm{extracted}} \leq N_k^{\mathrm{full}}` depends on boundary conditions.

**Mathematical Definition:**

The extraction operators are sparse matrices that select the admissible basis functions.
For a k-form with DOFs :math:`\hat{\mathbf{u}}^{\mathrm{full}} \in \mathbb{R}^{N_k^{\mathrm{full}}}`, the extracted DOFs are:

.. math::

    \hat{\mathbf{u}}^{\mathrm{extracted}} = E_k \, \hat{\mathbf{u}}^{\mathrm{full}}

The extraction operator :math:`E_k` has entries:

.. math::

    (E_k)_{ij} = \begin{cases}
        1 & \text{if basis function } j \text{ is admissible (satisfies BCs)} \\
        0 & \text{otherwise}
    \end{cases}

**Boundary Condition Enforcement:**

The extraction operators enforce boundary conditions by:

1. **Clamped (Dirichlet) boundary conditions**: Remove basis functions that are non-zero on the boundary where zero values are required.
   For example, in the radial direction with clamped BCs at :math:`r=0,1`, basis functions with support at the boundary are removed.

2. **Periodic boundary conditions**: Keep all basis functions but enforce periodicity constraints, reducing DOFs by the number of constraints.

3. **Polar coordinates**: Handle the axis singularity at :math:`r=0` by removing basis functions that would cause singular behavior.

**Usage:**

Extraction operators are applied:
- **After projection**: :math:`P_k(f) = E_k \, \hat{\mathbf{f}}` where :math:`\hat{\mathbf{f}}` are the projection coefficients before boundary conditions
- **In matrix assembly**: Mass matrices and derivative operators are assembled as :math:`M_k = E_k \, M_k^{\mathrm{full}} \, E_k^T` to ensure boundary conditions are satisfied
- **In solving linear systems**: The extracted DOFs :math:`\hat{\mathbf{u}}^{\mathrm{extracted}}` are solved for, and the full solution is reconstructed if needed

**Example:**

For a 0-form with clamped boundary conditions at :math:`r=0,1`:
- Full space: :math:`N_0^{\mathrm{full}} = n_r \cdot n_\theta \cdot n_\zeta` DOFs
- Extracted space: :math:`N_0^{\mathrm{extracted}} = (n_r-2) \cdot n_\theta \cdot n_\zeta` DOFs (removing 2 radial basis functions)
- The extraction operator :math:`E_0` removes the first and last radial basis functions at each :math:`(\theta, \zeta)` point

**Common Patterns**

For typical 2D problems (disc domain with :math:`n_\zeta = 1`, :math:`p_\zeta = 0`):
- :math:`N_0 = n_r \cdot n_\theta`
- :math:`N_1 = d_r \cdot n_\theta + n_r \cdot d_\theta`
- :math:`N_2 = n_r \cdot d_\theta + d_r \cdot n_\theta`
- :math:`N_3 = d_r \cdot d_\theta`

For typical 3D problems (toroidal domain):
- :math:`N_0 = n_r \cdot n_\theta \cdot n_\zeta`
- :math:`N_1 = d_r \cdot n_\theta \cdot n_\zeta + n_r \cdot d_\theta \cdot n_\zeta + n_r \cdot n_\theta \cdot d_\zeta`
- :math:`N_2 = n_r \cdot d_\theta \cdot d_\zeta + d_r \cdot n_\theta \cdot d_\zeta + d_r \cdot d_\theta \cdot n_\zeta`
- :math:`N_3 = d_r \cdot d_\theta \cdot d_\zeta`

**Geometric Mapping**

The transformation between logical coordinates :math:`(r, \theta, \zeta) \in [0,1]^3` and physical coordinates
uses a mapping function :math:`F: [0,1]^3 \to \Omega`. The Jacobian matrix :math:`DF(x)` and its determinant
:math:`\det(DF(x))` play crucial roles in the finite element discretization:

- **Metric tensor**: :math:`G(x) = DF(x)^T DF(x)` accounts for the non-orthogonal coordinate system
- **Inverse metric**: :math:`G^{-1}(x)` is used in Laplacian and gradient computations
- **Jacobian determinant**: :math:`J(x) = \det(DF(x))` appears in all volume integrals

These geometric factors ensure that the discretization maintains accuracy in curved geometries, which is
essential for toroidal and stellarator configurations.

