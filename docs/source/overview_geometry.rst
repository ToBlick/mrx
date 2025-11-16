Geometry and Coordinate Transformations
=======================================

**Notation**: Throughout this section:
- :math:`\hat{x} = (r, \theta, \zeta) \in [0,1]^3` denotes logical coordinates (hat indicates logical domain)
- :math:`x = (x, y, z) \in \Omega \subset \mathbb{R}^3` denotes physical coordinates (no hat indicates physical domain)
- :math:`\Phi: [0,1]^3 \to \Omega` (or :math:`F`) maps logical coordinates to physical coordinates

MRX uses differential geometry to transform between logical coordinates :math:`\hat{x} = (r, \theta, \zeta) \in [0,1]^3`
and physical coordinates :math:`x = (x, y, z) \in \Omega \subset \mathbb{R}^3` via a mapping function :math:`\Phi: [0,1]^3 \to \Omega`.

**Geometric Mapping**

The transformation is defined by a mapping function :math:`\Phi(\hat{x})` that maps logical coordinates to physical coordinates.
The Jacobian matrix :math:`D\Phi(\hat{x})` and its determinant :math:`\det(D\Phi(\hat{x}))` play crucial roles:

- **Jacobian matrix**: :math:`D\Phi(\hat{x}) = \frac{\partial \Phi}{\partial \hat{x}} \in \mathbb{R}^{3 \times 3}`
- **Jacobian determinant**: :math:`J(\hat{x}) = \det(D\Phi(\hat{x}))`
- **Metric tensor**: :math:`G(\hat{x}) = D\Phi(\hat{x})^T D\Phi(\hat{x}) \in \mathbb{R}^{3 \times 3}`
- **Inverse metric**: :math:`G^{-1}(\hat{x}) = (D\Phi(\hat{x})^T D\Phi(\hat{x}))^{-1}`

**Pullback and Pushforward Operations**

Differential forms transform between logical and physical domains using pullback and pushforward operations.
These operations ensure that physical laws (such as :math:`\operatorname{div} \mathbf{B} = 0`) are preserved under coordinate transformations.

**Pushforward** (:math:`\Phi_*`) transforms forms from logical to physical domain.
Note that pushforward is evaluated at logical coordinates :math:`\hat{x}`, and the result is a form in physical space at :math:`x = \Phi(\hat{x})`:

For **0-forms** (scalar functions):

.. math::

    (\Phi_* f)(\Phi(\hat{x})) = f(\hat{x})

For **1-forms** (vector fields):

.. math::

    (\Phi_* \mathbf{v})(\Phi(\hat{x})) = (D\Phi(\hat{x})^{-1})^T \mathbf{v}(\hat{x})

For **2-forms** (area forms, e.g., magnetic field :math:`\mathbf{B}`):

.. math::

    (\Phi_* \mathbf{B})(\Phi(\hat{x})) = \frac{D\Phi(\hat{x}) \mathbf{B}(\hat{x})}{\det(D\Phi(\hat{x}))}

For **3-forms** (volume forms):

.. math::

    (\Phi_* \rho)(\Phi(\hat{x})) = \frac{\rho(\hat{x})}{\det(D\Phi(\hat{x}))}

**Pullback** (:math:`\Phi^*`) transforms forms from physical to logical domain:

For **0-forms** (scalar functions):

.. math::

    (\Phi^* f)(\hat{x}) = f(\Phi(\hat{x}))

For **1-forms** (vector fields):

.. math::

    (\Phi^* \mathbf{v})(\hat{x}) = D\Phi(\hat{x})^T \mathbf{v}(\Phi(\hat{x}))

For **2-forms** (area forms):

.. math::

    (\Phi^* \mathbf{B})(\hat{x}) = D\Phi(\hat{x})^{-1} \mathbf{B}(\Phi(\hat{x})) \det(D\Phi(\hat{x}))

For **3-forms** (volume forms):

.. math::

    (\Phi^* \rho)(\hat{x}) = \rho(\Phi(\hat{x})) \det(D\Phi(\hat{x}))

**Projection with Pullback**

When projecting a function :math:`B_0: [0,1]^3 \to \mathbb{R}^3` (given in logical coordinates) onto the discrete 2-form space,
we pull the problem back to the logical domain. The Galerkin projection in physical space would be:

.. math::

    \sum_i \mathtt{B}_i \int_\Omega (\Phi_* \hat{\Lambda}_i^2)(x) \cdot (\Phi_* \hat{\Lambda}_j^2)(x) \, \mathrm{d}x = \int_\Omega B_0(\Phi^{-1}(x)) \cdot (\Phi_* \hat{\Lambda}_j^2)(x) \, \mathrm{d}x \quad \forall j

where :math:`\hat{\Lambda}_i^2` are the 2-form basis functions in logical coordinates, and :math:`\Phi_*` denotes pushforward.

Pulling back to the logical domain using the change of variables :math:`x = \Phi(\hat{x})`, :math:`\mathrm{d}x = \det(D\Phi(\hat{x})) \, \mathrm{d}\hat{x}`, this becomes:

.. math::

    \sum_i \mathtt{B}_i \int_{[0,1]^3} \hat{\Lambda}_i^2(\hat{x}) \cdot \frac{D\Phi(\hat{x})^T D\Phi(\hat{x})}{\det(D\Phi(\hat{x}))} \hat{\Lambda}_j^2(\hat{x}) \, \mathrm{d}\hat{x} = \int_{[0,1]^3} B_0(\hat{x}) \cdot D\Phi(\hat{x})^T \hat{\Lambda}_j^2(\hat{x}) \, \mathrm{d}\hat{x} \quad \forall j

where we used the pushforward formula for 2-forms and the fact that :math:`(\Phi_* \hat{\Lambda}_j^2)(\Phi(\hat{x})) = D\Phi(\hat{x}) \hat{\Lambda}_j^2(\hat{x}) / \det(D\Phi(\hat{x}))`.

In matrix form:

.. math::

    M_2 \, \mathtt{B} = \Pi^2(B_0)

where:
- :math:`M_2 \in \mathbb{R}^{N_2 \times N_2}` is the 2-form mass matrix with entries :math:`(M_2)_{ij} = \int \hat{\Lambda}_i^2(\hat{x}) \cdot G(\hat{x}) \hat{\Lambda}_j^2(\hat{x}) / \det(D\Phi(\hat{x})) \, \mathrm{d}\hat{x}`
- :math:`\Pi^2: L^2([0,1]^3; \mathbb{R}^3) \to \mathbb{R}^{N_2}` is the 2-form projection operator
- :math:`\mathtt{B} \in \mathbb{R}^{N_2}` are the degrees of freedom

**Mass Matrix with Pullback**

The mass matrix for k-forms incorporates the geometric transformation:

For **0-forms**:

.. math::

    (M_0)_{ij} = \int \Lambda_0^i(\hat{x}) \Lambda_0^j(\hat{x}) \det(D\Phi(\hat{x})) \, \mathrm{d}\hat{x}

For **1-forms**:

.. math::

    (M_1)_{ij} = \int \Lambda_1^i(\hat{x}) \cdot G^{-1}(\hat{x}) \Lambda_1^j(\hat{x}) \det(D\Phi(\hat{x})) \, \mathrm{d}\hat{x}

For **2-forms**:

.. math::

    (M_2)_{ij} = \int \Lambda_2^i(\hat{x}) \cdot G(\hat{x}) \Lambda_2^j(\hat{x}) \frac{1}{\det(D\Phi(\hat{x}))} \, \mathrm{d}\hat{x}

For **3-forms**:

.. math::

    (M_3)_{ij} = \int \Lambda_3^i(\hat{x}) \Lambda_3^j(\hat{x}) \frac{1}{\det(D\Phi(\hat{x}))} \, \mathrm{d}\hat{x}

**Field Line Integration**

Field lines are integrated in the logical domain. The magnetic field equation in physical space:

.. math::

    \frac{\mathrm{d}}{\mathrm{d}t} x_t = \frac{\mathbf{B}(x_t)}{|\mathbf{B}(x_t)|}

where :math:`\mathbf{B}(x_t)` is the magnetic field vector at physical point :math:`x_t`.

Transforming to logical coordinates using :math:`x_t = \Phi(\hat{x}_t)` and the chain rule:

.. math::

    \frac{\mathrm{d}}{\mathrm{d}t} \hat{x}_t = D\Phi(\hat{x}_t)^{-1} \frac{\mathrm{d}}{\mathrm{d}t} x_t = D\Phi(\hat{x}_t)^{-1} \frac{\mathbf{B}(\Phi(\hat{x}_t))}{|\mathbf{B}(\Phi(\hat{x}_t))|}

Using the pushforward relation for 2-forms, :math:`\mathbf{B}(\Phi(\hat{x})) = D\Phi(\hat{x}) \hat{\mathbf{B}}(\hat{x}) / \det(D\Phi(\hat{x}))`, where :math:`\hat{\mathbf{B}}(\hat{x})` is the magnetic field coefficient vector evaluated at logical coordinates. The magnitude is:

.. math::

    |\mathbf{B}(\Phi(\hat{x}))| = \frac{|D\Phi(\hat{x}) \hat{\mathbf{B}}(\hat{x})|}{\det(D\Phi(\hat{x}))}

Therefore, the field line equation in logical coordinates becomes:

.. math::

    \frac{\mathrm{d}}{\mathrm{d}t} \hat{x}_t = \frac{\hat{\mathbf{B}}(\hat{x}_t)}{|D\Phi(\hat{x}_t) \hat{\mathbf{B}}(\hat{x}_t)|}

where :math:`\hat{\mathbf{B}}(\hat{x})` is obtained by evaluating the discrete 2-form basis functions at :math:`\hat{x}` and multiplying by the degrees of freedom :math:`B_h`.

**Inverse Mapping**

To map from physical coordinates back to logical coordinates (e.g., for visualization or diagnostics),
MRX uses Newton's method to solve :math:`\Phi(\hat{x}) = y` for :math:`\hat{x}` given :math:`y`.
An approximate inverse mapping is used as an initial guess for the Newton iteration.

