Magnetic Relaxation Method
===========================

**Notation**: Throughout this section:

- Subscript :math:`h` indicates discretized (finite element) quantities, which are coefficient vectors (DOFs) in :math:`\mathbb{R}^{N_k}`:
  - :math:`B_h \in \mathbb{R}^{N_2}` denotes the magnetic field coefficients (2-form DOFs)
  - :math:`J_h \in \mathbb{R}^{N_1}` denotes the current density coefficients (1-form DOFs)
  - :math:`v_h \in \mathbb{R}^{N_2}` denotes the velocity field coefficients (2-form DOFs)
  - Boldface with subscript :math:`h` (e.g., :math:`\mathbf{f}_h`, :math:`\mathbf{E}_h`) also denotes coefficient vectors, emphasizing they represent vector fields conceptually
- Quantities without subscript :math:`h` are continuous (e.g., :math:`\mathbf{B}`, :math:`\mathbf{J}`)
- Superscripts :math:`n`, :math:`n+1` denote time step indices
- Hats (:math:`\hat{\cdot}`) indicate logical domain quantities (coefficients are already in logical coordinates, so hats are typically omitted for DOF vectors)
- See the :doc:`overview_discretization` page for complete notation conventions

MRX uses a magnetic relaxation approach to solve the magnetohydrostatic (MHS) equilibrium problem.
The method evolves a magnetic field from an initial non-equilibrium state toward an equilibrium state
that satisfies the force balance equation :math:`\mathbf{J} \times \mathbf{B} = \operatorname{grad} p`.

**Discretization Scheme**

The relaxation method uses a semi-implicit time-stepping scheme with a Picard iteration solver.
At each time step, the following fixed-point problem is solved:

.. math::

    \begin{align*}
        B_h^{n+1/2}     &= \frac{1}{2}(B_h^n + B_h^{n+1}) \\
        J_h             &= (\nabla \times)_h \, B_h^{n+1/2} \\
        H_h             &= P_{1,2} \, B_h^{n+1/2} \\
        \mathbf{f}_h    &= P_{1 \times 1 \to 2}(\mathbf{J}_h, \mathbf{H}_h) \\
        v_h             &= \mathcal{A} \, M_2^{-1} \mathbf{f}_h \\
        \mathbf{E}_h    &= M_1^{-1} P_{2 \times 1 \to 1}(\mathbf{v}_h, \mathbf{H}_h) - \eta \mathbf{J}_h \\
        B_h^{n+1}       &= B_h^n + \delta t \, (\nabla \times)_h \, \mathbf{E}_h
    \end{align*}

where:
- :math:`B_h^n, B_h^{n+1} \in \mathbb{R}^{N_2}` are the magnetic field coefficient vectors (2-form DOFs in logical coordinates) at time steps :math:`n` and :math:`n+1`
- :math:`B_h^{n+1/2} \in \mathbb{R}^{N_2}` is the midpoint coefficient vector used for the Crank-Nicolson scheme
- :math:`J_h \in \mathbb{R}^{N_1}` is the current density coefficient vector (1-form DOFs) computed via :math:`(\nabla \times)_h: V_2 \to V_1`
- :math:`H_h \in \mathbb{R}^{N_1}` is the magnetic field coefficient vector projected to 1-form space via :math:`P_{1,2} = M_1^{-1} M_{12}: V_2 \to V_1`, where :math:`M_{12}` is the mixed mass matrix with entries :math:`(M_{12})_{ij} = \int \Lambda_1^i \cdot \Lambda_2^j \, \mathrm{d}x`
- :math:`\mathbf{f}_h \in \mathbb{R}^{N_2}` is the Lorentz force coefficient vector (2-form DOFs) representing :math:`\mathbf{J}_h \times \mathbf{H}_h` projected to 2-form space
- :math:`v_h \in \mathbb{R}^{N_2}` is the velocity field coefficient vector (2-form DOFs)
- :math:`\mathcal{A}` is typically the Leray projection :math:`P_{\mathrm{Leray}}` to ensure divergence-free velocity
- :math:`\mathbf{E}_h \in \mathbb{R}^{N_1}` is the electric field coefficient vector (1-form DOFs)
- :math:`\eta \geq 0` is the resistivity (typically :math:`\eta = 0` for ideal MHD)
- :math:`\delta t > 0` is the time step size
- :math:`M_2 \in \mathbb{R}^{N_2 \times N_2}` and :math:`M_1 \in \mathbb{R}^{N_1 \times N_1}` are the mass matrices
- :math:`P_{1 \times 1 \to 2}: V_1 \times V_1 \to V_2` and :math:`P_{2 \times 1 \to 1}: V_2 \times V_1 \to V_1` are cross-product projection operators
- :math:`(\nabla \times)_h: V_1 \to V_2` is the curl operator (strong form)

**Picard Iteration**

The fixed-point problem is solved using Picard iteration. Starting with an initial guess :math:`B_h^{n+1,(0)} = B_h^n`,
the iteration proceeds as:

.. math::

    B_h^{n+1,(k+1)} = B_h^n + \delta t \, (\nabla \times)_h \, \mathbf{E}_h^{(k)}

where :math:`\mathbf{E}_h^{(k)}` is computed using the midpoint value :math:`B_h^{n+1/2,(k)} = \frac{1}{2}(B_h^n + B_h^{n+1,(k)})`
in the discretization scheme above.

The iteration continues until the residual:

.. math::

    \epsilon^{(k)} = \|B_h^{n+1,(k+1)} - B_h^{n+1,(k)}\|_{L^2} = ((B_h^{n+1,(k+1)} - B_h^{n+1,(k)})^T M_2 (B_h^{n+1,(k+1)} - B_h^{n+1,(k)}))^{1/2}

falls below a tolerance :math:`\epsilon_{\mathrm{tol}}` (typically :math:`10^{-12}`) or exceeds a maximum number of iterations (typically 20).
If convergence is not achieved, the time step is halved and the iteration is retried.

**Time Step Adaptation**

The time step :math:`\delta t` is adapted based on the convergence behavior of the Picard solver:

- If the Picard solver converges in fewer than 4 iterations, the time step is increased: :math:`\delta t_{\mathrm{new}} = \delta t \times 1.01`
- If the Picard solver requires many iterations, the time step is decreased: :math:`\delta t_{\mathrm{new}} = \delta t / 1.01`
- If the Picard solver fails to converge, the time step is halved and the iteration is retried

**Leray Projection**

The Leray projection :math:`P_{\mathrm{Leray}}` ensures that the velocity field is divergence-free:

.. math::

    P_{\mathrm{Leray}} = I - \nabla_h \circ ((\nabla \cdot)_h \circ M_1^{-1} \circ \nabla_h)^{-1} \circ (\nabla \cdot)_h

This projection is applied to the force term :math:`\mathbf{J}_h \times \mathbf{H}_h` to obtain a divergence-free velocity field.

**Cross-Product Projection**

The cross-product :math:`\mathbf{J}_h \times \mathbf{H}_h` is computed in the finite element space using projection operators.
The projection :math:`P_{1 \times 1 \to 2}(\mathbf{J}_h, \mathbf{H}_h)` computes the 2-form coefficients of :math:`\mathbf{J}_h \times \mathbf{H}_h`.
In logical coordinates, this is:

.. math::

    (P_{1 \times 1 \to 2}(\mathbf{J}_h, \mathbf{H}_h))_i = \int_{[0,1]^3} G(\hat{x}) (\mathbf{J}_h(\hat{x}) \times \mathbf{H}_h(\hat{x})) \cdot \hat{\Lambda}_2^i(\hat{x}) \frac{1}{\det(D\Phi(\hat{x}))} \, \mathrm{d}\hat{x}

where:
- :math:`\{\hat{\Lambda}_2^i\}` are the 2-form basis functions in logical coordinates
- :math:`G(\hat{x}) = D\Phi(\hat{x})^T D\Phi(\hat{x})` is the metric tensor
- :math:`\mathbf{J}_h(\hat{x})`, :math:`\mathbf{H}_h(\hat{x})` are the 1-form fields evaluated at logical coordinates using their pushforward relations

Similarly, :math:`P_{2 \times 1 \to 1}(\mathbf{v}_h, \mathbf{H}_h)` computes the 1-form coefficients of :math:`\mathbf{v}_h \times \mathbf{H}_h`:

.. math::

    (P_{2 \times 1 \to 1}(\mathbf{v}_h, \mathbf{H}_h))_i = \int_{[0,1]^3} (G(\hat{x}) \mathbf{v}_h(\hat{x}) \times \mathbf{H}_h(\hat{x})) \cdot \hat{\Lambda}_1^i(\hat{x}) \frac{1}{\det(D\Phi(\hat{x}))} \, \mathrm{d}\hat{x}

where:
- :math:`\{\hat{\Lambda}_1^i\}` are the 1-form basis functions in logical coordinates
- :math:`\mathbf{v}_h(\hat{x})` is the 2-form field evaluated at logical coordinates

**Force Balance Residual**

The force balance residual measures how well the magnetic field satisfies the equilibrium condition.
The residual is computed as the L2 norm of the velocity field:

.. math::

    \|\mathbf{J} \times \mathbf{B} - \operatorname{grad} p\|_{L^2(\Omega)} = \|v_h\|_{L^2(\Omega)} = (v_h^T M_2 v_h)^{1/2}

where :math:`M_2` is the 2-form mass matrix. This residual decreases as the magnetic field approaches equilibrium.
The relaxation continues until this residual falls below a specified tolerance (typically :math:`10^{-15}`).

**Helicity Conservation**

For ideal MHD (:math:`\eta = 0`), the magnetic helicity:

.. math::

    H = \int_\Omega \mathbf{A} \cdot \mathbf{B} \, \mathrm{d}x

where :math:`\mathbf{A}` is the vector potential satisfying :math:`\mathbf{B} = \operatorname{curl} \mathbf{A}`, should be conserved.
In the discrete setting, helicity is computed as:

.. math::

    H_h = B_h^T M_2 ((\nabla \times)_h)^{-1} B_h

where :math:`((\nabla \times)_h)^{-1}` denotes the pseudoinverse or solution of the curl equation.
MRX monitors the relative helicity change:

.. math::

    \frac{|H_h(t) - H_h(0)|}{|H_h(0)|}

to verify that helicity is approximately conserved during relaxation. Typically, the relative helicity change
remains below :math:`10^{-3}` throughout the relaxation process.

**Initial Condition**

The initial magnetic field :math:`B_0` is projected onto the discrete 2-form space:

.. math::

    M_2 \, \mathtt{B}_0 = \Pi^2(B_0)

The projected field is then made exactly divergence-free using the Leray projection:

.. math::

    \mathtt{B}_0 \leftarrow P_{\mathrm{Leray}} \, \mathtt{B}_0

and normalized to unit L2 norm:

.. math::

    \mathtt{B}_0 \leftarrow \frac{\mathtt{B}_0}{\|\mathtt{B}_0\|_{L^2}}

**State Object**

The relaxation state is tracked using a `State` object containing:

- :math:`B_n`: Current magnetic field DOFs
- :math:`B_{n+1}`: Next time step magnetic field DOFs (guess)
- :math:`\delta t`: Current time step size
- :math:`\eta`: Resistivity
- :math:`H`: Hessian matrix (for Newton method, if enabled)
- Picard iteration count and residual
- Force norm and velocity norm

**Convergence Criteria**

The relaxation is considered converged when:

1. The force balance residual :math:`\|\mathbf{J} \times \mathbf{B} - \operatorname{grad} p\|_{L^2}` falls below a tolerance (typically :math:`10^{-15}`)
2. The relative change in force residual between iterations becomes negligible
3. The magnetic field reaches a steady state (no significant change between time steps)

**Regularization (Optional)**

For smoother convergence, a regularization operator :math:`(I - \Delta)^{-\gamma}` can be applied to the velocity:

.. math::

    v_h = (I - \Delta)^{-\gamma} \, P_{\mathrm{Leray}} \, \Pi^2_0 (\mathbf{J}_h \times \mathbf{H}_h)

where :math:`\gamma \geq 0` is a regularization parameter. Typically :math:`\gamma = 0` (no regularization) is used.

