Mathematical Formulation
========================

**Notation**: Throughout this section:
- Boldface :math:`\mathbf{\cdot}` denotes vector quantities in physical space (e.g., :math:`\mathbf{B}`, :math:`\mathbf{J}`)
- Quantities without subscripts are continuous (not discretized)
- See the :doc:`overview_discretization` page for complete notation conventions including hats for logical domain and subscript :math:`h` for discretized quantities

**Magnetohydrostatic (MHS) Problem**

MRX solves the three-dimensional, static, ideal magnetohydrodynamic equilibrium problem, which seeks
a magnetic field :math:`\mathbf{B}: \Omega \to \mathbb{R}^3` on a bounded Lipschitz domain :math:`\Omega \subset \mathbb{R}^3`
such that:

.. math::

    \mathbf{J} \times \mathbf{B} = \operatorname{grad} p, \quad \operatorname{div} \mathbf{B} = 0

where :math:`\mathbf{J} = \operatorname{curl}(\mathbf{B})` is the current density (taking units where vacuum permeability :math:`\mu_0 = 1`),
and :math:`p` denotes the plasma pressure. This is the **magnetohydrostatic (MHS) problem**.

**Boundary Conditions**

The boundary conditions used in MRX are guided by the requirements of the variational formulation:

.. math::

    \mathbf{B} \cdot \mathbf{n} = 0, \quad \mathbf{J} \times \mathbf{n} = 0 \quad \mathrm{on\ } \partial \Omega

where :math:`\mathbf{n}` is the unit vector normal to the boundary :math:`\partial \Omega`.

**Magnetic Relaxation Method**

MRX uses a magnetic relaxation approach based on **admissible variations** of :math:`\mathbf{B}` and :math:`p`.
The method ensures certain bounds on the magnetic energy and uses differential geometry to transform
between logical and physical domains. The relaxation process allows a magnetic field in a perturbed or
non-minimum energy state to evolve to a lower energy equilibrium state.

**Key Features**

- **Exactly divergence-free magnetic fields**: The divergence constraint :math:`\operatorname{div} \mathbf{B} = 0` is enforced exactly
  at the discrete level through the use of appropriate finite element spaces (2-forms in the DeRham sequence)
- **No nested flux surface assumption**: Unlike traditional codes (VMEC, NSTAB, GVEC, DESC), MRX does not assume
  nested magnetic flux surfaces exist, making it suitable for studying magnetic islands and chaotic field lines
- **Differential geometry**: The code uses differential geometry to transform between logical coordinates
  :math:`(r, \theta, \zeta) \in [0,1]^3` and physical coordinates via a mapping function :math:`F: [0,1]^3 \to \Omega`
- **Differentiability**: Built on JAX, the entire computation graph is differentiable, enabling gradient-based
  optimization and sensitivity analysis

