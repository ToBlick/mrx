Iterative Island Calculation
============================

.. note::
   For general information about finite element discretization, basis functions, mesh parameters,
   polynomial degrees, boundary conditions, and matrix/operator dimensions, see :doc:`overview`.

This script performs iterative island calculations for magnetic configurations.
The script is located at ``scripts/config_scripts/iter_islands.py``.

**Problem Statement**

This script is similar to ``solovev.py`` but configured for studying magnetic islands
in tokamak configurations. It solves the magnetohydrostatic (MHS) equilibrium equation:

.. math::

    \mathbf{J} \times \mathbf{B} = \nabla p \quad \text{in } \Omega

with boundary conditions:

.. math::

    \mathbf{B} \cdot \mathbf{n} = 0 \quad \text{on } \partial\Omega

and the constraint:

.. math::

    \nabla \cdot \mathbf{B} = 0 \quad \text{in } \Omega

where:
- :math:`\mathbf{B}: \Omega \to \mathbb{R}^3` is the magnetic field (2-form)
- :math:`\mathbf{J} = \nabla \times \mathbf{B}` is the current density (1-form)
- :math:`p: \Omega \to \mathbb{R}` is the plasma pressure (0-form)
- :math:`\mathbf{n}` is the outward unit normal vector on the boundary
- :math:`\Omega` is the tokamak plasma domain

**Tokamak Geometry**

The tokamak uses a cerfon mapping (circular cross-section) with parameters:
- :math:`\epsilon = 0.33`: Inverse aspect ratio
- :math:`\kappa = 1.7`: Elongation parameter
- :math:`\delta = 0.33`: Triangularity parameter

**Initial Magnetic Field**

The initial magnetic field is defined in physical cylindrical coordinates:

.. math::

    B_R &= z R \\
    B_\phi &= \frac{\tau}{R} \\
    B_z &= -\left(\frac{\kappa^2}{2}(R^2 - R_0^2) + z^2\right)

where:
- :math:`\tau = q_* \kappa (1 + \kappa^2) / (\kappa + 1)` is the safety factor parameter
- :math:`R_0 = 1.0` is the major radius
- :math:`\kappa` is the elongation parameter

**Perturbation for Island Seeding**

A perturbation is applied to seed magnetic islands:

.. math::

    \delta\mathbf{B}_r = a(r) \sin(2\pi\theta m_{\mathrm{pol}}) \sin(2\pi\zeta n_{\mathrm{tor}}) \frac{\partial F}{\partial r}

where:
- :math:`a(r) = \exp(-(r - r_0)^2/(2\sigma^2))` is a Gaussian radial profile
- :math:`m_{\mathrm{pol}}` is the poloidal mode number
- :math:`n_{\mathrm{tor}}` is the toroidal mode number
- :math:`r_0` is the radial location of the perturbation
- :math:`\sigma` is the radial width

The perturbation strength is typically small (:math:`\sim 10^{-5}`) to avoid disrupting
the equilibrium while still seeding islands.

Usage:

.. code-block:: bash

    python scripts/config_scripts/iter_islands.py run_name=test_run

The script generates output files with island calculation results.

Code Walkthrough
----------------

**Block 1: Imports and Configuration (lines 1-16)**

Imports modules including ``cerfon_map`` for tokamak boundaries.

.. literalinclude:: ../../scripts/config_scripts/iter_islands.py
   :language: python
   :lines: 1-16

**Block 2: Main Function (lines 19-33)**

Sets default configuration for tokamak geometry:

- ``boundary_type="tokamak"``: Fixed for tokamak
- ``eps=0.33``: Inverse aspect ratio
- ``kappa=1.7``: Elongation
- ``delta=0.33``: Triangularity
- ``q_star=1.54``: Safety factor
- ``pert_strength=2e-5``: Small perturbation amplitude for island seeding
- ``save_every=10``: Save magnetic field snapshots every 10 iterations

.. literalinclude:: ../../scripts/config_scripts/iter_islands.py
   :language: python
   :lines: 19-33

**Block 3: Setup and Initial Condition (lines 36-111)**

The ``run()`` function:

- Creates output directory: ``script_outputs/iter/{run_name}/``
- Sets up cerfon mapping (circular tokamak cross-section)
- Defines initial magnetic field :math:`\mathbf{B}_0(p)` with tokamak-specific scaling
- Defines perturbation ``dB_xyz(p)`` with Gaussian radial profile and
  poloidal/toroidal mode structure
- Projects field into FEM space and applies Leray projection
- Applies perturbation to seed magnetic islands if ``apply_pert_after=0``

.. literalinclude:: ../../scripts/config_scripts/iter_islands.py
   :language: python
   :lines: 36-111

**Block 4: Magnetic Relaxation Loop (lines 114-116)**

Runs relaxation with perturbation function available for dynamic application.

.. literalinclude:: ../../scripts/config_scripts/iter_islands.py
   :language: python
   :lines: 114-116

**Block 5: Post-processing (lines 119-142)**

Saves final state and generates plots. The perturbation creates magnetic islands
that can be tracked through the relaxation process.

The script is designed to study how magnetic islands evolve during relaxation,
which is important for understanding plasma stability and transport in tokamaks.

.. literalinclude:: ../../scripts/config_scripts/iter_islands.py
   :language: python
   :lines: 119-142
