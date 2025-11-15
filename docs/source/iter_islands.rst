Iterative Island Calculation
============================

This script performs iterative island calculations for magnetic configurations.
The script is located at ``scripts/config_scripts/iter_islands.py``.

Usage:

.. code-block:: bash

    python scripts/config_scripts/iter_islands.py run_name=test_run

The script generates output files with island calculation results.

Code Walkthrough
================

This script is similar to ``solovev.py`` but configured for studying magnetic islands
in tokamak configurations:

**Block 1: Imports and Configuration (lines 1-16)**
   Imports modules including ``cerfon_map`` for tokamak boundaries.

**Block 2: Main Function (lines 19-33)**
   Sets default configuration for tokamak geometry:
   
   - ``boundary_type="tokamak"``: Fixed for tokamak
   - ``eps=0.33``: Inverse aspect ratio
   - ``kappa=1.7``: Elongation
   - ``delta=0.33``: Triangularity
   - ``delta_B=0.2``: Additional parameter
   - ``q_star=1.54``: Safety factor
   - ``pert_strength=2e-5``: Small perturbation amplitude for island seeding
   - ``save_every=10``: Save magnetic field snapshots every 10 iterations

**Block 3: Setup and Initial Condition (lines 36-111)**
   The ``run()`` function:
   
   - Creates output directory: ``script_outputs/iter/{run_name}/``
   - Sets up cerfon mapping (circular tokamak cross-section)
   - Defines initial magnetic field :math:`\mathbf{B}_0(p)` similar to stellarator but with
     tokamak-specific scaling:
   
     .. math::
     
         B_z = -\left(\frac{\kappa^2}{2}(R^2 - 1^2) + z^2\right)
     
     where :math:`\kappa` is the elongation parameter.
   - Defines perturbation ``dB_xyz(p)`` with Gaussian radial profile and
     poloidal/toroidal mode structure
   - Projects field into FEM space and applies Leray projection
   - Applies perturbation to seed magnetic islands if ``apply_pert_after=0``

**Block 4: Magnetic Relaxation Loop (lines 114-116)**
   Runs relaxation with perturbation function available for dynamic application.

**Block 5: Post-processing (lines 118-142)**
   Saves final state and generates plots. The perturbation creates magnetic islands
   that can be tracked through the relaxation process.

The script is designed to study how magnetic islands evolve during relaxation,
which is important for understanding plasma stability and transport in tokamaks.

Full script:

.. literalinclude:: ../../scripts/config_scripts/iter_islands.py
   :language: python
   :linenos:
