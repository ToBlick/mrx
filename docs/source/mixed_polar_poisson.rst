Mixed Polar Poisson Problem
============================

This tutorial demonstrates solving a mixed formulation of the Poisson problem on a disc.
The script is located at ``scripts/tutorials/mixed_polar_poisson.py``.

The mixed formulation rewrites the Poisson equation as a system:

.. math::

    \nabla \cdot \sigma = f \\
    -\nabla u = \sigma

The script demonstrates:

- Mixed finite element formulation
- Handling polar coordinates and axis singularity
- Convergence analysis for mixed methods
- Performance comparison with standard formulation

To run the script:

.. code-block:: bash

    python scripts/tutorials/mixed_polar_poisson.py

The script generates convergence plots and performance comparisons.

Full script:

.. literalinclude:: ../../scripts/tutorials/mixed_polar_poisson.py
   :language: python
   :linenos:

