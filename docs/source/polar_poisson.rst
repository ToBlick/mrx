Poisson Problem on a disc
=========================

This tutorial demonstrates solving a Poisson problem on a disc using polar coordinates.
The script is located at ``scripts/tutorials/polar_poisson.py``.

For this problem, we consider the source-solution pair $-\Delta u = f$

.. math::

    u(r) = \frac 1 {27} \left( r^3 (3 \log r - 2) + 2 \right),\\
    f(r) = - r \log r

The script demonstrates:

- Setting up finite element spaces with polar coordinates
- Handling the singularity at the axis using polar splines
- Assembling stiffness matrices and projectors
- Solving the Poisson equation and analyzing convergence

To run the script:

.. code-block:: bash

    python scripts/tutorials/polar_poisson.py

The script generates convergence plots showing error vs. mesh size for different polynomial orders.

Full script:

.. literalinclude:: ../../scripts/tutorials/polar_poisson.py
   :language: python
   :linenos: 