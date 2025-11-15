Toroid Poisson Problem
=======================

This tutorial demonstrates solving a Poisson problem on a toroidal domain.
The script is located at ``scripts/tutorials/toroid_poisson.py``.

The script demonstrates:

- Setting up finite element spaces on a toroidal domain
- Using toroidal mappings
- Solving Poisson equations in toroidal geometry
- Convergence analysis

To run the script:

.. code-block:: bash

    python scripts/tutorials/toroid_poisson.py

The script generates convergence plots showing error vs. mesh size.

Full script:

.. literalinclude:: ../../scripts/tutorials/toroid_poisson.py
   :language: python
   :linenos:

