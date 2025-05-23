Conjugate Gradient Method for Magnetic Field Relaxation
=======================================================

This tutorial walks through a script that implements a conjugate gradient method for relaxing magnetic fields while preserving magnetic helicity. The script demonstrates field setup, relaxation, error analysis, and conservation verification.

Introduction
------------
We use a conjugate gradient method to relax magnetic fields while preserving magnetic helicity. The script demonstrates initial field setup, relaxation, error analysis, and conservation properties.

Setup and Imports
-----------------
We start by importing the necessary libraries and setting up the output directory:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from pathlib import Path
    from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pullback
    from mrx.Quadrature import QuadratureRule
    from mrx.Projectors import Projector, CurlProjection
    from mrx.LazyMatrices import LazyMassMatrix, LazyDerivativeMatrix, LazyProjectionMatrix, LazyDoubleCurlMatrix, LazyStiffnessMatrix
    from mrx.Utils import curl

    jax.config.update("jax_enable_x64", True)
    output_dir = Path("script_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

Main Steps
----------
- Set up initial field and perturbation
- Perform conjugate gradient relaxation
- Analyze errors and visualize results
- Verify conservation of helicity and energy

Conclusion
----------
This script demonstrates how to use a conjugate gradient method to relax magnetic fields, analyze conservation properties, and visualize results.

**How to run:**
.. code-block:: bash

    python scripts/conjugate.py 