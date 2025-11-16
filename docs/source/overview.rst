Overview
========

MRX is a differentiable 3D magnetohydrodynamic (MHD) equilibrium solver that allows for magnetic relaxation
of magnetic fields in perturbed/non-minimum energy states to lower energy states. The code is designed to
address traditional challenges to 3D MHD equilibrium solvers, including exactly enforcing physical constraints
such as divergence-free magnetic fields, exhibiting high levels of numerical convergence, dealing with complex
geometries, and modeling stochastic field lines or chaotic behavior. By using differentiable Python (JAX), the
numerical method provides computational efficiency on modern computing architectures, high code accessibility,
and differentiability at each step.

The method is based on the concept of **admissible variations** of :math:`\mathbf{B}` and :math:`p` that allows
for magnetic relaxation without the assumption of nested flux surfaces. This makes MRX particularly suitable for
studying magnetic islands and chaos in stellarator fusion devices.

For more details, see the paper: `MRX: A differentiable 3D MHD equilibrium solver without nested flux surfaces <https://arxiv.org/pdf/2510.26986>`_.

.. toctree::
   :maxdepth: 1
   :caption: Overview Sections:

   overview_usage
   overview_mathematical
   overview_discretization
   overview_geometry
   overview_relaxation

For more details on specific scripts, see the individual script documentation pages.
