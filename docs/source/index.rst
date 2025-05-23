MRX documentation
=================

``MRX`` is a framework for performing 3D MHD equilibrium solves
without the assumption of nested flux surfaces. 

The design of ``MRX`` is guided by several principles:

- Thorough unit testing, regression testing, and continuous
  integration.
- Extensibility: It should be possible to add new codes and terms to
  the objective function without editing modules that already work,
  i.e. the `open-closed principle <https://en.wikipedia.org/wiki/Open%E2%80%93closed_principle>`_.
  This is because any edits to working code can potentially introduce bugs.
- JAX-based: Everything is written in a way that is compatible with JAX and jit-compilation.

``MRX`` is fully open-source, and anyone is welcome to use it,
make suggestions, and contribute.
We gratefully acknowledge funding from the `Simons Foundation's Hidden
symmetries and fusion energy project
<https://hiddensymmetries.princeton.edu>`_.

``MRX`` is one of several available codes for performing stellarator
optimization.  Others include `VMEC <https://github.com/hiddenSymmetries/VMEC2000>`_,
`DESC <https://github.com/PlasmaControl/DESC>`_, and
`SPEC <https://github.com/PrincetonUniversity/SPEC>`_.
The main difference is that ``MRX`` does not assume nested flux surfaces,
and is designed to be used in a JAX-based optimization framework.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   installation
   tracing
   source
   publications

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   mrx


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Examples
========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   mixed_poisson
   poisson
   Beltrami 
   conjugate 
   cube_relaxation
   oop_splines
   two_d_poisson
   two_d_helicity
   two_d_poisson_mixed
   two_d_helmholtz_decomposition
   three_d_poisson
   polar_helicity
   polar_poisson
   polar_poisson_mixed
   polar_poisson_constantangle
   polar_relaxation
   relaxation
   pullbacks

Unit Tests
==========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

  