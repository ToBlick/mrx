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
   generate_docs
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

Tutorials
==========

.. toctree::
   :maxdepth: 1
   :caption: Tutorials: 

   tutorials/polar_poisson
   tutorials/mixed_polar_poisson
   tutorials/toroid_poisson
   tutorials/harmonics_hollow_torus
   tutorials/gvec_mappings
   notebooks/10_tokamak_equilibrium
   notebooks/11_helicity_minimization
   notebooks/MRX_tokamak_demo

Examples
========

Tutorial Scripts
----------------

.. toctree::
   :maxdepth: 1
   :caption: Tutorial Scripts:

   polar_poisson
   mixed_polar_poisson
   toroid_poisson

Configuration Scripts
---------------------

.. toctree::
   :maxdepth: 1
   :caption: Configuration Scripts:

   solovev
   stell
   iter_islands
   hopf

Interactive Scripts
-------------------

.. toctree::
   :maxdepth: 1
   :caption: Interactive Scripts:

   test_gvec_tokamak
   test_gvec_stellarator
   toroid_poisson_interactive
   toroid_cavity
   cylinder_cavity
   cylinder_vector_poisson
   drumshape
  