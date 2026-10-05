API reference
=============

The modules a script or a tutorial calls. The assembly and solver internals
(``mrx.operators``, ``mrx.mass``, ``mrx.metric_lumping``, ...) are
documented in their source and in the concept pages.

.. toctree::
   :maxdepth: 1
   :caption: Discretisation

   geometry
   mappings
   derham_sequence
   differential_forms
   symmetry
   nullspace
   solvers
   precision

.. toctree::
   :maxdepth: 1
   :caption: Equilibria

   equilibria

.. toctree::
   :maxdepth: 1
   :caption: Relaxation

   relaxation.physics
   relaxation.loop
   relaxation.newton
   relaxation.initial_conditions
   relaxation.seeding
   relaxation.config
   relaxation.pressure_loop
   flux_label

.. toctree::
   :maxdepth: 1
   :caption: Diagnostics

   diagnostics.poincare
   diagnostics.islands
   diagnostics.plotting

.. toctree::
   :maxdepth: 1
   :caption: Optimization

   optimization.shape_ad
