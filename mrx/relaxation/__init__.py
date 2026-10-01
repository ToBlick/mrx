"""Magnetic relaxation: driving a field towards a magnetostatic equilibrium.

- :mod:`~mrx.relaxation.physics` computes the Lorentz force, the pressures, the helicity and the
  resistive step.
- :mod:`~mrx.relaxation.loop` holds the time stepper and :func:`~mrx.relaxation.loop.relax`, the
  relaxation loop that also writes the checkpoints.
- :mod:`~mrx.relaxation.newton` computes the Newton direction from the second variation of the energy.
- :mod:`~mrx.relaxation.initial_conditions` builds the starting field and
  :mod:`~mrx.relaxation.seeding` adds island seeds to it.
- :mod:`~mrx.relaxation.config` is the run configuration used by ``scripts/relax.py``.
"""
