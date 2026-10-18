"""The geometry of a de Rham sequence, and the one call that builds a sequence from a geometry file.

A sequence lives on the logical cube ``(r, theta, zeta)``. The map ``Phi`` from the cube to physical
space enters the discrete operators only through its Jacobian ``DPhi``: :class:`SequenceGeometry` holds
the metric ``G = DPhi^T DPhi``, its inverse and ``det DPhi`` at the quadrature points.
The geometry is data (a JAX pytree) that :meth:`~mrx.derham_sequence.DeRhamSequence.set_geometry_from_map` replaces. A new map of the same resolution does not
recompile the sequence's solves, but it drops the preconditioners (unless ``keep_preconditioners``) and the harmonic forms: call
:meth:`~mrx.derham_sequence.DeRhamSequence.build_preconditioners` again.

Most users call :func:`build_sequence`, which reads a geometry file (a GVEC state ``.dat``, a VMEC wout
``.nc`` or a DESC output ``.h5``), builds the sequence, installs the map and builds the preconditioners.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

import mrx
from mrx.differential_forms import inv33
from mrx.spline_bases import knot_vector


def map_jacobian_at(Phi: Callable, x: jnp.ndarray) -> jnp.ndarray:
    """The Jacobian ``DPhi[q, i, j] = dPhi_i/dx_j`` of the map ``Phi`` at the logical points ``x``, ``(N, 3)``."""
    return jax.lax.map(jax.jacfwd(Phi), x, batch_size=mrx.MAP_BATCH_SIZE_INNER)


class SequenceGeometry(eqx.Module):
    """The geometry of a sequence at its quadrature points: the metric ``metric_jkl = DPhi^T DPhi``, its
    inverse ``metric_inv_jkl`` (both ``(N_q, 3, 3)``) and ``jacobian_j = det DPhi`` (``(N_q,)``).

    The sequence fills in the mass weights (:func:`mrx.mass.attach_weights`) when the geometry is
    installed. ``DPhi`` itself is not kept, so code that needs it recomputes it from ``map``."""

    map: Any
    metric_jkl: jnp.ndarray = None
    metric_inv_jkl: jnp.ndarray = None
    jacobian_j: jnp.ndarray = None
    mass_weights: Optional[dict] = None

    @classmethod
    def from_map(cls, Phi: Callable, quad_x: jnp.ndarray) -> "SequenceGeometry":
        """The geometry of a differentiable map ``Phi: R^3 -> R^3`` at the quadrature points ``quad_x``."""
        DPhi_jkl = map_jacobian_at(Phi, quad_x)
        metric_jkl = jnp.einsum("qki,qkj->qij", DPhi_jkl, DPhi_jkl)
        return cls(Phi, metric_jkl, jax.vmap(inv33)(metric_jkl), jnp.linalg.det(DPhi_jkl))


#: The symmetry a map is assumed to have. In logical ``(r, theta, zeta)``, ``zeta`` in ``[0, 1]`` is one
#: field period.
#: ``"stellarator"``: ``nfp`` field periods and ``Phi(r, -theta, -zeta) = S Phi(r, theta, zeta)`` with
#: ``S = diag(1, -1, -1)``. A fitted map is projected onto this symmetry and the quadrature covers only
#: half a period, which halves the number of quadrature points.
#: ``"field-period"``: ``nfp`` field periods and no further symmetry.
#: ``"none"``: ``zeta`` in ``[0, 1]`` is the whole torus and ``nfp = 1``.
SYMMETRIES = ("stellarator", "field-period", "none")


def build_sequence(geometry, ns, p, maxiter=10_000, tol=None, nfp=None, knots=None,
                   symmetry="stellarator"):
    """Build the de Rham sequence of the geometry file ``geometry`` and return it.

    The sequence has its map installed and its preconditioners built (``seq.operators``). The
    harmonic forms are not computed here. Call :func:`mrx.nullspace.compute_nullspaces` next if you need
    them. The parsed file is kept as ``seq.equilibrium`` (the output of
    :func:`mrx.equilibria.read_equilibrium`).

    Args:
        ns: ``(n_r, n_theta, n_zeta)``, the number of splines per axis, also used for the map. An axis
            with breakpoints in ``knots`` takes its number from them instead.
        p: spline degree in all directions. The quadrature uses ``p + 1`` Gauss points per cell.
        maxiter, tol: iteration limit and relative tolerance of every solve on the sequence
            (``tol=None`` means :data:`mrx.precision.SOLVE_TOL`).
        nfp: overrides the ``nfp`` of the equilibrium file.
        knots: ``(r, theta, zeta)`` breakpoints (see :func:`mrx.spline_bases.knot_vector`), ``None`` for a uniform axis.
        symmetry: one of :data:`SYMMETRIES`, stored as ``seq.symmetry`` next to ``seq.nfp``. The
            half-period quadrature is only used with uniform angular knots. ``"none"`` needs ``nfp = 1``
            (see :class:`~mrx.derham_sequence.DeRhamSequence`).
    """
    from mrx.derham_sequence import AXIS_TYPES, DeRhamSequence      # both import this module
    from mrx.equilibria import build_map, is_stellarator_symmetric, read_equilibrium

    t0 = time.perf_counter()
    eq = read_equilibrium(geometry)
    nfp = eq["nfp"] if nfp is None else int(nfp)
    if symmetry == "stellarator" and not is_stellarator_symmetric(eq):
        raise ValueError(f"{geometry} is not stellarator-symmetric: use symmetry='field-period'")
    bps = tuple(knots) if knots is not None else (None, None, None)
    Ts = tuple(None if bp is None else knot_vector(bp, p, t == "periodic") for bp, t in zip(bps, AXIS_TYPES))
    ns = tuple(n if bp is None else len(bp) - 1 + (p if t == "clamped" else 0)
               for n, bp, t in zip(ns, bps, AXIS_TYPES))
    seq = DeRhamSequence(ns, p, nfp=nfp, symmetry=symmetry, knots=Ts, equilibrium=eq, tol=tol, maxiter=maxiter)
    # on a half-period sequence the map is projected onto the symmetry (see mrx.symmetry)
    Phi, info = build_map(eq, seq, nfp=seq.nfp, stellarator_symmetric=seq.half_period)
    defect = f"defect {info['symmetry_defect']:.1e}"
    note = (f"map projected, {defect}, half-period quadrature" if seq.half_period
            else f"{defect}, not exploited: non-uniform angular knots" if symmetry == "stellarator"
            else defect)
    reversed_note = " (theta reversed, the file is left-handed)" if eq["theta_reversed"] else ""
    print(f"[geom] {geometry}: nfp={info['nfp']}{reversed_note}, "
          f"det DPhi in [{info['det_range'][0]:.3e}, {info['det_range'][1]:.3e}], "
          f"symmetry {symmetry} ({note})", flush=True)
    seq.set_geometry_from_map(Phi)
    seq.build_preconditioners()
    print(f"[geom] sequence {seq.ns} p={p} built in {time.perf_counter() - t0:.0f} s", flush=True)
    return seq
