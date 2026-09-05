"""Geometry evaluation and map interpolation for mapped de Rham sequences.

Provides two paths for evaluating the map Jacobian ``DF`` at the quadrature
points and reducing it to the stored geometry (metric, inverse metric,
determinant):

- Generic path (:meth:`SequenceGeometry.from_map`): works for any
  differentiable map via ``jax.jacfwd``.
- Sum-factorized fast path (:meth:`SequenceGeometry.from_spline_map`):
  requires the map to be a tensor-product spline.  Avoids the black-box
  ``jacfwd`` pass by exploiting the Kronecker structure of the 1D basis
  evaluations stored on a :class:`~mrx.derham_sequence.DeRhamSequence`.

Also provides :func:`greville_interpolate_map`, which interpolates an analytic
map onto the spline basis of a sequence.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import mrx
from mrx.differential_forms import inv33


def grad_1d(d_basis, boundary_type):
    """Lift a derivative spline basis back to the scalar-basis space.

    Args:
        d_basis: ``(n-1, nq)`` or ``(n, nq)`` derivative basis values.
        boundary_type: ``'clamped'`` or ``'periodic'``.

    Returns:
        ``(n, nq)`` array suitable for contraction with the raw TP coefficient
        grid (same leading dimension as the primal basis).
    """
    if boundary_type == 'clamped':
        padded = jnp.pad(d_basis, ((1, 1), (0, 0)))
        return padded[:-1] - padded[1:]
    else:  # periodic
        return jnp.roll(d_basis, 1, axis=0) - d_basis


# ---------------------------------------------------------------------------
# Generic (jacfwd) path
# ---------------------------------------------------------------------------

def map_jacobian_at(map: Callable, x: jnp.ndarray) -> jnp.ndarray:
    """``DF`` of ``map`` at every point of ``x`` via batched ``jacfwd``.

    This is the only place the raw Jacobian is materialised on a point set.
    :meth:`SequenceGeometry.from_map` reduces it to the stored geometry at
    once; the physical-frame pullback (:func:`mrx.projectors.load` with
    ``frame='phys'``) calls it on demand at load time, since ``DF`` itself
    is not kept on the sequence.

    Args:
        map: Differentiable logical-to-physical map ``F: R^3 -> R^3``.
        x: Points, shape ``(N, 3)``.

    Returns:
        ``(N, 3, 3)`` with ``DF[q, i, j] = dF_i/dx_j``.
    """
    return jax.lax.map(jax.jacfwd(map), x, batch_size=mrx.MAP_BATCH_SIZE_INNER)


# ---------------------------------------------------------------------------
# SequenceGeometry
# ---------------------------------------------------------------------------

class SequenceGeometry(eqx.Module):
    """Geometry data attached to a de Rham sequence.

    An ``eqx.Module`` so that the quadrature-grid arrays are dynamic pytree
    leaves and can flow through ``jit`` / ``grad``. ``map`` is kept as a
    normal field so that if it is itself a pytree (e.g. a
    :class:`~mrx.mappings.SplineMap`), its coefficient leaves are tracked;
    plain ``Callable`` maps are treated as opaque leaves.

    Stored per quadrature point, built ONCE from ``DF`` by the constructors
    and never recomputed: the metric ``metric_jkl = DF^T DF`` ``(N_q, 3, 3)``,
    its inverse ``metric_inv_jkl`` ``(N_q, 3, 3)`` and the determinant
    ``jacobian_j = det DF`` ``(N_q,)``.  These are what every hot path wants
    -- the mass weights ``J``, ``J G^-1``, ``G/J``, ``1/J`` are elementwise
    products of them, ``cross_product_load`` contracts against ``G`` and
    ``G^-1`` on every force step, and the lumped preconditioners read them in
    the quadrature tensor layout.  ``DF`` itself is discarded: its only
    consumers are the physical-frame pullbacks at load time, which recompute
    it with :func:`map_jacobian_at`.
    """

    map: Any
    metric_jkl: jnp.ndarray = None
    metric_inv_jkl: jnp.ndarray = None
    jacobian_j: jnp.ndarray = None

    @classmethod
    def from_DF(cls, map, DF_jkl: jnp.ndarray) -> "SequenceGeometry":
        """Reduce the Jacobian on the quadrature grid to the stored geometry.

        Args:
            map: The map ``DF_jkl`` was evaluated from.
            DF_jkl: ``(N_q, 3, 3)`` with ``DF[q, i, j] = dF_i/dx_j``.

        Returns:
            A fully populated :class:`SequenceGeometry`.
        """
        metric_jkl = jnp.einsum("qki,qkj->qij", DF_jkl, DF_jkl)
        return cls(map, metric_jkl, jax.vmap(inv33)(metric_jkl),
                   jnp.linalg.det(DF_jkl))

    @classmethod
    def from_map(cls, map: Callable, quad_x: jnp.ndarray) -> "SequenceGeometry":
        """Build geometry by evaluating a map on the quadrature grid.

        Args:
            map: Differentiable logical-to-physical map ``F: R^3 -> R^3``.
            quad_x: Quadrature points, shape ``(N_q, 3)``.

        Returns:
            A fully populated :class:`SequenceGeometry`.
        """
        return cls.from_DF(map, map_jacobian_at(map, quad_x))

    @classmethod
    def from_spline_map(cls, spline_map, seq) -> "SequenceGeometry":
        """Sum-factorized geometry builder for tensor-product spline maps.

        Uses ``seq.basis_{r,t,z}_jk`` / ``seq.d_basis_{r,t,z}_jk`` on the
        map's raw coefficient grid.

        Args:
            spline_map: A :class:`~mrx.mappings.SplineMap`.
            seq: A :class:`~mrx.derham_sequence.DeRhamSequence` with.

        Returns:
            A fully populated :class:`SequenceGeometry`.
        """
        DF_jkl = spline_map_DF_at_quad(spline_map, seq)
        return cls.from_DF(spline_map, DF_jkl)


# ---------------------------------------------------------------------------
# Sum-factorized spline fast path
# ---------------------------------------------------------------------------

def _tp_evaluate(C_raw, M1, M2, M3):
    """Sum-factorized evaluation of a tensor-product spline.

    Computes ``sum_{a,b,c} C_raw[i,a,b,c] * M1[a,I] * M2[b,J] * M3[c,K]``
    in three sequential contractions, at cost
    ``O(N_q (n_r + n_t + n_z))`` rather than ``O(N_q n_r n_t n_z)``.

    Args:
        C_raw: ``(3, nr, nt, nz)`` coefficient array.
        M1: ``(nr, nqr)`` basis or derivative-basis matrix in r.
        M2: ``(nt, nqt)`` basis or derivative-basis matrix in t.
        M3: ``(nz, nqz)`` basis or derivative-basis matrix in z.

    Returns:
        ``(3, nqr, nqt, nqz)`` array of evaluated values.
    """
    T1 = jnp.einsum("iabc,aI->iIbc", C_raw, M1)
    T2 = jnp.einsum("iIbc,bJ->iIJc", T1, M2)
    return jnp.einsum("iIJc,cK->iIJK", T2, M3)


def spline_map_DF_at_quad(spline_map, seq):
    """``DF`` at the sequence's quadrature grid, ``(N_q, 3, 3)`` (axis 1 the
    Cartesian component, axis 2 the logical direction), by sum factorisation
    of ``spline_map.raw`` against the precomputed 1D basis / derivative
    values ``seq.basis_{r,t,z}_jk`` and ``seq.d_basis_{r,t,z}_jk``.
    """
    Br, Bt, Bz = seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk
    # The stored d_basis_*_jk live in the (p-1)-degree derived space and
    # have a different leading dimension than basis_*_jk; grad_1d lifts
    # them back to the full (nr, nt, nz) scalar-basis space so we can
    # contract against the same raw coefficient grid.
    types = seq.basis_0.types
    Dr = grad_1d(seq.d_basis_r_jk, types[0])
    Dt = grad_1d(seq.d_basis_t_jk, types[1])
    Dz = grad_1d(seq.d_basis_z_jk, types[2])

    C_raw = spline_map.raw

    dF_dx1 = _tp_evaluate(C_raw, Dr, Bt, Bz)
    dF_dx2 = _tp_evaluate(C_raw, Br, Dt, Bz)
    dF_dx3 = _tp_evaluate(C_raw, Br, Bt, Dz)

    # (3, nqr, nqt, nqz) flattens straight into the r-major seq.quad.x order.
    return jnp.stack(
        [dF_dx1.reshape(3, -1), dF_dx2.reshape(3, -1), dF_dx3.reshape(3, -1)],
        axis=-1,
    ).transpose(1, 0, 2)                                   # (N_q, 3, 3)


# ---------------------------------------------------------------------------
# Map interpolation onto spline DOFs
# ---------------------------------------------------------------------------

def greville_interpolate_map(F_analytic: Callable, seq) -> jnp.ndarray:
    """Interpolate an analytic map to spline coefficients via Greville collocation.

    Evaluates each Cartesian component of ``F_analytic`` at the
    tensor-product Greville points and solves the resulting 1-D collocation
    systems, returning a coefficient array suitable for
    :meth:`~mrx.derham_sequence.DeRhamSequence.set_spline_map`.

    Args:
        F_analytic: Analytic map ``F: R^3 -> R^3`` mapping logical coordinates
            ``(r, θ, ζ) ∈ [0, 1]^3`` to physical Cartesian coordinates
            ``(X, Y, Z)``.
        seq: :class:`~mrx.derham_sequence.DeRhamSequence` to interpolate into.
            Polar and periodic sequences
            are supported (the polar rows are restricted conformingly, see
            :func:`mrx.projectors.interpolate`).

    Returns:
        Coefficient array of shape ``(3, seq.n(0))`` — spline DOF vectors for
        the three Cartesian components stacked along axis 0.  Pass directly
        to ``seq.set_spline_map(coefficients)``.
    """
    component_dofs = [
        seq.interpolate(lambda x, i=i: F_analytic(x)[i], 0)
        for i in range(3)
    ]
    return jnp.stack(component_dofs, axis=0)


# ---------------------------------------------------------------------------
# Solve geometries for the driver scripts, by analytic name or by file path
# ---------------------------------------------------------------------------
#
# A geometry is a file: a GVEC state (``.dat``, :mod:`mrx.gvec`) or a VMEC
# wout (``.nc``, :mod:`mrx.vmec`), both read in closed form, or an analytic
# geometry (``.json``): a map of :mod:`mrx.mappings` with its parameters and
# the profiles of the analytic initial condition (``scripts/relax.py``).
# ``data/torus.json``, ``data/cylinder.json`` and ``data/rot_ellipse.json``
# are the shipped ones. :func:`build_sequence` turns a geometry into a polar
# sequence with the map installed and the preconditioners built; nullspaces
# are left to the caller.

#: The maps an analytic geometry file may name.
ANALYTIC_MAPS = ("torus", "cylinder", "rot-ellipse")


def read_analytic(path):
    """The analytic geometry file ``path``::

        {"map": "torus" | "cylinder" | "rot-ellipse",
         "map_params": {...},                       # keyword arguments of the map
         "profile": {"iota": [i0, i1], "iota_exp": e, "flux_exp": q,
                     "lambda": [[m, n, amp], ...]}}  # the logical-grid field

    ``map_params`` are ``toroid_map``'s ``epsilon, kappa, R0``,
    ``cylinder_map``'s ``a, h`` or ``rotating_ellipse_map``'s ``eps, kappa,
    R0, nfp``; the profile is ``iota = i0 + (i1 - i0) rho^e``, ``Phi' =
    rho^q`` and ``lambda = sum amp rho^|m| sin(2 pi (m theta - n zeta))``
    (``mrx.initial_conditions``).
    """
    with open(path) as fh:
        spec = json.load(fh)
    if spec.get("map") not in ANALYTIC_MAPS:
        raise ValueError(f"{path}: 'map' must be one of {', '.join(ANALYTIC_MAPS)}, "
                         f"got {spec.get('map')!r}")
    return spec


def geometry_kind(geometry):
    """``"gvec"`` for a state file, ``"vmec"`` for a wout, the map's name for
    an analytic geometry file; anything else raises."""
    if not os.path.isfile(geometry):
        raise ValueError(f"geometry {geometry!r} is not a file; MRX reads GVEC state files "
                         "(.dat), VMEC wout files (.nc) and analytic geometry files (.json)")
    if geometry.endswith(".dat"):
        return "gvec"
    if geometry.endswith(".nc"):
        return "vmec"
    if geometry.endswith(".json"):
        return read_analytic(geometry)["map"]
    raise ValueError(f"{geometry}: not a geometry file; MRX reads GVEC state files (.dat), "
                     "VMEC wout files (.nc) and analytic geometry files (.json)")


def geometry_nfp(geometry, nfp=None):
    """Field periods of a geometry.

    Args:
        geometry: the path of an equilibrium or analytic geometry file.
        nfp: overrides an equilibrium file's ``nfp``; ignored for an
            analytic geometry, whose map fixes it.

    Returns:
        The number of field periods spanned by logical zeta in [0, 1].
    """
    kind = geometry_kind(geometry)
    if kind == "gvec":
        if nfp is not None:
            return int(nfp)
        from mrx.gvec import read_state  # noqa: PLC0415  (imports this module)
        return read_state(geometry)["nfp"]
    if kind == "vmec":
        if nfp is not None:
            return int(nfp)
        from mrx.vmec import read_nfp  # noqa: PLC0415  (imports this module)
        return read_nfp(geometry)
    return int(read_analytic(geometry)["map_params"].get("nfp", 1))


def parse_r_refine(spec):
    """``"a:b:m,a:b:m"`` -> ``[(a, b, m), ...]``: radial windows ``[a, b]``
    that get ``m`` uniform cells each (:func:`radial_knots`); ``""`` -> ``[]``."""
    windows = []
    for w in filter(None, spec.split(",")):
        a, b, m = w.split(":")
        windows.append((float(a), float(b), int(m)))
    return windows


def radial_knots(n_r, p, windows):
    """The clamped radial knot vector of ``n_r`` degree-``p`` splines with
    ``n_r - p`` cells: ``m`` uniform cells inside every window ``(a, b, m)``,
    and the remaining cells spread over the gaps between and outside the
    windows in proportion to their length (largest-remainder rounding, at
    least one cell per gap). Without windows this is the uniform grid the
    sequence builds itself."""
    windows = sorted(windows)
    n_cells = n_r - p
    inside = sum(m for _, _, m in windows)
    gaps = []
    lo = 0.0
    for a, b, _ in windows:
        gaps.append((lo, a))
        lo = b
    gaps.append((lo, 1.0))
    gaps = [(a, b) for a, b in gaps if b > a]
    free = n_cells - inside
    if free < len(gaps):
        raise ValueError(f"{inside} window cells leave {free} for {len(gaps)} gaps of n_r - p = {n_cells}")
    length = sum(b - a for a, b in gaps)
    raw = [free * (b - a) / length for a, b in gaps]
    counts = [max(1, int(r)) for r in raw]
    order = sorted(range(len(gaps)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    for i in order[: free - sum(counts)]:
        counts[i] += 1
    for i in reversed(order):    # a gap forced up to one cell may have overshot
        if sum(counts) > free and counts[i] > 1:
            counts[i] -= 1
    segments = sorted([(a, b, m) for a, b, m in windows] + [(a, b, c) for (a, b), c in zip(gaps, counts)])
    bp = np.concatenate([np.linspace(a, b, m, endpoint=False) for a, b, m in segments] + [[1.0]])
    return np.concatenate([np.zeros(p), bp, np.ones(p)])


def build_sequence(geometry, ns, p, maxiter=10_000, tol=None, nfp=None, r_windows=()):
    """Build the sequence for a geometry and assemble its solver operators.

    Args:
        geometry:
            a GVEC state file (``.dat``, read in closed form, ``mrx.gvec``),
            a VMEC wout file (``.nc``, refit in closed form, ``mrx.vmec``),
            or an analytic geometry file (``.json``, :func:`read_analytic`).
        ns: ``(n_r, n_theta, n_zeta)``; also the map resolution for a file.
        p: spline degree, all directions; ``p + 1`` Gauss points per knot span.
        maxiter: iteration budget of every solve through the sequence.
        tol: solve tolerance; ``None`` is ``sqrt(eps)`` of working precision.
        nfp: overrides an equilibrium file's ``nfp`` (see ``mrx.gvec``);
            ignored for an analytic geometry.
        r_windows: radial refinement windows ``[(a, b, m), ...]`` for
            :func:`radial_knots`; empty for the uniform radial grid.

    Returns:
        ``(seq, ops)``: the sequence with its geometry installed and every
        preconditioner built (``seq.operators is ops``).

    Raises:
        ValueError: if ``geometry`` is not a file, is of another kind, or
            (from ``set_geometry``) if the map folds.
    """
    from mrx.derham_sequence import DeRhamSequence  # noqa: PLC0415  (imports this module)
    from mrx.gvec import build_gvec_map  # noqa: PLC0415
    from mrx.mappings import cylinder_map, rotating_ellipse_map, toroid_map  # noqa: PLC0415

    knots = (radial_knots(ns[0], p, r_windows), None, None) if r_windows else None
    seq = DeRhamSequence(ns, (p,) * 3, p + 1, ("clamped", "periodic", "periodic"),
                         polar=True, tol=tol, maxiter=maxiter, knots=knots,
                         betti_numbers=(1, 1, 0, 0))
    kind = geometry_kind(geometry)
    if kind in ("gvec", "vmec"):
        map_func, info = build_gvec_map(geometry, seq, nfp=nfp)
        print(f"[geom] {geometry}: nfp={info['nfp']} sign={info['sign']:+.0f} "
              f"det DF in [{info['det_range'][0]:.3e}, "
              f"{info['det_range'][1]:.3e}]", flush=True)
        seq.set_map(map_func)
    else:
        maps = {"torus": toroid_map, "cylinder": cylinder_map, "rot-ellipse": rotating_ellipse_map}
        seq.set_map(maps[kind](**read_analytic(geometry)["map_params"]))
    return seq, seq.build_preconditioners()
