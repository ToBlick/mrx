"""The initial magnetic field of a relaxation run.

:func:`initial_field` is the entry point. It returns the DoFs of a 2-form ``B`` with zero normal component on
the wall in the file's units (tesla for a field on a map in metres), built from the equilibrium file the sequence
was read from (``seq.equilibrium``, see :mod:`mrx.equilibria`). In logical coordinates it is a field with nested flux surfaces,

    det(DPhi) (B^r, B^theta, B^zeta) = Psi'(r) (0, iota(r) - d lambda / d zeta, 1 + d lambda / d theta),

where ``Psi`` is the toroidal flux and ``lambda`` the angle shift to straight field lines. Such a field has
``B^r = 0``, ``div B = 0`` and ``B . n = 0`` for any ``lambda`` and any geometry.

The discrete field is the discrete curl of a vector potential, so ``div B = 0`` holds exactly
(:func:`potential_two_form`). On a map re-sectioned into the planes normal to its axis
(:func:`mrx.relaxation.mesh_motion.axis_normal_map`, the default of :func:`mrx.geometry.build_sequence`) ``lambda``
is that of the new toroidal coordinate, so the field is the file's (:func:`_clebsch_potential`).

A state without a field, the map of a VMEC input namelist filled from its boundary (``kind`` ``"indata"``), starts
from the vacuum field instead: the harmonic 2-form with the namelist's toroidal flux ``PHIEDGE``
(:func:`vacuum_field`). It has no current and no pressure.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline

from mrx.equilibria import StateField
from mrx.relaxation.physics import compute_divergence_norm


def _clebsch_potential(st, nfp, basis):
    """The vector potential ``A' = (-lambda' Psi', 2 pi Psi, -(2 pi / nfp) chi)`` of the state ``st`` as a
    function of the logical coordinates of the map, with ``Psi(0) = chi(0) = 0`` and ``chi`` the poloidal flux.

    ``st`` is in GVEC units with ``lambda`` in radians. The curl of ``A'`` is the state's field up to a common
    factor ``2 pi``. Only the values of ``lambda`` are used. Its derivatives come from the discrete curl.

    On the file's sections ``lambda' = lambda``. On a map re-sectioned by
    :func:`mrx.relaxation.mesh_motion.axis_normal_map` the point ``(r, theta, zeta)`` is the file's point
    ``(r, theta, zeta + f)``, with the shift ``f`` the tensor coefficients ``st["zeta_shift"]`` in ``basis``, and
    ``lambda'(r, theta, zeta) = lambda(r, theta, zeta + f) - iota(r) (2 pi / nfp) f``. The straight-field-line angles
    of the file, ``theta + lambda`` and the file's toroidal angle, become ``theta + lambda'`` and ``zeta``: they
    differ by ``iota`` times the same shift of the toroidal angle, which moves no field line. So the field is the
    file's, not dragged by the slide of the points."""
    psi, iota = st["profiles"]["phi"], st["profiles"]["iota"]
    r = np.linspace(0.0, 1.0, 2001)
    dPsi = psi.derivative()(r)
    Psi = jnp.asarray(psi(r) - psi(0.0))
    chi = jnp.asarray(CubicSpline(r, iota(r) * dPsi).antiderivative()(r))
    io = jnp.asarray(iota(r))
    r, dPsi = jnp.asarray(r), jnp.asarray(dPsi)
    lam = StateField(st["LA"], nfp)
    shift = st.get("zeta_shift")
    two_pi = 2.0 * jnp.pi

    def lam_mesh(x):
        if shift is None:
            return lam(x)
        f = basis.contract(shift, x)
        return lam(x.at[2].add(f)) - jnp.interp(x[0], r, io) * two_pi / nfp * f

    def A_ref(x):
        return jnp.array([-lam_mesh(x) * jnp.interp(x[0], r, dPsi),
                          two_pi * jnp.interp(x[0], r, Psi),
                          -two_pi / nfp * jnp.interp(x[0], r, chi)])
    return A_ref


def potential_two_form(seq):
    """The field of the equilibrium file as the discrete curl of its vector potential. Returns ``(B, wall)``.

    ``B`` is in the file's units. Its flux through a section ``zeta = const`` is the file's toroidal flux. ``wall`` is the
    relative size of the normal component on the wall that is dropped when ``B`` is restricted to fields
    with ``B . n = 0``. It serves as a check and should be close to zero, because the tangential part of ``A'``
    on the wall depends on ``r`` alone."""
    seq = seq.odd
    free = seq.free
    # interpolate A' without a boundary condition: its tangential part on the wall carries the toroidal flux
    A = free.interpolate(_clebsch_potential(seq.equilibrium, seq.nfp, seq.basis_0.bases[0]), 1)
    B_full = free.G[1] @ A
    B = seq.restrict(B_full, 2)
    n_full, norm = float(free.l2_norm(B_full, 2)), float(seq.l2_norm(B, 2))
    wall = abs(n_full ** 2 - norm ** 2) ** 0.5 / norm
    return B, wall


def toroidal_flux(seq, B):
    """The flux of the 2-form ``B`` through a section ``zeta = const``, in the file's units (Wb for a map in metres
    and a field in tesla): the volume sum of its ``zeta`` density over the ``zeta`` extent of the quadrature."""
    return jnp.sum(seq.quad.w * seq.odd.evaluate_at_quadrature(B, 2)[:, 2]) / jnp.sum(seq.quad.w_z)


def vacuum_field(seq, flux):
    """The vacuum field of the domain with the toroidal flux ``flux``: the harmonic Dirichlet 2-form
    (``seq.odd.nullspace(2)``, which :func:`mrx.nullspace.compute_nullspaces` fills) scaled to it."""
    h = seq.odd.nullspace(2)[0]
    return h * (flux / toroidal_flux(seq, h))


def initial_field(seq):
    """The initial field of the sequence's geometry file. Returns ``(B, info)``.

    ``info`` holds numbers worth recording: ``kind``, the divergence ``div``, ``nfp``,
    ``iota_axis`` and ``iota_edge`` (per full turn) and ``wall_discarded``, or for the vacuum field of a namelist
    ``kind``, ``nfp``, ``flux`` and ``div``."""
    eq = seq.equilibrium
    if eq["kind"] == "indata":
        B = vacuum_field(seq, eq["indata"].phiedge)
        return B, dict(kind="indata", nfp=int(seq.nfp), flux=float(toroidal_flux(seq, B)),
                       div=float(compute_divergence_norm(B, seq)))
    B, wall = potential_two_form(seq)
    iota = eq["profiles"]["iota"]
    return B, dict(kind=eq["kind"], nfp=int(seq.nfp), iota_axis=float(iota(0.0)), iota_edge=float(iota(1.0)),
                   wall_discarded=float(wall),
                   div=float(compute_divergence_norm(B, seq)))
