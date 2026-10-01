"""The initial magnetic field of a relaxation run.

:func:`initial_field` is the entry point. It returns the DoFs of a 2-form ``B`` with zero normal component on
the wall, normalised to ``||B||_M = 1``, built from the equilibrium file the sequence was read from
(``seq.equilibrium``, see :mod:`mrx.equilibria`). In logical coordinates it is a field with nested flux surfaces,

    det(DPhi) (B^r, B^theta, B^zeta) = Psi'(r) (0, iota(r) - d lambda / d zeta, 1 + d lambda / d theta),

where ``Psi`` is the toroidal flux and ``lambda`` the angle shift to straight field lines. Such a field has
``B^r = 0``, ``div B = 0`` and ``B . n = 0`` for any ``lambda`` and any geometry.

The discrete field is the discrete curl of a vector potential, so ``div B = 0`` holds exactly
(:func:`potential_two_form`).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline

from mrx.equilibria import StateField


def _clebsch_potential(st, nfp):
    """The vector potential ``A' = (-lambda Psi', 2 pi Psi, -(2 pi / nfp) chi)`` of the state ``st`` as a
    function of logical coordinates, with ``Psi(0) = chi(0) = 0`` and ``chi`` the poloidal flux.

    ``st`` is in GVEC units with ``lambda`` in radians. The curl of ``A'`` is the state's field up to a common
    factor ``2 pi``. Only the values of ``lambda`` are used. Its derivatives come from the discrete curl."""
    psi, iota = st["profiles"]["phi"], st["profiles"]["iota"]
    r = np.linspace(0.0, 1.0, 2001)
    dPsi = psi.derivative()(r)
    Psi = jnp.asarray(psi(r) - psi(0.0))
    chi = jnp.asarray(CubicSpline(r, iota(r) * dPsi).antiderivative()(r))
    r, dPsi = jnp.asarray(r), jnp.asarray(dPsi)
    lam = StateField(st["LA"], nfp)
    two_pi = 2.0 * jnp.pi

    def A_ref(x):
        return jnp.array([-lam(x) * jnp.interp(x[0], r, dPsi),
                          two_pi * jnp.interp(x[0], r, Psi),
                          -two_pi / nfp * jnp.interp(x[0], r, chi)])
    return A_ref


def potential_two_form(seq):
    """The field of the equilibrium file as the discrete curl of its vector potential. Returns
    ``(B, norm, wall)``.

    ``B`` is normalised to ``||B||_M = 1`` and ``norm`` is its norm before normalisation. ``wall`` is the
    relative size of the normal component on the wall that is dropped when ``B`` is restricted to fields
    with ``B . n = 0``. It serves as a check and should be close to zero, because the tangential part of ``A'``
    on the wall depends on ``r`` alone."""
    seq = seq.odd
    free = seq.free
    # interpolate A' without a boundary condition: its tangential part on the wall carries the toroidal flux
    A = free.interpolate(_clebsch_potential(seq.equilibrium, seq.nfp), 1, frame='logical')
    B_full = free.G[1] @ A
    B = seq.restrict(B_full, 2)
    n_full, norm = float(free.l2_norm(B_full, 2)), float(seq.l2_norm(B, 2))
    wall = abs(n_full ** 2 - norm ** 2) ** 0.5 / norm
    return B / norm, norm, wall


def initial_field(seq):
    """The initial field of the sequence's geometry file. Returns ``(B, info)``.

    ``info`` holds numbers worth recording: ``kind``, ``B_norm_raw``, the divergence ``div``, ``nfp``,
    ``iota_axis`` and ``iota_edge`` (per full turn) and ``wall_discarded``."""
    from mrx.relaxation.physics import compute_divergence_norm  # noqa: PLC0415

    eq = seq.equilibrium
    B, norm, wall = potential_two_form(seq)
    iota = eq["profiles"]["iota"]
    return B, dict(kind=eq["kind"], nfp=int(seq.nfp), iota_axis=float(iota(0.0)), iota_edge=float(iota(1.0)),
                   B_norm_raw=float(norm), wall_discarded=float(wall),
                   div=float(compute_divergence_norm(B, seq)))
