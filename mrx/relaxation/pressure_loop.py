"""An outer loop that prescribes the pressure profile p(s) around the incompressible relaxation.

Incompressible ideal flow keeps the toroidal flux, the rotational transform and the volume of every flux shell
``V'(s)``. The relaxed pressure is the multiplier of that volume constraint, so the loop controls ``V'(s)``, one number per bin of the flux label ``s``
(:mod:`mrx.flux_label`):

1. Relax incompressibly for a few steps (:func:`mrx.relaxation.loop.relax`).
2. Read off the weak pressure ``p_w`` per bin (:func:`binned_profile`) and compare it with the target ``p*``.
3. Give each bin the relative volume change ``g = -gain (p_w - p*) / <|B|^2>``, so that a shell with too much
   pressure is compressed. At fixed flux a compressed shell carries a stronger field, and since ``p + B^2/2`` is
   nearly constant across the surfaces its pressure, measured from the wall, drops. (Expanding it raised ``p_w``
   by about 25 % per iteration on li383, 2026-10-05.) ``<|B|^2>`` is the bin's mean ``|B|^2``.
4. Move that volume with one ideal compressible displacement ``u`` with ``div u = g`` and ``u . n = 0`` on the wall
   (:func:`volume_velocity`, :func:`displace`). It keeps the topology and the wall fluxes.

:func:`prescribe_pressure` runs the loop. Islands and chaotic bands are one bin of ``s`` each, because the flux label
is flat across them. The bins beyond the plasma edge are included with the target zero.
"""
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from mrx.flux_label import enclosed_flux, flux_label
from mrx.relaxation.physics import compute_force, weak_pressure


class Profile(NamedTuple):
    """Volume-weighted means over the bins of the flux label: the bin centres ``s``, the volumes ``V``, the weak
    pressure ``p``, the target ``target`` and ``Bsq``, the mean of ``|B|^2``."""
    s: jnp.ndarray
    V: jnp.ndarray
    p: jnp.ndarray
    target: jnp.ndarray
    Bsq: jnp.ndarray


def file_profile(seq):
    """The shape of the equilibrium file's pressure as a function of the label ``s``, zero at ``s = 1`` (the weak
    pressure vanishes on the wall)."""
    prof = seq.equilibrium["profiles"]["pressure"]
    s = np.linspace(0.0, 1.0, 2001)
    values = prof(np.sqrt(s)) - prof(1.0)
    s, values = jnp.asarray(s), jnp.asarray(values / np.abs(values).max())
    return lambda x: jnp.interp(x, s, values)


@eqx.filter_jit
def label(seq, B, T_guess=None):
    """Return ``(s_q, T)``: the flux label ``s`` at the quadrature points and the temperature ``T`` behind it."""
    B_jk = seq.odd.evaluate_at_quadrature(B, 2)
    T, _ = flux_label(seq, B_jk, guess=T_guess)
    return enclosed_flux(seq, T, B_jk), T


@eqx.filter_jit
def binned_profile(seq, B, s_q, target_q, bins: int):
    """The :class:`Profile` of ``B`` over ``bins`` equal bins of ``s``, with the weak pressure of ``B`` and the
    target values ``target_q`` at the quadrature points. Returns ``(profile, p_w)``. A bin that holds no quadrature
    point has ``V = 0`` and undefined means."""
    odd, even = seq.odd, seq.even
    _, _, J, _ = compute_force(B, seq)
    p_w = weak_pressure(J, B, seq)
    p_q = even.evaluate_at_quadrature(p_w, 0)[:, 0]
    B_jk = odd.evaluate_at_quadrature(B, 2)
    Bsq_q = jnp.einsum('qi,qij,qj->q', B_jk, seq.metric_jkl, B_jk) / seq.jacobian_j ** 2
    wJ = seq.quad.w * seq.jacobian_j
    b = jnp.minimum((s_q * bins).astype(jnp.int32), bins - 1)             # s = 1 on the wall belongs to the last bin
    V = jnp.zeros(bins).at[b].add(wJ)

    def mean(f):
        return jnp.zeros(bins).at[b].add(wJ * f) / V
    return Profile((jnp.arange(bins) + 0.5) / bins, V, mean(p_q), mean(target_q), mean(Bsq_q)), p_w


@eqx.filter_jit
def volume_velocity(seq, g_q):
    """The even 2-form ``u`` with ``div u = g - <g>_V`` and ``u . n = 0`` on the wall, of least norm, for the
    quadrature values ``g_q``: ``u = M_2^-1 D_2^T psi`` with ``L_3 psi`` the load of ``g``."""
    even = seq.even
    wJ = seq.quad.w * seq.jacobian_j
    g_q = g_q - jnp.sum(wJ * g_q) / jnp.sum(wJ)            # the wall is fixed, so the total volume is too
    psi = even.L[3].solve(even._scalar_load_values(g_q, 3))
    return even.M[2].solve(even.D[2].T @ psi)


@eqx.filter_jit
def displace(seq, B, u):
    """One ideal step ``B + curl(u x B)`` of the field ``B`` by the displacement ``u``, as in the relaxation."""
    odd = seq.odd
    E = odd.M[1].solve(odd.cross_product_load_values(seq.even.evaluate_at_quadrature(u, 2),
                                                     odd.evaluate_at_quadrature(B, 2), 1, 2, 2))
    return B + odd.G[1] @ E


def prescribe_pressure(seq, B, ts, shape, beta, outer=10, steps=10, gain=0.5, bins=16, verbose=True):
    """Run the outer loop from the field ``B`` with the stepper ``ts`` (incompressible) and return
    ``(B, history)``.

    The target is ``p*(s) = c shape(s)`` with ``c`` fixed at the start so that ``int p* dV`` is ``beta`` times the
    magnetic energy of ``B``. Each of the ``outer`` iterations takes ``steps`` relaxation steps, then one volume
    displacement with the given ``gain``. ``history`` holds per iteration the relative error
    ``||p_w - p*||_V / ||p*||_V`` over the bins, the force residual, ``beta_w`` and the largest ``|g|``, plus the
    last :class:`Profile`.
    """
    from mrx.relaxation.loop import initial_state, relax  # noqa: PLC0415
    wJ = seq.quad.w * seq.jacobian_j
    s_q, T = label(seq, B)
    c = beta * 0.5 * seq.odd.l2_norm_sq(B, 2) / jnp.sum(wJ * shape(s_q))
    history = dict(err=[], resid=[], beta_w=[], g_max=[])
    for k in range(outer + 1):
        res = relax(initial_state(B, ts), ts, steps=steps, chunk=steps, verbose=False)
        B = res.state.B_n
        s_q, T = label(seq, B, T)
        prof, p_w = binned_profile(seq, B, s_q, c * shape(s_q), bins)
        # a bin without quadrature points (a narrow range of s on a coarse mesh) has no mean
        prof = Profile(*(np.asarray(v)[np.asarray(prof.V) > 0] for v in prof))
        err = float(np.sqrt(np.sum(prof.V * (prof.p - prof.target) ** 2) / np.sum(prof.V * prof.target ** 2)))
        beta_w = float(np.sum(prof.V * prof.p) / (0.5 * seq.odd.l2_norm_sq(B, 2)))
        g = -gain * (prof.p - prof.target) / prof.Bsq
        for key, v in zip(history, (err, res.trace["resid"][-1], beta_w, float(np.abs(g).max()))):
            history[key].append(v)
        if verbose:
            print(f"[outer {k:2d}] |p_w - p*|/|p*| {err:.3e}  resid {res.trace['resid'][-1]:.3e}  "
                  f"beta_w {beta_w:.4e}  max|g| {float(np.abs(g).max()):.2e}", flush=True)
        if k == outer:
            break
        B = displace(seq, B, volume_velocity(seq, jnp.interp(s_q, jnp.asarray(prof.s), jnp.asarray(g))))
    history["profile"] = prof
    return B, history
