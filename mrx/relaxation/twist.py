"""Twist moves: change the rotational transform of a field by adding poloidal flux on its flux surfaces.

The move is ``dB = curl(f grad zeta) = grad f x grad zeta`` with ``f = F(s)`` a function of the flux label ``s``
(:mod:`mrx.flux_label`, the normalised toroidal flux inside a level set of the label's temperature ``T``) and
``zeta`` the logical toroidal angle, one field period per unit. The sections ``zeta = const`` need not be planes.
``dB`` is divergence-free, tangent to the level sets of ``T`` and has no ``zeta`` component, so every flux surface
stays a flux surface and keeps its toroidal flux. It adds the poloidal flux ``nfp F(s)`` inside the surface ``s``
over a toroidal turn, so the rotational transform changes by ``diota = nfp dF/dPhi``. Where ``T`` is flat (an island
chain or a chaotic band) ``f`` is flat and no twist is added. The net toroidal current inside each surface changes
with ``dF/ds``. It is a one-off change of the field before a relaxation, like the island seeds of
:mod:`mrx.relaxation.seeding`. An ideal relaxation then keeps the new transform, since it holds the fluxes of every
surface.

- :func:`retarget_iota` is the entry point: the twist that takes the transform ``iota(s)``, or the net toroidal
  current ``I(s)``, to a target profile.
- :func:`twist_field` is the move for a given ``F(s)``.
- :func:`iota_fit`, :func:`iota_profile` and :func:`enclosed_current` measure the transform and the current on the
  label's level sets, :func:`layer_iota` the transform on the logical surfaces.
- :func:`load_iota_target` reads a target from a wout (its transform) or a VMEC input namelist (its transform, or
  with ``NCURR = 1`` its current).

The potential ``f dzeta`` lives in the free 1-forms. On the wall ``T = 0`` and ``f = F(1)``, so its trace there is
the closed form ``F(1) dzeta`` and ``dB`` has no flux through the wall. ``F(1) != 0`` changes the poloidal flux inside
the wall and with it the net toroidal current.
"""
from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

from mrx.differential_forms import DiscreteFunction
from mrx.flux_label import enclosed_flux_map, flux_label
from mrx.relaxation.physics import compute_divergence_norm, current as weak_current
from mrx.relaxation.shells import LABEL_TOL

#: The cumulative sums :func:`iota_fit` fits, sampled at this many points.
FIT_SAMPLES = 1024
#: The final transform may miss the target by this fraction of the target's largest magnitude before a warning.
WARN_TOL = 1e-2


def twist_field(seq, T, s_of, F_of_s):
    """Return the twist move ``dB = curl(f grad zeta) = grad f x grad zeta`` with ``f = F_of_s(s_of(T))`` as a
    Dirichlet 2-form, and the free 2-form it is the restriction of.

    ``T`` is the label's temperature (an even 0-form, zero on the wall), ``s_of`` maps its values to the label ``s``
    and ``F_of_s`` is traceable. The potential ``f dzeta`` is histopolated into the free 1-forms and ``dB`` is its
    curl. It is exactly divergence-free and tangent to the level sets of ``T``. Its flux through the wall vanishes,
    since the potential's trace there is ``F(1) dzeta``, so dropping the free form's wall DoFs loses nothing. It adds
    ``nfp F(s)`` to the poloidal flux enclosed by the surface ``s`` over a toroidal turn."""
    free = seq.odd.free
    T_fn = DiscreteFunction(T, seq.basis_0, seq.even.E(0))

    def A(x):
        f = F_of_s(s_of(T_fn(x)[0]))
        return jnp.stack([jnp.zeros_like(f), jnp.zeros_like(f), f])
    dB_free = free.G[1] @ free.interpolate(A, 1)
    return seq.odd.restrict(dB_free, 2), dB_free


def iota_profile(seq, B, s_q, bins):
    """``(s, iota, Phi)``: the rotational transform per full turn on ``bins`` bins of the label, equal in ``s``
    (empty bins dropped), at the bins' flux-weighted ``s``, and the toroidal flux.

    The volume sums of the 2-form's ``theta`` and ``zeta`` densities over a bin are the poloidal flux of the bin per
    field period and its toroidal flux, whatever the poloidal angle, when the bin's walls are flux surfaces. So
    ``iota = nfp dpsi / dPhi`` per bin. ``s_q`` is the label at the quadrature points."""
    Bq = np.asarray(seq.odd.evaluate_at_quadrature(B, 2))
    w = np.asarray(seq.quad.w)
    s = np.asarray(s_q)
    b = np.minimum((s * bins).astype(int), bins - 1)
    psi, phi = np.bincount(b, w * Bq[:, 1], bins), np.bincount(b, w * Bq[:, 2], bins)
    full = phi != 0
    s_b = np.bincount(b, w * Bq[:, 2] * s, bins)[full] / phi[full]
    return s_b, seq.nfp * psi[full] / phi[full], float(phi.sum() / np.sum(np.asarray(seq.quad.w_z)))


def iota_fit(seq, B, s_q, degree):
    """The transform per full turn as a smooth function of the label ``s``: the poloidal flux enclosed by the label's
    level sets against the toroidal flux they enclose (cumulative volume sums of the 2-form's ``theta`` and ``zeta``
    densities over the points sorted by ``s``), fitted by a polynomial of ``degree + 1`` without constant in the
    enclosed toroidal flux fraction and differentiated. The cumulative sums average the scatter of the label within
    a radial layer, which a bin's own sums (:func:`iota_profile`) see in full."""
    Bq = np.asarray(seq.odd.evaluate_at_quadrature(B, 2))
    w = np.asarray(seq.quad.w)
    order = np.argsort(np.asarray(s_q))
    psi = np.cumsum((w * Bq[:, 1])[order])
    phi = np.cumsum((w * Bq[:, 2])[order])
    pick = np.linspace(0, psi.size - 1, FIT_SAMPLES).astype(int)
    x = phi[pick] / phi[-1]
    c = np.linalg.lstsq(np.stack([x ** k for k in range(1, degree + 2)], axis=1), psi[pick] / phi[-1], rcond=None)[0]
    return seq.nfp * np.polynomial.Polynomial(np.concatenate([[0.0], c])).deriv()


def enclosed_current(seq, B, s_q, edges):
    """``mu0 I`` (T m) through the sections ``zeta = const`` inside the level sets ``s < edges``, the weak current's
    flux averaged over ``zeta``, positive along ``+zeta``."""
    J = seq.odd.evaluate_at_quadrature(weak_current(B, seq), 1)
    Jz = np.asarray(seq.jacobian_j * jnp.einsum("qj,qj->q", seq.metric_inv_jkl[:, 2, :], J))
    w, s = np.asarray(seq.quad.w), np.asarray(s_q)
    return np.array([np.sum((s < e) * w * Jz) for e in edges]) / np.sum(np.asarray(seq.quad.w_z))


def layer_iota(seq, B):
    """``(r, iota)``: the transform per full turn on the radial quadrature layers, the ratio of the layer's poloidal
    and toroidal flux. It is the surface's transform where the logical surfaces are flux surfaces (a field read from
    an equilibrium file), the form :func:`mrx.relaxation.seeding.energy_seed` takes."""
    Bq = np.asarray(seq.odd.evaluate_at_quadrature(B, 2))
    Bw = (np.asarray(seq.quad.w)[:, None] * Bq).reshape(*seq.quad.shape, 3)
    return np.asarray(seq.quad.x_x), seq.nfp * Bw[..., 1].sum((1, 2)) / Bw[..., 2].sum((1, 2))


def load_iota_target(path):
    """The target of :func:`retarget_iota` in a file: a wout (``.nc``, its ``iotaf``) or a VMEC input namelist (its
    transform, or with ``NCURR = 1`` its net toroidal current ``mu0 I(s)`` along the field). Returns the keyword
    ``iota`` or ``current``, a function of ``s``."""
    if path.endswith(".nc"):
        from mrx.equilibria.vmec import read_wout
        f = read_wout(path)["profiles"]["iota"]                          # a function of r = sqrt(s)
        return dict(iota=lambda s: np.asarray(f(np.sqrt(np.asarray(s)))))
    from mrx.equilibria.indata import read_indata
    nl = read_indata(path)
    if nl.ncurr == 0:
        return dict(iota=nl.iota)
    return dict(current=lambda s: nl.current(s) * np.sign(nl.phiedge))   # along phi, made along the field


def retarget_iota(seq, B, iota=None, current=None, T=None, bins=20, degree=6, corrections=2, label_tol=LABEL_TOL,
                  verbose=True):
    """Twist the field ``B`` (a Dirichlet 2-form) so that its rotational transform is ``iota(s)``, or its net toroidal
    current inside each surface ``current(s)``. Returns ``(B + dB, record)``.

    ``iota`` is per full turn, its magnitude taken with the sign of the field's transform, ``current`` is
    ``mu0 I(s)`` in T m along the field. Both are functions of the label ``s``. ``T`` is the label's temperature
    (computed if not given, to ``label_tol``).

    The transform is measured by :func:`iota_fit` (a polynomial of ``degree`` in ``s``) and reported on ``bins``
    label bins (:func:`iota_profile`). The twist ``F = Phi / nfp int_0^s diota`` with ``diota = iota* - iota`` is
    scaled by its measured response (the transform is linear in ``F`` at fixed toroidal flux, the response is 1 up to
    the label's resolution and its sign is the orientation of the angles), then ``corrections`` times corrected by
    the measured residual on the same label (the twist keeps its surfaces). A current target becomes a transform
    target through the measured local response ``dI(s) / diota(s)`` of a uniform trial twist (the current inside
    ``s`` depends on ``dF/ds`` there only). Where an island or a chaotic band keeps the label flat, no twist is added
    and the target is missed there: a final miss above :data:`WARN_TOL` of the target is a warning, not an error.

    ``record`` holds the bins, the transform and the current before and after, the target, ``F`` and the checks
    (``div dB``, the wall flux of the free ``dB``, the relative size of ``dB``), and the layer transform after the
    twist (:func:`layer_iota`, the argument :func:`mrx.relaxation.seeding.energy_seed` takes).
    """
    if (iota is None) == (current is None):
        raise ValueError("retarget_iota takes one of iota and current")
    odd = seq.odd
    nfp = seq.nfp
    B_jk = odd.evaluate_at_quadrature(B, 2)
    if T is None:
        T, info = flux_label(seq, B_jk, tol=label_tol)
        if int(info) <= 0:
            warnings.warn(f"retarget_iota: the flux label did not converge ({int(info)}), the twist follows it as is",
                          stacklevel=2)
    s_of = enclosed_flux_map(seq, T, B_jk)
    s_q = np.asarray(s_of(seq.even.evaluate_at_quadrature(T, 0)[:, 0]))
    edges = np.arange(1, bins + 1) / bins
    s_fine = np.linspace(0.0, 1.0, 2001)

    def measure(B_):
        s_b, i_b, phi = iota_profile(seq, B_, s_q, bins)
        return s_b, i_b, phi, iota_fit(seq, B_, s_q, degree)

    def F_of(diota):
        """``F = Phi / nfp int_0^s diota`` on ``s_fine``."""
        return phi0 / nfp * np.concatenate([[0.0], np.cumsum(0.5 * (diota[1:] + diota[:-1]) * np.diff(s_fine))])

    def move(F):
        return twist_field(seq, T, s_of, lambda s: jnp.interp(s, jnp.asarray(s_fine), jnp.asarray(F)))

    s0, i0, phi0, fit0 = measure(B)
    sign = float(np.sign(np.mean(i0)))
    I0 = enclosed_current(seq, B, s_q, edges)
    phi_sign = float(np.sign(phi0))

    # the transform's change per unit F, a scalar (1 up to the label's resolution, with the angles' orientation)
    dB_probe = move(F_of(np.ones_like(s_fine)))[0]
    fit1 = measure(B + dB_probe)[3]
    gain = float(np.mean((fit1 - fit0)(s_fine)))

    if iota is not None:
        def wanted(fit_, B_):
            return sign * np.abs(np.asarray(iota(s_fine))) - fit_(s_fine)
    else:
        # the current's response to a uniform transform change, per edge: I(s) depends on dF/ds at s only
        dI = (enclosed_current(seq, B + dB_probe, s_q, edges) - I0) / gain
        I_target = phi_sign * np.asarray(current(edges))

        def wanted(fit_, B_):
            miss = (I_target - enclosed_current(seq, B_, s_q, edges)) / dI
            return np.polynomial.Polynomial.fit(np.concatenate([[0.0], edges]), np.concatenate([[0.0], miss]),
                                                degree)(s_fine)

    F = F_of(wanted(fit0, B)) / gain
    history = []
    for k in range(corrections + 1):
        dB, dB_free = move(F)
        s_k, i_k, _, fit_k = measure(B + dB)
        diota = wanted(fit_k, B + dB)
        history.append(dict(correction=k, iota=i_k.tolist(), miss_max=float(np.abs(diota).max())))
        if k < corrections:
            F = F + F_of(diota) / gain
    B_new = B + dB
    I_new = enclosed_current(seq, B_new, s_q, edges)
    r_l, iota_l = layer_iota(seq, B_new)
    target_b = sign * np.abs(np.asarray(iota(s_k))) if iota is not None else None
    miss = (np.abs(fit_k(s_k) - target_b).max() / np.abs(target_b).max() if iota is not None
            else np.abs(I_new - I_target).max() / np.abs(I_target).max())
    raw = odd.free.E(2).T @ dB_free
    s2 = tuple(int(v) for v in seq.basis_2.shape[0])
    rec = dict(kind="iota" if iota is not None else "current", bins=bins, s=s_k.tolist(), iota_before=i0.tolist(),
               iota_after=i_k.tolist(), iota_fit_before=fit0(s0).tolist(), iota_fit_after=fit_k(s_k).tolist(),
               iota_target=None if target_b is None else target_b.tolist(), edges=edges.tolist(),
               current_before=I0.tolist(), current_after=I_new.tolist(),
               current_target=None if current is None else I_target.tolist(), miss=float(miss), gain=gain,
               F_s=s_fine[::40].tolist(), F=np.asarray(F)[::40].tolist(), F_edge=float(F[-1]),
               poloidal_flux_added=float(nfp * F[-1]), phi=phi0, history=history,
               dB_rel=float(odd.l2_norm(dB, 2) / odd.l2_norm(B, 2)),
               div_dB_rel=float(compute_divergence_norm(dB, seq) / odd.l2_norm(dB, 2)),
               wall_flux_dB=float(jnp.abs(raw[:s2[0] * s2[1] * s2[2]].reshape(s2)[-1]).max()),
               r_layer=r_l.tolist(), iota_layer=iota_l.tolist())
    if verbose:
        print(f"[iota] {rec['kind']} target on {len(s_k)} label bins, response {gain:+.4f} per unit F, "
              f"F(1) {F[-1]:+.4e} Wb per period, |dB| / |B| {rec['dB_rel']:.3e}, div dB {rec['div_dB_rel']:.1e}, "
              f"wall flux of dB {rec['wall_flux_dB']:.1e}", flush=True)
        print("[iota] s     " + " ".join(f"{v:7.3f}" for v in s_k), flush=True)
        print("[iota] before" + " ".join(f"{v:+7.4f}" for v in i0), flush=True)
        print("[iota] after " + " ".join(f"{v:+7.4f}" for v in i_k), flush=True)
        print("[iota] fit   " + " ".join(f"{v:+7.4f}" for v in fit_k(s_k)), flush=True)
        if target_b is not None:
            print("[iota] target" + " ".join(f"{v:+7.4f}" for v in target_b), flush=True)
        print(f"[iota] mu0 I(1) {I0[-1]:+.4e} -> {I_new[-1]:+.4e} T m, miss {miss:.2e} of the target", flush=True)
    if miss > WARN_TOL:
        warnings.warn(f"retarget_iota: the target is missed by {miss:.2e} of its size (islands or chaos keep the "
                      f"label flat there, or the target lies beyond the label's resolution)", stacklevel=2)
    return B_new, rec
