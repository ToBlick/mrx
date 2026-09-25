"""Island seeds by the energy criterion (the paper's Sec. 6.2, Tab. 4).

Every resonance ``(m, n)`` with ``m <= n_theta / 2`` and ``iota = nfp n / m``
inside the field's iota range gets SIESTA's parallel seed, ``dB = curl(A B /
|B|)`` with ``A = a(r) cos(2 pi (m theta - n zeta))``
(:func:`mrx.initial_conditions.parallel_seed`), its profile ``a(r)`` free in
the sequence's own radial basis within ``1 / m`` of the resonant radius
``r_mn`` (the wall function and the two innermost functions of the polar
surgery left out). The chains are solved together for the amplitudes of least
energy, ``a* = -G^-1 g`` with ``g_i = <B, dB_i>`` and ``G_ij = <dB_i, dB_j>``.

The seed has no phase: a phase shift multiplies it by ``cos(2 pi phase)``
and nothing else, because the odd part is removed exactly by the stellarator
parity projector (measured: ``||dB||^2`` at phase 1/4 is 1e-25 of its phase-0
value). The profile's sign is the only freedom and ``a*`` sets it.

:func:`energy_seed` takes ``iotas`` and ``amplitudes``: neither given, the
energy criterion over every resonance in range; ``iotas`` given, the criterion
over those chains only; both given, the amplitudes replace the criterion's
(each the resonant normal field ``|dB^r| / |B^zeta|`` at ``r_mn``, signed);
``amplitudes`` without ``iotas`` is ignored with a warning.
"""
from __future__ import annotations

import warnings
from math import gcd

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline

from mrx.differential_forms import DiscreteFunction
from mrx.gvec import load_clebsch
from mrx.initial_conditions import parallel_seed

#: The grid of the harmonic analysis (r, theta, zeta) and the evaluation batch.
N_R, N_THETA, N_ZETA = 320, 64, 32
BATCH = 65536


def resonances(iota_lo, iota_hi, nfp, m_max):
    """The chains that can sit between two rotational transforms: ``(m, n)``
    coprime with ``m <= m_max`` and ``iota_lo < nfp n / m < iota_hi``, by
    increasing ``m`` -- a chain at ``iota = nfp n / m`` closes after ``m``
    field periods (the seeds' convention, :mod:`mrx.initial_conditions`)."""
    out = []
    for m in range(1, int(m_max) + 1):
        n_lo, n_hi = int(np.floor(iota_lo * m / nfp)), int(np.ceil(iota_hi * m / nfp))
        for n in range(max(1, n_lo), n_hi + 1):
            if gcd(m, n) == 1 and iota_lo < nfp * n / m < iota_hi:
                out.append((m, n))
    return out


def energy_seed(seq, B, iotas=None, amplitudes=None, scale=1.0, verbose=True):
    """``(B + scale * dB, rows)``: the seeded field and one row per seeded chain
    (``m``, ``n``, ``r``, ``diota``, ``dBr`` the resonant normal field of the
    chain's amplitude, ``w`` its pendulum width ``sqrt(8 dBr nfp / (pi m
    |d_r iota|))``). ``seq`` is an equilibrium-file sequence (the iota profile
    and lambda come from its Clebsch data)."""
    if amplitudes is not None and iotas is None:
        warnings.warn("seed amplitudes without seed iotas are ignored: the energy criterion sets them",
                      stacklevel=2)
        amplitudes = None
    cb = load_clebsch(seq.equilibrium, nfp=seq.nfp)
    nfp = int(cb["nfp"])
    ns = seq.ns
    rb = seq.basis_0.bases[0].bases[0]                         # the radial basis
    h_r = 1.0 / (rb.n - rb.p)
    lam = lambda x: cb["lam_h"](x) / (2.0 * jnp.pi)   # noqa: E731  (lambda in turns)

    # the rotational transform per full turn, |dchi / dPhi| as mrx.initial_conditions counts it, a spline in rho
    # off the axis
    rho, dPhi, dchi = (np.asarray(cb[k], float) for k in ("rho", "dPhi", "dchi"))
    inside = dPhi != 0.0
    iota = CubicSpline(rho[inside], np.abs(dchi[inside] / dPhi[inside]))
    rr = np.linspace(1e-3, 1.0, 40001)

    # the radial basis, sampled: the profile of one seed block is one of its functions
    r_fine = np.linspace(0.0, 1.0, 2001)
    basis = np.asarray(jax.vmap(lambda x: jnp.array([rb(x, i) for i in range(rb.n)]))(jnp.asarray(r_fine))).T
    knots = np.asarray(rb.T)
    support = [(float(knots[i]), float(knots[i + rb.p + 1])) for i in range(rb.n)]

    chains = []                          # (m, n, r_mn, |d_r iota|, the block's basis functions)
    for m, n in resonances(float(iota(rr).min()), float(iota(rr).max()), nfp, ns[1] // 2):
        r0 = float(rr[int(np.argmin(np.abs(iota(rr) - nfp * n / m)))])
        if abs(float(iota(r0)) - nfp * n / m) < 1e-4 and 0.02 < r0 < 0.99:
            funcs = [i for i in range(2, rb.n - 1) if support[i][1] > r0 - 1.0 / m and support[i][0] < r0 + 1.0 / m]
            if funcs:
                chains.append((m, n, r0, abs(float(iota.derivative()(r0))), funcs))
    if iotas is not None:
        # the chains nearest the requested transforms, each once
        picked = []
        for target in iotas:
            c = int(np.argmin([abs(nfp * n / m - target) for m, n, *_ in chains]))
            if c in picked:
                raise ValueError(f"iota {target:g} names the chain {chains[c][:2]} twice")
            picked.append(c)
        chains = [chains[c] for c in picked]
    if verbose:
        print(f"[seed] {len(chains)} resonances with m <= {ns[1] // 2}: "
              + " ".join(f"({m},{n})" for m, n, *_ in chains) + f", h_r {h_r:.4f}", flush=True)

    dB, owner = [], []
    for c, (m, n, _, _, funcs) in enumerate(chains):
        for i in funcs:
            dB.append(parallel_seed(seq, B, m, n, basis[i], r_fine)[0])
            owner.append(c)
    owner = np.array(owner)
    MdB = [seq.apply_mass_matrix(d, 2, dirichlet=True) for d in dB]
    g = np.array([float(B @ Md) for Md in MdB])
    G = np.array([[float(d @ Md) for Md in MdB] for d in dB])
    a = -np.linalg.solve(0.5 * (G + G.T), g)
    energy = 0.5 * float(seq.l2_norm(B, 2)) ** 2
    if verbose:
        print(f"[seed] {len(dB)} blocks, the joint optimum lowers the energy by "
              f"{-0.5 * (a @ g) / energy * 1e6:.4f} ppm", flush=True)

    # the harmonics on the grid: sqrt(g) B^i is the logical 2-form's value. The (m, n) coefficient of dr/dzeta =
    # B^r / B^zeta in (theta*, zeta), theta* = theta + lambda: sqrt(g) B^zeta = Phi' (1 + d_theta lambda) in the
    # wout's coordinates cancels the Jacobian of theta -> theta*, so the sum over the theta grid has uniform
    # weights. The resonant angle takes the seed's sign of iota (parallel_seed's flux ratio)
    @jax.jit
    def two_form(dof, x):
        return jax.vmap(DiscreteFunction(dof, seq.basis_2, seq.E(2, True)))(x)

    def on_grid(f, *args):
        return np.concatenate([np.asarray(f(*args, x[k:k + BATCH])) for k in range(0, len(x), BATCH)])

    rg = np.linspace(0.02, 0.98, N_R)
    R, TH, ZE = np.meshgrid(rg, np.arange(N_THETA) / N_THETA, np.arange(N_ZETA) / N_ZETA, indexing="ij")
    x = jnp.asarray(np.stack([R.ravel(), TH.ravel(), ZE.ravel()], axis=1))
    shape = (N_R, N_THETA, N_ZETA)
    lam_x = on_grid(jax.jit(jax.vmap(lam))).reshape(shape)
    Bz = on_grid(two_form, B)[:, 2].reshape(shape).mean(axis=(1, 2))
    Bq, w = seq.evaluate_at_quadrature(B, 2, dirichlet=True), seq.quad.w
    s = float(jnp.sign(jnp.sum(w * Bq[:, 1]) / jnp.sum(w * Bq[:, 2])))

    def normal_field(dBc, m, n, r0):
        """The resonant normal field of a chain's perturbation at its radius (signed)."""
        br = on_grid(two_form, dBc)[:, 0].reshape(shape)
        # twice the complex coefficient of A cos(m theta* - s n zeta) is its peak amplitude A
        cr = 2.0 * (br * np.exp(-2j * np.pi * (m * (TH + lam_x) - s * n * ZE))).mean(axis=(1, 2))
        c = np.interp(r0, rg, cr.real) + 1j * np.interp(r0, rg, cr.imag)
        return float(np.sign(c.real) * abs(c) / abs(np.interp(r0, rg, Bz)))

    rows, dB_total = [], 0.0 * B
    for c, (m, n, r0, diota, _) in enumerate(chains):
        dBc = sum(float(a[i]) * dB[i] for i in np.nonzero(owner == c)[0])
        dBr = normal_field(dBc, m, n, r0)
        if amplitudes is not None:            # the amplitude replaces the criterion's, linearly
            dBc = dBc * (amplitudes[c] / dBr)
            dBr = amplitudes[c]
        dB_total = dB_total + dBc
        rows.append(dict(m=m, n=n, r=r0, diota=diota, dBr=dBr,
                         w=float(np.sqrt(8.0 * abs(dBr) * nfp / (np.pi * m * diota)))))
        if verbose:
            print(f"  ({m},{n}) r_mn {r0:.4f}  dBr {dBr:+.3e}  w {rows[-1]['w']:.4f} = {rows[-1]['w'] / h_r:.2f} h_r",
                  flush=True)
    seeded = B + scale * dB_total
    if verbose:
        print(f"[seed] ||dB|| / ||B|| = {float(seq.l2_norm(seeded - B, 2) / seq.l2_norm(B, 2)):.3e} (x {scale:g})",
              flush=True)
    return seeded, rows
