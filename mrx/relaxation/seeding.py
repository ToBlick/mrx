"""Island seeds: small resonant perturbations of an equilibrium field that let island chains open.

A field with nested flux surfaces (an equilibrium read from a file, or a relaxed field) gets, at its rational surfaces
``iota = nfp n / m``,
a perturbation ``dB = curl(A B / |B|)`` with ``A = a(sqrt(s)) cos(2 pi (m theta - n zeta))`` (the SIESTA seed), so
that the relaxation can open island chains there. The transform is that of the field itself, measured on its flux
label as a function of ``s`` (:func:`mrx.relaxation.twist.iota_projection`), not the file's, and the profiles ``a`` are
functions of the label ``s``: a seed sits on the resonant surface, a level set of ``s``, wherever the mesh puts it. The
angles of the phase are the logical ones. ``dB`` is divergence-free and tangent to the wall.

- :func:`energy_seed` is the entry point. It seeds every resonance in the field's iota range with
  ``m <= n_theta / 2`` (or only the chains nearest the given ``iotas``) at once. By default the profiles
  ``a`` are the ones that lower the energy the most. Each is a combination of the mesh's radial splines, as functions of
  ``sqrt(s)``, within ``1 / m`` of its resonant surface, and the optimum is ``a* = -G^-1 g`` with ``g_i = <B, dB_i>`` and
  ``G_ij = <dB_i, dB_j>``. Given ``amplitudes`` rescale these optimal seeds.
- :func:`parallel_seed` builds one seed on a given field.
- :func:`resonances` lists the chains between two rotational transforms.

The phase of a seed is not free. Under stellarator symmetry the part of the seed with the wrong parity under
``(theta, zeta) -> (-theta, -zeta)`` is removed, so a phase shift only rescales the seed and its sign is the one
choice left.
"""
from __future__ import annotations

import warnings
from math import gcd

import jax
import jax.numpy as jnp
import numpy as np

from mrx.differential_forms import DiscreteFunction
from mrx.equilibria import StateField
from mrx.flux_label import enclosed_flux, flux_label
from mrx.relaxation.shells import LABEL_TOL
from mrx.relaxation.twist import iota_projection
from mrx.spline_bases import basis_table

#: The (r, theta, zeta) grid on which the seed's resonant amplitude is measured, and the number of points
#: evaluated at once.
N_R, N_THETA, N_ZETA = 320, 64, 32
BATCH = 65536


def resonances(iota_lo, iota_hi, nfp, m_max):
    """List the island chains ``(m, n)`` with ``iota_lo < nfp n / m < iota_hi``, ``m`` and ``n`` coprime and
    ``m <= m_max``, sorted by increasing ``m``. ``n`` counts field periods."""
    out = []
    for m in range(1, int(m_max) + 1):
        n_lo, n_hi = int(np.floor(iota_lo * m / nfp)), int(np.ceil(iota_hi * m / nfp))
        for n in range(max(1, n_lo), n_hi + 1):
            if gcd(m, n) == 1 and iota_lo < nfp * n / m < iota_hi:
                out.append((m, n))
    return out


def _iota_sign(odd, Bq):
    """The sign of the field's iota (ratio of poloidal to toroidal flux), from its quadrature values ``Bq``."""
    w = odd.quad.w
    return float(jnp.sign(jnp.sum(w * Bq[:, 1]) / jnp.sum(w * Bq[:, 2])))


def parallel_seed(seq, B, m, n, a_q):
    """One seed ``dB = curl(A B / |B|)`` on the field ``B``, with ``A = a cos(2 pi (m theta - sigma n zeta))``.

    ``a_q`` are the values of the profile at the quadrature points and ``sigma`` is the sign of the field's iota.
    ``dB`` is a 2-form with zero normal component on the wall and exactly zero divergence."""
    odd = seq.odd
    Bq = odd.evaluate_at_quadrature(B, 2)                  # det(DPhi) B^i, shape (n_q, 3)
    g_B = jnp.einsum('qij,qj->qi', odd.metric_jkl, Bq)                     # g_ij B^j times det(DPhi)
    Bhat_cov = g_B / jnp.sqrt(jnp.einsum('qi,qi->q', g_B, Bq))[:, None]    # B_i / |B|, det(DPhi) cancels
    x = odd.quad.x
    A_par = (a_q * jnp.cos(2.0 * jnp.pi * (m * x[:, 1] - _iota_sign(odd, Bq) * n * x[:, 2])))
    # project A B / |B| onto the 1-forms with zero tangential trace, so that its curl is divergence-free
    load = odd.vector_load_values(A_par[:, None] * Bhat_cov, 1, 1)
    return odd.G[1] @ odd.M[1].solve(load)


def energy_seed(seq, B, iotas=None, amplitudes=None, scale=1.0, verbose=True, T=None):
    """Seed the field ``B`` of an equilibrium-file sequence. Returns ``(B + scale * dB, rows)``.

    ``rows`` has one dict per seeded chain with ``m``, ``n``, the resonant surface ``x = sqrt(s)``,
    ``diota = |d iota / dx|`` there, the signed resonant normal field ``dBr = |dB^r| / |B^zeta|`` at the logical radius
    ``x`` and the resulting island width estimate ``w = sqrt(8 |dBr| nfp / (pi m diota))``, in ``x``. The resonance
    and the seed are functions of ``s``. The amplitude ``dBr`` is measured in the logical radius and the file's
    straight-field-line angle, so it and ``w`` are estimates where the mesh is not cut along the field's surfaces. ``iotas`` restricts the seeds to the chains nearest those
    rotational transforms. ``amplitudes`` (one per entry of ``iotas``) rescales each chain's energy-optimal seed
    so that its ``dBr`` takes the given value. If no chain with ``m <= n_theta / 2`` lies in the field's iota range,
    the field is returned unseeded with a warning and ``rows`` is empty. The transform is measured on the flux label of
    ``B`` (:func:`mrx.relaxation.twist.iota_projection`), whose temperature ``T`` is solved here unless given. The
    function runs eagerly and cannot be traced by ``jax.jit``.
    """
    nfp = int(seq.nfp)
    ns = seq.ns
    odd = seq.odd                     # the field's parity view (the sequence itself without stellarator symmetry)
    rb = seq.basis_0.bases[0].bases[0]                         # the radial basis
    h_r = 1.0 / (rb.n - rb.p)
    lam_h = StateField(seq.equilibrium["LA"], nfp)
    lam = lambda x: lam_h(x) / (2.0 * jnp.pi)   # noqa: E731  (lambda in turns)
    if T is None:
        T, info = flux_label(seq, odd.evaluate_at_quadrature(B, 2), tol=LABEL_TOL)
        if verbose:
            print(f"[seed] flux label CG {int(info)}", flush=True)
    s_q = enclosed_flux(seq, T, odd.evaluate_at_quadrature(B, 2))
    iota_s, diota_s = iota_projection(seq, B, s_q)             # per full turn, functions of s
    rr = np.linspace(1e-3, 1.0, 40001)                        # x = sqrt(s)
    iota_rr = np.abs(iota_s(rr ** 2))

    # each seed profile is one radial spline of the mesh as a function of sqrt(s), at the quadrature points
    basis = basis_table(rb, jnp.sqrt(s_q))
    knots = np.asarray(rb.T)
    support = [(float(knots[i]), float(knots[i + rb.p + 1])) for i in range(rb.n)]

    chains = []                          # (m, n, r_mn, |d iota / dr|, indices of the splines used)
    for m, n in resonances(float(iota_rr.min()), float(iota_rr.max()), nfp, ns[1] // 2):
        k = int(np.argmin(np.abs(iota_rr - nfp * n / m)))
        r0 = float(rr[k])
        if abs(float(iota_rr[k]) - nfp * n / m) < 1e-4 and 0.02 < r0 < 0.99:
            # the last spline (at the wall) and the first two (at the axis) are not used
            funcs = [i for i in range(2, rb.n - 1) if support[i][1] > r0 - 1.0 / m and support[i][0] < r0 + 1.0 / m]
            if funcs:
                chains.append((m, n, r0, abs(float(diota_s(np.array([r0 ** 2]))[0])) * 2 * r0, funcs))
    if not chains:
        warnings.warn(f"no island chain with m <= {ns[1] // 2} lies in the field's iota range "
                      f"[{float(iota_rr.min()):.4f}, {float(iota_rr.max()):.4f}] (nfp = {nfp}): nothing is seeded")
        return B, []
    if iotas is not None:
        # the chain nearest each requested transform. A chain may be picked only once
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
            dB.append(parallel_seed(seq, B, m, n, basis[i]))
            owner.append(c)
    owner = np.array(owner)
    dB = jnp.stack(dB)
    MdB = jnp.stack([odd.M[2] @ d for d in dB])
    g, G = MdB @ B, dB @ MdB.T
    a = -jnp.linalg.solve(0.5 * (G + G.T), g)
    energy = 0.5 * odd.l2_norm(B, 2) ** 2
    if verbose:
        print(f"[seed] {len(dB)} blocks, the joint optimum lowers the energy by "
              f"{-0.5 * (a @ g) / energy * 1e6:.4f} ppm", flush=True)

    # Measure each seed's resonant amplitude on a grid. The 2-form's values are det(DPhi) B^i. We take the (m, n)
    # Fourier coefficient of dr/dzeta = B^r / B^zeta in the straight-field-line angle theta* = theta + lambda.
    # Since det(DPhi) B^zeta = Psi' (1 + d lambda / d theta) cancels the Jacobian of theta -> theta*, the sum over
    # the uniform theta grid needs no weights.
    @jax.jit
    def two_form(dof, x):
        return jax.vmap(DiscreteFunction(dof, seq.basis_2, odd.E(2)))(x)

    def on_grid(f, *args):
        return jnp.concatenate([f(*args, x[k:k + BATCH]) for k in range(0, len(x), BATCH)])

    rg = jnp.linspace(0.02, 0.98, N_R)
    R, TH, ZE = jnp.meshgrid(rg, jnp.arange(N_THETA) / N_THETA, jnp.arange(N_ZETA) / N_ZETA, indexing="ij")
    x = jnp.stack([R.ravel(), TH.ravel(), ZE.ravel()], axis=1)
    shape = (N_R, N_THETA, N_ZETA)
    lam_x = on_grid(jax.jit(jax.vmap(lam))).reshape(shape)
    Bz = on_grid(two_form, B)[:, 2].reshape(shape).mean(axis=(1, 2))
    s = _iota_sign(odd, odd.evaluate_at_quadrature(B, 2))

    def normal_field(dBc, m, n, r0):
        br = on_grid(two_form, dBc)[:, 0].reshape(shape)
        # twice the complex coefficient of A cos(m theta* - s n zeta) is its peak amplitude A
        cr = 2.0 * (br * jnp.exp(-2j * jnp.pi * (m * (TH + lam_x) - s * n * ZE))).mean(axis=(1, 2))
        c = jnp.interp(r0, rg, cr.real) + 1j * jnp.interp(r0, rg, cr.imag)
        return float(jnp.sign(c.real) * jnp.abs(c) / jnp.abs(jnp.interp(r0, rg, Bz)))

    rows, dB_total = [], 0.0 * B
    for c, (m, n, r0, diota, _) in enumerate(chains):
        dBc = a[owner == c] @ dB[owner == c]
        dBr = normal_field(dBc, m, n, r0)
        if amplitudes is not None:            # rescale the seed to the requested amplitude
            dBc = dBc * (amplitudes[c] / dBr)
            dBr = amplitudes[c]
        dB_total = dB_total + dBc
        rows.append(dict(m=m, n=n, x=r0, diota=diota, dBr=dBr,
                         w=float(np.sqrt(8.0 * abs(dBr) * nfp / (np.pi * m * diota)))))
        if verbose:
            print(f"  ({m},{n}) x_mn {r0:.4f}  dBr {dBr:+.3e}  w {rows[-1]['w']:.4f} = {rows[-1]['w'] / h_r:.2f} h_r",
                  flush=True)
    seeded = B + scale * dB_total
    if verbose:
        print(f"[seed] ||dB|| / ||B|| = {float(odd.l2_norm(seeded - B, 2) / odd.l2_norm(B, 2)):.3e} (x {scale:g})",
              flush=True)
    return seeded, rows
