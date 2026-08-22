"""The natural-BC face coefficient: ``exact`` vs the corrected ``ibp``.

Iteration counts are an indirect verdict on a scalar. This prints the scalar
itself, for every ``(k, c)`` that has a boundary trace, together with the
factor the analysis predicts separates them.

``E^T M_{k-1}^{-1} E`` reduces to ``alpha (e e^T) (x) M_t (x) M_z`` with

    alpha = < w_face >_{theta,zeta}(r=1) * mu

and the two implementations disagree in BOTH factors::

    exact   w_face = m_k sqrt(g^rr)   mu = (M_r[m_{k-1}]^-1)[last,last]
    ibp     w_face = m_k g^rr         mu = (M_r[1]^-1)[last,last]
                   = m_k^2 / m_{k-1}       the metric-free logical 1/h

``m_k`` is the V_k component mass weight, ``m_{k-1}`` the partner's. The two
face weights differ by ``sqrt(g_rr)``; the two amplifications by ``m_{k-1}``.
Their product is one surface element, so the prediction is

    alpha_ibp / alpha_exact  ~  J sqrt(g^rr) |_{r=1}

which this script also evaluates directly, as an independent check that the
discrepancy is the measure and not a coincidence of the geometry.
"""
from __future__ import annotations

import argparse
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx  # noqa: E402
from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.experimental.block_jacobi_laplacian import (  # noqa: E402
    _axis_bases, _face_metric_scalar, _mesh_amplification,
    _weak_inverse_amplification, bundled_axis_profiles, weight_fields)
from mrx.operators import (  # noqa: E402
    _assemble_weighted_1d_mass, _dense_incidence_1d)
from mrx.mappings import rotating_ellipse_map, toroid_map  # noqa: E402

mrx.MAP_BATCH_SIZE_INNER = int(os.environ.get("W7X_MAP_BATCH", "256"))

# (k, c) pairs whose radial axis is a derivative axis -- the ones with a trace.
TRACE_CASES = ((1, 0), (2, 1), (2, 2), (3, 0))


def build_sequence(geometry, ns, p):
    seq = DeRhamSequence(ns, (p,) * 3, 2 * p, ("clamped", "periodic", "periodic"),
                         polar=True, tol=1e-12, maxiter=1000,
                         betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    if geometry == "toroid":
        seq.set_map(toroid_map(epsilon=1 / 3, R0=1.0))
    elif geometry == "rot-ellipse":
        seq.set_map(rotating_ellipse_map(eps=0.33, kappa=1.5, nfp=3))
    else:
        from w7x_geometry import build_w7x_map  # noqa: PLC0415
        map_func, _ = build_w7x_map(map_ns=ns, p=p)
        seq.set_map(map_func)
    return seq


def face_mean(seq, field):
    """``<field>_{theta,zeta}`` on the last radial quadrature slice."""
    wy, wz = seq.quad.w_y, seq.quad.w_z
    return float(jnp.einsum('rs,r,s->', jnp.asarray(field)[-1], wy, wz)
                 / (jnp.sum(wy) * jnp.sum(wz)))


def roundtrip_reference(seq):
    r"""The k=1 radial boundary correction, from the EXACT 1-D round trip.

    ``F = M^d G A^-1 G^T M^d`` is the weak block's radial factor with no
    integration by parts and no approximation; ``Ktilde`` is what the atom uses
    instead. Their difference IS the boundary correction, cross term included::

        F - Ktilde  =  alpha e e^T  -  (e v^T + v e^T)
                       \__________/    \____________/
                        rank one         the cross term, rank TWO

    So the split is measurable. ``c_star`` is the best coefficient along
    ``e = dLam(1)`` -- directly comparable to the alphas above -- and
    ``off_frac`` is what is left after removing it: small means the correction
    really is rank one along ``e`` and the cross term is negligible; order one
    means it is not.
    """
    primal, deriv, quad_w = _axis_bases(seq)
    fields = weight_fields(seq)
    ginv, jac = fields["ginv_aa"], fields["jac"]
    c = a = 0

    from mrx.local_assembly import _second_derivative_tables  # noqa: PLC0415
    dd = _second_derivative_tables(seq)
    ktilde = np.asarray(_assemble_weighted_1d_mass(
        dd[a], quad_w[a] * bundled_axis_profiles(seq, ginv[c] * jac * ginv[a])[a]))
    m_d = np.asarray(_assemble_weighted_1d_mass(
        deriv[a], quad_w[a] * bundled_axis_profiles(seq, ginv[c] * jac)[a]))
    a_mat = np.asarray(_assemble_weighted_1d_mass(
        primal[a], quad_w[a] * bundled_axis_profiles(seq, jac)[a]))
    g = np.asarray(_dense_incidence_1d(int(a_mat.shape[0]), seq.basis_0.types[a]))

    dlam = seq.basis_0.dΛ[a]
    end = 1.0 - 1e-8 if dlam.type != "periodic" else 0.0
    e = np.asarray(jax.vmap(lambda i: jnp.sum(dlam(end, i)))(dlam.ns))

    out = {}
    n0 = a_mat.shape[0]
    for label, cols in (("free", np.arange(n0)), ("dbc", np.arange(n0 - 1))):
        gg = g[:, cols]
        f = m_d @ (gg @ np.linalg.solve(a_mat[np.ix_(cols, cols)], gg.T)) @ m_d
        d = f - ktilde
        ee = float(e @ e)
        c_star = float(e @ d @ e) / ee ** 2
        off = d - c_star * np.outer(e, e)
        sv = np.linalg.svd(d, compute_uv=False)
        out[label] = {
            "c_star": c_star,
            "off_frac": float(np.linalg.norm(off) / np.linalg.norm(d)),
            "s2_s1": float(sv[1] / sv[0]),
            "rel": float(np.linalg.norm(d) / np.linalg.norm(ktilde)),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="toroid",
                    choices=("toroid", "rot-ellipse", "w7x"))
    ap.add_argument("--ns", default="8,16,8")
    ap.add_argument("--p", type=int, default=3)
    cli = ap.parse_args()

    ns = tuple(int(v) for v in cli.ns.split(","))
    seq = build_sequence(cli.geometry, ns, cli.p)
    fields = weight_fields(seq)
    ginv, met, jac = fields["ginv_aa"], fields["met_aa"], fields["jac"]

    mu0 = _mesh_amplification(seq)
    n_r = int(ns[0])
    print(f"geometry={cli.geometry} ns={ns} p={cli.p}", flush=True)
    print(f"\nmu_0 (logical, metric-free) = {mu0:.6e}   "
          f"mu_0/n_r = {mu0 / n_r:.4f}   "
          f"-- must not depend on k, c or geometry\n", flush=True)

    surf = face_mean(seq, jac * jnp.sqrt(ginv[0]))
    print(f"predicted ratio  <J sqrt(g^rr)>_(r=1) = {surf:.4e}\n", flush=True)

    hdr = (f"{'k':>2} {'c':>2} {'partner':>9} {'a_exact':>12} {'a_ibp':>12} "
           f"{'a_ibpd':>12} {'a_ibps':>12} {'w_comp':>9} {'ibps/ibpd':>10}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for k, c in TRACE_CASES:
        m_k = {1: ginv[c] * jac, 2: met[c] / jac, 3: 1.0 / jac}[k]
        w_exact = m_k * jnp.sqrt(ginv[0])
        w_ibp = m_k * ginv[0]
        mu_ex = _weak_inverse_amplification(seq, k, c)
        a_exact = face_mean(seq, w_exact) * mu_ex
        a_ibp = face_mean(seq, w_ibp) * mu0
        # Under lumped="diag" the Kronecker factors carry only the k=0 weight
        # g^aa J; the component factor w_comp = mass_weight/J returns as the
        # D^-1/2 sandwich. So the face weight must be the k=0 one too.
        a_ibpd = _face_metric_scalar(seq, k, c, "diag") * mu0
        a_ibps = _face_metric_scalar(seq, k, c, "diag", separate=True) * mu0
        w_comp = face_mean(seq, m_k / jac)
        partner = 3 - c if k == 2 else c
        print(f"{k:>2} {c:>2} {f'V{k-1}[{partner}]':>9} "
              f"{a_exact:>12.4e} {a_ibp:>12.4e} {a_ibpd:>12.4e} "
              f"{a_ibps:>12.4e} {w_comp:>9.3f} "
              f"{a_ibps / a_ibpd:>10.4f}", flush=True)

    print("\nratio should track <J sqrt(g^rr)> above; any spread across rows "
          "is the angular/partner variation the scalar reduction drops.",
          flush=True)

    ref = roundtrip_reference(seq)
    print("\nk=1 radial boundary correction from the EXACT 1-D round trip "
          "F - Ktilde:", flush=True)
    print(f"{'':>6} {'c_star':>12} {'off_frac':>10} {'s2/s1':>10} "
          f"{'||D||/||K||':>12}", flush=True)
    for label, r in ref.items():
        print(f"{label:>6} {r['c_star']:>12.4e} {r['off_frac']:>10.3f} "
              f"{r['s2_s1']:>10.3f} {r['rel']:>12.3e}", flush=True)
    print("\noff_frac small  -> correction is rank one along e, cross term "
          "negligible\noff_frac O(1)   -> the cross term is real and the "
          "E^T M^-1 E piece is not the whole story", flush=True)


if __name__ == "__main__":
    main()
