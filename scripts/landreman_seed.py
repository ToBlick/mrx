#!/usr/bin/env python
"""Seed island chains into a converged state by the energy criterion (one GPU; Sec. 6.2, Tab. 4).

paper/seed.py of branch seed-width-fix (64fa2b1) for the Landreman equilibria, with the profile form of
``parallel_seed`` (a0bf002, not on this branch) inlined as :func:`parallel_seed_profile`, and the low-shear columns
of every resonance: ``d_r^2 iota``, the pendulum validity ``w |iota''| / |iota'|``, the shearless width ``w3``
(:func:`shearless_width`) and the radial extent over which ``|iota - nfp n / m|`` stays below the island's
resonance width (:func:`locking_extent`).

    python scripts/landreman_seed.py --run RUN --step N --out SEEDED.h5 [--chain M,N ...] [--scale Q]

Every resonance (m, n) with m <= n_theta / 2 and iota = nfp n / m inside the run's iota range gets SIESTA's
parallel seed, dB = curl(A B / |B|) with A = a(r) cos(2 pi (m theta - n zeta)) (mrx.initial_conditions.
parallel_seed), its profile a(r) free in the sequence's own radial basis within 1/m of the resonant radius r_mn
(the wall function and the two innermost functions of the polar surgery left out). All chains are solved together
for the amplitudes of least energy, a* = -G^-1 g with g_i = <B, dB_i> and G_ij = <dB_i, dB_j>, and B + Q sum a*_i
dB_i over the --chain chains (all by default) is written as a checkpoint for relax.py --restart.

Next to it, <out>.json lists every resonance: r_mn, |d_r iota|, the resonant normal field of its a*, dBr, the
amplitude of the (m, n) harmonic of sqrt(g) dB^r at r_mn over <sqrt(g) B^zeta>(r_mn) (the Fourier transform in the
straight-field-line angle theta* = theta + lambda of the wout), and the pendulum width
w = sqrt(8 dBr nfp / (pi m |d_r iota|)).

The one script of paper/ on mrx internals: no command-line tool seeds a checkpoint.
"""
import argparse
import json
import os

import numpy as np

N_R, N_THETA, N_ZETA = 320, 64, 32     # the grid of the harmonic analysis
BATCH = 65536


def parallel_seed_profile(seq, B, m, n, profile, r_grid):
    """``dB = curl(A_par B / |B|)``, ``A_par = profile(rho) cos(2 pi (m theta - s n zeta))``: the profile form of
    :func:`mrx.initial_conditions.parallel_seed` (seed-width-fix a0bf002: no ``(1 - rho^2)`` factor, the Dirichlet
    1-form space already pins the wall). Returns ``(dB, ||div dB|| / ||dB||)``."""
    import jax.numpy as jnp
    seq = seq.odd
    Bq = seq.evaluate_at_quadrature(B, 2, dirichlet=True)
    g_B = jnp.einsum('qij,qj->qi', seq.metric_jkl, Bq)
    Bhat_cov = g_B / jnp.sqrt(jnp.einsum('qi,qi->q', g_B, Bq))[:, None]
    x, w = seq.quad.x, seq.quad.w
    s = float(jnp.sign(jnp.sum(w * Bq[:, 1]) / jnp.sum(w * Bq[:, 2])))
    A_par = (jnp.interp(x[:, 0], jnp.asarray(r_grid), jnp.asarray(profile))
             * jnp.cos(2.0 * jnp.pi * (m * x[:, 1] - s * n * x[:, 2])))
    load = seq.vector_load_values(A_par[:, None] * Bhat_cov, 1, 1, dirichlet_n=True)
    dA = seq.apply_inverse_mass_matrix(load, 1, dirichlet=True)
    dB = seq.apply_incidence_matrix(dA, 1, dirichlet_in=True, dirichlet_out=True)
    div = float(seq.l2_norm(seq.apply_incidence_matrix(dB, 2, dirichlet_in=True, dirichlet_out=True), 3)
                / seq.l2_norm(dB, 2))
    return dB, div


def shearless_width(a, m, nfp, d2iota):
    """Island width at a shearless resonance. With x = r - r_mn, the resonant angle chi = 2 pi (m theta* - n zeta)
    and zeta in field periods, the field lines are dr/dzeta = a cos chi, dchi/dzeta = 2 pi m (iota - iota_mn) / nfp,
    i.e. the Hamiltonian K = (2 pi m / nfp) int (iota - iota_mn) dr - a sin chi. For iota = iota_mn + iota'' x^2 / 2
    (the resonance at the extremum) K = (pi m iota'' / (3 nfp)) x^3 - a sin chi; both fixed points sit at x = 0 and
    the two separatrices K = +-a reach F(x) = +-2a, x = +-(6 a nfp / (pi m |iota''|))^(1/3). The full width is
    w3 = 2 (6 a nfp / (pi m |iota''|))^(1/3) = (C a / (m |iota''|))^(1/3), C = 48 nfp / pi. (The pendulum, iota =
    iota_mn + iota' x, gives K = (pi m iota' / nfp) x^2 - a sin chi and w = sqrt(8 a nfp / (pi m |iota'|)).)"""
    return (48.0 * nfp * a / (np.pi * m * abs(d2iota))) ** (1.0 / 3.0)


def locking_extent(iota, r0, target, diota_res, rr):
    """The contiguous radial interval around ``r0`` on which ``|iota(r) - target| < diota_res``: the band of
    surfaces whose mismatch the resonance can absorb, which caps the chain's width."""
    inside = np.abs(iota(rr) - target) < diota_res
    i0 = int(np.argmin(np.abs(rr - r0)))
    if not inside[i0]:
        return 0.0
    lo = i0
    while lo > 0 and inside[lo - 1]:
        lo -= 1
    hi = i0
    while hi < len(rr) - 1 and inside[hi + 1]:
        hi += 1
    return float(rr[hi] - rr[lo])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="a scripts/relax.py run: its relax.json and checkpoints/")
    ap.add_argument("--step", type=int, required=True, help="seed checkpoints/state_<step>.h5")
    ap.add_argument("--out", required=True, help="the seeded checkpoint (.h5); the table goes to the .json beside it")
    ap.add_argument("--chain", action="append", default=None,
                    help="M,N: add only this chain's part of the joint optimum (repeatable) [every chain]")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the added perturbation [1]")
    cli = ap.parse_args()
    os.environ["MRX_DTYPE"] = os.environ["MRX_RESIDUAL_DTYPE"] = "float64"

    import jax
    import jax.numpy as jnp
    import scipy.io
    from scipy.interpolate import CubicSpline

    import mrx
    from mrx.differential_forms import DiscreteFunction
    from mrx.experimental.islands import resonances
    from mrx.geometry import build_sequence
    from mrx.gvec import load_clebsch
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import TimeStepper, initial_state, read_checkpoint, write_checkpoint

    params = json.load(open(os.path.join(cli.run, "relax.json")))["params"]
    ns, p = tuple(params["ns"]), params["p"]
    mrx.MAP_BATCH_SIZE_INNER = params["map_batch"]
    seq, _ = build_sequence(params["geometry_path"], ns, p, params["solve_maxiter"], tol=params["solve_tol"],
                            nfp=params["nfp"], knots=params["knots"], symmetry=params["symmetry"])
    compute_nullspaces(seq)
    ts = TimeStepper(seq=seq)
    ckpt = os.path.join(cli.run, "checkpoints", f"state_{cli.step:06d}.h5")
    state, step = read_checkpoint(ckpt, ts)
    B = state.B_n
    cb = load_clebsch(seq.equilibrium, nfp=seq.nfp)
    nfp = int(cb["nfp"])
    lam = lambda x: cb["lam_h"](x) / (2.0 * jnp.pi)   # noqa: E731  (lambda in turns)

    # the rotational transform of the wout, a spline in r = sqrt(s)
    with scipy.io.netcdf_file(params["geometry_path"], "r", mmap=False) as d:
        iota_f = np.abs(np.asarray(d.variables["iotaf"].data, float))
    iota = CubicSpline(np.sqrt(np.linspace(0.0, 1.0, iota_f.size)), iota_f)
    rr = np.linspace(1e-3, 1.0, 40001)
    h_r = 1.0 / (ns[0] - p)

    # the radial basis, sampled: the profile of one seed block is one of its functions
    rb = seq.basis_0.bases[0].bases[0]
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
    print(f"[seed] {ckpt}: step {step}, {len(chains)} resonances with m <= {ns[1] // 2}, h_r {h_r:.4f}", flush=True)

    dB, owner = [], []
    for c, (m, n, _, _, funcs) in enumerate(chains):
        for i in funcs:
            dB.append(parallel_seed_profile(seq, B, m, n, basis[i], r_fine)[0])
            owner.append(c)
    owner = np.array(owner)
    MdB = [seq.apply_mass_matrix(d, 2, dirichlet=True) for d in dB]
    g = np.array([float(B @ Md) for Md in MdB])
    G = np.array([[float(d @ Md) for Md in MdB] for d in dB])
    a = -np.linalg.solve(0.5 * (G + G.T), g)
    energy = 0.5 * float(seq.l2_norm(B, 2)) ** 2
    print(f"[seed] {len(dB)} blocks, the joint optimum lowers the energy by {-0.5 * (a @ g) / energy * 1e6:.4f} ppm",
          flush=True)

    # the harmonics on the grid: sqrt(g) B^i is the logical 2-form's value. The (m, n) coefficient of dr/dzeta =
    # B^r / B^zeta in (theta*, zeta), theta* = theta + lambda: sqrt(g) B^zeta = Phi' (1 + d_theta lambda) in the wout's
    # coordinates cancels the Jacobian of theta -> theta*, so the sum over the theta grid has uniform weights. The
    # resonant angle takes the seed's sign of iota (parallel_seed's flux ratio)
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

    rows = []
    for c, (m, n, r0, diota, _) in enumerate(chains):
        dBc = sum(float(a[i]) * dB[i] for i in np.nonzero(owner == c)[0])
        br = on_grid(two_form, dBc)[:, 0].reshape(shape)
        # twice the complex coefficient of A cos(m theta* - s n zeta) is its peak amplitude A
        cr = 2.0 * (br * np.exp(-2j * np.pi * (m * (TH + lam_x) - s * n * ZE))).mean(axis=(1, 2))
        dBr = abs(np.interp(r0, rg, cr.real) + 1j * np.interp(r0, rg, cr.imag)) / abs(np.interp(r0, rg, Bz))
        w = float(np.sqrt(8.0 * dBr * nfp / (np.pi * m * diota)))
        d2 = abs(float(iota.derivative(2)(r0)))
        w3 = shearless_width(dBr, m, nfp, d2)
        target = nfp * n / m
        # the resonance width in iota: the largest mismatch inside the separatrix, |iota'| w / 2 (pendulum) or
        # |iota''| (w3 / 2)^2 / 2 (shearless); the larger of the two models' bands is reported
        ext = locking_extent(iota, r0, target, max(diota * w / 2, d2 * (w3 / 2) ** 2 / 2), rr)
        rows.append(dict(m=m, n=n, r=r0, diota=diota, d2iota=d2, a_star=[float(v) for v in a[owner == c]],
                         dBr=float(dBr), w=w, w_over_hr=w / h_r, validity=w * d2 / diota, w3=w3,
                         locking_extent=ext))
        print(f"  ({m},{n}) r_mn {r0:.4f}  iota' {diota:.4f} iota'' {d2:.4f}  dBr {dBr:.3e}  w {w:.4f} = "
              f"{w / h_r:.2f} h_r  w iota''/iota' {w * d2 / diota:.3f}  w3 {w3:.4f}  locking extent {ext:.4f}",
              flush=True)

    keep = None if cli.chain is None else [tuple(int(v) for v in mn.split(",")) for mn in cli.chain]
    assert keep is None or set(keep) <= {c[:2] for c in chains}, (keep, [c[:2] for c in chains])
    blocks = [i for i in range(len(dB)) if keep is None or chains[owner[i]][:2] in keep]
    seeded = B + cli.scale * sum(float(a[i]) * dB[i] for i in blocks)
    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    write_checkpoint(cli.out, initial_state(seeded, ts, step=step), step)
    with open(os.path.splitext(cli.out)[0] + ".json", "w") as fh:
        json.dump(dict(checkpoint=ckpt, step=step, ns=list(ns), p=p, nfp=nfp, h_r=h_r, chains=keep, scale=cli.scale,
                       resonances=rows), fh, indent=1)
    print(f"[seed] wrote {cli.out} ({len(blocks)} of {len(dB)} blocks x {cli.scale:g}, "
          f"||dB|| / ||B|| = {float(seq.l2_norm(seeded - B, 2) / seq.l2_norm(B, 2)):.3e}) and its .json", flush=True)


if __name__ == "__main__":
    main()
