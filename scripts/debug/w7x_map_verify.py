"""Cheap verification that the W7-X projection / map construction works.

NO solves, NO mass-matrix assembly -- geometry only. Two checks:

  (1) Resolution & p independence.  The physical geometry is fixed, so the
      Jacobian extrema  J_min, J_max  and the volume integral
      V = sum_q w_q J(x_q)  (one field period) must converge to
      resolution-/p-independent values as the interpolation resolution rises.
      We sweep ns at fixed p, then p at fixed ns, and tabulate.

  (2) Visual sanity.  Render the map with mrx.plotting.plot_torus (a field
      period coloured by the Jacobian), plus a poloidal cross-section panel
      (the W7-X bean->triangle shaping) and a full 5-period torus.

The only non-trivial cost is det(jacfwd(F)) at the quadrature points; it is
batched via mrx.MAP_BATCH_SIZE_INNER (env W7X_MAP_BATCH, default 256).

Run:
  W7X_MAP_BATCH=256 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      .venv/bin/python scripts/debug/w7x_map_verify.py
Figures + table land under outputs/w7x_map_verify/<date>/<time>/.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")  # headless

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import mrx  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.differential_forms import jacobian_determinant  # noqa: E402
from mrx.plotting import get_2d_grids, plot_torus  # noqa: E402
from w7x_geometry import _interp_accuracy, build_w7x_map  # noqa: E402

TYPES = ("clamped", "periodic", "periodic")
NFP = 5


def _outdir():
    import datetime
    now = datetime.datetime.now()
    d = os.path.join("outputs", "w7x_map_verify",
                     now.strftime("%Y-%m-%d"), now.strftime("%H-%M-%S"))
    os.makedirs(d, exist_ok=True)
    return d


def jacobian_stats(map_func, ns, p):
    """J_min, J_max, volume integral over one field period -- no solves.

    Uses only the quadrature points/weights, which exist right after
    DeRhamSequence construction (no evaluate_1d / mass assembly needed).
    """
    seq = DeRhamSequence(ns, (p, p, p), 2 * p, TYPES, polar=False)
    jdet = jacobian_determinant(map_func)
    J = jax.lax.map(jdet, seq.quad.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    J = np.asarray(J)
    w = np.asarray(seq.quad.w)
    return dict(nq=J.size, jmin=float(J.min()), jmax=float(J.max()),
                vol=float(w @ J), nneg=int((J <= 0).sum()))


def run_sweep(outdir):
    """Sweep resolution (fixed p) and p (fixed ns); tabulate the invariants."""
    base_ns, base_p = (12, 24, 12), 3
    configs = [
        ("res", (8, 16, 8), 3),
        ("res", base_ns, base_p),
        ("res", (16, 32, 16), 3),
        ("p", base_ns, 2),
        ("p", base_ns, 4),
    ]
    header = (f"{'group':<5} {'ns':<12} {'p':<2} {'nq':>7} "
              f"{'J_min':>11} {'J_max':>11} {'volume':>13} "
              f"{'#J<=0':>6} {'interpR_max':>11} {'interpZ_max':>11}")
    lines = [header, "-" * len(header)]
    print(header, flush=True)
    rows = []
    for group, ns, p in configs:
        map_func, info = build_w7x_map(map_ns=ns, p=p)
        acc = _interp_accuracy(info)
        st = jacobian_stats(map_func, ns, p)
        row = (f"{group:<5} {str(ns):<12} {p:<2} {st['nq']:>7} "
               f"{st['jmin']:>11.4e} {st['jmax']:>11.4e} {st['vol']:>13.6e} "
               f"{st['nneg']:>6} {acc['R_max']:>11.2e} {acc['Z_max']:>11.2e}")
        print(row, flush=True)
        lines.append(row)
        rows.append((group, ns, p, st, acc))
        # keep the baseline map for plotting
        if (ns, p) == (base_ns, base_p):
            base_map = map_func

    # spread of each invariant across the sweep (how resolution/p-independent)
    jmins = np.array([r[3]["jmin"] for r in rows])
    jmaxs = np.array([r[3]["jmax"] for r in rows])
    vols = np.array([r[3]["vol"] for r in rows])

    def spread(a):
        return f"min={a.min():.6e} max={a.max():.6e} rel-spread={(a.max()-a.min())/abs(a.mean()):.2%}"
    lines += ["", "Resolution/p independence (spread across all configs):",
              f"  J_min : {spread(jmins)}",
              f"  J_max : {spread(jmaxs)}",
              f"  volume: {spread(vols)}"]
    for s in lines[-4:]:
        print(s, flush=True)

    with open(os.path.join(outdir, "jacobian_sweep.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return base_map, base_ns


def plot_field_period(map_func, outdir, n_cuts=6):
    """plot_torus: one field period, poloidal cuts coloured by the Jacobian."""
    p_h = jacobian_determinant(map_func)  # scalar per logical point
    cuts = jnp.linspace(0.0, 1.0, n_cuts, endpoint=False)
    grids_pol = [get_2d_grids(map_func, cut_axis=2, cut_value=float(v),
                              nx=28, ny=48, nz=1) for v in cuts]
    grid_surface = get_2d_grids(map_func, cut_axis=0, cut_value=1.0,
                                ny=96, nz=96)
    fig, ax = plot_torus(p_h, grids_pol, grid_surface,
                         gridlinewidth=1, cstride=6, elev=22, azim=40)
    ax.set_title("W7-X field period — surface = $\\rho{=}1$, colour = Jacobian $J$")
    out = os.path.join(outdir, "w7x_field_period.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}", flush=True)


def plot_crosssections(map_func, outdir, n=6):
    """Poloidal cross-sections across a field period -> bean->triangle shaping."""
    zetas = np.linspace(0.0, 1.0, n, endpoint=False)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2), squeeze=False)
    nr, nt = 9, 80
    rr = jnp.linspace(1e-3, 1.0, nr)
    tt = jnp.linspace(0.0, 1.0, nt)
    for j, z in enumerate(zetas):
        ax = axes[0][j]
        for r in rr:
            pts = jnp.stack([jnp.full(nt, r), tt, jnp.full(nt, z)], axis=1)
            y = jax.lax.map(map_func, pts,
                            batch_size=mrx.MAP_BATCH_SIZE_INNER)
            R = np.hypot(np.asarray(y[:, 0]), np.asarray(y[:, 1]))
            Z = np.asarray(y[:, 2])
            ax.plot(R, Z, lw=0.8, color="0.3")
        ax.set_aspect("equal")
        ax.set_title(f"$\\zeta={z:.2f}$ (period)")
        ax.set_xlabel("R");
        if j == 0:
            ax.set_ylabel("Z")
    fig.suptitle("W7-X poloidal cross-sections (flux surfaces) across one field period")
    fig.tight_layout()
    out = os.path.join(outdir, "w7x_crosssections.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}", flush=True)


def plot_full_torus(map_func, outdir, nt=96, nz=96):
    """Replicate the field period nfp times -> recognisable full stellarator."""
    surf = get_2d_grids(map_func, cut_axis=0, cut_value=1.0, ny=nt, nz=nz)
    X0, Y0, Z0 = (np.asarray(surf[2][i]) for i in range(3))
    # colour by Jacobian on the boundary
    p_h = jacobian_determinant(map_func)
    Jb = np.asarray(jax.lax.map(p_h, surf[0],
                                batch_size=mrx.MAP_BATCH_SIZE_INNER)
                    ).reshape(X0.shape)
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    vmin, vmax = float(Jb.min()), float(Jb.max())
    norm = (Jb - vmin) / max(vmax - vmin, 1e-30)
    colors = plt.cm.plasma(norm)
    for k in range(NFP):
        phi = -k * 2.0 * np.pi / NFP  # matches F's -sin handedness
        Xk = X0 * np.cos(phi) - Y0 * np.sin(phi)
        Yk = X0 * np.sin(phi) + Y0 * np.cos(phi)
        ax.plot_surface(Xk, Yk, Z0, facecolors=colors, rstride=1, cstride=1,
                        shade=False, linewidth=0, antialiased=False)
    ax.set_box_aspect((1, 1, 0.4))
    ax.set_title(f"W7-X full torus ({NFP} field periods), colour = Jacobian $J$")
    ax.view_init(elev=55, azim=30)
    out = os.path.join(outdir, "w7x_full_torus.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}", flush=True)


def main():
    mrx.MAP_BATCH_SIZE_INNER = int(os.environ.get("W7X_MAP_BATCH", "256"))
    outdir = _outdir()
    print(f"[w7x-verify] MAP_BATCH_SIZE_INNER={mrx.MAP_BATCH_SIZE_INNER}  "
          f"outdir={outdir}", flush=True)

    base_map, base_ns = run_sweep(outdir)

    print("[w7x-verify] rendering geometry ...", flush=True)
    plot_field_period(base_map, outdir)
    plot_crosssections(base_map, outdir)
    plot_full_torus(base_map, outdir)
    print(f"[w7x-verify] done. artifacts in {outdir}", flush=True)


if __name__ == "__main__":
    main()
