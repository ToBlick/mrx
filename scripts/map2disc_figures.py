"""Figures for the map2disc / stellarator-symmetry work: the PR's evidence.

Three figures, each reproducible from the shipped Landreman-Paul QA wout:

``qa_vs_vmec.png``
    The map2disc cross-sections against the wout's own last closed flux
    surface, plus the two convergence laws that govern the method: the
    spectral convergence of the fit in the Zernike degree ``M``, and the
    toroidal Nyquist condition on ``n_zeta``.

``stellarator_symmetry.png``
    A deliberately broken map before and after
    :func:`mrx.mappings.stellarator_symmetrize`, with the symmetry defect
    of each of the four routes to a symmetric map.

``qa_vacuum_both_maps.png``
    The QA vacuum field -- the harmonic 2-form of the Dirichlet complex --
    solved twice, once on ``build_gvec_map`` and once on the map2disc map
    of the SAME boundary. The harmonic form is a property of the domain,
    not of the coordinates used to mesh it, so the two must agree; this is
    a physics-level cross-validation of the map rather than a geometric
    one.

    python scripts/map2disc_figures.py --figure all --out docs/research/map2disc_2026-09-13

Options
    --figure {qa,symmetry,vacuum,all}
    --out DIR            figure directory
    --ns N_R,N_T,N_Z     vacuum-field resolution [12,24,12]
    --p P                vacuum-field spline degree [3]
    --seeds N            Poincare seeds per map [16]
    --periods N          field periods per traced line [100]
    --precision {float32,float64}
"""
from __future__ import annotations

import argparse
import io
import os

QA_WOUT = "data/wout_LandremanPaul2021_QA_lowres.nc"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", default="all", choices=("qa", "symmetry", "vacuum", "all"))
    ap.add_argument("--geometry", default=QA_WOUT)
    ap.add_argument("--out", default="docs/research/map2disc_2026-09-13")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--periods", type=int, default=100)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--precision", default="float64", choices=("float32", "float64"))
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Figure 1: map2disc against the VMEC boundary
# ---------------------------------------------------------------------------

def figure_qa(cli, out: str) -> str:
    """Draw the QA cross-sections and the two convergence laws.

    Args:
        cli: Parsed command line.
        out: Output directory.

    Returns:
        Path of the written PNG.
    """
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    from mrx.gvec import StateField, read_equilibrium
    from mrx.map2disc import (fit_disc_map, harmonic_map, lcfs_boundary, map2disc_map,
                              nyquist_n_zeta)

    st = read_equilibrium(cli.geometry)
    nfp = int(st["nfp"])
    boundary_of_zeta = lcfs_boundary(st)
    R_field, Z_field = StateField(st["X1"], nfp), StateField(st["X2"], nfp)

    def exact_lcfs(theta, zeta):
        """The wout's own LCFS in the (R, Z) plane, in closed form."""
        x = jnp.stack([jnp.ones_like(theta), theta, jnp.full_like(theta, zeta)], axis=-1)
        return jax.vmap(R_field)(x), jax.vmap(Z_field)(x)

    fig = plt.figure(figsize=(11.0, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=(1.35, 1.0), hspace=0.34, wspace=0.28)
    planes = (0.0, 0.25, 0.5)

    for col, zeta in enumerate(planes):
        ax = fig.add_subplot(gs[0, col])
        fh = fit_disc_map(boundary_of_zeta(zeta), M=8)

        # Interior coordinate lines: the level sets of the harmonic map.
        for rho in np.linspace(0.2, 1.0, 5):
            t = jnp.linspace(0.0, 2.0 * jnp.pi, 200)
            xy = jax.vmap(lambda a, r=rho: fh(r, a))(t)
            ax.plot(xy[:, 0], xy[:, 1], lw=0.6, color="0.72", zorder=1)
        for angle in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
            r = jnp.linspace(0.0, 1.0, 80)
            xy = jax.vmap(lambda rr, a=angle: fh(rr, a))(r)
            ax.plot(xy[:, 0], xy[:, 1], lw=0.6, color="0.72", zorder=1)

        theta = jnp.linspace(0.0, 1.0, 400, endpoint=False)
        Rx, Zx = exact_lcfs(theta, zeta)
        ax.plot(np.asarray(Rx), np.asarray(Zx), lw=2.6, color="#1f77b4",
                label="VMEC LCFS", zorder=2)
        got = jax.vmap(lambda a: fh(1.0, a))(2.0 * jnp.pi * theta)
        ax.plot(np.asarray(got[::8, 0]), np.asarray(got[::8, 1]), ls="none",
                marker="o", ms=3.2, mfc="none", color="#d62728",
                label="map2disc", zorder=3)
        err = float(jnp.max(jnp.hypot(got[:, 0] - Rx, got[:, 1] - Zx)))
        ax.set_title(rf"$\zeta = {zeta:g}$ of a period    max $|\Delta| = ${err:.1e}",
                     fontsize=9)
        ax.set_xlabel("$R$")
        ax.set_ylabel("$Z$" if col == 0 else "")
        ax.set_aspect("equal")
        if col == 0:
            ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # Spectral convergence of the fit in M, on one plane.
    curve = boundary_of_zeta(0.25)
    g = harmonic_map(curve.resample(4096))
    degrees = (4, 6, 8, 10, 12)
    rho, ang = np.meshgrid(np.array([0.3, 0.6, 0.9]),
                           np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False),
                           indexing="ij")
    rho, ang = jnp.asarray(rho.ravel()), jnp.asarray(ang.ravel())
    want = jnp.stack([rho * jnp.cos(ang), rho * jnp.sin(ang)], axis=1)
    errs = []
    for M in degrees:
        fh = fit_disc_map(curve, M=M)
        got = jax.vmap(g)(jax.vmap(fh)(rho, ang))
        errs.append(float(jnp.max(jnp.abs(got - want))))

    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(degrees, errs, "o-", color="#d62728")
    ax.set_xlabel("Zernike degree $M$")
    ax.set_ylabel("roundtrip error")
    ax.set_title("spectral convergence of the fit", fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # The toroidal Nyquist cliff: n_zeta is a sampling condition.
    n_per = max(abs(int(n)) for n in np.asarray(st["X1"]["n"])) / nfp
    counts = (5, 9, 13, 17, 21)
    theta = jnp.linspace(0.0, 1.0, 9, endpoint=False)
    zeta = jnp.linspace(0.0, 1.0, 11, endpoint=False)
    T, Zt = jnp.meshgrid(theta, zeta, indexing="ij")
    pts = jnp.stack([jnp.ones(T.size), T.ravel(), Zt.ravel()], axis=-1)

    def exact_xyz(x):
        R, phi = R_field(x), 2.0 * jnp.pi * x[2] / nfp
        return jnp.array([R * jnp.cos(phi), -R * jnp.sin(phi), Z_field(x)])

    ref = jax.vmap(exact_xyz)(pts)
    cliff = []
    for n_zeta in counts:
        F = map2disc_map(boundary_of_zeta, nfp=nfp, M=8, n_zeta=n_zeta, sign=-1.0)
        cliff.append(float(jnp.max(jnp.abs(jax.vmap(F)(pts) - ref))))

    ax = fig.add_subplot(gs[1, 1])
    ax.semilogy(counts, cliff, "s-", color="#1f77b4")
    ax.axvline(nyquist_n_zeta(st), color="0.4", ls="--", lw=1.0)
    ax.annotate(rf"$2\max|n|/n_{{fp}}+1 = {nyquist_n_zeta(st)}$",
                xy=(nyquist_n_zeta(st), max(cliff)), xytext=(-4, -6),
                textcoords="offset points", ha="right", va="top", fontsize=8)
    ax.set_xlabel(r"toroidal planes $n_\zeta$")
    ax.set_ylabel("boundary error")
    ax.set_title(rf"Nyquist, not a knob ($\max|n|/n_{{fp}} = {n_per:.0f}$)", fontsize=9)
    ax.set_xticks(counts)
    ax.grid(alpha=0.3, which="both")

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    rows = [rf"$M = {M}$:  {e:.1e}" for M, e in zip(degrees, errs)]
    rows += [""] + [rf"$n_\zeta = {c}$:  {e:.1e}" for c, e in zip(counts, cliff)]
    ax.text(0.0, 1.0, "roundtrip error\n\n" + "\n".join(rows[:len(degrees)])
            + "\n\nboundary error\n\n" + "\n".join(rows[len(degrees) + 1:]),
            va="top", ha="left", fontsize=8.5, family="monospace")

    fig.suptitle("map2disc against the Landreman-Paul QA boundary "
                 f"({cli.precision})", fontsize=11)
    path = os.path.join(out, "qa_vs_vmec.png")
    fig.savefig(path, dpi=cli.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2: stellarator symmetrisation
# ---------------------------------------------------------------------------

def figure_symmetry(cli, out: str) -> str:
    """Show the symmetry projector acting on a deliberately broken map.

    Args:
        cli: Parsed command line.
        out: Output directory.

    Returns:
        Path of the written PNG.
    """
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    from mrx.differential_forms import DifferentialForm
    from mrx.mappings import (STELLARATOR_REFLECTION, SplineMap, extend_map_half_period,
                              rotating_ellipse_map, stellarator_symmetrize,
                              stellarator_symmetry_defect)

    # nfp = 1 keeps the Cartesian components periodic in logical zeta, which
    # is what both the half-period fold and the periodic spline basis assume.
    nfp = 1
    base = rotating_ellipse_map(nfp=nfp)

    def broken(x):
        """A rotating ellipse with a ``Z`` term that is even where it must be odd."""
        r, t, z = x[0], x[1], x[2]
        shift = 0.12 * r * jnp.cos(2.0 * jnp.pi * z) * jnp.cos(2.0 * jnp.pi * t)
        return base(x) + jnp.array([0.0, 0.0, shift])

    fixed = stellarator_symmetrize(broken)
    half = extend_map_half_period(broken, nfp=nfp)

    # The same projector in coefficient space: an index permutation on the
    # two uniform periodic angular axes, applied once when the map is built.
    basis_0 = DifferentialForm(0, (6, 10, 10), (2, 2, 2),
                               ("clamped", "periodic", "periodic"))
    br, bt, bz = basis_0.Λ
    grid = jnp.meshgrid(br.greville_points(), bt.greville_points(),
                        bz.greville_points(), indexing="ij")
    vals = jax.vmap(broken)(jnp.stack(grid, axis=-1).reshape(-1, 3))
    coeffs = vals.T.reshape(3, br.n, bt.n, bz.n)
    for axis, basis in enumerate((br, bt, bz), start=1):
        moved = jnp.moveaxis(coeffs, axis, 0)
        coeffs = jnp.moveaxis(jnp.linalg.solve(
            basis.collocation_matrix(), moved.reshape(moved.shape[0], -1)
        ).reshape(moved.shape), 0, axis)
    spline = SplineMap(coeffs.reshape(3, -1), jnp.eye(basis_0.n), basis_0,
                       stellarator_symmetric=True)

    def reflected(F):
        """``S F(r, -theta, -zeta)`` -- what stellarator symmetry says ``F`` is."""
        return lambda x: STELLARATOR_REFLECTION * F(jnp.array([x[0], -x[1], -x[2]]))

    def section(F, zeta, rho=1.0, n=400):
        """The (R, Z) cross-section of ``F`` at one ``zeta``."""
        t = jnp.linspace(0.0, 1.0, n, endpoint=False)
        pts = jnp.stack([jnp.full(n, rho), t, jnp.full(n, zeta)], axis=-1)
        xyz = jax.vmap(F)(pts)
        return np.asarray(jnp.hypot(xyz[:, 0], xyz[:, 1])), np.asarray(xyz[:, 2])

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))
    zeta = 0.17
    for ax, (F, name) in zip(axes[:2], ((broken, "before"), (fixed, "after"))):
        for rho, alpha in ((1.0, 1.0), (0.6, 0.5)):
            R, Z = section(F, zeta, rho)
            Rr, Zr = section(reflected(F), zeta, rho)
            ax.plot(R, Z, lw=2.4, color="#1f77b4", alpha=alpha,
                    label=r"$F(r,\theta,\zeta)$" if rho == 1.0 else None)
            ax.plot(Rr[::6], Zr[::6], ls="none", marker="o", ms=3.0, mfc="none",
                    color="#d62728", alpha=alpha,
                    label=r"$S\,F(r,-\theta,-\zeta)$" if rho == 1.0 else None)
        defect = float(stellarator_symmetry_defect(F, jnp.array(
            [[0.4, 0.1, zeta], [0.7, 0.6, zeta], [1.0, 0.3, zeta]])))
        ax.set_title(f"{name} symmetrisation    defect {defect:.1e}", fontsize=9)
        ax.set_xlabel("$R$")
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("$Z$")
    axes[0].legend(fontsize=8, loc="upper right", framealpha=0.9)

    # Every route to a symmetric map, on the same probe points. The probe
    # avoids zeta = 0 and 1/2: those are the fold lines of the half-period
    # extension, where it returns F_half's own -- unsymmetrised -- value.
    probe = jnp.asarray(np.column_stack([
        np.linspace(0.1, 1.0, 24),
        np.linspace(0.0, 1.0, 24, endpoint=False),
        np.linspace(0.02, 0.98, 24)[::-1]]))
    routes = [("broken\nmap", broken), ("symmetrize", fixed),
              ("half-period\nextension", half), ("SplineMap\nprojector", spline)]
    defects = [max(float(stellarator_symmetry_defect(F, probe)), 1e-18)
               for _, F in routes]

    ax = axes[2]
    bars = ax.bar(range(len(routes)), defects,
                  color=["#d62728"] + ["#2ca02c"] * (len(routes) - 1))
    ax.set_yscale("log")
    ax.set_xticks(range(len(routes)))
    ax.set_xticklabels([n for n, _ in routes], fontsize=8)
    ax.set_ylabel(r"$\max\,|F(r,-\theta,-\zeta) - S\,F(r,\theta,\zeta)|$")
    ax.set_title(r"symmetry defect, 24 points off the $\zeta \in \{0, 1/2\}$ folds",
                 fontsize=9)
    ax.grid(alpha=0.3, axis="y", which="both")
    for bar, d in zip(bars, defects):
        ax.annotate(f"{d:.0e}", (bar.get_x() + bar.get_width() / 2, d),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7.5)

    fig.suptitle(f"stellarator symmetry in the mappings ({cli.precision})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out, "stellarator_symmetry.png")
    fig.savefig(path, dpi=cli.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 3: the QA vacuum field on both maps
# ---------------------------------------------------------------------------

def _trim(image, tol: float = 0.995):
    """Crop the uniform white margin a 3-D axes leaves around its render.

    Args:
        image: ``(h, w, 4)`` RGBA array.
        tol: Channel value above which a pixel counts as background.

    Returns:
        The cropped array.
    """
    import numpy as np

    ink = np.any(image[:, :, :3] < tol, axis=2)
    rows, cols = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return image
    return image[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def _vacuum_field(geometry, ns, p, map_source):
    """Solve the vacuum harmonic 2-form on one map of one geometry.

    Args:
        geometry: Path of the VMEC wout.
        ns: ``(n_r, n_theta, n_zeta)``.
        p: Spline degree.
        map_source: ``"equilibrium"`` or ``"map2disc"``.

    Returns:
        ``(seq, B, diagnostics)``: the sequence, the normalised 2-form DoF
        vector, and a dict with its divergence, curl ratio and Rayleigh
        quotient.
    """
    from mrx.geometry import build_sequence
    from mrx.nullspace import compute_nullspaces, get_nullspace, harmonic_rayleigh
    from mrx.relaxation import compute_divergence_norm, compute_force

    seq, _ = build_sequence(geometry, ns, p, map_source=map_source)
    compute_nullspaces(seq)
    B = get_nullspace(seq.get_operators(), 2, True)[0]
    B = B / float(seq.l2_norm(B, 2))
    _, _, J, _, _ = compute_force(B, seq)
    return seq, B, {"div": float(compute_divergence_norm(B, seq)),
                    "curl": float(seq.l2_norm(J, 1)),
                    "rayleigh": float(harmonic_rayleigh(seq, B, 2))}


def figure_vacuum(cli, out: str) -> str:
    """Solve the QA vacuum field on both maps and compare the two.

    Args:
        cli: Parsed command line.
        out: Output directory.

    Returns:
        Path of the written PNG.
    """
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    from mrx.differential_forms import DiscreteFunction, Pushforward
    from mrx.geometry import geometry_nfp
    from mrx.plotting import get_2d_grids, plot_torus
    from mrx.poincare import (logical_field, seed_from_axis, section_RZ,
                              surface_label, trace_and_classify)

    ns = tuple(int(v) for v in cli.ns.split(","))
    nfp = geometry_nfp(cli.geometry)
    sources = (("equilibrium", "#1f77b4"), ("map2disc", "#d62728"))
    panels, profiles, boundary, notes = {}, {}, {}, {}

    # Two Hodge solves and two field-line traces; cached under the ignored
    # outputs/ tree so the figure can be restyled without repeating them.
    os.makedirs("outputs/map2disc_figures", exist_ok=True)
    cache = os.path.join("outputs/map2disc_figures",
                         f"vacuum_{'x'.join(map(str, ns))}_p{cli.p}_"
                         f"{cli.precision}.npz")
    if os.path.exists(cache):
        blob = np.load(cache, allow_pickle=True)
        panels = {s: blob[f"panel_{s}"] for s, _ in sources}
        boundary = {s: blob[f"boundary_{s}"] for s, _ in sources}
        profiles = {s: (blob[f"a_{s}"], blob[f"iota_{s}"], str(blob["xlabel"]))
                    for s, _ in sources}
        notes = {s: dict(blob[f"notes_{s}"].item()) for s, _ in sources}
        print(f"[cache] reusing {cache}")

    for source, _ in sources:
        if source in panels:
            continue
        seq, B, diag = _vacuum_field(cli.geometry, ns, cli.p, source)
        notes[source] = diag
        print(f"[{source}] ||div B|| = {diag['div']:.2e}, "
              f"||curl B|| / ||B|| = {diag['curl']:.2e}, "
              f"Rayleigh = {diag['rayleigh']:.2e}")

        B_phys = Pushforward(DiscreteFunction(B, seq.basis_2, seq.E(2, True)), seq.map, 2)

        def B_mag(x, f=B_phys):
            return jnp.linalg.norm(f(x))

        # |B| on the torus. Both maps share the LCFS exactly, so the
        # surface these are drawn on is the same surface.
        n = 40
        grids_pol = [get_2d_grids(seq.map, cut_axis=2, cut_value=float(z),
                                  nx=n, ny=n, nz=1) for z in np.arange(4) / 4]
        grid_surface = get_2d_grids(seq.map, cut_axis=0, cut_value=1.0 - 1e-6,
                                    ny=3 * n, nz=3 * n, invert_z=True)
        fig, _ = plot_torus(B_mag, grids_pol, grid_surface, cstride=8,
                            gridlinewidth=0.3, elev=25, azim=40, cbar_label=r"$|B|$")
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=cli.dpi, bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        panels[source] = plt.imread(buffer)

        # |B| along the boundary at zeta = 0. r = 1 is the one place the
        # two maps agree pointwise, so this compares the same physical
        # points and needs no inversion.
        t = jnp.linspace(0.0, 1.0, 240, endpoint=False)
        pts = jnp.stack([jnp.full(240, 1.0 - 1e-6), t, jnp.zeros(240)], axis=-1)
        boundary[source] = np.asarray(jax.vmap(B_mag)(pts))

        # iota against the effective minor radius: both are physical, so
        # the profiles are directly comparable across the two maps.
        field = logical_field(seq, jnp.asarray(B), 2, True)
        seeds = seed_from_axis(field, cli.seeds, 8, n_rays=4, steps_per_period=32)
        res = trace_and_classify(field, seeds, nfp, n_periods=cli.periods,
                                 steps_per_period=32, saves_per_period=8)
        R, Z, aR, aZ, _, _, _, _ = section_RZ(seq, res["ys"], res["axis"], 8, 0.0)
        # Both midplane crossings of each line, shape (n_lines, 2): a
        # property of the physical curve, so the two maps are comparable.
        a_eff, xlabel = surface_label(R, Z, aR, aZ)
        good = np.asarray(~(res["escaped"] | ~res["ok"] | res["chaotic"]))
        profiles[source] = (np.asarray(a_eff)[good],
                            np.asarray(res["iota"])[good], xlabel)

    if not os.path.exists(cache):
        np.savez_compressed(cache, xlabel=profiles["equilibrium"][2], **{
            k: v for s, _ in sources for k, v in (
                (f"panel_{s}", panels[s]), (f"boundary_{s}", boundary[s]),
                (f"a_{s}", profiles[s][0]), (f"iota_{s}", profiles[s][1]),
                (f"notes_{s}", np.array(notes[s], dtype=object)))})

    fig = plt.figure(figsize=(11.5, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.45, 1.0), hspace=0.16, wspace=0.22)
    for col, (source, colour) in enumerate(sources):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(_trim(panels[source]))
        ax.axis("off")
        ax.set_title(f'map_source="{source}"', fontsize=10, color=colour)

    ax = fig.add_subplot(gs[1, 0])
    t = np.linspace(0.0, 1.0, 240, endpoint=False)
    for source, colour in sources:
        ax.plot(t, boundary[source], color=colour, lw=1.8, label=source)
    gap = np.max(np.abs(boundary["equilibrium"] - boundary["map2disc"]))
    rel = gap / np.mean(boundary["equilibrium"])
    ax.set_xlabel(r"poloidal angle $\theta / 2\pi$")
    ax.set_ylabel(r"$|B|$ on the boundary, $\zeta = 0$")
    ax.set_title(rf"same physical surface: max difference {rel:.1%} of $\langle|B|\rangle$",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    outboard = {}
    for source, colour in sources:
        a_eff, iota, xlabel = profiles[source]
        # A line that never reaches the midplane has no crossing to label.
        a_eff, iota = np.asarray(a_eff), np.asarray(iota)
        keep = np.all(np.isfinite(a_eff), axis=1) & np.isfinite(iota)
        a_eff, iota = a_eff[keep], iota[keep]
        ax.scatter(a_eff.ravel(),
                   np.broadcast_to(iota[:, None], a_eff.shape).ravel(),
                   s=14.0, color=colour, alpha=0.8,
                   label=f"{source}  ({iota.size} regular lines)")
        order = np.argsort(a_eff.max(axis=1))
        outboard[source] = (a_eff.max(axis=1)[order], iota[order])

    # Compare the two profiles where they overlap: iota against a physical
    # abscissa is the same function of the domain, whatever the coordinates.
    ref_R, ref_iota = outboard["equilibrium"]
    got_R, got_iota = outboard["map2disc"]
    lo, hi = max(ref_R.min(), got_R.min()), min(ref_R.max(), got_R.max())
    inside = (ref_R >= lo) & (ref_R <= hi)
    d_iota = np.max(np.abs(np.interp(ref_R[inside], got_R, got_iota)
                           - ref_iota[inside]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\iota$")
    ax.set_title(rf"coordinate-independent physics: $\max|\Delta\iota| = ${d_iota:.1e}"
                 rf" on $\iota \approx {ref_iota.mean():.3f}$", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    diag = notes["equilibrium"]
    fig.suptitle("the QA vacuum field is a property of the domain, so both maps "
                 f"must find it\nns={ns}, p={cli.p}, {cli.precision};  "
                 rf"$\|\mathrm{{curl}}\,B\| / \|B\| = ${diag['curl']:.1e} "
                 f"(equilibrium) vs {notes['map2disc']['curl']:.1e} (map2disc)",
                 fontsize=11)
    path = os.path.join(out, "qa_vacuum_both_maps.png")
    fig.savefig(path, dpi=cli.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def main(cli):
    """Render the requested figures.

    Args:
        cli: Parsed command line.
    """
    os.environ["MRX_DTYPE"] = cli.precision
    import matplotlib
    matplotlib.use("Agg")

    os.makedirs(cli.out, exist_ok=True)
    wanted = ("qa", "symmetry", "vacuum") if cli.figure == "all" else (cli.figure,)
    builders = {"qa": figure_qa, "symmetry": figure_symmetry, "vacuum": figure_vacuum}
    for name in wanted:
        print(f"  -> {builders[name](cli, cli.out)}")


if __name__ == "__main__":
    main(parse_args())
