"""Figures for the map2disc / stellarator-symmetry work: the PR's evidence.

Five figures, all reproducible from the shipped Landreman-Paul QA and
NCSX (li383) wout files.

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

``ncsx_vs_vmec.png``
    The same geometric comparison on NCSX, whose cross-sections are
    crescents rather than beans. The ``zeta = 0`` plane is the one that
    motivated :func:`mrx.map2disc._interior_seed`: its boundary centroid
    lies outside the plasma, so it gets its own panel.

``ncsx_relaxation.png``
    NCSX relaxed on both maps, before and after. The two maps start from
    DIFFERENT initial fields -- see :func:`_relaxation` -- so this is a
    comparison at unequal helicity, and the figure says so.

    python scripts/map2disc_figures.py --figure all
    python scripts/map2disc_figures.py --figure ncsx-relax --precision float32

``all`` is the four geometry and vacuum figures. ``ncsx-relax`` is two
nonlinear descents, about twenty minutes, so it is asked for by name; the
expensive solves of both are cached under ``outputs/map2disc_figures``.

Options
    --figure {qa,symmetry,vacuum,ncsx,ncsx-relax,all}
    --out DIR            figure directory
    --ns N_R,N_T,N_Z     vacuum-field resolution [12,24,12]
    --p P                vacuum-field spline degree [3]
    --relax-ns N_R,N_T,N_Z   relaxation resolution [10,16,16]
    --relax-p P          relaxation spline degree [2]
    --relax-steps N      relaxation step budget per map [500]
    --seeds N            Poincare seeds per map [16]
    --periods N          field periods per traced line [100]
    --precision {float32,float64}
"""
from __future__ import annotations

import argparse
import io
import os

QA_WOUT = "data/wout_LandremanPaul2021_QA_lowres.nc"
NCSX_WOUT = "data/wout_li383_low_res_reference.nc"

#: The two map builders compared throughout, and their colour everywhere.
SOURCES = (("equilibrium", "#1f77b4"), ("map2disc", "#d62728"))

#: Where the expensive solves are cached. Ignored by git.
CACHE_DIR = "outputs/map2disc_figures"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", default="all",
                    choices=("qa", "symmetry", "vacuum", "ncsx", "ncsx-relax", "all"))
    ap.add_argument("--geometry", default=QA_WOUT)
    ap.add_argument("--ncsx-geometry", default=NCSX_WOUT)
    ap.add_argument("--out", default="docs/research/map2disc_2026-09-13")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--relax-ns", default="10,16,16")
    ap.add_argument("--relax-p", type=int, default=2)
    ap.add_argument("--relax-steps", type=int, default=500)
    ap.add_argument("--relax-chunk", type=int, default=50)
    ap.add_argument("--relax-floor-tol", type=float, default=1e-6)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--periods", type=int, default=100)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--precision", default="float64", choices=("float32", "float64"))
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Shared helpers
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


def _cached(name: str, build):
    """Run ``build`` once and keep its arrays in :data:`CACHE_DIR`.

    Args:
        name: Cache file stem; the precision and resolution belong in it.
        build: Zero-argument callable returning a dict. Anything that is
            not an array is stored as a zero-dimensional object array, so
            read those back with ``.item()``.

    Returns:
        The dict, from the cache if it is there.
    """
    import numpy as np

    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}.npz")
    if os.path.exists(path):
        print(f"[cache] reusing {path}")
        blob = np.load(path, allow_pickle=True)
        return {k: blob[k] for k in blob.files}
    data = build()
    np.savez_compressed(path, **{
        k: v if isinstance(v, np.ndarray) else np.array(v, dtype=object)
        for k, v in data.items()})
    print(f"[cache] wrote {path}")
    return data


def _lcfs_fields(state):
    """The wout's own last closed flux surface, in closed form.

    Args:
        state: Parsed equilibrium from :func:`mrx.gvec.read_equilibrium`.

    Returns:
        ``(nfp, exact_RZ, exact_xyz)``: the field-period count, the LCFS in
        the ``(R, Z)`` plane at one ``zeta`` given an array of ``theta``,
        and the LCFS in Cartesian coordinates given a logical point.
    """
    import jax
    import jax.numpy as jnp

    from mrx.gvec import StateField

    nfp = int(state["nfp"])
    R_field, Z_field = StateField(state["X1"], nfp), StateField(state["X2"], nfp)

    def exact_RZ(theta, zeta):
        x = jnp.stack([jnp.ones_like(theta), theta, jnp.full_like(theta, zeta)], axis=-1)
        return jax.vmap(R_field)(x), jax.vmap(Z_field)(x)

    def exact_xyz(x):
        R, phi = R_field(x), 2.0 * jnp.pi * x[2] / nfp
        return jnp.array([R * jnp.cos(phi), -R * jnp.sin(phi), Z_field(x)])

    return nfp, exact_RZ, exact_xyz


def _cross_section(ax, fh, exact_RZ, zeta: float, legend: bool = False,
                   n_ring: int = 5, n_ray: int = 12) -> float:
    """Draw one map2disc cross-section over the VMEC boundary it came from.

    The grey mesh is the fit's own coordinate lines -- level sets of the
    harmonic map, NOT flux surfaces -- and the markers are ``f_h(1, theta)``
    against the wout's LCFS, which is the one place the two must agree.

    Args:
        ax: Axes to draw on.
        fh: The :class:`mrx.map2disc.ZernikeMap` of this plane.
        exact_RZ: From :func:`_lcfs_fields`.
        zeta: The toroidal plane, in periods.
        legend: Whether to label the two boundary curves.
        n_ring: Constant-``rho`` lines drawn.
        n_ray: Constant-``theta`` lines drawn.

    Returns:
        The largest distance between the fit's boundary and the wout's.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    for rho in np.linspace(0.2, 1.0, n_ring):
        t = jnp.linspace(0.0, 2.0 * jnp.pi, 200)
        xy = jax.vmap(lambda a, r=rho: fh(r, a))(t)
        ax.plot(xy[:, 0], xy[:, 1], lw=0.6, color="0.72", zorder=1)
    for angle in np.linspace(0.0, 2.0 * np.pi, n_ray, endpoint=False):
        r = jnp.linspace(0.0, 1.0, 80)
        xy = jax.vmap(lambda rr, a=angle: fh(rr, a))(r)
        ax.plot(xy[:, 0], xy[:, 1], lw=0.6, color="0.72", zorder=1)

    theta = jnp.linspace(0.0, 1.0, 400, endpoint=False)
    Rx, Zx = exact_RZ(theta, zeta)
    ax.plot(np.asarray(Rx), np.asarray(Zx), lw=2.6, color="#1f77b4",
            label="VMEC LCFS" if legend else None, zorder=2)
    got = jax.vmap(lambda a: fh(1.0, a))(2.0 * jnp.pi * theta)
    ax.plot(np.asarray(got[::8, 0]), np.asarray(got[::8, 1]), ls="none",
            marker="o", ms=3.2, mfc="none", color="#d62728",
            label="map2disc" if legend else None, zorder=3)
    ax.set_xlabel("$R$")
    ax.set_aspect("equal")
    if legend:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    return float(jnp.max(jnp.hypot(got[:, 0] - Rx, got[:, 1] - Zx)))


def _convergence_in_M(curve, degrees, rhos=(0.3, 0.6, 0.9), n_theta: int = 12):
    """Roundtrip error ``max |g(f_h(rho, theta)) - (rho, theta)|`` per degree.

    ``g`` is taken at a boundary resolution far beyond any the fit uses, so
    what this measures is the fit and not the reference.

    Args:
        curve: The boundary of one plane.
        degrees: Zernike degrees to try.
        rhos: Interior radii probed.
        n_theta: Angles per radius.

    Returns:
        One error per degree, ``nan`` where :func:`mrx.map2disc.fit_disc_map`
        refused the degree because its rings crowd the wall faster than
        ``N_BOUNDARY_MAX`` can resolve them.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from mrx.map2disc import fit_disc_map, harmonic_map

    g = harmonic_map(curve.resample(4096))
    rho, ang = np.meshgrid(np.asarray(rhos, dtype=float),
                           np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False),
                           indexing="ij")
    rho, ang = jnp.asarray(rho.ravel()), jnp.asarray(ang.ravel())
    want = jnp.stack([rho * jnp.cos(ang), rho * jnp.sin(ang)], axis=1)
    errs = []
    for M in degrees:
        try:
            got = jax.vmap(g)(jax.vmap(fit_disc_map(curve, M=M))(rho, ang))
            errs.append(float(jnp.max(jnp.abs(got - want))))
        except ValueError as exc:
            print(f"[conv] M = {M} refused: {str(exc)[:80]}")
            errs.append(float("nan"))
    return errs


def _field_magnitude(seq, B):
    """``|B|`` at a logical point, as a physical quantity.

    Args:
        seq: The sequence ``B`` lives on.
        B: Dirichlet 2-form DoFs.

    Returns:
        A callable taking a logical point.
    """
    import jax.numpy as jnp

    from mrx.differential_forms import DiscreteFunction, Pushforward

    pushed = Pushforward(DiscreteFunction(B, seq.basis_2, seq.E(2, True)), seq.map, 2)
    return lambda x: jnp.linalg.norm(pushed(x))


def _torus_image(seq, field, dpi: int, n: int = 40):
    """Render a scalar on the boundary and four poloidal cuts, as pixels.

    Rendered into a buffer rather than a file: a 3-D axes cannot be moved
    into another figure's grid, so the composite reads it back as an image.

    Args:
        seq: The sequence whose map defines the surfaces.
        field: Scalar function of a logical point.
        dpi: Render resolution.
        n: Grid points per axis on the poloidal cuts.

    Returns:
        The trimmed RGBA array.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from mrx.plotting import get_2d_grids, plot_torus

    grids_pol = [get_2d_grids(seq.map, cut_axis=2, cut_value=float(z),
                              nx=n, ny=n, nz=1) for z in np.arange(4) / 4]
    grid_surface = get_2d_grids(seq.map, cut_axis=0, cut_value=1.0 - 1e-6,
                                ny=3 * n, nz=3 * n, invert_z=True)
    fig, _ = plot_torus(field, grids_pol, grid_surface, cstride=8,
                        gridlinewidth=0.3, elev=25, azim=40, cbar_label=r"$|B|$")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return _trim(plt.imread(buffer))


def _poincare(seq, B, nfp: int, n_seeds: int, periods: int):
    """Trace field lines of ``B`` and reduce them to a profile and a section.

    Both outputs are physical -- ``iota`` against ``R`` on the midplane
    through the magnetic axis -- so they compare between two maps of the
    same domain, which logical ``r`` does not.

    Args:
        seq: The sequence ``B`` lives on.
        B: Dirichlet 2-form DoFs.
        nfp: Field periods.
        n_seeds: Seeds per ray.
        periods: Field periods traced per line.

    Returns:
        Dict of ``a`` (both midplane crossings of each regular line),
        ``iota``, the section's ``R`` and ``Z``, and ``xlabel``.
    """
    import jax.numpy as jnp
    import numpy as np

    from mrx.poincare import (logical_field, section_RZ, seed_from_axis,
                              surface_label, trace_and_classify)

    field = logical_field(seq, jnp.asarray(B), 2, True)
    seeds = seed_from_axis(field, n_seeds, 8, n_rays=4, steps_per_period=32)
    res = trace_and_classify(field, seeds, nfp, n_periods=periods,
                             steps_per_period=32, saves_per_period=8)
    R, Z, aR, aZ, _, _, _, _ = section_RZ(seq, res["ys"], res["axis"], 8, 0.0)
    a_eff, xlabel = surface_label(R, Z, aR, aZ)
    good = np.asarray(~(res["escaped"] | ~res["ok"] | res["chaotic"]))
    return {"a": np.asarray(a_eff)[good], "iota": np.asarray(res["iota"])[good],
            "R": np.asarray(R)[good], "Z": np.asarray(Z)[good], "xlabel": xlabel}


def _iota_scatter(ax, a_eff, iota, colour: str, label: str,
                  marker: str = "o", alpha: float = 0.8) -> None:
    """Scatter ``iota`` against the midplane crossings of each line.

    ``surface_label`` returns the inboard and outboard crossing of every
    line and ``nan`` where a line does not reach that side of the
    midplane. The mask is therefore per CROSSING, not per line: demanding
    both sides throws away lines that have a perfectly good one, which on
    the relaxed NCSX map2disc state is all of them.

    Args:
        ax: Axes to draw on.
        a_eff: ``(n_lines, 2)`` midplane crossings.
        iota: ``(n_lines,)`` rotational transforms.
        colour: Series colour.
        label: Legend label; the labelled-line count is appended.
        marker: Matplotlib marker.
        alpha: Marker opacity.
    """
    import numpy as np

    a_eff, iota = np.asarray(a_eff, dtype=float), np.asarray(iota, dtype=float)
    wide = np.broadcast_to(iota[:, None], a_eff.shape)
    keep = np.isfinite(a_eff) & np.isfinite(wide)
    ax.scatter(a_eff[keep], wide[keep], s=14.0, color=colour, alpha=alpha,
               marker=marker, label=f"{label}  ({int(keep.any(axis=1).sum())} lines)")


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


def _relaxation(cli, map_source: str):
    """Relax one geometry on one map and reduce the run to what is plotted.

    The initial condition is the geometry file's own field through the
    histopolated Clebsch potential, which reads the file's profiles at the
    LOGICAL radius (``mrx.initial_conditions``: ``r = clip(x[0], 0, 1)``).
    That treats ``r`` as a flux-surface label -- true for the equilibrium
    map, whose coordinate surfaces are the file's flux surfaces, and false
    for map2disc, whose surfaces are level sets of a harmonic map of the
    LCFS. So the two runs genuinely start from different fields with
    different helicity, and since the descent minimises energy at fixed
    helicity they need not land on the same equilibrium. Both are
    normalised to ``||B||_M = 1``, so their energies are on one scale; the
    helicities are reported so the reader can price the difference.

    Args:
        cli: Parsed command line.
        map_source: ``"equilibrium"`` or ``"map2disc"``.

    Returns:
        Dict of traces, torus renders before and after, Poincare data and
        a ``notes`` object array of the scalars quoted on the figure.
    """
    import jax.numpy as jnp
    import numpy as np

    from mrx.geometry import build_sequence
    from mrx.initial_conditions import initial_field
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import (TimeStepper, compute_helicity, initial_state, relax)

    ns = tuple(int(v) for v in cli.relax_ns.split(","))
    seq, _ = build_sequence(cli.ncsx_geometry, ns, cli.relax_p, map_source=map_source)
    compute_nullspaces(seq)
    nfp = int(seq.equilibrium["nfp"])

    B0, ic = initial_field(seq)
    H0 = float(compute_helicity(B0, seq, jnp.zeros(seq.n(1, True)))[0])
    print(f"[{map_source}] IC: H = {H0:+.6e}, iota {ic['iota_axis']:.4f} -> "
          f"{ic['iota_edge']:.4f}, ||div B|| = {ic['div']:.2e}", flush=True)

    ts = TimeStepper(seq=seq, cfl=0.5, history_size=1, velocity_smoothing_order=1)
    res = relax(initial_state(B0, ts), ts, steps=cli.relax_steps,
                chunk=cli.relax_chunk, floor_tol=cli.relax_floor_tol)
    B1 = res.state.B_n
    H1 = float(compute_helicity(B1, seq, jnp.zeros(seq.n(1, True)))[0])
    F = np.asarray(res.trace["F"], dtype=float)
    print(f"[{map_source}] {res.steps} steps ({res.stop}): ||F|| {F[0]:.3e} -> "
          f"{F[-1]:.3e}, H {H0:+.4e} -> {H1:+.4e}", flush=True)

    before = _poincare(seq, B0, nfp, cli.seeds, cli.periods)
    after = _poincare(seq, B1, nfp, cli.seeds, cli.periods)
    return {
        "F": F, "E": np.asarray(res.qoi["E"], dtype=float),
        "E_it": np.asarray(res.qoi["it"], dtype=float),
        "panel_before": _torus_image(seq, _field_magnitude(seq, B0), cli.dpi),
        "panel_after": _torus_image(seq, _field_magnitude(seq, B1), cli.dpi),
        "a_before": before["a"], "iota_before": before["iota"],
        "a_after": after["a"], "iota_after": after["iota"],
        "R_after": after["R"], "Z_after": after["Z"],
        "notes": np.array({"H0": H0, "H1": H1, "F0": float(F[0]),
                           "F1": float(F[-1]), "steps": int(res.steps),
                           "stop": str(res.stop), "wall": float(res.wall),
                           "iota_axis": float(ic["iota_axis"]),
                           "iota_edge": float(ic["iota_edge"]),
                           "xlabel": after["xlabel"]}, dtype=object)}


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

    from mrx.gvec import read_equilibrium
    from mrx.map2disc import fit_disc_map, lcfs_boundary, map2disc_map, nyquist_n_zeta

    st = read_equilibrium(cli.geometry)
    nfp, exact_RZ, exact_xyz = _lcfs_fields(st)
    boundary_of_zeta = lcfs_boundary(st)

    fig = plt.figure(figsize=(11.0, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=(1.35, 1.0), hspace=0.34, wspace=0.28)

    for col, zeta in enumerate((0.0, 0.25, 0.5)):
        ax = fig.add_subplot(gs[0, col])
        err = _cross_section(ax, fit_disc_map(boundary_of_zeta(zeta), M=8),
                             exact_RZ, zeta, legend=col == 0)
        ax.set_title(rf"$\zeta = {zeta:g}$ of a period    max $|\Delta| = ${err:.1e}",
                     fontsize=9)
        ax.set_ylabel("$Z$" if col == 0 else "")

    # Spectral convergence of the fit in M, on one plane.
    degrees = (4, 6, 8, 10, 12)
    errs = _convergence_in_M(boundary_of_zeta(0.25), degrees)

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
    ax.text(0.0, 1.0, "roundtrip error\n\n"
            + "\n".join(rf"$M = {M}$:  {e:.1e}" for M, e in zip(degrees, errs))
            + "\n\nboundary error\n\n"
            + "\n".join(rf"$n_\zeta = {c}$:  {e:.1e}" for c, e in zip(counts, cliff)),
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

    from mrx.geometry import geometry_nfp

    ns = tuple(int(v) for v in cli.ns.split(","))
    nfp = geometry_nfp(cli.geometry)

    def build():
        """Two Hodge solves and two field-line traces."""
        data = {}
        for source, _ in SOURCES:
            seq, B, diag = _vacuum_field(cli.geometry, ns, cli.p, source)
            print(f"[{source}] ||div B|| = {diag['div']:.2e}, "
                  f"||curl B|| / ||B|| = {diag['curl']:.2e}, "
                  f"Rayleigh = {diag['rayleigh']:.2e}")
            B_mag = _field_magnitude(seq, B)

            # |B| on the torus. Both maps share the LCFS exactly, so the
            # surface these are drawn on is the same surface.
            data[f"panel_{source}"] = _torus_image(seq, B_mag, cli.dpi)

            # |B| along the boundary at zeta = 0. r = 1 is the one place
            # the two maps agree pointwise, so this compares the same
            # physical points and needs no inversion.
            t = jnp.linspace(0.0, 1.0, 240, endpoint=False)
            pts = jnp.stack([jnp.full(240, 1.0 - 1e-6), t, jnp.zeros(240)], axis=-1)
            data[f"boundary_{source}"] = np.asarray(jax.vmap(B_mag)(pts))

            trace = _poincare(seq, B, nfp, cli.seeds, cli.periods)
            data[f"a_{source}"] = trace["a"]
            data[f"iota_{source}"] = trace["iota"]
            data[f"notes_{source}"] = np.array(dict(diag, xlabel=trace["xlabel"]),
                                               dtype=object)
        return data

    data = _cached(f"vacuum_{'x'.join(map(str, ns))}_p{cli.p}_{cli.precision}", build)
    notes = {s: data[f"notes_{s}"].item() for s, _ in SOURCES}

    fig = plt.figure(figsize=(11.5, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.45, 1.0), hspace=0.16, wspace=0.22)
    for col, (source, colour) in enumerate(SOURCES):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(data[f"panel_{source}"])
        ax.axis("off")
        ax.set_title(f'map_source="{source}"', fontsize=10, color=colour)

    ax = fig.add_subplot(gs[1, 0])
    t = np.linspace(0.0, 1.0, 240, endpoint=False)
    for source, colour in SOURCES:
        ax.plot(t, data[f"boundary_{source}"], color=colour, lw=1.8, label=source)
    gap = np.max(np.abs(data["boundary_equilibrium"] - data["boundary_map2disc"]))
    rel = gap / np.mean(data["boundary_equilibrium"])
    ax.set_xlabel(r"poloidal angle $\theta / 2\pi$")
    ax.set_ylabel(r"$|B|$ on the boundary, $\zeta = 0$")
    ax.set_title(rf"same physical surface: max difference {rel:.1%} of $\langle|B|\rangle$",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    outboard = {}
    for source, colour in SOURCES:
        a_eff, iota = data[f"a_{source}"], data[f"iota_{source}"]
        _iota_scatter(ax, a_eff, iota, colour, source)
        keep = np.isfinite(a_eff).any(axis=1) & np.isfinite(iota)
        edge = np.nanmax(a_eff[keep], axis=1)
        order = np.argsort(edge)
        outboard[source] = (edge[order], iota[keep][order])

    # Compare the two profiles where they overlap: iota against a physical
    # abscissa is the same function of the domain, whatever the coordinates.
    ref_R, ref_iota = outboard["equilibrium"]
    got_R, got_iota = outboard["map2disc"]
    lo, hi = max(ref_R.min(), got_R.min()), min(ref_R.max(), got_R.max())
    inside = (ref_R >= lo) & (ref_R <= hi)
    d_iota = np.max(np.abs(np.interp(ref_R[inside], got_R, got_iota)
                           - ref_iota[inside]))
    ax.set_xlabel(notes["equilibrium"]["xlabel"])
    ax.set_ylabel(r"$\iota$")
    ax.set_title(rf"coordinate-independent physics: $\max|\Delta\iota| = ${d_iota:.1e}"
                 rf" on $\iota \approx {ref_iota.mean():.3f}$", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("the QA vacuum field is a property of the domain, so both maps "
                 f"must find it\nns={ns}, p={cli.p}, {cli.precision};  "
                 rf"$\|\mathrm{{curl}}\,B\| / \|B\| = ${notes['equilibrium']['curl']:.1e} "
                 f"(equilibrium) vs {notes['map2disc']['curl']:.1e} (map2disc)",
                 fontsize=11)
    path = os.path.join(out, "qa_vacuum_both_maps.png")
    fig.savefig(path, dpi=cli.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 4: the NCSX mesh
# ---------------------------------------------------------------------------

def figure_ncsx(cli, out: str) -> str:
    """The NCSX cross-sections, the crescent that broke the seed, and ``M``.

    Args:
        cli: Parsed command line.
        out: Output directory.

    Returns:
        Path of the written PNG.
    """
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    from mrx.gvec import read_equilibrium
    from mrx.map2disc import fit_disc_map, harmonic_map, lcfs_boundary, nyquist_n_zeta

    st = read_equilibrium(cli.ncsx_geometry)
    _, exact_RZ, _ = _lcfs_fields(st)
    boundary_of_zeta = lcfs_boundary(st)
    M = 6

    fig = plt.figure(figsize=(11.5, 6.8))
    gs = fig.add_gridspec(2, 3, height_ratios=(1.35, 1.0), hspace=0.36, wspace=0.36)

    planes = (0.0, 0.25, 0.5)
    curves = {z: boundary_of_zeta(z) for z in planes}
    axes = {}
    for col, zeta in enumerate(planes):
        axes[zeta] = ax = fig.add_subplot(gs[0, col])
        err = _cross_section(ax, fit_disc_map(curves[zeta], M=M), exact_RZ, zeta,
                             legend=col == 0)
        ax.set_title(rf"$\zeta = {zeta:g}$ of a period    max $|\Delta| = ${err:.1e}",
                     fontsize=9)
        ax.set_ylabel("$Z$" if col == 0 else "")

    # The crescent, with the centroid that used to anchor every Newton.
    crescent = curves[0.0]
    g = harmonic_map(crescent)
    centroid = jnp.mean(crescent.samples, axis=0)
    g_centroid = float(jnp.linalg.norm(g(centroid)))
    ax = axes[0.0]
    ax.plot(float(centroid[0]), float(centroid[1]), marker="X", ms=9.0,
            color="#2ca02c", mec="k", mew=0.6, zorder=4)
    ax.annotate(rf"centroid, $|g| = {g_centroid:.1f}$", (float(centroid[0]),
                float(centroid[1])), xytext=(6, -12), textcoords="offset points",
                fontsize=7.5, color="#2ca02c")

    # Convergence in M on the crescent and on a mild plane of the same file.
    degrees = (3, 4, 6, 8, 10, 12)
    ax = fig.add_subplot(gs[1, 0])
    summary = []
    for zeta, colour, label in ((0.0, "#d62728", r"$\zeta = 0$ (crescent)"),
                                (0.25, "#1f77b4", r"$\zeta = 0.25$ (mild)")):
        errs = _convergence_in_M(curves[zeta], degrees)
        ax.semilogy(degrees, errs, "o-", color=colour, label=label)
        refused = [d for d, e in zip(degrees, errs) if not np.isfinite(e)]
        if refused:
            ax.axvspan(min(refused) - 0.4, max(degrees) + 0.4, color="0.9", zorder=0)
        summary.append((label, errs, refused))
    ax.set_xlabel("Zernike degree $M$")
    ax.set_ylabel("roundtrip error")
    ax.set_title("the crescent lags the mild plane by ~4 degrees", fontsize=9)
    ax.set_xticks(degrees)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, which="both")

    # Where the error lives: flat in the boundary resolution, peaked well
    # inside, so it is the Zernike basis and not close evaluation.
    ax = fig.add_subplot(gs[1, 1])
    fh = fit_disc_map(crescent, M=M)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 24, endpoint=False)
    for n_ref, colour, ls in ((2048, "#d62728", "-"), (4096, "0.3", "--")):
        g_ref = harmonic_map(crescent.resample(n_ref))
        rhos = np.linspace(0.1, 0.95, 12)
        errs = []
        for rho in rhos:
            pts = jnp.stack([jnp.full(24, float(rho)), theta], axis=1)
            got = jnp.stack([g_ref(fh(float(rho), float(t))) for t in theta])
            want = jnp.stack([pts[:, 0] * jnp.cos(theta),
                              pts[:, 0] * jnp.sin(theta)], axis=1)
            errs.append(float(jnp.max(jnp.abs(got - want))))
        ax.semilogy(rhos, errs, ls, color=colour, lw=1.8,
                    label=rf"$g$ at $n = {n_ref}$")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("roundtrip error")
    ax.set_title(rf"$M = {M}$: flat in boundary resolution", fontsize=9)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, which="both")

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    lines = [f"nfp = {int(st['nfp'])},  Nyquist $n_\\zeta$ = {nyquist_n_zeta(st)}",
             rf"$|g(\mathrm{{centroid}})|$ = {g_centroid:.2f}  at $\zeta = 0$", ""]
    for label, errs, refused in summary:
        lines.append(label)
        lines += [f"  M = {d:<3}{'refused' if not np.isfinite(e) else f'{e:.1e}'}"
                  for d, e in zip(degrees, errs)]
        lines.append("")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8,
            family="monospace")

    fig.suptitle(f"map2disc on the NCSX (li383) boundary ({cli.precision})", fontsize=11)
    path = os.path.join(out, "ncsx_vs_vmec.png")
    fig.savefig(path, dpi=cli.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 5: NCSX relaxed on both maps
# ---------------------------------------------------------------------------

def figure_ncsx_relax(cli, out: str) -> str:
    """Relax NCSX on both maps and compare before against after.

    Args:
        cli: Parsed command line.
        out: Output directory.

    Returns:
        Path of the written PNG.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    ns = cli.relax_ns.replace(",", "x")
    runs = {s: _cached(f"relax_ncsx_{s}_{ns}_p{cli.relax_p}_"
                       f"{cli.relax_steps}_{cli.precision}",
                       lambda s=s: _relaxation(cli, s)) for s, _ in SOURCES}
    notes = {s: runs[s]["notes"].item() for s, _ in SOURCES}

    fig = plt.figure(figsize=(13.0, 10.0))
    gs = fig.add_gridspec(3, 4, height_ratios=(1.0, 1.2, 1.05),
                          hspace=0.30, wspace=0.30)

    ax = fig.add_subplot(gs[0, :2])
    for source, colour in SOURCES:
        E = runs[source]["E"]
        ax.plot(runs[source]["E_it"], E[0] - E, color=colour, lw=1.8,
                label=f"{source}  (total {E[0] - E[-1]:.2e})")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$E_0 - E$")
    ax.set_title(r"energy removed ($\|B\|_M = 1$, so both start at $E = 0.5$)",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = fig.add_subplot(gs[0, 2:])
    for source, colour in SOURCES:
        F = runs[source]["F"]
        ax.semilogy(np.arange(F.size), F, color=colour, lw=1.2,
                    label=f"{source}  ({notes[source]['F0']:.2e} "
                          f"-> {notes[source]['F1']:.2e})")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\|F\|$")
    ax.set_title("force residual", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    for col, (source, colour) in enumerate(SOURCES):
        for half, when in enumerate(("before", "after")):
            ax = fig.add_subplot(gs[1, 2 * col + half])
            ax.imshow(runs[source][f"panel_{when}"])
            ax.axis("off")
            ax.set_title(f"{source}, {when}", fontsize=9, color=colour)

    ax = fig.add_subplot(gs[2, :2])
    for source, colour in SOURCES:
        _iota_scatter(ax, runs[source]["a_before"], runs[source]["iota_before"],
                      colour, f"{source}, before", marker="o", alpha=0.25)
        _iota_scatter(ax, runs[source]["a_after"], runs[source]["iota_after"],
                      colour, f"{source}, after", marker="x", alpha=0.9)
    ax.set_xlabel(notes["equilibrium"]["xlabel"])
    ax.set_ylabel(r"$\iota$")
    ax.set_title(r"$\iota$ against a physical abscissa (faint = before, $\times$ = after)",
                 fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # One colour per traced line, so nested surfaces read as nested rather
    # than as one smear of crossings.
    for col, (source, colour) in enumerate(SOURCES):
        ax = fig.add_subplot(gs[2, 2 + col])
        R, Z = runs[source]["R_after"], runs[source]["Z_after"]
        shades = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, R.shape[0]))
        for line in range(R.shape[0]):
            ax.plot(R[line], Z[line], ls="none", marker=".", ms=0.8,
                    color=shades[line], alpha=0.75)
        ax.set_xlabel("$R$")
        ax.set_ylabel("$Z$" if col == 0 else "")
        ax.set_aspect("equal")
        ax.set_title(f"{source}, converged  ({R.shape[0]} lines)", fontsize=9,
                     color=colour)

    # The caveat, stated rather than papered over.
    eq, m2d = notes["equilibrium"], notes["map2disc"]
    fig.suptitle(
        "NCSX relaxed on both maps, before and after  "
        f"(ns={cli.relax_ns}, p={cli.relax_p}, {cli.precision})\n"
        "The Clebsch initial condition reads the file's profiles at the LOGICAL "
        "radius, which labels flux surfaces for the equilibrium map and does NOT "
        "for map2disc, so the two\nruns start from different fields:  "
        rf"$H_0$ = {eq['H0']:+.4e} vs {m2d['H0']:+.4e},  "
        rf"$\|F\|_0$ = {eq['F0']:.2e} vs {m2d['F0']:.2e}.  "
        "The descent minimises energy at fixed helicity, so they need not agree.",
        fontsize=9.5)
    path = os.path.join(out, "ncsx_relaxation.png")
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
    builders = {"qa": figure_qa, "symmetry": figure_symmetry, "vacuum": figure_vacuum,
                "ncsx": figure_ncsx, "ncsx-relax": figure_ncsx_relax}
    # "all" leaves out ncsx-relax: it is two nonlinear descents, about twenty
    # minutes, and it is the one figure made in float32. Ask for it by name.
    wanted = ("qa", "symmetry", "vacuum", "ncsx") if cli.figure == "all" else (cli.figure,)
    for name in wanted:
        print(f"  -> {builders[name](cli, cli.out)}")


if __name__ == "__main__":
    main(parse_args())
