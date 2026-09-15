"""Figures and comparisons for the DESC interface writeup.

Owns its own compute. Every expensive result is cached under
``outputs/desc_figures/<case>/`` so ``--figure all`` is cheap the second
time, and the cache name carries ``--ns``, ``--p`` and ``--steps`` so a
longer budget cannot silently reload a shorter one::

    python -u scripts/desc_figures.py --figure traces
    python -u scripts/desc_figures.py --figure all --ns 8,12,12 --p 2 --steps 5000
    python -u scripts/desc_figures.py --figure grid --plot-only

``--precision`` is exported as ``MRX_DTYPE`` before ``mrx`` is imported.
Figures land in ``--out`` and are copied into ``docs/research/desc_interface/``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

#: Target number of plotted points in a per-step trace; the block width is
#: derived from the run length so a short run is not reduced to a polyline.
TRACE_POINTS = 200

#: Destination of the published PNGs, next to the writeup they illustrate.
RESEARCH_DIR = os.path.join("docs", "research", "desc_interface")

#: Radial, poloidal and toroidal samples of the comparison grid. Midpoints
#: in the angles and interior points in rho, so nothing lands on a spline
#: node, where an interpolatory refit is exact by construction.
GRID = (40, 64, 32)

#: Radius below which the lambda comparison is reported separately. DESC's
#: wout importer fits VMEC's HALF-mesh ``lmns`` as if it sat on the full
#: mesh and anchors it at the dummy axis row, so the two codes' lambda
#: disagree near the axis by a fixed amount that no fit resolution removes.
LAMBDA_CORE = 0.3

#: Cases of ``--figure cases``: name, an explicit path for the ones that
#: are not shipped DESC examples, and whether to publish the poloidal mesh.
CASE_FIGURES = (("HELIOTRON", None, True), ("W7-X", None, True),
                ("ATF", None, False))

FIGURES = ("grid", "gvec", "relax", "traces", "li383", "qa", "cases")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: argument list; ``None`` reads ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", default="all",
                    choices=(*FIGURES, "all"),
                    help="which figure to (re)build")
    ap.add_argument("--wout", default="data/wout_li383_low_res_reference.nc")
    ap.add_argument("--desc", default="data/desc_li383_lowres.h5")
    ap.add_argument("--qa-wout", default="data/wout_LandremanPaul2021_QA_lowres.nc")
    ap.add_argument("--qa-desc", default="data/desc_QA_lowres.h5")
    ap.add_argument("--examples", default=None,
                    help="DESC examples directory; default discovers the install")
    ap.add_argument("--ns", default="8,12,12")
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--history", type=int, default=1)
    ap.add_argument("--cfl", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, default=16,
                    help="field lines per ray of the Poincaré seed set")
    ap.add_argument("--rays", type=int, default=4)
    ap.add_argument("--periods", type=int, default=150)
    ap.add_argument("--trace-steps", type=int, default=24,
                    help="integration steps per field period; must divide --saves")
    ap.add_argument("--saves", type=int, default=8)
    ap.add_argument("--plane", type=float, default=0.0,
                    help="logical ζ of the section, in [0, 1)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="lines per JAX batch; None is a full vmap")
    ap.add_argument("--sweep", default="4,6,8,10",
                    help="comma-separated DESC fit resolutions; L = M = this, "
                         "N = min(this, the wout's ntor)")
    ap.add_argument("--grid", default=",".join(str(n) for n in GRID),
                    help="common evaluation grid n_rho,n_theta,n_zeta")
    ap.add_argument("--precision", default="float64", choices=("float32", "float64"))
    ap.add_argument("--out", default="outputs/desc_figures")
    ap.add_argument("--force", action="store_true",
                    help="ignore cached fields.npz / section.npz")
    ap.add_argument("--plot-only", action="store_true",
                    help="rebuild PNGs from cached result.json files")
    ap.add_argument("--no-mesh", action="store_true",
                    help="skip the plot_mesh.py subprocess of --figure cases")
    return ap.parse_args(argv)


def _log(msg: str) -> None:
    """Print a timestamped progress line.

    Args:
        msg: the message.
    """
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _run_tag(cli: argparse.Namespace) -> str:
    """Cache stem encoding the run parameters.

    Args:
        cli: parsed arguments.

    Returns:
        ``{nr}x{nt}x{nz}_p{p}_s{steps}``.
    """
    ns = cli.ns.replace(",", "x")
    return f"{ns}_p{cli.p}_s{cli.steps}"


def blocked(y: np.ndarray, log: bool = True, w: int | None = None
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Block means of a per-step trace and the ±1 sd band.

    ``log`` takes the statistics in log space for a log axis. The last
    partial block is kept. The default width yields about
    :data:`TRACE_POINTS` points regardless of the run length.

    Args:
        y: the per-step samples.
        log: whether to average in log space.
        w: block width in steps; ``None`` derives it from ``len(y)``.

    Returns:
        ``(centre_step, mean, lower, upper)``.
    """
    y = np.asarray(y, float)
    n = len(y)
    w = max(1, n // TRACE_POINTS) if w is None else w
    edges = list(range(0, n, w)) + [n]
    x, m, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        blk = y[a:b]
        if log:
            blk = np.log(blk[blk > 0])
            if len(blk) == 0:
                continue
        mu, sd = blk.mean(), blk.std()
        x.append(0.5 * (a + b + 1))
        m.append(mu)
        lo.append(mu - sd)
        hi.append(mu + sd)
    f = np.exp if log else np.asarray
    return (np.asarray(x), f(np.asarray(m)),
            f(np.asarray(lo)), f(np.asarray(hi)))


def plot_trace(ax: Any, y: np.ndarray, log: bool = True, raw: bool = True,
               **kw: Any) -> Any:
    """One block-averaged trace with its sd ribbon and a raw underlay.

    Args:
        ax: the matplotlib axes.
        y: the per-step samples.
        log: whether to average in log space.
        raw: whether to draw the per-step series at low alpha.
        **kw: forwarded to the block-mean line.

    Returns:
        The block-mean line artist.
    """
    y = np.asarray(y, float)
    x, m, lo, hi = blocked(y, log)
    (line,) = ax.plot(x, m, zorder=3, **kw)
    if raw:
        ax.plot(np.arange(1, len(y) + 1), y, color=line.get_color(),
                lw=0.4, alpha=0.22, zorder=1)
    ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.2, lw=0, zorder=2)
    return line


def publish(src: str, dest_name: str | None = None) -> str:
    """Copy ``src`` into the research-note figure directory.

    Args:
        src: path of the PNG just written.
        dest_name: optional basename under :data:`RESEARCH_DIR`; defaults
            to ``os.path.basename(src)``.

    Returns:
        The published path.
    """
    os.makedirs(RESEARCH_DIR, exist_ok=True)
    dest = os.path.join(RESEARCH_DIR, dest_name or os.path.basename(src))
    shutil.copy2(src, dest)
    _log(f"published {dest}")
    return dest


def examples_dir(given: str | None) -> str:
    """Where the DESC example files live.

    Args:
        given: an explicit directory, or ``None`` to discover one.

    Returns:
        ``given`` if set, else the installed DESC package's ``examples/``
        if importable, else ``data/``.
    """
    if given:
        return given
    try:
        import desc  # noqa: PLC0415  (optional dependency)
        return os.path.join(os.path.dirname(desc.__file__), "examples")
    except ImportError:
        return "data"


def locate(root: str, name: str) -> str | None:
    """The file of one shipped DESC example.

    Args:
        root: the directory to look in.
        name: the case name, e.g. ``"W7-X"``.

    Returns:
        The path, or ``None`` if the case is not there.
    """
    for pattern in (f"{name}_output.h5", f"desc_{name}.h5",
                    f"desc_{name}_lowres.h5", f"{name}.h5"):
        hit = os.path.join(root, pattern)
        if os.path.isfile(hit):
            return hit
    return None


def _cache_path(root: str, *parts: str) -> str:
    """``root/parts...`` as a path, with the parent created.

    Args:
        root: the figure output directory.
        *parts: path components under ``root``.

    Returns:
        The joined path.
    """
    path = os.path.join(root, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def relax_one(seq: Any, eq: dict[str, Any], ts: Any, cli: argparse.Namespace,
              cache: str, tag: str) -> dict[str, Any]:
    """Build one initial field, relax it, and cache the DoFs plus the trace.

    Omitting :func:`mrx.nullspace.compute_nullspaces` on ``seq`` before
    this is called makes the descent diverge to NaN.

    Args:
        seq: the shared sequence; ``seq.equilibrium`` is swapped to ``eq``.
        eq: the parsed equilibrium to start from.
        ts: the shared time stepper.
        cli: parsed arguments, for ``steps`` / ``chunk`` / ``force``.
        cache: ``fields.npz`` path.
        tag: a short label for the log lines.

    Returns:
        ``B0``, ``B_final``, per-step ``F`` and ``dE``, ``E0``, ``qoi``,
        ``steps``, ``stop``, ``wall``, and the initial-field ``info``.
    """
    from mrx.initial_conditions import initial_field
    from mrx.relaxation import initial_state, relax

    if os.path.isfile(cache) and not cli.force:
        data = np.load(cache, allow_pickle=True)
        rec = {k: data[k] for k in data.files}
        rec["info"] = rec["info"].item() if rec["info"].shape == () else rec["info"]
        rec["qoi"] = rec["qoi"].item() if rec["qoi"].shape == () else rec["qoi"]
        _log(f"{tag}: loaded {cache}")
        return rec

    seq.equilibrium = eq
    t0 = time.perf_counter()
    B0, info = initial_field(seq)
    _log(f"{tag}: IC in {time.perf_counter() - t0:.0f} s, "
         f"div {info['div']:.2e}, iota {info['iota_axis']:+.4f}.."
         f"{info['iota_edge']:+.4f}")
    res = relax(initial_state(B0, ts), ts, cli.steps, chunk=cli.chunk, verbose=False)
    qoi = {k: np.asarray(v) for k, v in res.qoi.items()}
    rec = dict(B0=np.asarray(B0), B_final=np.asarray(res.state.B_n),
               F=np.asarray(res.trace["F"]), dE=np.asarray(res.trace["dE"]),
               E0=np.asarray(res.E0), info=np.asarray(info, dtype=object),
               qoi=np.asarray(qoi, dtype=object),
               steps=np.asarray(res.steps), stop=np.asarray(res.stop),
               wall=np.asarray(res.wall))
    np.savez_compressed(cache, **rec)
    rec["info"] = info
    rec["qoi"] = qoi
    rec["steps"] = int(res.steps)
    rec["stop"] = res.stop
    rec["wall"] = float(res.wall)
    _log(f"{tag}: {res.steps} steps in {res.wall:.0f} s, stop={res.stop}, "
         f"wrote {cache}")
    return rec


def pair_from_wout(wout: str, desc: str, cli: argparse.Namespace,
                   name: str) -> tuple[Any, dict[str, Any], dict[str, Any], int]:
    """One sequence, two initial conditions, from a wout and its DESC refit.

    The DESC state is turned to the wout's poloidal orientation first:
    without that the DESC field is a mirrored equilibrium.

    Args:
        wout: the VMEC file the sequence is built from.
        desc: the DESC reading of the same equilibrium.
        cli: parsed arguments.
        name: cache subdirectory (``"li383"`` or ``"qa"``).

    Returns:
        ``(seq, rec_vmec, rec_desc, orientation)``.
    """
    from mrx.geometry import build_sequence
    from mrx.gvec import match_orientation, read_equilibrium
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import TimeStepper
    from mrx.vmec import read_wout

    ns = tuple(int(v) for v in cli.ns.split(","))
    t0 = time.perf_counter()
    seq, _ = build_sequence(wout, ns, cli.p)
    compute_nullspaces(seq)
    _log(f"{name}: sequence {ns} p={cli.p} in {time.perf_counter() - t0:.0f} s")
    ts = TimeStepper(seq=seq, cfl=cli.cfl, history_size=cli.history,
                     velocity_smoothing_order=1)

    eq_v = read_equilibrium(wout)
    eq_d, sign = match_orientation(read_equilibrium(desc), read_wout(wout))
    _log(f"{name}: DESC orientation relative to the wout: {sign:+d}")
    tag = _run_tag(cli)
    rec_v = relax_one(seq, eq_v, ts, cli,
                      _cache_path(cli.out, name, "vmec", f"fields_{tag}.npz"),
                      f"{name}/vmec")
    rec_d = relax_one(seq, eq_d, ts, cli,
                      _cache_path(cli.out, name, "desc", f"fields_{tag}.npz"),
                      f"{name}/desc")
    return seq, rec_v, rec_d, sign


def single_from_desc(path: str, cli: argparse.Namespace,
                     name: str) -> tuple[Any, dict[str, Any]]:
    """Relax MRX from a DESC-native file.

    Args:
        path: the DESC ``.h5``.
        cli: parsed arguments.
        name: cache subdirectory.

    Returns:
        ``(seq, rec)``.
    """
    from mrx.geometry import build_sequence
    from mrx.gvec import read_equilibrium
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import TimeStepper

    ns = tuple(int(v) for v in cli.ns.split(","))
    t0 = time.perf_counter()
    seq, _ = build_sequence(path, ns, cli.p)
    compute_nullspaces(seq)
    _log(f"{name}: sequence {ns} p={cli.p} in {time.perf_counter() - t0:.0f} s")
    ts = TimeStepper(seq=seq, cfl=cli.cfl, history_size=cli.history,
                     velocity_smoothing_order=1)
    rec = relax_one(seq, read_equilibrium(path), ts, cli,
                    _cache_path(cli.out, name, f"fields_{_run_tag(cli)}.npz"), name)
    return seq, rec


def section_of(seq: Any, B: np.ndarray, nfp: int, cli: argparse.Namespace,
               cache: str, label: str) -> dict[str, Any]:
    """Trace one field and return the Poincaré section at ``cli.plane``.

    Seeds from the magnetic axis, never ``r = 0``: a finite-beta
    equilibrium has a Shafranov shift of centimetres. Lets
    :func:`mrx.poincare.require_zeta_parameterisation` refuse rather than
    clamp. ``steps_per_period`` must be divisible by ``saves_per_period``.

    Args:
        seq: the sequence the field lives on.
        B: Dirichlet 2-form DoFs.
        nfp: field periods; must match the geometry.
        cli: parsed arguments (seeds, periods, plane, …).
        cache: ``section.npz`` path.
        label: a short name for the ζ-parameterisation gate.

    Returns:
        ``R``, ``Z``, ``iota``, ``keep``, ``axis`` of the section.
    """
    import jax.numpy as jnp
    from mrx.poincare import (logical_field, require_zeta_parameterisation,
                              section_RZ, seed_from_axis, trace_and_classify)

    if os.path.isfile(cache) and not cli.force:
        data = np.load(cache)
        _log(f"{label}: loaded {cache}")
        return {k: data[k] for k in data.files}

    if cli.trace_steps % cli.saves:
        raise ValueError(f"--trace-steps {cli.trace_steps} must be a "
                         f"multiple of --saves {cli.saves}")
    field = logical_field(seq, jnp.asarray(B), 2, True)
    require_zeta_parameterisation(field, name=label)
    seeds = seed_from_axis(field, cli.seeds, cli.saves, n_rays=cli.rays,
                           steps_per_period=cli.trace_steps)
    t0 = time.perf_counter()
    res = trace_and_classify(
        field, seeds, nfp, n_periods=cli.periods,
        steps_per_period=cli.trace_steps, saves_per_period=cli.saves,
        batch_size=cli.batch_size)
    R, Z, aR, aZ, _, _, _, _ = section_RZ(
        seq, res["ys"], res["axis"], cli.saves, cli.plane)
    keep = ~(res["escaped"] | ~res["ok"])
    out = dict(R=np.asarray(R), Z=np.asarray(Z), iota=np.asarray(res["iota"]),
               keep=np.asarray(keep), axis_R=np.asarray(aR), axis_Z=np.asarray(aZ))
    np.savez_compressed(cache, **out)
    _log(f"{label}: {int(keep.sum())}/{keep.size} lines kept, "
         f"{time.perf_counter() - t0:.0f} s, wrote {cache}")
    return out


def _iota_limits(*sections: dict[str, Any]) -> tuple[float, float]:
    """Shared iota colour scale: the union of the kept lines.

    Args:
        *sections: section dicts from :func:`section_of`.

    Returns:
        ``(lo, hi)``.
    """
    vals = [s["iota"][s["keep"]] for s in sections if np.any(s["keep"])]
    if not vals:
        return (0.0, 1.0)
    stacked = np.concatenate([np.asarray(v)[np.isfinite(v)] for v in vals])
    lo, hi = float(stacked.min()), float(stacked.max())
    if hi - lo < 1e-9:
        lo, hi = lo - 5e-3, hi + 5e-3
    return lo, hi


def _rz_limits(*sections: dict[str, Any]) -> tuple[tuple[float, float],
                                                   tuple[float, float]]:
    """Shared ``(R, Z)`` window of the kept crossings.

    Args:
        *sections: section dicts from :func:`section_of`.

    Returns:
        ``((R0, R1), (Z0, Z1))``.
    """
    rs, zs = [], []
    for s in sections:
        keep = s["keep"]
        if not np.any(keep):
            continue
        rs.append(np.asarray(s["R"])[keep].ravel())
        zs.append(np.asarray(s["Z"])[keep].ravel())
    R = np.concatenate(rs) if rs else np.array([0.0, 1.0])
    Z = np.concatenate(zs) if zs else np.array([0.0, 1.0])
    pad_r, pad_z = 0.06 * (R.max() - R.min() or 1.0), 0.06 * (Z.max() - Z.min() or 1.0)
    return ((float(R.min() - pad_r), float(R.max() + pad_r)),
            (float(Z.min() - pad_z), float(Z.max() + pad_z)))


def poincare_grid(panels: list[tuple[str, dict[str, Any]]], dest: str,
                  nfp: int, title: str) -> str:
    """Draw a row or 2×2 of Poincaré sections with one shared iota scale.

    ``render_section`` takes no multi-panel layout and produces a 16-inch
    page per field, so the comparison figure is the physical section only:
    same hue is the same transform in every panel.

    Args:
        panels: ``(title, section)`` pairs, row-major.
        dest: output PNG path.
        nfp: field periods, for the colour-bar Farey ticks.
        title: the figure title.

    Returns:
        ``dest``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mrx.plotstyle import SECTION_CMAP, house_style
    from mrx.plotting import resonant_rationals, save_figure

    n = len(panels)
    rows, cols = (2, 2) if n == 4 else (1, n)
    lo, hi = _iota_limits(*(s for _, s in panels))
    (r0, r1), (z0, z1) = _rz_limits(*(s for _, s in panels))
    ticks, labels = resonant_rationals(lo, hi, nfp, 12, 0.08)
    if len(ticks) > 6:
        ticks, labels = [], []

    with house_style():
        fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows),
                                 squeeze=False, layout="none")
        sc = None
        for ax, (lab, s) in zip(axes.ravel(), panels):
            keep = s["keep"]
            R, Z, iota = np.asarray(s["R"]), np.asarray(s["Z"]), np.asarray(s["iota"])
            colour = np.broadcast_to(iota[:, None], R.shape)
            npts = max(int(keep.sum()) * R.shape[1], 1)
            size = float(np.clip(3000.0 / npts, 0.35, 15.0))
            sc = ax.scatter(R[keep], Z[keep], c=colour[keep], s=size,
                            vmin=lo, vmax=hi, cmap=SECTION_CMAP, linewidths=0,
                            rasterized=True)
            if np.any(~keep):
                ax.scatter(R[~keep], Z[~keep], c="0.55", s=size, linewidths=0,
                           rasterized=True)
            aR, aZ = np.asarray(s["axis_R"]), np.asarray(s["axis_Z"])
            ax.plot(np.mean(aR), np.mean(aZ), "k+", ms=6, mew=1.1, zorder=5)
            ax.set_title(lab, fontsize=10)
            ax.set_aspect("equal")
            ax.set_xlim(r0, r1)
            ax.set_ylim(z0, z1)
            ax.set_xlabel("$R$")
            ax.set_ylabel("$Z$")
        fig.suptitle(title, fontsize=11)
        fig.subplots_adjust(right=0.88, wspace=0.28, hspace=0.32)
        if sc is not None:
            cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(sc, cax=cax)
            cbar.ax.set_ylabel(r"$\iota$")
            if ticks:
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(labels)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        save_figure(fig, dest, pgf=False, dpi=200)
        plt.close(fig)
    _log(f"wrote {dest}")
    publish(dest)
    return dest


def common_grid(shape: tuple[int, int, int], nfp: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The comparison grid: interior radii, midpoint angles.

    Args:
        shape: ``(n_rho, n_theta, n_zeta)``.
        nfp: field periods; zeta spans one of them, in RADIANS, since both
            readers' blocks are evaluated at physical angles.

    Returns:
        ``(rho, theta, zeta)``, each a 1-D array.
    """
    nr, nt, nz = shape
    rho = (np.arange(nr) + 0.5) / nr
    theta = 2.0 * np.pi * (np.arange(nt) + 0.5) / nt
    zeta = 2.0 * np.pi * (np.arange(nz) + 0.5) / (nz * nfp)
    return rho, theta, zeta


def _norms(got: Any, want: Any) -> dict[str, float]:
    """Relative sup and L2 differences of two arrays on the same grid.

    Args:
        got: the candidate values.
        want: the VMEC-side reference.

    Returns:
        ``sup`` and ``l2``, each divided by the reference's own norm (or by
        one, when the reference is identically zero).
    """
    got, want = np.asarray(got), np.asarray(want)
    sup_ref = max(float(np.abs(want).max()), 1e-30)
    l2_ref = max(float(np.sqrt((want ** 2).mean())), 1e-30)
    return dict(sup=float(np.abs(got - want).max()) / sup_ref,
                l2=float(np.sqrt(((got - want) ** 2).mean())) / l2_ref,
                ref_sup=sup_ref)


def _profile_spline(st: dict[str, Any], name: str) -> Any:
    """The kind's own profile spline; all three share this signature.

    Args:
        st: a parsed equilibrium, with ``kind`` set.
        name: ``"iota"`` or ``"pressure"``.

    Returns:
        A :class:`scipy.interpolate.BSpline` in the radial label.
    """
    if st.get("kind") == "desc":
        from mrx.desc import profile_spline
    elif st.get("kind") == "vmec":
        from mrx.vmec import profile_spline
    else:
        from mrx.gvec import profile_spline
    return profile_spline(st, name)


def _align_gvec(st: dict[str, Any]) -> dict[str, Any]:
    """Undo pyGVEC ``convert-wout``'s two sign conventions.

    The converter writes Fourier ``n`` with the opposite sign from VMEC's
    ``xn`` -- ``R`` agrees at ``zeta = 0`` and diverges elsewhere until
    every block's ``n`` is negated -- and sets ``phiedge = -phiedge``,
    which flips ``iota`` and ``phi``. Neither is a poloidal flip:
    :func:`mrx.gvec.flip_poloidal_angle` extra-negates the sine blocks and
    makes ``Z`` worse.

    Args:
        st: a GVEC state from :func:`mrx.gvec.read_state`.

    Returns:
        A new state dict in the wout's angle and flux conventions.
    """
    out = dict(st)
    for name in ("X1", "X2", "LA"):
        blk = dict(st[name])
        blk["n"] = -np.asarray(blk["n"])
        out[name] = blk
    prof = dict(st["profiles"])
    prof["iota"] = -np.asarray(prof["iota"])
    if "phi" in prof:
        prof["phi"] = -np.asarray(prof["phi"])
    out["profiles"] = prof
    return out


def compare_states(st: dict[str, Any], st_v: dict[str, Any],
                   shape: tuple[int, int, int]) -> dict[str, Any]:
    """Geometry, lambda and the profiles of one candidate against a wout.

    The candidate is first turned to the wout's poloidal orientation
    (:func:`mrx.gvec.match_orientation`), without which this compares two
    different points and every number is meaningless.

    Args:
        st: the candidate state (DESC or GVEC), with ``kind`` set.
        st_v: the VMEC state (:func:`mrx.vmec.read_wout`).
        shape: the comparison grid.

    Returns:
        Per-field ``sup`` / ``l2`` dicts under ``R``, ``Z``, ``lambda`` and
        ``lambda_outer`` (the latter outside :data:`LAMBDA_CORE`), the
        profile differences under ``iota`` and ``pressure``, and the
        measured ``orientation``.
    """
    from mrx.gvec import evaluate, match_orientation

    nfp = st_v["nfp"]
    if st.get("kind") == "gvec":
        st = _align_gvec(st)
    st, sign = match_orientation(st, st_v)
    rho, theta, zeta = common_grid(shape, nfp)
    out: dict[str, Any] = dict(orientation=sign, kind=st.get("kind"))
    for name, blk in (("R", "X1"), ("Z", "X2"), ("lambda", "LA")):
        got = evaluate(st[blk], rho, theta, zeta)
        want = evaluate(st_v[blk], rho, theta, zeta)
        out[name] = _norms(got, want)
        if name == "lambda":
            keep = rho >= LAMBDA_CORE
            out["lambda_outer"] = _norms(got[keep], want[keep])
    r = np.linspace(0.0, 1.0, 201)
    for name in ("iota", "pressure"):
        out[name] = _norms(_profile_spline(st, name)(r),
                           _profile_spline(st_v, name)(r))
    for tag, spline in (("cand", _profile_spline(st, "iota")),
                        ("vmec", _profile_spline(st_v, "iota"))):
        out[f"iota_axis_{tag}"] = float(spline(0.0))
        out[f"iota_edge_{tag}"] = float(spline(1.0))
    return out


def field_diagnostics(seq: Any, eq: dict[str, Any], tag: str) -> tuple[Any, dict]:
    """The production initial field of one state, and what it measures.

    Args:
        seq: the shared sequence; its ``equilibrium`` is swapped to ``eq``.
        eq: the parsed equilibrium dict to build the field from.
        tag: a short label for the log line.

    Returns:
        ``(B, info)``: the 2-form DoFs and the dict of
        :func:`mrx.initial_conditions.initial_field` extended with the
        toroidal flux and the Leray force residual.
    """
    from mrx.initial_conditions import initial_field
    from mrx.relaxation import compute_force

    seq.equilibrium = eq
    t0 = time.perf_counter()
    B, info = initial_field(seq)
    F, _, _, _, _ = compute_force(B, seq)
    info = dict(info)
    info["force_residual"] = float(seq.l2_norm(F, 2))
    info["toroidal_flux"] = float(seq.evaluate_at_quadrature(B, 2, True)[:, 2] @ seq.quad.w)
    _log(f"{tag}: IC {time.perf_counter() - t0:.0f} s, "
         f"div {info['div']:.2e}, |F| {info['force_residual']:.3e}, "
         f"iota {info['iota_axis']:+.4f}..{info['iota_edge']:+.4f}")
    return B, info


def compare_fields(seq: Any, eq_d: dict[str, Any], eq_v: dict[str, Any]
                   ) -> dict[str, Any]:
    """Both initial fields on one sequence, differenced in the mass norm.

    Args:
        seq: the shared sequence.
        eq_d: the DESC equilibrium dict.
        eq_v: the VMEC equilibrium dict.

    Returns:
        ``desc`` and ``vmec`` diagnostics and ``B_rel_diff``, the relative
        M-norm distance between the two fields.
    """
    B_d, info_d = field_diagnostics(seq, eq_d, "desc")
    B_v, info_v = field_diagnostics(seq, eq_v, "vmec")
    rel = float(seq.l2_norm(B_d - B_v, 2) / seq.l2_norm(B_v, 2))
    _log(f"||B_d - B_v||_M / ||B_v||_M = {rel:.4e}")
    return dict(desc=info_d, vmec=info_v, B_rel_diff=rel)


def refit(wout: str, lmn: tuple[int, int, int], path: str) -> str:
    """A DESC refit of ``wout`` at resolution ``lmn``, cached on disk.

    Args:
        wout: the VMEC file to fit.
        lmn: ``(L, M, N)`` of the Fourier-Zernike fit.
        path: where to write it; reused if it is already there.

    Returns:
        ``path``.
    """
    if os.path.isfile(path):
        _log(f"reusing {path}")
        return path
    from desc.vmec import VMECIO  # noqa: PLC0415  (optional dependency)
    t0 = time.perf_counter()
    L, M, N = lmn
    VMECIO.load(wout, L=L, M=M, N=N, profile="iota").save(path)
    _log(f"refit L={L} M={M} N={N} in {time.perf_counter() - t0:.0f} s -> {path}")
    return path


def gvec_state(wout: str, out_dir: str, nelems: int | None = None) -> str:
    """The GVEC refit of ``wout``, converted once and reused.

    Args:
        wout: the VMEC file to convert.
        out_dir: directory that will hold the ``*State*.dat``.
        nelems: GVEC radial elements; ``None`` keeps the converter default
            of 10.

    Returns:
        Path of the newest ``*State*.dat`` under ``out_dir``.

    Raises:
        ValueError: if the converted lambda block is identically zero,
            which means ``init_LA`` did not take.
    """
    os.makedirs(out_dir, exist_ok=True)
    hit = sorted(glob.glob(os.path.join(out_dir, "*State*.dat")))
    if hit:
        _log(f"reusing {hit[-1]}")
        return hit[-1]
    from gvec.scripts.convert_wout import convert_vmec_wout  # noqa: PLC0415

    extra: dict[str, Any] = {"init_LA": True}
    if nelems is not None:
        extra["sgrid"] = {"nelems": nelems}
    t0 = time.perf_counter()
    convert_vmec_wout(Path(wout), Path(out_dir), extra_parameters=extra)
    path = sorted(glob.glob(os.path.join(out_dir, "*State*.dat")))[-1]
    _log(f"convert-wout {os.path.basename(wout)} nelems={nelems} in "
         f"{time.perf_counter() - t0:.0f} s -> {path}")
    return path


def _require_gvec_lambda(st: dict[str, Any], path: str) -> None:
    """Refuse a GVEC state whose lambda was never initialised.

    Args:
        st: the parsed GVEC state.
        path: the file, for the error message.

    Raises:
        ValueError: if the ``LA`` coefficients are identically zero.
    """
    if np.abs(st["LA"]["coef"]).max() < 1e-12:
        raise ValueError(f"{path}: GVEC lambda is identically zero; "
                         "pass --param init_LA=True to convert-wout")


def _with_kind(st: dict[str, Any], kind: str, path: str) -> dict[str, Any]:
    """Attach ``kind`` and ``path`` the way :func:`read_equilibrium` does.

    Args:
        st: a parsed state.
        kind: ``"desc"``, ``"vmec"`` or ``"gvec"``.
        path: the file it came from.

    Returns:
        A new dict.
    """
    return dict(st, kind=kind, path=path)


def _log_geometry(tag: str, g: dict[str, Any]) -> None:
    """One compact line of geometry-comparison numbers.

    Args:
        tag: a short label.
        g: the dict of :func:`compare_states`.
    """
    _log(f"{tag}: orientation {g['orientation']:+d}  R sup {g['R']['sup']:.3e}  "
         f"Z sup {g['Z']['sup']:.3e}  lambda sup {g['lambda']['sup']:.3e} "
         f"(outside rho={LAMBDA_CORE}: {g['lambda_outer']['sup']:.3e})  "
         f"iota sup {g['iota']['sup']:.3e}")


def run_grid(cli: argparse.Namespace) -> None:
    """Sweep DESC fit resolution against the wout and write each rung.

    Args:
        cli: parsed arguments; see :func:`parse_args`.
    """
    from mrx.desc import read_desc
    from mrx.gvec import match_orientation, read_equilibrium
    from mrx.vmec import read_wout

    shape = tuple(int(v) for v in cli.grid.split(","))
    st_v = _with_kind(read_wout(cli.wout), "vmec", cli.wout)
    eq_v = dict(st_v)
    ntor = int(np.abs(st_v["X1"]["n"]).max() // st_v["nfp"])
    os.makedirs(cli.out, exist_ok=True)

    rungs = []
    for res in (int(v) for v in cli.sweep.split(",")):
        lmn = (res, res, min(res, ntor))
        tag = f"L{lmn[0]}M{lmn[1]}N{lmn[2]}"
        rungs.append((refit(cli.wout, lmn, os.path.join(cli.out, f"fit_{tag}.h5")), tag))

    seq = None
    for path, tag in rungs:
        _log(f"--- rung {tag}")
        st_d = _with_kind(read_desc(path), "desc", path)
        result = dict(tag=tag, desc_file=path, wout=cli.wout, grid=list(shape),
                      L=st_d["L"], M=st_d["M"], N=st_d["N"],
                      geometry=compare_states(st_d, st_v, shape))
        _log_geometry(tag, result["geometry"])
        if cli.ns:
            if seq is None:
                from mrx.geometry import build_sequence
                from mrx.nullspace import compute_nullspaces
                ns = tuple(int(v) for v in cli.ns.split(","))
                t0 = time.perf_counter()
                seq, _ = build_sequence(cli.wout, ns, cli.p)
                compute_nullspaces(seq)
                _log(f"sequence {ns} p={cli.p} in {time.perf_counter() - t0:.0f} s")
            result["ns"], result["p"] = [int(v) for v in cli.ns.split(",")], cli.p
            eq_d, _ = match_orientation(read_equilibrium(path), st_v)
            result["field"] = compare_fields(seq, eq_d, eq_v)
        rung_dir = os.path.join(cli.out, f"rung_{tag}")
        os.makedirs(rung_dir, exist_ok=True)
        with open(os.path.join(rung_dir, "result.json"), "w") as fh:
            json.dump(result, fh, indent=2)
    print(f"wrote {len(rungs)} rungs under {cli.out}", flush=True)


def figure_grid(cli: argparse.Namespace) -> None:
    """DESC-vs-VMEC fit-resolution sweep on a common grid.

    The DESC-vs-GVEC comparison is ``--figure gvec`` / ``gvec_grid.png``.

    Args:
        cli: parsed arguments.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mrx.plotstyle import house_style
    from mrx.plotting import save_figure

    if not cli.plot_only:
        run_grid(cli)

    results = []
    for path in sorted(glob.glob(os.path.join(cli.out, "rung_*", "result.json"))):
        with open(path) as fh:
            results.append(json.load(fh))
    if not results:
        raise SystemExit(f"no rungs under {cli.out}; run without --plot-only")
    results.sort(key=lambda r: (r["L"], r["M"], r["N"]))
    L = np.array([r["L"] for r in results])

    dest = os.path.join(cli.out, "grid_convergence.png")
    with house_style():
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), layout="none")
        for name, marker in (("R", "o"), ("Z", "s"), ("lambda", "^")):
            axes[0].semilogy(L, [r["geometry"][name]["sup"] for r in results],
                             marker=marker, label=f"{name} sup")
        if "lambda_outer" in results[0]["geometry"]:
            axes[0].semilogy(L, [r["geometry"]["lambda_outer"]["sup"] for r in results],
                             marker="^", ls="--", color="C2", mfc="none",
                             label=rf"lambda sup, $\rho \geq {LAMBDA_CORE}$")
        axes[0].semilogy(L, [r["geometry"]["iota"]["sup"] for r in results],
                         marker="v", ls=":", label="iota sup")
        axes[0].set_xlabel("DESC radial/poloidal fit resolution $L = M$")
        axes[0].set_ylabel("relative difference from VMEC")
        axes[0].set_title("geometry and lambda on a common grid", fontsize=9)
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        have_field = [r for r in results if "field" in r]
        if have_field:
            axes[1].semilogy([r["L"] for r in have_field],
                             [r["field"]["B_rel_diff"] for r in have_field],
                             marker="o", color="C3",
                             label=r"$\|B_d-B_v\|_M/\|B_v\|_M$")
            axes[1].semilogy([r["L"] for r in have_field],
                             [r["field"]["desc"]["force_residual"] for r in have_field],
                             marker="s", ls="--", label="DESC IC force residual")
            axes[1].axhline(have_field[0]["field"]["vmec"]["force_residual"],
                            color="k", ls=":", lw=1, label="VMEC IC force residual")
            axes[1].legend(fontsize=8)
        else:
            axes[1].text(0.5, 0.5, "no --ns rungs", ha="center", va="center",
                         transform=axes[1].transAxes)
        axes[1].set_xlabel("DESC radial/poloidal fit resolution $L = M$")
        axes[1].set_title("the initial 2-form, in the mass norm", fontsize=9)
        axes[1].grid(alpha=0.3)

        fig.suptitle(f"DESC vs VMEC, {os.path.basename(results[0]['wout'])}", fontsize=10)
        fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.16, wspace=0.28)
        os.makedirs(cli.out, exist_ok=True)
        save_figure(fig, dest, pgf=False, dpi=200)
        plt.close(fig)
    with open(os.path.join(cli.out, "grid_convergence.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    _log(f"wrote {dest}")
    publish(dest)


def _gvec_pair(wout: str, desc: str, cli: argparse.Namespace, name: str
               ) -> dict[str, Any]:
    """Three-way geometry comparison of one wout.

    Args:
        wout: the VMEC reference.
        desc: the DESC reading of the same equilibrium.
        cli: parsed arguments.
        name: cache subdirectory (``"li383"`` or ``"qa"``).

    Returns:
        The comparison dict written to ``gvec_<name>.json``.
    """
    from mrx.desc import read_desc
    from mrx.gvec import read_state
    from mrx.vmec import read_wout

    shape = tuple(int(v) for v in cli.grid.split(","))
    st_v = _with_kind(read_wout(wout), "vmec", wout)
    st_d = _with_kind(read_desc(desc), "desc", desc)
    out: dict[str, Any] = dict(wout=wout, desc=compare_states(st_d, st_v, shape),
                               grid=list(shape))
    _log_geometry(f"{name}/desc", out["desc"])
    for nelems in (10, 40):
        path = gvec_state(wout, os.path.join(cli.out, f"gvec_{name}_{nelems}"),
                          nelems=nelems)
        st_g = _with_kind(read_state(path), "gvec", path)
        _require_gvec_lambda(st_g, path)
        key = f"gvec_{nelems}"
        out[key] = compare_states(st_g, st_v, shape)
        out[f"{key}_file"] = path
        _log_geometry(f"{name}/{key}", out[key])
    dest = os.path.join(cli.out, f"gvec_{name}.json")
    os.makedirs(cli.out, exist_ok=True)
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def figure_gvec(cli: argparse.Namespace) -> None:
    """DESC and GVEC against the same wout, for li383 and QA.

    Args:
        cli: parsed arguments.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mrx.plotstyle import house_style
    from mrx.plotting import save_figure

    rows = []
    have_gvec = True
    try:
        import gvec  # noqa: F401, PLC0415
    except ImportError:
        have_gvec = False
    for name, wout, desc in (("li383", cli.wout, cli.desc),
                             ("qa", cli.qa_wout, cli.qa_desc)):
        path = os.path.join(cli.out, f"gvec_{name}.json")
        if (cli.plot_only or not have_gvec) and os.path.isfile(path):
            with open(path) as fh:
                rows.append((name, json.load(fh)))
            continue
        if not have_gvec:
            _log(f"{name}: pyGVEC is not importable and {path} is missing, skipping")
            continue
        rows.append((name, _gvec_pair(wout, desc, cli, name)))
    if not rows:
        _log("no GVEC comparisons to plot")
        return

    dest = os.path.join(cli.out, "gvec_grid.png")
    keys = ("desc", "gvec_10", "gvec_40")
    labels = ("DESC", "GVEC 10", "GVEC 40")
    qtys = (("R", "R"), ("Z", "Z"), ("lambda", r"$\lambda$"),
            ("lambda_outer", rf"$\lambda$, $\rho\geq{LAMBDA_CORE}$"))
    with house_style():
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), sharey=True, layout="none")
        for ax, (name, rec) in zip(axes, rows):
            xs = np.arange(len(keys))
            width = 0.18
            for i, (qty, lab) in enumerate(qtys):
                ax.semilogy(xs + (i - 1.5) * width,
                            [rec[k][qty]["sup"] for k in keys],
                            marker="o", ls="none", label=lab)
            ax.set_xticks(xs, labels)
            ax.set_title(name, fontsize=10)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("relative difference from VMEC")
        axes[0].legend(fontsize=7)
        fig.suptitle("DESC vs GVEC refit of the same wout", fontsize=11)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.14, wspace=0.22)
        os.makedirs(cli.out, exist_ok=True)
        save_figure(fig, dest, pgf=False, dpi=200)
        plt.close(fig)
    _log(f"wrote {dest}")
    publish(dest)


def _relax_record(tag: str, rec: dict[str, Any]) -> dict[str, Any]:
    """JSON-ready summary of one cached relaxation.

    Args:
        tag: ``"desc"`` or ``"vmec"``.
        rec: the dict of :func:`relax_one`.

    Returns:
        The record written into ``result.json``.
    """
    qoi = {k: np.asarray(v).tolist() for k, v in rec["qoi"].items()}
    first = {k: v[0] for k, v in qoi.items()}
    last = {k: v[-1] for k, v in qoi.items()}
    drift = abs(last["helicity"] - first["helicity"]) / max(abs(first["helicity"]), 1e-300)
    return dict(tag=tag, initial=rec["info"], steps=int(rec["steps"]),
                stop=str(rec["stop"]), wall=float(rec["wall"]),
                E0=float(np.asarray(rec["E0"])), qoi=qoi,
                energy_first=first["E"], energy_last=last["E"],
                resid_first=first["resid"], resid_last=last["resid"],
                helicity_first=first["helicity"], helicity_last=last["helicity"],
                helicity_drift=drift,
                energy_monotone=bool(np.all(np.asarray(rec["dE"]) <= 0.0)))


def figure_relax(cli: argparse.Namespace) -> None:
    """Relax from VMEC and DESC initial conditions on one sequence.

    Args:
        cli: parsed arguments.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mrx.plotstyle import house_style
    from mrx.plotting import save_figure

    dest_json = os.path.join(cli.out, "desc_vmec_relax.json")
    if cli.plot_only and os.path.isfile(dest_json):
        with open(dest_json) as fh:
            result = json.load(fh)
    else:
        seq, rec_v, rec_d, sign = pair_from_wout(cli.wout, cli.desc, cli, "li383")

        def rel(a: Any, b: Any) -> float:
            return float(seq.l2_norm(a - b, 2) / seq.l2_norm(b, 2))

        rec_v_rec, rec_d_rec = _relax_record("vmec", rec_v), _relax_record("desc", rec_d)
        initial_diff = rel(rec_d["B0"], rec_v["B0"])
        final_diff = rel(rec_d["B_final"], rec_v["B_final"])
        h_v, h_d = rec_v_rec["helicity_first"], rec_d_rec["helicity_first"]
        helicity_diff = abs(h_d - h_v) / max(abs(h_v), 1e-300)
        _log(f"||B_d - B_v||_M / ||B_v||_M: initial {initial_diff:.4e} -> "
             f"final {final_diff:.4e}   (relative helicity difference {helicity_diff:.4e})")
        comparison = dict(
            initial_B_rel_diff=initial_diff, final_B_rel_diff=final_diff,
            amplification=final_diff / max(initial_diff, 1e-300),
            initial_helicity_rel_diff=helicity_diff,
            energy_rel_diff=abs(rec_d_rec["energy_last"] - rec_v_rec["energy_last"])
            / max(abs(rec_v_rec["energy_last"]), 1e-300),
            orientation=int(sign))
        for key in ("beta_vol", "JoverB", "JB", "resid"):
            a, b = rec_d_rec["qoi"].get(key), rec_v_rec["qoi"].get(key)
            if a is not None and b is not None:
                comparison[f"{key}_desc"], comparison[f"{key}_vmec"] = a[-1], b[-1]
                comparison[f"{key}_rel_diff"] = abs(a[-1] - b[-1]) / max(abs(b[-1]), 1e-300)
        result = dict(wout=cli.wout, desc=cli.desc,
                      ns=[int(v) for v in cli.ns.split(",")], p=cli.p,
                      steps=cli.steps, precision=cli.precision,
                      vmec=rec_v_rec, desc_run=rec_d_rec, comparison=comparison)
        os.makedirs(cli.out, exist_ok=True)
        with open(dest_json, "w") as fh:
            json.dump(result, fh, indent=2)

    dest = os.path.join(cli.out, "desc_vmec_relax.png")
    with house_style():
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for rec, colour in ((result["vmec"], "C0"), (result["desc_run"], "C3")):
            it = np.asarray(rec["qoi"]["it"])
            axes[0].plot(it, np.asarray(rec["qoi"]["E"]), color=colour, label=rec["tag"])
            axes[1].semilogy(it, np.asarray(rec["qoi"]["resid"]), color=colour,
                             label=rec["tag"])
            h = np.asarray(rec["qoi"]["helicity"])
            axes[2].semilogy(it, np.abs(h / h[0] - 1.0) + 1e-18, color=colour,
                             label=f"{rec['tag']} helicity drift")
        axes[0].set_ylabel("energy")
        axes[1].set_ylabel("normalised force residual")
        axes[2].set_ylabel(r"$|H/H_0 - 1|$")
        c = result["comparison"]
        axes[2].axhline(c["initial_helicity_rel_diff"], color="k", ls=":", lw=1,
                        label="DESC-VMEC initial helicity gap")
        for ax in axes:
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"{os.path.basename(result['wout'])}: "
                     rf"$\|B_d-B_v\|_M/\|B_v\|_M$ {c['initial_B_rel_diff']:.2e} "
                     rf"$\to$ {c['final_B_rel_diff']:.2e}", fontsize=10)
        fig.tight_layout()
        os.makedirs(cli.out, exist_ok=True)
        save_figure(fig, dest, pgf=False, dpi=200)
        plt.close(fig)
    _log(f"wrote {dest}")
    publish(dest)


def figure_traces(cli: argparse.Namespace) -> None:
    """Block-averaged force residual and per-step energy release on li383.

    Args:
        cli: parsed arguments.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mrx.plotstyle import BLACK, TEAL, house_style
    from mrx.plotting import save_figure

    _, rec_v, rec_d, _ = pair_from_wout(cli.wout, cli.desc, cli, "li383")
    dest = os.path.join(cli.out, "desc_traces.png")
    with house_style():
        fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True)
        for rec, colour, ls, lab in (
                (rec_v, BLACK, "-", "VMEC"),
                (rec_d, TEAL, "--", "DESC")):
            F = np.asarray(rec["F"], float)
            release = -np.asarray(rec["dE"], float)
            total = float(release.sum())
            plot_trace(axes[0], F, log=True, color=colour, ls=ls, lw=1.2, label=lab)
            plot_trace(axes[1], release, log=True, color=colour, ls=ls, lw=1.2,
                       label=f"{lab} (released {total:.3e})")
        axes[0].set_ylabel(r"$\|F\|_M$")
        axes[0].set_yscale("log")
        axes[1].set_ylabel(r"$-\,\mathrm{d}E$ per step")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("step")
        axes[1].set_xscale("log")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("li383: VMEC vs DESC initial condition", fontsize=11)
        fig.tight_layout()
        os.makedirs(cli.out, exist_ok=True)
        save_figure(fig, dest, pgf=False, dpi=200)
        plt.close(fig)
    _log(f"wrote {dest}")
    publish(dest)


def figure_pair(cli: argparse.Namespace, name: str, wout: str, desc: str,
                title: str) -> None:
    """2×2 Poincaré of one wout: VMEC/DESC × initial/relaxed.

    Shared by ``--figure li383`` and ``--figure qa``.

    Args:
        cli: parsed arguments.
        name: cache subdirectory and ``poincare_<name>.png`` stem.
        wout: the VMEC file the sequence is built from.
        desc: the DESC reading of the same equilibrium.
        title: the figure title.
    """
    seq, rec_v, rec_d, _ = pair_from_wout(wout, desc, cli, name)
    nfp = int(seq.equilibrium["nfp"])
    panels = []
    for tag, rec in (("VMEC", rec_v), ("DESC", rec_d)):
        for state, key in (("initial", "B0"), ("relaxed", "B_final")):
            label = f"{tag} {state}"
            cache = _cache_path(cli.out, name,
                                f"section_{tag.lower()}_{state}_{_run_tag(cli)}.npz")
            panels.append((label, section_of(seq, rec[key], nfp, cli, cache, label)))
    poincare_grid(panels, os.path.join(cli.out, f"poincare_{name}.png"), nfp, title)


def _plot_case_mesh(path: str, cli: argparse.Namespace, slug: str) -> None:
    """Drive ``scripts/plot_mesh.py`` for one DESC file and publish the 2-D mesh.

    Args:
        path: the DESC ``.h5``.
        cli: parsed arguments.
        slug: published basename stem.
    """
    mesh_dir = os.path.join(cli.out, slug, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)
    cmd = [sys.executable, "scripts/plot_mesh.py", "--geometry", path,
           "--meshes", cli.ns, "--p", str(cli.p), "--planes", "0,0.5",
           "--out", mesh_dir, "--precision", cli.precision]
    _log(f"{slug}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    src = os.path.join(mesh_dir, "mesh_2d.png")
    if os.path.isfile(src):
        publish(src, f"mesh_{slug}.png")
    else:
        _log(f"{slug}: plot_mesh.py wrote no mesh_2d.png")


def figure_cases(cli: argparse.Namespace) -> None:
    """Mesh (selected cases) plus initial/relaxed Poincaré for the 3-D examples.

    Args:
        cli: parsed arguments.
    """
    root = examples_dir(cli.examples)
    for name, explicit, want_mesh in CASE_FIGURES:
        path = explicit or locate(root, name)
        if path is None:
            _log(f"{name}: not found under {root}, skipping")
            continue
        slug = name.lower().replace("-", "")
        if want_mesh and not cli.no_mesh:
            _plot_case_mesh(path, cli, slug)
        seq, rec = single_from_desc(path, cli, slug)
        nfp = int(seq.equilibrium["nfp"])
        panels = []
        for state, key in (("initial", "B0"), ("relaxed", "B_final")):
            label = f"{name} {state}"
            cache = _cache_path(cli.out, slug,
                                f"section_{state}_{_run_tag(cli)}.npz")
            panels.append((label, section_of(seq, rec[key], nfp, cli, cache, label)))
        poincare_grid(panels, os.path.join(cli.out, f"poincare_case_{slug}.png"), nfp,
                      f"{name}: initial vs relaxed")


def main(cli: argparse.Namespace) -> None:
    """Dispatch ``--figure``.

    Args:
        cli: parsed arguments.
    """
    import mrx
    # jax 0.8.1 (the Torch overlay) cannot take batch_size=0; 0.8.2+ can.
    # A positive MRX_MAP_BATCH_SIZE_INNER is the overlay workaround.
    _mbs = os.environ.get("MRX_MAP_BATCH_SIZE_INNER")
    if _mbs:
        mrx.MAP_BATCH_SIZE_INNER = int(_mbs)
    wanted = FIGURES if cli.figure == "all" else (cli.figure,)
    dispatch = dict(grid=figure_grid, gvec=figure_gvec, relax=figure_relax,
                    traces=figure_traces, cases=figure_cases)
    pairs = dict(
        li383=("li383", cli.wout, cli.desc,
               "li383: VMEC vs DESC, initial vs relaxed"),
        qa=("qa", cli.qa_wout, cli.qa_desc,
            "Landreman–Paul QA vacuum: VMEC vs DESC"),
    )
    for name in wanted:
        _log(f"=== {name} ===")
        if name in pairs:
            figure_pair(cli, *pairs[name])
        else:
            dispatch[name](cli)


if __name__ == "__main__":
    args = parse_args()
    os.environ["MRX_DTYPE"] = args.precision
    if args.trace_steps % args.saves:
        raise SystemExit("--trace-steps must be a multiple of --saves")
    main(args)
