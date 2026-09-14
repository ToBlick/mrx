"""Poincaré sections and relaxation traces for the DESC interface writeup.

Owns its own compute rather than depending on checkpoints the comparison
scripts do not write. Four relaxations share one helper, and every expensive
result is cached under ``outputs/desc_figures/<case>/`` so ``--figure all``
is cheap the second time::

    python -u scripts/desc_figures.py --figure traces
    python -u scripts/desc_figures.py --figure all --ns 8,12,12 --p 2 --steps 1000

``--precision`` is exported as ``MRX_DTYPE`` before ``mrx`` is imported.
Figures land in ``--out`` and are copied into
``docs/research/desc_interface_2026-09-13/``.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from typing import Any

import numpy as np

#: Block width of the house per-step traces
#: (``scripts/li383_pub_figures.blocked``).
BLOCK = 100

#: Destination of the published PNGs, next to the writeup they illustrate.
RESEARCH_DIR = os.path.join("docs", "research", "desc_interface_2026-09-13")

FIGURES = ("li383", "w7x", "qa", "traces")


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
    ap.add_argument("--steps", type=int, default=1000)
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
    ap.add_argument("--precision", default="float64", choices=("float32", "float64"))
    ap.add_argument("--out", default="outputs/desc_figures")
    ap.add_argument("--force", action="store_true",
                    help="ignore cached fields.npz / section.npz")
    return ap.parse_args(argv)


def _log(msg: str) -> None:
    """Print a timestamped progress line.

    Args:
        msg: the message.
    """
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def blocked(y: np.ndarray, log: bool = True, w: int = BLOCK
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Block means of a per-step trace and the ±1 sd band.

    Copied from ``scripts/li383_pub_figures.blocked`` so this script does
    not import that module (it pulls ``mrx`` at import time). ``log`` takes
    the statistics in log space for a log axis. The last partial block is
    kept.

    Args:
        y: the per-step samples.
        log: whether to average in log space.
        w: block width in steps.

    Returns:
        ``(centre_step, mean, lower, upper)``.
    """
    y = np.asarray(y, float)
    n = len(y)
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


def plot_trace(ax: Any, y: np.ndarray, log: bool = True, **kw: Any) -> Any:
    """One block-averaged trace with its sd ribbon.

    Args:
        ax: the matplotlib axes.
        y: the per-step samples.
        log: whether to average in log space.
        **kw: forwarded to the line.

    Returns:
        The line artist.
    """
    x, m, lo, hi = blocked(y, log)
    (line,) = ax.plot(x, m, **kw)
    ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.2, lw=0)
    return line


def publish(src: str) -> str:
    """Copy ``src`` into the research-note figure directory.

    Args:
        src: path of the PNG just written.

    Returns:
        The published path.
    """
    os.makedirs(RESEARCH_DIR, exist_ok=True)
    dest = os.path.join(RESEARCH_DIR, os.path.basename(src))
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
    for pattern in (f"{name}_output.h5", f"desc_{name}.h5", f"{name}.h5"):
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
        ``B0``, ``B_final``, per-step ``F`` and ``dE``, ``E0``, and the
        initial-field ``info``.
    """
    from mrx.initial_conditions import initial_field
    from mrx.relaxation import initial_state, relax

    if os.path.isfile(cache) and not cli.force:
        data = np.load(cache, allow_pickle=True)
        rec = {k: data[k] for k in data.files}
        rec["info"] = rec["info"].item() if rec["info"].shape == () else rec["info"]
        _log(f"{tag}: loaded {cache}")
        return rec

    seq.equilibrium = eq
    t0 = time.perf_counter()
    B0, info = initial_field(seq)
    _log(f"{tag}: IC in {time.perf_counter() - t0:.0f} s, "
         f"div {info['div']:.2e}, iota {info['iota_axis']:+.4f}.."
         f"{info['iota_edge']:+.4f}")
    res = relax(initial_state(B0, ts), ts, cli.steps, chunk=cli.chunk, verbose=False)
    rec = dict(B0=np.asarray(B0), B_final=np.asarray(res.state.B_n),
               F=np.asarray(res.trace["F"]), dE=np.asarray(res.trace["dE"]),
               E0=np.asarray(res.E0), info=np.asarray(info, dtype=object))
    np.savez_compressed(cache, **rec)
    rec["info"] = info
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
    from mrx.desc import match_orientation
    from mrx.geometry import build_sequence
    from mrx.gvec import read_equilibrium
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
    rec_v = relax_one(seq, eq_v, ts, cli,
                      _cache_path(cli.out, name, "vmec", "fields.npz"), f"{name}/vmec")
    rec_d = relax_one(seq, eq_d, ts, cli,
                      _cache_path(cli.out, name, "desc", "fields.npz"), f"{name}/desc")
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
                    _cache_path(cli.out, name, "fields.npz"), name)
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


def figure_li383(cli: argparse.Namespace) -> None:
    """2×2 Poincaré of li383: VMEC/DESC × initial/relaxed.

    Args:
        cli: parsed arguments.
    """
    seq, rec_v, rec_d, _ = pair_from_wout(cli.wout, cli.desc, cli, "li383")
    nfp = int(seq.equilibrium["nfp"])
    panels = []
    for tag, rec in (("VMEC", rec_v), ("DESC", rec_d)):
        for state, key in (("initial", "B0"), ("relaxed", "B_final")):
            label = f"{tag} {state}"
            cache = _cache_path(cli.out, "li383", f"section_{tag.lower()}_{state}.npz")
            panels.append((label, section_of(seq, rec[key], nfp, cli, cache, label)))
    poincare_grid(panels, os.path.join(cli.out, "poincare_li383.png"), nfp,
                  "li383: VMEC vs DESC, initial vs relaxed")


def figure_qa(cli: argparse.Namespace) -> None:
    """2×2 Poincaré of the Landreman–Paul QA vacuum.

    Args:
        cli: parsed arguments.
    """
    seq, rec_v, rec_d, _ = pair_from_wout(cli.qa_wout, cli.qa_desc, cli, "qa")
    nfp = int(seq.equilibrium["nfp"])
    panels = []
    for tag, rec in (("VMEC", rec_v), ("DESC", rec_d)):
        for state, key in (("initial", "B0"), ("relaxed", "B_final")):
            label = f"{tag} {state}"
            cache = _cache_path(cli.out, "qa", f"section_{tag.lower()}_{state}.npz")
            panels.append((label, section_of(seq, rec[key], nfp, cli, cache, label)))
    poincare_grid(panels, os.path.join(cli.out, "poincare_qa.png"), nfp,
                  "Landreman–Paul QA vacuum: VMEC vs DESC")


def figure_w7x(cli: argparse.Namespace) -> None:
    """W7-X finite beta, DESC-native, initial and relaxed.

    Args:
        cli: parsed arguments.

    Raises:
        FileNotFoundError: if no W7-X example is on disk.
    """
    path = locate(examples_dir(cli.examples), "W7-X")
    if path is None:
        raise FileNotFoundError(
            "W7-X_output.h5 not found: pass --examples or install DESC")
    seq, rec = single_from_desc(path, cli, "w7x")
    nfp = int(seq.equilibrium["nfp"])
    panels = []
    for state, key in (("initial", "B0"), ("relaxed", "B_final")):
        label = f"W7-X {state}"
        cache = _cache_path(cli.out, "w7x", f"section_{state}.npz")
        panels.append((label, section_of(seq, rec[key], nfp, cli, cache, label)))
    poincare_grid(panels, os.path.join(cli.out, "poincare_w7x.png"), nfp,
                  "W7-X (DESC-native, $\\Psi < 0$): initial vs relaxed")


def figure_traces(cli: argparse.Namespace) -> None:
    """Block-averaged force residual and energy release on li383.

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
            released = np.cumsum(-np.asarray(rec["dE"], float))
            plot_trace(axes[0], F, log=True, color=colour, ls=ls, lw=1.2, label=lab)
            plot_trace(axes[1], released, log=True, color=colour, ls=ls, lw=1.2,
                       label=lab)
        axes[0].set_ylabel(r"$\|F\|_M$")
        axes[0].set_yscale("log")
        axes[1].set_ylabel(r"$E_0 - E$")
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
    dispatch = dict(li383=figure_li383, w7x=figure_w7x, qa=figure_qa,
                    traces=figure_traces)
    for name in wanted:
        _log(f"=== {name} ===")
        dispatch[name](cli)


if __name__ == "__main__":
    args = parse_args()
    os.environ["MRX_DTYPE"] = args.precision
    if args.trace_steps % args.saves:
        raise SystemExit("--trace-steps must be a multiple of --saves")
    main(args)
