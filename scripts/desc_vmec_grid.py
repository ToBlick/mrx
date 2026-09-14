"""DESC against VMEC for the SAME equilibrium, on a common grid.

Both states come from one wout: ``mrx.vmec`` refits it into splines
directly, and ``VMECIO.load`` fits a Fourier-Zernike series to it which
``mrx.desc`` then refits into the same block layout. Identical boundary,
identical profiles, identical everything -- so every difference this script
measures is REPRESENTATION, not configuration. That is the point, and it is
why DESC's shipped ``NCSX_output.h5`` will not serve: it is a different
NCSX from the tracked li383 wout.

Two layers, both reported per rung.

**Geometry and lambda** (no sequence, milliseconds). ``R``, ``Z`` and
lambda from each state on a common ``(rho, theta, zeta)`` grid, as relative
sup and L2 norms; ``iota`` and ``pressure`` from each against the wout's
own profiles. This isolates DESC's Fourier-Zernike fit plus our conversion.

**The initial field** (one sequence, shared). The production IC route
(``initial_field``) from each state on the SAME sequence, compared in the
mass norm the relaxation actually uses: ``||B_d - B_v||_M / ||B_v||_M``,
with each field's ``div``, ``wall_discarded``, toroidal flux and Leray
force residual alongside. Same space, no transfer.

The sweep is the headline. DESC's fit resolution ``(L, M, N)`` is raised
rung by rung and the difference must fall TOWARD the VMEC reference: that
is what says the two codes are describing one equilibrium and our
conversion is faithful. It floors where the wout's own Fourier content runs
out (li383 is ``mpol = 7, ntor = 4``), which is the honest answer, not a
failure -- past that resolution there is nothing left in the file to fit.

Refitting needs DESC; comparing does not. ``--sweep`` writes each refit to
``<out>/fit_L<L>M<M>N<N>.h5`` and reuses it if present, so the expensive
half runs once, wherever DESC lives, and the analysis runs anywhere.

Usage::

    python -u scripts/desc_vmec_grid.py --wout data/wout_li383_low_res_reference.nc \
        --sweep 4,6,8,10 --ns 8,12,12 --p 2 --out outputs/desc_li383
    python -u scripts/desc_vmec_grid.py --plot outputs/desc_li383

``--precision`` is exported as ``MRX_DTYPE`` before ``mrx`` is imported.
Output: ``<out>/rung_L<L>M<M>N<N>/result.json`` per rung and, from
``--plot``, ``<out>/grid_convergence.json`` and ``.png``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

#: Radial, poloidal and toroidal samples of the comparison grid. Midpoints
#: in the angles and interior points in rho, so nothing lands on a spline
#: node, where an interpolatory refit is exact by construction.
GRID = (40, 64, 32)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: argument list; ``None`` reads ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wout", default="data/wout_li383_low_res_reference.nc",
                    help="the VMEC reference, and the file DESC is fit to")
    ap.add_argument("--sweep", default="4,6,8,10",
                    help="comma-separated DESC fit resolutions; L = M = this, "
                         "N = min(this, the wout's ntor)")
    ap.add_argument("--desc", default=None,
                    help="compare this single DESC file instead of sweeping a refit")
    ap.add_argument("--ns", default=None,
                    help="n_r,n_theta,n_zeta of the shared sequence; omit to do "
                         "the geometry comparison only (no sequence, seconds)")
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--grid", default=",".join(str(n) for n in GRID),
                    help="common evaluation grid n_rho,n_theta,n_zeta")
    ap.add_argument("--precision", default="float64", choices=("float32", "float64"))
    ap.add_argument("--out", default="outputs/desc_vmec_grid")
    ap.add_argument("--plot", default=None, help="merge the rungs under this directory and plot")
    return ap.parse_args(argv)


def _log(msg: str) -> None:
    """Print a timestamped progress line.

    Args:
        msg: the message.
    """
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def common_grid(shape: tuple[int, int, int], nfp: int):
    """The comparison grid: interior radii, midpoint angles.

    Args:
        shape: ``(n_rho, n_theta, n_zeta)``.
        nfp: field periods; zeta spans one of them, in RADIANS, since both
            readers' blocks are evaluated at physical angles.

    Returns:
        ``(rho, theta, zeta)``, each a 1-D array.
    """
    import numpy as np
    nr, nt, nz = shape
    rho = (np.arange(nr) + 0.5) / nr
    theta = 2.0 * np.pi * (np.arange(nt) + 0.5) / nt
    zeta = 2.0 * np.pi * (np.arange(nz) + 0.5) / (nz * nfp)
    return rho, theta, zeta


def _norms(got, want) -> dict[str, float]:
    """Relative sup and L2 differences of two arrays on the same grid.

    Args:
        got: the DESC-side values.
        want: the VMEC-side reference.

    Returns:
        ``sup`` and ``l2``, each divided by the reference's own norm (or by
        one, when the reference is identically zero).
    """
    import numpy as np
    got, want = np.asarray(got), np.asarray(want)
    sup_ref = max(float(np.abs(want).max()), 1e-30)
    l2_ref = max(float(np.sqrt((want ** 2).mean())), 1e-30)
    return dict(sup=float(np.abs(got - want).max()) / sup_ref,
                l2=float(np.sqrt(((got - want) ** 2).mean())) / l2_ref,
                ref_sup=sup_ref)


#: Radius below which the lambda comparison is reported separately. DESC's
#: wout importer fits VMEC's HALF-mesh ``lmns`` as if it sat on the full
#: mesh and anchors it at the dummy axis row, so the two codes' lambda
#: disagree near the axis by a fixed amount that no fit resolution removes
#: (``docs/research``). Reporting one number over the whole volume would
#: hide a converging comparison behind a non-converging artefact.
LAMBDA_CORE = 0.3


def compare_states(st_d: dict, st_v: dict, shape: tuple[int, int, int]) -> dict:
    """Geometry, lambda and the profiles of two parsed states.

    The DESC state is first turned to the wout's poloidal orientation
    (:func:`mrx.desc.match_orientation`), without which this compares two
    different points and every number is meaningless.

    Args:
        st_d: the DESC state (:func:`mrx.desc.read_desc`).
        st_v: the VMEC state (:func:`mrx.vmec.read_wout`).
        shape: the comparison grid.

    Returns:
        Per-field ``sup`` / ``l2`` dicts under ``R``, ``Z``, ``lambda`` and
        ``lambda_outer`` (the latter outside :data:`LAMBDA_CORE`), the
        profile differences under ``iota`` and ``pressure``, the axis and
        edge transform of each state, and the measured ``orientation``.
    """
    import numpy as np
    from mrx.desc import match_orientation
    from mrx.desc import profile_spline as desc_profile
    from mrx.gvec import evaluate
    from mrx.vmec import profile_spline as vmec_profile

    nfp = st_v["nfp"]
    st_d, sign = match_orientation(st_d, st_v)
    rho, theta, zeta = common_grid(shape, nfp)
    out = dict(orientation=sign)
    for name, blk in (("R", "X1"), ("Z", "X2"), ("lambda", "LA")):
        got = evaluate(st_d[blk], rho, theta, zeta)
        want = evaluate(st_v[blk], rho, theta, zeta)
        out[name] = _norms(got, want)
        if name == "lambda":
            keep = rho >= LAMBDA_CORE
            out["lambda_outer"] = _norms(got[keep], want[keep])
    r = np.linspace(0.0, 1.0, 201)
    for name in ("iota", "pressure"):
        out[name] = _norms(desc_profile(st_d, name)(r), vmec_profile(st_v, name)(r))
    for tag, spline in (("desc", desc_profile(st_d, "iota")),
                        ("vmec", vmec_profile(st_v, "iota"))):
        out[f"iota_axis_{tag}"] = float(spline(0.0))
        out[f"iota_edge_{tag}"] = float(spline(1.0))
    out["nfp_desc"], out["nfp_vmec"] = int(st_d["nfp"]), int(nfp)
    out["modes_desc"] = int(st_d["X1"]["coef"].shape[0])
    out["modes_vmec"] = int(st_v["X1"]["coef"].shape[0])
    return out


def field_diagnostics(seq, eq, tag: str) -> tuple:
    """The production initial field of one state, and what it measures.

    Args:
        seq: the shared sequence; its ``equilibrium`` is swapped to ``eq``.
        eq: the parsed equilibrium dict to build the field from.
        tag: ``"desc"`` or ``"vmec"``, for the log line.

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


def compare_fields(seq, eq_d: dict, eq_v: dict) -> dict:
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


def run(cli: argparse.Namespace) -> None:
    """Run every rung of the sweep and write its ``result.json``.

    Args:
        cli: parsed arguments; see :func:`parse_args`.
    """
    import numpy as np

    from mrx.desc import read_desc
    from mrx.gvec import read_equilibrium
    from mrx.vmec import read_wout

    shape = tuple(int(v) for v in cli.grid.split(","))
    st_v = read_wout(cli.wout)
    eq_v = dict(st_v, kind="vmec", path=cli.wout)
    ntor = int(np.abs(st_v["X1"]["n"]).max() // st_v["nfp"])
    os.makedirs(cli.out, exist_ok=True)

    if cli.desc:
        rungs = [(cli.desc, os.path.basename(cli.desc).removesuffix(".h5"))]
    else:
        rungs = []
        for res in (int(v) for v in cli.sweep.split(",")):
            lmn = (res, res, min(res, ntor))
            tag = f"L{lmn[0]}M{lmn[1]}N{lmn[2]}"
            rungs.append((refit(cli.wout, lmn, os.path.join(cli.out, f"fit_{tag}.h5")), tag))

    seq = None
    for path, tag in rungs:
        _log(f"--- rung {tag}")
        st_d = read_desc(path)
        result = dict(tag=tag, desc_file=path, wout=cli.wout, grid=list(shape),
                      L=st_d["L"], M=st_d["M"], N=st_d["N"],
                      geometry=compare_states(st_d, st_v, shape))
        g = result["geometry"]
        _log(f"{tag}: orientation {g['orientation']:+d}  R sup {g['R']['sup']:.3e}  "
             f"Z sup {g['Z']['sup']:.3e}  lambda sup {g['lambda']['sup']:.3e} "
             f"(outside rho={LAMBDA_CORE}: {g['lambda_outer']['sup']:.3e})  "
             f"iota sup {g['iota']['sup']:.3e}")
        if cli.ns:
            from mrx.desc import match_orientation
            from mrx.geometry import build_sequence
            if seq is None:                    # one sequence, built on the wout
                from mrx.nullspace import compute_nullspaces
                ns = tuple(int(v) for v in cli.ns.split(","))
                t0 = time.perf_counter()
                seq, _ = build_sequence(cli.wout, ns, cli.p)
                compute_nullspaces(seq)        # the force's Leray route deflates against them
                _log(f"sequence {ns} p={cli.p} in {time.perf_counter() - t0:.0f} s")
                result["ns"], result["p"] = list(ns), cli.p
            # The map is the wout's, so the DESC state must speak its angles.
            eq_d, _ = match_orientation(read_equilibrium(path), st_v)
            result["field"] = compare_fields(seq, eq_d, eq_v)
        rung_dir = os.path.join(cli.out, f"rung_{tag}")
        os.makedirs(rung_dir, exist_ok=True)
        with open(os.path.join(rung_dir, "result.json"), "w") as fh:
            json.dump(result, fh, indent=2)
    print(f"wrote {len(rungs)} rungs under {cli.out}", flush=True)


def plot(root: str) -> None:
    """Merge the rungs under ``root`` and draw the convergence figure.

    Args:
        root: the output directory ``run`` wrote into.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    results = []
    for path in sorted(glob.glob(os.path.join(root, "rung_*", "result.json"))):
        with open(path) as fh:
            results.append(json.load(fh))
    if not results:
        sys.exit(f"no rungs under {root}")
    results.sort(key=lambda r: (r["L"], r["M"], r["N"]))
    L = np.array([r["L"] for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, marker in (("R", "o"), ("Z", "s"), ("lambda", "^")):
        axes[0].semilogy(L, [r["geometry"][name]["sup"] for r in results],
                         marker=marker, label=f"{name} sup")
    if "lambda_outer" in results[0]["geometry"]:
        # Lambda splits in two: outside the axis region it converges like R
        # and Z, inside it does not, because DESC fits VMEC's half-mesh
        # lmns on the full mesh. Plotting only the total hides that.
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
                         marker="o", color="C3", label=r"$\|B_d-B_v\|_M/\|B_v\|_M$")
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
    fig.tight_layout()
    fig.savefig(os.path.join(root, "grid_convergence.png"), dpi=150)
    plt.close(fig)
    with open(os.path.join(root, "grid_convergence.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {root}/grid_convergence.json, grid_convergence.png", flush=True)


if __name__ == "__main__":
    args = parse_args()
    os.environ["MRX_DTYPE"] = args.precision
    if args.plot:
        plot(args.plot)
    else:
        run(args)
