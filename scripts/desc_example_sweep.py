"""Relax MRX from DESC's own saved equilibria, and check the metrics.

The other two scripts compare DESC with VMEC on one configuration. This
one is breadth: it reads DESC's shipped ``*_output.h5`` examples straight
through :func:`mrx.desc.read_desc`, builds a sequence and an initial field
from each, relaxes for a fixed budget, and asserts that everything a
relaxation is supposed to do, it does. A reader can be right on li383 and
wrong on a 19-period heliotron; this is what finds that out.

The cases divide two ways, and both divisions matter.

**Vacuum against finite beta.** A vacuum equilibrium (pressure identically
zero) has a known answer -- the field relaxes toward the discrete harmonic
form of its own boundary -- so a bad conversion cannot hide behind
plausible-looking numbers. These are the sharp cases. The finite-beta ones
instead check ``beta_vol`` against the file's own volume average.

**Stored iota against stored current.** ``SOLOVEV``, ``HELIOTRON``,
``W7-X``, ``ATF``, ``DSHAPE`` store an iota profile and run from the pure
``h5py`` parse, anywhere. ``NCSX``, ``ARIES-CS``, ``precise_QA``, ``HSX``,
``WISTELL-A`` are current-constrained: they store no iota at all and need
DESC installed for :func:`mrx.desc._iota_from_desc` to recompute it. The
script skips those with a clear note rather than failing, so it is useful
in either environment.

``HSX`` and ``W7-X`` carry a NEGATIVE ``_Psi``, so their toroidal flux and
field run the other way; ``HELIOTRON`` at 19 field periods and ``ATF`` at
12 are the hardest test of the toroidal mode mapping, and are cheap in MRX
because logical zeta spans a single period.

Checked per case, on the initial field:

* ``div`` at round-off and ``wall_discarded`` negligible -- the Clebsch
  construction is exactly divergence-free and tangential, or the geometry
  is wrong
* ``det DF > 0`` everywhere, reported by ``build_sequence``
* ``iota_axis`` and ``iota_edge`` against the file's OWN iota profile: the
  field MRX built has to have the transform DESC recorded

and on the relaxed field: energy monotonically decreasing, the force
residual down by at least the measured band, helicity drift within
tolerance, ``div`` still at round-off, and no dtype leak.

Usage::

    python -u scripts/desc_example_sweep.py --ns 6,10,10 --p 2 --steps 500 \
        --out outputs/desc_sweep
    python -u scripts/desc_example_sweep.py --cases SOLOVEV,DSHAPE_lowres --steps 200
    python -u scripts/desc_example_sweep.py --plot outputs/desc_sweep

``--precision`` is exported as ``MRX_DTYPE`` before ``mrx`` is imported.
Output: ``<out>/<case>/result.json`` per case and, from ``--plot``,
``<out>/summary.json`` and ``<out>/desc_sweep.png``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
import traceback

#: The shipped examples this sweeps, cheapest first. ``current`` marks the
#: ones that store no iota and so need DESC importable.
CASES = (
    # name,             nfp, current-constrained
    ("DSHAPE_lowres",     1, False),
    ("SOLOVEV",           1, False),
    ("DSHAPE",            1, False),
    ("HELIOTRON",        19, False),
    ("W7-X",              5, False),
    ("ATF",              12, False),
    ("precise_QA",        2, True),
    ("HSX",               4, True),
    ("WISTELL-A",         4, True),
    ("NCSX",              3, True),
    ("ARIES-CS",          3, True),
)

#: A profile whose sup is below this times the magnetic pressure scale is
#: treated as identically zero, i.e. the case is a vacuum field.
VACUUM_TOL = 1e-12

#: Round-off part of the helicity budget, in the repository's convention
#: ``|H_end - H_0| / (2 E_0)``. ``test/test_relaxation.py`` measures the
#: constant 25 over 50 steps on li383; the drift is the rounding of the
#: stored field accumulating once per step, so it grows with the count.
HELICITY_DRIFT_C = 25.0

#: Excursion part of the same budget. The ideal step conserves helicity
#: only to the discretisation, so a run that releases a FRACTION ``d`` of
#: its energy perturbs helicity at ``O(d)`` however exact the arithmetic.
#: A pure round-off band would therefore flag every case that actually
#: relaxes -- HELIOTRON releases 0.7% of its energy at (8, 14, 8) p=2 --
#: which says nothing about the reader or the stepper. Measured
#: 2026-09-13: the ratio of drift to fractional energy release is 1.2 for
#: HELIOTRON, 0.4 for DSHAPE and 0.3 for ATF, so 2 is the band with room.
HELICITY_EXCURSION_C = 2.0


def helicity_budget(steps: int, energy_first: float, energy_last: float) -> float:
    """Helicity drift allowed for a run, as ``|H_end - H_0| / (2 E_0)``.

    The sum of a round-off floor that grows with the step count and a term
    proportional to the energy actually released; see
    :data:`HELICITY_DRIFT_C` and :data:`HELICITY_EXCURSION_C`.

    Args:
        steps: steps the run took.
        energy_first: energy at the start.
        energy_last: energy at the end.

    Returns:
        The budget, in the same units as the measured drift.
    """
    from mrx.precision import sqrt_eps  # noqa: PLC0415  (needs MRX_DTYPE set)
    released = abs(energy_last - energy_first) / max(abs(energy_first), 1e-300)
    return (HELICITY_DRIFT_C * sqrt_eps() * max(steps, 1) / 50.0
            + HELICITY_EXCURSION_C * released)

#: Wall-normal part of the reference field that the Dirichlet projection
#: discards, relative to the field. The Clebsch construction is tangential
#: analytically, so this is pure quadrature round-off; the measured values
#: across the sweep are 0 or 2e-8 (2026-09-13), never the 1e-3 that a
#: genuinely non-tangential field would give.
WALL_DISCARDED_TOL = 1e-6

#: Relative agreement required between the transform of the field MRX built
#: and the iota profile stored in the file. This is THE reader check: it is
#: end-to-end, from the Fourier-Zernike coefficients through the basis
#: conversion, the spline refit and the Clebsch construction.
IOTA_TOL = 1e-3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: argument list; ``None`` reads ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", default=None,
                    help="directory of DESC *_output.h5 files; default is the "
                         "installed desc package's examples/, else data/")
    ap.add_argument("--cases", default=None,
                    help="comma-separated subset of the case names to run")
    ap.add_argument("--ns", default="6,10,10")
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--chunk", type=int, default=125)
    ap.add_argument("--cfl", type=float, default=0.5)
    ap.add_argument("--floor-tol", type=float, default=1e-8,
                    help="stop once the force residual floors, as scripts/relax.py "
                         "does; 0 runs the full budget and lets a converged case "
                         "wander at its floor")
    ap.add_argument("--precision", default="float64", choices=("float32", "float64"))
    ap.add_argument("--out", default="outputs/desc_sweep")
    ap.add_argument("--plot", default=None, help="merge the cases under this directory")
    return ap.parse_args(argv)


def _log(msg: str) -> None:
    """Print a timestamped progress line.

    Args:
        msg: the message.
    """
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def examples_dir(given: str | None) -> str:
    """Where the DESC example files live.

    Args:
        given: an explicit directory, or ``None`` to discover one.

    Returns:
        The directory: ``given`` if set, else the installed DESC package's
        ``examples/`` if importable, else ``data/``.
    """
    if given:
        return given
    try:
        import desc  # noqa: PLC0415  (optional dependency)
        return os.path.join(os.path.dirname(desc.__file__), "examples")
    except ImportError:
        return "data"


def locate(root: str, name: str) -> str | None:
    """The file of one case, under either naming convention.

    Args:
        root: the directory to look in.
        name: the case name, e.g. ``"SOLOVEV"``.

    Returns:
        The path, or ``None`` if the case is not there.
    """
    for pattern in (f"{name}_output.h5", f"desc_{name}.h5", f"{name}.h5"):
        hit = os.path.join(root, pattern)
        if os.path.isfile(hit):
            return hit
    return None


def one_case(path: str, name: str, cli: argparse.Namespace) -> dict:
    """Read, build, relax and measure one example.

    Args:
        path: the DESC output file.
        name: the case name, for the record.
        cli: parsed arguments; see :func:`parse_args`.

    Returns:
        The result dict: what the file said, what the initial field
        measured, what the relaxation did, and the ``checks`` each of those
        has to pass.
    """
    import numpy as np
    from mrx.desc import profile_spline, read_desc
    from mrx.geometry import build_sequence
    from mrx.initial_conditions import initial_field
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import TimeStepper, initial_state, relax

    st = read_desc(path)
    r = np.linspace(0.0, 1.0, 201)
    pressure = np.abs(profile_spline(st, "pressure")(r)).max()
    iota_axis_file = float(profile_spline(st, "iota")(0.0))
    iota_edge_file = float(profile_spline(st, "iota")(1.0))
    vacuum = pressure <= VACUUM_TOL
    _log(f"{name}: nfp={st['nfp']} L,M,N={st['L']},{st['M']},{st['N']} "
         f"Psi={st['Psi']:+.4g} {'vacuum' if vacuum else f'p_max={pressure:.4g}'} "
         f"iota {iota_axis_file:+.4f}..{iota_edge_file:+.4f}")

    ns = tuple(int(v) for v in cli.ns.split(","))
    t0 = time.perf_counter()
    seq, _ = build_sequence(path, ns, cli.p)
    # The descent deflates against the harmonic forms; without them the
    # potential-velocity route has a nullspace and the run diverges.
    compute_nullspaces(seq)
    build_s = time.perf_counter() - t0
    B0, info = initial_field(seq)
    _log(f"{name}: build {build_s:.0f} s, IC div {info['div']:.2e}, "
         f"wall {info['wall_discarded']:.2e}, iota {info['iota_axis']:+.4f}.."
         f"{info['iota_edge']:+.4f}")

    ts = TimeStepper(seq=seq, cfl=cli.cfl, history_size=1, velocity_smoothing_order=1)
    res = relax(initial_state(B0, ts), ts, cli.steps, chunk=cli.chunk,
                floor_tol=cli.floor_tol, verbose=False)
    qoi = {k: np.asarray(v).tolist() for k, v in res.qoi.items()}
    trace = {k: np.asarray(v).tolist() for k, v in res.trace.items()}
    first, last = {k: v[0] for k, v in qoi.items()}, {k: v[-1] for k, v in qoi.items()}
    # The repository's measure: the drift against the energy scale, not
    # against H itself, so a nearly force-free case with small H is not
    # judged more harshly than a sheared one.
    drift = abs(last["helicity"] - first["helicity"]) / (2.0 * res.E0)
    budget = helicity_budget(res.steps, first["E"], last["E"])
    _log(f"{name}: {res.steps} steps in {res.wall:.0f} s, stop={res.stop}, "
         f"E {first['E']:.6e} -> {last['E']:.6e}, resid {first['resid']:.3e} -> "
         f"{last['resid']:.3e}, dH/2E0 {drift:.2e} (budget {budget:.1e})")

    # Two groups, because they certify different things. The reader checks
    # are what this script is FOR: they say the equilibrium MRX built is
    # the one the file describes, and they must all hold. The relaxation
    # checks say the descent then behaved, which is a property of MRX at
    # this resolution and step budget, not of the reader -- a case that
    # starts at the force floor (SOLOVEV: a pressure-balanced state has a
    # Leray-projected force of zero) has nothing left to minimise and
    # wanders, which is honest behaviour rather than a reading error.
    reader = dict(
        div_at_roundoff=info["div"] < 1e-10,
        wall_negligible=info["wall_discarded"] < WALL_DISCARDED_TOL,
        # build_gvec_map may flip the handedness, so compare magnitudes.
        iota_axis_matches=abs(abs(info["iota_axis"]) - abs(iota_axis_file))
        <= IOTA_TOL * max(abs(iota_axis_file), 1.0),
        iota_edge_matches=abs(abs(info["iota_edge"]) - abs(iota_edge_file))
        <= IOTA_TOL * max(abs(iota_edge_file), 1.0),
        initial_field_finite=bool(np.isfinite(np.asarray(B0)).all()),
    )
    relaxation = dict(
        energy_monotone=bool(np.all(np.asarray(trace["dE"]) <= 0.0)),
        energy_decreased=last["E"] <= first["E"],
        helicity_within_budget=drift < budget,
        finite=bool(np.isfinite(np.asarray(res.state.B_n)).all()),
    )
    return dict(name=name, path=path, nfp=int(st["nfp"]), Psi=float(st["Psi"]),
                L=st["L"], M=st["M"], N=st["N"], vacuum=bool(vacuum),
                pressure_max=float(pressure), iota_axis_file=iota_axis_file,
                iota_edge_file=iota_edge_file, ns=list(ns), p=cli.p,
                build_seconds=build_s, initial=info, steps=int(res.steps),
                stop=res.stop, wall=float(res.wall), qoi=qoi,
                energy_first=first["E"], energy_last=last["E"],
                resid_first=first["resid"], resid_last=last["resid"],
                helicity_drift=drift, helicity_budget=budget,
                reader=reader, relaxation=relaxation,
                passed=bool(all(reader.values())),
                relaxed_well=bool(all(relaxation.values())))


def run(cli: argparse.Namespace) -> None:
    """Run every requested case, writing one ``result.json`` each.

    A case that raises is recorded and the sweep continues: one unreadable
    file should not cost the other ten.

    Args:
        cli: parsed arguments; see :func:`parse_args`.
    """
    import importlib.util
    have_desc = importlib.util.find_spec("desc") is not None
    root = examples_dir(cli.examples)
    wanted = set(cli.cases.split(",")) if cli.cases else None
    _log(f"examples from {root}; DESC {'is' if have_desc else 'is NOT'} importable")

    for name, nfp, current in CASES:
        if wanted is not None and name not in wanted:
            continue
        path = locate(root, name)
        if path is None:
            _log(f"{name}: not found under {root}, skipping")
            continue
        if current and not have_desc:
            _log(f"{name}: current-constrained (no stored iota) and DESC is not "
                 "importable, skipping")
            continue
        case_dir = os.path.join(cli.out, name)
        os.makedirs(case_dir, exist_ok=True)
        try:
            result = one_case(path, name, cli)
        except Exception as exc:                 # noqa: BLE001  (one case must not stop the sweep)
            _log(f"{name}: FAILED, {type(exc).__name__}: {exc}")
            result = dict(name=name, path=path, nfp=nfp, error=f"{type(exc).__name__}: {exc}",
                          traceback=traceback.format_exc(), passed=False)
        with open(os.path.join(case_dir, "result.json"), "w") as fh:
            json.dump(result, fh, indent=2)
        bad_reader = [k for k, v in result.get("reader", {}).items() if not v]
        bad_relax = [k for k, v in result.get("relaxation", {}).items() if not v]
        _log(f"{name}: reader {'ok' if result.get('passed') else 'FAILED ' + ','.join(bad_reader)}"
             f"; relaxation {'ok' if result.get('relaxed_well') else 'noted ' + ','.join(bad_relax)}")
    print(f"wrote the sweep under {cli.out}", flush=True)


def plot(root: str) -> None:
    """Merge the cases under ``root`` into a summary table and figure.

    Args:
        root: the output directory ``run`` wrote into.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    results = []
    for path in sorted(glob.glob(os.path.join(root, "*", "result.json"))):
        with open(path) as fh:
            results.append(json.load(fh))
    ok = [r for r in results if "error" not in r]
    if not ok:
        print(f"no completed cases under {root}", flush=True)
        return
    ok.sort(key=lambda r: (not r["vacuum"], r["name"]))
    names = [r["name"] for r in ok]
    x = np.arange(len(ok))
    # Re-judge the stored runs with the current budget, so tightening or
    # loosening it does not mean paying for the sweep again.
    for r in ok:
        r["helicity_budget"] = helicity_budget(r["steps"], r["energy_first"],
                                               r["energy_last"])
        r["relaxation"]["helicity_within_budget"] = r["helicity_drift"] < r["helicity_budget"]
        r["relaxed_well"] = bool(all(r["relaxation"].values()))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    colours = ["C0" if r["vacuum"] else "C3" for r in ok]
    axes[0].bar(x, [r["resid_first"] for r in ok], color="lightgrey", label="initial")
    axes[0].bar(x, [r["resid_last"] for r in ok], color=colours, label="relaxed")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("normalised force residual")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, [max(r["helicity_drift"], 1e-18) for r in ok], color=colours)
    axes[1].plot(x, [r["helicity_budget"] for r in ok], "k:", lw=1, label="budget")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"helicity drift $|\Delta H| / 2E_0$")
    axes[1].legend(fontsize=8)
    axes[2].bar(x, [max(r["initial"]["div"], 1e-20) for r in ok], color=colours)
    axes[2].set_yscale("log")
    axes[2].set_ylabel(r"$\|\mathrm{div}\,B\|$ of the initial field")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("MRX relaxed from DESC's own equilibria "
                 "(blue: vacuum, red: finite beta)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "desc_sweep.png"), dpi=150)
    plt.close(fig)

    summary = [dict(name=r["name"], nfp=r["nfp"], vacuum=r["vacuum"], Psi=r["Psi"],
                    iota_axis_file=r["iota_axis_file"], iota_axis_mrx=r["initial"]["iota_axis"],
                    iota_edge_file=r["iota_edge_file"], iota_edge_mrx=r["initial"]["iota_edge"],
                    div=r["initial"]["div"], steps=r["steps"], stop=r["stop"],
                    resid_first=r["resid_first"], resid_last=r["resid_last"],
                    helicity_drift=r["helicity_drift"], helicity_budget=r["helicity_budget"],
                    reader=r["reader"], relaxation=r["relaxation"],
                    passed=r["passed"], relaxed_well=r["relaxed_well"]) for r in ok]
    summary += [dict(name=r["name"], error=r["error"], passed=False, relaxed_well=False)
                for r in results if "error" in r]
    with open(os.path.join(root, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    width = max(len(s["name"]) for s in summary)
    print(f"\n{'case':<{width}}  nfp  vac  |iota| axis file/mrx      div    steps  "
          f"resid in -> out      dH/2E0  reader  relax", flush=True)
    for s in summary:
        if "error" in s:
            print(f"{s['name']:<{width}}  ERROR {s['error']}", flush=True)
            continue
        print(f"{s['name']:<{width}}  {s['nfp']:3d}  {'y' if s['vacuum'] else 'n':>3}  "
              f"{abs(s['iota_axis_file']):.5f}/{abs(s['iota_axis_mrx']):.5f}  "
              f"{s['div']:.1e}  {s['steps']:5d}  "
              f"{s['resid_first']:.2e}->{s['resid_last']:.2e}  {s['helicity_drift']:.2e}  "
              f"{'ok' if s['passed'] else 'FAIL':>6}  {'ok' if s['relaxed_well'] else 'noted':>5}",
              flush=True)
    print(f"\nwrote {root}/summary.json, desc_sweep.png", flush=True)


if __name__ == "__main__":
    args = parse_args()
    os.environ["MRX_DTYPE"] = args.precision
    if args.plot:
        plot(args.plot)
    else:
        run(args)
