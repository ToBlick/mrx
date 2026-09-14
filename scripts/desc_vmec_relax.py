"""Relax MRX from a VMEC and from a DESC initial condition, and compare.

One sequence, one :class:`~mrx.relaxation.TimeStepper`, two initial fields
built by the production route from the two readings of the SAME
equilibrium (``scripts/desc_fixtures.py`` writes the DESC refit of the wout
this runs on). Each is relaxed for the same budget and the converged states
are held against each other.

What this can and cannot assert, because the difference matters.

MRX's relaxation is a HELICITY-PRESERVING energy minimisation. Its fixed
point is therefore a function of the initial helicity, and two initial
fields that differ at all carry slightly different helicity and need not
converge to the same state. Demanding that they agree to round-off would be
demanding the wrong thing, and a test written that way would either be
vacuous or fail for a correct code. What is actually meaningful is that
relaxation does not AMPLIFY the discrepancy:

* both traces decrease in energy monotonically and reach the same force
  residual floor, to the measured band
* helicity is conserved along each run within the drift tolerance
* ``||B_f_d - B_f_v||_M / ||B_f_v||_M`` is no larger than the same ratio at
  step zero, and is commensurate with the relative helicity difference the
  two initial conditions started with -- the discrepancy is carried, not
  grown
* the converged energy, ``beta_vol``, ``J.B / |J||B|`` and the pressure
  diagnostics agree between the two

The DESC state is turned to the wout's poloidal orientation first
(:func:`mrx.desc.match_orientation`): the sequence's map comes from the
wout, so the two Clebsch datasets have to be speaking the same angles or
the DESC field is built on the wrong chart. This is not cosmetic -- without
it the DESC field is a mirrored equilibrium and every number below is
meaningless.

Usage::

    python -u scripts/desc_vmec_relax.py --wout data/wout_li383_low_res_reference.nc \
        --desc data/desc_li383_lowres.h5 --ns 8,12,12 --p 2 --steps 2000 \
        --out outputs/desc_relax_li383
    python -u scripts/desc_vmec_relax.py --plot outputs/desc_relax_li383

``--precision`` is exported as ``MRX_DTYPE`` before ``mrx`` is imported.
Output: ``<out>/result.json`` (both runs' traces, samples and the
comparison) and, from ``--plot``, ``<out>/desc_vmec_relax.png``.
"""
from __future__ import annotations

import argparse
import json
import os
import time


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: argument list; ``None`` reads ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wout", default="data/wout_li383_low_res_reference.nc")
    ap.add_argument("--desc", default="data/desc_li383_lowres.h5",
                    help="the DESC reading of the same equilibrium")
    ap.add_argument("--ns", default="8,12,12")
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--history", type=int, default=1)
    ap.add_argument("--cfl", type=float, default=0.5)
    ap.add_argument("--precision", default="float64", choices=("float32", "float64"))
    ap.add_argument("--out", default="outputs/desc_vmec_relax")
    ap.add_argument("--plot", default=None, help="plot the result.json under this directory")
    return ap.parse_args(argv)


def _log(msg: str) -> None:
    """Print a timestamped progress line.

    Args:
        msg: the message.
    """
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def relax_from(eq: dict, seq, ts, cli: argparse.Namespace, tag: str) -> tuple:
    """Build the initial field of ``eq`` and relax it.

    Args:
        eq: the parsed equilibrium dict to start from.
        seq: the shared sequence; its ``equilibrium`` is swapped to ``eq``.
        ts: the shared time stepper.
        cli: parsed arguments, for ``steps`` and ``chunk``.
        tag: ``"desc"`` or ``"vmec"``, for the log lines.

    Returns:
        ``(B0, res, record)``: the initial field, the
        :class:`~mrx.relaxation.RelaxResult` and the JSON-ready dict of
        what the run measured.
    """
    import numpy as np
    from mrx.initial_conditions import initial_field
    from mrx.relaxation import initial_state, relax

    seq.equilibrium = eq
    t0 = time.perf_counter()
    B0, info = initial_field(seq)
    _log(f"{tag}: IC in {time.perf_counter() - t0:.0f} s, div {info['div']:.2e}, "
         f"iota {info['iota_axis']:+.4f}..{info['iota_edge']:+.4f}")
    res = relax(initial_state(B0, ts), ts, cli.steps, chunk=cli.chunk, verbose=False)
    qoi = {k: np.asarray(v).tolist() for k, v in res.qoi.items()}
    trace = {k: np.asarray(v).tolist() for k, v in res.trace.items()}
    first, last = {k: v[0] for k, v in qoi.items()}, {k: v[-1] for k, v in qoi.items()}
    drift = abs(last["helicity"] - first["helicity"]) / max(abs(first["helicity"]), 1e-300)
    _log(f"{tag}: {res.steps} steps in {res.wall:.0f} s, stop={res.stop}, "
         f"E {first['E']:.8e} -> {last['E']:.8e}, resid {first['resid']:.3e} -> "
         f"{last['resid']:.3e}, helicity drift {drift:.2e}")
    record = dict(tag=tag, initial=info, steps=int(res.steps), stop=res.stop,
                  wall=float(res.wall), E0=float(res.E0), qoi=qoi,
                  energy_first=first["E"], energy_last=last["E"],
                  resid_first=first["resid"], resid_last=last["resid"],
                  helicity_first=first["helicity"], helicity_last=last["helicity"],
                  helicity_drift=drift,
                  energy_monotone=bool(np.all(np.asarray(trace["dE"]) <= 0.0)),
                  div_last=last.get("div", float("nan")))
    return B0, res, record


def run(cli: argparse.Namespace) -> dict:
    """Relax from both initial conditions and write ``result.json``.

    Args:
        cli: parsed arguments; see :func:`parse_args`.

    Returns:
        The result dict that was written.
    """
    from mrx.desc import match_orientation
    from mrx.geometry import build_sequence
    from mrx.gvec import read_equilibrium
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import TimeStepper
    from mrx.vmec import read_wout

    ns = tuple(int(v) for v in cli.ns.split(","))
    t0 = time.perf_counter()
    seq, _ = build_sequence(cli.wout, ns, cli.p)
    # The descent deflates against the harmonic forms; without them the
    # potential-velocity route has a nullspace and the run diverges.
    compute_nullspaces(seq)
    _log(f"sequence {ns} p={cli.p} in {time.perf_counter() - t0:.0f} s")
    ts = TimeStepper(seq=seq, cfl=cli.cfl, history_size=cli.history,
                     velocity_smoothing_order=1)

    eq_v = read_equilibrium(cli.wout)
    eq_d, sign = match_orientation(read_equilibrium(cli.desc), read_wout(cli.wout))
    _log(f"DESC orientation relative to the wout: {sign:+d}")

    B0_v, res_v, rec_v = relax_from(eq_v, seq, ts, cli, "vmec")
    B0_d, res_d, rec_d = relax_from(eq_d, seq, ts, cli, "desc")

    def rel(a, b):
        return float(seq.l2_norm(a - b, 2) / seq.l2_norm(b, 2))

    initial_diff = rel(B0_d, B0_v)
    final_diff = rel(res_d.state.B_n, res_v.state.B_n)
    h_v, h_d = rec_v["helicity_first"], rec_d["helicity_first"]
    helicity_diff = abs(h_d - h_v) / max(abs(h_v), 1e-300)
    _log(f"||B_d - B_v||_M / ||B_v||_M: initial {initial_diff:.4e} -> "
         f"final {final_diff:.4e}   (relative helicity difference {helicity_diff:.4e})")

    comparison = dict(
        initial_B_rel_diff=initial_diff, final_B_rel_diff=final_diff,
        amplification=final_diff / max(initial_diff, 1e-300),
        initial_helicity_rel_diff=helicity_diff,
        energy_rel_diff=abs(rec_d["energy_last"] - rec_v["energy_last"])
        / max(abs(rec_v["energy_last"]), 1e-300),
        orientation=int(sign))
    for key in ("beta_vol", "JoverB", "JB", "resid"):
        a, b = rec_d["qoi"].get(key), rec_v["qoi"].get(key)
        if a is not None and b is not None:
            comparison[f"{key}_desc"], comparison[f"{key}_vmec"] = a[-1], b[-1]
            comparison[f"{key}_rel_diff"] = abs(a[-1] - b[-1]) / max(abs(b[-1]), 1e-300)
    result = dict(wout=cli.wout, desc=cli.desc, ns=list(ns), p=cli.p,
                  steps=cli.steps, precision=cli.precision,
                  vmec=rec_v, desc_run=rec_d, comparison=comparison)
    os.makedirs(cli.out, exist_ok=True)
    with open(os.path.join(cli.out, "result.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {cli.out}/result.json", flush=True)
    return result


def plot(root: str) -> None:
    """Draw the two traces and their difference.

    Args:
        root: the directory holding ``result.json``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with open(os.path.join(root, "result.json")) as fh:
        result = json.load(fh)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for rec, colour in ((result["vmec"], "C0"), (result["desc_run"], "C3")):
        it = np.asarray(rec["qoi"]["it"])
        axes[0].plot(it, np.asarray(rec["qoi"]["E"]), color=colour, label=rec["tag"])
        axes[1].semilogy(it, np.asarray(rec["qoi"]["resid"]), color=colour, label=rec["tag"])
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
    fig.savefig(os.path.join(root, "desc_vmec_relax.png"), dpi=150)
    plt.close(fig)
    print(f"wrote {root}/desc_vmec_relax.png", flush=True)


if __name__ == "__main__":
    args = parse_args()
    os.environ["MRX_DTYPE"] = args.precision
    if args.plot:
        plot(args.plot)
    else:
        run(args)
