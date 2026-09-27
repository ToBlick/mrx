#!/usr/bin/env python
"""Seed island chains into a converged state by the energy criterion (one GPU; Sec. 6.2, Tab. 4).

    python scripts/paper_scripts/seed.py --run RUN --step N --out SEEDED.h5 [--chain M,N ...] [--scale Q]

:func:`mrx.seeding.energy_seed` on checkpoints/state_<step>.h5 of the run: every resonance in the field's iota
range (or the --chain ones) gets SIESTA's parallel seed at the amplitudes of least energy, and B + Q sum a*_i
dB_i is written as a checkpoint for relax.py --restart. Next to it, <out>.json lists every seeded resonance:
r_mn, |d_r iota|, the resonant normal field dBr of its amplitude and the pendulum width
w = sqrt(8 dBr nfp / (pi m |d_r iota|)).
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="a scripts/relax.py run: its relax.json and checkpoints/")
    ap.add_argument("--step", type=int, required=True, help="seed checkpoints/state_<step>.h5")
    ap.add_argument("--out", required=True, help="the seeded checkpoint (.h5); the table goes to the .json beside it")
    ap.add_argument("--chain", action="append", default=None,
                    help="M,N: seed only this chain (repeatable) [every chain in range]")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the added perturbation [1]")
    cli = ap.parse_args()
    os.environ["MRX_DTYPE"] = os.environ["MRX_RESIDUAL_DTYPE"] = "float64"

    import mrx
    from mrx.geometry import build_sequence
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import TimeStepper, initial_state, read_checkpoint, write_checkpoint
    from mrx.seeding import energy_seed

    params = json.load(open(os.path.join(cli.run, "relax.json")))["params"]
    ns, p = tuple(params["ns"]), params["p"]
    mrx.MAP_BATCH_SIZE_INNER = params["map_batch"]
    seq, _ = build_sequence(params["geometry_path"], ns, p, params["solve_maxiter"], tol=params["solve_tol"],
                            nfp=params["nfp"], knots=params["knots"], symmetry=params["symmetry"])
    compute_nullspaces(seq)
    ts = TimeStepper(seq=seq)
    ckpt = os.path.join(cli.run, "checkpoints", f"state_{cli.step:06d}.h5")
    state, step = read_checkpoint(ckpt, ts)
    print(f"[seed] {ckpt}: step {step}", flush=True)
    chains = None if cli.chain is None else [tuple(int(v) for v in mn.split(",")) for mn in cli.chain]
    iotas = None if chains is None else [seq.nfp * n / m for m, n in chains]
    seeded, rows = energy_seed(seq, state.B_n, iotas=iotas, scale=cli.scale)
    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    write_checkpoint(cli.out, initial_state(seeded, ts, step=step), step, seq)
    with open(os.path.splitext(cli.out)[0] + ".json", "w") as fh:
        json.dump(dict(checkpoint=ckpt, step=step, ns=list(ns), p=p, nfp=seq.nfp, chains=chains, scale=cli.scale,
                       resonances=rows), fh, indent=1)
    print(f"[seed] wrote {cli.out} and its .json", flush=True)


if __name__ == "__main__":
    main()
