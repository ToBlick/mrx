#!/usr/bin/env python3
"""Evaluate the toroidal boundary circulation of a saved harmonic nullspace DOF.

Rebuild the geometry and the de Rham sequence from the meta JSON that
accompanies a harmonic nullspace DOF file (the same rebuild the plotting
scripts use), load the DOF vector, and report ``Γ_MRX`` on the standard
loop ``(ρ=ρ_b, θ=0, ζ∈[0,1))`` via
:func:`mrx.circulation.boundary_toroidal_circulation`. The helpers come
from ``scripts/plotting/harmonic_nullspace_geometry``; the repository root
is put on ``sys.path`` for that import.

Arguments:
    --dof-npy (Path): Saved nullspace DOF vector. Required.
    --meta-json (Path): Meta JSON of the run. Default ``None`` selects the
        sibling ``hodge_k2_nullspace_meta.json`` (k=2) or
        ``hcurl_nullspace_meta.json`` (k=1) of the DOF file.
    --k (int): Form degree, 1 or 2. Default ``None`` infers 2 from a
        ``k2``/``hodge_k2`` filename and 1 otherwise.
    --n-quad (int): Quadrature points on the loop. Default 256.
    --rho-boundary (float): Loop radius ``ρ_b``. Default ``None`` selects
        0.6 for ``native`` and ``1 - 1e-10`` for ``extend_map``.
    --circulation-map (str): ``native`` or ``extend_map``. Default
        ``native``.
    --b-field (str): k=1 only: ``pushforward`` evaluates ``u``, ``curl``
        evaluates ``curl u``. Default ``pushforward``.
    -o, --output-json (Path): Write the result JSON here as well. Default
        ``None``.

Usage:
    ::

        python -u scripts/harmonic/eval_boundary_circulation.py \
            --dof-npy script_outputs/hodge_k2_quasr0009983_ns8_16_8_p3/hodge_k2_dbc_nullspace_dof.npy \
            --k 2 -o circulation.json

    Single GPU job through ``slurm/run.sh``::

        SCRIPT=scripts/harmonic/eval_boundary_circulation.py \
            ARGS="--dof-npy <dof.npy> --k 2 -o circulation.json" \
            JOB_NAME=circ bash slurm/run.sh

Runtime:
    Not measured.

Output:
    The result dict (circulation, loop parameters, ``dof_npy``,
    ``map_mode``, ``nfp_meta``) is printed as JSON to stdout and, with
    ``-o``, written to that file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mrx.circulation import boundary_toroidal_circulation, default_circulation_rho
from scripts.plotting.harmonic_nullspace_geometry import (
    infer_form_degree,
    load_dof_vector,
    load_meta,
    rebuild_sequence_from_meta,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dof-npy", type=Path, required=True)
    ap.add_argument("--meta-json", type=Path, default=None)
    ap.add_argument("--k", type=int, choices=(1, 2), default=None)
    ap.add_argument("--n-quad", type=int, default=256)
    ap.add_argument(
        "--rho-boundary",
        type=float,
        default=None,
        help="Loop radius ρ_b (default: 0.6 native, ~1 for extend_map).",
    )
    ap.add_argument(
        "--circulation-map",
        choices=("native", "extend_map"),
        default="native",
    )
    ap.add_argument(
        "--b-field",
        choices=("pushforward", "curl"),
        default="pushforward",
        help="For k=1 only: Pushforward(u) or Pushforward(curl u).",
    )
    ap.add_argument("-o", "--output-json", type=Path, default=None)
    args = ap.parse_args()

    dof_path = args.dof_npy.expanduser().resolve()
    k = infer_form_degree(dof_path, k=args.k)
    meta = load_meta(args.meta_json, dof_path, k=k)
    seq, _, map_raw, nfp = rebuild_sequence_from_meta(meta)
    u = load_dof_vector(dof_path, seq, k=k)

    rho = args.rho_boundary
    if rho is None:
        rho = default_circulation_rho(circulation_map=args.circulation_map)

    out = boundary_toroidal_circulation(
        seq,
        u,
        map_raw,
        k=k,
        n_quad=int(args.n_quad),
        rho_boundary=float(rho),
        circulation_map=str(args.circulation_map),
        nfp=int(nfp),
        b_field=str(args.b_field),
    )
    out["dof_npy"] = str(dof_path)
    out["map_mode"] = meta.get("map_mode")
    out["nfp_meta"] = int(nfp)

    print(json.dumps(out, indent=2))
    if args.output_json is not None:
        args.output_json.expanduser().resolve().write_text(json.dumps(out, indent=2))
        print(f"Wrote {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
