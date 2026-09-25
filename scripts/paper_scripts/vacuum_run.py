#!/usr/bin/env python
"""A stored vacuum field as a run of zero steps, so that scripts/poincare_trace.py traces it (Fig. 6).

    python scripts/paper_scripts/vacuum_run.py RUNG

RUNG is a scripts/paper_scripts/qa_vacuum_sweep.py rung (result.json, fields.npz). Writes RUNG/run/relax.json and
RUNG/run/checkpoints/state_000000.h5 with the rung's discrete harmonic 2-form h_dof as the field, in the layout
scripts/relax.py writes. No GPU.
"""
import json
import os
import sys

import h5py
import numpy as np

rung = sys.argv[1]
result = json.load(open(os.path.join(rung, "result.json")))
run = os.path.join(rung, "run")
os.makedirs(os.path.join(run, "checkpoints"), exist_ok=True)
with h5py.File(os.path.join(run, "checkpoints", "state_000000.h5"), "w") as fh:
    fh["B_n"] = np.load(os.path.join(rung, "fields.npz"))["h_dof"]
    fh["p"] = np.zeros(1)       # a vacuum has no pressure: the trace computes its own weak pressure, the page drops it
    fh.attrs["step"] = 0
# the sequence of the rung: vacuum_convergence.py builds it with build_sequence's defaults
params = dict(geometry_path=os.path.abspath(result["geometry"]), ns=result["ns"], p=result["p"], nfp=None,
              symmetry="stellarator", knots=[None, None, None], precision=result["precision"], steps=0, ic="vacuum",
              auxiliary_B_field=False)
with open(os.path.join(run, "relax.json"), "w") as fh:
    json.dump(dict(params=params, reconnect=[]), fh, indent=1)
print(f"wrote {run}: h_dof of {rung} as the field of step 0")
