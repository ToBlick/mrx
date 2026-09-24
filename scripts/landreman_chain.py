"""Run several scripts of this directory one after the other in ONE job (slurm/run.sh runs one command).

    python -u scripts/landreman_chain.py landreman_wout.py --case X --out W :: relax.py --geometry W ... :: ...

Each ``::``-separated group is ``python -u scripts/<group>``; the chain stops at the first failure.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    groups, cur = [], []
    for a in sys.argv[1:]:
        if a == "::":
            groups.append(cur)
            cur = []
        else:
            cur.append(a)
    groups.append(cur)
    for g in groups:
        print(f"[chain] {' '.join(g)}", flush=True)
        subprocess.run([sys.executable, "-u", os.path.join(HERE, g[0]), *g[1:]], check=True)
