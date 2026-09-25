"""Run several scripts of this directory one after the other in ONE job (slurm/run.sh runs one command).

    python -u landreman_chain.py landreman_wout.py --case X --out W :: relax.py --geometry W ... :: ...

Each ``::``-separated group is ``python -u <group>`` (a sibling of this file, else the repository's scripts/ production tool); the chain stops at the first failure.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
#: the production tools (relax.py, poincare_trace.py) when this file is not next to them
TOOLS = os.path.join(os.path.dirname(HERE), "scripts")


def resolve(name):
    """A sibling of this file, else the production tool of that name."""
    own = os.path.join(HERE, name)
    return own if os.path.isfile(own) else os.path.join(TOOLS, name)

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
        subprocess.run([sys.executable, "-u", resolve(g[0]), *g[1:]], check=True)
