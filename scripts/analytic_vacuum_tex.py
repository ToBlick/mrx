"""Emit LaTeX convergence tables from ``analytic_vacuum.json`` files.

Pure stdlib + numpy (no mrx, no GPU): safe to run on the login node.

    python scripts/analytic_vacuum_tex.py <pscan_dir> [--out tables.tex]

Scans ``<pscan_dir>/**/analytic_vacuum.json``, groups the records by route
(A/C) and degree p, and writes one booktabs table per route -- rows = element
count / mesh size ``h = 1/n_el``, columns = p, cells = relative M-error -- with
a fitted convergence order per column. Missing (still-running) cells are dashed.
Also prints the tables to stdout for copy-paste. Standalone: needs only
``\\usepackage{booktabs}`` (the scientific cells are plain math, no siunitx).
"""

import argparse
import glob
import json
import os

import numpy as np

ROUTE_TITLE = {"A": r"Route A: $H=\nabla f+\alpha h_1$ (1-form, $k{=}1$ free)",
               "C": r"Route C: $B=\operatorname{curl}A$ (2-form, $k{=}2$ free)"}


def collect(root):
    data, meta = {}, {}
    for jf in sorted(glob.glob(os.path.join(root, "**", "analytic_vacuum.json"),
                               recursive=True)):
        with open(jf) as fh:
            d = json.load(fh)
        meta.setdefault("geometry", os.path.basename(d.get("geometry", "")))
        meta.setdefault("field", d.get("field"))
        meta.setdefault("lam", d.get("lam"))
        for rec in d["records"]:
            ne, p = rec["n_elements"], rec["p"]
            for route, r in rec.get("routes", {}).items():
                if r.get("relerr") is not None:
                    data[(route, p, ne)] = (rec["h"], float(r["relerr"]))
    return data, meta


def sci(x):
    if x is None or not np.isfinite(x):
        return r"\textemdash"
    m, e = f"{x:.1e}".split("e")
    return fr"${m}\!\times\!10^{{{int(e)}}}$"


def fit_order(pairs):
    if len(pairs) < 2:
        return float("nan")
    h, e = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    m = np.isfinite(e) & (e > 0)
    return float(np.polyfit(np.log(h[m]), np.log(e[m]), 1)[0]) if m.sum() >= 2 else float("nan")


def table(data, route, ps, nels):
    L = [r"\begin{tabular}{rr" + "c" * len(ps) + "}", r"\toprule",
         r"$n_{\mathrm{el}}$ & $h$ & "
         + " & ".join(fr"$p{{=}}{p}$" for p in ps) + r" \\", r"\midrule"]
    for ne in nels:
        cells = [sci(data[(route, p, ne)][1]) if (route, p, ne) in data
                 else r"\textemdash" for p in ps]
        L.append(f"{ne} & {1.0 / ne:.4f} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    orders = [fit_order([data[(route, p, ne)] for ne in nels if (route, p, ne) in data])
              for p in ps]
    L.append(r"order & & " + " & ".join(f"{o:.2f}" if np.isfinite(o) else r"\textemdash"
                                         for o in orders) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def wrap(inner, route, meta):
    cap = (fr"Convergence of the discrete vacuum field to the analytic "
           fr"$B^\ast = e_\phi/R + \lambda\,\nabla(R^2\cos2\phi)$ "
           fr"($\lambda={meta.get('lam')}$) on the {meta.get('geometry', 'QA')} "
           fr"geometry. Relative $M$-error vs.\ mesh size; fitted order per $p$. "
           + ROUTE_TITLE.get(route, route) + ".")
    return ("\n".join([r"\begin{table}[t]", r"\centering", inner,
                       fr"\caption{{{cap}}}", fr"\label{{tab:analytic-vacuum-{route}}}",
                       r"\end{table}"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()

    data, meta = collect(cli.root)
    ps = sorted({p for _, p, _ in data})
    nels = sorted({ne for _, _, ne in data})
    routes = [r for r in ("A", "C") if any(k[0] == r for k in data)]
    blocks = [wrap(table(data, r, ps, nels), r, meta) for r in routes]
    out = ("% analytic_vacuum convergence tables -- needs \\usepackage{booktabs}\n"
           "% regenerate: python scripts/analytic_vacuum_tex.py " + cli.root + "\n\n"
           + "\n\n".join(blocks) + "\n")

    print(out)
    if cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
        with open(cli.out, "w") as fh:
            fh.write(out)
        print(f"% wrote {cli.out}")


if __name__ == "__main__":
    main()
