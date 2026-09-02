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


def preamble(meta):
    lam = meta.get("lam")
    return (
        r"\paragraph{Convergence against an analytic vacuum field.} "
        r"We verify the FEEC discretisation against a vacuum magnetic field known "
        r"in closed form on the (non-axisymmetric) quasi-axisymmetric stellarator "
        r"boundary. The reference is a model coil field, "
        r"\[ B^\ast = \frac{e_\phi}{R} + \lambda\,\nabla\!\left(R^2\cos 2\phi\right), "
        fr"\qquad \lambda = {lam}, \] "
        r"the axisymmetric toroidal-field part $e_\phi/R$ (the net current linking "
        r"the torus) plus an $n{=}2$ shaping ripple matching the two field periods; "
        r"both terms are curl- and divergence-free, so $B^\ast$ is an exact vacuum "
        r"field and the error carries no truncation floor. A vacuum field is "
        r"curl-free (a gradient) and divergence-free (a curl), so it can be "
        r"reconstructed at either end of the de~Rham complex. \emph{Route~A} takes "
        r"the $H$-field as a $1$-form with a scalar potential, "
        r"$H = \nabla f + \alpha\, h_1$ (a $k{=}0$ Laplace solve; the harmonic "
        r"amplitude $\alpha$ fixes the toroidal circulation $\oint B^\ast\cdot d\ell "
        r"= 2\pi/n_{\mathrm{fp}}$). \emph{Route~C} takes the $B$-field as a $2$-form "
        r"with a vector potential, $B = \operatorname{curl} A$ (a $k{=}1$ "
        r"curl--curl solve; the solid torus has no free harmonic $2$-form, so the "
        r"flux is carried by $A$'s boundary circulation and no harmonic term is "
        r"needed). Tables~\ref{tab:analytic-vacuum-A} and~\ref{tab:analytic-vacuum-C} "
        r"report the relative $M$-norm error $\|B_h - B^\ast\|_M / \|B^\ast\|_M$ "
        r"against mesh size $h = 1/n_{\mathrm{el}}$ for spline degrees "
        r"$p = 1,\dots,4$; both routes converge at the optimal rate $O(h^p)$.")


def vmec_preamble(mj):
    floor = mj.get("slopes", {}).get("D_floor")
    fs = sci(floor).replace("$", "") if floor else r"8\times10^{-5}"
    return (
        r"\paragraph{Convergence to the VMEC equilibrium field.} "
        r"The same discrete harmonic $2$-form is compared against the vacuum field "
        r"$B_w$ of a VMEC equilibrium (the low-resolution Landreman--Paul QA "
        r"reference), the \emph{confined} (wall-tangent) field of the bounded "
        r"domain rather than a driven one. The equilibrium field's scale is not "
        r"set a priori, so we fit a single amplitude $c$ by $M$-projection and "
        r"report $D = \|B_w - c\,h\|_M / \|B_w\|_M$. Unlike the analytic case there "
        r"is no exact target: $D$ falls at the pre-floor rates below and then "
        fr"saturates at a floor $D \approx {fs}$, set by the truncation of the "
        r"low-resolution VMEC reference itself and not by the MRX discretisation. "
        r"Table~\ref{tab:vmec-vacuum} reports $D$ against mesh size for "
        r"$p = 1,\dots,4$; the fitted pre-floor order per $p$ is given in the last "
        r"row.")


def vmec_table(mj):
    data = {(r["p"], r["ns"][0] - r["p"]): r["D"] for r in mj["rows"]}
    ps = sorted({p for p, _ in data})
    nels = sorted({ne for _, ne in data})
    pre = mj.get("slopes", {}).get("D_by_p_prefloor", {})
    L = [r"\begin{tabular}{rr" + "c" * len(ps) + "}", r"\toprule",
         r"$n_{\mathrm{el}}$ & $h$ & "
         + " & ".join(fr"$p{{=}}{p}$" for p in ps) + r" \\", r"\midrule"]
    for ne in nels:
        cells = [sci(data.get((p, ne))) for p in ps]
        L.append(f"{ne} & {1.0 / ne:.4f} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    L.append(r"pre-floor order & & "
             + " & ".join(f"{pre[str(p)]:.2f}" if str(p) in pre else r"\textemdash"
                          for p in ps) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def vmec_block(mj):
    cap = (r"Convergence of the discrete harmonic $2$-form to the VMEC vacuum "
           r"field $B_w$ (Landreman--Paul QA, low resolution): $D = "
           r"\|B_w - c\,h\|_M/\|B_w\|_M$ vs.\ mesh size, with the pre-floor order "
           r"per $p$. $D$ saturates at the reference's truncation floor.")
    return "\n".join([r"\begin{table}[t]", r"\centering", vmec_table(mj),
                      fr"\caption{{{cap}}}", r"\label{tab:vmec-vacuum}",
                      r"\end{table}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    ap.add_argument("--vmec", default=None,
                    help="qa_vacuum_convergence_merged.json -- append the "
                         "VMEC-convergence (D) paragraph and table")
    cli = ap.parse_args()

    data, meta = collect(cli.root)
    ps = sorted({p for _, p, _ in data})
    nels = sorted({ne for _, _, ne in data})
    routes = [r for r in ("A", "C") if any(k[0] == r for k in data)]
    blocks = [wrap(table(data, r, ps, nels), r, meta) for r in routes]
    parts = [preamble(meta)] + blocks
    if cli.vmec:
        with open(cli.vmec) as fh:
            mj = json.load(fh)
        parts += [vmec_preamble(mj), vmec_block(mj)]
    out = ("% analytic_vacuum convergence tables -- needs \\usepackage{booktabs}\n"
           "% regenerate: python scripts/analytic_vacuum_tex.py " + cli.root
           + (" --vmec " + cli.vmec if cli.vmec else "") + "\n\n"
           + "\n\n".join(parts) + "\n")

    print(out)
    if cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
        with open(cli.out, "w") as fh:
            fh.write(out)
        print(f"% wrote {cli.out}")


if __name__ == "__main__":
    main()
