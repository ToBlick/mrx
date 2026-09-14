"""Generate the tracked DESC fixtures in ``data/``.

Run once, in an environment that has DESC (it is the only script here that
needs it); the files it writes are committed so the test suite and the
comparison scripts run with no DESC dependency at all:

    python scripts/desc_fixtures.py

Two kinds of fixture, for two different jobs.

**Refits of MRX's own wout files** (``desc_li383_lowres.h5``,
``desc_QA_lowres.h5``). ``VMECIO.load`` fits a Fourier-Zernike series to
the same wout that ``mrx.vmec`` refits into splines, so the DESC and VMEC
states describe the SAME configuration -- identical boundary, identical
profiles -- and every difference MRX then measures is representation and
nothing else. This is what ``scripts/desc_vmec_grid.py`` and
``scripts/desc_vmec_relax.py`` compare, and it is why DESC's own
``NCSX_output.h5`` will not do: that is a different NCSX from the tracked
li383 wout (``Psi = 0.497`` against the wout's ``phi_edge = 0.514``).

``profile="iota"`` matters. Loaded with the default current constraint the
file would store no iota at all and could only be read back with DESC
installed (see :mod:`mrx.desc`); with an iota constraint it is pure
``h5py``. Both a FIT-ONLY and a SOLVED variant are written, so the
representation error and the solver difference can be told apart: the
fit-only file is VMEC's solution in DESC's basis, the solved file is
DESC's own equilibrium for the same boundary.

**Trimmed DESC examples** (``desc_SOLOVEV.h5``, ``desc_DSHAPE_lowres.h5``).
A shipped example carries its whole continuation family; only the last
member is the converged solution, so :func:`trim` copies that one out and
drops the rest. These are the cheap reader fixtures: small, iota-
constrained, and not written by us, so they catch anything our synthetic
writer and our parser happen to agree on wrongly.

The DESC examples are MIT-licensed, Copyright (c) 2020 Daniel Dudt, Rory
Conlin, Dario Panici, Egemen Kolemen; ``data/DESC_LICENSE`` carries the
notice that redistribution requires.
"""
from __future__ import annotations

import argparse
import os

#: Fit resolution of the wout refits, ``(L, M, N)``. Chosen to resolve the
#: tracked low-resolution wouts without inflating the fixtures: li383 is
#: ``mpol = 7, ntor = 4``, QA is ``mpol = 6, ntor = 5``.
FIT_LMN = (8, 8, 5)

#: The wouts refit into DESC, and the fixture stem each becomes.
WOUTS = (("data/wout_li383_low_res_reference.nc", "desc_li383_lowres"),
         ("data/wout_LandremanPaul2021_QA_lowres.nc", "desc_QA_lowres"))

#: Shipped DESC examples copied in trimmed, as ``desc_<name>.h5``.
EXAMPLES = ("SOLOVEV", "DSHAPE_lowres")

LICENSE_NOTICE = """\
The files data/desc_SOLOVEV.h5 and data/desc_DSHAPE_lowres.h5 are trimmed
copies of the example equilibria shipped with DESC (desc/examples/), used
as read-only test fixtures by test/test_desc.py.

DESC is distributed under the MIT License:

MIT License

Copyright (c) 2020 Daniel Dudt, Rory Conlin, Dario Panici, Egemen Kolemen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def trim(src: str, dst: str) -> None:
    """Copy a DESC output keeping only the last equilibrium of its family.

    The continuation family is the expensive part of a shipped example and
    none of it is read: :func:`mrx.desc.read_desc` takes the last member.

    Args:
        src: the DESC output to read.
        dst: the trimmed file to write (overwritten).
    """
    import h5py  # noqa: PLC0415

    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        for key in fin:
            if key != "_equilibria":
                fin.copy(key, fout)
        family = fin["_equilibria"]
        steps = sorted((k for k in family if k.isdigit()), key=int)
        out = fout.create_group("_equilibria")
        for key in family:
            if not key.isdigit():
                family.copy(key, out)
        family.copy(steps[-1], out, name="0")


def refit(wout: str, stem: str, out_dir: str, solve: bool) -> str:
    """Refit a VMEC wout into DESC and write it as a fixture.

    Args:
        wout: the wout file to fit.
        stem: fixture stem; ``_solved`` is appended when ``solve``.
        out_dir: directory to write into.
        solve: run ``eq.solve()`` after the fit, giving DESC's own
            equilibrium for the same boundary rather than VMEC's solution
            expressed in DESC's basis.

    Returns:
        The path written.
    """
    from desc.vmec import VMECIO  # noqa: PLC0415  (optional dependency)

    L, M, N = FIT_LMN
    eq = VMECIO.load(wout, L=L, M=M, N=N, profile="iota")
    if solve:
        eq.solve(verbose=2, ftol=1e-8, maxiter=100)
    path = os.path.join(out_dir, f"{stem}{'_solved' if solve else ''}.h5")
    eq.save(path)
    return path


def main(cli: argparse.Namespace) -> None:
    """Write every fixture into ``cli.out``.

    Args:
        cli: parsed arguments; see :func:`parse_args`.
    """
    import desc  # noqa: PLC0415  (optional dependency)

    os.makedirs(cli.out, exist_ok=True)
    examples = os.path.join(os.path.dirname(desc.__file__), "examples")
    written = []
    for name in EXAMPLES:
        dst = os.path.join(cli.out, f"desc_{name}.h5")
        trim(os.path.join(examples, f"{name}_output.h5"), dst)
        written.append(dst)
    for wout, stem in WOUTS:
        if not os.path.isfile(wout):
            print(f"[skip] {wout} is missing", flush=True)
            continue
        for solve in (False, True) if cli.solve else (False,):
            written.append(refit(wout, stem, cli.out, solve))
    notice = os.path.join(cli.out, "DESC_LICENSE")
    with open(notice, "w") as fh:
        fh.write(LICENSE_NOTICE)
    written.append(notice)
    for path in written:
        print(f"  {os.path.getsize(path) / 1024:8.0f} KiB  {path}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: argument list; ``None`` reads ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data", help="directory to write fixtures into")
    ap.add_argument("--no-solve", dest="solve", action="store_false",
                    help="only the fit-only refits, skipping the eq.solve() variants")
    return ap.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
