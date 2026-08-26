"""GVEC state files (``GVEC_State_*.dat``): read, evaluate, export.

A state file is GVEC's own representation of an equilibrium: the radial
B-spline basis (degree ``deg`` on the element grid ``sp``), the Fourier mode
table ``(m, n)`` -- ``n`` already multiplied by ``nfp`` -- and, per mode, the
radial coefficients of ``X1 = R`` (cosine series), ``X2 = Z`` and
``LA = lambda`` (sine series), followed by the profiles ``Phi``, ``chi``,
``iota``, ``p`` at the radial interpolation points of the ``X1`` basis. Its
angles are GVEC's ``theta`` and ``zeta`` in radians with the series
``sum f_mn(s) trig(m theta - n zeta)``; ``s`` is the radial label the
flat-schema exports call ``rho`` (``Phi = Phi_edge s^2``).

:class:`StateField` evaluates one of the three fields at a logical point in
JAX -- the radial basis through :class:`mrx.spline_bases.SplineBasis` on
GVEC's own knots, the angles as ``2 pi (m theta - n zeta / nfp)`` -- so the
map fit collocates ``R`` and ``Z`` at its Greville points and the initial
condition histopolates ``lambda`` at its quadrature points from the closed
form, with no intermediate grid: ``mrx.gvec.build_gvec_map`` and
``load_clebsch`` take a ``.dat`` directly. :func:`write_flat_schema`
evaluates the state on a tensor grid and writes the flat-schema file for
the tools that want a grid (``scripts/trim_gvec_export.py``). Validated
against the pyGVEC export of W7-X FMM002 to round-off
(``test/test_gvec_state.py``).
"""
from __future__ import annotations

import os

import h5py
import jax  # noqa: F401  (StateField is traced by the callers' vmaps)
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import BSpline

CLEBSCH_CONTRACT = ("sqrt(g) B^rho = 0; sqrt(g) B^theta = dchi_dr - dPhi_dr * dLA_dz; "
                    "sqrt(g) B^zeta = dPhi_dr * (1 + dLA_dt)")


def _numbers(line):
    return [float(v) for v in line.replace(",", " ").split()]


def read_state(path):
    """Parse a state file into a dict: ``nfp``, ``sp``, ``deg``, the three
    field blocks ``X1``, ``X2``, ``LA`` (``m``, ``n``, ``coef`` of shape
    ``(n_modes, n_base)``, ``sin_cos`` 1 = sine, 2 = cosine), ``profiles``
    (``s``, ``phi``, ``chi``, ``iota``, ``pressure`` at the interpolation
    points) and ``a_minor``, ``r_major``, ``volume``."""
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    heads = [i for i, ln in enumerate(lines) if ln.startswith("##")]
    blocks = []                                     # (header text, data lines)
    for j, i in enumerate(heads):
        end = heads[j + 1] if j + 1 < len(heads) else len(lines)
        blocks.append((lines[i][2:].strip(" #"), [ln for ln in lines[i + 1:end] if ln.strip()]))

    def block(prefix):
        for head, data in blocks:
            if head.startswith(prefix):
                return data
        raise ValueError(f"{path}: no '## {prefix}' block")

    st = {}
    n_elems = int(_numbers(block("grid: nElems")[0])[0])
    st["sp"] = np.array(_numbers(block("grid: sp")[0]))[: n_elems + 1]
    nfp, _, _, _, hmap = _numbers(block("global")[0])
    st["nfp"], st["hmap"] = int(nfp), int(hmap)
    for name in ("X1", "X2", "LA"):
        n_base, deg, _, n_modes, sin_cos, _ = (int(v) for v in _numbers(block(f"{name}_base")[0]))
        rows = np.array([_numbers(ln) for ln in block(f"{name}:")])
        if rows.shape != (n_modes, 2 + n_base):
            raise ValueError(f"{path}: {name} block is {rows.shape}, expected {(n_modes, 2 + n_base)}")
        st[name] = dict(m=rows[:, 0].astype(int), n=rows[:, 1].astype(int),
                        coef=rows[:, 2:], sin_cos=sin_cos, deg=deg)
    st["deg"] = st["X1"]["deg"]
    prof = np.array([_numbers(ln) for ln in block("at X1_base IP point positions")])
    st["profiles"] = dict(zip(("s", "phi", "chi", "iota", "pressure"), prof.T))
    st["a_minor"], st["r_major"], st["volume"] = _numbers(block("a_minor,r_major,volume")[0])
    return st


def knots(sp, deg):
    """Clamped knot vector of the degree-``deg`` B-splines on the element grid."""
    return np.concatenate([np.full(deg, sp[0]), sp, np.full(deg, sp[-1])])


def radial_design(sp, deg, s):
    """``(len(s), n_base)`` values of the radial basis at ``s``."""
    return BSpline.design_matrix(np.asarray(s, dtype=np.float64), knots(sp, deg), deg).toarray()


def evaluate(block, sp, s, theta, zeta):
    """A field block on the tensor grid ``s x theta x zeta`` (angles in
    radians, ``zeta`` the physical toroidal angle)."""
    A = radial_design(sp, block["deg"], s) @ block["coef"].T             # (n_s, n_modes)
    arg = (np.outer(block["m"], theta)[:, :, None]
           - np.outer(block["n"], zeta)[:, None, :])                      # (n_modes, n_t, n_z)
    F = np.cos(arg) if block["sin_cos"] == 2 else np.sin(arg)
    return np.einsum("sk,ktz->stz", A, F)


def profile_spline(st, name):
    """The radial spline through a profile's interpolation-point values."""
    prof, sp, deg = st["profiles"], st["sp"], st["deg"]
    c = np.linalg.solve(radial_design(sp, deg, prof["s"]), prof[name])
    return BSpline(knots(sp, deg), c, deg)


class StateField:
    """A state's ``X1``, ``X2`` or ``LA`` as a JAX function of the logical
    point ``(rho, theta, zeta)`` (angles on ``[0, 1)``, ``zeta`` per field
    period). ``vector=True`` returns a ``(1,)`` array, the convention of the
    map fit's scalar callables; otherwise a scalar."""

    def __init__(self, block, sp, nfp, vector=False):
        from mrx.spline_bases import SplineBasis
        self.basis = SplineBasis(block["coef"].shape[1], block["deg"], "clamped",
                                 T=jnp.asarray(knots(sp, block["deg"])))
        self.C = jnp.asarray(block["coef"])                              # (n_modes, n_base)
        self.m = jnp.asarray(block["m"], dtype=jnp.float64)
        self.n_per = jnp.asarray(block["n"], dtype=jnp.float64) / nfp    # per field period
        self.cos = block["sin_cos"] == 2
        self.vector = vector

    def __call__(self, x):
        vals, idx = self.basis.evaluate_local(jnp.clip(x[0], 0.0, 1.0))
        radial = self.C[:, idx] @ vals                                    # (n_modes,)
        arg = 2.0 * jnp.pi * (self.m * x[1] - self.n_per * x[2])
        f = (jnp.cos(arg) if self.cos else jnp.sin(arg)) @ radial
        return jnp.array([f]) if self.vector else f


def load_state_clebsch(path, n_rho=401):
    """The ``load_clebsch`` dict of a state file: profiles on ``n_rho``
    uniform radii from the profile splines (``chi' = iota Phi'``) and
    ``lam_h`` the closed-form :class:`StateField` of ``LA``."""
    st = read_state(path)
    rho = np.linspace(0.0, 1.0, n_rho)
    dPhi = profile_spline(st, "phi").derivative()(rho)
    return dict(nfp=st["nfp"], rho=rho, dPhi=dPhi,
                dchi=profile_spline(st, "iota")(rho) * dPhi,
                p=profile_spline(st, "pressure")(rho), iota_spread=0.0,
                lam_h=StateField(st["LA"], st["sp"], st["nfp"]), closed_axes=[])


def radial_axis(n):
    """``n`` radial samples: ``i / (n - 1)`` with the axis point moved to
    ``0.1 / (n - 1)``, as the pyGVEC exports place it."""
    rho = np.arange(n, dtype=np.float64) / (n - 1)
    rho[0] = 0.1 / (n - 1)
    return rho


def write_flat_schema(state_path, out_path, n_rho=50, n_theta=50, n_zeta=50):
    """Evaluate a state on an ``n_rho x n_theta x n_zeta`` grid and write the
    flat-schema file :mod:`mrx.gvec` reads. Logical ``theta`` and ``zeta``
    are uniform on ``[0, 1)``; ``zeta`` spans one field period."""
    st = read_state(state_path)
    nfp = st["nfp"]
    rho = radial_axis(n_rho)
    th = np.arange(n_theta) / n_theta
    ze = np.arange(n_zeta) / n_zeta
    theta, zeta = 2.0 * np.pi * th, 2.0 * np.pi * ze / nfp
    R = evaluate(st["X1"], st["sp"], rho, theta, zeta)
    Z = evaluate(st["X2"], st["sp"], rho, theta, zeta)
    LA = evaluate(st["LA"], st["sp"], rho, theta, zeta)
    dPhi = profile_spline(st, "phi").derivative()(rho)
    # GVEC's primary profile is iota; chi' = iota Phi' reproduces the pyGVEC
    # export to 1e-5 (the interpolation floor of the 15 profile samples),
    # differentiating the chi samples to 7e-5.
    dchi = profile_spline(st, "iota")(rho) * dPhi
    p = profile_spline(st, "pressure")(rho)
    shape = (n_rho, n_theta, n_zeta)
    pts = np.stack(np.meshgrid(rho, th, ze, indexing="ij"), axis=-1).reshape(-1, 3)
    with h5py.File(out_path, "w") as h:
        h.attrs["n_rho"], h.attrs["n_theta"], h.attrs["n_zeta"] = shape
        h.attrs["nfp"] = nfp
        h.attrs["gvec_source"] = os.path.abspath(state_path)
        h.attrs["exported_by"] = "mrx.gvec_state.write_flat_schema"
        h.attrs["clebsch_contract"] = CLEBSCH_CONTRACT
        h.attrs["a_minor"], h.attrs["r_major"], h.attrs["volume"] = st["a_minor"], st["r_major"], st["volume"]
        h.create_dataset("eval_points", data=pts, compression="gzip", shuffle=True)
        for k, v in (("R", R), ("Z", Z), ("pressure", np.broadcast_to(p[:, None, None], shape)),
                     ("clebsch/dPhi_dr", np.broadcast_to(dPhi[:, None, None], shape)),
                     ("clebsch/dchi_dr", np.broadcast_to(dchi[:, None, None], shape)),
                     ("clebsch/LA", LA)):
            h.create_dataset(k, data=np.ascontiguousarray(v, dtype=np.float64).ravel(),
                             compression="gzip", compression_opts=9, shuffle=True)
    return out_path


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Export a GVEC state file to the flat schema MRX reads.")
    ap.add_argument("state")
    ap.add_argument("out")
    ap.add_argument("--n-rho", type=int, default=50)
    ap.add_argument("--n-theta", type=int, default=50)
    ap.add_argument("--n-zeta", type=int, default=50)
    cli = ap.parse_args()
    write_flat_schema(cli.state, cli.out, cli.n_rho, cli.n_theta, cli.n_zeta)
    print(f"wrote {cli.out}: grid ({cli.n_rho}, {cli.n_theta}, {cli.n_zeta}), "
          f"{os.path.getsize(cli.out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

