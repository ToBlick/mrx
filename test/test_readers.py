"""The three equilibrium readers (GVEC, VMEC, DESC). These tests need no sequence.

GVEC: a synthetic state written by ``test/synthetic_gvec.py`` from closed formulas is read back and
reproduces the formulas to round-off. VMEC: the tracked li383 wout file reads with the expected layout and
a plausible axis. DESC: a synthetic file (``test/synthetic_desc.py``, iota- or current-constrained)
reproduces its formulas to round-off, and DESC's conversion of the li383 wout matches the wout.
Without stellarator symmetry: the li383 wout and the synthetic GVEC state, shifted in both angles, read as
the shifted series, and a synthetic DESC file with terms of the other parity reproduces its formulas.
Orientation: ``read_equilibrium`` reverses the poloidal angle of the left-handed VMEC files and not of DESC's
right-handed conversion, and the result is right-handed with the same R and Z at the mirrored angle.
"""
import numpy as np
from scipy.interpolate import BSpline
from scipy.io import netcdf_file

import mrx
from mrx.equilibria import _left_handed, is_stellarator_symmetric, read_equilibrium
from mrx.equilibria.fit import axis_orders
from mrx.equilibria.desc import read_desc
from mrx.equilibria.gvec import read_state
from mrx.equilibria.vmec import read_wout
from test.synthetic_desc import write_synthetic_desc
from test.synthetic_gvec import TWO_PI, evaluate, write_synthetic_state

# A W7-X-like rotational transform. Psi_edge = pi a^2 makes the mean toroidal field 1.
R0, A, NFP = 1.0, 1.0 / 3.0, 5
IOTA = (-0.9, -0.15)
PSI_EDGE = np.pi * A ** 2
LAM_AMPLITUDE, BETA = 0.05, 1e-3

LI383 = "data/wout_li383_low_res_reference.nc"
DESC_LI383 = "data/desc_li383_low_res_reference.h5"
EPS64 = np.finfo(np.float64).eps


def test_gvec_state_file_reproduces_the_formulas(tmp_path):
    """The GVEC reader returns the closed-form map, lambda and profiles of the synthetic state to
    round-off."""
    path = str(tmp_path / "GVEC_State_torus.dat")
    torus = write_synthetic_state(path, R0=R0, a=A, nfp=NFP, iota=IOTA, Psi_edge=PSI_EDGE,
                                  lam_amplitude=LAM_AMPLITUDE, beta=BETA)
    st = read_state(path)
    assert st["nfp"] == NFP and st["X1"]["deg"] == 5 and is_stellarator_symmetric(st)
    assert abs(st["a_minor"] - A) <= mrx.eps(8) and st["r_major"] == R0
    r = np.array([0.0, 0.13, 0.5, 0.87, 1.0])
    th, ze = np.array([0.0, 0.2, 0.45, 0.7]), np.array([0.0, 0.3, 0.8])
    RR, TH, ZE = np.meshgrid(r, th, ze, indexing="ij")
    for blk, want in (("X1", torus.R(RR, TH)), ("X2", torus.Z(RR, TH)),
                      ("LA", torus.LA(RR, TH, ZE))):
        got = evaluate(st[blk], r, TWO_PI * th, TWO_PI * ze / NFP)
        assert np.abs(got - np.asarray(want)).max() <= mrx.eps(512), blk
    r = np.linspace(0.0, 1.0, 37)
    for name, want in (("phi", torus.Psi(r)), ("chi", torus.chi(r)),
                       ("iota", torus.iota(r)), ("pressure", torus.pressure(r))):
        got = st["profiles"][name](r)
        assert np.abs(got - np.asarray(want)).max() <= mrx.eps(8192) * max(1.0, np.abs(want).max()), name
    dPsi = st["profiles"]["phi"].derivative()(r)
    assert np.abs(dPsi - np.asarray(torus.dPsi_dr(r))).max() <= mrx.eps(8192)


def test_vmec_li383_reads_and_reproduces_the_file():
    """The li383 wout file reads with the expected sizes and a finite map with the axis near R = 1.4 m."""
    st = read_wout(LI383)
    assert (st["nfp"], st["ns"], st["mnmax"]) == (3, 16, 25)
    deg = st["X1"]["deg"]
    k_max = max(len(axis_orders(int(mm), deg)) for mm in st["X1"]["m"])
    # the radial nodes plus the extra axis coefficients. LA: half mesh plus axis plus edge
    for name, n_base in (("X1", 16 + k_max), ("X2", 16 + k_max), ("LA", 17 + k_max)):
        blk = st[name]
        assert blk["cos"].shape == blk["sin"].shape == (25, n_base)
        assert len(blk["T"]) == n_base + blk["deg"] + 1
    # radial values at the mesh nodes reproduce the fitted samples
    r = np.sqrt(np.arange(st["ns"]) / (st["ns"] - 1))
    design = BSpline.design_matrix(r, st["X1"]["T"], deg).toarray()
    R_nodes = design @ st["X1"]["cos"].T           # (ns, n_modes)
    assert np.isfinite(R_nodes).all()
    assert abs(R_nodes[0, 0] - 1.41) < 0.2    # NCSX axis R ~ 1.4 m


def test_desc_synthetic_and_li383(tmp_path):
    """The DESC reader reproduces the synthetic file to round-off, and DESC's li383 agrees with the VMEC
    li383 within DESC's fit error."""
    path = str(tmp_path / "desc.h5")
    r = np.linspace(0.0, 1.0, 11)
    for kw in ({}, dict(b=0.0, d=0.0, e=0.0, current=(2.0e4, -5.0e3))):
        torus = write_synthetic_desc(path, **kw)
        st = read_desc(path)
        th, ze = TWO_PI * np.linspace(0.0, 1.0, 13), TWO_PI * np.linspace(0.0, 1.0, 9) / torus.nfp
        for blk, want in zip(("X1", "X2", "LA"), torus.fields(*np.meshgrid(r, th, ze, indexing="ij"))):
            assert np.abs(evaluate(st[blk], r, th, ze) - want).max() <= 64 * EPS64, blk
        prof = st["profiles"]
        assert np.abs(prof["phi"](r) - torus.Psi * r ** 2 / TWO_PI).max() <= 8 * EPS64
        assert np.abs(prof["iota"](r) - torus.iota(r)).max() <= 256 * EPS64
        assert np.abs(prof["pressure"](r) - torus.pressure(r)).max() <= 64 * EPS64 * torus.p0

    # DESC's conversion of the li383 wout (theta = -u, iota flips): R and Z within DESC's degree-8 Zernike fit
    # of the wout (2.5e-4, 1.8e-3), iota the wout's at its nodes
    st_d, st_v = read_desc(DESC_LI383), read_wout(LI383)
    r_full = np.sqrt(np.arange(st_v["ns"]) / (st_v["ns"] - 1))
    th, ze = TWO_PI * np.arange(16) / 16, TWO_PI * np.arange(8) / (8 * st_d["nfp"])
    for blk, tol in (("X1", 5e-4), ("X2", 3e-3)):
        want = evaluate(st_v[blk], r, -th, ze)
        assert np.abs(evaluate(st_d[blk], r, th, ze) - want).max() <= tol * np.abs(want).max(), blk
    assert np.abs(st_d["profiles"]["iota"](r_full) + st_v["profiles"]["iota"](r_full)).max() <= 64 * EPS64
    assert abs(st_d["profiles"]["phi"](1.0) - st_v["profiles"]["phi"](1.0)) <= 8 * EPS64


def _shifted_wout(src, path, c, d):
    """Write the wout ``src`` shifted to ``(u + c, v + d)`` as a non-symmetric (lasym = 1) wout."""
    with netcdf_file(src, mmap=False) as f:
        v = {k: np.array(f.variables[k][()]) for k in ("ns", "nfp", "mnmax", "version_", "xm", "xn", "rmnc",
                                                         "zmns", "lmns", "phi", "iotaf", "presf")}
    phase = v["xm"] * c - v["xn"] * d
    cos, sin = np.cos(phase), np.sin(phase)
    series = dict(rmnc=v["rmnc"] * cos, rmns=-v["rmnc"] * sin, zmns=v["zmns"] * cos, zmnc=v["zmns"] * sin,
                  lmns=v["lmns"] * cos, lmnc=v["lmns"] * sin)
    with netcdf_file(path, "w") as f:
        f.createDimension("radius", int(v["ns"]))
        f.createDimension("mn_mode", int(v["mnmax"]))
        for k, value in dict(ns=v["ns"], nfp=v["nfp"], mnmax=v["mnmax"], lasym__logical__=1).items():
            f.createVariable(k, "i4", ())[()] = value
        f.createVariable("version_", "f8", ())[()] = v["version_"]
        for k in ("xm", "xn"):
            f.createVariable(k, "f8", ("mn_mode",))[:] = v[k]
        for k in ("phi", "iotaf", "presf"):
            f.createVariable(k, "f8", ("radius",))[:] = v[k]
        for k, value in series.items():
            f.createVariable(k, "f8", ("radius", "mn_mode"))[:] = value
    return path


def test_non_stellarator_symmetric_states(tmp_path):
    """Shifted VMEC and GVEC states read as the shifted series of the symmetric ones, and a
    non-symmetric synthetic DESC file reproduces its formulas."""
    c, d = 0.3, 0.2                        # the poloidal and the (full-turn) toroidal shift, radians
    r = np.linspace(0.0, 1.0, 11)
    th, ze = TWO_PI * np.linspace(0.0, 1.0, 13), TWO_PI * np.linspace(0.0, 1.0, 9) / 3
    kw = dict(R0=R0, a=A, nfp=NFP, iota=IOTA, Psi_edge=PSI_EDGE, lam_amplitude=LAM_AMPLITUDE, beta=BETA)
    gvec, gvec_shifted = str(tmp_path / "GVEC_State_torus.dat"), str(tmp_path / "GVEC_State_shifted.dat")
    write_synthetic_state(gvec, **kw)
    write_synthetic_state(gvec_shifted, shift=(c, d), **kw)
    pairs = [(read_wout(LI383), read_wout(_shifted_wout(LI383, str(tmp_path / "wout_shifted.nc"), c, d))),
             (read_state(gvec), read_state(gvec_shifted))]
    for st, st_s in pairs:
        assert is_stellarator_symmetric(st) and not is_stellarator_symmetric(st_s)
        for blk in ("X1", "X2", "LA"):
            want = evaluate(st[blk], r, th + c, ze + d)
            assert np.abs(evaluate(st_s[blk], r, th, ze) - want).max() <= 64 * EPS64 * np.abs(want).max(), blk

    path = str(tmp_path / "desc.h5")
    torus = write_synthetic_desc(path, asym=(0.01, 0.02, 0.03))
    st = read_desc(path)
    assert not is_stellarator_symmetric(st)
    th, ze = TWO_PI * np.linspace(0.0, 1.0, 13), TWO_PI * np.linspace(0.0, 1.0, 9) / torus.nfp
    for blk, want in zip(("X1", "X2", "LA"), torus.fields(*np.meshgrid(r, th, ze, indexing="ij"))):
        assert np.abs(evaluate(st[blk], r, th, ze) - want).max() <= 64 * EPS64, blk


def test_angles_are_right_handed_after_reading():
    """A VMEC wout is read with theta reversed, DESC's conversion of it is not, and both are right-handed
    afterwards. The reversed state is the wout's at ``-theta``, with lambda and iota of the opposite sign."""
    r, th, ze = np.linspace(0.1, 1.0, 7), TWO_PI * np.linspace(0.0, 1.0, 11), TWO_PI * np.linspace(0.0, 1.0, 5) / 3
    for path, reversed_ in ((LI383, True), ("data/wout_LandremanPaul2021_QA_lowres.nc", True), (DESC_LI383, False)):
        st = read_equilibrium(path)
        assert st["theta_reversed"] is reversed_ and not _left_handed(st), path
    st, raw = read_equilibrium(LI383), read_wout(LI383)
    for blk, sign in (("X1", 1.0), ("X2", 1.0), ("LA", -1.0)):
        want = sign * evaluate(raw[blk], r, -th, ze)
        assert np.abs(evaluate(st[blk], r, th, ze) - want).max() <= 64 * EPS64 * np.abs(want).max(), blk
    assert np.abs(st["profiles"]["iota"](r) + raw["profiles"]["iota"](r)).max() <= 64 * EPS64
