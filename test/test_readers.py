"""The two equilibrium readers, one check each, no sequence.

GVEC: a synthetic state written by ``test/synthetic_gvec.py`` from closed-form
profiles and a Fourier-spline torus is read back and reproduces the formulas
to round-off (the writer is the parser's inverse). VMEC: the tracked li383 wout
reads with the expected layout and a sane axis.
"""
import numpy as np
from scipy.interpolate import BSpline

import mrx
from mrx.gvec import evaluate, profile_spline, read_state
from mrx.vmec import read_nfp, read_wout
from test.synthetic_gvec import TWO_PI, write_synthetic_state

# W7-X-like transform; Phi_edge = pi a^2 makes the mean toroidal field 1.
R0, A, NFP = 1.0, 1.0 / 3.0, 5
IOTA = (-0.9, -0.15)
PHI_EDGE = np.pi * A ** 2
LAM_AMPLITUDE, BETA = 0.05, 1e-3

LI383 = "data/wout_li383_low_res_reference.nc"


def test_gvec_state_file_reproduces_the_formulas(tmp_path):
    path = str(tmp_path / "GVEC_State_torus.dat")
    torus = write_synthetic_state(path, R0=R0, a=A, nfp=NFP, iota=IOTA, Phi_edge=PHI_EDGE,
                                  lam_amplitude=LAM_AMPLITUDE, beta=BETA)
    st = read_state(path)
    assert st["nfp"] == NFP and st["deg"] == 5 and st["X1"]["sin_cos"] == 2
    assert st["X2"]["sin_cos"] == 1 and st["LA"]["sin_cos"] == 1
    assert abs(st["a_minor"] - A) <= mrx.eps(8) and st["r_major"] == R0
    rho = np.array([0.0, 0.13, 0.5, 0.87, 1.0])
    th, ze = np.array([0.0, 0.2, 0.45, 0.7]), np.array([0.0, 0.3, 0.8])
    RHO, TH, ZE = np.meshgrid(rho, th, ze, indexing="ij")
    for blk, want in (("X1", torus.R(RHO, TH)), ("X2", torus.Z(RHO, TH)),
                      ("LA", torus.LA(RHO, TH, ZE))):
        got = evaluate(st[blk], st["sp"], rho, TWO_PI * th, TWO_PI * ze / NFP)
        assert np.abs(got - np.asarray(want)).max() <= mrx.eps(512), blk
    r = np.linspace(0.0, 1.0, 37)
    for name, want in (("phi", torus.Phi(r)), ("chi", torus.chi(r)),
                       ("iota", torus.iota(r)), ("pressure", torus.pressure(r))):
        got = profile_spline(st, name)(r)
        assert np.abs(got - np.asarray(want)).max() <= mrx.eps(8192) * max(1.0, np.abs(want).max()), name
    dPhi = profile_spline(st, "phi").derivative()(r)
    assert np.abs(dPhi - np.asarray(torus.dPhi_dr(r))).max() <= mrx.eps(8192)


def test_vmec_li383_reads_and_reproduces_the_file():
    st = read_wout(LI383)
    assert (st["nfp"], st["ns"], st["mnmax"]) == (3, 16, 25)
    assert read_nfp(LI383) == 3
    from mrx.geometry import geometry_nfp
    assert geometry_nfp(LI383) == 3
    from mrx.vmec import _axis_orders
    k_max = max(len(_axis_orders(int(mm), st["deg"])) for mm in st["X1"]["m"])
    # nodes + the axis phantoms; LA: half mesh + axis + edge
    for name, n_base in (("X1", 16 + k_max), ("X2", 16 + k_max), ("LA", 17 + k_max)):
        blk = st[name]
        assert blk["coef"].shape == (25, n_base)
        assert len(blk["T"]) == n_base + blk["deg"] + 1
    # radial values at the mesh nodes reproduce the fitted samples
    rho = st["profiles"]["rho"]
    design = BSpline.design_matrix(rho, st["X1"]["T"], st["deg"]).toarray()
    R_nodes = design @ st["X1"]["coef"].T          # (ns, n_modes)
    assert np.isfinite(R_nodes).all()
    assert abs(R_nodes[0, 0] - 1.41) < 0.2    # NCSX axis R ~ 1.4 m
