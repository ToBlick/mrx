"""``mrx.vmec``: wout files refit into GVEC-state blocks.

The synthetic tests need no data. The ``needs_data`` tests read the simsopt
reference files in ``data/`` (fetched from the simsopt repository's
``tests/test_files``): ``wout_li383_low_res_reference.nc`` (NCSX, nfp=3,
ns=16 -- the small parser/fit reference), ``wout_LandremanPaul2021_QA_lowres``
(vacuum, p = 0) and the W7-X beta=5%% file. The expensive initial-condition
gates additionally require ``MRX_WOUT_GATES=1`` and a GPU node."""
import os

import numpy as np
import pytest
from scipy.interpolate import BSpline

from mrx.vmec import (
    TWO_PI,
    _state_from_raw,
    load_wout_clebsch,
    profile_spline,
    read_nfp,
    read_wout,
)

LI383 = "data/wout_li383_low_res_reference.nc"
QA = "data/wout_LandremanPaul2021_QA_lowres.nc"
W7X = "data/wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"


def _needs(path):
    return pytest.mark.skipif(not os.path.isfile(path), reason=f"{path} is absent")


# ---------------------------------------------------------------------------
# synthetic: the fit and the guards
# ---------------------------------------------------------------------------

def _synthetic_raw(ns=9, nfp=3):
    """A tiny consistent wout dict: smooth mode profiles in s, junk in the
    rows the reader must ignore (lmns row 0, the m > 0 axis rows)."""
    s = np.arange(ns) / (ns - 1)
    m = np.array([0, 1, 2])
    n = np.array([0, 3, -6])
    rmnc = np.stack([3.0 + 0.1 * s, 0.9 * s ** 0.5 * (1 - 0.2 * s),
                     0.05 * s], axis=1)
    rmnc[0, 1:] = 123.0                       # junk axis extrapolation, m > 0
    zmns = np.stack([0.0 * s, 0.8 * s ** 0.5, -0.04 * s], axis=1)
    zmns[0, 1:] = -77.0
    lmns = np.stack([0.0 * s, 0.1 * s, 0.02 * s ** 2], axis=1)
    lmns[0] = 1e6                             # junk half-mesh row 0
    phi = 2.5 * s
    phipf = np.full(ns, 2.5)
    iotaf = 0.4 + 0.3 * s
    return dict(ns=ns, nfp=nfp, mnmax=len(m), xm=m, xn=n,
                lasym__logical__=0, signgs=-1, version_=9.0,
                rmnc=rmnc, zmns=zmns, lmns=lmns, phi=phi, phipf=phipf,
                chipf=iotaf * phipf, iotaf=iotaf, presf=1e4 * (1 - s))


def test_fit_interpolates_the_surface_data():
    """StateField on the fitted blocks reproduces the per-surface Fourier
    series at the mesh nodes (interpolation property), junk rows excluded."""
    import jax
    import jax.numpy as jnp
    from mrx.gvec import StateField

    raw = _synthetic_raw()
    st = _state_from_raw(raw)
    ns, nfp = raw["ns"], raw["nfp"]
    rho_full = np.sqrt(np.arange(ns) / (ns - 1))
    rho_half = np.sqrt((np.arange(1, ns) - 0.5) / (ns - 1))
    th, ze = 0.31, 0.77                       # logical angles, zeta per period
    arg = TWO_PI * (raw["xm"] * th - raw["xn"] / nfp * ze)
    for name, samples, rho, trig in (("X1", raw["rmnc"], rho_full, np.cos),
                                     ("X2", raw["zmns"], rho_full, np.sin),
                                     ("LA", raw["lmns"][1:], rho_half, np.sin)):
        samples = samples.copy()
        if name in ("X1", "X2"):
            samples[0, raw["xm"] > 0] = 0.0   # the reader pins these
        want = samples @ trig(arg)
        f = StateField(st[name], None, nfp)
        pts = jnp.array([[r, th, ze] for r in rho])
        got = np.asarray(jax.vmap(f)(pts))
        assert np.abs(got - want).max() < 1e-11


def test_axis_value_is_theta_independent():
    """m > 0 axis pinning: R at rho = 0 is a function of zeta only."""
    import jax
    import jax.numpy as jnp
    from mrx.gvec import StateField

    st = _state_from_raw(_synthetic_raw())
    f = StateField(st["X1"], None, 3)
    pts = jnp.array([[0.0, t, 0.4] for t in np.linspace(0, 1, 7)])
    vals = np.asarray(jax.vmap(f)(pts))
    assert np.abs(vals - vals[0]).max() < 1e-13


def test_profile_spline_is_exact_for_the_flux():
    """phi is linear in s, so the rho-spline is the exact quadratic and
    dPhi/drho = 2 rho Phi_edge / 2 pi."""
    raw = _synthetic_raw()
    st = _state_from_raw(raw)
    rho = np.linspace(0.0, 1.0, 33)
    dPhi = profile_spline(st, "phi").derivative()(rho)
    assert np.abs(dPhi - 2.0 * rho * raw["phi"][-1] / TWO_PI).max() < 1e-12


def test_guards():
    raw = _synthetic_raw()
    bad = dict(raw, version_=6.9)
    with pytest.raises(ValueError, match="version"):
        _state_from_raw(bad)
    bad = dict(raw, lasym__logical__=1)
    with pytest.raises(NotImplementedError, match="lasym"):
        _state_from_raw(bad)
    bad = dict(raw, chipf=raw["chipf"] + 0.01)
    with pytest.raises(ValueError, match="chipf"):
        _state_from_raw(bad)


# ---------------------------------------------------------------------------
# the simsopt reference files
# ---------------------------------------------------------------------------

@pytest.mark.needs_data
@_needs(LI383)
def test_li383_reads_and_reproduces_the_file():
    st = read_wout(LI383)
    assert (st["nfp"], st["ns"], st["mnmax"]) == (3, 16, 25)
    assert read_nfp(LI383) == 3
    from mrx.geometry import geometry_nfp
    assert geometry_nfp(LI383) == 3
    # blocks: coefficient layout and knot squareness
    for name, n_base in (("X1", 16), ("X2", 16), ("LA", 17)):   # LA: half mesh + axis + edge
        blk = st[name]
        assert blk["coef"].shape == (25, n_base)
        assert len(blk["T"]) == n_base + blk["deg"] + 1
    # radial values at the mesh nodes reproduce the fitted samples
    rho = st["profiles"]["rho"]
    A = BSpline.design_matrix(rho, st["X1"]["T"], st["deg"]).toarray()
    R_nodes = A @ st["X1"]["coef"].T          # (ns, n_modes)
    assert np.isfinite(R_nodes).all()
    assert abs(R_nodes[0, 0] - 1.41) < 0.2    # NCSX axis R ~ 1.4 m


@pytest.mark.needs_data
@_needs(LI383)
def test_li383_clebsch_dict_matches_the_state_route_contract():
    cb = load_wout_clebsch(LI383)
    assert set(cb) == {"nfp", "rho", "dPhi", "dchi", "p", "iota_spread",
                       "lam_h", "closed_axes"}
    assert cb["nfp"] == 3 and cb["closed_axes"] == [] and cb["iota_spread"] == 0.0
    # dPhi is the exact quadratic flux derivative in GVEC units
    st = read_wout(LI383)
    phi_edge = st["profiles"]["phi"][-1]      # already / 2 pi
    assert np.abs(cb["dPhi"] - 2.0 * cb["rho"] * phi_edge).max() < 1e-10
    assert abs(cb["dPhi"][0]) < 1e-12
    # dchi / dPhi is iota (checked off-axis where dPhi != 0)
    iota = cb["dchi"][1:] / cb["dPhi"][1:]
    assert 0.3 < np.abs(iota).min() and np.abs(iota).max() < 0.7
    # lambda evaluates finitely through the closed form
    import jax
    import jax.numpy as jnp
    vals = np.asarray(jax.vmap(cb["lam_h"])(
        jnp.array([[0.3, 0.1, 0.2], [0.9, 0.7, 0.6]])))
    assert np.isfinite(vals).all() and np.abs(vals).max() < 1.0


@pytest.mark.needs_data
@_needs(QA)
def test_qa_wout_is_vacuum():
    cb = load_wout_clebsch(QA)
    assert cb["nfp"] == 2
    assert np.abs(cb["p"]).max() == 0.0       # presf is identically zero


@pytest.mark.needs_data
@_needs(W7X)
def test_w7x_wout_reads():
    st = read_wout(W7X)
    assert (st["nfp"], st["ns"]) == (5, 201)
    assert st["version"] >= 9.0 and st["signgs"] == -1
    iota = st["profiles"]["iota"]
    assert 0.85 < iota.min() and iota.max() < 1.01


# ---------------------------------------------------------------------------
# initial-condition gates (GPU; MRX_WOUT_GATES=1)
# ---------------------------------------------------------------------------

def _ic_report(path, ns, p):
    """The production IC route of ``relax.py --ic clebsch``: B = dA' from the
    histopolated potential -- exactly divergence-free in the complex, no
    Leray cleaning (which would carry the interpolant's derivatives into the
    current)."""
    import jax.numpy as jnp
    from mrx.geometry import build_sequence
    from mrx.gvec import load_clebsch
    from mrx.initial_conditions import (
        clebsch_potential_form,
        divergence_norm,
        potential_two_form,
    )
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import compute_force

    seq, ops = build_sequence(path, ns, p)
    seq.set_operators(compute_nullspaces(seq, ops))
    cb = load_clebsch(path, seq.basis_0.types)
    B, norm, wall = potential_two_form(seq, clebsch_potential_form(cb))
    div = divergence_norm(seq, B)
    F, _, _, _, _ = compute_force(B, seq)
    F_norm = float(seq.l2_norm(F, 2))
    print(f"\n  {os.path.basename(path)} {ns} p={p}: ||B||_M raw {norm:.4e}, "
          f"||div B|| {div:.2e}, wall-normal discarded {wall:.2e}, "
          f"||F||_M {F_norm:.3e} at ||B||_M = 1")
    assert jnp.isfinite(B).all()
    assert div <= 10 * seq.tol
    return seq, B, F_norm


@pytest.mark.needs_data
@_needs(W7X)
def test_w7x_wout_ic_gates():
    if not os.environ.get("MRX_WOUT_GATES"):
        pytest.skip("set MRX_WOUT_GATES=1 (GPU) to run the wout IC gates")
    _, _, F_norm = _ic_report(W7X, (12, 24, 12), 3)
    assert F_norm < 0.1                       # near force balance


@pytest.mark.needs_data
@_needs(QA)
def test_qa_vacuum_wout_ic_is_the_harmonic_field():
    """A vacuum equilibrium is curl-free and divergence-free, so the
    projected Clebsch field must be (discretisation-)close to the k=2
    Dirichlet harmonic form -- and near force balance with p = 0."""
    if not os.environ.get("MRX_WOUT_GATES"):
        pytest.skip("set MRX_WOUT_GATES=1 (GPU) to run the wout IC gates")
    import jax.numpy as jnp

    seq, B, F_norm = _ic_report(QA, (12, 24, 12), 3)
    h = seq.nullspace(2, True)[0]             # (n_vectors, n_2) -> the form
    h = h / seq.l2_norm(h, 2)
    Bn = B / seq.l2_norm(B, 2)
    err = float(min(seq.l2_norm(Bn - h, 2), seq.l2_norm(Bn + h, 2)))
    print(f"  vacuum distance ||B_hat -+ h_hat||_M = {err:.3e}, "
          f"||F||_M {F_norm:.3e}")
    assert jnp.isfinite(err)
    assert err < 0.1
