"""The DESC reader: the basis conversion, the guards, and the dispatch.

Three tiers, cheapest first.

*Synthetic.* ``test/synthetic_desc.py`` writes a DESC-layout file of a
circular torus from closed formulas and :func:`mrx.desc.read_desc` reads it
back; since every radial function in it is ``1`` or ``rho``, which the
clamped cubic refit represents exactly, agreement is to round-off. The same
torus is written as a GVEC state by ``test/synthetic_gvec.py``, so the two
readers are also held against each other -- a conversion error that the
formulas somehow tolerate still has to survive being compared with an
independent parser of an independent format.

*Tracked fixtures.* ``data/desc_SOLOVEV.h5`` and ``data/desc_DSHAPE_lowres.h5``
are trimmed copies of DESC's own examples (``scripts/desc_fixtures.py``),
so they were not written by us and catch anything our writer and our parser
agree on wrongly. The sharp check on them is
:func:`test_desc_fixtures_reproduce_their_own_fourier_zernike_series`, which
rebuilds the DESC series in DESC's PRODUCT form, independently of the
product-to-sum conversion under test, and compares.

*Live DESC.* Skipped unless DESC is importable. This is the only check that
settles the angle conventions empirically rather than by reading DESC's
source: it asks DESC itself to evaluate ``R``, ``Z`` and ``lambda`` and
compares with what MRX parsed.

Nothing here builds a sequence except the one dispatch test, which uses the
coarsest mesh that still maps: the suite is compile-bound.
"""
import numpy as np
import pytest

import mrx
from mrx.desc import (_convert_block, _equilibrium, _nodes, _split_weights,
                      _zernike_radial, profile_spline, read_desc, read_nfp)
from mrx.geometry import geometry_kind, geometry_nfp
from mrx.gvec import evaluate, read_equilibrium
from test.synthetic_desc import write_synthetic_desc
from test.synthetic_gvec import LA_ZETA_MODULATION, TWO_PI, write_synthetic_state

# The same W7-X-like torus test_readers.py uses for the GVEC writer.
R0, A, NFP = 1.0, 1.0 / 3.0, 5
IOTA = (-0.9, -0.15)
PHI_EDGE = np.pi * A ** 2
LAM_AMPLITUDE, BETA = 0.05, 1e-3

SOLOVEV = "data/desc_SOLOVEV.h5"
DSHAPE = "data/desc_DSHAPE_lowres.h5"

#: Radii, poloidal and toroidal angles the fields are compared at. Chosen
#: off the sample nodes, where an interpolatory refit is unconstrained.
RHO = np.array([0.0, 0.137, 0.41, 0.63, 0.884, 1.0])
THETA = np.array([0.0, 0.2, 0.45, 0.7])
ZETA = np.array([0.0, 0.3, 0.8])


def _torus(tmp_path, **kwargs):
    """Write the synthetic DESC torus into ``tmp_path`` and read it back.

    Args:
        tmp_path: pytest's temporary directory.
        **kwargs: overrides of the default torus parameters.

    Returns:
        ``(path, torus, state)``: the file, its closed formulas and the
        parsed state, or ``(path, torus, None)`` if reading is expected to
        raise (``read=False``).
    """
    read = kwargs.pop("read", True)
    path = str(tmp_path / "desc_torus.h5")
    params = dict(R0=R0, a=A, nfp=NFP, iota=IOTA, Phi_edge=PHI_EDGE,
                  lam_amplitude=LAM_AMPLITUDE, beta=BETA)
    torus = write_synthetic_desc(path, **{**params, **kwargs})
    return path, torus, read_desc(path) if read else None


def test_desc_synthetic_file_reproduces_the_formulas(tmp_path):
    """The parser is the writer's inverse, to round-off."""
    path, torus, st = _torus(tmp_path)
    assert st["nfp"] == NFP and st["deg"] == 3
    assert (st["X1"]["sin_cos"], st["X2"]["sin_cos"], st["LA"]["sin_cos"]) == (2, 1, 1)
    assert read_nfp(path) == NFP

    grid = np.meshgrid(RHO, THETA, ZETA, indexing="ij")
    for blk, want in (("X1", torus.R(grid[0], grid[1])),
                      ("X2", torus.Z(grid[0], grid[1])),
                      ("LA", torus.LA(*grid))):
        got = evaluate(st[blk], RHO, TWO_PI * THETA, TWO_PI * ZETA / NFP)
        assert np.abs(got - np.asarray(want)).max() <= mrx.eps(512), blk

    r = np.linspace(0.0, 1.0, 37)
    for name, want in (("phi", torus.Phi(r)), ("iota", torus.iota(r)),
                       ("pressure", torus.pressure(r))):
        got = profile_spline(st, name)(r)
        scale = max(1.0, float(np.abs(want).max()))
        assert np.abs(got - np.asarray(want)).max() <= mrx.eps(8192) * scale, name
    dPhi = profile_spline(st, "phi").derivative()(r)
    assert np.abs(dPhi - np.asarray(torus.dPhi_dr(r))).max() <= mrx.eps(8192)


def test_desc_product_to_sum_splits_one_mode_into_the_pair(tmp_path):
    """The single sharpest check of the basis mapping.

    DESC stores the lambda modulation ``sin(theta) cos(nfp zeta)`` as ONE
    mode; MRX's single-angle basis needs the PAIR ``(1, +-nfp)`` at half
    amplitude each, which is exactly what ``test/synthetic_gvec.py`` writes
    by hand. So the conversion is right only if the parsed block has that
    pair, with those coefficients, and nothing else at ``|n| > 0``.
    """
    _, _, st = _torus(tmp_path)
    blk = st["LA"]
    modes = {(int(m), int(n)): row for m, n, row in zip(blk["m"], blk["n"], blk["coef"])}
    assert set(modes) == {(1, 0), (1, NFP), (1, -NFP)}

    # Each mode's radial function is its amplitude times rho; compare the
    # splines at the Greville-like interior point rho = 1, where rho = 1.
    from scipy.interpolate import BSpline
    for (_, n), amplitude in (((1, 0), LAM_AMPLITUDE),
                              ((1, NFP), 0.5 * LA_ZETA_MODULATION * LAM_AMPLITUDE),
                              ((1, -NFP), 0.5 * LA_ZETA_MODULATION * LAM_AMPLITUDE)):
        curve = BSpline(blk["T"], modes[(1, n)], blk["deg"])
        assert np.abs(curve(np.array([0.3, 1.0])) - amplitude * np.array([0.3, 1.0])
                      ).max() <= mrx.eps(64), n


def test_desc_zernike_radial_matches_the_low_order_polynomials():
    """``_zernike_radial`` against ``R_l^m`` written out by hand."""
    rho = np.linspace(0.0, 1.0, 11)
    closed = {(0, 0): np.ones_like(rho), (1, 1): rho,
              (2, 0): 2 * rho ** 2 - 1, (2, 2): rho ** 2,
              (3, 1): 3 * rho ** 3 - 2 * rho, (3, 3): rho ** 3,
              (4, 0): 6 * rho ** 4 - 6 * rho ** 2 + 1,
              (4, 2): 4 * rho ** 4 - 3 * rho ** 2}
    for (ell, m), want in closed.items():
        for signed in ({m, -m} if m else {0}):     # the sign of m is ignored
            got = _zernike_radial(rho, np.array(ell), np.array(signed))
            assert np.abs(got - want).max() <= mrx.eps(64), (ell, signed)
    # wrong parity (l - |m| odd) is identically zero, as in DESC
    assert np.abs(_zernike_radial(rho, np.array(2), np.array(1))).max() == 0.0


def test_desc_split_weights_cover_the_four_products():
    """The product-to-sum table, against the identities it encodes."""
    m = np.array([2, -2, -2, 2])                   # cos.cos, sin.sin, sin.cos, cos.sin
    n = np.array([1, -1, 1, -1])
    w_plus, w_minus = _split_weights(m, n)
    assert np.allclose(w_plus, [0.5, 0.5, 0.5, -0.5])
    assert np.allclose(w_minus, [0.5, -0.5, 0.5, 0.5])
    # with n = 0 the two branches are the same mode and must sum to one
    w_plus, w_minus = _split_weights(np.array([3, -3]), np.array([0, 0]))
    assert np.allclose(w_plus + w_minus, 1.0)


def test_desc_and_gvec_readers_agree_on_the_same_torus(tmp_path):
    """Two writers, two formats, two parsers, one field."""
    from mrx.gvec import profile_spline as gvec_profile
    from mrx.gvec import read_state

    _, _, st_d = _torus(tmp_path)
    gvec_path = str(tmp_path / "GVEC_State_torus.dat")
    write_synthetic_state(gvec_path, R0=R0, a=A, nfp=NFP, iota=IOTA, Phi_edge=PHI_EDGE,
                          lam_amplitude=LAM_AMPLITUDE, beta=BETA)
    st_g = read_state(gvec_path)
    for blk in ("X1", "X2", "LA"):
        got = evaluate(st_d[blk], RHO, TWO_PI * THETA, TWO_PI * ZETA / NFP)
        want = evaluate(st_g[blk], RHO, TWO_PI * THETA, TWO_PI * ZETA / NFP)
        assert np.abs(got - want).max() <= mrx.eps(512), blk
    r = np.linspace(0.0, 1.0, 21)
    for name in ("phi", "iota", "pressure"):
        want = gvec_profile(st_g, name)(r)
        scale = max(1.0, float(np.abs(want).max()))
        assert np.abs(profile_spline(st_d, name)(r) - want).max() <= mrx.eps(8192) * scale


def test_desc_negative_flux_is_carried_through(tmp_path):
    """``Psi < 0`` (HSX, W7-X) reverses the field; the geometry is untouched."""
    _, _, plus = _torus(tmp_path)
    _, torus, minus = _torus(tmp_path, Phi_edge=-PHI_EDGE)
    assert minus["Psi"] == -plus["Psi"] < 0
    r = np.linspace(0.0, 1.0, 21)
    assert np.abs(profile_spline(minus, "phi")(r) + profile_spline(plus, "phi")(r)).max() <= mrx.eps(64)
    assert (profile_spline(minus, "phi").derivative()(r[1:]) < 0).all()
    # iota is a ratio and does not flip with the flux
    assert np.abs(profile_spline(minus, "iota")(r) - np.asarray(torus.iota(r))).max() <= mrx.eps(64)
    for blk in ("X1", "X2"):
        a = evaluate(minus[blk], RHO, TWO_PI * THETA, TWO_PI * ZETA / NFP)
        b = evaluate(plus[blk], RHO, TWO_PI * THETA, TWO_PI * ZETA / NFP)
        assert np.abs(a - b).max() <= mrx.eps(64), blk


def test_desc_flip_poloidal_angle_is_an_involution_on_the_fields(tmp_path):
    """Flipping theta twice is the identity, and once is ``f(-theta)``."""
    from mrx.desc import flip_poloidal_angle

    _, _, st = _torus(tmp_path)
    flipped = flip_poloidal_angle(st)
    back = flip_poloidal_angle(flipped)
    for blk, sign in (("X1", 1.0), ("X2", 1.0), ("LA", -1.0)):
        straight = evaluate(st[blk], RHO, TWO_PI * THETA, TWO_PI * ZETA / NFP)
        at_minus = evaluate(st[blk], RHO, -TWO_PI * THETA, TWO_PI * ZETA / NFP)
        got = evaluate(flipped[blk], RHO, TWO_PI * THETA, TWO_PI * ZETA / NFP)
        assert np.abs(got - sign * at_minus).max() <= mrx.eps(64), blk
        assert np.abs(evaluate(back[blk], RHO, TWO_PI * THETA, TWO_PI * ZETA / NFP)
                      - straight).max() <= mrx.eps(64), blk
    r = np.linspace(0.0, 1.0, 21)
    assert np.abs(profile_spline(flipped, "iota")(r)
                  + profile_spline(st, "iota")(r)).max() <= mrx.eps(64)
    for name in ("phi", "pressure"):           # neither knows about theta
        assert np.abs(profile_spline(flipped, name)(r)
                      - profile_spline(st, name)(r)).max() <= mrx.eps(64), name


def test_desc_match_orientation_measures_rather_than_assumes(tmp_path):
    """A flipped state is turned back; an aligned one is left alone."""
    from mrx.desc import flip_poloidal_angle, match_orientation

    _, _, st = _torus(tmp_path)
    same, sign = match_orientation(st, st)
    assert sign == 1 and same is st
    fixed, sign = match_orientation(flip_poloidal_angle(st), st)
    assert sign == -1
    for blk in ("X1", "X2", "LA"):
        got = evaluate(fixed[blk], RHO, TWO_PI * THETA, TWO_PI * ZETA / NFP)
        want = evaluate(st[blk], RHO, TWO_PI * THETA, TWO_PI * ZETA / NFP)
        assert np.abs(got - want).max() <= mrx.eps(64), blk


def test_desc_refit_of_a_wout_matches_it_once_oriented():
    """The tracked li383 refit against the wout it was fit to.

    ``VMECIO.load`` runs ``ensure_positive_jacobian``, which flips theta for
    a left-handed wout, so the raw comparison is nonsense and the oriented
    one is tight. ``R`` and ``Z`` agree to the wout's own Fourier content;
    lambda does NOT, and deliberately is not asserted here -- DESC fits
    VMEC's half-mesh ``lmns`` as if it were on the full mesh, which
    ``scripts/desc_vmec_grid.py`` quantifies and ``docs/research`` records.
    """
    from mrx.desc import match_orientation
    from mrx.vmec import profile_spline as vmec_profile
    from mrx.vmec import read_wout

    st_v = read_wout("data/wout_li383_low_res_reference.nc")
    st_d, sign = match_orientation(read_desc("data/desc_li383_lowres.h5"), st_v)
    assert sign == -1                          # li383 is a left-handed wout
    theta = 2.0 * np.pi * np.array([0.1, 1.3, 2.9, 5.5]) / (2.0 * np.pi)
    theta, zeta = 2.0 * np.pi * theta, np.linspace(0.0, 2 * np.pi / st_v["nfp"], 5)
    rho = np.array([0.25, 0.5, 0.75, 1.0])
    for blk, tol in (("X1", 3e-4), ("X2", 3e-3)):
        got = evaluate(st_d[blk], rho, theta, zeta)
        want = evaluate(st_v[blk], rho, theta, zeta)
        assert np.abs(got - want).max() / np.abs(want).max() < tol, blk
    r = np.linspace(0.0, 1.0, 51)
    got, want = profile_spline(st_d, "iota")(r), vmec_profile(st_v, "iota")(r)
    assert np.abs(got - want).max() / np.abs(want).max() < 3e-3
    assert got[0] > 0 and want[0] > 0          # the flip restored iota's sign


def test_desc_takes_the_last_equilibrium_of_the_family(tmp_path):
    """A continuation family's converged member is the last one."""
    _, torus, st = _torus(tmp_path, n_family=3)
    on_axis = evaluate(st["X1"], np.array([0.0]), np.array([0.0]), np.array([0.0]))
    assert abs(float(on_axis.ravel()[0]) - R0) <= mrx.eps(64)      # not R0 + a


def test_desc_refuses_a_non_stellarator_symmetric_file(tmp_path):
    """``_sym = False`` needs the missing-parity partners, which MRX has not got."""
    path, _, _ = _torus(tmp_path, sym=False, read=False)
    with pytest.raises(NotImplementedError, match="_sym"):
        read_desc(path)


def test_desc_current_constrained_file_names_the_way_out(tmp_path):
    """Without a stored iota the reader needs DESC, and says which way out.

    Both ends of the fallback are error paths here, for different reasons,
    and each has to name its own. Without DESC there is nothing to ask. With
    DESC there is, but not about *this* file: it was written by hand and
    carries none of the class tags DESC's loader rebuilds from, so the
    honest report is that DESC could not read it -- not a traceback from
    inside DESC. The path itself is exercised on a real file by
    :func:`test_desc_iota_of_a_real_current_constrained_file`.
    """
    path, _, _ = _torus(tmp_path, store_iota=False, read=False)
    if _has_desc():
        with pytest.raises(ValueError, match="DESC could not load"):
            read_desc(path)
        return
    with pytest.raises(ImportError, match="current-constrained"):
        read_desc(path)


def test_desc_iota_of_a_real_current_constrained_file():
    """The DESC-backed iota path, on a file that really has no iota stored.

    The reader's one genuine gap: a current-constrained equilibrium does not
    store its rotational transform, so MRX has to ask DESC to recompute it.
    Checked against DESC's own value on the same radii, which is a
    round-trip of our grid and axis extrapolation rather than of the number
    itself -- the point is that the profile we hand downstream is the one
    DESC would compute, finite at the axis included.
    """
    pytest.importorskip("desc")
    from desc.grid import LinearGrid
    from desc.io import load

    path = _current_constrained_example()
    if path is None:
        pytest.skip("no current-constrained example shipped with this DESC")

    prof = read_desc(path)["profiles"]
    iota, rho = prof["iota"], prof["rho"]
    assert np.isfinite(iota).all() and rho[0] == 0.0

    eq = load(path)
    eq = eq[-1] if hasattr(eq, "__len__") else eq
    grid = LinearGrid(rho=rho[1:], M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    want = np.asarray(grid.compress(eq.compute("iota", grid=grid)["iota"]))
    assert np.allclose(iota[1:], want, rtol=1e-10, atol=1e-12)
    # The axis is extrapolated, not computed; it must at least continue the
    # profile rather than leave the range it is bracketed by.
    lo, hi = min(want[0], want[1]), max(want[0], want[1])
    assert lo - abs(hi - lo) - 1e-12 <= iota[0] <= hi + abs(hi - lo) + 1e-12


def test_desc_nodes_refuse_fewer_than_three():
    """Chebyshev-Lobatto needs the two endpoints and one interior point."""
    with pytest.raises(ValueError, match="n_rho"):
        _nodes(2)


def test_desc_convert_block_refuses_mixed_parity():
    """A cosine-of-the-combined-angle field cannot carry a sine partner."""
    rho = _nodes(9)
    # (l, m, n) = (0, 0, 0) is cos·cos; (1, -1, 0) is sin·cos. Together they
    # mix the two parities of (m theta - n zeta), which a stellarator-
    # symmetric file never does.
    modes = np.array([[0, 0, 0], [1, -1, 0]], dtype=int)
    coef = np.array([1.0, 0.3])
    with pytest.raises(ValueError, match="mixes cos and sin"):
        _convert_block(modes, coef, nfp=1, rho=rho, name="R")


def test_desc_identically_zero_lambda_is_a_zero_block(tmp_path):
    """An up-down-symmetric field stores no lambda; the block is still well-formed."""
    _, _, st = _torus(tmp_path, lam_amplitude=0.0)
    assert st["LA"]["coef"].shape[0] >= 1
    got = evaluate(st["LA"], RHO, TWO_PI * THETA, TWO_PI * ZETA / NFP)
    assert np.abs(got).max() <= mrx.eps(64)


def test_desc_spline_profile_is_resampled(tmp_path):
    """A DESC ``SplineProfile`` is read at the sample nodes, not refused."""
    import h5py

    path, _, _ = _torus(tmp_path, read=False)
    rho_knots = np.linspace(0.0, 1.0, 5)
    values = -0.9 - 0.15 * rho_knots ** 2
    with h5py.File(path, "a") as fh:
        eq = fh["_equilibria"]["0"]
        del eq["_iota"]
        node = eq.create_group("_iota")
        node["__class__"] = np.bytes_(b"desc.profiles.SplineProfile")
        node["_params"] = values.astype(np.float64)
        node["_knots"] = rho_knots.astype(np.float64)
    iota = read_desc(path)["profiles"]["iota"]
    assert np.isfinite(iota).all()
    # The cubic through five samples of a quadratic recovers it tightly.
    assert abs(float(iota[0]) - values[0]) < 1e-6


def test_desc_unknown_profile_class_names_the_way_out(tmp_path):
    """A profile class we do not parse is refused with the class name."""
    import h5py

    path, _, _ = _torus(tmp_path, read=False)
    with h5py.File(path, "a") as fh:
        eq = fh["_equilibria"]["0"]
        del eq["_iota"]["__class__"]
        eq["_iota"]["__class__"] = np.bytes_(b"desc.profiles.NotARealProfile")
    with pytest.raises(NotImplementedError, match="NotARealProfile"):
        read_desc(path)


def test_desc_empty_family_is_refused(tmp_path):
    """``/_equilibria`` with no numbered members is not a DESC output."""
    import h5py

    path, _, _ = _torus(tmp_path, n_family=0, read=False)
    with h5py.File(path, "r") as fh:
        with pytest.raises(ValueError, match="holds no equilibrium"):
            _equilibrium(fh, path)


def test_desc_missing_pressure_is_read_as_zero(tmp_path):
    """A vacuum file that stores no pressure profile is p = 0, not an error."""
    import h5py

    path, _, _ = _torus(tmp_path, read=False)
    with h5py.File(path, "a") as fh:
        eq = fh["_equilibria"]["0"]
        del eq["_pressure"]
        eq["_pressure"] = np.bytes_(b"None")
    pressure = read_desc(path)["profiles"]["pressure"]
    assert np.allclose(pressure, 0.0)


def test_desc_refuses_a_file_that_is_not_desc(tmp_path):
    """A non-HDF5 file, and an HDF5 file with no equilibrium family."""
    import h5py

    plain = tmp_path / "not.h5"
    plain.write_text("this is not HDF5\n")
    with pytest.raises(ValueError, match="not an HDF5 file"):
        read_desc(str(plain))
    empty = str(tmp_path / "empty.h5")
    with h5py.File(empty, "w") as fh:
        fh["something"] = 1
    with pytest.raises(ValueError, match="_equilibria"):
        read_desc(empty)


@pytest.mark.parametrize("path", [SOLOVEV, DSHAPE])
def test_desc_fixtures_reproduce_their_own_fourier_zernike_series(path):
    """The tracked examples, against DESC's series in its OWN product form.

    This is the conversion's acceptance test: the reference is built from
    the file's raw ``(l, m, n)`` table as
    ``Z_l^|m|(rho) P_m(theta) T_n(zeta)``, the form DESC evaluates, with no
    product-to-sum step anywhere in it. Agreement therefore exercises the
    identities of :func:`mrx.desc._split_weights` and nothing else; what is
    left is the spline refit error, which
    :data:`mrx.desc.NODES_PER_L` sets.
    """
    import h5py

    from mrx.desc import _equilibrium

    st = read_desc(path)
    with h5py.File(path, "r") as fh:
        eq = _equilibrium(fh, path)
        raw = {f: (np.asarray(eq[f"_{f}_basis/_modes"][()]),
                   np.asarray(eq[f"_{f}_lmn"][()])) for f in "RZL"}
    zeta = np.linspace(0.0, 2 * np.pi / st["nfp"], 5)          # physical radians
    for blk, field in (("X1", "R"), ("X2", "Z"), ("LA", "L")):
        modes, coef = raw[field]
        ell, m, n = modes[:, 0], modes[:, 1], modes[:, 2]
        radial = _zernike_radial(RHO[:, None], ell[None, :], m[None, :])
        poloidal = np.where(m >= 0, np.cos(np.abs(m) * THETA[:, None]),
                            np.sin(np.abs(m) * THETA[:, None]))
        toroidal = np.where(n >= 0, np.cos(np.abs(n) * st["nfp"] * zeta[:, None]),
                            np.sin(np.abs(n) * st["nfp"] * zeta[:, None]))
        want = np.einsum("rk,tk,zk,k->rtz", radial, poloidal, toroidal, coef)
        got = evaluate(st[blk], RHO, THETA, zeta)
        scale = max(float(np.abs(want).max()), 1e-12)
        assert np.abs(got - want).max() / scale < 1e-6, (path, field)


def test_desc_dispatch_reaches_the_reader_and_the_initial_field():
    """``.h5`` routes through ``geometry_kind`` to a divergence-free field.

    The coarsest mesh that still maps: this is the wiring, not the physics.
    """
    from mrx.geometry import build_sequence
    from mrx.initial_conditions import initial_field

    assert geometry_kind(SOLOVEV) == "desc"
    assert geometry_nfp(SOLOVEV) == 1 and geometry_nfp(SOLOVEV, nfp=2) == 2
    assert read_equilibrium(SOLOVEV)["kind"] == "desc"

    seq, _ = build_sequence(SOLOVEV, (6, 8, 4), 2)
    B, info = initial_field(seq)
    assert info["kind"] == "desc" and info["nfp"] == 1
    # SOLOVEV is stored with iota = 1 exactly, at every radius
    assert abs(info["iota_axis"] - 1.0) < 1e-6 and abs(info["iota_edge"] - 1.0) < 1e-6
    assert info["div"] <= mrx.eps(1024) and info["wall_discarded"] <= mrx.eps(1024)
    assert np.isfinite(np.asarray(B)).all()


def _has_desc():
    """Whether DESC is importable, for the tests that need it live.

    Returns:
        ``True`` if ``import desc`` succeeds.
    """
    import importlib.util
    return importlib.util.find_spec("desc") is not None


#: Current-constrained examples DESC ships, cheapest first. These store
#: ``_iota = None``: the transform is an output of the solve, so it is the
#: one thing the pure-``h5py`` read cannot recover.
CURRENT_CONSTRAINED = ("DSHAPE_CURRENT", "precise_QA", "HSX", "NCSX")


def _current_constrained_example():
    """A shipped DESC example whose iota has to be recomputed.

    Returns:
        The path of the first of :data:`CURRENT_CONSTRAINED` present in the
        installed DESC's ``examples/``, or ``None`` if none is.
    """
    import os

    import desc
    root = os.path.join(os.path.dirname(desc.__file__), "examples")
    for name in CURRENT_CONSTRAINED:
        path = os.path.join(root, f"{name}_output.h5")
        if os.path.exists(path):
            return path
    return None


@pytest.mark.parametrize("path", [SOLOVEV, DSHAPE])
def test_desc_parse_matches_desc_itself(path):
    """What MRX parsed against what DESC computes, the convention anchor.

    Reading DESC's source settles the basis; only DESC running settles that
    its ``theta`` and ``zeta`` are the angles MRX thinks they are. Skipped
    where DESC is absent, which is every CI run.
    """
    pytest.importorskip("desc", reason="the live cross-check needs DESC installed")
    from desc.grid import Grid
    from desc.io import load

    st = read_desc(path)
    eq = load(path)
    eq = eq[-1] if hasattr(eq, "__len__") else eq
    rho, theta = np.array([0.11, 0.4, 0.77, 1.0]), np.array([0.3, 2.1, 4.4])
    zeta = np.array([0.05, 0.5 * np.pi / st["nfp"]])
    nodes = np.array([[r, t, z] for r in rho for t in theta for z in zeta])
    data = eq.compute(["R", "Z", "lambda"], grid=Grid(nodes, sort=False, jitable=True))
    for blk, key in (("X1", "R"), ("X2", "Z"), ("LA", "lambda")):
        want = np.asarray(data[key]).reshape(len(rho), len(theta), len(zeta))
        got = evaluate(st[blk], rho, theta, zeta)
        scale = max(float(np.abs(want).max()), 1e-12)
        assert np.abs(got - want).max() / scale < 1e-6, (path, key)
