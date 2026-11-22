"""MRX against an exact equilibrium: the Solov'ev tokamak of :mod:`test.solovev`.

The wout holds the closed-form surfaces of the Solov'ev equilibrium (A = 0) in their straight-field-line angle, its transform
and its pressure, so every quantity MRX computes from it has an exact counterpart. On the graded production space at
p = 2, on two meshes whose cells differ by the factor 4/3 in r and theta, with 6 cells in zeta (the field does not
depend on zeta, and the slab mass model needs five):

- the start field, the 2-form of the file's fluxes, is the Solov'ev field to the discretisation, converging at order 3,
- its weak current is the projection of the Solov'ev current onto the Dirichlet 1-forms to the discretisation,
- the flux label is the normalised toroidal flux at first order, and the transform measured on it is the Solov'ev
  transform at order 3,
- ten Newton steps from the start field lower the energy at every step, take the force residual to the round-off of
  the precision and leave the field as close to the Solov'ev field as the start was. The discrete equilibrium
  approximates the exact one as well as the projection does.

The errors are relative L2 errors over the plasma volume at the quadrature points, measured 2026-11-21 in float64.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.flux_label import ANISOTROPY
from mrx.geometry import build_sequence
from mrx.nullspace import compute_nullspaces
from mrx.precision import DTYPE, eps
from mrx.relaxation.initial_conditions import initial_field
from mrx.relaxation.loop import TimeStepper, initial_state, relax
from mrx.relaxation.physics import current
from mrx.relaxation.shells import label
from mrx.relaxation.twist import iota_fit
from test.solovev import Solovev

#: The two meshes per field period (one field period, the whole torus), p = 2, graded.
COARSE, FINE = (14, 24, 6), (18, 32, 6)
#: The ratio of their cells in r (12 and 16, the clamped splines less p) and in theta.
H_RATIO = 4.0 / 3.0
#: 1.5x the errors measured on (COARSE, FINE) in float64: the start field 8.52e-4, 3.30e-4, the weak current against
#: the projection of the Solov'ev current 6.19e-3, 1.78e-3, the flux label (rms of s) 2.31e-3, 1.71e-3, the transform
#: (largest error on s in 0.05 .. 0.95 over the largest transform) 7.19e-5, 2.75e-5. In float32 the fitted transform
#: stops at 1.7e-4 and 2.4e-4, the round-off of its fit.
BANDS = {"B": (1.3e-3, 5.0e-4), "J": (9.3e-3, 2.7e-3), "s": (3.5e-3, 2.6e-3),
         "iota": (1.1e-4, 4.1e-5) if DTYPE == np.float64 else (3.6e-4, 3.6e-4)}
#: Lower bounds of the orders between the meshes, measured 3.30 (B), 4.33 (J), 1.05 (s), 3.34 (iota). The order of
#: the transform is checked in float64 only.
ORDERS = {"B": 2.5, "J": 2.0, "s": 0.9, **({"iota": 2.5} if DTYPE == np.float64 else {})}
STEPS = 10


@pytest.fixture(scope="module")
def solovev(tmp_path_factory):
    """``(eq, {mesh: (seq, B)})``: the equilibrium, and on each mesh the sequence of its wout with the harmonic
    forms and the start field."""
    eq = Solovev(angle="straight")
    path = eq.write_wout(str(tmp_path_factory.mktemp("solovev") / "wout_solovev.nc"), ns=201, mpol=40)
    out = {}
    for ns in (COARSE, FINE):
        seq = build_sequence(path, ns, 2, symmetry="stellarator", graded=True)
        compute_nullspaces(seq, verbose=False)
        out[ns] = (seq, initial_field(seq)[0])
    return eq, out


def _points(seq):
    """The Cartesian quadrature points and their volume weights."""
    return np.asarray(jax.vmap(seq.map)(seq.quad.x)), np.asarray(seq.quad.w * seq.jacobian_j)


def _field_error(eq, seq, B):
    """The relative L2 error of the 2-form ``B`` against the Solov'ev field."""
    X, w = _points(seq)
    Bh = np.asarray(jax.vmap(Pushforward(DiscreteFunction(B, seq.odd.basis_2, seq.odd.E(2)), seq.map, 2))(seq.quad.x))
    Be = eq.field(X)
    return float(np.sqrt(np.sum(w[:, None] * (Bh - Be) ** 2) / np.sum(w[:, None] * Be ** 2)))


def _order(errors):
    return float(np.log(errors[0] / errors[1]) / np.log(H_RATIO))


def _check(name, errors):
    print(f"\n  {name}: {errors[0]:.3e} on {COARSE}, {errors[1]:.3e} on {FINE}, order {_order(errors):.2f}")
    assert errors[0] < BANDS[name][0] and errors[1] < BANDS[name][1]
    if name in ORDERS:
        assert _order(errors) > ORDERS[name]


def test_the_start_field_is_the_solovev_field(solovev):
    """``||B_h - B|| / ||B||`` within its band on both meshes, falling at order 3."""
    eq, seqs = solovev
    _check("B", [_field_error(eq, *seqs[ns]) for ns in (COARSE, FINE)])


def test_the_current_is_solovevs(solovev):
    """The weak current ``M_1 J = D_1^T B`` against the L2 projection of the Solov'ev ``mu0 J`` onto the
    Dirichlet 1-forms, in the mass norm. The weak current has no tangential part on the wall, where the Solov'ev
    current has one, so it is compared with the projection onto its own space, not with the current itself."""
    eq, seqs = solovev
    errors = []
    for ns in (COARSE, FINE):
        seq, B = seqs[ns]
        odd = seq.odd

        def J_exact(x):
            X = seq.map(x)
            R, Z = jnp.hypot(X[0], X[1]), X[2]
            cos, sin = X[0] / R, X[1] / R
            psi_R = 2.0 * eq.a * R * Z ** 2 + 4.0 * eq.b * R * (R ** 2 - 1.0)
            psi_Z = 2.0 * eq.a * R ** 2 * Z + 2.0 * eq.d * Z
            psi = eq.a * R ** 2 * Z ** 2 + eq.b * (R ** 2 - 1.0) ** 2 + eq.d * Z ** 2
            dF = -2.0 * eq.d / jnp.sqrt(eq.F0 ** 2 - 4.0 * eq.d * psi)
            J_R, J_phi, J_Z = -dF * psi_Z / R, -(eq.c * R + 2.0 * eq.d / R), dF * psi_R / R
            return jnp.array([J_R * cos - J_phi * sin, J_R * sin + J_phi * cos, J_Z])
        J_proj = odd.M[1].solve(odd.load(J_exact, 1))
        errors.append(float(odd.l2_norm(current(B, seq) - J_proj, 1) / odd.l2_norm(J_proj, 1)))
    _check("J", errors)


def test_the_label_is_the_toroidal_flux_and_its_transform_is_solovevs(solovev):
    """The flux label of the start field (anisotropy :data:`~mrx.flux_label.ANISOTROPY`, solved to 1e-9) against the
    normalised toroidal flux of the Solov'ev surface through each quadrature point, in the root mean square, and the
    transform fitted on the label (:func:`mrx.relaxation.twist.iota_fit`, degree 6) against the Solov'ev transform on
    ``s`` in 0.05 .. 0.95. MRX reads the wout's angles reversed, so its transform is the negative of the file's.

    The field is tangent to the logical surfaces, so the temperature is a function of ``r`` alone and the anisotropy
    costs nothing. The label's error is that of :func:`mrx.flux_label.label_knots`: a radial quadrature layer gets the
    flux of the points before it plus half its own, and a Gauss node does not sit in the middle of its weight (at 0.41
    and 0.59 of it for the outer nodes of three), so the label is off by a fraction of a cell, at first order. That
    rule alone, with ``s = r^2`` on uniform cells, gives the measured errors to 1 %."""
    eq, seqs = solovev
    S = np.linspace(0.05, 0.95, 19)
    errors = {"s": [], "iota": []}
    for ns in (COARSE, FINE):
        seq, B = seqs[ns]
        X, w = _points(seq)
        s_q, T, info = label(seq, B, ANISOTROPY, None, 1e-9)
        assert int(info) > 0
        s_q = np.asarray(s_q)
        errors["s"].append(float(np.sqrt(np.sum(w * (s_q - eq.s_at(X)) ** 2) / np.sum(w))))
        iota = -np.asarray(iota_fit(seq, B, T, s_q, 6)(S))
        errors["iota"].append(float(np.abs(iota - eq.iota(S)).max() / eq.iota(S).max()))
    _check("s", errors["s"])
    _check("iota", errors["iota"])


def test_newton_steps_reach_the_discrete_equilibrium_next_to_solovevs(solovev):
    """Ten Newton steps on the coarse mesh from the start field. The energy never rises by more than its round-off,
    and every step taken above the floor of the squared force residual, ``1e3 eps^2``, moves along its descent
    direction. Below it the direction is round-off. The residual falls from 3.9e-7 to 4.6e-15 in float64 and to 4.1e-13
    in float32 (bound: by 1e6 or to the floor), and the end field is as close to the Solov'ev field as the start (8.25e-4
    against 8.52e-4 in both precisions, bound 1.05 times the start's error)."""
    eq, seqs = solovev
    seq, B = seqs[COARSE]
    ts = TimeStepper(seq=seq, newton=True)
    res = relax(initial_state(B, ts), ts, steps=STEPS, chunk=STEPS, verbose=False)
    resid = np.asarray(res.trace["resid"], dtype=float)
    dt_star = np.asarray(res.trace["dt_star"], dtype=float)
    dE = np.asarray(res.trace["dE"], dtype=float)
    floor = 1e3 * eps() ** 2
    start, end = _field_error(eq, seq, B), _field_error(eq, seq, res.state.B_n)
    print(f"\n  resid {' '.join(f'{r:.1e}' for r in resid)}\n  dt* {' '.join(f'{d:.2f}' for d in dt_star)}"
          f"\n  max dE / E0 {dE.max() / res.E0:+.1e}, error against Solov'ev {start:.3e} -> {end:.3e}")
    assert np.all(dE < eps() * res.E0)
    assert np.all(dt_star[resid > floor] > 0.0)
    assert resid[-1] < max(1e-6 * resid[0], floor)
    assert end < 1.05 * start
