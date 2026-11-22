"""The Bayesian fit of pressure data of ``scripts/pressure_fit.py``, an outer loop written outside the library.

The posterior of a profile given data (``posterior``) is calibrated with the evidence's prior weight, it follows the
data where they determine the profile and the prior where none lie, and a user's data file at points in space reaches
it through the kernel means of the splines on the field's label. The script is loaded from its path, it is no module of the
package.
"""
import importlib.util
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import BSpline

from mrx.flux_label import ANISOTROPY
from mrx.relaxation.shells import PROFILE_CELLS, PROFILE_DEGREE, label, profile_knots

_spec = importlib.util.spec_from_file_location(
    "pressure_fit", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "pressure_fit.py")
pf = importlib.util.module_from_spec(_spec)
sys.modules["pressure_fit"] = pf          # its dataclass looks its module up there
_spec.loader.exec_module(pf)

#: The coverage of a Gaussian at one and two standard deviations. The posterior's, measured: 0.705 and 0.967.
COVER_1, COVER_2 = 0.683, 0.954
#: The mean against the least-squares fit at a noise of 1e-9: 1.4e-11 measured (the evidence's weight at the lower end
#: of its window, 1e-16 of the data's total precision).
MEAN_LS = 1e-10
#: The posterior precision against the prior's on profiles that vanish where the data lie: 2.2e-15 measured.
PRIOR_PRECISION = 1e-10
#: The standard deviation beyond the data over its mean on them: 4.2 measured.
WIDEN = 3.0
#: The data file's fit against the file's profile and its coverage at two standard deviations: 1.61e-2 and 1.00
#: measured with kernels across / along the flux surfaces (2026-11-21).
DATA_ERROR, DATA_COVER = 2e-2, 0.85
#: The fewest effective quadrature points in a kernel of the data file: 29.8 measured.
KERNEL_POINTS = 10.0


def _posterior_draws(s_i, truth, level, draws, seed=0):
    """The posteriors of ``draws`` noisy copies of the data ``truth(s_i)``, noise ``level`` times the truth's
    maximum."""
    rng = np.random.default_rng(seed)
    sigma = level * truth(0.0)
    A = BSpline(profile_knots(PROFILE_CELLS + PROFILE_DEGREE), np.eye(PROFILE_CELLS + PROFILE_DEGREE), PROFILE_DEGREE)(s_i)
    return [pf.posterior(A, truth(s_i) + sigma * rng.standard_normal(s_i.size), sigma) for _ in range(draws)]


def test_profile_posterior():
    """The posterior of a profile given data in the default profile space (16 cells, 18 unknowns), with the truth
    ``(1 - s)^1.5`` (data at 30 labels uniform in ``sqrt(s)``, as a chord through the axis gives them).

    Calibrated: over 200 draws of 5 % noise with the evidence's weight, the truth lies within one posterior standard
    deviation at 0.705 of 100 labels uniform in ``sqrt(s)`` and draws and within two at 0.967, against
    COVER_1 and COVER_2 of a Gaussian. The data decide where they determine the profile: with 60 labels and a noise
    of 1e-9, the mean is the least-squares fit to 1.4e-11 relative. The prior decides where no data
    lie: with data at ``s < 0.25`` only, a profile ``v`` that vanishes there has the posterior precision
    ``v^T C^-1 v = tau v^T R v`` of the prior alone, to round-off (2.2e-15), and the standard deviation at
    ``s = 0.8`` is 4.2 times its mean on the data's range. Float32 gives the same numbers. 3 s on four CPU cores
    (2026-11-19).
    """
    def truth(s):
        return (1.0 - s) ** 1.5
    fits = _posterior_draws(np.linspace(0.03, 0.97, 30) ** 2, truth, 0.05, 200)
    s = (np.linspace(0.0, 1.0, 101) ** 2)[:-1]          # p(1) = 0 holds exactly, the standard deviation is zero there
    z = np.array([(np.asarray(f(jnp.asarray(s))) - truth(s)) / f.std(s) for f in fits])
    cover_1, cover_2 = float(np.mean(np.abs(z) < 1.0)), float(np.mean(np.abs(z) < 2.0))

    s_dense = np.linspace(0.03, 0.97, 60) ** 2
    (fit,) = _posterior_draws(s_dense, truth, 1e-9, 1)
    K = PROFILE_CELLS + PROFILE_DEGREE
    A = BSpline(profile_knots(K), np.eye(K), PROFILE_DEGREE)(s_dense)[:, :-1]
    a_ls = np.append(np.linalg.lstsq(A, fit.d, rcond=None)[0], 0.0)
    mean_ls = float(np.linalg.norm(fit.a - a_ls) / np.linalg.norm(a_ls))

    (core,) = _posterior_draws(np.linspace(0.03, 0.5, 30) ** 2, truth, 0.05, 1)
    lo = int(np.searchsorted(profile_knots(K), 0.25))     # the splines that vanish on s <= 0.25 start here
    v = np.zeros(K - 1)
    v[lo:] = np.random.default_rng(1).standard_normal(v.size - lo)
    R = pf.curvature_form(K)[:-1, :-1]
    prior_precision = abs(float(v @ np.linalg.solve(core.cov[:-1, :-1], v)) / (core.tau * float(v @ R @ v)) - 1.0)
    widen = float(core.std(np.array([0.8]))[0] / np.mean(core.std(np.linspace(0.03, 0.5, 30) ** 2)))
    print(f"\n  coverage 1 sd {cover_1:.3f}, 2 sd {cover_2:.3f}, mean against least squares {mean_ls:.1e}, "
          f"prior precision {prior_precision:.1e}, std at s 0.8 over the data's {widen:.1f}")
    assert abs(cover_1 - COVER_1) < 0.05 and abs(cover_2 - COVER_2) < 0.03
    assert mean_ls < MEAN_LS
    assert prior_precision < PRIOR_PRECISION
    assert widen > WIDEN


def test_pressure_data_at_points(seq, b0, tmp_path):
    """A user's data file of pressure measured in volumes about points in space with error bars
    (``scripts/pressure_fit.py --pressure.target data.json``): 30 points on two lines through the axis in the plane
    ``phi = 0``, Gaussian kernels of standard deviations 1 cm across the flux surfaces and 10 cm along them (resolved
    by the fixture's quadrature: at least 29.8 effective points each), the kernel means of the
    file's pressure on the file's field's label with 2 % noise. ``DataFit`` observes the splines by the kernels on the
    label and fits the posterior. Its mean misses the file's profile by 1.61e-2 relative, and the truth lies within two
    standard deviations at all of the labels. 10 s on an H200 after the fixture (2026-11-21).
    """
    p_r = seq.equilibrium["profiles"]["pressure"]                        # Pa, a function of r = sqrt(s)
    r = np.linspace(0.05, 0.95, 15)
    x = jnp.asarray(np.concatenate([np.stack([r, np.zeros(15), np.zeros(15)], 1),
                                    np.stack([r, np.full(15, 0.5), np.zeros(15)], 1)]))
    X = np.asarray(jax.vmap(seq.map)(x))
    s_q, T, _ = label(seq, b0, ANISOTROPY)

    def truth(s):
        return p_r(np.sqrt(s)) - p_r(1.0)
    K = PROFILE_CELLS + PROFILE_DEGREE
    s_dense = np.linspace(0.0, 1.0, 2001)
    c_truth = np.linalg.lstsq(BSpline(profile_knots(K), np.eye(K), PROFILE_DEGREE)(s_dense), truth(s_dense),
                              rcond=None)[0]
    width = np.array([0.01, 0.1])
    A, rec = pf.kernel_rows(seq, s_q, pf.surface_normals(seq, T), jax.vmap(seq.map)(seq.quad.x), X,
                            np.broadcast_to(width, (X.shape[0], 2)))
    sigma = 0.02 * truth(0.0)
    p = A @ c_truth + sigma * np.random.default_rng(0).standard_normal(X.shape[0])
    path = tmp_path / "data.json"
    path.write_text(json.dumps(dict(points=np.stack([np.hypot(X[:, 0], X[:, 1]), X[:, 2],
                                                     np.degrees(np.arctan2(X[:, 1], X[:, 0]))], 1).tolist(),
                                    kernel_m=width.tolist(), p_pa=p.tolist(), sigma_pa=sigma)))
    fit, _ = pf.DataFit(str(path))(seq, b0, T)
    s = np.linspace(0.0, 1.0, 101)[:-1] ** 2
    mean, std = np.asarray(fit(jnp.asarray(s))) / pf.MU0, fit.std(s) / pf.MU0
    error = float(np.linalg.norm(mean - truth(s)) / np.linalg.norm(truth(s)))
    cover = float(np.mean(np.abs(mean - truth(s)) < 2.0 * std))
    print(f"\n  kernel labels {min(rec['labels']):.3f} .. {max(rec['labels']):.3f}, kernel points "
          f"{min(rec['kernel_points']):.1f} .. {max(rec['kernel_points']):.1f}, error {error:.2e}, within 2 sd "
          f"{cover:.2f}, chi^2 {fit.chi2():.2f} per point, tau {fit.tau:.2e}")
    assert min(rec["kernel_points"]) > KERNEL_POINTS
    assert error < DATA_ERROR
    assert cover > DATA_COVER
