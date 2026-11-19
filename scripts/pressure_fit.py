"""Relax to the pressure that measured data give, with the profile a Bayesian posterior refitted on the flux label
before every chunk. This is ``scripts/relax.py`` with an outer loop of its own: it takes every flag of relax.py, and
``--pressure.target`` is the JSON file of the data.

The data are values ``d_i`` of the pressure with standard deviations ``sigma_i``, at points ``y_i`` in space
(``{"points": [[R, Z, phi_deg], ...], "p_pa": [...]}``, metres and degrees) or on flux surfaces (``{"rho": [...],
"p_pa": [...]}``, ``rho = sqrt(s)``), with an optional ``"sigma_pa"`` (pascal, one value or one per point). Data at
points get their labels ``s_i`` on the field before every chunk (:func:`mrx.relaxation.shells.point_labels`), the
profile is fitted to ``(s_i, d_i)`` and the chunk runs with the posterior mean as the prescribed pressure ``p*(s)``.
The loop is a fixed point of the labels: the labels give the fit, the fit moves the field and the field gives new
labels. Data on flux surfaces have their labels and are fitted once, less their value at the wall. Both are
multiplied by ``--pressure.scale``. Everything else of relax.py works as there: a current target, the transform's
loop, the mesh motion, a seed, a restart.

The model (:func:`posterior`). The profile is ``p(s) = sum_k c_k phi_k(s)`` in the profile space
(:func:`mrx.relaxation.shells.profile_basis`, cubic splines in ``s``, ``--pressure.cells`` cells), with ``p(1) = 0``.
Every profile of the space is smooth in ``s``, so it is regular at the magnetic axis without a condition there. The
data are ``d_i = p(s_i) + e_i`` with independent Gaussian errors of standard deviation ``sigma_i``. The prior is
Gaussian with the density ``exp(-tau / 2 int_0^1 (p'')^2 ds)``, the curvature of the function in ``s``
(:func:`curvature_form`). It is flat only along ``1 - s``, which the data determine. At fixed labels the model is
linear and the posterior is Gaussian with the covariance ``H^-1``, ``H = A^T S^-1 A + tau R``, and the mean
``H^-1 A^T S^-1 d``, with ``A`` the splines at the labels, ``S`` the errors' covariance and ``R`` the curvature form.
``tau`` is no setting: it maximises the evidence, the probability of the data given ``tau`` with the profile
integrated out. Without ``"sigma_pa"`` all points get one standard deviation, which the evidence sets together with
``tau``. The fit is computed in float64 whatever the precision of the run.

The posterior holds the labels fixed. In the equilibrium the labels move with the profile, so the uncertainty of the
profile also moves the labels, and the posterior of the equilibrium is wider than the band this script reports.

    python -u scripts/pressure_fit.py --geometry.path data/wout_li383_1.4m.nc --geometry.resolution 48 64 24 \\
        --geometry.graded --budget.steps 300 --pressure.target data.json

Output (in ``--output.out``): that of relax.py, whose ``pressure`` record has at every rebuild and at the end also
``fit`` (:meth:`ProfileFit.record`), and ``band.png``, the posterior mean at the end with its one and two standard
deviation bands and the data at their labels.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import NamedTuple

if __name__ == "__main__":
    # The precision must be in the environment before mrx is imported. The full parse follows.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--geometry.precision", dest="precision", default="float64", choices=("float64", "float32"))
    os.environ["MRX_DTYPE"] = pre.parse_known_args()[0].precision

import jax.numpy as jnp
import numpy as np
import scipy.linalg as sl
from scipy.interpolate import BSpline

from mrx.relaxation.shells import (PROFILE_CELLS, PROFILE_DEGREE, logical_points, point_labels, profile_basis,
                                   profile_knots)
from mrx.spline_bases import basis_table

MU0 = 4e-7 * np.pi

#: The search of the prior weight ``tau`` by the evidence: a grid of EVIDENCE_GRID points per decade over the
#: EVIDENCE_DECADES of the data's total precision ``sum_i 1 / sigma_i^2``, then a parabola through the best grid point
#: and its neighbours in ``log tau``. On point sets of 10 to 60 points, 1 to 10 % noise and 8 to 64 cells (20 draws
#: each) the maximum lay at 10^-5.8 to 10^-2.9 of that precision, or at the upper end, where the data do not tell the
#: profile from the straight one, ``1 - s``, within their noise, and the end is taken. Below, the data decide and the
#: mean tends to the least-squares fit. ``H`` was numerically positive definite from 10^-16 (the lowest tried) up to
#: 10^2.2 on 64 cells and 10^4.8 on 32: above, its flat direction drowns in the round-off of ``tau R`` (2026-11-19).
EVIDENCE_GRID = 20
EVIDENCE_DECADES = (-16, 1)


def _splines(n, s, derivative=0):
    """The ``derivative`` in ``s`` of each of the ``n`` splines of the profile space at the labels ``s``, ``(n_s, n)``,
    in float64."""
    B = BSpline(profile_knots(n), np.eye(n), PROFILE_DEGREE)
    return (B.derivative(derivative) if derivative else B)(np.asarray(s, dtype=np.float64))


def curvature_form(n):
    """``R`` ``(n, n)``: the squared curvature ``c^T R c = int_0^1 (d^2 p / ds^2)^2 ds`` of the profile
    ``p(s) = sum_k c_k phi_k(s)`` of the ``n`` splines of the profile space. The second derivative of a cubic is
    linear on each cell, so Gauss with two points per cell integrates the form exactly. It is a norm of the function,
    not of the coefficients, and vanishes only on the profiles linear in ``s``."""
    T = profile_knots(n)
    lo, hi = T[:-1][np.diff(T) > 0], T[1:][np.diff(T) > 0]
    g, w = np.polynomial.legendre.leggauss(PROFILE_DEGREE - 1)
    s = (0.5 * (lo + hi)[:, None] + 0.5 * (hi - lo)[:, None] * g[None, :]).ravel()
    D = _splines(n, s, 2)
    return D.T @ ((0.5 * (hi - lo)[:, None] * w[None, :]).ravel()[:, None] * D)


class ProfileFit(NamedTuple):
    """The posterior of a pressure profile ``p(s) = sum_k a_k phi_k(s)`` in the profile space given data
    (:func:`posterior`). Calling it gives the posterior mean at ``s``, :meth:`std` its standard deviation. ``a``
    holds the mean's coefficients, ``cov`` their posterior covariance (the wall's coefficient is zero), ``tau`` the
    prior weight, ``sigma`` the data's standard deviations (given, or estimated by the evidence), ``s`` and ``d`` the
    labels and values of the data."""
    a: np.ndarray
    cov: np.ndarray
    tau: float
    sigma: np.ndarray
    s: np.ndarray
    d: np.ndarray

    def __call__(self, s):
        return basis_table(profile_basis(self.a.size), s).T @ jnp.asarray(self.a)

    def std(self, s):
        """The posterior standard deviation of ``p(s)``."""
        A = _splines(self.a.size, s)
        return np.sqrt(np.einsum("ij,jk,ik->i", A, self.cov, A))

    def chi2(self):
        """The squared misfit of the mean at the data in units of their standard deviations, per point."""
        return float(np.mean(((_splines(self.a.size, self.s) @ self.a - self.d) / self.sigma) ** 2))

    def record(self, n=41):
        """The fit for a run's record: the mean ``p`` and its standard deviation ``std`` at ``n`` labels ``s``
        uniform in ``sqrt(s)``, the coefficients, ``tau``, ``sigma``, ``chi2`` and the data's labels and values."""
        s = np.linspace(0.0, 1.0, n) ** 2
        return dict(s=s.tolist(), p=(_splines(self.a.size, s) @ self.a).tolist(), std=self.std(s).tolist(),
                    coefficients=self.a.tolist(), tau=self.tau, sigma=self.sigma.tolist(), chi2=self.chi2(),
                    labels=self.s.tolist(), data=self.d.tolist())


def posterior(s_i, p_i, sigma=None, cells=PROFILE_CELLS):
    """The posterior of the profile ``p(s)`` in the profile space of ``cells`` cells, given the data ``p_i`` at the
    labels ``s_i``, as a :class:`ProfileFit`.

    The model: the profiles of the space with ``p(1) = 0`` (the wall's coefficient zero), the data
    ``p_i = p(s_i) + e_i`` with independent Gaussian errors of standard deviation ``sigma`` (one value or one per
    point), and the Gaussian smoothness prior of density ``exp(-tau c^T R c / 2)``, ``c^T R c`` the squared curvature
    of the profile in ``s`` (:func:`curvature_form`). The prior is flat along ``1 - s``, which the data alone
    determine. At fixed labels the posterior has the mean ``H^-1 A^T S^-1 p`` and the covariance ``H^-1``,
    ``H = A^T S^-1 A + tau R``. ``tau`` maximises the evidence with the flat direction integrated out (the restricted
    likelihood: the determinant of ``tau R`` on its range). Without ``sigma`` all points have one standard deviation,
    set by the evidence together with ``tau`` (it is ``Q / (n - 1)``, ``Q`` the penalised residual and ``n`` the
    number of points). ``tau`` and that ``sigma`` enter the posterior at their maximum, not integrated.

    The edge value is imposed because data rarely reach the wall, and the drop between the last datum and the wall
    sets the level of the relaxation's pressure everywhere (it vanishes on the wall), a constant no shell move
    changes."""
    K = cells + PROFILE_DEGREE
    R = curvature_form(K)[:-1, :-1]
    A = _splines(K, s_i)[:, :-1]
    d = np.asarray(p_i, dtype=np.float64)
    n, k = A.shape
    rank = k - 1                                          # R is flat along 1 - s only
    w = np.ones(n) if sigma is None else np.broadcast_to(np.asarray(sigma, dtype=np.float64), (n,)) ** -2.0
    AtWA, AtWd = A.T @ (w[:, None] * A), A.T @ (w * d)

    def solve(lam):
        """``(z, Q, H)``: the posterior mean, the penalised residual and the factor of ``H`` at the weight ``lam``."""
        H = sl.cho_factor(AtWA + lam * R)
        z = sl.cho_solve(H, AtWd)
        r = d - A @ z
        return z, float(r @ (w * r) + lam * z @ R @ z), H

    def log_evidence(log_lam):
        _, Q, H = solve(np.exp(log_lam))
        log_det = 2.0 * np.sum(np.log(np.diag(H[0])))
        fit = -0.5 * (n - (k - rank)) * np.log(Q) if sigma is None else -0.5 * Q
        return fit + 0.5 * rank * log_lam - 0.5 * log_det

    lo, hi = EVIDENCE_DECADES
    grid = np.log(w.sum()) + np.log(10.0) * np.linspace(lo, hi, (hi - lo) * EVIDENCE_GRID + 1)
    L = np.array([log_evidence(g) for g in grid])
    i = int(np.argmax(L))
    log_lam = grid[i]
    if 0 < i < grid.size - 1 and L[i + 1] - 2.0 * L[i] + L[i - 1] < 0.0:
        # an interior maximum: the vertex of the parabola through it and its neighbours, within them
        log_lam += 0.5 * (grid[1] - grid[0]) * (L[i - 1] - L[i + 1]) / (L[i + 1] - 2.0 * L[i] + L[i - 1])
    z, Q, H = solve(np.exp(log_lam))
    # without sigma the weights are 1 and lam = tau sigma^2, the evidence's sigma^2 is Q / (n - flat directions)
    var = Q / (n - (k - rank)) if sigma is None else 1.0
    P = np.eye(K)[:, :-1]
    return ProfileFit(P @ z, P @ (var * sl.cho_solve(H, np.eye(k))) @ P.T, float(np.exp(log_lam) / var),
                      np.sqrt(var / w), np.asarray(s_i, dtype=np.float64), d)


class DataFit:
    """The fit of the data in the JSON file ``path`` (times ``scale``, on the profile space of ``cells`` cells) as
    relax.py's pressure target: called with the field ``B`` and its label temperature ``T`` it returns the posterior
    (:class:`ProfileFit`, which is the profile) and the record ``{"fit": ...}``. Points are located on the map once
    and again after every move of the mesh, data on flux surfaces are fitted once."""

    def __init__(self, path, scale=1.0, cells=PROFILE_CELLS):
        with open(path) as fh:
            f = json.load(fh)
        self.path, self.cells = path, cells
        self.d = scale * MU0 * np.asarray(f["p_pa"], dtype=float)
        self.sigma = scale * MU0 * np.asarray(f["sigma_pa"], dtype=float) if "sigma_pa" in f else None
        self.surfaces = None
        if "rho" in f:
            rho = np.asarray(f["rho"], dtype=float)
            self.surfaces = posterior(rho ** 2, self.d - np.interp(1.0, rho, self.d), self.sigma, cells)
            return
        pts = np.asarray(f["points"], dtype=float)
        phi = np.radians(pts[:, 2])
        self.X = np.stack([pts[:, 0] * np.cos(phi), pts[:, 0] * np.sin(phi), pts[:, 1]], axis=1)
        self.size = float(np.abs(pts[:, :2]).max())
        self.map, self.x = None, None

    def __call__(self, seq, B, T):
        fit = self.surfaces
        if fit is None:
            if self.map is not seq.map:
                # the points are logical points of the map they were located in, a move of the mesh relocates them
                self.x, miss = logical_points(seq, self.X)
                if float(jnp.max(miss)) > 1e-6 * self.size:
                    raise ValueError(f"{self.path}: a point is outside the domain or the map inversion failed "
                                     f"(|Phi(x) - X| up to {float(jnp.max(miss)):.1e} m)")
                self.map = seq.map
            fit = posterior(np.asarray(point_labels(seq, B, T, self.x)), self.d, self.sigma, self.cells)
        print(f"[fit] tau {fit.tau:.3e}, sigma {fit.sigma[0] / MU0:.3e} Pa, chi^2 {fit.chi2():.3f} per point",
              flush=True)
        return fit, dict(fit=fit.record())


def plot_band(rec, path):
    """The posterior mean of the record ``rec`` (:meth:`ProfileFit.record`) with its one and two standard deviation
    bands over ``sqrt(s)``, in kPa, and the data at their labels with their standard deviations."""
    import matplotlib.pyplot as plt

    from mrx.diagnostics.plotstyle import finish_figure, house_style
    x = np.sqrt(np.asarray(rec["s"]))
    mean, std = (np.asarray(rec[k]) / MU0 / 1e3 for k in ("p", "std"))
    with house_style():
        fig, ax = plt.subplots(figsize=(4.5, 3.0))
        for k, alpha in ((2, 0.2), (1, 0.35)):
            ax.fill_between(x, mean - k * std, mean + k * std, color="C0", alpha=alpha, lw=0, label=f"{k} sd")
        ax.plot(x, mean, color="C0", label="posterior mean")
        ax.errorbar(np.sqrt(rec["labels"]), np.asarray(rec["data"]) / MU0 / 1e3,
                    yerr=np.asarray(rec["sigma"]) / MU0 / 1e3, fmt="o", ms=3, color="black", lw=0.8, label="data")
        ax.set_xlabel(r"$\sqrt{s}$")
        ax.set_ylabel(r"$p$ (kPa)")
        ax.legend()
        finish_figure(fig, path)


def main(cfg):
    from relax import main as relax_main
    pr = cfg.pressure
    if not (pr.target and pr.target.endswith(".json")):
        raise ValueError("--pressure.target must be the JSON file of the data")
    if pr.beta is not None:
        raise ValueError("--pressure.beta rescales a profile shape, measured data carry their own size")
    out = relax_main(cfg, fit=DataFit(pr.target, pr.scale, pr.cells))
    with open(os.path.join(out, "relax.json")) as fh:
        plot_band(json.load(fh)["pressure"]["rebuilds"][-1]["fit"], os.path.join(out, "band.png"))


if __name__ == "__main__":
    import tyro

    from mrx.relaxation.config import RelaxConfig
    main(tyro.cli(RelaxConfig, description=__doc__))
