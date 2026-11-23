"""Relax to the pressure that measured data give, with the profile a Bayesian posterior refitted on the flux label
before every chunk. This is ``scripts/relax.py`` with an outer loop of its own: it takes every flag of relax.py, and
``--pressure.target`` is the JSON file of the data.

The data are values ``d_i`` of the pressure with standard deviations ``sigma_i``, measured in volumes about points
``y_i`` in space (``{"points": [[R, Z, phi_deg], ...], "kernel_m": [w_across, w_along], "p_pa": [...]}``, metres and
degrees) or on flux surfaces (``{"rho": [...], "p_pa": [...]}``, ``rho = sqrt(s)``), with an optional ``"sigma_pa"``
(pascal, one value or one per point). A datum at a point is the mean of the pressure over its instrument kernel
``K_i``, a Gaussian about ``y_i`` with the standard deviation ``w_across`` across the flux surfaces (along the
gradient of the field's label at each quadrature point) and ``w_along`` along them (one pair or one per point): the
observation is the weak functional ``int K_i p dV / int K_i dV``
(:func:`kernel_rows`), integrated at the volume quadrature points of the field's label, linear in the profile's
coefficients. Before every chunk the rows are taken on the field's label and the chunk runs with the posterior mean as
the prescribed pressure ``p*(s)``. The loop is a fixed point of the labels: the labels give the fit, the fit moves the
field and the field gives new labels. A kernel must be wide enough for the quadrature to resolve it (its record holds
the effective number of quadrature points of each kernel). Since the pressure is a function of the label, a kernel
narrow across the surfaces and wide along them observes what a small instrument volume does, and the quadrature
resolves it. The data file states both widths. Data on flux surfaces have their
labels and are fitted once, less their value at the wall. Both are multiplied by ``--pressure.scale``. Everything
else of relax.py works as there: a current target, the transform's loop, the mesh motion, a seed, a restart.

The model (:func:`posterior`). The profile is ``p(s) = sum_k c_k phi_k(s)`` in the profile space
(:func:`mrx.relaxation.shells.profile_basis`, cubic splines in ``s``, ``--pressure.cells`` cells), with ``p(1) = 0``.
Every profile of the space is smooth in ``s``, so it is regular at the magnetic axis without a condition there. The
data are ``d_i = p(s_i) + e_i`` with independent Gaussian errors of standard deviation ``sigma_i``. The prior is
Gaussian with the density ``exp(-tau / 2 int_0^1 (p'')^2 ds)``, the curvature of the function in ``s``
(:func:`curvature_form`). It is flat only along ``1 - s``, which the data determine. At fixed labels the model is
linear and the posterior is Gaussian with the covariance ``H^-1``, ``H = A^T S^-1 A + tau R``, and the mean
``H^-1 A^T S^-1 d``, with ``A`` the observations of the splines (their kernel means, or their values on the data's
surfaces), ``S`` the errors' covariance and ``R`` the curvature form.
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

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sl
from scipy.interpolate import BSpline

from mrx.geometry import map_jacobian_at
from mrx.relaxation.shells import PROFILE_DEGREE, profile_basis, profile_cells, profile_knots, shell_table
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
    prior weight, ``sigma`` the data's standard deviations (given, or estimated by the evidence), ``A`` the
    observations of the splines (one row per datum) and ``d`` the data's values."""
    a: np.ndarray
    cov: np.ndarray
    tau: float
    sigma: np.ndarray
    A: np.ndarray
    d: np.ndarray

    def __call__(self, s):
        return basis_table(profile_basis(self.a.size), s).T @ jnp.asarray(self.a)

    def std(self, s):
        """The posterior standard deviation of ``p(s)``."""
        A = _splines(self.a.size, s)
        return np.sqrt(np.einsum("ij,jk,ik->i", A, self.cov, A))

    def chi2(self):
        """The squared misfit of the mean's observations at the data in units of their standard deviations, per
        point."""
        return float(np.mean(((self.A @ self.a - self.d) / self.sigma) ** 2))

    def record(self, n=41):
        """The fit for a run's record: the mean ``p`` and its standard deviation ``std`` at ``n`` labels ``s``
        uniform in ``sqrt(s)``, the coefficients, ``tau``, ``sigma``, ``chi2``, the data's values and the mean's
        observations of them."""
        s = np.linspace(0.0, 1.0, n) ** 2
        return dict(s=s.tolist(), p=(_splines(self.a.size, s) @ self.a).tolist(), std=self.std(s).tolist(),
                    coefficients=self.a.tolist(), tau=self.tau, sigma=self.sigma.tolist(), chi2=self.chi2(),
                    data=self.d.tolist(), observed=(self.A @ self.a).tolist())


def posterior(A_i, p_i, cells, sigma=None):
    """The posterior of the profile ``p(s)`` in the profile space of ``cells`` cells, given the data ``p_i`` with the
    observations ``A_i`` ``(n, K)`` of the ``K`` splines (row ``i`` the functional of datum ``i`` applied to each
    spline: :func:`kernel_rows`, or the splines' values on a datum's flux surface), as a :class:`ProfileFit`.

    The model: the profiles of the space with ``p(1) = 0`` (the wall's coefficient zero), the data
    ``p_i = (A c)_i + e_i`` with independent Gaussian errors of standard deviation ``sigma`` (one value or one per
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
    A = np.asarray(A_i, dtype=np.float64)[:, :-1]
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
                      np.sqrt(var / w), np.asarray(A_i, dtype=np.float64), d)


def _images(seq, X):
    """The physical points ``X`` (quadrature points of ``seq``) over the whole torus: the stellarator images of a
    half-period sequence (``(x, y, z) -> (x, -y, -z)``) and the ``nfp`` rotations about the Z axis, ``(n_img, n, 3)``.
    The maps are linear, so the normals of a symmetric label's surfaces go along the same way."""
    parts = [X] + ([X * jnp.array([1.0, -1.0, -1.0])] if seq.half_period else [])
    out = []
    for k in range(int(seq.nfp)):
        c, s = np.cos(2 * np.pi * k / seq.nfp), np.sin(2 * np.pi * k / seq.nfp)
        Rot = jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        out += [P @ Rot.T for P in parts]
    return jnp.stack(out)


def surface_normals(seq, T):
    """The unit normals ``(n_q, 3)`` of the label's level surfaces at the quadrature points of ``seq``: the physical
    gradient ``DPhi G^-1 dT`` of the label's temperature ``T`` (an even 0-form), normalised. The label is a monotone
    function of ``T``, so its gradient has this direction."""
    even = seq.even
    dT = even.evaluate_at_quadrature(even.G[0] @ T, 1)
    g = jnp.einsum("qij,qjk,qk->qi", map_jacobian_at(seq.map, seq.quad.x), seq.metric_inv_jkl, dT)
    return g / jnp.linalg.norm(g, axis=1, keepdims=True)


def kernel_rows(seq, s_q, n_q, X_q, y, width, cells):
    """``(A, record)``: the observations ``A_ik = int K_i phi_k(s) dV / int K_i dV`` of the ``K`` splines of the
    profile space by the instrument kernels of the data, ``K_i`` the Gaussian about the point ``y_i`` (Cartesian,
    ``(n, 3)``) with the standard deviations ``width`` ``(n, 2)`` across the flux surfaces (along their normal ``n_q``
    at each quadrature point, :func:`surface_normals`) and along them. The integrals are the volume quadrature of
    ``seq`` over the whole torus (``X_q`` the physical quadrature points), with the label ``s_q`` at the quadrature
    points. No point is located on the map and no label is read off at a point.

    The record holds per datum the effective number of quadrature points in its kernel, ``(sum K)^2 / sum K^2`` with
    the quadrature's volume weights, and the kernel's mean label, ``labels`` (for plots)."""
    wJ = seq.quad.w * seq.jacobian_j
    values, indices = shell_table(cells + PROFILE_DEGREE, s_q)
    Xi, Ni = _images(seq, X_q), _images(seq, n_q)

    def one(args):
        yi, wi = args
        e = Xi - yi
        across = jnp.sum(e * Ni, axis=-1)
        K = jnp.sum(jnp.exp(-0.5 * ((across / wi[0]) ** 2 + (jnp.sum(e * e, axis=-1) - across ** 2) / wi[1] ** 2)),
                    axis=0) * wJ
        norm = jnp.sum(K)
        row = jnp.zeros(cells + PROFILE_DEGREE, K.dtype).at[indices].add(values * K[:, None]) / norm
        return row, norm ** 2 / jnp.sum(K ** 2), jnp.sum(K * s_q) / jnp.sum(K)
    A, n_eff, s_mean = jax.lax.map(one, (jnp.asarray(y), jnp.asarray(width)))
    return np.asarray(A, dtype=np.float64), dict(kernel_points=np.asarray(n_eff).tolist(),
                                                 labels=np.asarray(s_mean).tolist())


class DataFit:
    """The fit of the data in the JSON file ``path`` (times ``scale``, on the profile space of ``cells`` cells,
    :func:`mrx.relaxation.shells.profile_cells` of the sequence unless given) as
    relax.py's pressure target: called with the field ``B`` and its label temperature ``T`` it returns the posterior
    (:class:`ProfileFit`, which is the profile) and the record ``{"fit": ...}``. Data at points are observed by their
    kernels on the field's label (:func:`kernel_rows`), data on flux surfaces are fitted once."""

    def __init__(self, path, scale=1.0, cells=None):
        with open(path) as fh:
            f = json.load(fh)
        self.path, self.cells = path, cells
        self.d = scale * MU0 * np.asarray(f["p_pa"], dtype=float)
        self.sigma = scale * MU0 * np.asarray(f["sigma_pa"], dtype=float) if "sigma_pa" in f else None
        self.surfaces, self.rho = None, None
        if "rho" in f:
            self.rho = np.asarray(f["rho"], dtype=float)
            self.labels = (self.rho ** 2).tolist()
            return
        pts = np.asarray(f["points"], dtype=float)
        phi = np.radians(pts[:, 2])
        self.y = np.stack([pts[:, 0] * np.cos(phi), pts[:, 0] * np.sin(phi), pts[:, 1]], axis=1)
        self.width = np.broadcast_to(np.asarray(f["kernel_m"], dtype=float), (self.y.shape[0], 2))
        self.map, self.X_q = None, None

    def __call__(self, seq, B, T):
        from mrx.flux_label import enclosed_flux
        cells = profile_cells(seq) if self.cells is None else self.cells
        if self.rho is not None:
            if self.surfaces is None:       # data on flux surfaces are fitted once
                self.surfaces = posterior(_splines(cells + PROFILE_DEGREE, self.rho ** 2),
                                          self.d - np.interp(1.0, self.rho, self.d), cells, self.sigma)
            fit, rec = self.surfaces, dict(labels=self.labels)
        else:
            if self.map is not seq.map:
                # the physical quadrature points of the installed map, again after every move of the mesh
                self.X_q, self.map = jax.vmap(seq.map)(seq.quad.x), seq.map
            s_q = enclosed_flux(seq, T, seq.odd.evaluate_at_quadrature(B, 2))
            A, rec = kernel_rows(seq, s_q, surface_normals(seq, T), self.X_q, self.y, self.width, cells)
            fit = posterior(A, self.d, cells, self.sigma)
        print(f"[fit] tau {fit.tau:.3e}, sigma {fit.sigma[0] / MU0:.3e} Pa, chi^2 {fit.chi2():.3f} per point"
              + (f", kernel points >= {min(rec['kernel_points']):.1f}" if "kernel_points" in rec else ""), flush=True)
        return fit, dict(fit=dict(fit.record(), **rec))


def plot_band(rec, path):
    """The posterior mean of the record ``rec`` (:meth:`ProfileFit.record`) with its one and two standard deviation
    bands over ``sqrt(s)``, in kPa, and the data at their kernels' mean labels (on flux surfaces, at their labels)
    with their standard deviations."""
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
