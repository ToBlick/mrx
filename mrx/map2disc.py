"""Native JAX map2disc: harmonic maps from a simply-connected domain to the disc.

Implements Babin, Hindenlang, Maj and Koeberl, *Construction of an
invertible mapping to boundary conforming coordinates for arbitrarily
shaped toroidal domains*, Plasma Phys. Control. Fusion **67** 035005
(2025). Two Dirichlet--Laplace problems are solved by a periodic
trapezoidal Nystrom discretisation of the double-layer operator; the
inverse is fitted in a Zernike basis of degree ``M``. There is no
dependency on the ``map2disc`` or ``pyBIE2D`` packages.

The paper's eq. (2a)--(3b) are

    lap xi = 0  in Omega,   xi = cos theta  on dOmega,
    lap eta = 0 in Omega,   eta = sin theta on dOmega,

where ``theta = arg gamma^{-1}`` is the parameter of the Jordan curve
``gamma : S^1 -> dOmega``. The harmonic map ``g = xi + i eta`` is a
diffeomorphism onto the unit disc; its inverse ``f = g^{-1}`` is the
boundary-conforming coordinate map.

The one thing that governs accuracy is the BOUNDARY resolution, not the
Zernike degree. A plain trapezoidal Nystrom evaluation loses accuracy
very near the boundary; the outermost Zernike ring sits at ``rho = 1``,
where ``f(e^{i theta}) = gamma(e^{i theta})`` exactly and nothing is
solved, but the next ring in is at ``cos(pi / M)``, which crowds the wall
as ``M`` grows. :func:`fit_disc_map` therefore refines the boundary until
its nodes are resolved and certifies the result, rather than taking a
resolution on faith: measured on the Landreman-Paul QA cross-sections, a
node seven boundary spacings from the wall is recovered to ``3e-7`` and
one two spacings in produces Zernike coefficients of order ``1e31``.
With the refinement the fit converges spectrally in ``M`` -- on the QA
cross-section at ``zeta = 1/4``, ``1.6e-3`` at ``M = 4`` down to
``2.6e-10`` at ``M = 12``.
"""
from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mrx.precision import DTYPE, sqrt_eps


# ---------------------------------------------------------------------------
# Periodic boundary curve
# ---------------------------------------------------------------------------

class BoundaryCurve(eqx.Module):
    """Equally spaced samples of a periodic Jordan curve and its derivatives.

    The parameter ``t`` lives in ``[0, 1)``. Samples are ordered
    counterclockwise so that ``(y', -x')`` is the outward normal.
    ``d1`` and ``d2`` are spectral (FFT) derivatives with respect to ``t``.
    """

    samples: jnp.ndarray
    d1: jnp.ndarray
    d2: jnp.ndarray

    @property
    def n(self) -> int:
        """Number of boundary samples."""
        return int(self.samples.shape[0])

    @property
    def speed(self) -> jnp.ndarray:
        """``|gamma'(t)|`` at each sample."""
        return jnp.hypot(self.d1[:, 0], self.d1[:, 1])

    def interpolate(self, t: jnp.ndarray) -> jnp.ndarray:
        """Spectral (DFT) interpolation of the periodic samples at ``t``.

        Exact for a curve whose Fourier content fits in ``n`` modes
        (ellipses, finite Fourier boundaries). ``t`` is wrapped into
        ``[0, 1)``.

        Args:
            t: Parameter in ``[0, 1)``, or any real (wrapped).

        Returns:
            Point ``(x, y)`` on the Fourier interpolant of the samples.
        """
        return _fourier_interp(self.samples, t)

    def resample(self, n: int) -> "BoundaryCurve":
        """The same curve on ``n`` equally spaced parameters.

        Trigonometric resampling, so this is EXACT for a curve whose
        Fourier content fits in ``min(self.n, n)`` modes -- a VMEC or GVEC
        cross-section is a finite Fourier series, so upsampling it adds no
        error. :func:`fit_disc_map` uses this to raise the boundary
        resolution until its near-boundary nodes are resolved.

        Args:
            n: Number of samples of the returned curve.

        Returns:
            The resampled curve, with its derivatives recomputed.

        Raises:
            ValueError: If ``n < 8``.
        """
        if n == self.n:
            return self
        if n < 8:
            raise ValueError(f"need at least 8 boundary samples, got {n}")
        d1, d2 = _spectral_derivatives(_fourier_resample(self.samples, n))
        return BoundaryCurve(samples=_fourier_resample(self.samples, n), d1=d1, d2=d2)


def _fourier_resample(samples: jnp.ndarray, m: int) -> jnp.ndarray:
    """Trigonometric resampling of real periodic ``samples`` to ``m`` points.

    Upsampling zero-pads the spectrum, which reproduces the band-limited
    interpolant exactly. For an even input length the Nyquist bin is
    halved on the way up, so that it is the symmetric ``cos(pi n t)`` mode
    rather than a one-sided exponential.

    Args:
        samples: Shape ``(n, ...)``, equally spaced over one period.
        m: Target number of samples.

    Returns:
        Array of shape ``(m,) + samples.shape[1:]``.
    """
    n = samples.shape[0]
    if m == n:
        return samples
    hat = jnp.fft.rfft(samples, axis=0)
    if n % 2 == 0 and m > n:
        hat = hat.at[n // 2].multiply(0.5)
    keep = m // 2 + 1
    if keep <= hat.shape[0]:
        hat = hat[:keep]
    else:
        pad = jnp.zeros((keep - hat.shape[0],) + hat.shape[1:], dtype=hat.dtype)
        hat = jnp.concatenate([hat, pad], axis=0)
    out = jnp.fft.irfft(hat, n=m, axis=0) * (m / n)
    return out.astype(samples.dtype)


def _spectral_derivatives(samples: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """First and second periodic derivatives of ``samples`` by FFT.

    Args:
        samples: Shape ``(n, 2)``, equally spaced in a parameter of period 1.

    Returns:
        ``(d1, d2)`` with the same shape, ``d/dt`` and ``d^2/dt^2``.
    """
    n = samples.shape[0]
    # ``fftfreq(n)`` is the integer mode over ``n``; ``d/dt`` of a period-1
    # series multiplies mode ``k`` by ``2 pi i k``.
    freq = 2.0 * jnp.pi * (jnp.fft.fftfreq(n) * n)
    hat = jnp.fft.fft(samples, axis=0)
    d1 = jnp.fft.ifft(1j * freq[:, None] * hat, axis=0).real
    d2 = jnp.fft.ifft(-(freq**2)[:, None] * hat, axis=0).real
    return d1.astype(samples.dtype), d2.astype(samples.dtype)


def boundary_from_samples(samples: jnp.ndarray) -> BoundaryCurve:
    """Build a :class:`BoundaryCurve` from counterclockwise ``(x, y)`` samples.

    Args:
        samples: Array of shape ``(n, 2)``. The first and last points are
            not repeated; the curve is closed by periodicity.

    Returns:
        A curve with spectrally differentiated ``gamma'`` and ``gamma''``.

    Raises:
        ValueError: If ``samples`` is not of shape ``(n, 2)`` with ``n >= 8``,
            or if the samples run clockwise. The double-layer kernel here
            takes ``(y', -x')`` as the OUTWARD normal, which is only true
            counterclockwise; a clockwise curve otherwise returns a
            plausible-looking but wrong harmonic map.
    """
    pts = jnp.asarray(samples, dtype=DTYPE)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"samples must have shape (n, 2), got {tuple(pts.shape)}")
    if pts.shape[0] < 8:
        raise ValueError(f"need at least 8 boundary samples, got {pts.shape[0]}")
    area = 0.5 * float(jnp.sum(pts[:, 0] * jnp.roll(pts[:, 1], -1)
                               - jnp.roll(pts[:, 0], -1) * pts[:, 1]))
    if area <= 0.0:
        raise ValueError(
            f"boundary samples must run counterclockwise (signed area {area:.3e} "
            "<= 0); reverse them with samples[::-1]")
    d1, d2 = _spectral_derivatives(pts)
    return BoundaryCurve(samples=pts, d1=d1, d2=d2)


def boundary_from_fourier(
        rcos: jnp.ndarray, rsin: jnp.ndarray,
        zcos: jnp.ndarray, zsin: jnp.ndarray,
        n_boundary: int = 256) -> BoundaryCurve:
    """Fourier series ``R(theta)``, ``Z(theta)`` sampled as a poloidal curve.

    ``R = sum_m rcos[m] cos(2 pi m t) + rsin[m] sin(2 pi m t)`` and
    likewise for ``Z``, with ``t`` in ``[0, 1)``.

    Args:
        rcos: Cosine coefficients of ``R``, index equal to the mode number.
        rsin: Sine coefficients of ``R`` (same length as ``rcos``).
        zcos: Cosine coefficients of ``Z``.
        zsin: Sine coefficients of ``Z``.
        n_boundary: Number of equally spaced samples.

    Returns:
        The sampled :class:`BoundaryCurve`.
    """
    rcos = jnp.asarray(rcos, dtype=DTYPE)
    rsin = jnp.asarray(rsin, dtype=DTYPE)
    zcos = jnp.asarray(zcos, dtype=DTYPE)
    zsin = jnp.asarray(zsin, dtype=DTYPE)
    n_modes = int(rcos.shape[0])
    t = jnp.arange(n_boundary, dtype=DTYPE) / n_boundary
    angles = 2.0 * jnp.pi * jnp.arange(n_modes, dtype=DTYPE)[:, None] * t[None, :]
    R = rcos @ jnp.cos(angles) + rsin @ jnp.sin(angles)
    Z = zcos @ jnp.cos(angles) + zsin @ jnp.sin(angles)
    return boundary_from_samples(jnp.stack([R, Z], axis=1))


# ---------------------------------------------------------------------------
# Double-layer Nystrom solve
# ---------------------------------------------------------------------------

def double_layer_matrix(curve: BoundaryCurve) -> jnp.ndarray:
    """Trapezoidal Nystrom matrix of the 2-D double-layer operator.

    Off-diagonal entries are ``A_jk = K(gamma_j, gamma_k) |gamma'_k| / N``
    with kernel ``K = (n_k · (gamma_j - gamma_k)) / (2 pi |diff|^2)`` and
    outward normal ``n = (y', -x') / |gamma'|``. The removable singularity
    on the diagonal is
    ``A_jj = (x' y'' - y' x'') / (4 pi N |gamma'|^2)``.

    Args:
        curve: Periodic boundary samples with spectral derivatives.

    Returns:
        Dense matrix of shape ``(n, n)``.
    """
    n = curve.samples.shape[0]
    idx = jnp.arange(n)
    diff = curve.samples[:, None, :] - curve.samples[None, :, :]
    # n_k |gamma'_k| = (y'_k, -x'_k), so the speed cancels in A_jk.
    nx = curve.d1[:, 1]
    ny = -curve.d1[:, 0]
    num = nx[None, :] * diff[:, :, 0] + ny[None, :] * diff[:, :, 1]
    dist2 = jnp.sum(diff * diff, axis=2)
    off = num / (2.0 * jnp.pi * dist2 * n)
    # Outward-normal double layer of a constant is -1/2 on the boundary.
    # Enforce that row-sum identity for the diagonal (singularity
    # subtraction): more accurate than the curvature formula, whose sign
    # is ``- (x' y'' - y' x'') / (4 pi N |gamma'|^2)``.
    off = off.at[idx, idx].set(0.0)
    return off.at[idx, idx].set(-0.5 - off.sum(axis=1))


def harmonic_map(curve: BoundaryCurve) -> Callable:
    """Harmonic map ``g : Omega -> D`` as a JAX callable ``(x, y) -> (xi, eta)``.

    Solves ``(-I/2 + A) sigma = f`` once for both Dirichlet data
    ``f = (cos 2 pi t, sin 2 pi t)``, then evaluates the double-layer
    potential in the interior. The parameter of ``curve`` *is* the disc
    angle, so ``g o gamma = Id`` on the boundary by construction of the
    Dirichlet data (paper, section 2).

    Args:
        curve: Counterclockwise Jordan curve, parametrised by the disc
            angle over ``[0, 1)``.

    Returns:
        Callable ``xy -> (xi, eta)``. ``xy`` has shape ``(2,)``.
    """
    A = double_layer_matrix(curve)
    n = curve.samples.shape[0]
    t = jnp.arange(n, dtype=DTYPE) / n
    rhs = jnp.stack([jnp.cos(2.0 * jnp.pi * t), jnp.sin(2.0 * jnp.pi * t)], axis=1)
    system = A - 0.5 * jnp.eye(n, dtype=A.dtype)
    sigma = jnp.linalg.solve(system, rhs)
    samples = curve.samples
    tang = curve.d1

    def g(xy):
        diff = xy[None, :] - samples
        num = tang[:, 1] * diff[:, 0] - tang[:, 0] * diff[:, 1]
        dist2 = jnp.sum(diff * diff, axis=1)
        kern = num / (2.0 * jnp.pi * dist2 * n)
        return kern @ sigma

    return g


# ---------------------------------------------------------------------------
# Zernike basis
# ---------------------------------------------------------------------------

def zernike_count(M: int) -> int:
    """Number of Zernike modes of maximum degree ``M``: ``(M+1)(M+2)/2``."""
    return (M + 1) * (M + 2) // 2


def zernike_indices(M: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Degrees ``l`` and azimuthal orders ``m`` of a degree-``M`` Zernike basis.

    For each ``l = 0, ..., M`` the orders are ``m = -l, -l+2, ..., l``
    (paper, appendix B).

    Args:
        M: Maximum polynomial degree. Must be a non-negative integer.

    Returns:
        Integer arrays ``(l, m)`` of length ``(M+1)(M+2)/2``.
    """
    if M < 0:
        raise ValueError(f"M must be non-negative, got {M}")
    ls, ms = [], []
    for ell in range(M + 1):
        for m in range(-ell, ell + 1, 2):
            ls.append(ell)
            ms.append(m)
    return jnp.asarray(ls), jnp.asarray(ms)


def _jacobi_p(n: int, alpha: float, beta: float, x: jnp.ndarray) -> jnp.ndarray:
    """Jacobi polynomial ``P_n^{alpha, beta}(x)`` by the three-term recurrence.

    Args:
        n: Degree (static Python integer).
        alpha: First Jacobi parameter.
        beta: Second Jacobi parameter.
        x: Evaluation points.

    Returns:
        ``P_n^{alpha, beta}`` at ``x``.
    """
    if n == 0:
        return jnp.ones_like(x)
    p0 = jnp.ones_like(x)
    p1 = 0.5 * ((alpha + beta + 2.0) * x + (alpha - beta))
    if n == 1:
        return p1
    for k in range(2, n + 1):
        a = 2.0 * k * (k + alpha + beta) * (2.0 * k + alpha + beta - 2.0)
        b = (2.0 * k + alpha + beta - 1.0) * (
            (2.0 * k + alpha + beta) * (2.0 * k + alpha + beta - 2.0) * x
            + alpha * alpha - beta * beta)
        c = 2.0 * (k + alpha - 1.0) * (k + beta - 1.0) * (2.0 * k + alpha + beta)
        p0, p1 = p1, (b * p1 - c * p0) / a
    return p1


def zernike_radial(ell: int, m: int, rho: jnp.ndarray) -> jnp.ndarray:
    """Radial Zernike polynomial ``R_l^{|m|}(rho)`` of the paper's eq. (B.2).

    ``R_l^m(rho) = (-1)^{(l-m)/2} rho^m P_{(l-m)/2}^{m, 0}(1 - 2 rho^2)``
    with ``m >= 0`` and ``l - m`` even.

    Args:
        ell: Degree.
        m: Absolute azimuthal order (non-negative).
        rho: Radial coordinate in ``[0, 1]``.

    Returns:
        ``R_l^m`` at ``rho``.
    """
    m = abs(m)
    n = (ell - m) // 2
    sign = -1.0 if (n % 2) else 1.0
    return sign * (rho**m) * _jacobi_p(n, float(m), 0.0, 1.0 - 2.0 * rho**2)


def zernike_eval(ell: jnp.ndarray, m: jnp.ndarray, rho: jnp.ndarray,
                 theta: jnp.ndarray) -> jnp.ndarray:
    """Evaluate every Zernike mode of a ``(l, m)`` list at ``(rho, theta)``.

    ``Z_l^m = R_l^{|m|}(rho) cos(m theta)`` for ``m >= 0`` and
    ``R_l^{|m|}(rho) sin(-m theta)`` for ``m < 0`` (paper, eq. B.1).

    Args:
        ell: Degrees, shape ``(K,)``.
        m: Azimuthal orders, shape ``(K,)``.
        rho: Radial coordinates, any broadcastable shape.
        theta: Polar angles in radians, same shape as ``rho``.

    Returns:
        Values of shape ``rho.shape + (K,)``.
    """
    rho = jnp.asarray(rho)
    theta = jnp.asarray(theta)
    vals = []
    for li, mi in zip(np_int_list(ell), np_int_list(m)):
        R = zernike_radial(li, abs(mi), rho)
        if mi >= 0:
            vals.append(R * jnp.cos(mi * theta))
        else:
            vals.append(R * jnp.sin(-mi * theta))
    return jnp.stack(vals, axis=-1)


def np_int_list(arr) -> list[int]:
    """Materialise a static integer array as a Python list of ints."""
    return [int(v) for v in np.asarray(arr).tolist()]


def _jacobi_batch(n_idx: np.ndarray, alpha: np.ndarray,
                  x: jnp.ndarray) -> jnp.ndarray:
    """``P_{n_k}^{alpha_k, 0}(x)`` for every mode ``k`` in one recurrence.

    :func:`_jacobi_p` runs one recurrence per mode, so a degree-``M`` basis
    costs ``(M+1)(M+2)/2`` separately traced loops. Here the per-mode
    ``alpha`` is carried as an array and the recurrence runs once, to
    ``max(n_idx)``, selecting each mode's degree on the way.

    Args:
        n_idx: Jacobi degrees ``(l - |m|) / 2``, shape ``(K,)``, static.
        alpha: Jacobi parameters ``|m|``, shape ``(K,)``, static.
        x: Evaluation points, any shape; broadcast against the mode axis.

    Returns:
        Values of shape ``x.shape + (K,)``.
    """
    n_idx = np.asarray(n_idx, dtype=int)
    xb = jnp.asarray(x)[..., None]
    a = jnp.asarray(alpha, dtype=xb.dtype)
    sel = jnp.asarray(n_idx)
    p0 = jnp.ones(jnp.broadcast_shapes(xb.shape, a.shape), dtype=xb.dtype)
    p1 = 0.5 * ((a + 2.0) * xb + a)
    out = jnp.where(sel == 0, p0, p1)
    for k in range(2, int(n_idx.max(initial=0)) + 1):
        ak = 2.0 * k * (k + a) * (2.0 * k + a - 2.0)
        bk = (2.0 * k + a - 1.0) * ((2.0 * k + a) * (2.0 * k + a - 2.0) * xb + a * a)
        ck = 2.0 * (k + a - 1.0) * (k - 1.0) * (2.0 * k + a)
        p0, p1 = p1, (bk * p1 - ck * p0) / ak
        out = jnp.where(sel == k, p1, out)
    return out


def zernike_eval_cartesian(ell: jnp.ndarray, m: jnp.ndarray,
                           xi: jnp.ndarray, eta: jnp.ndarray) -> jnp.ndarray:
    """The Zernike modes of :func:`zernike_eval` as polynomials in ``(xi, eta)``.

    ``rho^|m| cos(m theta) = Re[(xi + i eta)^|m|]`` and
    ``rho^|m| sin(|m| theta) = Im[(xi + i eta)^|m|]``, so with
    ``rho^2 = xi^2 + eta^2`` the paper's eq. (B.1)--(B.2) read

        Z_l^m = (-1)^n P_n^{|m|,0}(1 - 2 rho^2) (Re or Im)[(xi + i eta)^|m|],

    a POLYNOMIAL. The polar form goes through ``hypot`` and ``atan2``,
    whose derivatives are undefined at the origin -- ``jacfwd`` of it
    returns NaN at ``rho = 0``, which is the axis of every MRX map. This
    form is smooth there. The two agree wherever both are defined
    (``test_cartesian_and_polar_zernike_agree``).

    Args:
        ell: Degrees, shape ``(K,)``, static.
        m: Azimuthal orders, shape ``(K,)``, static.
        xi: Cartesian disc coordinate ``rho cos theta``.
        eta: Cartesian disc coordinate ``rho sin theta``.

    Returns:
        Values of shape ``xi.shape + (K,)``.
    """
    ell_np = np.asarray(ell, dtype=int)
    m_np = np.asarray(m, dtype=int)
    abs_m = np.abs(m_np)
    n_idx = (ell_np - abs_m) // 2
    xi, eta = jnp.broadcast_arrays(jnp.asarray(xi), jnp.asarray(eta))
    radial = _jacobi_batch(n_idx, abs_m.astype(float),
                           1.0 - 2.0 * (xi**2 + eta**2))
    # ``C_k + i S_k = (xi + i eta)^k`` by the real form of the complex
    # product, so no complex dtype (and no weak-typed ``1j``) is involved.
    cs, sn = jnp.ones_like(xi), jnp.zeros_like(xi)
    powers = [(cs, sn)]
    for _ in range(int(abs_m.max(initial=0))):
        cs, sn = xi * cs - eta * sn, xi * sn + eta * cs
        powers.append((cs, sn))
    angular = jnp.stack([powers[int(a)][0 if mi >= 0 else 1]
                         for a, mi in zip(abs_m, m_np)], axis=-1)
    sign = jnp.asarray((-1.0) ** (n_idx % 2), dtype=radial.dtype)
    return sign * radial * angular


def zernike_basis(M: int) -> Callable:
    """Return ``(rho, theta) -> Z`` evaluating all degree-``M`` Zernike modes.

    Args:
        M: Maximum polynomial degree.

    Returns:
        Callable whose result has a trailing axis of length
        :func:`zernike_count`.
    """
    ell, m = zernike_indices(M)

    def basis(rho, theta):
        return zernike_eval(ell, m, rho, theta)

    return basis


def concentric_nodes(M: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Concentric Zernike interpolation nodes of the paper's appendix B.

    Rings ``i = 0, ..., floor(M/2)`` sit at ``rho_i = cos(i pi / M)``
    with ``2 M + 1 - 4 i`` equally spaced angles. That is
    ``floor(M/2) + 1`` rings (the paper's ``M^2 + 1`` is a typo) and
    exactly ``(M+1)(M+2)/2`` points.

    Args:
        M: Maximum polynomial degree. Must be a positive integer.

    Returns:
        ``(rho, theta)`` of length ``(M+1)(M+2)/2``, ``theta`` in radians.

    Raises:
        ValueError: If ``M < 1``.
    """
    if M < 1:
        raise ValueError(f"M must be a positive integer, got {M}")
    rhos, thetas = [], []
    n_rings = M // 2 + 1
    for i in range(n_rings):
        rho = float(jnp.cos(i * jnp.pi / M))
        n_pts = 2 * M + 1 - 4 * i
        if n_pts <= 0:
            continue
        for j in range(n_pts):
            rhos.append(rho)
            thetas.append(2.0 * jnp.pi * j / n_pts)
    return jnp.asarray(rhos, dtype=DTYPE), jnp.asarray(thetas, dtype=DTYPE)


# ---------------------------------------------------------------------------
# Inverse harmonic map and Zernike fit
# ---------------------------------------------------------------------------

def invert_harmonic_map(
        g: Callable, targets: jnp.ndarray, x0: jnp.ndarray,
        max_iter: int = 25) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Newton inversion of ``g`` at a batch of disc targets.

    Args:
        g: Harmonic map ``(x, y) -> (xi, eta)``.
        targets: Disc coordinates of shape ``(n, 2)``.
        x0: Initial guesses of shape ``(n, 2)``.
        max_iter: Newton steps per point (the loop is fixed-step, so that
            its carry dtype cannot follow ``jax_enable_x64`` away from the
            working ``DTYPE``; the residual below is what says whether the
            steps were enough).

    Returns:
        ``(xy, residual)``: physical coordinates of shape ``(n, 2)`` and
        the final ``|g(xy) - target|`` of each point, shape ``(n,)``.
        Callers must check the residual -- Newton converging says nothing
        about ``g`` itself being accurate at ``xy``, which is what
        :func:`fit_disc_map` separately certifies.
    """
    jac = jax.jacfwd(g)
    eye = jnp.eye(2, dtype=DTYPE)
    shift = jnp.asarray(1e-14, dtype=DTYPE)

    def _one(target, guess):
        tgt = jnp.asarray(target, dtype=DTYPE)
        xy = jnp.asarray(guess, dtype=DTYPE)

        def body(i, xy):
            r = jnp.asarray(g(xy), dtype=DTYPE) - tgt
            J = jnp.asarray(jac(xy), dtype=DTYPE)
            dx = jnp.asarray(jnp.linalg.solve(J + shift * eye, -r), dtype=DTYPE)
            xy_new = jnp.asarray(xy + dx, dtype=DTYPE)
            return jnp.where(jnp.isfinite(xy_new), xy_new, xy)

        xy = jax.lax.fori_loop(0, max_iter, body, xy)
        return xy, jnp.linalg.norm(jnp.asarray(g(xy), dtype=DTYPE) - tgt)

    return jax.vmap(_one)(jnp.asarray(targets, dtype=DTYPE),
                          jnp.asarray(x0, dtype=DTYPE))


class ZernikeMap(eqx.Module):
    """Disc-to-domain map ``f_h(rho, theta) = (x, y)`` in a Zernike basis.

    ``coeffs`` has shape ``(2, K)`` with ``K = (M+1)(M+2)/2``: the first
    row is the ``x`` (or ``R``) expansion, the second the ``y`` (or ``Z``)
    expansion.
    """

    M: int = eqx.field(static=True)
    coeffs: jnp.ndarray

    def __call__(self, rho, theta=None) -> jnp.ndarray:
        """Evaluate the map at polar disc coordinates.

        Args:
            rho: Radial coordinate in ``[0, 1]``, or a length-2 array
                ``(rho, theta)`` if ``theta`` is omitted.
            theta: Polar angle in radians.

        Returns:
            Physical point ``(x, y)``.
        """
        if theta is None:
            rho, theta = rho[0], rho[1]
        return self.at_cartesian(rho * jnp.cos(theta), rho * jnp.sin(theta))

    def at_cartesian(self, xi, eta) -> jnp.ndarray:
        """Evaluate at Cartesian disc coordinates ``(xi, eta)``.

        Args:
            xi: Disc coordinate ``rho cos theta``.
            eta: Disc coordinate ``rho sin theta``.

        Returns:
            Physical point ``(x, y)``.
        """
        ell, m = zernike_indices(self.M)
        return self.coeffs @ zernike_eval_cartesian(ell, m, xi, eta)

    def jacobian_determinant(self, rho, theta) -> jnp.ndarray:
        """Jacobian determinant of ``f_h`` as a map of ``(xi, eta)``.

        ``(xi, eta) = (rho cos theta, rho sin theta)`` are Cartesian
        coordinates on the disc. A positive determinant is the discrete
        invertibility certificate of the paper's section 2.3 (v).
        Differentiating :meth:`at_cartesian` rather than the polar form
        keeps this finite at ``rho = 0``, the axis of every MRX map.

        Args:
            rho: Radial coordinate in ``[0, 1]``.
            theta: Polar angle in radians.

        Returns:
            Scalar ``det Df_h``.
        """
        def f_cart(xi_eta):
            return self.at_cartesian(xi_eta[0], xi_eta[1])

        xi_eta = jnp.array([rho * jnp.cos(theta), rho * jnp.sin(theta)])
        return jnp.linalg.det(jax.jacfwd(f_cart)(xi_eta))


#: Boundary spacings a concentric node must sit from the wall before the
#: plain trapezoidal Nystrom evaluation of ``g`` there is trusted. Measured
#: on the Landreman-Paul QA cross-sections: a node ``7`` spacings in is
#: recovered to ``3e-7``, ``3`` spacings to ``1e-5``, and below ``2`` the
#: Zernike solve blows up to coefficients of order ``1e31``.
GAP_RATIO = 6.0

#: Largest boundary resolution :func:`fit_disc_map` will refine to. The
#: Nystrom matrix is dense, so this caps one solve at ``4096^2``.
N_BOUNDARY_MAX = 4096


def _next_pow2(n: int) -> int:
    """Smallest power of two at least ``n`` (and at least 8)."""
    return max(8, 1 << max(0, int(n - 1)).bit_length())


def _invert_nodes(curve: BoundaryCurve, rho: jnp.ndarray, theta: jnp.ndarray,
                  interior: np.ndarray, max_iter: int
                  ) -> tuple[jnp.ndarray, float, float]:
    """Newton-invert the interior concentric nodes on one boundary resolution.

    Args:
        curve: Boundary at the resolution to use.
        rho: Node radii from :func:`concentric_nodes`.
        theta: Node angles from :func:`concentric_nodes`.
        interior: Indices of the nodes with ``rho < 1`` (the ``rho = 1``
            ring is ``gamma`` exactly and is never solved for).
        max_iter: Newton steps per node.

    Returns:
        ``(xy, residual, gap_ratio)``: the node positions, the largest
        Newton residual, and the distance from the closest interior node
        to the boundary in units of the boundary spacing.
    """
    xy = jax.vmap(curve.interpolate)(theta / (2.0 * jnp.pi))
    if interior.size == 0:
        return xy, 0.0, np.inf
    center = jnp.mean(curve.samples, axis=0)
    rho_in = rho[interior]
    xy0 = center[None, :] + (0.9 * rho_in[:, None]) * (xy[interior] - center[None, :])
    targets = jnp.stack([rho_in * jnp.cos(theta[interior]),
                         rho_in * jnp.sin(theta[interior])], axis=1)
    xy_in, residual = invert_harmonic_map(harmonic_map(curve), targets, xy0,
                                          max_iter=max_iter)
    step = curve.samples - jnp.roll(curve.samples, 1, axis=0)
    spacing = float(jnp.sum(jnp.hypot(step[:, 0], step[:, 1]))) / curve.n
    gap = float(jnp.min(jnp.linalg.norm(
        xy_in[:, None, :] - curve.samples[None, :, :], axis=2)))
    return xy.at[interior].set(xy_in), float(jnp.max(residual)), gap / spacing


def fit_disc_map(curve: BoundaryCurve, M: int = 8,
                 tol: float | None = None,
                 newton_max_iter: int = 25,
                 n_boundary: int | None = None,
                 gap_ratio: float = GAP_RATIO,
                 n_max: int = N_BOUNDARY_MAX) -> ZernikeMap:
    """Fit the inverse harmonic map in a Zernike basis of degree ``M``.

    Algorithm (paper, section 2.3): invert ``g`` by Newton at the
    concentric nodes, except the outermost ring ``rho = 1`` which is
    ``gamma(theta)`` exactly, then solve the square interpolation system
    for the Zernike coefficients.

    The accuracy of that inversion is set by the BOUNDARY resolution, not
    by ``M``: a plain trapezoidal Nystrom evaluation of ``g`` degrades as
    the target approaches the wall, and the outermost interior ring sits
    at ``cos(pi / M)``, which crowds the wall as ``M`` grows. This routine
    therefore refines the boundary itself. Starting from the curve's own
    resolution it repeatedly

    * measures ``gap_ratio``, the closest interior node's distance to the
      wall in boundary spacings, which is proportional to the resolution
      and so predicts the resolution that would suffice, and
    * compares the node positions against the previous, coarser
      resolution,

    and accepts only when the nodes have stopped moving AND the gap is
    resolved. Resampling is trigonometric (:meth:`BoundaryCurve.resample`),
    hence exact for the finite Fourier cross-section of an equilibrium
    file, so refining adds no boundary error of its own.

    Both checks are needed. The Newton residual alone certifies nothing:
    it says the iteration solved ``g_quadrature(x) = target``, not that
    ``g_quadrature`` is ``g``. Under-resolved nodes converge happily to
    the wrong points, and the resulting interpolation -- the system is
    square and exact -- spreads that error over the whole disc.

    One cheap probe sizes the refinement before it starts, so that the
    two resolutions actually compared are consecutive; comparing against
    the probe instead would compare against garbage and force a needless
    extra doubling, and each doubling costs eight times the last.

    Args:
        curve: Boundary, parametrised by the disc angle.
        M: Maximum Zernike degree. Must be a positive integer, and at
            least the boundary's own poloidal mode number or the exact
            ``rho = 1`` ring aliases.
        tol: Relative convergence tolerance: the Newton residual is in
            disc units and tested against it directly, the node movement
            under refinement is physical and tested against ``tol`` times
            the cross-section's half-extent. ``None`` scales with the
            working precision.
        newton_max_iter: Newton steps per interior node.
        n_boundary: Resolution to start refining from. ``None`` starts
            from the curve's own.
        gap_ratio: Boundary spacings an interior node must sit from the
            wall to be trusted (:data:`GAP_RATIO`).
        n_max: Resolution at which refinement gives up and raises.

    Returns:
        The discrete inverse map ``f_h``.

    Raises:
        ValueError: If ``M < 1``, or if the fit is still unresolved at
            ``n_max`` -- the message reports which of the two checks
            failed, so the caller can lower ``M`` or raise ``n_max``
            rather than receive a silently wrong map.
    """
    if M < 1:
        raise ValueError(f"M must be a positive integer, got {M}")
    rho, theta = concentric_nodes(M)
    ell, m = zernike_indices(M)
    V = zernike_eval(ell, m, rho, theta)
    interior = np.flatnonzero(np.asarray(rho) < 1.0 - 1e-10)
    extent = float(jnp.max(jnp.abs(curve.samples - jnp.mean(curve.samples, axis=0))))
    rel = sqrt_eps(1e1) if tol is None else float(tol)
    n = min(max(curve.n if n_boundary is None else int(n_boundary), 8), n_max)

    if interior.size == 0:
        # Every node of a degree-1 basis is on the boundary, where the map
        # is gamma exactly: no quadrature is involved and nothing to refine.
        xy, _, _ = _invert_nodes(curve, rho, theta, interior, newton_max_iter)
        return ZernikeMap(M=M, coeffs=jnp.linalg.solve(V, xy).T)

    # The probe only has to locate the nodes well enough to measure how
    # far they sit from the wall, so it runs a few Newton steps, not the
    # full budget, and its positions are discarded.
    _, _, probe = _invert_nodes(curve.resample(n), rho, theta, interior,
                                max(4, newton_max_iter // 5))
    if probe > 0.0:
        n = min(n_max, max(n, _next_pow2(int(n * gap_ratio / probe)) // 2))

    previous, change = None, np.inf
    while True:
        xy, residual, gap = _invert_nodes(curve.resample(n), rho, theta,
                                          interior, newton_max_iter)
        if previous is not None:
            change = float(jnp.max(jnp.abs(xy - previous)))
        if residual <= rel and gap >= gap_ratio and change <= rel * extent:
            return ZernikeMap(M=M, coeffs=jnp.linalg.solve(V, xy).T)
        if n >= n_max:
            raise ValueError(
                f"map2disc: a degree-{M} fit is not resolved at n_boundary="
                f"{n} (the cap n_max={n_max}). Closest interior node is "
                f"{gap:.1f} boundary spacings from the wall (need "
                f"{gap_ratio:.0f}); Newton residual {residual:.2e} (need "
                f"{rel:.2e}) and node movement under refinement {change:.2e} "
                f"(need {rel * extent:.2e}). Lower M, or raise n_max if the "
                f"memory for a dense {n_max}^2 Nystrom solve is available.")
        previous = xy
        n = min(n_max, 2 * n)


# ---------------------------------------------------------------------------
# 3-D MRX map
# ---------------------------------------------------------------------------

def _fourier_interp(values: jnp.ndarray, zeta: jnp.ndarray) -> jnp.ndarray:
    """Periodic Fourier interpolation of ``values[i]`` at ``zeta`` in ``[0, 1)``.

    Args:
        values: Samples at ``i / n``, ``i = 0, ..., n-1``. Leading axis is
            the periodic one.
        zeta: Evaluation parameter.

    Returns:
        Interpolant at ``zeta``, real part of the DFT interpolant.
    """
    n = values.shape[0]
    if n == 1:
        return values[0]
    chat = jnp.fft.fft(values, axis=0)
    freq = jnp.fft.fftfreq(n) * n
    phase = jnp.exp(2j * jnp.pi * freq * zeta)
    # Broadcast phase over the trailing axes of ``chat``.
    while phase.ndim < chat.ndim:
        phase = phase[..., None]
    return (phase * chat).sum(axis=0).real / n


def map2disc_map(
        boundary_of_zeta: Callable,
        nfp: int = 1,
        M: int = 8,
        n_boundary: int | None = None,
        n_zeta: int = 16,
        sign: float = -1.0,
        **fit_kwargs) -> Callable:
    """Automated logical-to-physical map from poloidal cross-sections.

    Applies :func:`fit_disc_map` at ``n_zeta`` equally spaced toroidal
    planes and interpolates the Zernike coefficients periodically. The
    returned map uses the GVEC convention
    ``(R cos 2 pi zeta / nfp, sign R sin 2 pi zeta / nfp, Z)``, so input
    ``zeta`` in ``[0, 1]`` is one field period.

    ``n_zeta`` is a Nyquist condition on the toroidal Fourier content of
    the boundary, NOT a convergence knob that can be left at a round
    number: a cross-section whose largest toroidal mode per field period
    is ``n_max`` needs ``n_zeta >= 2 n_max + 1``. Landreman-Paul QA has
    ``max|n| / nfp = 8`` and is reproduced to ``1.3e-15`` at ``n_zeta =
    16`` but only to ``3.8e-5`` at ``8``.

    Args:
        boundary_of_zeta: ``boundary_of_zeta(zeta)`` returns either an
            ``(n, 2)`` array of counterclockwise ``(R, Z)`` samples or a
            :class:`BoundaryCurve`. The argument ``zeta`` is the logical
            toroidal angle in ``[0, 1]``.
        nfp: Number of field periods. Must be a positive integer.
        M: Maximum Zernike degree of each poloidal fit.
        n_boundary: Boundary resolution each cross-section is resampled
            to before fitting (:meth:`BoundaryCurve.resample`, exact for
            a finite Fourier boundary). ``None`` leaves each curve at its
            own resolution; :func:`fit_disc_map` refines from there
            either way.
        n_zeta: Number of toroidal planes. ``1`` is an axisymmetric map.
        sign: Toroidal handedness of the Cartesian assembly.
        **fit_kwargs: Forwarded to :func:`fit_disc_map` (``tol``,
            ``gap_ratio``, ``n_max``).

    Returns:
        Logical-to-physical map ``(r, theta, zeta) -> (X, Y, Z)``.

    Raises:
        ValueError: If ``nfp`` or ``n_zeta`` is not a positive integer,
            if a cross-section is not ``(n, 2)`` samples or a
            :class:`BoundaryCurve`, or (from :func:`fit_disc_map`) if a
            cross-section cannot be resolved at degree ``M``.
    """
    if nfp <= 0:
        raise ValueError(f"nfp must be a positive integer, got {nfp}")
    if n_zeta <= 0:
        raise ValueError(f"n_zeta must be a positive integer, got {n_zeta}")

    planes = []
    for i in range(n_zeta):
        raw = boundary_of_zeta(i / n_zeta)
        if isinstance(raw, BoundaryCurve):
            curve = raw
        else:
            pts = jnp.asarray(raw, dtype=DTYPE)
            if pts.ndim != 2:
                raise ValueError(
                    "boundary_of_zeta must return (n, 2) samples or a "
                    f"BoundaryCurve, got shape {tuple(pts.shape)}")
            curve = boundary_from_samples(pts)
        if n_boundary is not None:
            curve = curve.resample(int(n_boundary))
        planes.append(fit_disc_map(curve, M=M, **fit_kwargs).coeffs)
    coeffs_planes = jnp.stack(planes, axis=0)

    def F(x):
        r, θ, ζ = x
        fh = ZernikeMap(M=M, coeffs=_fourier_interp(coeffs_planes, ζ))
        R, Z = fh(r, 2.0 * jnp.pi * θ)
        φ = 2.0 * jnp.pi * ζ / nfp
        return jnp.array([R * jnp.cos(φ), sign * R * jnp.sin(φ), Z])

    return F


def lcfs_boundary(st, nfp: int | None = None,
                  n_boundary: int = 256) -> Callable:
    """``boundary_of_zeta`` for the last closed flux surface of an equilibrium.

    Evaluates the state's ``R`` and ``Z`` series (:class:`mrx.gvec.StateField`)
    at ``rho = 1``. The poloidal angle of the file becomes the disc angle,
    which is what makes the resulting map boundary-conforming in the
    equilibrium's own parametrisation.

    Args:
        st: Parsed state of :func:`mrx.gvec.read_equilibrium`.
        nfp: Field periods; ``None`` takes the file's.
        n_boundary: Samples per cross-section. A VMEC or GVEC boundary is
            a finite Fourier series, so any value above its poloidal mode
            number is exact and :meth:`BoundaryCurve.resample` can raise
            it further without error.

    Returns:
        ``zeta -> BoundaryCurve`` over one field period.
    """
    from mrx.gvec import StateField  # noqa: PLC0415  (mrx.gvec imports this module)

    nfp = st["nfp"] if nfp is None else int(nfp)
    R_field, Z_field = StateField(st["X1"], nfp), StateField(st["X2"], nfp)
    t = jnp.arange(n_boundary, dtype=DTYPE) / n_boundary

    @jax.jit
    def _samples(zeta):
        def one(ti):
            x = jnp.array([jnp.ones_like(ti), ti, zeta])
            return jnp.array([R_field(x), Z_field(x)])
        return jax.vmap(one)(t)

    return lambda zeta: boundary_from_samples(_samples(jnp.asarray(zeta, dtype=DTYPE)))


def nyquist_n_zeta(st, nfp: int | None = None) -> int:
    """Toroidal planes that resolve a state's boundary: ``2 max|n|/nfp + 1``.

    Args:
        st: Parsed state of :func:`mrx.gvec.read_equilibrium`.
        nfp: Field periods; ``None`` takes the file's.

    Returns:
        The smallest alias-free plane count, at least 1.
    """
    nfp = st["nfp"] if nfp is None else int(nfp)
    n_per = max(abs(int(n)) for n in np.asarray(st["X1"]["n"])) / nfp
    return max(1, int(2 * np.ceil(n_per)) + 1)


def map2disc_from_equilibrium(st, seq=None, nfp: int | None = None,
                              M: int = 8, n_zeta: int | None = None,
                              n_boundary: int = 256,
                              **fit_kwargs) -> tuple[Callable, dict]:
    """A map2disc map of an equilibrium's boundary, as ``build_gvec_map`` returns.

    The interior of the file is IGNORED: only the last closed flux surface
    is used, and the interior coordinates are constructed by the harmonic
    map. That is the point of the method -- it needs a boundary and
    nothing else -- but it means the surfaces of this map are not the
    equilibrium's flux surfaces, only its boundary is shared. Use
    :func:`mrx.gvec.build_gvec_map` when the flux-surface labelling
    matters.

    Handedness is measured, not assumed: flipping ``sign`` mirrors the map
    in the ``y = 0`` plane and so negates ``det DF`` pointwise, which
    :func:`mrx.gvec.build_gvec_map` discovers by building both maps but
    which can be read off one of them.

    Args:
        st: Parsed state of :func:`mrx.gvec.read_equilibrium`.
        seq: Unused; present so this is interchangeable with
            :func:`mrx.gvec.build_gvec_map` in :func:`mrx.geometry.build_sequence`.
        nfp: Field periods; ``None`` takes the file's.
        M: Maximum Zernike degree per cross-section.
        n_zeta: Toroidal planes; ``None`` takes :func:`nyquist_n_zeta`.
        n_boundary: Samples per cross-section before refinement.
        **fit_kwargs: Forwarded to :func:`fit_disc_map`.

    Returns:
        ``(F, info)`` with ``info`` the ``nfp``, the measured ``sign``,
        the ``M`` and ``n_zeta`` used, and the sampled ``det_range``.

    Raises:
        RuntimeError: If neither handedness gives a positive Jacobian.
    """
    from mrx.geometry import map_jacobian_at  # noqa: PLC0415  (imports this module)

    nfp = st["nfp"] if nfp is None else int(nfp)
    n_zeta = nyquist_n_zeta(st, nfp) if n_zeta is None else int(n_zeta)
    boundary = lcfs_boundary(st, nfp, n_boundary)
    F = map2disc_map(boundary, nfp=nfp, M=M, n_zeta=n_zeta, sign=-1.0, **fit_kwargs)

    grid = jnp.stack(jnp.meshgrid(jnp.linspace(0.1, 1.0, 4),
                                  jnp.linspace(0.0, 1.0, 5)[:-1],
                                  jnp.linspace(0.0, 1.0, 5)[:-1],
                                  indexing="ij"), axis=-1).reshape(-1, 3)
    det = jnp.linalg.det(map_jacobian_at(F, grid))
    lo, hi = float(jnp.min(det)), float(jnp.max(det))
    if bool(jnp.all(jnp.isfinite(det))) and lo > 0.0:
        return F, {"nfp": nfp, "sign": -1.0, "M": M, "n_zeta": n_zeta,
                   "det_range": (lo, hi)}
    if bool(jnp.all(jnp.isfinite(det))) and hi < 0.0:
        return (map2disc_map(boundary, nfp=nfp, M=M, n_zeta=n_zeta, sign=1.0,
                             **fit_kwargs),
                {"nfp": nfp, "sign": 1.0, "M": M, "n_zeta": n_zeta,
                 "det_range": (-hi, -lo)})
    raise RuntimeError(
        f"{st.get('path', 'equilibrium')}: no handedness gives det DF > 0 for a "
        f"degree-{M} map2disc map; det of the left-handed assembly sampled in "
        f"[{lo:.3e}, {hi:.3e}]")
