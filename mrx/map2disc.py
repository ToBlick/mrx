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

Limitation (documented, not hidden): a plain trapezoidal Nystrom
evaluation loses accuracy very near the boundary. The outermost Zernike
ring sits at ``rho = 1``, where ``f(e^{i theta}) = gamma(e^{i theta})``
exactly, so that ring is set from the boundary samples and never solved.
The next ring in is at ``cos(pi / M)``; its accuracy is governed by
``n_boundary``.
"""
from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mrx.precision import DTYPE


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
        ValueError: If ``samples`` is not of shape ``(n, 2)`` with ``n >= 8``.
    """
    pts = jnp.asarray(samples, dtype=DTYPE)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"samples must have shape (n, 2), got {tuple(pts.shape)}")
    if pts.shape[0] < 8:
        raise ValueError(f"need at least 8 boundary samples, got {pts.shape[0]}")
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
        tol: float = 1e-10, max_iter: int = 25) -> jnp.ndarray:
    """Newton inversion of ``g`` at a batch of disc targets.

    Args:
        g: Harmonic map ``(x, y) -> (xi, eta)``.
        targets: Disc coordinates of shape ``(n, 2)``.
        x0: Initial guesses of shape ``(n, 2)``.
        tol: Residual tolerance on ``|g(x) - target|``.
        max_iter: Maximum Newton steps per point.

    Returns:
        Physical coordinates of shape ``(n, 2)``.
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

        # Fixed-step ``fori_loop`` avoids a ``while_loop`` carry whose
        # residual dtype would follow ``jax_enable_x64`` rather than
        # the working ``DTYPE``.
        return jax.lax.fori_loop(0, max_iter, body, xy)

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
        ell, m = zernike_indices(self.M)
        Z = zernike_eval(ell, m, rho, theta)
        return self.coeffs @ Z

    def jacobian_determinant(self, rho, theta) -> jnp.ndarray:
        """Jacobian determinant of ``f_h`` as a map of ``(xi, eta)``.

        ``(xi, eta) = (rho cos theta, rho sin theta)`` are Cartesian
        coordinates on the disc. A positive determinant is the discrete
        invertibility certificate of the paper's section 2.3 (v).

        Args:
            rho: Radial coordinate in ``[0, 1]``.
            theta: Polar angle in radians.

        Returns:
            Scalar ``det Df_h``.
        """
        def f_cart(xi_eta):
            xi, eta = xi_eta
            r = jnp.hypot(xi, eta)
            th = jnp.atan2(eta, xi)
            return self(r, th)

        xi_eta = jnp.array([rho * jnp.cos(theta), rho * jnp.sin(theta)])
        return jnp.linalg.det(jax.jacfwd(f_cart)(xi_eta))


def fit_disc_map(curve: BoundaryCurve, M: int = 15,
                 newton_tol: float = 1e-10,
                 newton_max_iter: int = 25,
                 rho_safe: float = 0.85) -> ZernikeMap:
    """Fit the inverse harmonic map in a Zernike basis of degree ``M``.

    Algorithm (paper, section 2.3): invert ``g`` by Newton at the
    concentric nodes, except the outermost ring ``rho = 1`` which is
    set to ``gamma(theta)`` exactly, then solve the square interpolation
    system for the Zernike coefficients.

    Nodes with ``rho_safe < rho < 1`` sit too close to the boundary for
    a plain Nystrom evaluation of ``g``. Those are inverted at
    ``rho_safe`` and linearly blended to the exact boundary value, which
    is exact for any elliptical cross-section (``f`` is linear in
    ``rho``) and a first-order approximation otherwise.

    Args:
        curve: Boundary, parametrised by the disc angle.
        M: Maximum Zernike degree. Must be a positive integer.
        newton_tol: Residual tolerance of the Newton inversion.
        newton_max_iter: Maximum Newton steps per interior node.
        rho_safe: Largest disc radius at which ``g`` is inverted directly.

    Returns:
        The discrete inverse map ``f_h``.
    """
    if M < 1:
        raise ValueError(f"M must be a positive integer, got {M}")
    g = harmonic_map(curve)
    rho, theta = concentric_nodes(M)
    center = jnp.mean(curve.samples, axis=0)
    on_bdry = rho >= 1.0 - 1e-10
    close = jnp.logical_and(rho > rho_safe, jnp.logical_not(on_bdry))
    t = theta / (2.0 * jnp.pi)
    xy_bdry = jax.vmap(curve.interpolate)(t)
    rho_inv = jnp.where(close, rho_safe, rho)
    xy0 = center[None, :] + (0.9 * rho_inv[:, None]) * (xy_bdry - center[None, :])
    targets = jnp.stack([rho_inv * jnp.cos(theta),
                         rho_inv * jnp.sin(theta)], axis=1)
    xy_int = invert_harmonic_map(g, targets, xy0, tol=newton_tol,
                                 max_iter=newton_max_iter)
    blend = (rho - rho_safe) / (1.0 - rho_safe)
    xy_close = (1.0 - blend)[:, None] * xy_int + blend[:, None] * xy_bdry
    xy = jnp.where(on_bdry[:, None], xy_bdry,
                   jnp.where(close[:, None], xy_close, xy_int))
    ell, m = zernike_indices(M)
    V = zernike_eval(ell, m, rho, theta)
    coeffs = jnp.linalg.solve(V, xy).T
    return ZernikeMap(M=M, coeffs=coeffs)


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
        M: int = 15,
        n_boundary: int = 256,
        n_zeta: int = 8) -> Callable:
    """Automated logical-to-physical map from poloidal cross-sections.

    Applies :func:`fit_disc_map` at ``n_zeta`` equally spaced toroidal
    planes and interpolates the Zernike coefficients periodically. The
    returned map uses the GVEC convention
    ``(R cos 2 pi zeta / nfp, -R sin 2 pi zeta / nfp, Z)``, so input
    ``zeta`` in ``[0, 1]`` is one field period.

    Args:
        boundary_of_zeta: ``boundary_of_zeta(zeta)`` returns either an
            ``(n_boundary, 2)`` array of counterclockwise ``(R, Z)``
            samples or a :class:`BoundaryCurve`. The argument ``zeta``
            is the logical toroidal angle in ``[0, 1]``.
        nfp: Number of field periods. Must be a positive integer.
        M: Maximum Zernike degree of each poloidal fit.
        n_boundary: Sample count used when ``boundary_of_zeta`` returns
            a callable that needs sampling. Ignored if the callable
            already returns ``n`` samples.
        n_zeta: Number of toroidal planes. ``1`` is an axisymmetric map.

    Returns:
        Logical-to-physical map ``(r, theta, zeta) -> (X, Y, Z)``.

    Raises:
        ValueError: If ``nfp`` or ``n_zeta`` is not a positive integer.
    """
    if nfp <= 0:
        raise ValueError(f"nfp must be a positive integer, got {nfp}")
    if n_zeta <= 0:
        raise ValueError(f"n_zeta must be a positive integer, got {n_zeta}")

    planes = []
    for i in range(n_zeta):
        zeta = i / n_zeta
        raw = boundary_of_zeta(zeta)
        if isinstance(raw, BoundaryCurve):
            curve = raw
        else:
            pts = jnp.asarray(raw, dtype=DTYPE)
            if pts.ndim != 2:
                raise ValueError(
                    "boundary_of_zeta must return (n, 2) samples or a "
                    f"BoundaryCurve, got shape {tuple(pts.shape)}")
            curve = boundary_from_samples(pts)
        planes.append(fit_disc_map(curve, M=M).coeffs)
    coeffs_planes = jnp.stack(planes, axis=0)

    def F(x):
        r, θ, ζ = x
        coeffs = _fourier_interp(coeffs_planes, ζ)
        fh = ZernikeMap(M=M, coeffs=coeffs)
        RZ = fh(r, 2.0 * jnp.pi * θ)
        R, Z = RZ[0], RZ[1]
        φ = 2.0 * jnp.pi * ζ / nfp
        return jnp.array([R * jnp.cos(φ), -R * jnp.sin(φ), Z])

    return F
