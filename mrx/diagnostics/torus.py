r"""Flux surfaces of a magnetic field given as a discrete 2-form, solved for one surface at a time.

The entry point is :func:`torus`. It returns the flux surface through a given point of the plane ``zeta = 0``
together with its rotational transform, without tracing a field line. :func:`toroidal_flux` gives the flux a
surface encloses.

A flux surface is a torus :math:`y(\vartheta, \zeta)` in the logical cross-section on which the field lines are
straight, :math:`\vartheta = \vartheta_0 + \iota \zeta`. With the field-line equation
:math:`dy/d\zeta = f(\zeta, y)` of :mod:`mrx.diagnostics.poincare` this means

.. math::
    (\partial_\zeta + \iota\, \partial_\vartheta)\, y = f(\zeta, y),

and the torus and :math:`\iota` are the unknowns. The torus is stored by its values on a uniform grid of the two
angles and the equation is asked to hold at the grid points, with the derivatives taken by FFT. The condition that
the torus passes through the given point where :math:`\vartheta = \zeta = 0` picks the surface and the origin of
:math:`\vartheta`. The parameter :math:`\vartheta` is the straight-field-line poloidal angle that belongs to the
logical toroidal angle :math:`\zeta`.

The equation is solved by Newton's method, and the linear system of every step by GMRES. The preconditioner
writes a correction in the frame of the tangent :math:`\partial_\vartheta y` and a normal scaled by the flux
density :math:`\hat B^\zeta`. In this frame the linearised equation is triangular up to terms of the size of the
error, because the field is divergence-free, and its inverse is two inversions of
:math:`\partial_\zeta + \iota\,\partial_\vartheta`, a division in Fourier space. Nothing is assembled. A Newton
step costs one evaluation of the field and its gradient on the grid and a few FFTs per GMRES iteration.

Only a field with a flux surface through the point has a solution. In an island chain or a chaotic layer the
iteration does not converge, or it converges to a torus that the field crosses, and the ``residual`` of the result
is large. Where :math:`\iota` per field period is close to a
rational :math:`-n/m` with ``m`` and ``n`` on the grid, the Fourier division is by a small number and the solve is
ill-conditioned. The field must have one-signed :math:`\hat B^\zeta`, as for a Poincare section.
"""
from __future__ import annotations

from functools import lru_cache, partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import gmres

from mrx.diagnostics.poincare import TWO_PI, logical_field, require_zeta_parameterisation, to_polar, to_uv
from mrx.precision import sqrt_eps

#: Relative tolerance, vectors per cycle and number of cycles of the GMRES solve inside a Newton step. With 30
#: vectors and 4 cycles the linear solves stalled on grids of 65 points and more and Newton's method took 15 to 40
#: steps where it now takes 4 to 6 (li383 (16, 32, 12) p=2, 2026-11-16).
GMRES_TOL, GMRES_RESTART, GMRES_CYCLES = sqrt_eps(), 80, 6


class Torus(NamedTuple):
    """A flux surface, the result of :func:`torus`."""

    y: jax.Array
    """The surface in the ``(u, v)`` chart of the logical cross-section (:func:`mrx.diagnostics.poincare.to_uv`),
    shape ``(n_theta, n_zeta, 2)``. Entry ``[i, k]`` is the point at the straight-field-line angle ``i / n_theta``
    and the logical toroidal angle ``k / n_zeta``, both in turns, ``zeta`` over one field period."""
    iota: float
    """The rotational transform in poloidal turns per full toroidal turn. It is positive when the field lines
    wind in the direction of increasing logical ``theta``. The field of a VMEC file reads negative."""
    residual: float
    """The root mean square of ``(d/dzeta + iota d/dtheta) y`` minus the field-line direction at the points
    halfway between the grid points, relative to the root mean square of ``(d/dzeta + iota d/dtheta) y``. It is
    small only on a flux surface that the grid resolves, and it falls with the grid at the rate the smoothness
    of the spline field allows."""
    iterations: int
    """The number of Newton steps, positive when the last step was below the tolerance and negative otherwise."""


# ---------------------------------------------------------------------------
# The field-line equation and its invariant density
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def cross_section_flow(field):
    r"""The function ``(zeta, y, coef) -> (dy/dzeta, omega)`` in the ``(u, v)`` chart, for ``field`` of
    :func:`mrx.diagnostics.poincare.logical_field` and its spline coefficients ``coef``
    (``field.coefficients(dof)``, taken once per field). ``omega`` is the density of toroidal flux in the chart,
    :math:`\hat B^\zeta / (2 \pi r)`. The field is divergence-free, so the field-line flow conserves
    ``omega du dv``. The same ``field`` always returns the same function object."""
    def flow(zeta, y, coef):
        r, theta = to_polar(y)
        b = field.at(jnp.array([r, theta, zeta % 1.0]), coef)
        dr, dtheta = b[0] / b[2], b[1] / b[2]
        c, s = jnp.cos(TWO_PI * theta), jnp.sin(TWO_PI * theta)
        f = jnp.array([c * dr - TWO_PI * r * s * dtheta, s * dr + TWO_PI * r * c * dtheta])
        return f, b[2] / (TWO_PI * r)
    return flow


def _on_grid(flow, coef, y, shift):
    """The field-line direction, the flux density and the Jacobian of the direction at the points ``y`` of the
    torus grid, whose toroidal angles are ``(k + shift) / n_zeta``."""
    zeta = (jnp.arange(y.shape[1]) + shift) / y.shape[1]

    def point(z, q):
        def both(q):
            f, omega = flow(z, q, coef)
            return f, (f, omega)
        return jax.jacfwd(both, has_aux=True)(q)

    Df, (f, omega) = jax.vmap(jax.vmap(point), (None, 0))(zeta, y)
    return f, omega, Df


# ---------------------------------------------------------------------------
# Derivatives along the torus
# ---------------------------------------------------------------------------

def _wavenumbers(shape):
    """The Fourier symbols of ``d/dtheta`` and ``d/dzeta`` on a grid of ``shape``, broadcast against it."""
    m = jnp.fft.fftfreq(shape[0], 1.0 / shape[0])
    n = jnp.fft.fftfreq(shape[1], 1.0 / shape[1])
    return TWO_PI * 1j * m[:, None], TWO_PI * 1j * n[None, :]


def _fft(g):
    return jnp.fft.fft2(g, axes=(0, 1))


def _ifft(gh):
    return jnp.fft.ifft2(gh, axes=(0, 1)).real


def _symbol(gh, iota):
    """The symbol of ``d/dzeta + iota d/dtheta`` shaped like the transform ``gh`` of a scalar or a vector."""
    dt, dz = _wavenumbers(gh.shape[:2])
    return (dz + iota * dt).reshape(gh.shape[:2] + (1,) * (gh.ndim - 2))


def _along(g, iota):
    """``(d/dzeta + iota d/dtheta) g``, the derivative of ``g`` along the field lines of the torus."""
    gh = _fft(g)
    return _ifft(_symbol(gh, iota) * gh)


def _solve_along(g, iota):
    """The function of zero mean whose derivative along the field lines is ``g`` minus its mean."""
    gh = _fft(g).at[0, 0].set(0.0)
    return _ifft(gh / _symbol(gh, iota).at[0, 0].set(1.0))


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _shifted(y):
    """The trigonometric interpolant of ``y`` at the points halfway between the grid points."""
    yh = _fft(y)
    dt, dz = _wavenumbers(y.shape[:2])
    phase = jnp.exp(dt / (2 * y.shape[0]) + dz / (2 * y.shape[1]))
    return _ifft(phase[..., None] * yh)


# ---------------------------------------------------------------------------
# Newton's method
# ---------------------------------------------------------------------------

def _frame(flow, coef, y, iota):
    r"""The tangent :math:`T = \partial_\vartheta y` of the torus ``y``, the normal
    :math:`N = T^\perp / (\omega |T|^2)`, the flux density, the Jacobian of the field-line direction, the shear
    :math:`S = \omega\, (D N - f'(y) N) \times N` and the error :math:`D y - f`, all on the grid. ``iota`` is the
    transform per field period and :math:`D = \partial_\zeta + \iota \partial_\vartheta`."""
    dt, _ = _wavenumbers(y.shape[:2])
    T = _ifft(dt[..., None] * _fft(y))
    f, omega, Df = _on_grid(flow, coef, y, 0.0)
    N = jnp.stack([-T[..., 1], T[..., 0]], axis=-1) / (omega * jnp.sum(T * T, axis=-1))[..., None]
    S = omega * _cross(_along(N, iota) - jnp.einsum("...ij,...j->...i", Df, N), N)
    return T, N, omega, Df, S, _along(y, iota) - f


def _linearised(frame, iota, x):
    r"""The linearised equation applied to ``x = (d, diota, kappa)``: the change
    :math:`D d - f'(y) d + \delta\iota\, T + \kappa N` of the error and the two components of ``d`` at the grid
    point ``[0, 0]``. The multiplier :math:`\kappa` takes up the one equation that the others imply, because the
    field has no net flux through a closed surface."""
    T, N, _, Df, _, _ = frame
    d, diota, kappa = x
    r = _along(d, iota) - jnp.einsum("...ij,...j->...i", Df, d) + diota * T + kappa * N
    return r, d[0, 0, 0], d[0, 0, 1]


def _reduced_solve(frame, iota, r):
    r"""The approximate inverse of :func:`_linearised`, exact where the torus is exact and resolved.

    A correction :math:`w_1 T + w_2 N` turns the linearised equation into :math:`D w_2 + \kappa = r_N` and
    :math:`D w_1 + S w_2 + \delta\iota = r_T`, with :math:`r_T, r_N` the components of the right-hand side. This
    uses that :math:`T` solves the homogeneous equation and that :math:`D N - f'(y) N` has no component along
    :math:`N`, which holds because the flow conserves :math:`\omega\, du\, dv`. Each equation is a division in
    Fourier space. The means of the two equations give :math:`\kappa` and :math:`\delta\iota`, and the means of
    :math:`w_1` and :math:`w_2` follow from the two components at ``[0, 0]``."""
    T, N, omega, _, S, _ = frame
    rg, g0, g1 = r
    r_T, r_N = omega * _cross(rg, N), omega * _cross(T, rg)
    w2 = _solve_along(r_N, iota)
    rhs = r_T - S * w2
    w1 = _solve_along(rhs, iota)
    w1_S = _solve_along(-S, iota)
    A = jnp.stack([w1_S[0, 0] * T[0, 0] + N[0, 0], T[0, 0]], axis=1)
    c, c1 = jnp.linalg.solve(A, jnp.stack([g0, g1]) - w1[0, 0] * T[0, 0] - w2[0, 0] * N[0, 0])
    d = (w1 + c * w1_S + c1)[..., None] * T + (w2 + c)[..., None] * N
    return d, jnp.mean(rhs) - c * jnp.mean(S), jnp.mean(r_N)


@partial(jax.jit, static_argnames=("flow",))
def _newton_step(flow, coef, y, iota, x0):
    """One Newton step for the torus ``y`` and its transform per field period ``iota`` through the point ``x0``.
    Returns the new ``y`` and ``iota``, the largest change of ``y`` and the relative residual that GMRES left in
    the linearised equation, which it solves with :func:`_reduced_solve` as the preconditioner."""
    frame = _frame(flow, coef, y, iota)
    b = (-frame[-1], x0[0] - y[0, 0, 0], x0[1] - y[0, 0, 1])
    x, _ = gmres(partial(_linearised, frame, iota), b, M=partial(_reduced_solve, frame, iota),
                 tol=GMRES_TOL, restart=GMRES_RESTART, maxiter=GMRES_CYCLES)
    left = jnp.linalg.norm(_linearised(frame, iota, x)[0] + frame[-1]) / jnp.linalg.norm(frame[-1])
    return y + x[0], iota + x[1], jnp.max(jnp.linalg.norm(x[0], axis=-1)), left


@partial(jax.jit, static_argnames=("flow",))
def _residual(flow, coef, y, iota):
    """The ``residual`` of :class:`Torus` for the torus ``y`` with the transform per field period ``iota``."""
    Dy = _shifted(_along(y, iota))
    f, _, _ = _on_grid(flow, coef, _shifted(y), 0.5)
    return jnp.sqrt(jnp.sum((Dy - f) ** 2) / jnp.sum(Dy ** 2))


@partial(jax.jit, static_argnames=("flow", "shape"))
def _circle(flow, coef, x0, centre, shape):
    """The circle about ``centre`` through ``x0``, the same at every ``zeta``, and the mean rate at which the
    field lines turn about ``centre`` on it, in turns per field period."""
    d = x0 - centre
    angle = TWO_PI * jnp.arange(shape[0]) / shape[0] + jnp.arctan2(d[1], d[0])
    ring = centre + jnp.linalg.norm(d) * jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=-1)
    y = jnp.broadcast_to(ring[:, None, :], shape + (2,))
    f, _, _ = _on_grid(flow, coef, y, 0.0)
    return y, jnp.mean(_cross(y - centre, f) / jnp.sum((y - centre) ** 2, axis=-1)) / TWO_PI


def torus(seq, dof, x0, *, shape=(33, 33), centre=(0.0, 0.0), tol=None, maxiter=40, verbose=False, name="field"):
    """The flux surface of the field given by the 2-form ``dof`` on ``seq``, a Dirichlet or a free one
    (:func:`mrx.diagnostics.poincare.logical_field`), through the logical point ``x0 = (r, theta)`` of the plane
    ``zeta = 0``. Returns a :class:`Torus`.

    ``shape`` is the number of grid points in the straight-field-line angle and in ``zeta`` over one field
    period. Both must be odd. The iteration starts from the circle about the ``(u, v)`` point ``centre`` through
    ``x0``, which is close to a surface when the logical radius is close to a flux label and ``centre`` is close
    to the magnetic axis. It stops when a step changes the torus by at most ``tol`` in the ``(u, v)`` chart
    (default the square root of the machine epsilon), or after ``maxiter`` steps. ``verbose`` prints the
    transform, the change and the residual of the linear solve of every step and ``name`` labels error messages.

    A new ``shape`` compiles a new solver. Another field on the same sequence, or the same sequence after a new
    geometry, reuses it."""
    dof, nfp = jnp.asarray(dof), seq.nfp
    field = logical_field(seq, dof)
    require_zeta_parameterisation(field, dof, name)
    flow, coef = cross_section_flow(field), field.coefficients(dof)
    x0 = to_uv(*jnp.asarray(x0, dtype=dof.dtype))
    y, iota = _circle(flow, coef, x0, jnp.asarray(centre, dtype=dof.dtype), tuple(shape))
    tol = sqrt_eps() if tol is None else tol
    step, k = np.inf, 0
    while step > tol and k < maxiter:
        y, iota, step, left = _newton_step(flow, coef, y, iota, x0)
        step, k = float(step), k + 1
        if verbose:
            print(f"[torus] step {k}: iota {float(iota) * nfp:+.12f}, change {step:.3e}, "
                  f"linear residual {float(left):.1e}", flush=True)
    return Torus(y, float(iota) * nfp, float(_residual(flow, coef, y, iota)), k if step <= tol else -k)


# ---------------------------------------------------------------------------
# Toroidal flux
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("flow", "points"))
def _toroidal_flux(flow, coef, ring, points):
    s, w = (jnp.asarray(a, dtype=ring.dtype) for a in np.polynomial.legendre.leggauss(points))
    s, w = 0.5 * (s + 1.0), 0.5 * w
    n = ring.shape[0]
    tangent = jnp.fft.ifft(TWO_PI * 1j * jnp.fft.fftfreq(n, 1.0 / n)[:, None] * jnp.fft.fft(ring, axis=0), axis=0).real
    centre = jnp.mean(ring, axis=0)
    d = ring - centre
    omega = jax.vmap(jax.vmap(lambda q: flow(0.0, q, coef)[1]))(centre + s[:, None, None] * d[None])
    return jnp.sum((w * s)[:, None] * omega * _cross(d, tangent)[None]) / n


def toroidal_flux(seq, dof, t, *, points=64):
    r"""The toroidal flux of the field ``dof`` through the cross-section of the :class:`Torus` ``t`` at
    ``zeta = 0``, in the units of ``dof``, with the sign of :math:`\hat B^\zeta`.

    The flux density is integrated over the rays from the mean of the section curve to the curve, with ``points``
    Gauss points on each ray. The curve has to be star-shaped about its mean."""
    dof = jnp.asarray(dof)
    field = logical_field(seq, dof)
    return float(_toroidal_flux(cross_section_flow(field), field.coefficients(dof), t.y[:, 0], points))
