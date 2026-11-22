"""A flux label for any field, from an anisotropic diffusion along it.

A field with nested surfaces has a flux function. One with islands or chaotic regions has none, but an
anisotropic heat diffusion along it still has a well-defined steady state. With ``b = B / |B|`` the temperature
``T``, a 0-form that vanishes on the wall, solves

    -div((b b^T + kperp Id) grad T) = 1,

with the parallel conductivity 1 and the perpendicular one ``kperp``, the ``anisotropy``. As ``kperp -> 0`` the
temperature becomes constant along the field lines wherever they cover a surface, so on nested surfaces it is a
function of the enclosed flux: on the li383 VMEC field its spread around a radial layer is 1.1e-3 of its maximum at
``kperp = 1e-5`` and 1.2e-4 at ``1e-6``. :func:`enclosed_flux` turns it into the label ``s``, the toroidal flux inside
the level set of ``T`` over the total flux, a spline in ``T`` fixed by weak moments (:func:`label_spline`). On a VMEC
field this is VMEC's ``s``.

Across an island ``T`` stays smooth and falls through it as across nested surfaces, so ``s`` there is about the flux
of the surfaces around it. It is not flat: an island flattens only when it is wider than about
``kperp^(1/4) (R L_s / n)^(1/2)`` (Fitzpatrick 1995, ``L_s`` the shear length, ``n`` the toroidal mode number). On li383
a seeded (6,1) chain 0.2 wide in ``r`` keeps 13 % of the drop of ``T`` across it at ``kperp = 1e-5``, 12 % at ``1e-6``
and 1 % at ``1e-8``. The uniform source does not give such an island a maximum of its own at any ``kperp`` tried
(down to 1e-12). Below ``kperp`` of about 1e-9 the discrete equation has spurious local maxima.

Solved once, the temperature is a flux function only to first order in ``kperp``. Expanded as
``T = T0 / kperp + T1 + ...``, ``T0`` is a flux function and ``T1`` solves ``div(b b.grad T1) = -(1 + div grad T0)``.
The part of ``div grad T0`` that varies over a flux surface drives a parallel gradient of ``T1`` of order one, so
``|B.grad T|`` is ``kperp`` times ``|B| |grad T|`` up to a factor of the geometry. At a magnetic axis that is not
straight and round this factor grows like ``1 / r``: next to the axis ``T0`` is a quadratic form in the distance, its
Laplacian (twice the trace of the form) varies along the axis, ``T1`` there is a function of the position along the
axis, and ``grad T0`` vanishes. :func:`flux_label` therefore solves twice with the same operator. The second source is
the perpendicular flux divergence of the first temperature, ``-div(kperp grad T1)``. To leading order it has the
same flux function, and the term of first order cancels, so its defect is of second order in ``kperp``.

The field enters only through its values at the quadrature points. On a stellarator-symmetric sequence ``T`` is
even and lives on ``seq.even``. All functions trace, so they can run inside a compiled relaxation step.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np

from mrx.mass import sumfact_apply
from mrx.precision import to_working
from mrx.solvers import preconditioned_cg
from mrx.spline_bases import DerivativeSpline, SplineBasis, basis_table

#: The default ``kperp``: ten times better aligned with the surfaces than 1e-5 at about 1.8 times the CG iterations
#: (li383 (16,32,32)), far from the spurious maxima below about 1e-9.
ANISOTROPY = 1e-6
#: The iteration cap of the label solve. A mesh whose logical surfaces are far from the flux surfaces needs more
#: iterations: the vacuum's label on li383 graded (48,64,24) at the tolerance 1e-5 takes 1624 on the wout's mesh and
#: 4191 on the mesh filled from the boundary alone, 2965 and 6544 at 1e-8 (one H200, 2026-11-18).
MAXITER = 20000


def _conductivity(seq, B_jk, anisotropy):
    """The conductivity ``J (kperp G^-1 + Bh Bh^T / (Bh^T G Bh))`` at the quadrature points, ``(n_q, 3, 3)``.

    ``Bh`` are the reference components of the 2-form ``B``, ``G`` the metric and ``J`` the Jacobian determinant.
    It maps the covariant components of ``grad T`` to the contravariant density of the heat flux. The parallel
    part needs no metric in its numerator."""
    J, g = seq.jacobian_j, seq.metric_jkl
    bb = jnp.einsum('qi,qj->qij', B_jk, B_jk) / jnp.einsum('qi,qij,qj->q', B_jk, g, B_jk)[:, None, None]
    return (anisotropy * seq.metric_inv_jkl + bb) * J[:, None, None]


def flux_label_operator(seq, B_jk, anisotropy=ANISOTROPY):
    """The anisotropic diffusion operator ``T -> int (b.grad T)(b.grad w) + kperp grad T.grad w dV`` on the
    0-forms of ``seq.even`` that vanish on the wall, as a function returning the dual 0-form.

    ``B_jk`` are the reference components of the field at the quadrature points
    (``seq.odd.evaluate_at_quadrature(B, 2)``). The operator is symmetric and positive definite for
    ``anisotropy > 0``.

    It is the stiffness ``S_0 = G_0^T M_1 G_0`` with the conductivity in place of the weight ``J G^-1`` of ``M_1``,
    applied by the sum-factorised kernel of the mass matrices. So like every operator it takes a float array or the
    accurate precision of the solves (a :class:`~mrx.double_single.DS` in float32) and returns the same.
    """
    even = seq.even
    K = _conductivity(seq, jnp.asarray(B_jk), anisotropy)
    weights = (K * seq.quad.w[:, None, None]).reshape(tuple(int(n) for n in seq.quad.shape) + (3, 3))
    plan, G, E = even.mass_plan[1], even.G[0], even.E(1)

    def apply(T):
        return G.T @ (E @ sumfact_apply(plan, weights, E.T @ (G @ T)))

    return apply


def flux_label_preconditioner(seq, B_jk, anisotropy=ANISOTROPY, radial_wavenumber=2.0):
    """An approximate inverse of :func:`flux_label_operator`: the k=0 Laplacian preconditioner ``P_L``,
    rescaled per Fourier mode by how much the anisotropy changes the operator, ``W P_L W``.

    The operator is the Laplacian with the conductivity ``K`` in place of ``J G^-1``. For each radial layer of
    DoFs and each Fourier mode ``(m, n)`` of the two angles the ratio of their symbols is

        lambda(r, m, n) = k^T <K> k / k^T <J G^-1> k,   k = (k_r, m, n),

    with ``<.>`` the average over the angles of the layer. Averaging the quadratic forms keeps a mode whose
    parallel derivative varies over the surface from being taken for resonant. ``k_r`` is the
    ``radial_wavenumber``, a typical radial scale of the error that the angular Fourier scaling cannot resolve.
    The radial-angular cross terms are dropped. ``W`` scales by ``lambda^-1/2`` (two FFTs), so the product is
    symmetric positive definite and needs no inner solve.

    Measured on the li383 VMEC field (float64, tol 1e-8, 2026-09-26): 70-90 iterations at (8,12,12) p=2 for every
    anisotropy from 1e-2 to 1e-6, where ``P_L`` alone needs 80 to 1000. At (16,32,32) p=3 it needs 206 / 313 / 894
    at 1e-2 / 1e-4 / 1e-6, ``P_L`` alone 206 / 1370 / more than 5000. Half of the remaining growth comes from the
    angles not being straight-field-line angles, where a Fourier mode is not a parallel eigenmode. A
    ``radial_wavenumber`` of 2 was within 10 % of the best value over 0.25 to 8 in all these cases.
    """
    even = seq.even
    K = _conductivity(seq, jnp.asarray(B_jk), anisotropy)
    L = seq.metric_inv_jkl * seq.jacobian_j[:, None, None]
    nx, ny, nz = seq.quad.shape
    w_ang = (seq.quad.w_y[:, None] * seq.quad.w_z[None, :]).reshape(1, ny * nz, 1, 1)

    def layer_mean(X):
        return (X.reshape(nx, ny * nz, 3, 3) * w_ang).sum(axis=1) / w_ang.sum()

    Kr, Lr = layer_mean(K), layer_mean(L)                 # (nx, 3, 3) on the radial quadrature points
    s1, s2, s3 = seq.basis_0.shape[0]
    r = seq.greville[0].point_rule[0][:, 0]               # the radial DoF layers
    m = np.fft.fftfreq(s2, d=1.0 / s2)[:, None]
    n = np.fft.fftfreq(s3, d=1.0 / s3)[None, :]

    def form(X):
        """``k^T X k`` per DoF layer and mode, ``(s1, s2, s3)``."""
        Xr = [[jnp.interp(r, seq.quad.x_x, X[:, i, j])[:, None, None] for j in range(3)] for i in range(3)]
        return (Xr[0][0] * radial_wavenumber ** 2 + Xr[1][1] * m ** 2 + 2 * Xr[1][2] * m * n
                + Xr[2][2] * n ** 2)

    scale = 1.0 / jnp.sqrt(form(Kr) / form(Lr))
    E = even.E(0)

    def W(x):
        X = (E.T @ x).reshape(s1, s2, s3)
        return E @ jnp.fft.ifft2(jnp.fft.fft2(X, axes=(1, 2)) * scale, axes=(1, 2)).real.ravel()

    def apply(x):
        return W(even.L[0].precondition(W(x)))

    return apply


def flux_label(seq, B_jk, anisotropy=ANISOTROPY, guess=None, tol=None, maxiter=MAXITER):
    """Return ``(T, info)``, the temperature of :func:`flux_label_operator` with the source 1 per volume, corrected
    to second order in ``kperp``.

    With ``A = A_par + kperp A_perp`` the operator and ``b`` the load of the source 1, the first solve is
    ``A T1 = b`` and the second ``A T = kperp A_perp T1``, written ``b - A_par T1``. The second source is the
    perpendicular flux divergence of ``T1``. ``T`` has the level sets of ``T1`` without their part of first order in
    ``kperp`` (module docstring).

    ``T`` is a 0-form on ``seq.even``. It scales like ``1 / kperp``. Both solves are preconditioned conjugate
    gradients with :func:`flux_label_preconditioner`, the first warm-started from ``guess`` and the second from
    ``T1``. ``tol`` is relative, in the preconditioner's norm, and defaults to the solve tolerance of the sequence
    (``seq.tol``). ``info`` is the signed sum of the two iteration counts (positive when both converged).
    """
    even = seq.even
    A = flux_label_operator(seq, B_jk, anisotropy)
    A_par = flux_label_operator(seq, B_jk, 0.0)
    P = flux_label_preconditioner(seq, B_jk, anisotropy)
    b = even._scalar_load_values(jnp.ones(seq.quad.w.shape), 0)
    tol = even.tol if tol is None else tol
    T1, info1 = preconditioned_cg(A, b, M=P, tol=tol, maxiter=maxiter, x0=guess)
    # the perpendicular flux divergence of T1, kperp A_perp T1 = A T1 - A_par T1 = b - A_par T1 to the tolerance,
    # in the accurate precision: T1 ~ 1 / kperp and A_par T1 is the small difference of large terms
    T, info2 = preconditioned_cg(A, b - A_par(T1), M=P, tol=tol, maxiter=maxiter, x0=T1)
    info = jnp.where((info1 > 0) & (info2 > 0), info1 + info2, -jnp.abs(info1) - jnp.abs(info2))
    return to_working(T), info


def label_spline(seq, T, B_jk, cells=None):
    """``(a, b, e)``: the label ``s = sigma(T)`` as a cubic spline in ``T`` on ``[a, b]``, ``a`` and ``b`` the smallest
    and the largest tensor coefficient of ``T`` (bounds of its values, ``a <= 0`` since ``T`` vanishes on the wall),
    ``e`` the coefficients of ``sigma`` in the clamped cubic splines of ``u = (T - a) / (b - a)`` on ``cells`` uniform
    cells (the number of radial cells of ``seq`` unless given).

    ``sigma(t)`` is the toroidal flux inside the level set ``T = t`` over the total flux. The flux through a plane
    ``zeta = const`` is ``int Bh^zeta dr dtheta``, its mean over ``zeta`` the volume integral of ``Bh^zeta`` over
    the logical cube over the ``zeta`` extent ``Z``. By the coarea formula the derivative of ``sigma`` has the moments

        int sigma'(u) M_k(u) du = -(1 / (Z Phi)) int M_k(u(T)) Bh^zeta dx

    against every derivative spline ``M_k`` (:class:`~mrx.spline_bases.DerivativeSpline`, quadratic): a volume
    integral of the field's own component and of the 0-form ``T`` at the quadrature points. ``sigma'`` is their L2
    projection onto the ``M_k`` (one Gram solve of the quadratic splines) and ``sigma`` its integral with
    ``sigma(b) = 0``. The constants lie in the span of the ``M_k``, so ``sigma(a) = 1`` exactly. Nothing is sorted,
    binned or read off at a point. Points of equal ``T`` have equal ``s``. The projection resolves the distribution
    of the flux over ``T`` on the cells: where the level sets meet the hottest value (the magnetic axis) the
    distribution ends with a jump inside the last cell, and there the projection is accurate to about a cell's share
    of the flux."""
    nc = (seq.basis_0.Lambda[0].n - seq.p) if cells is None else int(cells)
    basis = SplineBasis(nc + 3, 3, "clamped")
    d = DerivativeSpline(basis)
    c = seq.even.E(0).T @ T
    a, b = jnp.min(c), jnp.max(c)
    T_q = seq.even.evaluate_at_quadrature(T, 0)[:, 0]
    f = seq.quad.w * jnp.asarray(B_jk)[:, 2]
    values, indices = jax.vmap(d.evaluate_local)((T_q - a) / (b - a))
    m = -jnp.zeros(d.n, f.dtype).at[indices].add(values * f[:, None]) / jnp.sum(f)
    slope = jnp.asarray(_gram_inverse(nc), dtype=f.dtype) @ m
    e = -jnp.concatenate([jnp.cumsum(slope[::-1])[::-1], jnp.zeros(1, f.dtype)])
    return a, b, e


@functools.cache
def _gram_inverse(cells):
    """The inverse of the Gram matrix of the derivative splines of the clamped cubic splines on ``cells`` uniform
    cells of ``[0, 1]`` (quadratic, unit integral), by three Gauss points per cell, float64."""
    x, w = np.polynomial.legendre.leggauss(3)
    edges = np.linspace(0.0, 1.0, cells + 1)
    xq = (0.5 * (edges[1:] - edges[:-1])[:, None] * (x + 1.0) + edges[:-1, None]).ravel()
    wq = (0.5 * (edges[1:] - edges[:-1])[:, None] * w).ravel()
    with jax.ensure_compile_time_eval():          # called inside the compiled label
        M = np.asarray(basis_table(DerivativeSpline(SplineBasis(cells + 3, 3, "clamped")), jnp.asarray(xq)),
                       dtype=np.float64)
    return np.linalg.inv((M * wq) @ M.T)


def enclosed_flux(seq, T, B_jk):
    """The label ``s`` at the quadrature points: the toroidal flux inside the level set of ``T`` through each
    quadrature point, over the total toroidal flux (:func:`label_spline`)."""
    return enclosed_flux_map(seq, T, B_jk)(seq.even.evaluate_at_quadrature(T, 0)[:, 0])


def enclosed_flux_map(seq, T, B_jk):
    """The map ``T0 -> s`` of :func:`enclosed_flux` as a function, which also takes values of ``T`` off the
    quadrature points (a field line's, for example): the spline :func:`label_spline`, 1 at the smallest coefficient
    of ``T`` and 0 at the largest."""
    a, b, e = label_spline(seq, T, B_jk)
    basis = SplineBasis(e.size, 3, "clamped")

    def sigma(T0):
        values, indices = jax.vmap(basis.evaluate_local)(jnp.ravel((T0 - a) / (b - a)))
        return jnp.sum(values * e[indices], axis=1).reshape(jnp.shape(T0))
    return sigma
