"""A flux label for any field, from an anisotropic diffusion along it.

A field with nested surfaces has a flux function. One with islands or chaotic regions has none, but an
anisotropic heat diffusion along it still has a well-defined steady state. With ``b = B / |B|`` the temperature
``T``, a 0-form that vanishes on the wall, solves

    -div((b b^T + kperp Id) grad T) = 1,

with the parallel conductivity 1 and the perpendicular one ``kperp``, the ``anisotropy``. As ``kperp -> 0`` the
temperature becomes constant along the field lines wherever they cover a surface. It flattens across islands
wider than about ``kperp^(1/4)`` (Fitzpatrick 1995) and across chaotic regions, and is a function of the enclosed
flux elsewhere. :func:`enclosed_flux` turns it into the label ``s``, the toroidal flux inside the level set of
``T`` over the total flux. On a VMEC field this is VMEC's ``s``.

The field enters only through its values at the quadrature points. On a stellarator-symmetric sequence ``T`` is
even and lives on ``seq.even``. All functions trace, so they can run inside a compiled relaxation step.
"""
import jax
import jax.numpy as jnp
import numpy as np

from mrx.precision import default_tol
from mrx.solvers import preconditioned_cg

#: The default ``kperp``. Islands narrower than about ``kperp^(1/4) = 0.06`` of the minor radius keep a
#: temperature gradient across them.
ANISOTROPY = 1e-5


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
    """
    even = seq.even
    K = _conductivity(seq, jnp.asarray(B_jk, even.dtype), anisotropy)

    def apply(T):
        dT = even.G[0] @ T
        flux = jnp.einsum('qij,qj->qi', K, even.evaluate_at_quadrature(dT, 1))
        return even.G[0].T @ even.vector_load_values(flux, 2, 1)

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
    K = _conductivity(seq, jnp.asarray(B_jk, even.dtype), anisotropy)
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

    scale = (1.0 / jnp.sqrt(form(Kr) / form(Lr))).astype(even.dtype)
    E = even.E(0)

    def W(x):
        X = (E.T @ x).reshape(s1, s2, s3)
        return E @ jnp.fft.ifft2(jnp.fft.fft2(X, axes=(1, 2)) * scale, axes=(1, 2)).real.ravel()

    def apply(x):
        return W(even.L[0].precondition(W(x)))

    return apply


def flux_label(seq, B_jk, anisotropy=ANISOTROPY, guess=None, tol=None, maxiter=2000):
    """Return ``(T, info)``, the temperature of :func:`flux_label_operator` with the source 1 per volume.

    ``T`` is a 0-form on ``seq.even``. It scales like ``1 / kperp``. The solve is preconditioned conjugate
    gradients with :func:`flux_label_preconditioner`, warm-started from ``guess``. ``tol`` is relative, in the
    preconditioner's norm, and defaults to the plain solve tolerance of the working precision (1e-5 in float32).
    ``info`` is the signed iteration count (positive when converged).
    """
    even = seq.even
    A = flux_label_operator(seq, B_jk, anisotropy)
    P = flux_label_preconditioner(seq, B_jk, anisotropy)
    b = even._scalar_load_values(jnp.ones(seq.quad.w.shape, even.dtype), 0)
    tol = default_tol(even.dtype, refine=False) if tol is None else tol
    return preconditioned_cg(A, b, M=P, tol=tol, maxiter=maxiter, x0=guess)


def enclosed_flux(seq, T, B_jk):
    """The label ``s`` at the quadrature points: the toroidal flux inside the level set of ``T`` through each
    quadrature point, over the total toroidal flux.

    The flux through a ``zeta = const`` plane is ``int Bh^zeta dr dtheta``, and its average over ``zeta`` is the
    quadrature sum of ``Bh^zeta`` (exact on a half-period quadrature too). Each quadrature point carries its share
    ``w_q Bh^zeta_q`` of the flux. Sorted by falling ``T``, the flux of all hotter points gives ``s`` at the
    samples, but only to one radial layer of samples, whose points have nearly the same ``T``. So the sorted
    samples are cut into as many groups as there are radial quadrature layers, and ``s(T)`` is piecewise linear
    through the group means of ``T`` and ``s``, from ``s = 0`` at the hottest point to ``s = 1`` on the wall.
    Points of equal ``T`` have equal ``s``, and a layer's points get about the flux at its middle.
    """
    T_q = seq.even.evaluate_at_quadrature(T, 0)[:, 0]
    f = seq.quad.w * jnp.asarray(B_jk)[:, 2]
    order = jnp.argsort(-T_q)                                # descending T: the axis first
    s_sorted = (jnp.cumsum(f[order]) - 0.5 * f[order]) / f.sum()
    n_groups = seq.quad.shape[0]
    group = jnp.asarray(np.arange(T_q.size) * n_groups // T_q.size, jnp.int32)
    count = jax.ops.segment_sum(jnp.ones_like(T_q), group, n_groups)
    T_mean = jax.ops.segment_sum(T_q[order], group, n_groups) / count
    s_mean = jax.ops.segment_sum(s_sorted, group, n_groups) / count
    zero, one = jnp.zeros(1, T_q.dtype), jnp.ones(1, T_q.dtype)
    T_knots = jnp.concatenate([T_q[order][:1], T_mean, zero])[::-1]          # ascending, T = 0 on the wall
    s_knots = jnp.concatenate([zero, s_mean, one])[::-1]
    return jnp.interp(T_q, T_knots, s_knots)
