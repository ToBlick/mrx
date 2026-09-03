"""Preconditioner specifications and the Kronecker mass model behind the metric-lumping atoms.

:class:`MassPreconditionerSpec`, :class:`SchurPreconditionerSpec` and
:class:`SaddlePointPreconditionerSpec` name what a solve is preconditioned
with: kinds ``'none'``, ``'jacobi'`` (the probed diagonal,
``DeRhamSequence.build_preconditioners(jacobi=True)``) and
``'metric_lumping'`` (:func:`default_mass_preconditioner`, production).
:func:`_kron_mass_model_1d` is the separable Kronecker model
``Lam (x)_a A_a Lam`` of the mass matrices that
:class:`~mrx.metric_lumping_laplacian.MetricLumpingMass` inverts on the bulk
DoFs, and :func:`_simultaneous_diagonalize_pair` the generalised eigensolve
the Laplacian atoms' fast diagonalisation is built from.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field


import jax
import jax.numpy as jnp
import numpy as np

from mrx.precision import DTYPE


@dataclass(frozen=True)
class MassPreconditionerSpec:
    """A mass preconditioner, named by ``kind``.

    ``'none'`` is the identity and ``'metric_lumping'`` the production one: a separable Kronecker bulk with
    the polar core probed and inverted DENSELY, scaled by the metric-lumped
    diagonal. The field default is the production kind so that a bare spec
    and :func:`default_mass_preconditioner` agree.
    """
    kind: str = 'metric_lumping'


@dataclass(frozen=True)
class SchurPreconditionerSpec:
    inner: MassPreconditionerSpec = dataclass_field(
        default_factory=MassPreconditionerSpec)
    #: A bare spec must not silently mean SOME preconditioner. 'none' rather
    #: than 'metric_lumping' on purpose: 'none' fails VISIBLY at the first
    #: solve, where 'metric_lumping' would quietly work with something the
    #: caller never asked for. The authoritative answer is
    #: ``operators._materialize_default_saddle_preconditioner``, which needs a
    #: sequence and so cannot live in a field default at all. Every
    #: construction site in mrx/, test/ and scripts/ passes ``outer=``
    #: explicitly.
    outer: MassPreconditionerSpec = dataclass_field(
        default_factory=lambda: MassPreconditionerSpec(kind='none'))


@dataclass(frozen=True)
class SaddlePointPreconditionerSpec:
    mass: MassPreconditionerSpec = dataclass_field(
        default_factory=MassPreconditionerSpec)
    schur: SchurPreconditionerSpec = dataclass_field(
        default_factory=SchurPreconditionerSpec)
    coupled: bool = False


def default_mass_preconditioner() -> MassPreconditionerSpec:
    """The production mass preconditioner: metric_lumping.

    A separable Kronecker bulk with the polar core rows PROBED AND INVERTED
    DENSELY, scaled by the metric-lumped diagonal. The only mass preconditioner
    besides 'none'.

    MEASURED against its predecessor, the plain Kronecker model
    (docs/research/production_simplification_plan.md §10), 224 cells over
    four geometries, n = 8..20, p = 2..5:

    * the mass solve itself: median **0.83x** the iterations, and
      **0.70-0.77x** at k=1,2 where the cost is. The advantage HOLDS OR GROWS
      with h and is flat in p. Build time was equal (2.0 vs 2.2 s median).
    * the effect on ``L_k`` -- the mass preconditioner is the weak term's inner
      inverse, so this changes the OPERATOR at k >= 1, not just the solve:
      **median 0.91x, better in 12 of 16 cells**, up to 0.79x on the Dirichlet
      rows. Only regression was cylinder k=1 (1.07x).
    * the natural-BC scale SURVIVES: worst-case penalty against each cell's
      own optimum moves 1.14 -> 1.22, and only on the toroid, where the basin
      is flat. The shaped geometries are unchanged (1.01-1.04).

    Only regression anywhere was ~5% at k=0, on mass solves that take 7-17
    iterations either way.

    As the Schur-Jacobi probe backing it was measured separately -- six
    converged cells, five favouring it by 2.4-16.6%, one at +0.6% inside
    measured noise: docs/research/result_2026-08-25_schur_probe_ab.md.

    THE BUILD IS NOT JIT-SAFE, AND DOES NOT NEED TO BE. Its sparsity
    bookkeeping is host-side numpy and its core probe runs the matrix-free
    apply on concrete vectors. It runs once, in ``build_preconditioners``,
    outside every trace; only the apply is jit-safe.

    CAVEAT ON THE EVIDENCE: the mass A/B covers h = 8..20 and p = 2..5, but the
    effect on ``L_k`` was measured at n=12, p=3 only. The overnight sweep in
    ``outputs/diag_newstack/`` extends that to n = 8..32 and p = 2..5.
    """
    return MassPreconditionerSpec(kind='metric_lumping')


def _symmetrize(matrix: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (matrix + matrix.T)


def _simultaneous_diagonalize_pair(M: jnp.ndarray, A: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Simultaneously diagonalize an SPD ``M`` and a symmetric ``A``.

    Returns ``(V, lam)`` such that ``V.T @ M @ V = I`` and
    ``V.T @ A @ V = diag(lam)``, via Cholesky ``M = L L.T`` and a
    symmetric eigendecomposition of ``L^{-1} A L^{-T}``. This is the
    per-axis primitive of the Lynch fast-diagonalization (FD) method:
    given a 3D Kronecker sum ``M_r (x) M_t (x) M_z + A_r (x) A_t (x) A_z``
    with all ``M_axis`` SPD, applying the per-axis ``V`` reduces it to
    the diagonal ``1 + lam_r (x) lam_t (x) lam_z`` in the M-orthonormal
    basis. Reusable by the stiffness preconditioner.
    """
    M_sym = _symmetrize(jnp.asarray(M, dtype=DTYPE))
    A_sym = _symmetrize(jnp.asarray(A, dtype=DTYPE))
    L = jnp.linalg.cholesky(M_sym)
    Linv_A = jax.scipy.linalg.solve_triangular(L, A_sym, lower=True)
    B = jax.scipy.linalg.solve_triangular(L, Linv_A.T, lower=True).T
    B = _symmetrize(B)
    lam, U = jnp.linalg.eigh(B)
    V = jax.scipy.linalg.solve_triangular(L.T, U, lower=False)
    return V, lam


def _assemble_weighted_1d_mass(B: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    return (B * weights[None, :]) @ B.T


def _metric_lumping_diff_flags(k: int, c: int) -> tuple:
    """Differentiated-axis flags for component ``c`` of a k-form.

    Mirrors ``_component_axis_bases_k0/k1/k2/k3`` in :mod:`mrx.mass`:
    k=0 differentiates nothing, k=3 everything, k=1 only axis ``c``, and k=2
    every axis *except* ``c``.
    """
    match k:
        case 0:
            return (False, False, False)
        case 3:
            return (True, True, True)
        case 1:
            return tuple(a == c for a in range(3))
        case 2:
            return tuple(a != c for a in range(3))
    raise ValueError("k must be 0, 1, 2 or 3")


def _kron_mass_model_1d(seq, k: int, d_raw=None):
    """1-D factors of the Kronecker model of ``M_k``::

        M_k  ~  (+)_c  Lam_c (A^c_r x A^c_t x A^c_z) Lam_c

    with *unweighted* 1-D masses (degree ``p`` on primal axes, ``p-1`` on each
    differentiated axis) and the diagonal scaling ``Lam_c`` chosen so that the
    model reproduces ``diag(M_k)`` **exactly**: it is the support-averaged
    metric weight, ``sqrt(diag(M_k)_c / diag(A^c_r x A^c_t x A^c_z))``.

    This is the forward half of :func:`build_mass_metric_lumping_factors` -- that one
    inverts the 1-D masses and stores ``1/Lam`` -- and it is also the mass model
    the weak-term diagonal builds on. Measured ``||M~ - M||_F / ||M||_F`` on a
    spline toroid: ``3.1e-2`` at k=1, ``7e-3`` at k=2 and k=3.

    Returns ``(shapes, mass_1d, lam)``: the raw block shapes, the three 1-D
    masses per component, and the 3-D scaling per component.
    """
    from mrx.mass import build_mass_diagonal  # noqa: PLC0415

    form = getattr(seq, f"basis_{k}")
    shapes = [tuple(int(s) for s in sh) for sh in form.shape]

    if d_raw is None:
        d_raw = build_mass_diagonal(seq, k)
    d_raw = jnp.asarray(d_raw)

    primal = (seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk)
    deriv = (seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk)
    quad_w = (seq.quad.w_x, seq.quad.w_y, seq.quad.w_z)

    mass_1d, lam, start = [], [], 0
    for c, shape in enumerate(shapes):
        diff = _metric_lumping_diff_flags(k, c)
        bases = tuple(deriv[a] if diff[a] else primal[a] for a in range(3))
        m1 = tuple(_assemble_weighted_1d_mass(bases[a], quad_w[a]) for a in range(3))
        for a in range(3):
            if int(m1[a].shape[0]) != shape[a]:
                raise ValueError(
                    f"kron mass model k={k} component {c} axis {a}: 1D mass is "
                    f"{m1[a].shape[0]} but the raw block axis is {shape[a]}"
                )
        kron_diag = jnp.einsum('i,j,l->ijl', jnp.diag(m1[0]),
                               jnp.diag(m1[1]), jnp.diag(m1[2]))
        size = int(np.prod(shape))
        mass_1d.append(m1)
        lam.append(jnp.sqrt(d_raw[start:start + size].reshape(shape) / kron_diag))
        start += size
    return shapes, tuple(mass_1d), tuple(lam)


def build_shifted_mass_laplace_atom(seq, k: int, *, dirichlet: bool = False):
    """Fast-diagonalisation factors for ``M_k + eps L_k``, for EVERY ``eps``.

    The relaxation's velocity smoothing and both Leray solves invert
    ``M + eps L`` with ``eps = eta * (accumulated dt)``, which changes every
    step and again at every point of the line search. They are preconditioned
    today with the mass atom alone, which is an inverse of the ``eps = 0``
    end, and that is why they run to hundreds of iterations.

    ``docs/research/OPEN.md`` section 3.9 recorded two reasons not to build
    this. Both are answered here.

    *"The atoms share no generalised eigenbasis, because the mass atom lumps
    the metric and the Laplacian atom averages it differently."* Their 1-D
    factors are in fact **bit-identical**: :func:`_kron_mass_model_1d` and
    :func:`~mrx.metric_lumping_laplacian.component_factors` both build the
    axis mass as ``_assemble_weighted_1d_mass(basis, quad_w[a])`` on the same
    derivative-axis pattern, deliberately unweighted in both, and the metric
    is carried outside as a diagonal in both. Verified for k=1,2,3 and every
    component. What differs is only *which* diagonal, and that is a choice.
    This atom makes it once: the mass model's ``Lam``, which reproduces
    ``diag(M_k)`` exactly.

    *"``eps`` changes every step, so the factorisation would be rebuilt."* It
    would not. In the basis that :func:`_simultaneous_diagonalize_pair`
    returns, ``V.T M V = I`` and ``V.T L V = diag(mu)``, so ``M + eps L``
    becomes ``1 + eps * mu`` -- diagonal for every ``eps``, with the same
    ``V``. Only the reciprocal depends on ``eps``, and it is a runtime array
    op inside the jitted step. One build serves the whole line search.

    The two limits are the two existing atoms, which is the check that the
    construction is the right one: at ``eps = 0`` it is exactly the mass
    atom's Kronecker bulk (``V`` is M-orthonormal, so ``V V.T = A^-1``), and
    as ``eps`` grows it approaches the Laplacian atom, differing only by
    carrying ``Lam`` where that atom carries its own diagonal.

    MEASURED, AND NOT WIRED IN. It does what it was built to do and that turns
    out not to be worth having. ``tpu/shifted_atom_measure.py``, li383
    ``(8,16,8)`` p=3 float64, relative miss ``||P A x - x|| / ||x||`` at k=2:

    ==========  ============  =========
    eps         shifted atom  mass atom
    ==========  ============  =========
    0           3.26          2.92
    4.4e-4      **2.20**      6.02
    1e-2        **2.94**      116.0
    1e-1        **3.27**      1158.2
    ==========  ============  =========

    The miss stays bounded over four decades where the mass atom's grows
    linearly, so the shared-eigenbasis construction works. But CG counts
    iterations, not norms, and there it LOSES: 185 against 149 at the
    operative shift. The reason is visible at ``eps = 0``, where both model
    the same operator with the same 1-D factors and the only difference is
    that this atom has no polar-core block: 82 iterations against 57. That
    1.44x core deficit is larger than anything the shift handling wins back,
    and even removing it entirely projects only ~1.16x at the shift the
    VELOCITY SMOOTHING solve uses -- ``eps = 0.064 / n_r^2 = 4.4e-4``, where
    ``M + eps L`` is still mass-dominated and the mass atom is already close
    to the right preconditioner.

    THE RESISTIVE SOLVE IS A SEPARATE QUESTION AND IS NOT SETTLED. It is the
    other user of ``M + eps L`` and runs much further out: ``OPEN.md`` section
    3.9 measures ``eta = 1e-1`` giving ``eps ~ 0.17``, where the mass atom
    takes 612-1938 iterations and at ``eta = 1`` does not converge. That is
    where this atom's norm advantage is largest (3.27 against 1158 at ``eps =
    0.1``). Its iteration count there was worse too, but at that shift the
    operator is Laplacian-dominated and the missing core rows matter most, so
    the number confounds the core with the atom. Refuted for smoothing,
    untested for resistive.

    IT CANNOT HELP THE LERAY UPPER BLOCK EITHER, contrary to the plan that
    asked for it. That block's operator is the Schur complement ``S_3``, not
    ``M_3 + eps L_3``, and ``L_3`` is identically zero (there is no 4-form to
    take ``G_3`` into), so a shifted atom at k=3 IS the mass atom. Its 1.88
    miss needs a Schur model, which is a different object.

    Kept, tested and documented rather than deleted: it is the measured answer
    to ``docs/research/OPEN.md`` section 3.9, which had this as an open
    proposal resting on two reasons that are now known to be wrong. Anyone
    revisiting it should start by giving it a core block, and should expect
    ~1.16x for that work.

    Args:
        seq: the de Rham sequence, with a geometry installed.
        k: form degree, 0 to 3.
        dirichlet: whether the boundary rows are constrained, passed to
            ``component_factors`` for the natural-BC face term.

    Returns:
        ``(shapes, vs, mus, alphas, scales)``, one entry per component:
        the raw block shape, the three per-axis eigenvector matrices, the
        three per-axis eigenvalues, the three Kronecker weights, and the
        diagonal ``Lam`` in the block's own shape.

    Note:
        This is the BULK atom. The polar core rows are handled by their own
        dense block in the production preconditioners and are not treated
        here, so the miss on a polar sequence carries the core with it.
    """
    from mrx.metric_lumping_laplacian import component_factors  # noqa: PLC0415

    shapes, mass_1d, scales = _kron_mass_model_1d(seq, k)
    vs, mus, alphas = [], [], []
    for c in range(len(shapes)):
        masses, stiffs, alpha = component_factors(seq, k, c,
                                                  dirichlet=dirichlet)
        v_c, mu_c = [], []
        for a in range(3):
            # The shared-basis claim is the whole construction, so it is
            # checked rather than trusted: a future change to either builder
            # that breaks it must fail here and not silently precondition
            # with the wrong eigenvectors.
            if masses[a].shape != mass_1d[c][a].shape:
                raise ValueError(
                    f"shifted atom k={k} component {c} axis {a}: the mass "
                    f"model's 1-D factor is {mass_1d[c][a].shape} but the "
                    f"Laplacian atom's is {masses[a].shape}; they must be the "
                    f"same matrix for a shared eigenbasis to exist")
            v, mu = _simultaneous_diagonalize_pair(mass_1d[c][a], stiffs[a])
            v_c.append(v)
            mu_c.append(mu)
        vs.append(tuple(v_c))
        mus.append(tuple(mu_c))
        alphas.append(tuple(alpha))
    return shapes, tuple(vs), tuple(mus), tuple(alphas), scales


def apply_shifted_mass_laplace_atom(atom, x: jnp.ndarray, eps) -> jnp.ndarray:
    """Apply ``(M_k + eps L_k)^{-1}`` from :func:`build_shifted_mass_laplace_atom`.

    ``eps`` may be a traced scalar: it enters only the diagonal reciprocal, so
    a step that varies it does not retrace the factorisation.

    Args:
        atom: the tuple returned by :func:`build_shifted_mass_laplace_atom`.
        x: raw-DoF-space vector, components concatenated in order.
        eps: the shift, ``eta * (accumulated dt)`` in the relaxation.

    Returns:
        The preconditioned vector, same shape as ``x``.
    """
    from mrx.operators import _fd_apply_3d  # noqa: PLC0415

    shapes, vs, mus, alphas, scales = atom
    out, start = [], 0
    for c, shape in enumerate(shapes):
        size = int(np.prod(shape))
        block = x[start:start + size].reshape(shape) / scales[c]
        # _fd_apply_3d forms ``sum_a alpha_a mu_a + eps_arg``. Folding the
        # shift into alpha and passing eps_arg = 1 gives ``1 + eps sum_a
        # alpha_a mu_a``, which is what M + eps L is in this basis, without a
        # second apply primitive to keep in step with the first.
        y = _fd_apply_3d(vs[c][0], vs[c][1], vs[c][2],
                         mus[c][0], mus[c][1], mus[c][2],
                         tuple(eps * a for a in alphas[c]), block, eps=1.0)
        out.append((y / scales[c]).reshape(-1))
        start += size
    return jnp.concatenate(out)


