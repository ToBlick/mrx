"""The matrices of the discrete de Rham complex as operator objects, and their inverses.

For each form degree ``k = 0..3`` a sequence has the mass matrix ``M_k``, the exterior derivative
``G_k`` (a matrix of -1, 0, +1: grad, curl, div), the weak derivative ``D_k = M_{k+1} G_k``, the
stiffness ``S_k = G_k^T M_{k+1} G_k`` and the Hodge Laplacian
``L_k = S_k + D_{k-1} M_{k-1}^{-1} D_{k-1}^T``. None of these is ever assembled. Users reach them
through the sequence, ``seq.M[k] @ v``, ``seq.G[k].T @ w``, ``seq.L[k].solve(b)``, and the classes
here are what those expressions return. An operator acts on the spaces of the sequence it came
from: ``seq`` has the Dirichlet spaces, ``seq.free`` the free ones.

An operator object holds only its sequence and its degree. Creating one is free, also inside a
jitted function, and never triggers a recompile. The solves are compiled once per sequence, degree
and space and reuse the compiled program for every new right-hand side and every new geometry.

The solves need the preconditioners and harmonic forms of the installed geometry, a
:class:`SequenceOperators` bundle that ``seq.build_preconditioners()`` builds and that must be
built again after every ``set_geometry_from_map``.
"""
from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax.numpy as jnp

from mrx.mass import sumfact_apply
from mrx.solvers import deflation_projectors, refine, solve_saddle_point_minres, solve_singular_cg
from mrx.symmetry import symmetrize_like


class SequenceOperators(eqx.Module):
    """The preconditioners and harmonic forms of one geometry, each a dict keyed by ``(k, dirichlet)``.

    ``DeRhamSequence.build_preconditioners`` builds the bundle, and a new geometry needs a new one.
    A sequence and its free view share one bundle, and each picks the entries of its own spaces.
    The harmonic forms start as zeros and are filled in by :func:`mrx.nullspace.compute_nullspaces`.
    Until then the solves remove nothing from the kernel.
    """

    # the metric-lumped preconditioners of M_k and L_k (mrx.metric_lumping)
    mass_lumping: Optional[dict] = None
    laplacian_lumping: Optional[dict] = None
    # the harmonic k-forms as rows, shape (n_vectors, n_k)
    nullspaces: Optional[dict] = None


def _replace(operators, field, value):
    """A copy of ``operators`` with the dict ``field`` replaced by ``value``."""
    return eqx.tree_at(lambda ops: getattr(ops, field), operators, value,
                       is_leaf=lambda x: x is None or isinstance(x, dict))


def n_vectors(betti_numbers, k, dirichlet):
    """The number of harmonic ``k``-forms. ``betti_numbers`` belong to the spaces without boundary
    conditions. The Dirichlet spaces have ``b_{3-k}`` instead (Poincare-Lefschetz duality)."""
    b0, b1, b2, _b3 = betti_numbers
    if dirichlet:
        return (0, b2, b1, b0)[k]
    return (b0, b1, b2, 0)[k]


def init_nullspaces(seq, operators):
    """A copy of ``operators`` whose harmonic forms are all zero, with the shapes that
    ``seq.betti_numbers`` gives."""
    return _replace(operators, "nullspaces", {
        (k, d): jnp.zeros((n_vectors(seq.betti_numbers, k, d), seq.extraction[(k, d)].forward_shape[0]))
        for k in range(4) for d in (False, True)})


def new_operators(seq) -> SequenceOperators:
    """An empty bundle for ``seq``, with no preconditioners and zero harmonic forms."""
    return init_nullspaces(seq, SequenceOperators(mass_lumping={}, laplacian_lumping={},
                                                  nullspaces={}))


def set_nullspace(operators, k, dirichlet, values):
    """A copy of ``operators`` with the harmonic forms of ``(k, dirichlet)`` replaced by ``values``."""
    spaces = dict(operators.nullspaces)
    spaces[(int(k), bool(dirichlet))] = values
    return _replace(operators, "nullspaces", spaces)


def _bundle(operators):
    if operators is None:
        raise ValueError("no operator bundle: call seq.build_preconditioners() after set_geometry_from_map")
    return operators


def _atom(operators, kind: str, k: int, dirichlet: bool):
    """The preconditioner of ``(k, dirichlet)``, ``kind`` being ``'mass'`` or ``'laplacian'``."""
    try:
        return getattr(_bundle(operators), f"{kind}_lumping")[(int(k), bool(dirichlet))]
    except KeyError:
        raise ValueError(
            f"the metric-lumped {kind} atom of k={k}, dirichlet={dirichlet} is not built. "
            "seq.build_preconditioners() builds it for the installed geometry") from None


def _seq_atom(seq, kind: str, k: int):
    """The preconditioner of the ``k``-form space of ``seq``."""
    return _atom(seq.operators, kind, k, seq.dirichlet)


def _harmonic(seq, k):
    """The harmonic forms of the ``k``-form space of ``seq``, one per row."""
    return _bundle(seq.operators).nullspaces[(int(k), seq.dirichlet)]


def assemble_preconditioners(seq, operators):
    """A copy of ``operators`` with the mass and Laplacian preconditioners of the Dirichlet and the
    free spaces of every degree built for the installed geometry. This needs at least two radial
    elements (``n >= p + 2``)."""
    from mrx.metric_lumping import MetricLumpingLaplacian, MetricLumpingMass  # noqa: PLC0415
    views = [(k, view) for k in range(4) for view in (seq.free, seq)]
    mass = dict(operators.mass_lumping)
    for k, view in views:
        mass[(k, view.dirichlet)] = MetricLumpingMass(view, operators, k)
    # the Laplacian preconditioners are built using the mass ones
    operators = _replace(operators, "mass_lumping", mass)
    laplacian = dict(operators.laplacian_lumping)
    for k, view in views:
        laplacian[(k, view.dirichlet)] = MetricLumpingLaplacian(view, operators, k)
    return _replace(operators, "laplacian_lumping", laplacian)


def _geometry(seq):
    if seq.geometry is None:
        raise ValueError("no geometry installed: call seq.set_geometry_from_map first")
    return seq.geometry


def _mass_core(seq, k: int):
    """The function ``x -> M_k x`` on the full tensor-product spline DoFs."""
    plan, weights = seq.mass_plan[k], _geometry(seq).mass_weights[k]
    return _half_period_apply(seq, lambda x: sumfact_apply(plan, weights, x), k, k)


def _half_period_apply(seq, core, k_in, k_out):
    """On a half-period sequence, ``core`` (integrated over half a period) turned into the
    full-period apply. The output takes the stellarator parity of the input, which is exact for an
    input of one parity. ``core`` is returned unchanged on a full-period sequence or a parity view."""
    if not seq.half_period or seq.parity is not None:
        return core
    plan_in, plan_out = seq.reflection_plan[k_in], seq.reflection_plan[k_out]

    def apply(x):
        return symmetrize_like(core(x), x, plan_out, plan_in)
    return apply


# ---------------------------------------------------------------------------
# Applies. Every function acts on the spaces of ``seq`` (Dirichlet or free).
# ---------------------------------------------------------------------------

def _raw_incidence(seq, k: int):
    """``(G_k, G_k^T)`` on the full tensor-product DoFs (:mod:`mrx.incidence`)."""
    return getattr(seq, f"g{k}"), getattr(seq, f"g{k}_T")


def _grad(seq, v, k: int, transpose: bool = False):
    """``G_k v``, the exterior derivative of the ``k``-form ``v``, or ``G_k^T v`` if ``transpose``."""
    if k < 2:
        g = (seq.g0_grad if k == 0 else seq.g1_curl)[seq.dirichlet]
        return (g.T if transpose else g) @ v
    # div needs no polar stencil: the 3-form extraction is a plain selection (E_3 E_3^T = I),
    # unlike the output spaces of grad and curl, which have axis rows
    sp, sp_T = _raw_incidence(seq, 2)
    e_in, e_out = seq.E(2), seq.E(3)
    if transpose:
        return e_in @ (sp_T @ (e_out.T @ v))
    return e_out @ (sp @ (e_in.T @ v))


def _mass(seq, v, k: int):
    """``M_k v``."""
    e = seq.E(k)
    return e @ _mass_core(seq, k)(e.T @ v)


def _projection(seq, v, k_in: int, k_out: int):
    """``P v`` with the metric-free pairing ``P_ij = int Lambda^{k_out}_i . Lambda^{k_in}_j`` over the
    logical domain, for ``(k_in, k_out)`` = (1, 2), (2, 1), (0, 3) or (3, 0)."""
    plan = seq.projection_plan[(k_out, k_in)]
    weights = _geometry(seq).reference_weights[len(plan.shapes_r)]
    core = _half_period_apply(seq, lambda x: sumfact_apply(plan, weights, x), k_in, k_out)
    return seq.E(k_out) @ core(seq.E(k_in).T @ v)


def _weak_derivative(seq, v, k: int, transpose: bool = False):
    """``D_k v`` with ``D_k = M_{k+1} G_k``, or ``D_k^T v`` if ``transpose``."""
    g_sp, g_sp_T = _raw_incidence(seq, k)
    m_apply = _mass_core(seq, k + 1)
    e_in, e_out = seq.E(k), seq.E(k + 1)
    if transpose:
        return e_in @ (g_sp_T @ m_apply(e_out.T @ v))
    return e_out @ m_apply(g_sp @ (e_in.T @ v))


def _stiffness(seq, v, k: int):
    """``S_k v`` with ``S_k = G_k^T M_{k+1} G_k``, which is zero for k = 3."""
    if k == 3:
        return jnp.zeros_like(v)
    g_sp, g_sp_T = _raw_incidence(seq, k)
    m_apply = _mass_core(seq, k + 1)
    e = seq.E(k)
    return e @ (g_sp_T @ m_apply(g_sp @ (e.T @ v)))


def laplacian_with(seq, v, k: int, minv):
    """``S_k v + D_{k-1} minv(D_{k-1}^T v, k - 1)``, the Hodge Laplacian with ``minv(w, j)`` in
    place of ``M_j^{-1} w``."""
    strong = _stiffness(seq, v, k)
    if k == 0:
        return strong
    return strong + _weak_derivative(seq, minv(_weak_derivative(seq, v, k - 1, transpose=True), k - 1), k - 1)


@eqx.filter_jit
def _laplacian(seq, v, k: int):
    """``L_k v``. For k >= 1 this contains a solve with ``M_{k-1}``."""
    return laplacian_with(seq, v, k, lambda w, j: _mass_solve(seq, w, j))


def apply_laplacian_approx(seq, operators: SequenceOperators, v, k: int):
    """An approximation of ``L_k v`` with ``M_{k-1}^{-1}`` replaced by its preconditioner in
    ``operators``. It is linear and symmetric positive definite, so unlike the exact Hodge
    Laplacian it can be used inside a Krylov solver."""
    return laplacian_with(seq, v, k, lambda w, j: _atom(operators, "mass", j, seq.dirichlet).apply(w))


# ---------------------------------------------------------------------------
# Solves
# ---------------------------------------------------------------------------

def parity_projectors(seq, k: int, b):
    """The projections onto the stellarator parity of ``b`` in the ``k``-form space of a
    half-period sequence, or ``None`` on a full-period one."""
    pj = seq.free_projector(k)
    return None if pj is None else pj.projectors(b)


def _parity_pair(seq, k: int, b):
    """The parity projections of degrees ``k`` and ``k - 1``, both at the parity of ``b``."""
    pj = seq.free_projector(k)
    if pj is None:
        return None, None
    sign = pj.parity(b)
    return pj.with_sign(sign), seq.free_projector(k - 1).with_sign(sign)


def dual_norm(seq, k: int):
    """The norm ``sqrt(r^T P r)`` in which the solves measure a residual ``r`` of the ``k``-form
    space of ``seq``, where ``P`` is the preconditioner of ``M_k``. It approximates the ``M_k^{-1}``
    norm uniformly in the mesh size."""
    P = _seq_atom(seq, "mass", k).apply

    def norm(r):
        return jnp.sqrt(r @ P(r))
    return norm


@eqx.filter_jit
def _mass_solve(seq, rhs, k: int, guess=None, return_info: bool = False):
    """``M_k^{-1} rhs`` by preconditioned conjugate gradients."""
    x, info = solve_singular_cg(
        lambda x: _mass(seq, x, k),
        rhs,
        jnp.zeros((0, rhs.shape[0])),
        precond_matvec=_seq_atom(seq, "mass", k).apply,
        x0=guess,
        tol=seq.tol,
        maxiter=seq.maxiter,
        check_residual=True,
        norm=dual_norm(seq, k),
        parity=parity_projectors(seq, k, rhs),
    )
    return (x, info) if return_info else x


def _pair_loop(seq, k, eps, split, b, guess, vs):
    """Solve ``(eps M_k + L_k) x = b`` by repeated corrections until the true residual is small.

    The unknowns are ``x`` and ``w = M_{k-1}^{-1} G^T M_k x`` (``G = G_{k-1}``), and the residual is
    that of the block system::

        upper = b - S_k x - eps M_k x - M_k G w,   lower = G^T M_k x - M_{k-1} w

    The solve is judged on the residual of ``(eps M_k + L_k) x = b`` itself, in the norm of the
    ``k``-forms (:func:`dual_norm`): with ``w`` eliminated it is ``upper - M_k G M_{k-1}^{-1} lower``.

    ``split(r)`` must return ``(dx, dw, info)``, an approximate solution of
    ``(eps M_k + L_k) dx = r`` and its ``dw``. The kernel ``vs`` is removed from the upper residual.
    """
    n_k, n_l = seq.n(k), seq.n(k - 1)
    nu, P_lower = dual_norm(seq, k), _seq_atom(seq, "mass", k - 1).apply
    _, project_dual_k = deflation_projectors(vs, lambda v: _mass(seq, v, k))
    # on a half-period sequence both blocks lose the other parity (that of b)
    par_k, par_l = _parity_pair(seq, k, b)
    pd_k = (lambda r: r) if par_k is None else par_k[1]
    pd_l = (lambda r: r) if par_l is None else par_l[1]

    def project_dual(r):
        return jnp.concatenate([pd_k(project_dual_k(r[:n_k])), pd_l(r[n_k:])])

    def residual(p):
        x, w = p[:n_k], p[n_k:]
        Mx = _mass(seq, x, k)
        upper = b - _stiffness(seq, x, k) - eps * Mx - _mass(seq, _grad(seq, w, k - 1), k)
        lower = _grad(seq, Mx, k - 1, transpose=True) - _mass(seq, w, k - 1)
        return project_dual(jnp.concatenate([upper, lower]))

    def norm(r):
        # the residual of (eps M_k + L_k) x = b, which is upper - M_k G M_{k-1}^-1 lower once w is eliminated,
        # with the preconditioner of M_{k-1} for its inverse as in dual_norm
        return nu(r[:n_k] - _mass(seq, _grad(seq, P_lower(r[n_k:]), k - 1), k))

    def solve(r):
        r_u, r_l = r[:n_k], r[n_k:]
        y = _mass_solve(seq, r_l, k - 1)
        dx, dg, info = split(r_u - _mass(seq, _grad(seq, y, k - 1), k))
        return jnp.concatenate([dx, dg + y]), info

    x0 = jnp.zeros(n_k) if guess is None else guess
    p0 = jnp.concatenate([x0, jnp.zeros(n_l)])
    b_packed = jnp.concatenate([b, jnp.zeros(n_l)])
    p, info = refine(None, solve, b_packed, x0=p0, tol=seq.tol, norm=norm,
                     residual=residual, project_dual=project_dual)
    return p[:n_k], info


def _hat_solve(seq, b, k: int, tol, check_residual=False):
    """Solve ``L^_k x = b`` by conjugate gradients to tolerance ``tol``, harmonic forms removed.

    ``L^_k = S_k + M_k G W G^T M_k`` (``G = G_{k-1}``), with ``W`` the preconditioner of
    ``M_{k-1}``, is symmetric positive definite, while ``S_k`` alone is singular on the exact forms.
    For a ``b`` orthogonal to the exact forms both have the same solution, for any SPD ``W``. For any
    other ``b`` the co-exact part of ``x`` solves ``S_k x = b`` with the part of ``b`` that the exact
    forms see left out, and ``G_k x`` only depends on that part.

    With ``check_residual`` the true residual is checked after the conjugate gradients
    (:func:`mrx.solvers.refine`). Without it the solve stops on the residual of its recurrence, as it
    does inside :func:`_hodge_split`.
    """
    W = _seq_atom(seq, "mass", k - 1).apply

    def L_hat(x):
        Mx = _mass(seq, x, k)
        return _stiffness(seq, x, k) + _mass(seq, _grad(seq, W(_grad(seq, Mx, k - 1, transpose=True)), k - 1), k)

    return solve_singular_cg(
        L_hat, b,
        mass_matvec=lambda v: _mass(seq, v, k),
        precond_matvec=_seq_atom(seq, "laplacian", k).apply,
        vs=_harmonic(seq, k),
        tol=tol,
        maxiter=seq.maxiter,
        check_residual=check_residual,
        norm=dual_norm(seq, k),
        parity=parity_projectors(seq, k, b),
    )


def _k0_solve(seq, b, tol, guess=None, check_residual=False):
    """Solve ``S_0 x = b`` by conjugate gradients, harmonic forms removed.

    With ``check_residual`` the true residual is checked after the conjugate gradients
    (:func:`mrx.solvers.refine`). Without it the solve stops on the residual of its recurrence, as it
    does inside :func:`_hodge_split`.
    """
    return solve_singular_cg(
        lambda x: _stiffness(seq, x, 0),
        b,
        mass_matvec=lambda x: _mass(seq, x, 0),
        precond_matvec=_seq_atom(seq, "laplacian", 0).apply,
        x0=guess,
        vs=_harmonic(seq, 0),
        tol=tol,
        maxiter=seq.maxiter,
        check_residual=check_residual,
        norm=dual_norm(seq, 0),
        parity=parity_projectors(seq, 0, b),
    )


def _hodge_split(seq, b, k, guess):
    """``L_k^{-1} b`` for k = 1, 2, by three symmetric positive definite solves.

    Because ``G^T S_k = 0`` (``G = G_{k-1}``), the exact part of the solution splits off::

        S_{k-1} g = G^T b,   S_k x_perp = b - M_k G g,
        S_{k-1} a = M_{k-1} g - G^T M_k x_perp,   x = x_perp + G a

    Each singular ``S`` is solved through :func:`_hat_solve` (or :func:`_k0_solve` for degree 0),
    and :func:`_pair_loop` repeats the split until the residual is small. Each of the three stops at a
    third of the tolerance.
    """

    # the residual of the split is at most the sum of those of its three solves, so each gets a third
    def solve(b, j):
        if j == 0:
            return _k0_solve(seq, b, seq.tol / 3.0)
        return _hat_solve(seq, b, j, seq.tol / 3.0)

    def split(b):
        # g equals M_{k-1}^-1 G^T M x of the returned x, the second unknown of _pair_loop
        g, _ = solve(_grad(seq, b, k - 1, transpose=True), k - 1)
        x_perp, info = solve(b - _mass(seq, _grad(seq, g, k - 1), k), k)
        a, _ = solve(_mass(seq, g, k - 1) - _grad(seq, _mass(seq, x_perp, k), k - 1, transpose=True), k - 1)
        return x_perp + _grad(seq, a, k - 1), g, info

    return _pair_loop(seq, k, 0.0, split, b, guess, _harmonic(seq, k))


@eqx.filter_jit
def _laplacian_solve(seq, rhs, k: int, guess=None, return_info: bool = False):
    """``L_k^{-1} rhs``, the solution orthogonal (in ``M_k``) to the harmonic forms. k = 0 uses
    conjugate gradients, k = 1, 2 split into three conjugate-gradient solves, and k = 3 calls
    :func:`_saddle_solve`."""
    if k == 0:
        u, info = _k0_solve(seq, rhs, seq.tol, guess=guess, check_residual=True)
    elif k == 3:
        u, _, info = _saddle_solve(seq, rhs, guess=guess)
    else:
        u, info = _hodge_split(seq, rhs, k, guess)
    return (u, info) if return_info else u


def _coclosed_solve(seq, b, k: int, guess=None, tol=None):
    """``(x, info)`` with ``S_k x = b`` for a right-hand side ``b`` in the range of ``G_k^T``.

    ``S_k`` is singular. Its kernel are the closed forms, so ``x`` is only fixed up to a closed form and no
    gauge is imposed. Use this where only ``G_k x`` is needed. Conjugate gradients converge on the singular
    system because ``b`` is consistent, which holds when ``b`` is literally ``G_k^T`` of something. A right-hand side that is only approximately in that range, as inside
    the Hodge split, makes the iteration diverge. Use :func:`_laplacian_solve` for those.

    These are the steps of the Hodge split that such a right-hand side leaves. It is solved like every other
    system (:func:`mrx.solvers.solve_singular_cg`): conjugate gradients with the preconditioner of ``L_k``,
    harmonic forms removed, stopped on the residual in :func:`dual_norm` at ``tol`` (``seq.tol`` by default).
    ``info`` is the signed iteration count.
    """
    return solve_singular_cg(
        lambda v: _stiffness(seq, v, k), b, _harmonic(seq, k),
        precond_matvec=_seq_atom(seq, "laplacian", k).apply,
        tol=seq.tol if tol is None else tol,
        maxiter=seq.maxiter,
        mass_matvec=lambda v: _mass(seq, v, k),
        x0=guess,
        norm=dual_norm(seq, k),
        parity=parity_projectors(seq, k, b),
    )


#: The parts of a Laplacian solve (:meth:`Laplacian.solve`).
PARTS = ("full", "coexact", "exact")


def _coexact_solve(seq, c, j: int, gauge: bool = True, guess=None, tol=None):
    """``(y, info)`` with ``S_j y = c`` for a ``c`` orthogonal to the closed forms.
    With ``gauge`` ``y`` is co-exact (``G_{j-1}^T M_j y = 0``), which fixes it. Without it ``y`` is left with a
    closed form that nothing controls, and only ``G_j y`` is determined. Degree 0 has no exact forms. ``guess``
    starts the stiffness solve."""
    if j == 0:
        return _k0_solve(seq, c, seq.tol if tol is None else tol, guess=guess, check_residual=True)
    y, info = _coclosed_solve(seq, c, j, guess, tol)
    if gauge:
        # only G q enters, so q needs no gauge of its own
        q, info_q = _coexact_solve(seq, -_grad(seq, _mass(seq, y, j), j - 1, transpose=True), j - 1, gauge=False)
        y = y + _grad(seq, q, j - 1)
        info = _merge_info(info, info_q)
    return y, info


def _merge_info(a, b):
    """The signed iteration count of two solves: the sum, negative if either did not converge. A solve that
    starts converged counts zero."""
    return jnp.where((a >= 0) & (b >= 0), 1, -1) * (jnp.abs(a) + jnp.abs(b))


@eqx.filter_jit
def _part_solve(seq, b, k: int, part: str, guess=None, gauge: bool = True, tol=None, return_info: bool = False):
    """``L_k^{-1} b`` for a ``b`` that has only the co-exact or only the exact part (:meth:`Laplacian.solve`),
    k = 0 to 3::

        coexact:  S_k x = b,   x co-exact (with ``gauge``)
        exact:    S_{k-1} g = G^T b,   L^_{k-1} a = M_{k-1} g,   x = G a      (G = G_{k-1})

    The co-exact solve and the first of the exact part are solves of the stiffness alone on a consistent
    right-hand side (:func:`_coclosed_solve`). ``M_{k-1} g`` is consistent only as far as ``g`` is co-exact,
    and a stiffness solve does not get below what is left. So the second solve is one of the regularised
    ``L^_{k-1}`` (:func:`_hat_solve`), whose solution has the same ``G a`` for any right-hand side, and ``g``
    needs no gauge. At degree 0 there is nothing to regularise against: ``L^_0`` is ``S_0``, and for k = 1
    both solves are of ``S_0``.
    """
    if part == "coexact":
        x, info = _coexact_solve(seq, b, k, gauge, guess, tol)
    else:
        j = k - 1
        g, info_g = _coexact_solve(seq, _grad(seq, b, j, transpose=True), j, gauge=False)
        a, info_a = _regularised_solve(seq, _mass(seq, g, j), j)
        x, info = _grad(seq, a, j), _merge_info(info_g, info_a)
    return (x, info) if return_info else x


def _regularised_solve(seq, c, j: int):
    """``(a, info)`` with ``L^_j a = c`` (:func:`_hat_solve`), which is ``S_0 a = c`` at degree 0."""
    if j == 0:
        return _k0_solve(seq, c, seq.tol, check_residual=True)
    return _hat_solve(seq, c, j, seq.tol)


@eqx.filter_jit
def _saddle_solve(seq, rhs, guess=None, sigma_guess=None):
    """``L_3^{-1} rhs`` by MINRES on the saddle-point system, returning ``(u, sigma, info)``::

        | 0        D_2  | | u     |   | rhs |
        | D_2^T   -M_2  | | sigma | = | 0   |

    ``sigma = M_2^{-1} D_2^T u`` is the weak gradient of ``u``, the part that the Leray projection
    removes. Harmonic 3-forms are removed from ``u``. Pass ``guess`` and ``sigma_guess`` together to
    warm-start from a previous solve.
    """
    def saddle_res(u, s):
        return (_stiffness(seq, u, 3) + _weak_derivative(seq, s, 2),
                _weak_derivative(seq, u, 2, transpose=True) - _mass(seq, s, 2))

    P_lower = _seq_atom(seq, "mass", 2).apply

    def eliminate_lower(r):
        # D_2 M_2^-1 r, with the preconditioner of M_2 for its inverse
        return _weak_derivative(seq, P_lower(r), 2)

    parity_upper, parity_lower = _parity_pair(seq, 3, rhs)
    return solve_saddle_point_minres(
        stiffness_matvec=lambda x: _stiffness(seq, x, 3),
        derivative_matvec=lambda s: _weak_derivative(seq, s, 2),
        derivative_T_matvec=lambda u: _weak_derivative(seq, u, 2, transpose=True),
        mass_lower_matvec=lambda s: _mass(seq, s, 2),
        b_upper=rhs,
        n_upper=seq.n(3),
        n_lower=seq.n(2),
        precond_upper=_seq_atom(seq, "laplacian", 3).apply,
        precond_lower=_seq_atom(seq, "mass", 2).apply,
        mass_upper_matvec=lambda x: _mass(seq, x, 3),
        vs_upper=_harmonic(seq, 3),
        x0_upper=guess,
        x0_lower=sigma_guess,
        tol=seq.tol,
        maxiter=seq.maxiter,
        saddle_res=saddle_res,
        norm_upper=dual_norm(seq, 3),
        norm_lower=dual_norm(seq, 2),
        parity_upper=parity_upper, parity_lower=parity_lower,
        eliminate_lower=eliminate_lower,
    )


@eqx.filter_jit
def _shifted_solve(seq, rhs, k: int, eps, guess=None, return_info: bool = False):
    """``(M_k + eps L_k)^{-1} rhs``, by three symmetric positive definite solves, as the Hodge split.

    With ``G = G_{k-1}``, ``M = M_k``, ``M' = M_{k-1}``, and because ``G^T S_k = 0`` and ``G^T M G = S_{k-1}``::

        (M' + eps S_{k-1}) z = G^T b,      (M + eps S_k) x_perp = b - eps M G z,
        (M' + eps S_{k-1}) a = eps (M' z - G^T M x_perp),      x = x_perp + G a

    solves ``(M + eps L_k) x = b`` whatever the error of ``z``: it enters the right-hand sides of the second and
    the third solve so that it cancels, as ``g`` does in :func:`_hodge_split`. The third solve also takes up the
    gradient part ``G^T r`` of the residual ``r`` of the second, which ``S_k`` does not see. So every solve can be
    judged on its own residual, at a third of the tolerance, and :func:`_pair_loop` judges the result on the
    residual of the posed equation. With exact solves ``a = 0`` and ``x_perp`` is
    ``(M + eps S_k)^{-1} b - eps G (M' + eps S_{k-1})^{-1} G^T b``. For k = 0 only one solve is present.

    The preconditioner of ``M_j + eps S_j`` is the shifted atom plus, for ``j >= 1``, the ``(j-1)``-form
    Laplacian atom on gradients, ``G P_{j-1} G^T``. In the ``M``-orthogonal Hodge decomposition
    ``x = G phi + y`` the operator is block-diagonal, because ``S_j G = 0``: on gradients it is
    ``G^T M_j G = S_{j-1}`` for every ``eps``. The shifted atom is a model of ``M_j + eps L_j`` and is
    too small there by ``1 / (1 + eps lambda)``. Without the gradient part the 1-form solve on li383
    (48,64,24) takes 4228 iterations at ``eps = 1e-2`` and does not converge in 5000 at ``eps = 1``,
    with it 220 and 274 (2026-10-14).
    """
    def precondition(j):
        shifted = _seq_atom(seq, "laplacian", j).shifted_stiffness_apply(eps)
        if j == 0:
            return shifted
        lower = _seq_atom(seq, "laplacian", j - 1)
        return lambda x: shifted(x) + _grad(seq, lower.apply(_grad(seq, x, j - 1, transpose=True)), j - 1)

    def shifted_cg(j, b, tol, x0=None, check_residual=False):
        return solve_singular_cg(
            lambda x: _mass(seq, x, j) + eps * _stiffness(seq, x, j),
            b,
            jnp.zeros((0, b.shape[0])),
            precond_matvec=precondition(j),
            x0=x0,
            tol=tol,
            maxiter=seq.maxiter,
            check_residual=check_residual,
            norm=dual_norm(seq, j),
            parity=parity_projectors(seq, j, b),
        )

    if k == 0:
        x, info = shifted_cg(0, rhs, seq.tol, x0=guess, check_residual=True)
        return (x, info) if return_info else x

    def split(b):
        # the residual of the split is at most the sum of those of its three solves, so each gets a third
        z, info_z = shifted_cg(k - 1, _grad(seq, b, k - 1, transpose=True), seq.tol / 3.0)
        x_perp, info = shifted_cg(k, b - eps * _mass(seq, _grad(seq, z, k - 1), k), seq.tol / 3.0)
        a, info_a = shifted_cg(k - 1, eps * (_mass(seq, z, k - 1)
                                             - _grad(seq, _mass(seq, x_perp, k), k - 1, transpose=True)),
                               seq.tol / 3.0)
        # w = M'^-1 G^T M x of the returned x, the second unknown of _pair_loop: G^T M x = M' (z - a / eps)
        total = jnp.abs(info_z) + jnp.abs(info) + jnp.abs(info_a)
        ok = (info_z >= 0) & (info >= 0) & (info_a >= 0)
        return x_perp + _grad(seq, a, k - 1), z - a / eps, jnp.where(ok, total, -total)

    # solve (1/eps) M x + L x = b / eps. This operator is SPD, so there is no kernel to remove
    x, info = _pair_loop(seq, k, 1.0 / eps, lambda r: split(eps * r), rhs / eps, guess,
                         jnp.zeros((0, seq.n(k))))
    return (x, info) if return_info else x


# ---------------------------------------------------------------------------
# The operator objects behind seq.M, seq.L, seq.S, seq.G, seq.D, seq.P and seq.shifted
# ---------------------------------------------------------------------------

class _Operator(eqx.Module):
    """An operator of degree ``k`` on the spaces of ``seq``. It holds no arrays of its own."""
    seq: object
    k: int = eqx.field(static=True)


class _Symmetric(_Operator):
    @property
    def T(self):
        """The transpose, which is the operator itself."""
        return self


class MassMatrix(_Symmetric):
    """The mass matrix ``M_k``, the Gram matrix of the ``k``-form basis in the L2 inner product."""

    def __matmul__(self, v):
        return _mass(self.seq, v, self.k)

    def solve(self, b, guess=None, return_info=False):
        """``M_k^{-1} b`` by preconditioned conjugate gradients.

        ``guess`` warm-starts the solve. With ``return_info`` it returns ``(x, info)``, where ``info`` is the signed
        iteration count of :mod:`mrx.solvers` (positive when converged)."""
        return _mass_solve(self.seq, b, self.k, guess=guess, return_info=return_info)

    def precondition(self, v):
        """The preconditioner of ``M_k`` (an approximation of ``M_k^{-1}``) applied to ``v``."""
        return _seq_atom(self.seq, "mass", self.k).apply(v)


class Laplacian(_Symmetric):
    """The Hodge Laplacian ``L_k = S_k + D_{k-1} M_{k-1}^{-1} D_{k-1}^T``. For k >= 1 its apply
    contains a mass solve, so it costs far more than the stiffness ``S_k``."""

    def __matmul__(self, v):
        return _laplacian(self.seq, v, self.k)

    def solve(self, b, guess=None, return_info=False, part="full", gauge=True, tol=None):
        """``L_k^{-1} b``, the solution orthogonal (in ``M_k``) to the harmonic forms.

        The harmonic forms come from :func:`mrx.nullspace.compute_nullspaces`. While they are still
        zero nothing is removed. ``guess`` and ``return_info`` work as in :meth:`MassMatrix.solve`.

        The solution is the sum of a co-exact form (``G_{k-1}^T M_k x = 0``) and an exact one
        (``x = G_{k-1} a``). ``part`` says which of them ``b`` has, and leaves out the solves of the other:

        - ``"full"``: any ``b``. Three solves for k = 1, 2.
        - ``"coexact"``: ``b`` is orthogonal to the closed forms, as ``b = G_k^T w`` is. The solution
          is co-exact: one solve of the stiffness ``S_k x = b`` and a gauge solve of degree ``k - 1``.
        - ``"exact"``: ``b = M_k G_{k-1} c``. The solution is exact: two solves of degree ``k - 1``.

        With ``part="coexact"`` and ``gauge=False`` the gauge solve is left out. What comes back is then
        NOT the solution: it is the solution plus a closed form that nothing controls, and only its
        derivative ``G_k x`` is the same. Use it where only ``G_k x`` enters. ``tol`` replaces ``seq.tol``
        as the relative residual of the stiffness solve.

        ``guess`` starts the full solve, and the co-exact solve without its gauge (an earlier result of
        the same call; 15 % fewer iterations from one relaxation step to the next on li383 (48,64,24)).
        The other part solves take none: the exact part does not iterate on ``x``, and the gauge solve
        of an already co-exact guess has nothing but round-off on its right-hand side.

        The stiffness solve only converges on a right-hand side of that kind. A ``b`` with the other part as well
        returns a negative info. Every 0-form solution is co-exact and every 3-form solution is exact:
        asking for the other part raises."""
        if part not in PARTS:
            raise ValueError(f"part must be one of {PARTS}, got {part!r}")
        if (self.k, part) in ((0, "exact"), (3, "coexact")):
            raise ValueError(f"the solution of L_{self.k} has no {part} part")
        if part != "coexact" and (not gauge or tol is not None):
            raise ValueError("gauge=False and tol belong to part='coexact'")
        if part == "full":
            return _laplacian_solve(self.seq, b, self.k, guess=guess, return_info=return_info)
        if guess is not None and (part == "exact" or (gauge and self.k > 0)):
            raise ValueError("of the part solves only part='coexact' with gauge=False takes a guess")
        return _part_solve(self.seq, b, self.k, part, guess=guess, gauge=gauge, tol=tol, return_info=return_info)

    def precondition(self, v):
        """The preconditioner of ``L_k`` that the Laplacian solves use, applied to ``v``."""
        return _seq_atom(self.seq, "laplacian", self.k).apply(v)


class Stiffness(_Symmetric):
    """The stiffness ``S_k = G_k^T M_{k+1} G_k``, zero for k = 3."""

    def __matmul__(self, v):
        return _stiffness(self.seq, v, self.k)


class ExteriorDerivative(_Operator):
    """The exterior derivative ``G_k`` (grad, curl, div for k = 0, 1, 2), from the ``k``-forms to
    the ``(k+1)``-forms. ``.T`` is its transpose ``G_k^T``, from dual ``(k+1)``-forms to dual
    ``k``-forms."""
    transpose: bool = eqx.field(static=True, default=False)

    def __matmul__(self, v):
        return _grad(self.seq, v, self.k, transpose=self.transpose)

    @property
    def T(self):
        return ExteriorDerivative(self.seq, self.k, not self.transpose)


class WeakDerivative(_Operator):
    """The weak derivative ``D_k = M_{k+1} G_k``, the exterior derivative tested against the
    ``(k+1)``-forms. ``.T`` is ``D_k^T = G_k^T M_{k+1}``."""
    transpose: bool = eqx.field(static=True, default=False)

    def __matmul__(self, v):
        return _weak_derivative(self.seq, v, self.k, transpose=self.transpose)

    @property
    def T(self):
        return WeakDerivative(self.seq, self.k, not self.transpose)


class Projection(eqx.Module):
    """The metric-free pairing ``P_ij = int Lambda^{k_out}_i . Lambda^{k_in}_j`` over the logical
    domain. It takes a ``k_in``-form to a dual ``k_out``-form, for ``(k_in, k_out)`` = (1, 2),
    (2, 1), (0, 3) and (3, 0). ``.T`` is the pairing in the other direction."""
    seq: object
    k_in: int = eqx.field(static=True)
    k_out: int = eqx.field(static=True)

    def __matmul__(self, v):
        return _projection(self.seq, v, self.k_in, self.k_out)

    @property
    def T(self):
        return Projection(self.seq, self.k_out, self.k_in)


class ShiftedLaplacian(eqx.Module):
    """The shifted Laplacian ``M_k + eps L_k`` of ``seq.shifted(k, eps)``. ``eps`` may be a traced
    array. A Python float is compiled in, so every new float value recompiles."""
    seq: object
    k: int = eqx.field(static=True)
    eps: object

    def solve(self, b, guess=None, return_info=False):
        """``(M_k + eps L_k)^{-1} b`` by two symmetric positive definite solves. ``guess`` and
        ``return_info`` work as in :meth:`MassMatrix.solve`, and ``info`` counts the iterations of
        both solves together."""
        return _shifted_solve(self.seq, b, self.k, self.eps, guess=guess, return_info=return_info)


class OperatorFamily:
    """The operators of one kind on a sequence, indexed by the form degree: ``seq.M[k]``."""

    def __init__(self, seq, cls, degrees):
        self._seq, self._cls, self._degrees = seq, cls, degrees

    def __getitem__(self, k):
        if k not in self._degrees:
            raise IndexError(f"{self._cls.__name__} is defined for {self._degrees}, got {k!r}")
        return self._cls(self._seq, *k) if isinstance(k, tuple) else self._cls(self._seq, int(k))
