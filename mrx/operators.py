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

import mrx.double_single as ds
from mrx.mass import sumfact_apply
from mrx.precision import PAIRS, lift, to_working
from mrx.solvers import deflation_projectors, refine, solve_singular_cg
from mrx.symmetry import symmetrize_like


class SequenceOperators(eqx.Module):
    """The preconditioners and harmonic forms of one geometry, each a dict keyed by ``(k, dirichlet)``.

    ``DeRhamSequence.build_preconditioners`` builds the bundle, and a new geometry needs a new one.
    A sequence and its free view share one bundle, and each picks the entries of its own spaces.
    The harmonic forms start as zeros and are filled in by :func:`mrx.nullspace.compute_nullspaces`.
    Until then the solves remove nothing from the kernel.
    """

    # the atoms of M_k and L_k (mrx.metric_lumping): the norms of the
    # solves, the Laplacian preconditioners and every M^-1 inside an iteration
    mass_lumping: Optional[dict] = None
    laplacian_lumping: Optional[dict] = None
    # the preconditioners of the mass solves where they differ from the atoms (mrx.slab_mass for M_1, M_2)
    mass_model: Optional[dict] = None
    # the preconditioners of the co-closed stiffness solves where they differ from those of L_k (mrx.multigrid)
    stiffness: Optional[dict] = None
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
        (k, d): lift(jnp.zeros((n_vectors(seq.betti_numbers, k, d), seq.extraction[(k, d)].forward_shape[0])))
        for k in range(4) for d in (False, True)})


def new_operators(seq) -> SequenceOperators:
    """An empty bundle for ``seq``, with no preconditioners and zero harmonic forms."""
    return init_nullspaces(seq, SequenceOperators(mass_lumping={}, laplacian_lumping={}, mass_model={}, stiffness={},
                                                  nullspaces={}))


def set_nullspace(operators, k, dirichlet, values):
    """A copy of ``operators`` with the harmonic forms of ``(k, dirichlet)`` replaced by ``values``."""
    spaces = dict(operators.nullspaces)
    spaces[(int(k), bool(dirichlet))] = values
    return _replace(operators, "nullspaces", spaces)


def _atom(operators, kind: str, k: int, dirichlet: bool):
    """The preconditioner of ``(k, dirichlet)`` in the bundle ``operators``, ``kind`` being ``'mass'`` or
    ``'laplacian'``."""
    if operators is None:
        raise ValueError("no operator bundle: call seq.build_preconditioners() after set_geometry_from_map")
    try:
        return getattr(operators, f"{kind}_lumping")[(int(k), bool(dirichlet))]
    except KeyError:
        raise ValueError(
            f"the metric-lumped {kind} atom of k={k}, dirichlet={dirichlet} is not built. "
            "seq.build_preconditioners() builds it for the installed geometry") from None


def _seq_atom(seq, kind: str, k: int):
    """The preconditioner of the ``k``-form space of ``seq``. Every solve needs one, and a half-period sequence
    holds its fields on its parity views, so only those are solved on."""
    if seq.half_period and seq.parity is None:
        raise ValueError("a half-period sequence is solved on its parity views, seq.odd and seq.even")
    return _atom(seq.operators, kind, k, seq.dirichlet)


def _harmonic(seq, k):
    """The harmonic forms of the ``k``-form space of ``seq``, one per row."""
    return seq._require_operators().nullspaces[(int(k), seq.dirichlet)]


#: The preconditioners of the mass solves :func:`assemble_preconditioners` can build: the metric-lumping atom for
#: every degree, or the zeta-slab model (:mod:`mrx.slab_mass`) for ``k = 1, 2`` and the atom for ``k = 0, 3`` (the
#: default). A mass solve stops in the norm of its own preconditioner. The atoms stay the norms of the other solves
#: and every ``M^{-1}`` inside an iteration either way.
MASS_ATOMS = ("lumping", "slab")

#: The Laplacian preconditioners :func:`assemble_preconditioners` can build: the metric-lumping atom for
#: every degree, or in addition the geometric multigrid of ``S_2`` (:mod:`mrx.multigrid`) for the co-closed
#: stiffness solves on the Dirichlet 2-forms.
LAPLACIAN_ATOMS = ("lumping", "multigrid")


def assemble_preconditioners(seq, operators, mass_atom="slab", laplacian_atom="lumping"):
    """A copy of ``operators`` with the mass and Laplacian preconditioners of the Dirichlet and the
    free spaces of every degree built for the installed geometry. ``mass_atom`` is one of
    :data:`MASS_ATOMS` and ``laplacian_atom`` one of :data:`LAPLACIAN_ATOMS`. This needs at least two
    radial elements (``n >= p + 2``)."""
    from mrx.metric_lumping import MetricLumpingLaplacian, MetricLumpingMass   # imports this module
    from mrx.multigrid import StiffnessMultigrid                                # imports this module
    from mrx.slab_mass import SlabMass                                          # imports this module
    if mass_atom not in MASS_ATOMS:
        raise ValueError(f"mass_atom must be one of {MASS_ATOMS}, not {mass_atom!r}")
    if laplacian_atom not in LAPLACIAN_ATOMS:
        raise ValueError(f"laplacian_atom must be one of {LAPLACIAN_ATOMS}, not {laplacian_atom!r}")
    views = [(k, view) for k in range(4) for view in (seq.free, seq)]
    mass = dict(operators.mass_lumping)
    for k, view in views:
        mass[(k, view.dirichlet)] = MetricLumpingMass(view, k)
    # the Laplacian preconditioners are built using the mass atoms
    operators = _replace(operators, "mass_lumping", mass)
    if mass_atom == "slab":
        operators = _replace(operators, "mass_model", {(k, view.dirichlet): SlabMass(view, operators, k)
                                                       for k, view in views if k in (1, 2)})
    laplacian = dict(operators.laplacian_lumping)
    for k, view in views:
        laplacian[(k, view.dirichlet)] = MetricLumpingLaplacian(view, operators, k)
    operators = _replace(operators, "laplacian_lumping", laplacian)
    stiffness = {(2, True): StiffnessMultigrid(seq, operators, 2)} if laplacian_atom == "multigrid" else {}
    return _replace(operators, "stiffness", stiffness)


def _mass_core(seq, k: int):
    """The function ``x -> M_k x`` on the full tensor-product spline DoFs."""
    plan, weights = seq.mass_plan[k], seq._require_geometry().mass_weights[k]
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
    g = seq.incidence[k]
    return g, g.T


def _grad(seq, v, k: int, transpose: bool = False):
    """``G_k v``, the exterior derivative of the ``k``-form ``v``, or ``G_k^T v`` if ``transpose``."""
    g = (seq.g0_grad, seq.g1_curl, seq.g2_div)[k][seq.dirichlet]
    return (g.T if transpose else g) @ v


def _mass(seq, v, k: int):
    """``M_k v``."""
    e = seq.E(k)
    return e @ _mass_core(seq, k)(e.T @ v)


def _projection(seq, v, k_in: int, k_out: int):
    """``P v`` with the metric-free pairing ``P_ij = int Lambda^{k_out}_i . Lambda^{k_in}_j`` over the
    logical domain, for ``(k_in, k_out)`` = (1, 2), (2, 1), (0, 3) or (3, 0)."""
    plan = seq.projection_plan[(k_out, k_in)]
    weights = seq.quad.w.reshape(seq.quad.shape)       # the pairing has no metric
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
        return ds.zeros_like(v)
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
    return laplacian_with(seq, v, k, lambda w, j: _mass_solve(seq, w, j, accurate=isinstance(v, ds.DS)))


def apply_laplacian_approx(seq, operators: SequenceOperators, v, k: int):
    """An approximation of ``L_k v`` with ``M_{k-1}^{-1}`` replaced by its preconditioner in
    ``operators``. It is linear and symmetric positive definite, so unlike the exact Hodge
    Laplacian it can be used inside a Krylov solver."""
    return laplacian_with(seq, v, k, lambda w, j: _atom(operators, "mass", j, seq.dirichlet).apply(to_working(w)))


# ---------------------------------------------------------------------------
# Solves
# ---------------------------------------------------------------------------

def dual_norm(seq, k: int):
    """The norm ``sqrt(r^T P r)`` in which the solves measure a residual ``r`` of the ``k``-form
    space of ``seq``, where ``P`` is the preconditioner of ``M_k``. It approximates the ``M_k^{-1}``
    norm uniformly in the mesh size."""
    P = _seq_atom(seq, "mass", k).apply

    def norm(r):
        return jnp.sqrt(r @ P(r))
    return norm


def _mass_model(seq, k: int):
    """The preconditioner of the solves of ``M_k``: the slab model where it is built, else the atom."""
    atom = _seq_atom(seq, "mass", k)
    models, key = seq.operators.mass_model, (int(k), seq.dirichlet)
    return models[key] if key in models else atom


def _out(x, accurate):
    """A solution as its solve returns it: as it was accumulated if ``accurate``, else as a float array."""
    return x if accurate else to_working(x)


@eqx.filter_jit
def _mass_solve(seq, rhs, k: int, guess=None, tol=None, check_residual: bool = True, return_info: bool = False,
                accurate: bool = False):
    """``M_k^{-1} rhs`` by preconditioned conjugate gradients, preconditioned by :func:`_mass_model`.

    The solve stops in the norm of that preconditioner, which conjugate gradients have without a second apply:
    no ``norm`` is passed. For ``k = 1, 2`` with the slab model that is the slab's norm, for the atom it is the norm
    of :func:`dual_norm`. ``tol`` replaces ``seq.tol``. Without ``check_residual`` the solve stops on the residual of
    its recurrence, as a solve inside another iteration does."""
    x, info = solve_singular_cg(
        lambda x: _mass(seq, x, k),
        rhs,
        jnp.zeros((0, rhs.shape[0])),
        precond_matvec=_mass_model(seq, k).apply,
        x0=guess,
        tol=seq.tol if tol is None else tol,
        maxiter=seq.maxiter,
        check_residual=check_residual,
    )
    x = _out(x, accurate)
    return (x, info) if return_info else x


def _pair_loop(seq, k, eps, split, b, guess, vs, guess_lower=None):
    """Solve ``(eps M_k + L_k) x = b`` by repeated corrections until the true residual is small.

    The unknowns are ``x`` and ``w = M_{k-1}^{-1} G^T M_k x`` (``G = G_{k-1}``), and the residual is
    that of the block system::

        upper = b - S_k x - eps M_k x - M_k G w,   lower = G^T M_k x - M_{k-1} w

    The solve is judged on the residual of ``(eps M_k + L_k) x = b`` itself, in the norm of the
    ``k``-forms (:func:`dual_norm`): with ``w`` eliminated it is ``upper - M_k G M_{k-1}^{-1} lower``.

    ``split(r, rtol)`` must return ``(dx, dw, info)``, a solution of ``(eps M_k + L_k) dx = r`` to the relative
    residual ``rtol`` and its ``dw``. The kernel ``vs`` is removed from ``b`` and from the upper residual.
    ``guess`` and ``guess_lower`` start ``x`` and ``w``. Returns ``(x, w, info)``.
    """
    n_k, n_l = seq.n(k), seq.n(k - 1)
    nu, P_lower = dual_norm(seq, k), _seq_atom(seq, "mass", k - 1).apply
    _, project_dual_k = deflation_projectors(to_working(vs), lambda v: _mass(seq, v, k))
    # the kernel leaves b once, to the accuracy of b (:func:`mrx.solvers.refine`)
    b = (deflation_projectors(vs, lambda v: _mass(seq, v, k))[1] if PAIRS else project_dual_k)(lift(b))

    def project(r):
        return jnp.concatenate([project_dual_k(r[:n_k]), r[n_k:]])

    def residual(p):
        x, w = p[:n_k], p[n_k:]
        Mx = _mass(seq, x, k)
        upper = b - _stiffness(seq, x, k) - Mx * eps - _mass(seq, _grad(seq, w, k - 1), k)
        lower = _grad(seq, Mx, k - 1, transpose=True) - _mass(seq, w, k - 1)
        return ds.concatenate([upper, lower])

    def norm(r):
        # the residual of (eps M_k + L_k) x = b, which is upper - M_k G M_{k-1}^-1 lower once w is eliminated,
        # with the preconditioner of M_{k-1} for its inverse as in dual_norm
        return nu(r[:n_k] - _mass(seq, _grad(seq, P_lower(r[n_k:]), k - 1), k))

    def solve(r, rtol):
        r_u, r_l = r[:n_k], r[n_k:]
        y = _mass_solve(seq, r_l, k - 1)
        dx, dg, info = split(r_u - _mass(seq, _grad(seq, y, k - 1), k), rtol)
        return jnp.concatenate([dx, dg + y]), info

    x0 = jnp.zeros(n_k) if guess is None else guess
    w0 = jnp.zeros(n_l) if guess_lower is None else guess_lower
    p0 = jnp.concatenate([x0, w0])
    b_packed = ds.concatenate([b, jnp.zeros(n_l)])
    p, info = refine(None, solve, b_packed, x0=p0, tol=seq.tol, norm=norm,
                     residual=residual, project=project)
    return p[:n_k], p[n_k:], info


def _stiffness_solve(seq, b, k: int, tol, guess=None, check_residual=False):
    """``(x, info)`` with ``S_k x = b`` by conjugate gradients, harmonic forms removed, stopped on the residual in
    :func:`dual_norm` at ``tol``. The preconditioner is that of ``L_k``, or the stiffness preconditioner where one
    is built (the multigrid of :mod:`mrx.multigrid`).

    ``S_k`` is singular on the closed forms, so ``x`` is fixed only up to a closed form and only ``G_k x`` is
    determined. Conjugate gradients converge on the singular system only as far as ``b`` is consistent, in the
    range of ``G_k^T``. A right-hand side only approximately in that range makes the iteration stall at what is
    left and then diverge, so the callers inside the Hodge split and the exact part make it consistent to below
    their tolerance (:func:`_gauge`, :func:`_hodge_split`).

    With ``check_residual`` the true residual is checked after the conjugate gradients
    (:func:`mrx.solvers.refine`). Without it the solve stops on the residual of its recurrence, as it does
    inside :func:`_hodge_split`.
    """
    atom = _seq_atom(seq, "laplacian", k)
    stiffness, key = seq.operators.stiffness, (int(k), seq.dirichlet)
    return solve_singular_cg(
        lambda v: _stiffness(seq, v, k), b, _harmonic(seq, k),
        precond_matvec=(stiffness[key] if key in stiffness else atom).apply,
        mass_matvec=lambda v: _mass(seq, v, k),
        x0=guess,
        tol=tol,
        maxiter=seq.maxiter,
        check_residual=check_residual,
        norm=dual_norm(seq, k),
    )


def _shifted_cg(seq, j: int, delta, b, tol, x0=None, check_residual=False):
    """``(x, info)`` with ``(delta M_j + S_j) x = b`` by preconditioned conjugate gradients.

    The preconditioner is the shifted atom of ``M_j + eps S_j`` (``eps = 1 / delta``) plus, for ``j >= 1``,
    the ``(j-1)``-form Laplacian atom on gradients, ``G P_{j-1} G^T``. In the ``M``-orthogonal Hodge
    decomposition ``x = G phi + y`` the operator is block-diagonal, because ``S_j G = 0``: on gradients it is
    ``G^T M_j G = S_{j-1}`` up to the factor ``delta``. The shifted atom is a model of ``M_j + eps L_j`` and is
    too small there by ``1 / (1 + eps lambda)``. Without the gradient part the 1-form solve on li383
    (48,64,24) takes 4228 iterations at ``eps = 1e-2`` and does not converge in 5000 at ``eps = 1``, with it
    220 and 274 (2026-10-14).
    """
    shifted = _seq_atom(seq, "laplacian", j).shifted_stiffness_apply(1.0 / delta)
    if j == 0:
        P = shifted
    else:
        lower = _seq_atom(seq, "laplacian", j - 1)

        def P(x):
            return shifted(x) + _grad(seq, lower.apply(_grad(seq, x, j - 1, transpose=True)), j - 1)
    return solve_singular_cg(
        lambda x: _mass(seq, x, j) * delta + _stiffness(seq, x, j),
        b,
        jnp.zeros((0, b.shape[0])),
        precond_matvec=P,
        x0=x0,
        tol=tol,
        maxiter=seq.maxiter,
        check_residual=check_residual,
        norm=dual_norm(seq, j),
    )


def _lower_norm(seq, k: int):
    """The norm ``r -> dual_norm(seq, k)(M_k G P r)`` (``G = G_{k-1}``, ``P`` the preconditioner of
    ``M_{k-1}``). A residual ``r`` of the second unknown of :func:`_pair_loop`, a (k-1)-form, counts with
    this norm in the residual of the k-form equation."""
    nu, P = dual_norm(seq, k), _seq_atom(seq, "mass", k - 1).apply
    return lambda r: nu(_mass(seq, _grad(seq, P(r), k - 1), k))


def _lower_solve(seq, k: int, solve, apply, c, rtol, target, consistent=None):
    """``(y, info)`` with ``A y = c`` for the (k-1)-form unknown of a split of a k-form solve, stopped where
    the residual ``r = c - A y`` counts at most ``target`` in the residual of the k-form equation. ``rtol`` is
    the tolerance of the pass of the split.

    ``solve(c, tol)`` solves ``A y = c`` to ``tol`` relative to ``c`` in ``dual_norm(seq, k - 1)``, and
    ``apply`` is ``A``. The residual counts as ``M_k G P r`` (:func:`_lower_norm`), which weights its high
    modes against ``dual_norm(seq, k - 1)`` by up to the square root of the largest eigenvalue of ``S_{k-1}``
    against ``M_{k-1}``. How much depends on the right-hand side as much as on the mesh. So the first solve
    stops at a third of ``rtol``, the ratio of the two norms is read off its residual, and a second
    solve of that residual takes it to ``target``. The ratio changes little as a solve converges, and the
    outer check of :func:`_pair_loop` catches what is left. Measuring :func:`_lower_norm` in every iteration
    instead costs a mass apply each.

    A singular ``A`` (the stiffness ``S_{k-1}``) needs ``consistent(r, tol)``, which returns ``r`` without its part
    outside the range of ``A`` up to ``tol`` relative to ``r`` in ``dual_norm(seq, k - 1)``, and its count. ``c`` is
    consistent only as far as the gauge of the Hodge split made ``z`` co-exact. The first solve takes the rest of
    ``c`` to a third of the tolerance and leaves that part in ``r``, about 2e-2 of it in ``dual_norm(seq, k - 1)`` on
    li383 p = 2. The second solve asks for ``target / _lower_norm(r)`` of ``r``, which falls with the smallest
    angular cell at the axis. Below that 2e-2 the conjugate gradients on the inconsistent ``r`` diverge until
    ``seq.maxiter``: about 7e-3 at uniform (36,60,36), 1.6e-2 with 30 theta cells on the axis rings, against 2.1e-2
    with 15 (2026-10-20). So ``r`` is made consistent to a tenth of what the second solve asks for.
    """
    y, info = solve(c, rtol / 3.0)
    r = c - apply(y)
    tol = target / _lower_norm(seq, k)(r)
    if consistent is not None:
        r, info_c = consistent(r, tol / 10.0)
        info = _merge_info(info, info_c)
    dy, info_r = solve(r, tol)
    return y + dy, _merge_info(info, info_r)


@eqx.filter_jit
def _hodge_split(seq, b, k, guess, delta=None, guess_lower=None):
    """``(x, w, info)`` with ``(delta M_k + L_k) x = b`` by three symmetric solves, k = 1, 2, 3, and
    ``w = M_{k-1}^{-1} G^T M_k x`` (``G = G_{k-1}``). For k = 3 ``w`` is the ``sigma`` of the mixed form
    ``G_2 sigma = M_3^{-1} b``, the part of a 2-form that the Leray projection removes.

    ``delta=None`` is the Laplacian ``L_k`` itself, a positive ``delta`` the shifted Laplacian divided by its
    shift. Because ``G^T S_k = 0`` and ``G^T M_k G = S_{k-1}``, the exact part of the solution splits off::

        (delta M_{k-1} + S_{k-1}) z = G^T b,   (delta M_k + S_k) x_perp = b - M_k G z,
        (delta M_{k-1} + S_{k-1}) a = M_{k-1} z - G^T M_k x_perp,   x = x_perp + G a

    and :func:`_pair_loop` repeats the split until the residual of the posed equation is small, which
    normally takes one pass. With ``delta`` each solve is :func:`_shifted_cg`, positive definite. Without it
    each is a solve of the stiffness alone (:func:`_stiffness_solve`), singular for degrees 1 and 2, harmonic
    forms removed. A singular solve converges only as far as its right-hand side is consistent:

    - That of ``x_perp`` is consistent up to the residual of ``z``, which leaves the exact form ``M_k G e``
      in it (``S_{k-1} e`` that residual). So ``z`` stops where this is a thirtieth of the tolerance relative
      to ``b``, as in :func:`_gauge`. Relative to its own right-hand side the stop would ask the impossible of
      a co-exact ``b``, whose ``G^T b`` is rounding. With ``delta`` it stops at a tenth relative to its own
      right-hand side, since its residual counts up to about twice (li383, graded).
    - For k = 2, 3 that of ``a`` is consistent only as far as ``z`` is co-exact, and ``z`` is gauged.

    At k = 3 without a shift ``S_3 = 0``: the middle solve has nothing to do and is left out, and what remains is
    the exact route, ``x = G_2 a`` with ``S_2 a = M_2 z``. The residual that :func:`_pair_loop` judges is then that
    of the saddle-point system of ``(x, sigma)``.

    ``x_perp`` stops at a third of the tolerance relative to ``b``. Its residual counts at most once, and in a
    later pass ``b`` is a residual whose co-exact part can be far below what ``z`` leaves inconsistent. The
    residual of ``a`` counts through ``M_k G``, 5 to 70 times on li383 graded (48,64,24) and (64,80,32), and
    :func:`_lower_solve` takes it to a third.
    """
    nu, nu_lower = dual_norm(seq, k), dual_norm(seq, k - 1)

    def solve(c, j, tol):
        if delta is not None:
            return _shifted_cg(seq, j, delta, c, tol)
        return _stiffness_solve(seq, c, j, tol)

    def consistent(r, tol):
        # r without its exact part M G q (G = G_{k-2}): S_{k-2} q = G^T r, stopped where the exact form left by
        # the residual of q is at most tol relative to r, as in _gauge
        c = _grad(seq, r, k - 2, transpose=True)
        q, info = _consistent_solve(seq, c, k - 2, tol=tol * nu_lower(r) / dual_norm(seq, k - 2)(c))
        return r - _mass(seq, _grad(seq, q, k - 2), k - 1), info

    def apply_lower(y):
        S = _stiffness(seq, y, k - 1)
        return S if delta is None else _mass(seq, y, k - 1) * delta + S

    def split(b, rtol):
        c = _grad(seq, b, k - 1, transpose=True)
        if delta is None:
            z, info = solve(c, k - 1, rtol / 30.0 * nu(b) / nu_lower(c))
            if k >= 2:
                z, info_q = _gauge(seq, z, k - 1, rtol / 30.0)
                info = _merge_info(info, info_q)
        else:
            z, info = solve(c, k - 1, rtol / 10.0)
        if delta is None and k == 3:
            x_perp, info_x = jnp.zeros_like(b), jnp.zeros((), info.dtype)
        else:
            c = b - _mass(seq, _grad(seq, z, k - 1), k)
            x_perp, info_x = solve(c, k, rtol / 3.0 * nu(b) / nu(c))
        a, info_a = _lower_solve(seq, k, lambda c, tol: solve(c, k - 1, tol), apply_lower,
                                 _mass(seq, z, k - 1) - _grad(seq, _mass(seq, x_perp, k), k - 1, transpose=True),
                                 rtol, nu(b) * rtol / 3.0, consistent if delta is None and k >= 2 else None)
        # w = M_{k-1}^-1 G^T M_k x of the returned x, the second unknown of _pair_loop:
        # G^T M_k x = M_{k-1} (z - delta a)
        w = z if delta is None else z - delta * a
        return x_perp + _grad(seq, a, k - 1), w, _merge_info(_merge_info(info, info_x), info_a)

    if delta is None:
        return _pair_loop(seq, k, 0.0, split, b, guess, _harmonic(seq, k), guess_lower)
    # the shifted operator is positive definite, so there is no kernel to remove
    return _pair_loop(seq, k, delta, split, b, guess, jnp.zeros((0, seq.n(k))), guess_lower)


@eqx.filter_jit
def _laplacian_solve(seq, rhs, k: int, guess=None, return_info: bool = False, accurate: bool = False):
    """``L_k^{-1} rhs``, the solution orthogonal (in ``M_k``) to the harmonic forms. k = 0 uses
    conjugate gradients, k = 1, 2, 3 the Hodge split (:func:`_hodge_split`)."""
    if k == 0:
        u, info = _stiffness_solve(seq, rhs, 0, seq.tol, guess=guess, check_residual=True)
    else:
        u, _, info = _hodge_split(seq, rhs, k, guess)
    u = _out(u, accurate)
    return (u, info) if return_info else u


#: The parts of a Laplacian solve (:meth:`Laplacian.solve`).
PARTS = ("full", "coexact", "exact")


def _consistent_solve(seq, c, j: int, tol=None, guess=None, check_residual=False):
    """``(y, info)`` with ``S_j y = c`` for a ``c`` orthogonal to the closed forms, ``y`` up to a closed form.
    Degree 0 has no closed forms but the constants, and its solve is refined on the true residual. So is every
    solve with ``check_residual``. A solve that is not inside another one passes it in float32, where conjugate
    gradients alone do not reach the tolerance of the sequence."""
    return _stiffness_solve(seq, c, j, seq.tol if tol is None else tol, guess,
                            check_residual=check_residual or j == 0)


@eqx.filter_jit
def _stiffness_solve_jit(seq, b, k: int, guess=None, tol=None, return_info: bool = False, accurate: bool = False):
    """:meth:`Stiffness.solve`."""
    x, info = _consistent_solve(seq, b, k, tol, guess, check_residual=PAIRS)
    x = _out(x, accurate)
    return (x, info) if return_info else x


def _coexact_solve(seq, c, j: int):
    """``(y, info)``: the co-exact ``y`` (``G_{j-1}^T M_j y = 0``) with ``S_j y = c`` for a ``c`` orthogonal to the
    closed forms, the stiffness solve followed by a gauge solve of degree ``j - 1``."""
    y, info = _consistent_solve(seq, c, j, check_residual=PAIRS)
    if j == 0:
        return y, info
    # only G q enters, so q needs no gauge of its own
    q, info_q = _consistent_solve(seq, -_grad(seq, _mass(seq, y, j), j - 1, transpose=True), j - 1,
                                  check_residual=PAIRS)
    return y + _grad(seq, q, j - 1), _merge_info(info, info_q)


def _gauge(seq, y, j: int, tol, check_residual=False):
    """``(y + G q, info)``, the j-form ``y`` (j = 1, 2) made co-exact for a following stiffness solve on
    ``M_j (y + G q)`` (``G = G_{j-1}``, ``S_{j-1} q = -G^T M_j y``).

    The residual ``r`` of ``q`` leaves the exact form ``G e`` with ``S_{j-1} e = r``, and the stiffness solve
    that follows stalls at ``M_j G e``. In :func:`dual_norm` this is the ``S_{j-1}^{-1}`` norm of ``r``, which is
    at most a mesh-independent constant times ``dual_norm(seq, j - 1)(r)``. The right-hand side ``G^T M_j y``
    weights the high modes of ``y`` by up to ``h^-1`` in that norm, so ``q`` stops at ``tol`` relative to
    ``dual_norm(seq, j)(M_j y)``, not relative to its own right-hand side. ``check_residual`` works as in
    :func:`_consistent_solve`.
    """
    My = _mass(seq, y, j)
    c = -_grad(seq, My, j - 1, transpose=True)
    # the norms only set the tolerance, so they take a double-single y rounded
    nu_c = dual_norm(seq, j - 1)(to_working(c))
    # y = 0, the z of a co-exact right-hand side in the Hodge split, has nothing to gauge (0 / 0 otherwise)
    q, info = _consistent_solve(seq, c, j - 1,
                                tol=jnp.where(nu_c > 0, tol * dual_norm(seq, j)(to_working(My)) / nu_c, tol),
                                check_residual=check_residual)
    return y + _grad(seq, q, j - 1), info


def _merge_info(a, b):
    """The signed iteration count of two solves: the sum, negative if either did not converge. A solve that
    starts converged counts zero."""
    return jnp.where((a >= 0) & (b >= 0), 1, -1) * (jnp.abs(a) + jnp.abs(b))


@eqx.filter_jit
def _part_solve(seq, b, k: int, part: str, return_info: bool = False, accurate: bool = False):
    """``L_k^{-1} b`` for a ``b`` that has only the co-exact or only the exact part (:meth:`Laplacian.solve`),
    k = 0 to 3::

        coexact:  S_k x = b,   x co-exact
        exact:    S_{k-1} g = G^T b,   g made co-exact,   S_{k-1} a = M_{k-1} g,   x = G a      (G = G_{k-1})

    All solves are of the stiffness alone (:func:`_stiffness_solve`). ``M_{k-1} g`` is consistent only as far as
    ``g`` is co-exact, and the stiffness solve for ``a`` does not get below what is left. So ``g`` is gauged
    (:func:`_gauge`, one solve of degree ``k - 2``) to a thirtieth of the tolerance. At degree 0 there is nothing
    to gauge, and for k = 1 both solves are of ``S_0``. For k = 3 this is the Hodge split of :func:`_hodge_split`
    without its outer check.
    """
    if part == "coexact":
        x, info = _coexact_solve(seq, b, k)
    else:
        j = k - 1
        # G^T b in the arithmetic of the solve, so that it lies in the range of G^T to its accuracy
        g, info_g = _consistent_solve(seq, _grad(seq, lift(b), j, transpose=True), j, check_residual=PAIRS)
        if j > 0:
            g, info_q = _gauge(seq, g, j, seq.tol / 30.0, check_residual=PAIRS)
            info_g = _merge_info(info_g, info_q)
        a, info_a = _consistent_solve(seq, _mass(seq, g, j), j, check_residual=PAIRS)
        x, info = _grad(seq, a, j), _merge_info(info_g, info_a)
    x = _out(x, accurate)
    return (x, info) if return_info else x


@eqx.filter_jit
def _shifted_stiffness_solve(seq, rhs, k: int, eps, guess=None, return_info: bool = False):
    """``(M_k + eps S_k)^{-1} rhs`` (:meth:`ShiftedStiffness.solve`): one solve of ``delta M_k + S_k`` on
    ``rhs / eps``, ``delta = 1 / eps``."""
    delta = 1.0 / eps
    x, info = _shifted_cg(seq, k, delta, rhs * delta, seq.tol, x0=guess, check_residual=True)
    x = to_working(x)
    return (x, info) if return_info else x


@eqx.filter_jit
def _shifted_solve(seq, rhs, k: int, eps, guess=None, return_info: bool = False):
    """``(M_k + eps L_k)^{-1} rhs`` (:meth:`ShiftedLaplacian.solve`): :func:`_hodge_split` with ``delta = 1 / eps``
    on ``rhs / eps``, and at k = 0, where ``L_0 = S_0``, the one solve of :func:`_shifted_stiffness_solve`."""
    if k == 0:
        return _shifted_stiffness_solve(seq, rhs, 0, eps, guess, return_info)
    delta = 1.0 / eps
    x, _, info = _hodge_split(seq, rhs * delta, k, guess, delta=delta)
    x = to_working(x)
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

    def solve(self, b, guess=None, return_info=False, accurate=False):
        """``M_k^{-1} b`` by preconditioned conjugate gradients.

        ``guess`` warm-starts the solve. In float32 the solve accumulates its solution as a
        :class:`~mrx.double_single.DS` and returns it rounded, or as it is with ``accurate``. Ask for that
        where the solution is subtracted from something close to it. With ``return_info`` it returns
        ``(x, info)``, where ``info`` is the signed iteration count of :mod:`mrx.solvers` (positive when
        converged)."""
        return _mass_solve(self.seq, b, self.k, guess=guess, return_info=return_info, accurate=accurate)

    def precondition(self, v):
        """The mass atom of ``M_k`` (an approximation of ``M_k^{-1}``) applied to ``v``, symmetric to round-off. The
        solves of ``M_k`` may use the slab model."""
        return _seq_atom(self.seq, "mass", self.k).apply(v)


class Laplacian(_Symmetric):
    """The Hodge Laplacian ``L_k = S_k + D_{k-1} M_{k-1}^{-1} D_{k-1}^T``. For k >= 1 its apply
    contains a mass solve, so it costs far more than the stiffness ``S_k``."""

    def __matmul__(self, v):
        return _laplacian(self.seq, v, self.k)

    def solve(self, b, guess=None, return_info=False, part="full", accurate=False):
        """``L_k^{-1} b``, the solution orthogonal (in ``M_k``) to the harmonic forms.

        The harmonic forms come from :func:`mrx.nullspace.compute_nullspaces`. While they are still
        zero nothing is removed. ``guess``, ``return_info`` and ``accurate`` work as in
        :meth:`MassMatrix.solve`.

        The solution is the sum of a co-exact form (``G_{k-1}^T M_k x = 0``) and an exact one
        (``x = G_{k-1} a``). ``part`` says which of them ``b`` has, and leaves out the solves of the other:

        - ``"full"``: any ``b``. Three solves for k = 1, 2, two for k = 3. ``guess`` starts it.
        - ``"coexact"``: ``b`` is orthogonal to the closed forms, as ``b = G_k^T w`` is. The solution
          is co-exact: one stiffness solve (:meth:`Stiffness.solve`) and a gauge solve of degree ``k - 1``.
          Where only ``G_k x`` enters, :meth:`Stiffness.solve` alone is enough.
        - ``"exact"``: ``b = M_k G_{k-1} c``. The solution is exact: two solves of degree ``k - 1``.

        The part solves take no guess. Their stiffness solves only converge on a right-hand side of their kind: a
        ``b`` with the other part as well returns a negative info. In float32 form a co-exact ``b`` from a lifted
        vector, ``G_k^T @ lift(w)`` (:func:`mrx.precision.lift`): ``G_k^T w`` computed in float32 is in the
        range of ``G_k^T`` only to 1e-7, and the solve does not get below that. Every 0-form solution is co-exact and every
        3-form solution is exact: asking for the other part raises."""
        if part not in PARTS:
            raise ValueError(f"part must be one of {PARTS}, got {part!r}")
        if (self.k, part) in ((0, "exact"), (3, "coexact")):
            raise ValueError(f"the solution of L_{self.k} has no {part} part")
        if part == "full":
            return _laplacian_solve(self.seq, b, self.k, guess=guess, return_info=return_info, accurate=accurate)
        if guess is not None:
            raise ValueError("only the full solve takes a guess")
        return _part_solve(self.seq, b, self.k, part, return_info=return_info, accurate=accurate)

    def precondition(self, v):
        """The preconditioner of ``L_k`` that the Laplacian solves use, applied to ``v``."""
        return _seq_atom(self.seq, "laplacian", self.k).apply(v)


class Stiffness(_Symmetric):
    """The stiffness ``S_k = G_k^T M_{k+1} G_k``, zero for k = 3."""

    def __matmul__(self, v):
        return _stiffness(self.seq, v, self.k)

    def solve(self, b, guess=None, tol=None, return_info=False, accurate=False):
        """``x`` with ``S_k x = b``, for a ``b`` orthogonal to the closed forms, as ``b = G_k^T w`` is.

        ``S_k`` is singular on the closed forms, so ``x`` is fixed only up to a closed form that nothing
        controls: only ``G_k x`` is determined. Use it where only ``G_k x`` enters, a vector potential for
        example, and ``seq.L[k].solve(b, part="coexact")`` for the co-exact solution itself. Conjugate gradients
        converge on the singular system because ``b`` lies in its range. A ``b`` with an exact part returns a
        negative info. In float32 form ``b`` from a lifted vector, as for :meth:`Laplacian.solve`. ``guess`` starts
        the solve (an earlier result of the same call: 15 % fewer iterations from one relaxation step to the next
        on li383 (48,64,24)), ``tol`` replaces ``seq.tol``, and ``return_info`` and ``accurate`` work as in
        :meth:`MassMatrix.solve`. At k = 0 it is the Laplacian solve."""
        if self.k == 3:
            raise ValueError("S_3 is zero")
        return _stiffness_solve_jit(self.seq, b, self.k, guess=guess, tol=tol, return_info=return_info,
                                    accurate=accurate)


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
        """``(M_k + eps L_k)^{-1} b``: three solves for k >= 1, as the Hodge split of :meth:`Laplacian.solve`
        (:func:`_hodge_split`), with ``info`` counting them together, and one at k = 0. ``guess`` and
        ``return_info`` work as in :meth:`MassMatrix.solve`. Where only ``G_k x`` enters, the one solve of
        :class:`ShiftedStiffness` is enough."""
        return _shifted_solve(self.seq, b, self.k, self.eps, guess=guess, return_info=return_info)


class ShiftedStiffness(eqx.Module):
    """``M_k + eps S_k`` of ``seq.shifted_stiffness(k, eps)``, positive definite. ``eps`` may be a traced array.
    A Python float is compiled in, so every new float value recompiles."""
    seq: object
    k: int = eqx.field(static=True)
    eps: object

    def solve(self, b, guess=None, return_info=False):
        """``(M_k + eps S_k)^{-1} b`` by one preconditioned conjugate-gradient solve. For a ``b`` orthogonal to the
        closed forms, as ``b = G_k^T w`` is, it is also ``(M_k + eps L_k)^{-1} b``. For any other ``b`` the two
        differ by an exact form ``G_{k-1} c``, so ``G_k x`` is the same. Use it where only ``G_k x`` enters, as in
        the velocity smoothing. ``guess`` and ``return_info`` work as in :meth:`MassMatrix.solve`."""
        return _shifted_stiffness_solve(self.seq, b, self.k, self.eps, guess=guess, return_info=return_info)


class OperatorFamily:
    """The operators of one kind on a sequence, indexed by the form degree: ``seq.M[k]``."""

    def __init__(self, seq, cls, degrees):
        self._seq, self._cls, self._degrees = seq, cls, degrees

    def __getitem__(self, k):
        if k not in self._degrees:
            raise IndexError(f"{self._cls.__name__} is defined for {self._degrees}, got {k!r}")
        return self._cls(self._seq, *k) if isinstance(k, tuple) else self._cls(self._seq, int(k))
