"""Geometric multigrid for the stiffness ``S_2 = G_2^T M_3 G_2`` of the Dirichlet 2-forms.

The metric enters ``S_2`` only through the scalar weight ``1 / J`` of the 3-form mass ``M_3``, and ``G_2`` is
topological. In logical coordinates ``S_2`` is close to an isotropic div-div operator, without the skew of the radial
and poloidal directions that limits the atoms on shaped cross-sections. A plain V-cycle with a point smoother handles
it in a number of iterations that does not grow with the mesh.

:class:`StiffnessMultigrid` is that V-cycle, used as the preconditioner of the conjugate gradients on ``S_2``. The
levels are coarser sequences of the same degree and symmetry. Each coarsening halves every axis whose number of cells
is even and stays above a floor, up to :data:`LEVELS` levels. A graded sequence gets the graded theta counts of each
coarse radial mesh. Every level carries the fine sequence's installed map, so all levels discretise the same domain.
The prolongation is the knot insertion of the coarse splines into the fine ones along each axis, taken through the
extractions by least squares, ``P = (E_f E_f^T)^-1 E_f P_raw E_c^T``. Where the coarse space lies in the fine one,
as on graded meshes with dyadic theta counts, this is the exact embedding and commutes with the exterior derivative.
At the axis of a uniform mesh the spaces do not nest and it is a least-squares fit. The restriction is ``P^T``. Each
level smooths with one damped Jacobi step before and one after the coarse correction, with the weight
``4 / (3 lam_max)`` of the largest eigenvalue of ``D^-1 S_2``. The coarsest level is solved by a dense pseudo-inverse.

The cycle approximates ``S_2^+`` on the co-exact forms and does nothing designed for the exact ones. It serves the
co-closed solves ``S_2 x = G_2^T w``: ``seq.S[2].solve(b)``, both stiffness solves of the exact
part of ``L_3`` and the middle solve of the Hodge split of ``L_2``.

It is selected with ``laplacian_atom="multigrid"`` in
:meth:`mrx.derham_sequence.DeRhamSequence.build_preconditioners`, which builds it for the Dirichlet 2-forms next to
the atoms. On a half-period sequence each parity view runs the cycle on the same parity view of every level.
Storage is the transfers, the diagonals and the coarsest block, a few numbers per
degree of freedom. The build constructs the coarse sequences and assembles the transfers on the host.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import equinox as eqx
import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DERIV_AXES
from mrx.extraction_operators import SparseMatrix, graded_theta_counts, gram_inverse, prolongation, ring_layout
from mrx.incidence import diff_1d
from mrx.mass import sumfact_apply
from mrx.metric_lumping import _parity_split, _probe_rows
from mrx.operators import _stiffness
from mrx.precision import PRECOND_DTYPE, to_precond
from mrx.pytree import register_arrays
from mrx.spline_bases import SplineBasis, basis_table, knot_vector
from mrx.symmetry import mirror_component

#: The most levels of the hierarchy, the given sequence included.
LEVELS = 4

#: The fewest cells a coarse level keeps in ``(r, theta)``. In ``zeta`` the floor is ``2 p + 2``.
MIN_CELLS = (4, 8)

#: Power steps for the largest eigenvalue of ``D^-1 S_2`` on each level.
POWER_STEPS = 40

#: Eigenvalues of the coarsest ``S_2`` below this fraction of the largest are its kernel, the closed forms.
PINV_CUT = 1e-10


# --------------------------------------------------------------------------- #
# The coarse levels and the transfers (host, once per geometry)                #
# --------------------------------------------------------------------------- #

def _breakpoints(basis):
    """The distinct knots of a 1-D basis in ``[0, 1]``."""
    T = np.asarray(basis.T)
    return np.unique(T[(T >= 0.0) & (T <= 1.0)])


def coarse_sequence(seq):
    """The next coarser level of ``seq`` with its map installed, or ``None`` when no axis can be halved."""
    p = seq.p
    floors = (*MIN_CELLS, 2 * p + 2)
    knots, ns, halved = [], [], False
    for a in range(3):
        bp = _breakpoints(seq.basis_0.Lambda[a])
        cells = bp.size - 1
        # a half-period sequence needs an even number of zeta cells
        even = a == 2 and seq.half_period
        if cells % 2 == 0 and cells // 2 >= floors[a] and not (even and cells // 2 % 2):
            bp, cells, halved = bp[::2], cells // 2, True
        uniform = np.allclose(np.diff(bp), 1.0 / cells, rtol=0.0, atol=1e-12)
        knots.append(None if uniform else knot_vector(bp, p, a > 0))
        ns.append(cells + (p if a == 0 else 0))
    if not halved:
        return None
    counts = None
    if seq.theta_counts is not None:
        T_r = knots[0] if knots[0] is not None else SplineBasis(ns[0], p, "clamped").T
        counts = graded_theta_counts(np.asarray(T_r), p, ns[1])
    coarse = DeRhamSequence(ns, p, nfp=seq.nfp, symmetry=seq.symmetry, knots=knots, theta_counts=counts,
                            tol=seq.tol, maxiter=seq.maxiter, quad_points=seq.quad_points)
    coarse.set_geometry_from_map(seq.map)
    return coarse


def _knot_insertion(fine, coarse, breakpoints):
    """``(n_fine, n_coarse)``: the coarse 1-D basis functions as combinations of the fine ones (cells between
    ``breakpoints``), fitted by least squares at ``p + 2`` points per fine cell. Exact when the coarse knots are fine
    knots."""
    q = fine.p + 2
    x = (breakpoints[:-1, None] + np.diff(breakpoints)[:, None] * (np.arange(q) + 0.5) / q).ravel()
    F, C = (np.asarray(basis_table(b, jnp.asarray(x))).T for b in (fine, coarse))
    P = np.linalg.lstsq(F, C, rcond=None)[0]
    return np.where(np.abs(P) < 1e-13 * np.abs(P).max(), 0.0, P)


def _csr(m):
    """A :class:`SparseMatrix` as a scipy CSR matrix."""
    r, c, v = m.entries()
    return sp.csr_matrix((v, (r, c)), shape=m.forward_shape)


def _prolongation(fine, coarse):
    """The prolongation ``(E_f E_f^T)^-1 E_f P_raw E_c^T`` of the Dirichlet 2-forms of two full sequences, sparse."""
    one_d = [{d: _knot_insertion((fine.basis_0.dLambda if d else fine.basis_0.Lambda)[a],
                                 (coarse.basis_0.dLambda if d else coarse.basis_0.Lambda)[a],
                                 _breakpoints(fine.basis_0.Lambda[a]))
              for d in (False, True)} for a in range(3)]
    raw = sp.block_diag([sp.kron(sp.kron(one_d[0][0 in dax], one_d[1][1 in dax]), one_d[2][2 in dax])
                         for dax in DERIV_AXES[2]]).tocsr()
    key = (2, True)
    P = gram_inverse(fine.extraction[key], fine.core[key], fine._gram[key]) @ (
        _csr(fine.extraction[key]) @ (raw @ _csr(coarse.extraction[key]).T))
    return _clean(P)


def _clean(P):
    """``P`` without the round-off of the least squares."""
    P = sp.csr_matrix(P)
    P.data[np.abs(P.data) < 1e-12 * np.abs(P.data).max()] = 0.0
    P.eliminate_zeros()
    return P


def _device(P):
    """A scipy matrix as a :class:`SparseMatrix`."""
    P = P.tocoo()
    return SparseMatrix.from_coo(P.row, P.col, P.data, P.shape)


def _stiffness_diagonal(seq):
    """The diagonal of ``S_2`` on the Dirichlet 2-forms of a full sequence, on the host.

    Away from the axis a basis function is a tensor product, a graded one with a coarse theta function, and its
    divergence is the derivative of its primal factor. Its diagonal entry is the integral of that derivative squared
    against ``1 / J``, by sum factorisation. The polar rows are probed."""
    form = seq.basis_2
    deriv = seq.d_basis_jk
    weight = (seq.quad.w / seq.jacobian_j).reshape(seq.quad.shape)
    d = np.zeros(seq.n(2))
    tables = {}
    for c, a, m, derivative, first, nz in ring_layout(form, True, seq.theta_counts):
        if (c, m) not in tables:
            # the primal factor's derivative B'_l = D_{l-1} - D_l from the derivative splines
            tabs = [jnp.asarray(diff_1d(int(seq.basis_0.shape[0][b]), seq.basis_0.types[b]).T.toarray()) @ deriv[b]
                    if b == c else deriv[b] for b in range(3)]
            if m != form.shape[c][1]:
                tabs[1] = jnp.asarray(prolongation(m, form.shape[c][1], seq.p, derivative)).T @ tabs[1]
            t = jnp.einsum("ax,xyz->ayz", tabs[0] ** 2, weight)
            t = jnp.einsum("by,ayz->abz", tabs[1] ** 2, t)
            full = jnp.einsum("cz,abz->abc", tabs[2] ** 2, t)
            tables[(c, m)] = np.asarray(mirror_component(seq, full, DERIV_AXES[2][c]))
        d[first + np.arange(m)[:, None] * nz + np.arange(nz)[None, :]] = tables[(c, m)][a]
    core = np.asarray(seq.core[(2, True)])
    block = _probe_rows(lambda x: _stiffness(seq, x, 2), seq.n(2), core, _parity_split(seq, 2))
    d[core] = np.diag(np.asarray(block))
    return d


# --------------------------------------------------------------------------- #
# The cycle                                                                    #
# --------------------------------------------------------------------------- #

class _Stiffness(eqx.Module):
    """``S_2 = E G_2^T M_3 G_2 E^T`` on one level: the arrays of :func:`mrx.operators._stiffness` for a parity view or
    a full-period sequence, where the 3-form mass needs no reflection. Holding these instead of the level's sequence
    keeps the preconditioner, and every compiled call that takes the fine sequence, small."""
    E: SparseMatrix
    div: object
    div_T: object
    plan: object
    weight: jnp.ndarray

    def __init__(self, space):
        self.E, self.div, self.div_T = space.E(2), space.incidence[2], space.incidence[2].T
        self.plan, self.weight = space.mass_plan[3], space.geometry.mass_weights[3]

    def __call__(self, x):
        return self.E @ (self.div_T @ sumfact_apply(self.plan, self.weight, self.div @ (self.E.T @ x)))


class _Level(eqx.Module):
    """One level above the coarsest: its ``S_2``, the damped inverse diagonal and the prolongation from below."""
    stiffness: _Stiffness
    smoother: jnp.ndarray
    prolong: SparseMatrix


class _Cycle(eqx.Module):
    """The symmetric V-cycle on the 2-form spaces of one hierarchy, as an approximate inverse of ``S_2``."""
    levels: tuple
    coarsest: jnp.ndarray

    def __call__(self, r):
        # in PRECOND_DTYPE, returned in the precision of r
        return self._visit(0, r.astype(PRECOND_DTYPE)).astype(r.dtype)

    def _visit(self, i, r):
        if i == len(self.levels):
            return self.coarsest @ r
        level = self.levels[i]
        x = level.smoother * r
        x = x + level.prolong @ self._visit(i + 1, level.prolong.T @ (r - level.stiffness(x)))
        return x + level.smoother * (r - level.stiffness(x))


@eqx.filter_jit
def _lam_max(stiffness, dinv, v):
    """The largest eigenvalue of ``D^-1 S_2`` by power steps from ``v``."""
    def step(_, carry):
        v, _lam = carry
        w = dinv * stiffness(v)
        lam = jnp.linalg.norm(w)
        return w / lam, lam
    v = dinv * stiffness(v)
    return jax.lax.fori_loop(0, POWER_STEPS, step, (v / jnp.linalg.norm(v), 0.0))[1]


def _reduction(space):
    """The expansion ``X`` of a parity view's 2-forms as a scipy matrix, ``None`` on a full sequence."""
    return None if space.parity is None else _csr(space.reduction[(2, True)])


def _cycle(spaces, diagonals, transfers):
    """The cycle on ``spaces`` (fine first), from the diagonals and transfers of the full sequences."""
    X = [_reduction(s) for s in spaces]
    levels = []
    for i, space in enumerate(spaces[:-1]):
        d = diagonals[i] if X[i] is None else X[i].multiply(X[i]).T @ diagonals[i]
        P = transfers[i] if X[i] is None else _clean(X[i].T @ transfers[i] @ X[i + 1])
        stiffness = _Stiffness(space)
        dinv = jnp.asarray(1.0 / d)
        v = jnp.asarray(np.random.default_rng(0).standard_normal(d.size))
        levels.append(_Level(stiffness, (4.0 / 3.0) / _lam_max(stiffness, dinv, v) * dinv, _device(P)))
    n = spaces[-1].n(2)
    w, V = jnp.linalg.eigh(_probe_rows(_Stiffness(spaces[-1]), n, np.arange(n), None))
    keep = w > PINV_CUT * w[-1]
    return to_precond(_Cycle(tuple(levels), (V * jnp.where(keep, 1.0 / jnp.where(keep, w, 1.0), 0.0)) @ V.T))


@register_arrays
class StiffnessMultigrid:
    """Approximate inverse of ``S_2`` on the Dirichlet 2-forms of ``seq`` by a geometric V-cycle (see the module
    docstring).

    Built by ``DeRhamSequence.build_preconditioners(laplacian_atom="multigrid")`` for the installed geometry. Apply
    it with :meth:`apply`.
    """

    def __init__(self, seq, operators, k):
        if k != 2 or not seq.dirichlet:
            raise ValueError(f"the multigrid preconditioner is for the Dirichlet 2-forms, not k = {k}, "
                             f"dirichlet = {seq.dirichlet}")
        self.parity_projector = seq.free_projector(k)
        levels = [seq]
        while len(levels) < LEVELS:
            coarse = coarse_sequence(levels[-1])
            if coarse is None:
                break
            levels.append(coarse)
        diagonals = [_stiffness_diagonal(s) for s in levels[:-1]]
        transfers = [_prolongation(f, c) for f, c in zip(levels[:-1], levels[1:])]
        if not seq.half_period:
            self._cycles, self._X = {0: _cycle(levels, diagonals, transfers)}, None
            return
        # one cycle per parity, on the parity views of every level
        self._cycles = {s: _cycle([level.parity_view(s) for level in levels], diagonals, transfers) for s in (1, -1)}
        self._X = {s: seq.parity_view(s).reduction[(2, True)] for s in (1, -1)}

    def apply(self, x):
        """Apply the preconditioner to a coefficient vector of the degrees of freedom. On a half-period sequence it
        is the sum of the two parity cycles, ``X_e B_e X_e^T + X_o B_o X_o^T``."""
        x = jnp.asarray(x)
        if self._X is None:
            return self._cycles[0](x)
        return sum(X @ self._cycles[s](X.T @ x) for s, X in self._X.items())

    def reduced(self, X):
        """The payload of a parity view with the expansion ``X``: the cycle of its parity, which ``X`` has."""
        return self._cycles[1 if float(self.parity_projector.parity(X @ jnp.ones(X.shape[1]))) > 0 else -1]

    def apply_reduced(self, cycle, x):
        """The cycle of a parity view, the payload of :meth:`reduced`."""
        return cycle(jnp.asarray(x))

    def stored(self):
        """The number of stored floating-point numbers: transfers, diagonals and coarsest blocks."""
        return int(sum(level.prolong.vals.size + level.smoother.size
                       for c in self._cycles.values() for level in c.levels)
                   + sum(c.coarsest.size for c in self._cycles.values()))
