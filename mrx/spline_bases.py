"""1D B-spline bases (clamped, periodic, constant), their derivative bases, and local evaluation."""
import functools
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


def _nonzero_bsplines(T, p, x):
    """Values at ``x`` of the ``p + 1`` degree-``p`` B-splines on knots ``T`` that
    can be nonzero there, and the raw index of the first of them.

    One de Boor pass over the knot span ``T[s] <= x < T[s + 1]``.  The span is
    clipped to the nonempty ones, so at the right end of a clamped knot vector
    the last polynomial piece is evaluated -- the value at ``x = 1`` is the
    left limit, and so is every derivative autodiff takes through it.
    """
    s = jnp.clip(jnp.searchsorted(T, x, side='right') - 1, p, T.shape[0] - p - 2)
    N = [jnp.ones_like(x)]
    if p == 0:
        return jnp.stack(N), s
    # knots T[s-p+1 .. s+p]: t[m] = T[s - p + 1 + m]
    t = jax.lax.dynamic_slice(T, (s - p + 1,), (2 * p,))
    for j in range(1, p + 1):
        left = [x - t[p - 1 - m] for m in range(j)]    # x - T[s - m]
        right = [t[p + m] - x for m in range(j)]       # T[s + m + 1] - x
        saved = 0.0
        new = []
        for r in range(j):
            temp = N[r] / (right[r] + left[j - 1 - r])
            new.append(saved + right[r] * temp)
            saved = left[j - 1 - r] * temp
        new.append(saved)
        N = new
    return jnp.stack(N), s - p


def contract_local(coefficients, local):
    """Contract a coefficient tensor with local 1-D basis values.

    ``local`` is a triple of ``(values, indices)`` pairs, one per axis, as
    returned by ``evaluate_local``; only the ``prod(p_d + 1)`` coefficients in
    the window are touched.  Leading axes of ``coefficients`` are batched.
    """
    (vr, ir), (vt, it), (vz, iz) = local
    window = coefficients[..., ir[:, None, None], it[None, :, None], iz[None, None, :]]
    return jnp.einsum('i,j,k,...ijk->...', vr, vt, vz, window)


# --------------------------------------------------------------------------- #
# Jitted tables, keyed on the basis SHAPE, not the basis object                #
# --------------------------------------------------------------------------- #
#
# A basis evaluation run eagerly under vmap executes one primitive at a time,
# each compiled on its own (~450 compilations per axis, measured 2026-08-27).
# The jitted forms below take the knot vector as an ARRAY and ``(kind, n, p,
# type)`` as the static key, and rebuild the basis inside the trace: one
# executable per shape, shared by every basis of that shape and every
# sequence, and no basis object held as a jit-cache key (keying on object
# identity accumulated executables across a session and crashed the XLA CPU
# compiler in the full test suite).

def basis_key(basis):
    """``(static_key, knots)`` of a :class:`SplineBasis` or :class:`DerivativeSpline`."""
    if isinstance(basis, DerivativeSpline):
        par = basis.parent
        return ("d", par.n, par.p, par.type), par.T
    return ("s", basis.n, basis.p, basis.type), basis.T


def rebuild_basis(key, T):
    """The basis of :func:`basis_key` from its static key and (possibly traced) knots."""
    kind, n, p, typ = key
    base = SplineBasis(n, p, typ, T=T)
    return DerivativeSpline(base) if kind == "d" else base


@functools.partial(jax.jit, static_argnames=("key",))
def _collocation(T, points, *, key):
    """``basis(points[k], i)`` as a ``(len(points), n)`` matrix."""
    basis = rebuild_basis(key, T)
    return jax.vmap(lambda x: jax.vmap(lambda i: basis(x, i))(basis.ns))(points)


@functools.partial(jax.jit, static_argnames=("key", "periodic"))
def _histopolation(T, spans, xi_ref, w_ref, knots, *, key, periodic):
    """Integrals of every basis function over every span, ``(n_spans, n)``."""
    basis = rebuild_basis(key, T)

    def integrate_span(span):
        a, b = span
        cuts = jnp.clip(knots, a, b)
        cuts = jnp.sort(jnp.concatenate([jnp.array([a]), cuts, jnp.array([b])]))
        lo, hi = cuts[:-1], cuts[1:]
        centers = 0.5 * (lo + hi)
        halfwidths = 0.5 * (hi - lo)
        xs = centers[:, None] + halfwidths[:, None] * xi_ref[None, :]
        if periodic:
            xs = jnp.mod(xs, 1.0)
        values = jax.vmap(jax.vmap(
            lambda x: jax.vmap(lambda i: basis(x, i))(basis.ns)))(xs)
        return jnp.einsum('s,q,sqi->i', halfwidths, w_ref, values)

    return jax.vmap(integrate_span)(spans)


class SplineBasis:
    """A class representing a basis of spline functions.

    This class implements various types of spline bases including clamped, periodic,
    and constant splines of different degrees (0 to 3). The splines are evaluated
    using JAX for efficient computation and automatic differentiation.

    Attributes:
        n (int): The number of splines in the basis
        ns (jnp.ndarray): Array of spline indices
        p (int): The degree of the spline
        type (str): The type of spline ('clamped', 'periodic', or 'constant')
        T (jnp.ndarray): The knot vector defining the spline basis
    """

    n: int
    ns: jnp.ndarray
    p: int
    type: str
    T: jnp.ndarray

    def __init__(self, n: int, p: int, type: str, T: Optional[jnp.ndarray] = None) -> None:
        """Initialize a spline basis.

        Args:
            n: The number of splines in the basis
            p: The degree of the spline
            type: The type of spline ('clamped', 'periodic', or 'constant')
            T: Optional knot vector. If None, knots will be initialized based on type
        """
        self.n = n
        self.ns = jnp.arange(self.n)
        self.p = p
        self.type = type
        if T is not None:
            self.T = T
        else:
            self.T = self._init_knots()

        if p >= n and p != 1:  # n = p = 1 is allowed for ignoring the third dimension
            raise ValueError(
                f"Degree {p} is greater than or equal to the number of splines {n}")
        if type not in ['clamped', 'periodic', 'constant']:
            raise ValueError(f"Invalid spline type: {type}")

    def __call__(self, x: float, i: int) -> jnp.ndarray:
        """Alias for :meth:`evaluate`."""
        return self.evaluate(x, i)

    def _init_knots(self) -> jnp.ndarray:
        """Initialize the knot vector based on the spline type.

        Returns:
            The initialized knot vector

        Raises:
            ValueError: If an invalid spline type is provided
        """
        n = self.n
        p = self.p
        if self.type == 'periodic':
            _T = jnp.linspace(0, 1, n+1)
            T = jnp.concatenate([
                _T[-(p+1):-1] - 1,
                _T,
                _T[1:(p+1)] + 1
            ])
            return T
        elif self.type == 'clamped':
            T = jnp.concatenate([
                jnp.zeros(p),
                jnp.linspace(0, 1, n-p+1),
                jnp.ones(p)
            ])
            return T
        elif self.type == 'constant':
            T = jnp.array([0, 1])
            return T
        else:
            raise ValueError(f"Invalid spline type: {self.type}")

    def evaluate(self, x: float, i: int) -> jnp.ndarray:
        """Evaluate the ith spline at point x, handling special cases.

        Args:
            x: The point at which to evaluate the spline
            i: The index of the spline to evaluate

        Returns:
            The value of the ith spline at x
        """
        if self.type == 'periodic':
            return jnp.where(
                i < self.p,
                self._evaluate(x, i) + self._evaluate(x, self.n + i),
                self._evaluate(x, i),
            )
        elif self.type == 'clamped':
            return jnp.where(
                jnp.logical_and(i == self.n-1, x == self.T[-1]),
                1.0 * jnp.ones_like(x),
                self._evaluate(x, i))
        elif self.type == 'constant':
            return 1.0

    def evaluate_local(self, x: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return ``(values, indices)`` of the ``p + 1`` basis functions that can
        be nonzero at ``x``; all others vanish there.

        Periodic bases fold the raw indices modulo ``n`` (the same folding
        :meth:`evaluate` applies) and read ``x`` modulo the unit period.
        """
        if self.type == 'constant':
            return jnp.ones(1), jnp.zeros(1, dtype=jnp.int32)
        if self.type == 'periodic':
            x = jnp.mod(x, 1.0)
        values, first = _nonzero_bsplines(self.T, self.p, x)
        indices = first + jnp.arange(self.p + 1)
        if self.type == 'periodic':
            indices = indices % self.n
        return values, indices

    def greville_points(self) -> jnp.ndarray:
        """Return the Greville abscissae for this one-dimensional spline basis.

        For degree ``p > 0`` this uses the standard average of the interior
        knots of each basis function support. For degree ``p = 0`` it falls
        back to support midpoints.
        """
        if self.p == 0:
            points = 0.5 * (self.T[:self.n] + self.T[1:self.n + 1])
        else:
            offsets = jnp.arange(1, self.p + 1)
            knot_ids = self.ns[:, None] + offsets[None, :]
            points = jnp.mean(self.T[knot_ids], axis=1)

        if self.type == 'periodic':
            points = jnp.mod(points, 1.0)

        return points

    def collocation_matrix(self, points: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Assemble the point-collocation matrix for this basis.

        Args:
            points: Evaluation points. If omitted, the Greville abscissae are used.

        Returns:
            Array of shape ``(len(points), n)`` where entry ``[k, i]`` is
            the value of the ``i``-th basis function at ``points[k]``.
        """
        if points is None:
            points = self.greville_points()
        key, T = basis_key(self)
        return _collocation(T, jnp.asarray(points), key=key)

    def _evaluate(self, x: float, i: int) -> jnp.ndarray:
        """Evaluate the ith spline at x using the appropriate degree-specific method.

        Args:
            x: The point at which to evaluate the spline
            i: The index of the spline to evaluate

        Returns:
            The value of the ith spline at x
        """
        knot_slice = jax.lax.dynamic_slice(self.T, (i,), (self.p + 2,))
        return jnp.where(
            jnp.logical_and(self.T[i] <= x, x <= self.T[i+self.p+1]),
            self._p_spline(x, knot_slice, self.p),
            0)

    def _safe_divide(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Divide x by the knot-interval length y, returning 0 on a repeated knot.

        Repeated knots are the structural ones of a clamped knot vector and are
        bit-identical, so the test is exact.  Uses a dummy denominator of 1 in
        the empty branch so that ``x / safe_y`` is always finite — avoiding
        ``0 * NaN`` NaN-poisoning in JAX autodiff.

        Args:
            x: The numerator
            y: The denominator, a difference of two knots

        Returns:
            ``x / y`` where ``y != 0``, ``0`` elsewhere.
        """
        empty = y == 0
        safe_y = jnp.where(empty, jnp.ones_like(y), y)
        return jnp.where(empty, jnp.zeros_like(x), x / safe_y)

    def _const_spline(self, x: float, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate a constant (degree 0) spline.

        Args:
            x: The point at which to evaluate
            t: A vector of two elements - the start and end of the interval

        Returns:
            1.0 if t[0] ≤ x < t[1], 0.0 otherwise
        """
        # If knots coincide, value is 0
        return jnp.where(t[0] == t[1],
                         jnp.zeros_like(x),
                         jnp.where(jnp.logical_and(t[0] <= x, x < t[1]),
                                   jnp.ones_like(x),
                                   jnp.zeros_like(x))
                         )

    def _p_spline(self, x, t, p):
        """Evaluate a p-spline at point x.

        Args:
            x: The point at which to evaluate the spline
            t: The knot vector
            p: The degree of the spline

        Returns:
            The value of the p-spline at x
        """
        if p == 0:
            return self._const_spline(x, t)
        else:
            return self._safe_divide(x - t[0], t[p] - t[0]) * self._p_spline(x, t[:-1], p-1) + \
                self._safe_divide(t[p+1] - x, t[p+1] - t[1]) * \
                self._p_spline(x, t[1:], p-1)


class TensorBasis:
    """Tensor product of three 1-D spline bases, ``B_i(x) B_j(y) B_k(z)``.

    Production evaluation is span-local: :meth:`evaluate_local` returns, per
    axis, the ``p + 1`` nonzero 1-D values and their indices at a point, and
    :meth:`contract` sums a coefficient tensor against them
    (:func:`contract_local`); the sequence tabulates the 1-D bases at the
    quadrature points once. :meth:`evaluate` is the dense per-function
    reference the tests check that path against.

    Attributes:
        bases: the three :class:`SplineBasis`.
        shape: ``(n_x, n_y, n_z)``.
        n: ``prod(shape)``; ``ns`` its ``arange``.
    """

    def __init__(self, bases: list[SplineBasis]) -> None:
        """Initialize a tensor product basis.

        The number of basis functions needs to be tracked during JAX tracing/compilation,
        so we store it explicitly rather than computing it from the bases.

        Args:
            bases: List of one-dimensional SplineBasis objects to form the tensor product
        Raises:
            ValueError: If the number of bases is not exactly 3
        """
        if len(bases) != 3:
            raise ValueError(
                f"TensorBasis requires exactly 3 bases, got {len(bases)}")
        self.bases = bases
        self.n = bases[0].n * bases[1].n * bases[2].n
        self.ns = jnp.arange(self.n)

    def evaluate_local(self, x: jnp.ndarray) -> tuple:
        """Per-axis ``(values, indices)`` of the 1-D basis functions nonzero at ``x``."""
        return tuple(b.evaluate_local(xi) for b, xi in zip(self.bases, x))

    def contract(self, coefficients: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """``sum_ijk c[..., i, j, k] B_i(x_0) B_j(x_1) B_k(x_2)`` from the
        ``prod(p_d + 1)`` terms that are nonzero at ``x``."""
        return contract_local(coefficients, self.evaluate_local(x))


class DerivativeSpline:
    """A class representing the derivative of a spline basis.

    This class implements the derivative of a spline basis, supporting various types
    of splines (clamped, periodic, constant). It computes the derivative by adjusting
    the degree and number of basis functions based on the original spline type.

    Attributes:
        n (int): Number of derivative spline basis functions
        p (int): Degree of the derivative spline
        type (str): Type of spline ('clamped', 'periodic', or 'constant')
        T (jnp.ndarray): Knot vector for the derivative spline
        s (SplineBasis): The underlying spline basis used for derivative computation
    """

    def __init__(self, s: SplineBasis) -> None:
        """Initialize a derivative spline basis.

        Args:
            s: The original SplineBasis object to compute derivatives from
        """
        self.n = s.n - 1 if s.type == 'clamped' else s.n
        self.p = s.p if s.type == 'constant' else s.p - 1
        self.type = s.type
        # The derivative of a degree-p spline on knots T is a degree-(p-1)
        # spline on T[1:-1]: the outermost knot on each side drops out.
        self.T = s.T if s.type == 'constant' else s.T[1:-1]
        self.parent = s
        self.s = SplineBasis(self.n, self.p, self.type, self.T)
        self.ns = jnp.arange(self.n)

    def __call__(self, x: float, i: int) -> jnp.ndarray:
        """Alias for :meth:`evaluate`."""
        return self.evaluate(x, i)

    def _scale(self, i):
        """``D_i = (p + 1) / (T[i + p + 1] - T[i]) * B_i``: the unit-integral normalisation."""
        p = self.p
        return (p + 1) / (self.T[i + p + 1] - self.T[i])

    def evaluate(self, x: float, i: int) -> jnp.ndarray:
        """Evaluate the ith derivative basis function at point x.

        For clamped and periodic splines this is the degree-(p-1) B-spline on
        the trimmed knot vector, scaled to unit integral; for constant splines
        it is 1.0 (derivative of a constant function).

        Args:
            x: The point at which to evaluate the derivative
            i: The index of the spline derivative to evaluate

        Returns:
            The value of the derivative at x
        """
        if self.type == 'constant':
            return 1.0
        return self.s(x, i) * self._scale(i)

    def evaluate_local(self, x: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return ``(values, indices)`` of the ``p + 1`` derivative basis
        functions that can be nonzero at ``x``; see :meth:`SplineBasis.evaluate_local`."""
        values, indices = self.s.evaluate_local(x)
        if self.type == 'constant':
            return values, indices
        return values * self._scale(indices), indices

    def greville_spans(self) -> jnp.ndarray:
        """Return the consecutive Greville intervals of the parent spline basis.

        For clamped splines, the parent Greville points include the endpoints,
        so the consecutive intervals form the natural histopolation cells.

        Returns:
            Array of shape ``(n, 2)`` where each row is ``[a, b]`` defining
            an integration interval.
        """
        points = self.parent.greville_points()
        if self.type == 'periodic':
            # greville_points() applies mod(., 1) to the periodic abscissae,
            # which WRAPS the ones that fall outside [0, 1) and so destroys
            # their monotonicity for every p >= 2.  Rolling an unsorted array
            # produced one span of NEGATIVE width and one of width > 1 -- e.g.
            # n=6 gave widths spanning [-0.833, +1.167] at p = 2, 3 and 4 alike
            # -- so the spans did not tile the period and the histopolation
            # matrix was built over nonsense intervals.  Sorting first is safe:
            # the spans and the moments both come from this function, so they
            # stay consistent, and sorted points tile [g_0, g_0 + 1) exactly
            # once.
            points = jnp.sort(points)
            next_points = jnp.roll(points, -1)
            next_points = next_points.at[-1].set(next_points[-1] + 1.0)
            # NOTE: the last span is [g_{n-1}, g_0 + 1], which lies inside
            # [0, 1] only when g_0 = 0, i.e. for ODD p (Greville points on
            # knots) -- and even then only up to rounding: a point that is 0
            # up to eps can wrap to 1 - eps.  For EVEN p it always crosses
            # the seam.  Every consumer must integrate the PERIODIC extension
            # over it: histopolation_matrix wraps its quadrature points,
            # projectors._span_quadrature feeds a pullback that wraps.
            return jnp.stack([points, next_points], axis=1)
        if self.type != 'clamped':
            raise NotImplementedError(
                "Greville histopolation spans are currently implemented only "
                "for clamped and periodic splines."
            )
        return jnp.stack([points[:-1], points[1:]], axis=1)

    def histopolation_matrix(
        self,
        spans: Optional[jnp.ndarray] = None,
        quadrature_order: Optional[int] = None,
    ) -> jnp.ndarray:
        """Assemble the Greville-span histopolation matrix for this basis.

        Args:
            spans: Integration intervals of shape ``(n, 2)``. If omitted,
                the Greville spans from :meth:`greville_spans` are used.
            quadrature_order: Number of Gauss-Legendre quadrature points per
                span. Defaults to ``max(2, p + 2)``.

        Returns:
            Array of shape ``(n, n)`` where entry ``[k, i]`` is the integral
            of the ``i``-th derivative basis function over ``spans[k]``.
        """
        if spans is None:
            spans = self.greville_spans()
        if quadrature_order is None:
            quadrature_order = max(2, self.p + 2)

        xi_ref, w_ref = np.polynomial.legendre.leggauss(quadrature_order)
        # Every span is split at the knots it contains: a Greville span
        # straddles an interior knot whenever p is EVEN, and Gauss is exact
        # only for polynomials; across a knot the spline has a derivative
        # jump. _span_quadrature in projectors.py splits identically so H
        # and the moments use the SAME rule. Periodic spans can cross the
        # seam (even p, and odd p up to rounding), so their points are
        # wrapped: the basis is evaluated on its periodic extension.
        key, T = basis_key(self)
        return _histopolation(T, jnp.asarray(spans), jnp.asarray(xi_ref),
                              jnp.asarray(w_ref), jnp.unique(self.T),
                              key=key, periodic=self.type == 'periodic')


# --------------------------------------------------------------------------- #
# Basis tables at quadrature points (jitted, keyed on the basis shape)
# --------------------------------------------------------------------------- #
#
# Every table below is a vmap over a spline evaluation. Run eagerly, a vmap
# executes the batched trace one primitive at a time, each primitive compiled
# and dispatched on its own (~4000 compilations per (4,8,4) sequence build,
# measured 2026-08-27: 0.6 s per table). Under jit the whole table is one
# executable, keyed on the basis SHAPE (:func:`basis_key`),
# so every basis of a shape shares it.

def basis_table(basis, x):
    """``basis(x_q, i)`` for every ``i`` in ``basis.ns`` and every point in ``x``, ``(n, n_q)``."""
    key, T = basis_key(basis)
    return _basis_table(T, x, key=key)


@functools.partial(jax.jit, static_argnames=("key",))
def _basis_table(T, x, *, key):
    basis = rebuild_basis(key, T)
    return jax.vmap(jax.vmap(basis, (0, None)), (None, 0))(x, basis.ns)


def basis_derivative_table(basis, x):
    """``d/dx basis(x_q, i)`` by autodiff, ``(n, n_q)``."""
    key, T = basis_key(basis)
    return _basis_derivative_table(T, x, key=key)


@functools.partial(jax.jit, static_argnames=("key",))
def _basis_derivative_table(T, x, *, key):
    basis = rebuild_basis(key, T)

    def value(x, i):
        return jnp.sum(basis(x, i))
    return jax.vmap(jax.vmap(jax.grad(value, argnums=0), (0, None)), (None, 0))(x, basis.ns)


@functools.partial(jax.jit, static_argnames=("key",))
def _evaluate_basis_local(T, x_local, gdof, *, key):
    basis = rebuild_basis(key, T)

    def eval_e(x_e, dof_e):
        return jax.vmap(
            lambda x: jax.vmap(lambda i: basis(x, i))(dof_e)
        )(x_e)
    return jax.vmap(eval_e, in_axes=(0, 0))(x_local, gdof)


def evaluate_basis_local(basis, x_q_flat, q_per_elem):
    """Evaluate a 1D spline basis on each element at its local quad points.

    Works for both primal (``SplineBasis``) and derivative
    (``DerivativeSpline``) bases: the derivative basis simply reports a smaller
    degree, hence ``p`` locals per element instead of ``p+1``.

    Parameters
    ----------
    basis : SplineBasis or DerivativeSpline
        1D basis with ``.p``, ``.n`` and ``.type`` attributes and a callable
        ``basis(x, i)`` interface.
    x_q_flat : (n_elem * q,) array
        Composite Gauss quadrature points, ordered element-by-element.
    q_per_elem : int
        Number of Gauss points per knot interval.

    Returns
    -------
    B_loc : (n_elem, q_per_elem, p+1) array
        Values of the locally-active bases at the local quad points.
    gdof : (n_elem, p+1) int array
        Global DOF index of each local basis on each element.
    """
    p = basis.p
    n = basis.n
    n_local = p + 1
    if basis.type == "periodic":
        n_elem = n
        elems = jnp.arange(n_elem)
        ks = jnp.arange(n_local)
        gdof = (elems[:, None] + ks[None, :]) % n
    elif basis.type == "clamped":
        n_elem = n - p
        elems = jnp.arange(n_elem)
        ks = jnp.arange(n_local)
        gdof = elems[:, None] + ks[None, :]
    elif basis.type == "constant":
        # Single element, single DOF (p=0, n=1).
        n_elem = 1
        gdof = jnp.zeros((1, 1), dtype=jnp.int32)
    else:
        raise NotImplementedError(basis.type)

    x_local = x_q_flat.reshape(n_elem, q_per_elem)
    key, T = basis_key(basis)
    return _evaluate_basis_local(T, x_local, gdof, key=key), gdof
