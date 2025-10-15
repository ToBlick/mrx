from typing import Callable, Optional

import jax
import jax.numpy as jnp

__all__ = ['SplineBasis', 'TensorBasis', 'DerivativeSpline']


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
        if type not in ['clamped', 'periodic', 'constant', 'fourier']:
            raise ValueError(f"Invalid spline type: {type}")

    def __call__(self, x: float, i: int) -> jnp.ndarray:
        """Evaluate the ith spline at point x.

        Args:
            x: The point at which to evaluate the spline
            i: The index of the spline to evaluate

        Returns:
            The value of the ith spline at x
        """
        return self.evaluate(x, i)

    def __getitem__(self, i: int) -> Callable[[float], jnp.ndarray]:
        """Return a function that evaluates the ith spline.

        Args:
            i: The index of the spline

        Returns:
            A function that takes x and returns the value of the ith spline at x
        """
        return lambda x: self.evaluate(x, i)

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
        elif self.type == 'fourier':
            T = jnp.linspace(0, 1, n+1)
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
        elif self.type == 'fourier':
            return 1.0

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

    def __safe_divide(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Safely divide x by y, returning 0 when y is 0.

        Args:
            x: The numerator
            y: The denominator

        Returns:
            The result of the division, or 0 if y is 0
        """
        return jnp.where(jnp.isclose(y, 0.0), jnp.zeros_like(x), x / y)

    def _const_spline(self, x: float, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate a constant (degree 0) spline.

        Args:
            x: The point at which to evaluate
            t: A vector of two elements - the start and end of the interval

        Returns:
            1.0 if t[0] ≤ x < t[1], 0.0 otherwise
        """
        # If knots coincide, value is 0
        return jnp.where(jnp.isclose(t[0], t[1]),
                         jnp.zeros_like(x),
                         jnp.where(jnp.logical_and(t[0] <= x, x < t[1]),
                                   jnp.ones_like(x),
                                   jnp.zeros_like(x))
                         )

    def _p_spline(self, x, t, p):
        # jax.debug.print("Calling p_spline with p={p}, x={x}, t={t}", p=p, x=x, t=t)
        if p == 0:
            return self._const_spline(x, t)
        else:
            return self.__safe_divide(x - t[0], t[p] - t[0]) * self._p_spline(x, t[:-1], p-1) + \
                self.__safe_divide(t[p+1] - x, t[p+1] - t[1]) * \
                self._p_spline(x, t[1:], p-1)


class TensorBasis:
    """A class representing a tensor product of spline bases.

    This class implements a multidimensional basis formed by taking tensor products
    of one-dimensional spline bases. It is particularly useful for constructing
    basis functions in higher dimensions (2D or 3D) from one-dimensional splines.

    Attributes:
        bases (list[SplineBasis]): List of one-dimensional spline bases
        shape (jnp.ndarray): Array containing the number of basis functions in each dimension
        n (int): Total number of basis functions (product of individual dimensions)
        ns (jnp.ndarray): Array of indices for all basis functions
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
        self.shape = jnp.array([b.n for b in bases])
        self.n = bases[0].n * bases[1].n * bases[2].n
        self.ns = jnp.arange(self.n)

    def evaluate(self, x: jnp.ndarray, i: int) -> jnp.ndarray:
        """Evaluate the i-th tensor product basis function at point x.

        Computes the value by taking the product of the appropriate one-dimensional
        basis functions in each coordinate direction.

        Args:
            x: Point at which to evaluate the basis function (array of coordinates)
            i: Index of the tensor product basis function to evaluate

        Returns:
            Value of the i-th tensor product basis function at x
        """
        if x.shape[0] != len(self.bases):
            raise ValueError(
                f"Input point dimension {x.shape[0]} does not match number of bases {len(self.bases)}")

        ijk = jnp.unravel_index(i, self.shape)
        return self.bases[0](x[0], ijk[0]) * self.bases[1](x[1], ijk[1]) * self.bases[2](x[2], ijk[2])

    def __call__(self, x: jnp.ndarray, i: int) -> jnp.ndarray:
        """Evaluate the i-th tensor product basis function at point x.

        Args:
            x: Point at which to evaluate the basis function (array of coordinates)
            i: Index of the tensor product basis function to evaluate

        Returns:
            Value of the i-th tensor product basis function at x
        """
        return self.evaluate(x, i)

    def __getitem__(self, i: int) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Return a function that evaluates the i-th tensor product basis function.

        Args:
            i: Index of the tensor product basis function

        Returns:
            A function that takes a point x and returns the value of the i-th
            tensor product basis function at x
        """
        return lambda x: self.evaluate(x, i)


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
        self.T = s.T[1:-1] if s.type == 'periodic' else s.T
        self.s = SplineBasis(self.n, self.p, self.type, self.T)
        self.ns = jnp.arange(self.n)

    def __call__(self, x: float, i: int) -> jnp.ndarray:
        """Evaluate the derivative of the ith spline at point x.

        Args:
            x: The point at which to evaluate the derivative
            i: The index of the spline derivative to evaluate

        Returns:
            The value of the derivative of the ith spline at x
        """
        return self.evaluate(x, i)

    def __getitem__(self, i: int) -> Callable[[float], jnp.ndarray]:
        """Return a function that evaluates the derivative of the ith spline.

        Args:
            i: The index of the spline derivative

        Returns:
            A function that takes x and returns the derivative value at x
        """
        return lambda x: self.evaluate(x, i)

    def evaluate(self, x: float, i: int) -> jnp.ndarray:
        """Evaluate the derivative of the ith spline at point x.

        Computes the derivative based on the spline type:
        - For clamped splines: Uses a forward difference formula with appropriate scaling
        - For periodic splines: Handles wrapping of indices for periodic continuity
        - For constant splines: Returns 1.0 (derivative of constant function)

        Args:
            x: The point at which to evaluate the derivative
            i: The index of the spline derivative to evaluate

        Returns:
            The value of the derivative at x
        """
        p = self.p
        n = self.n
        if self.type == 'clamped':
            return jax.lax.cond(
                i < n,
                lambda x: self.s(x, i+1) * (p+1) /
                (self.s.T[i+p+2] - self.s.T[i+1]),
                lambda x: 0.0,
                operand=x
            )
        elif self.type == 'periodic':
            j = jnp.mod(i + n, n)
            return self.s(x, j) * (p+1) / (self.s.T[j+p+1] - self.s.T[j])
        else:
            return 1.0
