"""
Differential forms implementation for finite element analysis.

This module provides classes for working with differential forms in finite element
analysis, including discrete differential forms, pushforward and pullback operations,
and discrete function representations.

The implementation supports forms of different degrees (k = 0, 1, 2, 3) in
three-dimensional space and includes functionality for evaluation, transformation,
and basis manipulation.
"""

import jax.numpy as jnp
import jax

from mrx.SplineBases import SplineBasis, DerivativeSpline, TensorBasis
from mrx.Utils import inv33


class DifferentialForm:
    """
    A class representing differential forms of various degrees.

    This class implements differential forms using spline bases and supports
    operations like evaluation, indexing, and basis transformations.

    Attributes:
        d (int): Dimension of the space
        k (int): Degree of the differential form (0, 1, 2, or 3)
        n (int): Total number of basis functions
        nr (int): Number of basis functions in r direction
        nχ (int): Number of basis functions in χ direction
        nζ (int): Number of basis functions in ζ direction
        ns (jnp.ndarray): Array of indices for basis functions
        Λ (list): List of SplineBasis objects for each direction
        dΛ (list): List of derivative spline bases
        types (list): Boundary condition types for each direction
        bases (tuple): Tensor bases for the form
        shape (tuple): Shape of the form in each direction
    """

    d: int
    k: int
    n: int
    nr: int
    nχ: int
    nζ: int
    ns: jnp.ndarray

    def __init__(self, k, ns, ps, types, Ts=None):
        """
        Initialize a differential form.

        Args:
            k (int): Degree of the form
            ns (list): Number of basis functions in each direction
            ps (list): Polynomial degrees for each direction
            types (list): Boundary condition types for each direction
            Ts (list, optional): Knot vectors for each direction
        """
        self.d = len(ns)
        self.k = k
        if Ts is None:
            Ts = [None] * self.d
        self.Λ = [SplineBasis(n, p, type, T) for n, p, type, T in zip(ns, ps, types, Ts)]
        self.dΛ = [DerivativeSpline(b) for b in self.Λ]
        self.types = types

        self.nr, self.nχ, self.nζ = ns
        if types[0] == 'clamped':
            self.dr = self.nr - 1
        else:
            self.dr = self.nr
        if types[1] == 'clamped':
            self.dχ = self.nχ - 1
        else:
            self.dχ = self.nχ
        if types[2] == 'clamped':
            self.dζ = self.nζ - 1
        else:
            self.dζ = self.nζ

        self.vecs = jnp.eye(self.d)

        if k == 0:
            self.bases = (TensorBasis(self.Λ), )
            self.shape = ((self.nr, self.nχ, self.nζ), )
            self.n1 = self.nr * self.nχ * self.nζ
            self.n2 = 0
            self.n3 = 0
        elif k == 1:
            self.bases = (TensorBasis([self.dΛ[0], self.Λ[1], self.Λ[2]]),
                          TensorBasis([self.Λ[0], self.dΛ[1], self.Λ[2]]),
                          TensorBasis([self.Λ[0], self.Λ[1], self.dΛ[2]]))
            self.shape = ((self.dr, self.nχ, self.nζ),
                          (self.nr, self.dχ, self.nζ),
                          (self.nr, self.nχ, self.dζ))
            self.n1 = self.dr * self.nχ * self.nζ
            self.n2 = self.nr * self.dχ * self.nζ
            self.n3 = self.nr * self.nχ * self.dζ
        elif k == 2:
            self.bases = (TensorBasis([self.Λ[0], self.dΛ[1], self.dΛ[2]]),
                          TensorBasis([self.dΛ[0], self.Λ[1], self.dΛ[2]]),
                          TensorBasis([self.dΛ[0], self.dΛ[1], self.Λ[2]]))
            self.shape = ((self.nr, self.dχ, self.dζ),
                          (self.dr, self.nχ, self.dζ),
                          (self.dr, self.dχ, self.nζ))
            self.n1 = self.nr * self.dχ * self.dζ
            self.n2 = self.dr * self.nχ * self.dζ
            self.n3 = self.dr * self.dχ * self.nζ
        elif k == 3:
            self.bases = (TensorBasis(self.dΛ), )
            self.shape = ((self.dr, self.dχ, self.dζ), )
            self.n1 = self.dr * self.dχ * self.dζ
            self.n2 = 0
            self.n3 = 0
        self.n = self.n1 + self.n2 + self.n3
        self.ns = jnp.arange(self.n)

    def _vector_index(self, idx):
        """
        Convert linear index to vector component and local index.

        Args:
            idx (int): Linear index into the form

        Returns:
            tuple: (category, index) where category indicates the vector
                  component and index is the local index within that component
        """
        if self.k == 0 or self.k == 3:
            return 0, idx
        elif self.k == 1 or self.k == 2:
            n1, n2 = self.n1, self.n2
            category = jnp.int32(idx >= n1) + jnp.int32(idx >= n1 + n2)
            index = idx - n1 * jnp.int32(idx >= n1) - n2 * jnp.int32(idx >= n1 + n2)
            return category, index

    def _ravel_index(self, c, i, j, k):
        """
        Convert multi-dimensional indices to linear index.

        Args:
            c (int): Component index
            i (int): Index in r direction
            j (int): Index in χ direction
            k (int): Index in ζ direction

        Returns:
            int: Linear index into the form
        """
        if self.k == 0:
            return jnp.ravel_multi_index((i, j, k), (self.nr, self.nχ, self.nζ), mode='clip')
        elif self.k == 1:
            n1, n2 = self.n1, self.n2
            return jnp.where(
                c == 0,
                jnp.ravel_multi_index((i, j, k), (self.dr, self.nχ, self.nζ), mode='clip'),
                jnp.where(
                    c == 1,
                    n1 + jnp.ravel_multi_index((i, j, k), (self.nr, self.dχ, self.nζ), mode='clip'),
                    n1 + n2 + jnp.ravel_multi_index((i, j, k), (self.nr, self.nχ, self.dζ), mode='clip')
                )
            )
        elif self.k == 2:
            n1, n2 = self.n1, self.n2
            return jnp.where(
                c == 0,
                jnp.ravel_multi_index((i, j, k), (self.nr, self.dχ, self.dζ), mode='clip'),
                jnp.where(
                    c == 1,
                    n1 + jnp.ravel_multi_index((i, j, k), (self.dr, self.nχ, self.dζ), mode='clip'),
                    n1 + n2 + jnp.ravel_multi_index((i, j, k), (self.dr, self.dχ, self.nζ), mode='clip')
                )
            )
        elif self.k == 3:
            return jnp.ravel_multi_index((i, j, k), (self.dr, self.dχ, self.dζ), mode='clip')

    def _unravel_index(self, idx):
        """
        Convert linear index to multi-dimensional indices.

        Args:
            idx (int): Linear index into the form

        Returns:
            tuple: (category, i, j, k) where category is the component index
                  and (i,j,k) are the indices in each direction
        """
        if self.k == 0:
            return 0, *jnp.unravel_index(idx, (self.nr, self.nχ, self.nζ))
        elif self.k == 1:
            c, ijk = self._vector_index(idx)
            i, j, k = jnp.where(
                c == 0,
                jnp.array(jnp.unravel_index(ijk, (self.dr, self.nχ, self.nζ))),
                jnp.where(
                    c == 1,
                    jnp.array(jnp.unravel_index(ijk, (self.nr, self.dχ, self.nζ))),
                    jnp.array(jnp.unravel_index(ijk, (self.nr, self.nχ, self.dζ)))
                )
            )
            return c, i, j, k
        elif self.k == 2:
            c, ijk = self._vector_index(idx)
            i, j, k = jnp.where(
                c == 0,
                jnp.array(jnp.unravel_index(ijk, (self.nr, self.dχ, self.dζ))),
                jnp.where(
                    c == 1,
                    jnp.array(jnp.unravel_index(ijk, (self.dr, self.nχ, self.dζ))),
                    jnp.array(jnp.unravel_index(ijk, (self.dr, self.dχ, self.nζ)))
                )
            )
            return c, i, j, k
        elif self.k == 3:
            return 0, *jnp.unravel_index(idx, (self.dr, self.dχ, self.dζ))

    def __call__(self, x, i):
        """Evaluate the form at point x with basis function i."""
        return self.evaluate(x, i)

    def __getitem__(self, i):
        """Get the i-th basis function of the form."""
        return lambda x: self.evaluate(x, i)

    def __iter__(self):
        """Iterate over all basis functions of the form."""
        for i in range(self.n):
            yield self[i]

    def __len__(self):
        """Get the total number of basis functions."""
        return self.n

    def evaluate(self, x, i):
        """
        Evaluate the form at point x with basis function i.

        Args:
            x (array-like): Point at which to evaluate
            i (int): Index of basis function to evaluate

        Returns:
            array-like: Value of the form at x
        """
        category, index = self._vector_index(i)
        if self.k == 0 or self.k == 3:
            return jnp.ones(1) * self.bases[0](x, index)
        elif self.k == 1 or self.k == 2:
            e = jnp.zeros(3).at[category].set(1)
            val = jnp.where(
                category == 0,
                self.bases[0](x, index),
                jnp.where(
                    category == 1,
                    self.bases[1](x, index),
                    self.bases[2](x, index)
                )
            )
            return e * val


class DiscreteFunction:
    """
    A class representing discrete functions using differential forms.

    This class implements discrete functions as linear combinations of basis
    functions from a differential form.

    Attributes:
        dof (array-like): Degrees of freedom (coefficients)
        Λ (DifferentialForm): The underlying differential form
        n (int): Number of basis functions
        ns (array-like): Array of indices
        E (array-like): Transformation matrix
    """

    def __init__(self, dof, Λ, E=None):
        """
        Initialize a discrete function.

        Args:
            dof (array-like): Degrees of freedom (coefficients)
            Λ (DifferentialForm): The underlying differential form
            E (array-like, optional): Transformation matrix
        """
        self.dof = dof
        self.Λ = Λ
        self.n = Λ.n
        self.ns = jnp.arange(self.n)
        self.E = E if E is not None else jnp.eye(self.n)

    def __call__(self, x):
        """
        Evaluate the function at point x.

        Args:
            x (array-like): Point at which to evaluate

        Returns:
            array-like: Value of the function at x
        """
        return self.dof @ self.E @ jax.vmap(self.Λ, (None, 0))(x, self.ns)


class Pushforward:
    """
    A class implementing pushforward operations on differential forms.

    This class implements the pushforward of differential forms under a
    given transformation.

    Attributes:
        k (int): Degree of the form
        f (callable): The form to push forward
        F (callable): The transformation function
    """

    def __init__(self, f, F, k):
        """
        Initialize a pushforward operation.

        Args:
            f (callable): The form to push forward
            F (callable): The transformation function
            k (int): Degree of the form
        """
        self.k = k
        self.f = f
        self.F = F

    def __call__(self, x):
        """
        Apply the pushforward at point x.

        Args:
            x (array-like): Point at which to evaluate

        Returns:
            array-like: Value of the pushed-forward form at x
        """
        y = self.F(x)
        if self.k == 0:
            return self.f(y)
        elif self.k == 1:
            return jax.jacfwd(self.F)(x).T @ self.f(y)
        elif self.k == 2:
            return inv33(jax.jacfwd(self.F)(x)) @ self.f(y) * jnp.linalg.det(jax.jacfwd(self.F)(x))
        elif self.k == 3:
            return self.f(y) * jnp.linalg.det(jax.jacfwd(self.F)(x))


class Pullback:
    """
    A class implementing pullback operations on differential forms.

    This class implements the pullback of differential forms under a
    given transformation.

    Attributes:
        k (int): Degree of the form
        f (callable): The form to pull back
        F (callable): The transformation function
    """

    def __init__(self, f, F, k):
        """
        Initialize a pullback operation.

        Args:
            f (callable): The form to pull back
            F (callable): The transformation function
            k (int): Degree of the form
        """
        self.k = k
        self.f = f
        self.F = F

    def __call__(self, x):
        """
        Apply the pullback at point x.

        Args:
            x (array-like): Point at which to evaluate

        Returns:
            array-like: Value of the pulled-back form at x
        """
        y = self.F(x)
        if self.k == 0:
            return self.f(y)
        elif self.k == 1:
            return jax.jacfwd(self.F)(x).T @ self.f(y)
        elif self.k == 2:
            return inv33(jax.jacfwd(self.F)(x)) @ self.f(y) * jnp.linalg.det(jax.jacfwd(self.F)(x))
        elif self.k == 3:
            return self.f(y) * jnp.linalg.det(jax.jacfwd(self.F)(x))
