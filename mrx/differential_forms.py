"""
Differential forms implementation for finite element analysis.

This module provides classes for working with differential forms in finite element
analysis, including discrete differential forms, pushforward and pullback operations,
and discrete function representations.

The implementation supports forms of different degrees (k = 0, 1, 2, 3) in
three-dimensional space and includes functionality for evaluation, transformation,
and basis manipulation.
"""

import jax
import jax.numpy as jnp

from mrx.spline_bases import DerivativeSpline, SplineBasis, TensorBasis
from mrx.utils import inv33

__all__ = ['DifferentialForm', 'DiscreteFunction', 'Pushforward', 'Pullback']


class DifferentialForm:
    """
    A class representing differential forms of various degrees.

    This class implements differential forms using spline bases and supports
    operations like evaluation, indexing, and basis transformations.

    Attributes:
        d (int): Dimension of the space
        k (int): Degree of the differential form (0, 1, 2, or 3. -1 refers to a vector field)
        n (int): Total number of basis functions
        nr (int): Number of basis functions in r direction
        nt (int): Number of basis functions in θ direction
        nz (int): Number of basis functions in ζ direction
        pr (int): Polynomial degree in r direction
        pt (int): Polynomial degree in θ direction
        pz (int): Polynomial degree in ζ direction
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
    nt: int
    nz: int
    pr: int
    pt: int
    pz: int
    ns: jnp.ndarray

    def __init__(self, k, ns, ps, types, Ts=None):
        """
        Initialize a differential form.

        Args:
            k (int): Degree of the form, k = 0, 1, 2, 3 are supported.
            ns (list): Number of basis functions in each direction
            ps (list): Polynomial degrees for each direction
            types (list): Boundary condition types for each direction
            Ts (list, optional): Knot vectors for each direction
        """
        self.d = len(ns)
        self.k = k
        if Ts is None:
            Ts = [None] * self.d
        self.Λ = [
            SplineBasis(n, p, type, T) for n, p, type, T in zip(ns, ps, types, Ts)
        ]
        self.dΛ = [DerivativeSpline(b) for b in self.Λ]
        self.types = types

        self.pr, self.pt, self.pz = ps
        self.nr, self.nt, self.nz = ns
        if types[0] == "clamped":
            self.dr = self.nr - 1
        else:
            self.dr = self.nr
        if types[1] == "clamped":
            self.dt = self.nt - 1
        else:
            self.dt = self.nt
        if types[2] == "clamped":
            self.dz = self.nz - 1
        else:
            self.dz = self.nz

        self.vecs = jnp.eye(self.d)

        if k == 0:
            self.bases = (TensorBasis(self.Λ),)
            self.shape = ((self.nr, self.nt, self.nz),)
            self.n1 = self.nr * self.nt * self.nz
            self.n2 = 0
            self.n3 = 0
        elif k == 1:
            self.bases = (
                TensorBasis([self.dΛ[0], self.Λ[1], self.Λ[2]]),
                TensorBasis([self.Λ[0], self.dΛ[1], self.Λ[2]]),
                TensorBasis([self.Λ[0], self.Λ[1], self.dΛ[2]]),
            )
            self.shape = (
                (self.dr, self.nt, self.nz),
                (self.nr, self.dt, self.nz),
                (self.nr, self.nt, self.dz),
            )
            self.n1 = self.dr * self.nt * self.nz
            self.n2 = self.nr * self.dt * self.nz
            self.n3 = self.nr * self.nt * self.dz
        elif k == 2:
            self.bases = (
                TensorBasis([self.Λ[0], self.dΛ[1], self.dΛ[2]]),
                TensorBasis([self.dΛ[0], self.Λ[1], self.dΛ[2]]),
                TensorBasis([self.dΛ[0], self.dΛ[1], self.Λ[2]]),
            )
            self.shape = (
                (self.nr, self.dt, self.dz),
                (self.dr, self.nt, self.dz),
                (self.dr, self.dt, self.nz),
            )
            self.n1 = self.nr * self.dt * self.dz
            self.n2 = self.dr * self.nt * self.dz
            self.n3 = self.dr * self.dt * self.nz
        elif k == 3:
            self.bases = (TensorBasis(self.dΛ),)
            self.shape = ((self.dr, self.dt, self.dz),)
            self.n1 = self.dr * self.dt * self.dz
            self.n2 = 0
            self.n3 = 0
        elif k == -1:
            self.bases = (
                TensorBasis([self.Λ[0], self.Λ[1], self.Λ[2]]),
                TensorBasis([self.Λ[0], self.Λ[1], self.Λ[2]]),
                TensorBasis([self.Λ[0], self.Λ[1], self.Λ[2]]),
            )
            self.shape = (
                (self.nr, self.nt, self.nz),
                (self.nr, self.nt, self.nz),
                (self.nr, self.nt, self.nz),
            )
            self.n1 = self.nr * self.nt * self.nz
            self.n2 = self.nr * self.nt * self.nz
            self.n3 = self.nr * self.nt * self.nz
        else:
            raise ValueError(
                "Degree k must be 0, 1, 2, 3, or -1 (vector field)")
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
            return jnp.int32(0), idx
        elif self.k == 1 or self.k == 2 or self.k == -1:
            n1, n2 = self.n1, self.n2
            category = jnp.int32(idx >= n1) + jnp.int32(idx >= n1 + n2)
            index = jnp.int32(idx - n1 * (idx >= n1) - n2 * (idx >= n1 + n2))
            return category, index

    def _ravel_index(self, c, i, j, k):
        """
        Convert multi-dimensional indices to linear index.

        Args:
            c (int): Component index
            i (int): Index in radial direction
            j (int): Index in poloidal direction
            k (int): Index in toroidal direction

        Returns:
            int: Linear index into the form
        """
        if self.k == 0:
            rav = jnp.ravel_multi_index(
                (i, j, k), (self.nr, self.nt, self.nz), mode="clip"
            )
        elif self.k == 1:
            n1, n2 = self.n1, self.n2
            rav = jnp.where(
                c == 0,
                jnp.ravel_multi_index(
                    (i, j, k), (self.dr, self.nt, self.nz), mode="clip"
                ),
                jnp.where(
                    c == 1,
                    n1
                    + jnp.ravel_multi_index(
                        (i, j, k), (self.nr, self.dt, self.nz), mode="clip"
                    ),
                    n1
                    + n2
                    + jnp.ravel_multi_index(
                        (i, j, k), (self.nr, self.nt, self.dz), mode="clip"
                    ),
                ),
            )
        elif self.k == 2:
            n1, n2 = self.n1, self.n2
            rav = jnp.where(
                c == 0,
                jnp.ravel_multi_index(
                    (i, j, k), (self.nr, self.dt, self.dz), mode="clip"
                ),
                jnp.where(
                    c == 1,
                    n1
                    + jnp.ravel_multi_index(
                        (i, j, k), (self.dr, self.nt, self.dz), mode="clip"
                    ),
                    n1
                    + n2
                    + jnp.ravel_multi_index(
                        (i, j, k), (self.dr, self.dt, self.nz), mode="clip"
                    ),
                ),
            )
        elif self.k == 3:
            rav = jnp.ravel_multi_index(
                (i, j, k), (self.dr, self.dt, self.dz), mode="clip"
            )
        elif self.k == -1:
            n1, n2 = self.n1, self.n2
            _rav = jnp.ravel_multi_index(
                (i, j, k), (self.nr, self.nt, self.nz), mode="clip"
            )
            rav = jnp.where(
                c == 0,
                _rav,
                jnp.where(
                    c == 1,
                    n1 + _rav,
                    n1 + n2 + _rav,
                ),
            )
        return jnp.int32(rav)

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
            return jnp.int32(0), *jnp.unravel_index(idx, (self.nr, self.nt, self.nz))
        elif self.k == 1:
            c, ijk = self._vector_index(idx)
            i, j, k = jnp.where(
                c == 0,
                jnp.array(jnp.unravel_index(ijk, (self.dr, self.nt, self.nz))),
                jnp.where(
                    c == 1,
                    jnp.array(jnp.unravel_index(
                        ijk, (self.nr, self.dt, self.nz))),
                    jnp.array(jnp.unravel_index(
                        ijk, (self.nr, self.nt, self.dz))),
                ),
            )
            return c, i, j, k
        elif self.k == 2:
            c, ijk = self._vector_index(idx)
            i, j, k = jnp.where(
                c == 0,
                jnp.array(jnp.unravel_index(ijk, (self.nr, self.dt, self.dz))),
                jnp.where(
                    c == 1,
                    jnp.array(jnp.unravel_index(
                        ijk, (self.dr, self.nt, self.dz))),
                    jnp.array(jnp.unravel_index(
                        ijk, (self.dr, self.dt, self.nz))),
                ),
            )
            return c, i, j, k
        elif self.k == 3:
            return jnp.int32(0), *jnp.unravel_index(idx, (self.dr, self.dt, self.dz))
        elif self.k == -1:
            c, ijk = self._vector_index(idx)
            i, j, k = jnp.array(jnp.unravel_index(
                ijk, (self.nr, self.nt, self.nz)))
            return c, i, j, k

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
        elif self.k == 1 or self.k == 2 or self.k == -1:
            e = jnp.zeros(3).at[category].set(1)
            val = jnp.where(
                category == 0,
                self.bases[0](x, index),
                jnp.where(
                    category == 1, self.bases[1](
                        x, index), self.bases[2](x, index)
                ),
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
            x (array-like): Point at which to evaluate - always in the logical domain

        Returns:
            array-like: Value of the pushed-forward form at x
        """
        if self.k == 0:
            return self.f(x)
        elif self.k == 1:
            return inv33(jax.jacfwd(self.F)(x)).T @ self.f(x)
        elif self.k == 2:
            return (
                jax.jacfwd(self.F)(x)
                @ self.f(x)
                / jnp.linalg.det(jax.jacfwd(self.F)(x))
            )
        elif self.k == 3:
            return self.f(x) / jnp.linalg.det(jax.jacfwd(self.F)(x))
        elif self.k == -1:
            return (
                jax.jacfwd(self.F)(x)
                @ self.f(x)
            )


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
            return (
                inv33(jax.jacfwd(self.F)(x))
                @ self.f(y)
                * jnp.linalg.det(jax.jacfwd(self.F)(x))
            )
        elif self.k == 3:
            return self.f(y) * jnp.linalg.det(jax.jacfwd(self.F)(x))
        elif self.k == -1:
            return (
                inv33(jax.jacfwd(self.F)(x)) @ self.f(y)
            )
