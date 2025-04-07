import jax.numpy as jnp
import jax

from mrx.SplineBases import SplineBasis, DerivativeSpline, TensorBasis
from mrx.Utils import inv33


class DifferentialForm:
    d: int
    k: int
    n: int
    nr: int
    nχ: int
    nζ: int
    ns: jnp.ndarray

    def __init__(self, k, ns, ps, types, Ts=None):
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

        # if k == 0 or k == 3:
        #     self.n = self.bases[0].n
        # else:
        #     self.n = self.bases[0].n + self.bases[1].n + self.bases[2].n
        self.ns = jnp.arange(self.n)

    # def _get_tensor(self, derivs):
    #     return TensorBasis(tuple([self.Λ[i] if derivs[i]==0 else self.dΛ[i] for i in range(self.d)]))

    def _vector_index(self, i):
        if self.k == 0 or self.k == 3:
            return 0, i
        elif self.k == 1 or self.k == 2:
            n1, n2 = self.n1, self.n2  # self.bases[0].n, self.bases[1].n, self.bases[2].n
            # translate a 1D index into a 3D index
            category = jnp.int32(i >= n1) + jnp.int32(i >= n1 + n2)
            index = i - n1 * jnp.int32(i >= n1) - n2 * jnp.int32(i >= n1 + n2)
            return category, index

    def _ravel_index(self, c, i, j, k):
        if self.k == 0:
            return jnp.ravel_multi_index((i, j, k), (self.nr, self.nχ, self.nζ), mode='clip')  # c is always zero
        elif self.k == 1:
            n1, n2, n3 = self.n1, self.n2, self.n3  # self.bases[0].n, self.bases[1].n, self.bases[2].n
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
            n1, n2, n3 = self.n1, self.n2, self.n3  # self.bases[0].n, self.bases[1].n, self.bases[2].n
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

            # if c == 0:
            #     return jnp.ravel_multi_index((i,j,k), self.shape[0])
            # elif c == 1:
            #     return n1 + jnp.ravel_multi_index((i,j,k), self.shape[1])
            # elif c == 2:
            #     return n1 + n2 + jnp.ravel_multi_index((i,j,k), self.shape[2])

    def _unravel_index(self, I):
        if self.k == 0:
            return 0, *jnp.unravel_index(I, (self.nr, self.nχ, self.nζ))
        elif self.k == 1:
            c, ijk = self._vector_index(I)
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
            c, ijk = self._vector_index(I)
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
            return 0, *jnp.unravel_index(I, (self.dr, self.dχ, self.dζ))

    def __call__(self, x, i):
        return self.evaluate(x, i)

    def __getitem__(self, i):
        return lambda x: self.evaluate(x, i)

    def __iter__(self):
        for i in range(self.n):
            yield self[i]

    def __len__(self):
        return self.n

    def evaluate(self, x, i):
        category, index = self._vector_index(i)
        if self.k == 0 or self.k == 3:
            return jnp.ones(1) * self.bases[0](x, index)
        elif self.k == 1 or self.k == 2:
            e = jnp.zeros(3).at[category].set(1)
            # vals = jnp.array([b(x, index) for b in self.bases])
            val = jnp.where(
                category == 0,
                self.bases[0](x, index),  # lambda x: self.bases[0][index](x) = self.bases[0][index]
                jnp.where(
                    category == 1,
                    self.bases[1](x, index),
                    self.bases[2](x, index)
                )
            )
            return e * val  # s[category]


class DiscreteFunction:
    def __init__(self, dof, Λ, E=None):
        self.dof = dof
        self.Λ = Λ
        self.n = Λ.n
        self.ns = jnp.arange(self.n)
        self.E = E if E is not None else jnp.eye(self.n)

    def __call__(self, x):
        return self.dof @ self.E @ jax.vmap(self.Λ, (None, 0))(x, self.ns)


class Pushforward:
    def __init__(self, f, F, k):
        self.k = k
        self.f = f
        self.F = F

    def __call__(self, x):
        y = self.F(x)
        if self.k == 0:
            return self.f(y)
        elif self.k == 1:
            return jax.jacfwd(self.F)(x).T @ self.f(y)
        elif self.k == 2:
            inv33(jax.jacfwd(self.F)(x)) @ self.f(y) * jnp.linalg.det(jax.jacfwd(self.F)(x))
        elif self.k == 3:
            return self.f(y) * jnp.linalg.det(jax.jacfwd(self.F)(x))


class Pullback:
    def __init__(self, f, F, k):
        self.k = k
        self.f = f
        self.F = F

    def __call__(self, x):
        if self.k == 0:
            return self.f(x)
        elif self.k == 1:
            return inv33(jax.jacfwd(self.F)(x)).T @ self.f(x)
        elif self.k == 2:
            return jax.jacfwd(self.F)(x) @ self.f(x) / jnp.linalg.det(jax.jacfwd(self.F)(x))
        elif self.k == 3:
            return self.f(x) / jnp.linalg.det(jax.jacfwd(self.F)(x))
