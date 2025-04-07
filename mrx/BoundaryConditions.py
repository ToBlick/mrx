import jax
import jax.numpy as jnp
import numpy as np


# Bpundary extraction operator for cube-like domains


class LazyBoundaryOperator:
    def __init__(self, Λ, types):
        # types can be:
        # - 'dirichlet' (zero at 0 and 1)
        # - else (no boundary conditions)
        self.k = Λ.k
        self.Λ = Λ
        self.nr = Λ.nr - 2 if types[0] == 'dirichlet' else Λ.nr
        self.nχ = Λ.nχ - 2 if types[1] == 'dirichlet' else Λ.nχ
        self.nζ = Λ.nζ - 2 if types[2] == 'dirichlet' else Λ.nζ
        self.dr, self.dχ, self.dζ = Λ.dr, Λ.dχ, Λ.dζ

        if self.k == 0:
            self.n1 = self.nr * self.nχ * self.nζ
            self.n2 = 0
            self.n3 = 0
        if self.k == 1:
            self.n1 = self.dr * self.nχ * self.nζ
            self.n2 = self.nr * self.dχ * self.nζ
            self.n3 = self.nr * self.nχ * self.dζ
        elif self.k == 2:
            self.n1 = self.nr * self.dχ * self.dζ
            self.n2 = self.dr * self.nχ * self.dζ
            self.n3 = self.dr * self.dχ * self.nζ
        elif self.k == 3:
            self.n1 = self.dr * self.dχ * self.dζ
            self.n2 = 0
            self.n3 = 0
        self.n = self.n1 + self.n2 + self.n3

        self.M = self.assemble()

    def __getitem__(self, i):
        return self.M[i]

    def __array__(self):
        return np.array(self.M)

    def _vector_index(self, i):
        if self.k == 0 or self.k == 3:
            return 0, i
        elif self.k == 1 or self.k == 2:
            n1, n2 = self.n1, self.n2
            category = jnp.int32(i >= n1) + jnp.int32(i >= n1 + n2)
            index = i - n1 * jnp.int32(i >= n1) - n2 * jnp.int32(i >= n1 + n2)
            return category, index

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

    def _element(self, I, J):
        cI, i, j, k = self._unravel_index(I)
        cJ, l, m, n = self.Λ._unravel_index(J)
        if self.k == 0:
            # for example: dirichlet boundary condition in r and ζ:
            # I ∈ [0, (nr-2) nχ (nζ-2)]
            # J ∈ [0,   nr   nχ   nζ  ]
            return (
                (jnp.int32(self.nr == self.Λ.nr) * jnp.int32(i == l)
                    + jnp.int32(self.nr != self.Λ.nr) * jnp.int32(i == l-1))
                * (jnp.int32(self.nχ == self.Λ.nχ) * jnp.int32(j == m)
                   + jnp.int32(self.nχ != self.Λ.nχ) * jnp.int32(j == m-1))
                * (jnp.int32(self.nζ == self.Λ.nζ) * jnp.int32(k == n)
                   + jnp.int32(self.nζ != self.Λ.nζ) * jnp.int32(k == n-1))
            )
        elif self.k == 1:
            # for the x-component, it is dr x nχ x nζ
            return jnp.where(cI == cJ,
                             jnp.where(cI == 0,
                                       jnp.int32(i == l)
                                       * (jnp.int32(self.nχ == self.Λ.nχ) * jnp.int32(j == m)
                                          + jnp.int32(self.nχ != self.Λ.nχ) * jnp.int32(j == m-1))
                                       * (jnp.int32(self.nζ == self.Λ.nζ) * jnp.int32(k == n)
                                           + jnp.int32(self.nζ != self.Λ.nζ) * jnp.int32(k == n-1)),
                                       # for the y-component, it is nr x dχ x nζ
                                       jnp.where(cI == 1,
                                                 (jnp.int32(self.nr == self.Λ.nr) * jnp.int32(i == l)
                                                  + jnp.int32(self.nr != self.Λ.nr) * jnp.int32(i == l-1))
                                                 * jnp.int32(j == m)
                                                 * (jnp.int32(self.nζ == self.Λ.nζ) * jnp.int32(k == n)
                                                     + jnp.int32(self.nζ != self.Λ.nζ) * jnp.int32(k == n-1)),
                                                 # for the z-component, it is nr x nχ x dζ
                                                 (jnp.int32(self.nr == self.Λ.nr) * jnp.int32(i == l)
                                                     + jnp.int32(self.nr != self.Λ.nr) * jnp.int32(i == l-1))
                                                 * (jnp.int32(self.nχ == self.Λ.nχ) * jnp.int32(j == m)
                                                     + jnp.int32(self.nχ != self.Λ.nχ) * jnp.int32(j == m-1))
                                                 * jnp.int32(k == n)
                                                 )
                                       ),
                             0
                             )
        elif self.k == 2:
            # for the x-component, it is nr x dχ x dζ
            return jnp.where(cI == cJ,
                             jnp.where(cI == 0,
                                       (jnp.int32(self.nr == self.Λ.nr) * jnp.int32(i == l)
                                        + jnp.int32(self.nr != self.Λ.nr) * jnp.int32(i == l-1))
                                       * jnp.int32(j == m)
                                       * jnp.int32(k == n),
                                       # for the y-component, it is dr x nχ x dζ
                                       jnp.where(cI == 1,
                                                 jnp.int32(i == l)
                                                 * (jnp.int32(self.nχ == self.Λ.nχ) * jnp.int32(j == m)
                                                    + jnp.int32(self.nχ != self.Λ.nχ) * jnp.int32(j == m-1))
                                                 * jnp.int32(k == n),
                                                 # for the z-component, it is nr x nχ x dζ
                                                 jnp.int32(i == l)
                                                 * jnp.int32(j == m)
                                                 * (jnp.int32(self.nζ == self.Λ.nζ) * jnp.int32(k == n)
                                                     + jnp.int32(self.nζ != self.Λ.nζ) * jnp.int32(k == n-1))
                                                 )
                                       ),
                             0
                             )
        elif self.k == 3:
            return jnp.int32(I == J)

    def assemble(self):
        return jax.vmap(jax.vmap(self._element, (None, 0)), (0, None))(jnp.arange(self.n), jnp.arange(self.Λ.n))
