import jax
import jax.numpy as jnp

import numpy as np
from mrx.Projectors import Projector
from mrx.Quadrature import QuadratureRule
from mrx.LazyMatrices import LazyMassMatrix


class LazyExtractionOperator:
    def __init__(self, Λ, ξ, zero_bc):
        self.k = Λ.k
        self.Λ = Λ
        self.ξ = ξ
        self.nr, self.nχ, self.nζ = Λ.nr, Λ.nχ, Λ.nζ
        self.dr, self.dχ, self.dζ = Λ.dr, Λ.dχ, Λ.dζ
        self.o = 1 if zero_bc == True else 0
        # offset for the boundary conditions

        if self.k == 0:
            self.n1 = ((self.nr - 2 - self.o) * self.nχ + 3) * self.nζ
            self.n2 = 0
            self.n3 = 0
        if self.k == 1:
            self.n1 = (self.dr - 1) * self.nχ * self.nζ
            self.n2 = ((self.nr - 2 - self.o) * self.dχ + 2) * self.nζ
            self.n3 = ((self.nr - 2 - self.o) * self.nχ + 3) * self.dζ
        if self.k == 2:
            self.n1 = ((self.nr - 2 - self.o) * self.dχ + 2) * self.dζ
            self.n2 = (self.dr - 1) * self.nχ * self.dζ
            self.n3 = (self.dr - 1) * self.dχ * self.nζ
        if self.k == 3:
            self.n1 = (self.dr - 1) * self.dχ * self.dζ
            self.n2 = 0
            self.n3 = 0
        self.n = self.n1 + self.n2 + self.n3
        self.M = self.assemble()

    def __getitem__(self, i):
        return self.M[i]

    def __array__(self):
        return np.array(self.M)

    def _vector_index(self, i):
        n1, n2 = self.n1, self.n2
        if self.k == 0 or self.k == 3:
            return 0, i
        elif self.k == 1 or self.k == 2:
            category = jnp.int32(i >= n1) + jnp.int32(i >= n1 + n2)
            index = i - n1 * jnp.int32(i >= n1) - n2 * jnp.int32(i >= n1 + n2)
            return category, index

    def _element(self, I, J):
        if self.k == 0:
            # I ∈ [0, ((nr - 2) nχ + 3) nζ]
            # J ∈ [0, nr nχ nζ]
            return jnp.where(I < 3 * self.nζ,
                             self._inner_zeroform(I, J, self.nr, self.nχ, self.nζ),
                             self._outer_zeroform(I - 3 * self.nζ, J, self.nr, self.nχ, self.nζ))
        if self.k == 1:
            cI, I = self._vector_index(I)
            cJ, J = self.Λ._vector_index(J)
            return jnp.where(cI == 0,
                             # r-component
                             # last argument of _threeform determines if dirichlet BC is applied
                             self._threeform(I, J, self.dr, self.nχ, self.nζ) * jnp.int32(cI == cJ),
                             jnp.where(cI == 1,
                                       # chi-component
                                       jnp.where(I < 2 * self.nζ,
                                                 self.inner_oneform_r(I, J, self.dr, self.nχ, self.nζ) * jnp.int32(cJ == 0) +
                                                 self.inner_oneform_χ(I, J, self.nr, self.dχ, self.nζ) * jnp.int32(cJ == 1),
                                                 self._outer_zeroform(I - 2 * self.nζ, J, self.nr, self.dχ, self.nζ) * jnp.int32(cI == cJ)),
                                       # zeta-component
                                       jnp.where(I < 3 * self.dζ,
                                                 self._inner_zeroform(I, J, self.nr, self.nχ, self.dζ) * jnp.int32(cI == cJ),
                                                 self._outer_zeroform(I - 3 * self.dζ, J, self.nr, self.nχ, self.dζ) * jnp.int32(cI == cJ))
                                       )
                             )
        if self.k == 2:
            cI, I = self._vector_index(I)
            cJ, J = self.Λ._vector_index(J)
            return jnp.where(cI == 0,
                             # r-component
                             jnp.where(I < 2 * self.nζ,
                                       self.inner_oneform_χ(I, J, self.nr, self.dχ, self.dζ) * jnp.int32(cJ == 0) -
                                       self.inner_oneform_r(I, J, self.dr, self.nχ, self.dζ) * jnp.int32(cJ == 1),
                                       self._outer_zeroform(I - 2 * self.nζ, J, self.nr, self.dχ, self.dζ) * jnp.int32(cI == cJ)),
                             jnp.where(cI == 1,
                                       # chi-component
                                       self._threeform(I, J, self.dr, self.nχ, self.dζ) * jnp.int32(cI == cJ),
                                       # zeta-component
                                       self._threeform(I, J, self.dr, self.dχ, self.nζ) * jnp.int32(cI == cJ),
                                       )
                             )
        if self.k == 3:
            return self._threeform(I, J, self.nr, self.nχ, self.nζ)

    # Tensor product basis with three inner C1 bases
    # I and J are "local" indices, i.e. without the category index
    def _inner_zeroform(self, I, J, nr, nχ, nζ):
        l, m = jnp.unravel_index(I, (3, nζ))
        i, j, k = jnp.unravel_index(J, (nr, nχ, nζ))
        return jnp.int32(k == m) * jnp.int32(i < 2) * self.ξ[l, i, j]

    def _outer_zeroform(self, I, J, nr, nχ, nζ):
        i, j, k = jnp.unravel_index(I, (nr, nχ, nζ))
        # if self.o == 1 and i is in the outermost ring => zero
        return (
            jnp.int32(J == jnp.ravel_multi_index((i + 2, j, k), (nr, nχ, nζ), mode='clip'))
            * jnp.where(self.o == 1,
                        jnp.int32(i != nr-1),
                        1)
        )

    # first component of the inner basis: (ξ[l,1,j] - ξ[l,0,j]) Dr[0] Nχ[j] Nζ[k]
    def inner_oneform_r(self, I, J, nr, nχ, nζ):
        l, m = jnp.unravel_index(I, (2, nζ))
        l += 1
        i, j, k = jnp.unravel_index(J, (nr, nχ, nζ))
        return jnp.int32(k == m) * jnp.int32(i == 0) * (self.ξ[l, 1, j] - self.ξ[l, 0, j])

    # second component of the inner basis: (ξ[l,1,j+1] - ξ[l,1,j]) Nr[1] Dχ[j] Nζ[k]
    def inner_oneform_χ(self, I, J, nr, nχ, nζ):
        l, m = jnp.unravel_index(I, (2, nζ))
        l += 1
        i, j, k = jnp.unravel_index(J, (nr, nχ, nζ))
        return jnp.int32(k == m) * jnp.int32(i == 1) * (self.ξ[l, 1, jnp.mod(j+1, nχ)] - self.ξ[l, 1, j])

    # Tensor product basis excluding the inner ring
    def _threeform(self, I, J, nr, nχ, nζ):
        i, j, k = jnp.unravel_index(I, (nr, nχ, nζ))
        return jnp.int32(J == jnp.ravel_multi_index((i + 1, j, k), (nr, nχ, nζ), mode='clip'))

    def assemble(self):
        return jax.vmap(jax.vmap(self._element, (None, 0)), (0, None))(jnp.arange(self.n), jnp.arange(self.Λ.n))


def get_xi(_R, _Y, Λ0, q=3):

    nr, nχ, nζ = Λ0.nr, Λ0.nχ, Λ0.nζ
    Q = QuadratureRule(Λ0, q)
    P = Projector(Λ0, Q)
    M = LazyMassMatrix(Λ0, Q).M

    def R(x):
        return _R(x[0], x[1])

    def Y(x):
        return _Y(x[0], x[1])

    R0 = _R(0, 0)
    Y0 = _Y(0, 0)

    R_hat = jnp.linalg.solve(M, P(R))
    Y_hat = jnp.linalg.solve(M, P(Y))

    cR = R_hat.reshape(nr, nχ, nζ)
    cY = Y_hat.reshape(nr, nχ, nζ)
    ΔR = cR[1, :, 0] - R0
    ΔY = cY[1, :, 0] - Y0
    τ = jnp.max(
        jnp.array(
            [jnp.max(-2 * ΔR),
             jnp.max(ΔR - jnp.sqrt(3) * ΔY),
             jnp.max(ΔR + jnp.sqrt(3) * ΔY)]
        )
    )
    ξ00 = jnp.ones(nχ) / 3
    ξ01 = 1/3 + 2/(3*τ) * ΔR
    ξ10 = jnp.ones(nχ) / 3
    ξ11 = 1/3 - 1/(3*τ) * ΔR + jnp.sqrt(3)/(3*τ) * ΔY
    ξ20 = jnp.ones(nχ) / 3
    ξ21 = 1/3 - 1/(3*τ) * ΔR - jnp.sqrt(3)/(3*τ) * ΔY
    ξ = jnp.array([[ξ00, ξ01], [ξ10, ξ11], [ξ20, ξ21]])  # (3, 2, nχ) -> l, i, j
    return ξ, R_hat, Y_hat, Λ0, τ
