import jax
import jax.numpy as jnp


class SplineBasis:

    """
    Parameters:
    -----------
    n:
        The number of splines in the basis
    degree: int
        The degree of the spline
    type: {clamped, periodic}
        The type of the spline
    uniform_knots: bool
        Whether the knots are uniform or not
    """

    n: int
    ns: jnp.ndarray
    p: int
    type: str
    T: jnp.ndarray

    def __init__(self, n, p, type, T):
        self.n = n
        self.ns = jnp.arange(self.n)
        self.p = p
        self.type = type
        if T is not None:
            self.T = T
        else:
            self.T = self._init_knots()

    def __call__(self, x, i):
        return self.evaluate(x, i)

    def __getitem__(self, i):
        return lambda x: self.evaluate(x, i)

    def _init_knots(self):
        n = self.n
        p = self.p
        if self.type == 'periodic':
            _T = jnp.linspace(0, 1, n+1)
            T = jnp.concatenate([
                _T[-(p+1):-1] - 1,
                _T,
                _T[1:(p+1)] + 1
            ])
        elif self.type == 'clamped':
            T = jnp.concatenate([
                jnp.zeros(p),
                jnp.linspace(0, 1, n-p+1),
                jnp.ones(p)
            ])
        elif self.type == 'constant':
            T = jnp.array([0, 1])
        return T

    def evaluate(self, x, i):
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
        else:
            return 1.0

    def _evaluate(self, x, i):
        """
        Evaluate the ith spline at x

        Parameters:
        -----------
        x: float
            The point at which to evaluate the spline
        i: int
            The index of the spline to evaluate
        """
        if self.p == 0:
            return jnp.where(
                jnp.logical_and(self.T[i] <= x, x <= self.T[i+1]),
                self._const_spline(x, jax.lax.dynamic_slice(self.T, (i,), (2,))),
                0)
        elif self.p == 1:
            return jnp.where(
                jnp.logical_and(self.T[i] <= x, x <= self.T[i+2]),
                self._lin_spline(x, jax.lax.dynamic_slice(self.T, (i,), (3,))),
                0)
        elif self.p == 2:
            return jnp.where(
                jnp.logical_and(self.T[i] <= x, x <= self.T[i+3]),
                self._quad_spline(x, jax.lax.dynamic_slice(self.T, (i,), (4,))),
                0)
        elif self.p == 3:
            N_i_2 = jnp.where(
                jnp.logical_and(self.T[i] <= x, x <= self.T[i+3]),
                self._quad_spline(x, jax.lax.dynamic_slice(self.T, (i,), (4,))),
                0)
            N_iplus1_2 = jnp.where(
                jnp.logical_and(self.T[i+1] <= x, x <= self.T[i+4]),
                self._quad_spline(x, jax.lax.dynamic_slice(self.T, (i+1,), (4,))),
                0)
            return self.__safe_divide(x - self.T[i], self.T[i+3] - self.T[i]) * N_i_2 + \
                self.__safe_divide(self.T[i+4] - x, self.T[i+4] - self.T[i+1]) * N_iplus1_2

    def __safe_divide(self, x, y):
        return jax.lax.cond(
            y == 0,
            lambda x: jnp.zeros_like(x),
            lambda x: x/y,
            operand=x)

    def _const_spline(self, x, t):
        # t is a vector of two elements - the start and end of the interval where the spline is non-zero
        # S₀(x) = 1 if t₀ ≤ x < t₁,
        #         0 otherwise
        return jnp.where(jnp.logical_and(t[0] <= x, x < t[1]), 1.0, 0.0)

    def _lin_spline(self, x, t):
        # t is a vector of three elements
        # S₁(x) = (x - t₀)/(t₁ - t₀) if t₀ ≤ x < t₁,
        #         (t₂ - x)/(t₂ - t₁) if t₁ ≤ x < t₂,
        #                  0         otherwise
        return jnp.where(
            x < t[1],
            self.__safe_divide(x - t[0], t[1] - t[0]),
            self.__safe_divide(t[2] - x, t[2] - t[1])
        )

    def _quad_spline(self, x, t):
        # t is a vector of four elements
        # S₂(x) = (x - t₀)²/(t₁ - t₀)(t₂ - t₁)              if t₀ ≤ x < t₁,
        #         (x - t₀)(t₂ - x)/(t₀ - t₁)(t₂ - t₀) +
        #           (x - t₁)(t₃- x)/(t₂ - t₁)(t₃ - t₁)      if t₁ ≤ x < t₂,
        #         (t₃ - x)²/(t₃ - t₁)(t₂ - t₁)              if t₂ ≤ x < t₃,
        #         0 otherwise
        return jnp.where(
            x < t[1],
            self.__safe_divide((x - t[0])**2, (t[1] - t[0])*(t[2] - t[0])),
            jnp.where(
                x < t[2],
                self.__safe_divide((x - t[0])*(t[2] - x), (t[2] - t[0])*(t[2] - t[1])) +
                self.__safe_divide((x - t[1])*(t[3] - x), (t[2] - t[1])*(t[3] - t[1])),
                self.__safe_divide((t[3] - x)**2, (t[3] - t[1])*(t[3] - t[2]))
            )
        )


class TensorBasis:
    def __init__(self, bases):
        # we need to drag the n along with us here since JAX cannot infer it from the bases during tracing/compilation
        self.bases = bases
        self.shape = jnp.array([b.n for b in bases])
        self.n = bases[0].n * bases[1].n * bases[2].n
        self.ns = jnp.arange(self.n)

    def evaluate(self, x, i):
        ijk = jnp.unravel_index(i, self.shape)
        return self.bases[0](x[0], ijk[0]) * self.bases[1](x[1], ijk[1]) * self.bases[2](x[2], ijk[2])

    def __call__(self, x, i):
        return self.evaluate(x, i)

    def __getitem__(self, i):
        return lambda x: self.evaluate(x, i)


class DerivativeSpline:
    def __init__(self, s):
        self.n = s.n - 1 if s.type == 'clamped' else s.n
        self.p = s.p if s.type == 'constant' else s.p - 1
        self.type = s.type
        self.T = s.T[1:-1] if s.type == 'periodic' else s.T
        self.s = SplineBasis(self.n, self.p, self.type, self.T)

    def __call__(self, x, i):
        return self.evaluate(x, i)

    def __getitem__(self, i):
        return lambda x: self.evaluate(x, i)

    def evaluate(self, x, i):
        p = self.p
        n = self.n
        if self.type == 'clamped':
            return jax.lax.cond(
                i < n,
                lambda x: self.s(x, i+1) * (p+1) / (self.s.T[i+p+2] - self.s.T[i+1]),
                lambda x: 0.0,
                operand=x
            )
        elif self.type == 'periodic':
            j = jnp.mod(i + n, n)
            return self.s(x, j) * (p+1) / (self.s.T[j+p+1] - self.s.T[j])
            # return jnp.where(
            #     i + 1 - p > 0,
            #     self.s(x, i+1-p) * (p+1) / (self.s.T[i+p+2-p] - self.s.T[i+1-p]),
            #     self.s(x, 0) * (p+1) / (self.s.T[p+1] - self.s.T[0])
            # )
        # the derivative right now is not exact on the edge - but these functions should never be evaluated there anyway.
        else:
            return 1.0
