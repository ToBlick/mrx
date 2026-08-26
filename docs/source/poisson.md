# Solve a Poisson problem

This guide solves $-\Delta u = f$ on the unit disk and on a toroid with
manufactured solutions, first with 0-forms, then in mixed form, then for
every form degree and both boundary conditions. The complete scripts are
`scripts/tutorials/polar_poisson.py`, `toroid_poisson.py`, and
`mixed_polar_poisson.py`.

## The map

MRX is written for 3D problems in toroidal geometry, so every map is
$\Phi : [0, 1]^3 \to \mathbb{R}^3$. The unit disk is a disk in the
$(x, z)$ plane extruded along $-y$:

```python
import jax
import jax.numpy as jnp

def disk_map(x):
    r, θ, z = x
    return jnp.array([r * jnp.cos(2 * jnp.pi * θ), -z, r * jnp.sin(2 * jnp.pi * θ)])
```

`mrx.mappings` has the toroid, the cylinder, and a rotating ellipse.

## The manufactured solution

Scalar functions return arrays with one entry. Both $u$ and $f$ take
logical coordinates:

```python
def u(x):
    r, _, _ = x
    return jnp.ones(1) * (r**3 * (3 * jnp.log(r) - 2) / 27 + 2 / 27)

def f(x):
    r, _, _ = x
    return -jnp.ones(1) * r * jnp.log(r)
```

$u$ vanishes at $r = 1$ and is in $H^s$ only for $s < 4$, which caps the
convergence order.

## The sequence

`DeRhamSequence` holds the four spline spaces, the quadrature rule, and
the extraction operators. All code is 3D; a 2D problem uses one constant
basis function in the third direction:

```python
from mrx.derham_sequence import DeRhamSequence

n, p = 8, 3
seq = DeRhamSequence((n, n, 1), (p, p, 0), p + 1,
                     ("clamped", "periodic", "constant"), polar=True)
seq.evaluate_1d()
```

- `ns` is the number of basis functions and `ps` the degree per direction.
- `q = p + 1` is the number of Gauss points per knot span.
- `types` is `clamped`, `periodic`, or `constant` per direction.
- `polar=True` fuses the innermost rings so that fields are smooth at
  $r = 0$. See [The polar axis](concepts/polar.md).
- `evaluate_1d` tabulates the 1D bases at the quadrature points. Every
  assembly reads these tables.

Dirichlet conditions are not a property of the sequence. Every operator
takes `dirichlet=True` or `False` and picks the matching extraction
operator (`seq.e0_dbc` or `seq.e0` for 0-forms). See
[Architecture](concepts/architecture.md) section 3.

## The map and the preconditioners

Install the map and build the preconditioners for the solves you need:

```python
seq.set_map_and_preconditioners(disk_map, ks=(0,), dirichlets=(True,))
```

`set_map` evaluates the Jacobian of the map at the quadrature points and
stores it on `seq.geometry`. `build_preconditioners` assembles the
incidence operators, the mass preconditioners, and the metric-lumping
Laplacian preconditioners for the listed degrees and boundary conditions.
Call it again after every map change. See
[Solvers and preconditioners](concepts/preconditioning.md).

## Solve

The right-hand side is the dual load vector
$(\Pi_0 f)_i = \int f \, \Lambda^0_i \, \det D\Phi \, d\hat x$. The
solve is CG on the stiffness matrix with the metric-lumping
preconditioner:

```python
from mrx.differential_forms import DiscreteFunction

rhs = seq.load(f, 0, dirichlet=True)
u_hat, info = seq.apply_inverse_laplacian(rhs, 0, dirichlet=True, return_info=True)
u_h = DiscreteFunction(u_hat, seq.basis_0, seq.e0_dbc)
```

`info` is `-k` after `k` iterations if the solve converged and `+k` if it
did not. `u_h(x)` evaluates the solution at a logical point.

## Check the error

Evaluate both functions at the quadrature points and integrate with the
Jacobian:

```python
def relative_l2_error(seq, u_h, u_exact):
    diff = jax.vmap(lambda x: u_exact(x) - u_h(x))(seq.quad.x)
    exact = jax.vmap(u_exact)(seq.quad.x)
    w = seq.quad.w * seq.jacobian_j
    return float(jnp.sqrt((diff**2).sum(1) @ w) / jnp.sqrt((exact**2).sum(1) @ w))
```

Run `scripts/tutorials/polar_poisson.py --n 6 8 12 16 --p 1 2 3` to see
the error and the iteration count per resolution.

## A toroid

The toroid of major radius 1 and minor radius $\varepsilon$ is
`toroid_map(epsilon=ε)`. The exact solution
$u = (r^2 - r^4)\cos(2\pi\zeta)/4$ has the source listed in
[Manufactured solutions](concepts/manufactured_solutions.md). The only
changes are the map and the spaces:

```python
from mrx.mappings import toroid_map

seq = DeRhamSequence((n, 2 * n, n), (p, p, p), p + 1,
                     ("clamped", "periodic", "periodic"), polar=True)
seq.evaluate_1d()
seq.set_map_and_preconditioners(toroid_map(epsilon=1 / 3), ks=(0,), dirichlets=(True,))
```

This is the geometry of the convergence tests in the paper. The script
is `scripts/tutorials/toroid_poisson.py`.

## A spline map

A map given by data is fitted as three scalar splines. For an analytic
map, `greville_interpolate_map` fits by collocation at the Greville
points:

```python
from mrx.geometry import greville_interpolate_map

coeffs = greville_interpolate_map(toroid_map(epsilon=1 / 3), seq)   # shape (3, seq.n0)
seq.set_spline_map(coeffs)
seq.build_preconditioners(ks=(0,), dirichlets=(True,))
```

`set_spline_map` computes the geometry by sum factorisation of the
coefficients; the coefficients are a pytree, so a shape optimisation can
differentiate through them. GVEC files go through
`mrx.gvec.build_gvec_map`; see
[Solve a relaxation problem](relaxation.md).

## Mixed form

Write the Poisson equation as $\nabla \cdot \sigma = f$,
$-\nabla u = \sigma$ with $u$ a 3-form and $\sigma$ a 2-form. The first
equation holds strongly and the second weakly:

$$
\begin{bmatrix} \mathbb{M}_2 & -\mathbb{D}^T \\ \mathbb{D} & 0 \end{bmatrix}
\begin{bmatrix} \sigma \\ u \end{bmatrix} =
\begin{bmatrix} 0 \\ \Pi_3 f \end{bmatrix},
\qquad
\mathbb{D} \mathbb{M}_2^{-1} \mathbb{D}^T u = \Pi_3 f .
$$

The operator on the left is the 3-form Hodge Laplacian. Its natural
boundary condition is $u = 0$, so the solve uses the free space. The k=3
solve is a saddle-point solve with $\mathbb{M}_2^{-1}$ inside, so build
the k=2 mass and the k=3 Laplacian preconditioners, and initialise the
harmonic forms (none on a disk):

```python
from mrx.differential_forms import Pushforward
from mrx.nullspace import init_nullspaces

seq = DeRhamSequence((n, n, 1), (p, p, 0), p + 1,
                     ("clamped", "periodic", "constant"), polar=True,
                     betti_numbers=(1, 0, 0, 0))
seq.evaluate_1d()
seq.set_map_and_preconditioners(disk_map, ks=(2, 3), dirichlets=(False,))
seq.set_operators(init_nullspaces(seq, seq.get_operators()))

rhs = seq.load(f, 3)
u_hat, info = seq.apply_inverse_laplacian(rhs, 3, dirichlet=False, return_info=True)
u_h = Pushforward(DiscreteFunction(u_hat, seq.basis_3, seq.e3), disk_map, 3)
```

A 3-form load carries no Jacobian:
$(\Pi_3 f)_i = \int f \, \Lambda^3_i \, d\hat x$. The pushforward of a
3-form divides by $\det D\Phi$, which is what `Pushforward(..., 3)`
does. The manufactured solution
$u = -(r^4/16 - r^3/12 + 1/48)$, $f = r(r - 3/4)$ has zero-mean source
and $u(1) = 0$. The script is `scripts/tutorials/mixed_polar_poisson.py`.

## Every degree and both boundary conditions

The Hodge Laplacian of degree $k$ is
$L_k = K_k + D_{k-1} M_{k-1}^{-1} D_{k-1}^T$. `apply_inverse_laplacian`
solves it for every $k$: CG for $k = 0$, MINRES on the saddle-point system
for $k \geq 1$. On a solid torus, $L_k$ has a kernel of harmonic forms
whose dimension follows from the Betti numbers `(1, 1, 0, 0)`:

| | free | Dirichlet |
|---|---|---|
| k=0 | 1 (the constant) | 0 |
| k=1 | 1 (toroidal 1-form) | 0 |
| k=2 | 0 | 1 (toroidal 2-form) |
| k=3 | 0 | 1 (the constant) |

Build every preconditioner, then compute the harmonic forms. Solves
deflate them:

```python
from mrx.nullspace import compute_nullspaces

seq = DeRhamSequence((n, 2 * n, n), (p, p, p), p + 1,
                     ("clamped", "periodic", "periodic"), polar=True,
                     betti_numbers=(1, 1, 0, 0))
seq.evaluate_1d()
seq.set_map_and_preconditioners(toroid_map(epsilon=1 / 3))
seq.set_operators(compute_nullspaces(seq, seq.get_operators()))
```

Then, for a vector source `f_vec` returning physical components:

```python
rhs = seq.load(f_vec, 1, dirichlet=False)
a_hat = seq.apply_inverse_laplacian(rhs, 1, dirichlet=False)
```

The Dirichlet spaces enforce $u = 0$ for 0-forms, $u \times n = 0$ for
1-forms, and $u \cdot n = 0$ for 2-forms at $r = 1$; the free spaces
carry the complementary natural conditions. `load` takes physical
components by default and pulls them back; pass `frame='ref'` to hand
over reference components instead, in the convention of `load`'s
docstring, which is not the primal coefficient vector for 2-forms.

## The batch study

`scripts/poisson_study.py` is the convergence study on the toroid: all
eight `(k, boundary condition)` cases in one process, sharing one
sequence and one assembly pass per resolution. Its config is
`conf/config_poisson_test.yaml` with schema `mrx.config.PoissonTestConfig`.
Override any key on the command line:

```bash
python -u scripts/poisson_study.py p=3 n=[8,16]
python -u scripts/poisson_study.py p=2 n=16 precision=float32
```

Each run writes `outputs/<date>/<time>/result.json`, a list with one
entry per `n`, each a dict over the eight cases:

| key | meaning |
|---|---|
| `error` | relative L2 error against the manufactured solution |
| `iters`, `converged` | iteration count and convergence flag |
| `final_rel_residual` | $\|L u - b\| / \|b\|$ after the solve |
| `timings` | seconds per stage, compile and execute separately |

The file is rewritten after every `n`, so an out-of-memory failure at a
large `n` keeps the earlier results. The summary table at the end of the
log lists `n`, `error`, and total time per resolution; the error should
fall as $h^{p+1}$ in the L2 norm for 0-forms. To run on a cluster, see
[Running on a cluster](cluster.md).
