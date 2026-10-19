# Testing strategy

MRX has one lean test tier. Bare `pytest` runs the suite, in float64
([Precision](precision.md)). `bash slurm/suite.sh` submits it as a GPU job,
and the GitHub workflow runs it on a CPU. The suite reads only files
tracked in the repository.

## Four statements

The suite tests four mathematical statements about the discretisation, which between them run most of
the code, plus the field-line tracer, the shape derivative and the readers:

| file | statement |
|---|---|
| `test_derham.py` | the spaces form a de Rham complex: `G_{k+1} G_k = 0` on every space and parity view, the projections commute with the derivatives, the mass matrices are SPD with `D_k = M_{k+1} G_k` and adjoint projection pairs, and the harmonic forms span the cohomology `(1, 1, 0, 0)` of the solid torus |
| `test_solvers.py` | every solve inverts its operator: a known solution, mapped to its right-hand side by the operator's apply, comes back from `M[k]`, `L[k]` (every `part`), `S[k]`, `shifted(k, eps)`, `shifted_stiffness(k, eps)` and the Leray projection to a multiple of `seq.tol`, on the production spaces. Conjugate gradients and MINRES stop on the given norm of the true residual |
| `test_poisson.py` | the Hodge Laplacians k = 0..3 recover manufactured solutions within measured bands, with the same error on the two representations of the domain (below), preconditioners kept across a change of map still precondition the new operators, and the slab model conditions `M_1` |
| `test_newton.py` | the second variation reproduces the energy along the ideal flow, `E'(0) = -(u, J x B)` and `E''(0) = (u, H u)` with the bilinear form by polarisation, the Chebyshev mass inverse keeps the Hessian symmetric, and a short run takes positive steps, lowers the energy and the force, keeps `div B` at round-off and the helicity up to grid-scale effects, and checkpoints, and a few gradient-descent steps descend too |
| `test_poincare.py` | the field-line tracer measures the file's rotational transform on the equilibrium's own field, whose lines stay on their logical surfaces |
| `test_shape_ad.py` | the vacuum field of the shape derivative is harmonic, and its derivative in the map coefficients matches central differences |
| `test_readers.py` | the GVEC and DESC readers reproduce synthetic states, the VMEC reader reads li383, DESC's conversion of it agrees with the wout, non-symmetric files read as shifted symmetric ones, and the angles are right-handed after reading |

The fixtures of `test/conftest.py` build the li383 equilibrium
(`data/wout_li383_low_res_reference.nc`) at `(12, 20, 12)` per field period,
`p = 2`, in two representations with the same mesh: `half`
(the production sequence, stellarator symmetry, half a period integrated)
and `period` (one field period without the reflection). `seq` is `half`,
`b0` the equilibrium's own field on it. A manufactured solution with the
period and parity of li383 must give the same error on both, which tests the
half-period quadrature and the parity views end to end.

The suite is XLA-compile-bound: a test costs the distinct solves it
compiles, not the mesh. Compiled programs are cached on disk
(`MRX_XLA_CACHE`, default `outputs/xla_cache`).

## What a test asserts

Mathematical statements, phrased so that they fail when the mathematics is
wrong and not when an implementation detail moves:

- exact identities to a multiple of `mrx.eps()`.
- solver-based quantities to a multiple of `seq.tol`.
- measured bands, a fixed factor above a measured error, stated next to
  the value. A wrong metric factor or a broken preconditioner moves these
  by a factor, run-to-run noise by a few percent.

A new test states which mathematical claim it checks at which tolerance class, on the
fixtures above.
Dense references and anything that probes every degree of freedom do not
belong in the suite.
