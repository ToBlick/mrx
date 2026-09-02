# Testing strategy

One lean tier. `pytest` runs the whole suite in a few minutes on a GPU or on
four CPU cores, in float64 and float32, reading only files tracked in the
repository. The GitHub workflow runs exactly that, once per precision; a GPU
node runs the same command through `slurm/run.sh` (see `slurm/README.md`).

## One sequence

Every solve-based test runs on one session fixture, `seq` in
`test/conftest.py`: the li383 equilibrium (`data/wout_li383_low_res_reference.nc`,
the project's fruit-fly stellarator) at `(8, 12, 12)` p=3, built once by
`build_sequence` with the metric-lumping atoms for all eight `(k, BC)` pairs
and the harmonic forms. A second session fixture, `b0`, is the state's own
field `B = dA'` from the histopolated Clebsch potential. Nothing else builds
a sequence.

The suite is XLA-compile-bound: an eager solve traces and compiles its own
loop body, so a test costs what it compiles, not what it computes, and four
cores run the suite as fast as thirty-two. Two consequences:

- the persistent compilation cache (`JAX_COMPILATION_CACHE_DIR`) is the lever
  on wall time in CI;
- a long chain of distinct compilations in one process is what broke the
  old, larger suite on the CPU backend (XLA:CPU died after some thousands of
  executables). Twenty-odd tests do not get there.

## What is tested

| file | statement | tolerance |
|---|---|---|
| `test_assembly.py` | `M_k x` from the fused sum-factorised kernel equals evaluate -> metric weight -> integrate, one random vector per `k`; the projection pairs are transposes | `1e3 eps` |
| `test_complex.py` | `d d = 0` with the polar strong derivative, both BCs; the two harmonic forms have a roundoff Rayleigh quotient and an identity Gram matrix | `eps`-scaled; `seq.tol` |
| `test_poisson.py` | the analytic vacuum field `grad Psi` is recovered by the k=0 potential solve (Route A), the k=1 vector-potential solve (Route C, the Hodge split) and the k=2 shifted solve `(M_2 + eps L_2)` at the smoothing `eps`; the Leray projection (the k=3 solve) removes a co-exact perturbation of `b0` exactly | measured bands, 1.25x; `100 seq.tol` |
| `test_relaxation.py` | 100 production steps on `b0`: energy monotone at every recorded point, the force norm drops by the measured factor, helicity conserved to `25 seq.tol`, `div B` at roundoff | measured band; `seq.tol` |
| `test_readers.py` | the GVEC parser reproduces the closed-form synthetic state; the VMEC reader reads li383 with the expected layout | roundoff; exact |
| `test_spline_bases.py`, `test_quadrature.py`, `test_precision.py`, `test_preconditioner_kind_dispatch.py` | partition of unity and the histopolation de Rham identity; quadrature exactness; the working dtype and matmul precision; every accepted preconditioner kind is dispatched | roundoff; AST |

The vacuum construction (`test/manufactured.py`) is the one the convergence
study `scripts/analytic_vacuum.py` uses: the field is analytic in physical
coordinates and evaluated through the map, each problem is posed as a best
approximation, and the error against the exact field is computed from the
load, so no closed-form metric and no boundary data appear. Resolution-bound
accuracy claims (convergence rates, force balance of an equilibrium) belong to
the studies under `scripts/`, not here.

## What a test asserts

Mathematical statements, phrased so that they fail when the mathematics is
wrong and not when an implementation detail moves:

- exact identities to a multiple of `mrx.eps()`, so the same assertion is
  meaningful in both precisions;
- solver-based quantities to a multiple of `seq.tol` (`mrx.sqrt_eps()`);
- measured bands, 1.25x a measured error, stated next to the value, the date
  and the fixture. A wrong metric factor or a broken preconditioner moves
  these by a factor, precision and run-to-run noise by a few percent.

## Adding a test

A new test is the production configuration plus at most one contrasting
case, on `seq`. It states which mathematical claim it checks and at which
tolerance class; if it introduces a band it records the measured value and
the date. Dense references and anything that probes every degree of freedom
do not belong in the suite; the studies under `scripts/` and the GPU jobs in
`slurm/` are where expensive checks live.
