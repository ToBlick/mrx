# Testing strategy

One tier. `pytest` runs the whole suite on the CPU, in float64 and float32,
in a few minutes on four cores, with no data files. The GitHub workflow runs
exactly that, once per precision; a GPU node runs the same command through
`slurm/run.sh` (see `slurm/README.md`).

## Cost model

Assembly and compilation dominate, not arithmetic, so the suite shares one
session fixture and keeps every object small.

- `tiny_seq` (`test/conftest.py`) is the production setup on a (4, 6, 4) p=2
  spline-interpolated torus: `build_preconditioners` for all eight `(k, BC)`
  pairs and the harmonic forms. It is built once per session; every
  solve-based test runs on it, except the GVEC-route tests and the
  relaxation run, which share one module-scoped (4, 8, 4) p=2 sequence
  built by `build_sequence` from the synthetic export.
- Low-level tests (quadrature, spline bases, the evaluator, projector
  identities, operator identities on the rotating ellipse) build their own
  tiny objects, module-scoped.
- No `pytest-xdist`: the session fixture is shared, and XLA's compile is
  single-threaded, so four cores run the suite as fast as thirty-two. The
  persistent compilation cache (`JAX_COMPILATION_CACHE_DIR`) is the lever on
  wall time.
- A property that holds at any resolution is checked at tiny resolution.
  Resolution-bound accuracy claims (approximation error at production
  resolution, force balance of an equilibrium) belong to the studies under
  `scripts/`, not to the suite.

## What a test asserts

Mathematical statements, phrased so that they fail when the mathematics is
wrong and not when an implementation detail moves:

- exact identities (`d d = 0`, adjointness, projector idempotency, the
  Dirichlet invariant of a boundary term) to a multiple of `mrx.eps()`;
- solver-based quantities to a multiple of `seq.tol` (`mrx.sqrt_eps()`), so
  the same assertion is meaningful in both precisions;
- measured bands. A relative L2 error or an iteration count is asserted
  below a band stated next to the measured value and the date. Bands are
  1.25x a measured error and 2x a measured iteration count: a wrong metric
  factor or a broken preconditioner moves these by a factor, precision and
  run-to-run noise by a few percent.

Coverage of the production path, by test file:

- `test_poisson.py`: the eight Hodge Laplacians `(k = 0..3, free / Dirichlet)`
  solved with the production `'auto'` preconditioner against the
  manufactured solutions of `test/manufactured.py` (shared with
  `scripts/poisson_study.py`), and the Leray projection.
- `test_synthetic_gvec.py`: the GVEC route (`build_sequence` on a file,
  `load_clebsch`, `clebsch_form`, the projection) on a synthetic export
  written by `test/synthetic_gvec.py` -- the layout of the W7-X file, filled
  from closed formulas on a circular torus, so map, field, transform and
  lambda handling are checked against the formulas -- and, on that
  sequence's Clebsch initial condition, the one relaxation run of the
  suite with the most general stepper (CG, linesearch, CFL cap,
  hyperregularisation, resistivity from the first step): energy descent
  against the linesearch prediction, `div B`, the CFL invariant, and the
  helicity rate `dH/dt = -2 eta <J, B>` of the resistive step checked at
  two step sizes. `docs/source/concepts/gvec_mrx_interface.md` section 7 describes the
  synthetic file.
- the remaining files test the module they are named after.

Tests that read files outside the repository (`MRX_W7X_FILE`,
`MRX_RELAX_ARCHIVE`) carry the `needs_data` marker and skip with the missing
path in the reason when the file is absent.

## Adding a test

1. Can it run on `tiny_seq`? If it needs another geometry or parameter, build
   the smallest object that exhibits the property, module-scoped.
2. Which mathematical statement does it check, and at which tolerance class
   (identity, solve, measured band)?
3. If it introduces a band, state the measured value, the date and the
   fixture in a comment next to it.
4. Put it in the file of the module it tests.
