# Testing Strategy

This note records the intended direction for the `mrx` test suite.

The goal is not to maximize the raw number of tests. The goal is to build a
test suite that is fast enough to run routinely and strong enough to catch
mathematical regressions.

## 1. Default cost model

These are expensive operators. Tests must stay small.

- The default test resolution should be `n = (3, 5, 3)`.
- The default spline degree should be `p = 2` in every direction.
- A test should only go beyond this if the mathematical check genuinely
  requires it.
- If a test can be made cheaper by checking the same identity on a smaller
  object, do that.

The suite should be designed around the assumption that assembly is costly and
should not be repeated casually.

## 2. Default geometries

Most tests should use the rotating ellipse.

- The rotating ellipse is genuinely 3D.
- It is therefore the right default geometry for checking operator identities,
  metric-dependent assembly, Laplacian structure, projections, and
  preconditioners.

Some tests should use the torus.

- Use the torus when we need an analytical or axisymmetric example.
- In particular, tests that depend on known exact solutions or known topology
  should live on the torus.

So the default rule is:

- rotating ellipse for general-purpose tests,
- torus only when the mathematics of the test specifically asks for it.

## 3. Reuse assembled objects

Sequence assembly is expensive, so tests should reuse assembled objects as much
as possible.

- Shared fixtures should build a small number of canonical sequences once and
  reuse them across many tests.
- Prefer session-scoped or module-scoped fixtures for fully assembled test
  sequences.
- Keep at least one shared rotating-ellipse fixture and one shared torus
  fixture.
- Individual tests should assemble their own sequence only when they need a
  genuinely different geometry, resolution, boundary condition setup, or other
  special-case configuration.

This should be the main way we reconcile fast tests with broad mathematical
coverage.

## 4. Test-file organization

The ideal organization is one test file per source file.

- For a source file `mrx/foo.py`, the default target should be
  `test/test_foo.py`.
- The purpose of that file is to test the mathematical contracts owned by that
  module.

This clashes with the need to reuse assembled sequences, so the intended
compromise is:

- keep test ownership local, with one primary test file per source file,
- centralize expensive shared fixtures in `test/conftest.py` or a small number
  of shared test helpers,
- keep cross-module or end-to-end properties in a small number of dedicated
  integration tests.

In other words: distribute assertions by module, not assembly.

## 5. What tests should check

Tests should check mathematical quantities, not generic hygiene conditions.

Good examples include:

- exactness identities such as `d_{k+1} d_k = 0`,
- symmetry or adjointness relations,
- positive-definiteness or semidefiniteness where mathematically expected,
- correct nullspace dimensions and harmonic-space properties,
- consistency between weak and strong operators,
- projection identities and commuting-diagram properties,
- known analytical solutions on geometries where those are available,
- coarse-resolution convergence checks for manufactured or analytical problems,
- solver and preconditioner tests phrased in terms of residual reduction,
  Rayleigh quotients, or spectral behavior.

The suite should prefer tests that fail because the mathematics is wrong, not
because an incidental implementation detail changed.

## 6. What tests should not focus on

Do not spend effort on tests whose only content is checking for `NaN`, `Inf`,
or similar generic pathologies.

- Those issues should be caught indirectly by the real mathematical tests.
- A test that only says "the output is finite" is usually too weak to be worth
  the runtime cost.

The question for each test should be: what mathematical statement is this test
proving?

## 7. Practical guidance

When adding a new test, the default checklist should be:

1. Can this be checked on the shared small rotating ellipse?
2. If not, does it specifically require the torus or another analytical case?
3. Can it reuse an already assembled fixture?
4. What mathematical identity, invariant, or quantitative property is being
   checked?
5. Is this the right module-local test file, or is it truly an integration
   test?

If these questions are answered well, the suite should stay both fast and
useful.

## 8. Intended suite shape

The suite should eventually have three layers.

- Small module-local tests, ideally one file per source file, built around
  shared assembled fixtures.
- A small number of geometry-specific analytical tests, mostly on the torus.
- A small number of integration or regression tests for solver/preconditioner
  workflows that necessarily span several modules.

That should give us a test suite that is mathematically meaningful without
turning every run into a full assembly benchmark.