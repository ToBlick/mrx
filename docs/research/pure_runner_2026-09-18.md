# The chunk runner as a pure function of the sequence (2026-09-18)

Branch `worktree-pure-runner` (off `worktree-half-period`), commit 08c747a.
Tobias: "I dislike these large constant foldings. Pure functions look
better to me than closures."

## The problem

`chunk_runner` jitted a closure: `run(state, it0)` reached the time
stepper, the stepper the sequence, and every array an apply touched
became a CONSTANT of the compiled program. JAX's report
(`JAX_CAPTURED_CONSTANTS_WARN_BYTES=1`, `..._REPORT_FRAMES=3`, runs under
`outputs/half_period/const_report_{full,half}`) named them on the li383
(8,12,12) fixture: the metric at every quadrature point (`float32[N_q,3,3]`
through `cross_product_load` and `force_scale`), the cross-product
vectors (`float32[N_q,3]`), the element weight tensors of the
sum-factorised applies (`float64[ne_r,ne_t,ne_z,3,3,3]` through
`sumfact_apply`) -- 6 to 8 MB per chunk compile there, 3.5 GB at
(48,96,96), 10.6 GB for the whole torus at (48,96,288). XLA constant-folds
scatters over them (ten seconds a piece, the `slow_operation_alarm`
lines), keeps them on the host through the compile (the torus needed 320
GB of host memory; 128 was an OOM kill), and every new runner instance
recompiled with its own copy. The metric-lumping atoms had solved this
for themselves in August (their payload is a pytree, the arrays leaves,
one compile per treedef); the sequence had not.

## The design

`mrx/pytree.py`: `register_arrays(cls, static=())` makes a plain class a
JAX pytree of its attributes -- device arrays and nested pytrees are
children, host arrays (numpy), scalars, strings, functions and the named
`static` attributes are the treedef. The static part compares by the
identity of its values (scalars and tuples by value), so two flattenings
of the same object are equal and `jit`'s cache hits across calls.
Unflattening is a shallow copy with the arrays swapped for tracers; a
lazily filled cache on such a copy fills the copy, so the caches are
warmed before the jit (the setup does that already).

Registered: `DeRhamSequence` (static: `equilibrium`, `_gram_core`),
`QuadratureRule`, `MetricLumpingLaplacian`, `MetricLumpingMass`. Already
pytrees: `SequenceGeometry`, `Operators`, `MatrixFreeExtraction`,
`SumfactPlan`, `TimeStepper`, `State` (Equinox modules and NamedTuples).
Rewritten as Equinox modules, their arrays fields instead of closed-over
constants of per-method jits: `FreeProjector` (the half-period parity
projector) and the harmonic Newton atom (`HarmonicAtom`, called as
`atom(seq, x)` with the sequence it is traced with; the stepper stores
the module, not a closure).

The boundaries are `eqx.filter_jit` -- array leaves traced, everything
else static -- on functions that TAKE the sequence: `run(ts, state, it0)`
in `chunk_runner` (the returned callable binds `ts` and casts `it0` to an
array, since a Python int would be static and recompile every chunk),
`probe(seq, aux, ...)` in `make_sampler`, `force_scale_value(seq, B)`,
the reconnection solve `(seq, B, eps)`. The extra probes of the runner
take `(seq, state)`. Inside, the same methods run on the traced copy; no
apply changed.

## Measured

On the (8,12,12) fixture, 4 L-BFGS steps in chunks of 2 (runs
`outputs/half_period/pure_{full,half,newton}`): the chunk's captured
constants 2.4 / 6.2 MB (field period) and 3.4 / 4.6 MB (half) before,
21-57 KB after (one `int64[756]` index array remains). The suite in mixed
precision: 60/60 (17 min). Newton through the harmonic atom steps.

The gate, li383 (16,32,32) p=2 float64, L-BFGS 100 steps with one
reconnection at step 50, against the runs of the half-period note
(`compare_runs.py`, runs `lbfgs_{full,half}_pure` against
`lbfgs_{full,half}`): the reconnection identical to every printed digit
(eps 5.936e-05, |F| 1.976e-03 -> 1.333e-03, H -0.95%); field period: the
force to 2e-6 at step 100, energy to 2e-14, helicity 7e-14, the final
field to 5e-7 -- the round-off-seeded divergence of two builds
([[relaxation-trajectory-roundoff-divergence]]: the constants-as-inputs
program fuses differently); half period: the force to 3e-10, the final
field to 8e-13. Wall per step over the last chunk 0.80 s against 0.86
(the earlier gate quoted 1.30 and 1.40 s/step for the first chunk,
compile included).

At n=48, 20 L-BFGS steps in one chunk, refined float32, compile included
(runs `lbfgs48_{full,torus}_pure`):

| model | before: 20 steps [s] | host peak | pure: 20 steps [s] | host peak |
|---|---|---|---|---|
| one field period (48,96,96) | 607 | -- | 191 | 7.9 GB |
| whole torus (48,96,288) | 1333 | 309 GB (OOM at 128) | 490 | 10.2 GB |

The compile with gigabytes of constants was most of those "20 steps":
the chunk's wall drops 3.2x and 2.7x, and the torus runs in a 128 GB
job. The step kernels are unchanged, so the steady per-step rate is the
same; the paper's three-model table (`outputs/paper`) is remeasured on
the pure runner so that its compile-inclusive numbers mean the same
thing in every row.

## The three-model table, remeasured (runs `outputs/half_period/t2_*`)

Two chunks of 20 L-BFGS steps, refined float32; setup = operators +
harmonic forms, compile = the first chunk's excess over 20 steady steps,
s/step = the second chunk. (`table_t2.py` in the job scratch read the
qoi walls; the paper's `tables/symmetry_table_jcp.tex` is this.)

| model | resolution | DoFs | setup [s] | compile [s] | s/step |
|---|---|---|---|---|---|
| whole torus, no symmetry | (32,64,192) | 1,094,016 | 197 | 29 | 5.45 |
| one field period | (32,64,64) | 364,672 | 107 | 26 | 2.05 |
| half a period | (32,64,64) | 364,672 | 229 | 52 | 1.47 |
| whole torus, no symmetry | (48,96,288) | 3,788,352 | 548 | 34 | 22.9 |
| one field period | (48,96,96) | 1,262,784 | 168 | 27 | 8.20 |
| half a period | (48,96,96) | 1,262,784 | 293 | 45 | 4.78 |

Field period against torus 2.7x per step for 3x fewer DoFs; half against
field period 1.39x at n=32 and 1.72x at n=48. The earlier compile-inclusive
"s/step" (half-period note, section 5) mixed a constants-heavy compile
into the rate: 24.2 / 9.9 / 7.2 at n=32 there against 5.45 / 2.05 / 1.47
steady here. What the half period still pays: the compile (45-52 s
against 26, the projector's kernels) and the cold setup (229 against 107,
293 against 168; in a warm process the build is 70 against 71 s, so the
excess is compilation of the half-period apply path, `symmetrize_like`
and the projector, and the probes on both parities) -- open, small.
