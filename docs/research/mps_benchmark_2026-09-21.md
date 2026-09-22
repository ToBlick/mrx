# MRX on an Apple GPU: where the time goes

An M3 Pro, macOS 27, `jax-mps` 0.10.11 on jax 0.10.x, the plain float32
configuration (`MRX_X64=0`) throughout. The shift-assembly numbers are MRX
at `a1d0b26`; the indexed-assembly numbers are the change recorded below,
on the same machine. The reference mesh is li383 `(12,24,12)` p=3,
deliberately the one behind the v5e / H200 / H100 table in
`tpu_v5e_benchmark.md`.

## The headline

The GPU now wins the L-BFGS relaxation, which is not the default method.
The mass kernel was a dozen dense shifts where Metal wants one indexed read;
with that swapped, and only on Metal, 100 L-BFGS steps at this mesh take
283.8 s on the GPU against 352.5 s on the CPU. Newton is the default, and
its matvec is pinned back to the shift assembly: on Tutorial 4's mesh the
indexed form changed the direction. Both are below.

| measurement | CPU | MPS | |
|---|---|---|---|
| `relax.py`, 100 steps, indexed assembly | 352.5 s | **283.8 s** | MPS 1.24x |
| the same run with the shift assembly | **353.7 s** | 593.9 s | CPU 1.68x |
| per step, indexed | 3.52 s | **2.84 s** | MPS 1.24x |
| per step, shift assembly | 3.54 s | 5.94 s | CPU 1.68x |
| per step, `relaxation_bench.py`, shift assembly | 3.68 s | 4.72 s | CPU 1.28x |
| per step, `scan_scaling.py` chunk 5, shift assembly | 3.59 s | 5.51 s | CPU 1.53x |
| one nested-CG Laplacian solve, k=1 | 284.3 ms | **74.9 ms** | MPS 3.8x |
| inverse mass CG k=1, 20 iterations | 587 ms | **158 ms** | MPS 3.7x |
| inverse mass CG k=2, 24 iterations | 674 ms | **158 ms** | MPS 4.3x |

The per-step numbers disagree with each other by more than a quiet machine
would allow: the run-to-run spread of a fixed configuration here is about
17%, so nothing is claimed on a smaller margin than that. The two
end-to-end rows are the ones to quote. The CPU figure reproduced: the shift
assembly re-run on the same revision came back at 352.5 s against 353.7 s
the first time, and the printed trajectory matched to the last digit, which
is the check that `shift` is the kernel it was.

## The assembly change

`_structured_gather` and `_structured_accumulate` exist because a TPU has no
fast path for indexed memory. Measured on a v5e they were 33x faster than a
gather and a `segment_sum`, and that is the form the kernel has shipped
since. It is the wrong trade against a per-dispatch charge. One compiled
mass apply lowers to 115-126 `slice`s and 61-66 `concatenate`s on top of its
12 `dot_general`s; the indexed form is 6 gathers, 6 scatters and 1
concatenate, with the same 12 `dot_general`s. The index is built with NumPy
from the static shift plan at trace time, so it is a constant of the
executable and `SumfactPlan` does not change.

Inside a jitted scan, milliseconds per apply, best of three:

| | CPU shift | CPU indexed | MPS shift | MPS indexed |
|---|---|---|---|---|
| k=1 | 1.312 | 1.231 (1.07x) | 1.226 | **0.503 (2.44x)** |
| k=2 | 1.134 | 1.205 (0.94x) | 1.200 | **0.475 (2.52x)** |

The decision rule was fixed first: land it only if the indexed form beat
the shift form on MPS by more than the 17% noise. It beat it by 2.5x. On
the CPU it is inside the noise in both directions, so the CPU, and a TPU,
keep the shifts. A per-axis indexed form (one gather per axis rather than
one for the element) also won on MPS, by 1.9x, and lost to the flat form,
so it was not landed. `mps/assembly_ab.py` is the comparison.

What a step actually calls, counted with a host callback on one step so the
Krylov iterations are real ones: the mass kernel 2,757 times at k=1 and
1,228 times at k=2. Priced at the in-scan costs above, that is 4.9 s of a
5.9 s step, and the indexed form brings it down by 2.9 s. The measured step
went from 5.94 s to 2.84 s. The model and the run agree.

The other leaves do not deserve the same treatment. Inside a scan the
incidence stencils cost 0.06-0.14 ms and the mass preconditioner 0.27 ms,
against 1.2 ms for the mass kernel; the 5x the derivative lost as a
standalone call was the mass kernel inside it plus a launch, not the
stencil. 10,545 extraction applies remain, each on the dispatch floor, and
they are the next place the floor shows.

`MRX_ASSEMBLY=indexed` is the default on Metal and `shift` everywhere else,
from `jax.default_backend()`. It is a static argument of `_sumfact_kernel`,
so the two forms do not share a compilation cache entry and the shift trace
is the trace it has always been. The gather agrees exactly and the assembly
agrees to float32 roundoff (`test_indexed_assembly_matches_the_shift_form`). Over
the 100 steps the indexed and shifted GPU runs agree to 4e-6 in the force
at step 1 and then separate, the same float32 chaos as GPU against CPU.

## Newton, measured after the assembly change

`relax.py` defaults to `--method newton`. A step is 300 MINRES iterations
with no criterion of their own (`tol=0.0` inside `newton_direction`), and
each matvec is three k=1 mass solves, so the mass kernel is a larger share
of a Newton step than of an L-BFGS step. `newton_tol=0.1` is the truncation,
not a target the Laplacian atom reaches, so `newton_it` is `+300`.
`--floor-tol 1e-8` cannot fire in float32; these runs pass `--floor-tol 0`.

The suite, re-run after the kernel change: 61 passed in 150 s on MPS with
the indexed assembly, and 61 passed in 300 s on the CPU in the default mixed
configuration, which still traces `shift`. `test_hessian.py` on MPS: the
second variation matches the energy derivatives, and the Newton direction
has `||div u|| = 1.3e-9` and a descent cosine of +0.59.

Two Newton steps from the equilibrium field at `(12,24,12)` p=3, before the
matvec was pinned, so this is the indexed kernel against the shift kernel on
the same step:

| | CPU shift | MPS shift | MPS indexed |
|---|---|---|---|
| wall, 2 steps | 101.3 s | 120.9 s | 86.7 s |
| per step | 50.6 s | 60.5 s | 43.4 s |
| fallbacks | 1/2 | 1/2 | 1/2 |

Step 1 agrees between indexed and shift to 8e-7 in `|F|` and 3e-7 in the
descent cosine (0.994 on all three). The indexed kernel is 1.39x the shift
kernel and 1.17x the CPU. This is the mesh where the change is honest.

Tutorial 4 is not. `(10,16,16)` p=2, 10 steps, warm-started from
`data/tutorials/li383_relaxation/checkpoints/state_000500.h5`:

| | per step | fallbacks | minimum descent cosine |
|---|---|---|---|
| CPU, shift | 21.8 s | 0/10 | 0.20 |
| MPS, shift | 38.9 s | 0/10 | 0.13 |
| MPS, indexed | 31.3 s | 3/10 | 0.007 |

The indexed and shift forces agree at step 1 (9.811e-4 against 9.810e-4).
The directions do not: cosine 0.186 against 0.242 at step 1, 0.007 against
0.127 at step 3, and then three fallbacks and a force spike to 2.9e-3 that
the shift run never has. A matvec good to `sqrt(eps) = 3.5e-4`, repeated 300
times, makes a 1e-7 reordering visible. `newton_direction` therefore pins
`_assembly_override` to `shift` for its own trace. The force around it stays
on the backend's assembly. Re-run with the pin, five steps on the tutorial
mesh: no fallbacks, 39.0 s/step, the shift cost. The GPU still loses to the
CPU at this resolution, indexed or not.

The index constants are not the cost. Distinct component plans total 3.56 MB
at `(12,24,12)` p=3 and 9.13 MB at `(16,32,16)` p=3, under a millisecond to
build in NumPy and about 5 ms to copy to the device, once per trace.

## The per-dispatch floor, measured

`relaxation_bench.py` primitives at `(12,24,12)` p=3, steady ms per apply:

| primitive | CPU | MPS | MPS/CPU |
|---|---|---|---|
| `E` apply k=0 | 0.040 | 0.246 | 6.15x |
| `E` apply k=1 | 0.032 | 0.223 | 6.97x |
| `E` apply k=2 | 0.037 | 0.216 | 5.84x |
| `E^T` apply k=0 | 0.054 | 0.248 | 4.59x |
| `E^T` apply k=1 | 0.053 | 0.240 | 4.53x |
| `E^T` apply k=2 | 0.052 | 0.231 | 4.44x |
| mass gather `x[gather_idx]` | 0.043 | 0.167 | 3.88x |
| mass scatter `segment_sum` | 0.121 | 0.247 | 2.04x |
| `mass_core_apply` k=0 | 0.535 | 0.697 | 1.30x |
| `mass_core_apply` k=1 | 1.269 | 1.868 | 1.47x |
| `mass_core_apply` k=2 | 1.280 | 1.969 | 1.54x |
| `apply_derivative_matrix` k=0 | 1.544 | 5.001 | 3.24x |
| `apply_derivative_matrix` k=1 | 1.779 | 9.172 | 5.16x |
| `apply_derivative_matrix` k=2 | 0.817 | 4.247 | 5.20x |
| `apply_laplacian` k=0 free | 2.324 | 5.666 | 2.44x |
| `apply_laplacian` k=1 free (nested CG) | 284.257 | 74.916 | **0.26x** |

Read the MPS column down the first six rows: 0.246, 0.223, 0.216, 0.248,
0.240, 0.231. Three different form degrees, two different directions, two
different array sizes, and the cost does not move. **That flat ~0.22 ms is the
per-dispatch floor**, and it is not arithmetic -- the CPU does the same work
in 0.03-0.05 ms. Everything else in the table follows from it: an operation
smaller than the floor loses by whatever margin it is smaller, and the one
operation large enough to bury it wins 3.8x.

This is why the suite disagrees with itself. `test_poisson.py` is 24.8 s on
MPS against 45.3 s on CPU, because a Poisson solve is almost entirely the
bottom row. A relaxation step is a few hundred of the top rows.

## The op census

`JAX_MPS_DUMP_OPTIMIZED_IR=<dir>` writes the post-optimisation StableHLO. One
compiled relaxation chunk is ~21,000 ops:

| op | count | | op | count |
|---|---|---|---|---|
| `slice` | 3681 | | `select` | 1317 |
| `gather` | 3060 | | `dot_general` | **1271** |
| `reshape` | 2658 | | `transpose` | 882 |
| `concatenate` | 1969 | | `multiply` | 859 |
| `broadcast_in_dim` | 1759 | | `pad` | 464 |
| `add` | 1428 | | `scatter` | 352 |

1,271 of ~21,000 ops are the arithmetic. The other 94% is data movement that
XLA fuses into its kernels and that MLX, on this evidence, does not fuse as
aggressively. That ratio multiplied by a fixed per-dispatch cost is the whole
result, and it says the lever is op count in `mrx/`, not configuration.

## What was measured and rejected

**`lax.scan` trip count (jax-mps [#215](https://github.com/tillahoffmann/jax-mps/issues/215)) -- refuted.**
The issue reports per-iteration cost growing with the trip count, which would
have made the relaxation quadratic in the chunk. `mps/scan_scaling.py` at
`(8,12,12)` p=2, marginal ms per step between successive chunk lengths:

| chunk | 5 | 10 | 25 | 50 | 100 |
|---|---|---|---|---|---|
| CPU | -- | 175.6 | 177.1 | 147.3 | 170.2 |
| MPS | -- | 641.1 | 632.6 | 562.5 | 532.8 |

Both flat; if anything MPS improves. At `(12,24,12)` p=3 MPS gave 5506.8 and
5504.8 ms/step at chunks 5 and 10, which is flat to 0.04%. The effect does not
reach MRX. This also withdraws the library change that was planned on the
strength of it -- consolidating `chunk_runner`'s 13 separately stacked trace
scalars into one array -- which would have been unjustified work.

**The MLX command-buffer knobs -- no effect.** Against a fixed chunk, ms/step:
baseline 584.9, `ops=200+mb=1024` 548.7, `async_dispatch` 603.5,
`ops_per_buffer=50` 685.2, `=200` 703.8, `=1000` 729.3. The baseline
re-measured on its own gave 614.1 and 525.9, so the entire spread of the sweep
is the noise of the baseline. `MLX_METAL_FAST_SYNCH` is not merely useless but
broken on this build: it aborts with `[metal::Device] Unable to load kernel
input_coherent`.

**Controls confirming the stack is already working.** `JAX_MPS_NO_OPTIMIZE=1`
costs 1.28x (752.2 ms/step) and `MLX_DISABLE_COMPILE=1` costs 2.0x (1148.3
ms/step). The plugin's IR passes and MLX's kernel fusion are both on and both
earning their keep, which is a large part of why there is no configuration win
left to find.

**`jax-mps` 0.10.12.dev847 -- no movement.** 673.5 and 609.0 ms/step against
0.10.11's 614.1 and 525.9. Inside noise, trending slightly worse. Per the
gate set in advance, 0.11.x was not tried.

## The compilation cache does not work on MPS

Same script, same JAX cache settings, two consecutive runs:

| backend | cache entries written | compile, run 1 | compile, run 2 |
|---|---|---|---|
| CPU | 1118 | 9.02 s | 3.37 s |
| MPS | **0** | 7.50 s | 8.58 s |

The plugin never populates the persistent cache, so every MPS run recompiles
from scratch. That interacts badly with the other half of the scan result:
per-step cost is flat in the chunk, but *compile* cost is linear in it, about
7.9 s per step of chunk at `(12,24,12)` p=3. `scripts/relax.py` defaults to
chunk 500 under L-BFGS, which is roughly 65 minutes of uncacheable compile
before the first step is taken.

**This is the one actionable configuration finding.** For a 500-step run:

| | compile | stepping | total |
|---|---|---|---|
| `--chunk 500` | ~65 min | ~46 min | ~111 min |
| `--chunk 25` | ~3 min | ~46 min | ~49 min |

Worth about 2.3x, and it costs nothing but a command-line flag. It is baked
into `mps/env.sh`.

## The physics agrees; the trajectories do not

Both backends ran li383 `(12,24,12)` p=3, L-BFGS, 100 steps, chunk 25. The
force residual by step, and its relative difference:

| step | 1 | 2 | 5 | 10 | 12 | 30 | 100 |
|---|---|---|---|---|---|---|---|
| rel. diff in \|F\| | 6.0e-06 | 2.5e-04 | 1.3e-04 | 4.7e-04 | >1e-02 | 6.9e-01 | 1.6e-01 |

Step 1 agrees to 6e-6. That is float32 roundoff carried through solves whose
own tolerance is `sqrt(eps) = 3.5e-4`, so the two backends are computing the
same step. From there the difference grows exponentially and passes 1% at step
12: a gradient descent run at a loose tolerance amplifies a roundoff-sized
perturbation, and after 100 steps the two runs are on different trajectories
(`||J||/||B||` ends at 0.82 on MPS against 1.57 on CPU).

This is a property of plain float32 relaxation, not of the backend, and the
conserved quantities confirm it -- both runs stay physical:

| | MPS | CPU |
|---|---|---|
| energy removed | 6.03e-05 (0.0121%) | 3.55e-05 (0.0071%) |
| helicity drift, relative | -2.70e-06 | +7.33e-05 |
| `\|\|div B\|\|` max | 1.49e-06 | 1.24e-06 |
| energy increases | 16/100 steps | 19/100 steps |

Do not read the helicity column as the GPU being more accurate: these are two
samples of a chaotic trajectory, not a convergence study. What it does support
is that neither run is diverging or losing solenoidality, i.e. the GPU result
is a valid relaxation and not a broken one.

## What would actually make this faster

Not configuration -- that surface is exhausted above, and the two controls
show the plugin and MLX are already extracting what there is. The measurement
points at one thing: **94% of the ops in a relaxation chunk are data
movement**, and this backend charges ~0.22 ms per dispatch for them. Cutting
the slice / gather / reshape / concatenate count in `mrx/` is the lever, which
is the same kind of work as the data-movement rewrites in
[PR #17](https://github.com/ToBlick/mrx/pull/17) and is out of scope here.

Worth noting what that would be worth: it cannot make the GPU slower than it
already is at the small applies, and the nested-CG row shows a 3.8x waiting on
the other side of the floor. The hardware is not the limit.

## Reproducing

```bash
source mps/env.sh
python mps/probe_ops.py                        # 14 constructs vs CPU
python mps/scan_scaling.py                     # the #215 test
python mps/env_sweep.py                        # the knob sweep
python scripts/benchmark/relaxation_bench.py --ns 12,24,12 --p 3 \
    --precision float32 --relax-steps 5
python scripts/relax.py --ns 12,24,12 --p 3 --precision float32 \
    --method lbfgs --steps 100 --chunk 25
```

Every CPU comparison needs `JAX_PLATFORMS=cpu` explicitly, because installing
the plugin makes `mps` the default backend for the whole environment.
