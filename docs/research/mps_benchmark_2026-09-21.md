# MRX on an Apple GPU: where the time goes

An M3 Pro, macOS 27, `jax-mps` 0.10.11 on jax 0.10.x, MRX at `a1d0b26`, the
plain float32 configuration (`MRX_X64=0`) throughout. The reference mesh is
li383 `(12,24,12)` p=3, deliberately the one behind the v5e / H200 / H100
table in `tpu_v5e_benchmark.md`.

## The headline

The GPU loses the relaxation and wins the solves, and both are the same fact
about dispatch.

| measurement | CPU | MPS | |
|---|---|---|---|
| `relax.py` L-BFGS, 100 steps, end to end | **353.7 s** | 593.9 s | CPU 1.68x |
| per step, from that run | 3.54 s | 5.94 s | CPU 1.68x |
| per step, `relaxation_bench.py` (steady / 5) | 3.68 s | 4.72 s | CPU 1.28x |
| per step, `scan_scaling.py` at chunk 5 | 3.59 s | 5.51 s | CPU 1.53x |
| one nested-CG Laplacian solve, k=1 | 284.3 ms | **74.9 ms** | MPS 3.8x |
| inverse mass CG k=1, 20 iterations | 587 ms | **158 ms** | MPS 3.7x |
| inverse mass CG k=2, 24 iterations | 674 ms | **158 ms** | MPS 4.3x |

The per-step numbers disagree with each other by more than they disagree
between backends, which is worth stating plainly: the run-to-run spread of a
fixed configuration on this machine is about 17%, so nothing below is claimed
on a margin smaller than that. The 1.68x end-to-end figure is the one to
quote, because it is a single 100-step run of each and it is what a user
experiences.

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
