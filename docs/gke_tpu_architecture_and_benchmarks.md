# MRX on Cloud TPUs: benchmarks

What was measured on Cloud TPUs, how it compares with one NVIDIA H200 GPU, and where the time goes. [`gke/README.md`](../gke/README.md) explains how to run the jobs.

## Setup

* **Case:** `data/wout_li383_1.4m.nc`, spline degree p = 2, JAX 0.11.2, libtpu 0.0.48.1. S, M, L and XL are the MRX authors' default meshes: 32×44×20, 48×64×24, 64×80×32 and 80×104×40.
* **TPU:** float32 + double-single (`MRX_DTYPE=float32`), since TPUs have no float64. A node has 4 chips, and `make deploy-sharded` ([`scripts/bench/bench_multichip.py`](../scripts/bench/bench_multichip.py), `jax.shard_map`) runs 4 equilibria at once, one per chip. "Per equilibrium" is the node's time divided by 4; "one chip" is one equilibrium alone on one chip.
* **GPU:** one NVIDIA H200 in float64. The 8×12×12 numbers are from [`research/solvers_precision.md`](research/solvers_precision.md); the timings and energy drops at 48×64×24 and 64×80×32 are the MRX authors' figures (`main`, 2026-10-13), approximate since the solvers changed in #35.

| | TPU v6e (Trillium) | TPU v5p | NVIDIA H200 |
| :--- | :--- | :--- | :--- |
| Billed unit | node of 4 chips (`ct6e-standard-4t`) | node of 4 chips (`ct5p-hightpu-4t`) | 1 GPU |
| Memory per chip | 32 GB, 1.6 TB/s | 95 GB, 2.8 TB/s | 141 GB, 4.8 TB/s |
| Spot price | $2.20/h | $5.04/h | about $4.00/h (assumed: Google Cloud sells the H200 in 8-GPU VMs) |

## Results

**Newton**, seconds per step, per equilibrium:

| Mesh | H200 | v6e node | v5p node | One chip, v6e / v5p |
| :--- | ---: | ---: | ---: | ---: |
| 8×12×12 | 3.12 | **0.58** | 0.65 | 2.28 / 2.63 |
| 12×20×12 | — | **1.49** | 1.53 | 5.91 / 6.23 |
| S, 32×44×20 | — | 13.4 | **9.65** | — / 40.7 |
| M, 48×64×24 | **5.35** | — | 27.4–30.0 | — |
| L, 64×80×32 | 8.22 | not run | not run | — |

**Gradient**, milliseconds per step, per equilibrium:

| Mesh | H200 | v6e node | v5p node | One chip, v6e / v5p |
| :--- | ---: | ---: | ---: | ---: |
| 8×12×12 | 50 | **16.4** | 22.6 | 65 / 91 |
| 12×20×12 | — | **55** | 64 | 222 / 265 |
| S, 32×44×20 | — | 980 | **750** | 3,990 / 3,090 |
| M, 48×64×24 | **420** | — | 2,540–2,610 | — |
| L, 64×80×32 | 680 | not run | not run | — |

The node times at 12×20×12 and 32×44×20 are from reruns on 2026-10-17; those of the first sweep differ by up to 8%. The two values at 48×64×24 are from two runs (20 and 40 Newton steps, 200 and 400 gradient steps). XL was not run. v6e at 48×64×24 has no result: the setup fit in 32 GB per chip, but Spot preemptions stopped all 7 attempts.

* **8×12×12:** a v6e node is 5.4× faster than the H200 on Newton and about 10× cheaper (gradient: 3.0× faster, 5.5× cheaper). One v6e chip alone is 1.37× faster than the H200 on Newton.
* **v6e against v5p:** up to 12×20×12, v6e is as fast or up to 1.4× faster, and 2.4–3.2× cheaper. At 32×44×20, v5p is 1.3–1.4× faster and v6e 1.6–1.8× cheaper.
* **48×64×24:** the H200 is 5.1–5.6× faster on Newton and 6× on gradient, and 6–8× cheaper. From 8×12×12 to 48×64×24 (64× the cells), the H200's Newton step grows 1.7× and a TPU node's 42–46× (the H200 figure at 48×64×24 predates #35).
* **Four chips:** `shard_map` gives 3.9–4.1× the throughput of one chip. `jit(vmap)` reached only 1.6–1.8× on Newton: its batched `while_loop` keeps every chip stepping until the slowest equilibrium's Krylov solve converges.
* **Accuracy:** at 48×64×24 on v5p, float32 + double-single matches the H200's float64 energy drop within 0.09% after 200 gradient steps (1.149e-6 against 1.150e-6), and reaches 96.7% of it after 40 Newton steps (1.614e-6 against 1.669e-6), with best residuals of 3.9e-8 to 7.6e-8 on all four equilibria. Across hardware, compare converged states: after 20 Newton steps at 12×20×12, v5p and v6e agree on the residual but differ by 0.17% in the energy removed.

## Where the time goes at 48×64×24

Profile of one v5p chip (a one-off script, not in the repository). It is the op-level profile that [`research/open.md`](research/open.md) asks for, and it points at per-iteration gathers rather than solve depth (compare [`research/performance.md`](research/performance.md), section 8):

* **Not the matrix unit:** matmul precision `highest`, `high` or `bfloat16` changes the operator times by 1–2%, unlike on v5e (and `high` breaks `newton_direction`). The p = 2 sum-factorised contractions (`sumfact_apply`) have stencils 3 to 5 wide, too narrow for the 128×128 matrix unit, so they run on the vector unit.
* **The `E(k)` gather** (`Rows`, `seq.E(1) @ x`) is 52% of an `odd.M[1]` apply in float32 and 69% in double-single (13.4 of 19.3 ms).
* **`MassChebyshev`** (degree 36) costs 216 ms per apply, and one Newton MINRES iteration about 0.48 s (200 iterations: 96.6 s). The MINRES solve hits its 200-iteration cap at the start field, on the H200 too, and on every step of the 12×20×12 runs.
* **CG:** one iteration of the `even.L[1]` solve takes 7.5 ms on a TPU chip against 0.40 ms on the H200 (approximate: different runs).
* **Velocity smoothing:** the `even.shifted(1, mu)` solve of a gradient step did not converge in the profile and took about 60% of the step; `main` has since rewritten these solves (#35), so this needs a recheck.

Ideas: apply `E(k)` as an identity slice plus a small dense block instead of a gather; a tensor-product preconditioner inside `MassChebyshev` (degree 36 down to perhaps 10–20); a stronger MINRES preconditioner for Newton.

## The `two_prod` barrier

* **Without it,** XLA:TPU (libtpu 0.0.48.1) aborts while compiling the solver of the current `f32-device` at 12×20×12, on v5p and v6e, in `relax.py` and in the benchmark alike (`Cannot find a valid window for %subtract_add_fusion`). 32×44×20 compiles. The first sweep ran an earlier `f32-device` commit, which compiled at 12×20×12 without it.
* **With it** (one `jax.lax.optimization_barrier` in `two_prod`, `mrx/double_single.py`), every job compiles and runs. Where both versions compile, the results are bit-identical (48×64×24, Newton, step 20).
* **On arm64 CPUs,** without it XLA:CPU fuses the product into the error term: `test_kernels_against_float64_on_the_host` fails (1.1e-10 against a 1e-13 tolerance), and `test_sum_factorised_apply_against_float64_on_the_host` reaches only 8.6e-8.

## Several runs on one node (`RUNS`)

* **v6e:** three 12×20×12 runs and one 8×12×12 run in one job. Each process opened only its own chip, and the 12×20×12 runs are bit-identical to the run alone, at the same 6.13 s per Newton step. The job held its node 17 minutes, against 11 for one run alone: 2.6× less per equilibrium.
* **v5p:** four identical 8×12×12 runs, one chip each, bit-identical to one another, at 2.56 s per Newton step (2.63 s alone).

## Preemptions and capacity

* **Restart:** a preempted `relax.py` run resumes from its newest checkpoint (tested by deleting the pod mid-run: both runs resumed and finished). It is exact up to the restart, then close but not bit-identical: at 12×20×12 the residual differs by 5e-7 (relative) one step later and by 0.13% ten steps later, and the energy by 5e-9 (relative), which is 0.05% of the energy removed.
* **`relax.json`:** `relax.py` rewrites it in place after every chunk, and a preemption in the middle of a rewrite left it empty (the checkpoints were intact). Writing a temporary file and renaming it would avoid this.
* **Spot capacity** varies by region and hour: on 2026-10-17, `us-central1` had no Spot v6e VMs for over 28 minutes while `us-east1` had them. Hence `us-east1` first and `us-central1` as the backup.
* **Zones:** in `us-east1`, zone b lists the v6e machine type but rejects the nodes, so a job waited 4–14 minutes for its node, instead of about 3, until zone d took it. `make setup` now gives each TPU pool only the zones where the Cloud TPU API offers its TPU.
* **Both families in both regions:** GKE requires the same TPU topology in all priorities of a ComputeClass, so a class cannot fall back from v6e to v5p. Instead each cluster has a pool for every enabled family its region offers, and a job waits in all of them and runs in the first to get a node. On 2026-10-18 a test job waited in four pools: v5p in `us-east1-d` and v6e in `us-east5-c` had no capacity, v5p in `us-east5-a` got a node about 5 minutes after the submission, and its results matched earlier v5p runs bit for bit.

## Open items

* `relax.py`: `--geometry.precision` defaults to float64 and overrides `MRX_DTYPE` (the jobs pass float32); write `relax.json` atomically; make restarts bit-identical.
* H200 timings at 12×20×12 and 32×44×20 would locate the crossover; TPU runs at L and XL after the kernel work.
* TPU7x (Ironwood): 192 GB and 7.4 TB/s per chip. JAX sees each chip as two 96 GB devices, so a node runs 8 equilibria; `RUNS` would need an update.
