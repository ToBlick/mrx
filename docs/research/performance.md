# Performance: measurements

Timings with the counts measured alongside: quadrature, Newton step cost, per-apply profiles, symmetry models,
setup, memory, tracing and TPU. One H200 unless stated. relax.py prints s/step over the whole run, which includes
the compile of the first chunk. A steady rate is over a later chunk and says so.

## 1. Quadrature (p + 1, p, p) by default (li383 graded p = 2, multigrid, odd view, 2026-10-18)

p Gauss points per span in theta and zeta against p + 1, both in one job. Record:
`outputs/multipatch/graded/quad/NOTES.md`.

- Grid: M (48,64,24) 954k -> 424k points, L (64,80,32) 2.14M -> 952k.
- Applies of M_k and S_k are 1.5-1.7x faster on M and 1.5-2.1x on L (M_1 0.351 -> 0.172 ms on L). The atoms do not
  change.
- Solves, counts within 1 %: M_1 -26 / -37 %, S_1 -18 / -27 %, L_1 -21 / -32 %, L_3 split -19 / -26 % on M / L. The
  11 solves of table_bench -21 % on M and -29 % on L.
- Newton on M, MINRES capped at 200: 4.0 -> 2.6 s/step over the second chunk of 5 steps.
- Device memory after the build: M 1.78 -> 1.28 GB, L 4.18 -> 3.06 GB. Build time unchanged.
- Accuracy: test_poisson errors measured with p + 1 points in every direction move by +0.02-0.2 %. Harmonic
  Rayleigh quotients stay at round-off. P^-1 M_1 spectrum [0.068, 2.38] -> [0.067, 2.39]. 10 Newton steps: energy
  decrease +0.2 %, ||F|| +0.3-1.1 %, helicity drift 1.7e-8 -> 6.1e-8 relative.
- Measured with the reduced rule itself the test_poisson errors read 1.8-3.5x low. Its bands were re-measured on
  (12,20,12): 5.051e-3, 7.341e-3, 1.461e-2, 2.461e-2 for k = 0..3. Suite 13:52 against 14:37.

## 2. Newton step cost

Gauss-Newton by reorthogonalised MINRES, M_1 solves to 1e-6 (li383 graded (48,64,24), p = 2, H200, 2026-10-18,
study driver on gp-review before #38):
- One MINRES iteration 11.8 ms, of which the two M_1 solves (slab, 1e-6, about 11 its) 11.4 ms. The block
  Gauss-Newton system under flexible GCR needs 12.7 ms per iteration (two M_1 solves at 1e-4, outer norm,
  orthogonalisation) for the same iteration counts.
- Stored Lanczos vectors 2 x 300 of the 1-form space, 0.4 GB, against 1.8 GB for the 3 x 300 block vectors of GCR.
- Steady s/step over 40 steps from the file's field: 1.36 (float32 slab factors), GCR 1.58, production Newton-MR
  2.62.

Replaced 2026-10-18, Newton-MR on the full Hessian with MassChebyshev, a fixed Chebyshev polynomial for M_1^-1
(s/step without diagnostics, 2026-10-05):
- li383 (12,16,16): 3.23 with refined M_1 solves -> 0.83 with the polynomial (mass-tol 1e-3, degree 23) -> 0.44 with
  batched applies. W7-X FMM002 (16,32,32) 4.96 -> 0.83 (degree 20). Same floors and energy.
- The Hessian error was 3-4x that of the polynomial. mass-tol 1e-2 (degree 17) stalled li383 at 1e-6, at
  (32,64,64) 1e-3 stalled at 2e-6, 1e-4 reached 1e-9 like exact solves (2026-10-08). Wall time at (32,64,64): 118 s
  at 1e-3, 186 s at 1e-4, 260 s with exact solves.

## 3. Kernel-level profile (XProf, H100, 2026-10-05)

- Before the Chebyshev mass inverse a li383 (12,16,16) Newton step was 2.1M kernels of about 2 us, almost all in the
  refined M_1 solves. XLA command buffers for while and cond gained 7.5 % only.
- Per apply after batching, ms, M_1 / mass atom / Hessian: li383 12^3 0.034 / 0.024 / 3.0, W7-X 16^3 0.067 / 0.034
  / 3.6, 24^3 0.093 / 0.041 / 6.3, 32^3 0.19 / 0.03 / 10.7. Up to 16^3 the apply is bound by its kernel count. At
  32^3 the mass apply dominates: weight contraction 36 us, six skinny GEMMs 77 us, XLA transposes 40 us.
- Tried at 32^3 and dropped: a rotating layout with six unique weights 0.96x, E folded into the gathers 1.0x,
  Pallas element kernels 1.15-1.3x at best (2-3.7x at 12-16^3), CUDA FFI kernels slower with 5M atomic adds per
  apply. Records in `outputs/matvec_profile/README.md`.
- eqx.filter_jit of a function of the sequence costs about 10 ms of host dispatch per call. Batched axis probes cut
  build_preconditioners 13.5 -> 2.1 s per geometry.

## 4. Symmetry models (li383 p = 2, mixed, H100, 2026-09-18)

Setup / compile / steady s/step over the second chunk of 20 steps:

| model | mesh | DoFs | setup (s) | compile (s) | s/step |
|---|---|---|---|---|---|
| whole torus | (32,64,192) | 1,094,016 | 197 | 29 | 5.45 |
| field period | (32,64,64) | 364,672 | 107 | 26 | 2.05 |
| half period | (32,64,64) | - | 229 | 52 | 1.47 |
| whole torus | (48,96,288) | 3,788,352 | 548 | 34 | 22.9 |
| field period | (48,96,96) | - | 168 | 27 | 8.20 |
| half period | (48,96,96) | - | 293 | 45 | 4.78 |

- Period against torus: 2.7x per step for 3x fewer DoFs. Half against period: 1.39x at n = 32, 1.72x at n = 48.
  Energies agree to 1e-7..3e-7 after 20 steps.
- At (16,32,32) the half period gains nothing because the step is launch-bound: 1.40 against 1.30 s/step descent.
- Closure-captured constants, replaced by the pure chunk runner: 3.5 GB (n = 48 period) and 10.6 GB (n = 48 torus)
  of captured constants -> kilobytes. Host peak of the n = 48 torus 309 GB -> 10.2 GB. A 20-step chunk with compile
  607 -> 191 s (period) and 1333 -> 490 s (torus).
- Analytic QA vacuum, p = 2, 34x68x(68|34) splines, torus / period / half: 1788 / 1042 / 766 s in all, of which the
  scalar-potential solve 289 / 188 / 118 s. Errors identical.

## 5. Setup and memory

- Build of operators and preconditioners on li383 M, L and W7-X (32,64,36): 74-142 s (2026-10-17). The slab mass
  adds 32 s on M, the multigrid 26-33 s, the strips 43 s on L.
- JAX caps the H100 at 59.3 GiB of 80 (mem fraction 0.75). (33,64,32) p = 4 ran out of memory on a 10 GiB gather in
  the map evaluation. MAP_BATCH_SIZE_INNER = 262144 fixes it (2026-08-29).
- Persistent XLA cache: test suite cold 610 / 514 / 525 s against warm 378 / 345 / 324 s, about 38 % (2026-09-21).
- Map from series coefficients (2026-08-28): sampled fit 2.4-3.3 s, closed-form interpolant 0.03-0.12 s, L2
  projection 0.6-2.3 s.
- Removed setup routes: the Schur-Jacobi assembly of the old harmonic-form route grew as n^5 (W7-X n = 28 took 8h14m,
  2026-08-17). The old dense surgery coupling was O(N n_z), about 24 GB at 64x128x64 against 27 KB for (C C^T)^-1.
  A probed Jacobi diagonal took 541 s of a 2957 s setup at 12x24x12.

## 6. Field-line tracing (2026-08-24)

- Step schedule, 49 seeds x 20 periods, compile included. Prescribed steps under vmap / adaptive under vmap /
  adaptive in chunks of 8: W7-X k = 2 9.9 / 12.8 / 22.6 s, k = 1 26.0 / 43.0 / 66.9 s. quasr44970 k = 1 22.4
  against 215 s: one seed's step collapse drags the vmapped batch 9.6x, and chunking is worse.

## 7. TPU v5e against H200 and CPU (li383 float32, JAX 0.11.1, 2026-09-03 .. 05)

- Relaxation step (12,24,24) p = 3, tol 1e-6: v5e 2.862 s/step against H200 0.298 s/step, 9.6x. Compile 81.4 against
  61.0 s. One inverse-mass CG iteration 4.73 against 2.86 ms, the k = 1 mass core 0.800 against 0.114 ms. The gap
  grows with nesting depth.
- Per apply in a jitted scan of 50 at (12,24,12), ms, v5e / H200 / CPU: mass k = 1 0.716 / 0.090 / 1.417, stiffness
  k = 1 0.697 / 0.108 / 1.472, Laplacian atom k = 2 0.072 / 0.086 / 0.044. E and E^T are 7-11x cheaper on CPU than on
  either accelerator.
- Setup (12,24,12) p = 3, v5e / CPU / H200: build_sequence 38.8 / 42.9 / 61.6 s, compute_nullspaces 7.53 / 11.5 /
  19.2 s.
- On TPU an index-tensor gather in the mass kernel costs 30x a rolled-slice shift (1.624 against 0.049 ms). Folding
  the y and z sum-factorisation stages into one contraction: 1.48-1.70x on v5e, 1.23-1.49x on H200, 1.62x on CPU.
  Folding all three axes loses everywhere.
- Refuted TPU ideas: an indices_are_sorted scatter is slower (0.615 against 0.533 ms). A dense extraction matmul
  takes 0.408 against 0.533 ms but holds 303 MB.
- pmap of four initial states on four chips: 3.99x.
- Eager microbenchmarks overstate the scan form 1.0-1.6x on CPU, 1.3-6.8x on v5e and 5.9-66x on H200.

## 8. Relaxation benchmark, li383 1.4m, graded n_theta r^(1/3), float64 and float32 (H200, 2026-10-20)

`scripts/relax.py --geometry.path data/wout_li383_1.4m.nc --geometry.resolution <n> --geometry.graded
[--geometry.precision float32] --budget.steps 80`, Newton, from the file's field (beta 4.2 %). Record:
`outputs/benchmarks/relax_h200_li383_2026-10-20.md`.

- (32,44,20), float64 / float32: 1.23 / 1.47 s/step over 80 steps, residual 2.4e-4 -> 2.2e-10 / 2.6e-10, energy
  removed 1.7003e-6 / 1.6959e-6. Newton MINRES mean 82 / 87, unconverged on 15 / 60 of 80 steps. Float32 invariants
  at its round-off: div B 2.6e-6, dE - dE_ls 4e-10, 11 energy increases near the floor. Setup 201 s / 465 s.
- (72,120,72), float64 / float32: 9.23 / 9.23 s/step, setup 620 / 911 s, 23 / 31 min per job. Float64 reaches the
  1e-10 floor at step 70 (7.1e-11), float32 ends at 1.87e-10 after 80. Energy removed 1.689e-6 / 1.690e-6. Newton
  MINRES mean 145 / 147, at its cap of 300 on 54 of 70 / 69 of 80 steps. Float32 div B 1.5e-5.
- The CFL cap binds on every step at both sizes (dt/dt* mean 0.096-0.098). On the low-res reference wout (no
  pressure) it bound only on the first 1-5 steps.
- Float32 needs the SlabMass band checks in machine epsilon: with the fixed 1e-12 the build raised on float32
  round-off of 4.1e-9.
