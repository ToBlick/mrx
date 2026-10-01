# Performance: measurements

All GPU numbers on one H100 unless stated. "s/step" = steady per-step rate, diagnostics excluded,
compile excluded unless stated. The solver core got 1.4-1.9x faster on 2026-09-10 (CG carry kept in
the requested precision, refinement). Any rate dated earlier is stale by about that factor.

## 1. Relaxation step rates (released core e680ab4, li383 p=2 mixed, 2026-09-11)

Paper-arm rate -> rate on e680ab4:

| arm | before | e680ab4 |
|---|---|---|
| descent (12,24,24) | 0.330 | 0.229 |
| descent (16,32,32), smooth first | 0.643 | 0.411 |
| descent (24,48,48) | 1.96 | 1.29 |
| descent (32,64,64) | 5.45 | 2.98 |
| descent (16,32,32) p=1 / 3 / 4 / 5 | 0.215 / 1.23 / 2.56 / 4.94 | 0.197 / 0.814 / 1.57 / 3.70 |
| descent tol 1e-6 / 1e-10 | - | 0.193 / 0.486 |
| descent float64 order 1 / order 0 | - | 0.669 / 0.410 |
| descent plain float32 order 1 / order 0 | - | 0.125 / 0.062 |
| Newton (12,24,24), Laplacian atom, 300 MINRES | 10.7 | 8.0 |
| Newton (16,32,32) | 16.4 | 11.1 |
| Newton (24,48,48) | 36.5 | 23.6 |
| Newton (32,64,64) | 76.8 | 42.3 |

- The speedup equals the solve share of a step (1.4x at n = 12 to 1.8x at n = 32).
- Potential route with smoothing (m = 1, order 1) 0.303 vs Leray 0.411 s/step: 26-29% cheaper with
  smoothing, 1-10% without.
- Scaling: descent ~n^2.86 (n^2.65 over n = 12..32), Newton ~n^1.93. ~(p+1)^2.6. 0.8-1.5 us per
  quadrature point.
- Smoothing-constant cost: 0.21 (c = 0) -> 0.30 (c = 0.02) -> 0.63 (c = 0.64) s/step.
- m = 0 gradient descent (2026-09-17, li383 (16,32,32) p=2 mixed): 0.30 s/step with smoothing c = 0.02
  (Leray route 0.42, gamma = 0 0.26).
- Paper convention: s/step excludes diagnostics and includes the first-chunk compile (1-4% for
  Newton, < 1% for descent).

## 2. Newton step cost (adopted configuration: Newton-MR, parallel penalty 3 x strain, 200 MINRES)

- li383 (16,32,32) (2026-09-17/18): 100 it 4.6 s/step, 200 it 8.0-8.9 s/step. W7-X (16,32,32) 200 it
  5.8 s/step. (32,64,64) 13-16.6 s/step.
- One pass of 200 vs three passes of 100 (same residual, forcing 0.1 met after ~275 it): 8.9 vs 13.2
  s/step in one run. 8.0 vs 6.5 (li383) and 5.8 vs 7.4 (W7-X) in another.
- Hessian action = 3 k=1 mass solves + 4 cross-product loads, no k=2 or Leray solve: 0.12-0.13 s per
  MINRES iteration float64, 0.06-0.12 s mixed at (16,32,32). A descent step there 0.69 s mixed /
  1.44 s float64 (2026-09-06). Building the harmonic atom from B each step: one quadrature
  evaluation, negligible.
- Harmonic atom kappa = 3 at 100 MINRES (2026-09-13, superseded config): (12,24,24) 2.8, (16,32,32)
  4.05, (24,48,48) 8.8, (32,64,64) 16.7, (48,96,96) 53-61 s/step. First-chunk compile 104 s at
  (16,32,32), 30 min at (48,96,96). Setup 2-2.5 min at n = 16. Wall to F2 1e-8: 2.9 / 8.6 / 26 / 126
  min for n = 16 / 24 / 32 / 48 (steps to the floor ~n, cost/step ~DoFs).
- Harmonic atom kappa = 3 at 100 MINRES 5.0 s/step vs Laplacian atom 12.5 s/step (2026-09-11).
- A Newton step makes ~45k preconditioner applies (2026-09-17).

## 3. Symmetry models and the pure chunk runner (li383 p=2, refined float32, 2026-09-18)

Steady s/step on the 2nd chunk of 20 steps. Setup / compile / s/step:

| model | mesh | DoFs | setup (s) | compile (s) | s/step |
|---|---|---|---|---|---|
| whole torus | (32,64,192) | 1,094,016 | 197 | 29 | 5.45 |
| field period | (32,64,64) | 364,672 | 107 | 26 | 2.05 |
| half period | (32,64,64) | - | 229 | 52 | 1.47 |
| whole torus | (48,96,288) | 3,788,352 | 548 | 34 | 22.9 |
| field period | (48,96,96) | - | 168 | 27 | 8.20 |
| half period | (48,96,96) | - | 293 | 45 | 4.78 |

- Period vs torus: 2.7x per step for 3x fewer DoFs. Half vs period 1.39x (n = 32), 1.72x (n = 48).
  Energies agree to 1e-7..3e-7 after 20 float32 steps.
- At (16,32,32) half period gains nothing (launch-bound): 1.40 vs 1.30 s/step descent, Newton 21.77
  vs 21.70 (2026-09-17). Newton field period 12.3 vs half period 13.2 s/step under the same settings
  (2026-09-18).
- Parity projector as four float64 COO applies: 28.6 vs 21.7 s/step. As one gather 21.77. A host-side
  exact parity projection called 10k times at 40 ms was 409 s of a 541 s build. On device 70 vs 71 s.
- Closure-captured constants vs pure runner: captured constants 3.5 GB (n = 48 period) / 10.6 GB
  (n = 48 torus) -> kilobytes. XLA constant-folded scatters over them ~10 s each. Torus n = 48 host
  peak 309 GB (OOM at 128 GB) -> 10.2 GB. 20-step chunk incl. compile 607 -> 191 s (period),
  1333 -> 490 s (torus).
- Analytic QA vacuum, p=2, total torus / period / half: 34x68x(68|34) splines 1788 / 1042 / 766 s,
  scalar-potential solve 289 / 188 / 118 s, identical errors. 26x52x(52|26) 846 / 534 / 479 s. The
  gain is in the scalar-potential solve.

## 4. Setup, compile, chunking, caches

- Operators + nullspaces: 90 s (12,24,12) p=3, 152 s (16,32,32) p=2, 235 s (32,64,64) p=2 (2026-09-04).
- Chunk boundary 2-4 s at (32,64,64) vs 420 s of stepping per chunk of 10. One-time diagnostics probe
  compile 56 s (16,32,32) / 195 s (32,64,64). Loop speed independent of the chunk (one lax.scan)
  (2026-09-11). Chunked vs per-step driver: 0.557 / 0.563 / 0.558 s/step (2026-09-03).
- One QoI sample (helicity, force, weak pressure, beta) 3.4 s vs 58 s per 100 steps at (16,32,32) p=2.
  QoI + checkpoint every chunk ~6% at n = 16 (2026-09-03).
- Persistent XLA cache: test suite cold 610 / 514 / 525 s vs warm 378 / 345 / 324 s (~38%,
  2026-09-21). Without it eager lax.while_loop solves recompile every call (~10 s compile for ~20 ms
  of arithmetic per k=1 Laplacian apply) (2026-09-04).
- Helicity correction: 3% (mixed) / 5% (float64) of a descent step, 1% on the production route
  (0.300 -> 0.303 s/step), nothing on a Newton step (2026-09-11).
- compute_nullspaces (old code, W7-X, 2026-08-22): linear in dof, ~0.028 s/dof (92.6 / 228.9 / 547.1 /
  1193.6 s at n = 8/12/16/20). CG ~1.23e-6 s/it/dof. n_dof(k=1) = 43300 (n/20)^3 at (n,2n,n).
- Map from series coefficients (2026-08-28): sampled fit 2.4-3.3 s, closed-form interpolant
  0.03-0.12 s, L2 projection 0.6-2.3 s. The second de Rham sequence it used to build cost 16-27 s.
- Lambda solve per flux surface (2026-08-25): axisymmetric mpol 10 207 ms/surface. nfp 3 mpol 8
  ntor 6 (220 coefficients) 3.06 s/surface. 608 coefficients 3.38 s (+10% for 17-21x the asymptotic
  work: neither assembly nor the dense solve dominates).

## 5. Large meshes and memory

- (64,128,128) p=2 (2026-09-06, pre-pure-runner): 3.03e6 k=2 DoFs, operators + nullspaces 539 s, first
  chunk 1333 s (compile), 61 s/step (11x the n = 32 rate for 8x the cells), 320 GB host requested
  (n = 32 used 40 GB RSS). The Greville histopolation of the IC asked 17.4 GiB on GPU unbatched ->
  map batch 8192. n = 48 compile can sit 80 min on a shared node (host-side single-threaded constant
  folding, removed by the pure runner). float64 n = 48 28 s/step, compile 12 min.
- JAX caps the H100 at 59.3 GiB of 80 (mem fraction 0.75). Preallocate = false does not lift it.
  (33,64,32) p=4 OOMed on a ~10 GiB coefficient-window gather in the map evaluation.
  MRX_MAP_BATCH_SIZE_INNER = 262144 fixes it (2026-08-29).
- Vacuum (harmonic form) solve, float64 tol 1e-10 (2026-09-11): 39x78x39 p=2 (335k DoFs) build 103 s,
  harmonic form 40 / 19 s (first / second call), force 14 / 10 s, Rayleigh 3.2e-24, J/B 1.8e-12.
  32x64x32 p=3 (182k) 81 s, 38 / 19, 15 / 11, J/B 3.9e-12. 41x82x41 p=4 (390k) 127 s, 95 / 68, 34 / 30,
  J/B 2.3e-11 (p=4 needs map batch 8192, 14 GiB unbatched).
- QA vacuum rungs, float64 (2026-08-29): (8,16,8)..(32,64,32) p=3 132 / 164 / 297 / 580 / 1044 s.
  (49,96,48) p=4 112 min (59 min in the k=1 Hodge solve).

## 6. Solver-level costs (old code, 2026-08-17 .. 22, ratios still valid)

- Per CG iteration of the Laplacian, k=1 free: 40.9 ms rot-ell n=12, 36.2 ms n=20, 39.0 ms W7-X n=12,
  59.7 ms n=20. The atom apply is 0.09-0.20 ms (< 0.2%).
- Build at n = 20: scalar-term atom 2.6-2.7 s, fm3 6.4 s, Jacobi 7.8 s, o1 ~60 s, o2 ~120 s. r3
  28-38 s, r3o2 55-83 s, r3o4 96-125 s vs Jacobi 3-6.5 s (12^3-16^3).
- Storage (rot-ell n=12 k=1, 8700 dof): polar dense block 29 KB. o1 core block 34.5 MB (O(n^{4/3})).
  extra_rings 3 core 11.8 MB (~1215 rows). fm3 basis + image 20.6 MB (O(n q)).
- The old dense surgery coupling was O(N n_z): ~23 MB at 12^3, ~1.5 GB at 32x64x32, ~24 GB at
  64x128x64, vs (C C^T)^-1 27 KB. The old probed Jacobi diagonal 541 s and Schur-diagonal probe
  2236 s of a 2957 s setup (O(N^2), unbatched) at 12x24x12.
- Harmonic-form setup on the W7-X 32^3 h5 geometry (route deleted, 2026-08-17): set_map 67 / 243 /
  607 s (n = 8/12/16, ~n^3.2), Schur-Jacobi assembly 28.6 / 174 / 873 s (~n^5). n = 28 took 8h14m.
- Micro (2026-08, unexplained): (16,32,16) k=3 metric-lumping Laplacian apply 47 -> 60 us after the
  one-write apply (k = 1, 2 -15%). k=1 mass apply 61 -> 68 us (8,16,8) after the static selector. One
  dispatch ~7 us.
- Gradient descent (gamma = 0) line-search dt scales ~h^2 (0.022 / 0.0028 / 0.0012 at (8,16,8) /
  (12,24,12) / (16,32,32), W7-X): equal relaxation time costs ~h^-5 (2026-08-26).

## 7. Field-line tracing

- One Poincare field ~24 s at (16,32,32) after ~150 s of setup (2026-09-04). 100 seeds x 400 periods x
  8 planes, two fields: ~10 min, ~6 min of it setup. A small trace's 300 s was almost all JIT
  compile (2026-08-25).
- Island search li383 (16,32,32): section 61 s, all-chain search 116 s incl. compile (2026-09-18).
  The 0-form weak pressure in the tracer costs ~2 min per field (2026-09-09).
- Step schedule (49 seeds x 20 periods, compile incl., 2026-08-24): prescribed/vmap vs adaptive/vmap
  vs adaptive chunk 8: W7-X k2 9.9 / 12.8 / 22.6 s, k1 26.0 / 43.0 / 66.9 s. quasr44970 k1 22.4 vs
  215 s (one seed's step collapse drags the vmapped batch 9.6x, chunking is worse, each chunk pays its
  worst seed).

## 8. TPU v5e vs CPU vs H200 (li383, float32, matmul highest, JAX 0.11.1, 2026-09-03..05)

- Setup (12,24,12) p=3, v5e before -> after / VM CPU / H200: build_sequence (warm cache) 203 -> 38.8 s
  / 42.9 / 61.6. compute_nullspaces 222 -> 7.53 s / 11.5 / 19.2. Mass core apply k1 6.60 -> 0.420 ms /
  3.79. E apply k1 1.999 -> 0.182 ms / 0.100. k=1 Laplacian apply (nested CG) 10020 -> 85.7 ms / 102.
  At (12,24,24): build 36.8 / 36.0 / 65.1 s, nullspaces 34.5 / 10.9 / 19.4 s.
- Fresh-node setup 143 s without cache, 98 s warm GCS bucket, 53 s warm local disk.
- Index-tensor gather/scatter in the sum-factorised mass kernel (12,24,12) p=3: v5e 1.624 / 2.011 ms
  vs structured rolled-slice shifts 0.049 / 0.060 ms (CPU 0.070 / 0.398 vs 0.113 / 0.303): on TPU look
  for an index tensor first. Folding y,z sum-factorisation stages into one contraction (1.5x FLOPs):
  1.48-1.70x v5e, 1.23-1.49x H200, 1.62x CPU. Folding all three axes loses everywhere (4.8x FLOPs).
- Per apply in a jitted scan of 50, ms, (12,24,12) v5e / H200 f32 / CPU: mass k1 0.716 / 0.090 / 1.417.
  Mass k2 0.659 / 0.082 / 1.369. Stiffness k1 0.697 / 0.108 / 1.472, k2 0.390 / 0.047 / 0.561. D^T D k1
  1.354 / 0.170 / 2.832. Mass atom k2 0.065 / 0.036 / 0.030. Laplacian atom k2 0.072 / 0.086 / 0.044.
  H200 f64 mass k1 0.102. H200 wins every matvec (12.2x on mass k1). E / E^T are 7-11x cheaper on CPU
  than on either accelerator (pure data movement).
- Eager microbenchmarks overstate the scan form 1.0-1.6x (CPU), 1.3-6.8x (v5e), 5.9-66x (H200).
- Relaxation step (12,24,24) p=3 float32, tol 1e-6, smoothing 0.064/n_r^2: inverse-mass CG 95/98
  (k1/k2) on both. v5e 2.862 s/step vs H200 0.298 (9.6x, 14.8x on a 10-step call). Compile 81.4 vs
  61.0 s. One inverse-mass CG iteration 4.73 vs 2.86 ms (1.7x), mass core k1 0.800 vs 0.114 ms (7x):
  the gap grows with nesting depth. H200 at tol 3.5e-4: 0.352 s/step (H100 ~0.5 corroborates).
- Step cost rises along the trajectory when float32 cannot reach the tolerance (v5e tol 1e-6 steps
  6-10 vs 1-5: 6.270 vs 2.862 s, 2.19x). A 5-vs-10-step slope estimator is invalid when steps differ.
- As configured then: H200 refined float32 tol 1e-8 0.410 s/step (refinement 38%). v5e plain float32
  tol 3.5e-4 1.130 s/step.
- v5litepod-1 vs -4: 1.7998 vs 1.7999 s/step (single-device solve). pmap of a 4-member batch of
  initial states on 4 chips: 3.99x, members match sequential to 5.0e-5.
- Refuted TPU ideas: indices_are_sorted scatter 0.615 vs 0.533 ms (slower). Dense extraction matmul
  0.408 vs 0.533 ms but 303 MB resident. Narrow contractions cost flat K = 4..8 (not charged as 128).
  Extraction operator irrelevant (E / E^T ~700 of ~28700 applies per step).
- The early 13 s TPU step was apply count, not apply cost: the velocity-smoothing MINRES ran at
  thousands of iterations on every backend (led to the shifted split). The 17.03 s/step and 5.1x
  figures (compile / steps, unmatched tolerance) are withdrawn.
