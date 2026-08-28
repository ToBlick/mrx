> **Status:** current
> **Read this for:** why CG is the default descent method and what L-BFGS memory buys (nothing measurable)
> **Do not read for:** the time-step rule (see docs/source/concepts/relaxation.md) or the resistive step

# Descent method: CG vs L-BFGS

W7-X fmm002, `(8,16,8)`, `p=3`, Clebsch initial condition, analytic line search
with the CFL clip (C = 0.5), `--tol 1e-12`, 1000 steps, one H100 per arm.
L-BFGS stores `M s`, `M y` with the histories, so a step applies `M_2` three
times for every `m`.

| arm | E(100) | E(250) | E(1000) | \|F\|(1000) | s/step |
|---|---|---|---|---|---|
| CG | 4.998837137e-1 | 4.998792714e-1 | 4.998779714e-1 | 1.45e-4 | 0.464 |
| L-BFGS m=1 | 4.998836447e-1 | 4.998792302e-1 | 4.998779701e-1 | 1.34e-4 | 0.467 |
| L-BFGS m=3 | 4.998836930e-1 | 4.998792284e-1 | 4.998779711e-1 | 1.72e-4 | 0.467 |
| L-BFGS m=5 | 4.998836425e-1 | 4.998792518e-1 | 4.998779697e-1 | 1.25e-4 | 0.468 |
| L-BFGS m=10 | 4.998835527e-1 | 4.998791921e-1 | 4.998779640e-1 | 1.41e-4 | 0.470 |

The five arms are one trajectory to within its noise: the energies agree to
2e-8 at step 1000 and the force norms scatter without an ordering in `m`.
With an exact line search, memoryless BFGS (`m=1`) reproduces the
Polak-Ribière direction, and older curvature pairs add nothing because the
force is the gradient of a constrained variation, `delta B = curl(u x B)`,
whose metric changes with `B`.

Verdict: CG is the default. L-BFGS remains available (`--method lbfgs
--history m`) and its cost no longer grows with `m`; do not expect it to
converge faster on this problem.

## 2026-08-28: CG arm removed

W7-X wout (12,24,12) p=3, float32, 1000 steps, identical settings
(`static-dynamic-refactor` c25657c): CG |F| 2.593e-3 -> 1.911e-4 without
reaching the 1e-3 residual floor by step 1000, E(1000) = 4.99998927e-1,
0.36 s/step; L-BFGS m=1 floored at step 704, |F| -> 1.495e-4,
E(704) = 4.99998868e-1, 0.39 s/step (the extra is the per-20-step diagnostic
print of that branch). Analytically, with the exact line search
``<s, g_k>_M = 0`` kills every ``<s, g>`` term of the memoryless BFGS update
and leaves ``u = F + beta u_{k-1}`` with the Hestenes-Stiefel beta, which
equals Polak-Ribiere under the same orthogonality. The CG arm was deleted;
`DescentMethod = {GRADIENT, LBFGS}`, default L-BFGS with `history_size = 1`,
and a pair with ``<s, y>_M <= 0`` is skipped (rho = 0), the PR+ restart.
