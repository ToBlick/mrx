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
