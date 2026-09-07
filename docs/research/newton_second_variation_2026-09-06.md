# Newton's method on the second variation (the SIESTA idea), 2026-09-06

Branch `newton-second-variation` (worktree `.claude/worktrees/newton`, based on
li383-followups 5d2189f). Budget: 10 GPU h. This note explains the idea from the
energy to the code, then records the experiment. Sections 9 and 10 are filled in
as the jobs finish.

## 1. What the relaxation does today

The relaxation lowers the magnetic energy $E(B) = \tfrac12 \|B\|^2$ by moving $B$
with a divergence-free velocity $u$ through the ideal induction

$$B_{n+1} = B_n + \Delta t \, \mathrm{curl}(u \times B_n),$$

which keeps $\mathrm{div}\,B$ and the helicity exactly (the topology to the time
error). The velocity is the Leray-projected Lorentz force $F = P(J \times B)$,
combined by memoryless BFGS (= Polak-Ribière CG) and smoothed once,
$u = (M + \mu L)^{-1} M F$, and $\Delta t$ is the exact minimiser of the quadratic
energy along the increment, capped by a logical CFL number. The fixed point is
$J \times B = \nabla p$ with $p$ the Leray multiplier.

Measured on li383: the residual decays as a power law in the step, never a
plateau, and the tail is slow (2.3e-4 at step 5000, 1.5e-4 at 10000, 53 min more).
The slow directions are the ones the energy is flat along: surfaces sliding past
each other, current sheets thinning. Gradient descent scales every mode by the
same $\Delta t$; the stiffest modes (small scales) set $\Delta t$ and the flat
modes crawl. Newton scales each mode by the inverse of its own curvature. That is
the idea; the question is whether the curvature can be applied and inverted
cheaply, and whether the flat modes are flat enough that Newton still helps.

## 2. The energy along a flow: first and second variation

Take a divergence-free $u$ and the time-1 flow $\Phi_u = \exp(u)$. The
pushed-forward field solves $\partial_t B_t = \mathrm{curl}(u \times B_t)$, so to
second order

$$\Phi_{u*} B = B + Q + \tfrac12 R, \qquad Q = \mathrm{curl}(u \times B), \quad R = \mathrm{curl}(u \times Q).$$

Define the scalar function $\mathcal E(u) = \tfrac12 \|B + Q + \tfrac12 R\|^2$.
Its expansion in $u$ gives everything:

- **Gradient.** $\mathcal E'(0)[u] = (B, Q) = (B, \mathrm{curl}(u \times B)) = (J, u \times B) = -(u, J \times B)$.
  So the gradient of the energy with respect to the velocity is minus the Lorentz
  force, and on divergence-free $u$ only its Leray-projected part $F$ is seen.
  This is the force the code already computes.
- **Hessian.** The quadratic part of $\mathcal E$ is $\tfrac12\|Q\|^2 + \tfrac12 (B, R)$, so

$$(u, H v) = (Q_u, Q_v) + \tfrac12\big[(B, \mathrm{curl}(u \times Q_v)) + (B, \mathrm{curl}(v \times Q_u))\big].$$

  It is symmetric by construction, because it is the Hessian of a scalar
  function. The draft's iterated variation $\delta(\delta B(u))(v)$ gives only one
  of the two halves of the second term; the two halves differ by
  $\delta E([u, v])$, the first variation along the commutator, which is the
  asymmetry the draft remarks on. Defining $\delta^2 E$ as the Hessian of
  $u \mapsto E(\exp(u)_* B)$ removes the remark. (The draft's derivation ends at
  exactly this symmetrised form, so nothing changes in the Newton step.)

**Sign check against the energy principle.** With $u = v$,
$(u, H u) = \|Q\|^2 + (B, \mathrm{curl}(u \times Q)) = \|Q\|^2 + (J, u \times Q) = \|Q\|^2 - (u, J \times Q)$,
which is $2\,\delta W$ of ideal MHD at $p = 0$ with $\xi = u$ (Bernstein et al.
1958). The draft writes the cross term as $\mathrm{curl}(\delta B(u) \times v) = -\mathrm{curl}(v \times \delta B(u))$,
the opposite sign.

**At an equilibrium** ($J \times B = \nabla p$) the two halves agree and

$$H v = -\big[(\mathrm{curl}\, Q_v) \times B + J \times Q_v\big] = -\mathbf F(v),$$

minus the linearised force operator (Hain-Lüst), the operator SIESTA uses as its
preconditioner. Newton's equation $H u = J \times B$ therefore reads
$\mathbf F(u) = -(J \times B)$: find the displacement whose linearised force cancels
the current imbalance.

**Structure that matters for the solve.**

- *Kernel.* Every $u$ with $\mathrm{curl}(u \times B) = 0$ is in the kernel of the
  quadratic form: $u = B$ (flow along field lines) and, on a surface, every flow
  tangent to it that preserves $B$. Away from equilibrium $H B \neq 0$ but
  $(B, H B) = 0$ still.
- *Consistency.* The right-hand side annihilates the kernel:
  $(J \times B, u) = -(B, \mathrm{curl}(u \times B)) = 0$ whenever the curl vanishes.
  So the singular system is consistent and MINRES can solve it.
- *Definiteness.* $H$ is indefinite away from a minimum and only positive
  semi-definite at one. The flat directions of the tail are the near-kernel of
  $H$: Newton does not make them fast, it makes them no slower than the scale set
  by the regularisation.

## 3. Newton's step under the divergence constraint

Newton minimises the quadratic model $-(u, r) + \tfrac12 (u, H u)$ over
divergence-free $u$, $r = M_2 F$ the force as a covector. The constraint has a
Lagrange multiplier, the pressure, so the system is a saddle problem. Two ways to
avoid the saddle:

- Solve the unconstrained system and Leray-project the result: a different
  direction, and the projection is a k=3 saddle solve.
- Write $u = \mathrm{curl}\, a$ with $a$ a Dirichlet 1-form. In the discrete complex
  $\mathrm{div}\,\mathrm{curl} = 0$ exactly on the incidence matrices, so every such
  $u$ is divergence-free to roundoff, better than the Leray-projected force in
  float32. The system becomes

$$\mathrm{curl}^T (H + \lambda M_2)\, \mathrm{curl}\; a = \mathrm{curl}^T M_2 F,$$

  symmetric, singular on the gauge $a + \nabla\phi$ (which the curl removes from
  the answer) and on the preimages of the kernel of $H$ (consistent, as above).
  The one divergence-free direction it cannot represent is the harmonic 2-form
  of the Dirichlet complex, the net toroidal flux, one degree of freedom.

This is what `mrx.hessian.newton_direction` solves. $\lambda$ is a
Levenberg-Marquardt shift in the velocity's $L^2$ metric: $\lambda = 0$ is Newton,
$\lambda \to \infty$ gives $u = F / \lambda$, steepest descent (the line search
removes the scale), and in between the modes of $H$ with eigenvalue above
$\lambda$ are solved for and the ones below are descended along. A negative
eigenvalue of $H$ larger than $\lambda$ in magnitude makes the direction a
non-descent direction; the step detects that by the sign of $(u, F)_M$ and falls
back to the smoothed force for that step (`State.newton_fallback` counts it).

The right-hand side uses the Leray-projected force although the gradient is the
unprojected $J \times B$: the difference is a gradient $M_2^{-1} D_2^T q$, and
$\mathrm{curl}^T D_2^T = (D_2 \,\mathrm{curl})^T = 0$, so $\mathrm{curl}^T M_2 F = \mathrm{curl}^T \mathrm{load}(J \times B)$ exactly.

## 4. Applying the Hessian, matrix-free

Nothing is assembled. $H$ only ever acts on a vector, through the same loads and
mass solves the force uses. With $L_B u = \mathrm{load}_1(u \times B)$ (the 1-form load
of the cross product, the induction's right-hand side) and $G_1$ the topological
curl,

$$Q = G_1 M_1^{-1} L_B u .$$

Transposes are the same kind of load with the roles swapped, because the
integrand is a triple product: $(L_B u)^T W = \int W \cdot (u \times B) = \int u \cdot (B \times W)$,
so $L_B^T W = \mathrm{load}_2(B \times W)$. Working out the three terms of section 2:

| term | what it is | solve |
|---|---|---|
| $Q^T M_2 Q\, u = \mathrm{load}_2(B \times \delta J)$ | $\delta J = M_1^{-1} G_1^T M_2 Q$, the weak curl of $Q$, the current perturbation | k=1 |
| $\tfrac12 \mathrm{load}_2(Q \times J)$ | one half of the cross term | none |
| $\tfrac12 \mathrm{load}_2(B \times W)$ | $W = M_1^{-1} G_1^T \mathrm{load}_2(J \times u)$, the transpose half | k=1 |

plus the induction solve $E = M_1^{-1} L_B u$ for $Q$ itself. So one Hessian
action is three k=1 mass solves (PCG with the mass atom, refined to the solve
tolerance; the same solve as the weak curl $J$ in the force), four cross-product
loads and a handful of quadrature evaluations. No k=2 mass solve and no Leray
solve: it costs less than a force evaluation, whose k=3 saddle solve is the
expensive part of a step.

The result is a dual 2-form (a covector), which is what MINRES on the potential
system needs: $A a = \mathrm{curl}^T (H\, \mathrm{curl}\, a + \lambda M_2 \mathrm{curl}\, a)$ maps
1-form coefficients to dual 1-forms.

## 5. The solve: MINRES with the Laplacian atom

MINRES handles symmetric indefinite and singular-consistent systems, needs one
matvec per iteration and an SPD preconditioner. The preconditioner has to be a
fixed linear operator, not a solve (the no-Krylov-in-Krylov rule): the
metric-lumping atoms are exactly that.

Why the k=1 **Laplacian** atom: the dominant part of $H$ on velocities is
$\|Q_u\|^2 = \|\mathrm{curl}(u \times B)\|^2$, a curl-curl weighted by $|B|^2$. On
the potential this makes $A$ fourth order and singular on gradients. The
Laplacian atom approximates the inverse of $\mathrm{curl}\,\mathrm{curl} + \mathrm{grad}\,\mathrm{div}$:
it removes two of the four derivative orders and makes the gradient kernel
harmless. The preconditioned operator is second order with the $|B|^2$
anisotropy, so the iteration count should scale like $n$, the same as a
mass-preconditioned solve in velocity space would, without a Leray solve per
iteration. The atom **squared** removes all four orders at the price of a worse
constant; the probe tries both. Neither knows anything about $B$: the anisotropy
along field lines, the near-kernel of $H$, is not preconditioned, and if the
counts come out in the hundreds that is why.

Inexact Newton: the solve stops at a relative residual of 1e-3 (in the
preconditioner norm), capped at `newton_maxiter` iterations, and is warm-started
from the previous step's potential, which changes slowly. The Hessian's inner
mass solves run at the sequence tolerance (1e-8 refined), so the matvec is exact
for the purpose of a 1e-3 outer solve.

## 6. Inside the step

`TimeStepper(newton=True)` replaces only the direction. The rest is unchanged:

- the line search $\Delta t^* = (F, u)_M / \|Q_u\|^2$ minimises the exact quadratic
  energy along the first-order path. For a Newton direction at $\lambda = 0$ the
  model predicts $\Delta t^* \approx 1$; how far it is from 1 measures how good the
  quadratic model is;
- the CFL cap still applies: a Newton direction with a large displacement is
  chopped to half a cell per step, and the next step recomputes the direction.
  If that binds, the efficient variant is substeps along the frozen direction
  (cheap: one k=1 solve each), not implemented in this round;
- helicity and $\mathrm{div}\,B$ are exact for any $u$, the update is still a curl.

## 7. Cost, and what has to be true for Newton to pay

Per step: MINRES iterations times one Hessian action (three k=1 solves), plus
one force evaluation. At (16,32,32) p=2 in mixed precision a descent step costs
0.69 s, a Hessian action should be a fraction of that. If MINRES needs 100
iterations a Newton step costs a few tens of descent steps. It pays only if one
Newton step removes what that many descent steps remove, which in the power-law
tail means: the direction must correlate with the flat modes better than the
smoothed force does. The smoothing $(M + \mu L)^{-1}$ is itself a poor man's
Newton: it is spectrally equivalent to the $|B|^2$ curl-curl part of $H$ but
knows nothing about $B$. What Newton adds is the anisotropy and the $J \times Q$
term.

## 8. The experiment

Job A, `scripts/newton_probe.py`, float64, li383 (16,32,32) p=2, at the initial
field and at the anchor's step-5000 field (residual 2.3e-4):

1. identities: the first derivative against the force's pairing, the quadratic
   form against $\|Q\|^2 + (B, R)$, symmetry, and $(B, H B) = 0$;
2. Lanczos with full reorthogonalisation on $P M_2^{-1} H$ over divergence-free
   velocities in the $M_2$ metric, 150 steps: extreme Ritz values, the number of
   negative ones, the condition number Newton's system has with the exact
   projection;
3. the Newton direction for shifts 0, 1e-3 and 1e-2 of the largest Ritz value and
   both preconditioners, solved to 1e-1, 1e-2, 1e-3 in turn: MINRES counts, the
   descent cosine, $\Delta t^*$ and its CFL cap, the energy removed along the
   first-order path (exact) and the second-order path (evaluated), against the
   force and the smoothed force.

Job B, `scripts/relax.py --newton`, the anchor's settings (mixed precision, CFL
0.5, smoothing 1 as the fallback), restarted from the anchor's step-5000
checkpoint with the shift and preconditioner the probe favours, one hour of
stepping per arm, against the anchor's own continuation (1.46e-4 at step 10000
after 53 min). The comparison is residual against wall time and against force
evaluations.

Acceptance: Newton wins if it reaches the continuation's residual in clearly
less wall time, or reaches a lower one in the same time; a wash or a loss is a
result too, and the spectrum from job A says which of the two mechanisms of
section 7 is to blame.

## 9. Results of job A (probe)

*(pending)*

## 10. Results of job B (Newton arms)

*(pending)*

## 11. Corrections to the paper's second-variation section

- After the $\varepsilon$ derivative the first term is $t\,\delta\mathcal E(B)(u)$, not $(v)$.
- "Since $\mathcal E(B^{\mathrm{eq}}) = \Phi_* B$" should read $B^{\mathrm{eq}} = \Phi_* B$.
- The cross term's sign: $\delta(\delta B(u))(v)$ should be $\mathrm{curl}(v \times \delta B(u))$,
  from $\partial_{tt} B = \mathrm{curl}(u \times \partial_t B)$; then
  $\delta^2 E(u, u) = \|Q\|^2 - (u, J \times Q)$ matches $\delta W$.
- Define $\delta^2 E$ as the Hessian of $v \mapsto E(\exp(v)_* B)$: symmetric by
  definition, equal to the symmetrised form the derivation produces; the
  remark "symmetric at stationary points only" then concerns the iterated form
  and can go.
- The first display carries $(B, p)$ and $\nabla p$: in our formulation the
  pressure is the multiplier and the Hessian is purely magnetic.
- A shift $\epsilon\,\mathrm{Id}$ removes negative eigenvalues only if $\epsilon$
  exceeds their magnitude; what it guarantees is a nonsingular system. And
  $(\mathrm{Id} - \varepsilon\Delta)\,\delta^2 E$ is not symmetric; the symmetric
  regularisation is $\delta^2 E + \epsilon(-\Delta)$, which recovers the smoothed
  descent direction as $\epsilon \to \infty$.

## 12. Files

- `mrx/hessian.py`: `second_variation(seq, B, J)` (the Hessian action),
  `newton_direction(seq, B, J, MF, a_guess, shift, tol, maxiter, precond)`.
- `mrx/relaxation.py`: `TimeStepper.newton*`, `State.a / newton_it / newton_fallback`,
  the trace entries and the summary line.
- `scripts/relax.py --newton --newton-shift --newton-tol --newton-maxiter --newton-precond`.
- `scripts/newton_probe.py`, `test/test_hessian.py`.
- Outputs: `outputs/newton_probe/<arm>/probe.json`, `outputs/newton_relax/<arm>/relax.json`
  (copied to the main checkout's `outputs/newton_second_variation/`).
