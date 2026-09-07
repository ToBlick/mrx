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

## 7b. The potential form for the descent itself (Tobias's question)

Should the descent also write $v = \mathrm{curl}\,a$ and skip the Leray solve?
It can, but it trades one solve for another. The Leray projection of
$f = J \times B$ onto the normal-trace-free divergence-free fields is, by the
Hodge decomposition $f = \nabla q + \mathrm{curl}\,a + h$,

$$P f = \mathrm{curl}\,a + c\,h, \qquad (\mathrm{curl}\,a, \mathrm{curl}\,w) = (f, \mathrm{curl}\,w)\ \ \forall w,\quad c = (f, h) / (h, h),$$

i.e. $\mathrm{curl}\,\mathrm{curl}\,a = \mathrm{curl}\,f$ weakly with $a \times n = 0$, and
$h$ the one harmonic 2-form (the net toroidal flux). Discretely this is exact:
the spline complex is exact, so $\ker D_2 = \mathrm{range}\, G_1 \oplus \mathrm{span}\, h$,
and the $M_2$-orthogonal projection onto it has the normal equations
$S_1 a = G_1^T M_2 f$ and $c$ as above, because $G_1^T M_2 h = 0$ defines the
discrete harmonic form. The right-hand side $G_1^T M_2 f$ is orthogonal to
gradients, so the k=1 Hodge Laplacian solve $L_1 a = G_1^T M_2 f$ returns the
curl-curl solution in the Coulomb gauge: the same call `compute_helicity`
makes for the vector potential of $B$. One k=1 Hodge-split solve replaces the
k=3 saddle solve; no pressure comes out of it.

Do not descend in $a$ with the $M_1$ metric instead: that gives
$v = \mathrm{curl}(\mathrm{curl}_w f)$, the force differentiated twice, an
anti-smoothing that collapses the CFL step.

**Smoothing the potential.** The Hodge Laplacians commute with the curl,
$\Delta_2\,\mathrm{curl} = \mathrm{curl}\,\Delta_1$ (both are $\mathrm{curl}\,\delta\,\mathrm{curl}$),
so

$$(M_2 + \mu L_2)^{-1} M_2\, \mathrm{curl}\, a = \mathrm{curl}\,(M_1 + \mu L_1)^{-1} M_1\, a$$

exactly in the discrete complex (the proof: apply $M_2 G_1 M_1^{-1}$ to the k=1
equation and use $G_1 G_0 = 0$ on the weak half). The smoother maps gradients
to gradients, so the gauge stays harmless, and $h$ passes through unchanged.
The smoothing is therefore the k=1 shifted solve on $a$ followed by the free
curl: the same shifted-split machinery one level down.

**Implemented as `TimeStepper(potential_velocity=True)`** (`--potential-velocity
true`): the force $F = \mathrm{curl}\,a + c\,h$ from the k=1 Hodge solve
(warm-started from the previous potential), the smoothing on $a$, and L-BFGS on
the *smoothed* forces. That last point is an ordering difference to the Leray
route, which combines the unsmoothed forces and smooths the result (and thereby
smooths the already-smooth history vector a second time); the potential order
is the preconditioned-CG one. `test_potential_force_is_the_leray_force` checks
both identities on the fixture. The comparison against the sweep's anchor is in
section 10b; the m = 0 arms isolate the projection route from the ordering,
since steepest descent has no history to order.

**The harmonic direction.** $h$ is one degree of freedom, orthogonal to every
curl. At a fixed point $(J \times B, h) = (\nabla p, h) = -\int p\,\mathrm{div}\,h + \oint p\, h\cdot n = 0$,
so near the floor the force has no component along it anyway; the toroidal
flux of $B$ is conserved by every curl update whatever $u$ is. The descent
route carries $c\,h$ because it is one inner product; the Newton route drops it
and relies on the sign test.

## 7c. One family: the shift, the potential route, and a $B$-aware preconditioner

**The descent is the infinite-shift Newton.** Divide the Newton system by
$\lambda$ and let $\lambda \to \infty$: $S_1 a = \mathrm{curl}^T M_2 F / \lambda$,
whose solution is $\mathrm{curl}\,a = P F / \lambda$ up to the harmonic mode. That
is the potential route's projection solve, scale aside, and the line search
removes the scale. With $\lambda (M_2 + \mu L_2)$ in place of $\lambda M_2$ the
same limit is the smoothed force, by the commutation of 7b. So one system with
one parameter and one metric covers everything: $\lambda = 0$ Newton, $\lambda
= \infty$ in the $L^2$ metric the force, in the $H^1$ metric the smoothed
force, finite $\lambda$ in between. What differs at the two ends is the
solver: at $\lambda = 0$ the operator is symmetric indefinite and singular
(MINRES with the atom), at $\lambda = \infty$ it is the SPD curl-curl modulo
the gauge (the Hodge-split PCG). The finite shifts were the worst members in
the tail (section 9, 10): they keep the flat modes gradient-like while the
line search stretches the step to twice the Newton length for the resolved
ones.

**The Newton step can omit the Leray solve.** The direction never used it.
The right-hand side $\mathrm{curl}^T M_2 F$ equals $\mathrm{curl}^T \mathrm{load}(J \times B)$
exactly, and the line search $(F, u)_M / \|Q_u\|^2$ equals
$(\mathrm{load}(J \times B), u) / \|Q_u\|^2$ because $u$ is divergence-free. The
per-step residual and the fallback direction can come from the potential
route's force. So `potential_velocity` and `newton` together make a step with
no saddle solve anywhere; for the experiment it changes nothing (the Leray
solve is two percent of a Newton step), for the descent it is the 30% of 10b.

**The split identities do not apply; the tensor-atom idea does.** Both splits
(Hodge, shifted) live on $d \circ d = 0$. The Newton operator's core is, to a
percent, the Gauss-Newton form
$\mathrm{curl}^T L_B^T M_1^{-1} S_1 M_1^{-1} L_B\, \mathrm{curl}$ with
$L_B u = \mathrm{load}_1(u \times B)$: the curl-curl stiffness sandwiched by a
$B$-dependent map that respects no exact/co-exact decomposition. The only
split structure left is the gauge, which the potential form already uses
exactly. What transfers is the metric-lumping construction. For
$B = \nabla\psi \times \nabla(\theta - \iota\zeta)$ the map
$u \mapsto \mathrm{curl}(u \times B) = B \cdot \nabla u - u \cdot \nabla B$ acts on a Fourier
mode $(m, n)$ like $\iota(r) m + n$ times the mode, so per radial layer

$$\mathrm{curl}^T H\, \mathrm{curl} \;\approx\; \mathrm{curl}^T\, |B|^2 (\iota m + n)^2\, \mathrm{curl},$$

separable in the angles with the resonance denominators as eigenvalues, and
our logical coordinates are the VMEC flux coordinates, close to straight
field lines. A parallel-derivative atom (lumped $|B|^2$ and $\iota$, exact
Fourier diagonalisation in the angles, dense polar core like the other atoms)
is the $B$-aware preconditioner the Laplacian atom lacks, and the modes it
addresses are the resonant ones, $\iota m + n \approx 0$, the rational
surfaces, which are the near-kernel of the Hessian and the slow directions
of the descent.

**The harmonic version (Tobias 2026-09-07: "B in these states is mostly
harmonic").** `compute_helicity` splits $B = \mathrm{curl}\,A + B_{\mathrm{harm}}$ and
measures $\|B_{\mathrm{harm}}\| / \|B\|$ (0.974 on the quasr44970 initial field;
li383 at this beta is the same kind of field; not tallied per chunk today, one
line in the sampler). With $B \approx c\,h$, $h$ the harmonic 2-form the
sequence carries as its nullspace vector, $\|\mathrm{curl}(u \times h)\|^2$ is a
near-exact model of the Gauss-Newton operator (error: the $\mathrm{curl}\,A$
fraction, a few percent), built from a field known at construction and never
changing. And $h \propto R^{-1} e_\zeta$ is purely toroidal, so
$\mathrm{curl}(u \times h) \approx h^\zeta \partial_\zeta u$: a Kronecker term with a
lumped radial profile of $|h|^2$ and exact Fourier diagonalisation in $\zeta$
(eigenvalue $n^2$), plus the dense polar core -- the existing Laplacian-atom
construction with the $\zeta$-axis stiffness profile replaced. Blind spot: the
$n = 0$ modes get zero parallel derivative whatever their $m$, while the real
field gives them $\iota m$; promoting $n^2$ to $(\iota m + n)^2$ with the
$\iota(r)$ profile the sweeps already produce is the second step. Ladder:
Laplacian atom (every mode $k^2$, now) -> harmonic atom ($|h|^2 n^2$, everything
at hand) -> field-line atom ($|B|^2 (\iota m + n)^2$, one more profile); all
need the flat-mode floor below.

**Freezing the operator and recycling the solve (Tobias 2026-09-07: SIESTA
recomputes its operator every so many steps).** Matrix-free, freezing $H$ saves
nothing per action (three k=1 solves either way; the warm start of the
potential is already the zeroth-order reuse). What a frozen $H$ per chunk
enables is Krylov recycling: keep the lowest 20-50 Ritz vectors of the
preconditioned operator from one MINRES solve and deflate them from the next
(GCRO-DR and relatives). Those vectors are the flat modes the Laplacian atom
misses, learned rather than modelled, valid while the field changes slowly,
i.e. in the tail; storage plus one orthogonalisation per iteration. Combines
with the substeps along a frozen direction. Measurement: the MINRES count at
fixed direction quality. Not code now.

**Benchmark of the two models (job 18044754, float64, the step-5000 field,
`outputs/newton_second_variation/harmonic_probe.log`).** The field is 96.4%
harmonic ($\|B - c h\| / \|B\| = 0.036$, $c = 0.9994$; the harmonic part's
current is $10^{-13}$). On the Ritz vectors of 150 Lanczos steps of
$P M_2^{-1} H$, the ratio of the true quadratic form to the model's:

| Ritz value | $(y, Hy) / (y, H_h y)$, harmonic model | $(y, Hy) / (y, L_2 y)$, Laplacian model |
|---|---|---|
| 0.147 (smallest) | 0.08 | 1.7e-6 |
| 1.7, 4.6, 9.2, 15 | 0.52, 0.69, 0.83, 0.84 | 2e-5 .. 2e-4 |
| 23 .. 116 | 0.93 .. 0.99 | 4e-4 .. 2e-3 |
| 400 .. 3.9e4 (the rest) | 0.97 .. 1.03 | 7e-3 .. 0.11 |
| **spread = condition an exact inverse leaves** | **12.7** | **6.8e4** |
| no preconditioner | | 2.7e5 |

Above $\theta \approx 30$ the harmonic Gauss-Newton model is the Hessian to 3%;
it fails only on the seven lowest modes, down to 0.08 at the bottom: the
missing $\iota m$ term and $u \cdot \nabla B$, exactly where predicted. An exact
inverse of $\|\mathrm{curl}(u \times c h)\|^2$ would leave a condition number of
13 against 6.8e4 for the Laplacian model (whose ratio spans five orders of
magnitude: the anisotropy), i.e. MINRES in a handful of iterations instead of
300 -- before the atom's own lumping error, which is the same kind of error the
existing atoms carry. The remaining seven modes need no mechanism (Tobias 2026-09-07): isolated
outliers cost a Krylov method about one iteration each, so with the harmonic
model MINRES would need of the order of ten to fifteen iterations instead of
300, before the atom's lumping error; deflation would save those seven and no
more, and recycling becomes the fallback for the case where the flat end is a
ladder rather than seven points (denser rational surfaces, more shear, finer
meshes). This is the number that makes the harmonic atom the next thing to
build.

Two approximations are stacked in it, to be kept apart. (i) *Gauss-Newton*:
only $(\delta B(u), \delta B(v))$ is kept, the cross term
$\tfrac12[(B, \mathrm{curl}(u \times Q_v)) + (B, \mathrm{curl}(v \times Q_u))]$, at an
equilibrium the $J \times Q$ part of the force operator, is dropped. The probe
measured $(u, Hu) / \|Q_u\|^2 = 1.00$ to 1.015 at both fields, so on li383 the
dropped term is a percent. That is not general: $J \times Q$ carries the
current-driven physics (kink, tearing), it is small here because li383 at this
beta is far from any ideal instability, which the empty negative spectrum also
says; near marginal stability it is the whole story. The operator MINRES
applies stays the full $H$, only the preconditioner approximates. (ii) *The
mode structure*: only the parallel derivative $B \cdot \nabla u$ is kept and
$u \cdot \nabla B$ dropped, valid when $u$ varies on scales shorter than the
equilibrium's, and least accurate exactly on the resonant modes where the
kept term vanishes. A preconditioner that also sees the cross term is
block-tridiagonal in the mode index (the equilibrium's spectrum couples
neighbouring $(m, n)$), which is SIESTA's construction and the other end of
the effort scale. First measurement for either: the MINRES count on the
step-5000 field against the 300 of the Laplacian atom.

## 7d. The harmonic atom: what it is and how it is inverted

Built 2026-09-07 (Tobias: "build the harmonic atom and run, float64");
`mrx.hessian.harmonic_preconditioner`, `--newton-precond harmonic`.

**The model.** With $\mathrm{div}\,u = 0$ and $\mathrm{div}\,h = 0$,
$\mathrm{curl}(u \times h) = h \cdot \nabla u - u \cdot \nabla h$. The Hessian is the
Gauss-Newton form $\|\mathrm{curl}(u \times B)\|^2$ to a percent and $B = c\,h + \mathrm{curl}\,A$
is 96% harmonic, so the benchmark of 7c: $\|\mathrm{curl}(u \times c h)\|^2$ is the
Hessian to 3% on every Ritz vector but the seven lowest. The atom keeps the
first term, the parallel derivative, and lumps it. $h$ is the vacuum field
inside the boundary and carries its own rotational transform, so in the
logical angles $h \cdot \nabla = (h^\theta/J)\,\partial_\theta + (h^\zeta/J)\,\partial_\zeta$
(logical contravariant components), and on a Fourier mode $(m, n)$ of a
radial layer its symbol is $(2\pi)^2 (\bar h^\theta(r)\, m + \bar h^\zeta(r)\, n)^2$
with the angle-averaged profiles: the resonance denominators, zero on
$\bar h^\theta m + \bar h^\zeta n = 0$. The dropped second term is a
multiplication by the strain of $h$, a bounded operator of size
$|\nabla h| \sim |h| / R$ (below); it enters as the floor
$\varphi(r) = \kappa\, (2\pi)^2 (\bar h^{\theta 2} + \bar h^{\zeta 2})$, $\kappa$ the
one knob (`HARMONIC_FLOOR` = 1e-2; swept in the probe), which is what the
resonant modes see.

**The inversion.** The Newton operator on the potential is
$\mathrm{curl}^T H\,\mathrm{curl} \approx \mathrm{curl}^T (\text{parallel symbol})\,\mathrm{curl}$:
the curl-curl of the potential form times the symbol. The Laplacian atom
$P_L$ is the existing approximate inverse of the curl-curl (plus grad-div,
harmless on the gauge). The harmonic atom is the sandwich

$$P_h = W\, P_L\, W^T, \qquad W = E\, C\, E^T,$$

with $C$ the diagonal scaling by $(\lambda_\parallel + \varphi)^{-1/2}$ in the
2-D Fourier basis of the two angles on the tensor DoF grid of each component
of the 1-form (two FFTs per component per apply) and $E$ the Dirichlet
1-form extraction. In the bulk, where $E$ is the identity, $P_h$ is the
Laplacian atom with $\lambda_\parallel + \varphi$ multiplied into its
denominator, i.e. the fast-diagonalisation inverse of "curl-curl times the
parallel symbol"; on the polar rows the extraction lumps the scaling. It is
symmetric positive definite for any $C$, so MINRES stays correct whatever
the profiles are, and the quality is the measurement. The profiles are the
quadrature averages of $h/J$ over the angles, interpolated to the radial
DoF index; the Fourier frequencies are those of the DoF grid; no metric
factors beyond those in $P_L$. Cost: negligible next to the Hessian action.

**What is known about $\nabla h$.** $\mathrm{curl}\,h = 0$ makes $\nabla h$
symmetric, $\mathrm{div}\,h = 0$ makes it traceless, and together each
Cartesian component of $h$ is harmonic ($\Delta h = \nabla \mathrm{div}\,h - \mathrm{curl}\,\mathrm{curl}\,h = 0$),
so $|\nabla h|$ is bounded by $|h|$ over the scale of the geometry, $R$ for
the $1/R$ toroidal field and the boundary shaping. Hence $u \cdot \nabla h$ is a
bounded, mass-like term of size $|h||u|/R$, while $h \cdot \nabla u$ is
$|h|\, k_\parallel |u|$: the ratio is $k_\parallel R$. Neither is "very
small" in general: the derivative term dominates every stiff mode
($k_\parallel R \sim n$ at the grid scale), the strain term is all that is
left on a resonant mode, and on the exact kernel ($u = h$, the $E \times B$-type
flows $u = h \times \nabla\phi / |h|^2$) the two cancel identically. That is the
spectrum of 7c: a stiff end set by $k_\parallel$, a floor set by the strain,
and a few near-kernel modes below the floor where the cancellation is
partial.

**Measured (job 18045878, float64, the step-5000 field,
`harmonic_atom_probe.log`).** Direction quality per MINRES budget, the energy
the exact line search removes along the direction ($\Delta E^*$), against the
Laplacian atom:

| preconditioner | iterations (cumulative) | $\Delta E^*$ | $\Delta t^*$ |
|---|---|---|---|
| Laplacian atom | 300 (tol 0.1 not reached) | 1.06e-8 | 1.67 |
| harmonic, $\kappa$ = 1e-3 | 194 (tol 0.1) / 494 | 2.57e-8 / 6.19e-8 | 1.36 / 1.44 |
| harmonic, $\kappa$ = 1e-2 | 269 (tol 0.1) / 569 | 6.87e-8 / 9.78e-8 | 1.44 / 0.85 |
| harmonic, $\kappa$ = 0.1 | 300 | 8.55e-8 | 1.28 |
| harmonic, $\kappa$ = 1 | 300 | 5.32e-8 | 1.51 |

At 300 iterations the harmonic atom's direction removes 6-8x the energy of the
Laplacian atom's ($\kappa$ = 1e-2..0.1); at 569 iterations with $\kappa$ = 1e-2
the line search sits at $\Delta t^* = 0.85$, a nearly converged Newton
direction, which no Laplacian-atom solve reached. Far from the factor an exact
model inverse would give (condition 13 in 7c): the sandwich's lumping (radial
averaging of the profiles, the polar rows, the interplay with the Laplacian
atom's own approximations) eats most of it. Still the largest single gain of
the study; the iteration counts to a given tolerance are not comparable across
preconditioners (each measures the residual in its own norm), the direction
quality is. Default `HARMONIC_FLOOR` = 1e-2.

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

Jobs 18032142 (initial field) and 18032143 (the anchor's step-5000 field,
residual 2.3e-4), float64, li383 (16,32,32) p=2, 11 min each;
`outputs/newton_second_variation/probe_{ic,relaxed5000}.json`.

**Identities** (relative differences): gradient against the force's pairing
5e-9 / 7e-8, quadratic form against $\|Q\|^2 + (B, R)$ 8e-12 / 8e-12, symmetry
6e-9 / 6e-9, $(B, HB)/|B|^2$ 1e-20 / 1e-21. $|HB| / |Hu|$ for a random unit
$u$: 2.2e-4 at the initial field, 4.7e-6 at step 5000: the field's own direction
enters the kernel as the state relaxes.

**Spectrum** of $P M_2^{-1} H$ on divergence-free velocities, 150 Lanczos steps
with full reorthogonalisation, at both fields alike:

| | initial field | step 5000 |
|---|---|---|
| largest Ritz value | 3.93e4 | 3.89e4 |
| smallest | 0.117 | 0.123 |
| negative | 0 | 0 |
| condition number | 3.4e5 | 3.2e5 |
| next smallest | 1.7, 4.6, 9.1, 15, 23, 32, 42 | 1.7, 4.6, 9.1, 15, 22, 31, 42 |

The Hessian is positive definite on divergence-free velocities at both
fields (no ideally unstable direction on this mesh, at the initial VMEC field
as at the relaxed one), and its low end is a clean discrete ladder, not a dense
near-kernel. $(u, Hu) / \|Q_u\|^2 = 1.00$ to 1.5% for every direction tried:
the Hessian is the Gauss-Newton operator $\|\mathrm{curl}(u \times B)\|^2$ to
that accuracy, the $J \times Q$ terms are a percent.

**Direction quality.** $\Delta E^*$ is the energy the exact line search removes
along the direction (first-order path; the second-order path and the Newton
model agree with it to three digits everywhere). Cost is MINRES iterations,
three k=1 mass solves each, 0.12-0.13 s per iteration in float64 in the steady
state (a descent step is 0.69 s in mixed precision, 1.44 s in float64).

At the initial field (residual 1.6e-2): force $-9.4$e-7, smoothed force
$-9.2$e-7; Newton shift 0 at 139 iterations $-1.33$e-6, shift 1e-3 at 84
$-1.16$e-6, shift 1e-2 at 45 $-1.00$e-6. Newton buys 1.4x per step there and
nothing per second: the initial descent is not the problem.

At step 5000 (residual 2.3e-4):

| direction | iterations | cos | $\Delta t^*$ | $\Delta E^*$ | per smoothed step | per wall second |
|---|---|---|---|---|---|---|
| force | 0 | 1.00 | 7.5e-4 | $-1.8$e-11 | 0.7 | 2.6e-11 |
| smoothed force | 0 | 0.87 | 1.8e-3 | $-2.6$e-11 | 1 | 3.7e-11 |
| Newton, shift 0, tol 0.1 | 300 (not converged) | 0.48 | 1.67 | $-1.06$e-8 | 410 | 2.9e-10 |
| shift 39 (1e-3 of top), tol 0.1 | 89 | 0.69 | 6.2 | $-1.39$e-9 | 54 | 8.4e-11 |
| shift 39, tol 0.01 | 273 | 0.70 | 6.9 | $-1.82$e-9 | 70 | 5.3e-11 |
| shift 389 (1e-2), tol 0.1 | 35 | 0.85 | 5.6 | $-1.90$e-10 | 7 | 4.2e-11 |
| shift 389, tol 0.001 | 172 | 0.86 | 5.9 | $-2.28$e-10 | 9 | 1.1e-11 |

No Newton direction was CFL-bound: $\Delta t^*$ of 1.7 to 7 was taken in
full, against 1.8e-3 for the smoothed force. The Laplacian atom beats its
square at every shift and tolerance (the square needs 2-3x the iterations for
the same residual and never reached 1e-2 in 300). The full Newton direction,
even truncated at 300 iterations, removes 400 times the energy of a smoothed
step at 50 times its cost: 8x per wall second in the float64 accounting; the
shift-39 direction 2x; shift 389 breaks even. Section 10 says whether that
holds over a run.

## 10. Results of job B (Newton arms)

All arms restart from the anchor's step-5000 state (residual 2.1e-4, helicity
5.011146e-3), li383 (16,32,32) p=2, mixed precision, CFL 0.5, one hour of
stepping (`--seconds 3600`, chunks of 20), MINRES tolerance 0.1, the Laplacian
atom, `outputs/newton_relax/<arm>/`, jobs 18033081/83/88, 18034296, 18037672.
Reference: the anchor's own continuation, 5000 descent steps in 53 min.
Residuals are chunk means; "min" is the lowest chunk mean of the run; the
helicity drift is relative. Figure `outputs/newton_second_variation/newton_tail.png`.

| arm | steps | s/step | energy removed | residual first / min / last | helicity drift | dt |
|---|---|---|---|---|---|---|
| anchor continuation (descent) | 5000 | 0.64 | 6.3e-8 | 2.2e-4 / 1.5e-4 / 1.5e-4 | -1.2e-6 | 2.4 |
| shift 0, 300 it, uncapped | 300 | 12.2 | 7.3e-7 | 2.4e-4 / 6.5e-5 / 6.6e-4 | +4.0e-5 | 2.0 |
| shift 0, 100 it, uncapped (cancelled at 15 min) | 140 | 6.4 | 1.5e-7 | 3.7e-4 / 6.1e-5 / 6.1e-5 | -3.9e-6 | 1.75 |
| shift 39, 100 it | 4000 | 0.75 | 1.8e-7 | 7.6e-4 / 2.1e-4 / 4.1e-4 | -3.2e-6 | 2.0 |
| shift 0, 300 it, dt capped at 1 | 220 | 16.6 | 6.5e-7 | 5.5e-5 / 4.4e-5 / 9.7e-5 | +2.4e-5 | 1.0 |
| the same, midpoint induction on B: the Picard iteration halves dt four times to 1/16 and stays unconverged (sweep limit) on every step, so this is Newton at dt = 1/16 | 220 | 17.6 | 1.7e-7 | 2.2e-4 / 5.8e-5 / 1.2e-4 | -5.9e-6 | 0.03-0.06 |
| the same, float64, tol 1e-10 (is the floor the tolerance?) | 200 | 29.5 | 7.8e-7 | 2.3e-4 / 4.2e-5 / 1.3e-4 | +3.1e-5 | 1.0 |
| the same, midpoint induction, float64 (Picard converged, 5 sweeps, defect 3e-15) | 200 | 29.7 | 7.9e-7 | 2.3e-4 / 4.2e-5 / 1.2e-4 | +3.1e-5 | 1.0 |

(The dt = 1/16 arm's 17.6 s/step is 16.6 s of Newton direction plus the 101
failed Picard evaluations, so it paid a full direction for a sixteenth of a
step: sixteen times the explicit arm's price per unit of path. Explicit
stepping at dt = 1/16 would follow the same trajectory to solver noise at the
same per-step cost, and the substep variant, one direction and sixteen explicit
induction steps of 1/16 at one k=1 solve each, at about 17 s per full step.
What the arm measured is the price of the shorter steps: helicity -0.26e-5 at
the residual minimum and -0.60e-5 at the end against -0.33e-5 and +2.4e-5 for
the full-step arm, a quarter of the drift at the end; and that even at 1/16
the residual turns around after the floor, 3.4e-9 to 1.5e-8.)

(A midpoint arm with the auxiliary field was started and cancelled after 60
steps: its helicity was exact to the digit, but the auxiliary force $J \times H$
has a residual of 3.4e-3 at this state, which was relaxed under the
$J \times B$ force, so its Newton direction was fifteen times larger, the CFL
cap bound at dt = 2e-3, and the run relaxed a different functional from far
away from its floor. Not comparable.)

What the arms say.

1. **Newton reaches the mesh's residual floor in minutes.** Every shift-0
   arm drops the residual from 2.1e-4 to 4.4e-5..6.5e-5 within 40 to 100
   steps, i.e. 8 to 15 minutes, against the continuation's 1.5e-4 after 53
   minutes and a power-law tail that would need hours for the same. The
   lowest chunk mean of every Newton arm is the same number, 4.4e-5 to
   6.5e-5: that is this mesh's floor, which the descent approaches as
   $t^{-0.2}$ and Newton hits directly.
2. **Beyond the floor the explicit induction breaks the topology.** Once
   there, every uncapped or capped arm shows the same sequence: the energy
   removal accelerates (7e-7 per hour against the continuation's 6e-8), the
   helicity drifts linearly (2e-5 to 4e-5 relative, twenty to forty times the
   continuation's), and the residual climbs back to and above its start.
   That is numerical reconnection, and it is not a matter of large steps: the
   CFL cap held in every arm (it never bound), no step moved the field more
   than half a cell. Forward Euler's topology error per step is second order
   in that step's displacement $d$, so over a run it accumulates as
   $\sum d_i^2$; the descent covers the path to the tail in 5000 steps of
   about 1/60 of a cell, Newton in 300 steps of 0.3-0.5 cells, a comparable
   path in thirty times longer steps, and $5000 (1/60)^2 \approx 1.4$ against
   $300 \cdot 0.3^2 \approx 27$ is the measured drift ratio of 20-40. The cap
   bounds each step; the error is the step length integrated along the path,
   and Newton spends its path in the fewest, longest legal steps. Once the
   accumulated error is a small reconnection, the direction finds the energy
   that reconnection releases and the line search takes it. The 100-iteration arm, cancelled
   at 15 minutes, had reached 6.1e-5 with a drift of 4e-6 and had not entered
   this phase yet.
3. **The step cap helps and does not cure.** With dt = 1 the residual reaches
   4.4e-5 at 12 minutes instead of 20 and the drift is half the uncapped
   arm's, but it is still twenty times the descent's and it still grows.
   The line search wanted dt* = 8 on average: the unresolved flat modes.
4. **The LM shift is the worst member of the family.** Shift 39 (1e-3 of the
   top of the spectrum) makes MINRES converge in six iterations and the step
   cost 0.75 s, and leaves the residual at 4e-4, above the descent, because
   the flat modes stay gradient-like while the line search stretches to twice
   the Newton length for the resolved ones.
5. **Cost.** A shift-0 Newton step costs 6 to 17 s (100 to 300 Hessian
   actions at 0.06 to 0.12 s in mixed precision, the direction warm-started
   from the previous potential; MINRES converged to 0.1 on about half the
   steps of the 300-iteration arm and never on the capped one). The
   Hessian-action count per unit of residual removed is what the $B$-aware
   preconditioner of 7c would lower.

**Conclusion for the method.** Used as it is, Newton is a floor finder: from
a descended state it reaches the discretisation floor of the force residual
in ten minutes where the descent needs hours, and it must stop there (a floor
test on the residual, which the driver has, or a helicity budget). Used past
the floor with the explicit induction it reconnects. The two ways to make it a
relaxation rather than a floor finder are on the time integration, not the
direction: the midpoint induction (the last arm), or substeps along the frozen
direction (one direction, ten induction steps of a tenth of the length at one
k=1 solve each: the same path at a tenth of the summed square, for the price
of ten cheap solves instead of ten Newton directions); and on the direction, a
preconditioner that resolves the flat modes so that fewer, more accurate
steps are taken. A run from the initial field would be cap-bound step by step
too; its cost is the whole relaxation's path instead of the tail's, at the
same error per unit path, so it needs one of the two fixes first.

## 10c. What happens once the force is small: why the descent slows and what the floor is

1. **The descent's step is set by the flat modes.** The anchor's line search takes
   $\Delta t^* \approx 2.4$ throughout the tail. For a quadratic energy the exact
   line-search step is the inverse curvature along the direction, so the force it
   descends along has curvature about 0.4 in the velocity metric: the bottom of
   the measured spectrum (0.12, 1.7, 4.6, ...). By step 5000 the force lives in
   the flattest modes, the ones with $\iota m + n \approx 0$.
2. **A step of that size over-relaxes everything stiff.** With $\Delta t \approx 2.4$
   every mode with curvature above 1 is stepped past its minimum, the stiffest
   by $\Delta t \lambda \sim 10^5$. The exact line search keeps the energy from
   rising, so this is stable, but each flat-mode step kicks the stiff modes, and
   the residual norm, which the stiff modes dominate, records the kick and not
   the progress. Hence the non-monotone residual, the energy rising on half the
   steps in the tail, and the slow power law of the residual while the energy
   keeps falling: the descent zigzags, memoryless BFGS damps the zigzag without
   removing it, and the residual measures the zigzag.
3. **Newton removes the zigzag and exposes the real floor.** Solving the stiff
   modes exactly, it reaches a squared residual of 2.3-4.6e-9 in every shift-0
   arm, whatever the iteration budget or the cap, and cannot go lower. That
   common value is a floor of the problem, not of the direction. Two candidates,
   which the data cannot yet separate: the discretisation, since the ideal
   minimum at fixed topology carries current sheets at the rational surfaces
   that the mesh cannot represent, leaving a force of the sheet's truncation
   error; or the solve tolerance, since at 1e-8 the force's gradient-part
   remnant equals the descent at a residual of about 3e-5 (the
   $0.1\,\mathrm{tol}/\mathrm{resid}^2$ law of the velocity-Leray A/B), which is
   where Newton bottoms out. Decided 2026-09-07: the float64 arm at tolerance 1e-10
   reproduced the mixed floor to the digit (squared 2.03e-9 vs 2.34e-9): the
   sheets, not the tolerance. The refined-mesh anchors (32^3, and (16,32,32)
   refined around the 3/5 and the 1/2, 6/11, 3/5 surfaces) test whether the
   floor drops with radial resolution at the surfaces.
4. **Past the floor the force is remainder, and Newton amplifies remainder.** The
   direction is the inverse Hessian applied to the force; once the force is
   noise, the direction is noise scaled by the inverse of the smallest
   curvatures, i.e. concentrated in the flat modes, which are the motions of
   rational surfaces. The descent applies $\Delta t$ times the same noise and
   moves nothing; Newton moves up to half a cell per step along it, and the
   explicit induction turns that into the reconnection of section 10.

## 10d. What the helicity loss is (float64 pair, 2026-09-07): the pairing's projection error, not the time error

The float64 explicit arm (tol 1e-10) and the float64 midpoint-on-$B$ arm (Picard
converged in 5-6 sweeps at $\Delta t \approx 1$) have identical helicity drifts
at the same steps and path lengths: $-3.7$e-6 / $-3.8$e-6 at step 5040 (path
35), $-1.44$e-5 / $-1.44$e-5 at 5080 (path 74); the mixed explicit arm the
same. The time integrator makes no difference, so section 10's summed-square
explanation is wrong and is retracted. What the midpoint scheme on $B$ does not
remove is the projection error of the discrete helicity pairing, $E^T P B$
instead of $E^T M_1 H$ (li383 note 5f found the same for the descent: the
B-only midpoint isolates exactly this error). It is large along Newton paths
and small along descent paths because of what moves: the flattest modes of
the discrete Hessian are grid-scale oscillations aligned with the field (the
lowest Ritz vector's Laplacian Rayleigh quotient is at the grid scale), the
Newton direction resolves and excites them, and $u \times B$ with grid-scale
$u$ is what the 1-form projection cannot represent; the smoothed descent
direction never contains them. The harmonic atom resolves those modes best
and its explicit arm leaked fastest (helicity +3.9e-7 in 40 steps, energy
1.14e-6 removed, more than the whole descent from the initial field: job
18045879, cancelled at 40 steps; the harmonic midpoint-on-$B$ arm 18046954
was cancelled unstarted for the same reason). The $\Delta t = 1/16$ arm's
smaller drift (a half at a fourteenth of the path) is consistent with a
per-step leak set by the direction's grid-scale content rather than by the
step.

Consequences: (i) the exact fix is the auxiliary-field formulation,
$E^T M_1 H = 0$ for any $u$, which needs its own anchor (the $J \times H$
force has its own fixed point, 3.4e-3 away from the $J \times B$ state):
auxiliary descent, then harmonic Newton with the auxiliary midpoint; (ii) the
cheap fix is not to excite those modes: smooth the Newton direction with
$(M + \mu L)^{-1} M$ before stepping (one line), or a Levenberg-Marquardt
shift in the $H^1$ metric inside the solve; (iii) the residual floor of
section 10c stands (the float64 arm at tol 1e-10 reproduced the mixed floor,
4.5e-5, to the digit: the discretisation, not the tolerance), and the
"floor finder" reading of Newton stands; what changes is the diagnosis of what
happens past it.

## 10b. The potential route against the Leray route

Three arms from the initial field at the anchor's settings (li383 (16,32,32)
p=2, mixed precision, CFL 0.5, 5000 steps), `outputs/potential_relax/`, jobs
18033055-57, against the sweep arms of 2026-09-05 (`h16_p2`, `h16_p2_m0`) and
the mu sweep's order-0 arm (`mu_sweep/g0`, verified to be on the same method:
its `c0.02` twin reproduces the anchor block for block). Residuals are
1000-step block means; figure `outputs/newton_second_variation/potential_route.png`.

| arm | route | s/step | blocks 1..5 | energy removed | helicity drift |
|---|---|---|---|---|---|
| m=1, gamma=0 | Leray | 0.398 | 3.12e-3 1.24e-3 6.31e-4 6.78e-4 6.27e-4 | | |
| | potential | 0.318 | 3.12e-3 1.25e-3 6.31e-4 6.79e-4 6.28e-4 | 2.010e-6 | -8.5e-8 |
| m=0, smoothed | Leray | 0.645 | 1.24e-3 5.78e-4 4.50e-4 3.82e-4 3.37e-4 | 1.633e-6 | |
| | potential | 0.452 | 1.24e-3 5.78e-4 4.50e-4 3.82e-4 (running) | 1.624e-6 (4500) | |
| m=1, smoothed | Leray (anchor) | 0.672 | 9.45e-4 4.34e-4 3.32e-4 2.77e-4 2.37e-4 | 1.819e-6 | -4.4e-8 |
| | potential | 0.451 | 1.23e-3 4.98e-4 3.41e-4 1.59e-4 1.08e-4 | 1.974e-6 | -5.5e-8 |

Three findings.

1. **The routes are the same method.** Without smoothing (m=1, gamma=0) and
   with smoothing at m=0 the potential arm reproduces the Leray arm's
   residual to three digits in every block: the curl-curl solve plus the
   harmonic coefficient is the Leray projection, as section 7b proves, and
   the smoothing through the potential is the smoothing of the velocity.
2. **It is cheaper.** 20% per step without smoothing (the k=1 Hodge split
   against the k=3 saddle solve) and 30% with it (the k=1 shifted solve
   against the k=2 one on top). The velocity is divergence-free to roundoff
   by construction.
3b. **Confirmed on the Leray route** (`--smooth-first true`, job 18037645,
   `outputs/potential_relax/leray_m1_sf`): the saddle-point route with the
   potential route's order ends at a squared residual of 2.08e-8 (blocks
   1.23e-3, 5.49e-4, 3.34e-4, 2.40e-4, 1.43e-4 in the residual) against the
   anchor's 5.72e-8 and the potential arm's 1.18e-8, with the potential arm's
   energy removed (1.97e-6) and helicity drift (-1.14e-5) to the digit, at
   the anchor's 0.64 s/step. The ordering accounts for the bulk of the gain;
   the remaining 2.08 against 1.18 is within the scatter of single m=1
   trajectories (their block means swing by tens of percent). Making the PCG
   order the production default is a one-line change on either route.
3. **The ordering is worth a factor two in the tail.** The only difference
   left, at m=1 with smoothing, is that the potential route smooths the
   force and lets L-BFGS combine the smoothed forces (the preconditioned-CG
   order), while the Leray route combines the raw forces and smooths the
   combination, which smooths the already smooth history vector a second
   time. The potential arm ends at 1.08e-4 against the anchor's 2.37e-4 at
   step 5000, removes 8% more energy, in two thirds of the wall time. The
   helicity drift is the same to 25%. This is a change the Leray route can
   adopt on its own (smooth before the recursion); the m=0 identity says the
   projection route has nothing to do with it.

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

## 12. Towards production (thinking, not code; Tobias 2026-09-07)

**The midpoint arm's Picard "failure" is a tolerance artefact.** The run file
(140 steps): CFL number taken 0.003, contraction constant negligible, Picard
defect stalled at 4e-6..2.5e-5 on every step, i.e. the inner solves' noise
relative to a tiny increment; the tolerance 1.3e-6 (relative to $\|dt\, dB\|$)
is unattainable, so every step burns 100 sweeps and four halvings and goes out
at dt/16. A restrictive CFL would make it worse (smaller increment, larger
relative noise). Fix when the code is touched again: a floor on the Picard
tolerance relative to the field, not the increment. Until then midpoint +
Newton is Newton at dt/16 by construction.

**One long implicit induction step instead of many explicit substeps (Tobias
2026-09-07).** With the velocity frozen the midpoint induction
$(I - \tfrac{dt}{2} L_u) B_{n+1} = (I + \tfrac{dt}{2} L_u) B_n$, $L_u B = \mathrm{curl}(u \times B)$,
is a linear system, the Cayley transform of the generator: helicity exact
(auxiliary field; projection error on $B$ itself) at any step, topology error
$(dt\,\|L_u\|)^3/12$ per step, about 1% of the displacement at the Newton
length with CFL 0.5 -- equal to sixteen explicit Euler substeps of 1/32 in
total and forty times better than the single explicit step. Picard is the
Neumann series of that system (needs $dt\,\|L_u\| < 1$, stalls at noise in
float32 at small increments: the dt/16 arm). A Krylov solve instead: the
operator is $I$ minus something of norm ~0.5, condition ~3, CG on the normal
equations (the operator is not $M_2$-symmetric; $L_u$ is skew only in the
helicity pairing) in 10-20 iterations of one induction evaluation each, the
cost of the sixteen substeps. Ranking for stepping a Newton direction: one
explicit step (the arms) < sixteen explicit substeps ~ one implicit full step
(the latter helicity-exact) < many implicit substeps. In code: replace the
Picard loop of the midpoint solve by a normal-equations CG on the same
increment map. Not now.

**Newton in its own file.** `mrx/hessian.py` -> `mrx/newton.py`: the second
variation, the direction solve, a small config (shift, tol, maxiter, precond,
dt cap, inner tol). The stepper takes a direction strategy, L-BFGS (history,
order) or Newton (config), each owning its state as one sub-pytree of `State`
(histories / the potential warm start); no `newton_*` fields, no `if
self.newton` branches. The Newton step uses the potential force and never calls
the Leray solve; the per-step residual comes from that force. Stop at the floor
with the driver's floor test. Defaults from the experiment: shift 0, tol 0.1,
300 iterations, cap 1, Laplacian atom. One cheap relaxation test with a few
Newton steps on the fixture; a concepts section on the floor-finder role and
the reconnection caveat.

**The potential route as the production velocity.** Same iterates, 20-30%
cheaper, divergence-free to round-off, smooth-first order (validated on both
routes). It generalises to the auxiliary field for free (rhs = curl^T load(J x
X) for whichever X the cross product reads), so it can be the only route in the
step for every scheme; the Leray projection moves to where its pressure is
needed (sampler, weak pressure, diagnostics, tests). The step's Leray warm
starts (p, JxH, sigma) leave the state, the potential warm start stays.
Smooth-first becomes the only order, deleting the post-combination smoothing.
Caveats: single m=1 trajectories scatter by tens of percent (the factor two is
a trend, not a number); float32 storage puts the potential force 3e-5 off the
Leray force.

## 13. Files

- `mrx/hessian.py`: `second_variation(seq, B, J)` (the Hessian action),
  `newton_direction(seq, B, J, MF, a_guess, shift, tol, maxiter, precond)`.
- `mrx/relaxation.py`: `TimeStepper.newton*`, `State.a / newton_it / newton_fallback`,
  the trace entries and the summary line.
- `scripts/relax.py --newton --newton-shift --newton-tol --newton-maxiter --newton-precond`.
- `scripts/newton_probe.py`, `test/test_hessian.py`.
- Outputs: `outputs/newton_probe/<arm>/probe.json`, `outputs/newton_relax/<arm>/relax.json`
  (copied to the main checkout's `outputs/newton_second_variation/`).
