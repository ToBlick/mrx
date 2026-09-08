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

**The harmonic velocity the Newton solve omits (Tobias 2026-09-07: "our
Hessian method is assuming u = curl w, no harmonic part. Is that a
problem?"; jobs 18112659/18112660, `harmonic_velocity_probe.py`,
`harmonic_velocity_probe_16.log`, `_32.log`, float64).** The div-free
velocities with $u \cdot n = 0$ are $\mathrm{curl}\,V^1_0 \oplus \mathrm{span}(h)$,
$h$ the harmonic 2-form (net toroidal flux); the descent's Leray force keeps
the $h$ coefficient, the Newton solve on the potential drops it. Measured
with $h^T M_2 h = 1$: the share of the projected force along $h$,
$(h^T M F)^2 / (F^T M F)$; the Newton step on $\mathrm{span}(h)$ alone,
$c_h = h^T M F / h^T H h$, and its energy $\Delta E_h = -\tfrac12 (h^T M F)^2 / h^T H h$;
and, given the truncated direction $u_N$ (Laplacian atom, 300 iterations),
the residual of the omitted equation $h^T M F - h^T H u_N$ and the energy it
would add.

| state | $|PF|^2$ | share along $h$ | $c_h$ | $\Delta E_h$ | $\Delta E_h / \Delta E_N(300)$ | $\Delta E$ of the $h$-equation given $u_N$ |
|---|---|---|---|---|---|---|
| (16,32,32) step 5000 | 4.9e-8 | 2.7e-6 | 3.1e-4 | -5.6e-11 | 5e-3 | -2.8e-11 |
| (16,32,32) Newton floor (5040) | 2.8e-9 | 3.0e-9 | 2.5e-6 | -3.6e-15 | 2e-6 | -5.7e-15 |
| (32,32,32) step 5000 | 1.1e-7 | 1.6e-11 | 1.1e-6 | -7.2e-16 | 3e-7 | -1.6e-15 |
| (32,32,32) Newton floor (5200) | 5.4e-9 | 6.8e-10 | 1.6e-6 | -1.5e-15 | 3e-5 | -2.0e-15 |

$h^T H h = 1.18 \times 10^{-3}$ on both meshes (against curvatures 0.12 to
$4 \times 10^4$ on the curls): a velocity along $h$ is nearly parallel to
$B$ (the field is 96% harmonic), so its induction $\mathrm{curl}(h \times B)$ and the
work $h \cdot (J \times B)$ are both first order in the non-harmonic 4%, and their
ratio $c_h$ is $10^{-6}$ to $3 \times 10^{-4}$. Against the exact Newton
decrement of 1.13e-7 at the coarse step-5000 state the harmonic step is
worth $5 \times 10^{-4}$ of it; at every floor it is $10^{-15}$ in energy and
below $10^{-8}$ of the residual. Not a problem: the omission loses nothing
the residual or the energy can see, and the Leray force's harmonic
coefficient in the descent is a formality for the same reason. (If it ever
mattered, the fix is one scalar per step: the harmonic block is a single
coefficient with its own right-hand side and one inner product of coupling
to the curl block.)

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

**Lumping $u \cdot \nabla h$ instead of dropping it (Tobias 2026-09-07).** Per
mode the lumped derivative is $i k_\parallel I$ (anti-Hermitian) and the
angle-averaged strain $\bar S(r) = \langle \nabla h \rangle$ is real symmetric
($\mathrm{curl}\,h = 0$), so
$(i k_\parallel I - \bar S)^H (i k_\parallel I - \bar S) = k_\parallel^2 I + \bar S^2$:
the cross terms cancel and the lumped strain is a plain addition, a positive
$3 \times 3$ matrix per radial layer, the same for every mode -- the floor,
computed instead of tuned, $r$-dependent and anisotropic; within the
per-component atom its diagonal, $\varphi_c(r) = \langle (\nabla h)^T \nabla h \rangle_{cc}$,
and no $\kappa$. Needs $\nabla h$ at the quadrature points in the atom's
representation (a `jacfwd` of the discrete field, or grid differences: a
floor tolerates crudeness) and the angle average of $S^T S$ per layer; ~20
lines in the harmonic atom. Cannot restore the kernel cancellation (harmless
for a preconditioner) nor the mode coupling of the un-averaged strain. To be
built into the harmonic atom once the bands verdict is in, and measured with
it on the Newton count.

**The averaging must bundle the product (Tobias 2026-09-07).** For the $r$
component with $h^r \approx 0$ and $k = 2\pi(m, n)$,
$\int (g_{rr}/J) |(h \cdot \nabla) u^r|^2 = (2\pi)^2 [w_{\theta\theta} m^2 + 2 w_{\theta\zeta} m n + w_{\zeta\zeta} n^2]$
with $w_{ij}(r) = \langle (g_{rr}/J)(h^i/J)(h^j/J) \rangle_{\theta\zeta}$: three
bundled profiles per component and radius, a quadratic form in $(m, n)$ equal
to the Fourier-diagonal of the true operator (its Galerkin projection on one
mode, coupling dropped). The atom's $(\langle h^\theta/J \rangle m + \langle h^\zeta/J \rangle n)^2$
is a product of averages with the $g_{cc}/J$ left to the Laplacian atom: not
the same. Consequences: (i) $w_{\theta\zeta}^2 \le w_{\theta\theta} w_{\zeta\zeta}$
with equality only for a pitch $h^\theta/h^\zeta$ constant on the surface, so
$\langle k_\parallel^2 \rangle$ on a rational surface is positive -- the
resonance is smeared by the pitch variation and part of the floor is already
in the bundled coefficients; the number to compute per surface is
$1 - w_{\theta\zeta}^2/(w_{\theta\theta} w_{\zeta\zeta})$; (ii) the sandwich
multiplies the Laplacian atom's denominator (the $g^{aa}J$ curl-curl weights)
by a dimensionless symbol, but the model's weight sits inside the curl-curl
per component; that factorisation does not hold. The construction in which the
lumping is done right is the per-mode radial assembly with the bundled
weights as radial mass weights per component (+ the bundled strain, + $h^r$
terms if kept): the block-Jacobi of the true operator in Fourier space.

**The outer curls (Tobias 2026-09-07).** Per mode the incidence curl is a
$3 \times 3$ operator with radial derivatives,
$(\mathrm{curl}\,a)^r = D_\theta a_\zeta - D_\zeta a_\theta$,
$(\mathrm{curl}\,a)^\theta = D_\zeta a_r - \partial_r a_\zeta$,
$(\mathrm{curl}\,a)^\zeta = \partial_r a_\theta - D_\theta a_r$ ($D_\theta, D_\zeta$ the
incidence stencils' discrete Fourier symbols, $\partial_r$ the banded radial
incidence), so $\mathrm{curl}^T H\,\mathrm{curl}$ per mode is
$\mathrm{curl}_{mn}^T \mathrm{diag}_c(W_c(r; m, n))\, \mathrm{curl}_{mn}$: a $3 n_r \times 3 n_r$
block coupling the components through $\partial_r$ and the $r$ component.
The sandwich carries the curls through the Laplacian atom, one Kronecker sum
per component with the off-diagonal component blocks dropped -- an
approximation before the symbol enters. The per-mode assembly keeps the
coupling exactly; its gauge kernel ($n_r$ radial profiles of $\nabla \phi_{mn}$
per mode, $\mathrm{curl}\,\nabla = 0$ on the incidences) is completed to SPD by
the weak half $\nabla \mathrm{div}$ per mode as the Hodge Laplacian does (MINRES
never uses the gradient part of a consistent solve). Full block:
$\mathrm{curl}_{mn}^T W \mathrm{curl}_{mn} + (\text{weak half})_{mn}$, bandwidth
$\sim 3(2p+1)$, from the 1-D incidence, mass and stiffness routines the atoms
own; the only approximation left is the angle averaging.

**Measured (job 18065315, `pitch_probe.log`, li383 (16,32,32) p=2).** Lumped
pitch $w_{\theta\zeta}/w_{\zeta\zeta}$ = 0.16-0.20 = $\iota_h / n_{fp}$ (the logical
$\zeta$ covers one period), i.e. $\iota_h$ = 0.49-0.59. Smearing
$1 - w_{\theta\zeta}^2/(w_{\theta\theta} w_{\zeta\zeta})$ = 0.10-0.25 across the bulk
for all three components (0.21/0.25/0.31 at $r$ = 0.25, 0.11/0.16/0.24 at the
3/5 surface): the local pitch of $h$ varies by that much over a surface because
the logical $\theta$ is VMEC's angle, not straight-field-line ($\lambda$ is not
in the map). Hence a resonant direction is a packet of Fourier modes
$(m, n), (m, n \pm n_{fp}), \ldots$, each carrying 10-25% of a non-resonant
curvature in the Fourier diagonal: every mode-diagonal preconditioner (the
sandwich, the per-mode block, the bands) overestimates the flat packets'
curvature by that factor and leaves them as low outliers for Krylov or
deflation. A remedy at the preconditioner level is an internal change to the
field-aligned angle of $h$ (no longer a plain FFT). The strain floor is moot
for $m \gtrsim 3$: the bundling carries a floor of that size. And
$|h^r|/|h|$ = 0.05-0.08 over most of the radius (the harmonic field's surfaces
are not the VMEC surfaces): for the grid-scale-in-$r$ flat modes
$(h^r/J)\partial_r u$ is of the order of the missing $k_\parallel$ term; the
sandwich drops it, the per-mode block can assemble it (a radial
first-derivative term). Net: the model is right, its flat directions are wave
packets in these angles, and the mode-diagonal constructions can at best get
the stiff end right.

**The coarse space, measured (job 18065847, `coarse_probe.log`).** Tobias:
"we can afford to compute the stiff (harmonic) modes at the start of the run
once". Preconditioned Lanczos (250 steps, 51 s) on the sandwich-preconditioned
model $P_h A_h$: $\theta \in [7.4\text{e-}5, 7.9\text{e}3]$, 3 Ritz values below
0.1, 8 below 0.5, 234 above 2. Adding the $k$ lowest Ritz pairs as a coarse
space, $P = P_h + W \mathrm{diag}(1/\theta) W^T$: MINRES to 0.1 on the true system
269 / 266 / 266 / 264 iterations for $k$ = 0 / 10 / 30 / 100, direction quality
unchanged. The flat packets are three modes and cost nothing; the spread is at
the STIFF end -- the sandwich mis-scales the stiff modes by up to 3 orders
(bulk spread 4e3, against 7e4 for the Laplacian atom on the same model), i.e.
the commutation failures and the metric bookkeeping, not the resonances. A
coarse space is the wrong fix; the symbol variants (bundled weights, half
power) are the last cheap probe (job 18066512).

**Symbol variants, measured (job 18066512, `symbol_probe.log`; Lanczos of the
preconditioned model, spread = max / 10th Ritz value; MINRES to 0.1 on the true
system):** Laplacian atom spread 1.1e3, 300 (not reached); current sandwich
(averages, power 1) 2.1e3, 269; bundled weights power 1: 9.6e3, 300; averages
power 1/2: 1.1e3, 300; bundled 1/2: 1.1e3, 300; bundled 3/4: 1.2e3, 300. No
symbol or power beats the current one, and the whole "scaling in the angles
around the Laplacian atom" family sits at a bulk spread of 1e3-1e4: the
stiff-end mismatch is the commutation, not the averaging or the metric. Family
closed.

**The factored preconditioner (Tobias 2026-09-07: "apply the Laplacian twice
and figure out h").** The Gauss-Newton operator factors exactly,
$K = C^T X^T S_1 X C$ ($C$ = curl, $X: u \mapsto M_1^{-1} \mathrm{load}(u \times h)$,
$S_1 = \mathrm{curl}^T M_2 \mathrm{curl}$ on the 1-forms), so
$P = C^+ X^+ S_1^+ (X^+)^T (C^+)^T + \epsilon P_L$ with $S_1^+ = P_L$ (the k=1
Laplacian atom), $C^+ = P_L \mathrm{curl}^T M_2$, $X^+ E = M^{atom}_2 \mathrm{load}_2(h \times E / |h|^2)$
(pointwise, no averaging; $h$ or $B$ itself), $(X^+)^T y = \mathrm{load}_1((v \times h)/|h|^2)$
with $v = M^{atom}_2 y$; $\epsilon P_L$ for the kernel of $X$ (flows along $h$).
Three atoms, two mass atoms, two loads, no solve, no commutation assumption,
no coordinates. Probe: job 18066978 (`factored_probe.log`), Lanczos spread and
MINRES counts, $X$ from $h$ and from $B$, $\epsilon$ = 1e-2 / 1e-4.

**Factored, measured (jobs 18066978, 18067686; `factored_probe*.log`).** Raw:
at $\epsilon$ = 1e-2 the shift swallows the factored part (= the Laplacian atom);
at 1e-4 the spread of the preconditioned model is 3.7e2, MINRES reaches 0.1
in 243-246 iterations (best count of the study), but the direction removes
1.2e-8 (the Laplacian atom's, not the sandwich's 6.9e-8). Normalised (factored
part scaled to the atom on a random vector, $s \approx 1.1\text{e}3$), spread
and energy of the direction at fixed budgets:

| preconditioner | spread max/10th | $\Delta E^*$ at 100 it | at 300 it |
|---|---|---|---|
| Laplacian atom | 5.9e2 | 3.2e-9 | 1.07e-8 |
| harmonic sandwich | 1.2e3 | 2.5e-8 | 7.6e-8 |
| factored ($h$ or $B$), $\epsilon$ = 0.1 | 2.0e2 | 2.6e-9 | 7.6e-9 |
| factored, $\epsilon$ = 1e-2, 1e-3, 0 | 1.0e2 | 2.0e-9 | 3.7-4.6e-9 |

The best spread of the study on the model and the worst directions on the
true system, $h$ or $B$, any $\epsilon$. Either the true operator's difference
from the Gauss-Newton model (the parallel flows, the cross terms) is where the
energy lives, or the transposes are not exact and $P$ is not symmetric (then
MINRES converges in a meaningless norm). Symmetry check (job 18068767): asymmetry 5e-16 for the factored part, 2e-15
for the X+ transpose pair, 5e-14 / 5e-15 for the atoms -- the transposes are
exact, the result is a property. The energy-removing directions live in what
the true operator has and the model's inverse does not weigh: the nearly
parallel flows and the cross terms, the subspace the factored construction
hands to the shift. The preconditioner chapter closes here: the sandwich is
the best available (6-8x the Laplacian atom per iteration), and nothing
cheaper than field-aligned angles or the dense-in-r block improves on it.

**Convergence curves in fixed norms (job 18068950, `curve_probe.log`; the
true residual $\|rhs - K a\|$ of the true system relative to the right-hand
side, 2-norm and k=1 mass-atom norm, and $\Delta E^*$, after 25..300 MINRES
iterations):**

| preconditioner, 300 it | 2-norm | mass-atom norm | $\Delta E^*$ |
|---|---|---|---|
| Laplacian atom | 0.52 | 0.34 | 9.2e-9 |
| harmonic sandwich | 0.71 | 0.36 | 6.6e-8 |
| factored, $\epsilon$ = 0.1 | 0.61 | 0.39 | 6.7e-9 |
| factored, $\epsilon$ = 0.01 | 0.78 | 0.53 | 4.2e-9 |

No preconditioner converges the Newton system in any fixed norm in 300
iterations (a factor 1.3-3 on the residual); the sandwich's true residual is
the LARGEST in the 2-norm at every count while its direction removes 7x the
energy: residual norms are dominated by the stiff modes, energy removal by the
flat ones, and the preconditioners differ in which modes they resolve first,
not in how far they get. The factored one is worse than the Laplacian atom in
every measure on the true system: a genuine negative (transposes exact,
symmetry 5e-16). Consequences for the code: the Newton solve is a truncated
solve by design, with the iteration budget as the knob; its stopping test now
follows the house criterion (`refine` as the outer loop, the true residual
in the mass-atom norm of the dual 1-forms on the residual view), one pass --
a second pass would only double the cost (Tobias 2026-09-07: the Laplacian
solves measure in the mass norm, the Hessian solve did not). MINRES's own
criterion, the residual in the preconditioner's norm, is what made the
counts of the earlier probes incomparable across preconditioners; the
direction-quality columns were the comparable ones throughout.

**The curves to 4000 iterations (job 18070181, `curve_probe_long.log`, same
state, float64, 0.12 s per iteration; Tobias: "what would happen if we
allocate more MINRES iterations?"):**

| iterations | atom: mass-atom norm | atom: $\Delta E^*$ | sandwich: mass-atom norm | sandwich: $\Delta E^*$ |
|---|---|---|---|---|
| 300 | 0.34 | 9.2e-9 | 0.36 | 6.6e-8 |
| 600 | 0.22 | 1.6e-8 | 0.29 | 9.8e-8 |
| 1000 | 0.16 | 2.7e-8 | 0.16 | 1.05e-7 |
| 2000 | 0.11 | 5.0e-8 | 0.11 | 1.09e-7 |
| 4000 | 0.075 | 9.4e-8 | 0.090 | 1.12e-7 |

1. **The exact Newton decrement at this state is 1.13e-7.** The sandwich's
   $\Delta E^*$ saturates there (93% at 1000 iterations, 99% at 4000); the
   Laplacian atom has 8% of it at 300 and 83% at 4000, still climbing. So
   with the sandwich the exact Newton direction is within reach at about
   1000 iterations (two minutes per step in float64 at (16,32,32)); with the
   Laplacian atom it is not at any affordable budget.
2. **The residual norms say nothing of this.** In the mass-atom norm both
   preconditioners follow the same power law, roughly $N^{-1/2}$ (0.34 at 300,
   0.075-0.09 at 4000), and the sandwich still stands at 0.09 when its
   direction carries 99% of the decrement: that norm is the stiff modes, which
   carry no energy. The exponential rate guessed from the 25-300 window
   (halving per 275 iterations) does not hold; the decay is algebraic.
3. **Scale of the decrement.** The Newton arms remove 1.43e-7 from step 5000 to
   the floor in total; one exact Newton step from the step-5000 state removes
   1.13e-7 of that. An exact step would reach the floor in a handful of steps;
   the Laplacian-atom arm at 8% per step needs the observed 35-40; the
   sandwich arm at 58% per step should need a few, if its truncated iterate
   is a descent direction (16% fallbacks at 32^3, see 10e).
4. **More iterations, then:** with the Laplacian atom, cost-neutral to the floor
   (measured: 100 vs 300 iterations, 118 vs 37 steps, 12.7 vs 10.3 minutes),
   because the energy per direction grows about linearly with the budget while
   the cost does too; with the sandwich, 600-1000 iterations buy the whole
   decrement and anything beyond is waste; the floor itself moves with neither.

**The factored preconditioner with the same budget (job 18071871,
`curve_probe_factored_long.log`; Tobias: "how about the factored atom, how
does that do with more MINRES iterations?"):** fraction of the exact Newton
decrement (1.13e-7) recovered, and the true residual in the mass-atom norm.

| iterations | Laplacian atom | sandwich | factored, $\epsilon$ = 0.1 | factored, $\epsilon$ = 0.01 |
|---|---|---|---|---|
| 300 | 8% / 0.34 | 58% / 0.36 | 6% / 0.39 | 4% / 0.53 |
| 1000 | 24% / 0.16 | 93% / 0.16 | 16% / 0.20 | 9% / 0.33 |
| 4000 | 83% / 0.075 | 99% / 0.090 | 59% / 0.10 | 25% / 0.155 |

Worse than the plain Laplacian atom at every count in both measures, by a
factor 1.4 ($\epsilon$ = 0.1) to 3 ($\epsilon$ = 0.01) in the energy; the
recovery climbs linearly like the atom's and does not saturate like the
sandwich's, so the factored operator captures nothing of the flat modes that
the Laplacian atom does not, it only slows the atom down (its application
costs two Laplacian atoms and two cross products, 0.145 s per iteration
against 0.12). The smaller $\epsilon$, the worse: the $\epsilon P_L$ term is
what does the work. Closed: the factored preconditioner is a negative at every
budget.

**The sandwich in a relaxation** (Tobias: "diverged pretty early"): its 16-cell
arm did not diverge, it reconnected faster than the Laplacian-atom arm (10d):
a better direction resolves the flat modes sooner. The consistent test is
(32,32,32), where the Laplacian-atom arm ran an hour without a leak: job
18067197 (`anchor32_newton_harm`).

**Why the sandwich falls short of the model, and the fix (Tobias 2026-09-07).**
The model was validated on Ritz vectors (condition 13); the sandwich is not
its inverse: (i) the symbol $(a(r) m + b(r) n)^2$ passes through zero on each
resonant surface, and a pointwise-in-$r$ scaling before and after the
Laplacian atom's radial solve commutes with it only for $r$-independent
$a, b$ -- crudest exactly on the flat modes; (ii) the symbol is applied per
component of the potential $a$, i.e. $\mathrm{curl}(h \cdot \nabla a) = h \cdot \nabla(\mathrm{curl}\,a)$
is assumed, false with shear; (iii) angle-averaged profiles ($|h^\zeta/J|$
varies 2-3x over $\theta$), a scalar floor, the Laplacian's polar core. Each
departs from the model where it earned its condition number; the 6-8x is the
high end's share.

The fix: keep the Fourier diagonalisation in the angles (the lumping earns
it) and solve the radial direction exactly per mode. For a mode $(m, n)$ the
model $\mathrm{curl}^T \Lambda\, \mathrm{curl}$ is a dense-banded $3 n_r \times 3 n_r$
matrix in $r$ on the three components of $a$ (the mode's curl, the radial
masses and metric weights, the symbol as a radial profile with its zero at
the mode's resonant radius, the curl again): assemble the ~1000 mode matrices
once (they depend on $h$, not on $B$), factor once (~20 MB), apply = two FFTs
+ a batched banded solve, SPD per mode. The polar core by the dense probe or
the extraction as now. This is the field-line atom in the form the physics
dictates, SIESTA's block structure with the modes decoupled by the lumping.
Before building it (a day): two diagnostics, one job each -- MINRES with the
exact model inverse inside (an inner solve, diagnostic only) to confirm the
count the model promises; and the $(m, n, r)$ identification of the seven
flat Ritz vectors to check the floor mode by mode.

**Dense-in-$r$ for every atom (Tobias 2026-09-07).** The metric-lumping atoms
are Kronecker sums, which need each term's weight to be a product of one-axis
profiles: the $\theta$ term's $g^{\theta\theta} J \sim 1/r$ is averaged over
$r$ and the axis geometry leaves the bulk atom -- the documented weak spot
(k=0 $\kappa \sim n^{1.7}$ on the cylinder, k=1 counts $\sim n^{1.3}$). Fourier
in the angles with angle-averaged coefficients (today's lumping) and a
banded radial matrix per mode $(m, n)$ with the exact radial dependence of
every weight, $K_r + (2\pi m)^2 M_r[w_\theta(r)] + (2\pi n)^2 M_r[w_\zeta(r)]$
for a scalar, components coupled in the block, contains the Kronecker sum as
the $r$-independent special case and removes exactly that error; the polar
core stays the dense probe. One construction for the masses, the Laplacians,
the shifted $M + \epsilon L$ (a radial diagonal per mode) and the Newton
model (its symbol is one more radial weight per mode). Storage
$n_\theta n_\zeta (3 n_r)^2$ per atom (20 MB at (16,32,32); 300 MB dense,
30 MB banded at (32,64,64) for a vector form); apply = two FFTs + a batched
banded solve; build from the existing 1-D weighted assembly, no probing. The
successor of the metric-lumping family, to earn its place on the counts mesh
by mesh, the k=0 axis problem first, the harmonic model as the first customer.

**Without dense-in-$r$ (Tobias 2026-09-07).** The sandwich earns the stiff
end; what it misses is a low-dimensional, slowly varying set of flat
directions. Deflation with recycling learns them: keep the 20-50 lowest Ritz
vectors of the preconditioned operator from each MINRES solve and deflate
them from the next (valid with the operator frozen per chunk, slowly varying
even unfrozen). No atom, no model, no separability assumption; storage plus
one orthogonalisation per iteration; adapts to shear, $u \cdot \nabla h$ and
the polar rows alike. The first thing to try: one diagnostic job, MINRES with
the sandwich plus the Lanczos-learned low space at the step-5000 field,
against the 300. Radial bands / additive Schwarz (one Kronecker sum per band) are
NOT an option: the 2-D ring atom of 2026-08-19 was that construction at its
extreme and lost badly on the outer rings (toroid k=3 free 65 vs 24, W7-X k=2
free 616 vs 200), because their value is the nonlocal radial coupling
(Steklov/DtN) no separable factor carries (Tobias 2026-09-07). The dense
radial block per mode is the opposite experience: the k=0 `radial_dense` atom
(rank 2) was exact on the cylinder and converged in ~10 iterations with
Dirichlet conditions everywhere; its free-BC stall on curved geometries was
traced to the constant nullspace not being deflated in the per-mode inverse,
a solver detail. The Newton system is the Dirichlet k=1 potential, the case
where it worked. Coordinate
changes cannot remove shear; least-squares Krylov on the induction operator
has the same count in the end; an inner exact model solve moves the same
difficulty one level down.

**Radial bands, measured (2026-09-07, jobs 18064040/18064047, float64,
li383, consistent random right-hand sides, iterations of `apply_inverse_laplacian`):**

| bands | (12,24,24) p=3: k0F / k0D / k1F / k1D / k2F / k2D / k3F / k3D | (16,32,32) p=2 |
|---|---|---|
| 1 | 127 / 180 / 2968 / 830 / 3594 / 1012 / 1188 / 884 | 128 / 186 / 3122 / 883 / 3770 / 1200 / 1106 / 785 |
| 2 | 143 / 173 / 2806 / 749 / 3573 / 925 / 1185 / 884 | 128 / 163 / 2924 / 766 / 3768 / 1027 / 1086 / 810 |
| 3 | 150 / 88 / 2840 / 742 / 3631 / 926 / 1194 / 941 | 142 / 153 / 3030 / 750 / 3905 / 988 / 1088 / 798 |
| 4 | 164 / 167 / 2728 / 725 / 3485 / 900 / 1111 / 989 | 159 / 76 / 3098 / 725 / 4107 / 973 / 1075 / 906 |

Three bands: 10-18% fewer iterations on the Dirichlet k=1, 2 solves, nothing
on the free ones, 10-30% more on k=0 free, k=3 flat; the k=0 Dirichlet drop (180 -> 88 at three bands on the p=3
mesh, 186 -> 76 at four bands on the p=2 mesh) is real but erratic in the
band count: the one place the bands find the axis geometry, and not
reproducibly. Nothing on the k=0 axis problem the bands were meant
for, and no reason to expect them to place the Hessian's resonances. Verdict:
not worth extending to the Hessian; recommend removing the option (commit
1451700 keeps it in the history). The harmonic atom's next steps are the
lumped-strain floor and, for the shear, the per-mode radial solve.

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

Consequences: (i) the auxiliary-field formulation, $E^T M_1 H = 0$ for any
$u$, is exact but not consistent for our fields (Tobias 2026-09-07): the
identity needs $H$ in the Dirichlet space of $E$, i.e. $H \times n = 0$,
while the state has $|B| \sim 1$ tangential at the wall, so the proxy
$H = M_1^{-1} P B$ differs from $B$ by order one in a one-cell wall layer
($h^{1/2}$ in $L^2$), the force $J \times H$ has a different fixed point
(the 3.4e-3 residual of the auxiliary arm at the $J \times B$ state is
that layer), and a natural $H$ leaks through the same wall DoFs (midpoint
session). Not the remedy for the Newton leak; kept for fields with
$B \times n = 0$. The remedy is resolution: at 32 radial cells the leak is
the descent's own drift (10e); (ii) the
cheap fix is not to excite those modes: smooth the Newton direction with
$(M + \mu L)^{-1} M$ before stepping (one line), or a Levenberg-Marquardt
shift in the $H^1$ metric inside the solve; (iii) the residual floor of
section 10c stands (the float64 arm at tol 1e-10 reproduced the mixed floor,
4.5e-5, to the digit: the discretisation, not the tolerance), and the
"floor finder" reading of Newton stands; what changes is the diagnosis of what
happens past it.

## 10e. Radial resolution at the surfaces: the floor and the leak (2026-09-07)

Tobias: "then we need to launch another run at higher radial resolution";
three anchors, each the anchor descent (5000 steps, mixed, m=1, smoothed) on
its mesh, then the capped shift-0 Newton with the Laplacian atom for one hour
(jobs 18045954/18050908, 18046137/18050714, 18046424/18051672;
`outputs/newton_relax/anchor32*`, `anchor_ref35*`, `anchor_ref3*`). The
rational surfaces from the VMEC profile ($\rho = \sqrt s$): $\iota = 1/2$ at
$\rho = 0.544$, $6/11$ at $0.674$, $3/5$ at $0.794$ (the seeded arms sit on the
first and third).

| mesh | descent s/step | descent sq. resid at 5000 | Newton steps / s/step | sq. resid min / end | helicity drift at min / end |
|---|---|---|---|---|---|
| (16,32,32) uniform (the anchor) | 0.67 | 5.7e-8 | 220 / 16.6 | 2.3e-9 / 8.7e-9 | -3.3e-6 / +2.4e-5 |
| (32,32,32) uniform | 1.20 | 9.7e-8 | 140 / 26.3 | 1.7e-9 / 1.7e-9 (still falling) | +1.9e-7 / +1.9e-7 |
| (28,32,32), 12 cells in [0.74, 0.85] around 3/5 | 1.13 | 7.5e-7 | 160 / 22.8 | 5.6e-9 / 7.6e-9 | -2.9e-6 / -5.6e-6 |
| (40,32,32), 8/8/12 cells around 1/2, 6/11, 3/5 | 1.33 | 6.6e-7 | 140 / 28.8 | 4.5e-9 / 4.8e-9 | -2.2e-6 / -3.2e-6 |

Readings.

1. **Doubling the radial resolution uniformly removes the pathology.** At
   (32,32,32) the residual descends monotonically through the hour, the last
   chunk is the minimum (1.7e-9, below the (16,32,32) floor and not yet a
   floor), and the helicity drift is +1.9e-7, the size of the descent's own
   drift and 160x smaller than at (16,32,32). No turnaround. The leak and the
   reconnection of 10/10d are under-resolved radial structure at the surfaces,
   and $n_r = 32$ resolves it on li383 at this beta.
2. **Windows alone do not.** Both refined-window meshes reach floors above the
   uniform (16,32,32) one (5.6e-9 and 4.5e-9 against 2.3e-9) from descents
   whose residuals are 10x higher at step 5000 (the fine cells resolve force
   structure the uniform mesh smooths over, and the descent, CFL-bound by the
   smallest cell, is slower there), and leak helicity at a fifth to a tenth
   of the uniform (16,32,32) rate. The three-window mesh is between the
   single window and the uniform 32 in every column. So the leak is not
   confined to the 3/5 surface, nor to the three named surfaces: it is
   everywhere the radial resolution is 16, which is what a uniform 32 fixes.
   Whether a window set that follows more surfaces, or a $p = 3$ radial
   basis, does the same at lower cost is the sweep this leaves open.
2b. **The descents continued for the same wall time** (jobs 18063291/98;
   the (32,32,32) one, four hours, still running): 3/5-refined, 3500 more steps
   in the hour, last-chunk residual 6.4e-4 (squared 4.1e-7) against its Newton
   arm's floor 5.6e-9, helicity -1.0e-6; three-window, 3000 steps, 6.3e-4
   (squared 4.0e-7) against 4.5e-9, helicity -8.4e-7. Seventy to ninety times
   the Newton floor after an hour of descent from the same state.
3. **Cost.** A Newton step at 32 radial cells is 26 s in mixed precision
   (against 1.2 s per descent step), and the hour reached what the descent
   would need many hours for; the floor at 32 is still unknown because the
   arm had not reached it.

**The descent at (32,32,32) continued four hours (job 18063290,
`anchor32/cont`, 12000 steps at 1.21 s, to step 17000).** Energy removed
0.40e-7 against the Newton arm's 1.31e-7 in the same four hours; lowest
chunk 1.23e-8 at step 16784, last chunk 2.05e-8 (the zigzag), helicity
+4.1e-7 at its minimum and +3.7e-7 at the end. So after four hours the
descent stands at 7.7x the Newton floor of this mesh. Between step 5000
(9.65e-8) and 16784 the squared residual falls like $t^{-1.7}$; at that rate
the floor is another 40 000 steps, some 13 hours, away, against the Newton
arm's 52 minutes. The helicity drifts of the two arms are the same size and
opposite sign (+3.7e-7 against -7.4e-7 after four hours), which is what "no
leak at 32 radial cells" means quantitatively.

**The Newton arm at (32,32,32) continued to four hours (job 18062278,
`anchor32_newton/cont`, 440 more steps at 25.5 s from the step-5140
checkpoint; 580 steps and 4.14 h in all).** The floor at 32 radial cells is
1.6e-9 (lowest chunk, at step 5192, 52 steps into the continuation), against
2.0e-9 at 16: a fifth lower for double the radial resolution. Past it the
residual creeps up, to 2.3e-9 at the end (a chunk-to-chunk rise of 3-12% at
a time, never a jump), while the energy keeps going: 1.31e-7 removed in all,
0.54e-7 of it after the floor. The helicity drift stays small, -1.5e-7 at the
floor and -7.4e-7 after four hours, a tenth of the (16,32,32) arm's drift at
its floor and 30x smaller than that arm's drift after one hour: the leak of
10d is gone at 32 radial cells, and what remains is the projection error at
the rate of the descent. So at 32 the post-floor stepping is not a
reconnection, it is the valley of 10c, walked at a helicity cost that is
tolerable and an energy gain that is real; the result of the run is still
the floor checkpoint, by the paper's convention.

**The sandwich arm at (32,32,32) (job 18067197, `anchor32_newton_harm`,
66 minutes, 160 steps at 24.8 s).** No divergence this time. Against the
Laplacian-atom arm from the same state and hour (140 steps at 26.3 s):

| arm | energy removed | sq. resid, lowest chunk / last | helicity drift | fallbacks |
|---|---|---|---|---|
| Laplacian atom | 7.7e-8 | 1.7e-9 / 1.7e-9 | +1.9e-7 | 0 |
| harmonic sandwich | 1.44e-7 | 8.4e-9 / 8.4e-9 | -1.2e-6 | 16% of the steps |

The sandwich removes 1.9x the energy in the hour and sits at a 5x higher
residual: the same split as in the fixed-norm curves (7c), energy in the flat
modes, residual in the stiff ones, now in a relaxation. Its per-step residual
is noisy (jumps of up to 7x on the steps where the truncated MINRES iterate
was not a descent direction and the step fell back to the smoothed force;
16% of the steps), so its lowest 20-step chunk is the last one, and the
helicity drift is six times the Laplacian arm's. For the floor the Laplacian
atom is the better arm; for the energy the sandwich; a converged sandwich
solve (1000 iterations, 7c) would presumably give both, at 3x the step cost,
and is the arm to run if the sandwich is pursued.

**The sandwich at a converged budget (Tobias: "run that experiment"; job
18086925, `anchor32_newton_harm1000`, 1000 MINRES iterations per step, the
direction carrying >90% of the exact Newton decrement by 7c; 100 steps at
83 s, 2.3 hours).** The prediction was a floor in a third of the steps. The
result:

| arm | steps, s/step | energy removed | sq. resid lowest chunk / last | helicity at min / end | fallbacks | mean dt |
|---|---|---|---|---|---|---|
| Laplacian atom, 300 it (four hours) | 580, 25.7 | 1.31e-7 | 1.6e-9 / 2.3e-9 | -1.5e-7 / -7.4e-7 | 0 | 0.77 |
| sandwich, 300 it (one hour) | 160, 24.8 | 1.44e-7 | 8.4e-9 / 8.4e-9 | -1.2e-6 / -1.2e-6 | 16% | 0.73 |
| sandwich, 1000 it (two hours) | 100, 83.0 | 1.30e-7 | 4.5e-8 / 8.1e-8 | -1.6e-6 / -1.2e-6 | 1% | 0.31 |
| sandwich, sqrt(tol) stop, mean 62 it (two hours; job 18078295, `anchor32_newton_harm_sqrttol`) | 1160, 6.2 | 1.61e-7 | 1.3e-8 / 6.5e-8 | -1.8e-6 / +3.0e-6 | 2% | -- |

1. **The converged direction is a descent direction** (fallbacks 16% -> 1%),
   as it must be for a positive definite system. That part of the
   prediction held.
2. **It does not find the floor; it is the worst floor of any arm on this
   mesh**, 28x the Laplacian atom's. The second step, at the full Newton
   length, raised the squared residual 30x (1.1e-7 -> 3.5e-6) while lowering
   the energy; after that the line search took a mean step of 0.31 and the
   residual never came back below 2.4e-8 per step. This is 10c.4 at
   full strength: the exact direction is the inverse Hessian on the force,
   concentrated in the flat modes, and a full step along it moves the
   sheet-scale structure by a cell, outside the validity of the quadratic
   model. The truncated Laplacian-atom direction never contains those modes
   and steps safely; the truncated sandwich contains them partly; the
   converged one is nothing but them.
3. **The ordering of the four arms is the ordering of their flat-mode content**:
   the more of the decrement a direction carries, the more energy per step it
   removes and the worse its floor and helicity. The sqrt(tol) arm (62 cheap
   iterations, 1160 steps) removed the most energy of all and floored
   nowhere, with a helicity that turned positive at the end (a reconnection
   starting).
4. **Closed.** For the floor the Laplacian atom at 300 iterations is the best
   Newton arm at every budget tried; the harmonic preconditioner is a
   documented result, not a candidate. What a converged Newton step would need
   to be useful is a trust region on the flat modes, i.e. the shift family,
   which was measured to leave the flat modes to the gradient and lose to the
   descent (10). That circle is closed too.

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

**The auxiliary-field pair is off the plan** (the $H \times n = 0$ inconsistency
above); the plan for exact-enough helicity along Newton paths is radial
resolution at the surfaces, 10e, with the (32,32,32) continuation (job
18062278, three hours) as the number behind it.

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

## 14. The convergence section of the paper (2026-09-08)

Tobias: "I want to write a section on the convergence of relaxation methods
... 0) current sheets 1) the continuous case 2) the discrete case, the
floor, precision, solver tolerance 3) Newton 4) numerical results ...
Completely re-write it ... Be brief ... reduce the table to the main points
(newton, tolerance, resolution)". Written as
`outputs/newton_second_variation/convergence_section.tex` (five paragraphs,
`tab:convergence` with 11 rows, `fig:convergence`), citing only keys the
paper already uses (loizu_magnetic_2015, grad_toroidal_1967,
constantin_flexibility_2021, hudson_computation_2012, parker_spontaneous_1994,
moffatt_magnetostatic_1985, beekie_moffatts_2022, bae_local_2025,
cieliebak_note_2017, enciso_obstructions_2025) and the appendix
`sec:second_variation`; the hyperparameter-sweeps subsection has no label, so
the tolerance-1e-6 event is referenced by its figure `fig:h_p_m_tol_sweep_F`.

Results added there that the earlier block did not have: the descent rows at
tol 1e-6 / 1e-8 / 1e-10 from step 5000 to 10000 (the 1e-6 arm's re-ordering
event: energy 50.5e-7 against 0.63e-7, helicity +3.6e-5; 1e-8 and 1e-10 the
same to the residual floor and the energy), the power-law exponents, the
harmonic velocity's share, the converged-direction result, the sheet-width
scaling $\|F\|_{L^2} \sim \delta^{-1/2}$. The tolerance rows use the collected
sweep arms (`figures_2026-09/sweeps_tol.json`, `sweeps_h.json`) with the
same 20-step-chunk convention as the Newton rows, which changes the
descent-tol-1e-8 row's residual columns from the earlier 2.15 / 2.15 (a
last-1000-step mean) to 1.68 / 2.20. The float64 descent (floor study
`f64_g1`) is in the figure but not in the table: its run file predates the
energy diagnostic.

The figures (`newton_convergence_figure.py` -> `newton_convergence.json`,
`newton_convergence_steps.png`, `newton_convergence_wall.png`, `pgf/`; copies
in `figures_2026-09/`; Tobias's revision 2026-09-08: no tol-1e-6 arm, only
(16,32,32) and (32,32,32) in the resolution panel, linear x, one figure
against the step and one against the wall time, no block means "so it is
clear that we start from the same run"): the squared residual per step, raw,
descent solid / Newton dashed, the Newton arms branching off the descent at
step 5000 (on the wall axis offset by the descent's stepping wall at that
step; setup excluded for every arm), colour = tolerance or precision (left)
and mesh (right), legends below the panels. The json holds every arm in the
sweeps' collected format (params, trace, qoi with cumulative wall over the
continuations, summary) plus `start_step` and `t0_min`.
