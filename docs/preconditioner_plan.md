# Matrix-Free Preconditioning Strategy for Tensor-Spline Mass Matrices
### Algorithmic Blueprint & Implementation Plan

## Executive Summary
This document outlines a highly efficient, matrix-free preconditioning strategy for solving both scalar and vectorial mass matrices arising from tensor-spline discretizations. By avoiding dense multi-dimensional matrices and inner iterative Krylov methods, the strategy strictly maintains Symmetric Positive Definiteness (SPD) while minimizing memory overhead and computational cost.

## 1. The Core Strategy: Greville Collocation
Instead of relying on high-rank algebraic approximations (which are difficult to invert and costly to apply), this strategy geometrically separates the topological spline connectivity from the macroscopic geometry.

By evaluating the geometric Jacobian at the **Greville abscissae** (the natural spline nodes), the preconditioning matrix $P$ is formulated as:
$$P = D^{1/2} M_0 D^{1/2}$$
where:
* $M_0 = M_{0x} \otimes M_{0y} \otimes M_{0z}$ is the *unweighted* tensor-spline mass matrix.
* $D$ is a diagonal matrix containing the geometric weights (Jacobian determinants) evaluated at the Greville points.

Its inverse, applied during the Preconditioned Conjugate Gradient (PCG) solver, is simply:
$$P^{-1} = D^{-1/2} (M_{0x}^{-1} \otimes M_{0y}^{-1} \otimes M_{0z}^{-1}) D^{-1/2}$$

## 2. The Implementation Plan (Scalar Case)

### Phase A: Setup (Precomputation)
1. **1D Spline Inverses:** Form the standard, 1D unweighted mass matrices ($M_{0x}, M_{0y}, M_{0z}$). For a grid of size $100 \times 100 \times 100$, these are small $100 \times 100$ matrices. Compute their exact dense inverses (e.g., via Cholesky) and cache them.
2. **Geometric Weights:** Evaluate the Jacobian $f(x,y,z)$ at the 3D Greville points. Store the point-wise scaling array: $D^{-1/2} = 1 / \sqrt{f(x, y, z)}$.

### Phase B: Matrix-Free Execution (per PCG step)
To apply $w = P^{-1}v$ to a residual vector $v$:
1. **Initial Scaling:** $v^* = v \odot D^{-1/2}$ (element-wise multiplication).
2. **1D Tensor Contractions:** Apply the cached dense 1D inverses sequentially along each fiber using fast Matrix-Vector products:
   $$w^* = M_{0z}^{-1}(M_{0y}^{-1}(M_{0x}^{-1} v^*))$$
3. **Final Scaling:** $w = w^* \odot D^{-1/2}$.

## 3. Vectorial Extension (Diagonal Metric)
Assuming a diagonal metric tensor $g$, the vector components completely decouple, resulting in a strictly block-diagonal mass matrix. The scalar Greville trick is applied three separate times using component-specific geometric weights.

### Setup Modifications
Instead of just $f(x,y,z)$, compute three separate diagonal scaling arrays incorporating the diagonal entries of the metric tensor:
* $(D_1)_{ii} = 1 / \sqrt{g_{11} \cdot f(x_i,y_i,z_i)}$
* $(D_2)_{ii} = 1 / \sqrt{g_{22} \cdot f(x_i,y_i,z_i)}$
* $(D_3)_{ii} = 1 / \sqrt{g_{33} \cdot f(x_i,y_i,z_i)}$

The exact same unweighted 1D inverses ($M_{0x}^{-1}, \dots$) are reused for all three blocks.

### Execution Modifications
For a vectorial residual $v = [v_1, v_2, v_3]^T$:
1. Pass $v_1$ through the pipeline using $D_1^{-1/2}$ to yield $w_1$.
2. Pass $v_2$ through the pipeline using $D_2^{-1/2}$ to yield $w_2$.
3. Pass $v_3$ through the pipeline using $D_3^{-1/2}$ to yield $w_3$.
4. Assemble the preconditioned vector: $w = [w_1, w_2, w_3]^T$.

---

## Core Advantages
* **Strict Positivity:** Using the square root of the positive Jacobian guarantees the preconditioner remains perfectly Symmetric Positive Definite (SPD), keeping PCG stable.
* **Minimal Memory Footprint:** For a $100^3$ grid, caching requires only three $100 \times 100$ matrices ($\sim 240$ KB total) and 1D scaling arrays. Zero dense or sparse 3D matrix storage is needed.
* **Optimal Speed:** Execution reduces to pure dense matrix-vector multiplications along contiguous 1D fibers, scaling linearly as $\mathcal{O}(N)$ without any inner iterative loops.

---

# Part II — Implementation & Findings (2026-06)

Greville is applied **only to the bulk block**; the production surgery-Schur split
handles the polar axis separately, so the singular polar splines never enter the
tensor inverse.

## A. What is implemented + production config

**Mass $k=0,1,2,3$** — `scripts/debug/greville_bulk_precond.py` (iters),
`greville_bulk_speed.py` (jitted timings). Component-driven; per-component weight
$D$ collocated at that component's Greville abscissae (degree $p$ on primal axes,
$p-1$ on the differentiated axis):

| block | weight $D$ |
|---|---|
| $k=0$ | $J$ | $k=3$ | $1/J$ |
| $k=1$ comp $i$ | $J\,g^{ii}$ | $k=2$ comp $i$ | $g_{ii}/J$ |

**Laplacian $k=0$** — wired into the *production* k=0 Hodge preconditioner
(`assemble_tensor_laplacian_preconditioner(..., cp_kwargs={'greville':True})`),
reusing the surgery-Schur envelope and jitted `solve_singular_cg`. Bulk inverse =
exact additive fast-diagonalisation `_fd_apply_3d` on **unweighted** 1-D atoms,
sandwiched by $D^{\pm1/2}$ with directional constants:
$$P^{-1}=D^{-1/2}\Big(\textstyle\sum_a\alpha_a\,\text{atom}_a\Big)^{-1}D^{-1/2},\qquad
\text{denom}[i,j,k]=\alpha_r\lambda_r[i]+\alpha_t\lambda_t[j]+\alpha_z\lambda_z[k].$$
**Production config (`weight_mode="combined"`):** $D=\sqrt[3]{\alpha_{rr}\alpha_{\theta\theta}\alpha_{\zeta\zeta}}$
(geometric mean across channels), $\alpha_a=\langle\alpha_{aa}/D\rangle$ (arithmetic
mean). Head-to-head script: `scripts/debug/greville_laplacian_real_k0.py`. (See §D/§E
for why these particular means.)

## B. Two Greville-point gotchas (both fixed)

1. **Differentiated-axis double point.** Do *not* read the differentiated-axis Greville
   points from `dΛ[axis].s` — that inner basis carries the parent's degree-$p$ knots
   while declaring degree $p-1$, so a clamped endpoint gets a *spurious double Greville
   point* and a genuine bulk DOF appears to sit at $r=0$. Build a fresh
   `SplineBasis(dΛ.n, dΛ.p, dΛ.type)` (clamped → one point per endpoint, rest interior;
   periodic → all interior). Symptom: k=1 `arr` 56 it / `bad_D=72`; after fix 7 it / 0.
2. **Spline-map $J=0$ at the clamped boundary.** A spline (e.g. W7-X) map's clamped
   `evaluate()` has a constant branch at $x=T[-1]$, so `jacfwd` at exactly $r=1$ gives
   $\det=0$ over the whole $r=1$ Greville layer (288/2880 on W7-X; analytic maps fine).
   Fix: clip clamped-axis Greville coords to $[10^{-7},1-10^{-7}]$; median-floor any
   residual non-positive $D$ as a genuine-fold safety net.

## C. Results

**Mass — robust everywhere, `bad_D=0` in every cell** (cyl/toroid/W7-X, $p\in\{1..5\}$,
$n_s\in\{8^3,12^3,16^3\}$, free BC). Max CG iters per block:

| | k0 | k1 | k2 | k3 |
|---|---|---|---|---|
| cylinder | 8 | 9 | 9 | 8 |
| toroid | 12 | 14 | 14 | 10 |
| w7x | 27 | 136 | 96 | 18 |

W7-X scalar (k0,k3) is strong; vector (k1,k2) degrades at high $p$ (worst k1 `arr` p5
= 136, the hard $rR$ weight) but **converges where NTF rank-2 NaN'd**. Per-iteration
cost ≈ tensor (~6 einsums); on the mass **greville ≈ tensor**, ~40–50× over jacobi.

**Laplacian $k=0$ — real Schur+jit head-to-head (avg iters):**

| geom / BC | jacobi | tensor (r1 FD) | grev-combined |
|---|---|---|---|
| toroid dbc | 202 | **11** | 42 |
| toroid free | 336 | **18** | 59 |
| w7x dbc | 227 | 45 | **81** |
| **w7x free** | 414 | **5664 (stalls ✗)** | **128 ✅** |

**Greville wins decisively where it matters:** on W7-X free BC the production tensor
atom stalls (5664 it, fails); grev-combined converges in ~128. Tensor still wins on
smooth geometry (its rank-1 CP fits the smooth weight). grev-combined is robust but
**not $h$-independent** (toroid free $59\to81$, W7-X free $128\to178$ from $12^3\to16^3$)
— the single-$D$ residual analysed in §D. The A/B ablation showed the directional
constants $\alpha$ are *essential* (common-$D$-only `grev-spatial` diverges) while the
spatial $D$ sandwich itself adds little beyond them.

## D. The structural limit (why no single-$D$ atom is $h$-independent)

**The operator.** Diagonal-metric bulk stiffness: $a(u,v)=\int\sum_a\alpha_{aa}\,\partial_a u\,\partial_a v$
with $\alpha_{aa}=Jg^{aa}$. FD inverts $L$ in $O(n)$ einsums **iff** it has the canonical
form
$$L=K_r\!\otimes\!M_\theta\!\otimes\!M_\zeta+M_r\!\otimes\!K_\theta\!\otimes\!M_\zeta+M_r\!\otimes\!M_\theta\!\otimes\!K_\zeta\tag{$\star$}$$
with **exactly one $K_a$ and one $M_a$ per axis** (the same $M_r$ in both the
$\theta\theta$- and $\zeta\zeta$-terms, etc.); then one generalised eig per axis
$K_a v=\lambda M_a v$ gives the additive denominator $\lambda_r+\lambda_\theta+\lambda_\zeta$.

**Only two weightings preserve ($\star$):** (1) a single common spatial $D(\xi)$ as a
pointwise sandwich $D^{1/2}L_0D^{1/2}$ (the Greville move — a diagonal acting
basis-by-basis, leaving the 1-D atoms unweighted); (2) per-direction scalar constants
$\alpha$. Combined, the most general exactly-FD-invertible operator from one basis is
$\alpha_{aa}(\xi)=\alpha_a\,D(\xi)$.

**Theorem.** Exact FD from one basis $\iff$ the channel ratios
$\alpha_{rr}:\alpha_{\theta\theta}:\alpha_{\zeta\zeta}$ are **constant in space**.
*(If the $\theta\theta$- and $\zeta\zeta$-terms carried different radial weights they
would be two distinct $M_r$ matrices, and no single $V_r$ diagonalises both — a pencil
$(K,M)$ resolves two matrices, not three.)* So:
- The **mass** is a single channel → trivially $W=c\,D$ → exactly FD-able everywhere (κ≈1.3).
- The **Laplacian** has three channels whose ratios are *not* constant. Measured
  (`scripts/debug/laplacian_radial_profiles.py`, pure metric eval): clean power laws on
  **all** geometries — $\alpha_{rr}\sim r^{+1}$, $\alpha_{\theta\theta}\sim r^{-1}$,
  $\alpha_{\zeta\zeta}\sim r^{+1}$, so $\alpha_{rr}/\alpha_{\theta\theta}\sim r^2$ over
  4–5 decades. **No single $D$+$\alpha$ can represent a varying ratio** — the irreducible
  residual that makes grev-combined grow with resolution.

**Angular spread decides where radial methods could help.** The same diagnostic measures
the per-channel angular variation (std/mean at the edge): cylinder **0%** (purely
radial), toroid **~24%**, **W7-X ~60% (first-order)**. So on W7-X the angular variation
is irreducible by *any* purely-radial method, and the full pointwise $D$ (grev-combined)
is the correct base; further W7-X gains require an explicit angular term, not a radial one.

## E. Means for $D$ and $\alpha$ (settled empirically, 2026-06-27)

- **$D$ = geometric mean *across the 3 channels*** per point, $\sqrt[3]{\alpha_{rr}\alpha_{\theta\theta}\alpha_{\zeta\zeta}}$.
  Balanced; arithmetic would let the largest channel ($\sim r$) dominate $D$'s shape
  (= the rejected pair_d, measured worse), harmonic the smallest.
- **$\alpha_a$ = arithmetic mean** of $e_a=\alpha_{aa}/D$ over Greville points. The
  α-reduction sweep is decisive:

  | reduction | cyl/toroid dbc | W7-X $12^3$ | W7-X $16^3$ |
  |---|---|---|---|
  | **arith** | 40–58 | **81 / 129 ✅** | **113 / 178 ✅** |
  | geom | 45–66 | fail | 132 / 197 ⚠️ flaky |
  | minimax $\sqrt{\max\cdot\min}$ | 37–53 (best) | fail | fail ✗ |

  geom/minimax are ~5–10% better on smooth geometries but **break on W7-X**: the shaped
  metric gives $e_a$ extreme outliers, wrecking the log-average (geom, near-zeros) and
  the extremes-center (minimax, single extreme); geom is even resolution-flaky. Arithmetic
  is bulk-dominated and the **only reduction robust at all W7-X resolutions** — the
  production choice. Default `arith`; `geom`/`minimax` reachable via
  `cp_kwargs["greville_alpha_reduce"]`.

## F. Approaches explored and rejected

All target the §D residual (the per-channel radial divergence); all are **smooth-geometry
luxuries that fail on W7-X** and so are not the production path. Kept reachable as
`weight_mode=...` for the record.

- **radial-dense** (`radial_dense`): keep $r$ dense — diagonalise only $\theta,\zeta$
  and invert the exact dense $n_r\times n_r$ block $K_r[\rho_{rr}]+\lambda_\theta M_r[\rho_{\theta\theta}]+\lambda_\zeta M_r[\rho_{\zeta\zeta}]$
  per angular mode (the three radial matrices coexist — *never* co-diagonalised, which
  is how the $r,r,1/r$ non-proportionality is handled). **Exact & $h$-flat on smooth
  geometries** (1 it cylinder incl. free BC, 13 it toroid dbc) but **NaNs on W7-X** (dense
  breakdown on shaped $\rho$) and has a curved-geometry constant-null-mode bug on free BC.
  Also drops the 60% W7-X angular variation.
- **radial-Sylvester** (`radial_sylvester`): the structured form of radial-dense —
  co-diagonalise the two shared-profile channels, one symmetric eig per $\zeta$-mode for
  the third ($S_{jk}^{-1}=P_k\,\mathrm{diag}(1/(g_k+\lambda_\theta[j]))P_k^\top$).
  $n_\theta\times$ cheaper setup, cleaner null handling (per-$(j,k)$-block deflation).
  **Reproduces radial-dense exactly on dbc** but inherits the same W7-X NaN. Smooth-only.
- **radial-pencil** (`radial_pencil`): cheap robust attempt — co-diagonalise two channels,
  keep only the third's diagonal. **Worse than grev-combined everywhere and grows with
  resolution** — the dropped off-diagonal is first-order *because* $1/r$ vs $r$ are
  maximally different (the same fact that forces $r$ dense). Dead.
- **pair_d** ($D=\sqrt{\alpha_{rr}\alpha_{\zeta\zeta}}$): make the two $\sim r$ channels
  spatially exact, accept $\theta\theta$ wrong by $r^2$. **Worse than combined** (cyl 74 vs
  40) — the geomean $D$ balances all three better than nailing two and butchering one.

**Net:** grev-combined (full pointwise $D$ + arithmetic $\alpha$) is the robust production
$L_0$ atom; the radial-exact family cannot beat it on W7-X (angular) and isn't robust there.

## G. Open work

- **Laplacian $k=1,2,3$** — inherit the grev-combined $L_0$ atom through the existing
  Hiptmair–Xu machinery ($P_A+P_B$, gradient projection $\Pi$, nested Chebyshev
  $L_{k-1}^{-1}$); no fresh per-$k$ construction needed.
- **Mass production integration** — wire the bulk-only debug mass
  (`greville_bulk_precond.py`) into the production apply path for end-to-end
  (surgery-Schur + jit) solves/timings against the stored tensor/jacobi baselines.
- **W7-X angular term** — if better than grev-combined's W7-X iters is needed, add a few
  $(\theta,\zeta)$ cross-section terms / a $\zeta$-Fourier-block atom (the 60% residual,
  cf. [[metric-weight-separability-rule]]). This, not a radial method, is the W7-X lever.
- **Setup-cost accounting** — Greville pays one extra geometry evaluation at the Greville
  points (W7-X ~30–100 s, one-time); tabulate against the tensor CP-fit setup.
- **(optional, low priority)** fix radial-dense's curved-geometry null mode for an $h$-flat
  *smooth-geometry* atom — deprioritised since W7-X needs the angular term regardless.
