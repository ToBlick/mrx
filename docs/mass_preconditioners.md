# Mass Preconditioners

This note is the paper-ready writeup of the mass-block preconditioning
strategy used in MRX, the family of preconditioners we compare, and the
benchmark protocol used to evaluate them. The later sections (§6 onwards)
keep the development history — forward-model diagnostics, analytic-prior
experiments, and earlier sweeps — as supporting material.

The production policy is the tensor route with eager default rank `3` across
all four mass blocks. Formulas below are stated for one mass block at a
time; the same machinery applies for all four de Rham degrees
$k \in \{0, 1, 2, 3\}$.

## 1. Problem Setting

For each de Rham degree $k$, MRX assembles the symmetric positive
(semi-)definite mass operator

$$
M_k \in \mathbb{R}^{N_k \times N_k}, \qquad
(M_k)_{ij} = \int_{\Omega} \langle \Lambda_i^{(k)},\, \Lambda_j^{(k)} \rangle_{g}\,
\sqrt{\det g}\, d\xi,
$$

where $\{\Lambda_i^{(k)}\}$ are the mapped tensor-product B-spline basis
functions of the $k$-form space and $g$ is the metric induced by the smooth
chart $F : \hat\Omega \to \Omega$ on the logical cube
$\hat\Omega = [0,1]^3$. After polar/extraction surgery the assembled
operator $M_k$ is split block-wise into a small set of *surgery rows* (the
polar axis and periodicity identifications) and a large *bulk* block whose
basis is a true tensor product on $\hat\Omega$.

We solve $M_k u = f$ with conjugate gradients, preconditioned by an operator
$P_k^{-1}$ that we build matrix-free. Throughout, the iteration count and
wall-clock time are dominated by the quality of $P_k^{-1}$.

## 2. Preconditioner Family

We compare three preconditioner families, all matrix-free:

1. **Jacobi.** Inverse of the assembled mass diagonal,
   $P_{\mathrm{J}}^{-1} = \operatorname{diag}(M_k)^{-1}$.
2. **Polynomial (Chebyshev) on Jacobi.** A degree-$s$ Chebyshev polynomial in
   $M_k$ using $P_{\mathrm{J}}^{-1}$ as the inner smoother.
3. **Tensor (CP) preconditioner.** A separable rank-$r$ approximation of the
   bulk block combined with an exact dense solve on the surgery rows via a
   Schur complement. An optional Chebyshev acceleration of the bulk block,
   running $t$ Chebyshev steps in the bulk operator with the rank-$r$
   tensor apply as the inner smoother, sits inside the same envelope.

The first two are baselines; the third is the proposed method. All three
expose the same matrix-free interface to the outer CG.

### 2.1 Jacobi

The diagonal entries of $M_k$ are extracted at assembly time. The
preconditioner is one elementwise multiply per apply. It is the natural
cheap reference: its setup cost is negligible and it sets the floor for "do
nothing clever".

### 2.2 Chebyshev on Jacobi

For a target degree $s \ge 1$, we build the Chebyshev iteration that
approximates $M_k^{-1}$ on the spectrum interval
$[\lambda_{\min}, \lambda_{\max}]$ of the Jacobi-preconditioned operator
$P_{\mathrm{J}}^{-1} M_k$. Concretely, we

1. estimate $\lambda_{\min}, \lambda_{\max}$ once at setup using a short
   Lanczos run on $P_{\mathrm{J}}^{-1} M_k$ (with a small inflation /
   deflation safety margin),
2. emit the standard three-term Chebyshev recursion that, after $s$
   applications of $P_{\mathrm{J}}^{-1}$, returns an approximate solve of
   $M_k$ to a polynomial-residual tolerance set by $s$ and the conditioning
   of $P_{\mathrm{J}}^{-1} M_k$.

This is the standard sparse-matrix textbook baseline. It costs $s$ Jacobi
applies and $s$ matrix-vector products per outer CG iteration.

### 2.3 Tensor (CP) preconditioner

The mass block on the bulk reads

$$
(M_k^{\mathrm{bulk}} v)_{ijk} = \int_{\hat\Omega}
W^{(k)}(\xi)\, \Lambda_i(\xi_1)\Lambda_j(\xi_2)\Lambda_k(\xi_3)\,
\big[\Lambda \otimes \Lambda \otimes \Lambda\, v\big](\xi)\, d\xi,
$$

where the *coefficient field* $W^{(k)}$ collects the geometry- and
form-dependent factors (Jacobian, metric components). The active diagonal
coefficient fields are:

- $k = 0$: $J$,
- $k = 1$: $J g^{rr}$, $J g^{\theta\theta}$, $J g^{\zeta\zeta}$,
- $k = 2$: $g_{rr} / J$, $g_{\theta\theta} / J$, $g_{\zeta\zeta} / J$,
- $k = 3$: $1 / J$.

The exact $W^{(k)}$ is not separable, but on the rotating-ellipse and
toroidal mappings of interest it admits an accurate low-rank
canonical-polyadic (CP) approximation

$$
W^{(k)}(\xi) \approx \sum_{\rho=1}^{r}
w^{(k,\rho)}_1(\xi_1)\, w^{(k,\rho)}_2(\xi_2)\, w^{(k,\rho)}_3(\xi_3).
$$

For each separable summand, the bulk mass operator factors as a Kronecker
product of three 1-D mass-like operators, each banded. Inverting a single
rank-1 summand is therefore a sequence of three banded 1-D solves; the
rank-$r$ inverse uses a shared modal basis on the three axes followed by a
small dense solve in the modal coefficients.

The surgery rows are *not* approximated. Let $S$ be the set of surgery dofs
and $B$ the bulk dofs. We block $M_k$ as

$$
M_k = \begin{pmatrix} M_{SS} & M_{SB} \\ M_{BS} & M_{BB} \end{pmatrix},
$$

apply the tensor approximation only to $M_{BB}$, and wrap the surgery rows
in an exact dense Schur complement
$\Sigma = M_{SS} - M_{SB} M_{BB}^{-1} M_{BS}$. The full preconditioner is

$$
P_{\mathrm{T}}^{-1} = \begin{pmatrix}
\Sigma^{-1} & -\Sigma^{-1} M_{SB} M_{BB}^{-1} \\
-M_{BB}^{-1} M_{BS} \Sigma^{-1} &
M_{BB}^{-1} + M_{BB}^{-1} M_{BS} \Sigma^{-1} M_{SB} M_{BB}^{-1}
\end{pmatrix},
$$

with $M_{BB}^{-1}$ replaced by the rank-$r$ tensor apply. The dense Schur
block is small because the number of surgery dofs is independent of the
mesh resolution.

**Optional bulk Chebyshev acceleration.** Because the rank-$r$ tensor apply
is itself an approximate inverse of $M_{BB}$, we may sharpen it with a
Chebyshev polynomial in $M_{BB}$ that uses the tensor apply as its inner
smoother. The number of bulk Chebyshev steps is denoted $t$; $t = 0$ means
the tensor apply is used as-is. This is the natural way to trade extra
matvecs against fewer outer CG iterations once the tensor model is already
strong.

The tensor route exposes two hyperparameters:

- the CP rank $r$, controlling the fidelity of the separable model of
  $W^{(k)}$;
- the number of bulk Chebyshev steps $t$, controlling whether and how much
  the tensor apply is polished by Chebyshev.

For $k \in \{1, 2\}$ the bulk further splits into three vector components,
each treated as its own block; the Schur complement against the surgery
rows is handled identically. We expose a binary `inner_schur` toggle that
controls whether those three bulk components are coupled by an additional
inner Schur (`on`) or treated independently (`off`). Empirically `off` is
faster and equally robust on the tested geometry, so we use `off` in this
study.

## 3. Baselines

For paper-quality comparisons we use two sparse-matrix baselines and the
proposed tensor route. All three are wrapped in the same outer CG with
identical tolerances.

| Label | Family | Setup | Per-apply cost |
|---|---|---|---|
| `jacobi` | Jacobi | one diagonal extraction | one elementwise multiply |
| `cheb_J(s)`, $s \in \{2, 3, 5\}$ | Chebyshev on Jacobi | one Lanczos pass to bound the spectrum | $s$ Jacobi applies + $s$ matvecs |
| `tensor(r, t)`, $r \in \{1, 2, 3, 5\}$, $t \in \{0, 2, 3\}$ | CP + Schur | one greedy CP fit; one dense Schur factor | three banded 1-D solves per rank summand + optional $t$ bulk-Chebyshev steps |

The first two represent what a competent practitioner would reach for given
the assembled sparse $M_k$. The Chebyshev family covers the standard
"make Jacobi a polynomial preconditioner" trick; we sweep its degree $s$
to expose the polynomial-degree / iteration-count tradeoff.

We deliberately do not include incomplete-Cholesky or AMG baselines: both
require an explicit sparse $M_k$, which MRX never assembles for the bulk
block in the production solve path. They would not be a like-for-like
comparison against a matrix-free preconditioner.

## 4. Sweep Strategy

The benchmark is split into two phases.

### 4.1 Phase 1 — fixed reference point

A single de Rham sequence is built once at the reference geometry and
discretization, and then reused across the entire Cartesian sweep. This
isolates the *preconditioner* axes from the geometry / discretization axes
and keeps the cost of the study modest.

**Reference point.**

| Parameter | Value |
|---|---|
| Resolution $n_r \times n_\theta \times n_z$ | $16 \times 32 \times 16$ |
| Spline degree $p$ | $3$ |
| Aspect ratio $\varepsilon$ | $0.2$ |
| Major radius $R_0$ | $3.0$ |
| Mapping | toroidal, polar at the axis |
| Boundary | free |
| Form degrees $k$ | $\{0, 1, 2, 3\}$ |
| RHS regularity (Besov $s$) | $1$ |
| RHS seeds | $3$ independent draws |
| Outer CG tolerance | $10^{-12}$ |
| Outer CG max iterations | $1000$ |

**Cartesian grid.** Per $k$:

- baseline `jacobi`,
- baseline `cheb_J(s)` with $s \in \{2, 3, 5\}$,
- tensor `tensor(r, t)` with $r \in \{1, 2, 3, 5\}$ and $t \in \{0, 2, 3\}$,

against three independent right-hand sides drawn from a Besov-$s = 1$
random ensemble of fixed Fourier truncation. Per $k$ this is 16 cells
(1 Jacobi + 3 Chebyshev + 12 tensor); times four $k$ values, $64$ cells
total.

**Reported quantities.** For each cell:

- average and maximum CG iteration count over the three RHS draws,
- average wall-clock solve time (ms),
- one-time preconditioner setup cost (ms),
- final relative residual $\|M_k u - f\|_2 / \|f\|_2$.

The output is a tidy CSV with one row per cell. Phase 1 simultaneously
answers two questions: (i) does the tensor route beat both baselines on a
relevant reference problem, and (ii) which $(r, t)$ pair is the best
finalist for each $k$.

The driver is
[`scripts/benchmark_phase1_mass_baseline_vs_tensor.py`](../scripts/benchmark_phase1_mass_baseline_vs_tensor.py).

### 4.2 Phase 2 — scaling

Phase 2 now keeps one locked comparison set per $k$ from the Phase-1 results:

- the `jacobi` baseline,
- the `cheb_J(s)` baseline with $s = 3$,
- the tensor preconditioner with rank $r = 3$ and bulk Chebyshev disabled
  ($t = 0$).

These three strategies are then evaluated along five 1-D scaling axes with
the same driver logic in
[`scripts/benchmark_phase2_geometry.py`](../scripts/benchmark_phase2_geometry.py)
and the matching SLURM launcher
[`slurm/job_phase2_geometry.sh`](../slurm/job_phase2_geometry.sh). Each
Phase-2 axis varies one parameter at a time while holding the others at the
reference point

$$
(\kappa, \varepsilon, p, n_s, n_{\mathrm{fp}}, s)
= (1.25, 0.2, 3, (16, 32, 16), 3, 1).
$$

| Axis | Variable | Range | Reassemble? |
|---|---|---|---|
| 2a Aspect ratio | $\kappa$ | $\{1.0, 1.25, 1.5, 1.75\}$ | yes |
| 2b Minor radius | $\varepsilon$ | $\{0.1, 0.2, 0.33, 0.5\}$ | yes |
| 2c Spline order | $p$ | $\{1, 2, 3, 4\}$ | yes |
| 2d Resolution | $n_s$ | $\{(8,16,8), (16,32,16), (32,64,32), (64,128,64)\}$ | yes |
| 2e Right-hand side regularity | Besov exponent $s$ | $\{0, 1, 2, 3\}$ | yes |

The central baseline cell is run once and any axis entry equal to its
baseline value is deduplicated. `nfp` is not swept in Phase 2; it remains
fixed at $n_{\mathrm{fp}} = 3$. The CSV schema is shared across phases so the
same aggregator produces iteration- and time-vs-axis plots and the headline
tables.

### 4.3 Right-hand sides

All right-hand sides are drawn from a band-limited random Besov ensemble
parameterized by a smoothness exponent $s$, a Fourier upper limit, a number
of modes, and a normalization sample count. Holding the seed list fixed
across cells of the sweep makes the per-cell numbers directly comparable;
$s$ is one of the five Phase-2 axes. A higher $s$ produces smoother
right-hand sides and easier solves; a lower $s$ stresses the high-frequency
end of the spectrum and is where Jacobi is expected to weaken.

### 4.4 Scope

The sweep is designed to (i) establish a clean win for the tensor route on
a realistic reference problem against two textbook baselines, and (ii)
characterize how the win scales along the five axes that matter most for the
current study: $\kappa$, $\varepsilon$, spline order, mesh resolution, and RHS
regularity. It is deliberately not a fishing expedition over every CP rank,
Chebyshev degree, and damping parameter; only Phase 1 carries that
responsibility, and Phase 2 narrows to the three named strategies per $k$ to
keep the scaling plots interpretable.

### 4.5 Current sweep takeaways

The Phase 1 and Phase 2 sweeps justify the production defaults more cleanly
than the earlier small-case diagnostics.

1. **Phase 1 establishes a decisive tensor win at the reference point.** On
  all four mass blocks, the tensor route beats both whole-matrix baselines in
  CG iterations and in solve time at the reference geometry. The strongest
  raw iteration counts often come from either a larger rank or extra bulk
  Chebyshev polishing, but those gains usually do not survive wall-clock
  accounting once the tensor apply is already strong.
2. **`cheb_J(s = 3)` is the right locked polynomial baseline.** Increasing the
  Chebyshev degree beyond `3` continues to lower iteration counts, but the
  extra matrix-vector work usually erases the benefit in solve time. So Phase
  2 fixes the polynomial comparator at `s = 3` rather than chasing the
  iteration minimum.
3. **`tensor(r = 3, t = 0)` is the best production compromise.** Rank `1` is
  already strong on the scalar blocks and useful on the vector blocks, while
  ranks `2` and `3` capture most of the remaining gain. Rank `5` can still
  improve some scalar reference cases, but the improvement is small relative
  to the added setup and model complexity. The rank-`3`, no-polish tensor
  model is therefore the robust default across all four mass degrees.
4. **For `k = 1, 2`, `inner_schur = on` is an iteration-count knob, not a
  time-to-solution winner.** The coupled inner Schur sometimes lowers outer
  iteration counts on the rotating-ellipse family, but its bulk apply is much
  more expensive. In the current sweeps the uncoupled bulk model
  (`inner_schur = False`) is consistently faster in wall-clock time and is the
  production choice.
5. **The `\kappa` axis is the clearest stress test of the tensor model.** As
  `\kappa` increases, the tensor advantage narrows more than it does along the
  `\varepsilon`, `p`, or Besov-regularity axes. This is consistent with the
  rotating-ellipse map becoming less orthogonal and less tensor-separable away
  from `\kappa = 1`. Even so, on the completed moderate-resolution Phase-2
  points the rank-`3` tensor route remained the best solve-time strategy.
6. **At large resolution, setup and assembly become the practical bottleneck.**
  The most aggressive Phase-2 `n_s` points stress reference-mass assembly and
  geometry evaluation before they stress the outer CG iteration counts. So the
  Phase-2 plots should be read primarily as *solve-path* comparisons; the
  one-time setup column remains essential for understanding the full cost at
  high resolution.

## 5. Degree-by-Degree Structure

### `k = 0`

`k = 0` is the scalar surgery case.

- The extracted matrix is split into a small core block and one scalar bulk
  tensor block.
- The core is handled by a dense Schur solve.
- The bulk is handled by a scalar tensor inverse built from a fit of `J`.

So the active route is:

- outer scalar core Schur,
- scalar tensor bulk inverse.

### `k = 1`

`k = 1` uses a surgery-first extracted ordering.

- The extracted `theta` and `zeta` surgery rows form the outer Schur block.
- The bulk is split into `r`, `theta_bulk`, and `zeta_bulk` tensor blocks.
- The tensor route can optionally treat the bulk by an additional coupled inner
  Schur, but that coupling is not required for the outer surgery model.

So the active route is:

- outer surgery Schur,
- tensor bulk blocks for `r`, `theta_bulk`, and `zeta_bulk`,
- optional inner bulk Schur coupling.

The assembly-time toggle is:

- `cp_kwargs["k1_inner_schur"] = True` for the coupled bulk model,
- `cp_kwargs["k1_inner_schur"] = False` for pure diagonal tensor bulk blocks.

### `k = 2`

`k = 2` has the same overall philosophy with a smaller surgery block.

- The extracted `r` surgery rows form the outer Schur block.
- The bulk is split into `r_bulk`, `theta`, and `zeta` tensor blocks.
- The tensor route can optionally treat the bulk by an additional coupled inner
  Schur, but the outer surgery split remains the dominant structure.

So the active route is:

- outer surgery Schur,
- tensor bulk blocks for `r_bulk`, `theta`, and `zeta`,
- optional inner bulk Schur coupling.

The assembly-time toggle is:

- `cp_kwargs["k2_inner_schur"] = True` for the coupled bulk model,
- `cp_kwargs["k2_inner_schur"] = False` for pure diagonal tensor bulk blocks.

### `k = 3`

`k = 3` is the second scalar case.

- There is no surgery split.
- The extracted matrix is treated as one scalar tensor block.
- The inverse apply uses the tensor model built from a fit of `1 / J`.

So the active route is:

- direct scalar tensor inverse,
- no surgery Schur.

### 5.5 Practical winners on the rotating-ellipse family

For the historical record, the benchmark picture on the rotating-ellipse
family prior to the Phase 1 / Phase 2 sweep was:

- `k = 0` mass: scalar Schur plus tensor bulk decisively better than whole
  Jacobi and Jacobi-Chebyshev,
- `k = 3` mass: direct scalar tensor inversion decisively better than whole
  Jacobi and Jacobi-Chebyshev,
- `k = 1` and `k = 2` mass: the outer surgery Schur plus diagonal tensor
  bulk blocks already delivers most of the gain,
- the optional inner bulk Schur for `k = 1` and `k = 2` reduces iteration
  counts only slightly on the tested family, but increases runtime
  substantially,
- wrapping Chebyshev around an already strong tensor route often lowers
  iteration counts but usually does not improve wall-clock time.

This motivated the production defaults `k1_inner_schur = False`,
`k2_inner_schur = False`, and the rank-3 eager policy. The Phase 1 / Phase
2 sweep described in §4 is the formal version of these observations.

## 6. Forward-Model Diagnostics

The recent small-case forward-model checks help separate model quality from
solve-path effects.

- `k = 0` mass is a good rank-1 tensor model on the tested mapped case:
  about `1.6%` full extracted Frobenius error and about `4.7%` bulk-only.
- `k = 1` mass is a weak rank-1 tensor model on the same case: about `24%`
  Frobenius error both on the full extracted operator and on the bulk-only
  restriction. So this is a bulk-model issue, not a surgery artifact.
- `k = 2` mass is moderate at rank `1`: about `5.3%` Frobenius error, again
  with bulk-only error at essentially the same level.
- `k = 3` mass is also moderate at rank `1`, with about `5.5%` Frobenius
  error.

So the current rank-1 model-quality ordering is:

- good: `k = 0`,
- moderate: `k = 2`, `k = 3`,
- bad: `k = 1`.

Those rank-1 diagnostics turned out to be directionally correct but too
conservative about useful production ranks. The later higher-rank checks gave a
cleaner picture:

- `k = 0` mass is effectively a rank-2 geometry on the tested family. Forward
  error drops to near machine precision at rank `2`, and the solve count drops
  from about `11` iterations to about `3`.
- `k = 1` mass improves strongly from rank `1` to rank `2`, with a smaller
  further gain at rank `3`. On `ns = (8, 16, 8)`, the solve count dropped from
  about `28` to `14` to `13`.
- `k = 2` mass shows the same pattern, with the main gain at rank `2` and a
  smaller extra gain at rank `3`. On the same case, the solve count dropped
  from about `26` to `14` to about `12.5`.
- `k = 3` mass also benefits strongly from rank `2`, but shows no practical
  gain from rank `3`. On the same case, the solve count dropped from about
  `11` to `6`, then stayed there.

So the practical higher-rank conclusion is:

- rank `2` is a good default for all mass blocks,
- rank `3` is only a plausible extra option for `k = 1` or `k = 2`,
- and rank `2` already captures essentially all of the useful gain for `k = 0`
  and `k = 3`.

The larger mixed solve benchmark at `ns = (16, 32, 8)`, `p = 3` keeps that
overall recommendation but adds one useful refinement.

- `k = 0` mass still clearly prefers rank `2`: the solve count dropped from
  about `11.4` to `4`, with no further gain at rank `4`.
- `k = 1` mass still strongly prefers `inner_schur = off` in wall-clock time,
  but on this larger case rank `4` with `inner_schur = off` was the fastest of
  the tested tensor variants, at about `23.5` iterations / `153.2 ms` versus
  about `32.8` / `164.2 ms` at rank `1`.
- `k = 2` mass shows the same qualitative pattern: `inner_schur = off` remains
  the timing winner, and rank `4` with `inner_schur = off` was marginally the
  fastest tested tensor option at about `23` iterations / `147.2 ms`, versus
  about `29.6` / `147.6 ms` at rank `1`.
- `k = 3` mass is already saturated by rank `2`: the solve count dropped from
  about `11` to `6`, with rank `4` essentially tied in time.

So the updated practical reading is:

- keep rank `2` as the production default for all mass blocks,
- keep `inner_schur = off` as the production default for `k = 1` and `k = 2`,
- but treat rank `4` as a legitimate exposed tuning option for larger `k = 1`
  and `k = 2` cases where setup cost is acceptable and the extra iteration
  reduction matters.

One more free-boundary sweep at `ns = (16, 32, 16)`, `p = 3` is worth noting
because it sharpens that picture more convincingly.

- `k = 0` mass did **not** improve from rank `1` to rank `2` on this larger
  free case, but then improved strongly again at rank `3` and `4`: about
  `12.2` iterations / `8.52 ms` at rank `1`, about `13.0` / `9.63 ms` at rank
  `2`, and about `4.0` / `3.72 ms` at rank `3` and `4`.
- `k = 1` mass improved steadily across the ranks, with the main practical win
  now clearly at rank `3`: about `35.8` iterations / `348.5 ms` at rank `1`,
  about `30.0` / `297.6 ms` at rank `2`, and about `26.0` / `260.1 ms` at
  rank `3` and `4`.
- `k = 2` mass showed the same pattern: about `34.8` iterations / `332.2 ms`
  at rank `1`, about `29.0` / `282.4 ms` at rank `2`, and about `25.2` to
  `25.0` / `247.8` to `244.5 ms` at rank `3` and `4`.
- `k = 3` mass also kept improving beyond rank `2`, though more mildly: about
  `15` iterations / `4.20 ms` at rank `1`, about `7` / `2.35 ms` at rank `2`,
  and about `6` / `2.07 ms` at rank `3` and `4`.

So the current default should be read as a policy choice, not as the pointwise
best rank on every tested case:

- rank `3` is now the chosen eager default for all mass blocks,
- but free-boundary `k = 0` and `k = 3` now have concrete larger tests where
  rank `3` materially helps,
- and free-boundary `k = 1` / `k = 2` on the larger tested case now also look
  better at rank `3` than at rank `2`, with rank `4` giving little extra.

That said, the current evidence now favors the rank-3 policy directly: taken
at face value, this larger free-boundary sweep makes rank `3` the stronger
all-around candidate, while rank `4` adds little extra.

The current production default follows that recommendation in the eager
operator-assembly path: the mass blocks are assembled with per-degree tensor
ranks `k0 = k1 = k2 = k3 = 3`, while the scalar stiffness/Hodge fallback rank
remains at `1`.

## 7. Analytic Priors And Inversion Strategy

The recent toroidal-prior experiments are useful because they separate two
different questions:

1. how much analytic geometry should be built into the coefficient model,
2. how that modeled operator should then be inverted.

For the mass-side coefficient fields the natural leading toroidal factors are
the major-radius expressions

- `r R`,
- `R / r`,
- `r / R`,
- `1 / (r R)`,

with `R` shorthand for the toroidal major-radius factor. In the current prior
implementation those factors are used on the fit side, not as a separate exact
inverse formula. Concretely, if `W` is the coefficient field and `P` is the
known prior, the code fits the residual `C ≈ W / P` and then reconstructs the
modeled field as

$$
W_{\mathrm{model}} = P \cdot C_{\mathrm{fit}}.
$$

So the prior and the learned factors combine multiplicatively at the
coefficient-field level. Because both `P` and `C_fit` are represented as sums
of separable terms, the final assembled tensor block is still an additive sum
of separable Kronecker products after expansion.

That is different from an additive analytic expansion in `eps`. In an
`eps`-expansion route one writes the operator itself as

$$
A = A_0 + \varepsilon A_1 + \varepsilon^2 A_2 + \cdots,
$$

keeps a very simple analytic backbone `A_0`, and then uses that backbone as a
cheap inverse or smoother while the higher-order terms are treated as
corrections.

These two inverse strategies have different strengths.

Shared modal basis:

- Pros:
  - generic multirank inverse for arbitrary learned terms,
  - one uniform implementation for all tensor blocks,
  - currently the strongest fully automatic inverse when the coefficient model
    is not already extremely simple.
- Cons:
  - once the modeled field contains more than one separable term, the direct
    rank-1 inverse is lost and the block falls back to a dense shared-basis
    modal solve,
  - this does not exploit the original 1-D banded operator structure,
  - performance can become sensitive to the quality of the learned coefficient
    model.

Analytic expansion plus Richardson:

- Pros:
  - keeps the analytic rank-1 backbone explicit,
  - can exploit axis-by-axis 1-D solves against the underlying banded factors
    instead of building a dense modal basis,
  - Richardson correction uses the true residual of the full operator rather
    than relying on a truncated inverse series.
- Cons:
  - requires a problem-specific analytic expansion,
  - is less generic than the shared modal basis,
  - and if the analytic backbone is too weak, Richardson may need several
    correction steps before it is competitive.

This is why the current implementation uses the prior only to simplify the fit,
not yet to change the inverse. That is the safer step. The open design question
is whether the scalar and low-rank mass routes should eventually switch from
"prior plus shared modal basis" to an explicit analytic backbone plus a few
Richardson correction steps.

The current preferred next experiment is now more specific than a generic
additive `eps` expansion. Rather than giving every analytic branch the same
status, the better hybrid is:

- keep one leading analytic backbone term,
- require its learned coefficient to be rank `1`,
- and spend the remaining rank budget on a correction channel.

So instead of the current multiplicative form

$$
W_{\mathrm{model}} = P \cdot C_{\mathrm{fit}},
$$

or a fully symmetric additive model with many equally weighted learned
coefficients, the preferred ansatz is

$$
W_{\mathrm{model}} \approx B_0 \odot C_0 + B_1 \odot \widetilde C_1,
$$

where `B_0` is the leading analytic basis term, `C_0` is constrained to rank
`1`, `B_1` is the correction channel, and `\widetilde C_1` carries the rest of
the rank budget, for example rank `m - 1` if the total budget is `m`.

The reason this is attractive is solver-side rather than fit-side. Because
both `B_0` and `C_0` are rank `1`, the backbone term `B_0 \odot C_0` is still a
true rank-`1` tensor block and keeps the simple inverse structure. The
correction term can then be treated as the flexible residual, which is exactly
the setting where Richardson correction is most plausible.

For the toroidal families it is useful to write that analytic backbone
explicitly. Let

$$
R(r,\theta) = 1 + \varepsilon r \cos\theta.
$$

Then the four mass-side coefficient families are

- `k = 0`: `r R`,
- `k = 1` / `k = 2` angular or toroidal blocks: `R / r` or `r / R`,
- `k = 3`: `1 / (r R)`.

The `R`-type families are exact finite expansions:

$$
rR = r + \varepsilon r^2 \cos\theta,
$$

$$
\frac{R}{r} = \frac{1}{r} + \varepsilon \cos\theta.
$$

So those are exact two-term additive backbones. By contrast the inverse-type
families require a truncated series:

$$
\frac{1}{R} = 1 - \varepsilon r \cos\theta + \varepsilon^2 r^2 \cos^2\theta - \cdots,
$$

which gives

$$
\frac{r}{R} = r - \varepsilon r^2 \cos\theta + \varepsilon^2 r^3 \cos^2\theta - \cdots,
$$

$$
\frac{1}{rR} = \frac{1}{r} - \varepsilon \cos\theta + \varepsilon^2 r \cos^2\theta - \cdots.
$$

So the power of `r` grows with the order kept in the `eps` expansion, not with
the CP rank. This is one of the reasons the additive analytic route is often
easier to reason about than the multiplicative prior route.

For that hybrid rank-allocation plan, the natural leading backbones are:

- `k = 0`: use `B_0 = r` and put the toroidal correction into the `r^2 cos(theta)`
  channel,
- `k = 1` and `k = 2` for the `R / r` families: use `B_0 = 1 / r` and put the
  toroidal correction into the `cos(theta)` channel,
- `k = 1` and `k = 2` for the `r / R` families: use `B_0 = r` and treat the
  first inverse correction `- r^2 cos(theta)` as the residual channel,
- `k = 3`: use `B_0 = 1 / r` and treat `- cos(theta)` as the first correction
  channel.

That keeps the easiest analytically invertible part in the backbone and uses
the remaining rank budget only where the toroidal correction is actually
needed.

The practical consequence is that the most attractive Richardson backbones are:

- `k = 0`: the exact two-term `r + eps r^2 cos(theta)` backbone,
- `k = 3`: the zeroth- or first-order truncated `1/rR` backbone,
- `k = 1` and `k = 2`: the exact `R/r` backbone first, and then the first-order
  `r/R` truncation only if needed.

Those choices keep the backbone as simple as possible while still encoding the
dominant toroidal structure. They are therefore the best candidates for an
analytic expansion plus Richardson experiment.

For the first implementation, the fit itself should also stay simple. The
recommended procedure is a greedy two-stage residual fit rather than a fully
coupled additive optimization:

$$
R^{(0)} = W,
$$

fit rank-`1` `C_0` from

$$
B_0 \odot C_0 \approx R^{(0)},
$$

then update

$$
R^{(1)} = R^{(0)} - B_0 \odot C_0,
$$

and fit the remaining rank budget in the correction channel,

$$
B_1 \odot \widetilde C_1 \approx R^{(1)}.
$$

No extra normalization or polishing sweep is needed for the first pass. The
important structural point is simply that the leading term remains rank `1`
and easy to invert, while the correction term absorbs the remaining error.

The first forward-model checks refined that plan in an important way.

For `mass-k0` on the small rotating-ellipse case `ns = (4, 8, 4)`, `p = 3`, a
split rank-`2` model already behaved as intended structurally: the backbone
term carried about `99.3%` of the tensor norm, the correction-to-backbone norm
ratio was about `0.12`, and the residual left after the backbone alone was also
about `0.12`. So for `k = 0` the split really is in a "dominant backbone plus
small correction" regime.

But that same check also showed why the split and multiplicative paths cannot
be compared only by the printed rank parameter. The current multiplicative path
expands each learned residual mode through the multi-term toroidal prior, so a
printed rank-`m` fit can assemble more than `m` final separable terms. The
split path does not: split rank `m` literally means one rank-`1` backbone term
plus a free rank-`m - 1` residual.

The more important result came from `mass-k1`.

- At split rank `2`, the model was too rigid. The `arr` and `theta` blocks
  still had about `36%` to `38%` residual after the backbone alone, and the
  full forward error was much worse than the multiplicative baseline.
- At split rank `3`, the same `k = 1` path became competitive and in fact beat
  the multiplicative rank-`2` forward model on that test. The split-rank-`3`
  run gave about `6.23e-3` Frobenius error versus about `8.09e-3` for the
  multiplicative rank-`2` model.

The diagnostic picture explains why. On that same `k = 1`, split-rank-`3`
case:

- the `arr` and `theta` blocks were not in a "tiny correction" regime,
  with correction-to-backbone norm ratios around `0.39` to `0.40`,
- the `zeta` block was much closer to that regime, around `0.11`,
- and the backbone-only residuals were about `0.36` to `0.38` for `arr` and
  `theta`, versus about `0.11` for `zeta`.

So the current reading is:

- `k = 0` really does look like "easy backbone plus small correction",
- `k = 1` does not at split rank `2`,
- but `k = 1` becomes promising once the correction channel is allowed more
  than one rank-`1` term,
- and the backbone is still dominant enough that a backbone solve plus a few
  residual-correction steps is a plausible inverse strategy.

This is therefore the practical next-step policy for the split model:

- keep the rank-`1` backbone requirement,
- let the residual channel use the remaining rank budget freely,
- and evaluate inversion strategies on top of that split rather than forcing
  the correction itself to remain rank `1`.

## 8. Geometry Sensitivity: `eps = 1/3` vs `eps = 1/7`

The latest rotating-ellipse sweep compared the same benchmark at
`ns = (16, 16, 16)`, `p = 3`, ranks `1, 2, 4`, and inner Schur on/off for
`k = 1, 2`, with `eps = 1/3` and `eps = 1/7`.

Lower `eps` makes every mass block easier. The strongest and most consistent
effect is on model quality: the tensor coefficient fits become substantially
more low rank. The solve-time improvement is more modest once a block is
already near its practical floor.

Representative practical rows are:

| k | rank | inner | `cp_err` change | iteration change | time change |
| --- | ---: | :---: | ---: | ---: | ---: |
| 0 | 2 | n/a | `-57.7%` | `0.0%` | `-2.2%` |
| 1 | 2 | off | `-55.4%` | `-10.7%` | `-10.0%` |
| 2 | 2 | off | `-57.3%` | `-9.4%` | `-8.9%` |
| 3 | 2 | n/a | `-84.0%` | `-16.7%` | `-15.4%` |

So the practical reading is:

- `k = 0` and `k = 3` were already close to saturation at rank `2`, so lower
  `eps` mainly improves fit quality rather than runtime.
- `k = 1` and `k = 2` remain the most geometry-sensitive blocks in solve time,
  but the gain is still moderate: about `9%` to `10%` faster in the practical
  rank-`2`, inner-Schur-off regime.
- The inner-Schur conclusion does not change with `eps`: for `k = 1` and
  `k = 2`, turning the coupled inner Schur off is still much faster in
  wall-clock time on the tested family.

## 9. Final Summary

The final mass-preconditioner picture is simple.

- keep the extracted-space special rows exact through a small Schur solve,
- compress the diagonal mapped coefficient fields rather than the inverse,
- use tensor block inverses only on the regular bulk blocks,
- and keep the optional inner coupled bulk Schur for `k = 1` and `k = 2` as a
  benchmarked option rather than as the default practical choice,
- while treating rank `2` as the practical default tensor rank across
  `k = 0, 1, 2, 3` on the tested geometry family,
- with `k = 2` rank `3` left as an exposed tuning option rather than the
  default because the measured extra solve gain has not yet been weighed
  against additional setup cost.

See §10 for the W7-X (strongly non-separable) findings, which add an
important caveat: on a non-separable metric the CP factorization must be
**non-negative** (NTF) to stay SPD, and the exact tensor inverse is capped at
rank `2`.

## 10. Strongly non-separable geometry (W7-X) — NTF and the rank-2 ceiling

*Investigation 2026-06-26. Sweep: `outputs/overnight_sweep/2026-06-26/00-54-05`;
NTF re-runs under `outputs/ntf_retest/`, `outputs/ntf_validate/`,
`outputs/ntf_rank34/`.*

### 10.1 The failure and its real cause

On the W7-X stellarator the vector mass tensor preconditioner (`k = 1, 2`)
**diverges at high polynomial degree** — `k = 1` for `p >= 4`, `k = 2` for
`p >= 5` — while `k = 0, 3` (scalar) stay cheap at all `p`. The cause is *not*
the geometry and *not* the dropped off-diagonal metric:

- The geometry is healthy. The Jacobi baseline, which uses the **true** mass
  matrix built from the same metric/Jacobian fields, converges cleanly to
  `1e-11` at `p = 5` (`k=1`: 6768 it, `k=2`: 4516 it), only ~30% worse
  conditioned than the smooth geometries. So the true operator is SPD and
  well-behaved; no mesh fold, no near-singular Jacobian.
- The metric is mostly diagonal, so off-diagonal coupling is not the issue.

The real issue is **diagonal ≠ separable**. Each diagonal weight
`W^{(k)}(r,\theta,\zeta)` is a fully *non-separable* positive function. The
tensor route forces separability via the CP fit, and the **greedy** CP fit
(sequential rank-1 against a residual) produces sign-changing factors past
rank 1 — those make the assembled rank-2 Kronecker surrogate **indefinite**
(failed Cholesky anchor / non-positive fast-diagonalization denominator),
which is what blows PCG up to `nan`. Plain rank-1 stays SPD but is too poor an
approximation and stalls.

### 10.2 Fix: non-negative CP (NTF), now the default

We replace greedy CP with a **joint non-negative tensor factorization** (NTF,
Lee–Seung multiplicative updates) for the mass coefficient fields
(`_cp_ntf_3tensor` / `_ntf_terms` in `mrx/preconditioners.py`). Because
`W^{(k)} >= 0` and every factor is `>= 0`, each per-axis weighted mass
`B diag(quad·w) B^T` is SPSD and the assembled Kronecker surrogate is **SPD by
construction at any rank** — no clamp, no dense pseudo-inverse fallback. NTF is
now the default; `MRX_CP_GREEDY=1` restores the legacy greedy fit for A/B.

Result on W7-X (`12x24x12`, dbc, avg CG iters):

| case | greedy r2 | NTF r2 |
|---|---|---|
| `k=1` p4 | `nan` | **68** |
| `k=1` p5 | `nan` | **91** |
| `k=2` p4 | 94 | **67** |
| `k=2` p5 | `nan` | **fails** (rank-2 insufficient) |

NTF rank-1 is numerically identical to greedy rank-1 where the latter worked,
including **cylinder's exact 1-iteration convergence** (separable metric — the
NTF SVD init is exact for a rank-1 tensor), so there is no regression on the
easy cases.

### 10.3 Two hard limits this exposed

1. **Exact inversion is capped at rank 2.** Fast diagonalization inverts a sum
   of at most two Kronecker terms (one can simultaneously diagonalize a *pair*
   of symmetric matrices per axis, not three). For `rank >= 3` the code builds
   the modal basis from the leading two terms and keeps only the **diagonals**
   of the projected extra terms — the inverse is no longer exact. So rank-3+ is
   a better *fit* but an approximate *inverse*; gains are not guaranteed.

2. **NTF trades approximation power for the SPD guarantee, and rank does not
   buy it back.** On the toroid, greedy rank-2 reaches ~5 iters (its
   sign-changing second term fits the `1/R` coupling efficiently and happens to
   stay SPD), but NTF is stuck at ~12 iters at **every** rank tested
   (r2 = 12, r3 = 12, r4 = 11). Non-negativity cannot represent the toroidal
   metric as compactly, and higher NTF rank does not recover it. So
   "always-NTF" is robust but pays a real, persistent cost where greedy's
   sign-changing fit was both better and (by luck) PSD.

### 10.4 Safe envelope and open items

- **Safe today (NTF default):** `k=0,3` all `p`; **`k=1` through `p5`** (NTF
  rank-2: p4 = 68 it, p5 = 91 it; rank-1 stalls so rank-2 is required); `k=2`
  through `p4` (67 it). **`k=2` p5 is the only genuine wall** — NTF rank-2/3/4
  all fail to converge (residual crawls 0.83 -> 0.26 -> 0.10 but never reaches
  tol), so no rank-≤2 separable surrogate solves it and inexact rank-3/4 only
  creeps.
- **Proposed next step — single-out-one-axis / exact cross-section.** Factor
  out the most-separable axis as rank-1 and keep the hard cross-section as an
  *exact* 2-D weighted mass (SPD by construction, no separability assumption,
  no sign issues). This spends the exact 2-term budget where the geometry is
  hard and side-steps both greedy-indefiniteness and the NTF quality loss. It
  necessarily makes the singled-out axis rank-1 (a `2-2-1`-type structure);
  genuine rank-2-in-all-three needs 4 Kronecker terms and is not exactly
  invertible. Viability hinges on the singular-value decay of the weight
  tensor's mode unfoldings (which axis is near rank-1) — **diagnostic pending**.
- **Re-run scope** (all `MRX_CP_GREEDY` unset, `--groups mass`): NTF only moves
  the mass blocks (`k=0..3`); the stiffness preconditioners in `mrx/operators.py`
  call greedy CP directly and are unaffected. Necessary: W7-X all `p` + both
  resolutions; regression guard: cylinder/toroid `p3`/`p5` (done — clean except
  the documented toroid quality cost). Skip all `k`-stiffness baselines.

### 10.5 Which weights are hard, analytically

From the toroidal metric (orthogonal columns → diagonal `g`):
`g_rr = a^2`, `g_θθ = a^2 r^2`, `g_ζζ = R^2`, `J = a^2 r R`, with
`R = R0 + a r cosθ`. Cancelling constants, the diagonal weights reduce to:

| block | weight | ~form | type |
|---|---|---|---|
| M0 | `J` | `rR` | grows-with-r — **hard** |
| M3 | `1/J` | `1/(rR)` | `1/r` — **easy** |
| M1 `αrr` | `J g^{rr}` | `rR` | **hard** |
| M1 `αθθ` | `J g^{θθ}` | `R/r` | `1/r` — **easy** |
| M1 `αζζ` | `J g^{ζζ}` | `r/R` | **hard** |
| M2 `βrr` | `g_{rr}/J` | `1/(rR)` | `1/r` — **easy** |
| M2 `βθθ` | `g_{θθ}/J` | `r/R` | **hard** |
| M2 `βζζ` | `g_{ζζ}/J` | `R/r` | `1/r` — **easy** |

M1 is the diagonal of the IGA-paper coefficient
`Q = det(J_F) J_F^{-T} K J_F^{-1}` (`K=I`); see
`papers/preconditioner/IGA_sylvester_revised_submitted.tex` eq. (3.8), whose
Prop. 4.1 ties CG convergence to the spread of `Q`'s eigenvalues.

**Rule (magnitude-weighted SVD).** `1/r`-type weights peak near the axis where
they collapse to `1/r ⊗ const` → **rank-1**; the angular coupling lives only
where the weight is small. `r`-growing weights (`rR`, `r/R`) peak at large `r`,
exactly where `R`'s θ/ζ variation dominates → **high θ-rank (8–10)**. The
diagnostic `scripts/debug/w7x_weight_tensor_rank.py` confirms every case on
W7-X (M3 `1/J` = rank 1/1/1; M2 `βζζ` = 1/1/1; M0 `J` = 3/7/6; the four hard
weights = r-rank 3, θ-rank 8–10).

Consequences: **r is always the low-rank axis (2–3); θ-ζ is the coupled plane.**
M1 fails before M2 because M1 has 2/3 hard components vs M2's 1/3. Only the
**four `r·R`/`r/R` weights need a richer cross-section**; the four `1/r` weights
are already rank-1-trivial.

### 10.6 Strategy: symmetric CP vs. factor-out-r (open)

The exact tensor inverse is a 2-Kronecker-term budget (§10.3). Two ways to spend
it on the mass coefficient:

- **(A) symmetric CP-2** — `a1 b1 c1 + a2 b2 c2` (current default under NTF).
  Maximum freedom for two terms; already gives `k=1` to `p5`, `k=2` to `p4`.
- **(B) factor-out-r** — `a(r) (b1(θ) c1(ζ) + b2(θ) c2(ζ))`, i.e. rank-1 in `r`
  times a rank-2 CP cross-section. This is exactly invertible with **only 1-D
  eigendecompositions** — a 1-D banded solve in `r` tensored with a 2-D fast
  diagonalization of the two-term θ-ζ sum (no dense 2-D inverse) — and is low
  rank in every coordinate (1/2/2). It matches the metric structure (§10.5: `r`
  is rank-1–dominant; the coupling is θ-ζ), and by not spending a term on `r` it
  gives the cross-section a full rank-2 budget.

**Decision (measured, `scripts/debug/w7x_AB_fit_compare.py`): at rank-2, A and B
are equivalent — keep A.** B's fit residual equals A's on every weight at every
`p` (factoring `r` to a single rank-1 profile costs `<= 0.045`, `<= 0.006` on the
easy weights — r is the universally-separable axis), AND the cost is essentially
identical: both are 1-D-only fast diagonalization with three paired
decompositions and ~6 mode-multiplications per apply (B saves at most one
mode-mult by inverting `A_r` directly). The earlier claim that B is "cheaper /
better-structured" was wrong: A is also 1-D-only, dense-2-D-free, and low-rank
in every coordinate. So there is no reason to switch for the rank-2 case.

B becomes a genuinely different object only at `R >= 3` (the inexact regime),
where it keeps `r` exact and confines the diagonal-truncation approximation to
the 2-D cross-section — exactly where the fit error lives — whereas symmetric
A spreads the truncation across all three axes including `r`. Whether that
better-targeted truncation actually beats symmetric A-rank-R is untested.

**But the binding limit is the cross-section rank, not the structure.** At
rank-2 both A and B leave large residuals on the hard weights — `αrr`, `βθθ`
≈ 0.28, `αθθ`, `βrr` ≈ 0.26 — because the θ-ζ cross-section genuinely needs
**rank 3–7** (measured: M3/βζζ rank 1; αθθ/βrr rank 3; J/αζζ rank 5; αrr/βθθ
rank 7). This residual is `p`-independent, so it is also why W7-X merely "works"
rather than excels (M1 p3 = 93 it, not 1: the 28% fit *is* the iteration count),
and the p5 failures are conditioning tipping a chronically-weak preconditioner
over. A and B are equally capped here.

**The lever is therefore cross-section rank `R`** (not the A-vs-B structure,
which is a wash at rank-2). Both A and B can carry higher rank via inexact FD;
the open question is whether higher `R` helps at all, and if so whether B's
cross-section-confined truncation beats symmetric A's 3-D truncation. Next
experiment: test `R = 5` (symmetric A-rank-5, already implementable, vs B-rank-5)
on `k=2` p5 — does it rescue convergence, and how much does it accelerate the
working cases.
