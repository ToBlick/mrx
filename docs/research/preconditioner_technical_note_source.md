> **Status:** current; the canonical account of the production preconditioner
> **Read this for:** the construction, the derivation of the natural-BC coefficient, what was refuted and every measurement
> **Do not read for:** the scale value: it says 0.10, production is PRODUCTION_BC_SCALE = 3.0 (s_scale_2026-08-25.md)

# The MRX tensor block-Jacobi preconditioner — complete source for a technical note

Written 2026-08-22 to be read WITHOUT access to the cluster, the data, or the
session that produced it. Everything needed to write the note is here: the
construction, the derivation, the mechanism, every measurement, and the list of
what was tried and failed. Numbers are transcribed from the JSON of the runs
named in each section; the raw data is in `outputs/<name>/` on the cluster.

Companion: `preconditioner_lessons.md` (what is unfinished).

---

## 1. What the object is

MRX discretises a de Rham complex `V_0 -> V_1 -> V_2 -> V_3` with tensor-product
B-splines on a logically-cubic domain `(r, theta, zeta)`, mapped to the physical
torus/stellarator by `F`. The Hodge Laplacian at degree `k` is

    L_k  =  S_k + W_k ,     S_k = G_k^T M_{k+1} G_k ,   W_k = D_{k-1} M_{k-1}^{-1} D_{k-1}^T

with `G` the (metric-free) incidence operators, `M` the metric-carrying mass
matrices, and

    D_l  =  M_{l+1} G_l          (mass TIMES incidence, not the incidence alone)

so that written out in incidence operators only,

    L_k  =  G_k^T M_{k+1} G_k  +  M_k G_{k-1} M_{k-1}^{-1} G_{k-1}^T M_k .

`D_l` is the weak-derivative (codifferential-adjoint) matrix; in the code it also
carries the extraction operators, `D_l = E_{l+1} M_{l+1} G_l E_l^T`. The `M_k`
factors flanking `W_k` are not decoration -- they are why the weak term is the
part that carries the metric twice. Two facts drive everything:

* **`M_{k-1}^{-1}` is never formed.** `W_k` is applied with a mass
  PRECONDITIONER standing in for the inverse. So the mass preconditioner is
  part of the OPERATOR at `k >= 1`, not just part of the solve. Changing it
  changes `L_k`. (This is the single most load-bearing fact in the whole
  design, and it is easy to miss.)
* **The polar axis is special.** `r = 0` is a coordinate singularity; MRX uses a
  polar basis whose extraction operator `E` maps a ring of raw DOFs onto a
  smaller set of extracted ones. Those rows are not tensor-product functions.

The preconditioner is a **block-Jacobi atom**: one separable approximation of
`L_k` per vector component, inverted exactly by fast diagonalisation, plus a
small dense block for the rows the separable form cannot represent.

## 2. The atom

For component `c` of `V_k`, approximate the diagonal block of `L_k` by a
three-term Kronecker sum

    A_c  =  K_r (x) M_t (x) M_z  +  M_r (x) K_t (x) M_z  +  M_r (x) M_t (x) K_z

where each `K_a` is a 1-D stiffness carrying an axis-averaged metric weight and
each `M_a` an unweighted 1-D mass. This is inverted EXACTLY by fast
diagonalisation: solve the three 1-D generalised eigenproblems `K_a v = lam M_a v`
once at build time, then every apply is three small dense multiplications and a
pointwise divide by `lam_r + lam_t + lam_z`. Cost is `O(N (n_r + n_t + n_z))`
per apply with `O(n^2)` storage per axis -- no fill-in, no factorisation of
anything 3-D.

Three details that are not cosmetic:

**(a) The weights must be axis-averaged, and the masses must NOT be weighted.**
`g^{aa} J` is folded into `K_a` by averaging over the other two directions;
weighting the masses as well double-counts it.

**(b) `lumped="diag"`: the D sandwich.** The component factor `w_comp = m_k / J`
is pulled OUT of the atom and applied as a diagonal similarity transform
`D^{1/2} A D^{1/2}`. Anything else that carries `w_comp` -- notably the boundary
term of §4 -- must then drop it, or it is counted twice. (Getting this wrong is
how two hidden metric factors survived for weeks; see §7.)

**(c) `ktilde_mode="honest"`.** On a DERIVATIVE axis the exact factor is the
weak block's radial round trip `F = M^d G A^-1 G^T M^d`. The "honest" choice is
instead the 1-D stiffness OF THE DERIVATIVE SPLINES -- a weighted mass of their
tabulated derivatives, with no incidence and no `A^-1`, so nothing to mis-scale.
The alternative ("roundtrip") was the code's default and was never measured;
when it finally was, it lost **all 28 A/B rows**, by a median of 4.4x and up to
9.8x. At `k=0` the two are identical, as they must be -- `ktilde_mode` only
differs on derivative axes and `k=0` has none.

## 3. The polar core

The rows where `E` mixes a ring of raw DOFs are not tensor-product functions, so
no separable atom represents them. They are handled by **replacement**: probe
`L_k` on exactly those rows (one operator apply per row), invert the small dense
block, and use it there instead of the atom. `extra_rings`/`outer_rings` widen
that exact region inward/outward; both are diagnostics, not production (see §8).

## 4. The natural boundary condition — the derivation

Under a FREE (natural) condition at `r = 1` the weak block's integration by
parts leaves a surface term that the separable atom otherwise omits entirely.
In a tensor basis it is

    alpha . (e e^T) (x) M_t (x) M_z ,        e = dLam_r(1)

which has exactly the shape of the FIRST Kronecker term, so it merges into `K_r`
as a **rank-one update**. Fast diagonalisation, cost and storage are all
untouched: the term is free.

**`e` is one-hot.** Measured for `p = 1..4`: `DerivativeSpline` at a clamped end
has a single nonzero (the apparent second is `O(eps)` from evaluating at
`1 - 1e-8`). NOTE the naming trap: `basis_0.dLam` is the basis OF THE DERIVATIVE
SPACE (one-hot at the end); `Lam'(1)`, the derivative of the VALUE basis, has
two nonzeros and is the FLUX functional, a different object.

**The coefficient.** Take each matrix's weight at `r=1`, average over
`theta, zeta`, multiply: `E -> m_k`, `M_{k-1}^{-1} -> 1/m_{k-1}`, `E^T -> m_k`.
Using `J^2 = prod g_aa` this collapses at EVERY degree to

    alpha  =  mu_0 . <S> . <P> ,     S = J sqrt(g^rr),   P = sqrt(g^rr)
    mu_0   =  (M_r^logical)^-1 [last, last]          (METRIC-FREE)

with `w_comp` dropped because the `D` sandwich already carries it (§2b).

**Verification of the coefficient** (`bc_alpha_compare.py`, `diag_alphaverify`):
`mu_0` came out BIT-IDENTICAL across all three geometries at each degree --
66.73954 (p=2), 93.88387 (p=3), 134.4128 (p=5) -- which is a strong check
because any metric leakage would break it. `alpha` is identical down all four
`(k, c)` rows at every `p`, i.e. degree-independent as derived. Both hold at
p = 2, 3 and 5.

**Which components carry it.** Exactly those whose RADIAL axis is a derivative
axis:

| k | weak term | trace components |
| --- | --- | --- |
| 0 | `W_0 = 0` | **none** -- the atom's `w d_r u = 0` IS the operator's natural condition, exactly |
| 1 | `<u, grad tau> -> int (u.n) tau` | `c = r` (normal) |
| 2 | `<w, curl tau> -> int (w x n).tau` | `c = theta, zeta` (tangential) |
| 3 | `<om, div tau> -> int om (tau.n)` | the single component |

At k=2 the cross product pairs `c = theta` with `V_1`'s `c = zeta` (`3 - c`),
not with itself; the weights differ by `(R/a)^2` and fixing it took toroid
`12^3` k=2 free from 158 to 62 iterations.

Under a DIRICHLET condition the term vanishes identically (the partner DOF is
removed), and this is a hard invariant worth testing: every dbc row must be
bit-identical across any change to the boundary term. A missing guard once added
entries on the PERIODIC theta/zeta axes and this is the check that caught it.

## 5. The scale — the part that is empirical

`alpha` as derived is the operator's exact surface term. **It is the wrong SIZE
for a preconditioner.** Production uses

    PRODUCTION_BC_SCALE = 0.10          # multiplies alpha

### 5.1 Why a scale is needed at all (measured, Table F)

Without the term the atom is too SOFT at the boundary; with it at full strength
too STIFF. Neither is a statement about the surface integral -- both are about
`P` versus `L`:

* at `s = 0` the HIGH outliers of `spec(P L)` are boundary-localised and
  numerous;
* at `s = 1` the LOW outliers appear and `min eig(P)` collapses. Measured:
  **`min eig(P) = 1/(1 + r s)` to 3% over a 17x range in `s`**, where `r` is the
  face-row stiffening ratio (`alpha e_last^2 / K_r[-1,-1]` = 8.196 on the
  toroid, 7.27 fitted on rot-ellipse). The atom's boundary DOF is effectively
  DECOUPLED -- `e` is one-hot and the atom is a Kronecker sum, so that radial
  index is its own -- and `P` inverts it bare. A large penalty therefore does
  not impose the condition; it DELETES that row's preconditioning.
* the optimum is where the two families balance, and **at the optimum BOTH
  still exist**.

So the scale does not fix the boundary; it trades one error against the other.

### 5.2 What the factor IS (measured, Table G)

The clean experiment: on the outer `d` radial rings `R`, compare `L`'s block
`B_raw = L[R,R]` against the atom's implied block `A(s) = inv(P(s))[R,R]`, by
two criteria.

* `||A(s) - B_raw||` is minimised at **s ~ 1.0** -- `alpha` IS the best NORM
  approximation to `L`'s boundary block. The derivation is right.
* `cond(B_raw, A(s))` is minimised at **s ~ 0.06-0.55, ordered by geometry** --
  the best PRECONDITIONER of that block is a much smaller `alpha`.

**The best approximation is not the best preconditioner, and the gap between
them is the factor.** Cause: the atom's ring block is missing the WITHIN-RING
coupling (angular and cross-component) that `L` carries, and under-stiffening
the diagonal is the best a diagonal-only knob can do to compensate spectrally.

The ring-block criterion PREDICTS the measured optimum, ordering exact and
magnitudes within one sweep point on all four geometries (Table G vs Table A).
That is the basis for the open item "auto-compute `bc_scale`".

**A hypothesis that was tested and REFUTED**: that the factor is the interior
DtN / Schur reduction. The Schur complement removes only 17-34% of
`tr(L[R,R])` at depth 1 and 0.4-1.1% at depth 4 (it SHRINKS with depth), and
`B_schur` picks the same scale as `B_raw` or a LARGER one. Not the DtN.

### 5.3 The controlled experiment: the cylinder (Table F)

The cylinder metric has NO angular variation (angular spread: cylinder 0%,
toroid 24%, W7-X 60%). It is the zero-coupling limit, and there:

* **ZERO low outliers at every scale, including `s = 1`** (rot-ellipse: 8 -> 27);
* `lambda_min(PL)` moves 2% over the whole range (rot-ellipse: falls 60%);
* `cond` is MONOTONE DECREASING to `s = 1` -- **no interior minimum exists**.

Remove the coupling and the second error family vanishes, the optimum runs to 1,
and the derived coefficient is exactly right. The cost of `s = 1` tracks the
coupling across geometries: cylinder 1.00, toroid 1.00, rot-ellipse 1.73,
W7-X 2.03 (median over n, k=1). **This is the cleanest evidence in the whole
investigation that the factor is coupling-compensation and not a derivation
error** -- an arithmetic slip would be wrong on the cylinder too.

### 5.4 Why 0.10, and how safe it is

Minimax over 82 cells (4 geometries x k=0..3 x n=8..32 x p=2..5) under the OLD
mass, and re-confirmed over 168 cells under the new one (Table B/§9):

| band | 0.03 | 0.06 | **0.10** | 0.15 | 0.22 | 0.30 |
| --- | --- | --- | --- | --- | --- | --- |
| all cells | 1.76 | 1.45 | **1.19** | 1.23 | 1.36 | 1.37 |
| shaped only (rot-ell, w7x) | 1.36 | 1.19 | **1.11** | 1.23 | 1.36 | 1.37 |
| production n >= 16 | 1.63 | 1.27 | **1.11** | 1.23 | 1.36 | 1.37 |

The dependence is real but weak on every axis -- geometry (a 9x spread at n=12
narrowing to ~3x by n>=24), n (monotone DOWN), p (monotone down, factor 2-5
from p=2 to p=5), k (barely: k=1 ~ k=2; k=3 wants slightly more at low n). There
is no law to fit, and §5.2 says why: it is a kappa-balance point, not an
algebraic factor.

**Err LOW.** Too small is BOUNDED -- the limit is simply "no term", at most
3.7x. Too large is UNBOUNDED: 1.00 costs up to 2.24x, `bcp300` gave 39x, and
`x1e4` was catastrophic, all by the `1/(1 + r s)` mechanism of §5.1.

## 6. The mass preconditioner

Because `M^-1` sits inside `L_k` (§1), the mass preconditioner matters twice.
Production is `kind='block_jacobi'` since 2026-08-22 (was `raw_kron`).

`BlockJacobiMass` has the same separable-bulk shape as raw_kron, but the polar
core rows are PROBED AND INVERTED DENSELY rather than reached through the `E+`
pseudoinverse (whose "both sides must carry the full `(CC^T)^-1`" requirement
raw_kron's own docstring calls the easiest thing to get wrong). A mass is
structurally easier than a Laplacian -- a single Kronecker PRODUCT, not a sum --
so the bulk inverse is three 1-D solves with no fast diagonalisation.

Measured (Table D): **0.83x raw_kron's iterations at the median, 0.70-0.77x at
k=1,2 where the cost is**, holding or improving with h, flat in p, at equal
build cost. Only regression is ~5% at k=0, on solves that take 7-17 iterations.

Effect on `L_k` (which it changes, per §1): **0.91x median, better in 12 of 16
cells**, up to 0.79x on Dirichlet rows. `bc_scale = 0.10` survives -- see §9.

## 7. Baselines, and why the obvious one flatters

* **`kind='jacobi'`** -- per-DOF diagonal. For `k >= 1` its weak half is a
  CLOSED FORM UNDER THE KRONECKER MASS MODEL, i.e. a model of `D M^-1 D^T`, not
  the operator's own. Cheap, and the production fallback.
* **`kind='probed_jacobi'`** -- the same diagonal taken EXACTLY, one operator
  apply per DOF. The honest reference. `O(N)` applies to build (seconds at
  n=12, ~30 min extrapolated at n=32), so it is a reference and never a
  candidate.

Measured gap (Table E): at **k=0 they agree to 0.8%** -- they must, since
`L_0 = S_0` has no weak term for the model to get wrong, and that is the control
that validates the probe. At `k >= 1` the modelled diagonal costs up to 21%
extra iterations, geometry-correlated exactly as everything else is (cylinder
and toroid ~1.00, rot-ellipse 0.86-0.94, W7-X 0.79-0.96).

**So ratios quoted against `jacobi` are ~20% flattering at k=1/2 on shaped
geometries.** For the note, quote both or quote `probed_jacobi`.

## 8. What was tried and REFUTED (do not re-propose)

Each was measured; the note can state these as closed.

| candidate | verdict |
| --- | --- |
| `exact` (the shipped default before this work) | provably 8-14x too small AND worse than NO boundary term at k=1/2 free on rot-ellipse (1056 vs 636) |
| the cross term at the computed `rho` (`ibpr`) | makes `P` INDEFINITE on every geometry at every k |
| the full cross-term correction (`ibpf`) | SPD but 1.4-2.6x worse, worsening with resolution |
| exact 2-D face shape by quadrature (`wibp`, Woodbury) | 1.35-1.9x worse at the corrected scale |
| hard pin of the trace DOF | measured NO-OP on the high outliers; the failing modes carry zero `u_r` |
| hard `u.n = 0` by huge penalty | 250 it vs 76 -- the row is ABANDONED, not constrained (§5.1) |
| Nitsche cross-component consistency | diverges |
| mode-dependent beta (`tm`) | BROKEN, measured |
| a degree law for `bc_scale` (`1/(2p+1)`-ish) | argmin falls with p but k=1 and k=3 disagree ~2x at the same p |
| the Schur/DtN account of the factor | refuted, §5.2 |
| `outer_rings` 1/2 as production | 4-6x off on TOTAL time; on W7-X `o2` is slower than jacobi |
| the banded-capacitance route | exists to make `outer_rings` affordable, and they are 4-6x off before any banding |

**Rank by TOTAL TIME, not iterations.** Every arm costs the same per CG
iteration (the preconditioner apply is 0.09-0.20 ms, under 0.2% of one), so
build cost decides. rot-ellipse n=20 k=1 free: `fm3` 22.3s, **block@0.10 26.3s**,
`ibpd`@1.0 46.9s, jacobi 53.7s, `o1` 94.1s, `o2` 141.2s.

**Still alive but NOT production**: `fm`, a truncated-Fourier coarse correction
(`mrx/experimental/block_jacobi_coarse.py`). A genuine further 1.18-1.32x on
total time, but five parameters, storage LINEAR in `n_dof` (102 MB at n=20 vs
the atom's 0.1 MB, Table H), an ADDITIVE form that is structurally wrong for
this atom, `m95 ~ n_t/3` so a fixed mode box is asymptotically under-resolved,
and no coverage at k=0 or dbc.

## 9. Results — the production stack

Overnight 2026-08-22, 40 jobs, no failures, 168 cells. Block-Jacobi Laplacian at
`bc_scale=0.10` with `block_jacobi` mass. Full counts in Tables A and B.

**`bc_scale = 0.10` survives the mass swap.** Penalty against each cell's own
optimum, free BC, p=3:

| geom | k | n=8 | 12 | 16 | 20 | 24 | 28 | 32 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cylinder | 1 | 1.19 | 1.10 | 1.02 | 1.02 | 1.01 | 1.01 | **1.00** |
| toroid | 1 | 1.29 | 1.19 | 1.11 | 1.03 | 1.02 | 1.01 | **1.01** |
| rot-ellipse | 1 | 1.04 | 1.00 | 1.00 | 1.00 | 1.02 | 1.01 | **1.00** |
| w7x | 1 | 1.00 | 1.01 | 1.04 | 1.07 | 1.05 | 1.07 | **1.07** |
| cylinder | 3 | 1.15 | 1.05 | 1.00 | 1.00 | 1.01 | 1.00 | **1.00** |
| toroid | 3 | 1.24 | 1.11 | 1.05 | 1.00 | 1.00 | 1.00 | **1.00** |
| rot-ellipse | 3 | 1.07 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.01** |
| w7x | 3 | 1.14 | 1.02 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |

Median **1.01** over 96 free cells, IMPROVING with resolution. Worst anywhere is
1.55, and everything above 1.2 is at p=2 or n=8 -- worst exactly where the
problem is cheapest. In p at n=12: toroid k=3 1.55 (p=2) -> 1.11 (p=3) -> 1.00
(p=4) -> 1.00 (p=5); rot-ellipse k=1 is 1.00 at every p.

**Speedup vs point jacobi**, same runs:

| geom | k, bc | n=8 | 16 | 24 | 32 |
| --- | --- | --- | --- | --- | --- |
| toroid | 1 free | 0.29 | 0.19 | 0.14 | **0.12** |
| toroid | 3 free | 0.44 | 0.22 | 0.16 | **0.13** |
| cylinder | 1 free | 0.28 | 0.25 | 0.23 | **0.21** |
| w7x | 3 free | 0.54 | 0.31 | 0.23 | **0.19** |
| rot-ellipse | 1 free | 0.57 | 0.50 | 0.41 | **0.36** |
| w7x | 1 free | 0.56 | 0.67 | 0.60 | **0.52** |

Median **0.31** over 120 cells, **0.25 for n >= 24**, and the ratio IMPROVES
with refinement -- 3-8x at production resolution. The hardest cell is W7-X k=1
free (0.52), which has been the hardest case throughout.

**In p**: point jacobi degrades 7.6-12.2x over p = 2..5 while every block arm
grows only 2.3-2.8x, so the advantage grows strongly with degree.

## 10. Reading the numbers honestly

* **Iteration-count noise floor is ~1%, up to 2.4% at k=1 free** (measured from
  two nominally identical arms over 29 cells; the singular free rows carry a
  harmonic deflation that is not bit-reproducible). Ignore smaller differences.
* Two builds of the IDENTICAL configuration differ by ~1e-14 on ~1.7% of rows
  (the dense polar core is not bit-reproducible). Never assert bit-identity
  between builds.
* W7-X k=1 free had its `bc_scale` argmin at the BRACKET FLOOR in five separate
  sweeps. The curve is flat there (0.03 and 0.06 agree within noise while 0 is
  1.4x worse), so the true optimum is unknown and irrelevant -- 0.10 costs at
  most ~1.1 there.

## 11. Reproducing anything here

All sweeps go through slurm; nothing runs on the login node.

    SCRIPT=scripts/debug/verify_block_jacobi.py \
      ARGS="--geometry w7x --ns 12,24,12 --p 3 \
            --arms jacobi,bj_r0_nobc,bj_r0_ibpd_bcp10 \
            --ks 1,3 --bcs free,dbc --tol 1e-10 --maxiter 40000 \
            --out outputs/x/json/w7x.json" \
      JOB_NAME=x OUTSUB=x TIMEOUT_MIN=600 MEM_GB=96 \
      [EXTRA_ENV="MRX_MASS_KIND=raw_kron"] bash slurm/job_diag_run.sh

Arm grammar: `jacobi`, `probed_jacobi`, or `bj_rN[_oM]_<bc>[_bcpS][_fmM]` where
`_rN`/`_oM` are inner/outer exact rings (underscore-anchored), `<bc>` is `ibpd`
or `nobc`, `bcpS` sets the scale to S/100, and `fmM` selects the coarse
correction. Other harnesses: `block_jacobi_spectrum.py` (spectra),
`block_jacobi_mass_ab.py` (mass A/B), `bc_alpha_compare.py` (the coefficient),
`bc_schur_effective.py` (the ring-block predictor), `edge_vector_check.py`
(`e` and the stiffening ratio `r`).

Geometries: `cylinder` (zero angular metric variation), `toroid`
(epsilon=1/3, axisymmetric), `rot-ellipse` (eps=0.33, kappa=1.5, nfp=3), `w7x`
(nfp=5, from `data/W7-X.h5`).

---

# APPENDIX — the measurement tables

### TABLE A — new stack, iterations (p=3, tol 1e-10)

| geom | k | bc | n | jacobi | nobc | 0.06 | 0.10 | 0.15 | 0.30 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|---|
| cylinder | 1 | free | 8 | 226 | 110 | 75 | 63 | 59 | 53 | 53 |
| cylinder | 1 | free | 12 | 289 | 165 | 93 | 79 | 73 | 72 | 72 |
| cylinder | 1 | free | 16 | 363 | 203 | 106 | 89 | 88 | 87 | 87 |
| cylinder | 1 | free | 20 | 473 | 257 | 120 | 105 | 105 | 103 | 104 |
| cylinder | 1 | free | 24 | 544 | -- | 130 | 123 | 122 | 122 | -- |
| cylinder | 1 | free | 28 | 661 | -- | 144 | 141 | 140 | 140 | -- |
| cylinder | 1 | free | 32 | 742 | -- | 159 | 157 | 157 | 157 | -- |
| cylinder | 1 | dbc | 8 | 181 | 55 | 55 | 55 | 55 | 55 | 55 |
| cylinder | 1 | dbc | 12 | 231 | 73 | 72 | 72 | 73 | 72 | 72 |
| cylinder | 1 | dbc | 16 | 324 | 88 | 88 | 88 | 88 | 88 | 88 |
| cylinder | 1 | dbc | 20 | 412 | 107 | 108 | 108 | 108 | 108 | 108 |
| cylinder | 2 | free | 8 | 160 | 127 | 76 | 66 | 58 | 53 | 54 |
| cylinder | 2 | free | 12 | 234 | 183 | 93 | 79 | 73 | 73 | 73 |
| cylinder | 2 | free | 16 | 327 | 247 | 108 | 94 | 93 | 93 | 93 |
| cylinder | 2 | free | 20 | 414 | 304 | 120 | 110 | 110 | 110 | 110 |
| cylinder | 2 | dbc | 8 | 144 | 54 | 54 | 54 | 54 | 54 | 54 |
| cylinder | 2 | dbc | 12 | 213 | 76 | 71 | 74 | 76 | 76 | 76 |
| cylinder | 2 | dbc | 16 | 298 | 93 | 95 | 95 | 94 | 93 | 95 |
| cylinder | 2 | dbc | 20 | 378 | 114 | 108 | 114 | 114 | 114 | 114 |
| cylinder | 3 | free | 8 | 92 | 76 | 52 | 45 | 41 | 39 | 41 |
| cylinder | 3 | free | 12 | 140 | 123 | 72 | 63 | 60 | 60 | 60 |
| cylinder | 3 | free | 16 | 200 | 174 | 91 | 81 | 81 | 81 | 81 |
| cylinder | 3 | free | 20 | 267 | 219 | 107 | 100 | 100 | 101 | 100 |
| cylinder | 3 | free | 24 | 334 | -- | 121 | 120 | 119 | 120 | -- |
| cylinder | 3 | free | 28 | 383 | -- | 137 | 136 | 136 | 136 | -- |
| cylinder | 3 | free | 32 | 455 | -- | 154 | 154 | 154 | 155 | -- |
| cylinder | 3 | dbc | 8 | 86 | 38 | 38 | 38 | 38 | 38 | 38 |
| cylinder | 3 | dbc | 12 | 134 | 60 | 60 | 60 | 60 | 60 | 60 |
| cylinder | 3 | dbc | 16 | 192 | 81 | 81 | 81 | 81 | 81 | 81 |
| cylinder | 3 | dbc | 20 | 249 | 99 | 99 | 99 | 99 | 99 | 99 |
| toroid | 1 | free | 8 | 312 | 189 | 104 | 89 | 77 | 69 | 70 |
| toroid | 1 | free | 12 | 454 | 297 | 127 | 105 | 93 | 88 | 88 |
| toroid | 1 | free | 16 | 609 | 391 | 140 | 114 | 105 | 103 | 103 |
| toroid | 1 | free | 20 | 771 | 467 | 150 | 121 | 117 | 117 | 117 |
| toroid | 1 | free | 24 | 932 | -- | 157 | 130 | 129 | 128 | -- |
| toroid | 1 | free | 28 | 1081 | -- | 161 | 141 | 141 | 139 | -- |
| toroid | 1 | free | 32 | 1242 | -- | 163 | 151 | 150 | 150 | -- |
| toroid | 1 | dbc | 8 | 234 | 60 | 59 | 60 | 60 | 59 | 59 |
| toroid | 1 | dbc | 12 | 340 | 76 | 76 | 76 | 76 | 76 | 76 |
| toroid | 1 | dbc | 16 | 479 | 89 | 89 | 89 | 89 | 89 | 89 |
| toroid | 1 | dbc | 20 | 640 | 100 | 102 | 102 | 102 | 102 | 101 |
| toroid | 2 | free | 8 | 217 | 196 | 91 | 75 | 62 | 60 | 61 |
| toroid | 2 | free | 12 | 354 | 302 | 112 | 89 | 75 | 78 | 80 |
| toroid | 2 | free | 16 | 510 | 394 | 120 | 96 | 94 | 94 | 95 |
| toroid | 2 | free | 20 | 665 | 484 | 124 | 107 | 109 | 109 | 109 |
| toroid | 2 | dbc | 8 | 260 | 72 | 72 | 72 | 72 | 72 | 72 |
| toroid | 2 | dbc | 12 | 417 | 91 | 91 | 91 | 91 | 91 | 91 |
| toroid | 2 | dbc | 16 | 552 | 109 | 109 | 108 | 107 | 108 | 108 |
| toroid | 2 | dbc | 20 | 710 | 125 | 121 | 121 | 124 | 122 | 121 |
| toroid | 3 | free | 8 | 116 | 107 | 61 | 51 | 44 | 41 | 47 |
| toroid | 3 | free | 12 | 201 | 168 | 74 | 60 | 54 | 54 | 62 |
| toroid | 3 | free | 16 | 306 | 223 | 82 | 67 | 64 | 65 | 72 |
| toroid | 3 | free | 20 | 417 | 278 | 89 | 75 | 76 | 76 | 82 |
| toroid | 3 | free | 24 | 533 | -- | 95 | 86 | 86 | 86 | -- |
| toroid | 3 | free | 28 | 660 | -- | 98 | 96 | 96 | 96 | -- |
| toroid | 3 | free | 32 | 798 | -- | 106 | 106 | 106 | 106 | -- |
| toroid | 3 | dbc | 8 | 162 | 44 | 44 | 44 | 44 | 44 | 44 |
| toroid | 3 | dbc | 12 | 280 | 62 | 62 | 62 | 62 | 62 | 62 |
| toroid | 3 | dbc | 16 | 400 | 75 | 75 | 75 | 75 | 75 | 75 |
| toroid | 3 | dbc | 20 | 531 | 87 | 87 | 87 | 87 | 87 | 87 |
| rot-ellipse | 1 | free | 8 | 503 | 425 | 296 | 286 | 281 | 275 | 385 |
| rot-ellipse | 1 | free | 12 | 755 | 692 | 429 | 411 | 411 | 444 | 657 |
| rot-ellipse | 1 | free | 16 | 1094 | 914 | 543 | 542 | 553 | 605 | 930 |
| rot-ellipse | 1 | free | 20 | 1424 | 1076 | 651 | 649 | 649 | 753 | 1219 |
| rot-ellipse | 1 | free | 24 | 1829 | -- | 741 | 755 | 789 | 905 | -- |
| rot-ellipse | 1 | free | 28 | 2185 | -- | 844 | 849 | 879 | 1074 | -- |
| rot-ellipse | 1 | free | 32 | 2628 | -- | 936 | 939 | 1011 | 1211 | -- |
| rot-ellipse | 1 | dbc | 8 | 353 | 124 | 124 | 122 | 122 | 122 | 122 |
| rot-ellipse | 1 | dbc | 12 | 584 | 172 | 174 | 172 | 173 | 173 | 173 |
| rot-ellipse | 1 | dbc | 16 | 870 | 225 | 224 | 223 | 223 | 224 | 224 |
| rot-ellipse | 1 | dbc | 20 | 1176 | 274 | 275 | 275 | 275 | 273 | 275 |
| rot-ellipse | 2 | free | 8 | 392 | 470 | 295 | 282 | 271 | 280 | 382 |
| rot-ellipse | 2 | free | 12 | 649 | 786 | 455 | 439 | 437 | 477 | 704 |
| rot-ellipse | 2 | free | 16 | 918 | 1040 | 578 | 575 | 575 | 646 | 1002 |
| rot-ellipse | 2 | free | 20 | 1242 | 1286 | 700 | 690 | 711 | 823 | 1275 |
| rot-ellipse | 2 | dbc | 8 | 282 | 137 | 137 | 137 | 136 | 136 | 136 |
| rot-ellipse | 2 | dbc | 12 | 502 | 194 | 198 | 197 | 196 | 198 | 194 |
| rot-ellipse | 2 | dbc | 16 | 757 | 262 | 260 | 262 | 258 | 262 | 262 |
| rot-ellipse | 2 | dbc | 20 | 979 | 320 | 320 | 316 | 322 | 318 | 322 |
| rot-ellipse | 3 | free | 8 | 143 | 172 | 101 | 90 | 84 | 84 | 105 |
| rot-ellipse | 3 | free | 12 | 252 | 275 | 137 | 125 | 125 | 127 | 159 |
| rot-ellipse | 3 | free | 16 | 387 | 345 | 160 | 154 | 157 | 164 | 199 |
| rot-ellipse | 3 | free | 20 | 511 | 412 | 188 | 182 | 186 | 192 | 231 |
| rot-ellipse | 3 | free | 24 | 665 | -- | 209 | 210 | 213 | 221 | -- |
| rot-ellipse | 3 | free | 28 | 792 | -- | 232 | 233 | 238 | 245 | -- |
| rot-ellipse | 3 | free | 32 | 945 | -- | 253 | 255 | 257 | 264 | -- |
| rot-ellipse | 3 | dbc | 8 | 158 | 83 | 83 | 83 | 83 | 83 | 83 |
| rot-ellipse | 3 | dbc | 12 | 287 | 126 | 125 | 127 | 127 | 125 | 125 |
| rot-ellipse | 3 | dbc | 16 | 417 | 164 | 165 | 163 | 164 | 164 | 165 |
| rot-ellipse | 3 | dbc | 20 | 551 | 190 | 190 | 190 | 191 | 191 | 190 |
| w7x | 1 | free | 8 | 1043 | 829 | 593 | 584 | 599 | 657 | 917 |
| w7x | 1 | free | 12 | 1674 | 1524 | 1071 | 1087 | 1143 | 1295 | 1961 |
| w7x | 1 | free | 16 | 2299 | 2120 | 1482 | 1542 | 1658 | 1902 | 3044 |
| w7x | 1 | free | 20 | 3145 | 2659 | 1834 | 1959 | 2088 | 2472 | 4121 |
| w7x | 1 | free | 24 | 3887 | -- | 2209 | 2330 | 2471 | 3107 | -- |
| w7x | 1 | free | 28 | 4897 | -- | 2484 | 2650 | 2963 | 3579 | -- |
| w7x | 1 | free | 32 | 6021 | -- | 2904 | 3102 | 3438 | 4196 | -- |
| w7x | 1 | dbc | 8 | 467 | 169 | 172 | 170 | 169 | 169 | 170 |
| w7x | 1 | dbc | 12 | 853 | 254 | 254 | 254 | 254 | 256 | 254 |
| w7x | 1 | dbc | 16 | 1366 | 347 | 346 | 346 | 346 | 346 | 347 |
| w7x | 1 | dbc | 20 | 1955 | 437 | 440 | 439 | 439 | 437 | 436 |
| w7x | 2 | free | 8 | 877 | 880 | 588 | 583 | 583 | 616 | 860 |
| w7x | 2 | free | 12 | 1550 | 1715 | 1138 | 1138 | 1184 | 1324 | 1960 |
| w7x | 2 | free | 16 | 2376 | 2474 | 1581 | 1655 | 1759 | 2064 | 3204 |
| w7x | 2 | free | 20 | 3191 | 3171 | 2013 | 2134 | 2314 | 2799 | 4394 |
| w7x | 2 | dbc | 8 | 471 | 200 | 201 | 196 | 202 | 198 | 200 |
| w7x | 2 | dbc | 12 | 953 | 325 | 326 | 327 | 323 | 317 | 325 |
| w7x | 2 | dbc | 16 | 1441 | 452 | 452 | 452 | 450 | 448 | 455 |
| w7x | 2 | dbc | 20 | 2060 | 574 | 580 | 571 | 577 | 569 | 584 |
| w7x | 3 | free | 8 | 181 | 203 | 109 | 97 | 89 | 85 | 99 |
| w7x | 3 | free | 12 | 336 | 318 | 146 | 130 | 127 | 129 | 152 |
| w7x | 3 | free | 16 | 524 | 420 | 170 | 161 | 161 | 165 | 196 |
| w7x | 3 | free | 20 | 710 | 499 | 192 | 189 | 189 | 195 | 230 |
| w7x | 3 | free | 24 | 921 | -- | 213 | 212 | 212 | 219 | -- |
| w7x | 3 | free | 28 | 1150 | -- | 233 | 233 | 236 | 243 | -- |
| w7x | 3 | free | 32 | 1372 | -- | 257 | 256 | 256 | 262 | -- |
| w7x | 3 | dbc | 8 | 242 | 88 | 88 | 88 | 88 | 88 | 88 |
| w7x | 3 | dbc | 12 | 444 | 131 | 130 | 129 | 130 | 130 | 130 |
| w7x | 3 | dbc | 16 | 650 | 167 | 167 | 167 | 168 | 167 | 167 |
| w7x | 3 | dbc | 20 | 871 | 195 | 197 | 195 | 196 | 196 | 196 |

### TABLE B — new stack, p-sweep at n=12 (24,12), iterations

| geom | k | bc | p | jacobi | nobc | 0.06 | 0.10 | 0.15 | 0.30 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|---|
| cylinder | 1 | free | 2 | 134 | 121 | 77 | 66 | 58 | 52 | 51 |
| cylinder | 1 | free | 4 | 876 | 211 | 111 | 102 | 99 | 96 | 97 |
| cylinder | 1 | free | 5 | 3121 | 256 | 138 | 125 | 125 | 123 | 126 |
| cylinder | 1 | dbc | 2 | 132 | 52 | 51 | 51 | 52 | 52 | 52 |
| cylinder | 1 | dbc | 4 | 740 | 98 | 97 | 98 | 98 | 98 | 97 |
| cylinder | 1 | dbc | 5 | 2653 | 123 | 123 | 122 | 122 | 121 | 122 |
| cylinder | 3 | free | 2 | 108 | 113 | 73 | 62 | 55 | 47 | 47 |
| cylinder | 3 | free | 4 | 355 | 136 | 76 | 71 | 71 | 72 | 75 |
| cylinder | 3 | free | 5 | 1166 | 153 | 86 | 85 | 84 | 86 | 88 |
| cylinder | 3 | dbc | 2 | 115 | 48 | 48 | 48 | 48 | 48 | 48 |
| cylinder | 3 | dbc | 4 | 342 | 72 | 72 | 72 | 72 | 72 | 72 |
| cylinder | 3 | dbc | 5 | 1094 | 84 | 84 | 84 | 84 | 84 | 84 |
| toroid | 1 | free | 2 | 357 | 264 | 128 | 107 | 91 | 79 | 77 |
| toroid | 1 | free | 4 | 998 | 335 | 132 | 108 | 103 | 101 | 107 |
| toroid | 1 | free | 5 | 3635 | 394 | 143 | 126 | 121 | 120 | 130 |
| toroid | 1 | dbc | 2 | 273 | 67 | 67 | 67 | 67 | 67 | 67 |
| toroid | 1 | dbc | 4 | 815 | 87 | 87 | 87 | 87 | 87 | 87 |
| toroid | 1 | dbc | 5 | 3047 | 105 | 105 | 105 | 105 | 105 | 105 |
| toroid | 3 | free | 2 | 180 | 146 | 74 | 62 | 53 | 40 | 40 |
| toroid | 3 | free | 4 | 421 | 213 | 85 | 73 | 73 | 76 | 89 |
| toroid | 3 | free | 5 | 1311 | 257 | 99 | 92 | 95 | 97 | 115 |
| toroid | 3 | dbc | 2 | 262 | 51 | 51 | 51 | 51 | 51 | 51 |
| toroid | 3 | dbc | 4 | 443 | 75 | 75 | 75 | 75 | 75 | 75 |
| toroid | 3 | dbc | 5 | 1246 | 94 | 94 | 94 | 94 | 94 | 94 |
| rot-ellipse | 1 | free | 2 | 535 | 550 | 362 | 346 | 347 | 358 | 470 |
| rot-ellipse | 1 | free | 4 | 1820 | 737 | 476 | 465 | 471 | 542 | 828 |
| rot-ellipse | 1 | free | 5 | 6217 | 870 | 538 | 528 | 539 | 629 | 989 |
| rot-ellipse | 1 | dbc | 2 | 388 | 141 | 139 | 141 | 141 | 141 | 141 |
| rot-ellipse | 1 | dbc | 4 | 1365 | 200 | 200 | 200 | 200 | 201 | 200 |
| rot-ellipse | 1 | dbc | 5 | 4814 | 249 | 247 | 248 | 248 | 248 | 249 |
| rot-ellipse | 3 | free | 2 | 199 | 196 | 107 | 93 | 83 | 75 | 87 |
| rot-ellipse | 3 | free | 4 | 562 | 381 | 189 | 186 | 189 | 194 | 243 |
| rot-ellipse | 3 | free | 5 | 2000 | 493 | 268 | 264 | 266 | 283 | 354 |
| rot-ellipse | 3 | dbc | 2 | 261 | 81 | 83 | 83 | 83 | 82 | 83 |
| rot-ellipse | 3 | dbc | 4 | 559 | 187 | 189 | 187 | 189 | 189 | 189 |
| rot-ellipse | 3 | dbc | 5 | 1945 | 267 | 267 | 265 | 265 | 266 | 266 |
| w7x | 1 | free | 2 | 1332 | 1236 | 877 | 891 | 917 | 988 | 1346 |
| w7x | 1 | free | 4 | 3410 | 1765 | 1279 | 1339 | 1331 | 1678 | 2554 |
| w7x | 1 | free | 5 | 9827 | 1941 | 1383 | 1474 | 1569 | 1908 | 3074 |
| w7x | 1 | dbc | 2 | 598 | 207 | 209 | 207 | 207 | 207 | 209 |
| w7x | 1 | dbc | 4 | 1852 | 299 | 299 | 299 | 299 | 298 | 297 |
| w7x | 1 | dbc | 5 | 6171 | 356 | 355 | 355 | 357 | 356 | 355 |
| w7x | 3 | free | 2 | 303 | 232 | 114 | 99 | 89 | 78 | 85 |
| w7x | 3 | free | 4 | 722 | 469 | 213 | 202 | 204 | 207 | 251 |
| w7x | 3 | free | 5 | 2640 | 654 | 321 | 309 | 311 | 319 | 389 |
| w7x | 3 | dbc | 2 | 447 | 95 | 94 | 94 | 96 | 96 | 96 |
| w7x | 3 | dbc | 4 | 713 | 204 | 205 | 204 | 206 | 203 | 205 |
| w7x | 3 | dbc | 5 | 2672 | 307 | 306 | 305 | 307 | 305 | 307 |

### TABLE D — MASS solve M_k x = b, CG to 1e-8, iterations (build s)

| geom | n | p | k | bc | jacobi | raw_kron | block_jacobi | bj/rk |
|---|---|---|---|---|---|---|---|---|
| cylinder | 12 | 3 | 0 | free | 598 | 7 (2.5s) | 9 (1.6s) | 1.29 |
| cylinder | 12 | 3 | 0 | dbc | 556 | 8 (1.4s) | 9 (1.6s) | 1.12 |
| cylinder | 12 | 3 | 1 | free | 478 | 11 (2.7s) | 11 (2.4s) | 1.00 |
| cylinder | 12 | 3 | 1 | dbc | 458 | 12 (2.2s) | 11 (2.4s) | 0.92 |
| cylinder | 12 | 3 | 2 | free | 301 | 9 (2.2s) | 7 (2.3s) | 0.78 |
| cylinder | 12 | 3 | 2 | dbc | 292 | 10 (2.1s) | 7 (2.3s) | 0.70 |
| cylinder | 12 | 3 | 3 | free | 171 | 6 (0.7s) | 5 (0.7s) | 0.83 |
| cylinder | 12 | 3 | 3 | dbc | 168 | 6 (0.7s) | 5 (0.7s) | 0.83 |
| cylinder | 16 | 3 | 0 | free | 699 | 7 (2.6s) | 8 (1.7s) | 1.14 |
| cylinder | 16 | 3 | 0 | dbc | 675 | 8 (1.6s) | 8 (1.7s) | 1.00 |
| cylinder | 16 | 3 | 1 | free | 487 | 11 (2.8s) | 11 (2.5s) | 1.00 |
| cylinder | 16 | 3 | 1 | dbc | 480 | 11 (2.2s) | 11 (2.5s) | 1.00 |
| cylinder | 16 | 3 | 2 | free | 307 | 9 (2.3s) | 7 (2.3s) | 0.78 |
| cylinder | 16 | 3 | 2 | dbc | 304 | 10 (2.3s) | 7 (2.3s) | 0.70 |
| cylinder | 16 | 3 | 3 | free | 186 | 5 (0.7s) | 5 (0.7s) | 1.00 |
| cylinder | 16 | 3 | 3 | dbc | 184 | 5 (0.7s) | 5 (0.7s) | 1.00 |
| cylinder | 20 | 3 | 0 | free | 735 | 7 (2.7s) | 8 (1.7s) | 1.14 |
| cylinder | 20 | 3 | 0 | dbc | 708 | 8 (1.4s) | 8 (1.7s) | 1.00 |
| cylinder | 20 | 3 | 1 | free | 492 | 11 (2.9s) | 11 (2.6s) | 1.00 |
| cylinder | 20 | 3 | 1 | dbc | 477 | 11 (2.3s) | 11 (2.6s) | 1.00 |
| cylinder | 20 | 3 | 2 | free | 313 | 9 (2.3s) | 7 (2.3s) | 0.78 |
| cylinder | 20 | 3 | 2 | dbc | 303 | 10 (2.3s) | 7 (2.3s) | 0.70 |
| cylinder | 20 | 3 | 3 | free | 188 | 5 (0.7s) | 5 (0.7s) | 1.00 |
| cylinder | 20 | 3 | 3 | dbc | 189 | 5 (0.7s) | 5 (0.7s) | 1.00 |
| cylinder | 8 | 3 | 0 | free | 379 | 8 (2.4s) | 9 (1.6s) | 1.12 |
| cylinder | 8 | 3 | 0 | dbc | 327 | 9 (1.4s) | 9 (1.5s) | 1.00 |
| cylinder | 8 | 3 | 1 | free | 430 | 11 (2.8s) | 12 (2.3s) | 1.09 |
| cylinder | 8 | 3 | 1 | dbc | 404 | 12 (2.1s) | 12 (2.3s) | 1.00 |
| cylinder | 8 | 3 | 2 | free | 275 | 9 (2.2s) | 8 (2.2s) | 0.89 |
| cylinder | 8 | 3 | 2 | dbc | 265 | 11 (2.1s) | 8 (2.2s) | 0.73 |
| cylinder | 8 | 3 | 3 | free | 129 | 6 (0.7s) | 5 (0.7s) | 0.83 |
| cylinder | 8 | 3 | 3 | dbc | 128 | 6 (0.7s) | 5 (0.7s) | 0.83 |
| rot-ellipse | 12 | 3 | 0 | free | 643 | 15 (2.5s) | 16 (1.6s) | 1.07 |
| rot-ellipse | 12 | 3 | 0 | dbc | 617 | 15 (1.4s) | 16 (1.6s) | 1.07 |
| rot-ellipse | 12 | 3 | 1 | free | 606 | 44 (2.8s) | 31 (2.4s) | 0.70 |
| rot-ellipse | 12 | 3 | 1 | dbc | 583 | 43 (2.2s) | 31 (2.4s) | 0.72 |
| rot-ellipse | 12 | 3 | 2 | free | 394 | 42 (2.2s) | 31 (2.3s) | 0.74 |
| rot-ellipse | 12 | 3 | 2 | dbc | 377 | 42 (2.2s) | 32 (2.3s) | 0.76 |
| rot-ellipse | 12 | 3 | 3 | free | 178 | 8 (0.7s) | 6 (0.7s) | 0.75 |
| rot-ellipse | 12 | 3 | 3 | dbc | 176 | 8 (0.7s) | 6 (0.7s) | 0.75 |
| rot-ellipse | 16 | 3 | 0 | free | 722 | 15 (2.5s) | 16 (1.6s) | 1.07 |
| rot-ellipse | 16 | 3 | 0 | dbc | 699 | 15 (1.4s) | 16 (1.6s) | 1.07 |
| rot-ellipse | 16 | 3 | 1 | free | 636 | 46 (2.8s) | 32 (2.5s) | 0.70 |
| rot-ellipse | 16 | 3 | 1 | dbc | 621 | 46 (2.2s) | 32 (2.5s) | 0.70 |
| rot-ellipse | 16 | 3 | 2 | free | 410 | 44 (2.2s) | 31 (2.3s) | 0.70 |
| rot-ellipse | 16 | 3 | 2 | dbc | 399 | 44 (2.3s) | 31 (2.3s) | 0.70 |
| rot-ellipse | 16 | 3 | 3 | free | 190 | 7 (0.7s) | 6 (0.7s) | 0.86 |
| rot-ellipse | 16 | 3 | 3 | dbc | 189 | 7 (0.7s) | 6 (0.7s) | 0.86 |
| rot-ellipse | 20 | 3 | 0 | free | 747 | 14 (2.7s) | 14 (1.7s) | 1.00 |
| rot-ellipse | 20 | 3 | 0 | dbc | 721 | 15 (1.4s) | 15 (1.7s) | 1.00 |
| rot-ellipse | 20 | 3 | 1 | free | 658 | 47 (2.9s) | 33 (2.7s) | 0.70 |
| rot-ellipse | 20 | 3 | 1 | dbc | 632 | 47 (2.3s) | 32 (2.6s) | 0.68 |
| rot-ellipse | 20 | 3 | 2 | free | 422 | 45 (2.3s) | 31 (2.4s) | 0.69 |
| rot-ellipse | 20 | 3 | 2 | dbc | 407 | 45 (2.4s) | 31 (2.4s) | 0.69 |
| rot-ellipse | 20 | 3 | 3 | free | 191 | 7 (0.7s) | 6 (0.7s) | 0.86 |
| rot-ellipse | 20 | 3 | 3 | dbc | 190 | 7 (0.7s) | 6 (0.7s) | 0.86 |
| rot-ellipse | 8 | 3 | 0 | free | 488 | 17 (2.5s) | 18 (1.6s) | 1.06 |
| rot-ellipse | 8 | 3 | 0 | dbc | 445 | 17 (1.4s) | 18 (1.5s) | 1.06 |
| rot-ellipse | 8 | 3 | 1 | free | 536 | 39 (2.9s) | 30 (2.4s) | 0.77 |
| rot-ellipse | 8 | 3 | 1 | dbc | 497 | 39 (2.2s) | 30 (2.4s) | 0.77 |
| rot-ellipse | 8 | 3 | 2 | free | 342 | 37 (2.3s) | 30 (2.3s) | 0.81 |
| rot-ellipse | 8 | 3 | 2 | dbc | 333 | 37 (2.2s) | 29 (2.3s) | 0.78 |
| rot-ellipse | 8 | 3 | 3 | free | 147 | 9 (0.7s) | 7 (0.7s) | 0.78 |
| rot-ellipse | 8 | 3 | 3 | dbc | 146 | 9 (0.7s) | 7 (0.7s) | 0.78 |
| toroid | 12 | 3 | 0 | free | 625 | 9 (2.5s) | 11 (1.6s) | 1.22 |
| toroid | 12 | 3 | 0 | dbc | 592 | 10 (1.4s) | 11 (1.6s) | 1.10 |
| toroid | 12 | 3 | 1 | free | 501 | 13 (2.8s) | 14 (2.4s) | 1.08 |
| toroid | 12 | 3 | 1 | dbc | 483 | 13 (2.2s) | 13 (2.4s) | 1.00 |
| toroid | 12 | 3 | 2 | free | 322 | 12 (2.2s) | 9 (2.3s) | 0.75 |
| toroid | 12 | 3 | 2 | dbc | 304 | 12 (2.2s) | 9 (2.3s) | 0.75 |
| toroid | 12 | 3 | 3 | free | 176 | 7 (0.7s) | 5 (0.7s) | 0.71 |
| toroid | 12 | 3 | 3 | dbc | 173 | 7 (0.7s) | 5 (0.7s) | 0.71 |
| toroid | 16 | 3 | 0 | free | 709 | 9 (2.5s) | 10 (1.7s) | 1.11 |
| toroid | 16 | 3 | 0 | dbc | 686 | 10 (1.4s) | 10 (1.6s) | 1.00 |
| toroid | 16 | 3 | 1 | free | 512 | 13 (2.8s) | 13 (2.5s) | 1.00 |
| toroid | 16 | 3 | 1 | dbc | 497 | 13 (2.2s) | 13 (2.5s) | 1.00 |
| toroid | 16 | 3 | 2 | free | 323 | 12 (2.3s) | 9 (2.3s) | 0.75 |
| toroid | 16 | 3 | 2 | dbc | 324 | 12 (2.3s) | 9 (2.3s) | 0.75 |
| toroid | 16 | 3 | 3 | free | 188 | 6 (0.7s) | 5 (0.7s) | 0.83 |
| toroid | 16 | 3 | 3 | dbc | 187 | 6 (0.7s) | 5 (0.7s) | 0.83 |
| toroid | 20 | 3 | 0 | free | 742 | 9 (2.7s) | 10 (1.7s) | 1.11 |
| toroid | 20 | 3 | 0 | dbc | 712 | 10 (1.4s) | 10 (1.7s) | 1.00 |
| toroid | 20 | 3 | 1 | free | 525 | 13 (2.9s) | 11 (2.7s) | 0.85 |
| toroid | 20 | 3 | 1 | dbc | 503 | 13 (2.4s) | 11 (2.7s) | 0.85 |
| toroid | 20 | 3 | 2 | free | 328 | 11 (2.3s) | 9 (2.4s) | 0.82 |
| toroid | 20 | 3 | 2 | dbc | 309 | 12 (2.3s) | 9 (2.4s) | 0.75 |
| toroid | 20 | 3 | 3 | free | 190 | 6 (0.7s) | 5 (0.7s) | 0.83 |
| toroid | 20 | 3 | 3 | dbc | 190 | 6 (0.7s) | 5 (0.7s) | 0.83 |
| toroid | 8 | 3 | 0 | free | 447 | 11 (2.5s) | 12 (1.6s) | 1.09 |
| toroid | 8 | 3 | 0 | dbc | 391 | 11 (1.4s) | 12 (1.5s) | 1.09 |
| toroid | 8 | 3 | 1 | free | 443 | 14 (2.9s) | 15 (2.3s) | 1.07 |
| toroid | 8 | 3 | 1 | dbc | 425 | 14 (2.1s) | 15 (2.3s) | 1.07 |
| toroid | 8 | 3 | 2 | free | 290 | 13 (2.2s) | 10 (2.2s) | 0.77 |
| toroid | 8 | 3 | 2 | dbc | 287 | 13 (2.1s) | 10 (2.2s) | 0.77 |
| toroid | 8 | 3 | 3 | free | 140 | 8 (0.7s) | 6 (0.7s) | 0.75 |
| toroid | 8 | 3 | 3 | dbc | 140 | 7 (0.7s) | 6 (0.7s) | 0.86 |
| w7x | 12 | 3 | 0 | free | 639 | 14 (2.4s) | 15 (1.6s) | 1.07 |
| w7x | 12 | 3 | 0 | dbc | 621 | 15 (1.4s) | 15 (1.6s) | 1.00 |
| w7x | 12 | 3 | 1 | free | 665 | 76 (2.7s) | 53 (2.4s) | 0.70 |
| w7x | 12 | 3 | 1 | dbc | 624 | 71 (2.4s) | 50 (2.4s) | 0.70 |
| w7x | 12 | 3 | 2 | free | 435 | 75 (2.2s) | 53 (2.3s) | 0.71 |
| w7x | 12 | 3 | 2 | dbc | 422 | 71 (2.2s) | 52 (2.3s) | 0.73 |
| w7x | 12 | 3 | 3 | free | 178 | 7 (0.7s) | 6 (0.7s) | 0.86 |
| w7x | 12 | 3 | 3 | dbc | 174 | 7 (0.7s) | 6 (0.7s) | 0.86 |
| w7x | 16 | 3 | 0 | free | 720 | 14 (2.4s) | 14 (1.7s) | 1.00 |
| w7x | 16 | 3 | 0 | dbc | 695 | 14 (1.4s) | 14 (1.6s) | 1.00 |
| w7x | 16 | 3 | 1 | free | 681 | 83 (2.8s) | 55 (2.5s) | 0.66 |
| w7x | 16 | 3 | 1 | dbc | 667 | 78 (2.4s) | 53 (2.6s) | 0.68 |
| w7x | 16 | 3 | 2 | free | 463 | 83 (2.3s) | 57 (2.3s) | 0.69 |
| w7x | 16 | 3 | 2 | dbc | 446 | 78 (2.2s) | 55 (2.3s) | 0.71 |
| w7x | 16 | 3 | 3 | free | 190 | 7 (0.7s) | 6 (0.7s) | 0.86 |
| w7x | 16 | 3 | 3 | dbc | 188 | 7 (0.7s) | 6 (0.7s) | 0.86 |
| w7x | 20 | 3 | 0 | free | 749 | 13 (2.5s) | 14 (1.7s) | 1.08 |
| w7x | 20 | 3 | 0 | dbc | 720 | 13 (1.4s) | 14 (1.7s) | 1.08 |
| w7x | 20 | 3 | 1 | free | 704 | 88 (3.1s) | 58 (2.7s) | 0.66 |
| w7x | 20 | 3 | 1 | dbc | 684 | 83 (2.2s) | 55 (2.7s) | 0.66 |
| w7x | 20 | 3 | 2 | free | 477 | 87 (2.5s) | 59 (2.4s) | 0.68 |
| w7x | 20 | 3 | 2 | dbc | 456 | 83 (2.2s) | 58 (2.4s) | 0.70 |
| w7x | 20 | 3 | 3 | free | 190 | 7 (0.7s) | 5 (0.7s) | 0.71 |
| w7x | 20 | 3 | 3 | dbc | 190 | 7 (0.7s) | 5 (0.7s) | 0.71 |
| w7x | 8 | 3 | 0 | free | 494 | 16 (2.3s) | 16 (1.6s) | 1.00 |
| w7x | 8 | 3 | 0 | dbc | 440 | 16 (1.4s) | 16 (1.5s) | 1.00 |
| w7x | 8 | 3 | 1 | free | 578 | 62 (2.7s) | 48 (2.3s) | 0.77 |
| w7x | 8 | 3 | 1 | dbc | 544 | 57 (2.1s) | 45 (2.3s) | 0.79 |
| w7x | 8 | 3 | 2 | free | 378 | 60 (2.2s) | 47 (2.2s) | 0.78 |
| w7x | 8 | 3 | 2 | dbc | 360 | 56 (2.1s) | 46 (2.2s) | 0.82 |
| w7x | 8 | 3 | 3 | free | 150 | 8 (0.7s) | 7 (0.7s) | 0.88 |
| w7x | 8 | 3 | 3 | dbc | 144 | 8 (0.7s) | 7 (0.7s) | 0.88 |
| cylinder | 12 | 2 | 0 | free | 173 | 7 (1.7s) | 8 (0.9s) | 1.14 |
| cylinder | 12 | 2 | 0 | dbc | 164 | 8 (0.7s) | 8 (0.9s) | 1.00 |
| cylinder | 12 | 2 | 1 | free | 120 | 10 (1.8s) | 10 (1.4s) | 1.00 |
| cylinder | 12 | 2 | 1 | dbc | 115 | 10 (1.1s) | 10 (1.4s) | 1.00 |
| cylinder | 12 | 2 | 2 | free | 76 | 8 (1.2s) | 7 (1.2s) | 0.88 |
| cylinder | 12 | 2 | 2 | dbc | 74 | 8 (1.1s) | 7 (1.2s) | 0.88 |
| cylinder | 12 | 2 | 3 | free | 46 | 5 (0.4s) | 5 (0.4s) | 1.00 |
| cylinder | 12 | 2 | 3 | dbc | 47 | 5 (0.4s) | 5 (0.4s) | 1.00 |
| cylinder | 12 | 4 | 0 | free | 1995 | 8 (3.9s) | 9 (3.1s) | 1.12 |
| cylinder | 12 | 4 | 0 | dbc | 1887 | 9 (3.0s) | 9 (3.1s) | 1.00 |
| cylinder | 12 | 4 | 1 | free | 1901 | 12 (5.0s) | 12 (4.6s) | 1.00 |
| cylinder | 12 | 4 | 1 | dbc | 1825 | 12 (4.4s) | 12 (4.6s) | 1.00 |
| cylinder | 12 | 4 | 2 | free | 1185 | 10 (4.5s) | 8 (4.5s) | 0.80 |
| cylinder | 12 | 4 | 2 | dbc | 1156 | 10 (4.4s) | 8 (4.5s) | 0.80 |
| cylinder | 12 | 4 | 3 | free | 579 | 6 (1.5s) | 5 (1.5s) | 0.83 |
| cylinder | 12 | 4 | 3 | dbc | 579 | 6 (1.5s) | 5 (1.5s) | 0.83 |
| cylinder | 12 | 5 | 0 | free | 5000 | 9 (7.0s) | 10 (6.0s) | 1.11 |
| cylinder | 12 | 5 | 0 | dbc | 5000 | 10 (5.9s) | 10 (6.0s) | 1.00 |
| cylinder | 12 | 5 | 1 | free | 5000 | 13 (9.4s) | 13 (9.0s) | 1.00 |
| cylinder | 12 | 5 | 1 | dbc | 5000 | 14 (8.7s) | 13 (9.1s) | 0.93 |
| cylinder | 12 | 5 | 2 | free | 4789 | 11 (8.8s) | 9 (8.9s) | 0.82 |
| cylinder | 12 | 5 | 2 | dbc | 4464 | 12 (8.7s) | 9 (8.9s) | 0.75 |
| cylinder | 12 | 5 | 3 | free | 1910 | 6 (2.9s) | 6 (2.9s) | 1.00 |
| cylinder | 12 | 5 | 3 | dbc | 1908 | 6 (2.9s) | 6 (2.9s) | 1.00 |
| rot-ellipse | 12 | 2 | 0 | free | 179 | 13 (1.9s) | 13 (0.9s) | 1.00 |
| rot-ellipse | 12 | 2 | 0 | dbc | 170 | 13 (0.7s) | 13 (0.9s) | 1.00 |
| rot-ellipse | 12 | 2 | 1 | free | 158 | 37 (1.7s) | 27 (1.3s) | 0.73 |
| rot-ellipse | 12 | 2 | 1 | dbc | 154 | 37 (1.1s) | 27 (1.3s) | 0.73 |
| rot-ellipse | 12 | 2 | 2 | free | 105 | 37 (1.1s) | 28 (1.2s) | 0.76 |
| rot-ellipse | 12 | 2 | 2 | dbc | 102 | 37 (1.1s) | 28 (1.2s) | 0.76 |
| rot-ellipse | 12 | 2 | 3 | free | 47 | 6 (0.4s) | 5 (0.4s) | 0.83 |
| rot-ellipse | 12 | 2 | 3 | dbc | 47 | 6 (0.4s) | 5 (0.4s) | 0.83 |
| rot-ellipse | 12 | 4 | 0 | free | 2380 | 19 (4.2s) | 20 (3.1s) | 1.05 |
| rot-ellipse | 12 | 4 | 0 | dbc | 2281 | 19 (2.9s) | 20 (3.0s) | 1.05 |
| rot-ellipse | 12 | 4 | 1 | free | 2443 | 48 (5.0s) | 35 (4.6s) | 0.73 |
| rot-ellipse | 12 | 4 | 1 | dbc | 2356 | 48 (4.3s) | 34 (4.6s) | 0.71 |
| rot-ellipse | 12 | 4 | 2 | free | 1546 | 49 (4.4s) | 35 (4.4s) | 0.71 |
| rot-ellipse | 12 | 4 | 2 | dbc | 1473 | 48 (4.3s) | 34 (4.5s) | 0.71 |
| rot-ellipse | 12 | 4 | 3 | free | 650 | 9 (1.4s) | 7 (1.4s) | 0.78 |
| rot-ellipse | 12 | 4 | 3 | dbc | 650 | 10 (1.4s) | 7 (1.5s) | 0.70 |
| rot-ellipse | 12 | 5 | 0 | free | 5000 | 23 (7.0s) | 23 (6.0s) | 1.00 |
| rot-ellipse | 12 | 5 | 0 | dbc | 5000 | 24 (5.8s) | 24 (6.0s) | 1.00 |
| rot-ellipse | 12 | 5 | 1 | free | 5000 | 57 (9.2s) | 41 (8.7s) | 0.72 |
| rot-ellipse | 12 | 5 | 1 | dbc | 5000 | 57 (8.7s) | 42 (9.0s) | 0.74 |
| rot-ellipse | 12 | 5 | 2 | free | 5000 | 55 (8.8s) | 39 (8.9s) | 0.71 |
| rot-ellipse | 12 | 5 | 2 | dbc | 5000 | 54 (8.6s) | 38 (8.8s) | 0.70 |
| rot-ellipse | 12 | 5 | 3 | free | 2425 | 11 (2.9s) | 8 (2.9s) | 0.73 |
| rot-ellipse | 12 | 5 | 3 | dbc | 2457 | 12 (2.9s) | 8 (2.9s) | 0.67 |
| toroid | 12 | 2 | 0 | free | 176 | 8 (1.8s) | 10 (0.9s) | 1.25 |
| toroid | 12 | 2 | 0 | dbc | 167 | 8 (0.8s) | 10 (0.9s) | 1.25 |
| toroid | 12 | 2 | 1 | free | 127 | 11 (1.7s) | 11 (1.3s) | 1.00 |
| toroid | 12 | 2 | 1 | dbc | 121 | 11 (1.1s) | 10 (1.3s) | 0.91 |
| toroid | 12 | 2 | 2 | free | 80 | 10 (1.1s) | 8 (1.2s) | 0.80 |
| toroid | 12 | 2 | 2 | dbc | 76 | 10 (1.1s) | 8 (1.2s) | 0.80 |
| toroid | 12 | 2 | 3 | free | 47 | 5 (0.4s) | 5 (0.4s) | 1.00 |
| toroid | 12 | 2 | 3 | dbc | 47 | 5 (0.4s) | 5 (0.4s) | 1.00 |
| toroid | 12 | 4 | 0 | free | 2258 | 11 (3.9s) | 12 (3.1s) | 1.09 |
| toroid | 12 | 4 | 0 | dbc | 2162 | 12 (3.0s) | 12 (3.1s) | 1.00 |
| toroid | 12 | 4 | 1 | free | 1972 | 15 (5.0s) | 15 (4.6s) | 1.00 |
| toroid | 12 | 4 | 1 | dbc | 1870 | 16 (4.3s) | 15 (4.6s) | 0.94 |
| toroid | 12 | 4 | 2 | free | 1249 | 15 (4.4s) | 10 (4.5s) | 0.67 |
| toroid | 12 | 4 | 2 | dbc | 1251 | 14 (4.3s) | 10 (4.5s) | 0.71 |
| toroid | 12 | 4 | 3 | free | 614 | 8 (1.5s) | 6 (1.5s) | 0.75 |
| toroid | 12 | 4 | 3 | dbc | 610 | 8 (1.5s) | 6 (1.5s) | 0.75 |
| toroid | 12 | 5 | 0 | free | 5000 | 13 (7.0s) | 13 (6.0s) | 1.00 |
| toroid | 12 | 5 | 0 | dbc | 5000 | 14 (5.8s) | 13 (6.0s) | 0.93 |
| toroid | 12 | 5 | 1 | free | 5000 | 18 (9.3s) | 17 (9.0s) | 0.94 |
| toroid | 12 | 5 | 1 | dbc | 5000 | 18 (8.7s) | 17 (9.0s) | 0.94 |
| toroid | 12 | 5 | 2 | free | 5000 | 17 (8.6s) | 12 (8.7s) | 0.71 |
| toroid | 12 | 5 | 2 | dbc | 4881 | 16 (8.6s) | 12 (8.7s) | 0.75 |
| toroid | 12 | 5 | 3 | free | 2233 | 9 (2.9s) | 6 (2.9s) | 0.67 |
| toroid | 12 | 5 | 3 | dbc | 2215 | 9 (2.9s) | 6 (2.9s) | 0.67 |
| w7x | 12 | 2 | 0 | free | 178 | 11 (1.7s) | 12 (0.9s) | 1.09 |
| w7x | 12 | 2 | 0 | dbc | 168 | 11 (0.7s) | 12 (0.9s) | 1.09 |
| w7x | 12 | 2 | 1 | free | 168 | 65 (1.7s) | 48 (1.4s) | 0.74 |
| w7x | 12 | 2 | 1 | dbc | 163 | 61 (1.1s) | 45 (1.4s) | 0.74 |
| w7x | 12 | 2 | 2 | free | 115 | 64 (1.2s) | 49 (1.2s) | 0.77 |
| w7x | 12 | 2 | 2 | dbc | 109 | 59 (1.1s) | 46 (1.2s) | 0.78 |
| w7x | 12 | 2 | 3 | free | 47 | 6 (0.4s) | 5 (0.4s) | 0.83 |
| w7x | 12 | 2 | 3 | dbc | 47 | 6 (0.4s) | 5 (0.4s) | 0.83 |
| w7x | 12 | 4 | 0 | free | 2358 | 18 (3.9s) | 18 (3.0s) | 1.00 |
| w7x | 12 | 4 | 0 | dbc | 2250 | 19 (2.9s) | 18 (3.0s) | 0.95 |
| w7x | 12 | 4 | 1 | free | 2638 | 90 (4.9s) | 66 (4.5s) | 0.73 |
| w7x | 12 | 4 | 1 | dbc | 2489 | 84 (4.3s) | 63 (4.5s) | 0.75 |
| w7x | 12 | 4 | 2 | free | 1718 | 90 (4.4s) | 64 (4.4s) | 0.71 |
| w7x | 12 | 4 | 2 | dbc | 1667 | 83 (4.3s) | 60 (4.4s) | 0.72 |
| w7x | 12 | 4 | 3 | free | 627 | 10 (1.4s) | 8 (1.4s) | 0.80 |
| w7x | 12 | 4 | 3 | dbc | 632 | 10 (1.4s) | 8 (1.4s) | 0.80 |
| w7x | 12 | 5 | 0 | free | 5000 | 23 (6.7s) | 24 (6.0s) | 1.04 |
| w7x | 12 | 5 | 0 | dbc | 5000 | 24 (5.7s) | 23 (5.9s) | 0.96 |
| w7x | 12 | 5 | 1 | free | 5000 | 115 (9.5s) | 93 (8.9s) | 0.81 |
| w7x | 12 | 5 | 1 | dbc | 5000 | 107 (8.6s) | 89 (8.9s) | 0.83 |
| w7x | 12 | 5 | 2 | free | 5000 | 112 (8.7s) | 85 (8.8s) | 0.76 |
| w7x | 12 | 5 | 2 | dbc | 5000 | 102 (8.6s) | 80 (8.8s) | 0.78 |
| w7x | 12 | 5 | 3 | free | 2345 | 12 (2.9s) | 10 (2.9s) | 0.83 |
| w7x | 12 | 5 | 3 | dbc | 2332 | 12 (2.9s) | 10 (2.9s) | 0.83 |

### TABLE E — jacobi (modelled diag) vs probed_jacobi (exact diag) vs block@0.10

| geom | n | k | bc | jacobi | probed | probed/jac | block | blk/jac | blk/probed |
|---|---|---|---|---|---|---|---|---|---|
| cylinder | 12 | 0 | free | 269 | 269 | 1.00 | 35 | 0.13 | 0.13 |
| cylinder | 12 | 0 | dbc | 209 | 209 | 1.00 | 36 | 0.17 | 0.17 |
| cylinder | 12 | 1 | free | 263 | 261 | 0.99 | 70 | 0.27 | 0.27 |
| cylinder | 12 | 1 | dbc | 236 | 235 | 1.00 | 68 | 0.29 | 0.29 |
| cylinder | 12 | 2 | free | 209 | 210 | 1.00 | 75 | 0.36 | 0.36 |
| cylinder | 12 | 2 | dbc | 208 | 205 | 0.99 | 78 | 0.38 | 0.38 |
| cylinder | 12 | 3 | free | 127 | 126 | 0.99 | 63 | 0.50 | 0.50 |
| cylinder | 12 | 3 | dbc | 131 | 130 | 0.99 | 67 | 0.51 | 0.52 |
| cylinder | 8 | 0 | free | 197 | 196 | 0.99 | 25 | 0.13 | 0.13 |
| cylinder | 8 | 0 | dbc | 149 | 149 | 1.00 | 26 | 0.17 | 0.17 |
| cylinder | 8 | 1 | free | 226 | 227 | 1.00 | 56 | 0.25 | 0.25 |
| cylinder | 8 | 1 | dbc | 194 | 193 | 0.99 | 50 | 0.26 | 0.26 |
| cylinder | 8 | 2 | free | 163 | 162 | 0.99 | 58 | 0.36 | 0.36 |
| cylinder | 8 | 2 | dbc | 153 | 155 | 1.01 | 52 | 0.34 | 0.34 |
| cylinder | 8 | 3 | free | 87 | 89 | 1.02 | 46 | 0.53 | 0.52 |
| cylinder | 8 | 3 | dbc | 88 | 86 | 0.98 | 42 | 0.48 | 0.49 |
| rot-ellipse | 12 | 0 | free | 411 | 410 | 1.00 | 81 | 0.20 | 0.20 |
| rot-ellipse | 12 | 0 | dbc | 241 | 241 | 1.00 | 69 | 0.29 | 0.29 |
| rot-ellipse | 12 | 1 | free | 746 | 610 | 0.82 | 421 | 0.56 | 0.69 |
| rot-ellipse | 12 | 1 | dbc | 569 | 524 | 0.92 | 215 | 0.38 | 0.41 |
| rot-ellipse | 12 | 2 | free | 621 | 541 | 0.87 | 452 | 0.73 | 0.84 |
| rot-ellipse | 12 | 2 | dbc | 498 | 470 | 0.94 | 236 | 0.47 | 0.50 |
| rot-ellipse | 12 | 3 | free | 233 | 202 | 0.87 | 140 | 0.60 | 0.69 |
| rot-ellipse | 12 | 3 | dbc | 288 | 249 | 0.86 | 127 | 0.44 | 0.51 |
| rot-ellipse | 8 | 0 | free | 302 | 301 | 1.00 | 56 | 0.19 | 0.19 |
| rot-ellipse | 8 | 0 | dbc | 190 | 191 | 1.01 | 45 | 0.24 | 0.24 |
| rot-ellipse | 8 | 1 | free | 530 | 454 | 0.86 | 285 | 0.54 | 0.63 |
| rot-ellipse | 8 | 1 | dbc | 375 | 350 | 0.93 | 151 | 0.40 | 0.43 |
| rot-ellipse | 8 | 2 | free | 405 | 361 | 0.89 | 285 | 0.70 | 0.79 |
| rot-ellipse | 8 | 2 | dbc | 301 | 278 | 0.92 | 160 | 0.53 | 0.58 |
| rot-ellipse | 8 | 3 | free | 144 | 124 | 0.86 | 96 | 0.67 | 0.77 |
| rot-ellipse | 8 | 3 | dbc | 162 | 147 | 0.91 | 84 | 0.52 | 0.57 |
| toroid | 12 | 0 | free | 398 | 395 | 0.99 | 50 | 0.13 | 0.13 |
| toroid | 12 | 0 | dbc | 234 | 234 | 1.00 | 39 | 0.17 | 0.17 |
| toroid | 12 | 1 | free | 442 | 440 | 1.00 | 109 | 0.25 | 0.25 |
| toroid | 12 | 1 | dbc | 378 | 377 | 1.00 | 92 | 0.24 | 0.24 |
| toroid | 12 | 2 | free | 348 | 347 | 1.00 | 95 | 0.27 | 0.27 |
| toroid | 12 | 2 | dbc | 473 | 471 | 1.00 | 111 | 0.23 | 0.24 |
| toroid | 12 | 3 | free | 188 | 186 | 0.99 | 63 | 0.34 | 0.34 |
| toroid | 12 | 3 | dbc | 306 | 308 | 1.01 | 70 | 0.23 | 0.23 |
| toroid | 8 | 0 | free | 298 | 297 | 1.00 | 35 | 0.12 | 0.12 |
| toroid | 8 | 0 | dbc | 173 | 173 | 1.00 | 27 | 0.16 | 0.16 |
| toroid | 8 | 1 | free | 316 | 315 | 1.00 | 93 | 0.29 | 0.30 |
| toroid | 8 | 1 | dbc | 262 | 256 | 0.98 | 71 | 0.27 | 0.28 |
| toroid | 8 | 2 | free | 214 | 210 | 0.98 | 79 | 0.37 | 0.38 |
| toroid | 8 | 2 | dbc | 299 | 287 | 0.96 | 87 | 0.29 | 0.30 |
| toroid | 8 | 3 | free | 113 | 113 | 1.00 | 52 | 0.46 | 0.46 |
| toroid | 8 | 3 | dbc | 179 | 179 | 1.00 | 49 | 0.27 | 0.27 |
| w7x | 12 | 0 | free | 528 | 526 | 1.00 | 98 | 0.19 | 0.19 |
| w7x | 12 | 0 | dbc | 269 | 269 | 1.00 | 64 | 0.24 | 0.24 |
| w7x | 12 | 1 | free | 1663 | 1344 | 0.81 | 1105 | 0.66 | 0.82 |
| w7x | 12 | 1 | dbc | 857 | 758 | 0.88 | 320 | 0.37 | 0.42 |
| w7x | 12 | 2 | free | 1551 | 1430 | 0.92 | 1160 | 0.75 | 0.81 |
| w7x | 12 | 2 | dbc | 1015 | 806 | 0.79 | 401 | 0.40 | 0.50 |
| w7x | 12 | 3 | free | 305 | 256 | 0.84 | 143 | 0.47 | 0.56 |
| w7x | 12 | 3 | dbc | 442 | 372 | 0.84 | 142 | 0.32 | 0.38 |
| w7x | 8 | 0 | free | 370 | 369 | 1.00 | 71 | 0.19 | 0.19 |
| w7x | 8 | 0 | dbc | 200 | 200 | 1.00 | 43 | 0.21 | 0.21 |
| w7x | 8 | 1 | free | 1052 | 934 | 0.89 | 590 | 0.56 | 0.63 |
| w7x | 8 | 1 | dbc | 521 | 499 | 0.96 | 208 | 0.40 | 0.42 |
| w7x | 8 | 2 | free | 887 | 833 | 0.94 | 595 | 0.67 | 0.71 |
| w7x | 8 | 2 | dbc | 544 | 494 | 0.91 | 248 | 0.46 | 0.50 |
| w7x | 8 | 3 | free | 173 | 148 | 0.86 | 103 | 0.60 | 0.70 |
| w7x | 8 | 3 | dbc | 257 | 240 | 0.93 | 89 | 0.35 | 0.37 |
### TABLE F — cond(PL), outliers and lambda_min vs bc_scale

6,12,6, p=3, free BC (`diag_bcpspec`). The two outlier families move in
OPPOSITE directions with the scale; that trade IS the optimum.

**cylinder**

| k | scale | cond(PL) | high outliers | low outliers | lambda_min |
|---|---|---|---|---|---|
| 1 | 0.00 | 486 | 23 | 0 | 0.3341 |
| 1 | 0.06 | 61 | 11 | 0 | 0.3340 |
| 1 | 0.10 | 41 | 5 | 0 | 0.3339 |
| 1 | 0.15 | 31 | 1 | 0 | 0.3338 |
| 1 | 0.22 | 26 | 1 | 0 | 0.3336 |
| 1 | 0.30 | 23 | 0 | 0 | 0.3333 |
| 1 | 0.55 | 20 | 0 | 0 | 0.3323 |
| 1 | 1.00 | 19 | 0 | 0 | 0.3274 |
| 3 | 0.00 | 195 | 9 | 0 | 0.5494 |
| 3 | 0.06 | 23 | 1 | 0 | 0.5433 |
| 3 | 0.10 | 15 | 0 | 0 | 0.5390 |
| 3 | 0.15 | 10 | 0 | 0 | 0.5333 |
| 3 | 0.22 | 9 | 0 | 0 | 0.5248 |
| 3 | 0.30 | 9 | 0 | 0 | 0.5143 |
| 3 | 0.55 | 10 | 0 | 0 | 0.4766 |
| 3 | 1.00 | 12 | 0 | 0 | 0.3985 |

**rot-ellipse**

| k | scale | cond(PL) | high outliers | low outliers | lambda_min |
|---|---|---|---|---|---|
| 1 | 0.00 | 3118 | 56 | 8 | 0.0662 |
| 1 | 0.06 | 626 | 46 | 9 | 0.0649 |
| 1 | 0.10 | 596 | 42 | 9 | 0.0639 |
| 1 | 0.15 | 581 | 41 | 11 | 0.0626 |
| 1 | 0.22 | 577 | 37 | 11 | 0.0606 |
| 1 | 0.30 | 605 | 34 | 13 | 0.0563 |
| 1 | 0.55 | 788 | 28 | 19 | 0.0414 |
| 1 | 1.00 | 1224 | 28 | 27 | 0.0260 |
| 3 | 0.00 | 443 | 15 | 0 | 0.1821 |
| 3 | 0.06 | 65 | 2 | 0 | 0.1760 |
| 3 | 0.10 | 47 | 0 | 0 | 0.1717 |
| 3 | 0.15 | 40 | 0 | 0 | 0.1661 |
| 3 | 0.22 | 43 | 0 | 0 | 0.1579 |
| 3 | 0.30 | 45 | 0 | 0 | 0.1483 |
| 3 | 0.55 | 56 | 0 | 0 | 0.1199 |
| 3 | 1.00 | 80 | 1 | 3 | 0.0840 |

**w7x**

| k | scale | cond(PL) | high outliers | low outliers | lambda_min |
|---|---|---|---|---|---|
| 1 | 0.00 | 8464 | 67 | 18 | 0.0286 |
| 1 | 0.06 | 2634 | 61 | 19 | 0.0258 |
| 1 | 0.10 | 2698 | 59 | 20 | 0.0241 |
| 1 | 0.15 | 2842 | 57 | 22 | 0.0221 |
| 1 | 0.22 | 3100 | 54 | 25 | 0.0198 |
| 1 | 0.30 | 3433 | 52 | 27 | 0.0175 |
| 1 | 0.55 | 4729 | 49 | 33 | 0.0123 |
| 1 | 1.00 | 7303 | 48 | 42 | 0.0078 |
| 3 | 0.00 | 450 | 17 | 0 | 0.1517 |
| 3 | 0.06 | 82 | 5 | 0 | 0.1490 |
| 3 | 0.10 | 60 | 1 | 0 | 0.1471 |
| 3 | 0.15 | 47 | 0 | 0 | 0.1446 |
| 3 | 0.22 | 42 | 0 | 0 | 0.1408 |
| 3 | 0.30 | 43 | 0 | 0 | 0.1360 |
| 3 | 0.55 | 50 | 1 | 0 | 0.1191 |
| 3 | 1.00 | 66 | 2 | 0 | 0.0896 |


### TABLE G — the ring-block predictor (`diag_bcschur`), k=1

`B_raw = L[R,R]` on the outer rings; `A(s) = inv(P(s))[R,R]`.

| geom | mesh | DtN removes, depth 1 | depth 4 | argmin cond(B_raw,A) d3 |
|---|---|---|---|---|
| cylinder | 6,12,6 | 17.7% | 0.5% | 0.55 |
| toroid | 6,12,6 | 17.6% | 0.4% | 0.55 |
| rot-ellipse | 6,12,6 | 23.8% | 1.0% | 0.22 |
| rot-ellipse | 8,16,8 | 25.6% | 0.9% | 0.15 |
| w7x | 6,12,6 | 29.1% | 1.1% | 0.06 |
| w7x | 8,16,8 | 34.4% | 1.0% | 0.06 |

### TABLE H — storage and build (rot-ellipse k=1, `diag_fmcost`)

| n | arm | build | MB total | MB coarse | MB core |
|---|---|---|---|---|---|
| 12 | ibpd_r3 | 50.0s | 11.8 | 0.0 | 11.8 |
| 12 | ibpd_r3_fm2 | 29.8s | 22.3 | 10.5 | 11.8 |
| 12 | ibpd_r3_fm3 | 30.9s | 32.4 | 20.6 | 11.8 |
| 12 | ibpd_r3_fm3_fr2 | 34.1s | 53.4 | 41.6 | 11.8 |
| 12 | ibpd_r3_o1 | 49.2s | 34.5 | 0.0 | 34.5 |
| 16 | ibpd_r0 | 23.7s | 0.1 | 0.0 | 0.1 |
| 16 | ibpd_r0_fm2 | 3.8s | 26.0 | 25.9 | 0.1 |
| 16 | ibpd_r0_fm3 | 5.4s | 51.0 | 50.9 | 0.1 |
| 16 | ibpd_r0_o1 | 37.8s | 20.9 | 0.0 | 20.9 |
| 16 | ibpd_r0_o2 | 74.9s | 79.5 | 0.0 | 79.5 |
| 20 | ibpd_r0 | 26.6s | 0.1 | 0.0 | 0.1 |
| 20 | ibpd_r0_fm3 | 6.3s | 102.2 | 102.0 | 0.1 |
| 20 | ibpd_r0_o1 | 59.6s | 50.1 | 0.0 | 50.0 |
