> **Status:** in production since 2026-09-02 (`apply_inverse_mass_plus_eps_laplace_matrix`, every k)
> **Read this for:** the split identity for `(M_k + eps L_k)^-1`, its proof, the measured iteration counts against the saddle MINRES it replaced, and the shifted-stiffness atom that preconditions each split system (§6)
> **Do not read for:** the Hodge split of the unshifted Laplacian (`precond_h_scaling_2026-09-02.md`) or the construction of the metric-lumped atoms themselves (`preconditioner_technical_note_source.md`)

# The shifted Laplacian `M_k + eps L_k` as two SPD solves

## 1. Objects

All matrices act on extracted DoF vectors of one boundary-condition family
(Dirichlet or natural), as in `precond_h_scaling_2026-09-02.md`.

| symbol | definition | properties |
|---|---|---|
| `M_k` | mass matrix of k-forms | SPD |
| `D_k` | strong exterior derivative, k-form coefficients to (k+1)-form coefficients | `D_k D_{k-1} = 0` |
| `S_k = D_k^T M_{k+1} D_k` | strong stiffness ("delta d": grad div on 2-forms) | symmetric positive semidefinite; kernel = closed forms |
| `K_k = M_k D_{k-1} M_{k-1}^{-1} D_{k-1}^T M_k` | weak half ("d delta": curl curl on 2-forms) | symmetric positive semidefinite; kernel = co-closed forms |
| `L_k = S_k + K_k` | Hodge Laplacian | symmetric positive semidefinite |

The velocity smoothing `(M_2 + mu L_2) v = M_2 F` and the resistive step
`(M_2 + eps L_2) delta = -eps L_2 B` both solve with `M_k + eps L_k`, which
is SPD for every `eps >= 0`. Until 2026-09-02 that solve was MINRES on the
saddle system whose Schur complement is `M_k + eps L_k`, because applying
`K_k` needs `M_{k-1}^{-1}` and a Krylov solve must not nest inside another.
The smoothing solve was 75% of a relaxation step (GitHub issue #18).

## 2. The identity

For every `k >= 1` and every `eps`,

```
(M_k + eps L_k)^{-1}  =  (M_k + eps S_k)^{-1}  -  eps D_{k-1} (M_{k-1} + eps S_{k-1})^{-1} D_{k-1}^T .      (*)
```

Both matrices being inverted on the right are a mass matrix plus a
positive semidefinite stiffness, hence SPD, and both are applied
matrix-free without any inner inverse. Nothing is approximated: (*) is an
algebraic identity that uses only `D_k D_{k-1} = 0`.

## 3. Proof

Abbreviate, for a fixed `k`,

```
M = M_k,   N = M_{k-1},   D = D_{k-1},
A = M + eps S_k            (SPD),
B = N + eps S_{k-1}        (SPD).
```

Note that `D^T M D = D_{k-1}^T M_k D_{k-1} = S_{k-1}`, so `K_k = M D N^{-1} D^T M`.

**Lemma 1 (the stiffness kills exact forms).**
`S_k D = 0` and `D^T S_k = 0`.

*Proof.* `S_k D = D_k^T M_{k+1} (D_k D_{k-1}) = 0`; transpose for the second.

**Lemma 2.** `A D = M D`, hence `A^{-1} M D = D`; and `D^T A = D^T M`, hence `D^T M A^{-1} = D^T`.

*Proof.* `A D = M D + eps S_k D = M D` by Lemma 1; multiply by `A^{-1}` on the left. Transpose for the second pair.

### 3a. Via the Woodbury identity

Write the operator as `A + U C V` with

```
U = M D,      C = eps N^{-1},      V = D^T M,
```

so that `U C V = eps M D N^{-1} D^T M = eps K_k` and `A + U C V = M + eps S_k + eps K_k = M + eps L_k`.

The Woodbury identity reads

```
(A + U C V)^{-1} = A^{-1} - A^{-1} U (C^{-1} + V A^{-1} U)^{-1} V A^{-1} .
```

Evaluate the three pieces with Lemma 2:

```
A^{-1} U         = A^{-1} M D            = D ,
V A^{-1}         = D^T M A^{-1}          = D^T ,
V A^{-1} U       = D^T M A^{-1} M D      = D^T M D        = S_{k-1} ,
C^{-1} + V A^{-1} U = N / eps + S_{k-1}  = B / eps .
```

Substituting,

```
(M + eps L_k)^{-1} = A^{-1} - D (B / eps)^{-1} D^T = A^{-1} - eps D B^{-1} D^T ,
```

which is (*). Woodbury needs `A`, `C` and `C^{-1} + V A^{-1} U` invertible: `A` and `N` are SPD and `B / eps` is SPD. ∎

### 3b. By direct multiplication (no Woodbury)

Let `X = A^{-1} - eps D B^{-1} D^T`. We show `X (M + eps L_k) = I`.
Since `M + eps L_k = A + eps K_k`,

```
X (A + eps K_k) = X A + eps X K_k .
```

*First term.* Using Lemma 2 (`D^T A = D^T M`):

```
X A = I - eps D B^{-1} D^T A = I - eps D B^{-1} D^T M .
```

*Second term.* Two ingredients. By Lemma 2, `A^{-1} K_k = A^{-1} M D N^{-1} D^T M = D N^{-1} D^T M`.
And `D^T K_k = (D^T M D) N^{-1} D^T M = S_{k-1} N^{-1} D^T M`. Hence

```
eps X K_k = eps D N^{-1} D^T M  -  eps^2 D B^{-1} S_{k-1} N^{-1} D^T M .
```

*Sum.* Everything except `I` has the form `eps D [ ... ] D^T M`:

```
X (A + eps K_k) = I + eps D [ -B^{-1} + N^{-1} - eps B^{-1} S_{k-1} N^{-1} ] D^T M .
```

Factor `B^{-1}` out of the bracket on the left:

```
-B^{-1} + N^{-1} - eps B^{-1} S_{k-1} N^{-1}
   = B^{-1} [ -I + B N^{-1} - eps S_{k-1} N^{-1} ]
   = B^{-1} [ -I + (N + eps S_{k-1}) N^{-1} - eps S_{k-1} N^{-1} ]
   = B^{-1} [ -I + I + eps S_{k-1} N^{-1} - eps S_{k-1} N^{-1} ]
   = 0 .
```

So `X (M + eps L_k) = I`, and since everything is square, `X = (M + eps L_k)^{-1}`. ∎

### 3c. What it says in Hodge language

In the `M`-inner product, `M^{-1} S_k = delta d` and `M^{-1} K_k = d delta`
act on complementary pieces of the Hodge decomposition
`V_k = exact + coexact + harmonic`, and each vanishes on the other's range
(because `d d = 0`). So `I + eps M^{-1} L_k` is block diagonal:
`I + eps delta d` on the coexact part, `I + eps d delta` on the exact part,
`I` on the harmonic part. The first solve of (*) handles the coexact and
harmonic parts correctly and returns `M^{-1} b` on the exact part; the
second term corrects the exact part, using the intertwining
`(I + eps d delta)^{-1} d = d (I + eps delta d)^{-1}` to move the solve one
level down, where `delta d` is the matrix-free stiffness `S_{k-1}`. The
inner `M_{k-1}^{-1}` of `K_k` never has to be applied.

Special cases: `k = 0` is the first solve alone (there is no `D_{-1}`),
and at `k = 3` the first solve is a mass solve because `S_3 = 0`.

## 4. Numerical confirmation of the premises

The identity is only as exact as `D_k D_{k-1} = 0` and the agreement between
the strong `D_{k-1}`, the weak `M_k D_{k-1}` and the stiffness
`S_{k-1} = D_{k-1}^T M_k D_{k-1}` in the code, which on a polar sequence go
through the analytic axis stencils. Measured on li383, k = 2 Dirichlet,
p = 3, float64, three meshes, Gaussian test vectors:

| check | value |
|---|---|
| `\|\|M_2 D_1 v - (weak D_1) v\|\| / \|\|(weak D_1) v\|\|` | 2.5e-16 |
| `\|\|D_1^T M_2 w - (weak D_1)^T w\|\| / \|\|(weak D_1)^T w\|\|` | 2.7e-16 to 3.1e-16 |
| `\|\|D_1^T M_2 D_1 v - S_1 v\|\| / \|\|S_1 v\|\|` | 3.0e-16 to 3.2e-16 |
| `\|\|S_2 D_1 v\|\| / \|\|S_2 w\|\|` | 2.0e-16 to 2.5e-16 |

## 5. Measured iteration counts

li383 (`data/wout_li383_low_res_reference.nc`), p = 3, float64, Dirichlet
k = 2, tol `sqrt(eps)` = 1.5e-8, `rhs = M_2 u` for Gaussian `u`. "PR 19" is
the `(1/eps) P_L` upper-block kind proposed in that pull request. Counts
are MINRES iterations for the saddle solves and the sum of the two CG
counts for the split; the last column is `||x_split - x_MINRES||_M / ||x_MINRES||_M`.
Probe: `~/.claude/jobs/b4470d78/tmp/probe_split_shifted.py`.

| mesh | eps | eps n_r^2 | MINRES, mass atom | MINRES, PR 19 | split (k2 + k1) | split vs MINRES |
|---|---|---|---|---|---|---|
| (8,16,8) | 1e-6 | | 497 | 3623 | 100 (46 + 54) | 3.6e-8 |
| (8,16,8) | 1e-4 | | 745 | 2947 | 144 (66 + 78) | 4.2e-8 |
| (8,16,8) | 1e-3 | 0.064 | 2134 | 1675 | 330 (152 + 178) | 5.0e-8 |
| (8,16,8) | 1e-2 | | 5798 | 943 | 808 (396 + 412) | 8.2e-8 |
| (12,24,12) | 4.44e-4 | 0.064 | 8478 | 3650 | 772 (373 + 399) | 7.7e-8 |
| (12,24,12) | 1e-3 | | 13093 | 2842 | 1057 (520 + 537) | 9.6e-8 |
| (16,32,16) | 2.5e-4 | 0.064 | 20362 | 6397 | 1326 (650 + 676) | 8.8e-8 |

At the production `eps = 0.064 / n_r^2` the split needs 6.5x / 11x / 15x
fewer iterations than the MINRES with the same atom and 5.1x / 4.7x / 4.8x
fewer than PR 19, and it wins at every eps, so there is no crossover to
pick a kind by. The residual `||(M + eps L) x - b|| / ||b||` through the
exact nested Laplacian is 6e-8 to 7e-7 for the split and 3e-8 to 1e-7 for
MINRES; both are at tolerance, the split's being two solves' worth.

## 6. The preconditioner of `M_k + eps S_k`

The counts above used the mass atom on both split systems. On the closed
forms of `M_j + eps S_j` (the kernel of `S_j`) that atom is right, since the
operator is exactly `M_j` there; on the coexact part it leaves
`kappa ~ 1 + eps lambda_max(M^{-1} S)`, and the count grows
330 -> 772 -> 1326, about `n_r^2`.

The metric-lumped Laplacian atom (`metric_lumping_laplacian.py`) is, per
component, a Kronecker SUM of three 1-D terms in a common per-axis
eigenbasis (`V_a^T m_a V_a = I`, `V_a^T K_a V_a = diag(lam_a)`, `m_a` the
UNWEIGHTED 1-D mass) inside the sandwich `D_c^{-1/2} (.) D_c^{-1/2}`. The
strong half `S_k` is the terms on the component's PRIMAL axes (k=2: axis
`c`; k=1: the two axes other than `c`), the weak half `K_k` the terms on
the derivative axes. Keeping the strong-half terms only and dividing by
`1 + eps lambda` instead of `lambda` in the eigenbasis gives exactly

```
(M^ + eps S^)^-1,      M^ = D_c^{1/2} (m_r x m_t x m_z) D_c^{1/2}
```

for the atom's own separable mass: `M^-1` as `eps -> 0`, `(1/eps) S^-1`
as `eps -> inf`. The core rows get the dense `(M + eps S)^-1` from probed
`M` and `S` core blocks, formed once per solve (eps may be traced). This is
`MetricLumpingLaplacian.shifted_stiffness_apply` and the diffusion slot's
`'auto'`.

Measured, CG iterations on `(M_k + eps S_k) x = M_k u`, li383 p=3,
Dirichlet, `eps = 0.064 / n_r^2`, tol 1.5e-8. `jointJ` / `jointM` are two
"consistent" alternatives that put the Jacobian into the 1-D masses
(bundled `J` profiles with matching stiffness profiles; the mass atom's
exact-diagonal sandwich `Lam_c` with `<g^{aa}>` stiffness profiles).
Probe: `~/.claude/jobs/b4470d78/tmp/probe_shifted_atoms.py`.

| mesh | k | mass atom | shifted S atom | jointJ | jointM |
|---|---|---|---|---|---|
| (8,16,8) | 2 | 153 | **69** | 100 | 99 |
| (8,16,8) | 1 | 181 | **74** | 111 | 108 |
| (12,24,12) | 2 | 371 | **117** | 203 | 199 |
| (12,24,12) | 1 | 422 | **128** | 222 | 222 |

The plain shift wins by 2.2x / 3.2x over the mass atom and by 1.4x / 1.7x
over the Jacobian-consistent variants, so the Jacobian in the implied mass
is not the limiting factor. Growth is `n_r^1.3` (69 -> 117) against
`n_r^2.2` for the mass atom. Through the production entry point the whole
smoothing solve is then 145 / 249 iterations at (8,16,8) / (12,24,12),
from 2134 / 8478 with the saddle MINRES.

The same at (12,24,12) p=3 on the other geometries (analytic `toroid`;
`wout_LandremanPaul2021_QA_lowres.nc`, nfp 2;
`wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc`, nfp 5):

| geometry | k | mass atom | shifted S atom | jointJ | jointM |
|---|---|---|---|---|---|
| toroid | 2 | 82 | **44** | 87 | 79 |
| toroid | 1 | 99 | **49** | 103 | 97 |
| QA | 2 | 618 | **205** | 348 | 335 |
| QA | 1 | 734 | **220** | 424 | 408 |
| W7-X | 2 | 233 | **89** | 151 | 146 |
| W7-X | 1 | 270 | **84** | 171 | 167 |

1.9x to 3.3x on every geometry; on the toroid, where the metric is
orthogonal and the lumping is closest to exact, the Jacobian-consistent
variants are no better than the mass atom, so their loss is not a
stellarator artefact.

**Where it stops paying.** The atom's implied mass
`D_c^{1/2} (m x m x m) D_c^{1/2}` is a worse approximation of `M_k` than
the mass atom, which reproduces `diag(M_k)` exactly, so as `eps -> 0` the
shifted atom loses. Through the production entry point, li383 (8,16,8),
both split solves together, shifted atom vs the same split with the mass
atom: `eps = 1e-6` 195 vs 100; `1e-4` 150 vs 145; `1e-3` (smoothing) 145
vs 329; `1e-2` 309 vs 810. The crossover is near `eps n_r^2 ~ 0.006`. The
smoothing eps sits 10x above it; the resistive step's `eta dt`
(1e-6 .. 1e-4 in the production range) sits below it and pays up to 2x on
~100 iterations. The shifted atom is the default because the smoothing
solve is the cost that matters (issue #18).

## 7. What is left

The `n_r^1.3` growth. The atom drops the cross-component blocks of `S_k`
(`d_c^T M_{k+1} d_{c'}` for `c != c'`, which for div-div and curl-curl are
not small) and does not couple the polar core to the bulk; either could be
the residual mechanism. Both are measurable with the same probe.
