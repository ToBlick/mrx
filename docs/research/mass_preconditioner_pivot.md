> **Status:** superseded by preconditioner_technical_note_source.md
> **Read this for:** the E+ pseudoinverse analysis that explains why the dense polar core is worth it
> **Do not read for:** the production mass preconditioner; raw_kron is deleted

# Mass preconditioners: pivot from Schur to the pow2 extraction sandwich

**Decided 2026-08-17.** Two independent outcomes in this document:

- **Mass matrices `M_k`, k=0..3 (sections 1-6):** pivot from the surgery/Schur
  split to the pow2 extraction sandwich. Decided, pending the k=2/k=3 and
  stellarator gates in Phase 3.
- **k=0 Laplacian (section 7):** the opposite conclusion -- the core Schur
  **stays**, but its atom is replaced by a modal-radial solve (1.7-2.3x fewer
  iterations, mesh-independent, drop-in). Open handoff item: rotating ellipse.

All numbers measured on CPU, polar toroid + elongated torus, p=2/3/4.
`docs/source/concepts/PRODUCTION.md` should be updated when Phase 5 lands, not before.

## Decision

Mass solves move to **pow2**: a per-component diagonally-scaled Kronecker
inverse applied on the full raw grid, lifted and projected with the
Moore-Penrose pseudoinverse of the extraction operator. The surgery/bulk split,
the dense surgery Schur complement, the `A_ss` probe and the `coupling_sb`
block are all removed from the mass path.

The trade is explicit and accepted: **~1.4x solve wall clock** in exchange for
deleting 801 lines across 47 definitions (27% of `mrx/preconditioners.py`),
most of the tensor-mass setup cost, and a storage term that reaches an
estimated 24 GB at 64x128x64.

## 1. What pow2 is

The preconditioner never splits the space. It applies a Kronecker model on the
full raw (unextracted) grid and moves between raw and extracted coordinates
with the pseudoinverse `E+ = E^T (E E^T)^-1`:

```
M^-1  ~  (E+)^T [ (+)_c  D_c^-1/2 ( M_r^-1 x M_t^-1 x M_z^-1 )_c  D_c^-1/2 ] E+

  M_a      unweighted 1D masses, degree p (p-1 on the differentiated axis)
  D_c,a    = M_aa / (M_unw)_aa = int(w phi_a^2) / int(phi_a^2)
  (EE^T)^-1 = diag( (CC^T)^-1 , I )   -- identity on every bulk row
```

`D` is the phi^2-weighted support average of the metric weight, taken straight
from the exact mass diagonal. It is the **same model class** as the existing
Greville-collocation sandwich -- both are `M_ab ~ sqrt(v_a v_b) (M_unw)_ab` --
differing only in whether the weight is sampled at a point or averaged over the
support. The averaged form is what makes the model well-defined on the
innermost rings, where a Greville point sits at `r ~ 0` and `J -> 0`
(`_build_greville_mass_block_factors` already carries an eps-nudge and a median
fallback for exactly this, and is restricted to `radial_start = 2`).

**The pseudoinverse is not optional.** Using `E^T` in its place (`pow0`) costs
2.3x the iterations at k=1 and drifts under refinement, because the mis-scaled
subspace has dimension O(n_z) and grows. Applying the correction once rather
than twice (`pow1`) recovers about half of it. Both sides must carry the full
`(EE^T)^-1`.

### Apply, before and after

```
REMOVED -- surgery + Schur                ADOPTED -- pow2 sandwich
  split rows -> surgery / bulk              x <- (EE^T)^-1 x     (blocks <= 3x3)
  y = bulk_inv(r_b)          [Kron #1]      lift to raw: E^T     (sparse scatter)
  z = Sigma^-1(r_s - M_sb y) [dense+coup]   (+)_c D^-1/2 Kron D^-1/2  [Kron, once]
  x_b = y - bulk_inv(M_bs z) [Kron #2+coup] project: E, then (EE^T)^-1
  needs A_ss, Sigma^-1, coupling_sb         needs 1D masses, D, (CC^T)^-1
  storage O(N*n_z)                          storage O(n_z)
```

## 2. Evidence

CG to relative residual 1e-10, polar toroid, p=3, fully jitted PCG loop (no
per-iteration host sync). Wall clock is min-of-5.

| Preconditioner | 8x16x8 its | 12x24x12 its | 12x24x12 wall | vs best |
| --- | --- | --- | --- | --- |
| **k = 0 mass** | | | | |
| jacobi | 538 | 785 | 1744 ms | 76x |
| tensor (current production) | 11 | **10** | 22.8 ms | 1.00x |
| block-diagonal, exact A_ss | 18 | 16 | 35.5 ms | 1.55x |
| sandwich pow0 | 23 | 26 | 58.6 ms | 2.57x |
| sandwich pow1 | 20 | 19 | 43.0 ms | 1.88x |
| **sandwich pow2** | **15** | **13** | **28.9 ms** | **1.27x** |
| **k = 1 mass** | | | | |
| jacobi | 538 | 608 | 9464 ms | 49x |
| tensor (current production) | 13 | **12** | 192 ms | 1.00x |
| block-diagonal, exact A_ss | 21 | 20 | 305 ms | 1.59x |
| sandwich pow0 | 37 | 39 | 598 ms | 3.11x |
| **sandwich pow2** | **19** | **17** | **266 ms** | **1.38x** |

pow2 is mesh-independent at both degrees (15->13, 19->17) and beats the
block-diagonal variant at k=1. pow0 drifts upward -- the signature of a growing
mis-scaled subspace.

### Geometry stress

Elongated axisymmetric torus (Cerfon `one_size_fits_all_map`, kappa=2.0) at
8x16x8 -- the case that previously moved k=1 from 14 to 28 iterations. The
scalar k=0 block is insensitive to cross-section shaping (pow2 = 15 on both
geometries), so only k=1 is informative.

| k = 1 | toroid | elongated kappa=2 | degradation |
| --- | --- | --- | --- |
| jacobi | 538 | 636 | 1.18x |
| tensor (current production) | 13 | 28 | 2.15x |
| block-diagonal, exact A_ss | 21 | 30 | 1.43x |
| sandwich pow2 | 19 | 33 | 1.74x |

Everything degrades together and pow2 degrades *less* than production. The
per-component `D` ranges blow out on this geometry (139 / 397 / 58 against
39 / 123 / 58) and the Kronecker factors still absorb them.

### Iteration count is the only currency

The matrix-free mass apply dominates an iteration so heavily that preconditioner
apply cost is not a design variable. At 12x24x12, k=1: matvec 11961 us,
preconditioner applies 10-295 us, giving us/iteration flat to +-4% across every
variant tested.

This is structural. The sum-factorized matvec costs **O(N p^4)** -- quadrature
points scale with degree, so per-element work is `q^3 (p+1)` with `q = 2p`
(`DeRhamSequence.__init__` takes `q` = points per direction; callers pass
`2*P`). The Kronecker solve costs **O(N (n_r+n_t+n_z))** with no p dependence.
Measured at fixed 8x16x8:

| degree | A matvec | tensor apply | sandwich apply | ratio |
| --- | --- | --- | --- | --- |
| p = 2 | 317 us | 37.7 us | 44.0 us | 8x |
| p = 3 | 558 us | 46.2 us | 42.4 us | 12x |
| p = 4 | 1328 us | 45.2 us | 47.6 us | 29x |

The p=3->4 matvec jump is 2.38x against a predicted 2.37x. Preconditioner apply
is flat in p, so at higher order the apply-cost argument weakens further.

## 3. The structure that makes it work

Four facts, each verified numerically rather than assumed.

| Fact | Evidence |
| --- | --- |
| `E E^T = diag(CC^T, I)` -- bulk block exactly identity, cross block exactly zero, selection values exactly 1.0. Holds for k=0,1,2 x both BCs at every resolution. | `max\|cross\| = 0.0` |
| Coupled rows of `E` are *identically* the hand-derived surgery rows, so the ~100 lines of per-degree layout arithmetic re-derive something readable from the sparsity pattern. | Jaccard 1.000 |
| `CC^T` is block-diagonal with blocks of size <= 3, found basis-free via connected components. Coupled counts are exactly 3n_z / 5n_z / 2n_z / 0 for k=0/1/2/3. | 52*n_z doubles total |
| The exact mass diagonal is computable with zero probes -- one sum-factorized contraction against squared basis tables -- and agrees with the probed diagonal to machine precision. This is both the `D` for pow2 and the fix for the Jacobi setup cost. | 4-6 x 1e-16 |

### Storage, before and after

| resolution | n_z | `(CC^T)^-1` blocked | dense fallback | `coupling_sb` replaced |
| --- | --- | --- | --- | --- |
| 12x24x12 | 12 | 4.9 KB | 88 KB | ~23 MB |
| 32x64x32 | 32 | 13 KB | 623 KB | ~1.5 GB |
| 64x128x64 | 64 | **27 KB** | 2.5 MB | **~24 GB** |

Totals across k=0,1,2 and both BC variants. `coupling_sb` is O(N n_z) --
asymptotically O(n^4), larger than the solution vector by a factor of n -- in a
code whose stated premise is matrix-free (`mass_core_apply`: "removing the
high-(n,p) storage bottleneck (notably for M1)"). `(CC^T)^-1` is O(n_z) and
depends only on sparsity, so it never rebuilds when the geometry changes.

## 4. Plan

Ordered so each phase has standalone value and nothing is deleted before its
replacement is measured. Phase 1 is worth doing regardless of this decision.

### Phase 1 -- exact mass diagonal (independent value)

One routine beside `build_matrixfree_mass_apply` sharing its `_bases_for_form` /
`_split_field` tables, contracting squared basis tables to produce `diag(M_k)`
in raw space. Two consumers: `D` for pow2, and `build_mass_jacobi_pair`, which
currently costs O(n) full applies.

- Replaces the O(n)-probe Jacobi assembly -- 541 s of the 2957 s total setup at
  12x24x12.
- Consistent by construction: derived from the same tables the solver applies,
  which is exactly what the deleted `diag_EAET_direct` route failed to guarantee.
- Ship with a small-resolution test asserting `direct == probed`, so the
  2026-08-14 stall cannot recur silently.

*Done when:* Jacobi diagonals match probed values to 1e-14 for k=0..3 x both
BCs, and assembly is O(1) applies.

### Phase 2 -- build pow2 for k=0..3

Three ingredients, none requiring an operator probe.

- Per-component unweighted 1D masses and inverses -- the axis convention already
  exists in `_build_greville_mass_block_factors` (`deriv` on axis c, `primal`
  elsewhere).
- `D_c` from the Phase-1 diagonal divided by the Kronecker diagonal.
- `(CC^T)^-1` from `E.indices`: find coupled rows, form the small Gram, split
  into blocks by connected components, invert blocks of size <= 3.

*Done when:* k=0 and k=1 reproduce 13 and 17 iterations at 12x24x12, and k=3
(no coupled rows) is a plain tensor block.

### Phase 3 -- validation gates (BLOCKING)

The decision rests on k=0 and k=1 on two axisymmetric geometries.

- **k=2 and k=3.** k=2 has 2n_z coupled rows and `g_ij/J` weights; structurally
  like k=1, but unmeasured. k=3 has zero coupled rows and should be trivial.
- **A stellarator geometry.** Both geometries tested so far are axisymmetric.
  The rotating-ellipse case needs adequate zeta resolution -- at n_z=8 with
  nfp=3 the discretization itself fails and the test measures nothing.
- **GPU.** All timings are CPU. The matvec is one large kernel and the
  preconditioner several small ones, so the 1-2% apply share is the number most
  likely to move.

*Proceed if:* k=2 lands within ~1.5x of production and the stellarator case
degrades no worse than production does.

### Phase 4 -- repoint dependents (watch this one)

`assemble_schur_jacobi_preconditioner` with `schur_diag_mode='tensor_probe'`
uses the tensor mass preconditioner as `M^-1` in the weak term and raises if it
is not assembled for k-1. Removing the tensor mass path breaks the k>=1 saddle
preconditioner unless that reference is moved to the pow2 apply first.

Independently of this pivot, that same probe is the single largest setup cost in
the code -- 2236 s of 2957 s at 12x24x12, scaling as O(N^2) -- and it runs
through an unbatched `jax.lax.map` (`operators.py:_diagonal_from_matvec`) while
`diag_matvec` batches 16. Batching it is a small diff with no numerical change.

*Done when:* the saddle path builds against pow2 and k>=1 saddle iteration
counts are unchanged.

### Phase 5 -- remove the surgery machinery

801 lines across 47 definitions: `build_mass_surgery_preconditioner`,
`_apply_surgery_schur`, `_assemble_surgery_schur_inverse_from_applies`, the
K0/K1/K2 surgery factor classes, every `_apply_k1_*` / `_apply_k2_*` coupling
function, `_surgery_slices_k1/k2`, `_k1_layout_sizes`, `_component_sizes_k2`,
`_tensor_block_indices_k1/k2`, `coupling_sb`, and the `precompute_coupling`
toggle. The CP/NTF fits and surgery probes go with them -- most of the 127 s of
tensor-mass setup at 12x24x12.

*Done when:* the mass test suite passes against pow2 and no module imports the
removed symbols. Update `docs/source/concepts/PRODUCTION.md`.

## 5. Explicitly out of scope

- **k=0 Laplacian core Schur**
  (`operators.py:_assemble_k0_tensor_hodge_preconditioner`). Different
  machinery, different justification -- an exact Schur from assembly-time CG
  solves, documented as worth 53/80 against 56/87 iterations on W7-X. It
  **stays**: five separate attempts to remove it failed (section 7). Its *atom*
  is replaced (section 7.1), which is a drop-in change to the same structure.
- **The saddle-point Schur-outer Jacobi** for k>=1 solves. Only its `M^-1`
  reference is repointed (Phase 4); the method stays.
- **Mass Jacobi** stays as the cheap fallback and reference, and gets much
  cheaper to assemble via Phase 1.

## 6. Risks

- **Coverage.** k=2/k=3 and stellarator geometry are unmeasured. Phase 3 exists
  to close this; Phase 5 must not start before it does.
- **The 1.4x is real and permanent.** It is an iteration-count gap, not an
  apply-cost gap, so it will not be recovered by optimization. If mass solves
  later become the production bottleneck, revisit the decision rather than tune it.
- **pow0 is a trap.** It looks acceptable at k=0 (23 iterations) and is badly
  wrong at k=1 (37, drifting). An implementation that applies `E^T` where `E+`
  belongs will appear to work on the easiest test case.
- **Off-diagonal metric blocks are dropped** in the k=1/k=2 component model.
  Production already does this (`inner_schur=False` selects
  `_apply_k1_bulk_diagonal_preconditioner`), so it is not a new approximation --
  but it is now the only one absorbing anisotropy.
- **CPU-only timings.** Ratios could shift on GPU; direction unknown, Phase 3
  measures it.

## 7. The k=0 Laplacian: separate investigation, separate outcome

The mass pivot does **not** carry over. The stiffness needs a *sum* of three
Kronecker terms with three different metric weights (`g^rr J`, `g^tt J`,
`g^zz J`), and fast diagonalization requires the three terms to share their mass
factors. The current atom achieves that by keeping the 1D masses unweighted and
pushing each metric into its own 1D stiffness factor, with per-axis averaged
profiles (`_k0_bundled_axis_profiles`). Those averages deliberately **skip the
polar element** (`wx_cut`), and the docstring says why: "core DOFs are handled
exactly by the Schur envelope."

Conclusion after five experiments: **the core Schur stays; replace the atom.**

### 7.1 ADOPT -- modal-radial atom (the one clear win)

Average the weights over theta and zeta **only**, keeping the full radial
dependence, then diagonalize only those two pencils:

```
K ~ K_r[a] (x) M_t (x) M_z + M_r[b] (x) K_t (x) M_z + M_r[c] (x) M_t (x) K_z
    a(r)=<g^rr J>_tz         b(r)=<g^tt J>_tz         c(r)=<g^zz J>_tz

(M_t,K_t) -> mu_j ,  (M_z,K_z) -> nu_k
  =>  block diagonal over (j,k) with  A_jk = K_r[a] + mu_j M_r[b] + nu_k M_r[c]
      (n_r x n_r, banded, solved EXACTLY -- no radial averaging anywhere)

apply:  V_t^T, V_z^T  ->  batched radial solve  ->  V_t, V_z
```

Why the `1/r` stops mattering: `mu_0 = 0`, so the constant theta mode never sees
the singular weight at all, and higher modes see it multiplied by basis
functions that vanish like `r^m` at the axis. Averaging over `r` destroys that
pairing; modal decomposition preserves it exactly.

Full extracted `K_0`, dbc, **core Schur kept, only the atom swapped**:

| geometry | production atom | modal-radial atom |
| --- | --- | --- |
| toroid 8x16x8 | 22 | **13** |
| toroid 12x24x12 | **32** | **14** |
| elongated kappa=2, 8x16x8 | 28 | **24** |

Production degrades under refinement (22->32) while modal-radial is flat
(13->14): the radial averaging was the *mesh-dependent* error, so removing it
buys mesh-independence, not just a constant. Bulk-block-only numbers agree
(production 25 free / 22 dbc -> modal-radial 13 / 13), confirming the gain is in
the atom rather than an interaction with the Schur.

Setup is ~0.01-0.06 s warm (n_t*n_z tiny dense eigendecompositions); block
conditioning on the bulk is 1.4e1 - 6.9e2.

Two properties that make this a genuine drop-in:

- The stored `schur_inv` is built from **exact** bulk CG solves (converged to
  1e-12), so it is atom-independent -- the atom only preconditions those solves
  and never enters the result. It can be reused verbatim with a different bulk
  inverse at runtime.
- Those same assembly solves are atom-preconditioned, so the better atom also
  makes the core-Schur assembly (`6*n_z` CG solves) roughly 2x cheaper. Better
  runtime and cheaper assembly come from one change.

### 7.2 ADOPT -- harmonic profile averaging

`_k0_bundled_axis_profiles` uses a quadrature-weighted *arithmetic* mean. For
`g^tt J ~ 1/r` that mean is dominated by the small-r end and diverges, which is
exactly why `wx_cut` exists. The harmonic mean is finite and behaves.

Bulk block, free / dbc:

| rule | with `wx_cut` | without cut |
| --- | --- | --- |
| arithmetic (production) | 25 / 22 | **33 / 27** |
| harmonic | 24 / 23 | **24 / 21** |
| geometric | 24 / 23 | 25 / 21 |

On iterations it is a wash; the finding is the second column. **Arithmetic needs
the cut, harmonic does not**, which retires `wx_cut` and with it the documented
dependence on "core DOFs are handled exactly by the Schur envelope". Harmonic
also wins outright on the full grid (73 vs 82). One-line change, never worse.

### 7.3 REFUTED -- do not re-try these

1. **Exact modal denominator.** Keep the FD transform `V`, replace the additive
   `lam_r+lam_t+lam_z` with the exact modal diagonal
   `d_ijk = <psi_ijk, K psi_ijk>` (computable probe-free by the same
   sum-factorized contraction as the mass diagonal, all 9 metric blocks
   including the off-diagonals production drops). Bulk: 25->24 (~4%). Full grid:
   82->**107**, clearly worse. Reason: it converts the preconditioner from *the
   exact inverse of a nearby SPD operator* into *a diagonal truncation of the
   true operator in a basis that does not diagonalize it*. The off-diagonal
   energy of `V^T K V` exceeds the averaging error it removes.

2. **Per-basis-function collapse of the three metrics into one diagonal**
   (`D_a` from arithmetic / harmonic / geometric means of the directional
   ratios `k_s(a) = int w_ss (d_s phi_a)^2 / int_ref`). On an *unweighted*
   reference the harmonic rule helps a lot (140->108). On top of good profiles
   nothing helps: best is 70 vs 73 for no D, and the arithmetic rule -- which is
   `K_aa/(K_ref)_aa`, the direct analogue of the mass `D` -- is the **worst** at
   126. Reason: profiles change the 1D *eigenvalues*, representing a spatially
   varying stiffness spectrum; a per-DOF diagonal only *rescales* and cannot
   represent a varying spectrum. They are not interchangeable instruments, and
   profiles are strictly the better way to spend the same metric information.
   Secondary but consistent signal: rules dominated by the **softest** direction
   (harmonic) beat rules dominated by the **stiffest** (arithmetic), because an
   inverse is governed by the soft direction.

3. **Schur-free sandwich, every variant tried.** Best result 66 (modal-radial +
   pow2) against 13 with the Schur. The extraction scaling was verified optimal
   -- pow0 106, pow1 73, **pow2 66**, the same monotone ordering as the mass --
   so the lift is not the cause.

   A note on a wrong diagnosis, recorded so it is not repeated: the `2.7e15`
   block conditioning on the full raw grid is **not** a `1/r` pathology. For
   `j=k=0`, `mu_0 = nu_0 = 0` so `A_00 = K_r[a]`, the bare radial stiffness on
   the raw basis with no BC (E imposes the BC downstream), which is singular by
   construction. Regularizing it does not rescue the sandwich -- a Tikhonov
   sweep gives sigma=0 -> 66, 1e-10 -> **90** (worse than zeroing), 1e-8 -> 77,
   1e-6 -> 68, 1e-4 -> 60, 1e-2 -> 42. Only a large shift helps, and that is
   preconditioning `K + sigma M` rather than `K`.

### 7.4 Why the sandwich works for the mass but not the stiffness

The k=1 mass has the *same* `1/r` -- component theta carries `g^tt J ~ R/r`, and
it shows in the measured per-component `D` ranges (39, **123**, 58). It works
there because for the mass the `1/r` is a **pure diagonal magnitude**:
`D^-1/2 M D^-1/2` has unit diagonal by construction, so the divergence is
*removed*, not approximated (radial cond 950 -> 21, mesh-independent), and the
Kronecker factors only ever see the benign shape.

For the stiffness the same `1/r` sits in one of three *competing* terms, so no
diagonal can rebalance it. And structurally: the mass sandwich approximates a
well-conditioned operator, where a bounded multiplicative error near the axis
costs a constant; the stiffness has conditioning O(h^-2), CG convergence is
governed by its **low** modes, and the low modes at the axis are precisely what
`E` supplies (`r^m` regularity plus the boundary condition) and a raw-grid model
lacks. The core Schur reconstructs exactly that, from exact bulk solves.

### 7.5 Open -- rotating ellipse (handoff item)

Not yet measured. Both geometries tested are axisymmetric, so the one
approximation modal-radial still makes -- averaging over theta and zeta -- is
untested where it matters. A stellarator metric varies strongly in zeta.

The analytic map is fine (`det J` = +0.26 .. +1.61 at kappa=1.5, checked
directly). The failure at (8,16,24) was in the **spline projection**: projected
profiles came out negative (`a[-25.6, -0.159]`), i.e. the projected geometry
folds, and production returns `nan` on it too.

**Gate any rotating-ellipse run on the sign of the projected Jacobian before
solving.** Needs higher poloidal resolution (n_t=16 is thin for kappa=1.5 with
nfp=3) and more map samples than the 40^3 grid used here.

### 7.6 Harness caveats for whoever re-runs this

- The free/Neumann full `K_0` is **singular** (constants). A plain PCG diverges
  on it -- production uses `solve_singular_cg` with harmonic deflation. All
  full-grid numbers above are dbc only. Bulk-block-only numbers are valid for
  both BCs (the bulk block is nonsingular).
- `apply_stiffness` builds host-side numpy state lazily on first call; it must
  be warmed eagerly before entering a `lax.while_loop`, or it raises
  `TracerArrayConversionError`. `_diagonal_from_matvec` does the same warmup.
- `core_coupling` in the k=0 Laplacian factors is a dense `bulk x 3*n_z` block --
  the identical O(N*n_z) storage problem as `coupling_sb` on the mass side.
  Banding it is independent of everything above and carries no numerical risk.

### 7.7 Reproduction

`modal_radial_test.py` (modal-radial, 3 parts: bulk / Schur-free sandwich /
Schur-kept atom swap), `laplace_atom_test.py` (averaging rules + exact modal
denominator, bulk), `laplace_full_test.py` (full grid, sandwich),
`laplace_D_test.py` (per-DOF metric collapse), `pow_sweep_laplace.py` (pow0/1/2
+ Tikhonov sweep).


## Addendum (2026-08-26): the modal-radial k=0 bulk atom

Measurement kept from the docstring of `mrx/experimental/modal_radial.py` (greville-prod 53a71ed), the per-mode radial pencil atom; CG iterations to 1e-10 on the bulk block, p=3, `fd -> modal` per (dbc/free):

```
Measured 2026-08-17/18, CG to 1e-10, bulk block, p=3, fd -> per-k:

    toroid        8x16x16  24/22 -> 13/13     12x24x24  36/32 -> 14/13
    rot-ellipse   8x16x16  59/48 -> 45/36     12x24x24  83/71 -> 49/42
    W7-X          8x16x16  61/45 -> 47/34     12x24x24  83/66 -> 50/40

```
Not in production; the candidate replacement was gated on the full-grid comparison, which was never run.
