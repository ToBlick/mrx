> **Status:** superseded by preconditioner_technical_note_source.md
> **Read this for:** the day-by-day record of the natural-BC coefficient, every dead end and why it died (read section 0, then 15-19)
> **Do not read for:** the current value of the scale; s_scale_2026-08-25.md settles it at 3.0

# Natural-BC coefficient and the k=1/2 free gap — handoff

Started 2026-08-21 (§1-§12), continued 2026-08-22 (§13-§18).

## READ THIS FIRST — current state, 2026-08-22

**THE DELIVERABLE: land the derived boundary term at a CORRECTED SCALE.**
`BlockJacobiLaplacian(bc_entry="ibpd", bc_scale=0.10)`. One float, a rank-one
update that already exists, FD and the shared eigenbasis untouched, vanishes
identically under Dirichlet. It buys ~2x over jacobi and 2-3x over the shipped
default at k=1/2 free, and captures 66-74% of the entire gain available between
jacobi and the best known arm at ANY cost. See **§15** (cost-adjusted) and
**§16** (the hyperparameter).

**The shipped default is actively harmful.** `bc_entry="exact"` at scale 1.0 is
WORSE THAN NO BOUNDARY TERM AT ALL at k=1/2 free on rot-ellipse (1056 vs 636),
and `ibpd` at scale 1.0 loses to jacobi on W7-X for n >= 12. §12.11.

**Where the 0.10 comes from: a kappa-BALANCE point, measured, not derived.**
`alpha` is right for `L` -- it is the best NORM approximation to `L`'s boundary
block (§17.5) -- but `P` minimises `kappa(P^-1 L)`, a different question with a
different optimum. §14.1 has the mechanism (`min eig(P) = 1/(1+r s)` to 3%);
§17.5 has the origin (the WITHIN-RING coupling the separable atom drops -- NOT
the radial DtN, which §17.5 refutes) and shows the scale is COMPUTABLE from a
local ring-block match with no solve.

**Retired, do not re-propose** (§9, §12.3, §14.3, §15):
`wibp` (exact 2-D face shape), `ibpf`/`ibpr` (the cross term -- `ibpr` is
INDEFINITE), the `pin`, Nitsche, `tm`/`mode_beta_correction`, and -- on
TOTAL TIME -- `o1`/`o2`/`o3` and the banded-capacitance route built on them.
`fm3` survives as an OPTION (a further 1.18-1.32x) but not as a default.

**Noise floor ~1%, up to 2.4% at k=1 free** (§13.1). Ignore smaller differences
anywhere in this file.

**NOTHING IS COMMITTED.** `bc_entry="exact"` is still the default and is
byte-identical to before.

### Map

| § | what |
| --- | --- |
| 1-8 | the derivation of `alpha`, and the state of play on 2026-08-21 morning |
| 9 | dead ends, do not re-run |
| 10-11 | the pin, and the 2026-08-21 overnight launch |
| 12 | the 33 overnight jobs read out (§12.9's "wrong object" verdict is WITHDRAWN by §13.4) |
| 13 | `diag_bcpn` -- the scale, not the h-scaling, was the problem |
| 14 | what 0.15 is; `fm` revives; **§14.8: 0.15 was an n<=20 artefact, use 0.10** |
| **15** | **COST-ADJUSTED -- rank by TOTAL TIME. `o1`/`o2` are dead.** |
| **16** | **THE HYPERPARAMETER -- what to ship and how reliable it is** |
| **17** | **where the factor comes from; §17.5 = the DtN test (refuted) and the answer** |
| 18 | open items 
---

## 1. Scope: the coefficient changes exactly three cases

The boundary term only ever executes at **k=1, k=2, k=3 FREE**.

* **All Dirichlet**: `_boundary_entry_direct` returns `None`; the partner DOF is
  removed so `E^T M^-1 E` genuinely vanishes. Every dbc row was identical across
  every arm all day.
* **k=0 free**: `W_0 = 0` and `deriv_axes` is empty, so the block never runs —
  and it needs nothing: the atom's `w d_r u = 0` IS the operator's natural
  condition, same weight, exactly. Best-conditioned case anywhere (43 vs jacobi
  398, 9.2x).

So the measurable benefit is **k=3 free**; the rest of the value is documentary.

---

## 2. What the coefficient is

Take each matrix's weight at `r=1`, average over `theta,zeta`, multiply:
`E -> m_k`, `M_{k-1}^{-1} -> 1/m_{k-1}`, `E^T -> m_k`. Under `lumped="diag"` the
component factor `w_comp = m_k/J` is carried outside as the `D` sandwich and must
come out of the face weight too:

```
alpha = mu_0 . <S> . <P>       S = J sqrt(g^rr)   surface element
                               P = sqrt(g^rr)     normal pullback
mu_0  = (M_r^{logical})^-1[last,last]      metric-FREE, ~ c(p)/h_last
```

`m_k J / m_{k-1} = J g^rr = S.P` for EVERY k (use `J^2 = prod g_aa`), so the
per-degree spread was an artefact. Verified: `mu_0` bit-identical across all
three geometries (`5.226074e+01` at n=8, `9.388387e+01` at n=12); `a_ibpd`
identical down all four `(k,c)` rows on every geometry.

**Averaging convention**: `<S><P>` (product of averages), not `<S.P>`. Identical
to 1.0000 on the toroid, where `g^rr` is constant on the face so the covariance
vanishes -- a free correctness check. 0.945 rot-ellipse, 0.912 W7-X.
Deliberately the OPPOSITE convention to `bundled_axis_profiles`, whose bundling
exists to keep `g^tt J ~ 1/r` integrable toward the AXIS; there is no such
singularity on the r=1 face.

### CONFIRMED as an optimum, not a fit

Joint `(alpha, beta)` sweep, k=3 free, n=12. Scale 1.00 IS the derived alpha:

| alpha scale | 0.30 | **1.00 (derived)** | 3.00 |
| --- | --- | --- | --- |
| toroid | 49 | **34** | 44 |
| rot-ellipse | 80 | **75** | 111 |
| w7x | 82 | **75** | 108 |

Clean interior minimum at the derived value on all three geometries. At k=3 --
the one degree where the atom is otherwise faithful (single component, nothing
dropped) -- the derived coefficient IS the CG optimum. No tuning.

---

## 3. The two errors in `exact`, both from implicit metric factors

1. `_weak_inverse_amplification` builds `M_r` **with** the partner's mass weight
   and inverts it, so `1/m_{k-1}` never appears as a factor. The face weight was
   then "corrected" `g^rr -> sqrt(g^rr)` to compensate. Wrong: `u.n` and `ds`
   each carry `sqrt(g^rr)` and they MULTIPLY back to `g^rr J`.
2. `lumped="diag"` factors `w(c,a) = w_comp . (g^aa J)` and builds the Kronecker
   factors from the `g^aa J` half alone, returning `w_comp` as the
   `D^-1/2 . D^-1/2` sandwich (`component_diagonal`). The face weight used the
   full `m_k`, so `w_comp` was counted **twice** -- 9.0 toroid, 15.0
   rot-ellipse, 6.7 W7-X.

Net: `exact` is `w_comp . (1/rho)` off -- 13.9x toroid, 12.1x W7-X. Against the
exact 1-D round trip, `exact` is 8-14x too small.

**The dangerous part is not the wrong number, it is the fitted compensation.**
Both hidden factors were cancelled by something tuned empirically, so the wrong
version passed its own sweep (§6.3 claimed "the residual sweep peaks at x1").
See memory `metric-factors-must-be-explicit`.

---

## 4. The cross term (real, confirmed at k=3)

`W_k` has three pieces: `d_s^T M d_s - 2 d_s^T E u + u^T E^T M^-1 E u`. The atom
carries the outer two. The middle is interference between the boundary spike
(`V_{k-1}` cannot hold a delta, so it holds a spike of height `~u_n/h`) and the
interior divergence. Same order in `h`, negative sign. Structurally
`-(e v^T + v e^T)`; splitting `v = gamma e + v_perp` shows the dominant effect is
`alpha -> alpha - 2 gamma`, a shrunk coefficient on the same `e e^T`.

Measured: `s2/s1 = 0.02-0.06` (rank one), `off_frac = 0.21-0.28`,
`rho = c_star/alpha` = 0.647/0.629 (toroid n=8/12, h-INDEPENDENT), 0.396 (W7-X).
Scale sweep: **k=3 confirms it** (interior minimum at 0.55-0.70 on both shaped
geometries, exactly where `rho` puts it; 1.00 on the toroid). k=1/2: no minimum.

If landing the coefficient, `rho` should be COMPUTED from the round-trip ratio
(dimensionless, numerator and denominator share the same profiles so the
bundling cancels), never fitted.

---

## 5. THE DIAGNOSIS (Tobias): a scalar Laplacian per component cannot carry a
## vector BC

The atom solves a scalar Laplacian per component. The natural BCs of a VECTOR
operator couple components at the boundary. Everything else here follows.

### The BC table

At `r=1`. "Atom imposes" = the natural condition of the assembled radial factor
(a stiffness on the unconstrained space asserts homogeneous Neumann, silently),
plus the `alpha e e^T` update where one exists (`has_trace = 0 in deriv_axes`).

| k | BC | operator wants | atom imposes | status |
| --- | --- | --- | --- | --- |
| 0 | free | `w d_r u = 0`, `w = J g^rr` | `w d_r u = 0`, same `w` | exact |
| 0 | dbc | `u = 0` | DOF removed | exact |
| 1 | free | penalised `u_r=0`; `d_r u_t = d_t u_r`; `d_r u_z = d_z u_r` | penalised `u_r≈0` (Robin, correct); **`d_r u_t = 0`**; **`d_r u_z = 0`** | normal ok, TANGENTIAL WRONG |
| 1 | dbc | `u_t=u_z=0`; `div u = 0` => `d_r(J g^rr u_r) = 0` | DOFs removed; `d_r u_r = 0` | weight OUTSIDE the derivative |
| 2 | free | penalised `w_t=w_z=0`; `div w = 0` | penalised `w_t,w_z≈0`; **`d_r w_r = 0`** | tangential ok, NORMAL WRONG |
| 2 | dbc | `w_r=0`; `n x curl w = 0` => `d_r w_t = 0` | DOF removed; `w d_r w_t = 0` | exact |
| 3 | free | penalised `omega = 0` | penalised `omega ≈ 0` | exact (one component) |
| 3 | dbc | as k=1 dbc | `w d_r omega = 0` | weight placement |

**Over-determination** (Tobias): asserting `d_r u_t = 0` while the operator asks
only that `d_r u_t = d_t u_r` forces `d_t u_r = 0` too, which the operator never
requires. The atom imposes two conditions where the operator has one.

### Why Dirichlet is immune: essential conditions are enforced by the SPACE

At k=1 dbc both conditions are structural -- `u_t=u_z=0` because those DOFs are
absent, `div u = 0` because `V_0` loses its outer boundary DOF. **A wrong
boundary term cannot violate a condition the space already satisfies.** So the
weight-placement mismatch only mis-weights an already-correctly-constrained
problem. Under a free BC the conditions are enforced by TERMS IN THE OPERATOR
and nothing else holds them, so a wrong term gives a genuinely different
near-null space.

Quantified by the same probe: `||F - Ktilde||/||Ktilde||` is **0.29 dbc vs 4.63
free** (toroid; 0.50 vs 2.85 W7-X). Dirichlet omits a 29% perturbation; free
omits a term 4.6x LARGER than the thing it corrects. §6.5's spectrum reference
says the same: k=1 free has 44 outliers of 894 with the extreme mode entirely on
the outer boundary; k=1 dbc has ZERO outliers and condition 12.7.

**Rule: spend boundary effort only where the condition is NATURAL.**

### What the iteration numbers split on

Not the BC table's dbc entries -- the ordering is backwards there (k=3, flagged
mismatched, is the strongest dbc case; k=2, flagged exact, the weakest). The
split is **component count**. Dirichlet speedup jacobi/ibpd, n=12:

| | k=0 | k=1 | k=2 | k=3 |
| --- | --- | --- | --- | --- |
| toroid | 7.31 | 6.06 | 6.10 | 7.44 |
| rot-ellipse | 4.58 | 4.52 | 3.85 | 5.14 |
| w7x | 5.49 | 4.88 | 4.73 | 5.93 |

Single-component (k=0, k=3) beats three-component (k=1, k=2) in all three
geometries, even under an essential BC where most coupling is killed.

---

## 6. Results

Free BC, p=3, `extra_rings=3`, `lumped="diag"`, tol 1e-10, n=12.
**Every Dirichlet row identical across every arm, all day** (to +-1 iteration;
GPU reductions are not bit-reproducible run to run).

### k=3 free — the deliverable

| | jacobi | `exact` (today) | derived `ibpd` | `wibp` (dense face) |
| --- | --- | --- | --- | --- |
| toroid | 188 | 36 | 34 | **33** |
| rot-ellipse | 234 | 88 | 75 | **58** |
| w7x | 306 | 123 | **75** | 74 |

### k=1/2 free — the derived coefficient is HARMFUL

| k=1 free | jacobi | nobc | direct | exact | ibpd | best `tg` | joint best |
| --- | --- | --- | --- | --- | --- | --- | --- |
| toroid | 440 | 296 | 117 | 77 | 76 | 76 (tg0) | 76 |
| rot-ellipse | 738 | 636 | 420 | 1063 | 618 | 368 (tg10) | **317** |
| w7x | 1657 | 1321 | 1054 | 1194 | 1802 | 808 (tg30) | **710** |

`ibpd` (correct) is worse than `nobc` (nothing) on W7-X. k=2 mirrors this;
joint best W7-X k=2 free **726** vs jacobi 1555.

### The `tg` lever

`MRX_BJ_TANG_BC` adds `beta e_last e_last^T` to the radial factor of the
components that get NO term. **The optimum is one number per GEOMETRY, shared
across k=1 and k=2** (0 / 10 / 30 for toroid / rot-ell / W7-X) and ZERO on the
orthogonal toroid. That is a property of the boundary geometry, not a per-case
fit. Joint `(alpha, beta)`: the two are INDEPENDENT (`beta_opt` flat or rising
in `alpha`) -- expected, since `alpha` is a preconditioner parameter and cannot
move the operator's boundary residual.

In Nitsche language `tg` is the **stabilisation term with no consistency term**
-- a penalty method. That is why it has an interior optimum and why the optimum
is geometry-dependent.

### Outer rings — corrected understanding

W7-X k=1 free: p=3 `o0/o1/o2/o3` = 1796/561/391/349; p=5 = 2892/1173/781/690.
**Relative gain per ring is identical at p=3 and p=5** (`o1/o2` 1.44 vs 1.50,
`o2/o3` 1.12 vs 1.13). So:

* `outer_rings=2` is a cost/benefit knee, NOT a derived number, and needs **no
  `p` qualifier**. (The Bezier endpoint property makes the boundary flux a
  two-coefficient object at every degree -- but the rings are buying general
  exactness near the boundary, not representing the BC, else `o2` would
  saturate.)
* **k=0: outer rings HURT** -- 65->182/114/89 (rot-ell), 83->288/181/145 (W7-X),
  never recovering to `o0`. Rings are exact within the ring but DECOUPLED from
  the bulk; at k=0 the atom's BC is already exact so decoupling is pure loss.
  §7's "k>=1 only" is now measured, not asserted.
* **k=3 saturates at `o2`** (45 then 46) -- the only degree that does, and the
  only one whose free condition is a pure value condition.
* At p=5, jacobi **diverges** (5000 it, 1.3e-4) where the atom converges. The
  atom is not optional at high degree.

---

## 7. Code state (all uncommitted)

`mrx/experimental/block_jacobi_laplacian.py`
- `_mesh_amplification(seq)` -- metric-free `1/h`. No k, c, or geometry.
- `_face_metric_scalar(seq, k, c, lumped, separate)` -- all the metric, one
  number. `separate=True` gives `<S><P>`.
- `_edge_vector(seq, axis, window)` -- `e = dLam(1)`, shared by all boundary
  updates.
- `bc_entry`: `"ibp"` (non-lumped `m_k g^rr`), **`"ibpd"` (diag-lumped `J g^rr`,
  the correct one)**, `"ibps"` (ibpd with product-of-averages), `"wibp"` /
  `"wibpd"` (corrected dense face via Woodbury).
- `nitsche_consistency(...)` -- cross-component boundary coupling, k=1 (see §8).
- `mode_beta_correction(...)` -- **BROKEN, see §9.**
- `face_operator(..., corrected=True)` -- weight `J g^rr`, full angular profile,
  x `mu_0`.
- Env knobs (diagnostic only): `MRX_BJ_TANG_BC`, `MRX_BJ_TANG_MODE`,
  `MRX_BJ_DBC_BC`, `MRX_BJ_NITSCHE`.
- `"exact"`, `"direct"`, `"face"`, `"woodbury"`, `"wdiag"` all unchanged.

`scripts/debug/verify_block_jacobi.py`
- `rot-ellipse` geometry (`eps=0.33, kappa=1.5, nfp=3`).
- `--ks` / `--bcs` to restrict which (k, dbc) rows are solved (big cost saver).
- arm keys: `ibps|ibpd|ibp|wibpd|wibp`, `bcpN` (alpha scale N/100), `tgN`,
  `tmN`, `dbN`, `ntN`, `oN`, `rN`.

New probes (all quadrature-only, no operator applications, seconds to run):
`bc_alpha_compare.py` (all alphas, `w_comp`, round-trip reference),
`component_coupling_probe.py` (metric correlation + tilt),
`face_weight_spectrum.py` (2-D FFT of the face weight -> `B_hat` bandwidth).

---

## 8. THE PLAN (morning) — SUPERSEDED by §11, except the capacitance route

Read for the storage analysis and the measured `B_hat` bandwidths, which stand
and are route (2) of §11. Its two action items are dead: landing the
coefficient is on hold (§11 shows `alpha` is still geometry-dependent once a
penalty is present), and the Nitsche flux stabilisation is not worth building
(§11: the failing modes violate no condition, so consistency+stabilisation has
nothing to restore for them).


**Land the coefficient, with an honest scope.** It changes three cases, helps in
one (k=3 free, W7-X 123->75). The larger value is removing a fitted compensation
for two hidden errors so the next person doesn't inherit confidently-wrong
reasoning. Add `rho` computed (not fitted) and `ibps`.

**Do NOT raise alpha to the derived value at k=1/2 free.** It is measurably
worse than what ships. Reason documented (§5).

**The real fix is genuinely low-rank, and untested.** The couplings live only on
the `r=1` face. The atom's angular masses are unweighted, so `M_t`, `K_t` AND
`D_t` are circulant and simultaneously Fourier-diagonalised by the basis FD
already computes -- **`d_t` is diagonal there**, so the coupling does not mix
angular modes:

    per angular mode (j,k):  3x3 coupling among the components' boundary rows
    total:                   rank 3 n_t n_z, block-diagonal, 3x3 blocks
    Woodbury capacitance:    3x3 per mode, closed form

`O(n_t n_z)` with tiny blocks, no dense face, no probes.

**But consistency alone is INDEFINITE and that is now measured.** `ntN` (the
consistency term `-oint w [u_t d_t v_r + v_t d_t u_r]`) diverges: toroid k=1
76 -> 293/1422/2723 at nt25/50/100; W7-X diverges at nt25. `nt0` reproduces
`ibpd` exactly (wiring confirmed). Only rot-ellipse `nt25` showed a flicker of
improvement (602 vs 626) before blowing up.

The missing half is the **correct stabilisation**, which was NOT implemented:

    gamma/h . oint ( d_r u_t - d_t u_r )^2

This is on the FLUX, not the value. `tg` penalises `u_t(1)^2` -- the value --
so `nt + tg` is not Nitsche and cannot restore coercivity for the right
quantity. The flux term needs radial vectors `e_val` (unit on last, for `u_r`)
and `e_flux = Lam'(1)` (TWO nonzero entries, for `u_t`) -- still low-rank, still
probe-free, but rank-2 radially. **That is the next thing to build**, and until
it exists Nitsche is untested, not refuted.

**Storage for the outer-ring / capacitance route.** Dense is `(n_t n_z)^2 =
N^(4/3)` memory and `N^2` to factor -- unacceptable at scale. Structure that is
currently thrown away: `B` is sparse (bandwidth `p+1` in each angular
direction); `D = E^T A^-1 E` is DIAGONAL in the angular eigenbasis (the closed
form already computed) so it should never be formed; and `B_hat = Q^T B Q` is
banded in mode space if the face weight has limited Fourier content. Then
`(I + D B_hat)` is sparse: `O(n_t n_z . bw)` memory, `O((n_t n_z)^{3/2})` work.
MEASURED (`face_weight_spectrum.py`, n=12, 2-D FFT of `J g^rr` at r=1) --
modes needed for a given fraction of the weight's energy, and the implied
half-bandwidths of `B_hat`:

| geometry | 99% | `m_max`,`n_max` | 99.9% | `m_max`,`n_max` |
| --- | --- | --- | --- | --- |
| toroid | 3 | 1, 0 | 3 | 1, 0 |
| rot-ellipse | 9 | 2, 2 | 25 | 4, 2 |
| w7x | 11 | 3, 2 | 39 | 26, 11 |

**At 99% the bandwidth is tiny**: W7-X needs `(2.3+1)(2.2+1) = 35` entries per
row against 288 dense -- 8x less storage, and `O(n_t n_z . 35)` instead of
`(n_t n_z)^2`. That is the O(N) route, and 99% truncation of a weight is
obviously acceptable inside a preconditioner.

The jump from 99% to 99.9% on W7-X (`m_max` 3 -> 26) is the tail going
broadband, i.e. quadrature-level noise rather than structure -- so truncate at
99% and do not chase the tail. The toroid needs 3 modes with `m_max=1`, which is
why a scalar face weight works there at all.

Use an explicit FFT basis, not `eigh` eigenvectors: degenerate cos/sin pairs are
determined only up to rotation and would smear a banded `B_hat` into a dense
one. Cheap fallback already in the code: `wdiag`, `O(n_t n_z)`, no solve.

**Diagnostic before more mechanisms.** Dense spectrum probe
(`block_jacobi_spectrum.py`) on k=1 free rot-ellipse, `nobc`/`ibpd`/`ibpd+tg`.
Watch where the knobs move the 44 outliers. Constrains every hypothesis at once
instead of one sweep each -- five were proposed today and four refuted.

**Open anomaly**: outer rings hurt at k=0 on both geometries. Explained
plausibly (exact-but-decoupled) but not verified; it means "outer rings are
exact and therefore safe" is false as stated.

---

## 9. Things NOT to redo

- **`ibp`** (non-lumped weight under diag lumping) -- double counts `w_comp`,
  diverges on W7-X n=12 k=1. Superseded by `ibpd`.
- **A fixed `rho` at k=1/2** -- no minimum exists there. At k=3 `rho` is real
  and should be COMPUTED.
- **`wibp`/`wibpd`** -- exact 2-D face by quadrature x `mu_0`, Woodbury. Within
  noise of the scalar at k=1/2, WORSE on W7-X; `wibpd` (diagonal-only
  capacitance) DIVERGES once `B` carries `mu_0`. The k=1/2 gap is not the face
  shape.
- **The trace's component coupling** -- face tilt `cos^2 phi` = 0.986-0.991
  (k=1), 0.84-0.98 (k=2). A 1-16% effect; cannot explain damping of 0.07-0.16.
  TRAP: `|g^ab|/sqrt(g^aa g^bb)` (0.39/0.51) is the MASS-coupling measure and is
  ~5.5x larger than `|g^rb|/g^rr`, which is what the boundary term needs. I read
  one as the other and it cost a wrong diagnosis.
- **An `alpha`-`beta` trade-off** -- `alpha` is a PRECONDITIONER parameter and
  cannot move the operator's boundary residual, so `beta_opt` is
  `alpha`-independent by construction. Measured; it is.
- **A Dirichlet-side boundary term (`dbN`)** -- catastrophic even at 5% (toroid
  k=1 dbc 62 -> 382; rot-ell 125 -> 1232), non-monotone at larger values
  (indefinite). Nothing to correct: the constraints are already exact (§5).
  §6.3's invariant should STAY; only its stated reason needs fixing.
- **`mode_beta_correction` (`tmN`)** -- BROKEN, now MEASURED. The equivalence
  check §9 asked for exists (`scripts/debug/tm_equivalence.py`, knob
  `MRX_BJ_TANG_MODE_FLAT=1` pins the mode factor to 1; the FD eigenvectors are
  M-orthonormal, so the Woodbury update must then reproduce `tg` EXACTLY).
  Toroid 6,12,6 p=3, `||P_tm - P_tg|| / ||P_tg||`:

  | case | ratio | moved by tg | moved by tm | |
  | --- | --- | --- | --- | --- |
  | k=1 free | **3.39e+03** | 4.6e-1 | 3.4e+03 | MISMATCH |
  | k=2 free | 8.3e-1 | 8.8e-1 | 4.2e-2 | MISMATCH |
  | k=1 dbc | 4.5e-13 | 3.1e-13 | 2.5e-13 | match (both inert) |
  | k=2 dbc | 7.9e-13 | 1.2e-12 | 5.5e-13 | match |

  It perturbs `P` by 3390x the norm of `P` at k=1 and by 20x too LITTLE at k=2 --
  two different failure modes, so not one sign or scale slip. The dbc rows being
  inert confirms the harness, not the mechanism. The near-null guard is NOT the
  explanation (`d >= 0` with the guard, so `coef = beta/(1+beta d) <= beta`);
  suspect the basis in which the capacitance is applied (`q = kron(v_t, v_z)`).
  DO NOT debug it to rescue the route: §10 shows `|m|` separates the two outlier
  families only weakly (1.4 vs 2.1), so a mode-dependent beta is not what the
  k=1/2 free gap wants. Retire or delete.
- **`ns = 4,8,4` smoke** -- every block arm throws `LinAlgError` including
  unmodified ones; `extra_rings` degenerates the bulk window at `n_r=4`. Use
  `8,16,8`.
- **Ring count vs `p`** -- measured p-independent. No need to re-sweep.

Data: `outputs/diag_{bcibp,bcref,bcref2,bcref3,ibpd,xterm,wibp,coupling,tilt,
tilt2,tang,tmode,joint,dbc,nitsche2,rings,psweep,spec}/` (all 2026-08-21).

---

## 10. THE PIN (Tobias, 2026-08-21 afternoon) — the idea that reframes §5

### The observation

The Dirichlet cases are all strong (4-7x) and their spectra are clean (k=1 dbc:
cond 45.7, 7 outliers, and those sit at `outer=0.27`, i.e. NOT boundary modes).
Yet the atom does not represent the dbc problem's conditions correctly either:
at k=1 dbc the operator wants `div u = 0` on the face, at k=2 dbc it wants
`n x curl w = 0`, and the atom asserts neither. **It gets the natural half wrong
under Dirichlet and pays nothing.**

§5 explained this as "essential conditions are enforced by the SPACE, so a wrong
boundary term cannot violate them". True but incomplete, and the incompleteness
matters: Tobias's objection is that `n x curl u = 0` -- the very condition §5
blames for the k=1 free failure -- is ALSO mis-represented at k=2 dbc, where it
costs nothing. So "the atom cannot carry a vector natural condition" cannot by
itself be the reason free lags.

### The mechanism (this is the part worth keeping)

An essential condition on the face does not merely protect itself. **It kills
the tangential derivatives of the constrained component, and the operator's
natural conditions then COLLAPSE onto exactly the scalar per-component
conditions the atom already imposes.**

* k=2 dbc: `w_r = 0` on the whole face `=> d_t w_r = d_z w_r = 0` there, so
  `n x curl w = 0` degenerates from `d_r w_t = d_t w_r` to `d_r w_t = 0`.
  That IS the atom's condition. The atom is not getting away with a wrong
  condition; the essential condition makes its condition right.
* k=1 dbc: `u_t = u_z = 0` on the face `=> d_t u_t = d_z u_z = 0`, so
  `div u = 0` collapses onto the radial term alone.

So the rule is not "natural conditions do not matter". It is:

> **Pin one component's trace and the others' natural conditions become the
> scalar conditions a per-component Laplacian can express.**

### The prescription

Under a FREE condition nothing is pinned, every coupling survives, and the atom
asserts `d_r u_c = 0` per component where the operator asks for something
coupled. Impose the missing pin IN THE PRECONDITIONER and the same degeneracy
buys the same thing:

| k | pin on the face | then the atom's own conditions become |
| --- | --- | --- |
| 0 | nothing (already exact) | -- |
| 1 free | `u_r = 0` | `d_t u_r = 0`, so `d_r u_t = 0` and `d_r u_z = 0` ARE `n x curl u = 0` |
| 2 free | `w_t = w_z = 0` | `d_t w_t = d_z w_z = 0`, so `d_r w_r = 0` IS `div w = 0` |
| 3 free | `omega = 0` | single component; the free condition is already a value condition |

The component to pin is in every case the one whose radial axis is a DERIVATIVE
axis -- the `has_trace` set, i.e. exactly the component that carries the
natural-BC term `alpha e e^T`. So **the pin REPLACES the penalty on the same
component**; it is the hard-constraint limit of the thing §2 derived a
coefficient for, not an addition to it.

The preconditioner is then a different operator from `L` -- it has a boundary
condition `L` does not have. That is allowed and is the whole bet: the dbc
spectra say the pinned problem is the well-conditioned one (cond 45 vs 800),
and the mismatch is confined to the boundary trace, which CG handles as a few
isolated eigenvalues rather than a smeared band.

### Why `tg` is the WRONG lever, now with evidence

`tg` penalises the NON-trace components (`u_t, u_z` at k=1). Pinning `u_t` does
nothing for `d_r u_t = d_t u_r` -- the coupling is through `u_r` -- so it buys
no degeneracy, it only adds stiffness. The mode-resolved spectrum (n=6,12,6,
p=3, `diag_pinspec`/`diag_spec3`) shows what that costs:

* toroid, the modes `tg` damages are `|m| = 0.00` and `comp[0.00/0.00/0.98]` --
  PURE `u_z`, zero normal content. For those `d_z u_r = 0` already, so the
  atom's condition was exact and the penalty is pure damage:
  `lambda_min` 3.8e-2 -> 1.3e-2 -> 5.4e-3 for tg 0/10/30, low outliers 0/5/16.
* rot-ellipse: `tg30` removes every high outlier (9 -> 0) and creates 50 low
  ones; cond 837 -> 361 but `lambda_min` 2.5e-2 -> 1.4e-2.

That trade -- cure the too-soft modes, damage the already-correct ones -- is why
the best `tg` is an interior optimum and why it needs one number per geometry.
A mode-dependent `beta` does NOT rescue it: `|m|` separates the two families
only weakly (high outliers `|m|`=1.4, low ones 2.1 on rot-ellipse), so the
discriminator is component content, not mode number. `tm`/`mode_beta_correction`
is therefore no longer the priority it was in §8.

Pinning `u_r` instead constrains the component the free-BC operator ALREADY
wants to vanish (`u.n = 0` is its natural condition), so it is the smallest
perturbation available, not the largest.

### Implementation (in the tree, uncommitted)

`core_rows(..., pin_rings, pin_comps)` + `BlockJacobiLaplacian(pin_trace=N)`,
arm key `pinN`, and `trace_components(k)`.

Moving a component's outer radial ring out of the bulk shrinks THAT component's
window by one, and the windowed radial factors are the Dirichlet-eliminated
ones -- so the pin is exact, not a penalty, and needs no coefficient. The
removed rows are not dropped: they go to the dense probe, which is where the
Steklov/DtN coupling the atom cannot carry gets handled exactly anyway.

Two consequences worth noting:

* `bc_entry` is forced OFF on pinned components (the term would otherwise land
  on the last WINDOW row, which is no longer the face). Hence `ibpd_pinN` and
  `nobc_pinN` are the SAME operator -- the pin replaces the natural term.
* Cost is `pin_rings * n_t * n_z` probe applies for the TRACE components only,
  versus `outer_rings`, which takes the ring on all three: 9x smaller dense
  block at k=1 (one component of three), 2.25x at k=2.

### The pin REPLACES the derived coefficient -- and at k=3 it already wins

The pin forces `bc_entry=False` on precisely the components that carry the
natural-BC term, so `ibpd_pinN` IS `nobc_pinN`. The two are mutually exclusive
on the same component: with the face DOF evicted there is no face row for
`alpha e e^T` to land on, and putting it on the new window edge would be wrong.
So the pin is not a complement to §2's coefficient, it is a rival to it.

At k=3 free the trace set is the single component, so `outer_rings=1` IS the
pin, and `diag_rings` already scores the two head to head (n=12, p=3):

| k=3 free | jacobi | `ibpd` (derived) | `o1` == pin | `o2` |
| --- | --- | --- | --- | --- |
| rot-ellipse | 235 | 75 | **50** | 45 |
| w7x | 306 | 75 | **53** | 49 |

The pin beats the derived coefficient by ~1.4x. §6's k=3 deliverable
(W7-X 123 -> 75) is superseded by 75 -> 53 on an arm that throws the
coefficient away.

**But eviction DECOUPLES the row from the bulk atom, and where the atom was
already right that is pure loss** -- the other half of the same table:

| | `ibpd` | `o1` |
| --- | --- | --- |
| k=0 free, w7x | 83 | **288** |
| k=3 dbc, w7x | 75 | **183** |
| k=1 dbc, w7x | 178 | 230 |
| k=2 dbc, w7x | 212 | 361 |

Hence the guard: trace components, free BC, only. And hence a second prediction
-- **`pin1` should beat `o1` outright, not merely cost less**: `outer_rings`
evicts all three components' rings, including the ones whose scalar condition
was already correct, and pays the decoupling on each; the pin evicts only the
component whose condition is wrong.

The coefficient is NOT thereby refuted. On the toroid `exact` beats `o2` in all
eight cases at 16^3 at half the build (memory `tensor-precond-natural-bc`), and
that fits the mechanism: the toroid is the one geometry where `g^rr` is constant
on the face, so the scalar collapse is EXACT (`<S><P>/<SP>` = 1.0000, §2) and the
term is faithful. On shaped geometries the collapse is the defect `wibp` failed
to repair. Expected split, to be confirmed by the toroid `pin1` row:

> **The derived penalty is right exactly where the face weight is genuinely
> scalar; the pin takes over where it is not.**

That is the same toroid-vs-shaped, k=3-vs-k=1/2 split that runs through the
whole investigation.

### This is NOT the refuted "hard u.n = 0"

Memory (`tensor-precond-natural-bc`) and §9 both record that hard `u.n = 0` was
refuted: a penalty of x1e4 gave 250 it at k=1 free and 334 at k=2, worse than no
term at all. That is a DIFFERENT construction and does not refute the pin.

A penalty driven to infinity leaves the boundary DOF **in the atom**, with a
huge diagonal entry. The atom's inverse on that row then goes to ZERO, so those
DOFs receive essentially no correction from the preconditioner -- the rows are
not constrained, they are ABANDONED. Unsurprisingly that is worse than a finite
penalty.

The pin removes the DOF from the atom and hands the row to the **exact probe**.
The constraint is imposed by restriction, not by a large number, and the row is
still preconditioned -- better than anywhere else in the operator, in fact,
since the probe is exact. Same boundary condition, opposite treatment of the
DOF that carries it.

This also explains `bcp300` (2998 it, W7-X k=1 free): pushing alpha up is the
first few percent of the road to x1e4, and it degrades monotonically for the
same reason.

### The pin has two halves, and only one of them is the idea

1. **Evicting the row** -- free. The window is `[i_r.min(), i_r.max()]` over the
   rows the component still owns, and the radial factors are `cut(K_r, window)`;
   restricting a stiffness to interior indices IS Dirichlet elimination. The
   atom gets SMALLER and fast diagonalisation is untouched.
2. **Preconditioning the evicted row** -- not free. `pin_mode="probe"` (the
   default) hands it to `probe_core_block`: one operator apply per row and a
   dense inverse, `O((n_t n_z)^2)`, the same thing §8 flags as unscalable.

Only (1) is the argument. If the two are not separated the experiment is
uninterpretable -- `pin1 ~ o1` would only re-prove that dense-probing the
boundary works, which `outer_rings` established long ago.

So `pin_mode="diag"` (arm `pindN`) evicts the row and gives it the operator's
JACOBI DIAGONAL: no applies, no dense block, O(1) storage.

* `pind1 ~ pin1` -- the benefit is the degeneracy argument, and the pin is
  essentially free: a smaller atom plus a diagonal.
* `pind1 >> pin1` -- the benefit was the exact boundary treatment all along,
  the pin is cosmetic, and §8's banded-capacitance route is the only way out.

Cost note either way: at k=1, n=12 the pinned ring is `n_t n_z` = 288 rows
against the ~2600 `extra_rings=3` already probes, so the MARGINAL probe cost is
small -- but the probe as a whole is what does not scale, and §8's banded
`B_hat` applies to the pinned ring exactly as it does to the core.

### MEASURED: the trace pin is real but SECONDARY -- prediction falsified

`diag_pinspec`, rot-ellipse 6,12,6 p=3, free BC. The prediction in this section
was "once the trace is pinned, `tg` should stop helping". It does not.

| k=1 free | cond | high | low | where the HIGH ones live |
| --- | --- | --- | --- | --- |
| `ibpd` | 837 | 9 | 33 | comp[0.00/0.78/0.21] tangential |
| `ibpd_pin1` | 661 | **9** | 23 | comp[0.00/0.80/0.20] -- UNCHANGED |
| `ibpd_pin1_tg10` | 264 | 1 | 27 | |
| `ibpd_o1` | 176 | 0 | 34 | |

| k=2 free | cond | high | low | where the HIGH ones live |
| --- | --- | --- | --- | --- |
| `ibpd` | 747 | 12 | 53 | comp[1.00/0.00/0.00] = `w_r` |
| `ibpd_pin1` | 611 | **12** | 29 | comp[1.00/0.00/0.00] -- UNCHANGED |
| `ibpd_pin1_tg10` | 237 | 0 | 37 | |
| `ibpd_o1` | 131 | 0 | 26 | |

The pin does exactly what it claims ON ITS OWN COMPONENT -- 30-45% of the LOW
outliers go (the trace component's over-stiff modes), `lambda_min` rises, cond
falls ~20%. But the HIGH outliers are untouched to the mode (9 -> 9, 12 -> 12,
same components, same `|m|`), and `tg` still cures them after pinning.

**Why the degeneracy argument does not transfer.** The atom is BLOCK-DIAGONAL
per component. Pinning `u_r` changes `u_r`'s block only; the tangential blocks
are bit-identical. It makes **P** self-consistent -- P's own solution has
`u_r = 0` on the face, so P's tangential conditions are right FOR P -- but L's
`u_r` is not pinned, so `d_t u_r != 0` in L and the two still disagree on
exactly those tangential boundary modes. The argument says which condition is
CORRECT; it does not make P match L.

**What the data says instead**, consistently at both degrees: the high outliers
always live on the components that get NO boundary term (k=1 tangential, k=2
`w_r`), where the atom silently asserts `d_r u_c = 0` and is too SOFT. They are
cured by acting on THOSE components -- evicting them (`o1`: high -> 0) or
stiffening them (`tg`: high -> 1). That is the opposite component set.

So the prescription inverts: **pin the NON-trace components** (`pin_set="other"`,
arm `pinoN` -- the hard limit of `tg`, evicting the row instead of abandoning it
under a huge penalty) **and keep the derived coefficient on the trace
components**, where §6/`diag_alphapen` prove it worth 479 -> 124 on the toroid.
Both mechanisms stay where each is demonstrated, rather than one replacing the
other.

### Why a penalty cannot be driven to the hard limit IN A PRECONDITIONER

Worth stating separately, because it is counter-intuitive and it is the reason
the pin exists at all.

`alpha e e^T` IS a penalty on `oint (u.n)^2 ds` -- it is not an approximation to
a condition we are failing to impose, it is the OPERATOR's own term, because the
free-BC weak block enforces `u.n = 0` weakly rather than by removing a DOF.
`MRX_BJ_BC_SCALE` is exactly "penalise it harder", and it degrades monotonically
past the derived value (toroid k=1 free 76 at x1, 86 at x3, 250 at x1e4).

The reason is that a large penalty leaves the DOF **inside the atom** with a
huge diagonal, so the atom's INVERSE on that row goes to zero: the DOF receives
essentially no correction, `lambda(PL) -> 0` there, and CG stalls on it. The
penalty does not impose the condition, it DELETES the row's preconditioning.

Under a true Dirichlet condition the DOF is gone from `L` and `P` alike, so
there is no row left to precondition and nothing is lost. Under free BC `L`
still has that row. That asymmetry is why the hard limit must be taken by
EVICTION (pin) rather than by a large coefficient.

### HOW MANY DOFS CARRY THE TRACE -- MEASURED, and it is one

Raised by Tobias: "more than one derivative spline is non-zero at r=1 when
p > 1". This matters structurally, because **if `e` had more than one nonzero
then `pin_trace=1` would NOT impose `u_r(1) = 0`** -- evicting the last radial
row sets the last COEFFICIENT to zero, and row removal equals the trace
constraint only when `e` is one-hot. The pin would then never have been tested.

MEASURED (`scripts/debug/dspline_endpoint.py`, bases only, no geometry, no
solve), `DerivativeSpline` at the clamped end:

| p | n | nonzeros at `1-1e-8` | at `1-1e-12` | `|e|_2nd/|e|_max` |
| --- | --- | --- | --- | --- |
| 1 | 12 | 1, `[11.0]` | 1 | 0 |
| 2 | 12 | 2, `[1e-6, 20.0]` | **1**, `[20.0]` | 5e-8 |
| 3 | 12 | 2, `[2.4e-6, 27.0]` | **1**, `[27.0]` | 9e-8 |
| 4 | 12 | 2, `[3.8e-6, 32.0]` | **1**, `[32.0]` | 1.2e-7 |

The second entry is O(eps) -- it scales linearly with the offset and is exactly
zero at the endpoint. So `e` IS one-hot and the trace is a single dof.

**The distinction that makes both readings right.** `basis_0.dLam` is NOT
`d/dx` of the value basis. It is the basis OF THE DERIVATIVE SPACE:
`DerivativeSpline` returns `s(x, i+1) . (p+1)/(T[i+p+2]-T[i+1])`, a SINGLE
scaled degree-(p-1) B-spline (the magnitudes 20/27/32 are the unit-integral
normalisation, O(1/h)), and a single B-spline on a clamped end is one-hot
there. The object with TWO nonzeros at r=1 is `Lam'(1)` -- the derivative of the
VALUE basis -- which is the FLUX functional (§8's `e_flux`), not the trace.
The name `dLam` for the D-spline basis invites exactly this collision; note it
before reasoning about boundary functionals.

Consequences:

* `pin_trace=1` does impose `u_r(1) = 0`, so §10's falsified prediction is a
  GENUINE result and not an artefact of constraining the wrong functional.
* No rank-one-constraint rebuild (`K_r -> Z^T K_r Z`) is needed -- row eviction
  already IS the trace constraint. Keep that construction in mind only for the
  FLUX condition, where `e_flux` really is rank-2 radially.
* `pin2` should therefore be "more of the same" rather than qualitatively
  different. That is a live prediction of the running `diag_pin` job.

`scripts/debug/edge_vector_check.py` repeats this on the real sequence
(polar basis, window, geometry) and additionally reports how hard the derived
penalty stiffens the boundary row relative to `K_r[-1,-1]`.

---

## 11. RESULTS AND PLAN — end of 2026-08-21

**If you are picking this up cold: read "TOMORROW: WHAT TO SUMMARIZE, in order"
near the end of this section first.** 33 jobs were launched overnight and two
of them are GATES whose outcome decides whether the others mean anything.


### The numbers (n=12, p=3, extra_rings=3, tol 1e-10)

k=1 free / k=2 free iterations. `--` = job still running when this was written;
read it from the logs listed at the end of this section.

| k=1 free | toroid | rot-ellipse | w7x |
| --- | --- | --- | --- |
| jacobi | 442 | 739 | 1657 |
| `ibpd` (derived coefficient) | **76** | 620 | 1834 |
| `ibpd_pin1` (pin TRACE) | 76 | 607 | 1874 |
| `ibpd_pin2` | 76 | 661 | 2116 |
| `ibpd_pind1` (trace, no probe) | 195 | 932 | 1974 |
| `ibpd_pino1` (pin OTHERS) | (spec only) | 432 | 907 |
| `ibpd_pinod1` (others, no probe) | (spec only) | 481 | 1323 |
| `ibpd_pina1` (pin ALL) | (spec only) | **307** | **559** |
| `ibpd_o1` | 200 | 308 | 562 |
| `ibpd_tg10` | 124 | 368 | 864 |
| `ibpd_pin1_tg10` | 124 | 339 | 884 |
| `ibpd_bcp30_tg10` | 130 | 316 | 709 |
| `nobc_tg10` (alpha = 0) | 479 | 802 | 1608 |

| k=2 free | toroid | rot-ellipse | w7x |
| --- | --- | --- | --- |
| jacobi | 345 | 621 | 1555 |
| `ibpd` | **62** | 652 | 1790 |
| `ibpd_pin1` | 61 | 625 | 1998 |
| `ibpd_pin2` | 61 | 670 | 2230 |
| `ibpd_pind1` | 155 | 684 | 1701 |
| `ibpd_pino1` | (spec only) | 431 | 931 |
| `ibpd_pinod1` | (spec only) | 452 | 1280 |
| `ibpd_pina1` | (spec only) | **261** | **497** |
| `ibpd_o1` | 111 | 261 | 497 |
| `ibpd_tg10` | 73 | 376 | 887 |
| `ibpd_bcp30_tg10` | 78 | 308 | 725 |
| `nobc_tg10` (alpha = 0) | 357 | 749 | 1641 |

`pina1` == `o1` on every geometry and both degrees (307/308, 261/261, 559/562,
497/497), which VALIDATES the pin implementation -- they are the same object up
to `bc_entry` being correctly switched off on evicted components. Evicting ALL
components is worth ~1.6x over evicting only the non-trace ones (559 vs 907),
so the trace ring does carry weight -- but only once the others are gone
(alone it is worth nothing: 1874).

On the TOROID every eviction arm loses to the plain derived coefficient
(`o1` 200 vs `ibpd` 76 at k=1, 111 vs 62 at k=2), and the spectra agree
(`pino1`/`pina1` cond 464 vs 75). The geometry split is structural.

### What is now settled

1. **The trace pin is REFUTED as a mechanism.** 76/76/76 on the toroid,
   620 -> 607 on rot-ellipse, 1834 -> 1874 on W7-X. `pin2` is "more of the
   same", confirming `e` is one-hot and that the trace condition really was
   imposed. The spectrum said why in advance: it moves LOW outliers only, and
   the failing modes carry ZERO `u_r`, so `u_r = 0` was already true for them.
   The degeneracy argument is correct and VACUOUS where it is needed.
2. **Eviction is not free.** `pind`/`pinod` (evict, then Jacobi diagonal on the
   evicted rows) costs ~1.5x on the trace components (toroid 76 -> 195,
   rot-ell 620 -> 932). The dense probe on the evicted rows is load-bearing.
   Note the asymmetry: `pinod1` (others, diagonal) still IMPROVES on the
   baseline (620 -> 481) where `pind1` (trace, diagonal) degrades it (932).
3. **The gains are non-additive, and the "other" components carry most of it.**
   rot-ell k=1: pin-trace alone 607 (nothing), pin-other alone 432,
   pin-both 307. So the trace ring does contribute -- but only once the others
   are gone. `pina1` = 307 vs `o1` = 308 also VALIDATES the implementation
   (they are the same object up to `bc_entry` being correctly switched off).
4. **`alpha` earns its place on every geometry, and `alpha=0` is the worst arm
   anywhere.** With `tg10` on: 479/802/1608 at alpha=0 against 130/316/709 at
   alpha=0.30. The optimum with a penalty present is alpha ~ 0.3 on the shaped
   geometries and >= 1.0 on the toroid -- still geometry dependent, and NOT
   explained by the cross-term `rho` (0.63 toroid would predict the opposite
   ordering).
5. **`bcp30_tg10` is the best PROBE-FREE arm on shaped geometries**: 316 vs
   `o1` 308 on rot-ellipse (97% of the benefit) and 709 vs 562 on W7-X (80%),
   at zero extra dense rows. If a cheap default is wanted today, that is it --
   but it needs one fitted number per geometry, which is why it is not landed.
6. **`mode_beta_correction` is broken** -- see §9, now measured.

### The bad modes are boundary-LOCALISED but not a boundary CONDITION

Everything that fixes a condition has now failed on them: the correct
coefficient (`ibpd`), the correct face SHAPE (`wibp`), the consistent
cross-component term (Nitsche `nt`, diverges), and the hard trace constraint
(`pin`, no-op). Everything that treats the boundary REGION more exactly works,
and keeps working deeper: `o1/o2/o3` = 1796/561/391/349 on W7-X k=1 free. A
boundary condition lives on the FACE, so one exact ring should capture it; that
gains continue at the third ring says the defect is a LAYER.

The pattern that ties it together, two for two -- **the bad modes live on
exactly the boundary DOFs that the DIRICHLET problem deletes**:

| | high outliers live on | dbc deletes | atom's boundary term |
| --- | --- | --- | --- |
| k=1 free | `u_t, u_z` (comp 0.00/0.78/0.21) | `u_t = u_z = 0` | none (non-trace) |
| k=2 free | `w_r` (comp 1.00/0.00/0.00) | `w_r = 0` | none (non-trace) |

"Where the high outliers live", "what dbc deletes" and "which components get no
boundary term" pick out the SAME set. So dbc's immunity is not that its
conditions are well represented (they are not -- §10) but that **those DOFs do
not exist there**. Under a free condition they do, the atom is a poor model at
them, and no condition-level fix reaches them.

### Spectra that landed after the tables above

**W7-X, first spectrum ever taken there (6,12,6, `ibpd`, free):** k=1 cond
4648, 73 outliers (20 high, 53 low), high on comp[0.00/0.49/0.51] TANGENTIAL;
k=2 cond 4751, 91 outliers (16 high, 75 low), high on comp[1.00/0.00/0.00] =
`w_r`. So "high outliers live on the components with no boundary term" is now
**4 for 4** (rot-ell k=1/k=2, W7-X k=1/k=2).

**`pino` does what `pin` could not -- it kills the high outliers**
(`diag_pinospec`, 6,12,6):

| rot-ellipse | cond | high | low |
| --- | --- | --- | --- |
| k=1 `ibpd` | 837 | 9 | 33 |
| k=1 `pino1` | 291 | **0** | 62 |
| k=1 `pina1` | 176 | **0** | 34 |
| k=1 `pinod1` (no probe) | 586 | 8 | 78 |
| k=2 `ibpd` | 747 | 12 | 53 |
| k=2 `pino1` | 298 | **0** | 56 |

but it pays in LOW outliers (33 -> 62 at k=1), which is the decoupling cost, and
`pina1` pays less than `pino1` -- consistent with the non-additivity in the
iteration counts. On the TOROID the same move is a net loss (k=1 cond 75 -> 464,
2 -> 16 low outliers; k=2 cond 13.9 -> 20.0), which is exactly why `o1` costs
76 -> 200 there. The geometry split is structural, not a tuning artefact.

**The derived `alpha` already stiffens the boundary row 8.2x**
(`diag_edge`, toroid 8,16,8): `alpha` = 2.049e3, `alpha e_last^2` = 4.611e5
against `K_r[-1,-1]` = 5.626e4. IDENTICAL for (k=1,c=0), (k=2,c=1), (k=2,c=2),
(k=3,c=0) -- an independent confirmation of §2's claim that `alpha` is the same
for every degree. So "penalise harder" starts from 8x, and the refuted x1e4
would be ~8e4 x. `e` on the real sequence (polar basis, window, geometry) is
one-hot to 5e-8, confirming §10.

### `L V` DECAYS -- the storage constant is not irreducible

MEASURED (`fm_cost.py`, rot-ellipse n=12 k=1), `|LV|^2` by depth from the outer
ring:

    d0 = 0.995   d1 = 0.005   d2..d5 = 0.000   deeper = 0.000

`LV` is 99.5% supported on the outer ring ITSELF. The `raw_kron` spreading is
energetically negligible against the boundary rows' own `O(1/h^2)` magnitudes.
So both `V` and `LV` live on a thin slab and the dominant storage collapses:

* untruncated `2 n q` = 20.6 MB (n=12, q=147, `fm3`);
* slab of 2 rings `2 q (2 n_t n_z)` = 147 x 576 x 2 x 8 B ~ **0.7 MB**.

~30x, and the SCALING changes character: `O(n^{2/3} q)` instead of `O(n q)`,
which beats `o1`'s `O(n^{4/3})` by a factor `n_t n_z / q`. `A_0 = V^T (L V)`
samples `LV` only on `V`'s support, so the Galerkin block is UNCHANGED by
truncation; only the `(I - L Q)` factors move, by 0.5%, inside a
preconditioner.

Implemented as `coarse_trunc=D` (arm `ftD`): `V` and `LV` are held on a slab
`D` rings deeper than the coarse rings, with a gather/scatter in the apply.
**`ft9` must reproduce the untruncated arm EXACTLY** -- that equivalence check
is `outputs/diag_ftcheck/` and should be read BEFORE trusting any `ft` number
(same discipline as the `tm` check in §9, which caught a 3390x error).

### ROUTE (1) IS WEAKENED -- read this before building the coarse space

`diag_spech`, rot-ellipse k=1 free `ibpd`, refinement 6,12,6 -> 8,16,8:

| | n | ring `n_t n_z` | outliers | high | cond | high `\|m\|` |
| --- | --- | --- | --- | --- | --- | --- |
| 6,12,6 | 894 | 72 | 42 | 9 | 837 | 1.39 |
| 8,16,8 | 2344 | 128 | **98** | 31 | 2247 | **1.38** |

The outlier COUNT grows 2.3x while the ring grows 1.8x -- so it tracks
`n_t n_z`, NOT fixed. A coarse space of fixed size will not span it, and
route (1) as stated ("`O(1)` vectors") is dead in that form. Note also cond
WORSENS under refinement (837 -> 2247), i.e. the atom degrades with h at k=1
free on shaped geometry -- consistent with `o1/o2/o3` never saturating.

BUT the mode content does NOT drift: high-outlier `|m|` is 1.39 -> 1.38 across
the refinement. Those two facts are in tension and the mean is too blunt to
resolve them -- if the outliers were spreading over all angular modes the mean
`|m|` would grow like `n_t`. The resolution is probably that the count grows in
the RADIAL direction (more boundary-layer profiles per angular mode) at fixed
low `|m|`, which would keep a TRUNCATED-mode treatment viable: keep
`|m| <= 3, |n| <= 3` for each of a few radial layers.

**FIRST JOB TOMORROW: re-run the probe reporting the `|m|` HISTOGRAM and MAX,
not the mean** (a two-line change in `block_jacobi_spectrum.py`'s
`mode_content`). If max `|m|` stays ~3 while the count grows, mode truncation
works and routes (1) and (2) merge into "treat the outer rings in a truncated
Fourier basis". If max `|m|` grows like `n_t`, only route (2) survives.

### THE PLAN FOR TOMORROW

The question is now: handle those boundary DOFs better WITHOUT an
`O((n_t n_z)^2)` dense probe. Three routes, ranked.

**CONSTRAINT (Tobias): keep the boundary handling INSIDE the preconditioner --
no deflation.** This is not a restriction, it is the natural form, and the code
is already shaped for it: `core_inv = (R L R^T)^-1` applied as
`out.at[core].set(core_inv @ x[core])` IS an in-preconditioner coarse
correction with `R` = a selection of rows. Everything below is a change to `R`,
not a change to CG. Keeping it in `P` also keeps `P` SPD (a sum of SPD pieces)
and composes with the harmonic-nullspace deflation already in the driver
instead of competing with it.

**(1) A TRUNCATED-MODE coarse correction, additive, inside P. IMPLEMENTED 2026-08-21, results in `outputs/diag_fm*/`.**
The outliers' measured mode content is `|m| = 1.4-2.3`, `|n| = 1.3-2.1` on both
geometries and in BOTH the high and low families -- a handful of low modes, not
a mesh-scaling family. The face weight is banded the same way (99% inside
`|m|<=3, |n|<=2` on W7-X). Generalise `R` from a selection of the
ring's rows to a RESTRICTION of those rows onto `|m| <= M`, `|n| <= N`:

* now: `R` picks all `n_t n_z` rows -> `(n_t n_z)^2` block, `n_t n_z` probe
  applies;
* proposed: `(2M+1)(2N+1)` columns -> one probe apply per coarse VECTOR rather
  than per row. At `M = N = 3` that is 49 per component-ring against 288, and
  it stops growing as the mesh refines while `n_t n_z` keeps growing.

One design change comes with it: the present core block REPLACES the atom on
its rows, which is only valid because `R` is a selection and those rows were
removed from the bulk. A truncated basis does not span the rows, so it must be
ADDITIVE -- keep the rows in the bulk atom and add `R^T (R L R^T)^-1 R`.
Textbook two-level additive Schwarz.

  PRECONDITION: MEASURED AND IT FAILED IN THE STATED FORM -- the count tracks
  `n_t n_z` (42 -> 98 as the ring goes 72 -> 128). The mode NUMBERS do not
  drift though (`|m|` 1.39 -> 1.38), so the surviving version is a
  TRUNCATED-MODE treatment rather than a fixed-size space. Resolve with the
  histogram run described above before building anything.

IMPLEMENTATION (in the tree, uncommitted):

* `coarse_ring_basis(seq, k, dirichlet, rings, m_max, n_max, comps, exclude)`
  -- cos/sin over the `(m, n)` box on each outer ring of each component,
  scattered onto that ring's extracted rows, zeroed on the rows the dense core
  REPLACES (so nothing is double counted), then a QR to drop the redundant
  columns the box generates (m=0 pairs n with -n). Returns orthonormal columns.
* `coarse_correction(...)` -- `V (V^T L V)^-1 V^T`, one operator apply per
  coarse COLUMN, pseudo-inverted by `eigh` with the same tolerance pattern as
  `core_inv`.
* `BlockJacobiLaplacian(coarse_rings=R, coarse_modes=(M, N), coarse_set=...)`,
  applied ADDITIVELY and LAST in the jitted apply -- every other branch writes
  with `.set()`, so the coarse term must come after them or be overwritten.
* Arm keys: `fmM` (mode cutoff, implies 1 ring), `frR` (rings), `fso`/`fst`
  (restrict to the non-trace / trace components).

Cost at n=12 (`n_t, n_z` = 24, 12), `M = N = 3`: 49 modes per component-ring x
3 components = 147 coarse vectors, so 147 applies and a 147^2 block, against
`o1`'s 864 rows -> 864 applies and an 864^2 block. ~6x cheaper to build, ~35x
smaller dense inverse, and it does NOT grow with the mesh while `o1` does.

**ADDITIVE IS THE WRONG COMBINATION -- measured, and it is structural.**
First run (rot-ellipse k=1 free, 6,12,6, `fm2`, ADDITIVE):

| | cond | high | low | `lambda_min` | `lambda_max` |
| --- | --- | --- | --- | --- | --- |
| `ibpd` | 837 | 9 | 33 | 2.52e-2 | 21.05 |
| `ibpd_fm2` additive | 617 | **10** | 29 | 3.55e-2 | **21.90** |

It raises `lambda_min` and removes LOW outliers -- and leaves the HIGH ones
untouched with `lambda_max` slightly WORSE. That is forced: a high outlier means
`P` is too LARGE there (the atom is too soft), and `P + V A_0^-1 V^T` can only
make `P` larger. Additive two-level Schwarz assumes the local part
UNDER-resolves the coarse modes; this atom OVER-resolves them. No choice of
`V` fixes that -- the COMBINATION is wrong.

So use the symmetric HYBRID (balancing) form, still entirely inside `P`:

    P = Q + (I - Q L) M (I - L Q) ,     Q = V A_0^-1 V^T

which REMOVES the atom's action on the coarse space instead of adding to it.
Crucially it costs **no extra operator apply**: `L V` is already formed when
building `A_0 = V^T (L V)`, so `L Q x = (L V) A_0^-1 (V^T x)` is a dense matvec
against a stored `n x q` array (~10 MB at n=12, q=147). One atom apply per
iteration, as before. `coarse_mode="hybrid"` is the DEFAULT; `"additive"`
(arm `fadd`) is retained as the diagnostic contrast.

This is also the general lesson for anything else added to this atom: **the
atom's boundary defect is being too SOFT, so corrections must replace or
subtract, never add.** `core_inv` gets this right by using `.set()`; `tg` gets
it right by stiffening; the additive coarse term got it wrong.

**HYBRID RESULT (rot-ellipse k=1 free, 6,12,6) -- it beats `o1`:**

| | cond | high | low | `lambda_max` |
| --- | --- | --- | --- | --- |
| `ibpd` | 837 | 9 | 33 | 21.05 |
| `fm2` ADDITIVE | 617 | 10 | 29 | 21.90 |
| **`fm2` hybrid** | **134** | **0** | 26 | 5.00 |
| **`fm3` hybrid** | **104** | **0** | 20 | 4.47 |
| `o1` (full ring, all components) | 176 | 0 | 34 | 3.85 |

Every high outlier is gone and cond 104 is BETTER than `o1`'s 176, on 49 modes
per component-ring against 72 rows (at n=12: 49 against 288, and the gap widens
with refinement because the mode count is FIXED while the ring grows).

**And it is harmless where the atom is already good** -- toroid k=1 free
`ibpd` 75.3 -> `fm2` 68.7 -> `fm3` 68.6, against `o1` at 464 (200 iterations).
Nothing is decoupled, so there is no ring-eviction penalty on the
well-conditioned geometry. That removes the geometry split which made every
earlier mechanism (`tg`, `o1`, `pino`) a per-case judgement call: this one can
be on by default.

**CONFIRMED IN ITERATIONS -- but it does NOT beat `o1` in general.** n=12,
p=3, `extra_rings=3`:

| | `ibpd` | `fm2` | `fm3` | `fm3_fr2` | `o1` |
| --- | --- | --- | --- | --- | --- |
| rot-ell k=1 | 618 | 361 | 291 | **268** | 306 |
| rot-ell k=2 | 652 | 413 | 313 | 298 | **262** |
| W7-X k=1 | 1839 | 1106 | 814 | 736 | **562** |

It wins on rot-ellipse k=1 and gets 70-80% of `o1`'s benefit elsewhere; the gap
GROWS with geometry difficulty. That fits the measured face-weight bandwidth
(§8): W7-X needs `m_max=3, n_max=2` for 99% of the weight but 26/11 for 99.9%,
so W7-X plausibly wants more modes than `fm3` supplies. Mode sweep
`fm3..fm6` in `outputs/diag_fmwide/`.

**MEASURED COST** (`scripts/debug/fm_cost.py`, rot-ellipse n=12 k=1, n=8700,
jitted apply, warm, `block_until_ready`):

| arm | build | q | apply | MB core | MB coarse | MB total |
| --- | --- | --- | --- | --- | --- | --- |
| `ibpd_r3` | 50.0s | -- | 0.09 ms | 11.8 | 0 | 11.8 |
| `ibpd_r3_fm2` | 29.8s | 75 | 0.10 ms | 11.8 | 10.5 | 22.3 |
| `ibpd_r3_fm3` | 30.9s | 147 | 0.12 ms | 11.8 | 20.6 | 32.4 |
| `ibpd_r3_fm3_fr2` | 34.1s | 294 | 0.14 ms | 11.8 | 41.6 | 53.4 |
| `ibpd_r3_o1` | 49.2s | -- | **0.09 ms** | 34.5 | 0 | **34.5** |

**STORAGE ANATOMY -- what is a "block" and what is not** (Tobias asked why the
polar block is negligible when it is `O(n_zeta)`):

| object | size | at n=12, k=1 | grows as |
| --- | --- | --- | --- |
| polar block (dense, inverted) | `O(n_z)` rows -> `O(n_z^2)` | ~60 rows, **29 KB** | `n^{1/3}` rows |
| `o1` core block (dense, inverted) | `(3 n_t n_z)^2` | **34.5 MB** | `n^{4/3}` |
| fm Galerkin block `A_0` | `q^2`, q FIXED | **173 KB** | constant |
| fm basis `V` | `n x q` dense as stored | 10.2 MB | `O(n)` |
| fm image `L V` | `n x q` dense | 10.2 MB | `O(n)` |

So the polar block is kilobytes because it is `O(n_z^2)` on ~5 rows per
zeta-plane, and **fm's dense BLOCK is tiny too** -- 173 KB, ~200x smaller than
`o1`'s and constant under refinement. What costs 20.6 MB is not a block at all
but the basis and its image, `O(n q)`, i.e. proportional to the TOTAL dof count
rather than to the boundary. `o1` and `fm` therefore have completely different
storage shapes: a big dense block over boundary rows versus two skinny dense
arrays over the whole domain.

Two consequences:

* **`V` is pure waste as stored.** Each column lives on ONE ring of ONE
  component, so it is `q x n_t n_z` = 42k nonzeros = 0.34 MB sparse against
  10.2 MB dense. 30x, free, not yet done.
* **`L V` is the real question.** It is dense because
  `apply_hodge_laplacian_approx` uses the `raw_kron` inner inverse -- a
  Kronecker product of dense 1-D inverses -- which spreads a ring-supported
  vector over the whole domain. But dense is not the same as non-decaying: if
  `L V` decays away from the boundary it truncates to the outer few rings and
  fm drops from `O(n q)` to `O(n^{2/3} q)`. Measured by the
  `|LV|^2`-by-depth profile in `fm_cost.py` (`fm_lv_decay` run).
  NOT storing it is the wrong trade -- the hybrid needs `L Q x` and `Q L w`,
  which would be two extra operator applies per iteration (~3x per-iteration
  cost against 2.1x fewer iterations).

**AT n=12, `o1` IS SIMPLY BETTER** -- same storage as `fm3` (34.5 vs 32.4 MB),
CHEAPER per apply (0.09 vs 0.12 ms; a dense matvec is very GPU-friendly), and
better iterations (562 vs 814 on W7-X k=1). `fm`'s entire case therefore rests
on SCALING, which is asserted above and NOT yet demonstrated:
`fm` is `2 n q` with `q` FIXED -> `O(n)`, while `o1` is `(rows)^2` with
rows proportional to `n_t n_z` -> `O(n^{4/3})`. Estimate at n=16: ~48 MB
(`fm3`) against ~109 MB (`o1`). Measured in `outputs/diag_fmcost/`
(`fm_cost_n16`, `fm_cost_n20`). **If the crossover is not there, `fm` has no
case and route (2) -- the banded capacitance, which compresses the SAME object
without a Galerkin space -- is the one to build.**

With `extra_rings=0` the core is 0.0 MB (the polar block is negligible), so
`fm` is then the only stored object: 10.5 / 20.6 / 41.6 MB for
`fm2 / fm3 / fm3_fr2`.

Build times in the FIRST row of each `fm_cost` table are inflated by JIT
warm-up of the operator apply (`ibpd_r0` 24.0s vs `fm2` 3.1s in the same run);
compare non-first rows only.

Apply overhead is **+33%** for `fm3` against 2.1x fewer iterations -- a clear
net win on time. Two corrections to earlier claims in this section:

* the `extra_rings=3` dense core is 11.8 MB (~1215 rows), NOT the 54 MB / 2592
  rows estimated above by counting `3 rings x 3 components x n_t n_z`. Do not
  re-derive it; measure it.
* with `fm3` the COARSE SPACE is now the larger storage item (20.6 MB vs 11.8).
  `2 n q` doubles is `O(n)` but with a heavy constant -- 2.3 KB/DOF at q=147,
  which would be ~2.3 GB at `n = 10^6`. Halving it is easy (`V` is
  structurally sparse -- q columns x `n_t n_z` entries -- and is currently
  stored dense); `LV` is the irreducible half unless it turns out to DECAY,
  which is worth measuring before assuming.

**WHAT THE `fm` NUMBERS STILL CARRY (Tobias's question -- read before quoting
any cost claim).** The arms are `bj_r3_ibpd_fm*`, and `_r3` is
`extra_rings=3`: three INNER rings (the axis side) in the DENSE probed core.
At n=12, k=1 that core is polar + 3 rings x 3 components x `n_t n_z` = 2592
rows, i.e. a `(9 n_t n_z)^2` dense matrix (~54 MB) eigendecomposed at build.

So `fm` did NOT remove dense probing -- it removed the OUTER one. The
configuration measured is "dense inner core, Fourier outer edge". `fm` replaced
`o1`'s outer probe (+864 rows, +41 MB) with a fixed-size coarse space; the
inner core is pre-existing and untouched.

The inner core is the part already known to be compressible: §9 records that a
separable 2-D ring atom MATCHES the dense probe on INNER rings at a tenth of
the build cost and only loses on OUTER ones -- exactly the split we now have.
`core_mode="atom2d"` (arm `a2d`) makes the inner rings separable ring atoms
(`O(n_t^2 + n_z^2)`) and probes only the polar rows, a handful independent of
the ring size.

**`a2d` + `fm` would be the first configuration with NO `(n_t n_z)^2` dense
block anywhere**: separable atom in the bulk, separable ring atoms at the axis,
Fourier-truncated correction at the edge, tiny polar probe. Under test in
`outputs/diag_a2dfm/` (arms `ibpd / fm3 / a2d / fm3_a2d / fm3_fr2_a2d`, n=12,
rot-ellipse and W7-X, k=1 and k=2 free) with costs in
`outputs/diag_fmcost/`.

Cost accounting for `fm` ALONE (q = components x rings x (2M+1)(2N+1); at n=12
k=1, `fm3` gives q = 3 x 49 = 147):

* storage `2 n q + q^2` doubles = 20.5 MB (`fm3`), 10.5 MB (`fm2`), 41 MB
  (`fm3_fr2`) -- against `o1`'s +41 MB;
* apply: four dense `n x q` matvecs, `4 n q` ~ 5.1M flops, comparable to what
  `o1` adds to the `core_inv` matvec (+5.2M);
* `q` is FIXED under refinement while the ring grows as `n^{2/3}`, so `fm` is
  `O(n)` storage and apply against `o1`'s `O(n^{4/3})` and `O(n^2)` factor.
* caveat: `V` is structurally SPARSE (each column lives on one ring of one
  component, 288 of 8700 entries) and is currently stored dense by
  `np.stack` -- about half the storage is recoverable. `LV` is genuinely dense,
  because the approximate Laplacian's inner `raw_kron` inverse is a Kronecker
  product and therefore global.

NEXT, in order:

* finish the table (W7-X, k=2, and the toroid, where the spectrum says
  harmless: cond 75.3 -> 68.6);
* h-scaling: the count of LOW outliers grew with `n_t n_z` for the plain atom.
  The hybrid leaves 20-26 of them -- do THEY scale? Run 8,16,8 and compare.
  This is what decides whether the coarse size can stay fixed;
* sweep `M` and rings properly (`fm2/fm3/fm4`, `fr1/fr2/fr3`) for the knee --
  `fm3_fr2` was the best of four points, not a located optimum;
* the residual defect is now entirely `lambda_min` (4.3e-2), modes the atom is
  too STIFF on -- OPPOSITE sign to everything chased today, so expect a
  different mechanism; do not reuse today's tools on it;
* decide the dbc question: the correction is NOT guarded on `dirichlet` and
  `coarse_set="all"` will build there too. The dbc cases are already strong,
  and §11 shows eviction HURTS them; check the dbc rows before defaulting it on.
* if this holds up, it likely SUPERSEDES `outer_rings` entirely (cheaper,
  better, mesh-independent, and harmless on the toroid where `o1` costs 200 vs
  76). That would also retire the `extra_rings`/`o` cost/benefit knee of §6.

**(2) The banded capacitance (§8). Safe fallback, already designed+measured.**
`D = E^T A^-1 E` is DIAGONAL in the angular eigenbasis FD already computes and
should never be formed; `B_hat = Q^T B Q` is BANDED because the face weight has
limited Fourier content (half-bandwidths at 99% energy: toroid (1,0),
rot-ellipse (2,2), W7-X (3,2)). Then `(I + D B_hat)` is sparse:
`O(n_t n_z . 35)` storage and `O((n_t n_z)^{3/2})` work. Use an explicit FFT
basis, NOT `eigh` eigenvectors (degenerate cos/sin pairs would smear the band).

**(3) Schwarz relaxation on the boundary block.** Keep the rows; apply a few
matrix-free Chebyshev/Jacobi sweeps restricted to the outer rings instead of an
exact inverse. `pind`/`pinod` are the zeroth-order member (one diagonal
application) and are not enough, but a few sweeps carry some radial coupling at
the cost of applies per iteration rather than storage.

RULED OUT, do not re-propose: a better SEPARABLE model of the ring (§9: a 2-D
ring atom with the ring's exact metric loses badly on outer rings -- what the
probe supplies is nonlocal radial coupling), and a rank>1 weight fit (Lynch
fast diagonalisation is exact only at rank 1, see memory
`rank-gt1-slower-laplacian`).

### ACCOUNTING FOR THE CROSS TERM -- exactly, and for free (2026-08-21 late)

What was tried before was FITTING a multiplier (the `bcp` sweeps). What was
never tried is using the exact object, which the code already builds.

§4: the cross term is structurally `-(e v^T + v e^T)`. Splitting
`v = gamma e + v_perp` gives `alpha -> alpha - 2 gamma` -- that is the scalar
`rho` -- but `off_frac` = 0.21-0.28 says **21-28% of it lies OFF the `e e^T`
direction**, so no scalar can represent it. The exact 1-D radial correction is
`F - Ktilde`, `F = M^d G A^-1 G^T M^d`, which `_boundary_entry` builds and
which the old path then threw away down to `sv[0] u0 u0^T`.

**There is no cost to taking it in full.** `K_r` is a dense `n_r x n_r` matrix
in a 1-D generalised eigenproblem, so any SYMMETRIC radial update merges in and
fast diagonalisation is untouched -- rank is irrelevant. So the exact cross
term is free; it was only ever missing because the analytic route replaced the
round-trip route.

Two new arms:

* **`ibpr`** -- rank-1 projection of the EXACT correction,
  `c_star = e^T (F - Ktilde) e / (e^T e)^2`. This is the honest, COMPUTED
  version of `rho` that §4 asked for, with no fitting.
* **`ibpf`** -- the whole correction, cross term and all.

`ibpd` vs `ibpr` isolates "is the computed rho right?"; `ibpr` vs `ibpf`
isolates "does the off-`e` part matter?", i.e. whether the 21-28% is real
signal or noise.

**RISK, and it is the reason to watch the spectrum and not just iterations:**
the cross term is NEGATIVE, so `ibpf` can make `K_r` INDEFINITE and floor CG --
the same failure mode as the k=0 raw-atom Schur rebuild
(memory `laplacian-mg-k0-status`). `min eig(P)` in the spectrum probe shows it
directly; `MRX_BJ_BC_SCALE` still scales the whole correction if a partial step
is wanted. Runs: `outputs/diag_xterm2/` (three geometries, k=1,2,3, free AND
dbc, n=12) and `outputs/diag_xterm2spec/` (toroid + rot-ellipse spectra,
`ibpd_r0 / ibpr_r0 / ibpf_r0`).

PREDICTIONS worth checking against: at k=3 free the fine sweep put the optimum
at scale 0.55-0.70 on shaped geometries and 1.00 on the toroid, and computed
`rho` = 0.63 (toroid, h-INDEPENDENT) / 0.40 (W7-X). If `ibpr` reproduces the
0.55-0.70 gain at k=3 with no knob, `rho` is settled. If `ibpf` then beats
`ibpr`, the off-`e` part is real and the scalar story in §4 is incomplete.
The dbc rows must stay invariant (the term vanishes identically under an
essential condition).

### HOW CONFIDENT ARE WE IN `ibpd`'s NUMERICAL FACTORS? (asked 2026-08-21)

**Solid.**

* `mu_0 = (M_r^logical)^-1[last,last]` is metric-FREE and came out BIT-IDENTICAL
  across all three geometries (52.26074 at n=8, 93.88387 at n=12) -- exactly the
  derivation's prediction, and a strong check because it would break under any
  metric leakage.
* Degree-independence (`m_k J / m_{k-1} = J g^rr` for every k, via
  `J^2 = prod g_aa`) verified TWICE independently: `a_ibpd` identical down all
  four `(k,c)` rows, and `edge_check` giving `alpha` = 2.0492e3 identical for
  (1,0), (2,1), (2,2), (3,0).
* At k=3 free -- the one degree where the operator is PURELY the weak block --
  the derived value is a clean interior CG minimum on all three geometries.

**Known MISSING: the cross term.** `-2 d_s^T E u` is real (rank-one,
`off_frac` 0.21-0.28) and the exact 1-D round trip says the true coefficient is
**0.4-0.65x** what `ibpd` ships. The FINE k=3 sweep confirms it (shaped
geometries prefer scale 0.55-0.70, where computed `rho` puts it); the COARSE
sweep in §2 reads 1.00 only because its next point down was 0.30. So `ibpd` is
knowingly ~1.5-2.5x too large on shaped geometries and `rho` is NOT
implemented. §2 and §4 should be read together, not separately.

**A convention, not a derivation:** `<S><P>` vs `<SP>` -- exact on the toroid
(1.0000, since `g^rr` is constant on the face) but 0.945 / 0.912 on
rot-ellipse / W7-X. A 5-9% modelling choice.

**Unexplained:** the k=1/2 free alpha optimum is ~0.3 on shaped geometries and
>= 1.0 on the toroid, and `rho` predicts the WRONG ORDERING there (0.63 toroid
vs 0.40 W7-X). Something in the k=1/2 story is still not understood.

**And the standing warning**: `exact` is PROVABLY 8-14x too small and still
beats `ibpd` at k=1/2 free. "The arm wins" is NOT evidence its factors are
right -- that is exactly how two hidden metric factors survived, each cancelled
by a fitted compensation (§3, memory `metric-factors-must-be-explicit`).

`outputs/diag_alphaverify/` (9 jobs: 3 geometries x p = 2,3,5 at ns=12,24,12)
re-runs `bc_alpha_compare.py` across p. It prints `alpha` per `(k,c)` against
the exact round-trip `c_star`, so it tests both predictions at degrees they
were never checked at: does `mu_0` stay metric-free, and does `alpha` stay
degree-independent, at p != 3? A failure there would be diagnostic.

### OVERNIGHT RUNS LAUNCHED 2026-08-21 ~18:00 (read these first)

**`outputs/diag_hsweep/`** -- THE scaling experiment, 8 jobs: rot-ellipse and
W7-X x `ns` = 8,16,8 / 12,24,12 / 16,32,16 / 20,40,20, k=1 free, p=3,
`extra_rings=0` (inner rings are a separate problem -- Tobias), tol 1e-10,
**maxiter 40000**. Arms:

    jacobi, bj_r0_exact, bj_r0_ibpd, bj_r0_ibpd_fm3,
    bj_r0_ibpd_fm3_ft2, bj_r0_ibpd_o1, bj_r0_ibpd_o2

`bj_r0_ibpd` is the BASELINE Tobias asked for -- fix the BC and do nothing else
on the outer ring. To be precise about what that arm IS (asked 2026-08-21, and
easy to conflate with the pin): it adds the PENALTY `alpha e e^T` to the trace
components' radial stiffness -- k=1 `u_r`, k=2 `u_t, u_z`, k=3 the single
component, k=0 NONE (`W_0 = 0`, and the atom's `w d_r u = 0` is already exact
there). It does NOT set anything to zero: alpha is finite, measured to stiffen
`K_r[-1,-1]` by 8.2x. The hard-zero version is `pin`, refuted in §11.
Under Dirichlet the term vanishes identically at every k. `exact` is alongside it so the h-trend of the coefficient
correction itself is visible. What the sweep decides:

* does `fm3` (FIXED modes) keep pace with `o1`/`o2` (GROWING rings) as h -> 0,
  or fall behind because more modes are needed? At n=12 `o1` beats `fm3`
  (562 vs 814 on W7-X), so `fm` needs the trend to be in its favour or it has
  no case at all;
* does the plain `ibpd` baseline degrade with h (§10 saw cond 837 -> 2247 from
  6,12,6 to 8,16,8), and how fast;
* the ratio to `jacobi`, which memory says IMPROVES with refinement on the
  toroid -- check whether that survives on shaped geometries.

**`outputs/diag_psweep2/`** -- p-sweep, 8 jobs: rot-ellipse and W7-X x
p = 2,3,4,5 at ns = 12,24,12, k=1 free, `extra_rings=0`, maxiter 40000, same
arms as the h-sweep. What exists ALREADY (§6, `diag_psweep`) is p=3 vs p=5 for
the RING arms only -- W7-X k=1 free `o0/o1/o2/o3` = 1796/561/391/349 at p=3 and
2892/1173/781/690 at p=5, with the per-ring gain p-INDEPENDENT, and jacobi
DIVERGING at p=5 where the atom converges. Untested and the point of this
sweep: the coarse space is angular, so `q` does not depend on p -- but the
stencil widens with p, so both the required mode box and the `LV` slab depth
could move. `outputs/diag_fmcost/fm_lv_p{2,3,5}` measures the slab depth vs p
directly (the `|LV|^2`-by-depth profile), which is what the storage claim
rests on.

**`outputs/diag_ftcheck/`** -- the slab-truncation equivalence check
(`fm3` vs `fm3_ft9` vs `ft2` vs `ft1`, rot-ellipse 8,16,8). `ft9` == `fm3` or
the `ft` arms in the h-sweep mean nothing.

**`outputs/diag_m95/`** -- mode percentile under refinement (rot-ellipse
6,12,6 / 8,16,8 / 10,20,10, `ibpd_r0` and `ibpd_r0_fm3`). The probe now reports
`m95`/`n95` -- the smallest cutoff containing 95% of a mode's energy, i.e. the
coarse cutoff that mode DEMANDS -- alongside the mean, which was too blunt
(§11: count grew with `n_t n_z` while mean `|m|` did not move). If `m95` stays
~3 while the count grows, a FIXED cutoff works and `fm` scales; if `m95` grows
like `n_t`, it does not.

**`outputs/diag_fmwide/`** -- mode sweep `fm3..fm6` on the two cases where
`fm3` trails `o1` (W7-X k=1, rot-ell k=2). Does a wider box close the gap, as
the face-weight bandwidth (99% at `m_max=3` but 99.9% at 26 on W7-X) suggests?

**`outputs/diag_xterm2/`** -- THE CROSS TERM, taken exactly. 3 jobs: all three
geometries, `ns` = 12,24,12, p=3, **k=1,2,3 and free AND dbc**, maxiter 40000.
Arms `bj_r0_ibpd, bj_r0_ibpr, bj_r0_ibpf, bj_r0_ibpd_fm3, bj_r0_ibpf_fm3,
bj_r0_ibpd_o1`. `ibpd -> ibpr` tests the COMPUTED `rho`; `ibpr -> ibpf` tests
whether the 21-28% that lies off `e e^T` is real signal. dbc rows must be
INVARIANT (the term vanishes identically under an essential condition) -- that
is the wiring check.

**`outputs/diag_xterm2spec/`** -- 2 jobs (toroid, rot-ellipse, 6,12,6,
k=1,2,3 free, `ibpd_r0 / ibpr_r0 / ibpf_r0`). **Read this BEFORE the iteration
counts**: the cross term is NEGATIVE, so `ibpf` can make `K_r` indefinite;
`min eig(P)` in this probe shows it directly, and a floored CG would otherwise
look like "the correction is bad" when it is really "the atom stopped being
SPD".

**`outputs/diag_alphaverify/`** -- 9 jobs, `bc_alpha_compare.py` on 3
geometries x p = 2,3,5 at 12,24,12. Prints `alpha` per `(k,c)` against the
exact round-trip `c_star`. Tests two predictions at degrees they were never
checked at: does `mu_0` stay METRIC-FREE, and does `alpha` stay
DEGREE-INDEPENDENT, at p != 3? A failure is diagnostic.

**`outputs/diag_fmr0/`** -- `fm` isolated from the axis treatment, n=12.
**`outputs/diag_fmcost/`** -- `fm_cost_n16`, `fm_cost_n20` (the storage
crossover), `fm_lv_decay` and `fm_lv_p{2,3,5}` (the `|LV|^2`-by-depth profile
and how it moves with p).

### TOMORROW: WHAT TO SUMMARIZE, in order -- **DONE, see §12**

All 33 jobs completed and are read out in **§12**, in this order. Both
gates passed as tests (§12.1 pass, §12.2 fired -- `ibpr` indefinite).
The verdict on the whole boundary-term programme is **§12.9**.

Kept below for the reasoning behind each item:

1. **`diag_ftcheck`** -- gate. If `ft9 != fm3`, every `ft` number is void and
   the h-sweep must be re-read using the untruncated `fm3` column only.
2. **`diag_xterm2spec`** -- gate. If `min eig(P) < 0` for `ibpf`, its iteration
   counts measure indefiniteness, not the cross term.
3. **`diag_xterm2`** -- is the cross term accountable? Table `ibpd / ibpr /
   ibpf` x (k=1,2,3) x 3 geometries. Compare k=3 against the fine-sweep
   optimum (0.55-0.70 shaped, 1.00 toroid) and computed `rho` (0.63 toroid,
   0.40 W7-X). Check the dbc invariance.
4. **`diag_alphaverify`** -- do `mu_0` and degree-independence survive p != 3?
5. **`diag_hsweep`** -- the scaling verdict. Plot iterations vs `n` for
   `jacobi / exact / ibpd / fm3 / fm3_ft2 / o1 / o2` on both geometries. THE
   question: does the `fm3`-vs-`o1` gap (562 vs 814 at n=12, W7-X) CLOSE with
   refinement? If not, `fm` has no case and route (2) is next.
6. **`diag_m95`** -- does the demanded cutoff `m95` stay ~3 while the outlier
   count grows? That is the mechanism behind whatever (5) shows.
7. **`diag_psweep2` + `fm_lv_p*`** -- p-dependence of the arms and of the `LV`
   slab depth.
8. **`diag_fmwide`, `diag_fmr0`, `diag_fmcost`** -- the mode-box knee, `fm`
   without the axis treatment, and the storage crossover at n=16/20.

Then update §11's tables and decide between: land `ibpr`/`ibpf` (if the cross
term is accountable), pursue `fm` (if it scales), or drop to route (2), the
banded capacitance.

### Older runs -- COMPLETE and already folded into the tables above

Nothing to re-read here unless a number is being checked:
`diag_pin`, `diag_pindiag`, `diag_pino`, `diag_pinspec`, `diag_alphapen`,
`diag_spec2`, `diag_spec3`, `diag_spech`, `diag_edge`, `diag_tmequiv`,
`diag_dspline`, `diag_fmspec` (ADDITIVE, superseded), `diag_fmhspec`,
`diag_fmh`.

### ARM GRAMMAR (current -- several keys were added or fixed 2026-08-21)

Arm grammar (fixed today; the ring regexes are underscore-anchored so that
`pino1` is not read as `outer_rings=1`):
`pin[o|a][d]N` -- N rings; `o` = non-trace components, `a` = all, default
trace; `d` = evicted rows get the Jacobi diagonal instead of a probe column.

* `_rN` / `_oN` -- `extra_rings` (inner) / `outer_rings`. UNDERSCORE-ANCHORED,
  so `pino1` is not misread as `outer_rings=1` (it was, and it silently
  confounded a comparison).
* `fmM` -- truncated-Fourier coarse correction, `|m|,|n| <= M`, 1 ring;
  `frR` -- R rings; `fso`/`fst` -- restrict to non-trace / trace components;
  `fadd` -- ADDITIVE instead of hybrid (diagnostic only, it cannot cure a high
  outlier); `ftD` -- hold `V`/`LV` on a slab D rings deep (`ft9` must reproduce
  the untruncated arm exactly).
* `ibpr` / `ibpf` -- rank-1 projection of the EXACT 1-D correction
  `F - Ktilde` (the computed `rho`) / the FULL correction including the
  off-`e e^T` part of the cross term.
* `bcpN` (alpha x N/100), `tgN`, `tmN` (BROKEN), `dbN`, `ntN`, `a2d`, `d0sN`.

### The questions those runs were designed to answer -- ANSWERED

Kept so the same arms are not re-run:

* `pin1` vs `ibpd` -- **no**, the hard pin does not beat the derived penalty;
  it does not even move (76/76, 620/607, 1834/1874).
* `pin1` vs `o1` -- the boundary problem is NOT specific to the trace
  component. `o1` (all components) 308 against `pin1` 607 on rot-ellipse.
* `pin1_tg10` vs `pin1` -- **prediction FALSIFIED**: `tg` still helps after
  pinning (rot-ell 607 -> 339, W7-X 1874 -> 884). The degeneracy argument is
  not what drives the numbers, for the reason in §10 (the failing modes carry
  no `u_r`, so the condition was already satisfied for them).
* the dbc invariant HELD: every dbc row identical across `ibpd`, `pin1`,
  `pin2`, `pin1_tg10`, `tg10` (toroid k=1 dbc 62, k=2 dbc 77; rot-ell k=1 dbc
  125; W7-X k=1 dbc 177). `o1` is the exception and perturbs them (98, 169,
  157, 229) because `outer_rings` is NOT guarded on Dirichlet while
  `pin_trace` is.
* `diag_alphapen` -- **yes**, the derived term earns its place: `alpha = 0` is
  the worst arm on all three geometries with a penalty present.

Added after the overnight set (§12); do not re-run these either:

* the cross term at the computed `rho` (`ibpr`) -- **REFUTED, and indefinite**:
  `min eig(P) < 0` on every geometry at every k (§12.2), counts 5-15x worse or
  non-convergent (§12.3). The full correction `ibpf` stays SPD and is still
  3-7x worse. §4's route is closed.
* `ft` truncation -- **FREE, at depth 1**: `ft9 == ft2 == ft1 == fm3` to within
  1 iteration, `|LV|^2` = 99.1-99.9% in the outer ring alone, p-independent.
* `mu_0` metric-free and `alpha` degree-independent at **p = 2 and 5**, not
  just p=3 (§12.4). Both hold bit-identically.
* does the `fm3`-vs-`o1` gap close under refinement? -- **NO, it inverts on
  W7-X** (412 vs 568 at n=8; 2033 vs 1742 at n=20). `fm` inherits `ibpd`'s
  exponent because it is a flat ~2.1x on top of it (§12.5).
* does the demanded cutoff `m95` stay fixed? -- **NO, `m95` ~ n_t/3** (§12.6).
  The fixed box does not scale; that is the mechanism behind the exponent.
* is `ibpd` safe as a default at k=1/2 free? -- **NO.** It loses to jacobi on
  W7-X for n >= 12 and to `exact` at every n (§12.5). Guard it to k=3 and the
  toroid before anything lands.

---

## 12. OVERNIGHT RESULTS — all 33 jobs, read out after the ~18:00 launch

All 33 completed, none failed. (The three `a2dfm` / `fm_cost_a2d` jobs cancelled
at 17:43 are pre-overnight casualties and are not part of this set.)

Read in the order §11 asked for. The two gates are §12.1 and §12.2; §12.9 is the
verdict on the whole boundary-term programme and is the part that changes what
happens next.

### 12.1 GATE — `diag_ftcheck`: PASS, and better than needed

rot-ellipse 8,16,8, k=1 free:

| `fm3` | `fm3_ft9` | `fm3_ft2` | `fm3_ft1` |
| --- | --- | --- | --- |
| 226 it, 14.7s build | 224 it, 4.4s | 226 it, 4.4s | 227 it, 4.5s |

`ft9` reproduces `fm3`, so every `ft` number below is valid. The bonus: **`ft1`
is also within noise**, at 3.3x cheaper build. `diag_fmcost/fm_lv_decay` gives
the mechanism: `|LV|^2` is 99.1-99.9% at depth 0, <= 0.9% at d1, exactly 0
beyond d2 -- and it is **p-INDEPENDENT** (0.999/0.000 at p=2, 0.991/0.009 at
p=5). The slab does not widen with the stencil. §11's storage claim holds.

### 12.2 GATE — `diag_xterm2spec`: `ibpr` is INDEFINITE, its counts are void

`min eig(P)` for `ibpr`, all negative: toroid -1.36e-1 / -6.07e-2 / -1.23e-3
(k=1/2/3), rot-ellipse -6.59e-1 / -3.79e-2 / -6.96e-4. `ibpd` and `ibpf` stay
SPD throughout. The gate fired exactly as designed -- the cross term is
negative, and applied at the computed `rho` it takes `K_r` out of SPD.

`ibpf` survives SPD but is spectrally WORSE than `ibpd`: cond 3048 vs 115
(toroid k=1), 152 vs 16 (toroid k=3), with 90-102 high outliers where `ibpd`
has 0-2.

### 12.3 `diag_xterm2` — THE CROSS TERM IS REFUTED

n=12, p=3, `extra_rings=0`, iterations (`*` = hit maxiter 40000):

| geom | k, bc | `ibpd` | `ibpr` | `ibpf` | `ibpd_fm3` | `ibpd_o1` |
| --- | --- | --- | --- | --- | --- | --- |
| toroid | 1 free | **96** | 925 | 520 | 95 | 291 |
| toroid | 2 free | **92** | 1281 | 502 | 93 | 192 |
| toroid | 3 free | 71 | 517 | 358 | 71 | **59** |
| rot-ell | 1 free | 671 | 9258 | 1313 | **374** | 512 |
| rot-ell | 2 free | 737 | 14089 | 1395 | **439** | 465 |
| rot-ell | 3 free | 181 | 2134 | 614 | 173 | **123** |
| w7x | 1 free | 2009 | 40000* | 2733 | **900** | 994 |
| w7x | 2 free | 2027 | 40000* | 3398 | 1091 | **953** |
| w7x | 3 free | 175 | 1937 | 600 | 172 | **130** |

`ibpd` as shipped beats both corrections at every degree on every geometry. The
21-28% lying off `e e^T` is not usable signal: `ibpf` is 3-7x worse at k=1/2 and
3-4x at k=3.

**The wiring check PASSED**: every dbc row is identical across `ibpd`/`ibpr`/
`ibpf` (216/217/217, 236/236/236, 319/321/319, 141/140/140, ...), confirming
the term vanishes identically under an essential condition. `o1` perturbs them,
as §11 already recorded.

Consequence: `rho` is settled as NOT implementable this way. §4 and §11.ACCOUNTING
are closed. The positive scalar knob `bcpN` remains the only route to the
0.55-0.70 shaped-geometry optimum the fine sweep found.

### 12.4 `diag_alphaverify` — both predictions HOLD at p != 3, and a new defect

* `mu_0` is **bit-identical across all three geometries** at each p: 66.73954
  (p=2), 93.88387 (p=3), 134.4128 (p=5). Metric-free, confirmed off p=3.
* `a_ibpd` is **identical down all four `(k,c)` rows** for every geometry at
  every p. Degree-independence, confirmed off p=3.

Both §11.CONFIDENCE claims survive. But the same tables show something that was
not being looked at -- the spread of `a_exact / a_ibpd` ACROSS PARTNER ROWS:

| geom | p=3, four `(k,c)` rows | spread |
| --- | --- | --- |
| toroid | 0.70, 0.70, 0.67, 0.67 | **1.04x** |
| rot-ellipse | 3.45, 3.22, 1.74, 1.62 | **2.13x** |
| w7x | 0.265, 0.239, 0.092, 0.084 | **3.15x** |

`a_ibpd` is `(k,c)`-CONSTANT by construction (that is the degree-independence
result). The exact round trip is not. So on shaped geometries the true
coefficient wants 2-3 DIFFERENT numbers where the formula can only supply one.
That is a far larger modelling error than the 5-9% `<S><P>` vs `<SP>`
convention §2 flags, and it is not a factor that can be corrected -- see §12.9.

`off_frac` also grows with p (toroid 0.136 / 0.214 / 0.244 at p=2/3/5;
rot-ell 0.206 / 0.259 / 0.286), so the cross term becomes MORE real at high p
even though correcting for it fails (§12.3).

### 12.5 `diag_hsweep` — THE SCALING VERDICT

k=1 free, p=3, `extra_rings=0`, tol 1e-10. Iterations; build seconds in the
n=20 row.

**rot-ellipse**

| n | jacobi | `exact` | `ibpd` | `fm3` | `fm3_ft2` | `o1` | `o2` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 523 | 678 | 392 | 226 | 226 | 309 | 205 |
| 12 | 740 | 1170 | 677 | 374 | 374 | 512 | 361 |
| 16 | 967 | 1647 | 966 | 517 | 518 | 708 | 501 |
| 20 | 1269 (7.8s) | 2201 | 1224 (2.7s) | 640 (6.4s) | 643 | 937 (60.2s) | 620 (118.9s) |

**W7-X**

| n | jacobi | `exact` | `ibpd` | `fm3` | `fm3_ft2` | `o1` | `o2` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 1060 | 652 | 946 | 412 | 412 | 568 | 374 |
| 12 | 1641 | 1268 | 1963 | 900 | 904 | 988 | 678 |
| 16 | 2265 | 1878 | 3116 | 1507 | 1505 | 1397 | 976 |
| 20 | 3023 (7.9s) | 2465 | 4150 (2.6s) | 2033 (6.4s) | 2023 | 1742 (60.5s) | 1222 (120.4s) |

Fitted exponents over n = 8 -> 20:

| | jacobi | `exact` | `ibpd` | `fm3` | `o1` | `o2` |
| --- | --- | --- | --- | --- | --- | --- |
| rot-ellipse | n^0.97 | n^1.29 | n^1.24 | n^1.14 | n^1.21 | n^1.21 |
| w7x | n^1.14 | n^1.45 | **n^1.62** | **n^1.74** | n^1.23 | n^1.29 |

Three findings, the first being the important one:

1. **`ibpd` CROSSES jacobi on W7-X and keeps going.** 946 < 1060 at n=8, but
   4150 > 3023 at n=20. The derived coefficient is actively HARMFUL on W7-X
   above n ~ 10. And `exact` -- the provably 8-14x too small one -- beats
   `ibpd` at EVERY n on W7-X (652/1268/1878/2465), with a better exponent. On
   rot-ellipse the ordering is reversed at every n. §6's "the derived
   coefficient is HARMFUL at k=1/2" is now h-resolved, and it gets worse, not
   better, under refinement.
2. **`fm` does not change the exponent; it divides by a constant.** `ibpd/fm3`
   on W7-X = 2.30, 2.18, 2.07, 2.04 -- a flat ~2.1x. `fm3` therefore inherits
   `ibpd`'s n^1.62 and lands at the worst exponent in the table. The ring arms
   sit at n^1.21-1.29, tracking jacobi. **The `fm3`-vs-`o1` gap does NOT close
   on W7-X, it INVERTS**: 412 vs 568 at n=8, 2033 vs 1742 at n=20. On
   rot-ellipse `fm3` stays ahead at every n (ratio a stable 0.68-0.73).
3. **`ft2` is free at every resolution**, confirming §12.1 across the sweep.

On wall-clock the picture is friendlier to `fm` than iterations alone suggest:
at n=20 rot-ellipse `o2` buys 3% fewer iterations for a 19x more expensive
build (118.9s vs 6.4s); on W7-X n=20 `o1`'s 14% iteration win costs 60.5s
against 6.4s. **A per-iteration timing was not measured and is what would
settle it.**

### 12.6 `diag_m95` — the mechanism, and it kills the FIXED box

rot-ellipse k=1 free, `ibpd` baseline, high-outlier statistics:

| n_r, n_t | cond | outliers | `m95` | `m95`/n_t | `n95` |
| --- | --- | --- | --- | --- | --- |
| 6, 12 | 1224 | 55 | 4.11 | 0.34 | 2.79 |
| 8, 16 | 3126 | 137 | 5.23 | 0.33 | 3.59 |
| 10, 20 | 6127 | 239 | 6.81 | 0.34 | 4.43 |

**`m95` grows PROPORTIONALLY to `n_t` -- a constant ~1/3 fraction, not a fixed
number.** That is the branch §11 named as "it does not scale", and it is the
mechanism behind §12.5's exponent. A fixed cutoff captures a shrinking share of
what the modes demand.

`fm3` does still cut cond 4-7x at every n (319/527/881 against 1224/3126/6127),
and `diag_fmwide` says the same thing from the other side -- more modes keep
paying at fixed n: W7-X n=12 k=1, `fm3` 896 -> `fm4` 731 -> `fm5` 650 ->
`fm6` 622, `fm5_fr2` 586, against `o1` 994. So a GROWN box beats `o1` by 1.6x;
a fixed one does not. `fm` with a box growing like `n` is still open, but it is
then no longer a fixed-cost method.

### 12.7 `diag_psweep2` — the one unambiguously good result

k=1 free, n=12, `extra_rings=0`:

| geom | p | jacobi | `ibpd` | `fm3` | `fm3_fr2` | `o1` | `o2` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rot-ell | 2 | 544 | 468 | 255 | 239 | 342 | 268 |
| rot-ell | 3 | 742 | 674 | 374 | 369 | 511 | 360 |
| rot-ell | 4 | 1882 | 882 | 524 | 519 | 676 | 450 |
| rot-ell | 5 | 6640 | 1091 | 709 | 704 | 865 | 577 |
| w7x | 2 | 1328 | 1324 | 664 | 593 | 703 | 536 |
| w7x | 3 | 1649 | 1973 | 897 | 831 | 993 | 680 |
| w7x | 4 | 3364 | 2629 | 1180 | 1125 | 1350 | 874 |
| w7x | 5 | 10095 | 3194 | 1508 | 1432 | 1709 | 1087 |

**Jacobi degrades catastrophically with p** (12.2x on rot-ellipse, 7.6x on W7-X
across p=2->5) while every block arm grows only 2.3-2.8x. At p=5 W7-X `fm3` is
6.7x jacobi and `o2` is 9.3x. The arm ORDERING (`o2` < `fm3` < `o1` < `ibpd` <
jacobi) is p-independent, `fr2` gives a small consistent extra few percent, and
the `LV` slab depth does not move with p (§12.1). The p story is clean and
favourable; it is where the block route's case is strongest.

### 12.8 `diag_fmcost` — the storage crossover

rot-ellipse k=1, MB total (build seconds):

| n | `ibpd` | `fm3` | `o1` | `o2` |
| --- | --- | --- | --- | --- |
| 12 | 0.0 (24.0s) | 20.7 (4.7s) | 6.8 (21.9s) | -- |
| 16 | 0.1 (23.7s) | 51.0 (5.4s) | 20.9 (37.8s) | 79.5 (74.9s) |
| 20 | 0.1 (26.6s) | 102.2 (6.3s) | 50.1 (59.6s) | -- |

`fm`'s memory is LINEAR in `n_dof` (20.7/51.0/102.2 against n_dof
8700/21584/43300 -- ratios match to three digits), so it is not fixed-cost in
memory despite fixed `q`=147: it is `V` at `n x q`. `o1` is smaller in absolute
MB at every n but grows faster and costs 6-10x more to build.

**MISSING MEASUREMENT**: the `ft` truncation should collapse `fm`'s storage to
`O(n_t n_z q)` and §12.1 shows it costs nothing in iterations -- but memory
under `ft` was never reported. That single number decides whether `fm` is
cheap. Add `ft` arms to the `fm_cost` table.

### 12.9 VERDICT: is the boundary correction a WRONG NUMBER or a WRONG OBJECT?

Asked by Tobias. The answer the data supports is **wrong object, and the number
is right** -- with one h-scaling loophole that is cheap to close (below).

**The number passes every independent check available.** `mu_0` metric-free
across three geometries and p=2,3,5 (bit-identical); degree-independence across
all four `(k,c)` rows and all three p; a clean interior CG minimum at k=3, the
one degree where the operator IS purely the weak block. Those are strong checks
of a kind that would break under a stray metric factor -- which is exactly how
the two errors in §3 were caught. Nothing in them has moved.

**The mechanism is exactly right where its own assumption holds.** The scalar
reduction assumes the face weight is constant over the face. On the toroid it
IS (`<S><P>/<SP>` = 1.0000, `g^rr` constant there), and there the derived term
is the best arm at every k and every h -- k=1 free 442 -> 76, and it beats
every ring and mode arm in §12.3. This is the decisive asymmetry: **an
arithmetic slip would be wrong on the toroid too.** It is not.

**And it fails where the assumption fails, in four ways no coefficient can
repair:**

1. *The true coefficient is not a coefficient.* §12.4: `a_exact` varies 2.13x
   (rot-ellipse) and 3.15x (W7-X) across the four `(k,c)` rows, while any
   `alpha = mu_0 <S> <P>` is constant across them by construction. One number
   cannot satisfy four rows that want four different numbers. On the toroid the
   spread is 1.04x -- and that is the case where the term works.
2. *Every enrichment of the term has failed, in both directions.* Restoring the
   dropped angular structure (`wibp`, exact 2-D face by quadrature) -- within
   noise, worse on W7-X. Raising the rank (`ibpf`) -- 3-7x worse (§12.3).
   Taking the penalty to its hard limit (`pin`) -- a no-op on the high
   outliers (§10). Adding the consistent cross-component term (Nitsche) --
   diverges. A wrong NUMBER is repaired by a better number; none of these are
   number changes, and all of them fail.
3. *The defect has an h-exponent.* §12.5: `ibpd` is n^1.62 on W7-X where doing
   nothing at the boundary is n^1.14. Adding the term makes the SCALING worse,
   not just the constant. And `exact`, which is provably 8-14x too small, has a
   better exponent (n^1.45) than the correct value.
4. *The object is a layer, not a face.* `o1/o2/o3` = 561/391/349 on W7-X k=1
   free: gains keep coming at the second and third ring. A boundary CONDITION
   lives on the face; one exact ring would capture any of them, with any
   coefficient. §11 said this already; §12.5 adds that the ring arms are also
   the only ones that keep jacobi's exponent.

**The loophole, and how to close it.** `alpha` is itself h-dependent (`mu_0` =
52.26 at n=8, 93.88 at n=12 -- the `O(1/h)` unit-integral normalisation), and
the derivation fixes that h-dependence. If the derivation has the wrong POWER
of h, that would be a numerical-factor problem and it would produce exactly the
exponent defect in (3). Point (4) and the partner spread in (1) would still be
unexplained -- they are h-independent arguments -- so this cannot be the whole
story, but it could be part of it.

> **The experiment**: sweep `bcpN` at each n on the `diag_hsweep` grid
> (rot-ellipse and W7-X, n = 8,12,16,20, k=1 free, `extra_rings=0`) and read
> off the optimal scale as a function of n. If `bcp_opt(n)` is FLAT, `alpha`'s
> h-scaling is right and the residual is entirely mechanism. If it drifts like
> a power of n, the derivation is missing an h-factor and that power names it.
> Cheap -- same harness as `diag_alphapen`, ~8 jobs.

**What follows either way.**

* The natural-BC term is FINISHED as a piece of derivation. Keep it where it is
  demonstrated: k=3 free (the §6 deliverable), and the toroid at all k.
* **It must NOT be on by default at k=1/2 free on shaped geometries.** §12.5
  shows it losing to jacobi on W7-X for n >= 12 and to `exact` everywhere. That
  is a guard to add before anything lands, and it is the one actionable item
  from tonight.
* The k=1/2 free gap is a DIFFERENT object -- a boundary layer, i.e. a DtN /
  Steklov operator with radial extent -- and only region treatments have ever
  touched it. §8's banded capacitance is now the strongest remaining candidate,
  because §12.5 shows the ring arms are the only family whose exponent tracks
  jacobi, and their sole defect (60-120s dense builds at n=20) is precisely
  what banding attacks.

### 12.10 Where that leaves the three options §11 named

| route | verdict |
| --- | --- |
| land `ibpr`/`ibpf` | **DEAD.** Indefinite (`ibpr`) or worse than what it corrects (`ibpf`), everywhere. |
| pursue `fm` | **HALF DEAD.** A real, robust ~2.1x on top of `ibpd`, cheap to build, truncates to one ring for free, improves with p. But `m95` ~ n_t/3 means the FIXED box does not scale, and `o1` overtakes it on W7-X by n=16. A box grown like `n` still works (`fmwide`) but is no longer fixed-cost. |
| route (2), banded capacitance | **NOW THE LEAD.** The only family that keeps jacobi's exponent; banding attacks its only weakness. |

New problem not on the old list, and it propagates: `ibpd` loses to jacobi on
W7-X under refinement, and `fm` sits on top of `ibpd`, so the best cheap arm
inherits the baseline's defect. See §12.9.

Data: `outputs/diag_{ftcheck,xterm2,xterm2spec,alphaverify,hsweep,m95,psweep2,
fmwide,fmr0,fmcost}/` (all 2026-08-21).

### 12.11 THE MISSING CONTROL — `nobc`, and a correction to §12.9's argument (3)

Asked by Tobias: how does the atom do with NO boundary correction at all?

**This was never the control being used, and it should have been.** §12.5
compares `ibpd` against `jacobi` -- but `jacobi` is POINT Jacobi, a different
preconditioner entirely. The correct control for "does the boundary term help"
is `nobc`: the SAME block atom, same rings, `alpha = 0`. It exists at
`extra_rings=3`, n=8 and 12 only (`outputs/diag_bcibp/` + `outputs/diag_ibpd/`).

Free BC, p=3, `extra_rings=3`:

| geom | n | k | jacobi | **`nobc`** | `direct` | `exact` | `ibpd` | **`ibpd`/`nobc`** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toroid | 8 | 1 | 317 | 168 | 78 | 57 | 58 | **0.35** |
| toroid | 12 | 1 | 442 | 296 | 117 | 76 | 77 | **0.26** |
| toroid | 8 | 2 | 214 | 165 | 137 | 45 | 46 | **0.28** |
| toroid | 12 | 2 | 345 | 292 | 240 | 62 | 62 | **0.21** |
| toroid | 8 | 3 | 113 | 98 | 97 | 29 | 28 | **0.29** |
| toroid | 12 | 3 | 188 | 186 | 183 | 36 | 34 | **0.18** |
| rot-ell | 8 | 3 | 144 | 152 | 130 | 67 | 58 | **0.38** |
| rot-ell | 12 | 3 | 236 | 249 | 208 | 88 | 75 | **0.30** |
| w7x | 8 | 3 | 173 | 165 | 163 | 96 | 58 | **0.35** |
| w7x | 12 | 3 | 306 | 275 | 274 | 123 | 75 | **0.27** |
| rot-ell | 8 | 1 | 524 | 391 | 280 | 601 | 360 | **0.92** |
| rot-ell | 12 | 1 | 744 | 636 | 426 | 1056 | 622 | **0.98** |
| rot-ell | 8 | 2 | 407 | 405 | 281 | 565 | 349 | **0.86** |
| rot-ell | 12 | 2 | 621 | 689 | 466 | 1063 | 652 | **0.95** |
| w7x | 8 | 1 | 1060 | 695 | 575 | 602 | 840 | **1.21** |
| w7x | 12 | 1 | 1635 | 1321 | 1066 | 1194 | 1795 | **1.36** |
| w7x | 8 | 2 | 888 | 746 | 703 | 576 | 781 | **1.05** |
| w7x | 12 | 2 | 1552 | 1498 | 1393 | 1217 | 1793 | **1.20** |

Read against the RIGHT baseline the picture is much sharper than §12.5's:

* **Where the scalar-face assumption holds the term is a 2.6-5.5x win over no
  term**, and it IMPROVES with refinement: toroid all k (0.35->0.26, 0.28->0.21,
  0.29->0.18) and k=3 on every geometry (rot-ell 0.38->0.30, w7x 0.35->0.27).
  Eight rows, all falling.
* **Where it fails the term does NOTHING (rot-ellipse, 0.86-0.98) or actively
  HARMS (W7-X, 1.05-1.36)**, and it DEGRADES with refinement: 0.92->0.98,
  0.86->0.95, 1.21->1.36, 1.05->1.20. Four rows, all rising.

**CORRECTION to §12.9 argument (3).** I stated the exponent defect against
`jacobi`, which is not the control, and flagged internally that n^1.62 might
belong to the block atom rather than to the boundary term. It does not: the
`ibpd`/`nobc` RATIO itself moves with n, in opposite directions on the two
sides of the split. Argument (3) stands, and it now rests on the same-atom
comparison rather than on a cross-preconditioner one. Two resolutions only;
`diag_bcpn` extends it to four at `extra_rings=0`.

**PRODUCTION CONCERN, wider than `ibpd`.** `exact` is the SHIPPED DEFAULT and
it is 1.66x WORSE than no boundary term at all on rot-ellipse k=1 free (1056 vs
636) and 1.54x at k=2 (1063 vs 689). §12.9's guard is not just about landing
`ibpd` -- the default in the tree today is harmful at k=1/2 free on rot-ellipse.
`direct` (the pre-2026-08-19 weight, ~35-70x below the true value) is the best
of the boundary arms at k=1/2 on every shaped row, which is the same "the atom
wants an alpha an order of magnitude under the operator's" signal from §6, now
measured against `alpha = 0`.

### 12.12 LAUNCHED: `diag_bcpn` — the bcp-vs-n sweep (§12.9's experiment)

12 jobs, submitted 2026-08-21 23:43 (`16434318`-`16434332`). 3 geometries x
n = 8,12,16,20, p=3, `extra_rings=0`, free BC, tol 1e-10, maxiter 40000 --
config IDENTICAL to `diag_hsweep` so the rows drop straight into §12.5's table.
`ks` = 1,2,3 at n <= 12 and 1,3 at n = 16,20 (k=2 tracks k=1 throughout; k=3 is
the clean control since `S_3 = 0` makes `L_3` purely the weak block).

Arms: `jacobi, bj_r0_nobc, bj_r0_ibpd_bcp0, bj_r0_ibpd_bcp15,
bj_r0_ibpd_bcp30, bj_r0_ibpd_bcp55, bj_r0_ibpd, bj_r0_ibpd_bcp200`.

It answers three things at once:

1. **`bcp_opt(n)`** -- FLAT means `alpha`'s h-dependence is right and the
   k=1/2 residual is entirely mechanism; DRIFTING like a power of n names the
   h-factor the derivation is missing. Three geometries separate the h axis
   from the geometry axis (§11: the optimum is ~0.3 shaped, >= 1.0 toroid), and
   if it is BOTH then a rescaling is still a per-geometry fit and still not
   landable.
2. **`nobc` at four resolutions and `extra_rings=0`** -- the control §12.11
   only has at two, and the one that decides whether §12.9(3) is the term's
   defect or the atom's.
3. **Does `bcp_opt -> 0` at k=1/2 on shaped geometries?** §6's scale sweep was
   monotone falling and still falling at its lowest point (0.4); this one goes
   to 0 at four resolutions. If the optimum is a BOUNDARY point at zero rather
   than a drifting interior one, the k=1/2 prescription is simply "switch the
   term off", which is cleaner and more landable than any rescaling.

**Wiring check built in**: `bcp0` must reproduce `nobc` exactly.

**WHAT THIS SWEEP CANNOT DO**, so it is not over-read: it cannot rescue the
boundary term as the k=1/2 MECHANISM. §12.9's arguments (1) -- the true
coefficient varies 2-3x across the `(k,c)` rows at FIXED h, where a global
h-power multiplies all rows equally -- and (4) -- rings 2 and 3 keep paying at
fixed h -- are h-independent, as is the mode content (the failing k=1 modes
carry ZERO `u_r` and live on components with NO boundary term at all, so no
scaling of a term that is not on them can reach them).

**What it CAN change, and this is the part of §12 genuinely at risk: the `fm`
verdict.** `fm3` is a flat ~2.1x multiple of `ibpd` at every n on W7-X
(2.30/2.18/2.07/2.04), so it inherits `ibpd`'s exponent BY CONSTRUCTION. If a
rescaled `alpha` repairs `ibpd`'s exponent, `fm3`'s repairs with it and §12.5's
"`o1` overtakes `fm3` at n=16" could reverse. `m95 ~ n_t/3` (§12.6) is measured
independently and still says the fixed box captures a shrinking share -- but
the ITERATION evidence for that conclusion was inherited from the baseline, not
intrinsic to `fm`. §12.5 and §12.10 should be read with that caveat until
`diag_bcpn` lands.

---

## 13. `diag_bcpn` RESULTS — the h-loophole is CLOSED, and §12.9 was PREMATURE

All 12 jobs COMPLETE. p=3, `extra_rings=0`, free BC, tol 1e-10.

### 13.1 WIRING CHECK — passes, and it measures a noise floor

`bcp0` must reproduce `nobc` (both are `alpha = 0`). Over 29 cells: mean
**+0.13%**, max |delta| **2.4%**, signs mixed (12 up, 8 down, 9 exact). Same
operator. The RHS is deterministic (`PRNGKey(31 k + dbc)`, drawn OUTSIDE the arm
loop), so the spread is run-to-run numerical noise, and it is concentrated where
it should be: k=3 rows are 0 to +2 (<= 0.6%), k=1 rows -26 to +11. **k=1 free is
the singular case with the harmonic projection reapplied each iteration** -- that
is where the nondeterminism lives.

> **NOISE FLOOR ~1%, up to 2.4% at k=1 free. Differences below ~2% anywhere in
> §12 are not meaningful.** This retro-validates the "within noise" readings
> (`ft9` 224 vs `fm3` 226; `fm3` 374 vs `ft2` 374; `ibpd` 676 vs `nobc` 678) and
> should be applied to every table above.

### 13.2 THE ANSWER: `bcp_opt` DOES NOT DRIFT WITH n

k=1 free, iterations by penalty scale:

| geom | n | jacobi | `nobc` | 0.15 | 0.30 | 0.55 | **1.00** | 2.00 | argmin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toroid | 8 | 318 | 197 | 83 | 77 | 77 | 77 | 87 | plateau |
| toroid | 12 | 440 | 310 | 100 | 99 | 96 | 96 | 104 | plateau |
| toroid | 16 | 600 | 391 | 113 | 113 | 113 | 113 | 118 | plateau |
| toroid | 20 | 774 | 476 | 128 | 127 | 127 | 127 | 132 | plateau |
| rot-ell | 8 | 527 | 446 | **282** | 291 | 321 | 403 | 521 | 0.15 |
| rot-ell | 12 | 740 | 678 | **423** | 463 | 537 | 676 | 896 | 0.15 |
| rot-ell | 16 | 968 | 921 | **554** | 625 | 747 | 947 | 1304 | 0.15 |
| rot-ell | 20 | 1274 | 1103 | **674** | 776 | 962 | 1210 | 1643 | 0.15 |
| w7x | 8 | 1056 | 835 | **612** | 680 | 788 | 961 | 1261 | 0.15 |
| w7x | 12 | 1644 | 1546 | **1147** | 1278 | 1572 | 2007 | 2652 | 0.15 |
| w7x | 16 | 2288 | 2117 | **1662** | 1972 | 2386 | 3081 | 4234 | 0.15 |
| w7x | 20 | 3016 | 2657 | **2074** | 2514 | 3193 | 4096 | 5688 | 0.15 |

k=3 free is the same shape (shaped argmin 0.15 at every n; toroid a plateau),
and k=2 free tracks k=1. **The argmin does not move with n on any geometry at
any degree.** §12.9's loophole is CLOSED: `alpha`'s h-dependence is right, and
the k=1/2 residual is not an h-scaling error.

CAVEAT, and it is not small: on the shaped geometries the argmin sits AT the
low edge of the bracket, so "flat" is established only against drift ABOVE
0.15. And the high side steepens with n -- `bcp15/bcp30` = 0.969 / 0.914 /
0.886 / **0.869** (rot-ellipse k=1, n=8..20) and 0.900 / 0.897 /
0.843 / **0.825** (W7-X k=1) -- which is what residual DOWNWARD drift would
look like. At k=3 it is flat (0.96-0.99). See §13.5.

### 13.3 THE REVISION: the optimum is ~0.15, and it is GEOMETRY-INDEPENDENT

`bcp15` against the best scale swept, the shipped scale, `nobc`, and jacobi:

| k | geom | n | 15/best | 15/**1.00** | 15/`nobc` | 15/jacobi |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | toroid | 8..20 | 1.078 / 1.042 / 1.000 / 1.008 | ~1.0 | 0.42 -> 0.27 | 0.26 -> 0.17 |
| 1 | rot-ell | 8..20 | **1.000** (all) | 0.70 / 0.63 / 0.59 / **0.56** | 0.63 / 0.62 / 0.60 / 0.61 | ~0.55 |
| 1 | w7x | 8..20 | **1.000** (all) | 0.64 / 0.57 / 0.54 / **0.51** | 0.73 / 0.74 / 0.79 / 0.78 | 0.58 / 0.70 / 0.73 / 0.69 |
| 2 | rot-ell | 8,12 | **1.000** | 0.70 / 0.61 | 0.59 / 0.57 | 0.69 / 0.72 |
| 2 | w7x | 8,12 | **1.000** | 0.67 / 0.60 | 0.66 / 0.70 | 0.68 / 0.79 |
| 3 | rot-ell | 8..20 | **1.000** (all) | 0.78 / 0.77 / 0.80 / 0.81 | ~0.52 | 0.65 -> 0.44 |
| 3 | w7x | 8..16 | **1.000** (all) | 0.84 / 0.83 / 0.82 | ~0.45 | 0.55 / 0.47 / 0.37 |
| 3 | toroid | 8..20 | 1.000 / 1.000 / 1.014 / 1.011 | 0.87 / 0.87 / 0.89 / 0.93 | 0.44 -> 0.31 | 0.42 -> 0.21 |

**A single scale of 0.15 is the best point swept on all 14 shaped rows and is
within 1.4% of best on the toroid for n >= 12** (worst case 7.8%, toroid k=1
n=8). Three geometries x three degrees x four resolutions, one constant.

That directly contradicts §11's "it needs one fitted number per geometry".
The term is not geometry-dependent -- **it is over-scaled by ~6.7x, uniformly.**

Consequences at the corrected scale:
* k=1/2 free on shaped geometries: **1.4-1.9x fewer iterations than the shipped
  scale** and 1.3-1.8x fewer than no term at all. The term is NOT inert there.
* k=3: 0.77-0.93 of the derived value everywhere -- so §6's k=3 deliverable and
  §12.9's "clean interior minimum at the derived value" are both superseded.
* The toroid is unharmed: a plateau from 0.15 to 1.00 at every n.

### 13.4 WHY §12.9 WAS PREMATURE — three of its four arguments are confounded

Every experiment behind the "wrong object" verdict was run with the term at
**~6.7x its best scale**. Re-reading them:

* **(3) the h-exponent -- REFUTED as stated.** `ibpd`/`nobc` at scale 1.00
  RISES with n (rot-ell 0.90 / 0.98 / 1.02 / 1.09; W7-X 1.14 / 1.30 / 1.46 /
  1.54). At scale 0.15 it is FLAT on rot-ellipse (0.63 / 0.62 / 0.60 / 0.61)
  and FLAT on W7-X too once n >= 12 (0.73 / 0.74 / 0.79 / 0.78). **The
  degradation with h is a
  property of the over-scaled term, not of the term.** §12.11's "four rows all
  rising" is real but is a fact about scale 1.00.
* **(2) the failed enrichments -- CONFOUNDED.** `wibp` (correct coefficient +
  full angular face shape), `ibpf`, Nitsche and the `pin` were ALL tested at
  scale 1.00. `wibp`'s failure was read as "the shape hypothesis is dead" -- but
  a right shape carrying a 6.7x-too-large coefficient would fail exactly that
  way. These refutations do not stand at the corrected scale and must be re-run.
* **(4) rings 2 and 3 keep paying -- CONFOUNDED.** `o1/o2/o3` were measured on
  top of `ibpd` at scale 1.00. Whether the layer survives once the face term is
  correctly scaled is now an open question, not a settled one.
* **(1) the 2-3x partner-row spread -- WEAKENED, and arguably cuts the other
  way.** If the true coefficient really varied 2-3x per `(k,c)` row in a way
  that mattered, one global 0.15 would not be optimal on all three geometries
  and all three degrees. It is.

**Revised verdict: the derivation's STRUCTURE and h-SCALING are right; a
geometry-independent constant of ~1/6.7 is missing from it.** "Wrong object" is
withdrawn pending §13.5 and the re-runs. What is NOT withdrawn: the failing
k=1 modes carry zero `u_r` and live on components with no boundary term at all
(§10) -- that is a statement about mode content, independent of any scale, and
it still says a trace-component term cannot be the whole story.

### 13.5 WHAT THE BRACKET COST — every scale sweep so far has bottomed out at
its own low edge

§2's coarse sweep read the optimum at 1.00 with 0.30 as its next point down.
§6's fine sweep read 0.55-0.70 with 0.40 as its lowest. This sweep reads 0.15
with 0.15 as its lowest. **Three sweeps, three optima at or near the bracket
floor.** The bracket has been the binding constraint every time, and it was set
from §11's "~0.3 with `tg10` on" -- but `tg` and `alpha` trade against each
other, so the with-`tg` optimum was never the right bracket for the no-`tg` run.

REQUIRED FOLLOW-UP (`diag_bcplow`): the same grid with scales
**0.03, 0.06, 0.10, 0.15, 0.22, 0.30** plus `nobc`. Until it runs, §13.2's
"no drift" is established only above 0.15, and the widening `bcp15/bcp30`
ratio at k=1 is unexplained.

### 13.6 Standing items unchanged

* `exact` (the shipped default) is still 1.66x worse than no boundary term on
  rot-ellipse k=1 free (§12.11). The guard is still needed -- but the fix is now
  plausibly "ship `ibpd` at 0.15" rather than "switch the term off at k=1/2".
* The `fm` verdict (§12.5, §12.10) remains provisional for the reason in §12.12,
  and now doubly so: `fm3` rides on `ibpd` at scale 1.00, whose exponent defect
  §13.4 attributes to the scale. `fm` must be re-measured on a `bcp15` baseline
  before "`o1` overtakes `fm3` at n=16" can be believed.

Data: `outputs/diag_bcpn/` (2026-08-22).

### 13.7 LAUNCHED 2026-08-22 — the two follow-ups

**`outputs/diag_bcplow/`** (12 jobs, `16434536`-`16434548`). The true floor.
3 geometries x n = 8,12,16,20, k=1,2,3 at n<=12 and k=1,3 above, free BC,
p=3, `extra_rings=0`, maxiter 40000. Arms: `nobc` + scales
**0.03, 0.06, 0.10, 0.15, 0.22, 0.30**. Settles §13.5 -- whether the argmin is
a genuine interior point at ~0.15 or is still running downhill, and whether it
drifts with n once the bracket no longer pins it.

**`outputs/diag_bcp15base/`** (8 jobs, `16434549`-`16434556`). The confounded
refutations, re-run at the corrected scale. rot-ellipse and W7-X x
n = 8,12,16,20, k=1 free (plus k=2 at n<=12), same config. Arms:
`bj_r0_ibpd` (scale-1.0 anchor), `bj_r0_ibpd_bcp15`, `bj_r0_wibp_bcp15`,
`bj_r0_ibpf_bcp15`, `bj_r0_ibpd_bcp15_fm3`, `bj_r0_ibpd_bcp15_o1`,
`bj_r0_ibpd_bcp15_o2`. Pairs directly against `outputs/diag_hsweep/`, which is
the same geometries/ns/config at scale 1.0.

What each arm decides:

| arm | the claim it re-tests | why it was confounded |
| --- | --- | --- |
| `wibp_bcp15` | the FACE SHAPE hypothesis (§9, memory) | "exact 2-D face does not rescue k=1/2" was measured with a 6.7x-too-large coefficient on top of the correct shape |
| `ibpf_bcp15` | the off-`e e^T` cross term (§12.3) | same; `ibpf` was 3-7x worse at scale 1.0 |
| `o1/o2_bcp15` | the LAYER (§12.9 argument 4) | ring gains were measured on top of scale-1.0 `ibpd` |
| `fm3_bcp15` | **"`o1` overtakes `fm3` at n=16"** (§12.5) | `fm3` is a flat ~2.1x multiple of `ibpd`, so it inherits whatever exponent the baseline has |

`fm3` is the one that matters most: §12.10's whole "`fm` is half dead" verdict
rests on an exponent that §13.4 attributes to the scale, not to `fm`.

NOT re-run, and why: the `pin` (it forces `bc_entry` OFF on the pinned
components, so the scale does not reach them -- `pin1` == `nobc_pin1` there) and
Nitsche (it diverged, a different failure mode from being over-scaled). Revisit
only if §13.7 shifts the picture again.

---

## 14. WHAT 0.15 IS — a kappa-BALANCE point, not a missing factor (2026-08-22)

`diag_bcpspec` COMPLETE (2/2). `diag_bcplow` 6/12, `diag_bcpp` 1/8,
`diag_bcp15base` 2/8 -- partial results below are final for the cells shown.

### 14.1 THE MECHANISM — confirmed, and quantitatively

`cond(PL)` vs penalty scale (6,12,6, p=3, free):

| scale | 0 | 0.06 | 0.10 | 0.15 | 0.22 | 0.30 | 0.55 | 1.00 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rot-ell k=1 | 3118 | 626 | 596 | 581 | **577** | 605 | 789 | 1224 |
| w7x k=1 | 8464 | **2634** | 2698 | 2842 | 3100 | 3433 | 4729 | 7303 |
| rot-ell k=3 | 443 | 65 | 47 | **40.5** | 42.5 | 45.3 | 56.0 | 79.8 |
| w7x k=3 | 450 | 82 | 60 | 46.5 | **42.0** | 43.4 | 49.6 | 65.9 |

**Every case has a clean interior minimum in 0.06-0.22**, and the outlier
populations show exactly the predicted trade (rot-ell k=1):

| scale | 0 | 0.06 | 0.10 | 0.15 | 0.22 | 0.30 | 0.55 | 1.00 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HIGH outliers | 56 | 46 | 42 | 41 | 37 | 34 | 28 | 28 |
| LOW outliers | 8 | 9 | 9 | 11 | 11 | 13 | 19 | 27 |

High (face row too SOFT) falls monotonically; low (row over-stiffened) rises
monotonically; `cond` bottoms out where they cross. W7-X k=1 is identical in
structure (high 67->48, low 18->42).

**The smoking gun.** `min eig(P)` decays as `1/(1 + r s)` with `r` = the face-row
stiffening ratio (`diag_edge` measures 8.2 on the toroid; the rot-ellipse fit
gives 7.27):

| s | 0.06 | 0.10 | 0.15 | 0.22 | 0.30 | 0.55 | 1.00 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| measured `min eig(P)`/s=0 | 0.714 | 0.596 | 0.491 | 0.394 | 0.320 | 0.202 | 0.121 |
| `1/(1 + 7.27 s)` | 0.696 | 0.579 | 0.478 | 0.384 | 0.314 | 0.200 | 0.121 |

Agreement to ~3% across a 17x range. **`P`'s smallest eigenvalue IS the
boundary row's inverse, and the penalty dilutes it exactly as
`1/(1 + r s)`.** §10's "a large penalty ABANDONS the row rather than
constraining it" is no longer an argument -- it is a measured law.

> **VERDICT: 0.15 is a kappa-BALANCE point, not a missing algebraic factor.**
> `alpha` is right for `L`; the preconditioner is minimising `kappa(P^-1 L)`,
> which is a different question with a different optimum. There is nothing to
> derive. §13.3's framing ("a constant is missing from the derivation") is
> WITHDRAWN.

### 14.2 AND THAT QUALIFIES "GEOMETRY-INDEPENDENT" — the basin is BROAD, not the
constant universal

`diag_bcplow`, k=1 free, iterations:

| geom | n | `nobc` | 0.03 | 0.06 | 0.10 | 0.15 | 0.22 | 0.30 | argmin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toroid | 8 | 196 | 130 | 108 | 92 | 83 | 78 | **77** | 0.30 |
| toroid | 12 | 310 | 166 | 133 | 109 | 100 | 99 | **96** | 0.30 |
| toroid | 16 | 395 | 184 | 143 | 120 | **113** | 114 | 113 | 0.15 |
| rot-ell | 8 | 439 | 322 | 296 | 283 | **280** | 284 | 292 | 0.15 |
| rot-ell | 12 | 687 | 460 | 430 | 427 | **415** | 446 | 460 | 0.15 |
| w7x | 8 | 811 | 626 | 602 | **594** | 612 | 634 | 668 | 0.10 |
| w7x | 12 | 1536 | 1091 | **1037** | 1098 | 1161 | 1214 | 1307 | 0.06 |

**The true argmin IS geometry-dependent** -- 0.30 (toroid) down to 0.06 (W7-X),
a 5x spread -- **and on W7-X it drifts DOWN with n** (0.10 -> 0.06). §13.2's
"no drift" was an artefact of the bracket floor, exactly as §13.5 warned.

What survives is weaker but still useful: **the basin is broad and the basins
overlap.** rot-ellipse k=1 n=8 varies only 283/280/284 over 0.10-0.22; the
`cond` curve varies 8% over 0.06-0.30. Cost of using a flat 0.15 against each
case's own best: 0% (rot-ell, toroid n=16), 3-4% (toroid n=12, w7x n=8), and
**12% at worst so far** (w7x n=12, 1161 vs 1037).

> Restated: **0.15 is a robust DEFAULT sitting inside every basin, not a
> universal constant.** Against the shipped scale 1.0 (a 2x penalty at n=20)
> a 12% miss is cheap -- but "geometry-independent" in §13.3 overclaimed and is
> corrected here.

### 14.3 `wibp` AND `ibpf` ARE STILL REFUTED AT THE CORRECTED SCALE

`diag_bcp15base`, the confound tested (n=8,12; k=1 and k=2 free):

| geom | n | k | `ibpd` 1.00 | `bcp15` | `wibp15` | `ibpf15` |
| --- | --- | --- | --- | --- | --- | --- |
| rot-ell | 8 | 1 | 400 | **280** | 397 | 394 |
| rot-ell | 12 | 1 | 675 | **421** | 711 | 733 |
| w7x | 8 | 1 | 945 | **611** | 986 | 828 |
| rot-ell | 8 | 2 | 401 | **277** | 422 | 428 |
| rot-ell | 12 | 2 | 737 | **449** | 758 | 762 |
| w7x | 8 | 2 | 902 | **594** | 988 | 900 |

Both are **1.35-1.7x WORSE than the plain scalar term at the same scale**. The
confound did not save them. So §12.9's argument (2) is RESTORED for these two:
the exact 2-D face shape (`wibp`) and the off-`e e^T` cross term (`ibpf`) are
genuinely refuted, not artefacts of over-scaling. §13.4 was right to flag the
confound and wrong to expect it to matter here.

### 14.4 `fm` IS BACK — and this is the headline

Same jobs, against `outputs/diag_hsweep/` at scale 1.0 (identical config; the
`ibpd` anchor matches to 0.3%, 675 vs 677):

| rot-ell n=12 k=1 | scale 1.0 | scale 0.15 | change |
| --- | --- | --- | --- |
| `ibpd` | 677 | 421 | -38% |
| **`fm3`** | 374 | **275** | **-26%** |
| `o1` | 512 | 506 | -1% |
| `o2` | 361 | 361 | 0% |

**The ring arms are INSENSITIVE to the boundary scale and `fm` is not.** That
makes sense and is the mechanism: `o1/o2` EVICT the boundary rows to the dense
probe, so the atom's face term never acts on them; `fm` rides on the atom's own
boundary treatment and inherits every improvement to it.

Consequence at the corrected scale, all cells so far:

| geom | n | k | `bcp15` | `fm3` | `o1` | `o2` |
| --- | --- | --- | --- | --- | --- | --- |
| rot-ell | 8 | 1 | 280 | **185** | 309 | 206 |
| rot-ell | 12 | 1 | 421 | **275** | 506 | 361 |
| w7x | 8 | 1 | 611 | **329** | 569 | 375 |
| rot-ell | 8 | 2 | 277 | **197** | 265 | 207 |
| rot-ell | 12 | 2 | 449 | **313** | 464 | 342 |
| w7x | 8 | 2 | 594 | 366 | 481 | **355** |

`fm3` now beats `o2` outright in 5 of 6 cells (the sixth, w7x k=2, is 366 vs
355 -- inside the 2% noise floor of §13.1) **at roughly a tenth of the build
cost** (4.7s vs ~45s at n=12, §12.8). At scale 1.0 it was roughly tied with
`o2` and losing to `o1` on W7-X at n>=16.

§12.10's "`fm` is HALF DEAD" is provisionally WITHDRAWN. The asymptotic
question is still open and is exactly what the pending n=16,20 cells decide:
at scale 1.0 `o1` overtook `fm3` on W7-X at n=16 (1397 vs 1507). If that
crossover moves out or disappears at 0.15, `fm` is the lead route, not the
banded capacitance.

### 14.5 `bcpp` — too early, but it does NOT look like a clean degree law

Only rot-ellipse p=2 and p=3 in. k=1 argmin 0.15 (p=2) and 0.11 (p=3), which
is loosely consistent with a `1/(2p+1)`-ish decline (predicting 0.20 / 0.143) --
but k=3 at p=2 puts the argmin at **0.28**, nearly 2x the k=1 value at the same
p. A clean degree factor would not split by degree like that. Combined with
§14.1, the likelier reading is that there is no law: the basin is broad, its
centre wanders with geometry, degree and h, and 0.15 is simply a good place to
stand. p=4,5 pending.

### 14.6 LAUNCHED: `diag_bcphi` — is 0.15 decent at HIGH resolution?

11 jobs (`bcphi_*`), 2026-08-22. The doubt is specific and comes from §14.2:
on W7-X the true argmin was still SLIDING DOWN with n (0.10 at n=8 -> 0.06 at
n=12) while the toroid approaches 0.15 from ABOVE (0.30 -> 0.30 -> 0.15 at
n=8/12/16). If the shaped drift continues, a flat 0.15 stops being safe
somewhere above n=20 -- and n=20 is the entire evidence base for §13/§14.

Grid: n = **24, 28, 32** on all three geometries, plus n = **40** on the two
shaped ones (the toroid is converging toward 0.15 from above and is not where
the risk is). With the existing n = 8..20 that is a **5x span in linear
resolution and 125x in DOFs** (2344 -> 346k at k=1).

Arms: `bcp3, bcp6, bcp10, bcp15, bcp22` -- five scales spanning 7x around
0.15, enough to localise an argmin that is still moving; plus `ibpd` (scale
1.0) at n=24 only, since it is the most expensive arm and its verdict is
settled. `nobc` dropped for the same reason. k=1 and k=3 free; k=2 tracks k=1
in every cell measured and would double the cost.

Cost model fitted to the n=8..20 logs, which is what made this affordable:

* `compute_nullspaces` is a matrix-free DIRECT Hodge decomposition (a fixed pair
  of Hodge solves per form, `mrx/nullspace.py:354`), **linear in DOFs at
  ~0.028 s/dof** -- 92.6 / 228.9 / 547.1 / 1193.6 s at n = 8/12/16/20 on W7-X.
  Not a dense factorisation, so there is no `n_dof^2` memory wall; that is the
  fact that makes n=40 (346k DOFs) possible at all.
* CG ~ `1.23e-6` s/it/dof; `n_dof(k=1) = 43300 (n/20)^3`.
* => W7-X n=32 ~ 2.7 h, n=40 ~ 5.5 h. Partition limit is 48 h; jobs submitted
  with `TIMEOUT_MIN=1440`, `MEM_GB=160` (200 at n=40).

READ IT AS: `bcp15 / min(swept)` per cell. If that ratio stays within a few
percent out to n=40, 0.15 is a safe default and §14.2's "robust default in a
broad basin" holds at production resolution. If it grows past ~1.2 on the
shaped geometries, the basin centre is genuinely running away with h and the
default must be a function of n (or the scale must be chosen per-problem),
which would also re-open §14.4's `fm` result -- `fm` rides on the atom's face
term and inherits whatever the scale does.

### 14.7 `diag_bcp15base` COMPLETE — the `o1` crossover was an ARTEFACT, and
`fm` is the lead route

`fm3` vs the ring arms, scale 1.0 (`diag_hsweep`) against scale 0.15, k=1 free:

| geom | n | `fm3` 1.0 | **`fm3` .15** | `o1` 1.0 | `o1` .15 | `o2` 1.0 | `o2` .15 | `fm3`/`o1` | `fm3`/`o2` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rot-ell | 8 | 226 | **185** | 309 | 309 | 205 | 206 | 0.60 | 0.90 |
| rot-ell | 12 | 374 | **275** | 512 | 506 | 361 | 361 | 0.54 | 0.76 |
| rot-ell | 16 | 517 | **359** | 708 | 705 | 501 | 502 | 0.51 | 0.72 |
| rot-ell | 20 | 640 | **439** | 937 | 938 | 620 | 618 | 0.47 | 0.71 |
| w7x | 8 | 412 | **329** | 568 | 569 | 374 | 375 | 0.58 | 0.88 |
| w7x | 12 | 900 | **659** | 988 | 994 | 678 | 680 | 0.66 | 0.97 |
| w7x | 16 | 1507 | **1033** | 1397 | 1413 | 976 | 979 | 0.73 | 1.06 |
| w7x | 20 | 2033 | **1308** | 1742 | 1735 | 1222 | 1221 | 0.75 | 1.07 |

1. **The ring arms are SCALE-BLIND** -- `o1` 309/512/708/937 -> 309/506/705/938,
   `o2` unchanged to within noise. Exactly the predicted mechanism: they EVICT
   the boundary rows to the dense probe, so the atom's face term never acts on
   them. `fm` rides on that term and gains 20-31% everywhere.
2. **§12.5's "`o1` overtakes `fm3` at n=16 on W7-X" is REFUTED.** At scale 1.0
   `fm3`/`o1` went 0.58 / 0.66 / 1.08 / 1.17 -- a crossover at n=16. At 0.15 it
   is 0.58 / 0.66 / **0.73 / 0.75** and never crosses. The crossover was an
   artefact of the over-scaled baseline, precisely as §12.12 warned it might be.
3. **`o2` still edges `fm3` on W7-X at n >= 16, but by 6-7%** (1033 vs 979,
   1308 vs 1221) -- against a build that is ~20x more expensive (`fm3` 6.4s vs
   `o2` ~120s at n=20, §12.8). On TOTAL time `fm3` wins outright: at n=20 W7-X
   it needs 87 more iterations at ~0.053 s/it = 4.6s, against 114s more build.
   On rot-ellipse `fm3` beats `o2` on iterations alone at every n.
4. `wibp`/`ibpf` get WORSE with resolution, decisively: rot-ell n=20 `bcp15`
   661 vs `wibp15` 1283 (1.9x) and `ibpf15` 1727 (2.6x). Dead beyond doubt.

> **§12.10 is superseded: `fm` is the LEAD ROUTE, not the banded capacitance.**
> `fm3` at the corrected scale is the best arm on rot-ellipse at every n and
> within 7% of the best on W7-X at a twentieth of the build cost.

### 14.8 `diag_bcphi` (8/11) — 0.15 is the WRONG DEFAULT at production
resolution; use 0.10

Worst-case penalty of a FIXED scale against each cell's own best:

| band | 0.03 | 0.06 | **0.10** | 0.15 | 0.22 |
| --- | --- | --- | --- | --- | --- |
| n = 8..20 (the old evidence base) | 1.74 | 1.43 | 1.20 | **1.14** | 1.25 |
| n = 24..32 (production) | 1.49 | 1.12 | **1.11** | 1.23 | 1.36 |

**The best fixed scale MOVES DOWN with resolution, and 0.15 was an artefact of
stopping at n=20.** The mechanism is a genuine crossover between two geometries
pulling in opposite directions (k=1, penalty vs that cell's own best):

| n | 8 | 12 | 16 | 20 | 24 | 28 | 32 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| toroid at 0.06 | 1.40 | 1.39 | 1.27 | 1.20 | 1.12 | 1.06 | **1.02** |
| w7x at 0.15 | 1.03 | 1.12 | 1.10 | 1.14 | 1.18 | **1.23** | -- |

The toroid's demand for a LARGE scale FADES with n (by n=32 it is flat: 167 at
0.10, 0.15 and 0.22 alike); W7-X's demand for a SMALL scale GROWS. So the
compromise slides down, and at production resolution the toroid no longer votes.

**And the W7-X argmin is STILL at the bracket floor** (0.03 at every n >= 16 --
the fifth sweep in this investigation to bottom out at its own low edge, cf.
§13.5). The true W7-X optimum is unknown and below 0.03, so 0.10's penalty
there is a LOWER bound.

By degree, `bcplow` + `bcphi` + `bcpp`:

* **k=3 is easy and wants ~0.15** (worst case 1.02 over every cell). It is also
  insensitive: 0.10-0.30 are all within 6%.
* **k=1 on W7-X is the hard case and wants <= 0.03**, worsening with BOTH n
  (1.03 -> 1.23) and p (`bcpp`: 15/best = 1.046 / 1.090 / 1.118 / **1.188** at
  p = 2/3/4/5).
* `bcpp` also kills the degree-law idea properly: the argmin falls with p
  (k=3 rot-ell 0.28 / 0.15 / 0.05 / 0.05 at p=2..5) but not by any clean rule,
  and k=1 and k=3 disagree by ~2x at the same p. Consistent with §14.1 --
  a broad basin whose centre wanders, not a derivable factor.

**RECOMMENDATION: default 0.10, or 0.15 at k=3 and 0.06 at k=1/2.** But keep the
magnitudes straight -- this is a 10-25% argument sitting on top of a 2.4x one:
at W7-X n=24 k=1 the shipped scale 1.00 gives 5204 against 2170 at the best
scale. **Getting off 1.00 is the whole prize; choosing within 0.03-0.15 is
second order.**

Still running: `bcphi` rot-ell n=40, w7x n=32, w7x n=40.

---

## 15. COST-ADJUSTED: the scalar term + scale IS the right default (2026-08-22)

Tobias's objection, and it is correct: §14.7 ranked arms by ITERATIONS. That is
the wrong figure of merit when the arms differ by 20x in build cost and 1000x
in storage, and it inverted the ranking.

Per-CG-iteration cost, derived from the `diag_hsweep` logs
(`elapsed - nullspace - builds` / total iterations; the preconditioner apply is
0.09-0.20 ms, under 0.2% of an iteration, so the operator apply dominates and
all arms cost the same per iteration):

| | rot-ellipse n=12 | rot-ellipse n=20 | w7x n=12 | w7x n=20 |
| --- | --- | --- | --- | --- |
| ms / CG iteration | 40.9 | 36.2 | 39.0 | 59.7 |

### 15.1 TOTAL SOLVE TIME (build + iterations), k=1 free, n=20

**rot-ellipse**

| arm | iters | build | CG | **TOTAL** | vs best |
| --- | --- | --- | --- | --- | --- |
| `fm3` @0.15 | 439 | 6.4s | 15.9s | **22.3s** | 1.00 |
| **`bcp10`** | 652 | 2.7s | 23.6s | **26.3s** | **1.18** |
| `ibpd` @1.00 | 1224 | 2.7s | 44.2s | 46.9s | 2.11 |
| jacobi | 1269 | 7.8s | 45.9s | 53.7s | 2.41 |
| `o1` @0.15 | 938 | 60.2s | 33.9s | 94.1s | 4.23 |
| `o2` @0.15 | 618 | 118.9s | 22.3s | 141.2s | 6.34 |

**W7-X**

| arm | iters | build | CG | **TOTAL** | vs best |
| --- | --- | --- | --- | --- | --- |
| `fm3` @0.15 | 1308 | 6.4s | 78.1s | **84.5s** | 1.00 |
| **`bcp3`** | 1831 | 2.6s | 109.4s | **112.0s** | **1.32** |
| `bcp10` | 1987 | 2.6s | 118.7s | 121.3s | 1.43 |
| `o1` @0.15 | 1735 | 60.5s | 103.7s | 164.2s | 1.94 |
| jacobi | 3023 | 7.9s | 180.6s | 188.5s | 2.23 |
| `o2` @0.15 | 1221 | 120.4s | 73.0s | 193.4s | 2.29 |
| `ibpd` @1.00 | 4150 | 2.6s | 248.0s | 250.6s | 2.96 |

**`o1` and `o2` are DEAD on total time** -- 4.2x and 6.3x worse than `fm3` on
rot-ellipse, and on W7-X `o2` is worse than JACOBI. They only ever looked
competitive because the tables reported iterations. Every "o1/o2 leads"
statement in §12 and §14.7 is an artefact of that.

**`bcp` alone captures 74% (rot-ellipse) and 66% (W7-X) of the entire iteration
gain available between jacobi and the best known arm** -- and lands within
18-32% of the best on total time, at zero structural cost.

CAVEAT, stated honestly: if the build is amortised over many solves (a
time-stepping loop, one build many RHS) the ranking reverts toward iterations,
where `fm3` (1308) and `o2` (1221) lead on W7-X. Break-even for `o2` vs
`bcp10` there is `117.8s / ((1987-1221) x 59.7ms)` = **2.6 solves**. So the
total-time argument alone does not kill `o2` in a production loop -- its
storage and scaling do (§15.2).

### 15.2 STRUCTURAL COST — the part that is not a number

| | `bcp` (scalar + rank-1) | `fm3` (coarse space) | `o1`/`o2` (dense probe) |
| --- | --- | --- | --- |
| extra storage @n=20 | **0.1 MB** | 102 MB, **LINEAR in n_dof** | 50 / ~200 MB, `O((n_t n_z)^2)` |
| extra build | **0s** (folds into `K_r`) | +3.7s | +58s / +118s |
| knobs | **one scalar** | `coarse_rings`, `coarse_modes`, `coarse_set`, `coarse_mode`, `coarse_trunc` | `outer_rings` |
| effect on the atom | **none** -- rank-1 update, FD and the shared eigenbasis untouched | hybrid/balancing sandwich around the atom | evicts rows to a dense block |
| known traps | -- | ADDITIVE form is structurally WRONG and was measured as such; `m95 ~ n_t/3` so the fixed box is asymptotically wrong | perturbs the DIRICHLET rows (`outer_rings` is NOT guarded on dbc, §11) |
| coverage tested | k=0,1,2,3 x free+dbc x 3 geometries x n=8..32 x p=2..5; never diverged | k=1,2 free only | k=1,2,3 |

`bcp` changes ONE FLOAT in an update that already exists. It cannot change the
cost, the storage, the eigenbasis, or the Dirichlet behaviour, because the term
it scales vanishes identically under an essential condition (verified invariant,
§12.3) and merges into `K_r` as a rank one.

### 15.3 THE RECOMMENDATION

> **Land the scalar term at a corrected scale. That is the deliverable.**
> `bcp10` (or 0.15 at k=3 / 0.06 at k=1/2) buys ~2x over jacobi and ~2-3x over
> the shipped `ibpd`/`exact` at k=1/2 free, with zero added complexity, zero
> storage, and the widest test coverage of anything here.

`fm3` is a real further **1.18x (rot-ellipse) / 1.32x (W7-X)** on total time. It
is worth keeping as an OPTION for the hardest cases, but on a
complexity-and-robustness budget it should not be the default: it adds a coarse
space with five parameters, memory linear in `n_dof`, a correctness trap that
has already been hit once, and it is untested at k=0 and under Dirichlet.

`o1`/`o2` should be retired as production candidates. They remain useful as a
DIAGNOSTIC -- an "exact boundary treatment" upper bound on what any boundary
method can buy -- which is what they were originally for.

§14.7's "`fm` is the LEAD ROUTE" is withdrawn as a production statement; it is
true only on iterations. §12.10's banded capacitance drops further: it is an
attempt to make `o1`/`o2` affordable, and §15.1 says the thing it is trying to
rescue is 4-6x off the pace before any banding.

---

## 16. THE HYPERPARAMETER — what to ship, and how reliable it is

### 16.1 The name

`bcp` is debug-harness arm grammar, not an API name: `bcs`N multiplies the
natural-BC penalty by the integer N, `bcp`N does the same by **N/100** ("BC
penalty, in per cent"), added when the interesting range turned out to be
sub-unity (`verify_block_jacobi.py:202-207`). The real knob is the env var
`MRX_BJ_BC_SCALE`, read at `block_jacobi_laplacian.py:457`. **On landing it
should become a constructor argument** -- `BlockJacobiLaplacian(bc_scale=0.10)`
-- and the env var kept only as an override.

### 16.2 It depends on everything, weakly. Argmin by cell (p=3 unless noted)

| k | geom | n=8 | 12 | 16 | 20 | 24 | 28 | 32 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | toroid | 0.30 | 0.55 | 0.15 | 0.22 | 0.10 | 0.10 | 0.10 |
| 1 | rot-ell | 0.15 | 0.11 | 0.10 | 0.10 | 0.06 | 0.06 | 0.10 |
| 1 | w7x | 0.10 | 0.06 | <0.03 | <0.03 | <0.03 | <0.03 | -- |
| 3 | toroid | 0.22 | 0.15 | 0.10 | 0.22 | 0.06 | 0.06 | 0.06 |
| 3 | rot-ell | 0.15 | 0.10 | 0.06 | 0.06 | 0.06 | 0.06 | <0.03 |
| 3 | w7x | 0.15 | 0.15 | 0.10 | 0.06 | 0.06 | 0.06 | -- |

and in p (n=12): rot-ell k=1 **0.15 / 0.11 / 0.11 / 0.05** at p=2/3/4/5;
rot-ell k=3 **0.28 / 0.10 / 0.05 / 0.05**; w7x k=3 **0.28 / 0.15 / 0.08 / 0.08**.

* **geometry: YES**, and it is the strongest axis at low n -- a 9x spread at
  n=12 (toroid 0.55, rot-ell 0.11, w7x 0.06) -- narrowing to ~3x by n>=24.
* **n: YES**, monotone DOWN on every geometry and degree.
* **p: YES**, monotone DOWN, a factor 2-5 from p=2 to p=5.
* **k: barely.** k=1 and k=2 track each other; k=3 wants slightly more at low n
  and converges to the same 0.06 by n>=24. Not worth encoding. (The toroid k=2
  "1.00" entries are an artefact of a FLAT plateau -- 67/68/68/67 over
  0.15-1.00 -- where the argmin is meaningless.)

There is no law to fit here, and §14.1 says why: this is a `kappa`-balance
point, not an algebraic factor. Do not try to derive it.

### 16.3 Why it is safe anyway — MINIMAX over all 58 cells

Worst-case ratio to each cell's OWN best:

| band | 0.03 | 0.06 | 0.10 | 0.15 | 0.22 | 0.30 |
| --- | --- | --- | --- | --- | --- | --- |
| all 58 cells | 1.76 | 1.45 | **1.19** | 1.23 | 1.36 | 1.37 |
| production (n>=16) | 1.63 | 1.27 | **1.11** | 1.23 | 1.36 | 1.37 |
| k=1,2 only | 1.76 | 1.45 | **1.19** | 1.23 | 1.36 | 1.37 |
| k=3 only | 1.53 | 1.30 | 1.11 | 1.10 | **1.04** | 1.07 |

**0.10 is the minimax choice**: never worse than 1.19x anywhere, 1.11x at
production resolution. Over the whole range 0.06-0.15 -- a 2.5x span in the
knob -- nothing is worse than 1.45x, and at n>=16 nothing is worse than 1.27x.
That is a genuinely flat basin.

For contrast, the two things it replaces: the SHIPPED scale 1.00 costs up to
**2.24x** (W7-X k=1 n=20), and 0 (no term) costs **1.40-3.74x**.

### 16.4 The asymmetry — err LOW

* Too small is BOUNDED: the limit is `nobc`, at most 3.74x (and that worst case
  is the toroid, where the term matters most and the scale is least critical).
* Too large is UNBOUNDED: 1.00 gives 2.24x, `bcp300` gave 2998 vs ~76 (39x),
  x1e4 was catastrophic. §14.1 gives the mechanism -- `min eig(P) = 1/(1+r s)`,
  so a large scale drives the atom's inverse on the boundary row to zero.

So when in doubt, go smaller. 0.06 is a defensible conservative alternative to
0.10 (1.27x worst at n>=16 against 1.11x) and is strictly better on W7-X.

### 16.5 The censored cell, and why it does not matter

W7-X k=1 has its argmin AT the bracket floor 0.03 for every n>=16 -- the fifth
sweep in this investigation to bottom out at its own low edge (§13.5). The true
optimum there is unknown. **It is also irrelevant**, because the curve is flat
across the floor:

| n | 0 | 0.03 | 0.06 | 0.10 | 0.15 | 0.10/best |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 2128 | 1515 | 1519 | 1568 | 1662 | 1.03 |
| 20 | 2696 | 1831 | 1835 | 1987 | 2074 | 1.09 |
| 24 | -- | 2170 | 2250 | 2329 | 2571 | 1.07 |
| 28 | -- | 2470 | 2580 | 2737 | 3034 | 1.11 |

0.03 and 0.06 are identical to within the 1% noise floor (§13.1), and 0 is
1.4x worse -- so the minimum lies in (0, 0.06), is bounded below by ~1800 at
n=20, and 0.10 costs at most 1.03-1.11 across n=16-28. Chasing the true floor
would buy under 10% on one geometry at one degree.

### 16.6 WHAT TO LAND

1. `BlockJacobiLaplacian(bc_scale=0.10)`, `MRX_BJ_BC_SCALE` retained as an
   override. Default `bc_entry` stays `ibpd` (the derived alpha) -- the scale
   multiplies it.
2. Document that it is an EMPIRICAL `kappa`-balance constant, not derived, and
   that the optimum drifts DOWN with n and p -- at very high resolution or
   p >= 5 a user may prefer 0.05.
3. **The decision is not close.** The current default (`exact`, scale 1.0) is
   WORSE THAN NO BOUNDARY TERM AT ALL at k=1/2 free on rot-ellipse (§12.11).
   Any value in 0.03-0.30 is a large improvement on the status quo; the choice
   within that range is a <=20% question.
4. Do NOT make it k-, p- or geometry-dependent. The gain over a flat 0.10 is
   at most ~15% and it would turn one documented constant into a table.

---

## 17. WHERE THE FACTOR COMES FROM — what is measured, what is inferred

Asked by Tobias: do we know where the ~0.1 comes from and why we need it?
Answer: the MECHANISM is measured and solid; the ORIGIN has a coherent account
that is not proven; and there is a ~5x quantitative gap that is still open.

### 17.1 MEASURED — why a scale is needed at all

Without the term the atom is too SOFT at the boundary; with it at full strength
it is too STIFF. Neither is a statement about the surface integral -- both are
statements about `P` versus `L`.

* `s = 0`: 56 HIGH outliers on rot-ell k=1, boundary-localised. `P` under-
  stiffens the face row, those modes get `lambda(P^-1 L) >> 1`.
* `s = 1`: 27 LOW outliers, and `min eig(P)` has fallen 8.3x. The face row is
  stiffened ~9.2x (`diag_edge`: `alpha e_l^2 / K_r[-1,-1]` = 8.196), and
  **`min eig(P) = 1/(1 + r s)` to 3% over a 17x range in `s`** -- the atom's
  boundary DOF is effectively DECOUPLED (`e` is one-hot and the atom is a
  Kronecker sum, so that radial index is its own), and `P` inverts it bare.
* The optimum is where the two families balance. **At the optimum BOTH still
  exist** (~41 high + ~11 low at `s = 0.15`). The scale does not fix the
  boundary; it trades one error against the other.

That last point is the answer to "why do we need it": not because a factor is
missing from the derivation, but because **the boundary error is a layer with
structure a rank-one face term cannot represent**, so the best a scalar can do
is balance. §12's independent finding says the same thing from the other side
-- `o1/o2/o3` keep paying at rings 2 and 3, so the defect has radial extent.

### 17.2 MEASURED — what the factor is NOT

Every derivable local correction has been tried and each one fails:

| candidate | verdict |
| --- | --- |
| a metric factor (`<S><P>` vs `<SP>`) | 5-9%, geometry-shaped; the optimum is not |
| a missing power of `h` | `mu_0`'s `1/h` is right; the optimum barely drifts with n |
| the cross term `rho` (the EXACT 1-D round trip) | **`ibpr` made `P` INDEFINITE** on every geometry at every k (§12.2). The exact local correction is WORSE than the approximate one. |
| the exact 2-D face shape (`wibp`) | 1.35-1.9x worse at the corrected scale (§14.3) |
| higher rank (`ibpf`) | 1.4-2.6x worse (§14.3) |
| a degree factor `1/(2p+1)`-ish | argmin falls with `p` but k=1 and k=3 disagree ~2x at the same `p` (§14.8) |

**The exact correction being indefinite is the strongest single clue.** If the
factor were a missing term in the surface integral, computing that term exactly
would help. It does the opposite.

### 17.3 INFERRED — the account that fits everything

`L`'s boundary DOF is coupled to the interior; `P`'s is not (Kronecker sum,
one-hot `e`). Matching `P`'s DIAGONAL to `L`'s diagonal -- which is what
`s = 1` does, since `alpha` is exactly `L`'s surface term -- therefore
over-stiffens `P` relative to `L`'s EFFECTIVE action on the modes that matter,
because coupling to the interior lowers the Rayleigh quotient there. The right
number for `P` is the ratio of `L`'s effective to its diagonal boundary
stiffness: a **DtN / Steklov-type quantity -- nonlocal, with radial extent.**

This account is consistent with all of: `s_opt < 1`; no closed form (a DtN is
not a local integral); drift with n, p and geometry (a DtN drifts with all
three); every exact LOCAL correction failing; the exact RING treatments
(`o1/o2`, which invert `L`'s true block and so carry the within-ring coupling)
working on iterations; and the residual being a layer rather than a face.

**Quantitative gap, stated plainly.** The cross term is the 1-D version of
exactly this coupling correction and it is measured: `rho` = 0.63 (toroid) /
0.40 (W7-X), i.e. it accounts for `1.0 -> ~0.5`. The observed optimum is
0.06-0.15. **A factor of ~4-5 is attributed to the 3-D DtN and has NOT been
measured.** Until it is, "it is the Schur/DtN reduction" is a hypothesis with
one supporting order of magnitude, not a result.

### 17.4 THE TEST THAT WOULD CLOSE IT -- and the prize

`probe_core_block` (`:1030`) builds `L`'s exact dense block on a chosen set of
rows by one operator apply per row. So for the outer ring:

1. form `B_exact` = `L` restricted to the outer ring (exact, includes the
   angular and cross-component coupling the atom drops);
2. form `B_atom(s)` = the atom's own action on the same rows as a function of
   `s`;
3. report the generalised spectrum of `(B_exact, B_atom(s))` and find the `s`
   that clusters it best.

That number is a directly computable "effective boundary coefficient". If it
lands on the `s_opt` measured by the iteration sweeps, §17.3 is confirmed and
the factor is explained. If it does not, §17.3 is wrong and the factor is
something else.

**The prize is that the hyperparameter disappears.** The ring probe is
`n_t n_z` applies per component -- too expensive as a preconditioner (that IS
`o1`, retired in §15) -- but it only has to be done ONCE at setup to pick a
scalar, and it could be done on a COARSE ring or a few modes rather than the
full ring. A self-computed `bc_scale` would adapt to k, p, n and geometry for
free and remove the one fitted constant in the design. Cost: a few hundred
applies at setup, versus the ~2600 an `o1` build already spends per solve.

Cheap and worth doing before landing a constant. `diag_bcpn`..`bcphi` already
give the target values to check against on 58 cells.

### 17.5 THE DtN TEST — §17.3 REFUTED, and the real answer found

`scripts/debug/bc_schur_effective.py`, `outputs/diag_bcschur/`, 5 jobs, all
complete. For the outer `d` rings `R`: `B_raw = L[R,R]`,
`B_schur = L[R,R] - L[R,I] inv(L[I,I]) L[I,R]`, `A(s) = inv(P(s))[R,R]`.
(First submission crashed on a guessed API -- `seq.assemble_operators()` does
not exist; the path is `op.assemble_incidence_operators` +
`assemble_mass_jacobi_preconditioner` + `seq.set_operators`.)

**§17.3 IS REFUTED on both of its predictions.**

1. *The DtN is small, and it shrinks with depth* -- exactly backwards for an
   account that needs it to remove ~90% of `alpha`. Fraction of `tr(L[R,R])`
   removed by the Schur complement, k=1:

   | depth | 1 | 2 | 3 | 4 |
   | --- | --- | --- | --- | --- |
   | toroid | 17.6% | 3.2% | 3.4% | 0.4% |
   | rot-ell | 23.8% | 4.8% | 3.5% | 1.0% |
   | w7x | 29.1% | 6.1% | 3.5% | 1.1% |

2. *`B_raw` and `B_schur` pick the SAME scale* at depth >= 3 in almost every
   case, and where they differ (depth 1) the Schur version prefers a LARGER
   `s`, not smaller -- toroid depth 1: raw 0.55, schur **2.0**. The prediction
   was raw -> 1, schur -> 0.1. Neither happened.

**BUT THE TEST FOUND THE ANSWER ANYWAY, in the arm that was meant to be the
control.** Two criteria on the SAME raw block disagree, and the disagreement
IS the factor:

| criterion | what it picks | meaning |
| --- | --- | --- |
| `||A(s) - B_raw|| / ||B_raw||` | **s ~ 1.0** (k=1, both shaped geoms, every depth >= 2) | `alpha` IS the best NORM approximation to `L`'s boundary block -- the derivation is right |
| `cond(B_raw, A(s))` | **s ~ 0.06-0.55, geometry-ordered** | the best PRECONDITIONER of that block is a much smaller `alpha` |

> **The best approximation is not the best preconditioner, and the gap between
> them IS the ~10x.** Not a missing term, not a metric factor, not the DtN:
> the atom's ring block is missing the WITHIN-RING coupling (angular and
> cross-component) that `L`'s block carries, and under-stiffening the diagonal
> is the best a diagonal-only knob can do to compensate spectrally.

**And it PREDICTS the right scale, per geometry, with no solve.**
`cond(B_raw, A(s))` argmin at depth 3, k=1, against the ITERATION argmin from
`diag_bcplow` at the nearest resolution:

| geom | ring-block argmin | iteration argmin | |
| --- | --- | --- | --- |
| toroid 6,12,6 | 0.55 | 0.30 (n=8) / 0.55 (n=12) | within one sweep point |
| rot-ell 6,12,6 | 0.22 | 0.15 (n=8) | within one sweep point |
| **rot-ell 8,16,8** | **0.15** | **0.15 (n=8)** | **exact** |
| w7x 6,12,6 | 0.06 | 0.10 (n=8) | within one sweep point |
| **w7x 8,16,8** | **0.06** | **0.10 (n=8) / 0.06 (n=12)** | **exact at n=12** |

The geometry ORDERING is reproduced exactly (toroid > rot-ellipse > W7-X), and
the magnitudes agree to within one point of the iteration sweep in every case.

**Consequences.**

* §17.3's Schur/DtN story is dead. Delete it as an explanation; §17.1-17.2
  (the measured mechanism and the list of what the factor is NOT) stand.
* The factor is now EXPLAINED: it is the spectral-vs-norm gap on the boundary
  ring, caused by coupling the separable atom cannot represent. That is the
  same defect §12 identified as "a layer" -- but the relevant coupling is
  ANGULAR and CROSS-COMPONENT within the ring, not radial into the interior.
* **The hyperparameter is computable.** A depth-3 `probe_core_block` plus a
  small generalised eigenproblem, ONCE at setup, reproduces the measured
  optimum. Cost is `3 n_t n_z` applies per component -- the same order as an
  `o1` build, but paid once for a scalar rather than every solve for a dense
  block, and it could be cut further by probing a coarse ring or a few modes.
* Caveat: validated on 3 geometries x 2 meshes at k=1 (plus k=3, where the
  ring-block criterion is noisier -- argmins 0.15-1.0 -- because k=3 is the
  single-component case with no cross-component coupling to detect, which is
  itself consistent with the explanation).

---

## 18. OPEN ITEMS

1. **Land it** (§16.6): `bc_scale` as a constructor arg, default 0.10.
2. **Optional, and now justified**: auto-compute `bc_scale` from the depth-3
   ring-block match (§17.5) instead of shipping a constant. Would remove the
   only fitted number in the design and adapt to k, p, n and geometry. Needs a
   cheaper probe (coarse ring / few modes) to be worth it at production sizes.
3. `fm3` at the corrected scale is a further 1.18-1.32x on total time (§15.1)
   -- keep as an option, not a default; still untested at k=0 and under dbc.
4. Still running when this was written: `bcphi` rot-ell n=40, w7x n=32, n=40.
   They extend §16.2's argmin table to n=40 and cannot change the
   recommendation.
5. The W7-X k=1 argmin has been bracket-censored at 0.03 in five separate
   sweeps (§16.5). Harmless -- the curve is flat there -- but note it before
   designing a sixth.

### 17.6 THREE MORE CONFIRMATIONS — less coupling wants a LARGER scale

Asked by Tobias: should `s = 1` at k=0 and k=3, since those are the cases with
no cross-component coupling? Answer: k=0 is vacuous, k=3 is directionally right
but the number is 0.15 not 1.0 -- and the iteration data confirms the §17.5
mechanism three separate ways.

**k=0 has no boundary term at all.** `trace_components(0) = ()` -- `W_0 = 0`,
no component's radial axis is a derivative axis, so `_boundary_entry_direct`
never runs (§1). `s` is a NO-OP at k=0. And nothing is needed there: the atom's
`w d_r u = 0` IS the operator's natural condition exactly, and k=0 free is the
best-conditioned case anywhere (43 vs jacobi 398, 9.2x).

**What `s = 1` costs, per cell (ratio to that cell's own best):**

| k | geom | n=8 | 12 | 16 | 20 | 24 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **toroid** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| 1 | rot-ell | 1.43 | 1.60 | 1.73 | 1.86 | 2.01 |
| 1 | w7x | 1.62 | 1.94 | 2.03 | 2.24 | 2.40 |
| 2 | **toroid** | **1.00** | **1.00** | -- | -- | -- |
| 2 | rot-ell | 1.43 | 1.64 | -- | -- | -- |
| 2 | w7x | 1.50 | 1.76 | -- | -- | -- |
| 3 | toroid | 1.17 | 1.15 | 1.14 | 1.09 | 1.05 |
| 3 | rot-ell | 1.28 | 1.31 | 1.28 | 1.25 | 1.27 |
| 3 | w7x | 1.19 | 1.21 | 1.24 | 1.22 | 1.22 |

Minimax over each degree's cells: **k=1,2 -> 0.10** (worst 1.19);
**k=3 -> 0.15** (worst 1.02). `s = 1` is worst 2.40 at k=1,2 and 1.31 at k=3.

**The three confirmations, all the same direction -- fewer coupling channels
for the scale to compensate, optimum closer to 1 and penalty flatter:**

1. **k=3 vs k=1/2.** k=3 is SINGLE-COMPONENT: no cross-component coupling,
   only angular. It wants 1.5x the scale (0.15 vs 0.10) and is ~2x more
   tolerant of `s=1` (1.31 vs 2.40 worst case).
2. **The toroid at k=1/2 wants `s = 1` EXACTLY** -- cost 1.00 at every
   resolution, both degrees. Its metric is near-diagonal and `g^rr` is constant
   on the face (§2: `<S><P>/<SP>` = 1.0000), so there is almost no within-ring
   coupling for the scale to compensate. **On the one geometry where the atom's
   separability assumption is nearly exact, the DERIVED coefficient is exactly
   right.** That is the cleanest possible confirmation that the factor is a
   coupling-compensation and not a derivation error.
3. **The shaped geometries scale with their coupling** -- rot-ellipse 1.43-2.01,
   W7-X 1.62-2.40, and W7-X (the most strongly coupled metric) is worst
   everywhere.

**Should the default be k-dependent?** Marginal, and the answer is still no.
Global 0.10 gives worst 1.19 (k=1,2) / 1.11 (k=3). Splitting to 0.15 at k=3
improves only the k=3 arm, 1.11 -> 1.02; the overall worst case is unchanged at
1.19 because it is set by k=1/2. So the split buys ~9% on the degree that is
already easiest. Keep the single constant (§16.6.4); note 0.15 at k=3 as a
documented refinement for anyone who wants it.

---

## 19. CYLINDER — the zero-coupling end, and the mechanism CONFIRMED

Added 2026-08-22 at Tobias's request to complete the tables. `cylinder` is now
a geometry choice in `verify_block_jacobi.py`, `block_jacobi_spectrum.py` and
`bc_schur_effective.py`, wired to `cylinder_map(a=0.33, h=1.0)` (`a` matching
the toroid's minor radius, per `benchmark_graddiv_k1_preconditioner.py`).
Additive only -- the existing three geometries are untouched.

The cylinder metric has NO angular variation at all (memory
`laplacian-radial-profile-diagnostic`: angular spread cyl 0% / toroid 24% /
w7x 60%), so it is the zero-coupling limit of §17.5's mechanism.

### 19.1 The n-sweep — the penalty surface is FLAT

k=1 free, p=3 (`outputs/diag_cyl/`):

| n | jacobi | `nobc` | 0.03 | 0.06 | 0.10 | 0.15 | 0.22 | 0.30 | 0.55 | 1.00 | 2.00 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 226 | 93 | 73 | 63 | 56 | 52 | 50 | 49 | 49 | 51 | 57 |
| 12 | 262 | 140 | 97 | 81 | 70 | 67 | 67 | 67 | 66 | 66 | 71 |
| 16 | 308 | 182 | 115 | 93 | 84 | 83 | 83 | 83 | 83 | 83 | 84 |
| 20 | 379 | 224 | 131 | 105 | 99 | 99 | 98 | 98 | 97 | 97 | 98 |
| 24 | -- | -- | 146 | 118 | 114 | 114 | 114 | -- | -- | 114 | -- |
| 28 | -- | -- | 160 | 133 | 132 | 132 | 132 | -- | -- | 132 | -- |

**`s = 1` costs 1.00-1.04 at every k, n and p.** From n=16 up the curve is flat
to within the noise floor across a 20x range in the knob (0.10 to 2.00).

### 19.2 THE SPECTRA — the second error family does not exist here

This is the cleanest confirmation of §17.5 anywhere in this file
(`outputs/diag_bcpspec/`, cylinder 6,12,6 k=1 free):

| scale | 0 | 0.06 | 0.10 | 0.15 | 0.22 | 0.30 | 0.55 | 1.00 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cond | 486 | 61 | 41 | 31 | 26 | 23 | 20 | **19.4** |
| HIGH outliers | 23 | 11 | 5 | 1 | 1 | 0 | 0 | 0 |
| **LOW outliers** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |
| `lambda_min(PL)` | 0.3341 | 0.3340 | 0.3339 | 0.3338 | 0.3336 | 0.3333 | 0.3323 | 0.3274 |

* **ZERO low outliers at every scale, including `s = 1`.** On rot-ellipse the
  same sweep gives 8 -> 27 (§14.1).
* **`lambda_min` moves 2%** over the whole range (0.3341 -> 0.3274). On
  rot-ellipse it falls 60% (0.0662 -> 0.0260).
* **`cond` is MONOTONE DECREASING to `s = 1`** -- there is no interior minimum
  at all, so there is no kappa-balance to strike.

> **The "too stiff" error family is CREATED by the coupling the atom drops.**
> Remove the coupling and it vanishes, the optimum runs to 1, and the derived
> coefficient is exactly right. That is §17.5's mechanism stated as a controlled
> experiment rather than an inference.

### 19.3 The trend, complete over four geometries

Cost of `s = 1` at k=1 (worst / median over n):

| geometry | angular metric variation | worst | median |
| --- | --- | --- | --- |
| **cylinder** | **0%** | **1.04** | **1.00** |
| toroid | 24% | 1.00 | 1.00 |
| rot-ellipse | -- | 2.01 | 1.73 |
| w7x | 60% | 2.40 | 2.03 |

And the §17.5 ring-block predictor now validates on FOUR geometries, ordering
exact and magnitudes within one sweep point everywhere:

| geom | `cond(B_raw, A(s))` argmin, depth 3 | iteration argmin |
| --- | --- | --- |
| cylinder | 0.55 | 0.55 (flat 0.10-2.0) |
| toroid | 0.55 | 0.30-0.55 |
| rot-ell 6,12,6 / 8,16,8 | 0.22 / 0.15 | 0.15 |
| w7x 6,12,6 / 8,16,8 | 0.06 / 0.06 | 0.06-0.10 |

(The cylinder's Schur arm again picks the same or a LARGER scale than the raw
arm -- depth 1: raw 0.55, schur 2.0 -- reconfirming §17.5's refutation of the
DtN account on a fourth geometry.)

### 19.4 The default is unchanged, and now better justified

Minimax over **82 cells, four geometries**:

| band | 0.03 | 0.06 | **0.10** | 0.15 | 0.22 | 0.30 | 0.55 | 1.00 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all cells | 1.76 | 1.45 | **1.19** | 1.23 | 1.36 | 1.37 | 1.74 | 2.40 |
| **shaped only (rot-ell, w7x)** | 1.36 | 1.19 | **1.11** | 1.23 | 1.36 | 1.37 | 1.74 | 2.40 |
| simple only (cyl, toroid) | 1.76 | 1.45 | 1.19 | 1.12 | 1.03 | 1.03 | 1.06 | 1.17 |
| production n >= 16 | 1.63 | 1.27 | **1.11** | 1.23 | 1.36 | 1.37 | 1.74 | 2.40 |

**0.10 still wins, and the case for it is stronger than §16 could make it.**
Its all-cells worst case of 1.19 is set by the SIMPLE geometries -- exactly
where the basin is flat and being wrong is cheap. Restricted to the geometries
that actually matter it is **1.11**, and restricted to production resolution
also 1.11.

The simple geometries would individually prefer 0.22-0.30 (1.03), but they cost
at most 1.19 at 0.10 and typically under 1.05. No reason to split.

Also landed: **W7-X n=32** (`diag_bcphi`) -- k=1 argmin 0.03 (censored),
0.10/best **1.08**, 0.15/best 1.20; k=3 argmin 0.06, 0.10/best 1.01. Consistent
with §16.5: 0.10's penalty on the hard cell stays ~1.1 and does not run away.
