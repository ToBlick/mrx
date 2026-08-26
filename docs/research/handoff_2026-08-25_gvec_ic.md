> **Status:** superseded by docs/gvec_mrx_interface.md and relaxation_ic_2026-08-25.md
> **Read this for:** the session record behind the GVEC export contract and the hegna-trust measurements
> **Do not read for:** how to build an IC today; use mrx/gvec.py and mrx/initial_conditions.py

# Handoff 2026-08-25 — GVEC interface, relaxation ICs, hegna trust

Written for redistribution. Companion to `relaxation_ic_2026-08-25.md` §13 (the
other session's side); this covers what THIS session holds. Same shape, so the
two merge.

**Branch warning.** Everything below lives on `worktree-gvec-interface`, which
was branched from `origin/main` (`c0e616a`) because the default `worktree.baseRef`
is `fresh`. It therefore does NOT contain the `greville-prod` work. Two commits
need cherry-picking onto `greville-prod`:

```
ef7191c  docs: the GVEC -> MRX export interface
d92b247  docs: pin the GVEC radial label -- rho = sqrt(s), measured
<this>   docs: handoff
```

---

## 1. Deliverable in hand

**`docs/gvec_mrx_interface.md`** — the GVEC → MRX export contract, written for
an external collaborator producing the re-export. Published (private) at
https://claude.ai/code/artifact/82c70f8b-f2e8-40ea-abaa-b785c757d248 — republish
the same file path to update that URL.

Covers: the premise (our reference 2-form components ARE GVEC's `sqrt(g) B^i`),
the contract identity with measured residuals, the required quantity list in
GVEC `state.evaluate()` names, the derivative-vs-parent storage rule, the
conventions that must be stated in attributes, the eight load-time checks, and
the traps already paid for.

**A re-export is being supplied by the user.** The alternative — reconstructing
`clebsch/` ingredients for `w7x_ini_mrx.h5` from its `B`, `R`, `Z` — is
therefore DROPPED, not merely unstarted.

## 2. Settled — do not re-derive

| result | where verified |
| --- | --- |
| Reference 2-form components = GVEC `sqrt(g) B^i`; the three Clebsch identities | vs hegna's own B: ratio 1.00000000, std 2.9e-13 / 1.7e-16, `B^rho` 3.8e-16 |
| Unit conversions: `Phi' = 2pi dPhi_dr`, `iota = (1/nfp) dchi_dr/dPhi_dr`, `lambda = LA/2pi` | FD of `LA` vs stored derivatives: 6.274 vs 2pi, 2.0905 vs 2pi/nfp |
| **Radial label is `rho = sqrt(s)`**: `Phi(rho) = rho^2/2pi` exactly, `dPhi_dr` linear in rho | `int dPhi_dr d(rho)` vs stored `Phi`: **3.5e-16** |
| `dPhi_dr` is d/d(rho) w.r.t. the SAME rho indexing `eval_points` — no chain rule | same test |
| Storage rule: store the derivative UNLESS two derivatives are tied by an exactness identity, then store the parent | mixed-partial identity on hegna's stored derivatives: 6.6e-3 relative |
| Helicity `H = int Phi^2 W' drho`, metric-free; **zero for constant iota** | closed form vs quadrature 1e-13 over six shapes; `lic_cyl_noshear` returns **exactly +0.000000e+00** on the real solve path |
| `compute_helicity` uses a different gauge — at constant iota its entire value is the harmonic term | same run: +7.81 against analytic 0 |
| The lambda equation: `dW/dlambda = 0` <=> `(curl B).grad rho = 0`; elliptic, decoupled per surface | `lws_toroid`: general solve vs closed form, worst residual **1.30e-08** |
| Fixed-geometry lambda reproduces GVEC's own lambda at **median corr +0.998 / +0.999** | `lws_hegna`, 220 coefficients |
| That residual is MODEL error, not truncation | mpol 8->14, ntor 6->10 (220 -> 608 coeffs): median 0.0606 -> 0.0607, some values worse |
| **Metric-orthogonality criterion**: the L2 route preserves `B^rho = 0` exactly iff `g_rho-chi = g_rho-zeta = 0` | re-established WITHOUT hegna — see §3.1 |
| `load(frame='ref')` wants `g omega / J`, not `omega` — fails SILENTLY | `PRODUCTION.md` Knobs; push forward, use `frame='phys'` |
| Physical histopolation pullbacks were wrong at k=1 and k=2 | `Pushforward` (`differential_forms.py:301`) is the authority: `omega = DF^T v` at k=1, `adj(DF) v` at k=2; code had `DF^-1` and `DF^T`. **Fixed and confirmed**: k=1 went nan -> 1.545e+00 (finite), the axis singularity gone because `DF^T` has no inverse. Build `adj` as cofactors, NOT `det*inv33` — that is `0*inf` at the axis |
| **`E . Pi_full` is NOT a projector**, so MRX's extraction is not the conforming `P_Z` and dropping the guards is NOT sufficient | round-trip (interpolate a function already in the target space, expect its own DOFs): k=0 free **5.290e-01**, k=0 dbc **3.609e-01**. The explicit local `P_Z` port is genuinely required — the ~a-day branch, not the hours branch |

## 3. hegna: audited, found bad, and DELETED

`gvec_nfp3_hegna_80cubed_clebsch.h5` **no longer exists** — the user deleted it
after the audit below. It was the ONLY file with a `clebsch/` group, so there is
currently no test data for the Clebsch reconstruction route at all until the
re-export lands. The audit is recorded not as an action item but because it
determines which conclusions survive its deletion (§3.2).

### 3.1 What survives without it — the criterion, re-anchored

The metric-orthogonality criterion was the session's headline result and was
illustrated only on hegna. It has been re-established from `R`, `Z` alone on
three surviving shaped geometries, so it does not depend on the deleted file:

```
|off-diagonal| / sqrt(g_ii g_jj)        g_rt          g_rz        g_tz
quasr9983  (nfp=2)  rho 0.20 -> 0.86   0.54 -> 0.64  0.18-0.20   0.18-0.21
quasr44970 (nfp=3)  rho 0.20 -> 0.86   0.87 -> 0.91  0.19-0.38   0.18-0.39
w7x-gvec   (nfp=5)  rho 0.20 -> 0.86   0.88 -> 0.93  0.36-0.51   0.33-0.44
cylinder / toroid                      IDENTICALLY ZERO (analytic)
```

Every real device is strongly non-orthogonal in the rho-chi block and grows more
so with radius. So the prediction — the L2 route cannot hold `B^rho = 0` on any
shaped geometry — stands on three files that exist, and `logical_profile_ic.py`
can re-measure the leak itself on any of them WITHOUT any GVEC field data, since
it builds its own analytic profiles.

### 3.2 The audit, and the triage that matters

Audited from the file alone, no MRX involved. **The distrust was justified.**

What is FINE:

* Flux functions are exact. Angular spread of `p`, `iota`, `Phi`, `chi` at fixed
  rho is **1e-16** — machine precision.
  *(This corrects an earlier claim of mine that they varied 0.2-5% over a
  surface. That was my measurement bug: `|rho - r0| < 0.01` with spacing 1/79
  selects TWO radial slices, so I was measuring radial variation, not angular.)*
* `|J| ~ 4.6e5 A/m^2` against `|B| ~ 0.5 T` is consistent with `mu0 J = curl B`
  at device scale, so `J` is a real current density in SI.
* The Clebsch identities hold internally to 1e-13 (§2), so the `clebsch/` group
  and `B` are mutually consistent.

What is BROKEN:

| test | result | expected |
| --- | --- | --- |
| `sin(angle(J, B))`, median rho>0.1 | **0.022** | near 0 only if force-free |
| `\|J x B\| / \|grad p\|`, median | **0.042** | 1 |
| `\|J x B - grad p\| / \|grad p\|`, median | **1.00** | << 1 |
| `2 mu0 p / B^2` at rho=0.1 (raw p) | **1.09** | a few % |

So the field is **nearly force-free**, while the stored `pressure` implies
beta ~ 100% on axis and a gradient **~25x larger** than the Lorentz force can
support.

**`pres_scale_pa` does NOT reconcile it.** The attribute is 4000.0 while
`p(axis) = 1.147e5`, implying a factor 28.7. Applying it makes the force-balance
residual **worse**, 1.00 -> 1.36. So this is not a simple unit or scale error and
I have no resolution.

**Triage — what this does and does not invalidate:**

* **INVALIDATED**: any end-to-end pressure test against hegna, including the
  comparison in `gvec_clebsch_ic.py`. A `p` that the field cannot support is not
  a reference. (Independently, that run's field leaked `B^rho` at 6.8e-3, which
  is on its own too large to test `p` against.)
* **NOT invalidated — pure geometry**: the metric off-diagonals (§2) come from
  `R`, `Z` only. The `B^rho` leak and the metric-orthogonality criterion stand.
* **NOT invalidated — internally consistent**: the Clebsch identities, the unit
  conversions, and the radial label are all checks of the file against ITSELF,
  and hold at 1e-13 to 1e-16 regardless of whether the equilibrium converged.
* **WEAKENED**: `lws_hegna`'s corr +0.998 compares our lambda to GVEC's lambda.
  Both come from this file, so the agreement is real, but "our fixed-geometry
  lambda reproduces a CONVERGED equilibrium's lambda" is no longer supported —
  only "reproduces THIS file's lambda".

### 3.3 ACTION: gate the replacement file on this audit

A replacement clebsch-carrying export is coming. **Run the four tests of §3.2 on
it before trusting anything downstream** — they need only `B`, `J`, `p` and
`grad_rho`, take seconds, and would have caught this on day one:

```
sin(angle(J, B))              near 0 only if force-free
|J x B| / |grad p|            should be ~1
|J x B - grad p| / |grad p|   should be << 1
2 mu0 p / B^2                 should be a few %, not ~1
```

This is why interface doc §2.4 asks for GVEC's own force residual `F`: it makes
the check a single stored number instead of a reconstruction. Worth also asking
the collaborator explicitly whether the export is a converged finite-beta
equilibrium or a force-free case with a decorative pressure profile, and what
normalisation `pressure` needs — `pres_scale_pa` demonstrably was not it.

**The companion doc has the same exposure.** `relaxation_ic_2026-08-25.md` §5,
§5.4 and §7.2 quote hegna numbers throughout (the Clebsch identity residuals, the
lambda magnitudes, the `lws_hegna` correlations, the 6.8e-3 leak). None of those
are reproducible now. They are not WRONG — they were correctly measured on the
file as it stood — but they are unverifiable, and §3.2's triage applies to them
identically.

## 4. Decided against — do not restart

* **Reconstructing `clebsch/` for `w7x_ini_mrx.h5`** — superseded by the user's
  re-export.
* **A designed HTML artifact for internal docs** — user instruction: publish the
  plain markdown.

## 5. Known-broken and cheap

* `gvec_clebsch_ic.py`'s pressure comparison reads raw `pressure` with no scale
  applied. Given §3 the right fix is unclear, but it should not silently compare
  against an unbalanced profile.
* `logical_profile_ic.py:409` normalises the axis/bulk `B^rho` bands by a GLOBAL
  `max|B^zeta|`, while `B^zeta ~ Phi'(rho) ~ rho` is ~7x smaller in the axis
  band. Deliberately left unfixed — its two runs have landed and the cylinder has
  zero metric off-diagonal, so it cannot exercise the mechanism anyway.

## 6. Open, no work started

1. **The toroid's 4.6e-09 `B^rho` leak** with an EXACTLY diagonal metric is
   unexplained. Six orders below hegna's 6.8e-3, so it is a different and much
   smaller effect — not the same mechanism. Candidates: polar extraction,
   conditioning.
2. **Whether histopolation actually removes the hegna leak.** The argument is
   structural (histopolation never forms the dual pairing, so `g_rho-chi` cannot
   enter) but unmeasured.
3. **`aic_tor_vac2` (16764594)** landed but I never read it. It is the corrected
   `--flux vacuum` arm; the prediction on record is that the closed-form force
   collapses while `lambda=0` stays near 3.9e-05, i.e. the ratio flips.
4. **The lambda INVARIANCE arms were never run** — `toroid_lam`, `w7x_lam`, and
   the `--no-lambda` pair. These are the falsifiable test of the §2 claim that
   lambda changes energy and force but never the fluxes, iota or the helicity.
   Cheap, and the most load-bearing untested claim in the whole thread.

## 7. Method notes worth keeping

**Three arms in this session were declared decisive while being structurally
unable to decide**, and I wrote all three: the `tor_vacuum` spec (a correct
lambda paired with a flux profile that made it not-a-vacuum-field), the global
scalar at `gvec_clebsch_ic.py:329` where a radial profile was needed, and a claim
that a test run would print a k=2 number when `interpolate` is never called at
k=2. Roughly a dozen arms were designated in total, so that is a real rate.

What caught all three was the same move: **reading the source of the thing meant
to produce the evidence, rather than the claim about it.** Cheap, and it worked
every time.

**Accuracy tests pass on wrong operators; identity tests do not.** Two
independent instances, one at each end of this thread. The k=0 Greville
interpolation of a smooth function returns 2.225e-02 — comfortably inside the L2
bound — while the very same operator fails its round-trip at 5.290e-01. An
accuracy assertion would have passed and hidden the fact that `E . Pi_full` is
not a projector at all. Symmetrically, a finiteness assertion at k=2 would have
passed on `DF^T v`, which is finite at the axis and also the wrong object. In
both cases the test that discriminates is the one asserting an EXACT identity
the operator must satisfy — idempotency on its own space, or round-tripping a
known primal field — not one asserting that an error is small.

Second note: I twice generalised a geometric property from the two easy maps.
"The metric is diagonal" is true of cylinder and toroid and spectacularly false
of hegna (0.77-0.95). Both times the fix was to measure the shaped case.
