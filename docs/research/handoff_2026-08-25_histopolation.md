> **Status:** resolved; leftovers in OPEN.md 3.5
> **Read this for:** why interpolation/histopolation never ran on polar or DBC spaces and how the parity fix works
> **Do not read for:** the spline boundary-evaluation defect of section 10; the local-support evaluator of 2026-08-26 removed it

# Interpolation and histopolation for all k — 2026-08-25

Task: continue the histopolation diagnosis from `fd39a81`, and provide working
interpolation/histopolation operators at **k = 0, 1, 2, 3**.

Branch `worktree-histopolation-ops`. The operators all *existed*; none of them
worked on the sequences MRX actually builds, because the code had never
executed — two guards blocked every polar and every Dirichlet case, so four
`xfail(strict)` tests hid the fact that nothing behind them was correct.

> # *** RESOLVED 2026-08-25: exact at BOTH parities ***
> **The operators are EXACT at every k = 0,1,2,3, both BCs, at EVEN and ODD
> `p`** (round-trip ~1e-16; table in §7).
>
> The even-p defect (F) was the periodic seam: at even `p` the Greville points
> sit at half-knots, so the last sorted periodic span is `[1 - h/2, 1 + h/2]`
> and crosses `x = 1`.  The moments wrap their points and integrate the
> periodic extension; `histopolation_matrix` evaluated the basis UNWRAPPED
> there, where `SplineBasis.evaluate` is not periodic (the image of basis
> function `p-1` is missing from the extended knot vector and comes back as
> zero).  So H and the moments shared the RULE but not the INTEGRAND -- the
> loophole in the "same rule => m = H c" argument.  Fix: wrap the matrix's
> quadrature points.  See §7.
>
> **PRODUCTION EXPOSURE.** `conf/config_relax_from_nfs.yaml` runs `ps = 4`
> (even).  Checked repo-wide: the only files that reach `histopolation_matrix`
> / `greville_spans` are `mrx/projectors.py`, `mrx/spline_bases.py` and their
> two test files.  `relax_from_nfs.py` ingests via `interpolate_B` /
> `interpolate_map_from_points` in `mrx/io.py`, which uses `collocation_matrix`
> only.  So no production path reached the defect while it was open.
>
> **`interpolate` IS called in production -- at k = 0.** `mrx/geometry.py:380`
> and `:425-426` build the R/Z spline maps with `seq.interpolate(..., 0)`. That
> is pure collocation, never forms a span, and was exact at both parities
> throughout (2.672e-16 at p=2).  It does now route through
> `_conforming_restriction`, an improvement rather than a risk: k=0 accuracy
> went `2.225e-02 -> 1.540e-02` and the round-trip `5.290e-01 -> ~2.7e-16`.
> Flagged because map construction goes through it.

**Five distinct defects were found (A-D, F). All are fixed.**

---

## 1. Status at a glance

| # | finding | status |
| --- | --- | --- |
| A | **Physical pullbacks were the wrong objects** (k=1, k=2) | **SETTLED — measured** |
| B | **Extraction is non-biorthogonal**; needs `(E E^T)^-1` explicitly | **FIXED, measured at k=0..3** |
| C | **Periodic Greville spans are unsorted** → negative-width spans | **FIXED, verified** (16776655) |
| D | **Even-p Greville spans straddle knots** → moments inexact | **FIXED** (accuracy 3.272e-01 → 2.960e-01) |
| F | **Even-p, k>=1 is not a projector (7e-2 … 1.3e-1)**: periodic spans cross the seam, H evaluated unwrapped | **FIXED, measured** (§7) |
| E | The 1cf9cbd justification for removing the guards | **RETRACTED** (see §6) |

All of A-F are measured. Evidence: jobs 16770362 (B+C), 16773716 (B+C+D) and
16776655 (B+C+D+F), all over p=2 and p=3 x k=0..3 x free/dbc; 16776655 also
carries the 1-D seam discriminator (§7).

---

## 2. SETTLED: the physical pullbacks (defect A)

`Pushforward` (`differential_forms.py:301`) is the authority:

```
k=1   F_* omega = (DF^T)^-1 omega   =>   omega = DF^T   v_phys
k=2   F_* omega = DF omega / J      =>   omega = adj(DF) v_phys
```

The code had `DF^-1` at k=1 — that is the **k=-1 vector-field rule**, off by
`G^-1` — and `DF^T` at k=2, which is `load`'s **dual** pairing, off by `g/J`.
Both are correct *for `load`*, because `M_k^{-1}` converts the dual result back
to primal. Histopolation has no mass solve, so reusing them silently returned
the wrong object. It passed every structural gate: being off by a metric factor
is still finite, smooth and divergence-free-looking.

**Measured, `test_phys_pullback_inverts_pushforward`:**

```
k=1 phys pullback vs Pushforward inverse:  2.883e-16
k=2 phys pullback vs Pushforward inverse:  2.147e-16
k=0 / k=2 non-finite DOFs:                 0 of 162 / 0 of 444
```

`adj(DF)` must be built from **cofactors** (`differential_forms.adj33`), not as
`det * inv33`, which is `0 * inf = nan` exactly on the polar axis — the point
that motivated using it. `adj33` was added additively; `inv33` now returns
`adj33(mat)/det`, same formula, identical behaviour.

This finding is closed and independent of everything below.

---

## 3. FIXED: the extraction is not a projector (defect B)

`e @ c_full` alone is **not** a projector. Measured
(`scripts/debug/extraction_unitarity_probe.py`):

```
k=1 polar   ||E E^T - I||_max = 1.556    30 of 606 rows    ||P^2-P|| = 1.500
k=2 polar   ||E E^T - I||_max = 0.352    12 of 588 rows    ||P^2-P|| = 0.036
k=3 polar   ||E E^T - I||_max = 0.000     0 rows           pure selection
```

The violating rows are exactly the polar ring rows; every other row is a clean
0/1 selection. `E E^T` is therefore **identity plus small dense blocks** — one
per zeta slice per affected component, of size `n_polar`, because the polar
surgery acts only in (rho, theta) while the zeta index rides along. The
component sizes predict the counts exactly (`2*nz + 3*dz = 30` at k=1,
`2*dz = 12` at k=2, none at k=3).

So the correct restriction is `a = (E E^T)^{-1} E c_full`, with the blocks
inverted **densely** — the same separable-bulk-plus-dense-core idiom as
`BlockJacobiMass`, and for the same reason: an `E+` pseudoinverse is what that
analysis rejected. `_conforming_restriction` builds the full `E E^T` from the
extraction's COO triplets and uses `connected_components`, so it assumes no
per-component structure and handles any extraction; for a pure selection it
returns immediately.

**Measured: k=0 round-trip 5.290e-01 -> 2.796e-16.** k>=1 was still failing at
that point, for reason F; once that was fixed the same restriction carried
k=1,2,3 to ~1e-16 at both parities (§7), so B is sufficient at every k.

---

## 4. FIXED: periodic Greville spans are unsorted (defect C)

`greville_points()` applies `mod(., 1)` to the periodic abscissae, wrapping
those outside `[0,1)` and **destroying their monotonicity for every p >= 2**.
`greville_spans()` then does `roll(points, -1)` plus a single `+1.0` patch on
the last element — a construction that repairs exactly ONE wrap and is only
correct if `mod` moved at most the final point.

```
p=1 n=6:  sorted=True    widths [+0.1667, +0.1667]
p=2 n=6:  sorted=False   widths [-0.8333, +1.1667]   BROKEN
p=3 n=6:  sorted=False   widths [-0.8333, +1.1667]   BROKEN
p=4 n=6:  sorted=False   widths [-0.8750, +1.1250]   BROKEN
```

One span of **negative width** and one wider than the whole period: the spans
do not tile the domain and the histopolation matrix is assembled over nonsense
intervals.

**Every MRX sequence is `("clamped","periodic","periodic")`**, so this hits both
angular axes of every sequence we build, at every `p >= 2`, whenever
histopolation is used. It is a production defect, not a test artefact.

Fix: sort before forming spans, periodic branch only. Safe because the spans and
the moments both come from that one function, so they take the same permutation.

**The permutation cancels, and that was checked rather than assumed.** `H[k,i]`
has rows indexed by SPAN and columns by BASIS; `m[k]` is indexed by span; so
`solve(PH, Pm) = (PH)^-1 P m = H^-1 m`. The coefficients are indexed by the
columns, untouched. And the sort is **never the identity** — a pure cyclic
rotation in every case (`perm=[1,2,3,4,5,0]` at p=2 n=6), so the concern
genuinely arises and is genuinely handled, rather than being vacuously absent.
The index-consuming backstop is `test_k1_histopolation_error_is_small`, which
goes through `Pushforward` and would blow up on a permuted DOF vector.

---

## 5. FIXED: even-p spans straddle knots (defect D)

A Greville point of degree p is the mean of p consecutive knots. On uniform
knots, `g_i = i + (p+1)/2` — an **integer for odd p** (lands on a knot) and a
**half-integer for even p** (lands exactly midway).

```
 p   straddling  quadrature  offset/h
 1        0        exact       0.000
 2       10      INEXACT       0.500
 3        0        exact       0.000
 4       11      INEXACT       0.500
 5        0        exact       0.000
 6       13      INEXACT       0.500
```

For even p **every** span straddles an interior knot. Gauss-Legendre is exact
for *polynomials*; a spline is only *piecewise* polynomial, with a derivative
jump at each knot, so a single rule spanning one is inexact **at any quadrature
order**. Measured on an off-centre knot: 40 Gauss points still leave 3.6e-07,
while splitting at the knot is exact at 2 points.

**FIXED** — `_interval_rule` and `SplineBasis.histopolation_matrix` both split
every span at the knots it contains, so `H` and the moments keep an identical
rule. Effect: `k=1` histopolation accuracy `3.272e-01 -> 2.960e-01`.

**This did NOT fix the identity failures, and was never going to.** See §7.

(Those two paragraphs above are the whole of defect D. The half-knot offset
tabulated here is also what puts the last PERIODIC span across the seam, which
is defect F — one geometric fact, two independent consequences: an accuracy one
on every axis, and an identity one on the periodic axes only.)

(`conf/config_relax_from_nfs.yaml` runs at EVEN `p = 4`; `config_stell.yaml`
at 3, and those are the only `ps_*` entries in the repo. Production reach is
settled in the header banner.)

---

## 6. RETRACTION (defect E)

`1cf9cbd`'s commit message, and an earlier version of the comment in
`projectors.py`, said idempotency "comes from the coefficient rules being
self-consistent, not from any biorthogonality condition on the extraction."

**That is true of the paper's `P_Z` and false of MRX's extraction.** The claim
was imported across an operator boundary it does not cross — MRX's `E` is not
`P_Z`, as `||E E^T - I|| = 1.556` shows. The biorthogonality question was the
right one after all.

The same retracted claim was the stated justification for removing the two
guards. Removing them still looks right on the k=0 evidence, but **the reason
recorded at the time was not sound**. History is not rewritten; the retraction
is in `projectors.py` at the site, above the guards it justified.

---

## 7. FIXED: even-p periodic spans cross the seam (defect F)

**Before (jobs 16770362 / 16773716) and after (job 16776655):**

```
                 p = 3 (ODD)                    p = 2 (EVEN), before         p = 2 (EVEN), after
k=0   7.677e-16 / 5.375e-16   PASS     2.672e-16 / 2.619e-16   PASS     3.008e-16 / 2.649e-16   PASS
k=1   5.198e-16 / 6.131e-16   PASS     7.111e-02 / 6.755e-02   FAIL     2.248e-16 / 3.135e-16   PASS
k=2   4.556e-16 / 3.870e-16   PASS     8.626e-02 / 1.137e-01   FAIL     2.689e-16 / 2.247e-16   PASS
k=3   3.630e-16 / 5.234e-16   PASS     1.026e-01 / 1.293e-01   FAIL     2.977e-16 / 3.471e-16   PASS
```
(free / dirichlet.  The p=3 column is from the earlier job; in 16776655 the
p=3 cases stayed exact -- 7.402e-16 / 4.673e-16, 3.837e-16 / 5.297e-16,
3.341e-16 / 3.831e-16, 4.198e-16 / 3.812e-16 at k=0..3 -- so the wrap costs the
odd-p path nothing.  `test_pi_full_is_idempotent` on the non-polar,
identity-extraction fixture: 2.849e-16 / 2.571e-16 / 2.761e-16 / 2.392e-16 at
k=0/1/2/3, all PASS.  Whole file: **32 passed, 0 failed** in 1:06:34, against
9 failed / 23 passed before.)

### The mechanism

The periodic Greville point of degree `p` is `g_i = (i + (1-p)/2) / n`: ON a
knot for odd `p`, HALFWAY between knots for even `p`.  After the wrap-and-sort
of defect C the spans tile `[g_0, g_0 + 1)`.  For odd `p`, `g_0 = 0` and every
span lies in `[0, 1]` (up to rounding -- see below).  For even `p`,
`g_0 = h/2` and the last span is `[1 - h/2, 1 + h/2]` -- it **crosses the
period seam**.

Over that span the integrand must be the PERIODIC extension of the basis.  The
moments get it: every `_interval_rule` site feeds a pullback that calls
`_wrap_periodic_point`, so `f` is evaluated at `x mod 1`.
`histopolation_matrix` did not: it evaluated `self(x, i)` at the raw `x > 1`,
and `SplineBasis.evaluate` is NOT periodic there.  It folds only the `p'` raw
functions `n .. n+p'-1` that exist in the extended knot vector; the raw image
of basis function `p'` (support `[0, (p'+1)h]`) would be index `n+p'`, which
does not exist, so on `(1, 1 + h/2]` that basis function evaluates to ZERO.

So `H` and the moments shared the quadrature RULE but not the INTEGRAND.  That
is precisely the loophole in the linearity argument of §5: "same rule =>
`m = H c`" assumes the rule is applied to the same functions on both sides.

**1-D discriminator (job 16776655, `histo_seam_check.py`)**, `p=2, n=4`, de
Rham ground truth `H dc = s(b) - s(a)` with `s` evaluated periodically at the
span endpoints:

* unwrapped `H`: seam row off by **1.250e-01**, every other row ~1e-16.  That
  number is analytic: the dropped `D_1 = 16 x` integrated over `(0, 1/8]` is
  `8 (1/8)^2 = 0.125`.  `solve(H, ds) - dc = 1.717e-01`.
* wrapped `H`: `1.110e-16` on every row; `H` equals the wrapped moments to
  `1.110e-16`; `solve(H, ds) - dc = 3.331e-16`.
* the same holds at `n = 6, 8` and `p = 2, 4, 5`: unwrapped seam-row defects
  `1.25e-01 / 2.6e-03 / 8.3e-03` (p = 2 / 4 / 5), wrapped ~1e-16 throughout.

**The odd-p exactness was a rounding accident.**  At `n = 6, p = 3` the
Greville point that should be exactly `0` came out as `-eps`, wrapped to
`1 - eps`, and the last span became `[1 - eps, 1 + 1/6]` -- across the seam
at ODD p.  The unwrapped matrix was then wrong by `8.333e-02` at `p = 3`, and
`8.333e-03` at `p = 5`.  At `n = 4` and `n = 8` the knots are exact binary
fractions, the point is exactly `0`, and odd p never crosses -- which is why
the `(4,4,4)` fixtures passed at p=3.  Any production resolution that is not
a power of two would have hit this at odd p as well.  With the wrap the
matrix is exact regardless of where rounding puts the seam.

Every measured constraint from the earlier hunt follows: it is a parity effect
in `p` (only even `p` crosses the seam), it scales with the number of
histopolated axes (one seam row per periodic histopolated axis), it survives a
correct extraction and exact quadrature (neither touches the integrand), and
`k = 0` is immune (collocation points are wrapped, no spans).  The radial axis
is clamped and never crosses.

### The two mechanisms this replaced, and what killed them

Both were pursued for a day; neither is recoverable from the code, because
neither left a diff.

* **The extraction (`(E E^T)^-1 E`) is still incomplete at k >= 1.** REFUTED by
  k = 3, which was named as the discriminator *before* the run: at k = 3 the
  extraction is a pure selection, `||E E^T - I||_max = 0.000`, 0 violating rows
  (§3) — and k = 3 failed anyway, at 1.026e-01 / 1.293e-01. An operator that
  is exactly the identity cannot be the defect.
* **The quadrature is inexact because even-p spans straddle knots.** REFUTED by
  its own fix: splitting every span at its knots made the round-trip slightly
  WORSE, 5.736e-02 -> 7.111e-02 at k = 1. It should have: `solve(H, m) = c`
  needs only `m = H c`, which holds by LINEARITY whenever `H` and the moments
  share a rule — exact or not. Exactness was never what the identity depended
  on. (The split was kept: it is a genuine ACCURACY fix, defect D.)

### Where the earlier reasoning went wrong

The refutation of mechanism 2 was correct as far as it went -- exactness was
not the issue -- but the linearity argument was then held as a proof that the
rule could not be the problem at all.  It assumed the integrand was the same
on both sides.  The moments wrap; the matrix did not.  The column-by-column
comparison suggested at the end of the earlier §7 found it in one 1-D run.

### The fix

`SplineBasis.histopolation_matrix` reduces its quadrature points `mod 1` for
periodic bases (one line; the comment at the site records the mechanism), and
`greville_spans` documents that its last span crosses the seam at even `p`.
`test_histopolation_de_rham_periodic` in `test_spline_bases.py` asserts the de
Rham commutation on the periodic axis at `p = 2, 3, 4` against the periodic
ground truth -- it fails on the seam row without the wrap and needs no
sequence.  The alternative, making `SplineBasis.evaluate` itself periodic in
`x`, was NOT done: it would touch every assembly path for a defect confined to
one consumer, and it changes the value of the `p = 0` periodic basis at
`x = 1.0` exactly.

---

## 8. Method notes — these cost real time this week

* **Accuracy tests pass on wrong operators; identity tests do not.** k=0
  Greville interpolation of a smooth function returned 2.225e-02, comfortably
  inside the L2 bound, while the *same* operator failed its round-trip at
  5.290e-01. Assert the exact identity the operator must satisfy, not an error
  bound.
* **Put the discriminator IN the suite, not in a one-off run.** Parametrising
  the fixture over p=2 and p=3 is what refuted "even-p parity explains
  everything" — k=1 fails at p=3 too, which is what forced the second-defect
  conclusion. A single p=3 confirmation run would have been read as agreement.
* **Commit to what a result will mean BEFORE it lands.** k=3 was named in
  advance as the discriminator (`E E^T = I` exactly, so a k=3 failure cannot be
  the extraction). It then failed, in two independent runs.
* **A test that has never run has no known cost, and an unknown cost is
  indistinguishable from a hang.** A 9-minute test was read as non-termination
  by a separate gate and nearly cost a GPU investigation. Cost here is
  `O(n^2 q^d)` — `DiscreteFunction` evaluates all `n` basis functions per point,
  inside a `q^d` quadrature. Identity tests do not need resolution; they run on
  a `(4,4,4)` fixture.
* **Always use `--output=...%j.log` for slurm jobs.** Three concurrent runs of
  one script wrote to one path and clobbered each other. The log was
  *unreadable*, not wrong — nothing errors, you simply cannot trust what you
  read.
* **`faulthandler_exit_on_timeout` on JAX code manufactures fake segfaults.** It
  kills the process mid-compilation and the stack lands in JAX tracing. Use
  `faulthandler_timeout` alone, which dumps without killing.
* **Evidence can outrank a proof you already hold — do not let it.** The span
  split was run despite an argument, already made and written down, that it
  could not fix the identity: `solve(H, m) = c` needs only `m = H c`, which
  holds by LINEARITY whenever `H` and the moments share a rule, exact or not.
  A strong empirical signal (parity) was allowed to override reasoning already
  in hand. This is a different failure from insufficient evidence, and harder
  to notice: the measurement looked like it was leading, when it was actually
  overruling something settled. When a measurement contradicts a proof you
  have, find the flaw in the proof FIRST — do not just follow the data.
* **Check what a run can EMIT before calling it decisive.** Twice an arm was
  designated decisive while structurally unable to decide: one whose input
  parameterisation could not express the intended field, one whose output was a
  scalar where a profile was needed. A third was aborted by `pytest -x` before
  reaching the deciding test.

---

## 9. What is left

1. **Tighten the accuracy tolerances.** They are still the `< 1.0` placeholders
   from the file's first commit and are nearly vacuous. Now that the operators
   are exact at both parities, set them from measured values.
2. The `frame='phys'` convention for `interpolate` at k=3 is unresolved; its
   histopolation carries no Jacobian factor, so `frame='ref'` is rejected there
   rather than defined.
3. Whether `SplineBasis.evaluate` should be periodic in `x` for periodic bases
   (see §7, "The fix") is a design question, not a defect: nothing else feeds
   it points outside `[0, 1]`.
4. **The closed-last-piece convention (§10, "Proposed fix") is offered and not
   applied.** It touches every assembly path, so it wants its own commit behind
   a full GPU suite run. It would also let `block_jacobi_laplacian.py` drop its
   `end = 1.0 - 1e-8` in `_edge_vector` and `face_operator` (relative error
   ~`(p-1) * 1e-8 / h`) and take the true face metric instead of the last Gauss
   node (`field[-1]`, an O(h) bias currently absorbed by the fitted
   `bc_scale = 3.0`).
5. **`block_jacobi_laplacian.py:682`'s docstring attributes the 4-6x
   free-vs-dbc iteration lag to `det(DF) = 0` at the last knot. That looks
   wrong** — §10 shows the zero is an autodiff artefact at `x = 1.0` exactly,
   and quadrature never samples `rho = 1`. Treat the lag as unexplained.

---

## 10. Spline evaluation at knots and boundary points (measured, NOT fixed)

Asked after §7, because that bug was a boundary-evaluation defect.  Job
16778427 (`spline_boundary_check.py`), clamped `n = 6`, `p = 1..4`, and
periodic `p = 1..3`.  Every number below is measured.

**Interior knots are fine.** Values and `jax.grad` agree from the left, the
right, and exactly at the knot for all `p` (p=1 gets the right-derivative,
which is the only possible convention).  Partition of unity holds everywhere.

**The clamped right boundary `x = 1.0` EXACTLY is a dead point for autodiff.**
`_const_spline` is half-open `[t0, t1)`, so at the last knot every degree-0
piece selects the zero branch of its `jnp.where`.  The VALUE is rescued by the
`i == n-1 and x == T[-1]` patch in `evaluate` -- but a patch is a constant:

```
p=2   grad at 1.0     : [0 0 0 0  0  0]
      grad at 1-1e-12 : [0 0 0 0 -8  8]     = left finite difference
```

So **every clamped basis function has gradient 0 at `x = 1.0`**, for every
`p`.  That is the whole content of the "spline map `DF` singular at the outer
knot, `det DF = 0` at `rho = 1`" finding of 2026-08-19: the map is fine, its
`jacfwd` at `rho = 1.0` is structurally zero because of this.  The
`1 - 1e-7` workaround in the eval scripts, and the clipping of clamped
Greville points to `[eps, 1 - eps]`, were both treating a symptom of this.

**`DerivativeSpline` at `x = 1.0` is wrong, not just undefined.**  Its inner
basis `s` has `n - 1` functions on the parent's knot vector, so the patch
fires on index `n - 2` where index `n - 1` is the one that is nonzero at the
wall, and it fires with the wrong scale:

```
p=2   Dspl at 1.0     : [0 0 0 4 0]
      Dspl at 1-1e-12 : [0 0 0 0 8]
```

The docstring says "cannot be evaluated at a clamped boundary"; it returns a
plausible-looking wrong vector rather than failing.  `x = 0.0` is correct for
both bases (the first piece is closed on the left).

**Periodic bases are not periodic in `x`.**  Values at `x = 1.0` fold
correctly, but outside `[0, 1]` the image of basis function `p` is missing
(`value at 1.05` lacks the `0.3 / 0.045 / 0.0045` entry that `0.05` has at
p = 1/2/3) -- the mechanism of §7.  At `p = 1` the gradient at `1.0` is also
missing its `+n` entry (`[-6, 0, ...]` vs `[-6, 6, ...]` at `0.0`); `p >= 2`
is fine there because the true derivative of that function vanishes at 0.

**Stability.** `_safe_divide` (`isclose(y, 0)` with a dummy denominator of 1
in the zero branch) is the correct NaN-safe pattern for the repeated end
knots and survives `jacfwd(jacfwd(.))`.  The Cox-de Boor recursion in
`_p_spline` re-evaluates the lower-degree pieces `2^p` times; at `p <= 5`
that is cost, not accuracy.  The only exact float comparison, `x == T[-1]`, is
against a knot that is exactly representable, so it does what it says.

**Proposed fix (one place, deletes a patch).** Make the LAST piece closed:
in `_const_spline`, `t0 <= x < t1` becomes `t0 <= x <= t1` when `t1` is the
final knot `T[-1]`, and the `i == n-1 and x == T[-1]` patch in `evaluate` is
deleted.  Then the value at `1.0` comes from the polynomial branch, the
gradient is the left-derivative (the finite-difference value above), and
`DerivativeSpline` needs no special case because its inner basis inherits the
closed last piece.  For periodic bases `T[-1] = 1 + p h` is never reached, so
they are unaffected.  Periodicity in `x` for periodic bases is a separate
design choice (§9 item 3).  Neither is applied here: this section is an
assessment, and the fix touches every assembly path.
