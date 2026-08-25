# Interpolation and histopolation for all k — 2026-08-25

Task: continue the histopolation diagnosis from `fd39a81`, and provide working
interpolation/histopolation operators at **k = 0, 1, 2, 3**.

Branch `worktree-histopolation-ops`. The operators all *existed*; none of them
worked on the sequences MRX actually builds, because the code had never
executed — two guards blocked every polar and every Dirichlet case, so four
`xfail(strict)` tests hid the fact that nothing behind them was correct.

**Three distinct defects were found. Two are fixed and one is not.**

---

## 1. Status at a glance

| # | finding | status |
| --- | --- | --- |
| A | **Physical pullbacks were the wrong objects** (k=1, k=2) | **SETTLED — measured** |
| B | **Extraction is non-biorthogonal**; needs `(E E^T)^-1` explicitly | **FIXED at k=0, measured**; k>=1 pending |
| C | **Periodic Greville spans are unsorted** → negative-width spans | **FIXED, verification pending** |
| D | **Even-p Greville spans straddle knots** → quadrature inexact | **CONFIRMED, NOT FIXED** |
| E | The 1cf9cbd justification for removing the guards | **RETRACTED** (see §6) |

Everything marked pending rests on job **16770362**, the first run with B and C
both in place, over p=2 and p=3 x k=0..3 x free/dbc.

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
that point for reasons C and D, so B's sufficiency at k>=1 is what 16770362
decides.

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

## 5. CONFIRMED BUT NOT FIXED: even-p spans straddle knots (defect D)

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
order**. `Pi_1D` is then not idempotent and no restriction can repair it.

**THE FIX IS NOT APPLIED.** It is: split each Greville span at its interior
knots and apply the Gauss rule per sub-interval. Exact for piecewise polynomials
at any p and any parity, confined to the span quadrature, and only marginally
more expensive (one extra sub-interval per span).

**This is required, not optional.** `conf/config_relax_from_nfs.yaml` sets
`ps_r/ps_theta/ps_zeta = 4` at lines 19-21 and 30-32 — a production relaxation
config at EVEN p. (`config_stell.yaml` is 3; those are the only `ps_*` entries.)
Whether that driver actually reaches Greville histopolation is **unsettled** —
its ingest goes through `interpolate_B` / `load_grid_field`, which use
collocation (`n_basis = n_data`), so it is *probably* safe. Do not let that
"probably" be rounded up.

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

## 7. Method notes — these cost real time this week

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
* **Check what a run can EMIT before calling it decisive.** Twice an arm was
  designated decisive while structurally unable to decide: one whose input
  parameterisation could not express the intended field, one whose output was a
  scalar where a profile was needed. A third was aborted by `pytest -x` before
  reaching the deciding test.

---

## 8. What is left

1. **Read 16770362.** It decides whether B+C are sufficient. If k=3 still fails,
   defect D is the next suspect — the fixture includes p=2 and k=3 histopolates
   all three axes at once.
2. **Implement defect D's fix** (split spans at interior knots). Required
   regardless of (1), because of the p=4 production config.
3. **Tighten the accuracy tolerances.** They are still the `< 1.0` placeholders
   from the file's first commit and are nearly vacuous. Once the operators are
   correct, set them from measured values.
4. **Settle whether `config_relax_from_nfs` reaches Greville histopolation.**
   Per call site, not by inference.
5. The `frame='phys'` convention for `interpolate` at k=3 is unresolved; its
   histopolation carries no Jacobian factor, so `frame='ref'` is rejected there
   rather than defined.
