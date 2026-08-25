# Handoff 2026-08-25 — the Poisson convergence studies

Branch `worktree-poisson-k1`, three commits, none on `greville-prod`:
`c2c28ca`, `1fb1e3f`, `4c789e3`.

| # | finding | status |
| --- | --- | --- |
| 1 | The task premise was wrong: **k=1 passed, k=0 failed** | SETTLED |
| 2 | `k0_sparse` never called `assemble_incidence_operators` | **FIXED** |
| 3 | k=1 and k=2 RHS off by one factor of the metric | **FIXED** |
| 4 | k=2 source had the wrong sign in its middle slot | **FIXED** |
| 5 | `nbc_k1` converges at order ~3.3, not 4 | OPEN |
| 6 | The atom's `n >= p+2` floor should raise a named error | OPEN (Stage D) |
| 7 | `debug_poisson_convergence.py` is 6x-duplicated; ruff sees 3 findings | OPEN |
| 8 | fp32/submitit as the cause of the k=0 crash | REFUTED |
| 9 | F811 shadowing as the cause of a k=1 failure | REFUTED |

Final state, all three studies, `n = 6/8/10`, `p = 3`:

```
        k0_sparse      nbc_k1        dbc_k2
 6      2.072e-03    8.564e-03    1.700623e-03
 8      4.771e-04    3.244e-03    4.601128e-04
10      1.737e-04    1.576e-03    1.760652e-04
order        ~4.5         ~3.3     4.55 / 4.31
```

---

## 1. The premise was inverted (SETTLED)

The task was "why did the Poisson runs fail for k=1". From `sacct` on the batch
submitted 04:30:14:

```
16763035  pois_k0_sparse       FAILED     exit 1:0   00:00:27
16763036  pois_nbc_k0_sparse   COMPLETED  exit 0:0   00:06:37
16763037  pois_k1_sparse       COMPLETED  exit 0:0   00:07:29
16763038  pois_dbc_k2_sparse   COMPLETED  exit 0:0   00:11:25
```

k=1 completed cleanly. Only k=0 failed. Infrastructure was ruled out: the failed
job's WorkDir (`.claude/worktrees/poincare`, since deleted) still existed, as
`pois_nbc_k0_sparse` started **5 seconds later from it** and ran 6:37.

## 2. `k0_sparse` never assembled its incidence operators (FIXED, `c2c28ca`)

`ValueError: Incidence operator G0 is required to apply K0`, raised at
`operators.py:2723` from `apply_stiffness` <- `apply_hodge_laplacian_approx`
<- `probe_core_block` <- `BlockJacobiLaplacian.__init__`.

`7cead35` swapped `assemble_tensor_laplacian_preconditioner` for the
block-Jacobi atom. The atom's constructor PROBES the Laplacian and therefore
needs `G0`; the retired tensor path built its factors from the metric alone and
did not. The dependency arrived silently.

**The discriminator**: `k0_sparse` is the ONLY one of the nine
`test_torus_poisson_*` scripts that never calls `assemble_incidence_operators`.
`nbc_k0` builds the same atom, calls it, and completed fine in the same batch
(1.657e-03 / 4.443e-04 / 1.673e-04). Fixed by mirroring `nbc_k0`.

Verified on the exact config the failing job ran (job 16769192): order ~4.5,
CG residual 2.7e-10.

## 3. The k=1 and k=2 RHS were off by a metric factor (FIXED, `1fb1e3f`)

`nbc_k1` and `dbc_k2` both returned a **FLAT** relative L2 error — 3.7256e+01
and 1.7818, unchanged to four figures across n — with the solver reporting
converged.

**The experiment that localised it** (Tobias's suggestion): the two studies are
the same problem, `f0 = cos(2 pi zeta)/R^2` with `f1 = d f0`, `f2 = *(d f0)`,
exact solutions Hodge dual. Running `nbc_k1` showed it fails too. **k=1 free and
k=2 Dirichlet failing alike rules out the boundary conditions** and leaves only
the shared analytic problem.

Three further candidates were eliminated before the RHS:

* the manufactured pair is CORRECT — `omega_1` is closed, so
  `L_1 omega_1 = d(delta omega_1)`, and on the toroid metric
  `delta omega_1 = cos(2 pi zeta)/R^2 = f0` exactly;
* the error metric is self-consistent — computed and exact are pushed forward by
  the same `DF G^-1`, the correct 1-form pushforward;
* the discrete space really is covariant — `M_1 = int Lambda^T G^-1 Lambda J`.

**The defect.** `load` pairs its argument directly against the basis, while
`M_k` carries a metric weight. Recovering a primal form from `M_k^-1 load`
therefore needs the load integrand to be `G^-1 omega` at k=1 and
`g omega / J` at k=2, NOT the bare reference components. The `make_f*_phys`
helpers had been written to INVERT `load`'s internal pullback — their docstrings
said so, *"load applies DF^-1, recovering f1_ref"* — so both frames delivered the
bare components. **That made the two frames agree with each other while both
were wrong**, which is why it looked self-consistent and survived. A metric
factor does not vanish under refinement; hence a flat error with a converged
solver, and the differing factors at k=1 and k=2 are why the two flat constants
differed.

Measured three ways at k=1, reproducing the study's own metric
(`scripts/debug/poisson_rhs_frame_probe.py`, job 16775777):

```
 n     bare (study)     G^-1 (ref)      DF^-T (phys)
 6     3.725597e+01     8.564482e-03    8.564482e-03
 8     3.725945e+01     3.244121e-03    3.244121e-03
 order       -0.00           +3.37           +3.37
```

`bare` reproduces the study to every digit, so the probe is faithful; the two
corrections agree to seven digits despite different code paths.

**`all_k_sparse` was already correct and is UNTOUCHED** — its `make_f1_ref`
applies `G^-1` and its `make_f2_ref` applies `G/J`, with the rule stated
verbatim in its k=2 docstring. It is the reference the two standalone scripts
diverged from. I nearly "fixed" it, having grepped the surface pattern
`DF(x) @ f1r(x)` without checking that its `f1r` already contained `G^-1`.

## 4. The k=2 source had a sign error (FIXED, `4c789e3`)

After §3, `dbc_k2` was still flat, at 1.6796e-01. Two probes, neither needing a
solve:

**(a) The reference is fine.** `scripts/debug/poisson_k2_reference_probe.py`
projects `w2_exact` into V2 and measures it with the study's own error metric:
1.699e-03 / 4.600e-04 at n = 6/8, order 4.54, best-fit scalar exactly 1.000000,
the two unused slots at 3e-17 and 9e-18. So `w2_exact`, the error metric and the
discrete space all agree, and the space represents the exact field at full
order. That exonerates the reference and leaves the source.

**(b) The source is wrong, found with zero compute.** Ratio of `dbc_k2`'s
hand-written `f2` to the WORKING Hodge star in `all_k_sparse`
(`_hodge_star_1to2_ref`, cyclic convention `(J/g_rr, J/g_cc, J/g_zz)`, all
positive), applied to the same `f1`:

```
   r    chi   zeta |     chi-zeta      r-zeta       r-chi
  0.30  0.15  0.20 |   +0.333333    -0.333333    +0.333333
  0.80  0.62  0.35 |   +0.333333    -0.333333    +0.333333
```

exact at four unrelated points. The uniform `eps` is HARMLESS — `w2` carries the
same spurious factor, so it cancels between source and solution, which is why
the pair looked self-consistent and why (a) passed while the study was broken.
The **middle slot's sign** does not cancel: it is the `dr^dzeta` vs `dzeta^dr`
orientation. One character.

Verified (job 16784998): order 4.55 then 4.31 — **and matching the projection
errors from (a) to four digits**, so the solve now sits exactly on the best
approximation the space admits and the RHS and solver add nothing on top. That
is a sharper check than the convergence order.

The spurious uniform `eps` in both `f2` and `w2` is left in DELIBERATELY: it
cancels exactly, so removing it is cosmetic, and folding a rescale into a
correctness fix makes the diff harder to trust.

## 5. OPEN — `nbc_k1` converges at ~3.3, not 4

Real and unexplained. **Under-integration is eliminated**: `dbc_k2` reaches
~4.4 at the same `quad_order_offset: 0`, so `q = 2p` is not the limiter.

*Cheapest next experiment*: run the §4(a) projection test on `omega_1` —
project it into V1 and measure with the study's own metric. If that also comes
back at ~3.3, the ceiling is the space or the reference and the solve is
innocent; if it comes back at 4, the fault is in the RHS or the solve. It needs
no solve and it is exactly what settled k=2. Next suspects after that: the NBC
harmonic deflation, or a subtler defect in the k=1 pair of the same family as §4.

## 6. OPEN — the atom's `n >= p+2` floor (Stage D)

A second, latent defect in the same call as §2, surfacing only when the sweep
dips below `p+2`: `component_factors` goes non-finite and numpy raises
`LinAlgError` from inside `eigvals` (job 16767210 at n=4). It did NOT cause the
observed failure — 16763035 ran `n=[6,8,10]` — and `conf/config_poisson_verify.yaml`
already pins `n: [6, 8, 10]` for it.

The agreed fix is a NAMED precondition error raised where the constraint is
violated, citing the constraint that `operators.py:3467` already documents. Not
a clamp and not a silent bump of `n`. Queued as a Stage D item so it lands once,
after the `block_jacobi -> tensor` rename, since it touches the builder.

## 7. OPEN — `debug_poisson_convergence.py` is heavily duplicated

Not the cause of anything here (it is k=0-only and is not what `pois_k0_sparse`
runs), but a real result about that file: `_require_valid_resolution` is defined
**6 times**, `l2_relative_error` **6 times**, and `u` / `make_f` /
`exact_u_at_quad` **4 times each**, in 1711 lines. Ruff reports only 3 findings
because F811 fires only on the *unused* redefinitions — so the linter looks
clean while the file is mostly copies. Worth a dedicated pass.

## 8-9. REFUTED, with reasons

**fp32 on the submitit worker** (`poisson_convergence_submitit_bug.md`). The doc
is real and its diagnostic ladder is good, but fp32 produces a WRONG ANSWER, not
an exit-1 crash 27 seconds in. It cannot explain the k=0 failure. The related
observation that `config_poisson_verify.yaml` inherits x64 from an unspecified
caller with no warning stands on its own as a fragility.

**F811 shadowing in `debug_poisson_convergence.py`.** Right `k`, wrong reason
and wrong file: that script is entirely k=0 (`ks=(0,)` throughout), so it could
never have caused a k=1 failure — and it is not what `pois_k0_sparse` runs.
`test_torus_poisson_k0_sparse.py` lints clean on F821/F811.

## Traps that generalise

**`frame='ref'` does not take the natural components.** Third and fourth
independent appearance today (the relaxation-IC work, then both of these). Same
shape every time: same slot names, different object, differing by a metric
factor, silent, and it survives because the wrong convention is applied
*consistently* so everything looks self-consistent. Already in
`docs/PRODUCTION.md` under Knobs.

**A round trip that inverts a transform is not a test.** The `make_f*_phys`
helpers were built to undo `load`'s internal pullback, so the two frames agreed
with each other by construction and neither was checked against anything
external. What finally worked was comparing against an INDEPENDENT
implementation — `all_k_sparse`'s Hodge star for §4, and the projection test for
the reference.

**A converged solver with a wrong answer: two modes, and a free discriminator.**
These look identical from the summary line — *"the solver reports convergence
and the answer is wrong"* — and they want opposite responses:

> **Mode A.** The solver passes its OWN criterion while an INDEPENDENT residual
> disagrees. Real; documented in `poisson_convergence_submitit_bug.md`, where CG
> reported 11 iterations and met its M-norm criterion while the recomputed
> Euclidean residual was 4.93e-07 against 8.17e-14. The recheck instinct is
> correct here.
>
> **Mode B.** The solver and the residual AGREE, and the SOURCE is wrong. Both
> studies here. An independent residual recheck comes back CLEAN and sends you
> to the solver, which is the one place the fault is not.
>
> **Discriminator: is the error FLAT IN N?** Any positive weight gives 0 as
> `w_h -> w_ex`, so a relative error that is constant in n proves `w_h` is
> converging to something that is not `w_ex` — which cannot be a solver problem
> at all. Mode A degrades with refinement; mode B does not move.

This costs nothing: it reads off the convergence table you already have, before
reaching for any instrument. It ruled out the error metric as a suspect in both
studies and pointed at the source both times.

**Check what an arm can emit before calling it decisive.** Three arms today were
declared decisive while structurally unable to decide. Reading the source of the
thing meant to produce the evidence — rather than the claim about it — caught
every one.
