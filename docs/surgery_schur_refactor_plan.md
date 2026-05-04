# Surgery-Schur Mass Architecture

This note records the current mass-side architecture after the surgery-Schur
cleanup.

## 1. Separation Of Roles

The mass-preconditioner stack now treats the extracted-space surgery elimination
and the bulk approximation as separate layers.

- surgery data describes the exact small extracted-space block structure,
- the bulk preconditioner describes how the regular complement is approximated,
- and the final apply is built by wrapping the chosen bulk apply in the outer
  surgery Schur solve.

That is the final architectural split.

## 2. Stored Preconditioner Families

The mass-side payloads are now organized by role.

- Jacobi data stores the extracted inverse diagonal,
- surgery data stores the extracted Schur structure,
- tensor data stores the tensor block factors,
- the operator layer composes them into the final preconditioner applies.

The tensor payloads for `k = 0`, `k = 1`, and `k = 2` no longer try to own the
whole surgery structure. They store the tensor bulk factors, while the surgery
data lives separately.

## 3. Active Public Picture

The practical public picture is now small.

- `jacobi` is the minimal fallback.
- whole-operator Richardson and Chebyshev remain comparison baselines.
- the preferred production tensor route for `k = 0`, `k = 1`, and `k = 2`
  uses the outer surgery Schur wrapper.
- `k = 3` uses the direct tensor route with no surgery wrapper.

Under that outer wrapper, the active tensor bulk models are:

- `k = 0`: one scalar tensor bulk block,
- `k = 1`: tensor bulk blocks on `r`, `theta_bulk`, and `zeta_bulk`, with an
  optional inner coupled bulk Schur,
- `k = 2`: tensor bulk blocks on `r_bulk`, `theta`, and `zeta`, with an
  optional inner coupled bulk Schur.

The practical finding from the current benchmark set is that the optional inner
bulk Schur for `k = 1` and `k = 2` is useful as a comparison mode, but not the
default runtime winner on the tested geometry family.

## 4. Polynomial Wrappers

Richardson and Chebyshev still exist as wrappers around simpler mass
preconditioners, but the current benchmark conclusions are:

- whole-matrix Chebyshev on Jacobi is a useful baseline,
- Chebyshev wrapped around already-strong tensor routes often reduces
  iterations but usually does not improve runtime,
- nested expensive inner iterations are not the preferred practical policy.

So the settled mass-side design favors simple leaves plus exact outer Schur
structure over deeply nested iterative inners.

## 5. Final Summary

The final surgery-Schur story is:

- outer extracted-space Schur is a reusable wrapper,
- tensor bulk models are separate payloads,
- `k = 1` and `k = 2` may optionally use an additional coupled bulk model,
- but the current production recommendation is the cheaper diagonal tensor bulk
  route unless later benchmarks show a robust need for the coupled option.
