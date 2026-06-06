---
name: mrx-coding
description: 'Write and modify code in the MRX package. Use for MRX operator, solver, preconditioner, JAX, sparse assembly, and validation work in this repository.'
argument-hint: 'What MRX code change do you want to make?'
---

# MRX Coding

Use this skill when working on code in the MRX repository, especially under `mrx/`, `scripts/`, `test/`, and `docs/`.

## What This Skill Covers

- Matrix-free operator and solver changes
- Mass and Hodge-Laplacian preconditioner work
- JAX-based numerical code and sparse assembly paths
- Debug scripts and focused validation for MRX behavior
- Developer documentation that explains current solver policy

## Project Cues

- MRX is a Python and JAX codebase for 3D MHD equilibrium solves.
- Full 3D assembly and validation checks are expensive; treat large sequence builds as costly, not routine.
- Production inverse operators are applied matrix-free.
- Scalar problems and mixed saddle-point problems use different Krylov paths.
- Krylov-in-Krylov solves are not a valid default tactic here.
- Prefer interactive notebook-style diagnostics when exploratory work is needed.
- If a Python check is more than a trivial CPU-only sanity check, prepare the snippet clearly and hand it to the user to run on GPU.
- Existing docs are usually the fastest source of local design intent before widening exploration.

## Procedure

1. Anchor the task locally.
   Start from the file, symbol, failing script, solver wrapper, or doc section closest to the requested behavior. Prefer the code that directly computes or applies the operator over wiring layers.

2. Read only enough context to form one local hypothesis.
   Identify the smallest controlling code path and one cheap check that could falsify the current understanding. Avoid broad repo mapping before the first edit.

3. Preserve MRX solver structure.
   Keep the scalar versus saddle-point split intact unless the task explicitly changes solver policy. Treat user-facing preconditioner specs separately from internal callable applies.

4. Respect existing numerical conventions.
   Prefer matrix-free changes when the current path is matrix-free. Do not introduce dense or assembled shortcuts unless the task requires them. Keep JAX-compatible array and operator behavior intact. Do not propose Krylov methods whose inner actions are themselves unresolved Krylov solves.

5. Make the smallest grounded edit.
   Prefer a focused change in the controlling implementation or its nearest test or doc. Avoid unrelated cleanup or API churn.

6. Validate immediately after the first substantive edit.
   Run the cheapest focused check available: a narrow test, a targeted debug script, or a limited error check on the touched file. Treat full 3D builds as expensive. If the useful validation requires nontrivial Python execution, package it for the user to run and wait for the result before making further numerical claims.

7. Close with evidence.
   Summarize what changed, what was validated, and any remaining risk around numerical accuracy, nullspaces, preconditioners, or performance.

8. Prefer notebook workflows for exploratory diagnostics.
   When a task benefits from interactive investigation, favor a compact VS Code notebook or notebook-ready snippet over an ad hoc throwaway script.

## Decision Points

- If the change touches `M_k`-style mass behavior, check whether the intended path is `jacobi`, `tensor`, or `auto` and whether assembled tensor data is expected to exist.
- If the change touches `L_0`, distinguish unshifted singular solves from shifted solves. Do not treat harmonic handling as the same problem.
- If the change touches `L_k` for `k = 1, 2, 3`, assume the mixed saddle-point MINRES path unless nearby code or docs show a deliberate exception.
- If a proposed preconditioner or reformulation would require solving an inner Krylov problem inside an outer Krylov iteration, reject that route and choose a well-defined approximate apply instead.
- If a bug appears in extracted-space behavior, inspect extraction and surgery effects before blaming tensor approximations globally.
- If documentation and code disagree, fix the code path first when behavior is clearly wrong; otherwise update docs to match the implemented policy once validated.

## Completion Checks

- The change matches the nearby MRX abstraction and does not bypass an existing operator or preconditioner interface.
- Focused validation was run after editing, not replaced by diff inspection alone.
- Expensive runtime checks were treated as user-run validations when they required substantial Python or GPU work.
- Any documentation update reflects current production behavior rather than an intended future design.
- The final explanation names the validation performed and any unresolved numerical or performance caveats.

## Testing Conventions

- Tests must be cheap. No heavy 3D assembly, large DeRham sequences, or GPU work in tests.
- Share computation across tests wherever possible: build bases and evaluate on grids at module level, not inside each test function.
- One test file per source file.
- No basic inf/nan tests. Every test must verify a specific mathematical property or compare against an analytic baseline.
- Good test targets: partition of unity, positivity, analytic formulas for low-order splines (e.g. Bernstein), de Rham commutation (histopolation vs. finite-difference coboundary), interpolation/histopolation roundtrips.
- Removed tests: implementation roundtrips (knot roundtrip), API consistency checks (getitem vs call) — these are not mathematical properties.

## Good Prompts

- Refactor the MRX mass preconditioner path for k=1 without changing the public API.
- Trace why this shifted `L_0` solve is using the wrong harmonic handling.
- Add a focused test for this saddle-point Schur preconditioner regression.
- Update the iterative solver docs to match the current operator implementation.
- Prepare a notebook cell that I can run on GPU to compare this extracted-space operator against the tensor model.