# Hiptmair-Xu Preconditioner Plan

This note is a draft implementation plan for the higher-form preconditioner
work in `mrx`.

The current strategic split is:

- mass preconditioners are essentially done,
- scalar `k = 0` stiffness / Hodge stays on the tensor route,
- and higher forms `k = 1, 2, 3` move toward a Hiptmair-Xu (HX)
  auxiliary-space preconditioner rather than more tuning of Schur-outers as an
  end state.

The point of this note is not to freeze the exact formula yet. It is to list
the concrete parts that still have to exist before an HX route can be wired in
cleanly.

## 1. Goal

For `k = 1, 2, 3`, keep the existing mixed / saddle-point solve structure, but
replace the current "better Schur outer" direction with an auxiliary-space
preconditioner that:

- uses cheap local interpolation / projection operators,
- reuses scalar Poisson-style preconditioning where possible,
- avoids expensive nested exact solves inside the outer Krylov method,
- and respects the discrete de Rham structure already present in the code.

## 2. Current Starting Point

What already exists and should be reused:

- robust tensor mass preconditioners,
- a working scalar `k = 0` tensor Hodge / Poisson-style preconditioner,
- the extracted-space derivative blocks and incidence structure,
- saddle-point MINRES infrastructure for higher forms,
- transition benchmark baselines:
  - Jacobi + `exact_jacobi` Schur as a reference,
  - tensor lower + Chebyshev outer as an interim production-style comparison.

What is still missing:

- the local interpolation / projection operators that connect
  `H(curl)` / `H(div)` unknowns to the auxiliary scalar spaces,
- the concrete HX apply assembled from those maps and scalar auxiliary solves,
- nullspace / harmonic handling integrated with that auxiliary-space route.

Current preferred first interpolation idea:

- Greville-point interpolation / histopolation built from the existing spline
  structure.

That is attractive for two reasons:

- it is a plausible cheap, local bridge for the HX auxiliary-space maps,
- and it is also useful more generally as infrastructure for putting analytic
  functions into the discrete spaces.

## 3. Required Pieces

### 3.1 Auxiliary-space design table

First make the degree-by-degree table explicit.

For each of `k = 1, 2, 3`, record:

- the source discrete space in the extracted system,
- the auxiliary scalar space or spaces that the HX route needs,
- which exact topological maps already exist in the de Rham sequence,
- which interpolation / projection maps are still missing,
- which scalar Poisson-style operator each auxiliary solve should use,
- and which boundary-condition variants must be supported first.

This table should be written down before implementation so the code does not
drift into ad hoc case-by-case logic.

### 3.2 Local interpolation / projection operators

This is the main missing ingredient.

We need cheap, local operators that map extracted higher-form unknowns into the
auxiliary scalar spaces used by the HX construction.

Requirements:

- locality: no dense global fit or dense global solve during setup,
- sparsity: assembly and apply should stay cheap enough for routine use,
- compatibility with extraction: the maps must respect the extracted dof layout,
- boundary-awareness: Dirichlet / free variants must not be mixed up,
- geometry robustness: the maps should not depend on fragile global tuning,
- adjoint availability: the transpose action must be usable in the HX apply.

Current preferred first prototype:

- Greville-point interpolation for the nodal / point-evaluation side,
- Greville-based histopolation for the integral / moment side,
- assembled from local one-dimensional ingredients wherever possible.

Why start there:

- Greville data is already natural for the spline setting,
- the resulting maps should stay local and structured,
- and the same machinery is useful outside HX whenever analytic fields need to
  be injected into the function spaces.

Still open within that choice:

- whether the extracted-space maps should be built directly on extracted dofs or
  pulled through extraction from tensor-product operators,
- which pieces need strict commuting behavior in the first prototype,
- and how much of the histopolation side can be expressed by reusing existing
  one-dimensional quadrature / basis machinery.

The first prototype does not need to be perfect. It does need to be cheap,
local, and structurally compatible with the existing extracted sequence.

### 3.3 Auxiliary scalar solves

The auxiliary solves should reuse as much of the scalar `k = 0` machinery as
possible.

Tasks:

- decide which scalar operator each HX term should call,
- decide which boundary conditions the auxiliary scalar solve should use,
- decide whether the auxiliary solves are pure Poisson, shifted Poisson, or a
  closely related scalar Hodge block,
- and keep the resulting apply free of nested exact inner solves.

The working assumption is that the scalar tensor route is the right backbone
here, but the exact auxiliary operator per term still needs to be written down
carefully.

### 3.4 Smoother and topological pieces

HX is not only the auxiliary interpolation. It also needs the higher-form
native piece that stays in the original space.

Tasks:

- choose the higher-form smoother / native block correction,
- decide whether that native piece is Jacobi, mass-based, or another cheap
  local block,
- identify which existing incidence / derivative operators already supply the
  exact-sequence part of the decomposition,
- and decide whether the overall HX apply should be additive first, with any
  multiplicative variant deferred until later.

The initial implementation should prefer the simplest additive composition that
is easy to inspect and benchmark.

### 3.5 Nullspace and harmonic handling

The higher-form route will not be finished without a compatible coarse/nullspace
story.

Tasks:

- make the first HX prototype work on the already safe cases:
  - `k = 1` with Dirichlet,
  - `k = 2` without Dirichlet,
- decide how the auxiliary-space route interacts with the current deflation and
  harmonic handling,
- and only then widen to the cases where the nullspace path is still under
  cleanup.

The nullspace work should be staged after the first auxiliary-space prototype,
but the prototype should be built so it does not block that follow-up.

### 3.6 Runtime interface and assembly storage

Once the ingredients exist, they need a clean place in the operator stack.

Tasks:

- decide whether HX should be a new user-facing preconditioner spec or an
  extension of the current higher-form saddle spec,
- decide which assembled interpolation data belongs on `SequenceOperators`,
- keep setup separate from benchmark timing,
- and keep the actual apply compatible with the current JAX / matrix-free
  solver flow.

The first version should optimize for clarity of structure rather than maximum
API compression.

## 4. Suggested Order Of Work

1. Write the degree-by-degree auxiliary-space table.
2. Prototype one Greville-based interpolation / histopolation operator on the
  smallest useful case.
3. Check basic structural properties of that map:
   - sparsity,
   - locality,
   - correct shape,
   - sensible behavior under the chosen boundary conditions.
4. Wire one auxiliary scalar solve through that map into an HX-style additive
   apply for the first target case.
5. Benchmark that prototype against the current transition baselines.
6. Extend the same structure to the next higher-form case.
7. Revisit nullspace / harmonic handling once the auxiliary-space route exists
   end-to-end.

The preferred first target is the smallest harmonic-safe higher-form case that
already behaves well enough for repeated experiments.

## 5. First Milestones

### Milestone A: Paper design

Deliverables:

- a degree-by-degree auxiliary-space table,
- a decision to start from Greville-point interpolation / histopolation,
- a decision on the first scalar auxiliary operator to reuse,
- and a clear statement of the first benchmark case.

### Milestone B: First local map prototype

Deliverables:

- one assembled Greville-based local interpolation / projection operator,
- shape and sparsity checks,
- and one small diagnostic showing that the map behaves sensibly on the chosen
  test case.

### Milestone C: First HX apply

Deliverables:

- one higher-form auxiliary-space preconditioner apply,
- one benchmark comparison against the current Schur-based transition
  baselines,
- and one short note on whether the scalar auxiliary solve is carrying the
  expected part of the work.

### Milestone D: Nullspace integration

Deliverables:

- the first compatible nullspace / harmonic handling pass,
- and widened benchmarks beyond the initial safe case.

## 6. Validation Checklist

The HX route should not be called usable until it has all of the following.

- Local interpolation operators have explicit sparsity and setup-cost checks.
- The transpose actions used in the apply are tested, not assumed.
- The first prototype is benchmarked against the current transition baselines.
- The preconditioned higher-form operator is checked on the deflated positive
  subspace, not only by raw iteration counts.
- Nullspace-sensitive cases are tested separately from harmonic-safe cases.
- Setup time and apply time are reported separately in benchmarks.

## 7. Open Design Questions

- What is the cleanest auxiliary scalar space for each of `k = 1, 2, 3` in the
  extracted formulation used here?
- Should the first Greville-based map be defined directly on extracted dofs, or
  built first on a tensor-product space and then pulled through extraction?
- Which higher-form native smoother is cheap enough to be routine, but still
  strong enough that the auxiliary scalar pieces do not have to do everything?
- Does `k = 3` need a distinct HX layer, or does it mostly reuse the same
  auxiliary infrastructure with a smaller specialization?
- Which boundary-condition combinations should be declared in-scope for the
  first full implementation, beyond the current harmonic-safe cases?
- How much of the analytic-function-to-space interpolation utility should be
  designed now, rather than as an HX-only special case?

## 8. Non-Goals For The First Draft

To keep scope under control, the first HX implementation should not try to do
all of the following at once.

- It should not try to solve the full nullspace / harmonic problem first.
- It should not rely on expensive exact inner solves.
- It should not begin with a heavily optimized multiplicative variant.
- It should not try to replace the scalar tensor route.
- It should not treat the current Schur benchmark tuning as the final
  production design.