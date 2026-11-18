# MRX code map

Condensed from `docs/source/concepts/` (architecture, mass, polar, preconditioning, precision,
relaxation, equilibrium_input). Read the full page before changing what it describes.

## Spaces and operators (`concepts/architecture.md`)

- `mrx/spline_bases.py`: `SplineBasis(n, p, type)` ("clamped" in r, "periodic" in theta, zeta),
  `DerivativeSpline` (degree p-1, unit integral).
- `mrx/differential_forms.py`: `DifferentialForm(k, ns, ps, types)`, `DiscreteFunction(dof, Lambda, E)`,
  `Pushforward(f, Phi, k)`. A DoF vector is the concatenation of components, each C-order raveled.
- `mrx/quadrature.py`: Gauss rules with `p + 1` points per span in r and `p` in the angles by default
  (`quad_points` overrides), flat points **r-major** (`seq.quad.shape = (nr, nt, nz)`).
- `mrx/incidence.py`: the exterior derivative `G_k` is topological (entries -1/0/+1), plus the polar
  stencils `seq.g0_grad`, `seq.g1_curl`, `seq.g2_div`.
- `mrx/operators.py`: operator objects indexed by degree, applied with `@`:
  `seq.M[k]` (mass, `.solve`, `.precondition`), `seq.G[k]` (strong derivative), `seq.D[k] = M_{k+1} G_k`
  (weak), `seq.S[k]` (stiffness, `.solve` for co-closed right-hand sides), `seq.L[k]` (Hodge Laplacian,
  `.solve(b, part=...)`), `seq.shifted(k, eps)` (`M_k + eps L_k`), `seq.shifted_stiffness(k, eps)`
  (`M_k + eps S_k`), `seq.P[k, l]` (metric-free pairings), `seq.leray(v, k)`. Creating one inside jit is free.
- Products are L2 loads: `cross_product_load_values`, `dot_product_load_values`, ... 1-forms are
  covariant, 2-forms contravariant densities, so wedge/contraction are metric-free and induction
  conserves flux exactly. The metric enters only through a Hodge star.
- `mrx/mass.py`: no matrix is stored. Global batched sum factorisation: the components are zero-padded to one
  grid, three contractions to the quadrature grid, the full `n x n` weight per point, three back. Weights per k:
  `J`, `J g^-1`, `g / J`, `1 / J` with `g = DPhi^T DPhi`, `J = det DPhi`, stored on the geometry (`mass_weights[k]`).

## Boundary conditions belong to the space

`seq` holds the Dirichlet spaces (B.n = 0, tangential E/A = 0, p = 0), the working spaces of B, J, E
and the force. `seq.free` is the natural-BC view sharing all arrays. No operator takes a BC argument.
`seq.restrict(v, k)` takes a free form to the Dirichlet space. Take `seq.free` anew after
`set_geometry_from_map`, `build_preconditioners`, `compute_nullspaces`.

## Polar axis (`concepts/polar.md`)

C^1 polar splines: rings 0 and 1 replaced by 3 barycentric functions per zeta plane
(`get_xi`, `build_extraction` in `mrx/extraction_operators.py`). `E E^T != I` for k = 0, 1, 2. The
polar rows are a dense "core" in the preconditioners. The map is singular at r = 0: evaluate near the
axis, not on it. `graded=True` gives each radial ring its own theta count, fewer near the axis
(`graded_theta_counts`), with the map on `map_theta` uniform theta cells. The measured meshes of
`docs/research/` are graded.

## Build order (`build_sequence` in `mrx/geometry.py`)

```python
seq = build_sequence(path, (nr, nt, nz), p, symmetry="stellarator", graded=True)  # topology, map, preconditioners
compute_nullspaces(seq)                                                            # harmonic forms, SEPARATE call
B0, info = initial_field(seq)                                                      # mrx.relaxation.initial_conditions
```

Nothing is built on first use. By default a geometry change discards the operator bundle: after
`seq.set_geometry_from_map(Phi)` rerun `build_preconditioners()` and `compute_nullspaces(seq)`. With
`keep_preconditioners=True` the preconditioners of the old geometry stay. They are still valid
preconditioners of the new operators, and good ones after a small change of the map. The harmonic forms are
set to zero either way, because a form of the old geometry is not in the kernel of the new Laplacian: run
`compute_nullspaces(seq)` again before a solve that removes them. A map that folds (`det DPhi <= 0`) is
refused.

## Data model

- `DeRhamSequence` (`mrx/derham_sequence.py`): static topology, a pytree (`mrx/pytree.py`).
- `SequenceGeometry` (`mrx/geometry.py`, eqx.Module): map, `metric`, `metric_inv`, `jacobian` at the
  quadrature points. DPhi is not stored.
- `SequenceOperators` (`mrx/operators.py`): mass atoms, Laplacian atoms, harmonic forms, keyed `(k, dirichlet)`.
- Harmonic forms: Betti numbers of the solid torus `(1, 1, 0, 0)`. `mrx/nullspace.py`.

## Stellarator symmetry (`mrx/symmetry.py`)

`symmetry="stellarator"` (default): half-period quadrature (zeta in [0, 1/2], doubled weights) and
parity views `seq.odd` / `seq.even` with half the DoFs. Odd: B, A, J, E, harmonic 2-form. Even:
velocity, force, pressures. `"field-period"` and `"none"` give full-period sequences where both views
are the sequence. A state without the symmetry needs `"field-period"`. Needs uniform angular knots.

## Solvers (`concepts/preconditioning.md`, `mrx/solvers.py`)

| solve | method | preconditioner |
|---|---|---|
| `M_k` | PCG | slab model (k = 1, 2), mass atom (k = 0, 3) |
| `L_0` | PCG on `S_0`, kernel deflated | k = 0 Laplacian atom |
| `L_1`, `L_2` | Hodge split: three PCGs on the stiffnesses | atoms of levels k, k-1 |
| `S_k x = G_k^T w` | PCG on the singular `S_k` (consistent right-hand side only) | Laplacian atom of level k |
| `L_3`, Leray k = 2 | Hodge split at k = 3: two PCGs on `S_2`, gauge on `S_1` | atoms of levels 2, 1 |
| `M_k + eps L_k` | shifted split: three SPD PCGs | shifted-stiffness atoms |

- Atoms: `mrx/metric_lumping.py` (`MetricLumpingMass`, `MetricLumpingLaplacian`). Bulk = Kronecker sum
  with axis-mean metric weights, inverted by fast diagonalisation. Core = probed dense polar block.
  Natural-BC rank-one term scaled by `PRODUCTION_BC_SCALE = 3.0`. Graded spaces get level blocks plus
  strips. Applied as one batch over the components (`_Batched`, `_apply_batched`), gathers composed into
  one input and one output gather. `ReducedAtom(atom, X)` for the parity views. The axis block is probed in
  batches of `PROBE_BATCH` (default 8, `MRX_PROBE_BATCH`).
- `mrx/slab_mass.py`: `SlabMass`, the default preconditioner of `M_1`, `M_2` (`mass_atom="slab"`). The
  atom stays the norm of the solves and every `M^{-1}` inside an iteration.
- `mrx/multigrid.py`: `StiffnessMultigrid`, a V-cycle for the co-closed `S_2` solves, opt-in by
  `laplacian_atom="multigrid"`.
- No inner Krylov under MINRES/CG: where `M^{-1}` sits inside a preconditioner the mass atom replaces it.
  Singular stiffness solves need a right-hand side in the range of `G_k^T`.
- Every solve: warm start `guess`, `return_info=True` gives the signed count, stops on the true
  residual in the mass-atom norm (`mrx.solvers.refine`), up to `MAX_PASSES = 6` passes.

## Precision (`concepts/precision.md`, `mrx/precision.py`, `mrx/double_single.py`)

| `MRX_DTYPE` | arrays | JAX 64-bit | solves |
|---|---|---|---|
| `float64` (default) | float64 | on | Krylov, true residual checked |
| `float32` | float32 | off | Krylov in float32, solution and residual in double-single (`DS`) |

Default tol `SOLVE_TOL = 1e-8` in both. Preconditioners, Krylov directions, norms and MINRES are float32
(`LOW_DTYPE`) in both. Matmul precision "highest" (no TF32). An operator applied to a `DS` returns a `DS`.
`lift`, `to_working`, `low` convert. Ask a solve for `accurate=True` where its solution is subtracted from
something close (`J x B` in `compute_force`). Test tolerances are multiples of `mrx.eps()` or `seq.tol`.

## Relaxation (`concepts/relaxation.md`, `mrx/relaxation/`)

- `physics.py`: force by the potential route (`J = M_1^-1 D_1^T B`, `load(J x B)`, `L_1 a = curl^T load`,
  `F = curl a + c h`). `compute_force` = Leray route, gives the strong pressure. `weak_pressure`,
  `beta_vol`, `compute_helicity`, `compute_divergence_norm`. `force_scale` is in `loop.py`.
- `loop.py`: `TimeStepper` (eqx.Module), `relaxation_step`, `State` (`B_n`, `warm`, `best`),
  `initial_state`, `relax(...)` = THE loop (compiled chunks via `lax.scan`, sampler per chunk),
  `write_checkpoint` / `read_checkpoint` / `read_field` / `checkpoint_attrs`.
- Step: direction u (Newton or smoothed force), `E = M_1^-1 load(u x B)`, `dB = G_1 E` (div B exact),
  `dt_star` exact line search, CFL clip, `dt <= 1` with Newton. Optional resistive step
  `(M_2 + eps L_2) delta = -eps L_2 (B - B_ref)`, done as one `shifted_stiffness(1, eps)` solve.
- `newton.py`: Gauss-Newton part of the Hessian `H`, `newton_direction` solves `curl^T H curl a = curl^T load(J x B)`
  by Newton-MR (MINRES with reorthogonalised Lanczos vectors), spline-atom preconditioner (`spline_preconditioner`), parallel penalty
  `kappa * strain` lifting the field-aligned null space. Defaults: penalty 3, tol 0.1, maxiter 300, the two
  `M_1` solves per Hessian apply by PCG to `newton_mass_solve_tol = 1e-6`.
- `seeding.py`: `energy_seed`, island seeds at every resonance in the iota range (or the given `iotas`),
  profiles by the energy criterion `a* = -G^-1 g`. No resonance: warns and returns the field unseeded.
- `initial_conditions.py`: Clebsch potential `A'` interpolated on the free 1-forms, `B = dA'`, normalised
  to `||B||_M = 1` (`B_norm_raw` in the info).
- `config.py`: tyro dataclasses `RelaxConfig(geometry, seed, descent, newton, budget, drive, output)`.

## Equilibrium input (`concepts/equilibrium_input.md`, `mrx/equilibria/`)

`read_equilibrium(path)` picks by extension: VMEC `wout_*.nc`, GVEC `*.dat`, DESC `*.h5`. One state
dict (`X1`, `X2`, `LA` blocks, profiles in `r = sqrt(s)`: `phi` (flux / 2 pi), `iota`, `pressure`).
`build_map` = L2 projection of the series onto the polar 0-forms, a `FrameMap` in the frame the file's
solver supplies: the cylindrical frame (`X1 = R`, `X2 = Z`) for VMEC, DESC and cylindrical GVEC states,
GVEC's own frame for a G-frame state (example in `data/gvec_gframe/`). `X3`, out of the frame's plane, is
zero for every file. The initial field is rebuilt from scalars, so
`div B = 0`, `B.n = 0` and nested surfaces are exact. No separate analytic-shape input: write a mock file
(`test/synthetic_gvec.py`, `test/synthetic_desc.py`).

## jit and host/device rules

- Host (NumPy, float64, once per sequence): readers, fits, topology that fixes shapes and index maps.
- Device (JAX): everything geometry-dependent (metric, atoms, harmonic forms, solves, h_r^2, CFL
  weights). Use JAX linear algebra, not `np.linalg`.
- Module-level `eqx.filter_jit` functions taking the sequence/stepper as arguments. Never jitted
  closures per call, never functions stored on a pytree (recompiles per build).
- A Python float in an eqx.Module is static: changing it recompiles. Geometry-derived scalars are
  device arrays. A map given as a pytree (`FrameMap`) reuses compiled solves across
  geometries.
- `--geometry.max-batch` (default 262144) bounds the quadrature points evaluated at once when those loops
  run out of device memory.
