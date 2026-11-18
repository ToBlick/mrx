# MRX code map

Where things are and what they are called. How they work and why is in `design.md` and in the concept pages
`docs/source/concepts/` named in each heading. Read the page before changing what it describes.

## Spaces and operators (`concepts/architecture.md`, `concepts/mass.md`)

- `mrx/spline_bases.py`: `SplineBasis(n, p, type)` ("clamped" in r, "periodic" in theta, zeta),
  `DerivativeSpline` (degree p-1, unit integral).
- `mrx/differential_forms.py`: `DifferentialForm(k, ns, ps, types)`, `DiscreteFunction(dof, Lambda, E)`,
  `Pushforward(f, Phi, k)`. A DoF vector is the concatenation of components, each C-order raveled.
- `mrx/quadrature.py`: Gauss rules, `quad_points` overrides the default count. Flat points are **r-major**
  (`seq.quad.shape = (nr, nt, nz)`).
- `mrx/incidence.py`: the exterior derivative `G_k` (entries -1/0/+1), plus the polar stencils `seq.g0_grad`,
  `seq.g1_curl`, `seq.g2_div`.
- `mrx/operators.py`: operator objects indexed by degree, applied with `@`:
  `seq.M[k]` (mass, `.solve`, `.precondition`), `seq.G[k]` (strong derivative), `seq.D[k] = M_{k+1} G_k`
  (weak), `seq.S[k]` (stiffness, `.solve` for co-closed right-hand sides), `seq.L[k]` (Hodge Laplacian,
  `.solve(b, part=...)`), `seq.shifted(k, eps)` (`M_k + eps L_k`), `seq.shifted_stiffness(k, eps)`
  (`M_k + eps S_k`), `seq.P[k, l]` (metric-free pairings), `seq.leray(v, k)`. Creating one inside jit is free.
- Products as L2 loads: `cross_product_load_values`, `dot_product_load_values`, ...
- `mrx/mass.py`: the sum-factorised mass apply. The weights per k are stored on the geometry (`mass_weights[k]`).

## Boundary conditions belong to the space

`seq` holds the Dirichlet spaces (B.n = 0, tangential E/A = 0, p = 0), the working spaces of B, J, E
and the force. `seq.free` is the natural-BC view sharing all arrays. No operator takes a BC argument.
`seq.restrict(v, k)` takes a free form to the Dirichlet space. Take `seq.free` anew after
`set_geometry_from_map`, `build_preconditioners`, `compute_nullspaces`.

## Polar axis (`concepts/polar.md`)

`get_xi`, `build_extraction`, `graded_theta_counts`, `prolongation` in `mrx/extraction_operators.py`,
`build_polar_derivative` in `mrx/incidence.py`, `ring_projections` in `mrx/projectors.py`. `E E^T != I` for
k = 0, 1, 2. `seq.core_rows(k)` are the polar rows. `graded=True` and `map_theta` are arguments of `build_sequence`.

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
- Harmonic forms: `compute_nullspaces`, `harmonic_rayleigh` in `mrx/nullspace.py`.

## Stellarator symmetry (`mrx/symmetry.py`, `concepts/architecture.md` 6)

`symmetry="stellarator"` (default) gives the half-period sequence and the parity views `seq.odd` / `seq.even`.
Odd: B, A, J, E, harmonic 2-form. Even: velocity, force, pressures. `"field-period"` and `"none"` give
full-period sequences where both views are the sequence. A state without the symmetry needs `"field-period"`.

## Solvers (`concepts/preconditioning.md`, `mrx/solvers.py`)

Which solve uses which method and preconditioner: the table in `concepts/preconditioning.md` 1.

- Atoms: `mrx/metric_lumping.py` (`MetricLumpingMass`, `MetricLumpingLaplacian`, `PRODUCTION_BC_SCALE`,
  `STIFFNESS_PROFILE_EXPONENT`). Applied as one batch over the components (`_Batched`, `_apply_batched`).
  `ReducedAtom(atom, X)` for the parity views. The axis block is probed in batches of `PROBE_BATCH` (default 8,
  `MRX_PROBE_BATCH`).
- `mrx/slab_mass.py`: `SlabMass`, the preconditioner of `M_1`, `M_2` (`mass_atom="slab"`, `"lumping"` keeps the atom).
- `mrx/multigrid.py`: `StiffnessMultigrid`, opt-in by `laplacian_atom="multigrid"`.
- Every solve: warm start `guess`, `return_info=True` gives the signed count, the outer loop is
  `mrx.solvers.refine`, up to `MAX_PASSES = 6` passes.

## Precision (`concepts/precision.md`, `mrx/precision.py`, `mrx/double_single.py`)

`MRX_DTYPE` (`float64` default, `float32`), `SOLVE_TOL = 1e-8`, `LOW_DTYPE`, `eps()`. The pair type `DS` and the
conversions `lift`, `to_working`, `low`. A solve returns its accumulated solution with `accurate=True`.

## Relaxation (`concepts/relaxation.md`, `mrx/relaxation/`)

- `physics.py`: the force by the potential route, `compute_force` (Leray route, strong pressure),
  `resistive_step`, `weak_pressure`, `beta_vol`, `compute_helicity`, `compute_divergence_norm`. `force_scale` is in
  `loop.py`.
- `loop.py`: `TimeStepper` (eqx.Module), `relaxation_step`, `State` (`B_n`, `warm`, `best`),
  `initial_state`, `relax(...)` = THE loop (compiled chunks via `lax.scan`, sampler per chunk),
  `write_checkpoint` / `read_checkpoint` / `read_field` / `checkpoint_attrs`. Constants `CFL`, `SMOOTHING_C`.
- `newton.py`: `second_variation`, `newton_direction`, `spline_preconditioner`, the defaults `NEWTON_PENALTY`,
  `NEWTON_TOL`, `NEWTON_MAXITER`, `NEWTON_MASS_SOLVE_TOL`.
- `seeding.py`: `energy_seed`, `parallel_seed`, `resonances`. No resonance in the iota range: warns and returns the
  field unseeded.
- `initial_conditions.py`: `initial_field`, `potential_two_form`. The field is normalised to `||B||_M = 1`
  (`B_norm_raw` in the info).
- `config.py`: tyro dataclasses `RelaxConfig(geometry, seed, descent, newton, budget, drive, output)`.

## Equilibrium input (`concepts/equilibrium_input.md`, `mrx/equilibria/`)

`read_equilibrium(path)` picks by extension: VMEC `wout_*.nc`, GVEC `*.dat`, DESC `*.h5`. One state dict (`X1`, `X2`,
`LA` blocks, profiles in `r = sqrt(s)`). `build_map` returns a `FrameMap`, `cylindrical_frame()` is the frame of
VMEC, DESC and cylindrical GVEC states, `data/gvec_gframe/` an example of a G-frame state. No separate
analytic-shape input: write a mock file (`test/synthetic_gvec.py`, `test/synthetic_desc.py`).

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
