# Relaxation

`mrx/relaxation.py` descends the magnetic energy `E = ½ ||B||²_{M_2}` of a
divergence-free 2-form `B` with `B · n = 0` under an incompressible,
helicity-conserving flow. The fixed point is `J × B = ∇p` with `p` the Leray
multiplier: a finite-beta equilibrium. `scripts/relax.py` is the driver.

## 1. The force

`compute_force(B, seq, dirichlet_H, p_guess, H_guess, JxH_guess)` returns
`(F, p, J, H, JxH)`:

1. `H = M_1^{-1} P_{21} B`: the field as a 1-form, one k=1 mass solve
   (`apply_projection_matrix`, `apply_inverse_mass_matrix`).
2. `J = apply_weak_curl(B)`: the current, a second k=1 mass solve.
3. `JxH = M_2^{-1} cross_product_load(J, H, ...)`: a k=2 mass solve.
4. `F, p = seq.apply_leray_projection(JxH, k=2, p_guess=p_guess)`: removes
   the gradient part with one k=3 Hodge solve; `p` is the pressure.

`F` is the Riesz representative of `-∇E` in the `M_2` inner product. Every
solve is warm-started from the previous step's value.

## 2. The step

`TimeStepper(seq, gamma, mu, descent_method, dt_mode, timestep_mode,
history_size, dirichlet_H, ...)` is an `eqx.Module`; `_relaxation_step` does
one step and `relaxation_step` dispatches on `timestep_mode`
(`IntegrationScheme.EXPLICIT` or `IMPLICIT_MIDPOINT`, the latter through
`midpoint_picard_step`).

1. `compute_force` at `B_n`; `MF = M_2 F` once. It serves `||F||_M`, the CG
   coefficient, and the L-BFGS secant.
2. Direction `u` by `descent_method`:
   - `DescentMethod.GRADIENT`: `u = F`.
   - `CONJUGATE_GRADIENT`: Polak-Ribière, `u = F + β v` with
     `β = max((F·MF - F·MF_prev) / (F_prev·MF_prev), 0)`.
   - `LBFGS`: `_lbfgs_direction(F, s, y, Ms, My)`, the two-loop recursion in
     the `M_2` inner product. The state stores `M s` and `M y` next to `s`
     and `y`, so the recursion applies `M` zero times. The descent variable
     is the velocity: `s = dt · u`, `y = F_prev - F`.
3. `apply_regularization(u)`: `gamma` times `u = (M_2 + mu L_2)^{-1} M_2 u`,
   the hyperregularisation `v = (I - mu Δ)^{-gamma} F`. Off at `gamma = 0`.
4. `u, p_v = seq.apply_leray_projection(u, k=2)`: the flow is
   incompressible.
5. `E = M_1^{-1} cross_product_load(u, H, ...)`: the ideal electric
   field, one k=1 mass solve.
6. `dB = seq.apply_incidence_matrix(E, 1)`: the topological curl, so
   `div B` is conserved to `1e-16` along the trajectory.
7. `dt_star`: `TimeStepChoice.ANALYTIC_LINESEARCH` takes
   `dt_star = F·Mu / ||dB||²_M`, the minimiser of the energy along the
   direction and the largest step that still lowers `E`; `FIXED` uses
   `state.dt`. Then the CFL cap, `dt = min(dt_star, cfl / cfl_max)`:
   `cfl_max = max_{q,i} |u_ref^i(x_q)| / (J(x_q) h_i)` is the largest
   logical CFL number of the velocity (`u_ref^i = J ξ̇^i` are the 2-form's
   reference components at the quadrature points, already evaluated for the
   cross product; `h_i` the knot spacing), with theta not counted inside the
   first radial span, where the polar cell degenerates. `cfl` is a
   `TimeStepper` field (`--cfl`, default 0.5; `inf` disables the cap and
   reproduces the uncapped trajectory bit for bit). The minimiser cannot
   raise the energy, but a large `dt_star` leaves the ideal-induction flow
   (frozen-in topology violated at `O(dt²)`) and diverges when `||dB||`
   collapses. `state.dt_star` and `state.cfl_max` record the cap's activity.
8. `B_ideal = B_n + dt · dB`.
9. Resistivity, backward Euler: `(M_2 + dt · eta · L_2) B_{n+1} = M_2 B_ideal`
   through `apply_inverse_mass_plus_eps_laplace_matrix` (k=2, Dirichlet, the
   saddle form under MINRES with the metric-lumping preconditioners, warm
   start `B_n`), with the same capped `dt` as the ideal step; `eta` is the
   resistivity carried on the state. A
   `lax.cond` skips the solve at `eta = 0`, so the step is then the ideal
   one bit for bit. The signed MINRES count is `state.resistive_info`.

`M_2` is applied three times per step (`M F`, `M u`, `M dB`) for every
method, four with `eta > 0`. With the line search, `dE/dt <= 0` is a
guarantee at every `eta`: the ideal step is the line minimiser and the
implicit diffusion `(I + dt·eta·M_2^{-1} L_2)^{-1}` is an `M_2`-contraction,
so `E(B_{n+1}) <= E(B_ideal) <= E(B_n)`. It also maps `ker(div)` into itself,
so `div B` stays at the solver's tolerance rather than at `1e-16`. The
resistive part used to be explicit (`E - eta · J` inside the curl), which is
stable only for `dt · eta <~ h²`, a limit the line search does not see; that
is why `eta` had to be small and scheduled. It is now unconditionally stable,
and the schedule is a physics choice (how much helicity to spend), not a
stability one. The force residual is not monotone and is not used as a
stopping criterion. Helicity is conserved only at `eta = 0`. Explicit Euler
on the ideal part keeps the frozen-in flux to `O(dt²)`, so a large
line-search step can change the field-line topology with `div B` and
monotone energy intact; a fixed small `--dt0` is the control.

`State` holds `B_n`, the warm-start guesses (`p`, `p_v`, `H`, `JxH`, `E`,
`A`), `F_prev`, `MF_prev`, the four history arrays, `dt`, `dt_star`,
`cfl_max`, `eta`, `resistive_info`, `F_norm`, `v_norm`, `lbfgs_sy`. Build it with `initial_state(B_dof, ts, dt)`, which
runs one `compute_force` so the first secant and CG coefficient see a true
previous gradient. `relaxation_loop(B_dof, ts, num_iters_outer,
num_iters_inner, ...)` runs `num_iters_inner` steps in a `jax.lax.scan` per
outer iteration and records energy, force norm, helicity, divergence, and
step size.

## 3. Diagnostics

- `compute_helicity(B, seq, A_guess)`: one k=1 Hodge solve,
  `A = L_1^{-1} D_1^T B`, then `H = A · P_{21}(B + B_harm)` with
  `B_harm = B - curl A`. The right-hand side is the dual 1-form `D_1^T B`,
  not the weak curl. `||B_harm|| <= ||B||` is an identity (`b_2 = 1` in the
  Dirichlet complex) and a check on the solve.
- `compute_divergence_norm(B, seq)`: `||G_2 B||` through the incidence
  operator, so it measures the field and not a solver residual.
- The energy is `0.5 * seq.l2_norm_sq(B, 2)`.

## 4. Initial conditions

`mrx/initial_conditions.py` builds every field in the reference 2-form frame,
components `(dχ∧dζ, dr∧dζ, dr∧dχ)`. That frame is GVEC's `sqrt(g) B^i`, so

```
B̂^ρ = 0,   B̂^χ = Φ'(ρ) (ι(ρ) - ∂_ζ λ),   B̂^ζ = Φ'(ρ) (1 + ∂_χ λ)
```

is divergence-free and tangent to the boundary for any `λ` and any geometry
before a solve. `λ` moves the field within a surface: it changes force and
energy, not fluxes, `ι`, or helicity.

| function | builds |
|---|---|
| `make_profiles(iota0, iota1, iota_exp, flux_exp)` | `ι = ι₀ + (ι₁-ι₀) ρ^e`, `Φ' = ρ^q` |
| `make_lambda(modes)`, `parse_lambda(spec)` | `λ` from `"m,n,amp;..."` |
| `logical_profile_form(iota, dPhi, dlam)` | the reference 2-form above |
| `clebsch_form(cb, use_lambda)` | the same from a GVEC file's `dPhi_dr`, `dchi_dr`, `LA` (`load_clebsch` in `mrx/gvec.py`) |
| `dzeta_form()` | the constant `(0, 0, 1)`; relaxes to the harmonic field |
| `analytic_helicity(...)` | the closed-form helicity of the logical profiles |
| `project_reference_two_form(seq, omega_ref)` | pushes forward `B = DF ω / J` and L²-projects onto the Dirichlet k=2 space |
| `leray_clean(seq, B)`, `divergence_norm(seq, B)` | remove and measure the projection's divergence |

Units from a GVEC file: `Φ' = 2π dPhi_dr`, `ι = dchi_dr / (nfp · dPhi_dr)`,
`λ = LA / 2π`, because MRX's `ζ` spans one field period and the file's
angular derivatives are per radian. See [gvec_mrx_interface.md](gvec_mrx_interface.md).

## 5. Geometries

`build_sequence(geometry, ns, p, maxiter, tol)` in `mrx/geometries.py`
returns `(seq, ops)` with the map installed and every solver operator built.
Names: `toroid`, `cylinder`, `rot-ellipse` (analytic, `mrx/mappings.py`),
`w7x` (`build_w7x_map`, reads `W7-X.h5`), and every key of
`GVEC_GEOMETRIES` in `mrx/gvec.py` (`build_gvec_map`, flat-schema GVEC
exports: `quasr44970`, `w7x-fmm002`, `hegna`, ...). Files are read from
`MRX_DATA` (default `data`). `geometry_nfp(geometry)` returns the field
periods. `build_gvec_map` measures the handedness of the file and mirrors it
so that `det DF > 0`, and detects whether the angular samples are half-open
or closed.

## 6. Running `scripts/relax.py`

```
SCRIPT=scripts/relax.py JOB_NAME=relax_w1 TIMEOUT_MIN=90 \
  ARGS="--geometry w7x-fmm002 --ic clebsch --ns 8,16,8 --p 3 --steps 3000" \
  bash slurm/run.sh
```

Every run is a GPU job through `slurm/run.sh` (see `slurm/README.md`). One
method per run. Flags, defaults in brackets:

| flag | meaning |
|---|---|
| `--geometry NAME [quasr44970]` | a name from section 5 |
| `--ns R,T,Z [8,16,8]`, `--p P [3]` | resolution (also the map's) and degree |
| `--maxiter N [10000]`, `--tol TOL [sqrt(eps)]` | budget and tolerance of every inner solve |
| `--precision {float64,float32} [float64]` | exported as `MRX_DTYPE` before `mrx` is imported |
| `--ic {logical,clebsch,dzeta} [logical]` | initial condition (section 4) |
| `--iota I0,I1 [0.4,0.9]`, `--iota-exp E [2.0]`, `--flux-exp Q [1.0]`, `--lam SPEC [""]` | the logical profiles |
| `--no-lambda`, `--no-leray-ic` | clebsch with `λ = 0`; skip the Leray clean-up |
| `--method {gradient,cg,lbfgs} [cg]`, `--history M [1]` | descent method and history length |
| `--gamma G [0]`, `--mu MU [0.0]` | hyperregularisation |
| `--dt-mode {linesearch,fixed} [linesearch]`, `--dt0 DT [1.0]`, `--cfl C [0.5]` | step choice and its CFL cap |
| `--eta-max ETA [0.0]`, `--eta-schedule {tanh,constant,linear} [tanh]` | resistivity (implicit, any size); `tanh` drops it to zero over the middle third of the run |
| `--steps N [3000]`, `--seconds S [none]` | outer guards |
| `--floor-tol TOL [10*eps]`, `--floor-window W [100]` | stopping criterion |
| `--diag-every N [250]` | steps between helicity samples (each a k=1 Hodge solve) |
| `--out DIR [outputs/relax/<date>/<time>]` | output directory |

The run stops when the energy decrease over the last `W` steps, relative to
the energy and per step, `(E[i-W] - E[i]) / (W |E[i]|)`, falls below
`--floor-tol` (`energy_floor_reached`), or when a budget runs out. The
criterion is replayed on archived traces in `test/test_relax_floor.py`
without a GPU.

Output: `relax.json` with the parameters, the per-step trace (`E`, `F`,
`dt`, `dt_star`, `cfl`, `div`, `eta`, `res_it`, `dE_meas`, `dE_pred`), the sampled diagnostics, the
initial-condition summary, and the stopping reason; and `B.h5` with the
final field.

At the reference resolution (`w7x-fmm002`, `(8,16,8)`, `p = 3`, one H100):
setup about 330 s, first step about 90 s of compilation, then 0.7-0.9 s
per step.
