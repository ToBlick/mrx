# Relaxation

`mrx/relaxation.py` descends the magnetic energy `E = ½ ||B||²_{M_2}` of a
divergence-free 2-form `B` with `B · n = 0` under an incompressible,
helicity-conserving flow. The fixed point is `J × B = ∇p` with `p` the Leray
multiplier: a finite-beta equilibrium. `scripts/relax.py` is the driver.

## 1. The force

`compute_force(B, seq, auxiliary_B_field, p_guess, H_guess, JxH_guess)`
returns `(F, p, J, X, JxX)`:

1. `J = apply_weak_curl(B)`: the current, one k=1 mass solve.
2. `X`, the field the cross products read. Without the auxiliary field
   (the default) it is the 2-form `B` itself. With it, `H = M_1^{-1} P_{21}
   B`: the field as a Dirichlet 1-form (`H_t = 0` on the wall), a second
   k=1 mass solve (`apply_projection_matrix`, `apply_inverse_mass_matrix`),
   the auxiliary variable of the helicity-conserving scheme below.
3. `JxX = M_2^{-1} cross_product_load(J, X, ...)`: a k=2 mass solve.
4. `F, p = seq.apply_leray_projection(JxX, k=2, p_guess=p_guess,
   sigma_guess=sigma_guess)`: removes the gradient part with one k=3 Hodge
   solve; `p` is the pressure. The solve is the saddle MINRES in `(p,
   sigma)` and its lower unknown `sigma = M_2^{-1} D_2^T p` IS the gradient
   part, so `F = JxX - sigma` costs no further mass solve.

`F` is the Riesz representative of `-∇E` in the `M_2` inner product. Every
solve is warm-started from the previous step's value: `J`, `H`, `JxX`, `p`,
and `sigma` as the previous `JxX - F` (a warm start on `p` alone leaves
`D^T p` in the initial residual's lower block, so the two go together).

## 2. The step

`TimeStepper(seq, auxiliary_B_field, velocity_smoothing_order,
velocity_smoothing_scale, cfl)` is an `eqx.Module`;
`relaxation_step(state)` does one step, explicit Euler on the descent
velocity, `B_{n+1} = B_n + dt · curl(u × X)`.
The step is ideal: reconnection is a separate resistive solve between
chunks (section 2a).

1. `compute_force` at `B_n`; `MF = M_2 F` once. It serves `||F||_M` and
   the Newton right-hand side.
2. Direction `u`: the smoothed force (gradient descent; the L-BFGS memory
   was removed 2026-09-17, its arms were too finicky for the paper), or
   Newton's direction with `newton=True`.
3. `smooth_velocity(u)`: `velocity_smoothing_order` times
   `u = (M_2 + mu L_2)^{-1} M_2 u` with `mu = velocity_smoothing_scale`, the
   smoothed direction `v = (I - mu Δ)^{-order} F`. Off at order 0. The
   default scale is `SMOOTHING_C h_r^2 = 0.075 h_r^2`, $h_r^2 = \langle g_{rr} \rangle \Delta r^2$
   the squared physical radial cell (`mrx.relaxation.radial_cell_sq`): the
   radial two-cell mode damped by 1/1.7 on any device; swept 2026-09-05 on
   li383 (16,32,32) p=2 (flat optimum 0.074-0.24 per step, 0.074 cheapest
   per second, the helicity drift independent of the scale).
   The flow is incompressible without a projection of its own: the
   force is Leray-projected and the smoothing commutes with the
   divergence. A second Leray projection of the velocity was measured to
   change nothing in float64 and in mixed precision and to cost 1.5-4x
   the step (`docs/research/velocity_leray_ab_2026-09-04.md`); until
   2026-09-05 it was step 4.
4. `E = M_1^{-1} cross_product_load(u, X, ...)`: the ideal electric
   field, one k=1 mass solve (`X` as in section 1).
5. `dB = seq.apply_incidence_matrix(E, 1)`: the topological curl, so
   `div B` is conserved to `1e-16` along the trajectory.
6. `dt_star = F·Mu / ||dB||²_M`, the minimiser of the energy along the
   direction and the largest step that still lowers `E` (the analytic line
   search). Then the CFL cap, `dt = min(dt_star, cfl / cfl_max)`:
   `cfl_max = max_{q,i} |u_ref^i(x_q)| / (J(x_q) h_i)` is the largest
   logical CFL number of the velocity (`u_ref^i = J ξ̇^i` are the 2-form's
   reference components at the quadrature points, already evaluated for the
   cross product; `h_i` the knot spacing), with theta not counted inside the
   first radial span, where the polar cell degenerates. `cfl` is a
   `TimeStepper` field (`cfl`, 0.5; `inf` disables the cap and
   reproduces the uncapped trajectory bit for bit). The minimiser cannot
   raise the energy, but a large `dt_star` leaves the ideal-induction flow
   (frozen-in topology violated at `O(dt²)`) and diverges when `||dB||`
   collapses. `state.dt_star` and `state.cfl_max` record the cap's activity.
7. `B_{n+1} = B_n + dt · dB`.

`M_2` is applied three times per step (`M F`, `M u`, `M dB`). With the line search `dE/dt <= 0` is a guarantee: the step is the
line minimiser. The force residual is not monotone, which is why the
stopping criterion (section 6) averages it over a window. Explicit Euler
keeps the frozen-in flux to `O(dt²)`, so a large line-search step can
change the field-line topology with `div B` and monotone energy intact;
the CFL cap is the control.

### 2a. Reconnection

`resistive_step(B, seq, eps)` is one backward-Euler step of
`dB/dt = -eta curl curl B` over `eps = eta dt`, in defect form:
`(M_2 + eps L_2) delta = -eps L_2 B`, `B + delta`, with initial guess 0,
through `apply_inverse_mass_plus_eps_laplace_matrix` (k=2, Dirichlet, two
SPD CG solves through the split identity with the shifted-stiffness atom;
`L_2 B` is one Hodge-Laplacian apply). The defect form is what keeps the
solve meaningful in float32: solving for `B` itself with a tolerance
relative to `||B||` returns `B` unchanged when the correction is a few
ulps, whereas the tolerance here is relative to `delta` in both
precisions. The solve is unconditionally stable and dissipative,
`(I + eps·M_2^{-1} L_2)^{-1}` an `M_2`-contraction, and maps `ker(div)`
into itself, so `div B` stays at the solver's tolerance.

The descent never applies it inside the step: the drive of `scripts/relax.py`
(`--drive-resistivity`) adds it AFTER every ideal step towards a reference
current (section 7 of the paper), and a single reconnection is
`resistive_step` itself (Tutorial 6).

**The helicity correction** (`TimeStepper.helicity_correction`,
`scripts/paper_scripts/relax_paper.py --helicity-correction`) keeps `H` natural and removes the leak
instead. Over one explicit step the identity above reads, exactly on the
mesh, `K_{n+1} - K_n = 2 dt <E, P B_n> + dt^2 <E, P curl E>`, and the
pairing `<E, P B>` is one number per step: the projection residual of
`u x B` paired with the tangential wall DoFs of the natural proxy. With
`E - lambda H_D`, `H_D = M_1^{-1} P B` the Dirichlet proxy of the field, the
change is a quadratic in `lambda` whose root near zero the explicit step
takes (two pairings of curls, one warm-started k=1 mass solve for `H_D`).
The pairings are formed in the residual
precision, a small total of large terms. The helicity is then flat to the
solves and to the stored field's rounding, `E` stays Dirichlet so `B` keeps
its wall condition, and there is no wall layer. What the correction removes
is the component of `E` along `B`, which the exact `u x B` does not have
and which is the only one that changes helicity; the induction picks up
`-lambda curl H_D`, of the size of the leak, and the energy decrease is
perturbed by `lambda` times the `J . B` pairing. The step is no longer
variational: the energy is monotone up to that term. The trace records
`lambda` as `hcorr`.

`State` holds `B_n`, `B_nplus1`, `dt`, `dt_star`, `cfl_max` and three
subtrees: `warm` (the warm starts `p`, `H`, `JxH`, `J`, `E`, `a`, `A`,
`resistive_delta`), `last` (the last step's `F`, `F_norm`, `v`, `v_norm`
and its diagnostics) and `best` (the best field, its residual and step). Build it with `initial_state(B_dof, ts, dt)`, which
runs one `compute_force` so the first step's solves start from the true
previous force. `relax(state, ts, steps, chunk, ...)` runs the steps in
`jax.lax.scan` chunks of `chunk` (`chunk_runner`), samples the diagnostics
once per chunk (`make_sampler`: the energy of the stored field in the
residual precision, helicity, the two pressures, beta), applies
the floor, wall-budget and reconnection rules and returns a `RelaxResult`
(the state, the per-step trace, the per-chunk samples, the reconnection
records); `write_checkpoint` / `read_checkpoint` store and restore a state.

## 3. Diagnostics

- `compute_helicity(B, seq, A_guess)`: one k=1 Hodge solve,
  `A = L_1^{-1} D_1^T B`, then `H = A · P_{21}(B + B_harm)` with
  `B_harm = B - curl A`. The right-hand side is the dual 1-form `D_1^T B`,
  not the weak curl. `||B_harm|| <= ||B||` is an identity (`b_2 = 1` in the
  Dirichlet complex) and a check on the solve.
- `compute_divergence_norm(B, seq)`: `||G_2 B||` through the incidence
  operator, so it measures the field and not a solver residual.
- The energy is `0.5 * seq.l2_norm_sq(B, 2)`.
- `weak_pressure(J, X, seq, auxiliary_B_field, p_guess)` and
  `pressure_diagnostics(B, p, p_w, F_w, v, seq)`: the second pressure and
  the plasma beta, below.

### Two pressures

The relaxation has two pressures. They agree at a fixed point whose
pressure is constant on the wall and differ everywhere else, in a way
that is itself a diagnostic.

**Strong** (`p`, from `compute_force`): the Leray multiplier of the
constrained energy principle. `J × H` is projected onto the Dirichlet
2-form space first, which discards its normal component `(J × H) · n`,
and the k=3 Hodge solve of `apply_leray_projection(k=2)` removes the
gradient part with `σ · n = 0` built in. So `p` satisfies `dp/dn = 0` on
the wall by construction and is defined up to a constant. It is the
right multiplier for the descent: `F = J × H - ∇p` is exactly the force
the constrained flow (`u · n = 0`) can see. It is blind to the wall
force.

**Weak** (`p_w`, from `weak_pressure`): `J × H` is projected onto the
NATURAL 1-form space, `v = M_1^{-1} load(J × H)` (no boundary condition,
so `v · n` is `(J × H) · n`), and Helmholtz-decomposed there,
`v = F_w + ∇p_w`, with `p_w` in the Dirichlet 0-form space:
`(∇φ, ∇p_w) = (∇φ, v)` for every `φ` with `φ = 0` on the wall
(`apply_leray_projection(v, k=1, dirichlet_p=True)`, one k=0 Dirichlet
Laplacian solve, CG with the metric-lumping atom). `p_w = 0` on the wall
by construction, so it has no gauge; `F_w` is divergence-free in the
interior and keeps its normal trace, and on the wall
`(J × H) · n = dp_w/dn + F_w · n`. At a fixed point `J × H` is a
gradient, `F_w` vanishes, and `dp_w/dn` is the wall force. `J` and `H`
come from `compute_force`, so the current is computed once.

Read `p` for the descent and the force residual; read `p_w` for the
pressure profile, the wall force and beta. `scripts/relax.py` records at
every qoi sample, in `qoi`, `ic` and `summary` of `relax.json`:

| key | definition |
|---|---|
| `gradp_cmp` | `‖Π_2 ∇p_w - ∇_w p‖_{M_2} / ‖Π_2 ∇p_w‖_{M_2}`, gauge-free. `∇_w p` is the weak gradient of the 3-form in the Dirichlet 2-form space (the `σ` the Leray step subtracts): the L2 projection of the true gradient onto that space, so its normal trace is zero whatever `dp/dn` is. `∇p_w` is the exact strong gradient of the 0-form (incidence matrix, natural 1-form space), projected onto the same space, `Π_2 = M_2^{-1} P_{12}`, so both sides lose the same normal trace and the ratio compares the pressures, not the projection: against the unprojected `∇p_w` it reads 0.6 for identical pressures on the (4,6,4) test torus, the wall layer |
| `p_cmp` | `‖(p/J - ⟨p/J⟩) - (p_w - ⟨p_w⟩)‖_{L2} / ‖p_w - ⟨p_w⟩‖_{L2}` at the quadrature points, `⟨·⟩` the volume mean: the two pressures as functions, the strong one's gauge removed |
| `weak_resid` | `‖F_w‖_{M_1} / ‖v‖_{M_1}`: the part of `J × H` that is not the gradient of a function vanishing on the wall |
| `dpdn_wall` | `max |dp_w/dn|` over the wall (`r = 1` at the angular quadrature points) relative to `max |∇p_w|` over the quadrature points; `p_w = 0` on the wall, so its gradient there is normal |
| `JxBn_wall` | `max |(J × H) · n|` on the same wall points, from `v`, relative to the same `max |∇p_w|`: the wall force the strong pressure cannot see |
| `beta_vol` | `⟨p_w, 1⟩_{M_0} / E` with `E = ½ B^T M_2 B = ∫ B²/2 dV`. Code units: the magnetic pressure is `B²/2`, so `β = ∫ p dV / ∫ B²/2 dV` |
| `beta_axis` | `⟨p_w⟩ / ⟨|B|²/2⟩` on the COORDINATE axis, logical `r = 0`: both averaged (quadrature weights) over the innermost radial quadrature layer, `r = x_r[0]`, a few percent of the first knot span, all `θ` and `ζ`. The 2-form's magnitude `B_ref^T G B_ref / J²` is 0/0 on the polar axis itself, and the polar 2-form space does not pin `B_ref(0)` to zero, so a limit `r → 0` reads the solver's residual there (measured: 50% off at `r = sqrt(eps)`) |

The strong pressure `p` is state (a field of `State`, so it is in every
checkpoint); the weak pressure is a diagnostic, computed by the sampler at
every chunk and by the plotters on demand (two solves per field).
`scripts/poincare_trace.py` archives `p_w` at the crossings and
`scripts/poincare_plot.py` draws it normalised by the field's mean magnetic
pressure, `p_w / <B^2/2>`: a 0-form, its value is the spline evaluation and
it is not shifted. The tutorials draw `p_w` on the torus and in poloidal
cuts (`mrx.plotting.plot_torus`, `plot_crossections_separate`) and `‖F‖_M`
against `E` on twin axes (`plot_twin_axis`); `scripts/compare_relaxations.py OUT label=run ...`
overlays several runs' traces (`‖F‖`, `E_0 - E`, `-dE/dt`, `dH/H_0`, `dt`,
the CFL number, `‖J‖/‖B‖`, `beta_vol`, the line-search cosine) against
relaxation time and against step, plus a runtime view (relaxation time
per wall hour, seconds per step). `test/test_weak_pressure.py` checks the
decomposition (`v = ∇q` returns `p_w = q`) and the closed-form beta of
`1 - r²` against `e_φ / R` on the analytic torus.

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
| `make_lambda(modes)` | `λ` from `[(m, n, amp), ...]` |
| `analytic_profile_form(iota, dPhi, dlam)` | the reference 2-form above; the initial condition of an analytic geometry file, whose `profile` block supplies the numbers |
| `clebsch_potential_form(cb)`, `potential_two_form(seq, A_ref)` | the reference 1-form `A' = (-LA dPhi_dr, 2π Φ, -(2π/nfp) χ)` (the GVEC potential with the gauge term `d(Φ LA)` dropped; `Φ`, `χ` integrated from the profiles) histopolated on the FREE 1-form space -- its wall trace `2π Φ_edge` is the toroidal flux, the Dirichlet harmonic content -- and `B = dA'` by the exact incidence curl into the Dirichlet 2-form space: `div B = 0` to round-off, no Leray step, and no derivative of the sampled `LA` is ever taken (the discrete `d` differentiates), so a coarse export cannot inject grid-scale current through its interpolant; the initial condition of every equilibrium file |
| `project_reference_two_form(seq, omega_ref)` | pushes forward `B = DF ω / J` and L²-projects onto the Dirichlet k=2 space |
| `leray_clean(seq, B)`, `compute_divergence_norm(B, seq)` (`mrx.relaxation`) | remove and measure the projection's divergence |

Units from a GVEC file: `Φ' = 2π dPhi_dr`, `ι = dchi_dr / (nfp · dPhi_dr)`,
`λ = LA / 2π`, because MRX's `ζ` spans one field period and the file's
angular derivatives are per radian. See [gvec_mrx_interface.md](gvec_mrx_interface.md).

## 5. Geometries

`build_sequence(geometry, ns, p, maxiter, tol, nfp, knots)` in
`mrx/geometry.py` returns `(seq, ops)` with the map installed and every
solver operator built. `geometry` is the path of a GVEC state (`.dat`) or
a VMEC wout (`.nc`) (`build_gvec_map` in `mrx/gvec.py`), or of an analytic
geometry file (`.json`, `read_analytic`: a map of `mrx/mappings.py`,
`torus`, `cylinder` or `rot-ellipse`, with its parameters and the profiles
of the analytic initial condition; `data/torus.json` and its siblings are
the shipped ones). Anything else raises. The file is parsed once and kept
on the sequence as `seq.equilibrium` (the state dict of
`read_equilibrium`, with `kind`, or the analytic file's dict), which
`initial_field(seq, seed)` and `load_clebsch(seq.equilibrium)` read; a
VMEC wout's refit into the GVEC blocks is therefore done once per run.
`nfp` overrides an equilibrium
file's value for a file that declares it wrong. `geometry_kind(geometry)`
returns `vmec`, `gvec` or the map's name, `geometry_nfp(geometry, nfp)` the
field periods. `build_gvec_map`
measures the handedness of the file and mirrors it so that `det DF > 0`.
Nothing is resolved from names or from the environment: every reader takes
the path.

## 6. Running `scripts/relax.py`

```
SCRIPT=scripts/relax.py JOB_NAME=relax_w7x TIMEOUT_MIN=60 \
  ARGS="--geometry data/GVEC_State_final.dat" bash slurm/run.sh
```

Every run is a GPU job through `slurm/run.sh` (see `slurm/README.md`). One
method per run. Flags, defaults in brackets:

| flag | meaning |
|---|---|
| `--geometry PATH` (required) | a VMEC wout (`.nc`), a GVEC state (`.dat`) or an analytic geometry (`.json`, which carries its `nfp`); the geometry and the initial condition |
| `--symmetry {stellarator,field-period,none} [stellarator]` | what the map satisfies (`mrx.geometry.SYMMETRIES`): `nfp` field periods and stellarator symmetry (the spline map is projected onto it, the quadrature covers half the period and the fields keep their parity), field periods only, or nothing (`zeta` in `[0, 1]` is the whole torus, `nfp = 1`) |
| `--resolution R,T,Z [32,64,64]`, `--spline-degree P [2]` | spline resolution (also the map's) and degree |
| `--knots-r LIST`, `--knots-theta LIST`, `--knots-zeta LIST` | the breakpoints of that axis, comma-separated from 0 to 1, instead of the uniform grid; the axis takes its `n` from them (`mrx.geometry.knot_vector`) |
| `--precision {mixed,float32,float64} [mixed]` | `mixed` is float32 fields and solves with a float64 residual (tolerance 1e-8), `float32` and `float64` are both (1e-5, 1e-10); exported as `MRX_DTYPE` and `MRX_RESIDUAL_DTYPE` before `mrx` is imported |
| `--solve-tol TOL [the precision's]`, `--solve-maxiter N [2000]` | residual tolerance and budget of every solve (`concepts/precision.md`) |
| `--max-batch N [0]` | cells per batch of the quadrature loops; 0 evaluates all points in one `vmap`; bound it at high resolution (8192 at (64,128,128)) |
| `--seed` / `--no-seed` [off] | equilibrium files only: the energy-criterion seed of `mrx.seeding` on the start field (the initial condition or the `--restart` checkpoint): every resonance `nfp n / m` in the field's iota range gets SIESTA's parallel seed at the amplitudes of least energy; no phase (the stellarator parity fixes it) |
| `--seed-iotas I,...`, `--seed-amplitudes A,...`, `--seed-scale Q [1]` | the chains to seed by their rotational transforms [all in range]; their amplitudes, the resonant normal field `|dB^r| / |B^zeta|` at `r_mn`, signed, instead of the criterion's (one per iota; amplitudes without iotas are ignored with a warning); a multiplier of the added perturbation |
| `--method {newton,gradient} [newton]` | the direction: Newton on the second variation (`mrx.hessian.newton_direction`, the rows below), or gradient descent on the smoothed force |
| `--scheme {explicit,midpoint} [explicit]` | the induction step: forward Euler, or midpoint-implicit at the predictor's velocity (Picard on the increment) |
| `--newton-penalty KAPPA [3]` | the parallel-flow penalty, `KAPPA` times the strain along the field: the one number of the Newton configuration |
| `--newton-tol TOL [0.1]`, `--newton-maxiter N [200]` | the forcing term that ends the Newton solve (its residual below `TOL` of the right-hand side) and the MINRES iterations at most |
| `--steps N [100 Newton, 2000 gradient]` | the step budget; a job's time limit is no stop, the checkpoint of every chunk restarts it (`--restart`) |
| `--chunk N [10 Newton, 200 gradient]` | steps per compiled chunk (one `lax.scan`, `mrx.relaxation.chunk_runner`): the per-step trace comes back, the qoi are sampled, a checkpoint and the outputs are written, and the floor test runs once per chunk; `--steps` is a multiple of it |
| `--floor-tol TOL [1e-10]` | stopping criterion: the last chunk's mean squared normalised force residual `‖F‖²_M / ‖grad(B²/2)‖²` below it |
| `--drive-resistivity C [0]`, `--drive-reference PATH`, `--drive-reference-smoothing c [0.1]` | the resistive steady state: a backward-Euler resistive step of dose $C h_r^2$ in every step, $E = \eta (J - J^*)$, $J^*$ the current of the reference checkpoint's field (required: a converged nested equilibrium) after one heat step of $c h_r^2$ |
| `--drive-chain I`, `--drive-eps A` | a resonant seed of the reference at the chain of rotational transform `I`, amplitude `A` as `--seed-amplitudes`: a drive that stays |
| `--out DIR [outputs/relax/<date>/<time>]` | output directory |
| `--restart PATH` | continue from a `checkpoints/state_<step>.h5` of the same geometry, mesh, degree and precision |

The CFL cap (0.5), the velocity smoothing (order 1, scale 0.075 $h_r^2$) and the Newton solve's passes (one) are constants of the library; the paper's variants of them (the auxiliary B field, the helicity correction, the potential velocity, other smoothings) are `scripts/paper_scripts/relax_paper.py`'s flags, the same run on an extended configuration.

The initial condition is always Leray-projected. The run stops when the
mean over the last `W` steps of the squared normalised force residual
`||F||²_M / ||grad(B²/2)||²` falls below `--floor-tol`
(`force_floor_reached`), or when a budget runs out. The residual is not
monotone, so the window mean is the quantity, never the last value.
Calibration: on the W7-X Clebsch run at `(8,16,8)`, `p = 3`, float64, the
residual reaches `2.9e-6` at step 500 and floors around `1e-6` by step
1000-3000. A float32 run's solves are refined against a float64 residual
(`precision.md`), so its floor is no longer the solve tolerance (until
2026-09-04 it was, `~4e-6` at tol `1e-5`).

Output: `relax.json` with the parameters, the per-step trace (`dE` the
exact energy change of the step, `dE_ls` the line search's prediction,
`F`, `resid`, `dt`, `dt_star`, `cfl`, `div`, `cos`, `gain`), the sampled quantities of interest
`qoi` (`it`, `wall`, `F`, `resid`, `helicity`, `JoverB`, `JB`, and the
pressure diagnostics of section 3: `gradp_cmp`, `p_cmp`, `weak_resid`,
`dpdn_wall`, `JxBn_wall`, `beta_vol`, `beta_axis`), the initial field's
numbers `ic` and the `summary` with the stopping reason; and
`checkpoints/best.h5`, the field with the lowest per-step residual
of the run (`State.B_best`, kept inside the compiled loop at the cost of one
`where` per step; the run's answer when it went past its floor), and
`checkpoints/state_<step>.h5`, the `State` at every chunk boundary and at
step 0 (`write_checkpoint` / `read_checkpoint`), from which the plotters
read the field and the strong pressure. The loop itself is
`mrx.relaxation.relax(state, ts, steps, chunk, ...)`: `chunk_runner` for
the steps, `make_sampler` for the diagnostics, the floor, wall-budget and
reconnection rules, and an `on_chunk` callback the driver uses to write;
the script is its command line plus `build_sequence`, `initial_field` and
the JSON writer.

At the reference resolution (W7-X FMM002, `(8,16,8)`,
`p = 3`, float64, one H100): setup about 330 s, first step about 90 s of
compilation, then 0.7-0.9 s per step.
