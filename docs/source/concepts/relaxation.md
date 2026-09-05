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
velocity_smoothing_scale, history_size, cfl, scheme)` is an `eqx.Module`;
`relaxation_step(state)` does one step, explicit Euler on the descent
velocity, `B_{n+1} = B_n + dt · curl(u × X)`, or the midpoint rule below.
The step is ideal: reconnection is a separate resistive solve between
chunks (section 2a).

1. `compute_force` at `B_n`; `MF = M_2 F` once. It serves `||F||_M` and the
   L-BFGS secant.
2. Direction `u` by `history_size`:
   - `0`: steepest descent, `u = F`.
   - `1` (default): `_lbfgs_direction(F, s, y, Ms, My)`,
     the two-loop recursion in the `M_2` inner product. With one pair and the
     exact line search this is Polak-Ribière CG (the classical memoryless-BFGS
     identity; measured identical on W7-X), and a pair with `<s, y>_M <= 0` is
     skipped -- the PR+ restart. Longer histories add nothing measurable. The state stores `M s` and `M y` next to `s`
     and `y`, so the recursion applies `M` zero times. The descent variable
     is the velocity: `s = dt · u`, `y = F_prev - F`.
3. `smooth_velocity(u)`: `velocity_smoothing_order` times
   `u = (M_2 + mu L_2)^{-1} M_2 u` with `mu = velocity_smoothing_scale`, the
   smoothed direction `v = (I - mu Δ)^{-order} F`. Off at order 0.
4. `u, p_v = seq.apply_leray_projection(u, k=2)`: the flow is
   incompressible.
5. `E = M_1^{-1} cross_product_load(u, X, ...)`: the ideal electric
   field, one k=1 mass solve (`X` as in section 1).
6. `dB = seq.apply_incidence_matrix(E, 1)`: the topological curl, so
   `div B` is conserved to `1e-16` along the trajectory.
7. `dt_star = F·Mu / ||dB||²_M`, the minimiser of the energy along the
   direction and the largest step that still lowers `E` (the analytic line
   search). Then the CFL cap, `dt = min(dt_star, cfl / cfl_max)`:
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
8. `B_{n+1} = B_n + dt · dB`.

`M_2` is applied three times per step (`M F`, `M u`, `M dB`) whatever the
history. With the line search `dE/dt <= 0` is a guarantee: the step is the
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

The descent never applies it inside the step. `scripts/relax.py
--reconnect-every K` runs the ideal descent, and every `K` steps
checkpoints the field, applies one such solve, and restarts the optimiser
on the diffused field (`initial_state`): reconnection as a discrete event
whose dose is a physics choice, how much helicity to spend, never a
stability one. The dose is set by that budget: to first order the
increment changes the helicity by `dH = -2 eps ∫ J·B`, so `eps = X |H| /
(2 |∫ J·B|)` spends the fraction `X` (`--reconnect-helicity`), and the
run records the helicity actually spent next to the target. The ideal
descent between solves is a power law in the step, never a plateau
(`docs/research/li383_sweep_results_2026-09-02.md`, section 5h), so the
interval is a choice, not something to detect.

**Midpoint-implicit induction** (`IntegrationScheme.IMPLICIT_MIDPOINT`,
`--scheme midpoint`) keeps steps 1-7 and replaces step 8 by
`B_{n+1} = B_n + dt · curl(u × X_mid)` at the midpoint field
`(B_n + B_{n+1})/2`, with the predictor's velocity `u` and `dt`. With the
auxiliary field, `X_mid = H_mid = M_1^{-1} P (B_n + B_{n+1})/2`, the 1-form
proxy of the midpoint field: the auxiliary-variable scheme, which exists
for one exact identity.
The discrete helicity `<A, B + B_harm>` of `compute_helicity` is conserved
by the semi-discrete flow for ANY velocity: the pairing of the 2-form `B`
with a 1-form `E` goes through the proxy `H = M_1^{-1} P B`, so
`E^T P B = H^T load(u × H) = ∫ H_h · (u_h × H_h) = 0` at every quadrature
node, and with the exact discrete Stokes identity
`d/dt <A, B + B_harm> = 2 <B, E> = 0` (`B_harm` is constant, `dB/dt` lies
in `range D_1`). The helicity is a quadratic form in `B`, and evaluating
`E` at the midpoint field keeps it exactly, whatever `u` is; the explicit
scheme's drift is entirely the time-integration error of evaluating `H`
at `B_n`. One condition: `E^T P B = E^T M_1 H` needs `E` and `H` in the
same 1-form space, which is why the auxiliary `H` is a Dirichlet 1-form
like `E` (so that `D_1 E` keeps `B · n = 0`): with a natural `H` (the
proxy of a wall-tangent `B` has a tangential trace) the load `load(u × H)`
loses its tangential wall DoFs on the way to `E` and both schemes leak
helicity through that wall layer at the same rate (li383 (8,16,16) p=2,
float64, 1000 L-BFGS steps: -5.5e-7 explicit, -6.6e-7 midpoint); with the
Dirichlet `H` the midpoint scheme is exact to the solves (+2.2e-7
explicit, +5e-12 midpoint), at the price of `H_t = 0` at the wall. Without
the auxiliary field (`X = B`) the midpoint scheme has no time error either;
what remains is the grid's projection error of the pairing, about 1e-6
on that mesh. See `docs/research/implicit_midpoint_2026-09-04.md`. The
energy change is `-dt <u, F_mid>_M` with the force at the
midpoint field: descent while the predictor's velocity still correlates
with the midpoint force, second order in `dt`, not the line search's
guarantee.

The velocity stays explicit on purpose. Taking `u` at the midpoint too
makes the step a nonlinear fixed point in `B` through the force, whose
linearisation is the descent operator `|H|² curl curl` with largest
eigenvalue `|H|²/h²`; on li383 (8,16,16) p=2 the line-search `dt` is 35x
above the Picard contraction limit (the iterates blow up in six sweeps,
Anderson acceleration and a Laplacian preconditioner do not rescue it
because the operator is soft on the force-free perturbations the
Laplacian is stiff on, and Newton is a Krylov solve inside a Krylov
solve). With `u` frozen the map `x -> dt · curl(u × H(B_n + x/2))` is
linear in the increment `x` with contraction constant `dt |u| / 2h`, small
because `u` is the force, so plain Picard (`_midpoint_solve`) converges in a
few sweeps of one k=1 mass solve for `H_mid`, one for `E` and the
topological curl, warm-started from the previous sweep. Convergence is
judged on the defect `||g(x) - x||_M` relative to the predictor's increment
`||dt · dB(B_n)||_M` (the defect form again) against `picard_tol`,
`PICARD_TOL_FACTOR` (10) times `seq.tol` because the inner solves define
the map. Should the defect blow up (`PICARD_BLOWUP` times the predictor's
increment, NaN included) or `PICARD_MAX` (20) sweeps not converge, `dt` is
halved and the solve restarts from the predictor, at most
`PICARD_RESTARTS` (4) times, after which the step goes out unconverged
with `state.picard_residual` above the tolerance; no run has ever halved.
The state records `picard_iterations` (1 for the explicit
step, the predictor plus the sweeps otherwise), `picard_restarts` and
`picard_residual`; `F`, `u` and the L-BFGS pair are the predictor's, as in
the explicit step, and `H`, `E` carried as warm starts are the midpoint's.
Cost: one explicit step plus a few pairs of k=1 mass solves (one, for
`E`, without the auxiliary field).

`State` holds `B_n`, `B_nplus1`, `v`, the warm-start guesses (`p`, `p_v`,
`H`, `JxH`, `J`, `E`, `A`), `F_prev`, `MF_prev`, the four history arrays,
`dt`, `dt_star`, `cfl_max`, `F_norm`, `v_norm`, `lbfgs_sy`,
`picard_iterations`, `picard_restarts`, `picard_residual`. Build it with `initial_state(B_dof, ts, dt)`, which
runs one `compute_force` so the first secant and CG coefficient see a true
previous gradient. `relax(state, ts, steps, chunk, ...)` runs the steps in
`jax.lax.scan` chunks of `chunk` (`chunk_runner`), samples the diagnostics
once per chunk (`make_sampler`: helicity, the two pressures, beta), applies
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
`scripts/poincare_relax.py --pressure weak|strong` (default `weak`) draws
either one: `p_w` is a 0-form, its value is the spline evaluation
and it is not shifted; `p` is a 3-form, `p / det DF`, shifted so the
outermost kept line reads zero. `scripts/plot_relaxation.py` draws `p_w`
on the torus and in poloidal cuts (`mrx.plotting.plot_torus`,
`plot_crossections_separate`) and `‖F‖_M` against `E` on twin axes
(`plot_twin_axis`); `scripts/compare_relaxations.py OUT label=run ...`
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

`build_sequence(geometry, ns, p, maxiter, tol, nfp, r_windows)` in
`mrx/geometry.py` returns `(seq, ops)` with the map installed and every
solver operator built. `geometry` is the path of a GVEC state (`.dat`) or
a VMEC wout (`.nc`) (`build_gvec_map` in `mrx/gvec.py`), or of an analytic
geometry file (`.json`, `read_analytic`: a map of `mrx/mappings.py`,
`torus`, `cylinder` or `rot-ellipse`, with its parameters and the profiles
of the analytic initial condition; `data/torus.json` and its siblings are
the shipped ones). Anything else raises. `nfp` overrides an equilibrium
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
| `--geometry PATH` (required) | a VMEC wout (`.nc`), a GVEC state (`.dat`) or an analytic geometry (`.json`): the geometry and the initial condition (sections 4, 5) |
| `--nfp N [file value]` | field periods, for a file that declares them wrong |
| `--ns R,T,Z [8,16,16]`, `--p P [2]` | resolution (also the map's) and degree |
| `--r-refine a:b:m,... [""]` | radial refinement windows, `m` uniform cells in each `[a, b]` (`radial_knots`) |
| `--solve-maxiter N [2000]`, `--solve-tol TOL [sqrt(eps)]` | budget and tolerance of every inner solve |
| `--precision {float32,float64} [float32]` | exported as `MRX_DTYPE` before `mrx` is imported |
| `--seed m,n,rho0,width [""]`, `--seed-eps EPS [0]` | equilibrium files only: adds the resonant term `eps |Φ'(rho0)|/m · g(rho) cos(2π(m θ − s n ζ))` to `A'_ζ` (`g` a Gaussian of that width tapered to zero at the wall, `s` the sign of the file's iota) before `B = dA'`, so `div B = 0` and `B·n = 0` stay exact; `EPS` is the resonant normal field `|δB^ρ|/|B^ζ|` at `rho0`, the chain sits where `|iota| = nfp n / m` (`resonant_rho`, printed) and opens an island of full width about `1.6 sqrt(EPS nfp/(m |iota'|))` in `rho`. A stability probe: under ideal descent the topology is frozen, so a seeded island that grows to an `EPS`-independent width marks a tearing-unstable surface, one that shrinks back to the seed width a stable one -- sweep `EPS` |
| `--auxiliary-B-field {false,true} [false]` | `false` reads the 2-form `B` itself in both cross products; `true` routes them through the auxiliary Dirichlet 1-form `H = M_1^{-1} P B` (section 1), the variable that makes the midpoint scheme conserve the discrete helicity exactly |
| `--scheme {explicit,midpoint} [explicit]` | forward Euler, or midpoint-implicit induction with the explicit velocity (section 2): Picard on the increment to `PICARD_TOL_FACTOR` times the solver tolerance, `dt` halved after `PICARD_MAX` sweeps or a blow-up, at most `PICARD_RESTARTS` times; the trace records `picard_it`, `picard_resid` |
| `--history M [1]` | L-BFGS secant pairs; 0 is steepest descent, 1 memoryless BFGS (= CG) |
| `--velocity-smoothing-order G [0]`, `--velocity-smoothing-scale MU [0.0]` | `v = (I - MU L)^{-G} F` |
| `--cfl C [0.5]` | the CFL cap on the line-search step |
| `--chunk N [500]` | steps per compiled chunk (one `lax.scan`, `mrx.relaxation.chunk_runner`; the per-step trace is the scan's stacked output, the state its carry): once per chunk the qoi are sampled (section 3), the checkpoint `checkpoints/state_<step>.h5` and `relax.json` are written, and the floor, reconnect and wall-time tests run; `--steps` is a multiple of it. The checkpoints serve `scripts/poincare_relax.py --fields snapshots`, which traces every stored step at the chosen plane and writes one frame per step with every axis, colour scale and the split line held fixed (`render_section(limits=...)`); `ffmpeg -framerate 4 -i frame_zeta0.5_%04d.png -c:v mpeg4 -q:v 2 movie.mp4` assembles them (`--snapshot-steps 0:500:2,500:2501:8` renders a subset, dense where the flow is fast; if the system ffmpeg lacks H.264, `pip install imageio-ffmpeg` provides one with libx264) |
| `--steps N [3000]`, `--seconds S [none]` | outer guards |
| `--floor-tol TOL [1e-3]` | stopping criterion: the last chunk's mean relative force residual below it |
| `--reconnect-every K [0]`, `--reconnect-helicity X [0.01]` | the reconnection series (section 2a): every `K` steps (rounded to whole chunks) the field is written to `<out>/reconnect/<k>/` (`B.h5` in the layout of the run's, `state.eqx` to `--restart` from) and reconnected by one `resistive_step` spending the fraction `X` of the helicity, after which the descent restarts on the diffused field; `results["reconnect"]` records each solve with the helicity actually spent, `scripts/poincare_relax.py --fields ic,final,reconnect` traces the series on one colour scale |
| `--out DIR [outputs/relax/<date>/<time>]` | output directory |
| `--restart PATH` | continue from a `checkpoints/state_<step>.h5` |

The initial condition is always Leray-projected. The run stops when the
mean over the last `W` steps of the relative force residual
`||F||_M / ||grad(B²/2)||` falls below `--floor-tol`
(`force_floor_reached`), or when a budget runs out. The residual is not
monotone, so the window mean is the quantity, never the last value.
Calibration: on the W7-X Clebsch run at `(8,16,8)`, `p = 3`, float64, the
residual reaches `1.7e-3` at step 500 and floors around `1e-3` by step
1000-3000. In float32 the residual floors at the solve-tolerance level
(`~2e-3` at tol `1e-5`), so a `--floor-tol` below that never fires.

Output: `relax.json` with the parameters, the per-step trace (`dE` the
exact energy change of the step, `dE_ls` the line search's prediction,
`F`, `resid`, `dt`, `dt_star`, `cfl`, `div`, `cos`, `gain`, `picard_it`,
`picard_resid`), the sampled quantities of interest
`qoi` (`it`, `wall`, `F`, `resid`, `helicity`, `JoverB`, `JB`, and the
pressure diagnostics of section 3: `gradp_cmp`, `p_cmp`, `weak_resid`,
`dpdn_wall`, `JxBn_wall`, `beta_vol`, `beta_axis`), the initial field's
numbers `ic` and the `summary` with the stopping reason; and
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
