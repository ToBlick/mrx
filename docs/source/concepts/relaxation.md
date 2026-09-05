# Relaxation

`mrx/relaxation.py` descends the magnetic energy `E = ½ ||B||²_{M_2}` of a
divergence-free 2-form `B` with `B · n = 0` under an incompressible,
helicity-conserving flow. The fixed point is `J × B = ∇p` with `p` the Leray
multiplier: a finite-beta equilibrium. `scripts/relax.py` is the driver
([Solve a relaxation problem](../relaxation.md)).

## 1. The force

`compute_force(B, seq, auxiliary_B_field, p_guess, H_guess, JxH_guess,
J_guess, F_guess)` returns `(F, p, J, X, JxX)`:

1. `J = apply_weak_curl(B)`: the current, one k=1 mass solve.
2. `X`, the field the cross products read. Without the auxiliary field
   (the default) it is the 2-form `B` itself. With it, `H = M_1^{-1} P_{21}
   B`: the field as a Dirichlet 1-form (`H_t = 0` on the wall), a second
   k=1 mass solve, the auxiliary variable of the helicity-conserving scheme
   below.
3. `JxX = M_2^{-1} cross_product_load(J, X, ...)`: a k=2 mass solve, in the
   residual precision.
4. `F, p = seq.apply_leray_projection(JxX, k=2, p_guess=p_guess,
   sigma_guess=sigma_guess)`: removes the gradient part with one k=3 Hodge
   solve; `p` is the pressure. The solve is the saddle MINRES in `(p,
   sigma)` and its lower unknown `sigma = M_2^{-1} D_2^T p` IS the gradient
   part, so `F = JxX - sigma` costs no further mass solve and is formed in
   float64 before it is stored ([precision.md](precision.md)).

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
     identity), and a pair with `<s, y>_M <= 0` is skipped -- the PR+ restart.
     Longer histories add nothing measurable. The state stores `M s` and
     `M y` next to `s` and `y`, so the recursion applies `M` zero times. The
     descent variable is the velocity: `s = dt · u`, `y = F_prev - F`.
3. `smooth_velocity(u)`: `velocity_smoothing_order` times
   `u = (M_2 + mu L_2)^{-1} M_2 u` with `mu = velocity_smoothing_scale`, the
   smoothed direction `v = (I - mu Δ)^{-order} F`. Off at order 0. The
   default scale is `SMOOTHING_C / n_r^2 = 0.02 / n_r^2`: the two-cell
   mode damped by 1/1.2, nothing resolved touched
   (`docs/research/mu_sweep_2026-09-05.md`). The flow is incompressible
   without a projection of its own: the force is Leray-projected, the
   L-BFGS direction combines projected forces and their steps, and the
   smoothing commutes with the divergence
   (`docs/research/velocity_leray_ab_2026-09-04.md`).
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
   `TimeStepper` field (`--cfl`, default 0.5; `inf` disables the cap). The
   minimiser cannot raise the energy, but a large `dt_star` leaves the
   ideal-induction flow (frozen-in topology violated at `O(dt²)`) and
   diverges when `||dB||` collapses. `state.dt_star` and `state.cfl_max`
   record the cap's activity.
7. `B_{n+1} = B_n + dt · dB`.

`M_2` is applied three times per step (`M F`, `M u`, `M dB`) whatever the
history. With the line search `dE/dt <= 0` is a guarantee: the step is the
line minimiser. The force residual is not monotone, which is why the
stopping criterion averages it over a window.

### 2a. Reconnection

`resistive_step(B, seq, eps)` is one backward-Euler step of
`dB/dt = -eta curl curl B` over `eps = eta dt`, in defect form:
`(M_2 + eps L_2) delta = -eps L_2 B`, `B + delta`, with initial guess 0,
through `apply_inverse_mass_plus_eps_laplace_matrix` (k=2, Dirichlet, two
SPD CG solves through the split identity with the shifted-stiffness atom).
The defect form is what keeps the solve meaningful in float32: the
tolerance is relative to `delta`, not to `B`. The solve is unconditionally
stable and dissipative and maps `ker(div)` into itself, so `div B` stays at
the solver's tolerance.

The descent never applies it inside the step. `relax(reconnect_every=K)`
runs the ideal descent and, every `K` steps, checkpoints the field, applies
one such solve, and restarts the optimiser on the diffused field
(`initial_state`): reconnection as a discrete event whose dose is a physics
choice, how much helicity to spend, never a stability one. To first order
the increment changes the helicity by `dH = -2 eps ∫ J·B`, so `eps = X |H| /
(2 |∫ J·B|)` spends the fraction `X` (`reconnect_helicity`), and the run
records the helicity actually spent next to the target. The ideal descent
between solves is a power law in the step, never a plateau
(`docs/research/li383_sweep_results_2026-09-02.md`), so the interval is a
choice, not something to detect.

**Midpoint-implicit induction** (`IntegrationScheme.IMPLICIT_MIDPOINT`,
`--scheme midpoint`) keeps steps 1-6 and replaces step 7 by
`B_{n+1} = B_n + dt · curl(u × X_mid)` at the midpoint field
`(B_n + B_{n+1})/2`, with the predictor's velocity `u` and `dt`. With the
auxiliary field, `X_mid = H_mid = M_1^{-1} P (B_n + B_{n+1})/2`: the
auxiliary-variable scheme, which conserves the discrete helicity `<A, B +
B_harm>` of `compute_helicity` exactly for ANY velocity, because the pairing
of `B` with a 1-form `E` goes through the proxy `H` in the same Dirichlet
1-form space (`E^T P B = H^T load(u × H) = 0` at every quadrature node) and
the helicity is a quadratic form in `B`. With a natural `H` both schemes
leak helicity through the wall layer alike; without the auxiliary field
(`X = B`) what remains is the grid's projection error of the pairing. The
velocity stays explicit on purpose: taking `u` at the midpoint too makes
the step a nonlinear fixed point whose line-search `dt` sits far above the
Picard contraction limit, while with `u` frozen the map is linear in the
increment and plain Picard (`_midpoint_solve`) converges in a few sweeps of
one k=1 mass solve each, judged on the defect relative to the predictor's
increment against `picard_tol`; `dt` is halved and the solve restarted on a
blow-up, at most `PICARD_RESTARTS` times. The derivation, the measured
numbers and the failed alternatives are in the `_midpoint_solve` docstring
and `docs/research/implicit_midpoint_2026-09-04.md`.

`State` holds `B_n`, `v`, the warm-start guesses (`p`, `H`, `JxH`, `J`,
`E`, `A`), `F_prev`, `MF_prev`, the four history arrays, `dt`, `dt_star`,
`cfl_max`, `F_norm`, `v_norm`, `picard_iterations`, `picard_residual`.
Build it with `initial_state(B_dof, ts, dt)`, which runs one
`compute_force` so the first secant sees a true previous gradient.
`relax(state, ts, steps, chunk, ...)` runs the steps in `jax.lax.scan`
chunks of `chunk` (`chunk_runner`), samples the diagnostics once per chunk
(`make_sampler`: the energy of the stored field in the residual
precision, the helicity, the two pressures, beta),
applies the floor, wall-budget and reconnection rules and returns a
`RelaxResult` (the state, the per-step trace, the per-chunk samples, the
reconnection records); `write_checkpoint` / `read_checkpoint` store and
restore a state.

## 3. Diagnostics

- `compute_helicity(B, seq, A_guess)`: one k=1 Hodge solve,
  `A = L_1^{-1} D_1^T B`, then `H = A · P_{21}(B + B_harm)` with
  `B_harm = B - curl A`. The right-hand side is the dual 1-form `D_1^T B`,
  not the weak curl. `||B_harm|| <= ||B||` is an identity (`b_2 = 1` in the
  Dirichlet complex) and a check on the solve.
- `compute_divergence_norm(B, seq)`: `||G_2 B||` through the incidence
  operator, so it measures the field and not a solver residual.
- The energy is `0.5 * seq.l2_norm_sq(B, 2)`, sampled per chunk in the
  residual precision (`qoi["E"]`); the trace's `dE` is the exact energy
  change of each step.
- `weak_pressure(J, X, seq, auxiliary_B_field, p_guess)` and
  `pressure_diagnostics(B, p, p_w, F_w, v, seq)`: the second pressure and
  the plasma beta, below.

### Two pressures

The relaxation has two pressures. They agree at a fixed point whose
pressure is constant on the wall and differ everywhere else, in a way
that is itself a diagnostic.

**Strong** (`p`, from `compute_force`): the Leray multiplier of the
constrained energy principle. `J × X` is projected onto the Dirichlet
2-form space first, which discards its normal component, and the k=3 Hodge
solve of `apply_leray_projection(k=2)` removes the gradient part with
`σ · n = 0` built in. So `p` satisfies `dp/dn = 0` on the wall by
construction and is defined up to a constant. It is the right multiplier
for the descent: `F = J × X - ∇p` is exactly the force the constrained
flow (`u · n = 0`) can see. It is blind to the wall force.

**Weak** (`p_w`, from `weak_pressure`): `J × X` is projected onto the
NATURAL 1-form space, `v = M_1^{-1} load(J × X)` (no boundary condition,
so `v · n` is `(J × X) · n`), and Helmholtz-decomposed there,
`v = F_w + ∇p_w`, with `p_w` in the Dirichlet 0-form space:
`(∇φ, ∇p_w) = (∇φ, v)` for every `φ` with `φ = 0` on the wall
(`apply_leray_projection(v, k=1, dirichlet_p=True)`, one k=0 Dirichlet
Laplacian solve). `p_w = 0` on the wall by construction, so it has no
gauge; `F_w` is divergence-free in the interior and keeps its normal
trace, and on the wall `(J × X) · n = dp_w/dn + F_w · n`. At a fixed point
`J × X` is a gradient, `F_w` vanishes, and `dp_w/dn` is the wall force.

Read `p` for the descent and the force residual; read `p_w` for the
pressure profile, the wall force and beta. `relax` records at every qoi
sample:

| key | definition |
|---|---|
| `gradp_cmp` | `‖Π_2 ∇p_w - ∇_w p‖_{M_2} / ‖Π_2 ∇p_w‖_{M_2}`, gauge-free. `∇_w p` is the weak gradient of the 3-form in the Dirichlet 2-form space (the `σ` the Leray step subtracts): the L2 projection of the true gradient onto that space, so its normal trace is zero whatever `dp/dn` is. `∇p_w` is the exact strong gradient of the 0-form, projected onto the same space, `Π_2 = M_2^{-1} P_{12}`, so both sides lose the same normal trace and the ratio compares the pressures, not the projection |
| `p_cmp` | `‖(p/J - ⟨p/J⟩) - (p_w - ⟨p_w⟩)‖_{L2} / ‖p_w - ⟨p_w⟩‖_{L2}` at the quadrature points, `⟨·⟩` the volume mean: the two pressures as functions, the strong one's gauge removed |
| `weak_resid` | `‖F_w‖_{M_1} / ‖v‖_{M_1}`: the part of `J × X` that is not the gradient of a function vanishing on the wall |
| `dpdn_wall` | `max |dp_w/dn|` over the wall (`r = 1` at the angular quadrature points) relative to `max |∇p_w|` over the quadrature points; `p_w = 0` on the wall, so its gradient there is normal |
| `JxBn_wall` | `max |(J × X) · n|` on the same wall points, from `v`, relative to the same `max |∇p_w|`: the wall force the strong pressure cannot see |
| `beta_vol` | `⟨p_w, 1⟩_{M_0} / E` with `E = ½ B^T M_2 B = ∫ B²/2 dV`. Code units: the magnetic pressure is `B²/2`, so `β = ∫ p dV / ∫ B²/2 dV` |
| `beta_axis` | `⟨p_w⟩ / ⟨|B|²/2⟩` on the COORDINATE axis, logical `r = 0`: both averaged (quadrature weights) over the innermost radial quadrature layer, all `θ` and `ζ`. The 2-form's magnitude is 0/0 on the polar axis itself, so a limit `r → 0` reads the solver's residual there |

The strong pressure `p` is state (a field of `State`, so it is in every
checkpoint); the weak pressure is a diagnostic, computed by the sampler at
every chunk and by the plotters on demand (two solves per field).
`scripts/poincare_relax.py --pressure weak|strong` (default `weak`) draws
either one: `p_w` is a 0-form, its value is the spline evaluation and it is
not shifted; `p` is a 3-form, `p / det DF`, shifted so the outermost kept
line reads zero.

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
| `initial_field(seq, seed)` | the field the sequence's geometry file decides on (below), `||B||_M = 1`, with the numbers a driver records |
| `make_profiles(iota0, iota1, iota_exp, flux_exp)` | `ι = ι₀ + (ι₁-ι₀) ρ^e`, `Φ' = ρ^q` |
| `make_lambda(modes)` | `λ` from `[(m, n, amp), ...]` |
| `analytic_profile_form(iota, dPhi, dlam)` | the reference 2-form above; the initial condition of an analytic geometry file, whose `profile` block supplies the numbers |
| `clebsch_potential_form(cb, seed)`, `potential_two_form(seq, A_ref)` | the reference 1-form `A' = (-LA dPhi_dr, 2π Φ, -(2π/nfp) χ)` (the GVEC potential with the gauge term `d(Φ LA)` dropped; `Φ`, `χ` integrated from the profiles) histopolated on the FREE 1-form space -- its wall trace `2π Φ_edge` is the toroidal flux, the Dirichlet harmonic content -- and `B = dA'` by the exact incidence curl into the Dirichlet 2-form space: `div B = 0` to round-off, no Leray step, and no derivative of the sampled `LA` is ever taken (the discrete `d` differentiates); the initial condition of every equilibrium file. `seed = (m, n, rho0, width, eps)` adds a resonant term to `A'_ζ` that opens an island at the `|iota| = nfp n / m` surface (`resonant_rho`) |
| `project_reference_two_form(seq, omega_ref)` | pushes forward `B = DF ω / J` and L²-projects onto the Dirichlet k=2 space |
| `leray_clean(seq, B)` | removes the projection's divergence |

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
of the analytic initial condition). Anything else raises. The file is
parsed once and kept on the sequence as `seq.equilibrium`, which
`initial_field` and `load_clebsch` read. `nfp` overrides an equilibrium
file's value for a file that declares it wrong. `build_gvec_map` measures
the handedness of the file and mirrors it so that `det DF > 0`. Nothing
is resolved from names or from the environment: every reader takes the
path.

The command line, the flags and the run layout are in
[Solve a relaxation problem](../relaxation.md).
