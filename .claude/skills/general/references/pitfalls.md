# Pitfalls and measured lessons

The first half is `docs/source/sharp_bits.md` in short. The second half is lessons from runs.

## Sharp bits

- **Precision at import.** `MRX_DTYPE` is read once at the first `import mrx`. `relax.py
  --geometry.precision` sets it before importing. Every checkpoint records the precision.
- **float32 means double-single.** In float32 a solve is refined until its double-single residual meets the
  tolerance. A stall at 1e-7 means something accurate was formed in float32 (a projection, `G^T w`, a seed
  interpolation, a normalisation): lift it. More iterations never fix a precision floor.
- **Signed counts.** Negative = not converged. Laplacian and pressure counts sum up to six passes, so
  they can exceed `solve-maxiter`.
- **Harmonic forms.** `[nullspace]` prints `v^T L v / v^T M v`: round-off (<= 1e-9) is good, ~1e-3
  means the solve behind it failed and everything built on it is off.
- **Parity views.** A field's norm on `seq` instead of `seq.odd`/`seq.even` is a different (wrong)
  number with no error.
- **Axis.** `det DPhi = 0` at r = 0: evaluate near, not on. Averages skip the first radial span.
- **Wall.** r = 1 evaluates fine. `jnp.clip(r, 0, 1)` halves the autodiff derivative at r = 1: pass r as is.
- **Angles.** VMEC iota reads negative after the `theta -> -theta` flip.
- **Checkpoints** record their discretisation (resolution, degree, nfp, symmetry, graded, `map_theta`,
  precision) and are refused on another one. Rebuild the sequence from `checkpoint_attrs`.
- **`1/nfp`.** MRX zeta spans one period. A missed `1/nfp` makes iota nfp times too large and passes
  every structural check. A wrong `nfp` wraps one period through the wrong angle with a healthy Jacobian.
  A profile in `s` read as `r = sqrt(s)` distorts the radial structure but leaves iota alone.
- **Frames.** `seq.load(f, k)` takes physical components. `seq.interpolate(f, k)` takes logical components.
- **Compilation.** First call compiles. Static Python numbers in eqx modules recompile on change. A
  pytree map (`CylindricalMap`, `FrameMap`) reuses compiled solves after `set_geometry_from_map` (0.14 s vs
  18 s compile). A map given as a plain Python function does not.
- **Force residual is not monotone**, and every run reaches a discretisation floor set by unresolved
  current sheets on rational surfaces. Steps along the floor only reconnect at grid scale. With the
  resistive drive the energy is not monotone either.
- **Not reproducible to the last digit.** Round-off separates trajectories by about a decade per ten
  steps on li383. Compare energy and floor. Validate kernel changes per apply (tests at ~1e3 eps) and by
  energy and step time.
- **Ideal relaxation keeps topology.** It moves islands but neither opens nor closes them, and conserves
  helicity up to grid-scale drift (Tutorial 3: dH/H0 = -2.4e-4 over 20 Newton steps). Islands need a
  seed, losing them needs the resistive drive. A drive whose start already has the reference current
  does nothing: smooth the reference (`--drive.reference-smoothing`) when start = reference.
- **Poincare.** Needs one-signed `B^zeta`. Axis-nearest iota is unreliable (axis off r = 0). `drift`
  is meaningful only on regular lines (on chaotic lines it measures the Lyapunov exponent).

- Host dispatch: calling a filter_jit function of the sequence costs ~10 ms per call whatever it computes.
  Loops of such calls on the host (probes, diagnostics) are slow for that reason, batch them inside one call.
- At (12,16,16)-(16,32,32) GPU matvecs are kernel-count bound (~2 us per kernel), at 32^3 the mass apply is
  memory bound. Count kernels before optimising flops.

## Lessons from runs

- **Newton null space.** The Hessian vanishes on field-aligned flows `u = f B`, a continuum on rational
  surfaces. Without the parallel penalty Newton divides force round-off by it and destroys surfaces past
  the floor. Penalty `3 x strain` fixes it on li383 and W7-X. Any Newton preconditioner work must treat
  this null space explicitly. Krylov extremes are not modes (use LOBPCG for the low end).
- **MINRES, not CG, for Newton**, with its Lanczos vectors reorthogonalised (1.6-2.6x fewer iterations).
  The mass solves inside the Hessian need 1e-6: at 1e-4 the relaxation stalls.
- **Islands and seeds.** Seeds narrower than ~3 radial cells are removed by numerical reconnection.
  High-m seeds are damped by the mesh. Seed `n` counts field periods (chain at `|iota| = nfp n / m`).
  Never quote a seeded width without an amplitude sweep. The residue is not a width (it overestimates
  1.35-1.6x), use it for existence and phase only.
- **Initial conditions matter more than the solver.** Check the file's own initial force residual before
  blaming the relaxation.
- **Floors.** Set `--budget.floor-tol` well below the expected floor and let `--budget.steps` end the
  run, otherwise tables show the floor-tol instead of the bottom. Judge h-refinement by the floor.
- **Two pressures.** Strong `p` (Neumann Leray, multiplier of the descent, blind to the wall force) vs
  weak `p_w` (Dirichlet, zero on the wall). Profiles and `beta_vol` use `p_w`.
- **Open questions** live in `docs/research/`. Check there before "discovering" a known effect.
