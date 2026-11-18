# Pitfalls

Most traps are explained in the docs. This page names each one and points to its page under `docs/source/` or to
`docs/research/`. Read the page before acting on the trap. What follows the pointers is only what helps when working
in the code and does not belong in user docs.

## In the docs

| trap | where |
|---|---|
| a negative iteration count, counts above maxiter | `sharp_bits.md` "A negative iteration count" |
| harmonic forms zero until `compute_nullspaces`, the `[nullspace]` Rayleigh line | `sharp_bits.md` "The harmonic forms are computed separately" |
| a norm on the wrong parity view | `sharp_bits.md` "Fields live on the view of their parity" |
| evaluating on the axis | `sharp_bits.md` "The map is singular on the axis" |
| `jnp.clip(r, 0, 1)` halves the derivative at the wall | `sharp_bits.md` "The wall r = 1 is safe" |
| VMEC iota reads negative | `sharp_bits.md` "The angles are right-handed" |
| recompiles from static Python numbers, non-pytree maps, `--geometry.max-batch` | `sharp_bits.md` "Compilation" |
| `\|\|F\|\|` rises, the discretisation floor | `sharp_bits.md` "The force residual need not decrease" |
| trajectories differ from round-off | `sharp_bits.md` "Runs are not reproducible to the last digit" |
| islands need a seed, losing them needs the drive, a drive with start = reference does nothing | `sharp_bits.md` "The ideal relaxation keeps the topology" |
| one-signed `B^zeta`, axis iota, `drift`, `locked_width` resolution | `sharp_bits.md` "Poincare sections" |
| precision fixed at import, float32 = double-single | `concepts/precision.md`, `faq.md` "Which precision" |
| a float32 solve that stalls near 1e-7: something accurate was formed in float32, lift it | `concepts/precision.md` "Double-single", `docs/research/solvers_precision.md` 8 |
| a missed `1/nfp`, a wrong `nfp`, `s` read as `r`, `load` vs `interpolate` components | `concepts/equilibrium_input.md` 4 |
| a new file: check `div B`, the discarded wall part, iota, the initial force residual | `concepts/equilibrium_input.md` 5 |
| strong `p` vs weak `p_w` | `concepts/relaxation.md` "Two pressures" |
| the field-aligned null space of the Newton Hessian, the penalty | `concepts/relaxation.md` "Newton", `docs/research/relaxation.md` 3 |
| MINRES needs its Lanczos vectors reorthogonalised, mass solves in the Hessian at 1e-6 | `concepts/relaxation.md` "Newton" |
| a seed narrower than about 3 radial cells is reconnected away | `docs/research/relaxation.md` 5 |
| the residue overestimates an island width 1.35-1.6x | `docs/research/poincare_islands.md` 3 |
| about 10 ms host dispatch per jitted call, small meshes kernel-count bound | `docs/research/performance.md` 3 |

## Working in the code

- **Checkpoints are tied to their discretisation.** `read_field` raises on a checkpoint written on another mesh,
  degree, `nfp`, symmetry, knots, ring theta counts or `map_theta`. `read_checkpoint` checks the same.
  Rebuild the sequence from `checkpoint_attrs(path)` first.
- **Krylov extremes are not modes.** Ritz values from a MINRES or CG run do not give the low end of a spectrum. Use
  LOBPCG for that.
- **Quote no seeded island width without an amplitude sweep.** A seeded width depends on the seed amplitude.
- **Set `--budget.floor-tol` well below the expected floor** and let `--budget.steps` end the run. Otherwise a
  table shows the floor-tol instead of the bottom. Judge h-refinement by the floor.
- **Known effects are recorded** in `docs/research/`. Check there before "discovering" one.
