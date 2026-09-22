# Plan: stellarator symmetry as a reduction of the DoF space (2026-09-20)

Branch `parity-extraction` off `static-dynamic-refactor` 3c120db. Tobias's
question: the half-period sequence halves the quadrature but not the DoF
count; make it halve the DoFs too, because that is cleaner.

## 1. Where we are

A stellarator-symmetric map has `F(r, -theta, -zeta) = S F(r, theta, zeta)`,
`S = diag(1, -1, -1)` the rotation by pi about X. Every field of the
relaxation has a definite parity under it, a property of the PHYSICAL field,
not of the form degree: `B`, `A`, `J`, `E`, `H` are odd (`B(Sx) = -S B(x)`:
the flux is even, the straight-field-line angle odd), `u`, `F = J x B`, `p`,
`grad p` and every smoothing solve on them are even. `d` preserves parity,
products multiply it, `M^-1` and the Laplacians preserve it. On the raw DoF
grid the rotation is a signed permutation `R` of the two angular axes
(`reflection_plan`: `perm_theta`, `perm_zeta`, `COMPONENT_SIGNS = (1, -1,
-1)` for the vectorial forms), and a field of parity `s` has `c = s R c`.

Today (`mrx.symmetry`, 2026-09-17) the full coefficient vector is kept and
the symmetry is a CONSTRAINT enforced around every kernel: the quadrature
covers `zeta in [0, 1/2]` with doubled weights, every load is `symmetrize`d
on the raw grid, every mass/projection apply detects the parity of its input
(`parity_of`: a reflection and a global reduction) and projects its output
(`_half_period_apply`), every atom is wrapped `Pi P Pi^T` (`FreeProjector`,
whose `post`/`dual` cast to float64 and gather twice), every Krylov loop
composes parity projectors with its deflation (`_compose_parity`,
`_parity`, `_parity_pair`), and the dense-core probes split unit vectors by
parity (`_parity_split`). The same k-form space holds both parities (`B`
odd and `u` even in the Dirichlet 2-forms), which is why the detection is
per vector. Measured: the half-period step is 1.4x (n=32) to 1.7x (n=48)
the field-period one, not 2x; the efficiency review of 2026-09-20 counted
~50-60 small kernels of parity handling per PCG iteration.

## 2. Where we go

The symmetry becomes a REDUCTION of the DoF space, composed into the
extraction, so that every kernel sees pure fields by construction and no
runtime symmetry logic remains.

**The reduced space.** For `(k, dirichlet, s)` the DoFs are the orbits of
the free-space reflection `R_free = (E E^T)^-1 E R E^T` (`E` the polar +
boundary extraction, raw -> free), which `FreeProjector.__init__` already
shows to be a signed permutation on the bulk rows and a small dense block on
the polar core rows: one DoF per bulk pair `{i, R(i)}`, the fixed points on
the fold planes `zeta = 0, 1/2` kept when their sign matches `s` and dropped
otherwise (like Dirichlet rows), and on the core an orthonormal basis of the
`s`-eigenspace of the core block. `X_s` (`n_free x n_red`, orthonormal
columns) is the expansion, `X_s^T` the reduction, and

    E_{k,d,s} = X_s^T E_{k,d}      (n_red x n_raw),

built ONCE on the host as COO triplets and applied exactly as today's `E`:
one gather + `segment_sum` (`MatrixFreeExtraction`), two entries `+-1/sqrt2`
per bulk row, a few per core row. `n_red = n_free / 2` up to the fold planes.

**The invariants, asserted at build time** (host, numpy, cheap):

    E_{k,d,s} R = s E_{k,d,s}        (every reduced DoF is a field of parity s)
    E_{k,d,s} E_{k,d,s}^T = I        (orthonormal rows)

The first is what makes every kernel exact WITHOUT symmetrize: the raw
half-period apply returns `2 * (half moments)`, and `Pi_raw (2 m_half) =
m_full` for a field of definite parity (the docstring of `mrx.symmetry`),
so `E_red (2 m_half) = E_red Pi_raw (2 m_half) = E_red m_full` -- the
reduction IS the combination of the mirror images. Loads, mass applies,
projections, derivatives: all `E_red (raw kernel) E_red^T`, as now with `E`.

**The parity lives in two places, both static.**

1. On the space: the extraction table is keyed `(k, dirichlet, parity)`.
   The sequence exposes the two parity classes as VIEWS, `seq.odd` and
   `seq.even`, the way `seq.residual` is the float64 view: the same
   geometry, quadrature and raw kernels (shared arrays), each with its own
   extraction table, operator bundle (atoms, nullspaces) and DoF counts.
   Call sites pick the view of the field they hold, `seq.odd.apply_mass_matrix(B, 2)`,
   `seq.even.apply_inverse_laplacian(rhs, 2)`; a product load is called on
   the view of its OUTPUT, `seq.odd.cross_product_load_values(u_jk, B_jk, 1, 2, 2)`.
   On a full-period sequence `seq.odd is seq.even is seq`. No apply gains
   an argument; the `parity=` kwargs of the loads go.
2. On the physics: one table in `mrx/relaxation.py` (and the IC / hessian
   code that builds fields) names the view of every variable -- `B, A, J,
   E, H` odd; `u, F, p, grad p` even -- and every step uses it. The runtime
   `parity_of` is gone; a vector on the wrong view is a shape error at `E`,
   not something to detect.

**The atoms.** They never saw the symmetry (the projector was composed
around them) and keep not seeing it: the metric-lumping atoms are built on
the reduced space directly. Bulk: the 1-D tensor factors act on the raw
grid; the reduced atom is `X^T P_polar X`, the same sandwich as today's
`Pi P Pi^T` but static and exact (one gather each side, no detection, no
float64 round trip). Core: probed in the REDUCED basis -- `E_red M E_red^T
e_j` for the reduced core DoFs, which are symmetric fields, so the
half-period raw apply is exact on them and `_parity_split` is not needed;
half the probes of today. The 1-D zeta assemblies of the lumped factors
still need the full-period 1-D matrix: hand `_axis_bases` a full-period
1-D zeta rule for the 1-D assemblies (the altitude review's suggestion) and
`mirror_zeta_1d` / `mirror_component` go.

**Everything downstream** (checkpoints, `poincare`, `islands`, plotting,
the readers) sees a shorter DoF vector and expands through `E_red^T` the
way it expands through the polar extraction today (`DiscreteFunction(dof,
basis, E)`). Fields without a definite parity cannot be represented on a
half-period sequence; they run on `field-period` or `none` (Tobias).

## 3. What goes

`mrx/symmetry.py`: `FreeProjector`, `free_projector`, `parity_of`,
`symmetrize`, `symmetrize_like`, `mirror_zeta_1d`, `mirror_component`,
`_extraction_gram_core` (folded into the builder); what stays is
`reflection_permutation`, `reflection_plan`, `is_uniform_periodic`,
`COMPONENT_SIGNS` and the new builder. `mrx/operators.py`:
`_half_period_apply`, `_parity`, `_parity_pair`, the `parity=` kwargs on
the composite solves. `mrx/solvers.py`: `_compose_parity` and the
`parity`/`parity_upper`/`parity_lower` arguments. `mrx/derham_sequence.py`:
`symmetrize`, `project_parity`, `parity`, `free_projector`, the
`half_period` constructor flag (the sequence learns `symmetry`, `nfp` and
the parity views; `build_sequence` passes them through -- the altitude
review's item 4). `mrx/metric_lumping_laplacian.py`: `_parity_split`,
`parity_projector`, the `split` argument of `_probe_rows`. `mrx/mass.py`:
the inline mirror of the diagonal. `mrx/hessian.py`, `mrx/relaxation.py`,
`mrx/initial_conditions.py`, `mrx/nullspace.py`, `mrx/projectors.py`: the
`parity=` arguments (the view carries it).

## 4. Phases (each leaves the suite green; the suite runs as three GPU jobs)

1. **The builder.** `mrx.symmetry.parity_extraction(seq, k, dirichlet, parity)`
   -> `MatrixFreeExtraction`, with the two invariants asserted; a unit test
   of the invariants and of `n_red` on the session sequence. Nothing uses
   it yet.
2. **The views.** `DeRhamSequence(..., symmetry, nfp)`; `seq.odd`, `seq.even`
   with their extraction tables; loads and the mass / projection /
   derivative applies through `E_red`, `symmetrize` and
   `_half_period_apply` removed; `test_symmetry` rewritten to check the
   half-period view against a full-period sequence through `E_red^T`.
3. **Solves and atoms.** Nullspaces per view (the harmonic forms are odd;
   the even view's k=2 Dirichlet space has none), the metric-lumping atoms
   on the reduced space, the Krylov loops without parity projectors; the
   manufactured vacuum solves (`test_vacuum`) are the acceptance test.
4. **The relaxation.** The parity table; `TimeStepper`, `compute_force`,
   `second_variation`, `newton_direction`, the initial conditions and the
   checkpoint layout on the views; `poincare`/`islands`/plotting expand
   through the view's `E`. Old half-period checkpoints (full-length `B_n`)
   are converted by `X^T`, once, in `read_checkpoint`, keyed on the stored
   length -- or dropped, Tobias's call.
5. **Delete** what section 3 lists; docs (`docs/source/concepts`, the
   `mrx.symmetry` docstring) rewritten to the reduction.
6. **Measure**: the half-period step against the field-period one at n=32
   and 48 (the 1.4x / 1.7x of 2026-09-17 are the baseline), the DoF counts,
   compile time.

## 5. Open

- Views versus a `parity` argument on every apply: the plan takes views
  (no API sprawl, the parity of a variable is written once); to confirm on
  the branch.
- Whether the core eigenvectors are those of `R_free`'s core block or its
  transpose is settled by the invariant `E_red R = s E_red` at build time,
  not by derivation.
- Old half-period checkpoints: convert or drop.

## 6. Log

- Phase 1 (02f9f79, de90fca): the builder, verified on (6,8,8) p=2: every
  `(k, dirichlet)` splits `n_free` exactly, each parity about half. The
  invariant `E_red E_red^T = I` of section 2 was wrong -- neither `E` nor
  `E_red` is row-orthonormal on the polar core; what holds is `X^T X = I`.
- Phase 2 (d872c49): the views; mass, projections, grad/curl/div, evaluation
  and loads agree with the projected applies to 2e-7 on a torus.
- The core rows are DETERMINED, not probed (89227fe, Tobias): the extraction
  records the rows its surgery writes, the parity basis its core columns,
  `seq.core_rows(k, dirichlet)` serves both; the `bincount > 1`
  discriminator would have called every orbit row of `E_red` core.
- Phase 3a (3d4b2d7 + fixes): `ReducedAtom = X^T P X` of the base atoms (one
  build for base and views), per-view Betti numbers (odd `(0, b1, 0, 0)`,
  even `(b0, 0, 0, 0)`). Two bugs found by the CPU check: the view's float64
  twin must be a cached attribute so it shares the bundle; and the reduced
  basis is a CHOICE (the SVD basis of the core eigenspace), so working view
  and twin must build it from the same float64 extraction -- built
  independently they disagreed and every solve with a polar core was garbage
  while k=3 (no core) was exact. Result on the torus: the odd view's mass,
  Laplacian (k=0,1,2) and Leray solves agree with the base to 1e-7..2e-6
  at the same iteration counts (61/60, 40/40, 32/32); the harmonic 2-form
  to 4e-7; `n(2, dbc)` 720 -> 354.
- Open before phase 4: the even view was not re-checked after the basis fix
  (same code path); the Krylov loops still carry the `parity` arguments
  (None on views) until phase 5.
- 2026-09-21: `get_xi(nt, p)` puts the surgery weights at the splines'
  centres (Tobias, option 2): the free-space reflection is then a signed
  permutation on the polar core too and the parity basis is closed form
  (orbit pairs everywhere, no eigen-decomposition). Phase 4 written and
  smoke-tested on li383 (6,8,8) on the CPU: Newton, gradient descent and the
  helicity correction run on the views (`n(2, dbc)` 720 -> 354 for B, 366 for
  u); midpoint waits for the GPU. Suite on 1a6434f (before phase 4): 52/54,
  cold 610/514/525 s vs warm 378/345/324 s (the persistent cache, ~38%).
  Test lessons: the gradient of an even scalar is an EVEN vector; the natural
  k=3 Laplacian is the Dirichlet Laplacian on the density (its weak gradient
  is the adjoint of the free divergence), so its manufactured density must
  vanish on the wall; the equilibrium field's divergence is round-off on
  both sides of a comparison.
- 2026-09-21 evening: the suite on the phase-4 tip (cc2cd3e) was 53/54 in all
  three configurations (cold 507/439/439 s): the one failure was the k=0
  manufactured test's constant -- on the even view the constant function has
  reduced coefficients `X^T 1` (sqrt 2 on every orbit pair), not the all-ones
  vector (7355af0; error 37% -> 2.8%). Lesson for every consumer of a view: a
  "known" coefficient vector of the base must be reduced through `X^T`, never
  rewritten on the view. Suite resubmitted (18713216-8).
