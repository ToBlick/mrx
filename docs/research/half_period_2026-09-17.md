# Half-period quadrature for stellarator-symmetric runs (2026-09-17)

> **Read for:** what `symmetry="stellarator"` now does to a run, why the
> DoFs stay full-size, where the parity enters, and what was measured.
> **Do not read for:** the map projector itself (`mrx/mappings.py`, from
> PR #23) or the Poincare API (`docs/source/relaxation.md`).

Branch `worktree-half-period` (off `worktree-poincare-dof-arg`).

## 1. What it is

A stellarator-symmetric map satisfies `F(r, -theta, -zeta) = S F(r, theta,
zeta)`, `S = diag(1, -1, -1)`, the rotation by pi about the X axis. Every
field of the relaxation then has a definite parity under it: `B`, `A`, `J`,
`H` odd (`B(Sx) = -S B(x)`, the usual `(B_R, B_phi, B_Z)(R, -phi, -Z) =
(-B_R, B_phi, B_Z)`), velocities, forces, pressures even. The relaxation
preserves the parity, so the symmetric half of the device determines the
rest and the quadrature can stop at `zeta = 1/2`.

Tobias's brief (2026-09-17): the island phase drifting off the symmetry
plane is not physical, so keep the fields symmetric; save compute, keep
the DoFs as they are.

## 2. How

* `QuadratureRule(..., half_zeta=True)`: the zeta rule covers the spans
  in `[0, 1/2]` with the weights doubled. `1/2` must be a knot (an even
  number of uniform zeta cells; `build_sequence` exploits the symmetry
  only on uniform angular knots). The sum-factorised kernels already
  accept a partial element range on an axis (the accumulate pads, the
  gather slices), so `mass.py` changes in two lines: the element counts
  come from the rule, and the diagonal is mirrored.
* `mrx/symmetry.py`: on a uniform periodic axis the reflection is the
  index permutation `j -> (p - 1 - j) mod n` (the derivative basis with
  its own `(n, p - 1)`); with the component signs `(1, -1, -1)` it is the
  reflection `R` of every raw DoF grid, and `Pi = (I + parity R) / 2` the
  projector. **The doubled half-period moments, projected, are the
  full-period moments** for a field of definite parity: the mirror image
  of a basis function `j` is `R j`, and `Pi` adds the two halves.
* Where the parity comes from: the mass and projection applies read it
  off their input (`sign(x . R x)`, one gather and a dot, fused with the
  projector in `symmetrize_like`); the loads of pointwise products take
  it explicitly (`parity=` on `cross_product_load_values` and friends;
  the DoF-vector variants multiply the factors' parities). Scalar
  integrals of even integrands (energy, helicity, norms) are right as
  they stand.
* Preconditioners: the 1-D zeta assemblies and the per-DoF support
  averages are mirrored (`mirror_zeta_1d`, `mirror_component`); the
  dense-core probes split the unit vector into its two parities
  (`_parity_split`, the exact host-side `project_parity`); and every atom
  returns the parity of its input (`free_projector`, `Pi P Pi`).
* The initial field is projected onto the odd parity (`initial_field`,
  `info["parity_discarded"]`: 3e-8 on li383, i.e. the histopolated `B =
  dA'` was odd already).

## 3. What went wrong on the way (kept because each one looks plausible)

1. `lax.cond` in the apply to handle mixed-parity inputs (probes): the
   branch closures capture the input, so every eager call recompiled;
   the (8,12,12) fixture took 1322 s to build. Removed: the apply is for
   pure inputs, the probes split explicitly.
2. The probe split through `_conforming_restriction` returns in the
   module `DTYPE` (float32) while the cores are probed on the float64
   view: cores accurate to 1e-7, inverted at a 1e-12 cut-off. Replaced by
   the exact host projection through the sparse free-space reflection
   `R_free = (E E^T)^-1 E R E^T`.
3. **The metric-lumping atoms are not reflection-equivariant** (the
   polar rows). Every operator -- `M_k`, `P_kl`, `D_k`, `S_k`, `L_k` --
   matched the full-period twin to 1e-14 on pure vectors, the `M_0` and
   `M_3` solves converged identically, and the `M_1`, `M_2`, `L_1`, `L_2`
   solves ran to 10000 iterations: the atom's output leaks the other
   parity, the half-period apply projects it away, CG loses conjugacy.
   Fixed by `Pi P Pi` on every atom (measured in section 4).
4. A sparse shortcut for the free-space reflection (`(E E^T)^-1 E R E^T`
   assembled with per-component blocks) was wrong at 1e-3; the exact
   `_conforming_restriction` is kept (returned in the input's dtype), and
   the device projector of the atoms is checked against it (1e-16).
5. Primal against dual purity. A free vector `x` is pure when its lift
   `E^T x` is; a DUAL vector (a load, `M x`) is pure when `E R E^T (E
   E^T)^-1 r = +-r`. On the polar rows, where `E E^T` is not the identity,
   the two differ: a primal-pure random vector used as a right-hand side is
   dual-impure, the projected preconditioner annihilates that part, and CG
   stalls. That was my test, not the code -- every load and apply produces
   dual-pure vectors -- and it cost an evening: the `M_k^-1` lines of the
   operator diagnostic have to use `M_full x` as the right-hand side.
6. The round-off of the other parity. A residual assembled by cancellation
   (`b - S x - M D w` in the pair loop, `r - alpha A p` in a CG) carries an
   impure part at round-off; the half-period operators cannot reduce it,
   and once the pure part has converged the impure one dominates -- the
   inner CG of the k=1 Hodge split then ran to its 10000 cap and returned
   garbage (the operator diagnostic: 10152 iterations against 357, the
   solution 40% impure on the third pass). Two pieces fix it: the atoms
   are `Pi P Pi^T` (the dual projector on the input, the primal one on the
   output: symmetric, and the CG's `P`-norm cannot see the impure part),
   and every solve projects its right-hand side, residuals and iterates
   onto the parity of its right-hand side (`parity=` on
   `solve_singular_cg`, `parity_upper/lower` on the saddle MINRES, the
   pair loop's `project_dual`, the Newton `refine`), composed with the
   harmonic-form deflation it already had. After that the k=1 Dirichlet
   Laplacian solve matches the full one to 6e-14 in 314 iterations
   against 357.

## 4. Measurements

Operator diagnostic (`half_period_2026-09-17/diag_ops.py`, li383 (8,12,12)
p=2 float64, the half-period sequence against a full-period twin on the
same map, vectors of definite parity): `M_k`, `P_kl`, `D_k`, `D_k^T`,
`S_k`, `L_k` agree to 1e-14 in every degree and both boundary classes;
the mass and Laplacian atoms differ by 1% and 2-6% (`Pi P Pi^T` against
`P`); the mass solves agree to 1e-13 with identical iteration counts
(k=0..3: 14, 57/62, 54/61, 7); the k=0 Laplacian solves to 1e-9 with
44/73 against 44/72 iterations; the k=1 Dirichlet Laplacian solve to
6e-14 in 314 against 357; the harmonic forms to 2e-15 (k=1 free), 7e-15
(k=2 Dirichlet), 7e-15 (k=3); the initial field's even part 5e-16.

The suite on the li383 (8,12,12) fixture as a half-period sequence: 60/60
in float64 (13.5 min) and 60/60 in refined float32 (15.8 min), the normal
suite speed. Two tests changed: the mass-apply oracle and the projection
transposition draw their random vectors of one parity, the Hessian test
projects its random velocity even (a half-period sequence is defined on
fields of definite parity; a random vector is not one). `test_symmetry.py`
adds the 1-D reflection identities, the projector algebra, the parity of
the equilibrium field and harmonic forms, and the half-against-full mass
apply at 1e3 eps.

The gate, li383 (16,32,32) p=2 float64 (runs under the worktree's
`outputs/half_period/`, compared with `half_period_2026-09-17/compare_runs.py`):

* L-BFGS, 100 steps, one reconnection at step 50: the reconnection
  identical to every printed digit (eps 5.936e-05, |F| 1.976e-03 ->
  1.333e-03, H -0.95%, 608 against 620 solver iterations); energy and
  helicity to 1e-12 over the run; per-step residual to 2.4e-4 and the
  final B to 5e-5 at step 100 -- round-off-seeded divergence, as between
  any two builds ([[relaxation-trajectory-roundoff-divergence]]). Wall
  1.40 s/step against 1.30: 6% SLOWER.
* Newton, 20 steps: same floor (7.8e-9 against 1.07e-8), but the paths
  differ at the 1e-3 level from the first step: MINRES exhausts its 300
  iterations on 19 of 20 steps, so the direction is a truncated solve, and
  the projected preconditioner (2-6% different) truncates it differently.
  Wall: 28.6 s/step against 21.7 with the projector through the extraction
  operators (a Newton step makes ~45k preconditioner applies, each then
  four COO applies in float64 -- the difference), 21.77 against 21.70 with
  the projector as one gather (a signed permutation of the bulk rows plus
  the dense polar core block, equal to the exact projector to 1e-16).

So at (16,32,32) the half-period step costs the same as the full one: the
step is launch-bound there, and halving the kernel work buys nothing. The
production size, li383 (32,64,64) p=2 refined float32, 20 L-BFGS steps in
one chunk (compile included): full 198.2 s (9.91 s/step), half 166.8 s
(8.34 s/step) on the slow projector, 144.3 s (7.21 s/step) on the gather
one: 27% faster, energies equal to 2e-7 and helicities to 3e-7 after 20
float32 steps. The steady-state per-step gain is larger (the compile is a
fixed part of both), and it grows with the mesh; a (48,96,96) pair would
say where it saturates. Not the 2x of the kernel count: the Krylov
overhead, the atoms, the incidence applies and the polar core are
unchanged, and the scheme adds O(n) per apply.
