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

## 4. Measurements

(to be filled: the operator diagnostic after the atom fix, the suite in
both precisions, the li383 (16,32,32) float64 identity run half vs full
-- Newton 20 steps, L-BFGS 100 steps with a reconnection -- and the
per-step wall time.)
