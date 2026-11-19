# Testing strategy

MRX has one lean test tier in two precisions. Bare `pytest` runs the suite
in float64, and `MRX_DTYPE=float32 pytest` in float32 with 64-bit mode off
([Precision](precision.md)). `bash slurm/suite.sh` submits both as GPU jobs,
and the GitHub workflow runs both on a CPU. The suite reads only files
tracked in the repository.

## Four statements

The suite tests four mathematical statements about the discretisation, which between them run most of
the code, plus the prescribed pressure, current and transform, the field-line tracer, the shape derivative, the readers, the profiles of a
VMEC input namelist, the map from a boundary alone, the mesh motion and the flux label:

| file | statement |
|---|---|
| `test_derham.py` | the spaces form a de Rham complex: `G_{k+1} G_k = 0` on every space and parity view, the mass matrices are SPD with `D_k = M_{k+1} G_k` and adjoint projection pairs, the extracted derivatives are the raw ones, interpolation commutes with the derivative, axis included, and the harmonic forms span the cohomology `(1, 1, 0, 0)` of the solid torus, and the distance of `grad zeta` from the harmonic 1-form vanishes on an axisymmetric torus |
| `test_solvers.py` | every solve inverts its operator: a known solution, mapped to its right-hand side by the operator's apply, comes back from `M[k]`, `L[k]` (every `part`), `S[k]`, `shifted(k, eps)`, `shifted_stiffness(k, eps)` and the Leray projection to a multiple of `seq.tol`, on the production spaces. Conjugate gradients and MINRES stop on the given norm of the true residual |
| `test_poisson.py` | the Hodge Laplacians k = 0..3 recover manufactured solutions within measured bands, with the same error on the two representations of the domain (below), preconditioners kept across a change of map still precondition the new operators, and the slab model conditions `M_1` |
| `test_newton.py` | the second variation reproduces the energy along the ideal flow, `E'(0) = -(u, J x B)` and `E''(0) = (u, H u)` with the bilinear form by polarisation, the Chebyshev mass inverse keeps the Hessian symmetric, and a short run takes positive steps, lowers the energy and the force, keeps `div B` at round-off and the helicity up to grid-scale effects, and checkpoints, and a few gradient-descent steps descend too |
| `test_shells.py` | a shell velocity of the prescribed pressure has the projection of its shell function less the mean as its divergence to the tolerance of the Poisson solve and is orthogonal to every curl to round-off, the energy changes along its flow at the rate `-(xi, J x B)`, and the pressure part of its row is `int p* div xi dV` |
| `test_twist.py` | the slips of the current target change the field by a divergence-free form without a `zeta` component, their map's transpose is its transpose to round-off, the energy along a slip is quadratic with the assembled derivative and block to the round-off of the energy, the moments are the spline means of the weak current to the discretisation and the block is positive definite. A `relax.py --current.target --pressure.target` run with one `NCURR = 1` namelist and a `relax.py --iota.target --pressure.target --seed` run with one `NCURR = 0` namelist meet a current, or a transform, and a pressure a few percent off the file's in twenty Newton steps, lowering `W` at every step, and the seed waits for the transform and sits at the target's resonances (float64 only) |
| `test_mesh_independence.py` | the profile control does not depend on the mesh: from the vacuum, with the file's current as the target, the file's mesh and the same domain with its axis moved 4 cm meet the target and end at the same field and the same transform on their own labels to the two meshes' discretisation difference (float64 only) |
| `test_poincare.py` | the field-line tracer measures the file's rotational transform on the equilibrium's own field, whose lines stay on their logical surfaces, the padding lines stay on their surfaces both ways, the crossings of a closed-form field are its exact lines forwards and backwards, a lost line is cut at the boundary, and on a map in the cylindrical frame the sections `phi = const` are the sections `zeta = const` |
| `test_shape_ad.py` | the vacuum field of the shape derivative is harmonic, its derivative in the map coefficients matches central differences, and on a frame that changes along zeta with points out of its planes the hand-derived metric and `det DPhi` are those of the map's Jacobian |
| `test_readers.py` | the GVEC and DESC readers reproduce synthetic states, the VMEC reader reads li383, DESC's conversion of it agrees with the wout, non-symmetric files read as shifted symmetric ones, and the angles are right-handed after reading |
| `test_indata.py` | every profile type of a VMEC input namelist is its closed form, or its density's integral for a current, the current is `CURTOR` at the edge, the pressure is `mu0 p` in T^2, and a namelist that tabulates the geometry file's pressure gives the file's target |
| `test_fill.py` | the map filled from a namelist's boundary is the axis at `rho = 0` and the boundary at `rho = 1` mode by mode, the guessed axis leaves the fill unfolded and moving it by 1 cm lowers the guess's objective, the spline map meets the boundary within its projection error with a positive Jacobian at every quadrature point, and the vacuum start has the namelist's toroidal flux through every plane, no divergence and no current |
| `test_mesh_motion.py` | a move hardly moves the li383 file mesh, whose surfaces are the flux surfaces. On that mesh with its axis displaced by 4 cm one move puts the mesh axis back on the label's maximum, keeps `div B` and the toroidal flux to round-off and the physical field within the transport's error, the undo transport there and back returns the field to its time error, a relaxation with the motion ends where the relaxation on the file's mesh ends, and the harmonic section brings `zeta` onto the harmonic angle and a move keeps it there |
| `test_flux_label.py` | the flux label of the equilibrium's field is constant on its logical surfaces up to the anisotropy, falls from the axis to the wall and encloses the flux `r^2`, and the label at the shells' tolerance, solved cold or warm-started from the label of another field, is the converged one to twice that tolerance |

The fixtures of `test/conftest.py` build the li383 equilibrium
(`data/wout_li383_low_res_reference.nc`) at `(16, 32, 12)` per field period,
`p = 2` on the graded polar space of production, in two representations with the same mesh: `half`
(the production sequence, stellarator symmetry, half a period integrated)
and `period` (one field period without the reflection). `seq` is `half`,
`b0` the equilibrium's own field on it. A manufactured solution with the
period and parity of li383 must give the same error on both, which tests the
half-period quadrature and the parity views end to end.

The suite is XLA-compile-bound: a test costs the distinct solves it
compiles, not the mesh. Compiled programs are cached on disk
(`MRX_XLA_CACHE`, default `$MRX_DATA/cache/xla`, or `outputs/xla_cache` in
the checkout without `MRX_DATA`).

## What a test asserts

Mathematical statements, phrased so that they fail when the mathematics is
wrong and not when an implementation detail moves:

- exact identities to a multiple of `mrx.eps()`.
- solver-based quantities to a multiple of `seq.tol`.
- measured bands, a fixed factor above a measured error, stated next to
  the value. A wrong metric factor or a broken preconditioner moves these
  by a factor, run-to-run noise by a few percent.

A new test states which mathematical claim it checks at which tolerance class, on the
fixtures above.
Dense references and anything that probes every degree of freedom do not
belong in the suite.
