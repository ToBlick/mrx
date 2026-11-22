# Testing strategy

MRX has one lean test tier in two precisions. Bare `pytest` runs the suite
in float64, and `MRX_DTYPE=float32 pytest` in float32 with 64-bit mode off
([Precision](precision.md)). `bash slurm/suite.sh` submits both as GPU jobs,
and the GitHub workflow runs both on a CPU. The suite reads only files
tracked in the repository.

## Four statements

The suite tests four mathematical statements about the discretisation, which between them run most of
the code, plus the prescribed pressure and current in one run of `relax.py`, the Bayesian fit of pressure data, the field-line tracer, the shape derivative, the readers, the profiles of a
VMEC input namelist, the map from a boundary alone, the mesh motion and the flux label:

| file | statement |
|---|---|
| `test_derham.py` | the spaces form a de Rham complex: `G_{k+1} G_k = 0` on every space and parity view, the mass matrices are SPD with `D_k = M_{k+1} G_k` and adjoint projection pairs, the extracted derivatives are the raw ones, interpolation commutes with the derivative, axis included, and the harmonic forms span the cohomology `(1, 1, 0, 0)` of the solid torus, and the distance of `grad zeta` from the harmonic 1-form vanishes on an axisymmetric torus |
| `test_solvers.py` | every solve inverts its operator: a known solution, mapped to its right-hand side by the operator's apply, comes back from `M[k]`, `L[k]` (every `part`), `S[k]`, `shifted(k, eps)`, `shifted_stiffness(k, eps)` and the Leray projection to a multiple of `seq.tol`, on the production spaces. Conjugate gradients and MINRES stop on the given norm of the true residual |
| `test_poisson.py` | the Hodge Laplacians k = 0..3 recover manufactured solutions within measured bands, with the same error on the two representations of the domain (below), preconditioners kept across a change of map still precondition the new operators, and the slab model conditions `M_1` |
| `test_newton.py` | the second variation reproduces the energy along the ideal flow, `E'(0) = -(u, J x B)` and `E''(0) = (u, H u)` with the bilinear form by polarisation, the Chebyshev mass inverse keeps the Hessian symmetric, and a short run takes positive steps, lowers the energy and the force, keeps `div B` at round-off and the helicity up to grid-scale effects, and checkpoints, and a few gradient-descent steps descend too |
| `test_shells.py` | a shell velocity of the prescribed pressure has the projection of its shell function less the mean as its divergence to the tolerance of the Poisson solve and is orthogonal to every curl to round-off, the energy changes along its flow at the rate `-(xi, J x B)`, and the pressure part of its row is `int p* div xi dV` |
| `test_pressure_fit.py` | the Bayesian fit of `scripts/pressure_fit.py`, loaded from its path: with the evidence's weight the posterior covers the truth as a Gaussian does, it is the least-squares fit when the noise vanishes and the prior where no data lie, and a data file at points reaches it through the means of the splines over kernels narrow across the field's flux surfaces and wide along them |
| `test_twist.py` | the slips of the current target change the field by a divergence-free form without a `zeta` component, their map's transpose is its transpose to round-off, the energy along a slip is quadratic with the assembled derivative and block to the round-off of the energy, the moments are the spline means of the weak current to the discretisation and the block is positive definite |
| `test_poincare.py` | the field-line tracer measures the file's rotational transform on the equilibrium's own field, whose lines stay on their logical surfaces, the padding lines stay on their surfaces both ways, the crossings of a closed-form field are its exact lines forwards and backwards, a lost line is cut at the boundary, and on a map in the cylindrical frame the sections `phi = const` are the sections `zeta = const` |
| `test_shape_ad.py` | the vacuum field of the shape derivative is harmonic, its derivative in the map coefficients matches central differences, and on a frame that changes along zeta with points out of its planes the hand-derived metric and `det DPhi` are those of the map's Jacobian |
| `test_readers.py` | the GVEC and DESC readers reproduce synthetic states, the VMEC reader reads li383, DESC's conversion of it agrees with the wout, non-symmetric files read as shifted symmetric ones, and the angles are right-handed after reading |
| `test_indata.py` | every profile type of a VMEC input namelist is its closed form, or its density's integral for a current, the current is `CURTOR` at the edge, the pressure is `mu0 p` in T^2, and a namelist that tabulates the geometry file's pressure gives the file's target |
| `test_fill.py` | the map filled from a namelist's boundary is the axis at `rho = 0` and the boundary at `rho = 1` mode by mode, and the guessed axis leaves the fill unfolded and moving it by 1 cm lowers the guess's objective |
| `test_mesh_motion.py` | the axis finder puts the magnetic axis of the li383 file's field on its pole and a move hardly moves that mesh. On that mesh with its pole displaced by 4 cm the field kept by its coefficients has the same logical axis, the field carried there and back keeps `div B` and the toroidal flux at round-off and returns to the time error of the transfer, the harmonic section brings `zeta` onto the harmonic angle and a move of the pole leaves it within a measured band, and the cut by the planes normal to the axis puts the sections normal to the axis without folding, keeps every point on its surface `r = const` to the error of the splines and carries the file's field, which the points' slide alone would drag |
| `test_relax_run.py` | one `scripts/relax.py` run with the pole kept on the magnetic axis and the pressure and current targets from a namelist alone, and one with the same targets on the wout: the fill meets the boundary with a positive Jacobian, the vacuum start has the namelist's flux and no divergence or current, both runs descend and end at the targets, the start's mesh has its pole on the vacuum's axis, the pole moves exactly at the boundaries below the bar with the axis off it by more than the tolerance and without folding the mesh, and the two runs end at the same field and transform (the runs in float64 only) |
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
