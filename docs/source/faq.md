# FAQ

## Can MRX run in free-boundary mode?

No. MRX is a fixed-boundary code. The computational domain is the mapped
logical cube, and its outer surface `r = 1` is the boundary of the input
equilibrium: the last closed surface of a GVEC or VMEC file, or the surface
of an analytic map. The magnetic field lives in the Dirichlet 2-form space,
so `B · n = 0` on that surface, and the relaxation velocity is tangent to
it, so the boundary never moves during a run. What MRX does not assume is
anything *inside* the boundary: no nested surfaces, no flux label, no
profile in a flux coordinate.

A free-boundary calculation needs a vacuum region between the plasma and a
coil field, and a plasma-vacuum interface that moves until the total
pressure balances across it. MRX has neither. The closest thing you can do
is choose a larger fixed boundary that encloses the plasma and a region of
(near) vacuum; the relaxation then decides where the current and the
pressure end up inside it, but the outer surface stays where you put it.

## What about pressure?

Pressure is not an input. MRX minimises the magnetic energy under the
constraint that the field moves with a divergence-free, wall-tangent
velocity, and the pressure is the Lagrange multiplier of that constraint:
at a fixed point `J × B = ∇p`. There is no prescribed `p(ψ)`, because there
is no `ψ`. The initial field sets how much pressure the equilibrium will
carry (for the li383 Clebsch initial condition the relaxed volume-averaged
beta is a few per cent), and the relaxation finds the pressure that
balances the force it produces. Where the relaxed field has islands or
chaotic regions the pressure flattens across them, which is the physics the
code is built to show.

The driver records two pressures. The strong pressure `p` is the multiplier
itself, a 3-form with `dp/dn = 0` on the wall by construction; it cannot
see a force on the wall. The weak pressure `p_w` is the gradient part of
`J × B` in the natural 1-form space with `p_w = 0` on the wall; it does see
the wall force, and its normal derivative there is the force the strong
pressure misses. The two agree in the interior of a well-converged
equilibrium and differ where the equilibrium is not one; the diagnostics
`gradp_cmp`, `p_cmp`, `weak_resid`, `dpdn_wall` and `JxBn_wall` in the
[relaxation concepts](concepts/relaxation.md) page quantify that. Beta is
`β = ∫ p dV / ∫ B²/2 dV` in code units, reported as `beta_vol` from `p_w`.

Resistivity lowers the pressure. A resistive step reconnects the field,
removes current, and the pressure follows: along a reconnection series the
current ratio `‖J‖/‖B‖` and `beta_vol` drop together with the helicity.

## How expensive is MRX to run?

A single GPU. Everything is matrix-free and jit-compiled with JAX; the cost
of a relaxation step is a handful of conjugate-gradient and MINRES solves
(the force, the Leray projection, the velocity smoothing, the line search)
on the tensor-product B-spline spaces, so it scales with the number of
degrees of freedom times the solver iteration counts. Measured on one
H100 for the li383 stellarator (`nfp = 3`), relaxation in float32:

| mesh `(n_r, n_θ, n_ζ)`, degree | setup (operators, preconditioners, harmonic forms) | one relaxation step |
|---|---|---|
| (16, 32, 32), p = 2 | 150 s | 0.57 s |
| (32, 64, 64), p = 2 | 235 s | 4.6 s |

The setup happens once per geometry and includes the compile. A run to the
ideal floor on the (16, 32, 32) mesh is a few thousand steps, under an
hour; a reconnection series of 18000 steps is about three hours. A
Poincaré section (160 field lines, 400 crossings each, float32) takes about
25 s per field on that mesh plus the same setup; the lean test suite runs
in about four minutes on a GPU. Memory is not the limit at these meshes;
the largest arrays are the metric at the quadrature points, which are kept
resident, and a (32, 64, 64) p = 2 run fits comfortably on an 80 GB card.
MRX runs on a CPU as well, which is fine for the Poisson tutorials and the
tests, but not for a production relaxation.
