# FAQ

## Can MRX run in free-boundary mode?

Yes. Nothing in MRX identifies the computational boundary with the plasma
boundary. The computational domain is the mapped logical cube, its outer
surface `r = 1` is fixed, the field lives in the Dirichlet 2-form space
with `B · n = 0` there, and the relaxation velocity is tangent to it. That
surface can be the last closed surface of a GVEC or VMEC file, which is
what the fixed-boundary tutorials do, but it can equally lie far out in
vacuum. The relaxation then decides where the plasma ends: the pressure is
the Lagrange multiplier of the constrained energy minimisation (next
question), it is not prescribed on any surface, and it goes to zero where
the field carries no current. Islands, chaotic regions and the plasma edge
are all outcomes of the same run.

What stays fixed is the computational boundary itself. A vacuum region is
part of the same domain and the same spline spaces, and MRX has no coil
model, so the field in the vacuum region is whatever the initial condition
put there and the relaxation makes of it under `B · n = 0` on the outer
surface. Choose that surface far enough out that it does not matter.

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

A run records two pressures, the strong multiplier `p` and the weak
pressure `p_w` that sees the wall force; they agree in the interior of a
converged equilibrium, and the diagnostics that quantify their difference
are in [Relaxation](concepts/relaxation.md). Beta is
`β = ∫ p dV / ∫ B²/2 dV` in code units, reported as `beta_vol` from `p_w`.

Resistivity lowers the pressure. A resistive step reconnects the field,
removes current, and the pressure follows: along a reconnection series the
current ratio `‖J‖/‖B‖` and `beta_vol` drop together with the helicity.

## How expensive is MRX to run?

A single GPU. Everything is matrix-free and jit-compiled with JAX; a
relaxation step is a handful of CG and MINRES solves on the tensor-product
B-spline spaces, under a second per step on the li383 `(16, 32, 32)`
`p = 2` mesh on one H100 after a setup of a few minutes (the compile, the
preconditioners, the harmonic forms). A run to the ideal floor is a few
thousand steps; a `(32, 64, 64)` `p = 2` run fits comfortably on an 80 GB
card. MRX runs on a CPU as well, which is fine for the tutorials and the
tests, but not for a production relaxation.
