# FAQ

## Can MRX run in free-boundary mode?

Nothing in MRX identifies the computational boundary with the plasma
boundary. The domain is the mapped logical cube, its outer surface `r = 1`
is fixed, the field lives in the Dirichlet 2-form space with `B . n = 0`
there, and the relaxation velocity is tangent to it. That surface can be
the last closed surface of a GVEC or VMEC file, as in the tutorials, or lie
far out in vacuum. The relaxation then decides where the plasma ends: the
pressure is not prescribed on any surface, and it goes to zero where the
field carries no current. The plasma edge is an outcome of the run, and so
are islands and chaotic regions, which come from the initial field, a seed or
the resistive drive, since the ideal relaxation keeps the topology up to
grid-scale effects.

What stays fixed is the computational boundary. MRX has no coil model (yet), 
so the field in a vacuum region is what the initial condition put there and
the relaxation makes of it under `B . n = 0` on the outer surface. Choose
that surface far enough out that it does not matter.

## What about pressure?

Pressure is not an input. MRX minimises the magnetic energy under the
constraint that the field moves with a divergence-free, wall-tangent
velocity, and the pressure is the Lagrange multiplier of that constraint.
At a fixed point `J x B = grad p`. There is no prescribed profile `p(Psi)`,
because there are no flux surfaces for `Psi` to label. The initial field 
sets how much pressure the equilibrium carries, and where the relaxed field 
has islands or chaotic regions the pressure flattens across them. 

Beta is `beta = int p dV / int |B|^2/2 dV` in code units, reported as 
`beta_vol`. [Relaxation](concepts/relaxation.md) explains the two pressures 
a run records.

## How expensive is MRX to run? Do I need a GPU?

It depends on the resolution. On a single GPU, setup (the compile, building
preconditioners, computing harmonic forms) takes minutes, and a Newton run
reaches the force residual floor in tens to hundreds of steps.

MRX runs on a CPU as well, which is fine for the tutorials (all six take about
40 minutes on 8 cores, see [Tutorials](tutorials.md)) and the tests.

We do not yet have benchmark numbers on strong consumer laptops. Feel free to
share them.

## Which precision should I use?

The default. MRX fixes its precision when it is imported, from `MRX_DTYPE`
and `MRX_RESIDUAL_DTYPE` ([Precision](concepts/precision.md)), and by default
it runs in mixed precision: the fields and the Krylov iterations in float32,
every residual in float64. This is what the paper's runs and the tutorials
use. Plain float32 (`MRX_RESIDUAL_DTYPE=float32`) is faster but can stall on a
strongly shaped geometry. On the QA map of the tutorials the solve behind the
vacuum field stops near a relative residual of `5e-5` however many
iterations it gets, while mixed precision converges in a few hundred.
float64 throughout (`MRX_DTYPE=float64`) is for derivatives by finite
differences and for the shape optimization, where the gradient needs the
extra digits.

## Which equilibrium files can MRX read?

VMEC `wout_*.nc` files, GVEC state files (`*.dat`) and DESC output files
(`*.h5`). The extension selects the reader. Each gives the map of the domain
and the equilibrium's own field as the initial condition. Files without
stellarator symmetry are read as well, and `--geometry.symmetry` then has to
be `field-period` or `none` ([Equilibrium input](concepts/equilibrium_input.md)).

## How do I read the solver output?

A solve reports a signed iteration count: positive when it converged,
negative when it stopped without reaching its tolerance. Solves with an outer
correction loop (the Laplacians and the pressure) add up the iterations of all
passes, so the count can exceed `--geometry.solve-maxiter`. The
`[nullspace]` lines give the Rayleigh quotient `v^T L v / v^T M v` of each
harmonic form. A harmonic form reads round-off, `1e-9` or smaller. A value
near `1e-3` means the solve that produced it did not converge.
