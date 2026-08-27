# Concepts

This page names the objects you meet in every MRX script and points to the
page that explains each one. The pages under *Concepts* in the sidebar are
the documentation of the code as it is: every identifier named there exists in `mrx/` or `scripts/`.

## The discrete complex

MRX discretises

```
V0 --grad--> V1 --curl--> V2 --div--> V3
```

with tensor-product B-splines on the logical cube `[0, 1]^3` in coordinates
`(r, θ, ζ)`. A scalar field is a 0-form, a vector field is a 1-form or a
2-form depending on how it transforms, a density is a 3-form. The exterior
derivative on coefficients is a topological difference stencil. Geometry
enters only through the mass matrices.

Read [Architecture](concepts/architecture.md) for the spaces, the
extraction operators, the Laplacians, the data model, and the assembly
order.

## The map

A map `F` sends the logical cube to the physical domain. Analytic maps are
callables in `mrx.mappings`; maps from data are fitted as three scalar
splines. The map's Jacobian at the quadrature points is stored on a
`SequenceGeometry`. Radial direction `r = 0` is the magnetic axis; the
`polar=True` sequence fuses the innermost rings so that fields are smooth
there.

Read [The polar axis](concepts/polar.md) for the constraint at `r = 0`.

## Operators

Nothing larger than a dense polar core is stored. Every operator is a
callable matvec built from 1D basis tables and the metric weight at the
quadrature points. `DeRhamSequence.apply_*` methods apply them;
`DeRhamSequence.apply_inverse_*` methods solve with them.

Read [Mass operators](concepts/mass.md) for the kernel, the
weights, the memory, and the quadrature rule.

## Solvers and preconditioners

Every inverse is a Krylov solve. The k=0 Laplacian and every mass matrix
use CG; the k>=1 Laplacians use MINRES on a saddle-point system. The
production preconditioner is `metric_lumping`: a separable Kronecker bulk
and a dense polar core.

Read [Solvers and preconditioners](concepts/preconditioning.md) for which
solver and preconditioner each operator uses and how to build them.

## Precision

`MRX_DTYPE` selects `float64` or `float32` once per process. Solver
tolerances default to `sqrt(eps)` of the working precision.

Read [Precision](concepts/precision.md).

## Relaxation

`scripts/relax.py` descends the magnetic energy of a divergence-free
2-form under a helicity-conserving flow. The fixed point is
`J × B = ∇p`.

Read [Relaxation](concepts/relaxation.md) for the force, the step, the
diagnostics, and the initial conditions.

## What runs today

[Production](concepts/PRODUCTION.md) is the one-page list of the
settings, solvers, and traps that are current. Read it before changing a
default.

## Also

- [GVEC → MRX interface](concepts/gvec_mrx_interface.md): what MRX needs
  from a GVEC export.
- [Manufactured solutions](concepts/manufactured_solutions.md): the exact
  solutions behind the Poisson convergence studies.
- `docs/research/` in the repository is the campaign record. Its
  `README.md` indexes it by topic; `OPEN.md` lists every open item.
