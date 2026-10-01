# Sharp bits

Things that behave differently from what one might expect, and how to notice
them before they cost a run.

## Precision is fixed when MRX is imported

`MRX_DTYPE` and `MRX_RESIDUAL_DTYPE` are read once, when `mrx` is first
imported ([Precision](concepts/precision.md)). Setting them later has no
effect on the arrays. `scripts/relax.py` sets them from
`--geometry.precision` before its import, and the tutorials stop with a
message when `--geometry.precision` disagrees with them. In your own code, set the
variables before the first `import mrx`:

```python
import os
os.environ["MRX_RESIDUAL_DTYPE"] = "float32"   # plain float32 instead of the default mixed precision
import mrx
```

## Plain float32 can stall on strongly shaped devices

Plain float32 (`MRX_RESIDUAL_DTYPE=float32`) is the fastest configuration, but on a
strongly shaped map a Krylov solve can stop improving long before its
tolerance. On the QA stellarator of the tutorials at (12, 16, 16) the solve
behind the vacuum field stalls near a relative residual of `5e-5`, with 2000,
20000 or 100000 iterations alike, while li383 at the same mesh converges.
More iterations do not help, only more precision in the residual does: mixed
precision, the default, converges in about 300 iterations. Use plain float32
only for quick looks.

## A negative iteration count means the solve did not converge

Every solve returns a signed iteration count: positive when it reached its
tolerance, negative when it stopped without. The Laplacian and pressure solves
repeat their inner solves in up to six correction passes and report the sum,
so the count can be larger than `--geometry.solve-maxiter`. A negative count
is the first thing to look for when a result looks off.

## The harmonic forms are computed separately

`build_sequence` does not compute the harmonic forms. Until
`mrx.nullspace.compute_nullspaces(seq)` has run, `seq.nullspace(k)` is zero,
and every solve that removes the harmonic part removes nothing. Call it once
after the build, before any relaxation (`scripts/relax.py` and the tutorials
do).

It prints the Rayleigh quotient `v^T L v / v^T M v` of every form. A harmonic
form reads round-off, `1e-9` or smaller in float32 or mixed precision and
`1e-20` or smaller in float64. A quotient near `1e-3` means the solve that
produced the form did not converge (see the sections above), and the
vacuum field and every projection built on it are off by that much.

## Fields live on the view of their parity

On a stellarator-symmetric geometry the sequence integrates over half a field
period, and a field has a parity. `B`, `A`, `J` and `E` are odd and live on
`seq.odd`, the velocity, the force, the pressures and the constants are even
and live on `seq.even`.
Norms, inner products, mass solves and pushforwards of a field must use the
view of its parity. The full sequence `seq` gives different numbers for the
same coefficient vector. Without stellarator symmetry (`--geometry.symmetry
field-period` or `none`) the views are the sequence itself.

## The map is singular on the axis

The polar map has `det DPhi = 0` at `r = 0`, so anything divided by the
Jacobian, such as the pushforward of a 2-form, is `0 / 0` there. Evaluate
physical fields near the axis, not on it. Quantities that are averaged over the
domain, such as the quasi-symmetry criterion of the shape optimization, leave
out the first radial knot span `r < h_r` for the same reason.

## The wall r = 1 is safe, clipping r to it is not

The map, its derivatives and the fields evaluate correctly at `r = 1` itself.
On QA and li383, `det DPhi` by `jax.jacfwd`, the pushforward of the vacuum
field and the derivative with respect to the map coefficients agree at `r = 1`
and `r = 1 - 1e-6` to the size of the step. There is no need to stay inside
the domain by a small margin, as older code did. What breaks autodiff at the
wall is clipping: `jnp.clip(r, 0, 1)` at `r = 1` halves the derivative,
because JAX splits the gradient of a tie between the two branches. The spline
evaluators continue their end pieces past `r = 1` instead, so pass `r` as it
is.

## The angles are right-handed, so a VMEC file's iota reads negative

MRX's logical coordinates `(r, theta, zeta)` are right-handed, with the map
`Phi = (R cos phi, R sin phi, Z)` of positive Jacobian. The angles of a VMEC
wout are left-handed, so `read_equilibrium` reverses its poloidal angle,
`theta -> -theta`, as DESC does when it converts a wout. Lambda and iota then
change sign: the QA and li383 files read with a negative iota, and
`theta_reversed` in the state (and the `[geom]` line) says so. The device and
every result that does not depend on the orientation (energies, residuals,
`|iota|`, island widths) are unchanged. A checkpoint written before this
convention (2026-09-30) has no `angles` attribute and describes the mirror
image on such a file, so `read_checkpoint`, the drive reference,
`scripts/poincare_trace.py` and the tutorials refuse it.

## Compilation

Every function of the sequence is compiled on first use, so the first call is
slow and the first chunk of a relaxation takes much longer than the rest.
Python numbers stored in `equinox` modules are static: changing one of them,
for example a step size held as a Python `float`, compiles the function again.
A new geometry of the same mesh reuses every compiled solve when its map is
a pytree, as the `CylindricalMap` of `build_map` is: after `seq.set_map` and
`build_preconditioners` the first Laplacian solve takes 0.14 s, not the 18 s
of its compile (li383 at (12, 16, 16)). A map given as a plain Python
function is a static part of the sequence, so every function of the sequence
compiles again when it changes. At
high resolution the quadrature loops can run out of GPU memory, and
`--geometry.max-batch` bounds the number of cells evaluated at once.

## The force residual need not decrease

The relaxation guarantees that the energy decreases, not that the force
residual `||F||` does. `||F||` can rise while the run behaves correctly, since
the gradient can steepen near the minimum. Each run also reaches a
discretisation floor, set by the current sheets on rational surfaces that the
mesh cannot resolve, and further steps along the floor only reconnect at the
grid scale. Judge a run by its floor and its energy, and stop at the floor
(`--budget.floor-tol`). With the resistive drive the energy is not monotone
either.

## Runs are not reproducible to the last digit

Two relaxations that differ only in round-off, for example on another GPU or
with a different summation order, separate exponentially along the trajectory,
by about a decade per ten steps on li383. Their energies and floors agree,
their states after many steps do not. Compare changes by the energy and the
floor, not by the end state.

## The ideal relaxation keeps the topology

Up to grid-scale effects, an ideal run moves and reshapes islands but never
opens or closes one, and it conserves the helicity. The grid-scale
reconnection shows as a small helicity drift (`dH/H_0 = -2.4e-4` over the 20
Newton steps of Tutorial 3) that shrinks with the mesh. A field
that should have islands needs a seed (`--seed`), and one that should lose
them needs the resistive drive. The drive pulls the current towards that of
its reference, so a start that already has the reference's current does not
move at all. Smooth the reference (`--drive.reference-smoothing`, the default)
when the start is the reference itself, and use it unsmoothed when the start
is a different field that should return to it.

## Poincare sections

The tracer uses the toroidal angle as time and needs `B^zeta` of one sign
everywhere, which it checks before tracing. The rotational transform of the
lines closest to the axis is not reliable, because the magnetic axis sits
slightly off `r = 0`. The step-size check (`drift`) is measured on regular
lines only. On a chaotic line it would measure how fast neighbouring lines
separate. `locked_width` resolves an island only to the spacing of the seed
lines, so trace more lines (`--lines`) before reading small widths.
