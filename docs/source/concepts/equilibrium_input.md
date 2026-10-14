# Equilibrium input

`mrx.equilibria` reads a GVEC state file (`GVEC_State_*.dat`,
`mrx.equilibria.gvec`), a VMEC wout (`wout_*.nc`, `mrx.equilibria.vmec`) or
a DESC output (`*.h5`, `mrx.equilibria.desc`) into one representation, and everything downstream (the map, the initial
field) is built from it in closed form, with no evaluation grid in between.

## 1. The premise

MRX's reference 2-form components are the flux densities `sqrt(g) B^i`, the
same object all three codes describe (`sqrt(g) = det DPhi`, with `DPhi` the
Jacobian of the map `Phi`). A field is therefore not resampled as a
vector. It is rebuilt from scalars, which gives `div B = 0`, `B . n = 0`
and nested surfaces exactly, and leaves the fluxes, `iota` and the helicity
exact, independently of the map fit. In GVEC's variables, with `Psi` the
toroidal flux and `chi` the poloidal flux,

```
sqrt(g) B^r     = 0
sqrt(g) B^theta = chi' - Psi' dLA/dzeta
sqrt(g) B^zeta  = Psi' (1 + dLA/dtheta)
```

## 2. The state

`read_equilibrium(path)` picks the reader by the extension and returns the
state, a dict with

- `nfp`, `kind` (`"gvec"`, `"vmec"` or `"desc"`) and `path`.
- the blocks `X1 = R`, `X2 = Z` and `LA = lambda` (radians):
  `sum_mn c_mn(r) cos(m theta - n zeta) + s_mn(r) sin(m theta - n zeta)` in
  radian angles with `zeta` the full-turn toroidal angle. A block holds `m`,
  `n` (`n` a multiple of `nfp`) and the radial B-spline coefficients `cos`
  and `sin` `(n_modes, n_base)` of degree `deg` on the clamped knots `T`.
- `profiles`: `phi` (the toroidal flux `Psi` over `2 pi`), `iota` (per full turn)
  and `pressure`, scipy `BSpline`s in `r`.

The radial label `r` is the square root of the normalised toroidal flux
(`Psi = Psi_edge r^2`). `StateField(block, nfp)` evaluates a block at a
logical point in JAX. A stellarator-symmetric state has `R` a cosine and
`Z`, `lambda` sine series. `is_stellarator_symmetric(st)` checks the
coefficients (every other one zero), and `build_sequence` refuses
`symmetry="stellarator"` for a state without the symmetry
(`symmetry="field-period"` takes it).

**GVEC.** The state file holds the blocks on GVEC's own radial basis and
the profiles at the interpolation points of the `X1` basis. `read_state`
keeps the knots and interpolates each profile on them. A block's `sin_cos`
is 1 (sine), 2 (cosine) or 3 (both). With both, the sine modes are listed
first (`m = 0`, `n = 1..n_max`, then `m = 1..m_max`, `n = -n_max..n_max`),
and the cosine modes follow in the same order with `m = n = 0` in front.

**GVEC G-frame.** The state's `hmap` (the last number of its `global`
line) says how `X1`, `X2` place a point in space. `hmap = 1` is
cylindrical (`X1 = R`, `X2 = Z`). `hmap = 21` is GVEC's G-frame
(`hmap_axisNB`), for devices whose axis is too 3D for planar
cross-sections, such as a figure-8 stellarator:

```
x = a(zeta) + X1 N(zeta) + X2 B(zeta)
```

with the curve `a` and the vectors `N`, `B` sampled in a netCDF file. The
state does not name that file. The GVEC parameter file next to it does
(`hmap_ncfile` in `*.ini`), and `read_state` follows it. Like GVEC, the
reader rotates the samples back by `zeta` about the `z` axis
(`R_z(-zeta) a` is periodic in one field period) and keeps the
trigonometric series of one field period with modes up to
`(nzeta - 1) / 2`. The state's `frame` holds that series. Any other `hmap`
is refused.

**VMEC.** The wout holds `rmnc`, `zmns` on the full radial mesh
`s_j = j / (ns - 1)` and `lmns` on the half mesh, and with `lasym = 1` the
other parities `rmns`, `zmnc`, `lmnc` on the same meshes. `read_wout` interpolates
every mode and profile to a clamped cubic B-spline in `r = sqrt(s)` with the
axis behaviour of a smooth field imposed (a mode `m` is `r^m` times an even
function of `r`, see `mrx.equilibria.fit`). The half-mesh `lambda` is extended
linearly in `s` to the axis and the edge. Files older than VMEC 8 are
refused.

**DESC.** The file holds an `Equilibrium` (or an `EquilibriaFamily`, whose
last member is read) as Fourier-Zernike series: the mode `(l, m, n)` of
`R_lmn`, `Z_lmn`, `L_lmn` is `Z_l^|m|(r) F_m(theta) F_n(nfp zeta)`, with
`F_m = cos(|m| x)` for `m >= 0` and `sin(|m| x)` for `m < 0`, `Z_l^|m|` the
Zernike radial polynomial (`Z(1) = 1`) and `zeta` the full-turn toroidal
angle. `read_desc` converts it exactly: each product of the two Fourier
factors is the pair `trig(|m| theta -+ |n| nfp zeta) / 2` (a cosine for
`sign(m) = sign(n)`, `sign(0) = 1`, a sine otherwise), and each mode's
radial function, a polynomial of degree `L`, is one Bezier segment of that
degree. The profiles are the file's own functions: a `PowerSeriesProfile`
as a polynomial, a `SplineProfile` (method `cubic2`) as its not-a-knot cubic,
`phi = Psi_edge r^2 / (2 pi)` with `Psi_edge` the file's total toroidal
flux. A current-constrained file stores the net toroidal current `I`
instead of `iota`, and the reader evaluates DESC's
flux-surface average

```
iota = (mu0 I / (2 Psi_edge r) + <(dLA/dzeta g_tt - (1 + dLA/dtheta) g_tz) / sqrt g>) / <g_tt / sqrt g>
```

(`<.>` the mean over `theta, zeta`, `g` the metric of `(r, theta, zeta)`)
on a uniform angular grid and interpolates it as a polynomial in `r^2`.
DESC keeps its Jacobian positive, so a file converted from a VMEC wout has
`theta = -u`, and `LA` and `iota` change sign with it. MRX does the same
for every file: `read_equilibrium` reverses `theta` of a file whose angles
are left-handed (a VMEC wout), so the logical coordinates are always
right-handed, and `theta_reversed` in the state records it. Kinetic or anisotropic pressure and other
profile classes are refused.

## 3. The map

`build_map(st, seq, nfp, stellarator_symmetric)` returns `(Phi, info)`, the
map `Phi = (R cos phi, R sin phi, Z)`, `phi = 2 pi zeta / nfp`, on
`seq.basis_0`. The coefficients of `R` and `Z` are the L2 projection of the
series onto the polar 0-form space, mode by mode. The radial splines are
projected exactly by Gauss quadrature on the union of the two knot sets,
the angular modes by the projection of `exp(2 pi i (m theta - n zeta))` onto
the periodic splines. With `stellarator_symmetric` they are projected onto
`R` even and `Z` odd. `info` holds `nfp`, the sampled `det_range` (checked
to be positive), the `symmetry_defect` and the raw coefficients `raw_R`,
`raw_Z`.

A G-frame state gives a `FrameMap`, `Phi = R_z(phi) P` with
`P = a + X1 N + X2 B` the position in the frame that turns with the field
periods. `P` is periodic in one field period whatever the axis does, and
a `CylindricalMap` is the case `P = (R, 0, Z)`. Each Cartesian component of
`P` is a sum of the state's modes times the frame's modes
(`frame_blocks`), projected as above. With `stellarator_symmetric`, `P_x`
is projected like `R` and `P_y`, `P_z` like `Z`. `info` holds `raw_P`.
`state_position(st)` is the state's own map, cylindrical or G-frame, as a
JAX function, the reference the spline map approximates.

## 4. Conventions that must be stated

**Angles.** The state's angles are radians: `theta_G = 2 pi theta`,
`zeta_G = 2 pi zeta / nfp`. MRX's `zeta` spans one field period, so the
transform per MRX toroidal turn is `iota / nfp`. A missed `1/nfp` makes
`iota` `nfp` times too large while passing every structural check.

**Radial label.** A profile in `s` read as a profile in `r = sqrt(s)`
distorts the field's radial structure by `ds/dr = 2 r` and leaves
`iota` untouched. `dPsi/dr` proportional to `r` means the label is `r`.

**`nfp`.** A wrong `nfp` wraps one field period through the
wrong angle with a healthy Jacobian. `build_sequence` takes an `nfp`
override for a file that declares it wrong.

**Frames.** `seq.load(f, k)` takes physical components, so a field given
in logical components is pushed forward first. `seq.interpolate(f, k)` takes
physical components, or logical ones with `frame='logical'`.

## 5. What to check on a new file

1. `||div B||` of the initial condition: round-off for `B = dA'`.
2. The wall-normal part the Dirichlet restriction discards (`initial_field`
   reports it as `wall_discarded`): zero when `A'` on the wall depends on
   `r` alone.
3. `iota` at the axis and the edge against the file (`iota_axis`,
   `iota_edge`, with the opposite sign when `theta_reversed`), and the
   Poincare iota profile of the initial field.
4. The force residual of the initial field: the end-to-end test of map,
   representation and force operator, floored by how well the file's
   equilibrium is converged.
5. `symmetry_defect` of the map, for `symmetry="stellarator"`.

`scripts/relax.py --geometry.path <file>` prints 1-4 at the start of a run.

## 6. The synthetic state

`test/synthetic_gvec.py` (`write_synthetic_state`) writes a state file of
a circular torus from closed formulas: `R = R0 + a r cos theta_G`,
`Z = a r sin theta_G`, `Psi = Psi_edge r^2`, `iota = iota0 + iota1 r^2`, a
`lambda` with a toroidal modulation and a parabolic pressure. Every radial
function is in the spline space, so a correct parser reproduces the
formulas to round-off. `test/test_readers.py` checks it, and it checks the VMEC
reader on the tracked li383 wout. `test/synthetic_desc.py`
(`write_synthetic_desc`) does the same for a DESC file, with every parity
of the product basis, an `iota`- and a current-constrained variant. The
test also holds DESC's conversion of the li383 wout
(`data/desc_li383_low_res_reference.h5`) against the wout. Without
stellarator symmetry, the GVEC writer's `shift = (c, d)` writes the fields
at `(theta_G + c, zeta_G + d)` with both series (`sin_cos = 3`), the test
writes the li383 wout shifted the same way as a `lasym = 1` wout, and both
must read as the unshifted state at the shifted angles. The DESC writer's
`asym` adds terms of the other parity.

MRX has no separate input for analytic shapes. An analytic shape is used
by writing a mock equilibrium file from its formulas, for example a VMEC
wout or a GVEC state, as `test/synthetic_gvec.py` and
`test/synthetic_desc.py` do.
