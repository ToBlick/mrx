# External equilibria → MRX

MRX reads three equilibrium formats — GVEC state files, VMEC `wout` files
and DESC outputs — and reads all three into **one** representation. This
page is the contract they share and the conversion table for each.
`concepts/gvec_mrx_interface` is the companion: it explains *why* MRX
rebuilds a field from scalars rather than resampling a vector, and
everything it says applies here verbatim.

## 1. The shared block dict

Every reader returns the same dict, the one `mrx.gvec.read_state` produces:

```python
dict(nfp=int, deg=int,
     X1=block,                   # R, a cosine series
     X2=block,                   # Z, a sine series
     LA=block,                   # lambda, a sine series
     profiles=dict(rho, phi, iota, pressure))
```

and a `block` is

```python
dict(m, n,                       # (n_modes,) integer mode tables
     coef,                       # (n_modes, n_base) radial B-spline coefficients
     T,                          # the knot vector
     deg, sin_cos)               # 2 = cosine, 1 = sine
```

`mrx.gvec.StateField` evaluates a block as

```
f(rho, theta, zeta) = sum_mn c_mn(rho) trig(m theta_G - n zeta_G)
```

with a **single trig of the combined angle**, `theta_G = 2 pi theta` and
`zeta_G = 2 pi zeta / nfp` the radian angles, and `n` the **full-turn**
toroidal index (VMEC's `xn`, already multiplied by `nfp`). `phi` is the
toroidal flux **per radian**, `Phi / 2 pi`, and the radial label is
`rho = sqrt(s)` with `s` the normalised toroidal flux.

Because the three readers agree on this, everything downstream is shared and
untouched by a new format: `build_gvec_map`, `series_spline_dofs`,
`load_clebsch`, `clebsch_potential_form`, `initial_field`. Adding a format is
writing a parser and extending six dispatch points, nothing else.

## 2. What each reader has to convert

| | GVEC (`.dat`) | VMEC (`.nc`) | DESC (`.h5`) |
|---|---|---|---|
| module | `mrx/gvec.py` | `mrx/vmec.py` | `mrx/desc.py` |
| angular basis | already the combined angle | already the combined angle | **product of two trig factors** |
| radial basis | B-splines | values per flux surface | **Zernike polynomials** |
| radial label | `s = rho` | `s = rho^2`, **remapped** | `rho`, no remap |
| toroidal index | full turn | full turn (`xn`) | **per period, times `nfp`** |
| flux units | `Phi / 2 pi` | `phi` array | **`Psi` Webers, divided by `2 pi`** |
| lambda mesh | full | **half mesh** | full |
| poloidal orientation | as written | as written | **may be flipped** (§5) |
| iota | always stored | always stored | **may be absent** (§4) |

Only the bold entries are work. The rest is bookkeeping.

## 3. DESC's basis conversion

A DESC field is a Fourier-Zernike series evaluated as a **product**,

```
f = sum_lmn c_lmn  Z_l^|m|(rho) . P_m(theta) . T_n(zeta)
P_m(theta) = cos(|m| theta)  if m >= 0 else sin(|m| theta)
T_n(zeta)  = cos(|n| nfp zeta) if n >= 0 else sin(|n| nfp zeta)
```

so converting to MRX's single combined angle is the product-to-sum
identities. With `a = |m|` and `b = |n| nfp`:

```
cos(a t) cos(b z) = ( cos(a t - b z) + cos(a t + b z) ) / 2
sin(a t) sin(b z) = ( cos(a t - b z) - cos(a t + b z) ) / 2
sin(a t) cos(b z) = ( sin(a t - b z) + sin(a t + b z) ) / 2
cos(a t) sin(b z) = ( sin(a t + b z) - sin(a t - b z) ) / 2
```

Each DESC mode therefore becomes the **pair** of MRX modes `(a, +b)` and
`(a, -b)` at half weight, collapsing onto one at full weight when `n = 0`
(`mrx.desc._split_weights`). The identities also settle the parity for
free: DESC's `sym='cos'` filter keeps the modes with `sign(m) = sign(n)`,
and those are exactly the ones whose product is a *cosine* of the combined
angle, so `R` lands in a cosine block and `Z` and lambda in sine blocks
with no extra bookkeeping.

Radially, each `(m, n)` pair's function `sum_l c_lmn Z_l^|m|(rho)` is
sampled at Chebyshev-Lobatto nodes and refit as a clamped interpolatory
B-spline by `mrx.vmec._fit_block`, shared with the wout reader. DESC's
`rho` *is* MRX's `rho`, so unlike the VMEC path there is no radial remap,
and a Zernike series already satisfies the `rho^m` axis parity that
`_fit_block` imposes — there the conditions confirm the fit rather than
correct it. The refit is fourth-order accurate and its error is the only
thing separating MRX's reading from DESC's own evaluation. Measured below
`1e-6` relative on the tracked HELIOTRON and QA fixtures
(`test_desc_fixtures_reproduce_their_own_fourier_zernike_series`), against
a discretisation error many orders larger. `read_desc` records that
midpoint error as `refit_error` and refuses an under-resolved `n_rho`
above `REFIT_TOL = 1e-4`.

## 4. The one thing that may be missing: iota

A DESC equilibrium is constrained by **either** an iota profile **or** a
current profile. In the current case `_iota` is the literal string `None`
in the file: the rotational transform is an *output* of the solve and is
not stored, so no amount of parsing recovers it. This is not a corner case
— NCSX, ARIES-CS, ESTELL, HSX, WISTELL-A, precise_QA and precise_QH are all
current-constrained.

`mrx.desc.read_desc` therefore has two paths. With `_iota` present
(HELIOTRON, W7-X, ATF, DSHAPE, SOLOVEV, and anything saved from
`VMECIO.load(..., profile="iota")`) the read is pure `h5py` and works with
no DESC installed, which is what lets CI cover it. With `_iota` absent it
falls back to DESC itself, `eq.compute("iota")`, and raises an `ImportError`
naming the file and the way out if DESC is not importable.

## 5. The one thing that may be wrong: poloidal orientation

`VMECIO.load` runs DESC's `ensure_positive_jacobian`, which for a
**left-handed** wout — most of them, li383 included — silently flips the
sign of theta: it negates every `m < 0` mode of `R` and `Z`, every `m >= 0`
mode of lambda, and `iota`. The result is the same torus with the opposite
poloidal orientation.

Nothing in MRX *needs* this undone to read the file. `build_gvec_map`
measures the handedness that gives `det DF > 0` rather than assuming one,
so a flipped DESC file builds a perfectly good sequence; only the sign of
the reported `iota` differs. But comparing a DESC state with the VMEC state
it came from at equal `theta` compares two different points, and every
number that falls out is meaningless. `mrx.gvec.match_orientation` measures
the relative orientation (from `R` **and** `Z` — `R` alone cannot see the
flip on an up-down-symmetric cross-section) and
`mrx.gvec.flip_poloidal_angle` undoes it exactly, in the block
representation. Both are re-exported from `mrx.desc`. They apply to a
wout or a GVEC state as well: pyGVEC's `convert-wout` is the second
instance, with its own flux-sign flip (`phiedge = -phiedge`) and an `n`
sign that has to be undone separately before the labels agree.

| | flip |
|---|---|
| `X1` (cosine) | `n -> -n` |
| `X2` (sine) | `n -> -n`, coefficients negated |
| `LA` (sine) | `n -> -n` (the two sign flips cancel) |
| `iota` | negated |
| `phi`, `pressure` | unchanged |

## 6. A known inaccuracy in DESC's wout importer

Two distinct mechanisms, both in DESC, and both visible against a GVEC
control that refits the same wout.

DESC's `desc/vmec.py` reads VMEC's lambda as

```python
lmns = file.variables["lmns"][:].filled()
```

with no `[1:]`, keeping the dummy first row. It then hands that array to
the same fit as `R` and `Z`. That fit, `fourier_to_zernike` in
`desc/vmec_utils.py`, uses

```python
surfs = x_mn.shape[0]
rho = np.sqrt(np.linspace(0, 1, surfs))
```

the **full** mesh, with no half-mesh branch. So every `lmns` row, whose
value belongs at `s_{j-1/2}`, is fit at `s_j` (a half-cell outward shift
that hurts most where the half-cell is a large fraction of `rho`), and
row 0 is fit at `rho = 0`, pulling lambda toward zero on the axis where
the `m = 0` component is not zero.

MRX's own wout reader handles both: it drops row 0 at the call site
(`raw["lmns"][1:]`) and `mrx.vmec._lambda_nodes` puts the rest on the
half mesh with an axis and an edge node added, `m > 0` pinned to zero
from the `rho^m` behaviour and `m = 0` extrapolated linearly in `s`.

The signature is unmistakable: `R` and `Z` converge toward the wout as
DESC's fit resolution rises, while lambda *diverges* near the axis,
because a higher resolution only tracks the wrong nodes more faithfully.
Outside `rho = 0.3` lambda converges like everything else. A GVEC
`convert-wout` of the same file, whose radial B-splines have no half-mesh
confusion, beats DESC on lambda by an order of magnitude — that is the
control. Measured on li383 in `docs/research/desc_interface.md`; this is
a DESC-side issue, recorded rather than worked around.

## 7. Dispatch points

Adding a format touches exactly these:

- `mrx/gvec.py` `read_equilibrium` — extension to reader, sets `kind`
- `mrx/gvec.py` `load_clebsch` — routes `kind` to the profile spline
- `mrx/geometry.py` `geometry_kind` — extension to `kind`
- `mrx/geometry.py` `geometry_nfp` — `nfp` without fitting anything
- `mrx/geometry.py` `build_sequence` — the equilibrium-file branch
- `mrx/initial_conditions.py` `initial_field` — the equilibrium-file branch
- `scripts/relax.py` — `--geometry` help and validation
- `scripts/plot_mesh.py`, `scripts/poincare_trace.py` — `--geometry` help

## 8. Testing

Each reader has a **synthetic writer that inverts it**
(`test/synthetic_gvec.py`, `test/synthetic_desc.py`): a closed-form
circular torus written in the file's own layout and read back, which must
reproduce the formulas to round-off. Both writers describe the *same*
torus, so the two readers are also held against each other — a conversion
error that the formulas tolerate still has to survive an independent parser
of an independent format.

On top of that, `test/test_desc.py` checks the tracked DESC fixtures
against the Fourier-Zernike series rebuilt in DESC's own product form
(independent of the conversion under test), and, when DESC is importable,
against `eq.compute` itself. That last one is the only check that settles
the angle conventions *empirically* rather than by reading DESC's source.

## 9. Reader parity

What each reader exposes, so a new comparison does not have to rediscover
it.

- **Parse.** `read_state` (`.dat`), `read_wout` (`.nc`), `read_desc`
  (`.h5`). `read_equilibrium` dispatches on the extension and sets
  `kind`.
- **Cheap `nfp`.** VMEC and DESC have `read_nfp`. GVEC does not:
  `geometry_nfp` parses a whole state for a `.dat`. That is a
  consistency wart, not a performance bug, and is left as-is.
- **`profile_spline(st, name)`.** All three share the signature and
  return a `BSpline` in the radial label. GVEC's knots are the element
  grid `sp`; VMEC and DESC fit on the sample nodes with the `m = 0`
  parity of `_fit_block`.
- **Resolution control.** Only DESC takes `n_rho` and `deg`. A wout
  refit uses the file's `ns` and a hard-wired degree 3.
- **Orientation.** `flip_poloidal_angle` and `match_orientation` live
  on the shared block dict in `mrx.gvec` and apply to all three.
- **What each refuses.** VMEC: non-NetCDF3, missing wout variables,
  `version_ < 8`, `lasym`, `chipf != iotaf * phipf`. DESC: non-HDF5,
  `n_rho < 3`, `_sym = False`, mixed cos/sin of the combined angle,
  unknown profile class, empty `_equilibria` family,
  current-constrained without DESC, midpoint refit above `REFIT_TOL`.
  GVEC: missing or misshapen blocks only.
- **Not in this interface.** Current HEAD of `mrx.vmec` no longer reads
  `signgs`; handedness is measured by `build_gvec_map`.
