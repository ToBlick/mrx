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
`1e-6` relative on the tracked SOLOVEV and DSHAPE fixtures
(`test_desc_fixtures_reproduce_their_own_fourier_zernike_series`), against
a discretisation error many orders larger.

## 4. The one thing that may be missing: iota

A DESC equilibrium is constrained by **either** an iota profile **or** a
current profile. In the current case `_iota` is the literal string `None`
in the file: the rotational transform is an *output* of the solve and is
not stored, so no amount of parsing recovers it. This is not a corner case
— NCSX, ARIES-CS, ESTELL, HSX, WISTELL-A, precise_QA and precise_QH are all
current-constrained.

`mrx.desc.read_desc` therefore has two paths. With `_iota` present
(SOLOVEV, HELIOTRON, W7-X, ATF, DSHAPE, and anything saved from
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
number that falls out is meaningless. `mrx.desc.match_orientation` measures
the relative orientation (from `R` **and** `Z` — `R` alone cannot see the
flip on an up-down-symmetric cross-section) and
`mrx.desc.flip_poloidal_angle` undoes it exactly, in the block
representation:

| | flip |
|---|---|
| `X1` (cosine) | `n -> -n` |
| `X2` (sine) | `n -> -n`, coefficients negated |
| `LA` (sine) | `n -> -n` (the two sign flips cancel) |
| `iota` | negated |
| `phi`, `pressure` | unchanged |

## 6. A known inaccuracy in DESC's wout importer

DESC's `fourier_to_zernike` fits VMEC's lambda at
`rho = sqrt(linspace(0, 1, ns))`, but `lmns` in a wout lives on the **half
mesh**, and its first row is a dummy zero. So the fit is misregistered by
half a radial cell and anchored to zero on the axis, where VMEC's lambda is
not zero. MRX's own wout reader handles the half mesh correctly
(`mrx.vmec._lambda_nodes`).

The signature is unmistakable: `R` and `Z` converge toward the wout as
DESC's fit resolution rises, while lambda *diverges* near the axis, because
a higher resolution only tracks the wrong nodes more faithfully. Outside
`rho = 0.3` lambda converges like everything else. Measured on li383 in
`docs/research/desc_interface_2026-09-13.md`; this is a DESC-side issue,
recorded rather than worked around.

## 7. Dispatch points

Adding a format touches exactly these:

- `mrx/gvec.py` `read_equilibrium` — extension to reader, sets `kind`
- `mrx/gvec.py` `load_clebsch` — routes `kind` to the profile spline
- `mrx/geometry.py` `geometry_kind` — extension to `kind`
- `mrx/geometry.py` `geometry_nfp` — `nfp` without fitting anything
- `mrx/geometry.py` `build_sequence` — the equilibrium-file branch
- `mrx/initial_conditions.py` `initial_field` — the equilibrium-file branch
- `scripts/relax.py` — `--geometry` help and validation

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
